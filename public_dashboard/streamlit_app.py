#!/usr/bin/env python3
"""
Public Streamlit dashboard for Kalshi bot performance.

This app is deploy-only and contains no strategy/execution logic.
It can read from:
1) DATABASE_URL (recommended for live cloud deploy)
2) SQLITE_PATH (local fallback)
"""

import ast
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text


def secret_or_env(name, default=None):
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    if value is None:
        value = os.getenv(name, default)
    return value


def as_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def parse_ts(value):
    if value is None:
        return pd.NaT
    return pd.to_datetime(value, utc=True, errors="coerce")


def db_config():
    db_url = secret_or_env("DATABASE_URL", None)
    sqlite_path = secret_or_env("SQLITE_PATH", "data/kalshi_quotes.sqlite")
    max_risk = float(secret_or_env("MAX_TOTAL_RISK_DOLLARS", "60.0"))
    return db_url, sqlite_path, max_risk


@st.cache_resource(show_spinner=False)
def get_engine_and_label():
    db_url, sqlite_path, _ = db_config()
    if db_url:
        engine = create_engine(db_url, pool_pre_ping=True)
        return engine, "DATABASE_URL"

    path = Path(sqlite_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    engine = create_engine(f"sqlite:///{path}")
    return engine, f"SQLITE_PATH ({path})"


def fetch_query(sql, params=None):
    engine, _ = get_engine_and_label()
    params = params or {}
    try:
        with engine.connect() as conn:
            return pd.read_sql_query(text(sql), conn, params=params)
    except Exception:
        return pd.DataFrame()


def parse_live_order_payload(raw_json):
    if not raw_json:
        return {}
    try:
        return json.loads(raw_json)
    except Exception:
        try:
            return ast.literal_eval(raw_json)
        except Exception:
            return {}


def filled_qty_from_payload(payload):
    if not isinstance(payload, dict):
        return 0
    order = payload.get("order") if isinstance(payload.get("order"), dict) else payload
    for key in ("fill_count", "filled_quantity", "filled_count", "quantity_filled", "executed_count", "count"):
        value = order.get(key) if isinstance(order, dict) else None
        if value is not None:
            return max(0, as_int(value, 0))
    return 0


def fill_price_from_payload(payload, fallback_price):
    if not isinstance(payload, dict):
        return fallback_price
    order = payload.get("order") if isinstance(payload.get("order"), dict) else payload
    for key in ("yes_price", "execution_price", "average_price", "price"):
        value = order.get(key) if isinstance(order, dict) else None
        if value is not None:
            try:
                return int(round(float(value)))
            except Exception:
                return fallback_price
    return fallback_price


def load_trade_events():
    trades_df = fetch_query(
        """
        SELECT
            ts_utc,
            ticker,
            UPPER(side) AS side,
            price,
            qty,
            COALESCE(note, '') AS note,
            'trades' AS source
        FROM trades
        """
    )

    live_df = fetch_query(
        """
        SELECT
            ts_utc,
            ticker,
            LOWER(action) AS action,
            yes_price,
            raw_json,
            status
        FROM live_orders
        WHERE LOWER(action) IN ('buy', 'sell')
        """
    )

    trade_keys = set()
    if not trades_df.empty:
        for _, row in trades_df.iterrows():
            trade_keys.add(
                (
                    str(row.get("ts_utc")),
                    str(row.get("ticker")),
                    str(row.get("side")).upper(),
                    as_int(row.get("price"), 0),
                    as_int(row.get("qty"), 0),
                )
            )

    live_rows = []
    if not live_df.empty:
        for _, row in live_df.iterrows():
            payload = parse_live_order_payload(row.get("raw_json"))
            fill_qty = filled_qty_from_payload(payload)
            if fill_qty <= 0:
                continue
            action = str(row.get("action") or "").lower()
            side = "BUY" if action == "buy" else "SELL"
            base_price = as_int(row.get("yes_price"), 0)
            fill_price = max(0, min(100, fill_price_from_payload(payload, base_price)))
            key = (
                str(row.get("ts_utc")),
                str(row.get("ticker")),
                side,
                as_int(fill_price, 0),
                as_int(fill_qty, 0),
            )
            if key in trade_keys:
                continue

            live_rows.append(
                {
                    "ts_utc": row.get("ts_utc"),
                    "ticker": row.get("ticker"),
                    "side": side,
                    "price": fill_price,
                    "qty": fill_qty,
                    "note": f"live status={row.get('status') or ''}",
                    "source": "live_orders",
                }
            )

    live_fills_df = pd.DataFrame(live_rows)
    all_trades = pd.concat([trades_df, live_fills_df], ignore_index=True, sort=False)
    if all_trades.empty:
        return all_trades

    all_trades["qty"] = pd.to_numeric(all_trades["qty"], errors="coerce").fillna(0).astype(int)
    all_trades["price"] = pd.to_numeric(all_trades["price"], errors="coerce").fillna(0.0).astype(float)
    all_trades["side"] = all_trades["side"].astype(str).str.upper()
    all_trades["ts"] = pd.to_datetime(all_trades["ts_utc"], utc=True, errors="coerce")
    all_trades = all_trades.dropna(subset=["ts"])
    all_trades = all_trades[all_trades["qty"] > 0]
    all_trades = all_trades[all_trades["side"].isin(["BUY", "SELL"])]
    all_trades = all_trades.sort_values("ts", ascending=False).reset_index(drop=True)
    return all_trades


def compute_win_rate(trades):
    if trades.empty:
        return 0.0, 0
    ordered = trades.sort_values("ts")
    state = {}
    wins = 0
    losses = 0
    for _, t in ordered.iterrows():
        ticker = str(t["ticker"])
        side = str(t["side"])
        qty = int(t["qty"])
        px = float(t["price"])
        if ticker not in state:
            state[ticker] = {"qty": 0, "avg": 0.0}

        cur_qty = state[ticker]["qty"]
        cur_avg = state[ticker]["avg"]
        if side == "BUY":
            new_qty = cur_qty + qty
            if new_qty > 0:
                state[ticker]["avg"] = ((cur_qty * cur_avg) + (qty * px)) / new_qty
            state[ticker]["qty"] = new_qty
            continue

        if side == "SELL" and cur_qty > 0:
            closed_qty = min(qty, cur_qty)
            pnl = (px - cur_avg) * closed_qty / 100.0
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1
            state[ticker]["qty"] = cur_qty - closed_qty
            if state[ticker]["qty"] == 0:
                state[ticker]["avg"] = 0.0

    decided = wins + losses
    return (100.0 * wins / decided) if decided > 0 else 0.0, decided


st.set_page_config(page_title="Kalshi Bot Dashboard", layout="wide")
st.title("Kalshi Bot Dashboard")
st.caption("Public performance view. No trading logic is included in this repo.")

if st.button("Refresh Now"):
    st.rerun()

engine, source_label = get_engine_and_label()
_, _, MAX_RISK = db_config()

trade_events = load_trade_events()

with st.expander("Data Source"):
    st.write(f"Source: `{source_label}`")
    ts_info = {
        "Latest trade table ts": fetch_query("SELECT MAX(ts_utc) AS ts FROM trades"),
        "Latest live_orders ts": fetch_query("SELECT MAX(ts_utc) AS ts FROM live_orders"),
        "Latest quote ts": fetch_query("SELECT MAX(ts_utc) AS ts FROM quotes"),
    }
    for label, df in ts_info.items():
        value = df.iloc[0, 0] if not df.empty else None
        st.write(f"- {label}: `{value}`")

col1, col2, col3, col4, col5 = st.columns(5)

if trade_events.empty:
    realized_pnl = 0.0
    daily = pd.DataFrame(columns=["date", "daily_pnl"])
else:
    signed = np.where(
        trade_events["side"].values == "BUY",
        -trade_events["qty"].values * trade_events["price"].values / 100.0,
        trade_events["qty"].values * trade_events["price"].values / 100.0,
    )
    realized_pnl = float(np.sum(signed))
    daily = (
        trade_events.assign(
            signed_cash=signed,
            date=trade_events["ts"].dt.date.astype(str),
        )
        .groupby("date", as_index=False)["signed_cash"]
        .sum()
        .rename(columns={"signed_cash": "daily_pnl"})
        .sort_values("date")
    )

unrealized_df = fetch_query(
    """
    SELECT
        COALESCE(SUM(
            CASE
                WHEN qty > 0 THEN qty * (100 - avg_price) / 100.0
                WHEN qty < 0 THEN qty * (avg_price) / 100.0
                ELSE 0
            END
        ), 0) AS unrealized_pnl
    FROM positions
    WHERE qty != 0
    """
)
unrealized_pnl = float(unrealized_df["unrealized_pnl"].iloc[0]) if not unrealized_df.empty else 0.0
total_pnl = realized_pnl + unrealized_pnl

win_rate, win_samples = compute_win_rate(trade_events)
open_pos_df = fetch_query("SELECT COUNT(*) AS cnt FROM positions WHERE qty != 0")
open_pos = int(open_pos_df["cnt"].iloc[0]) if not open_pos_df.empty else 0

risk_df = fetch_query(
    """
    SELECT
        COALESCE(SUM(
            CASE
                WHEN qty > 0 THEN qty * avg_price / 100.0
                WHEN qty < 0 THEN (-qty) * (100 - avg_price) / 100.0
                ELSE 0
            END
        ), 0) AS total_risk
    FROM positions
    WHERE qty != 0
    """
)
total_risk = float(risk_df["total_risk"].iloc[0]) if not risk_df.empty else 0.0
risk_pct = (100.0 * total_risk / MAX_RISK) if MAX_RISK > 0 else 0.0

if daily.empty:
    sharpe = 0.0
else:
    recent = daily.tail(30)["daily_pnl"].values
    std = np.std(recent) if len(recent) > 1 else 0.0
    sharpe = (np.mean(recent) / std * np.sqrt(252)) if std > 0 else 0.0

with col1:
    st.metric("Total P&L", f"${total_pnl:.2f}")
with col2:
    st.metric("Win Rate", f"{win_rate:.1f}%", delta=f"{win_samples} closed sells")
with col3:
    st.metric("Open Positions", open_pos)
with col4:
    st.metric("Portfolio Risk", f"${total_risk:.2f}", delta=f"{risk_pct:.0f}% of ${MAX_RISK:.0f}")
with col5:
    st.metric("Sharpe (30d)", f"{sharpe:.2f}")

st.divider()
left, right = st.columns(2)
with left:
    st.subheader("Equity Curve")
    if not daily.empty:
        d = daily.copy()
        d["cum"] = d["daily_pnl"].cumsum()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=d["date"],
                y=d["cum"],
                mode="lines+markers",
                line=dict(width=2),
                fill="tozeroy",
                name="Equity",
            )
        )
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=24, b=0), yaxis_title="P&L ($)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trades yet.")

with right:
    st.subheader("Daily P&L")
    if not daily.empty:
        colors = ["green" if x > 0 else "red" for x in daily["daily_pnl"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=daily["date"], y=daily["daily_pnl"], marker_color=colors))
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=24, b=0), yaxis_title="P&L ($)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trades yet.")

st.divider()
st.subheader("Recent Trades")
if not trade_events.empty:
    recent = trade_events.head(30).copy()
    recent["timestamp"] = recent["ts"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    recent["price"] = recent["price"].map(lambda x: f"{x:.1f}c")
    st.dataframe(
        recent[["timestamp", "ticker", "side", "qty", "price", "source"]],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No trades yet.")

st.divider()
st.subheader("Series Performance")
if not trade_events.empty:
    t = trade_events.copy()
    t["series"] = t["ticker"].astype(str).str.extract(r"^([A-Z0-9]+)")
    t["cash"] = np.where(
        t["side"] == "BUY",
        -t["qty"] * t["price"] / 100.0,
        t["qty"] * t["price"] / 100.0,
    )
    series_df = (
        t.groupby("series", as_index=False)
        .agg(trades=("ticker", "count"), pnl=("cash", "sum"))
        .sort_values("pnl", ascending=False)
    )
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.bar(series_df, x="series", y="pnl", color="pnl", color_continuous_scale=["red", "gray", "green"])
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.dataframe(series_df, use_container_width=True, hide_index=True)
else:
    st.info("No series data yet.")

st.divider()
st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
