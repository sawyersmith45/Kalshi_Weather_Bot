#!/usr/bin/env python3
"""
Kalshi Trading Bot - Live Dashboard
"""

import ast
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

def get_setting(name, default=None):
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    if value is None:
        value = os.getenv(name, default)
    return value


MAX_RISK = float(get_setting("MAX_TOTAL_RISK_DOLLARS", "60.0"))
DATABASE_URL = str(get_setting("DATABASE_URL", "") or "").strip()


def parse_ts(value):
    if value is None:
        return pd.NaT
    return pd.to_datetime(value, utc=True, errors="coerce")


def as_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def table_exists(conn, table_name):
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def max_table_ts(conn, table_name):
    if not table_exists(conn, table_name):
        return pd.NaT
    row = conn.execute(f"SELECT MAX(ts_utc) FROM {table_name}").fetchone()
    return parse_ts(row[0] if row else None)


def resolve_db_path():
    root = Path(__file__).resolve().parent.parent
    candidates = []

    env_db = get_setting("KALSHI_DB_PATH", None) or get_setting("DB_PATH", None)
    if env_db:
        candidates.append(Path(env_db).expanduser())

    candidates.append(root / "data" / "kalshi_quotes.sqlite")
    candidates.append(root / "kalshi.db")

    unique_candidates = []
    for c in candidates:
        if c not in unique_candidates:
            unique_candidates.append(c)

    scored = []
    for path in unique_candidates:
        if not path.exists():
            continue
        try:
            conn = sqlite3.connect(str(path), check_same_thread=False, timeout=5)
            latest_trade = max_table_ts(conn, "trades")
            latest_live = max_table_ts(conn, "live_orders")
            latest_quote = max_table_ts(conn, "quotes")
            candidates_ts = [t for t in (latest_trade, latest_live, latest_quote) if not pd.isna(t)]
            latest = max(candidates_ts) if candidates_ts else pd.NaT
            conn.close()
            scored.append(
                {
                    "path": path,
                    "latest": latest,
                    "latest_trade": latest_trade,
                    "latest_live": latest_live,
                    "latest_quote": latest_quote,
                }
            )
        except Exception:
            continue

    if not scored:
        return root / "data" / "kalshi_quotes.sqlite", []

    def freshness_key(entry):
        ts = entry.get("latest")
        if pd.isna(ts):
            return -1
        return int(ts.value)

    scored.sort(key=freshness_key, reverse=True)
    return scored[0]["path"], scored


def get_sqlite_db(path):
    db = sqlite3.connect(str(path), check_same_thread=False, timeout=10)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=5000")
    db.execute("PRAGMA temp_store=MEMORY")
    return db


@st.cache_resource(show_spinner=False)
def get_engine(url):
    return create_engine(url, pool_pre_ping=True)


def get_db_handle():
    if DATABASE_URL:
        try:
            engine = get_engine(DATABASE_URL)
            return {
                "kind": "engine",
                "ready": True,
                "engine": engine,
                "source": "DATABASE_URL",
                "path": None,
                "exists": True,
                "candidates": [],
            }
        except Exception:
            return {
                "kind": "engine",
                "ready": False,
                "engine": None,
                "source": "DATABASE_URL",
                "path": None,
                "exists": False,
                "candidates": [],
            }

    db_path, db_candidates = resolve_db_path()
    db_exists = db_path.exists()
    db = get_sqlite_db(db_path) if db_exists else None
    return {
        "kind": "sqlite",
        "ready": db is not None,
        "conn": db,
        "source": f"sqlite:{db_path}",
        "path": db_path,
        "exists": db_exists,
        "candidates": db_candidates,
    }


def fetch_query(db_handle, sql, params=()):
    try:
        if db_handle["kind"] == "engine":
            with db_handle["engine"].connect() as conn:
                if isinstance(params, dict):
                    return pd.read_sql_query(text(sql), conn, params=params)
                return pd.read_sql_query(text(sql), conn)
        return pd.read_sql_query(sql, db_handle["conn"], params=params)
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
            return as_int(round(float(value)), fallback_price)
    return fallback_price


def load_trade_events(db_handle):
    trades_df = fetch_query(
        db_handle,
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
        """,
    )

    live_df = fetch_query(
        db_handle,
        """
        SELECT
            ts_utc,
            ticker,
            LOWER(action) AS action,
            yes_price,
            raw_json,
            order_id,
            client_order_id,
            status
        FROM live_orders
        WHERE LOWER(action) IN ('buy', 'sell')
        """,
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
            fill_price = fill_price_from_payload(payload, base_price)
            fill_price = max(0, min(100, fill_price))
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
    all_trades["price"] = pd.to_numeric(all_trades["price"], errors="coerce").fillna(0).astype(float)
    all_trades["side"] = all_trades["side"].astype(str).str.upper()
    all_trades = all_trades[all_trades["qty"] > 0]
    all_trades = all_trades[all_trades["side"].isin(["BUY", "SELL"])]
    all_trades["ts"] = pd.to_datetime(all_trades["ts_utc"], utc=True, errors="coerce")
    all_trades = all_trades.dropna(subset=["ts"])

    all_trades = all_trades.drop_duplicates(
        subset=["ts_utc", "ticker", "side", "price", "qty", "source"],
        keep="last",
    )
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
        ticker = t["ticker"]
        side = t["side"]
        qty = int(t["qty"])
        price = float(t["price"])

        if ticker not in state:
            state[ticker] = {"qty": 0, "avg": 0.0}

        pos_qty = state[ticker]["qty"]
        pos_avg = state[ticker]["avg"]

        if side == "BUY":
            new_qty = pos_qty + qty
            if new_qty > 0:
                state[ticker]["avg"] = ((pos_qty * pos_avg) + (qty * price)) / new_qty
            state[ticker]["qty"] = new_qty
            continue

        if side == "SELL" and pos_qty > 0:
            closed_qty = min(qty, pos_qty)
            pnl = (price - pos_avg) * closed_qty / 100.0
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1
            state[ticker]["qty"] = pos_qty - closed_qty
            if state[ticker]["qty"] == 0:
                state[ticker]["avg"] = 0.0

    decided = wins + losses
    return (100.0 * wins / decided) if decided > 0 else 0.0, decided


st.set_page_config(page_title="Trading Bot Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("Kalshi Trading Bot - Live Dashboard")

if st.button("Refresh Now", key="refresh_btn"):
    st.rerun()

db_handle = get_db_handle()
trade_events = load_trade_events(db_handle) if db_handle["ready"] else pd.DataFrame()

with st.expander("Debug Info"):
    st.write(f"**Data Source:** `{db_handle['source']}`")
    if db_handle["kind"] == "sqlite":
        st.write(f"**Selected DB Path:** `{db_handle['path']}`")
        st.write(f"**Database Exists:** {db_handle['exists']}")
    if db_handle["kind"] == "sqlite" and db_handle["candidates"]:
        st.write("**DB Freshness (newest first):**")
        for c in db_handle["candidates"]:
            st.write(
                f"- `{c['path']}` | latest={c['latest']} | trades={c['latest_trade']} | "
                f"live_orders={c['latest_live']} | quotes={c['latest_quote']}"
            )
    if db_handle["ready"]:
        debug_queries = {
            "Total Trades (table)": "SELECT COUNT(*) AS cnt FROM trades",
            "Total Live Orders": "SELECT COUNT(*) AS cnt FROM live_orders",
            "Unified Trade Events": f"SELECT {len(trade_events)} AS cnt",
            "Latest Unified Trade": (
                "SELECT MAX(ts_utc) AS ts FROM ("
                "SELECT ts_utc FROM trades UNION ALL SELECT ts_utc FROM live_orders)"
            ),
            "Total Positions": "SELECT COUNT(*) AS cnt FROM positions WHERE qty != 0",
            "Recent Quotes (1h)": "SELECT COUNT(*) AS cnt FROM quotes WHERE ts_utc > datetime('now', '-1 hour')",
        }
        for label, query in debug_queries.items():
            result = fetch_query(db_handle, query)
            value = result.iloc[0, 0] if not result.empty else "N/A"
            st.write(f"- {label}: **{value}**")

if not db_handle["ready"]:
    st.error(
        "No database found. Expected one of: secret/env `DATABASE_URL`, "
        "`KALSHI_DB_PATH`, `data/kalshi_quotes.sqlite`, `kalshi.db`."
    )
    st.stop()

col1, col2, col3, col4, col5 = st.columns(5)

if trade_events.empty:
    realized_pnl = 0.0
else:
    cashflow = np.where(
        trade_events["side"].values == "BUY",
        -trade_events["qty"].values * trade_events["price"].values / 100.0,
        trade_events["qty"].values * trade_events["price"].values / 100.0,
    )
    realized_pnl = float(np.sum(cashflow))

unrealized_df = fetch_query(
    db_handle,
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
    """,
)
unrealized_pnl = float(unrealized_df["unrealized_pnl"].iloc[0]) if not unrealized_df.empty else 0.0
total_pnl = realized_pnl + unrealized_pnl

with col1:
    st.metric("Total P&L", f"${total_pnl:.2f}")

win_rate, win_samples = compute_win_rate(trade_events)
with col2:
    st.metric("Win Rate", f"{win_rate:.1f}%", delta=f"{win_samples} closed sells")

pos_df = fetch_query(db_handle, "SELECT COUNT(*) AS cnt FROM positions WHERE qty != 0")
open_pos = int(pos_df["cnt"].iloc[0]) if not pos_df.empty else 0
with col3:
    st.metric("Open Positions", open_pos)

risk_df = fetch_query(
    db_handle,
    """
    SELECT COALESCE(SUM(
        CASE
            WHEN qty > 0 THEN qty * avg_price / 100.0
            WHEN qty < 0 THEN (-qty) * (100 - avg_price) / 100.0
            ELSE 0
        END
    ), 0) AS total_risk
    FROM positions
    WHERE qty != 0
    """,
)
total_risk = float(risk_df["total_risk"].iloc[0]) if not risk_df.empty else 0.0
risk_pct = (total_risk / MAX_RISK * 100.0) if MAX_RISK > 0 else 0.0
with col4:
    st.metric("Portfolio Risk", f"${total_risk:.2f}", delta=f"{risk_pct:.0f}% of ${MAX_RISK:.0f}")

if trade_events.empty:
    sharpe = 0.0
    daily_pnl = pd.DataFrame(columns=["date", "daily_pnl"])
else:
    daily_pnl = (
        trade_events.assign(
            signed_pnl=np.where(
                trade_events["side"] == "BUY",
                -trade_events["qty"] * trade_events["price"] / 100.0,
                trade_events["qty"] * trade_events["price"] / 100.0,
            ),
            date=trade_events["ts"].dt.date.astype(str),
        )
        .groupby("date", as_index=False)["signed_pnl"]
        .sum()
        .rename(columns={"signed_pnl": "daily_pnl"})
        .sort_values("date")
    )
    recent_daily = daily_pnl.tail(30)["daily_pnl"].values
    daily_std = np.std(recent_daily) if len(recent_daily) > 1 else 0
    sharpe = (np.mean(recent_daily) / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0

with col5:
    st.metric("Sharpe Ratio (30d)", f"{sharpe:.2f}")

st.divider()
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Equity Curve")
    if not daily_pnl.empty:
        equity_data = daily_pnl.copy()
        equity_data["cumulative_pnl"] = equity_data["daily_pnl"].cumsum()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=equity_data["date"],
                y=equity_data["cumulative_pnl"],
                mode="lines+markers",
                name="Equity",
                line=dict(color="#1f77b4", width=2),
                fill="tozeroy",
            )
        )
        fig.update_layout(
            height=350,
            hovermode="x unified",
            margin=dict(l=0, r=0, t=20, b=0),
            yaxis_title="P&L ($)",
            xaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trades yet")

with col_right:
    st.subheader("Daily P&L")
    if not daily_pnl.empty:
        colors = ["green" if x > 0 else "red" for x in daily_pnl["daily_pnl"]]
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=daily_pnl["date"],
                y=daily_pnl["daily_pnl"],
                marker_color=colors,
                name="Daily P&L",
            )
        )
        fig.update_layout(
            height=350,
            hovermode="x",
            margin=dict(l=0, r=0, t=20, b=0),
            yaxis_title="P&L ($)",
            xaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trades yet")

st.divider()
st.subheader("Recent Trades (Last 20)")

if not trade_events.empty:
    recent_trades = trade_events.head(20).copy()
    recent_trades["timestamp"] = recent_trades["ts"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    recent_trades["price"] = recent_trades["price"].map(lambda x: f"{x:.1f}c")
    recent_trades["cashflow"] = np.where(
        recent_trades["side"] == "BUY",
        -recent_trades["qty"] * recent_trades["price"].str.replace("c", "").astype(float) / 100.0,
        recent_trades["qty"] * recent_trades["price"].str.replace("c", "").astype(float) / 100.0,
    )
    recent_trades["cashflow"] = recent_trades["cashflow"].map(lambda x: f"${x:.2f}")

    st.dataframe(
        recent_trades[["timestamp", "ticker", "side", "qty", "price", "cashflow", "source"]],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No trades yet")

st.divider()
st.subheader("Performance by Series")

if not trade_events.empty:
    series_df = trade_events.copy()
    series_df["series"] = series_df["ticker"].astype(str).str.extract(r"^([A-Z0-9]+)")
    series_df["signed_cashflow"] = np.where(
        series_df["side"] == "BUY",
        -series_df["qty"] * series_df["price"] / 100.0,
        series_df["qty"] * series_df["price"] / 100.0,
    )
    series_pnl = (
        series_df.groupby("series", as_index=False)
        .agg(trade_count=("ticker", "count"), series_pnl=("signed_cashflow", "sum"))
        .sort_values("series_pnl", ascending=False)
    )
else:
    series_pnl = pd.DataFrame(columns=["series", "trade_count", "series_pnl"])

if not series_pnl.empty:
    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        fig = px.bar(
            series_pnl,
            x="series",
            y="series_pnl",
            color="series_pnl",
            color_continuous_scale=["red", "gray", "green"],
            title="P&L by Series",
        )
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with col_table:
        table_df = series_pnl.rename(
            columns={"series": "Series", "trade_count": "Trades", "series_pnl": "P&L"}
        )
        st.dataframe(table_df, use_container_width=True, hide_index=True)
else:
    st.info("No trades yet")

st.divider()
st.subheader("Forecast Accuracy by Series")

forecast_data = fetch_query(
    db_handle,
    """
    SELECT
        series,
        COUNT(*) AS predictions,
        SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END) AS completed,
        AVG(
            CASE
                WHEN outcome IS NOT NULL AND
                     (
                        (forecast_prob > 1 AND ((outcome = 1 AND forecast_prob > 50) OR (outcome = 0 AND forecast_prob < 50)))
                        OR
                        (forecast_prob <= 1 AND ((outcome = 1 AND forecast_prob > 0.5) OR (outcome = 0 AND forecast_prob < 0.5)))
                     )
                THEN 1 ELSE 0
            END
        ) AS accuracy
    FROM forecast_predictions
    WHERE outcome IS NOT NULL
    GROUP BY series
    ORDER BY accuracy DESC
    LIMIT 20
    """,
)

if not forecast_data.empty:
    forecast_data["accuracy_pct"] = forecast_data["accuracy"].apply(
        lambda x: f"{x * 100:.1f}%" if pd.notna(x) else "N/A"
    )
    st.dataframe(
        forecast_data[["series", "predictions", "completed", "accuracy_pct"]].rename(
            columns={
                "series": "Series",
                "predictions": "Total Preds",
                "completed": "Completed",
                "accuracy_pct": "Accuracy",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No forecast accuracy data yet")

st.divider()
st.subheader("Open Positions")

positions = fetch_query(
    db_handle,
    """
    SELECT
        ticker,
        qty,
        avg_price,
        CASE
            WHEN qty > 0 THEN qty * (100 - avg_price) / 100.0
            WHEN qty < 0 THEN qty * (avg_price) / 100.0
            ELSE 0
        END AS unrealized_pnl
    FROM positions
    WHERE qty != 0
    ORDER BY ABS(unrealized_pnl) DESC
    LIMIT 50
    """,
)

if not positions.empty:
    positions["avg_price"] = positions["avg_price"].map(lambda x: f"{x:.1f}c")
    positions["unrealized_pnl"] = positions["unrealized_pnl"].map(lambda x: f"${x:.2f}")
    st.dataframe(
        positions[["ticker", "qty", "avg_price", "unrealized_pnl"]].rename(
            columns={
                "ticker": "Ticker",
                "qty": "Qty",
                "avg_price": "Avg Price",
                "unrealized_pnl": "Unrealized P&L",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No open positions")

if db_handle["kind"] == "sqlite" and db_handle.get("conn") is not None:
    db_handle["conn"].close()

st.divider()
col_time, col_refresh = st.columns([4, 1])
with col_time:
    st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
with col_refresh:
    if st.button("Refresh", use_container_width=True):
        st.rerun()
