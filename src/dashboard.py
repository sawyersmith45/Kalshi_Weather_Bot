#!/usr/bin/env python3
"""
Kalshi Trading Bot - Live Dashboard
Displays portfolio performance, P&L, forecast accuracy, and trading activity.
Deploy to Streamlit Cloud for resume showcase.
"""

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
import os

# --- Config ---
load_dotenv()
DB_PATH = Path(__file__).parent.parent / "data" / "kalshi_quotes.sqlite"
MAX_RISK = float(os.getenv("MAX_TOTAL_RISK_DOLLARS", "60.0"))

# --- Setup ---
st.set_page_config(page_title="Trading Bot Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("🎯 Kalshi Trading Bot - Live Dashboard")

# Add manual refresh button
if st.button("🔄 Refresh Now", key="refresh_btn"):
    st.rerun()

# Create fresh DB connection on each run (no caching)
def get_db():
    return sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=10)

conn = get_db()

def fetch_query(sql, params=()):
    """Fetch query results into DataFrame"""
    try:
        return pd.read_sql_query(sql, conn, params=params)
    except Exception as e:
        st.error(f"Query error: {e}")
        return pd.DataFrame()

# --- Metrics ---
col1, col2, col3, col4, col5 = st.columns(5)

# Total P&L
pnl_query = """
    SELECT 
        SUM(CASE WHEN side = 'BUY' THEN -qty * price / 100.0 ELSE qty * price / 100.0 END) as realized_pnl
    FROM trades
"""
realized_pnl = fetch_query(pnl_query)['realized_pnl'].iloc[0] if not fetch_query(pnl_query).empty else 0
realized_pnl = realized_pnl or 0

# Unrealized P&L (only active markets with recent quotes)
unrealized_query = """
    SELECT 
        COALESCE(SUM(
            CASE 
                WHEN qty > 0 THEN qty * (100 - avg_price) / 100.0
                WHEN qty < 0 THEN qty * (avg_price - 0) / 100.0
                ELSE 0
            END
        ), 0) as unrealized_pnl
    FROM positions p
    WHERE qty != 0
    AND p.ticker IN (
        SELECT DISTINCT ticker FROM quotes
        WHERE ts_utc > datetime('now', '-1 hour')
    )
"""
unrealized_pnl = fetch_query(unrealized_query)['unrealized_pnl'].iloc[0] if not fetch_query(unrealized_query).empty else 0

total_pnl = realized_pnl + unrealized_pnl

with col1:
    st.metric("Total P&L", f"${total_pnl:.2f}", delta=f"{total_pnl/abs(total_pnl+0.001)*100:.1f}%" if total_pnl != 0 else "0%")

# Win Rate
trades_df = fetch_query("""
    SELECT 
        (side = 'BUY') as is_buy,
        price,
        qty,
        ts_utc
    FROM trades
    ORDER BY ts_utc DESC
    LIMIT 1000
""")

if not trades_df.empty:
    # Simple win rate: assume selling > buying price is a win
    wins = len(trades_df[trades_df['is_buy'] == False]) 
    total_trades = len(trades_df)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
else:
    win_rate = 0

with col2:
    st.metric("Win Rate", f"{win_rate:.1f}%", delta=f"{int(total_trades)} trades" if not trades_df.empty else "0 trades")

# Open Positions
pos_df = fetch_query("SELECT COUNT(*) as cnt FROM positions WHERE qty != 0")
open_pos = pos_df['cnt'].iloc[0] if not pos_df.empty else 0

with col3:
    st.metric("Open Positions", int(open_pos))

# Portfolio Risk
risk_query = """
    SELECT COALESCE(SUM(
        CASE 
            WHEN qty > 0 THEN qty * avg_price / 100.0
            WHEN qty < 0 THEN (-qty) * (100 - avg_price) / 100.0
            ELSE 0
        END
    ), 0) as total_risk
    FROM positions WHERE qty != 0
"""
risk_df = fetch_query(risk_query)
total_risk = risk_df['total_risk'].iloc[0] if not risk_df.empty else 0
risk_pct = (total_risk / MAX_RISK * 100) if MAX_RISK > 0 else 0

with col4:
    st.metric("Portfolio Risk", f"${total_risk:.2f}", delta=f"{risk_pct:.0f}% of ${MAX_RISK}")

# Sharpe Ratio (approximation)
daily_pnl = fetch_query("""
        SELECT DATE(ts_utc) as date, SUM(CASE WHEN side = 'BUY' THEN -qty * price / 100.0 ELSE qty * price / 100.0 END) as daily_pnl
        FROM trades
        GROUP BY DATE(ts_utc)
    LIMIT 30
""")

if not daily_pnl.empty and len(daily_pnl) > 1:
    daily_returns = daily_pnl['daily_pnl'].values
    daily_std = np.std(daily_returns) if len(daily_returns) > 1 else 1
    sharpe = (np.mean(daily_returns) / daily_std * np.sqrt(252)) if daily_std > 0 else 0
else:
    sharpe = 0

with col5:
    st.metric("Sharpe Ratio (30d)", f"{sharpe:.2f}")

st.divider()

# --- Charts ---
col_left, col_right = st.columns(2)

# Equity Curve
with col_left:
    st.subheader("📈 Equity Curve")
    
    equity_data = fetch_query("""
        SELECT 
            DATE(ts_utc) as date,
            SUM(CASE WHEN side = 'BUY' THEN -qty * price / 100.0 ELSE qty * price / 100.0 END) as daily_pnl
        FROM trades
        GROUP BY DATE(ts_utc)
        ORDER BY date ASC
    """)
    
    if not equity_data.empty:
        equity_data['cumulative_pnl'] = equity_data['daily_pnl'].cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_data['date'],
            y=equity_data['cumulative_pnl'],
            mode='lines+markers',
            name='Equity',
            line=dict(color='#1f77b4', width=2),
            fill='tozeroy'
        ))
        fig.update_layout(
            height=350,
            hovermode='x unified',
            margin=dict(l=0, r=0, t=20, b=0),
            yaxis_title="P&L ($)",
            xaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trades yet")

# Daily P&L
with col_right:
    st.subheader("📊 Daily P&L")
    
    if not daily_pnl.empty:
        colors = ['green' if x > 0 else 'red' for x in daily_pnl['daily_pnl']]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=daily_pnl['date'],
            y=daily_pnl['daily_pnl'],
            marker_color=colors,
            name='Daily P&L'
        ))
        fig.update_layout(
            height=350,
            hovermode='x',
            margin=dict(l=0, r=0, t=20, b=0),
            yaxis_title="P&L ($)",
            xaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trades yet")

st.divider()

# --- Recent Trades ---
st.subheader("📝 Recent Trades (Last 20)")

recent_trades = fetch_query("""
    SELECT 
        ts_utc as timestamp,
        ticker,
        side,
        qty,
        price,
        CASE WHEN side = 'BUY' THEN -qty * price / 100.0 ELSE qty * price / 100.0 END as pnl
    FROM trades
    ORDER BY ts_utc DESC
    LIMIT 20
""")

if not recent_trades.empty:
    recent_trades['timestamp'] = pd.to_datetime(recent_trades['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    recent_trades['price'] = recent_trades['price'].apply(lambda x: f"{x:.1f}¢")
    recent_trades['pnl'] = recent_trades['pnl'].apply(lambda x: f"${x:.2f}")
    
    st.dataframe(
        recent_trades[['timestamp', 'ticker', 'side', 'qty', 'price', 'pnl']],
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No trades yet")

st.divider()

# --- Series Breakdown ---
st.subheader("🏙️ Performance by Series")

series_pnl = fetch_query("""
    SELECT 
        SUBSTR(ticker, 1, INSTR(ticker, '-') - 1) as series,
        COUNT(*) as trade_count,
        SUM(CASE WHEN side = 'BUY' THEN -qty * price / 100.0 ELSE qty * price / 100.0 END) as series_pnl
    FROM trades
    GROUP BY series
    ORDER BY series_pnl DESC
""")

if not series_pnl.empty:
    col_chart, col_table = st.columns([2, 1])
    
    with col_chart:
        fig = px.bar(
            series_pnl,
            x='series',
            y='series_pnl',
            color='series_pnl',
            color_continuous_scale=['red', 'gray', 'green'],
            title="P&L by City Series"
        )
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col_table:
        st.dataframe(
            series_pnl[['series', 'trade_count', 'series_pnl']].rename(columns={
                'series': 'Series',
                'trade_count': 'Trades',
                'series_pnl': 'P&L'
            }),
            use_container_width=True,
            hide_index=True
        )
else:
    st.info("No trades yet")

st.divider()

# --- Forecast Accuracy ---
st.subheader("🎯 Forecast Accuracy by Series")

forecast_data = fetch_query("""
    SELECT 
        series,
        COUNT(*) as predictions,
        SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END) as completed,
        AVG(CASE WHEN outcome IS NOT NULL AND 
                      ((outcome = 1 AND forecast_prob > 50) OR (outcome = 0 AND forecast_prob < 50))
                 THEN 1 ELSE 0 END) as accuracy
    FROM forecast_predictions
    WHERE outcome IS NOT NULL
    GROUP BY series
    ORDER BY accuracy DESC
    LIMIT 20
""")

if not forecast_data.empty:
    forecast_data['accuracy_pct'] = forecast_data['accuracy'].apply(lambda x: f"{x*100:.1f}%" if x else "N/A")
    
    st.dataframe(
        forecast_data[['series', 'predictions', 'completed', 'accuracy_pct']].rename(columns={
            'series': 'Series',
            'predictions': 'Total Preds',
            'completed': 'Completed',
            'accuracy_pct': 'Accuracy'
        }),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No forecast accuracy data yet")

st.divider()

# --- Current Positions ---
st.subheader("💼 Open Positions")

positions = fetch_query("""
    SELECT 
        p.ticker,
        qty,
        avg_price,
        CASE WHEN qty > 0 THEN qty * (100 - avg_price) / 100.0
             WHEN qty < 0 THEN qty * (avg_price) / 100.0
             ELSE 0
        END as unrealized_pnl
    FROM positions p
    WHERE qty != 0
    AND p.ticker IN (
        SELECT DISTINCT ticker FROM quotes
        WHERE ts_utc > datetime('now', '-1 hour')
    )
    ORDER BY ABS(unrealized_pnl) DESC
    LIMIT 50
""")

if not positions.empty:
    positions['avg_price'] = positions['avg_price'].apply(lambda x: f"{x:.1f}¢")
    positions['unrealized_pnl'] = positions['unrealized_pnl'].apply(lambda x: f"${x:.2f}")
    
    st.dataframe(
        positions[['ticker', 'qty', 'avg_price', 'unrealized_pnl']].rename(columns={
            'ticker': 'Ticker',
            'qty': 'Qty',
            'avg_price': 'Avg Price',
            'unrealized_pnl': 'Unrealized P&L'
        }),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No open positions")

# --- Footer ---
st.divider()
col_time, col_refresh = st.columns([4, 1])

with col_time:
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")

with col_refresh:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()
