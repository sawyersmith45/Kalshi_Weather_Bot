#!/usr/bin/env python3
"""
Diagnose trading activity: Why is the bot not generating trades?
Check forecast counts, edge distribution, and recent decision logs.
"""

import sys
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import open_db
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    db_path = Path(__file__).parent.parent / "data" / "kalshi_quotes.sqlite"
    
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)
    
    conn = open_db(db_path)
    
    print("\n" + "="*100)
    print("TRADING ACTIVITY DIAGNOSTICS")
    print("="*100 + "\n")
    
    # 1. Forecast count and freshness
    print("[FORECASTS]")
    print("-" * 100)
    
    try:
        # Check forecast_predictions table
        fc = conn.execute("SELECT COUNT(*) FROM forecast_predictions").fetchone()[0]
        print(f"Total forecast_predictions in DB: {fc}")
        
        # Recent forecasts
        recent = conn.execute("""
            SELECT method, COUNT(*) as count, MAX(ts_utc) as last_ts
            FROM forecast_predictions
            WHERE datetime(ts_utc) > datetime('now', '-6 hours')
            GROUP BY method
            ORDER BY count DESC
        """).fetchall()
        
        if recent:
            print(f"\nRecent forecasts (last 6 hours):")
            for method, count, last_ts in recent:
                print(f"  {method:20s}: {count:5d} forecasts  (latest: {last_ts})")
        else:
            print("  ⚠️  NO recent forecasts!")
        
        # Check for empirical forecasts specifically
        empirical_count = conn.execute("""
            SELECT COUNT(*) FROM forecast_predictions 
            WHERE method = 'empirical' AND datetime(ts_utc) > datetime('now', '-24 hours')
        """).fetchone()[0]
        print(f"\nEmpirical forecasts (last 24h): {empirical_count}")
        
    except Exception as e:
        print(f"Error checking forecasts: {e}\n")
    
    # 2. Recent trades
    print("\n[RECENT TRADES]")
    print("-" * 100)
    
    try:
        trades = conn.execute("""
            SELECT ts_utc, ticker, side, price, qty, note
            FROM trades
            ORDER BY ts_utc DESC
            LIMIT 20
        """).fetchall()
        
        if trades:
            print(f"Recent trades ({len(trades)} total):")
            for ts, ticker, side, price, qty, note in trades:
                print(f"  {ts:22s} | {ticker:20s} | {side:4s} | price={price:3d}c qty={qty:3d} | note={note}")
        else:
            print("  No trades in database")
    
    except Exception as e:
        print(f"Error checking trades: {e}\n")
    
    # 3. Position summary
    print("\n[CURRENT POSITIONS]")
    print("-" * 100)
    
    try:
        positions = conn.execute("""
            SELECT ticker, qty, avg_price
            FROM positions
            WHERE qty != 0
            ORDER BY qty DESC
        """).fetchall()
        
        if positions:
            print(f"Open positions ({len(positions)} total):")
            for ticker, qty, avg_px in positions:
                print(f"  {ticker:20s} | qty={qty:3d} | avg_price={avg_px:.1f}c")
        else:
            print("  No open positions")
    
    except Exception as e:
        print(f"Error checking positions: {e}\n")
    
    # 4. Edge persistence state
    print("\n[EDGE PERSISTENCE STATE]")
    print("-" * 100)
    
    try:
        # Get all persistence keys (edge tracking)
        persist_keys = conn.execute("""
            SELECT k, v
            FROM state
            WHERE k LIKE 'persist:%'
            ORDER BY k
            LIMIT 20
        """).fetchall()
        
        if persist_keys:
            print(f"Active edge persistence keys ({len(persist_keys)} total):")
            for key, value in persist_keys:
                # Format: persist:series:ticker:direction -> first_ts|edge_value
                parts = key.split(":")
                if len(parts) >= 4:
                    series = parts[1]
                    ticker = parts[2]
                    direction = parts[3]
                    if value:
                        ts_str, edge_str = value.split("|", 1) if "|" in value else (value, "?")
                        print(f"  {series:12s} {ticker:20s} {direction:4s} | edge={edge_str:6s} | since={ts_str}")
        else:
            print("  No active persistence states")
    
    except Exception as e:
        print(f"Error checking persistence: {e}\n")
    
    # 5. Config check
    print("\n[KEY CONFIG VALUES]")
    print("-" * 100)
    
    print(f"MIN_EDGE_CENTS: {os.getenv('MIN_EDGE_CENTS', 'not set')}")
    print(f"EMPIRICAL_MIN_MEMBERS: {os.getenv('EMPIRICAL_MIN_MEMBERS', 'not set')}")
    print(f"PERSIST_SECONDS: {os.getenv('PERSIST_SECONDS', 'not set')}")
    print(f"BASE_TARGET: {os.getenv('BASE_TARGET', 'not set')}")
    print(f"MAX_TOTAL_RISK_DOLLARS: {os.getenv('MAX_TOTAL_RISK_DOLLARS', 'not set')}")
    print(f"COST_BUFFER_CENTS: {os.getenv('COST_BUFFER_CENTS', 'not set')}")
    print(f"LOG_CANDIDATE_REJECTION: {os.getenv('LOG_CANDIDATE_REJECTION', 'not set')}")
    
    # 6. Market and weather data freshness
    print("\n[DATA FRESHNESS]")
    print("-" * 100)
    
    try:
        # Check market snapshots
        ms_count = conn.execute("SELECT COUNT(*) FROM markets_snapshots").fetchone()[0]
        ms_recent = conn.execute(
            "SELECT COUNT(*) FROM markets_snapshots WHERE datetime(ts_utc) > datetime('now', '-1 hours')"
        ).fetchone()[0]
        print(f"Market snapshots: {ms_count} total, {ms_recent} in last 1h")
        
        # Check weather features
        wf_count = conn.execute("SELECT COUNT(*) FROM weather_features").fetchone()[0]
        wf_recent = conn.execute(
            "SELECT COUNT(*) FROM weather_features WHERE datetime(ts_utc) > datetime('now', '-1 hours')"
        ).fetchone()[0]
        print(f"Weather features: {wf_count} total, {wf_recent} in last 1h")
        
        # Check provider readings
        pr_count = conn.execute("SELECT COUNT(*) FROM provider_readings").fetchone()[0]
        pr_recent = conn.execute(
            "SELECT COUNT(*) FROM provider_readings WHERE datetime(ts_utc) > datetime('now', '-1 hours')"
        ).fetchone()[0]
        print(f"Provider readings: {pr_count} total, {pr_recent} in last 1h")
        
    except Exception as e:
        print(f"Error checking data freshness: {e}\n")
    
    conn.close()
    print("\n" + "="*100 + "\n")

if __name__ == "__main__":
    main()
