#!/usr/bin/env python3
"""
Check trade fills and execution quality.
Shows recent trades, average fill prices vs fair value, slippage.
"""

import sys
from pathlib import Path
import sqlite3
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import open_db
from dotenv import load_dotenv

load_dotenv()

def main():
    db_path = Path(__file__).parent.parent / "data" / "kalshi_quotes.sqlite"
    
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)
    
    conn = open_db(db_path)
    
    print("\n" + "="*120)
    print("TRADE FILLS & EXECUTION QUALITY")
    print("="*120 + "\n")
    
    # 1. Recent trades
    print("[RECENT TRADES]")
    print("-" * 120)
    
    recent = conn.execute("""
        SELECT 
            ts_utc,
            ticker,
            side,
            qty,
            price,
            CASE WHEN side = 'BUY' THEN -qty * price / 100.0 ELSE qty * price / 100.0 END as pnl
        FROM trades
        ORDER BY ts_utc DESC
        LIMIT 20
    """).fetchall()
    
    if recent:
        for ts, ticker, side, qty, price, pnl in recent:
            ts_fmt = ts[:19] if ts else "?"
            print(f"  {ts_fmt}  {side:4s} {qty:3d}x {ticker:25s} @ {price:6.1f}c  PnL: ${pnl:+7.2f}")
    else:
        print("  No trades yet")
    
    print()
    
    # 2. Fill statistics by side
    print("[FILL STATISTICS]")
    print("-" * 120)
    
    stats = conn.execute("""
        SELECT 
            side,
            COUNT(*) as count,
            AVG(price) as avg_price,
            MIN(price) as min_price,
            MAX(price) as max_price
        FROM trades
        GROUP BY side
    """).fetchall()
    
    for side, count, avg_px, min_px, max_px in stats:
        print(f"  {side:4s}: {count:4d} trades | avg=${avg_px/100:6.2f} | range ${min_px/100:6.2f} - ${max_px/100:6.2f}")
    
    print()
    
    # 3. Profitability by time of day
    print("[PROFITABILITY BY HOUR]")
    print("-" * 120)
    
    hourly = conn.execute("""
        SELECT 
            SUBSTR(ts_utc, 12, 2) as hour,
            COUNT(*) as trades,
            SUM(CASE WHEN side = 'BUY' THEN -qty * price / 100.0 ELSE qty * price / 100.0 END) as pnl
        FROM trades
        GROUP BY hour
        ORDER BY hour ASC
    """).fetchall()
    
    for hour, count, pnl in hourly:
        pnl = pnl or 0
        status = "✓" if pnl > 0 else "✗"
        print(f"  {hour}:00 UTC {status} {count:3d} trades | P&L: ${pnl:+7.2f}")
    
    print()
    
    # 4. Top performers / losers
    print("[TOP PERFORMERS (Last 50 trades)]")
    print("-" * 120)
    
    top_winners = conn.execute("""
        SELECT 
            ticker,
            COUNT(*) as count,
            SUM(CASE WHEN side = 'BUY' THEN -qty * price / 100.0 ELSE qty * price / 100.0 END) as pnl
        FROM trades
        ORDER BY ts_utc DESC
        LIMIT 50
    """).fetchall()
    
    if top_winners:
        # Group by ticker from the recent trades
        from collections import defaultdict
        by_ticker = defaultdict(tuple)
        for t, c, p in top_winners:
            by_ticker[t] = (c, p)
        
        print("  WINNERS:")
        for ticker, (count, pnl) in sorted(by_ticker.items(), key=lambda x: x[1][1], reverse=True)[:10]:
            print(f"    {ticker:25s} {count:2d} trades | P&L: ${pnl:+7.2f}")
    else:
        print("    No data")
    
    print()
    
    # 5. Series breakdown
    print("[TRADES BY SERIES (Last 100)]")
    print("-" * 120)
    
    series_stats = conn.execute("""
        SELECT 
            SUBSTR(ticker, 1, INSTR(ticker, '-') - 1) as series,
            COUNT(*) as trades,
            SUM(CASE WHEN side = 'BUY' THEN -qty * price / 100.0 ELSE qty * price / 100.0 END) as pnl
        FROM (
            SELECT * FROM trades ORDER BY ts_utc DESC LIMIT 100
        )
        GROUP BY series
        ORDER BY pnl DESC
    """).fetchall()
    
    for series, count, pnl in (series_stats or []):
        pnl = pnl or 0
        status = "✓" if pnl > 0 else "✗"
        print(f"  {series:10s} {status} {count:3d} trades | P&L: ${pnl:+7.2f}")
    
    print()
    print("="*120)

if __name__ == "__main__":
    main()
