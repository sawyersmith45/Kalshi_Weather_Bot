#!/usr/bin/env python3
"""Direct database check to see what data exists"""
import sqlite3
from pathlib import Path

db_path = Path("data/kalshi_quotes.sqlite")

try:
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    
    # Check latest trade
    latest_trade = conn.execute("SELECT MAX(ts_utc) FROM trades").fetchone()[0]
    total_trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    
    # Check latest quote
    latest_quote = conn.execute("SELECT MAX(ts_utc) FROM quotes").fetchone()[0]
    total_quotes = conn.execute("SELECT COUNT(*) FROM quotes").fetchone()[0]
    
    # Check positions
    total_positions = conn.execute("SELECT COUNT(*) FROM positions WHERE qty != 0").fetchone()[0]
    
    print("\n" + "="*60)
    print("DATABASE STATUS")
    print("="*60)
    print(f"Latest Trade:        {latest_trade}")
    print(f"Total Trades:        {total_trades}")
    print(f"Latest Quote:        {latest_quote}")
    print(f"Total Quotes:        {total_quotes}")
    print(f"Open Positions:      {total_positions}")
    print("="*60)
    
    # Show last 3 trades
    print("\nLast 3 trades:")
    trades = conn.execute("""
        SELECT ts_utc, ticker, side, qty, price 
        FROM trades 
        ORDER BY ts_utc DESC 
        LIMIT 3
    """).fetchall()
    
    for ts, ticker, side, qty, price in trades:
        print(f"  {ts} | {ticker} {side:4} {qty:3} @ {price:6.1f}¢")
    
    conn.close()
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
