#!/usr/bin/env python3
"""Quick debug script to check database content"""
import sqlite3
from pathlib import Path

db_path = Path("data/kalshi_quotes.sqlite")

conn = sqlite3.connect(str(db_path), timeout=10)
conn.row_factory = sqlite3.Row

# Check latest trades
print("=" * 80)
print("LATEST TRADES (Last 5)")
print("=" * 80)
trades = conn.execute("""
    SELECT ts_utc, ticker, side, qty, price
    FROM trades
    ORDER BY ts_utc DESC
    LIMIT 5
""").fetchall()

for row in trades:
    print(f"{row['ts_utc']} | {row['ticker']:15} | {row['side']:4} | qty={row['qty']:3} | price={row['price']:6.2f}¢")

print()
print("=" * 80)
print("OPEN POSITIONS (Current)")
print("=" * 80)
positions = conn.execute("""
    SELECT ticker, qty, avg_price
    FROM positions
    WHERE qty != 0
    ORDER BY ticker
    LIMIT 10
""").fetchall()

for row in positions:
    print(f"{row['ticker']:15} | qty={row['qty']:3} | avg_price={row['avg_price']:6.2f}¢")

print()
total_open = conn.execute("SELECT COUNT(*) FROM positions WHERE qty != 0").fetchone()[0]
print(f"Total open positions: {total_open}")

conn.close()
