#!/usr/bin/env python3
"""
Check P&L tracking and profitability metrics.
Shows realized/unrealized P&L, daily breakdown, Sharpe ratio.
"""

import sys
from pathlib import Path
import sqlite3
from datetime import datetime, timedelta
import numpy as np

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
    print("P&L & PROFITABILITY ANALYSIS")
    print("="*120 + "\n")
    
    # 1. Overall P&L
    print("[OVERALL P&L]")
    print("-" * 120)
    
    pnl_query = conn.execute("""
        SELECT 
            SUM(CASE WHEN side = 'BUY' THEN -qty * price / 100.0 ELSE qty * price / 100.0 END) as realized_pnl,
            COUNT(*) as trade_count
        FROM trades
    """).fetchone()
    
    realized_pnl, trade_count = (pnl_query or (0, 0))
    realized_pnl = realized_pnl or 0
    
    # Only count unrealized P&L for ACTIVE positions (tickers with recent quotes)
    unrealized_query = conn.execute("""
        SELECT 
            SUM(
                CASE 
                    WHEN qty > 0 THEN qty * (100 - avg_price) / 100.0
                    WHEN qty < 0 THEN qty * (avg_price) / 100.0
                    ELSE 0
                END
            ) as unrealized_pnl
        FROM positions p
        WHERE qty != 0
        AND p.ticker IN (
            SELECT DISTINCT ticker FROM quotes 
            WHERE ts_utc > datetime('now', '-1 hour')
        )
    """).fetchone()
    
    unrealized_pnl = (unrealized_query[0] or 0) if unrealized_query else 0
    total_pnl = realized_pnl + unrealized_pnl
    
    print(f"  Realized P&L:     ${realized_pnl:+10.2f}  ({trade_count} trades)")
    print(f"  Unrealized P&L:   ${unrealized_pnl:+10.2f}")
    print(f"  Total P&L:        ${total_pnl:+10.2f}")
    print()
    
    # 2. Daily breakdown
    print("[DAILY P&L]")
    print("-" * 120)
    
    daily = conn.execute("""
        SELECT 
            DATE(ts_utc) as date,
            COUNT(*) as trades,
            SUM(CASE WHEN side = 'BUY' THEN -qty * price / 100.0 ELSE qty * price / 100.0 END) as daily_pnl
        FROM trades
        GROUP BY DATE(ts_utc)
        ORDER BY date DESC
        LIMIT 30
    """).fetchall()
    
    daily_pnls = []
    for date, trades, pnl in daily:
        pnl = pnl or 0
        daily_pnls.append(pnl)
        status = "✓ " if pnl > 0 else "✗ "
        print(f"  {date} {status} {trades:3d} trades | P&L: ${pnl:+8.2f}")
    
    print()
    
    # 3. Statistical metrics
    print("[PERFORMANCE METRICS]")
    print("-" * 120)
    
    if daily_pnls:
        daily_array = np.array(daily_pnls)
        win_days = np.sum(daily_array > 0)
        lose_days = np.sum(daily_array < 0)
        break_even_days = np.sum(daily_array == 0)
        
        win_rate = win_days / len(daily_array) * 100
        avg_win = np.mean(daily_array[daily_array > 0]) if np.any(daily_array > 0) else 0
        avg_loss = np.mean(daily_array[daily_array < 0]) if np.any(daily_array < 0) else 0
        
        daily_std = np.std(daily_array)
        daily_mean = np.mean(daily_array)
        sharpe = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0
        
        print(f"  Win Rate:         {win_rate:6.1f}%  ({win_days} wins, {lose_days} losses, {break_even_days} break-even)")
        print(f"  Avg Win:          ${avg_win:+8.2f}")
        print(f"  Avg Loss:         ${avg_loss:+8.2f}")
        print(f"  Win/Loss Ratio:   {abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'):6.2f}x")
        print(f"  Daily Std Dev:    ${daily_std:8.2f}")
        print(f"  Sharpe Ratio:     {sharpe:6.2f}  (annualized)")
    else:
        print("  Insufficient data")
    
    print()
    
    # 4. Trade frequency
    print("[TRADE FREQUENCY]")
    print("-" * 120)
    
    if trade_count > 0:
        first_trade = conn.execute("SELECT MIN(ts_utc) FROM trades").fetchone()[0]
        last_trade = conn.execute("SELECT MAX(ts_utc) FROM trades").fetchone()[0]
        
        if first_trade and last_trade:
            from datetime import datetime
            t1 = datetime.fromisoformat(first_trade.replace('Z', '+00:00'))
            t2 = datetime.fromisoformat(last_trade.replace('Z', '+00:00'))
            elapsed_hours = (t2 - t1).total_seconds() / 3600
            
            if elapsed_hours > 0:
                trades_per_hour = trade_count / elapsed_hours if elapsed_hours > 0 else 0
                print(f"  Total Trades:     {trade_count}")
                print(f"  Time Span:        {elapsed_hours:.1f} hours")
                print(f"  Trade Rate:       {trades_per_hour:.1f} trades/hour")
    
    print()
    print("="*120)

if __name__ == "__main__":
    main()
