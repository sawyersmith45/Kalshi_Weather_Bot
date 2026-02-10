#!/usr/bin/env python
"""Quick script to check current P&L without running the full bot."""

import sys
from datetime import timedelta
from src.db import open_db, compute_pnl_metrics, compute_pnl_by_series

def main():
    lookback = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    
    conn = open_db("data/kalshi_quotes.sqlite")
    
    pnl = compute_pnl_metrics(conn, lookback_hours=lookback)
    by_series = compute_pnl_by_series(conn, lookback_hours=lookback)
    
    print(f"\n=== P&L Summary (last {lookback}h) ===")
    print(f"Total P&L: ${pnl['total_pnl']:.2f}")
    print(f"Trades: {pnl['count_trades']} ({pnl['count_wins']} wins, {pnl['count_losses']} losses)")
    if pnl['count_trades'] > 0:
        print(f"Win Rate: {pnl['win_rate']:.1f}%")
        print(f"Avg Win: ${pnl['avg_win']:.2f} | Avg Loss: ${pnl['avg_loss']:.2f}")
        print(f"Max Win: ${pnl['max_win']:.2f} | Max Loss: ${pnl['max_loss']:.2f}")
    
    print(f"\n=== By Series ===")
    for series in sorted(by_series.keys()):
        print(f"{series}: ${by_series[series]:.2f}")
    
    print(f"\n=== By Ticker ===")
    for ticker, pnl_val in sorted(pnl['pnl_by_ticker'].items(), key=lambda x: x[1], reverse=True):
        print(f"{ticker}: ${pnl_val:.2f}")
    
    conn.close()

if __name__ == "__main__":
    main()
