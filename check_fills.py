#!/usr/bin/env python
"""Quick script to check fill/execution analytics without running the full bot."""

import sys
from src.db import open_db, compute_fill_analytics

def main():
    lookback = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    
    conn = open_db("data/kalshi_quotes.sqlite")
    
    fills = compute_fill_analytics(conn, lookback_hours=lookback)
    
    print(f"\n=== Fill Analytics (last {lookback}h) ===\n")
    
    if fills["total_orders"] == 0:
        print("No orders found in lookback period")
        conn.close()
        return
    
    print(f"Total Orders Placed:      {fills['total_orders']}")
    print(f"Orders with Any Fill:     {fills['filled_orders']}")
    print(f"Partial Fills:            {fills['partial_fills']}")
    print(f"Rejected (0 fill):        {fills['rejected_orders']}")
    print(f"\nFill Rate:                {fills['fill_rate']:.1f}%")
    print(f"Average Fill %:           {fills['avg_fill_pct']:.1f}%")
    print(f"\nTotal Slippage:           {fills['total_slippage_cents']:.2f}c")
    print(f"Avg Slippage per Order:   {fills['avg_slippage_per_order']:.2f}c")
    
    # By side breakdown
    if fills["by_side"]:
        print(f"\n=== Breakdown by Side ===")
        for side in ["BUY", "SELL"]:
            side_stats = fills["by_side"].get(side, {})
            if side_stats and side_stats.get("total", 0) > 0:
                total = side_stats["total"]
                filled = side_stats["filled"]
                partial = side_stats["partial"]
                fill_rate_side = 100.0 * (filled + partial) / total if total > 0 else 0
                print(f"\n{side}:")
                print(f"  Orders:           {filled} fully filled + {partial} partial = {filled + partial} / {total} ({fill_rate_side:.1f}%)")
                print(f"  Total Slippage:   {side_stats['slippage']:.2f}c")
                if total > 0:
                    print(f"  Avg Slippage:     {side_stats['slippage'] / total:.2f}c per order")
    
    conn.close()

if __name__ == "__main__":
    main()
