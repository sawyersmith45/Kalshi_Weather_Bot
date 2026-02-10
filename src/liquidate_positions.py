#!/usr/bin/env python3
"""
Liquidate old losing positions to free up capital and reset edge persistence.

This is useful when the bot gets stuck with 100+ old positions from paper trading.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import open_db, set_state, get_state
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
    print("POSITION LIQUIDATION TOOL")
    print("="*100 + "\n")
    
    # Get all open positions with age
    try:
        old_positions = conn.execute("""
            SELECT p.ticker, p.qty, p.avg_price, COALESCE(MAX(t.ts_utc), '') as last_entry
            FROM positions p
            LEFT JOIN trades t ON p.ticker = t.ticker
            WHERE p.qty != 0
            GROUP BY p.ticker
            ORDER BY last_entry ASC
            LIMIT 50
        """).fetchall()
        
        print(f"Found {len(old_positions)} open positions\n")
        
        # Show oldest positions
        print("OLDEST POSITIONS (first 20):")
        print("-" * 100)
        for i, (ticker, qty, avg_px, last_entry) in enumerate(old_positions[:20]):
            days_ago = "(unknown)"
            if last_entry:
                try:
                    entry_dt = datetime.fromisoformat(last_entry.replace("+00:00", "+00:00"))
                    now_dt = datetime.now(timezone.utc)
                    delta = (now_dt - entry_dt).days
                    days_ago = f"{delta}d ago"
                except:
                    days_ago = last_entry.split("T")[0]
            
            print(f"  {i+1:2d}. {ticker:27s} qty={qty:3d} avg={avg_px:6.1f}c  entered={days_ago}")
        
        # Ask user what to do
        print("\n" + "-" * 100)
        print("Options:")
        print("  1. Reset edge persistence (clears 'persist:' keys, allows bot to retry edges)")
        print("  2. Mark old positions for liquidation (by setting qty to 0 in DB)")
        print("  3. Show position P&L (to decide what to liquidate)")
        print("  4. Exit")
        
        choice = input("\nChoose action (1-4): ").strip()
        
        if choice == "1":
            # Reset edge persistence
            print("\nResetting edge persistence state...")
            persist_keys = conn.execute("SELECT k FROM state WHERE k LIKE 'persist:%'").fetchall()
            deleted = 0
            for (key,) in persist_keys:
                conn.execute("DELETE FROM state WHERE k = ?", (key,))
                deleted += 1
            conn.commit()
            print(f"  Deleted {deleted} persistence keys")
            print("  Bot will now retry edges from scratch on next run")
        
        elif choice == "2":
            # Let user specify which positions to liquidate
            keep_days = input("Keep positions newer than (days, e.g. 3 for last 3 days): ").strip()
            try:
                keep_days = int(keep_days)
            except:
                print("Invalid input")
                return
            
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=keep_days)
            
            positions_to_close = []
            for ticker, qty, avg_px, last_entry in old_positions:
                if last_entry:
                    try:
                        entry_dt = datetime.fromisoformat(last_entry.replace("+00:00", "+00:00"))
                        if entry_dt < cutoff_date:
                            positions_to_close.append((ticker, qty))
                    except:
                        pass
            
            if positions_to_close:
                print(f"\nWill close {len(positions_to_close)} positions:")
                for ticker, qty in positions_to_close[:10]:
                    print(f"  {ticker} (qty={qty})")
                if len(positions_to_close) > 10:
                    print(f"  ... and {len(positions_to_close) - 10} more")
                
                confirm = input("\nConfirm liquidation? (yes/no): ").strip()
                if confirm.lower() in ["yes", "y"]:
                    for ticker, qty in positions_to_close:
                        conn.execute("UPDATE positions SET qty = 0 WHERE ticker = ?", (ticker,))
                    conn.commit()
                    print(f"\n✓ Closed {len(positions_to_close)} positions")
                    print("  Bot can now enter new trades with freed capital")
            else:
                print("No positions to close")
        
        elif choice == "3":
            # Show P&L with latest prices
            print("\nFetching latest market prices...")
            # Use latest quotes table for current prices (quotes.mid_yes or best_yes_bid/ask)
            pnl_data = conn.execute("""
                SELECT p.ticker, p.qty, p.avg_price,
                       COALESCE(m.best_yes_bid, m.best_yes_ask, 0) as bid,
                       COALESCE(m.best_yes_ask, m.best_yes_bid, 100) as ask,
                       ROUND(p.qty * (CASE WHEN p.qty > 0 THEN (COALESCE(m.best_yes_bid, m.mid_yes, 0) - p.avg_price)
                                           ELSE (p.avg_price - COALESCE(m.best_yes_ask, m.mid_yes, 100))
                                      END) / 100.0, 2) as pnl
                FROM positions p
                LEFT JOIN (
                    SELECT q.ticker, q.best_yes_bid, q.best_yes_ask, q.mid_yes
                    FROM quotes q
                    WHERE q.ts_utc = (
                        SELECT MAX(q2.ts_utc) FROM quotes q2 WHERE q2.ticker = q.ticker
                    )
                ) m ON p.ticker = m.ticker
                WHERE p.qty != 0
                ORDER BY pnl ASC
            """).fetchall()
            
            print("\nAll positions (sorted by P&L, worst losses first):")
            print("-" * 100)
            total_pnl = 0.0
            for ticker, qty, avg_px, bid, ask, pnl in pnl_data:
                bid = bid or 0
                ask = ask or 0
                pnl = pnl or 0
                total_pnl += pnl
                status = "✓" if pnl > 0 else "✗"
                print(f"  {ticker:27s} qty={qty:3d} avg={avg_px:6.1f} bid={bid:5.1f} ask={ask:5.1f}  {status} ${pnl:8.2f}")
            
            print(f"\nTotal P&L: ${total_pnl:.2f}")
            
            # Suggest liquidating losers
            losers = [x for x in pnl_data if (x[5] or 0) < -0.5]  # pnl < -$0.50
            if losers:
                print(f"\n⚠️  Found {len(losers)} positions with >$0.50 loss")
                confirm = input("Close all positions with >$0.50 loss? (yes/no): ").strip()
                if confirm.lower() in ["yes", "y"]:
                    for ticker, _, _, _, _, _ in losers:
                        conn.execute("UPDATE positions SET qty = 0 WHERE ticker = ?", (ticker,))
                    conn.commit()
                    print(f"✓ Closed {len(losers)} losing positions")
    
    except Exception as e:
        print(f"Error: {e}\n")
        import traceback
        traceback.print_exc()
    
    finally:
        conn.close()
        print("\n" + "="*100 + "\n")

if __name__ == "__main__":
    main()
