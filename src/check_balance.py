#!/usr/bin/env python3
"""
Check current balance, risk, and position concentration to understand why bot isn't trading.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import open_db, get_position
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    db_path = Path(__file__).parent.parent / "data" / "kalshi_quotes.sqlite"
    
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)
    
    conn = open_db(db_path)
    
    MAX_TOTAL_RISK_DOLLARS = float(os.getenv("MAX_TOTAL_RISK_DOLLARS", "60.0"))
    MIN_BALANCE_CENTS = int(os.getenv("MIN_BALANCE_CENTS", "200"))
    
    print("\n" + "="*100)
    print("BALANCE & RISK DIAGNOSTICS")
    print("="*100 + "\n")
    
    # 1. Portfolio positions
    print("[POSITIONS]")
    print("-" * 100)
    
    try:
        positions = conn.execute("""
            SELECT COUNT(*) as cnt, SUM(qty) as total_qty, SUM(qty * avg_price / 100.0) as notional
            FROM positions WHERE qty != 0
        """).fetchone()
        
        pos_count, total_qty, notional = positions
        notional = notional or 0.0
        
        print(f"Open positions: {pos_count}")
        print(f"Total contracts: {total_qty}")
        print(f"Total notional value: ${notional:.2f}")
        
        # Risk analysis (local computation to avoid importing runtime-only helpers)
        live_pos = {}
        rows = conn.execute("SELECT ticker, qty, avg_price FROM positions WHERE qty != 0").fetchall()
        for ticker, qty, avg_px in rows:
            live_pos[ticker] = (int(qty), float(avg_px))

        def _compute_risk_from_positions(positions: dict):
            # Simple worst-case per-contract loss approximation (same as in run loop)
            risk = 0.0
            for _ticker, (q, a) in positions.items():
                if q > 0:
                    risk += q * (a / 100.0)
                elif q < 0:
                    risk += (-q) * ((100.0 - a) / 100.0)
            return float(risk)

        pos_risk = _compute_risk_from_positions(live_pos)
        pend_risk = 0.0
        total_risk = pos_risk + pend_risk

        print(f"\nPortfolio Risk: ${total_risk:.2f}")
        print(f"  Position risk: ${pos_risk:.2f}")
        print(f"  Pending risk: ${pend_risk:.2f}")
        print(f"  Max allowed: ${MAX_TOTAL_RISK_DOLLARS:.2f}")
        
        pct_used = (total_risk / MAX_TOTAL_RISK_DOLLARS * 100) if MAX_TOTAL_RISK_DOLLARS > 0 else 0
        print(f"  % of budget used: {pct_used:.1f}%")
        
        if pct_used > 95:
            print(f"\n  ⚠️  RISK IS MAXED OUT - No room for new trades!")
            print(f"      Need to close ${total_risk - MAX_TOTAL_RISK_DOLLARS:.2f} in positions")
        
    except Exception as e:
        print(f"Error analyzing positions: {e}\n")
    
    # 2. Recent position activity
    print("\n[RECENT POSITION CHANGES]")
    print("-" * 100)
    
    try:
        # Find positions that are winning/losing
        positions_status = conn.execute("""
            SELECT ticker, qty, avg_price,
                   CASE WHEN qty > 0 THEN bid -- for long, bid is exit price
                        ELSE ask  -- for short, ask is exit price
                   END as current_price,
                   ROUND(qty * (CASE WHEN qty > 0 THEN (bid - avg_price)
                                     ELSE (avg_price - ask)
                                END) / 100.0, 2) as pnl
            FROM (
                SELECT p.ticker, p.qty, p.avg_price,
                       COALESCE(m.yes_bid, m.yes_ask, 0) as bid,
                       COALESCE(m.yes_ask, m.yes_bid, 100) as ask
                FROM positions p
                LEFT JOIN (
                    SELECT ticker, yes_bid, yes_ask 
                    FROM markets_snapshots
                    WHERE ts_utc = (SELECT MAX(ts_utc) FROM markets_snapshots)
                ) m ON p.ticker = m.ticker
                WHERE p.qty != 0
            )
            ORDER BY ABS(pnl) DESC
            LIMIT 20
        """).fetchall()
        
        if positions_status:
            print("Top 20 positions by P&L magnitude:")
            total_pnl = 0.0
            for ticker, qty, avg_px, curr_px, pnl in positions_status:
                curr_px = curr_px or 0
                total_pnl += (pnl or 0)
                status = "✓ WIN " if (pnl or 0) > 0 else "✗ LOSS"
                print(f"  {ticker:25s} qty={qty:3d} avg={avg_px:6.1f}c curr={curr_px:5.1f}c {status} ${pnl or 0:7.2f}")
            print(f"\nTotal P&L from open positions: ${total_pnl:.2f}")
        
    except Exception as e:
        print(f"Error analyzing P&L: {e}\n")
    
    # 3. Oldest positions
    print("\n[OLDEST POSITIONS]")
    print("-" * 100)
    
    try:
        oldest = conn.execute("""
            SELECT ticker, qty, avg_price, MIN(ts_utc) as entered_at
            FROM (
                SELECT t.ticker, t.side, t.qty as trade_qty, p.qty, p.avg_price,
                       MAX(t.ts_utc) as ts_utc
                FROM trades t
                JOIN positions p ON t.ticker = p.ticker
                WHERE p.qty != 0
                GROUP BY t.ticker
            )
            GROUP BY ticker
            ORDER BY entered_at ASC
            LIMIT 10
        """).fetchall()
        
        if oldest:
            print("Oldest 10 open positions:")
            for ticker, qty, avg_px, entered_at in oldest:
                days_old = "(unknown)" if not entered_at else entered_at.split("T")[0]
                print(f"  {ticker:25s} qty={qty:3d} avg={avg_px:6.1f}c entered={days_old}")
        
    except Exception as e:
        print(f"Error fetching oldest: {e}\n")
    
    # 4. Balance check (if we have access via DB)
    print("\n[CURRENT BALANCE]")
    print("-" * 100)
    print("(To get current balance, run with LIVE_TRADING=1 and check bot logs)")
    print(f"MIN_BALANCE_CENTS required: {MIN_BALANCE_CENTS}c = ${MIN_BALANCE_CENTS/100:.2f}")
    print(f"MAX_TOTAL_RISK_DOLLARS: ${MAX_TOTAL_RISK_DOLLARS:.2f}")
    print(f"Available for position sizing: ${MAX_TOTAL_RISK_DOLLARS - total_risk:.2f} (if balance > ${(MIN_BALANCE_CENTS + MAX_TOTAL_RISK_DOLLARS*100)/100:.2f})")
    
    conn.close()
    print("\n" + "="*100 + "\n")

if __name__ == "__main__":
    main()
