#!/usr/bin/env python3
"""
Check portfolio correlations and hedging effectiveness.
Shows correlated series exposure and diversification metrics.
"""

import sys
from pathlib import Path
import sqlite3
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import open_db, compute_series_correlations, compute_correlated_exposure
from dotenv import load_dotenv

load_dotenv()

def main():
    db_path = Path(__file__).parent.parent / "data" / "kalshi_quotes.sqlite"
    
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)
    
    conn = open_db(db_path)
    MAX_TOTAL_RISK = float(os.getenv("MAX_TOTAL_RISK_DOLLARS", "60.0"))
    MAX_CORR_PCT = float(os.getenv("MAX_CORRELATED_EXPOSURE_PCT", "30.0"))
    
    print("\n" + "="*120)
    print("PORTFOLIO CORRELATION & HEDGING ANALYSIS")
    print("="*120 + "\n")
    
    # 1. Current positions by series
    print("[POSITIONS BY SERIES]")
    print("-" * 120)
    
    positions = conn.execute("""
        SELECT 
            SUBSTR(ticker, 1, INSTR(ticker, '-') - 1) as series,
            COUNT(*) as pos_count,
            SUM(qty) as total_qty,
            SUM(qty * avg_price / 100.0) as notional,
            SUM(
                CASE 
                    WHEN qty > 0 THEN qty * (100 - avg_price) / 100.0
                    WHEN qty < 0 THEN qty * (avg_price) / 100.0
                    ELSE 0
                END
            ) as unrealized_pnl
        FROM positions
        WHERE qty != 0
        GROUP BY series
        ORDER BY notional DESC
    """).fetchall()
    
    total_exposure = 0
    for series, count, total_qty, notional, pnl in positions:
        total_exposure += notional
        pnl = pnl or 0
        print(f"  {series:10s}  {count:3d} positions | Qty: {total_qty:4d} | Notional: ${notional:7.2f} | Unrealized: ${pnl:+7.2f}")
    
    print(f"  {'TOTAL':10s}                                   Notional: ${total_exposure:7.2f}")
    print()
    
    # 2. Correlation matrix (if available)
    print("[SERIES CORRELATIONS (Last 50 trades)]")
    print("-" * 120)
    
    try:
        correls = compute_series_correlations(conn, lookback_hours=24)
        if correls:
            print("  Correlation matrix (sample):")
            series_list = sorted(set(s for pair in correls.keys() for s in pair))
            
            for s1 in series_list[:5]:  # Show first 5
                for s2 in series_list[:5]:
                    if s1 <= s2:
                        key = (s1, s2) if s1 <= s2 else (s2, s1)
                        corr = correls.get(key, correls.get((s2, s1), 0.0))
                        print(f"    {s1:10s} <-> {s2:10s} : {corr:+.3f}")
        else:
            print("  Insufficient data for correlation matrix")
    except Exception as e:
        print(f"  Unable to compute correlations: {e}")
    
    print()
    
    # 3. Correlated exposure
    print("[CORRELATED EXPOSURE CHECK]")
    print("-" * 120)
    
    try:
        live_pos = {}
        rows = conn.execute("SELECT ticker, qty, avg_price FROM positions WHERE qty != 0").fetchall()
        for ticker, qty, avg_px in rows:
            live_pos[ticker] = (int(qty), float(avg_px))
        
        corr_exposure = compute_correlated_exposure(conn, live_pos, lookback_hours=24)
        max_allowed = MAX_TOTAL_RISK * (MAX_CORR_PCT / 100.0)
        
        for series, exposure in sorted(corr_exposure.items(), key=lambda x: x[1], reverse=True):
            status = "⚠️ " if exposure > max_allowed else "✓ "
            print(f"  {status} {series:10s} exposure: ${exposure:7.2f} / ${max_allowed:7.2f} max ({exposure/max_allowed*100:5.1f}%)")
    except Exception as e:
        print(f"  Error computing correlated exposure: {e}")
    
    print()
    
    # 4. Diversification score
    print("[DIVERSIFICATION METRICS]")
    print("-" * 120)
    
    series_count = len(positions) if positions else 0
    print(f"  Active Series:       {series_count}")
    print(f"  Avg Pos per Series:  {sum(c for _, c, *_ in positions) / series_count:.1f}" if series_count > 0 else "  No positions")
    
    # Simple Herfindahl index
    if positions and total_exposure > 0:
        hh_index = sum((notional / total_exposure) ** 2 for _, _, _, notional, _ in positions)
        print(f"  Herfindahl Index:    {hh_index:.3f}  (0=perfect, 1=concentrated)")
        if hh_index < 0.2:
            print(f"                       → Well diversified")
        elif hh_index < 0.5:
            print(f"                       → Moderately concentrated")
        else:
            print(f"                       → Highly concentrated")
    
    print()
    print("="*120)

if __name__ == "__main__":
    main()
