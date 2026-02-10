#!/usr/bin/env python
"""Quick script to check series correlations and portfolio concentration risk."""

import sys
from src.db import open_db, compute_series_correlations, compute_correlated_exposure, get_position

def main():
    lookback = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    
    conn = open_db("data/kalshi_quotes.sqlite")
    
    # Get current positions
    live_pos = {}
    try:
        positions = conn.execute("SELECT ticker, qty, avg_price FROM positions WHERE qty > 0").fetchall()
        for ticker, qty, avg_px in positions:
            live_pos[ticker] = (qty, avg_px)
    except Exception:
        pass
    
    print(f"\n=== Series Correlation Analysis (last {lookback}h) ===\n")
    
    # Get correlations
    corr = compute_series_correlations(conn, lookback_hours=lookback)
    
    if corr:
        print("Significant Series Correlations (|corr| > 0.3):")
        print("=" * 50)
        for (s1, s2), corr_val in sorted(corr.items(), key=lambda x: abs(x[1]), reverse=True):
            corr_type = "POSITIVE" if corr_val > 0 else "NEGATIVE"
            strength = "high" if abs(corr_val) > 0.7 else "medium"
            print(f"  {s1:15} <-> {s2:15}: {corr_val:7.3f} ({strength:8} {corr_type})")
    else:
        print("No significant correlations found (need at least 3+ trades per series)")
    
    # Get correlated exposure
    if live_pos:
        print(f"\n=== Current Portfolio Concentration Risk ===\n")
        print(f"Open Positions: {len(live_pos)} tickers")
        
        corr_exp = compute_correlated_exposure(conn, live_pos, lookback_hours=lookback)
        
        max_allowed = 60.0 * 40.0 / 100.0  # MAX_TOTAL_RISK * MAX_CORRELATED_EXPOSURE_PCT
        
        if corr_exp:
            print(f"Max Allowed Correlated Exposure: ${max_allowed:.2f}\n")
            for series in sorted(corr_exp.keys()):
                exp = corr_exp[series]
                status = "⚠️ ALERT" if exp > max_allowed else "✓ OK"
                pct = 100.0 * exp / max_allowed if max_allowed > 0 else 0
                print(f"  {series:15} ${exp:8.2f} ({pct:5.1f}%) {status}")
        else:
            print("No positions or correlations to evaluate")
    else:
        print("No open positions to analyze")
    
    conn.close()

if __name__ == "__main__":
    main()
