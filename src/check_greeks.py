#!/usr/bin/env python3
"""
Greeks-lite monitoring: Portfolio delta and vol spike detection.
"""

import sys
import sqlite3
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import open_db, compute_portfolio_delta, detect_vol_spikes, get_position
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    db_path = Path(__file__).parent.parent / "data" / "kalshi_quotes.sqlite"
    
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)
    
    conn = open_db(db_path)
    
    print("\n" + "="*80)
    print("GREEKS-LITE MONITORING")
    print("="*80 + "\n")
    
    # 1. Portfolio Delta (directional exposure)
    print("[PORTFOLIO DELTA]")
    print("-" * 80)
    
    try:
        # Collect live positions
        live_pos = {}
        positions = conn.execute("SELECT ticker, qty, avg_entry_price FROM positions WHERE qty != 0").fetchall()
        for ticker, qty, avg_px in positions:
            live_pos[ticker] = (qty, avg_px)
        
        if not live_pos:
            print("No open positions\n")
        else:
            # Get latest forecasts
            forecasts = {}
            forecasts_data = conn.execute(
                "SELECT ticker, prob FROM forecasts ORDER BY ts_utc DESC LIMIT 500"
            ).fetchall()
            for ticker, prob in forecasts_data:
                if ticker not in forecasts:
                    forecasts[ticker] = prob
            
            delta_result = compute_portfolio_delta(conn, live_pos, forecasts)
            
            portfolio_delta = delta_result.get('portfolio_delta', 0.0)
            print(f"Portfolio Delta (net): {portfolio_delta:+.2f} contracts")
            print(f"  → Positive delta means portfolio benefits if probabilities increase")
            print(f"  → Negative delta means portfolio benefits if probabilities decrease\n")
            
            if delta_result.get('delta_by_series'):
                print("Delta by series:")
                for series in sorted(delta_result['delta_by_series'].keys()):
                    delta = delta_result['delta_by_series'][series]
                    print(f"  {series:20s}: {delta:+8.2f}")
                print()
    
    except Exception as e:
        print(f"Error computing portfolio delta: {e}\n")
    
    # 2. Volatility Spikes
    print("[VOLATILITY SPIKES]")
    print("-" * 80)
    
    try:
        vol_spike_threshold = float(os.getenv("VOL_SPIKE_THRESHOLD", "2.0"))
        vol_spikes = detect_vol_spikes(conn, lookback_hours=1, vol_spike_threshold=vol_spike_threshold)
        
        if not vol_spikes:
            print("No volatility data available\n")
        else:
            # Separate spikes from normal
            spikes_detected = []
            normal_vol = []
            
            for series, data in vol_spikes.items():
                if data.get('is_spike'):
                    spikes_detected.append((series, data))
                else:
                    normal_vol.append((series, data))
            
            if spikes_detected:
                print(f"⚠️  SPIKES DETECTED ({len(spikes_detected)} series):")
                for series, data in sorted(spikes_detected, key=lambda x: x[1]['spike_ratio'], reverse=True):
                    print(
                        f"  {series:20s}: recent_vol={data['recent_vol']:6.3f}, "
                        f"historical_vol={data['historical_vol']:6.3f}, "
                        f"ratio={data['spike_ratio']:5.2f}x (threshold={vol_spike_threshold:.1f}x)"
                    )
                print()
            else:
                print("✓ No volatility spikes detected\n")
            
            if normal_vol:
                print("Normal volatility series:")
                for series, data in sorted(normal_vol, key=lambda x: x[1]['recent_vol'], reverse=True):
                    print(
                        f"  {series:20s}: recent_vol={data['recent_vol']:6.3f}, "
                        f"historical_vol={data['historical_vol']:6.3f}, "
                        f"ratio={data['spike_ratio']:5.2f}x"
                    )
                print()
    
    except Exception as e:
        print(f"Error detecting volatility spikes: {e}\n")
    
    # 3. Position Summary
    print("[POSITION SUMMARY]")
    print("-" * 80)
    
    try:
        positions = conn.execute(
            """SELECT ticker, qty, avg_entry_price, 
                      ROUND(qty * avg_entry_price, 2) as notional
               FROM positions WHERE qty != 0
               ORDER BY notional DESC"""
        ).fetchall()
        
        if positions:
            total_notional = sum(p[3] for p in positions)
            print(f"{'Ticker':20s} {'Qty':>10s} {'Avg Price':>12s} {'Notional':>12s} {'% of Total':>12s}")
            print("-" * 68)
            
            for ticker, qty, avg_px, notional in positions:
                pct = (notional / total_notional * 100) if total_notional != 0 else 0
                print(f"{ticker:20s} {qty:>10d} {avg_px:>12.4f} {notional:>12.2f} {pct:>11.1f}%")
            
            print("-" * 68)
            print(f"{'TOTAL':20s} {' ':>10s} {' ':>12s} {total_notional:>12.2f}")
        else:
            print("No open positions\n")
    
    except Exception as e:
        print(f"Error fetching positions: {e}\n")
    
    conn.close()
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
