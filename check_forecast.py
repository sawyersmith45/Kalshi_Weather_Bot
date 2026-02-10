#!/usr/bin/env python
"""Quick script to check forecast accuracy dashboard without running the full bot."""

import sys
from src.db import (
    open_db,
    compute_provider_accuracy,
    get_provider_rankings,
    compute_forecast_method_accuracy,
)

def main():
    lookback = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    
    conn = open_db("data/kalshi_quotes.sqlite")
    
    print(f"\n=== Forecast Accuracy Dashboard (last {lookback}h) ===\n")
    
    # Provider accuracy
    rankings = get_provider_rankings(conn, lookback_hours=lookback)
    provider_acc = compute_provider_accuracy(conn, lookback_hours=lookback)
    
    if rankings:
        print("=== Provider Rankings (by RMSE - lower is better) ===")
        for i, (provider, rmse, count) in enumerate(rankings, 1):
            if count > 0:
                m = provider_acc.get(provider, {})
                print(f"{i}. {provider:20} RMSE={rmse:7.3f} Bias={m.get('bias', 0):7.2f} MAE={m.get('mae', 0):7.2f} [n={count}]")
    else:
        print("No provider data available yet")
    
    # Forecast method accuracy
    method_acc = compute_forecast_method_accuracy(conn, lookback_hours=lookback)
    if method_acc:
        print(f"\n=== Forecast Method Accuracy ===")
        for method in sorted(method_acc.keys()):
            m = method_acc[method]
            print(f"{method:15} Brier={m['brier']:.4f} Accuracy={m['accuracy']:6.1f}% [n={m['count']}]")
    else:
        print("No forecast method data available yet")
    
    conn.close()

if __name__ == "__main__":
    main()
