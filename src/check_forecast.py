#!/usr/bin/env python3
"""
Check forecast accuracy and method performance.
Shows which providers/methods are most accurate.
"""

import sys
from pathlib import Path
import sqlite3
from datetime import datetime, timedelta

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
    print("FORECAST ACCURACY & METHOD PERFORMANCE")
    print("="*120 + "\n")
    
    # 1. Forecast freshness
    print("[FORECAST FRESHNESS]")
    print("-" * 120)
    
    freshness = conn.execute("""
        SELECT 
            COUNT(*) as total_predictions,
            SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END) as resolved,
            SUM(CASE WHEN outcome IS NULL THEN 1 ELSE 0 END) as pending
        FROM forecast_predictions
    """).fetchone()
    
    total, resolved, pending = freshness or (0, 0, 0)
    resolved = resolved or 0
    pending = pending or 0
    
    print(f"  Total Predictions:  {total}")
    print(f"  Resolved:           {resolved}")
    print(f"  Pending:            {pending}")
    print()
    
    # 2. Accuracy by series
    print("[ACCURACY BY SERIES]")
    print("-" * 120)
    
    by_series = conn.execute("""
        SELECT 
            series,
            COUNT(*) as predictions,
            SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END) as resolved,
            SUM(CASE WHEN outcome IS NOT NULL AND 
                          ((outcome = 1 AND forecast_prob > 50) OR (outcome = 0 AND forecast_prob < 50))
                 THEN 1 ELSE 0 END) as correct,
            AVG(CASE WHEN outcome IS NOT NULL THEN 
                      ABS(forecast_prob - outcome) 
                 ELSE NULL END) as mae
        FROM forecast_predictions
        GROUP BY series
        ORDER BY resolved DESC
    """).fetchall()
    
    for series, preds, resolved, correct, mae in by_series:
        reso = resolved or 0
        corr = correct or 0
        mae = mae or 0
        acc = (corr / reso * 100) if reso > 0 else 0
        print(f"  {series:10s}  {preds:4d} preds | {reso:3d} resolved | Acc: {acc:6.1f}% | MAE: {mae:.3f}")
    
    print()
    
    # 3. Forecast method accuracy
    print("[FORECAST METHOD PERFORMANCE]")
    print("-" * 120)
    
    by_method = conn.execute("""
        SELECT 
            forecast_method,
            COUNT(*) as predictions,
            SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END) as resolved,
            SUM(CASE WHEN outcome IS NOT NULL AND 
                          ((outcome = 1 AND forecast_prob > 50) OR (outcome = 0 AND forecast_prob < 50))
                 THEN 1 ELSE 0 END) as correct,
            AVG(CASE WHEN outcome IS NOT NULL THEN forecast_prob ELSE NULL END) as avg_conf
        FROM forecast_predictions
        GROUP BY forecast_method
        ORDER BY resolved DESC
    """).fetchall()
    
    for method, preds, resolved, correct, avg_conf in by_method:
        reso = resolved or 0
        corr = correct or 0
        avg_conf = avg_conf or 0
        acc = (corr / reso * 100) if reso > 0 else 0
        print(f"  {method:15s}  {preds:4d} preds | {reso:3d} resolved | Acc: {acc:6.1f}% | Avg Conf: {avg_conf:.1f}%")
    
    print()
    
    # 4. Recent predictions
    print("[RECENT PREDICTIONS (Last 20)]")
    print("-" * 120)
    
    recent = conn.execute("""
        SELECT 
            ts_utc,
            ticker,
            forecast_prob,
            forecast_method,
            outcome
        FROM forecast_predictions
        ORDER BY ts_utc DESC
        LIMIT 20
    """).fetchall()
    
    for ts, ticker, prob, method, outcome in recent:
        ts_fmt = ts[:19] if ts else "?"
        outcome_str = f"{bool(outcome)}" if outcome is not None else "PENDING"
        correct = "✓" if outcome is not None and ((outcome == 1 and prob > 50) or (outcome == 0 and prob < 50)) else "✗" if outcome is not None else "?"
        print(f"  {ts_fmt} {correct} {ticker:25s} prob={prob*100:6.1f}% ({method:12s}) outcome={outcome_str}")
    
    print()
    print("="*120)

if __name__ == "__main__":
    main()
