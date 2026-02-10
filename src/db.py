# src/db.py
import sqlite3
import json


def open_db(path: str = "data/kalshi_quotes.sqlite"):
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")  # Faster writes, still safe with WAL
    conn.execute("PRAGMA foreign_keys=ON;")     # Enable referential integrity

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS quotes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            ticker TEXT NOT NULL,
            best_yes_bid INTEGER,
            best_yes_ask INTEGER,
            best_no_bid INTEGER,
            best_no_ask INTEGER,
            mid_yes REAL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_quotes_ticker ON quotes(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_quotes_ts_utc ON quotes(ts_utc DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_quotes_ticker_ts ON quotes(ticker, ts_utc DESC)")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            ticker TEXT NOT NULL,
            side TEXT NOT NULL,
            price INTEGER NOT NULL,
            qty INTEGER NOT NULL,
            note TEXT,
            CHECK(side IN ('BUY', 'SELL') OR side=''),
            CHECK(price >= 0 AND price <= 100),
            CHECK(qty >= 0)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_ts_utc ON trades(ts_utc DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_ticker_ts ON trades(ticker, ts_utc DESC)")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS positions (
            ticker TEXT PRIMARY KEY,
            qty INTEGER NOT NULL,
            avg_price REAL NOT NULL,
            CHECK(qty >= 0 AND avg_price >= 0)
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS last_trade (
            ticker TEXT PRIMARY KEY,
            ts_utc TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS state (
            k TEXT PRIMARY KEY,
            v TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS live_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            ticker TEXT NOT NULL,
            side TEXT NOT NULL,
            action TEXT NOT NULL,
            yes_price INTEGER,
            count INTEGER NOT NULL,
            order_id TEXT,
            client_order_id TEXT,
            status TEXT,
            raw_json TEXT,
            CHECK(side IN ('BUY', 'SELL') OR side=''),
            CHECK(count >= 0)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_live_orders_ticker ON live_orders(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_live_orders_ts_utc ON live_orders(ts_utc DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_live_orders_ticker_ts ON live_orders(ticker, ts_utc DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_live_orders_order_id ON live_orders(order_id)")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS markets_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            event_ticker TEXT NOT NULL,
            markets_json TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_markets_snap_event_ticker ON markets_snapshots(event_ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_markets_snap_ts_utc ON markets_snapshots(ts_utc DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_markets_snap_event_ts ON markets_snapshots(event_ticker, ts_utc DESC)")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS forecast_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            event_ticker TEXT NOT NULL,
            series TEXT NOT NULL,
            target_date TEXT NOT NULL,
            forecast_high_f REAL NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_forecast_snap_event_ticker ON forecast_snapshots(event_ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_forecast_snap_ts_utc ON forecast_snapshots(ts_utc DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_forecast_snap_event_ts ON forecast_snapshots(event_ticker, ts_utc DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_forecast_snap_series ON forecast_snapshots(series)")

    # ---- Forecast Calibration Tracking ----
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS forecast_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            ticker TEXT NOT NULL,
            series TEXT NOT NULL,
            event_ticker TEXT NOT NULL,
            strike_type TEXT NOT NULL,
            strike_floor REAL,
            strike_cap REAL,
            forecast_prob REAL NOT NULL,
            forecast_method TEXT NOT NULL,
            num_ensemble_members INTEGER,
            mu REAL,
            sigma REAL,
            outcome REAL,
            outcome_ts_utc TEXT,
            CHECK(forecast_prob >= 0 AND forecast_prob <= 1)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_forecast_pred_ticker ON forecast_predictions(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_forecast_pred_ts_utc ON forecast_predictions(ts_utc DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_forecast_pred_series ON forecast_predictions(series)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_forecast_pred_has_outcome ON forecast_predictions(outcome)")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS weather_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            series TEXT NOT NULL,
            event_ticker TEXT NOT NULL,
            target_date TEXT NOT NULL,
            mu REAL NOT NULL,
            sigma REAL NOT NULL,
            ens_members INTEGER,
            ens_std REAL,
            disagree_sigma REAL,
            front_risk REAL,
            cloud_risk REAL,
            meta_json TEXT,
            CHECK(sigma > 0)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_weather_feat_event_ticker ON weather_features(event_ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_weather_feat_series ON weather_features(series)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_weather_feat_series_target ON weather_features(series, target_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_weather_feat_ts_utc ON weather_features(ts_utc DESC)")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS provider_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            series TEXT NOT NULL,
            event_ticker TEXT NOT NULL,
            target_date TEXT NOT NULL,
            provider TEXT NOT NULL,
            mu REAL,
            std REAL,
            meta_json TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_provider_readings_event_ticker ON provider_readings(event_ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_provider_readings_series ON provider_readings(series)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_provider_readings_series_target ON provider_readings(series, target_date, provider)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_provider_readings_ts_utc ON provider_readings(ts_utc DESC)")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS outcomes (
            event_ticker TEXT PRIMARY KEY,
            series TEXT NOT NULL,
            target_date TEXT NOT NULL,
            observed_high_f REAL,
            ts_utc TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_series ON outcomes(series)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_ts_utc ON outcomes(ts_utc DESC)")

    # ---- Climatology (historical) ----
    # series = e.g. KXHIGHNY
    # doy = day-of-year 1..366
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS climo_daily (
            series TEXT NOT NULL,
            doy INTEGER NOT NULL,
            mean_high_f REAL NOT NULL,
            std_high_f REAL NOT NULL,
            n INTEGER NOT NULL,
            PRIMARY KEY (series, doy),
            CHECK(doy >= 1 AND doy <= 366)
        )
        """
    )

    conn.commit()
    return conn


def record_trade(conn, ts_utc: str, ticker: str, side: str, price: int, qty: int, note: str | None = None):
    conn.execute(
        "INSERT INTO trades(ts_utc, ticker, side, price, qty, note) VALUES (?,?,?,?,?,?)",
        (ts_utc, ticker, side, int(price), int(qty), note),
    )
    conn.commit()


def get_position(conn, ticker: str):
    row = conn.execute("SELECT qty, avg_price FROM positions WHERE ticker=?", (ticker,)).fetchone()
    if not row:
        return 0, 0.0
    return int(row[0]), float(row[1])


def upsert_position(conn, ticker: str, qty: int, avg_price: float):
    conn.execute(
        """
        INSERT INTO positions(ticker, qty, avg_price)
        VALUES (?,?,?)
        ON CONFLICT(ticker) DO UPDATE SET
            qty=excluded.qty,
            avg_price=excluded.avg_price
        """,
        (ticker, int(qty), float(avg_price)),
    )
    conn.commit()


def get_last_trade_ts(conn, ticker: str):
    row = conn.execute("SELECT ts_utc FROM last_trade WHERE ticker=?", (ticker,)).fetchone()
    return None if not row else row[0]


def set_last_trade_ts(conn, ticker: str, ts_utc: str):
    conn.execute(
        """
        INSERT INTO last_trade(ticker, ts_utc)
        VALUES (?,?)
        ON CONFLICT(ticker) DO UPDATE SET ts_utc=excluded.ts_utc
        """,
        (ticker, ts_utc),
    )
    conn.commit()


def set_state(conn, k: str, v: str):
    conn.execute(
        """
        INSERT INTO state(k, v) VALUES (?,?)
        ON CONFLICT(k) DO UPDATE SET v=excluded.v
        """,
        (k, str(v)),
    )
    conn.commit()


def get_state(conn, k: str, default: str = ""):
    row = conn.execute("SELECT v FROM state WHERE k=?", (k,)).fetchone()
    return default if not row else row[0]


def log_live_order(
    conn,
    ts_utc: str,
    ticker: str,
    side: str,
    action: str,
    yes_price,
    count: int,
    order_id=None,
    client_order_id=None,
    status=None,
    raw_json=None,
):
    conn.execute(
        """
        INSERT INTO live_orders(
            ts_utc, ticker, side, action, yes_price, count, order_id, client_order_id, status, raw_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (
            ts_utc,
            str(ticker),
            str(side),
            str(action),
            None if yes_price is None else int(yes_price),
            int(count),
            None if order_id is None else str(order_id),
            None if client_order_id is None else str(client_order_id),
            None if status is None else str(status),
            None if raw_json is None else str(raw_json),
        ),
    )
    conn.commit()


def snapshot_markets(conn, ts_utc: str, event_ticker: str, markets: list[dict]):
    conn.execute(
        "INSERT INTO markets_snapshots(ts_utc, event_ticker, markets_json) VALUES (?,?,?)",
        (ts_utc, str(event_ticker), json.dumps(markets)),
    )
    conn.commit()


def snapshot_forecast(conn, ts_utc: str, event_ticker: str, target_date: str, mu: float, series: str):
    conn.execute(
        """
        INSERT INTO forecast_snapshots(ts_utc, event_ticker, series, target_date, forecast_high_f)
        VALUES (?,?,?,?,?)
        """,
        (ts_utc, str(event_ticker), str(series), str(target_date), float(mu)),
    )
    conn.commit()


def snapshot_weather_features(
    conn,
    ts_utc: str,
    series: str,
    event_ticker: str,
    target_date: str,
    mu: float,
    sigma: float,
    ens_members: int | None = None,
    ens_std: float | None = None,
    disagree_sigma: float | None = None,
    front_risk: float | None = None,
    cloud_risk: float | None = None,
    meta: dict | None = None,
):
    conn.execute(
        """
        INSERT INTO weather_features(
            ts_utc, series, event_ticker, target_date, mu, sigma, ens_members, ens_std, disagree_sigma,
            front_risk, cloud_risk, meta_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            ts_utc,
            str(series),
            str(event_ticker),
            str(target_date),
            float(mu),
            float(sigma),
            None if ens_members is None else int(ens_members),
            None if ens_std is None else float(ens_std),
            None if disagree_sigma is None else float(disagree_sigma),
            None if front_risk is None else float(front_risk),
            None if cloud_risk is None else float(cloud_risk),
            None if meta is None else json.dumps(meta),
        ),
    )
    conn.commit()


def snapshot_provider_readings(conn, ts_utc: str, series: str, event_ticker: str, target_date: str, rows: list[dict]):
    for r in rows:
        conn.execute(
            """
            INSERT INTO provider_readings(
                ts_utc, series, event_ticker, target_date, provider, mu, std, meta_json
            ) VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                ts_utc,
                str(series),
                str(event_ticker),
                str(target_date),
                str(r.get("provider") or ""),
                None if r.get("mu") is None else float(r.get("mu")),
                None if r.get("std") is None else float(r.get("std")),
                None if r.get("meta") is None else json.dumps(r.get("meta")),
            ),
        )
    conn.commit()


def upsert_outcome(conn, event_ticker: str, series: str, target_date: str, observed_high_f: float | None, ts_utc: str):
    conn.execute(
        """
        INSERT INTO outcomes(event_ticker, series, target_date, observed_high_f, ts_utc)
        VALUES (?,?,?,?,?)
        ON CONFLICT(event_ticker) DO UPDATE SET
            series=excluded.series,
            target_date=excluded.target_date,
            observed_high_f=excluded.observed_high_f,
            ts_utc=excluded.ts_utc
        """,
        (
            event_ticker,
            series,
            target_date,
            None if observed_high_f is None else float(observed_high_f),
            ts_utc,
        ),
    )
    conn.commit()


def get_latest_forecast_for_event(conn, event_ticker: str):
    row = conn.execute(
        """
        SELECT series, target_date, forecast_high_f, ts_utc
        FROM forecast_snapshots
        WHERE event_ticker=?
        ORDER BY id DESC
        LIMIT 1
        """,
        (event_ticker,),
    ).fetchone()
    if not row:
        return None
    return {"series": row[0], "target_date": row[1], "mu": row[2], "ts_utc": row[3]}


def get_latest_provider_readings_for_event(conn, event_ticker: str):
    rows = conn.execute(
        """
        SELECT provider, mu, std
        FROM provider_readings
        WHERE event_ticker=?
        ORDER BY id DESC
        """,
        (event_ticker,),
    ).fetchall()
    out = {}
    for prov, mu, std in rows:
        if prov not in out:
            out[str(prov)] = {"mu": mu, "std": std}
    return out


def vacuum_and_analyze(conn, verbose=False):
    """Optimize database: vacuum fragmentation and recompute statistics.
    
    Should be run periodically (e.g., daily or when DB size balloons).
    """
    if verbose:
        print("[DB] Running VACUUM...")
    conn.execute("VACUUM")
    if verbose:
        print("[DB] Running ANALYZE...")
    conn.execute("ANALYZE")
    conn.commit()
    if verbose:
        print("[DB] Optimization complete")


def cleanup_old_snapshots(conn, keep_hours: int = 72, verbose=False):
    """Delete snapshot records older than keep_hours to prevent DB bloat.
    
    This removes old weather_features, provider_readings, markets_snapshots,
    and forecast_snapshots, keeping only the most recent keep_hours worth.
    """
    from datetime import datetime, timezone, timedelta
    
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=keep_hours)).isoformat()
    
    tables = [
        "weather_features",
        "provider_readings", 
        "markets_snapshots",
        "forecast_snapshots",
    ]
    
    total_deleted = 0
    for table in tables:
        cursor = conn.execute(f"DELETE FROM {table} WHERE ts_utc < ?", (cutoff,))
        deleted = cursor.rowcount
        total_deleted += deleted
        if verbose and deleted > 0:
            print(f"[DB] Deleted {deleted} old rows from {table}")
    
    conn.commit()
    if verbose and total_deleted > 0:
        print(f"[DB] Total deleted {total_deleted} old snapshot records")
    return total_deleted


def check_db_integrity(conn, verbose=False):
    """Run database integrity checks and report issues.
    
    Returns dict like {'errors': [...], 'warnings': [...], 'ok': bool}
    """
    issues = {"errors": [], "warnings": [], "ok": True}
    
    # Check for constraint violations
    try:
        result = conn.execute("PRAGMA integrity_check").fetchone()
        if result[0] != "ok":
            issues["errors"].append(f"Integrity check failed: {result[0]}")
            issues["ok"] = False
    except Exception as e:
        issues["warnings"].append(f"Could not run integrity check: {e}")
    
    # Check for invalid trades
    invalid_trades = conn.execute("""
        SELECT COUNT(*) FROM trades 
        WHERE side NOT IN ('BUY', 'SELL') 
           OR price < 0 OR price > 100 
           OR qty < 0
    """).fetchone()[0]
    if invalid_trades > 0:
        issues["errors"].append(f"{invalid_trades} invalid trade records found")
        issues["ok"] = False
    
    # Check for invalid positions
    invalid_pos = conn.execute("""
        SELECT COUNT(*) FROM positions 
        WHERE qty < 0 OR avg_price < 0
    """).fetchone()[0]
    if invalid_pos > 0:
        issues["errors"].append(f"{invalid_pos} invalid position records found")
        issues["ok"] = False
    
    # Check for orphaned live_orders (orders with no matching position)
    orphan_orders = conn.execute("""
        SELECT COUNT(*) FROM live_orders lo
        WHERE lo.order_id IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM positions p WHERE p.ticker = lo.ticker)
    """).fetchone()[0]
    if orphan_orders > 0:
        issues["warnings"].append(f"{orphan_orders} live orders referencing non-existent tickers")
    
    # Get database stats
    try:
        db_size = conn.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()").fetchone()[0]
        if verbose:
            print(f"[DB] Database size: {db_size / 1024 / 1024:.1f} MB")
    except Exception:
        pass
    
    if verbose:
        if issues["ok"]:
            print("[DB] ✓ Database integrity OK")
        else:
            for e in issues["errors"]:
                print(f"[DB] ERROR: {e}")
        for w in issues["warnings"]:
            print(f"[DB] WARNING: {w}")
    
    return issues


def get_db_stats(conn):
    """Return dict of database statistics."""
    stats = {}
    
    tables = [
        "quotes", "trades", "positions", "live_orders",
        "markets_snapshots", "forecast_snapshots", "weather_features",
        "provider_readings", "outcomes"
    ]
    
    for table in tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            stats[table] = count
        except Exception:
            stats[table] = 0
    
    try:
        db_size = conn.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()").fetchone()[0]
        stats["db_size_mb"] = db_size / 1024 / 1024
    except Exception:
        stats["db_size_mb"] = 0
    
    return stats


def record_forecast_prediction(conn, ts_utc: str, ticker: str, series: str, event_ticker: str,
                               strike_type: str, strike_floor: float | None, strike_cap: float | None,
                               forecast_prob: float, forecast_method: str, num_members: int | None = None,
                               mu: float | None = None, sigma: float | None = None):
    """Record a forecast prediction for calibration tracking."""
    conn.execute(
        """INSERT INTO forecast_predictions 
           (ts_utc, ticker, series, event_ticker, strike_type, strike_floor, strike_cap, 
            forecast_prob, forecast_method, num_ensemble_members, mu, sigma)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (ts_utc, ticker, series, event_ticker, strike_type, strike_floor, strike_cap,
         float(forecast_prob), forecast_method, num_members, mu, sigma)
    )
    conn.commit()


def update_forecast_outcome(conn, ticker: str, event_ticker: str, outcome: float, outcome_ts_utc: str):
    """Update outcome for all predictions of a given ticker/event_ticker.

    Converts the observed numeric outcome (e.g., observed high temperature)
    into a binary outcome per prediction based on its strike type and
    strike thresholds, then updates each prediction row individually.
    """
    try:
        rows = conn.execute(
            """SELECT id, strike_type, strike_floor, strike_cap FROM forecast_predictions
               WHERE ticker = ? AND event_ticker = ? AND outcome IS NULL""",
            (ticker, event_ticker),
        ).fetchall()
        if not rows:
            return

        for rid, strike_type, strike_floor, strike_cap in rows:
            st = (strike_type or "").lower()
            obs = float(outcome)
            result = None
            try:
                if st == "between" and strike_floor is not None and strike_cap is not None:
                    lo = float(strike_floor)
                    hi = float(strike_cap)
                    result = 1.0 if (obs >= lo and obs < hi) else 0.0
                elif st in ("greater", "greater_than", "greater_or_equal", "above") and strike_floor is not None:
                    lo = float(strike_floor)
                    result = 1.0 if (obs > lo) else 0.0
                elif st in ("less", "less_than", "less_or_equal", "below") and strike_cap is not None:
                    hi = float(strike_cap)
                    result = 1.0 if (obs < hi) else 0.0
                else:
                    # Unknown strike type — store raw numeric for inspection
                    result = obs
            except Exception:
                result = obs

            conn.execute(
                """UPDATE forecast_predictions SET outcome = ?, outcome_ts_utc = ? WHERE id = ?""",
                (result, outcome_ts_utc, rid),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def update_provider_mse(conn, series: str, event_ticker: str, observed_high_f: float, alpha: float = 0.05):
    """Update exponential-moving-average MSE per provider based on provider_readings.

    For each provider reading for the event, compute squared error between
    the provider's mu and the observed value and update stored state
    key `mse:{series}:{provider}` using EMA: new = alpha*err + (1-alpha)*old.
    Returns a dict of updated MSEs.
    """
    out = {}
    try:
        rows = conn.execute(
            "SELECT provider, mu FROM provider_readings WHERE event_ticker = ?",
            (event_ticker,),
        ).fetchall()
        if not rows:
            return out

        for provider, mu in rows:
            if provider is None or mu is None:
                continue
            try:
                mu_v = float(mu)
                err = (mu_v - float(observed_high_f)) ** 2
            except Exception:
                continue

            key = f"mse:{series}:{provider}"
            old_raw = conn.execute("SELECT v FROM state WHERE k = ?", (key,)).fetchone()
            old = None
            if old_raw and old_raw[0] is not None:
                try:
                    old = float(old_raw[0])
                except Exception:
                    old = None

            if old is None:
                new = float(err)
            else:
                new = float(alpha) * float(err) + (1.0 - float(alpha)) * float(old)

            conn.execute(
                "INSERT INTO state(k,v) VALUES (?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                (key, str(new)),
            )
            out[provider] = new
        conn.commit()
        return out
    except Exception:
        conn.rollback()
        raise


def compute_heat_map(conn, lookback_hours: int = 24, out_csv: str = "data/forecast_heatmap.csv"):
    """Compute simple heat map data (avg abs error) per ticker over recent predictions.

    Writes a CSV `out_csv` with columns: ticker, count, avg_abs_error, avg_brier
    and returns a dict mapping ticker -> (count, avg_abs_error, avg_brier).
    """
    import csv
    cutoff_clause = "datetime(outcome_ts_utc) > datetime('now', '-' || ? || ' hours')"
    rows = conn.execute(
        f"SELECT ticker, forecast_prob, outcome FROM forecast_predictions WHERE outcome IS NOT NULL AND {cutoff_clause}",
        (lookback_hours,),
    ).fetchall()
    if not rows:
        return {}

    stats = {}
    for ticker, prob, outcome in rows:
        try:
            p = float(prob)
            o = float(outcome)
        except Exception:
            continue
        if ticker not in stats:
            stats[ticker] = {"count": 0, "sum_abs": 0.0, "sum_brier": 0.0}
        stats[ticker]["count"] += 1
        stats[ticker]["sum_abs"] += abs(p - o)
        stats[ticker]["sum_brier"] += (p - o) ** 2

    out = {}
    # Write CSV
    try:
        with open(out_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["ticker", "count", "avg_abs_error", "avg_brier"])
            for t, s in stats.items():
                cnt = s["count"]
                avg_abs = s["sum_abs"] / cnt
                avg_brier = s["sum_brier"] / cnt
                writer.writerow([t, cnt, f"{avg_abs:.4f}", f"{avg_brier:.6f}"])
                out[t] = (cnt, avg_abs, avg_brier)
    except Exception:
        # If file write fails, still return computed dict
        for t, s in stats.items():
            cnt = s["count"]
            out[t] = (cnt, s["sum_abs"] / cnt, s["sum_brier"] / cnt)

    return out


def compute_calibration_metrics(conn, lookback_hours: int = 24):
    """Compute calibration metrics for predictions with outcomes.
    
    Returns:
        dict: Brier score, accuracy, log-loss, count, grouped by method
    """
    import time
    import math
    
    cutoff_ts = time.time() - (lookback_hours * 3600)
    
    rows = conn.execute("""
        SELECT forecast_prob, outcome, forecast_method 
        FROM forecast_predictions 
        WHERE outcome IS NOT NULL 
          AND datetime(outcome_ts_utc) > datetime('now', '-' || ? || ' hours')
    """, (lookback_hours,)).fetchall()
    
    if not rows:
        return {}
    
    metrics_by_method = {}
    
    for prob, outcome, method in rows:
        if method not in metrics_by_method:
            metrics_by_method[method] = {
                "count": 0,
                "brier": 0.0,  # Mean squared error
                "log_loss": 0.0,
                "accuracy": 0,
                "calibration_bins": [0] * 10,  # 10 bins for calibration curve
            }
        
        m = metrics_by_method[method]
        m["count"] += 1
        
        # Brier score: (forecast - outcome)^2
        brier_contrib = (float(prob) - float(outcome)) ** 2
        m["brier"] += brier_contrib
        
        # Log loss: -[y*log(p) + (1-y)*log(1-p)]
        p = max(1e-6, min(1 - 1e-6, float(prob)))
        out = float(outcome)
        log_loss_contrib = -(out * math.log(p) + (1 - out) * math.log(1 - p))
        m["log_loss"] += log_loss_contrib
        
        # Accuracy: 1 if |forecast - outcome| < 0.5, else 0 (rounded classification)
        if abs(float(prob) - out) < 0.5:
            m["accuracy"] += 1
        
        # Calibration bin: assign to bin 0-9 based on forecast probability
        bin_idx = min(9, int(float(prob) * 10))
        m["calibration_bins"][bin_idx] += 1
    
    # Normalize metrics
    for method, m in metrics_by_method.items():
        count = m["count"]
        if count > 0:
            m["brier"] /= count
            m["log_loss"] /= count
            m["accuracy"] /= count
    
    return metrics_by_method


class BatchTransaction:
    """Context manager for batching multiple DB writes into a single transaction.
    
    Usage:
        with BatchTransaction(conn) as bat:
            bat.execute("INSERT ...", params)
            bat.execute("UPDATE ...", params)
        # All commits at once on exit
    """
    def __init__(self, conn):
        self.conn = conn
        self.operations = []
    
    def execute(self, sql: str, params: tuple = ()):
        """Queue an execute operation (no immediate commit)."""
        self.operations.append((sql, params))
    
    def __enter__(self):
        self.conn.execute("BEGIN TRANSACTION")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Execute all queued operations
            for sql, params in self.operations:
                self.conn.execute(sql, params)
            self.conn.commit()
        else:
            self.conn.rollback()
        return False


def compute_pnl_metrics(conn, lookback_hours: int = 24):
    """Compute P&L metrics from trades table.
    
    Returns dict with keys:
    - total_pnl: sum of all closed trades
    - count_trades: number of completed round trips
    - count_wins: trades with positive P&L
    - count_losses: trades with negative P&L
    - win_rate: % of winning trades
    - avg_win: average profit on winning trades
    - avg_loss: average loss on losing trades
    - max_win: largest single win
    - max_loss: largest single loss
    - pnl_by_ticker: dict of {ticker: pnl_value}
    """
    try:
        # Get all trades in lookback period
        trades = conn.execute(
            f"""SELECT ticker, side, price, qty, ts_utc FROM trades
               WHERE datetime(ts_utc) > datetime('now', '-' || ? || ' hours')
               ORDER BY ticker, ts_utc""",
            (lookback_hours,),
        ).fetchall()
        
        if not trades:
            return {
                "total_pnl": 0.0, "count_trades": 0, "count_wins": 0, "count_losses": 0,
                "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "max_win": 0.0, "max_loss": 0.0,
                "pnl_by_ticker": {}
            }
        
        # Group by ticker and pair BUYs with SELLs (FIFO)
        pnl_list = []
        by_ticker = {}
        
        for ticker, side, price, qty, ts_utc in trades:
            if ticker not in by_ticker:
                by_ticker[ticker] = {"buys": [], "sells": []}
            
            if side == "BUY":
                by_ticker[ticker]["buys"].append((price, qty, ts_utc))
            elif side == "SELL":
                by_ticker[ticker]["sells"].append((price, qty, ts_utc))
        
        # Match buy/sell pairs
        pnl_by_ticker = {}
        for ticker, trades_dict in by_ticker.items():
            buys = trades_dict["buys"]
            sells = trades_dict["sells"]
            ticker_pnl = 0.0
            
            # Simple FIFO matching
            buy_idx, sell_idx = 0, 0
            while buy_idx < len(buys) and sell_idx < len(sells):
                buy_price, buy_qty, _ = buys[buy_idx]
                sell_price, sell_qty, _ = sells[sell_idx]
                
                match_qty = min(buy_qty, sell_qty)
                trade_pnl = (sell_price - buy_price) * match_qty / 100.0  # Convert cents to dollars
                pnl_list.append(trade_pnl)
                ticker_pnl += trade_pnl
                
                # Consume matched qty
                buys[buy_idx] = (buy_price, buy_qty - match_qty, _)
                sells[sell_idx] = (sell_price, sell_qty - match_qty, _)
                
                if buys[buy_idx][1] == 0:
                    buy_idx += 1
                if sells[sell_idx][1] == 0:
                    sell_idx += 1
            
            pnl_by_ticker[ticker] = ticker_pnl
        
        # Compute metrics
        total_pnl = sum(pnl_list)
        count_wins = len([p for p in pnl_list if p > 0])
        count_losses = len([p for p in pnl_list if p < 0])
        count_trades = len(pnl_list)
        win_rate = 100.0 * count_wins / count_trades if count_trades > 0 else 0.0
        
        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p < 0]
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        max_win = max(wins) if wins else 0.0
        max_loss = min(losses) if losses else 0.0
        
        return {
            "total_pnl": round(total_pnl, 2),
            "count_trades": count_trades,
            "count_wins": count_wins,
            "count_losses": count_losses,
            "win_rate": round(win_rate, 1),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "max_win": round(max_win, 2),
            "max_loss": round(max_loss, 2),
            "pnl_by_ticker": {k: round(v, 2) for k, v in pnl_by_ticker.items()},
        }
    except Exception as e:
        import logging
        logging.warning(f"Error computing P&L metrics: {e}")
        return {
            "total_pnl": 0.0, "count_trades": 0, "count_wins": 0, "count_losses": 0,
            "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "max_win": 0.0, "max_loss": 0.0,
            "pnl_by_ticker": {}
        }


def compute_pnl_by_series(conn, lookback_hours: int = 24):
    """Compute P&L grouped by series (from ticker name).
    
    Extracts series name from ticker prefix and sums P&L.
    Returns dict mapping series_name -> pnl_value.
    """
    metrics = compute_pnl_metrics(conn, lookback_hours)
    by_series = {}
    
    for ticker, pnl in metrics.get("pnl_by_ticker", {}).items():
        # Extract series: ticker format is like KXHIGHNY_YYYYMDD_CEILING_80
        parts = ticker.split("_")
        series = parts[0] if parts else ticker
        
        if series not in by_series:
            by_series[series] = 0.0
        by_series[series] += pnl
    
    return {k: round(v, 2) for k, v in by_series.items()}


def compute_provider_accuracy(conn, lookback_hours: int = 24):
    """Compute accuracy metrics per weather provider.
    
    Returns dict mapping provider_name -> {
        'count': predictions made,
        'rmse': root mean squared error,
        'bias': mean error (positive = over-forecast),
        'mae': mean absolute error,
        'brier': mean Brier score (for binary outcomes),
    }
    """
    try:
        # Get provider readings with corresponding outcomes
        rows = conn.execute(
            f"""SELECT pr.provider, pr.mu, fp.outcome 
               FROM provider_readings pr
               JOIN forecast_predictions fp ON pr.event_ticker = fp.event_ticker 
               WHERE pr.mu IS NOT NULL AND fp.outcome IS NOT NULL
               AND datetime(fp.outcome_ts_utc) > datetime('now', '-' || ? || ' hours')""",
            (lookback_hours,),
        ).fetchall()
        
        if not rows:
            return {}
        
        # Accumulate metrics per provider
        metrics = {}
        for provider, mu, outcome in rows:
            if provider not in metrics:
                metrics[provider] = {
                    "count": 0,
                    "sum_error": 0.0,
                    "sum_sq_error": 0.0,
                    "sum_abs_error": 0.0,
                    "sum_brier": 0.0,
                }
            
            try:
                mu = float(mu)
                outcome = float(outcome)
            except (ValueError, TypeError):
                continue
            
            # For numeric (high temp): compute RMSE, bias, MAE
            error = mu - outcome
            metrics[provider]["sum_error"] += error
            metrics[provider]["sum_sq_error"] += error ** 2
            metrics[provider]["sum_abs_error"] += abs(error)
            
            # For binary outcome: compute Brier score
            # If outcome is 0 or 1, treat as probability match
            if outcome in (0.0, 1.0):
                metrics[provider]["sum_brier"] += (mu / 100.0 - outcome) ** 2
            
            metrics[provider]["count"] += 1
        
        # Compute final metrics
        out = {}
        for provider, m in metrics.items():
            cnt = m["count"]
            if cnt == 0:
                continue
            
            rmse = (m["sum_sq_error"] / cnt) ** 0.5
            bias = m["sum_error"] / cnt
            mae = m["sum_abs_error"] / cnt
            brier = m["sum_brier"] / cnt
            
            out[provider] = {
                "count": cnt,
                "rmse": round(rmse, 3),
                "bias": round(bias, 2),
                "mae": round(mae, 2),
                "brier": round(brier, 4),
            }
        
        return out
    except Exception as e:
        import logging
        logging.warning(f"Error computing provider accuracy: {e}")
        return {}


def get_provider_rankings(conn, lookback_hours: int = 24):
    """Get providers ranked by overall accuracy (lower RMSE is better).
    
    Returns list of (provider_name, rmse, count) tuples sorted by RMSE ascending.
    """
    acc = compute_provider_accuracy(conn, lookback_hours)
    ranked = sorted([(p, m["rmse"], m["count"]) for p, m in acc.items()], 
                    key=lambda x: x[1])
    return ranked


def compute_forecast_method_accuracy(conn, lookback_hours: int = 24):
    """Compute accuracy by forecast method (empirical, students_t, normal).
    
    Returns dict mapping method_name -> {
        'count': number of predictions,
        'brier': Brier score,
        'accuracy': % correct (for binary outcomes),
    }
    """
    try:
        rows = conn.execute(
            f"""SELECT forecast_method, forecast_prob, outcome 
               FROM forecast_predictions 
               WHERE forecast_method IS NOT NULL AND outcome IS NOT NULL
               AND datetime(outcome_ts_utc) > datetime('now', '-' || ? || ' hours')""",
            (lookback_hours,),
        ).fetchall()
        
        if not rows:
            return {}
        
        metrics = {}
        for method, prob, outcome in rows:
            if method not in metrics:
                metrics[method] = {"count": 0, "sum_brier": 0.0, "correct": 0}
            
            try:
                prob = float(prob) / 100.0  # Convert to 0-1
                outcome = float(outcome)
            except (ValueError, TypeError):
                continue
            
            brier_score = (prob - outcome) ** 2
            metrics[method]["sum_brier"] += brier_score
            metrics[method]["count"] += 1
            
            # Count correct predictions (if prob > 0.5 and outcome == 1, or vice versa)
            if (prob > 0.5 and outcome == 1.0) or (prob <= 0.5 and outcome == 0.0):
                metrics[method]["correct"] += 1
        
        out = {}
        for method, m in metrics.items():
            cnt = m["count"]
            if cnt == 0:
                continue
            
            accuracy = 100.0 * m["correct"] / cnt
            brier = m["sum_brier"] / cnt
            out[method] = {
                "count": cnt,
                "brier": round(brier, 4),
                "accuracy": round(accuracy, 1),
            }
        
        return out
    except Exception as e:
        import logging
        logging.warning(f"Error computing forecast method accuracy: {e}")
        return {}


def compute_fill_analytics(conn, lookback_hours: int = 24):
    """Compute fill/execution analytics from live_orders table.
    
    Returns dict with:
    - total_orders: number of orders placed
    - filled_orders: orders with non-zero fills
    - partial_fills: orders filled < 100%
    - rejected_orders: orders with 0 fills
    - fill_rate: % of orders that got any fill
    - avg_fill_pct: average % of order qty that filled
    - total_slippage_cents: sum of (filled_px - requested_px) * qty
    - avg_slippage_per_order: mean slippage per order
    by_side: breakdown by BUY/SELL
    """
    try:
        # Get all live orders placed in lookback period
        orders = conn.execute(
            f"""SELECT id, ticker, side, yes_price, count, status, raw_json
               FROM live_orders
               WHERE datetime(ts_utc) > datetime('now', '-' || ? || ' hours')
               AND action = 'place'""",
            (lookback_hours,),
        ).fetchall()
        
        if not orders:
            return {
                "total_orders": 0,
                "filled_orders": 0,
                "partial_fills": 0,
                "rejected_orders": 0,
                "fill_rate": 0.0,
                "avg_fill_pct": 0.0,
                "total_slippage_cents": 0.0,
                "avg_slippage_per_order": 0.0,
                "by_side": {"BUY": {}, "SELL": {}},
            }
        
        import json
        
        stats = {
            "total": 0,
            "filled": 0,
            "partial": 0,
            "rejected": 0,
            "total_slippage": 0.0,
            "by_side": {"BUY": {"total": 0, "filled": 0, "partial": 0, "slippage": 0.0}, 
                       "SELL": {"total": 0, "filled": 0, "partial": 0, "slippage": 0.0}},
        }
        
        for order_id, ticker, side, yes_price, count, status, raw_json in orders:
            stats["total"] += 1
            stats["by_side"][side]["total"] += 1
            
            # Try to parse JSON for fill information
            try:
                data = json.loads(raw_json) if raw_json else {}
            except Exception:
                data = {}
            
            # Extract fill qty (look for various possible response fields)
            filled_qty = 0
            filled_price = yes_price
            
            if isinstance(data, dict):
                # Try common response keys
                filled_qty = data.get("filled_quantity") or data.get("quantity") or 0
                filled_price = data.get("average_price") or data.get("execution_price") or yes_price
            
            # Categorize fill
            if filled_qty == 0:
                stats["rejected"] += 1
                stats["by_side"][side]["partial"] += 1
            elif filled_qty < count:
                stats["partial"] += 1
                stats["by_side"][side]["partial"] += 1
                stats["filled"] += 1
                stats["by_side"][side]["filled"] += 1
            else:
                stats["filled"] += 1
                stats["by_side"][side]["filled"] += 1
            
            # Compute slippage (in cents * qty)
            if filled_qty > 0:
                slippage_cents = (filled_price - yes_price) * filled_qty
                stats["total_slippage"] += slippage_cents
                stats["by_side"][side]["slippage"] += slippage_cents
        
        # Compute final metrics
        fill_rate = 100.0 * stats["filled"] / stats["total"] if stats["total"] > 0 else 0.0
        
        # Average fill % (sum of filled_qty / requested_qty)
        avg_fill_pct = 0.0
        filled_count = stats["filled"] + stats["partial"]
        
        if stats["total"] > 0:
            # Recount for fill %
            fill_pcts = []
            for order_id, ticker, side, yes_price, count, status, raw_json in orders:
                try:
                    data = json.loads(raw_json) if raw_json else {}
                    filled_qty = data.get("filled_quantity") or data.get("quantity") or 0
                    if count > 0 and filled_qty > 0:
                        fill_pcts.append(100.0 * filled_qty / count)
                except Exception:
                    pass
            
            avg_fill_pct = sum(fill_pcts) / len(fill_pcts) if fill_pcts else 0.0
        
        avg_slippage = stats["total_slippage"] / stats["total"] if stats["total"] > 0 else 0.0
        
        return {
            "total_orders": stats["total"],
            "filled_orders": stats["filled"],
            "partial_fills": stats["partial"],
            "rejected_orders": stats["rejected"],
            "fill_rate": round(fill_rate, 1),
            "avg_fill_pct": round(avg_fill_pct, 1),
            "total_slippage_cents": round(stats["total_slippage"], 2),
            "avg_slippage_per_order": round(avg_slippage, 2),
            "by_side": {
                "BUY": {
                    "total": stats["by_side"]["BUY"]["total"],
                    "filled": stats["by_side"]["BUY"]["filled"],
                    "partial": stats["by_side"]["BUY"]["partial"],
                    "slippage": round(stats["by_side"]["BUY"]["slippage"], 2),
                } if stats["by_side"]["BUY"]["total"] > 0 else {},
                "SELL": {
                    "total": stats["by_side"]["SELL"]["total"],
                    "filled": stats["by_side"]["SELL"]["filled"],
                    "partial": stats["by_side"]["SELL"]["partial"],
                    "slippage": round(stats["by_side"]["SELL"]["slippage"], 2),
                } if stats["by_side"]["SELL"]["total"] > 0 else {},
            }
        }
    except Exception as e:
        import logging
        logging.warning(f"Error computing fill analytics: {e}")
        return {
            "total_orders": 0,
            "filled_orders": 0,
            "partial_fills": 0,
            "rejected_orders": 0,
            "fill_rate": 0.0,
            "avg_fill_pct": 0.0,
            "total_slippage_cents": 0.0,
            "avg_slippage_per_order": 0.0,
            "by_side": {},
        }


def compute_series_correlations(conn, lookback_hours: int = 24, min_trades: int = 3):
    """Compute correlation matrix between series based on recent P&L.
    
    Uses realized P&L per trade to estimate series correlations:
    - Positive correlation: series tend to both profit or both lose together
    - Negative correlation: series offset each other
    
    Returns dict mapping (series1, series2) -> correlation_coefficient.
    Only includes pairs with correlation > 0.3 or < -0.3 (significant).
    """
    try:
        import numpy as np
        
        # Get all trades with their series in lookback period
        trades = conn.execute(
            f"""SELECT ticker, side, price, qty, ts_utc FROM trades
               WHERE datetime(ts_utc) > datetime('now', '-' || ? || ' hours')
               ORDER BY ticker, ts_utc""",
            (lookback_hours,),
        ).fetchall()
        
        if len(trades) < min_trades * 2:
            return {}
        
        # Group by series and compute P&L per trade
        series_pnls = {}
        by_ticker = {}
        
        for ticker, side, price, qty, ts_utc in trades:
            series = ticker.split("_")[0]  # Extract series prefix
            
            if series not in series_pnls:
                series_pnls[series] = []
            if ticker not in by_ticker:
                by_ticker[ticker] = {"buys": [], "sells": []}
            
            if side == "BUY":
                by_ticker[ticker]["buys"].append((price, qty, ts_utc))
            elif side == "SELL":
                by_ticker[ticker]["sells"].append((price, qty, ts_utc))
        
        # Match buy/sell pairs and accumulate per-series P&L
        for ticker, trades_dict in by_ticker.items():
            series = ticker.split("_")[0]
            buys = trades_dict["buys"]
            sells = trades_dict["sells"]
            
            buy_idx, sell_idx = 0, 0
            while buy_idx < len(buys) and sell_idx < len(sells):
                buy_price, buy_qty, _ = buys[buy_idx]
                sell_price, sell_qty, _ = sells[sell_idx]
                
                match_qty = min(buy_qty, sell_qty)
                trade_pnl = (sell_price - buy_price) * match_qty / 100.0
                series_pnls[series].append(trade_pnl)
                
                buys[buy_idx] = (buy_price, buy_qty - match_qty, _)
                sells[sell_idx] = (sell_price, sell_qty - match_qty, _)
                
                if buys[buy_idx][1] == 0:
                    buy_idx += 1
                if sells[sell_idx][1] == 0:
                    sell_idx += 1
        
        # Filter series with enough trades
        valid_series = {s: pnls for s, pnls in series_pnls.items() if len(pnls) >= min_trades}
        
        if len(valid_series) < 2:
            return {}
        
        # Compute correlation matrix
        correlations = {}
        series_list = sorted(valid_series.keys())
        
        for i, s1 in enumerate(series_list):
            for s2 in series_list[i+1:]:
                pnls1 = np.array(valid_series[s1])
                pnls2 = np.array(valid_series[s2])
                
                # Ensure same length by truncating
                min_len = min(len(pnls1), len(pnls2))
                pnls1 = pnls1[-min_len:]
                pnls2 = pnls2[-min_len:]
                
                if min_len > 0:
                    corr = np.corrcoef(pnls1, pnls2)[0, 1]
                    if not np.isnan(corr) and abs(corr) > 0.3:
                        key = tuple(sorted([s1, s2]))
                        correlations[key] = round(float(corr), 3)
        
        return correlations
    except Exception as e:
        import logging
        logging.warning(f"Error computing series correlations: {e}")
        return {}


def compute_correlated_exposure(conn, live_positions: dict, lookback_hours: int = 24):
    """Compute portfolio risk considering correlated positions.
    
    For each series with open position, sum up correlated exposures.
    Returns dict mapping series -> correlated_notional_dollars.
    
    High correlated exposure = concentration risk (series move together).
    """
    try:
        correlations = compute_series_correlations(conn, lookback_hours)
        
        if not correlations or not live_positions:
            return {}
        
        # Get current positions by series
        positions_by_series = {}
        for ticker, (qty, avg_px) in live_positions.items():
            series = ticker.split("_")[0]
            if series not in positions_by_series:
                positions_by_series[series] = 0.0
            # Notional = qty * price (in dollars, assuming price is in cents/dollar scale)
            positions_by_series[series] += qty * avg_px / 100.0
        
        # For each series, sum correlated positions
        correlated_exposure = {}
        for series, notional in positions_by_series.items():
            total_corr_exposure = notional  # Start with own exposure
            
            for (s1, s2), corr in correlations.items():
                if s1 == series and s2 in positions_by_series:
                    # Add (correlation * other_series_notional)
                    total_corr_exposure += abs(corr) * positions_by_series[s2]
                elif s2 == series and s1 in positions_by_series:
                    total_corr_exposure += abs(corr) * positions_by_series[s1]
            
            correlated_exposure[series] = round(total_corr_exposure, 2)
        
        return correlated_exposure
    except Exception as e:
        import logging
        logging.warning(f"Error computing correlated exposure: {e}")
        return {}


def compute_portfolio_delta(conn, live_positions: dict, forecasts: dict):
    """Compute portfolio delta (directional exposure) based on positions and forecasts.
    
    Delta = qty * (forecast_prob - 0.5)
    - Positive delta = portfolio benefits if outcome probability increases
    - Negative delta = portfolio benefits if outcome probability decreases
    
    Args:
        live_positions: dict of ticker -> (qty, avg_price)
        forecasts: dict of ticker -> forecast_prob (0-100)
    
    Returns dict with:
        - portfolio_delta: sum of all deltas
        - delta_by_series: breakdown by series
        - delta_by_ticker: breakdown by ticker
    """
    try:
        if not live_positions:
            return {"portfolio_delta": 0.0, "delta_by_series": {}, "delta_by_ticker": {}}
        
        delta_by_ticker = {}
        delta_by_series = {}
        total_delta = 0.0
        
        for ticker, (qty, avg_px) in live_positions.items():
            if qty == 0:
                continue
            
            # Get forecast probability for this ticker
            prob = forecasts.get(ticker, 50.0)  # Default to 50% if no forecast
            
            # Delta = qty * (prob/100 - 0.5)
            # Normalize prob to 0-1 scale
            prob_decimal = prob / 100.0
            delta = qty * (prob_decimal - 0.5)
            
            delta_by_ticker[ticker] = round(delta, 2)
            total_delta += delta
            
            # Aggregate by series
            series = ticker.split("_")[0]
            if series not in delta_by_series:
                delta_by_series[series] = 0.0
            delta_by_series[series] += delta
        
        return {
            "portfolio_delta": round(total_delta, 2),
            "delta_by_series": {k: round(v, 2) for k, v in delta_by_series.items()},
            "delta_by_ticker": delta_by_ticker,
        }
    except Exception as e:
        import logging
        logging.warning(f"Error computing portfolio delta: {e}")
        return {"portfolio_delta": 0.0, "delta_by_series": {}, "delta_by_ticker": {}}


def detect_vol_spikes(conn, lookback_hours: int = 1, vol_spike_threshold: float = 2.0):
    """Detect recent volatility spikes per series by comparing recent vs historical vol.
    
    Returns dict mapping series -> {
        'recent_vol': current volatility,
        'historical_vol': baseline volatility,
        'spike_ratio': recent_vol / historical_vol,
        'is_spike': bool (spike_ratio > threshold)
    }
    """
    try:
        # Get recent forecast volatility
        recent = conn.execute(
            f"""SELECT series, AVG(sigma) as avg_sigma FROM weather_features
               WHERE datetime(ts_utc) > datetime('now', '-' || ? || ' hours')
               GROUP BY series""",
            (lookback_hours,),
        ).fetchall()
        
        # Get historical baseline (last 7 days)
        historical = conn.execute(
            f"""SELECT series, AVG(sigma) as avg_sigma FROM weather_features
               WHERE datetime(ts_utc) > datetime('now', '-7 days')
               GROUP BY series""",
        ).fetchall()
        
        recent_dict = {s: vol for s, vol in recent}
        hist_dict = {s: vol for s, vol in historical}
        
        spikes = {}
        for series in set(list(recent_dict.keys()) + list(hist_dict.keys())):
            recent_vol = recent_dict.get(series, 0.0) or 0.0
            hist_vol = hist_dict.get(series, 0.0) or 0.0
            
            if hist_vol > 0:
                ratio = recent_vol / hist_vol
                is_spike = ratio > vol_spike_threshold
                spikes[series] = {
                    "recent_vol": round(float(recent_vol), 2),
                    "historical_vol": round(float(hist_vol), 2),
                    "spike_ratio": round(ratio, 2),
                    "is_spike": is_spike,
                }
        
        return spikes
    except Exception as e:
        import logging
        logging.warning(f"Error detecting vol spikes: {e}")
        return {}



