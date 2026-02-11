#!/usr/bin/env python3
"""
Sync local SQLite bot data into a remote Postgres database for Streamlit Cloud.

Usage:
  python -m src.sync_dashboard_db --once
  python -m src.sync_dashboard_db --interval 30

Environment:
  DATABASE_URL                Required. Postgres connection string used by dashboard.
  SOURCE_SQLITE_PATH          Optional. Defaults to data/kalshi_quotes.sqlite
  SYNC_FORECAST               Optional. 1 to sync forecast_predictions table, default 0
  SYNC_QUOTES                 Optional. 1 to sync quotes table (can be large), default 0
  SYNC_BATCH_SIZE             Optional. default 2000
"""

import argparse
import logging
import os
import sqlite3
import time
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()

LOG_LEVEL = os.getenv("SYNC_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SOURCE_SQLITE_PATH = os.getenv("SOURCE_SQLITE_PATH", "data/kalshi_quotes.sqlite")
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
SYNC_FORECAST = os.getenv("SYNC_FORECAST", "0") == "1"
SYNC_QUOTES = os.getenv("SYNC_QUOTES", "0") == "1"
SYNC_BATCH_SIZE = int(os.getenv("SYNC_BATCH_SIZE", "2000"))


INCREMENTAL_TABLES = {
    "trades": ["id", "ts_utc", "ticker", "side", "price", "qty", "note"],
    "live_orders": [
        "id",
        "ts_utc",
        "ticker",
        "side",
        "action",
        "yes_price",
        "count",
        "order_id",
        "client_order_id",
        "status",
        "raw_json",
    ],
}

if SYNC_FORECAST:
    INCREMENTAL_TABLES["forecast_predictions"] = [
        "id",
        "ts_utc",
        "ticker",
        "series",
        "event_ticker",
        "strike_type",
        "strike_floor",
        "strike_cap",
        "forecast_prob",
        "forecast_method",
        "num_ensemble_members",
        "mu",
        "sigma",
        "outcome",
        "outcome_ts_utc",
    ]

if SYNC_QUOTES:
    INCREMENTAL_TABLES["quotes"] = [
        "id",
        "ts_utc",
        "ticker",
        "best_yes_bid",
        "best_yes_ask",
        "best_no_bid",
        "best_no_ask",
        "mid_yes",
    ]


def sqlite_table_exists(conn, table_name):
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def ensure_remote_schema(engine):
    def try_execute(conn, sql_text, allow_fail=False):
        try:
            conn.execute(text(sql_text))
            return True
        except SQLAlchemyError as e:
            if allow_fail:
                logger.warning("Schema step skipped: %s", str(e).split("\n")[0])
                return False
            raise

    with engine.connect() as conn:
        conn = conn.execution_options(isolation_level="AUTOCOMMIT")
        try_execute(
            conn,
            """
                CREATE TABLE IF NOT EXISTS trades (
                    id BIGINT PRIMARY KEY,
                    ts_utc TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    price INTEGER NOT NULL,
                    qty INTEGER NOT NULL,
                    note TEXT
                )
                """,
        )
        try_execute(conn, "CREATE INDEX IF NOT EXISTS idx_trades_ts_utc ON trades(ts_utc DESC)", allow_fail=True)

        try_execute(
            conn,
            """
                CREATE TABLE IF NOT EXISTS live_orders (
                    id BIGINT PRIMARY KEY,
                    ts_utc TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    action TEXT NOT NULL,
                    yes_price INTEGER,
                    count INTEGER NOT NULL,
                    order_id TEXT,
                    client_order_id TEXT,
                    status TEXT,
                    raw_json TEXT
                )
                """,
        )
        try_execute(conn, "CREATE INDEX IF NOT EXISTS idx_live_orders_ts_utc ON live_orders(ts_utc DESC)", allow_fail=True)

        try_execute(
            conn,
            """
                CREATE TABLE IF NOT EXISTS positions (
                    ticker TEXT PRIMARY KEY,
                    qty INTEGER NOT NULL,
                    avg_price DOUBLE PRECISION NOT NULL
                )
                """,
        )

        if SYNC_FORECAST:
            try_execute(
                conn,
                """
                    CREATE TABLE IF NOT EXISTS forecast_predictions (
                        id BIGINT PRIMARY KEY,
                        ts_utc TEXT NOT NULL,
                        ticker TEXT NOT NULL,
                        series TEXT NOT NULL,
                        event_ticker TEXT NOT NULL,
                        strike_type TEXT NOT NULL,
                        strike_floor DOUBLE PRECISION,
                        strike_cap DOUBLE PRECISION,
                        forecast_prob DOUBLE PRECISION NOT NULL,
                        forecast_method TEXT NOT NULL,
                        num_ensemble_members INTEGER,
                        mu DOUBLE PRECISION,
                        sigma DOUBLE PRECISION,
                        outcome DOUBLE PRECISION,
                        outcome_ts_utc TEXT
                    )
                    """,
            )
            try_execute(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_forecast_predictions_ts_utc ON forecast_predictions(ts_utc DESC)",
                allow_fail=True,
            )

        try_execute(
            conn,
            """
                CREATE TABLE IF NOT EXISTS quotes (
                    id BIGINT PRIMARY KEY,
                    ts_utc TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    best_yes_bid INTEGER,
                    best_yes_ask INTEGER,
                    best_no_bid INTEGER,
                    best_no_ask INTEGER,
                    mid_yes DOUBLE PRECISION
                )
                """,
        )
        try_execute(conn, "CREATE INDEX IF NOT EXISTS idx_quotes_ts_utc ON quotes(ts_utc DESC)", allow_fail=True)


def remote_max_id(conn, table_name):
    return int(conn.execute(text(f"SELECT COALESCE(MAX(id), 0) FROM {table_name}")).scalar() or 0)


def sync_incremental_table(src_conn, engine, table_name, columns, batch_size=2000):
    if not sqlite_table_exists(src_conn, table_name):
        return 0

    inserted_total = 0
    col_csv = ", ".join(columns)
    placeholders = ", ".join([f":{c}" for c in columns])
    sql_insert = text(
        f"""
        INSERT INTO {table_name} ({col_csv})
        VALUES ({placeholders})
        ON CONFLICT (id) DO NOTHING
        """
    )

    with engine.begin() as dst_conn:
        max_id = remote_max_id(dst_conn, table_name)
        while True:
            rows = src_conn.execute(
                f"SELECT {col_csv} FROM {table_name} WHERE id > ? ORDER BY id ASC LIMIT ?",
                (max_id, int(batch_size)),
            ).fetchall()
            if not rows:
                break

            records = []
            for row in rows:
                rec = {}
                for idx, col in enumerate(columns):
                    rec[col] = row[idx]
                records.append(rec)

            dst_conn.execute(sql_insert, records)
            inserted_total += len(records)
            max_id = int(rows[-1][0])

    return inserted_total


def sync_positions(src_conn, engine):
    if not sqlite_table_exists(src_conn, "positions"):
        return 0

    rows = src_conn.execute("SELECT ticker, qty, avg_price FROM positions").fetchall()
    if not rows:
        return 0

    records = [{"ticker": r[0], "qty": int(r[1]), "avg_price": float(r[2])} for r in rows]
    sql = text(
        """
        INSERT INTO positions (ticker, qty, avg_price)
        VALUES (:ticker, :qty, :avg_price)
        ON CONFLICT (ticker) DO UPDATE SET
            qty = EXCLUDED.qty,
            avg_price = EXCLUDED.avg_price
        """
    )
    with engine.begin() as conn:
        conn.execute(sql, records)
    return len(records)


def sync_once(src_conn, engine):
    totals = {}
    for table_name, cols in INCREMENTAL_TABLES.items():
        inserted = sync_incremental_table(src_conn, engine, table_name, cols, batch_size=SYNC_BATCH_SIZE)
        totals[table_name] = inserted

    positions_upserted = sync_positions(src_conn, engine)
    totals["positions_upserted"] = positions_upserted
    return totals


def parse_args():
    parser = argparse.ArgumentParser(description="Sync local SQLite data to remote Postgres.")
    parser.add_argument("--once", action="store_true", help="Run a single sync and exit.")
    parser.add_argument("--interval", type=int, default=30, help="Loop interval in seconds (default 30).")
    return parser.parse_args()


def main():
    args = parse_args()

    if not DATABASE_URL:
        raise SystemExit("DATABASE_URL is required.")

    sqlite_path = Path(SOURCE_SQLITE_PATH)
    if not sqlite_path.exists():
        raise SystemExit(f"SQLite source file not found: {sqlite_path}")

    src_conn = sqlite3.connect(str(sqlite_path), check_same_thread=False, timeout=10)
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    ensure_remote_schema(engine)
    logger.info(
        "Sync started. source=%s sync_forecast=%s sync_quotes=%s batch=%s",
        sqlite_path,
        int(SYNC_FORECAST),
        int(SYNC_QUOTES),
        SYNC_BATCH_SIZE,
    )

    if args.once:
        totals = sync_once(src_conn, engine)
        logger.info("Sync done: %s", totals)
        src_conn.close()
        return

    interval = max(5, int(args.interval))
    while True:
        try:
            totals = sync_once(src_conn, engine)
            logger.info("Sync done: %s", totals)
        except Exception as e:
            logger.error("Sync error: %r", e)
        time.sleep(interval)


if __name__ == "__main__":
    main()
