import os
import time
import math
import re
import logging
import traceback
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from src.kalshi_client import KalshiClient, best_bid_ask_from_orderbook
from src.db import (
    open_db,
    get_state,
    set_state,
    log_live_order,
    upsert_position,
    get_position,
    get_last_trade_ts,
    set_last_trade_ts,
)

from src.weather_sources import (
    prob_between_empirical,
)

from src.weather_sources import OpenMeteoMonthlySnowEnsembleProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

LIVE_TRADING = os.getenv("LIVE_TRADING", "0") == "1"
KILL_SWITCH = os.getenv("KILL_SWITCH", "0") == "1"

RUN_EVERY_SECONDS = int(os.getenv("RUN_EVERY_SECONDS", "120"))
FAST_EVERY_SECONDS = int(os.getenv("FAST_EVERY_SECONDS", "20"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "900"))

MAX_LIVE_ORDERS_PER_LOOP = int(os.getenv("SNOW_MAX_LIVE_ORDERS_PER_LOOP", os.getenv("MAX_LIVE_ORDERS_PER_LOOP", "6")))
MAX_LIVE_ORDERS_PER_DAY  = int(os.getenv("SNOW_MAX_LIVE_ORDERS_PER_DAY",  os.getenv("MAX_LIVE_ORDERS_PER_DAY",  "300")))
LIVE_QTY_CAP             = int(os.getenv("SNOW_LIVE_QTY_CAP",             os.getenv("LIVE_QTY_CAP",             "1")))

CANCEL_STALE_ORDERS  = os.getenv("SNOW_CANCEL_STALE_ORDERS", os.getenv("CANCEL_STALE_ORDERS", "1")) == "1"
STALE_ORDER_SECONDS  = int(os.getenv("SNOW_STALE_ORDER_SECONDS", os.getenv("STALE_ORDER_SECONDS", "60")))

STALE_QUOTE_CENTS        = int(os.getenv("SNOW_STALE_QUOTE_CENTS", os.getenv("STALE_QUOTE_CENTS", "5")))
MIN_LIQUIDITY_SZ         = int(os.getenv("SNOW_MIN_LIQUIDITY_SZ", os.getenv("MIN_LIQUIDITY_SZ", "1")))
ORDERBOOK_VALIDATE_TOPN  = int(os.getenv("SNOW_ORDERBOOK_VALIDATE_TOPN", os.getenv("ORDERBOOK_VALIDATE_TOPN", "12")))

EMPIRICAL_MIN_MEMBERS = int(os.getenv("EMPIRICAL_MIN_MEMBERS", "8"))
SNOW_SMOOTH_SIGMA_IN = float(os.getenv("SNOW_SMOOTH_SIGMA_IN", "0.4"))

# --- snow-only tuning (falls back to generic if SNOW_* not set) ---
MIN_EDGE_CENTS     = float(os.getenv("SNOW_MIN_EDGE_CENTS", os.getenv("MIN_EDGE_CENTS", "1.0")))
COST_BUFFER_CENTS  = float(os.getenv("SNOW_COST_BUFFER_CENTS", os.getenv("COST_BUFFER_CENTS", "0.0")))
MAX_SPREAD_CENTS   = int(os.getenv("SNOW_MAX_SPREAD_CENTS", os.getenv("MAX_SPREAD_CENTS", "30")))

PERSIST_SECONDS    = int(os.getenv("SNOW_PERSIST_SECONDS", os.getenv("PERSIST_SECONDS", str(RUN_EVERY_SECONDS))))

MAKER_MODE         = os.getenv("SNOW_MAKER_MODE", os.getenv("MAKER_MODE", "1")) == "1"
MAKER_IMPROVE_CENTS= int(os.getenv("SNOW_MAKER_IMPROVE_CENTS", os.getenv("MAKER_IMPROVE_CENTS", "1")))


SERIES = "KXNYCSNOWM"
LAT = 40.7829
LON = -73.9654
TZNAME = "America/New_York"

MONTH_MAP = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
             "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}

def iso_now():
    return datetime.now(timezone.utc).isoformat()

def parse_iso(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def clamp01(p):
    return max(0.0, min(1.0, float(p)))

def make_client_order_id(ticker: str) -> str:
    raw = f"snow-{ticker}-{time.time_ns()}"
    raw = re.sub(r"[^A-Za-z0-9_-]", "_", raw)
    return raw[:80]

def get_best_yes_bid_ask(ob: dict):
    b = best_bid_ask_from_orderbook(ob) if isinstance(ob, dict) else {}
    yb = b.get("best_yes_bid")
    ya = b.get("best_yes_ask")
    yb_sz = int(b.get("best_yes_bid_sz") or 0)
    ya_sz = int(b.get("best_yes_ask_sz") or 0)

    nb = b.get("best_no_bid")
    nb_sz = int(b.get("best_no_bid_sz") or 0)

    if ya is None and nb is not None:
        try:
            ya = int(100 - int(nb))
            ya_sz = nb_sz
        except Exception:
            pass

    return yb, yb_sz, ya, ya_sz

def month_from_event_ticker(event_ticker: str):
    parts = (event_ticker or "").split("-")
    if len(parts) < 2:
        return None
    code = parts[1].upper()
    if len(code) != 5:
        return None
    yy = int(code[0:2])
    mon = code[2:5]
    if mon not in MONTH_MAP:
        return None
    year = 2000 + yy
    month = MONTH_MAP[mon]
    return year, month

def pick_this_month_event(client: KalshiClient, series: str, now_utc: datetime):
    cursor = None
    best_et = None
    for _ in range(10):
        resp = client.get_events(limit=200, cursor=cursor, series_ticker=series)
        events = resp.get("events", []) or []
        for e in events:
            et = e.get("event_ticker") or e.get("ticker")
            if not et:
                continue
            ym = month_from_event_ticker(et)
            if not ym:
                continue
            y, m = ym
            local = now_utc.astimezone(ZoneInfo(TZNAME))
            if y == local.year and m == local.month:
                best_et = et
                break
        if best_et:
            break
        cursor = resp.get("cursor")
        if not cursor:
            break
    return best_et

def cancel_stale_orders(client: KalshiClient, conn, now_dt: datetime, ts: str):
    if not (LIVE_TRADING and CANCEL_STALE_ORDERS and STALE_ORDER_SECONDS > 0):
        return 0

    canceled = 0
    cursor = None
    for _ in range(10):
        resp = client.get_orders(limit=200, cursor=cursor)
        orders = resp.get("orders", []) or []
        if not orders:
            break

        for o in orders:
            st = (o.get("status") or "").lower()
            if st not in ("open", "resting"):
                continue

            oid = o.get("order_id") or o.get("id")
            if not oid:
                continue

            created = parse_iso(o.get("created_time"))
            if not created:
                continue

            age = (now_dt - created).total_seconds()
            if age < STALE_ORDER_SECONDS:
                continue

            try:
                client.cancel_order(oid)
                canceled += 1
                log_live_order(
                    conn,
                    ts,
                    ticker=o.get("ticker") or "",
                    side=o.get("side") or "",
                    action="cancel",
                    yes_price=o.get("yes_price"),
                    count=int(o.get("initial_count") or 0),
                    order_id=oid,
                    client_order_id=o.get("client_order_id"),
                    status="canceled",
                    raw_json=str(o),
                )
            except Exception:
                continue

        cursor = resp.get("cursor")
        if not cursor:
            break

    return canceled

def edge_persistence_ok(conn, key: str, edge: float, ts: str, seconds_needed: int):
    raw = get_state(conn, key, "")
    now_dt = parse_iso(ts) or datetime.now(timezone.utc)

    if raw:
        try:
            first_ts, _ = raw.split("|", 1)
            first_dt = parse_iso(first_ts)
            if first_dt and (now_dt - first_dt).total_seconds() >= seconds_needed:
                set_state(conn, key, f"{first_ts}|{edge:.4f}")
                return True
        except Exception:
            pass

    set_state(conn, key, f"{ts}|{edge:.4f}")
    return False

def clear_persistence(conn, key: str):
    set_state(conn, key, "")

def fair_prob_market_snow(m: dict, members_in: list[float]):
    st = (m.get("strike_type") or "").lower()
    floor_ = m.get("floor_strike")
    cap_ = m.get("cap_strike")

    if not members_in or len(members_in) < EMPIRICAL_MIN_MEMBERS:
        return None

    if st == "between" and floor_ is not None and cap_ is not None:
        lo = float(floor_)
        hi = float(cap_)
        p = prob_between_empirical(lo, hi, members_in, SNOW_SMOOTH_SIGMA_IN)
        return None if p is None else clamp01(p)

    if st in ("greater", "greater_than", "greater_or_equal", "above") and floor_ is not None:
        x = float(floor_)
        p = clamp01(sum(1.0 for v in members_in if float(v) >= x) / float(len(members_in)))
        return p

    if st in ("less", "less_than", "less_or_equal", "below") and cap_ is not None:
        x = float(cap_)
        p = clamp01(sum(1.0 for v in members_in if float(v) <= x) / float(len(members_in)))
        return p

    return None

def main():
    client = KalshiClient()
    conn = open_db("data/kalshi_quotes.sqlite")
    prov = OpenMeteoMonthlySnowEnsembleProvider()

    mode = "LIVE" if LIVE_TRADING else "PAPER"
    logger.info(f"{mode} snow bot running. Series={SERIES} BaseURL={client.base_url} MAKER_MODE={int(MAKER_MODE)}")

    while True:
        ts = iso_now()
        now_dt = datetime.now(timezone.utc)

        try:
            if KILL_SWITCH:
                time.sleep(RUN_EVERY_SECONDS)
                continue

            canceled = cancel_stale_orders(client, conn, now_dt, ts)
            if canceled:
                logger.info(f"Canceled {canceled} stale orders")

            event_ticker = pick_this_month_event(client, SERIES, now_dt)
            if not event_ticker:
                logger.info("No this-month event found; sleeping")
                time.sleep(RUN_EVERY_SECONDS)
                continue

            ym = month_from_event_ticker(event_ticker)
            if not ym:
                time.sleep(RUN_EVERY_SECONDS)
                continue
            year, month = ym

            resp = client.get_markets(series_ticker=SERIES, limit=200)
            ms = resp.get("markets", []) or []
            markets = [m for m in ms if m.get("event_ticker") == event_ticker and (m.get("status") in ("active", "open"))]
            if not markets:
                time.sleep(RUN_EVERY_SECONDS)
                continue

            members = prov.get_month_members_inches(LAT, LON, year, month, TZNAME)
            if not members or len(members) < EMPIRICAL_MIN_MEMBERS:
                logger.info(f"{SERIES} no members (n={0 if not members else len(members)}); sleeping")
                time.sleep(RUN_EVERY_SECONDS)
                continue

            checked = 0
            candidates = []

            for m in markets:
                ticker = m.get("ticker")
                bid = m.get("yes_bid")
                ask = m.get("yes_ask")
                if not ticker or bid is None or ask is None:
                    continue
                if m.get("yes_bid") is None:
                    continue

                bid = int(bid)
                ask = int(ask)
                spread = ask - bid
                if spread < 0 or spread > MAX_SPREAD_CENTS:
                    continue

                p = fair_prob_market_snow(m, members)
                if p is None:
                    continue

                fair = p * 100.0
                buy_edge = fair - float(ask)
                eff_buy = buy_edge - (spread / 2.0) - COST_BUFFER_CENTS

                checked += 1

                if eff_buy >= MIN_EDGE_CENTS:
                    cur_qty, cur_avg = get_position(conn, ticker)
                    candidates.append((float(eff_buy), ticker, bid, ask, fair, spread, cur_qty, cur_avg))

            candidates.sort(reverse=True, key=lambda x: x[0])
            logger.info(f"SUMMARY {SERIES} event={event_ticker} ym={year:04d}-{month:02d} cand={len(candidates)} checked={checked}")

            placed_live = 0
            validated = 0

            for edge_eff, ticker, snap_bid, snap_ask, fair, spread, cur_qty, cur_avg in candidates:
                if placed_live >= MAX_LIVE_ORDERS_PER_LOOP:
                    break

                last_ts = get_last_trade_ts(conn, ticker)
                if last_ts:
                    dt = (parse_iso(ts) or now_dt) - (parse_iso(last_ts) or now_dt)
                    if dt.total_seconds() < COOLDOWN_SECONDS:
                        continue

                ob_bid = None
                ob_ask = None
                ob_bid_sz = 0
                ob_ask_sz = 0

                if validated < ORDERBOOK_VALIDATE_TOPN or MAKER_MODE:
                    ob = client.get_orderbook(ticker)
                    yb, yb_sz, ya, ya_sz = get_best_yes_bid_ask(ob)
                    ob_bid, ob_ask = yb, ya
                    ob_bid_sz, ob_ask_sz = int(yb_sz), int(ya_sz)
                    validated += 1

                if ob_bid is not None and ob_ask is not None:
                    spr = int(ob_ask) - int(ob_bid)
                    if spr < 0 or spr > MAX_SPREAD_CENTS:
                        continue

                if ob_ask is not None and int(ob_ask_sz) < MIN_LIQUIDITY_SZ:
                    continue

                if ob_ask is not None and abs(int(ob_ask) - int(snap_ask)) > STALE_QUOTE_CENTS:
                    continue

                persist_key = f"persist_snow:{SERIES}:{ticker}:BUY"
                if not edge_persistence_ok(conn, persist_key, float(edge_eff), ts, PERSIST_SECONDS):
                    continue

                px = None
                post_only = None

                if MAKER_MODE:
                    if ob_bid is not None:
                        px = int(ob_bid) + MAKER_IMPROVE_CENTS
                    else:
                        px = int(math.floor(float(fair) - float(COST_BUFFER_CENTS)))

                    cap = int(math.floor(float(fair) - float(COST_BUFFER_CENTS)))
                    if cap < 1:
                        clear_persistence(conn, persist_key)
                        continue
                    if px > cap:
                        px = cap

                    if ob_ask is not None and px >= int(ob_ask):
                        if int(ob_ask) <= 1:
                            clear_persistence(conn, persist_key)
                            continue
                        px = int(ob_ask) - 1

                    post_only = True
                else:
                    if ob_ask is None:
                        continue
                    px = int(ob_ask)

                if px is None or px < 1 or px > 99:
                    clear_persistence(conn, persist_key)
                    continue

                if not LIVE_TRADING:
                    logger.info(f"PAPER BUY YES {ticker} px={px} edge={edge_eff:.1f}")
                    set_last_trade_ts(conn, ticker, ts)
                    clear_persistence(conn, persist_key)
                    continue

                client_order_id = make_client_order_id(ticker)

                resp = client.create_order(
                    ticker=ticker,
                    side="yes",
                    action="buy",
                    count=LIVE_QTY_CAP,
                    type_="limit",
                    yes_price=px,
                    client_order_id=client_order_id,
                    post_only=post_only,
                    reduce_only=None,
                    time_in_force=None,
                )

                order_obj = resp.get("order") if isinstance(resp, dict) else None
                status = order_obj.get("status") if isinstance(order_obj, dict) else None
                fill_count = int(order_obj.get("fill_count") or 0) if isinstance(order_obj, dict) else 0

                log_live_order(
                    conn,
                    ts,
                    ticker=ticker,
                    side="yes",
                    action="buy",
                    yes_price=px,
                    count=LIVE_QTY_CAP,
                    order_id=order_obj.get("order_id") if isinstance(order_obj, dict) else None,
                    client_order_id=client_order_id,
                    status=status,
                    raw_json=str(resp),
                )

                if fill_count <= 0:
                    logger.info(f"LIVE BUY YES {ticker} px={px} qty=1 -> NO_FILL status={status}")
                    clear_persistence(conn, persist_key)
                    continue

                new_qty = int(cur_qty) + 1
                new_avg = (float(cur_qty) * float(cur_avg) + float(px)) / float(new_qty) if new_qty else 0.0
                upsert_position(conn, ticker, new_qty, new_avg)
                set_last_trade_ts(conn, ticker, ts)
                clear_persistence(conn, persist_key)
                placed_live += 1
                logger.info(f"LIVE BUY YES {ticker} px={px} qty=1 FILLED status={status}")
                break

            time.sleep(FAST_EVERY_SECONDS if placed_live > 0 else RUN_EVERY_SECONDS)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"ERROR: {repr(e)}")
            logger.error(traceback.format_exc())
            time.sleep(RUN_EVERY_SECONDS)

if __name__ == "__main__":
    main()
