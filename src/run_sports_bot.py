import os
import re
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from src.kalshi_client import KalshiClient, best_bid_ask_from_orderbook
from src.db import open_db, snapshot_markets
from src.sports_sources import build_totals_by_matchup

load_dotenv()
TZ = ZoneInfo("America/New_York")

TOTAL_OVER_RE = re.compile(r"\bover\s+(\d+(?:\.\d+)?)\s+points\s+scored\b", re.IGNORECASE)
TOTAL_UNDER_RE = re.compile(r"\bunder\s+(\d+(?:\.\d+)?)\s+points\s+scored\b", re.IGNORECASE)
MARGIN_OVER_RE = re.compile(r"\bwins\s+by\s+over\s+(\d+(?:\.\d+)?)\s+points\b", re.IGNORECASE)
MARGIN_UNDER_RE = re.compile(r"\bwins\s+by\s+under\s+(\d+(?:\.\d+)?)\s+points\b", re.IGNORECASE)


def _envb(k: str, d: str = "0") -> bool:
    return os.getenv(k, d) == "1"


def _envi(k: str, d: str) -> int:
    return int(os.getenv(k, d))


def _envf(k: str, d: str) -> float:
    return float(os.getenv(k, d))


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_quotes(c: KalshiClient, ticker: str, m: dict):
    yes_bid = m.get("yes_bid")
    yes_ask = m.get("yes_ask")
    no_bid = m.get("no_bid")
    no_ask = m.get("no_ask")

    if yes_bid is not None or yes_ask is not None or no_bid is not None or no_ask is not None:
        return {
            "best_yes_bid": yes_bid,
            "best_yes_ask": yes_ask,
            "best_no_bid": no_bid,
            "best_no_ask": no_ask,
        }

    ob = c.get_orderbook(ticker)
    return best_bid_ask_from_orderbook(ob)


def extract_total_and_margin(title: str):
    t = (title or "").strip()
    if not t:
        return None

    tl = t.lower()
    if not (tl.startswith("yes ") or tl.startswith("no ")):
        return None

    yn = "yes" if tl.startswith("yes ") else "no"
    is_bundle = ("," in t)

    overs = [float(x) for x in TOTAL_OVER_RE.findall(t)]
    unders = [float(x) for x in TOTAL_UNDER_RE.findall(t)]
    mov = [float(x) for x in MARGIN_OVER_RE.findall(t)]
    mun = [float(x) for x in MARGIN_UNDER_RE.findall(t)]

    total_legs = [("over", x) for x in overs] + [("under", x) for x in unders]
    margin_legs = [("over", x) for x in mov] + [("under", x) for x in mun]

    return {
        "yn": yn,
        "bundle": is_bundle,
        "title": t,
        "total_legs": total_legs,
        "margin_legs": margin_legs,
    }


def pick_reference_total(line: float, totals: dict, tol: float):
    best = None
    best_d = None
    for meta in totals.values():
        x = meta.get("total")
        if x is None:
            continue
        d = abs(float(x) - float(line))
        if d <= tol and (best_d is None or d < best_d):
            best = meta
            best_d = d
    return best


def fetch_all_markets(c: KalshiClient):
    limit = _envi("MARKETS_LIMIT", "200")
    max_pages = _envi("SPORTS_MAX_PAGES", "10")

    first = c.get_markets(limit=limit)
    markets = list(first.get("markets", []))
    cursor = first.get("cursor")

    pages = 1
    while cursor and pages < max_pages:
        nxt = c.get_markets(limit=limit, cursor=cursor)
        markets.extend(nxt.get("markets", []))
        cursor = nxt.get("cursor")
        pages += 1

    return markets


def main():
    if not _envb("SPORTS_ENABLED", "0"):
        print("SPORTS_ENABLED=0, exiting")
        return

    live = _envb("LIVE_TRADING", "0")
    kill = _envb("KILL_SWITCH", "0")

    allow_bundles = _envb("SPORTS_ALLOW_BUNDLES", "0")
    allow_mixed = _envb("SPORTS_ALLOW_MIXED_BUNDLES", "0")

    enable_totals = _envb("SPORTS_ENABLE_TOTALS", "1")

    run_every = _envi("RUN_EVERY_SECONDS", "30")
    min_edge = _envi("SPORTS_MIN_EDGE_CENTS", "8")
    max_qty = _envi("SPORTS_MAX_CONTRACTS_PER_TRADE", "1")
    max_live_per_loop = _envi("MAX_LIVE_PER_LOOP", "2")

    tol = _envf("SPORTS_TOTAL_LINE_TOL", "0.5")
    max_spread = _envi("SPORTS_MAX_SPREAD", "25")
    min_liq = _envi("SPORTS_MIN_LIQUIDITY_DOLLARS", "0")

    max_bundle_yes_ask = _envi("SPORTS_MAX_BUNDLE_YES_ASK", "20")
    max_mixed_bundle_yes_ask = _envi("SPORTS_MAX_MIXED_BUNDLE_YES_ASK", "10")
    min_yes_price = _envi("SPORTS_MIN_YES_PRICE", "1")

    db_path = os.getenv("DB_PATH", "kalshi.db")
    conn = open_db(db_path)
    c = KalshiClient()

    print(f"LIVE={int(live)} KILL={int(kill)} DB={db_path}")

    while True:
        try:
            totals = build_totals_by_matchup()
            markets = fetch_all_markets(c)

            snapshot_markets(conn, now_utc_iso(), "SPORTS_SCAN", markets)

            placed = 0
            candidates = 0

            for m in markets:
                if placed >= max_live_per_loop:
                    break

                liq = m.get("liquidity_dollars")
                if liq is not None and float(liq) < float(min_liq):
                    continue

                title = (m.get("title") or "").strip()
                parsed = extract_total_and_margin(title)
                if parsed is None:
                    continue

                if parsed["yn"] != "yes":
                    continue

                if parsed["bundle"] and not allow_bundles:
                    continue

                total_legs = parsed["total_legs"]
                margin_legs = parsed["margin_legs"]

                if not enable_totals:
                    continue

                if len(total_legs) != 1:
                    continue

                mixed = (len(margin_legs) == 1)

                if mixed and not allow_mixed:
                    continue

                if len(margin_legs) > 1:
                    continue

                ticker = m.get("ticker") or ""
                if not ticker:
                    continue

                q = get_quotes(c, ticker, m)
                yes_ask = q.get("best_yes_ask")
                yes_bid = q.get("best_yes_bid")

                if yes_ask is None:
                    continue

                try:
                    px = int(yes_ask)
                except Exception:
                    continue

                if px < min_yes_price:
                    continue

                if yes_bid is not None:
                    try:
                        spr = int(px) - int(yes_bid)
                    except Exception:
                        spr = 0
                    if spr < 0:
                        spr = 0
                    if spr > max_spread:
                        continue

                if parsed["bundle"]:
                    if mixed:
                        if px > max_mixed_bundle_yes_ask:
                            continue
                    else:
                        if px > max_bundle_yes_ask:
                            continue

                d, line = total_legs[0]
                ref = pick_reference_total(line, totals, tol)
                if ref is None:
                    continue

                fair = 50
                if px > (fair - min_edge):
                    continue

                candidates += 1

                if not live:
                    print(
                        f"SIM BUY YES {ticker} px={px} fair={fair} edge={fair-px} "
                        f"mixed={int(mixed)} '{parsed['title']}'"
                    )
                    placed += 1
                    continue

                if kill:
                    continue

                cid = f"sports-{int(time.time()*1000)}"
                c.create_order(
                    ticker=ticker,
                    side="yes",
                    action="buy",
                    count=max_qty,
                    type_="limit",
                    yes_price=px,
                    no_price=None,
                    client_order_id=cid,
                    reduce_only=None,
                    time_in_force=None,
                )
                placed += 1

            print(
                f"{datetime.now(TZ).isoformat()} totals={len(totals)} markets={len(markets)} "
                f"candidates={candidates} placed={placed}"
            )

        except Exception as e:
            print(f"ERROR: {e}")

        time.sleep(run_every)


if __name__ == "__main__":
    main()
