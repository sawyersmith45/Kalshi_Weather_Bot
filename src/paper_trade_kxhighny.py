import math
from datetime import datetime, timezone

from kalshi_client import KalshiClient
from nws import get_day_high_f
from db import open_db, record_trade, get_position, upsert_position

SERIES = "KXHIGHNY"
EVENT = "KXHIGHNY-26JAN15"

CENTRAL_PARK_LAT = 40.7829
CENTRAL_PARK_LON = -73.9654

MIN_EDGE_CENTS = 3.0
MAX_SPREAD_CENTS = 10
MIN_VOL_24H = 50
MIN_OPEN_INTEREST = 200

TRADE_QTY = 25
MAX_ABS_QTY_PER_TICKER = 100


def normal_cdf(x, mu, sigma):
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1.0 + math.erf(z))


def clamp01(p):
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return p


def prob_between(lo, hi, mu, sigma):
    if sigma <= 0:
        return 1.0 if (mu >= lo and mu <= hi) else 0.0
    return clamp01(normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma))


def fair_prob_from_market(m: dict, mu: float, sigma: float):
    st = m.get("strike_type")
    floor_ = m.get("floor_strike")
    cap_ = m.get("cap_strike")

    if st == "between" and floor_ is not None and cap_ is not None:
        return prob_between(float(floor_), float(cap_), mu, sigma)

    return None


def apply_fill_to_position(qty: int, avg: float, side: str, fill_price: int, fill_qty: int):
    if fill_qty <= 0:
        return qty, avg

    if side == "BUY":
        if qty >= 0:
            new_qty = qty + fill_qty
            new_avg = (qty * avg + fill_qty * fill_price) / new_qty
            return new_qty, new_avg

        cover = min(-qty, fill_qty)
        remaining_buy = fill_qty - cover
        new_qty = qty + cover
        if new_qty == 0:
            avg = 0.0
        if remaining_buy > 0:
            new_qty = remaining_buy
            avg = float(fill_price)
        return new_qty, avg

    if side == "SELL":
        if qty <= 0:
            new_qty = qty - fill_qty
            abs_old = -qty
            abs_new = -new_qty
            new_avg = (abs_old * avg + fill_qty * fill_price) / abs_new if abs_new != 0 else 0.0
            return new_qty, new_avg

        sell = min(qty, fill_qty)
        remaining_sell = fill_qty - sell
        new_qty = qty - sell
        if new_qty == 0:
            avg = 0.0
        if remaining_sell > 0:
            new_qty = -remaining_sell
            avg = float(fill_price)
        return new_qty, avg

    raise ValueError("side must be BUY or SELL")


def mark_from_market_obj(market_obj: dict):
    bid = market_obj.get("yes_bid")
    ask = market_obj.get("yes_ask")
    if bid is None and ask is None:
        return None
    if bid is None:
        return float(ask)
    if ask is None:
        return float(bid)
    return (float(bid) + float(ask)) / 2.0


def main():
    client = KalshiClient()
    conn = open_db("data/kalshi_quotes.sqlite")

    target_date = "2026-01-15"
    forecast_high = get_day_high_f(CENTRAL_PARK_LAT, CENTRAL_PARK_LON, target_date)
    if forecast_high is None:
        print("Could not find NWS daytime forecast for", target_date)
        return

    mu = float(forecast_high)
    sigma = 3.5

    resp = client.get_markets(series_ticker=SERIES, status="open", limit=200)
    markets = [m for m in resp.get("markets", []) if m.get("event_ticker") == EVENT]
    if not markets:
        print("No markets matched:", EVENT)
        return

    ts = datetime.now(timezone.utc).isoformat()

    candidates = []
    for m in markets:
        ticker = m.get("ticker")
        title = m.get("title", "")
        bid = m.get("yes_bid")
        ask = m.get("yes_ask")
        vol24 = m.get("volume_24h") or 0
        oi = m.get("open_interest") or 0

        if not ticker or bid is None or ask is None:
            continue

        spread = ask - bid
        if spread > MAX_SPREAD_CENTS:
            continue
        if vol24 < MIN_VOL_24H or oi < MIN_OPEN_INTEREST:
            continue

        p = fair_prob_from_market(m, mu, sigma)
        if p is None:
            continue

        fair = p * 100.0
        buy_edge = fair - float(ask)
        sell_edge = float(bid) - fair

        if buy_edge >= MIN_EDGE_CENTS:
            candidates.append((buy_edge, "BUY", ticker, title, int(bid), int(ask), fair))
        if sell_edge >= MIN_EDGE_CENTS:
            candidates.append((sell_edge, "SELL", ticker, title, int(bid), int(ask), fair))

    candidates.sort(reverse=True, key=lambda x: x[0])

    print("Paper trades to place:")
    print("edge side  fair  bid ask qty ticker  title")
    print("-" * 140)

    placed_any = False
    for edge, side, ticker, title, bid, ask, fair in candidates:
        cur_qty, cur_avg = get_position(conn, ticker)

        if side == "BUY":
            fill_price = ask
            new_abs = abs(cur_qty + TRADE_QTY)
            if new_abs > MAX_ABS_QTY_PER_TICKER:
                continue
            qty = TRADE_QTY
        else:
            fill_price = bid
            new_abs = abs(cur_qty - TRADE_QTY)
            if new_abs > MAX_ABS_QTY_PER_TICKER:
                continue
            qty = TRADE_QTY

        new_qty, new_avg = apply_fill_to_position(cur_qty, cur_avg, side, fill_price, qty)

        record_trade(conn, ts, ticker, side, fill_price, qty, note=f"edge={edge:.1f}, fair={fair:.1f}")
        upsert_position(conn, ticker, new_qty, new_avg)

        placed_any = True
        print(f"{edge:4.1f} {side:4} {fair:5.1f} {bid:3d} {ask:3d} {qty:3d} {ticker}  {title}")

    if not placed_any:
        print("No trades placed (filters met, but position caps prevented fills or no candidates).")

    print()
    print("Positions (marked):")
    print("qty  avg  mark  unreal_PnL$  ticker")
    print("-" * 80)

    pos_rows = conn.execute("SELECT ticker, qty, avg_price FROM positions WHERE qty != 0").fetchall()
    if not pos_rows:
        print("No open positions.")
        return

    for ticker, qty, avg in pos_rows:
        m = client.get_market(ticker).get("market")
        mark = mark_from_market_obj(m)
        if mark is None:
            continue

        if qty > 0:
            unreal = qty * (mark - float(avg)) / 100.0
        else:
            unreal = (-qty) * (float(avg) - mark) / 100.0

        print(f"{qty:4d} {float(avg):5.1f} {mark:5.1f} {unreal:10.2f}  {ticker}")


if __name__ == "__main__":
    main()
