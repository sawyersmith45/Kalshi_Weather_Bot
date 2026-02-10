from datetime import datetime, timezone
from kalshi_client import KalshiClient, best_bid_ask_from_orderbook
from db import open_db, insert_quote

SERIES = "KXHIGHNY"

def parse_dt(s: str | None):
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None

def main():
    client = KalshiClient()
    conn = open_db("data/kalshi_quotes.sqlite")
    ts = datetime.now(timezone.utc).isoformat()

    cursor = None
    events = []
    for _ in range(10):
        resp = client.get_events(limit=200, cursor=cursor, series_ticker=SERIES)
        events.extend(resp.get("events", []))
        cursor = resp.get("cursor")
        if not cursor:
            break

    if not events:
        print("No events found for series:", SERIES)
        return

    def event_sort_key(e):
        for k in ("close_time", "event_close_time", "settlement_time", "strike_time"):
            dt = parse_dt(e.get(k))
            if dt:
                return dt
        return datetime(1970, 1, 1, tzinfo=timezone.utc)

    events.sort(key=event_sort_key, reverse=True)
    chosen = events[0]
    event_ticker = chosen.get("event_ticker") or chosen.get("ticker")
    print("Chosen event:", event_ticker, "|", chosen.get("title"))

    ev = client.get_event(event_ticker)
    markets = ev.get("markets", [])

    if not markets:
        print("No markets on that event. Raw keys:", ev.keys())
        return

    tickers = [m.get("ticker") for m in markets if m.get("ticker")]
    print(f"Found {len(tickers)} markets in event. Logging quotes...")

    for ticker in tickers:
        ob = client.get_orderbook(ticker)
        q = best_bid_ask_from_orderbook(ob)
        insert_quote(conn, ts, ticker, q)
        print(ticker, q)

    conn.close()
    print("Done. Saved to data/kalshi_quotes.sqlite")

if __name__ == "__main__":
    main()
