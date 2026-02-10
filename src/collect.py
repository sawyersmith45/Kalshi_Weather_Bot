from datetime import datetime, timezone
from kalshi_client import KalshiClient, best_bid_ask_from_orderbook
from db import open_db, insert_quote

WEATHER_KEYWORDS = [
    "temperature", "temp", "rain", "snow", "weather", "wind", "storm", "hurricane",
    "high", "low", "precip", "inch", "mph"
]

def looks_weather(title: str) -> bool:
    t = (title or "").lower()
    return any(k in t for k in WEATHER_KEYWORDS)

def main():
    client = KalshiClient()
    conn = open_db("data/kalshi_quotes.sqlite")
    ts = datetime.now(timezone.utc).isoformat()

    cursor = None
    tickers = []

    for _ in range(6):
        resp = client.get_markets(limit=200, cursor=cursor)
        markets = resp.get("markets", [])
        for m in markets:
            title = m.get("title", "")
            ticker = m.get("ticker")
            if ticker and looks_weather(title):
                tickers.append(ticker)

        cursor = resp.get("cursor")
        if not cursor:
            break

    tickers = list(dict.fromkeys(tickers))[:25]

    print(f"Found {len(tickers)} weather-ish markets. Logging quotes...")
    for ticker in tickers:
        ob = client.get_orderbook(ticker)
        q = best_bid_ask_from_orderbook(ob)
        insert_quote(conn, ts, ticker, q)
        print(ticker, q)

    conn.close()
    print("Done. Saved to data/kalshi_quotes.sqlite")

if __name__ == "__main__":
    main()
