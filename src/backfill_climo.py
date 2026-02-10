import math
import os
import requests
from datetime import date, datetime
from collections import defaultdict

from src.db import open_db

SERIES_LOC = {
    "KXHIGHNY": (40.7829, -73.9654),
}
SERIES_TZ = {
    "KXHIGHNY": "America/New_York",
}

def mean(xs):
    return sum(xs) / len(xs)

def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def fetch_daily_tmax_f(lat, lon, start, end, tz):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_max",
        "timezone": tz,
        "temperature_unit": "fahrenheit",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    j = r.json() or {}
    daily = j.get("daily") or {}
    t = daily.get("time") or []
    mx = daily.get("temperature_2m_max") or []
    return list(zip(t, mx))

def doy_of(ymd: str) -> int:
    y, m, d = map(int, ymd.split("-"))
    return date(y, m, d).timetuple().tm_yday

def main():
    conn = open_db("data/kalshi_quotes.sqlite")

    start_year = int(os.getenv("CLIMO_START_YEAR", "1990"))
    end_year = int(os.getenv("CLIMO_END_YEAR", str(datetime.now().year - 1)))

    for series, (lat, lon) in SERIES_LOC.items():
        tz = SERIES_TZ.get(series, "UTC")
        start = f"{start_year:04d}-01-01"
        end = f"{end_year:04d}-12-31"

        rows = fetch_daily_tmax_f(lat, lon, start, end, tz)

        bucket = defaultdict(list)
        for ymd, tmax in rows:
            if tmax is None:
                continue
            bucket[doy_of(ymd)].append(float(tmax))

        for doy, vals in bucket.items():
            m = mean(vals)
            s = std(vals)
            n = len(vals)
            conn.execute(
                "INSERT OR REPLACE INTO climo_daily(series, doy, mean_high_f, std_high_f, n) VALUES (?,?,?,?,?)",
                (series, int(doy), float(m), float(s), int(n)),
            )

        conn.commit()
        print(f"backfilled {series}: days={len(bucket)} years={start_year}-{end_year}")

if __name__ == "__main__":
    main()
