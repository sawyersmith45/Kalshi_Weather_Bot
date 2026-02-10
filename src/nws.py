import requests

NWS_BASE = "https://api.weather.gov"

def get_forecast_periods(lat: float, lon: float, timeout_s: int = 20):
    headers = {"User-Agent": "kalshi-weather-bot (contact: you@example.com)"}

    p = requests.get(f"{NWS_BASE}/points/{lat},{lon}", headers=headers, timeout=timeout_s)
    p.raise_for_status()
    points = p.json()

    forecast_url = points["properties"]["forecast"]
    f = requests.get(forecast_url, headers=headers, timeout=timeout_s)
    f.raise_for_status()
    forecast = f.json()

    return forecast["properties"]["periods"]

def get_day_high_f(lat: float, lon: float, yyyy_mm_dd: str):
    periods = get_forecast_periods(lat, lon)
    for per in periods:
        if not per.get("isDaytime"):
            continue
        start = per.get("startTime", "")
        if start[:10] == yyyy_mm_dd:
            return per.get("temperature")
    return None
