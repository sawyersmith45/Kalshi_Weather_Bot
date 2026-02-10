# src/weather_sources.py
import os
import math
import requests
import requests
from datetime import datetime
from calendar import monthrange


try:
    from src.nws import get_day_high_f as nws_high_f
except Exception:
    nws_high_f = None


class WeatherProvider:
    name = "base"

    def get_high_f(self, lat: float, lon: float, target_date: str, timezone: str = "UTC") -> float | None:
        raise NotImplementedError

    def get_high_dist(self, lat: float, lon: float, target_date: str, timezone: str = "UTC"):
        return None


class NWSProvider(WeatherProvider):
    name = "nws"

    def get_high_f(self, lat: float, lon: float, target_date: str, timezone: str = "UTC") -> float | None:
        if nws_high_f is None:
            return None
        v = nws_high_f(lat, lon, target_date)
        return None if v is None else float(v)


class OpenMeteoProvider(WeatherProvider):
    name = "open_meteo"

    def __init__(self, timeout_s: int = 15):
        self.timeout_s = timeout_s

    def get_high_f(self, lat: float, lon: float, target_date: str, timezone: str = "UTC") -> float | None:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "timezone": timezone,
            "daily": "temperature_2m_max",
            "temperature_unit": "fahrenheit",
            "start_date": target_date,
            "end_date": target_date,
        }
        r = requests.get(url, params=params, timeout=self.timeout_s)
        r.raise_for_status()
        j = r.json() or {}
        daily = j.get("daily") or {}
        tmax = (daily.get("temperature_2m_max") or [None])[0]
        return None if tmax is None else float(tmax)


class OpenMeteoEnsembleProvider(WeatherProvider):
    name = "open_meteo_ens"

    def __init__(self, timeout_s: int = 15):
        self.timeout_s = timeout_s

    def get_high_f(self, lat: float, lon: float, target_date: str, timezone: str = "UTC") -> float | None:
        d = self.get_high_dist(lat, lon, target_date, timezone=timezone)
        if not d:
            return None
        mu = d.get("mu")
        return None if mu is None else float(mu)

    def get_high_dist(self, lat: float, lon: float, target_date: str, timezone: str = "UTC"):
        url = "https://ensemble-api.open-meteo.com/v1/ensemble"
        params = {
            "latitude": lat,
            "longitude": lon,
            "timezone": timezone,
            "daily": "temperature_2m_max",
            "temperature_unit": "fahrenheit",
            "start_date": target_date,
            "end_date": target_date,
            "models": "gfs_seamless,ecmwf_ifs04,gem_global,icon_global,meteofrance_arpege,ukmo_global_deterministic,jma_seamless",
        }
        r = requests.get(url, params=params, timeout=self.timeout_s)
        r.raise_for_status()
        j = r.json() or {}
        daily = j.get("daily") or {}
        series = daily.get("temperature_2m_max")
        if not isinstance(series, list) or not series:
            return None
        vals = [float(x) for x in series if x is not None]
        if not vals:
            return None
        mu = sum(vals) / len(vals)
        var = sum((x - mu) ** 2 for x in vals) / max(1, (len(vals) - 1))
        std = math.sqrt(var)
        return {"mu": float(mu), "std": float(std), "members": vals}


class AnalogEnsembleProvider(WeatherProvider):
    """Pull historical analogs from reanalysis data to generate a distribution.
    
    For each day, fetch the past 30-50 years of similar-calendar-date observations.
    This provides a rich empirical ensemble (40-50 members) for better tail behavior.
    """
    name = "analog_ens"

    def __init__(self, timeout_s: int = 20):
        self.timeout_s = timeout_s

    def get_high_f(self, lat: float, lon: float, target_date: str, timezone: str = "UTC") -> float | None:
        d = self.get_high_dist(lat, lon, target_date, timezone=timezone)
        if not d:
            return None
        mu = d.get("mu")
        return None if mu is None else float(mu)

    def get_high_dist(self, lat: float, lon: float, target_date: str, timezone: str = "UTC"):
        """Fetch historical data for this date ±5 days across 45 years."""
        try:
            # Parse target date (YYYY-MM-DD)
            parts = target_date.split("-")
            if len(parts) != 3:
                return None
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            
            # Fetch data from 30 years ago to present, for this date ±5 days
            members = []
            start_year = year - 45
            
            # Window around target date: target_date-5 to target_date+5
            from datetime import datetime as dt, timedelta
            target_dt = dt(year, month, day)
            window_start = (target_dt - timedelta(days=5)).strftime("%Y-%m-%d")
            window_end = (target_dt + timedelta(days=5)).strftime("%Y-%m-%d")
            
            # Fetch archive data for this window across all past years
            for y in range(start_year, year):
                hist_start = window_start.replace(str(year), str(y), 1)
                hist_end = window_end.replace(str(year), str(y), 1)
                
                try:
                    url = "https://archive-api.open-meteo.com/v1/archive"
                    params = {
                        "latitude": lat,
                        "longitude": lon,
                        "timezone": timezone,
                        "daily": "temperature_2m_max",
                        "temperature_unit": "fahrenheit",
                        "start_date": hist_start,
                        "end_date": hist_end,
                    }
                    r = requests.get(url, params=params, timeout=self.timeout_s)
                    r.raise_for_status()
                    j = r.json() or {}
                    daily = j.get("daily") or {}
                    highs = daily.get("temperature_2m_max") or []
                    for h in highs:
                        if h is not None:
                            members.append(float(h))
                except Exception:
                    pass  # Skip failed years
            
            if not members:
                return None
            
            mu = sum(members) / len(members)
            var = sum((x - mu) ** 2 for x in members) / max(1, (len(members) - 1))
            std = math.sqrt(var)
            return {"mu": float(mu), "std": float(std), "members": members}
        except Exception:
            return None


class OpenMeteoHourlyFeaturesProvider(WeatherProvider):
    name = "open_meteo_hourly_features"

    def __init__(self, timeout_s: int = 15):
        self.timeout_s = timeout_s

    def get_features(self, lat: float, lon: float, target_date: str, timezone: str = "UTC"):
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "timezone": timezone,
            "hourly": "cloud_cover,precipitation_probability,pressure_msl,temperature_2m",
            "start_date": target_date,
            "end_date": target_date,
            "temperature_unit": "fahrenheit",
        }
        r = requests.get(url, params=params, timeout=self.timeout_s)
        r.raise_for_status()
        j = r.json() or {}
        hourly = j.get("hourly") or {}
        cc = hourly.get("cloud_cover") or []
        pp = hourly.get("precipitation_probability") or []
        pr = hourly.get("pressure_msl") or []
        def _avg(xs):
            ys = [float(x) for x in xs if x is not None]
            return 0.0 if not ys else sum(ys) / len(ys)
        cloud = _avg(cc) / 100.0
        precip = _avg(pp) / 100.0
        pressure = _avg(pr)
        front_risk = 0.0
        if len(pr) >= 2:
            p2 = [float(x) for x in pr if x is not None]
            if len(p2) >= 2:
                dp = max(p2) - min(p2)
                front_risk = min(1.0, max(0.0, dp / 15.0))
        cloud_risk = min(1.0, max(0.0, 0.65 * cloud + 0.35 * precip))
        return {"front_risk": float(front_risk), "cloud_risk": float(cloud_risk), "pressure": float(pressure)}


class OpenMeteoObservedHighProvider(WeatherProvider):
    name = "open_meteo_observed_high"

    def __init__(self, timeout_s: int = 15):
        self.timeout_s = timeout_s

    def get_high_f(self, lat: float, lon: float, target_date: str, timezone: str = "UTC") -> float | None:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "timezone": timezone,
            "daily": "temperature_2m_max",
            "temperature_unit": "fahrenheit",
            "start_date": target_date,
            "end_date": target_date,
            "past_days": 2,
        }
        r = requests.get(url, params=params, timeout=self.timeout_s)
        r.raise_for_status()
        j = r.json() or {}
        daily = j.get("daily") or {}
        dates = daily.get("time") or []
        highs = daily.get("temperature_2m_max") or []
        if not dates or not highs or len(dates) != len(highs):
            return None
        for d, h in zip(dates, highs):
            if str(d) == str(target_date):
                return None if h is None else float(h)
        return None


class OpenMeteoArchiveProvider(WeatherProvider):
    name = "open_meteo_archive"

    def __init__(self, timeout_s: int = 20):
        self.timeout_s = timeout_s

    def get_high_f(self, lat: float, lon: float, target_date: str, timezone: str = "UTC") -> float | None:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "timezone": timezone,
            "daily": "temperature_2m_max",
            "temperature_unit": "fahrenheit",
            "start_date": target_date,
            "end_date": target_date,
        }
        r = requests.get(url, params=params, timeout=self.timeout_s)
        r.raise_for_status()
        j = r.json() or {}
        daily = j.get("daily") or {}
        tmax = (daily.get("temperature_2m_max") or [None])[0]
        return None if tmax is None else float(tmax)


def prob_between_normal(lo: float, hi: float, mu: float, sigma: float):
    if sigma <= 0:
        return 1.0 if (mu >= lo and mu < hi) else 0.0
    def cdf(x):
        z = (x - mu) / (sigma * math.sqrt(2.0))
        return 0.5 * (1.0 + math.erf(z))
    return max(0.0, min(1.0, cdf(hi) - cdf(lo)))


def prob_between_empirical(lo: float, hi: float, highs: list[float], smooth_sigma: float):
    if not highs:
        return None
    s = smooth_sigma if smooth_sigma > 0 else 0.9
    def cdf(x, m):
        z = (x - m) / (s * math.sqrt(2.0))
        return 0.5 * (1.0 + math.erf(z))
    ps = []
    for m in highs:
        ps.append(max(0.0, min(1.0, cdf(hi, float(m)) - cdf(lo, float(m)))))
    return max(0.0, min(1.0, sum(ps) / len(ps)))


def fuse_weather(
    lat: float,
    lon: float,
    target_date: str,
    base_sigma_f: float,
    providers: list[WeatherProvider],
    weights: dict,
    provider_bias_f: dict | None = None,
    hourly_feature_provider: OpenMeteoHourlyFeaturesProvider | None = None,
    timezone: str = "UTC",
):
    dist = {}
    readings = []
    members_all = []

    for p in providers:
        try:
            d = p.get_high_dist(lat, lon, target_date, timezone=timezone)
        except Exception:
            d = None

        mu = None
        std = None
        members = None

        if isinstance(d, dict):
            mu = d.get("mu")
            std = d.get("std")
            members = d.get("members")
        else:
            try:
                mu = p.get_high_f(lat, lon, target_date, timezone=timezone)
            except Exception:
                mu = None

        if mu is None:
            continue

        mu = float(mu)
        if provider_bias_f and p.name in provider_bias_f:
            mu = mu + float(provider_bias_f.get(p.name) or 0.0)

        if std is None:
            std = float(base_sigma_f)
        else:
            std = float(std)
            if std <= 0:
                std = float(base_sigma_f)

        dist[p.name] = {"mu": float(mu), "std": float(std)}
        readings.append((p.name, float(mu)))

        if isinstance(members, list):
            for x in members:
                if x is not None:
                    members_all.append(float(x))

    if not dist:
        return None

    wts = []
    mus = []
    vars_ = []

    for name, d in dist.items():
        w = float(weights.get(name, 1.0))
        wts.append(w)
        mus.append(float(d["mu"]))
        vars_.append(float(d["std"]) ** 2)

    wsum = sum(wts)
    if wsum <= 0:
        return None

    mu_f = sum(w * m for w, m in zip(wts, mus)) / wsum
    var_f = sum(w * v for w, v in zip(wts, vars_)) / wsum
    sigma_f = math.sqrt(max(0.01, var_f))

    meta = {"readings": readings, "dist": dist}

    if hourly_feature_provider is not None:
        try:
            feats = hourly_feature_provider.get_features(lat, lon, target_date, timezone=timezone)
        except Exception:
            feats = None
        if isinstance(feats, dict):
            meta["features"] = feats

    if members_all:
        mu_m = sum(members_all) / len(members_all)
        var_m = sum((x - mu_m) ** 2 for x in members_all) / max(1, (len(members_all) - 1))
        meta["ens_members"] = int(len(members_all))
        meta["ens_std"] = float(math.sqrt(max(0.0, var_m)))
        disagree = []
        for name, d in dist.items():
            disagree.append(abs(float(d["mu"]) - mu_f))
        meta["disagree_sigma"] = float(sum(disagree) / len(disagree)) if disagree else 0.0

    return {"mu": float(mu_f), "sigma": float(sigma_f), "high_members": members_all, "meta": meta}
# Snow ensemble provider
class OpenMeteoMonthlySnowEnsembleProvider:
    name = "open_meteo_monthly_snow_ens"

    def __init__(self, timeout_s: int = 20, models: str = "gfs_seamless,ecmwf_ifs"):
        self.timeout_s = timeout_s
        self.models = models

    def get_month_members_inches(self, lat: float, lon: float, year: int, month: int, timezone: str):
        days = monthrange(year, month)[1]
        start = f"{year:04d}-{month:02d}-01"
        end = f"{year:04d}-{month:02d}-{days:02d}"

        url = "https://ensemble-api.open-meteo.com/v1/ensemble"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "daily": "snowfall_sum",
            "timezone": timezone,
            "models": self.models,
        }

        r = requests.get(url, params=params, timeout=self.timeout_s)
        r.raise_for_status()
        j = r.json() or {}

        daily = j.get("daily") or {}
        times = daily.get("time") or []
        if not times:
            return None

        members = []

        for k, v in daily.items():
            if not isinstance(k, str):
                continue
            if not k.startswith("snowfall_sum_"):
                continue
            if not isinstance(v, list) or len(v) != len(times):
                continue
            try:
                total_mm = sum(0.0 if x is None else float(x) for x in v)
            except Exception:
                continue
            total_in = total_mm / 25.4
            members.append(float(total_in))

        if len(members) == 0:
            return None

        return members
