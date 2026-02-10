import os
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


def _env(k: str, default: str = "") -> str:
    v = os.getenv(k)
    return v if v is not None and v != "" else default


def norm_team(s: str) -> str:
    return "".join(ch.lower() for ch in s.strip() if ch.isalnum())


def parse_iso_utc(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


def fetch_nba_totals() -> List[dict]:
    api_key = _env("ODDS_API_KEY")
    sport = _env("ODDS_SPORT", "basketball_nba")
    regions = _env("ODDS_REGIONS", "us")
    markets = _env("ODDS_MARKETS", "totals")
    odds_format = _env("ODDS_ODDS_FORMAT", "american")

    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": "iso",
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def extract_consensus_total(game: dict) -> Optional[Tuple[float, float]]:
    home = game.get("home_team")
    away = game.get("away_team")
    if not home or not away:
        return None

    best = None
    best_dev = None

    for bm in (game.get("bookmakers") or []):
        for m in (bm.get("markets") or []):
            if m.get("key") != "totals":
                continue
            outs = m.get("outcomes") or []
            over = None
            under = None
            for o in outs:
                name = o.get("name")
                point = o.get("point")
                if name == "Over" and point is not None:
                    over = float(point)
                if name == "Under" and point is not None:
                    under = float(point)
            if over is None or under is None:
                continue

            dev = abs(over - under)
            if best is None or (best_dev is not None and dev < best_dev):
                best = (over, under)
                best_dev = dev

    return best


def build_totals_by_matchup() -> Dict[str, dict]:
    raw = fetch_nba_totals()
    out: Dict[str, dict] = {}

    for g in raw:
        commence_time = g.get("commence_time")
        home = g.get("home_team")
        away = g.get("away_team")
        if not commence_time or not home or not away:
            continue

        total = extract_consensus_total(g)
        if total is None:
            continue

        over, under = total
        key = f"{norm_team(away)}@{norm_team(home)}"
        out[key] = {
            "commence_time_utc": parse_iso_utc(commence_time),
            "home": home,
            "away": away,
            "total": (over + under) / 2.0,
        }

    return out
