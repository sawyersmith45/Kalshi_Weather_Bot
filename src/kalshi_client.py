import base64
import os
import time
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

load_dotenv()

TRADE_API_PREFIX = "/trade-api/v2"


def _as_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def _parse_levels(levels):
    """
    Accepts multiple formats:
      - [{"price": 47, "size": 3}, ...]
      - [[47, 3], [46, 10], ...]
      - [{"yes_price": 47, "count": 3}, ...]
      - {"47": 3, "46": 10, ...}
    Returns list of (price:int, size:int)
    """
    out = []

    if levels is None:
        return out

    if isinstance(levels, dict):
        for k, v in levels.items():
            p = _as_int(k, None)
            s = _as_int(v, 0)
            if p is not None:
                out.append((p, s))
        return out

    if not isinstance(levels, (list, tuple)):
        return out

    for lvl in levels:
        p = None
        s = 0

        if isinstance(lvl, dict):
            p = lvl.get("price")
            if p is None:
                p = lvl.get("yes_price") or lvl.get("px") or lvl.get("p")
            s = lvl.get("size")
            if s is None:
                s = lvl.get("quantity") or lvl.get("count") or lvl.get("sz") or lvl.get("q")
        elif isinstance(lvl, (list, tuple)):
            if len(lvl) > 0:
                p = lvl[0]
            if len(lvl) > 1:
                s = lvl[1]
        else:
            continue

        p = _as_int(p, None)
        s = _as_int(s, 0)

        if p is not None:
            out.append((p, s))

    return out


def best_bid_ask_from_orderbook(ob: dict):
    book = ob.get("orderbook") if isinstance(ob, dict) and isinstance(ob.get("orderbook"), dict) else ob
    if not isinstance(book, dict):
        return {
            "best_yes_bid": None, "best_yes_bid_sz": 0,
            "best_yes_ask": None, "best_yes_ask_sz": 0,
            "best_no_bid": None,  "best_no_bid_sz": 0,
            "best_no_ask": None,  "best_no_ask_sz": 0,
        }

    yes_levels = book.get("yes")
    no_levels = book.get("no")

    def levels_to_pairs(levels):
        out = []
        if not isinstance(levels, list):
            return out
        for lvl in levels:
            if isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                try:
                    out.append((int(lvl[0]), int(lvl[1])))
                except Exception:
                    pass
            elif isinstance(lvl, dict):
                p = lvl.get("price")
                s = lvl.get("size") or lvl.get("count")
                try:
                    out.append((int(p), int(s)))
                except Exception:
                    pass
        return out

    ys = levels_to_pairs(yes_levels)
    ns = levels_to_pairs(no_levels)

    best_yes_bid = None
    best_yes_bid_sz = 0
    best_no_bid = None
    best_no_bid_sz = 0

    if ys:
        best_yes_bid, best_yes_bid_sz = max(ys, key=lambda x: x[0])
    if ns:
        best_no_bid, best_no_bid_sz = max(ns, key=lambda x: x[0])

    best_yes_ask = (100 - best_no_bid) if best_no_bid is not None else None
    best_yes_ask_sz = best_no_bid_sz if best_yes_ask is not None else 0

    best_no_ask = (100 - best_yes_bid) if best_yes_bid is not None else None
    best_no_ask_sz = best_yes_bid_sz if best_no_ask is not None else 0

    return {
        "best_yes_bid": best_yes_bid,
        "best_yes_bid_sz": int(best_yes_bid_sz or 0),
        "best_yes_ask": best_yes_ask,
        "best_yes_ask_sz": int(best_yes_ask_sz or 0),
        "best_no_bid": best_no_bid,
        "best_no_bid_sz": int(best_no_bid_sz or 0),
        "best_no_ask": best_no_ask,
        "best_no_ask_sz": int(best_no_ask_sz or 0),
    }




class KalshiClient:
    def __init__(self, base_url: str | None = None, timeout_s: int = 20):
        self.base_url = (base_url or os.getenv("KALSHI_BASE_URL", "https://api.elections.kalshi.com")).rstrip("/")
        self.timeout_s = timeout_s

        self.key_id = os.getenv("KALSHI_KEY_ID")
        self.private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")

        self._private_key = None
        self._session = requests.Session()

    def _timestamp_ms(self) -> str:
        return str(int(time.time() * 1000))

    def _load_private_key(self):
        if self._private_key is not None:
            return self._private_key
        if not self.private_key_path:
            raise RuntimeError("KALSHI_PRIVATE_KEY_PATH not set")

        from cryptography.hazmat.primitives import serialization

        with open(self.private_key_path, "rb") as f:
            raw = f.read().strip()

        if b"-----BEGIN" not in raw:
            raw = b"-----BEGIN PRIVATE KEY-----\n" + raw + b"\n-----END PRIVATE KEY-----\n"

        self._private_key = serialization.load_pem_private_key(raw, password=None)
        return self._private_key

    def _sign(self, message: str) -> str:
        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography.hazmat.primitives import hashes

        key = self._load_private_key()
        sig = key.sign(
            message.encode("utf-8"),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode("utf-8")

    def _auth_headers(self, method: str, full_path_with_query: str) -> dict:
        if not self.key_id or not self.private_key_path:
            return {}

        ts = self._timestamp_ms()
        path_no_query = full_path_with_query.split("?", 1)[0]
        payload = f"{ts}{method.upper()}{path_no_query}"

        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": self._sign(payload),
        }

    def _request(self, method: str, path: str, params=None, json=None):
        if not path.startswith("/"):
            path = "/" + path

        url = f"{self.base_url}{path}"
        req = requests.Request(method.upper(), url, params=params, json=json)
        prepared = self._session.prepare_request(req)

        parsed = urlparse(prepared.url)
        full_path_with_query = parsed.path + (f"?{parsed.query}" if parsed.query else "")

        headers = self._auth_headers(method, full_path_with_query)
        for k, v in headers.items():
            prepared.headers[k] = v

        r = self._session.send(prepared, timeout=self.timeout_s)
        if not r.ok:
            try:
                detail = r.json()
            except Exception:
                detail = r.text
            raise requests.HTTPError(f"{r.status_code} {r.reason} for {r.url} | {detail}", response=r)

        if r.content:
            return r.json()
        return {}

    def get_series(self, limit: int = 200, cursor: str | None = None, **params):
        q = {"limit": limit, **params}
        if cursor:
            q["cursor"] = cursor
        return self._request("GET", f"{TRADE_API_PREFIX}/series", params=q)

    def get_markets(self, limit: int = 200, cursor: str | None = None, **params):
        q = {"limit": limit, **params}
        if cursor:
            q["cursor"] = cursor
        return self._request("GET", f"{TRADE_API_PREFIX}/markets", params=q)

    def get_market(self, ticker: str):
        return self._request("GET", f"{TRADE_API_PREFIX}/markets/{ticker}")

    def get_orderbook(self, ticker: str):
        return self._request("GET", f"{TRADE_API_PREFIX}/markets/{ticker}/orderbook")

    def get_events(self, limit: int = 200, cursor: str | None = None, **params):
        q = {"limit": limit, **params}
        if cursor:
            q["cursor"] = cursor
        return self._request("GET", f"{TRADE_API_PREFIX}/events", params=q)

    def get_event(self, event_ticker: str):
        return self._request("GET", f"{TRADE_API_PREFIX}/events/{event_ticker}")

    def get_portfolio_balance(self):
        return self._request("GET", f"{TRADE_API_PREFIX}/portfolio/balance")

    def get_positions(self, limit: int = 200, cursor: str | None = None, **params):
        q = {"limit": limit, **params}
        if cursor:
            q["cursor"] = cursor
        return self._request("GET", f"{TRADE_API_PREFIX}/portfolio/positions", params=q)

    def get_orders(self, limit: int = 50, cursor: str | None = None, status: str | None = None):
        params = {"limit": int(limit)}
        if cursor:
            params["cursor"] = cursor
        if status:
            params["status"] = status
        return self._request("GET", f"{TRADE_API_PREFIX}/portfolio/orders", params=params)

    def get_order(self, order_id: str):
        return self._request("GET", f"{TRADE_API_PREFIX}/portfolio/orders/{order_id}")

    def cancel_order(self, order_id: str):
        return self._request("DELETE", f"{TRADE_API_PREFIX}/portfolio/orders/{order_id}")

    def create_order(
        self,
        ticker: str,
        side: str,
        action: str,
        count: int,
        type_: str = "limit",
        yes_price: int | None = None,
        no_price: int | None = None,
        client_order_id: str | None = None,
        post_only: bool | None = None,
        reduce_only: bool | None = None,
        time_in_force: str | None = None,
    ):
        payload = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": int(count),
            "type": type_,
        }

        if yes_price is not None and no_price is not None:
            raise ValueError("Provide only one of yes_price or no_price")

        if yes_price is not None:
            payload["yes_price"] = int(yes_price)
        if no_price is not None:
            payload["no_price"] = int(no_price)

        if client_order_id:
            payload["client_order_id"] = client_order_id

        if post_only is not None:
            payload["post_only"] = bool(post_only)

        if reduce_only is not None:
            payload["reduce_only"] = bool(reduce_only)

        if time_in_force is not None:
            payload["time_in_force"] = time_in_force

        return self._request("POST", f"{TRADE_API_PREFIX}/portfolio/orders", json=payload)
