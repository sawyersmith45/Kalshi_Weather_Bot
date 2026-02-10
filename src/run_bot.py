import os
import time
import math
import re
import logging
import traceback
from datetime import timedelta, datetime, timezone
from zoneinfo import ZoneInfo

import src.kalshi_client as kc
print("USING kalshi_client FROM:", kc.__file__)

from dotenv import load_dotenv

from src.kalshi_client import KalshiClient, best_bid_ask_from_orderbook
from src.db import (
    open_db,
    record_trade,
    get_position,
    upsert_position,
    get_last_trade_ts,
    set_last_trade_ts,
    set_state,
    get_state,
    log_live_order,
    snapshot_forecast,
    snapshot_markets,
    snapshot_weather_features,
    snapshot_provider_readings,
    upsert_outcome,
    get_db_stats,
    check_db_integrity,
    cleanup_old_snapshots,
    vacuum_and_analyze,
    record_forecast_prediction,
    update_forecast_outcome,
    update_provider_mse,
    compute_heat_map,
    compute_calibration_metrics,
    compute_pnl_metrics,
    compute_pnl_by_series,
    compute_provider_accuracy,
    get_provider_rankings,
    compute_forecast_method_accuracy,
    compute_fill_analytics,
    compute_series_correlations,
    compute_correlated_exposure,
    compute_portfolio_delta,
    detect_vol_spikes,
    BatchTransaction,
)
from src.weather_sources import (
    NWSProvider,
    OpenMeteoProvider,
    VisualCrossingProvider,
    OpenMeteoEnsembleProvider,
    AnalogEnsembleProvider,
    OpenMeteoHourlyFeaturesProvider,
    OpenMeteoObservedHighProvider,
    OpenMeteoArchiveProvider,
    fuse_weather,
    prob_between_empirical,
    prob_between_normal,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Environment variables
LIVE_TRADING = os.getenv("LIVE_TRADING", "0") == "1"
KILL_SWITCH = os.getenv("KILL_SWITCH", "0") == "1"

RUN_EVERY_SECONDS = int(os.getenv("RUN_EVERY_SECONDS", "120"))
FAST_EVERY_SECONDS = int(os.getenv("FAST_EVERY_SECONDS", "20"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "900"))

MIN_EDGE_CENTS = float(os.getenv("MIN_EDGE_CENTS", "3.0"))
COST_BUFFER_CENTS = float(os.getenv("COST_BUFFER_CENTS", "2.0"))
MAX_SPREAD_CENTS = int(os.getenv("MAX_SPREAD_CENTS", "10"))

MIN_VOL_24H = int(os.getenv("MIN_VOL_24H", "0"))
MIN_OPEN_INTEREST = int(os.getenv("MIN_OPEN_INTEREST", "0"))

TRADE_STEP = int(os.getenv("TRADE_STEP", "25"))
BASE_TARGET = int(os.getenv("BASE_TARGET", "25"))
MAX_TARGET = int(os.getenv("MAX_TARGET", "75"))
MAX_ABS_QTY_PER_TICKER = int(os.getenv("MAX_ABS_QTY_PER_TICKER", "100"))

MAX_TOTAL_RISK_DOLLARS = float(os.getenv("MAX_TOTAL_RISK_DOLLARS", "50.0"))
MIN_BALANCE_CENTS = int(os.getenv("MIN_BALANCE_CENTS", "200"))

# Risk management settings
RISK_INCLUDE_PENDING_ORDERS = os.getenv("RISK_INCLUDE_PENDING_ORDERS", "1") == "1"  # Account for orders in this loop
USE_LIVE_POSITIONS_FOR_RISK = os.getenv("USE_LIVE_POSITIONS_FOR_RISK", "1") == "1"  # Use API instead of DB
RISK_LOG_INTERVAL_SECS = float(os.getenv("RISK_LOG_INTERVAL_SECS", "10"))  # Log risk every N seconds

MAX_LIVE_ORDERS_PER_LOOP = int(os.getenv("MAX_LIVE_ORDERS_PER_LOOP", "10"))
MAX_LIVE_ORDERS_PER_DAY = int(os.getenv("MAX_LIVE_ORDERS_PER_DAY", "500"))
LIVE_QTY_CAP = int(os.getenv("LIVE_QTY_CAP", "1"))

MAX_SELL_ORDERS_PER_LOOP = int(os.getenv("MAX_SELL_ORDERS_PER_LOOP", str(MAX_LIVE_ORDERS_PER_LOOP)))
MAX_BUY_ORDERS_PER_LOOP = int(os.getenv("MAX_BUY_ORDERS_PER_LOOP", str(MAX_LIVE_ORDERS_PER_LOOP)))

MAKER_MODE = os.getenv("MAKER_MODE", "0") == "1"
MAKER_IMPROVE_CENTS = int(os.getenv("MAKER_IMPROVE_CENTS", "1"))

STALE_ORDER_SECONDS = int(os.getenv("STALE_ORDER_SECONDS", "120"))
CANCEL_STALE_ORDERS = os.getenv("CANCEL_STALE_ORDERS", "0") == "1"
ORDER_AMENDING_ENABLED = os.getenv("ORDER_AMENDING_ENABLED", "1") == "1"  # Smart amend instead of cancel+relist

ORDERBOOK_VALIDATE_TOPN = int(os.getenv("ORDERBOOK_VALIDATE_TOPN", "12"))
MIN_LIQUIDITY_SZ = int(os.getenv("MIN_LIQUIDITY_SZ", "1"))

# Order management improvements
MIN_QUEUE_DEPTH_FOR_MAKER = int(os.getenv("MIN_QUEUE_DEPTH_FOR_MAKER", "5"))  # Min liquidity to trust maker
ORDERBOOK_MAX_AGE_SECONDS = int(os.getenv("ORDERBOOK_MAX_AGE_SECONDS", "30"))  # Max age of orderbook data
FILL_VALIDATION_TOLERANCE = float(os.getenv("FILL_VALIDATION_TOLERANCE", "0.95"))  # Accept fills >= 95% of requested
MAKER_FALLBACK_ATTEMPTS = int(os.getenv("MAKER_FALLBACK_ATTEMPTS", "1"))  # Only fallback once per order
ORDER_FILL_TIMEOUT_SECONDS = int(os.getenv("ORDER_FILL_TIMEOUT_SECONDS", "10"))  # How long to wait for fill
REJECT_ORDER_IF_NO_FILL_PCT = float(os.getenv("REJECT_ORDER_IF_NO_FILL_PCT", "0.1"))  # Reject if <10% fills after timeout

EMPIRICAL_MIN_MEMBERS = int(os.getenv("EMPIRICAL_MIN_MEMBERS", "5"))  # Lower threshold to use empirical more often
EMPIRICAL_SMOOTH_SIGMA = float(os.getenv("EMPIRICAL_SMOOTH_SIGMA", "0.9"))

# ---- Weather Provider Configuration ----
USE_VISUAL_CROSSING = os.getenv("USE_VISUAL_CROSSING", "0") == "1"  # Optional independent provider
VISUAL_CROSSING_TIMEOUT = int(os.getenv("VISUAL_CROSSING_TIMEOUT", "15"))
USE_ANALOG_ENSEMBLE = os.getenv("USE_ANALOG_ENSEMBLE", "0") == "1"  # Disable by default (slow, 45+ archive API calls)
ANALOG_ENSEMBLE_TIMEOUT = int(os.getenv("ANALOG_ENSEMBLE_TIMEOUT", "30"))  # Timeout per fetch in seconds
WEATHER_FETCH_DEBUG = os.getenv("WEATHER_FETCH_DEBUG", "0") == "1"  # Log each weather provider result

# ---- Resilience / Circuit Breaker ----
MAX_API_ERRORS_BEFORE_PAUSE = int(os.getenv("MAX_API_ERRORS_BEFORE_PAUSE", "5"))
API_PAUSE_SECONDS_BASE = int(os.getenv("API_PAUSE_SECONDS_BASE", "60"))  # base pause seconds after threshold
API_PAUSE_MAX_SECONDS = int(os.getenv("API_PAUSE_MAX_SECONDS", "3600"))  # maximum pause
API_RETRY_ATTEMPTS = int(os.getenv("API_RETRY_ATTEMPTS", "3"))
API_RETRY_BACKOFF = float(os.getenv("API_RETRY_BACKOFF", "1.5"))  # exponential backoff multiplier

# In-memory service failure tracker for runtime resilience
_service_failures = {}

class ServicePaused(Exception):
    pass

def _now_ts():
    return time.time()

def _is_paused(service: str):
    info = _service_failures.get(service)
    if not info:
        return False
    until = info.get("pause_until", 0)
    return _now_ts() < until

def _record_failure(service: str, exc: Exception = None):
    info = _service_failures.get(service) or {"count": 0, "last": 0, "pause_until": 0}
    info["count"] = info.get("count", 0) + 1
    info["last"] = _now_ts()
    # If exceed threshold, set pause
    if info["count"] >= MAX_API_ERRORS_BEFORE_PAUSE:
        extra = info["count"] - MAX_API_ERRORS_BEFORE_PAUSE
        pause = min(API_PAUSE_SECONDS_BASE * (2 ** extra), API_PAUSE_MAX_SECONDS)
        info["pause_until"] = _now_ts() + pause
        logger.warning(f"[RESILIENCE] Service {service} paused for {pause}s after {info['count']} errors")
        # reset count so pauses escalate from pauses only
        info["count"] = 0
    _service_failures[service] = info

def _record_success(service: str):
    if service in _service_failures:
        _service_failures[service]["count"] = 0
        _service_failures[service]["last"] = _now_ts()
        _service_failures[service]["pause_until"] = 0

def safe_api_call(service: str, fn, *args, **kwargs):
    """Call external API with retries and circuit-breaker pause on repeated failures.

    Returns the function result or raises ServicePaused/last exception.
    """
    if _is_paused(service):
        raise ServicePaused(service)

    attempt = 0
    err = None
    backoff = 1.0
    while attempt < API_RETRY_ATTEMPTS:
        try:
            res = fn(*args, **kwargs)
            _record_success(service)
            return res
        except ServicePaused:
            raise
        except Exception as e:
            err = e
            attempt += 1
            logger.warning(f"[RESILIENCE] {service} call failed attempt={attempt}: {repr(e)}")
            time.sleep(backoff)
            backoff *= API_RETRY_BACKOFF
    # After retries, record failure
    _record_failure(service, err)
    raise err

BIAS_ALPHA = float(os.getenv("BIAS_ALPHA", "0.05"))
MSE_EPS = float(os.getenv("MSE_EPS", "4.0"))
W_MIN = float(os.getenv("W_MIN", "0.25"))
W_MAX = float(os.getenv("W_MAX", "3.00"))
DYNAMIC_WEIGHTS = os.getenv("DYNAMIC_WEIGHTS", "1") == "1"

SIGMA_HORIZON_K = float(os.getenv("SIGMA_HORIZON_K", "0.10"))
FRONT_SIGMA_ADD = float(os.getenv("FRONT_SIGMA_ADD", "2.0"))
CLOUD_SIGMA_ADD = float(os.getenv("CLOUD_SIGMA_ADD", "1.0"))
CLOUD_MU_ADJ = float(os.getenv("CLOUD_MU_ADJ", "0.6"))

EDGE_QTY_SCALE = float(os.getenv("EDGE_QTY_SCALE", "8.0"))
MAX_STEP_MULT = int(os.getenv("MAX_STEP_MULT", "3"))

STALE_QUOTE_CENTS = int(os.getenv("STALE_QUOTE_CENTS", "5"))

# Data staleness settings
MAX_ORDERBOOK_AGE_SECONDS = float(os.getenv("MAX_ORDERBOOK_AGE_SECONDS", "15"))  # Fresh orderbook < 15s
MAX_FORECAST_AGE_SECONDS = float(os.getenv("MAX_FORECAST_AGE_SECONDS", "300"))  # Fresh forecast < 5 min
MAX_WEATHER_PROVIDER_AGE_SECONDS = float(os.getenv("MAX_WEATHER_PROVIDER_AGE_SECONDS", "600"))  # Provider data < 10 min
REQUIRE_FRESH_ORDERBOOK = os.getenv("REQUIRE_FRESH_ORDERBOOK", "1") == "1"  # Enforce orderbook age
REQUIRE_FRESH_FORECAST = os.getenv("REQUIRE_FRESH_FORECAST", "1") == "1"  # Enforce forecast age
WARN_STALE_DATA = os.getenv("WARN_STALE_DATA", "1") == "1"  # Log warnings on stale data

NEW_POS_EDGE_BONUS = float(os.getenv("NEW_POS_EDGE_BONUS", "3.0"))

TAKE_PROFIT_CENTS = float(os.getenv("TAKE_PROFIT_CENTS", "5.0"))
STOP_LOSS_CENTS = float(os.getenv("STOP_LOSS_CENTS", "3.0"))
SELL_ONLY_IF_PROFIT = os.getenv("SELL_ONLY_IF_PROFIT", "1") == "1"
SYNC_PORTFOLIO_POSITIONS = os.getenv("SYNC_PORTFOLIO_POSITIONS", "1") == "1"
SELL_DIAGNOSTICS = os.getenv("SELL_DIAGNOSTICS", "1") == "1"
EXIT_SCAN_ALL_POSITIONS = os.getenv("EXIT_SCAN_ALL_POSITIONS", "1") == "1"
AGGRESSIVE_EDGE_CENTS = float(os.getenv("AGGRESSIVE_EDGE_CENTS", "12.0"))

# Time-based exit strategy
MAX_HOLD_SECONDS = int(os.getenv("MAX_HOLD_SECONDS", "3600"))  # 1 hour default
SCALE_OUT_PROFIT_CENTS = float(os.getenv("SCALE_OUT_PROFIT_CENTS", "2.0"))  # Scale out at 2 cents
SCALE_OUT_QTY_FRACTION = float(os.getenv("SCALE_OUT_QTY_FRACTION", "0.5"))  # Sell 50% at scale target
TIME_BASED_EXIT_EDGE = float(os.getenv("TIME_BASED_EXIT_EDGE", "1.0"))  # Looser edge requirement after time

# Forecast improvements
EXPLORE_EVENT_PAGES = int(os.getenv("EXPLORE_EVENT_PAGES", "50"))  # Explore more events (was 10)
EXPLORE_MULTIPLE_EVENTS = os.getenv("EXPLORE_MULTIPLE_EVENTS", "1") == "1"  # Trade multiple event dates per series
MAX_EVENTS_PER_SERIES = int(os.getenv("MAX_EVENTS_PER_SERIES", "3"))  # Trade top N nearest event dates
VOLATILITY_SMILE_ENABLED = os.getenv("VOLATILITY_SMILE_ENABLED", "1") == "1"  # Adjust vol by strike distance
VOLATILITY_SMILE_FACTOR = float(os.getenv("VOLATILITY_SMILE_FACTOR", "0.02"))  # Vol increase per 1% away from ATM
USE_STUDENTS_T = os.getenv("USE_STUDENTS_T", "1") == "1"  # Model fat tails with Student's t-distribution
STUDENTS_T_DF = float(os.getenv("STUDENTS_T_DF", "5.0"))  # Degrees of freedom for t-distribution (lower = fatter tails)
FORECAST_SKEW_ENABLED = os.getenv("FORECAST_SKEW_ENABLED", "1") == "1"  # Adjust forecast for skew
FORECAST_SKEW_FACTOR = float(os.getenv("FORECAST_SKEW_FACTOR", "0.5"))  # How much to weight empirical skew
LOG_FORECAST_DIAGNOSTICS = os.getenv("LOG_FORECAST_DIAGNOSTICS", "1") == "1"  # Log which forecast method was used (empirical, t, normal)
LOG_CANDIDATE_REJECTION = os.getenv("LOG_CANDIDATE_REJECTION", "0") == "1"  # Verbose logging of why candidates rejected
CALIBRATION_TRACKING_ENABLED = os.getenv("CALIBRATION_TRACKING_ENABLED", "1") == "1"  # Record forecasts and compute calibration
CALIBRATION_LOG_INTERVAL_HOURS = int(os.getenv("CALIBRATION_LOG_INTERVAL_HOURS", "6"))  # Log calibration metrics every N hours

# ===== P&L TRACKING =====
PNL_TRACKING_ENABLED = os.getenv("PNL_TRACKING_ENABLED", "1") == "1"  # Track and log P&L metrics
PNL_LOG_INTERVAL_HOURS = float(os.getenv("PNL_LOG_INTERVAL_HOURS", "1"))  # Log P&L summary every N hours
PNL_LOOKBACK_HOURS = int(os.getenv("PNL_LOOKBACK_HOURS", "24"))  # Compute P&L over last N hours

# ===== FORECAST ACCURACY TRACKING =====
FORECAST_ACCURACY_TRACKING_ENABLED = os.getenv("FORECAST_ACCURACY_TRACKING_ENABLED", "1") == "1"  # Track provider accuracy
FORECAST_ACCURACY_LOG_INTERVAL_HOURS = float(os.getenv("FORECAST_ACCURACY_LOG_INTERVAL_HOURS", "2"))  # Log every N hours
FORECAST_ACCURACY_LOOKBACK_HOURS = int(os.getenv("FORECAST_ACCURACY_LOOKBACK_HOURS", "24"))  # Analyze last N hours

# ===== FILL ANALYTICS TRACKING =====
FILL_ANALYTICS_TRACKING_ENABLED = os.getenv("FILL_ANALYTICS_TRACKING_ENABLED", "1") == "1"  # Track execution quality
FILL_ANALYTICS_LOG_INTERVAL_HOURS = float(os.getenv("FILL_ANALYTICS_LOG_INTERVAL_HOURS", "1"))  # Log every N hours
FILL_ANALYTICS_LOOKBACK_HOURS = int(os.getenv("FILL_ANALYTICS_LOOKBACK_HOURS", "24"))  # Analyze last N hours

# ===== CORRELATION HEDGING =====
CORRELATION_HEDGING_ENABLED = os.getenv("CORRELATION_HEDGING_ENABLED", "1") == "1"  # Prevent over-concentration in correlated series
CORRELATION_THRESHOLD = float(os.getenv("CORRELATION_THRESHOLD", "0.3"))  # Detect correlations > this
MAX_CORRELATED_EXPOSURE_PCT = float(os.getenv("MAX_CORRELATED_EXPOSURE_PCT", "40.0"))  # Max % of portfolio in correlated positions
CORRELATION_LOG_INTERVAL_HOURS = float(os.getenv("CORRELATION_LOG_INTERVAL_HOURS", "2"))  # Log correlation matrix every N hours

# ===== GREEKS TRACKING (Delta) =====
GREEKS_TRACKING_ENABLED = os.getenv("GREEKS_TRACKING_ENABLED", "1") == "1"  # Track portfolio delta
GREEKS_LOG_INTERVAL_HOURS = float(os.getenv("GREEKS_LOG_INTERVAL_HOURS", "1"))  # Log every N hours

# ===== VOLATILITY ALERTS =====
VOL_ALERT_ENABLED = os.getenv("VOL_ALERT_ENABLED", "1") == "1"  # Enable spike detection
VOL_SPIKE_THRESHOLD = float(os.getenv("VOL_SPIKE_THRESHOLD", "2.0"))  # Alert if recent vol > historical vol * threshold
VOL_ALERT_LOG_INTERVAL_HOURS = float(os.getenv("VOL_ALERT_LOG_INTERVAL_HOURS", "1"))  # Log spikes every N hours

# ===== LOSS PROTECTION / ADAPTIVE EDGE GATE =====
DAILY_DRAWDOWN_STOP_PCT = float(os.getenv("DAILY_DRAWDOWN_STOP_PCT", "8.0"))  # Block new buys for day if balance drawdown exceeds this %
MAX_CONSECUTIVE_LOSSES = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "4"))  # Block new buys after this many consecutive closed losing trades
CONSECUTIVE_LOSS_LOOKBACK_HOURS = int(os.getenv("CONSECUTIVE_LOSS_LOOKBACK_HOURS", "72"))  # Trade history window for streak detection
LOSS_STREAK_COOLDOWN_MINUTES = int(os.getenv("LOSS_STREAK_COOLDOWN_MINUTES", "180"))  # Buy block duration after streak trigger

ADAPTIVE_EDGE_GATE_ENABLED = os.getenv("ADAPTIVE_EDGE_GATE_ENABLED", "1") == "1"  # Penalize required edge when execution quality degrades
ADAPTIVE_EDGE_LOOKBACK_HOURS = int(os.getenv("ADAPTIVE_EDGE_LOOKBACK_HOURS", "6"))  # Fill analytics lookback for adaptive edge
ADAPTIVE_EDGE_MIN_ORDERS = int(os.getenv("ADAPTIVE_EDGE_MIN_ORDERS", "20"))  # Minimum sample size before applying penalty
ADAPTIVE_EDGE_TARGET_FILL_RATE = float(os.getenv("ADAPTIVE_EDGE_TARGET_FILL_RATE", "70.0"))  # Fill-rate target (%)
ADAPTIVE_EDGE_REJECT_WEIGHT = float(os.getenv("ADAPTIVE_EDGE_REJECT_WEIGHT", "0.05"))  # Cents penalty per 1% fill-rate gap
ADAPTIVE_EDGE_SLIPPAGE_WEIGHT = float(os.getenv("ADAPTIVE_EDGE_SLIPPAGE_WEIGHT", "1.0"))  # Multiplier on avg slippage penalty
ADAPTIVE_EDGE_MAX_PENALTY_CENTS = float(os.getenv("ADAPTIVE_EDGE_MAX_PENALTY_CENTS", "6.0"))  # Hard cap on added penalty

# ===== ENTRY QUALITY FILTERS =====
SERIES_EXPECTANCY_FILTER_ENABLED = os.getenv("SERIES_EXPECTANCY_FILTER_ENABLED", "1") == "1"
SERIES_EXPECTANCY_LOOKBACK_HOURS = int(os.getenv("SERIES_EXPECTANCY_LOOKBACK_HOURS", "168"))
SERIES_EXPECTANCY_MIN_CLOSED_TRADES = int(os.getenv("SERIES_EXPECTANCY_MIN_CLOSED_TRADES", "20"))
SERIES_EXPECTANCY_MIN_EDGE_DOLLARS = float(os.getenv("SERIES_EXPECTANCY_MIN_EDGE_DOLLARS", "0.0"))

HOURLY_FILTER_ENABLED = os.getenv("HOURLY_FILTER_ENABLED", "1") == "1"
HOURLY_FILTER_LOOKBACK_HOURS = int(os.getenv("HOURLY_FILTER_LOOKBACK_HOURS", "336"))
HOURLY_FILTER_MIN_CLOSED_TRADES = int(os.getenv("HOURLY_FILTER_MIN_CLOSED_TRADES", "20"))
HOURLY_FILTER_MIN_EXPECTANCY_DOLLARS = float(os.getenv("HOURLY_FILTER_MIN_EXPECTANCY_DOLLARS", "0.0"))

MAX_TRADES_PER_TICKER_PER_DAY = int(os.getenv("MAX_TRADES_PER_TICKER_PER_DAY", "8"))

# ===== PROBABILITY / PRICE ENTRY GUARDS =====
MIN_FAIR_PROB_PCT = float(os.getenv("MIN_FAIR_PROB_PCT", "8.0"))  # Skip BUYs if model fair probability is too low
MIN_BUY_ASK_CENTS = int(os.getenv("MIN_BUY_ASK_CENTS", "4"))  # Skip BUYs below this ask price
MAX_BUY_ASK_CENTS = int(os.getenv("MAX_BUY_ASK_CENTS", "80"))  # Skip BUYs above this ask price

TAIL_PENALTY_ENABLED = os.getenv("TAIL_PENALTY_ENABLED", "1") == "1"  # Penalize extreme-tail probabilities
TAIL_PENALTY_CENTER_PCT = float(os.getenv("TAIL_PENALTY_CENTER_PCT", "50.0"))  # Center point where penalty is zero
TAIL_PENALTY_FREE_BAND_PCT = float(os.getenv("TAIL_PENALTY_FREE_BAND_PCT", "10.0"))  # No penalty within +/- this band
TAIL_PENALTY_SLOPE_CENTS_PER_PCT = float(os.getenv("TAIL_PENALTY_SLOPE_CENTS_PER_PCT", "0.06"))  # Cents penalty per 1% tail distance
TAIL_PENALTY_CAP_CENTS = float(os.getenv("TAIL_PENALTY_CAP_CENTS", "4.0"))  # Cap tail penalty

# ===== CALIBRATION DIAGNOSTICS =====
CALIBRATION_DIAG_LOOKBACK_HOURS = int(os.getenv("CALIBRATION_DIAG_LOOKBACK_HOURS", "168"))
CALIBRATION_DIAG_BINS = int(os.getenv("CALIBRATION_DIAG_BINS", "10"))
CALIBRATION_DIAG_MIN_SAMPLES = int(os.getenv("CALIBRATION_DIAG_MIN_SAMPLES", "30"))

DB_CLEANUP_EVERY_MINUTES = int(os.getenv("DB_CLEANUP_EVERY_MINUTES", "120"))  # Run cleanup every N minutes
DB_CLEANUP_KEEP_HOURS = int(os.getenv("DB_CLEANUP_KEEP_HOURS", "72"))  # Keep snapshots for N hours, delete older
DB_VACUUM_EVERY_HOURS = int(os.getenv("DB_VACUUM_EVERY_HOURS", "24"))  # Run VACUUM every N hours
DB_INTEGRITY_CHECK = os.getenv("DB_INTEGRITY_CHECK", "1") == "1"  # Run integrity checks on startup
DB_LOG_STATS = os.getenv("DB_LOG_STATS", "1") == "1"  # Log database stats periodically

# ===== DYNAMIC POSITION SIZING =====
DYNAMIC_POSITION_SIZING = os.getenv("DYNAMIC_POSITION_SIZING", "1") == "1"  # Enable vol-based position sizing
VOL_LOW_THRESH = float(os.getenv("VOL_LOW_THRESH", "2.0"))  # Sigma threshold for low-vol regime
VOL_HIGH_THRESH = float(os.getenv("VOL_HIGH_THRESH", "5.0"))  # Sigma threshold for high-vol regime
VOL_POSITION_MULT_LOW = float(os.getenv("VOL_POSITION_MULT_LOW", "1.5"))  # Position multiplier in low vol
VOL_POSITION_MULT_HIGH = float(os.getenv("VOL_POSITION_MULT_HIGH", "0.5"))  # Position multiplier in high vol

# ===== REGIME DETECTION & THROTTLING =====
REGIME_DETECTION_ENABLED = os.getenv("REGIME_DETECTION_ENABLED", "1") == "1"  # Enable market regime detection
REGIME_CHECK_WINDOW_MINUTES = int(os.getenv("REGIME_CHECK_WINDOW_MINUTES", "10"))  # Look back N minutes for vol spike
VOLATILITY_SPIKE_THRESHOLD = float(os.getenv("VOLATILITY_SPIKE_THRESHOLD", "2.0"))  # Spike if vol > avg * factor
LIQUIDITY_DROP_THRESHOLD = float(os.getenv("LIQUIDITY_DROP_THRESHOLD", "0.3"))  # Throttle if liquidity < threshold
PRICE_VELOCITY_THRESHOLD = float(os.getenv("PRICE_VELOCITY_THRESHOLD", "3.0"))  # Cents/minute alert threshold
REGIME_THROTTLE_LOOPS = int(os.getenv("REGIME_THROTTLE_LOOPS", "3"))  # How many loops to throttle for
REGIME_EDGE_MULTIPLIER = float(os.getenv("REGIME_EDGE_MULTIPLIER", "1.5"))  # Multiply MIN_EDGE_CENTS during throttle

# ===== PORTFOLIO CONCENTRATION LIMITS =====
CONCENTRATION_LIMITS_ENABLED = os.getenv("CONCENTRATION_LIMITS_ENABLED", "1") == "1"  # Enable concentration checks
MAX_NOTIONAL_PER_SERIES = float(os.getenv("MAX_NOTIONAL_PER_SERIES", "1000.0"))  # Max total position value per series ($)
MAX_NOTIONAL_PER_EVENT = float(os.getenv("MAX_NOTIONAL_PER_EVENT", "500.0"))  # Max total position value per event ($)
MAX_CONTRACTS_PER_SERIES = int(os.getenv("MAX_CONTRACTS_PER_SERIES", "150"))  # Max total contracts per series

SERIES_LIST_RAW = [s.strip() for s in os.getenv("SERIES_LIST", "KXHIGHNY").split(",") if s.strip()]

SERIES_LOC = {
    "KXHIGHNY": (40.7829, -73.9654),
    "KXHIGHNY0": (40.7829, -73.9654),
    "KXHIGHTDC": (38.9072, -77.0369),
    "KXHIGHCHI": (41.8781, -87.6298),
    "KXHIGHTSFO": (37.7749, -122.4194),
    "KXHIGHLAX": (34.0522, -118.2437),
    "KXHIGHDEN": (39.7392, -104.9903),
    "KXHIGHTEMPDEN": (39.7392, -104.9903),
    "KXHIGHAUS": (30.2672, -97.7431),
    "KXHIGHHOU": (29.7604, -95.3698),
    "KXHIGHMIA": (25.7617, -80.1918),
    "KXHIGHPHIL": (39.9526, -75.1652),
    "KXHIGHTSEA": (47.6062, -122.3321),
    "KXHIGHTNOLA": (29.9511, -90.0715),
    "KXHIGHTLV": (36.1699, -115.1398),
    "KXHIGHOU": (35.4676, -97.5164),
}

SERIES_TZ = {
    "KXHIGHNY": "America/New_York",
    "KXHIGHNY0": "America/New_York",
    "KXHIGHTDC": "America/New_York",
    "KXHIGHCHI": "America/Chicago",
    "KXHIGHTSFO": "America/Los_Angeles",
    "KXHIGHLAX": "America/Los_Angeles",
    "KXHIGHDEN": "America/Denver",
    "KXHIGHTEMPDEN": "America/Denver",
    "KXHIGHAUS": "America/Chicago",
    "KXHIGHHOU": "America/Chicago",
    "KXHIGHMIA": "America/New_York",
    "KXHIGHPHIL": "America/New_York",
    "KXHIGHTSEA": "America/Los_Angeles",
    "KXHIGHTNOLA": "America/Chicago",
    "KXHIGHTLV": "America/Los_Angeles",
    "KXHIGHOU": "America/Chicago",
}

BASE_SIGMA_BY_SERIES_MONTH = {
    "KXHIGHNY":  {1: 5.0, 2: 5.0, 3: 4.6, 4: 4.2, 5: 3.6, 6: 3.4, 7: 3.3, 8: 3.3, 9: 3.5, 10: 3.9, 11: 4.3, 12: 4.8},
    "KXHIGHCHI": {1: 5.4, 2: 5.4, 3: 5.0, 4: 4.5, 5: 3.9, 6: 3.6, 7: 3.5, 8: 3.5, 9: 3.8, 10: 4.3, 11: 4.8, 12: 5.2},
    "KXHIGHTDC": {1: 4.8, 2: 4.7, 3: 4.3, 4: 3.9, 5: 3.4, 6: 3.2, 7: 3.1, 8: 3.1, 9: 3.3, 10: 3.8, 11: 4.2, 12: 4.6},
    "KXHIGHTSFO": {1: 3.2, 2: 3.1, 3: 3.0, 4: 2.8, 5: 2.7, 6: 2.6, 7: 2.6, 8: 2.6, 9: 2.7, 10: 2.8, 11: 3.0, 12: 3.1},
    "KXHIGHLAX": {1: 3.0, 2: 3.0, 3: 2.9, 4: 2.8, 5: 2.7, 6: 2.7, 7: 2.8, 8: 2.9, 9: 2.9, 10: 2.9, 11: 3.0, 12: 3.0},
    "KXHIGHDEN": {1: 6.0, 2: 5.8, 3: 5.4, 4: 4.8, 5: 4.3, 6: 4.0, 7: 4.0, 8: 4.1, 9: 4.4, 10: 5.0, 11: 5.5, 12: 5.9},
    "KXHIGHTEMPDEN": {1: 6.0, 2: 5.8, 3: 5.4, 4: 4.8, 5: 4.3, 6: 4.0, 7: 4.0, 8: 4.1, 9: 4.4, 10: 5.0, 11: 5.5, 12: 5.9},
    "KXHIGHAUS": {1: 4.0, 2: 3.9, 3: 3.6, 4: 3.3, 5: 3.1, 6: 3.0, 7: 3.0, 8: 3.1, 9: 3.3, 10: 3.6, 11: 3.8, 12: 3.9},
    "KXHIGHHOU": {1: 4.0, 2: 3.9, 3: 3.6, 4: 3.3, 5: 3.1, 6: 3.0, 7: 3.0, 8: 3.1, 9: 3.3, 10: 3.6, 11: 3.8, 12: 3.9},
    "KXHIGHMIA": {1: 2.8, 2: 2.7, 3: 2.6, 4: 2.5, 5: 2.5, 6: 2.6, 7: 2.6, 8: 2.6, 9: 2.6, 10: 2.6, 11: 2.7, 12: 2.8},
    "KXHIGHPHIL": {1: 5.0, 2: 5.0, 3: 4.6, 4: 4.2, 5: 3.7, 6: 3.5, 7: 3.4, 8: 3.4, 9: 3.6, 10: 4.0, 11: 4.4, 12: 4.8},
    "KXHIGHTSEA": {1: 3.8, 2: 3.7, 3: 3.6, 4: 3.4, 5: 3.2, 6: 3.0, 7: 3.0, 8: 3.0, 9: 3.1, 10: 3.3, 11: 3.6, 12: 3.8},
    "KXHIGHTNOLA": {1: 4.0, 2: 3.9, 3: 3.6, 4: 3.3, 5: 3.1, 6: 3.0, 7: 3.0, 8: 3.1, 9: 3.3, 10: 3.6, 11: 3.8, 12: 3.9},
    "KXHIGHTLV": {1: 3.4, 2: 3.3, 3: 3.2, 4: 3.1, 5: 3.0, 6: 3.0, 7: 3.1, 8: 3.2, 9: 3.2, 10: 3.2, 11: 3.3, 12: 3.4},
    "KXHIGHOU": {1: 4.8, 2: 4.7, 3: 4.3, 4: 3.9, 5: 3.5, 6: 3.3, 7: 3.3, 8: 3.4, 9: 3.6, 10: 4.0, 11: 4.4, 12: 4.7},
}


def iso_now():
    return datetime.now(timezone.utc).isoformat()


def parse_iso(s: str | None):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _extract_first_list(d: dict):
    if not isinstance(d, dict):
        return []
    for _, v in d.items():
        if isinstance(v, list):
            return v
    return []


def fetch_live_positions(client: KalshiClient) -> dict:
    """Fetch live positions with robust parsing and error handling."""
    try:
        resp = safe_api_call("positions", client.get_positions, limit=200) or {}
    except ServicePaused as e:
        logger.warning(f"Positions API paused: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error fetching positions: {repr(e)}")
        return {}

    # Try multiple possible response formats
    items = None
    for key in ["market_positions", "positions", "portfolio_positions"]:
        items = resp.get(key)
        if isinstance(items, list):
            break
    
    # Fallback: try to find any list in response
    if items is None:
        for v in resp.values():
            if isinstance(v, list) and len(v) > 0:
                items = v
                break
    
    if not isinstance(items, list):
        logger.warning("Could not parse positions response")
        return {}

    out = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        
        # Try multiple possible ticker keys
        t = None
        for key in ["ticker", "market_ticker", "contract_ticker"]:
            t = it.get(key)
            if t:
                break
        
        if not t:
            continue
        
        # Try multiple possible quantity keys
        qty = None
        for key in ["position", "qty", "quantity", "shares"]:
            qty = it.get(key)
            if qty is not None:
                break
        
        try:
            qty = int(qty)
        except (ValueError, TypeError):
            continue
        
        if qty == 0:
            continue

        # Try multiple possible cost keys
        total = None
        for key in ["total_traded", "market_exposure", "total_cost"]:
            total = it.get(key)
            if total is not None:
                break
        
        try:
            total = float(total) if total is not None else None
        except (ValueError, TypeError):
            total = None

        avg = 0.0
        if total is not None and qty != 0:
            avg = total / float(abs(qty))

        out[t] = (qty, float(avg))

    return out


def reconcile_positions_to_db(conn, live_pos: dict):
    if not isinstance(live_pos, dict):
        return
    for t, (q, ap) in live_pos.items():
        try:
            upsert_position(conn, t, int(q), float(ap))
        except Exception as e:
            logger.error(f"Error reconciling position {t}: {repr(e)}")


def get_position_effective(conn, ticker: str, live_pos: dict | None):
    if live_pos and ticker in live_pos:
        q, ap = live_pos[ticker]
        return int(q), float(ap)
    return get_position(conn, ticker)


def clamp01(p):
    return max(0.0, min(1.0, p))


def normal_cdf(x, mu, sigma):
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def prob_ge_normal(x, mu, sigma):
    if sigma <= 0:
        return 1.0 if mu >= x else 0.0
    return clamp01(1.0 - normal_cdf(x, mu, sigma))


def prob_le_normal(x, mu, sigma):
    if sigma <= 0:
        return 1.0 if mu <= x else 0.0
    return clamp01(normal_cdf(x, mu, sigma))


def prob_ge_empirical(x, highs: list[float], smooth_sigma: float):
    if not highs or len(highs) < EMPIRICAL_MIN_MEMBERS:
        return None
    s = smooth_sigma if smooth_sigma > 0 else 0.9
    return clamp01(sum(prob_ge_normal(x, float(h), s) for h in highs) / len(highs))


def prob_le_empirical(x, highs: list[float], smooth_sigma: float):
    if not highs or len(highs) < EMPIRICAL_MIN_MEMBERS:
        return None
    s = smooth_sigma if smooth_sigma > 0 else 0.9
    return clamp01(sum(prob_le_normal(x, float(h), s) for h in highs) / len(highs))


def make_client_order_id(ticker: str) -> str:
    raw = f"bot-{ticker}-{time.time_ns()}"
    raw = re.sub(r"[^A-Za-z0-9_-]", "_", raw)
    return raw[:80]


# ===== REGIME DETECTION & DYNAMIC SIZING FUNCTIONS =====

def volatility_regime_for_sigma(sigma: float):
    """Classify volatility regime. Returns 'low', 'normal', or 'high'."""
    if sigma < VOL_LOW_THRESH:
        return "low"
    elif sigma > VOL_HIGH_THRESH:
        return "high"
    else:
        return "normal"


def compute_dynamic_position_multiplier(sigma: float):
    """Scale position size based on volatility. Lower vol = larger positions allowed."""
    if not DYNAMIC_POSITION_SIZING:
        return 1.0
    
    regime = volatility_regime_for_sigma(sigma)
    
    if regime == "low":
        return VOL_POSITION_MULT_LOW
    elif regime == "high":
        return VOL_POSITION_MULT_HIGH
    else:
        # Linear interpolation between low and high
        if VOL_HIGH_THRESH <= VOL_LOW_THRESH:
            return 1.0
        frac = (sigma - VOL_LOW_THRESH) / (VOL_HIGH_THRESH - VOL_LOW_THRESH)
        return VOL_POSITION_MULT_LOW - frac * (VOL_POSITION_MULT_LOW - VOL_POSITION_MULT_HIGH)


def detect_market_regime(conn, now_dt: datetime):
    """Detect market stress: volatility spikes, liquidity drops, price velocity.
    
    Returns dict: {'throttle': bool, 'reason': str, 'severity': float}
    """
    if not REGIME_DETECTION_ENABLED:
        return {"throttle": False, "reason": None, "severity": 0.0}
    
    try:
        # Check recent quote variance (proxy for vol spike)
        cutoff = (now_dt - timedelta(minutes=REGIME_CHECK_WINDOW_MINUTES)).isoformat()
        rows = conn.execute("""
            SELECT mid_yes FROM quotes 
            WHERE ts_utc > ? AND mid_yes IS NOT NULL
            ORDER BY ts_utc DESC 
            LIMIT 100
        """, (cutoff,)).fetchall()
        
        if len(rows) > 5:
            mids = [float(r[0]) for r in rows if r[0] is not None]
            if len(mids) > 1:
                mean_mid = sum(mids) / len(mids)
                variance = sum((x - mean_mid) ** 2 for x in mids) / len(mids)
                std_mid = math.sqrt(variance) if variance > 0 else 0.1
                
                # Check for unusual price velocity (rapid moves)
                if len(mids) >= 2:
                    velocity = abs(mids[0] - mids[-1]) / max(1, len(mids) - 1)
                    if velocity > PRICE_VELOCITY_THRESHOLD:
                        return {
                            "throttle": True,
                            "reason": f"PRICE_VELOCITY {velocity:.2f} cents/quote",
                            "severity": min(1.0, velocity / PRICE_VELOCITY_THRESHOLD)
                        }
        
        return {"throttle": False, "reason": None, "severity": 0.0}
    except Exception as e:
        logger.warning(f"Regime detection error: {e}")
        return {"throttle": False, "reason": None, "severity": 0.0}


def compute_series_notional_exposure(live_pos: dict, series: str):
    """Compute total notional $ exposure for all tickers in a series.
    
    Requires tickers to be traceable to series (ticker format: SERIESXXXX)
    """
    total = 0.0
    for ticker, (qty, price) in live_pos.items():
        # Heuristic: ticker usually contains series name
        if series.upper() in ticker.upper():
            pos_value = abs(int(qty)) * float(price) / 100.0  # Convert to mid price equivalent
            total += pos_value
    return total


def compute_event_notional_exposure(live_pos: dict, event_ticker: str):
    """Compute total notional $ exposure for a specific event.
    
    Event ticker is specific, sum all positions in that event's markets.
    (Requires markets list or cross-reference)
    """
    # For position-only tracking, we approximate by ticker prefix matching
    # In practice, would need market->event mapping from API
    total = 0.0
    for ticker, (qty, price) in live_pos.items():
        # All markets for an event share event prefix (e.g., KXHIGHNY-26JAN15-##)
        if event_ticker in ticker:
            pos_value = abs(int(qty)) * float(price) / 100.0
            total += pos_value
    return total


def check_concentration_limits(conn, live_pos: dict, series: str, event_ticker: str, 
                               dry_run_qty: int, dry_run_price: float, direction: str):
    """Check if adding position would violate concentration limits.
    
    Returns: (allowed: bool, reason: str or None)
    """
    if not CONCENTRATION_LIMITS_ENABLED:
        return True, None
    
    try:
        # Current exposures
        series_exposure = compute_series_notional_exposure(live_pos, series)
        event_exposure = compute_event_notional_exposure(live_pos, event_ticker)
        
        # Compute total contracts in series
        series_qty = sum(abs(int(q)) for ticker, (q, p) in live_pos.items() 
                        if series.upper() in ticker.upper())
        
        # Dry run: add proposed position
        dry_notional = abs(int(dry_run_qty)) * float(dry_run_price) / 100.0
        
        new_series_exposure = series_exposure + dry_notional
        new_event_exposure = event_exposure + dry_notional
        new_series_qty = series_qty + abs(int(dry_run_qty))
        
        # Check limits
        if new_series_exposure > MAX_NOTIONAL_PER_SERIES:
            return False, f"Series ${new_series_exposure:.0f} > limit ${MAX_NOTIONAL_PER_SERIES:.0f}"
        
        if new_event_exposure > MAX_NOTIONAL_PER_EVENT:
            return False, f"Event ${new_event_exposure:.0f} > limit ${MAX_NOTIONAL_PER_EVENT:.0f}"
        
        if new_series_qty > MAX_CONTRACTS_PER_SERIES:
            return False, f"Series qty {new_series_qty} > limit {MAX_CONTRACTS_PER_SERIES}"
        
        return True, None
    except Exception as e:
        logger.warning(f"Concentration check error: {e}")
        return True, None  # Fail open


def fair_prob_market(m: dict, mu: float, sigma: float, highs_members: list[float] | None, market_ticker: str = ""):
    st = (m.get("strike_type") or "").lower()
    floor_ = m.get("floor_strike")
    cap_ = m.get("cap_strike")
    
    # Apply skew adjustment to mu based on ensemble members
    mu_adjusted = compute_forecast_skew(highs_members, mu)
    method_used = "unknown"

    if st == "between" and floor_ is not None and cap_ is not None:
        lo = float(floor_)
        hi = float(cap_)
        pe = prob_between_empirical(lo, hi, highs_members or [], EMPIRICAL_SMOOTH_SIGMA)
        if pe is not None:
            if LOG_FORECAST_DIAGNOSTICS:
                logger.info(f"FORECAST_METHOD {market_ticker} between=[{lo:.0f},{hi:.0f}] method=empirical p={pe:.3f} members={len(highs_members) if highs_members else 0}")
            return pe
        
        # Apply volatility smile and Student's t
        sigma_low = adjusted_sigma_for_strike(sigma, lo, mu_adjusted)
        sigma_hi = adjusted_sigma_for_strike(sigma, hi, mu_adjusted)
        sigma_avg = (sigma_low + sigma_hi) / 2.0
        
        # Try Student's t first
        prob_t = prob_between_t_dist(lo, hi, mu_adjusted, sigma_avg)
        if prob_t is not None:
            if LOG_FORECAST_DIAGNOSTICS:
                logger.info(f"FORECAST_METHOD {market_ticker} between=[{lo:.0f},{hi:.0f}] method=students_t p={prob_t:.3f} sigma={sigma_avg:.2f}")
            return prob_t
        
        prob_normal = clamp01(prob_between_normal(lo, hi, mu_adjusted, sigma_avg))
        if LOG_FORECAST_DIAGNOSTICS:
            logger.info(f"FORECAST_METHOD {market_ticker} between=[{lo:.0f},{hi:.0f}] method=normal p={prob_normal:.3f} sigma={sigma_avg:.2f}")
        return prob_normal

    if st in ("greater", "greater_than", "greater_or_equal", "above") and floor_ is not None:
        x = float(floor_)
        pe = prob_ge_empirical(x, highs_members or [], EMPIRICAL_SMOOTH_SIGMA)
        if pe is not None:
            if LOG_FORECAST_DIAGNOSTICS:
                logger.info(f"FORECAST_METHOD {market_ticker} greater={x:.0f} method=empirical p={pe:.3f} members={len(highs_members) if highs_members else 0}")
            return pe
        
        # Apply volatility smile
        sigma_adj = adjusted_sigma_for_strike(sigma, x, mu_adjusted)
        
        # Try Student's t first
        prob_t = prob_with_students_t(x, mu_adjusted, sigma_adj, direction="greater")
        if prob_t is not None:
            if LOG_FORECAST_DIAGNOSTICS:
                logger.info(f"FORECAST_METHOD {market_ticker} greater={x:.0f} method=students_t p={prob_t:.3f} sigma={sigma_adj:.2f}")
            return prob_t
        
        prob_normal = prob_ge_normal(x, mu_adjusted, sigma_adj)
        if LOG_FORECAST_DIAGNOSTICS:
            logger.info(f"FORECAST_METHOD {market_ticker} greater={x:.0f} method=normal p={prob_normal:.3f} sigma={sigma_adj:.2f}")
        return prob_normal

    if st in ("less", "less_than", "less_or_equal", "below") and cap_ is not None:
        x = float(cap_)
        pe = prob_le_empirical(x, highs_members or [], EMPIRICAL_SMOOTH_SIGMA)
        if pe is not None:
            if LOG_FORECAST_DIAGNOSTICS:
                logger.info(f"FORECAST_METHOD {market_ticker} less={x:.0f} method=empirical p={pe:.3f} members={len(highs_members) if highs_members else 0}")
            return pe
        
        # Apply volatility smile
        sigma_adj = adjusted_sigma_for_strike(sigma, x, mu_adjusted)
        
        # Try Student's t first
        prob_t = prob_with_students_t(x, mu_adjusted, sigma_adj, direction="less")
        if prob_t is not None:
            if LOG_FORECAST_DIAGNOSTICS:
                logger.info(f"FORECAST_METHOD {market_ticker} less={x:.0f} method=students_t p={prob_t:.3f} sigma={sigma_adj:.2f}")
            return prob_t
        
        prob_normal = prob_le_normal(x, mu_adjusted, sigma_adj)
        if LOG_FORECAST_DIAGNOSTICS:
            logger.info(f"FORECAST_METHOD {market_ticker} less={x:.0f} method=normal p={prob_normal:.3f} sigma={sigma_adj:.2f}")
        return prob_normal

    return None


def fair_prob_market_with_calibration(conn, ts_utc: str, m: dict, mu: float, sigma: float,
                                       highs_members: list[float] | None, market_ticker: str = "",
                                       series: str = "", event_ticker: str = ""):
    """Call fair_prob_market and record the prediction for calibration tracking (if enabled)."""
    p = fair_prob_market(m, mu, sigma, highs_members, market_ticker)
    
    if p is None:
        return None
    
    if CALIBRATION_TRACKING_ENABLED:
        try:
            st = (m.get("strike_type") or "").lower()
            floor_ = m.get("floor_strike")
            cap_ = m.get("cap_strike")
            
            # Determine forecast method by trying the same logic as fair_prob_market
            method = "unknown"
            if st == "between" and floor_ is not None and cap_ is not None:
                lo, hi = float(floor_), float(cap_)
                if prob_between_empirical(lo, hi, highs_members or [], EMPIRICAL_SMOOTH_SIGMA) is not None:
                    method = "empirical"
                elif prob_between_t_dist(lo, hi, mu, sigma) is not None:
                    method = "students_t"
                else:
                    method = "normal"
            elif st in ("greater", "greater_than", "greater_or_equal", "above") and floor_ is not None:
                x = float(floor_)
                if prob_ge_empirical(x, highs_members or [], EMPIRICAL_SMOOTH_SIGMA) is not None:
                    method = "empirical"
                elif prob_with_students_t(x, mu, sigma, direction="greater") is not None:
                    method = "students_t"
                else:
                    method = "normal"
            elif st in ("less", "less_than", "less_or_equal", "below") and cap_ is not None:
                x = float(cap_)
                if prob_le_empirical(x, highs_members or [], EMPIRICAL_SMOOTH_SIGMA) is not None:
                    method = "empirical"
                elif prob_with_students_t(x, mu, sigma, direction="less") is not None:
                    method = "students_t"
                else:
                    method = "normal"
            
            record_forecast_prediction(
                conn, ts_utc, market_ticker, series, event_ticker,
                st, floor_, cap_, p, method,
                num_members=len(highs_members) if highs_members else None,
                mu=mu, sigma=sigma
            )
        except Exception as e:
            logger.warning(f"Calibration record error for {market_ticker}: {e}")
    
    return p


def date_from_event_ticker(event_ticker: str):
    """Parse date from event ticker with improved error handling."""
    if not isinstance(event_ticker, str):
        return None
    
    parts = event_ticker.split("-")
    if len(parts) < 2:
        return None
    
    code = parts[1]
    
    # Try standard format: YYMONDD (e.g., 24JAN15)
    if len(code) == 7:
        try:
            yy = int(code[0:2])
            mon = code[2:5].upper()
            dd = int(code[5:7])
            
            month_map = {
                "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
            }
            
            if mon not in month_map:
                return None
            
            year = 2000 + yy
            month = month_map[mon]
            return f"{year:04d}-{month:02d}-{dd:02d}"
        except (ValueError, IndexError):
            pass
    
    return None


def baseline_sigma_for(series: str, target_date: str):
    try:
        month = int(target_date[5:7])
    except Exception:
        return 4.0
    return float(BASE_SIGMA_BY_SERIES_MONTH.get(series, {}).get(month, 4.0))


def min_edge_for_time(series: str, now_utc: datetime):
    tzname = SERIES_TZ.get(series, "UTC")
    try:
        local = now_utc.astimezone(ZoneInfo(tzname))
        h = local.hour
    except Exception:
        h = now_utc.hour
    mult = 1.0
    if 9 <= h < 17:
        mult = 1.25
    return float(MIN_EDGE_CENTS) * mult, mult


def target_for_edge(edge: float, position_mult: float = 1.0):
    """Calculate target position size for edge, adjusted by volatility regime multiplier."""
    t = BASE_TARGET
    if edge >= 10:
        t = min(50, MAX_TARGET)
    if edge >= 25:
        t = min(75, MAX_TARGET)
    
    # Apply dynamic sizing multiplier
    adjusted = int(round(float(t) * position_mult))
    return max(BASE_TARGET // 2, adjusted)  # Don't shrink below half BASE_TARGET


def balance_aware_target(bal_cents: int, base_target: int = None, max_target: int = None):
    """Scale target position size based on available balance.
    
    If balance < MAX_TOTAL_RISK_DOLLARS, scale down positions proportionally.
    Available balance = bal_cents - MIN_BALANCE_CENTS
    Scale factor = available / (MAX_TOTAL_RISK_DOLLARS * 100)
    
    Examples:
        - balance=$20, MIN_BALANCE_CENTS=500, MAX_TOTAL_RISK_DOLLARS=60 → scale=0.33x
        - balance=$60, MIN_BALANCE_CENTS=500, MAX_TOTAL_RISK_DOLLARS=60 → scale=1.0x (no change)
    """
    if base_target is None:
        base_target = BASE_TARGET
    if max_target is None:
        max_target = MAX_TARGET
    
    available_cents = max(0, bal_cents - MIN_BALANCE_CENTS)
    max_risk_cents = MAX_TOTAL_RISK_DOLLARS * 100.0
    
    if available_cents < max_risk_cents:
        scale = available_cents / max_risk_cents
        scaled_base = max(1, int(round(base_target * scale)))
        scaled_max = max(1, int(round(max_target * scale)))
        return scaled_base, scaled_max, scale
    else:
        # Plenty of balance, use normal targets
        return base_target, max_target, 1.0


def per_contract_worst_loss_dollars(direction: str, px_cents: int):
    if direction == "BUY":
        return float(px_cents) / 100.0
    return (100.0 - float(px_cents)) / 100.0


def compute_current_risk_dollars(conn, positions: dict = None, source: str = "db"):
    """Compute risk from positions. 
    
    Args:
        conn: Database connection
        positions: Optional dict of {ticker: (qty, avg_price)}. If provided, uses these instead of DB.
        source: 'db' reads from database, 'live' uses provided positions dict, 'both' combines them
    
    Returns:
        float: Risk in dollars
    """
    risk = 0.0
    
    if source == "both" and positions:
        # Merge live positions with DB, preferring live
        db_rows = conn.execute("SELECT ticker, qty, avg_price FROM positions WHERE qty != 0").fetchall() or []
        combined = {}
        for ticker, qty, avg in db_rows:
            combined[ticker] = (int(qty), float(avg))
        combined.update(positions)
        items = combined.items()
    elif source == "live" and positions:
        items = positions.items()
    else:
        # Read from DB
        rows = conn.execute("SELECT qty, avg_price FROM positions WHERE qty != 0").fetchall() or []
        items = [(f"ticker_{i}", (int(qty), float(avg))) for i, (qty, avg) in enumerate(rows)]
    
    for ticker, (q, a) in items:
        if q > 0:
            risk += q * (a / 100.0)
        elif q < 0:
            risk += (-q) * ((100.0 - a) / 100.0)
    
    return float(risk)


def compute_risk_with_pending_orders(conn, live_pos: dict, pending_orders: dict):
    """Compute total risk including pending orders.
    
    Args:
        conn: Database connection
        live_pos: Dict of {ticker: (qty, avg_price)} from live API
        pending_orders: Dict of {ticker: (direction, qty, px)} or
                        {ticker: (direction, qty, px, accounted)} for unfilled/being-placed orders.
                        The optional `accounted` boolean indicates whether the pending
                        order's risk has already been applied to `current_risk`.
    
    Returns:
        tuple: (total_risk_dollars, position_risk_dollars, pending_risk_dollars)
    """
    # Risk from actual positions
    position_risk = compute_current_risk_dollars(conn, live_pos, source="live")
    
    # Risk from pending orders
    pending_risk = 0.0
    for ticker, payload in pending_orders.items():
        # support both legacy (direction, qty, px) and new (direction, qty, px, accounted)
        if len(payload) == 4:
            direction, qty, px, accounted = payload
        else:
            direction, qty, px = payload
            accounted = False
        # Only include pending orders that have NOT already been applied to current risk
        if accounted:
            continue
        if direction == "BUY":
            pending_risk += qty * per_contract_worst_loss_dollars("BUY", int(px))
        # Do not add SELL orders to pending_risk. A pending sell will
        # reduce exposure if it fills, but until filled the live positions
        # represent current exposure; including SELL here created incorrect
        # increases in reported risk. If desired, we could subtract an
        # estimated reduction, but for safety we omit SELL from pending risk.
    
    return position_risk + pending_risk, position_risk, pending_risk


def adjusted_sigma_for_strike(sigma: float, strike: float, mu: float, atm_vol: float = None):
    """Apply volatility smile: adjust sigma based on strike distance from ATM.
    
    Implements a simple volatility smile where OTM and ITM options have higher vol.
    Formula: sigma_adj = sigma * (1 + smile_factor * ((strike - mu) / mu)^2)
    """
    if not VOLATILITY_SMILE_ENABLED or sigma <= 0:
        return sigma
    
    try:
        # Distance from ATM as fraction
        distance = abs(float(strike) - float(mu)) / max(1, abs(float(mu)))
        # Quadratic smile: vol increases as we move away from ATM
        smile_multiplier = 1.0 + float(VOLATILITY_SMILE_FACTOR) * (distance ** 2)
        return float(sigma) * smile_multiplier
    except Exception:
        return sigma


def prob_with_students_t(x: float, mu: float, sigma: float, df: float = None, direction: str = "greater"):
    """Compute probability using Student's t-distribution instead of normal.
    
    Args:
        x: Strike price
        mu: Mean
        sigma: Standard deviation
        df: Degrees of freedom (lower = fatter tails)
        direction: 'greater', 'less'
    
    Returns:
        Probability (0-1) or None to fall back to normal
    """
    if not USE_STUDENTS_T:
        return None  # Use normal distribution
    
    if df is None:
        df = STUDENTS_T_DF
    
    try:
        from scipy import stats
        # Standardize
        z = (x - mu) / (sigma * math.sqrt(2.0)) if sigma > 0 else 0
        
        if direction == "greater":
            return clamp01(1.0 - stats.t.cdf(z, df=df))
        elif direction == "less":
            return clamp01(stats.t.cdf(z, df=df))
        else:
            return None
    except ImportError:
        # scipy not available, fall back to normal
        return None
    except Exception:
        return None


def prob_between_t_dist(lo: float, hi: float, mu: float, sigma: float, df: float = None):
    """Compute P(lo <= X <= hi) using Student's t-distribution.
    
    Returns None if scipy not available or Student's t not enabled.
    """
    if not USE_STUDENTS_T:
        return None
    
    if df is None:
        df = STUDENTS_T_DF
    
    try:
        from scipy import stats
        # Standardize
        z_lo = (lo - mu) / (sigma * math.sqrt(2.0)) if sigma > 0 else 0
        z_hi = (hi - mu) / (sigma * math.sqrt(2.0)) if sigma > 0 else 0
        
        cdf_lo = stats.t.cdf(z_lo, df=df)
        cdf_hi = stats.t.cdf(z_hi, df=df)
        
        return clamp01(cdf_hi - cdf_lo)
    except ImportError:
        return None
    except Exception:
        return None


def compute_forecast_skew(highs_members: list, mu: float):
    """Compute skewness from ensemble members and adjust forecast accordingly.
    
    Returns adjusted mu that accounts for forecast skew.
    """
    if not FORECAST_SKEW_ENABLED or not highs_members or len(highs_members) < 3:
        return mu
    
    try:
        # Compute sample skewness
        vals = [float(h) for h in highs_members]
        mean = sum(vals) / len(vals)
        variance = sum((x - mean) ** 2 for x in vals) / max(1, len(vals) - 1)
        std = math.sqrt(variance) if variance > 0 else 1.0
        
        # Fisher-Pearson skewness
        skewness = sum((x - mean) ** 3 for x in vals) / (len(vals) * (std ** 3)) if std > 0 else 0
        
        # Adjust mu by a fraction of the skewness
        # Positive skew = right tail, so true mean is lower than sample mean
        skew_adjustment = skewness * float(FORECAST_SKEW_FACTOR) * std
        adjusted_mu = float(mu) - skew_adjustment
        
        if abs(skew_adjustment) > 0.1:  # Log if significant
            logger.info(f"SKEW_ADJUSTMENT skewness={skewness:.3f} adjustment={skew_adjustment:.2f} mu={mu:.2f}->{adjusted_mu:.2f}")
        
        return adjusted_mu
    except Exception:
        return mu


def pick_best_event_ticker(client: KalshiClient, series: str, now_utc: datetime):
    """Pick the best event ticker with improved error handling."""
    cursor = None
    best_dt = None
    best_et = None
    fallback = None
    for _ in range(10):
        try:
            resp = safe_api_call("events", client.get_events, limit=200, cursor=cursor, series_ticker=series)
        except ServicePaused as e:
            logger.warning(f"Events API paused: {e}")
            break
        except Exception as e:
            logger.error(f"Error fetching events for {series}: {repr(e)}")
            break
        
        events = resp.get("events", []) or []
        for e in events:
            et = e.get("event_ticker") or e.get("ticker")
            if not et:
                continue
            if fallback is None:
                fallback = et
            ct = parse_iso(e.get("close_time") or e.get("event_close_time"))
            if not ct:
                continue
            if ct <= now_utc:
                continue
            if best_dt is None or ct < best_dt:
                best_dt = ct
                best_et = et
        cursor = resp.get("cursor")
        if not cursor:
            break

    return best_et or fallback


def cancel_stale_orders(client: KalshiClient, conn, now_dt: datetime, ts: str):
    if not (LIVE_TRADING and CANCEL_STALE_ORDERS and STALE_ORDER_SECONDS > 0):
        return 0

    canceled = 0
    cursor = None
    max_pages = 10

    for page in range(max_pages):
        try:
            resp = safe_api_call("orders", client.get_orders, limit=200, cursor=cursor)  # no status filter
        except Exception as e:
            logger.error(f"Error fetching orders: {repr(e)}")
            break

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
                    count=int(o.get("count") or o.get("initial_count") or 0),
                    order_id=oid,
                    client_order_id=o.get("client_order_id"),
                    status="canceled",
                    raw_json=str(o),
                )
            except Exception as e:
                logger.error(f"Error canceling order {oid}: {repr(e)}")
                continue

        cursor = resp.get("cursor")
        if not cursor:
            break

    return canceled


def amend_stale_orders(client: KalshiClient, conn, now_dt: datetime, ts: str, 
                       latest_orderbook: dict, current_prices: dict):
    """
    Instead of canceling stale orders, try to amend their prices to current market.
    This is smarter than cancel+relist as it preserves queue position and reduces slippage.
    
    Args:
        client: Kalshi API client
        conn: DB connection
        now_dt: current datetime
        ts: ISO timestamp
        latest_orderbook: dict mapping ticker -> orderbook snapshot
        current_prices: dict mapping ticker -> current mid price
    
    Returns: tuple (amended_count, failed_count)
    """
    if not LIVE_TRADING:
        return 0, 0

    amended = 0
    failed = 0
    cursor = None
    max_pages = 10

    for page in range(max_pages):
        try:
            resp = safe_api_call("orders", client.get_orders, limit=200, cursor=cursor)
        except Exception as e:
            logger.error(f"Error fetching orders for amend: {repr(e)}")
            break

        orders = resp.get("orders", []) or []
        if not orders:
            break

        for o in orders:
            st = (o.get("status") or "").lower()
            if st not in ("open", "resting"):
                continue

            oid = o.get("order_id") or o.get("id")
            ticker = o.get("ticker") or ""
            if not oid or not ticker:
                continue

            created = parse_iso(o.get("created_time"))
            if not created:
                continue

            age = (now_dt - created).total_seconds()
            
            # Only amend if old enough AND amending makes sense
            if age < STALE_ORDER_SECONDS * 0.5:
                # Not stale yet, skip
                continue
            
            # Check if price is far from current market
            current_price = current_prices.get(ticker)
            old_price = o.get("yes_price") or o.get("price") or 0.0
            
            if current_price is None or abs(current_price - old_price) < 0.01:
                # Not worth amending, skip
                continue
            
            try:
                # Try to amend to current market price
                new_price = round(current_price, 4)
                direction = (o.get("side") or "").lower()
                qty = int(o.get("count") or o.get("initial_count") or 0)
                
                # Kalshi API: try to amend order
                # Note: not all order types support amendment; if it fails, fall back to cancel
                if hasattr(client, 'amend_order'):
                    try:
                        client.amend_order(oid, yes_price=new_price)
                        amended += 1
                        log_live_order(
                            conn,
                            ts,
                            ticker=ticker,
                            side=direction,
                            action="amend",
                            yes_price=new_price,
                            count=qty,
                            order_id=oid,
                            client_order_id=o.get("client_order_id"),
                            status="amended",
                            raw_json=f"{{'old_price': {old_price}, 'new_price': {new_price}}}",
                        )
                    except Exception as amend_err:
                        # Fall back to cancel if amend not supported
                        logger.debug(f"Amend failed for {oid} ({amend_err}), falling back to cancel")
                        client.cancel_order(oid)
                        failed += 1
                        log_live_order(
                            conn,
                            ts,
                            ticker=ticker,
                            side=direction,
                            action="cancel",
                            yes_price=old_price,
                            count=qty,
                            order_id=oid,
                            client_order_id=o.get("client_order_id"),
                            status="canceled",
                            raw_json=str(o),
                        )
                else:
                    # API doesn't support amend, fall back to cancel
                    logger.debug(f"Amend not supported for {oid}, falling back to cancel")
                    failed += 1
                    
            except Exception as e:
                logger.error(f"Error amending/canceling order {oid}: {repr(e)}")
                failed += 1
                continue

        cursor = resp.get("cursor")
        if not cursor:
            break

    return amended, failed



def edge_persistence_ok(conn, series: str, ticker: str, direction: str, edge: float, ts: str, seconds_needed: int):
    key = f"persist:{series}:{ticker}:{direction}"
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


def clear_persistence(conn, series: str, ticker: str, direction: str):
    set_state(conn, f"persist:{series}:{ticker}:{direction}", "")


def track_order_attempt(conn, series: str, ticker: str, direction: str, ts: str):
    """Track attempt to place an order for monitoring fill rates."""
    key = f"order_attempt:{series}:{ticker}:{direction}"
    set_state(conn, key, ts)


def get_order_attempt_age(conn, series: str, ticker: str, direction: str, now_ts: str):
    """Get age (seconds) of last order attempt for this direction on ticker."""
    key = f"order_attempt:{series}:{ticker}:{direction}"
    raw = get_state(conn, key, "")
    if not raw:
        return None
    try:
        last_ts = parse_iso(raw)
        now_dt = parse_iso(now_ts)
        if last_ts and now_dt:
            return int((now_dt - last_ts).total_seconds())
    except Exception:
        pass
    return None


def should_skip_order_due_to_fill_rate(conn, series: str, ticker: str, direction: str, now_ts: str):
    """Skip order if recent attempt didn't result in acceptable fill rate."""
    age = get_order_attempt_age(conn, series, ticker, direction, now_ts)
    if age is None or age > ORDER_FILL_TIMEOUT_SECONDS:
        return False
    
    # Check fill rate from last Trade table entry for this ticker/direction
    rows = conn.execute(
        "SELECT fill_count FROM live_orders WHERE ticker=? AND action=? ORDER BY id DESC LIMIT 1",
        (ticker, direction.lower()),
    ).fetchall()
    
    if not rows:
        return False
    
    # If we're still within timeout and previous fill was poor, skip
    fill_count = rows[0][0] if rows[0] else 0
    if fill_count == 0 and age < ORDER_FILL_TIMEOUT_SECONDS * 0.5:
        return True  # Skip if no fill yet and still in early timeout
    
    return False


def check_orderbook_freshness(ob: dict, now_dt: datetime, max_age_sec: float = MAX_ORDERBOOK_AGE_SECONDS):
    """Check if orderbook data is fresh. Returns (is_fresh, age_seconds, reason)."""
    if not isinstance(ob, dict):
        return False, None, "orderbook_not_dict"
    
    # Try multiple possible timestamp field names
    ts_raw = None
    for ts_key in ["timestamp", "ts", "updated_at", "last_update", "updated", "time"]:
        ts_raw = ob.get(ts_key)
        if ts_raw:
            break
    
    if not ts_raw:
        # Many orderbook payloads omit a top-level timestamp. Treat missing
        # timestamp as "unknown/fresh" rather than stale to avoid skipping
        # valid markets. Return a distinct reason so callers can log if needed.
        if WARN_STALE_DATA:
            logger.warning("ORDERBOOK missing timestamp; treating as unknown/fresh")
        return True, None, "no_timestamp"
    
    try:
        ts_dt = parse_iso(ts_raw) if isinstance(ts_raw, str) else None
        if not ts_dt:
            try:
                ts_dt = datetime.fromtimestamp(float(ts_raw), tz=timezone.utc)
            except Exception:
                if REQUIRE_FRESH_ORDERBOOK:
                    return False, None, "invalid_timestamp_format"
                return True, None, "unparseable_timestamp_but_not_required"
        
        age_sec = (now_dt - ts_dt).total_seconds()
        if age_sec < 0:
            return False, age_sec, "future_timestamp"
        
        is_fresh = age_sec <= max_age_sec
        if not is_fresh and WARN_STALE_DATA:
            logger.warning(f"STALE_ORDERBOOK age={age_sec:.1f}s max={max_age_sec}s")
        
        return is_fresh, age_sec, "ok" if is_fresh else "too_old"
    except Exception as e:
        if REQUIRE_FRESH_ORDERBOOK:
            return False, None, f"timestamp_check_error: {repr(e)}"
        return True, None, f"timestamp_check_error_but_not_required: {repr(e)}"


def check_forecast_freshness(snapshot_ts: str, now_dt: datetime, max_age_sec: float = MAX_FORECAST_AGE_SECONDS):
    """Check if forecast data is fresh. Returns (is_fresh, age_seconds)."""
    try:
        snap_dt = parse_iso(snapshot_ts)
        if not snap_dt:
            if REQUIRE_FRESH_FORECAST:
                return False, None
            return True, None
        
        age_sec = (now_dt - snap_dt).total_seconds()
        if age_sec < 0:
            return False, age_sec
        
        is_fresh = age_sec <= max_age_sec
        if not is_fresh and WARN_STALE_DATA:
            logger.warning(f"STALE_FORECAST age={age_sec:.1f}s max={max_age_sec}s")
        
        return is_fresh, age_sec
    except Exception:
        if REQUIRE_FRESH_FORECAST:
            return False, None
        return True, None


def log_data_freshness_report(conn, client: KalshiClient, now_dt: datetime, series_list: list):
    """Log diagnostics about data freshness across the system."""
    logger.info("=== DATA_FRESHNESS_REPORT ===")
    logger.info(f"Now UTC: {now_dt.isoformat()}")
    
    # Check a sample orderbook
    if series_list:
        series = series_list[0]
        try:
            resp = safe_api_call("events", client.get_events, limit=1, series_ticker=series)
            events = resp.get("events", []) or []
            if events and events[0].get("event_ticker"):
                et = events[0].get("event_ticker")
                try:
                    markets_resp = safe_api_call("markets", client.get_markets, series_ticker=series, limit=5)
                    markets = markets_resp.get("markets", []) or []
                    if markets and markets[0].get("ticker"):
                        ticker = markets[0].get("ticker")
                        try:
                            ob = safe_api_call("orderbook", client.get_orderbook, ticker)
                            is_fresh, age, reason = check_orderbook_freshness(ob, now_dt)
                            logger.info(f"Sample OB {ticker}: fresh={is_fresh} age={age} reason={reason}")
                        except ServicePaused:
                            logger.info(f"Sample OB {ticker}: orderbook service paused")
                        except Exception as e:
                            logger.info(f"Could not fetch sample orderbook {ticker}: {repr(e)}")
                except Exception:
                    # ignore sample check failures
                    pass
        except Exception as e:
            logger.warning(f"Could not check sample orderbook: {repr(e)}")
    
    # Check forecast age from DB
    rows = conn.execute(
        "SELECT ts_utc, series, target_date FROM forecast_snapshots ORDER BY id DESC LIMIT 1"
    ).fetchall()
    if rows:
        ts_utc, series, target_date = rows[0]
        is_fresh, age = check_forecast_freshness(ts_utc, now_dt)
        logger.info(f"Latest forecast: age={age:.0f}s fresh={is_fresh} series={series} target={target_date}")
    
    logger.info("=== END DATA_FRESHNESS_REPORT ===")


def ticker_prefix(ticker: str) -> str:
    if not isinstance(ticker, str):
        return ""
    return ticker.split("-", 1)[0]


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


def get_orderbook_depth(ob: dict, direction: str, target_px: int, levels: int = 3):
    """Get cumulative depth near target price. direction='BUY'->need ask side, direction='SELL'->need bid side."""
    try:
        book = ob.get("orderbook") if isinstance(ob, dict) and isinstance(ob.get("orderbook"), dict) else ob
        if not isinstance(book, dict):
            return 0
        
        if direction == "BUY":
            # Check no_bid side (inverse of yes_ask)
            no_levels = book.get("no") or []
            if not isinstance(no_levels, list):
                return 0
            depth = 0
            count = 0
            for lvl in no_levels:
                if isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                    try:
                        px = int(lvl[0])
                        sz = int(lvl[1])
                        # For a BUY, we look at no_bid which translates to yes_ask
                        yes_ask_equiv = 100 - px
                        if yes_ask_equiv <= target_px and count < levels:
                            depth += sz
                            count += 1
                    except Exception:
                        pass
            return depth
        else:  # SELL
            # Check yes_bid side
            yes_levels = book.get("yes") or []
            if not isinstance(yes_levels, list):
                return 0
            depth = 0
            count = 0
            for lvl in yes_levels:
                if isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                    try:
                        px = int(lvl[0])
                        sz = int(lvl[1])
                        if px >= target_px and count < levels:
                            depth += sz
                            count += 1
                    except Exception:
                        pass
            return depth
    except Exception:
        return 0


def build_exit_candidates_from_positions(client: KalshiClient, conn, ts: str, live_pos: dict, now_dt: datetime = None):
    """Build exit candidates with enhanced logic: stop-loss, take-profit, scale-outs, time-based exits."""
    forced = []
    if not isinstance(live_pos, dict) or not live_pos:
        return forced
    
    if now_dt is None:
        now_dt = datetime.now(timezone.utc)

    for ticker, (qty, avg) in live_pos.items():
        try:
            qty_i = int(qty)
        except Exception:
            continue
        if qty_i <= 0:
            continue

        try:
            ob = safe_api_call("orderbook", client.get_orderbook, ticker)
        except ServicePaused as e:
            if SELL_DIAGNOSTICS:
                logger.info(f"SELL_DIAG ticker={ticker} qty={qty_i} avg={avg:.2f} ob=PAUSED: {e}")
            continue
        except Exception as e:
            if SELL_DIAGNOSTICS:
                logger.info(f"SELL_DIAG ticker={ticker} qty={qty_i} avg={avg:.2f} ob=ERR: {repr(e)}")
            continue

        yes_bid, yes_bid_sz, yes_ask, yes_ask_sz = get_best_yes_bid_ask(ob)

        profit_edge = None
        if yes_bid is not None:
            profit_edge = float(yes_bid) - float(avg)

        # Position age tracking
        last_ts = get_last_trade_ts(conn, ticker)
        position_age_sec = 0
        if last_ts:
            try:
                last_dt = parse_iso(last_ts)
                if last_dt:
                    position_age_sec = (now_dt - last_dt).total_seconds()
            except Exception:
                pass

        will_sell = False
        reason = "HOLD"
        sell_qty = qty_i
        sell_rank = 1e8  # Base rank for forced sells

        # Stop-loss always takes priority
        if STOP_LOSS_CENTS > 0 and profit_edge is not None and profit_edge <= -float(STOP_LOSS_CENTS):
            will_sell = True
            reason = "STOP_LOSS"
            sell_rank = 1e9  # Highest priority

        # Take-profit with scaling
        elif profit_edge is not None and profit_edge >= float(TAKE_PROFIT_CENTS):
            will_sell = True
            reason = "TAKE_PROFIT"
            sell_rank = 1e9 - 1  # Very high priority

        # Scale-out on partial profit
        elif profit_edge is not None and SCALE_OUT_PROFIT_CENTS > 0 and profit_edge >= float(SCALE_OUT_PROFIT_CENTS):
            will_sell = True
            reason = "SCALE_OUT"
            sell_qty = max(1, int(qty_i * float(SCALE_OUT_QTY_FRACTION)))
            sell_rank = 1e8 + 5

        # Time-based exit: after max hold time, exit on looser edge or any bid
        elif position_age_sec >= MAX_HOLD_SECONDS:
            if yes_bid is not None:
                will_sell = True
                reason = "TIME_EXPIRE"
                sell_rank = 1e8 + 1

        if SELL_DIAGNOSTICS:
            logger.info(
                f"SELL_DIAG ticker={ticker} qty={qty_i} avg={float(avg):.2f} age_s={position_age_sec:.0f} "
                f"yes_bid={yes_bid}({yes_bid_sz}) yes_ask={yes_ask}({yes_ask_sz}) "
                f"pnl_edge={profit_edge if profit_edge is not None else 'NA'} "
                f"-> {reason} (sell_qty={sell_qty})"
            )

        if not EXIT_SCAN_ALL_POSITIONS and reason == "HOLD":
            continue

        if will_sell and yes_bid is not None:
            series = ticker_prefix(ticker)
            if series not in SERIES_TZ or series not in SERIES_LOC:
                continue
            bid = int(yes_bid)
            ask = int(yes_ask) if yes_ask is not None else bid
            spread = max(0, ask - bid)
            forced.append((sell_rank, series, "SELL", ticker, bid, ask, 0.0, spread, sell_qty, float(avg)))

    return forced


def _parse_date_ymd(s: str):
    try:
        y = int(s[0:4])
        m = int(s[5:7])
        d = int(s[8:10])
        return y, m, d
    except Exception:
        return None


def _days_ahead_local(series: str, target_date: str, now_utc: datetime):
    tzname = SERIES_TZ.get(series, "UTC")
    try:
        local_now = now_utc.astimezone(ZoneInfo(tzname))
        today = local_now.date()
    except Exception:
        today = now_utc.date()

    t = _parse_date_ymd(target_date)
    if not t:
        return 0
    ty, tm, td = t
    try:
        from datetime import date
        dt = date(ty, tm, td)
        return int((dt - today).days)
    except Exception:
        return 0


def _series_weights(conn, series: str, default_weights: dict):
    if not DYNAMIC_WEIGHTS:
        return dict(default_weights)

    out = {}
    for prov, default_w in default_weights.items():
        raw_mse = get_state(conn, f"mse:{series}:{prov}", "")
        mse = None
        if raw_mse:
            try:
                mse = float(raw_mse)
            except Exception:
                mse = None
        if mse is None or mse <= 0:
            out[prov] = float(default_w)
            continue
        w = 1.0 / (mse + float(MSE_EPS))
        if w < W_MIN:
            w = W_MIN
        if w > W_MAX:
            w = W_MAX
        out[prov] = float(w)
    return out


def _maybe_settle_event(conn, series: str, event_ticker: str, target_date: str, lat: float, lon: float, tzname: str, ts: str):
    days_ahead = _days_ahead_local(series, target_date, datetime.now(timezone.utc))
    if days_ahead >= 0:
        return

    done_key = f"outcome_done:{event_ticker}"
    if get_state(conn, done_key, "") == "1":
        return

    obs = OpenMeteoObservedHighProvider()
    obs_high = None
    try:
        obs_high = obs.get_high_f(lat, lon, target_date, timezone=tzname)
    except Exception:
        obs_high = None

    if obs_high is None:
        return

    upsert_outcome(conn, event_ticker, series, target_date, float(obs_high), ts)

    rows = conn.execute(
        """
        SELECT provider, mu
        FROM provider_readings
        WHERE event_ticker=?
        ORDER BY id DESC
        """,
        (event_ticker,),
    ).fetchall()

    seen = set()
    for prov, mu in rows:
        if prov in seen:
            continue
        seen.add(prov)
        if mu is None:
            continue
        try:
            mu_f = float(mu)
        except Exception:
            continue

        err = float(obs_high) - mu_f
        prev_bias = get_state(conn, f"bias:{series}:{prov}", "")
        prev_mse = get_state(conn, f"mse:{series}:{prov}", "")

        b0 = 0.0
        m0 = 25.0
        if prev_bias:
            try:
                b0 = float(prev_bias)
            except Exception:
                b0 = 0.0
        if prev_mse:
            try:
                m0 = float(prev_mse)
            except Exception:
                m0 = 25.0

        a = float(BIAS_ALPHA)
        b1 = (1.0 - a) * b0 + a * err
        m1 = (1.0 - a) * m0 + a * (err * err)

        set_state(conn, f"bias:{series}:{prov}", f"{b1:.6f}")
        set_state(conn, f"mse:{series}:{prov}", f"{m1:.6f}")

    set_state(conn, done_key, "1")


def _dynamic_trade_step(edge_eff: float):
    base = int(TRADE_STEP)
    if base <= 1:
        return 1
    k = float(EDGE_QTY_SCALE)
    if k <= 0:
        return base
    mult = int(1 + max(0.0, float(edge_eff)) / k)
    if mult < 1:
        mult = 1
    if mult > int(MAX_STEP_MULT):
        mult = int(MAX_STEP_MULT)
    return int(base * mult)


def _matched_closed_trade_pnls(conn, lookback_hours: int = 72):
    """Return closed trade legs as (ts_utc, pnl_dollars) using FIFO buy/sell matching."""
    try:
        rows = conn.execute(
            f"""SELECT ticker, side, price, qty, ts_utc
                FROM trades
                WHERE datetime(ts_utc) > datetime('now', '-' || ? || ' hours')
                ORDER BY datetime(ts_utc) ASC, id ASC""",
            (int(lookback_hours),),
        ).fetchall()
    except Exception:
        return []

    lots_by_ticker = {}
    closed = []

    for ticker, side, price, qty, ts_utc in rows:
        ticker = str(ticker)
        side = str(side).upper()
        price = float(price)
        qty = int(qty or 0)
        if qty <= 0:
            continue

        if ticker not in lots_by_ticker:
            lots_by_ticker[ticker] = []

        if side == "BUY":
            lots_by_ticker[ticker].append([price, qty])
            continue

        if side != "SELL":
            continue

        remaining = qty
        buy_lots = lots_by_ticker[ticker]
        while remaining > 0 and buy_lots:
            buy_price, buy_qty = buy_lots[0]
            match_qty = min(int(remaining), int(buy_qty))
            pnl = (float(price) - float(buy_price)) * float(match_qty) / 100.0
            closed.append((str(ts_utc), float(pnl)))
            remaining -= match_qty
            buy_qty -= match_qty
            if buy_qty <= 0:
                buy_lots.pop(0)
            else:
                buy_lots[0][1] = buy_qty

    return closed


def compute_consecutive_closed_losses(conn, lookback_hours: int = 72):
    """Count trailing consecutive losing closed trade legs."""
    closed = _matched_closed_trade_pnls(conn, lookback_hours=lookback_hours)
    if not closed:
        return 0, 0

    streak = 0
    for _, pnl in reversed(closed):
        if pnl < 0:
            streak += 1
        elif pnl > 0:
            break
    return streak, len(closed)


def compute_adaptive_edge_penalty_cents(conn):
    """Execution-quality penalty (in cents) to subtract from buy edge."""
    if not ADAPTIVE_EDGE_GATE_ENABLED:
        return 0.0, {}

    fills = compute_fill_analytics(conn, lookback_hours=ADAPTIVE_EDGE_LOOKBACK_HOURS)
    total_orders = int((fills or {}).get("total_orders", 0))
    if total_orders < ADAPTIVE_EDGE_MIN_ORDERS:
        return 0.0, {"reason": "insufficient_sample", "total_orders": total_orders}

    avg_slippage = float((fills or {}).get("avg_slippage_per_order", 0.0))
    fill_rate = float((fills or {}).get("fill_rate", 0.0))

    slippage_penalty = max(0.0, avg_slippage) * float(ADAPTIVE_EDGE_SLIPPAGE_WEIGHT)
    fill_gap = max(0.0, float(ADAPTIVE_EDGE_TARGET_FILL_RATE) - fill_rate)
    reject_penalty = fill_gap * float(ADAPTIVE_EDGE_REJECT_WEIGHT)

    penalty = min(float(ADAPTIVE_EDGE_MAX_PENALTY_CENTS), slippage_penalty + reject_penalty)
    return float(penalty), {
        "total_orders": total_orders,
        "fill_rate": fill_rate,
        "avg_slippage_per_order": avg_slippage,
        "slippage_penalty": slippage_penalty,
        "reject_penalty": reject_penalty,
    }


def _matched_closed_trade_legs(conn, lookback_hours: int = 72):
    """Return matched closed trade legs using FIFO as dict records."""
    try:
        rows = conn.execute(
            f"""SELECT ticker, side, price, qty, ts_utc
                FROM trades
                WHERE datetime(ts_utc) > datetime('now', '-' || ? || ' hours')
                ORDER BY datetime(ts_utc) ASC, id ASC""",
            (int(lookback_hours),),
        ).fetchall()
    except Exception:
        return []

    lots_by_ticker = {}
    out = []

    for ticker, side, price, qty, ts_utc in rows:
        ticker = str(ticker)
        side = str(side).upper()
        price = float(price)
        qty = int(qty or 0)
        if qty <= 0:
            continue

        if ticker not in lots_by_ticker:
            lots_by_ticker[ticker] = []

        if side == "BUY":
            lots_by_ticker[ticker].append([price, qty])
            continue
        if side != "SELL":
            continue

        remaining = qty
        buy_lots = lots_by_ticker[ticker]
        while remaining > 0 and buy_lots:
            buy_price, buy_qty = buy_lots[0]
            match_qty = min(int(remaining), int(buy_qty))
            pnl = (float(price) - float(buy_price)) * float(match_qty) / 100.0
            close_dt = parse_iso(str(ts_utc))
            series = ticker.split("_")[0] if "_" in ticker else ticker
            out.append(
                {
                    "ticker": ticker,
                    "series": series,
                    "close_ts": str(ts_utc),
                    "close_hour_utc": int(close_dt.hour) if close_dt else None,
                    "pnl": float(pnl),
                }
            )
            remaining -= match_qty
            buy_qty -= match_qty
            if buy_qty <= 0:
                buy_lots.pop(0)
            else:
                buy_lots[0][1] = buy_qty

    return out


def compute_series_expectancy_stats(conn, lookback_hours: int = 168, min_closed_trades: int = 20):
    """Compute rolling expectancy per series from closed trade legs."""
    legs = _matched_closed_trade_legs(conn, lookback_hours=lookback_hours)
    by_series = {}
    for leg in legs:
        s = str(leg.get("series") or "")
        pnl = float(leg.get("pnl") or 0.0)
        if not s:
            continue
        if s not in by_series:
            by_series[s] = {"count": 0, "wins": 0, "losses": 0, "sum_pnl": 0.0, "sum_win": 0.0, "sum_loss": 0.0}
        m = by_series[s]
        m["count"] += 1
        m["sum_pnl"] += pnl
        if pnl > 0:
            m["wins"] += 1
            m["sum_win"] += pnl
        elif pnl < 0:
            m["losses"] += 1
            m["sum_loss"] += pnl

    out = {}
    for s, m in by_series.items():
        cnt = int(m["count"])
        if cnt < int(min_closed_trades):
            continue
        wins = int(m["wins"])
        losses = int(m["losses"])
        avg_win = (m["sum_win"] / wins) if wins > 0 else 0.0
        avg_loss = (m["sum_loss"] / losses) if losses > 0 else 0.0
        win_rate = (100.0 * wins / cnt) if cnt > 0 else 0.0
        expectancy = m["sum_pnl"] / cnt if cnt > 0 else 0.0
        out[s] = {
            "count": cnt,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": expectancy,
            "total_pnl": m["sum_pnl"],
        }
    return out


def compute_hourly_expectancy(conn, lookback_hours: int = 336):
    """Compute closed-trade expectancy per UTC hour."""
    legs = _matched_closed_trade_legs(conn, lookback_hours=lookback_hours)
    by_hour = {}
    for leg in legs:
        hr = leg.get("close_hour_utc")
        if hr is None:
            continue
        hr = int(hr)
        pnl = float(leg.get("pnl") or 0.0)
        if hr not in by_hour:
            by_hour[hr] = {"count": 0, "sum_pnl": 0.0}
        by_hour[hr]["count"] += 1
        by_hour[hr]["sum_pnl"] += pnl

    out = {}
    for hr, m in by_hour.items():
        cnt = int(m["count"])
        exp = (m["sum_pnl"] / cnt) if cnt > 0 else 0.0
        out[hr] = {"count": cnt, "expectancy": exp, "total_pnl": m["sum_pnl"]}
    return out


def compute_calibration_diagnostics(conn, lookback_hours: int = 168, bins: int = 10):
    """Detailed calibration diagnostics (ECE/MCE/reliability) overall, by method, and by series."""
    bins = max(5, min(20, int(bins)))
    rows = conn.execute(
        """
        SELECT forecast_prob, outcome, forecast_method, series
        FROM forecast_predictions
        WHERE outcome IS NOT NULL
          AND datetime(outcome_ts_utc) > datetime('now', '-' || ? || ' hours')
        """,
        (int(lookback_hours),),
    ).fetchall()

    def normalize_prob(x):
        p = float(x)
        if p > 1.0:
            p = p / 100.0
        return max(1e-6, min(1.0 - 1e-6, p))

    def bucket_add(store, prob, outcome):
        idx = min(bins - 1, int(prob * bins))
        if idx not in store["bins"]:
            store["bins"][idx] = {"count": 0, "sum_prob": 0.0, "sum_outcome": 0.0}
        b = store["bins"][idx]
        b["count"] += 1
        b["sum_prob"] += prob
        b["sum_outcome"] += outcome

    def new_group():
        return {"count": 0, "brier": 0.0, "log_loss": 0.0, "accuracy": 0, "bins": {}}

    groups = {"overall": new_group(), "method": {}, "series": {}}

    for prob_raw, outcome_raw, method_raw, series_raw in rows:
        try:
            p = normalize_prob(prob_raw)
            o = float(outcome_raw)
        except Exception:
            continue
        if o not in (0.0, 1.0):
            continue

        method = str(method_raw or "unknown")
        series = str(series_raw or "unknown")
        for key, name in (("overall", "overall"), ("method", method), ("series", series)):
            g = groups[key] if key == "overall" else groups[key].setdefault(name, new_group())
            g["count"] += 1
            g["brier"] += (p - o) ** 2
            g["log_loss"] += -(o * math.log(p) + (1.0 - o) * math.log(1.0 - p))
            if (p >= 0.5 and o == 1.0) or (p < 0.5 and o == 0.0):
                g["accuracy"] += 1
            bucket_add(g, p, o)

    def finalize(group):
        cnt = int(group["count"])
        if cnt <= 0:
            return {"count": 0}
        ece = 0.0
        mce = 0.0
        rel = []
        for idx in sorted(group["bins"].keys()):
            b = group["bins"][idx]
            bc = int(b["count"])
            if bc <= 0:
                continue
            avg_p = b["sum_prob"] / bc
            avg_o = b["sum_outcome"] / bc
            gap = abs(avg_p - avg_o)
            ece += gap * (bc / cnt)
            mce = max(mce, gap)
            rel.append(
                {
                    "bin": idx,
                    "count": bc,
                    "avg_pred": avg_p,
                    "avg_outcome": avg_o,
                    "gap": gap,
                }
            )
        return {
            "count": cnt,
            "brier": group["brier"] / cnt,
            "log_loss": group["log_loss"] / cnt,
            "accuracy": group["accuracy"] / cnt,
            "ece": ece,
            "mce": mce,
            "reliability": rel,
        }

    out = {
        "overall": finalize(groups["overall"]),
        "method": {k: finalize(v) for k, v in groups["method"].items()},
        "series": {k: finalize(v) for k, v in groups["series"].items()},
    }
    return out


def settle_yesterday_outcomes(client: KalshiClient, conn, now_dt: datetime, ts: str):
    """Settle outcomes for yesterday's events with robust error handling."""
    archive_truth = OpenMeteoArchiveProvider(timeout_s=30)
    
    yday = (now_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    
    for series in SERIES_LIST_RAW:
        if series not in SERIES_LOC or series not in SERIES_TZ:
            continue
            
        lat, lon = SERIES_LOC[series]
        tzname = SERIES_TZ.get(series, "UTC")

        try:
            resp = safe_api_call("events", client.get_events, limit=200, series_ticker=series)
            events = resp.get("events", []) or []
        except ServicePaused as e:
            logger.warning(f"Events API paused while settling {series}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error fetching events for {series}: {repr(e)}")
            continue
            
        for e in events:
            et = e.get("event_ticker") or e.get("ticker")
            if not et:
                continue
                
            target_date = date_from_event_ticker(et)
            if target_date != yday:
                continue
            
            # Check if already settled
            row = conn.execute(
                "SELECT 1 FROM outcomes WHERE event_ticker=? LIMIT 1",
                (et,),
            ).fetchone()
            if row:
                continue
            
            # Try archive API first, then observed API as fallback
            obs = None
            try:
                obs = archive_truth.get_high_f(lat, lon, target_date, timezone=tzname)
            except Exception as e1:
                logger.warning(f"Archive API failed for {et}: {repr(e1)}, trying ObservedHigh...")
                try:
                    obs_provider = OpenMeteoObservedHighProvider()
                    obs = obs_provider.get_high_f(lat, lon, target_date, timezone=tzname)
                except Exception as e2:
                    logger.error(f"ObservedHigh also failed for {et}: {repr(e2)}")
                    continue
            
            if obs is None:
                logger.warning(f"No observed data available for {et} on {target_date}")
                continue
            
            # Verify we're settling yesterday's market (not a future date)
            try:
                target_dt = datetime.strptime(target_date, "%Y-%m-%d").date()
                if target_dt >= now_dt.date():
                    logger.warning(f"Skipping settlement for future/today date {target_date}")
                    continue
            except Exception as e:
                logger.warning(f"Could not validate target date {target_date}: {repr(e)}")
                pass
            
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO outcomes(event_ticker, series, target_date, observed_high_f, ts_utc) VALUES (?,?,?,?,?)",
                    (et, series, target_date, float(obs), ts),
                )
                conn.commit()
                logger.info(f"SETTLED outcome series={series} event={et} date={target_date} obs_high={float(obs):.1f}")
                
                # Update forecast calibration data with the actual outcome
                if CALIBRATION_TRACKING_ENABLED:
                    try:
                        # Get all tickers that have predictions for this event
                        ticker_rows = conn.execute(
                            "SELECT DISTINCT ticker FROM forecast_predictions WHERE event_ticker=? AND outcome IS NULL",
                            (et,)
                        ).fetchall()
                        for (ticker,) in ticker_rows:
                            update_forecast_outcome(conn, ticker, et, float(obs), ts)
                        # Update provider MSEs (adaptive weight input)
                        try:
                            updated = update_provider_mse(conn, series, et, float(obs), alpha=BIAS_ALPHA)
                            if updated:
                                logger.info(f"Updated provider MSEs for {et}: {updated}")
                        except Exception as e:
                            logger.warning(f"Error updating provider MSEs for {et}: {e}")

                        # Compute heatmap CSV for recent predictions
                        try:
                            hm = compute_heat_map(conn, lookback_hours=24)
                            if hm:
                                # log worst offenders
                                worst = sorted(hm.items(), key=lambda kv: kv[1][1], reverse=True)[:5]
                                logger.info(f"Forecast heatmap top issues (ticker,avg_abs_err): {[(t,v[1]) for t,v in worst]}")
                        except Exception as e:
                            logger.warning(f"Error computing heatmap for {et}: {e}")
                    except Exception as e:
                        logger.warning(f"Error updating calibration outcomes for {et}: {e}")
            except Exception as e:
                logger.error(f"Error saving outcome for {et}: {repr(e)}")
                continue


def main():
    client = KalshiClient()
    # Use absolute path based on this file's location
    from pathlib import Path
    db_path = Path(__file__).parent.parent / "data" / "kalshi_quotes.sqlite"
    conn = open_db(str(db_path))
    # Log the DB path being used so we can diagnose mismatches between processes
    try:
        logger.info(f"[DB] Using database path: {db_path}")
    except Exception:
        print(f"[DB] Using database path: {db_path}")
    
    # Run integrity check on startup
    if DB_INTEGRITY_CHECK:
        logger.info("[DB] Running startup integrity check...")
        issues = check_db_integrity(conn, verbose=True)
        if not issues["ok"]:
            logger.warning(f"[DB] Startup check found issues, but continuing")
    
    if DB_LOG_STATS:
        stats = get_db_stats(conn)
        logger.info(f"[DB] Initial stats: {stats}")

    providers = [NWSProvider(), OpenMeteoProvider(), OpenMeteoEnsembleProvider()]
    if USE_VISUAL_CROSSING:
        vc_provider = VisualCrossingProvider(timeout_s=VISUAL_CROSSING_TIMEOUT)
        providers.append(vc_provider)
        if not getattr(vc_provider, "api_key", ""):
            logger.warning("USE_VISUAL_CROSSING=1 but VISUAL_CROSSING_API_KEY is not set; provider will be skipped.")
    if USE_ANALOG_ENSEMBLE:
        providers.append(AnalogEnsembleProvider(timeout_s=ANALOG_ENSEMBLE_TIMEOUT))
    
    hourly_feat = OpenMeteoHourlyFeaturesProvider()

    default_weights = {
        "nws": float(os.getenv("W_NWS", "1.0")),
        "open_meteo": float(os.getenv("W_OPEN_METEO", "0.9")),
        "open_meteo_ens": float(os.getenv("W_OPEN_METEO_ENS", "1.5")),
    }
    if USE_VISUAL_CROSSING:
        default_weights["visual_crossing"] = float(os.getenv("W_VISUAL_CROSSING", "1.1"))
    if USE_ANALOG_ENSEMBLE:
        default_weights["analog_ens"] = float(os.getenv("W_ANALOG_ENS", "1.8"))

    SERIES_LIST = []
    skipped = []
    for s in SERIES_LIST_RAW:
        if s in SERIES_LOC and s in SERIES_TZ:
            SERIES_LIST.append(s)
        else:
            skipped.append(s)

    mode = "LIVE" if LIVE_TRADING else "PAPER"
    logger.info(f"{mode} bot running. SeriesList={SERIES_LIST} BaseURL={client.base_url}")
    if skipped:
        logger.warning(f"Skipping series with no loc/tz mapping: {skipped}")
    logger.info(f"KILL_SWITCH={int(KILL_SWITCH)} MAKER_MODE={int(MAKER_MODE)}")
    logger.info(
        f"Global caps: MAX_TOTAL_RISK=${MAX_TOTAL_RISK_DOLLARS:.2f} "
        f"MAX_LIVE_PER_LOOP={MAX_LIVE_ORDERS_PER_LOOP} "
        f"(SELL={MAX_SELL_ORDERS_PER_LOOP}, BUY={MAX_BUY_ORDERS_PER_LOOP}) "
        f"MAX_LIVE_PER_DAY={MAX_LIVE_ORDERS_PER_DAY}"
    )

    default_persist = max(FAST_EVERY_SECONDS, RUN_EVERY_SECONDS)
    PERSIST_SECONDS = int(os.getenv("PERSIST_SECONDS", str(default_persist)))
    
    db_last_cleanup = time.time()
    db_last_vacuum = time.time()
    db_last_stats = time.time()
    calibration_last_log = time.time()
    pnl_last_log = time.time()
    forecast_acc_last_log = time.time()
    fill_analytics_last_log = time.time()
    correlation_last_log = time.time()
    greeks_last_log = time.time()
    vol_alert_last_log = time.time()
    
    consecutive_errors = 0
    max_consecutive_errors = 5
    live_pos = {}

    while True:
        ts = iso_now()
        now_dt = datetime.now(timezone.utc)
        
        # Periodic database maintenance
        now_seconds = time.time()
        
        # Cleanup old snapshots
        if now_seconds - db_last_cleanup > DB_CLEANUP_EVERY_MINUTES * 60:
            try:
                deleted = cleanup_old_snapshots(conn, keep_hours=DB_CLEANUP_KEEP_HOURS, verbose=True)
                db_last_cleanup = now_seconds
            except Exception as e:
                logger.warning(f"[DB] Cleanup error: {e}")
        
        # Vacuum and analyze (less frequently)
        if now_seconds - db_last_vacuum > DB_VACUUM_EVERY_HOURS * 3600:
            try:
                logger.info("[DB] Running VACUUM and ANALYZE...")
                vacuum_and_analyze(conn, verbose=True)
                db_last_vacuum = now_seconds
            except Exception as e:
                logger.warning(f"[DB] Vacuum error: {e}")
        
        # Log statistics
        if DB_LOG_STATS and now_seconds - db_last_stats > 3600:  # Every hour
            try:
                stats = get_db_stats(conn)
                logger.info(f"[DB] Stats: {stats}")
                db_last_stats = now_seconds
            except Exception as e:
                logger.warning(f"[DB] Stats error: {e}")
        
        # Log calibration metrics periodically
        if CALIBRATION_TRACKING_ENABLED and now_seconds - calibration_last_log > CALIBRATION_LOG_INTERVAL_HOURS * 3600:
            try:
                # Keep legacy summary
                metrics = compute_calibration_metrics(conn, lookback_hours=CALIBRATION_LOG_INTERVAL_HOURS)
                if metrics:
                    for method, m in metrics.items():
                        logger.info(
                            f"[CALIBRATION] method={method} count={m['count']} "
                            f"brier={m['brier']:.4f} log_loss={m['log_loss']:.4f} accuracy={m['accuracy']:.2%}"
                        )

                # Enhanced diagnostics
                diag = compute_calibration_diagnostics(
                    conn,
                    lookback_hours=CALIBRATION_DIAG_LOOKBACK_HOURS,
                    bins=CALIBRATION_DIAG_BINS,
                )
                overall = diag.get("overall", {})
                if overall and int(overall.get("count", 0)) > 0:
                    logger.info(
                        f"[CALIBRATION_DIAG] overall count={overall['count']} brier={overall['brier']:.4f} "
                        f"log_loss={overall['log_loss']:.4f} acc={overall['accuracy']:.2%} "
                        f"ece={overall['ece']:.4f} mce={overall['mce']:.4f}"
                    )
                    rel = sorted(overall.get("reliability", []), key=lambda x: x.get("gap", 0.0), reverse=True)
                    if rel:
                        w = rel[0]
                        logger.info(
                            f"[CALIBRATION_DIAG] worst_bin bin={w['bin']} count={w['count']} "
                            f"avg_pred={w['avg_pred']:.3f} avg_outcome={w['avg_outcome']:.3f} gap={w['gap']:.3f}"
                        )

                method_diag = [
                    (k, v) for k, v in (diag.get("method") or {}).items()
                    if int(v.get("count", 0)) >= CALIBRATION_DIAG_MIN_SAMPLES
                ]
                for method, m in sorted(method_diag, key=lambda kv: kv[1].get("ece", 0.0), reverse=True)[:5]:
                    logger.info(
                        f"[CALIBRATION_METHOD] {method} count={m['count']} acc={m['accuracy']:.2%} "
                        f"brier={m['brier']:.4f} ece={m['ece']:.4f} mce={m['mce']:.4f}"
                    )

                series_diag = [
                    (k, v) for k, v in (diag.get("series") or {}).items()
                    if int(v.get("count", 0)) >= CALIBRATION_DIAG_MIN_SAMPLES
                ]
                for series_name, m in sorted(series_diag, key=lambda kv: kv[1].get("ece", 0.0), reverse=True)[:8]:
                    logger.info(
                        f"[CALIBRATION_SERIES] {series_name} count={m['count']} acc={m['accuracy']:.2%} "
                        f"brier={m['brier']:.4f} ece={m['ece']:.4f} mce={m['mce']:.4f}"
                    )

                calibration_last_log = now_seconds
            except Exception as e:
                logger.warning(f"Calibration logging error: {e}")

        # Log P&L metrics periodically
        if PNL_TRACKING_ENABLED and now_seconds - pnl_last_log > PNL_LOG_INTERVAL_HOURS * 3600:
            try:
                pnl = compute_pnl_metrics(conn, lookback_hours=PNL_LOOKBACK_HOURS)
                pnl_by_series = compute_pnl_by_series(conn, lookback_hours=PNL_LOOKBACK_HOURS)
                if pnl and pnl["count_trades"] > 0:
                    logger.info(
                        f"[PNL] total=${pnl['total_pnl']:.2f} trades={pnl['count_trades']} "
                        f"wins={pnl['count_wins']} ({pnl['win_rate']:.1f}%) "
                        f"avg_win=${pnl['avg_win']:.2f} avg_loss=${pnl['avg_loss']:.2f} "
                        f"max_win=${pnl['max_win']:.2f} max_loss=${pnl['max_loss']:.2f}"
                    )
                    # Log per-series breakdown
                    for series, series_pnl in pnl_by_series.items():
                        logger.info(f"[PNL_SERIES] {series} ${series_pnl:.2f}")
                pnl_last_log = now_seconds
            except Exception as e:
                logger.warning(f"P&L logging error: {e}")

        # Log forecast accuracy metrics periodically
        if FORECAST_ACCURACY_TRACKING_ENABLED and now_seconds - forecast_acc_last_log > FORECAST_ACCURACY_LOG_INTERVAL_HOURS * 3600:
            try:
                # Log provider accuracy
                provider_acc = compute_provider_accuracy(conn, lookback_hours=FORECAST_ACCURACY_LOOKBACK_HOURS)
                rankings = get_provider_rankings(conn, lookback_hours=FORECAST_ACCURACY_LOOKBACK_HOURS)
                
                if rankings:
                    logger.info("[FORECAST_ACCURACY] Provider rankings (RMSE):")
                    for provider, rmse, count in rankings:
                        if count > 0:
                            macc = provider_acc.get(provider, {})
                            logger.info(
                                f"  {provider}: RMSE={rmse:.3f} bias={macc.get('bias', 0):.2f} "
                                f"MAE={macc.get('mae', 0):.2f} count={count}"
                            )
                
                # Log forecast method accuracy
                method_acc = compute_forecast_method_accuracy(conn, lookback_hours=FORECAST_ACCURACY_LOOKBACK_HOURS)
                if method_acc:
                    logger.info("[FORECAST_METHOD] Accuracy by method:")
                    for method, m in sorted(method_acc.items()):
                        logger.info(
                            f"  {method}: brier={m['brier']:.4f} accuracy={m['accuracy']:.1f}% count={m['count']}"
                        )
                
                forecast_acc_last_log = now_seconds
            except Exception as e:
                logger.warning(f"Forecast accuracy logging error: {e}")

        # Log fill analytics periodically
        if FILL_ANALYTICS_TRACKING_ENABLED and now_seconds - fill_analytics_last_log > FILL_ANALYTICS_LOG_INTERVAL_HOURS * 3600:
            try:
                fills = compute_fill_analytics(conn, lookback_hours=FILL_ANALYTICS_LOOKBACK_HOURS)
                if fills and fills["total_orders"] > 0:
                    logger.info(
                        f"[FILL_ANALYTICS] total_orders={fills['total_orders']} "
                        f"filled={fills['filled_orders']} partial={fills['partial_fills']} "
                        f"rejected={fills['rejected_orders']} fill_rate={fills['fill_rate']:.1f}% "
                        f"avg_fill_pct={fills['avg_fill_pct']:.1f}% "
                        f"avg_slippage={fills['avg_slippage_per_order']:.2f}c"
                    )
                    # Log by-side breakdown
                    for side in ["BUY", "SELL"]:
                        side_stats = fills["by_side"].get(side, {})
                        if side_stats.get("total", 0) > 0:
                            logger.info(
                                f"  {side}: {side_stats['filled']}/{side_stats['total']} filled, "
                                f"partial={side_stats['partial']}, "
                                f"total_slippage={side_stats['slippage']:.2f}c"
                            )
                fill_analytics_last_log = now_seconds
            except Exception as e:
                logger.warning(f"Fill analytics logging error: {e}")

        # Log correlation matrix periodically
        if CORRELATION_HEDGING_ENABLED and now_seconds - correlation_last_log > CORRELATION_LOG_INTERVAL_HOURS * 3600:
            try:
                correlations = compute_series_correlations(conn, lookback_hours=24)
                if correlations:
                    logger.info("[CORRELATION_MATRIX] Significant correlations (|corr| > 0.3):")
                    for (s1, s2), corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
                        corr_type = "positive" if corr > 0 else "negative"
                        logger.info(f"  {s1} <-> {s2}: {corr:.3f} ({corr_type})")
                
                # Log correlated exposure per series
                if live_pos:
                    corr_exposure = compute_correlated_exposure(conn, live_pos, lookback_hours=24)
                    max_corr = MAX_TOTAL_RISK_DOLLARS * MAX_CORRELATED_EXPOSURE_PCT / 100.0
                    if corr_exposure:
                        logger.info(f"[CORRELATED_EXPOSURE] max_allowed=${max_corr:.2f}:")
                        for series, exposure in sorted(corr_exposure.items(), key=lambda x: x[1], reverse=True):
                            status = "⚠️ ALERT" if exposure > max_corr else "OK"
                            logger.info(f"  {series}: ${exposure:.2f} {status}")
                
                correlation_last_log = now_seconds
            except Exception as e:
                logger.warning(f"Correlation logging error: {e}")
        
        # Log portfolio delta (Greeks-lite) periodically
        if GREEKS_TRACKING_ENABLED and now_seconds - greeks_last_log > GREEKS_LOG_INTERVAL_HOURS * 3600:
            try:
                if live_pos:
                    delta_result = compute_portfolio_delta(conn, live_pos, {})
                    logger.info(f"[PORTFOLIO_DELTA] Total delta: {delta_result['portfolio_delta']:.2f} contracts")
                    if delta_result['delta_by_series']:
                        logger.info("  By series:")
                        for series, delta in sorted(delta_result['delta_by_series'].items(), key=lambda x: abs(x[1]), reverse=True):
                            logger.info(f"    {series}: {delta:.2f}")
                greeks_last_log = now_seconds
            except Exception as e:
                logger.debug(f"Greeks logging (non-critical): {e}")
        
        # Log volatility spikes periodically
        if VOL_ALERT_ENABLED and now_seconds - vol_alert_last_log > VOL_ALERT_LOG_INTERVAL_HOURS * 3600:
            try:
                vol_spikes = detect_vol_spikes(conn, lookback_hours=1, vol_spike_threshold=VOL_SPIKE_THRESHOLD)
                if vol_spikes:
                    spikes_detected = [s for s, data in vol_spikes.items() if data.get('is_spike')]
                    if spikes_detected:
                        logger.warning(f"[VOL_ALERT] {len(spikes_detected)} series with vol spikes:")
                        for series in spikes_detected:
                            data = vol_spikes[series]
                            logger.warning(
                                f"  {series}: recent_vol={data['recent_vol']:.3f}, "
                                f"historical_vol={data['historical_vol']:.3f}, "
                                f"ratio={data['spike_ratio']:.2f}x (threshold={VOL_SPIKE_THRESHOLD:.1f}x)"
                            )
                    else:
                        logger.info("[VOL_ALERT] No spikes detected")
                vol_alert_last_log = now_seconds
            except Exception as e:
                logger.warning(f"Vol spike logging error: {e}")
        
        # Safe settlement with error recovery
        try:
            settle_yesterday_outcomes(client, conn, now_dt, ts)
            consecutive_errors = 0  # Reset on success
        except Exception as e:
            logger.error(f"Error in settle_yesterday_outcomes: {repr(e)}")
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                logger.critical("Too many consecutive errors, exiting")
                break

        next_sleep = RUN_EVERY_SECONDS

        placed = 0
        placed_live = 0
        placed_live_sell = 0
        placed_live_buy = 0

        try:
            if KILL_SWITCH:
                time.sleep(RUN_EVERY_SECONDS)
                continue

            bal_cents = None
            if LIVE_TRADING:
                try:
                    b = safe_api_call("portfolio", client.get_portfolio_balance) or {}
                    bal_cents = int(b.get("balance", 0))
                except ServicePaused as e:
                    logger.warning(f"Portfolio API paused: {e}")
                    bal_cents = 0
                except Exception as e:
                    logger.error(f"Error fetching balance: {repr(e)}")
                    bal_cents = 0
                
                if bal_cents < MIN_BALANCE_CENTS:
                    logger.warning(f"BALANCE too low: {bal_cents}c < {MIN_BALANCE_CENTS}c, sleeping.")
                    time.sleep(RUN_EVERY_SECONDS)
                    continue

            # Try to amend stale orders (smarter than cancel+relist)
            # NOTE: latest_orderbook is populated per-series in the loop below
            # For now, skip order amending here and rely on cancel flow
            if ORDER_AMENDING_ENABLED:
                try:
                    # Will be called inside series loop where latest_orderbook is available
                    pass
                except Exception as e:
                    logger.warning(f"Error in order amending: {e}")

            canceled = cancel_stale_orders(client, conn, now_dt, ts)
            if canceled:
                logger.info(f"Canceled {canceled} stale orders")


            day_key = now_dt.strftime("%Y-%m-%d")
            used_today = int(get_state(conn, f"live_orders_{day_key}", "0"))

            live_pos = {}
            if LIVE_TRADING and SYNC_PORTFOLIO_POSITIONS:
                live_pos = fetch_live_positions(client)
                if live_pos:
                    reconcile_positions_to_db(conn, live_pos)

            # ---- Safety breakers: block only new BUY entries; allow SELL exits ----
            block_new_buys = False
            block_reasons = []

            if LIVE_TRADING and bal_cents is not None and DAILY_DRAWDOWN_STOP_PCT > 0:
                day_start_key = f"risk:day_start_balance_cents:{day_key}"
                day_start_raw = get_state(conn, day_start_key, "")
                if day_start_raw:
                    try:
                        day_start_bal = int(day_start_raw)
                    except Exception:
                        day_start_bal = int(bal_cents)
                        set_state(conn, day_start_key, str(day_start_bal))
                else:
                    day_start_bal = int(bal_cents)
                    set_state(conn, day_start_key, str(day_start_bal))

                if day_start_bal > 0:
                    drawdown_pct = 100.0 * max(0, day_start_bal - int(bal_cents)) / float(day_start_bal)
                    drawdown_latch_key = "risk:drawdown_stop_day"
                    latched_day = get_state(conn, drawdown_latch_key, "")

                    if drawdown_pct >= float(DAILY_DRAWDOWN_STOP_PCT) and latched_day != day_key:
                        set_state(conn, drawdown_latch_key, day_key)
                        logger.warning(
                            f"[RISK_BREAKER] Daily drawdown triggered: drawdown={drawdown_pct:.2f}% "
                            f"threshold={DAILY_DRAWDOWN_STOP_PCT:.2f}% day_start={day_start_bal}c now={bal_cents}c"
                        )

                    if get_state(conn, drawdown_latch_key, "") == day_key:
                        block_new_buys = True
                        block_reasons.append(f"daily_drawdown_stop day={day_key}")

            if MAX_CONSECUTIVE_LOSSES > 0:
                pause_key = "risk:loss_streak_pause_until_utc"
                pause_until_raw = get_state(conn, pause_key, "")
                pause_until_dt = parse_iso(pause_until_raw) if pause_until_raw else None

                if pause_until_dt and now_dt < pause_until_dt:
                    block_new_buys = True
                    mins_left = int((pause_until_dt - now_dt).total_seconds() / 60.0)
                    block_reasons.append(f"loss_streak_cooldown {max(0, mins_left)}m_left")
                else:
                    loss_streak, closed_count = compute_consecutive_closed_losses(
                        conn,
                        lookback_hours=CONSECUTIVE_LOSS_LOOKBACK_HOURS,
                    )
                    if closed_count >= MAX_CONSECUTIVE_LOSSES and loss_streak >= MAX_CONSECUTIVE_LOSSES:
                        pause_until = now_dt + timedelta(minutes=max(1, LOSS_STREAK_COOLDOWN_MINUTES))
                        set_state(conn, pause_key, pause_until.isoformat())
                        block_new_buys = True
                        block_reasons.append(
                            f"loss_streak_trigger losses={loss_streak} lookback={CONSECUTIVE_LOSS_LOOKBACK_HOURS}h"
                        )
                        logger.warning(
                            f"[RISK_BREAKER] Consecutive closed losses={loss_streak} >= {MAX_CONSECUTIVE_LOSSES}. "
                            f"Blocking BUYs until {pause_until.isoformat()}"
                        )

            adaptive_edge_penalty = 0.0
            adaptive_diag = {}
            try:
                adaptive_edge_penalty, adaptive_diag = compute_adaptive_edge_penalty_cents(conn)
            except Exception as e:
                logger.warning(f"Adaptive edge penalty calculation failed: {repr(e)}")
                adaptive_edge_penalty, adaptive_diag = 0.0, {}

            if adaptive_edge_penalty > 0:
                logger.warning(
                    f"[ADAPTIVE_EDGE] buy_penalty={adaptive_edge_penalty:.2f}c "
                    f"orders={adaptive_diag.get('total_orders', 0)} fill_rate={adaptive_diag.get('fill_rate', 0.0):.1f}% "
                    f"slippage={adaptive_diag.get('avg_slippage_per_order', 0.0):.2f}c "
                    f"(slip_pen={adaptive_diag.get('slippage_penalty', 0.0):.2f} "
                    f"rej_pen={adaptive_diag.get('reject_penalty', 0.0):.2f})"
                )

            if block_new_buys:
                logger.warning(f"[RISK_BREAKER] BUY entries blocked: {'; '.join(block_reasons)}")

            # ---- Entry quality filters (BUY-only) ----
            series_expectancy = {}
            if SERIES_EXPECTANCY_FILTER_ENABLED:
                try:
                    series_expectancy = compute_series_expectancy_stats(
                        conn,
                        lookback_hours=SERIES_EXPECTANCY_LOOKBACK_HOURS,
                        min_closed_trades=SERIES_EXPECTANCY_MIN_CLOSED_TRADES,
                    )
                    if series_expectancy:
                        worst = sorted(series_expectancy.items(), key=lambda kv: kv[1].get("expectancy", 0.0))[:5]
                        logger.info(
                            f"[SERIES_EXPECTANCY] sample={len(series_expectancy)} "
                            f"worst={[(s, round(m.get('expectancy', 0.0), 4), m.get('count', 0)) for s, m in worst]}"
                        )
                except Exception as e:
                    logger.warning(f"Series expectancy filter error: {repr(e)}")
                    series_expectancy = {}

            bad_hours = set()
            if HOURLY_FILTER_ENABLED:
                try:
                    hourly = compute_hourly_expectancy(conn, lookback_hours=HOURLY_FILTER_LOOKBACK_HOURS)
                    for hr, m in hourly.items():
                        if int(m.get("count", 0)) < HOURLY_FILTER_MIN_CLOSED_TRADES:
                            continue
                        if float(m.get("expectancy", 0.0)) < float(HOURLY_FILTER_MIN_EXPECTANCY_DOLLARS):
                            bad_hours.add(int(hr))
                    if bad_hours:
                        logger.warning(f"[HOURLY_FILTER] blocking_buy_hours_utc={sorted(list(bad_hours))}")
                except Exception as e:
                    logger.warning(f"Hourly filter error: {repr(e)}")
                    bad_hours = set()

            ticker_day_trade_counts = {}
            try:
                rows = conn.execute(
                    "SELECT ticker, COUNT(*) FROM trades WHERE DATE(ts_utc)=? GROUP BY ticker",
                    (day_key,),
                ).fetchall()
                for t, cnt in rows:
                    ticker_day_trade_counts[str(t)] = int(cnt)
            except Exception as e:
                logger.warning(f"Ticker day trade count preload error: {repr(e)}")
                ticker_day_trade_counts = {}

            candidates = []

            if LIVE_TRADING and (SELL_DIAGNOSTICS or EXIT_SCAN_ALL_POSITIONS):
                forced_sells = build_exit_candidates_from_positions(client, conn, ts, live_pos, now_dt)
                if forced_sells:
                    candidates.extend(forced_sells)

            for series in SERIES_LIST:
                lat, lon = SERIES_LOC[series]
                tzname = SERIES_TZ.get(series, "UTC")

                event_ticker = pick_best_event_ticker(client, series, now_dt)
                if not event_ticker:
                    continue

                target_date = date_from_event_ticker(event_ticker)
                if not target_date:
                    logger.warning(f"Could not parse date from event ticker: {event_ticker}")
                    continue

                try:
                    resp = safe_api_call("markets", client.get_markets, series_ticker=series, limit=200)
                    ms = resp.get("markets", []) or []
                except ServicePaused as e:
                    logger.warning(f"Markets API paused: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error fetching markets for {series}: {repr(e)}")
                    continue
                
                markets = [m for m in ms if m.get("event_ticker") == event_ticker and (m.get("status") in ("active", "open"))]
                if not markets:
                    continue

                snapshot_markets(conn, ts, event_ticker, markets)

                base_sigma = baseline_sigma_for(series, target_date)

                biases = {}
                bias_names = ["nws", "open_meteo", "open_meteo_ens"]
                if USE_VISUAL_CROSSING:
                    bias_names.append("visual_crossing")
                if USE_ANALOG_ENSEMBLE:
                    bias_names.append("analog_ens")
                for name in bias_names:
                    v = get_state(conn, f"bias:{series}:{name}", "")
                    biases[name] = float(v) if v else 0.0

                weights = _series_weights(conn, series, default_weights)

                if WEATHER_FETCH_DEBUG:
                    logger.info(f"WEATHER_START series={series} target_date={target_date} providers={len(providers)}")
                
                try:
                    fused = fuse_weather(
                        lat=lat,
                        lon=lon,
                        target_date=target_date,
                        base_sigma_f=base_sigma,
                        providers=providers,
                        weights=weights,
                        provider_bias_f=biases,
                        hourly_feature_provider=hourly_feat,
                        timezone=tzname,
                    )
                    if WEATHER_FETCH_DEBUG:
                        logger.info(f"WEATHER_DONE series={series} mu={fused.get('mu') if fused else None}")

                except Exception as e:
                    logger.error(f"Error fusing weather for {series}: {repr(e)}")
                    continue
                
                if not fused:
                    logger.warning(f"No weather data available for {series}")
                    continue

                mu = float(fused["mu"])
                sigma = float(fused["sigma"])
                meta = fused.get("meta") or {}
                highs_members = fused.get("high_members")

                feat = (meta.get("features") or {}) if isinstance(meta, dict) else {}
                fr = float(feat.get("front_risk", 0.0)) if isinstance(feat, dict) else 0.0
                cr = float(feat.get("cloud_risk", 0.0)) if isinstance(feat, dict) else 0.0

                mu = float(mu) - float(CLOUD_MU_ADJ) * cr
                sigma = float(sigma) + float(FRONT_SIGMA_ADD) * fr + float(CLOUD_SIGMA_ADD) * cr

                da = _days_ahead_local(series, target_date, now_dt)
                if da > 0:
                    sigma = float(sigma) * (1.0 + float(SIGMA_HORIZON_K) * float(da))

                snapshot_forecast(conn, ts, event_ticker, target_date, mu, series=series)
                
                # Check forecast freshness - skip if data is too old
                is_forecast_fresh, forecast_age = check_forecast_freshness(ts, now_dt, MAX_FORECAST_AGE_SECONDS)
                if not is_forecast_fresh and REQUIRE_FRESH_FORECAST:
                    logger.warning(
                        f"SKIP_STALE_FORECAST {series} {event_ticker} age={forecast_age:.0f}s max={MAX_FORECAST_AGE_SECONDS}s"
                    )
                    continue

                dist = meta.get("dist") if isinstance(meta, dict) else {}
                prov_rows = []
                if isinstance(dist, dict):
                    for prov, d in dist.items():
                        if not isinstance(d, dict):
                            continue
                        prov_rows.append(
                            {
                                "provider": str(prov),
                                "mu": d.get("mu"),
                                "std": d.get("std"),
                                "meta": {k: v for k, v in d.items() if k not in ("mu", "std")},
                            }
                        )
                else:
                    if isinstance(meta, dict) and isinstance(meta.get("readings"), list):
                        for prov, v in meta.get("readings"):
                            prov_rows.append({"provider": str(prov), "mu": v, "std": None, "meta": None})

                snapshot_provider_readings(conn, ts, series, event_ticker, target_date, prov_rows)

                snapshot_weather_features(
                    conn,
                    ts_utc=ts,
                    series=series,
                    event_ticker=event_ticker,
                    target_date=target_date,
                    mu=mu,
                    sigma=sigma,
                    ens_members=int(meta.get("ens_members") or 0) if isinstance(meta, dict) else 0,
                    ens_std=float(meta.get("ens_std") or 0.0) if isinstance(meta, dict) else 0.0,
                    disagree_sigma=float(meta.get("disagree_sigma") or 0.0) if isinstance(meta, dict) else 0.0,
                    front_risk=fr,
                    cloud_risk=cr,
                    meta=meta if isinstance(meta, dict) else None,
                )

                _maybe_settle_event(conn, series, event_ticker, target_date, lat, lon, tzname, ts)

                min_edge_now, mult = min_edge_for_time(series, now_dt)
                
                # Regime detection: throttle if market stress detected
                regime_check = detect_market_regime(conn, now_dt)
                if regime_check["throttle"]:
                    logger.warning(f"REGIME_THROTTLE {series} reason={regime_check['reason']} severity={regime_check['severity']:.2f}")
                    min_edge_now = min_edge_now * REGIME_EDGE_MULTIPLIER
                
                # Dynamic position sizing based on volatility
                position_mult = compute_dynamic_position_multiplier(sigma)
                vol_regime = volatility_regime_for_sigma(sigma)
                
                logger.info(f"{series} forecast mu={mu:.2f} sigma={sigma:.2f} regime={vol_regime} mult={position_mult:.2f} min_edge_now={min_edge_now:.1f} (x{mult:.2f})")

                checked = 0
                cand = 0
                buy_reject_ask_low = 0
                buy_reject_ask_high = 0
                buy_reject_fair_low = 0
                buy_reject_edge = 0

                for m in markets:
                    ticker = m.get("ticker")
                    bid = m.get("yes_bid")
                    ask = m.get("yes_ask")
                    vol24 = int(m.get("volume_24h") or 0)
                    oi = int(m.get("open_interest") or 0)

                    if m.get("yes_bid") is None:
                        continue

                    if not ticker or bid is None or ask is None:
                        continue

                    bid = int(bid)
                    ask = int(ask)
                    if bid < 0 or ask < 0:
                        continue

                    spread = ask - bid
                    if spread < 0 or spread > MAX_SPREAD_CENTS:
                        continue
                    if vol24 < MIN_VOL_24H or oi < MIN_OPEN_INTEREST:
                        continue

                    p = fair_prob_market_with_calibration(
                        conn, ts, m, mu, sigma,
                        highs_members if isinstance(highs_members, list) else None,
                        market_ticker=ticker, series=series, event_ticker=event_ticker
                    )
                    if p is None:
                        continue

                    fair = p * 100.0
                    buy_edge = fair - float(ask)
                    sell_edge = float(bid) - fair

                    tail_penalty = 0.0
                    if TAIL_PENALTY_ENABLED:
                        tail_distance = abs(float(fair) - float(TAIL_PENALTY_CENTER_PCT))
                        penalized_distance = max(0.0, tail_distance - float(TAIL_PENALTY_FREE_BAND_PCT))
                        tail_penalty = min(
                            float(TAIL_PENALTY_CAP_CENTS),
                            penalized_distance * float(TAIL_PENALTY_SLOPE_CENTS_PER_PCT),
                        )

                    eff_buy = buy_edge - (spread / 2.0) - COST_BUFFER_CENTS - adaptive_edge_penalty - tail_penalty
                    eff_sell = sell_edge - (spread / 2.0) - COST_BUFFER_CENTS

                    checked += 1
                    cur_qty, cur_avg = get_position_effective(conn, ticker, live_pos)

                    buy_thresh = min_edge_now + (NEW_POS_EDGE_BONUS if cur_qty == 0 else 0.0)
                    buy_gate_ok = True
                    gate_reason = ""
                    if int(ask) < int(MIN_BUY_ASK_CENTS):
                        buy_gate_ok = False
                        gate_reason = "ASK_TOO_LOW"
                        buy_reject_ask_low += 1
                    elif int(ask) > int(MAX_BUY_ASK_CENTS):
                        buy_gate_ok = False
                        gate_reason = "ASK_TOO_HIGH"
                        buy_reject_ask_high += 1
                    elif float(fair) < float(MIN_FAIR_PROB_PCT):
                        buy_gate_ok = False
                        gate_reason = "FAIR_TOO_LOW"
                        buy_reject_fair_low += 1

                    if buy_gate_ok and eff_buy >= buy_thresh:
                        candidates.append((float(eff_buy), series, "BUY", ticker, bid, ask, float(fair), int(spread), cur_qty, cur_avg))
                        cand += 1
                    elif not buy_gate_ok:
                        if LOG_CANDIDATE_REJECTION:
                            logger.debug(
                                f"REJECT_BUY {ticker} reason={gate_reason} "
                                f"(fair={fair:.1f}%, ask={ask}, min_fair={MIN_FAIR_PROB_PCT:.1f}%, "
                                f"ask_range=[{MIN_BUY_ASK_CENTS},{MAX_BUY_ASK_CENTS}])"
                            )
                    elif LOG_CANDIDATE_REJECTION and eff_buy > -2.0:  # Only log near-misses to avoid spam
                        buy_reject_edge += 1
                        logger.debug(
                            f"REJECT_BUY {ticker} eff_buy={eff_buy:.2f}c < thresh={buy_thresh:.2f}c "
                            f"(fair={fair:.1f}, ask={ask}, buy_edge={buy_edge:.2f}, "
                            f"min_edge={min_edge_now:.2f}, adaptive_penalty={adaptive_edge_penalty:.2f}, "
                            f"tail_penalty={tail_penalty:.2f})"
                        )
                    elif not LOG_CANDIDATE_REJECTION:
                        buy_reject_edge += 1
                    else:
                        buy_reject_edge += 1

                    if cur_qty > 0:
                        profit_edge = float(bid) - float(cur_avg)
                        tp_hit = (TAKE_PROFIT_CENTS > 0) and (profit_edge >= TAKE_PROFIT_CENTS)
                        sl_hit = (STOP_LOSS_CENTS > 0) and ((float(cur_avg) - float(bid)) >= STOP_LOSS_CENTS)
                        
                        # Time-based exit: force exit after max hold time
                        last_ts = get_last_trade_ts(conn, ticker)
                        time_expired = False
                        if last_ts:
                            try:
                                last_dt = parse_iso(last_ts)
                                if last_dt:
                                    hold_time = (now_dt - last_dt).total_seconds()
                                    time_expired = hold_time >= MAX_HOLD_SECONDS
                            except Exception:
                                pass

                        alpha_sell = eff_sell >= min_edge_now
                        
                        # Scale-out: sell partial if profitable but not at full take-profit
                        scale_out = (SCALE_OUT_PROFIT_CENTS > 0 and profit_edge >= float(SCALE_OUT_PROFIT_CENTS) 
                                     and profit_edge < float(TAKE_PROFIT_CENTS))

                        should_sell = False
                        sell_reason = "HOLD"
                        
                        if sl_hit:
                            should_sell = True
                            sell_reason = "STOP_LOSS"
                        elif tp_hit:
                            should_sell = True
                            sell_reason = "TAKE_PROFIT"
                        elif time_expired and bid is not None:
                            should_sell = True
                            sell_reason = "TIME_EXPIRE"
                        elif scale_out:
                            should_sell = True
                            sell_reason = "SCALE_OUT"
                        else:
                            if SELL_ONLY_IF_PROFIT:
                                should_sell = False
                            else:
                                should_sell = alpha_sell
                                if should_sell:
                                    sell_reason = "ALPHA"

                        if should_sell and bid is not None:
                            rank = profit_edge if profit_edge is not None else 0
                            if sl_hit:
                                rank = 1e7  # High priority
                            elif tp_hit:
                                rank = 1e6  # Very high priority
                            elif time_expired:
                                rank = 1e5  # Medium-high priority
                            candidates.append((float(rank), series, "SELL", ticker, bid, ask, float(fair), int(spread), cur_qty, cur_avg))
                            cand += 1

                logger.info(
                    f"SUMMARY {series} event={event_ticker} date={target_date} cand={cand} checked={checked} "
                    f"buy_rejects=(ask_low={buy_reject_ask_low}, ask_high={buy_reject_ask_high}, "
                    f"fair_low={buy_reject_fair_low}, edge={buy_reject_edge})"
                )

            candidates.sort(reverse=True, key=lambda x: x[0])

            # Initialize risk tracking with live positions for accuracy
            current_risk = compute_current_risk_dollars(conn, live_pos, source="both")
            pending_orders = {}  # Track orders placed in this loop: values may be (direction, qty, px, accounted)
            validated = 0
            
            # Calculate balance-aware position scaling
            balance_scale = 1.0
            if bal_cents is not None:
                _, _, balance_scale = balance_aware_target(bal_cents)
                if balance_scale < 1.0:
                    available = (bal_cents - MIN_BALANCE_CENTS) / 100.0
                    logger.info(
                        f"BALANCE_AWARE_SIZING bal=${bal_cents/100:.2f} available=${available:.2f} "
                        f"max_risk=${MAX_TOTAL_RISK_DOLLARS:.2f} scale={balance_scale:.2f}x"
                    )
            
            logger.info(f"RISK_INIT position_risk=${current_risk:.2f} live_pos_count={len(live_pos)}")

            for edge_eff, series, direction, ticker, snap_bid, snap_ask, fair, spread, cur_qty, cur_avg in candidates:
                if placed_live >= MAX_LIVE_ORDERS_PER_LOOP:
                    break

                if direction == "SELL" and placed_live_sell >= MAX_SELL_ORDERS_PER_LOOP:
                    continue
                if direction == "BUY" and placed_live_buy >= MAX_BUY_ORDERS_PER_LOOP:
                    continue
                if direction == "BUY" and block_new_buys:
                    if SELL_DIAGNOSTICS:
                        logger.info(
                            f"BUY_DIAG {ticker} edge={edge_eff:.2f}c: BLOCKED_BY_RISK_BREAKER "
                            f"reasons={'|'.join(block_reasons)}"
                        )
                    continue
                if direction == "BUY" and HOURLY_FILTER_ENABLED and now_dt.hour in bad_hours:
                    if SELL_DIAGNOSTICS:
                        logger.info(
                            f"BUY_DIAG {ticker} edge={edge_eff:.2f}c: BLOCKED_BY_HOURLY_FILTER "
                            f"hour_utc={now_dt.hour}"
                        )
                    continue
                if direction == "BUY" and MAX_TRADES_PER_TICKER_PER_DAY > 0:
                    tcount = int(ticker_day_trade_counts.get(str(ticker), 0))
                    if tcount >= MAX_TRADES_PER_TICKER_PER_DAY:
                        if SELL_DIAGNOSTICS:
                            logger.info(
                                f"BUY_DIAG {ticker} edge={edge_eff:.2f}c: TICKER_DAILY_CAP "
                                f"{tcount}/{MAX_TRADES_PER_TICKER_PER_DAY}"
                            )
                        continue
                if direction == "BUY" and SERIES_EXPECTANCY_FILTER_ENABLED:
                    sstats = series_expectancy.get(str(series))
                    if sstats is not None:
                        exp = float(sstats.get("expectancy", 0.0))
                        scnt = int(sstats.get("count", 0))
                        if scnt >= SERIES_EXPECTANCY_MIN_CLOSED_TRADES and exp < SERIES_EXPECTANCY_MIN_EDGE_DOLLARS:
                            if SELL_DIAGNOSTICS:
                                logger.info(
                                    f"BUY_DIAG {ticker} edge={edge_eff:.2f}c: SERIES_EXPECTANCY_FILTER "
                                    f"series={series} expectancy=${exp:.4f}/trade count={scnt}"
                                )
                            continue
                if LIVE_TRADING and used_today >= MAX_LIVE_ORDERS_PER_DAY:
                    break

                last_ts = get_last_trade_ts(conn, ticker)
                if last_ts:
                    dt = (parse_iso(ts) or now_dt) - (parse_iso(last_ts) or now_dt)
                    if dt.total_seconds() < COOLDOWN_SECONDS:
                        continue


                if direction == "SELL" and cur_qty <= 0:
                    continue

                if direction == "BUY":
                    # Scale position targets based on available balance
                    if bal_cents is not None:
                        scaled_base, scaled_max, balance_scale = balance_aware_target(bal_cents)
                        target = target_for_edge(float(edge_eff), position_mult)
                        # Scale the target by balance factor
                        target = max(1, int(round(target * balance_scale)))
                        if balance_scale < 1.0:
                            logger.debug(
                                f"BALANCE_SCALING bal=${bal_cents/100:.2f} available=$"
                                f"{(bal_cents - MIN_BALANCE_CENTS)/100:.2f} scale={balance_scale:.2f}x "
                                f"target_raw={int(target_for_edge(float(edge_eff), position_mult))} "
                                f"target_scaled={target}"
                            )
                    else:
                        target = target_for_edge(float(edge_eff), position_mult)
                    
                    if cur_qty >= target:
                        if SELL_DIAGNOSTICS:
                            logger.info(f"BUY_DIAG {ticker} edge={edge_eff:.2f}c: Already at target (cur={cur_qty} >= tgt={target})")
                        continue
                    remaining = target - cur_qty
                    qty_step = _dynamic_trade_step(float(edge_eff))
                    qty = int(min(qty_step, remaining))
                    if qty <= 0:
                        continue
                    
                    # Check correlation hedging
                    if CORRELATION_HEDGING_ENABLED:
                        try:
                            corr_exposure = compute_correlated_exposure(conn, live_pos, lookback_hours=24)
                            ticker_series = ticker.split("_")[0]
                            if ticker_series in corr_exposure:
                                # Scale max correlation exposure based on available balance
                                base_max_corr = MAX_TOTAL_RISK_DOLLARS * MAX_CORRELATED_EXPOSURE_PCT / 100.0
                                if bal_cents is not None:
                                    available_dollars = (bal_cents - MIN_BALANCE_CENTS) / 100.0
                                    if available_dollars < MAX_TOTAL_RISK_DOLLARS:
                                        scale = available_dollars / MAX_TOTAL_RISK_DOLLARS
                                        max_corr_dollars = base_max_corr * scale
                                    else:
                                        max_corr_dollars = base_max_corr
                                else:
                                    max_corr_dollars = base_max_corr
                                
                                if corr_exposure[ticker_series] > max_corr_dollars:
                                    logger.warning(
                                        f"CORRELATION_LIMIT {ticker_series} "
                                        f"corr_exposure=${corr_exposure[ticker_series]:.2f} "
                                        f"> max=${max_corr_dollars:.2f}, skipping BUY"
                                    )
                                    continue
                        except Exception as e:
                            logger.debug(f"Correlation check error: {e}")
                    
                    live_qty = min(qty, LIVE_QTY_CAP)
                else:
                    qty_step = _dynamic_trade_step(float(edge_eff))
                    qty = int(min(qty_step, cur_qty))
                    if qty <= 0:
                        continue
                    live_qty = min(qty, LIVE_QTY_CAP)

                if LIVE_TRADING and live_qty <= 0:
                    continue

                if abs(cur_qty + (live_qty if direction == "BUY" else -live_qty)) > MAX_ABS_QTY_PER_TICKER:
                    continue

                if direction == "BUY":
                    incr_risk = live_qty * per_contract_worst_loss_dollars("BUY", int(snap_ask))
                    # Calculate pending buy orders already tracked in this loop,
                    # but exclude any that were already applied to current_risk
                    total_pend_buy_risk = 0.0
                    for _, payload in pending_orders.items():
                        if len(payload) == 4:
                            d, qty, px, accounted = payload
                        else:
                            d, qty, px = payload
                            accounted = False
                        if d == "BUY" and not accounted:
                            total_pend_buy_risk += qty * per_contract_worst_loss_dollars("BUY", int(px))
                    if (current_risk + incr_risk + total_pend_buy_risk) > MAX_TOTAL_RISK_DOLLARS:
                        if SELL_DIAGNOSTICS:
                            logger.info(
                                f"BUY_DIAG {ticker} edge={edge_eff:.2f}c: RISK_LIMIT curr=${current_risk:.2f} + incr=${incr_risk:.2f} + pend=${total_pend_buy_risk:.2f} > max=${MAX_TOTAL_RISK_DOLLARS}"
                            )
                        continue

                ok = edge_persistence_ok(conn, series, ticker, direction, float(edge_eff), ts, PERSIST_SECONDS)
                if not ok:
                    if SELL_DIAGNOSTICS and direction == "BUY":
                        logger.info(f"BUY_DIAG {ticker} edge={edge_eff:.2f}c: PERSISTENCE_CHECK failed (< {PERSIST_SECONDS}s)")
                    if LOG_CANDIDATE_REJECTION and direction == "SELL":
                        logger.debug(f"REJECT_PERSIST {ticker} {direction} edge={edge_eff:.2f}c, needs {PERSIST_SECONDS}s")
                    continue

                ob_bid = None
                ob_ask = None
                ob_bid_sz = 0
                ob_ask_sz = 0

                if validated < ORDERBOOK_VALIDATE_TOPN or MAKER_MODE:
                    try:
                        ob = safe_api_call("orderbook", client.get_orderbook, ticker)
                        yb, yb_sz, ya, ya_sz = get_best_yes_bid_ask(ob)
                        ob_bid = yb
                        ob_ask = ya
                        ob_bid_sz = int(yb_sz)
                        ob_ask_sz = int(ya_sz)
                    except ServicePaused as e:
                        logger.warning(f"Orderbook API paused while fetching {ticker}: {e}")
                        ob_bid = None
                        ob_ask = None
                        ob_bid_sz = 0
                        ob_ask_sz = 0
                    except Exception as e:
                        logger.error(f"Error fetching orderbook for {ticker}: {repr(e)}")
                        ob_bid = None
                        ob_ask = None
                        ob_bid_sz = 0
                        ob_ask_sz = 0
                    validated += 1

                if not MAKER_MODE:
                    if direction == "BUY" and ob_ask is None:
                        continue
                    if direction == "SELL" and ob_bid is None:
                        continue

                if ob_bid is not None and ob_ask is not None:
                    spr = int(ob_ask) - int(ob_bid)
                    if spr < 0 or spr > MAX_SPREAD_CENTS:
                        continue
                
                # Check concentration limits before proceeding further
                conc_ok, conc_reason = check_concentration_limits(
                    conn, live_pos, series, event_ticker, live_qty, float(fair), direction
                )
                if not conc_ok:
                    if SELL_DIAGNOSTICS:
                        logger.info(f"BUY_DIAG {ticker} edge={edge_eff:.2f}c: CONCENTRATION_LIMIT: {conc_reason}")
                    continue

                # Validate liquidity at the price we're targeting for execution
                if direction == "BUY" and ob_ask is not None:
                    bid_depth = get_orderbook_depth(ob, direction, int(ob_ask), levels=3)
                    if bid_depth < int(live_qty):
                        if SELL_DIAGNOSTICS:
                            logger.info(f"BUY_DIAG {ticker} edge={edge_eff:.2f}c: INSUFFICIENT_LIQUIDITY at ask={ob_ask} depth={bid_depth} < qty={live_qty}")
                        continue
                if direction == "SELL" and ob_bid is not None:
                    ask_depth = get_orderbook_depth(ob, direction, int(ob_bid), levels=3)
                    if ask_depth < int(live_qty):
                        logger.info(f"INSUFFICIENT_LIQUIDITY {ticker} SELL at {ob_bid} depth={ask_depth} < qty={live_qty}")
                        continue

                if direction == "BUY" and ob_ask is not None and int(ob_ask_sz) < MIN_LIQUIDITY_SZ:
                    continue
                if direction == "SELL" and ob_bid is not None and int(ob_bid_sz) < MIN_LIQUIDITY_SZ:
                    continue
                
                # Check queue depth for maker mode reliability
                queue_depth = 0
                if MAKER_MODE:
                    queue_depth = get_orderbook_depth(ob, direction, int(ob_ask) if direction == "BUY" else int(ob_bid), levels=5)
                    if queue_depth < MIN_QUEUE_DEPTH_FOR_MAKER:
                        if SELL_DIAGNOSTICS:
                            logger.info(f"QUEUE_CHECK {ticker} dir={direction} depth={queue_depth} < min={MIN_QUEUE_DEPTH_FOR_MAKER}")
                        continue

                if direction == "BUY" and ob_ask is not None:
                    if abs(int(ob_ask) - int(snap_ask)) > STALE_QUOTE_CENTS:
                        if SELL_DIAGNOSTICS:
                            logger.info(f"BUY_DIAG {ticker} edge={edge_eff:.2f}c: STALE_QUOTE ask_moved={abs(int(ob_ask) - int(snap_ask))}c > {STALE_QUOTE_CENTS}c")
                        continue
                if direction == "SELL" and ob_bid is not None:
                    if abs(int(ob_bid) - int(snap_bid)) > STALE_QUOTE_CENTS:
                        logger.info(f"STALE_QUOTE {ticker} SELL bid_move={abs(int(ob_bid) - int(snap_bid))}c > {STALE_QUOTE_CENTS}c")
                        continue
                
                # Check orderbook freshness by timestamp AND price movement
                ob_fresh, ob_age, ob_reason = check_orderbook_freshness(ob, now_dt, MAX_ORDERBOOK_AGE_SECONDS)
                if not ob_fresh and REQUIRE_FRESH_ORDERBOOK:
                    if SELL_DIAGNOSTICS:
                        logger.info(f"STALE_OB_TIMESTAMP {ticker} {direction} age={ob_age} reason={ob_reason}")
                    continue

                px = None
                if MAKER_MODE:
                    if direction == "BUY":
                        if ob_bid is not None:
                            px = int(ob_bid) + MAKER_IMPROVE_CENTS
                        else:
                            px = int(round(float(fair) - float(COST_BUFFER_CENTS)))

                        fair_cap = int(math.floor(float(fair) - float(COST_BUFFER_CENTS)))
                        if fair_cap < 1:
                            continue
                        if px > fair_cap:
                            px = fair_cap

                        if ob_ask is not None:
                            if int(ob_ask) <= 1:
                                continue
                            if px >= int(ob_ask):
                                px = int(ob_ask) - 1
                        if direction == "BUY" and ob_ask is not None and float(edge_eff) >= AGGRESSIVE_EDGE_CENTS:
                            px = int(ob_ask)

                    else:
                        if ob_bid is None:
                            continue
                        px = int(ob_bid)
                else:
                    px = int(ob_ask) if direction == "BUY" else int(ob_bid)

                if px is None or px < 1 or px > 99:
                    continue

                if direction == "BUY":
                    incr_risk = live_qty * per_contract_worst_loss_dollars("BUY", px)
                    # Recompute pending BUY risk excluding buys already accounted
                    total_pend_buy_risk = 0.0
                    for _, payload in pending_orders.items():
                        if len(payload) == 4:
                            d, qty, px_pend, accounted = payload
                        else:
                            d, qty, px_pend = payload
                            accounted = False
                        if d == "BUY" and not accounted:
                            total_pend_buy_risk += qty * per_contract_worst_loss_dollars("BUY", int(px_pend))
                    if (current_risk + incr_risk + total_pend_buy_risk) > MAX_TOTAL_RISK_DOLLARS:
                        continue

                if LIVE_TRADING:
                    if bal_cents is not None:
                        # Calculate total balance needed for pending orders. Include all
                        # pending BUYs regardless of whether their risk was already
                        # accounted in current_risk because balance is separate.
                        pending_balance_needed = 0
                        for _, payload in pending_orders.items():
                            if len(payload) == 4:
                                d, qty, px_pend, _ = payload
                            else:
                                d, qty, px_pend = payload
                            if d == "BUY":
                                pending_balance_needed += qty * px_pend
                        balance_needed = live_qty * px + pending_balance_needed
                        if balance_needed > bal_cents:
                            logger.info(
                                f"INSUFFICIENT_BALANCE {ticker} needed={balance_needed}c (qty={live_qty}@{px} + "
                                f"pending={pending_balance_needed}c) > available={bal_cents}c"
                            )
                            continue

                    action = "buy" if direction == "BUY" else "sell"
                    reduce_only_flag = True if direction == "SELL" else None
                    time_in_force = "immediate_or_cancel" if direction == "SELL" else None

                    client_order_id = make_client_order_id(ticker)

                    final_post_only = True if (MAKER_MODE and direction == "BUY") else None
                    if direction == "BUY" and ob_ask is not None and float(edge_eff) >= AGGRESSIVE_EDGE_CENTS:
                        final_post_only = None

                    if direction == "SELL":
                        final_post_only = None
                        time_in_force = "immediate_or_cancel"
                        if ob_bid is None:
                            continue
                        px = int(ob_bid)
                        if px < 1 or px > 99:
                            continue

                    if MAKER_MODE and direction == "BUY":
                        # Double-check queue depth hasn't collapsed
                        try:
                            ob2 = safe_api_call("orderbook", client.get_orderbook, ticker)
                            yb2, _, ya2, _ = get_best_yes_bid_ask(ob2)
                            
                            if ya2 is not None:
                                depth2 = get_orderbook_depth(ob2, "BUY", int(ya2), levels=3)
                                if depth2 < MIN_QUEUE_DEPTH_FOR_MAKER:
                                    # Queue collapsed, skip this order
                                    continue
                            
                            fa = ya2
                        except ServicePaused:
                            fa = None
                        except Exception:
                            fa = None

                        if fa is not None and px >= int(fa):
                            if int(fa) > 1:
                                px = int(fa) - 1
                            else:
                                final_post_only = None
                                px = int(fa)

                    if px < 1 or px > 99:
                        continue

                    used_client_order_id = client_order_id

                    try:
                        if direction == "BUY":
                            resp = client.create_order(
                                ticker=ticker,
                                side="yes",
                                action="buy",
                                count=live_qty,
                                type_="limit",
                                yes_price=px,
                                client_order_id=client_order_id,
                                post_only=final_post_only,
                                reduce_only=None,
                                time_in_force=None,
                            )
                        else:
                            resp = client.create_order(
                                ticker=ticker,
                                side="yes",
                                action="sell",
                                count=live_qty,
                                type_="limit",
                                yes_price=px,
                                client_order_id=client_order_id,
                                post_only=None,
                                reduce_only=True,
                                time_in_force="immediate_or_cancel",
                            )

                    except Exception as e:
                        msg_l = str(e).lower()

                        # Only attempt one fallback from post-only to taker
                        if direction == "BUY" and MAKER_MODE and final_post_only and ("post only cross" in msg_l):
                            used_client_order_id = f"{client_order_id}-taker"
                            logger.info(
                                f"POST_ONLY_CROSS {series} {ticker} px={px} qty={live_qty}. "
                                f"Falling back to taker. REBATE LOST."
                            )
                            try:
                                resp = client.create_order(
                                    ticker=ticker,
                                    side="yes",
                                    action=action,
                                    count=live_qty,
                                    type_="limit",
                                    yes_price=px,
                                    client_order_id=used_client_order_id,
                                    post_only=None,
                                    reduce_only=reduce_only_flag,
                                    time_in_force=time_in_force,
                                )
                            except Exception as e2:
                                msg2_l = str(e2).lower()
                                if "order_already_exists" in msg2_l:
                                    resp = {"order": {"client_order_id": used_client_order_id, "status": "already_exists"}}
                                else:
                                    logger.error(f"Error placing fallback taker order for {ticker}: {repr(e2)}")
                                    raise
                        elif "order_already_exists" in msg_l:
                            resp = {"order": {"client_order_id": used_client_order_id, "status": "already_exists"}}
                        else:
                            logger.error(f"Error placing order for {ticker}: {repr(e)}")
                            raise

                    order_obj = resp.get("order") if isinstance(resp, dict) else None
                    order_id = order_obj.get("order_id") if isinstance(order_obj, dict) else None
                    status = order_obj.get("status") if isinstance(order_obj, dict) else None

                    log_live_order(
                        conn,
                        ts,
                        ticker=ticker,
                        side="yes",
                        action=action,
                        yes_price=px,
                        count=live_qty,
                        order_id=order_id,
                        client_order_id=used_client_order_id,
                        status=status,
                        raw_json=str(resp),
                    )

                    filled = 0
                    if isinstance(order_obj, dict):
                        try:
                            filled = int(order_obj.get("fill_count") or 0)
                        except Exception:
                            filled = 0

                    # Validate fill amount
                    if filled > 0:
                        fill_pct = float(filled) / float(live_qty) if live_qty > 0 else 0
                        if fill_pct < float(FILL_VALIDATION_TOLERANCE):
                            logger.warning(
                                f"PARTIAL_FILL {series} {action} {ticker} px={px} requested={live_qty} filled={filled} "
                                f"fill_pct={fill_pct:.2%} (below {FILL_VALIDATION_TOLERANCE:.2%} threshold)"
                            )

                    if filled <= 0:
                        logger.info(f"LIVE {series} {action.upper():4} YES {ticker} px={px} qty={live_qty} -> NO_FILL status={status}")
                        clear_persistence(conn, series, ticker, direction)
                        continue

                    exec_qty = min(int(filled), int(live_qty))
                    
                    # Sanity check: filled should not exceed requested
                    if exec_qty > live_qty:
                        logger.error(
                            f"FILL_OVERFLOW {series} {action} {ticker} requested={live_qty} filled={exec_qty}. "
                            f"Capping to requested amount."
                        )
                        exec_qty = live_qty

                    # Persist live fills in trades so downstream analytics/dashboard stay in sync.
                    record_trade(
                        conn,
                        ts,
                        ticker,
                        direction,
                        int(px),
                        int(exec_qty),
                        note=(
                            f"mode=live action={action} status={status or ''} "
                            f"order_id={order_id or ''} client_order_id={used_client_order_id or ''}"
                        ).strip(),
                    )
                    ticker_day_trade_counts[str(ticker)] = int(ticker_day_trade_counts.get(str(ticker), 0)) + 1

                    set_state(conn, f"live_orders_{day_key}", str(used_today + 1))
                    used_today += 1

                    new_qty = cur_qty + exec_qty if direction == "BUY" else cur_qty - exec_qty
                    new_avg = float(cur_avg)
                    if direction == "BUY":
                        new_avg = (cur_qty * cur_avg + exec_qty * px) / new_qty if new_qty != 0 else 0.0

                    upsert_position(conn, ticker, new_qty, new_avg)
                    set_last_trade_ts(conn, ticker, ts)
                    clear_persistence(conn, series, ticker, direction)

                    placed += 1
                    placed_live += 1
                    if direction == "SELL":
                        placed_live_sell += 1
                    else:
                        placed_live_buy += 1

                    # Track pending order for risk accounting in this loop
                    # Include an `accounted` flag so we can avoid double-counting
                    # pending BUY risk when current_risk was already incremented.
                    pending_orders[ticker] = (direction, exec_qty, px, True if direction == "BUY" else False)
                    
                    # Update current_risk: for BUY increment, for SELL recalculate from updated positions
                    if direction == "BUY":
                        current_risk += exec_qty * per_contract_worst_loss_dollars("BUY", px)
                        logger.info(f"RISK_AFTER_BUY {ticker} qty={exec_qty} px={px} new_total=${current_risk:.2f}")
                    else:
                        # For SELL, recalculate to get accurate updated risk
                        updated_pos = dict(live_pos) if live_pos else {}
                        updated_pos[ticker] = (cur_qty - exec_qty, cur_avg)  # Update position
                        current_risk = compute_current_risk_dollars(conn, updated_pos, source="live")
                        logger.info(f"RISK_AFTER_SELL {ticker} qty={exec_qty} px={px} new_total=${current_risk:.2f}")


                    spr_txt = "NA"
                    if ob_bid is not None and ob_ask is not None:
                        spr_txt = str(int(ob_ask) - int(ob_bid))

                    logger.info(
                        f"LIVE {series} {action.upper():4} YES {ticker} px={px} qty={exec_qty} "
                        f"eff_edge={edge_eff:.1f} fair={fair:.1f} spr={spr_txt} "
                        f"bid={ob_bid} ask={ob_ask} bsz={ob_bid_sz} asz={ob_ask_sz}"
                    )
                    
                    track_order_attempt(conn, series, ticker, direction, ts)

                else:
                    record_trade(
                        conn,
                        ts,
                        ticker,
                        direction,
                        int(px),
                        int(live_qty),
                        note=f"mode={'maker' if MAKER_MODE else 'cross'} eff_edge={edge_eff:.1f} fair={fair:.1f}",
                    )
                    ticker_day_trade_counts[str(ticker)] = int(ticker_day_trade_counts.get(str(ticker), 0)) + 1
                    new_qty = cur_qty + int(live_qty) if direction == "BUY" else cur_qty - int(live_qty)
                    new_avg = float(cur_avg)
                    if direction == "BUY":
                        new_avg = (cur_qty * cur_avg + int(live_qty) * int(px)) / new_qty if new_qty != 0 else 0.0
                    upsert_position(conn, ticker, new_qty, new_avg)
                    set_last_trade_ts(conn, ticker, ts)
                    clear_persistence(conn, series, ticker, direction)
                    placed += 1
                    logger.info(f"PAPER {series} {direction:4} YES {ticker} px={int(px)} qty={int(live_qty)} eff_edge={edge_eff:.1f} pos={new_qty}@{new_avg:.1f}")

            next_sleep = FAST_EVERY_SECONDS if placed_live > 0 else RUN_EVERY_SECONDS
            
            # Calculate risk breakdown including pending orders
            total_risk, position_risk, pending_risk = compute_risk_with_pending_orders(conn, live_pos, pending_orders)
            pending_count = 0
            for _, payload in pending_orders.items():
                if len(payload) == 4:
                    d, _, _, _ = payload
                else:
                    d, _, _ = payload
                if d == "BUY":
                    pending_count += 1
            
            logger.info(
                f"LOOP_DONE series={len(SERIES_LIST)} placed={placed} "
                f"livePlaced={placed_live} (sell={placed_live_sell}, buy={placed_live_buy}) "
                f"usedToday={used_today} sleep={next_sleep}s "
                f"risk_total=${total_risk:.2f} (pos=${position_risk:.2f} pend=${pending_risk:.2f} pend_orders={pending_count})"
            )

        except KeyboardInterrupt:
            logger.info("Stopping bot...")
            break
        except Exception as e:
            logger.error(f"ERROR in main loop: {repr(e)}")
            logger.error(traceback.format_exc())
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                logger.critical("Too many consecutive errors, exiting")
                break

        time.sleep(next_sleep)


if __name__ == "__main__":
    main()
