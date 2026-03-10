"""
Hybrid Regime-Switching Trading Bot v6
=======================================
- Trend-following LONG when ADX > 25 (momentum + breakout)
- Trend-following SHORT when ADX > 25 (breakdown + bearish momentum)
- Mean-reversion long when ADX < 20 (Bollinger Band + RSI)
- Oversold bounce entries for ranging stocks (RSI < 45)
- Conservative in neutral zone (ADX 20-25)
- ATR-based dynamic stops, trailing stops, time stops
- Volume confirmation on all entries
- Daily equity curve tracking
- Max daily drawdown kill switch
- Bear market capable: short-selling support for downtrending markets
"""

import os
import math
import json
import time
import logging
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = SCRIPT_DIR / "config" / ".env"
LOG_DIR = SCRIPT_DIR / "logs"
TRADE_LOG_FILE = LOG_DIR / "trade_log.json"
EQUITY_LOG_FILE = LOG_DIR / "equity_curve.csv"
BOT_LOG_FILE = LOG_DIR / "bot.log"

WATCHLIST = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA",
             "GOOG", "AMD", "NFLX", "AVGO"]

# Risk parameters
RISK_PER_TRADE = 0.015           # 1.5% of equity risked per trade
MAX_POSITION_PCT = 0.10          # Max 10% of equity in single position
MAX_OPEN_POSITIONS = 6
MAX_DAILY_LOSS_PCT = 0.03        # 3% daily drawdown kill switch
ATR_STOP_MULTIPLIER_TREND = 2.0  # 2x ATR stop for trend trades
ATR_STOP_MULTIPLIER_RANGE = 1.5  # 1.5x ATR stop for range trades
ATR_TARGET_MULTIPLIER_TREND = 4.0  # 4x ATR target for trend (2:1 R:R)
ATR_TARGET_MULTIPLIER_RANGE = 2.5  # 2.5x ATR target for range (~1.7:1 R:R)
TRAILING_ACTIVATION_R = 1.0      # Activate trailing stop after 1R profit
TRAILING_ATR_MULTIPLIER = 2.5    # Trail at 2.5x ATR once activated
MAX_HOLD_DAYS = 15               # Time stop: exit after 15 trading days
MIN_PRICE = 15.0

# Regime thresholds
ADX_TREND_THRESHOLD = 25         # ADX > 25 = trending
ADX_RANGE_THRESHOLD = 20         # ADX < 20 = ranging

# Volatility filter
MIN_ATR_PCT = 0.003              # Min ATR/price ratio (0.3%) - skip dead markets
MAX_ATR_PCT = 0.06               # Max ATR/price ratio (6%) - skip chaos

# Volume filter (lowered from 1.1 — bear markets often see vol_ratios 0.3-0.7)
VOLUME_CONFIRMATION_MULT = 0.8   # Volume must be > 0.8x 20-day average

# Indicator periods
RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
EMA_FAST = 9
EMA_SLOW = 21
SMA_TREND = 50
SMA_LONG = 200
VOLUME_MA_PERIOD = 20

# Bollinger Band entry distance (widened from 0.5% — was too tight)
BB_DISTANCE_PCT = 0.02           # Price within 2% of lower/upper BB to qualify

# Mean-reversion volume threshold (lower than trend — less volume in ranging)
MEAN_REVERSION_VOL_MULT = 0.6   # Lowered from 1.0 for bear market conditions

# Short-selling parameters
ATR_STOP_MULTIPLIER_SHORT = 2.0  # 2x ATR stop for short trades
ATR_TARGET_MULTIPLIER_SHORT = 3.5  # 3.5x ATR target for short trades
SHORT_RSI_MIN = 30               # Short entry: RSI floor (avoid shorting extremes)
SHORT_RSI_MAX = 60               # Short entry: RSI ceiling

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2


# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("trading_bot")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(BOT_LOG_FILE)
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


log = setup_logging()


# ============================================================
# UTILITIES
# ============================================================

def ensure_directories():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_environment():
    load_dotenv(str(ENV_PATH))


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_utc_iso() -> str:
    return now_utc().isoformat()


def utc_today_str() -> str:
    return now_utc().date().isoformat()


# ============================================================
# TRADE LOG PERSISTENCE
# ============================================================

def load_trade_log() -> list:
    ensure_directories()
    if not TRADE_LOG_FILE.exists():
        return []
    try:
        with open(TRADE_LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_trade_log(data: list) -> None:
    ensure_directories()
    with open(TRADE_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def append_trade_log(entry: dict) -> None:
    data = load_trade_log()
    data.append(entry)
    save_trade_log(data)


# ============================================================
# EQUITY CURVE TRACKING
# ============================================================

def log_equity_curve(equity: float, positions_count: int) -> None:
    ensure_directories()
    header_needed = not EQUITY_LOG_FILE.exists()
    with open(EQUITY_LOG_FILE, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("timestamp,date,equity,positions_count\n")
        f.write(f"{now_utc_iso()},{utc_today_str()},{equity:.2f},{positions_count}\n")


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI using exponential moving average."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index with +DI and -DI."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    alpha = 1.0 / period
    atr_smooth = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    plus_di = 100 * plus_dm_smooth / atr_smooth.replace(0, np.nan)
    minus_di = 100 * minus_dm_smooth / atr_smooth.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    result = pd.DataFrame({
        "plus_di": plus_di,
        "minus_di": minus_di,
        "adx": adx,
    }, index=df.index)
    return result


def compute_bollinger_bands(series: pd.Series, period: int = 20,
                            std_mult: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands."""
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return pd.DataFrame({
        "bb_upper": sma + std_mult * std,
        "bb_middle": sma,
        "bb_lower": sma - std_mult * std,
        "bb_width": (std_mult * 2 * std) / sma.replace(0, np.nan),
    }, index=series.index)


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators on a daily OHLCV DataFrame."""
    df = df.copy()

    # Moving averages
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["sma50"] = df["close"].rolling(SMA_TREND).mean()
    df["sma200"] = df["close"].rolling(SMA_LONG).mean()

    # RSI
    df["rsi"] = compute_rsi(df["close"], RSI_PERIOD)

    # ATR
    df["atr"] = compute_atr(df, ATR_PERIOD)
    df["atr_pct"] = df["atr"] / df["close"]

    # ADX
    adx_df = compute_adx(df, ADX_PERIOD)
    df["adx"] = adx_df["adx"]
    df["plus_di"] = adx_df["plus_di"]
    df["minus_di"] = adx_df["minus_di"]

    # Bollinger Bands
    bb_df = compute_bollinger_bands(df["close"], BB_PERIOD, BB_STD)
    df["bb_upper"] = bb_df["bb_upper"]
    df["bb_middle"] = bb_df["bb_middle"]
    df["bb_lower"] = bb_df["bb_lower"]
    df["bb_width"] = bb_df["bb_width"]

    # Volume moving average
    df["volume_sma"] = df["volume"].rolling(VOLUME_MA_PERIOD).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"].replace(0, np.nan)

    # Price rate of change (momentum)
    df["roc_5"] = df["close"].pct_change(5) * 100
    df["roc_10"] = df["close"].pct_change(10) * 100

    return df


# ============================================================
# REGIME DETECTION
# ============================================================

def detect_regime(row: pd.Series) -> str:
    """Classify market regime based on ADX.
    Returns: 'trending', 'ranging', or 'neutral'
    """
    adx = row.get("adx", np.nan)
    if pd.isna(adx):
        return "neutral"
    if adx >= ADX_TREND_THRESHOLD:
        return "trending"
    if adx <= ADX_RANGE_THRESHOLD:
        return "ranging"
    return "neutral"


def passes_volatility_filter(row: pd.Series) -> tuple[bool, str]:
    """Check if volatility is in tradeable range."""
    atr_pct = row.get("atr_pct", np.nan)
    if pd.isna(atr_pct):
        return False, "ATR data unavailable"
    if atr_pct < MIN_ATR_PCT:
        return False, f"Volatility too low (ATR%={atr_pct:.4f})"
    if atr_pct > MAX_ATR_PCT:
        return False, f"Volatility too high (ATR%={atr_pct:.4f})"
    return True, "Volatility OK"


def has_volume_confirmation(row: pd.Series) -> bool:
    """Check if current volume exceeds threshold vs. average."""
    vr = row.get("volume_ratio", np.nan)
    if pd.isna(vr):
        return False
    return vr >= VOLUME_CONFIRMATION_MULT


# ============================================================
# ENTRY SIGNALS
# ============================================================

def signal_trend_long(df: pd.DataFrame) -> tuple[bool, str]:
    """Trend-following long signal (ADX > 25 regime).

    Confluence requirements:
    1. Price > SMA200 (long-term uptrend)
    2. EMA9 > EMA21 (short-term momentum)
    3. +DI > -DI (directional strength confirms bullish)
    4. RSI between 40-70 (momentum without overbought)
    5. Volume confirmation
    6. Close above prior 5-bar high (breakout)
    """
    if len(df) < SMA_LONG + 5:
        return False, "Not enough data for trend signal"

    latest = df.iloc[-1]
    prev_bars = df.iloc[-6:-1]  # Last 5 bars before current

    # Check for NaN in critical indicators
    required = ["close", "ema_fast", "ema_slow", "sma200", "rsi",
                "adx", "plus_di", "minus_di", "atr"]
    for field in required:
        if pd.isna(latest.get(field)):
            return False, f"Missing indicator: {field}"

    if latest["close"] < MIN_PRICE:
        return False, "Price below minimum"

    # 1. Long-term trend filter
    if latest["close"] <= latest["sma200"]:
        return False, "Price below SMA200 — no uptrend"

    # 2. Short-term momentum
    if latest["ema_fast"] <= latest["ema_slow"]:
        return False, "EMA9 < EMA21 — no short-term momentum"

    # 3. Directional strength
    if latest["plus_di"] <= latest["minus_di"]:
        return False, "+DI < -DI — bears in control"

    # 4. RSI filter (wider than original)
    if not (40 <= latest["rsi"] <= 70):
        return False, f"RSI {latest['rsi']:.1f} outside 40-70 range"

    # 5. Volume confirmation
    if not has_volume_confirmation(latest):
        return False, f"Volume ratio {latest.get('volume_ratio', 0):.2f} below threshold"

    # 6. Breakout: close above highest high of prior 5 bars
    prior_high = prev_bars["high"].max()
    if latest["close"] <= prior_high:
        return False, f"No breakout — close {latest['close']:.2f} <= prior 5-bar high {prior_high:.2f}"

    return True, (f"TREND LONG: ADX={latest['adx']:.1f}, RSI={latest['rsi']:.1f}, "
                  f"+DI={latest['plus_di']:.1f}, EMA9>21, breakout above {prior_high:.2f}")


def signal_trend_short(df: pd.DataFrame) -> tuple[bool, str]:
    """Trend-following short signal (ADX > 25 regime, bearish).

    Confluence requirements:
    1. EMA9 < EMA21 (bearish momentum — primary downtrend filter)
    2. Price below recent 10-bar swing high (confirms downtrend structure)
    3. -DI > +DI (directional strength confirms bearish)
    4. RSI between SHORT_RSI_MIN-SHORT_RSI_MAX (momentum without oversold)
    5. Volume confirmation
    6. Close below prior 5-bar low (breakdown)
    """
    if len(df) < SMA_LONG + 5:
        return False, "Not enough data for short signal"

    latest = df.iloc[-1]
    prev_bars = df.iloc[-6:-1]

    required = ["close", "ema_fast", "ema_slow", "sma200", "rsi",
                "adx", "plus_di", "minus_di", "atr"]
    for field in required:
        if pd.isna(latest.get(field)):
            return False, f"Missing indicator: {field}"

    if latest["close"] < MIN_PRICE:
        return False, "Price below minimum"

    # 1. Bearish momentum: EMA9 < EMA21 (primary downtrend filter)
    if latest["ema_fast"] >= latest["ema_slow"]:
        return False, "EMA9 >= EMA21 — no bearish momentum for short"

    # 2. Price below recent swing high (10-bar high confirms downtrend structure)
    swing_bars = df.iloc[-11:-1]  # prior 10 bars
    recent_swing_high = swing_bars["high"].max()
    if latest["close"] >= recent_swing_high:
        return False, f"Price {latest['close']:.2f} >= swing high {recent_swing_high:.2f} — not in downtrend"

    # 3. Directional strength (bears)
    if latest["minus_di"] <= latest["plus_di"]:
        return False, "-DI <= +DI — bulls still in control"

    # 4. RSI filter (avoid shorting into oversold)
    if not (SHORT_RSI_MIN <= latest["rsi"] <= SHORT_RSI_MAX):
        return False, f"RSI {latest['rsi']:.1f} outside {SHORT_RSI_MIN}-{SHORT_RSI_MAX} range for short"

    # 5. Volume confirmation
    if not has_volume_confirmation(latest):
        return False, f"Volume ratio {latest.get('volume_ratio', 0):.2f} below threshold"

    # 6. Breakdown: close below lowest low of prior 5 bars
    prior_low = prev_bars["low"].min()
    if latest["close"] >= prior_low:
        return False, f"No breakdown — close {latest['close']:.2f} >= prior 5-bar low {prior_low:.2f}"

    return True, (f"TREND SHORT: ADX={latest['adx']:.1f}, RSI={latest['rsi']:.1f}, "
                  f"-DI={latest['minus_di']:.1f}, EMA9<21, breakdown below {prior_low:.2f}")


def signal_mean_reversion_long(df: pd.DataFrame) -> tuple[bool, str]:
    """Mean-reversion long signal (ADX < 20 regime).

    Confluence requirements:
    1. Price near or below lower Bollinger Band
    2. RSI < 45 (oversold for ranging regime)
    3. Price > SMA200 (only mean-revert in uptrend context)
    4. Volume confirmation (capitulation volume)
    5. Bounce: current close > current open (bullish candle)
    """
    if len(df) < SMA_LONG + 5:
        return False, "Not enough data for mean reversion signal"

    latest = df.iloc[-1]

    required = ["close", "open", "bb_lower", "bb_middle", "sma200", "rsi", "atr"]
    for field in required:
        if pd.isna(latest.get(field)):
            return False, f"Missing indicator: {field}"

    if latest["close"] < MIN_PRICE:
        return False, "Price below minimum"

    # 1. Price near lower BB (within BB_DISTANCE_PCT of lower band or below it)
    bb_distance = (latest["close"] - latest["bb_lower"]) / latest["bb_lower"]
    if bb_distance > BB_DISTANCE_PCT:
        return False, f"Price not near lower BB (distance={bb_distance:.4f}, need <{BB_DISTANCE_PCT})"

    # 2. RSI oversold (45 threshold catches more mean-reversion opportunities in ranging markets)
    if latest["rsi"] >= 45:
        return False, f"RSI {latest['rsi']:.1f} not oversold (need < 45)"

    # 3. Long-term context: only buy dips in uptrend
    if latest["close"] <= latest["sma200"] * 0.95:
        return False, "Price too far below SMA200 — potential breakdown"

    # 4. Volume: want some volume (lowered threshold for bear markets)
    vol_ratio = latest.get("volume_ratio", 0)
    if vol_ratio < MEAN_REVERSION_VOL_MULT:
        return False, f"Volume too low for capitulation ({vol_ratio:.2f}, need >={MEAN_REVERSION_VOL_MULT})"

    # 5. Bullish candle (close > open = buyers stepping in)
    if latest["close"] <= latest["open"]:
        return False, "No bullish candle — close <= open"

    return True, (f"MEAN REVERSION LONG: RSI={latest['rsi']:.1f}, "
                  f"BB_dist={bb_distance:.4f}, vol_ratio={vol_ratio:.2f}")


def signal_oversold_bounce(df: pd.DataFrame) -> tuple[bool, str]:
    """Oversold bounce signal for ranging stocks (ADX < 20).

    Lighter requirements than full mean-reversion — catches RSI < 35 bounces
    in stocks that aren't necessarily at the lower BB but are deeply oversold.

    Confluence requirements:
    1. RSI < 45 (oversold for ranging regime)
    2. RSI turned up: current RSI > prior bar RSI (momentum shifting)
    3. Price > SMA200 * 0.90 (not in freefall — within 10% of SMA200)
    4. Bullish candle (close > open)
    5. Minimum volume (not a dead market)
    """
    if len(df) < SMA_LONG + 5:
        return False, "Not enough data for oversold bounce signal"

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    required = ["close", "open", "sma200", "rsi", "atr"]
    for field in required:
        if pd.isna(latest.get(field)):
            return False, f"Missing indicator: {field}"

    if latest["close"] < MIN_PRICE:
        return False, "Price below minimum"

    # 1. RSI oversold (45 threshold for ranging markets)
    if latest["rsi"] >= 45:
        return False, f"RSI {latest['rsi']:.1f} not oversold (need < 45)"

    # 2. RSI turning up (momentum shift)
    if pd.isna(prev.get("rsi")) or latest["rsi"] <= prev["rsi"]:
        return False, f"RSI not turning up ({latest['rsi']:.1f} vs prior {prev.get('rsi', 0):.1f})"

    # 3. Not in total freefall — within 10% of SMA200
    if latest["close"] <= latest["sma200"] * 0.90:
        return False, "Price too far below SMA200 — potential crash, skip bounce"

    # 4. Bullish candle
    if latest["close"] <= latest["open"]:
        return False, "No bullish candle — close <= open"

    # 5. Minimum volume
    vol_ratio = latest.get("volume_ratio", 0)
    if vol_ratio < MEAN_REVERSION_VOL_MULT:
        return False, f"Volume too low ({vol_ratio:.2f})"

    return True, (f"OVERSOLD BOUNCE: RSI={latest['rsi']:.1f} (turning up from "
                  f"{prev['rsi']:.1f}), vol_ratio={vol_ratio:.2f}")


def signal_neutral_long(df: pd.DataFrame) -> tuple[bool, str]:
    """Conservative signal for neutral regime (ADX 20-25).

    Requires BOTH trend AND mean-reversion conditions partially met:
    1. Price > SMA200
    2. EMA9 > EMA21
    3. RSI between 35-55 (pullback within uptrend)
    4. Price within 1 ATR of SMA50 (pullback to key level)
    5. Volume confirmation
    6. Bullish candle
    """
    if len(df) < SMA_LONG + 5:
        return False, "Not enough data for neutral signal"

    latest = df.iloc[-1]

    required = ["close", "open", "ema_fast", "ema_slow", "sma50", "sma200",
                "rsi", "atr"]
    for field in required:
        if pd.isna(latest.get(field)):
            return False, f"Missing indicator: {field}"

    if latest["close"] < MIN_PRICE:
        return False, "Price below minimum"

    if latest["close"] <= latest["sma200"]:
        return False, "Price below SMA200"

    if latest["ema_fast"] <= latest["ema_slow"]:
        return False, "EMA9 < EMA21"

    if not (35 <= latest["rsi"] <= 55):
        return False, f"RSI {latest['rsi']:.1f} outside neutral sweet spot 35-55"

    # Price near SMA50 (within 1 ATR)
    distance_to_sma50 = abs(latest["close"] - latest["sma50"])
    if distance_to_sma50 > latest["atr"]:
        return False, "Price too far from SMA50 pullback zone"

    if not has_volume_confirmation(latest):
        return False, "No volume confirmation"

    if latest["close"] <= latest["open"]:
        return False, "No bullish candle"

    return True, (f"NEUTRAL LONG: RSI={latest['rsi']:.1f}, "
                  f"near SMA50, EMA9>21, bullish candle")


def evaluate_entry(df: pd.DataFrame) -> tuple[bool, str, str, str]:
    """Master entry evaluation. Returns (signal, reason, regime, direction).
    direction is 'long' or 'short'.
    """
    if len(df) < SMA_LONG + 5:
        return False, "Insufficient data", "unknown", "long"

    latest = df.iloc[-1]
    regime = detect_regime(latest)

    # Volatility gate — applies to ALL regimes
    vol_ok, vol_reason = passes_volatility_filter(latest)
    if not vol_ok:
        return False, vol_reason, regime, "long"

    if regime == "trending":
        # Determine bias from EMA crossover for correct direction reporting
        ema_bearish = latest.get("ema_fast", 0) < latest.get("ema_slow", 0)
        if ema_bearish:
            # Bearish bias — evaluate short signal only
            signal, reason = signal_trend_short(df)
            return signal, reason, regime, "short"
        else:
            # Bullish bias — evaluate long signal only
            signal, reason = signal_trend_long(df)
            return signal, reason, regime, "long"

    elif regime == "ranging":
        # Try mean-reversion long first
        signal, reason = signal_mean_reversion_long(df)
        if signal:
            return True, reason, regime, "long"
        # Try oversold bounce
        signal, reason = signal_oversold_bounce(df)
        if signal:
            return True, reason, regime, "long"
        return False, reason, regime, "long"

    else:
        signal, reason = signal_neutral_long(df)
        return signal, reason, regime, "long"


# ============================================================
# POSITION SIZING
# ============================================================

def position_size_from_atr(
    equity: float,
    entry_price: float,
    atr: float,
    stop_multiplier: float,
    risk_per_trade: float,
    max_position_pct: float,
    direction: str = "long"
) -> tuple[int, float]:
    """Calculate position size based on ATR stop distance.

    Returns (qty, stop_price).
    For shorts, stop is placed ABOVE entry price.
    """
    stop_distance = atr * stop_multiplier
    if stop_distance <= 0 or entry_price <= 0:
        return 0, 0.0

    risk_dollars = equity * risk_per_trade
    qty = math.floor(risk_dollars / stop_distance)

    # Enforce max position size
    max_qty = math.floor((equity * max_position_pct) / entry_price)
    qty = min(qty, max_qty)
    qty = max(qty, 0)

    if direction == "short":
        stop_price = round(entry_price + stop_distance, 2)
    else:
        stop_price = round(entry_price - stop_distance, 2)
    return qty, stop_price


# ============================================================
# API CLIENTS
# ============================================================

def create_clients():
    api_key = get_required_env("ALPACA_KEY")
    api_secret = get_required_env("ALPACA_SECRET")
    trading_client = TradingClient(api_key, api_secret, paper=True)
    data_client = StockHistoricalDataClient(api_key, api_secret)
    return trading_client, data_client


def retry_api_call(func, *args, max_retries=MAX_RETRIES, **kwargs):
    """Execute an API call with exponential backoff retry."""
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries:
                raise
            delay = RETRY_DELAY_BASE ** (attempt + 1)
            log.warning(f"API call failed (attempt {attempt + 1}): {e}. "
                        f"Retrying in {delay}s...")
            time.sleep(delay)


# ============================================================
# DATA FETCHING
# ============================================================

def fetch_daily_bars(data_client: StockHistoricalDataClient,
                     symbol: str) -> pd.DataFrame:
    start_date = datetime.now(timezone.utc) - timedelta(days=500)

    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        start=start_date,
        adjustment="raw"
    )

    bars = retry_api_call(data_client.get_stock_bars, request)
    df = bars.df

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level=0).reset_index()
    else:
        df = df.reset_index()

    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    log.info(f"{symbol}: fetched {len(df)} daily bars")
    return df


# ============================================================
# POSITION / ACCOUNT HELPERS
# ============================================================

def get_open_positions(trading_client: TradingClient) -> dict:
    positions = retry_api_call(trading_client.get_all_positions)
    return {p.symbol: p for p in positions}


def get_account_equity(trading_client: TradingClient) -> float:
    account = retry_api_call(trading_client.get_account)
    return float(account.equity)


def place_market_buy(trading_client: TradingClient, symbol: str, qty: int):
    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY
    )
    return retry_api_call(trading_client.submit_order, order_data=order_data)


def place_market_sell(trading_client: TradingClient, symbol: str, qty: int):
    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY
    )
    return retry_api_call(trading_client.submit_order, order_data=order_data)


def place_short_entry(trading_client: TradingClient, symbol: str, qty: int):
    """Sell short — opens a short position."""
    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY
    )
    return retry_api_call(trading_client.submit_order, order_data=order_data)


def place_short_exit(trading_client: TradingClient, symbol: str, qty: int):
    """Buy to cover — closes a short position."""
    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY
    )
    return retry_api_call(trading_client.submit_order, order_data=order_data)


# ============================================================
# TRADE LOG HELPERS
# ============================================================

def find_latest_open_buy(log_data: list, symbol: str) -> dict | None:
    for entry in reversed(log_data):
        if (entry.get("symbol") == symbol
                and entry.get("side") == "BUY"
                and not entry.get("closed", False)):
            return entry
    return None


def find_latest_open_short(log_data: list, symbol: str) -> dict | None:
    for entry in reversed(log_data):
        if (entry.get("symbol") == symbol
                and entry.get("side") == "SHORT"
                and not entry.get("closed", False)):
            return entry
    return None


def has_exited_today(log_data: list, symbol: str) -> bool:
    today = utc_today_str()
    for entry in reversed(log_data):
        if entry.get("symbol") != symbol:
            continue
        if entry.get("side") in ("SELL", "COVER"):
            ts = entry.get("timestamp_utc", "")
            if ts.startswith(today):
                return True
        if (entry.get("side") in ("BUY", "SHORT") and entry.get("closed")
                and entry.get("exit")):
            exit_ts = entry["exit"].get("timestamp_utc", "")
            if exit_ts.startswith(today):
                return True
    return False


def get_realized_pnl_today(log_data: list) -> float:
    today = utc_today_str()
    realized = 0.0
    for entry in log_data:
        if entry.get("side") in ("SELL", "COVER"):
            ts = entry.get("timestamp_utc", "")
            if ts.startswith(today):
                realized += float(entry.get("realized_pnl", 0.0))
    return realized


def daily_loss_limit_hit(log_data: list,
                         account_equity: float) -> tuple[bool, float, float]:
    realized_today = get_realized_pnl_today(log_data)
    loss_limit_dollars = account_equity * MAX_DAILY_LOSS_PCT
    hit = realized_today <= -loss_limit_dollars
    return hit, realized_today, loss_limit_dollars


def calculate_realized_pnl(trade_entry: dict, exit_price: float,
                           qty: int) -> float:
    entry_price = float(trade_entry["entry_reference_price"])
    direction = trade_entry.get("direction", "long")
    if direction == "short":
        return round((entry_price - exit_price) * qty, 2)
    return round((exit_price - entry_price) * qty, 2)


def win_or_loss(realized_pnl: float) -> str:
    if realized_pnl > 0:
        return "WIN"
    if realized_pnl < 0:
        return "LOSS"
    return "BREAKEVEN"


def count_trading_days_held(entry_date_str: str) -> int:
    """Approximate trading days held (weekdays only)."""
    try:
        entry_date = datetime.fromisoformat(entry_date_str).date()
    except (ValueError, TypeError):
        return 0
    today = now_utc().date()
    days = 0
    current = entry_date
    while current < today:
        current += timedelta(days=1)
        if current.weekday() < 5:
            days += 1
    return days


# ============================================================
# EXIT LOGIC — DYNAMIC STOPS, TRAILING, TIME
# ============================================================

def manage_open_positions(trading_client: TradingClient,
                          data_client: StockHistoricalDataClient) -> None:
    """Manage all open positions with dynamic exit logic (long and short)."""
    open_positions = get_open_positions(trading_client)
    log_data = load_trade_log()

    if not open_positions:
        log.info("No open positions to manage.")
        return

    for symbol, position in open_positions.items():
        try:
            log.info(f"Managing position: {symbol}")
            bars_df = fetch_daily_bars(data_client, symbol)
            if bars_df.empty:
                log.warning(f"Skipping {symbol}: no market data")
                continue

            bars_df = calculate_indicators(bars_df)
            latest = bars_df.iloc[-1]
            current_price = float(latest["close"])
            current_atr = float(latest["atr"]) if not pd.isna(latest.get("atr")) else 0
            qty = abs(int(float(position.qty)))

            # Determine if this is a long or short position
            # Alpaca: negative qty = short position
            is_short = float(position.qty) < 0

            if is_short:
                trade_entry = find_latest_open_short(log_data, symbol)
                if not trade_entry:
                    log.warning(f"No matching SHORT log for {symbol}; skipping")
                    continue
            else:
                trade_entry = find_latest_open_buy(log_data, symbol)
                if not trade_entry:
                    log.warning(f"No matching BUY log for {symbol}; skipping")
                    continue

            entry_price = float(trade_entry["entry_reference_price"])
            original_stop = float(trade_entry["stop_loss_price"])
            take_profit_price = float(trade_entry["take_profit_price"])
            direction = trade_entry.get("direction", "long")

            # Calculate R-multiple
            if is_short:
                stop_distance = original_stop - entry_price
                if stop_distance > 0:
                    r_multiple = (entry_price - current_price) / stop_distance
                else:
                    r_multiple = 0
            else:
                stop_distance = entry_price - original_stop
                if stop_distance > 0:
                    r_multiple = (current_price - entry_price) / stop_distance
                else:
                    r_multiple = 0

            # --- Determine effective stop ---
            effective_stop = original_stop

            if is_short:
                # Short trailing stop: ratchets DOWN
                if r_multiple >= TRAILING_ACTIVATION_R and current_atr > 0:
                    trailing_stop = current_price + (TRAILING_ATR_MULTIPLIER * current_atr)
                    effective_stop = min(effective_stop, trailing_stop)
                    # Move stop to at least breakeven
                    effective_stop = min(effective_stop, entry_price)
            else:
                # Long trailing stop: ratchets UP
                if r_multiple >= TRAILING_ACTIVATION_R and current_atr > 0:
                    trailing_stop = current_price - (TRAILING_ATR_MULTIPLIER * current_atr)
                    effective_stop = max(effective_stop, trailing_stop)
                    effective_stop = max(effective_stop, entry_price)

            # --- Check exit conditions ---
            exit_reason = None
            exit_qty = qty

            if is_short:
                # Short: stop is above entry, take profit below
                if current_price >= effective_stop:
                    if r_multiple >= TRAILING_ACTIVATION_R:
                        exit_reason = "TRAILING_STOP"
                    else:
                        exit_reason = "STOP_LOSS"
                elif current_price <= take_profit_price:
                    exit_reason = "TAKE_PROFIT"
            else:
                # Long: stop is below entry, take profit above
                if current_price <= effective_stop:
                    if r_multiple >= TRAILING_ACTIVATION_R:
                        exit_reason = "TRAILING_STOP"
                    else:
                        exit_reason = "STOP_LOSS"
                elif current_price >= take_profit_price:
                    exit_reason = "TAKE_PROFIT"

            # Time stop (both directions)
            if not exit_reason:
                entry_ts = trade_entry.get("timestamp_utc", "")
                days_held = count_trading_days_held(entry_ts)
                if days_held >= MAX_HOLD_DAYS and r_multiple < TRAILING_ACTIVATION_R:
                    exit_reason = f"TIME_STOP ({days_held} days)"

            # Regime change exit
            regime = detect_regime(latest)
            trade_regime = trade_entry.get("regime", "unknown")
            if (trade_regime == "trending" and regime == "ranging"
                    and r_multiple < 0.5 and not exit_reason):
                if is_short:
                    tight_stop = current_price + current_atr
                    if current_price >= tight_stop or tight_stop < effective_stop:
                        effective_stop = min(effective_stop, tight_stop)
                        if current_price >= effective_stop:
                            exit_reason = "REGIME_CHANGE_EXIT"
                else:
                    tight_stop = current_price - current_atr
                    if current_price <= tight_stop or tight_stop > effective_stop:
                        effective_stop = max(effective_stop, tight_stop)
                        if current_price <= effective_stop:
                            exit_reason = "REGIME_CHANGE_EXIT"

            dir_label = "SHORT" if is_short else "LONG"
            log.info(
                f"{symbol} [{dir_label}]: price={current_price:.2f}, "
                f"entry={entry_price:.2f}, stop={effective_stop:.2f}, "
                f"target={take_profit_price:.2f}, R={r_multiple:.2f}, "
                f"regime={regime}"
            )

            if not exit_reason:
                log.info(f"{symbol}: HOLD (R={r_multiple:.2f})")
                # Update the trailing stop in the log
                stop_changed = (effective_stop > original_stop if not is_short
                                else effective_stop < original_stop)
                if stop_changed:
                    entry_side = "SHORT" if is_short else "BUY"
                    for entry in reversed(log_data):
                        if (entry.get("order_id") == trade_entry["order_id"]
                                and entry.get("side") == entry_side
                                and not entry.get("closed", False)):
                            entry["stop_loss_price"] = round(effective_stop, 2)
                            break
                    save_trade_log(log_data)
                continue

            log.info(f"{symbol}: EXIT -> {exit_reason} (qty={exit_qty})")

            if is_short:
                close_order = place_short_exit(trading_client, symbol, exit_qty)
                exit_side = "COVER"
            else:
                close_order = place_market_sell(trading_client, symbol, exit_qty)
                exit_side = "SELL"

            realized_pnl = calculate_realized_pnl(trade_entry, current_price,
                                                  exit_qty)

            exit_info = {
                "timestamp_utc": now_utc_iso(),
                "reason": exit_reason,
                "exit_reference_price": current_price,
                "sell_order_id": str(close_order.id),
                "qty": exit_qty,
                "realized_pnl": realized_pnl,
                "result": win_or_loss(realized_pnl),
                "r_multiple": round(r_multiple, 2),
                "days_held": count_trading_days_held(
                    trade_entry.get("timestamp_utc", "")),
                "exit_regime": regime,
            }

            entry_side = "SHORT" if is_short else "BUY"
            for entry in reversed(log_data):
                if (entry.get("order_id") == trade_entry["order_id"]
                        and entry.get("side") == entry_side
                        and not entry.get("closed", False)):
                    entry["closed"] = True
                    entry["exit"] = exit_info
                    break

            exit_log = {
                "timestamp_utc": now_utc_iso(),
                "trade_date": utc_today_str(),
                "symbol": symbol,
                "side": exit_side,
                "direction": direction,
                "qty": exit_qty,
                "entry_reference_price": entry_price,
                "exit_reference_price": current_price,
                "realized_pnl": realized_pnl,
                "result": win_or_loss(realized_pnl),
                "r_multiple": round(r_multiple, 2),
                "strategy": "hybrid_regime_v6",
                "order_id": str(close_order.id),
                "notes": {
                    "reason": exit_reason,
                    "linked_entry_order_id": trade_entry["order_id"],
                    "days_held": count_trading_days_held(
                        trade_entry.get("timestamp_utc", "")),
                    "exit_atr": round(current_atr, 4),
                    "exit_regime": regime,
                }
            }
            log_data.append(exit_log)
            save_trade_log(log_data)

            log.info(
                f"{symbol}: {exit_side} | Order: {close_order.id} | "
                f"P/L: ${realized_pnl:.2f} | R: {r_multiple:.2f} | "
                f"Reason: {exit_reason}"
            )

        except Exception as e:
            log.error(f"Error managing {symbol}: {e}\n{traceback.format_exc()}")


# ============================================================
# ENTRY SCANNING
# ============================================================

def scan_for_new_entries(trading_client: TradingClient,
                         data_client: StockHistoricalDataClient) -> None:
    """Scan watchlist for new entry signals."""
    log_data = load_trade_log()
    open_positions = get_open_positions(trading_client)
    account_equity = get_account_equity(trading_client)

    log.info(f"Open positions: {sorted(open_positions.keys())}")
    log.info(f"Account equity: ${account_equity:.2f}")

    loss_limit_hit, realized_today, loss_limit_dollars = daily_loss_limit_hit(
        log_data, account_equity)
    log.info(f"Realized P/L today: ${realized_today:.2f} | "
             f"Daily loss limit: -${loss_limit_dollars:.2f}")

    if loss_limit_hit:
        log.warning("DAILY LOSS LIMIT HIT — no new trades today.")
        return

    if len(open_positions) >= MAX_OPEN_POSITIONS:
        log.info("Max open positions reached — no new trades.")
        return

    entries_this_run = 0
    max_entries_per_run = MAX_OPEN_POSITIONS - len(open_positions)

    for symbol in WATCHLIST:
        if entries_this_run >= max_entries_per_run:
            log.info("Max entries for this run reached.")
            break

        try:
            if symbol in open_positions:
                log.info(f"{symbol}: SKIP — already in position")
                continue

            if has_exited_today(log_data, symbol):
                log.info(f"{symbol}: SKIP — same-day re-entry blocked")
                continue

            df = fetch_daily_bars(data_client, symbol)
            if df.empty:
                log.info(f"{symbol}: SKIP — no bar data")
                continue

            df = calculate_indicators(df)
            signal, reason, regime, direction = evaluate_entry(df)
            latest = df.iloc[-1]

            # Log the analysis for every symbol
            log.info(
                f"{symbol}: regime={regime}, direction={direction}, "
                f"signal={signal} | {reason} | "
                f"close={latest['close']:.2f}, ADX={latest.get('adx', 0):.1f}, "
                f"RSI={latest.get('rsi', 0):.1f}, "
                f"ATR={latest.get('atr', 0):.2f}, "
                f"vol_ratio={latest.get('volume_ratio', 0):.2f}"
            )

            if not signal:
                continue

            entry_price = float(latest["close"])
            current_atr = float(latest["atr"])

            # Determine stop/target multipliers based on regime and direction
            if direction == "short":
                stop_mult = ATR_STOP_MULTIPLIER_SHORT
                target_mult = ATR_TARGET_MULTIPLIER_SHORT
            elif regime == "trending":
                stop_mult = ATR_STOP_MULTIPLIER_TREND
                target_mult = ATR_TARGET_MULTIPLIER_TREND
            elif regime == "ranging":
                stop_mult = ATR_STOP_MULTIPLIER_RANGE
                target_mult = ATR_TARGET_MULTIPLIER_RANGE
            else:
                stop_mult = (ATR_STOP_MULTIPLIER_TREND + ATR_STOP_MULTIPLIER_RANGE) / 2
                target_mult = (ATR_TARGET_MULTIPLIER_TREND + ATR_TARGET_MULTIPLIER_RANGE) / 2

            qty, stop_price = position_size_from_atr(
                equity=account_equity,
                entry_price=entry_price,
                atr=current_atr,
                stop_multiplier=stop_mult,
                risk_per_trade=RISK_PER_TRADE,
                max_position_pct=MAX_POSITION_PCT,
                direction=direction
            )

            if qty < 1:
                log.info(f"{symbol}: SKIP — calculated qty < 1")
                continue

            # Target price depends on direction
            if direction == "short":
                take_profit_price = round(entry_price - (current_atr * target_mult), 2)
            else:
                take_profit_price = round(entry_price + (current_atr * target_mult), 2)
            estimated_value = qty * entry_price

            dir_label = "SHORT" if direction == "short" else "LONG"
            log.info(
                f"{symbol}: {dir_label} ENTRY SIGNAL — qty={qty}, "
                f"value=${estimated_value:.2f}, "
                f"stop={stop_price:.2f}, target={take_profit_price:.2f}, "
                f"regime={regime}"
            )

            if direction == "short":
                order = place_short_entry(trading_client, symbol, qty)
                log_side = "SHORT"
            else:
                order = place_market_buy(trading_client, symbol, qty)
                log_side = "BUY"

            log_entry = {
                "timestamp_utc": now_utc_iso(),
                "trade_date": utc_today_str(),
                "symbol": symbol,
                "side": log_side,
                "direction": direction,
                "qty": qty,
                "entry_reference_price": entry_price,
                "estimated_position_value": estimated_value,
                "risk_dollars": round(account_equity * RISK_PER_TRADE, 2),
                "stop_loss_price": stop_price,
                "take_profit_price": take_profit_price,
                "entry_atr": round(current_atr, 4),
                "regime": regime,
                "strategy": "hybrid_regime_v6",
                "order_id": str(order.id),
                "closed": False,
                "notes": {
                    "reason": reason,
                    "close": entry_price,
                    "adx": round(float(latest.get("adx", 0)), 2),
                    "rsi": round(float(latest.get("rsi", 0)), 2),
                    "atr": round(current_atr, 4),
                    "atr_pct": round(float(latest.get("atr_pct", 0)), 5),
                    "ema_fast": round(float(latest.get("ema_fast", 0)), 2),
                    "ema_slow": round(float(latest.get("ema_slow", 0)), 2),
                    "sma50": round(float(latest.get("sma50", 0)), 2),
                    "sma200": round(float(latest.get("sma200", 0)), 2),
                    "plus_di": round(float(latest.get("plus_di", 0)), 2),
                    "minus_di": round(float(latest.get("minus_di", 0)), 2),
                    "volume_ratio": round(float(latest.get("volume_ratio", 0)), 2),
                    "bb_lower": round(float(latest.get("bb_lower", 0)), 2),
                    "bb_upper": round(float(latest.get("bb_upper", 0)), 2),
                    "stop_multiplier": stop_mult,
                    "target_multiplier": target_mult,
                }
            }
            append_trade_log(log_entry)

            log.info(f"{symbol}: {dir_label} ORDER PLACED | ID: {order.id}")
            entries_this_run += 1

        except Exception as e:
            log.error(f"Error scanning {symbol}: {e}\n{traceback.format_exc()}")


# ============================================================
# DAILY SUMMARY
# ============================================================

def print_daily_summary(trading_client: TradingClient) -> None:
    log_data = load_trade_log()
    account_equity = get_account_equity(trading_client)
    open_positions = get_open_positions(trading_client)

    realized_today = get_realized_pnl_today(log_data)
    loss_limit_dollars = account_equity * MAX_DAILY_LOSS_PCT

    today = utc_today_str()
    today_sells = [
        x for x in log_data
        if x.get("side") in ("SELL", "COVER")
        and x.get("timestamp_utc", "").startswith(today)
    ]

    wins = sum(1 for x in today_sells if x.get("realized_pnl", 0) > 0)
    losses = sum(1 for x in today_sells if x.get("realized_pnl", 0) < 0)
    breakeven = sum(1 for x in today_sells if x.get("realized_pnl", 0) == 0)

    # Win rate and average R
    total_trades = wins + losses + breakeven
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    avg_r = 0
    if today_sells:
        r_values = [x.get("r_multiple", 0) for x in today_sells]
        avg_r = sum(r_values) / len(r_values) if r_values else 0

    summary = (
        f"\n{'='*60}\n"
        f"  DAILY SUMMARY — {today}\n"
        f"{'='*60}\n"
        f"  Equity:           ${account_equity:>12,.2f}\n"
        f"  Open Positions:   {len(open_positions):>12}\n"
        f"  Closed Today:     {total_trades:>12}\n"
        f"  Wins/Losses/BE:   {wins}/{losses}/{breakeven}\n"
        f"  Win Rate:         {win_rate:>11.1f}%\n"
        f"  Avg R-Multiple:   {avg_r:>12.2f}\n"
        f"  Realized P/L:     ${realized_today:>12,.2f}\n"
        f"  Daily Loss Limit: ${-loss_limit_dollars:>12,.2f}\n"
        f"{'='*60}\n"
    )
    log.info(summary)

    # Log equity curve
    log_equity_curve(account_equity, len(open_positions))


# ============================================================
# MAIN
# ============================================================

def run_bot() -> None:
    ensure_directories()
    load_environment()

    log.info("=" * 60)
    log.info("Hybrid Regime-Switching Bot v6 — Starting Run")
    log.info("=" * 60)

    trading_client, data_client = create_clients()

    account = retry_api_call(trading_client.get_account)
    log.info(f"Connected to Alpaca | Status: {account.status}")
    log.info(f"Equity: ${float(account.equity):,.2f} | "
             f"Buying Power: ${float(account.buying_power):,.2f}")
    log.info("-" * 60)

    # Phase 1: Manage existing positions (exits, trailing stops)
    log.info("PHASE 1: Managing open positions...")
    manage_open_positions(trading_client, data_client)

    # Phase 2: Scan for new entries
    log.info("\nPHASE 2: Scanning for new entries...")
    scan_for_new_entries(trading_client, data_client)

    # Phase 3: Daily summary + equity curve
    print_daily_summary(trading_client)

    log.info("Run complete.\n")


def test_entry_signals():
    """Test entry signal evaluation with synthetic data for TSLA and GOOG.

    Generates realistic downtrending price data to verify that:
    - Short entries trigger when EMA9 < EMA21 + price below swing high
    - Direction field correctly shows 'short' for downtrending stocks
    - RSI < 45 threshold works for ranging regime longs
    """
    print("=" * 60)
    print("ENTRY SIGNAL TEST — Synthetic Data")
    print("=" * 60)

    np.random.seed(42)

    for symbol, start_price, trend in [("TSLA", 280.0, "down"), ("GOOG", 170.0, "down")]:
        n_bars = 250
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_bars)

        # Generate downtrending OHLCV
        prices = [start_price]
        for i in range(1, n_bars):
            # Downtrend: -0.15% drift with noise
            daily_return = -0.0015 + np.random.normal(0, 0.015)
            prices.append(prices[-1] * (1 + daily_return))

        closes = np.array(prices)
        highs = closes * (1 + np.abs(np.random.normal(0.005, 0.003, n_bars)))
        lows = closes * (1 - np.abs(np.random.normal(0.005, 0.003, n_bars)))
        opens = closes * (1 + np.random.normal(0, 0.003, n_bars))
        volumes = np.random.randint(5_000_000, 30_000_000, n_bars).astype(float)

        df = pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }, index=dates)

        df = calculate_indicators(df)
        signal, reason, regime, direction = evaluate_entry(df)
        latest = df.iloc[-1]

        print(f"\n{'─' * 50}")
        print(f"  {symbol} (synthetic {trend}trend)")
        print(f"{'─' * 50}")
        print(f"  Close:       {latest['close']:.2f}")
        print(f"  EMA9:        {latest['ema_fast']:.2f}")
        print(f"  EMA21:       {latest['ema_slow']:.2f}")
        print(f"  SMA200:      {latest['sma200']:.2f}")
        print(f"  ADX:         {latest['adx']:.1f}")
        print(f"  RSI:         {latest['rsi']:.1f}")
        print(f"  +DI:         {latest['plus_di']:.1f}")
        print(f"  -DI:         {latest['minus_di']:.1f}")
        print(f"  Vol Ratio:   {latest['volume_ratio']:.2f}")
        print(f"  Regime:      {regime}")
        print(f"  Direction:   {direction}")
        print(f"  Signal:      {signal}")
        print(f"  Reason:      {reason}")

        if signal:
            print(f"  >>> ENTRY WOULD TRIGGER: {direction.upper()} on {symbol}")
        else:
            print(f"  >>> No entry (reason above)")

    print(f"\n{'=' * 60}")
    print("Test complete.")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test-signals":
        test_entry_signals()
    else:
        run_bot()
