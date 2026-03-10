"""
Hybrid Regime-Switching Trading Bot v5
=======================================
- Trend-following when ADX > 25 (momentum + breakout)
- Mean-reversion when ADX < 20 (Bollinger Band + RSI)
- Conservative in neutral zone (ADX 20-25)
- ATR-based dynamic stops, trailing stops, time stops
- Volume confirmation on all entries
- Daily equity curve tracking
- Max daily drawdown kill switch
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

# Volume filter
VOLUME_CONFIRMATION_MULT = 1.1   # Volume must be > 1.1x 20-day average

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


def signal_mean_reversion_long(df: pd.DataFrame) -> tuple[bool, str]:
    """Mean-reversion long signal (ADX < 20 regime).

    Confluence requirements:
    1. Price near or below lower Bollinger Band
    2. RSI < 35 (oversold)
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

    # 1. Price near lower BB (within 0.5% of lower band or below it)
    bb_distance = (latest["close"] - latest["bb_lower"]) / latest["bb_lower"]
    if bb_distance > 0.005:
        return False, f"Price not near lower BB (distance={bb_distance:.4f})"

    # 2. RSI oversold
    if latest["rsi"] >= 35:
        return False, f"RSI {latest['rsi']:.1f} not oversold (need < 35)"

    # 3. Long-term context: only buy dips in uptrend
    if latest["close"] <= latest["sma200"] * 0.95:
        return False, "Price too far below SMA200 — potential breakdown"

    # 4. Volume: want elevated volume (capitulation/selling climax)
    vol_ratio = latest.get("volume_ratio", 0)
    if vol_ratio < 1.0:
        return False, f"Volume too low for capitulation ({vol_ratio:.2f})"

    # 5. Bullish candle (close > open = buyers stepping in)
    if latest["close"] <= latest["open"]:
        return False, "No bullish candle — close <= open"

    return True, (f"MEAN REVERSION LONG: RSI={latest['rsi']:.1f}, "
                  f"BB_dist={bb_distance:.4f}, vol_ratio={vol_ratio:.2f}")


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


def evaluate_entry(df: pd.DataFrame) -> tuple[bool, str, str]:
    """Master entry evaluation. Returns (signal, reason, regime)."""
    if len(df) < SMA_LONG + 5:
        return False, "Insufficient data", "unknown"

    latest = df.iloc[-1]
    regime = detect_regime(latest)

    # Volatility gate — applies to ALL regimes
    vol_ok, vol_reason = passes_volatility_filter(latest)
    if not vol_ok:
        return False, vol_reason, regime

    if regime == "trending":
        signal, reason = signal_trend_long(df)
        return signal, reason, regime
    elif regime == "ranging":
        signal, reason = signal_mean_reversion_long(df)
        return signal, reason, regime
    else:
        signal, reason = signal_neutral_long(df)
        return signal, reason, regime


# ============================================================
# POSITION SIZING
# ============================================================

def position_size_from_atr(
    equity: float,
    entry_price: float,
    atr: float,
    stop_multiplier: float,
    risk_per_trade: float,
    max_position_pct: float
) -> tuple[int, float]:
    """Calculate position size based on ATR stop distance.

    Returns (qty, stop_price).
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


def has_exited_today(log_data: list, symbol: str) -> bool:
    today = utc_today_str()
    for entry in reversed(log_data):
        if entry.get("symbol") != symbol:
            continue
        if entry.get("side") == "SELL":
            ts = entry.get("timestamp_utc", "")
            if ts.startswith(today):
                return True
        if (entry.get("side") == "BUY" and entry.get("closed")
                and entry.get("exit")):
            exit_ts = entry["exit"].get("timestamp_utc", "")
            if exit_ts.startswith(today):
                return True
    return False


def get_realized_pnl_today(log_data: list) -> float:
    today = utc_today_str()
    realized = 0.0
    for entry in log_data:
        if entry.get("side") == "SELL":
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


def calculate_realized_pnl(buy_entry: dict, exit_price: float,
                           qty: int) -> float:
    entry_price = float(buy_entry["entry_reference_price"])
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
    """Manage all open positions with dynamic exit logic."""
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
            qty = int(float(position.qty))

            buy_entry = find_latest_open_buy(log_data, symbol)
            if not buy_entry:
                log.warning(f"No matching BUY log for {symbol}; skipping")
                continue

            entry_price = float(buy_entry["entry_reference_price"])
            original_stop = float(buy_entry["stop_loss_price"])
            take_profit_price = float(buy_entry["take_profit_price"])
            entry_atr = float(buy_entry.get("entry_atr", current_atr))

            # Calculate R-multiple (how many R's of profit)
            stop_distance = entry_price - original_stop
            if stop_distance > 0:
                r_multiple = (current_price - entry_price) / stop_distance
            else:
                r_multiple = 0

            # --- Determine effective stop ---
            effective_stop = original_stop

            # Trailing stop: if profit > 1R, trail at 2.5x current ATR
            if r_multiple >= TRAILING_ACTIVATION_R and current_atr > 0:
                trailing_stop = current_price - (TRAILING_ATR_MULTIPLIER * current_atr)
                # Trailing stop only ratchets UP, never down
                effective_stop = max(effective_stop, trailing_stop)
                # Also move stop to at least breakeven once 1R is hit
                effective_stop = max(effective_stop, entry_price)

            # --- Check exit conditions ---
            exit_reason = None
            exit_qty = qty

            # 1. Stop loss (static or trailing)
            if current_price <= effective_stop:
                if r_multiple >= TRAILING_ACTIVATION_R:
                    exit_reason = "TRAILING_STOP"
                else:
                    exit_reason = "STOP_LOSS"

            # 2. Take profit target
            elif current_price >= take_profit_price:
                exit_reason = "TAKE_PROFIT"

            # 3. Time stop
            elif not exit_reason:
                entry_ts = buy_entry.get("timestamp_utc", "")
                days_held = count_trading_days_held(entry_ts)
                if days_held >= MAX_HOLD_DAYS and r_multiple < TRAILING_ACTIVATION_R:
                    exit_reason = f"TIME_STOP ({days_held} days)"

            # 4. Regime change exit — if regime shifted to ranging and we're
            #    in a trend trade with minimal profit, tighten stop
            regime = detect_regime(latest)
            trade_regime = buy_entry.get("regime", "unknown")
            if (trade_regime == "trending" and regime == "ranging"
                    and r_multiple < 0.5 and not exit_reason):
                # Tighten stop to 1 ATR
                tight_stop = current_price - current_atr
                if current_price <= tight_stop or tight_stop > effective_stop:
                    effective_stop = max(effective_stop, tight_stop)
                    if current_price <= effective_stop:
                        exit_reason = "REGIME_CHANGE_EXIT"

            log.info(
                f"{symbol}: price={current_price:.2f}, "
                f"entry={entry_price:.2f}, stop={effective_stop:.2f}, "
                f"target={take_profit_price:.2f}, R={r_multiple:.2f}, "
                f"regime={regime}"
            )

            if not exit_reason:
                log.info(f"{symbol}: HOLD (R={r_multiple:.2f})")
                # Update the trailing stop in the log for next check
                if effective_stop > original_stop:
                    for entry in reversed(log_data):
                        if (entry.get("order_id") == buy_entry["order_id"]
                                and entry.get("side") == "BUY"
                                and not entry.get("closed", False)):
                            entry["stop_loss_price"] = round(effective_stop, 2)
                            break
                    save_trade_log(log_data)
                continue

            log.info(f"{symbol}: EXIT -> {exit_reason} (qty={exit_qty})")
            sell_order = place_market_sell(trading_client, symbol, exit_qty)

            realized_pnl = calculate_realized_pnl(buy_entry, current_price,
                                                  exit_qty)

            exit_info = {
                "timestamp_utc": now_utc_iso(),
                "reason": exit_reason,
                "exit_reference_price": current_price,
                "sell_order_id": str(sell_order.id),
                "qty": exit_qty,
                "realized_pnl": realized_pnl,
                "result": win_or_loss(realized_pnl),
                "r_multiple": round(r_multiple, 2),
                "days_held": count_trading_days_held(
                    buy_entry.get("timestamp_utc", "")),
                "exit_regime": regime,
            }

            for entry in reversed(log_data):
                if (entry.get("order_id") == buy_entry["order_id"]
                        and entry.get("side") == "BUY"
                        and not entry.get("closed", False)):
                    entry["closed"] = True
                    entry["exit"] = exit_info
                    break

            sell_log = {
                "timestamp_utc": now_utc_iso(),
                "trade_date": utc_today_str(),
                "symbol": symbol,
                "side": "SELL",
                "qty": exit_qty,
                "entry_reference_price": entry_price,
                "exit_reference_price": current_price,
                "realized_pnl": realized_pnl,
                "result": win_or_loss(realized_pnl),
                "r_multiple": round(r_multiple, 2),
                "strategy": "hybrid_regime_v5",
                "order_id": str(sell_order.id),
                "notes": {
                    "reason": exit_reason,
                    "linked_buy_order_id": buy_entry["order_id"],
                    "days_held": count_trading_days_held(
                        buy_entry.get("timestamp_utc", "")),
                    "exit_atr": round(current_atr, 4),
                    "exit_regime": regime,
                }
            }
            log_data.append(sell_log)
            save_trade_log(log_data)

            log.info(
                f"{symbol}: SOLD | Order: {sell_order.id} | "
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
            signal, reason, regime = evaluate_entry(df)
            latest = df.iloc[-1]

            # Log the analysis for every symbol
            log.info(
                f"{symbol}: regime={regime}, signal={signal} | {reason} | "
                f"close={latest['close']:.2f}, ADX={latest.get('adx', 0):.1f}, "
                f"RSI={latest.get('rsi', 0):.1f}, "
                f"ATR={latest.get('atr', 0):.2f}, "
                f"vol_ratio={latest.get('volume_ratio', 0):.2f}"
            )

            if not signal:
                continue

            entry_price = float(latest["close"])
            current_atr = float(latest["atr"])

            # Determine stop/target multipliers based on regime
            if regime == "trending":
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
                max_position_pct=MAX_POSITION_PCT
            )

            if qty < 1:
                log.info(f"{symbol}: SKIP — calculated qty < 1")
                continue

            take_profit_price = round(entry_price + (current_atr * target_mult), 2)
            estimated_value = qty * entry_price

            log.info(
                f"{symbol}: ENTRY SIGNAL — qty={qty}, "
                f"value=${estimated_value:.2f}, "
                f"stop={stop_price:.2f}, target={take_profit_price:.2f}, "
                f"regime={regime}"
            )

            order = place_market_buy(trading_client, symbol, qty)

            log_entry = {
                "timestamp_utc": now_utc_iso(),
                "trade_date": utc_today_str(),
                "symbol": symbol,
                "side": "BUY",
                "qty": qty,
                "entry_reference_price": entry_price,
                "estimated_position_value": estimated_value,
                "risk_dollars": round(account_equity * RISK_PER_TRADE, 2),
                "stop_loss_price": stop_price,
                "take_profit_price": take_profit_price,
                "entry_atr": round(current_atr, 4),
                "regime": regime,
                "strategy": "hybrid_regime_v5",
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

            log.info(f"{symbol}: ORDER PLACED | ID: {order.id}")
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
        if x.get("side") == "SELL"
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
    log.info("Hybrid Regime-Switching Bot v5 — Starting Run")
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


if __name__ == "__main__":
    run_bot()
