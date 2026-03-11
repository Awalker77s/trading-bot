"""Crypto paper trading bot entry point.

Runs a crypto-only strategy on Alpaca paper trading for BTC/USD, ETH/USD, and SOL/USD.
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

import main as shared
import monitor

# Load .env early so os.getenv() calls below pick up config values at import time.
shared.load_environment()

SCRIPT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = SCRIPT_DIR / "logs"
CRYPTO_LOG_FILE = LOG_DIR / "crypto_bot.log"
CRYPTO_TRADE_LOG_FILE = LOG_DIR / "crypto_trade_log.json"

# fmt: off
CRYPTO_WATCHLIST = [
    "BTC/USD",   # Bitcoin
    "ETH/USD",   # Ethereum
    "SOL/USD",   # Solana
    "LINK/USD",  # Chainlink          # verify Alpaca support before live
    "AVAX/USD",  # Avalanche          # verify Alpaca support before live
    "DOT/USD",   # Polkadot           # verify Alpaca support before live
    "UNI/USD",   # Uniswap            # verify Alpaca support before live
    "AAVE/USD",  # Aave               # verify Alpaca support before live
    "LTC/USD",   # Litecoin           # verify Alpaca support before live
]
# fmt: on

# Crypto risk tuning (separate from stocks) — all overridable via config/.env
CRYPTO_RISK_PER_TRADE         = float(os.getenv("CRYPTO_RISK_PER_TRADE",          "0.01"))
CRYPTO_POSITION_SIZE_PCT      = float(os.getenv("CRYPTO_POSITION_SIZE_PCT",       "0.12"))  # 6 pos × 12% ≈ 72% buying power
MAX_CRYPTO_POSITIONS          = int(os.getenv("MAX_CRYPTO_POSITIONS",             "6"))
CRYPTO_MIN_NOTIONAL_USD       = float(os.getenv("CRYPTO_MIN_NOTIONAL_USD",        "25.0"))
CRYPTO_STOP_ATR_MULTIPLIER    = float(os.getenv("CRYPTO_STOP_ATR_MULTIPLIER",     "2.2"))
CRYPTO_TARGET_ATR_MULTIPLIER  = float(os.getenv("CRYPTO_TARGET_ATR_MULTIPLIER",   "3.8"))
CRYPTO_MIN_ATR_PCT            = float(os.getenv("CRYPTO_MIN_ATR_PCT",             "0.004"))
CRYPTO_MAX_ATR_PCT            = float(os.getenv("CRYPTO_MAX_ATR_PCT",             "0.15"))
CRYPTO_LOOKBACK_DAYS          = int(os.getenv("CRYPTO_LOOKBACK_DAYS",             "220"))
CRYPTO_MAX_DAILY_LOSS_PCT     = float(os.getenv("CRYPTO_MAX_DAILY_LOSS_PCT",      "0.02"))
# OPT: volume confirmation gate — only enter when current bar volume > 20-period average
CRYPTO_VOLUME_FILTER          = os.getenv("CRYPTO_VOLUME_FILTER", "true").strip().lower() in {"1", "true", "yes", "on"}
CROSSOVER_LOOKBACK_CANDLES    = 3


def setup_crypto_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("crypto_trading_bot")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [CRYPTO] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(CRYPTO_LOG_FILE)
        ch = logging.StreamHandler()
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


log = setup_crypto_logging()


def load_crypto_trade_log() -> list[dict]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not CRYPTO_TRADE_LOG_FILE.exists():
        return []
    try:
        return json.loads(CRYPTO_TRADE_LOG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_crypto_trade_log(rows: list[dict]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CRYPTO_TRADE_LOG_FILE.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def validate_crypto_startup_config() -> dict[str, str | bool]:
    api_key = shared.get_required_env("ALPACA_KEY", "APCA_API_KEY_ID")
    api_secret = shared.get_required_env("ALPACA_SECRET", "APCA_API_SECRET_KEY")
    paper_flag = os.getenv("ALPACA_PAPER", "true").strip().lower() in {"1", "true", "yes", "on"}

    if not paper_flag:
        raise ValueError("Crypto bot requires ALPACA_PAPER=true (paper trading only).")

    return {"api_key": api_key, "api_secret": api_secret, "paper": True}


def create_clients(config: dict[str, str | bool]) -> tuple[TradingClient, CryptoHistoricalDataClient]:
    trading_client = TradingClient(config["api_key"], config["api_secret"], paper=True)
    data_client = CryptoHistoricalDataClient(config["api_key"], config["api_secret"])
    return trading_client, data_client


def fetch_crypto_bars(data_client: CryptoHistoricalDataClient, symbol: str) -> pd.DataFrame:
    request = CryptoBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        start=datetime.now(timezone.utc) - timedelta(days=CRYPTO_LOOKBACK_DAYS),
    )
    bars = shared.retry_api_call(data_client.get_crypto_bars, request)
    df = bars.df
    if df.empty:
        return pd.DataFrame()

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level=0).reset_index()
    else:
        df = df.reset_index()

    return shared.validate_bars_df(df, symbol)


def evaluate_crypto_entry(df: pd.DataFrame) -> tuple[bool, str]:
    if len(df) < 60:
        return False, "Insufficient data"

    latest = df.iloc[-1]
    required = ["close", "ema_fast", "ema_slow", "sma50", "rsi", "adx", "atr_pct"]
    for field in required:
        if pd.isna(latest.get(field)):
            return False, f"Missing {field}"

    if latest["atr_pct"] < CRYPTO_MIN_ATR_PCT:
        return False, f"ATR% too low ({latest['atr_pct']:.4f})"
    if latest["atr_pct"] > CRYPTO_MAX_ATR_PCT:
        return False, f"ATR% too high ({latest['atr_pct']:.4f})"

    crossed_up_recently = False
    for i in range(1, CROSSOVER_LOOKBACK_CANDLES + 1):
        older = df.iloc[-i - 1]
        newer = df.iloc[-i]
        if any(
            pd.isna(row.get("ema_fast")) or pd.isna(row.get("ema_slow"))
            for row in (older, newer)
        ):
            continue
        if older["ema_fast"] <= older["ema_slow"] and newer["ema_fast"] > newer["ema_slow"]:
            crossed_up_recently = True
            break
    if not crossed_up_recently:
        return False, (
            f"No EMA crossover UP in last {CROSSOVER_LOOKBACK_CANDLES} candles "
            f"(cur fast={latest['ema_fast']:.2f} slow={latest['ema_slow']:.2f})"
        )
    if latest["close"] <= latest["sma50"]:
        return False, "Close <= SMA50"
    if latest["adx"] < 18:
        return False, f"ADX weak ({latest['adx']:.1f})"
    if not (45 <= latest["rsi"] <= 72):
        return False, f"RSI out of range ({latest['rsi']:.1f})"

    # OPT: volume confirmation — current bar must exceed 20-period average volume to reduce
    # false signals (toggled by CRYPTO_VOLUME_FILTER env flag)
    if CRYPTO_VOLUME_FILTER:
        vol = latest.get("volume", 0)
        vol_sma = latest.get("volume_sma", 0)
        if pd.isna(vol_sma) or vol_sma <= 0 or vol <= vol_sma:
            return False, f"Volume below 20-period avg ({vol:.0f} <= {vol_sma:.0f})"

    return True, "Momentum breakout continuation"


def evaluate_crypto_short_entry(df: pd.DataFrame) -> tuple[bool, str]:
    """Short entry signal — triggers when fast EMA is currently below slow EMA (downtrend state).

    No fresh crossover is required; we check current EMA state instead, then confirm with:
    - RSI < 52 (momentum is down or neutral, not overbought)
    - ADX > 15 (some trend strength present)
    - Close < SMA50 (price in downtrend context)
    - ATR% within the allowed threshold (same filter as longs)
    - Volume > 20-period average (if CRYPTO_VOLUME_FILTER=true)
    The "not already in a short position" guard is enforced in scan_for_entries().
    """
    if len(df) < 60:
        return False, "Insufficient data"

    latest = df.iloc[-1]
    required = ["close", "ema_fast", "ema_slow", "sma50", "rsi", "adx", "atr_pct"]
    for field in required:
        if pd.isna(latest.get(field)):
            return False, f"Missing {field}"

    if latest["atr_pct"] < CRYPTO_MIN_ATR_PCT:
        return False, f"ATR% too low ({latest['atr_pct']:.4f})"
    if latest["atr_pct"] > CRYPTO_MAX_ATR_PCT:
        return False, f"ATR% too high ({latest['atr_pct']:.4f})"

    # Primary condition: fast EMA currently below slow EMA (in downtrend)
    if latest["ema_fast"] >= latest["ema_slow"]:
        return False, (
            f"Fast EMA not below slow EMA "
            f"(fast={latest['ema_fast']:.2f} slow={latest['ema_slow']:.2f})"
        )

    if latest["rsi"] >= 52:
        return False, f"RSI too high ({latest['rsi']:.1f} >= 52)"

    if latest["adx"] < 15:
        return False, f"ADX weak ({latest['adx']:.1f} < 15)"

    if latest["close"] >= latest["sma50"]:
        return False, "Close >= SMA50 (not in downtrend context)"

    if CRYPTO_VOLUME_FILTER:
        vol = latest.get("volume", 0)
        vol_sma = latest.get("volume_sma", 0)
        if pd.isna(vol_sma) or vol_sma <= 0 or vol <= vol_sma:
            return False, f"Volume below 20-period avg ({vol:.0f} <= {vol_sma:.0f})"

    return True, (
        f"Downtrend: fast EMA < slow EMA "
        f"(fast={latest['ema_fast']:.2f} slow={latest['ema_slow']:.2f})"
    )


def calculate_notional_size(equity: float, price: float, atr: float) -> float:
    if equity <= 0 or price <= 0 or atr <= 0:
        return 0.0
    stop_distance = atr * CRYPTO_STOP_ATR_MULTIPLIER
    risk_dollars = equity * CRYPTO_RISK_PER_TRADE
    risk_based_notional = (risk_dollars * price) / stop_distance
    cap_notional = equity * CRYPTO_POSITION_SIZE_PCT
    notional = min(risk_based_notional, cap_notional)
    if notional < CRYPTO_MIN_NOTIONAL_USD:
        return 0.0
    return round(notional, 2)


def get_open_crypto_positions(trading_client: TradingClient) -> dict[str, object]:
    positions = shared.retry_api_call(trading_client.get_all_positions)
    crypto_positions = [p for p in positions if str(getattr(p, "asset_class", "")).lower() == "crypto"]
    return {p.symbol: p for p in crypto_positions}


def submit_crypto_entry_order(trading_client: TradingClient, symbol: str, notional: float):
    order = MarketOrderRequest(
        symbol=symbol,
        notional=notional,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.GTC,
    )
    log.info(f"{symbol}: submitting crypto BUY market order with notional=${notional:.2f}")
    return shared.retry_api_call(trading_client.submit_order, order_data=order)


def submit_crypto_exit_order(trading_client: TradingClient, symbol: str, qty: float):
    order = MarketOrderRequest(
        symbol=symbol,
        qty=abs(qty),
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC,
    )
    log.info(f"{symbol}: submitting crypto SELL market order with qty={abs(qty)}")
    return shared.retry_api_call(trading_client.submit_order, order_data=order)


def submit_crypto_short_entry_order(trading_client: TradingClient, symbol: str, notional: float):
    """Sell short — opens a crypto short position."""
    order = MarketOrderRequest(
        symbol=symbol,
        notional=notional,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC,
    )
    log.info(f"{symbol}: submitting crypto SHORT (sell) market order with notional=${notional:.2f}")
    return shared.retry_api_call(trading_client.submit_order, order_data=order)


def submit_crypto_short_exit_order(trading_client: TradingClient, symbol: str, qty: float):
    """Buy to cover — closes a crypto short position."""
    order = MarketOrderRequest(
        symbol=symbol,
        qty=abs(qty),
        side=OrderSide.BUY,
        time_in_force=TimeInForce.GTC,
    )
    log.info(f"{symbol}: submitting crypto BUY (cover short) market order with qty={abs(qty)}")
    return shared.retry_api_call(trading_client.submit_order, order_data=order)


def _check_daily_loss_limit(log_rows: list[dict], equity: float) -> bool:
    """Return True if today's realized losses have breached CRYPTO_MAX_DAILY_LOSS_PCT.

    When True, callers should halt new entries for the remainder of the day.
    Open position management is unaffected.
    """
    today = datetime.now(timezone.utc).date().isoformat()
    realized_today = sum(
        float(r.get("realized_pnl", 0.0))
        for r in log_rows
        if r.get("side") in {"SELL", "COVER"} and str(r.get("timestamp_utc", "")).startswith(today)
    )
    limit = equity * CRYPTO_MAX_DAILY_LOSS_PCT
    if realized_today < -limit:
        log.warning(
            "Daily loss limit reached: realized_today=$%.2f limit=-$%.2f — halting new entries",
            realized_today,
            limit,
        )
        monitor.notify_daily_loss_limit("crypto", realized_today, limit, equity)
        return True
    return False


def scan_for_entries(trading_client: TradingClient, data_client: CryptoHistoricalDataClient) -> None:
    open_positions = get_open_crypto_positions(trading_client)
    equity = float(shared.retry_api_call(trading_client.get_account).equity)
    log_rows = load_crypto_trade_log()

    if _check_daily_loss_limit(log_rows, equity):
        return

    log.info(f"Crypto open positions: {sorted(open_positions.keys())}")
    for symbol in CRYPTO_WATCHLIST:
        position_key = symbol.replace("/", "")
        if symbol in open_positions or position_key in open_positions:
            log.info(f"{symbol}: SKIP already open")
            continue
        if len(open_positions) >= MAX_CRYPTO_POSITIONS:
            log.info("Max crypto open positions reached")
            break

        try:
            raw = fetch_crypto_bars(data_client, symbol)
            if raw.empty:
                log.info(f"{symbol}: SKIP no bars")
                continue

            df = shared.calculate_indicators(raw)
            latest = df.iloc[-1]

            # Evaluate long signal
            long_signal, long_reason = evaluate_crypto_entry(df)
            short_signal, short_reason = evaluate_crypto_short_entry(df)
            log.info(
                f"{symbol}: long={long_signal} | short={short_signal} | "
                f"close={latest['close']:.2f} RSI={latest.get('rsi', 0):.1f} ADX={latest.get('adx', 0):.1f} | "
                f"reason=long:{long_reason}; short:{short_reason}"
            )

            if long_signal:
                notional = calculate_notional_size(equity, float(latest["close"]), float(latest["atr"]))
                if notional <= 0:
                    log.info(f"{symbol}: SKIP notional too small")
                    continue

                order = submit_crypto_entry_order(trading_client, symbol, notional)
                stop = round(float(latest["close"]) - float(latest["atr"]) * CRYPTO_STOP_ATR_MULTIPLIER, 4)
                target = round(float(latest["close"]) + float(latest["atr"]) * CRYPTO_TARGET_ATR_MULTIPLIER, 4)
                log_rows.append(
                    {
                        "timestamp_utc": shared.now_utc_iso(),
                        "symbol": symbol,
                        "side": "BUY",
                        "order_id": str(order.id),
                        "entry_reference_price": round(float(latest["close"]), 4),
                        "stop_loss_price": stop,
                        "take_profit_price": target,
                        "strategy": "crypto_momentum_v1",
                        "closed": False,
                        "notes": {"reason": long_reason, "notional": notional},
                    }
                )
                save_crypto_trade_log(log_rows)
                open_positions[symbol] = True  # count toward cap for remainder of loop
                monitor.notify_trade_entry(
                    bot="crypto",
                    symbol=symbol,
                    direction="long",
                    qty=f"${notional:.2f} notional",
                    entry_price=round(float(latest["close"]), 2),
                    stop=stop,
                    target=target,
                )

            elif short_signal:
                notional = calculate_notional_size(equity, float(latest["close"]), float(latest["atr"]))
                if notional <= 0:
                    log.info(f"{symbol}: SKIP notional too small")
                    continue

                order = submit_crypto_short_entry_order(trading_client, symbol, notional)
                # For shorts: stop is ABOVE entry, target is BELOW entry
                stop = round(float(latest["close"]) + float(latest["atr"]) * CRYPTO_STOP_ATR_MULTIPLIER, 4)
                target = round(float(latest["close"]) - float(latest["atr"]) * CRYPTO_TARGET_ATR_MULTIPLIER, 4)
                log_rows.append(
                    {
                        "timestamp_utc": shared.now_utc_iso(),
                        "symbol": symbol,
                        "side": "SHORT",
                        "order_id": str(order.id),
                        "entry_reference_price": round(float(latest["close"]), 4),
                        "stop_loss_price": stop,
                        "take_profit_price": target,
                        "strategy": "crypto_momentum_v1",
                        "closed": False,
                        "notes": {"reason": short_reason, "notional": notional},
                    }
                )
                save_crypto_trade_log(log_rows)
                open_positions[symbol] = True  # count toward cap for remainder of loop
                monitor.notify_trade_entry(
                    bot="crypto",
                    symbol=symbol,
                    direction="short",
                    qty=f"${notional:.2f} notional",
                    entry_price=round(float(latest["close"]), 2),
                    stop=stop,
                    target=target,
                )

        except Exception as exc:
            log.error("%s: error during entry scan, skipping: %s", symbol, exc)


def manage_open_positions(trading_client: TradingClient, data_client: CryptoHistoricalDataClient) -> None:
    positions = get_open_crypto_positions(trading_client)
    if not positions:
        log.info("No open crypto positions")
        return

    log_rows = load_crypto_trade_log()
    for symbol_key, position in positions.items():
        try:
            symbol = symbol_key if "/" in symbol_key else f"{symbol_key[:-3]}/{symbol_key[-3:]}"
            raw = fetch_crypto_bars(data_client, symbol)
            if raw.empty:
                continue
            df = shared.calculate_indicators(raw)
            latest = df.iloc[-1]
            price = float(latest["close"])
            qty = float(position.qty)

            # Alpaca returns negative qty for short positions
            is_short = qty < 0

            open_entry = next(
                (
                    row
                    for row in reversed(log_rows)
                    if row.get("symbol", "").replace("/", "") == symbol_key.replace("/", "")
                    and row.get("side") == ("SHORT" if is_short else "BUY")
                    and not row.get("closed", False)
                ),
                None,
            )
            if not open_entry:
                continue

            stop_price = float(open_entry["stop_loss_price"])
            target_price = float(open_entry["take_profit_price"])
            exit_reason = None

            if is_short:
                # Short exits: stop triggered when price moves UP, target when price moves DOWN
                if price >= stop_price:
                    exit_reason = "STOP_LOSS"
                elif price <= target_price:
                    exit_reason = "TAKE_PROFIT"
                elif latest.get("ema_fast", 0) >= latest.get("ema_slow", 0):
                    # EMA crossed back UP — trend reversed against short
                    exit_reason = "TREND_REVERSAL"
            else:
                # Long exits: stop triggered when price moves DOWN, target when price moves UP
                if price <= stop_price:
                    exit_reason = "STOP_LOSS"
                elif price >= target_price:
                    exit_reason = "TAKE_PROFIT"
                elif latest.get("ema_fast", 0) < latest.get("ema_slow", 0):
                    exit_reason = "TREND_REVERSAL"

            if not exit_reason:
                dir_label = "SHORT" if is_short else "LONG"
                log.info(
                    f"{symbol}: HOLD {dir_label} | px={price:.2f} stop={stop_price:.2f} target={target_price:.2f}"
                )
                continue

            if is_short:
                order = submit_crypto_short_exit_order(trading_client, symbol, qty)
                # Short PnL: profit when price falls below entry
                pnl = round((float(open_entry["entry_reference_price"]) - price) * abs(qty), 2)
                exit_side = "COVER"
                direction = "short"
            else:
                order = submit_crypto_exit_order(trading_client, symbol, qty)
                pnl = round((price - float(open_entry["entry_reference_price"])) * abs(qty), 2)
                exit_side = "SELL"
                direction = "long"

            open_entry["closed"] = True
            open_entry["exit"] = {
                "timestamp_utc": shared.now_utc_iso(),
                "reason": exit_reason,
                "exit_reference_price": round(price, 4),
                "order_id": str(order.id),
                "realized_pnl": pnl,
            }
            log_rows.append(
                {
                    "timestamp_utc": shared.now_utc_iso(),
                    "symbol": symbol,
                    "side": exit_side,
                    "order_id": str(order.id),
                    "realized_pnl": pnl,
                    "notes": {"reason": exit_reason},
                }
            )
            save_crypto_trade_log(log_rows)
            monitor.notify_trade_exit(
                bot="crypto",
                symbol=symbol,
                direction=direction,
                reason=exit_reason,
                exit_price=price,
                pnl=pnl,
            )
            log.info(f"{symbol}: EXIT {exit_reason} {direction.upper()} | pnl=${pnl:.2f}")
        except Exception as exc:
            log.error("%s: error during position management, skipping: %s", symbol_key, exc)


def print_crypto_summary(trading_client: TradingClient) -> None:
    log_rows = load_crypto_trade_log()
    open_positions = get_open_crypto_positions(trading_client)
    equity = float(shared.retry_api_call(trading_client.get_account).equity)
    today = datetime.now(timezone.utc).date().isoformat()
    realized = sum(
        float(row.get("realized_pnl", 0.0))
        for row in log_rows
        if row.get("side") in {"SELL", "COVER"} and str(row.get("timestamp_utc", "")).startswith(today)
    )
    log.info(
        "CRYPTO SUMMARY | equity=$%.2f | open_positions=%d | realized_today=$%.2f",
        equity,
        len(open_positions),
        realized,
    )
    monitor.log_equity_curve("crypto", equity, len(open_positions))


def run_bot() -> None:
    shared.load_environment()
    log.info("=" * 70)
    log.info("CRYPTO PAPER TRADING BOT STARTUP | universe=%s", ", ".join(CRYPTO_WATCHLIST))
    log.info("=" * 70)

    config = validate_crypto_startup_config()
    trading_client, data_client = create_clients(config)

    account = shared.retry_api_call(trading_client.get_account)
    log.info(
        "Connected to Alpaca paper account | status=%s | equity=$%.2f | buying_power=$%.2f",
        account.status,
        float(account.equity),
        float(account.buying_power),
    )
    log.info("daily_loss_limit=%.1f%%", CRYPTO_MAX_DAILY_LOSS_PCT * 100)

    manage_open_positions(trading_client, data_client)
    scan_for_entries(trading_client, data_client)
    print_crypto_summary(trading_client)

    # Monitoring — heartbeat + optional run-complete notification
    try:
        _account = shared.retry_api_call(trading_client.get_account)
        _equity = float(_account.equity)
        _open_n = len(get_open_crypto_positions(trading_client))
        _log_rows = load_crypto_trade_log()
        today = datetime.now(timezone.utc).date().isoformat()
        _realized = sum(
            float(r.get("realized_pnl", 0.0))
            for r in _log_rows
            if r.get("side") in {"SELL", "COVER"} and str(r.get("timestamp_utc", "")).startswith(today)
        )
        monitor.write_heartbeat("crypto", _equity, _open_n)
        monitor.notify_run_complete("crypto", _equity, _open_n, _realized)
    except Exception as _mon_exc:
        log.warning("monitor phase failed (non-critical): %s", _mon_exc)

    log.info("Crypto run complete — exiting")


if __name__ == "__main__":
    run_bot()
