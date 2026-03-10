import os
import math
import json
from datetime import datetime, timezone, timedelta, date

import pandas as pd
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


ENV_PATH = "/root/trading-bot/config/.env"
LOG_DIR = "/root/trading-bot/logs"
TRADE_LOG_FILE = os.path.join(LOG_DIR, "trade_log.json")

WATCHLIST = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA"]

RISK_PER_TRADE = 0.01
STOP_LOSS_PCT = 0.01
TAKE_PROFIT_PCT = 0.02
MAX_OPEN_POSITIONS = 6
MAX_DAILY_LOSS_PCT = 0.03

RSI_MIN = 50
RSI_MAX = 70
PULLBACK_THRESHOLD = 0.015
MIN_PRICE = 20


def ensure_directories() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


def load_environment() -> None:
    load_dotenv(ENV_PATH)


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


def load_trade_log() -> list:
    ensure_directories()
    if not os.path.exists(TRADE_LOG_FILE):
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


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["sma200"] = df["close"].rolling(200).mean()
    df["rsi14"] = compute_rsi(df["close"], 14)
    return df


def is_bullish_trend(row: pd.Series) -> bool:
    return (
        row["close"] > row["sma200"]
        and row["sma20"] > row["sma200"]
    )


def near_pullback_zone(row: pd.Series) -> bool:
    if pd.isna(row["sma20"]) or row["sma20"] == 0:
        return False
    distance = abs(row["close"] - row["sma20"]) / row["sma20"]
    return distance <= PULLBACK_THRESHOLD


def has_bounce_confirmation(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    return latest["close"] > prev["close"] and latest["low"] >= prev["low"]


def signal_long(df: pd.DataFrame) -> tuple[bool, str]:
    if len(df) < 200:
        return False, "Not enough bars"

    latest = df.iloc[-1]

    required_fields = ["close", "sma20", "sma50", "sma200", "rsi14"]
    for field in required_fields:
        if pd.isna(latest[field]):
            return False, f"Missing indicator: {field}"

    if latest["close"] < MIN_PRICE:
        return False, "Price below minimum threshold"

    if not is_bullish_trend(latest):
        return False, "Trend conditions not met"

    if not (RSI_MIN <= latest["rsi14"] <= RSI_MAX):
        return False, "RSI not in valid range"

    if not near_pullback_zone(latest):
        return False, "Not near 20 SMA pullback zone"

    if not has_bounce_confirmation(df):
        return False, "Bounce confirmation failed"

    return True, "Long signal detected"


def position_size_from_risk(
    equity: float,
    entry_price: float,
    stop_loss_pct: float,
    risk_per_trade: float
) -> int:
    risk_dollars = equity * risk_per_trade
    stop_distance = entry_price * stop_loss_pct

    if stop_distance <= 0:
        return 0

    qty = math.floor(risk_dollars / stop_distance)
    return max(qty, 0)


def create_clients():
    api_key = get_required_env("ALPACA_KEY")
    api_secret = get_required_env("ALPACA_SECRET")

    trading_client = TradingClient(api_key, api_secret, paper=True)
    data_client = StockHistoricalDataClient(api_key, api_secret)

    return trading_client, data_client


def fetch_daily_bars(data_client: StockHistoricalDataClient, symbol: str) -> pd.DataFrame:
    start_date = datetime.now(timezone.utc) - timedelta(days=450)

    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        start=start_date,
        adjustment="raw"
    )

    bars = data_client.get_stock_bars(request)
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

    print(f"{symbol}: fetched {len(df)} bars")
    return df


def get_open_positions(trading_client: TradingClient) -> dict:
    positions = trading_client.get_all_positions()
    return {p.symbol: p for p in positions}


def get_account_equity(trading_client: TradingClient) -> float:
    account = trading_client.get_account()
    return float(account.equity)


def place_market_buy(trading_client: TradingClient, symbol: str, qty: int):
    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY
    )
    return trading_client.submit_order(order_data=order_data)


def place_market_sell(trading_client: TradingClient, symbol: str, qty: int):
    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY
    )
    return trading_client.submit_order(order_data=order_data)


def find_latest_open_buy(log_data: list, symbol: str) -> dict | None:
    for entry in reversed(log_data):
        if (
            entry.get("symbol") == symbol
            and entry.get("side") == "BUY"
            and not entry.get("closed", False)
        ):
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

        if entry.get("side") == "BUY" and entry.get("closed") and entry.get("exit"):
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


def daily_loss_limit_hit(log_data: list, account_equity: float) -> tuple[bool, float, float]:
    realized_today = get_realized_pnl_today(log_data)
    loss_limit_dollars = account_equity * MAX_DAILY_LOSS_PCT

    hit = realized_today <= -loss_limit_dollars
    return hit, realized_today, loss_limit_dollars


def calculate_realized_pnl(buy_entry: dict, exit_price: float, qty: int) -> float:
    entry_price = float(buy_entry["entry_reference_price"])
    return round((exit_price - entry_price) * qty, 2)


def win_or_loss(realized_pnl: float) -> str:
    if realized_pnl > 0:
        return "WIN"
    if realized_pnl < 0:
        return "LOSS"
    return "BREAKEVEN"


def manage_open_positions(trading_client: TradingClient, data_client: StockHistoricalDataClient) -> None:
    open_positions = get_open_positions(trading_client)
    log_data = load_trade_log()

    if not open_positions:
        print("No open positions to manage.")
        return

    for symbol, position in open_positions.items():
        try:
            print(f"\nManaging open position: {symbol}")
            bars_df = fetch_daily_bars(data_client, symbol)
            if bars_df.empty:
                print(f"Skipping {symbol}: no market data for exit check")
                continue

            latest = bars_df.iloc[-1]
            current_price = float(latest["close"])
            qty = int(float(position.qty))

            buy_entry = find_latest_open_buy(log_data, symbol)
            if not buy_entry:
                print(f"No matching BUY log found for {symbol}; skipping exit logic")
                continue

            stop_price = float(buy_entry["stop_loss_price"])
            take_profit_price = float(buy_entry["take_profit_price"])

            exit_reason = None
            if current_price <= stop_price:
                exit_reason = "STOP_LOSS"
            elif current_price >= take_profit_price:
                exit_reason = "TAKE_PROFIT"

            print(
                f"{symbol}: current={current_price:.2f}, "
                f"stop={stop_price:.2f}, target={take_profit_price:.2f}"
            )

            if not exit_reason:
                print(f"{symbol}: hold")
                continue

            print(f"{symbol}: exit triggered -> {exit_reason}")
            sell_order = place_market_sell(trading_client, symbol, qty)

            realized_pnl = calculate_realized_pnl(buy_entry, current_price, qty)

            exit_info = {
                "timestamp_utc": now_utc_iso(),
                "reason": exit_reason,
                "exit_reference_price": current_price,
                "sell_order_id": str(sell_order.id),
                "qty": qty,
                "realized_pnl": realized_pnl,
                "result": win_or_loss(realized_pnl),
            }

            for entry in reversed(log_data):
                if (
                    entry.get("order_id") == buy_entry["order_id"]
                    and entry.get("side") == "BUY"
                    and not entry.get("closed", False)
                ):
                    entry["closed"] = True
                    entry["exit"] = exit_info
                    break

            sell_log = {
                "timestamp_utc": now_utc_iso(),
                "trade_date": utc_today_str(),
                "symbol": symbol,
                "side": "SELL",
                "qty": qty,
                "entry_reference_price": float(buy_entry["entry_reference_price"]),
                "exit_reference_price": current_price,
                "realized_pnl": realized_pnl,
                "result": win_or_loss(realized_pnl),
                "strategy": "trend_pullback_momentum_v3",
                "order_id": str(sell_order.id),
                "notes": {
                    "reason": exit_reason,
                    "linked_buy_order_id": buy_entry["order_id"]
                }
            }
            log_data.append(sell_log)
            save_trade_log(log_data)

            print(
                f"{symbol}: sell order submitted. "
                f"Order ID: {sell_order.id} | Realized P/L: {realized_pnl:.2f}"
            )

        except Exception as e:
            print(f"Error managing {symbol}: {e}")


def scan_for_new_entries(trading_client: TradingClient, data_client: StockHistoricalDataClient) -> None:
    log_data = load_trade_log()
    open_positions = get_open_positions(trading_client)
    account_equity = get_account_equity(trading_client)

    print(f"Current open positions: {sorted(open_positions.keys())}")

    loss_limit_hit, realized_today, loss_limit_dollars = daily_loss_limit_hit(log_data, account_equity)
    print(
        f"Realized P/L today: {realized_today:.2f} | "
        f"Daily loss limit: -{loss_limit_dollars:.2f}"
    )

    if loss_limit_hit:
        print("Daily loss limit hit. No new trades will be opened today.")
        return

    if len(open_positions) >= MAX_OPEN_POSITIONS:
        print("Max open positions reached. No new trades.")
        return

    for symbol in WATCHLIST:
        print(f"\nChecking {symbol}...")
        try:
            if symbol in open_positions:
                print(f"Skipping {symbol}: already in position")
                continue

            if has_exited_today(log_data, symbol):
                print(f"Skipping {symbol}: same-day re-entry blocked")
                continue

            df = fetch_daily_bars(data_client, symbol)
            if df.empty:
                print(f"Skipping {symbol}: no bar data")
                continue

            df = calculate_indicators(df)
            signal, reason = signal_long(df)
            latest = df.iloc[-1]

            print(f"Signal result: {signal} | Reason: {reason}")
            print(
                f"Close={latest['close']:.2f}, "
                f"SMA20={latest['sma20']:.2f}, "
                f"SMA50={latest['sma50']:.2f}, "
                f"SMA200={latest['sma200']:.2f}, "
                f"RSI14={latest['rsi14']:.2f}"
            )

            if not signal:
                continue

            entry_price = float(latest["close"])
            qty = position_size_from_risk(
                equity=account_equity,
                entry_price=entry_price,
                stop_loss_pct=STOP_LOSS_PCT,
                risk_per_trade=RISK_PER_TRADE
            )

            if qty < 1:
                print(f"Skipping {symbol}: calculated quantity < 1")
                continue

            estimated_position_value = qty * entry_price
            print(f"Placing paper BUY for {symbol} | qty={qty} | est value=${estimated_position_value:.2f}")

            order = place_market_buy(trading_client, symbol, qty)

            stop_price = round(entry_price * (1 - STOP_LOSS_PCT), 2)
            take_profit_price = round(entry_price * (1 + TAKE_PROFIT_PCT), 2)

            log_entry = {
                "timestamp_utc": now_utc_iso(),
                "trade_date": utc_today_str(),
                "symbol": symbol,
                "side": "BUY",
                "qty": qty,
                "entry_reference_price": entry_price,
                "estimated_position_value": estimated_position_value,
                "risk_dollars": round(account_equity * RISK_PER_TRADE, 2),
                "stop_loss_price": stop_price,
                "take_profit_price": take_profit_price,
                "strategy": "trend_pullback_momentum_v3",
                "order_id": str(order.id),
                "closed": False,
                "notes": {
                    "reason": reason,
                    "close": entry_price,
                    "sma20": float(latest["sma20"]),
                    "sma50": float(latest["sma50"]),
                    "sma200": float(latest["sma200"]),
                    "rsi14": float(latest["rsi14"]),
                }
            }
            append_trade_log(log_entry)

            print(f"Order submitted for {symbol}. Order ID: {order.id}")
            print(f"Logged trade to {TRADE_LOG_FILE}")
            break

        except Exception as e:
            print(f"Error on {symbol}: {e}")


def print_daily_summary(trading_client: TradingClient) -> None:
    log_data = load_trade_log()
    account_equity = get_account_equity(trading_client)

    realized_today = get_realized_pnl_today(log_data)
    loss_limit_dollars = account_equity * MAX_DAILY_LOSS_PCT

    today_sells = [
        x for x in log_data
        if x.get("side") == "SELL" and x.get("timestamp_utc", "").startswith(utc_today_str())
    ]

    wins = sum(1 for x in today_sells if x.get("realized_pnl", 0) > 0)
    losses = sum(1 for x in today_sells if x.get("realized_pnl", 0) < 0)
    breakeven = sum(1 for x in today_sells if x.get("realized_pnl", 0) == 0)

    print("\n" + "=" * 50)
    print("DAILY SUMMARY")
    print(f"Date: {utc_today_str()}")
    print(f"Closed trades today: {len(today_sells)}")
    print(f"Wins: {wins} | Losses: {losses} | Breakeven: {breakeven}")
    print(f"Realized P/L today: {realized_today:.2f}")
    print(f"Daily loss limit: -{loss_limit_dollars:.2f}")
    print("=" * 50 + "\n")


def run_bot() -> None:
    ensure_directories()
    load_environment()

    trading_client, data_client = create_clients()

    account = trading_client.get_account()
    print("Connected to Alpaca paper account")
    print(f"Status: {account.status}")
    print(f"Equity: {account.equity}")
    print(f"Buying Power: {account.buying_power}")
    print("-" * 50)

    manage_open_positions(trading_client, data_client)
    print("\n" + "=" * 50 + "\n")
    scan_for_new_entries(trading_client, data_client)
    print_daily_summary(trading_client)


if __name__ == "__main__":
    run_bot()
