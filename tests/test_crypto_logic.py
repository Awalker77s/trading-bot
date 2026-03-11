import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import crypto_main  # noqa: E402


def _make_short_df(fast_above_slow_count: int = 5, bars_below_before_current: int = 0) -> pd.DataFrame:
    """Build a minimal DataFrame that passes all short-entry filters except EMA crossover.

    fast_above_slow_count: how many leading candles have fast > slow (before the cross).
    bars_below_before_current: how many candles immediately before the current bar have fast < slow.
    """
    n = 65
    close = np.full(n, 100.0)
    sma50 = np.full(n, 110.0)   # close < sma50 → downtrend context
    rsi = np.full(n, 60.0)      # > 55
    adx = np.full(n, 25.0)      # > 20
    atr_pct = np.full(n, 0.01)  # within [MIN, MAX]
    volume = np.full(n, 1000.0)
    volume_sma = np.full(n, 500.0)

    # EMA layout: fast > slow for first N bars, then fast < slow for the rest
    ema_fast = np.full(n, 101.0)
    ema_slow = np.full(n, 100.0)

    # bars where fast is below slow: the last (bars_below_before_current + 1) rows
    cross_start = n - bars_below_before_current - 1
    ema_fast[cross_start:] = 99.0  # fast < slow

    df = pd.DataFrame({
        "close": close,
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "sma50": sma50,
        "rsi": rsi,
        "adx": adx,
        "atr_pct": atr_pct,
        "volume": volume,
        "volume_sma": volume_sma,
    })
    return df


def test_calculate_notional_size_returns_positive_for_valid_inputs():
    notional = crypto_main.calculate_notional_size(
        equity=10000,
        price=70000,
        atr=1200,
    )
    assert notional >= crypto_main.CRYPTO_MIN_NOTIONAL_USD


def test_calculate_notional_size_returns_zero_for_bad_inputs():
    assert crypto_main.calculate_notional_size(0, 100, 2) == 0.0
    assert crypto_main.calculate_notional_size(10000, 0, 2) == 0.0
    assert crypto_main.calculate_notional_size(10000, 100, 0) == 0.0


def test_evaluate_crypto_entry_insufficient_data():
    df = pd.DataFrame({"close": [1, 2, 3]})
    signal, reason = crypto_main.evaluate_crypto_entry(df)
    assert signal is False
    assert reason == "Insufficient data"


def test_short_entry_primary_crossover_fires():
    """Exact crossover candle (prev fast >= slow, current fast < slow) should signal."""
    df = _make_short_df(bars_below_before_current=0)
    signal, reason = crypto_main.evaluate_crypto_short_entry(df)
    assert signal is True
    assert reason == "Fresh EMA crossover DOWN"


def test_short_entry_fallback_1_candle_ago():
    """Fast crossed below slow 1 candle ago → fallback should fire."""
    df = _make_short_df(bars_below_before_current=1)
    signal, reason = crypto_main.evaluate_crypto_short_entry(df)
    assert signal is True
    assert "fallback" in reason


def test_short_entry_fallback_at_lookback_boundary():
    """Fast crossed below slow exactly CROSSOVER_LOOKBACK candles ago → still valid."""
    lookback = crypto_main.CROSSOVER_LOOKBACK
    df = _make_short_df(bars_below_before_current=lookback - 1)
    signal, reason = crypto_main.evaluate_crypto_short_entry(df)
    assert signal is True
    assert "fallback" in reason


def test_short_entry_fallback_expired_beyond_lookback():
    """Fast crossed below slow more than CROSSOVER_LOOKBACK candles ago → should NOT fire."""
    lookback = crypto_main.CROSSOVER_LOOKBACK
    df = _make_short_df(bars_below_before_current=lookback)
    signal, reason = crypto_main.evaluate_crypto_short_entry(df)
    assert signal is False
    assert "EMA crossover" in reason
