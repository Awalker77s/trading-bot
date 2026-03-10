import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import main  # noqa: E402


def test_position_size_long_stop_below_entry():
    qty, stop = main.position_size_from_atr(
        equity=10000,
        entry_price=100,
        atr=2,
        stop_multiplier=2,
        risk_per_trade=0.01,
        max_position_pct=0.1,
        direction="long",
    )
    assert qty > 0
    assert stop < 100


def test_position_size_short_stop_above_entry():
    qty, stop = main.position_size_from_atr(
        equity=10000,
        entry_price=100,
        atr=2,
        stop_multiplier=2,
        risk_per_trade=0.01,
        max_position_pct=0.1,
        direction="short",
    )
    assert qty > 0
    assert stop > 100


def test_evaluate_entry_insufficient_data():
    df = pd.DataFrame({"close": [1, 2, 3]})
    signal, reason, regime, direction = main.evaluate_entry(df)
    assert signal is False
    assert reason == "Insufficient data"
    assert direction == "long"


def test_validate_bars_df_rejects_invalid_rows():
    df = pd.DataFrame(
        {
            "timestamp": [1, 2],
            "open": [10, -1],
            "high": [11, 2],
            "low": [9, 1],
            "close": [10.5, 1.5],
            "volume": [1000, 2000],
        }
    )
    cleaned = main.validate_bars_df(df, "TEST")
    assert len(cleaned) == 1
