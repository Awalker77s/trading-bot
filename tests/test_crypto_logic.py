import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import crypto_main  # noqa: E402


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
