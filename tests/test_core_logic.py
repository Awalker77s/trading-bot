import sys
from pathlib import Path

import pytest
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


def test_get_required_env_rejects_placeholder(monkeypatch):
    monkeypatch.setenv("ALPACA_KEY", "your_api_key_here")
    monkeypatch.setenv("APCA_API_KEY_ID", "")
    with pytest.raises(ValueError):
        main.get_required_env("ALPACA_KEY", "APCA_API_KEY_ID")


def test_retry_non_retryable_error_raises_immediately():
    class DummyError(Exception):
        pass

    calls = {"n": 0}

    def failing_call():
        calls["n"] += 1
        raise DummyError("400 bad request")

    with pytest.raises(DummyError):
        main.retry_api_call(failing_call, max_retries=3)
    assert calls["n"] == 1


def test_retry_retryable_error_retries(monkeypatch):
    calls = {"n": 0}

    def flaky_call():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("503 service unavailable")
        return "ok"

    monkeypatch.setattr(main.time, "sleep", lambda _x: None)
    result = main.retry_api_call(flaky_call, max_retries=3)
    assert result == "ok"
    assert calls["n"] == 3


def test_get_open_positions_filters_non_stock_and_logs(monkeypatch):
    class DummyPosition:
        def __init__(self, symbol, asset_class):
            self.symbol = symbol
            self.asset_class = asset_class

    class DummyClient:
        def get_all_positions(self):
            return [
                DummyPosition("AAPL", "us_equity"),
                DummyPosition("BTCUSD", "crypto"),
            ]

    logs = []
    monkeypatch.setattr(main.log, "info", lambda msg, *args: logs.append(msg % args if args else msg))

    positions = main.get_open_positions(DummyClient())

    assert list(positions.keys()) == ["AAPL"]
    assert any("Ignoring non-stock open position" in message for message in logs)


def test_scan_for_new_entries_ignores_recently_submitted_exits(monkeypatch):
    monkeypatch.setattr(main, "load_trade_log", lambda: [])
    monkeypatch.setattr(main, "get_account_equity", lambda _client: 100000.0)
    monkeypatch.setattr(main, "daily_loss_limit_hit", lambda *_args: (False, 0.0, 0.0))
    monkeypatch.setattr(main, "WATCHLIST", [])

    class DummyPosition:
        def __init__(self, symbol):
            self.symbol = symbol
            self.asset_class = "us_equity"

    class DummyClient:
        def get_all_positions(self):
            return [DummyPosition("SPY")]

    captured = []
    monkeypatch.setattr(main.log, "info", lambda message, *args: captured.append(message % args if args else message))

    main.scan_for_new_entries(
        DummyClient(),
        data_client=None,
        analysis_cache={},
        recently_submitted_exits={"SPY"},
    )

    assert any("Ignoring recently submitted exits" in line for line in captured)
