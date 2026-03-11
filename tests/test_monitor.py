"""Tests for src/monitor.py — heartbeat, equity curve, and notification helpers."""

import json
import sys
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import monitor  # noqa: E402


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

def test_write_heartbeat_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "_LOG_DIR", tmp_path)
    monitor.write_heartbeat("stocks", equity=50000.0, open_positions=3)
    hb_file = tmp_path / "heartbeat_stocks.json"
    assert hb_file.exists()
    data = json.loads(hb_file.read_text())
    assert data["bot"] == "stocks"
    assert data["equity"] == 50000.0
    assert data["open_positions"] == 3
    assert data["status"] == "ok"
    assert "timestamp_utc" in data


def test_write_heartbeat_does_not_raise_on_permission_error(monkeypatch):
    """Heartbeat failures must be swallowed, not propagate."""
    monkeypatch.setattr(monitor, "_LOG_DIR", Path("/nonexistent/path/that/cannot/be/created"))
    # Should log a warning but not raise
    monitor.write_heartbeat("stocks", equity=1000.0, open_positions=0)


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

def test_log_equity_curve_creates_csv_with_header(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "_LOG_DIR", tmp_path)
    monitor.log_equity_curve("crypto", equity=12345.67, open_positions=2)
    csv_file = tmp_path / "crypto_equity_curve.csv"
    assert csv_file.exists()
    lines = csv_file.read_text().splitlines()
    assert lines[0] == "timestamp,date,equity,open_positions"
    assert "12345.67" in lines[1]
    assert "2" in lines[1]


def test_log_equity_curve_appends_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "_LOG_DIR", tmp_path)
    monitor.log_equity_curve("crypto", 10000.0, 1)
    monitor.log_equity_curve("crypto", 10500.0, 2)
    lines = (tmp_path / "crypto_equity_curve.csv").read_text().splitlines()
    assert len(lines) == 3  # header + 2 data rows


# ---------------------------------------------------------------------------
# Webhook — _send_webhook
# ---------------------------------------------------------------------------

def test_send_webhook_skipped_when_no_url(monkeypatch):
    monkeypatch.delenv("WEBHOOK_URL", raising=False)
    # Should not raise and should not attempt a network call
    with patch("urllib.request.urlopen") as mock_open:
        monitor._send_webhook({"content": "test"})
        mock_open.assert_not_called()


def test_send_webhook_posts_json(monkeypatch):
    monkeypatch.setenv("WEBHOOK_URL", "https://example.com/hook")
    mock_response = MagicMock()
    mock_response.read.return_value = b""
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_response) as mock_open:
        monitor._send_webhook({"content": "hello"})
        mock_open.assert_called_once()
        req = mock_open.call_args[0][0]
        assert req.full_url == "https://example.com/hook"
        assert json.loads(req.data) == {"content": "hello"}


def test_send_webhook_swallows_network_error(monkeypatch):
    monkeypatch.setenv("WEBHOOK_URL", "https://example.com/hook")
    with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
        # Must not raise
        monitor._send_webhook({"content": "test"})


# ---------------------------------------------------------------------------
# Notification helpers honour env flags
# ---------------------------------------------------------------------------

def test_notify_trade_entry_skipped_when_flag_off(monkeypatch):
    monkeypatch.setenv("NOTIFY_ON_TRADES", "false")
    with patch.object(monitor, "_send_webhook") as mock_wh:
        monitor.notify_trade_entry("stocks", "AAPL", "long", 10, 150.0, 145.0, 162.0)
        mock_wh.assert_not_called()


def test_notify_trade_exit_sends_when_flag_on(monkeypatch):
    monkeypatch.setenv("NOTIFY_ON_TRADES", "true")
    monkeypatch.setenv("WEBHOOK_URL", "https://example.com/hook")
    with patch.object(monitor, "_send_webhook") as mock_wh:
        monitor.notify_trade_exit("stocks", "AAPL", "long", "TAKE_PROFIT", 162.0, 120.0, 2.0)
        mock_wh.assert_called_once()
        payload = mock_wh.call_args[0][0]
        assert "AAPL" in payload["content"]
        assert "TAKE_PROFIT" in payload["content"]
        assert "+120.00" in payload["content"]


def test_notify_daily_loss_limit_skipped_when_flag_off(monkeypatch):
    monkeypatch.setenv("NOTIFY_ON_ERRORS", "false")
    with patch.object(monitor, "_send_webhook") as mock_wh:
        monitor.notify_daily_loss_limit("stocks", -3000.0, 3000.0, 97000.0)
        mock_wh.assert_not_called()


def test_notify_run_complete_skipped_by_default(monkeypatch):
    monkeypatch.delenv("NOTIFY_ON_SUMMARY", raising=False)
    with patch.object(monitor, "_send_webhook") as mock_wh:
        monitor.notify_run_complete("stocks", 100000.0, 3, 250.0)
        mock_wh.assert_not_called()


def test_notify_run_complete_sends_when_opted_in(monkeypatch):
    monkeypatch.setenv("NOTIFY_ON_SUMMARY", "true")
    monkeypatch.setenv("WEBHOOK_URL", "https://example.com/hook")
    with patch.object(monitor, "_send_webhook") as mock_wh:
        monitor.notify_run_complete("crypto", 25000.0, 2, -50.0)
        mock_wh.assert_called_once()
        payload = mock_wh.call_args[0][0]
        assert "CRYPTO" in payload["content"]
        assert "-50.00" in payload["content"]


# ---------------------------------------------------------------------------
# health_check module
# ---------------------------------------------------------------------------

def test_health_check_missing_heartbeat(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import health_check
    monkeypatch.setattr(health_check, "_LOG_DIR", tmp_path)
    healthy = health_check.check_bots(["stocks"], max_age_minutes=60)
    assert healthy is False


def test_health_check_fresh_heartbeat(tmp_path, monkeypatch):
    import health_check
    monkeypatch.setattr(health_check, "_LOG_DIR", tmp_path)
    # Write a fresh heartbeat
    monkeypatch.setattr(monitor, "_LOG_DIR", tmp_path)
    monitor.write_heartbeat("stocks", 50000.0, 2)
    healthy = health_check.check_bots(["stocks"], max_age_minutes=60)
    assert healthy is True


def test_health_check_overdue_heartbeat(tmp_path, monkeypatch):
    import health_check
    monkeypatch.setattr(health_check, "_LOG_DIR", tmp_path)
    # Write a heartbeat with a timestamp far in the past
    old_ts = "2000-01-01T00:00:00+00:00"
    hb = {"bot": "crypto", "status": "ok", "timestamp_utc": old_ts,
          "equity": 1000.0, "open_positions": 0}
    (tmp_path / "heartbeat_crypto.json").write_text(json.dumps(hb))
    healthy = health_check.check_bots(["crypto"], max_age_minutes=60)
    assert healthy is False
