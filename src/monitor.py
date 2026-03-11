"""Bot monitoring utilities — heartbeat files and optional webhook notifications.

All public functions are internally wrapped so that a monitoring failure can
NEVER interrupt live trading execution.

Configuration (via .env / environment variables):
    WEBHOOK_URL         Discord/Slack/generic webhook URL. Leave unset to
                        disable all webhook notifications.
    NOTIFY_ON_TRADES    Send notifications for entries and exits. Default: true
    NOTIFY_ON_ERRORS    Send notifications for errors and loss-limit hits.
                        Default: true
    NOTIFY_ON_SUMMARY   Send a run-complete summary notification. Default: false

Heartbeat files are always written regardless of webhook config.
"""

import json
import logging
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("trading_bot")

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
_LOG_DIR = _SCRIPT_DIR / "logs"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _flag(env_name: str, default: bool) -> bool:
    val = os.getenv(env_name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _send_webhook(payload: dict) -> None:
    """POST JSON to WEBHOOK_URL with a 5-second timeout. Fire-and-forget."""
    url = os.getenv("WEBHOOK_URL", "").strip()
    if not url:
        return
    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp.read()
    except Exception as exc:
        log.warning("monitor: webhook delivery failed (non-critical): %s", exc)


def _discord(text: str) -> dict:
    """Minimal Discord-compatible payload (also accepted by many other services)."""
    return {"content": text}


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

def write_heartbeat(bot_name: str, equity: float, open_positions: int,
                    status: str = "ok") -> None:
    """Write logs/heartbeat_{bot_name}.json after each successful run.

    External monitors (cron jobs, uptime tools, etc.) can read this file to
    verify the bot has run recently and report its last known equity.
    """
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = _LOG_DIR / f"heartbeat_{bot_name}.json"
        payload = {
            "bot": bot_name,
            "status": status,
            "timestamp_utc": _now_utc_iso(),
            "equity": round(float(equity), 2),
            "open_positions": int(open_positions),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("monitor: heartbeat written → %s", path.name)
    except Exception as exc:
        log.warning("monitor: write_heartbeat failed (non-critical): %s", exc)


# ---------------------------------------------------------------------------
# Equity curve (separate CSV per bot)
# ---------------------------------------------------------------------------

def log_equity_curve(bot_name: str, equity: float, open_positions: int) -> None:
    """Append a row to logs/{bot_name}_equity_curve.csv."""
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = _LOG_DIR / f"{bot_name}_equity_curve.csv"
        header_needed = not path.exists()
        today = datetime.now(timezone.utc).date().isoformat()
        with open(path, "a", encoding="utf-8") as f:
            if header_needed:
                f.write("timestamp,date,equity,open_positions\n")
            f.write(f"{_now_utc_iso()},{today},{equity:.2f},{open_positions}\n")
    except Exception as exc:
        log.warning("monitor: log_equity_curve failed (non-critical): %s", exc)


# ---------------------------------------------------------------------------
# Trade notifications
# ---------------------------------------------------------------------------

def notify_trade_entry(
    bot: str,
    symbol: str,
    direction: str,
    qty,
    entry_price: float,
    stop: float,
    target: float,
    regime: str = "",
) -> None:
    """Notify on a new trade entry."""
    if not _flag("NOTIFY_ON_TRADES", True):
        return
    try:
        emoji = "📈" if direction == "long" else "📉"
        regime_str = f" | regime={regime}" if regime else ""
        text = (
            f"{emoji} **{bot.upper()} ENTRY** | {symbol} {direction.upper()}\n"
            f"qty={qty} | entry≈${entry_price:.2f} | stop=${stop:.2f}"
            f" | target=${target:.2f}{regime_str}"
        )
        _send_webhook(_discord(text))
    except Exception as exc:
        log.warning("monitor: notify_trade_entry failed (non-critical): %s", exc)


def notify_trade_exit(
    bot: str,
    symbol: str,
    direction: str,
    reason: str,
    exit_price: float,
    pnl: float,
    r_multiple: float = 0.0,
) -> None:
    """Notify on a trade exit."""
    if not _flag("NOTIFY_ON_TRADES", True):
        return
    try:
        emoji = "✅" if pnl >= 0 else "❌"
        text = (
            f"{emoji} **{bot.upper()} EXIT** | {symbol} {direction.upper()}\n"
            f"reason={reason} | exit≈${exit_price:.2f}"
            f" | P/L=${pnl:+.2f} | R={r_multiple:.2f}"
        )
        _send_webhook(_discord(text))
    except Exception as exc:
        log.warning("monitor: notify_trade_exit failed (non-critical): %s", exc)


def notify_daily_loss_limit(
    bot: str, realized_today: float, limit: float, equity: float
) -> None:
    """Notify when the daily loss kill-switch fires."""
    if not _flag("NOTIFY_ON_ERRORS", True):
        return
    try:
        text = (
            f"🚨 **{bot.upper()} DAILY LOSS LIMIT HIT**\n"
            f"realized_today=${realized_today:+.2f}"
            f" | limit=-${limit:.2f} | equity=${equity:.2f}"
        )
        _send_webhook(_discord(text))
    except Exception as exc:
        log.warning("monitor: notify_daily_loss_limit failed (non-critical): %s", exc)


def notify_error(bot: str, context: str, error: str) -> None:
    """Notify on an unexpected error."""
    if not _flag("NOTIFY_ON_ERRORS", True):
        return
    try:
        text = f"⚠️ **{bot.upper()} ERROR** | {context}\n`{error[:500]}`"
        _send_webhook(_discord(text))
    except Exception as exc:
        log.warning("monitor: notify_error failed (non-critical): %s", exc)


def notify_run_complete(
    bot: str, equity: float, open_positions: int, realized_today: float
) -> None:
    """Notify with a run-complete summary (opt-in, disabled by default)."""
    if not _flag("NOTIFY_ON_SUMMARY", False):
        return
    try:
        pnl_str = f"${realized_today:+.2f}"
        text = (
            f"📊 **{bot.upper()} RUN COMPLETE**\n"
            f"equity=${equity:.2f} | open={open_positions}"
            f" | realized_today={pnl_str}"
        )
        _send_webhook(_discord(text))
    except Exception as exc:
        log.warning("monitor: notify_run_complete failed (non-critical): %s", exc)
