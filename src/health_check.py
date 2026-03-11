#!/usr/bin/env python3
"""Standalone bot liveness checker.

Reads logs/heartbeat_*.json and reports the age of each bot's last run.
Exits with code 1 if any bot is overdue so this can be used in cron
alerts or monitoring pipelines.

Usage:
    python health_check.py                  # default 60-min threshold
    python health_check.py --max-age 30     # warn if older than 30 min
    python health_check.py --bot stocks     # check only the stock bot
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
_LOG_DIR = _SCRIPT_DIR / "logs"

# Default: how many minutes without a heartbeat counts as "overdue".
# Stocks run on market hours; crypto is 24/7. A single threshold is
# intentionally conservative — adjust with --max-age for your schedule.
DEFAULT_MAX_AGE_MINUTES = 60


def _read_heartbeat(bot_name: str) -> dict | None:
    path = _LOG_DIR / f"heartbeat_{bot_name}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _age_minutes(timestamp_utc: str) -> float:
    """Return minutes since the given ISO-8601 UTC timestamp."""
    try:
        ts = datetime.fromisoformat(timestamp_utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - ts
        return delta.total_seconds() / 60
    except Exception:
        return float("inf")


def check_bots(bot_names: list[str], max_age_minutes: float) -> bool:
    """Check heartbeat files for the given bots.

    Returns True if all bots are healthy, False if any are overdue.
    """
    all_healthy = True

    for bot in bot_names:
        hb = _read_heartbeat(bot)
        if hb is None:
            print(f"[MISSING ] {bot}: no heartbeat file found at logs/heartbeat_{bot}.json")
            all_healthy = False
            continue

        age = _age_minutes(hb.get("timestamp_utc", ""))
        status = hb.get("status", "unknown")
        equity = hb.get("equity", "N/A")
        open_pos = hb.get("open_positions", "N/A")
        ts = hb.get("timestamp_utc", "N/A")

        if age > max_age_minutes:
            flag = "OVERDUE "
            all_healthy = False
        else:
            flag = "OK      "

        print(
            f"[{flag}] {bot}: last_run={ts} | age={age:.1f}min"
            f" | status={status} | equity=${equity} | open={open_pos}"
        )

    return all_healthy


def main() -> None:
    parser = argparse.ArgumentParser(description="Trading bot liveness checker")
    parser.add_argument(
        "--max-age",
        type=float,
        default=DEFAULT_MAX_AGE_MINUTES,
        metavar="MINUTES",
        help=f"Alert if a bot's last heartbeat is older than this many minutes (default: {DEFAULT_MAX_AGE_MINUTES})",
    )
    parser.add_argument(
        "--bot",
        nargs="+",
        default=["stocks", "crypto"],
        metavar="BOT",
        help="Which bots to check (default: stocks crypto)",
    )
    args = parser.parse_args()

    print(f"=== Bot Health Check | threshold={args.max_age:.0f}min ===")
    healthy = check_bots(args.bot, args.max_age)
    print("=" * 45)

    sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
