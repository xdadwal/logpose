"""Built-in ``current_time`` tool: the current date and time in an IANA zone."""

from __future__ import annotations

import functools
from datetime import datetime, tzinfo
from datetime import timezone as _dt_timezone
from zoneinfo import ZoneInfo, available_timezones

from logpose.tools import tool

__all__ = ["current_time"]

_FORMAT = "%A, %d %B %Y, %H:%M %Z"
_UTC_NAMES = {"utc", "etc/utc"}


def _now(tz: tzinfo) -> datetime:
    return datetime.now(tz)


@functools.lru_cache(maxsize=1)
def _tzdata_available() -> bool:
    return bool(available_timezones())


def _format(tz: tzinfo, label: str) -> str:
    return f"{_now(tz).strftime(_FORMAT)} ({label})"


@tool
def current_time(timezone: str = "UTC") -> str:
    """Get the current date and time in a time zone.

    Use it whenever you need today's date, the current time, or to work out
    relative dates ("tomorrow", "next Friday").

    Args:
        timezone: IANA time zone name, e.g. "Asia/Kolkata" or "America/New_York".
            Defaults to UTC.
    """
    if timezone.strip().lower() in _UTC_NAMES:
        return _format(_dt_timezone.utc, timezone)
    try:
        zone = ZoneInfo(timezone)
    except (ValueError, OSError, KeyError):
        # ZoneInfoNotFoundError is a KeyError; odd names raise ValueError/OSError.
        if not _tzdata_available():
            return (
                f"{_now(_dt_timezone.utc).strftime(_FORMAT)} "
                "(time zone data is not installed on this server, so this is UTC)"
            )
        return (
            f"Unknown time zone {timezone[:64]!r}. Use an IANA name such as "
            "'Asia/Kolkata' or 'America/New_York'."
        )
    return _format(zone, timezone)
