"""Built-in ``current_time`` tool: the current date and time in an IANA zone."""

from __future__ import annotations

from datetime import datetime, tzinfo
from zoneinfo import ZoneInfo

from logpose.tools import tool

__all__ = ["current_time"]

_FORMAT = "%A, %d %B %Y, %H:%M %Z"


def _now(tz: tzinfo) -> datetime:
    return datetime.now(tz)


@tool
def current_time(timezone: str = "UTC") -> str:
    """Get the current date and time in a time zone.

    Args:
        timezone: IANA time zone name, e.g. "Asia/Kolkata" or "America/New_York".
            Defaults to UTC.
    """
    try:
        zone = ZoneInfo(timezone)
    except (ValueError, OSError, KeyError):
        # ZoneInfoNotFoundError is a KeyError; odd names raise ValueError/OSError.
        return (
            f"Unknown time zone {timezone!r}. Use an IANA name such as "
            "'Asia/Kolkata' or 'America/New_York'."
        )
    return f"{_now(zone).strftime(_FORMAT)} ({timezone})"
