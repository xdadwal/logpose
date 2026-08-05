"""A tool one level deeper than the package root."""

from logpose import tool


@tool
def delta() -> str:
    """Delta tool."""
    return "delta"
