"""Underscore-prefixed: discovery must skip this module entirely."""

from logpose import tool


@tool
def skipped_tool() -> str:
    """Must never be discovered."""
    return "nope"
