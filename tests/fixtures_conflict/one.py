"""Defines a tool named ``search``."""

from logpose import tool


@tool
def search(q: str) -> str:
    """Search the web.

    Args:
        q: The query.
    """
    return f"web:{q}"
