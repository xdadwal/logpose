"""Defines a *different* tool that is also named ``search``."""

from logpose import tool


@tool(name="search")
def vector_search(q: str) -> str:
    """Search a vector store.

    Args:
        q: The query.
    """
    return f"vector:{q}"
