"""Two tools, to prove a single module yields all of them."""

from logpose import tool


@tool
def alpha(x: int) -> str:
    """Alpha tool.

    Args:
        x: A number.
    """
    return f"alpha:{x}"


@tool
def beta(y: str) -> str:
    """Beta tool.

    Args:
        y: Some text.
    """
    return f"beta:{y}"
