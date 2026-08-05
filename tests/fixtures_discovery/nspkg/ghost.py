"""Lives in a directory with no __init__.py.

pkgutil.iter_modules does not report such a directory as a package, so recursion
does not descend into it. This file pins that documented limitation.
"""

from logpose import tool


@tool
def ghost_tool() -> str:
    """Must never be discovered by a recursive walk."""
    return "boo"
