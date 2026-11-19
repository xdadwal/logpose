"""Read-only workspace tools: paged file reads, listings, and content search.

Pure stdlib. Path safety is centralized in :func:`resolve_path` — every path
is resolved against the configured root and escapes (``..``, absolute paths
outside it, or symlinks pointing out) become clear error strings rather than
file access. The factories return ``ToolDef`` objects; the functions behind
them are importable for direct, tool-free use.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterator
from pathlib import Path

from logpose.tools import ToolDef, tool

_SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules"}
_MAX_LINE = 200
_MAX_PAGE_CHARS = 8_000
_TARGETS = ("content", "names", "both")


class PathOutsideWorkspace(ValueError):
    """A requested path resolved outside the configured root."""


def resolve_path(path: str, base_path: Path) -> Path:
    """Resolve ``path`` against ``base_path``, rejecting any escape.

    Fully resolves both sides (``..`` collapsed, symlinks followed), then
    requires the result to be the root itself or contained under it. An
    absolute ``path`` overrides the root and is rejected unless it already
    lives inside.
    """
    candidate = (base_path / path).resolve()
    if candidate != base_path and base_path not in candidate.parents:
        raise PathOutsideWorkspace(f"{path!r} resolves outside the workspace ({base_path}).")
    return candidate


def read_file_page(
    path: str,
    *,
    base_path: Path,
    start_line: int = 1,
    max_lines: int | None = None,
    cursor: int = 0,
) -> str:
    """Read a page of a text file, bounded by characters (and optionally lines)."""
    bad_lines = max_lines is not None and max_lines < 1
    if start_line < 1 or bad_lines or cursor < 0 or (cursor and start_line != 1):
        return (
            "Invalid read range: start_line >= 1, max_lines >= 1 (or omitted), "
            "cursor >= 0; use start_line=1 with a cursor."
        )
    try:
        target = resolve_path(path, base_path)
    except PathOutsideWorkspace as exc:
        return f"Refused: {exc}"
    if not target.is_file():
        return f"No file at {path!r}."
    with target.open(encoding="utf-8") as stream:
        offset = 0
        while offset < cursor:
            skipped = stream.read(min(8192, cursor - offset))
            if not skipped:
                return "Invalid cursor: beyond the current file contents."
            offset += len(skipped)
        line = 1
        while line < start_line:
            skipped = stream.readline(8192)
            if not skipped:
                return "(no text in the requested range)"
            offset += len(skipped)
            line += skipped.endswith("\n")
        chunks: list[str] = []
        length = 0
        read = 0
        while max_lines is None or read < max_lines:
            read += 1
            chunk = stream.readline(_MAX_PAGE_CHARS - length)
            chunks.append(chunk)
            length += len(chunk)
            if not chunk or length == _MAX_PAGE_CHARS:
                break
        more = bool(stream.read(1))
    content = "".join(chunks)
    if more:
        content += (
            f"\n[More text; continue read_file(path={json.dumps(path)}, "
            f"cursor={offset + length}"
            + (f", max_lines={max_lines}" if max_lines is not None else "")
            + ")]"
        )
    return content


def list_directory(
    path: str = ".",
    *,
    base_path: Path,
    after: str = "",
    max_entries: int | None = None,
) -> str:
    """List directory entries (directories suffixed with '/'), paginated.

    Pages end at ``max_entries`` entries when given, and always at the character
    budget; a continuation names the last entry shown.
    """
    if max_entries is not None and max_entries < 1:
        return "Invalid page size: max_entries >= 1 (or omitted)."
    try:
        target = resolve_path(path, base_path)
    except PathOutsideWorkspace as exc:
        return f"Refused: {exc}"
    if not target.is_dir():
        return f"No directory at {path!r}."
    names = sorted(
        name
        for name in (p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
        if name > after
    )
    shown: list[str] = []
    length = 0
    for name in names:
        if max_entries is not None and len(shown) == max_entries:
            break
        if shown and length + len(name) + 1 > _MAX_PAGE_CHARS:
            break
        shown.append(name)
        length += len(name) + 1
    content = "\n".join(shown) if shown else "(empty)"
    if len(shown) < len(names):
        content += (
            f"\n[More entries; continue list_dir(path={json.dumps(path)}, "
            f"after={json.dumps(shown[-1])}"
            + (f", max_entries={max_entries}" if max_entries is not None else "")
            + ")]"
        )
    return content


def grep_workspace(
    pattern: str,
    *,
    base_path: Path,
    path: str = ".",
    ignore_case: bool = False,
    target: str = "content",
    output_char_cap: int | None = None,
    on_output: Callable[[str], None] | None = None,
    skip_dirs: set[str] | None = None,
) -> str:
    """Search file contents/names under the root. See ``make_grep_tool``."""
    if target not in _TARGETS:
        return f"Invalid target {target!r}; use 'content', 'names', or 'both'."
    try:
        root = resolve_path(path, base_path)
    except PathOutsideWorkspace as exc:
        return f"Refused: {exc}"
    if not root.exists():
        return f"No file or directory at {path!r}."
    try:
        content_re = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        name_re = _name_regex(pattern)
    except re.error as exc:
        return f"Invalid pattern {pattern!r}: {exc}"

    files = [root] if root.is_file() else list(_walk(root, skip_dirs))
    results: list[str] = []
    if target in ("names", "both"):
        results.extend(_match_names(files, name_re, base_path))
    if target in ("content", "both"):
        results.extend(_match_content(files, content_re, base_path))
    if not results:
        return (
            f"No matches for {pattern!r}. If you expected one, try ignore_case=true, "
            'a shorter pattern, or target="names" to find files by name — or '
            "list_dir to see what's here."
        )
    output = "\n".join(results)
    if on_output is not None:
        on_output(output)
    if output_char_cap is not None and len(output) > output_char_cap:
        output = output[:output_char_cap] + "\n...[truncated]"
    return output


def make_read_file_tool(
    base_path: str | Callable[[], str], name: str = "read_file", description: str | None = None
) -> ToolDef:
    """Build a paged file-read ToolDef bound to ``base_path``."""

    def root() -> Path:
        return Path(base_path() if callable(base_path) else base_path).resolve()

    summary = description or (
        "Read a bounded page of a text file before editing it. Continue with "
        "the returned cursor when a page is incomplete."
    )

    @tool
    def read_file(
        path: str, start_line: int = 1, max_lines: int | None = None, cursor: int = 0
    ) -> str:
        """Read a page of a text file."""
        return read_file_page(
            path, base_path=root(), start_line=start_line, max_lines=max_lines, cursor=cursor
        )

    return _rebind(read_file, name, summary)


def make_list_dir_tool(
    base_path: str | Callable[[], str], name: str = "list_dir", description: str | None = None
) -> ToolDef:
    """Build a paginated directory-listing ToolDef bound to ``base_path``."""

    def root() -> Path:
        return Path(base_path() if callable(base_path) else base_path).resolve()

    summary = description or (
        "List entries in a directory (defaults to the workspace root). "
        "Directories are suffixed with '/'."
    )

    @tool
    def list_dir(path: str = ".", after: str = "", max_entries: int | None = None) -> str:
        """List directory entries, paginated."""
        return list_directory(path, base_path=root(), after=after, max_entries=max_entries)

    return _rebind(list_dir, name, summary)


def make_grep_tool(
    base_path: str | Callable[[], str],
    *,
    name: str = "grep",
    description: str | None = None,
    output_char_cap: int | None = 8_000,
    on_output: Callable[[str], None] | None = None,
    skip_dirs: set[str] | None = None,
) -> ToolDef:
    """Build a content/name search ToolDef bound to ``base_path``."""

    def root() -> Path:
        return Path(base_path() if callable(base_path) else base_path).resolve()

    summary = description or (
        'Search files for a pattern. target="content" searches file contents '
        '(like grep); target="names" finds files by path (case-insensitive, '
        'partial match); "both" does both.'
    )

    @tool
    def grep(
        pattern: str, path: str = ".", ignore_case: bool = False, target: str = "content"
    ) -> str:
        """Search files for a pattern."""
        return grep_workspace(
            pattern,
            base_path=root(),
            path=path,
            ignore_case=ignore_case,
            target=target,
            output_char_cap=output_char_cap,
            on_output=on_output,
            skip_dirs=skip_dirs,
        )

    return _rebind(grep, name, summary)


def _rebind(decorated: ToolDef, name: str, description: str) -> ToolDef:
    """Re-emit a decorated tool with the caller's name and summary."""
    return ToolDef(
        name=name,
        description=description,
        input_schema=decorated.input_schema,
        handler=decorated.handler,
        is_async=decorated.is_async,
        args_model=decorated.args_model,
        positional_only=decorated.positional_only,
    )


def _name_regex(pattern: str) -> re.Pattern:
    """Forgiving file-name matcher: case-insensitive; multi-word queries match
    a path containing ANY of the words."""
    tokens = pattern.split()
    body = "|".join(re.escape(token) for token in tokens) if len(tokens) > 1 else pattern
    return re.compile(body, re.IGNORECASE)


def _match_names(files: list[Path], regex: re.Pattern, base: Path) -> list[str]:
    return [str(f.relative_to(base)) for f in files if regex.search(str(f.relative_to(base)))]


def _match_content(files: list[Path], regex: re.Pattern, base: Path) -> list[str]:
    out = []
    for file in files:
        try:
            text = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                line = line.rstrip()
                if len(line) > _MAX_LINE:
                    line = line[:_MAX_LINE] + "..."
                out.append(f"{file.relative_to(base)}:{lineno}: {line}")
    return out


def _walk(root: Path, skip_dirs: set[str] | None) -> Iterator[Path]:

    prune = _SKIP_DIRS if skip_dirs is None else skip_dirs
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in prune)
        for name in sorted(filenames):
            yield Path(dirpath) / name
