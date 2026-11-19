"""Read-only workspace tools: pagination, safety, and search."""

from __future__ import annotations

import pytest

from logpose.workspace import (
    PathOutsideWorkspace,
    grep_workspace,
    list_directory,
    make_grep_tool,
    make_list_dir_tool,
    make_read_file_tool,
    read_file_page,
    resolve_path,
)


@pytest.fixture
def root(tmp_path):
    (tmp_path / "hello.txt").write_text("line one\nline two\nline three\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "deep.txt").write_text("deep content\n")
    return tmp_path


def test_resolve_path_allows_inside_and_rejects_escape(root):
    assert resolve_path("hello.txt", root) == root / "hello.txt"
    with pytest.raises(PathOutsideWorkspace):
        resolve_path("../outside.txt", root)
    with pytest.raises(PathOutsideWorkspace):
        resolve_path(str(root.parent / "elsewhere"), root)


def test_read_file_page_paginates_by_cursor(root):
    first = read_file_page("hello.txt", base_path=root, max_lines=2)
    assert "line two" in first
    assert "[More text" in first
    second = read_file_page("hello.txt", base_path=root, cursor=len("line one\nline two\n"))
    assert "line three" in second
    assert "[More" not in second


def test_list_directory_paginates_and_marks_dirs(root):
    out = list_directory(base_path=root, max_entries=1)
    assert "hello.txt" in out
    assert "[More entries" in out
    assert "sub/" in list_directory(base_path=root)


def test_grep_finds_content_and_names(root):
    out = grep_workspace("line two", base_path=root)
    assert "hello.txt:2: line two" in out
    names = grep_workspace("deep", base_path=root, target="names")
    assert "sub/deep.txt" in names


def test_factories_bind_the_root(root):
    read = make_read_file_tool(str(root))
    listing = make_list_dir_tool(str(root))
    search = make_grep_tool(str(root))
    assert "line one" in read("hello.txt")
    assert "hello.txt" in listing()
    assert "line three" in search("three")


# ---------------------------------------------------------------------------
# web-app safe limits
# ---------------------------------------------------------------------------

import re  # noqa: E402


def _follow_read(root, name):
    """Read a file to the end by following cursor continuations."""
    out, cursor = [], 0
    while True:
        page = read_file_page(name, base_path=root, cursor=cursor)
        match = re.search(r"\n\[More text; continue read_file\(.*cursor=(\d+).*\)\]$", page)
        if not match:
            out.append(page)
            return "".join(out), len(out)
        out.append(page[: match.start()])
        cursor = int(match.group(1))


def test_read_file_has_no_line_cap(tmp_path):
    (tmp_path / "f.txt").write_text("".join(f"{i}\n" for i in range(300)))
    page = read_file_page("f.txt", base_path=tmp_path, max_lines=500)
    assert page == "".join(f"{i}\n" for i in range(300))


def test_read_file_page_is_char_bounded(tmp_path):
    text = "".join(f"line {i:05d} " + "x" * 30 + "\n" for i in range(1000))
    assert len(text) > 40_000
    (tmp_path / "big.txt").write_text(text)
    first = read_file_page("big.txt", base_path=tmp_path)
    assert len(first) <= 8_000 + 200
    assert "cursor=" in first
    rebuilt, pages = _follow_read(tmp_path, "big.txt")
    assert rebuilt == text
    assert pages >= 3


def test_list_dir_has_no_entry_cap(tmp_path):
    for i in range(250):
        (tmp_path / f"f{i:03d}").write_text("")
    out = list_directory(base_path=tmp_path)
    assert len(out.splitlines()) == 250


def test_list_dir_char_budget_continues_without_gaps(tmp_path):
    names = [f"entry_{i:05d}_padding_xx" for i in range(3000)]
    for name in names:
        (tmp_path / name).write_text("")
    seen, after = [], ""
    while True:
        out = list_directory(base_path=tmp_path, after=after)
        assert len(out) <= 8_000 + 200
        match = re.search(r'\n\[More entries; continue list_dir\(.*after="([^"]+)".*\)\]$', out)
        body = out[: match.start()] if match else out
        seen.extend(body.splitlines())
        if not match:
            break
        after = match.group(1)
    assert seen == sorted(names)


def test_grep_output_is_capped_by_default(tmp_path):
    for i in range(200):
        (tmp_path / f"f{i}.txt").write_text("needle " * 50 + "\n")
    grep = make_grep_tool(str(tmp_path))
    out = grep("needle")
    assert len(out) <= 8_000 + 200
