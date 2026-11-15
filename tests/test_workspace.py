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
