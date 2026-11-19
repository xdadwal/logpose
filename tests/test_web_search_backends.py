"""The backend-bound search tool factory."""

from __future__ import annotations

from logpose.web import SearchFailure, make_search_tool


def test_make_search_tool_formats_success_and_caps():
    search = make_search_tool(
        backend=lambda q, n: ["A — https://a", "B — https://b"],
        output_char_cap=10,
    )
    out = search("q")
    assert out.startswith("Web results for 'q':")
    assert out.endswith("...[truncated]")


def test_make_search_tool_preserves_safe_failures_and_hides_payloads():
    def safe_failure(q, n):
        raise SearchFailure("blocked")

    def unsafe_failure(q, n):
        raise RuntimeError("payload with credentials")

    assert make_search_tool(backend=safe_failure)("q") == "blocked"
    out = make_search_tool(backend=unsafe_failure)("q")
    assert "credentials" not in out
    assert "RuntimeError" in out


def test_make_search_tool_empty_and_blank_query():
    search = make_search_tool(backend=lambda q, n: [])
    assert "No web results" in search("q")
    assert "non-empty" in search("   ")
