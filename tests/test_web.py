"""Web capabilities against injected responses — no network."""

from __future__ import annotations

import requests

from logpose.web import fetch_readable_text, search_duckduckgo


class _FakeStream:
    encoding = "utf-8"
    status_code = 200

    def __init__(self, body: bytes) -> None:
        self._body = body
        self.text = body.decode("utf-8", errors="replace")

    def raise_for_status(self) -> None:
        pass

    def iter_content(self, chunk_size: int):
        for offset in range(0, len(self._body), chunk_size):
            yield self._body[offset : offset + chunk_size]

    def close(self) -> None:
        pass


def test_fetch_extracts_readable_text_and_strips_markup():
    html = b"<html><script>evil()</script><style>p{}</style><p>Hello World</p></html>"
    out = fetch_readable_text("http://example.test", get=lambda *a, **k: _FakeStream(html))
    assert "Hello World" in out
    assert "evil" not in out


def test_fetch_labels_oversize_download_as_incomplete():
    body = b"<p>" + b"word " * 50000 + b"</p>"
    out = fetch_readable_text(
        "http://example.test", get=lambda *a, **k: _FakeStream(body), max_bytes=100
    )
    assert "Partial page" in out


def test_fetch_reports_blocked_sites():
    class Blocked(_FakeStream):
        status_code = 403

        def raise_for_status(self):
            raise requests.HTTPError(response=self)

    out = fetch_readable_text("http://example.test", get=lambda *a, **k: Blocked(b""))
    assert "blocked the request" in out


def test_search_formats_results():
    html = (
        '<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg='
        "https%3A%2F%2Fdocs.python.org%2F3%2F\">Python Docs</a>"
        '<a class="result__snippet">Reference pages</a></div>'
    )
    out = search_duckduckgo("python docs", post=lambda *a, **k: _FakeStream(html.encode()))
    assert "Python Docs" in out
    assert "https://docs.python.org/3/" in out
    assert "Reference pages" in out


def test_search_reports_challenge_without_fake_empty():
    html = '<form id="challenge-form"></form>'
    out = search_duckduckgo("q", post=lambda *a, **k: _FakeStream(html.encode()))
    assert "human verification" in out
