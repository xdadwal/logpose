"""Z.ai Coding Plan MCP search and the backend-bound search tool factory."""

from __future__ import annotations

import json
import json as json_mod

import pytest

from logpose.web import SearchFailure, make_search_tool, search_zai


class _SSEClient:
    """Minimal MCP session: replies to initialize/tools-call with plain JSON."""

    def __init__(self, tool_result: dict) -> None:
        self.tool_result = tool_result
        self._rpc_responses: list[object] | None = None
        self.closed = False

    def post(self, url, payload=None, json=None, headers=None, **kwargs):  # noqa: A002
        if json and json.get("method") == "initialize":
            body = {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}}
        else:
            body = {"jsonrpc": "2.0", "id": 2, "result": self.tool_result}

        class _Resp:
            status_code = 200
            headers = {"Content-Type": "application/json"}
            text = json_mod.dumps(body)

            def close(self):
                pass

        return _Resp()

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _tool_result(items: list[dict]) -> dict:
    wrapped = json.dumps(json.dumps(items))
    return {"content": [{"type": "text", "text": wrapped}]}


def test_search_zai_formats_entries_and_registers_terminator():
    client = _SSEClient(_tool_result([
        {"title": "Python", "link": "https://python.org/", "content": "Home"},
    ]))
    entries = search_zai("q", api_key="key", session_factory=lambda: client)
    assert entries == ["Python — https://python.org/\n  Home"]
    assert client.closed is True


def test_search_zai_rejects_empty_key():
    with pytest.raises(SearchFailure):
        search_zai("q", api_key="  ", session_factory=lambda: _SSEClient(_tool_result([])))


def test_search_zai_maps_is_error_to_safe_failure():
    client = _SSEClient({"isError": True, "content": [{"type": "text", "text": "secret"}]})
    with pytest.raises(SearchFailure) as excinfo:
        search_zai("q", api_key="key", session_factory=lambda: client)
    assert "secret" not in str(excinfo.value)


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


def test_zai_rpc_parses_sse_matching_response_after_notification():
    from logpose.web import _zai_rpc

    response = _RPC(sse=True)
    response.text = (': keepalive\n\ndata: {"jsonrpc":"2.0","method":"notice"}\n\n'
                     'event: message\ndata: {"jsonrpc":"2.0","id":2,\n'
                     'data: "result":{"content":[]}}\n\n')

    class Client:
        def post(self, *args, **kwargs):
            return response

    assert _zai_rpc(Client(), {}, "https://zai.test", "tools/call", {}, 2) == {"content": []}


def test_zai_rpc_error_does_not_echo_server_secret():
    from logpose.web import _zai_rpc

    response = _RPC({"jsonrpc": "2.0", "id": 2,
                     "error": {"code": -32000, "message": "test-secret"}})

    class Client:
        def post(self, *args, **kwargs):
            return response

    with pytest.raises(SearchFailure) as raised:
        _zai_rpc(Client(), {}, "https://zai.test", "tools/call", {}, 2)
    assert "test-secret" not in str(raised.value)


def test_zai_rpc_sse_returns_matching_result_without_waiting_for_eof():
    from logpose.web import _zai_rpc

    class OpenStream(_RPC):
        def iter_lines(self, **kwargs):
            yield 'data: {"jsonrpc":"2.0","id":2,"result":{"content":[]}}'
            yield ''
            pytest.fail("waited for EOF after a complete matching event")

    response = OpenStream(sse=True)

    class Client:
        def post(self, *args, **kwargs):
            assert kwargs["stream"] is True
            return response

    assert _zai_rpc(Client(), {}, "https://zai.test", "tools/call", {}, 2) == {"content": []}
    assert response.closed


class _RPC:
    def __init__(self, payload=None, status_code=200, *, sse=False, headers=None):
        self.status_code = status_code
        self.headers = headers or {}
        self.closed = False
        self.text = "" if payload is None else json.dumps(payload)
        if sse:
            self.headers = {"Content-Type": "text/event-stream", **(headers or {})}
            self.encoding = None

    def raise_for_status(self):
        pass

    def iter_lines(self, **kwargs):
        yield from self.text.splitlines()

    def close(self):
        self.closed = True
