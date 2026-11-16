"""Web capabilities: readable page fetch and DuckDuckGo search.

Capability logic only — the model-facing ToolDef, provider configuration
(e.g. a paid search backend), guarding, and artifact capture stay with the
embedding application. Every function accepts the requests callable as a
parameter so tests can inject responses without network access, and both
register a session close with the current tool deadline (see
:mod:`logpose.cancellation`) so a timed-out request is actually released.

Requires the ``web`` extra: ``pip install logpose[web]``.
"""

from __future__ import annotations

from collections.abc import Callable

from logpose.tools import ToolDef, tool

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError as _exc:  # pragma: no cover - environment shape, not logic
    raise ImportError(
        "logpose.web requires the 'web' extra: pip install logpose[web]"
    ) from _exc

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
_SEARCH_URL = "https://html.duckduckgo.com/html/"
_REQUEST_TIMEOUT = 15
_SEARCH_TIMEOUT = 15
_ZAI_URL = "https://api.z.ai/api/mcp/web_search_prime/mcp"


class SearchFailure(ValueError):
    """A provider failure whose message is safe to show to the model.

    Raise (or receive) this instead of leaking raw HTTP/RPC payloads — response
    bodies may contain credentials. Callers that catch plain provider
    exceptions should surface only the exception type, never its text.
    """


def fetch_readable_text(
    url: str,
    *,
    get: Callable[..., requests.Response] | None = None,
    timeout: float = _REQUEST_TIMEOUT,
    max_bytes: int = 2 * 1024 * 1024,
    output_char_cap: int | None = None,
    on_output: Callable[[str, bool], None] | None = None,
) -> str:
    """Fetch ``url`` and return its readable text, scripts/styles stripped.

    Downloads stop at ``max_bytes`` decompressed bytes; pages are labelled
    when the download or decoding was incomplete. ``on_output(content,
    complete)`` observes the full (uncapped) extracted text for capture.
    """
    session = None
    if get is None:
        from logpose.cancellation import current_tool_deadline

        session = requests.Session()
        handle = current_tool_deadline()
        if handle is not None:
            handle.register_terminator(session.close)
        getter = session.get
    else:
        getter = get
    response = None
    body = bytearray()
    complete = True
    decode_note = ""
    try:
        response = getter(url, headers=_HEADERS, timeout=timeout, allow_redirects=True, stream=True)
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=65536):
            remaining = max_bytes - len(body)
            body.extend(chunk[:remaining])
            if len(chunk) > remaining:
                complete = False
                break
        raw = bytes(body)
        from requests.compat import chardet

        detected = chardet.detect(raw).get("encoding") if chardet else None
        encoding = response.encoding or detected or "utf-8"
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            text = raw.decode(encoding, errors="replace")
            complete = False
            decode_note = "[Text decoding was lossy; source capture is incomplete]\n"
        except LookupError:
            text = raw.decode("utf-8", errors="replace")
            complete = False
            decode_note = (
                "[Declared text encoding was invalid or decoding was lossy; "
                "source capture is incomplete]\n"
            )
    except requests.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else "?"
        if code in (401, 403):
            return (
                f"{url} blocked the request (HTTP {code}) — the site refuses "
                "automated access (bot protection). Try a different source."
            )
        return f"Fetching {url} failed: HTTP {code}."
    except requests.RequestException as exc:
        return f"Could not reach {url}: {exc}"
    finally:
        if response is not None:
            response.close()
        if session is not None:
            session.close()

    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    lines = [line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip()]
    content = "\n".join(lines)
    if not content:
        tail = " Download was limited or decoding was incomplete." if not complete else ""
        return f"{url} returned no readable text.{tail}"
    limit_note = (
        f"[Partial page: download limited to {max_bytes} decompressed response bytes]\n"
        if len(body) == max_bytes and not complete
        else ""
    ) + decode_note
    full = f"Content from {url}:\n\n{limit_note}{content}"
    if on_output is not None:
        on_output(full, complete)
    if output_char_cap is not None and len(full) > output_char_cap:
        full = full[:output_char_cap] + "\n...[truncated]"
    return full


def search_duckduckgo(
    query: str,
    max_results: int = 5,
    *,
    post: Callable[..., requests.Response] | None = None,
    timeout: float = _SEARCH_TIMEOUT,
) -> str:
    """Search DuckDuckGo's HTML endpoint and return formatted results.

    String convenience wrapper over :func:`duckduckgo_search_entries`: failures
    become the failure message, an empty result set becomes a retry-hint line.
    """
    try:
        entries = duckduckgo_search_entries(query, max_results, post=post, timeout=timeout)
    except SearchFailure as exc:
        return str(exc)
    if not entries:
        return f"No web results for {query!r}. Try different or broader search terms."
    return f"Web results for {query!r}:\n\n" + "\n\n".join(entries)


def duckduckgo_search_entries(
    query: str,
    max_results: int = 5,
    *,
    post: Callable[..., requests.Response] | None = None,
    timeout: float = _SEARCH_TIMEOUT,
) -> list[str]:
    """Search DuckDuckGo's HTML endpoint and return `title — url` entries.

    Raises:
        SearchFailure: On blocked/challenge pages, unexpected HTTP statuses,
            or unrecognized responses — with a model-safe message that
            distinguishes failure from an empty result set.
    """
    if post is None:
        session = requests.Session()
        from logpose.cancellation import current_tool_deadline

        handle = current_tool_deadline()
        if handle is not None:
            handle.register_terminator(session.close)
        poster = session.post
    else:
        poster = post
    try:
        response = poster(
            _SEARCH_URL, data={"q": query}, headers=_HEADERS, timeout=timeout, allow_redirects=True
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SearchFailure(f"Web search for {query!r} failed: {exc}") from exc
    soup = BeautifulSoup(response.text, "html.parser")
    if soup.select_one("#challenge-form, #anomaly-form, .anomaly-modal"):
        raise SearchFailure(
            "Web search for "
            f"{query!r} blocked by DuckDuckGo human verification "
            f"(HTTP {response.status_code}). This is not an empty search. "
            "Do not retry the same request; use a paid search backend or "
            "fetch a known source URL directly."
        )
    if response.status_code != 200:
        raise SearchFailure(
            f"Web search for {query!r} failed: unexpected "
            f"DuckDuckGo HTTP {response.status_code}."
        )
    results = parse_duckduckgo_results(response.text, max_results)
    if not results and not soup.select_one(".no-results, .result--no-result"):
        raise SearchFailure(
            f"Web search for {query!r} failed: unrecognized DuckDuckGo response. "
            "No result count could be established."
        )
    return results


def parse_duckduckgo_results(html: str, max_results: int) -> list[str]:
    """Pull `title — url` entries with snippets out of a DDG results page."""
    soup = BeautifulSoup(html, "html.parser")
    out: list[str] = []
    for result in soup.select(".result"):
        anchor = result.select_one(".result__a")
        snippet = result.select_one(".result__snippet")
        if anchor is None:
            continue
        title = anchor.get_text(strip=True)
        link = clean_ddg_url(str(anchor.get("href", "")))
        if not link:
            continue
        snippet_text = snippet.get_text(" ", strip=True) if snippet else ""
        entry = f"{title} — {link}"
        if snippet_text:
            entry += f"\n  {snippet_text}"
        out.append(entry)
        if len(out) >= max_results:
            break
    return out


def clean_ddg_url(href: str) -> str:
    """DDG wraps result links in a ``/l/?uddg=<real-url>`` redirect; unwrap it.

    Returns an empty string for anything that isn't a usable http(s) URL.
    """
    from urllib.parse import parse_qs, urlparse

    if not isinstance(href, str) or not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    try:
        parsed = urlparse(href)
        host = parsed.hostname or ""
        is_ddg = host == "duckduckgo.com" or host.endswith(".duckduckgo.com") or not host
        if is_ddg and parsed.path.startswith("/l/"):
            target = parse_qs(parsed.query).get("uddg", [])
            href = target[0] if target else ""
            parsed = urlparse(href)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return ""
        return href
    except ValueError:
        return ""


def _zai_rpc(
    client: requests.Session,
    headers: dict[str, str],
    base_url: str,
    method: str,
    params: dict | None = None,
    request_id: int | None = None,
) -> dict:
    """One bounded MCP HTTP exchange; notifications may return empty 202."""

    payload: dict = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        payload["params"] = params
    if request_id is not None:
        payload["id"] = request_id
    response = client.post(base_url, json=payload, headers=headers, timeout=30,
                           allow_redirects=False, stream=True)
    try:
        session_id = response.headers.get("Mcp-Session-Id")
        if session_id:
            headers["Mcp-Session-Id"] = session_id
        if request_id is None and response.status_code in (200, 202):
            return {}  # Notifications have no JSON-RPC response body.
        if response.status_code != 200:
            hints = {401: "check the API key", 403: "check plan access",
                     429: "plan quota or rate limit reached"}
            hint = hints.get(response.status_code, "service request was unsuccessful")
            raise SearchFailure(f"HTTP {response.status_code}: {hint}.")
        message = _zai_rpc_message(response, request_id)
    finally:
        # A matching result can arrive long before the SSE connection closes.
        response.close()
    if not isinstance(message, dict) or message.get("id") != request_id:
        raise SearchFailure("Unrecognized MCP response or request identifier.")
    if "error" in message:
        raise SearchFailure("MCP request rejected; check credentials and quota.")
    result = message.get("result")
    if not isinstance(result, dict):
        raise SearchFailure("MCP response is missing its result.")
    return result


def _zai_rpc_message(response: requests.Response, request_id: int | None) -> object:
    import json

    content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
    if content_type != "text/event-stream":
        try:
            return json.loads(response.text)
        except ValueError as exc:
            raise SearchFailure("Malformed MCP JSON response.") from exc
    response.encoding = "utf-8"  # SSE is UTF-8, including non-ASCII result snippets.
    data: list[str] = []
    received = 0
    stream = response.iter_lines(chunk_size=1, decode_unicode=True)
    while True:
        try:
            raw_line = next(stream)
        except StopIteration:
            break
        line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8", errors="replace")
        received += len(line.encode("utf-8"))
        if received > 2_000_000:
            raise SearchFailure("MCP response exceeded the search size limit.")
        if line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
        elif not line and data:
            try:
                candidate = json.loads("\n".join(data))
            except ValueError as exc:
                raise SearchFailure("Malformed MCP event response.") from exc
            data.clear()
            if isinstance(candidate, dict) and candidate.get("id") == request_id:
                return candidate
    raise SearchFailure("MCP stream ended without a matching result.")


def search_zai(
    query: str,
    max_results: int = 5,
    *,
    api_key: str,
    base_url: str = _ZAI_URL,
    session_factory: Callable[[], requests.Session] | None = None,
) -> list[str]:
    """Search via a Z.ai Coding Plan MCP endpoint; return `title — url` entries.

    Args:
        api_key: Bearer token for the endpoint. Required — logpose never reads
            environment variables itself; the embedding app owns credentials.
        base_url: MCP endpoint override.
        session_factory: Injectable session constructor for tests.

    Raises:
        SearchFailure: With a model-safe, credential-free message on any
            provider-level problem (auth, quota, malformed payloads).
    """
    import json

    if not api_key.strip():
        raise SearchFailure("A non-empty API key is required.")
    factory = session_factory or requests.Session
    headers = {"Authorization": f"Bearer {api_key.strip()}",
               "Accept": "application/json, text/event-stream"}
    with factory() as client:
        from logpose.cancellation import current_tool_deadline

        handle = current_tool_deadline()
        if handle is not None and hasattr(client, "close"):
            handle.register_terminator(client.close)
        initial = _zai_rpc(client, headers, base_url, "initialize", {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "logpose", "version": "0.1.0"},
        }, 1)
        version = initial.get("protocolVersion")
        if not isinstance(version, str) or not version:
            raise SearchFailure("MCP initialization did not negotiate a protocol version.")
        headers["MCP-Protocol-Version"] = version
        _zai_rpc(client, headers, base_url, "notifications/initialized")
        result = _zai_rpc(client, headers, base_url, "tools/call", {
            "name": "web_search_prime", "arguments": {"search_query": query, "location": "us"},
        }, 2)
    if result.get("isError"):
        raise SearchFailure("Search tool returned an error; check plan access and quota.")
    content = result.get("content")
    if not isinstance(content, list):
        raise SearchFailure("Search tool response has no content blocks.")
    items: list[dict] = []
    decoded = False
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        value = block.get("text")
        # The live service wraps its JSON result list in a JSON string.
        for _ in range(3):
            if not isinstance(value, str):
                break
            try:
                value = json.loads(value)
            except ValueError as exc:
                raise SearchFailure("Unrecognized search result format.") from exc
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            raise SearchFailure("Unrecognized search result list.")
        decoded = True
        items.extend(value)
    if not decoded:
        raise SearchFailure("Search tool returned no recognized results payload.")
    out = []
    for item in items:
        title, url = item.get("title"), item.get("link")
        if not isinstance(title, str) or not title.strip() or not isinstance(url, str):
            raise SearchFailure("Search result is missing a title or URL.")
        cleaned = clean_ddg_url(url)
        if not cleaned:
            raise SearchFailure("Search result has an invalid URL.")
        entry = f"{title.strip()} — {cleaned}"
        snippet = item.get("content")
        if isinstance(snippet, str) and snippet.strip():
            entry += "\n  " + snippet.strip()
        out.append(entry)
        if len(out) >= max_results:
            break
    return out


def make_search_tool(
    *,
    backend: Callable[[str, int], list[str]],
    name: str = "web_search",
    description: str | None = None,
    output_char_cap: int | Callable[[], int | None] | None = None,
    on_output: Callable[[str], None] | None = None,
) -> ToolDef:
    """Build a web-search ToolDef bound to a backend entry provider.

    Args:
        backend: ``(query, max_results) -> entries``; raise
            :class:`SearchFailure` with a model-safe message for provider
            problems, return ``[]`` for a genuine empty result set. Any other
            exception surfaces as its type name only — response payloads may
            contain credentials and are never shown.
        name: Tool name (default ``web_search``).
        description: Override the model-facing summary.
        output_char_cap: Longest result body returned to the model; a callable
            is resolved per call, so embedding apps can track live config.
        on_output: Observes the full formatted result body (e.g. artifact
            capture) before any truncation.
    """
    summary = description or (
        "Search the web and return the top results as `title — url` with a "
        "snippet. Reports provider blocking/errors distinctly from an empty "
        "search."
    )

    @tool
    def web_search(query: str, max_results: int = 5) -> str:
        """Search the web; returns `title — url` entries with snippets."""
        max_results = max(1, min(max_results, 25))
        query = query.strip()
        if not query:
            return "Web search needs a non-empty query."
        try:
            results = backend(query, max_results)
        except SearchFailure as exc:
            return str(exc)
        except Exception as exc:  # noqa: BLE001 - payload text may hold credentials
            return f"Web search for {query!r} failed: {type(exc).__name__}."
        if not results:
            return f"No web results for {query!r}. Try different or broader search terms."
        body = "\n\n".join(results)
        full = f"Web results for {query!r}:\n\n{body}"
        if on_output is not None:
            on_output(full)
        cap = output_char_cap() if callable(output_char_cap) else output_char_cap
        if cap is not None and len(body) > cap:
            body = body[:cap] + "\n...[truncated]"
            full = f"Web results for {query!r}:\n\n{body}"
        return full

    return ToolDef(
        name=name,
        description=summary,
        input_schema=web_search.input_schema,
        handler=web_search.handler,
        is_async=web_search.is_async,
        args_model=web_search.args_model,
        positional_only=web_search.positional_only,
    )
