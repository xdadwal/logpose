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

    Model-oriented output: a clean failure reason on block/unknown pages (never
    an empty-looking success), or `title — url` lines with snippets.
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
        return f"Web search for {query!r} failed: {exc}"
    soup = BeautifulSoup(response.text, "html.parser")
    if soup.select_one("#challenge-form, #anomaly-form, .anomaly-modal"):
        return (
            "Web search for "
            f"{query!r} blocked by DuckDuckGo human verification "
            f"(HTTP {response.status_code}). This is not an empty search. "
            "Do not retry the same request; use a paid search backend or "
            "fetch a known source URL directly."
        )
    if response.status_code != 200:
        return (
            f"Web search for {query!r} failed: unexpected "
            f"DuckDuckGo HTTP {response.status_code}."
        )
    results = parse_duckduckgo_results(response.text, max_results)
    if not results and not soup.select_one(".no-results, .result--no-result"):
        return (
            f"Web search for {query!r} failed: unrecognized DuckDuckGo response. "
            "No result count could be established."
        )
    if not results:
        return f"No web results for {query!r}. Try different or broader search terms."
    lines = [f"Web results for {query!r}:"]
    for result in results:
        lines.append(result)
    return "\n".join(lines)


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
