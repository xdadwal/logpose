# Built-in `current_time` tool; remove the tool builders

Status: approved 2026-11-26 (short design; owner decided in conversation).

## Intent

The xdadwal.in chat needs to know the current date and time. The owner chose a
simple tool over injecting the time into each request, and wants no `make_*`
tool-builder functions: logpose keeps base implementations, and apps wrap what
they need with `@tool`. The shell and workspace code is a leftover from
logpose's CLI-agent days and has no users.

## Decided

- **Add** `current_time` to `logpose` (exported): a plain `@tool` function
  `current_time(timezone: str = "UTC") -> str` returning
  `"%A, %d %B %Y, %H:%M %Z"` for `datetime.now(ZoneInfo(timezone))` plus the
  IANA name, e.g. `Thursday, 26 November 2026, 14:05 IST (Asia/Kolkata)`.
  Unknown or invalid time zone → a short message naming the bad value and an
  example (`'Asia/Kolkata'`), never an exception. Its docstring tells the model
  to use it for today, now, or relative dates and that `timezone` is an IANA
  name.
- **Configuration (Option B, owner's choice):** no new API. The default is UTC;
  an app that wants the model to use its time zone says so in the system prompt
  (e.g. "Site time zone: Asia/Kolkata; pass it to current_time"). README shows
  this, and shows wrapping it with `@tool` for a fixed zone as an alternative.
- **Remove** `src/logpose/shell.py` and `src/logpose/workspace.py` entirely, with
  `tests/test_shell.py` and `tests/test_workspace.py` (no other users).
- **Remove** `make_search_tool` from `src/logpose/web.py` and its tests; keep the
  base functions (`fetch_readable_text`, `search_duckduckgo`,
  `duckduckgo_search_entries`, `SearchFailure`, parsing helpers) and their
  tests.
- **Rename** compaction's internal `make_recall_tool` to `_recall_tool` (not a
  public tool builder; built per run).
- **Docs:** README tool-output paragraph and the deadline paragraph stop
  mentioning shell/`read_file`/`list_dir`/grep; README gets a short "Built-in
  tools" note for `current_time`. CHANGELOG `[Unreleased]`: Added
  `current_time`; Removed (Breaking) `logpose.shell`, `logpose.workspace`,
  `make_search_tool` (they were importable from their modules, never exported
  from `logpose`).

## Non-goals

Per-run time zones; injecting the time into requests; a generic way to fix tool
defaults.
