# `current_time` tool and builder removal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an exported `current_time` tool and remove the unused tool builders and CLI-era modules.

**Architecture:** `current_time` is a module-level `@tool` function in a new `src/logpose/clock.py`, exported from `logpose`. Removals delete whole modules (shell, workspace) and one builder (web); compaction's builder becomes private.

**Tech Stack:** Python 3.10+ (`zoneinfo`), `uv`, pytest.

**Spec:** `docs/superpowers/specs/2026-11-26-current-time-tool-design.md`

## Global Constraints

- Output format exactly `"%A, %d %B %Y, %H:%M %Z"` + ` (<IANA name>)`; default `timezone="UTC"`.
- Invalid/unknown zone → returned message containing the bad value and `'Asia/Kolkata'` as an example; never raises.
- No new public API beyond `current_time`. Gates: `uv run --all-extras pytest -q`, `uv run ruff check`, `uv run --all-extras mypy src/`.

## Review Focus

1. A time zone name the model invents with odd characters (`"../etc"`, `""`, `"Mars/Base"`) returns the friendly message, not an exception — Task 1 test.
2. `current_time` used through a real `Agent` tool round (FakeProvider) produces a tool result string — Task 1 test.
3. Nothing in `src/` or `tests/` still imports `logpose.shell`, `logpose.workspace` or `make_search_tool` — Task 1 grep check.

---

### Task 1: Add `current_time`; remove builders; docs

**Files:**
- Create: `src/logpose/clock.py`, `tests/test_clock.py`
- Modify: `src/logpose/__init__.py` (export `current_time`, `__all__`), `tests/test_public_api.py`, `src/logpose/web.py` (drop `make_search_tool` and now-unused imports), `src/logpose/compaction.py` + `src/logpose/agent.py` + `tests/test_compaction.py` (rename `make_recall_tool` → `_recall_tool`), `README.md`, `CHANGELOG.md`
- Delete: `src/logpose/shell.py`, `src/logpose/workspace.py`, `tests/test_shell.py`, `tests/test_workspace.py`, the `make_search_tool` tests in `tests/test_web_search_backends.py` (delete the file if nothing else remains)

**Interfaces:**
- Produces: `logpose.current_time` — a `ToolDef` named `current_time` with one optional string parameter `timezone` (default `"UTC"`).

- [ ] **Step 1: Failing tests** in `tests/test_clock.py` (freeze time by monkeypatching the module's clock source, e.g. a module-level `_now(tz)` helper):
  - `test_formats_time_in_zone`: 2026-11-26 08:35 UTC with `timezone="Asia/Kolkata"` → `"Thursday, 26 November 2026, 14:05 IST (Asia/Kolkata)"`.
  - `test_default_is_utc`: → ends with `"UTC (UTC)"`.
  - `test_invalid_zone_returns_message`: for `"Mars/Base"`, `""`, `"../etc/passwd"` → no exception; the text contains the bad value repr and `'Asia/Kolkata'`.
  - `test_tool_schema`: `current_time.spec()` name `current_time`, `timezone` optional string with a description.
  - `test_agent_tool_round`: FakeProvider tool_use `current_time` → the run's tool result message contains the formatted time.
  - `test_exported`: `from logpose import current_time` works (also add to test_public_api).
- [ ] **Step 2:** Run them — expect failures.
- [ ] **Step 3:** Implement `clock.py`; export; delete modules/tests; remove `make_search_tool`; rename the recall builder; update README (tool-output paragraph → about logpose's tool output budget generally / `recall_tool_output` and web helpers; deadline paragraph without "shell commands"; a "Built-in tools" note with the system-prompt time zone tip and an `@tool` wrapper example for a fixed zone) and CHANGELOG (Added; Removed — Breaking).
- [ ] **Step 4:** `grep -rn "logpose.shell\|logpose.workspace\|make_search_tool\|make_shell_tool\|make_read_file_tool\|make_list_dir_tool\|make_grep_tool\|make_recall_tool" src tests README.md` → no hits (CHANGELOG history entries may stay). Full gates pass.
- [ ] **Step 5: Commit** (explicit paths, deletions via `git rm`).
