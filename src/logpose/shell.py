"""Bounded, hard-killable shell execution as a reusable agent capability.

Two layers:

:func:`bounded_run` — subprocess execution with merged output drained
non-blockingly, a byte-capped capture, and a timeout that ``SIGKILL``s the
whole process group (children included). This is the logpose answer to "a
cancelled ``to_thread`` handler cannot be stopped": the handler ends its own
process, deterministically, at its deadline.

:func:`make_shell_tool` — a ``ToolDef`` factory wrapping :func:`bounded_run`
for model use. Consumers keep policy (guarding, workspace confinement) on
their side by choosing ``cwd`` and how they register the result; logpose owns
the execution mechanics.
"""

from __future__ import annotations

import os
import selectors
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable

from logpose.cancellation import current_tool_deadline
from logpose.tools import ToolDef, tool

_DEFAULT_CAPTURE_LIMIT = 1024 * 1024
_DEFAULT_TIMEOUT = 25.0
# Shorter than the agent's 30 s tool_timeout, so this timer fires first and the
# model sees the partial output instead of a bare "exceeded its limit" result.
_DEFAULT_OUTPUT_CAP = 8_000


def _kill_group(pid: int) -> None:
    """SIGKILL a process group, ignoring one that has already exited."""
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def bounded_run(
    command: str,
    *,
    cwd: str,
    input: str = "",
    timeout: float = _DEFAULT_TIMEOUT,
    capture_limit: int = _DEFAULT_CAPTURE_LIMIT,
) -> subprocess.CompletedProcess:
    """Run ``command`` with merged output, a capture cap, and a group kill.

    On POSIX the command starts in its own session and, if it outlives
    ``timeout``, every process in its group is killed before the resulting
    :class:`subprocess.TimeoutExpired` (carrying partial output) propagates.
    On non-POSIX platforms this degrades to :func:`subprocess.run` semantics.

    The returned ``CompletedProcess`` carries two extra attributes when the
    bounded path ran: ``capture_complete`` (``False`` when output was clipped
    or not valid UTF-8) and ``capture_note`` explaining why.
    """
    if os.name != "posix":
        return subprocess.run(  # noqa: S602 - the command is the tool's contract
            command, cwd=cwd, input=input, timeout=timeout, shell=True,
            capture_output=True, text=True,
        )
    kept = bytearray()
    observed = 0
    deadline = time.monotonic() + timeout
    with tempfile.TemporaryFile() as incoming:
        incoming.write(input.encode("utf-8"))
        incoming.seek(0)
        process = subprocess.Popen(  # noqa: S602 - see above
            command, shell=True, cwd=cwd, stdin=incoming,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        assert process.stdout is not None
        reaped = threading.Event()
        handle = current_tool_deadline()
        if handle is not None:
            # A tool or run deadline in the agent kills the whole group, so no
            # child outlives the call that started it. Once this process is
            # reaped its pid may be reused, so the terminator becomes a no-op.
            def terminate(pid: int = process.pid) -> None:
                if not reaped.is_set():
                    _kill_group(pid)

            handle.register_terminator(terminate)
        settled = False
        try:
            os.set_blocking(process.stdout.fileno(), False)
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(
                            command, timeout, output=kept.decode("utf-8", errors="replace"))
                    for key, _ in selector.select(min(remaining, 0.1)):
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        observed += len(chunk)
                        kept.extend(chunk[:max(0, capture_limit - len(kept))])
            try:
                code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired as exc:
                raise subprocess.TimeoutExpired(
                    command, timeout, output=kept.decode("utf-8", errors="replace")) from exc
            settled = True
            try:
                decoded = kept.decode("utf-8")
                decoded_complete = True
            except UnicodeDecodeError:
                decoded = kept.decode("utf-8", errors="replace")
                decoded_complete = False
            result = subprocess.CompletedProcess(command, code, stdout=decoded, stderr="")
            result.capture_complete = observed <= capture_limit and decoded_complete  # type: ignore[attr-defined]
            result.capture_note = (  # type: ignore[attr-defined]
                "byte limit" if observed > capture_limit else "invalid UTF-8"
            )
            return result
        finally:
            # Include inherited pipes/children in the deadline and cleanup, even
            # when the shell has exited but a child still holds stdout open.
            if not settled:
                _kill_group(process.pid)
            process.stdout.close()
            process.wait()
            reaped.set()


def make_shell_tool(
    *,
    cwd: str | Callable[[], str],
    timeout: float = _DEFAULT_TIMEOUT,
    output_char_cap: int | None = _DEFAULT_OUTPUT_CAP,
    name: str = "run_shell",
    description: str | None = None,
) -> ToolDef:
    """Build a model-facing shell tool bound to ``cwd`` and ``timeout``.

    Args:
        cwd: Working directory for commands; a callable is resolved per call,
            which lets embedding applications bind a session root lazily.
        timeout: Seconds before the process group is killed.
        output_char_cap: Longest output returned to the model; ``None``
            disables truncation.
        name: Tool name the model calls (default ``run_shell``).
        description: Override the model-facing description.

    Returns:
        A :class:`~logpose.tools.ToolDef`. Registration is the caller's
        decision: pass it to ``Agent(tools=[...])`` as-is.
    """
    summary = description or (
        "Run a shell command and return its combined stdout+stderr. Use this "
        "for anything the other tools cannot do. It runs with a time limit "
        "and very long output is truncated. The exit code is reported so you "
        "can tell success from failure. If the command might prompt for "
        "input, supply the answer(s) via `stdin`, each ending in a newline."
    )

    def _cap(text: str) -> str:
        if output_char_cap is None or len(text) <= output_char_cap:
            return text
        return text[:output_char_cap] + "\n...[truncated]"

    @tool
    def run_shell(command: str, stdin: str = "") -> str:
        """Run a shell command and return its combined output with exit code."""
        resolved_cwd = cwd() if callable(cwd) else cwd
        try:
            completed = bounded_run(command, cwd=resolved_cwd, input=stdin, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            partial = exc.output or ""
            return _cap(f"Timed out after {timeout:g}s\n{partial}".strip())
        output = (completed.stdout or "") + (completed.stderr or "")
        status = "exit 0" if completed.returncode == 0 else f"exit {completed.returncode}"
        complete = getattr(completed, "capture_complete", True)
        if not complete:
            output += "\n...[capture incomplete: " + str(
                getattr(completed, "capture_note", "bounded capture")
            ) + "]"
        return _cap(f"[{status}]\n{output.strip() or '(no output)'}")

    decorated: ToolDef = run_shell
    # The decorator derives the schema from the stub; re-emit with the
    # caller-chosen name and model-facing summary.
    return ToolDef(
        name=name,
        description=summary,
        input_schema=decorated.input_schema,
        handler=decorated.handler,
        is_async=decorated.is_async,
        args_model=decorated.args_model,
        positional_only=decorated.positional_only,
    )
