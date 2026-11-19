"""Bounded shell execution: capture caps, group kill, and the tool factory."""

from __future__ import annotations

import subprocess
import sys

import pytest

from logpose.shell import bounded_run, make_shell_tool

pytestmark = pytest.mark.skipif(
    __import__("os").name != "posix", reason="bounded capture is POSIX-only"
)


def test_bounded_run_returns_output_and_exit_code(tmp_path):
    result = bounded_run("echo hello; exit 3", cwd=str(tmp_path))
    assert result.returncode == 3
    assert "hello" in result.stdout
    assert result.capture_complete is True


def test_bounded_run_kills_the_group_on_timeout(tmp_path):
    # The child outlives the shell; the group kill must take it down.
    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        bounded_run("sleep 30 & wait", cwd=str(tmp_path), timeout=0.5)
    assert excinfo.value.output is not None


def test_bounded_run_caps_capture(tmp_path):
    result = bounded_run("yes | head -c 200000", cwd=str(tmp_path), capture_limit=1000)
    assert result.capture_complete is False
    assert result.capture_note == "byte limit"


def test_bounded_run_drains_excess_without_retaining_it(tmp_path):
    command = f"{sys.executable} -c \"print('x'*10000)\""
    result = bounded_run(command, cwd=str(tmp_path), capture_limit=64)
    assert result.returncode == 0
    assert len(result.stdout) == 64
    assert result.capture_complete is False


def test_inherited_stdout_pipe_obeys_the_deadline(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        bounded_run("sleep 5 & exit 0", cwd=str(tmp_path), timeout=0.1)


def test_timeout_after_stdout_closes_retains_observed_output(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired) as error:
        bounded_run("printf before; exec 1>&- 2>&-; sleep 2", cwd=str(tmp_path), timeout=0.1)
    assert error.value.output == "before"


def test_make_shell_tool_reports_exit_code_and_truncates(tmp_path):
    shell = make_shell_tool(cwd=str(tmp_path), timeout=10, output_char_cap=50)
    out = shell("echo " + "x" * 200)
    assert out.startswith("[exit 0]")
    assert out.endswith("...[truncated]")
    assert shell("exit 7").startswith("[exit 7]")


def test_make_shell_tool_reports_timeout(tmp_path):
    shell = make_shell_tool(cwd=str(tmp_path), timeout=0.3)
    assert "Timed out after 0.3s" in shell("sleep 5")


def test_make_shell_tool_resolves_callable_cwd(tmp_path):
    shell = make_shell_tool(cwd=lambda: str(tmp_path), timeout=10)
    assert shell(f"ls {tmp_path}").startswith("[exit 0]")


def test_shell_defaults_are_web_app_safe(tmp_path):
    shell = make_shell_tool(cwd=str(tmp_path))
    out = shell("yes x | head -c 20000")
    assert out.endswith("...[truncated]")
    assert len(out) <= 16_000 + 20
