"""Tests for ``logpose.discover_tools``.

Two kinds of fixture are used. Real packages under ``tests/fixtures_*`` exercise the
import machinery — recursion, skip-before-import, failing imports — which cannot be
faked. Synthetic ``ModuleType`` objects cover namespace *shapes* (a tool in a class
body, two names for one object) without needing a file each.

No network and no provider: the one end-to-end test drives ``FakeProvider``.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

import tests.fixtures_discovery as discovery_pkg
from logpose import Agent, ToolDef, discover_tools, tool
from logpose.errors import LogposeError
from tests.fake_provider import FakeProvider, ScriptedTurn, tool_call

PKG = "tests.fixtures_discovery"


def synthetic(name: str = "synthetic", **attributes: object) -> ModuleType:
    """Build a throwaway module with the given attributes."""
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


@tool
def module_level_tool() -> str:
    """Defined at module level so the zero-argument form can find it."""
    return "ok"


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------


def test_a_string_target_imports_the_module_and_finds_its_tools():
    names = [t.name for t in discover_tools(f"{PKG}.first")]
    assert names == ["alpha", "beta"]


def test_a_module_object_target_is_used_as_is():
    names = [t.name for t in discover_tools(discovery_pkg, recursive=False)]
    assert names == ["alpha"]


def test_several_targets_are_scanned_in_argument_order():
    names = [t.name for t in discover_tools(f"{PKG}.second", f"{PKG}.first")]
    assert names == ["gamma", "alpha", "beta"]


def test_the_same_target_twice_yields_each_tool_once():
    names = [t.name for t in discover_tools(f"{PKG}.first", f"{PKG}.first")]
    assert names == ["alpha", "beta"]


def test_overlapping_targets_do_not_duplicate_tools():
    names = [t.name for t in discover_tools(PKG, f"{PKG}.first")]
    assert names == sorted(set(names))


def test_a_package_with_no_tools_returns_an_empty_list():
    assert discover_tools("tests.fixtures_empty") == []


def test_a_missing_target_raises_naming_it():
    with pytest.raises(LogposeError, match="tests.fixtures_does_not_exist"):
        discover_tools("tests.fixtures_does_not_exist")


# ---------------------------------------------------------------------------
# recursion
# ---------------------------------------------------------------------------


def test_a_package_target_walks_its_submodules():
    names = {t.name for t in discover_tools(PKG)}
    assert {"alpha", "beta", "gamma"} <= names


def test_recursion_reaches_a_nested_subpackage():
    assert "delta" in {t.name for t in discover_tools(PKG)}


def test_recursive_false_scans_only_the_package_namespace():
    names = [t.name for t in discover_tools(PKG, recursive=False)]
    assert names == ["alpha"], "only the re-export in __init__ should be seen"


def test_recursive_is_a_no_op_for_a_plain_module():
    names = [t.name for t in discover_tools(f"{PKG}.second", recursive=True)]
    assert names == ["gamma"]


def test_underscore_prefixed_submodules_are_skipped():
    assert "skipped_tool" not in {t.name for t in discover_tools(PKG)}


def test_a_skipped_submodule_is_never_imported():
    """The reason for iter_modules over walk_packages: filter before importing."""
    sys.modules.pop(f"{PKG}._hidden", None)
    discover_tools(PKG)
    assert f"{PKG}._hidden" not in sys.modules


def test_an_underscore_module_named_explicitly_is_still_scanned():
    """The skip rule applies to walked submodules, never to an explicit target."""
    names = [t.name for t in discover_tools(f"{PKG}._hidden")]
    assert names == ["skipped_tool"]


def test_a_namespace_subpackage_is_not_walked():
    """A directory with no __init__.py is invisible to iter_modules."""
    assert "ghost_tool" not in {t.name for t in discover_tools(PKG)}


def test_a_submodule_that_fails_to_import_raises_naming_it():
    with pytest.raises(LogposeError, match=r"tests\.fixtures_broken\.bad") as excinfo:
        discover_tools("tests.fixtures_broken")
    assert isinstance(excinfo.value.__cause__, ImportError)


# ---------------------------------------------------------------------------
# dedupe and conflicts
# ---------------------------------------------------------------------------


def test_a_re_exported_tool_is_returned_once():
    """`from .first import alpha` in __init__ puts one object in two namespaces."""
    alphas = [t for t in discover_tools(PKG) if t.name == "alpha"]
    assert len(alphas) == 1


def test_a_re_exported_tool_does_not_trip_the_agent_duplicate_check():
    """The point of identity dedupe: the result must be Agent-compatible."""
    agent = Agent(FakeProvider(), tools=discover_tools(PKG))
    assert "alpha" in {t.name for t in agent.tools}


def test_two_names_for_one_tool_object_yield_one_tool():
    module = synthetic(primary=module_level_tool, alias=module_level_tool)
    assert len(discover_tools(module)) == 1


def test_two_different_tools_with_the_same_name_raise():
    with pytest.raises(LogposeError, match="Duplicate tool name 'search'"):
        discover_tools("tests.fixtures_conflict")


def test_the_conflict_message_names_both_defining_modules():
    with pytest.raises(LogposeError) as excinfo:
        discover_tools("tests.fixtures_conflict")
    message = str(excinfo.value)
    assert "tests.fixtures_conflict.one" in message
    assert "tests.fixtures_conflict.two" in message


def test_the_same_function_decorated_twice_is_not_a_conflict():
    def handler(x: int) -> str:
        """Do a thing.

        Args:
            x: A number.
        """
        return str(x)

    first = tool(name="dup")(handler)
    second = tool(name="dup")(handler)
    assert first is not second

    module = synthetic(a=first, b=second)
    assert len(discover_tools(module)) == 1


# ---------------------------------------------------------------------------
# predicate
# ---------------------------------------------------------------------------


def test_a_predicate_filters_the_result():
    tools = discover_tools(PKG, predicate=lambda t: t.name != "beta")
    assert "beta" not in {t.name for t in tools}
    assert "alpha" in {t.name for t in tools}


def test_a_predicate_runs_before_conflict_detection():
    """Which is what makes it a usable escape hatch for a name clash."""
    tools = discover_tools(
        "tests.fixtures_conflict",
        predicate=lambda t: t.handler.__module__.endswith(".one"),
    )
    assert [t.name for t in tools] == ["search"]


def test_a_predicate_exception_propagates_unchanged():
    def boom(_: ToolDef) -> bool:
        raise ZeroDivisionError("from the predicate")

    with pytest.raises(ZeroDivisionError, match="from the predicate"):
        discover_tools(f"{PKG}.first", predicate=boom)


# ---------------------------------------------------------------------------
# ordering
# ---------------------------------------------------------------------------


def test_tools_are_returned_in_traversal_order_not_alphabetically():
    """Regression guard: appending must not reshuffle earlier entries.

    Traversal is the package namespace first (alpha, the re-export), then
    submodules in sorted name order: first (beta), nested.deep (delta), second
    (gamma). That happens to read alphabetically here, so the sibling assertion
    below is what actually pins non-sorting.
    """
    assert [t.name for t in discover_tools(PKG)] == ["alpha", "beta", "delta", "gamma"]


def test_argument_order_wins_over_alphabetical_order():
    names = [t.name for t in discover_tools(f"{PKG}.second", f"{PKG}.first")]
    assert names == ["gamma", "alpha", "beta"]
    assert names != sorted(names), "results must not be sorted by name"


def test_definition_order_within_a_module_is_preserved():
    module = synthetic(zebra=module_level_tool)
    other = tool(name="aardvark")(lambda: "x")
    module.aardvark = other  # bound second, so it must come second
    assert [t.name for t in discover_tools(module)] == ["module_level_tool", "aardvark"]


# ---------------------------------------------------------------------------
# the zero-argument form
# ---------------------------------------------------------------------------


def test_no_targets_scans_the_calling_modules_globals():
    # Membership, not equality: this module gains tools as tests are added.
    assert "module_level_tool" in {t.name for t in discover_tools()}


def test_no_targets_from_inside_a_helper_still_sees_module_globals():
    def helper() -> list[ToolDef]:
        return discover_tools()

    assert "module_level_tool" in {t.name for t in helper()}


def test_no_targets_does_not_see_tools_defined_in_a_local_scope():
    local_tool = tool(name="local_only")(lambda: "x")
    assert local_tool.name not in {t.name for t in discover_tools()}


# ---------------------------------------------------------------------------
# what is not discovered
# ---------------------------------------------------------------------------


def test_a_tool_in_a_class_body_is_not_discovered():
    class Holder:
        held = module_level_tool

    assert discover_tools(synthetic(Holder=Holder)) == []


def test_a_tool_held_only_in_a_list_is_not_discovered():
    assert discover_tools(synthetic(TOOLS=[module_level_tool])) == []


def test_dunder_all_is_not_consulted():
    module = synthetic(exposed=module_level_tool)
    module.__all__ = []  # would hide it, if it were honoured
    assert len(discover_tools(module)) == 1


# ---------------------------------------------------------------------------
# result shape and use
# ---------------------------------------------------------------------------


def test_the_result_is_a_plain_mutable_list_of_tooldefs():
    tools = discover_tools(PKG)
    assert isinstance(tools, list)
    assert all(isinstance(t, ToolDef) for t in tools)
    tools.append(module_level_tool)  # must not be frozen or a view


async def test_discovered_tools_reach_the_provider_and_can_be_called():
    provider = FakeProvider(
        [
            ScriptedTurn.tool_use(tool_call("alpha", {"x": 7})),
            ScriptedTurn.text("done"),
        ]
    )
    agent = Agent(provider, tools=discover_tools(PKG))
    result = await agent.run("go")

    advertised = {spec.name for spec in provider.requests[0].tools}
    assert {"alpha", "beta", "gamma", "delta"} <= advertised
    assert result.text == "done"
    assert "alpha:7" in str(provider.requests[1].messages[-1].content)

