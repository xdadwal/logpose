"""Tests for the tool definition API (``logpose.tools``).

No network, no provider: everything here is pure schema generation and handler
dispatch.
"""

import asyncio
import json
import threading
from typing import Any, Literal, Optional

import pytest
from pydantic import BaseModel, Field

from logpose.errors import ToolExecutionError, ToolSchemaError
from logpose.providers.base import ToolSpec
from logpose.tools import ToolDef, tool


class Address(BaseModel):
    """A postal address used to exercise nested-model schema generation."""

    street: str
    zip_code: str | None = None


class Node(BaseModel):
    """A self-referencing model used to exercise the recursive-schema path."""

    value: int
    child: "Node | None" = None


Node.model_rebuild()


# ---------------------------------------------------------------------------
# Decorator forms
# ---------------------------------------------------------------------------


def test_bare_decorator_builds_a_tooldef():
    @tool
    def ping() -> str:
        """Check liveness."""
        return "pong"

    assert isinstance(ping, ToolDef)
    assert ping.name == "ping"
    assert ping.description == "Check liveness."
    assert ping.is_async is False


def test_called_decorator_overrides_name_and_description():
    @tool(name="weather", description="Look up the weather.")
    def _weather(city: str) -> str:
        """This docstring is ignored."""
        return "sunny"

    assert _weather.name == "weather"
    assert _weather.description == "Look up the weather."


def test_called_decorator_with_no_arguments():
    @tool()
    def ping() -> str:
        """Check liveness."""
        return "pong"

    assert isinstance(ping, ToolDef)
    assert ping.name == "ping"


def test_description_is_the_first_paragraph_of_the_docstring():
    @tool
    def summarize(text: str) -> str:
        """Summarize a document
        across two source lines.

        This second paragraph is not part of the summary.

        Args:
            text: The text.
        """
        return text

    assert summarize.description == "Summarize a document across two source lines."


def test_missing_docstring_yields_an_empty_description():
    @tool
    def bare(x: int) -> str:
        return str(x)

    assert bare.description == ""


def test_tooldef_is_still_callable_as_a_plain_function():
    @tool
    def double(x: int) -> int:
        """Double a number."""
        return x * 2

    assert double(21) == 42


async def test_callable_objects_are_supported():
    class Fetcher:
        async def __call__(self, url: str) -> str:
            """Fetch a URL.

            Args:
                url: The URL to fetch.
            """
            return f"got {url}"

    fetch = tool(name="fetch")(Fetcher())

    assert fetch.is_async is True
    assert fetch.description == "Fetch a URL."
    assert fetch.input_schema["properties"]["url"]["description"] == "The URL to fetch."
    assert await fetch.invoke({"url": "http://x"}) == "got http://x"


def test_decorating_a_tooldef_twice_raises():
    @tool
    def ping() -> str:
        """Check liveness."""
        return "pong"

    with pytest.raises(ToolSchemaError, match="already a ToolDef"):
        tool(ping)


# ---------------------------------------------------------------------------
# Schema shape
# ---------------------------------------------------------------------------


def test_schema_matches_the_provider_contract():
    @tool
    def noop() -> str:
        """Do nothing."""
        return ""

    assert noop.input_schema == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }


def test_scalar_types_map_to_json_schema_types():
    @tool
    def scalars(name: str, count: int, ratio: float, flag: bool) -> str:
        """Take one of every scalar."""
        return "ok"

    props = scalars.input_schema["properties"]
    assert props["name"] == {"type": "string"}
    assert props["count"] == {"type": "integer"}
    assert props["ratio"] == {"type": "number"}
    assert props["flag"] == {"type": "boolean"}
    assert scalars.input_schema["required"] == ["name", "count", "ratio", "flag"]
    assert scalars.input_schema["additionalProperties"] is False


def test_literal_becomes_an_enum():
    @tool
    def convert(unit: Literal["c", "f"]) -> str:
        """Convert a temperature."""
        return unit

    assert convert.input_schema["properties"]["unit"]["enum"] == ["c", "f"]
    assert convert.input_schema["properties"]["unit"]["type"] == "string"


def test_single_value_literal_becomes_a_const():
    @tool
    def only(mode: Literal["strict"]) -> str:
        """Run in one mode."""
        return mode

    assert only.input_schema["properties"]["mode"]["const"] == "strict"


def test_container_types():
    @tool
    def containers(tags: list[str], meta: dict[str, int]) -> str:
        """Take containers."""
        return "ok"

    props = containers.input_schema["properties"]
    assert props["tags"] == {"type": "array", "items": {"type": "string"}}
    assert props["meta"] == {"type": "object", "additionalProperties": {"type": "integer"}}


def test_optional_and_union_with_none():
    @tool
    # Both spellings are supported; the legacy `Optional[...]` form is exercised
    # on purpose, so the modernization lint is suppressed here.
    def optionals(a: str | None, b: Optional[int] = None) -> str:  # noqa: UP045
        """Take optionals."""
        return "ok"

    props = optionals.input_schema["properties"]
    assert props["a"]["anyOf"] == [{"type": "string"}, {"type": "null"}]
    assert props["b"]["anyOf"] == [{"type": "integer"}, {"type": "null"}]
    assert props["b"]["default"] is None
    # `a` has no default, so it is required even though it is nullable.
    assert optionals.input_schema["required"] == ["a"]


def test_defaults_decide_required_vs_optional():
    @tool
    def mixed(required_one: str, optional_one: int = 5, optional_two: str = "x") -> str:
        """Mix required and optional parameters."""
        return "ok"

    schema = mixed.input_schema
    assert schema["required"] == ["required_one"]
    assert set(schema["properties"]) == {"required_one", "optional_one", "optional_two"}
    assert schema["properties"]["optional_one"]["default"] == 5
    assert schema["properties"]["optional_two"]["default"] == "x"


def test_pydantic_model_parameter_is_inlined():
    @tool
    def ship(addr: Address) -> str:
        """Ship a package."""
        return "ok"

    schema = ship.input_schema
    assert "$defs" not in schema
    assert "$ref" not in json.dumps(schema)

    addr_schema = schema["properties"]["addr"]
    assert addr_schema["type"] == "object"
    assert addr_schema["additionalProperties"] is False
    assert addr_schema["required"] == ["street"]
    assert addr_schema["properties"]["street"] == {"type": "string"}
    assert addr_schema["properties"]["zip_code"]["anyOf"] == [
        {"type": "string"},
        {"type": "null"},
    ]


def test_recursive_model_keeps_defs_and_stays_self_contained():
    @tool
    def walk(root: Node) -> str:
        """Walk a tree."""
        return str(root.value)

    schema = walk.input_schema
    # A recursive model cannot be inlined; the $defs block is preserved instead,
    # so every $ref still resolves within the emitted schema.
    assert "$defs" in schema
    assert schema["properties"]["root"]["$ref"] == "#/$defs/Node"
    assert "Node" in schema["$defs"]


def test_schema_carries_no_title_noise():
    @tool
    def titled(title: str, addr: Address) -> str:
        """Check that pydantic titles are stripped.

        Args:
            title: A parameter that is literally named "title".
            addr: An address.
        """
        return "ok"

    schema = titled.input_schema
    # The pydantic-generated "title" keywords are gone...
    assert "title" not in schema
    assert "title" not in schema["properties"]["addr"]
    assert "title" not in schema["properties"]["addr"]["properties"]["street"]
    # ...but a parameter *named* "title" survives.
    assert schema["properties"]["title"]["type"] == "string"
    assert (
        schema["properties"]["title"]["description"]
        == 'A parameter that is literally named "title".'
    )


# ---------------------------------------------------------------------------
# Docstring-driven parameter descriptions
# ---------------------------------------------------------------------------


def test_docstring_arg_descriptions_land_in_the_schema():
    @tool
    def get_weather(location: str, unit: Literal["c", "f"] = "c") -> str:
        """Get current weather.

        Args:
            location: City name.
            unit: Temperature unit.

        Returns:
            A description of the weather.
        """
        return "sunny"

    props = get_weather.input_schema["properties"]
    assert props["location"]["description"] == "City name."
    assert props["unit"]["description"] == "Temperature unit."


def test_docstring_arg_descriptions_join_continuation_lines():
    @tool
    def wrapped(value: int) -> str:
        """Take a value.

        Args:
            value: A number that needs
                a long explanation.
        """
        return str(value)

    description = wrapped.input_schema["properties"]["value"]["description"]
    assert description == "A number that needs a long explanation."


def test_docstring_arg_descriptions_tolerate_type_annotations_in_parens():
    @tool
    def annotated(value: int) -> str:
        """Take a value.

        Args:
            value (int): The number.
        """
        return str(value)

    assert annotated.input_schema["properties"]["value"]["description"] == "The number."


def test_args_section_stops_at_the_next_section():
    @tool
    def sectioned(value: int) -> str:
        """Take a value.

        Args:
            value: The number.

        Raises:
            ValueError: never.
        """
        return str(value)

    props = sectioned.input_schema["properties"]
    assert props["value"]["description"] == "The number."
    assert "ValueError" not in json.dumps(props)


def test_explicit_field_description_wins_over_the_docstring():
    @tool
    def described(value: int = Field(default=1, description="From Field.")) -> str:
        """Take a value.

        Args:
            value: From the docstring.
        """
        return str(value)

    assert described.input_schema["properties"]["value"]["description"] == "From Field."


# ---------------------------------------------------------------------------
# spec()
# ---------------------------------------------------------------------------


def test_spec_returns_a_toolspec_without_the_handler():
    @tool
    def get_weather(location: str) -> str:
        """Get current weather.

        Args:
            location: City name.
        """
        return "sunny"

    spec = get_weather.spec()
    assert isinstance(spec, ToolSpec)
    assert spec.name == "get_weather"
    assert spec.description == "Get current weather."
    assert spec.input_schema == get_weather.input_schema
    assert not hasattr(spec, "handler")


# ---------------------------------------------------------------------------
# invoke(): dispatch
# ---------------------------------------------------------------------------


async def test_sync_handler_runs_off_the_event_loop():
    @tool
    def where(x: int) -> str:
        """Report the thread it ran on.

        Args:
            x: Ignored.
        """
        return threading.current_thread().name

    result = await where.invoke({"x": 1})
    assert result != threading.current_thread().name


async def test_async_handler_is_awaited_directly():
    @tool
    async def fetch(url: str) -> str:
        """Fetch a URL.

        Args:
            url: The URL.
        """
        await asyncio.sleep(0)
        return f"fetched {url}"

    assert fetch.is_async is True
    assert await fetch.invoke({"url": "http://x"}) == "fetched http://x"


async def test_arguments_are_coerced_before_the_handler_sees_them():
    seen: dict[str, Any] = {}

    @tool
    def take(count: int) -> str:
        """Take a count.

        Args:
            count: How many.
        """
        seen["count"] = count
        return "ok"

    await take.invoke({"count": "3"})
    assert seen["count"] == 3


async def test_optional_arguments_fall_back_to_their_defaults():
    @tool
    def greet(name: str, greeting: str = "hello") -> str:
        """Greet someone.

        Args:
            name: Who to greet.
            greeting: The greeting to use.
        """
        return f"{greeting} {name}"

    assert await greet.invoke({"name": "ada"}) == "hello ada"


async def test_pydantic_model_arguments_arrive_as_model_instances():
    @tool
    def ship(addr: Address) -> str:
        """Ship a package.

        Args:
            addr: Where to ship it.
        """
        assert isinstance(addr, Address)
        return addr.street

    assert await ship.invoke({"addr": {"street": "1 Main St"}}) == "1 Main St"


async def test_positional_only_parameters_are_passed_positionally():
    @tool
    def positional(a: int, /, b: int = 2) -> str:
        """Add two numbers.

        Args:
            a: First.
            b: Second.
        """
        return str(a + b)

    assert positional.positional_only == ("a",)
    assert await positional.invoke({"a": 1, "b": 5}) == "6"


# ---------------------------------------------------------------------------
# invoke(): return-value coercion
# ---------------------------------------------------------------------------


async def test_str_return_passes_through():
    @tool
    def echo(x: str) -> str:
        """Echo.

        Args:
            x: The text.
        """
        return x

    assert await echo.invoke({"x": "plain"}) == "plain"


async def test_dict_return_is_json_encoded():
    @tool
    def payload(x: int) -> dict:
        """Return a dict.

        Args:
            x: A number.
        """
        return {"value": x, "ok": True, "none": None}

    result = await payload.invoke({"x": 7})
    assert json.loads(result) == {"value": 7, "ok": True, "none": None}


async def test_list_return_is_json_encoded():
    @tool
    def listing(x: int) -> list:
        """Return a list.

        Args:
            x: A number.
        """
        return [x, "two", None]

    assert json.loads(await listing.invoke({"x": 1})) == [1, "two", None]


async def test_unserializable_container_falls_back_to_str():
    @tool
    def weird(x: int) -> dict:
        """Return a dict with a non-JSON value.

        Args:
            x: A number.
        """
        return {"obj": object()}

    result = await weird.invoke({"x": 1})
    assert "obj" in result


async def test_other_return_types_are_stringified():
    @tool
    def number(x: int) -> int:
        """Return a number.

        Args:
            x: A number.
        """
        return x * 2

    assert await number.invoke({"x": 4}) == "8"


async def test_none_return_is_stringified():
    @tool
    def silent(x: int) -> None:
        """Return nothing.

        Args:
            x: A number.
        """
        return None

    assert await silent.invoke({"x": 1}) == "None"


# ---------------------------------------------------------------------------
# invoke(): error contract
# ---------------------------------------------------------------------------


async def test_handler_exception_becomes_tool_execution_error():
    @tool
    def boom(x: int) -> str:
        """Explode.

        Args:
            x: Ignored.
        """
        raise RuntimeError("kaboom")

    with pytest.raises(ToolExecutionError) as info:
        await boom.invoke({"x": 1})

    assert "kaboom" in str(info.value)
    assert "boom" in str(info.value)
    assert isinstance(info.value.__cause__, RuntimeError)


async def test_async_handler_exception_becomes_tool_execution_error():
    @tool
    async def boom(x: int) -> str:
        """Explode asynchronously.

        Args:
            x: Ignored.
        """
        raise ValueError("async kaboom")

    with pytest.raises(ToolExecutionError, match="async kaboom"):
        await boom.invoke({"x": 1})


async def test_wrongly_typed_argument_becomes_tool_execution_error():
    @tool
    def take(count: int) -> str:
        """Take a count.

        Args:
            count: How many.
        """
        return str(count)

    with pytest.raises(ToolExecutionError) as info:
        await take.invoke({"count": "not-a-number"})

    assert "count" in str(info.value)
    assert "take" in str(info.value)
    assert type(info.value.__cause__).__name__ == "ValidationError"


async def test_missing_argument_becomes_tool_execution_error():
    @tool
    def take(count: int) -> str:
        """Take a count.

        Args:
            count: How many.
        """
        return str(count)

    with pytest.raises(ToolExecutionError, match="Field required"):
        await take.invoke({})


async def test_unknown_argument_becomes_tool_execution_error():
    @tool
    def take(count: int) -> str:
        """Take a count.

        Args:
            count: How many.
        """
        return str(count)

    with pytest.raises(ToolExecutionError, match="Extra inputs are not permitted"):
        await take.invoke({"count": 1, "surprise": 2})


async def test_non_mapping_arguments_become_tool_execution_error():
    @tool
    def take(count: int) -> str:
        """Take a count.

        Args:
            count: How many.
        """
        return str(count)

    with pytest.raises(ToolExecutionError, match="expected an object"):
        await take.invoke(["not", "a", "mapping"])  # type: ignore[arg-type]


async def test_cancellation_is_not_swallowed():
    @tool
    async def cancels(x: int) -> str:
        """Raise CancelledError.

        Args:
            x: Ignored.
        """
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await cancels.invoke({"x": 1})


async def test_tool_execution_error_from_a_handler_is_not_double_wrapped():
    @tool
    def strict(x: int) -> str:
        """Raise a logpose error directly.

        Args:
            x: Ignored.
        """
        raise ToolExecutionError("already shaped")

    with pytest.raises(ToolExecutionError) as info:
        await strict.invoke({"x": 1})

    assert str(info.value) == "already shaped"


# ---------------------------------------------------------------------------
# Decoration-time schema errors
# ---------------------------------------------------------------------------


def test_var_positional_raises_at_decoration():
    with pytest.raises(ToolSchemaError, match=r"\*args"):

        @tool
        def variadic(*args: int) -> str:
            """Take anything."""
            return "ok"


def test_var_keyword_raises_at_decoration():
    with pytest.raises(ToolSchemaError, match=r"\*\*kwargs"):

        @tool
        def variadic(**kwargs: int) -> str:
            """Take anything."""
            return "ok"


def test_unannotated_parameter_raises_at_decoration():
    with pytest.raises(ToolSchemaError, match="no type annotation"):

        @tool
        def untyped(x) -> str:
            """Take an unannotated parameter."""
            return "ok"


def test_unresolvable_annotation_raises_at_decoration():
    with pytest.raises(ToolSchemaError):

        @tool
        def broken(x: "DefinitelyNotDefined") -> str:  # noqa: F821
            """Reference a name that does not exist."""
            return "ok"


def test_var_positional_raises_even_when_a_name_is_supplied():
    with pytest.raises(ToolSchemaError, match=r"\*args"):
        tool(name="explicit")(lambda *args: "ok")


@pytest.mark.parametrize("param", ["model_config", "_private", "__dunder"])
def test_a_pydantic_reserved_parameter_name_raises_at_decoration(param: str):
    """Regression: pydantic swallows these names — ``model_config`` becomes the
    model's config, a leading underscore becomes a private attribute — so the
    parameter vanished from the emitted schema with no error.

    The model then could never call the tool: the parameter is not advertised,
    ``_bind`` builds kwargs from ``model_fields`` which lacks it, and every
    invocation came back as ``TypeError: missing 1 required positional argument``.
    """
    namespace: dict[str, Any] = {}
    exec(  # noqa: S102 - the parameter name is the thing under test
        f"def reserved(name: str, {param}: str) -> str:\n"
        f'    """Do a thing.\n\n'
        f"    Args:\n"
        f"        name: The name.\n"
        f"        {param}: The other one.\n"
        f'    """\n'
        f"    return name\n",
        namespace,
    )

    with pytest.raises(ToolSchemaError, match="reserved by pydantic"):
        tool(namespace["reserved"])


def test_a_schematizable_signature_still_keeps_every_parameter():
    @tool
    def fine(name: str, config: str) -> str:
        """Do a thing.

        Args:
            name: The name.
            config: The config.
        """
        return name + config

    assert sorted(fine.input_schema["properties"]) == ["config", "name"]
    assert sorted(fine.args_model.model_fields) == ["config", "name"]


@pytest.mark.parametrize(
    ("name", "match"),
    [
        ("get weather now!", "characters providers reject"),
        ("weather.now", "characters providers reject"),
        ("тула", "characters providers reject"),
        ("x" * 129, "at most 128"),
    ],
)
def test_an_illegal_tool_name_raises_at_decoration(name: str, match: str):
    """Regression: names were passed through untouched and only failed as an
    opaque provider 400 on the first live request — potentially in production,
    long after the decorator ran. Anthropic requires ``^[a-zA-Z0-9_-]{1,128}$``."""
    with pytest.raises(ToolSchemaError, match=match):

        @tool(name=name)
        def named(city: str) -> str:
            """Look something up.

            Args:
                city: City name.
            """
            return city


@pytest.mark.parametrize("name", ["get_weather", "get-weather", "GetWeather9", "x" * 128])
def test_a_legal_tool_name_is_accepted(name: str):
    definition = tool(name=name)(lambda: "ok")
    assert definition.name == name
