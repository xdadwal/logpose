"""Tool definition API.

Consumers describe a tool by writing an ordinary Python function and decorating
it with :func:`tool`. The decorator inspects the signature and docstring and
produces a :class:`ToolDef` that carries everything the loop and the provider
need: a JSON Schema for the arguments, a human description, and a handler.

.. code-block:: python

    from typing import Literal

    from logpose import tool


    @tool
    def get_weather(location: str, unit: Literal["c", "f"] = "c") -> str:
        '''Get the current weather.

        Args:
            location: City name, e.g. "Pune".
            unit: Temperature unit.
        '''
        return f"22 degrees {unit} in {location}"

Schema generation
-----------------
The schema is produced by pydantic from the annotations, then normalized to the
shape providers expect::

    {"type": "object", "properties": {...}, "required": [...],
     "additionalProperties": false}

``$ref``/``$defs`` produced by nested :class:`pydantic.BaseModel` parameters are
**inlined** so the emitted schema is flat and self-contained. The one exception
is a *recursive* model, which cannot be inlined without looping: there the
``$defs`` block is preserved alongside the ``$ref`` (still self-contained, just
not flat).

Error contract
--------------
:meth:`ToolDef.invoke` never lets a handler exception escape. Both argument
validation failures and handler failures are re-raised as
:class:`~logpose.errors.ToolExecutionError` with the original exception chained
as ``__cause__``; the loop is expected to convert that into a
``ToolResultBlock(is_error=True)`` so the model can adapt. Only
:class:`BaseException` subclasses that are not :class:`Exception` (notably
``asyncio.CancelledError``) propagate untouched.

Un-schematizable signatures (``*args``, ``**kwargs``, a parameter with no
annotation, a parameter whose name pydantic reserves) raise
:class:`~logpose.errors.ToolSchemaError` at decoration time, as does a tool name
outside ``[A-Za-z0-9_-]{1,128}`` — the charset every targeted provider accepts.
Nothing that would fail as an opaque provider 400 is allowed to reach a request.

Discovery
---------
:func:`tool` is a pure factory — it returns a :class:`ToolDef` and registers it
nowhere, so this module holds no mutable state. :func:`discover_tools` is the
consequence: collecting the tools in a package means importing its modules and
scanning their namespaces, rather than reading a registry.

That import step is why the underscore-prefix skip is applied *before* a submodule
is imported, and why a failing import is reported rather than swallowed: a scan
executes the top-level code of everything it touches, and silently missing a tool
is worse than a loud error.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import pkgutil
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any, overload

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from logpose.errors import LogposeError, ToolExecutionError, ToolSchemaError
from logpose.providers.base import ToolSpec

__all__ = ["ToolDef", "discover_tools", "tool"]


# ---------------------------------------------------------------------------
# Docstring parsing (Google style)
# ---------------------------------------------------------------------------

_ARGS_HEADERS = frozenset(
    {"args", "arguments", "parameters", "params", "keyword args", "keyword arguments"}
)
_SECTION_HEADERS = _ARGS_HEADERS | frozenset(
    {
        "attributes",
        "example",
        "examples",
        "note",
        "notes",
        "raises",
        "references",
        "return",
        "returns",
        "see also",
        "todo",
        "warning",
        "warnings",
        "yield",
        "yields",
    }
)

_HEADER_RE = re.compile(r"^(?P<name>[A-Za-z][A-Za-z ]*):$")
_PARAM_RE = re.compile(
    r"^(?P<name>\*{0,2}[A-Za-z_][A-Za-z0-9_]*)\s*(?:\((?P<type>[^)]*)\))?\s*:\s?(?P<desc>.*)$"
)


def _indent_of(line: str) -> int:
    """Return the number of leading whitespace characters in ``line``."""
    return len(line) - len(line.lstrip())


def _section_name(stripped_line: str) -> str | None:
    """Return the lowercase section name if ``stripped_line`` is a section header.

    Args:
        stripped_line: A docstring line with surrounding whitespace removed.

    Returns:
        The recognized section name, or ``None`` if the line is not a header.
    """
    match = _HEADER_RE.match(stripped_line)
    if match is None:
        return None
    name = match.group("name").strip().lower()
    return name if name in _SECTION_HEADERS else None


def _parse_args_block(block: list[str]) -> dict[str, str]:
    """Parse the body of a Google-style ``Args:`` section.

    Args:
        block: The raw lines making up the section body.

    Returns:
        A mapping of parameter name to its description.
    """
    indents = [_indent_of(line) for line in block if line.strip()]
    if not indents:
        return {}
    base = min(indents)

    entries: dict[str, str] = {}
    current: str | None = None
    for line in block:
        if not line.strip():
            continue
        if _indent_of(line) == base:
            match = _PARAM_RE.match(line.strip())
            if match is None:
                current = None
                continue
            current = match.group("name").lstrip("*")
            entries[current] = match.group("desc").strip()
        elif current is not None:
            continuation = line.strip()
            entries[current] = f"{entries[current]} {continuation}".strip()
    return entries


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a docstring into its summary and its per-parameter descriptions.

    Args:
        doc: The raw docstring, or ``None``.

    Returns:
        A ``(summary, param_descriptions)`` pair. The summary is the first
        paragraph joined into a single line; ``param_descriptions`` maps
        parameter names to the text from the ``Args:`` section.
    """
    if not doc:
        return "", {}

    lines = inspect.cleandoc(doc).splitlines()
    total = len(lines)

    summary_parts: list[str] = []
    index = 0
    while index < total:
        stripped = lines[index].strip()
        if not stripped or _section_name(stripped) is not None:
            break
        summary_parts.append(stripped)
        index += 1
    summary = " ".join(summary_parts).strip()

    params: dict[str, str] = {}
    while index < total:
        stripped = lines[index].strip()
        if _section_name(stripped) not in _ARGS_HEADERS:
            index += 1
            continue
        header_indent = _indent_of(lines[index])
        index += 1
        block: list[str] = []
        while index < total:
            line = lines[index]
            if line.strip() and _indent_of(line) <= header_indent:
                break
            block.append(line)
            index += 1
        params.update(_parse_args_block(block))
    return summary, params


# ---------------------------------------------------------------------------
# JSON Schema generation
# ---------------------------------------------------------------------------

_NAME_MAPPING_KEYS = frozenset({"properties", "$defs", "definitions", "patternProperties"})
_OPAQUE_KEYS = frozenset({"const", "default", "enum", "examples"})
_REF_PREFIX = "#/$defs/"


class _RecursiveSchema(Exception):
    """Internal signal that a schema contains a ``$ref`` cycle."""


def _resolve_refs(node: Any, defs: Mapping[str, Any], stack: frozenset[str]) -> Any:
    """Recursively replace ``$ref`` pointers with the definitions they name.

    Args:
        node: The schema fragment being rewritten.
        defs: The ``$defs`` mapping the refs point into.
        stack: Definition names currently being inlined, used for cycle detection.

    Returns:
        A copy of ``node`` with every resolvable ``$ref`` inlined.

    Raises:
        _RecursiveSchema: If a reference cycle is detected.
    """
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith(_REF_PREFIX):
            key = ref[len(_REF_PREFIX) :]
            if key in stack:
                raise _RecursiveSchema(key)
            target = defs.get(key)
            if target is not None:
                merged = dict(_resolve_refs(target, defs, stack | {key}))
                for sibling_key, sibling_value in node.items():
                    if sibling_key != "$ref":
                        merged[sibling_key] = _resolve_refs(sibling_value, defs, stack)
                return merged
        return {key: _resolve_refs(value, defs, stack) for key, value in node.items()}
    if isinstance(node, list):
        return [_resolve_refs(item, defs, stack) for item in node]
    return node


def _inline_defs(schema: dict[str, Any]) -> dict[str, Any]:
    """Flatten a pydantic schema by inlining its ``$defs``.

    Args:
        schema: The schema produced by ``model_json_schema()``.

    Returns:
        A flat, self-contained copy. If the schema is recursive it is returned
        unchanged (``$defs`` preserved) because inlining would not terminate.
    """
    defs = schema.get("$defs")
    if not isinstance(defs, dict) or not defs:
        return schema
    body = {key: value for key, value in schema.items() if key != "$defs"}
    try:
        resolved = _resolve_refs(body, defs, frozenset())
    except _RecursiveSchema:
        return schema
    return dict(resolved)


def _normalize(node: Any, *, is_schema: bool = True) -> Any:
    """Drop ``title`` keys and force ``additionalProperties: false`` on objects.

    Args:
        node: The schema fragment being normalized.
        is_schema: ``True`` when ``node`` is itself a schema; ``False`` when it
            is a mapping of names to schemas (e.g. the value of ``properties``),
            where keys are user-controlled and must not be filtered.

    Returns:
        A normalized copy of ``node``.
    """
    if isinstance(node, dict):
        if not is_schema:
            return {key: _normalize(value, is_schema=True) for key, value in node.items()}
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key == "title":
                continue
            if key in _NAME_MAPPING_KEYS:
                out[key] = _normalize(value, is_schema=False)
            elif key in _OPAQUE_KEYS:
                out[key] = value
            else:
                out[key] = _normalize(value, is_schema=True)
        is_closed_object = out.get("type") == "object" and "properties" in out
        if is_closed_object and "additionalProperties" not in out:
            out["additionalProperties"] = False
        return out
    if isinstance(node, list):
        return [_normalize(item, is_schema=True) for item in node]
    return node


_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
"""Charset every provider logpose targets accepts for a tool name.

Anthropic enforces ``^[a-zA-Z0-9_-]{1,128}$`` and rejects anything else with an
opaque 400 on the first live request; validating at decoration time turns that
into an import-time error.
"""


def _check_tool_name(name: str) -> None:
    """Reject a tool name no provider will accept.

    Args:
        name: The resolved tool name.

    Raises:
        ToolSchemaError: If ``name`` is empty, longer than 128 characters, or
            contains anything outside ``[A-Za-z0-9_-]``.
    """
    if _TOOL_NAME_RE.match(name):
        return
    if not name:
        raise ToolSchemaError("Tool name must not be empty.")
    if len(name) > 128:
        raise ToolSchemaError(
            f"Tool name {name!r} is {len(name)} characters; providers accept at most 128."
        )
    illegal = "".join(sorted({char for char in name if not re.match(r"[A-Za-z0-9_-]", char)}))
    raise ToolSchemaError(
        f"Tool name {name!r} contains characters providers reject: {illegal!r}. "
        "Use only letters, digits, underscores, and hyphens."
    )


def _sanitize_model_name(name: str) -> str:
    """Turn a tool name into a valid pydantic model name.

    Args:
        name: The tool name.

    Returns:
        A name safe to hand to ``pydantic.create_model``.
    """
    cleaned = re.sub(r"\W", "_", name).strip("_")
    return f"{cleaned or 'Tool'}Args"


def _dunder_call(func: Callable[..., Any]) -> Any:
    """Return the class-level ``__call__`` of a callable object, if it has one.

    Used to look at the real function behind a callable *instance* (annotations,
    coroutine-ness). Plain functions and methods carry that information directly,
    so callers check those first.

    Args:
        func: The callable to inspect.

    Returns:
        The unbound ``__call__`` attribute, or ``None`` if there is none.
    """
    try:
        return type(func).__call__
    except AttributeError:
        return None


def _is_async_callable(func: Callable[..., Any]) -> bool:
    """Report whether calling ``func`` returns an awaitable.

    Args:
        func: The callable to inspect.

    Returns:
        ``True`` for coroutine functions and for objects whose ``__call__`` is one.
    """
    if inspect.iscoroutinefunction(func):
        return True
    dunder_call = _dunder_call(func)
    return dunder_call is not None and inspect.iscoroutinefunction(dunder_call)


def _resolve_doc(func: Callable[..., Any]) -> str | None:
    """Return the docstring describing ``func``'s call signature.

    For a callable *object*, the signature and annotations come from its
    ``__call__``, so the ``Args:`` block does too — falling back to the object's
    own docstring when ``__call__`` has none.

    Args:
        func: The callable being turned into a tool.

    Returns:
        The docstring to parse, or ``None``.
    """
    doc = inspect.getdoc(func)
    if inspect.isfunction(func) or inspect.ismethod(func):
        return doc
    dunder_call = _dunder_call(func)
    if inspect.isfunction(dunder_call):
        call_doc = inspect.getdoc(dunder_call)
        if call_doc:
            return call_doc
    return doc


def _resolve_hints(func: Callable[..., Any], tool_name: str) -> dict[str, Any]:
    """Resolve a callable's type hints, including string annotations.

    Args:
        func: The callable being turned into a tool.
        tool_name: Name used in error messages.

    Returns:
        The resolved annotation mapping.

    Raises:
        ToolSchemaError: If an annotation cannot be resolved.
    """
    from typing import get_type_hints

    target: Any = func
    if not (inspect.isfunction(func) or inspect.ismethod(func)):
        dunder_call = _dunder_call(func)
        if dunder_call is not None:
            target = dunder_call
    try:
        return dict(get_type_hints(target, include_extras=True))
    except Exception as exc:  # noqa: BLE001 - re-raised as a logpose error
        raise ToolSchemaError(
            f"Could not resolve type annotations for tool {tool_name!r}: {exc}"
        ) from exc


def _build_args_model(
    func: Callable[..., Any], tool_name: str
) -> tuple[type[BaseModel], tuple[str, ...]]:
    """Build the pydantic model that validates a tool's arguments.

    Args:
        func: The callable being turned into a tool.
        tool_name: Name used in error messages and for the model name.

    Returns:
        A ``(model, positional_only_names)`` pair. ``positional_only_names`` are
        the parameters that must be passed positionally when calling the handler.

    Raises:
        ToolSchemaError: If the signature cannot be turned into a JSON Schema.
    """
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError) as exc:
        raise ToolSchemaError(f"Cannot inspect the signature of tool {tool_name!r}: {exc}") from exc

    hints = _resolve_hints(func, tool_name)

    fields: dict[str, Any] = {}
    positional_only: list[str] = []
    for param_name, param in signature.parameters.items():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            raise ToolSchemaError(
                f"Tool {tool_name!r} declares *{param_name}; variadic positional arguments "
                "have no JSON Schema equivalent. Use explicit parameters or a list parameter."
            )
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            raise ToolSchemaError(
                f"Tool {tool_name!r} declares **{param_name}; variadic keyword arguments "
                "have no JSON Schema equivalent. Use explicit parameters or a dict parameter."
            )
        if param_name not in hints:
            raise ToolSchemaError(
                f"Parameter {param_name!r} of tool {tool_name!r} has no type annotation; "
                "every tool parameter must be annotated."
            )
        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional_only.append(param_name)
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param_name] = (hints[param_name], default)

    try:
        model = create_model(
            _sanitize_model_name(tool_name),
            __config__=ConfigDict(extra="forbid"),
            **fields,
        )
    except Exception as exc:  # noqa: BLE001 - re-raised as a logpose error
        raise ToolSchemaError(
            f"Could not build an argument schema for tool {tool_name!r}: {exc}"
        ) from exc

    # pydantic silently swallows parameters whose names it reserves:
    # `model_config` becomes the model's config and a leading underscore makes a
    # private attribute. Either way the parameter vanishes from the schema, the
    # model can never supply it, and every invocation fails with a TypeError at
    # call time. Fail at decoration time instead, as the module contract promises.
    missing = [name for name in fields if name not in model.model_fields]
    if missing:
        names = ", ".join(repr(name) for name in missing)
        raise ToolSchemaError(
            f"Parameter(s) {names} of tool {tool_name!r} cannot be schematized: the "
            "name is reserved by pydantic (names starting with an underscore, and "
            "'model_config', are not usable as fields). Rename the parameter."
        )
    return model, tuple(positional_only)


def _build_input_schema(model: type[BaseModel], param_docs: Mapping[str, str]) -> dict[str, Any]:
    """Render a tool's JSON Schema from its argument model.

    Args:
        model: The pydantic model describing the arguments.
        param_docs: Per-parameter descriptions parsed from the docstring. A
            description is only injected when the property does not already
            carry one (an explicit ``Field(description=...)`` wins).

    Returns:
        The provider-facing schema, always of the form
        ``{"type": "object", "properties": {...}, "required": [...],
        "additionalProperties": false}`` (plus ``$defs`` for recursive models).

    Raises:
        ToolSchemaError: If pydantic cannot produce a JSON Schema.
    """
    try:
        raw = model.model_json_schema()
    except Exception as exc:  # noqa: BLE001 - re-raised as a logpose error
        raise ToolSchemaError(f"Could not generate a JSON Schema: {exc}") from exc

    flattened = _inline_defs(raw)
    normalized = _normalize(flattened)

    properties = normalized.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    required = normalized.get("required")
    required_list = [str(name) for name in required] if isinstance(required, list) else []

    for param_name, description in param_docs.items():
        prop = properties.get(param_name)
        if isinstance(prop, dict) and "description" not in prop:
            prop["description"] = description

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required_list,
        "additionalProperties": False,
    }
    defs = normalized.get("$defs")
    if isinstance(defs, dict) and defs:
        schema["$defs"] = defs
    return schema


# ---------------------------------------------------------------------------
# Result / error coercion
# ---------------------------------------------------------------------------


def _to_text(value: object) -> str:
    """Coerce a handler's return value to the text sent back to the model.

    Args:
        value: Whatever the handler returned.

    Returns:
        ``value`` unchanged if it is already a string, a JSON encoding if it is
        a ``dict`` or ``list``, and ``str(value)`` otherwise.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, default=str)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _format_validation_error(exc: ValidationError) -> str:
    """Render a pydantic validation error as a short, model-readable string.

    Only the field location and message are included; input values are omitted
    so a mis-passed secret is never echoed back.

    Args:
        exc: The validation error raised while coercing tool arguments.

    Returns:
        A ``"field: message"`` summary, one entry per problem.
    """
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", ())) or "<arguments>"
        parts.append(f"{location}: {error.get('msg', 'invalid value')}")
    return "; ".join(parts) or "invalid arguments"


# ---------------------------------------------------------------------------
# ToolDef
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolDef:
    """A consumer-defined tool: schema, description, and handler.

    Build these with the :func:`tool` decorator rather than by hand.

    Attributes:
        name: Name the model uses to call the tool.
        description: What the tool does and when to use it.
        input_schema: JSON Schema describing the tool's arguments.
        handler: The underlying callable. May be sync or async.
        is_async: Whether ``handler`` is a coroutine function.
        args_model: Pydantic model used to validate and coerce incoming
            arguments. ``None`` disables validation and passes arguments through
            unchanged (only useful for hand-built instances).
        positional_only: Parameter names that must be passed positionally when
            calling ``handler``.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]
    is_async: bool = False
    args_model: type[BaseModel] | None = None
    positional_only: tuple[str, ...] = ()

    def spec(self) -> ToolSpec:
        """Return the provider-facing description of this tool.

        Returns:
            A :class:`~logpose.providers.base.ToolSpec` carrying the name,
            description, and input schema. Deliberately drops the handler:
            providers advertise tools, only the loop executes them.
        """
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )

    async def invoke(self, args: dict[str, Any]) -> str:
        """Validate ``args``, run the handler, and return its result as text.

        Sync handlers run on a worker thread via :func:`asyncio.to_thread` so the
        event loop is never blocked; async handlers are awaited directly.

        Args:
            args: The already-parsed arguments from the model's tool-use block.

        Returns:
            The handler's return value coerced to ``str``.

        Raises:
            ToolExecutionError: If the arguments fail validation or the handler
                raises. The original exception is attached as ``__cause__``. The
                loop converts this into an error tool result so the model can
                correct itself.
        """
        call_args, call_kwargs = self._bind(args)
        try:
            if self.is_async:
                result = await self.handler(*call_args, **call_kwargs)
            else:
                result = await asyncio.to_thread(self.handler, *call_args, **call_kwargs)
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(
                f"Tool {self.name!r} failed: {type(exc).__name__}: {exc}"
            ) from exc
        return _to_text(result)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the underlying handler directly.

        Decorating a function does not make it unusable from ordinary Python
        code; ``ToolDef`` forwards calls straight to ``handler``.

        Args:
            *args: Positional arguments for the handler.
            **kwargs: Keyword arguments for the handler.

        Returns:
            Whatever the handler returns (a coroutine for async handlers).
        """
        return self.handler(*args, **kwargs)

    def _bind(self, args: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Validate incoming arguments and split them for the handler call.

        Args:
            args: The raw argument mapping from the model.

        Returns:
            A ``(positional, keyword)`` pair ready to splat into the handler.

        Raises:
            ToolExecutionError: If ``args`` is not a mapping or fails validation.
        """
        if not isinstance(args, Mapping):
            raise ToolExecutionError(
                f"Tool {self.name!r} expected an object of arguments, "
                f"got {type(args).__name__}.",
                safe_to_expose=True,
            )
        if self.args_model is None:
            return (), dict(args)
        try:
            validated = self.args_model.model_validate(dict(args))
        except ValidationError as exc:
            raise ToolExecutionError(
                f"Invalid arguments for tool {self.name!r}: {_format_validation_error(exc)}",
                safe_to_expose=True,
            ) from exc

        values = {name: getattr(validated, name) for name in type(validated).model_fields}
        positional = tuple(values.pop(name) for name in self.positional_only)
        return positional, values


# ---------------------------------------------------------------------------
# The decorator
# ---------------------------------------------------------------------------


def _build_tool(
    func: Callable[..., Any],
    *,
    name: str | None,
    description: str | None,
) -> ToolDef:
    """Turn a callable into a :class:`ToolDef`.

    Args:
        func: The callable to wrap.
        name: Explicit tool name; defaults to ``func.__name__``.
        description: Explicit description; defaults to the docstring summary.

    Returns:
        The built tool definition.

    Raises:
        ToolSchemaError: If ``func`` is not a usable, schematizable callable.
    """
    if isinstance(func, ToolDef):
        raise ToolSchemaError(f"{func.name!r} is already a ToolDef; do not apply @tool twice.")
    if not callable(func):
        raise ToolSchemaError(f"@tool expects a callable, got {type(func).__name__}.")

    tool_name = name or getattr(func, "__name__", None)
    if not tool_name:
        raise ToolSchemaError(
            "Could not determine a tool name; pass one explicitly as @tool(name=...)."
        )
    _check_tool_name(tool_name)

    summary, param_docs = _parse_docstring(_resolve_doc(func))
    model, positional_only = _build_args_model(func, tool_name)
    schema = _build_input_schema(model, param_docs)

    return ToolDef(
        name=tool_name,
        description=description if description is not None else summary,
        input_schema=schema,
        handler=func,
        is_async=_is_async_callable(func),
        args_model=model,
        positional_only=positional_only,
    )


@overload
def tool(func: Callable[..., Any], /) -> ToolDef: ...


@overload
def tool(
    *,
    name: str | None = ...,
    description: str | None = ...,
) -> Callable[[Callable[..., Any]], ToolDef]: ...


def tool(
    func: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
) -> ToolDef | Callable[[Callable[..., Any]], ToolDef]:
    """Turn a function into a :class:`ToolDef`.

    Usable bare or called::

        @tool
        def ping() -> str:
            '''Check liveness.'''
            return "pong"

        @tool(name="weather", description="Look up the weather.")
        def _weather(city: str) -> str:
            return "sunny"

    The argument schema comes from the annotations; the description comes from
    ``description`` if given, otherwise from the docstring summary. Descriptions
    for individual parameters are read from a Google-style ``Args:`` block and
    injected into the property schemas — the model relies on them, so write them.

    Args:
        func: The function to wrap, when used bare.
        name: Override the tool name (defaults to the function's ``__name__``).
        description: Override the description (defaults to the docstring summary).

    Returns:
        A :class:`ToolDef` when used bare, or a decorator returning one when
        called with keyword arguments.

    Raises:
        ToolSchemaError: If the name is not ``[A-Za-z0-9_-]{1,128}``, or the
            signature cannot be schematized — ``*args``, ``**kwargs``, a
            parameter with no annotation, or a parameter whose name pydantic
            reserves (``model_config``, anything leading-underscore).
    """

    def decorate(target: Callable[..., Any]) -> ToolDef:
        return _build_tool(target, name=name, description=description)

    if func is None:
        return decorate
    return decorate(func)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _origin_of(tool_def: ToolDef, fallback: str) -> str:
    """Name the module a tool was defined in.

    Reads the handler's ``__module__`` rather than where the tool was *found*, so
    a re-exported tool is reported against the file that defines it.

    Args:
        tool_def: The tool to locate.
        fallback: Module name to use when the handler has no ``__module__``.

    Returns:
        A module name suitable for an error message.
    """
    origin = getattr(tool_def.handler, "__module__", None)
    return origin if isinstance(origin, str) and origin else fallback


def _import(name: str) -> ModuleType:
    """Import a module by name, reporting failure with the module named.

    Args:
        name: Fully qualified module name.

    Returns:
        The imported module.

    Raises:
        LogposeError: If the module cannot be imported. The original exception is
            chained as ``__cause__``.
    """
    try:
        return importlib.import_module(name)
    except Exception as exc:
        raise LogposeError(
            f"Failed to import {name!r} while discovering tools: {type(exc).__name__}: {exc}"
        ) from exc


def _walk(module: ModuleType, recursive: bool, visited: set[str]) -> Iterator[ModuleType]:
    """Yield ``module`` and, when recursive, every submodule beneath it.

    Uses :func:`pkgutil.iter_modules` rather than :func:`pkgutil.walk_packages`
    for three reasons that matter here: the underscore filter is applied *before*
    the submodule is imported (``walk_packages`` imports subpackages itself, so a
    ``_private`` package would have its top-level code executed anyway), an import
    failure keeps its ``__cause__`` (``walk_packages``'s ``onerror`` receives only
    a name, and silently swallows the error when no handler is given), and the
    traversal order is ours to fix.

    Submodules are visited in sorted name order. That is explicit rather than
    inherited: the default path finder happens to sort, but zip importers and
    third-party path hooks do not.

    Namespace subpackages — a directory with no ``__init__.py`` — are not
    descended into, because ``iter_modules`` does not report them as packages.

    Args:
        module: The module or package to start from.
        recursive: Whether to descend into subpackages.
        visited: Module names already yielded, mutated in place so overlapping
            targets do not scan the same module twice.

    Yields:
        Modules to scan, the parent before its children.

    Raises:
        LogposeError: If a submodule cannot be imported.
    """
    yield module
    path = getattr(module, "__path__", None)
    if not recursive or path is None:
        return

    for info in sorted(pkgutil.iter_modules(list(path)), key=lambda item: item.name):
        # Filter before importing: this is the whole reason for iter_modules.
        # Covers dunders too, so `__main__.py` is never executed by a scan.
        if info.name.startswith("_"):
            continue
        qualified = f"{module.__name__}.{info.name}"
        if qualified in visited:
            continue
        visited.add(qualified)
        yield from _walk(_import(qualified), recursive, visited)


def _caller_namespace() -> tuple[Mapping[str, Any], str]:
    """Return the calling module's globals, for a bare ``discover_tools()``.

    Reads the frame's globals directly rather than looking the module up in
    ``sys.modules``, so this also works in a REPL, under ``python -c``, and
    inside ``exec`` with a custom namespace.

    Returns:
        The caller's global namespace and a label for error messages.

    Raises:
        LogposeError: If frame introspection is unavailable, which is possible on
            non-CPython implementations.
    """
    frame = inspect.currentframe()
    # _caller_namespace -> discover_tools -> the actual caller.
    for _ in range(2):
        frame = frame.f_back if frame is not None else None
    if frame is None:
        raise LogposeError(
            "discover_tools() cannot inspect the calling frame on this Python "
            "implementation; pass an explicit module or module name instead."
        )
    namespace = frame.f_globals
    name = namespace.get("__name__")
    return namespace, name if isinstance(name, str) else "<caller>"


def discover_tools(
    *targets: str | ModuleType,
    recursive: bool = True,
    predicate: Callable[[ToolDef], bool] | None = None,
) -> list[ToolDef]:
    """Collect every :func:`tool`-decorated function in one or more modules.

    ``@tool`` keeps no registry — it simply returns a :class:`ToolDef` — so
    discovery works by importing modules and scanning their namespaces::

        agent = Agent("docker", tools=discover_tools("myapp.tools"))

    The result is an ordinary list, so it composes with hand-written tools::

        tools = [*discover_tools("myapp.tools"), extra_tool]

    A tool re-exported into a package's ``__init__`` is returned **once**: results
    are deduplicated by object identity, so the usual
    ``from .weather import get_weather`` is not mistaken for a duplicate. Two
    *different* tools sharing a name is a genuine conflict and raises.

    **Order is traversal order, not alphabetical**: targets in the order given,
    each module's own namespace in definition order, then its submodules sorted by
    name, depth-first. Appending a tool therefore leaves the preceding request
    bytes untouched, which preserves automatic prefix caching on providers that do
    it; sorting by name would shift every entry after an insertion. Sort the
    result yourself if you want alphabetical.

    **Discovery imports every module under a target**, which executes its
    top-level code. Never pass a target derived from untrusted input, and call
    this at startup rather than inside a running event loop, since importing a
    package tree blocks.

    Only module-level names are found. A tool defined inside a class body or a
    function, or held only in a list or dict, is not discovered, and ``__all__``
    is not consulted. A :class:`ToolDef` imported into a scanned module from a
    third-party library *is* discovered — use ``predicate`` to exclude it.

    Args:
        *targets: Modules to scan, each a dotted module path or an
            already-imported module object. With no targets, the **calling
            module** is scanned; note it sees only names bound above the call, so
            put the call at the bottom of the file.
        recursive: Descend into subpackages when a target is a package.
            Submodules whose name starts with an underscore are skipped without
            being imported. Ignored for a plain module.
        predicate: Optional filter applied as tools are collected. A rejected tool
            is ignored entirely, so this also resolves a name conflict. Exceptions
            it raises propagate unchanged.

    Returns:
        The discovered tools in traversal order. Empty when the targets imported
        cleanly but defined no tools.

    Raises:
        LogposeError: If a target or submodule cannot be imported, or if two
            different tools share a name.
    """
    found: dict[int, ToolDef] = {}
    claimed: dict[str, tuple[ToolDef, str]] = {}
    visited: set[str] = set()

    def collect(namespace: Mapping[str, Any], where: str) -> None:
        for value in namespace.values():
            # Keyed by id() because ToolDef is an unhashable frozen dataclass
            # (input_schema is a dict). Safe because `found` holds a strong
            # reference to every object it has keyed, so no id can be reused.
            if not isinstance(value, ToolDef) or id(value) in found:
                continue
            if predicate is not None and not predicate(value):
                continue
            origin = _origin_of(value, where)
            previous = claimed.get(value.name)
            if previous is not None:
                if previous[0].handler is value.handler:
                    # The same function decorated twice: two ToolDef objects, but
                    # unambiguously the same tool. Keep the first.
                    continue
                raise LogposeError(
                    f"Duplicate tool name {value.name!r} found while discovering tools:\n"
                    f"  {previous[1]}\n"
                    f"  {origin}\n"
                    "Rename one of them, or exclude it with predicate=."
                )
            found[id(value)] = value
            claimed[value.name] = (value, origin)

    if not targets:
        namespace, label = _caller_namespace()
        collect(namespace, label)

    for target in targets:
        module = _import(target) if isinstance(target, str) else target
        if module.__name__ in visited:
            continue
        visited.add(module.__name__)
        for scanned in _walk(module, recursive, visited):
            collect(vars(scanned), scanned.__name__)

    return list(found.values())
