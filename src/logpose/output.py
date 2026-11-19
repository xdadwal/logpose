"""Structured-output helpers: strict JSON schema, ``response_format`` and parsing.

Pure functions, no I/O. A pydantic model is converted into an OpenAI strict
``json_schema`` ``response_format``, and the model's reply is parsed back into
the typed object.

Security note: error details never echo model-provided input values; only field
paths, messages and error types are kept.
"""

from __future__ import annotations

import copy
import json
import re
import types
from typing import Any, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError

__all__ = [
    "OutputParseFailure",
    "format_errors",
    "parse_output",
    "response_format_for",
    "schema_name",
    "strip_disallowed_nulls",
    "strict_json_schema",
]

T = TypeVar("T", bound=BaseModel)

_NULL: dict[str, Any] = {"type": "null"}


class OutputParseFailure(Exception):
    """Raised when model output is not valid JSON or fails schema validation.

    Attributes:
        errors: Sanitized error items, each with only ``loc`` (tuple), ``msg``
            (str) and ``type`` (str). Never contains the offending input.
    """

    def __init__(self, errors: list[dict[str, Any]]) -> None:
        super().__init__(format_errors(errors))
        self.errors = errors


def _is_nullable_schema(schema: dict[str, Any]) -> bool:
    if schema.get("type") == "null":
        return True
    return any(isinstance(s, dict) and s.get("type") == "null" for s in schema.get("anyOf", []))


def _strictify(schema: Any) -> None:
    """Mutate ``schema`` in place so every object is strict."""
    if not isinstance(schema, dict):
        return
    props = schema.get("properties")
    if schema.get("type") == "object" or isinstance(props, dict):
        props = props if isinstance(props, dict) else {}
        schema["properties"] = props
        schema["additionalProperties"] = False
        schema["required"] = list(props)
        for name, prop in list(props.items()):
            if isinstance(prop, dict) and "default" in prop:
                prop = {k: v for k, v in prop.items() if k != "default"}
                if not _is_nullable_schema(prop):
                    prop = {"anyOf": [prop, dict(_NULL)]}
                props[name] = prop
        for prop in props.values():
            _strictify(prop)
    for key in ("$defs", "definitions"):
        for sub in schema.get(key, {}).values():
            _strictify(sub)
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        for sub in schema.get(key, []):
            _strictify(sub)
    _strictify(schema.get("items"))


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return ``model``'s JSON schema converted to OpenAI strict form.

    Every object gets ``additionalProperties: false`` and lists all properties
    as required. Defaulted fields become nullable and lose their ``default``.

    Args:
        model: The pydantic model describing the answer.

    Returns:
        A new schema dict; ``model``'s own schema is not mutated.
    """
    schema = copy.deepcopy(model.model_json_schema())
    _strictify(schema)
    return schema


def schema_name(model: type[BaseModel]) -> str:
    """Return a ``response_format`` schema name derived from ``model.__name__``."""
    name = re.sub(r"[^A-Za-z0-9_-]", "_", model.__name__)[:64]
    return name or "output"


def response_format_for(model: type[BaseModel]) -> dict[str, Any]:
    """Build the strict ``json_schema`` ``response_format`` for ``model``."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name(model),
            "schema": strict_json_schema(model),
            "strict": True,
        },
    }


def _accepts_none(annotation: Any) -> bool:
    if annotation is None or annotation is type(None) or annotation is Any:
        return True
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        return any(_accepts_none(a) for a in get_args(annotation))
    return False


def _strip_value(annotation: Any, value: Any) -> Any:
    """Strip disallowed nulls inside ``value`` according to ``annotation``."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return strip_disallowed_nulls(annotation, value)
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Union or origin is types.UnionType:
        for arg in args:
            if arg is type(None):
                continue
            stripped = _strip_value(arg, value)
            if stripped is not value:
                return stripped
        return value
    if origin in (list, set, frozenset) and isinstance(value, list) and args:
        return [_strip_value(args[0], v) for v in value]
    if origin is tuple and isinstance(value, list) and args:
        if len(args) == 2 and args[1] is Ellipsis:
            return [_strip_value(args[0], v) for v in value]
        if len(args) == len(value):
            return [_strip_value(a, v) for a, v in zip(args, value, strict=True)]
        return value
    if origin is dict and isinstance(value, dict) and len(args) == 2:
        return {k: _strip_value(args[1], v) for k, v in value.items()}
    return value


def strip_disallowed_nulls(model: type[BaseModel], data: Any) -> Any:
    """Remove ``None`` values for fields whose annotation does not accept ``None``.

    Strict schemas force the model to emit ``null`` for defaulted fields; dropping
    them lets pydantic apply the declared default. Recurses through nested models,
    list/tuple items and dict values.

    Args:
        model: The model ``data`` is meant to validate against.
        data: Decoded JSON; returned unchanged if it is not an object.

    Returns:
        A new structure with the disallowed nulls removed.
    """
    if not isinstance(data, dict):
        return data
    by_key = {(f.alias or name): f.annotation for name, f in model.model_fields.items()}
    by_key.update({name: f.annotation for name, f in model.model_fields.items() if not f.alias})
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key not in by_key:
            out[key] = value
            continue
        annotation = by_key[key]
        if value is None:
            if _accepts_none(annotation):
                out[key] = None
            continue
        out[key] = _strip_value(annotation, value)
    return out


def _sanitize(exc: ValidationError) -> list[dict[str, Any]]:
    return [
        {"loc": tuple(e["loc"]), "msg": str(e["msg"]), "type": str(e["type"])}
        for e in exc.errors(include_url=False, include_context=False, include_input=False)
    ]


def parse_output(model: type[T], text: str) -> T:
    """Parse ``text`` as JSON and validate it into ``model``.

    Args:
        model: Target pydantic model.
        text: The model's raw reply.

    Returns:
        The validated instance.

    Raises:
        OutputParseFailure: If ``text`` is not JSON or fails validation. The
            errors never contain the offending input values.
    """
    try:
        data = json.loads(text)
    except ValueError:
        raise OutputParseFailure(
            [{"loc": (), "msg": "Invalid JSON", "type": "json_invalid"}]
        ) from None
    try:
        return model.model_validate(strip_disallowed_nulls(model, data))
    except ValidationError as exc:
        raise OutputParseFailure(_sanitize(exc)) from None


def format_errors(errors: list[dict[str, Any]]) -> str:
    """Render errors as ``"<dotted loc or '(root)'>: <msg>"`` lines."""
    lines = []
    for err in errors:
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        lines.append(f"{loc}: {err['msg']}")
    return "\n".join(lines)
