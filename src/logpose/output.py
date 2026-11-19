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
from typing import Annotated, Any, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, RootModel, TypeAdapter, ValidationError

from logpose.errors import LogposeError

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


_DATA_KEYS = frozenset({"enum", "const", "default", "examples"})
_REF_PREFIX = "#/$defs/"


def _normalize(node: Any, defs: dict[str, Any], inlining: tuple[str, ...] = ()) -> Any:
    """Return ``node`` rewritten into shapes OpenAI strict mode accepts.

    * A ``$ref`` with sibling keys is replaced by the referenced definition merged
      with those siblings (strict mode rejects ``$ref`` siblings). On a recursive
      cycle the siblings are dropped and the plain ``$ref`` is kept.
    * ``oneOf`` becomes ``anyOf`` and ``discriminator`` is dropped (strict mode only
      allows ``anyOf``). Parsing validates against the original model, so the
      discriminator is still enforced on the way back.
    """
    if isinstance(node, list):
        return [_normalize(item, defs, inlining) for item in node]
    if not isinstance(node, dict):
        return node
    ref = node.get("$ref")
    if isinstance(ref, str) and len(node) > 1 and "$defs" not in node:
        name = ref[len(_REF_PREFIX) :] if ref.startswith(_REF_PREFIX) else None
        if name is None or name not in defs or name in inlining:
            return {"$ref": ref}
        siblings = {k: v for k, v in node.items() if k != "$ref"}
        merged = {**copy.deepcopy(defs[name]), **siblings}
        return _normalize(merged, defs, (*inlining, name))
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "discriminator":
            continue
        if key in _DATA_KEYS:
            out[key] = value
        elif key in ("properties", "$defs", "definitions") and isinstance(value, dict):
            out[key] = {k: _normalize(v, defs, inlining) for k, v in value.items()}
        else:
            out[key] = _normalize(value, defs, inlining)
    if "oneOf" in out and "anyOf" not in out:
        out["anyOf"] = out.pop("oneOf")
    return out


def _open_map_error(path: str) -> LogposeError:
    return LogposeError(
        f"output_type field {path!r} is an open-ended mapping (e.g. dict[str, ...]), which "
        "strict json_schema structured output cannot express; use a nested model with "
        "named fields instead."
    )


def _strictify(schema: Any, path: str) -> None:
    """Mutate ``schema`` in place so every object is strict.

    Raises:
        LogposeError: If an object allows arbitrary keys (an open ``dict``).
    """
    if not isinstance(schema, dict):
        return
    props = schema.get("properties")
    if schema.get("type") == "object" or isinstance(props, dict):
        extra = schema.get("additionalProperties")
        if extra is True or isinstance(extra, dict) or not isinstance(props, dict):
            raise _open_map_error(path)
        schema["additionalProperties"] = False
        schema["required"] = list(props)
        for name, prop in list(props.items()):
            if isinstance(prop, dict) and "default" in prop:
                fixed = "const" in prop and prop["const"] == prop["default"]
                prop = {k: v for k, v in prop.items() if k != "default"}
                # A defaulted single-value Literal (e.g. a discriminator tag) stays
                # required: its only value is the default, and the tag must be present.
                if not fixed and not _is_nullable_schema(prop):
                    prop = {"anyOf": [prop, dict(_NULL)]}
                props[name] = prop
        for name, prop in props.items():
            _strictify(prop, f"{path}.{name}" if path else name)
    for key in ("$defs", "definitions"):
        for def_name, sub in schema.get(key, {}).items():
            _strictify(sub, def_name)
    for key in ("anyOf", "oneOf", "allOf"):
        for sub in schema.get(key, []):
            _strictify(sub, path)
    for sub in schema.get("prefixItems", []):
        _strictify(sub, f"{path}[]")
    _strictify(schema.get("items"), f"{path}[]")


def _not_object_error(model: type[BaseModel]) -> LogposeError:
    return LogposeError(
        f"output_type {model.__name__} must be an object model (a BaseModel with named "
        "fields); strict json_schema structured output requires an object at the root."
    )


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return ``model``'s JSON schema converted to OpenAI strict form.

    Every object gets ``additionalProperties: false`` and lists all properties
    as required. Defaulted fields become nullable and lose their ``default``.
    ``$ref`` with sibling keys is inlined, and ``oneOf`` becomes ``anyOf``.

    Args:
        model: The pydantic model describing the answer.

    Returns:
        A new schema dict; ``model``'s own schema is not mutated.

    Raises:
        LogposeError: If ``model`` is not an object model (e.g. a ``RootModel``)
            or has an open-ended ``dict`` field, which strict mode cannot express.
            The message names the offending field path.
    """
    if issubclass(model, RootModel):
        raise _not_object_error(model)
    raw = copy.deepcopy(model.model_json_schema())
    defs = raw.get("$defs", {})
    root_ref = raw.get("$ref")
    if isinstance(root_ref, str) and root_ref.startswith(_REF_PREFIX):
        # A self-referencing model's root is a $ref into $defs; hoist it.
        target = defs.get(root_ref[len(_REF_PREFIX) :])
        if isinstance(target, dict):
            raw = {**copy.deepcopy(target), **{k: v for k, v in raw.items() if k != "$ref"}}
    if raw.get("type") != "object":
        raise _not_object_error(model)
    schema = _normalize(raw, defs)
    _strictify(schema, "")
    if not isinstance(schema, dict):  # pragma: no cover - _normalize keeps dicts
        raise _not_object_error(model)
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


def _unwrap_annotated(annotation: Any) -> Any:
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation


def _accepts_none(annotation: Any) -> bool:
    annotation = _unwrap_annotated(annotation)
    if annotation is None or annotation is type(None) or annotation is Any:
        return True
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        return any(_accepts_none(a) for a in get_args(annotation))
    return False


def _strip_value(annotation: Any, value: Any) -> Any:
    """Strip disallowed nulls inside ``value`` according to ``annotation``."""
    annotation = _unwrap_annotated(annotation)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return strip_disallowed_nulls(annotation, value)
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Union or origin is types.UnionType:
        for arg in args:
            if arg is type(None):
                continue
            stripped = _strip_value(arg, value)
            try:
                TypeAdapter(arg).validate_python(stripped)
            except ValidationError:
                continue
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
