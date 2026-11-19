"""Tests for logpose.output: strict schema, response_format and typed parsing."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, create_model

from logpose.output import (
    OutputParseFailure,
    format_errors,
    parse_output,
    response_format_for,
    schema_name,
    strict_json_schema,
)


class Item(BaseModel):
    name: str
    qty: int = 1


class Order(BaseModel):
    customer: str
    items: list[Item]
    note: str | None = None
    priority: int = 3


def _objects(schema: dict) -> list[dict]:  # type: ignore[type-arg]
    found = [s for s in schema.get("$defs", {}).values() if s.get("type") == "object"]
    if schema.get("type") == "object":
        found.append(schema)
    return found


def test_strict_schema_requires_every_property_and_forbids_extras() -> None:
    schema = strict_json_schema(Order)
    objects = _objects(schema)
    assert len(objects) == 2
    for obj in objects:
        assert set(obj["required"]) == set(obj["properties"])
        assert obj["additionalProperties"] is False


def test_defaulted_fields_become_nullable_without_default() -> None:
    schema = strict_json_schema(Order)
    for prop in (schema["properties"]["priority"], schema["$defs"]["Item"]["properties"]["qty"]):
        assert "default" not in prop
        assert {"type": "null"} in prop["anyOf"]
    assert {"type": "null"} in schema["properties"]["note"]["anyOf"]
    assert schema["required"].count("note") == 1


def test_response_format_shape() -> None:
    assert response_format_for(Order) == {
        "type": "json_schema",
        "json_schema": {"name": "Order", "schema": strict_json_schema(Order), "strict": True},
    }


def test_schema_name_is_sanitized() -> None:
    assert schema_name(create_model("My Model!")) == "My_Model_"


def test_null_for_defaulted_field_takes_the_default() -> None:
    out = parse_output(
        Order,
        '{"customer":"a","items":[{"name":"x","qty":null}],"note":null,"priority":null}',
    )
    assert out.priority == 3
    assert out.items[0].qty == 1
    assert out.note is None


def test_invalid_json_is_a_parse_failure() -> None:
    with pytest.raises(OutputParseFailure) as exc:
        parse_output(Order, "not json")
    assert exc.value.errors[0]["type"] == "json_invalid"


def test_validation_errors_never_echo_input() -> None:
    with pytest.raises(OutputParseFailure) as exc:
        parse_output(Order, '{"customer": 123456789, "items": "secret-value"}')
    errors = exc.value.errors
    for text in (str(errors), format_errors(errors)):
        assert "secret-value" not in text
        assert "123456789" not in text
    assert all(set(e) == {"loc", "msg", "type"} for e in errors)


def test_format_errors_lists_paths() -> None:
    with pytest.raises(OutputParseFailure) as exc:
        parse_output(Order, '{"customer": "a", "items": [{"qty": 2}]}')
    assert "items.0.name: " in format_errors(exc.value.errors)
