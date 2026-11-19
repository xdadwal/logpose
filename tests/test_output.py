"""Tests for logpose.output: strict schema, response_format and typed parsing."""

from __future__ import annotations

from typing import Annotated, Any, Literal

import pytest
from pydantic import BaseModel, Field, RootModel, create_model

from logpose.errors import LogposeError
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


class A(BaseModel):
    a: str


class B(BaseModel):
    b: str
    n: int = 5


class Holder(BaseModel):
    x: A | B


class Registry(BaseModel):
    by_name: dict[str, Item]


class Batch(BaseModel):
    rows: tuple[Item, ...]


class Aliased(BaseModel):
    qty: int = Field(default=7, alias="Qty")


class Wrapper(BaseModel):
    inner: Item | None = None


def test_union_of_models_picks_the_fitting_arm() -> None:
    out = parse_output(Holder, '{"x":{"b":"q","n":null}}')
    assert out.x == B(b="q", n=5)


def test_dict_of_models_strips_nested_nulls() -> None:
    out = parse_output(Registry, '{"by_name":{"k":{"name":"x","qty":null}}}')
    assert out.by_name["k"].qty == 1


def test_tuple_of_models_strips_nested_nulls() -> None:
    out = parse_output(Batch, '{"rows":[{"name":"x","qty":null}]}')
    assert out.rows[0].qty == 1


def test_aliased_defaulted_field_null_takes_default() -> None:
    assert parse_output(Aliased, '{"Qty":null}').qty == 7


def test_optional_model_field_null_and_nested_null() -> None:
    assert parse_output(Wrapper, '{"inner":null}').inner is None
    out = parse_output(Wrapper, '{"inner":{"name":"x","qty":null}}')
    assert out.inner == Item(name="x", qty=1)


# ---------------------------------------------------------------------------
# strict-mode limits: open maps, non-object roots, refs with siblings, oneOf
# ---------------------------------------------------------------------------


class WithMeta(BaseModel):
    name: str
    meta: dict[str, int]


class AnyMap(BaseModel):
    extra: dict[str, Any]


class InnerMap(BaseModel):
    tags: dict[str, str]


class OuterMap(BaseModel):
    inner: list[InnerMap]


@pytest.mark.parametrize(
    ("model", "path"),
    [(WithMeta, "meta"), (AnyMap, "extra"), (OuterMap, "tags")],
)
def test_open_dict_field_is_rejected(model: type[BaseModel], path: str) -> None:
    with pytest.raises(LogposeError, match=path) as excinfo:
        strict_json_schema(model)
    assert "dict" in str(excinfo.value)
    with pytest.raises(LogposeError):
        response_format_for(model)


@pytest.mark.parametrize("model", [RootModel[list[Item]], RootModel[Item], RootModel[int]])
def test_non_object_root_is_rejected(model: type[BaseModel]) -> None:
    with pytest.raises(LogposeError, match="object model"):
        strict_json_schema(model)


class Sub(BaseModel):
    a: int
    b: int = 2


class Described(BaseModel):
    sub: Sub = Field(description="the sub")
    titled: Sub = Field(title="Titled")
    plain: Sub
    maybe: Sub | None = Field(default=None, description="optional sub")


class Cat(BaseModel):
    kind: Literal["cat"] = "cat"
    lives: int = 9


class Dog(BaseModel):
    kind: Literal["dog"]
    good: bool = True


class Pets(BaseModel):
    pet: Cat | Dog = Field(discriminator="kind")
    more: list[Annotated[Cat | Dog, Field(discriminator="kind")]]


def _walk(node: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(node, dict):
        out.append(node)
        for value in node.values():
            out.extend(_walk(value))
    elif isinstance(node, list):
        for value in node:
            out.extend(_walk(value))
    return out


def _assert_strict_compatible(schema: dict[str, Any]) -> None:
    for node in _walk(schema):
        if "$ref" in node:
            assert set(node) == {"$ref"}, node
        assert "oneOf" not in node, node
        assert "discriminator" not in node, node


def test_ref_with_siblings_is_inlined() -> None:
    schema = strict_json_schema(Described)
    _assert_strict_compatible(schema)
    props = schema["properties"]
    assert props["plain"] == {"$ref": "#/$defs/Sub"}
    assert props["sub"]["description"] == "the sub"
    assert props["sub"]["type"] == "object"
    assert props["sub"]["additionalProperties"] is False
    assert props["sub"]["required"] == ["a", "b"]
    assert "default" not in props["sub"]["properties"]["b"]


def test_ref_with_siblings_round_trip() -> None:
    text = (
        '{"sub":{"a":1,"b":null},"titled":{"a":2,"b":3},'
        '"plain":{"a":4,"b":null},"maybe":null}'
    )
    out = parse_output(Described, text)
    assert out.sub == Sub(a=1, b=2)
    assert out.titled == Sub(a=2, b=3)
    assert out.plain == Sub(a=4)
    assert out.maybe is None


def test_discriminated_union_uses_any_of() -> None:
    schema = strict_json_schema(Pets)
    _assert_strict_compatible(schema)
    assert len(schema["properties"]["pet"]["anyOf"]) == 2
    assert len(schema["properties"]["more"]["items"]["anyOf"]) == 2
    # The defaulted tag stays a required, non-nullable const.
    assert schema["$defs"]["Cat"]["properties"]["kind"] == {
        "const": "cat",
        "title": "Kind",
        "type": "string",
    }


def test_discriminated_union_round_trip() -> None:
    text = (
        '{"pet":{"kind":"dog","good":null},'
        '"more":[{"kind":"cat","lives":null},{"kind":"dog","good":false}]}'
    )
    out = parse_output(Pets, text)
    assert out.pet == Dog(kind="dog", good=True)
    assert out.more == [Cat(), Dog(kind="dog", good=False)]


def test_recursive_ref_with_siblings_terminates() -> None:
    schema = strict_json_schema(Node)
    _assert_strict_compatible(schema)


class Node(BaseModel):
    value: int
    child: Node | None = Field(default=None, description="next node")
    kids: list[Node] = Field(default_factory=list, description="children")
