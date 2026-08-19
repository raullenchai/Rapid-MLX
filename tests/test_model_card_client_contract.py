# SPDX-License-Identifier: Apache-2.0
"""``/v1/models`` vendor extensions are a contract with real clients.

``ModelInfo``'s trailing fields exist so a client can ask the SERVER what
it is talking to instead of making the operator retype it into a config
file. Two consumers depend on that today:

* **In-tree** — ``vllm_mlx.agents.adapter.fetch_reasoning_support`` reads
  ``reasoning_parser`` to decide whether ``agents dsh --setup`` should
  advertise graded reasoning to DeepSeek Harness (#1984).
* **Out-of-tree** — the native Rapid-MLX provider for DSH
  (``raullenchai/rapid-mlx-dsh-provider``) reads ``context_window``,
  ``max_model_len``, ``reasoning_parser`` and ``capabilities`` to answer
  Harness's ``resolveModel()``. The capacity it reports feeds
  ``dsh-compaction-basic``, which multiplies it by its threshold ratio to
  decide when to compact a session; the provider prefers ``max_model_len``
  (the memory-fitted ceiling) over ``context_window`` (the native window)
  when present, so both are part of the contract.

The out-of-tree consumer is the reason this file exists. Renaming or
dropping one of these fields is a silent break for it: nothing in this
repo imports it, its CI does not run here, and the failure surfaces as
compaction firing at the wrong time in somebody's long session. The
fields are owned here, so the guard belongs here.

This is deliberately NOT a test of the values — aliases come and go and
that is not a contract. It pins the wire SHAPE: the names, the
nullability, and the one serialization behaviour the three-state logic
rests on.
"""

from __future__ import annotations

import json

import pytest

from vllm_mlx.api.models import ModelInfo

#: Field -> why an external client needs it. Deleting a row here is a
#: deliberate act that should be visible in review, which is the point.
CLIENT_FIELDS: dict[str, str] = {
    "id": "route identity; the model id sent back on a request",
    "context_window": "native window — the model's own max context",
    "max_model_len": "memory-fitted capacity — the preferred DSH compaction basis",
    "reasoning_parser": "whether the model can emit reasoning at all",
    "tool_call_parser": "whether it can emit OpenAI-shape tool_calls",
    "capabilities": "text / tools / vision, for request shaping",
    "recommended_sampling": "per-model sampling a client should adopt",
}

#: Fields whose ``None`` must reach the wire as an explicit ``null``.
#: See ``test_null_extensions_are_serialized_not_dropped`` for why.
TRISTATE_FIELDS = (
    "reasoning_parser",
    "tool_call_parser",
    "context_window",
    "max_model_len",
)


@pytest.mark.parametrize("field", sorted(CLIENT_FIELDS))
def test_model_card_still_carries_the_field(field: str) -> None:
    assert field in ModelInfo.model_fields, (
        f"ModelInfo lost {field!r} ({CLIENT_FIELDS[field]}). This is a wire "
        f"break for out-of-tree clients — see this module's docstring. If the "
        f"rename is intended, update raullenchai/rapid-mlx-dsh-provider in the "
        f"same change and edit CLIENT_FIELDS here."
    )


@pytest.mark.parametrize("field", TRISTATE_FIELDS)
def test_tristate_fields_accept_none(field: str) -> None:
    """``None`` must be a legal value, not just an unset default.

    A client distinguishes "this model has no reasoning parser" from "this
    server is too old to say" by whether the key is present-and-null or
    absent. That only works if ``None`` is expressible.
    """
    assert ModelInfo.model_fields[field].default is None, (
        f"{field!r} no longer defaults to None; the present-null vs absent "
        f"distinction that fetch_reasoning_support relies on is gone"
    )
    card = ModelInfo(id="m", **{field: None})
    assert getattr(card, field) is None


def test_null_extensions_are_serialized_not_dropped() -> None:
    """The single load-bearing serialization behaviour.

    ``ModelInfo`` deliberately does not set ``exclude_none`` (see its
    docstring). That is what lets a reasoning-less model send
    ``"reasoning_parser": null`` while a server too old to know the field
    omits the key entirely — the two states a client must tell apart.

    If someone adds ``exclude_none`` to shrink the payload, every such
    client silently collapses to "unknown" and stops downgrading models
    that genuinely have no reasoning parser. That is a behaviour change
    with no error attached to it, which is exactly the kind this test is
    here to make loud.
    """
    payload = json.loads(ModelInfo(id="m").model_dump_json())
    for field in TRISTATE_FIELDS:
        assert field in payload, (
            f"{field!r} vanished from the serialized model card. Something "
            f"turned on exclude_none (or equivalent); clients can no longer "
            f"distinguish 'explicitly none' from 'server too old to say'."
        )
        assert payload[field] is None


def test_capabilities_is_a_list_not_optional() -> None:
    """``capabilities`` is absence-as-empty, unlike the tri-state fields.

    A client reads it as a positive capability list, so an unknown model
    must yield ``[]`` rather than ``None`` — otherwise every consumer
    needs a null check that the type does not advertise.
    """
    assert ModelInfo(id="m").capabilities == []
    payload = json.loads(ModelInfo(id="m").model_dump_json())
    assert payload["capabilities"] == []


def test_contract_fields_are_documented_here() -> None:
    """Every tri-state field must also be in CLIENT_FIELDS.

    Cheap guard against the two lists drifting apart as fields are added.
    """
    assert set(TRISTATE_FIELDS) <= set(CLIENT_FIELDS)
