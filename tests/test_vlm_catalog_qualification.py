# SPDX-License-Identifier: Apache-2.0
"""Offline qualification of catalog architecture metadata and loader coverage."""

import ast
import importlib.util
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from rapid_mlx.model_aliases import _coerce, list_profiles
from rapid_mlx.utils.model_file_guard import (
    EXTERNAL_TOKENIZER_VLM_MODEL_TYPES,
    FORCED_TRUST_VLM_MODEL_TYPES,
)

UNQUALIFIED_IMAGE_ALIASES = {
    "qwen3.8-27b-abliterated-4bit": "gated repository; pinned metadata not publicly "
    "readable; the load-boundary processor guard checks its config at load",
}


def _dependency_registry():
    spec = importlib.util.find_spec("mlx_vlm")
    if spec is None:
        return None, {}
    root = Path(next(iter(spec.submodule_search_locations)))
    # Read the literal registry without importing the tensor runtime.
    tree = ast.parse((root / "utils.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "MODEL_REMAPPING"
            for target in node.targets
        ):
            return root, ast.literal_eval(node.value)
    pytest.fail("Installed dependency no longer declares a literal MODEL_REMAPPING")


def _normalized_types(raw, remapping):
    normalized = raw.lower().replace("-", "_")
    return {
        normalized,
        remapping.get(raw, raw).lower().replace("-", "_"),
        remapping.get(normalized, normalized).lower().replace("-", "_"),
    }


def test_image_capable_aliases_declare_model_type():
    profiles = list_profiles()
    missing = {
        alias
        for alias, profile in profiles.items()
        if profile.supports_image_input and not profile.model_type
    }
    assert missing == UNQUALIFIED_IMAGE_ALIASES.keys()
    for alias in UNQUALIFIED_IMAGE_ALIASES:
        assert alias in profiles
        assert profiles[alias].supports_image_input
        assert profiles[alias].model_type is None


def test_shared_pins_agree_on_model_type():
    pins = {}
    for alias, profile in list_profiles().items():
        if profile.model_type is None:
            continue
        pin = (profile.hf_path, profile.revision, profile.subfolder)
        previous = pins.setdefault(pin, (alias, profile.model_type))
        assert previous[1] == profile.model_type, (previous, alias, pin)


def test_declared_types_avoid_unreviewed_processor_routes():
    _, remapping = _dependency_registry()
    guarded = FORCED_TRUST_VLM_MODEL_TYPES | EXTERNAL_TOKENIZER_VLM_MODEL_TYPES
    for alias, profile in list_profiles().items():
        if profile.model_type:
            assert not (_normalized_types(profile.model_type, remapping) & guarded), (
                alias
            )


def test_declared_types_have_a_loader():
    root, remapping = _dependency_registry()
    if root is None:
        pytest.skip("VLM dependency is not installed")
    # Reviewed dispatch: models/media_backend.py, models/mllm.py, and
    # spec_decode/mtp/gemma4_inject.py (_ASSISTANT_MODEL_TYPES).
    native = {
        "lfm3_vl",
        "minicpmv4_7",
        "prism_hadamard_qwen35",
        "gemma4_assistant",
        "gemma4_unified_assistant",
    }
    installed = {path.name for path in (root / "models").iterdir() if path.is_dir()}
    for alias, profile in list_profiles().items():
        if profile.model_type:
            types = _normalized_types(profile.model_type, remapping)
            assert types & (installed | native), (alias, profile.model_type)


@pytest.mark.parametrize("model_type", [None, "", "  ", 1, True, [], {}])
def test_catalog_rejects_invalid_model_type(model_type):
    with pytest.raises(ValueError, match="model_type.*non-empty string"):
        _coerce("test", {"hf_path": "owner/model", "model_type": model_type})


def test_catalog_optional_model_type():
    assert _coerce("test", {"hf_path": "owner/model"}).model_type is None
    assert _coerce("test", "owner/model").model_type is None
    assert (
        _coerce("test", {"hf_path": "owner/model", "model_type": "example"}).model_type
        == "example"
    )


@pytest.mark.requires_network
@pytest.mark.skipif(
    os.environ.get("RAPID_MLX_ONLINE_CATALOG_CHECK") != "1",
    reason="Set RAPID_MLX_ONLINE_CATALOG_CHECK=1 to check pinned Hub metadata",
)
def test_online_pinned_model_types_match():
    headers = {}
    try:
        from huggingface_hub import get_token
    except ImportError:
        pass
    else:
        if token := get_token():
            headers["Authorization"] = f"Bearer {token}"
    failures = []
    for alias, profile in list_profiles().items():
        if not profile.supports_image_input or not profile.model_type:
            continue
        subfolder = f"{profile.subfolder}/" if profile.subfolder else ""
        path = urllib.parse.quote(
            f"{profile.hf_path}/resolve/{profile.revision}/{subfolder}config.json",
            safe="/",
        )
        request = urllib.request.Request(
            f"https://huggingface.co/{path}", headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                actual = json.load(response).get("model_type")
            if actual != profile.model_type:
                failures.append(
                    f"{alias}: declared {profile.model_type!r}, got {actual!r}"
                )
        except (OSError, ValueError) as exc:
            failures.append(f"{alias}: metadata unavailable ({type(exc).__name__})")
    assert not failures, "\n".join(failures)


def test_catalog_drafters_are_pinned():
    import re

    from rapid_mlx.model_aliases import resolve_revision
    from rapid_mlx.speculative.native_mtp.eligibility import _QUALIFIED_PAIRS

    failures = []
    for alias, profile in list_profiles().items():
        for field in (
            "mtp_draft_model",
            "native_mtp_draft_model",
            "ddtree_draft_model",
            "dflash_draft_model",
        ):
            repo = getattr(profile, field)
            if not repo:
                continue
            revisions = [resolve_revision(repo)]
            if field in {"ddtree_draft_model", "dflash_draft_model"}:
                revisions.append(getattr(profile, field.replace("_model", "_revision")))
            if pair := _QUALIFIED_PAIRS.get(alias):
                if pair.target_repo == profile.hf_path and pair.drafter_repo == repo:
                    revisions.append(pair.drafter_revision)
            if not any(
                isinstance(pin, str) and re.fullmatch(r"[0-9a-f]{40}", pin)
                for pin in revisions
            ):
                failures.append((alias, field, repo))
    assert not failures, f"Unpinned catalog drafters: {failures}"


@pytest.mark.parametrize("size", ["4", "9"])
def test_qwen35_mtp_entries_are_draft_only(size):
    from rapid_mlx.community_bench.workspace import benchmark_catalog
    from rapid_mlx.model_aliases import draft_only_conflict

    profiles = list_profiles()
    alias = f"qwen3.5-{size}b-mtp-4bit"
    profile = profiles[alias]
    parent_alias = f"qwen3.5-{size}b-4bit"
    parent = profiles[parent_alias]
    assert profile.hf_path == parent.mtp_draft_model
    assert not profile.supports_image_input
    assert not profile.supports_spec_decode
    assert alias not in {
        entry["alias"] for entry in benchmark_catalog(memory_gib=256)["models"]
    }
    for field in (
        "tool_call_parser",
        "reasoning_parser",
        "is_hybrid",
        "is_hybrid_explicit",
    ):
        assert getattr(profile, field) == getattr(parent, field)
    for ref in (alias, profile.hf_path):
        conflict = draft_only_conflict(ref)
        assert conflict is not None
        assert parent_alias in conflict
