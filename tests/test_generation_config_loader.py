# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm_mlx.utils.generation_config.load_generation_config_sampling."""

from __future__ import annotations

import json
import os

import pytest

from vllm_mlx.utils.generation_config import (
    load_generation_config_eos_ids,
    load_generation_config_sampling,
)


def _write(tmp_path, payload):
    """Write a generation_config.json into ``tmp_path`` and return the dir."""
    if payload is _MISSING:
        return str(tmp_path)
    path = tmp_path / "generation_config.json"
    if isinstance(payload, str):
        path.write_text(payload)
    else:
        path.write_text(json.dumps(payload))
    return str(tmp_path)


_MISSING = object()


class TestLoadGenerationConfigSampling:
    def test_curated_qwen_style(self, tmp_path):
        d = _write(
            tmp_path,
            {
                "do_sample": True,
                "temperature": 0.6,
                "top_k": 20,
                "top_p": 0.95,
                "eos_token_id": [151643, 151645],
                "transformers_version": "4.57.0",
            },
        )
        assert load_generation_config_sampling(d) == {
            "temperature": 0.6,
            "top_k": 20,
            "top_p": 0.95,
        }

    def test_drops_non_sampling_keys(self, tmp_path):
        d = _write(
            tmp_path,
            {
                "bos_token_id": 1,
                "eos_token_id": 2,
                "_from_model_config": True,
                "pad_token_id": 11,
            },
        )
        assert load_generation_config_sampling(d) == {}

    def test_repetition_penalty_passes_through(self, tmp_path):
        d = _write(
            tmp_path,
            {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 1.05,
            },
        )
        result = load_generation_config_sampling(d)
        assert result["repetition_penalty"] == 1.05

    def test_missing_file_returns_empty(self, tmp_path):
        # tmp_path exists but no generation_config.json inside it
        d = _write(tmp_path, _MISSING)
        assert load_generation_config_sampling(d) == {}

    def test_nonexistent_path_returns_empty(self):
        assert load_generation_config_sampling("/nonexistent/path/xyz") == {}

    def test_none_path_returns_empty(self):
        assert load_generation_config_sampling(None) == {}

    def test_empty_string_returns_empty(self):
        assert load_generation_config_sampling("") == {}

    def test_malformed_json_returns_empty(self, tmp_path):
        d = _write(tmp_path, "this is { not [ json")
        assert load_generation_config_sampling(d) == {}

    def test_non_dict_payload_returns_empty(self, tmp_path):
        d = _write(tmp_path, "[1, 2, 3]")
        assert load_generation_config_sampling(d) == {}

    def test_drops_bool_temperature(self, tmp_path):
        """JSON ``true`` would otherwise sneak through int isinstance check."""
        d = _write(tmp_path, {"temperature": True, "top_p": 0.9})
        assert load_generation_config_sampling(d) == {"top_p": 0.9}

    def test_drops_string_temperature(self, tmp_path):
        d = _write(tmp_path, {"temperature": "0.7", "top_p": 0.9})
        assert load_generation_config_sampling(d) == {"top_p": 0.9}

    def test_drops_nan_infinity(self, tmp_path):
        # JSON spec rejects NaN/inf, but some tooling emits them. Manual write.
        path = tmp_path / "generation_config.json"
        path.write_text('{"temperature": NaN, "top_p": Infinity, "top_k": 20}')
        # Python's json accepts these by default; we should still drop them.
        assert load_generation_config_sampling(str(tmp_path)) == {"top_k": 20}

    def test_glm47_partial_with_from_model_config(self, tmp_path):
        """GLM-4.7-Flash ships ``_from_model_config: True`` *and* a
        curated ``temperature`` — must extract the temperature."""
        d = _write(
            tmp_path,
            {"_from_model_config": True, "temperature": 1.0},
        )
        assert load_generation_config_sampling(d) == {"temperature": 1.0}

    def test_only_sampling_subset_extracted(self, tmp_path):
        """Future HF additions outside our subset must not leak through."""
        d = _write(
            tmp_path,
            {
                "temperature": 0.7,
                "typical_p": 0.9,  # NOT in our subset
                "epsilon_cutoff": 0.0,  # NOT in our subset
                "length_penalty": 1.0,  # NOT in our subset
            },
        )
        assert load_generation_config_sampling(d) == {"temperature": 0.7}

    def test_hf_hub_snapshot_layout(self, tmp_path, monkeypatch):
        """org/repo paths must resolve through the HF hub cache."""
        hub = tmp_path / "hf"
        repo_dir = hub / "models--mlx-community--Fakemodel-4bit" / "snapshots" / "abc"
        repo_dir.mkdir(parents=True)
        (repo_dir / "generation_config.json").write_text(
            json.dumps({"temperature": 0.4, "top_p": 0.7})
        )
        monkeypatch.setenv("HF_HUB_CACHE", str(hub))
        assert load_generation_config_sampling("mlx-community/Fakemodel-4bit") == {
            "temperature": 0.4,
            "top_p": 0.7,
        }

    def test_hf_hub_missing_repo_returns_empty(self, tmp_path, monkeypatch):
        """Repo not pulled locally → no network fetch, return empty."""
        hub = tmp_path / "hf"
        hub.mkdir()
        monkeypatch.setenv("HF_HUB_CACHE", str(hub))
        assert (
            load_generation_config_sampling("mlx-community/NeverDownloaded-4bit") == {}
        )

    def test_hf_hub_refs_main_resolution(self, tmp_path, monkeypatch):
        """Prefer ``refs/main`` SHA over a sorted-first stale snapshot."""
        hub = tmp_path / "hf"
        repo = hub / "models--mlx-community--Fakemodel-4bit"
        (repo / "refs").mkdir(parents=True)
        (repo / "refs" / "main").write_text("zzzcurrentsha\n")

        # Stale snapshot — would win on sorted() but shouldn't.
        old_snap = repo / "snapshots" / "aaa000oldstale"
        old_snap.mkdir(parents=True)
        (old_snap / "generation_config.json").write_text(
            json.dumps({"temperature": 99.9})
        )

        # Canonical snapshot
        new_snap = repo / "snapshots" / "zzzcurrentsha"
        new_snap.mkdir(parents=True)
        (new_snap / "generation_config.json").write_text(
            json.dumps({"temperature": 0.6, "top_p": 0.95})
        )

        monkeypatch.setenv("HF_HUB_CACHE", str(hub))
        assert load_generation_config_sampling("mlx-community/Fakemodel-4bit") == {
            "temperature": 0.6,
            "top_p": 0.95,
        }

    def test_hf_hub_refs_main_stale_falls_back_to_snapshot_scan(
        self, tmp_path, monkeypatch
    ):
        """If refs/main points at a SHA no longer on disk, scan snapshots."""
        hub = tmp_path / "hf"
        repo = hub / "models--mlx-community--Fakemodel-4bit"
        (repo / "refs").mkdir(parents=True)
        (repo / "refs" / "main").write_text("missing_sha\n")
        snap = repo / "snapshots" / "actuallypresent"
        snap.mkdir(parents=True)
        (snap / "generation_config.json").write_text(json.dumps({"top_p": 0.8}))
        monkeypatch.setenv("HF_HUB_CACHE", str(hub))
        assert load_generation_config_sampling("mlx-community/Fakemodel-4bit") == {
            "top_p": 0.8
        }

    def test_top_k_fractional_float_dropped(self, tmp_path):
        """``top_k`` must be a whole number; fractions hide bad configs."""
        d = _write(tmp_path, {"top_k": 20.5, "top_p": 0.9})
        assert load_generation_config_sampling(d) == {"top_p": 0.9}

    def test_top_k_integer_float_normalized_to_int(self, tmp_path):
        """``top_k: 20.0`` is a JSON whole number; pass through as int."""
        d = _write(tmp_path, {"top_k": 20.0})
        result = load_generation_config_sampling(d)
        assert result == {"top_k": 20}
        assert isinstance(result["top_k"], int)

    def test_local_directory_with_no_config(self, tmp_path):
        # Has weights file but no generation_config.json
        (tmp_path / "model.safetensors").write_text("dummy")
        assert load_generation_config_sampling(str(tmp_path)) == {}

    @pytest.mark.parametrize(
        "key, value",
        [
            ("temperature", 0.5),
            ("top_p", 0.9),
            ("top_k", 20),
            ("min_p", 0.05),
            ("repetition_penalty", 1.1),
            ("presence_penalty", 0.3),
            ("frequency_penalty", 0.2),
        ],
    )
    def test_all_supported_sampling_keys(self, tmp_path, key, value):
        d = _write(tmp_path, {key: value})
        assert load_generation_config_sampling(d) == {key: value}


class TestSafeWithWeirdFilesystem:
    """Don't let a bad model dir crash the server at startup."""

    def test_permission_denied_returns_empty(self, tmp_path):
        path = tmp_path / "generation_config.json"
        path.write_text("{}")
        try:
            os.chmod(path, 0o000)
            # Should not raise; should return empty
            assert load_generation_config_sampling(str(tmp_path)) == {}
        finally:
            os.chmod(path, 0o644)


class TestLoadGenerationConfigEosIds:
    """Coverage for the chat-template-terminator harvest path.

    Motivating bug: Gemma 3 / 3n ship ``eos_token: <eos>`` (id 1) in
    ``tokenizer_config.json`` but declare the actual chat-template
    terminator ``<end_of_turn>`` (id 106) only in
    ``generation_config.json``'s ``eos_token_id`` array. Without this
    helper the scheduler stop set misses id 106 and the model emits
    ``<end_of_turn>`` as a literal token until ``max_tokens``.
    """

    def test_gemma3_style_list_extracts_extra_ids(self, tmp_path):
        d = _write(
            tmp_path,
            {
                "bos_token_id": 2,
                "eos_token_id": [1, 106],
                "pad_token_id": 0,
            },
        )
        assert load_generation_config_eos_ids(d) == (1, 106)

    def test_single_int_eos_returns_empty(self, tmp_path):
        # A bare int is already covered by tokenizer.eos_token_id —
        # returning it here would add no new information.
        d = _write(tmp_path, {"eos_token_id": 2})
        assert load_generation_config_eos_ids(d) == ()

    def test_missing_eos_key_returns_empty(self, tmp_path):
        d = _write(tmp_path, {"temperature": 0.6})
        assert load_generation_config_eos_ids(d) == ()

    def test_missing_file_returns_empty(self, tmp_path):
        d = _write(tmp_path, _MISSING)
        assert load_generation_config_eos_ids(d) == ()

    def test_none_path_returns_empty(self):
        assert load_generation_config_eos_ids(None) == ()

    def test_empty_string_returns_empty(self):
        assert load_generation_config_eos_ids("") == ()

    def test_malformed_json_returns_empty(self, tmp_path):
        d = _write(tmp_path, "not { valid")
        assert load_generation_config_eos_ids(d) == ()

    def test_non_dict_payload_returns_empty(self, tmp_path):
        d = _write(tmp_path, "[1, 2]")
        assert load_generation_config_eos_ids(d) == ()

    def test_drops_bool_inside_list(self, tmp_path):
        # JSON ``true`` decodes to ``int``; filter it.
        d = _write(tmp_path, {"eos_token_id": [True, 106]})
        assert load_generation_config_eos_ids(d) == (106,)

    def test_drops_strings_inside_list(self, tmp_path):
        d = _write(tmp_path, {"eos_token_id": ["<eos>", 106]})
        assert load_generation_config_eos_ids(d) == (106,)

    def test_all_strings_returns_empty(self, tmp_path):
        d = _write(tmp_path, {"eos_token_id": ["<eos>", "<end_of_turn>"]})
        assert load_generation_config_eos_ids(d) == ()


class TestAugmentEosFromGenerationConfig:
    """Integration coverage for the tokenizer-load-layer augment.

    The fix is a single mutation point at load time. Two tokenizer
    shapes flow through Rapid-MLX:

    1. mlx-lm ``TokenizerWrapper`` — mutate its ``_eos_token_ids``
       set directly.
    2. Raw HF tokenizer (mlx-vlm processors hand these out) — set
       the plural ``eos_token_ids`` instance attribute that the
       schedulers' source-3 union branch reads.

    After augmentation every downstream consumer (text + MLLM
    schedulers, mlx-lm BatchGenerator, DFlash) sees the full stop
    set without per-consumer plumbing.
    """

    def test_shape1_mutates_wrapper_set(self, tmp_path):
        from vllm_mlx.utils.tokenizer import (
            augment_eos_token_ids_from_generation_config,
        )

        class _WrapperStub:
            def __init__(self, primary: int):
                self._eos_token_ids: set[int] = {primary}

        d = _write(tmp_path, {"bos_token_id": 2, "eos_token_id": [1, 106]})
        tok = _WrapperStub(primary=1)
        augment_eos_token_ids_from_generation_config(tok, d)
        assert tok._eos_token_ids == {1, 106}

    def test_shape2_stashes_extras_on_rapid_attr(self, tmp_path):
        """For raw HF tokenizers (mlx-vlm processors), the augment
        cannot assign to ``eos_token_ids`` directly — HF defines it
        as a property descriptor that rejects non-string values.
        Instead the extras are stashed on RAPID_EXTRA_EOS_ATTR and
        the scheduler's source-4 union reads them from there."""
        from vllm_mlx.utils.tokenizer import (
            RAPID_EXTRA_EOS_ATTR,
            augment_eos_token_ids_from_generation_config,
        )

        class _HFTok:
            eos_token_id = 1

        d = _write(tmp_path, {"eos_token_id": [1, 106]})
        tok = _HFTok()
        augment_eos_token_ids_from_generation_config(tok, d)
        assert tuple(getattr(tok, RAPID_EXTRA_EOS_ATTR)) == (1, 106)
        # The HF eos_token_id property is left untouched so other HF
        # code paths (encode w/ EOS, etc.) keep working.
        assert tok.eos_token_id == 1

    def test_shape2_idempotent_unions_with_prior_stash(self, tmp_path):
        from vllm_mlx.utils.tokenizer import (
            RAPID_EXTRA_EOS_ATTR,
            augment_eos_token_ids_from_generation_config,
        )

        class _HFTok:
            eos_token_id = 1

        d = _write(tmp_path, {"eos_token_id": [1, 106]})
        tok = _HFTok()
        # Caller pre-seeded a sibling extra (e.g. a custom finetune).
        setattr(tok, RAPID_EXTRA_EOS_ATTR, (42,))
        augment_eos_token_ids_from_generation_config(tok, d)
        assert tuple(getattr(tok, RAPID_EXTRA_EOS_ATTR)) == (1, 42, 106)

    def test_no_extras_is_no_op(self, tmp_path):
        from vllm_mlx.utils.tokenizer import (
            augment_eos_token_ids_from_generation_config,
        )

        class _WrapperStub:
            def __init__(self):
                self._eos_token_ids = {1}

        d = _write(tmp_path, {"eos_token_id": 2})  # singular, not a list
        tok = _WrapperStub()
        augment_eos_token_ids_from_generation_config(tok, d)
        assert tok._eos_token_ids == {1}

    def test_scheduler_reads_wrapper_set_via_get_stop_tokens(self):
        """End-to-end union: a wrapper-shaped tokenizer with a grown
        ``_eos_token_ids`` set is correctly unioned by
        ``Scheduler._get_stop_tokens``."""
        from vllm_mlx.scheduler import Scheduler, SchedulerConfig

        class _WrapperStub:
            def __init__(self):
                self._eos_token_ids = {1, 106}
                self.eos_token_id = 1  # __getattr__ surface

        tok = _WrapperStub()
        sched = Scheduler.__new__(Scheduler)
        sched.tokenizer = tok
        sched._actual_tokenizer = tok
        sched.config = SchedulerConfig()
        assert sched._get_stop_tokens() == {1, 106}

    def test_mllm_scheduler_reads_rapid_stash_via_get_stop_tokens(self):
        """End-to-end union: a raw HF tokenizer with the Rapid-MLX
        extras stash on RAPID_EXTRA_EOS_ATTR is correctly unioned
        by ``MLLMScheduler._get_stop_tokens``."""
        from vllm_mlx.mllm_scheduler import MLLMScheduler
        from vllm_mlx.utils.tokenizer import RAPID_EXTRA_EOS_ATTR

        class _HFTok:
            eos_token_id = 1

        tok = _HFTok()
        setattr(tok, RAPID_EXTRA_EOS_ATTR, (1, 106))

        class _Processor:
            tokenizer = tok

        sched = MLLMScheduler.__new__(MLLMScheduler)
        sched.processor = _Processor()
        assert sched._get_stop_tokens() == {1, 106}
