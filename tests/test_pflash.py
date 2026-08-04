# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PFlash long-prompt compression (#287).

Ported and extended from @michaelasper's reference implementation on the
``pflash-qwen36-ttft`` fork (tests/test_pflash.py at commit d7a2797).
The fork's edge-preservation, determinism, and skip-reason tests carry
over directly; the budget invariant and the prompt-integrity / tool
skip tests are kept as written because they encode the contract the
scheduler depends on.
"""

from types import SimpleNamespace

from vllm_mlx.pflash import (
    PFlashConfig,
    compress_request_tokens,
    compress_tokens,
    config_from_args,
    resolve_pflash_config,
    resolve_pflash_keep_ratio_default,
    resolve_pflash_mode_default,
    validate_model_support,
)


class TestPFlashCompressor:
    def test_compresses_long_prompt_preserving_edges_and_order(self):
        tokens = (
            list(range(20)) + [100] * 64 + list(range(200, 264)) + list(range(900, 920))
        )
        config = PFlashConfig(
            mode="auto",
            threshold=64,
            keep_ratio=0.35,
            min_keep_tokens=32,
            sink_tokens=8,
            tail_tokens=8,
            block_size=8,
            query_window=8,
            stride_blocks=0,
        )

        result = compress_tokens(tokens, config)

        assert result.compressed is True
        assert result.original_tokens == len(tokens)
        assert len(result.tokens) < len(tokens)
        assert result.tokens[:8] == tokens[:8]
        assert result.tokens[-8:] == tokens[-8:]
        # Original order preserved across kept tokens.
        assert result.tokens == sorted(result.tokens, key=tokens.index)

    def test_keeps_query_overlap_blocks_over_repetitive_filler(self):
        prefix = list(range(10))
        filler = [7] * 96
        needle_block = [501, 502, 503, 504, 900, 901, 902, 903]
        more_filler = [8] * 96
        tail = [900, 901, 902, 903, 1000, 1001, 1002, 1003]
        tokens = prefix + filler + needle_block + more_filler + tail
        config = PFlashConfig(
            mode="always",
            threshold=1,
            keep_ratio=0.20,
            min_keep_tokens=24,
            sink_tokens=4,
            tail_tokens=8,
            block_size=8,
            query_window=8,
            stride_blocks=0,
        )

        result = compress_tokens(tokens, config)

        assert result.compressed is True
        # Needle block shares tokens 900–903 with the tail so it should
        # rank above the repetitive filler blocks.
        assert all(token in result.tokens for token in needle_block)

    def test_deterministic_across_runs(self):
        tokens = list(range(2000))
        config = PFlashConfig(
            mode="always",
            threshold=1,
            keep_ratio=0.10,
            sink_tokens=16,
            tail_tokens=32,
            block_size=16,
            query_window=32,
        )

        runs = [compress_tokens(tokens, config).tokens for _ in range(5)]
        assert all(r == runs[0] for r in runs)

    def test_skips_tool_prompts_by_default(self):
        tokens = list(range(200))
        config = PFlashConfig(
            mode="auto",
            threshold=10,
            keep_ratio=0.10,
            skip_when_tools=True,
        )

        result = compress_tokens(tokens, config, has_tools=True)

        assert result.compressed is False
        assert result.reason == "tools"
        assert result.tokens is tokens

    def test_skips_prompt_integrity_requests(self):
        tokens = list(range(200))
        config = PFlashConfig(
            mode="always",
            threshold=1,
            keep_ratio=0.10,
            skip_when_tools=True,
        )

        result = compress_tokens(tokens, config, requires_prompt_integrity=True)

        assert result.compressed is False
        assert result.reason == "protected_prompt"
        assert result.tokens is tokens

    def test_does_not_exceed_keep_budget_when_blocks_are_large(self):
        tokens = list(range(100))
        config = PFlashConfig(
            mode="always",
            threshold=1,
            keep_ratio=0.03,
            min_keep_tokens=3,
            sink_tokens=0,
            tail_tokens=0,
            block_size=8,
        )

        result = compress_tokens(tokens, config)

        assert result.compressed is True
        # Original fork allowed a small overshoot; the adapted compressor
        # truncates each candidate block at the remaining slot budget.
        assert len(result.tokens) <= 3

    def test_empty_prompt_short_circuits(self):
        config = PFlashConfig(mode="always", threshold=1)
        result = compress_tokens([], config)
        assert result.compressed is False
        assert result.reason == "empty"

    def test_below_threshold_skips_in_auto_mode(self):
        tokens = list(range(100))
        config = PFlashConfig(mode="auto", threshold=1024, keep_ratio=0.1)
        result = compress_tokens(tokens, config)
        assert result.compressed is False
        assert result.reason == "threshold"

    def test_disabled_mode_returns_unchanged(self):
        tokens = list(range(10_000))
        config = PFlashConfig(mode="off")
        result = compress_tokens(tokens, config)
        assert result.compressed is False
        assert result.reason == "off"
        assert result.tokens is tokens


class TestPFlashConfig:
    def test_validate_rejects_invalid_values(self):
        invalid_configs = [
            PFlashConfig(mode="unknown"),  # type: ignore[arg-type]
            PFlashConfig(threshold=-1),
            PFlashConfig(keep_ratio=0),
            PFlashConfig(keep_ratio=1.1),
            PFlashConfig(min_keep_tokens=-1),
            PFlashConfig(sink_tokens=-1),
            PFlashConfig(tail_tokens=-1),
            PFlashConfig(block_size=0),
            PFlashConfig(query_window=0),
            PFlashConfig(stride_blocks=-1),
        ]
        for config in invalid_configs:
            try:
                config.validate()
            except ValueError:
                pass
            else:
                raise AssertionError(f"expected invalid PFlash config: {config!r}")

    def test_config_from_args_maps_include_tools_inversion(self):
        args = SimpleNamespace(
            pflash="auto",
            pflash_threshold=1024,
            pflash_keep_ratio=0.25,
            pflash_min_keep_tokens=128,
            pflash_sink_tokens=16,
            pflash_tail_tokens=64,
            pflash_block_size=32,
            pflash_query_window=128,
            pflash_stride_blocks=4,
            pflash_include_tools=True,
        )

        config = config_from_args(args)

        assert config.mode == "auto"
        assert config.keep_ratio == 0.25
        # CLI exposes the positive form (--pflash-include-tools); the
        # config field uses the inverted predicate so the default
        # behaviour is the conservative skip.
        assert config.skip_when_tools is False

    def test_config_from_args_none_keep_ratio_falls_back_to_default(self):
        # The CLI default for --pflash-keep-ratio is now a None sentinel
        # (resolved to an alias override or 0.20 before construction). If the
        # resolver never ran (bare SimpleNamespace, or the --enable-dflash
        # path that skips PFlash resolution), config_from_args must fall back
        # to 0.20 rather than fail PFlashConfig.validate on ``None``.
        args = SimpleNamespace(
            pflash="always",
            pflash_threshold=1024,
            pflash_keep_ratio=None,
            pflash_min_keep_tokens=128,
            pflash_sink_tokens=16,
            pflash_tail_tokens=64,
            pflash_block_size=32,
            pflash_query_window=128,
            pflash_stride_blocks=4,
            pflash_include_tools=False,
        )
        config = config_from_args(args)
        assert config.keep_ratio == 0.20

    def test_validate_rejects_multimodal_models(self):
        config = PFlashConfig(mode="auto")
        try:
            validate_model_support(config, model_name="qwen-vl", is_mllm=True)
        except ValueError as exc:
            assert "multimodal" in str(exc)
        else:
            raise AssertionError("expected multimodal PFlash config to be rejected")

    def test_validate_allows_text_models(self):
        config = PFlashConfig(mode="auto")
        validate_model_support(config, model_name="qwen3-coder", is_mllm=False)

    def test_validate_allows_mllm_when_disabled(self):
        config = PFlashConfig(mode="off")
        validate_model_support(config, model_name="qwen-vl", is_mllm=True)


class TestResolvePFlashModeDefault:
    """Per-alias tier-based default for ``--pflash`` (#287).

    The contract:
    * If the user passed ``--pflash {off,auto,always}`` (i.e. ``args.pflash``
      is not None), the resolver returns that value unchanged.
    * Otherwise the alias's ``pflash_tier`` decides: ``"verified"`` →
      ``"always"``, anything else → ``"off"``.
    """

    def _ns(self, pflash):
        return SimpleNamespace(pflash=pflash)

    def test_verified_alias_with_no_flag_defaults_to_always(self):
        # qwen3.5-4b-4bit is tagged pflash_tier=verified in aliases.json
        # (PR #649). Mirror the alias-driven default the engine wires up.
        mode = resolve_pflash_mode_default(self._ns(None), model_name="qwen3.5-4b-4bit")
        assert mode == "always"

    def test_verified_alias_default_branch_emits_log_without_error(self, caplog):
        # Regression guard for the module-level ``logger`` binding: the
        # verified-alias default path calls ``logger.info(...)`` and must
        # resolve cleanly (a stray NameError here would break the exact code
        # this PR touches). Assert the branch both returns "always" AND emits
        # its INFO line, so the logging call is provably exercised.
        import logging as _logging

        with caplog.at_level(_logging.INFO, logger="vllm_mlx.pflash"):
            mode = resolve_pflash_mode_default(
                self._ns(None), model_name="qwen3.5-4b-4bit"
            )
        assert mode == "always"
        assert any(
            "pflash_tier=verified" in rec.message and "qwen3.5-4b-4bit" in rec.message
            for rec in caplog.records
        ), "verified-alias default branch did not emit its INFO log"

    def test_multimodal_suppression_log_does_not_advise_a_flag_that_errors(
        self, caplog
    ):
        # codex #2 nit on #1178: the multimodal-suppression log must NOT tell
        # the user to "Pass --pflash always" as an override — that flag is
        # rejected downstream by validate_model_support for the MLLM lane. The
        # message should keep the user's mental model correct instead.
        import logging as _logging

        with caplog.at_level(_logging.INFO, logger="vllm_mlx.pflash"):
            mode = resolve_pflash_mode_default(
                self._ns(None), model_name="qwen3.5-4b-4bit", is_multimodal=True
            )
        assert mode == "off"
        msgs = [rec.message for rec in caplog.records if "multimodal" in rec.message]
        assert msgs, "multimodal-suppression branch did not emit its INFO log"
        joined = " ".join(msgs)
        assert "Pass --pflash always to override" not in joined
        # It should instead signal that PFlash is unavailable / an override errors.
        assert "unavailable" in joined and "rejected" in joined

    def test_verified_alias_multimodal_suppresses_always(self):
        # A verified alias that ALSO routes multimodally (a vision-config
        # Qwen3.6-27B checkpoint is both) must NOT auto-enable PFlash — the
        # MLLM lane is rejected by validate_model_support, so the naive
        # default-serve command would otherwise die on a --pflash flag the
        # user never set (#352 dogfood P1-②). The caller passes the same
        # is_mllm verdict it feeds validate_model_support.
        mode = resolve_pflash_mode_default(
            self._ns(None), model_name="qwen3.5-4b-4bit", is_multimodal=True
        )
        assert mode == "off"

    def test_explicit_always_wins_even_when_multimodal(self):
        # is_multimodal only suppresses the AUTO tier default; an explicit
        # --pflash always still wins (and is then rejected loudly downstream
        # by validate_model_support — the user asked for it).
        mode = resolve_pflash_mode_default(
            self._ns("always"), model_name="qwen3.5-4b-4bit", is_multimodal=True
        )
        assert mode == "always"

    def test_unknown_alias_with_no_flag_defaults_to_off(self):
        # qwen3-0.6b-4bit is an explicit non-Qwen3.5/3.6 entry; its
        # default pflash_tier is "unknown".
        mode = resolve_pflash_mode_default(self._ns(None), model_name="qwen3-0.6b-4bit")
        assert mode == "off"

    def test_unrecognized_model_path_defaults_to_off(self):
        # No alias profile match → detect_model_config returns None →
        # resolver falls through to the conservative "off".
        mode = resolve_pflash_mode_default(
            self._ns(None), model_name="some/unmapped-model-path"
        )
        assert mode == "off"

    def test_explicit_off_wins_over_verified_default(self):
        mode = resolve_pflash_mode_default(
            self._ns("off"), model_name="qwen3.5-4b-4bit"
        )
        assert mode == "off"

    def test_explicit_auto_wins_over_unknown_default(self):
        mode = resolve_pflash_mode_default(
            self._ns("auto"), model_name="qwen3-0.6b-4bit"
        )
        assert mode == "auto"

    def test_explicit_always_wins_for_unknown_alias(self):
        mode = resolve_pflash_mode_default(
            self._ns("always"), model_name="some/unmapped-model-path"
        )
        assert mode == "always"

    def test_verified_aliases_in_registry_match_qwen35_or_qwen36(self):
        # Defense-in-depth alongside the contract test in
        # tests/test_aliases_contract.py: verify the resolver returns
        # "always" for every verified alias in the registry, not just
        # the qwen3.5-4b-4bit sample. Catches the case where a future
        # contributor edits the JSON-level tag but the model_auto_config
        # → pflash threading regresses.
        from vllm_mlx.model_aliases import list_profiles

        verified = [
            a for a, p in list_profiles().items() if p.pflash_tier == "verified"
        ]
        assert verified, "no verified aliases — see PR #649 / aliases.json"
        for alias in verified:
            mode = resolve_pflash_mode_default(self._ns(None), model_name=alias)
            assert mode == "always", (
                f"{alias}: tier=verified but resolver returned {mode!r}"
            )

    def test_config_from_args_treats_none_mode_as_off(self):
        # Defensive: config_from_args is the public surface other tests
        # exercise via SimpleNamespace. Callers that build args.pflash=None
        # and skip the resolver should still get a valid (off) config
        # rather than a ValueError or silent compression.
        args = SimpleNamespace(
            pflash=None,
            pflash_threshold=1024,
            pflash_keep_ratio=0.25,
            pflash_min_keep_tokens=128,
            pflash_sink_tokens=16,
            pflash_tail_tokens=64,
            pflash_block_size=32,
            pflash_query_window=128,
            pflash_stride_blocks=4,
            pflash_include_tools=False,
        )
        config = config_from_args(args)
        assert config.mode == "off"


class TestResolvePFlashKeepRatioDefault:
    """Per-alias ``pflash_keep_ratio`` override (#287 follow-up).

    Contract mirrors the mode resolver:
    * explicit ``--pflash-keep-ratio`` (args value not None) wins;
    * else the alias's ``pflash_keep_ratio`` if it pins one;
    * else the engine default 0.20.
    """

    def _ns(self, keep_ratio):
        return SimpleNamespace(pflash_keep_ratio=keep_ratio)

    def test_alias_override_applies_when_no_flag(self):
        # bonsai-27b-2bit is verified BUT only recall-safe at 0.50 (1/5 needle
        # at the 0.20 default); it pins pflash_keep_ratio=0.5 in aliases.json.
        ratio = resolve_pflash_keep_ratio_default(
            self._ns(None), model_name="bonsai-27b-2bit"
        )
        assert ratio == 0.5

    def test_explicit_flag_wins_over_alias_override(self):
        ratio = resolve_pflash_keep_ratio_default(
            self._ns(0.33), model_name="bonsai-27b-2bit"
        )
        assert ratio == 0.33

    def test_verified_alias_without_override_uses_engine_default(self):
        # qwen3.5-4b-4bit is verified at the default 0.20 and pins no override.
        ratio = resolve_pflash_keep_ratio_default(
            self._ns(None), model_name="qwen3.5-4b-4bit"
        )
        assert ratio == 0.20

    def test_unknown_alias_without_override_uses_engine_default(self):
        ratio = resolve_pflash_keep_ratio_default(
            self._ns(None), model_name="qwen3-0.6b-4bit"
        )
        assert ratio == 0.20

    def test_unrecognized_model_path_uses_engine_default(self):
        ratio = resolve_pflash_keep_ratio_default(
            self._ns(None), model_name="/no/such/model-xyz"
        )
        assert ratio == 0.20

    def _full_ns(self, **overrides):
        base = dict(
            pflash=None,
            pflash_keep_ratio=None,
            pflash_threshold=32_768,
            pflash_min_keep_tokens=2_048,
            pflash_sink_tokens=256,
            pflash_tail_tokens=2_048,
            pflash_block_size=128,
            pflash_query_window=512,
            pflash_stride_blocks=8,
            pflash_include_tools=False,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_resolve_pflash_config_wires_bonsai_mode_and_ratio_end_to_end(self):
        # Guards the actual serve/bench WIRING, not just the resolvers: a bare
        # ``serve bonsai-27b-2bit`` routes through resolve_pflash_config, which
        # must resolve mode→"always" AND keep_ratio→0.5 from the alias and bake
        # both into the built PFlashConfig. If either resolver call is dropped
        # from the helper, one of these assertions fails (a test that called the
        # resolvers directly would still pass — codex #1458 BLOCKING).
        args = self._full_ns()
        config = resolve_pflash_config(args, model_name="bonsai-27b-2bit")
        assert config.mode == "always"
        assert config.keep_ratio == 0.5
        # The helper also materializes the resolved values back onto args so
        # later readers (engine wiring) see them, not the None sentinels.
        assert args.pflash == "always"
        assert args.pflash_keep_ratio == 0.5

    def test_resolve_pflash_config_explicit_keep_ratio_flag_wins(self):
        # An explicit --pflash-keep-ratio must survive the shared wiring even
        # when the alias pins its own override.
        args = self._full_ns(pflash_keep_ratio=0.33)
        config = resolve_pflash_config(args, model_name="bonsai-27b-2bit")
        assert config.mode == "always"
        assert config.keep_ratio == 0.33

    def test_resolve_pflash_config_unknown_alias_stays_off_at_default(self):
        # A non-verified alias: mode stays off and keep_ratio falls to 0.20.
        args = self._full_ns()
        config = resolve_pflash_config(args, model_name="qwen3-0.6b-4bit")
        assert config.mode == "off"
        assert config.keep_ratio == 0.20


class TestCompressRequestTokens:
    def test_reports_compression_metrics(self):
        tokens = list(range(256))
        config = PFlashConfig(
            mode="always",
            threshold=1,
            keep_ratio=0.25,
            min_keep_tokens=32,
            sink_tokens=8,
            tail_tokens=16,
            block_size=8,
        )

        compressed, metadata = compress_request_tokens(tokens, config, has_tools=False)

        assert len(compressed) < len(tokens)
        assert metadata["compressed"] is True
        assert metadata["reason"] == "compressed"
        assert metadata["original_tokens"] == 256
        assert metadata["kept_tokens"] == len(compressed)
        assert metadata["dropped_tokens"] == 256 - len(compressed)
        assert metadata["compression_ratio"] == len(compressed) / 256

    def test_skip_metadata_for_tools(self):
        tokens = list(range(64))
        config = PFlashConfig(mode="always", threshold=1, keep_ratio=0.5)
        _, metadata = compress_request_tokens(tokens, config, has_tools=True)
        assert metadata["compressed"] is False
        assert metadata["reason"] == "tools"
        assert metadata["dropped_tokens"] == 0
