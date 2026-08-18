# SPDX-License-Identifier: Apache-2.0
"""An alias that declares MTP must actually serve MTP (#1998).

``rapid-mlx models`` renders ``✓ MTP`` and ``MTP@<repo>@<k>`` for any alias
carrying ``mtp_draft_model``. Until this fix those two fields were read by that
renderer and by nothing else, so the command the listing implies —

    rapid-mlx serve qwen3.8-27b-mixed-3.5bpw --speculative-config '{"method":"mtp"}'

— reached the injector with ``sidecar=None`` and hard-failed at boot, suggesting
an unrelated model (``mlx-community/Qwen3.5-9B-MTP-4bit``) in the remedy. The
declared depth was ignored too: the alias says 3, the controller ran ``max_k=1``.

The fail-closed boot is correct and stays. What changes is that the serve path
now reads the declaration the registry already advertises.

Precedence is the substance of these tests: an explicit ``--speculative-config``
must always win, and an alias with NO declaration must keep hard-failing rather
than silently acquiring a sidecar.
"""

from types import SimpleNamespace

import pytest

# Real registry entries: one that declares MTP, one that does not. Using the
# real aliases (not fixtures) is deliberate — the bug was precisely that the
# registry and the serve path disagreed, so the test has to read the registry.
ALIAS_WITH_MTP = "qwen3.8-27b-mixed-3.5bpw"
ALIAS_WITHOUT_MTP = "qwen3.6-35b-8bit"


def _args(**overrides):
    data = {
        "model": None,
        "speculative_config": None,
        "enable_ddtree": False,
        "enable_dflash": False,
        "spec_decode": "none",
        "dflash_drafter_path": "",
        "mtp_num_draft_tokens": 1,
        "mtp_optimistic": False,
        "mtp_sidecar": None,
        "mtp_max_k": None,
        "mtp_disable_auto_k": False,
        "force_spec_decode": False,
        "suffix_decoding": False,
        "suffix_max_draft": None,
        "suffix_max_suffix_len": None,
        "suffix_min_confidence": None,
        "suffix_min_draft_len": None,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def _normalize(args):
    from vllm_mlx.cli import _normalize_speculative_config_or_exit

    _normalize_speculative_config_or_exit(args)
    return args


# --------------------------------------------------------------------- registry
def test_registry_precondition_the_alias_still_declares_mtp():
    """If this fails, the rest of the file is testing nothing."""
    from vllm_mlx.model_aliases import resolve_profile

    profile = resolve_profile(ALIAS_WITH_MTP)
    assert profile is not None
    assert profile.mtp_draft_model, (
        f"{ALIAS_WITH_MTP} no longer declares mtp_draft_model — update this test "
        "to another declaring alias, or the #1998 wiring is now untested"
    )
    assert profile.mtp_speculative_tokens == 3

    other = resolve_profile(ALIAS_WITHOUT_MTP)
    assert other is not None
    assert not other.mtp_draft_model


# ------------------------------------------------------------------ the fix
def test_alias_declaration_supplies_the_sidecar_and_depth():
    from vllm_mlx.model_aliases import resolve_profile

    declared = resolve_profile(ALIAS_WITH_MTP)
    args = _normalize(
        _args(model=ALIAS_WITH_MTP, speculative_config='{"method":"mtp"}')
    )
    assert args.spec_decode == "mtp"
    assert args.mtp_sidecar == declared.mtp_draft_model
    assert args.mtp_max_k == declared.mtp_speculative_tokens


def test_alias_without_a_declaration_still_gets_no_sidecar():
    """The injector's hard-fail must stay reachable — no silent acquisition."""
    args = _normalize(
        _args(model=ALIAS_WITHOUT_MTP, speculative_config='{"method":"mtp"}')
    )
    assert args.spec_decode == "mtp"
    assert args.mtp_sidecar is None


def test_unknown_model_is_left_alone():
    args = _normalize(
        _args(
            model="not-a-real-org/not-a-real-model",
            speculative_config='{"method":"mtp"}',
        )
    )
    assert args.mtp_sidecar is None


# --------------------------------------------------------------- precedence
def test_explicit_model_beats_the_alias_declaration():
    args = _normalize(
        _args(
            model=ALIAS_WITH_MTP,
            speculative_config='{"method":"mtp","model":"org/explicit-sidecar"}',
        )
    )
    assert args.mtp_sidecar == "org/explicit-sidecar"


def test_explicit_depth_beats_the_alias_declaration():
    args = _normalize(
        _args(
            model=ALIAS_WITH_MTP,
            speculative_config='{"method":"mtp","num_speculative_tokens":7}',
        )
    )
    assert args.mtp_max_k == 7
    # …while the sidecar still comes from the alias: the two fill independently.
    assert args.mtp_sidecar


def test_alias_depth_beats_the_generic_force_spec_decode_default(monkeypatch):
    """--force-spec-decode's K=3 is a generic fallback; the alias is specific.

    The declared depth is forced to 5 here ON PURPOSE. The real alias declares
    3, which is also the force-fallback value, so asserting against the real
    registry would pass whether or not the alias is consulted at all — the
    mutation run proved exactly that. 5 makes the two sources distinguishable.
    """
    import vllm_mlx.model_aliases as aliases

    real = aliases.resolve_profile(ALIAS_WITH_MTP)
    monkeypatch.setattr(
        aliases,
        "resolve_profile",
        lambda _n: SimpleNamespace(
            mtp_draft_model=real.mtp_draft_model, mtp_speculative_tokens=5
        ),
    )
    args = _normalize(
        _args(
            model=ALIAS_WITH_MTP,
            speculative_config='{"method":"mtp"}',
            force_spec_decode=True,
        )
    )
    assert args.mtp_max_k == 5


def test_force_spec_decode_fallback_survives_for_undeclared_aliases():
    args = _normalize(
        _args(
            model=ALIAS_WITHOUT_MTP,
            speculative_config='{"method":"mtp"}',
            force_spec_decode=True,
        )
    )
    assert args.mtp_max_k == 3
    assert args.mtp_sidecar is None


# ------------------------------------------------------------- helper totality
@pytest.mark.parametrize("model", [None, "", "org/unknown-model"])
def test_declaration_lookup_is_total(model):
    from vllm_mlx.cli import _alias_mtp_declaration

    assert _alias_mtp_declaration(model) == (None, None)


def test_declaration_lookup_never_raises_when_the_registry_is_broken(monkeypatch):
    """A registry failure must degrade to "no default", not crash the serve."""
    import vllm_mlx.model_aliases as aliases
    from vllm_mlx.cli import _alias_mtp_declaration

    def _boom(_name):
        raise RuntimeError("registry unreadable")

    monkeypatch.setattr(aliases, "resolve_profile", _boom)
    assert _alias_mtp_declaration(ALIAS_WITH_MTP) == (None, None)


@pytest.mark.parametrize("bad", [7, ["org/repo"], {"a": 1}, object()])
def test_a_non_string_sidecar_does_not_raise(monkeypatch, bad):
    """Totality has to survive the value being the wrong TYPE, not just absent."""
    import vllm_mlx.model_aliases as aliases
    from vllm_mlx.cli import _alias_mtp_declaration

    monkeypatch.setattr(
        aliases,
        "resolve_profile",
        lambda _n: SimpleNamespace(mtp_draft_model=bad, mtp_speculative_tokens=3),
    )
    assert _alias_mtp_declaration(ALIAS_WITH_MTP) == (None, None)


def test_depth_without_a_sidecar_yields_neither(monkeypatch):
    """A hand-edited registry could carry a depth alone; it is meaningless."""
    import vllm_mlx.model_aliases as aliases
    from vllm_mlx.cli import _alias_mtp_declaration

    monkeypatch.setattr(
        aliases,
        "resolve_profile",
        lambda _n: SimpleNamespace(mtp_draft_model=None, mtp_speculative_tokens=5),
    )
    assert _alias_mtp_declaration(ALIAS_WITH_MTP) == (None, None)


# ------------------------------------------------------------------- labelling
def test_info_reports_the_sidecar_lane_instead_of_disabled():
    """`MTP path: disabled` was wrong for a model that decodes with MTP."""
    from vllm_mlx.model_aliases import resolve_profile
    from vllm_mlx.model_auto_config import _mtp_path_label

    declaring = resolve_profile(ALIAS_WITH_MTP)
    label = _mtp_path_label(declaring.hf_path, declaring)
    assert "sidecar" in label and "--speculative-config" in label

    plain = resolve_profile(ALIAS_WITHOUT_MTP)
    assert _mtp_path_label(plain.hf_path, plain) == "disabled"
