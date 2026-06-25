# SPDX-License-Identifier: Apache-2.0
"""R15 task #297 — load-time guardrail for the MoE+MXFP4+multi-device cliff.

Covers the detection matrix for the two upstream MLX issues exposed by
``vllm_mlx/_mxfp4_moe_guardrail.py``:

* mlx#3402 — MoE + MXFP4 + multi-device throughput cliff
  (3-of-3 → fire; any 2-of-3 → silent).
* mlx#2962 — MoE + NVFP4 dynamic-range loss
  (fires regardless of device count, MoE required).

The tests are pure: every signal is constructed by hand via
:class:`GuardrailSignal`, so we don't pull any MoE weights from disk
just to exercise the guardrail (per the task disk-hygiene constraint).
The mlx.distributed probe is exercised through ``check_from_profile``
with monkeypatched detectors so we don't need a real MPI run either.
"""

from __future__ import annotations

import logging

import pytest

from vllm_mlx import _mxfp4_moe_guardrail as g


@pytest.fixture(autouse=True)
def _reset_guardrail_state():
    """Zero the module counters around every test for isolation."""
    g.reset_for_tests()
    yield
    g.reset_for_tests()


# ---------------------------------------------------------------------------
# _detect_quant_format — path-name heuristic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hf_path,expected",
    [
        ("mlx-community/MiniMax-M2.7-4bit-mxfp4", "mxfp4"),
        ("nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx", "mxfp4"),
        ("mlx-community/gpt-oss-20b-MXFP4-Q8", "mxfp4"),  # case-insensitive
        ("vendor/Some-Model-NVFP4", "nvfp4"),
        ("vendor/some-nvfp4-moe", "nvfp4"),
        ("mlx-community/Qwen3-7B-4bit", None),
        ("mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit", None),
        ("", None),
        (None, None),
    ],
)
def test_detect_quant_format(hf_path, expected):
    """Path-name heuristic detects mxfp4/nvfp4 case-insensitively."""
    assert g._detect_quant_format(hf_path) == expected


def test_detect_quant_format_mxfp4_wins_over_nvfp4():
    """When both tokens are present, the louder mxfp4 warning wins."""
    # Synthetic — no real model carries both, but the priority matters.
    assert g._detect_quant_format("vendor/weird-mxfp4-nvfp4") == "mxfp4"


# ---------------------------------------------------------------------------
# check_load_time_guardrails — the 3-flag matrix
# ---------------------------------------------------------------------------


def _signal(is_moe, quant, world):
    return g.GuardrailSignal(
        is_moe=is_moe,
        quant_format=quant,
        distributed_world_size=world,
    )


def test_three_tuple_fires_mxfp4_moe_distributed_warning(caplog):
    """The full 3-of-3 match fires the cliff warning and bumps the counter."""
    caplog.set_level(logging.WARNING, logger=g.logger.name)
    fired = g.check_load_time_guardrails(
        _signal(is_moe=True, quant="mxfp4", world=2),
        hf_path="vendor/some-moe-mxfp4",
        alias="some-alias",
    )
    assert fired == ["mxfp4_moe_distributed"]
    assert g.snapshot()["mxfp4_moe_distributed_warnings_total"] == 1
    assert g.snapshot()["nvfp4_moe_warnings_total"] == 0
    # Warning carries the upstream issue link so operators can self-route.
    joined = " ".join(r.message for r in caplog.records)
    assert g.MLX_3402_URL in joined
    assert "mxfp4" in joined.lower() or "MXFP4" in joined


@pytest.mark.parametrize(
    "is_moe,quant,world,label",
    [
        # 2-of-3 misses: MoE off
        (False, "mxfp4", 4, "no_moe"),
        # 2-of-3 misses: quant off
        (True, None, 4, "no_mxfp4"),
        (True, "int4", 4, "wrong_quant"),
        # 2-of-3 misses: single-device
        (True, "mxfp4", 1, "single_device"),
        # 1-of-3 / 0-of-3
        (False, None, 1, "none"),
        (False, "mxfp4", 1, "mxfp4_only"),
        (True, None, 1, "moe_only"),
        (False, None, 4, "distributed_only"),
    ],
)
def test_two_of_three_does_not_fire_cliff(is_moe, quant, world, label):
    """Any 2-of-3 combo stays silent for the cliff guardrail."""
    fired = g.check_load_time_guardrails(
        _signal(is_moe=is_moe, quant=quant, world=world),
        hf_path=f"vendor/{label}",
        alias=label,
    )
    # The MoE+NVFP4 guardrail is independent — it must not fire for any
    # of these cases either (none of them carry nvfp4).
    assert "mxfp4_moe_distributed" not in fired
    assert "nvfp4_moe" not in fired
    assert g.snapshot()["mxfp4_moe_distributed_warnings_total"] == 0
    assert g.snapshot()["nvfp4_moe_warnings_total"] == 0


def test_nvfp4_moe_fires_regardless_of_device_count(caplog):
    """mlx#2962 dynamic-range loss bites even single-device."""
    caplog.set_level(logging.WARNING, logger=g.logger.name)
    for world in (1, 2, 8):
        g.reset_for_tests()
        caplog.clear()
        fired = g.check_load_time_guardrails(
            _signal(is_moe=True, quant="nvfp4", world=world),
            hf_path="vendor/nvfp4-moe",
            alias="alias-x",
        )
        assert fired == ["nvfp4_moe"], f"world={world}"
        assert g.snapshot()["nvfp4_moe_warnings_total"] == 1
        joined = " ".join(r.message for r in caplog.records)
        assert g.MLX_2962_URL in joined


def test_nvfp4_without_moe_silent():
    """NVFP4 + dense (non-MoE) does not trip the guardrail."""
    fired = g.check_load_time_guardrails(
        _signal(is_moe=False, quant="nvfp4", world=4),
        hf_path="vendor/dense-nvfp4",
        alias=None,
    )
    assert fired == []
    assert g.snapshot()["nvfp4_moe_warnings_total"] == 0


def test_counters_monotonic_across_repeated_fires():
    """Counters must accumulate across repeated three-tuple loads."""
    for _ in range(3):
        g.check_load_time_guardrails(
            _signal(is_moe=True, quant="mxfp4", world=2),
            hf_path="vendor/x",
            alias="x",
        )
    assert g.snapshot()["mxfp4_moe_distributed_warnings_total"] == 3


# ---------------------------------------------------------------------------
# _detect_distributed_world_size — env-var-based, non-mutating
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_var",
    ["MLX_WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE"],
)
def test_world_size_reads_launcher_env_vars(monkeypatch, env_var):
    """Each launcher env var on its own should be enough to report > 1."""
    # Make sure none of the *other* launcher vars leak through from the
    # test runner's environment.
    for v in (
        "MLX_WORLD_SIZE",
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "MLX_HOSTFILE",
    ):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv(env_var, "8")
    assert g._detect_distributed_world_size() == 8


def test_world_size_defaults_to_one_when_no_env(monkeypatch):
    """No launcher vars set → assume single-device (size 1)."""
    for v in (
        "MLX_WORLD_SIZE",
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "MLX_HOSTFILE",
    ):
        monkeypatch.delenv(v, raising=False)
    assert g._detect_distributed_world_size() == 1


def test_world_size_ignores_garbage(monkeypatch):
    """Non-integer values fall through to the next var / default."""
    for v in (
        "MLX_WORLD_SIZE",
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "MLX_HOSTFILE",
    ):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("MLX_WORLD_SIZE", "not-an-int")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    assert g._detect_distributed_world_size() == 4


def test_world_size_from_mlx_hostfile_ring_backend(monkeypatch, tmp_path):
    """Ring backend (Apple Silicon default) — count entries in MLX_HOSTFILE.

    The ring backend is the precise target of mlx#3402. Upstream
    ``mlx/distributed_run.py`` sets ``MLX_HOSTFILE`` (path to a JSON
    array of host descriptors) but NOT ``MLX_WORLD_SIZE`` on the ring
    path. The guardrail must still report world_size > 1 here.
    """
    import json

    for v in (
        "MLX_WORLD_SIZE",
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "MLX_HOSTFILE",
    ):
        monkeypatch.delenv(v, raising=False)
    # Match the upstream hostfile format from
    # ``mlx.distributed_run.parse_hostfile``: a JSON array of
    # ``{ssh, ips}`` host objects.
    hostfile = tmp_path / "hosts.json"
    hostfile.write_text(
        json.dumps(
            [
                {"ssh": "node-0", "ips": ["10.0.0.1"]},
                {"ssh": "node-1", "ips": ["10.0.0.2"]},
                {"ssh": "node-2", "ips": ["10.0.0.3"]},
            ]
        )
    )
    monkeypatch.setenv("MLX_HOSTFILE", str(hostfile))
    assert g._detect_distributed_world_size() == 3


def test_world_size_from_mlx_hostfile_single_host(monkeypatch, tmp_path):
    """A single-host hostfile reports 1 — not multi-device."""
    import json

    for v in (
        "MLX_WORLD_SIZE",
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "MLX_HOSTFILE",
    ):
        monkeypatch.delenv(v, raising=False)
    hostfile = tmp_path / "hosts.json"
    hostfile.write_text(json.dumps([{"ssh": "node-0", "ips": ["10.0.0.1"]}]))
    monkeypatch.setenv("MLX_HOSTFILE", str(hostfile))
    assert g._detect_distributed_world_size() == 1


def test_world_size_unreadable_hostfile_falls_through(monkeypatch):
    """A garbage / missing hostfile path is silently treated as size 1."""
    for v in (
        "MLX_WORLD_SIZE",
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "MLX_HOSTFILE",
    ):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("MLX_HOSTFILE", "/nonexistent/path/that/should/not/be/here")
    assert g._detect_distributed_world_size() == 1


def test_ring_backend_three_tuple_fires_e2e(monkeypatch, tmp_path, caplog):
    """End-to-end: ring backend hostfile + MoE + MXFP4 → cliff warning.

    This is the exact scenario mlx#3402 documents (M3 Ultra distributed
    GLM-5.1 / DeepSeek-V3.2), so it's the most important guardrail case
    to cover end-to-end through the adapter.
    """
    import json
    import logging

    caplog.set_level(logging.WARNING, logger=g.logger.name)
    for v in (
        "MLX_WORLD_SIZE",
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "MLX_HOSTFILE",
    ):
        monkeypatch.delenv(v, raising=False)
    hostfile = tmp_path / "hosts.json"
    hostfile.write_text(
        json.dumps(
            [
                {"ssh": "node-0", "ips": ["10.0.0.1"]},
                {"ssh": "node-1", "ips": ["10.0.0.2"]},
            ]
        )
    )
    monkeypatch.setenv("MLX_HOSTFILE", str(hostfile))

    class _RingProfile:
        # Inline stand-in for AliasProfile so this test stays
        # self-contained (the broader _FakeProfile is defined below).
        hf_path = "nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx"
        is_moe = True

    fired = g.check_from_profile(
        model_name="qwen3.5-122b-mxfp4",
        profile=_RingProfile(),
        alias="qwen3.5-122b-mxfp4",
    )
    assert fired == ["mxfp4_moe_distributed"]
    joined = " ".join(r.message for r in caplog.records)
    assert "world_size=2" in joined


def test_guardrail_module_never_references_mx_distributed_init():
    """Codex #297 BLOCKING regression: the guardrail must NOT call init().

    ``mx.distributed.init()`` creates the global communication group and
    can block while a backend handshakes — calling it from a warning-only
    guardrail would be a real side effect on the engine that follows.

    We pin the contract by **static source inspection** rather than a
    runtime monkeypatch on the live ``mlx.core`` module. Importing
    ``mlx.core`` purely for the patch reintroduced a heavy MLX import
    into an otherwise pure guardrail test (codex round 5); on a host
    with MLX installed but no usable Metal device, ``import mlx.core``
    can abort the process with ``NSRangeException``. Reading the
    source text instead is cheap, robust across hosts, and
    semantically equivalent to "the code doesn't call init()".
    """
    import inspect
    import re

    src = inspect.getsource(g)
    # Strip comments and docstrings: triple-quoted strings + ``# ...``
    # tails. The contract is "no executable call to .init(", so
    # documentation mentioning ``mx.distributed.init()`` (the entire
    # raison d'être of the guardrail) must not trip the check.
    stripped = re.sub(r'"""[\s\S]*?"""', "", src)
    stripped = re.sub(r"#.*", "", stripped)
    # The forbidden pattern is any executable call to ``.init(`` on
    # mx.distributed / mlx.core.distributed. Plain ``distributed.init``
    # without an open paren is harmless (it's just an attribute ref
    # passed to e.g. monkeypatch.setattr in test scaffolding, which
    # this production module never does).
    forbidden = re.compile(r"distributed\s*\.\s*init\s*\(")
    matches = forbidden.findall(stripped)
    assert matches == [], (
        f"Guardrail module must not call mx.distributed.init() "
        f"(see codex review on #297 BLOCKING #1) — found "
        f"{len(matches)} match(es) in source after docstring/comment "
        f"strip."
    )


# ---------------------------------------------------------------------------
# check_from_profile — adapter from server.load_model() call site
# ---------------------------------------------------------------------------


class _FakeProfile:
    """Minimal stand-in for AliasProfile — just is_moe + hf_path."""

    def __init__(self, *, hf_path, is_moe):
        self.hf_path = hf_path
        self.is_moe = is_moe


def test_check_from_profile_three_tuple(monkeypatch, caplog):
    """Adapter path: AliasProfile + mocked distributed size triggers the cliff."""
    caplog.set_level(logging.WARNING, logger=g.logger.name)
    monkeypatch.setattr(g, "_detect_distributed_world_size", lambda: 4)
    profile = _FakeProfile(
        hf_path="nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx",
        is_moe=True,
    )
    fired = g.check_from_profile(
        model_name="qwen3.5-122b-mxfp4",
        profile=profile,
        alias="qwen3.5-122b-mxfp4",
    )
    assert fired == ["mxfp4_moe_distributed"]
    assert g.snapshot()["mxfp4_moe_distributed_warnings_total"] == 1


def test_check_from_profile_single_device_silent(monkeypatch):
    """Adapter path: single-device default (world=1) stays silent on MoE+MXFP4."""
    monkeypatch.setattr(g, "_detect_distributed_world_size", lambda: 1)
    profile = _FakeProfile(
        hf_path="nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx",
        is_moe=True,
    )
    fired = g.check_from_profile(
        model_name="qwen3.5-122b-mxfp4",
        profile=profile,
    )
    assert fired == []
    assert g.snapshot()["mxfp4_moe_distributed_warnings_total"] == 0


def test_check_from_profile_no_profile_treated_as_non_moe(monkeypatch):
    """A bare HF path with no alias is conservatively treated as non-MoE."""
    monkeypatch.setattr(g, "_detect_distributed_world_size", lambda: 4)
    # Path carries mxfp4 + we're distributed, but profile=None → no is_moe
    # signal, so the guardrail stays silent. The conservative bias is
    # documented in the module docstring.
    fired = g.check_from_profile(
        model_name="some/unknown-mxfp4-model",
        profile=None,
    )
    assert fired == []


def test_check_from_profile_nvfp4_moe(monkeypatch, caplog):
    """Adapter path: NVFP4 + MoE fires regardless of mocked device count."""
    caplog.set_level(logging.WARNING, logger=g.logger.name)
    monkeypatch.setattr(g, "_detect_distributed_world_size", lambda: 1)
    profile = _FakeProfile(
        hf_path="vendor/some-nvfp4-moe",
        is_moe=True,
    )
    fired = g.check_from_profile(
        model_name="vendor/some-nvfp4-moe",
        profile=profile,
    )
    assert fired == ["nvfp4_moe"]


# ---------------------------------------------------------------------------
# /metrics rendering — counters surface in Prometheus output
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Real-aliases regression: every MXFP4/NVFP4 alias must carry is_moe so the
# guardrail actually fires in production (codex round 6 BLOCKING).
# ---------------------------------------------------------------------------


def test_all_mxfp4_aliases_carry_is_moe_metadata():
    """Every MXFP4 alias in ``aliases.json`` MUST have ``is_moe: true``.

    Codex round 6 BLOCKING: ``gpt-oss-20b-mxfp4-q8`` and
    ``minimax-m2.7-mxfp4`` were shipped with implicit ``is_moe=false``
    (the field defaulted) even though both are MoE architectures
    (gpt-oss is the OpenAI mixture-of-experts family; MiniMax-Text is
    ~456B total / ~46B active). With ``is_moe=false`` the guardrail
    returns BEFORE the warning, leaving production alerts inert on the
    exact aliases the cliff most affects.

    This test walks every alias whose HF path embeds the ``mxfp4``
    token (case-insensitive) and asserts ``is_moe=True`` is wired
    through ``resolve_profile`` — the same code path
    ``server.load_model()`` uses at runtime. Any future MXFP4 alias
    added without ``is_moe: true`` will trip this test, so the
    guardrail can never silently regress to inert again.
    """
    from vllm_mlx.model_aliases import list_profiles, resolve_profile

    mxfp4_aliases: list[str] = []
    for alias, profile in list_profiles().items():
        if "mxfp4" in profile.hf_path.lower():
            mxfp4_aliases.append(alias)

    # Defensive — if the alias registry no longer carries any MXFP4
    # entries (all retired), the test would silently pass. Pin a floor
    # so the test still catches the regression next time someone adds
    # an MXFP4 alias.
    assert mxfp4_aliases, (
        "Expected at least one MXFP4 alias in the registry; if all are "
        "retired, remove this test along with the rest of the guardrail."
    )

    for alias in mxfp4_aliases:
        profile = resolve_profile(alias)
        assert profile is not None
        assert profile.is_moe, (
            f"alias {alias!r} (hf_path={profile.hf_path!r}) is an MXFP4 "
            f"variant but ``is_moe=False`` — the mlx#3402 guardrail will "
            f"NEVER fire for this alias. Set ``is_moe: true`` in "
            f"aliases.json."
        )


def test_all_nvfp4_aliases_carry_is_moe_metadata():
    """Every NVFP4 alias must also carry ``is_moe: true`` for mlx#2962.

    Symmetric counterpart to the MXFP4 test. There are currently no
    NVFP4 aliases shipped, but a future one added without ``is_moe``
    would silently bypass the dynamic-range-loss warning.
    """
    from vllm_mlx.model_aliases import list_profiles, resolve_profile

    nvfp4_aliases = [
        alias
        for alias, profile in list_profiles().items()
        if "nvfp4" in profile.hf_path.lower()
    ]

    for alias in nvfp4_aliases:
        profile = resolve_profile(alias)
        assert profile is not None
        assert profile.is_moe, (
            f"alias {alias!r} (hf_path={profile.hf_path!r}) is an NVFP4 "
            f"variant but ``is_moe=False`` — the mlx#2962 guardrail will "
            f"NEVER fire. Set ``is_moe: true`` in aliases.json."
        )


def test_render_prometheus_lines_exposes_both_counters():
    """The pure render helper emits HELP/TYPE/sample for both counters.

    This test deliberately calls ``render_prometheus_lines()`` directly
    instead of going through ``vllm_mlx.routes.metrics``. The route
    module's transitive import closure pulls in ``vllm_mlx.config`` →
    ``BaseEngine`` → the engine stack, which codex rounds 3/4 flagged
    as too heavy for a focused unit test. The new pure helper lives
    in the guardrail module so a single ``import`` is enough.
    """
    # Trip both counters once each so the rendered body shows non-zero
    # values rather than just the HELP/TYPE lines.
    g.check_load_time_guardrails(
        _signal(is_moe=True, quant="mxfp4", world=2),
        hf_path="vendor/a",
        alias="a",
    )
    g.check_load_time_guardrails(
        _signal(is_moe=True, quant="nvfp4", world=1),
        hf_path="vendor/b",
        alias="b",
    )

    lines = g.render_prometheus_lines()
    body = "\n".join(lines)

    assert "rapid_mlx_mxfp4_moe_distributed_warnings_total" in body
    assert "rapid_mlx_nvfp4_moe_warnings_total" in body
    # Counter type discoverable via TYPE line — required by Prometheus
    # exposition format spec.
    assert (
        "# TYPE rapid_mlx_mxfp4_moe_distributed_warnings_total counter" in body
    )
    assert "# TYPE rapid_mlx_nvfp4_moe_warnings_total counter" in body
    # Values were both bumped to 1.
    assert "rapid_mlx_mxfp4_moe_distributed_warnings_total 1" in body
    assert "rapid_mlx_nvfp4_moe_warnings_total 1" in body


def test_render_prometheus_lines_zero_state():
    """At fresh state both counters render as 0 (Prometheus contract).

    Operators alerting on the cliff need the metric series visible
    from process startup, BEFORE any guardrail has fired. The pure
    renderer must always emit both series with at-least-zero values.
    """
    # Counters reset by the autouse fixture, so call render directly.
    lines = g.render_prometheus_lines()
    body = "\n".join(lines)
    assert "rapid_mlx_mxfp4_moe_distributed_warnings_total 0" in body
    assert "rapid_mlx_nvfp4_moe_warnings_total 0" in body
