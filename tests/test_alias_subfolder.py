# SPDX-License-Identifier: Apache-2.0
"""``subfolder``: one repo, one folder per quantization.

Liquid AI ships ``LiquidAI/LFM2.5-2.6B-MLX`` as eight complete MLX
checkpoints in sibling directories (``4bit/``, ``5bit/``, ``6bit/``,
``8bit/``, ``bf16/``, ``mxfp4/``, ``mxfp8/``, ``nvfp4/``) instead of one
repo per quant, and there is no flat conversion on the Hub. ``mlx_lm.load``
has no subfolder parameter, so the repo id has to become a concrete
directory somewhere. These tests pin *where* — and, just as importantly,
pin that it does NOT happen anywhere else: the download gate, the R2
mirror catalog, ``model_sizes`` and telemetry all key on the bare repo id.

mlx-free by construction (no weights, no ``mlx`` import) so it runs on the
Linux CI leg as well as Apple Silicon.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vllm_mlx.model_aliases import (
    _coerce,
    resolve_model,
    resolve_profile,
    resolve_subfolder,
    subfolder_allow_patterns,
)
from vllm_mlx.utils.tokenizer import _resolve_subfolder_checkpoint

ALIAS = "lfm2.5-2.6b-4bit"
REPO = "LiquidAI/LFM2.5-2.6B-MLX"


# --------------------------------------------------------------------------
# The shipped alias
# --------------------------------------------------------------------------


def test_alias_declares_the_4bit_subfolder():
    profile = resolve_profile(ALIAS)
    assert profile is not None, f"{ALIAS} missing from aliases.json"
    assert profile.hf_path == REPO
    assert profile.subfolder == "4bit"


def test_hf_path_stays_a_bare_repo_id():
    """The subfolder must NOT be folded into ``hf_path``.

    ``resolve_model`` feeds the download gate, ``model_sizes``, the mirror
    catalog and telemetry redaction — all of which treat the value as an
    ``org/name`` repo id. A three-segment path would 404 against the HF
    API and silently miss the mirror.
    """
    resolved = resolve_model(ALIAS)
    assert resolved == REPO
    assert resolved.count("/") == 1


def test_most_aliases_have_no_subfolder():
    """Guard against a copy-paste that pins a subfolder repo-wide."""
    from vllm_mlx.model_aliases import list_profiles

    with_subfolder = {
        alias for alias, p in list_profiles().items() if p.subfolder is not None
    }
    assert with_subfolder == {ALIAS}, (
        "A new subfolder alias landed without updating this test. That is "
        "fine — but confirm the download path passes allow_patterns for it, "
        "or the pull fetches every quant in the repo."
    )


# --------------------------------------------------------------------------
# Schema validation — subfolder becomes a filesystem path, so it is
# validated as a relative downward one at registry load.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "/etc",  # absolute
        "../../etc",  # escapes the snapshot
        "4bit/../../../etc",  # escapes mid-path
        "4bit\\weights",  # backslash separator
        "C:/Users/x",  # drive-qualified — os.path.isabs is False on POSIX
        "c:weights",  # drive-relative, same discard behaviour on Windows
        # Legal relative paths, but the value is also spliced into an HF
        # allow_patterns glob — each of these widens the download past the
        # one folder the alias declares.
        "4bit*",  # also matches 4bit-extra/, 4bitfoo/
        "[48]bit",  # matches both 4bit/ and 8bit/
        "**",  # the whole repo
        "?bit",
        "!4bit",
        "4bit/",  # trailing slash → join produces a stray separator
        "",  # empty
        123,  # not a string
    ],
)
def test_rejects_paths_that_leave_the_snapshot(bad):
    with pytest.raises(ValueError, match="subfolder"):
        _coerce("evil", {"hf_path": "org/repo", "subfolder": bad})


@pytest.mark.parametrize("good", ["4bit", "mxfp4", "quants/4bit"])
def test_accepts_a_relative_downward_path(good):
    assert _coerce("ok", {"hf_path": "org/repo", "subfolder": good}).subfolder == good


def test_absent_subfolder_is_none_not_empty_string():
    """``None`` and ``""`` must not both mean "no subfolder" — the loader
    branches on truthiness, but ``resolve_subfolder`` is also compared for
    equality in the ambiguity guard."""
    assert _coerce("plain", {"hf_path": "org/repo"}).subfolder is None


# --------------------------------------------------------------------------
# The ambiguity guard: two quants of one repo can't both be recovered
# --------------------------------------------------------------------------


def test_two_aliases_on_one_repo_with_different_subfolders_is_rejected():
    """The loader recovers the subfolder by ``hf_path → first alias``.

    Two aliases on one repo agree about every other profile field, so
    "first wins" is harmless there. ``subfolder`` is the exception: 4-bit
    and 8-bit legitimately disagree, and picking the first would serve the
    wrong weights under the right name. Fail at registry load instead.
    """
    from vllm_mlx.model_aliases import _assert_subfolder_is_unambiguous

    profiles = {
        "m-4bit": _coerce("m-4bit", {"hf_path": REPO, "subfolder": "4bit"}),
        "m-8bit": _coerce("m-8bit", {"hf_path": REPO, "subfolder": "8bit"}),
    }
    with pytest.raises(ValueError, match="different 'subfolder' values"):
        _assert_subfolder_is_unambiguous(profiles)


def test_two_aliases_on_one_repo_agreeing_is_fine():
    from vllm_mlx.model_aliases import _assert_subfolder_is_unambiguous

    profiles = {
        "a": _coerce("a", {"hf_path": REPO, "subfolder": "4bit"}),
        "b": _coerce("b", {"hf_path": REPO, "subfolder": "4bit"}),
    }
    _assert_subfolder_is_unambiguous(profiles)  # does not raise


def test_shipped_registry_is_unambiguous():
    """The real aliases.json, not a fixture."""
    from vllm_mlx.model_aliases import _assert_subfolder_is_unambiguous, list_profiles

    _assert_subfolder_is_unambiguous(list_profiles())


# --------------------------------------------------------------------------
# Reverse lookup — by the time the text lane loads, the alias the user
# typed is long gone and only the resolved repo id remains.
# --------------------------------------------------------------------------


def test_subfolder_is_recoverable_from_the_resolved_repo_id():
    assert resolve_subfolder(REPO) == "4bit"
    assert resolve_subfolder(ALIAS) == "4bit"


def test_unknown_and_flat_models_report_no_subfolder():
    assert resolve_subfolder("mlx-community/LFM2.5-8B-A1B-MLX-4bit") is None
    assert resolve_subfolder("some/model-nobody-registered") is None


# --------------------------------------------------------------------------
# Download filtering — the whole point of the field is not paying for the
# seven quants you didn't ask for.
# --------------------------------------------------------------------------


def test_allow_patterns_fetch_only_the_declared_folder():
    assert subfolder_allow_patterns(ALIAS) == ["4bit/*"]
    assert subfolder_allow_patterns(REPO) == ["4bit/*"]


def test_allow_patterns_is_none_for_flat_repos():
    """``None``, not ``["*"]`` — callers branch on it to keep the
    historical single-argument ``snapshot_download(repo)`` call shape."""
    assert subfolder_allow_patterns("mlx-community/LFM2.5-8B-A1B-MLX-4bit") is None


def test_allow_pattern_prefix_matches_the_real_repo_layout():
    """The pattern is also sliced to a prefix for the size estimate; a
    ``4bit/*`` glob and a ``4bit/`` prefix must select the same files."""
    patterns = subfolder_allow_patterns(ALIAS)
    prefix = patterns[0][:-1]
    assert prefix == "4bit/"
    for name in ("4bit/config.json", "4bit/model.safetensors"):
        assert name.startswith(prefix)
    for name in ("8bit/model.safetensors", "README.md", "bf16/config.json"):
        assert not name.startswith(prefix)


# --------------------------------------------------------------------------
# The load-time join — the one place a repo id becomes a directory.
# --------------------------------------------------------------------------


def test_flat_repo_id_passes_through_untouched():
    name = "mlx-community/LFM2.5-8B-A1B-MLX-4bit"
    assert _resolve_subfolder_checkpoint(name) == name


def test_local_path_passes_through_even_if_its_name_matches_an_alias(
    tmp_path, monkeypatch
):
    """A local checkpoint directory is already the thing to load. Deriving
    a subfolder for it from a reverse alias lookup would append a folder
    that isn't there.

    The directory is named EXACTLY the alias and referred to by that
    relative name, so the registry lookup genuinely fires and only the
    ``os.path.exists`` guard can save it. A tmp path that matches no alias
    would pass with the guard deleted, testing nothing.
    """
    local = tmp_path / ALIAS
    local.mkdir()
    (local / "config.json").write_text("{}")
    monkeypatch.chdir(tmp_path)

    assert resolve_subfolder(ALIAS) == "4bit", "precondition: name IS an alias"
    assert os.path.exists(ALIAS), "precondition: it is also a real directory"
    assert _resolve_subfolder_checkpoint(ALIAS) == ALIAS


def test_joins_the_subfolder_onto_the_snapshot(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshots" / "deadbeef"
    (snapshot / "4bit").mkdir(parents=True)
    (snapshot / "4bit" / "config.json").write_text("{}")
    # Complete checkpoint: the returned directory is validated against
    # mlx-lm's own ``model*.safetensors`` glob before it is handed over.
    (snapshot / "4bit" / "model.safetensors").write_bytes(b"\x00" * 16)

    seen: dict[str, object] = {}

    def fake_snapshot_download(repo_id, **kwargs):
        seen["repo_id"] = repo_id
        seen["allow_patterns"] = kwargs.get("allow_patterns")
        return str(snapshot)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    out = _resolve_subfolder_checkpoint(REPO)

    assert out == os.path.join(str(snapshot), "4bit")
    assert seen["repo_id"] == REPO, "must download the repo, not repo/subfolder"
    assert seen["allow_patterns"] == ["4bit/*"], (
        "without the filter this pulls all eight quantizations (~20 GB)"
    )
    assert json.loads(Path(out, "config.json").read_text()) == {}


def test_download_failure_raises_instead_of_returning_the_repo_id(monkeypatch):
    """Returning the bare repo id would send mlx-lm to its own UNFILTERED
    snapshot_download: the user waits out ~20 GB and then fails anyway,
    because the repo root is not a checkpoint. Fail immediately instead."""
    import huggingface_hub

    def boom(repo_id, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)
    with pytest.raises(RuntimeError, match="not in the local cache"):
        _resolve_subfolder_checkpoint(REPO)


def test_warm_complete_subfolder_never_touches_the_network(monkeypatch, tmp_path):
    """A warm, COMPLETE cache resolves offline-first with zero network.

    The online ``snapshot_download`` used to run first; on a poisoned-DNS
    network it hangs in SYN_SENT indefinitely rather than raising, so the
    cached fallback (reached only inside ``except``) never ran and an
    already-downloaded subfolder sat at "Starting" until the outer deadline.
    A verified-complete on-disk checkpoint must now short-circuit before any
    networked call is made.
    """
    import huggingface_hub

    snapshot = tmp_path / "snap"
    ckpt = snapshot / "4bit"
    ckpt.mkdir(parents=True)
    # A real cached checkpoint, not just an empty directory — an empty one
    # would satisfy the isdir() check while being unloadable, so the test
    # would pass on a path that cannot actually serve.
    (ckpt / "config.json").write_text('{"model_type": "lfm2"}')
    (ckpt / "model.safetensors").write_bytes(b"\x00" * 16)

    calls: list[bool] = []

    def offline_first(repo_id, **kwargs):
        local_only = kwargs.get("local_files_only", False)
        calls.append(local_only)
        if not local_only:
            raise AssertionError(
                "warm complete cache must resolve offline; the network call "
                "is exactly what hangs on a poisoned-DNS start"
            )
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", offline_first)

    resolved = _resolve_subfolder_checkpoint(REPO)
    assert resolved == str(ckpt)
    assert calls == [True], "offline-first: the cache is tried before the Hub"
    # What the loader is handed must be openable: mlx-lm globs
    # ``model*.safetensors`` at the directory it is given, non-recursively.
    assert Path(resolved, "config.json").exists()
    assert list(Path(resolved).glob("model*.safetensors")), (
        "the returned directory must be where mlx-lm's loader glob will hit"
    )


def test_incomplete_cached_subfolder_falls_through_to_the_hub(monkeypatch, tmp_path):
    """A half-pulled cache must NOT short-circuit — it still hits the Hub to
    finish the download, rather than loading a shard-less checkpoint that
    would fail (or render garbage). Only a *complete* offline resolve wins."""
    import huggingface_hub

    partial = tmp_path / "partial"
    complete = tmp_path / "complete"
    # Offline cache: subfolder present but NO weight shards → incomplete.
    (partial / "4bit").mkdir(parents=True)
    (partial / "4bit" / "config.json").write_text('{"model_type": "lfm2"}')
    # What the Hub returns once the pull finishes.
    (complete / "4bit").mkdir(parents=True)
    (complete / "4bit" / "config.json").write_text('{"model_type": "lfm2"}')
    (complete / "4bit" / "model.safetensors").write_bytes(b"\x00" * 16)

    calls: list[bool] = []

    def offline_incomplete_then_online(repo_id, **kwargs):
        local_only = kwargs.get("local_files_only", False)
        calls.append(local_only)
        return str(partial) if local_only else str(complete)

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", offline_incomplete_then_online
    )

    resolved = _resolve_subfolder_checkpoint(REPO)
    assert resolved == str(complete / "4bit")
    assert calls == [True, False], (
        "incomplete offline resolve must fall through to the networked pull"
    )


def test_flat_repos_are_untouched_by_the_strict_path(monkeypatch):
    """The strictness applies only to subfolder aliases — a flat repo must
    never reach snapshot_download here at all."""
    import huggingface_hub

    def tripwire(repo_id, **kwargs):
        raise AssertionError("flat repo must not be downloaded by this helper")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", tripwire)
    name = "mlx-community/LFM2.5-8B-A1B-MLX-4bit"
    assert _resolve_subfolder_checkpoint(name) == name


def test_missing_subfolder_after_download_raises(monkeypatch, tmp_path):
    """Publisher renamed or dropped the folder. Say so — silently loading
    the repo root would either fail obscurely or, worse, succeed against
    some other checkpoint that happens to sit there."""
    snapshot = tmp_path / "snap"
    snapshot.mkdir()

    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda repo_id, **kw: str(snapshot)
    )
    with pytest.raises(RuntimeError, match="reorganized the repo"):
        _resolve_subfolder_checkpoint(REPO)


# --------------------------------------------------------------------------
# Everything that reaches INTO the checkpoint. Each of these was found by
# running ``rapid-mlx serve lfm2.5-2.6b-4bit`` and watching it fail a
# different way — the unit tests above all passed while the command was
# still unusable, so these pin the specific symptoms.
# --------------------------------------------------------------------------


def test_prefix_is_empty_for_flat_repos_so_callers_can_prepend_blindly():
    from vllm_mlx.model_aliases import checkpoint_prefix

    assert checkpoint_prefix(REPO) == "4bit/"
    assert checkpoint_prefix("mlx-community/LFM2.5-8B-A1B-MLX-4bit") == ""
    assert checkpoint_prefix("nobody/registered-this") == ""
    # The whole point of "" over None: ``f"{prefix}config.json"`` is
    # correct for both kinds of repo with no branch at the call site.
    assert f"{checkpoint_prefix('nobody/registered-this')}config.json" == "config.json"


def test_metadata_read_looks_inside_the_checkpoint(monkeypatch, tmp_path):
    """Symptom: ``Could not materialize the checkpoint config ... before
    selecting the serving lane`` — a hard startup failure. ``config.json``
    is at ``4bit/config.json``, so the root probe found nothing and the
    MLLM-vs-text routing had no evidence to work from."""
    import vllm_mlx.model_metadata as mm

    snapshot = tmp_path / "snapshots" / "cafe"
    (snapshot / "4bit").mkdir(parents=True)
    (snapshot / "4bit" / "config.json").write_text('{"model_type": "lfm2"}')

    asked: list[str] = []

    def fake_cached(repo_id, filename, **kwargs):
        asked.append(filename)
        target = snapshot / filename
        return target if target.exists() else None

    monkeypatch.setattr(mm, "_cached_file", lambda name, fn: fake_cached(name, fn))

    meta = mm.read_cached_model_metadata(REPO)

    assert asked[0] == "4bit/config.json", (
        f"probed {asked[0]!r} — the repo root has no config.json"
    )
    assert meta is not None
    assert meta.config == {"model_type": "lfm2"}
    assert meta.snapshot_dir.name == "4bit", (
        "snapshot_dir must be the checkpoint directory, not the repo root — "
        "everything downstream resolves sibling files against it"
    )


def test_size_estimate_counts_only_the_alias_checkpoint(monkeypatch):
    """Symptom: ``Estimated size: 18.7 GiB`` on a 1.6 GB model, then a
    kernel-panic warning and a confirm prompt. The repo holds eight
    complete quantizations; seven of them are not being downloaded."""
    import vllm_mlx._download_gate as gate

    class Sib:
        def __init__(self, name, size):
            self.rfilename, self.size, self.lfs = name, size, None

    info = type(
        "Info",
        (),
        {
            "siblings": [
                Sib("4bit/model.safetensors", 1_583_152_892),
                Sib("4bit/config.json", 2202),
                Sib("8bit/model.safetensors", 2_866_086_056),
                Sib("bf16/model-00001-of-00002.safetensors", 5_329_406_264),
                Sib("README.md", 2804),
            ]
        },
    )()
    monkeypatch.setattr(gate, "_model_info_with_timeout", lambda r, t: info)

    total = gate.estimate_repo_size_bytes(REPO)
    assert total == 1_583_152_892 + 2202
    assert total < 2 * 1024**3, "must not sum the repo's other quantizations"


def test_size_estimate_unchanged_for_flat_repos(monkeypatch):
    import vllm_mlx._download_gate as gate

    class Sib:
        def __init__(self, name, size):
            self.rfilename, self.size, self.lfs = name, size, None

    info = type(
        "Info",
        (),
        {"siblings": [Sib("model.safetensors", 100), Sib("config.json", 5)]},
    )()
    monkeypatch.setattr(gate, "_model_info_with_timeout", lambda r, t: info)
    assert gate.estimate_repo_size_bytes("mlx-community/LFM2.5-8B-A1B-MLX-4bit") == 105


def test_cache_probe_descends_before_asking_if_complete(tmp_path):
    """Symptom: the download gate re-prompts on every serve because a
    fully-cached checkpoint reads as absent — ``model*.safetensors`` is
    inside ``4bit/``, and the loader glob is non-recursive."""
    from vllm_mlx._download_gate import _descend_to_checkpoint

    snap = tmp_path / "sha"
    (snap / "4bit").mkdir(parents=True)
    assert _descend_to_checkpoint(str(snap), REPO) == str(snap / "4bit")
    # Flat repo: identity.
    assert _descend_to_checkpoint(
        str(snap), "mlx-community/LFM2.5-8B-A1B-MLX-4bit"
    ) == str(snap)


def test_cache_probe_does_not_invent_a_directory(tmp_path):
    """Publisher reorganised the repo: report the root (which will then
    read as incomplete) rather than a path that does not exist."""
    from vllm_mlx._download_gate import _descend_to_checkpoint

    snap = tmp_path / "sha"
    snap.mkdir()
    assert _descend_to_checkpoint(str(snap), REPO) == str(snap)


def test_mirror_declines_subfolder_repos(monkeypatch):
    """The R2 mirror hydrates whole repos and has no per-folder mode.

    Letting it run on a subfolder repo would pull all eight quantizations
    behind a progress bar that says "Pulling LiquidAI/LFM2.5-2.6B-MLX".
    None of these repos is mirrored today; this guard is what keeps
    mirroring one later from silently reintroducing the 20 GB pull.
    """
    import vllm_mlx.cli as cli

    called: list[str] = []

    def tripwire(*a, **kw):
        called.append("mirror ran")
        return True

    import vllm_mlx._mirror as mirror

    monkeypatch.setattr(mirror, "download_with_mirror_fallback", tripwire)

    assert cli._try_mirror_prefetch(REPO) is False
    assert called == [], "mirror must not be consulted for a subfolder repo"

    # Control: a flat repo still goes through the mirror.
    assert cli._try_mirror_prefetch("mlx-community/LFM2.5-8B-A1B-MLX-4bit") is True
    assert called == ["mirror ran"]


def test_registry_fails_closed_on_every_load_not_just_the_first(monkeypatch):
    """A caught validation error must not leave a usable registry behind.

    ``_load`` memoizes into module globals. Publishing before validating
    meant the first call raised, the caller caught it, and every call
    after that hit the memoized fast path and happily used the registry
    the assertion had just rejected — fail-open after a single swallowed
    exception.
    """
    import json as _json

    import vllm_mlx.model_aliases as ma

    ambiguous = {
        "m-4bit": {"hf_path": REPO, "subfolder": "4bit"},
        "m-8bit": {"hf_path": REPO, "subfolder": "8bit"},
    }
    monkeypatch.setattr(ma, "_aliases", None)
    monkeypatch.setattr(ma, "_hf_to_alias", None)
    monkeypatch.setattr(_json, "load", lambda fh: ambiguous)

    for attempt in range(3):
        with pytest.raises(ValueError, match="different 'subfolder' values"):
            ma._load()
        assert ma._aliases is None, (
            f"attempt {attempt}: registry was published despite failing validation"
        )


def test_subfolder_cannot_widen_the_download_glob():
    """The field's promise is "only this folder". A subfolder that is a
    legal path but an active glob quietly breaks that promise — and it is
    the download filter, not the path join, where it does damage."""
    import fnmatch

    patterns = subfolder_allow_patterns(ALIAS)
    assert patterns == ["4bit/*"]
    for other in ("8bit/model.safetensors", "bf16/config.json", "README.md"):
        assert not fnmatch.fnmatch(other, patterns[0]), (
            f"{other} must not match the 4-bit alias's allow_patterns"
        )


def test_alias_spelling_downloads_the_repo_not_the_alias(monkeypatch, tmp_path):
    """``server.load_model`` is a public entry point: a programmatic caller
    reaches it with the bare alias, skipping the CLI's pre-resolution.
    ``resolve_subfolder`` answers for both spellings, so the subfolder is
    detected either way — but the Hub only knows the repo id, and asking it
    for "lfm2.5-2.6b-4bit" is a 404."""
    import huggingface_hub

    snapshot = tmp_path / "snap"
    ckpt = snapshot / "4bit"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text("{}")
    (ckpt / "model.safetensors").write_bytes(b"\x00" * 16)
    asked: list[str] = []

    def fake(repo_id, **kwargs):
        asked.append(repo_id)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)

    out = _resolve_subfolder_checkpoint(ALIAS)
    assert out == os.path.join(str(snapshot), "4bit")
    assert asked == [REPO], f"asked the Hub for {asked} — the alias is not a repo id"


# --------------------------------------------------------------------------
# The reasoning_parser flip on lfm2.5-8b-a1b-4bit. It shipped as null while
# the model emits <think> and, at a 256-token budget, never closes it — so
# the raw tag reached users of our 16 GB "fast" recommendation. These pin
# the parser now declared for it against the shapes actually observed on an
# M2 Pro, including the unclosed one.
# --------------------------------------------------------------------------


def test_lfm_family_declares_a_reasoning_parser():
    """Both LFM2.5 aliases we have measured emit <think>; neither may ship
    with the parser left null again."""
    for alias in ("lfm2.5-2.6b-4bit", "lfm2.5-8b-a1b-4bit"):
        profile = resolve_profile(alias)
        assert profile is not None, alias
        assert profile.reasoning_parser == "qwen3", (
            f"{alias} emits <think> — a null parser leaks the raw tag into "
            "the user's chat window"
        )
        assert profile.tool_call_parser == "lfm"


def test_declared_parser_separates_the_output_these_models_actually_emit():
    from vllm_mlx.reasoning import get_parser

    parser = get_parser("qwen3")()

    # Shape 1 — lfm2.5-8b-a1b: emits the opening tag itself, closes it.
    reasoning, content = parser.extract_reasoning(
        "<think>Let me work this out. 17 * 23 = 391.</think>391"
    )
    assert content == "391"
    assert "391" in reasoning and "<think>" not in (content or "")

    # Shape 2 — lfm2.5-2.6b: the chat template pre-injects <think>, so the
    # completion starts INSIDE the reasoning block with no opening tag.
    reasoning, content = parser.extract_reasoning(
        "The user asks for a product. 17 * 20 = 340, plus 51.</think>391"
    )
    assert content == "391", "implicit-think mode must still yield clean content"
    assert "340" in reasoning


def test_unclosed_think_does_not_leak_into_content():
    """The 256-token observation: both models were still reasoning when the
    budget ran out, so NO closing tag was ever emitted. Whatever the split,
    a bare ``<think>`` must not reach the content field."""
    from vllm_mlx.reasoning import get_parser

    parser = get_parser("qwen3")()
    truncated = "<think>We need to produce exactly three sentences. The content"
    reasoning, content = parser.extract_reasoning(truncated)
    assert "<think>" not in (content or "")
    assert "</think>" not in (content or "")


def test_pull_narrows_to_the_checkpoint_and_says_so(monkeypatch, capsys, tmp_path):
    """``pull`` fetches only the served folder — for the bare repo id too.

    Every other consumer already reaches inside on the repo id: the gate
    quotes the subfolder's size, ``is_repo_cached`` probes it, the loader
    opens it. Pulling the whole repo would make that quote a lie and leave
    seven checkpoints on disk that nothing can serve. What the earlier
    version got wrong was doing it silently.
    """
    import argparse

    import vllm_mlx.cli as cli

    snapshot = tmp_path / "snap"
    (snapshot / "4bit").mkdir(parents=True)
    seen: dict = {}

    def fake_dl(rid, **kw):
        seen["repo"] = rid
        seen["allow"] = kw.get("allow_patterns")
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_dl)
    monkeypatch.setattr(cli, "_try_mirror_prefetch", lambda *a, **kw: False)

    cli.pull_command(argparse.Namespace(model=REPO, _original_alias=None))

    assert seen["allow"] == ["4bit/*"]
    out = capsys.readouterr().out
    assert "4bit/" in out and "one checkpoint per quantization" in out, (
        f"narrowing must be announced, got: {out!r}"
    )


def test_pull_of_a_flat_repo_is_unfiltered(monkeypatch, tmp_path):
    """The generic contract is untouched for every ordinary repo."""
    import argparse

    import vllm_mlx.cli as cli

    seen: dict = {}

    def fake_dl(rid, **kw):
        seen["allow"] = kw.get("allow_patterns", "ABSENT")
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_dl)
    monkeypatch.setattr(cli, "_try_mirror_prefetch", lambda *a, **kw: False)

    cli.pull_command(
        argparse.Namespace(
            model="mlx-community/LFM2.5-8B-A1B-MLX-4bit", _original_alias=None
        )
    )
    assert seen["allow"] == "ABSENT", (
        "flat repos must keep the historical single-argument call shape"
    )


def test_incomplete_cached_checkpoint_is_rejected(monkeypatch, tmp_path):
    """A directory is not a checkpoint.

    An earlier pull that ran out of disk leaves ``4bit/`` present with its
    shards missing. Returning that path produces a confusing mid-load
    failure inside mlx-lm instead of "your cache is incomplete"."""
    import huggingface_hub

    snapshot = tmp_path / "snap"
    (snapshot / "4bit").mkdir(parents=True)
    (snapshot / "4bit" / "config.json").write_text("{}")  # no weights

    def offline_then_cache(repo_id, **kwargs):
        if not kwargs.get("local_files_only", False):
            raise OSError("no network")
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", offline_then_cache)

    with pytest.raises(RuntimeError, match="incomplete"):
        _resolve_subfolder_checkpoint(REPO)


def test_disk_check_keeps_the_prefix_when_the_cache_probe_throws(monkeypatch):
    """The cache probe is best-effort and swallows its own failures. It must
    not take the size estimate down with it.

    Free space here (4 GB) fits the 1.6 GB checkpoint but not the repo's
    9.7 GB of quantizations. Before the fix, one flaky
    ``try_to_load_from_cache`` reset the prefix and this machine was told to
    free up disk for seven checkpoints it was never going to download.
    """
    import huggingface_hub

    import vllm_mlx.cli as cli

    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("cache unreadable")),
    )

    class Sib:
        def __init__(self, name, size):
            self.rfilename, self.size = name, size

    monkeypatch.setattr(
        huggingface_hub,
        "model_info",
        lambda repo_id, **kw: type(
            "I",
            (),
            {
                "siblings": [
                    Sib("4bit/model.safetensors", 1_600_000_000),
                    Sib("bf16/model.safetensors", 5_300_000_000),
                    Sib("8bit/model.safetensors", 2_800_000_000),
                ]
            },
        )(),
    )

    four_gb = 4 * 1000**3
    monkeypatch.setattr(
        cli.os,
        "statvfs",
        lambda p: type("S", (), {"f_bavail": four_gb // 4096, "f_frsize": 4096})(),
    )

    # Returns normally: only the 1.6 GB checkpoint is counted.
    cli._check_disk_space(REPO)

    # Control — a flat repo of the same total genuinely does not fit, and
    # the check must still refuse it. This is what proves the assertion
    # above is about the prefix and not about the check being inert.
    monkeypatch.setattr(
        huggingface_hub,
        "model_info",
        lambda repo_id, **kw: type(
            "I", (), {"siblings": [Sib("model.safetensors", 9_700_000_000)]}
        )(),
    )
    with pytest.raises(SystemExit):
        cli._check_disk_space("mlx-community/LFM2.5-8B-A1B-MLX-4bit")


def test_completeness_is_checked_on_the_success_path_too(monkeypatch, tmp_path):
    """A publisher who reorganizes the repo can leave ``config.json`` with no
    weights beside it. That arrives via a perfectly SUCCESSFUL download, so
    checking only after the offline fallback would let it through."""
    import huggingface_hub

    snapshot = tmp_path / "snap"
    (snapshot / "4bit").mkdir(parents=True)
    (snapshot / "4bit" / "config.json").write_text("{}")  # no shards

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda rid, **kw: str(snapshot)
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        _resolve_subfolder_checkpoint(REPO)


def test_local_snapshot_resolves_a_cached_repo_to_its_path_offline(monkeypatch):
    """A verified-complete cached repo is resolved to its local snapshot dir
    with ``local_files_only=True`` — so the loader gets a path and never
    round-trips. No process-global offline toggling, no network.

    A fail-fast fake asserts the resolve requests the cache (never the Hub) and
    that the returned local path is what the loader is handed.
    """
    import huggingface_hub

    from vllm_mlx.utils import tokenizer as tok

    calls: list[bool] = []

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append(kwargs.get("local_files_only", False))
        assert repo_id == "org/warm-repo"
        return "/cache/models--org--warm-repo/snapshots/abc"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr("vllm_mlx._download_gate.is_repo_cached", lambda _name: True)

    resolved = tok._local_snapshot_if_cached("org/warm-repo")
    assert resolved == "/cache/models--org--warm-repo/snapshots/abc"
    assert calls == [True], "must resolve from the cache, never the network"


def test_local_snapshot_leaves_a_cold_repo_untouched(monkeypatch):
    """A cold cache is NOT resolved locally — the bare repo id passes through so
    the loader's own online pull still runs. snapshot_download is a tripwire."""
    import huggingface_hub

    from vllm_mlx.utils import tokenizer as tok

    def tripwire(repo_id, **kwargs):
        raise AssertionError("a cold repo must not be resolved from the cache")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", tripwire)
    monkeypatch.setattr("vllm_mlx._download_gate.is_repo_cached", lambda _name: False)

    assert tok._local_snapshot_if_cached("org/cold-repo") == "org/cold-repo"


def test_local_snapshot_falls_back_when_the_local_resolve_fails(monkeypatch):
    """If the completeness probe says cached but the offline resolve raises
    (unexpected cache state), return the bare id so the normal path can still
    try — never propagate a resolve error out of a supposedly-warm start."""
    import huggingface_hub

    from vllm_mlx.utils import tokenizer as tok

    def boom(repo_id, **kwargs):
        raise OSError("cache surprised us")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)
    monkeypatch.setattr("vllm_mlx._download_gate.is_repo_cached", lambda _name: True)

    assert tok._local_snapshot_if_cached("org/warm-repo") == "org/warm-repo"
