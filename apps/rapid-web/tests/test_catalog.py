# SPDX-License-Identifier: Apache-2.0
"""Tests for the model catalog.

The `rapid-mlx` CLI is replaced by a fake, so these run without an
install. The payload shapes below were captured from the real
``rapid-mlx models --json`` / ``--cached --json`` on 2026-08-27.
"""

from __future__ import annotations

import json
import time

import pytest

from rmlx_web.catalog import (
    CatalogError,
    ModelCatalog,
    RemovalError,
    _image_capability,
)

# Trimmed from the real payload. `flux2-klein-4b` is retained on purpose:
# it is an image-gen alias that can be present, cached and "ok", and it
# must never appear in a chat picker.
AVAILABLE = {
    "text": [
        {
            "alias": "qwen3.5-9b-4bit",
            "hf_path": "mlx-community/Qwen3.5-9B-4bit",
            "size_bytes": 5977075377,
            "tool_call_parser": "hermes",
            "reasoning_parser": None,
            "modality": "text",
            "is_text_only": False,
        },
        {
            "alias": "bonsai-1.7b-2bit",
            "hf_path": "prism-ml/Ternary-Bonsai-1.7B-mlx-2bit",
            "size_bytes": 495525300,
            "tool_call_parser": "hermes",
            "reasoning_parser": None,
            "modality": "text",
            "is_text_only": True,
        },
    ],
    "image": [
        {
            "alias": "flux2-klein-4b",
            "hf_path": "Runpod/FLUX.2-klein-4B-mflux-4bit",
            "size_bytes": 4619695783,
            "modality": "image-gen",
        }
    ],
    "video": [
        {
            "alias": "wan-2.2-t2v",
            "hf_path": "mlx-community/Wan2.2-T2V",
            "size_bytes": 1,
            "modality": "video-gen",
        }
    ],
    "audio": [
        {
            "alias": "whisper-large-v3",
            "hf_id": "mlx-community/whisper-large-v3",
            "kind": "stt",
            "family": "whisper",
            "modality": "audio",
        }
    ],
}

CACHED = {
    "cached": [
        {
            "alias": "qwen3.5-9b-4bit",
            "repo": "mlx-community/Qwen3.5-9B-4bit",
            "size_bytes": 5977075377,
            "state": "ok",
            "external": False,
        },
        {
            "alias": "flux2-klein-4b",
            "repo": "Runpod/FLUX.2-klein-4B-mflux-4bit",
            "size_bytes": 4619704407,
            "state": "ok",
            "external": False,
        },
        {
            # A partial download: config.json plus some shards. Counting
            # this as cached makes a switch fail inside the engine
            # minutes later rather than here.
            "alias": None,
            "repo": "prism-ml/Ternary-Bonsai-1.7B-mlx-2bit",
            "size_bytes": 1000,
            "state": "incomplete",
            "external": False,
        },
        {
            # Complete and usable, but not in the TEXT alias registry —
            # which is what every downloaded audio repo looks like, since
            # audio aliases live in their own registry the cached scan
            # does not consult.
            "alias": None,
            "repo": "mlx-community/whisper-large-v3",
            "size_bytes": 1610000000,
            "state": "unmapped",
            "external": False,
        },
    ],
    "count": 4,
    "total_bytes": 12207780784,
}


@pytest.fixture
def fake_cli(tmp_path):
    """A stub `rapid-mlx` that answers the two JSON queries.

    Payloads are written as JSON files and read back at runtime rather than
    interpolated into the script source: JSON's ``null``/``true``/``false``
    are not Python literals and would raise ``NameError``.

    Each invocation is logged so tests can assert on caching without
    reaching into the catalog's private state.
    """
    calls = tmp_path / "calls.log"
    available = tmp_path / "available.json"
    cached = tmp_path / "cached.json"
    available.write_text(json.dumps(AVAILABLE))
    cached.write_text(json.dumps(CACHED))

    # /bin/sh rather than python3: the catalog spawns this for every
    # query, and a fresh interpreter per spawn made the suite take ~35s
    # for what is really a few file reads.
    script = tmp_path / "rapid-mlx"
    script.write_text(
        "#!/bin/sh\n"
        f'echo "$@" >> {calls}\n'
        '[ "$1" = "rm" ] && exit 0\n'
        'for arg in "$@"; do\n'
        f'  [ "$arg" = "--cached" ] && exec cat {cached}\n'
        "done\n"
        f"exec cat {available}\n"
    )
    script.chmod(0o755)
    return script, calls


class TestListModels:
    @pytest.mark.asyncio
    async def test_text_image_and_audio_are_listed_and_tagged(self, fake_cli):
        script, _ = fake_cli
        entries = {e.alias: e for e in await ModelCatalog(str(script)).list_models()}

        assert entries["qwen3.5-9b-4bit"].kind == "text"
        assert entries["flux2-klein-4b"].kind == "image"
        assert entries["whisper-large-v3"].kind == "audio"

    @pytest.mark.asyncio
    async def test_video_is_omitted(self, fake_cli):
        script, _ = fake_cli
        entries = await ModelCatalog(str(script)).list_models()

        # The video lane needs extras a plain install does not ship, so a
        # 64 GiB download that cannot be served would be a trap.
        assert "wan-2.2-t2v" not in [e.alias for e in entries]

    @pytest.mark.asyncio
    async def test_audio_is_loadable_too(self, fake_cli):
        script, _ = fake_cli
        entries = {e.alias: e for e in await ModelCatalog(str(script)).list_models()}

        assert entries["qwen3.5-9b-4bit"].loadable is True
        assert entries["flux2-klein-4b"].loadable is True
        # `serve <audio-alias>` has a dedicated fork in the CLI
        # (`_serve_audio_mode`): the child binds, `/health/ready` reports
        # ready, and `/v1/models` describes it as `modality: audio`.
        # Verified live 2026-08-28 with `serve kokoro`. It is the last
        # resort, not the normal path — with any model already serving,
        # the `--enable-audio` lane answers from that process — but the
        # catalog's job is to say what CAN be served, and this can.
        assert entries["whisper-large-v3"].loadable is True

    @pytest.mark.asyncio
    async def test_an_image_row_merges_its_cached_state(self, fake_cli):
        script, _ = fake_cli
        entries = {e.alias: e for e in await ModelCatalog(str(script)).list_models()}

        assert entries["flux2-klein-4b"].cached is True
        assert entries["flux2-klein-4b"].cached_bytes == 4619704407

    @pytest.mark.asyncio
    async def test_image_capability_is_derived_from_the_repo_id(self, fake_cli):
        script, _ = fake_cli
        entries = {e.alias: e for e in await ModelCatalog(str(script)).list_models()}

        # FLUX.2 accepts both request shapes, which is what lets the surface
        # offer Edit on it at all. Repeats the CLI's `[image:both]` rule.
        assert entries["flux2-klein-4b"].image_capability == "both"
        # Only image rows carry one; anything else branching on it would be
        # reading a field that means nothing.
        assert entries["qwen3.5-9b-4bit"].image_capability is None
        assert entries["whisper-large-v3"].image_capability is None

    @pytest.mark.parametrize(
        ("hf_path", "expected"),
        [
            ("Runpod/FLUX.2-klein-4B-mflux-4bit", "both"),
            ("mlx-community/flux2-dev", "both"),
            # Underscores are folded, as the CLI folds them.
            ("org/Qwen_Image_Edit-2509", "editing"),
            ("mlx-community/Qwen-Image-Edit-2509-4bit", "editing"),
            # Plain Qwen-Image is text-to-image: the edit token must win only
            # where it is actually present.
            ("mlx-community/Qwen-Image-4bit", "generation"),
            ("mlx-community/Z-Image-Turbo", "generation"),
        ],
    )
    def test_the_capability_rule_matches_the_cli(self, hf_path, expected):
        assert _image_capability(hf_path) == expected

    @pytest.mark.asyncio
    async def test_an_audio_row_is_normalised(self, fake_cli):
        script, _ = fake_cli
        entries = {e.alias: e for e in await ModelCatalog(str(script)).list_models()}
        whisper = entries["whisper-large-v3"]

        # ``hf_id`` becomes ``hf_path`` and the registry's tts/stt
        # ``kind`` moves aside, so one shape serves every kind.
        assert whisper.hf_path == "mlx-community/whisper-large-v3"
        assert whisper.audio_kind == "stt"
        assert whisper.family == "whisper"
        # No size manifest covers audio repos, so the pull gate — which
        # fails closed on an unknown size — must see None, never zero.
        assert whisper.size_bytes is None


class TestListChatModels:
    @pytest.mark.asyncio
    async def test_only_text_models_are_listed(self, fake_cli):
        script, _ = fake_cli
        entries = await ModelCatalog(str(script)).list_chat_models()

        aliases = [e.alias for e in entries]
        assert "qwen3.5-9b-4bit" in aliases
        assert "bonsai-1.7b-2bit" in aliases
        # Image, video and audio aliases have no chat surface; offering
        # them would dead-end on the first send.
        assert "flux2-klein-4b" not in aliases
        assert "wan-2.2-t2v" not in aliases
        assert "whisper-large-v3" not in aliases

    @pytest.mark.asyncio
    async def test_cached_state_is_merged_by_hf_path(self, fake_cli):
        script, _ = fake_cli
        entries = {
            e.alias: e for e in await ModelCatalog(str(script)).list_chat_models()
        }

        assert entries["qwen3.5-9b-4bit"].cached is True
        assert entries["qwen3.5-9b-4bit"].cached_bytes == 5977075377

    @pytest.mark.asyncio
    async def test_incomplete_download_does_not_count_as_cached(self, fake_cli):
        script, _ = fake_cli
        entries = {
            e.alias: e for e in await ModelCatalog(str(script)).list_chat_models()
        }

        # The bonsai row is present on disk but "incomplete". Reporting
        # it as downloaded would make a switch fail inside the engine.
        # This is the boundary "unmapped" must not be widened past: both
        # carry alias=None, and only one of them is loadable.
        assert entries["bonsai-1.7b-2bit"].cached is False
        assert entries["bonsai-1.7b-2bit"].cached_bytes is None

    @pytest.mark.asyncio
    async def test_an_unmapped_snapshot_still_counts_as_downloaded(self, fake_cli):
        script, _ = fake_cli
        entries = {e.alias: e for e in await ModelCatalog(str(script)).list_models()}

        # "unmapped" means complete and usable but absent from the TEXT
        # alias registry — which is every audio repo, because audio aliases
        # live in their own registry the cached scan does not consult.
        # Accepting only "ok" reported every downloaded voice model as not
        # downloaded, with no way to fix it from this surface.
        assert entries["whisper-large-v3"].cached is True
        assert entries["whisper-large-v3"].cached_bytes == 1610000000

    @pytest.mark.asyncio
    async def test_downloaded_models_sort_first(self, fake_cli):
        script, _ = fake_cli
        entries = await ModelCatalog(str(script)).list_chat_models()

        # Alphabetically bonsai precedes qwen; the cached one must win,
        # because on a 179-alias catalog an alphabetical wall buries the
        # handful the user actually has.
        assert entries[0].alias == "qwen3.5-9b-4bit"
        assert entries[0].cached is True

    @pytest.mark.asyncio
    async def test_metadata_is_carried_through(self, fake_cli):
        script, _ = fake_cli
        entries = {
            e.alias: e for e in await ModelCatalog(str(script)).list_chat_models()
        }

        assert entries["qwen3.5-9b-4bit"].tool_call_parser == "hermes"
        assert entries["bonsai-1.7b-2bit"].is_text_only is True
        assert entries["qwen3.5-9b-4bit"].size_bytes == 5977075377


class TestCaching:
    @pytest.mark.asyncio
    async def test_alias_list_is_fetched_once(self, fake_cli):
        script, calls = fake_cli
        catalog = ModelCatalog(str(script))

        await catalog.list_chat_models()
        await catalog.list_chat_models()

        lines = calls.read_text().splitlines()
        # The alias list only changes when the rapid-mlx package is
        # upgraded, which cannot happen under a running supervisor.
        assert lines.count("models --json") == 1

    @pytest.mark.asyncio
    async def test_disk_scan_is_reused_within_the_ttl(self, fake_cli):
        script, calls = fake_cli
        catalog = ModelCatalog(str(script))

        await catalog.list_chat_models()
        await catalog.list_chat_models()

        lines = calls.read_text().splitlines()
        assert lines.count("models --cached --json") == 1

    @pytest.mark.asyncio
    async def test_force_rescans_the_disk(self, fake_cli):
        script, calls = fake_cli
        catalog = ModelCatalog(str(script))

        await catalog.list_chat_models()
        await catalog.list_chat_models(force=True)

        lines = calls.read_text().splitlines()
        assert lines.count("models --cached --json") == 2
        # The alias list is still not re-read: only the disk changed.
        assert lines.count("models --json") == 1

    @pytest.mark.asyncio
    async def test_invalidate_forces_the_next_scan(self, fake_cli):
        script, calls = fake_cli
        catalog = ModelCatalog(str(script))

        await catalog.list_chat_models()
        catalog.invalidate_cache()
        await catalog.list_chat_models()

        assert calls.read_text().count("models --cached --json") == 2


class TestAliasValidation:
    @pytest.mark.asyncio
    async def test_known_chat_alias_is_accepted(self, fake_cli):
        script, _ = fake_cli
        assert await ModelCatalog(str(script)).is_known_chat_alias("qwen3.5-9b-4bit")

    @pytest.mark.asyncio
    async def test_non_chat_alias_is_rejected(self, fake_cli):
        script, _ = fake_cli
        # Present in the catalog, but as an image model.
        assert not await ModelCatalog(str(script)).is_known_chat_alias("flux2-klein-4b")

    @pytest.mark.asyncio
    async def test_arbitrary_repo_is_rejected(self, fake_cli):
        script, _ = fake_cli
        catalog = ModelCatalog(str(script))
        # Accepting this would turn the model picker into a
        # general-purpose remote fetch primitive.
        assert not await catalog.is_known_chat_alias("attacker/anything")
        assert not await catalog.is_known_chat_alias("../../etc/passwd")


class TestRemove:
    @pytest.mark.asyncio
    async def test_the_repo_is_passed_not_the_alias(self, fake_cli):
        script, calls = fake_cli
        freed = await ModelCatalog(str(script)).remove("qwen3.5-9b-4bit")

        # `rm` scans the cache for ``models--<owner>--<repo>``, which a
        # bare alias can never match. Passing the catalog's own hf_path
        # also means the argv token is one this install published rather
        # than one the caller chose.
        assert "rm mlx-community/Qwen3.5-9B-4bit --yes" in calls.read_text()
        assert freed == 5977075377

    @pytest.mark.asyncio
    async def test_an_unknown_alias_is_refused(self, fake_cli):
        script, calls = fake_cli
        with pytest.raises(RemovalError):
            await ModelCatalog(str(script)).remove("attacker/anything")

        assert "rm" not in calls.read_text()

    @pytest.mark.asyncio
    async def test_a_cached_image_model_can_be_removed(self, fake_cli):
        script, calls = fake_cli
        freed = await ModelCatalog(str(script)).remove("flux2-klein-4b")

        # Removal is kind-agnostic on purpose: it frees disk, and an
        # image checkpoint is one of the largest things on it.
        assert "rm Runpod/FLUX.2-klein-4B-mflux-4bit --yes" in calls.read_text()
        assert freed == 4619704407

    @pytest.mark.asyncio
    async def test_a_model_that_is_not_downloaded_is_refused(self, fake_cli):
        script, calls = fake_cli
        # The bonsai row is on disk but "incomplete", so it never counted
        # as cached; there is nothing here to report as freed.
        with pytest.raises(RemovalError) as excinfo:
            await ModelCatalog(str(script)).remove("bonsai-1.7b-2bit")

        assert "not downloaded" in str(excinfo.value)
        assert "rm" not in calls.read_text()

    @pytest.mark.asyncio
    async def test_the_disk_scan_is_invalidated(self, fake_cli):
        script, calls = fake_cli
        catalog = ModelCatalog(str(script))

        await catalog.remove("qwen3.5-9b-4bit")
        await catalog.list_chat_models()

        # Without this the page keeps showing "on disk" for up to the
        # TTL after the row was deleted.
        assert calls.read_text().count("models --cached --json") == 2

    @pytest.mark.asyncio
    async def test_a_failing_rm_raises_with_its_output(self, tmp_path):
        available = tmp_path / "available.json"
        cached = tmp_path / "cached.json"
        available.write_text(json.dumps(AVAILABLE))
        cached.write_text(json.dumps(CACHED))

        script = tmp_path / "rapid-mlx"
        script.write_text(
            "#!/bin/sh\n"
            '[ "$1" = "rm" ] && { echo "permission denied" >&2; exit 1; }\n'
            'for arg in "$@"; do\n'
            f'  [ "$arg" = "--cached" ] && exec cat {cached}\n'
            "done\n"
            f"exec cat {available}\n"
        )
        script.chmod(0o755)

        with pytest.raises(RemovalError) as excinfo:
            await ModelCatalog(str(script)).remove("qwen3.5-9b-4bit")

        assert "permission denied" in str(excinfo.value)


class TestFailureModes:
    @pytest.mark.asyncio
    async def test_missing_binary_raises_catalog_error(self):
        catalog = ModelCatalog("/nonexistent/rapid-mlx")
        with pytest.raises(CatalogError):
            await catalog.list_chat_models()

    @pytest.mark.asyncio
    async def test_non_zero_exit_raises_with_stderr(self, tmp_path):
        script = tmp_path / "failing"
        script.write_text("#!/bin/sh\necho 'boom' >&2\nexit 2\n")
        script.chmod(0o755)

        with pytest.raises(CatalogError) as excinfo:
            await ModelCatalog(str(script)).list_chat_models()

        assert "exit 2" in str(excinfo.value)
        assert "boom" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_non_json_output_raises(self, tmp_path):
        script = tmp_path / "chatty"
        script.write_text("#!/bin/sh\necho 'not json at all'\n")
        script.chmod(0o755)

        with pytest.raises(CatalogError) as excinfo:
            await ModelCatalog(str(script)).list_chat_models()

        assert "did not emit JSON" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_timeout_kills_the_subprocess(self, tmp_path, monkeypatch):
        from rmlx_web import catalog as catalog_module

        # Short ceiling: this measures the timeout path, not startup
        # speed, and the child never returns by design.
        monkeypatch.setattr(catalog_module, "_SUBPROCESS_TIMEOUT_S", 0.5)

        script = tmp_path / "hanging"
        # `sleep` as a CHILD of the shell, which is the case that used to
        # break: killing only the shell leaves the sleep holding the
        # stdout pipe open, and asyncio does not report the process as
        # finished until those pipes close. The timeout then blocked for
        # the child's full runtime — precisely what it exists to prevent.
        script.write_text("#!/bin/sh\nsleep 30\n")
        script.chmod(0o755)

        started = time.monotonic()
        with pytest.raises(CatalogError) as excinfo:
            await ModelCatalog(str(script)).list_chat_models()
        elapsed = time.monotonic() - started

        assert "timed out" in str(excinfo.value)
        # Deliberately loose (0.5s ceiling, 10s bound): this asserts the
        # timeout fires at all, not how fast. The regression it guards
        # took the full 30s.
        assert elapsed < 10.0
