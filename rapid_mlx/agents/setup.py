# SPDX-License-Identifier: Apache-2.0
"""Safe plan/apply flow for first-class agent integrations."""

from __future__ import annotations

import difflib
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rapid_mlx.agents.config_merge import (
    deep_merge,
    is_id_list,
    merge_by_id,
    merge_patch_layers,
)
from rapid_mlx.agents.telemetry import (
    track_agent_configure_failed,
)
from rapid_mlx.launch import _common as launch_common
from rapid_mlx.launch import claude_code, continue_dev

# Agents whose ``--setup`` goes through the plan/apply flow below: an exact
# diff preview, consent (or --yes), a timestamped backup of the existing file
# and an atomic write. Every CLI entry point routes on this one set.
FIRST_CLASS_SETUP_AGENTS = frozenset(
    {"claude-code", "continue", "deepseek-harness", "pi", "qwen-code"}
)


@dataclass(frozen=True)
class SetupPlan:
    agent: str
    display_name: str
    path: Path
    # A mapping for every JSON/mapping plan; a top-level LIST of Cordis patch
    # layers for the dsh plan (dsh >= 0.2 patch files are lists, #4040).
    before: dict[str, Any] | list[Any]
    after: dict[str, Any] | list[Any]
    base_url: str
    model: str
    format: str = "json"
    credentials_path: Path | None = None
    credentials_before: dict[str, Any] | None = None
    credentials_after: dict[str, Any] | None = None

    @property
    def changed(self) -> bool:
        return self.before != self.after or (
            self.credentials_path is not None
            and self.credentials_before != self.credentials_after
        )

    def diff(self) -> str:
        before_data = self.before
        after_data = self.after
        secret_changed = False
        if (
            self.agent == "claude-code"
            and isinstance(self.before, dict)
            and isinstance(self.after, dict)
        ):
            # A keyed local server needs the real bearer in Claude's settings,
            # but the setup preview must never print it or an existing token.
            def hidden(data: dict[str, Any]) -> dict[str, Any]:
                result = dict(data)
                env = dict(result.get("env", {}))
                visible_fields = {
                    "ANTHROPIC_BASE_URL",
                    "ANTHROPIC_MODEL",
                    "CLAUDE_CODE_MAX_CONTEXT_TOKENS",
                }
                for name in env:
                    if name not in visible_fields and env[name]:
                        env[name] = "<redacted>"
                result["env"] = env
                return result

            before_data = hidden(self.before)
            after_data = hidden(self.after)
            secret_changed = any(
                self.before.get("env", {}).get(name)
                != self.after.get("env", {}).get(name)
                for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
            )
        before = _serialize(before_data, self.format)
        after = _serialize(after_data, self.format)
        primary = "\n".join(
            difflib.unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile=str(self.path),
                tofile=str(self.path) + " (proposed)",
                lineterm="",
            )
        )
        if secret_changed:
            primary = "\n".join(
                part
                for part in (
                    primary,
                    "  Claude credentials will be updated (values hidden).",
                )
                if part
            )
        if self.credentials_path is None or self.credentials_after is None:
            return primary
        # DSH's credential store may contain real keys for unrelated remote
        # providers.  A setup preview must never echo those secrets.  Show
        # only the presence/absence of the one Rapid-managed key.
        managed_key = "RAPID_MLX_API_KEY"
        credentials_before = (
            {managed_key: "<configured; preserved>"}
            if managed_key in (self.credentials_before or {})
            else {}
        )
        credentials_after = (
            {managed_key: "<configured; preserved>"}
            if managed_key in (self.credentials_before or {})
            else {managed_key: "not-needed"}
        )
        credentials = "\n".join(
            difflib.unified_diff(
                _serialize(credentials_before, "yaml").splitlines(),
                _serialize(credentials_after, "yaml").splitlines(),
                fromfile=str(self.credentials_path),
                tofile=str(self.credentials_path) + " (proposed)",
                lineterm="",
            )
        )
        return "\n".join(part for part in (primary, credentials) if part)


def _serialize(data: dict[str, Any] | list[Any], format: str) -> str:
    if format == "yaml":
        import yaml

        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True).rstrip()
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)


def _load_yaml_mapping(
    path: Path, agent: str | None = None, *, emit_telemetry: bool = True
) -> dict[str, Any]:
    import yaml

    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return {}
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        if emit_telemetry:
            track_agent_configure_failed("config_invalid", agent)
        raise
    if not isinstance(value, dict):
        if emit_telemetry:
            track_agent_configure_failed("config_invalid", agent)
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _load_patch_layers(
    path: Path, agent: str | None = None, *, emit_telemetry: bool = True
) -> list[Any]:
    """Load a Cordis patch-layer file — a top-level YAML list of ``{id, config}``.

    dsh's own parser rejects anything else ("must be a top-level YAML array of
    loader patch entries"), so a stray mapping is reported as invalid config
    rather than silently swallowed into an empty merge.
    """
    import yaml

    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        if emit_telemetry:
            track_agent_configure_failed("config_invalid", agent)
        raise
    if not isinstance(value, list):
        if emit_telemetry:
            track_agent_configure_failed("config_invalid", agent)
        raise ValueError(f"{path} must contain a YAML list of patch layers")
    return value


def _dsh_patch_path() -> Path:
    """dsh's home-level Cordis patch layer: ``$DSH_HOME/cordis.patch.yml``.

    dsh 0.2 composes each profile boot from patch layers: bundle patches, the
    per-profile ``cordis.patch.yml``, the home-level
    ``$DSH_HOME/cordis.patch.yml`` (machine-local preferences that apply to
    every profile, verified in @deepseek-ai/dsh-app-boot's
    ``readProfilePatches``), then ``--patch`` overlays. Writing there means a
    plain ``dsh --profile headless '<task>'`` picks the provider up with no
    extra flags. dsh 0.1.x instead read ``$DSH_HOME/settings.yaml``, which
    0.2.x ignores — that move is exactly what issue #4040 tracks.
    """
    configured = os.environ.get("DSH_HOME", "").strip()
    root = Path(configured).expanduser() if configured else Path.home() / ".dsh"
    return root / "cordis.patch.yml"


def _pi_profile() -> Any:
    from rapid_mlx.agents import get_profile

    profile = get_profile("pi")
    assert profile is not None, "the pi profile ships with rapid-mlx"
    return profile


def _qwen_code_profile() -> Any:
    from rapid_mlx.agents import get_profile

    profile = get_profile("qwen-code")
    assert profile is not None, "the qwen-code profile ships with rapid-mlx"
    return profile


def _qwen_code_settings_path() -> Path:
    """Qwen Code's settings file, resolved like the generic setup writer.

    A profile in ``~/.rapid-mlx/agents`` may shadow the shipped one. This flow
    merges a JSON settings file, so a shadowing profile of any other shape is
    refused here rather than crashing further down.
    """
    from rapid_mlx.agents.adapter import _resolve_config_path

    cfg = _qwen_code_profile().get_config_for_version(None)
    if (
        cfg.type != "json"
        or not (isinstance(cfg.path, str) and cfg.path)
        or not (isinstance(cfg.template, str) and cfg.template)
    ):
        raise ValueError(
            "the installed qwen-code profile does not describe a JSON settings "
            "file; fix or remove its override in ~/.rapid-mlx/agents"
        )
    try:
        return _resolve_config_path(cfg).resolve()
    except RuntimeError as exc:  # a symlink loop, before Python 3.13
        raise ValueError(f"cannot resolve the Qwen Code settings path: {exc}") from exc


def _pi_models_path() -> Path:
    """pi's ``<agent-dir>/models.json``, honouring ``PI_CODING_AGENT_DIR``.

    Resolved through the same helper the generic writer uses, so the
    relocation contract lives in one place (the profile's ``home_env``).
    The path is fully resolved: a ``models.json`` symlinked into a dotfiles
    repo is read, backed up and atomically replaced at its real target, so
    the rename never swaps the link itself for a disconnected file (the
    generic writer resolves symlinks the same way).
    """
    from rapid_mlx.agents.adapter import _resolve_config_path

    return _resolve_config_path(_pi_profile().get_config_for_version(None)).resolve()


def _atomic_write_secure_text(path: Path, text: str) -> None:
    """Atomically publish owner-only text beside a potentially secret config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".new", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_setup_plan(
    agent: str,
    base_url: str,
    model: str,
    context_length: int | None = None,
    supports_reasoning: bool | None = None,
    *,
    emit_telemetry: bool = True,
) -> SetupPlan:
    """Build a side-effect-free setup plan for a supported client."""
    if agent in {"claude", "claude-code"}:
        path = claude_code.current_config_path()
        assert path is not None
        # The union annotation is the contract: only the dsh plan carries a
        # patch-layer list; every other flow is a mapping.
        loaded: dict[str, Any] = launch_common.load_json_lenient(path)
        before: dict[str, Any] | list[Any] = loaded
        after: dict[str, Any] | list[Any] = claude_code.patched_config(
            loaded,
            base_url,
            model,
            api_key=os.environ.get("RAPID_MLX_API_KEY") or "sk-noop",
            context_length=context_length,
        )
        return SetupPlan(
            "claude-code", "Claude Code", path, before, after, base_url, model
        )
    if agent in {"continue", "continue-dev"}:
        path = continue_dev.current_config_path()
        assert path is not None
        before = launch_common.load_json_lenient(path)
        after = continue_dev.patched_config(before, base_url, model)
        return SetupPlan(
            "continue", "Continue.dev", path, before, after, base_url, model
        )
    if agent in {"deepseek-harness", "dsh"}:
        # Resolve each managed file independently before backup + atomic
        # replace so dotfile-managed symlinks survive.  Keep the logical DSH
        # home for locating credentials: cordis.patch.yml may point into an
        # unrelated repository whose parent is not DSH_HOME.
        logical_path = _dsh_patch_path()
        path = logical_path.resolve()
        before = _load_patch_layers(
            path, "deepseek-harness", emit_telemetry=emit_telemetry
        )
        credentials_path = (logical_path.parent / ".credentials.yaml").resolve()
        credentials_before = _load_yaml_mapping(
            credentials_path,
            "deepseek-harness",
            emit_telemetry=emit_telemetry,
        )
        context = context_length if context_length and context_length > 0 else 32768
        # Report the model's ACTUAL reasoning capability rather than
        # asserting the graded ladder for everything we serve.
        #
        # Harness renders a reasoning-effort control from this block, so a
        # model with no reasoning parser used to get an off/low/medium/high
        # selector that changed nothing the user could observe. pi-ai
        # accepts ``reasoningEfforts: false`` for exactly this case
        # ("set false for a non-reasoning model").
        #
        # Only a definite ``False`` downgrades. ``None`` means we could not
        # find out (server unreachable, model not listed, or a rapid-mlx too
        # old to report the field), and silently deleting a working control
        # on a guess is worse than the cosmetic over-claim this fixes — see
        # ``fetch_reasoning_support`` for why the three states are kept apart.
        reasoning_capable = supports_reasoning is not False
        model_entry: dict[str, Any] = {
            "id": model,
            "name": f"{model} (Rapid-MLX)",
            "contextWindow": context,
            "maxTokens": 8192,
            "reasoningEfforts": (
                {
                    "off": "none",
                    "low": "low",
                    "medium": "medium",
                    "high": "high",
                }
                if reasoning_capable
                else False
            ),
        }
        # Cordis patch layers, in dsh's composition order. The verified dsh
        # 0.2 contract (issues #4040, harness-lab 2026-10-02): the same
        # provider block the old settings.yaml carried, expressed as a
        # top-level patch-layer list — a mapping there fails dsh's parser
        # ("must be a top-level YAML array of loader patch entries").
        patch: list[dict[str, Any]] = [
            {
                "id": "llm-pi-ai",
                "config": {
                    "providers": {
                        "rapid-mlx": {
                            "displayName": "Rapid-MLX Local",
                            "apiKeyEnv": "RAPID_MLX_API_KEY",
                            "api": "openai-completions",
                            "baseURL": base_url.rstrip("/"),
                            "defaultContextWindow": context,
                            "defaultMaxTokens": 8192,
                            "compat": {"supportsReasoningEffort": reasoning_capable},
                            "models": [model_entry],
                        }
                    }
                },
            },
            {
                "id": "agent-default-model",
                "config": {"provider": "rapid-mlx", "model": model},
            },
        ]
        credentials_after = dict(credentials_before)
        credentials_after.setdefault("RAPID_MLX_API_KEY", "not-needed")
        return SetupPlan(
            "deepseek-harness",
            "DeepSeek Harness",
            path,
            before,
            merge_patch_layers(before, patch),
            base_url,
            model,
            "yaml",
            credentials_path,
            credentials_before,
            credentials_after,
        )
    if agent == "pi":
        path = _pi_models_path()
        try:
            loaded_pi = launch_common.load_json_lenient(path)
        except json.JSONDecodeError:
            if emit_telemetry:
                track_agent_configure_failed("config_invalid", "pi")
            raise
        if not isinstance(loaded_pi, dict):
            if emit_telemetry:
                track_agent_configure_failed("config_invalid", "pi")
            raise ValueError(f"{path} must contain a JSON object")
        # Render the shipped profile template so the plan and the profile can
        # never drift; merge so the user's other providers AND their other
        # models under providers.rapid-mlx survive (models merge by ``id``).
        template = json.loads(
            _pi_profile().render_config(base_url, model, context_length=context_length)
        )
        return SetupPlan(
            "pi",
            "Pi Coding Agent",
            path,
            loaded_pi,
            deep_merge(loaded_pi, template),
            base_url,
            model,
        )
    if agent == "qwen-code":
        path = _qwen_code_settings_path()
        try:
            loaded_qwen = launch_common.load_json_lenient(path)
        except OSError:
            if emit_telemetry:
                track_agent_configure_failed("other", "qwen-code")
            raise
        except (ValueError, RecursionError) as exc:
            # Undecodable bytes and pathological nesting are refused like any
            # other file we cannot round-trip, never surfaced as a traceback.
            if emit_telemetry:
                track_agent_configure_failed("config_invalid", "qwen-code")
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc
        if not isinstance(loaded_qwen, dict):
            if emit_telemetry:
                track_agent_configure_failed("config_invalid", "qwen-code")
            raise ValueError(f"{path} must contain a JSON object")
        profile = _qwen_code_profile()
        template = json.loads(
            profile.render_config(base_url, model, context_length=context_length)
        )
        incoming_providers = (
            template.get("modelProviders") if isinstance(template, dict) else None
        )
        incoming_openai = (
            incoming_providers.get("openai")
            if isinstance(incoming_providers, dict)
            else None
        )
        if not is_id_list(incoming_openai):
            raise ValueError(
                "the installed qwen-code profile template must define "
                "modelProviders.openai entries with an id"
            )
        after = deep_merge(loaded_qwen, template)

        # ``modelProviders.openai`` is an id-keyed registry shared with the
        # user's other OpenAI-compatible endpoints. Generic setup replaces
        # lists, so preserve existing entries and update only our model id.
        existing_providers = loaded_qwen.get("modelProviders")
        existing_openai = (
            existing_providers.get("openai")
            if isinstance(existing_providers, dict)
            else None
        )
        if isinstance(existing_openai, list):
            after["modelProviders"]["openai"] = merge_by_id(
                existing_openai, incoming_openai
            )
        return SetupPlan(
            "qwen-code",
            "Qwen Code",
            path,
            loaded_qwen,
            after,
            base_url,
            model,
        )
    # Reserved for defensive callers. The CLIs only route
    # FIRST_CLASS_SETUP_AGENTS here, so this outcome is currently unreachable.
    if emit_telemetry:
        track_agent_configure_failed("no_safe_setup_flow", agent)
    raise ValueError(f"{agent} does not have a first-class safe setup flow")


def apply_setup_plan(plan: SetupPlan) -> Path:
    """Back up the existing config and atomically apply an unchanged plan."""
    # Re-read to prevent overwriting an edit made between preview and consent.
    # The shape check on ``after`` picks the loader: a patch-layer plan (dsh)
    # re-reads a top-level list, everything else a mapping / JSON object.
    if plan.format == "yaml" and isinstance(plan.after, list):
        current: Any = _load_patch_layers(plan.path, plan.agent)
    elif plan.format == "yaml":
        current = _load_yaml_mapping(plan.path, plan.agent)
    else:
        try:
            current = launch_common.load_json_lenient(plan.path)
        except (ValueError, RecursionError) as exc:
            # The plan was built from a readable file, so one that no longer
            # parses was edited after the preview.
            track_agent_configure_failed("config_changed", plan.agent)
            raise RuntimeError(
                f"{plan.path} changed after preview; re-run --setup"
            ) from exc
    if current != plan.before:
        track_agent_configure_failed("config_changed", plan.agent)
        raise RuntimeError(f"{plan.path} changed after preview; re-run --setup")
    if plan.credentials_path is not None:
        credentials_current = _load_yaml_mapping(plan.credentials_path, plan.agent)
        if credentials_current != (plan.credentials_before or {}):
            track_agent_configure_failed("config_changed", plan.agent)
            raise RuntimeError(
                f"{plan.credentials_path} changed after preview; re-run --setup"
            )
    try:
        launch_common.backup_existing(plan.path)
        if plan.credentials_path is not None and plan.credentials_after is not None:
            launch_common.backup_existing(plan.credentials_path)
            # Publish the harmless loopback sentinel first.  If the later settings
            # write fails, DSH retains its prior provider selection; the reverse
            # order would leave a newly-selected Rapid route unable to authenticate.
            _atomic_write_secure_text(
                plan.credentials_path,
                _serialize(plan.credentials_after, "yaml") + "\n",
            )
        if plan.format == "yaml":
            _atomic_write_secure_text(plan.path, _serialize(plan.after, "yaml") + "\n")
        else:
            launch_common.atomic_write_json(plan.path, plan.after)
    except BaseException as exc:
        if isinstance(exc, OSError):
            track_agent_configure_failed("config_write_failed", plan.agent)
        raise
    return plan.path


def verify_server(
    base_url: str,
    expected_model: str,
    timeout: float = 2.0,
    *,
    agent: str,
) -> str:
    """Verify health and model discovery without performing inference."""
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{root}/health", timeout=timeout) as response:
            if response.status != 200:
                track_agent_configure_failed("server_not_ready", agent)
                raise RuntimeError(f"health returned HTTP {response.status}")
        from rapid_mlx.http_auth import rapid_mlx_auth_headers

        request = urllib.request.Request(
            f"{root}/v1/models", headers=rapid_mlx_auth_headers()
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        track_agent_configure_failed("server_not_ready", agent)
        raise RuntimeError(f"server is not ready at {root}: {exc}") from exc
    models = payload.get("data", []) if isinstance(payload, dict) else []
    ids = [item.get("id") for item in models if isinstance(item, dict)]
    if not ids:
        track_agent_configure_failed("server_no_models", agent)
        raise RuntimeError(f"server at {root} reported no models")
    if expected_model != "default" and expected_model not in ids and len(ids) != 1:
        track_agent_configure_failed("model_not_advertised", agent)
        raise RuntimeError(
            f"server does not advertise model {expected_model!r} (found: {', '.join(ids)})"
        )
    return (
        expected_model
        if expected_model != "default" and expected_model in ids
        else ids[0]
    )


def confirm_plan(plan: SetupPlan) -> bool:
    """Ask before writing; non-interactive callers must use --yes."""
    if not sys.stdin.isatty():
        return False
    try:
        return input("Apply this configuration? [y/N] ").strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        return False
