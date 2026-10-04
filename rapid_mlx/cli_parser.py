# SPDX-License-Identifier: Apache-2.0
"""Argparse construction for the ``rapid-mlx`` CLI.

Split out of :mod:`rapid_mlx.cli` so the parser surface (pinned by
``tests/test_cli_parser_snapshot.py``) lives apart from command execution.
:mod:`rapid_mlx.cli` re-exports every public-ish name defined here, so
``from rapid_mlx.cli import build_parser`` keeps working. Patch these names
on this module (not on :mod:`rapid_mlx.cli`) to affect parser construction.
"""

import argparse

from rapid_mlx._completion import alias_completer
from rapid_mlx.runtime.optional_runtime import optional_extra_install_hint


def _stamp_port_explicit(args: argparse.Namespace) -> argparse.Namespace:
    """Stamp bind-port provenance from the parsed server namespace."""
    if not hasattr(args, "port"):
        return args
    if getattr(args, "listen_fd", None) is not None:
        args._port_explicit = None
    else:
        args._port_explicit = args.port is not None
    return args


class _PortContextArgumentParser(argparse.ArgumentParser):
    """Argument parser that records the effective bind-port provenance."""

    def parse_args(self, args=None, namespace=None):
        if args is None and namespace is None:
            parsed = super().parse_args()
        elif namespace is None:
            parsed = super().parse_args(args)
        else:
            parsed = super().parse_args(args, namespace)
        return _stamp_port_explicit(parsed)


def _log_level_choice(value: str) -> str:
    """Argparse ``type`` callable: normalize to upper-case so
    ``--log-level info`` is accepted as ``INFO``. Named (not a lambda)
    so argparse's error messages read sensibly instead of
    ``invalid <lambda> value``.
    """
    return value.upper()


def _add_video_job_args(parser: argparse.ArgumentParser) -> None:
    """Register the shared video artifact-store option on a serve parser."""
    parser.add_argument(
        "--video-output-dir",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Persist completed video jobs and MP4 files under PATH so they "
            "remain available after a server restart. The default uses a "
            "process-temporary directory."
        ),
    )


def _port_arg(value: str) -> int:
    """Argparse ``type`` callable: validate ``--port`` is in [1, 65535].

    Without this, ``rapid-mlx chat --port 99999`` parsed successfully and
    dropped the user into a REPL whose first turn failed with a confusing
    ``Failed to parse: http://127.0.0.1:99999/...``. Validate early so the
    user sees a one-line argparse error instead.
    """
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"port must be an integer, got {value!r}"
        ) from None
    if not (1 <= port <= 65535):
        raise argparse.ArgumentTypeError(
            f"port must be between 1 and 65535, got {port}"
        )
    return port


def _listen_fd_arg(value: str) -> int:
    """Argparse ``type`` callable: validate ``--listen-fd`` is a sane fd.

    ``--listen-fd`` enables socket activation — the supervisor (launchd,
    systemd, an external parent process) binds the listening socket
    itself and execve's into ``rapid-mlx serve`` with the pre-bound fd.
    This closes the bind→auth TOCTOU window: by the time rapid-mlx
    runs, the socket is already bound but no requests can be accepted
    until ``uvicorn.run`` calls ``accept()`` — at which point the
    FastAPI app (with all route auth dependencies wired) is already
    constructed. See ``rapid_mlx/server.py`` and the regression test
    pinning the bind→auth invariant.

    Accept integers in ``[3, 1023]``:

    * 0/1/2 are stdin/stdout/stderr — never a listening socket.
    * 3 is the conventional "first non-stdio fd" (systemd's
      ``LISTEN_FDS_START`` and launchd both follow this convention).
    * 1023 is the SysV soft-limit ceiling — anything higher is almost
      certainly a typo, not a real fd.
    """
    try:
        fd = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--listen-fd must be an integer, got {value!r}"
        ) from None
    if not (3 <= fd <= 1023):
        raise argparse.ArgumentTypeError(
            f"--listen-fd must be between 3 and 1023, got {fd}"
        )
    return fd


def non_negative_int(value: str) -> int:
    """Argparse ``type`` callable: parse a ``>= 0`` integer.

    Rejects a negative value at parse time so a bad ``--response-cache-
    entries -5`` fails immediately with a clear argparse error, before any
    model download or load. ``SchedulerConfig.__post_init__`` also rejects
    negatives, but for ``serve`` that check runs only after the expensive
    download/load, so the early argparse guard gives the user faster,
    clearer feedback. The construction-time check stays as defense in
    depth.
    """
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative integer, got {value!r}"
        ) from None
    if n < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {n}")
    return n


def positive_int(value: str) -> int:
    """Argparse ``type`` callable: parse a strictly positive integer."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {value!r}"
        ) from None
    if n <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {n}")
    return n


def positive_finite_float(value: str) -> float:
    """Argparse type for positive, finite resource-budget values."""
    import math

    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a positive finite number, got {value!r}"
        ) from None
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError(
            f"expected a positive finite number, got {value!r}"
        )
    return number


def _add_pflash_args(parser) -> None:
    """Attach PFlash long-prompt-compression CLI flags to an argparse parser.

    Used by both ``serve`` and ``bench`` so the flag surface stays in
    sync. The default for ``--pflash`` is intentionally ``None``
    (sentinel for "user passed nothing") so the per-alias resolver in
    ``pflash.resolve_pflash_mode_default`` can switch the engine to
    ``always`` for ``pflash_tier="verified"`` aliases (Qwen3.5 /
    Qwen3.6 family per #287) without breaking the explicit-override
    contract: passing ``--pflash off`` still wins.
    """
    parser.add_argument(
        "--pflash",
        choices=["off", "auto", "always"],
        default=None,
        help="Enable PFlash long-prompt prefill compression "
        "(off, auto, always). Default: 'always' for verified aliases "
        "(Qwen3.5 / Qwen3.6 family per #287), 'off' for everything else.",
    )
    parser.add_argument(
        "--pflash-threshold",
        type=int,
        default=32_768,
        help="Minimum prompt tokens before --pflash auto compresses (default: 32768).",
    )
    parser.add_argument(
        "--pflash-keep-ratio",
        type=float,
        default=None,
        help="Fraction of prompt tokens to keep when compressing. Unset lets "
        "the engine resolve it: a per-alias ``pflash_keep_ratio`` override if "
        "the alias pins one (e.g. 0.50 for a ternary arch), else the default "
        "0.20 (the bench-validated profile in PR #649: TTFT 3.87x-8.5x, needle "
        "recall 5/5). An explicit value here always wins.",
    )
    parser.add_argument(
        "--pflash-min-keep-tokens",
        type=int,
        default=2_048,
        help="Minimum tokens to keep when compressing (default: 2048).",
    )
    parser.add_argument(
        "--pflash-sink-tokens",
        type=int,
        default=256,
        help="Leading prompt tokens always kept by PFlash (default: 256).",
    )
    parser.add_argument(
        "--pflash-tail-tokens",
        type=int,
        default=2_048,
        help="Trailing prompt tokens always kept by PFlash (default: 2048).",
    )
    parser.add_argument(
        "--pflash-block-size",
        type=int,
        default=128,
        help="Middle-token scoring block size (default: 128).",
    )
    parser.add_argument(
        "--pflash-query-window",
        type=int,
        default=512,
        help="Trailing query window used to score middle blocks (default: 512).",
    )
    parser.add_argument(
        "--pflash-stride-blocks",
        type=int,
        default=8,
        help="Keep every Nth middle block as an anchor during scoring "
        "(0 disables anchors, default: 8).",
    )
    parser.add_argument(
        "--pflash-include-tools",
        action="store_true",
        help="Allow PFlash compression on prompts with tool definitions. "
        "By default tool prompts are skipped for tool-call reliability.",
    )


def _resolve_cli_version() -> str:
    from importlib.metadata import version as pkg_version

    try:
        return pkg_version("rapid-mlx")
    except Exception:
        return "dev"


def _add_system_one_parser(
    subparsers: "argparse._SubParsersAction[_PortContextArgumentParser]",
) -> None:
    """Register the ``system-one`` subcommand."""
    system_one_parser = subparsers.add_parser(
        "system-one",
        help="Serve a typed decision model",
        description=(
            "Start a TypeSafe-compatible decision server with POST "
            "/v1/systemone and POST /v1/rank. This service is independent "
            "from the OpenAI-compatible generative server."
        ),
        allow_abbrev=False,
    )
    system_one_parser.add_argument(
        "model",
        nargs="?",
        default="convaiinnovations/laya",
        help="Laya model id/path, CLM public name, or clef/clef-flash",
    )
    system_one_parser.add_argument(
        "--backend", choices=("auto", "laya", "clm", "clef"), default="auto"
    )
    system_one_parser.add_argument("--host", default="127.0.0.1")
    system_one_parser.add_argument("--port", type=_port_arg, default=None)
    system_one_parser.add_argument("--api-key", default=None)
    system_one_parser.add_argument(
        "--log-level",
        type=_log_level_choice,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    system_one_parser.add_argument(
        "--device",
        choices=("gpu", "cpu"),
        default="gpu",
        help="MLX device for Laya/CLM; Metal/MPS device for Clef",
    )
    system_one_parser.add_argument(
        "--dtype",
        choices=("float16", "float32", "bfloat16"),
        default="float16",
        help="Laya weight dtype",
    )
    system_one_parser.add_argument("--batch-size", type=positive_int, default=16)
    system_one_parser.add_argument(
        "--encoder",
        default="Qwen/Qwen3-8B",
        help="CLM backbone; use the BF16 Qwen3-8B reference for calibrated output",
    )
    system_one_parser.add_argument(
        "--head",
        help="Converted CLM head directory (config.json + model.safetensors)",
    )
    system_one_parser.add_argument(
        "--cache-entries", type=non_negative_int, default=20_000
    )
    system_one_parser.add_argument("--max-tokens", type=positive_int, default=2048)
    system_one_parser.add_argument(
        "--max-work-tokens",
        type=positive_int,
        default=32_768,
        help="Maximum aggregate CLM encoder tokens accepted in one request",
    )
    system_one_parser.add_argument(
        "--max-concurrent-requests",
        type=positive_int,
        default=8,
        help="Maximum outstanding System One backend requests",
    )


def _add_cua_parser(
    subparsers: "argparse._SubParsersAction[_PortContextArgumentParser]",
) -> None:
    """Register the ``cua`` subcommand."""
    # Serve command. ``allow_abbrev=False`` blocks unique-prefix matches
    # like ``--no-thin`` resolving silently to ``--no-thinking``: with the
    # hidden ``--no-think`` cross-alias added in D4, both flags share the
    # ``--no-thi`` prefix and prefix matching becomes ambiguous (an
    # ambiguity which argparse does NOT report by default for hidden
    # aliases). Force users to type the flag in full.
    cua_parser = subparsers.add_parser(
        "cua",
        help="Native-accessibility computer-use agent with configurable planner",
        description=(
            "Run the computer-use agent loop. Fast thinking (outcome routing, "
            "fixation detection) is always local; slow thinking (planning) uses "
            "the preset or URL you choose: `rapid-mlx cua run --planner local-9b`."
        ),
    )
    cua_parser.add_argument(
        "cua_args",
        nargs=argparse.REMAINDER,
        help="arguments passed to the cua subcommand (run/config/planners)",
    )


def _add_serve_parser(
    subparsers: "argparse._SubParsersAction[_PortContextArgumentParser]",
) -> None:
    """Register the ``serve`` subcommand."""
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start OpenAI-compatible server",
        description=(
            "Start a local OpenAI-compatible inference server.\n"
            "\n"
            "  rapid-mlx serve qwen3.5-4b-4bit\n"
            "    <model>    pick yours: a short alias (rapid-mlx models) or HF repo\n"
            "    --port     bind port (default: first free in 8000-8009)\n"
            "    --host     bind host (default 127.0.0.1, loopback-only)\n"
            "    --api-key  require a bearer token on every request\n"
            "\n"
            "Once warmed up the server prints its 'Ready:' URL; the "
            "OpenAI-compatible\n"
            "endpoints (/v1/models, /v1/chat/completions, /v1/audio/*, ...) "
            "serve from\n"
            "that base URL. Most options below are advanced tuning; the "
            "common journey\n"
            "needs only a model (--port/--host default to local use)."
        ),
        epilog=(
            "First-time tips:\n"
            "  rapid-mlx models  lists what you can serve;\n"
            "  rapid-mlx recipe  recommends the best model for this Mac."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    serve_parser.add_argument(  # type: ignore[attr-defined]  # argcomplete attribute
        "model", nargs="?", type=str, help="Model to serve"
    ).completer = alias_completer
    serve_parser.add_argument(
        "--cua-only",
        action="store_true",
        help=(
            "Start the authenticated Computer Use API without resolving, "
            "downloading, or loading a model"
        ),
    )
    serve_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="assume yes for prompts such as installing a missing optional extra",
    )
    serve_parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="The model name used in the API. If not specified, the model argument is used.",
    )
    serve_parser.add_argument(
        "--force-disk-check",
        action="store_true",
        help=(
            "Skip the pre-flight disk-space check that aborts when the model "
            "is larger than free disk. Use only if you know the HF cache lives "
            "on a different filesystem (e.g. external drive via HF_HOME)."
        ),
    )
    serve_parser.add_argument(
        "--image-weight-precision",
        choices=("q4", "bf16"),
        default=None,
        help=(
            "Explicit FLUX.2 Klein weight precision. q4 keeps the compact "
            "default checkpoint; bf16 selects the full-precision checkpoint "
            "measured faster on an M2 Pro large-matrix image workload. "
            "Currently limited to FLUX.2 Klein; no automatic hardware switch."
        ),
    )
    # Disk-streaming MoE weight loading (PRD-rapid-mlx-integration.md).
    # Strictly opt-in: default behavior for every existing invocation is
    # unchanged. When set, the model loads lazily (routed-expert weights
    # never materialized) and rapid_mlx.disk_stream_patch.install() patches
    # its MoE blocks to stream selected experts off disk through a
    # byte-budgeted LRU cache instead of holding them resident — lets an
    # operator run a model whose declared min_memory_gb floor
    # (_check_alias_min_memory above) exceeds this Mac's RAM. Does NOT
    # suppress that warning: resident components (attention, KV cache,
    # dense layers, the cache budget itself) still consume real RAM.
    serve_parser.add_argument(
        "--disk-stream",
        action="store_true",
        default=False,
        help=(
            "Stream MoE routed-expert weights from disk instead of holding "
            "them resident (opt-in). Loads the model lazily and installs "
            "rapid_mlx.disk_stream_patch on every MoE layer before serving "
            "starts. Only architectures registered in rapid_mlx.registry "
            "are supported; an unregistered model_type fails at load time."
        ),
    )
    serve_parser.add_argument(
        "--request",
        action="store_true",
        default=False,
        help=(
            "If the pre-download check refuses a public Hugging Face model "
            "(unsupported architecture or GGUF/.bin-only), file a support "
            "request without asking. Sends only the repo id, architecture, "
            "format and Rapid-MLX version."
        ),
    )
    serve_parser.add_argument(
        "--no-preflight",
        action="store_true",
        default=False,
        help=(
            "Skip the pre-download check for models outside the Rapid-MLX "
            "catalog (format, architecture and memory fit)."
        ),
    )
    serve_parser.add_argument(
        "--disk-stream-cache-gb",
        type=positive_finite_float,
        default=1.0,
        help=(
            "Byte budget (GB) for the disk-stream expert LRU cache. Only "
            "used when --disk-stream is set. Default: 1.0 GB, matching "
            "rapid_mlx.expert_cache.ExpertCache's default."
        ),
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help=(
            "Host to bind (default: 127.0.0.1, loopback-only). Pass "
            '0.0.0.0 (or "") to expose the server on every '
            "interface (LAN reachable) — only do this once the "
            "bearer-auth posture has been reviewed. The wildcard "
            "bind also widens the PortSweep collision window: macOS "
            "lets a wildcard listener coexist with a more-specific "
            "(127.0.0.1) listener on the same port, so a second "
            "server may start and silently shadow the first on the "
            "loopback path. The pre-flight bind check below probes "
            "127.0.0.1 explicitly whenever --host is a wildcard "
            "alias to keep that bypass closed."
        ),
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=(
            "Port to bind (default when omitted: first free port in 8000-8009; "
            "an explicit port never falls back)"
        ),
    )
    _add_video_job_args(serve_parser)
    # Socket activation — let an external supervisor (launchd, systemd,
    # parent process) bind the listening socket and execve into
    # ``rapid-mlx`` with the pre-bound fd. This closes the bind→auth
    # TOCTOU window described in issue #574: no co-located process can
    # land an unauthenticated request between socket bind and FastAPI
    # auth dependency registration, because by the time
    # ``rapid-mlx serve`` runs, the app (with auth dependencies wired
    # into chat/embeddings/audio/models routers) is already constructed
    # before ``uvicorn.run`` starts ``accept()``-ing on the fd.
    #
    # When ``--listen-fd`` is set, ``--host``/``--port`` are IGNORED:
    # the supervisor controls the bind address. The "Ready:" banner
    # prints the inherited fd shape (``Ready: inherited fd N``) — NOT
    # the user-supplied host/port, since those don't reflect the
    # supervisor's actual bind. Setting both ``--listen-fd`` and a
    # non-default ``--port`` is allowed but the port has no effect;
    # the active listener is the inherited fd.
    serve_parser.add_argument(
        "--listen-fd",
        type=_listen_fd_arg,
        default=None,
        metavar="FD",
        help=(
            "File descriptor of a pre-bound listening socket (3-1023). "
            "Used for socket activation (launchd/systemd/parent-process "
            "supervision) — supervisor binds the loopback socket, "
            "validates auth secret, then execve's into rapid-mlx. "
            "When set, --host/--port are ignored for binding. Native MTP, "
            "DSpark K4, DFlash, and DDTree reject --listen-fd with rc 2."
        ),
    )
    serve_parser.add_argument(
        "--log-level",
        type=_log_level_choice,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level for Python logging and uvicorn (case-insensitive)",
    )
    serve_parser.add_argument(
        "--max-num-seqs", type=int, default=256, help="Max concurrent sequences"
    )
    serve_parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=256,
        help=(
            "Admission cap on in-flight requests (queued + running). When "
            "exceeded, new requests return HTTP 503 with Retry-After. "
            "Default 256; operators on memory-constrained devices may want "
            "to set this near ``--max-num-seqs`` to limit queue depth."
        ),
    )
    serve_parser.add_argument(
        "--prefill-batch-size",
        type=int,
        default=8,
        help=(
            "Max prompts prefilled together in one cold wave (default: 8). "
            "Lower it to cut first-token latency under concurrent cold load — "
            "requests start decoding sooner instead of all sharing one "
            "full-wave prefill — at an aggregate-throughput cost on large MoE "
            "models, where staggered rows carry ragged offsets that push "
            "batched attention onto a slower path (see #1861)."
        ),
    )
    serve_parser.add_argument(
        "--completion-batch-size", type=int, default=32, help="Completion batch size"
    )
    serve_parser.add_argument(
        "--scheduling-policy",
        choices=("fcfs", "shortest_validated_tail"),
        default="fcfs",
        help=(
            "Prompt-slot admission order (default: fcfs). "
            "shortest_validated_tail favors cache-hot and short prompts while "
            "bounding how often an older compatible request may be deferred."
        ),
    )
    serve_parser.add_argument(
        "--scheduling-max-deferrals",
        type=int,
        default=8,
        metavar="N",
        help=(
            "Maximum compatible prompt-slot grants that may pass over a request "
            "under shortest_validated_tail before it is forced FIFO (default: 8)."
        ),
    )
    serve_parser.add_argument(
        "--enable-prefix-cache",
        action="store_true",
        default=True,
        help="Enable prefix caching for repeated prompts (default: enabled)",
    )
    serve_parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help="Disable prefix caching",
    )
    serve_parser.add_argument(
        "--prefix-cache-size",
        type=int,
        default=100,
        help="Max entries in prefix cache (default: 100, legacy mode only)",
    )
    # Memory-aware cache options (recommended for large models)
    serve_parser.add_argument(
        "--cache-memory-mb",
        type=int,
        default=None,
        help="Cache memory limit in MB (default: auto-detect ~20%% of RAM)",
    )
    serve_parser.add_argument(
        "--cache-memory-percent",
        type=float,
        default=None,
        help=(
            "Fraction of available RAM for cache if auto-detecting (default: "
            "0.20, raised to the agent-session floor; an explicit value is kept)"
        ),
    )
    serve_parser.add_argument(
        "--idle-cache-clear-seconds",
        type=float,
        default=None,
        help=(
            "Clear reusable prefix/KV cache after this many seconds with no "
            "active requests, preserving loaded model weights. 0 disables; "
            "default: RAPID_MLX_IDLE_CACHE_CLEAR_SECONDS or disabled."
        ),
    )
    # #1103: bounded trim-free prefix reuse for "non-trimmable" cache entries.
    # Opt-in: the default 0 keeps the #1075 policy of dropping them at store
    # time. Two model families produce non-trimmable layers and both benefit:
    #   * hybrid recurrent-state (GatedDeltaNet / Mamba MoE) — ArraysCache;
    #   * sliding-window attention (Gemma 4, GPT-OSS) — RotatingKVCache once
    #     the ring has rotated (offset >= sliding_window → is_trimmable False).
    # The store gate and the trim-free fetch paths are class-agnostic — they
    # key off is_trimmable() — so lifting the store drop for N>0 recovers
    # within-conversation prefix reuse for BOTH families (generalized from
    # recurrent-state to sliding-window). NOTE: prefix-EXTENSION hits (stable
    # prefix + new suffix) are served trim-free; an EXACT re-request of a
    # rotated sliding-window prompt instead full-prefills, because the
    # scheduler's trim(1) exact-hit compensation is unavailable on a rotated
    # cache (see Scheduler._resolve_exact_hit_tokens).
    serve_parser.add_argument(
        "--hybrid-cache-entries",
        type=int,
        default=0,
        help=(
            "Retain up to N non-trimmable prefix-cache entries for "
            "prefix-extension reuse (a stable prefix + a new suffix each turn); "
            "0 disables (default: 0). Covers both hybrid recurrent-state "
            "(GatedDeltaNet/Mamba) AND sliding-window (Gemma 4, GPT-OSS) "
            "models. Best for stable-system-prompt / long-context agent "
            "workloads. An identical exact re-request of a rotated "
            "sliding-window prompt falls back to a full prefill (byte-equal to "
            "cold)."
        ),
    )
    # Operator override for the D-METAL-CAP admission projection. The
    # auto-derived figure assumes an UNCOMPRESSED fp16 KV cache — see
    # ``Scheduler._infer_kv_dtype_bytes``, which documents that quantized-KV
    # deployments are not auto-detected and names this knob as the escape
    # hatch. It was reachable only from the Python API, so a CLI user running
    # ``--kv-cache-turboquant`` / ``--kv-cache-quantization`` got an admission
    # projection that ignored the codec entirely and 503'd long prompts the
    # codec would have fit. Default 0 preserves auto-derivation exactly.
    serve_parser.add_argument(
        "--metal-cap-kv-bytes-per-token",
        type=non_negative_int,
        default=0,
        metavar="BYTES",
        help=(
            "Override the per-token KV-cache size the D-METAL-CAP admission "
            "gate projects, in bytes. 0 (default) auto-derives an "
            "architecture-aware fp16 figure. Set this when running a "
            "quantized KV cache (--kv-cache-turboquant / "
            "--kv-cache-quantization), whose real footprint the auto-derived "
            "figure over-estimates — an over-estimate only costs you spurious "
            "503s, but on a memory-tight Mac that is the difference between a "
            "long prompt being served and being rejected. UNDER-setting it "
            "risks the OOM cliff the gate exists to prevent: lower it only to "
            "a value you have measured. Overrides the architecture-aware "
            "estimator wholesale (sliding-window and recurrent terms included)."
        ),
    )
    # Opt-in prompt-deterministic RESPONSE CACHE (exact-match short-circuit).
    # Distinct from the prefix/KV cache above: this returns the ENTIRE stored
    # completion for a completely repeated GREEDY request (temperature==0 or
    # top_k==1), doing zero GPU decode. Default 0 = fully disabled.
    serve_parser.add_argument(
        "--response-cache-entries",
        type=non_negative_int,
        default=0,
        help=(
            "Retain up to N fully-computed deterministic (greedy) chat "
            "responses; a completely repeated request returns the stored "
            "completion verbatim with zero GPU decode. 0 disables (default: 0). "
            "Only temperature==0 / top_k==1 requests are cached — sampled "
            "requests are never short-circuited."
        ),
    )
    serve_parser.add_argument(
        "--no-memory-aware-cache",
        action="store_true",
        help="Disable memory-aware cache, use legacy entry-count based cache",
    )
    # R15-P1 (task #303): radix-tree prefix-cache index. Default ``radix``
    # accelerates lookup and accounts for cross-request prefix dedup on
    # shared-system-prompt workloads. ``hash`` is the legacy bisect path,
    # kept as an escape hatch if a regression is found in production.
    serve_parser.add_argument(
        "--prefix-cache-index",
        type=str,
        default="radix",
        choices=("radix", "hash"),
        help=(
            "Prefix-cache lookup index: 'radix' (default, R15-P1) uses a "
            "token trie for O(prefix_len) lookups and surfaces dedup-bytes-"
            "saved on /metrics; 'hash' falls back to the legacy bisect-over-"
            "sorted-keys path."
        ),
    )
    serve_parser.add_argument(
        "--mllm-singleton-fastpath",
        type=str,
        default="auto",
        choices=["auto", "off"],
        help=(
            "Serialized MLLM lane cache handling (default: auto). 'auto' "
            "skips repacking a single eligible request's cache leaves into "
            "batched form (structural B=1 batches on the serialized hybrid "
            "lane only; dense lanes keep the merge); 'off' always takes the "
            "legacy merge/rebatch path. Operator rollback for the singleton "
            "fast path."
        ),
    )
    serve_parser.add_argument(
        "--mllm-media-prefix-cache",
        type=str,
        default="auto",
        choices=["auto", "off"],
        help=(
            "Prior-turn media boundary reuse on the serialized MLLM lane "
            "(default: auto). 'auto' snapshots each eligible media prefill's "
            "stable turn boundary and resumes it when the next turn verifies "
            "as a strict token prefix; 'off' disables store and lookup and "
            "keeps the cold image path. Operator rollback for the media "
            "boundary cache."
        ),
    )
    # KV cache quantization options
    # ``--kv-cache-dtype`` (R15 task #300) is the canonical knob. Default
    # is bf16 (#1853): the R15 int4 default was justified by "4×-smaller
    # KV cuts decode bandwidth proportionally", but the live serve path
    # (QuantizedBatchKVCache, #1197) implements quantization as
    # dequant-on-read — it MATERIALIZES full-precision K/V on every
    # decode step, so per-token cost grows with context instead of
    # shrinking. Measured on qwen3.5-4b, 16k context, N=2 (parity
    # server, disk checkpoints off): bf16 134.6 tok/s, int4 98.2
    # (-27%), int8 86.1 (-36%); at 128-ctx: bf16 167, int4 161,
    # int8 160. The #910 numbers that motivated the int4 default were
    # short-context (292-tok prompt), where the regression is invisible.
    # int4/int8 remain available as explicit opt-ins for
    # memory-constrained hosts (KV is 4×/2× smaller); ``--reasoning``
    # still pins to int8.
    serve_parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "int8", "int4"],
        help=(
            "KV cache dtype (R15 #300, default: bf16). int8/int4 shrink the "
            "KV cache 2x/4x for memory-constrained hosts, but the live-cache "
            "dequant-on-read costs O(context) per decode step — measured "
            "-27%% (int4) / -36%% (int8) at 16k context (#1853). "
            "An explicit int8/int4 on a sliding-window (Gemma 3/4, "
            "GPT-OSS) or MLA (DeepSeek V3+, Kimi K2.5) model is rejected "
            "before the server reports ready; only auto/profile-selected "
            "quantization downgrades to bf16. Use --reasoning "
            "for AIME / hard math."
        ),
    )
    serve_parser.add_argument(
        "--reasoning",
        action="store_true",
        default=False,
        help=(
            "Reasoning profile: pins --kv-cache-dtype to int8 regardless of "
            "the dtype flag (sub-4-bit drops -20pt on AIME-class math for "
            "Qwen3 thinking variants)."
        ),
    )
    serve_parser.add_argument(
        "--kv-cache-quantization",
        action="store_true",
        help=(
            "[deprecated alias of --kv-cache-dtype int8] Quantize stored "
            "KV caches to reduce memory (8-bit by default). When both "
            "flags are passed, this one wins for backwards compatibility."
        ),
    )
    serve_parser.add_argument(
        "--kv-cache-quantization-bits",
        type=int,
        default=8,
        choices=[4, 8],
        help="Bit width for KV cache quantization (default: 8)",
    )
    serve_parser.add_argument(
        "--kv-cache-quantization-group-size",
        type=int,
        default=64,
        help="Group size for KV cache quantization (default: 64)",
    )
    serve_parser.add_argument(
        "--kv-cache-min-quantize-tokens",
        type=int,
        default=256,
        help="Minimum tokens for quantization to apply (default: 256)",
    )
    # TurboQuant KV cache compression (experimental, R15 Phase 4).
    #
    # Accepts an optional mode value:
    #   --kv-cache-turboquant              → V-only legacy (v4)
    #   --kv-cache-turboquant v4           → V-only explicit
    #   --kv-cache-turboquant k8v4         → K-8bit + V-4bit mix (R15 Phase 4)
    #   --kv-cache-turboquant none         → explicit off-switch (overrides
    #                                        alias ``turboquant_tier=k8v4_verified``
    #                                        auto-resolution; see
    #                                        ``resolve_turboquant_mode_default``)
    #
    # The bare-flag form preserves PR #157 backward compatibility. Mode
    # is mutually exclusive with --kv-cache-quantization.
    serve_parser.add_argument(
        "--kv-cache-turboquant",
        nargs="?",
        const="v4",
        default=None,
        choices=["v4", "k8v4", "none"],
        help="Enable TurboQuant KV-cache compression. ``v4`` (default when "
        "the flag is bare) is V-only 3-4 bit Lloyd-Max with K in FP16; "
        "``k8v4`` is the R15 Phase 4 mix — K at 8-bit Walsh-Hadamard + V at "
        "4-bit Lloyd-Max (~4.6x KV compression on dense models); ``none`` "
        "is the explicit off-switch — overrides the alias-driven "
        "``turboquant_tier=k8v4_verified`` default so the operator can A/B "
        "the bare FP16 KV path. Experimental — mutually exclusive with "
        "--kv-cache-quantization.",
    )
    serve_parser.add_argument(
        "--kv-cache-turboquant-bits",
        type=int,
        default=None,
        choices=[3, 4],
        help="V-side bit width for TurboQuant (default: auto-select by head_dim — "
        "3-bit for head_dim>=96, 4-bit for head_dim=64). Ignored when "
        "--kv-cache-turboquant=k8v4 (V is pinned to 4-bit there).",
    )
    serve_parser.add_argument(
        "--kv-cache-turboquant-group-size",
        type=int,
        default=32,
        help="Group size for TurboQuant V-side quantization (default: 32)",
    )
    # R15-P1 (task #296): disk-backed KV checkpointing. 0 (default)
    # disables the feature entirely (no scheduler-hot-path cost, no
    # ~/.cache/rapid-mlx/kv_checkpoints/ directory creation). Opt-in
    # only: each snapshot serializes the full KV cache synchronously on
    # the decode thread — O(context) per boundary, which degraded 16k
    # decode by up to 45% when this defaulted to 256 (#1853). When
    # enabling, use a multiple of 256 to match MLX-LM's KVCache.step and
    # LMCache's external-chunk size so the on-disk shape aligns with the
    # in-memory shape on reload.
    serve_parser.add_argument(
        "--kv-disk-checkpoint-interval",
        type=int,
        default=0,
        help=(
            "Token interval at which the scheduler snapshots KV state to "
            "~/.cache/rapid-mlx/kv_checkpoints/ (R15 #296). 0 (default) "
            "disables. Write-only today: no engine path reloads the "
            "snapshots yet, and each one blocks decode for O(context) — "
            "enable only for external tooling that consumes the files "
            "(#1853). Pairs with the RAPID_MLX_KV_CHECKPOINT_MAX_BYTES "
            "env var (default 20 GiB) for the oldest-first disk-cap "
            "eviction policy."
        ),
    )
    serve_parser.add_argument(
        "--stream-interval",
        type=int,
        default=1,
        help="Tokens to batch before streaming (1=smooth, higher=throughput)",
    )
    serve_parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Default max tokens for generation (default: 32768)",
    )
    serve_parser.add_argument(
        "--speculative-config",
        dest="speculative_config",
        default=None,
        help=(
            "vLLM-style speculative decoding JSON config. This frontend "
            "parses method/model/num_speculative_tokens now. DFlash "
            "requires the rapid-mlx[dflash] extra and is available with "
            '\'{"method":"dflash"}\', DDTree with '
            '\'{"method":"ddtree"}\', and MTP with '
            '\'{"method":"mtp","num_speculative_tokens":3,'
            '"disable_auto_k":false,"continuous_batching":false,'
            '"allow_dynamic_membership":false}\'. '
            "Continuous self-MTP and dynamic membership are default-off. "
            "SuffixDecoding is an explicit, "
            "workload-specific flag for high prompt/output-overlap traffic "
            "and is available with "
            '\'{"method":"suffix","num_speculative_tokens":8}\'.'
        ),
    )
    # Hidden deprecated aliases. They are intentionally absent from help;
    # normalization folds them into the same SpeculativeConfig path as
    # --speculative-config so old commands do not revive old implementations.
    serve_parser.add_argument(
        "--enable-dflash",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--enable-ddtree",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--spec-decode",
        dest="spec_decode",
        choices=["none", "dflash", "mtp"],
        default="none",
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--dflash-drafter-path",
        default="",
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--enable-mtp",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--mtp-num-draft-tokens",
        type=int,
        default=1,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--mtp-optimistic",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--mtp-sidecar",
        default=None,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--mtp-max-k",
        dest="mtp_max_k",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--mtp-disable-auto-k",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--suffix-decoding",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--suffix-max-draft",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--suffix-max-suffix-len",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--suffix-min-confidence",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--suffix-min-draft-len",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    # Deprecated no-op flags — accepted-but-ignored for backward compat.
    # These once controlled removed engine paths (the single BatchedEngine,
    # legacy KV-bit quant, the --draft-model / --num-draft-tokens speculation
    # frontend, the --specprefill prototype, and the legacy chunked-prefill
    # monkey-patch that mlx-lm 0.31+ made unreachable). The implementations are
    # gone, but the launcher must still PARSE these flags without an argparse
    # hard-fail so existing user launch scripts (and older docs) keep booting.
    # They are consumed-and-discarded: stored on ``args`` but never read. Hidden
    # from --help (argparse.SUPPRESS); slated for removal in a future release.
    serve_parser.add_argument(
        "--continuous-batching",
        action="store_true",
        default=True,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--simple-engine",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--kv-bits",
        type=int,
        default=None,
        choices=[4, 8],
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--kv-group-size",
        type=int,
        default=64,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--draft-model",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=4,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--specprefill",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--specprefill-threshold",
        type=int,
        default=8192,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--specprefill-keep-pct",
        type=float,
        default=0.3,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--specprefill-draft-model",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--chunked-prefill-tokens",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Fraction of device memory for the Metal allocation limit and "
        "admission cap (0.0-1.0). Default: auto — the budget is sized to the "
        "loaded model (measured weights + headroom, between 0.90 and 0.97 of "
        "the device working-set budget). Pass an explicit value only as an "
        "advanced override.",
    )
    serve_parser.add_argument(
        "--resident-memory-limit-gb",
        type=float,
        default=0.0,
        help=(
            "Process-wide resident model ceiling in GiB. Loading another model "
            "evicts the least-recently-used idle unpinned model first. 0 disables "
            "the ceiling (default: 0)."
        ),
    )
    serve_parser.add_argument(
        "--resident-model-idle-ttl",
        type=float,
        default=0.0,
        help=(
            "Evict idle unpinned secondary models after this many seconds. "
            "0 disables idle eviction (default: 0)."
        ),
    )
    serve_parser.add_argument(
        "--lazy-load",
        action="store_true",
        help=(
            "Bind the API endpoint without loading the configured model's "
            "weights; the first inference request loads and warms it."
        ),
    )
    serve_parser.add_argument(
        "--idle-unload-seconds",
        type=float,
        default=0.0,
        help=(
            "Release the configured primary model after this many idle "
            "seconds while keeping the API endpoint online. A later request "
            "reloads it. 0 disables primary idle unload (default: 0)."
        ),
    )
    # Paged cache options (experimental)
    serve_parser.add_argument(
        "--use-paged-cache",
        action="store_true",
        help=(
            "Use paged KV cache for memory efficiency (experimental). "
            "Requires a model whose prompt cache is plain full-attention "
            "KV on every layer; startup fails with an actionable error "
            "for rotating/hybrid/recurrent/quantized cache layouts."
        ),
    )
    serve_parser.add_argument(
        "--paged-cache-block-size",
        type=int,
        default=64,
        help="Tokens per cache block (default: 64)",
    )
    serve_parser.add_argument(
        "--max-cache-blocks",
        type=int,
        default=1000,
        help="Maximum number of cache blocks (default: 1000)",
    )
    # Task #292: opt-in for ``/v1/audio/*`` routes on a text-only server.
    # The audio-mode boot path (``rapid-mlx serve kokoro`` etc.) auto-
    # enables the routes via the registry hit — this flag is the
    # escape hatch for operators who want the audio router mounted
    # alongside a text engine (e.g. side-car deployments that proxy the
    # audio paths to a separate process).
    serve_parser.add_argument(
        "--enable-audio",
        action="store_true",
        default=False,
        help="Mount the ``/v1/audio/*`` routes even when the loaded model "
        "is text-only. Useful for side-car deployments that proxy audio "
        "requests to a separate process. Audio-capable models "
        "(kokoro / whisper / parakeet / chatterbox / vibevoice / voxcpm) "
        "auto-mount the routes — this flag is only needed on text-mode boots.",
    )
    # Prefill step size
    serve_parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=2048,
        help="Chunk size for prompt prefill processing. Larger values use more memory "
        "but can improve prefill throughput. (default: 2048; bench-verified model "
        "profiles may recommend a smaller value unless explicitly set)",
    )
    serve_parser.add_argument(
        "--vision-prefill-token-budget",
        type=positive_int,
        default=None,
        help=(
            "Advanced: maximum prompt tokens per vision-bearing request. "
            "Defaults to 8192 for automatic profiles; an explicit "
            "--prefill-step-size preserves the legacy shared limit."
        ),
    )
    serve_parser.add_argument(
        "--vision-min-pixels",
        type=non_negative_int,
        default=0,
        help=(
            "Minimum pixels used by dynamic-resolution VLM image processors. "
            "0 keeps the model default (default: 0)."
        ),
    )
    serve_parser.add_argument(
        "--vision-max-pixels",
        type=non_negative_int,
        default=0,
        help=(
            "Maximum pixels used by dynamic-resolution VLM image processors. "
            "Lower values trade image detail for lower TTFT and memory. "
            "0 keeps the model default (default: 0)."
        ),
    )
    # MCP options
    serve_parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP configuration file (JSON/YAML) for tool integration",
    )
    # Security options
    # ``--api-key`` accepts an inline value OR falls back to the
    # ``RAPID_MLX_API_KEY`` env var. ``rapid-mlx share`` uses the env-var
    # form so the bearer key never lands in argv (visible to ``ps`` for
    # any local user). Inline value still works for backwards-compat
    # with existing scripts; if both are set, the inline value wins.
    serve_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "API key for authentication (if not set, falls back to the "
            "RAPID_MLX_API_KEY env var; if neither, no auth required)"
        ),
    )
    serve_parser.add_argument(
        "--cors-origins",
        type=str,
        nargs="+",
        default=None,
        metavar="ORIGIN",
        help=(
            "Allowed CORS origins (default: * for all origins). "
            "Example: --cors-origins http://localhost:3000 https://myapp.com"
        ),
    )
    serve_parser.add_argument(
        "--trusted-hosts",
        type=str,
        nargs="+",
        default=None,
        metavar="HOST",
        help=(
            "OPT-IN Host-header allowlist (DNS-rebinding hardening): only "
            "requests whose Host header matches one of these values are "
            "accepted; everything else gets 400. Off by default so "
            "rapid-mlx share and LAN access keep working. Values may be "
            "space- or comma-separated. Example: --trusted-hosts localhost "
            "127.0.0.1 (also settable via "
            "RAPID_MLX_TRUSTED_HOSTS)."
        ),
    )
    serve_parser.add_argument(
        "--rate-limit",
        type=int,
        default=0,
        help="Rate limit requests per minute per client (0 = disabled)",
    )
    # Hard cap on per-request body size — DoS defense.
    # See ``rapid_mlx/middleware/body_size.py`` for the rationale (pre-fix:
    # a 10 MB body silently ran a ~60 s full prefill on a 27B alias before
    # the client timed out; rapid-desktop#273 + #463). Default 8 MiB fits
    # a 128k-token prompt with tool schemas; 0 disables the cap.
    serve_parser.add_argument(
        "--max-request-bytes",
        type=int,
        default=None,
        help=(
            "Maximum HTTP request body size in bytes (default: 8 MiB = "
            "8388608). Requests over this cap are rejected with HTTP 413 "
            "before JSON parsing or tokenization runs. 0 disables the cap. "
            "Falls back to the RAPID_MLX_MAX_REQUEST_BYTES env var if unset."
        ),
    )
    serve_parser.add_argument(
        "--max-prompt-tokens",
        type=positive_int,
        default=None,
        metavar="TOKENS",
        help=(
            "Operational prompt-token admission ceiling. Requests above this "
            "limit are rejected with HTTP 400 context_length_exceeded before "
            "prefill, even when the model supports a larger context window."
        ),
    )
    serve_parser.add_argument(
        "--context-length",
        type=positive_int,
        default=None,
        metavar="TOKENS",
        help=(
            "Per-request context window (prompt plus output), up to the model's "
            "declared limit. Overrides the automatic memory estimate; an "
            "explicit --gpu-memory-utilization cap still applies."
        ),
    )
    serve_parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="Default request timeout in seconds (default: 1800 = 30 min)",
    )
    # Tool calling options
    serve_parser.add_argument(
        "--enable-auto-tool-choice",
        action="store_true",
        help="Enable auto tool choice for supported models. Use --tool-call-parser to specify which parser to use.",
    )
    serve_parser.add_argument(
        "--tool-call-parser",
        type=str,
        default=None,
        # Choices NOT enforced at argparse level — the canonical set is the
        # ToolParserManager registry, which has ~39 entries (canonical
        # names + per-family aliases like ``deepseek_v31``, ``llama4``,
        # ``moonshot`` for kimi, ``nous`` for hermes). The argparse hard-
        # coded list drifted to 19 over multiple releases and rejected
        # legitimate aliases users discovered via ``rapid-mlx info``.
        # Validation now happens post-parse in
        # ``_validate_tool_call_parser_choice`` against the live registry.
        # v0.6.63 onboarding sweep finding #1.
        help=(
            "Select the tool call parser for the model. Canonical options: "
            "auto (auto-detect), mistral, qwen/qwen3/qwen3_xml (reasoning models, "
            "<tool_call>JSON</tool_call> format), qwen3_coder/qwen3_coder_xml "
            "(Coder model, <function=NAME> XML format), llama/llama3/llama4, "
            "hermes/nous, deepseek/deepseek_v3/deepseek_v31, kimi/moonshot/kimi_k2, "
            "granite/granite3, nemotron/nemotron3, xlam, functionary/meetkai, "
            "glm47/glm4, minimax/minimax_m2, harmony/gpt-oss/gpt_oss, "
            "gemma4/gemma_4, seed_oss/seed. "
            "Run `python -c 'from rapid_mlx.tool_parsers import ToolParserManager;"
            "print(sorted(ToolParserManager.tool_parsers))'` for the live list. "
            "Required for --enable-auto-tool-choice."
        ),
    )
    # Tool logits bias (jump-forward decoding for tool call structural tokens)
    serve_parser.add_argument(
        "--enable-tool-logits-bias",
        action="store_true",
        default=False,
        help="Bias logits toward structural tool call tokens for faster generation. "
        "Only active when --tool-call-parser is also set. Currently supports minimax.",
    )
    # Reasoning parser options - choices loaded dynamically from registry
    from .api.models import _VALID_REASONING_EFFORTS
    from .reasoning import list_parsers

    reasoning_choices = list_parsers()
    serve_parser.add_argument(
        "--reasoning-parser",
        type=str,
        default=None,
        choices=reasoning_choices,
        help=(
            "Enable reasoning content extraction with specified parser. "
            "Extracts <think>...</think> tags into reasoning_content field. "
            f"Options: {', '.join(reasoning_choices)}."
        ),
    )
    serve_parser.add_argument(
        "--default-reasoning-effort",
        type=str,
        default=None,
        choices=list(_VALID_REASONING_EFFORTS),
        metavar="EFFORT",
        help=(
            "OpenAI reasoning_effort applied to requests that send no "
            "reasoning knob (reasoning_effort / reasoning_max_tokens / "
            "enable_thinking / chat_template_kwargs.reasoning_effort). "
            "Translated exactly like a client value: a template that "
            "publishes its own effort levels (GLM-5.3, Qwen3.8) gets the "
            "nearest native level in the prompt, any other template gets "
            "the matching thinking-token cap. Use it for models whose "
            "template default is the most expensive level (GLM-5.3 renders "
            "'Reasoning Effort: Max' unless told otherwise). Options: "
            "none, minimal, low, medium, high, xhigh."
        ),
    )
    serve_parser.add_argument(
        "--no-thinking",
        action="store_true",
        default=False,
        help=(
            "Disable reasoning/thinking parser even if auto-detected. "
            "Thinking tokens will appear as regular content. "
            "Useful for faster responses when chain-of-thought is not needed."
        ),
    )
    # Hidden cross-alias mirroring ``chat --no-thinking`` (see the chat
    # parser for the full rationale). ``serve --no-think`` lands on the
    # same ``no_thinking`` destination so users who reach for the shorter
    # name don't get an ``unrecognized arguments`` error.
    serve_parser.add_argument(
        "--no-think",
        dest="no_thinking",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--no-tool-call-parser",
        dest="no_tool_call_parser",
        action="store_true",
        default=False,
        help=(
            "Force-disable tool-call parser auto-detection from the alias "
            "profile. Escape hatch (SOP §10) when AliasProfile's auto-"
            "selected parser misfires for a specific deployment. Mutually "
            "exclusive with --tool-call-parser."
        ),
    )
    serve_parser.add_argument(
        "--no-reasoning-parser",
        dest="no_reasoning_parser",
        action="store_true",
        default=False,
        help=(
            "Force-disable reasoning parser auto-detection from the alias "
            "profile. Distinct from --no-thinking (which also suppresses "
            "the chain-of-thought prompt template) — this flag ONLY skips "
            "the auto-config step. Mutually exclusive with --reasoning-parser."
        ),
    )
    # SOP §10 profile-override escape hatches. Pair every binary
    # auto-routing field with both force-on and force-off CLI flags so
    # users always have an override path when the AliasProfile
    # auto-detection misfires. Registered in
    # tests/test_no_mllm_flag.py::test_auto_routing_flags_have_force_on_and_force_off_pair.
    serve_parser.add_argument(
        "--force-hybrid",
        dest="force_hybrid",
        action="store_true",
        default=False,
        help=(
            "Force-treat the model as a hybrid (linear-attention / Mamba) "
            "architecture even when AliasProfile says otherwise. Disables "
            "spec/suffix decode paths that are unsound on hybrids. "
            "Mutually exclusive with --no-hybrid."
        ),
    )
    serve_parser.add_argument(
        "--no-hybrid",
        dest="no_hybrid",
        action="store_true",
        default=False,
        help=(
            "Force-treat the model as non-hybrid (full attention) even when "
            "AliasProfile says it's hybrid. Use when the profile mis-labels "
            "your model and you want spec/suffix decode enabled. "
            "Mutually exclusive with --force-hybrid."
        ),
    )
    serve_parser.add_argument(
        "--force-spec-decode",
        dest="force_spec_decode",
        action="store_true",
        default=False,
        help=(
            "Force-enable speculative-decode eligibility even when "
            "AliasProfile says the model doesn't support it. Risky on "
            "hybrid models — use only when you've verified the profile "
            "is wrong. Mutually exclusive with --no-spec-decode."
        ),
    )
    serve_parser.add_argument(
        "--no-spec-decode",
        dest="no_spec_decode",
        action="store_true",
        default=False,
        help=(
            "Force-disable speculative-decode eligibility (suffix / MTP / "
            "DFlash / DDTree) even when AliasProfile says the model supports it. "
            "Mutually exclusive with --force-spec-decode."
        ),
    )
    # #516 — HarmonyStreamingRouter auto-upgrade escape hatches (G11).
    # PR #515 introduced an auto-upgrade from the legacy harmony state
    # machine to openai-harmony's StreamableParser for matched-vocab
    # gpt-oss tokenizers. The auto-detection is conservative (three-layer
    # compat check) but the SOP requires every binary auto-routing
    # decision expose both force-on and force-off CLI flags.
    serve_parser.add_argument(
        "--force-openai-harmony-streaming",
        dest="force_openai_harmony_streaming",
        action="store_true",
        default=False,
        help=(
            "Force-on: construct HarmonyStreamingRouter even when the "
            "compat gate would reject. Use to debug a regression in the "
            "gate itself; production should leave this off. Mutually "
            "exclusive with --no-openai-harmony-streaming."
        ),
    )
    serve_parser.add_argument(
        "--no-openai-harmony-streaming",
        dest="no_openai_harmony_streaming",
        action="store_true",
        default=False,
        help=(
            "Force-off: skip the HarmonyStreamingRouter upgrade and use "
            "the legacy custom harmony state machine even on matched-vocab "
            "gpt-oss tokenizers. Escape hatch for a hypothetical false "
            "positive in the compat gate. Mutually exclusive with "
            "--force-openai-harmony-streaming."
        ),
    )
    # GC control (Tier 0 optimization)
    serve_parser.add_argument(
        "--gc-control",
        action="store_true",
        default=True,
        help="Enable Python GC pausing during generation to avoid latency spikes (default: enabled)",
    )
    serve_parser.add_argument(
        "--no-gc-control",
        action="store_true",
        help="Disable GC control (allow normal Python GC during generation)",
    )
    # Pinned prefix cache (Tier 0 optimization)
    serve_parser.add_argument(
        "--pin-system-prompt",
        action="store_true",
        default=False,
        help="Auto-pin system prompt in prefix cache to prevent eviction under memory pressure",
    )
    serve_parser.add_argument(
        "--relocate-mid-conversation-system",
        action="store_true",
        default=False,
        help=(
            "Keep a mid-conversation system message at its position (folded "
            "into the next user turn) instead of hoisting it into the leading "
            "system block. Preserves the prefix cache for clients that inject "
            "reminders mid-session (Claude Code); OFF by default because the "
            "relocated text carries user authority rather than system "
            "authority."
        ),
    )
    # Multimodal option
    serve_parser.add_argument(
        "--mllm",
        action="store_true",
        help="Force load model as multimodal (vision) even if name doesn't match auto-detection patterns. Also DISABLES automatic text-only fallbacks and alias-owned speculative-decoding defaults; an explicitly requested speculative decoder conflicts and is rejected. A vision-config checkpoint with no usable vision tower hard-fails instead of silently downgrading (#1187).",
    )
    serve_parser.add_argument(
        "--no-mllm",
        "--text-only",
        dest="no_mllm",
        action="store_true",
        help="Force load model as text-only LLM even when auto-detection would route it to the multimodal/VLM path. Escape hatch for incomplete vision-tower checkpoints (#393) and text-only forks of multimodal architectures whose config.json still declares vision_config.",
    )
    # Generation defaults
    serve_parser.add_argument(
        "--default-temperature",
        type=float,
        default=None,
        help="Override default temperature for all requests (default: use model default)",
    )
    serve_parser.add_argument(
        "--default-top-p",
        type=float,
        default=None,
        help="Override default top_p for all requests (default: use model default)",
    )
    serve_parser.add_argument(
        "--default-top-k",
        type=int,
        default=None,
        help="Override default top_k for all requests (default: use model default)",
    )
    serve_parser.add_argument(
        "--default-min-p",
        type=float,
        default=None,
        help="Override default min_p for all requests (default: use model default)",
    )
    serve_parser.add_argument(
        "--default-repetition-penalty",
        type=float,
        default=None,
        help="Override default repetition_penalty for all requests (default: use model default)",
    )
    serve_parser.add_argument(
        "--default-presence-penalty",
        type=float,
        default=None,
        help="Override default presence_penalty for all requests (default: use model default)",
    )
    serve_parser.add_argument(
        "--default-frequency-penalty",
        type=float,
        default=None,
        help="Override default frequency_penalty for all requests (default: use model default)",
    )
    # Embedding model option
    serve_parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help=(
            "Pre-load an embedding model at startup (e.g. "
            "mlx-community/embeddinggemma-300m-6bit). Requires the "
            "[embeddings] extra. "
            + optional_extra_install_hint("embeddings", include_paths=False)
        ),
    )
    # Embedding input-length controls (issue #1381). Prevents silent
    # 512-token truncation: derive a model-aware limit and make overflow
    # observable / configurable.
    serve_parser.add_argument(
        "--embedding-max-length",
        type=str,
        default="auto",
        metavar="TOKENS",
        help=(
            "Max input length (tokens) for --embedding-model. 'auto' "
            "(default) derives it from the model's declared maximum "
            "(config.max_position_embeddings, else tokenizer.model_max_length); "
            "or pass a positive integer to set a lower operational ceiling. "
            "Inputs above the effective limit are handled per "
            "--embedding-overflow-policy (never truncated silently)."
        ),
    )
    serve_parser.add_argument(
        "--embedding-overflow-policy",
        type=str,
        choices=["truncate", "error"],
        default="truncate",
        help=(
            "How to handle embedding inputs longer than "
            "--embedding-max-length: 'truncate' (default) discards the tail "
            "but logs a warning and increments the "
            "rapid_mlx_embedding_truncations_total metric (never silent); "
            "'error' rejects the request with a 400 carrying the observed "
            "and allowed token counts."
        ),
    )
    # Parent-PID watchdog (rapid-desktop issue #449). When set, the
    # sidecar polls ``os.getppid()`` every 2 s and self-terminates if
    # the parent dies (re-parent to launchd / init on macOS/Linux). The
    # supervisor passes its own PID at spawn so a SIGKILL on the desktop
    # cannot leave a 30 GB orphan holding the model + port. ``0`` /
    # negative / unset disables. The ``RAPID_MLX_WATCHDOG_PPID`` env var
    # is honoured as a fallback when the CLI flag is omitted; the flag
    # wins when both are present.
    serve_parser.add_argument(
        "--watchdog-ppid",
        type=int,
        default=None,
        metavar="PID",
        help=(
            "Self-terminate when the parent with this PID dies (defeats "
            "orphan-sidecar after SIGKILL on the supervisor). Honors "
            "$RAPID_MLX_WATCHDOG_PPID as a fallback. Set to 0 / unset to "
            "disable."
        ),
    )
    # PFlash long-prompt prefill compression (#287). Off by default; see
    # rapid_mlx/pflash.py for the design and the prefix-cache bypass.
    _add_pflash_args(serve_parser)


def _add_bench_parser(
    subparsers: "argparse._SubParsersAction[_PortContextArgumentParser]",
) -> None:
    """Register the ``bench`` subcommand."""
    # Bench command
    bench_parser = subparsers.add_parser("bench", help="Run benchmark")
    bench_parser.add_argument(  # type: ignore[attr-defined]  # argcomplete attribute
        "model", type=str, help="Model to benchmark"
    ).completer = alias_completer

    bench_parser.add_argument(
        "--force-disk-check",
        action="store_true",
        help=(
            "Skip the pre-flight disk-space check that aborts when the model "
            "is larger than free disk. Use only if you know the HF cache lives "
            "on a different filesystem (e.g. external drive via HF_HOME)."
        ),
    )
    # Disk-streaming MoE weight loading — same opt-in flags as `serve`,
    # see the `serve_parser` registration above for the full rationale.
    bench_parser.add_argument(
        "--disk-stream",
        action="store_true",
        default=False,
        help=(
            "Stream MoE routed-expert weights from disk instead of holding "
            "them resident (opt-in). See `rapid-mlx serve --help`."
        ),
    )
    bench_parser.add_argument(
        "--disk-stream-cache-gb",
        type=positive_finite_float,
        default=1.0,
        help="Byte budget (GB) for the disk-stream expert LRU cache.",
    )
    bench_parser.add_argument(
        "--num-prompts", type=int, default=10, help="Number of prompts"
    )
    bench_parser.add_argument(
        "--max-tokens", type=int, default=100, help="Max tokens per prompt"
    )
    bench_parser.add_argument(
        "--max-num-seqs", type=int, default=32, help="Max concurrent sequences"
    )
    bench_parser.add_argument(
        "--prefill-batch-size", type=int, default=8, help="Prefill batch size"
    )
    bench_parser.add_argument(
        "--completion-batch-size", type=int, default=16, help="Completion batch size"
    )
    bench_parser.add_argument(
        "--enable-prefix-cache",
        action="store_true",
        default=True,
        help="Enable prefix caching (default: enabled)",
    )
    bench_parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help="Disable prefix caching",
    )
    bench_parser.add_argument(
        "--prefix-cache-size",
        type=int,
        default=100,
        help="Max entries in prefix cache (default: 100, legacy mode only)",
    )
    # Memory-aware cache options (recommended for large models)
    bench_parser.add_argument(
        "--cache-memory-mb",
        type=int,
        default=None,
        help="Cache memory limit in MB (default: auto-detect ~20%% of RAM)",
    )
    bench_parser.add_argument(
        "--cache-memory-percent",
        type=float,
        default=None,
        help=(
            "Fraction of available RAM for cache if auto-detecting (default: "
            "0.20, raised to the agent-session floor; an explicit value is kept)"
        ),
    )
    bench_parser.add_argument(
        "--no-memory-aware-cache",
        action="store_true",
        help="Disable memory-aware cache, use legacy entry-count based cache",
    )
    # KV cache quantization options
    bench_parser.add_argument(
        "--kv-cache-quantization",
        action="store_true",
        help="Quantize stored KV caches to reduce memory (8-bit by default)",
    )
    bench_parser.add_argument(
        "--kv-cache-quantization-bits",
        type=int,
        default=8,
        choices=[4, 8],
        help="Bit width for KV cache quantization (default: 8)",
    )
    bench_parser.add_argument(
        "--kv-cache-quantization-group-size",
        type=int,
        default=64,
        help="Group size for KV cache quantization (default: 64)",
    )
    bench_parser.add_argument(
        "--kv-cache-min-quantize-tokens",
        type=int,
        default=256,
        help="Minimum tokens for quantization to apply (default: 256)",
    )
    # #1103 codex BLOCKING-2: the bench path reads args.hybrid_cache_entries
    # (see the MemoryCacheConfig assembly above) but the flag was only
    # registered on serve_parser, so `rapid-mlx bench --hybrid-cache-entries N`
    # was rejected and the getattr fell back to 0. Register it here too, with
    # the same semantics/default as serve, so bench honors the knob.
    bench_parser.add_argument(
        "--hybrid-cache-entries",
        type=int,
        default=0,
        help=(
            "Retain up to N hybrid (recurrent-state) prefix-cache entries for "
            "exact/prefix-extension reuse; 0 disables (default: 0). Useful for "
            "stable-system-prompt agent workloads on GatedDeltaNet/Mamba models."
        ),
    )
    # --response-cache-entries is intentionally NOT registered on the bench
    # parser. The prompt-deterministic response cache is a chat/serve feature
    # whose lookup/store logic lives only in the chat route; `rapid-mlx bench`
    # never consumes it, so exposing the flag here would advertise a no-op
    # (and wiring bench to the cache would change its measurement semantics).
    # The flag stays serve-only.
    # Paged cache options (experimental)
    bench_parser.add_argument(
        "--use-paged-cache",
        action="store_true",
        help=(
            "Use paged KV cache for memory efficiency (experimental). "
            "Requires a model whose prompt cache is plain full-attention "
            "KV on every layer; startup fails with an actionable error "
            "for rotating/hybrid/recurrent/quantized cache layouts."
        ),
    )
    bench_parser.add_argument(
        "--paged-cache-block-size",
        type=int,
        default=64,
        help="Tokens per cache block (default: 64)",
    )
    bench_parser.add_argument(
        "--max-cache-blocks",
        type=int,
        default=1000,
        help="Maximum number of cache blocks (default: 1000)",
    )
    # Community benchmark submission. Mutually-exclusive with the
    # freeform bench above — when --submit is set the standardized
    # B=1 runner takes over and every other knob is ignored.
    bench_parser.add_argument(
        "--submit",
        action="store_true",
        help=(
            "Removed: exits with an error pointing to `rapid-mlx benchmark "
            "run` + `rapid-mlx benchmark share`, which replace it. Nothing is "
            "run or uploaded."
        ),
    )
    bench_parser.add_argument(
        "--spec-decode",
        type=str,
        default="none",
        choices=["none", "mtp"],
        help=(
            "Speculative-decoding arm for --submit. 'none' (default) is the "
            "baseline. Run the same model twice with a shared --run-group to "
            "put a same-machine A/B on the board."
        ),
    )
    bench_parser.add_argument(
        "--run-group",
        type=str,
        default=None,
        metavar="HEX12",
        help=(
            "12 hex chars linking the arms of one A/B. The board only reports "
            "a speedup for two arms that share this AND ran on one machine; "
            "without it the runs are published as independent rows."
        ),
    )
    bench_parser.add_argument(
        "--sampled",
        action="store_true",
        help=(
            "With --submit, run the bench at temp=0.7/top_p=0.9 instead of "
            "greedy. Stored as a separate 'sampled' bucket — useful for "
            "comparing against Artificial Analysis-style real-world numbers."
        ),
    )
    bench_parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help=(
            "Optional free-text annotation attached to the submission "
            "(e.g. 'on battery', 'fresh boot'). Max 200 chars."
        ),
    )
    bench_parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help=(
            "Path to the Rapid-MLX git checkout. Defaults to the current "
            "working directory. The --submit flow writes the JSON file and "
            "opens the PR from this checkout."
        ),
    )
    # --tier: user-facing tier dispatcher (PR #2). Mutually-exclusive
    # with --submit (PR #3 will consolidate them, but for now the two
    # are independent code paths).
    bench_parser.add_argument(
        "--tier",
        type=str,
        choices=["smoke", "speed", "harness", "all"],
        default=None,
        help=(
            "Run one of the standardized validation tiers: "
            "'smoke' (boot + 1 prompt), "
            "'speed' (B=1 perf probe), "
            "'harness' (5 first-class agent harnesses: "
            "codex/opencode/qwen-code/hermes/aider), "
            "'all' (smoke → speed → harness sequentially, abort on smoke "
            "fail). Boots the model server exactly once per invocation."
        ),
    )
    bench_parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help=(
            "For --tier: attach to an already-running server at this URL "
            "(e.g. http://localhost:8000) instead of booting one. Used by "
            "release_check_m3.sh G7b to reuse the gauntlet's server."
        ),
    )
    bench_parser.add_argument(
        "--long-prompt-tokens",
        type=int,
        default=0,
        help="Approximate reference-context tokens to prepend to each "
        "benchmark prompt. Used with --pflash auto/always for "
        "long-prompt TTFT replication (#287).",
    )
    _add_pflash_args(bench_parser)


def _add_benchmark_parser(
    subparsers: "argparse._SubParsersAction[_PortContextArgumentParser]",
) -> None:
    """Register the ``benchmark`` subcommand."""
    # Local-first, model-first Community Benchmark workspace. Keep the legacy
    # freeform `bench` command intact while this replacement matures. Register
    # this only after every `bench_parser` argument so AST-based docs contract
    # checks continue to attribute the legacy defaults to `bench`.
    community_parser = subparsers.add_parser(
        "benchmark", help="Run or inspect reproducible local benchmarks"
    )
    community_subparsers = community_parser.add_subparsers(
        dest="benchmark_action", required=True
    )
    community_catalog = community_subparsers.add_parser(
        "catalog", help="List models with a registered benchmark protocol"
    )
    community_catalog.add_argument(
        "--memory-gib",
        type=positive_int,
        default=None,
        help="Compute the fit column for a Mac with this much unified memory instead of this one",
    )
    community_catalog.add_argument(
        "--all",
        action="store_true",
        help="List every model with a protocol, not only the recommended ones",
    )
    community_catalog.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of the text summary",
    )
    community_plan = community_subparsers.add_parser(
        "plan", help="Preview the exact local workload for a model"
    )
    community_plan.add_argument(
        "benchmark_model", help="Model alias from `rapid-mlx benchmark catalog`"
    )
    community_plan.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of the text summary",
    )
    community_run = community_subparsers.add_parser(
        "run", help="Run the registered protocol and save the result locally"
    )
    community_run.add_argument(
        "benchmark_model", help="Model alias from `rapid-mlx benchmark catalog`"
    )
    community_run.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of the text summary",
    )
    community_run.add_argument(
        "--progress",
        action="store_true",
        help=argparse.SUPPRESS,  # emit machine-readable progress under --json
    )
    community_run.add_argument(
        "--inherit-process-group",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    community_results = community_subparsers.add_parser(
        "results", help="List locally saved benchmark results"
    )
    community_results.add_argument(
        "--limit", type=positive_int, default=None, help="Return only the latest N runs"
    )
    community_results.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of the text summary",
    )
    community_inspect = community_subparsers.add_parser(
        "inspect", help="Print one locally saved benchmark result"
    )
    community_inspect.add_argument(
        "run_id",
        help="Run id printed by `benchmark run` or listed by `benchmark results`",
    )
    community_inspect.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of the text summary",
    )
    community_share = community_subparsers.add_parser(
        "share", help="Explicitly upload one locally saved benchmark result"
    )
    community_share.add_argument(
        "run_id",
        help="Run id printed by `benchmark run` or listed by `benchmark results`",
    )
    community_share.add_argument(
        "--yes",
        action="store_true",
        help="Confirm upload (for callers that already presented a consent dialog)",
    )
    community_share.add_argument(
        "--preview",
        action="store_true",
        help="Print the exact upload payload without writing or sending it",
    )
    community_share.add_argument(
        "--install-id",
        help=argparse.SUPPRESS,
    )
    community_share.add_argument(
        "--payload-digest",
        help=argparse.SUPPRESS,
    )
    community_share.add_argument(
        "--body-digest",
        help=argparse.SUPPRESS,
    )
    community_share.add_argument(
        "--target",
        help=argparse.SUPPRESS,
    )
    community_share.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of the text summary",
    )


def _add_models_parser(
    subparsers: "argparse._SubParsersAction[_PortContextArgumentParser]",
) -> None:
    """Register the ``models`` subcommand."""
    # Models command. ``ls`` is registered as a top-level alias that
    # defaults to ``models --cached`` (the locally-cached view) — two
    # muscle-memory entry points, one underlying impl.
    models_parser = subparsers.add_parser("models", help="List available model aliases")
    models_parser.add_argument(
        "--cached",
        action="store_true",
        default=False,
        help="Only list models that are downloaded to the local HuggingFace "
        "cache (alias, HF repo, size on disk, last modified).",
    )
    models_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit the model list as machine-readable JSON instead of the "
        "human table (stable keys; pairs with --cached). Prefer this over "
        "scraping the text columns.",
    )
    models_parser.add_argument(
        "--search",
        metavar="TERM",
        default=None,
        help="Case-insensitive substring match against the alias name. "
        "Narrows the 200+-line catalog to rows containing TERM "
        "(e.g. --search qwen picks only qwen aliases). Applies to the "
        "human available-models table; cannot be combined with --json or "
        "--cached.",
    )
    models_parser.add_argument(
        "--modality",
        choices=("text", "video-gen", "image-gen", "audio"),
        default=None,
        help="Show only models of this modality (text, audio, video-gen, "
        "image-gen). Omit the flag to show the full catalog: the text chat "
        "table plus every tagged section. Applies to the human "
        "available-models table; cannot be combined with --json or --cached.",
    )


def _add_recipe_parser(
    subparsers: "argparse._SubParsersAction[_PortContextArgumentParser]",
) -> None:
    """Register the ``recipe`` subcommand."""
    recipe_parser = subparsers.add_parser(
        "recipe", help="Recommend the smart and fast models for this Mac"
    )
    recipe_parser.add_argument(
        "--max-ram",
        type=float,
        default=None,
        metavar="GB",
        help="Use this RAM size instead of auto-detecting the current Mac",
    )
    recipe_parser.add_argument(
        "--json", action="store_true", help="Print the recommendation as JSON"
    )
    subparsers.add_parser(
        "ls",
        help="List models in the local HuggingFace cache (alias for `models --cached`)",
    )

    # Version + help — utility commands that mirror the existing flags but
    # are scriptable as plain subcommands.
    subparsers.add_parser("version", help="Show version number")


def _add_help_parser(
    subparsers: "argparse._SubParsersAction[_PortContextArgumentParser]",
) -> None:
    """Register the ``help`` subcommand."""
    help_parser = subparsers.add_parser("help", help="Show help for a subcommand")
    help_parser.add_argument(
        "subcommand", nargs="?", help="Subcommand to show help for (omit for top-level)"
    )


def _add_pull_parser(
    subparsers: "argparse._SubParsersAction[_PortContextArgumentParser]",
) -> None:
    """Register the ``pull`` subcommand."""
    # Pull / rm / ps — Ollama-style cache and process management.
    pull_parser = subparsers.add_parser(
        "pull", help="Download a model to the HuggingFace cache (no server)"
    )
    pull_parser.add_argument(  # type: ignore[attr-defined]  # argcomplete attribute
        "model", help="Model alias (e.g. qwen3.5-4b-4bit) or HF repo (org/name)"
    ).completer = alias_completer
    # #2145: a multi-variant repo ships every quantization side by side as
    # top-level folders (e.g. LiquidAI/LFM2.5-2.6B-MLX holds 4bit/ 5bit/ 6bit/
    # 8bit/ mxfp4/...). Without selection, `pull <repo>` fetches ALL of them.
    # These flags let a constrained Mac fetch only the variant it can serve.
    # They select the SAME dimension (one variant folder), so --bits and
    # --format are mutually exclusive — passing both would be ambiguous about
    # which single variant the caller wants.
    _variant_group = pull_parser.add_mutually_exclusive_group()
    _variant_group.add_argument(
        "--bits",
        metavar="N",
        help=(
            "Pull only the <N>bit variant of a multi-variant repo "
            "(e.g. --bits 4 fetches only 4bit/; any N the repo ships works)."
        ),
    )
    _variant_group.add_argument(
        "--format",
        metavar="name",
        help=(
            "Pull only the named format variant of a multi-variant repo "
            "(e.g. --format mxfp4, when the repo ships one). GGUF is not "
            "supported: Rapid-MLX cannot run GGUF files."
        ),
    )
    pull_parser.add_argument(
        "--request",
        action="store_true",
        default=False,
        help=(
            "If the pre-download check refuses a public Hugging Face model "
            "(unsupported architecture or GGUF/.bin-only), file a support "
            "request without asking. Sends only the repo id, architecture, "
            "format and Rapid-MLX version."
        ),
    )
    pull_parser.add_argument(
        "--no-preflight",
        action="store_true",
        default=False,
        help=(
            "Skip the pre-download check for models outside the Rapid-MLX "
            "catalog (format and architecture)."
        ),
    )


def _add_import_parser(
    subparsers: "argparse._SubParsersAction[_PortContextArgumentParser]",
) -> None:
    """Register the ``import`` subcommand."""
    import_parser = subparsers.add_parser(
        "import",
        help="Convert a bf16/fp16 safetensors model to quantized MLX (explicit, "
        "cancel-safe)",
    )
    import_parser.add_argument(
        "source", help="Hugging Face repo id (org/name) or local model directory"
    )
    import_parser.add_argument(
        "--quantize",
        type=int,
        choices=[2, 3, 4, 6, 8],
        default=4,
        metavar="BITS",
        help="Quantization bits: 2, 3, 4, 6 or 8 (default: 4).",
    )
    import_parser.add_argument(
        "--name",
        default=None,
        help="Name to serve it by (default: <source-name>-<bits>bit).",
    )
    import_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Replace an existing import of the same name from another source.",
    )


def _add_rm_parser(
    subparsers: "argparse._SubParsersAction[_PortContextArgumentParser]",
) -> None:
    """Register the ``rm`` subcommand."""
    rm_parser = subparsers.add_parser(
        "rm", help="Remove a cached model from the HuggingFace cache"
    )
    rm_parser.add_argument(  # type: ignore[attr-defined]  # argcomplete attribute
        "model", help="Model alias (e.g. qwen3.5-4b-4bit) or HF repo (org/name)"
    ).completer = alias_completer
    rm_parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt and remove the model immediately.",
    )


def _add_alias_parser(
    subparsers: "argparse._SubParsersAction[_PortContextArgumentParser]",
) -> None:
    """Register the ``alias`` subcommand."""
    alias_parser = subparsers.add_parser(
        "alias", help="Manage user-owned model aliases"
    )
    alias_subparsers = alias_parser.add_subparsers(
        dest="alias_action", required=True, help="Alias action"
    )
    alias_set = alias_subparsers.add_parser("set", help="Create or replace an alias")
    alias_set.add_argument("name", help="Private alias name")
    alias_set.add_argument("target", help="Built-in alias or Hugging Face repo id")
    alias_remove = alias_subparsers.add_parser("remove", help="Remove an alias mapping")
    alias_remove.add_argument("name", help="Private alias name")
    alias_subparsers.add_parser("list", help="List user alias mappings")
    subparsers.add_parser("ps", help="List running rapid-mlx servers")


def _add_upgrade_parser(
    subparsers: "argparse._SubParsersAction[_PortContextArgumentParser]",
) -> None:
    """Register the ``upgrade`` subcommand."""
    # Upgrade — detect install method and run the right upgrade command
    # ``update`` is exposed as a subparser alias purely for muscle-memory
    # parity (``npm update`` / ``brew update`` / ``claude update`` /
    # ``rustup update`` all spell it "update"); both names route to
    # ``upgrade_command``. argparse reports the user-typed name on
    # ``args.command``, so the dispatch below matches both.
    upgrade_parser = subparsers.add_parser(
        "upgrade",
        aliases=["update"],
        help="Upgrade rapid-mlx to the latest version (brew / pip / install.sh)",
        description=(
            "Upgrade rapid-mlx to the latest version.\n\n"
            "Note: 'rapid-mlx update' is an alias for 'upgrade'."
        ),
    )
    upgrade_parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt and run the upgrade immediately.",
    )
    upgrade_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the detected install method and the upgrade command, "
            "then exit without running it."
        ),
    )


def _add_chat_parser(
    subparsers: "argparse._SubParsersAction[_PortContextArgumentParser]",
) -> None:
    """Register the ``chat`` subcommand."""
    # Chat — interactive REPL backed by a (spawned or existing) server.
    # ``run`` is exposed as a subparser alias purely for Ollama-muscle-memory
    # parity (``ollama run <model>``). Both names route to ``chat_command``.
    chat_parser = subparsers.add_parser(
        "chat",
        aliases=["run"],
        help="Interactive chat REPL with a model",
        description=(
            "Interactive chat REPL with a model.\n\n"
            "Note: 'rapid-mlx run' is an alias for 'chat'."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # See serve_parser for the rationale: ``--think``/``--no-think`` +
        # ``--thinking``/``--no-thinking`` cross-aliases create ambiguous
        # prefixes that argparse silently resolves to whichever flag was
        # added first.
        allow_abbrev=False,
    )
    chat_parser.add_argument(  # type: ignore[attr-defined]  # argcomplete attribute
        "model",
        nargs="?",
        default=None,
        help="Model alias (e.g. qwen3.5-4b-4bit) or HF repo (org/name). "
        "When omitted, defaults to the qwen3.5-4b-4bit starter — a "
        "dogfood-tested, tool-call-reliable model — downloaded once on first "
        "use. See `rapid_mlx.first_run.select_chat_default`.",
    ).completer = alias_completer
    chat_parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="System prompt prepended to the conversation",
    )
    chat_parser.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable thinking/reasoning mode (default: off in chat REPL — "
        "reasoning models like Qwen3.5 otherwise leak raw chain-of-thought "
        "and can loop until max-tokens). Use --think to surface reasoning, "
        "--no-think is also accepted for back-compat.",
    )
    # Hidden cross-alias for users who picked up the ``--no-thinking`` muscle
    # memory from ``rapid-mlx serve``. ``serve --no-thinking`` and
    # ``chat --no-think`` mean different things internally (server-side
    # parser disable vs. per-request ``enable_thinking=false``), but the
    # flag-name difference trips users. We accept the wrong-side name as
    # an alias for the right-side semantics: ``chat --no-thinking`` simply
    # forwards to the same destination as ``--no-think``.
    chat_parser.add_argument(
        "--no-thinking",
        dest="think",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    chat_parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens per assistant response (default: 2048; raised to "
        "4096 when --think is set so reasoning + answer fit the budget).",
    )
    chat_parser.add_argument(
        "--context-length",
        type=positive_int,
        default=None,
        metavar="TOKENS",
        help="Per-request context window for the server started by chat.",
    )
    chat_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    chat_parser.add_argument(
        "--port",
        type=_port_arg,
        default=None,
        help="Connect to existing server on 127.0.0.1:<port> instead of spawning",
    )
    chat_parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Connect to existing server URL (e.g. http://host:8000) "
        "instead of spawning. Overrides --port.",
    )
    chat_parser.add_argument(
        "--ready-timeout",
        type=int,
        default=600,
        help="Seconds to wait for the spawned server to become ready (default: 600)",
    )
    chat_parser.add_argument(
        "--response-timeout",
        type=int,
        default=600,
        help="Seconds to wait for a single assistant response (default: 600)",
    )
    chat_parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to an MCP config file whose tools are available in this chat",
    )
    chat_parser.add_argument(
        "--mcp-max-rounds",
        type=positive_int,
        default=8,
        help=(
            "Maximum tool-call rounds per turn when --mcp-config is set "
            "(default: 8). Multi-step tasks may need more."
        ),
    )
    chat_parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help=(
            "Disable reusable prefix-cache persistence in the server spawned "
            "by chat, so prompt token IDs are not written to disk. Has no "
            "effect with --port or --base-url; configure that server directly."
        ),
    )


def _add_info_parser(
    subparsers: "argparse._SubParsersAction[_PortContextArgumentParser]",
) -> None:
    """Register the ``info`` subcommand."""
    # Info command — show the per-model profile (parsers + capability gates)
    info_parser = subparsers.add_parser(
        "info",
        help="Show the per-model profile for a model name or alias",
    )
    info_parser.add_argument(  # type: ignore[attr-defined]  # argcomplete attribute
        "model",
        help="Model alias (e.g. qwen3.5-4b-4bit) or HF repo (e.g. mlx-community/SmolLM3-3B-4bit)",
    ).completer = alias_completer


def _add_agents_parser(
    subparsers: "argparse._SubParsersAction[_PortContextArgumentParser]",
) -> None:
    """Register the ``agents`` subcommand."""
    # Agents command
    agents_parser = subparsers.add_parser(
        "agents", help="List, configure, and test agent integrations"
    )
    agents_parser.add_argument(
        "agent_name",
        nargs="?",
        default=None,
        help=(
            "Agent name (e.g. codex, opencode, qwen-code, aider; "
            "continue-dev is accepted for continue). Omit to list all."
        ),
    )
    agents_parser.add_argument(
        "--setup",
        action="store_true",
        help="Auto-configure the agent to point at this server",
    )
    agents_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview setup changes without writing configuration",
    )
    agents_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Apply setup without an interactive confirmation",
    )
    agents_parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip the server health and model check (allow offline setup)",
    )
    agents_parser.add_argument(
        "--test",
        action="store_true",
        help="Run integration tests for this agent",
    )
    agents_parser.add_argument(  # type: ignore[attr-defined]  # argcomplete attribute
        "--model",
        type=str,
        default=None,
        help="Model to use (default: auto-detect from running server)",
    ).completer = alias_completer
    agents_parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="Rapid-MLX server URL (default: http://localhost:8000/v1)",
    )
    agents_parser.add_argument(
        "--agent-version",
        type=str,
        default=None,
        help="Agent version for version-specific config (e.g. 0.8.5)",
    )


def _add_connect_parser(
    subparsers: "argparse._SubParsersAction[_PortContextArgumentParser]",
) -> None:
    """Register the ``connect`` subcommand."""
    # Connect command — the single place to learn "the server is up, now
    # point a tool at it." Renders from the same SSOT as the serve banner
    # (:mod:`rapid_mlx.connect`) so ``ready``/``openai``/``anthropic`` and the
    # ``--json`` machine form can never drift from what the server prints.
    connect_parser = subparsers.add_parser(
        "connect",
        help="Show the server's connection info and wire up a tool",
    )
    connect_parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help=(
            "Tool to set up: claude-code, continue, or openai-python. "
            "Omit to print the connection banner."
        ),
    )
    connect_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the rendered banner",
    )
    connect_parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Server host (default: auto-detect or localhost)",
    )
    connect_parser.add_argument(
        "--port",
        type=_port_arg,
        default=None,
        help="Server port 1-65535 (default: auto-detect or 8000)",
    )
    connect_parser.add_argument(  # type: ignore[attr-defined]  # argcomplete attribute
        "--model",
        type=str,
        default=None,
        help="Model name to advertise (default: auto-detect from server)",
    ).completer = alias_completer
    connect_parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help=(
            "Explicit OpenAI-style base URL of the running server "
            "(e.g. http://localhost:8123/v1) — the banner's pasted commands "
            "carry this so the snippet targets the live host/port, not the "
            "default. Overrides --host/--port only when those are unset."
        ),
    )


def _add_doctor_parser(
    subparsers: "argparse._SubParsersAction[_PortContextArgumentParser]",
) -> None:
    """Register the ``doctor`` subcommand."""
    # Doctor command — pure env-health probe (≤5 s, no model load, no server).
    # Model-validation tiers (smoke/check/full/benchmark) moved to
    # ``rapid-mlx bench --tier ...`` as of v0.7.22.
    #
    # The legacy positional ``tier`` plus ``--model``, ``--models``, and
    # ``--update-baselines`` are intentionally retained (SUPPRESSed from
    # --help) for one release so users hitting the old form
    # ``rapid-mlx doctor check --model qwen3.5-9b-4bit`` get the actionable
    # bench redirect from ``doctor_command`` instead of an argparse
    # ``unrecognized arguments`` wall. Codex review round 1 flagged this:
    # rejecting at argparse-time defeated the redirect. Drop these in a
    # future release once telemetry confirms no one's still calling them.
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check environment health (Python, packages, HF cache, network, ...)",
    )
    doctor_parser.add_argument(
        "tier",
        nargs="?",
        default=None,
        choices=["smoke", "check", "full", "benchmark"],
        help=argparse.SUPPRESS,
    )
    doctor_output = doctor_parser.add_mutually_exclusive_group()
    doctor_output.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print the underlying probe detail for each check",
    )
    doctor_output.add_argument(
        "--json",
        action="store_true",
        help="Emit the versioned machine-readable report as JSON",
    )
    doctor_output.add_argument(
        "--summary",
        action="store_true",
        help="Print only the one-line result summary",
    )
    doctor_parser.add_argument(
        "--deep",
        action="store_true",
        help="Run opt-in dependency, DNS, and route probes (up to 30 seconds)",
    )
    doctor_parser.add_argument(
        "--fix",
        action="store_true",
        help=(
            "Plan and apply only verified repairs (bounded stages may total "
            "up to 90 seconds)"
        ),
    )
    doctor_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --fix, print the repair plan without making changes",
    )
    doctor_parser.add_argument(
        "--yes",
        action="store_true",
        help="With --fix, accept the repair plan non-interactively",
    )
    doctor_section_ids = (
        "system",
        "python",
        "packages.required",
        "updates",
        "packages.optional",
        "cache.huggingface",
        "network",
        "shell",
        "tools.optional",
        "agents",
        "service",
        "deep",
    )
    doctor_parser.add_argument(
        "--only",
        action="append",
        choices=doctor_section_ids,
        metavar="SECTION",
        help="Run only this section ID (repeatable)",
    )
    doctor_parser.add_argument(
        "--skip",
        action="append",
        choices=doctor_section_ids,
        metavar="SECTION",
        help="Skip this section ID (repeatable)",
    )
    # Legacy compatibility shims — accepted-but-ignored so the redirect
    # message in ``doctor_command`` can fire (see comment above).
    doctor_parser.add_argument(
        "--model",
        default=None,
        help=argparse.SUPPRESS,
    )
    doctor_parser.add_argument(
        "--models",
        default=None,
        help=argparse.SUPPRESS,
    )
    doctor_parser.add_argument(
        "--update-baselines",
        action="store_true",
        help=argparse.SUPPRESS,
    )


def _add_telemetry_parser(
    subparsers: "argparse._SubParsersAction[_PortContextArgumentParser]",
) -> None:
    """Register the ``telemetry`` subcommand."""
    # Telemetry subcommand — default-on anonymous usage data.
    # See rapid_mlx/telemetry/ for what we collect / don't collect, and
    # the README "Telemetry" section for the user-facing summary.
    telemetry_parser = subparsers.add_parser(
        "telemetry",
        help="Manage anonymous usage telemetry",
    )
    telemetry_subparsers = telemetry_parser.add_subparsers(
        dest="telemetry_action",
        help="Telemetry actions",
    )
    telemetry_subparsers.add_parser(
        "status", help="Show whether telemetry is enabled and why"
    )
    telemetry_subparsers.add_parser("on", help="Turn anonymous usage telemetry on")
    telemetry_subparsers.add_parser("off", help="Turn anonymous usage telemetry off")
    telemetry_subparsers.add_parser("enable", help="Alias for telemetry on")
    telemetry_subparsers.add_parser("disable", help="Alias for telemetry off")
    telemetry_subparsers.add_parser(
        "preview",
        help="Print a sample payload showing exactly what telemetry would send",
    )
    telemetry_subparsers.add_parser(
        "reset-id",
        help="Rotate the client ID without changing consent",
    )
    telemetry_subparsers.add_parser(
        "reset",
        help="Delete the stored preference and rotate the client ID",
    )


def _add_feedback_parser(
    subparsers: "argparse._SubParsersAction[_PortContextArgumentParser]",
) -> None:
    """Register the ``feedback`` subcommand."""
    # Feedback — the voice channel. Telemetry says what people do; only
    # people say why. Read-only and send-nothing by construction: it
    # prints an invite link and (interactively) opens it.
    feedback_parser = subparsers.add_parser(
        "feedback",
        help="Tell us what you want from Rapid-MLX (opens the community Discord)",
    )
    feedback_parser.add_argument(
        "--no-open",
        action="store_true",
        help="Print the invite link without opening a browser",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the full CLI parser (extracted from ``main`` so tests
    can assert effective flag defaults on the parsed namespace instead
    of scraping source or help text)."""
    _version = _resolve_cli_version()

    parser = _PortContextArgumentParser(
        description=(
            "Rapid-MLX — OpenAI- and Anthropic-compatible LLM server and Mac app "
            "for Apple Silicon, built on MLX, focused on reliable tool calling "
            "for coding agents."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  rapid-mlx chat                                      # interactive REPL (defaults to qwen3.5-4b-4bit)
  rapid-mlx chat qwen3.5-9b-4bit --think                   # larger model, surface reasoning
  rapid-mlx serve qwen3.5-9b-4bit --port 8000              # OpenAI-compatible server
  rapid-mlx serve mlx-community/Qwen3.5-9B-4bit       # full HF repo also works
  rapid-mlx models                                    # list all aliases
  rapid-mlx info qwen3.5-9b-4bit                           # show per-alias profile
""",
    )
    parser.add_argument(
        "--version", "-V", action="version", version=f"rapid-mlx {_version}"
    )
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        help="Disable anonymous usage telemetry for this run "
        "(equivalent to RAPID_MLX_TELEMETRY=0).",
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Do not print the cheetah launch banner. Top-level only "
        "(place it before the subcommand, e.g. 'rapid-mlx --no-banner "
        "serve', like --no-telemetry); equivalent to RAPID_MLX_NO_BANNER=1.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    _add_system_one_parser(subparsers)
    _add_cua_parser(subparsers)
    _add_serve_parser(subparsers)
    _add_bench_parser(subparsers)
    _add_benchmark_parser(subparsers)
    _add_models_parser(subparsers)
    _add_recipe_parser(subparsers)
    _add_help_parser(subparsers)
    _add_pull_parser(subparsers)
    _add_import_parser(subparsers)
    _add_rm_parser(subparsers)
    _add_alias_parser(subparsers)
    _add_upgrade_parser(subparsers)
    _add_chat_parser(subparsers)
    _add_info_parser(subparsers)
    _add_agents_parser(subparsers)

    # Start command — one-command agent startup (#150). Deferred-import so
    # ``rapid_mlx.run`` (and its heavy deps: recommendations, agents, etc.)
    # are only loaded when the verb is actually used.
    from rapid_mlx.run.cli import register as _register_start

    _register_start(subparsers)
    _add_connect_parser(subparsers)
    _add_doctor_parser(subparsers)
    _add_telemetry_parser(subparsers)
    _add_feedback_parser(subparsers)

    # Share subcommand — expose a local serve behind a public rapidmlx.com URL.
    from rapid_mlx.share.cli import register as _register_share

    _register_share(subparsers)

    # Launch subcommand — one-shot bootstrap that patches IDE/agent
    # client configs (Cline, Claude Code, Continue, Cursor) to route
    # at the local rapid-mlx server. See GH issue #566 for motivation.
    # Registered AFTER share so the help-text ordering reads
    # serve→…→share→launch, matching the rough "more common first" flow.
    from rapid_mlx.launch.cli import register as _register_launch

    _register_launch(subparsers)

    # Service subcommand — supported headless macOS service lifecycle
    # (system LaunchDaemon). GH issue #2859. Lives in headless_service to
    # stay distinct from rapid_mlx.service (the engine's helper/post-process
    # layer). Registered after launch so the help ordering keeps the common
    # interactive verbs first.
    from rapid_mlx.headless_service.cli import register as _register_service

    _register_service(subparsers)

    return parser
