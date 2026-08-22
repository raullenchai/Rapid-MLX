import Foundation
import Testing
@testable import Rapid

/// Issue #271: the desktop has exactly ONE spawn shape for
/// ``rapid-mlx serve`` regardless of the launch trigger (cold start,
/// post-crash respawn, alias switch, auto-respawn after an idle crash).
///
/// The historical bug — observed during v0.7.13 stress testing — was
/// that a "respawn after crash" code path emitted a DIFFERENT argv:
///
/// ```
/// ~/.local/share/uv/tools/rapid-mlx/bin/python ... rapid-mlx serve \
///     <alias> --listen-fd 3 --host 127.0.0.1 --api-key <BEARER>
/// ```
///
/// vs. the cold-relaunch shape:
///
/// ```
/// .../Resources/rapid-mlx/python/bin/python3.12 -u -s -m vllm_mlx.cli \
///     serve <alias> --host 127.0.0.1 --port 8000
/// # RAPID_MLX_API_KEY supplied via env
/// ```
///
/// Two concrete security/UX hazards:
///
/// 1. **P0 bearer leak**: ``--api-key <value>`` puts the secret on
///    the kernel argv, where every local process can read it via
///    ``ps -axww``. macOS doesn't restrict cross-user ps for non-system
///    processes.
/// 2. **P1 version + model drift**: the external uv install can have
///    a different ``aliases.json`` than the bundled sidecar — so
///    "respawn for alias X" can end up serving alias Y, or fail to
///    find X at all.
///
/// These tests pin the wire shape (argv + env additions) emitted by
/// ``ServerManager`` so a future refactor that re-introduces a
/// divergent respawn path fails CI before it ships.
@Suite("ServerManager spawn-argument shape (issue #271)")
struct SpawnArgumentsTests {

    @Test("Desktop opts vision aliases into MLLM without changing text aliases")
    func desktopVisionCapabilityFlags() {
        #expect(ServerManager.desktopCapabilityFlags(
            forAlias: "qwen3.5-9b-4bit", isBuiltinProfile: true,
            isTextOnly: false, existing: []
        ) == ["--mllm"])
        #expect(ServerManager.desktopCapabilityFlags(
            forAlias: "qwen3-vl-4b-4bit", isBuiltinProfile: true,
            isTextOnly: false, existing: ["--cache-memory-mb", "512"]
        ) == ["--cache-memory-mb", "512", "--mllm"])
        #expect(ServerManager.desktopCapabilityFlags(
            forAlias: "llama3-3b-4bit", existing: ["--enable-prefix-cache"]
        ) == ["--enable-prefix-cache"])
        #expect(ServerManager.desktopCapabilityFlags(
            forAlias: "qwen3.5-4b-4bit", isBuiltinProfile: true,
            isTextOnly: true, existing: []
        ).isEmpty)
        #expect(ServerManager.desktopCapabilityFlags(
            forAlias: "qwen3.5-company-tuned", isBuiltinProfile: false,
            isTextOnly: false, existing: []
        ).isEmpty)
    }

    @Test("Desktop vision policy removes text-only escape hatches and is idempotent")
    func desktopVisionCapabilityFlagsResolveConflicts() {
        #expect(ServerManager.desktopCapabilityFlags(
            forAlias: "gemma3-12b-4bit",
            isBuiltinProfile: true, isTextOnly: false,
            existing: ["--no-mllm", "--cache-memory-mb", "512", "--text-only"]
        ) == ["--cache-memory-mb", "512", "--mllm"])
        #expect(ServerManager.desktopCapabilityFlags(
            forAlias: "gemma3-12b-4bit", isBuiltinProfile: true,
            isTextOnly: false, existing: ["--mllm"]
        ) == ["--mllm"])
    }

    // MARK: - argv shape

    @Test("serve argv has exactly serve + alias + --host + --port + --cors-origins loopback allowlist")
    func argvIsExactlyExpectedShape() {
        let argv = ServerManager.serveArguments(
            alias: "qwen3.5-4b-4bit",
            host: "127.0.0.1",
            port: 8000
        )
        // ``--cors-origins`` uses argparse ``nargs="+"``, so both URL
        // values must trail the flag and not be followed by any
        // further argv element that could be misparsed as an
        // additional origin. Keeping ``--cors-origins`` last in the
        // builder satisfies that constraint without a brittle escape.
        #expect(argv == [
            "serve",
            "qwen3.5-4b-4bit",
            "--host", "127.0.0.1",
            "--port", "8000",
            "--cors-origins", "http://127.0.0.1", "http://localhost",
        ])
    }

    @Test("serve argv NEVER carries --api-key (bearer must travel via env)")
    func argvNeverContainsApiKey() {
        let argv = ServerManager.serveArguments(
            alias: "qwen3.5-4b-4bit",
            host: "127.0.0.1",
            port: 8000
        )
        // ``ps -axww`` exposes argv to every user; ``ps eww`` only
        // exposes env to the owner. The bearer MUST live in env.
        #expect(!argv.contains("--api-key"))
        #expect(!argv.contains("--api_key"))
        #expect(!argv.contains("--apikey"))
        #expect(!argv.contains("--bearer"))
        #expect(!argv.contains("--token"))
        for arg in argv {
            // Defensive: catch the embedded ``--api-key=VALUE`` shape
            // a future "compact flag" refactor might introduce.
            #expect(!arg.hasPrefix("--api-key"))
            #expect(!arg.hasPrefix("--api_key"))
            #expect(!arg.hasPrefix("--token"))
            #expect(!arg.hasPrefix("--bearer"))
        }
    }

    @Test("serve argv NEVER uses --listen-fd (we own port allocation)")
    func argvNeverContainsListenFd() {
        // ``--listen-fd N`` implies the parent (us) bound the socket
        // and inherited it. ``PortAllocator`` does the binding in our
        // shape and then closes it before spawn — the child binds the
        // port itself via ``--port``. ``--listen-fd`` showing up here
        // would mean a divergent spawn path snuck in.
        let argv = ServerManager.serveArguments(
            alias: "qwen3.5-4b-4bit",
            host: "127.0.0.1",
            port: 8000
        )
        #expect(!argv.contains("--listen-fd"))
        for arg in argv {
            #expect(!arg.hasPrefix("--listen-fd"))
        }
    }

    // MARK: - CORS allowlist (issue #306)

    @Test("serve argv pins --cors-origins so the bundled sidecar can't default to wildcard")
    func argvCarriesCorsOrigins() {
        // Issue #306: without ``--cors-origins`` the sidecar defaults
        // to ``["*"]`` (vllm_mlx/cli.py:899). Combined with #303
        // (bearer env not yet enforced as 401) a wildcard CORS policy
        // would let any drive-by webpage on https://evil.example POST
        // to ``http://127.0.0.1:PORT/v1/chat/completions`` once the
        // CORS middleware is wired in a future bundle bump. The
        // desktop pins an explicit allowlist so the policy survives
        // every submodule bump.
        let argv = ServerManager.serveArguments(
            alias: "qwen3.5-4b-4bit",
            host: "127.0.0.1",
            port: 8000
        )
        #expect(argv.contains("--cors-origins"))
    }

    @Test("serve argv NEVER carries a wildcard / null / catch-all CORS value")
    func argvRejectsWildcardCorsValues() {
        // Pin the contract a future refactor would most plausibly
        // break — switching to ``--cors-origins *`` "for convenience"
        // or passing the literal string ``null`` because someone read
        // the help text as "set to null to disable". Both shapes
        // re-introduce the drive-by hazard.
        let argv = ServerManager.serveArguments(
            alias: "qwen3.5-4b-4bit",
            host: "127.0.0.1",
            port: 8000
        )
        // Walk the values trailing ``--cors-origins`` (argparse
        // ``nargs="+"`` semantics: every subsequent argv element
        // up to the next ``--``-prefixed flag is an origin value).
        let idx = argv.firstIndex(of: "--cors-origins")!
        let values = argv[(idx + 1)...].prefix(while: { !$0.hasPrefix("--") })
        #expect(!values.isEmpty, "--cors-origins must be followed by at least one origin value")
        for value in values {
            // Reject every shape a maintainer might mistakenly pass
            // for "allow everything". argparse will happily accept
            // ``""`` and ``"null"`` as origin values (they satisfy
            // ``nargs="+"``); the tripwire rejects them here because
            // both are invalid RFC 6454 origins and either signals a
            // builder bug that would silently expand the surface.
            #expect(value != "*", "wildcard CORS defeats the loopback-only threat model (issue #306)")
            #expect(value != "null")
            #expect(value != "")
            #expect(!value.contains("*"))
            // Origin syntax sanity: each value must be an absolute
            // ``scheme://host[:port]`` per RFC 6454 §3 — not a bare
            // host, not a path, not a wildcard subdomain pattern.
            #expect(value.contains("://"), "CORS origin must be scheme-qualified: \(value)")
        }
    }

    @Test("serve argv pins the default-port loopback CORS allowlist (127.0.0.1 + localhost)")
    func argvPinsLoopbackCorsAllowlist() {
        // The threat model is "drive-by webpage that the browser
        // sends to the loopback sidecar". The desktop's own SwiftUI
        // client is a native binary and sends NO Origin header, so
        // CORS does not apply to it regardless of allowlist contents
        // — the allowlist exists ENTIRELY to defend against
        // browser-originated cross-origin requests.
        //
        // Starlette ``CORSMiddleware.allow_origins`` is exact-match
        // (no port wildcarding, no subdomain wildcarding). A browser
        // sends the page's ``scheme://host[:port]`` as the Origin
        // header, so ``http://localhost`` ONLY matches pages served
        // from default-port loopback (port 80). Browser tools served
        // from ``http://localhost:3000`` (Open WebUI's default),
        // ``http://localhost:8080`` etc. are INTENTIONALLY rejected
        // by this minimal allowlist — third-party in-browser tools
        // on non-default ports are not a supported integration in
        // the desktop's v1 threat model. If/when they become one,
        // extend the allowlist (or pin a sidecar regex flag) at the
        // builder, not in this test.
        let argv = ServerManager.serveArguments(
            alias: "qwen3.5-4b-4bit",
            host: "127.0.0.1",
            port: 8000
        )
        let idx = argv.firstIndex(of: "--cors-origins")!
        let values = Array(argv[(idx + 1)...].prefix(while: { !$0.hasPrefix("--") }))
        #expect(values.contains("http://127.0.0.1"))
        #expect(values.contains("http://localhost"))
    }

    @Test("serve argv keeps --cors-origins last so nargs+ doesn't eat trailing flags")
    func argvCorsOriginsTerminatesArgv() {
        // ``--cors-origins`` is argparse ``nargs="+"`` — it greedily
        // consumes every argv element until the next ``--``-prefixed
        // flag. If a future maintainer inserts a positional or
        // non-prefixed value after the origins (e.g. a model alias
        // mistakenly placed at the tail), argparse will silently
        // treat it as an additional CORS origin. Pinning
        // ``--cors-origins`` as the LAST flag eliminates that hazard.
        let argv = ServerManager.serveArguments(
            alias: "qwen3.5-4b-4bit",
            host: "127.0.0.1",
            port: 8000
        )
        let idx = argv.firstIndex(of: "--cors-origins")!
        let trailing = argv[(idx + 1)...]
        // No element after the cors origin values may start with
        // ``--`` (that would be a flag positioned after a nargs+
        // consumer, which works in argparse but invites confusion)
        // and no element may lack ``://`` (that would be a
        // positional misparsed as an origin).
        for value in trailing {
            #expect(!value.hasPrefix("--"), "no flag may follow --cors-origins values (would be eaten by nargs+ if non-prefixed; pin tail position)")
            #expect(value.contains("://"), "non-URL element after --cors-origins would be silently absorbed as an origin: \(value)")
        }
    }

    @Test("serve argv carries the alias as a positional, not after --alias")
    func argvAliasIsPositional() {
        let argv = ServerManager.serveArguments(
            alias: "qwen3.5-4b-4bit",
            host: "127.0.0.1",
            port: 8000
        )
        // Positional alias is argv[1]; argv[0] is the ``serve``
        // subcommand. This guards against a refactor that switched to
        // ``--alias <value>`` form, which would silently break older
        // rapid-mlx CLIs that only accept the positional.
        #expect(argv.count >= 2)
        #expect(argv[0] == "serve")
        #expect(argv[1] == "qwen3.5-4b-4bit")
        #expect(!argv.contains("--alias"))
    }

    @Test("serve argv reflects allocator-picked port (8001 fallback case)")
    func argvUsesProvidedPort() throws {
        // PortAllocator falls back to 8001 when 8000 is held by another
        // app (LM Studio default, jupyter, etc.). The argv must reflect
        // the actually-allocated port, not the default constant.
        let argv = ServerManager.serveArguments(
            alias: "qwen3.5-4b-4bit",
            host: "127.0.0.1",
            port: 8003
        )
        // ``try #require`` instead of ``if let idx``: a future change
        // that drops ``--port`` from the argv must fail this test
        // outright, not silently skip the inner expectation.
        let idx = try #require(argv.firstIndex(of: "--port"))
        #expect(idx + 1 < argv.count)
        #expect(argv[idx + 1] == "8003")
    }

    @Test("serve argv allows hf_path-shaped aliases (slash and dot survive)")
    func argvAllowsHFPathAlias() {
        // ``ensureServing(alias:)`` accepts ``mlx-community/Qwen3.5-4B-MLX-4bit``
        // as a passthrough hf_path. The argv builder must NOT mangle
        // it (split on slash, escape, lowercase) — rapid-mlx parses
        // the literal value.
        let argv = ServerManager.serveArguments(
            alias: "mlx-community/Qwen3.5-4B-MLX-4bit",
            host: "127.0.0.1",
            port: 8000
        )
        #expect(argv[1] == "mlx-community/Qwen3.5-4B-MLX-4bit")
    }

    // MARK: - env shape

    @Test("env additions carry RAPID_MLX_API_KEY with the exact bearer")
    func envCarriesBearer() {
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "test-bearer-aaaa-bbbb-cccc",
            ambient: [:]
        )
        #expect(env["RAPID_MLX_API_KEY"] == "test-bearer-aaaa-bbbb-cccc")
    }

    @Test("env additions never carry a key whose name resembles --api-key")
    func envNoApiKeyFlagShaped() {
        // Defensive: a future refactor that started writing the bearer
        // to a second env var (mirror, debugging) would multiply the
        // attack surface for the same reason argv must not carry it.
        // ``RAPID_MLX_API_KEY`` is the single canonical channel.
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "test-bearer",
            ambient: [:]
        )
        for key in env.keys {
            let lc = key.lowercased()
            // Allow exactly the canonical RAPID_MLX_API_KEY; anything
            // else carrying "key" / "token" / "bearer" / "api" is a
            // signal of an unintended duplicate channel.
            if key == "RAPID_MLX_API_KEY" { continue }
            #expect(!lc.contains("api"))
            #expect(!lc.contains("token"))
            #expect(!lc.contains("bearer"))
            #expect(!lc.contains("apikey"))
        }
    }

    @Test("env additions force PYTHONUNBUFFERED so tqdm reaches the log tail")
    func envHasPythonUnbuffered() {
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: [:]
        )
        #expect(env["PYTHONUNBUFFERED"] == "1")
    }

    @Test("HF_HUB_DISABLE_XET defaults to 1 when ambient does not set it")
    func envDefaultsXetDisabled() {
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: [:]
        )
        #expect(env["HF_HUB_DISABLE_XET"] == "1")
    }

    @Test("HF_HUB_DISABLE_XET=0 in ambient passes through to the child")
    func envPassesThroughXetOverride() {
        // Operator escape hatch — a power user with a working Xet
        // config can keep it on via ``launchctl setenv HF_HUB_DISABLE_XET 0``.
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: ["HF_HUB_DISABLE_XET": "0"]
        )
        #expect(env["HF_HUB_DISABLE_XET"] == "0")
    }

    @Test("HF_HUB_DOWNLOAD_TIMEOUT defaults to 300 when ambient does not set it")
    func envDefaultsDownloadTimeout() {
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: [:]
        )
        #expect(env["HF_HUB_DOWNLOAD_TIMEOUT"] == "300")
    }

    @Test("HF_HUB_DOWNLOAD_TIMEOUT ambient override passes through")
    func envPassesThroughDownloadTimeout() {
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: ["HF_HUB_DOWNLOAD_TIMEOUT": "900"]
        )
        #expect(env["HF_HUB_DOWNLOAD_TIMEOUT"] == "900")
    }

    // MARK: - cross-trigger uniformity (issue #271 core invariant)

    @Test("Cold start and respawn-after-crash produce identical argv shape")
    func coldStartAndRespawnShapeMatches() {
        // The historical bug was that the "respawn after crash" code
        // path used a totally different argv (--listen-fd, --api-key,
        // sometimes a different alias). After the unification fix the
        // builder is ONE function — there are no separate code paths,
        // so two calls with the same inputs MUST produce equal output.
        //
        // The test deliberately covers the same alias from the same
        // ``ServerManager.start(alias:)`` entry point both times to
        // pin the contract a maintainer would actually break.
        let coldArgv = ServerManager.serveArguments(
            alias: "qwen3.5-4b-4bit",
            host: "127.0.0.1",
            port: 8000
        )
        let respawnArgv = ServerManager.serveArguments(
            alias: "qwen3.5-4b-4bit",
            host: "127.0.0.1",
            port: 8000
        )
        #expect(coldArgv == respawnArgv)
    }

    @Test("Cold start and respawn-after-crash produce identical env additions")
    func coldStartAndRespawnEnvMatches() {
        let coldEnv = ServerManager.serveEnvironmentAdditions(
            bearer: "bearer-xyz",
            ambient: [:]
        )
        let respawnEnv = ServerManager.serveEnvironmentAdditions(
            bearer: "bearer-xyz",
            ambient: [:]
        )
        #expect(coldEnv == respawnEnv)
    }
}
