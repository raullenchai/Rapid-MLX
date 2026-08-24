import Foundation
import Testing
@testable import Rapid

/// Issue #272: the spawn env handed to the bundled ``rapid-mlx``
/// sidecar is constructed from an explicit allowlist applied to the
/// launcher's ambient env, layered with the desktop's own injected
/// vars. Anything not on the allowlist (``ANTHROPIC_API_KEY``,
/// ``BRAVE_API_KEY``, ``OPENAI_*`` ...) is DROPPED so it can't show
/// up in ``ps eww`` against the sidecar PID or leak into crash-
/// reporter / telemetry snapshots of the child's env.
///
/// These tests pin the allowlist contract on the pure
/// ``ServerManager.serveEnvironmentAdditions`` helper so a future
/// edit that re-adds full-env passthrough fails CI before the leak
/// ships in a DMG.
@MainActor
@Suite("Spawn env allowlist (issue #272)")
struct SpawnEnvAllowlistTests {
    @Test("Desktop disables background prefix-cache restore")
    func desktopDisablesPrefixCacheAutoload() {
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "test-bearer",
            ambient: ["RAPID_MLX_PREFIX_CACHE_AUTOLOAD": "1"]
        )

        #expect(env["RAPID_MLX_PREFIX_CACHE_AUTOLOAD"] == "0")
    }

    // MARK: - PATH augmentation for Dock-launched runs

    /// The PATH a GUI app actually inherits when launched from Finder or
    /// the Dock: launchd's default, with no Homebrew and no ``~/.local/bin``.
    private static let launchdPATH = "/usr/bin:/bin:/usr/sbin:/sbin"

    @Test("Dock-launched PATH gains Homebrew and ~/.local/bin so stdio MCP servers resolve")
    func dockLaunchPATHGainsToolchainDirs() {
        // The reported failure: a `time` connector configured as
        // `uvx --with mcp==1.9.4 mcp-server-time` reports
        // "Command 'uvx' not found in PATH" when the app is opened from the
        // Dock, but works when the app is started from a terminal. The
        // engine resolves the command with `shutil.which` against the
        // sidecar's PATH (vllm_mlx/mcp/security.py), so launchd's minimal
        // PATH means Homebrew's uvx is invisible.
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: ["PATH": Self.launchdPATH, "HOME": "/Users/test"]
        )
        let entries = (env["PATH"] ?? "").split(separator: ":").map(String.init)
        #expect(entries.contains("/opt/homebrew/bin"))
        #expect(entries.contains("/usr/local/bin"))
        #expect(entries.contains("/Users/test/.local/bin"))
        // launchd's own entries survive, still ahead of the fallbacks.
        #expect(env["PATH"]?.hasPrefix(Self.launchdPATH) == true)
    }

    @Test("PATH augmentation does not duplicate directories already present")
    func pathAugmentationDeduplicates() {
        // A terminal-launched run already has Homebrew on PATH. Appending
        // blindly would ship a PATH that grows on every launch path and
        // makes the env harder to read in `ps eww` / bug reports.
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: ["PATH": "/opt/homebrew/bin:/usr/bin", "HOME": "/Users/test"]
        )
        let entries = (env["PATH"] ?? "").split(separator: ":").map(String.init)
        #expect(entries.filter { $0 == "/opt/homebrew/bin" }.count == 1)
        #expect(entries.filter { $0 == "/usr/bin" }.count == 1)
        // The operator's ordering is preserved — Homebrew stays first.
        #expect(entries.first == "/opt/homebrew/bin")
    }

    @Test("A non-absolute HOME contributes no ~/.local/bin entry")
    func nonAbsoluteHomeIsIgnoredForLocalBin() {
        // Defensive: a missing or relative HOME must not inject a relative
        // directory into the child's PATH, where it would resolve against
        // the sidecar's cwd.
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: ["PATH": Self.launchdPATH, "HOME": "relative/path"]
        )
        let entries = (env["PATH"] ?? "").split(separator: ":").map(String.init)
        #expect(entries.allSatisfy { $0.hasPrefix("/") })
        #expect(entries.contains { $0.hasSuffix(".local/bin") } == false)
    }

    @Test("Missing ambient PATH still yields a usable toolchain PATH")
    func absentAmbientPATHStillGetsFallbacks() {
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: ["HOME": "/Users/test"]
        )
        let entries = (env["PATH"] ?? "").split(separator: ":").map(String.init)
        #expect(entries.contains("/opt/homebrew/bin"))
        #expect(entries.contains("/usr/bin"))
        #expect(entries.contains("/bin"))
    }

    // MARK: - allowlist drops third-party secrets

    @Test("Third-party API keys in the launcher's env are dropped")
    func dropsThirdPartySecrets() {
        // The exact env shape observed during v0.7.13 stress testing
        // when the desktop was launched from a Terminal that had
        // `export ANTHROPIC_API_KEY=...` set in the user's shell rc.
        let ambient: [String: String] = [
            "ANTHROPIC_API_KEY": "sk-ant-api03-leak",
            "BRAVE_API_KEY": "BSA-leak",
            "OPENAI_API_KEY": "sk-leak",
            "GOOGLE_API_KEY": "AIza-leak",
            "AWS_SECRET_ACCESS_KEY": "leak",
            "GH_TOKEN": "ghp_leak",
            "GITHUB_TOKEN": "ghp_leak2",
            "NPM_TOKEN": "npm_leak",
            "PATH": "/usr/bin",
        ]
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "real-bearer",
            ambient: ambient
        )
        #expect(env["ANTHROPIC_API_KEY"] == nil)
        #expect(env["BRAVE_API_KEY"] == nil)
        #expect(env["OPENAI_API_KEY"] == nil)
        #expect(env["GOOGLE_API_KEY"] == nil)
        #expect(env["AWS_SECRET_ACCESS_KEY"] == nil)
        #expect(env["GH_TOKEN"] == nil)
        #expect(env["GITHUB_TOKEN"] == nil)
        #expect(env["NPM_TOKEN"] == nil)
        // PATH is on the allowlist, so it passes through — with the
        // toolchain fallbacks appended (see augmentedToolchainPATH).
        #expect(env["PATH"]?.split(separator: ":").first.map(String.init) == "/usr/bin")
        // The legitimate bearer is the canonical one.
        #expect(env["RAPID_MLX_API_KEY"] == "real-bearer")
    }

    // MARK: - allowlist passes through legitimate ambient vars

    @Test("Allowlisted system vars pass through to the child")
    func allowlistedSystemVarsPassThrough() {
        let ambient: [String: String] = [
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": "/Users/test",
            "USER": "test",
            "LOGNAME": "test",
            "LANG": "en_US.UTF-8",
            "LC_ALL": "en_US.UTF-8",
            "LC_CTYPE": "UTF-8",
            "TMPDIR": "/var/folders/x/T/",
            "TZ": "America/Los_Angeles",
        ]
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: ambient
        )
        // Ambient entries keep their order and precedence; the toolchain
        // fallbacks are appended after them.
        #expect(env["PATH"]?.hasPrefix("/usr/local/bin:/usr/bin:/bin") == true)
        #expect(env["HOME"] == "/Users/test")
        #expect(env["USER"] == "test")
        #expect(env["LOGNAME"] == "test")
        #expect(env["LANG"] == "en_US.UTF-8")
        #expect(env["LC_ALL"] == "en_US.UTF-8")
        #expect(env["LC_CTYPE"] == "UTF-8")
        #expect(env["TMPDIR"] == "/var/folders/x/T/")
        #expect(env["TZ"] == "America/Los_Angeles")
    }

    @Test("Python launcher pointers and CA bundle paths pass through")
    func pythonAndTLSAmbientPassThrough() {
        // ``PYTHONHOME`` / ``PYTHONPATH`` are how the desktop tells the
        // bundled Python where its stdlib + site-packages live; the
        // child literally won't start without them. ``SSL_CERT_*`` are
        // how HF Hub finds a working CA bundle for outbound TLS — also
        // on the must-pass-through list.
        let ambient: [String: String] = [
            "PYTHONHOME": "/Applications/Rapid-MLX Desktop.app/Contents/Resources/rapid-mlx/python",
            "PYTHONPATH": "/Applications/.../site-packages",
            "SSL_CERT_FILE": "/etc/ssl/cert.pem",
            "SSL_CERT_DIR": "/etc/ssl/certs",
            "__CFBundleIdentifier": "com.rapidmlx.rapid",
            "XPC_SERVICE_NAME": "application.com.rapidmlx.rapid.123456.78",
        ]
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: ambient
        )
        #expect(env["PYTHONHOME"]?.hasSuffix("/python") == true)
        #expect(env["PYTHONPATH"] == "/Applications/.../site-packages")
        #expect(env["SSL_CERT_FILE"] == "/etc/ssl/cert.pem")
        #expect(env["SSL_CERT_DIR"] == "/etc/ssl/certs")
        #expect(env["__CFBundleIdentifier"] == "com.rapidmlx.rapid")
        #expect(env["XPC_SERVICE_NAME"] == "application.com.rapidmlx.rapid.123456.78")
    }

    // MARK: - desktop-injected overrides win

    @Test("Desktop-injected RAPID_MLX_API_KEY overrides ambient same-named var")
    func desktopBearerOverridesAmbient() {
        // If a malicious / confused ambient already exports
        // ``RAPID_MLX_API_KEY``, the desktop's per-launch bearer must
        // still win — otherwise the child would honor the wrong key
        // and the chat surface would refuse every request signed by
        // the real bearer.
        let ambient: [String: String] = [
            "RAPID_MLX_API_KEY": "stale-or-attacker-controlled",
            "PATH": "/usr/bin",
        ]
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "real-bearer-xyz",
            ambient: ambient
        )
        #expect(env["RAPID_MLX_API_KEY"] == "real-bearer-xyz")
        #expect(env["PATH"]?.hasPrefix("/usr/bin") == true)
    }

    // MARK: - empty bearer ships no sentinel

    @Test("Empty bearer omits RAPID_MLX_API_KEY entirely (no sentinel value)")
    func emptyBearerEmitsNoKey() {
        // Defense in depth: an upstream bug that called the builder
        // with an empty bearer should NOT ship an empty-string
        // ``RAPID_MLX_API_KEY=""`` to the child, where it would be
        // indistinguishable from "the user explicitly disabled auth".
        // Drop the key entirely and let the child's own missing-key
        // handling surface the bug.
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "",
            ambient: ["PATH": "/usr/bin"]
        )
        #expect(env["RAPID_MLX_API_KEY"] == nil)
        #expect(env.keys.contains("RAPID_MLX_API_KEY") == false)
        // The rest of the layered env still ships correctly.
        #expect(env["PYTHONUNBUFFERED"] == "1")
        #expect(env["PATH"]?.hasPrefix("/usr/bin") == true)
    }

    // MARK: - desktop prefix-cache ceiling (issue #1412)

    @Test("Desktop sidecar caps prefix cache at 8 percent of physical RAM")
    func desktopPrefixCacheUsesPhysicalRAMFraction() {
        let sixteenGiB = UInt64(16) << 30
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: ["RAPID_MLX_PREFIX_CACHE_MAX_BYTES": "999999999999"],
            physicalRAMBytes: sixteenGiB,
            availableRAMBytes: sixteenGiB
        )

        let expected = (sixteenGiB / 100) * 8
        #expect(env["RAPID_MLX_PREFIX_CACHE_MAX_BYTES"] == String(expected))
    }

    @Test("Desktop prefix-cache ceiling tops out at 4 GiB")
    func desktopPrefixCacheHasAbsoluteCeiling() {
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: [:],
            physicalRAMBytes: UInt64(256) << 30,
            availableRAMBytes: UInt64(256) << 30
        )

        #expect(
            env["RAPID_MLX_PREFIX_CACHE_MAX_BYTES"]
                == String(UInt64(4) << 30)
        )
    }

    @Test("Memory pressure cannot raise engine's available-RAM default")
    func desktopPrefixCacheClampsToAvailableRAM() {
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: [:],
            physicalRAMBytes: UInt64(32) << 30,
            availableRAMBytes: UInt64(8) << 30
        )

        #expect(
            env["RAPID_MLX_PREFIX_CACHE_MAX_BYTES"]
                == String((UInt64(8) << 30) / 5)
        )
    }

    @Test("Unavailable physical RAM probe preserves engine fallback")
    func unavailablePhysicalRAMProbeKeepsEngineFallback() {
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: ["RAPID_MLX_PREFIX_CACHE_MAX_BYTES": "999999999999"],
            physicalRAMBytes: 0,
            availableRAMBytes: UInt64(8) << 30
        )

        #expect(env["RAPID_MLX_PREFIX_CACHE_MAX_BYTES"] == nil)
    }

    @Test("Unavailable free RAM probe preserves engine fallback")
    func unavailableFreeRAMProbeKeepsEngineFallback() {
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: ["RAPID_MLX_PREFIX_CACHE_MAX_BYTES": "999999999999"],
            physicalRAMBytes: UInt64(32) << 30,
            availableRAMBytes: 0
        )

        #expect(env["RAPID_MLX_PREFIX_CACHE_MAX_BYTES"] == nil)
    }

    // MARK: - empty ambient

    @Test("Empty ambient yields only the desktop-injected layer")
    func emptyAmbientYieldsOnlyInjected() {
        // No allowlisted var present means the result is the desktop-
        // injected layer — bearer + PYTHONUNBUFFERED + HF pinning + the
        // Desktop prefix-cache autoload opt-out — plus
        // ``PATH``, which is deliberately always set (see
        // ``augmentedToolchainPATH``): a sidecar with no PATH at all
        // cannot resolve any stdio MCP server command. Anything BEYOND
        // this set would mean we accidentally hard-coded a system var
        // into the helper, which is what this test exists to catch.
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: [:]
        )
        let expectedKeys: Set<String> = [
            "RAPID_MLX_API_KEY",
            "PYTHONUNBUFFERED",
            "HF_HUB_DISABLE_PROGRESS_BARS",
            "HF_HUB_DISABLE_XET",
            "HF_HUB_DOWNLOAD_TIMEOUT",
            "RAPID_MLX_PREFIX_CACHE_AUTOLOAD",
            "PATH",
        ]
        #expect(Set(env.keys) == expectedKeys)
        // The always-set PATH carries only the fallback toolchain dirs —
        // no HOME means no ``~/.local/bin`` entry.
        #expect(env["PATH"] == "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin")
    }

    // MARK: - allowlist is documented in source as a Set

    @Test("Allowlist is a named Set, not an inline literal hiding in the body")
    func allowlistIsNamedAndStable() {
        // Pin the exact allowlist contents so a future edit that
        // sneaks in a new key (e.g., re-adding ``AWS_PROFILE`` because
        // some downstream lib "needs it") has to update this test and
        // explain why in code review. Allowlist edits are a security-
        // sensitive surface (#272).
        let expected: Set<String> = [
            "PATH", "HOME", "USER", "LOGNAME",
            "LANG", "LC_ALL", "LC_CTYPE",
            "TMPDIR", "TZ",
            "PYTHONHOME", "PYTHONPATH",
            "SSL_CERT_FILE", "SSL_CERT_DIR",
            "__CFBundleIdentifier", "XPC_SERVICE_NAME",
            "HF_HOME", "HF_HUB_CACHE", "XDG_CACHE_HOME",
            "RAPID_MLX_EXTRA_MODEL_ROOTS",
            "HF_ENDPOINT", "HF_HUB_OFFLINE",
            "HF_HUB_DISABLE_TELEMETRY", "HF_HUB_ENABLE_HF_TRANSFER",
        ]
        #expect(ServerManager.serveEnvironmentAllowlist == expected)
    }

    @Test("Drop-if-empty set is exactly the three cache-root keys")
    func dropIfEmptySetIsStable() {
        // Pin the drop-if-empty contract so a future edit that adds a
        // behavior-knob key to the empty-suppression list has to
        // explain why in code review. ``HF_HOME=""`` and friends
        // re-introduce #277 if forwarded verbatim; the behavior knobs
        // (``HF_ENDPOINT``, ``HF_HUB_OFFLINE``, ...) intentionally
        // mirror ``DownloadManager.augmentedEnv``'s verbatim shape.
        let expected: Set<String> = [
            "HF_HOME", "HF_HUB_CACHE", "XDG_CACHE_HOME",
        ]
        #expect(ServerManager.serveEnvironmentDropIfEmpty == expected)
    }

    // MARK: - HF cache-root passthrough (issue #277)

    /// Issue #277 (codex #275 post-merge audit): the launcher's
    /// ``BundledModel.userHFCacheURL`` resolves the HF cache root via
    /// ``HF_HUB_CACHE`` → ``HF_HOME`` + ``/hub`` → ``XDG_CACHE_HOME`` +
    /// ``/huggingface/hub`` → ``HOME`` + ``/.cache/huggingface/hub``,
    /// mirroring ``huggingface_hub.constants``. The launcher's HF byte
    /// monitor at ``installStartupByteMonitor`` watches that directory.
    /// If the child rapid-mlx process doesn't observe the same env
    /// vars, ``huggingface_hub`` inside the child resolves a DIFFERENT
    /// cache root and writes there — the launcher monitors an empty
    /// directory (0% progress) while the model downloads twice across
    /// re-launches. Pin the three path vars on the passthrough side.
    @Test("HF_HOME ambient override passes through to the child")
    func hfHomeAmbientPassesThrough() {
        let ambient: [String: String] = [
            "HF_HOME": "/Volumes/External/hf",
            "PATH": "/usr/bin",
        ]
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: ambient
        )
        #expect(env["HF_HOME"] == "/Volumes/External/hf")
    }

    @Test("HF_HUB_CACHE ambient override passes through to the child")
    func hfHubCacheAmbientPassesThrough() {
        let ambient: [String: String] = [
            "HF_HUB_CACHE": "/explicit/path/to/hub",
            "PATH": "/usr/bin",
        ]
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: ambient
        )
        #expect(env["HF_HUB_CACHE"] == "/explicit/path/to/hub")
    }

    @Test("XDG_CACHE_HOME ambient override passes through to the child")
    func xdgCacheHomeAmbientPassesThrough() {
        // Tier 3 of huggingface_hub's resolution. Less common on macOS
        // than HF_HOME but the launcher's userHFCacheURL honors it, so
        // the child must see it too or the cache splits in the same
        // way HF_HOME would.
        let ambient: [String: String] = [
            "XDG_CACHE_HOME": "/Volumes/External/xdg",
            "PATH": "/usr/bin",
        ]
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: ambient
        )
        #expect(env["XDG_CACHE_HOME"] == "/Volumes/External/xdg")
    }

    @Test("Unset HF cache-root vars stay unset (no forced default)")
    func hfCacheRootVarsNotForcedWhenUnset() {
        // When the user hasn't exported any of HF_HOME / HF_HUB_CACHE /
        // XDG_CACHE_HOME, the desktop must NOT manufacture a value —
        // huggingface_hub's own default (HOME/.cache/huggingface/hub)
        // is the right answer and the launcher's userHFCacheURL falls
        // through to the same default. Pinning an empty value would
        // suppress that fallback in the child.
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: ["PATH": "/usr/bin", "HOME": "/Users/test"]
        )
        #expect(env["HF_HOME"] == nil)
        #expect(env["HF_HUB_CACHE"] == nil)
        #expect(env["XDG_CACHE_HOME"] == nil)
        // The keys aren't present (not just nil values).
        #expect(env.keys.contains("HF_HOME") == false)
        #expect(env.keys.contains("HF_HUB_CACHE") == false)
        #expect(env.keys.contains("XDG_CACHE_HOME") == false)
    }

    @Test("HF auth tokens are still dropped — only cache-root vars cross")
    func hfAuthTokensStillDropped() {
        // The fix for #277 widens the allowlist with PATH config only.
        // HF_TOKEN / HUGGINGFACE_TOKEN / HUGGING_FACE_HUB_TOKEN are
        // SECRETS — they auth HF Hub downloads against the user's
        // account — and stay on the drop list. Otherwise an ambient
        // HF_TOKEN could surface in ``ps eww`` against the sidecar PID.
        let ambient: [String: String] = [
            "HF_HOME": "/Volumes/External/hf",  // passes
            "HF_HUB_CACHE": "/explicit/hub",    // passes
            "HF_TOKEN": "hf_leak",              // drops
            "HUGGINGFACE_TOKEN": "hf_leak",     // drops
            "HUGGING_FACE_HUB_TOKEN": "hf_leak",// drops
        ]
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: ambient
        )
        #expect(env["HF_HOME"] == "/Volumes/External/hf")
        #expect(env["HF_HUB_CACHE"] == "/explicit/hub")
        #expect(env["HF_TOKEN"] == nil)
        #expect(env["HUGGINGFACE_TOKEN"] == nil)
        #expect(env["HUGGING_FACE_HUB_TOKEN"] == nil)
    }

    // MARK: - Codex #279 r1 — empty cache-root values + behavior knobs

    @Test("Empty HF_HOME / HF_HUB_CACHE / XDG_CACHE_HOME are dropped, not forwarded as \"\"")
    func emptyCacheRootValuesAreDropped() {
        // huggingface_hub treats an empty env var the same as "unset"
        // and falls through to the next precedence tier. The launcher
        // does the same in BundledModel.userHFCacheURL via
        // ``!explicit.isEmpty``. If the spawn helper forwarded
        // ``HF_HOME=""`` verbatim, the child's resolver would treat
        // the var as "set" — re-introducing the #277 launcher/child
        // desync.
        let ambient: [String: String] = [
            "HF_HOME": "",
            "HF_HUB_CACHE": "",
            "XDG_CACHE_HOME": "",
            "PATH": "/usr/bin",
        ]
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: ambient
        )
        #expect(env["HF_HOME"] == nil)
        #expect(env["HF_HUB_CACHE"] == nil)
        #expect(env["XDG_CACHE_HOME"] == nil)
        #expect(env.keys.contains("HF_HOME") == false)
        #expect(env.keys.contains("HF_HUB_CACHE") == false)
        #expect(env.keys.contains("XDG_CACHE_HOME") == false)
        // Other allowlisted keys still cross.
        #expect(env["PATH"]?.hasPrefix("/usr/bin") == true)
    }

    @Test("Both HF_HUB_CACHE and HF_HOME set — both cross verbatim, child resolves precedence")
    func bothCacheRootKeysSetCrossVerbatim() {
        // Don't try to be clever about precedence in the launcher;
        // forward what the user set and let huggingface_hub apply its
        // own resolution. The launcher's userHFCacheURL prefers
        // HF_HUB_CACHE over HF_HOME, and so does the child resolver,
        // so forwarding both keeps them in sync.
        let ambient: [String: String] = [
            "HF_HUB_CACHE": "/explicit/hub",
            "HF_HOME": "/somewhere/else",
            "PATH": "/usr/bin",
        ]
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: ambient
        )
        #expect(env["HF_HUB_CACHE"] == "/explicit/hub")
        #expect(env["HF_HOME"] == "/somewhere/else")
    }

    @Test("HF behavior knobs pass through — HF_ENDPOINT / OFFLINE / TELEMETRY / HF_TRANSFER")
    func hfBehaviorKnobsPassThrough() {
        // Codex #279 r1 MAJOR: DownloadManager.augmentedEnv forwards
        // the FULL ambient env to ``rapid-mlx pull``. If the serve-
        // side allowlist drops these knobs, the in-band
        // ``rapid-mlx serve`` cold-download path silently disagrees
        // with the background pull on offline mode / private mirror /
        // privacy / perf — e.g. a user with HF_HUB_OFFLINE=1 sees the
        // pull respect offline mode but a serve cold path try the
        // network and fail opaquely.
        let ambient: [String: String] = [
            "HF_ENDPOINT": "https://hf-mirror.internal.example.com",
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PATH": "/usr/bin",
        ]
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: ambient
        )
        #expect(env["HF_ENDPOINT"] == "https://hf-mirror.internal.example.com")
        #expect(env["HF_HUB_OFFLINE"] == "1")
        #expect(env["HF_HUB_DISABLE_TELEMETRY"] == "1")
        #expect(env["HF_HUB_ENABLE_HF_TRANSFER"] == "1")
    }

    @Test("Unset HF behavior knobs stay unset (no manufactured defaults)")
    func hfBehaviorKnobsNotForcedWhenUnset() {
        // Mirror the cache-root contract: if the user hasn't exported
        // a behavior knob, don't manufacture one. The child's own
        // defaults (online, mirror=upstream, telemetry-on, transfer-
        // off) are the right answer, and pinning empty would suppress
        // huggingface_hub's own envvar-respecting fallback.
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: ["PATH": "/usr/bin"]
        )
        #expect(env.keys.contains("HF_ENDPOINT") == false)
        #expect(env.keys.contains("HF_HUB_OFFLINE") == false)
        #expect(env.keys.contains("HF_HUB_DISABLE_TELEMETRY") == false)
        #expect(env.keys.contains("HF_HUB_ENABLE_HF_TRANSFER") == false)
    }

    @Test("Empty HF behavior knobs are FORWARDED verbatim (only cache-root keys empty-suppress)")
    func emptyBehaviorKnobsForwardedVerbatim() {
        // Codex #279 r2 NIT: pin the explicit empty-passthrough
        // contract for the behavior knobs so a future edit that
        // widens ``serveEnvironmentDropIfEmpty`` to the knobs has to
        // explain why. ``DownloadManager.augmentedEnv`` forwards
        // ambient verbatim — a user who set ``HF_HUB_OFFLINE=""`` on
        // the pull path sees an empty string in the child; the serve
        // path must match that shape so the two cold-download
        // surfaces stay byte-for-byte identical on the same launchctl
        // env.
        let ambient: [String: String] = [
            "HF_ENDPOINT": "",
            "HF_HUB_OFFLINE": "",
            "HF_HUB_DISABLE_TELEMETRY": "",
            "HF_HUB_ENABLE_HF_TRANSFER": "",
        ]
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: ambient
        )
        #expect(env["HF_ENDPOINT"] == "")
        #expect(env["HF_HUB_OFFLINE"] == "")
        #expect(env["HF_HUB_DISABLE_TELEMETRY"] == "")
        #expect(env["HF_HUB_ENABLE_HF_TRANSFER"] == "")
        // The keys are present (not just nil values) — this is the
        // contract that differentiates them from the cache-root keys.
        #expect(env.keys.contains("HF_ENDPOINT"))
        #expect(env.keys.contains("HF_HUB_OFFLINE"))
        #expect(env.keys.contains("HF_HUB_DISABLE_TELEMETRY"))
        #expect(env.keys.contains("HF_HUB_ENABLE_HF_TRANSFER"))
    }

    // MARK: - Models folder override (issue #503)

    @Test("Models folder override injects HF_HUB_CACHE for the engine")
    func modelsFolderOverrideInjectsHubCache() {
        // The desktop "Models folder" preference resolves to an absolute
        // path (validated to exist at the call site); the spawn helper
        // must hand it to the engine as HF_HUB_CACHE so downloads + loads
        // land there.
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: ["PATH": "/usr/bin", "HOME": "/Users/test"],
            modelsFolderOverride: "/Volumes/T7/models"
        )
        #expect(env["HF_HUB_CACHE"] == "/Volumes/T7/models")
    }

    @Test("Models folder override WINS over a stray ambient HF_HUB_CACHE")
    func modelsFolderOverrideBeatsAmbient() {
        // The desktop preference is authoritative: a user who set the
        // folder in Settings must not be silently overridden by an
        // HF_HUB_CACHE exported in the shell that launched the app.
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: [
                "PATH": "/usr/bin",
                "HF_HUB_CACHE": "/some/stale/shell/export",
            ],
            modelsFolderOverride: "/Volumes/T7/models"
        )
        #expect(env["HF_HUB_CACHE"] == "/Volumes/T7/models")
    }

    @Test("No models folder override leaves HF_HUB_CACHE to the ambient/default path")
    func noOverrideLeavesAmbientHubCache() {
        // nil override (no folder set, or the drive is unplugged so the
        // call site resolved to nil) must not manufacture an HF_HUB_CACHE
        // — the ambient value (or the engine's own default) applies, same
        // as before #503.
        let ambientPresent = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: ["PATH": "/usr/bin", "HF_HUB_CACHE": "/ambient/hub"],
            modelsFolderOverride: nil
        )
        #expect(ambientPresent["HF_HUB_CACHE"] == "/ambient/hub")

        let ambientAbsent = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: ["PATH": "/usr/bin"],
            modelsFolderOverride: nil
        )
        #expect(ambientAbsent["HF_HUB_CACHE"] == nil)
        #expect(ambientAbsent.keys.contains("HF_HUB_CACHE") == false)
    }

    @Test("Empty models folder override is ignored (no HF_HUB_CACHE=\"\")")
    func emptyOverrideIgnored() {
        // Defense in depth: an empty string must not ship
        // HF_HUB_CACHE="" which huggingface_hub would treat as "set" and
        // fail to fall through to its default.
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: ["PATH": "/usr/bin"],
            modelsFolderOverride: ""
        )
        #expect(env["HF_HUB_CACHE"] == nil)
        #expect(env.keys.contains("HF_HUB_CACHE") == false)
    }

    // MARK: - regression fence: third-party shape sweep

    @Test("Mixed third-party shape sweep — none of the common leakers survive")
    func commonThirdPartyShapesAllDropped() {
        // Snapshot of the actual ``ps eww`` output the user saw on
        // v0.7.13 plus a sweep of the next obvious shapes a future
        // user could carry into the launch shell. ``RAPID_MLX_*`` is
        // the desktop's own namespace and stays allowlisted via the
        // bearer override path only — there is no general ``RAPID_*``
        // passthrough rule.
        let ambient: [String: String] = [
            // observed in #272
            "ANTHROPIC_API_KEY": "x",
            "BRAVE_API_KEY": "x",
            // next obvious shapes
            "OPENAI_API_KEY": "x",
            "OPENAI_ORG_ID": "x",
            "GOOGLE_API_KEY": "x",
            "GOOGLE_APPLICATION_CREDENTIALS": "/x/path.json",
            "AWS_ACCESS_KEY_ID": "x",
            "AWS_SECRET_ACCESS_KEY": "x",
            "AWS_SESSION_TOKEN": "x",
            "AZURE_OPENAI_API_KEY": "x",
            "DEEPSEEK_API_KEY": "x",
            "MISTRAL_API_KEY": "x",
            "GROQ_API_KEY": "x",
            "REPLICATE_API_TOKEN": "x",
            "HF_TOKEN": "x",                  // the HF hub auth token specifically
            "HUGGINGFACE_TOKEN": "x",
            "HUGGING_FACE_HUB_TOKEN": "x",
            "GH_TOKEN": "x",
            "GITHUB_TOKEN": "x",
            "NPM_TOKEN": "x",
            "DOCKER_PASSWORD": "x",
            "STRIPE_SECRET_KEY": "x",
            "DATABASE_URL": "postgres://leak@host/db",
            "REDIS_URL": "redis://leak@host:6379",
            "SLACK_TOKEN": "xoxb-leak",
            "DISCORD_TOKEN": "x",
            // also drop the legit-sounding home-style stragglers that
            // some users export
            "EDITOR": "vim",
            "SHELL": "/bin/zsh",
            "TERM": "xterm-256color",
            "OLDPWD": "/Users/x/prev",
            "PWD": "/Users/x/cwd",
            "SHLVL": "1",
            "_": "/usr/bin/open",
        ]
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: ambient
        )
        for key in ambient.keys {
            #expect(env[key] == nil, "ambient key \(key) leaked through allowlist")
        }
    }
}
