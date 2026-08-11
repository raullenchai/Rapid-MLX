import Foundation
import Testing
@testable import Rapid

/// Issue #449 (Persona 3 / Sam dogfood): when the desktop is killed
/// with SIGKILL — force-quit, OOM kill, kernel panic — the bundled
/// ``rapid-mlx`` sidecar is re-parented to launchd (PID 1) and keeps
/// running forever. The sidecar holds 20-30 GB of model weights in
/// unified memory AND the loopback port the next launch wants to
/// bind. Two crashes can stack 60+ GB of phantom RSS the operator
/// never notices, and the next launch hits "port in use" until the
/// orphan is reaped by hand.
///
/// vllm-mlx PR #942 added a sidecar-side watchdog that polls
/// ``os.getppid()`` every 2 s and self-terminates the moment the
/// live PPID stops matching the supervisor's stamp. This test suite
/// pins the desktop half of the protocol: the sidecar spawn MUST
/// stamp ``RAPID_MLX_WATCHDOG_PPID=<launcher PID>`` on the env so
/// the watchdog has the right PID to compare against.
///
/// The watchdog stamp lives in ``serveEnvironmentAdditions`` Layer 2
/// (desktop-injected, always — same tier as ``RAPID_MLX_API_KEY``);
/// the prod call site in ``start(alias:)`` passes
/// ``ProcessInfo.processInfo.processIdentifier``. Both invariants
/// are pinned below.
@MainActor
@Suite("Watchdog spawn env stamp (issue #449)")
struct WatchdogSpawnEnvTests {

    // MARK: - Layer 2 contract

    @Test("Positive supervisorPID stamps RAPID_MLX_WATCHDOG_PPID")
    func stampsWatchdogPpidForPositivePid() {
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: ["PATH": "/usr/bin"],
            supervisorPID: 12345
        )
        #expect(env["RAPID_MLX_WATCHDOG_PPID"] == "12345")
    }

    @Test("Zero / negative / PID-1 sentinel skips the stamp")
    func skipsStampForSentinelPids() {
        // ``supervisorPID <= 1`` mirrors the rapid-mlx side's
        // ``ppid <= 1`` early-out — PID 0 (impossible), -1 (test
        // default), and 1 (launchd, "no real parent to watch") are
        // all treated as "do not stamp" so the watchdog never fires
        // on a degenerate ancestry.
        for sentinel: Int32 in [-1, 0, 1] {
            let env = ServerManager.serveEnvironmentAdditions(
                bearer: "b",
                ambient: ["PATH": "/usr/bin"],
                supervisorPID: sentinel
            )
            #expect(
                env["RAPID_MLX_WATCHDOG_PPID"] == nil,
                "supervisorPID=\(sentinel) must not stamp the watchdog env"
            )
        }
    }

    @Test("Stamp overrides a stale ambient RAPID_MLX_WATCHDOG_PPID")
    func stampOverridesStaleAmbient() {
        // If the launcher was itself started from a shell that
        // exported ``RAPID_MLX_WATCHDOG_PPID=<grandparent_pid>``,
        // an inherited stale value would mis-target the sidecar's
        // watchdog at the wrong PID and the watchdog would fire on
        // the first poll. The allowlist already drops the var from
        // ambient (it's not in ``serveEnvironmentAllowlist``), but
        // pin the override explicitly so a future allowlist edit
        // that accidentally re-admits the key still gets the
        // launcher's own PID, not the stale grandparent's.
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "b",
            ambient: [
                "PATH": "/usr/bin",
                "RAPID_MLX_WATCHDOG_PPID": "99999",
            ],
            supervisorPID: 12345
        )
        #expect(env["RAPID_MLX_WATCHDOG_PPID"] == "12345")
    }

    @Test("Stamp is independent of bearer / HF override layers")
    func stampCoexistsWithOtherLayers() {
        // Layer 1 (allowlist) + Layer 2 (bearer + watchdog) + Layer 3
        // (HF overrides) must all populate without conflict.
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "real-bearer-123",
            ambient: [
                "PATH": "/usr/bin",
                "HOME": "/Users/test",
            ],
            supervisorPID: 7777
        )
        #expect(env["RAPID_MLX_API_KEY"] == "real-bearer-123")
        #expect(env["PATH"]?.hasPrefix("/usr/bin") == true)
        #expect(env["HOME"] == "/Users/test")
        #expect(env["RAPID_MLX_WATCHDOG_PPID"] == "7777")
        #expect(env["PYTHONUNBUFFERED"] == "1")
        #expect(env["HF_HUB_DISABLE_XET"] == "1")
    }

    // MARK: - Prod call site contract (source-pinned)

    @Test("Prod call site in start(alias:) passes the launcher's PID")
    func prodCallSitePassesProcessIdentifier() throws {
        // Source-pin guard. Codex round-1 PR #942 review caught that
        // a unit-test-only argument is easy to forget at the prod
        // call site — the helper accepts ``supervisorPID: -1`` (the
        // legacy / no-stamp shape) by default, so a refactor that
        // drops the explicit argument from ``start(alias:)`` would
        // silently re-introduce the orphan-sidecar bug.
        //
        // Walk the ``start(alias:)`` body once and assert it contains
        // the exact ``supervisorPID: ProcessInfo.processInfo.
        // processIdentifier`` token sequence. The string-grep is
        // crude but cheap; the alternative (running the spawn under
        // a fake Process and inspecting env at the wire level)
        // requires a much deeper test harness that doesn't exist
        // for ServerManager today.
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // RapidTests/
            .deletingLastPathComponent()  // Tests/
            .deletingLastPathComponent()  // repo root
            .appendingPathComponent("Sources/Rapid/Server/ServerManager.swift")
        let source = try String(contentsOf: url, encoding: .utf8)
        // We don't slice the function body (it's >200 LoC and the
        // call site is unique to ServerManager) — grep the whole
        // file. The token includes the named argument so it cannot
        // collide with an unrelated PID read elsewhere.
        let needle =
            "supervisorPID: ProcessInfo.processInfo.processIdentifier"
        #expect(
            source.contains(needle),
            """
            rapid-desktop #449 regression: ServerManager.swift no \
            longer passes the launcher's PID to serveEnvironmentAdditions. \
            Without the stamp, a SIGKILL of the desktop process orphans \
            the bundled rapid-mlx sidecar (30 GB RAM + bound port). \
            Restore ``supervisorPID: ProcessInfo.processInfo.processIdentifier`` \
            on the prod call site.
            """
        )
    }
}
