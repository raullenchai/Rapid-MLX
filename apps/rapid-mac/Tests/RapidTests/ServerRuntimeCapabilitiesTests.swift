import Darwin
import Foundation
import Testing
@testable import Rapid

@Suite("Server runtime capability probing")
struct ServerRuntimeCapabilitiesTests {
    @Test("serve help with resident flags enables both residency arguments")
    func parseCurrentServeHelp() {
        let capabilities = ServerRuntimeCapabilities.parse(serveHelp: """
        usage: rapid-mlx serve [-h] [--resident-memory-limit-gb RESIDENT_MEMORY_LIMIT_GB]
                               [--resident-model-idle-ttl RESIDENT_MODEL_IDLE_TTL]
        """)

        #expect(capabilities.supportsResidentMemoryLimitGB)
        #expect(capabilities.supportsResidentModelIdleTTL)
    }

    @Test("serve help without resident flags disables both residency arguments")
    func parseOldServeHelp() {
        let capabilities = ServerRuntimeCapabilities.parse(serveHelp: """
        usage: rapid-mlx serve [-h] [--served-model-name SERVED_MODEL_NAME]
        """)

        #expect(!capabilities.supportsResidentMemoryLimitGB)
        #expect(!capabilities.supportsResidentModelIdleTTL)
    }

    @Test("resident launch flags are omitted for older runtimes")
    func oldRuntimeDoesNotReceiveResidentFlags() {
        let flags = ServerManager.residentLaunchFlags(
            memoryCeilingGB: 14,
            capabilities: .conservative
        )

        #expect(flags.isEmpty)
    }

    @Test("resident launch flags are emitted for runtimes that advertise them")
    func currentRuntimeReceivesResidentFlags() {
        let flags = ServerManager.residentLaunchFlags(
            memoryCeilingGB: 14,
            capabilities: .allKnown
        )

        #expect(flags == [
            "--resident-memory-limit-gb", "14",
            "--resident-model-idle-ttl", "1800",
        ])
    }

    @Test("probe reads the selected runtime serve help")
    func probeRuntimeHelp() async throws {
        let runtime = try makeRuntimeScript()

        let capabilities = await ServerRuntimeCapabilities.probe(
            binary: runtime,
            timeoutSeconds: 5
        )

        #expect(capabilities == .allKnown)
    }

    @Test("probe drains help output larger than the pipe buffer")
    func probeLargeRuntimeHelp() async throws {
        let runtime = try makeRuntimeScript(helpPaddingLines: 4_096)

        let capabilities = await ServerRuntimeCapabilities.probe(
            binary: runtime,
            timeoutSeconds: 2
        )

        #expect(capabilities == .allKnown)
    }

    @Test("probe ignores help text from a failed runtime command")
    func probeFailedRuntimeHelp() async throws {
        let runtime = try makeRuntimeScript(helpExitStatus: 7)

        let capabilities = await ServerRuntimeCapabilities.probe(
            binary: runtime,
            timeoutSeconds: 2
        )

        #expect(capabilities == .conservative)
    }

    @Test("probe uses the server spawn allowlist instead of ambient secrets")
    func probeSanitizesEnvironment() async throws {
        let environmentCapture = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-runtime-probe-env-\(UUID().uuidString)")
        let runtime = try makeRuntimeScript(environmentCapture: environmentCapture)

        let capabilities = await ServerRuntimeCapabilities.probe(
            binary: runtime,
            timeoutSeconds: 2,
            ambientEnvironment: [
                "ANTHROPIC_API_KEY": "must-not-reach-probe",
                "PATH": "/usr/bin:/bin",
                "HOME": "/Users/test",
            ]
        )

        #expect(capabilities == .allKnown)
        let captured = try String(contentsOf: environmentCapture, encoding: .utf8)
        #expect(captured.contains("secret=unset"))
        #expect(captured.contains("path=/usr/bin:/bin"))
    }

    @Test("timed-out probe terminates its descendant process group")
    func probeTimeoutTerminatesDescendant() async throws {
        let descendantPIDFile = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-runtime-probe-child-\(UUID().uuidString)")
        let runtime = try makeRuntimeScript(
            hangingDescendantPIDFile: descendantPIDFile
        )

        let capabilities = await ServerRuntimeCapabilities.probe(
            binary: runtime,
            timeoutSeconds: 1
        )

        #expect(capabilities == .conservative)
        let pidText = try String(contentsOf: descendantPIDFile, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let descendantPID = try #require(Int32(pidText))
        let deadline = Date().addingTimeInterval(2)
        while processExists(descendantPID), Date() < deadline {
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(!processExists(descendantPID), "the probe helper must not outlive its process group")
    }

    @Test("cancelling a probe terminates its descendant process group")
    func probeCancellationTerminatesDescendant() async throws {
        let descendantPIDFile = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-runtime-probe-cancel-child-\(UUID().uuidString)")
        let runtime = try makeRuntimeScript(
            hangingDescendantPIDFile: descendantPIDFile
        )
        let probe = Task {
            await ServerRuntimeCapabilities.probe(
                binary: runtime,
                timeoutSeconds: 30
            )
        }

        let fileDeadline = Date().addingTimeInterval(2)
        while !FileManager.default.fileExists(atPath: descendantPIDFile.path),
              Date() < fileDeadline {
            try await Task.sleep(for: .milliseconds(20))
        }
        let pidText = try String(contentsOf: descendantPIDFile, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let descendantPID = try #require(Int32(pidText))

        probe.cancel()
        #expect(await probe.value == .conservative)
        let exitDeadline = Date().addingTimeInterval(2)
        while processExists(descendantPID), Date() < exitDeadline {
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(!processExists(descendantPID), "cancel must kill the probe helper before returning")
    }

    @Test("only one start capability probe can own the pre-spawn window")
    @MainActor
    func startProbeReservationRejectsConcurrentCaller() async {
        let gate = RuntimeProbeGate()
        let manager = ServerManager(
            testingState: .idle,
            binaryPath: URL(fileURLWithPath: "/usr/bin/true")
        )
        manager.runtimeCapabilitiesProvider = { _ in
            await gate.wait()
            return .allKnown
        }

        let first = Task { @MainActor in
            await manager._testProbeRuntimeCapabilitiesForStart(
                binary: URL(fileURLWithPath: "/usr/bin/true")
            )
        }
        await gate.waitUntilEntered()
        let second = await manager._testProbeRuntimeCapabilitiesForStart(
            binary: URL(fileURLWithPath: "/usr/bin/true")
        )
        #expect(second == nil)
        #expect(await gate.entryCount == 1)

        await gate.release()
        #expect(await first.value == .allKnown)
    }

    @Test("app shutdown cancels an owned pre-spawn capability probe")
    @MainActor
    func shutdownCancelsStartProbe() async {
        let witness = RuntimeProbeCancellationWitness()
        let manager = ServerManager(
            testingState: .idle,
            binaryPath: URL(fileURLWithPath: "/usr/bin/true")
        )
        manager.runtimeCapabilitiesProvider = { _ in
            await witness.run()
        }
        let probe = Task { @MainActor in
            await manager._testProbeRuntimeCapabilitiesForStart(
                binary: URL(fileURLWithPath: "/usr/bin/true")
            )
        }

        await witness.waitUntilEntered()
        manager.beginShutdown()

        #expect(await probe.value == nil)
        #expect(await witness.wasCancelled)
    }

    @Test("probe bounds a descendant that retains the output pipe")
    func probeBoundsRetainedOutputPipe() async throws {
        let runtime = try makeRuntimeScript(retainedOutputPipe: true)
        let clock = ContinuousClock()

        let elapsed = await clock.measure {
            let capabilities = await ServerRuntimeCapabilities.probe(
                binary: runtime,
                timeoutSeconds: 1
            )

            #expect(capabilities == .conservative)
        }

        #expect(elapsed < .seconds(1.5))
    }

    @Test("probe falls back conservatively when the runtime does not run")
    func probeFailureIsConservative() async {
        let missing = URL(fileURLWithPath: "/tmp/rapid-mlx-missing-\(UUID().uuidString)")

        let capabilities = await ServerRuntimeCapabilities.probe(
            binary: missing,
            timeoutSeconds: 1
        )

        #expect(capabilities == .conservative)
    }

    private func makeRuntimeScript(
        helpPaddingLines: Int = 0,
        helpExitStatus: Int = 0,
        retainedOutputPipe: Bool = false,
        environmentCapture: URL? = nil,
        hangingDescendantPIDFile: URL? = nil
    ) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-runtime-capabilities-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        let script = directory.appendingPathComponent("rapid-mlx")
        let environmentCaptureCommand = environmentCapture.map {
            "printf 'secret=%s\\npath=%s\\n' \"${ANTHROPIC_API_KEY-unset}\" \"${PATH-unset}\" > '\($0.path)'"
        } ?? ":"
        let hangingDescendantCommand = hangingDescendantPIDFile.map {
            "( sleep 30 ) & echo $! > '\($0.path)'; sleep 30"
        } ?? ":"
        try """
        #!/bin/sh
        if [ "$1" = "serve" ] && [ "$2" = "--help" ]; then
          \(environmentCaptureCommand)
          \(hangingDescendantCommand)
          if \(retainedOutputPipe); then
            ( sleep 2 ) &
          fi
          i=0
          while [ "$i" -lt \(helpPaddingLines) ]; do
            echo '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef'
            i=$((i + 1))
          done
          echo '--resident-memory-limit-gb'
          echo '--resident-model-idle-ttl'
          exit \(helpExitStatus)
        fi
        exit 2
        """.write(to: script, atomically: true, encoding: .utf8)
        chmod(script.path, 0o755)
        return script
    }

    private func processExists(_ pid: Int32) -> Bool {
        if kill(pid, 0) == 0 { return true }
        return errno == EPERM
    }
}

private actor RuntimeProbeGate {
    private var entered = 0
    private var released = false
    private var continuation: CheckedContinuation<Void, Never>?

    var entryCount: Int { entered }

    func wait() async {
        entered += 1
        guard !released else { return }
        await withCheckedContinuation { continuation = $0 }
    }

    func waitUntilEntered() async {
        while entered == 0 {
            await Task.yield()
        }
    }

    func release() {
        released = true
        continuation?.resume()
        continuation = nil
    }
}

private actor RuntimeProbeCancellationWitness {
    private var entered = false
    private(set) var wasCancelled = false

    func run() async -> ServerRuntimeCapabilities {
        entered = true
        do {
            try await Task.sleep(for: .seconds(30))
        } catch is CancellationError {
            wasCancelled = true
        } catch {}
        return .conservative
    }

    func waitUntilEntered() async {
        while !entered {
            await Task.yield()
        }
    }
}
