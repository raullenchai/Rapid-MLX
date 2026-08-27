import Foundation
import Testing
@testable import Rapid

/// ``URLProtocol`` stub that stands in for the sidecar's in-process residency
/// endpoint. The lean, deterministic seam for #1838: a rejected resident load
/// must publish the engine's reason to the surface that initiated it, not be
/// swallowed into the log pane.
///
/// The default rejection answer mirrors the exact failure an engine gives when
/// its bundle cannot serve the requested model — e.g. a stock desktop build
/// whose sidecar has no ``[image]`` extra (#1840):
///
///     image generation requires the 'rapid-mlx[image]' Python extra
///     (pip install 'rapid-mlx[image]')
///
/// It returns 422 (a non-2xx, non-404/405 response), so it reaches the
/// ``.rejected(detail)`` branch of ``ServerResidencyClient.load`` rather than
/// the legacy stop/start fallback — the exact path the issue pins.
///
/// Configuration lives on process-wide singleton statics (``URLProtocol`` has
/// no per-session request config channel), so the consuming suite must be
/// ``.serialized``: parallel tests toggling them in opposite directions would
/// race and see each other's value. Each test pins the value it needs before
/// use and restores the defaults in a ``defer`` so a mid-test failure cannot
/// leak configuration into the next test.
final class ResidentLoadRejectProtocol: URLProtocol, @unchecked Sendable {
    /// The engine's own, actionable `detail` string, preserved verbatim.
    static let rejectionDetail =
        "image generation requires the 'rapid-mlx[image]' Python extra "
        + "(pip install 'rapid-mlx[image]')"

    /// When set, only loads whose request body names this alias are rejected
    /// with 422; every other load succeeds. When `nil`, ALL ``POST
    /// /v1/models/load`` requests are controlled by ``rejectLoad`` instead.
    ///
    /// This is what lets a test load model A (rejected) and model B
    /// (succeeding) within one scenario and assert the two outcomes stay
    /// independent — the exact cross-talk regression the single-slot design
    /// had.
    nonisolated(unsafe) static var rejectOnlyAlias: String?

    /// When ``rejectOnlyAlias`` is `nil`, `true` makes every ``POST
    /// /v1/models/load`` answer 422 (rejected) and `false` makes every one
    /// answer 200 (success).
    nonisolated(unsafe) static var rejectLoad = true

    /// Controllable in-order gate for the same-alias concurrency test. When
    /// ``gateAlias`` is set, the FIRST ``/v1/models/load`` whose target
    /// matches it is HELD (its response deferred) rather than answered
    /// immediately, so a second, newer attempt for the same alias can
    /// complete first. The test signals ``releaseGate()`` to deliver the held
    /// first load's response last — recreating the interleaving where an
    /// OLDER ``ensureServing`` attempt returns AFTER a NEWER one, which the
    /// per-alias ``residentLoadFailures`` dictionary alone cannot express.
    ///
    /// The deferred delivery must NOT block the URL loading thread with a
    /// semaphore: URLProtocol ``startLoading`` runs on a shared transport
    /// thread, and blocking it stalls every other request on the same
    /// ``URLSession`` (the second, newer load would fail to start and the
    /// whole test would deadlock). Instead we park the ``URLProtocol``
    /// instance and deliver its response asynchronously on ``releaseGate()``.
    nonisolated(unsafe) static var gateAlias: String?
    nonisolated(unsafe) static var gateHasHeldOne = false
    nonisolated(unsafe) private static var heldProtocol: ResidentLoadRejectProtocol?

    /// Restore the defaults so one test can never observe another test's
    /// leftover configuration — the leak that made the previous global-slot
    /// tests flaky under parallel execution.
    static func reset() {
        rejectLoad = true
        rejectOnlyAlias = nil
        gateAlias = nil
        gateHasHeldOne = false
        heldProtocol = nil
    }

    /// Release the held first gated load (if any), delivering its response
    /// asynchronously after the newer attempt has already completed. The
    /// response uses the current ``rejectLoad`` / ``rejectOnlyAlias`` values,
    /// so the test toggles them just before releasing to choose whether the
    /// OLDER attempt resolves to a rejection or a success.
    static func releaseGate() {
        guard let held = heldProtocol else { return }
        heldProtocol = nil
        let reject = rejectOnlyAlias
        let rejectAll = rejectLoad
        DispatchQueue.global(qos: .userInitiated).async {
            held.finish(resolveWith: rejectAll, rejectOnlyAlias: reject)
        }
    }

    static func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [ResidentLoadRejectProtocol.self] + (config.protocolClasses ?? [])
        return URLSession(configuration: config)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        if request.httpMethod == "POST", request.url?.path == "/v1/models/load" {
            let requestedAlias = Self.alias(from: request)
            // Hold the FIRST load for the gated alias (defer its response)
            // until the test calls ``releaseGate()``, so a second, newer
            // attempt can complete while the first is still in flight —
            // recreating the exact interleaving where an older attempt
            // returns last (#1838 follow-up). The hold parks the protocol
            // instance without blocking the URL loading thread, so the newer
            // attempt's request can still be serviced.
            if let ga = Self.gateAlias, requestedAlias == ga, !Self.gateHasHeldOne {
                Self.gateHasHeldOne = true
                Self.heldProtocol = self
                return
            }
            finish(resolveWith: Self.rejectLoad, rejectOnlyAlias: Self.rejectOnlyAlias)
            return
        }
        if request.httpMethod == "GET", request.url?.path == "/v1/models/residency" {
            // A healthy residency snapshot so a successful load's
            // ``refreshResidency`` has something to read.
            let body = Data(#"""
                {
                  "memory_limit_bytes": 34359738368,
                  "memory_used_bytes": 10737418240,
                  "memory_available_bytes": 23622320128,
                  "idle_ttl_seconds": 1800,
                  "loads_total": 1,
                  "evictions_total": 0,
                  "models": [{
                    "id": "flux2-klein-4b",
                    "model_path": "Runware/FLUX.2-klein-4B",
                    "aliases": ["flux-klein"],
                    "modality": "image-gen",
                    "state": "resident",
                    "pinned": false,
                    "primary": false,
                    "active_requests": 0,
                    "estimated_bytes": 1000,
                    "measured_bytes": 500,
                    "idle_seconds": 0.0
                  }]
                }
                """#.utf8)
            respond(status: 200, body: body)
        } else {
            respond(status: 404, body: Data("{\"error\":\"not_found\"}".utf8))
        }
    }

    /// Deliver this request's answer, computing the body from the given
    /// rejection configuration. Also the delayed path for a held gated load
    /// (dispatched from ``releaseGate`` on a background queue, so it never
    /// blocks the URL loading thread).
    private func finish(
        resolveWith rejectAll: Bool,
        rejectOnlyAlias: String?
    ) {
        let body: Data
        let status: Int
        if let rejectOnlyAlias = rejectOnlyAlias {
            if Self.alias(from: request) == rejectOnlyAlias {
                status = 422
                body = Data("{\"detail\": \"\(Self.rejectionDetail)\"}".utf8)
            } else {
                status = 200
                body = Self.successLoadBody
            }
        } else if rejectAll {
            status = 422
            body = Data("{\"detail\": \"\(Self.rejectionDetail)\"}".utf8)
        } else {
            status = 200
            body = Self.successLoadBody
        }
        respond(status: status, body: body)
    }

    private func respond(status: Int, body: Data) {
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: status,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "application/json"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }

    /// The `model` field of the `POST /v1/models/load` body, so the stub can
    /// reject one specific alias while letting others succeed.
    ///
    /// ``URLSession`` hands a ``URLProtocol`` the request body as an input
    /// stream, coalescing ``URLRequest.httpBody`` into
    /// ``URLRequest.httpBodyStream`` (and niling ``httpBody``), so the alias
    /// must be read from whichever representation carries the bytes — reading
    /// only ``httpBody`` would silently see ``nil`` and never match the gated
    /// alias (the reason the per-alias branch returned 200 instead of 422).
    private static func alias(from request: URLRequest) -> String? {
        guard let body = Self.bodyData(from: request),
              let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any]
        else { return nil }
        return object["model"] as? String
    }

    private static func bodyData(from request: URLRequest) -> Data? {
        if let httpBody = request.httpBody {
            return httpBody
        }
        guard let stream = request.httpBodyStream else { return nil }
        stream.open()
        defer { stream.close() }
        var data = Data()
        let bufferSize = 4096
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferSize)
        defer { buffer.deallocate() }
        while stream.hasBytesAvailable {
            let count = stream.read(buffer, maxLength: bufferSize)
            if count <= 0 { break }
            data.append(buffer, count: count)
        }
        return data
    }

    private static let successLoadBody = Data(#"""
        {
          "id": "flux2-klein-4b",
          "model_path": "Runware/FLUX.2-klein-4B",
          "aliases": ["flux-klein"],
          "modality": "image-gen",
          "state": "resident",
          "pinned": false,
          "primary": false,
          "active_requests": 0,
          "estimated_bytes": 1000,
          "measured_bytes": 500,
          "idle_seconds": 0.0
        }
        """#.utf8)

    override func stopLoading() {}
}

/// ``.serialized`` because ``ResidentLoadRejectProtocol``'s configuration is
/// process-wide singleton state (``URLProtocol`` cannot carry per-request
/// config). Serializing the suite guarantees that tests which toggle the shared
/// slot in opposite directions never race on it, and the ``defer { reset() }``
/// in each test restores defaults so a failure mid-test cannot leak state
/// forward (#1838 follow-up).
///
/// RAM independence (#2480): every server built by ``makeServer`` injects a
/// fixed, generous ``MemoryProbe.Snapshot`` (64 GiB / 0 used) through
/// ``ServerManager.memorySnapshotProvider`` — the same seam
/// ``SessionModelRestoreTests`` uses. On a low-RAM CI runner the pre-#2478
/// tests projected >100% memory and ``ensureServing`` parked forever awaiting
/// a memory-confirmation decision no test responds to. With the injected
/// snapshot every load projects ``.safe``, so ``requiresMemoryConfirmation``
/// can never fire and no responder is needed: the suite asserts the resident
/// rejection/load behaviour on fixtures, not on the host's real RAM.
@MainActor
@Suite("Resident-load rejection feedback", .serialized)
struct ResidentLoadFeedbackTests {
    @Test("Resident admission publishes alias-scoped working state immediately", .timeLimit(.minutes(1)))
    func publishesResidentLoadInFlightState() async {
        defer { ResidentLoadRejectProtocol.reset() }
        let server = makeServer()
        let alias = "flux2-klein-4b"
        ResidentLoadRejectProtocol.gateAlias = alias

        let load = Task { @MainActor in
            await server.ensureServing(alias: alias, hfPath: nil)
        }
        #expect(await pollUntil { ResidentLoadRejectProtocol.gateHasHeldOne })
        #expect(server.isResidentLoadInFlight(alias))

        ResidentLoadRejectProtocol.releaseGate()
        _ = await load.value
        #expect(!server.isResidentLoadInFlight(alias))
    }

    /// The core defect (#1838): the engine returns an actionable reason, the
    /// ``ServerResidencyClient`` maps it to ``.rejected(detail)``, but the
    /// GUI layer previously dropped that result — the failure reached only the
    /// log pane. This pins that ``ServerManager.ensureServing`` now publishes
    /// it so the initiating surface can show it.
    ///
    /// The per-alias ``residentLoadFailure(for:)`` lookup did not exist before
    /// this fix, so this test does not merely fail — it does not compile
    /// against old `main`, guaranteeing it cannot silently rot into a pass.
    @Test("A rejected resident load publishes the engine's reason", .timeLimit(.minutes(1)))
    func publishesRejectedLoadFailure() async {
        defer { ResidentLoadRejectProtocol.reset() }
        ResidentLoadRejectProtocol.rejectLoad = true
        let server = makeServer()

        let ok = await server.ensureServing(
            alias: "flux2-klein-4b",
            hfPath: "Runware/FLUX.2-klein-4B"
        )

        #expect(ok == false)
        #expect(server.residentLoadFailure(for: "flux2-klein-4b")?.alias == "flux2-klein-4b")
        #expect(server.residentLoadFailure(for: "flux2-klein-4b")?.message == ResidentLoadRejectProtocol.rejectionDetail)
    }

    /// A successful in-process load clears any prior rejection, so the banner
    /// does not keep showing last round's failure once the model loads.
    @Test("A successful resident load clears a prior rejection", .timeLimit(.minutes(1)))
    func successfulLoadClearsRejection() async {
        defer { ResidentLoadRejectProtocol.reset() }
        // First a rejection (sets the published failure)…
        ResidentLoadRejectProtocol.rejectLoad = true
        let server = makeServer()

        _ = await server.ensureServing(alias: "flux2-klein-4b", hfPath: nil)
        #expect(server.residentLoadFailure(for: "flux2-klein-4b")?.alias == "flux2-klein-4b")

        // …then a successful load clears it.
        ResidentLoadRejectProtocol.rejectLoad = false
        let ok = await server.ensureServing(alias: "flux2-klein-4b", hfPath: nil)
        #expect(ok == true)
        #expect(server.residentLoadFailure(for: "flux2-klein-4b") == nil)
    }

    @Test("Assistant resident load persists authoritative catalog provenance without a hint", .timeLimit(.minutes(1)))
    func residentAssistantPersistsProbedChatAlias() async throws {
        defer { ResidentLoadRejectProtocol.reset() }
        ResidentLoadRejectProtocol.rejectLoad = false
        let suite = "ResidentLoadFeedbackTests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        defaults.set("previous-chat", forKey: SessionModelRestore.chatAliasStorageKey)
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let fakeServer = packageRoot.appendingPathComponent("scripts/fake-rapid-mlx.sh")
        let wrapper = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-resident-catalog-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: wrapper) }
        try "#!/bin/sh\nFAKE_CACHED_CURATED_TRADEUP=1 exec '\(fakeServer.path)' \"$@\"\n"
            .write(to: wrapper, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755],
            ofItemAtPath: wrapper.path
        )
        let server = makeServer(
            initialAlias: "previous-chat",
            binaryPath: wrapper,
            sessionDefaults: defaults
        )

        let ok = await server.ensureServing(
            alias: "qwen3.5-4b-4bit",
            hfPath: "fake/model",
            estimatedMemoryGB: nil,
            replacementGroup: .assistant
        )

        #expect(ok)
        #expect(defaults.string(forKey: SessionModelRestore.chatAliasStorageKey) == "qwen3.5-4b-4bit")
    }

    /// The cross-talk regression from the earlier single-slot design (#1838
    /// follow-up): a rejection for model B must NOT be cleared by model A
    /// successfully loading, and a rejection for model A must not surface for
    /// model B. Failures are keyed per alias, so concurrent/interleaved loads
    /// of different models each keep their own outcome.
    @Test("Rejections are independent per alias (no cross-talk)", .timeLimit(.minutes(1)))
    func rejectionsArePerAliasIndependent() async {
        defer { ResidentLoadRejectProtocol.reset() }
        // Only model A is rejected; model B and a second attempt at A that
        // follows a different alias succeed. This holds even when A's load is
        // still in flight conceptually, because each alias owns its key.
        ResidentLoadRejectProtocol.rejectOnlyAlias = "flux2-klein-4b"
        let server = makeServer()

        // A is rejected → its failure is recorded.
        let aResult = await server.ensureServing(alias: "flux2-klein-4b", hfPath: nil)
        #expect(aResult == false)
        #expect(server.residentLoadFailure(for: "flux2-klein-4b") != nil)

        // B (a distinct model that takes the same in-process load path)
        // succeeds → its own key stays clear AND it does not clear A.
        let bResult = await server.ensureServing(alias: "llama-3.2-3b", hfPath: nil)
        #expect(bResult == true)
        #expect(server.residentLoadFailure(for: "llama-3.2-3b") == nil)
        #expect(server.residentLoadFailure(for: "flux2-klein-4b") != nil,
                "loading a different, succeeding model must not wipe A's rejection")
    }

    /// The same-alias ordering guarantee the per-alias dictionary cannot
    /// provide by itself. ``ensureServing`` is an ``@MainActor`` async method,
    /// so two attempts for the SAME alias can interleave across the
    /// ``await residencyClient.load`` hop: an attempt that STARTED earlier may
    /// RETURN later. Without the per-alias attempt token, that older
    /// rejection would overwrite the newer success and the UI would show an
    /// expired failure even though the latest load succeeded.
    ///
    /// The stub's in-order gate holds the FIRST load for the alias in flight
    /// (in ``startLoading``, steering clear of the main actor) while a second,
    /// newer attempt completes first — reproducing exactly the interleaving
    /// where latest-attempt-wins must hold.
    @Test("An older rejection cannot clobber a newer success for the same alias", .timeLimit(.minutes(1)))
    func olderRejectionDoesNotClobberNewerSuccess() async throws {
        defer { ResidentLoadRejectProtocol.reset() }
        let server = makeServer()
        let alias = "flux2-klein-4b"

        ResidentLoadRejectProtocol.rejectLoad = true // older attempt, when released, rejects
        ResidentLoadRejectProtocol.gateAlias = alias

        // Attempt 1 (older): mints the first token, clears the failure, and
        // is held by the gate (its response deferred) until releaseGate().
        let attempt1: Task<Bool, Never> = Task { @MainActor in
            await server.ensureServing(alias: alias, hfPath: nil)
        }
        #expect(await pollUntil { ResidentLoadRejectProtocol.gateHasHeldOne },
                "the first load never reached the gate")

        // Attempt 2 (newer) succeeds and takes over the alias's slot.
        ResidentLoadRejectProtocol.rejectLoad = false
        let attempt2Result = await server.ensureServing(alias: alias, hfPath: nil)
        #expect(attempt2Result == true)
        #expect(server.residentLoadFailure(for: alias) == nil,
                "the newer success leaves no rejection in its own slot")

        // Release attempt 1 so it resolves as a REJECTION — but it is no
        // longer the newest attempt, so its stale rejection must be ignored.
        ResidentLoadRejectProtocol.rejectLoad = true
        ResidentLoadRejectProtocol.releaseGate()
        let attempt1Result = await attempt1.value
        #expect(attempt1Result == false)
        #expect(server.residentLoadFailure(for: alias) == nil,
                "an older attempt's rejection must not clobber the newer success")
    }

    /// The mirror image: an older attempt's SUCCESS must not clear a newer
    /// attempt's rejection. The per-alias dictionary allows an old success to
    /// wipe a fresh failure unless the return-time write is also guarded by
    /// which attempt is newest (#1838 follow-up).
    @Test("An older success cannot clear a newer rejection for the same alias", .timeLimit(.minutes(1)))
    func olderSuccessDoesNotClearNewerRejection() async throws {
        defer { ResidentLoadRejectProtocol.reset() }
        let server = makeServer()
        let alias = "flux2-klein-4b"

        ResidentLoadRejectProtocol.rejectLoad = false // older attempt, when released, succeeds
        ResidentLoadRejectProtocol.gateAlias = alias

        let attempt1: Task<Bool, Never> = Task { @MainActor in
            await server.ensureServing(alias: alias, hfPath: nil)
        }
        #expect(await pollUntil { ResidentLoadRejectProtocol.gateHasHeldOne },
                "the first load never reached the gate")

        // Attempt 2 (newer) is rejected and takes over the alias's slot.
        ResidentLoadRejectProtocol.rejectLoad = true
        let attempt2Result = await server.ensureServing(alias: alias, hfPath: nil)
        #expect(attempt2Result == false)
        #expect(server.residentLoadFailure(for: alias) != nil,
                "the newer rejection is recorded")

        // Release attempt 1 so it resolves as SUCCESS — but it is not newest,
        // so its success must not wipe the newer rejection.
        ResidentLoadRejectProtocol.rejectLoad = false
        ResidentLoadRejectProtocol.releaseGate()
        let attempt1Result = await attempt1.value
        #expect(attempt1Result == true)
        #expect(server.residentLoadFailure(for: alias) != nil,
                "an older attempt's success must not clear the newer rejection")
    }

    /// Poll a condition while yielding the main actor, bounded so a stuck
    /// condition fails the test rather than hanging it.
    private func pollUntil(
        _ condition: @escaping () -> Bool,
        timeout: TimeInterval = 3
    ) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return condition()
    }

    /// Build a ``ServerManager`` in the resident-ready state with the stub
    /// transport and a stub child, so ``ensureServing`` takes the in-process
    /// ``/v1/models/load`` path rather than the cold-start fallback.
    ///
    /// Every server also gets a fixed, generous memory snapshot so the suite
    /// is RAM-independent (#2480): on a low-RAM CI runner the host probe can
    /// project >100% utilisation, which parks ``ensureServing`` on a
    /// memory-confirmation decision that no test responder answers — hanging
    /// the whole serial job. Injecting ``safeMemorySnapshot`` (64 GiB / 0 used)
    /// keeps every load ``.safe``, so no confirmation is ever requested and the
    /// tests exercise the resident rejection/load contract on fixtures rather
    /// than on the host's actual memory.
    private func makeServer(
        initialAlias: String = "qwen3.5-4b-4bit",
        binaryPath: URL? = nil,
        sessionDefaults: UserDefaults? = nil
    ) -> ServerManager {
        var client = ServerResidencyClient()
        client.session = ResidentLoadRejectProtocol.session()
        let server = ServerManager(
            testingState: .ready(alias: initialAlias),
            binaryPath: binaryPath,
            sessionDefaults: sessionDefaults
        )
        server._testSetResidencyClient(client)
        server._testInstallChild(ProcessGroupChild.testStub())
        // Capture the Sendable snapshot VALUE on the main actor before the
        // closure: ``memorySnapshotProvider`` is a ``@Sendable`` closure, and
        // referencing the ``@MainActor``-isolated static from inside one is a
        // Swift 6 strict-concurrency error. ``MemoryProbe.Snapshot`` is
        // ``Sendable``, so capturing the value is safe.
        let safeSnapshot = Self.safeMemorySnapshot
        server.memorySnapshotProvider = { safeSnapshot }
        return server
    }

    /// Plenty of headroom above any model footprint in this suite, so
    /// ``ModelSizing.memorySafety`` always returns ``.safe`` and the
    /// `>100%` memory-confirmation path can never be reached. Same shape as
    /// ``SessionModelRestoreTests.safeMemorySnapshot``.
    private static var safeMemorySnapshot: MemoryProbe.Snapshot {
        MemoryProbe.Snapshot(
            totalBytes: 64 * 1_024 * 1_024 * 1_024,
            usedBytes: 0
        )
    }
}
