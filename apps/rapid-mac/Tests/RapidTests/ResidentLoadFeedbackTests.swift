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

    /// Restore the defaults so one test can never observe another test's
    /// leftover configuration — the leak that made the previous global-slot
    /// tests flaky under parallel execution.
    static func reset() {
        rejectLoad = true
        rejectOnlyAlias = nil
    }

    static func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [ResidentLoadRejectProtocol.self] + (config.protocolClasses ?? [])
        return URLSession(configuration: config)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let body: Data
        let status: Int
        if request.httpMethod == "POST", request.url?.path == "/v1/models/load" {
            let requestedAlias = Self.alias(from: request)
            if let rejectOnlyAlias = Self.rejectOnlyAlias {
                if requestedAlias == rejectOnlyAlias {
                    status = 422
                    body = Data("{\"detail\": \"\(Self.rejectionDetail)\"}".utf8)
                } else {
                    status = 200
                    body = Self.successLoadBody
                }
            } else if Self.rejectLoad {
                status = 422
                body = Data("{\"detail\": \"\(Self.rejectionDetail)\"}".utf8)
            } else {
                status = 200
                body = Self.successLoadBody
            }
        } else if request.httpMethod == "GET", request.url?.path == "/v1/models/residency" {
            // A healthy residency snapshot so a successful load's
            // ``refreshResidency`` has something to read.
            status = 200
            body = Data(#"""
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
        } else {
            status = 404
            body = Data("{\"error\":\"not_found\"}".utf8)
        }
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
    private static func alias(from request: URLRequest) -> String? {
        guard let httpBody = request.httpBody,
              let object = try? JSONSerialization.jsonObject(with: httpBody) as? [String: Any]
        else { return nil }
        return object["model"] as? String
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
@MainActor
@Suite("Resident-load rejection feedback", .serialized)
struct ResidentLoadFeedbackTests {
    /// The core defect (#1838): the engine returns an actionable reason, the
    /// ``ServerResidencyClient`` maps it to ``.rejected(detail)``, but the
    /// GUI layer previously dropped that result — the failure reached only the
    /// log pane. This pins that ``ServerManager.ensureServing`` now publishes
    /// it so the initiating surface can show it.
    ///
    /// The per-alias ``residentLoadFailure(for:)`` lookup did not exist before
    /// this fix, so this test does not merely fail — it does not compile
    /// against old `main`, guaranteeing it cannot silently rot into a pass.
    @Test("A rejected resident load publishes the engine's reason")
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
    @Test("A successful resident load clears a prior rejection")
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

    /// The cross-talk regression from the earlier single-slot design (#1838
    /// follow-up): a rejection for model B must NOT be cleared by model A
    /// successfully loading, and a rejection for model A must not surface for
    /// model B. Failures are keyed per alias, so concurrent/interleaved loads
    /// of different models each keep their own outcome.
    @Test("Rejections are independent per alias (no cross-talk)")
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

    /// Build a ``ServerManager`` in the resident-ready state with the stub
    /// transport and a stub child, so ``ensureServing`` takes the in-process
    /// ``/v1/models/load`` path rather than the cold-start fallback.
    private func makeServer() -> ServerManager {
        var client = ServerResidencyClient()
        client.session = ResidentLoadRejectProtocol.session()
        let server = ServerManager(testingState: .ready(alias: "qwen3.5-4b-4bit"))
        server._testSetResidencyClient(client)
        server._testInstallChild(ProcessGroupChild.testStub())
        return server
    }
}
