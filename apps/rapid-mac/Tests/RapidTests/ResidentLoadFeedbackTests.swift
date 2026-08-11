import Foundation
import Testing
@testable import Rapid

/// ``URLProtocol`` stub that stands in for the sidecar's in-process residency
/// endpoint. The lean, deterministic seam for #1838: a rejected resident load
/// must publish the engine's reason to the surface that initiated it, not be
/// swallowed into the log pane.
///
/// The default answer mirrors the exact failure an engine gives when its
/// bundle cannot serve the requested model — e.g. a stock desktop build whose
/// sidecar has no ``[image]`` extra (#1840):
///
///     image generation requires the 'rapid-mlx[image]' Python extra
///     (pip install 'rapid-mlx[image]')
///
/// It returns 422 (a non-2xx, non-404/405 response), so it reaches the
/// ``.rejected(detail)`` branch of ``ServerResidencyClient.load`` rather than
/// the legacy stop/start fallback — the exact path the issue pins.
final class ResidentLoadRejectProtocol: URLProtocol, @unchecked Sendable {
    /// The engine's own, actionable `detail` string, preserved verbatim.
    static let rejectionDetail =
        "image generation requires the 'rapid-mlx[image]' Python extra "
        + "(pip install 'rapid-mlx[image]')"

    nonisolated(unsafe) static var rejectLoad = true
    nonisolated(unsafe) static var method = "POST"
    nonisolated(unsafe) static var path = ""

    static func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [ResidentLoadRejectProtocol.self] + (config.protocolClasses ?? [])
        return URLSession(configuration: config)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.method = request.httpMethod ?? "GET"
        Self.path = request.url?.path ?? ""
        let body: Data
        let status: Int
        if Self.method == "POST", Self.path == "/v1/models/load", Self.rejectLoad {
            status = 422
            body = Data("{\"detail\": \"\(Self.rejectionDetail)\"}".utf8)
        } else if Self.method == "GET", Self.path == "/v1/models/residency" {
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
        } else if Self.method == "POST", Self.path == "/v1/models/load" {
            // Success branch for the clear-on-success assertion.
            status = 200
            body = Data(#"""
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

    override func stopLoading() {}
}

@MainActor
@Suite("Resident-load rejection feedback")
struct ResidentLoadFeedbackTests {
    /// The core defect (#1838): the engine returns an actionable reason, the
    /// ``ServerResidencyClient`` maps it to ``.rejected(detail)``, but the
    /// GUI layer previously dropped that result — the failure reached only the
    /// log pane. This pins that ``ServerManager.ensureServing`` now publishes
    /// it so the initiating surface can show it.
    ///
    /// The `lastResidentLoadFailure` property did not exist before this fix, so
    /// this test does not merely fail — it does not compile against old `main`,
    /// guaranteeing it cannot silently rot into a pass.
    @Test("A rejected resident load publishes the engine's reason")
    func publishesRejectedLoadFailure() async {
        ResidentLoadRejectProtocol.rejectLoad = true
        var client = ServerResidencyClient()
        client.session = ResidentLoadRejectProtocol.session()
        let server = ServerManager(testingState: .ready(alias: "qwen3.5-4b-4bit"))
        server._testSetResidencyClient(client)
        server._testInstallChild(ProcessGroupChild.testStub())

        let ok = await server.ensureServing(
            alias: "flux2-klein-4b",
            hfPath: "Runware/FLUX.2-klein-4B"
        )

        #expect(ok == false)
        #expect(server.lastResidentLoadFailure?.alias == "flux2-klein-4b")
        #expect(server.lastResidentLoadFailure?.message == ResidentLoadRejectProtocol.rejectionDetail)
    }

    /// A successful in-process load clears any prior rejection, so the banner
    /// does not keep showing last round's failure once the model loads.
    @Test("A successful resident load clears a prior rejection")
    func successfulLoadClearsRejection() async {
        // First a rejection (sets the published failure)…
        ResidentLoadRejectProtocol.rejectLoad = true
        var client = ServerResidencyClient()
        client.session = ResidentLoadRejectProtocol.session()
        let server = ServerManager(testingState: .ready(alias: "qwen3.5-4b-4bit"))
        server._testSetResidencyClient(client)
        server._testInstallChild(ProcessGroupChild.testStub())

        _ = await server.ensureServing(alias: "flux2-klein-4b", hfPath: nil)
        #expect(server.lastResidentLoadFailure?.alias == "flux2-klein-4b")

        // …then a successful load clears it.
        ResidentLoadRejectProtocol.rejectLoad = false
        let ok = await server.ensureServing(alias: "flux2-klein-4b", hfPath: nil)
        #expect(ok == true)
        #expect(server.lastResidentLoadFailure == nil)
    }
}
