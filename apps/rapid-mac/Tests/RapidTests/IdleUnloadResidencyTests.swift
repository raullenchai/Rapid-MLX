import Foundation
import Testing
@testable import Rapid

@Suite("Idle unload residency")
struct IdleUnloadResidencyTests {
    @Test func standbyRemainsRoutable() throws {
        let snapshot = try Self.snapshot()
        let model = try #require(snapshot.models.first)
        #expect(model.primary)
        #expect(model.isRoutable)
        #expect(!model.isHoldingMemory)
        #expect(!model.isTransitioning)
        #expect(snapshot.contains("chat"))
        #expect(snapshot.contains("example/chat"))
        #expect(!snapshot.isHoldingMemory(for: "chat"))
        #expect(snapshot.preferredTextAlias(fallback: "speech") == "chat")
        #expect(snapshot.modality(for: "chat") == "text")
        #expect(snapshot.activeRequests(for: "chat") == 0)
        #expect(!snapshot.hasActiveRequests)
        #expect(ModelSwitchRisk.evaluate(currentAlias: "chat", targetAlias: "other", residency: snapshot) == nil)
        #expect(!SidebarView.hasResidentWorkloads(snapshot))
        #expect(SidebarView.residentWorkloadCount(snapshot) == 0)
        #expect(SidebarView.residentMemoryLabel(model) == String(localized: "Unloaded"))
    }

    @Test func lifecycleMemoryStates() throws {
        for state in ["resident", "loading", "unloading", "standby", "evicting", "error"] {
            let snapshot = try Self.snapshot(state: state)
            let model = try #require(snapshot.models.first)
            let holdsMemory = state == "resident" || state == "loading"
            #expect(model.isHoldingMemory == holdsMemory)
            #expect(model.isRoutable == (state != "evicting"))
            #expect(snapshot.contains("chat") == model.isRoutable)
            #expect(SidebarView.residentWorkloadCount(snapshot) == (holdsMemory ? 1 : 0))
            if state == "loading" {
                #expect(SidebarView.residentMemoryLabel(model) == String(localized: "Reloading…"))
            }
        }
    }

    @Test func standbyDoesNotCreditReclaimableMemory() throws {
        #expect(ServerManager.memoryAdmissionForTransition(
            host: MemoryProbe.Snapshot(totalBytes: 32 << 30, usedBytes: 20 << 30),
            residency: try Self.snapshot(),
            plan: .releaseResidentModels
        ) == nil)
    }

    @MainActor
    @Test func standbyIsReadyAndManualEjectStillStops() async throws {
        var client = ServerResidencyClient()
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StandbyResidencyProtocol.self]
        client.session = URLSession(configuration: config)
        let server = ServerManager(testingState: .ready(alias: "chat"), residency: try Self.snapshot())
        server._testSetResidencyClient(client)
        server._testInstallChild(ProcessGroupChild.testStub())
        #expect(server.isModelResident("chat"))
        #expect(server.readinessState(for: "chat") == .ready(alias: "chat"))
        #expect(await server.unloadResidentModelsIfIdle() == .stopped)
        #expect(server.state == .stopped)
    }

    static func snapshot(state: String = "standby") throws -> ModelResidencySnapshot {
        try JSONDecoder().decode(ModelResidencySnapshot.self, from: payload(state: state))
    }

    static func payload(state: String = "standby") -> Data {
        Data("""
        {"memory_limit_bytes":15032385536,"memory_used_bytes":0,"idle_ttl_seconds":300,
         "loads_total":1,"evictions_total":0,"activity":{"in_flight":0,"idle_seconds":301},
         "models":[{"id":"example/chat","model_path":"example/chat","aliases":["chat"],
         "modality":"text","state":"\(state)","pinned":true,"primary":true,"active_requests":0,
         "estimated_bytes":4294967296,"measured_bytes":null,"idle_seconds":301}]}
        """.utf8)
    }
}

private final class StandbyResidencyProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: nil)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: IdleUnloadResidencyTests.payload())
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}
