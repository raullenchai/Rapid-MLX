import Foundation
import XCTest
@testable import Rapid

final class IdleReleaseSelectionTests: XCTestCase {
    @MainActor
    func testAutomaticIdleReleasePreservesSavedChatModel() async throws {
        let defaults = UserDefaults.standard
        let key = IdleModelReleasePreference.storageKey
        let previousValue = defaults.object(forKey: key)
        defaults.set(900, forKey: key)
        defer { defaults.set(previousValue, forKey: key) }

        try await assertRelease(idleSetting: 900, expectedAlias: "chosen-chat-model")
    }

    @MainActor
    func testManualUnloadClearsSavedChatModel() async throws {
        try await assertRelease(idleSetting: nil, expectedAlias: nil)
    }

    @MainActor
    private func assertRelease(idleSetting: Int?, expectedAlias: String?) async throws {
        let suite = "IdleReleaseSelectionTests.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        defaults.set("chosen-chat-model", forKey: SessionModelRestore.chatAliasStorageKey)

        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [IdleReleaseSelectionProtocol.self]
        let session = URLSession(configuration: configuration)
        defer { session.invalidateAndCancel() }
        var client = ServerResidencyClient()
        client.session = session
        let server = ServerManager(
            testingState: .ready(alias: "chosen-chat-model"),
            residency: .empty,
            sessionDefaults: defaults
        )
        server._testSetResidencyClient(client)
        server._testInstallChild(ProcessGroupChild.testStub())

        XCTAssertEqual(ServerManager.lastServedAlias(defaults: defaults), "chosen-chat-model")
        let result = await server.unloadResidentModelsIfIdle(idleSetting: idleSetting)
        XCTAssertEqual(result, .stopped)
        XCTAssertEqual(server.state, .stopped)
        XCTAssertEqual(ServerManager.lastServedAlias(defaults: defaults), expectedAlias)
    }
}

private final class IdleReleaseSelectionProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let payload = Data(#"{"memory_limit_bytes":10,"memory_used_bytes":1,"memory_available_bytes":9,"idle_ttl_seconds":900,"loads_total":1,"evictions_total":0,"models":[{"id":"chosen-chat-model","model_path":"test/chosen-chat-model","aliases":[],"modality":"text","state":"resident","pinned":true,"primary":true,"active_requests":0,"estimated_bytes":1,"measured_bytes":null,"idle_seconds":900}],"audio_lanes":[],"activity":{"in_flight":0,"idle_seconds":900}}"#.utf8)
        let response = HTTPURLResponse(
            url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: nil
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: payload)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}
