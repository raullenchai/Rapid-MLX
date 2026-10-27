import Foundation
import Testing
@testable import Rapid

private final class ChatOwnerStub: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var epoch = ""
    nonisolated(unsafe) static var store = ""
    nonisolated(unsafe) static var headers: [String] = []
    nonisolated(unsafe) static var rejected = ""
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        Self.headers.append(request.value(forHTTPHeaderField: "Authorization") ?? "")
        if request.value(forHTTPHeaderField: "Authorization") == Self.rejected {
            client?.urlProtocol(self, didReceive: HTTPURLResponse(url: request.url!, statusCode: 401, httpVersion: nil, headerFields: nil)!, cacheStoragePolicy: .notAllowed)
            client?.urlProtocolDidFinishLoading(self)
            return
        }
        let body: String
        if request.url!.path.hasSuffix("capabilities") {
            body = "{\"wire_version\":1,\"schema_version\":4,\"owner_epoch\":\"\(Self.epoch)\",\"store_uuid\":\"\(Self.store)\"}"
        } else if request.httpMethod == "POST" {
            var data = request.httpBody ?? Data()
            if data.isEmpty, let stream = request.httpBodyStream {
                stream.open()
                defer { stream.close() }
                var buffer = [UInt8](repeating: 0, count: 4096)
                while stream.hasBytesAvailable {
                    let count = stream.read(&buffer, maxLength: buffer.count)
                    if count <= 0 { break }
                    data.append(buffer, count: count)
                }
            }
            let json = try! JSONSerialization.jsonObject(with: data) as! [String: String]
            body = "{\"client_id\":\"\(json["client_id"]!)\",\"lease_seconds\":45}"
        } else { body = "{}" }
        client?.urlProtocol(self, didReceive: HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(body.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

private final class LaunchCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0
    func increment() { lock.withLock { value += 1 } }
    var count: Int { lock.withLock { value } }
}

@Suite(.serialized) struct ChatOwnerConnectionTests {
    private func fixture() throws -> (URL, URL, String, String) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let run = root.appendingPathComponent("default/run")
        try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        return (root, run, UUID().uuidString.lowercased(), UUID().uuidString.lowercased())
    }
    private func write(_ run: URL, epoch: String, store: String, token: String, changes: [String: Any] = [:]) throws {
        var record: [String: Any] = ["format_version": 1, "wire_version": 1, "schema_version": 4, "owner_epoch": epoch, "store_uuid": store, "pid": 123, "process_start_time": 100.0, "endpoint": "http://127.0.0.1:4321", "credential_file": "bearer-\(epoch)"]
        record.merge(changes) { _, new in new }
        let file = run.appendingPathComponent("discovery.json")
        try JSONSerialization.data(withJSONObject: record).write(to: file)
        try Data(token.utf8).write(to: run.appendingPathComponent("bearer-\(epoch)"))
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: file.path)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: run.appendingPathComponent("bearer-\(epoch)").path)
    }
    @Test func parsingAndStaleIdentity() throws {
        let (root, run, epoch, store) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        try write(run, epoch: epoch, store: store, token: "secret")
        let profile = root.appendingPathComponent("default")
        let result = try ChatOwnerDiscovery.read(profileDirectory: profile, processStartTime: { _ in 100 })
        let found = try #require(result)
        #expect(found.status.ownerEpoch.uuidString.lowercased() == epoch)
        #expect(try ChatOwnerDiscovery.read(profileDirectory: profile, processStartTime: { _ in 101 }) == nil)
        try write(run, epoch: epoch, store: store, token: "secret", changes: ["endpoint": "http://example.com:4321"])
        #expect(throws: ChatOwnerConnectionError.self) { try ChatOwnerDiscovery.read(profileDirectory: profile, processStartTime: { _ in 100 }) }
        try write(run, epoch: epoch, store: store, token: "secret", changes: ["schema_version": 999])
        #expect(throws: ChatOwnerConnectionError.incompatibleOwner) { try ChatOwnerDiscovery.read(profileDirectory: profile, processStartTime: { _ in 100 }) }
    }
    @Test func launcherAndRotatedBearer() async throws {
        let (root, run, epoch, store) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        ChatOwnerStub.epoch = epoch; ChatOwnerStub.store = store; ChatOwnerStub.headers = []; ChatOwnerStub.rejected = ""
        try write(run, epoch: epoch, store: store, token: "first")
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [ChatOwnerStub.self]
        let session = URLSession(configuration: config)
        let counter = LaunchCounter()
        let connection = try ChatOwnerConnection(runtime: ChatOwnerRuntime(interpreter: URL(fileURLWithPath: "/bin/false")), environment: ["RAPID_MLX_CHAT_HOME": root.path], session: session, launcher: { _, _ in
            counter.increment()
            try write(run, epoch: epoch, store: store, token: "second")
        }, processStartTime: { _ in 100 })
        _ = try await connection.status()
        ChatOwnerStub.rejected = "Bearer first"
        _ = try await connection.status()
        await connection.close()
        #expect(ChatOwnerStub.headers.contains("Bearer first"))
        #expect(ChatOwnerStub.headers.contains("Bearer second"))
        #expect(ChatOwnerStub.headers.last == "Bearer second")
        #expect(counter.count == 1)
        #expect(ChatOwnerStub.headers.filter { $0 == "Bearer first" }.count == 3)
    }
    @Test func invalidDiscoveryAndPermissions() throws {
        let (root, run, epoch, store) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let profile = root.appendingPathComponent("default")
        let file = run.appendingPathComponent("discovery.json")
        try write(run, epoch: epoch, store: store, token: "secret")
        try Data("{".utf8).write(to: file)
        #expect(throws: ChatOwnerConnectionError.invalidDiscovery("json")) {
            try ChatOwnerDiscovery.read(profileDirectory: profile, processStartTime: { _ in 100 })
        }
        try Data(repeating: 65, count: ChatOwnerDiscovery.maximumFileBytes + 1).write(to: file)
        #expect(throws: ChatOwnerConnectionError.self) {
            try ChatOwnerDiscovery.read(profileDirectory: profile, processStartTime: { _ in 100 })
        }
        for field in ["wire_version", "schema_version"] {
            try write(run, epoch: epoch, store: store, token: "secret", changes: [field: 999])
            #expect(throws: ChatOwnerConnectionError.incompatibleOwner) {
                try ChatOwnerDiscovery.read(profileDirectory: profile, processStartTime: { _ in 100 })
            }
        }
        try write(run, epoch: epoch, store: store, token: "secret", changes: ["format_version": 999])
        #expect(throws: ChatOwnerConnectionError.invalidDiscovery("format")) {
            try ChatOwnerDiscovery.read(profileDirectory: profile, processStartTime: { _ in 100 })
        }
        try write(run, epoch: epoch, store: store, token: "secret")
        #expect(try ChatOwnerDiscovery.read(profileDirectory: profile, processStartTime: { _ in nil }) == nil)
        #expect(try ChatOwnerDiscovery.read(profileDirectory: profile, processStartTime: { _ in 101 }) == nil)
        for url in [run, file, run.appendingPathComponent("bearer-\(epoch)")] {
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
            #expect(throws: ChatOwnerConnectionError.self) {
                try ChatOwnerDiscovery.read(profileDirectory: profile, processStartTime: { _ in 100 })
            }
            try FileManager.default.setAttributes([.posixPermissions: url == run ? 0o700 : 0o600], ofItemAtPath: url.path)
        }
    }

    @Test func changedEpochAndBearer() async throws {
        let (root, run, epoch, store) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        ChatOwnerStub.epoch = epoch; ChatOwnerStub.store = store; ChatOwnerStub.headers = []; ChatOwnerStub.rejected = ""
        try write(run, epoch: epoch, store: store, token: "first")
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [ChatOwnerStub.self]
        let connection = try ChatOwnerConnection(runtime: ChatOwnerRuntime(interpreter: URL(fileURLWithPath: "/bin/false")), environment: ["RAPID_MLX_CHAT_HOME": root.path], session: URLSession(configuration: config), launcher: { _, _ in Issue.record("unexpected launch") }, processStartTime: { _ in 100 })
        #expect(try await connection.status().ownerEpoch.uuidString.lowercased() == epoch)
        try write(run, epoch: epoch, store: store, token: "rotated")
        _ = try await connection.status()
        let next = UUID().uuidString.lowercased()
        try write(run, epoch: next, store: store, token: "new-epoch")
        ChatOwnerStub.epoch = next
        #expect(try await connection.status().ownerEpoch.uuidString.lowercased() == next)
        await connection.close()
        #expect(ChatOwnerStub.headers.contains("Bearer rotated"))
        #expect(ChatOwnerStub.headers.contains("Bearer new-epoch"))
    }

    @Test func launcherFailure() async throws {
        let (root, _, _, _) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let connection = try ChatOwnerConnection(runtime: ChatOwnerRuntime(interpreter: URL(fileURLWithPath: "/bin/false")), environment: ["RAPID_MLX_CHAT_HOME": root.path], launcher: { _, _ in throw ChatOwnerConnectionError.ownerUnavailable }, processStartTime: { _ in nil })
        await #expect(throws: ChatOwnerConnectionError.ownerUnavailable) { try await connection.status() }
    }

}
