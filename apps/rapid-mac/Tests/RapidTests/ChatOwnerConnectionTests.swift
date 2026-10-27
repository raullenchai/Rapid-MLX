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
        let connection = try ChatOwnerConnection(runtime: ChatOwnerRuntime(interpreter: URL(fileURLWithPath: "/bin/false")), environment: ["RAPID_MLX_CHAT_HOME": root.path], session: session, launcher: { _, _ in
            try write(run, epoch: epoch, store: store, token: "second")
        }, processStartTime: { _ in 100 })
        _ = try await connection.status()
        ChatOwnerStub.rejected = "Bearer first"
        _ = try await connection.status()
        await connection.close()
        #expect(ChatOwnerStub.headers.contains("Bearer first"))
        #expect(ChatOwnerStub.headers.contains("Bearer second"))
        #expect(ChatOwnerStub.headers.last == "Bearer second")
    }
}
