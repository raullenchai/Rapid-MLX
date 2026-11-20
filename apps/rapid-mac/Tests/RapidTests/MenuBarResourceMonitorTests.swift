import Foundation
import Testing
@testable import Rapid

@MainActor
struct MenuBarResourceMonitorTests {
    private func clients() -> (ServerResidencyClient, MetalMemoryClient) {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [ResourceCardProtocol.self]
        let session = URLSession(configuration: config)
        return (ServerResidencyClient(session: session), MetalMemoryClient(session: session))
    }

    private func server(_ bearer: String = "complete") -> ServerManager {
        ServerManager(testingState: .ready(alias: "chat"), activePort: 7661, activeBearer: bearer)
    }

    @Test func refreshPublishesBothEndpointsAndClearsOnStopOrSessionChange() async {
        let (residency, status) = clients()
        let server = server()
        let monitor = MenuBarResourceMonitor(residencyClient: residency, statusClient: status)
        await monitor.refresh(server: server)
        #expect(monitor.model(for: server).memoryLabel.hasPrefix("4.00 GB / "))
        #expect(monitor.model(for: server).rows.count == 1)
        #expect(monitor.totalBytes == ProcessInfo.processInfo.physicalMemory)
        #expect(monitor.status?.generationTPS == 23)
        server._testReplaceActiveServerSession(bearer: "complete")
        #expect(monitor.model(for: server).rows.isEmpty)
        await monitor.refresh(server: server)
        #expect(monitor.model(for: server).rows.count == 1)
        server._testSetState(.stopped)
        #expect(monitor.model(for: server).rows.isEmpty)
        await monitor.refresh(server: server)
        #expect(monitor.residency == nil && monitor.status == nil)
        server._testSetState(.ready(alias: "chat"))
        server._testReplaceActiveServerSession(bearer: nil)
        await monitor.refresh(server: server)
        #expect(monitor.residency == nil && monitor.status == nil)
    }

    @Test func failedEndpointDoesNotBecomeZeroAndStatusRemainsIndependent() async {
        let (residency, status) = clients()
        for bearer in ["bad-residency", "bad-status", "unauthorized", "malformed", "transport"] {
            let server = server(bearer)
            let monitor = MenuBarResourceMonitor(residencyClient: residency, statusClient: status)
            await monitor.refresh(server: server)
            if bearer == "bad-status" {
                #expect(monitor.residency != nil && monitor.status == nil)
            } else if bearer == "bad-residency" {
                #expect(monitor.residency == nil && monitor.status != nil)
                #expect(monitor.model(for: server).memoryLabel.hasPrefix("— / "))
            } else {
                #expect(monitor.residency == nil && monitor.status == nil)
            }
        }
    }

    @Test func cancelledRefreshPublishesNothing() async {
        let (residency, status) = clients()
        let server = server()
        let monitor = MenuBarResourceMonitor(residencyClient: residency, statusClient: status)
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            await monitor.refresh(server: server)
        }
        await task.value
        #expect(monitor.sampledSession == nil)
        #expect(monitor.residency == nil && monitor.status == nil)
    }

    @Test func inFlightSampleFromPreviousLaunchIsDiscarded() async {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [SuspendedResourceProtocol.self]
        let session = URLSession(configuration: config)
        let monitor = MenuBarResourceMonitor(residencyClient: ServerResidencyClient(session: session),
                                             statusClient: MetalMemoryClient(session: session))
        let server = server()
        let refresh = Task { await monitor.refresh(server: server) }
        await SuspendedResourceProtocol.gate.waitForStart()
        server._testReplaceActiveServerSession(bearer: "complete")
        await SuspendedResourceProtocol.gate.release()
        await refresh.value
        #expect(monitor.sampledSession == nil)
        #expect(monitor.residency == nil && monitor.status == nil)
    }

    @Test func individualUnloadUsesEncodedIDAndEngineDenialsStayVisible() async {
        let (residency, status) = clients()
        #expect(await residency.unload(id: "model #1/part?%", port: 7661, bearer: "complete"))
        #expect(!(await residency.unload(id: "", port: 7661, bearer: "complete")))
        for bearer in ["denied", "unauthorized", "transport"] {
            #expect(!(await residency.unload(id: "model #1/part?%", port: 7661, bearer: bearer)))
        }
        for bearer in ["complete", "denied"] {
            let server = server(bearer)
            server._testSetResidencyClient(residency)
            let monitor = MenuBarResourceMonitor(residencyClient: residency, statusClient: status)
            await monitor.unload(id: "model #1/part?%", server: server)
            #expect((monitor.notice != nil) == (bearer == "denied"))
            #expect(!monitor.unloading)
            #expect(server.state == .ready(alias: "chat"))
        }
    }

    @Test func unloadRechecksBusyPinnedAndOperatingState() async {
        let (residency, status) = clients()
        for bearer in ["busy", "pinned", "bad-residency"] {
            let server = server(bearer)
            let monitor = MenuBarResourceMonitor(residencyClient: residency, statusClient: status)
            // The protocol fails the test if DELETE is ever sent in these states.
            await monitor.unload(id: "model #1/part?%", server: server)
            #expect(monitor.notice == nil)
            #expect(!monitor.unloading)
        }
        let server = server()
        server._testSetOperating(true)
        let monitor = MenuBarResourceMonitor(residencyClient: residency, statusClient: status)
        await monitor.unload(id: "model #1/part?%", server: server)
        #expect(monitor.sampledSession == nil)
    }

    @Test func unloadAllUsesExistingProcessRelease() async {
        let (residency, status) = clients()
        let server = server()
        server._testSetResidencyClient(residency)
        server._testInstallChild(ProcessGroupChild.testStub())
        let monitor = MenuBarResourceMonitor(residencyClient: residency, statusClient: status)
        await monitor.unload(id: nil, server: server)
        #expect(server.state == .stopped)
        #expect(monitor.model(for: server).rows.isEmpty)
        #expect(!monitor.unloading)
    }
}

private actor ResourceResponseGate {
    private var started = false
    private var released = false
    private var starts: [CheckedContinuation<Void, Never>] = []
    private var responses: [CheckedContinuation<Void, Never>] = []

    func enter() async {
        started = true
        starts.forEach { $0.resume() }
        starts.removeAll()
        if !released { await withCheckedContinuation { responses.append($0) } }
    }

    func waitForStart() async {
        if !started { await withCheckedContinuation { starts.append($0) } }
    }

    func release() {
        released = true
        responses.forEach { $0.resume() }
        responses.removeAll()
    }
}

private final class SuspendedResourceProtocol: URLProtocol, @unchecked Sendable {
    // URLProtocol's callbacks are thread-safe, but its SDK Sendable
    // conformance is unavailable. Only this immutable response crosses tasks.
    private struct PendingResponse: @unchecked Sendable {
        let owner: URLProtocol

        func send() {
            let body = owner.request.url?.path == "/v1/status"
                ? #"{"num_running":1,"generation_tps":23}"#
                : #"{"memory_limit_bytes":10,"memory_used_bytes":4,"idle_ttl_seconds":0,"loads_total":0,"evictions_total":0,"models":[]}"#
            let response = HTTPURLResponse(url: owner.request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
            owner.client?.urlProtocol(owner, didReceive: response, cacheStoragePolicy: .notAllowed)
            owner.client?.urlProtocol(owner, didLoad: Data(body.utf8))
            owner.client?.urlProtocolDidFinishLoading(owner)
        }
    }

    static let gate = ResourceResponseGate()
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}
    override func startLoading() {
        let pending = PendingResponse(owner: self)
        Task {
            await SuspendedResourceProtocol.gate.enter()
            pending.send()
        }
    }
}

private final class ResourceCardProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}

    override func startLoading() {
        let bearer = request.value(forHTTPHeaderField: "Authorization") ?? ""
        #expect(request.url?.port == 7661)
        #expect(request.value(forHTTPHeaderField: "Accept") == "application/json")
        if bearer == "Bearer transport" {
            client?.urlProtocol(self, didFailWithError: URLError(.cannotConnectToHost))
            return
        }
        var code = bearer == "Bearer unauthorized" ? 401 : 200
        let body: String
        if request.httpMethod == "DELETE" {
            #expect(!["Bearer busy", "Bearer pinned", "Bearer bad-residency"].contains(bearer))
            #expect(request.url?.absoluteString == "http://127.0.0.1:7661/v1/models/model%20%231%2Fpart%3F%25")
            code = bearer == "Bearer denied" ? 409 : (code == 401 ? 401 : 204)
            body = ""
        } else if bearer == "Bearer malformed" {
            body = "not json"
        } else if request.url?.path == "/v1/status" {
            if bearer == "Bearer bad-status" { code = 503 }
            body = #"{"num_running":0,"generation_tps":23}"#
        } else {
            #expect(request.url?.path == "/v1/models/residency")
            if bearer == "Bearer bad-residency" { code = 503 }
            let active = bearer == "Bearer busy" ? 1 : 0
            let pinned = bearer == "Bearer pinned" ? "true" : "false"
            body = """
            {"memory_limit_bytes":12000000000,"memory_used_bytes":4000000000,
             "idle_ttl_seconds":0,"loads_total":0,"evictions_total":0,
             "models":[{"id":"model #1/part?%","model_path":"org/model","aliases":["model"],
             "modality":"text","state":"resident","pinned":\(pinned),"primary":false,
             "active_requests":\(active),"estimated_bytes":4000000000,"idle_seconds":0}]}
            """
        }
        let response = HTTPURLResponse(url: request.url!, statusCode: code, httpVersion: nil, headerFields: nil)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(body.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }
}
