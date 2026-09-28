import Foundation
import Testing
@testable import Rapid

// MARK: - Fakes

private final class MockAgentAPI: CUAAPI, @unchecked Sendable {
    enum Failure: Error { case requested }

    var plannersResult: [CUAPlannerOption] = []
    var createdRequests: [CUARunRequest] = []
    var scriptedEvents: [CUAEvent] = []
    var finalSummary = "opened the article"
    var approveCalls = 0
    var cancelCalls = 0
    var runStatus = "running"
    var approveShouldFail = false
    var cancelShouldFail = false
    var eventsShouldFail = false

    var addedPlanners: [CUAPlannerCreateRequest] = []
    var deletedPlannerNames: [String] = []
    var addShouldFail = false

    func planners() async throws -> [CUAPlannerOption] {
        plannersResult
    }

    func addPlanner(_ request: CUAPlannerCreateRequest) async throws {
        if addShouldFail { throw Failure.requested }
        addedPlanners.append(request)
    }

    func deletePlanner(name: String) async throws {
        deletedPlannerNames.append(name)
    }

    func create(_ request: CUARunRequest) async throws -> String {
        createdRequests.append(request)
        return "run123"
    }

    func events(runID: String, after: Int) async throws -> CUARunView {
        if eventsShouldFail { throw Failure.requested }
        var events: [CUAEvent] = []
        if after == 0 {
            events.append(CUAEvent(seq: 1, kind: "started", step: nil, action: nil, stepInstruction: nil, outcome: nil, targetLabel: nil, status: nil, finalSummary: nil, reason: nil))
            events.append(contentsOf: scriptedEvents.enumerated().map { index, event in
                CUAEvent(seq: index + 2, kind: event.kind, step: event.step, action: event.action, stepInstruction: event.stepInstruction, outcome: event.outcome, targetLabel: event.targetLabel, status: event.status, finalSummary: event.finalSummary, reason: event.reason, error: event.error)
            })
        }
        return CUARunView(
            runID: runID,
            app: "Google Chrome",
            goal: "g",
            status: runStatus,
            finalSummary: finalSummary,
            error: "",
            planner: "local-9b [local]",
            eventsAfterSeq: after,
            events: events,
            runDir: "/tmp/runs/x"
        )
    }

    func approve(runID: String) async throws {
        approveCalls += 1
        if approveShouldFail { throw Failure.requested }
    }

    func cancel(runID: String) async throws {
        cancelCalls += 1
        if cancelShouldFail { throw Failure.requested }
    }
}

private func drain() async {
    await Task.yield()
    try? await Task.sleep(nanoseconds: 50_000_000)
    await Task.yield()
}

// MARK: - View model

@Suite("Agent Task Panel")
@MainActor
struct CUAViewModelTests {
    private func makeEvent(
        seq: Int, kind: String, step: Int? = nil, action: String? = nil,
        instruction: String? = nil, outcome: String? = nil,
        status: String? = nil, summary: String? = nil, error: String? = nil
    ) -> CUAEvent {
        CUAEvent(
            seq: seq, kind: kind, step: step, action: action,
            stepInstruction: instruction, outcome: outcome,
            targetLabel: nil, status: status, finalSummary: summary, reason: nil,
            error: error
        )
    }

    @Test("Start creates a run and finishes with the server summary")
    func startToFinished() async throws {
        let api = MockAgentAPI()
        api.scriptedEvents = [
            makeEvent(seq: 2, kind: "plan", step: 1, instruction: "click the search box"),
            makeEvent(seq: 3, kind: "executed", step: 1, outcome: "success"),
            makeEvent(seq: 4, kind: "terminal", status: "completed", summary: "opened Apple Silicon"),
        ]
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        viewModel.goal = "open Apple Silicon"
        await viewModel.start()
        await drain()
        #expect(viewModel.phase == .finished(summary: "opened Apple Silicon"))
        #expect(api.createdRequests.count == 1)
        #expect(api.createdRequests[0].goal == "open Apple Silicon")
        #expect(api.createdRequests[0].humanLogin)
        #expect(viewModel.events.count == 4)
    }

    @Test("Gate event flips to awaitingApproval and approve() hits the API once")
    func gateApproval() async throws {
        let api = MockAgentAPI()
        api.scriptedEvents = [
            makeEvent(seq: 2, kind: "plan", step: 2, instruction: "fill email"),
        ]
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        viewModel.goal = "check flights"
        await viewModel.start()
        api.scriptedEvents = [
            makeEvent(seq: 3, kind: "gate"),
        ]
        await drain()
        #expect(viewModel.phase == .awaitingApproval)
        await viewModel.approve()
        #expect(api.approveCalls == 1)
    }

    @Test("Approval failure keeps the active run controllable")
    func gateApprovalFailure() async throws {
        let api = MockAgentAPI()
        api.scriptedEvents = [makeEvent(seq: 2, kind: "gate")]
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        viewModel.goal = "check flights"
        await viewModel.start()
        await drain()
        #expect(viewModel.phase == .awaitingApproval)

        api.approveShouldFail = true
        await viewModel.approve()

        #expect(viewModel.phase == .awaitingApproval)
        #expect(viewModel.actionError?.hasPrefix("Approval failed:") == true)
        #expect(viewModel.phase.isBusy)
    }

    @Test("Cancel hits the API and is a no-op when idle")
    func cancel() async {
        let api = MockAgentAPI()
        let viewModel = CUAViewModel(api: api)
        await viewModel.cancel()
        #expect(api.cancelCalls == 0)
        viewModel.goal = "g"
        await viewModel.start()
        await viewModel.cancel()
        #expect(api.cancelCalls == 1)
    }

    @Test("Stop failure keeps the Stop action available")
    func cancelFailure() async {
        let api = MockAgentAPI()
        api.cancelShouldFail = true
        let viewModel = CUAViewModel(api: api)
        viewModel.goal = "g"
        await viewModel.start()
        await viewModel.cancel()

        #expect(api.cancelCalls == 1)
        #expect(viewModel.phase.isBusy)
        #expect(viewModel.actionError?.hasPrefix("Stop failed:") == true)
    }

    @Test("Polling failure keeps the active run controllable and retries")
    func pollingFailure() async {
        let api = MockAgentAPI()
        api.eventsShouldFail = true
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        viewModel.goal = "g"
        await viewModel.start()
        await drain()

        #expect(viewModel.phase == .running)
        #expect(viewModel.phase.isBusy)
        #expect(viewModel.actionError?.hasPrefix("Connection interrupted:") == true)
        await viewModel.cancel()
        #expect(api.cancelCalls == 1)
        #expect(viewModel.phase == .idle)
    }

    @Test("Empty goal cannot start")
    func cannotStartEmptyGoal() async {
        let api = MockAgentAPI()
        let viewModel = CUAViewModel(api: api)
        #expect(!viewModel.canStart)
        viewModel.goal = "   "
        #expect(!viewModel.canStart)
        await viewModel.start()
        #expect(api.createdRequests.isEmpty)
    }

    @Test("Terminal failure surfaces the reason")
    func stalledRunFails() async throws {
        let api = MockAgentAPI()
        api.finalSummary = ""
        api.scriptedEvents = [
            makeEvent(seq: 2, kind: "terminal", status: "stalled", summary: ""),
        ]
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        viewModel.goal = "long task"
        await viewModel.start()
        await drain()
        guard case .failed = viewModel.phase else {
            Issue.record("expected failed phase, got \(viewModel.phase)")
            return
        }
        #expect(!viewModel.phase.isBusy)
    }

    @Test("Terminal server errors remain visible")
    func terminalErrorIsVisible() async throws {
        let api = MockAgentAPI()
        api.scriptedEvents = [
            makeEvent(seq: 2, kind: "terminal", status: "failed", error: "planner unavailable"),
        ]
        let viewModel = CUAViewModel(api: api, pollIntervalNanos: 5_000_000)
        viewModel.goal = "open an article"
        await viewModel.start()
        await drain()

        #expect(viewModel.phase == .failed(message: "planner unavailable"))
    }
}

// MARK: - Client

struct RecordingStub: @unchecked Sendable {
    var request: URLRequest?
    var body: Data = Data("{}".utf8)
    var status = 200
}

@Suite("Agent Task Client", .serialized)
struct CUAClientTests {
    private func makeClient() -> CUAClient {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [RecordingURLProtocol.self]
        let session = URLSession(configuration: config)
        guard let client = CUAClient(
            host: "127.0.0.1", port: 8899, bearerToken: "tok", session: session
        ) else {
            Issue.record("client init failed")
            fatalError("unreachable")
        }
        return client
    }

    @Test("Rejects non-loopback hosts and empty tokens")
    func loopbackGuard() {
        #expect(CUAClient(host: "10.0.0.5", port: 8000, bearerToken: "t") == nil)
        #expect(CUAClient(host: "127.0.0.1", port: 8000, bearerToken: "") == nil)
        #expect(CUAClient(host: "localhost", port: 8000, bearerToken: "t") == nil)
    }

    @Test("Create posts JSON with bearer auth and decodes run_id")
    func createDecodes() async throws {
        RecordingURLProtocol.stubResponse(
            path: "/v1/cua/runs",
            body: Data(#"{"run_id":"abc","status":"running"}"#.utf8)
        )
        let client = makeClient()
        let runID = try await client.create(
            CUARunRequest(
                app: "Google Chrome", goal: "g", planner: "local-9b",
                openURL: "", allowedDomain: "wikipedia.org", maxSteps: 8,
                humanLogin: true
            )
        )
        #expect(runID == "abc")
        let captured = try #require(RecordingURLProtocol.captured["/v1/cua/runs"])
        #expect(captured.request.value(forHTTPHeaderField: "Authorization") == "Bearer tok")
        let sent = captured.body
        let json = try #require(JSONSerialization.jsonObject(with: sent) as? [String: Any])
        #expect(json["allowed_domain"] as? String == "wikipedia.org")
        #expect(json["max_steps"] as? Int == 8)
    }

    @Test("Events decode snake_case payload")
    func eventsDecode() async throws {
        let payload = """
        {"run_id":"abc","app":"Google Chrome","goal":"g","status":"running",
         "final_summary":"","error":"","planner":"p","events_after_seq":1,
         "events":[{"seq":2,"kind":"plan","step":1,"step_instruction":"click it",
                    "target_label":"Search","latency_s":0.4}],"run_dir":"/tmp/x"}
        """
        RecordingURLProtocol.stubResponse(path: "/v1/cua/runs/abc/events", body: Data(payload.utf8))
        let client = makeClient()
        let view = try await client.events(runID: "abc", after: 1)
        #expect(view.events.count == 1)
        #expect(view.events[0].stepInstruction == "click it")
        #expect(view.events[0].targetLabel == "Search")
    }

    @Test("HTTP errors surface as typed failures")
    func httpError() async {
        RecordingURLProtocol.stubResponse(
            path: "/v1/cua/runs/abc/approval",
            body: Data(#"{"detail":"run is not awaiting approval"}"#.utf8),
            status: 409
        )
        let client = makeClient()
        do {
            try await client.approve(runID: "abc")
            Issue.record("expected throw")
        } catch let error as CUAClientError {
            #expect(error == .http(409, "run is not awaiting approval"))
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }
}

/// Minimal URLProtocol double: stubs and captures are keyed by request path so
/// parallel tests never observe each other's responses.
final class RecordingURLProtocol: URLProtocol {
    struct Captured {
        let request: URLRequest
        let body: Data
    }

    nonisolated(unsafe) static var stubs: [String: RecordingStub] = [:]
    nonisolated(unsafe) static var captured: [String: Captured] = [:]

    static func stubResponse(path: String, body: Data = Data("{}".utf8), status: Int = 200) {
        stubs[path] = RecordingStub(body: body, status: status)
    }

    static func reset() {
        stubs = [:]
        captured = [:]
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let url = request.url else {
            client?.urlProtocol(self, didFailWithError: URLError(.badURL))
            return
        }
        Self.captured[url.path] = Captured(request: request, body: requestBody(request))
        let stub = Self.stubs[url.path] ?? RecordingStub(body: Data(), status: 404)
        let response = HTTPURLResponse(
            url: url, statusCode: stub.status, httpVersion: "HTTP/1.1", headerFields: nil
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: stub.body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}

    private func requestBody(_ request: URLRequest) -> Data {
        if let body = request.httpBody { return body }
        guard let stream = request.httpBodyStream else { return Data() }
        stream.open()
        defer { stream.close() }
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: 65_536)
        defer { buffer.deallocate() }
        var data = Data()
        while stream.hasBytesAvailable {
            let read = stream.read(buffer, maxLength: 65_536)
            if read <= 0 { break }
            data.append(buffer, count: read)
        }
        return data
    }
}

// MARK: - Add-brain settings

@Suite(.serialized)
struct CUAAddBrainTests {
    @MainActor
    @Test func saveBrainPostsRequestAndReloads() async throws {
        let api = MockAgentAPI()
        let vm = CUAViewModel(api: api)
        vm.newBrainName = "deepseek"
        vm.newBrainURL = "https://api.example.com/v1/chat/completions"
        vm.newBrainModel = "deepseek-reasoner"
        vm.newBrainAPIKey = "sk-test"
        vm.newBrainAllowRemote = true
        await vm.saveBrain()
        #expect(api.addedPlanners.count == 1)
        #expect(api.addedPlanners[0].name == "deepseek")
        #expect(api.addedPlanners[0].apiKey == "sk-test")
        #expect(api.addedPlanners[0].allowRemote == true)
        #expect(vm.showAddBrain == false)
        #expect(vm.brainError == nil)
        #expect(vm.newBrainName.isEmpty)
    }

    @MainActor
    @Test func saveBrainSurfacesError() async throws {
        let api = MockAgentAPI()
        api.addShouldFail = true
        let vm = CUAViewModel(api: api)
        vm.showAddBrain = true
        vm.newBrainName = "bad"
        vm.newBrainURL = "https://api.example.com/v1"
        vm.newBrainModel = "m"
        await vm.saveBrain()
        #expect(vm.brainError != nil)
        #expect(vm.showAddBrain == true)
    }

    @MainActor
    @Test func addBrainIsValidRequiresAllFields() {
        let api = MockAgentAPI()
        let vm = CUAViewModel(api: api)
        #expect(vm.addBrainIsValid == false)
        vm.newBrainName = "x"
        vm.newBrainURL = "https://api.example.com/v1"
        #expect(vm.addBrainIsValid == false)
        vm.newBrainModel = "m"
        #expect(vm.addBrainIsValid == true)
    }

    @MainActor
    @Test func plannerDisclosureSeparatesMacExecutionFromExternalBrainData() {
        let vm = CUAViewModel(api: MockAgentAPI())
        vm.plannerName = "external"
        vm.plannerOptions = [
            CUAPlannerOption(
                name: "external", model: "m", url: "https://planner.example/v1",
                textOnly: false, allowRemote: true
            )
        ]
        #expect(vm.plannerDisclosure.contains("Actions run on this Mac"))
        #expect(vm.plannerDisclosure.contains("external brain"))
        #expect(vm.plannerDisclosure.contains("screenshot"))

        vm.plannerOptions[0].textOnly = true
        #expect(!vm.plannerDisclosure.contains("screenshot"))
        #expect(vm.plannerDisclosure.contains("Accessibility snapshot"))
    }

    @MainActor
    @Test func loopbackEndpointClassificationMatchesServerRules() {
        #expect(CUAViewModel.isLoopbackEndpoint("http://localhost:1234/v1"))
        #expect(CUAViewModel.isLoopbackEndpoint("http://localhost.:1234/v1"))
        #expect(CUAViewModel.isLoopbackEndpoint("http://127.0.0.1:1234/v1"))
        #expect(CUAViewModel.isLoopbackEndpoint("http://[::1]:1234/v1"))
        #expect(!CUAViewModel.isLoopbackEndpoint("https://planner.example/v1"))
        #expect(!CUAViewModel.isLoopbackEndpoint("not a url"))
    }
}

@Suite(.serialized)
struct CUAPlannerDecodeTests {
    @Test func decodesLegacyServerJSONWithoutNewFields() throws {
        // Older sidecars predate has_api_key/user_created; the client must
        // still decode their planner list instead of failing the picker.
        let legacy = #"{"name":"local-9b","model":"m9","url":"http://127.0.0.1:1/v1","text_only":true}"#
        let data = Data(legacy.utf8)
        let option = try JSONDecoder().decode(CUAPlannerOption.self, from: data)
        #expect(option.name == "local-9b")
        #expect(option.hasApiKey == false)
        #expect(option.userCreated == false)
    }

    @Test func decodesSnakeCaseFields() throws {
        let full = #"{"name":"my-cloud","model":"m","url":"https://x/v1","text_only":false,"note":"","has_api_key":true,"user_created":true,"allow_remote":true}"#
        let option = try JSONDecoder().decode(CUAPlannerOption.self, from: Data(full.utf8))
        #expect(option.hasApiKey == true)
        #expect(option.userCreated == true)
        #expect(option.allowRemote == true)
    }
}
