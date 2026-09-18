import Foundation
import Testing
@testable import Rapid

@Suite("Agent Runtime desktop transport", .serialized)
struct AgentRuntimeClientTests {
    private func makeClient() -> AgentRuntimeClient {
        AgentRuntimeStubProtocol.reset()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [AgentRuntimeStubProtocol.self]
        return AgentRuntimeClient(
            baseURL: URL(string: "http://127.0.0.1:8123")!,
            session: URLSession(configuration: configuration)
        )
    }

    @Test("Create uses client execution, bearer auth, and exact tool selection")
    func createWireContract() async throws {
        let client = makeClient()
        AgentRuntimeStubProtocol.response = (202, Self.awaitingModel)

        let run = try await client.create(
            goal: "Find the answer",
            model: "minicpm5-2b-4bit",
            toolNames: ["files__read_file"],
            trustedInstructions: "Always answer concisely",
            localContext: "Preference: concise",
            recentUserMessages: ["Search my Documents folder"],
            execution: .client,
            bearerToken: "secret"
        )

        #expect(run.status == .awaitingModel)
        let request = try #require(AgentRuntimeStubProtocol.requests.first)
        #expect(request.url?.path == "/v1/agent/runs")
        #expect(request.httpMethod == "POST")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer secret")
        let body = try Self.jsonBody(at: 0)
        #expect(body["goal"] as? String == "Find the answer")
        #expect(body["model"] as? String == "minicpm5-2b-4bit")
        #expect(body["execution"] as? String == "client")
        #expect(body["tool_names"] as? [String] == ["files__read_file"])
        #expect(body["trusted_instructions"] as? String == "Always answer concisely")
        #expect(body["local_context"] as? String == "Preference: concise")
        #expect(body["recent_user_messages"] as? [String] == ["Search my Documents folder"])
    }

    @Test("Create retries without structured history on an older server")
    func createLegacyRecentUserMessagesRetry() async throws {
        let client = makeClient()
        AgentRuntimeStubProtocol.responses = [
            (422, Data(#"{"detail":"unknown field: recent_user_messages"}"#.utf8)),
            (200, Self.awaitingModel),
        ]

        _ = try await client.create(
            goal: "Search again",
            recentUserMessages: ["Search my Documents folder"],
            execution: .client
        )

        #expect(try Self.jsonBody(at: 0)["recent_user_messages"] != nil)
        #expect(try Self.jsonBody(at: 1)["recent_user_messages"] == nil)
    }

    @Test("Create can keep MCP execution pinned to the server run")
    func createWithServerExecution() async throws {
        let client = makeClient()
        AgentRuntimeStubProtocol.response = (202, Self.awaitingModel)

        _ = try await client.create(
            goal: "Search safely",
            toolNames: ["search__query"],
            execution: .server
        )

        let body = try Self.jsonBody(at: 0)
        #expect(body["execution"] as? String == "server")
    }

    @Test("Pending approvals decode summaries without exposing arguments")
    func approvalDecodeAndResolve() async throws {
        let client = makeClient()
        AgentRuntimeStubProtocol.responses = [
            (200, Self.awaitingApproval),
            (200, Self.awaitingToolResult),
        ]

        let pending = try await client.get(
            runID: "01234567-89ab-cdef-0123-456789abcdef",
            bearerToken: nil
        )
        #expect(pending.status == .awaitingApproval)
        #expect(pending.pendingAction?.callID == "call-1")
        #expect(pending.pendingAction?.arguments == [:])
        #expect(pending.pendingAction?.approvalSummary?["path"] == .string("notes.md"))
        let requestURL = try #require(AgentRuntimeStubProtocol.requests[0].url)
        let requestComponents = try #require(URLComponents(
            url: requestURL,
            resolvingAgainstBaseURL: false
        ))
        #expect(
            requestComponents.percentEncodedPath
                == "/v1/agent/runs/01234567-89ab-cdef-0123-456789abcdef"
        )

        let approved = try await client.resolveApproval(
            runID: pending.id,
            callID: "call-1",
            approved: true,
            bearerToken: "token"
        )
        #expect(approved.status == .awaitingToolResult)
        #expect(approved.pendingAction?.arguments["path"] == .string("notes.md"))
        let body = try Self.jsonBody(at: 1)
        #expect(body["call_id"] as? String == "call-1")
        #expect(body["approved"] as? Bool == true)
    }

    @Test("Run IDs cannot escape through dot segments")
    func rejectsDotSegmentRunIDs() async {
        let client = makeClient()

        for runID in ["", ".", "..", "../victim", "run/events", "RUN-1"] {
            do {
                _ = try await client.get(runID: runID)
                Issue.record("Expected \(runID.debugDescription) to be rejected")
            } catch let error as AgentRuntimeClientError {
                #expect(error == .invalidRunID)
            } catch {
                Issue.record("Unexpected error: \(error)")
            }
        }
        #expect(AgentRuntimeStubProtocol.requests.isEmpty)
    }

    @Test("Tool results preserve executed accounting")
    func toolResultWireContract() async throws {
        let client = makeClient()
        AgentRuntimeStubProtocol.response = (200, Self.awaitingModel)

        _ = try await client.submitToolResult(
            runID: "01234567-89ab-cdef-0123-456789abcdef",
            callID: "call-1",
            content: "not dispatched",
            isError: true,
            executed: false,
            bearerToken: nil
        )

        let body = try Self.jsonBody(at: 0)
        #expect(body["call_id"] as? String == "call-1")
        #expect(body["content"] as? String == "not dispatched")
        #expect(body["is_error"] as? Bool == true)
        #expect(body["executed"] as? Bool == false)
        #expect(body["declined"] == nil)

        _ = try await client.submitToolResult(
            runID: "01234567-89ab-cdef-0123-456789abcdef",
            callID: "call-2",
            content: "declined",
            isError: true,
            executed: false,
            declined: true,
            bearerToken: nil
        )
        let declinedBody = try Self.jsonBody(at: 1)
        #expect(declinedBody["declined"] as? Bool == true)
    }

    @Test("Declined result retries without the new field on an older server")
    func declinedResultLegacyRetry() async throws {
        let client = makeClient()
        AgentRuntimeStubProtocol.responses = [
            (422, Data(#"{"detail":"unknown field: declined"}"#.utf8)),
            (200, Self.awaitingModel),
        ]

        _ = try await client.submitToolResult(
            runID: "01234567-89ab-cdef-0123-456789abcdef",
            callID: "call-1",
            content: "The user declined this action.",
            isError: true,
            executed: false,
            declined: true,
            bearerToken: nil
        )

        #expect(try Self.jsonBody(at: 0)["declined"] as? Bool == true)
        #expect(try Self.jsonBody(at: 1)["declined"] == nil)
    }

    @Test("Declined result does not retry unrelated validation failures")
    func declinedResultDoesNotRetryUnrelated422() async {
        let client = makeClient()
        AgentRuntimeStubProtocol.response = (
            422,
            Data(#"{"detail":"a declined client tool cannot be executed"}"#.utf8)
        )

        do {
            _ = try await client.submitToolResult(
                runID: "01234567-89ab-cdef-0123-456789abcdef",
                callID: "call-1",
                content: "The user declined this action.",
                isError: true,
                executed: true,
                declined: true,
                bearerToken: nil
            )
            Issue.record("Expected the contradictory result to be rejected")
        } catch let error as AgentRuntimeClientError {
            #expect(error == .http(
                status: 422,
                message: "a declined client tool cannot be executed"
            ))
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
        #expect(AgentRuntimeStubProtocol.requests.count == 1)
    }

    @Test("Declined result retries only an unknown-field response")
    func declinedResultDoesNotRetryGeneric422() async {
        let client = makeClient()
        AgentRuntimeStubProtocol.response = (
            422,
            Data(#"{"detail":"tool result does not match the pending action"}"#.utf8)
        )

        do {
            _ = try await client.submitToolResult(
                runID: "01234567-89ab-cdef-0123-456789abcdef",
                callID: "call-1",
                content: "The user declined this action.",
                isError: true,
                executed: false,
                declined: true,
                bearerToken: nil
            )
            Issue.record("Expected the mismatched result to be rejected")
        } catch let error as AgentRuntimeClientError {
            #expect(error == .http(
                status: 422,
                message: "tool result does not match the pending action"
            ))
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
        #expect(AgentRuntimeStubProtocol.requests.count == 1)
    }

    @Test("Event cursors are monotonic client inputs")
    func eventCursor() async throws {
        let client = makeClient()
        AgentRuntimeStubProtocol.response = (200, Self.events)

        let response = try await client.events(
            runID: "01234567-89ab-cdef-0123-456789abcdef",
            after: 7,
            bearerToken: nil
        )

        #expect(response.runID == "01234567-89ab-cdef-0123-456789abcdef")
        #expect(response.nextAfter == 8)
        #expect(response.events.map(\.sequence) == [8])
        let url = try #require(AgentRuntimeStubProtocol.requests.first?.url)
        let components = try #require(URLComponents(
            url: url,
            resolvingAgainstBaseURL: false
        ))
        #expect(components.queryItems == [URLQueryItem(name: "after", value: "7")])
    }

    @Test("Server detail is the actionable HTTP error")
    func serverError() async throws {
        let client = makeClient()
        AgentRuntimeStubProtocol.response = (
            409,
            Data(#"{"detail":"tool result does not match the pending action"}"#.utf8)
        )

        do {
            _ = try await client.cancel(
                runID: "01234567-89ab-cdef-0123-456789abcdef",
                bearerToken: nil
            )
            Issue.record("Expected the HTTP error")
        } catch let error as AgentRuntimeClientError {
            #expect(error == .http(
                status: 409,
                message: "tool result does not match the pending action"
            ))
        }
    }

    @Test("Personal Intelligence recognizes only model-specific harness bindings")
    func personalIntelligenceModelSupport() {
        let qualified = ServerModelProfile(
            id: "minicpm5-2b-4bit",
            toolCallParser: "minicpm",
            personalIntelligenceProfile: "minicpm5-2b",
            personalIntelligenceQualification: "minicpm5-2b-q4-v1"
        )
        #expect(PersonalIntelligenceConfig.supportsModel(
            "minicpm5-2b-4bit",
            serverProfile: qualified
        ))
        #expect(!PersonalIntelligenceConfig.supportsModel(
            "qwen3.5-4b-4bit",
            serverProfile: qualified
        ))
        #expect(!PersonalIntelligenceConfig.supportsModel(
            "MiniCPM5-2B-4bit",
            serverProfile: qualified
        ))
        #expect(!PersonalIntelligenceConfig.supportsModel(
            "minicpm5-2b-4bit",
            serverProfile: ServerModelProfile(id: "minicpm5-2b-4bit")
        ))
        #expect(!PersonalIntelligenceConfig.supportsModel(
            "minicpm5-2b-4bit",
            serverProfile: ServerModelProfile(
                id: "minicpm5-2b-4bit",
                personalIntelligenceProfile: "minicpm5-2b",
                personalIntelligenceQualification: "minicpm5-2b-q4-v1"
            )
        ))
        #expect(!PersonalIntelligenceConfig.supportsModel(
            "minicpm5-2b-4bit",
            serverProfile: ServerModelProfile(
                id: "minicpm5-2b-4bit",
                toolCallParser: "minicpm",
                personalIntelligenceProfile: "minicpm5-2b"
            )
        ))
        #expect(PersonalIntelligenceConfig.isEnabled(
            conversationEnabled: true,
            alias: "minicpm5-2b-4bit",
            serverProfile: qualified
        ))
        #expect(!PersonalIntelligenceConfig.isEnabled(
            conversationEnabled: true,
            alias: "qwen3.5-4b-4bit",
            serverProfile: qualified
        ))
    }

    @Test("Only new conversations inherit the post-consent default")
    func personalIntelligenceConversationDefaults() {
        let oldA = UUID()
        let oldB = UUID()
        let newDraft = UUID()

        let existing = PersonalIntelligenceConfig.reconciledConversationStates(
            [:],
            activeConversationID: oldA,
            storedConversationIDs: [oldA, oldB],
            introductionCompleted: true,
            preferredEnabled: true
        )
        #expect(existing[oldA] == false)
        #expect(existing[oldB] == false)

        // Conversation persistence inserts the row before selecting it. The
        // insertion transaction must record the inherited preference so the
        // subsequent active-ID reconciliation cannot mistake it for an old
        // saved conversation.
        let insertedBeforeSelection = PersonalIntelligenceConfig.reconciledConversationStates(
            existing,
            activeConversationID: oldA,
            storedConversationIDs: [oldA, oldB, newDraft],
            newlyCreatedConversationIDs: [newDraft],
            introductionCompleted: true,
            preferredEnabled: true
        )
        #expect(insertedBeforeSelection[newDraft] == true)

        let created = PersonalIntelligenceConfig.reconciledConversationStates(
            insertedBeforeSelection,
            activeConversationID: newDraft,
            storedConversationIDs: [oldA, oldB, newDraft],
            introductionCompleted: true,
            preferredEnabled: true
        )
        #expect(created[newDraft] == true)

        let afterDeletingOldConversations =
            PersonalIntelligenceConfig.reconciledConversationStates(
                created,
                activeConversationID: newDraft,
                storedConversationIDs: [newDraft],
                introductionCompleted: true,
                preferredEnabled: true
            )
        #expect(afterDeletingOldConversations == [newDraft: true])

        let declined = PersonalIntelligenceConfig.reconciledConversationStates(
            [:],
            activeConversationID: newDraft,
            storedConversationIDs: [],
            introductionCompleted: false,
            preferredEnabled: true
        )
        #expect(declined[newDraft] == false)
    }

    @Test("Per-conversation choices round-trip through preferences")
    func personalIntelligencePersistence() throws {
        let suite = "AgentRuntimeClientTests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let enabled = UUID()
        let disabled = UUID()

        PersonalIntelligenceConfig.saveConversationStates(
            [enabled: true, disabled: false],
            to: defaults
        )

        #expect(PersonalIntelligenceConfig.loadConversationStates(from: defaults) == [
            enabled: true,
            disabled: false,
        ])
    }

    private static func jsonBody(at index: Int) throws -> [String: Any] {
        let body = try #require(AgentRuntimeStubProtocol.bodies[safe: index])
        return try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
    }

    private static let awaitingModel = Data(#"""
    {
      "id":"01234567-89ab-cdef-0123-456789abcdef","model":"minicpm5-2b-4bit","profile":"minicpm5-2b",
      "status":"awaiting_model","model_turns":1,"tool_rounds":0,
      "final_synthesis":false,"failure_code":null,"output":null,"pending_action":null
    }
    """#.utf8)

    private static let awaitingApproval = Data(#"""
    {
      "id":"01234567-89ab-cdef-0123-456789abcdef","model":"minicpm5-2b-4bit","profile":"minicpm5-2b",
      "status":"awaiting_approval","model_turns":1,"tool_rounds":0,
      "final_synthesis":false,"failure_code":null,"output":null,
      "pending_action":{"call_id":"call-1","name":"files__write_file",
        "approval_summary":{"path":"notes.md"},"risk":"external_side_effect",
        "approval_required":true}
    }
    """#.utf8)

    private static let awaitingToolResult = Data(#"""
    {
      "id":"01234567-89ab-cdef-0123-456789abcdef","model":"minicpm5-2b-4bit","profile":"minicpm5-2b",
      "status":"awaiting_tool_result","model_turns":1,"tool_rounds":0,
      "final_synthesis":false,"failure_code":null,"output":null,
      "pending_action":{"call_id":"call-1","name":"files__write_file",
        "arguments":{"path":"notes.md","content":"hello"},"approval_summary":null,
        "risk":"external_side_effect","approval_required":false}
    }
    """#.utf8)

    private static let events = Data(#"""
    {
      "run_id":"01234567-89ab-cdef-0123-456789abcdef","status":"awaiting_model","next_after":8,
      "events":[{"schema_version":1,"sequence":8,"type":"model.requested",
        "created_at":1.25,"data":{"round":2}}]
    }
    """#.utf8)
}

private extension Array {
    subscript(safe index: Index) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}

private final class AgentRuntimeStubProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var requests: [URLRequest] = []
    nonisolated(unsafe) static var bodies: [Data] = []
    nonisolated(unsafe) static var response: (Int, Data) = (200, Data())
    nonisolated(unsafe) static var responses: [(Int, Data)] = []

    static func reset() {
        requests = []
        bodies = []
        response = (200, Data())
        responses = []
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.requests.append(request)
        Self.bodies.append(Self.readBody(from: request))
        let item = Self.responses.isEmpty ? Self.response : Self.responses.removeFirst()
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: item.0,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "application/json"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: item.1)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}

    private static func readBody(from request: URLRequest) -> Data {
        if let body = request.httpBody { return body }
        guard let stream = request.httpBodyStream else { return Data() }
        stream.open()
        defer { stream.close() }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while true {
            let count = stream.read(&buffer, maxLength: buffer.count)
            if count <= 0 { break }
            data.append(buffer, count: count)
        }
        return data
    }
}
