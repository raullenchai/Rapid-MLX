import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Agent Runtime session controller", .serialized)
struct AgentSessionControllerTests {
    @Test("A server-owned run reaches a completed UI state")
    func completesRun() async throws {
        let transport = AgentSessionTransportStub(
            created: try Self.run(status: "awaiting_model"),
            gets: [try Self.run(status: "completed", output: "Finished locally.")],
            eventPages: [
                try Self.events(sequence: 1, nextAfter: 1),
                try Self.events(sequence: 2, nextAfter: 2),
            ]
        )
        let controller = AgentSessionController(
            transportFactory: { _ in transport },
            pollDelay: { await Task.yield() }
        )

        controller.start(
            goal: "Organize these notes",
            model: "minicpm5-2b-4bit",
            baseURL: URL(string: "http://127.0.0.1:8000")!,
            bearerToken: "secret"
        )
        await controller._testingWaitForDriver()

        #expect(controller.phase == .completed)
        #expect(controller.run?.output == "Finished locally.")
        #expect(controller.events.map(\.sequence) == [1, 2])
        #expect(controller.eventCursor == 2)
        #expect(transport.createExecution == .server)
        #expect(transport.createBearer == "secret")
        #expect(transport.eventCursors == [0, 1])
    }

    @Test("Approval pauses and resumes only the pending call")
    func approvalRoundTrip() async throws {
        let transport = AgentSessionTransportStub(
            created: try Self.run(
                status: "awaiting_approval",
                pendingAction: """
                {"call_id":"call-7","name":"notes__write","approval_summary":{"path":"Notes/report.md"},"risk":"local_change","approval_required":true}
                """
            ),
            gets: [try Self.run(status: "completed", output: "Saved.")],
            eventPages: [
                try Self.events(sequence: 1, nextAfter: 1),
                try Self.events(sequence: 2, nextAfter: 2),
            ],
            approvalResult: try Self.run(status: "awaiting_model")
        )
        let controller = AgentSessionController(
            transportFactory: { _ in transport },
            pollDelay: { await Task.yield() }
        )

        controller.start(
            goal: "Save a summary",
            model: nil,
            baseURL: URL(string: "http://127.0.0.1:8000")!,
            bearerToken: nil
        )
        await controller._testingWaitForDriver()
        #expect(controller.phase == .awaitingApproval)
        #expect(controller.pendingApproval?.arguments == [:])
        #expect(controller.pendingApproval?.approvalSummary?["path"] == .string("Notes/report.md"))

        controller.resolvePendingApproval(approved: true)
        await controller._testingWaitForDriver()

        #expect(transport.approvals == [Approval(callID: "call-7", approved: true)])
        #expect(controller.phase == .completed)
        #expect(controller.run?.output == "Saved.")
    }

    @Test("Cancellation during create cleans up the late server run")
    func cancellationDuringCreate() async throws {
        let transport = AgentSessionTransportStub(
            created: try Self.run(status: "awaiting_model"),
            suspendsCreate: true,
            throwsIfCreateCancelled: true
        )
        let controller = AgentSessionController(
            transportFactory: { _ in transport },
            pollDelay: { await Task.yield() }
        )

        controller.start(
            goal: "Slow start",
            model: nil,
            baseURL: URL(string: "http://127.0.0.1:8000")!,
            bearerToken: "secret"
        )
        await transport.waitUntilCreateIsSuspended()
        controller.cancel()
        transport.releaseCreate()
        await transport.waitForCancellation()

        #expect(controller.phase == .cancelled)
        #expect(transport.cancelledRunIDs == [Self.runID])
        #expect(transport.cancelBearers == ["secret"])
    }

    @Test("Cancellation invalidates the local run before remote acknowledgement")
    func cancellationIsImmediate() async throws {
        let transport = AgentSessionTransportStub(
            created: try Self.run(
                status: "awaiting_approval",
                pendingAction: """
                {"call_id":"call-8","name":"notes__write","approval_summary":{},"risk":"local_change","approval_required":true}
                """
            )
        )
        let controller = AgentSessionController(
            transportFactory: { _ in transport },
            pollDelay: { await Task.yield() }
        )

        controller.start(
            goal: "Write",
            model: nil,
            baseURL: URL(string: "http://127.0.0.1:8000")!,
            bearerToken: nil
        )
        await controller._testingWaitForDriver()
        controller.cancel()
        await Task.yield()

        #expect(controller.phase == .cancelled)
        #expect(controller.pendingApproval == nil)
        #expect(transport.cancelledRunIDs == [Self.runID])
    }

    @Test("Cancellation during the final event drain still cancels the server run")
    func cancellationDuringFinalDrain() async throws {
        let transport = AgentSessionTransportStub(
            created: try Self.run(status: "completed", output: "Done."),
            suspendsEvents: true
        )
        let controller = AgentSessionController(
            transportFactory: { _ in transport },
            pollDelay: { await Task.yield() }
        )

        controller.start(
            goal: "Finish then stop",
            model: nil,
            baseURL: URL(string: "http://127.0.0.1:8000")!,
            bearerToken: "secret"
        )
        await transport.waitUntilEventsAreSuspended()
        controller.cancel()
        transport.releaseEvents()
        await transport.waitForCancellation()

        #expect(controller.phase == .cancelled)
        #expect(transport.cancelledRunIDs == [Self.runID])
        #expect(transport.cancelBearers == ["secret"])
    }

    @Test("A second start cannot replace an active approval")
    func activeRunCannotBeReplaced() async throws {
        let transport = AgentSessionTransportStub(
            created: try Self.run(
                status: "awaiting_approval",
                pendingAction: """
                {"call_id":"call-9","name":"notes__write","approval_summary":{},"risk":"local_change","approval_required":true}
                """
            )
        )
        let controller = AgentSessionController(
            transportFactory: { _ in transport },
            pollDelay: { await Task.yield() }
        )

        controller.start(
            goal: "First",
            model: nil,
            baseURL: URL(string: "http://127.0.0.1:8000")!,
            bearerToken: nil
        )
        await controller._testingWaitForDriver()
        controller.start(
            goal: "Second",
            model: nil,
            baseURL: URL(string: "http://127.0.0.1:8000")!,
            bearerToken: nil
        )

        #expect(transport.createGoals == ["First"])
        #expect(controller.pendingApproval?.callID == "call-9")
    }

    @Test("Observation failure cancels a remotely live run")
    func observationFailureCancelsRemoteRun() async throws {
        let transport = AgentSessionTransportStub(
            created: try Self.run(status: "awaiting_model"),
            eventError: AgentSessionStubError.disconnected
        )
        let controller = AgentSessionController(
            transportFactory: { _ in transport },
            pollDelay: { await Task.yield() }
        )

        controller.start(
            goal: "Observe",
            model: nil,
            baseURL: URL(string: "http://127.0.0.1:8000")!,
            bearerToken: nil
        )
        await controller._testingWaitForDriver()
        await transport.waitForCancellation()

        #expect(controller.phase == .failed)
        #expect(transport.cancelledRunIDs == [Self.runID])
    }

    @Test("A cancelled final event drain cannot hide a completed answer")
    func cancelledFinalDrainPreservesCompletion() async throws {
        let transport = AgentSessionTransportStub(
            created: try Self.run(status: "completed", output: "Still finished."),
            eventError: CancellationError()
        )
        let controller = AgentSessionController(
            transportFactory: { _ in transport },
            pollDelay: { await Task.yield() }
        )

        controller.start(
            goal: "Finish",
            model: nil,
            baseURL: URL(string: "http://127.0.0.1:8000")!,
            bearerToken: nil
        )
        await controller._testingWaitForDriver()

        #expect(controller.phase == .completed)
        #expect(controller.run?.output == "Still finished.")
        #expect(transport.cancelledRunIDs.isEmpty)
    }

    @Test("Dropping an active controller cancels its remote run")
    func deinitCancelsRemoteRun() async throws {
        let transport = AgentSessionTransportStub(
            created: try Self.run(
                status: "awaiting_approval",
                pendingAction: """
                {"call_id":"call-10","name":"notes__write","approval_summary":{},"risk":"local_change","approval_required":true}
                """
            )
        )
        var controller: AgentSessionController? = AgentSessionController(
            transportFactory: { _ in transport },
            pollDelay: { await Task.yield() }
        )
        weak var weakController = controller

        controller?.start(
            goal: "Close the window",
            model: nil,
            baseURL: URL(string: "http://127.0.0.1:8000")!,
            bearerToken: nil
        )
        await controller?._testingWaitForDriver()
        controller = nil
        await transport.waitForCancellation()

        #expect(weakController == nil)
        #expect(transport.cancelledRunIDs == [Self.runID])
    }

    private static let runID = "01234567-89ab-cdef-0123-456789abcdef"

    fileprivate static func run(
        status: String,
        output: String? = nil,
        pendingAction: String = "null"
    ) throws -> AgentRunView {
        let outputJSON = output.map { "\"\($0)\"" } ?? "null"
        return try JSONDecoder().decode(AgentRunView.self, from: Data("""
        {"id":"\(runID)","model":"minicpm5-2b-4bit","profile":"minicpm5-2b","status":"\(status)","model_turns":1,"tool_rounds":0,"final_synthesis":false,"failure_code":null,"output":\(outputJSON),"pending_action":\(pendingAction)}
        """.utf8))
    }

    private static func events(sequence: Int, nextAfter: Int) throws -> AgentEventsView {
        try JSONDecoder().decode(AgentEventsView.self, from: Data("""
        {"run_id":"\(runID)","status":"awaiting_model","next_after":\(nextAfter),"events":[{"schema_version":1,"sequence":\(sequence),"type":"model.requested","created_at":1.0,"data":{}}]}
        """.utf8))
    }

    fileprivate static func emptyEvents(nextAfter: Int) throws -> AgentEventsView {
        try JSONDecoder().decode(AgentEventsView.self, from: Data("""
        {"run_id":"\(runID)","status":"awaiting_model","next_after":\(nextAfter),"events":[]}
        """.utf8))
    }
}

private struct Approval: Equatable {
    let callID: String
    let approved: Bool
}

private enum AgentSessionStubError: Error {
    case disconnected
}

@MainActor
private final class AgentSessionTransportStub: AgentRuntimeTransport, @unchecked Sendable {
    let created: AgentRunView
    var gets: [AgentRunView]
    var eventPages: [AgentEventsView]
    let approvalResult: AgentRunView?
    let suspendsCreate: Bool
    let throwsIfCreateCancelled: Bool
    let suspendsEvents: Bool
    let eventError: Error?

    var createGoals: [String] = []
    var createExecution: AgentExecutionMode?
    var createBearer: String?
    var eventCursors: [Int] = []
    var approvals: [Approval] = []
    var cancelledRunIDs: [String] = []
    var cancelBearers: [String?] = []
    private var createContinuation: CheckedContinuation<Void, Never>?
    private var createSuspendedContinuation: CheckedContinuation<Void, Never>?
    private var eventsContinuation: CheckedContinuation<Void, Never>?
    private var eventsSuspendedContinuation: CheckedContinuation<Void, Never>?
    private var cancelContinuation: CheckedContinuation<Void, Never>?

    init(
        created: AgentRunView,
        gets: [AgentRunView] = [],
        eventPages: [AgentEventsView] = [],
        approvalResult: AgentRunView? = nil,
        suspendsCreate: Bool = false,
        throwsIfCreateCancelled: Bool = false,
        suspendsEvents: Bool = false,
        eventError: Error? = nil
    ) {
        self.created = created
        self.gets = gets
        self.eventPages = eventPages
        self.approvalResult = approvalResult
        self.suspendsCreate = suspendsCreate
        self.throwsIfCreateCancelled = throwsIfCreateCancelled
        self.suspendsEvents = suspendsEvents
        self.eventError = eventError
    }

    func create(
        goal: String,
        model _: String?,
        toolNames _: [String]?,
        execution: AgentExecutionMode,
        bearerToken: String?
    ) async throws -> AgentRunView {
        createGoals.append(goal)
        createExecution = execution
        createBearer = bearerToken
        if suspendsCreate {
            await withCheckedContinuation { continuation in
                createContinuation = continuation
                createSuspendedContinuation?.resume()
                createSuspendedContinuation = nil
            }
        }
        if throwsIfCreateCancelled {
            try Task.checkCancellation()
        }
        return created
    }

    func get(runID _: String, bearerToken _: String?) async throws -> AgentRunView {
        gets.removeFirst()
    }

    func events(
        runID _: String,
        after: Int,
        bearerToken _: String?
    ) async throws -> AgentEventsView {
        eventCursors.append(after)
        if suspendsEvents {
            await withCheckedContinuation { continuation in
                eventsContinuation = continuation
                eventsSuspendedContinuation?.resume()
                eventsSuspendedContinuation = nil
            }
        }
        if let eventError { throw eventError }
        if eventPages.isEmpty {
            return try AgentSessionControllerTests.emptyEvents(nextAfter: after)
        }
        return eventPages.removeFirst()
    }

    func resolveApproval(
        runID _: String,
        callID: String,
        approved: Bool,
        bearerToken _: String?
    ) async throws -> AgentRunView {
        approvals.append(Approval(callID: callID, approved: approved))
        return try #require(approvalResult)
    }

    func submitToolResult(
        runID _: String,
        callID _: String,
        content _: String,
        isError _: Bool,
        executed _: Bool,
        bearerToken _: String?
    ) async throws -> AgentRunView {
        Issue.record("Client execution is outside this session slice")
        return created
    }

    func cancel(runID: String, bearerToken: String?) async throws -> AgentRunView {
        cancelledRunIDs.append(runID)
        cancelBearers.append(bearerToken)
        cancelContinuation?.resume()
        cancelContinuation = nil
        return try AgentSessionControllerTests.run(status: "cancelled")
    }

    func waitUntilCreateIsSuspended() async {
        if createContinuation != nil { return }
        await withCheckedContinuation { createSuspendedContinuation = $0 }
    }

    func releaseCreate() {
        createContinuation?.resume()
        createContinuation = nil
    }

    func waitUntilEventsAreSuspended() async {
        if eventsContinuation != nil { return }
        await withCheckedContinuation { eventsSuspendedContinuation = $0 }
    }

    func releaseEvents() {
        eventsContinuation?.resume()
        eventsContinuation = nil
    }

    func waitForCancellation() async {
        if !cancelledRunIDs.isEmpty { return }
        await withCheckedContinuation { cancelContinuation = $0 }
    }
}
