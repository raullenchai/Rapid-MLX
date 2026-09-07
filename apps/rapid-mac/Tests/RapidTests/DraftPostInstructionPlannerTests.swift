import Foundation
import Testing
@testable import Rapid

@Suite("Computer Use natural-language draft planning")
struct DraftPostInstructionPlannerTests {
    @Test("A ready response becomes a bounded review plan")
    func readyPlan() async throws {
        let transport = PlannerTransport(response: try Self.response(content: [
            "status": "ready",
            "purpose": "Launch Rapid 0.13.4",
            "audience": "Mac developers",
            "talking_points": ["Faster local inference", "No cloud upload"],
            "tone": "Concise and enthusiastic",
            "destination": "x.com",
            "draft": "Rapid 0.13.4 is here — faster and fully local.",
            "clarifying_question": "",
        ]))
        let result = try await Self.planner(transport: transport).analyze(
            instruction: "Write a launch post for X for Mac developers.",
            browserApplication: "Google Chrome",
            destinationHost: "x.com"
        )
        #expect(result == .ready(DraftPostPlan(
            purpose: "Launch Rapid 0.13.4",
            audience: "Mac developers",
            talkingPoints: ["Faster local inference", "No cloud upload"],
            tone: "Concise and enthusiastic",
            destination: "x.com",
            draft: "Rapid 0.13.4 is here — faster and fully local."
        )))

        let request = try #require(await transport.requests.first)
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer secret")
        let body = try #require(request.httpBody)
        let json = try #require(
            JSONSerialization.jsonObject(with: body) as? [String: Any]
        )
        #expect(json["stream"] as? Bool == false)
        #expect(json["tools"] == nil)
        let responseFormat = try #require(json["response_format"] as? [String: Any])
        #expect(responseFormat["type"] as? String == "json_schema")
        let schemaEnvelope = try #require(
            responseFormat["json_schema"] as? [String: Any]
        )
        #expect(schemaEnvelope["strict"] as? Bool == true)
    }

    @Test("A model clarification response is exposed without a guessed plan")
    func clarification() async throws {
        let transport = PlannerTransport(response: try Self.response(content: [
            "status": "needs_clarification",
            "purpose": "",
            "audience": "",
            "talking_points": [],
            "tone": "",
            "destination": "",
            "draft": "",
            "clarifying_question": "Which site should I prepare this for?",
        ]))
        let result = try await Self.planner(transport: transport).analyze(
            instruction: "Write something about the release.",
            browserApplication: "Safari",
            destinationHost: "x.com"
        )
        #expect(result == .needsClarification("Which site should I prepare this for?"))
    }

    @Test("The planning request requires clarification for incomplete or mismatched intent")
    func clarificationPromptContract() throws {
        let data = try LocalDraftPostInstructionPlanner.requestBody(
            instruction: "Write something about the release.",
            browserApplication: "Safari",
            destinationHost: "x.com",
            model: "qwen3.5-9b-4bit"
        )
        let body = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let messages = try #require(body["messages"] as? [[String: String]])
        let system = try #require(messages.first?["content"])
        #expect(system.contains("did not name the destination service or site"))
        #expect(system.contains("names a destination inconsistent with the trusted browser hostname"))
        #expect(system.contains("return needs_clarification and ask exactly one concise question"))
        let user = try #require(messages.last?["content"])
        #expect(user.contains("Trusted destination hostname: x.com"))
        #expect(user.contains("Write something about the release."))
    }

    @Test("Unknown fields and incomplete ready plans fail closed")
    func strictOutputBoundary() async throws {
        var extra: [String: Any] = [
            "status": "ready",
            "purpose": "Launch",
            "audience": "Developers",
            "talking_points": ["Local"],
            "tone": "Concise",
            "destination": "x.com",
            "draft": "A draft",
            "clarifying_question": "",
        ]
        extra["publish_now"] = true
        let extraTransport = PlannerTransport(response: try Self.response(content: extra))
        await #expect(throws: DraftPostPlanningError.invalidResponse) {
            _ = try await Self.planner(transport: extraTransport).analyze(
                instruction: "Draft a launch post for X.",
                browserApplication: "Safari",
                destinationHost: "x.com"
            )
        }

        let emptyDraft = PlannerTransport(response: try Self.response(content: [
            "status": "ready",
            "purpose": "Launch",
            "audience": "Developers",
            "talking_points": ["Local"],
            "tone": "Concise",
            "destination": "x.com",
            "draft": "",
            "clarifying_question": "",
        ]))
        await #expect(throws: DraftPostPlanningError.invalidResponse) {
            _ = try await Self.planner(transport: emptyDraft).analyze(
                instruction: "Draft a launch post for X.",
                browserApplication: "Safari",
                destinationHost: "x.com"
            )
        }

        let wrongDestination = PlannerTransport(response: try Self.response(content: [
            "status": "ready",
            "purpose": "Launch",
            "audience": "Developers",
            "talking_points": ["Local"],
            "tone": "Concise",
            "destination": "example.com",
            "draft": "A draft",
            "clarifying_question": "",
        ]))
        await #expect(throws: DraftPostPlanningError.invalidResponse) {
            _ = try await Self.planner(transport: wrongDestination).analyze(
                instruction: "Draft a launch post for X.",
                browserApplication: "Safari",
                destinationHost: "x.com"
            )
        }

        let multipleQuestions = PlannerTransport(response: try Self.response(content: [
            "status": "needs_clarification",
            "purpose": "",
            "audience": "",
            "talking_points": [],
            "tone": "",
            "destination": "",
            "draft": "",
            "clarifying_question": "Who is this for? Which tone should I use?",
        ]))
        await #expect(throws: DraftPostPlanningError.invalidResponse) {
            _ = try await Self.planner(transport: multipleQuestions).analyze(
                instruction: "Draft a launch post for X.",
                browserApplication: "Safari",
                destinationHost: "x.com"
            )
        }

        let punctuationOnly = PlannerTransport(response: try Self.response(content: [
            "status": "needs_clarification",
            "purpose": "",
            "audience": "",
            "talking_points": [],
            "tone": "",
            "destination": "",
            "draft": "",
            "clarifying_question": "?",
        ]))
        await #expect(throws: DraftPostPlanningError.invalidResponse) {
            _ = try await Self.planner(transport: punctuationOnly).analyze(
                instruction: "Draft a launch post for X.",
                browserApplication: "Safari",
                destinationHost: "x.com"
            )
        }
    }

    @Test("Empty and oversized instructions never reach the model")
    func instructionBounds() async {
        let transport = PlannerTransport(response: LocalComputerUseGroundingHTTPResponse(
            statusCode: 200,
            contentType: "application/json",
            body: Data()
        ))
        await #expect(throws: DraftPostPlanningError.instructionMissing) {
            _ = try await Self.planner(transport: transport).analyze(
                instruction: "   ",
                browserApplication: "Safari",
                destinationHost: "x.com"
            )
        }
        await #expect(throws: DraftPostPlanningError.instructionTooLarge) {
            _ = try await Self.planner(transport: transport).analyze(
                instruction: String(
                    repeating: "x",
                    count: LocalDraftPostInstructionPlanner.maximumInstructionBytes + 1
                ),
                browserApplication: "Safari",
                destinationHost: "x.com"
            )
        }
        #expect(await transport.requests.isEmpty)
    }

    @Test("Only a resident text-capable exact model can plan")
    func runtimeEligibility() throws {
        let validator: DraftPostLanguageRuntime.SessionValidator = { true }
        let profile = ServerModelProfile(id: "qwen3.5-9b-4bit", modality: "text")
        let runtime = try #require(DraftPostLanguageRuntime(
            profile: profile,
            selectedAlias: "QWEN3.5-9B-4BIT",
            host: "127.0.0.1",
            port: 7659,
            bearerToken: "secret",
            sessionValidator: validator
        ))
        #expect(runtime.model == profile.id)
        #expect(DraftPostLanguageRuntime(
            profile: ServerModelProfile(id: "flux", modality: "image-gen"),
            selectedAlias: "flux",
            host: "127.0.0.1",
            port: 7659,
            bearerToken: "secret",
            sessionValidator: validator
        ) == nil)
        #expect(DraftPostLanguageRuntime(
            profile: profile,
            selectedAlias: "another-model",
            host: "127.0.0.1",
            port: 7659,
            bearerToken: "secret",
            sessionValidator: validator
        ) == nil)
        #expect(DraftPostLanguageRuntime(
            profile: ServerModelProfile(id: profile.id),
            selectedAlias: profile.id,
            host: "127.0.0.1",
            port: 7659,
            bearerToken: "secret",
            sessionValidator: validator
        ) == nil)
    }

    @MainActor
    @Test("A rotated server session is rejected before the planning request")
    func staleRuntimeSession() async throws {
        let transport = PlannerTransport(response: try Self.response(content: [
            "status": "needs_clarification",
            "purpose": "",
            "audience": "",
            "talking_points": [],
            "tone": "",
            "destination": "",
            "draft": "",
            "clarifying_question": "Which site should I prepare this for?",
        ]))
        let validator: DraftPostLanguageRuntime.SessionValidator = { false }
        let optionalRuntime = DraftPostLanguageRuntime(
            host: "127.0.0.1",
            port: 7659,
            model: "qwen3.5-9b-4bit",
            bearerToken: "secret",
            sessionValidator: validator
        )
        let runtime = try #require(optionalRuntime)
        let planner = runtime.makePlanner(transport: transport)
        await #expect(throws: DraftPostPlanningError.modelUnavailable) {
            _ = try await planner.analyze(
                instruction: "Draft a launch update for X.",
                browserApplication: "Safari",
                destinationHost: "x.com"
            )
        }
        #expect(await transport.requests.isEmpty)
    }

    @MainActor
    @Test("The UI reviews generated text before invoking browser execution")
    func reviewBeforeExecution() async throws {
        let plan = DraftPostPlan(
            purpose: "Launch",
            audience: "Developers",
            talkingPoints: ["Local"],
            tone: "Concise",
            destination: "x.com",
            draft: "Original draft"
        )
        let planner = ScriptedInstructionPlanner(result: .ready(plan))
        let driver = RecordingPreparedDraftDriver()
        let viewModel = DraftPostInstructionFlowViewModel(
            catalog: InstructionWindowCatalog(options: [Self.destination]),
            planner: planner,
            destinationInspector: InstructionDestinationInspector(),
            driver: driver
        )
        await viewModel.load()
        viewModel.destinationID = Self.destination.id
        viewModel.instruction = "Launch Rapid for developers on X."
        viewModel.analyze()
        await Self.waitUntil { viewModel.phase == .reviewing }
        #expect(await driver.drafts.isEmpty)
        viewModel.editableDraft = "User-edited draft"
        viewModel.execute()
        await Self.waitUntil {
            if case .readyForReview = viewModel.phase { return true }
            return false
        }
        #expect(await driver.drafts == ["User-edited draft"])
        #expect(await driver.destinations == [Self.destinationIdentity])
    }

    @MainActor
    @Test("A clarification returns to the editable request without execution")
    func clarificationState() async throws {
        let planner = ScriptedInstructionPlanner(
            result: .needsClarification("Who is the audience?")
        )
        let driver = RecordingPreparedDraftDriver()
        let viewModel = DraftPostInstructionFlowViewModel(
            catalog: InstructionWindowCatalog(options: [Self.destination]),
            planner: planner,
            destinationInspector: InstructionDestinationInspector(),
            driver: driver
        )
        await viewModel.load()
        viewModel.destinationID = Self.destination.id
        viewModel.instruction = "Write a launch post for X."
        viewModel.analyze()
        await Self.waitUntil { viewModel.clarificationQuestion != nil }
        #expect(viewModel.phase == .ready)
        #expect(viewModel.clarificationQuestion == "Who is the audience?")
        #expect(await driver.drafts.isEmpty)
        await viewModel.load()
        #expect(viewModel.clarificationQuestion == nil)
    }

    @Test("Prepared execution retries only recoverable pre-write failures")
    func boundedPreparedRecovery() async {
        let driver = ScriptedPreparedDraftDriver(
            outcomes: [.failure(.targetUnavailable), .success(())]
        )
        let outcome = await PreparedDraftPostFlowCoordinator(driver: driver).run(
            draft: "Reviewed draft",
            destination: Self.destination,
            expectedDestination: Self.destinationIdentity
        )
        #expect(outcome == .readyForReview(DraftPostFlowMetrics(
            attempts: 2,
            automaticRecoveries: 1,
            completedSteps: 3
        )))
        #expect(await driver.attempts == 2)

        let terminalDriver = ScriptedPreparedDraftDriver(
            outcomes: [.failure(.verificationFailed), .success(())]
        )
        let terminalOutcome = await PreparedDraftPostFlowCoordinator(
            driver: terminalDriver
        ).run(
            draft: "Reviewed draft",
            destination: Self.destination,
            expectedDestination: Self.destinationIdentity
        )
        #expect(terminalOutcome == .failed(
            .verificationFailed,
            DraftPostFlowMetrics(attempts: 1)
        ))
        #expect(await terminalDriver.attempts == 1)
        #expect(DraftPostFlowFailure.composerNotEmpty.permitsReviewedRetry)
        #expect(DraftPostFlowFailure.destinationMismatch.permitsReviewedRetry)
        #expect(!DraftPostFlowFailure.writeRejected.permitsReviewedRetry)
        #expect(!DraftPostFlowFailure.verificationFailed.permitsReviewedRetry)
        #expect(!DraftPostFlowFailure.dependencyFailure.permitsReviewedRetry)
    }

    @MainActor
    @Test("Stopping after a possible browser write reports the definitive result")
    func latePreparedCancellationIsHonest() async {
        let plan = DraftPostPlan(
            purpose: "Launch",
            audience: "Developers",
            talkingPoints: ["Local"],
            tone: "Concise",
            destination: "x.com",
            draft: "Reviewed draft"
        )
        let driver = CancellationIgnoringPreparedDraftDriver()
        let viewModel = DraftPostInstructionFlowViewModel(
            catalog: InstructionWindowCatalog(options: [Self.destination]),
            planner: ScriptedInstructionPlanner(result: .ready(plan)),
            destinationInspector: InstructionDestinationInspector(),
            driver: driver
        )
        await viewModel.load()
        viewModel.destinationID = Self.destination.id
        viewModel.instruction = "Draft a launch update for X."
        viewModel.analyze()
        await Self.waitUntil { viewModel.phase == .reviewing }
        viewModel.execute()
        while !(await driver.didStart) {
            await Task.yield()
        }
        viewModel.stop()
        #expect(viewModel.phase == .stopping)
        await driver.complete()
        await Self.waitUntil {
            if case .readyForReview = viewModel.phase { return true }
            return false
        }
        guard case .readyForReview = viewModel.phase else {
            Issue.record("A possibly completed browser mutation was reported as cancelled")
            return
        }
    }

    @Test("Browser addresses normalize to a stable hostname")
    func destinationHostNormalization() {
        #expect(MacOSDraftPostFlowDriver.normalizedDestinationHost(
            from: "https://X.com/compose/post?draft=1"
        ) == "x.com")
        #expect(MacOSDraftPostFlowDriver.normalizedDestinationHost(
            from: "mail.example.com/inbox"
        ) == "mail.example.com")
        #expect(MacOSDraftPostFlowDriver.normalizedDocumentIdentity(
            from: "HTTPS://X.COM:443/compose/post?draft=1"
        ) == "https://x.com/compose/post?draft=1")
        let expected = ComputerUseBrowserDestinationIdentity(
            host: "x.com",
            documentIdentity: "https://x.com/compose/post"
        )
        #expect(MacOSDraftPostFlowDriver.browserDestinationMatches(
            currentAddress: "HTTPS://X.COM:443/compose/post",
            expected: expected
        ))
        #expect(!MacOSDraftPostFlowDriver.browserDestinationMatches(
            currentAddress: "https://x.com/another-account/compose",
            expected: expected
        ))
        #expect(MacOSDraftPostFlowDriver.normalizedDestinationHost(from: "   ") == nil)
    }

    private static func planner(
        transport: PlannerTransport
    ) -> LocalDraftPostInstructionPlanner {
        LocalDraftPostInstructionPlanner(
            baseURL: URL(string: "http://127.0.0.1:7659/v1")!,
            model: "qwen3.5-9b-4bit",
            bearerToken: "secret",
            transport: transport
        )
    }

    private static func response(
        content: [String: Any]
    ) throws -> LocalComputerUseGroundingHTTPResponse {
        let contentData = try JSONSerialization.data(withJSONObject: content)
        let contentString = try #require(String(data: contentData, encoding: .utf8))
        let envelopeData = try JSONSerialization.data(withJSONObject: [
            "choices": [["message": ["content": contentString]]],
        ])
        return LocalComputerUseGroundingHTTPResponse(
            statusCode: 200,
            contentType: "application/json; charset=utf-8",
            body: envelopeData
        )
    }

    @MainActor
    private static func waitUntil(
        _ predicate: @escaping @MainActor () -> Bool
    ) async {
        let deadline = ContinuousClock.now + .seconds(2)
        while !predicate() {
            guard ContinuousClock.now < deadline else {
                Issue.record("Timed out waiting for the expected flow state")
                return
            }
            try? await Task.sleep(for: .milliseconds(10))
        }
    }

    private static let destination = ComputerUseWindowOption(
        id: "chrome:42",
        applicationName: "Google Chrome",
        windowTitle: "X / Home",
        selection: ComputerUseWindowSelection(
            bundleIdentifier: "com.google.Chrome",
            processIdentifier: 123,
            processLaunchDate: Date(timeIntervalSince1970: 1_700_000_000),
            windowID: 42
        )
    )

    private static let destinationIdentity = ComputerUseBrowserDestinationIdentity(
        host: "x.com",
        documentIdentity: "https://x.com/compose/post"
    )
}

private actor PlannerTransport: LocalComputerUseGroundingTransport {
    let response: LocalComputerUseGroundingHTTPResponse
    private(set) var requests: [URLRequest] = []

    init(response: LocalComputerUseGroundingHTTPResponse) {
        self.response = response
    }

    func send(
        _ request: URLRequest,
        maximumResponseBytes _: Int
    ) async throws -> LocalComputerUseGroundingHTTPResponse {
        requests.append(request)
        return response
    }
}

private struct InstructionWindowCatalog: ComputerUseWindowListing {
    let options: [ComputerUseWindowOption]

    func windows() async throws -> [ComputerUseWindowOption] {
        options
    }
}

private struct InstructionDestinationInspector: ComputerUseBrowserDestinationInspecting {
    func destinationIdentity(
        for _: ComputerUseWindowOption
    ) async throws -> ComputerUseBrowserDestinationIdentity {
        ComputerUseBrowserDestinationIdentity(
            host: "x.com",
            documentIdentity: "https://x.com/compose/post"
        )
    }
}

private actor ScriptedInstructionPlanner: DraftPostInstructionPlanning {
    let result: DraftPostPlanningResult

    init(result: DraftPostPlanningResult) {
        self.result = result
    }

    func analyze(
        instruction _: String,
        browserApplication _: String,
        destinationHost _: String
    ) async throws -> DraftPostPlanningResult {
        result
    }
}

private actor RecordingPreparedDraftDriver: PreparedDraftPostFlowDriving {
    private(set) var drafts: [String] = []
    private(set) var destinations: [ComputerUseBrowserDestinationIdentity] = []

    func transferPreparedDraft(
        _ draft: String,
        to _: ComputerUseWindowOption,
        expectedDestination: ComputerUseBrowserDestinationIdentity
    ) async throws {
        drafts.append(draft)
        destinations.append(expectedDestination)
    }
}

private actor ScriptedPreparedDraftDriver: PreparedDraftPostFlowDriving {
    private var outcomes: [Result<Void, DraftPostFlowFailure>]
    private(set) var attempts = 0

    init(outcomes: [Result<Void, DraftPostFlowFailure>]) {
        self.outcomes = outcomes
    }

    func transferPreparedDraft(
        _ draft: String,
        to _: ComputerUseWindowOption,
        expectedDestination _: ComputerUseBrowserDestinationIdentity
    ) async throws {
        attempts += 1
        guard !draft.isEmpty, !outcomes.isEmpty else {
            throw DraftPostFlowFailure.dependencyFailure
        }
        try outcomes.removeFirst().get()
    }
}

private actor CancellationIgnoringPreparedDraftDriver: PreparedDraftPostFlowDriving {
    private(set) var didStart = false
    private var continuation: CheckedContinuation<Void, Never>?

    func transferPreparedDraft(
        _: String,
        to _: ComputerUseWindowOption,
        expectedDestination _: ComputerUseBrowserDestinationIdentity
    ) async throws {
        didStart = true
        await withCheckedContinuation { continuation = $0 }
    }

    func complete() {
        continuation?.resume()
        continuation = nil
    }
}
