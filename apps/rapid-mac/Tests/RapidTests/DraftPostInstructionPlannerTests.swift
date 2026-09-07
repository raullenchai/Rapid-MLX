import Foundation
import Testing
@testable import Rapid

@Suite("Computer Use natural-language draft planning")
struct DraftPostInstructionPlannerTests {
    @Test("A ready response becomes a bounded review plan")
    func readyPlan() async throws {
        let transport = PlannerTransport(response: try Self.response(content: [
            "status": "ready",
            "purpose": "launch post",
            "audience": "Mac developers",
            "talking_points": ["faster local inference", "no cloud upload"],
            "tone": "Concise and enthusiastic",
            "destination": "x.com",
            "purpose_evidence": "launch post",
            "audience_evidence": "Mac developers",
            "talking_points_evidence": ["faster local inference", "no cloud upload"],
            "tone_evidence": "Concise and enthusiastic",
            "destination_evidence": "X",
            "clarifying_question": "",
        ]))
        let result = try await Self.planner(transport: transport).analyze(
            instruction: "Write a Concise and enthusiastic launch post for X for Mac developers about faster local inference and no cloud upload.",
            browserApplication: "Google Chrome",
            destinationHost: "x.com"
        )
        #expect(result == .ready(DraftPostPlan(
            purpose: "launch post",
            audience: "Mac developers",
            talkingPoints: ["faster local inference", "no cloud upload"],
            tone: "Concise and enthusiastic",
            destination: "x.com",
            draft: "For Mac developers: launch post — faster local inference; no cloud upload!"
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
        let schema = try #require(schemaEnvelope["schema"] as? [String: Any])
        let properties = try #require(schema["properties"] as? [String: Any])
        let clarification = try #require(
            properties["clarifying_question"] as? [String: Any]
        )
        #expect(clarification["pattern"] as? String == #"^(?:[^?？؟\r\n]*[?？؟])?$"#)
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
            "purpose_evidence": "",
            "audience_evidence": "",
            "talking_points_evidence": [],
            "tone_evidence": "",
            "destination_evidence": "",
            "clarifying_question": "Which site should I prepare this for?",
        ]))
        let result = try await Self.planner(transport: transport).analyze(
            instruction: "Write something about the release.",
            browserApplication: "Safari",
            destinationHost: "x.com"
        )
        #expect(result == .needsClarification("Which site should I prepare this for?"))
    }

    @Test("A clarification response cannot smuggle an unreviewed partial plan")
    func clarificationRejectsPartialPlan() async throws {
        let transport = PlannerTransport(response: try Self.response(content: [
            "status": "needs_clarification",
            "purpose": "Launch Rapid",
            "audience": "",
            "talking_points": [],
            "tone": "",
            "destination": "",
            "purpose_evidence": "",
            "audience_evidence": "",
            "talking_points_evidence": [],
            "tone_evidence": "",
            "destination_evidence": "",
            "clarifying_question": "Which site should I prepare this for?",
        ]))
        await #expect(throws: DraftPostPlanningError.invalidResponse) {
            _ = try await Self.planner(transport: transport).analyze(
                instruction: "Write something about the release.",
                browserApplication: "Safari",
                destinationHost: "x.com"
            )
        }
    }

    @Test("The planning request requires clarification for incomplete or mismatched intent")
    func clarificationPromptContract() async throws {
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
        #expect(system.contains("ending with one question mark"))
        #expect(system.contains("For ready, clarifying_question must be an empty string"))
        #expect(system.contains("purpose_evidence"))
        let user = try #require(messages.last?["content"])
        #expect(user.contains("Trusted destination hostname: x.com"))
        #expect(user.contains("Write something about the release."))

        let fabricatedReady = PlannerTransport(response: try Self.response(content: [
            "status": "ready",
            "purpose": "Announce a major launch",
            "audience": "Developers",
            "talking_points": ["Breakthrough performance"],
            "tone": "Excited",
            "destination": "x.com",
            "purpose_evidence": "Write something",
            "audience_evidence": "Developers",
            "talking_points_evidence": ["breakthrough performance"],
            "tone_evidence": "Excited",
            "destination_evidence": "X",
            "clarifying_question": "",
        ]))
        let fabricatedResult = try await Self.planner(
            transport: fabricatedReady
        ).analyze(
            instruction: "Write something about the release.",
            browserApplication: "Safari",
            destinationHost: "x.com"
        )
        #expect(fabricatedResult == .needsClarification(
            "What should this update accomplish, which points should it include, and where should it be posted?"
        ))

        let reusedEvidence = PlannerTransport(response: try Self.response(content: [
            "status": "ready",
            "purpose": "launch post",
            "audience": "Developers",
            "talking_points": ["launch post"],
            "tone": "Concise",
            "destination": "x.com",
            "purpose_evidence": "launch post",
            "audience_evidence": "Developers",
            "talking_points_evidence": ["launch post"],
            "tone_evidence": "Concise",
            "destination_evidence": "X",
            "clarifying_question": "",
        ]))
        let reusedResult = try await Self.planner(transport: reusedEvidence).analyze(
            instruction: "Write a launch post for X.",
            browserApplication: "Safari",
            destinationHost: "x.com"
        )
        #expect(reusedResult == .needsClarification(
            "What should this update accomplish, which points should it include, and where should it be posted?"
        ))

        let substringDestination = PlannerTransport(response: try Self.response(content: [
            "status": "ready",
            "purpose": "Write text",
            "audience": "Developers",
            "talking_points": ["about the release"],
            "tone": "Concise",
            "destination": "x.com",
            "purpose_evidence": "Write text",
            "audience_evidence": "Developers",
            "talking_points_evidence": ["about the release"],
            "tone_evidence": "Concise",
            "destination_evidence": "X",
            "clarifying_question": "",
        ]))
        let substringResult = try await Self.planner(
            transport: substringDestination
        ).analyze(
            instruction: "Write text about the release.",
            browserApplication: "Safari",
            destinationHost: "x.com"
        )
        #expect(substringResult == .needsClarification(
            "What should this update accomplish, which points should it include, and where should it be posted?"
        ))
    }

    @Test("Audience and tone must also come from the user's brief")
    func audienceAndToneEvidence() async throws {
        let transport = PlannerTransport(response: try Self.response(content: [
            "status": "ready",
            "purpose": "launch post",
            "audience": "Enterprise buyers",
            "talking_points": ["local inference"],
            "tone": "Urgent",
            "destination": "x.com",
            "purpose_evidence": "launch post",
            "audience_evidence": "Enterprise buyers",
            "talking_points_evidence": ["local inference"],
            "tone_evidence": "Urgent",
            "destination_evidence": "X",
            "clarifying_question": "",
        ]))
        let result = try await Self.planner(transport: transport).analyze(
            instruction: "Write a launch post for X about local inference.",
            browserApplication: "Safari",
            destinationHost: "x.com"
        )
        #expect(result == .needsClarification(
            "What should this update accomplish, which points should it include, and where should it be posted?"
        ))
    }

    @Test("A service named in the brief matches its recognized browser subdomain")
    func destinationServiceSubdomain() async throws {
        let transport = PlannerTransport(response: try Self.response(content: [
            "status": "ready",
            "purpose": "launch post",
            "audience": "developers",
            "talking_points": ["local inference"],
            "tone": "concise",
            "destination": "mobile.twitter.com",
            "purpose_evidence": "launch post",
            "audience_evidence": "developers",
            "talking_points_evidence": ["local inference"],
            "tone_evidence": "concise",
            "destination_evidence": "Twitter",
            "clarifying_question": "",
        ]))
        let result = try await Self.planner(transport: transport).analyze(
            instruction: "Write a concise launch post for developers on Twitter about local inference.",
            browserApplication: "Google Chrome",
            destinationHost: "mobile.twitter.com"
        )
        #expect(result == .ready(DraftPostPlan(
            purpose: "launch post",
            audience: "developers",
            talkingPoints: ["local inference"],
            tone: "concise",
            destination: "mobile.twitter.com",
            draft: "For developers: launch post — local inference."
        )))
    }

    @Test("A known service name matches only its explicit hostname allowlist")
    func allowlistedDestinationServiceName() async throws {
        func output(destination: String) -> [String: Any] {
            [
                "status": "ready",
                "purpose": "release update",
                "audience": "contributors",
                "talking_points": ["local inference"],
                "tone": "concise",
                "destination": destination,
                "purpose_evidence": "release update",
                "audience_evidence": "contributors",
                "talking_points_evidence": ["local inference"],
                "tone_evidence": "concise",
                "destination_evidence": "GitHub",
                "clarifying_question": "",
            ]
        }
        let brief = "Write a concise release update for contributors on GitHub about local inference."
        let github = try await Self.planner(transport: PlannerTransport(
            response: try Self.response(content: output(destination: "github.com"))
        )).analyze(
            instruction: brief,
            browserApplication: "Safari",
            destinationHost: "github.com"
        )
        guard case .ready = github else {
            Issue.record("An explicit generic service did not match its verified host")
            return
        }

        let deceptiveSubdomain = try await Self.planner(transport: PlannerTransport(
            response: try Self.response(content: output(destination: "github.evil.com"))
        )).analyze(
            instruction: brief,
            browserApplication: "Safari",
            destinationHost: "github.evil.com"
        )
        #expect(deceptiveSubdomain == .needsClarification(
            "What should this update accomplish, which points should it include, and where should it be posted?"
        ))

        let privateSuffix = try await Self.planner(transport: PlannerTransport(
            response: try Self.response(content: output(destination: "attacker.github.io"))
        )).analyze(
            instruction: brief,
            browserApplication: "Safari",
            destinationHost: "attacker.github.io"
        )
        #expect(privateSuffix == .needsClarification(
            "What should this update accomplish, which points should it include, and where should it be posted?"
        ))

        let deceptiveTopLevelDomain = try await Self.planner(transport: PlannerTransport(
            response: try Self.response(content: output(destination: "github.zip"))
        )).analyze(
            instruction: brief,
            browserApplication: "Safari",
            destinationHost: "github.zip"
        )
        #expect(deceptiveTopLevelDomain == .needsClarification(
            "What should this update accomplish, which points should it include, and where should it be posted?"
        ))
    }

    @Test("Verified audience and tone materially shape the grounded draft")
    func groundedDraftStyle() async throws {
        func output(tone: String) -> [String: Any] {
            [
                "status": "ready",
                "purpose": "announce Rapid",
                "audience": "Mac developers",
                "talking_points": ["local inference"],
                "tone": tone,
                "destination": "x.com",
                "purpose_evidence": "announce Rapid",
                "audience_evidence": "Mac developers",
                "talking_points_evidence": ["local inference"],
                "tone_evidence": tone,
                "destination_evidence": "X",
                "clarifying_question": "",
            ]
        }
        let enthusiastic = try await Self.planner(transport: PlannerTransport(
            response: try Self.response(content: output(tone: "enthusiastic"))
        )).analyze(
            instruction: "For Mac developers, announce Rapid on X with an enthusiastic tone and mention local inference.",
            browserApplication: "Safari",
            destinationHost: "x.com"
        )
        let professional = try await Self.planner(transport: PlannerTransport(
            response: try Self.response(content: output(tone: "professional"))
        )).analyze(
            instruction: "For Mac developers, announce Rapid on X with a professional tone and mention local inference.",
            browserApplication: "Safari",
            destinationHost: "x.com"
        )
        guard case .ready(let enthusiasticPlan) = enthusiastic,
              case .ready(let professionalPlan) = professional
        else {
            Issue.record("Evidence-backed plans did not reach review")
            return
        }
        #expect(enthusiasticPlan.draft == "For Mac developers: announce Rapid. local inference!")
        #expect(professionalPlan.draft == "For Mac developers: announce Rapid. local inference.")
        #expect(enthusiasticPlan.draft != professionalPlan.draft)
    }

    @Test("Grounded rendering normalizes existing terminal punctuation")
    func groundedDraftPunctuation() async throws {
        let transport = PlannerTransport(response: try Self.response(content: [
            "status": "ready",
            "purpose": "announce Rapid.",
            "audience": "Mac developers",
            "talking_points": ["Released today!"],
            "tone": "professional",
            "destination": "x.com",
            "purpose_evidence": "announce Rapid.",
            "audience_evidence": "Mac developers",
            "talking_points_evidence": ["Released today!"],
            "tone_evidence": "professional",
            "destination_evidence": "X",
            "clarifying_question": "",
        ]))
        let result = try await Self.planner(transport: transport).analyze(
            instruction: "For Mac developers, announce Rapid. Use a professional tone on X: Released today!",
            browserApplication: "Safari",
            destinationHost: "x.com"
        )
        guard case .ready(let plan) = result else {
            Issue.record("The punctuated evidence-backed plan did not reach review")
            return
        }
        #expect(plan.draft == "For Mac developers: announce Rapid. Released today.")
        #expect(!plan.draft.contains(".."))
        #expect(!plan.draft.contains("!."))
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
            "purpose_evidence": "launch post",
            "audience_evidence": "Developers",
            "talking_points_evidence": ["Local"],
            "tone_evidence": "Concise",
            "destination_evidence": "X",
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

        let missingEvidence = PlannerTransport(response: try Self.response(content: [
            "status": "ready",
            "purpose": "Launch",
            "audience": "Developers",
            "talking_points": ["Local"],
            "tone": "Concise",
            "destination": "x.com",
            "purpose_evidence": "launch post",
            "audience_evidence": "Developers",
            "talking_points_evidence": ["Local"],
            "destination_evidence": "X",
            "clarifying_question": "",
        ]))
        await #expect(throws: DraftPostPlanningError.invalidResponse) {
            _ = try await Self.planner(transport: missingEvidence).analyze(
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
            "purpose_evidence": "launch post",
            "audience_evidence": "Developers",
            "talking_points_evidence": ["Local"],
            "tone_evidence": "Concise",
            "destination_evidence": "X",
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
            "purpose_evidence": "",
            "audience_evidence": "",
            "talking_points_evidence": [],
            "tone_evidence": "",
            "destination_evidence": "",
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
            "purpose_evidence": "",
            "audience_evidence": "",
            "talking_points_evidence": [],
            "tone_evidence": "",
            "destination_evidence": "",
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
        #expect(runtime.sessionID == nil)
        let sharedSession = UUID()
        let firstIdentity = try #require(DraftPostLanguageRuntime(
            host: "127.0.0.1",
            port: 7659,
            model: "qwen3.5-9b-4bit",
            bearerToken: "secret",
            sessionID: sharedSession,
            sessionValidator: validator
        )).viewIdentity
        let switchedIdentity = try #require(DraftPostLanguageRuntime(
            host: "127.0.0.1",
            port: 7659,
            model: "gemma-4-12b-4bit",
            bearerToken: "secret",
            sessionID: sharedSession,
            sessionValidator: validator
        )).viewIdentity
        #expect(firstIdentity != switchedIdentity)
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
            "purpose_evidence": "",
            "audience_evidence": "",
            "talking_points_evidence": [],
            "tone_evidence": "",
            "destination_evidence": "",
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
    @Test("A retained runtime rejects a restarted server that reuses its connection values")
    func restartedServerSession() async throws {
        let alias = "qwen3.5-9b-4bit"
        let bearer = "persisted-secret"
        let profile = ServerModelProfile(id: alias, modality: "text")
        let server = ServerManager(
            testingState: .ready(alias: alias),
            activePort: 7659,
            activeBearer: bearer
        )
        server.applyActiveModelProfile(profile, forAlias: alias)
        let runtime = try #require(DraftPostLanguageRuntime(
            profile: profile,
            selectedAlias: alias,
            host: server.host,
            port: server.activePort,
            bearerToken: bearer,
            liveServer: server
        ))
        let originalSessionID = try #require(runtime.sessionID)
        let transport = PlannerTransport(response: try Self.response(content: [
            "status": "needs_clarification",
            "purpose": "",
            "audience": "",
            "talking_points": [],
            "tone": "",
            "destination": "",
            "purpose_evidence": "",
            "audience_evidence": "",
            "talking_points_evidence": [],
            "tone_evidence": "",
            "destination_evidence": "",
            "clarifying_question": "Which site should I prepare this for?",
        ]))

        // Simulate a new launch reusing alias, port, and a persisted bearer.
        server._testReplaceActiveServerSession(bearer: bearer)
        #expect(server.activeServerSessionID != originalSessionID)
        server.applyActiveModelProfile(profile, forAlias: alias)

        await #expect(throws: DraftPostPlanningError.modelUnavailable) {
            _ = try await runtime.makePlanner(transport: transport).analyze(
                instruction: "Draft a launch update for X.",
                browserApplication: "Safari",
                destinationHost: "x.com"
            )
        }
        #expect(await transport.requests.isEmpty)
    }

    @MainActor
    @Test("A long-lived embedded credential cannot authorize a private brief")
    func persistentCredentialIsIneligible() throws {
        let alias = "qwen3.5-9b-4bit"
        let profile = ServerModelProfile(id: alias, modality: "text")
        let server = ServerManager(
            testingState: .ready(alias: alias),
            activePort: 7659,
            activeBearer: "persisted-secret"
        )
        server.applyActiveModelProfile(profile, forAlias: alias)
        server.setEmbeddedBearerLifetime(.daily)
        #expect(DraftPostLanguageRuntime(
            profile: profile,
            selectedAlias: alias,
            host: server.host,
            port: server.activePort,
            bearerToken: server.activeBearer,
            liveServer: server
        ) == nil)
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
    @Test("A model-session change preserves the reviewed flow")
    func modelSessionChangePreservesReviewedFlow() async throws {
        let originalPlan = DraftPostPlan(
            purpose: "Launch",
            audience: "Developers",
            talkingPoints: ["Local"],
            tone: "Concise",
            destination: "x.com",
            draft: "Reviewed draft"
        )
        let replacementPlan = DraftPostPlan(
            purpose: "Replacement",
            audience: "Operators",
            talkingPoints: ["Later"],
            tone: "Direct",
            destination: "x.com",
            draft: "Replacement draft"
        )
        let viewModel = DraftPostInstructionFlowViewModel(
            catalog: InstructionWindowCatalog(options: [Self.destination]),
            planner: ScriptedInstructionPlanner(result: .ready(originalPlan)),
            destinationInspector: InstructionDestinationInspector(),
            driver: RecordingPreparedDraftDriver()
        )
        await viewModel.load()
        viewModel.destinationID = Self.destination.id
        viewModel.instruction = "Launch Rapid for developers on X."
        viewModel.analyze()
        await Self.waitUntil { viewModel.phase == .reviewing }

        viewModel.updatePlanner(
            ScriptedInstructionPlanner(result: .ready(replacementPlan))
        )

        #expect(viewModel.phase == .reviewing)
        #expect(viewModel.plan == originalPlan)
        #expect(viewModel.editableDraft == "Reviewed draft")
    }

    @MainActor
    @Test("Navigation during local planning invalidates the plan before review")
    func navigationDuringPlanning() async throws {
        let plan = DraftPostPlan(
            purpose: "Launch",
            audience: "Developers",
            talkingPoints: ["Local"],
            tone: "Concise",
            destination: "x.com",
            draft: "Reviewed draft"
        )
        let inspector = ScriptedDestinationInspector(identities: [
            Self.destinationIdentity,
            ComputerUseBrowserDestinationIdentity(
                host: "x.com",
                documentIdentity: "https://x.com/another-account"
            ),
        ])
        let viewModel = DraftPostInstructionFlowViewModel(
            catalog: InstructionWindowCatalog(options: [Self.destination]),
            planner: ScriptedInstructionPlanner(result: .ready(plan)),
            destinationInspector: inspector,
            driver: RecordingPreparedDraftDriver()
        )
        await viewModel.load()
        viewModel.destinationID = Self.destination.id
        viewModel.instruction = "Launch Rapid for developers on X."
        viewModel.analyze()
        await Self.waitUntil {
            viewModel.phase == .planningFailed(.destinationUnavailable)
        }
        #expect(viewModel.plan == nil)
        #expect(await inspector.inspectionCount == 2)
    }

    @MainActor
    @Test("Permission loss during destination inspection stays actionable")
    func planningPermissionLoss() async throws {
        let viewModel = DraftPostInstructionFlowViewModel(
            catalog: InstructionWindowCatalog(options: [Self.destination]),
            planner: ScriptedInstructionPlanner(result: .needsClarification("Who?")),
            destinationInspector: FailingDestinationInspector(failure: .permissionMissing),
            driver: RecordingPreparedDraftDriver()
        )
        await viewModel.load()
        viewModel.destinationID = Self.destination.id
        viewModel.instruction = "Draft a launch update for X."
        viewModel.analyze()
        await Self.waitUntil { viewModel.phase == .planningFailed(.permissionMissing) }
        #expect(DraftPostPlanningError.permissionMissing.userMessage.contains("Accessibility"))
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
            outcomes: [.failure(.focusChanged), .success(())]
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

        let missingTargetDriver = ScriptedPreparedDraftDriver(
            outcomes: [.failure(.targetUnavailable), .success(())]
        )
        let missingTargetOutcome = await PreparedDraftPostFlowCoordinator(
            driver: missingTargetDriver
        ).run(
            draft: "Reviewed draft",
            destination: Self.destination,
            expectedDestination: Self.destinationIdentity
        )
        #expect(missingTargetOutcome == .failed(
            .targetUnavailable,
            DraftPostFlowMetrics(attempts: 1)
        ))
        #expect(await missingTargetDriver.attempts == 1)

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
        #expect(!DraftPostFlowFailure.targetUnavailable.permitsReviewedRetry)
        #expect(!DraftPostFlowFailure.writeRejected.permitsReviewedRetry)
        #expect(!DraftPostFlowFailure.verificationFailed.permitsReviewedRetry)
        #expect(!DraftPostFlowFailure.cancelled.permitsReviewedRetry)
        #expect(!DraftPostFlowFailure.dependencyFailure.permitsReviewedRetry)
    }

    @MainActor
    @Test("Cancellation after a simulated write cannot return to the reviewed plan")
    func cancellationAfterWriteRequiresStartOver() async throws {
        let plan = DraftPostPlan(
            purpose: "Launch",
            audience: "Developers",
            talkingPoints: ["Local"],
            tone: "Concise",
            destination: "x.com",
            draft: "Reviewed draft"
        )
        let driver = WrittenThenCancelledPreparedDraftDriver()
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
        await Self.waitUntil {
            if case .executionFailed(.cancelled, _) = viewModel.phase { return true }
            return false
        }
        #expect(await driver.didWrite)

        viewModel.returnToPlan()
        guard case .executionFailed(.cancelled, _) = viewModel.phase else {
            Issue.record("An ambiguous cancellation returned to the stale reviewed plan")
            return
        }
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
        let hashRouted = ComputerUseBrowserDestinationIdentity(
            host: "example.com",
            documentIdentity: "https://example.com/app#account-a/compose"
        )
        #expect(!MacOSDraftPostFlowDriver.browserDestinationMatches(
            currentAddress: "https://example.com/app#account-b/compose",
            expected: hashRouted
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

private actor ScriptedDestinationInspector: ComputerUseBrowserDestinationInspecting {
    let identities: [ComputerUseBrowserDestinationIdentity]
    private(set) var inspectionCount = 0

    init(identities: [ComputerUseBrowserDestinationIdentity]) {
        self.identities = identities
    }

    func destinationIdentity(
        for _: ComputerUseWindowOption
    ) async throws -> ComputerUseBrowserDestinationIdentity {
        let index = min(inspectionCount, identities.count - 1)
        inspectionCount += 1
        return identities[index]
    }
}

private struct FailingDestinationInspector: ComputerUseBrowserDestinationInspecting {
    let failure: DraftPostFlowFailure

    func destinationIdentity(
        for _: ComputerUseWindowOption
    ) async throws -> ComputerUseBrowserDestinationIdentity {
        throw failure
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

private actor WrittenThenCancelledPreparedDraftDriver: PreparedDraftPostFlowDriving {
    private(set) var didWrite = false

    func transferPreparedDraft(
        _: String,
        to _: ComputerUseWindowOption,
        expectedDestination _: ComputerUseBrowserDestinationIdentity
    ) async throws {
        didWrite = true
        throw CancellationError()
    }
}
