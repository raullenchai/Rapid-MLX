import Foundation
import Testing
@testable import Rapid

/// Contract tests for #2306: driving the Desktop's role-aware conflict UX off
/// the typed `insufficient_capacity_error` 507 envelope (#2305), and the
/// cross-surface #2383 fix that retiring/failed snapshots are never treated as
/// available/serving.
@Suite("Role-aware residency conflict (#2306)")
struct ResidentRoleConflictTests {
    // MARK: - A. Cross-surface serving-state filter (#2383)

    @Test(
        "isServingState fails closed: only resident is serving",
        arguments: [
            ("resident", true),
            ("evicting", false),
            ("retiring", false),
            ("failed", false),
            ("loading", false),
            ("unknown-future-state", false),
        ] as [(String, Bool)]
    )
    func servingStateTruth(state: String, expected: Bool) {
        #expect(isServingState(state) == expected)
    }

    @Test("The residency snapshot never reports a retiring model as present")
    func snapshotTruthfulness() {
        let snapshot = ModelResidencySnapshot(
            memoryLimitBytes: 1,
            memoryUsedBytes: 1,
            memoryAvailableBytes: nil,
            idleTTLSeconds: 60,
            loadsTotal: 0,
            evictionsTotal: 0,
            models: [
                model(id: "assistant-1", alias: "qwen3.8-27b-4bit", modality: "text", state: "retiring", primary: true),
                model(id: "speech-out", alias: "tts-engine", modality: "audio", state: "failed", primary: false),
                model(id: "stt", alias: "whisper-small", modality: "audio", state: "resident", primary: false),
            ]
        )

        // The retiring assistant and failed speech-output engines must not be
        // treated as serving; only the resident STT lane is.
        #expect(!snapshot.contains("qwen3.8-27b-4bit"))
        #expect(!snapshot.contains("tts-engine"))
        #expect(snapshot.contains("whisper-small"))
        #expect(snapshot.modality(for: "qwen3.8-27b-4bit") == nil)
        #expect(snapshot.modality(for: "whisper-small") == "audio")
        #expect(snapshot.activeRequests(for: "qwen3.8-27b-4bit") == nil)
    }

    @Test("preferredTextAlias skips retiring text models")
    func preferredTextSkipsRetiring() {
        let snapshot = ModelResidencySnapshot(
            memoryLimitBytes: 1,
            memoryUsedBytes: 1,
            memoryAvailableBytes: nil,
            idleTTLSeconds: 60,
            loadsTotal: 0,
            evictionsTotal: 0,
            models: [
                model(id: "a", alias: "qwen3.8-27b-4bit", modality: "text", state: "retiring", primary: true),
                model(id: "b", alias: "qwen3.5-4b-4bit", modality: "text", state: "resident", primary: false),
            ]
        )
        #expect(snapshot.preferredTextAlias(fallback: "fallback") == "qwen3.5-4b-4bit")
    }

    @Test("Model switch risk ignores a retiring current model")
    func switchRiskIgnoresRetiring() {
        let snapshot = ModelResidencySnapshot(
            memoryLimitBytes: 1,
            memoryUsedBytes: 1,
            memoryAvailableBytes: nil,
            idleTTLSeconds: 60,
            loadsTotal: 0,
            evictionsTotal: 0,
            models: [
                model(id: "a", alias: "old", modality: "text", state: "retiring", primary: true, activeRequests: 5),
            ]
        )
        #expect(ModelSwitchRisk.evaluate(
            currentAlias: "old",
            targetAlias: "new",
            residency: snapshot
        ) == nil)
    }

    // MARK: - B. Typed 507 role-capacity envelope decode

    @Test("A speech-input 507 conflict decodes the requested role + resident roles + recovery actions")
    func decodesSpeechInputConflict() async {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [RoleConflictStubProtocol.self]
        var client = ServerResidencyClient()
        client.session = URLSession(configuration: configuration)

        let result = await client.load(
            alias: "whisper-large-v3",
            hfPath: nil,
            estimatedSizeGB: 3,
            port: 8000,
            bearer: nil
        )

        guard case .conflicted(let conflict) = result else {
            Issue.record("Expected a typed role-capacity conflict, got \(result)")
            return
        }
        #expect(conflict.requestedRole == "speech-input")
        #expect(conflict.reason == "role_capacity_speech_input")
        #expect(conflict.requestedBytes == 3_221_225_472)
        #expect(conflict.residentRoles.count == 2)

        let assistant = conflict.residentRoles.first { $0.role == "assistant" }
        let speechOutput = conflict.residentRoles.first { $0.role == "speech-output" }
        #expect(assistant?.modelID == "mlx-community/Qwen3.5-4B-MLX-4bit")
        #expect(assistant?.reservedBytes == 5_368_709_120)
        #expect(assistant?.state == "resident")
        #expect(speechOutput?.modelID == "mlx-community/tts-qwen3")
        #expect(speechOutput?.reservedBytes == 1_610_612_736)

        // The speech-input role's valid recovery actions are surfaced verbatim.
        #expect(conflict.recoveryActions.contains("select_smaller_speech_input"))
        #expect(conflict.recoveryActions.contains("stop_speech_output"))
        #expect(conflict.recoveryActions.contains("unload_assistant"))
        #expect(conflict.scopedRecoveryActions.contains(.selectSmallerSpeechInput))
        #expect(conflict.scopedRecoveryActions.contains(.stopSpeechOutput))
        #expect(conflict.scopedRecoveryActions.contains(.unloadAssistant))
    }

    @Test("Unknown recovery_action values never become buttons")
    func unknownRecoveryActionsFiltered() throws {
        // Even when the server declares an action Desktop has never seen, it
        // must not surface as a button — the closed enum is the gate.
        let conflict = ResidentRoleConflict(
            message: "insufficient capacity",
            requestedRole: "speech-input",
            requestedBytes: 1,
            limitBytes: 1,
            usedBytes: 1,
            reason: "role_capacity_speech_input",
            residentRoles: [],
            recoveryActions: [
                "select_smaller_speech_input",
                "stop_speech_output",
                "brand_new_server_action",
                "unload_assistant",
            ]
        )
        #expect(conflict.scopedRecoveryActions == [
            .selectSmallerSpeechInput,
            .stopSpeechOutput,
            .unloadAssistant,
        ])
        // The unknown server action is dropped entirely — never a button.
        #expect(!conflict.scopedRecoveryActions.map(\.rawValue).contains("brand_new_server_action"))
        // And it is not even a known action value.
        #expect(ResidentRecoveryAction(rawValue: "brand_new_server_action") == nil)
    }

    @Test("GiB presentation formats user-readable quantities")
    func gibFormatting() throws {
        let conflict = ResidentRoleConflict(
            message: "no room",
            requestedRole: "speech-input",
            requestedBytes: 3_221_225_472,
            limitBytes: 34_359_738_368,
            usedBytes: 30_000_000_000,
            reason: "x",
            residentRoles: [],
            recoveryActions: []
        )
        #expect(conflict.requestedGib != nil)
        #expect(abs((conflict.requestedGib ?? 0) - 3.0) < 0.01)
        #expect(abs((conflict.limitGib ?? 0) - 32.0) < 0.01)

        // Roles with no reported bytes must not present a fabricated measure.
        let noBytes = ResidentRoleStatus(role: "assistant", modelID: "x", reservedBytes: nil, state: "resident")
        #expect(noBytes.reservedGib == nil)
    }

    @Test("An x-role with nil fields decodes defensively and stays a plain rejection")
    func nilRoleFieldsSafe() throws {
        // The typed envelope with every role field null/absent must not crash;
        // with no role identity it degrades to a plain rejection (additive-safe).
        let json = Data(
            #"""
            {"error":{
              "message":"insufficient capacity",
              "type":"insufficient_capacity_error",
              "code":"insufficient_capacity_error",
              "param":"model",
              "requested_bytes": null,
              "requested_role": null,
              "resident_roles": [],
              "recovery_actions": []
            }}
            """#.utf8
        )
        let envelope = try JSONDecoder().decode(
            ServerResidencyClient_errorEnvelopeStub.self,
            from: json
        )
        #expect(envelope.error?.requestedRole == nil)
        #expect((envelope.error?.residentRoles ?? []) == [])
        #expect(envelope.error?.recoveryActions == [])
    }

    @Test("Unknown/new envelope fields never crash the decode")
    func unknownEnvelopeFieldsSafe() throws {
        // Forward/backward compatibility: a server that adds a new sibling
        // field (or a new resident_roles key) must not crash the Desktop.
        let json = Data(
            #"""
            {"error":{
              "message":"insufficient capacity",
              "type":"insufficient_capacity_error",
              "code":"insufficient_capacity_error",
              "param":"model",
              "reason":"role_capacity_speech_input",
              "requested_role":"speech-input",
              "requested_bytes": 3221225472,
              "limit_bytes": 34359738368,
              "used_bytes": 30100000000,
              "resident_roles":[
                {"role":"assistant","model_id":"qwen","reserved_bytes":5368709120,"state":"resident","future_field":123}
              ],
              "recovery_actions":["stop_speech_output"],
              "some_brand_new_field":{"nested":[1,2,3]}
            },
            "another_new_top_level_field": "ignored"
            }
            """#.utf8
        )
        let envelope = try JSONDecoder().decode(
            ServerResidencyClient_errorEnvelopeStub.self,
            from: json
        )
        #expect(envelope.error?.requestedRole == "speech-input")
        #expect(envelope.error?.recoveryActions == ["stop_speech_output"])
        #expect(envelope.error?.residentRoles?.first?.role == "assistant")
    }

    // MARK: - Role matrix (speech input → LLM/VLM → speech output)

    @Test("Speech-input conflict surfaces assistant + speech-output residents (full matrix)")
    func roleMatrixConflict() async {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [RoleMatrixStubProtocol.self]
        var client = ServerResidencyClient()
        client.session = URLSession(configuration: configuration)

        let result = await client.load(
            alias: "whisper-large-v3",
            hfPath: nil,
            estimatedSizeGB: 3,
            port: 8000,
            bearer: nil
        )

        guard case .conflicted(let conflict) = result else {
            Issue.record("Expected role-matrix conflict, got \(result)")
            return
        }
        // The matrix: speech-input (requested) conflicts with resident
        // assistant/conversation + speech-output roles.
        #expect(conflict.requestedRole == "speech-input")
        let roles = Set(conflict.residentRoles.map(\.role))
        #expect(roles == ["assistant", "speech-output"])
        // Recovery actions come from the requested (speech-input) role.
        #expect(conflict.scopedRecoveryActions.contains(.selectSmallerSpeechInput))
        #expect(conflict.scopedRecoveryActions.contains(.unloadAssistant))
    }

    @Test("A plain 507 without role fields remains a legacy rejection (#2305 additive-safe)")
    func plainRejectionPreserved() async {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [Plain507StubProtocol.self]
        var client = ServerResidencyClient()
        client.session = URLSession(configuration: configuration)

        let result = await client.load(
            alias: "not-a-role-model",
            hfPath: nil,
            estimatedSizeGB: 1,
            port: 8000,
            bearer: nil
        )
        guard case .rejected(let message) = result else {
            Issue.record("Expected a plain rejection, got \(result)")
            return
        }
        // The server's message is preserved verbatim; critically it must NOT
        // be misclassified as a typed `.conflicted` role conflict.
        #expect(message.contains("memory ceiling exceeded"))
    }

    // MARK: - Helpers

    private func model(
        id: String,
        alias: String,
        modality: String,
        state: String,
        primary: Bool,
        activeRequests: Int = 0
    ) -> ResidentModelStatus {
        ResidentModelStatus(
            id: id,
            modelPath: "test/\(id)",
            aliases: [alias],
            modality: modality,
            state: state,
            pinned: false,
            primary: primary,
            activeRequests: activeRequests,
            estimatedBytes: 1,
            measuredBytes: nil,
            idleSeconds: 0
        )
    }
}

// MARK: - URLProtocol stubs

/// A typed 507 `insufficient_capacity_error` for a speech-input admission
/// conflicting with a resident assistant + speech-output role.
private final class RoleConflictStubProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}

    override func startLoading() {
        let payload = Data(
            #"""
            {"detail":{"error":{
              "message":"insufficient capacity for role 'speech-input': requested=3.00 GiB, used=8.00 GiB, limit=32.00 GiB; no idle unpinned model is eligible for eviction",
              "type":"insufficient_capacity_error",
              "code":"insufficient_capacity_error",
              "reason":"role_capacity_speech_input",
              "param":"model",
              "requested_bytes": 3221225472,
              "limit_bytes": 34359738368,
              "used_bytes": 8589934592,
              "requested_role": "speech-input",
              "resident_roles": [
                {"role":"assistant","model_id":"mlx-community/Qwen3.5-4B-MLX-4bit","reserved_bytes":5368709120,"state":"resident"},
                {"role":"speech-output","model_id":"mlx-community/tts-qwen3","reserved_bytes":1610612736,"state":"resident"}
              ],
              "recovery_actions":["select_smaller_speech_input","stop_speech_output","unload_assistant"]
            }}}
            """#.utf8
        )
        let response = HTTPURLResponse(
            url: request.url!, statusCode: 507, httpVersion: "HTTP/1.1", headerFields: nil
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: payload)
        client?.urlProtocolDidFinishLoading(self)
    }
}

/// The full voice conversation matrix: a speech-input admission conflicting
/// with resident assistant + speech-output roles.
private final class RoleMatrixStubProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}

    override func startLoading() {
        let payload = Data(
            #"""
            {"detail":{"error":{
              "message":"speech input cannot coexist with the resident conversation and speech output",
              "type":"insufficient_capacity_error",
              "code":"insufficient_capacity_error",
              "reason":"role_capacity_speech_input",
              "param":"model",
              "requested_bytes": 3221225472,
              "limit_bytes": 34359738368,
              "used_bytes": 16106127360,
              "requested_role": "speech-input",
              "resident_roles": [
                {"role":"assistant","model_id":"mlx-community/Qwen3.5-4B-MLX-4bit","reserved_bytes":6442450944,"state":"resident"},
                {"role":"speech-output","model_id":"mlx-community/tts-qwen3","reserved_bytes":1610612736,"state":"resident"}
              ],
              "recovery_actions":["select_smaller_speech_input","unload_assistant"]
            }}}
            """#.utf8
        )
        let response = HTTPURLResponse(
            url: request.url!, statusCode: 507, httpVersion: "HTTP/1.1", headerFields: nil
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: payload)
        client?.urlProtocolDidFinishLoading(self)
    }
}

/// A plain 507 with no typed role fields — must stay a legacy rejection.
private final class Plain507StubProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}

    override func startLoading() {
        let payload = Data(
            #"""
            {"detail":{"error":{
              "message":"resident model memory ceiling exceeded: usage=8.00 GiB, incoming=1.00 GiB, limit=32.00 GiB; no idle unpinned model is eligible for eviction",
              "type":"insufficient_capacity_error",
              "code":"insufficient_capacity_error",
              "param":"estimated_size_gb"
            }}}
            """#.utf8
        )
        let response = HTTPURLResponse(
            url: request.url!, statusCode: 507, httpVersion: "HTTP/1.1", headerFields: nil
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: payload)
        client?.urlProtocolDidFinishLoading(self)
    }
}

/// A decodable mirror of the envelope's internal `Error` shape so defensive
/// decoding of null/unknown fields can be asserted without network I/O.
struct ServerResidencyClient_errorEnvelopeStub: Decodable {
    struct Error: Decodable {
        struct RoleStatusStub: Decodable, Equatable {
            let role: String?
            let modelID: String?
            let reservedBytes: UInt64?
            let state: String?
            enum CodingKeys: String, CodingKey {
                case role
                case modelID = "model_id"
                case reservedBytes = "reserved_bytes"
                case state
            }
        }
        let message: String?
        let requestedRole: String?
        let requestedBytes: UInt64?
        let limitBytes: UInt64?
        let usedBytes: UInt64?
        let reason: String?
        let residentRoles: [RoleStatusStub]?
        let recoveryActions: [String]?
        enum CodingKeys: String, CodingKey {
            case message
            case requestedRole = "requested_role"
            case requestedBytes = "requested_bytes"
            case limitBytes = "limit_bytes"
            case usedBytes = "used_bytes"
            case reason
            case residentRoles = "resident_roles"
            case recoveryActions = "recovery_actions"
        }
    }
    let error: Error?
}
