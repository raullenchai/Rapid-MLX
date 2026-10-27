import Foundation
import Testing
@testable import Rapid

@Suite("Owner wire-v1 boundaries")
struct ChatWireBoundaryTests {
    @Test("Reasoning is transient even before terminal, omitted replay is inert")
    func reasoningPrivacy() throws {
        var reducer = ChatTurnReducer(turnID: "t")
        let live = try ChatWireFixtures.envelope(#"{"type":"reasoning.delta","part_id":"p","text":"PRIVATE_SENTINEL"}"#)
        #expect(try reducer.apply(live) == .applied)
        #expect(reducer.liveReasoning["p"] == "PRIVATE_SENTINEL")
        let settled = String(decoding: try JSONEncoder().encode(reducer.settledProjection), as: UTF8.self)
        #expect(!settled.contains("PRIVATE_SENTINEL"))
        let omitted = try ChatWireFixtures.envelope(#"{"type":"reasoning.delta","part_id":"p","text":"DO_NOT_DISPLAY","omitted":true}"#, seq: 2)
        _ = try reducer.apply(omitted)
        #expect(reducer.liveReasoning["p"] == "PRIVATE_SENTINEL")
        _ = try reducer.apply(ChatWireFixtures.envelope(#"{"type":"turn.terminal","outcome":"completed"}"#, seq: 3))
        #expect(reducer.liveReasoning.isEmpty)
        #expect(!String(decoding: try JSONEncoder().encode(reducer.settledProjection), as: UTF8.self).contains("PRIVATE_SENTINEL"))
    }

    @Test("Unknown events consume sequence; one terminal fences all future changes")
    func futureEventsAndTerminal() throws {
        var reducer = ChatTurnReducer(turnID: "t")
        _ = try reducer.apply(ChatWireFixtures.envelope(#"{"type":"future.optional","anything":{"nested":true}}"#))
        #expect(reducer.lastApplied == 1)
        #expect(reducer.projection.parts.isEmpty)
        let terminal = try ChatWireFixtures.envelope(#"{"type":"turn.terminal","outcome":"cancelled"}"#, seq: 2)
        #expect(try reducer.apply(terminal) == .applied)
        #expect(try reducer.apply(terminal) == .duplicate)
        #expect(try reducer.apply(ChatWireFixtures.envelope(#"{"type":"turn.terminal","outcome":"completed"}"#, seq: 3)) == .afterTerminal)
        #expect(reducer.projection.terminal?.outcome == .cancelled)
        #expect(reducer.lastApplied == 2)
        #expect(throws: ChatWireError.wrongTurn) {
            try reducer.apply(ChatWireFixtures.envelope(#"{"type":"usage"}"#, turn: "other"))
        }
    }

    @Test("Defaults are applied only to missing fields and known schema violations fail")
    func strictDecoding() throws {
        for event in [
            #"{"type":"text.delta","part_id":"p"}"#,
            #"{"type":"text.delta","part_id":"p","text":"x","extra":0}"#,
            #"{"type":"usage","input_tokens":-1}"#,
            #"{"type":"usage","requests":null}"#,
            #"{"type":"usage","requests":true}"#,
            #"{"type":"usage","requests":1.5}"#,
            #"{"type":"turn.terminal","outcome":"interrupted"}"#,
            #"{"type":"tool.result","call_id":"c","result":"x","details":null}"#,
            #"{"type":"tool.result","call_id":"c","result":"x","details_kind":"html"}"#,
            #"{"type":"context","used_tokens":0,"limit_tokens":1,"system_tools_tokens":0,"conversation_tokens":0,"tool_output_tokens":0,"free_tokens":1,"percent":101}"#
        ] {
            #expect(throws: (any Error).self) { try ChatWireFixtures.envelope(event) }
        }
        for raw in [
            #"{"wire_version":2,"turn_id":"t","seq":1,"event":{"type":"usage"}}"#,
            #"{"turn_id":"t","seq":0,"event":{"type":"usage"}}"#,
            #"{"turn_id":"t","seq":1,"event":{"type":"usage"},"extra":true}"#
        ] {
            #expect(throws: (any Error).self) { try JSONDecoder().decode(ChatWireEnvelope.self, from: Data(raw.utf8)) }
        }
        if case .toolResult(let result) = try ChatWireFixtures.envelope(#"{"type":"tool.result","call_id":"c","result":"ok"}"#).event {
            #expect(result.details.isEmpty)
            #expect(result.detailsKind == .text)
            #expect(result.outcome == .success)
            #expect(result.omittedLines == 0)
        } else { Issue.record("Wrong event") }
    }

    @Test("SSE multiline data, BOM, CR-only, comments, unknown fields, and invalid IDs")
    func sseFraming() throws {
        let lines = ["\u{feff}: hello", "", "retry: 50", "extension: ignored", "id: t:1", "event: usage",
                     #"data: {"turn_id":"t","seq":1,"event":{"type":"usage", "#,
                     #"data: "input_tokens":4}}"#, "", ""]
        for newline in ["\n", "\r\n", "\r"] {
            let raw = Data(lines.joined(separator: newline).utf8)
            for split in 0...raw.count {
                var decoder = ChatSSEDecoder()
                let result = try decoder.append(Data(raw.prefix(split))) + decoder.append(Data(raw.dropFirst(split)))
                #expect(result.count == 1)
                #expect(result.first?.event.type == "usage")
                try decoder.finish()
            }
        }
        for header in ["id: other:1\nevent: usage", "id: t:2\nevent: usage", "id: t:1\nevent: context", "event: usage", "id: t:1"] {
            var decoder = ChatSSEDecoder()
            #expect(throws: ChatWireError.invalidFrame) {
                try decoder.append(Data((header + "\ndata: {\"turn_id\":\"t\",\"seq\":1,\"event\":{\"type\":\"usage\"}}\n\n").utf8))
            }
        }
        var decoder = ChatSSEDecoder()
        #expect(throws: ChatWireError.invalidFrame) { try decoder.append(Data([0xff, 10])) }
    }

    @Test("Every incomplete prefix remains uncommitted at EOF and resumes safely")
    func truncatedFrame() throws {
        let raw = try ChatWireFixtures.frames("completed")[1]
        for split in 1..<raw.count {
            var decoder = ChatSSEDecoder()
            let events = try decoder.append(Data(raw.prefix(split)))
            // A keepalive prefix is allowed to finish, but partial data never emits.
            #expect(events.isEmpty)
            _ = try? decoder.finish()
        }
        var decoder = ChatSSEDecoder()
        _ = try decoder.append(Data("data: {".utf8))
        #expect(throws: ChatWireError.unexpectedEOF) { try decoder.finish() }
        var bounded = ChatSSEDecoder()
        #expect(throws: ChatWireError.frameTooLarge) { try bounded.append(Data(repeating: 65, count: ChatSSEDecoder.maxLineBytes + 1)) }
        var eventBound = ChatSSEDecoder()
        let manyLines = Data(String(repeating: ":" + String(repeating: "x", count: 1000) + "\n", count: 2200).utf8)
        #expect(throws: ChatWireError.frameTooLarge) { try eventBound.append(manyLines) }
    }

    @Test("Usage replaces totals, sources retain associations, tool rows retain order and details")
    func projectionUpdates() throws {
        let events = [
            #"{"type":"text.delta","part_id":"p","text":"before"}"#,
            #"{"type":"tool.started","call_id":"c","name":"patch","label":"Edit","target":"a.txt","summary":"Working"}"#,
            #"{"type":"text.delta","part_id":"q","text":"after"}"#,
            #"{"type":"text.delta","part_id":"p","text":"!"}"#,
            #"{"type":"tool.result","call_id":"c","result":"done","label":"Edit","target":"a.txt","summary":"Changed","details":["+line"],"details_kind":"diff","omitted_lines":3}"#,
            #"{"type":"source","source_id":"s","call_id":"c","title":"Old","attachment_id":"a"}"#,
            #"{"type":"source","source_id":"s","call_id":"c","title":"New","attachment_id":"a"}"#,
            #"{"type":"usage","input_tokens":10,"output_tokens":3,"requests":1}"#,
            #"{"type":"usage","input_tokens":15,"output_tokens":4,"requests":2}"#
        ]
        var reducer = ChatTurnReducer(turnID: "t")
        for (index, event) in events.enumerated() { _ = try reducer.apply(ChatWireFixtures.envelope(event, seq: index + 1)) }
        #expect(reducer.projection.parts.count == 3)
        #expect(reducer.projection.parts[0] == .text(id: "p", text: "before!"))
        #expect(reducer.projection.parts[2] == .text(id: "q", text: "after"))
        guard case .tool(let tool) = reducer.projection.parts[1] else { Issue.record("Missing tool"); return }
        #expect(tool.result?.details == ["+line"])
        #expect(tool.result?.detailsKind == .diff)
        #expect(tool.result?.omittedLines == 3)
        #expect(reducer.projection.sources.count == 1)
        #expect(reducer.projection.sources[0].title == "New")
        #expect(reducer.projection.sources[0].attachmentID == "a")
        #expect(reducer.projection.usage == ChatWireUsage(inputTokens: 15, outputTokens: 4, requests: 2))
    }

    @Test("Pending permission preserves revision, expiry, choices, and child identity")
    func permissionState() throws {
        let frames = try ChatWireFixtures.frames("delegation-cancelled")
        let reducer = try ChatWireFixtures.replay(Array(frames.prefix(5)), turnID: "delegated-stop")
        let pending = try #require(reducer.projection.pendingPermission)
        #expect(pending.opID == "op")
        #expect(pending.revision == 0)
        #expect(pending.expiresAt == 2_000_000_000)
        #expect(pending.subagentID == "helper")
        #expect(pending.options == [.once, .deny])
        #expect(pending.previewLanguage == .diff)
        // There is no permission-resolved event in v1; later adapters reconcile
        // another client's decision with the operation receipt.
    }

    @Test("Concurrent approvals retain order and child completion clears only that child's requests")
    func concurrentPermissions() throws {
        func permission(_ id: String, revision: Int = 0, child: String? = nil) -> String {
            let subagent = child.map { ",\"subagent_id\":\"\($0)\"" } ?? ""
            return "{\"type\":\"permission.request\",\"op_id\":\"\(id)\",\"revision\":\(revision),\"tool\":\"write_file\",\"summary\":\"Edit\",\"preview\":\"+line\",\"risk\":\"file\",\"reason\":\"Approval\",\"options\":[\"once\",\"deny\"],\"expires_at\":2000000000\(subagent)}"
        }
        var reducer = ChatTurnReducer(turnID: "t")
        let events = [
            permission("parent"), permission("child", child: "helper"),
            permission("sibling", child: "other"), permission("parent", revision: 2),
            permission("parent", revision: 1),
            #"{"type":"subagent.finished","id":"helper","call_id":"delegate","title":"Helper","status":"cancelled"}"#
        ]
        for (index, event) in events.enumerated() { _ = try reducer.apply(ChatWireFixtures.envelope(event, seq: index + 1)) }
        #expect(reducer.projection.pendingPermissions.map(\.opID) == ["parent", "sibling"])
        #expect(reducer.projection.pendingPermission?.revision == 2)
        let prefix = try ChatWireFixtures.replay(Array(ChatWireFixtures.frames("delegation-cancelled").prefix(6)), turnID: "delegated-stop")
        #expect(prefix.projection.terminal == nil)
        #expect(prefix.projection.pendingPermissions.isEmpty)
    }

    @Test("Direct context decoding enforces an optional-present discriminator")
    func contextDiscriminator() throws {
        let fields = #""used_tokens":0,"limit_tokens":512,"system_tools_tokens":0,"conversation_tokens":0,"tool_output_tokens":0,"free_tokens":512,"percent":0"#
        for type in ["null", "5", #""usage""#] {
            #expect(throws: (any Error).self) {
                try JSONDecoder().decode(ChatWireContextEvent.self, from: Data(("{\"type\":" + type + "," + fields + "}").utf8))
            }
        }
        #expect(try JSONDecoder().decode(ChatWireContextEvent.self, from: Data(("{" + fields + "}").utf8)).limitTokens == 512)
    }

    @Test("Live tool output is bounded without changing the final result")
    func outputBound() throws {
        let text = String(repeating: "x", count: ChatTurnReducer.maxToolOutputCharacters + 17)
        var reducer = ChatTurnReducer(turnID: "t")
        _ = try reducer.apply(ChatWireFixtures.envelope("{\"type\":\"tool.output\",\"call_id\":\"c\",\"text\":\"\(text)\"}"))
        guard case .tool(let tool) = reducer.projection.parts[0] else { Issue.record("Missing tool"); return }
        #expect(tool.output.count == ChatTurnReducer.maxToolOutputCharacters)
        #expect(tool.omittedOutputCharacters == 17)
    }

    @Test("Experimental backend preference defaults to legacy and persists separately")
    @MainActor
    func backendSetting() throws {
        let suite = "ChatBackendTests-" + UUID().uuidString
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let config = ChatBackendConfig(defaults: defaults)
        #expect(config.chatBackend == .legacy)
        config.chatBackend = .ownerV1
        #expect(ChatBackendConfig(defaults: defaults).chatBackend == .ownerV1)
        defaults.set("future", forKey: ChatBackendConfig.storageKey)
        #expect(ChatBackendConfig(defaults: defaults).chatBackend == .legacy)
    }
}
