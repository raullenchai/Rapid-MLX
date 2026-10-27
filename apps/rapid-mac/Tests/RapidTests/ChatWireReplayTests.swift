import Foundation
import Testing
@testable import Rapid

/// Reads the repository corpus in place: changes to the Python wire contract
/// must reach these tests, not a separately maintained Swift copy.
enum ChatWireFixtures {
    static let directory: URL = {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { url.deleteLastPathComponent() }
        return url.appendingPathComponent("tests/fixtures/chat_runtime")
    }()

    static let names = [
        "completed", "cancelled", "interrupted", "tool-round", "web-search",
        "reasoning-redacted", "approved-action", "denied-action", "context-compaction",
        "delegation", "delegation-cancelled"
    ]

    static func objects(_ name: String) throws -> [[String: Any]] {
        try String(contentsOf: directory.appendingPathComponent(name + ".jsonl"), encoding: .utf8)
            .split(separator: "\n").map {
                try JSONSerialization.jsonObject(with: Data($0.utf8)) as! [String: Any]
            }
    }

    static func frames(_ name: String, newline: String = "\n") throws -> [Data] {
        // Preserve the checked-in JSON bytes, including multibyte text.
        try String(contentsOf: directory.appendingPathComponent(name + ".jsonl"), encoding: .utf8)
            .split(separator: "\n").map { line in
                let object = try JSONSerialization.jsonObject(with: Data(line.utf8)) as! [String: Any]
                let event = object["event"] as! [String: Any]
                return Data((": keepalive" + newline + newline
                    + "id: \(object["turn_id"]!):\(object["seq"]!)" + newline
                    + "event: \(event["type"]!)" + newline
                    + "data: " + line + newline + newline).utf8)
            }
    }

    static func envelope(_ event: String, seq: Int = 1, turn: String = "t") throws -> ChatWireEnvelope {
        try JSONDecoder().decode(ChatWireEnvelope.self, from: Data(
            "{\"turn_id\":\"\(turn)\",\"seq\":\(seq),\"event\":\(event)}".utf8
        ))
    }

    static func replay(_ chunks: [Data], turnID: String) throws -> ChatTurnReducer {
        var decoder = ChatSSEDecoder()
        var reducer = ChatTurnReducer(turnID: turnID)
        for chunk in chunks {
            for envelope in try decoder.append(chunk) { _ = try reducer.apply(envelope) }
        }
        try decoder.finish()
        return reducer
    }
}

@Suite("Owner wire-v1 golden replay")
struct ChatWireReplayTests {
    @Test("All checked-in traces traverse production SSE at every byte split", arguments: ChatWireFixtures.names)
    func everyByteSplit(_ name: String) throws {
        let objects = try ChatWireFixtures.objects(name)
        let turnID = objects[0]["turn_id"] as! String
        let frames = try ChatWireFixtures.frames(name)
        let baseline = try ChatWireFixtures.replay([frames.reduce(Data(), +)], turnID: turnID)
        try assertSemantics(name, baseline.settledProjection, objects)
        for newline in ["\n", "\r\n"] {
            let framed = try ChatWireFixtures.frames(name, newline: newline)
            let middle = framed.count / 2
            let scenarios = [
                framed,
                Array(framed.prefix(middle)) + framed,
                Array(framed.prefix(middle)) + Array(framed.dropFirst(middle + 1)) + Array(framed.dropFirst(middle))
            ]
            for (scenario, frames) in scenarios.enumerated() {
                let raw = frames.reduce(Data(), +)
                for split in 0...raw.count {
                    let replay = try ChatWireFixtures.replay([Data(raw.prefix(split)), Data(raw.dropFirst(split))], turnID: turnID)
                    #expect(replay.settledProjection == baseline.settledProjection, "\(name) scenario \(scenario) split \(split)")
                    #expect(replay.lastApplied == objects.count)
                }
                let bytes = try ChatWireFixtures.replay(raw.map { Data([$0]) }, turnID: turnID)
                #expect(bytes.settledProjection == baseline.settledProjection)
            }
        }
    }

    @Test("Duplicate prefixes and each dropped frame recover from the applied cursor", arguments: ChatWireFixtures.names)
    func replayRecovery(_ name: String) throws {
        let frames = try ChatWireFixtures.frames(name)
        let objects = try ChatWireFixtures.objects(name)
        let turnID = objects[0]["turn_id"] as! String
        let expected = try ChatWireFixtures.replay(frames, turnID: turnID).settledProjection
        for index in frames.indices {
            let duplicates = Array(frames.prefix(index + 1)) + frames
            #expect(try ChatWireFixtures.replay(duplicates, turnID: turnID).settledProjection == expected)
            var decoder = ChatSSEDecoder()
            var reducer = ChatTurnReducer(turnID: turnID)
            for (position, frame) in frames.enumerated() where position != index {
                // Deliver one byte at a time, including the reconnect suffix.
                for byte in frame {
                    if let event = try decoder.append(byte) {
                        let effect = try reducer.apply(event)
                        if position > index { #expect(effect == .resume(after: index)) }
                    }
                }
            }
            #expect(reducer.lastApplied == index)
            #expect(reducer.projection.terminal == nil)
            for frame in frames.dropFirst(index) {
                for byte in frame {
                    if let event = try decoder.append(byte) { #expect(try reducer.apply(event) == .applied) }
                }
            }
            #expect(reducer.settledProjection == expected)
            #expect(reducer.lastApplied == frames.count)
            try decoder.finish()
        }
    }

    @Test("Schema, fixture discriminators, and exhaustive reducer enum stay aligned")
    func schemaCoverage() throws {
        let schema = try JSONSerialization.jsonObject(with: Data(contentsOf:
            ChatWireFixtures.directory.appendingPathComponent("wire-v1.schema.json"))) as! [String: Any]
        let properties = schema["properties"] as! [String: Any]
        let event = properties["event"] as! [String: Any]
        let discriminator = event["discriminator"] as! [String: Any]
        let mapping = discriminator["mapping"] as! [String: String]
        #expect(Set(mapping.keys) == Set(ChatWireEventKind.allCases.map(\.rawValue)))
        var observed: Set<String> = []
        let checkedIn = try FileManager.default.contentsOfDirectory(at: ChatWireFixtures.directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "jsonl" }.map { $0.deletingPathExtension().lastPathComponent }
        #expect(Set(checkedIn) == Set(ChatWireFixtures.names))
        for name in checkedIn {
            var decoder = ChatSSEDecoder()
            let events = try decoder.append(ChatWireFixtures.frames(name).reduce(Data(), +))
            var reducer = ChatTurnReducer(turnID: events[0].turnID)
            for envelope in events {
                if case .unknown = envelope.event { Issue.record("Known fixture decoded as unknown") }
                observed.insert(envelope.event.type)
                #expect(try reducer.apply(envelope) == .applied)
            }
        }
        #expect(observed == Set(mapping.keys))
        // ChatTurnReducer switches exhaustively over ChatWireEvent, so adding a
        // typed event without a reducer case also fails the production build.
    }

    @Test("Request goldens decode defaults and round-trip their full intent")
    func requestGoldens() throws {
        struct Golden: Decodable { let request: ChatTurnRequest; let surface: String }
        let values = try JSONDecoder().decode([Golden].self, from: Data(contentsOf:
            ChatWireFixtures.directory.appendingPathComponent("turn-requests.json")))
        #expect(values.count == 4)
        for value in values {
            #expect(value.request.surface.rawValue == value.surface)
            #expect(value.request.maxTokens == 32768)
            #expect(value.request.contextTokens == 32768)
            #expect(value.request.attachmentIDs.isEmpty)
            #expect(value.request.actionTools)
            #expect(try JSONDecoder().decode(ChatTurnRequest.self, from: JSONEncoder().encode(value.request)) == value.request)
        }
    }

    private func assertSemantics(_ name: String, _ projection: ChatTurnProjection, _ objects: [[String: Any]]) throws {
        let events = objects.map { $0["event"] as! [String: Any] }
        let start = events[0]
        #expect(projection.started?.messageID == start["message_id"] as? String)
        #expect(projection.started?.branchID == start["branch_id"] as? String)
        #expect(projection.started?.model == start["model"] as? String)
        #expect(projection.started?.thinking == start["thinking"] as? Bool)
        let terminal = events.last!
        #expect(projection.terminal?.outcome.rawValue == terminal["outcome"] as? String)
        #expect(projection.terminal?.finalMessageID == terminal["final_message_id"] as? String)
        #expect(projection.terminal?.errorCode == terminal["error_code"] as? String)
        let usage = terminal["usage"] as? [String: Int] ?? [:]
        #expect(projection.usage.inputTokens == usage["input_tokens", default: 0])
        #expect(projection.usage.outputTokens == usage["output_tokens", default: 0])
        #expect(projection.usage.requests == usage["requests", default: 0])
        #expect(projection.pendingPermission == nil)
        var expectedOrder: [String] = []
        for event in events {
            if event["type"] as? String == "text.delta", let id = event["part_id"] as? String {
                if !expectedOrder.contains("text:" + id) { expectedOrder.append("text:" + id) }
            }
            if event["type"] as? String == "tool.started", let id = event["call_id"] as? String {
                expectedOrder.append("tool:" + id)
            }
        }
        let order = projection.parts.map { part -> String in
            switch part {
            case .text(let id, let text):
                let deltas = events.filter { $0["type"] as? String == "text.delta" && $0["part_id"] as? String == id }
                #expect(text == deltas.compactMap { $0["text"] as? String }.joined())
                return "text:" + id
            case .tool(let tool):
                let start = events.first { $0["type"] as? String == "tool.started" && $0["call_id"] as? String == tool.callID }!
                #expect(tool.started?.name == start["name"] as? String)
                #expect(tool.started?.label == start["label"] as? String ?? "")
                #expect(tool.started?.target == start["target"] as? String ?? "")
                #expect(tool.started?.summary == start["summary"] as? String ?? "")
                if let result = events.last(where: { $0["type"] as? String == "tool.result" && $0["call_id"] as? String == tool.callID }) {
                    #expect(tool.result?.result == result["result"] as? String)
                    #expect(tool.result?.summary == result["summary"] as? String ?? "")
                    #expect(tool.result?.label == result["label"] as? String ?? "")
                    #expect(tool.result?.target == result["target"] as? String ?? "")
                    #expect(tool.result?.details == result["details"] as? [String] ?? [])
                    #expect(tool.result?.detailsKind.rawValue == result["details_kind"] as? String ?? "text")
                    #expect(tool.result?.outcome.rawValue == result["outcome"] as? String ?? "success")
                }
                let output = events.filter { $0["type"] as? String == "tool.output" && $0["call_id"] as? String == tool.callID }
                #expect(tool.output == output.compactMap { $0["text"] as? String }.joined())
                return "tool:" + tool.callID
            }
        }
        #expect(order == expectedOrder)
        if name == "web-search" {
            #expect(projection.sources.count == 1)
            #expect(projection.sources.first?.sourceID == "1")
            #expect(projection.sources.first?.callID == "search")
            #expect(projection.sources.first?.url == "https://example.com/news")
        } else { #expect(projection.sources.isEmpty) }
        if name == "context-compaction" {
            #expect(projection.context?.usedTokens == 3100)
            #expect(projection.context?.limitTokens == 8192)
            #expect(projection.context?.percent == 38)
            #expect(projection.context?.compacted == true)
            #expect(projection.context?.notice == "Context compacted — older details summarized.")
        } else { #expect(projection.context == nil) }
        if name.hasPrefix("delegation") {
            #expect(projection.children.count == 1)
            #expect(projection.children[0].id == "helper")
            #expect(projection.children[0].callID == "delegate")
            #expect(projection.children[0].status.rawValue == (name == "delegation" ? "completed" : "cancelled"))
            #expect(projection.children[0].toolCount == (name == "delegation" ? 2 : 1))
        } else { #expect(projection.children.isEmpty) }
        let encoded = String(decoding: try JSONEncoder().encode(projection), as: UTF8.self)
        #expect(!encoded.contains("reasoning"))
    }
}
