import Foundation

/// Sanitized presentation data only. No event archive or reasoning is retained
/// here, so a settled turn can be encoded without leaking live reasoning.
struct ChatTurnProjection: Codable, Equatable, Sendable {
    enum Part: Codable, Equatable, Sendable {
        case text(id: String, text: String)
        case tool(Tool)
    }

    struct Tool: Codable, Equatable, Sendable {
        let callID: String
        var started: ChatWireToolStarted?
        var output = ""
        var omittedOutputCharacters = 0
        var result: ChatWireToolResult?
    }

    let turnID: String
    var started: ChatWireTurnStarted?
    var parts: [Part] = []
    var sources: [ChatWireSource] = []
    var usage = ChatWireUsage()
    var context: ChatWireContextEvent?
    var pendingPermissions: [ChatWirePermissionRequest] = []
    var pendingPermission: ChatWirePermissionRequest? { pendingPermissions.first }
    var children: [ChatWireSubagent] = []
    var terminal: ChatWireTurnTerminal?
}

/// Pure, per-turn state machine. The cursor is inseparable from its projection;
/// callers start a fresh reducer at zero after restart until snapshots have a
/// contractual watermark. Transport errors never manufacture a terminal event.
struct ChatTurnReducer: Sendable {
    enum Effect: Equatable, Sendable {
        case applied
        case duplicate
        case resume(after: Int)
        case afterTerminal
    }

    static let maxToolOutputCharacters = 65_536
    static let maxReasoningCharacters = 65_536
    private(set) var projection: ChatTurnProjection
    private(set) var lastApplied = 0
    private(set) var liveReasoning: [String: String] = [:]

    init(turnID: String) {
        projection = ChatTurnProjection(turnID: turnID)
    }

    var lastEventID: String { "\(projection.turnID):\(lastApplied)" }
    var settledProjection: ChatTurnProjection { projection }

    mutating func apply(_ envelope: ChatWireEnvelope) throws -> Effect {
        guard envelope.turnID == projection.turnID else { throw ChatWireError.wrongTurn }
        if envelope.seq <= lastApplied { return .duplicate }
        if projection.terminal != nil { return .afterTerminal }
        guard envelope.seq == lastApplied + 1 else { return .resume(after: lastApplied) }

        switch envelope.event {
        case .turnStarted(let event):
            projection.started = event
        case .textDelta(let event):
            if let index = projection.parts.firstIndex(where: {
                if case .text(let id, _) = $0 { return id == event.partID }
                return false
            }), case .text(let id, let text) = projection.parts[index] {
                projection.parts[index] = .text(id: id, text: text + event.text)
            } else {
                projection.parts.append(.text(id: event.partID, text: event.text))
            }
        case .reasoningDelta(let event):
            if !event.omitted {
                liveReasoning[event.partID] = String(
                    (liveReasoning[event.partID, default: ""] + event.text)
                        .suffix(Self.maxReasoningCharacters)
                )
            }
        case .toolStarted(let event):
            updateTool(event.callID) { $0.started = event }
        case .toolOutput(let event):
            updateTool(event.callID) { tool in
                let combined = tool.output + event.text
                let omitted = max(0, combined.count - Self.maxToolOutputCharacters)
                tool.output = String(combined.suffix(Self.maxToolOutputCharacters))
                tool.omittedOutputCharacters += omitted
            }
        case .toolResult(let event):
            updateTool(event.callID) { $0.result = event }
            // Only an explicit op_id association is authoritative. Tool names
            // cannot resolve permissions: two calls can have the same name.
            let operationID = projection.parts.compactMap { part -> String? in
                guard case .tool(let tool) = part, tool.callID == event.callID,
                      case .string(let opID) = tool.started?.args["op_id"] else { return nil }
                return opID
            }.first
            if let operationID { projection.pendingPermissions.removeAll { $0.opID == operationID } }
        case .permissionRequest(let event):
            if let index = projection.pendingPermissions.firstIndex(where: { $0.opID == event.opID }) {
                if event.revision >= projection.pendingPermissions[index].revision {
                    projection.pendingPermissions[index] = event
                }
            } else {
                projection.pendingPermissions.append(event)
            }
        case .subagentStarted(let event), .subagentProgress(let event):
            updateChild(event)
        case .subagentFinished(let event):
            updateChild(event)
            if event.status != .queued && event.status != .running {
                projection.pendingPermissions.removeAll { $0.subagentID == event.id }
            }
        case .source(let event):
            if let index = projection.sources.firstIndex(where: { $0.sourceID == event.sourceID }) {
                projection.sources[index] = event
            } else {
                projection.sources.append(event)
            }
        case .usage(let event):
            projection.usage = ChatWireUsage(
                inputTokens: event.inputTokens, outputTokens: event.outputTokens, requests: event.requests
            )
        case .context(let event):
            projection.context = event
        case .turnTerminal(let event):
            projection.terminal = event
            projection.usage = event.usage
            projection.pendingPermissions.removeAll()
            liveReasoning.removeAll()
        case .unknown:
            break
        }
        lastApplied = envelope.seq
        return .applied
    }

    private mutating func updateChild(_ event: ChatWireSubagent) {
        if let index = projection.children.firstIndex(where: { $0.id == event.id }) {
            projection.children[index] = event
        } else {
            projection.children.append(event)
        }
    }

    private mutating func updateTool(_ callID: String, update: (inout ChatTurnProjection.Tool) -> Void) {
        if let index = projection.parts.firstIndex(where: {
            if case .tool(let tool) = $0 { return tool.callID == callID }
            return false
        }), case .tool(var tool) = projection.parts[index] {
            update(&tool)
            projection.parts[index] = .tool(tool)
        } else {
            var tool = ChatTurnProjection.Tool(callID: callID)
            update(&tool)
            projection.parts.append(.tool(tool))
        }
    }
}
