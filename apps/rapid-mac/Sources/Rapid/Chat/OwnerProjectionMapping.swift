import Foundation
import CryptoKit

/// Converts sanitized owner state to transcript rows; reasoning is never copied.
func ownerProjectionMessages(_ projection: ChatTurnProjection, assistantID: UUID, createdAt: Date) -> [ChatMessage] {
    func rowID(_ key: String) -> UUID {
        let bytes = Array(SHA256.hash(data: Data("\(projection.turnID):\(key)".utf8)).prefix(16))
        return UUID(uuid: (bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7], bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]))
    }
    let text = projection.parts.compactMap { part -> String? in
        if case .text(_, let text) = part { return text }; return nil
    }.joined()
    let tools = projection.parts.compactMap { part -> ChatTurnProjection.Tool? in
        if case .tool(let tool) = part { return tool }; return nil
    }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let calls = tools.map { tool in
        let data = (try? encoder.encode(tool.started?.args ?? [:])) ?? Data("{}".utf8)
        return ToolCall(id: tool.callID, name: tool.started?.name ?? "", arguments: String(decoding: data, as: UTF8.self))
    }
    let terminal = projection.terminal
    let status: ChatMessage.Status = terminal == nil ? .streaming : (terminal?.outcome == .completed ? .complete : .failed)
    let error: String? = terminal.flatMap { event in
        guard event.outcome != .completed else { return nil }
        if event.errorCode == "owner_restarted" { return "Interrupted: the chat owner restarted." }
        if event.outcome == .cancelled { return "Interrupted." }
        return event.errorCode ?? "The turn failed."
    }
    var rows = [ChatMessage(id: assistantID, role: .assistant, content: text, status: status, errorMessage: error, toolCalls: calls.isEmpty ? nil : calls, createdAt: createdAt)]
    for tool in tools {
        let output = tool.result.map { tool.output.isEmpty || tool.output == $0.result ? $0.result : tool.output + "\n" + $0.result } ?? tool.output
        if !output.isEmpty || tool.result != nil {
            rows.append(ChatMessage(id: rowID("tool:" + tool.callID), role: .tool, content: output, status: tool.result == nil ? .streaming : (tool.result?.outcome == .error ? .failed : .complete), toolCallID: tool.callID, createdAt: createdAt))
        }
    }
    for source in projection.sources {
        let detail = source.url ?? source.attachmentID ?? ""
        rows.append(ChatMessage(id: rowID("source:" + source.sourceID), role: .system, content: detail.isEmpty ? source.title : "\(source.title): \(detail)", wireVisibility: .transcriptOnly, createdAt: createdAt))
    }
    if let context = projection.context, context.compacted {
        rows.append(ChatMessage(id: rowID("context"), role: .system, content: context.notice.isEmpty ? "Context compacted." : context.notice, wireVisibility: .transcriptOnly, createdAt: createdAt))
    }
    return rows
}
