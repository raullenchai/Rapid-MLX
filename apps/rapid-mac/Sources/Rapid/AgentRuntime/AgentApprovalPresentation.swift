import Foundation

/// Stable, redacted copy for Desktop's approval alert.
///
/// The raw `arguments` dictionary is deliberately never accepted here. The
/// runtime owns redaction and supplies `approval_summary`; Desktop formats only
/// that bounded summary so secrets and large model-generated payloads cannot
/// leak into a system alert.
enum AgentApprovalPresentation {
    static func title(for action: AgentPendingAction) -> String {
        "Allow \(toolName(action.name))?"
    }

    static func message(for action: AgentPendingAction) -> String {
        var lines = [riskDescription(action.risk)]
        let summary = action.approvalSummary ?? [:]
        for key in summary.keys.sorted().prefix(4) {
            guard let value = summary[key] else { continue }
            let safeKey = bounded(ChatTextSanitizer.sanitizeForDisplay(key), limit: 40)
            lines.append("\(safeKey): \(display(value))")
        }
        if summary.count > 4 { lines.append("…and \(summary.count - 4) more details") }
        return lines.joined(separator: "\n")
    }

    private static func toolName(_ raw: String) -> String {
        let safe = ChatTextSanitizer.sanitizeForDisplay(raw)
            .replacingOccurrences(of: "__", with: " · ")
            .replacingOccurrences(of: "_", with: " ")
        return bounded(safe, limit: 72)
    }

    private static func riskDescription(_ risk: AgentToolRisk) -> String {
        switch risk {
        case .readOnly:
            "This action reads local or connected information."
        case .localChange:
            "This action will change something on this Mac."
        case .externalSideEffect:
            "This action can affect an external service or other people."
        }
    }

    private static func display(_ value: CodableJSON) -> String {
        switch value {
        case .null: "None"
        case .bool(let value): value ? "Yes" : "No"
        // Approval copy must describe the exact value the runtime classified.
        // `String(Double)` is locale-independent and round-trippable; display
        // formatting with a precision cap could make 1.2349 appear as 1.235.
        case .number(let value): String(value)
        case .string(let value):
            bounded(ChatTextSanitizer.sanitizeForDisplay(value), limit: 120)
        case .array(let values): "\(values.count) items"
        case .object(let values): "\(values.count) fields"
        }
    }

    private static func bounded(_ value: String, limit: Int) -> String {
        guard value.count > limit else { return value }
        return String(value.prefix(limit)) + "…"
    }
}
