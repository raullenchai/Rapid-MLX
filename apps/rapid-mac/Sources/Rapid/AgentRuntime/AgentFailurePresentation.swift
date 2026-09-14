import Foundation

/// User-facing recovery copy for stable server failure codes.
///
/// Codes remain useful in events and logs, but exposing them in Chat makes a
/// recoverable compact-model limit look like an application crash. Keep this
/// mapping exhaustive by category and never include raw server/tool text.
enum AgentFailurePresentation {
    static func message(for code: String?) -> String {
        switch code {
        case "tool_call_during_final_synthesis":
            return "Rapid reached this model’s action limit before it could finish. Try splitting the request into two smaller steps."
        case "parallel_tool_call_limit_exceeded":
            return "The model requested several actions at once. Those requested actions weren’t run; try again."
        case "unadvertised_tool_call", "invalid_tool_arguments", "reused_tool_call_id":
            return "Rapid couldn’t safely run the latest requested action. It wasn’t run; try again."
        case "model_request_failed", "empty_model_turn":
            return "The model stopped before finishing. Try again."
        case "agent_adapter_cancelled", "agent_adapter_failure":
            return "Something interrupted the agent task. Check whether any approved action completed before trying again."
        case .none:
            return "Rapid couldn’t complete the agent task. Try again."
        default:
            return "Rapid couldn’t complete the agent task. Try again or split it into smaller steps."
        }
    }
}
