import Foundation
import Observation

/// The single ``ToolRegistry`` the chat loop sees, composed of the built-in
/// tools and whatever MCP connectors are currently up.
///
/// Issue #1716 asks that an MCP tool call "render like the built-in tool calls
/// already do". Composing at the registry boundary is what buys that for free:
/// ``ChatViewModel`` keeps talking to one registry, ``ChatView`` keeps
/// rendering `message.toolCalls` generically, and neither learns the word
/// "MCP".
///
/// Dispatch is by ownership, not by name pattern. Matching on the `server__`
/// shape would misroute the moment a built-in tool is ever named with a double
/// underscore, and would silently swallow a name neither side owns.
@MainActor
@Observable
final class CompositeToolRegistry: ToolRegistry {
    let builtin: BuiltinToolRegistry
    let mcp: MCPToolRegistry

    init(builtin: BuiltinToolRegistry, mcp: MCPToolRegistry) {
        self.builtin = builtin
        self.mcp = mcp
    }

    /// Built-ins first so the stable, always-present tools lead the list the
    /// model reads, and a flapping connector can't reorder them mid-session.
    ///
    /// A connector tool whose name collides with a built-in is DROPPED rather
    /// than shadowing it: `browse` means the SSRF-guarded, user-approved
    /// built-in, and a connector that claims the name must not inherit the
    /// trust the user has already placed in it.
    var definitions: [ToolDefinition] {
        let builtinNames = Set(builtin.definitions.map { $0.function.name })
        return builtin.definitions
            + mcp.definitions.filter { !builtinNames.contains($0.function.name) }
    }

    func run(_ call: ToolCall) async -> ToolCallResult {
        let name = call.function.name
        if builtin.definitions.contains(where: { $0.function.name == name }) {
            return await builtin.run(call)
        }
        if mcp.catalog.tools.contains(where: { $0.function.name == name }) {
            return await mcp.run(call)
        }
        // Neither side owns it. ``ChatViewModel/toolRefusalMessage`` normally
        // catches this before dispatch, so reaching here means a non-chat
        // caller. Answer with the same shape the built-in registry uses — a
        // recoverable prose nudge — but list the names ACROSS both sides, not
        // just the built-in three.
        let available = definitions.map { $0.function.name }.sorted().joined(separator: ", ")
        return ToolCallResult(
            toolCallID: call.id,
            content: "unknown tool '\(name)'\(available.isEmpty ? "" : " — available: \(available)")",
            isError: true,
            failureKind: .toolFailed
        )
    }
}
