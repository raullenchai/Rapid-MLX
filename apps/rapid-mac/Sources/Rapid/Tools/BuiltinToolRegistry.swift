import Foundation

/// Concrete ``ToolRegistry`` that ships the built-in tools the chat
/// surface exposes:
///
///   * ``weather`` — no approval, hits Open-Meteo over HTTPS
///   * ``web_search`` — no approval, backend per ``WebSearchConfig``
///     (Keenable keyless by default; Parallel / Tavily / Brave with a
///     key; DuckDuckGo backstop)
///   * ``browse`` — USER-approved per fetch (``BrowseApprovalStore``),
///     SSRF-guarded, byte-capped
///   * ``read_document`` — no approval, reads only documents the user
///     already attached (see below)
///
/// One instance is constructed by ``RapidApp`` and shared by the chat
/// view model. Filesystem / shell tools are deliberately absent: this
/// build has no ``SandboxManager``, and a tool that touches the user's
/// disk must not ship without one.
///
/// ``read_document`` is not an exception to that rule. It accepts no
/// path — only an attachment UUID minted when the user dropped or picked
/// a file — and resolves it solely through ``DocumentContentCache``. It
/// can therefore reach nothing the user did not already hand over, which
/// is also why it needs no approval prompt of its own.
@MainActor
final class BuiltinToolRegistry: ToolRegistry {
    /// Per-invocation approval gate for ``browse``. Held on the shared registry
    /// so the SwiftUI approval dialog + the Settings auto-approve switch bind to
    /// the same object the tool runner consults.
    let browseApproval: BrowseApprovalStore
    /// Which backend ``web_search`` dispatches to + the stored API key. Owned by
    /// the registry so the chat loop doesn't need to thread a separate
    /// environment value through every tool call.
    let webSearch: WebSearchConfig

    init(
        browseApproval: BrowseApprovalStore = BrowseApprovalStore(),
        webSearch: WebSearchConfig = WebSearchConfig()
    ) {
        self.browseApproval = browseApproval
        self.webSearch = webSearch
    }

    var definitions: [ToolDefinition] {
        [
            WebSearchTool.definition,
            BrowseTool.definition,
            WeatherTool.definition,
            ReadDocumentTool.definition,
        ]
    }

    func run(_ call: ToolCall) async -> ToolCallResult {
        let result: ToolCallResult
        switch call.function.name {
        case "web_search":
            let provider = webSearch.provider
            let key = webSearch.apiKey(for: provider)
            result = await WebSearchTool.run(
                arguments: call.function.arguments,
                provider: provider,
                apiKey: key
            )
        case "browse":
            result = await BrowseTool.run(
                arguments: call.function.arguments,
                approval: browseApproval
            )
        case "weather":
            result = await WeatherTool.run(arguments: call.function.arguments)
        case "read_document":
            result = await ReadDocumentTool.run(arguments: call.function.arguments)
        default:
            // The model invented a tool name we don't ship — return an
            // error result so it gets a chance to recover instead of
            // throwing and tearing the chat loop down.
            result = ToolCallResult(
                toolCallID: call.id,
                content: "unknown tool '\(call.function.name)' — available: web_search, browse, weather, read_document",
                isError: true
            )
        }
        // The individual tools don't know the toolCallID at run time, so
        // fill it in here. Classification is centralised at this boundary:
        // raw content continues to the model, but the transcript gets only a
        // stable diagnosis.
        let failureKind = result.failureKind ?? FailureDiagnoser.toolFailureKind(
            toolName: call.function.name,
            content: result.content,
            isError: result.isError
        )
        return ToolCallResult(
            toolCallID: call.id,
            content: result.content,
            isError: result.isError || failureKind != nil,
            failureKind: failureKind
        )
    }
}
