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
    typealias WebSearchRunner = (
        _ arguments: String,
        _ provider: WebSearchProvider,
        _ apiKey: String?
    ) async -> ToolCallResult

    /// Model-visible audit note for the one credential-recovery transition.
    /// It deliberately names the mode change without echoing any credential.
    static func rejectedKeyRecoveryNote(for provider: WebSearchProvider) -> String {
        "Note: Rapid removed a rejected saved \(provider.displayName) key and retried this search using \(provider.displayName)'s keyless mode."
    }

    /// Per-invocation approval gate for ``browse``. Held on the shared registry
    /// so the SwiftUI approval dialog + the Settings auto-approve switch bind to
    /// the same object the tool runner consults.
    let browseApproval: BrowseApprovalStore
    /// Which backend ``web_search`` dispatches to + the stored API key. Owned by
    /// the registry so the chat loop doesn't need to thread a separate
    /// environment value through every tool call.
    let webSearch: WebSearchConfig
    /// Injected at the service boundary so the state transition can be tested
    /// without a live provider. Production always uses ``WebSearchTool/run``.
    private let webSearchRunner: WebSearchRunner
    private struct RejectedKeyRecoveryTransition {
        let rejectedRevision: UInt64
        let keylessRevision: UInt64
    }
    /// Records only transitions performed by this registry so an overlapping
    /// stale response can distinguish automatic recovery from a user's manual
    /// clear. Revisions also make same-value credential replacement observable.
    private var rejectedKeyRecoveryTransitions: [
        WebSearchProvider: RejectedKeyRecoveryTransition
    ] = [:]

    init(
        browseApproval: BrowseApprovalStore = BrowseApprovalStore(),
        webSearch: WebSearchConfig = WebSearchConfig(),
        webSearchRunner: @escaping WebSearchRunner = { arguments, provider, apiKey in
            await WebSearchTool.run(
                arguments: arguments,
                provider: provider,
                apiKey: apiKey
            )
        }
    ) {
        self.browseApproval = browseApproval
        self.webSearch = webSearch
        self.webSearchRunner = webSearchRunner
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
            result = await runWebSearchWithCredentialRecovery(call)
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

    /// Execute one web-search call, with one narrowly-scoped configuration
    /// recovery: a rejected optional Keenable credential can transition to the
    /// provider's supported keyless mode and replay the SAME call once.
    ///
    /// This is not a generic retry loop. Producer-owned failure metadata is the
    /// gate, so network failures, quota/rate limits, malformed queries, and
    /// prose that happens to mention a key never enter this path. The rejected
    /// key is removed before replay, which makes resending it impossible and
    /// persists the selected recovery for later searches. If Keychain cannot
    /// establish that post-condition, the original failure remains visible.
    private func runWebSearchWithCredentialRecovery(
        _ call: ToolCall
    ) async -> ToolCallResult {
        let provider = webSearch.provider
        let credential = webSearch.credentialSnapshot(for: provider)
        let key = credential.key
        let first = await webSearchRunner(call.function.arguments, provider, key)

        guard provider.recoversRejectedKeyKeylessly,
              key != nil,
              first.failureKind == .webSearchKeyRejected,
              !Task.isCancelled
        else {
            return first
        }

        let current = webSearch.credentialSnapshot(for: provider)
        if current == credential {
            // This request still owns the rejected value. Establish keyless
            // mode persistently before replaying so it cannot be resent.
            guard webSearch.setAPIKey(nil, for: provider) else { return first }
            let keyless = webSearch.credentialSnapshot(for: provider)
            guard keyless.key == nil else { return first }
            rejectedKeyRecoveryTransitions[provider] = RejectedKeyRecoveryTransition(
                rejectedRevision: credential.revision,
                keylessRevision: keyless.revision
            )
        } else if current.key == nil,
                  let transition = rejectedKeyRecoveryTransitions[provider],
                  transition.rejectedRevision == credential.revision,
                  transition.keylessRevision == current.revision
        {
            // An overlapping rejection already established keyless mode. This
            // call can share that exact registry-owned transition.
        } else {
            // Settings saved, re-saved, or cleared a credential while this
            // request was suspended. The stale response has no authority to
            // clear, test, or narrate that user-owned mutation.
            return first
        }

        // Once the persistent transition begins, invoke its one bounded replay
        // even if cancellation races with the synchronous Keychain mutation.
        // Otherwise the call could return the old credential failure after
        // having already removed the key. The real transport still observes
        // task cancellation and can terminate the replay attempt promptly.
        let recovered = await webSearchRunner(call.function.arguments, provider, nil)
        return ToolCallResult(
            toolCallID: recovered.toolCallID,
            content: recovered.content + "\n\n" + Self.rejectedKeyRecoveryNote(for: provider),
            isError: recovered.isError,
            failureKind: recovered.failureKind
        )
    }
}
