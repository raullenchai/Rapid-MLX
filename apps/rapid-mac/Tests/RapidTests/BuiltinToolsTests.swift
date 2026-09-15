import Foundation
import Testing
@testable import Rapid

/// Contracts for the built-in tool surface: what the registry exposes, what
/// reaches the wire, and what is refused at dispatch.
///
/// Three of the shipped tools (``web_search``, ``browse``, ``weather``) are
/// network-facing, so the gates below are load-bearing rather than cosmetic:
/// a tool stripped from the request body can STILL be named by a malformed
/// model, and only the dispatch-side refusal stops it running.
/// ``read_document`` reaches no network and no path — only documents the user
/// attached — but is held to the same dispatch contract.
@MainActor
@Suite("Built-in tools")
final class BuiltinToolsTests {
    nonisolated(unsafe) private var createdSuiteNames: [String] = []
    deinit { TestDefaultsScope.cleanup(suiteNames: createdSuiteNames) }

    private func freshDefaults() -> UserDefaults {
        let name = TestDefaultsScope.mintSuiteName(prefix: "rapid-tools-test-")
        createdSuiteNames.append(name)
        let d = UserDefaults(suiteName: name)!
        d.removePersistentDomain(forName: name)
        return d
    }

    private func makeRegistry() -> BuiltinToolRegistry {
        BuiltinToolRegistry(
            browseApproval: BrowseApprovalStore(defaults: freshDefaults()),
            webSearch: WebSearchConfig(defaults: freshDefaults(), keychain: InMemoryKeychain())
        )
    }

    // MARK: - Registry surface

    @Test("Registry exposes exactly web_search, browse, weather, and read_document")
    func registryDefinitions() {
        let names = makeRegistry().definitions.map { $0.function.name }
        #expect(names == ["web_search", "browse", "weather", "read_document"])
    }

    @Test("An unknown tool name returns an error result naming what IS available")
    func unknownToolIsRefusedNotCrashed() async {
        // A model that invents a name must get a recoverable error result, not
        // a thrown error that tears the chat loop down.
        let result = await makeRegistry().run(
            ToolCall(id: "call_1", name: "read_file", arguments: "{}")
        )
        #expect(result.isError)
        #expect(result.toolCallID == "call_1")
        #expect(result.content.contains("web_search"))
        #expect(result.content.contains("browse"))
        #expect(result.content.contains("weather"))
        #expect(!result.executed)
    }

    @Test("Registry stamps the call id onto a result the tool produced without one")
    func registryFillsToolCallID() async {
        // Tools build their results before they know the id, so the registry
        // is the single place that stamps it. A dropped id produces a
        // ``role: "tool"`` row the model can't match to its call.
        let result = await makeRegistry().run(
            ToolCall(id: "call_xyz", name: "weather", arguments: "not json")
        )
        #expect(result.toolCallID == "call_xyz")
        #expect(result.isError)
    }

    @Test("Native executor filters arguments by the advertised schema")
    func nativeExecutorNormalizesArguments() throws {
        let call = ToolCall(
            id: "weather_1",
            name: "weather",
            arguments: #"{"location":"Tokyo","country":"Japan","invented":"drop me"}"#
        )
        let normalized = try #require(NativeToolCallExecutor.normalized(
            call,
            for: WeatherTool.definition
        ))
        let data = try #require(normalized.function.arguments.data(using: .utf8))
        let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: String])
        #expect(object["location"] == "Tokyo")
        #expect(object["country"] == "Japan")
        #expect(object["invented"] == nil)
    }

    @Test("Native executor rejects unknown arguments for strict tool schemas")
    func nativeExecutorEnforcesStrictSchema() {
        let call = ToolCall(
            id: "document_1",
            name: "read_document",
            arguments: #"{"document_id":"00000000-0000-0000-0000-000000000000","offset_len":117524}"#
        )
        #expect(NativeToolCallExecutor.normalized(
            call,
            for: ReadDocumentTool.definition
        ) == nil)
    }

    /// A rejection the model cannot read is a rejection it repeats. The
    /// executor used to answer every schema violation with "arguments must be
    /// a JSON object matching the advertised schema", which names neither the
    /// offending key nor the accepted ones — so a 4B model re-sent the same
    /// `offset_len` call. Fail closed, but say what closed it.
    @Test("A strict-schema rejection names the unknown key and the allowed ones")
    func strictSchemaRejectionNamesKeys() throws {
        switch NativeToolCallExecutor.normalize(
            ToolCall(
                id: "document_1",
                name: "read_document",
                arguments: #"{"document_id":"00000000-0000-0000-0000-000000000000","offset_len":117524}"#
            ),
            for: ReadDocumentTool.definition
        ) {
        case .success:
            Issue.record("an unknown key must not normalize")
        case .failure(let rejection):
            #expect(rejection.reason.contains("offset_len"))
            #expect(rejection.reason.contains("document_id, grep, mode, offset"))
        }
    }

    @Test("An unbounded key list is truncated in the rejection text")
    func strictSchemaRejectionBoundsItsEcho() throws {
        let junk = (0..<9).map { "\"k\($0)\": 1" }.joined(separator: ",")
        switch NativeToolCallExecutor.normalize(
            ToolCall(
                id: "document_2",
                name: "read_document",
                arguments: #"{"document_id":"x",\#(junk)}"#
            ),
            for: ReadDocumentTool.definition
        ) {
        case .success:
            Issue.record("unknown keys must not normalize")
        case .failure(let rejection):
            #expect(rejection.reason.contains("…"))
            #expect(!rejection.reason.contains("k8"))
        }
    }

    @Test("Native executor rejects non-object or malformed arguments generically")
    func nativeExecutorRejectsMalformedArguments() {
        #expect(NativeToolCallExecutor.normalized(
            ToolCall(id: "bad_1", name: "weather", arguments: #"["Tokyo"]"#),
            for: WeatherTool.definition
        ) == nil)
        #expect(NativeToolCallExecutor.normalized(
            ToolCall(id: "bad_2", name: "weather", arguments: #"{"location":"Tokyo""#),
            for: WeatherTool.definition
        ) == nil)
    }

    // MARK: - Per-tool enable/disable

    @Test("A tool toggled off is stripped from the definitions sent to the model")
    func disabledToolIsStrippedFromWire() {
        let registry = makeRegistry()
        let total = registry.definitions.count
        let vm = ChatViewModel(tools: registry, toolDefaults: freshDefaults())
        #expect(vm.enabledDefinitions.count == total)
        vm.setToolEnabled("browse", false)
        #expect(!vm.enabledDefinitions.contains { $0.function.name == "browse" })
        #expect(vm.enabledDefinitions.count == total - 1)
    }

    @Test("Tool toggles persist across a fresh view model on the same defaults")
    func toolToggleIsPersisted() {
        let defaults = freshDefaults()
        let first = ChatViewModel(tools: makeRegistry(), toolDefaults: defaults)
        first.setToolEnabled("weather", false)

        let second = ChatViewModel(tools: makeRegistry(), toolDefaults: defaults)
        #expect(second.disabledTools.contains("weather"))
        #expect(!second.enabledDefinitions.contains { $0.function.name == "weather" })
    }

    @Test("An untouched tool defaults to enabled so a shipped tool needs no opt-in")
    func unsetToolDefaultsToEnabled() {
        let vm = ChatViewModel(tools: makeRegistry(), toolDefaults: freshDefaults())
        #expect(vm.disabledTools.isEmpty)
    }

    @Test("Personal Intelligence projects only enabled live-data built-ins")
    func personalIntelligenceToolProjection() {
        let vm = ChatViewModel(tools: makeRegistry(), toolDefaults: freshDefaults())
        #expect(vm.personalIntelligenceDefinitions.map { $0.function.name } == [
            "web_search", "browse", "weather",
        ])

        vm.setToolEnabled("browse", false)
        #expect(vm.personalIntelligenceDefinitions.map { $0.function.name } == [
            "web_search", "weather",
        ])
    }

    @Test("Personal Intelligence refuses invalid arguments before dispatch")
    func personalIntelligenceRejectsInvalidArguments() async throws {
        let vm = ChatViewModel(tools: makeRegistry(), toolDefaults: freshDefaults())
        let action = try JSONDecoder().decode(
            AgentPendingAction.self,
            from: Data(#"{"call_id":"bad","name":"weather","arguments":{},"approval_summary":null,"risk":"read_only","approval_required":false}"#.utf8)
        )

        let result = await vm.executePersonalIntelligenceTool(
            action,
            advertised: vm.personalIntelligenceDefinitions
        )

        #expect(result.isError)
        #expect(!result.executed)
        #expect(result.content.contains("location"))
    }

    @Test("Personal Intelligence records a declined browse as unexecuted")
    func personalIntelligenceRecordsDeclinedBrowse() async throws {
        let registry = DeclinedBrowseRegistry()
        let vm = ChatViewModel(tools: registry, toolDefaults: freshDefaults())
        let action = try JSONDecoder().decode(
            AgentPendingAction.self,
            from: Data(#"{"call_id":"declined","name":"browse","arguments":{"url":"https://example.com"},"approval_summary":null,"risk":"read_only","approval_required":false}"#.utf8)
        )

        let result = await vm.executePersonalIntelligenceTool(
            action,
            advertised: registry.definitions
        )

        #expect(result.isError)
        #expect(!result.executed)
    }

    // MARK: - Dispatch refusal

    @Test("A call for a tool that wasn't advertised this round is refused, not run")
    func unadvertisedToolIsRefused() {
        // Omitting a tool from the request body does NOT stop a malformed model
        // emitting a call for it — this refusal is the gate that actually
        // prevents the network fetch.
        let refusal = ChatViewModel.toolRefusalMessage(
            name: "browse",
            allowed: ["web_search", "weather"],
            known: ["web_search", "weather", "browse"]
        )
        #expect(refusal != nil)
        #expect(refusal?.contains("browse") == true)
    }

    @Test("A call for a tool the model invented outright is refused, not run")
    func unknownToolIsRefused() {
        // Not advertised AND not a shipped tool — must be refused before
        // dispatch, never handed to tools.run.
        let refusal = ChatViewModel.toolRefusalMessage(
            name: "run_shell",
            allowed: ["web_search", "weather", "browse"],
            known: ["web_search", "weather", "browse"]
        )
        #expect(refusal != nil)
        #expect(refusal?.contains("run_shell") == true)
    }

    @Test("A call for an advertised tool is allowed through")
    func advertisedToolIsAllowed() {
        let all: Set<String> = ["web_search", "weather", "browse"]
        #expect(ChatViewModel.toolRefusalMessage(name: "weather", allowed: all, known: all) == nil)
        #expect(ChatViewModel.toolRefusalMessage(
            name: "weather", allowed: ["weather"], known: all) == nil)
    }

    // MARK: - Broken-alias wire strip

    @Test("Tools are stripped entirely for an alias known to mishandle tool calls")
    func brokenAliasGetsNoTools() {
        // Sending tools to a model empirically proven to ignore them produces a
        // confidently-hallucinated answer with no chip to warn the user.
        let enabled = makeRegistry().definitions
        #expect(ChatViewModel.wireDefinitions(forAlias: "hermes3-8b-4bit", enabled: enabled).isEmpty)
        #expect(ChatViewModel.wireDefinitions(forAlias: "bonsai-8b-2bit", enabled: enabled).isEmpty)
        // An alias outside the broken list passes every enabled tool through,
        // whatever the registry's size (read_document joined the three web
        // tools in this PR).
        #expect(
            ChatViewModel.wireDefinitions(forAlias: "qwen3.5-4b-4bit", enabled: enabled).count
                == enabled.count
        )
    }

    // MARK: - Ambient guidance

    private static func toolTurn(_ text: String = "weather in Tokyo?") -> [ChatMessage] {
        [
            ChatMessage(role: .system, content: "[CURRENT DATE]\nToday is Friday.", status: .complete),
            ChatMessage(role: .user, content: text, status: .complete),
            ChatMessage(role: .assistant, toolCalls: [ToolCall(id: "w1", name: "weather", arguments: "{}")]),
            ChatMessage(role: .tool, content: "{\"temp_c\": 29.2}", toolCallID: "w1"),
        ]
    }

    @Test("The anti-confabulation guidance rides the newest user row once a tool result is in play")
    func ambientGuidanceGatedOnToolResult() {
        let turn = Self.toolTurn()
        let stamped = ChatViewModel.stampingToolGuidance(on: turn, toolsAdvertised: true)
        // The system row is untouched: the guidance must never move the head
        // of the prompt, or the engine re-prefills the whole conversation.
        #expect(stamped[0] == turn[0])
        #expect(stamped[1].content == "weather in Tokyo?")
        #expect(stamped[1].wireSuffix == ChatViewModel.toolGuidance)
        #expect(stamped[1].modelContent.hasSuffix(ChatViewModel.toolGuidance))
        #expect(stamped[2] == turn[2])
        #expect(stamped[3] == turn[3])
        #expect(ChatViewModel.toolGuidance.contains("state the reason written in that result"))
        #expect(ChatViewModel.toolGuidance.contains("never claim the tool lacks a capability"))
    }

    @Test("A tool merely being advertised does not summon the preamble (#1549)")
    func ambientGuidanceStaysHomeUntilThereIsAResult() {
        // The regression this guards is the whole first-turn experience. The
        // built-in web tools are advertised by default, so before #1549 every
        // opening message shipped a preamble telling the model that anything
        // absent from "the tool result" was unknown to it — with no tool result
        // in context. The shipped starter answered "I don't have access to
        // current or external data" to *what is the capital of France?*.
        let plain = Array(Self.toolTurn().prefix(2))
        #expect(ChatViewModel.stampingToolGuidance(on: plain, toolsAdvertised: true) == plain)
    }

    @Test("A stale tool result cannot re-bind the model once the tool is gone")
    func ambientGuidanceNeedsTheToolStillAdvertised() {
        // A transcript keeps its ``.tool`` rows after the user disables the
        // tool in Settings. Re-asserting "your only source of truth is the tool
        // result" would then pin the model to a result it can no longer
        // refresh, which is the same failure wearing older evidence.
        let turn = Self.toolTurn()
        #expect(ChatViewModel.stampingToolGuidance(on: turn, toolsAdvertised: false) == turn)
    }

    @Test("A tool result only counts for the turn it belongs to")
    func toolResultScopedToTheCurrentTurn() {
        func msg(_ role: ChatMessage.Role, _ text: String) -> ChatMessage {
            ChatMessage(role: role, content: text, status: .complete)
        }

        // Turn 1 used a tool; turn 2 is an ordinary question. Scanning the
        // whole transcript would re-arm the preamble here and reproduce
        // #1549 from the second question onward.
        let laterPlainTurn = [
            msg(.user, "weather in Tokyo?"),
            msg(.assistant, ""),
            msg(.tool, "{\"temp_c\": 29.2}"),
            msg(.assistant, "It's 29.2°C in Tokyo."),
            msg(.user, "what is the capital of France?"),
        ]
        #expect(!ChatViewModel.carriesToolResultForThisTurn(laterPlainTurn))

        // The round right after a tool returned — this is what the preamble
        // exists for, so it must still count.
        let midToolLoop = [
            msg(.user, "weather in Tokyo?"),
            msg(.assistant, ""),
            msg(.tool, "{\"temp_c\": 29.2}"),
        ]
        #expect(ChatViewModel.carriesToolResultForThisTurn(midToolLoop))

        // A transcript with no user row at all (defensive: the tool loop
        // never produces one, but the helper must not crash or misread it).
        #expect(!ChatViewModel.carriesToolResultForThisTurn([]))
        #expect(ChatViewModel.carriesToolResultForThisTurn([msg(.tool, "{}")]))
    }

    @Test("The guidance never adds a row: the message count and the system row are unchanged")
    func ambientGuidanceNeverAddsARow() {
        // Two competing system messages is a documented chat-template
        // foot-gun, and a new row anywhere would move the prompt bytes behind
        // it. The guidance only extends the newest user row's wire trailer.
        let turn = Self.toolTurn()
        let stamped = ChatViewModel.stampingToolGuidance(on: turn, toolsAdvertised: true)
        #expect(stamped.count == turn.count)
        #expect(stamped.map(\.role) == turn.map(\.role))
        #expect(stamped.filter { $0.role == .system } == turn.filter { $0.role == .system })
    }
}

@MainActor
private final class DeclinedBrowseRegistry: ToolRegistry {
    let definitions = [BrowseTool.definition]

    func run(_ call: ToolCall) async -> ToolCallResult {
        ToolCallResult(
            toolCallID: call.id,
            content: "User declined to open this page.",
            isError: true,
            failureKind: .userDeclined,
            executed: false
        )
    }
}

/// Keychain double so the web-search key tests never touch the real login
/// keychain (which would prompt, and would leak between runs).
private final class InMemoryKeychain: KeychainStoring, @unchecked Sendable {
    private let lock = NSLock()
    private var store: [String: String] = [:]

    func read(account: String) -> String? {
        lock.lock(); defer { lock.unlock() }
        return store[account]
    }

    func write(account: String, secret: String) -> Bool {
        lock.lock(); defer { lock.unlock() }
        store[account] = secret
        return true
    }

    func delete(account: String) -> Bool {
        lock.lock(); defer { lock.unlock() }
        store.removeValue(forKey: account)
        return true
    }
    @Test("The rejected-key echo is bounded by scalars, not characters")
    func rejectionEchoBoundsCombiningMarks() throws {
        // Adversarial review round 11 (codex, blocking): `prefix(80)` counts
        // grapheme clusters, so one cluster carrying thousands of combining
        // marks was not bounded at all.
        let zalgo = "k" + String(repeating: "\u{0301}", count: 20_000)
        #expect(zalgo.count == 1)   // one grapheme cluster, 20_001 scalars
        switch NativeToolCallExecutor.normalize(
            ToolCall(
                id: "document_1",
                name: "read_document",
                arguments: "{\"document_id\":\"00000000-0000-0000-0000-000000000000\",\"\(zalgo)\":1}"
            ),
            for: ReadDocumentTool.definition
        ) {
        case .success:
            Issue.record("an unknown key must not normalize")
        case .failure(let rejection):
            // 80 scalars of key plus the surrounding copy — not 20 001.
            #expect(rejection.reason.unicodeScalars.count < 400)
            #expect(rejection.reason.contains("unknown argument(s)"))
        }
    }

}
