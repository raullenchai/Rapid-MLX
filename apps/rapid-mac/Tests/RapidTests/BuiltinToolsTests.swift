import Foundation
import Testing
@testable import Rapid

/// Contracts for the built-in tool surface: what the registry exposes, what
/// reaches the wire, and what is refused at dispatch.
///
/// The three shipped tools (``web_search``, ``browse``, ``weather``) are all
/// network-facing, so the gates below are load-bearing rather than cosmetic:
/// a tool stripped from the request body can STILL be named by a malformed
/// model, and only the dispatch-side refusal stops it running.
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

    @Test("Registry exposes exactly web_search, browse, and weather")
    func registryDefinitions() {
        let names = makeRegistry().definitions.map { $0.function.name }
        #expect(names == ["web_search", "browse", "weather"])
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

    // MARK: - Per-tool enable/disable

    @Test("A tool toggled off is stripped from the definitions sent to the model")
    func disabledToolIsStrippedFromWire() {
        let vm = ChatViewModel(tools: makeRegistry(), toolDefaults: freshDefaults())
        #expect(vm.enabledDefinitions.count == 3)
        vm.setToolEnabled("browse", false)
        #expect(!vm.enabledDefinitions.contains { $0.function.name == "browse" })
        #expect(vm.enabledDefinitions.count == 2)
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
        #expect(ChatViewModel.wireDefinitions(forAlias: "qwen3.5-4b-4bit", enabled: enabled).count == 3)
    }

    // MARK: - Ambient guidance

    @Test("The anti-confabulation preamble rides along once a tool result is in play")
    func ambientGuidanceGatedOnToolResult() {
        let withResult = ChatViewModel.ambientSystemMessages(
            historyOpensWithSystem: false,
            toolsAdvertised: true,
            toolResultPresent: true
        )
        #expect(withResult.count == 1)
        #expect(withResult.first?.role == .system)
        #expect(withResult.first?.content == ChatViewModel.toolGuidancePreamble)
    }

    @Test("A tool merely being advertised does not summon the preamble (#1549)")
    func ambientGuidanceStaysHomeUntilThereIsAResult() {
        // The regression this guards is the whole first-turn experience. The
        // built-in web tools are advertised by default, so before #1549 every
        // opening message shipped a preamble telling the model that anything
        // absent from "the tool result" was unknown to it — with no tool result
        // in context. The shipped starter answered "I don't have access to
        // current or external data" to *what is the capital of France?*.
        #expect(ChatViewModel.ambientSystemMessages(
            historyOpensWithSystem: false,
            toolsAdvertised: true,
            toolResultPresent: false
        ).isEmpty)
    }

    @Test("A stale tool result cannot re-bind the model once the tool is gone")
    func ambientGuidanceNeedsTheToolStillAdvertised() {
        // A transcript keeps its ``.tool`` rows after the user disables the
        // tool in Settings. Re-asserting "your only source of truth is the tool
        // result" would then pin the model to a result it can no longer
        // refresh, which is the same failure wearing older evidence.
        #expect(ChatViewModel.ambientSystemMessages(
            historyOpensWithSystem: false,
            toolsAdvertised: false,
            toolResultPresent: true
        ).isEmpty)
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

    @Test("No second system row is injected when the transcript already opens with one")
    func ambientGuidanceDefersToExistingSystemRow() {
        // Two competing system messages is a documented chat-template foot-gun.
        #expect(ChatViewModel.ambientSystemMessages(
            historyOpensWithSystem: true,
            toolsAdvertised: true,
            toolResultPresent: true
        ).isEmpty)
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
}
