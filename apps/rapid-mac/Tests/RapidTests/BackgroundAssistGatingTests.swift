import Foundation
import Testing
@testable import Rapid

/// Who owns a conversation's name, and what happens when two owners disagree.
///
/// The derivation in ``ChatViewModel/persistActive`` re-reads the first user
/// turn on *every* save, so any title that did not come from it needs a flag
/// to survive the next streamed token. There are now two such flags with
/// deliberately different meanings, and the interesting behaviour is all in
/// how they interact.
@MainActor
@Suite("Generated conversation titles")
struct GeneratedTitleTests {

    private func isolatedStoreURL() throws -> (root: URL, file: URL) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-generated-title-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return (root, root.appendingPathComponent("conversations.json"))
    }

    /// One persisted single-turn conversation, the same way
    /// ``SidebarConversationActionsTests`` builds one: the snapshot into
    /// `conversations` hangs off the end of a turn, and `stopAndPersist`
    /// cancels the unroutable request before it can stream.
    private func seededModel(_ store: URL) -> ChatViewModel {
        let model = ChatViewModel(conversationStoreURL: store)
        model.send("how does euler's theorem work", alias: "test-model")
        model.stopAndPersist()
        return model
    }

    // MARK: - The flag earns its keep

    /// The whole reason ``hasGeneratedTitle`` exists. Without it the next
    /// persisted turn reverts a generated title to the opening words of the
    /// user's prompt — the same regression ``hasCustomTitle`` was added for.
    @Test("A generated title survives later saves")
    func generatedTitleSurvivesLaterPersists() throws {
        let store = try isolatedStoreURL()
        defer { try? FileManager.default.removeItem(at: store.root) }
        let model = seededModel(store.file)
        let id = model.activeConversationID

        model.applyGeneratedTitle("Euler's theorem", to: id)
        #expect(model.conversations.first { $0.id == id }?.title == "Euler's theorem")

        model.send("and fermat?", alias: "test-model")
        model.stopAndPersist()

        #expect(model.conversations.first { $0.id == id }?.title == "Euler's theorem")
    }

    // MARK: - The user always wins

    @Test("A rename beats a title that arrives afterwards")
    func renameBeatsLaterGeneratedTitle() throws {
        let store = try isolatedStoreURL()
        defer { try? FileManager.default.removeItem(at: store.root) }
        let model = seededModel(store.file)
        let id = model.activeConversationID

        #expect(model.renameConversation(id, to: "My notes"))
        model.applyGeneratedTitle("Euler's theorem", to: id)

        #expect(model.conversations.first { $0.id == id }?.title == "My notes")
    }

    @Test("A rename beats a title that arrived before it")
    func renameBeatsEarlierGeneratedTitle() throws {
        let store = try isolatedStoreURL()
        defer { try? FileManager.default.removeItem(at: store.root) }
        let model = seededModel(store.file)
        let id = model.activeConversationID

        model.applyGeneratedTitle("Euler's theorem", to: id)
        #expect(model.renameConversation(id, to: "My notes"))

        model.send("more", alias: "test-model")
        model.stopAndPersist()
        #expect(model.conversations.first { $0.id == id }?.title == "My notes")
    }

    // MARK: - One shot, and quiet about it

    @Test("A second generated title is ignored")
    func secondGeneratedTitleIsIgnored() throws {
        let store = try isolatedStoreURL()
        defer { try? FileManager.default.removeItem(at: store.root) }
        let model = seededModel(store.file)
        let id = model.activeConversationID

        model.applyGeneratedTitle("First name", to: id)
        model.applyGeneratedTitle("Second name", to: id)

        #expect(model.conversations.first { $0.id == id }?.title == "First name")
    }

    @Test("Unparseable output leaves the derived title alone")
    func unparseableTitleIsIgnored() throws {
        let store = try isolatedStoreURL()
        defer { try? FileManager.default.removeItem(at: store.root) }
        let model = seededModel(store.file)
        let id = model.activeConversationID
        let derived = model.conversations.first { $0.id == id }?.title

        model.applyGeneratedTitle("Sure! Here's a title for your conversation.", to: id)

        #expect(model.conversations.first { $0.id == id }?.title == derived)
        // And the door stays open — this was not the one shot.
        model.applyGeneratedTitle("Euler's theorem", to: id)
        #expect(model.conversations.first { $0.id == id }?.title == "Euler's theorem")
    }

    @Test("A title for a conversation that no longer exists is dropped")
    func titleForMissingConversationIsDropped() throws {
        let store = try isolatedStoreURL()
        defer { try? FileManager.default.removeItem(at: store.root) }
        let model = seededModel(store.file)
        let before = model.conversations
        model.applyGeneratedTitle("Euler's theorem", to: UUID())
        #expect(model.conversations == before)
    }

    // MARK: - Naming is not working on it

    /// A rename does not bump `updatedAt`, and neither does this — otherwise
    /// the sidebar reshuffles a second after the answer lands, moving a row
    /// the reader is looking at for a reason they cannot see.
    @Test("A generated title does not reorder the sidebar")
    func generatedTitleDoesNotTouchUpdatedAt() throws {
        let store = try isolatedStoreURL()
        defer { try? FileManager.default.removeItem(at: store.root) }
        let model = seededModel(store.file)
        let id = model.activeConversationID
        let before = model.conversations.first { $0.id == id }?.updatedAt
        let order = model.conversations.map(\.id)

        model.applyGeneratedTitle("Euler's theorem", to: id)

        #expect(model.conversations.first { $0.id == id }?.updatedAt == before)
        #expect(model.conversations.map(\.id) == order)
    }

    // MARK: - On-disk migration

    /// History written before this shipped must decode, and must decode as
    /// "still derived" so the row keeps behaving the way it always has.
    @Test("History without the key decodes as not-yet-generated")
    func legacyHistoryDecodes() throws {
        let json = """
            [{"id":"\(UUID().uuidString)","title":"Old chat","messages":[],
              "createdAt":0,"updatedAt":0}]
            """
        let decoded = try JSONDecoder().decode(
            [ChatConversation].self, from: Data(json.utf8)
        )
        #expect(decoded.first?.hasGeneratedTitle == false)
        #expect(decoded.first?.hasCustomTitle == false)
    }

    @Test("The flag round-trips")
    func flagRoundTrips() throws {
        let original = ChatConversation(
            id: UUID(), title: "Euler's theorem", messages: [],
            createdAt: Date(), updatedAt: Date(), hasGeneratedTitle: true
        )
        let decoded = try JSONDecoder().decode(
            ChatConversation.self, from: JSONEncoder().encode(original)
        )
        #expect(decoded.hasGeneratedTitle)
        #expect(!decoded.hasCustomTitle)
    }
}

/// When the app is allowed to ask the model something on its own account,
/// and when a reply that arrives late is thrown away.
@MainActor
@Suite("Background assist gating")
struct BackgroundAssistGatingTests {

    private func answer(
        _ content: String = "Because it scatters blue light.",
        status: ChatMessage.Status = .complete,
        errorMessage: String? = nil,
        toolCalls: [ToolCall]? = nil
    ) -> ChatMessage {
        ChatMessage(
            role: .assistant, content: content, status: status,
            errorMessage: errorMessage, toolCalls: toolCalls
        )
    }

    private let ask = ChatMessage(role: .user, content: "why is the sky blue")

    // MARK: - Which turns qualify

    @Test("A settled answer qualifies")
    func settledAnswerQualifies() {
        let turn = ChatViewModel.settledTurn([ask, answer()])
        #expect(turn?.lastUserText == "why is the sky blue")
    }

    @Test("A turn that does not qualify", arguments: [
        "streaming", "failed", "stopped", "empty", "tool-shell", "no-user", "user-last", "empty-transcript",
    ])
    func nonQualifyingTurns(_ shape: String) {
        let messages: [ChatMessage]
        switch shape {
        case "streaming":  messages = [ask, answer(status: .streaming)]
        case "failed":     messages = [ask, answer(status: .failed)]
        // The marker ``finaliseCancellation`` writes. Suggesting more of an
        // answer the reader just stopped is the opposite of listening.
        case "stopped":    messages = [ask, answer(errorMessage: "Stopped.")]
        case "empty":      messages = [ask, answer("   ")]
        case "tool-shell": messages = [ask, answer("", toolCalls: [
            ToolCall(id: "1", name: "web_search", arguments: "{}"),
        ])]
        case "no-user":    messages = [answer()]
        case "user-last":  messages = [ask, answer(), ask]
        default:           messages = []
        }
        #expect(ChatViewModel.settledTurn(messages) == nil, "\(shape) should not qualify")
    }

    // MARK: - Which turn gets a title

    @Test("Only the first exchange is titled")
    func onlyFirstExchangeIsTitled() {
        #expect(ChatViewModel.isFirstExchange([ask, answer()]))
        #expect(!ChatViewModel.isFirstExchange([ask, answer(), ask, answer()]))
        #expect(!ChatViewModel.isFirstExchange([]))
    }

    // MARK: - Which lane will batch us

    private func snapshot(modality: String, active: Int) -> ModelResidencySnapshot {
        ModelResidencySnapshot(
            memoryLimitBytes: 0, memoryUsedBytes: 0, memoryAvailableBytes: nil,
            idleTTLSeconds: 0, loadsTotal: 0, evictionsTotal: 0,
            models: [ResidentModelStatus(
                id: "m", modelPath: "m", aliases: ["m"], modality: modality,
                state: "ready", pinned: false, primary: true,
                activeRequests: active, estimatedBytes: 0, measuredBytes: nil,
                idleSeconds: 0
            )]
        )
    }

    @Test("The batched text lane allows background work")
    func textLaneAllows() {
        #expect(ChatViewModel.laneAllowsBackgroundWork(snapshot(modality: "text", active: 0), alias: "m"))
    }

    /// The `--mllm` lane runs one request at a time, so anything we send
    /// there is not batched alongside the reader's turn — it is queued in
    /// front of their next one.
    @Test("A serialised lane refuses")
    func visionLaneRefuses() {
        #expect(!ChatViewModel.laneAllowsBackgroundWork(snapshot(modality: "vision", active: 0), alias: "m"))
    }

    @Test("A busy model refuses")
    func busyModelRefuses() {
        #expect(!ChatViewModel.laneAllowsBackgroundWork(snapshot(modality: "text", active: 1), alias: "m"))
    }

    /// A sidecar too old to report residency would otherwise disable both
    /// features permanently. The cost of guessing wrong there is bounded by
    /// max_tokens, the deadline, and cancel-on-send.
    @Test("An unknown model allows")
    func unknownModelAllows() {
        #expect(ChatViewModel.laneAllowsBackgroundWork(.empty, alias: "m"))
        #expect(ChatViewModel.laneAllowsBackgroundWork(snapshot(modality: "text", active: 0), alias: "other"))
    }

    // MARK: - Late replies

    @Test("Suggestions for a turn that is no longer last are dropped")
    func staleSuggestionsAreDropped() {
        let model = ChatViewModel()
        model.devSeedMessages([ask, answer()])
        // Anchored to a row that is not the last one any more.
        model.publishFollowUps("A?\nB?\nC?", anchoredTo: UUID(), excluding: "")
        #expect(model.followUp == .idle)
        #expect(model.followUpAnchorID == nil)
    }

    @Test("Suggestions for the visible turn are published")
    func freshSuggestionsArePublished() {
        let model = ChatViewModel()
        let last = answer()
        model.devSeedMessages([ask, last])
        model.publishFollowUps("A one?\nB two?\nC three?", anchoredTo: last.id, excluding: "")
        #expect(model.followUp == .ready(["A one?", "B two?", "C three?"]))
    }

    @Test("Nothing parseable leaves the rail empty")
    func unparseableSuggestionsLeaveItEmpty() {
        let model = ChatViewModel()
        let last = answer()
        model.devSeedMessages([ask, last])
        model.publishFollowUps("Here are some ideas for you.", anchoredTo: last.id, excluding: "")
        #expect(model.followUp == .idle)
        model.publishFollowUps(nil, anchoredTo: last.id, excluding: "")
        #expect(model.followUp == .idle)
    }
}

/// Shape rules a SwiftUI body cannot be asked about from this target
/// (ViewInspector is not linked — #1492), pinned by reading the source.
@MainActor
@Suite("Follow-up rail shape")
struct FollowUpSuggestionRailSourceTests {

    /// Comments stripped, because several of these rules are stated in prose
    /// in the very file they govern — a naive grep for "brandPrimary" finds
    /// the comment explaining why it is not used.
    private func source(_ path: String) throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // RapidTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // rapid-mac
            .appendingPathComponent(path)
        return CapabilityChipRenderGateSourceGuardTests.stripCommentsAndWhitespace(
            try String(contentsOf: url, encoding: .utf8)
        )
    }

    private let rail = "Sources/Rapid/UI/FollowUpSuggestionRail.swift"

    /// The no-yank contract, in one line of source.
    ///
    /// The rail must hold a constant height in every state, because
    /// ``TranscriptScrollPositionProbe`` follows any document growth while
    /// the reader is pinned and its release valve is gated on `isStreaming`
    /// — already false by the time chips could arrive. If filling the rail
    /// ever changes its height, a pinned reader gets pulled down with
    /// nothing to stop them.
    @Test("The rail's height does not depend on its contents")
    func heightIsConstant() throws {
        let text = try source(rail)
        #expect(text.contains(".frame(height:Self.reservedHeight"))
        // One line, so the height cannot depend on the text a model wrote.
        #expect(text.contains(".lineLimit(1)"))
        #expect(text.contains("ScrollView(.horizontal"))
    }

    /// The composer's send disc is this surface's only amber moment
    /// (`CoreWorkspaceVisualFoundationTests.composerControlsAreIntact`).
    @Test("Chips do not compete with the send button")
    func chipsAreNotBranded() throws {
        #expect(!(try source(rail).contains("brandPrimary")))
    }

    /// Identifiers must not carry the `ChatView.Message.` prefix:
    /// `gui-golden-flows.sh`'s `transcript_only` scopes its slice by it, and
    /// a chip inside that slice would freeze model-authored text into a
    /// baseline. They must also key on the index, never the label.
    @Test("Chip identifiers stay out of the transcript slice")
    func identifiersAreOutOfSlice() throws {
        let text = try source(rail)
        #expect(text.contains(#".accessibilityIdentifier("ChatView.FollowUp.\(index)")"#))
        #expect(!text.contains("ChatView.Message."))
        #expect(!text.contains("accessibilityIdentifier(question"))
    }

    /// Anything re-entering send from the view answers to the readiness gate
    /// — `ChatViewModel.send` carries none of its own.
    @Test("A chip send answers to the readiness gate")
    func chipSendIsGated() throws {
        let text = try source("Sources/Rapid/UI/ChatView.swift")
        guard let start = text.range(of: "privatefuncsendSuggestion(") else {
            Issue.record("sendSuggestion is gone")
            return
        }
        let body = text[start.lowerBound...].prefix(600)
        #expect(body.contains("acknowledgeIfNotReady()"))
    }

    /// Both background calls must keep thinking off and tools out: a hybrid
    /// model with thinking on spends the whole budget on a reasoning trace
    /// and returns empty content, and a call advertising tools can finish
    /// with `tool_calls` and no text.
    @Test("Background requests stay cheap and plain")
    func backgroundRequestIsPlain() throws {
        let text = try source("Sources/Rapid/Chat/BackgroundCompletionClient.swift")
        #expect(text.contains("enableThinking:false"))
        #expect(text.contains("tools:nil"))
    }
}
