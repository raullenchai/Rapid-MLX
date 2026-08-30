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

    @Test("The first exchange is recognisable")
    func firstExchangeIsRecognisable() {
        #expect(ChatViewModel.isFirstExchange([ask, answer()]))
        #expect(!ChatViewModel.isFirstExchange([ask, answer(), ask, answer()]))
        #expect(!ChatViewModel.isFirstExchange([]))
    }

    // MARK: - Which lane will batch us

    private func profile(
        id: String = "m", servingLane: String?, modality: String = "text"
    ) -> ServerModelProfile {
        ServerModelProfile(
            id: id,
            servingLane: servingLane,
            modality: modality
        )
    }

    @Test("The batched text lane allows background work")
    func textLaneAllows() {
        #expect(ChatViewModel.laneAllowsBackgroundWork(
            profile(servingLane: "text"), alias: "m"
        ))
    }

    /// The `--mllm` lane runs one request at a time, so anything we send
    /// there is not batched alongside the reader's turn — it is queued in
    /// front of their next one.
    ///
    /// The server still reports `modality: text` for this payload. The live
    /// `serving_lane` is what distinguishes its execution path.
    @Test("A serialised vision lane with text modality refuses")
    func visionLaneRefuses() throws {
        let payload = #"{"id":"m","serving_lane":"vision","modality":"text"}"#
        let liveProfile = try JSONDecoder().decode(
            ServerModelProfile.self, from: Data(payload.utf8)
        )
        #expect(!ChatViewModel.laneAllowsBackgroundWork(
            liveProfile, alias: "m"
        ))
    }

    @Test("An older profile without a serving lane allows")
    func missingServingLaneAllows() {
        #expect(ChatViewModel.laneAllowsBackgroundWork(
            profile(servingLane: nil), alias: "m"
        ))
    }

    /// A sidecar too old to report a live profile would otherwise disable both
    /// features permanently. The cost of guessing wrong there is bounded by
    /// max_tokens, the deadline, and cancel-on-send.
    @Test("An unknown model allows")
    func unknownModelAllows() {
        #expect(ChatViewModel.laneAllowsBackgroundWork(nil, alias: "m"))
        #expect(ChatViewModel.laneAllowsBackgroundWork(
            profile(id: "other", servingLane: "vision"), alias: "m"
        ))
    }

    @Test("A server transition invalidates the scheduled background target")
    func serverTransitionInvalidatesTarget() {
        let scheduled = BackgroundCompletionClient.Target(
            port: 8_000,
            bearer: "old-bearer",
            alias: "old-model"
        )

        #expect(ChatViewModel.revalidatedBackgroundTarget(
            scheduled,
            state: .ready(alias: "old-model"),
            activePort: 8_000,
            activeBearer: "old-bearer"
        ) == scheduled)

        // The transition happened after scheduling but before the task ran.
        // A stale alias must not be paired with the replacement endpoint.
        #expect(ChatViewModel.revalidatedBackgroundTarget(
            scheduled,
            state: .ready(alias: "new-model"),
            activePort: 8_001,
            activeBearer: "new-bearer"
        ) == nil)

        // A restart of the same alias still invalidates a changed endpoint or
        // credential; the task cannot silently retarget itself.
        #expect(ChatViewModel.revalidatedBackgroundTarget(
            scheduled,
            state: .ready(alias: "old-model"),
            activePort: 8_001,
            activeBearer: "new-bearer"
        ) == nil)
    }

    @Test("Follow-ups publish without waiting for a stalled title")
    func followUpsPublishBeforeTitle() async {
        let titleGate = BackgroundAssistTestGate()
        var publications: [String] = []

        let delivery = Task { @MainActor in
            await ChatViewModel.deliverBackgroundReplies(
                title: {
                    await titleGate.wait()
                    return "A generated title"
                },
                followUp: { "One?\nTwo?\nThree?" },
                onFollowUp: { reply in
                    publications.append("follow-up:\(reply ?? "nil")")
                    return true
                },
                onTitle: { reply in
                    publications.append("title:\(reply ?? "nil")")
                }
            )
        }

        // Give the immediately-completing follow-up child repeated chances
        // to publish while the title child remains deliberately suspended.
        for _ in 0..<100 where publications.isEmpty { await Task.yield() }
        #expect(publications == ["follow-up:One?\nTwo?\nThree?"])

        await titleGate.open()
        await delivery.value
        #expect(publications == [
            "follow-up:One?\nTwo?\nThree?",
            "title:A generated title",
        ])
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
        // Both halves, or the rail would never mount: `ChatView` keys on the
        // anchor, so a `.ready` state with no anchor is a combination the
        // screen can never show.
        #expect(model.followUpAnchorID == last.id)
    }

    /// Starts from `.ready`, because the previous version of this test
    /// asserted `.idle` on a model that was already `.idle` — it passed with
    /// the entire body of `publishFollowUps` replaced by a no-op.
    @Test("Nothing parseable puts the rail away")
    func unparseableSuggestionsClearTheRail() {
        let model = ChatViewModel()
        let last = answer()
        model.devSeedMessages([ask, last])

        model.publishFollowUps("A one?\nB two?\nC three?", anchoredTo: last.id, excluding: "")
        #expect(model.followUp == .ready(["A one?", "B two?", "C three?"]))
        #expect(model.followUpAnchorID == last.id)

        model.publishFollowUps("Here are some ideas for you.", anchoredTo: last.id, excluding: "")
        #expect(model.followUp == .idle)
        // The anchor goes too: the rail is mounted on it and holds 40pt
        // whatever it shows, so leaving it set is an empty band under the
        // answer until the next turn.
        #expect(model.followUpAnchorID == nil)
    }

    @Test("A nil reply puts the rail away")
    func nilReplyClearsTheRail() {
        let model = ChatViewModel()
        let last = answer()
        model.devSeedMessages([ask, last])
        model.publishFollowUps("A one?\nB two?\nC three?", anchoredTo: last.id, excluding: "")
        #expect(model.followUp != .idle)
        model.publishFollowUps(nil, anchoredTo: last.id, excluding: "")
        #expect(model.followUp == .idle)
        #expect(model.followUpAnchorID == nil)
    }

    // MARK: - Cancellation

    /// Cancelling the assist has to put the rail away *synchronously*.
    ///
    /// A cancelled assist exits without publishing, so nothing downstream
    /// clears the anchor — and ``ChatView`` mounts the rail on the anchor
    /// alone, at a fixed height whatever it shows. The task's own
    /// continuation does clear it, but only once it gets to run, and
    /// ``stop()`` never cleared it at all.
    ///
    /// The routes are listed rather than folded into one case because the
    /// defect was that three of the six cancellation sites remembered to put
    /// the rail away and three did not — which a single-route test cannot
    /// see.
    @Test("Cancelling puts the rail away", arguments: [
        "stop", "stopAndPersist", "newConversation", "delete",
    ])
    func cancellingClearsTheRail(_ route: String) {
        let model = ChatViewModel()
        let last = answer()
        model.devSeedMessages([ask, last])
        model.publishFollowUps("A one?\nB two?\nC three?", anchoredTo: last.id, excluding: "")
        // The premise: there is a rail to strand. Without it the assertions
        // below would hold on a model that never had one.
        #expect(model.followUp == .ready(["A one?", "B two?", "C three?"]))
        #expect(model.followUpAnchorID == last.id)

        switch route {
        case "stop": model.stop()
        case "stopAndPersist": model.stopAndPersist()
        case "newConversation": model.newConversation()
        default: #expect(model.deleteMessage(id: last.id))
        }

        #expect(model.followUp == .idle, "\(route) left the rail showing")
        #expect(model.followUpAnchorID == nil, "\(route) left the anchor set")
    }

    @Test("A stale cancellation cannot clear a newer turn's rail")
    func staleCancellationPreservesNewerRail() {
        let model = ChatViewModel()
        let staleAnswer = answer()
        let currentAnswer = answer()
        model.devSeedMessages([ask, currentAnswer])
        model.publishFollowUps(
            "Current one?\nCurrent two?\nCurrent three?",
            anchoredTo: currentAnswer.id,
            excluding: ""
        )

        // The older assist finally observes cancellation after the current
        // turn has already published. Its cleanup no longer owns this rail.
        model.clearFollowUps(anchoredTo: staleAnswer.id)
        #expect(model.followUp == .ready([
            "Current one?", "Current two?", "Current three?",
        ]))
        #expect(model.followUpAnchorID == currentAnswer.id)

        // The matching assist still tears down its own rail.
        model.clearFollowUps(anchoredTo: currentAnswer.id)
        #expect(model.followUp == .idle)
        #expect(model.followUpAnchorID == nil)
    }

    @Test("Replay cancels suggestions for the replaced answer", arguments: [
        "regenerate", "retry",
    ])
    func replayCancelsReplacedSuggestions(_ route: String) {
        let model = ChatViewModel(persistsConversations: false)
        let last = answer()
        model.devSeedMessages([ask, last])
        model.publishFollowUps(
            "Old one?\nOld two?\nOld three?",
            anchoredTo: last.id,
            excluding: ""
        )

        if route == "regenerate" {
            model.regenerateLast(alias: "test-model")
        } else {
            #expect(model.retryAssistantMessage(id: last.id, alias: "test-model"))
        }
        defer { model.stopAndPersist() }

        #expect(model.followUp == .idle)
        #expect(model.followUpAnchorID == nil)
    }
}

private actor BackgroundAssistTestGate {
    private var isOpen = false
    private var waiter: CheckedContinuation<Void, Never>?

    func wait() async {
        if isOpen { return }
        await withCheckedContinuation { waiter = $0 }
    }

    func open() {
        isOpen = true
        waiter?.resume()
        waiter = nil
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

/// Whether a conversation still wants a title.
///
/// The gate used to be "is this the first exchange", which forfeits the one
/// chance permanently: `send` cancels the outstanding call, so a reader who
/// replies within the fifteen-second deadline leaves the conversation on its
/// derived title forever. `hasGeneratedTitle` already means "a title has
/// landed", so gating on that keeps the once-per-conversation promise while
/// letting a cancelled or failed attempt be retried on the next turn.
@MainActor
@Suite("Title retry")
struct TitleRetryTests {

    private func isolatedStore() throws -> (root: URL, file: URL) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-title-retry-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return (root, root.appendingPathComponent("conversations.json"))
    }

    @Test("A conversation whose first attempt failed is still eligible later")
    func retriedAfterAFailedFirstAttempt() throws {
        let store = try isolatedStore()
        defer { try? FileManager.default.removeItem(at: store.root) }
        let model = ChatViewModel(conversationStoreURL: store.file)
        model.send("first question", alias: "test-model")
        model.stopAndPersist()
        let id = model.activeConversationID

        // The first attempt produced nothing — cancelled, or unparseable.
        #expect(model.conversations.first { $0.id == id }?.hasGeneratedTitle == false)

        // A second turn. The conversation is no longer a first exchange, but
        // it still has no title, so it is still eligible.
        model.send("second question", alias: "test-model")
        model.stopAndPersist()
        model.applyGeneratedTitle("Euler's theorem", to: id)
        #expect(model.conversations.first { $0.id == id }?.title == "Euler's theorem")
    }

    /// And once one lands, it is never asked for again.
    @Test("A titled conversation stops being eligible")
    func notRetriedOnceTitled() throws {
        let store = try isolatedStore()
        defer { try? FileManager.default.removeItem(at: store.root) }
        let model = ChatViewModel(conversationStoreURL: store.file)
        model.send("first question", alias: "test-model")
        model.stopAndPersist()
        let id = model.activeConversationID

        model.applyGeneratedTitle("First name", to: id)
        model.applyGeneratedTitle("Second name", to: id)
        #expect(model.conversations.first { $0.id == id }?.title == "First name")
    }
}
