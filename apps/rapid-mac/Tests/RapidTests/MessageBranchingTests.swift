import Foundation
import Testing
@testable import Rapid

/// End-to-end cover for branching through the view model.
///
/// ``MessageTreeTests`` pins the tree arithmetic in isolation; these tests
/// exercise the promise the feature actually makes to the user — that
/// regenerating an answer no longer destroys the one it replaced — by driving
/// the same entry points the buttons call.
@MainActor
@Suite("Message branching")
struct MessageBranchingTests {
    private func seededModel() -> (ChatViewModel, first: ChatMessage, answer: ChatMessage) {
        let viewModel = ChatViewModel(persistsConversations: false)
        let user = ChatMessage(role: .user, content: "question")
        let assistant = ChatMessage(role: .assistant, content: "first answer")
        viewModel.devSeedMessages([user, assistant])
        return (viewModel, user, assistant)
    }

    @Test("Regenerating keeps the replaced answer as a sibling")
    func regenerateKeepsTheOldAnswer() {
        // The whole point of the feature: before branching this transcript
        // slice was `messages.prefix(idx)` and "first answer" was gone.
        let (viewModel, _, answer) = seededModel()
        viewModel.regenerateLast(alias: "test-model")
        defer { viewModel.stopAndPersist() }

        // The visible path has rewound to the prompt and opened a new turn…
        #expect(viewModel.messages.count == 2)
        #expect(viewModel.messages[1].status == .streaming)
        #expect(!viewModel.messages.contains { $0.id == answer.id })
        // …and the replaced answer is still reachable as an alternative.
        let group = viewModel.siblings(of: viewModel.messages[1].id)
        #expect(group.count == 2)
        #expect(group.contains { $0.id == answer.id })
    }

    @Test("Retrying an older answer keeps it as a sibling")
    func retryKeepsTheOldAnswer() {
        let viewModel = ChatViewModel(persistsConversations: false)
        let user = ChatMessage(role: .user, content: "question")
        let answer = ChatMessage(role: .assistant, content: "first answer")
        let laterUser = ChatMessage(role: .user, content: "follow-up")
        let laterAnswer = ChatMessage(role: .assistant, content: "second answer")
        viewModel.devSeedMessages([user, answer, laterUser, laterAnswer])

        #expect(viewModel.retryAssistantMessage(id: answer.id, alias: "test-model"))
        defer { viewModel.stopAndPersist() }

        #expect(viewModel.messages.count == 2)
        let group = viewModel.siblings(of: viewModel.messages[1].id)
        #expect(group.contains { $0.id == answer.id })
    }

    @Test("Editing a prompt keeps the original wording as a sibling")
    func editKeepsTheOriginalPrompt() {
        let (viewModel, user, _) = seededModel()
        #expect(
            viewModel.editUserMessage(
                id: user.id, newContent: "revised question", alias: "test-model"
            )
        )
        defer { viewModel.stopAndPersist() }

        #expect(viewModel.messages[0].content == "revised question")
        let group = viewModel.siblings(of: viewModel.messages[0].id)
        #expect(group.count == 2)
        #expect(group.contains { $0.id == user.id })
    }

    @Test("A turn that was never regenerated reports no branch position")
    func unbranchedTurnHasNoPosition() {
        // What keeps the `‹ 1/1 ›` control off every row of an ordinary
        // transcript.
        let (viewModel, _, answer) = seededModel()
        #expect(viewModel.branchPosition(of: answer.id) == nil)
    }

    @Test("Branch position is 1-based and counts every alternative")
    func branchPositionCountsAlternatives() {
        let (viewModel, _, answer) = seededModel()
        viewModel.regenerateLast(alias: "test-model")
        viewModel.stopAndPersist()

        let replacement = viewModel.messages[1]
        let original = try? #require(viewModel.branchPosition(of: answer.id))
        let latest = try? #require(viewModel.branchPosition(of: replacement.id))
        #expect(original?.index == 1)
        #expect(original?.count == 2)
        #expect(latest?.index == 2)
        #expect(latest?.count == 2)
    }

    @Test("Switching back to an earlier branch restores its answer")
    func switchingRestoresTheEarlierAnswer() {
        let (viewModel, _, answer) = seededModel()
        viewModel.regenerateLast(alias: "test-model")
        viewModel.stopAndPersist()
        let replacement = viewModel.messages[1]

        #expect(viewModel.stepBranch(from: replacement.id, by: -1))
        #expect(viewModel.messages.last?.id == answer.id)
        #expect(viewModel.messages.last?.content == "first answer")

        // And forward again — the round trip is what makes the control
        // trustworthy rather than a one-way escape hatch.
        #expect(viewModel.stepBranch(from: answer.id, by: 1))
        #expect(viewModel.messages.last?.id == replacement.id)
    }

    @Test("Switching branches discards suggestions from the abandoned path")
    func switchingBranchesClearsFollowUps() {
        let (viewModel, _, answer) = seededModel()
        viewModel.regenerateLast(alias: "test-model")
        viewModel.stopAndPersist()
        let replacement = viewModel.messages[1]
        viewModel.publishFollowUps(
            "Old one?\nOld two?\nOld three?",
            anchoredTo: replacement.id,
            excluding: ""
        )

        #expect(viewModel.stepBranch(from: replacement.id, by: -1))
        #expect(viewModel.messages.last?.id == answer.id)
        #expect(viewModel.followUp == .idle)
        #expect(viewModel.followUpAnchorID == nil)
    }

    @Test("Stepping past either end of the group is refused")
    func steppingOffTheEndIsRefused() {
        // Bounded, not wrapping — matches the arrows' disabled states.
        let (viewModel, _, answer) = seededModel()
        viewModel.regenerateLast(alias: "test-model")
        viewModel.stopAndPersist()

        #expect(!viewModel.stepBranch(from: answer.id, by: -1))
        #expect(!viewModel.stepBranch(from: viewModel.messages[1].id, by: 1))
    }

    @Test("Deleting a turn removes its whole subtree")
    func deleteRemovesTheSubtree() {
        let viewModel = ChatViewModel(persistsConversations: false)
        let user = ChatMessage(role: .user, content: "question")
        let answer = ChatMessage(role: .assistant, content: "answer")
        let laterUser = ChatMessage(role: .user, content: "follow-up")
        viewModel.devSeedMessages([user, answer, laterUser])

        #expect(viewModel.deleteMessage(id: answer.id))
        #expect(viewModel.messages.map(\.id) == [user.id])
        #expect(viewModel.siblings(of: user.id).count == 1)
    }

    @Test("Deletion impact counts the hidden branches too")
    func deletionImpactCountsEveryBranch() {
        // What the confirmation dialog quotes. The user can see one answer
        // under this prompt; the count has to include the one they branched
        // away from, or the dialog understates what it is about to destroy.
        let (viewModel, _, answer) = seededModel()
        viewModel.regenerateLast(alias: "test-model")
        viewModel.stopAndPersist()

        // prompt + both answers
        #expect(viewModel.deletionImpact(of: viewModel.messages[0].id) == 3)
        #expect(viewModel.deletionImpact(of: answer.id) == 1)
        #expect(viewModel.deletionImpact(of: UUID()) == 0)
    }

    @Test("The delete confirmation names how many turns go")
    func deleteConfirmationCopyIsHonest() {
        #expect(ChatViewModel.deleteConfirmationTitle(turnCount: 1) == "Delete this message?")
        #expect(
            ChatViewModel.deleteConfirmationTitle(turnCount: 2)
                == "Delete this message and the 1 turn below it?"
        )
        #expect(
            ChatViewModel.deleteConfirmationTitle(turnCount: 4)
                == "Delete this message and the 3 turns below it?"
        )
    }

    @Test("Branches survive a save/load round trip")
    func branchesSurvivePersistence() throws {
        // The tree is split across `messages` + `branchedAway` in memory and
        // re-joined only on save, so persistence is where that split would
        // silently lose a branch.
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-branching-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("conversations.json")

        let viewModel = ChatViewModel(conversationStoreURL: file)
        let user = ChatMessage(role: .user, content: "question")
        let answer = ChatMessage(role: .assistant, content: "first answer")
        viewModel.devSeedMessages([user, answer])
        viewModel.regenerateLast(alias: "test-model")
        viewModel.stopAndPersist()
        ConversationStore.flush()

        let restored = ConversationStore.load(from: file)
        let conversation = try #require(restored.first)
        // The replaced answer is on disk, but in `branches` — NOT in
        // `messages`. That split is the downgrade contract: `messages` stays a
        // linear transcript so a build without branching renders the
        // conversation the user was looking at rather than a flattened pile of
        // alternatives it would then re-save in that scrambled order.
        #expect(!conversation.messages.contains { $0.id == answer.id })
        #expect(conversation.branches.contains { $0.id == answer.id })
        #expect(conversation.allMessages.contains { $0.id == answer.id })
        #expect(conversation.hasBranches)
        #expect(conversation.activePath.count == 2)
        #expect(conversation.activePath.last?.id != answer.id)
    }

    @Test("A pre-branching transcript loads as an ordinary linear chat")
    func legacyTranscriptStillRenders() throws {
        // The upgrade path: a conversations.json written before branching has
        // no parent links and no activeLeafID at all.
        let json = """
        [{"id":"\(UUID().uuidString)","title":"Old chat",
          "messages":[
            {"id":"\(UUID().uuidString)","role":"user","content":"q1","reasoning":"",
             "status":"complete","reasoningTruncated":false,"contentTruncated":false,
             "toolNotCalledFlagged":false,"toolCallArtifactSuppressed":false,
             "createdAt":0},
            {"id":"\(UUID().uuidString)","role":"assistant","content":"a1","reasoning":"",
             "status":"complete","reasoningTruncated":false,"contentTruncated":false,
             "toolNotCalledFlagged":false,"toolCallArtifactSuppressed":false,
             "createdAt":1}],
          "createdAt":0,"updatedAt":1}]
        """
        let decoded = try JSONDecoder().decode([ChatConversation].self, from: Data(json.utf8))
        let conversation = try #require(decoded.first)

        #expect(conversation.activePath.map(\.content) == ["q1", "a1"])
        #expect(!conversation.hasBranches)
    }

    @Test("A never-branched save carries no conversation-level branching keys")
    func neverBranchedSaveOmitsBranchingKeys() throws {
        // The `branches` key's ABSENCE is the schema marker the decoder keys
        // on, and `activeLeafID` / `branchChoices` ride along with it: a
        // conversation that never branched must not emit any of the three.
        // (Rows still carry `parentID` — an additive key old builds ignore.)
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-branching-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("conversations.json")

        let viewModel = ChatViewModel(conversationStoreURL: file)
        // A real send, not a dev seed: ``persistActive`` is what writes the
        // file, and it runs on the send path.
        viewModel.send("question", alias: "test-model")
        viewModel.stopAndPersist()
        ConversationStore.flush()

        let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: file))
        let stored = try #require((raw as? [[String: Any]])?.first)
        #expect(stored["branches"] == nil)
        #expect(stored["activeLeafID"] == nil)
        #expect(stored["branchChoices"] == nil)
    }

    @Test("Deleting every alternative returns the file to the linear shape intact")
    func drainedBranchesRoundTripSafely() throws {
        // The migration marker is the `branches` key's presence, so the
        // dangerous transition is branching and then deleting every
        // alternative: the re-encoded file has no marker again, and the next
        // decode must NOT re-chain it into something else. It survives
        // because the remaining path still carries its parent links, which
        // the legacy repair (keyed on "NO row has a parent") leaves alone.
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-branching-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("conversations.json")

        let viewModel = ChatViewModel(conversationStoreURL: file)
        let user = ChatMessage(role: .user, content: "question")
        let answer = ChatMessage(role: .assistant, content: "first answer")
        viewModel.devSeedMessages([user, answer])
        viewModel.regenerateLast(alias: "test-model")
        viewModel.stopAndPersist()
        #expect(viewModel.deleteMessage(id: answer.id))
        ConversationStore.flush()

        let restored = try #require(ConversationStore.load(from: file).first)
        #expect(!restored.hasBranches)
        #expect(restored.activePath.count == 2)
        #expect(restored.activePath.first?.content == "question")
    }

    @Test("Editing the opening prompt, then deleting the old branch, reloads intact")
    func editedRootThenDrainedBranchesRoundTrip() throws {
        // The nastiest shape for the old shape-inferred migration: editing
        // the opening prompt makes a SECOND parentless root. Once the stale
        // branch is deleted the file has no `branches` key again — reload
        // must keep the surviving root's chain, not splice anything.
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-branching-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("conversations.json")

        let viewModel = ChatViewModel(conversationStoreURL: file)
        let user = ChatMessage(role: .user, content: "original question")
        let answer = ChatMessage(role: .assistant, content: "original answer")
        viewModel.devSeedMessages([user, answer])
        #expect(
            viewModel.editUserMessage(
                id: user.id, newContent: "revised question", alias: "test-model"
            )
        )
        viewModel.stopAndPersist()
        #expect(viewModel.deleteMessage(id: user.id))
        ConversationStore.flush()

        let restored = try #require(ConversationStore.load(from: file).first)
        #expect(!restored.hasBranches)
        #expect(restored.activePath.first?.content == "revised question")
        #expect(!restored.allMessages.contains { $0.id == user.id })
        #expect(!restored.allMessages.contains { $0.id == answer.id })
    }

    @Test("Deleting the only answer keeps its prompt")
    func deletingOnlyAnswerKeepsPrompt() {
        // The subtree stops at the selected turn: a childless prompt is still
        // something the user said, and the seed for a fresh regeneration.
        let (viewModel, user, answer) = seededModel()
        #expect(viewModel.deleteMessage(id: answer.id))
        #expect(viewModel.messages.map(\.id) == [user.id])
        #expect(viewModel.deletionImpact(of: user.id) == 1)
    }

    @Test("A snapshot built mid-stream never overwrites the finished answer")
    func midStreamSnapshotDoesNotClobberFinalContent() {
        // The branch-metadata snapshot is rebuilt only on SHAPE changes, so
        // one built while an answer streams caches that row's early value —
        // and the stream finishing is a content write that leaves the shape
        // version untouched. Switching branches afterwards must re-read the
        // live rows: adopting the stale snapshot would silently roll the
        // finished answer back to its mid-stream prefix and persist that.
        let (viewModel, _, answer) = seededModel()
        viewModel.regenerateLast(alias: "test-model")
        // Force the snapshot to exist while the replacement is streaming —
        // exactly what rendering the transcript does.
        _ = viewModel.branchPosition(of: viewModel.messages[1].id)
        viewModel.stopAndPersist()
        let finished = viewModel.messages[1]
        #expect(finished.status != .streaming)

        // Away to the original and back: both hops re-adopt the tree.
        #expect(viewModel.selectBranch(at: 0, forSiblingOf: finished.id))
        #expect(viewModel.messages[1].id == answer.id)
        #expect(viewModel.selectBranch(at: 1, forSiblingOf: answer.id))
        #expect(viewModel.messages[1].id == finished.id)
        #expect(viewModel.messages[1].status == finished.status)
        #expect(viewModel.messages[1].content == finished.content)
    }

    @Test("Structural mutations are refused while a stream is in flight")
    func structuralMutationsRefusedWhileStreaming() {
        // The in-flight turn writes by INDEX into the visible path; every
        // structural entry point must refuse rather than move the rows under
        // it.
        let (viewModel, _, answer) = seededModel()
        viewModel.regenerateLast(alias: "test-model")
        defer { viewModel.stopAndPersist() }

        #expect(viewModel.isStreaming)
        let replacement = viewModel.messages[1]
        #expect(!viewModel.deleteMessage(id: answer.id))
        #expect(!viewModel.stepBranch(from: replacement.id, by: -1))
        #expect(!viewModel.selectBranch(at: 0, forSiblingOf: replacement.id))
        #expect(viewModel.messages.count == 2)
    }

    // MARK: - Remembered position within a branch

    /// Two answers to one prompt, where the FIRST was continued for several
    /// more turns. Returning to it must land where the user left, not at its
    /// deepest tip.
    private func branchedModelWithContinuation() -> (
        model: ChatViewModel, original: ChatMessage, middle: ChatMessage, tip: ChatMessage
    ) {
        let viewModel = ChatViewModel(persistsConversations: false)
        let user = ChatMessage(role: .user, content: "question")
        let original = ChatMessage(role: .assistant, content: "first answer")
        let middleUser = ChatMessage(role: .user, content: "follow-up")
        let tip = ChatMessage(role: .assistant, content: "deep answer")
        viewModel.devSeedMessages([user, original, middleUser, tip])
        // Branch at the prompt: a second answer alongside `original`.
        viewModel.retryAssistantMessage(id: original.id, alias: "test-model")
        viewModel.stopAndPersist()
        return (viewModel, original, middleUser, tip)
    }

    @Test("Returning to a branch lands where the user left it")
    func returningToABranchRestoresPosition() {
        let (viewModel, original, middle, tip) = branchedModelWithContinuation()

        // Go back to the original answer — it resolves to its own tip…
        #expect(viewModel.stepBranch(from: viewModel.messages.last!.id, by: -1))
        #expect(viewModel.messages.last?.id == tip.id)

        // …then step UP inside that branch, so the user's position is no
        // longer the tip.
        #expect(viewModel.deleteMessage(id: tip.id))
        #expect(viewModel.messages.last?.id == middle.id)

        // Leave for the sibling and come back. Without a remembered choice
        // this would resolve to whatever is newest rather than the branch the
        // user was actually reading.
        #expect(viewModel.stepBranch(from: original.id, by: 1))
        #expect(viewModel.stepBranch(from: viewModel.messages.last!.id, by: -1))
        #expect(viewModel.messages.contains { $0.id == original.id })
    }

    @Test("Remembered positions survive a save/load round trip")
    func branchChoicesArePersisted() throws {
        // Regression cover for a real trap: a hand-written `init(from:)` with
        // a SYNTHESISED encoder writes `[UUID: UUID]` as a flat array while
        // the decoder asks for an object, so the map would look correct all
        // session and silently reset on relaunch.
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-branch-memory-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("conversations.json")

        let viewModel = ChatViewModel(conversationStoreURL: file)
        let user = ChatMessage(role: .user, content: "question")
        let answer = ChatMessage(role: .assistant, content: "first answer")
        viewModel.devSeedMessages([user, answer])
        viewModel.regenerateLast(alias: "test-model")
        viewModel.stopAndPersist()
        // Step back to the original, which records a choice at the prompt.
        #expect(viewModel.stepBranch(from: viewModel.messages.last!.id, by: -1))
        ConversationStore.flush()

        let conversation = try #require(ConversationStore.load(from: file).first)
        #expect(!conversation.branchChoices.isEmpty)
        #expect(conversation.branchChoices[user.id] == answer.id)
        // And the stored map actually steers the derived path.
        #expect(conversation.activePath.last?.id == answer.id)
    }

    @Test("A remembered choice pointing at a deleted branch is ignored")
    func staleChoicesDegradeGracefully() {
        // Stale entries are left in the map rather than pruned everywhere, so
        // the read path has to tolerate them: a hint that no longer names a
        // child of the fork must fall back, not strand the walk.
        let user = ChatMessage(role: .user, content: "q", createdAt: Date(timeIntervalSince1970: 0))
        let answer = ChatMessage(
            role: .assistant, content: "a",
            parentID: user.id, createdAt: Date(timeIntervalSince1970: 1)
        )
        let leaf = MessageTree.deepestLeaf(
            from: user.id,
            in: [user, answer],
            preferring: [user.id: UUID()]
        )
        #expect(leaf == answer.id)
    }

    // MARK: - Review regressions

    @Test("A regenerated tool-assisted answer is still switchable")
    func toolTurnsExposeTheSwitcher() {
        // A tool round-trip writes FOUR rows for one logical answer:
        // user -> assistant(tool_calls) -> tool -> assistant(final).
        // The branch forks at the dispatch row, but the row the user reads is
        // the final answer — so asking for the final answer's own siblings
        // found none and the switcher disappeared exactly when a tool was
        // used, stranding the user on the new answer with no way back.
        let viewModel = ChatViewModel(persistsConversations: false)
        let user = ChatMessage(role: .user, content: "weather?")
        let dispatch = ChatMessage(
            role: .assistant, content: "",
            toolCalls: [ToolCall(id: "c1", name: "weather", arguments: "{}")]
        )
        let toolRow = ChatMessage(role: .tool, content: "{\"t\":20}", toolCallID: "c1")
        let final = ChatMessage(role: .assistant, content: "It is 20 degrees.")
        viewModel.devSeedMessages([user, dispatch, toolRow, final])

        viewModel.regenerateLast(alias: "test-model")
        viewModel.stopAndPersist()

        let position = try? #require(viewModel.branchPosition(of: final.id))
        #expect(position?.count == 2)
        // …and stepping from that row actually moves, rather than reporting a
        // position the arrows cannot act on.
        #expect(viewModel.stepBranch(from: viewModel.messages.last!.id, by: -1))
        #expect(viewModel.messages.contains { $0.id == final.id })
    }

    @Test("A parent cycle does not hang the branch switcher", .timeLimit(.minutes(1)))
    func branchPositionSurvivesACycle() {
        // ``MessageTree/activePath`` TRUNCATES a corrupt parent chain instead
        // of rejecting it, so a looping tree from a hand-edited
        // conversations.json still reaches the screen — and every rendered row
        // then asks for its branch position. The anchor walk has to carry the
        // same cycle guard or that query spins on the main actor and the
        // window wedges.
        let viewModel = ChatViewModel(persistsConversations: false)
        var a = ChatMessage(role: .assistant, content: "a")
        var b = ChatMessage(role: .assistant, content: "b")
        a.parentID = b.id
        b.parentID = a.id
        // A non-nil parent anywhere skips the legacy re-chaining, so the
        // cycle survives seeding intact.
        viewModel.devSeedMessages([a, b])

        // Terminating at all is the assertion; the value only has to be sane.
        let position = viewModel.branchPosition(of: a.id)
        #expect(position == nil || position?.count ?? 0 > 1)
        #expect(viewModel.siblings(of: a.id).count <= 2)
    }

    @Test("Editing the opening prompt survives a reload as a real branch")
    func multipleRootsAreNotMistakenForLegacy() throws {
        // Editing the FIRST prompt legitimately produces two parentless roots,
        // which is shape-identical to a pre-branching linear array. Inferring
        // the migration from that shape re-chained the two prompts into one
        // thread: the branch entry vanished and the next request would have
        // sent both prompts as consecutive turns of one conversation.
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-multiroot-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("conversations.json")

        let viewModel = ChatViewModel(conversationStoreURL: file)
        let first = ChatMessage(role: .user, content: "prompt1")
        viewModel.devSeedMessages([first, ChatMessage(role: .assistant, content: "answer1")])
        _ = viewModel.editUserMessage(id: first.id, newContent: "prompt2", alias: "test-model")
        viewModel.stopAndPersist()
        ConversationStore.flush()

        let conversation = try #require(ConversationStore.load(from: file).first)
        #expect(conversation.hasBranches)
        // The two prompts are alternatives, never consecutive turns.
        let path = conversation.activePath.map(\.content)
        #expect(path.contains("prompt2"))
        #expect(!path.contains("prompt1"))
    }

    @Test("An old build reads the visible path, not a flattened tree")
    func downgradeSeesALinearTranscript() throws {
        // `messages` is the field every shipped build already reads. Writing
        // the whole node bag there would hand a downgraded build a pile of
        // alternatives with no parent links, which it would present as one
        // linear conversation and then re-save in that scrambled order.
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-downgrade-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("conversations.json")

        let viewModel = ChatViewModel(conversationStoreURL: file)
        viewModel.devSeedMessages([
            ChatMessage(role: .user, content: "q"),
            ChatMessage(role: .assistant, content: "answer-A"),
        ])
        viewModel.regenerateLast(alias: "test-model")
        viewModel.stopAndPersist()
        ConversationStore.flush()

        let raw = try #require(
            try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [[String: Any]]
        )
        let legacyView = try #require(raw.first?["messages"] as? [[String: Any]])
        // Exactly the active path: prompt + the current answer.
        #expect(legacyView.count == 2)
        #expect(legacyView.compactMap { $0["content"] as? String }.first == "q")
        // The replaced answer is parked under a key an old build ignores.
        #expect((raw.first?["branches"] as? [[String: Any]])?.count == 1)
    }
}
