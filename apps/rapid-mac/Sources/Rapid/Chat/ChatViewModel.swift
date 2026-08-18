import AppKit
import Foundation
import Observation

/// Per-window controller that ties ``SessionStore`` to a
/// ``ChatStreamClient``. The view owns a ``ChatViewModel`` and the model
/// owns the active streaming ``Task`` — that makes "stop" a single
/// ``task.cancel()`` instead of state-machine bookkeeping.
///
/// v0.3: the chat loop now drives the tool round-trip. When the model
/// returns ``finish_reason: "tool_calls"``, we run the referenced tools
/// (in parallel via ``TaskGroup``), append the ``role: "tool"`` results
/// to the transcript, open a fresh assistant placeholder, and start
/// another stream. The loop terminates on any non-tool finish reason
/// or after a tools-disabled synthesis round when ``maxToolExecutions`` is hit.
@MainActor
@Observable
final class ChatViewModel {
    /// The ACTIVE conversation's live message buffer. All streaming /
    /// append / update goes through here; ``persistActive()`` snapshots it
    /// into ``conversations`` + disk at each turn boundary (M3 history).
    private(set) var messages: [ChatMessage] = []

    /// Saved conversations, newest-updated first — the sidebar "Older"
    /// list. Loaded from disk on init; upserted whenever the active
    /// conversation gains a user turn.
    private(set) var conversations: [ChatConversation] = []

    /// User-created folders for filing conversations. Loaded from their own
    /// `folders.json` on init; see ``ConversationFolderStore`` for why the
    /// list is not stored alongside the transcripts.
    private(set) var folders: [ChatFolder] = []

    /// Identity of the conversation ``messages`` currently holds. A fresh
    /// UUID on launch (opens to an empty "Ask anything"); ``persistActive``
    /// upserts under this id once the user sends.
    private(set) var activeConversationID = UUID()

    /// User-authored instructions for the open conversation. They are kept
    /// outside the visible transcript and merged into the wire-only system row.
    private(set) var conversationInstructions: String = ""

    /// Bumped on every conversation switch (new / select / delete). Each
    /// send captures the epoch; any streaming write or completion whose
    /// captured epoch no longer matches is discarded — so a stream that
    /// outlives a switch can't bleed tokens into, or persist over, the
    /// conversation the user moved to (codex BLOCKING).
    private var conversationEpoch = 0

    /// ``var`` not ``let`` because ``ChatStreamClient`` is a struct
    /// and ``send()`` re-targets ``client.baseURL`` to track
    /// ``ServerManager.activePort`` (v0.5.6 port-fallback) before each
    /// request. A class wrapper would have been an alternative but
    /// would force every test that constructs a model to either
    /// inject a real URLSession or thread a mock through both layers.
    private var client: ChatStreamClient

    /// Tool runner. An ``EmptyToolRegistry`` short-circuits the tool-call
    /// request shape (we send no ``tools:`` field and the server emits plain
    /// content), which is what unit tests and the no-tools build get.
    let tools: any ToolRegistry

    /// Hard cap on tool executions within a single user turn. Three leaves
    /// room for the useful search → open page → refine pattern without making
    /// a user wait through a long local-model loop. After the budget is spent
    /// we give the model one tools-disabled round to synthesize what it has.
    private let maxToolExecutions: Int = 3

    nonisolated private static let toolBudgetSynthesisPreamble = """
    The tool-use budget for this turn is exhausted. Do not request or describe any more tool calls. Answer the user's question now using the evidence already present in the conversation. If that evidence is insufficient, say what remains uncertain.
    """

    /// Per-tool on/off flags, persisted in ``UserDefaults`` under keys of the
    /// form ``rapid.tools.enabled.<name>``. Reads fall through to ``true``
    /// (every tool is enabled by default) so a fresh install picks up tools we
    /// ship without the user opting in. ``enabledDefinitions`` filters disabled
    /// tools out before they reach the model, and the loop also gates dispatch
    /// so a model that invents a disabled name gets a clean refusal instead.
    private(set) var disabledTools: Set<String>

    /// ``UserDefaults`` suite the tool flags live in. Injectable so tests can
    /// use a fresh in-memory suite per case.
    private let toolDefaults: UserDefaults

    private static func toolEnabledKey(_ name: String) -> String {
        "rapid.tools.enabled.\(name)"
    }

    /// Set while a stream is in flight. UI reads this to show the stop
    /// button instead of send.
    /// The assistant message currently streaming, as (id, text).
    ///
    /// A projection rather than a stored property: it exists so exactly one
    /// view can observe the growing string, instead of every row observing
    /// the whole `messages` array. Nil once nothing is streaming.
    var streamingBody: StreamingBody? {
        guard isStreaming,
              let m = messages.last(where: { $0.role == .assistant && $0.status == .streaming })
        else { return nil }
        return StreamingBody(id: m.id, text: m.content)
    }

    struct StreamingBody: Equatable {
        let id: UUID
        let text: String
    }

    private(set) var isStreaming: Bool = false {
        didSet {
            // A turn just ended (stream finished, failed, or was stopped) —
            // snapshot the final exchange into history + disk. Covers every
            // completion path without hooking each one.
            if oldValue && !isStreaming { persistActive() }
        }
    }
    /// Most recent transport / parse error. Cleared on the next ``send``.
    /// Shown as a banner above the compose bar so the user knows what
    /// went wrong without scraping the log tail.
    private(set) var lastError: String?
    /// Structured counterpart to ``lastError`` used by the shared failure
    /// view to choose a recovery action without parsing display copy.
    private(set) var lastFailureKind: FailureDiagnosis.Kind?
    /// Which model ``lastError`` is about.
    ///
    /// Without this the readiness surface had to guess, and it guessed
    /// "whatever is selected right now" — so after `kimi-k2.6` failed to
    /// start, choosing `bonsai-1.7b-2bit` re-labelled Kimi's error as
    /// Bonsai's. A failure belongs to the model that actually failed;
    /// recording the alias is what lets a later surface decide whether
    /// the failure is still the user's current problem.
    ///
    /// Cleared in lockstep with ``lastError``. Note this scopes only the
    /// BANNER — the failed turn stays in the transcript either way.
    private(set) var lastFailureAlias: String?

    private var inflight: Task<Void, Never>?

    /// Tests that exercise turn replay can disable disk I/O so seeded
    /// transcripts never read or overwrite the user's conversation history.
    /// Production keeps the default enabled.
    private let persistsConversations: Bool
    private let conversationStoreURL: URL?

    /// v0.4.14: user-mutable sampling knobs. Optional in the init
    /// signature so existing tests don't have to spin one up — they
    /// fall back to the v0.4.12 hard-coded defaults via
    /// ``ChatStreamClient.Request``'s own default parameters.
    /// Production code (``RapidApp.init``) always passes a real
    /// ``SamplingConfig`` reading from ``UserDefaults``.
    let sampling: SamplingConfig?

    /// Global custom instructions shared with Settings.
    let customInstructions: CustomInstructionsConfig

    /// v0.5.1: live handle on the embedded server so the chat loop can
    /// resolve `request.model` against the alias currently being served
    /// instead of trusting the picker bar's state. Mirrors the global
    /// "loaded model" model used by Ollama and LM Studio — outgoing
    /// requests follow whatever the backend has resident. Optional so
    /// unit tests can construct a viewmodel without spinning up a real
    /// process; production wires this from ``RapidApp.init``.
    private weak var server: ServerManager?

    init(
        client: ChatStreamClient = ChatStreamClient(),
        tools: any ToolRegistry = EmptyToolRegistry(),
        toolDefaults: UserDefaults = .standard,
        sampling: SamplingConfig? = nil,
        customInstructions: CustomInstructionsConfig? = nil,
        server: ServerManager? = nil,
        persistsConversations: Bool = true,
        conversationStoreURL: URL? = nil
    ) {
        self.client = client
        self.tools = tools
        self.toolDefaults = toolDefaults
        self.sampling = sampling
        self.customInstructions = customInstructions ?? CustomInstructionsConfig()
        self.server = server
        self.persistsConversations = persistsConversations
        self.conversationStoreURL = conversationStoreURL
        // Seed disabledTools from the persistent store. Anything explicitly set
        // to ``false`` in UserDefaults goes in; unknown keys default to enabled.
        var disabled = Set<String>()
        for def in tools.definitions {
            // ``object(forKey:)`` so we can distinguish "absent" (default to
            // enabled) from "explicitly false".
            if let raw = toolDefaults.object(forKey: Self.toolEnabledKey(def.function.name)) as? Bool,
               raw == false {
                disabled.insert(def.function.name)
            }
        }
        self.disabledTools = disabled
        self.conversations = persistsConversations
            ? ConversationStore.load(from: conversationStoreURL)
            : []
        self.folders = persistsConversations
            ? ConversationFolderStore.load(from: Self.folderStoreURL(for: conversationStoreURL))
            : []
    }

    /// The folder file that sits beside whichever conversation store is in
    /// use, so an injected test store keeps its folders in the same temp
    /// directory instead of touching the real one.
    private static func folderStoreURL(for conversationStore: URL?) -> URL? {
        ConversationFolderStore.companionURL(forConversationStore: conversationStore)
    }

    /// Toggle a tool from the UI. Persists to ``UserDefaults`` so the choice
    /// survives an app restart.
    func setToolEnabled(_ name: String, _ enabled: Bool) {
        toolDefaults.set(enabled, forKey: Self.toolEnabledKey(name))
        if enabled {
            disabledTools.remove(name)
        } else {
            disabledTools.insert(name)
        }
    }

    /// Active tool definitions — the registry minus anything the user has
    /// toggled off. Computed every send so a mid-session toggle takes effect on
    /// the next turn without re-initialising the chat loop.
    var enabledDefinitions: [ToolDefinition] {
        tools.definitions.filter { !disabledTools.contains($0.function.name) }
    }

    /// Just the built-in tools, for Settings → Tools.
    ///
    /// Issue #1716: since the registry became a ``CompositeToolRegistry``,
    /// ``tools/definitions`` also carries connector tools. Those get their own
    /// switch in Settings → Connectors, backed by a different defaults key
    /// (``MCPToolRegistry``). Listing them in both panels would give one tool
    /// two independent off switches — flip the wrong one and the tool stays
    /// live with no indication why. Each surface owns exactly its own set.
    ///
    /// Falls back to the whole registry when it isn't a composite, which is
    /// what unit tests and the dev-snapshot harness construct.
    var builtinDefinitions: [ToolDefinition] {
        (tools as? CompositeToolRegistry)?.builtin.definitions ?? tools.definitions
    }

    // MARK: - Conversation history (M3)

    /// Snapshot the active conversation into ``conversations`` + disk. A
    /// no-op until the user has actually sent something (no empty rows in
    /// the sidebar). Title is derived from the first user message.
    ///
    /// - Parameter touching: whether this counts as *activity*. Only a real
    ///   change to the transcript should refresh ``updatedAt`` and bubble the
    ///   row to the top. Merely opening a conversation to read it, or leaving
    ///   it to start a new one, archives the buffer without claiming the user
    ///   did anything to it — otherwise clicking through history silently
    ///   reshuffles the sidebar, and the list stops reflecting when each
    ///   conversation was last *worked on*.
    private func persistActive(touching: Bool = true) {
        guard messages.contains(where: { $0.role == .user }) else { return }
        let now = Date()
        let title = ConversationStore.title(from: messages)
        if conversations.contains(where: { $0.id == activeConversationID }) {
            conversations = ConversationOrdering.updating(
                conversations,
                id: activeConversationID,
                touching: touching,
                at: now
            ) { conversation in
                conversation.messages = messages
                // A renamed row keeps its name. Re-deriving unconditionally
                // meant the next streamed token silently reverted the user's
                // rename back to the first prompt's opening words.
                if !conversation.hasCustomTitle {
                    conversation.title = title
                }
                conversation.customInstructions = Self.normalizedInstruction(
                    conversationInstructions
                )
            }
        } else {
            conversations.insert(
                ChatConversation(
                    id: activeConversationID,
                    title: title,
                    messages: messages,
                    createdAt: now,
                    updatedAt: now,
                    customInstructions: Self.normalizedInstruction(conversationInstructions)
                ),
                at: 0
            )
        }
        saveConversations()
    }

    // MARK: - Conversation row actions (rename / pin / archive)

    /// Rename a saved conversation.
    ///
    /// Trimmed, and blank input is rejected rather than accepted as an empty
    /// row label — an unnamed row already has a sensible derived title, so
    /// "clear the name" is better served by the caller not committing.
    /// Setting ``hasCustomTitle`` is what stops ``persistActive`` re-deriving
    /// the title from the transcript on the next save.
    ///
    /// A rename is not conversation *activity*, so ``updatedAt`` and the row's
    /// position are left alone — the same reasoning ``ConversationOrdering``
    /// applies to merely opening a conversation.
    @discardableResult
    func renameConversation(_ id: UUID, to newTitle: String) -> Bool {
        let trimmed = newTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        guard let index = conversations.firstIndex(where: { $0.id == id }) else { return false }
        conversations[index].title = trimmed
        conversations[index].hasCustomTitle = true
        saveConversations()
        return true
    }

    /// Pin / unpin a conversation. Pinned rows get their own sidebar section
    /// above the date buckets.
    func setConversationPinned(_ id: UUID, _ pinned: Bool) {
        guard let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        guard conversations[index].isPinned != pinned else { return }
        conversations[index].isPinned = pinned
        // Pinning something archived is contradictory — the row would be
        // pinned to a list it isn't in. Surfacing it is the intent.
        if pinned { conversations[index].isArchived = false }
        saveConversations()
    }

    /// Archive / unarchive a conversation. Archiving is deliberately NOT a
    /// delete: the transcript stays on disk and the row remains reachable
    /// from the sidebar's Archived section, which is why — unlike Delete —
    /// it needs no confirmation and is one click to undo.
    ///
    /// Archiving the OPEN conversation leaves it open. Closing the transcript
    /// out from under the user would turn a filing action into an unexpected
    /// navigation, and re-reading what you just archived is normal.
    func setConversationArchived(_ id: UUID, _ archived: Bool) {
        guard let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        guard conversations[index].isArchived != archived else { return }
        conversations[index].isArchived = archived
        if archived { conversations[index].isPinned = false }
        saveConversations()
    }

    /// Update the instruction layer for the open conversation. Existing saved
    /// chats are written immediately; a brand-new empty chat is persisted with
    /// its first user turn, avoiding an empty row in the sidebar.
    func setConversationInstructions(_ value: String) {
        guard value != conversationInstructions else { return }
        conversationInstructions = value
        guard let index = conversations.firstIndex(where: { $0.id == activeConversationID }) else {
            return
        }
        conversations[index].customInstructions = Self.normalizedInstruction(value)
        saveConversations()
    }

    private func saveConversations() {
        guard persistsConversations else { return }
        ConversationStore.save(conversations, to: conversationStoreURL)
    }

    // MARK: - Folders

    /// Create a folder. Returns nil for a blank name rather than making an
    /// unnamed row the user then can't tell apart from any other.
    ///
    @discardableResult
    func createFolder(named rawName: String) -> ChatFolder? {
        guard let name = ChatFolder.normalizedName(rawName) else { return nil }
        guard !folderNameExists(name) else { return nil }
        let folder = ChatFolder(name: name)
        folders.append(folder)
        saveFolders()
        return folder
    }

    @discardableResult
    func renameFolder(_ id: UUID, to rawName: String) -> Bool {
        guard let name = ChatFolder.normalizedName(rawName) else { return false }
        guard let index = folders.firstIndex(where: { $0.id == id }) else { return false }
        guard folders[index].name != name else { return true }
        guard !folderNameExists(name, excluding: id) else { return false }
        folders[index].name = name
        saveFolders()
        return true
    }

    /// Delete a folder WITHOUT deleting the conversations filed in it.
    ///
    /// The transcripts are the valuable thing; the folder is just where the
    /// user put them. Unfiling them returns the rows to the date buckets,
    /// which is recoverable — deleting them would not be. Same restraint
    /// ``setConversationArchived`` shows for the same reason.
    ///
    /// Clearing ``folderID`` eagerly (rather than leaning on the render-time
    /// orphan fallback) is what keeps a later folder created with a recycled
    /// id from silently adopting rows that were never filed into it.
    func deleteFolder(_ id: UUID) {
        guard folders.contains(where: { $0.id == id }) else { return }
        folders.removeAll { $0.id == id }
        var unfiled = false
        for index in conversations.indices where conversations[index].folderID == id {
            conversations[index].folderID = nil
            unfiled = true
        }
        saveFolders()
        if unfiled { saveConversations() }
    }

    /// File a conversation into a folder, or pass nil to unfile it.
    ///
    /// Filing is not conversation *activity*: ``updatedAt`` and the row's
    /// position are left alone, matching rename / pin / archive and the
    /// contract ``ConversationOrdering`` states.
    ///
    /// **Filing un-archives.** The sidebar shows archived rows only in the
    /// Archived disclosure, ahead of any folder, so filing one without this
    /// would record the folder and change nothing you can see — the row stays
    /// where it was and the action reads as broken. Putting something in a
    /// folder means wanting it in that folder; surfacing it is what makes the
    /// gesture honest. Unfiling (`nil`) deliberately does NOT re-archive:
    /// there is no earlier state to restore, and silently archiving a row the
    /// user only wanted out of a folder would hide it entirely.
    func moveConversation(_ id: UUID, toFolder folderID: UUID?) {
        guard let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        // An id for a folder that no longer exists would file the row into a
        // section that never renders — i.e. it would look deleted.
        if let folderID, !folders.contains(where: { $0.id == folderID }) { return }
        let needsSurfacing = folderID != nil && conversations[index].isArchived
        guard conversations[index].folderID != folderID || needsSurfacing else { return }
        conversations[index].folderID = folderID
        if needsSurfacing { conversations[index].isArchived = false }
        saveConversations()
    }

    private func saveFolders() {
        guard persistsConversations else { return }
        ConversationFolderStore.save(
            folders,
            to: Self.folderStoreURL(for: conversationStoreURL)
        )
    }

    func folderNameExists(_ name: String, excluding excludedID: UUID? = nil) -> Bool {
        folders.contains {
            $0.id != excludedID
                && $0.name.compare(name, options: [.caseInsensitive, .diacriticInsensitive])
                    == .orderedSame
        }
    }

    /// Load a saved conversation into the transcript, archiving whatever is
    /// currently open first. Cancels any in-flight stream.
    func selectConversation(_ id: UUID) {
        guard id != activeConversationID else { return }
        inflight?.cancel()
        inflight = nil
        conversationEpoch &+= 1
        // Archive + unstick BEFORE swapping buffers, so the old transcript
        // is what gets persisted and a mid-stream switch doesn't leave the
        // incoming conversation showing Stop.
        isStreaming = false
        persistActive(touching: false)
        guard let conv = conversations.first(where: { $0.id == id }) else { return }
        messages = conv.messages
        conversationInstructions = conv.customInstructions ?? ""
        activeConversationID = id
        lastError = nil
        lastFailureKind = nil
        lastFailureAlias = nil
    }

    /// Delete a saved conversation. If it was the open one, drop to a fresh
    /// empty transcript.
    func deleteConversation(_ id: UUID) {
        // If deleting the OPEN conversation, tear down the live transcript
        // FIRST — otherwise the `isStreaming = false` below fires
        // persistActive() via didSet while the deleted messages + id are
        // still active, re-inserting the conversation we just removed.
        if id == activeConversationID {
            inflight?.cancel()
            inflight = nil
            conversationEpoch &+= 1
            messages.removeAll()
            conversationInstructions = ""
            activeConversationID = UUID()
            isStreaming = false          // messages now empty → persistActive no-ops
            lastError = nil
            lastFailureKind = nil
            lastFailureAlias = nil
        }
        conversations.removeAll { $0.id == id }
        saveConversations()
    }

    // MARK: - In-memory message storage

    /// Append a message and return its index in ``messages``.
    @discardableResult
    private func appendMessage(_ message: ChatMessage) -> Int {
        messages.append(message)
        return messages.count - 1
    }

    /// Overwrite the message at ``index`` when it is still in range.
    private func updateMessage(at index: Int, with message: ChatMessage) {
        guard messages.indices.contains(index) else { return }
        messages[index] = message
    }

    /// Streaming write guarded on the sending conversation's epoch: a
    /// no-op once the user has switched conversations under the stream, so
    /// a stale/cancelled stream can't overwrite a message in — or, via the
    /// completion path, persist — the conversation now on screen.
    private func writeStreamMessage(at index: Int, epoch: Int, _ message: ChatMessage) {
        guard epoch == conversationEpoch else { return }
        updateMessage(at: index, with: message)
    }

    /// Snapshot of the message at ``index``, or ``nil`` when out of range.
    private func currentMessage(index: Int) -> ChatMessage? {
        guard messages.indices.contains(index) else { return nil }
        return messages[index]
    }

    /// Start a fresh conversation — drops the transcript and any stale
    /// error banner. The in-flight stream (if any) is cancelled first.
    func newConversation() {
        inflight?.cancel()
        inflight = nil
        conversationEpoch &+= 1
        // Fix: without this a New Chat during a stream leaves the empty
        // chat stuck showing Stop (isStreaming never reset). Setting it
        // false here also archives the just-closed conversation via didSet.
        isStreaming = false
        persistActive(touching: false)
        messages.removeAll()
        conversationInstructions = ""
        activeConversationID = UUID()
        lastError = nil
        lastFailureKind = nil
        lastFailureAlias = nil
    }

    /// DEV-ONLY: replace the transcript with a fixed set of messages so
    /// the `DevSnapshot` harness can render a populated chat. Never called
    /// in normal use (only from the env-gated snapshot path).
    func devSeedMessages(_ seeded: [ChatMessage]) {
        messages = seeded
    }

    /// Append a locally authored assistant message to the open conversation.
    ///
    /// The one caller is the onboarding completion transaction, which lands
    /// its welcome message in the chat the user is about to be dropped into.
    /// There is no network round trip and no stream: the text is written
    /// straight into the transcript as a finished assistant turn, so it can
    /// never wedge the typing indicator or the streaming gate.
    ///
    /// A transcript that already holds messages is left alone. Onboarding
    /// completion is a one-shot event, but the app can reach it with a live
    /// conversation on screen (a user who skipped setup, chatted, then came
    /// back to it), and injecting a stray intro into the middle of somebody's
    /// conversation is worse than skipping the pleasantry.
    ///
    /// - Returns: ``true`` when the message landed in the transcript.
    @discardableResult
    func seedAssistantWelcome(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        guard messages.isEmpty else { return false }
        appendMessage(ChatMessage(role: .assistant, content: trimmed))
        return true
    }

    /// Append the user message, open a placeholder assistant row, and
    /// kick off the streaming task. The text field clears immediately on
    /// the caller's side.
    func send(
        _ text: String,
        alias: String,
        imageAttachments: [ChatImageAttachment] = [],
        fileAttachments: [ChatFileAttachment] = []
    ) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty || !imageAttachments.isEmpty || !fileAttachments.isEmpty else { return }
        guard !isStreaming else { return }

        // Small local models are unreliable at the first step of tool use:
        // deciding that an explicitly live/dated question needs the web. Do
        // that narrow piece of routing in the app, while leaving query wording
        // and answer synthesis to the model. Follow-ups inherit the intent of
        // the preceding user turn ("What about technology?") so restoring a
        // conversation cannot silently turn search back into plain chat.
        let forcedTool = Self.forcedToolForUserTurn(
            trimmed,
            priorMessages: messages,
            enabledToolNames: Set(enabledDefinitions.map { $0.function.name })
        )
        let forcedWebSearchQuery = forcedTool == "web_search"
            ? Self.webSearchQuery(for: trimmed, priorMessages: messages)
            : nil

        let user = ChatMessage(
            role: .user,
            content: trimmed,
            imageAttachments: imageAttachments,
            fileAttachments: ChatFileAttachment.fittedForMessage(fileAttachments),
            status: .complete
        )
        _ = appendMessage(user)
        // Surface the conversation in the sidebar immediately — before the
        // (possibly cold, ~minute) model load — so "Older" reflects the
        // send the instant it happens, not only once the reply lands.
        persistActive()

        let placeholder = ChatMessage(role: .assistant, status: .streaming)
        let placeholderIndex = appendMessage(placeholder)

        lastError = nil
        lastFailureKind = nil
        lastFailureAlias = nil
        isStreaming = true

        let epoch = conversationEpoch
        // Freeze both user-authored layers for this turn. Changing Settings in
        // another window takes effect on the next send, not halfway through a
        // multi-round tool exchange.
        let globalInstruction = customInstructions.global
        let chatInstruction = conversationInstructions
        inflight = Task { [weak self] in
            guard let self else { return }

            // Bring the model up if it isn't serving yet. The user's
            // turn is already in the transcript, so a load that takes a
            // minute — or fails — costs them nothing they typed.
            // `ensureServing` short-circuits when we are already serving
            // this alias, so the warm path pays only a state read.
            if let server {
                let ready = await server.ensureServing(
                    alias: alias,
                    hfPath: startupHFPath,
                    estimatedMemoryGB: nil,
                    replacementGroup: .assistant
                )
                // A user Stop during the (possibly cold, multi-second)
                // bring-up cancels THIS task. That is a deliberate
                // cancel, not a start failure — route it through the
                // cancel contract, never the "Couldn't start" failure
                // row + error banner. Checked BEFORE `ready`: a Stop
                // that lands the instant the model reaches ``.ready``
                // returns `ready == true`, yet must still not stream and
                // must still reset `isStreaming`.
                guard !Task.isCancelled else {
                    finishStartupCancellation(placeholderIndex: placeholderIndex, epoch: epoch)
                    return
                }
                guard ready else {
                    finishWithStartupFailure(
                        placeholderIndex: placeholderIndex,
                        alias: alias,
                        epoch: epoch
                    )
                    return
                }
            }
            guard !Task.isCancelled else {
                finishStartupCancellation(placeholderIndex: placeholderIndex, epoch: epoch)
                return
            }

            // v0.5.6: re-target the chat client at
            // ``ServerManager.activePort`` so a PortAllocator fallback
            // from the default port → +1 routes to the actual rapid-mlx
            // the child started. MUST run after the ensureServing await:
            // ``activePort`` is only assigned inside ``start``.
            if let server, server.activePort != 0 {
                client.baseURL = ChatStreamClient.loopbackURL(port: server.activePort)
            }

            await self.runToolLoop(
                alias: alias,
                initialPlaceholder: placeholderIndex,
                epoch: epoch,
                forcedWebSearchQuery: forcedWebSearchQuery,
                globalInstruction: globalInstruction,
                conversationInstruction: chatInstruction
            )
        }
    }

    /// Deterministic routing for prompts whose answer is explicitly time
    /// sensitive. This is intentionally narrower than a general semantic
    /// classifier: a false negative falls back to normal `tool_choice:auto`,
    /// while a false positive performs an unnecessary network search.
    nonisolated static func forcedToolForUserTurn(
        _ prompt: String,
        priorMessages: [ChatMessage],
        enabledToolNames: Set<String>
    ) -> String? {
        guard enabledToolNames.contains("web_search") else { return nil }
        if promptRequiresFreshWebEvidence(prompt) { return "web_search" }

        // A short follow-up often omits the live-time words carried by the
        // previous turn. Inherit across the immediately preceding user turn;
        // this also covers a restored thread where the first broad search ran
        // but the user now asks for a narrower, fresh query.
        guard let previous = priorMessages.last(where: { $0.role == .user }),
              promptLooksLikeFollowUp(prompt),
              promptRequiresFreshWebEvidence(previous.content)
        else { return nil }
        return "web_search"
    }

    nonisolated static func promptRequiresFreshWebEvidence(_ prompt: String) -> Bool {
        let value = prompt.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !value.isEmpty else { return false }
        let phrases = [
            "latest", "recent", "today", "yesterday", "this week", "last week",
            "this month", "this year", "right now", "breaking news",
            "news story", "news about", "world cup 2026", "2026 world cup",
            "current price", "current weather", "current version", "current president",
            "current status", "current score", "current exchange rate",
            "最新", "最近", "今天", "昨天", "本周", "上周", "这个月", "本月",
            "今年", "当前", "现在", "刚刚", "新闻", "今年世界杯"
        ]
        return phrases.contains(where: value.contains)
    }

    nonisolated static func promptLooksLikeFollowUp(_ prompt: String) -> Bool {
        let value = prompt.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !value.isEmpty, value.count <= 240 else { return false }
        let referentialPhrases = [
            "what about", "how about", "tell me more", "summarize it",
            "that story", "those results", "the same topic", "one concrete story",
            "那这个呢", "那件事", "这件事", "这些结果", "继续说", "总结一下这个"
        ]
        if referentialPhrases.contains(where: value.contains) { return true }
        // Bare elliptical replies are referential precisely because they have
        // no independent subject. Do not broaden this to arbitrary questions
        // containing "why" (e.g. "Why is the sky blue?").
        let bareFollowUps: Set<String> = [
            "why?", "why", "more", "and?", "还有呢？", "为什么？", "然后呢？"
        ]
        return bareFollowUps.contains(value)
    }

    nonisolated static func webSearchArguments(query: String) -> String {
        let data = try? JSONSerialization.data(
            withJSONObject: ["query": query],
            options: [.sortedKeys]
        )
        return data.flatMap { String(data: $0, encoding: .utf8) }
            ?? #"{"query":""}"#
    }

    /// Preserve the live-time scope when a follow-up is elliptical. Searching
    /// only "What about technology?" loses the preceding "last week" filter
    /// and lets stale or fictional high-ranking pages dominate the results.
    nonisolated static func webSearchQuery(
        for prompt: String,
        priorMessages: [ChatMessage]
    ) -> String {
        if promptRequiresFreshWebEvidence(prompt) { return prompt }
        if let previous = priorMessages.last(where: {
            $0.role == .user && promptRequiresFreshWebEvidence($0.content)
        }) {
            return "\(previous.content)\nFollow-up focus: \(prompt)"
        }
        return prompt
    }

    struct GroundingSource: Equatable, Sendable {
        let title: String
        let url: String
    }

    nonisolated static func groundingSources(from toolResult: String) -> [GroundingSource] {
        let lines = toolResult.split(separator: "\n", omittingEmptySubsequences: false)
        var sources: [GroundingSource] = []
        for index in lines.indices where sources.count < 3 {
            let titleLine = lines[index].trimmingCharacters(in: .whitespaces)
            guard titleLine.range(of: #"^\d+\.\s+"#, options: .regularExpression) != nil,
                  lines.indices.contains(index + 1)
            else { continue }
            let url = lines[index + 1].trimmingCharacters(in: .whitespaces)
            guard url.hasPrefix("https://") || url.hasPrefix("http://") else { continue }
            let title = titleLine.replacingOccurrences(
                of: #"^\d+\.\s+"#,
                with: "",
                options: .regularExpression
            )
            sources.append(GroundingSource(title: title, url: url))
        }
        return sources
    }

    private func appendGroundingSources(
        _ sources: [GroundingSource],
        to index: Int,
        epoch: Int
    ) {
        guard epoch == conversationEpoch,
              !sources.isEmpty,
              var message = currentMessage(index: index),
              message.status == .complete,
              !message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return }
        let missing = sources.filter { !message.content.contains($0.url) }
        guard !missing.isEmpty else { return }
        let rows = missing.map { source in
            let safeTitle = source.title
                .replacingOccurrences(of: "[", with: "\\[")
                .replacingOccurrences(of: "]", with: "\\]")
            return "- [\(safeTitle)](\(source.url))"
        }
        message.content += "\n\nSources:\n" + rows.joined(separator: "\n")
        updateMessage(at: index, with: message)
    }

    /// HF repo for the alias the next Send will start, supplied by the
    /// view from the catalog snapshot. Only used to install the
    /// bytes-on-disk progress monitor on a cold pull; a nil value just
    /// means a less informative progress strip, never a failure.
    var startupHFPath: String?

    /// Turn the streaming placeholder into a failure row when the
    /// model could not be brought up at all. Mirrors the shape
    /// ``runToolLoop`` uses for a transport failure so the existing
    /// Retry affordance applies.
    ///
    /// Internal (not `private`) so ``StreamCancelTests`` can pin the
    /// banner-classification contract directly, as it does for
    /// ``finishStartupCancellation``.
    func finishWithStartupFailure(
        placeholderIndex: Int,
        alias: String,
        epoch: Int? = nil
    ) {
        if let epoch, epoch != conversationEpoch { return }
        let message = "Couldn't start \(alias). Try again, or pick a different model in the box below."
        if var placeholder = currentMessage(index: placeholderIndex) {
            placeholder.status = .failed
            if placeholder.content.isEmpty { placeholder.content = message }
            updateMessage(at: placeholderIndex, with: placeholder)
        }
        lastError = message
        // Attribute the failure to the alias that actually failed, not to
        // whatever the picker happens to hold later.
        lastFailureAlias = alias
        // Classify the banner as a model-load failure (#590) rather than
        // letting it fall back to chatFailureKind("Couldn't start …") →
        // the generic `.requestFailed` diagnosis, whose Retry action just
        // re-runs the same failing start. `.modelLoadFailed` surfaces the
        // "check the model files or choose another model" copy + an Open
        // Model Management action, which matches the placeholder guidance.
        lastFailureKind = .modelLoadFailed
        isStreaming = false
    }

    /// The user hit Stop while the model was still being brought up — or
    /// in the instant it reached ``.ready``. Unlike
    /// ``finishWithStartupFailure`` this is a *deliberate* cancel, so it
    /// must NOT paint a "Couldn't start" failure row or raise the
    /// ``lastError`` banner: the same "Stop no longer masquerades as a
    /// failure" contract that governs mid-stream cancel applies to the
    /// cold-start path too.
    ///
    /// It resets the streaming state the ``runToolLoop`` ``defer`` would
    /// normally own — without this reset ``isStreaming`` sticks ``true``
    /// and the ``guard !isStreaming`` at the top of ``send`` wedges every
    /// future send in every session — and finalises the (usually empty)
    /// placeholder through the shared ``finaliseCancellation`` contract so
    /// it resolves to ``.complete`` + "Stopped." instead of spinning as
    /// ``.streaming`` forever.
    ///
    /// Internal (not `private`) purely so ``StreamCancelTests`` can pin
    /// the state-transition contract directly, exactly as it pins
    /// ``finaliseCancellation``.
    func finishStartupCancellation(
        placeholderIndex: Int,
        epoch: Int? = nil
    ) {
        if let epoch, epoch != conversationEpoch { return }
        if var placeholder = currentMessage(index: placeholderIndex) {
            Self.finaliseCancellation(message: &placeholder)
            updateMessage(at: placeholderIndex, with: placeholder)
        }
        isStreaming = false
    }

    /// Cancel the in-flight stream, finalising whatever the assistant
    /// has produced so far. The placeholder transitions to ``.complete``
    /// with whatever bytes already arrived.
    func stop() {
        inflight?.cancel()
    }

    /// Stop AND snapshot, synchronously, for the app-termination path.
    ///
    /// ``stop()`` only *requests* cancellation; the streaming task's own
    /// cleanup — ``isStreaming = false`` plus ``persistActive()`` — runs later
    /// on the main actor. During termination the main actor is then blocked
    /// for seconds reaping the server child, so that continuation never gets
    /// to run before ``ConversationStore.flush()``, and the partial final turn
    /// is written nowhere. Snapshot here instead, while we still hold the
    /// actor. The task's own cleanup remains harmless: it re-persists the same
    /// state if it ever gets to run.
    func stopAndPersist() {
        inflight?.cancel()
        guard isStreaming else { return }
        // Finalise through the SHARED cancellation contract before snapshotting.
        // Persisting a message still marked ``.streaming`` writes a turn that
        // reopens after relaunch as a permanent typing indicator, with no live
        // task left to ever complete or cancel it. Same transition the normal
        // stop path uses: ``.complete`` + "Stopped.", keeping whatever bytes
        // already arrived.
        if let idx = messages.indices.last,
           messages[idx].role == .assistant,
           messages[idx].status == .streaming,
           var last = currentMessage(index: idx) {
            Self.finaliseCancellation(message: &last)
            updateMessage(at: idx, with: last)
        }
        isStreaming = false
        persistActive()
    }

    /// Drop a stale chat-level error banner once the server provably
    /// reaches ``.ready`` again. The banner's copy is advice about a
    /// PAST failure ("Couldn't reach the model. Restart it from the
    /// model bar…") — the moment a restart succeeds, that advice has
    /// been followed and the banner is a lie. 2026-07 dogfood: the
    /// banner survived a manual stop → start cycle and kept accusing
    /// a model that was back to Ready. ContentView calls this on the
    /// ``.ready`` transition.
    ///
    /// Guarded on ``isStreaming`` out of caution only: ``send()``
    /// already clears ``lastError`` at turn start, so a live stream
    /// should never coexist with a stale banner.
    func clearStaleErrorBanner() {
        guard !isStreaming else { return }
        lastError = nil
        lastFailureKind = nil
        lastFailureAlias = nil
    }

    /// Pure transformation applied to the assistant placeholder when
    /// the user clicks Stop mid-stream. Lifted out of
    /// ``runOneStream`` so the contract — partial content stays, the
    /// row is marked ``.complete`` (NOT ``.failed`` — failed gets the
    /// red Retry bubble and visually drops the partial reply) with a
    /// short "Stopped." footer — can be pinned by tests without
    /// standing up the full async stream loop.
    ///
    /// Codex audit r1 (ChatViewModel.swift:737): if the model emitted
    /// a ``finish_reason: tool_calls`` chunk JUST before the user hit
    /// Stop, ``message.toolCalls`` was populated by the stream-loop
    /// handler but the tool-loop will never execute them — clearing
    /// the field here prevents a stale ``assistant(tool_calls)`` row
    /// from going out on the next wire body with no matching
    /// ``role: "tool"`` results (which most chat templates 400 on).
    static func finaliseCancellation(message: inout ChatMessage) {
        message.status = .complete
        message.errorMessage = "Stopped."
        message.toolCalls = nil
    }

    /// True when ``error`` is a cancellation, whichever shape it
    /// arrives in. ``Task.cancel()`` surfaces as Swift's
    /// ``CancellationError`` when it interrupts one of our own
    /// suspension points — but when the task is parked inside
    /// URLSession's async machinery at that moment, URLSession
    /// reports it as ``URLError(.cancelled)`` (NSURLErrorDomain
    /// -999) instead. 2026-07 dogfood: Stop mid-stream took the
    /// generic failure path and painted "Couldn't reach the model.
    /// Restart it…" over a perfectly healthy model, because only
    /// the ``CancellationError`` shape was routed to
    /// ``finaliseCancellation``. Both shapes must land there.
    nonisolated static func isCancellation(_ error: Error) -> Bool {
        if error is CancellationError { return true }
        let ns = error as NSError
        return ns.domain == NSURLErrorDomain && ns.code == NSURLErrorCancelled
    }

    /// v0.4.35: filter "empty-prose assistant" turns out of the wire
    /// body. Lifted out of ``runToolLoop`` so the contract — keep
    /// assistants whose prose is non-empty OR that carry tool_calls,
    /// drop the rest — can be pinned by tests. See the call site for
    /// the full motivation (model-switch silent-failure bug).
    static func filterEmptyAssistantsForWire(_ messages: [ChatMessage]) -> [ChatMessage] {
        var filtered: [ChatMessage] = []
        for msg in messages {
            guard msg.role == .assistant else {
                filtered.append(msg)
                continue
            }
            let proseEmpty = msg.content
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .isEmpty
            let noToolCalls = (msg.toolCalls?.isEmpty ?? true)
            guard proseEmpty && noToolCalls else {
                filtered.append(msg)
                continue
            }

            // If Stop landed before the first token, the transcript contains
            // an unanswered user prompt followed by an empty assistant row
            // captioned "Stopped.". Dropping only the assistant leaves that
            // prompt as the last historical turn, so the next send answers
            // the cancelled request instead of the new one. Remove the pair
            // from the wire history while keeping both rows visible in the UI.
            if msg.errorMessage == "Stopped.", filtered.last?.role == .user {
                filtered.removeLast()
            }
        }
        return filtered
    }

    /// Issue #477: strip forward-incompatible ``.unknown``-role messages
    /// out of the outbound wire body. Such rows only arise from decoding a
    /// ``sessions.json`` written by a newer build (a role the OpenAI
    /// schema grew, or a value an older build can't map) — the load path
    /// degrades them to ``.unknown`` so the sidebar isn't wiped, and the
    /// UI renders them as a neutral system note. But serialising
    /// ``{"role":"unknown", ...}`` into the next request would make the
    /// server 400 the send, so they MUST NOT reach the wire. This is
    /// wire-only: the rows stay visible in the transcript.
    static func filterUnknownRolesForWire(_ messages: [ChatMessage]) -> [ChatMessage] {
        messages.filter { $0.role != .unknown }
    }

    /// Issue #308 helper: returns the trimmed prose of the most
    /// recent ``.user`` message strictly before
    /// ``placeholderIndex`` in ``messages``, or ``""`` if no such
    /// message exists. Used by ``runOneStream`` to feed the
    /// calculator-shape detector in
    /// ``ChatMessage.shouldFlagToolNotCalled``.
    ///
    /// Walks backwards because a tool-call loop can have multiple
    /// assistant/tool rows between the user's prompt and the
    /// terminal assistant turn we're about to flag.
    static func lastUserPromptBefore(
        messages: [ChatMessage],
        placeholderIndex: Int
    ) -> String {
        guard placeholderIndex > 0 else { return "" }
        let upper = min(placeholderIndex, messages.count)
        for i in stride(from: upper - 1, through: 0, by: -1) {
            let m = messages[i]
            if m.role == .user {
                return m.content.trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        return ""
    }

    /// True when the assistant turn ending at ``placeholderIndex`` was
    /// preceded — since the most recent user message — by a tool that
    /// SUCCEEDED (a ``.tool`` result message with status ``.complete``).
    /// Walking back stops at the turn boundary (the user message) so a
    /// tool used in an EARLIER turn never counts.
    ///
    /// Feeds ``ChatMessage.shouldFlagToolNotCalled``'s
    /// ``toolSucceededThisTurn`` gate: a multi-step tool turn (the model
    /// calls e.g. ``calculator``, gets a good result, then writes a
    /// plain-language summary) leaves the final assistant message with no
    /// ``toolCalls`` of its own, so without this the "didn't call a tool"
    /// caption false-positives even though the tool-call chip is right
    /// there on screen.
    ///
    /// Success matters, not merely attempt: a tool that ERRORED (status
    /// ``.failed``, encoded at the ``.tool`` message per ``runToolLoop``'s
    /// result mapping) and left the model to hallucinate a raw answer is
    /// exactly the #308 failure mode, so that turn must still flag. The
    /// gate is DEFINITIVE success — ``.complete`` only, not "anything but
    /// failed": a ``.streaming`` (mid-flight) or ``.unknown`` (malformed /
    /// forward-compat) ``.tool`` row does not prove the tool produced a
    /// real result, and suppressing the warning is the unsafe direction
    /// (it would hide a raw-numeric hallucination). ``runToolLoop`` only
    /// ever writes ``.complete`` or ``.failed`` here, so this is purely
    /// defensive against restored / future envelopes.
    static func turnHadSuccessfulTool(
        messages: [ChatMessage],
        placeholderIndex: Int
    ) -> Bool {
        guard placeholderIndex > 0 else { return false }
        let upper = min(placeholderIndex, messages.count)
        for i in stride(from: upper - 1, through: 0, by: -1) {
            let m = messages[i]
            if m.role == .user { return false }        // turn boundary
            if m.role == .tool && m.status == .complete { return true }
        }
        return false
    }

    /// v0.5.11: silent sliding-window trim. ChatGPT / Claude desktop
    /// don't show users a token meter — they drop oldest turns behind
    /// the scenes when the conversation would exceed the model's
    /// context window. The previous "9.3k / 8k red chip" was both
    /// confusing (users don't know what 8k means) and useless (no
    /// affordance to fix the overflow). rapid-mlx's server doesn't
    /// enforce a window either — it just hands the full prompt to
    /// mlx-lm, which RoPE-extrapolates past training context and
    /// degrades quality silently. So the client has to do it.
    ///
    /// Contract:
    ///   * If ``contextWindow`` is ``nil`` or estimated tokens fit
    ///     under ``keepFraction * contextWindow``, return unchanged.
    ///   * Otherwise: split off a leading system row, walk the body
    ///     newest-to-oldest accumulating ``content.count / 4`` tokens,
    ///     stop when adding the next row would exceed the budget, and
    ///     drop everything before that cut point.
    ///   * The most recent message (the current user turn) is always
    ///     kept — even if it alone overshoots the budget, since
    ///     dropping it would mean sending no question at all.
    ///   * After cutting, drop leading non-user rows so the kept tail
    ///     never starts mid-tool-chain (a bare ``tool`` or
    ///     ``assistant(tool_calls)`` row at the head of a wire body is
    ///     a 400 with most chat templates).
    ///   * Re-attach the system row at index 0 if one was present.
    ///
    /// Token estimate is ``content.count / 4`` per message —
    /// OpenAI's published English rule-of-thumb. Order-of-magnitude
    /// is enough; the goal is keeping quality high, not hitting a
    /// precise count.
    static func trimMessagesForContextWindow(
        _ messages: [ChatMessage],
        contextWindow: Int?,
        keepFraction: Double = 0.75
    ) -> [ChatMessage] {
        guard let ctx = contextWindow, ctx > 0 else { return messages }
        guard !messages.isEmpty else { return messages }
        let budget = max(1, Int(Double(ctx) * keepFraction))
        // Codex audit r1 (ChatViewModel.swift:282): the pre-audit
        // shape only counted ``content.count`` and ignored
        // ``toolCalls.arguments`` — a model that emits a 50 KB JSON
        // tool argument blob (e.g. a stringified web-search payload)
        // would slip past the budget because the trimming logic
        // saw a near-empty assistant turn. Fold the serialized
        // tool-call arguments into the per-row cost so the budget
        // reflects the actual wire body. ``modelContent`` includes locally
        // extracted document text while keeping it out of the visible chat
        // bubble. Images use multimodal content parts and
        // are excluded here (token-count-per-image is model-specific
        // and not estimable from byte count alone).
        let perRowCost: (ChatMessage) -> Int = { msg in
            let contentChars = msg.modelContent.count
            let toolArgsChars = (msg.toolCalls ?? [])
                .reduce(0) { $0 + $1.function.arguments.count }
            return max(1, (contentChars + toolArgsChars) / 4)
        }
        let totalTokens = max(1, messages.reduce(0) { $0 + perRowCost($1) })
        if totalTokens <= budget { return messages }

        var system: ChatMessage? = nil
        var body = messages
        if body.first?.role == .system {
            system = body.removeFirst()
        }
        let systemTokens = system.map(perRowCost) ?? 0
        let bodyBudget = max(1, budget - systemTokens)

        var keep: [ChatMessage] = []
        var running = 0
        for msg in body.reversed() {
            let cost = perRowCost(msg)
            if keep.isEmpty {
                keep.append(msg)
                running += cost
                continue
            }
            if running + cost > bodyBudget { break }
            keep.append(msg)
            running += cost
        }
        keep.reverse()

        while let first = keep.first, first.role != .user {
            keep.removeFirst()
        }
        if keep.isEmpty, let last = body.last {
            keep = [last]
        }
        if let sys = system {
            keep.insert(sys, at: 0)
        }
        return keep
    }

    /// v0.4.35: classify a terminal stream as a soft failure when it
    /// produced no visible text and no tool calls. Lifted out of
    /// ``runOneStream`` so the contract can be tested independently
    /// of the async URLProtocol stream — the call site reads exactly
    /// the same fields out of the streamed ``current`` snapshot.
    ///
    /// Returns ``nil`` when the terminal is a real completion (any
    /// prose or any tool call present). Returns a non-nil error
    /// message when zero-content + zero-tool-calls — the caller
    /// applies it as ``errorMessage`` and flips ``status`` to
    /// ``.failed``.
    static func zeroContentFailureMessage(
        proseContent: String,
        toolCalls: [ToolCall]?,
        finishReason: String?,
        thinkingEnabled: Bool = false
    ) -> String? {
        // Back-compat shim around the richer ``classifyTerminal``
        // helper. Callers that don't know about ``reasoningContent``
        // (or that only need the hard-failure subset) keep working
        // unchanged. Anything that needs the cycle-2 reasoning
        // fallback path uses ``classifyTerminal`` directly.
        let outcome = classifyTerminal(
            proseContent: proseContent,
            reasoningContent: "",
            toolCalls: toolCalls,
            finishReason: finishReason,
            thinkingEnabled: thinkingEnabled
        )
        switch outcome {
        case .realCompletion, .reasoningOnlyTruncated:
            return nil
        case .emptyTurnFailure(let copy):
            return copy
        }
    }

    /// Classification of one stream's terminal moment. Cycle-2 fix
    /// for the F-002 / verbose-reasoning UX gap (filed
    /// 2026-06-19) where a reasoning-only assistant turn that hits
    /// ``max_tokens`` mid-think landed as an empty failed bubble
    /// — even though ``reasoning_content`` was populated.
    ///
    /// Three buckets:
    ///   * ``.realCompletion`` — prose OR a tool call landed. No
    ///     action needed.
    ///   * ``.reasoningOnlyTruncated(hint:)`` — prose empty, NO tool
    ///     calls, ``finish_reason == "length"``, AND reasoning is
    ///     populated. Caller leaves the row ``.complete`` and the
    ///     reasoning lane visible (the UI auto-expands the
    ///     disclosure for this state). ``hint`` is an actionable
    ///     soft caption ("hit max_tokens mid-reasoning — raise the
    ///     budget") that renders in the secondary, NOT red, lane.
    ///   * ``.emptyTurnFailure(message:)`` — every other empty
    ///     terminal (stop / nil / length-without-reasoning). Caller
    ///     flips the row to ``.failed`` with the message as the red
    ///     caption, exactly as the pre-cycle-2 code did.
    enum TerminalOutcome: Equatable {
        case realCompletion
        case reasoningOnlyTruncated(hint: String)
        case emptyTurnFailure(message: String)
    }

    static func classifyTerminal(
        proseContent: String,
        reasoningContent: String,
        toolCalls: [ToolCall]?,
        finishReason: String?,
        thinkingEnabled: Bool = false
    ) -> TerminalOutcome {
        let proseEmpty = proseContent
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .isEmpty
        let noToolCalls = (toolCalls?.isEmpty ?? true)
        if !(proseEmpty && noToolCalls) {
            return .realCompletion
        }
        let reasoningEmpty = reasoningContent
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .isEmpty
        // Cycle-2 F-002 fallback: the model burned its budget inside
        // the reasoning trace but produced *something* — the user
        // should see the (truncated) trace, not a red empty bubble.
        // Only fires on ``finish_reason == "length"`` because a
        // clean ``"stop"`` with reasoning-only output is much more
        // likely a parser / chat-template bug than a budget hit,
        // and the existing "switching models" copy is the right
        // pointer in that case.
        if finishReason == "length" && !reasoningEmpty {
            return .reasoningOnlyTruncated(
                hint: "Hit the Max Tokens limit with the answer still inside the reasoning trace. Open the Reasoning section to see the partial trace, then raise Max Tokens (Settings → Sampling) or simplify the prompt to get a final answer."
            )
        }
        if finishReason == "length" {
            // #161: hybrid models with thinking ON routinely burn
            // the entire 4 K default ``max_tokens`` budget inside
            // ``<think>...</think>`` on a 4 B / 9 B-class model and
            // emit zero answer tokens. Point the user at the toggle
            // directly when that's the likely cause.
            if thinkingEnabled {
                return .emptyTurnFailure(
                    message: "Hit the Max Tokens limit with the answer still inside the reasoning trace. Turn off Show reasoning (Settings → Sampling), or raise Max Tokens."
                )
            }
            return .emptyTurnFailure(
                message: "Hit the Max Tokens limit before any output. Raise Max Tokens, or try a shorter prompt."
            )
        }
        return .emptyTurnFailure(
            message: "The model returned no text. This sometimes happens right after switching models — try Regenerate, or start a new session."
        )
    }

    /// Edit a user turn in place: replace its prose, drop everything
    /// that came after it, and re-send. Matches ChatGPT Desktop's
    /// "edit message" pattern — the edit point becomes the new
    /// conversation tip.
    ///
    /// The replay stays on the CURRENT conversation id. An earlier build
    /// forked to a fresh id here so the pre-edit transcript survived as a
    /// recoverable branch, but the branch was indistinguishable from a real
    /// chat in the sidebar: every edit and every Retry silently spawned a
    /// duplicate row with the same title, so a few regenerations buried the
    /// history list under near-identical entries. Rewinding in place is what
    /// ChatGPT and Claude desktop do, and it is what the row the user is
    /// looking at appears to promise.
    @discardableResult
    func editUserMessage(
        id: UUID,
        newContent: String,
        alias: String
    ) -> Bool {
        guard !isStreaming else { return false }
        guard let idx = messages.firstIndex(where: { $0.id == id && $0.role == .user }) else { return false }
        let imageAttachments = messages[idx].imageAttachments
        let fileAttachments = messages[idx].fileAttachments
        let trimmed = newContent.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty || !imageAttachments.isEmpty || !fileAttachments.isEmpty else {
            return false
        }
        messages = Array(messages.prefix(idx))
        send(
            trimmed,
            alias: alias,
            imageAttachments: imageAttachments,
            fileAttachments: fileAttachments
        )
        return true
    }

    /// Drop the most recent assistant turn and resend the user turn that
    /// preceded it. Powers the per-message Regenerate button under each
    /// assistant bubble. No-op while a stream is in flight.
    func regenerateLast(alias: String) {
        guard !isStreaming else { return }
        guard let lastUserIndex = messages.lastIndex(where: { $0.role == .user }) else { return }
        let userMessage = messages[lastUserIndex]
        messages = Array(messages.prefix(lastUserIndex))
        send(
            userMessage.content,
            alias: alias,
            imageAttachments: userMessage.imageAttachments,
            fileAttachments: userMessage.fileAttachments
        )
    }

    /// Retry the turn that produced a specific assistant message. This is
    /// intentionally message-addressed: retrying an older response rewinds
    /// to the user prompt immediately before it instead of regenerating the
    /// latest turn by accident.
    @discardableResult
    func retryAssistantMessage(id: UUID, alias: String) -> Bool {
        guard !isStreaming else { return false }
        guard let assistantIndex = messages.firstIndex(where: {
            $0.id == id && $0.role == .assistant
        }) else { return false }
        guard let userIndex = messages[..<assistantIndex].lastIndex(where: {
            $0.role == .user
        }) else { return false }

        let userMessage = messages[userIndex]
        guard !userMessage.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                || !userMessage.imageAttachments.isEmpty
                || !userMessage.fileAttachments.isEmpty else {
            return false
        }
        // In place, on the SAME conversation id — see ``editUserMessage``
        // for why the old fork-into-a-branch behaviour was removed.
        messages = Array(messages.prefix(userIndex))
        send(
            userMessage.content,
            alias: alias,
            imageAttachments: userMessage.imageAttachments,
            fileAttachments: userMessage.fileAttachments
        )
        return true
    }

    /// Same as ``regenerateLast(alias:)`` but brings up ``newAlias``
    /// first so the regenerated turn is answered by the newly selected
    /// model. Surfaces a friendly error and bails without dropping the
    /// transcript tail if the new model fails to load.
    func regenerateLast(usingAlias newAlias: String) async {
        guard !isStreaming else { return }
        let trimmed = newAlias.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        if let server {
            let ok = await server.ensureServing(
                alias: trimmed,
                hfPath: nil,
                estimatedMemoryGB: nil,
                replacementGroup: .assistant
            )
            guard ok else {
                lastFailureKind = .modelLoadFailed
                lastError = FailureDiagnoser.diagnosis(
                    for: .modelLoadFailed,
                    modelAlias: trimmed
                ).message
                lastFailureAlias = trimmed
                return
            }
        }
        regenerateLast(alias: trimmed)
    }

    // MARK: - Tool round-trip loop

    /// Drive one user turn to completion.
    ///
    /// Each iteration streams one assistant turn into the placeholder at
    /// ``currentPlaceholder``. When the model finishes with
    /// ``finish_reason: "tool_calls"`` we run the referenced tools, append the
    /// results as ``role: "tool"`` rows, open a fresh assistant placeholder,
    /// and loop. Any other finish reason (or a transport failure, or Stop)
    /// ends the turn. Bounded by ``maxToolExecutions`` plus one tools-disabled
    /// synthesis round so a misbehaving model can't pin the loop forever.
    ///
    /// The KEEP-path wire hygiene is preserved on every round: empty-prose
    /// and forward-incompatible ``.unknown`` rows are stripped from the wire
    /// body, and the transcript is silently context-window trimmed (ChatGPT /
    /// Claude desktop behaviour).
    private func runToolLoop(
        alias: String,
        initialPlaceholder: Int,
        epoch: Int,
        forcedWebSearchQuery: String? = nil,
        globalInstruction: String = "",
        conversationInstruction: String = ""
    ) async {
        defer {
            // A stream that outlived a conversation switch must not reset
            // the NEW conversation's streaming state or clear a newer
            // in-flight task handle.
            if epoch == conversationEpoch {
                isStreaming = false
                inflight = nil
            }
        }
        var currentPlaceholder = initialPlaceholder
        var toolExecutionsLeft = maxToolExecutions
        var appGroundingSources: [GroundingSource] = []
        var isFinalSynthesisRound = false
        // dogfood-0810 BUG C: one-shot grounding-correction retry. Set when a
        // grounded synthesis answer denies real-time access; forces a single
        // tools-disabled correction round carrying ``groundingCorrectionPreamble``.
        var groundingCorrectionUsed = false
        var forceGroundingCorrection = false
        // The confabulated draft we are replacing, kept until the correction
        // round produces a usable answer. Restored if the retry stream fails,
        // is cancelled, or comes back empty — a wrong-but-present answer beats
        // a blank message.
        var draftBeforeCorrection: String?

        // `tool_choice:function` is advisory in several local chat templates:
        // the shipped 1.2B starter can ignore it and answer "I can search".
        // For an unambiguous fresh-information prompt, dispatch the harmless
        // search directly and give the model the same assistant(tool_calls) +
        // tool(result) transcript it would have produced itself. The model's
        // job is then only evidence synthesis, which is much more reliable.
        if let query = forcedWebSearchQuery,
           toolExecutionsLeft > 0,
           !query.isEmpty
        {
            let call = ToolCall(
                id: "app_search_\(UUID().uuidString)",
                name: "web_search",
                arguments: Self.webSearchArguments(query: query)
            )
            if var staged = currentMessage(index: currentPlaceholder) {
                staged.toolCalls = [call]
                staged.status = .complete
                updateMessage(at: currentPlaceholder, with: staged)
            }
            let result = await tools.run(call)
            appGroundingSources = Self.groundingSources(from: result.content)
            guard epoch == conversationEpoch, !Task.isCancelled else {
                finaliseCancelledPlaceholder(at: currentPlaceholder, epoch: epoch)
                return
            }
            let failureKind = result.failureKind ?? FailureDiagnoser.toolFailureKind(
                toolName: "web_search",
                content: result.content,
                isError: result.isError
            )
            _ = appendMessage(ChatMessage(
                role: .tool,
                content: result.content,
                status: (result.isError || failureKind != nil) ? .failed : .complete,
                errorMessage: failureKind.map { FailureDiagnoser.diagnosis(for: $0).message },
                failureKind: failureKind,
                toolCallID: result.toolCallID
            ))
            currentPlaceholder = appendMessage(ChatMessage(role: .assistant, status: .streaming))
            toolExecutionsLeft -= 1
        }

        while toolExecutionsLeft > 0 || isFinalSynthesisRound {
            // History for this request: everything BEFORE the streaming
            // placeholder. The placeholder itself is excluded because the
            // assistant hasn't said anything yet.
            var history = Array(messages.prefix(currentPlaceholder))
            // v0.4.35: strip empty-prose assistant turns from the wire body.
            // The UI still shows them — this is wire-only — but sending
            // ``{"role":"assistant","content":""}`` into a chat template is a
            // documented foot-gun (several templates treat an empty assistant
            // slot as "the model already finished" and immediately EOS).
            // Tool-call assistants (empty prose but ``tool_calls`` populated)
            // stay — they're load-bearing for the tool loop.
            history = ChatViewModel.filterEmptyAssistantsForWire(history)
            // Issue #477: drop any forward-incompatible ``.unknown``-role rows
            // so a serialised ``{"role":"unknown"}`` never 400s the send.
            history = ChatViewModel.filterUnknownRolesForWire(history)
            // Multi-model servers route each request by this selected alias.
            // ``servingAlias`` is the protected startup/default engine and is
            // no longer authoritative once secondary models are resident.
            let wireAlias = alias
            let definitions = isFinalSynthesisRound ? [] : ChatViewModel.wireDefinitions(
                forAlias: wireAlias,
                enabled: enabledDefinitions
            )
            let allowedToolNames = Set(definitions.map { $0.function.name })
            let knownToolNames = Set(tools.definitions.map { $0.function.name })
            // Ambient anti-confabulation guidance, prepended for the wire body
            // only (never appended to the transcript) so the user's history
            // stays prose-only. Skipped when no tools are advertised and — the
            // point of #1549 — on rounds that carry no tool result for it to
            // talk about. Existing/custom instructions are merged into the
            // same system row below.
            let ambientPreamble = !definitions.isEmpty
                && ChatViewModel.carriesToolResultForThisTurn(history)
                ? ChatViewModel.toolGuidancePreamble
                : nil
            // Inserted BEFORE the trim so its tokens are inside the budget the
            // trim works to, not added on top of a body already sized to fill
            // the window.
            history = ChatViewModel.addingInstructionLayers(
                to: history,
                ambientPreamble: ambientPreamble,
                global: globalInstruction,
                conversation: conversationInstruction
            )
            // v0.5.11 / issue #363: silent context-window trim against the
            // engine-reported window (captured on the last profile fetch),
            // falling back to the per-family heuristic in ``ModelInfoCatalog``.
            let ctxWindow = ModelInfoCatalog
                .info(
                    for: wireAlias,
                    hfRepo: nil,
                    serverContextWindow: sampling?.activeContextWindow
                )
                .contextWindow
            history = ChatViewModel.trimMessagesForContextWindow(
                history,
                contextWindow: ctxWindow
            )
            // The trim drops the oldest rows to fit and deliberately preserves
            // a leading system row, so on an over-budget turn it can carry the
            // preamble through while taking the tool result it describes. That
            // puts "your only source of truth is the tool result" on the wire
            // with no tool result behind it — #1549 again, just needing a long
            // enough conversation to reach. If the evidence didn't survive,
            // neither does the instruction.
            if let ambientPreamble,
                !ChatViewModel.carriesToolResultForThisTurn(history)
            {
                history = ChatViewModel.removingLeadingSystemComponent(
                    ambientPreamble,
                    from: history
                )
            }
            // Add this after the ambient/evidence consistency check above so
            // combining the two system instructions cannot defeat that guard.
            if isFinalSynthesisRound {
                history = ChatViewModel.addingToolBudgetSynthesisPreamble(to: history)
            }
            // BUG C: on the forced correction round, name the exact failure so
            // the retry has a stronger steer than the preamble that already
            // rode on the draft it is correcting.
            if forceGroundingCorrection {
                history = ChatViewModel.addingGroundingCorrectionPreamble(to: history)
            }
            let request: ChatStreamClient.Request
            if let s = sampling {
                let resolved = s.resolved(toolsEnabled: !definitions.isEmpty)
                request = ChatStreamClient.Request(
                    alias: wireAlias,
                    messages: history,
                    temperature: resolved.temperature,
                    topP: resolved.topP,
                    maxTokens: resolved.maxTokens,
                    repetitionPenalty: resolved.repetitionPenalty,
                    tools: definitions.isEmpty ? nil : definitions,
                    enableThinking: resolved.enableThinking
                )
            } else {
                request = ChatStreamClient.Request(
                    alias: wireAlias,
                    messages: history,
                    tools: definitions.isEmpty ? nil : definitions,
                    enableThinking: false
                )
            }
            let outcome = await runOneStream(
                placeholderIndex: currentPlaceholder,
                request: request,
                epoch: epoch
            )
            switch outcome {
            case .terminal:
                // BUG C: if this .terminal ends a grounding-correction round
                // that failed to produce a usable answer (transport error,
                // cancellation, or an empty stream), restore the original draft
                // — a wrong-but-present answer is better than a blank message.
                // A successful, non-empty correction is kept as-is.
                if let original = draftBeforeCorrection {
                    draftBeforeCorrection = nil
                    guard epoch == conversationEpoch else { return }
                    let corrected = currentMessage(index: currentPlaceholder)
                    let correctedText =
                        (corrected?.content ?? "")
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    // Keep the correction only if it produced a usable, non-
                    // failed answer of its own. Restore the original draft on
                    // failure, an empty stream, OR cancellation — a cancelled
                    // retry must never leave the blanked placeholder empty.
                    let correctionUsable =
                        !Task.isCancelled
                        && !correctedText.isEmpty
                        && corrected?.status != .failed
                    if !correctionUsable,
                        var restore = currentMessage(index: currentPlaceholder)
                    {
                        restore.content = original
                        restore.status = .complete
                        restore.errorMessage = nil
                        restore.failureKind = nil
                        updateMessage(at: currentPlaceholder, with: restore)
                    }
                    appendGroundingSources(
                        appGroundingSources,
                        to: currentPlaceholder,
                        epoch: epoch
                    )
                    return
                }
                // The model finished in prose, but if it denied having
                // real-time/current data while a SUCCESSFUL tool result for
                // THIS turn is on the wire, that answer confabulates a refusal
                // over live evidence. Force exactly one tools-disabled
                // correction round. Guarded so it fires at most once, only with
                // a successful tool result present, and never on a cancelled/
                // superseded turn. The ``!answerReliesOnEvidence`` clause spares
                // the caveat case ("I can't browse directly, but the results
                // show X"): a disclaimer in front of a grounded answer is not a
                // refusal.
                if !groundingCorrectionUsed,
                    epoch == conversationEpoch,
                    !Task.isCancelled,
                    ChatViewModel.carriesSuccessfulToolResultForThisTurn(
                        Array(messages.prefix(currentPlaceholder))
                    ),
                    let produced = currentMessage(index: currentPlaceholder)?.content,
                    ChatViewModel.looksLikeUngroundedRefusal(produced),
                    !ChatViewModel.answerReliesOnEvidence(produced)
                {
                    groundingCorrectionUsed = true
                    forceGroundingCorrection = true
                    isFinalSynthesisRound = true
                    // Stash the draft and reset the placeholder so the
                    // correction streams fresh instead of appending onto the
                    // refusal text. The stash is restored above if the retry
                    // does not produce a usable answer.
                    draftBeforeCorrection = produced
                    if var staged = currentMessage(index: currentPlaceholder) {
                        staged.content = ""
                        staged.toolCalls = nil
                        staged.status = .streaming
                        updateMessage(at: currentPlaceholder, with: staged)
                    }
                    continue
                }
                appendGroundingSources(
                    appGroundingSources,
                    to: currentPlaceholder,
                    epoch: epoch
                )
                return
            case .toolCallsPending(let calls):
                // The user can press Stop in the gap between the stream
                // returning .toolCallsPending and the loop below dispatching
                // the first tool. Honour cancellation here, and after each
                // tool, clearing the staged tool_calls on the assistant row so
                // the next wire body doesn't ship a half-finished tool round
                // with no matching results (most chat templates 400 on that).
                if Task.isCancelled {
                    finaliseCancelledPlaceholder(at: currentPlaceholder, epoch: epoch)
                    return
                }
                // A tools-disabled synthesis request should never produce a
                // structured call. Keep a defensive failure for malformed
                // model output rather than looping forever.
                if isFinalSynthesisRound {
                    failWithToolRoundCap(at: currentPlaceholder, epoch: epoch)
                    return
                }
                // Run each tool sequentially. Parallel execution via TaskGroup
                // trips Swift 6's region-based isolation analyzer on
                // @MainActor protocols; the tools here are network calls the
                // model rarely emits more than two of at once.
                var results: [ToolCallResult] = []
                for call in calls {
                    if Task.isCancelled {
                        finaliseCancelledPlaceholder(at: currentPlaceholder, epoch: epoch)
                        return
                    }
                    // A model may batch many calls into one assistant turn.
                    // Enforce the budget per requested call, and still emit a
                    // matching result for every skipped call so the transcript
                    // remains a valid assistant(tool_calls) → tool sequence.
                    guard toolExecutionsLeft > 0 else {
                        results.append(ToolCallResult(
                            toolCallID: call.id,
                            content: "Tool budget exhausted. Answer using the results already available.",
                            isError: true,
                            failureKind: .toolFailed
                        ))
                        continue
                    }
                    toolExecutionsLeft -= 1
                    // Refuse rather than dispatch when the tool was not
                    // advertised this round — a malformed model can emit a
                    // tool_call for a tool we never offered, and ``tools.run``
                    // would happily execute it. The refusal goes back as an
                    // error result so the model can recover in prose.
                    if let refusal = ChatViewModel.toolRefusalMessage(
                        name: call.function.name,
                        allowed: allowedToolNames,
                        known: knownToolNames
                    ) {
                        results.append(ToolCallResult(
                            toolCallID: call.id,
                            content: refusal,
                            isError: true,
                            failureKind: .toolFailed
                        ))
                        continue
                    }
                    let r = await tools.run(call)
                    results.append(r)
                    // A Stop pressed AFTER the tool resolved but BEFORE we
                    // append the result rows must still win. Exit BEFORE the
                    // append so the placeholder gets the standard cancel
                    // finalisation instead of a dangling tool_calls row.
                    if Task.isCancelled {
                        finaliseCancelledPlaceholder(at: currentPlaceholder, epoch: epoch)
                        return
                    }
                }
                guard epoch == conversationEpoch else { return }
                // Append role:"tool" messages for each result.
                for r in results {
                    let failureKind = r.failureKind ?? FailureDiagnoser.toolFailureKind(
                        toolName: calls.first(where: { $0.id == r.toolCallID })?.function.name ?? "",
                        content: r.content,
                        isError: r.isError
                    )
                    let msg = ChatMessage(
                        role: .tool,
                        content: r.content,
                        status: (r.isError || failureKind != nil) ? .failed : .complete,
                        errorMessage: failureKind.map { FailureDiagnoser.diagnosis(for: $0).message },
                        failureKind: failureKind,
                        toolCallID: r.toolCallID
                    )
                    _ = appendMessage(msg)
                }
                // Open the next assistant placeholder and loop.
                currentPlaceholder = appendMessage(ChatMessage(role: .assistant, status: .streaming))
                if toolExecutionsLeft == 0 {
                    isFinalSynthesisRound = true
                }
            }
        }
    }

    /// Finalise the streaming placeholder through the shared cancel contract.
    /// Used by the tool loop's several cancellation checkpoints.
    private func finaliseCancelledPlaceholder(at index: Int, epoch: Int) {
        guard epoch == conversationEpoch else { return }
        guard var stale = currentMessage(index: index) else { return }
        ChatViewModel.finaliseCancellation(message: &stale)
        updateMessage(at: index, with: stale)
    }

    /// Defensive fallback when a model emits a structured call even though
    /// the final synthesis request advertised no tools.
    private func failWithToolRoundCap(at index: Int, epoch: Int) {
        guard epoch == conversationEpoch else { return }
        let message = ChatViewModel.toolRoundCapMessage(cap: maxToolExecutions)
        if var capped = currentMessage(index: index) {
            capped.status = .failed
            capped.failureKind = .toolFailed
            capped.errorMessage = message
            capped.toolCalls = nil
            updateMessage(at: index, with: capped)
        }
        lastFailureKind = .toolFailed
        lastError = message
    }

    /// Copy for the round-cap failure. Static so a test can pin it without
    /// driving a full loop.
    static func toolRoundCapMessage(cap: Int) -> String {
        "The model could not finish after \(cap) tool calls. Try rephrasing, or turn a tool off."
    }

    /// Add the tools-disabled final-round instruction without introducing a
    /// second system row, which several local chat templates reject.
    nonisolated static func addingToolBudgetSynthesisPreamble(
        to messages: [ChatMessage]
    ) -> [ChatMessage] {
        var result = messages
        if result.first?.role == .system {
            result[0].content += "\n\n" + toolBudgetSynthesisPreamble
        } else {
            result.insert(
                ChatMessage(role: .system, content: toolBudgetSynthesisPreamble, status: .complete),
                at: 0
            )
        }
        return result
    }

    /// Wire-side filter — what actually ends up in the request body's ``tools``
    /// array. When ``alias`` is marked broken in ``ToolUseCapability``, returns
    /// ``[]`` so the model never sees tools it has been empirically proven to
    /// silently ignore or schema-leak. Static + pure so the strip can be
    /// pinned without spinning up a ``ChatViewModel``.
    nonisolated static func wireDefinitions(
        forAlias alias: String,
        enabled: [ToolDefinition]
    ) -> [ToolDefinition] {
        if ToolUseCapability.shouldDisableToolsChip(alias: alias) {
            return []
        }
        return enabled
    }

    /// Decide whether a model-emitted tool call should be REFUSED (never
    /// dispatched to ``tools.run``) rather than executed, and with what
    /// model-facing explainer. Returns nil when the call is allowed to run.
    ///
    /// Omitting a tool from the request body does NOT stop a malformed model
    /// emitting a call for it, so this is the load-bearing gate — not the
    /// wire filter. Refusing (rather than hard-discarding) lets the model
    /// answer in prose on the next round; the round cap backstops a model that
    /// keeps trying.
    nonisolated static func toolRefusalMessage(
        name: String,
        allowed: Set<String>,
        known: Set<String>
    ) -> String? {
        // Only a tool advertised (and enabled) THIS round may run. Everything
        // else is refused before dispatch — a disabled-but-shipped tool and a
        // name the model invented outright both get a recoverable prose nudge
        // rather than reaching ``tools.run``.
        if allowed.contains(name) { return nil }
        if known.contains(name) {
            return "tool '\(name)' isn't available in this conversation — answer directly, or ask the user to enable it in Settings."
        }
        let list = allowed.sorted().joined(separator: ", ")
        return "unknown tool '\(name)'\(list.isEmpty ? "" : " — available: \(list)"). Answer directly instead."
    }

    /// Ambient anti-confabulation guidance — prepended to the wire body on
    /// rounds where a tool result is actually in play.
    ///
    /// Small models routinely fire a tool, get faithful snippets back, then
    /// fabricate the rest of the list from training-data priors. The preamble
    /// is the cheapest mitigation against that.
    ///
    /// It is gated on a tool RESULT being present, not merely on a tool being
    /// advertised. The rules it states ("if a fact is not in the tool result,
    /// you DO NOT KNOW IT") describe how to read a result that exists; a model
    /// shown them with no result in context can only conclude it knows nothing.
    /// That is not hypothetical — issue #1549: with the built-in web tools
    /// advertised by default, the preamble rode along on every first turn and
    /// the shipped starter model answered "I don't have access to current or
    /// external data" to *what is the capital of France?*, a question it
    /// answers correctly the moment the preamble is absent.
    ///
    /// Both conditions are required rather than just the result: a transcript
    /// can carry ``.tool`` rows from an earlier round after the user has since
    /// disabled the tool, and re-asserting "your only source of truth is the
    /// tool result" would then bind the model to a result it can no longer
    /// refresh.
    ///
    /// Returns an empty array when the transcript already opens with a
    /// ``role: "system"`` row so we never ship competing system messages.
    /// Does this wire body carry a tool result for the turn being answered?
    ///
    /// Scoped to the rows after the last ``.user`` message, because a
    /// ``.tool`` row from an earlier question is not evidence about this one.
    /// Asking the whole transcript instead means a single weather lookup
    /// re-arms the preamble for every ordinary question that follows it —
    /// #1549 again, wearing a longer conversation.
    static func carriesToolResultForThisTurn(_ history: [ChatMessage]) -> Bool {
        let start =
            history.lastIndex { $0.role == .user }
            .map { history.index(after: $0) } ?? history.startIndex
        return history[start...].contains { $0.role == .tool }
    }

    /// Like ``carriesToolResultForThisTurn`` but requires usable current data
    /// from one of the built-in live-data tools. The grounding
    /// correction (BUG C) asserts "the tool result above was fetched just now
    /// and IS the current data", so it must not fire when the only tool result
    /// this turn is a failed/empty one (e.g. a search that errored): there the
    /// model's "I can't get current data" answer is correct, not a
    /// confabulation. Tool rows are joined back to their calls by ID so a
    /// successful calculator or MCP result cannot accidentally arm this
    /// live-data-specific retry.
    static func carriesSuccessfulToolResultForThisTurn(_ history: [ChatMessage]) -> Bool {
        let start =
            history.lastIndex { $0.role == .user }
            .map { history.index(after: $0) } ?? history.startIndex
        let turn = history[start...]
        let liveToolNames: Set<String> = ["web_search", "browse", "weather"]
        var liveCallIDs: Set<String> = []
        for message in turn where message.role == .assistant {
            for call in message.toolCalls ?? []
            where liveToolNames.contains(call.function.name) {
                liveCallIDs.insert(call.id)
            }
        }
        return turn.contains { message in
            guard message.role == .tool,
                message.status == .complete,
                let toolCallID = message.toolCallID,
                liveCallIDs.contains(toolCallID)
            else { return false }
            let content = message.content.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !content.isEmpty else { return false }
            // A successful transport can still produce no evidence. In that
            // case a refusal may be accurate, and the correction preamble must
            // not claim that current data exists.
            return !content.lowercased().contains("no results found")
        }
    }

    static func ambientSystemMessages(
        historyOpensWithSystem: Bool,
        toolsAdvertised: Bool,
        toolResultPresent: Bool
    ) -> [ChatMessage] {
        guard !historyOpensWithSystem, toolsAdvertised, toolResultPresent else {
            return []
        }
        return [ChatMessage(role: .system, content: toolGuidancePreamble, status: .complete)]
    }

    /// Merge app, pre-existing, global, and conversation instruction layers
    /// into one leading system row. Local chat templates often reject a second
    /// system message, so every caller must go through this transformation.
    nonisolated static func addingInstructionLayers(
        to messages: [ChatMessage],
        ambientPreamble: String?,
        global: String,
        conversation: String
    ) -> [ChatMessage] {
        var result = messages
        let existing = result.first?.role == .system ? result.removeFirst().content : nil
        var parts = [ambientPreamble, existing]
            .compactMap { $0.flatMap(normalizedInstruction) }
        if let global = normalizedInstruction(global) {
            parts.append("""
            [GLOBAL USER INSTRUCTIONS]
            These user preferences apply unless this conversation has a conflicting instruction:
            \(global)
            """)
        }
        if let conversation = normalizedInstruction(conversation) {
            parts.append("""
            [CONVERSATION INSTRUCTIONS - HIGHEST USER PRIORITY]
            These instructions apply only to this conversation. If they conflict with the global user instructions above, follow THESE conversation instructions. They do not override earlier application, safety, or tool instructions:
            \(conversation)
            """)
        }
        guard !parts.isEmpty else { return result }
        result.insert(
            ChatMessage(role: .system, content: parts.joined(separator: "\n\n"), status: .complete),
            at: 0
        )
        return result
    }

    /// Remove an exact first component from the merged system row. Used when
    /// context trimming drops the tool evidence that armed ambient guidance.
    nonisolated static func removingLeadingSystemComponent(
        _ component: String,
        from messages: [ChatMessage]
    ) -> [ChatMessage] {
        var result = messages
        guard result.first?.role == .system,
              let normalized = normalizedInstruction(component)
        else { return result }
        let separator = "\n\n"
        let prefix = normalized + separator
        if result[0].content == normalized {
            result.removeFirst()
        } else if result[0].content.hasPrefix(prefix) {
            result[0].content.removeFirst(prefix.count)
        }
        return result
    }

    nonisolated static func normalizedInstruction(_ value: String) -> String? {
        CustomInstructionsConfig.normalized(value)
    }

    static let toolGuidancePreamble: String = """
You have access to tools that fetch real-time information. When you use one of these tools, follow these rules — they OVERRIDE your training data:

1. Your ONLY source of truth for this turn is the tool result text. If a fact is not in the tool result, you DO NOT KNOW IT for the purposes of this answer. Your training data on this topic is OUT OF DATE and MUST NOT be used.

2. NEVER enumerate a list (teams, products, countries, dates, scores, names) from memory. If the user asks for a list and the tool result does NOT name the specific items, say so plainly. Do not produce any items from training data.

3. Forbidden phrases — they always signal you are reaching past the snippet: "based on common knowledge", "as is widely reported", "the following are typically considered", "generally speaking".

4. If only one or two items appear in the tool result, list ONLY those — do not extrapolate the rest of the bracket / list / table. State explicitly that this is partial coverage.

5. When the user's question is ambiguous about which subject the tool result covers, ask a clarifying question before answering.

6. For web results, cite the supporting source inline as a Markdown link using its exact title and URL. Never give a current-events answer without at least one clickable source.

7. If the search result only contains homepages or snippets that do not support a concrete answer, call web_search again with a more specific subject/date query. Do not merely offer to search and do not ask permission to use a tool that is already available.

8. If a tool result is an error, refusal, or user decline, state the reason written in that result. Never replace it with a different explanation, and never claim the tool lacks a capability unless the result itself says so.

These rules apply to every tool, not just web search.
"""

    /// Failure-specific instruction for the one correction round the tool loop
    /// forces when a grounded synthesis still denies having current data
    /// (dogfood-0810 BUG C). The anti-confabulation preamble already rode on
    /// the draft that failed, so a plain retry would likely repeat it; this
    /// names the exact mistake so the retry has a different, stronger steer.
    nonisolated static let groundingCorrectionPreamble: String = """
Your previous draft refused the question by claiming you lack real-time access or that your knowledge has a cutoff. That is FALSE for this turn: the tool result above was fetched just now and IS the current, real-time data. Answer the user's question again using ONLY the tool result. Do NOT mention a knowledge cutoff, a training date, or any inability to access real-time / current information. If the tool result genuinely does not contain the answer, say specifically what it is missing — do not fall back to a blanket refusal.
"""

    /// Prepend/merge the grounding-correction instruction onto the wire body of
    /// the forced correction round. Mirrors ``addingToolBudgetSynthesisPreamble``
    /// so both can stack on a single leading system row.
    nonisolated static func addingGroundingCorrectionPreamble(
        to messages: [ChatMessage]
    ) -> [ChatMessage] {
        var result = messages
        if result.first?.role == .system {
            result[0].content += "\n\n" + groundingCorrectionPreamble
        } else {
            result.insert(
                ChatMessage(
                    role: .system,
                    content: groundingCorrectionPreamble,
                    status: .complete
                ),
                at: 0
            )
        }
        return result
    }

    /// True when an assistant answer is a FIRST-PERSON temporal-denial refusal
    /// — "I can't access real-time information", "my knowledge is only up to
    /// 2024", etc. Every phrase carries its own first-person subject (``i ``,
    /// ``my ``, ``as of my ``, or the Chinese ``我``) so a grounded answer that
    /// reports a cutoff about a third party ("the model's knowledge cutoff is
    /// 2023") or quotes a tool result ("users were 'unable to browse'") is not
    /// flagged. Typographic apostrophes are normalized first so a curly
    /// ``can't`` matches too. Used ONLY with ``carriesToolResultForThisTurn``
    /// so the signal is "denied despite having evidence", never "no data".
    nonisolated static func looksLikeUngroundedRefusal(_ text: String) -> Bool {
        let value =
            text
            .lowercased()
            .replacingOccurrences(of: "\u{2019}", with: "'")  // ’ right single quote
            .replacingOccurrences(of: "\u{2018}", with: "'")  // ‘ left single quote
            .replacingOccurrences(of: "\u{02BC}", with: "'")  // ʼ modifier apostrophe
        guard !value.isEmpty else { return false }
        let firstPersonDenials = [
            "i can't access real-time", "i cannot access real-time",
            "i can't access current", "i cannot access current",
            "i can't access the internet", "i cannot access the internet",
            "i can't provide real-time", "i cannot provide real-time",
            "i can't provide current", "i cannot provide current",
            "i can't give you real-time", "i cannot give you real-time",
            "i don't have access to real-time", "i do not have access to real-time",
            "i don't have real-time", "i do not have real-time",
            "i don't have access to current", "i do not have access to current",
            "i don't have internet access", "i do not have internet access",
            "i have no access to real-time", "i have no access to current",
            "i'm unable to access real-time", "i am unable to access real-time",
            "i'm unable to access current", "i am unable to access current",
            "i'm unable to provide real-time", "i am unable to provide real-time",
            "i can't browse the internet", "i cannot browse the internet",
            "i can't browse", "i cannot browse",
            "i'm not able to browse", "i am not able to browse",
            "i don't have browsing", "i do not have browsing",
            "my knowledge is only up to", "my knowledge only goes up to",
            "my knowledge cutoff", "my knowledge is current up to",
            "my knowledge only extends to", "my knowledge ends in",
            "as of my last update", "as of my last training",
            "as of my knowledge cutoff", "my training data only goes",
            "my training only goes", "i was last trained", "my last training",
            // Chinese equivalents (the app ships a zh UI), anchored on the
            // first-person 我 so a third-party report does not match.
            "我无法访问实时", "我无法获取实时", "我无法提供实时",
            "我无法访问最新", "我无法获取最新", "我没有实时",
            "我不能访问实时", "我无法联网", "我无法上网", "我不能联网",
            "我的知识截止", "我的知识只到", "我的训练数据截止"
        ]
        return firstPersonDenials.contains(where: value.contains)
    }

    /// True when an answer visibly draws on the tool result — a citation link,
    /// or a CONCRETE reference to the fetched results. Used to spare the caveat
    /// case ("I can't browse directly, BUT the tool results show X"): the model
    /// prefaced a grounded answer with a disclaimer, so it did NOT confabulate.
    ///
    /// Deliberately concrete: only a link or an explicit reference to the
    /// fetched *results* counts. Vague connectors like "according to" or "the
    /// source" are excluded because they attach just as readily to a refusal
    /// ("According to my knowledge cutoff, I cannot provide current data") and
    /// would let a confabulation slip through the correction.
    nonisolated static func answerReliesOnEvidence(_ text: String) -> Bool {
        let value = text.lowercased()
        guard !value.isEmpty else { return false }
        let groundingSignals = [
            "](",  // a Markdown-linked source inside a formatted answer
            "the results", "the result show", "the result indicate",
            "search result", "the tool result", "the tool output",
            "the search returned", "the fetched",
            // Chinese concrete-result references.
            "搜索结果", "结果显示", "工具结果", "抓取"
        ]
        return groundingSignals.contains(where: value.contains)
    }

    // MARK: - Single-stream driver

    /// Outcome of one streamed assistant turn.
    private enum StreamOutcome {
        /// Either we got a non-tool finish reason, the user pressed Stop,
        /// or the transport failed. Either way, no further automatic
        /// requests should fire — the chat loop is done.
        case terminal
        /// finish_reason: "tool_calls" with a non-empty call list.
        /// Caller runs the tools, appends results, opens a new
        /// placeholder, and re-enters the loop.
        case toolCallsPending([ToolCall])
    }

    private func runOneStream(
        placeholderIndex: Int,
        request: ChatStreamClient.Request,
        epoch: Int
    ) async -> StreamOutcome {
        var current = currentMessage(index: placeholderIndex)
            ?? ChatMessage(role: .assistant, status: .streaming)
        var capturedCalls: [ToolCall] = []
        var capturedFinish: String?
        // v0.4.12: stream-start timestamp. Captured here rather
        // than via ``current.createdAt`` because the placeholder
        // may have been inserted seconds ago by the tool-call
        // loop (between rounds), and the user reads "elapsed
        // time" as "time the model spent on THIS round."
        let streamStart = ContinuousClock.now
        // #478: VoiceOver live-region feedback. Streaming replies are
        // otherwise silent to a screen-reader user — no start / progress
        // / completion signal. ``AssistantStreamAnnouncer`` is the pure,
        // throttled decision core; ``VoiceOverAnnouncer`` posts the
        // AppKit announcement. We read ``isVoiceOverEnabled`` ONCE here
        // and gate every announce path on it, so when VoiceOver is off
        // there is zero scanning / posting on the hot streaming path.
        let voiceOverActive = NSWorkspace.shared.isVoiceOverEnabled
        var announcer = AssistantStreamAnnouncer()
        // v0.4.13: capture server-reported usage if the server
        // honours ``stream_options.include_usage``. Either /
        // both fields may stay nil — non-conforming servers and
        // mid-stream cancellations leave them empty and the
        // stats caption falls back to the char-count estimate.
        var capturedPromptTokens: Int?
        var capturedCompletionTokens: Int?
        // The instant the first GENERATED token lands, on whichever channel
        // carries it. Everything before it is prefill (prompt processing),
        // which must not be charged to the decode rate the caption reports —
        // see ``MessageStats/decodeSeconds``.
        //
        // Reasoning counts. A thinking model emits its whole reasoning block
        // before the first prose token, and the server's `completion_tokens`
        // includes those tokens; starting the clock at the first `.content`
        // delta would divide every reasoning token by the prose-only window
        // and report a rate several times the real one.
        var firstTokenAt: ContinuousClock.Instant?
        do {
            // #17 desktop-half: thread the per-launch bearer through
            // every chat request. ``server.activeBearer`` rotates
            // each ServerManager.start() and clears on stop/crash,
            // so a stale leaked token is bounded to the live session.
            try await client.send(request, bearerToken: server?.activeBearer) { [weak self] event in
                guard let self else { return }
                switch event {
                case .firstToken(let at):
                    // The stream says the first generated token landed, on
                    // whichever lane carried it. Stamping per-lane here
                    // instead would miss a turn that opens with a tool-call
                    // fragment and time the later prose, reporting a decode
                    // window that excludes real generation.
                    if firstTokenAt == nil { firstTokenAt = at }
                case .content(let delta):
                    current.content += delta
                    // #478: announce the response start once, then the
                    // trailing un-announced sentence(s) on a throttled
                    // cadence. Both self-gate (start cue once; chunk only
                    // on a terminator + interval) so there is no
                    // per-token spam.
                    if voiceOverActive {
                        if let cue = announcer.firstTokenCue(fullContent: current.content) {
                            VoiceOverAnnouncer.announce(cue)
                        }
                        if let chunk = announcer.onDelta(
                            fullContent: current.content,
                            now: Date()
                        ) {
                            VoiceOverAnnouncer.announce(chunk)
                        }
                    }
                case .reasoning(let delta):
                    current.reasoning += delta
                case .toolCalls(let calls):
                    capturedCalls = calls
                    current.toolCalls = calls
                case .usage(let prompt, let completion):
                    capturedPromptTokens = prompt
                    capturedCompletionTokens = completion
                case .finished(let reason):
                    capturedFinish = reason
                    current.status = .complete
                    // v0.4.35 + cycle-2 (2026-06-19): classify the
                    // terminal moment via ``classifyTerminal``.
                    //
                    // Three buckets the helper distinguishes:
                    //
                    //   * ``.realCompletion`` — prose or tool call
                    //     landed. Leave the row ``.complete`` with no
                    //     caption.
                    //
                    //   * ``.emptyTurnFailure`` — every other empty
                    //     terminal. Flip to ``.failed`` with the red
                    //     caption, exact pre-cycle-2 behaviour.
                    //     Covers v0.4.35's three causes:
                    //       - ``reason == "length"`` — max_tokens hit
                    //         before the first token landed.
                    //       - ``reason == "stop"`` — clean stop with
                    //         empty content (model-switch history-tail
                    //         mismatch, or mid-warmup chat template).
                    //       - ``reason == nil`` — non-conforming
                    //         server emitted [DONE] with no choice
                    //         content.
                    //
                    //   * ``.reasoningOnlyTruncated`` — cycle-2
                    //     F-002 fix. ``reason == "length"`` AND
                    //     ``reasoning_content`` is populated AND the
                    //     visible ``content`` is empty. Don't flag as
                    //     a failure; surface a SOFT (secondary, NOT
                    //     red) caption pointing at Max Tokens and let
                    //     the ChatView's auto-expand path reveal the
                    //     partial reasoning trace. Without this branch
                    //     a reasoning model (phi-4-mini-reasoning,
                    //     deepseek_r1, nanbeige4.1) that exceeds the
                    //     budget mid-think left the user staring at
                    //     an empty red bubble even though useful
                    //     state was already on the row.
                    let outcome = ChatViewModel.classifyTerminal(
                        proseContent: current.content,
                        reasoningContent: current.reasoning,
                        toolCalls: current.toolCalls,
                        finishReason: reason,
                        thinkingEnabled: request.enableThinking
                    )
                    switch outcome {
                    case .realCompletion:
                        // Cycle-13 (2026-06-20) F-5: verbose-output
                        // length-truncation badge. A non-reasoning
                        // dense model (nemotron-30b-4bit and similar)
                        // that exhausts ``max_tokens`` mid-answer
                        // lands here as ``.realCompletion`` because
                        // ``content`` is non-empty — but the body the
                        // user sees is half-finished and reads as a
                        // real answer. Mark the row so the chat view
                        // can paint a subtle "Answer cut off (Max
                        // Tokens hit). Increase Max Tokens to see the
                        // rest." caption inline at the bottom of the
                        // bubble.
                        //
                        // The helper's gates make this a no-op for
                        // every other ``.realCompletion`` shape:
                        //   * clean ``stop`` / ``tool_calls`` — the
                        //     reason gate short-circuits.
                        //   * ``length`` + populated reasoning +
                        //     populated content — the reasoning-empty
                        //     gate short-circuits. This shape is
                        //     intentionally NOT badged: the row already
                        //     surfaces its reasoning disclosure and the
                        //     visible answer body landed, so the
                        //     "cut off mid-answer" framing of the
                        //     verbose-output badge doesn't apply.
                        //     ``reasoningTruncated`` does NOT cover this
                        //     shape either — its branch only fires on
                        //     empty content + populated reasoning + length
                        //     via ``.reasoningOnlyTruncated``.
                        //   * Codex r1 NIT (2026-06-20): comment used to
                        //     misattribute this case to PR #317; corrected
                        //     above.
                        current.contentTruncated = ChatMessage.shouldFlagContentTruncated(
                            content: current.content,
                            reasoning: current.reasoning,
                            finishReason: reason
                        )
                        // Issue #308 (2026-06-20): caption the row when
                        // the request offered tools but the model
                        // emitted zero ``tool_calls`` AND landed on a
                        // raw/numeric answer to a calculator- or
                        // search-shaped prompt. Catches the
                        // gemma3-1b-qat-4bit-style hallucinated-
                        // arithmetic failure mode on any model the
                        // user might pick. ``shouldFlagToolNotCalled``
                        // gates so a well-prosed answer or a non-
                        // tool-shaped prompt is left alone.
                        let lastUserPrompt = ChatViewModel.lastUserPromptBefore(
                            messages: self.messages,
                            placeholderIndex: placeholderIndex
                        )
                        current.toolNotCalledFlagged = ChatMessage.shouldFlagToolNotCalled(
                            userPrompt: lastUserPrompt,
                            assistantContent: current.content,
                            toolCalls: current.toolCalls,
                            finishReason: reason,
                            toolsRequested: !(request.tools?.isEmpty ?? true),
                            toolSucceededThisTurn: ChatViewModel.turnHadSuccessfulTool(
                                messages: self.messages,
                                placeholderIndex: placeholderIndex
                            )
                        )
                        // Issue #513 (defense-in-depth, layer 3): when
                        // the request offered tools but the model emitted
                        // zero ``tool_calls`` AND its content is just a
                        // malformed tool-call artifact (a raw
                        // ``<tool_call>`` / ``[TOOL_CALLS]`` /
                        // ``<｜tool▁calls▁begin｜>`` fragment or a bare
                        // tool-call-shaped JSON object the parser
                        // couldn't recover), flag the row so the chat
                        // view replaces the raw envelope dump with a
                        // quiet caption instead of surfacing machine
                        // syntax. ``shouldSuppressToolCallArtifact``
                        // gates tightly so a genuine JSON/code answer is
                        // never suppressed.
                        current.toolCallArtifactSuppressed = ChatMessage.shouldSuppressToolCallArtifact(
                            content: current.content,
                            toolCalls: current.toolCalls,
                            finishReason: reason,
                            toolsRequested: !(request.tools?.isEmpty ?? true)
                        )
                    case .reasoningOnlyTruncated(let hint):
                        current.errorMessage = hint
                        // Keep status .complete — the row carries
                        // useful reasoning state and the soft caption
                        // is informational, not an error. The
                        // .failed lane would paint the row red and
                        // hide the message behind Regenerate prompts.
                        // Codex r1 MAJOR-1: structural marker (not
                        // copy-string inference) so the chat-view
                        // can distinguish this case from
                        // ``finaliseCancellation``'s "Stopped." path,
                        // which ALSO lands as ``.complete`` with
                        // empty content + populated reasoning.
                        current.reasoningTruncated = true
                    case .emptyTurnFailure(let message):
                        current.errorMessage = message
                        current.status = .failed
                    }
                    // #478: announce the terminal moment ONCE, with a
                    // short cue — never the reply body. Skipped for the
                    // intermediate ``tool_calls`` finish (the turn is not
                    // over; a follow-up round streams next and announces
                    // its own completion). ``.failed`` speaks the error
                    // string; every other terminal is "Response complete".
                    if voiceOverActive, reason != "tool_calls" {
                        let terminal: AssistantStreamAnnouncer.Terminal =
                            current.status == .failed ? .failed : .complete
                        if let cue = announcer.onTerminal(
                            terminal,
                            errorMessage: current.errorMessage
                        ) {
                            VoiceOverAnnouncer.announce(cue)
                        }
                    }
                }
                self.writeStreamMessage(at: placeholderIndex, epoch: epoch, current)
            }
        } catch where Self.isCancellation(error) {
            // v0.4.29 pin: cancel MUST land as .complete (not .failed)
            // so the half-streamed content the user already saw stays
            // in the transcript with normal styling. Flipping to
            // .failed would paint a red "Retry" bubble and drop the
            // partial reply visually. Logic lifted into a static
            // helper so the contract is testable without async fan-out.
            ChatViewModel.finaliseCancellation(message: &current)
            writeStreamMessage(at: placeholderIndex, epoch: epoch, current)
            // #478: tell a screen-reader user the reply was stopped.
            if voiceOverActive, let cue = announcer.onTerminal(.cancelled, errorMessage: nil) {
                VoiceOverAnnouncer.announce(cue)
            }
            return .terminal
        } catch {
            current.status = .failed
            // Raw error → log for support; the user only ever sees
            // humanize()'s clean, jargon-free copy.
            print("[chat] stream failed: \(error.localizedDescription)")
            let failureKind = FailureDiagnoser.chatFailureKind(error: error)
            let actionable = FailureDiagnoser.diagnosis(
                for: failureKind,
                modelAlias: request.alias
            ).message
            current.errorMessage = actionable
            current.failureKind = failureKind
            lastFailureKind = failureKind
            lastError = actionable
            lastFailureAlias = request.alias
            writeStreamMessage(at: placeholderIndex, epoch: epoch, current)
            // #478: speak the failure to a screen-reader user (the error
            // string), so a silent red bubble isn't the only signal.
            if voiceOverActive, let cue = announcer.onTerminal(.failed, errorMessage: actionable) {
                VoiceOverAnnouncer.announce(cue)
            }
            return .terminal
        }
        // v0.4.12: end-of-stream stats. We attach for the plain-
        // text path (capturedFinish nil/"stop"/"length") AND for
        // the tool-call path — even though tool-call turns
        // continue to a follow-up round, the user sees the
        // elapsed time for THIS turn's reasoning + decision.
        // Skipped only when the message ended up ``.failed`` (a
        // crashed mid-stream "produced" 1.7 s and 41 chars but
        // that's noise, not throughput).
        if current.status == .complete && !current.content.isEmpty {
            let elapsed = streamStart.duration(to: .now).seconds
            current.stats = MessageStats(
                elapsedSeconds: elapsed,
                charCount: current.content.count,
                promptTokens: capturedPromptTokens,
                completionTokens: capturedCompletionTokens,
                timeToFirstTokenSeconds: firstTokenAt
                    .map { streamStart.duration(to: $0).seconds },
                reasoningEmitted: !current.reasoning.isEmpty
            )
            writeStreamMessage(at: placeholderIndex, epoch: epoch, current)
        }
        if capturedFinish == "tool_calls" && !capturedCalls.isEmpty {
            return .toolCallsPending(capturedCalls)
        }
        return .terminal
    }

    /// Map a raw transport / SSE error into an actionable single-
    /// sentence message the user can read without knowing what
    /// "URLError -1001" means. The categories match the realistic
    /// failure modes we hit against rapid-mlx + a local server:
    ///
    ///   * ``streamTruncated`` — server died mid-response (#896);
    ///     the dedicated copy already lives on the error itself.
    ///   * Timeout — model is generating but URLSession's idle
    ///     limit (600 s) expired; suggest raising max_tokens
    ///     budget or restarting.
    ///   * Connection refused / can't reach host — server isn't
    ///     listening; suggest Restart from the picker / banner.
    ///   * HTTP non-2xx — the request was rejected; show a plain,
    ///     actionable recovery message. The raw status code + server
    ///     body are engine internals and are NOT shown to the user
    ///     (they belong in the logs).
    ///   * Anything else — fall back to localizedDescription so a
    ///     rare error still has SOMETHING readable instead of an
    ///     empty bubble.
    nonisolated static func humanize(_ error: Error) -> String {
        if let chat = error as? ChatStreamError {
            switch chat {
            case .streamTruncated:
                // The stream ended early — almost always the model
                // crashing mid-reply. Give a plain recovery path; the
                // raw engine detail stays in the logs (principle: error
                // copy must be human + actionable).
                return "Rapid lost the model mid-reply — it may have crashed. Restart it from the model bar at the top and try again."
            case .httpStatus(_, let body):
                // #471: a genuine capacity rejection (out-of-memory
                // admission cap, or the server busy finishing another
                // reply) has a *specific* recovery the generic message
                // hides. The raw status code + body still never reach the
                // user — only the classification does; diagnostics stay in
                // the logs.
                switch capacityKind(from: body) {
                case .outOfMemory: return outOfMemoryMessage
                case .serverBusy: return serverBusyMessage
                case .none:
                    return "Rapid couldn't complete that request. Try again, or restart the model from the bar at the top."
                }
            case .transport(let message):
                // #471: a memory cap that trips mid-generation surfaces
                // here (the SSE stream had already opened), not as an HTTP
                // status. Same classification so the user gets the OOM /
                // busy recovery path regardless of *when* the cap fired.
                switch capacityKind(from: message) {
                case .outOfMemory: return outOfMemoryMessage
                case .serverBusy: return serverBusyMessage
                case .none:
                    return "Rapid lost its connection to the model. Restart it from the bar at the top and try again."
                }
            }
        }
        let ns = error as NSError
        if ns.domain == NSURLErrorDomain {
            switch ns.code {
            case NSURLErrorTimedOut:
                return "The model stopped responding (nothing for 10 minutes). It may be stuck — restart it from the model bar at the top, or try a shorter message."
            case NSURLErrorCannotConnectToHost, NSURLErrorCannotFindHost:
                return "Can't reach the model. Use the model bar at the top to restart it."
            case NSURLErrorNetworkConnectionLost:
                return "The model disconnected mid-reply. Restart it from the model bar at the top and try again."
            case NSURLErrorNotConnectedToInternet:
                return "macOS says the network is off, but Rapid runs entirely on your Mac, so this usually doesn't matter. Restart the model from the model bar at the top; if that fails, restart your Mac."
            default:
                // Don't surface the raw NSURLError code/body (e.g.
                // "NSURLErrorDomain error -1004") — the diagnostic is
                // logged at the call site; the user gets a clean path.
                return "Couldn't reach the model. Restart it from the model bar at the top and try again."
            }
        }
        // Anything else (system / library error — e.g. a decode failure,
        // a cancelled task) is NOT one of our authored user-facing errors;
        // its localizedDescription is a raw diagnostic. The caller already
        // logged the raw error, so the user gets a plain, actionable path.
        return "Rapid couldn't complete that request. Try again, or restart the model from the bar at the top."
    }

    /// #471: the capacity failure modes that deserve a *specific* recovery
    /// message instead of the generic "couldn't complete that request".
    enum CapacityKind: Equatable { case outOfMemory, serverBusy, none }

    /// Classify a raw sidecar error body/message into a capacity failure
    /// mode. Matching is substring + case-insensitive against the server's
    /// own wording — the ``BackpressureError`` text from the D-METAL-CAP
    /// memory-admission gate (a genuine OOM: the model + KV won't fit) and
    /// the max-concurrent-requests gate (transient busy). Engine internals
    /// never reach the user; only this classification does. The signals are
    /// distinctive phrases (not bare tokens like "oom") so they don't
    /// collide with unrelated errors.
    nonisolated static func capacityKind(from raw: String) -> CapacityKind {
        let t = raw.lowercased()
        // Genuine out-of-memory: the D-METAL-CAP admission gate rejected
        // the request because weights + projected KV exceed the GPU cap.
        // Signals are OOM-SPECIFIC on purpose — a bare "would exceed" would
        // collide with the context-length error ("... would exceed model
        // context ...") whose fix is a shorter *prompt*, not a smaller
        // model. The METAL-CAP 503 body still matches on the specific
        // phrases below.
        let memorySignals = [
            "metal-cap", "gpu_memory_utilization",
            "projected kv", "metal active", "out of memory",
            "insufficient memory",
        ]
        if memorySignals.contains(where: { t.contains($0) }) { return .outOfMemory }
        // Concurrency backpressure — the server is busy, not out of memory.
        // Waiting fixes it; a smaller model does not.
        let busySignals = ["max_concurrent_requests", "server is busy", "max concurrent"]
        if busySignals.contains(where: { t.contains($0) }) { return .serverBusy }
        return .none
    }

    /// #471 acceptance: a genuine OOM must read as human + actionable, not
    /// "Transport error / Internal error during streaming". Points at the
    /// two levers the user actually controls. (The #968 sidecar fix removes
    /// the *false* rejects; this copy is for the requests that truly don't
    /// fit.)
    nonisolated static let outOfMemoryMessage =
        "This reply needs more memory than your Mac has free right now. Try a smaller model from the bar at the top, or ask for a shorter response."

    /// #471: concurrency backpressure is transient — distinct from OOM so
    /// the user waits instead of needlessly downsizing their model.
    nonisolated static let serverBusyMessage =
        "Rapid is busy finishing another reply. Give it a moment, then try again."
}
