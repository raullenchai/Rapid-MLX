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
/// or when ``maxToolRounds`` is hit.
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

    /// Identity of the conversation ``messages`` currently holds. A fresh
    /// UUID on launch (opens to an empty "Ask anything"); ``persistActive``
    /// upserts under this id once the user sends.
    private(set) var activeConversationID = UUID()

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

    /// Set while a stream is in flight. UI reads this to show the stop
    /// button instead of send.
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

    /// v0.4.14: user-mutable sampling knobs. Optional in the init
    /// signature so existing tests don't have to spin one up — they
    /// fall back to the v0.4.12 hard-coded defaults via
    /// ``ChatStreamClient.Request``'s own default parameters.
    /// Production code (``RapidApp.init``) always passes a real
    /// ``SamplingConfig`` reading from ``UserDefaults``.
    let sampling: SamplingConfig?

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
        sampling: SamplingConfig? = nil,
        server: ServerManager? = nil
    ) {
        self.client = client
        self.sampling = sampling
        self.server = server
        self.conversations = ConversationStore.load()
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
                conversation.title = title
            }
        } else {
            conversations.insert(
                ChatConversation(
                    id: activeConversationID,
                    title: title,
                    messages: messages,
                    createdAt: now,
                    updatedAt: now
                ),
                at: 0
            )
        }
        ConversationStore.save(conversations)
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
            activeConversationID = UUID()
            isStreaming = false          // messages now empty → persistActive no-ops
            lastError = nil
            lastFailureKind = nil
            lastFailureAlias = nil
        }
        conversations.removeAll { $0.id == id }
        ConversationStore.save(conversations)
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

    /// Append the user message, open a placeholder assistant row, and
    /// kick off the streaming task. The text field clears immediately on
    /// the caller's side.
    func send(
        _ text: String,
        alias: String
    ) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        guard !isStreaming else { return }

        let user = ChatMessage(
            role: .user,
            content: trimmed,
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
        inflight = Task { [weak self] in
            guard let self else { return }

            // Bring the model up if it isn't serving yet. The user's
            // turn is already in the transcript, so a load that takes a
            // minute — or fails — costs them nothing they typed.
            // `ensureServing` short-circuits when we are already serving
            // this alias, so the warm path pays only a state read.
            if let server {
                let ready = await server.ensureServing(alias: alias, hfPath: startupHFPath)
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

            await self.runSingleStream(
                alias: alias,
                placeholderIndex: placeholderIndex,
                epoch: epoch
            )
        }
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
        messages.filter { msg in
            guard msg.role == .assistant else { return true }
            let proseEmpty = msg.content
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .isEmpty
            let noToolCalls = (msg.toolCalls?.isEmpty ?? true)
            return !(proseEmpty && noToolCalls)
        }
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
        // reflects the actual wire body. Attachments are already
        // inlined into ``content`` by ``composeProseWithFileAttachments``
        // at send-time, so the content-byte count already covers
        // text attachments. Images use multimodal content parts and
        // are excluded here (token-count-per-image is model-specific
        // and not estimable from byte count alone).
        let perRowCost: (ChatMessage) -> Int = { msg in
            let contentChars = msg.content.count
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
    /// conversation tip, no branching, no orphan replies.
    @discardableResult
    func editUserMessage(
        id: UUID,
        newContent: String,
        alias: String
    ) -> Bool {
        guard !isStreaming else { return false }
        let trimmed = newContent.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        guard let idx = messages.firstIndex(where: { $0.id == id && $0.role == .user }) else { return false }
        // Truncate everything from the edited row onward, then resend.
        messages = Array(messages.prefix(idx))
        send(trimmed, alias: alias)
        return true
    }

    /// Drop the most recent assistant turn and resend the user turn that
    /// preceded it. Powers the per-message Regenerate button under each
    /// assistant bubble. No-op while a stream is in flight.
    func regenerateLast(alias: String) {
        guard !isStreaming else { return }
        guard let lastUserIndex = messages.lastIndex(where: { $0.role == .user }) else { return }
        let userText = messages[lastUserIndex].content
        messages = Array(messages.prefix(lastUserIndex))
        send(userText, alias: alias)
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
            let ok = await server.ensureServing(alias: trimmed)
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

    // MARK: - Single-stream driver

    /// Stream one assistant turn into the placeholder at
    /// ``placeholderIndex``. The minimal menu-bar app is a plain
    /// streaming chat — no tools, no tool-call round-trip — so a single
    /// stream is the whole turn. The KEEP-path wire hygiene is preserved:
    /// empty-prose and forward-incompatible ``.unknown`` rows are stripped
    /// from the wire body, and the transcript is silently context-window
    /// trimmed (ChatGPT / Claude desktop behaviour).
    private func runSingleStream(
        alias: String,
        placeholderIndex: Int,
        epoch: Int
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
        // History for this request: everything BEFORE the streaming
        // placeholder. The placeholder itself is excluded because the
        // assistant hasn't said anything yet.
        var history = Array(messages.prefix(placeholderIndex))
        // v0.4.35: strip empty-prose assistant turns from the wire body.
        // The UI still shows them — this is wire-only — but sending
        // ``{"role":"assistant","content":""}`` into a chat template is a
        // documented foot-gun (several templates treat an empty assistant
        // slot as "the model already finished" and immediately EOS).
        history = ChatViewModel.filterEmptyAssistantsForWire(history)
        // Issue #477: drop any forward-incompatible ``.unknown``-role rows
        // so a serialised ``{"role":"unknown"}`` never 400s the send.
        history = ChatViewModel.filterUnknownRolesForWire(history)
        // v0.5.1: outgoing ``model:`` is the alias the server is ACTUALLY
        // serving right now, falling back to the caller-supplied alias
        // until the server reports ``.ready``.
        let wireAlias = server?.servingAlias ?? alias
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
        let request: ChatStreamClient.Request
        if let s = sampling {
            let resolved = s.resolved(toolsEnabled: false)
            request = ChatStreamClient.Request(
                alias: wireAlias,
                messages: history,
                temperature: resolved.temperature,
                topP: resolved.topP,
                maxTokens: resolved.maxTokens,
                repetitionPenalty: resolved.repetitionPenalty,
                enableThinking: resolved.enableThinking
            )
        } else {
            request = ChatStreamClient.Request(
                alias: wireAlias,
                messages: history,
                enableThinking: false
            )
        }
        _ = await runOneStream(
            placeholderIndex: placeholderIndex,
            request: request,
            epoch: epoch
        )
    }

    // MARK: - Legacy tool round-trip loop (removed)

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
        let streamStart = Date()
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
        do {
            // #17 desktop-half: thread the per-launch bearer through
            // every chat request. ``server.activeBearer`` rotates
            // each ServerManager.start() and clears on stop/crash,
            // so a stale leaked token is bounded to the live session.
            try await client.send(request, bearerToken: server?.activeBearer) { [weak self] event in
                guard let self else { return }
                switch event {
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
            let elapsed = Date().timeIntervalSince(streamStart)
            current.stats = MessageStats(
                elapsedSeconds: elapsed,
                charCount: current.content.count,
                promptTokens: capturedPromptTokens,
                completionTokens: capturedCompletionTokens
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
