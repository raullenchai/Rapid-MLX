import AppKit
import MarkdownUI
import SwiftUI
import UniformTypeIdentifiers

/// Sanitizes untrusted transcript text for UI and clipboard surfaces
/// without mutating the stored chat history or wire payload.
///
/// ## Delta-safety contract (#296)
///
/// ``sanitize`` is a pure per-scalar filter — each unicode scalar maps
/// to either itself, ``lineFeed``, or nothing, with no context-sensitive
/// operations (no ``precomposedStringWithCanonicalMapping``, no
/// whitespace-run normalisation, no ZWJ-sequence handling). That makes
/// the function **delta-safe**:
///
///   ``sanitize(a + b) == sanitize(a) + sanitize(b)``
///
/// for every pair of strings. The streaming UI exploits this via
/// ``Memo`` to skip O(buffer) work on every coalescer flush — a 20K-char
/// buffer pays only the delta-sanitise cost, not the full re-sanitise.
enum ChatTextSanitizer {
    private static let lineFeed = UnicodeScalar(0x0A)!

    static func sanitizeForDisplay(_ raw: String) -> String {
        sanitize(raw)
    }

    static func sanitizeForPasteboard(_ raw: String) -> String {
        sanitize(raw)
    }

    /// Delta-aware memoiser for the streaming chat surface (#296).
    ///
    /// The streaming display buffer grows monotonically — each new
    /// coalescer flush appends a small suffix to the previous buffer.
    /// Naive ``sanitizeForDisplay`` ran the whole buffer through the
    /// per-scalar filter on every flush: 62 µs @ 500 chars, 903 µs @
    /// 8K chars, 2.3 ms @ 20K chars. At 60 flushes/sec on a 2K-token
    /// reply that's ~54 ms/sec of pure CPU re-sanitising the same
    /// prefix.
    ///
    /// ``Memo`` caches the last sanitised prefix + its source-byte
    /// count. On the common growing-buffer path we sanitise only the
    /// suffix delta and append.
    ///
    /// ## Caller contract: monotone extension
    ///
    /// The memo is fast because it **trusts** that consecutive calls
    /// see strings that monotonically extend (or shrink to ≤ 0 bytes,
    /// or replace entirely with a new prefix). Verifying the prefix
    /// on every call would require O(buffer) byte compares — exactly
    /// what we're trying to avoid.
    ///
    /// Defence-in-depth: the memo detects shrunken buffers
    /// (``raw.utf8.count < lastRawUtf8Count``) and falls back to a
    /// full re-sanitise. A NEW prefix of the SAME length OR a
    /// LONGER buffer whose first ``lastRawUtf8Count`` bytes differ
    /// from the previous call is **caller error** — the View must
    /// call ``reset()`` on every non-monotonic transition (different
    /// message id, regenerate-from-here, edit-and-resend).
    ///
    /// In the production SwiftUI usage (``MessageRow`` ``@State``),
    /// SwiftUI rebuilds the row on every ``ChatMessage.id`` change,
    /// which gives us a fresh ``Memo`` automatically — so the caller-
    /// error path is unreachable from production code today.
    ///
    /// Thread model: not synchronised. Designed for SwiftUI ``@State``
    /// where the owning view is main-actor-bound; the memo lives
    /// alongside one ``ChatMessage`` and is read+written from the
    /// same actor.
    ///
    /// Correctness rests on the ``sanitize(a + b) == sanitize(a) +
    /// sanitize(b)`` invariant in the type-level docstring above. If
    /// ``sanitize`` is ever changed to use context-sensitive
    /// operations (Unicode normalisation, ZWJ handling, etc.) the
    /// delta-safety claim breaks AND the ``ChatTextSanitizerTests``
    /// memo-equivalence suite below must also be updated.
    @MainActor
    final class Memo {
        /// UTF-8 byte count of the last raw input we sanitised. We
        /// key on byte count (not character count) so the prefix
        /// check is cheap — ``String.utf8.count`` is O(1) under
        /// COW; ``String.count`` is O(n).
        private var lastRawUtf8Count: Int = 0
        /// Cached sanitised output for the prefix of length
        /// ``lastRawUtf8Count`` in UTF-8 bytes of the raw stream.
        private var lastSanitisedPrefix: String = ""

        /// Reset the memo. Call when the streaming source switches
        /// (different message id, regenerate, edit-and-resend) so
        /// the next ``sanitised`` call runs a full pass against the
        /// fresh source. In production SwiftUI usage SwiftUI's
        /// row-rebuild on identity change makes this implicit — the
        /// API is exposed for tests and for any future caller that
        /// pools memos across message identities.
        func reset() {
            lastRawUtf8Count = 0
            lastSanitisedPrefix = ""
        }

        /// Returns ``sanitize(raw)`` using the cached sanitised
        /// prefix when the new raw input monotonically extends the
        /// previous one. **Trusts** that the unchanged prefix bytes
        /// equal what we cached — see the type-level "caller
        /// contract" note. Falls back to a full sanitise when the
        /// new buffer is shorter than the cached prefix.
        func sanitised(_ raw: String) -> String {
            let newCount = raw.utf8.count
            if newCount == lastRawUtf8Count {
                // No change — return cached value, no per-scalar work.
                return lastSanitisedPrefix
            }
            if newCount > lastRawUtf8Count && lastRawUtf8Count > 0 {
                // Hot path: sanitise only the new suffix. We slice
                // the suffix off the raw String via UTF-8 view so
                // we avoid the Array<UInt8> copy the previous draft
                // took. ``String(Substring.UTF8View)`` is failable
                // (sub-UTF-8-boundary slices) — production callers
                // only ever extend at scalar boundaries, but we
                // fall through to the cold path on a nil decode
                // anyway, so the failure mode is a slightly slower
                // sanitise, never wrong output.
                let utf8 = raw.utf8
                let suffixStart = utf8.index(utf8.startIndex, offsetBy: lastRawUtf8Count)
                if let suffix = String(utf8[suffixStart...]) {
                    let sanitisedSuffix = ChatTextSanitizer.sanitize(suffix)
                    let combined = lastSanitisedPrefix + sanitisedSuffix
                    lastRawUtf8Count = newCount
                    lastSanitisedPrefix = combined
                    return combined
                }
                // Fall through to the cold path — the suffix slice
                // straddles a multi-byte UTF-8 scalar boundary.
                // Shouldn't happen during normal streaming (the
                // model emits whole scalars per chunk) but rather
                // be safe than emit U+FFFD replacement noise.
            }
            // Cold path: first call OR raw shrunk OR explicit reset.
            let combined = ChatTextSanitizer.sanitize(raw)
            lastRawUtf8Count = newCount
            lastSanitisedPrefix = combined
            return combined
        }
    }

    fileprivate static func sanitize(_ raw: String) -> String {
        let scalars = raw.unicodeScalars.compactMap(sanitizedScalar)
        return String(String.UnicodeScalarView(scalars))
    }

    private static func sanitizedScalar(_ scalar: UnicodeScalar) -> UnicodeScalar? {
        switch scalar.value {
        case 0x09, 0x0A:
            return scalar
        case 0x0D:
            return lineFeed
        case 0x00...0x08, 0x0B...0x1F, 0x7F...0x9F:
            return nil
        case 0x061C, 0x200E, 0x200F, 0x202A...0x202E, 0x2066...0x2069:
            return nil
        default:
            return scalar
        }
    }
}

@MainActor
private func copySanitizedToPasteboard(_ raw: String) {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(ChatTextSanitizer.sanitizeForPasteboard(raw), forType: .string)
}
/// Main chat surface. ChatGPT Desktop's layout: messages scroll the
/// top region, the compose bar is pinned at the bottom. Streaming
/// responses pin the scroll to the trailing edge so the user sees
/// tokens land in real time.
///
/// The minimal menu-bar app keeps a single ephemeral conversation
/// (``ChatViewModel.messages``) — no sidebar, history, presets, tools,
/// or attachments.
struct ChatView: View {
    @Bindable var viewModel: ChatViewModel
    /// Compiled blocks for the streaming message. Keeps the raw streamed
    /// string out of the streaming view's inputs — see
    /// `StreamingMarkdownStore`. (`MessageRow` still receives the raw
    /// `ChatMessage` for non-markdown concerns like tool chips.)
    @State private var streamingMarkdown = StreamingMarkdownStore()
    @Bindable var server: ServerManager
    @Binding var alias: String
    /// The single readiness value for this window, resolved once by
    /// ``ContentView`` and shared with the Launch page. Both the chat
    /// hero and the composer read their copy off this, so they cannot
    /// describe the same lifecycle differently.
    var readiness: ModelReadiness
    /// First launch must not inspect model caches behind the consent sheet.
    /// The parent flips this after the user makes that one-time decision.
    var catalogRefreshEnabled: Bool = true
    /// Explicit picker gesture signal used by launch auto-start arbitration.
    /// Catalog-driven default selection intentionally does not call this.
    var onUserModelSelection: (String) -> Void = { _ in }
    /// Perform the readiness banner's next-step action (start, download
    /// & start, retry). Owned by ``ContentView`` because starting a model
    /// is a window-level concern, not a chat-surface one.
    var onReadinessAction: (ModelReadiness.Action) -> Void = { _ in }
    /// Monotonic counter the parent bumps when something outside this view
    /// wants the composer to take keyboard focus — today, the onboarding
    /// completion transaction dropping the user into their first chat.
    ///
    /// A counter rather than a Bool so a second request is distinguishable
    /// from the first and nothing has to be reset afterwards. Zero is the
    /// "never asked" value; ``ComposeTextEditor`` ignores it, so a plain
    /// mount does not steal focus from whatever the user was doing.
    var composerFocusRequest: Int = 0

    /// Backing state for the composer's inline model picker (Ollama-style).
    /// The picker lives in the compose box now, not a top control bar.
    @Environment(DownloadManager.self) private var downloads
    @Environment(QuickstartCoordinator.self) private var quickstart

    @State private var draft: String = ""
    @State private var imageAttachments: [ChatImageAttachment] = []
    @State private var fileAttachments: [ChatFileAttachment] = []
    /// Source path per attachment, image or document, so the same file cannot
    /// be added twice.
    ///
    /// Kept beside the attachments rather than on the attachment types: both
    /// are `Codable` and persist with the conversation, and an absolute path
    /// written into chat history carries the user's account name and stops
    /// being true the moment the file moves.
    @State private var attachedSourcePaths: [UUID: String] = [:]
    @State private var attachmentNotice: String?
    @State private var isAttachmentDropTarget = false
    @State private var isImportingFiles = false
    @State private var composeFocusToken: Int = 0
    /// Incremented every time the user tries to send while gated. Drives
    /// the banner's brief emphasis so a blocked Return is never silent.
    @State private var blockedSendAttempts: Int = 0
    @State private var showsConversationInstructions = false
    /// Refreshed from the active conversation every time the popover opens.
    /// SwiftUI may otherwise reuse the popover's old local `@State` when the
    /// same conversation closes and reopens it.
    @State private var conversationInstructionsDraft = ""

    private let contentMaxWidth: CGFloat = RapidTheme.Layout.contentMaxWidth
    /// A live user gesture must reach the actual trailing edge before
    /// stream following resumes. The AppKit probe then owns following
    /// through every subsequent SwiftUI layout pass.
    private let bottomResumeSlack: CGFloat = 2

    /// Updated directly from the hosting NSScrollView. A SwiftUI geometry
    /// preference arrives after the scroll event, which leaves a window
    /// where the next streamed token can still yank the reader downward.
    @State private var isPinnedToBottom = true
    /// Incremented to ask the probe for an explicit scroll — see
    /// ``TranscriptScrollPositionProbe/scrollToBottomRequest``.
    @State private var scrollToBottomRequest = 0

    private var messages: [ChatMessage] { viewModel.messages }

    var body: some View {
        // The reader exists for one value: the surface's own width, which
        // the lifecycle band needs in order to pick its height on the
        // FIRST layout pass. Reading it here — where the pane already has
        // a definite size — rather than letting the band measure itself
        // is what keeps the band's geometry a pure function of its
        // inputs, and therefore reproducible in a single-pass capture.
        GeometryReader { proxy in
            VStack(spacing: 0) {
                // The band opens ABOVE the transcript, never between it
                // and the composer: it is context for the whole surface,
                // and wedging it into the compose zone is exactly the
                // treatment that made a multi-gigabyte download read as
                // an inline footnote. It replaces the readiness banner
                // for the duration — see ``composeBar`` — rather than
                // joining it, so the same sentence is never on screen
                // twice.
                if showsLifecycleBand {
                    LifecycleBand(
                        readiness: readiness,
                        attentionToken: blockedSendAttempts,
                        width: proxy.size.width
                    )
                    .transition(.move(edge: .top).combined(with: .opacity))
                }
                transcript
                Divider()
                composeBar
            }
            .frame(width: proxy.size.width, height: proxy.size.height)
        }
        .rapidAnimation(RapidMotion.standard, value: showsLifecycleBand)
        .background(RapidTheme.surfaceCanvas)
        // Drop a stale error banner once the server is provably ready.
        .onChange(of: server.state) { _, newState in
            if case .ready = newState { viewModel.clearStaleErrorBanner() }
        }
        // Forward an external focus request onto the composer's own token.
        // Routed through the existing token rather than a second mechanism
        // so every focus request in this view — Send, Cmd+L, onboarding
        // completion — reaches the editor by the same path.
        .onChange(of: composerFocusRequest) { _, request in
            guard request != 0 else { return }
            composeFocusToken &+= 1
        }
    }

    // MARK: - Transcript

    @ViewBuilder
    private var transcript: some View {
        if messages.isEmpty {
            // v1.0: the empty state is centred in the transcript region
            // instead of living inside the scroll flow behind a 96pt top
            // pad. In the scroll flow it sat high and left the bottom
            // two-thirds of the window blank, which is what made an
            // otherwise-fine screen read as an unfinished poster.
            emptyState
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ScrollView {
                transcriptRows
                    .onChange(of: viewModel.streamingBody) { _, body in
                        // The one place the raw string is fed into the
                        // debounced compiler. The row itself is still
                        // invalidated per SSE batch — `MessageRow` holds the
                        // raw `ChatMessage` — but the expensive part (markdown
                        // parse + block rebuild) now runs on the store's
                        // 100 ms beat instead of once per delta.
                        if let body {
                            streamingMarkdown.enqueue(id: body.id, text: body.text)
                        } else {
                            streamingMarkdown.finish()
                        }
                    }
                    .background(
                        TranscriptScrollPositionProbe(
                            isPinnedToBottom: $isPinnedToBottom,
                            bottomResumeSlack: bottomResumeSlack,
                            isStreaming: viewModel.isStreaming,
                            scrollToBottomRequest: scrollToBottomRequest
                        )
                    )
            }
            // The probe is the single owner of transcript positioning.
            // A new message is deliberate navigation to the conversation
            // tip; streamed frame changes then keep following from there.
            .onAppear { isPinnedToBottom = true }
            .onChange(of: messages.count) { _, _ in isPinnedToBottom = true }
            // Switching conversations is navigation to a DIFFERENT tip, but it
            // need not change `messages.count` — two conversations can have the
            // same number of turns, and then the count-keyed reset above never
            // fires. Without this, opening B inherits A's paused state and lands
            // at A's old scroll offset instead of B's latest message.
            .onChange(of: viewModel.activeConversationID) { _, _ in
                isPinnedToBottom = true
            }
            .overlay(alignment: .bottom) { jumpToBottomOverlay }
        }
    }

    /// Offered whenever the reader has scrolled away from the tip — the same
    /// basis native-chat uses ("there is somewhere below to go"), rather than
    /// gating on streaming, so "get me back" is never a state the reader has
    /// to infer.
    @ViewBuilder
    private var jumpToBottomOverlay: some View {
        if !isPinnedToBottom {
            JumpToBottomButton(isStreaming: viewModel.isStreaming) {
                // The probe owns positioning, so ask it — re-pinning alone
                // only restores following, and following has nothing to react
                // to once an answer has finished arriving.
                isPinnedToBottom = true
                scrollToBottomRequest += 1
            }
            .padding(.bottom, RapidTheme.Space.md)
            .transition(.opacity.combined(with: .scale(scale: 0.9)))
        }
    }

    /// The message rows. Factored out so the snapshot harness can render
    /// the transcript in a fixed frame (``ImageRenderer`` collapses
    /// ``ScrollView`` content to zero height).
    @ViewBuilder
    var transcriptRows: some View {
        // Tool results are rendered INSIDE the assistant row that dispatched
        // them (as an expandable chip), never as standalone transcript rows —
        // a raw JSON blob in the scroll reads as debug output.
        let toolResults = ChatView.toolResultsByCallID(messages)
        // `VStack`, not `LazyVStack`.
        //
        // Lazy row building is what made the transcript's own height a
        // moving target: a row scrolled out of view is released and reports an
        // *estimate*, so the same transcript measured 3 439 pt from the top
        // and 5 174 pt from the bottom. Every "scroll to the end" then aimed
        // at a length that changed the moment it got there, which is why
        // jumping back from the top landed mid-document or in space that had
        // not been built.
        //
        // ChatGPT keeps laziness AND correctness by pairing an
        // `NSCollectionView` with a persistent per-item height cache
        // (`ChatCollectionViewLayout.itemPlacements`), so a recycled row still
        // reports its real height. Reproducing that is a transcript rewrite;
        // building every row is the same guarantee for a fraction of the work,
        // and the cost is bounded by conversation length rather than
        // transcript length. Settled rows are not recompiled on streamed
        // deltas — `ForEach` re-instantiates only the row whose identity
        // changed, and their markdown views read no per-delta state; profiled
        // at ~0.2 % of the main thread during a stream. `MessageRow` itself
        // still re-evaluates per delta (it holds the raw `ChatMessage`), but
        // its expensive part, the markdown compile, is what this keeps cheap.
        VStack(alignment: .leading, spacing: RapidTheme.Space.lg) {
            ForEach(messages) { message in
                if message.role != .tool {
                    MessageRow(
                        message: message,
                        isStreaming: viewModel.isStreaming,
                        streamingMarkdown: streamingMarkdown,
                        toolResults: toolResults,
                        onEdit: { newContent in
                            // Edit and Retry re-enter ``send`` inside the view
                            // model, so they answer to the same readiness gate
                            // the composer does. Without this, a message action
                            // could kick off a turn the Send button is refusing
                            // to allow one row below.
                            guard acknowledgeIfNotReady() else { return false }
                            return viewModel.editUserMessage(
                                id: message.id,
                                newContent: newContent,
                                alias: alias
                            )
                        },
                        onRetry: {
                            guard acknowledgeIfNotReady() else { return false }
                            return viewModel.retryAssistantMessage(
                                id: message.id,
                                alias: alias
                            )
                        },
                        // Retry re-enters ``send``, so it answers to the same
                        // lifecycle gate — and must LOOK like it does. It used
                        // to render at full weight in every state and then do
                        // nothing at all when the model wasn't running: the
                        // gate fired, the banner flashed 400pt away at the
                        // bottom of the window, and the button the user
                        // actually pressed gave no feedback whatsoever. It now
                        // dims and carries the Send tooltip's sentence, which
                        // is the pattern ``sendOrStopButton`` already uses.
                        retryEnabled: readiness.sendAllowed,
                        retryTooltip: readiness.sendTooltip
                    )
                    .frame(maxWidth: contentMaxWidth, alignment: .leading)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .id(message.id)
                }
            }
            Color.clear
                .frame(height: 1)
        }
        .padding(.horizontal, RapidTheme.Space.xl)
        .padding(.vertical, RapidTheme.Space.xl)
    }

    /// Index every ``role: .tool`` row by the ``toolCallID`` it answers, so an
    /// assistant row can pair each of its ``toolCalls`` with its result.
    static func toolResultsByCallID(_ messages: [ChatMessage]) -> [String: ChatMessage] {
        var out: [String: ChatMessage] = [:]
        for m in messages where m.role == .tool {
            if let id = m.toolCallID { out[id] = m }
        }
        return out
    }

    /// Whether the lifecycle band owns the current moment.
    ///
    /// Exactly the two states in which the app is doing work the user is
    /// waiting on. Everything else — nothing chosen, not downloaded, not
    /// running, failed — is a DECISION, and a decision belongs beside the
    /// control that resolves it, which is the composer's notice slot.
    ///
    /// Streaming is deliberately absent. A reply arriving is work, but it
    /// is work the user can already see landing token by token, and it
    /// does not block the surface; opening a graphite band over every
    /// answer would make the ordinary case of using the product feel like
    /// an interruption.
    private var showsLifecycleBand: Bool { readiness.isWorking }

    private var emptyState: some View {
        // Optically centred, not geometrically. The composer is a fixed
        // object pinned to the bottom of the window, so a block centred
        // on the transcript's true midpoint reads as sitting low —
        // drifting toward the composer rather than balancing against it.
        // Lifting the block by 2.8% of the region's height puts it where
        // the eye expects the centre to be. The lift is proportional
        // rather than a fixed offset so it stays correct from the 560pt
        // window floor to a full-screen 1440.
        GeometryReader { proxy in
            VStack(spacing: 0) {
                Spacer(minLength: RapidTheme.Space.lg)
                heroBlock
                Spacer(minLength: RapidTheme.Space.xl)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding(.bottom, proxy.size.height * 0.056)
        }
    }

    private var heroBlock: some View {
        EmptyState(
            title: "Ask anything",
            message: emptyStateSubtitle,
            hint: downloadHint,
            // No disc. The plate was framing an illustration that already
            // has its own silhouette, and — being amber-tinted — it was
            // spending a second amber moment on a surface whose whole
            // budget is one (that one is the send disc).
            //
            // 116 is deliberately ≥ 64: ``CheetahLogo`` switches to the
            // 440×390 master above that threshold, so the artwork is
            // downsampled from a large source (crisp at @2x) rather than
            // upscaled from the 56×50 crop. ``scaledToFit`` inside the
            // square frame preserves the asset's own aspect ratio, which
            // is why the mark renders ~116×103 rather than square.
            markDiameter: 116,
            marksOnBackplate: false,
            // The chat surface at rest has nothing else in it. A 20pt
            // line alone in 1440pt of canvas reads as a caption that lost
            // its picture; at 34/40 the greeting is the object it should
            // be.
            titleEmphasis: .display,
            mark: { CheetahLogo(size: 116) }
        )
    }

    /// The line under "Ask anything", and the quieter hint below it.
    ///
    /// Both come straight off ``ModelReadiness`` rather than being
    /// re-derived here. That is the whole point of the type: the hero
    /// and the composer are two renderings of one state, so they can no
    /// longer say different things about the same moment. The two
    /// strings the approved empty state already shipped ("Choose a model
    /// to start", "Chatting with <alias>") are preserved verbatim by
    /// ``ModelReadiness/emptyStateSubtitle``.
    private var emptyStateSubtitle: String { readiness.emptyStateSubtitle }

    private var downloadHint: String? { readiness.emptyStateHint }

    // MARK: - Compose bar

    private var composeBar: some View {
        VStack(spacing: RapidTheme.Space.sm) {
            // Exactly one notice slot. While the model is not ready the
            // readiness banner owns it — it already folds the failure
            // message in, so rendering the raw error underneath would be
            // the same fact twice. Once the model is ready the slot
            // reverts to turn-level errors (a 500 from a healthy server),
            // which readiness has no opinion about.
            if !readiness.isReady && !showsLifecycleBand {
                // Suppressed while the band is open. The band renders the
                // same ``ModelReadiness`` — same headline, same detail,
                // same fraction — so leaving the banner here as well
                // would print one fact twice, 400pt apart, in two
                // different visual languages. Nothing is lost: neither
                // ``downloading`` nor ``starting`` carries a renderable
                // action, so no control moves and no identifier goes
                // missing (see ``ModelReadiness.action``).
                ReadinessBanner(
                    readiness: readiness,
                    attentionToken: blockedSendAttempts,
                    onAction: onReadinessAction
                )
                .frame(maxWidth: contentMaxWidth)
                .frame(maxWidth: .infinity)
            } else if let error = viewModel.lastError,
                      ModelReadiness.turnErrorApplies(
                          failureAlias: viewModel.lastFailureAlias,
                          selectedAlias: alias
                      ) {
                // Only the CURRENT model's turn error (or an unattributed
                // one) belongs here. Without the alias gate, switching to a
                // healthy model and letting it reach ``ready`` would render
                // the previous model's error under the fresh composer until
                // the next send cleared it — a model-scoped error leaking
                // across a model switch (#1505 follow-up).
                InlineNotice(message: error, tone: .error)
                    .frame(maxWidth: contentMaxWidth)
                    .frame(maxWidth: .infinity)
            } else if let attachmentNotice {
                InlineNotice(message: attachmentNotice, tone: .error)
                    .frame(maxWidth: contentMaxWidth)
                    .frame(maxWidth: .infinity)
            }
            // One input, not a pill containing a second pill.
            //
            // v1.0 proportions: radius 22 → 12 (the single input
            // radius), padding 14/12 → 10/8, inner spacing 10 → 6, and
            // the field/controls stack now sits on ``surfaceRaised``
            // with a hairline instead of a heavy grey ``composePill``
            // fill. The old treatment made a two-line composer ~110pt
            // tall and read as a card that happened to contain a text
            // area; this reads as a text field with controls in it.
            VStack(spacing: RapidTheme.Space.sm - 2) {
                if !imageAttachments.isEmpty || !fileAttachments.isEmpty {
                    attachmentStrip
                }
                // The field stays live in every state — a user may draft
                // while the model downloads, and that draft survives the
                // transition to ready untouched. Only the send ACTION is
                // gated; the placeholder names the blocking step so the
                // reason is visible before a single character is typed.
                ComposeField(
                    text: $draft,
                    focusToken: composeFocusToken,
                    isStreaming: viewModel.isStreaming,
                    placeholder: readiness.composerPlaceholder,
                    onSubmit: send,
                    onCancel: { viewModel.stop() },
                    onPasteAttachments: pasteAttachmentsFromClipboard,
                    onRecallLastUser: {
                        messages.last(where: { $0.role == .user })?.content
                    }
                )
                composerControls
            }
            .padding(.horizontal, RapidTheme.Space.md - 2)
            .padding(.vertical, RapidTheme.Space.sm)
            .background(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.input, style: .continuous)
                    .fill(RapidTheme.surfaceRaised)
            )
            .overlay(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.input, style: .continuous)
                    .strokeBorder(
                        isAttachmentDropTarget ? RapidTheme.brandPrimary : RapidTheme.hairlineStrong,
                        lineWidth: isAttachmentDropTarget ? 2 : 1
                    )
            )
            .dropDestination(for: URL.self) { urls, _ in
                addAttachmentURLs(urls)
            } isTargeted: { targeted in
                isAttachmentDropTarget = targeted
            }
            .frame(maxWidth: contentMaxWidth)
            .frame(maxWidth: .infinity)
        }
        .padding(.horizontal, RapidTheme.Space.xl)
        .padding(.top, RapidTheme.Space.md)
        .padding(.bottom, RapidTheme.Space.lg)
    }

    /// Bottom row of the compose box: the inline model picker on the
    /// right, then the send/stop button — Ollama's `model ▾  ⬆` cluster.
    private var composerControls: some View {
        HStack(spacing: RapidTheme.Space.sm) {
            Button(action: chooseAttachments) {
                Image(systemName: "plus")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Color.primary)
                    .frame(width: 28, height: 28)
                    .background(Circle().fill(Color.primary.opacity(0.06)))
            }
            .buttonStyle(.plain)
            .disabled(viewModel.isStreaming || isImportingFiles)
            .help(supportsImageInput ? "Add files or photos" : "Add PDF, CSV, or TXT files")
            .accessibilityLabel("Add attachments")
            .accessibilityIdentifier("ChatView.AddAttachments")
            Button {
                conversationInstructionsDraft = viewModel.conversationInstructions
                showsConversationInstructions = true
            } label: {
                Image(systemName: "text.bubble")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(
                        CustomInstructionsConfig.normalized(
                            viewModel.conversationInstructions
                        ) == nil
                            ? Color.secondary
                            : RapidTheme.brandPrimary
                    )
                    .frame(width: 28, height: 28)
                    .background(Circle().fill(Color.primary.opacity(0.06)))
            }
            .buttonStyle(.plain)
            .disabled(viewModel.isStreaming)
            .help("Conversation instructions")
            .accessibilityLabel("Conversation instructions")
            .accessibilityIdentifier("ChatView.ConversationInstructions")
            .popover(isPresented: $showsConversationInstructions, arrowEdge: .bottom) {
                ConversationInstructionsPopover(
                    draft: $conversationInstructionsDraft,
                    onSave: { value in
                        viewModel.setConversationInstructions(value)
                        showsConversationInstructions = false
                    },
                    onCancel: { showsConversationInstructions = false }
                )
                .id(viewModel.activeConversationID)
            }
            Spacer(minLength: 0)
            ModelPickerBar(
                server: server,
                downloads: downloads,
                alias: $alias,
                quickstart: quickstart,
                composerStyle: true,
                catalogRefreshEnabled: catalogRefreshEnabled,
                onUserSelection: onUserModelSelection
            )
            sendOrStopButton
        }
    }

    /// Send / stop. v1.0 gives the send action the amber hierarchy:
    /// when there is something to send it is the brightest thing in the
    /// composer, and when there isn't it recedes to a neutral outline
    /// rather than a filled-but-dead grey disc. Stop stays neutral-solid
    /// — it is a correction, not the primary path.
    @ViewBuilder
    private var sendOrStopButton: some View {
        if viewModel.isStreaming {
            Button(action: { viewModel.stop() }) {
                Image(systemName: "stop.fill")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundStyle(RapidTheme.sendButtonIcon)
                    .frame(width: 28, height: 28)
                    .background(Circle().fill(RapidTheme.sendButton))
            }
            .buttonStyle(.plain)
            .help("Stop generating")
            .accessibilityLabel("Stop generating")
            .accessibilityIdentifier("ChatView.SendOrStopButton")
        } else {
            Button(action: send) {
                Image(systemName: "arrow.up")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundStyle(
                        sendEnabled ? RapidTheme.onBrandPrimary : Color.secondary
                    )
                    .frame(width: 28, height: 28)
                    .background(
                        Circle().fill(
                            sendEnabled ? RapidTheme.brandPrimary : Color.clear
                        )
                    )
                    .overlay(
                        Circle().strokeBorder(
                            sendEnabled ? .clear : RapidTheme.hairlineStrong,
                            lineWidth: 1
                        )
                    )
            }
            .buttonStyle(.plain)
            .disabled(!sendEnabled)
            .help(readiness.sendTooltip)
            .accessibilityLabel("Send message")
            .accessibilityIdentifier("ChatView.SendOrStopButton")
            // The tooltip alone is mouse-only. VoiceOver users get the
            // same sentence as the control's hint, so "why is this
            // dimmed" is answerable without a pointer.
            .accessibilityHint(readiness.sendAllowed ? "" : readiness.sendTooltip)
        }
    }

    /// Two independent gates. ``hasDraft`` is the ordinary "nothing to
    /// send" case; ``readiness.sendAllowed`` is the lifecycle gate that
    /// replaces the old behaviour where pressing Send on a cold model
    /// silently kicked off a multi-gigabyte download behind a spinner.
    private var sendEnabled: Bool {
        hasDraft && readiness.sendAllowed && !isImportingFiles
            && (imageAttachments.isEmpty || supportsImageInput)
    }

    private var hasDraft: Bool {
        !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || !imageAttachments.isEmpty
            || !fileAttachments.isEmpty
    }

    // MARK: - Actions

    private func send() {
        let text = draft
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                || !imageAttachments.isEmpty
                || !fileAttachments.isEmpty else { return }
        guard !viewModel.isStreaming else { return }
        guard !isImportingFiles else { return }
        guard imageAttachments.isEmpty || supportsImageInput else {
            rejectImageInputForCurrentModel()
            return
        }
        // Return reaches here even when the Send button is disabled —
        // AppKit routes the key through ``ComposeTextEditor`` regardless
        // of SwiftUI's button state. Previously that landed on a silent
        // no-op: the keystroke did nothing and nothing said why.
        //
        // Two guarantees here. The draft is NOT cleared (it is still
        // exactly what the user typed, ready to send the moment the
        // model is up), and the attempt is acknowledged — the banner
        // flashes and VoiceOver speaks the same sentence the Send
        // tooltip carries.
        guard acknowledgeIfNotReady() else { return }
        draft = ""
        let images = imageAttachments
        let files = fileAttachments
        imageAttachments = []
        fileAttachments = []
        attachedSourcePaths = [:]
        attachmentNotice = nil
        composeFocusToken &+= 1
        viewModel.send(
            text,
            alias: alias,
            imageAttachments: images,
            fileAttachments: files
        )
    }

    private var supportsImageInput: Bool {
        ModelBrandStyle.supportsImageInput(forAlias: alias)
    }

    private var attachmentStrip: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: RapidTheme.Space.sm) {
                ForEach(imageAttachments) { attachment in
                    ZStack(alignment: .topTrailing) {
                        if let image = NSImage(data: attachment.data) {
                            Image(nsImage: image)
                                .resizable()
                                .scaledToFill()
                                .frame(width: 64, height: 64)
                                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                        }
                        Button {
                            imageAttachments.removeAll { $0.id == attachment.id }
                            attachedSourcePaths[attachment.id] = nil
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .symbolRenderingMode(.palette)
                                .foregroundStyle(.white, .black.opacity(0.7))
                        }
                        .buttonStyle(.plain)
                        .offset(x: 5, y: -5)
                        .accessibilityLabel("Remove \(attachment.filename)")
                        .accessibilityIdentifier(
                            "ChatView.Attachment.Remove.\(attachment.filename)"
                        )
                    }
                }
                ForEach(fileAttachments) { attachment in
                    HStack(spacing: RapidTheme.Space.sm) {
                        Image(systemName: attachment.kind.systemImage)
                            .font(.system(size: 20))
                            .foregroundStyle(RapidTheme.brandPrimary)
                            .frame(width: 28)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(attachment.filename)
                                .font(.caption.weight(.medium))
                                .lineLimit(1)
                            Text(attachment.detailText)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                        Button {
                            fileAttachments.removeAll { $0.id == attachment.id }
                            attachedSourcePaths[attachment.id] = nil
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel("Remove \(attachment.filename)")
                        .accessibilityIdentifier(
                            "ChatView.Attachment.Remove.\(attachment.filename)"
                        )
                    }
                    .padding(.horizontal, RapidTheme.Space.sm)
                    .frame(width: 220, height: 54, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: RapidTheme.Radius.input, style: .continuous)
                            .fill(Color.primary.opacity(0.05))
                    )
                }
            }
            .padding(.top, 5)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func chooseAttachments() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = supportsImageInput
            ? [.pdf, .commaSeparatedText, .plainText, .png, .jpeg, .gif]
            : [.pdf, .commaSeparatedText, .plainText]
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        guard panel.runModal() == .OK else { return }
        _ = addAttachmentURLs(panel.urls)
    }

    @discardableResult
    private func addAttachmentURLs(_ urls: [URL]) -> Bool {
        guard !isImportingFiles else { return false }
        // Filter before splitting, so images and documents get the same
        // answer to "is this already here". Re-attaching is not merely
        // redundant for a document: the per-message character budget is split
        // evenly across attachments (``fittedForMessage``), so the same PDF
        // added four times sends a quarter of it four times over instead of
        // the whole thing once, and says so only with a "partial" chip.
        let (urls, duplicates) = Self.withoutAlreadyAttached(
            urls, attached: Set(attachedSourcePaths.values)
        )
        guard !urls.isEmpty else {
            attachmentNotice = duplicates == 1
                ? "That file is already attached."
                : "Those files are already attached."
            return false
        }
        var imageURLs: [URL] = []
        var fileURLs: [URL] = []
        var unsupported = false
        var accepted = false
        for url in urls {
            let type = UTType(filenameExtension: url.pathExtension)
            if ChatFileAttachment.recognizesDocument(at: url) {
                fileURLs.append(url)
            } else if type?.conforms(to: .image) == true {
                imageURLs.append(url)
            } else {
                unsupported = true
            }
        }

        if !imageURLs.isEmpty {
            if supportsImageInput {
                addImageURLs(imageURLs)
                accepted = true
            } else {
                rejectImageInputForCurrentModel()
            }
        }
        if !fileURLs.isEmpty {
            accepted = addFileURLs(fileURLs) || accepted
        }
        if unsupported {
            attachmentNotice = "Choose PDF, CSV, TXT, PNG, JPEG, or GIF files."
        }
        return accepted
    }

    private func addImageURLs(_ urls: [URL]) {
        var accepted: [ChatImageAttachment] = []
        var rejection: String?
        for url in urls {
            do {
                let attachment = try ChatImageAttachment(contentsOf: url)
                attachedSourcePaths[attachment.id] = Self.attachmentKey(for: url)
                accepted.append(attachment)
            }
            catch { rejection = error.localizedDescription }
        }
        imageAttachments.append(contentsOf: accepted)
        attachmentNotice = rejection
    }

    @discardableResult
    private func addFileURLs(_ urls: [URL]) -> Bool {
        let selection = ChatFileAttachment.importCandidates(
            urls,
            existingCount: fileAttachments.count
        )
        guard !selection.accepted.isEmpty else {
            attachmentNotice = "Attach up to \(ChatFileAttachment.maxAttachmentsPerMessage) PDF, CSV, or TXT files per message."
            return false
        }
        isImportingFiles = true
        Task { @MainActor in
            let outcome = await Task.detached(priority: .userInitiated) {
                Self.loadFileAttachments(selection.accepted)
            }.value

            let combined = fileAttachments + outcome.0.map(\.attachment)
            fileAttachments = ChatFileAttachment.fittedForMessage(combined)
            for imported in outcome.0 {
                attachedSourcePaths[imported.attachment.id] = Self.attachmentKey(
                    for: imported.sourceURL
                )
            }
            if selection.rejectedCount > 0 {
                attachmentNotice = "Attach up to \(ChatFileAttachment.maxAttachmentsPerMessage) PDF, CSV, or TXT files per message."
            } else {
                attachmentNotice = outcome.1
            }
            isImportingFiles = false
        }
        return true
    }

    /// Parse candidates without losing which source produced each attachment.
    /// Failed candidates may appear anywhere in the batch, so pairing a
    /// filtered attachments array with the original URL array by index would
    /// associate every success after a failure with the wrong path.
    nonisolated static func loadFileAttachments(
        _ urls: [URL]
    ) -> (
        accepted: [(attachment: ChatFileAttachment, sourceURL: URL)],
        rejection: String?
    ) {
        var accepted: [(attachment: ChatFileAttachment, sourceURL: URL)] = []
        var rejection: String?
        for url in urls {
            let accessed = url.startAccessingSecurityScopedResource()
            defer {
                if accessed { url.stopAccessingSecurityScopedResource() }
            }
            do {
                accepted.append((
                    attachment: try ChatFileAttachment(contentsOf: url),
                    sourceURL: url
                ))
            } catch {
                rejection = error.localizedDescription
            }
        }
        return (accepted, rejection)
    }

    private func pasteAttachmentsFromClipboard() -> Bool {
        let pasteboard = NSPasteboard.general
        let urls = (pasteboard.readObjects(
            forClasses: [NSURL.self],
            options: [.urlReadingFileURLsOnly: true]
        ) as? [URL] ?? []).filter {
            let type = UTType(filenameExtension: $0.pathExtension)
            return type?.conforms(to: .image) == true
                || ChatFileAttachment.recognizesDocument(at: $0)
        }
        let pastedImage = NSImage(pasteboard: pasteboard)
        guard !urls.isEmpty || pastedImage != nil else { return false }
        if !urls.isEmpty {
            _ = addAttachmentURLs(urls)
        } else if let image = pastedImage,
                  let tiff = image.tiffRepresentation,
                  let rep = NSBitmapImageRep(data: tiff),
                  let png = rep.representation(using: .png, properties: [:]) {
            guard supportsImageInput else {
                rejectImageInputForCurrentModel()
                return true
            }
            do {
                imageAttachments.append(try ChatImageAttachment(
                    filename: "Pasted image.png", mimeType: "image/png", data: png
                ))
                attachmentNotice = nil
            } catch { attachmentNotice = error.localizedDescription }
        }
        return true
    }

    /// Identity for "the same file". Symlinks and `..` segments are resolved
    /// so two spellings of one path do not read as two files; the same bytes
    /// living at two real paths deliberately still count as two, because
    /// deciding otherwise would mean reading every candidate before we know
    /// whether we want it.
    nonisolated static func attachmentKey(for url: URL) -> String {
        url.standardizedFileURL.resolvingSymlinksInPath().path
    }

    /// Split incoming URLs into ones not yet attached and a count of the rest.
    /// Also collapses repeats WITHIN one batch — selecting the same file twice
    /// in the open panel is the same mistake as pasting it twice.
    nonisolated static func withoutAlreadyAttached(
        _ urls: [URL], attached: Set<String>
    ) -> (fresh: [URL], duplicates: Int) {
        var seen = attached
        var fresh: [URL] = []
        for url in urls {
            let key = attachmentKey(for: url)
            if seen.insert(key).inserted { fresh.append(url) }
        }
        return (fresh, urls.count - fresh.count)
    }

    private func rejectImageInputForCurrentModel() {
        attachmentNotice = "\(alias) doesn't support image input. Choose a vision model to add photos."
        VoiceOverAnnouncer.announce(attachmentNotice ?? "This model doesn't support images.")
    }

    /// The shared lifecycle gate for every path that would start a turn:
    /// Send, and the per-message Edit / Retry actions in the transcript.
    ///
    /// All of them ultimately call ``ChatViewModel/send(_:alias:)``, so
    /// all of them need the same answer to "is there a live model?".
    /// Returns `true` when the caller may proceed; when it returns
    /// `false` the attempt has already been acknowledged — the banner
    /// flashes and VoiceOver speaks the same sentence the Send tooltip
    /// carries, so a blocked action is never silent.
    ///
    /// Nothing is mutated on the blocked path. The draft still holds
    /// exactly what the user typed, and an edit-in-progress keeps its
    /// text, ready to go through the moment the model is up.
    @discardableResult
    private func acknowledgeIfNotReady() -> Bool {
        guard readiness.sendAllowed else {
            blockedSendAttempts &+= 1
            VoiceOverAnnouncer.announce(readiness.sendTooltip)
            return false
        }
        return true
    }
}

/// One transcript row — a user prompt bubble, an assistant answer
/// (markdown + optional reasoning disclosure + stats/error captions),
/// or a neutral system note.
private struct MessageRow: View {
    let message: ChatMessage
    let isStreaming: Bool
    let streamingMarkdown: StreamingMarkdownStore
    /// Tool-result rows keyed by the ``ToolCall.id`` they answer. Used to
    /// pair each dispatched call with its outcome inside this row.
    var toolResults: [String: ChatMessage] = [:]
    var onEdit: (String) -> Bool = { _ in false }
    var onRetry: () -> Bool = { false }
    /// Whether Retry can actually start a turn right now. Mirrors the
    /// composer's Send gate so a dead button is never shown at full
    /// weight — see the call site in ``ChatView.transcriptRows``.
    var retryEnabled: Bool = true
    /// The one sentence explaining a disabled Retry. Same string the
    /// Send button uses, so both channels say the same thing.
    var retryTooltip: String = "Retry response"

    @State private var reasoningExpanded: Bool = false
    @State private var isEditing: Bool = false
    @State private var editDraft: String = ""
    @State private var copiedRecently: Bool = false
    @State private var selectTextPresented: Bool = false
    @FocusState private var editFieldFocused: Bool

    var body: some View {
        Group {
            switch message.role {
            case .user:
                userBubble
            case .assistant:
                assistantBlock
            default:
                systemNote
            }
        }
        .sheet(isPresented: $selectTextPresented) {
            SelectTextSheet(text: SelectTextSheet.selectableText(for: selectableText))
        }
    }

    // MARK: User

    private var userBubble: some View {
        VStack(alignment: .trailing, spacing: RapidTheme.Space.xs) {
            HStack {
                Spacer(minLength: 40)
                if isEditing {
                    userEditor
                } else {
                    VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
                        if !message.fileAttachments.isEmpty {
                            VStack(alignment: .leading, spacing: RapidTheme.Space.xs) {
                                ForEach(message.fileAttachments) { attachment in
                                    HStack(spacing: RapidTheme.Space.sm) {
                                        Image(systemName: attachment.kind.systemImage)
                                            .foregroundStyle(RapidTheme.userBubbleText.opacity(0.8))
                                            .frame(width: 20)
                                        VStack(alignment: .leading, spacing: 1) {
                                            Text(attachment.filename)
                                                .font(.caption.weight(.medium))
                                                .lineLimit(1)
                                            Text(attachment.detailText)
                                                .font(.caption2)
                                                .foregroundStyle(RapidTheme.userBubbleText.opacity(0.7))
                                                .lineLimit(1)
                                        }
                                    }
                                    .accessibilityElement(children: .combine)
                                    .accessibilityLabel(
                                        "\(attachment.kind.displayName) file, \(attachment.filename), \(attachment.detailText)"
                                    )
                                }
                            }
                        }
                        if !message.imageAttachments.isEmpty {
                            LazyVGrid(
                                columns: [GridItem(.adaptive(minimum: 120, maximum: 220))],
                                spacing: RapidTheme.Space.sm
                            ) {
                                ForEach(message.imageAttachments) { attachment in
                                    if let image = NSImage(data: attachment.data) {
                                        Image(nsImage: image)
                                            .resizable()
                                            .scaledToFit()
                                            .frame(maxHeight: 240)
                                            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                                            .accessibilityLabel(attachment.filename)
                                    }
                                }
                            }
                        }
                        if !message.content.isEmpty {
                            Text(message.content)
                                .textSelection(.enabled)
                                .foregroundStyle(RapidTheme.userBubbleText)
                        }
                    }
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        .background(
                            RoundedRectangle(cornerRadius: RapidTheme.Radius.bubble, style: .continuous)
                                .fill(RapidTheme.userBubble)
                        )
                }
            }
            userActions
        }
        .frame(maxWidth: .infinity, alignment: .trailing)
    }

    /// Editor for a sent user message.
    ///
    /// The focus request lives HERE, on the editor, rather than on the
    /// enclosing bubble: a `.task(id: isEditing)` on the bubble runs in the
    /// same update that flips ``isEditing``, i.e. before this ``TextEditor``
    /// is in the responder chain, and a focus request made then is silently
    /// dropped — the editor opened unfocused and everything the user typed
    /// went to the chat composer instead (the same defect the sidebar's
    /// inline rename had). Attaching `.task` to the editor means it cannot run
    /// before the editor exists, and the yield defers the write to the next
    /// scheduling point so it lands after the update that installs the backing
    /// text view. (A yield is a scheduler hop, not a guaranteed runloop turn;
    /// it is empirically sufficient here and preferable to guessing at a
    /// sleep — same wording as ``SidebarView.renameField(_:)``.)
    private var userEditor: some View {
        TextEditor(text: $editDraft)
            .font(.body)
            .foregroundStyle(RapidTheme.userBubbleText)
            .scrollContentBackground(.hidden)
            .focused($editFieldFocused)
            .accessibilityIdentifier(actionIdentifier("EditField"))
            .task {
                await Task.yield()
                guard !Task.isCancelled else { return }
                editFieldFocused = true
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .frame(minWidth: 240, idealWidth: 420, maxWidth: 560, minHeight: 72, maxHeight: 160)
            .background(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.bubble, style: .continuous)
                    .fill(RapidTheme.userBubble)
            )
            .overlay(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.bubble, style: .continuous)
                    .stroke(RapidTheme.utilityActionHover.opacity(0.7), lineWidth: 1)
            )
    }

    /// Accessibility identifier for one of this row's action buttons.
    ///
    /// Two things it deliberately is NOT: the SF Symbol name (which is what
    /// these buttons leaked before — `doc.on.doc`, `pencil`,
    /// `arrow.clockwise`, all of which change the moment the glyph does),
    /// and the button's spoken label (localizable copy). The action half is
    /// a fixed English key; the message id makes it addressable in a
    /// transcript where every row offers the same actions.
    private func actionIdentifier(_ action: String) -> String {
        "ChatView.Message.\(action).\(message.id.uuidString)"
    }

    @ViewBuilder
    private var userActions: some View {
        HStack(spacing: 2) {
            if isEditing {
                QuietIconButton(
                    symbol: "xmark",
                    label: "Cancel editing",
                    size: RapidTheme.ControlHeight.mini
                ) {
                    cancelEditing()
                }
                .accessibilityIdentifier(actionIdentifier("CancelEdit"))
                QuietIconButton(
                    symbol: "checkmark",
                    label: "Save edited message",
                    size: RapidTheme.ControlHeight.mini
                ) {
                    saveEditing()
                }
                .disabled(
                    isStreaming
                        || (
                            editDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                                && message.imageAttachments.isEmpty
                                && message.fileAttachments.isEmpty
                        )
                )
                .accessibilityIdentifier(actionIdentifier("SaveEdit"))
            } else {
                copyButton(text: message.content, label: "Copy message")
                selectTextButton(text: message.content)
                QuietIconButton(
                    symbol: "pencil",
                    label: "Edit message",
                    size: RapidTheme.ControlHeight.mini
                ) {
                    editDraft = message.content
                    isEditing = true
                }
                .disabled(isStreaming)
                .accessibilityIdentifier(actionIdentifier("Edit"))
            }
        }
    }

    private func cancelEditing() {
        editFieldFocused = false
        editDraft = message.content
        isEditing = false
    }

    private func saveEditing() {
        guard onEdit(editDraft) else { return }
        editFieldFocused = false
        isEditing = false
    }

    @ViewBuilder
    private func copyButton(text: String, label: String) -> some View {
        QuietIconButton(
            symbol: copiedRecently ? "checkmark" : "doc.on.doc",
            label: label,
            tint: copiedRecently ? RapidTheme.utilityActionSuccess : nil,
            size: RapidTheme.ControlHeight.mini
        ) {
            copySanitizedToPasteboard(text)
            copiedRecently = true
        }
        .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        .accessibilityIdentifier(actionIdentifier("Copy"))
        .task(id: copiedRecently) {
            guard copiedRecently else { return }
            try? await Task.sleep(nanoseconds: 1_200_000_000)
            guard !Task.isCancelled else { return }
            copiedRecently = false
        }
    }

    @ViewBuilder
    private func selectTextButton(text: String) -> some View {
        QuietIconButton(
            symbol: "text.cursor",
            label: "Select text",
            size: RapidTheme.ControlHeight.mini
        ) {
            selectTextPresented = true
        }
        .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        .accessibilityIdentifier(actionIdentifier("SelectText"))
    }

    // MARK: Assistant

    private var assistantBlock: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !message.reasoning.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                reasoningDisclosure
            }
            if message.toolCallArtifactSuppressed {
                // The model tried to call a tool and the parser couldn't read
                // the request. Show the quiet explainer instead of dumping the
                // raw envelope syntax at the user.
                Text(ChatMessage.toolCallArtifactSuppressedCaptionCopy)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            } else if !message.content.isEmpty {
                // #1843: both paths now render through TextKit 2.
                //
                // Streaming goes through a debounced compiler that caches per
                // message, so a flush reparses on a 100 ms beat instead of
                // rebuilding a SwiftUI subtree from the whole accumulated
                // string every time. That is the O(length²) term `8f7e0847`
                // could only bound rather than remove.
                //
                // The split is per-message (`message.status`), not the view
                // model's `isStreaming` — the latter is true for the whole
                // transcript while the last answer arrives, so keying off it
                // would swap every earlier message's renderer the moment a new
                // one starts.
                if message.status == .streaming {
                    StreamingTextKitMarkdownView(store: streamingMarkdown)
                } else {
                    TextKitMarkdownView(content: message.content)
                }
            } else if let caption = toolDispatchCaption {
                Text(caption)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            } else if showTypingIndicator {
                typingIndicator
            }
            if let calls = message.toolCalls, !calls.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(calls) { call in
                        ToolCallChip(call: call, result: toolResults[call.id])
                    }
                }
            }
            if message.toolNotCalledFlagged {
                Text(ChatMessage.toolNotCalledCaptionCopy)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .accessibilityLabel(ChatMessage.toolNotCalledCaptionAccessibilityLabel)
            }
            if message.status == .failed {
                failureCaption
            } else if let hint = softCaption {
                Text(hint)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            if message.contentTruncated {
                Text(ChatMessage.lengthTruncationBadgeCopy)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            if let stats = statsCaption {
                Text(stats)
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            if showsAssistantActions {
                assistantActions
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// "Calling web_search…" filler for the window between a tool_calls
    /// envelope landing with no preamble prose and the result coming back.
    /// Clears itself once every dispatched call has a result — the chips then
    /// tell the whole story.
    private var toolDispatchCaption: String? {
        ChatMessage.toolDispatchPlaceholderCaption(
            content: message.content,
            reasoning: message.reasoning,
            toolCalls: message.toolCalls,
            settledToolCallIDs: Set(toolResults.keys)
        )
    }

    private var assistantActions: some View {
        HStack(spacing: 2) {
            copyButton(text: assistantCopyText, label: "Copy response")
            selectTextButton(text: assistantCopyText)
            QuietIconButton(
                symbol: "arrow.clockwise",
                label: "Retry response",
                help: retryEnabled ? "Retry response" : retryTooltip,
                size: RapidTheme.ControlHeight.mini
            ) {
                _ = onRetry()
            }
            .disabled(isStreaming || !retryEnabled)
            .accessibilityHint(retryEnabled ? "" : retryTooltip)
            .accessibilityIdentifier(actionIdentifier("Retry"))
        }
    }

    /// Whether this assistant row is a finished ANSWER the user can copy
    /// or regenerate — as opposed to a mid-turn tool dispatch.
    ///
    /// A tool round-trip writes two assistant rows into the transcript:
    /// the one that requested the tool (prose empty, ``toolCalls``
    /// populated, and the chips rendered under it), then the one that
    /// reads the results and actually answers. Rendering the action row
    /// under BOTH put a stray copy/retry pair between the weather chip
    /// and the sentence it produced — pointing at nothing the user
    /// thinks of as a response, since the visible answer is the row
    /// below. Copy would have yielded the empty string, and Retry would
    /// have rewound the same user turn the second row's Retry already
    /// covers.
    ///
    /// Keyed on the tool-dispatch shape rather than "is the last row" so
    /// it holds for every round of a multi-round turn, and for a row
    /// still streaming its follow-up prose.
    private var showsAssistantActions: Bool {
        // Not until the answer is finished. Copy and Select Text would hand
        // back a half-written response, and Retry was already dead here (it
        // carries `.disabled(isStreaming …)`) — a row of controls that either
        // lie about what they will give you or visibly do nothing is worse
        // than no row at all. ChatGPT gates the same way: its
        // `MessageRowInlineActionsPart` carries both a `streaming` flag and a
        // `streamingAppearanceDelay`.
        //
        // Keyed on THIS message's `status`, not the view model's
        // `isStreaming`: the latter is true for the whole transcript while the
        // last answer arrives, so it would also strip the actions from every
        // earlier, settled message.
        guard message.status != .streaming else { return false }
        guard let calls = message.toolCalls, !calls.isEmpty else { return true }
        return !message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var assistantCopyText: String {
        message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? message.reasoning
            : message.content
    }


    private var selectableText: String {
        message.role == .assistant ? assistantCopyText : message.content
    }

    private var reasoningDisclosure: some View {
        DisclosureGroup(isExpanded: $reasoningExpanded) {
            Text(message.reasoning)
                .font(.callout)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            HStack(spacing: RapidTheme.Space.xs) {
                Label(message.reasoningTruncated ? "Thinking trace (cut off)" : reasoningTitle,
                      systemImage: "brain")
                if reasoningInProgress {
                    ProgressView()
                        .controlSize(.mini)
                        .accessibilityHidden(true)
                }
            }
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)
                .accessibilityElement(children: .combine)
                .accessibilityLabel(reasoningAccessibilityLabel)
        }
        .accessibilityIdentifier(actionIdentifier("ReasoningDisclosure"))
        .onAppear {
            // Auto-expand a truncated reasoning-only turn so the user
            // sees the partial trace instead of an empty bubble.
            if message.reasoningTruncated { reasoningExpanded = true }
        }
    }

    /// A reasoning trace can begin well before answer tokens arrive. Keeping
    /// its disclosure label static during that interval made a healthy stream
    /// look frozen. Key this to the ROW's status (not the view model's global
    /// streaming flag) so completed history never keeps animating while a
    /// later turn is running.
    private var reasoningInProgress: Bool {
        message.status == .streaming && !message.reasoningTruncated
    }

    private var reasoningTitle: String {
        reasoningInProgress ? "Reasoning…" : "Reasoning"
    }

    private var reasoningAccessibilityLabel: String {
        if message.reasoningTruncated { return "Thinking trace, cut off" }
        return reasoningInProgress ? "Reasoning in progress" : "Reasoning"
    }

    private var showTypingIndicator: Bool {
        message.status == .streaming
            && message.reasoning.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var typingIndicator: some View {
        ProgressView()
            .controlSize(.small)
            .padding(.vertical, 2)
    }

    private var failureCaption: some View {
        Text(message.errorMessage ?? "The model couldn't complete that request.")
            .font(.footnote)
            .foregroundStyle(RapidTheme.statusError)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// A ``.complete`` row can still carry a soft, non-error caption —
    /// the "Stopped." footer or the reasoning-only-truncated hint.
    private var softCaption: String? {
        guard message.status == .complete, let msg = message.errorMessage else { return nil }
        return msg
    }

    /// "131 tok/s · 0.7 s to first token · 2.4 s".
    ///
    /// The rate covers generation only; time-to-first-token is named
    /// separately rather than being silently blended into it, so a slow
    /// prefill reads as a slow prefill instead of a slow model.
    private var statsCaption: String? {
        guard message.status == .complete, let stats = message.stats else { return nil }
        var parts: [String] = []
        if let reported = stats.reportedTokensPerSecond {
            parts.append("\(AssistantStatsFormatter.formatTPS(reported)) tok/s")
        } else if let estimated = stats.estimatedTokensPerSecond {
            parts.append("~\(AssistantStatsFormatter.formatTPS(estimated)) tok/s")
        }
        // ``validTimeToFirstToken``, not the raw field — the same value the
        // rate arithmetic accepted. A TTFT at or past the end of the turn
        // is rejected there, and rendering it here anyway would print
        // "1.2 s to first token · 1.0 s".
        if let ttft = stats.validTimeToFirstToken {
            parts.append("\(AssistantStatsFormatter.formatElapsed(ttft)) to first token")
        }
        parts.append(AssistantStatsFormatter.formatElapsed(stats.elapsedSeconds))
        return parts.joined(separator: " · ")
    }

    // MARK: System

    private var systemNote: some View {
        Text(message.content)
            .font(.footnote)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .center)
    }
}

/// One dispatched tool call, with its arguments and (once it lands) its
/// result behind a disclosure. Expanded while the tool is still running so
/// the user sees what is happening; auto-collapses on success so a long
/// transcript stays skimmable, and stays expanded on failure because the
/// error detail is the useful part.
private struct ToolCallChip: View {
    let call: ToolCall
    let result: ChatMessage?

    /// Deep-link channel into Settings. Optional so the chip still renders in
    /// any host that hasn't injected the router (previews, snapshot harness) —
    /// the non-optional form traps at lookup time. Absence also suppresses the
    /// inline button entirely; see ``FailureDiagnosis.inlineToolCardAction``.
    @Environment(SettingsRouter.self) private var settingsRouter: SettingsRouter?
    /// ``openWindow(id: "settings")``, NOT ``@Environment(\.openSettings)``.
    /// This app has no SwiftUI ``Settings`` scene — it declares a real
    /// ``Window("Settings", id: "settings")`` so the tray item can reach it
    /// (see ``RapidApp``), and ``OpenSettingsAction`` against a missing
    /// ``Settings`` scene is a silent no-op. ⌘, and the tray's "Settings…"
    /// item both go through ``openWindow`` for the same reason.
    @Environment(\.openWindow) private var openWindow

    /// Manual override. Tracks the user's last toggle so a click always wins
    /// over the auto-collapse-on-success rule; without it, expanding a
    /// completed chip would be silently re-collapsed on the next body pass.
    @State private var userToggled: Bool = false
    @State private var manualExpanded: Bool = false

    private var expanded: Bool {
        if userToggled { return manualExpanded }
        guard let result else { return true }
        return result.status == .failed
    }

    /// Pretty-printed if the model emitted real JSON, sanitised raw string if
    /// not — smaller models occasionally drop unparseable junk in here.
    private var prettyArguments: String {
        let raw = call.function.arguments
        guard !raw.isEmpty else { return "(no arguments)" }
        guard let data = raw.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data),
              let pretty = try? JSONSerialization.data(
                withJSONObject: obj,
                options: [.prettyPrinted, .sortedKeys]
              ),
              let str = String(data: pretty, encoding: .utf8) else {
            return ChatTextSanitizer.sanitizeForDisplay(raw)
        }
        return ChatTextSanitizer.sanitizeForDisplay(str)
    }

    /// Stable diagnosis for a failed result. ``nil`` while the tool is still
    /// running or when it succeeded.
    ///
    /// A tool the USER declined lands here too — it produced nothing, so it is
    /// still ``.failed`` for every purpose that asks "did this tool deliver a
    /// result?" (notably ``ChatViewModel.turnHadSuccessfulTool``). What the
    /// diagnosis's ``FailureDiagnosis/severity`` changes is only how the chip
    /// PAINTS it, and whether it offers a button — see ``inlineAction``.
    private var failureDiagnosis: FailureDiagnosis? {
        guard let result, result.status == .failed else { return nil }
        return result.toolFailureDiagnosis(toolName: call.function.name)
    }

    /// A failed result renders its stable diagnosis, never the raw tool
    /// payload — that stays in the model's context, not on screen.
    private var resultBody: String? {
        guard let result else { return nil }
        if let failureDiagnosis { return failureDiagnosis.message }
        return ChatTextSanitizer.sanitizeForDisplay(result.content)
    }

    /// The one recovery action the chip offers inline, or nil for no button.
    /// Policy lives in ``FailureDiagnosis`` so it can be pinned by a test —
    /// this view is private and a SwiftUI body isn't reachable from the suite.
    private var inlineAction: FailureDiagnosis.Action? {
        FailureDiagnosis.inlineToolCardAction(
            for: failureDiagnosis,
            canRouteToSettings: settingsRouter != nil
        )
    }

    private func perform(_ action: FailureDiagnosis.Action) {
        switch action {
        case .openWebSearchSettings:
            // Order matters: the router field must be set BEFORE the window
            // opens. ``SettingsView`` consumes it from ``.onAppear`` (first
            // open of the session) and ``.onChange`` (already-open window
            // being re-focused); setting it after the open would race the
            // ``.onAppear`` read and land the user on the last-used tab.
            // ``route`` owns that ordering and the tab choice; the same call
            // ``QuickstartView`` uses for its deep-links.
            settingsRouter?.route(action) { openWindow(id: "settings") }
        case .retry, .restart, .openModelManagement, .switchDownloadSource:
            break
        }
    }

    private var statusIcon: String {
        guard result != nil else { return "ellipsis.circle" }
        guard let failureDiagnosis else { return "checkmark.circle.fill" }
        if failureDiagnosis.severity == .notice { return "hand.raised" }
        if failureDiagnosis.kind == .browsePageTooLarge { return "exclamationmark.triangle.fill" }
        return "exclamationmark.octagon.fill"
    }

    private var statusColor: Color {
        guard result != nil else { return .secondary }
        guard let failureDiagnosis else { return .green }
        if failureDiagnosis.severity == .notice { return .secondary }
        if failureDiagnosis.kind == .browsePageTooLarge { return .orange }
        return RapidTheme.statusError
    }

    /// Red is reserved for something that actually went wrong. A decline reads
    /// in the same quiet secondary tone the transcript uses for its other
    /// "this ended early, and that's fine" footers (e.g. "Stopped.").
    private var resultBodyColor: Color {
        guard let failureDiagnosis else { return .secondary }
        if failureDiagnosis.severity == .notice { return .secondary }
        if failureDiagnosis.kind == .browsePageTooLarge { return .orange }
        return RapidTheme.statusError
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button {
                userToggled = true
                manualExpanded = !expanded
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "wrench.and.screwdriver.fill")
                        .font(.system(size: 11))
                        .foregroundStyle(RapidTheme.brand)
                    Text(ChatTextSanitizer.sanitizeForDisplay(call.function.name))
                        .font(.system(size: 12, weight: .medium, design: .monospaced))
                    Spacer(minLength: 0)
                    Image(systemName: statusIcon)
                        .font(.system(size: 11))
                        .foregroundStyle(statusColor)
                    Image(systemName: expanded ? "chevron.up" : "chevron.down")
                        .font(.system(size: 10))
                        .foregroundStyle(.tertiary)
                }
                .padding(.vertical, 6)
                .padding(.horizontal, 10)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Tool call \(call.function.name)")
            .accessibilityIdentifier("ToolCallChip.Toggle.\(call.id)")

            if expanded {
                VStack(alignment: .leading, spacing: 6) {
                    Divider()
                    Text(prettyArguments)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    if let body = resultBody {
                        Divider()
                        Text(body)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(resultBodyColor)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    if let action = inlineAction {
                        Button {
                            perform(action)
                        } label: {
                            Label(action.title, systemImage: action.systemImage)
                        }
                        .buttonStyle(.link)
                        .font(.system(size: 11))
                        .accessibilityIdentifier("ToolCallChip.\(action.rawValue)")
                    }
                }
                .padding(.horizontal, 10)
                .padding(.bottom, 8)
            }
        }
        .background(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.input, style: .continuous)
                .fill(RapidTheme.surfaceRaised)
        )
        .overlay(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.input, style: .continuous)
                .strokeBorder(RapidTheme.hairline, lineWidth: 1)
        )
    }
}

/// The chat/image compose text editor: an autosizing NSTextView with a
/// placeholder, ⏎-to-submit and Esc-to-cancel. Shared by ChatView and
/// ImagesView so both tabs get an identical input field.
struct ComposeField: View {
    @Binding var text: String
    /// Counter the parent bumps when it wants the editor to grab
    /// keyboard focus. See ``ChatView.composeFocusToken``.
    var focusToken: Int
    /// True while the chat view's ViewModel is mid-stream. Forwarded
    /// to ``ComposeTextEditor`` so its ``cancelOperation:`` handler
    /// (Esc) can stop the stream instead of being swallowed.
    var isStreaming: Bool
    /// Greyed text shown when the editor is empty. Defaults to
    /// "Send a message…" (the v0.4 copy); ChatView swaps in
    /// "Model is loading…" while the not-ready gate is active so a
    /// user who clicks into the empty editor sees the WHY before
    /// they type a single character (cycle-13 P3).
    var placeholder: String = "Send a message…"
    var onSubmit: () -> Void
    /// Called when the user presses Esc while a stream is in flight.
    /// No-op (returns control to AppKit's default Esc handling)
    /// when nothing is streaming.
    var onCancel: () -> Void
    var onPasteAttachments: () -> Bool = { false }
    /// Resolves the text of the last user message in the active
    /// session, or ``nil`` when there's nothing to recall. Bound to
    /// the Up-arrow-in-empty-compose recall affordance (Claude /
    /// Raycast convention). Default ``{ nil }`` so existing call
    /// sites that don't wire it stay quiet.
    var onRecallLastUser: () -> String? = { nil }
    /// Accessibility identity forwarded to the underlying ``NSTextView``.
    /// Defaults to the chat compose field; the Images tab overrides it so the
    /// two surfaces are distinguishable to VoiceOver and to automation.
    var axIdentifier: String = AutosizingTextView.composeAccessibilityIdentifier
    var axLabel: String = AutosizingTextView.composeAccessibilityLabel
    var axRoleDescription: String = AutosizingTextView.composeAccessibilityRoleDescription

    /// One text line + the editor's vertical inset. Floor for the
    /// field so a single line never collapses below a tappable row.
    private let minHeight: CGFloat = 22
    /// Growth ceiling. Past this the editor scrolls internally instead
    /// of pushing the whole window around.
    private let maxHeight: CGFloat = 120

    /// Measured content height reported by the NSTextView, clamped to
    /// ``[minHeight, maxHeight]``. This is THE height of the field —
    /// no reliance on intrinsic sizing (which previously let the
    /// NSTextView balloon to a giant centred textarea).
    @State private var contentHeight: CGFloat = 22

    /// True while an input method is showing pre-edit text. ``text`` is
    /// empty for that whole phase, so this is what keeps the placeholder
    /// from rendering underneath a half-typed pinyin / kana run.
    @State private var isComposing = false

    var body: some View {
        // v0.5 (Phase 5b): explicit height. The editor measures its own
        // text and reports it; we clamp and apply it via `.frame(height:)`.
        // Top-aligned (NSTextView default), so the caret/placeholder sit
        // at the top-left and the field hugs one line by default.
        ZStack(alignment: .topLeading) {
            // ``ComposeTextEditor`` reports the NSTextView's measured
            // content height on every edit; we clamp it into
            // ``[minHeight, maxHeight]`` and apply it below. Beyond the
            // cap the editor scrolls internally (it is hosted in an
            // NSScrollView) rather than pushing the window around.
            ComposeTextEditor(
                text: $text,
                focusToken: focusToken,
                isStreaming: isStreaming,
                onSubmit: onSubmit,
                onCancel: onCancel,
                onPasteAttachments: onPasteAttachments,
                onRecallLastUser: onRecallLastUser,
                axIdentifier: axIdentifier,
                axLabel: axLabel,
                axRoleDescription: axRoleDescription,
                onComposingChange: { isComposing = $0 },
                onMeasuredHeight: { measured in
                    let clamped = min(max(measured, minHeight), maxHeight)
                    if abs(clamped - contentHeight) > 0.5 {
                        contentHeight = clamped
                    }
                }
            )
            if text.isEmpty, !isComposing {
                Text(placeholder)
                    // Overlays the 15pt NSTextView — must match its
                    // size or the placeholder visibly shrinks the
                    // moment the first character lands.
                    .scaledSystemFont(15)
                    .foregroundStyle(.tertiary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .allowsHitTesting(false)
            }
        }
        .frame(height: contentHeight)
    }
}

/// ``NSTextView`` subclass that measures its own laid-out content and
/// reports the height back to SwiftUI. The stock ``NSTextView`` returns
/// ``noIntrinsicMetric`` for both axes, so a SwiftUI parent has no way
/// to size the field to its text; ``ComposeField`` therefore drives the
/// height explicitly from ``onMeasuredHeight``. ``didChangeText`` and
/// width changes both re-measure, so wrapping caused by resizing the
/// window grows the field just like typing does.
final class AutosizingTextView: NSTextView {
    /// Called with the measured content height whenever the text or the
    /// view's width changes. The receiver owns the clamping; this only
    /// reports what the layout manager actually used.
    var onMeasuredHeight: ((CGFloat) -> Void)?

    /// Called when input-method pre-edit text (marked text) appears or
    /// clears.
    ///
    /// AppKit does NOT post a text-did-change notification while an IME
    /// is composing, so ``text`` stays empty through the whole pinyin /
    /// kana / jamo phase. ``ComposeField`` keyed its placeholder off
    /// that binding alone, so "Send a message…" kept rendering
    /// underneath the candidate text the user was typing. Marked text is
    /// the only signal that the field is non-empty here.
    var onComposingChange: ((Bool) -> Void)?
    /// Gives the chat composer first refusal on file/image pasteboard items.
    /// AppKit routes Command-V through ``paste(_:)`` directly; it does not
    /// reliably consult the text-view delegate's ``doCommandBy`` hook.
    var onPasteAttachments: (() -> Bool)?
    private var lastReportedCompositionState = false

    private func reportCompositionState() {
        let isComposing = hasMarkedText()
        guard isComposing != lastReportedCompositionState else { return }
        lastReportedCompositionState = isComposing
        onComposingChange?(isComposing)
    }

    override func setMarkedText(
        _ string: Any,
        selectedRange: NSRange,
        replacementRange: NSRange
    ) {
        super.setMarkedText(
            string,
            selectedRange: selectedRange,
            replacementRange: replacementRange
        )
        reportCompositionState()
        // Pre-edit text occupies real lines — a long pinyin run wraps
        // and must grow the field like committed text does.
        remeasure()
    }

    override func unmarkText() {
        super.unmarkText()
        reportCompositionState()
        remeasure()
    }

    override func paste(_ sender: Any?) {
        if onPasteAttachments?() == true { return }
        super.paste(sender)
    }

    /// Height of the laid-out text plus the editor's vertical insets.
    var measuredHeight: CGFloat {
        guard let lm = layoutManager, let tc = textContainer else { return 28 }
        lm.ensureLayout(for: tc)
        return ceil(lm.usedRect(for: tc).height) + textContainerInset.height * 2
    }

    override var intrinsicContentSize: NSSize {
        NSSize(width: NSView.noIntrinsicMetric, height: measuredHeight)
    }

    override func didChangeText() {
        super.didChangeText()
        remeasure()
    }

    override func setFrameSize(_ newSize: NSSize) {
        let widthChanged = abs(newSize.width - frame.width) > 0.5
        super.setFrameSize(newSize)
        // Re-wrap on a width change means a different line count, so the
        // height the parent is holding is now stale.
        if widthChanged { remeasure() }
    }

    /// Re-measures and republishes the content height. Call after any
    /// programmatic edit — assigning ``string`` directly does not route
    /// through ``didChangeText``.
    func remeasure() {
        invalidateIntrinsicContentSize()
        guard let onMeasuredHeight else { return }
        let height = measuredHeight
        // Both callers can run inside a layout or view-update pass;
        // publishing SwiftUI state from there would mutate the view graph
        // mid-update. Hop to the next runloop turn so the write lands in
        // a clean cycle.
        DispatchQueue.main.async { onMeasuredHeight(height) }
    }

    /// Bug 3-A residual P2: AppleScript / cliclick / VoiceOver target
    /// NSTextView by ``accessibilityIdentifier``, but NSTextView ships
    /// without one. Setting these here (rather than inline in
    /// ``ComposeTextEditor.makeNSView``) lets an isolated unit test
    /// guard against a future refactor accidentally dropping them and
    /// breaking external tooling that depends on the IDs.
    static let composeAccessibilityLabel = "Message compose field"
    static let composeAccessibilityIdentifier = "rapid.chat.compose"
    static let composeAccessibilityRoleDescription = "Chat message input"

    /// The Images tab reuses ``ComposeField``, so before these existed its
    /// editor announced itself as the CHAT compose field: one identifier on
    /// two different surfaces. The semantic ``Images.Prompt`` identifier sits
    /// on the SwiftUI wrapper, which resolves to the placeholder static text
    /// and the scroll area — not to the NSTextView — so anything driving the
    /// prompt by identifier (VoiceOver, cliclick, the GUI golden flows) either
    /// hit the wrong element or had to pretend the Images tab was chat.
    static let imagePromptAccessibilityLabel = "Image prompt field"
    static let imagePromptAccessibilityIdentifier = "rapid.images.compose"
    static let imagePromptAccessibilityRoleDescription = "Image prompt input"

    static func applyComposeAccessibility(
        _ tv: NSTextView,
        identifier: String = composeAccessibilityIdentifier,
        label: String = composeAccessibilityLabel,
        roleDescription: String = composeAccessibilityRoleDescription
    ) {
        tv.setAccessibilityLabel(label)
        tv.setAccessibilityIdentifier(identifier)
        tv.setAccessibilityRoleDescription(roleDescription)
    }
}

/// ``NSTextView`` wrapped just enough to intercept ``insertNewline:``
/// and turn plain Return into ``onSubmit``. ``Shift+Return`` falls
/// through to a real newline insertion; ``Cmd+Return`` (which AppKit
/// routes to ``insertLineBreak:``) is also treated as submit so users
/// coming from Slack / Linear keep their muscle memory.
///
/// Smart-substitutions are explicitly off — chat with an LLM is
/// code-heavy enough that auto-quoting / dash substitution corrupts
/// snippets the user pastes.
struct ComposeTextEditor: NSViewRepresentable {
    @Binding var text: String
    var focusToken: Int
    var isStreaming: Bool
    var onSubmit: () -> Void
    var onCancel: () -> Void
    var onPasteAttachments: () -> Bool
    var onRecallLastUser: () -> String?
    /// Accessibility identity of the underlying ``NSTextView``. Defaults to
    /// the chat compose field so every existing call site — and the external
    /// tooling pinned to ``rapid.chat.compose`` — is unchanged.
    ///
    /// Declared BEFORE ``onMeasuredHeight`` on purpose: the memberwise
    /// initialiser takes arguments in declaration order, and the call site
    /// passes these ahead of the trailing height closure.
    var axIdentifier: String = AutosizingTextView.composeAccessibilityIdentifier
    var axLabel: String = AutosizingTextView.composeAccessibilityLabel
    var axRoleDescription: String = AutosizingTextView.composeAccessibilityRoleDescription
    /// Reports whether an input method is currently showing pre-edit
    /// text, so the placeholder can yield to it. Defaults to a no-op:
    /// call sites that don't draw a placeholder don't need to care.
    var onComposingChange: (Bool) -> Void = { _ in }
    /// Reports the editor's laid-out content height so ``ComposeField``
    /// can size the field to the draft.
    var onMeasuredHeight: (CGFloat) -> Void

    /// Keep this decision pure so the IME regression stays testable without
    /// constructing SwiftUI's private representable context. Marked text is
    /// owned by AppKit until the input method commits it; the binding must not
    /// overwrite that temporary editor value.
    static func shouldApplyBindingText(
        viewHasMarkedText: Bool,
        editorText: String,
        bindingText: String
    ) -> Bool {
        !viewHasMarkedText && editorText != bindingText
    }

    func makeNSView(context: Context) -> NSScrollView {
        let tv = AutosizingTextView()
        tv.delegate = context.coordinator
        tv.isRichText = false
        tv.allowsUndo = true
        tv.drawsBackground = false
        tv.backgroundColor = .clear
        tv.textContainerInset = NSSize(width: 4, height: 6)
        tv.isAutomaticQuoteSubstitutionEnabled = false
        tv.isAutomaticDashSubstitutionEnabled = false
        tv.isAutomaticTextReplacementEnabled = false
        tv.isAutomaticSpellingCorrectionEnabled = false
        tv.isAutomaticLinkDetectionEnabled = false
        tv.isAutomaticTextCompletionEnabled = false
        // 15 to match the transcript's body size (2026-07 typography
        // sweep) — was NSFont.systemFontSize (13). NSTextView ignores
        // Dynamic Type either way (documented in DynamicTypeClamp);
        // this only aligns the default-size look with the chat.
        tv.font = NSFont.systemFont(ofSize: 15)
        // Width tracks the view (so wrapping matches the visible width),
        // height is unbounded so ``usedRect`` reflects every line. This
        // is what lets us measure the true content height below.
        tv.isVerticallyResizable = true
        tv.isHorizontallyResizable = false
        tv.autoresizingMask = [.width]
        tv.textContainer?.widthTracksTextView = true
        tv.textContainer?.heightTracksTextView = false
        tv.textContainer?.containerSize = NSSize(
            width: 0, height: CGFloat.greatestFiniteMagnitude
        )
        tv.onMeasuredHeight = onMeasuredHeight
        tv.onComposingChange = onComposingChange
        tv.onPasteAttachments = onPasteAttachments
        // Bug 3-A residual P2: NSTextView already advertises role
        // ``.textArea`` by default, but with no label / identifier
        // AppleScript and cliclick can't tell which text area is the
        // chat compose vs the system-prompt editor or search bar.
        AutosizingTextView.applyComposeAccessibility(
            tv,
            identifier: axIdentifier,
            label: axLabel,
            roleDescription: axRoleDescription
        )

        // Hosting the editor in a scroll view is what makes the height
        // cap usable: past ``ComposeField.maxHeight`` the draft scrolls
        // inside the field (and the caret stays visible) instead of
        // being clipped out of reach.
        let scroll = NSScrollView()
        scroll.documentView = tv
        scroll.hasVerticalScroller = false
        scroll.hasHorizontalScroller = false
        scroll.drawsBackground = false
        scroll.borderType = .noBorder
        scroll.autohidesScrollers = true
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        guard let view = scroll.documentView as? AutosizingTextView else { return }
        // Never write over an in-flight IME composition. While marked
        // text is up, ``view.string`` holds the pre-edit run and ``text``
        // is still empty (AppKit posts no text-did-change until the user
        // commits), so this branch would look like a stale draft and
        // assigning ``text`` would erase the pinyin mid-word. Any
        // unrelated SwiftUI update landing on this view — including the
        // placeholder's own state change — is enough to trigger it.
        if Self.shouldApplyBindingText(
            viewHasMarkedText: view.hasMarkedText(),
            editorText: view.string,
            bindingText: text
        ) {
            view.string = text
            // A programmatic assignment does not fire ``didChangeText``,
            // so the parent would keep the height of the previous draft
            // (e.g. after Send clears it, or after Up-arrow recall).
            view.remeasure()
        }
        view.onMeasuredHeight = onMeasuredHeight
        view.onComposingChange = onComposingChange
        view.onPasteAttachments = onPasteAttachments
        context.coordinator.onSubmit = onSubmit
        context.coordinator.onCancel = onCancel
        context.coordinator.onPasteAttachments = onPasteAttachments
        context.coordinator.isStreaming = isStreaming
        context.coordinator.onRecallLastUser = onRecallLastUser
        // Cmd+L (or any other external focus request) bumps the
        // token; we compare-and-store so a single bump triggers
        // exactly one ``makeFirstResponder`` call.
        if focusToken != context.coordinator.lastFocusToken {
            context.coordinator.lastFocusToken = focusToken
            if focusToken != 0 {
                DispatchQueue.main.async {
                    view.window?.makeFirstResponder(view)
                }
            }
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(
            text: $text,
            onSubmit: onSubmit,
            onCancel: onCancel,
            onPasteAttachments: onPasteAttachments,
            isStreaming: isStreaming,
            onRecallLastUser: onRecallLastUser
        )
    }

    @MainActor
    final class Coordinator: NSObject, NSTextViewDelegate {
        var text: Binding<String>
        var onSubmit: () -> Void
        var onCancel: () -> Void
        var onPasteAttachments: () -> Bool
        var isStreaming: Bool
        /// Resolves text of the last user message for Up-arrow recall;
        /// nil = nothing to recall, fall through to AppKit default.
        var onRecallLastUser: () -> String?
        /// Last focus token applied. ``updateNSView`` compares and
        /// only calls ``makeFirstResponder`` when this lags behind.
        var lastFocusToken: Int = 0

        init(
            text: Binding<String>,
            onSubmit: @escaping () -> Void,
            onCancel: @escaping () -> Void,
            onPasteAttachments: @escaping () -> Bool,
            isStreaming: Bool,
            onRecallLastUser: @escaping () -> String?
        ) {
            self.text = text
            self.onSubmit = onSubmit
            self.onCancel = onCancel
            self.onPasteAttachments = onPasteAttachments
            self.isStreaming = isStreaming
            self.onRecallLastUser = onRecallLastUser
        }

        func textDidChange(_ notification: Notification) {
            guard let tv = notification.object as? NSTextView else { return }
            text.wrappedValue = tv.string
        }

        func textView(_ textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
            if commandSelector == #selector(NSText.paste(_:)), onPasteAttachments() {
                return true
            }
            if commandSelector == #selector(NSResponder.insertNewline(_:)) {
                // Shift held → real newline. Otherwise → submit. Probing
                // ``NSApp.currentEvent`` is the documented way to read
                // modifier flags from inside ``doCommandBy``; the
                // selector itself doesn't carry them.
                let event = NSApp.currentEvent
                let shiftPressed = event?.modifierFlags.contains(.shift) ?? false
                if shiftPressed {
                    textView.insertText("\n", replacementRange: textView.selectedRange())
                    return true
                }
                onSubmit()
                return true
            }
            if commandSelector == #selector(NSResponder.insertLineBreak(_:)) {
                // Cmd+Return — also submit. The SwiftUI ``.keyboardShortcut``
                // on the send button is a belt-and-suspenders fallback for
                // when the editor isn't first responder.
                onSubmit()
                return true
            }
            if commandSelector == #selector(NSResponder.cancelOperation(_:)) {
                // Esc. Two roles: during a stream it stops generation
                // (same as clicking the Stop button — the muscle
                // memory most chat surfaces have settled on). When
                // nothing is streaming we hand the event back so the
                // surrounding window's default Esc handling (close a
                // popover, dismiss a sheet) still works.
                if isStreaming {
                    onCancel()
                    return true
                }
                return false
            }
            if commandSelector == #selector(NSResponder.moveUp(_:)) {
                // ⬆ in an empty compose = recall the last user
                // message into the editor for editing / resending.
                // Claude and Raycast both ship this; the rule is
                // "only when the field is empty" so multi-line
                // editing's natural caret-up-a-line behaviour stays
                // intact. Whitespace-only counts as empty — a stray
                // space-then-Up shouldn't lock the user out of the
                // affordance.
                let trimmed = textView.string.trimmingCharacters(in: .whitespacesAndNewlines)
                guard trimmed.isEmpty,
                      let recalled = onRecallLastUser(),
                      !recalled.isEmpty
                else { return false }
                textView.string = recalled
                text.wrappedValue = recalled
                // Direct ``string`` assignment skips ``didChangeText``, so
                // the recalled draft would render at the old (one-line)
                // height until the next keystroke.
                (textView as? AutosizingTextView)?.remeasure()
                // Park the caret at the END so the user can
                // immediately append or hit ⌘A → retype. Anchoring
                // at start would force a ⌘→ before the first edit.
                let end = (recalled as NSString).length
                textView.setSelectedRange(NSRange(location: end, length: 0))
                return true
            }
            return false
        }
    }
}

/// Theme that styles MarkdownUI to feel like ChatGPT Desktop's
/// assistant transcript. Calibrated against the model's typical
/// long-form output: a paragraph or two of prose, an H3 section
/// header, a numbered list of bolded items, occasional inline code.
///
/// The big wins over Apple's ``AttributedString(markdown:)``:
///   * Headings get real font-size jumps and top/bottom margins
///     instead of rendering as plain body text.
///   * Lists use a hanging indent so wrapped lines align under the
///     first character of the item, not under the bullet/number.
///   * Code blocks get a distinct background, padding, and a
///     monospaced font.
///   * Paragraph spacing is honoured — multi-paragraph answers no
///     longer collapse into a wall of text.
extension MarkdownUI.Theme {
    // MarkdownUI.Theme isn't Sendable, and SwiftUI evaluates view
    // bodies on the main actor, so isolate the literal to MainActor
    // rather than tripping Swift 6's "main-actor-isolated default in
    // a nonisolated context" diagnostic on a plain ``static let``.
    //
    // #546: the transcript body already honours Dynamic Type — the
    // ``Markdown`` view wraps this theme's root ``FontSize`` in its own
    // ``@ScaledMetric(relativeTo: .body)`` (`Markdown.swift`
    // `ScaledFontSizeModifier`) and every other size here is `.em(...)`
    // relative to that root, so the whole answer scales off MarkdownUI's
    // single built-in pass. The root therefore stays a FIXED 13pt: a
    // second `@ScaledMetric` at the call site would double-scale it
    // (~13 × scale²). Display math is scaled separately in ``MathView``.
    @MainActor
    static let rapidChat: MarkdownUI.Theme = MarkdownUI.Theme()
        .text {
            // 15pt on a Claude-Desktop-calibrated reading rhythm.
            // History, because this number has flip-flopped: v0.3
            // shipped 15, v0.4 reverted to 13 ("2pt too large vs the
            // system baseline"), and 2026-07 dogfood reversed that
            // again — explicit user feedback that 13 was hard to read
            // across the whole app, with Claude Desktop (~15-16px
            // body) as the named reference. If this ever feels big,
            // the fallback is 14 — do NOT "fix" it back to 13, and
            // move the streaming Text + MathView base in lockstep
            // (three literals, one size; see the streaming branch).
            FontSize(15)
            // v1.0.1: system sans, not New York.
            //
            // The serif was an editorial device — "serif for the
            // model's voice, sans for the chrome" — and it read as a
            // different application embedded inside this one. A
            // desktop tool should feel like one product; the
            // separation between model content and app chrome is
            // carried by the 720pt measure, the paragraph rhythm
            // below, and weight — not by a typeface switch.
            ForegroundColor(.primary)
        }
        .code {
            FontFamilyVariant(.monospaced)
            FontSize(.em(0.92))
            BackgroundColor(.secondary.opacity(0.15))
        }
        .strong { FontWeight(.semibold) }
        // Steel blue, spelled semantically: a markdown link is exactly
        // the sanctioned use of the secondary brand colour. Same value
        // as the legacy ``brand`` alias it replaces.
        .link { ForegroundColor(RapidTheme.linkLabel) }
        // v1.0.1: restrained heading scale. With the serif gone the
        // old 1.45/1.25/1.1 ramp read as oversized — a serif carries
        // a large size gracefully, a sans at 21.75pt inside a 15pt
        // answer just looks like a heading pasted in from a document.
        // 1.27/1.13/1.0-with-weight keeps three distinguishable
        // levels while staying inside the answer's own rhythm.
        .heading1 { config in
            config.label
                .relativePadding(.top, length: .em(0.35))
                .relativePadding(.bottom, length: .em(0.1))
                .markdownTextStyle {
                    FontWeight(.semibold)
                    FontSize(.em(1.27))
                }
        }
        .heading2 { config in
            config.label
                .relativePadding(.top, length: .em(0.3))
                .relativePadding(.bottom, length: .em(0.1))
                .markdownTextStyle {
                    FontWeight(.semibold)
                    FontSize(.em(1.13))
                }
        }
        .heading3 { config in
            config.label
                .relativePadding(.top, length: .em(0.25))
                .relativePadding(.bottom, length: .em(0.1))
                .markdownTextStyle {
                    FontWeight(.semibold)
                    FontSize(.em(1.0))
                }
        }
        .paragraph { config in
            // 2026-07 recalibration, replacing the v0.5.9
            // ChatGPT-matched 0.15em/9pt: the reference is now
            // Claude Desktop's ~1.5-1.6 leading, per the same
            // dogfood feedback that raised the base size.
            // Natural leading ~1.2 + 0.35em ≈ 1.55 effective, and
            // the 12pt bottom margin restores real paragraph
            // rhythm at the bigger size.
            config.label
                .relativeLineSpacing(.em(0.35))
                .markdownMargin(top: 0, bottom: 12)
        }
        .listItem { config in
            // v0.5.9: tighten list-item gutter from 0.15 → 0.05
            // em. ChatGPT renders consecutive bullets nearly
            // touching with the leading doing the visual work;
            // 0.15 em on 13 pt base was a perceptible gap that
            // made numbered lists feel sparse.
            config.label
                .markdownMargin(top: .em(0.05), bottom: .em(0.05))
        }
        .codeBlock { config in
            CodeBlockWithCopy(config: config)
                .markdownMargin(top: 8, bottom: 8)
        }
        .blockquote { config in
            HStack(spacing: 0) {
                Rectangle()
                    .fill(Color.secondary.opacity(0.5))
                    .frame(width: 3)
                config.label
                    .padding(.leading, 10)
                    .markdownTextStyle { ForegroundColor(.secondary) }
            }
            .markdownMargin(top: 8, bottom: 8)
        }
        .table { config in
            let accessibilityTable = MarkdownTableAccessibility.parse(
                config.content.renderMarkdown()
            )
            config.label
                // ``fixedSize`` vertically: without it a cell whose text
                // wraps gets its height clipped to one line, because the
                // table lays rows out before the wrapped measurement lands.
                .fixedSize(horizontal: false, vertical: true)
                .markdownTableBorderStyle(.init(color: .secondary.opacity(0.4)))
                .markdownMargin(top: 8, bottom: 8)
                .accessibilityRepresentation {
                    if let accessibilityTable {
                        AccessibleMarkdownTable(model: accessibilityTable)
                    }
                }
        }
        // 2026-08: the theme set a table BORDER but never a cell style, so
        // MarkdownUI fell back to ``Theme.tableCell``'s default — the bare
        // label, zero padding. Rendered output put the border hard against
        // the glyphs and ran adjacent cells together ("列1列2" with no gap),
        // which reads as a layout bug rather than a table. Padding is the
        // whole fix; the header weight and fill are what make the first row
        // scan as a header once the columns are actually separated.
        //
        // Values follow the surrounding rhythm rather than MarkdownUI's
        // GitHub theme: 13pt horizontal there is calibrated for a 16px web
        // body, and at our 15pt root inside a 720pt message column it
        // pushes three-column tables into horizontal overflow. 10/5 keeps
        // the columns distinct without spending the measure.
        .tableCell { config in
            config.label
                .markdownTextStyle {
                    if config.row == 0 { FontWeight(.semibold) }
                }
                // The default block text style cannot override an inline
                // code run: ``Theme.code`` is applied afterwards. Replace
                // that specific style inside table cells so inline code
                // keeps its face and size without painting a grey pill that
                // fights the row fill below.
                .markdownTextStyle(\.code) {
                    FontFamilyVariant(.monospaced)
                    FontSize(.em(0.92))
                    BackgroundColor(nil)
                }
                .fixedSize(horizontal: false, vertical: true)
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .relativeLineSpacing(.em(0.2))
                // ``frame`` BEFORE ``background``, and it is not
                // optional. A cell's label is only as wide as its text,
                // while the column is as wide as its widest cell, so
                // filling the label paints a header stripe that stops
                // partway across every column it doesn't own — which
                // reads as phantom empty columns in the header row
                // rather than as a fill. Stretching first makes the
                // fill cover the cell it is supposed to describe.
                .frame(maxWidth: .infinity, alignment: .leading)
                // Header fill only. Alternating row stripes were tried and
                // dropped: against the transcript's warm parchment they
                // read as selection highlights on the model's own answer.
                .background(
                    config.row == 0
                        ? Color.secondary.opacity(0.08)
                        : Color.clear
                )
        }
}
/// Carries the code block's scrollable content span up to
/// ``CodeBlockWithCopy``. Measured inside the scroll view, so the
/// value tracks the scroll offset as well as the content width.
private struct CodeBlockContentSpanKey: PreferenceKey {
    static let defaultValue = CodeBlockOverflow.ContentSpan.unmeasured

    static func reduce(
        value: inout CodeBlockOverflow.ContentSpan,
        nextValue: () -> CodeBlockOverflow.ContentSpan
    ) {
        let next = nextValue()
        if next.isMeasured { value = next }
    }
}

/// Code block with a hover-revealed Copy button. ChatGPT Desktop
/// hangs the copy affordance off the top-right; we mirror that and
/// fade in on hover so the button doesn't distract during reading.
/// The button briefly flips to a checkmark on click so the user
/// knows the clipboard load actually happened.
///
/// Wide code scrolls horizontally rather than wrapping — a wrapped
/// line mangles the indentation that Python and YAML carry meaning
/// in. That scroll used to be undiscoverable: `showsIndicators:
/// false` installs no scroller at all, so a line running past the
/// 720pt message column read as truncated and users assumed the tail
/// of the answer was gone (2026-08 dogfood). The indicator is on and
/// the edges dissolve wherever content is hidden — see
/// ``CodeBlockOverflow`` for the geometry and the measurement that
/// established the text was never actually clipped.
private struct CodeBlockWithCopy: View {
    let config: CodeBlockConfiguration

    @State private var hovering: Bool = false
    @State private var copiedRecently: Bool = false
    @State private var contentSpan = CodeBlockOverflow.ContentSpan.unmeasured
    @State private var syntaxHighlightMemo = SyntaxHighlighter.Memo()
    @ScaledMetric(relativeTo: .body)
    private var highlightedCodeLineSpacing: CGFloat = 15 * 0.92 * 0.2

    /// Highlighted when we have a grammar for the fence's language,
    /// otherwise MarkdownUI's own label.
    ///
    /// The two branches are NOT interchangeable. ``config.label`` carries
    /// MarkdownUI's inline styling and reacts to ``markdownTextStyle``;
    /// the highlighted branch is a plain ``Text`` over an
    /// ``AttributedString``, so it has to set the monospaced face and the
    /// 0.92em size itself. Both end up at the same measurements — keep
    /// them in lockstep if either moves, or a highlighted block will
    /// render a different size from an unhighlighted one in the same
    /// reply.
    ///
    /// ``.textSelection`` is not applied here: the assistant block
    /// already enables it for the whole message, and re-applying it to
    /// the inner ``Text`` makes the horizontal scroll gesture fight
    /// text-drag selection.
    @ViewBuilder
    private var codeBody: some View {
        if SyntaxHighlighter.supports(language: config.language) {
            Text(syntaxHighlightMemo.highlight(config.content, language: config.language))
                .scaledSystemFont(15 * 0.92, design: .monospaced)
                .lineSpacing(highlightedCodeLineSpacing)
                .fixedSize(horizontal: true, vertical: false)
                .frame(maxWidth: .infinity, alignment: .leading)
        } else {
            config.label
                .relativeLineSpacing(.em(0.2))
                .markdownTextStyle {
                    FontFamilyVariant(.monospaced)
                    FontSize(.em(0.92))
                }
        }
    }

    var body: some View {
        ZStack(alignment: .topTrailing) {
            ScrollView(.horizontal, showsIndicators: true) {
                codeBody
                    .padding(10)
                    .background(contentSpanReader)
            }
            // Order matters twice here. ``mask`` before ``background``
            // fades only the code, leaving the block's fill solid —
            // the alternative, painting a gradient of the fill colour
            // over the text, has to guess the composited colour and
            // gets it wrong in one appearance or the other. And the
            // mask sits OUTSIDE the scroll view so it stays anchored
            // to the viewport instead of scrolling away with the
            // text. Rendering-only: the AppKit scroll view underneath
            // is untouched, so the gesture still works inside the
            // faded strip.
            .mask(fadeMask)
            .background(Color.secondary.opacity(0.12))
            .clipShape(RoundedRectangle(cornerRadius: 6))
            // ``$contentSpan`` rather than ``self``: the action is
            // ``@Sendable`` and ``CodeBlockConfiguration`` is not.
            .onPreferenceChange(CodeBlockContentSpanKey.self) { [$contentSpan] span in
                $contentSpan.wrappedValue = span
            }

            copyButton
                .padding(.top, 6)
                .padding(.trailing, 6)
                .opacity(hovering || copiedRecently ? 1.0 : 0.0)
                .rapidAnimation(RapidMotion.quick, value: hovering)
        }
        .onHover { h in hovering = h }
    }

    /// Publishes the content's horizontal span. Lives in a
    /// ``background`` so it inherits the content's frame exactly and
    /// contributes nothing to layout.
    private var contentSpanReader: some View {
        GeometryReader { proxy in
            let frame = proxy.frame(in: .global)
            Color.clear.preference(
                key: CodeBlockContentSpanKey.self,
                value: CodeBlockOverflow.ContentSpan(minX: frame.minX, maxX: frame.maxX)
            )
        }
    }

    /// Opaque through the middle, transparent at whichever edge is
    /// hiding content. Falls back to a flat opaque fill before the
    /// first measurement lands and whenever the block fits, so a short
    /// code block keeps crisp edges.
    private var fadeMask: some View {
        GeometryReader { proxy in
            let frame = proxy.frame(in: .global)
            let fade = CodeBlockOverflow.fade(
                content: contentSpan,
                viewportMinX: frame.minX,
                viewportMaxX: frame.maxX
            )
            if fade.isEmpty || frame.width <= 0 {
                Color.black
            } else {
                LinearGradient(
                    stops: [
                        .init(color: .black.opacity(fade.leading > 0 ? 0 : 1), location: 0),
                        .init(color: .black, location: fade.leading / frame.width),
                        .init(color: .black, location: 1 - fade.trailing / frame.width),
                        .init(color: .black.opacity(fade.trailing > 0 ? 0 : 1), location: 1)
                    ],
                    startPoint: .leading,
                    endPoint: .trailing
                )
            }
        }
    }

    private var copyButton: some View {
        Button {
            copySanitizedToPasteboard(config.content)
            copiedRecently = true
        } label: {
            HStack(spacing: 4) {
                Image(systemName: copiedRecently ? "checkmark" : "doc.on.doc")
                    .font(.system(size: 11, weight: .medium))
                Text(copiedRecently ? "Copied" : "Copy")
                    .font(.caption2.weight(.medium))
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(
                RoundedRectangle(cornerRadius: 5, style: .continuous)
                    .fill(Color(nsColor: .controlBackgroundColor).opacity(0.9))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 5, style: .continuous)
                    .stroke(Color.secondary.opacity(0.25), lineWidth: 0.5)
            )
            .foregroundStyle(copiedRecently ? Color.green : .secondary)
        }
        .buttonStyle(.plain)
        .help("Copy code")
        .accessibilityIdentifier(
            "ChatView.CodeBlock.Copy.\(config.content.utf8.count)"
        )
        .task(id: copiedRecently) {
            guard copiedRecently else { return }
            // 1.2 s feels like ChatGPT Desktop's flash; long enough
            // to register, short enough not to linger.
            try? await Task.sleep(nanoseconds: 1_200_000_000)
            guard !Task.isCancelled else { return }
            copiedRecently = false
        }
    }
}
/// Free-standing formatters for the v0.4.12 assistant stats
/// caption. Lifted out of ``MessageRow`` (which is ``private``) so
/// the format-and-display contract is testable without standing up
/// a SwiftUI host. Pure value transforms — no dependency on
/// SwiftUI types, no @MainActor needed.
enum AssistantStatsFormatter {
    /// Format the TPS number — sub-10 gets one decimal so a slow
    /// 4-bit 27B doesn't render as "9 tok/s"; ≥10 rounds to int
    /// because nobody cares about the tenths at 80 tok/s.
    static func formatTPS(_ tps: Double) -> String {
        if tps < 10 { return String(format: "%.1f", tps) }
        return "\(Int(tps.rounded()))"
    }

    /// Elapsed formatter — milliseconds for sub-second turns so the
    /// hot-cache case is legible; "X.Xs" for the common second
    /// range; "Xm Ys" past 60 s so a tool-call round that ran
    /// search → summarise → answer doesn't read as "94.7s".
    static func formatElapsed(_ seconds: Double) -> String {
        if seconds < 1.0 {
            return "\(Int((seconds * 1000).rounded())) ms"
        }
        if seconds < 60.0 {
            return String(format: "%.1f s", seconds)
        }
        let mins = Int(seconds) / 60
        let secs = Int(seconds) % 60
        return "\(mins)m \(secs)s"
    }

    /// VoiceOver-friendly composite for the caption. Screen readers
    /// would otherwise stumble over the tilde + middle-dot
    /// separator; this resolves to a plain English sentence.
    static func accessibilityCaption(for stats: MessageStats) -> String {
        var parts: [String] = []
        if let tps = stats.reportedTokensPerSecond {
            parts.append("\(formatTPS(tps)) tokens per second")
        } else if let est = stats.estimatedTokensPerSecond {
            parts.append("approximately \(formatTPS(est)) tokens per second")
        }
        if let ttft = stats.validTimeToFirstToken {
            parts.append("\(formatElapsed(ttft)) to the first token")
        }
        if stats.elapsedSeconds > 0 {
            parts.append("took \(formatElapsed(stats.elapsedSeconds))")
        }
        return parts.joined(separator: ", ")
    }
}
