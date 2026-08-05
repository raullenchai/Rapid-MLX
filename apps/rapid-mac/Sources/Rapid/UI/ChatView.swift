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
    @Bindable var server: ServerManager
    @Binding var alias: String
    var serverReady: Bool
    /// When the next Send would trigger a cold model download, the
    /// alias + a human size string so the empty state can hint at it.
    var autoStartPendingDownload: (alias: String, sizeText: String?)? = nil

    /// Backing state for the composer's inline model picker (Ollama-style).
    /// The picker lives in the compose box now, not a top control bar.
    @Environment(DownloadManager.self) private var downloads
    @Environment(QuickstartCoordinator.self) private var quickstart

    @State private var draft: String = ""
    @State private var composeFocusToken: Int = 0
    @State private var showConnectTools = false
    @State private var showBenchmark = false

    private let contentMaxWidth: CGFloat = RapidTheme.Layout.contentMaxWidth
    private let bottomSentinelID = "rapid-bottom-sentinel"

    private var messages: [ChatMessage] { viewModel.messages }

    var body: some View {
        VStack(spacing: 0) {
            transcript
            Divider()
            composeBar
        }
        .background(RapidTheme.surfaceCanvas)
        // Drop a stale error banner once the server is provably ready.
        .onChange(of: server.state) { _, newState in
            if case .ready = newState { viewModel.clearStaleErrorBanner() }
        }
        .sheet(isPresented: $showConnectTools) {
            ConnectToolsView(
                host: "127.0.0.1",
                port: server.activePort,
                bearer: server.activeBearer ?? "",
                alias: alias,
                onClose: { showConnectTools = false }
            )
        }
        .sheet(isPresented: $showBenchmark) {
            BenchmarkView(
                binary: server.binaryPath,
                alias: alias,
                hardware: MacHardware.detect(),
                onClose: { showBenchmark = false }
            )
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
            ScrollViewReader { proxy in
                ScrollView {
                    transcriptRows
                }
                // This branch mounts fresh the moment the transcript goes
                // from empty to populated — selecting a long saved
                // conversation, or launching straight into one. The
                // `messages.count` change that mounts it predates the
                // `.onChange` handlers below, so a fresh ScrollView would
                // open at the OLDEST message. Anchor to the latest on
                // appear (no animation: this is initial positioning, not a
                // scroll the user should see move).
                .onAppear { scrollToBottom(proxy, animated: false) }
                .onChange(of: messages.last?.content) { _, _ in scrollToBottom(proxy) }
                .onChange(of: messages.last?.reasoning) { _, _ in scrollToBottom(proxy) }
                .onChange(of: messages.count) { _, _ in scrollToBottom(proxy) }
            }
        }
    }

    /// The message rows. Factored out so the snapshot harness can render
    /// the transcript in a fixed frame (``ImageRenderer`` collapses
    /// ``ScrollView`` content to zero height).
    @ViewBuilder
    var transcriptRows: some View {
        LazyVStack(alignment: .leading, spacing: RapidTheme.Space.lg) {
            ForEach(messages) { message in
                MessageRow(
                    message: message,
                    isStreaming: viewModel.isStreaming,
                    onRegenerate: regenerate
                )
                .frame(maxWidth: contentMaxWidth, alignment: .leading)
                .frame(maxWidth: .infinity, alignment: .center)
                .id(message.id)
            }
            Color.clear
                .frame(height: 1)
                .id(bottomSentinelID)
        }
        .padding(.horizontal, RapidTheme.Space.xl)
        .padding(.vertical, RapidTheme.Space.xl)
    }

    private func scrollToBottom(_ proxy: ScrollViewProxy, animated: Bool = true) {
        guard animated else {
            proxy.scrollTo(bottomSentinelID, anchor: .bottom)
            return
        }
        withAnimation(.easeOut(duration: 0.15)) {
            proxy.scrollTo(bottomSentinelID, anchor: .bottom)
        }
    }

    private var emptyState: some View {
        EmptyState(
            title: "Ask anything",
            message: emptyStateSubtitle,
            hint: downloadHint,
            markDiameter: 92,
            mark: {
                // The brand moment on the app's main surface. 68pt
                // inside a 92pt disc — at the previous 28/44 the mascot
                // read as a favicon rather than the product's mark.
                //
                // 68 is deliberately ≥ 64: ``CheetahLogo`` switches to
                // the 440×390 master above that threshold, so the
                // artwork is downsampled from a large source (crisp at
                // @2x) instead of being upscaled from the 56×50 crop.
                // ``scaledToFit`` inside a square frame preserves the
                // asset's own aspect ratio.
                CheetahLogo(size: 68)
            },
            actions: {
                // #CTA-bug: both actions used to be wrapped in
                // `if serverReady`, so on a cold first launch — exactly
                // when a new user most needs the second call-to-action —
                // the row was absent entirely, and it then popped into
                // existence mid-session and shifted the layout.
                //
                // Now both always render whenever the transcript is
                // empty, in every lifecycle state. Availability is
                // expressed by ENABLEMENT, not by presence:
                //
                //   * Connect your tools is always actionable. The sheet
                //     itself explains when the endpoint isn't ready yet
                //     and refuses to hand out incomplete values.
                //   * Speed needs a live model to measure, so it
                //     disables with a tooltip that says why.
                Button {
                    showConnectTools = true
                } label: {
                    Label("Connect your tools", systemImage: "link")
                }
                .help("Point an editor or agent at your local server")

                Button {
                    showBenchmark = true
                } label: {
                    Label("Speed on this Mac", systemImage: "gauge.with.dots.needle.67percent")
                }
                // Enablement is derived from live server state, so the
                // button flips to enabled on its own the moment the
                // model reaches .ready — no user action, no re-render
                // trigger needed beyond the @Observable read.
                .disabled(!benchmarkAvailable)
                .help(
                    benchmarkAvailable
                        ? "Measure this model's tokens per second on your Mac"
                        : "Start a model to run a speed test."
                )
            }
        )
    }

    /// The line under "Ask anything".
    ///
    /// Three distinct states, none of which may render an internal
    /// placeholder (`Loading`, `Starting`, …) where a model name goes:
    ///
    ///   * nothing chosen yet  → an instruction, not a claim
    ///   * coming up           → a status sentence
    ///   * resolved            → the real alias
    private var emptyStateSubtitle: String {
        if case .starting = server.state {
            return "Preparing your local model…"
        }
        if ModelDisplayName.isUnresolved(alias) {
            // "Choose", not "Select" — one verb for this flow, matching
            // the composer control and the picker's own tooltip.
            return "Choose a model to start"
        }
        return "Chatting with \(alias)"
    }

    /// The download hint under the subtitle. Names the model only when
    /// it is a real alias; otherwise stays generic rather than
    /// interpolating a placeholder into the sentence.
    private var downloadHint: String? {
        guard let pending = autoStartPendingDownload else { return nil }
        guard !ModelDisplayName.isUnresolved(pending.alias) else {
            return "Your first message will download the selected model."
        }
        let size = pending.sizeText.map { " (\($0))" } ?? ""
        return "Your first message will download \(pending.alias)\(size)."
    }

    /// Speed can only measure a model that is actually up. Keyed on the
    /// live server state rather than the ``serverReady`` flag so the
    /// button re-enables the moment the model finishes starting.
    private var benchmarkAvailable: Bool {
        guard server.binaryPath != nil else { return false }
        // Routed through the shared predicate rather than a bare
        // ``isEmpty`` so every readiness decision in the app agrees on
        // what counts as "no model" — an internal placeholder is not a
        // model you can benchmark.
        if case .ready = server.state {
            return !ModelDisplayName.isUnresolved(alias)
        }
        return false
    }

    // MARK: - Compose bar

    private var composeBar: some View {
        VStack(spacing: RapidTheme.Space.sm) {
            if let error = viewModel.lastError {
                InlineNotice(message: error, tone: .error)
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
                ComposeField(
                    text: $draft,
                    focusToken: composeFocusToken,
                    isStreaming: viewModel.isStreaming,
                    onSubmit: send,
                    onCancel: { viewModel.stop() },
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
                    .strokeBorder(RapidTheme.hairlineStrong, lineWidth: 1)
            )
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
            Spacer(minLength: 0)
            ModelPickerBar(
                server: server,
                downloads: downloads,
                alias: $alias,
                quickstart: quickstart,
                composerStyle: true
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
            .help("Send")
            .accessibilityLabel("Send message")
        }
    }

    private var sendEnabled: Bool {
        !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    // MARK: - Actions

    private func send() {
        let text = draft
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        guard !viewModel.isStreaming else { return }
        draft = ""
        composeFocusToken &+= 1
        viewModel.send(text, alias: alias)
    }

    private func regenerate() {
        guard !viewModel.isStreaming else { return }
        viewModel.regenerateLast(alias: alias)
    }
}

/// One transcript row — a user prompt bubble, an assistant answer
/// (markdown + optional reasoning disclosure + stats/error captions),
/// or a neutral system note.
private struct MessageRow: View {
    let message: ChatMessage
    let isStreaming: Bool
    var onRegenerate: () -> Void = {}

    @State private var reasoningExpanded: Bool = false

    var body: some View {
        switch message.role {
        case .user:
            userBubble
        case .assistant:
            assistantBlock
        default:
            systemNote
        }
    }

    // MARK: User

    private var userBubble: some View {
        HStack {
            Spacer(minLength: 40)
            Text(message.content)
                .textSelection(.enabled)
                .foregroundStyle(RapidTheme.userBubbleText)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(
                    RoundedRectangle(cornerRadius: RapidTheme.Radius.bubble, style: .continuous)
                        .fill(RapidTheme.userBubble)
                )
        }
        .frame(maxWidth: .infinity, alignment: .trailing)
    }

    // MARK: Assistant

    private var assistantBlock: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !message.reasoning.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                reasoningDisclosure
            }
            if !message.content.isEmpty {
                LaTeXMarkdownView(content: message.content)
                    .textSelection(.enabled)
            } else if showTypingIndicator {
                typingIndicator
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
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var reasoningDisclosure: some View {
        DisclosureGroup(isExpanded: $reasoningExpanded) {
            Text(message.reasoning)
                .font(.callout)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Label(message.reasoningTruncated ? "Thinking trace (cut off)" : "Reasoning",
                  systemImage: "brain")
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)
        }
        .onAppear {
            // Auto-expand a truncated reasoning-only turn so the user
            // sees the partial trace instead of an empty bubble.
            if message.reasoningTruncated { reasoningExpanded = true }
        }
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
        HStack(spacing: 10) {
            // A failed turn is an ERROR, so it takes the error token.
            // It previously rendered in deep amber, which under this
            // palette means brand / active / working — the same hue the
            // product uses for a model that is starting up. Red is the
            // only colour that means "this went wrong".
            Text(message.errorMessage ?? "The model couldn't complete that request.")
                .font(.footnote)
                .foregroundStyle(RapidTheme.statusError)
            Button("Regenerate", action: onRegenerate)
                .buttonStyle(.link)
                .font(.footnote)
                .disabled(isStreaming)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// A ``.complete`` row can still carry a soft, non-error caption —
    /// the "Stopped." footer or the reasoning-only-truncated hint.
    private var softCaption: String? {
        guard message.status == .complete, let msg = message.errorMessage else { return nil }
        return msg
    }

    private var statsCaption: String? {
        guard message.status == .complete, let stats = message.stats else { return nil }
        let elapsed = AssistantStatsFormatter.formatElapsed(stats.elapsedSeconds)
        if let reported = stats.reportedTokensPerSecond {
            return "\(AssistantStatsFormatter.formatTPS(reported)) tok/s · \(elapsed)"
        }
        if let estimated = stats.estimatedTokensPerSecond {
            return "~\(AssistantStatsFormatter.formatTPS(estimated)) tok/s · \(elapsed)"
        }
        return elapsed
    }

    // MARK: System

    private var systemNote: some View {
        Text(message.content)
            .font(.footnote)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .center)
    }
}

private struct ComposeField: View {
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
    /// Resolves the text of the last user message in the active
    /// session, or ``nil`` when there's nothing to recall. Bound to
    /// the Up-arrow-in-empty-compose recall affordance (Claude /
    /// Raycast convention). Default ``{ nil }`` so existing call
    /// sites that don't wire it stay quiet.
    var onRecallLastUser: () -> String? = { nil }

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

    var body: some View {
        // v0.5 (Phase 5b): explicit height. The editor measures its own
        // text and reports it; we clamp and apply it via `.frame(height:)`.
        // Top-aligned (NSTextView default), so the caret/placeholder sit
        // at the top-left and the field hugs one line by default.
        ZStack(alignment: .topLeading) {
            // ``AutosizingTextView`` (the NSTextView subclass behind
            // ``ComposeTextEditor``) reports its content's measured
            // height as intrinsic size, so SwiftUI's
            // ``.frame(minHeight: 28, maxHeight: 160)`` on the parent
            // becomes a true elastic range: 28 pt for an empty draft
            // (hugging the placeholder) → grows one line at a time
            // → caps at 160 pt with internal scrolling beyond. The
            // ghost-Text autosize trick that this view used to need
            // was a no-op as long as the stock NSTextView returned
            // ``noIntrinsicMetric`` (it always grew to the cap), so
            // the trick is gone now.
            ComposeTextEditor(
                text: $text,
                focusToken: focusToken,
                isStreaming: isStreaming,
                onSubmit: onSubmit,
                onCancel: onCancel,
                onRecallLastUser: onRecallLastUser
            )
            if text.isEmpty {
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

/// ``NSTextView`` subclass that reports its content's measured height
/// as ``intrinsicContentSize``. The stock ``NSTextView`` returns
/// ``noIntrinsicMetric`` for both axes, which lets SwiftUI hand it the
/// whole vertical budget of its parent — so a parent ``.frame(maxHeight:
/// 160)`` becomes a *fixed* 160 pt box rather than a 28→160 pt elastic
/// pill. By overriding ``intrinsicContentSize`` to ``usedRect`` +
/// inset, SwiftUI's ``minHeight: 28, maxHeight: 160`` clamp becomes a
/// real elastic range that hugs single-line content and grows to the
/// cap as the user types. ``didChangeText`` invalidates so each edit
/// re-asks SwiftUI to re-measure.
final class AutosizingTextView: NSTextView {
    override var intrinsicContentSize: NSSize {
        guard let lm = layoutManager, let tc = textContainer else {
            return NSSize(width: NSView.noIntrinsicMetric, height: 28)
        }
        lm.ensureLayout(for: tc)
        let used = lm.usedRect(for: tc)
        return NSSize(
            width: NSView.noIntrinsicMetric,
            height: ceil(used.height) + textContainerInset.height * 2
        )
    }
    override func didChangeText() {
        super.didChangeText()
        invalidateIntrinsicContentSize()
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

    static func applyComposeAccessibility(_ tv: NSTextView) {
        tv.setAccessibilityLabel(composeAccessibilityLabel)
        tv.setAccessibilityIdentifier(composeAccessibilityIdentifier)
        tv.setAccessibilityRoleDescription(composeAccessibilityRoleDescription)
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
private struct ComposeTextEditor: NSViewRepresentable {
    @Binding var text: String
    var focusToken: Int
    var isStreaming: Bool
    var onSubmit: () -> Void
    var onCancel: () -> Void
    var onRecallLastUser: () -> String?

    func makeNSView(context: Context) -> NSTextView {
        let tv = AutosizingTextView()
        tv.delegate = context.coordinator
        tv.isRichText = false
        tv.font = .systemFont(ofSize: NSFont.systemFontSize)
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
        tv.textContainer?.widthTracksTextView = true
        tv.textContainer?.heightTracksTextView = false
        tv.textContainer?.containerSize = NSSize(
            width: 0, height: CGFloat.greatestFiniteMagnitude
        )
        // Bug 3-A residual P2: NSTextView already advertises role
        // ``.textArea`` by default, but with no label / identifier
        // AppleScript and cliclick can't tell which text area is the
        // chat compose vs the system-prompt editor or search bar.
        AutosizingTextView.applyComposeAccessibility(tv)
        return tv
    }

    func updateNSView(_ view: NSTextView, context: Context) {
        if view.string != text {
            view.string = text
        }
        context.coordinator.onSubmit = onSubmit
        context.coordinator.onCancel = onCancel
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
            isStreaming: isStreaming,
            onRecallLastUser: onRecallLastUser
        )
    }

    @MainActor
    final class Coordinator: NSObject, NSTextViewDelegate {
        var text: Binding<String>
        var onSubmit: () -> Void
        var onCancel: () -> Void
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
            isStreaming: Bool,
            onRecallLastUser: @escaping () -> String?
        ) {
            self.text = text
            self.onSubmit = onSubmit
            self.onCancel = onCancel
            self.isStreaming = isStreaming
            self.onRecallLastUser = onRecallLastUser
        }

        func textDidChange(_ notification: Notification) {
            guard let tv = notification.object as? NSTextView else { return }
            text.wrappedValue = tv.string
        }

        func textView(_ textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
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
            config.label
                .markdownTableBorderStyle(.init(color: .secondary.opacity(0.4)))
                .markdownMargin(top: 8, bottom: 8)
        }
}
/// LazyVStack's bottom edge in the transcript's named coord space.
/// Streams into ``ChatView`` so it can derive "is the user at the
/// bottom?" without polling.
private struct ContentBottomKey: PreferenceKey {
    static let defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = nextValue()
    }
}

/// ScrollView's visible height. Paired with ``ContentBottomKey`` to
/// compute the bottom-distance dead-band.
private struct ViewportHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = nextValue()
    }
}

/// Code block with a hover-revealed Copy button. ChatGPT Desktop
/// hangs the copy affordance off the top-right; we mirror that and
/// fade in on hover so the button doesn't distract during reading.
/// The button briefly flips to a checkmark on click so the user
/// knows the clipboard load actually happened.
private struct CodeBlockWithCopy: View {
    let config: CodeBlockConfiguration

    @State private var hovering: Bool = false
    @State private var copiedRecently: Bool = false

    var body: some View {
        ZStack(alignment: .topTrailing) {
            ScrollView(.horizontal, showsIndicators: false) {
                config.label
                    .relativeLineSpacing(.em(0.2))
                    .markdownTextStyle {
                        FontFamilyVariant(.monospaced)
                        FontSize(.em(0.92))
                    }
                    .padding(10)
            }
            .background(Color.secondary.opacity(0.12))
            .clipShape(RoundedRectangle(cornerRadius: 6))

            copyButton
                .padding(.top, 6)
                .padding(.trailing, 6)
                .opacity(hovering || copiedRecently ? 1.0 : 0.0)
                .rapidAnimation(RapidMotion.quick, value: hovering)
        }
        .onHover { h in hovering = h }
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
        if stats.elapsedSeconds > 0 {
            parts.append("took \(formatElapsed(stats.elapsedSeconds))")
        }
        return parts.joined(separator: ", ")
    }
}
