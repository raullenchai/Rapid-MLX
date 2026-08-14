import Foundation
import ImageIO
import UniformTypeIdentifiers

struct ChatImageAttachment: Codable, Equatable, Hashable, Identifiable, Sendable {
    static let maxBytes = 20 * 1024 * 1024

    let id: UUID
    let filename: String
    let mimeType: String
    let data: Data

    init(id: UUID = UUID(), filename: String, mimeType: String, data: Data) throws {
        guard data.count <= Self.maxBytes else { throw ValidationError.tooLarge }
        guard ["image/png", "image/jpeg", "image/gif"].contains(mimeType) else {
            throw ValidationError.unsupportedType
        }
        if mimeType == "image/gif",
           let source = CGImageSourceCreateWithData(data as CFData, nil),
           CGImageSourceGetCount(source) > 1 {
            throw ValidationError.animatedGIF
        }
        self.id = id
        self.filename = filename
        self.mimeType = mimeType
        self.data = data
    }

    init(contentsOf url: URL) throws {
        let values = try url.resourceValues(forKeys: [.fileSizeKey, .contentTypeKey])
        guard (values.fileSize ?? 0) <= Self.maxBytes else { throw ValidationError.tooLarge }
        let type = values.contentType
        let mime: String
        if type?.conforms(to: .png) == true { mime = "image/png" }
        else if type?.conforms(to: .jpeg) == true { mime = "image/jpeg" }
        else if type?.conforms(to: .gif) == true { mime = "image/gif" }
        else { throw ValidationError.unsupportedType }
        try self.init(filename: url.lastPathComponent, mimeType: mime, data: Data(contentsOf: url))
    }

    var dataURL: String { "data:\(mimeType);base64,\(data.base64EncodedString())" }

    enum ValidationError: LocalizedError {
        case tooLarge, unsupportedType, animatedGIF
        var errorDescription: String? {
            switch self {
            case .tooLarge: return "Images must be 20 MB or smaller."
            case .unsupportedType: return "Choose a PNG, JPEG, or non-animated GIF."
            case .animatedGIF: return "Animated GIFs aren't supported."
            }
        }
    }
}

/// One chat message. Mirrors the OpenAI chat-completions schema closely
/// enough that the stream client can serialise an array of these directly
/// into the wire body.
///
/// ``id`` is local-only — used by SwiftUI for diffing and by the session
/// store for indexed mutation while a stream is in flight.
struct ChatMessage: Identifiable, Codable, Equatable, Hashable {
    enum Role: String, Codable, Sendable {
        case user
        case assistant
        case system
        /// Tool-result message. Carries ``toolCallID`` so the model
        /// can match the response to the call it asked for. Created
        /// programmatically by ``ChatViewModel`` during the tool
        /// round-trip; the user never types one directly.
        case tool
        /// Issue #477 forward-compatibility fallback. A ``sessions.json``
        /// written by a NEWER build (or hand-edited) can carry a role the
        /// current build doesn't know — e.g. a new OpenAI-schema role, or
        /// a downgrade after an auto-update. The custom ``init(from:)``
        /// below maps any unrecognised raw string here instead of
        /// throwing, so one forward-incompatible message no longer wipes
        /// the whole library. Rendered as a neutral system note in the UI
        /// and FILTERED OUT of the outbound wire body (serialising
        /// ``{"role":"unknown"}`` would 400 the next send). Encodes back
        /// to the stable ``"unknown"`` sentinel via the synthesised
        /// ``encode(to:)`` — the original raw string is intentionally not
        /// preserved (plain case, not ``unknown(String)``, so the
        /// compiler flags every exhaustive switch that must handle it).
        case unknown

        /// Forward-tolerant decode: an unrecognised raw string degrades
        /// to ``.unknown`` rather than throwing. Non-string values (a
        /// role encoded as a number, say) still throw — that element is
        /// then dropped by ``FailableDecodable`` one level up.
        init(from decoder: Decoder) throws {
            let raw = try decoder.singleValueContainer().decode(String.self)
            self = Role(rawValue: raw) ?? .unknown
        }
    }

    /// Streaming phase for assistant messages. User messages are always
    /// ``.complete`` from the moment they're created; system messages
    /// likewise. Only the assistant placeholder cycles through
    /// ``.streaming`` / ``.complete`` / ``.failed``.
    enum Status: String, Codable, Sendable {
        case complete
        case streaming
        case failed
        /// Issue #477 forward-compatibility fallback — same rationale as
        /// ``Role.unknown``. An unrecognised status from a newer / edited
        /// envelope degrades here instead of throwing. Treated exactly
        /// like ``.complete`` for all runtime purposes: it is NOT
        /// ``.streaming``, so ``SessionStore``'s streaming-count seeding
        /// and the typing-dot UI never wedge on a restored ``.unknown``
        /// row. Encodes back to the stable ``"unknown"`` sentinel.
        case unknown

        /// Forward-tolerant decode: an unrecognised raw string degrades
        /// to ``.unknown`` rather than throwing.
        init(from decoder: Decoder) throws {
            let raw = try decoder.singleValueContainer().decode(String.self)
            self = Status(rawValue: raw) ?? .unknown
        }
    }

    let id: UUID
    let role: Role
    /// The visible assistant prose / user prompt body. For Qwen3.5/3.6
    /// hybrid-thinking responses this excludes any ``reasoning_content``
    /// the model produced — that goes into ``reasoning`` so the UI can
    /// render it in a collapsed disclosure block.
    var content: String
    var imageAttachments: [ChatImageAttachment]
    /// Locally extracted document text. Kept separate from ``content`` so a
    /// multi-page source does not flood the transcript or copied user prose.
    var fileAttachments: [ChatFileAttachment]
    /// Hybrid-thinking trace (mlx-lm ``reasoning_content`` field). Only
    /// populated for assistant messages from hybrid models; empty string
    /// is treated as "no trace" by the UI.
    var reasoning: String
    var status: Status
    /// Optional inline error string for ``.failed`` rows. Shown under the
    /// (possibly partial) content with a red caption.
    var errorMessage: String?
    /// Rule-based diagnosis rendered instead of raw tool/transport details.
    /// Optional so sessions written by older builds decode unchanged.
    var failureKind: FailureDiagnosis.Kind?
    /// Tool calls returned by an assistant turn. ``nil`` for any other
    /// role and for assistant turns that produced plain text. The chat
    /// loop reads this to decide whether to run tools and continue.
    var toolCalls: [ToolCall]?
    /// Set on ``role == .tool`` messages — the ``ToolCall.id`` the
    /// content is responding to. Required by the OpenAI spec.
    var toolCallID: String?
    /// v0.4.12: streaming-time + token-throughput stats. Populated
    /// at end-of-stream for assistant messages so the UI can show
    /// a small caption ("~84 tok/s · 2.4 s"). ``nil`` for all
    /// older sessions decoded from disk — the custom init below
    /// defaults it on missing-key, mirroring the pre-existing
    /// pattern for ``ChatSession.isPinned``.
    var stats: MessageStats?
    /// Cycle-2 (2026-06-19) F-002 marker: set to ``true`` ONLY when a
    /// reasoning model exhausted its ``max_tokens`` budget mid-think,
    /// producing empty ``content`` + populated ``reasoning`` +
    /// ``finish_reason: "length"`` (see
    /// ``ChatViewModel.TerminalOutcome.reasoningOnlyTruncated``).
    /// The chat view keys on THIS flag (not on
    /// ``content.isEmpty && !reasoning.isEmpty``) to decide whether
    /// to auto-expand the reasoning disclosure, relabel it as
    /// "Thinking trace (cut off)", and route the VoiceOver
    /// accessibility caption — because a user-cancelled stream and a
    /// chat-template parser bug can ALSO land with empty content +
    /// populated reasoning, and they need different UX.
    ///
    /// Defaults to ``false`` everywhere. Old on-disk sessions decoded
    /// before the cycle-2 release have no key for this field; the
    /// custom ``init(from:)`` below uses ``decodeIfPresent`` with a
    /// ``false`` fallback so they load cleanly. (Swift's synthesised
    /// ``Decodable`` would have thrown on a missing non-optional
    /// ``Bool``; codex r1 NIT clarified.)
    var reasoningTruncated: Bool
    /// Cycle-13 (2026-06-20) F-5 marker: set to ``true`` ONLY when a
    /// non-reasoning assistant turn exhausted its ``max_tokens`` budget
    /// mid-answer — i.e. ``finish_reason == "length"`` AND ``content``
    /// is non-empty AND ``reasoning`` is empty (see
    /// ``ChatViewModel.runOneStream``'s ``.finished`` handler).
    /// Verbose-output dense models (nemotron-30b-4bit and similar)
    /// would emit a 200-token LaTeX derivation for "what is 17*23?"
    /// against the default 200-token cap and the row would render as
    /// a normal completed answer with no indication the model was
    /// cut off — so the user reads a half-finished derivation as the
    /// real reply.
    ///
    /// The chat view keys on THIS flag (not on ``finish_reason ==
    /// "length"`` directly) to decide whether to paint a subtle
    /// "Answer cut off (Max Tokens hit). Increase Max Tokens to see
    /// the rest." caption inline at the bottom of the bubble.
    ///
    /// Disjoint from ``reasoningTruncated``:
    ///   * ``reasoningTruncated`` — empty content + populated
    ///     reasoning + length (PR #317's reasoning-only fallback).
    ///   * ``contentTruncated`` — populated content + empty reasoning
    ///     + length (this cycle-13 verbose-output fix).
    ///   * Both false on a content + reasoning + length shape, since
    ///     ``classifyTerminal`` already treats that as a real
    ///     completion (the answer body landed). See
    ///     ``ChatViewVerboseOutputBadgeTests`` for the 4-cell truth
    ///     table.
    ///
    /// Defaults to ``false`` everywhere. Old on-disk sessions decoded
    /// before this cycle's release have no key for this field; the
    /// custom ``init(from:)`` below uses ``decodeIfPresent`` with a
    /// ``false`` fallback so they load cleanly (same back-compat
    /// shim used for ``reasoningTruncated``).
    var contentTruncated: Bool
    /// Issue #308 (2026-06-20) marker: set to ``true`` ONLY when an
    /// assistant turn met every gate in ``shouldFlagToolNotCalled``
    /// — tools were sent in the request body, zero ``tool_calls``
    /// were emitted, prose body looks like a raw / numeric answer,
    /// AND the user's prompt looked calculator- / search-shaped.
    /// The chat view paints a lightweight dismissible caption above
    /// the bubble ("This model didn't call a tool — verify the
    /// answer.") so the user isn't silently misled by a small
    /// model's hallucinated calculation.
    ///
    /// Defaults to ``false`` everywhere. Old on-disk sessions
    /// decoded before this release have no key for this field; the
    /// custom ``init(from:)`` below uses ``decodeIfPresent`` with a
    /// ``false`` fallback so they load cleanly (same back-compat
    /// shim used for ``reasoningTruncated`` /
    /// ``contentTruncated``).
    var toolNotCalledFlagged: Bool
    /// Issue #513 marker (defense-in-depth, layer 3): set to ``true``
    /// ONLY when a finished assistant turn had tools advertised on the
    /// request, fired **zero** ``tool_calls``, AND its ``content`` is
    /// essentially just a malformed tool-call artifact the engine parser
    /// couldn't recover (a raw ``<tool_call>`` / ``<parameter=`` /
    /// ``[TOOL_CALLS]`` / ``<｜tool▁calls▁begin｜>`` fragment or a bare
    /// tool-call-shaped JSON object). The chat view then replaces the
    /// raw dump with a quiet "the model tried to use a tool but its
    /// request couldn't be read" caption instead of rendering the
    /// envelope — so a rare per-turn glitch on an otherwise-good model
    /// doesn't surface confusing machine syntax to the user. See
    /// ``shouldSuppressToolCallArtifact``.
    ///
    /// Defaults to ``false`` everywhere. Old on-disk sessions decoded
    /// before this release have no key for this field; the custom
    /// ``init(from:)`` below uses ``decodeIfPresent`` with a ``false``
    /// fallback so they load cleanly (same back-compat shim used for
    /// ``reasoningTruncated`` / ``contentTruncated`` /
    /// ``toolNotCalledFlagged``).
    var toolCallArtifactSuppressed: Bool
    let createdAt: Date

    init(
        id: UUID = UUID(),
        role: Role,
        content: String = "",
        imageAttachments: [ChatImageAttachment] = [],
        fileAttachments: [ChatFileAttachment] = [],
        reasoning: String = "",
        status: Status = .complete,
        errorMessage: String? = nil,
        failureKind: FailureDiagnosis.Kind? = nil,
        toolCalls: [ToolCall]? = nil,
        toolCallID: String? = nil,
        stats: MessageStats? = nil,
        reasoningTruncated: Bool = false,
        contentTruncated: Bool = false,
        toolNotCalledFlagged: Bool = false,
        toolCallArtifactSuppressed: Bool = false,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.imageAttachments = imageAttachments
        self.fileAttachments = fileAttachments
        self.reasoning = reasoning
        self.status = status
        self.errorMessage = errorMessage
        self.failureKind = failureKind
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
        self.stats = stats
        self.reasoningTruncated = reasoningTruncated
        self.contentTruncated = contentTruncated
        self.toolNotCalledFlagged = toolNotCalledFlagged
        self.toolCallArtifactSuppressed = toolCallArtifactSuppressed
        self.createdAt = createdAt
    }

    /// Codex r1 MAJOR-1: keep ``reasoningTruncated`` decodable from
    /// pre-cycle-2 session envelopes (the on-disk JSON has no such
    /// key). Swift's synthesised init(from:) throws on a missing
    /// non-optional, so the custom init below falls back to ``false``
    /// for that one key and defers all other fields to the standard
    /// container shape.
    enum CodingKeys: String, CodingKey {
        case id, role, content, imageAttachments, fileAttachments, reasoning, status
        case errorMessage, failureKind, toolCalls, toolCallID
        case stats, reasoningTruncated, contentTruncated
        case toolNotCalledFlagged
        case toolCallArtifactSuppressed
        case createdAt
        /// The outcome as THIS build understands it. ``failureKind`` carries
        /// the same outcome narrowed to a value older builds can decode — see
        /// ``encode(to:)``.
        case failureKindV2
    }

    /// Hand-written so ``failureKind`` is persisted TWICE.
    ///
    /// ``FailureDiagnosis.Kind`` decodes strictly in every build already in
    /// users' hands, and ``ConversationStore.load`` turns ONE undecodable
    /// message into "the whole history is corrupt": the file is sided to
    /// `conversations.corrupt-<uuid>.json` and the sidebar comes up empty.
    /// So a raw value added after a release must never be written into the
    /// key those builds read — that would cost a user who downgrades their
    /// entire visible history, which is precisely the failure ``Role.unknown``
    /// and ``Status.unknown`` exist to prevent on the other two enums.
    ///
    /// Hence: ``failureKind`` gets ``legacyPersistedKind`` (always a value
    /// every shipped build knows, so an old build reads a slightly coarser
    /// outcome and keeps the conversation), and ``failureKindV2`` — a key old
    /// builds simply ignore — gets the real one. Adding a future kind needs
    /// nothing here beyond its ``legacyPersistedKind`` mapping.
    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(role, forKey: .role)
        try c.encode(content, forKey: .content)
        try c.encode(imageAttachments, forKey: .imageAttachments)
        try c.encode(fileAttachments, forKey: .fileAttachments)
        try c.encode(reasoning, forKey: .reasoning)
        try c.encode(status, forKey: .status)
        try c.encodeIfPresent(errorMessage, forKey: .errorMessage)
        try c.encodeIfPresent(failureKind?.legacyPersistedKind, forKey: .failureKind)
        try c.encodeIfPresent(failureKind, forKey: .failureKindV2)
        try c.encodeIfPresent(toolCalls, forKey: .toolCalls)
        try c.encodeIfPresent(toolCallID, forKey: .toolCallID)
        try c.encodeIfPresent(stats, forKey: .stats)
        try c.encode(reasoningTruncated, forKey: .reasoningTruncated)
        try c.encode(contentTruncated, forKey: .contentTruncated)
        try c.encode(toolNotCalledFlagged, forKey: .toolNotCalledFlagged)
        try c.encode(toolCallArtifactSuppressed, forKey: .toolCallArtifactSuppressed)
        try c.encode(createdAt, forKey: .createdAt)
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try c.decode(UUID.self, forKey: .id)
        self.role = try c.decode(Role.self, forKey: .role)
        self.content = try c.decode(String.self, forKey: .content)
        self.imageAttachments = try c.decodeIfPresent([ChatImageAttachment].self, forKey: .imageAttachments) ?? []
        self.fileAttachments = try c.decodeIfPresent([ChatFileAttachment].self, forKey: .fileAttachments) ?? []
        self.reasoning = try c.decode(String.self, forKey: .reasoning)
        self.status = try c.decode(Status.self, forKey: .status)
        self.errorMessage = try c.decodeIfPresent(String.self, forKey: .errorMessage)
        // Prefer the finer v2 value; fall back to the original key for rows
        // written before it existed (and for rows an older build re-saved).
        //
        // Read v2 as a raw string rather than through ``Kind``'s tolerant
        // decode: a value this build doesn't recognise came from a NEWER one,
        // which wrote its closest known ancestor into ``failureKind`` — and
        // that is strictly better than the blanket degrade to ``.toolFailed``.
        let modernRaw = try c.decodeIfPresent(String.self, forKey: .failureKindV2)
        let legacyKind = try c.decodeIfPresent(FailureDiagnosis.Kind.self, forKey: .failureKind)
        self.failureKind = modernRaw.flatMap(FailureDiagnosis.Kind.init(rawValue:)) ?? legacyKind
        self.toolCalls = try c.decodeIfPresent([ToolCall].self, forKey: .toolCalls)
        self.toolCallID = try c.decodeIfPresent(String.self, forKey: .toolCallID)
        self.stats = try c.decodeIfPresent(MessageStats.self, forKey: .stats)
        self.reasoningTruncated = try c.decodeIfPresent(Bool.self, forKey: .reasoningTruncated) ?? false
        // Cycle-13 (2026-06-20) F-5: same back-compat shim as
        // ``reasoningTruncated`` — sessions saved before this cycle's
        // release have no key, so we default to false.
        self.contentTruncated = try c.decodeIfPresent(Bool.self, forKey: .contentTruncated) ?? false
        // Issue #308 (2026-06-20): same back-compat shim — sessions
        // saved before this release have no key for this field; we
        // default to false so old transcripts decode cleanly.
        self.toolNotCalledFlagged = try c.decodeIfPresent(Bool.self, forKey: .toolNotCalledFlagged) ?? false
        // Issue #513: same back-compat shim — sessions saved before this
        // release have no key for this field; default to false so old
        // transcripts decode cleanly.
        self.toolCallArtifactSuppressed = try c.decodeIfPresent(Bool.self, forKey: .toolCallArtifactSuppressed) ?? false
        self.createdAt = try c.decode(Date.self, forKey: .createdAt)
    }

    /// Text sent to the model. Document extracts stay out of the visible
    /// ``content`` property but remain part of this turn on every retry and
    /// follow-up request.
    var modelContent: String {
        guard !fileAttachments.isEmpty else { return content }
        let request = content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? "Analyze the attached file and summarize the important findings."
            : content
        return ([request] + fileAttachments.map(\.promptText))
            .joined(separator: "\n\n")
    }

    /// Resolve a tool failure outside the SwiftUI render tree. This lets the
    /// classifier inspect the raw model-facing payload while keeping that
    /// payload entirely out of the failed-result display branch.
    func toolFailureDiagnosis(toolName: String) -> FailureDiagnosis {
        let kind = failureKind
            ?? FailureDiagnoser.toolFailureKind(
                toolName: toolName,
                content: content,
                isError: true
            )
            ?? .toolFailed
        return FailureDiagnoser.diagnosis(for: kind)
    }

    // MARK: - Cycle-8 (F-CORR-3): tool-dispatch placeholder caption

    /// Returns a "Calling <tool_name>…" caption for an assistant
    /// message that has dispatched one or more tool calls without any
    /// preamble prose AND without a reasoning trace. ``nil`` for every
    /// other shape — see the case table below for the exact gates.
    ///
    /// ## Background
    ///
    /// Cycle-6 fuzz-correctness F-CORR-3 (filed 2026-06-19 against
    /// gemma-4-26b) caught this user-facing failure mode: when an
    /// assistant turn emits a ``tool_calls`` envelope with empty
    /// ``content`` and no ``reasoning_content`` (the model dispatches
    /// the tool with no preamble narration), the chat surface
    /// rendered only the small ``ToolCallChip`` row — a wrench-icon
    /// chip with a chevron. To a casual user that reads as debug
    /// metadata, not "the assistant is dispatching a tool right
    /// now"; the bubble looks blank for the 1-2 seconds before the
    /// tool result arrives and the chip flips to a checkmark.
    ///
    /// ## Case table
    ///
    ///   * **(a)** ``content`` non-empty (after whitespace trim) →
    ///     ``nil``. The model already narrated, the prose body
    ///     speaks for itself; a redundant "Calling…" caption would
    ///     just clutter the bubble.
    ///   * **(b)** ``reasoning`` non-empty (after whitespace trim) →
    ///     ``nil``. PR #317's reasoning fallback owns this shape
    ///     (auto-expanded "Thinking trace (cut off)" disclosure for
    ///     ``reasoningTruncated`` rows, "Thinking…" / "Reasoning"
    ///     for everything else). Returning ``nil`` keeps the two
    ///     fallbacks from double-painting.
    ///   * **(c)** ``content`` empty + ``reasoning`` empty +
    ///     ``toolCalls`` non-empty → "Calling
    ///     `<name>`…" / "Calling `<a>`, `<b>`…" — the F-CORR-3
    ///     fix path.
    ///   * **(d)** Everything empty / nil → ``nil``. Manufacturing a
    ///     caption out of thin air would lie about the model's
    ///     behaviour; the existing ``…`` ProgressView spinner is
    ///     correct for the still-streaming-nothing case.
    ///
    /// ## Hardening
    ///
    /// Tool names are sanitised through ``ChatTextSanitizer`` so a
    /// malicious or crash-corrupted SSE chunk can't inject NUL bytes
    /// or bidi-override controls (``U+202A`` … ``U+202E``,
    /// ``U+2066`` … ``U+2069``) into the caption. The bidi case
    /// matters: an unbalanced RTL override would re-flow every
    /// subsequent character in the bubble (and arguably in the
    /// surrounding row) right-to-left. ``ChatTextSanitizer`` already
    /// strips those scalars and is the right common-path for any
    /// untrusted text the chat view paints.
    ///
    /// ## Handoff to the ``ToolCallChip``
    ///
    /// The placeholder is an in-flight affordance ONLY — once every
    /// dispatched call has a matching tool-result message in the
    /// transcript, the ``ToolCallChip`` row owns the completed state
    /// (checkmark / error icon + expanded result body), and a
    /// lingering "Calling web_search…" caption above it would lie
    /// about the dispatch still being in flight. ``settledToolCallIDs``
    /// carries the set of ``ToolCall.id`` values that already have a
    /// result; when EVERY call's id appears in that set the helper
    /// returns ``nil`` and the chip(s) take over the row.
    ///
    /// Pre-cycle-8 / pre-codex-r1 the helper ignored the result set,
    /// which made the placeholder stick even after the tool round
    /// completed. The view layer passes the same
    /// ``[ToolCall.id: ChatMessage]`` map it already computes for the
    /// chip row, so no extra plumbing.
    ///
    /// ## Caller contract
    ///
    /// The view layer is responsible for only invoking this on
    /// assistant rows — see ``MessageRow.assistantBlock`` (which
    /// short-circuits to nil for any non-assistant role before
    /// calling through). The helper itself is role-agnostic so it
    /// can be unit-tested without standing up a ``ChatMessage``
    /// instance for every shape; see
    /// ``ToolCallDispatchPlaceholderTests`` for the case-table +
    /// settlement coverage.
    static func toolDispatchPlaceholderCaption(
        content: String,
        reasoning: String,
        toolCalls: [ToolCall]?,
        settledToolCallIDs: Set<String> = []
    ) -> String? {
        // Case (a): visible prose already speaks for the model.
        let contentEmpty = content
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .isEmpty
        guard contentEmpty else { return nil }

        // Case (b): reasoning trace path is owned by PR #317.
        let reasoningEmpty = reasoning
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .isEmpty
        guard reasoningEmpty else { return nil }

        // Case (d): nothing to surface.
        guard let calls = toolCalls, !calls.isEmpty else { return nil }

        // Codex r1 BLOCKING-1: handoff to ``ToolCallChip``. Once every
        // dispatched call has a matching tool-result row in the
        // transcript, the placeholder MUST step aside so the chip's
        // own completed state (checkmark / error / result body) reads
        // as the source of truth. Partial settlement (e.g. one of two
        // parallel calls back, the other still in flight) keeps the
        // placeholder up so the user still sees the in-flight signal
        // for the pending call — the caption itself only lists pending
        // calls in that branch, so a half-settled multi-dispatch reads
        // honestly.
        let pendingCalls = calls.filter { !settledToolCallIDs.contains($0.id) }
        guard !pendingCalls.isEmpty else { return nil }

        // Case (c): build the placeholder. Sanitise every tool name
        // through the shared chat-text sanitiser so control chars
        // and bidi overrides can't ride into the bubble. Empty
        // names (after sanitise) fall back to a generic "tool"
        // token so the caption still reads as in-flight.
        let sanitisedNames: [String] = pendingCalls.map { call in
            let cleaned = ChatTextSanitizer.sanitizeForDisplay(call.function.name)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return cleaned.isEmpty ? "tool" : cleaned
        }

        // Single-call shape reads as: "Calling web_search…"
        // Multi-call shape reads as: "Calling web_search, weather…"
        // We deliberately avoid an Oxford-comma "and" join so the
        // line stays grammatically neutral on three-or-more dispatches
        // ("a, b, c…") and stays narrow enough to fit a single
        // bubble line on a typical chat column.
        let joined = sanitisedNames.joined(separator: ", ")
        return "Calling \(joined)…"
    }

    // MARK: - Cycle-13 (2026-06-20) F-5: length-truncation badge

    /// Returns ``true`` when a non-reasoning assistant turn was cut
    /// off by ``max_tokens`` mid-answer — i.e. an answer body exists
    /// (or at least started) but the model never reached its natural
    /// stop token. The chat view paints a subtle inline caption
    /// ("Answer cut off (Max Tokens hit). Increase Max Tokens to see
    /// the rest.") under the bubble for this shape so the user
    /// understands the half-finished body isn't a real answer.
    ///
    /// ## Gates
    ///
    ///   * ``finish_reason == "length"`` — the server explicitly told
    ///     us the cap was hit. Without this gate every short answer
    ///     a verbose model produced would falsely wear the badge.
    ///   * ``content`` non-empty after whitespace trim — there's an
    ///     answer body to be "cut off". The empty-content + length
    ///     shape is owned by ``classifyTerminal`` (it routes to either
    ///     ``.reasoningOnlyTruncated`` or ``.emptyTurnFailure``); the
    ///     badge path stays out of that lane.
    ///   * ``reasoning`` empty after whitespace trim — a populated
    ///     reasoning trace on a length-truncated turn is PR #317's
    ///     domain (the reasoning-only fallback owns the
    ///     ``reasoningTruncated`` flag + auto-expanded disclosure
    ///     copy). The badge path stays out of that lane too, so a
    ///     reasoning model that hit the cap mid-answer-after-thinking
    ///     gets the reasoning fallback's copy + auto-expand, not a
    ///     redundant length-truncated badge.
    ///
    /// Role-agnostic so unit tests can exercise the gates directly;
    /// the view layer enforces "only paint on assistant rows".
    ///
    /// See ``ChatViewVerboseOutputBadgeTests`` for the 4-cell truth
    /// table covering (finish_reason ∈ {.length, .stop}) ×
    /// (reasoning ∈ {empty, non-empty}).
    static func shouldFlagContentTruncated(
        content: String,
        reasoning: String,
        finishReason: String?
    ) -> Bool {
        guard finishReason == "length" else { return false }
        let contentEmpty = content
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .isEmpty
        guard !contentEmpty else { return false }
        let reasoningEmpty = reasoning
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .isEmpty
        guard reasoningEmpty else { return false }
        return true
    }

    /// Cycle-13 F-5 (2026-06-20): pinned visible copy for the
    /// length-truncation badge. Lifted onto ``ChatMessage`` (the model
    /// layer) instead of ``MessageRow`` (the view layer) because
    /// ``MessageRow`` is ``private`` to ``ChatView.swift`` and the
    /// test suite needs to snapshot the exact string an accidental
    /// reword would change. ``MessageRow`` reads through this
    /// constant so the painted string and the snapshotted string
    /// can never drift apart.
    ///
    /// Copy reference: ChatGPT-Desktop's "Continue generating" pattern.
    /// We surface (i) what happened ("Answer cut off") and (ii) the
    /// user-facing knob to raise ("Max Tokens"). See
    /// ``ChatViewVerboseOutputBadgeTests.badgeCopySnapshot``.
    static let lengthTruncationBadgeCopy: String =
        "Answer cut off (Max Tokens hit). Increase Max Tokens to see the rest."

    /// Cycle-13 F-5 (2026-06-20): VoiceOver caption for the
    /// length-truncation badge. Reads the same information as the
    /// visible text but as a single well-formed sentence so the screen
    /// reader's pacing doesn't break on the parenthetical. Pinned
    /// alongside the visible copy so an accessibility regression
    /// surfaces in the tests, not silently in production.
    static let lengthTruncationBadgeAccessibilityLabel: String =
        "Answer cut off because the Max Tokens limit was hit. Increase Max Tokens in Settings to see the rest of the answer."

    // MARK: - Issue #308: tool-not-called caption

    /// Returns ``true`` for an assistant turn that landed with
    /// ``finish_reason: "stop"``-like terminal AND the request was
    /// built with a non-empty ``tools`` array AND the assistant
    /// produced ZERO ``tool_calls`` AND the prose body looks like
    /// a raw numeric answer to a calculator-style prompt.
    ///
    /// ## Background — issue #308
    ///
    /// The original Quickstart "Speed" pick (``gemma3-1b-qat-4bit``)
    /// is too small to reliably emit ``tool_calls`` for arithmetic
    /// prompts. Its sibling fix in this PR is to swap to a tool-
    /// capable alias (``qwen3.5-4b-4bit``), but the underlying class
    /// of failure — "model could have called a tool but silently
    /// answered raw instead, with a wrong answer" — applies to any
    /// future small / weak / out-of-distribution model the user
    /// picks themselves. A lightweight caption above the bubble
    /// ("This model didn't call the calculator — verify the
    /// answer") is a cheap, dismissible signal that prevents the
    /// silent-wrong-answer failure mode.
    ///
    /// Trigger gates (ALL must hold):
    ///
    ///   * ``toolsRequested == true`` — the request body actually
    ///     carried a non-empty ``tools`` array. Without this gate
    ///     every short numeric answer would wear the caption.
    ///   * ``toolSucceededThisTurn == false`` — no tool SUCCEEDED
    ///     earlier in this same turn. A multi-step tool turn (call
    ///     ``calculator`` → read a good result → summarise it) ends on a
    ///     summary message with no ``toolCalls`` of its own, but a tool
    ///     WAS used and the chip already says so; captioning that summary
    ///     would contradict the chip on screen. A tool that ERRORED does
    ///     NOT count as succeeded — a hallucinated raw answer after a
    ///     failed tool is exactly the shape we still want to flag.
    ///   * ``finishReason`` is ``nil`` or anything OTHER than
    ///     ``"tool_calls"`` — a real tool-call turn doesn't need
    ///     the caption (the chip row already speaks for it). A
    ///     ``"length"`` truncation also fires the caption: a
    ///     half-finished raw-numeric answer is still suspect.
    ///   * ``toolCalls`` is nil or empty — the model produced no
    ///     ``tool_calls`` for this turn.
    ///   * Prose body matches ``shouldFlagToolNotCalled``'s
    ///     numeric-or-short heuristic AND the user's prompt looks
    ///     calculator-shaped (see ``promptLooksCalculatorish``).
    ///
    /// The heuristic is intentionally loose — a false-positive
    /// caption ("model probably tool-called; this caption is wrong")
    /// is annoying but harmless; a false-negative (silent wrong
    /// answer) is the bug we're fixing. The view layer is
    /// responsible for making the caption dismissible (one-shot per
    /// session) so a user who knows better can mute it.
    ///
    /// Role-agnostic (assertion-only check); the view layer enforces
    /// "only paint on assistant rows".
    static func shouldFlagToolNotCalled(
        userPrompt: String,
        assistantContent: String,
        toolCalls: [ToolCall]?,
        finishReason: String?,
        toolsRequested: Bool,
        toolSucceededThisTurn: Bool = false
    ) -> Bool {
        // Gate 1: tools must have actually been advertised. Without
        // this gate every short numeric answer would wear the caption.
        guard toolsRequested else { return false }
        // Gate 1b: no tool SUCCEEDED earlier in this turn. A multi-step
        // tool turn — the model calls e.g. ``calculator``, gets a good
        // result back, then writes a plain-language summary of it —
        // leaves the FINAL assistant message with an empty ``toolCalls``
        // array of its own. But a tool WAS used this turn and the
        // visible tool-call chip already says so, so captioning that
        // summary "didn't call a tool" is a false positive that flatly
        // contradicts the chip on screen. Note "succeeded", not merely
        // "attempted": a tool that ERRORED and left the model to
        // hallucinate a raw answer is exactly the #308 failure mode, so
        // that case must still fire. ``toolSucceededThisTurn`` is
        // computed at the call site from the turn's message history (see
        // ``ChatViewModel.turnHadSuccessfulTool``).
        guard !toolSucceededThisTurn else { return false }
        // Gate 2: model must have produced no tool_calls. A real
        // tool-call turn doesn't need the caption.
        let noToolCalls = (toolCalls?.isEmpty ?? true)
        guard noToolCalls else { return false }
        // Gate 3: don't fire on a turn that finished with
        // finish_reason == "tool_calls" (a tool-call landed at the
        // last moment but the captured array is empty — corner case,
        // but be conservative).
        if finishReason == "tool_calls" { return false }
        // Gate 4: prose must look like a "raw answer" — short or
        // numeric-dominated. A long, well-cited prose reply is not
        // the failure shape we're guarding against.
        guard contentLooksLikeRawAnswer(assistantContent) else { return false }
        // Gate 5: user prompt must look calculator-shaped or
        // tool-shaped (a math question, a "what's the weather"
        // question, a "search for" prompt). Without this gate every
        // "yes" / "no" assistant reply to a casual question would
        // wear the caption.
        guard promptLooksCalculatorish(userPrompt) else { return false }
        return true
    }

    // MARK: - Issue #513: raw tool-call artifact suppression

    /// User-facing caption rendered in place of a suppressed raw
    /// tool-call artifact (issue #513). Deliberately jargon-free — no
    /// "envelope" / "parser" / "tool_call" machine syntax — the user
    /// only needs to know the turn didn't yield a usable answer and what
    /// to do next. Pinned by a test so a reword stays intentional.
    static let toolCallArtifactSuppressedCaptionCopy =
        "This model tried to use a tool but its request couldn't be read. Try again, or pick a different model."

    /// Gate for the render-time safety net (issue #513). True ONLY when
    /// a finished assistant turn:
    ///   1. had tools advertised on the request, AND
    ///   2. produced zero ``tool_calls`` (and did not finish as
    ///      ``finish_reason == "tool_calls"``), AND
    ///   3. its ``content`` is essentially just a malformed tool-call
    ///      artifact the engine parser couldn't recover
    ///      (``contentLooksLikeToolCallArtifact``).
    ///
    /// When it holds, the chat view replaces the raw envelope dump with
    /// a quiet caption instead of rendering machine syntax to the user.
    /// This is defense-in-depth layer 3: recommendation curation removes
    /// systematically-broken aliases, but can't catch a rare per-turn
    /// glitch on an otherwise-good (or manually-selected `.unknown`)
    /// model.
    static func shouldSuppressToolCallArtifact(
        content: String,
        toolCalls: [ToolCall]?,
        finishReason: String?,
        toolsRequested: Bool
    ) -> Bool {
        // Gate 1: tools must have been advertised — otherwise a model
        // that legitimately answered with a JSON object was never asked
        // to call a tool, so there's nothing to suppress.
        guard toolsRequested else { return false }
        // Gate 2: a real tool-call turn (non-empty array) is handled by
        // the normal tool-dispatch path, never suppressed.
        guard (toolCalls?.isEmpty ?? true) else { return false }
        // Gate 3: a turn that finished as "tool_calls" landed a call at
        // the last moment even if the captured array reads empty — be
        // conservative and leave it alone.
        if finishReason == "tool_calls" { return false }
        // Gate 4: the content must actually look like a raw tool-call
        // artifact, not a genuine answer that merely embeds JSON.
        //
        // Accepted residual (issue #513 enumerates these exact shapes as
        // targets and scopes the risk to "content that merely CONTAINS
        // JSON — code answers, here's-a-JSON-example"): a user who, in a
        // tools-enabled session, explicitly asks the model to return ONLY
        // a raw tool-call example gets the same shape a leak would, and no
        // content-based test can separate the two. This is deliberately
        // tolerated because (a) it is the exact defense the issue author
        // requested, (b) suppression is NON-DESTRUCTIVE — the raw content
        // stays on the message and copy / export reproduce it verbatim; it
        // is only the inline render that swaps to a caption, and (c) in the
        // target population (glitchy local models whose parser can't
        // recover a call) a whole-content call shape with tools advertised
        // + zero tool_calls is far more likely a parser miss than a
        // deliberately-requested example. Prose-FRAMED examples ("here's a
        // JSON example: …") are NOT suppressed — see the detector.
        return contentLooksLikeToolCallArtifact(content)
    }

    /// True when ``content`` is *essentially just* a malformed tool-call
    /// artifact — a raw envelope/fragment the engine parser couldn't turn
    /// into a real ``tool_calls`` array — rather than a genuine answer
    /// that merely mentions or embeds JSON (issue #513).
    ///
    /// Deliberately conservative, because the headline risk is eating
    /// legitimate content (a code answer, a "here's a JSON example"):
    ///   * Envelope / fragment markers (`<tool_call>`, `<parameter=`,
    ///     `<function=`, `<JSON>`) match only when the RAW (un-fenced)
    ///     content STARTS with them. A leaked call begins the assistant
    ///     turn dumped raw; an answer that *discusses* `<tool_call>` has
    ///     prose before it, and a ```` ```xml ```` example the user asked
    ///     for is never unwrapped into these checks.
    ///   * The DeepSeek `tool▁calls▁begin` (U+2581 separators) token is
    ///     literal tool-call syntax that never appears in prose, so a
    ///     contained match is safe.
    ///   * A bare JSON object matches only when the WHOLE content is one
    ///     object (raw or in a single ```` ```json ```` fence) shaped like
    ///     a tool CALL — the OpenAI `{"type":"function","function":{…}}`
    ///     wire shape, or a Hermes/ReAct object that names a tool AND
    ///     carries an args container with no definition/schema key. A tool
    ///     DEFINITION (`{"name":…,"parameters":{"type":"object"}}`) and a
    ///     data record that merely reuses `name` are NOT hit.
    static func contentLooksLikeToolCallArtifact(_ content: String) -> Bool {
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }

        // The DeepSeek `tool▁calls▁begin` token uses U+2581 separators
        // that never occur in human prose or a normal answer, so a
        // CONTAINED match is safe anywhere in the body.
        if trimmed.contains("tool\u{2581}calls\u{2581}begin") { return true }

        // Envelope / fragment shapes that must LEAD the raw (un-fenced)
        // turn AND be followed by a structured payload — not prose. The
        // checks run on `trimmed`, never on a fence-stripped body:
        //   * A real leak is dumped raw, not inside a ```` ``` ```` fence.
        //   * A legitimate ```` ```xml ```` / ```` ```html ```` answer whose
        //     body happens to begin `<tool_call>` (example markup the user
        //     asked for) must NOT be unwrapped and then mistaken for a leak.
        // Anchoring to the prefix keeps an answer that merely DISCUSSES the
        // syntax safe; the payload check keeps an answer that *leads* with
        // the token but explains it (`"<JSON> is a wrapper used by…"`,
        // `"<tool_call> is a special tag…"`) from being suppressed.
        if leadingEnvelopeLeak(in: trimmed) { return true }

        // Bare tool-call JSON object — raw, or wrapped in a single
        // whole-content ```` ```json ```` fence. Fence-unwrapping is
        // confined to THIS branch (and to a `{`-leading body), so it can
        // never feed the envelope checks above.
        let jsonBody = strippedSingleJSONFence(trimmed)
        guard jsonBody.hasPrefix("{") else { return false }
        return jsonObjectLooksLikeToolCall(jsonBody)
    }

    /// True when the trimmed content LEADS with a tool-call envelope
    /// marker AND is followed by that format's structured payload (JSON,
    /// a nested tag, or the format's arg delimiter). The payload gate is
    /// what separates a leaked call from an answer that merely opens with
    /// the token to explain it — the headline false-positive risk (#513).
    private static func leadingEnvelopeLeak(in trimmed: String) -> Bool {
        // Hermes / qwen XML envelope. A real leak is `<tool_call>{…` or the
        // truncated `<tool_call><parameter=…>` repro, or carries a closing
        // tag; a prose answer is `<tool_call> is a tag…`.
        if trimmed.hasPrefix("<tool_call") || trimmed.hasPrefix("</tool_call")
            || trimmed.hasPrefix("<function_call") {
            return trimmed.contains("</tool_call")
                || trimmed.contains("<parameter=")
                || trimmed.range(
                    of: #"^</?(tool_call|function_call)[^>]*>\s*[\{<]"#,
                    options: .regularExpression) != nil
        }
        // Bare malformed-tag fragments (`<function=name>`, `<parameter=x>`):
        // the `=` immediately after the tag name is not valid markup and
        // never occurs in legitimate prose or HTML/XML the user asked for,
        // so the prefix alone is a reliable leak signal.
        if trimmed.hasPrefix("<function=") || trimmed.hasPrefix("<parameter=") {
            return true
        }
        // Documented `<JSON>…</JSON>` raw wrapper: real leak is `<JSON>{…`;
        // a prose answer is `<JSON> is a wrapper…`.
        if trimmed.hasPrefix("<JSON>") {
            let rest = trimmed.dropFirst("<JSON>".count)
                .drop { $0 == " " || $0 == "\n" || $0 == "\t" }
            return rest.first == "{" || rest.first == "["
        }
        // Mistral `[TOOL_CALLS]`: real emit carries an `[ARGS]` delimiter
        // or an inline `[`/`{` array/object; a prose answer is
        // `[TOOL_CALLS] is Mistral's prefix marker…`.
        if trimmed.hasPrefix("[TOOL_CALLS]") {
            let rest = trimmed.dropFirst("[TOOL_CALLS]".count)
                .drop { $0 == " " || $0 == "\n" || $0 == "\t" }
            return rest.hasPrefix("[") || rest.hasPrefix("{")
                || trimmed.contains("[ARGS]")
        }
        return false
    }

    /// True when a `{`-leading string is (or, if truncated, is opening as)
    /// a leaked tool CALL — as opposed to a tool DEFINITION/JSON-Schema, a
    /// data object, or an ordinary record that merely reuses a key name.
    private static func jsonObjectLooksLikeToolCall(_ jsonBody: String) -> Bool {
        // A tool DEFINITION or the args' JSON-Schema carries these keys;
        // their presence means "this describes a tool", not "call it".
        let schemaKeys: Set<String> = ["description", "parameters", "properties", "required"]
        // Keys that NAME the tool, and keys that carry a call's ARGUMENTS.
        // `parameters` is deliberately NOT an args key: it is the schema
        // key of a tool definition, and treating it as args suppresses
        // definitions like `{"name":"get_weather","parameters":{"type":"object"}}`.
        let namesKeys: Set<String> = ["name", "action", "tool", "tool_name"]
        let argsKeys: Set<String> = ["arguments", "action_input", "tool_input"]

        if jsonBody.hasSuffix("}"),
           let data = jsonBody.data(using: .utf8),
           let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
            let keys = Set(obj.keys.map { $0.lowercased() })
            // Canonical OpenAI wire shape — the project's own `ToolCall`:
            // `{"id":"call_1","type":"function","function":{"name":…,"arguments":…}}`.
            // `id`/`type` aren't call vocab and `arguments` is nested, so
            // it is matched structurally. The nested `function` must itself
            // look like a CALL (name + arguments, no schema key) — an
            // OpenAI tool DEFINITION nests
            // `{"name":…,"description":…,"parameters":{…}}` under the same
            // `function` key and must NOT be suppressed.
            if let fn = obj["function"] as? [String: Any] {
                let fnKeys = Set(fn.keys.map { $0.lowercased() })
                if fnKeys.contains("name") && fnKeys.contains("arguments")
                    && fnKeys.isDisjoint(with: schemaKeys) {
                    return true
                }
            }
            // Hermes / ReAct inner shape: names a tool AND carries an args
            // container, with NO definition/schema key and NO key outside
            // the call vocabulary. That excludes a tool DEFINITION and a
            // data record that merely reuses `name`.
            guard keys.isDisjoint(with: schemaKeys) else { return false }
            return !keys.isDisjoint(with: namesKeys)
                && !keys.isDisjoint(with: argsKeys)
                && keys.isSubset(of: namesKeys.union(argsKeys))
        }

        // A malformed / truncated dump that never parses. A genuine JSON
        // answer parses cleanly (handled above); a broken envelope leak
        // does not. `mentionsArgs` is the shared guard that separates a
        // half-emitted CALL (cut off mid-`arguments`) from a truncated
        // DEFINITION (which carries `parameters`, never `arguments`) or a
        // truncated ordinary record.
        let mentionsArgs = argsKeys.contains { jsonBody.contains("\"\($0)\"") }
        guard mentionsArgs else { return false }

        // Truncated canonical OpenAI wire dump — cut off mid-`arguments`,
        // so its FIRST key is `id`, not a call key. `"type":"function"`
        // (a discriminator that essentially never appears in ordinary
        // data) + an args token pins it as a leaked call, not a truncated
        // tool definition (`parameters`, no `arguments`).
        if jsonBody.range(
            of: #"^\{[^}]*"type"\s*:\s*"function""#,
            options: .regularExpression) != nil {
            return true
        }

        // Other malformed dumps that OPEN with a tool-call key AND carry an
        // args token. Requiring the args token keeps a truncated ordinary
        // object like `{"name": "Bob", "age": …` (a person record cut off
        // by max_tokens) from being mistaken for a call.
        return jsonBody.range(
            of: #"^\{\s*"(name|action|tool|tool_name|function)"\s*:"#,
            options: .regularExpression
        ) != nil
    }

    /// Strip a single leading+trailing Markdown code fence when it wraps
    /// the WHOLE content AND its info string is empty or `json` — so a
    /// model that dumped its raw tool call inside a ```` ```json ```` (or
    /// bare ```` ``` ````) block is still recognised. A non-JSON language
    /// tag (```` ```xml ````, ```` ```html ````) or any interior fence is
    /// left untouched, so an example-markup answer is never unwrapped.
    private static func strippedSingleJSONFence(_ s: String) -> String {
        guard s.hasPrefix("```"), s.hasSuffix("```"), s.count > 6,
              let firstNL = s.firstIndex(of: "\n") else { return s }
        let info = s[s.index(s.startIndex, offsetBy: 3)..<firstNL]
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        guard info.isEmpty || info == "json" else { return s }
        let afterOpen = s.index(after: firstNL)
        let closeStart = s.index(s.endIndex, offsetBy: -3)
        guard afterOpen <= closeStart else { return s }
        let inner = String(s[afterOpen..<closeStart])
        // Reject a multi-block answer: any interior fence means this isn't
        // a single top-to-bottom code block.
        guard !inner.contains("```") else { return s }
        return inner.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// True when the assistant's prose body reads as a "raw answer"
    /// — numeric-dominated enough that the model probably skipped
    /// tool-calling and just emitted a guess. Tightened from the
    /// initial draft (codex r1 MAJOR-1, #308 PR): requiring numeric
    /// content rules out the "Paris." false-positive on
    /// ``"What is the capital of France?"`` — a short prose reply
    /// with no digits is a perfectly good answer and must NOT wear
    /// the warning. The gate stays effective against the canonical
    /// issue #308 repro (``43.92504669599178``) because the
    /// hallucination shape is by definition numeric.
    ///
    /// Two satisfying shapes (either fires):
    ///
    ///   * **Short-and-contains-digit**: trimmed length ≤ 80 chars
    ///     AND at least one digit. Catches "15% of 2650 is 397.5;
    ///     sqrt(781) ≈ 27.95; sum is 425.45" (caption acceptable —
    ///     dismissible) while leaving "Paris." / "Yes." / "It depends."
    ///     alone.
    ///   * **Numeric-dominated** (regardless of length): ≥ 40% of
    ///     trimmed chars are digits, dot, or sign. Catches the
    ///     bare-number bug shape (``43.92504669599178``,
    ///     ``425.45``, ``=425``).
    static func contentLooksLikeRawAnswer(_ content: String) -> Bool {
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        let hasDigit = trimmed.unicodeScalars.contains { scalar in
            scalar.value >= 0x30 && scalar.value <= 0x39
        }
        // Short reply with at least one digit — likely a raw answer.
        // The digit requirement rules out the "Paris." /
        // "It depends." false-positive on non-numeric factual
        // prompts (codex r1 MAJOR-1).
        if hasDigit && trimmed.count <= 80 { return true }
        // Numeric-dominated reply (regardless of length).
        guard hasDigit else { return false }
        let numericChars = trimmed.unicodeScalars.filter { scalar in
            (scalar.value >= 0x30 && scalar.value <= 0x39)  // 0..9
                || scalar == "."
                || scalar == "-"
                || scalar == "+"
                || scalar == ","
                || scalar == "="
        }.count
        let ratio = Double(numericChars) / Double(trimmed.unicodeScalars.count)
        return ratio >= 0.4
    }

    /// True when the user's prompt reads as a calculator-, web-
    /// search-, or weather-style query — i.e. the kind of question
    /// where a tool-call SHOULD have been the right shape. The
    /// match is keyword-based and intentionally inclusive; the
    /// caption is dismissible and a false-positive is harmless.
    ///
    /// Heuristics:
    ///   * Math operators (``+``, ``-``, ``*``, ``/``, ``%``, ``=``,
    ///     ``^``) AND at least one digit — catches arithmetic
    ///     prompts.
    ///   * Number words / math keywords (``square root``, ``sqrt``,
    ///     ``percent``, ``calculate``, ``compute``, ``solve``,
    ///     ``divide``, ``multiply``, ``sum``, ``product``).
    ///   * Web-search shaped keywords (``search``, ``look up``,
    ///     ``what is``, ``who is``, ``where is``, ``latest``,
    ///     ``news``, ``today``, ``current``).
    ///   * Weather-shaped keywords (``weather``, ``temperature``,
    ///     ``forecast``).
    static func promptLooksCalculatorish(_ prompt: String) -> Bool {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        let lowered = trimmed.lowercased()

        // Math: operator + digit.
        let hasDigit = lowered.unicodeScalars.contains { scalar in
            scalar.value >= 0x30 && scalar.value <= 0x39
        }
        let mathOperators: Set<Character> = ["+", "-", "*", "/", "%", "=", "^"]
        let hasOperator = lowered.contains(where: { mathOperators.contains($0) })
        if hasDigit && hasOperator { return true }

        // Math keywords.
        let mathKeywords: [String] = [
            "square root", "sqrt", "percent", "calculate", "compute", "solve",
            "divide", "multiply", "sum of", "product of", "plus", "minus",
            "times", "divided by"
        ]
        for kw in mathKeywords where lowered.contains(kw) { return true }

        // Web-search keywords. Note: codex r1 MAJOR-1 (#308 PR)
        // dropped the bare ``"what is the"`` keyword — it matched
        // every plain factual question ("What is the capital of
        // France?") and false-flagged short prose answers like
        // "Paris." Web-search keywords must point at LIVE / DATED
        // information; an evergreen factual lookup is not the
        // failure mode this caption guards against.
        let webKeywords: [String] = [
            "search for", "google for", "look up", "look it up",
            "latest news", "latest version", "news about",
            "today's", "this week's", "right now",
            "current price", "stock price", "exchange rate",
            "current weather"
        ]
        for kw in webKeywords where lowered.contains(kw) { return true }

        // Weather keywords (looser than the web list above).
        let weatherKeywords: [String] = ["weather", "temperature", "forecast"]
        for kw in weatherKeywords where lowered.contains(kw) { return true }

        return false
    }

    /// Visible copy for the issue #308 tool-not-called caption.
    /// Lifted out so the test can snapshot the exact string an
    /// accidental reword would change. Kept short — the caption
    /// renders above the assistant bubble in ``MessageRow``.
    static let toolNotCalledCaptionCopy: String =
        "This model didn't call a tool — verify the answer."

    /// VoiceOver caption for the issue #308 tool-not-called caption.
    /// Reads the same information as a complete sentence. Pinned
    /// alongside the visible copy so an accessibility regression
    /// surfaces in the tests.
    static let toolNotCalledCaptionAccessibilityLabel: String =
        "Caution: this model answered without calling any of the available tools. The answer may be a guess. Verify before relying on it."
}

/// End-of-stream summary stats for an assistant turn. Carried on
/// the message itself so the rendered "~84 tok/s · 2.4 s" caption
/// survives session reload — recomputing from ``createdAt`` would
/// be wrong after a reload (the cold-start latency leaks in).
///
/// ``charCount`` is the only field we can reliably populate today:
/// neither rapid-mlx nor the OpenAI streaming spec emits per-chunk
/// usage by default, and ``stream_options.include_usage`` is on
/// the v0.4.13 backlog. We use char count as a proxy ("~" prefix
/// in the UI signals estimate, not authoritative). When the
/// include_usage wiring lands, ``promptTokens`` / ``completionTokens``
/// will become populated and the UI will drop the tilde.
struct MessageStats: Codable, Equatable, Hashable {
    /// Wall-clock seconds from the assistant placeholder being
    /// inserted to the [DONE] event arriving. Doesn't include
    /// cold-start (the model was already loaded by then).
    var elapsedSeconds: Double
    /// Characters in ``content`` at end-of-stream. Estimating
    /// tokens at ~4 chars/token gets us roughly within 15% on
    /// English; non-English / code is noisier but still useful.
    var charCount: Int
    /// Server-reported prompt token count from the final
    /// stream chunk's ``usage`` block. ``nil`` until v0.4.13
    /// wires ``stream_options.include_usage``.
    var promptTokens: Int?
    /// Server-reported completion token count. ``nil`` until
    /// v0.4.13.
    var completionTokens: Int?
    /// Seconds from dispatching the request to the FIRST content token
    /// arriving — the prompt-processing (prefill) half of the turn.
    ///
    /// ``nil`` on transcripts persisted before this field existed and on
    /// turns that produced no content, in which case every rate below
    /// degrades to the pre-existing whole-turn arithmetic.
    var timeToFirstTokenSeconds: Double?
    /// Did this turn emit a reasoning trace?
    ///
    /// Only the char-count estimate needs it, to know that its numerator
    /// (visible prose) and its denominator (a window that opened on the
    /// first reasoning token) are not describing the same generation.
    /// Optional for the same reason as ``timeToFirstTokenSeconds``: older
    /// transcripts have no such field and decode as ``nil``.
    var reasoningEmitted: Bool?

    /// ``reasoningEmitted`` with the legacy default. A transcript written
    /// before the field existed is treated as prose-only, which is what
    /// those rows were captioned as at the time.
    var emittedReasoning: Bool { reasoningEmitted ?? false }

    /// Seconds spent generating — everything after the first token landed.
    ///
    /// This, not ``elapsedSeconds``, is the denominator for a throughput
    /// number. Dividing by the whole turn folds prefill into "tok/s" and
    /// makes the answer's LENGTH the dominant term: the desktop advertised
    /// ~61 tok/s for `qwen3.5-4b-4bit` while the same model, in the same
    /// second, captioned a short reply at 13 tok/s — prefill of a
    /// ~950-token tool-carrying prompt was 93 % of that turn. Same
    /// machine, same model, two numbers 5x apart, both labelled "tok/s".
    /// The recorded TTFT, but only when it describes an interval that can
    /// exist: inside the turn it belongs to, and not negative.
    ///
    /// A persisted transcript can carry nonsense here — a hand-edited
    /// session file, or a row written by a build that measured this with a
    /// wall clock rather than the monotonic one used now — and a value at
    /// or past ``elapsedSeconds`` is not a prefill measurement.
    /// One accessor so arithmetic and presentation can never disagree
    /// about which values are real: rejecting it for the rate while still
    /// rendering "1.2 s to first token · 1.0 s" would just move the lie
    /// into the caption.
    var validTimeToFirstToken: Double? {
        guard let timeToFirstTokenSeconds,
              timeToFirstTokenSeconds >= 0,
              timeToFirstTokenSeconds < elapsedSeconds else { return nil }
        return timeToFirstTokenSeconds
    }

    /// Whether this row came from a build that measures prefill at all.
    ///
    /// The field's PRESENCE is the discriminator, not its validity, and the
    /// distinction decides what an unusable TTFT falls back to. A transcript
    /// persisted before ``timeToFirstTokenSeconds`` existed decodes as `nil`
    /// here and is legitimately captioned with the whole-turn arithmetic it
    /// was first shown with — there is no better measurement to be had, and
    /// re-rendering history is its own kind of wrong.
    ///
    /// A row that carries a value the guards then REJECT is a different
    /// animal: a corrupt measurement from a build that should have produced
    /// a good one. Falling back to the whole turn *there* would reintroduce
    /// the plausible-but-wrong number this change exists to remove, on
    /// precisely the rows already known to be untrustworthy, and it would be
    /// indistinguishable from a real reading. Those get no rate at all.
    var measuresPrefill: Bool { timeToFirstTokenSeconds != nil }

    var decodeSeconds: Double? {
        guard let ttft = validTimeToFirstToken else {
            return measuresPrefill ? nil : elapsedSeconds
        }
        let decode = elapsedSeconds - ttft
        return decode > 0 ? decode : nil
    }

    /// Heuristic tokens/sec from char count, for a server that reports no
    /// usage at all. UI prefixes with "~" to signal estimate.
    ///
    /// Gated on ``completionTokens == nil`` so it is a fallback for
    /// *missing* data and never a second opinion that overrides a
    /// deliberate ``nil`` from ``reportedTokensPerSecond``. Without that
    /// gate a one-token reply — which the reported path declines to rate
    /// because there is no interval to divide by — fell straight through
    /// to this estimate and printed "~1 tok/s" anyway.
    ///
    /// Suppressed on a turn that emitted reasoning, because the two halves
    /// of the fraction would then measure different things: the decode
    /// window opens at the first token on ANY channel (a reasoning model's
    /// trace comes first), while ``charCount`` counts only the visible
    /// prose. A long think followed by one short sentence would divide a
    /// handful of characters by the whole reasoning window and understate
    /// the rate by however long the model thought — reintroducing, in the
    /// fallback path, the exact distortion this change removed from the
    /// main one. No number beats a wrong one; the caption still carries
    /// time-to-first-token and the total.
    /// The estimate carries the same inverse-TPOT shape as the reported
    /// path, and for the same reason: when there IS a measured prefill, the
    /// first token is what ended it, so it was not produced inside
    /// ``decodeSeconds`` and does not belong in the numerator. Leaving the
    /// subtraction out here would make a four-character reply — one
    /// estimated token, zero token intervals — divide by a window it never
    /// occupied and print a rate, which is exactly what the reported path
    /// declines to do one accessor below.
    var estimatedTokensPerSecond: Double? {
        guard completionTokens == nil, !emittedReasoning else { return nil }
        guard let decodeSeconds, decodeSeconds > 0.05 else { return nil }  // < 50 ms is noise
        let estTokens = Double(charCount) / 4.0
        guard validTimeToFirstToken != nil else {
            return estTokens / decodeSeconds
        }
        guard estTokens > 1 else { return nil }
        return (estTokens - 1) / decodeSeconds
    }

    /// Authoritative tokens/sec when the server reported usage.
    ///
    /// With a TTFT, ``completionTokens - 1`` intervals span the decode
    /// window: the first token is what ENDS prefill, so it is not produced
    /// during ``decodeSeconds``. That is the standard inverse-TPOT
    /// definition, and it needs at least two tokens to mean anything — a
    /// one-token reply returns ``nil`` and the caption omits a rate rather
    /// than dividing by a window the token never occupied.
    ///
    /// **Without** a TTFT the subtraction has nothing to stand on: the
    /// denominator is the whole turn, which begins before the first token,
    /// so all N tokens fall inside it. Transcripts written before this
    /// field existed therefore keep the exact arithmetic they were
    /// captioned with — `N / elapsed` — rather than silently shifting to
    /// `(N - 1) / elapsed` and re-rendering history slightly differently
    /// than it was first shown. That branch is reachable *only* for those
    /// legacy rows: see ``measuresPrefill``, which withholds
    /// ``decodeSeconds`` entirely from a modern row whose stamp is corrupt,
    /// so this accessor returns `nil` there instead of falling through to
    /// the whole-turn number.
    var reportedTokensPerSecond: Double? {
        guard let completionTokens, completionTokens > 0,
              let decodeSeconds, decodeSeconds > 0.05 else { return nil }
        guard validTimeToFirstToken != nil else {
            return Double(completionTokens) / decodeSeconds
        }
        guard completionTokens > 1 else { return nil }
        return Double(completionTokens - 1) / decodeSeconds
    }
}

extension Duration {
    /// Whole plus fractional seconds, for storing an interval in the
    /// `Double` fields of ``MessageStats``.
    ///
    /// The durations here come from ``ContinuousClock`` rather than
    /// differences of `Date`, because both numbers this feeds are the
    /// *difference* between two readings taken seconds apart. A wall clock
    /// can be stepped by NTP or by the user between those two reads, and a
    /// step of a few hundred milliseconds is enough to turn a real decode
    /// window into a nonsensical one while still landing inside every
    /// range guard — a silently wrong rate, which is the failure mode this
    /// whole change set exists to eliminate. A monotonic clock cannot be
    /// stepped, so that class of corruption stops being possible rather
    /// than being detected after the fact.
    var seconds: Double {
        let parts = components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }
}
