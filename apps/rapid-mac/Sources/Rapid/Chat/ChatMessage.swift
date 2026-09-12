import Foundation
import ImageIO
import UniformTypeIdentifiers

struct ChatImageAttachment: Codable, Equatable, Hashable, Identifiable, Sendable {
    static let maxBytes = 20 * 1024 * 1024
    /// Per-message image budget, mirroring the document attachment budget
    /// (``ChatFileAttachment/maxAttachmentsPerMessage`` and friends). The
    /// per-file 20 MB cap is not enough on its own: a multi-select or drop can
    /// present dozens of individually valid images, and decoding every one and
    /// base64-encoding the whole set into a single request would freeze or
    /// exhaust the Desktop process. `maxImagesPerMessage` bounds the count and
    /// ``maxCombinedEncodedImageBytes`` bounds the aggregate **encoded** bytes.
    static let maxImagesPerMessage = 4
    /// Combined budget for the images in one message, measured in exact
    /// encoded `data:<mime>;base64,…` byte counts, not raw `Data.count`.
    /// Each image travels over the wire as a base64 data URL, so raw bytes
    /// under-count the request by ~4/3 and can deterministically blow past the
    /// engine's 8 MiB request-body cap with a 413. This budget stays below
    /// that cap while leaving dedicated room for text, history, tools, and
    /// JSON framing. **Value under review: Pixel (product).**
    static let maxCombinedEncodedImageBytes = 6 * 1024 * 1024

    /// A human-readable form of ``maxCombinedEncodedImageBytes`` for notices.
    static var formattedCombinedImageBudget: String {
        let bytes = maxCombinedEncodedImageBytes
        let mb = Double(bytes) / Double(1024 * 1024)
        if mb == mb.rounded() { return "\(Int(mb)) MB" }
        return String(format: "%.1f MB", mb)
    }

    /// User-facing explanation for attachments rejected by the per-message
    /// image budget. Keeping this copy beside the limits lets both the pre-read
    /// picker gate and the authoritative post-normalization gate report the
    /// same count and binding reason.
    static func budgetNotice(
        rejectedCount: Int = 0,
        limit: ImageBudgetLimit = .count
    ) -> String {
        let budget = formattedCombinedImageBudget
        let base = "Attach up to \(maxImagesPerMessage) images or \(budget) of images per message."
        guard rejectedCount > 0 else { return base }
        let reason: String = switch limit {
        case .count: "too many images"
        case .bytes: "their combined size exceeds the \(budget) budget"
        }
        let plural = rejectedCount == 1 ? "image was" : "images were"
        return "\(base) \(rejectedCount) \(plural) not added — \(reason)."
    }

    /// Why ``importCandidates(_:existingCount:existingBytes:)`` dropped a
    /// candidate. Kept distinct so a notice can tell the user whether the
    /// count or the combined-byte budget was the binding limit.
    enum ImageBudgetLimit {
        case count
        case bytes
    }

    /// Exact number of bytes a `data:<mime>;base64,<payload>` URL occupies for
    /// `rawBytes` of payload, without materialising the base64 string.
    /// `"data:"` + `<mime>` + `";base64,"` + base64(data); base64 of `n` bytes
    /// is `4 * ceil(n / 3)` and the integer `(n + 2) / 3 * 4` avoids floats.
    static func encodedDataURLByteCount(mimeType: String, rawBytes: Int) -> Int {
        let base64 = ((rawBytes + 2) / 3) * 4
        return 5 + mimeType.utf8.count + 8 + base64
    }
    /// Keep image preprocessing below the embedded engine's per-request
    /// vision-token budget. A 2048 × 1536 image is roughly 4K vision tokens
    /// on the recommended model, leaving useful margin below its 8K cap.
    static let maxVisionLongEdge = 2_048
    /// Bounds decoded memory before ImageIO creates a full bitmap. This still
    /// admits 48 MP iPhone captures while rejecting compressed image bombs.
    static let maxPixelCount = 64_000_000
    static let maxPixelDimension = 16_384

    static func dimensionsFit(width: Int, height: Int) -> Bool {
        width > 0
            && height > 0
            && width <= maxPixelDimension
            && height <= maxPixelDimension
            && width <= maxPixelCount / height
    }

    static func normalizedPixelSize(width: Int, height: Int) -> (width: Int, height: Int) {
        let longEdge = max(width, height)
        guard longEdge > maxVisionLongEdge else { return (width, height) }
        let scale = Double(maxVisionLongEdge) / Double(longEdge)
        return (
            max(1, Int((Double(width) * scale).rounded())),
            max(1, Int((Double(height) * scale).rounded()))
        )
    }

    /// Bound work before reading any selected image. The per-file 20 MB cap
    /// is enforced at read time; this pre-read gate stops the count and the
    /// aggregate bytes from growing unbounded across a multi-select/drop, which
    /// would otherwise decode and retain every image and base64 them all into a
    /// single request. Accepted order matches the selection; each candidate's
    /// on-disk file size is the pre-read estimate for the aggregate byte gate,
    /// charged at the encoded data-URL rate. The exact post-decode gate
    /// (``ChatImageAttachment/fittedForMessage(_:)``) is authoritative and is
    /// also applied at the wire, so a small pre-read skew cannot admit an
    /// over-budget request.
    static func importCandidates(
        _ urls: [URL],
        existingCount: Int,
        existingBytes: Int
    ) -> (accepted: [URL], rejectedCount: Int, limit: ImageBudgetLimit) {
        var accepted: [URL] = []
        var remainingCount = max(0, maxImagesPerMessage - max(0, existingCount))
        var remainingBytes = max(0, maxCombinedEncodedImageBytes - max(0, existingBytes))
        var limit: ImageBudgetLimit = .count
        for url in urls {
            guard remainingCount > 0 else { break }
            let size = (
                try? url.resourceValues(forKeys: [.fileSizeKey])
            )?.fileSize ?? 0
            // MIME is unknown before read; its prefix differs by one byte, which
            // is immaterial at this pre-read stage.
            let estimatedEncoded = encodedDataURLByteCount(mimeType: "image/jpeg", rawBytes: size)
            guard estimatedEncoded <= remainingBytes else {
                limit = .bytes
                continue
            }
            accepted.append(url)
            remainingCount -= 1
            remainingBytes -= estimatedEncoded
        }
        return (accepted, max(0, urls.count - accepted.count), limit)
    }

    /// Returns the ordered subset that fits the per-message image budget (count
    /// and combined bytes). Images cannot be truncated the way document text is
    /// (``ChatFileAttachment/fittedForMessage(_:)``), so an image that does not
    /// fit is dropped rather than reduced.
    static func fittedForMessage(
        _ attachments: [ChatImageAttachment]
    ) -> [ChatImageAttachment] {
        var result: [ChatImageAttachment] = []
        var remainingBytes = maxCombinedEncodedImageBytes
        for attachment in attachments {
            guard result.count < maxImagesPerMessage,
                  attachment.encodedDataURLByteCount <= remainingBytes else { continue }
            result.append(attachment)
            remainingBytes -= attachment.encodedDataURLByteCount
        }
        return result
    }

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
        guard values.contentType?.conforms(to: .image) == true,
              let source = CGImageSourceCreateWithURL(url as CFURL, nil)
        else { throw ValidationError.unsupportedType }
        guard CGImageSourceGetCount(source) == 1 else { throw ValidationError.animatedGIF }
        let sourceProperties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
            as? [CFString: Any] ?? [:]
        guard let width = (sourceProperties[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
              let height = (sourceProperties[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue,
              Self.dimensionsFit(width: width, height: height)
        else { throw ValidationError.tooManyPixels }

        let type = values.contentType
        let fitsVisionBudget = max(width, height) <= Self.maxVisionLongEdge
        if type?.conforms(to: .png) == true, fitsVisionBudget {
            try self.init(
                filename: url.lastPathComponent,
                mimeType: "image/png",
                data: Data(contentsOf: url)
            )
        } else if type?.conforms(to: .jpeg) == true, fitsVisionBudget {
            try self.init(
                filename: url.lastPathComponent,
                mimeType: "image/jpeg",
                data: Data(contentsOf: url)
            )
        } else if type?.conforms(to: .gif) == true, fitsVisionBudget {
            try self.init(
                filename: url.lastPathComponent,
                mimeType: "image/gif",
                data: Data(contentsOf: url)
            )
        } else {
            let normalized = try Self.normalizedStaticImage(
                source: source,
                sourceProperties: sourceProperties,
                at: url
            )
            try self.init(
                filename: normalized.filename,
                mimeType: normalized.mimeType,
                data: normalized.data
            )
        }
    }

    /// Normalize native still-image formats at the attachment boundary so the
    /// persisted attachment, preview, and wire request all share one truthful
    /// MIME/byte contract. Small PNG/JPEG/GIF files stay byte-for-byte
    /// unchanged above. Larger or native still images become JPEG, or PNG when
    /// alpha must be preserved. The ordinary initializer applies the same 20 MB
    /// wire cap to the normalized result.
    private static func normalizedStaticImage(
        source: CGImageSource,
        sourceProperties: [CFString: Any],
        at url: URL
    ) throws -> (filename: String, mimeType: String, data: Data) {
        let thumbnailOptions: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxVisionLongEdge,
            kCGImageSourceShouldCacheImmediately: true,
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(
            source,
            0,
            thumbnailOptions as CFDictionary
        ) else {
            throw ValidationError.unsupportedType
        }

        let preservesAlpha: Bool
        if let sourceHasAlpha = sourceProperties[kCGImagePropertyHasAlpha] as? NSNumber {
            preservesAlpha = sourceHasAlpha.boolValue
        } else {
            preservesAlpha = Self.containsTransparentPixel(image)
        }
        let targetType: UTType = preservesAlpha ? .png : .jpeg
        let mimeType = preservesAlpha ? "image/png" : "image/jpeg"
        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            output,
            targetType.identifier as CFString,
            1,
            nil
        ) else { throw ValidationError.unsupportedType }

        var properties = sourceProperties
        // The thumbnail transform has already applied EXIF orientation.
        properties[kCGImagePropertyOrientation] = 1
        if !preservesAlpha {
            properties[kCGImageDestinationLossyCompressionQuality] = 0.9
        }
        CGImageDestinationAddImage(destination, image, properties as CFDictionary)
        guard CGImageDestinationFinalize(destination) else {
            throw ValidationError.unsupportedType
        }

        let base = url.deletingPathExtension().lastPathComponent
        let suffix = preservesAlpha ? "png" : "jpg"
        return ("\(base).\(suffix)", mimeType, output as Data)
    }

    /// Thumbnail backing stores may carry an alpha channel for an opaque
    /// source. Inspect the already-bounded thumbnail instead of treating the
    /// channel's mere presence as transparency.
    private static func containsTransparentPixel(_ image: CGImage) -> Bool {
        switch image.alphaInfo {
        case .none, .noneSkipFirst, .noneSkipLast: return false
        default: break
        }
        let bytesPerRow = image.width * 4
        var pixels = [UInt8](repeating: 255, count: bytesPerRow * image.height)
        guard let context = CGContext(
            data: &pixels,
            width: image.width,
            height: image.height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return true }
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return stride(from: 3, to: pixels.count, by: 4).contains { pixels[$0] < 255 }
    }

    var dataURL: String { "data:\(mimeType);base64,\(data.base64EncodedString())" }

    /// Exact number of bytes this attachment's `dataURL` occupies on the wire,
    /// without materialising `dataURL` just to measure it.
    var encodedDataURLByteCount: Int {
        Self.encodedDataURLByteCount(mimeType: mimeType, rawBytes: data.count)
    }

    enum ValidationError: LocalizedError {
        case tooLarge, tooManyPixels, unsupportedType, animatedGIF
        var errorDescription: String? {
            switch self {
            case .tooLarge: return "Images must be 20 MB or smaller."
            case .tooManyPixels: return "Images must be 64 megapixels or smaller."
            case .unsupportedType: return "Choose a supported still image."
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

    /// Whether a locally stored row belongs in the model-facing transcript.
    /// UI-authored affordances such as the onboarding welcome remain visible
    /// and persisted, but are not synthetic assistant turns on the wire.
    enum WireVisibility: String, Codable, Sendable {
        case model
        case transcriptOnly
    }

    /// Whether an image-bearing user turn was accepted by the model lane.
    /// Stored on the user turn because empty failed assistant rows are omitted
    /// from later wire history, while a rejected image must not poison every
    /// subsequent plain-text follow-up. A direct Retry still sends the image.
    enum ImageDeliveryStatus: String, Codable, Sendable {
        case pending
        /// One pre-token failure has already been allowed to retry through a
        /// later plain-text turn. Persisting this state on the image-bearing
        /// message keeps the retry budget tied to that exact turn.
        case retryable
        case accepted
        case rejected

        var permitsFollowUpInheritance: Bool {
            self == .retryable || self == .accepted
        }

        /// A newer outcome must fail closed for attachment inheritance rather
        /// than make one message undecodable and side the whole conversation
        /// as corrupt. The image remains visible and directly retryable.
        init(from decoder: Decoder) throws {
            let raw = try decoder.singleValueContainer().decode(String.self)
            self = ImageDeliveryStatus(rawValue: raw) ?? .rejected
        }
    }

    /// Messages written before ``wireVisibility`` shipped need one narrow
    /// migration: Quickstart's product-authored welcome was persisted as an
    /// ordinary assistant turn. Match only the two stable Rapid copy shapes;
    /// this is history-schema migration, not classification of user text.
    private static func isLegacyQuickstartWelcome(_ content: String) -> Bool {
        guard content.hasPrefix("You're chatting with ") else { return false }
        return content.hasSuffix(
            ", running entirely on your Mac. Open the picker any time to switch models."
        ) || (
            content.contains(" — a model picked so you can start chatting in about a minute. ")
                && content.hasSuffix("great first upgrade when you want more.")
        )
    }

    let id: UUID
    let role: Role
    /// The visible assistant prose / user prompt body. For Qwen3.5/3.6
    /// hybrid-thinking responses this excludes any ``reasoning_content``
    /// the model produced — that goes into ``reasoning`` so the UI can
    /// render it in a collapsed disclosure block.
    var content: String
    var imageAttachments: [ChatImageAttachment]
    var imageDeliveryStatus: ImageDeliveryStatus?
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
    var wireVisibility: WireVisibility
    /// Parent node in the conversation tree, or ``nil`` for a root turn.
    ///
    /// A conversation stores EVERY branch, not just the visible transcript;
    /// the path on screen is derived by walking up from
    /// ``ChatConversation.activeLeafID`` through this link (see
    /// ``ChatConversation/activePath``). Two messages sharing a ``parentID``
    /// are siblings — alternative continuations of the same point — which is
    /// what Regenerate / Retry / editing a prompt now produce instead of
    /// truncating the tail away.
    ///
    /// Siblings are deliberately NOT mirrored in a `childrenIDs` array on the
    /// parent: one authoritative edge per node cannot drift out of sync,
    /// whereas a bidirectional pair has to be repaired on every insert,
    /// delete, and decode. Sibling order is derived from ``createdAt``.
    ///
    /// Optional, and absent from every conversation written before branching
    /// shipped. ``ChatConversation``'s decode reconnects such a legacy linear
    /// array into a degenerate tree (each row parented to the one before it),
    /// so old transcripts render exactly as they always did.
    var parentID: UUID?
    let createdAt: Date

    /// Wire-only trailer appended after this row's prose AND its attachment
    /// extracts — see ``modelContent``.
    ///
    /// Deliberately absent from ``CodingKeys``: it is set on the throwaway
    /// array ``ChatViewModel`` builds for one request, never on the
    /// transcript, so it is not part of the conversation and must not reach
    /// `conversations.json`. ``init(from:)`` therefore restores it as `nil`.
    ///
    /// The one producer is ``ChatViewModel/stampingClockContext(on:calendar:)``,
    /// which needs each user turn's wall clock to land after that turn's
    /// attachment extracts. Writing it into ``content`` instead would put it
    /// in FRONT of the extract, so the first request carrying a document and
    /// the next one would diverge before the document rather than after it,
    /// and the engine's prefix cache could not reuse the document.
    var wireSuffix: String?

    init(
        id: UUID = UUID(),
        role: Role,
        content: String = "",
        imageAttachments: [ChatImageAttachment] = [],
        imageDeliveryStatus: ImageDeliveryStatus? = nil,
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
        wireVisibility: WireVisibility = .model,
        parentID: UUID? = nil,
        createdAt: Date = Date(),
        wireSuffix: String? = nil
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.imageAttachments = imageAttachments
        self.imageDeliveryStatus = imageDeliveryStatus
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
        self.wireVisibility = wireVisibility
        self.parentID = parentID
        self.createdAt = createdAt
        self.wireSuffix = wireSuffix
    }

    /// Codex r1 MAJOR-1: keep ``reasoningTruncated`` decodable from
    /// pre-cycle-2 session envelopes (the on-disk JSON has no such
    /// key). Swift's synthesised init(from:) throws on a missing
    /// non-optional, so the custom init below falls back to ``false``
    /// for that one key and defers all other fields to the standard
    /// container shape.
    enum CodingKeys: String, CodingKey {
        case id, role, content, imageAttachments, imageDeliveryStatus
        case fileAttachments, reasoning, status
        case errorMessage, failureKind, toolCalls, toolCallID
        case stats, reasoningTruncated, contentTruncated
        case toolNotCalledFlagged
        case toolCallArtifactSuppressed
        case wireVisibility
        case parentID
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
        try c.encodeIfPresent(imageDeliveryStatus, forKey: .imageDeliveryStatus)
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
        try c.encode(wireVisibility, forKey: .wireVisibility)
        // Omitted for root turns. Non-root rows always carry it — including
        // in conversations that never branched — which is safe because it is
        // an additive key every shipped decoder ignores; the conversation-
        // level schema marker is the `branches` key, not this one.
        try c.encodeIfPresent(parentID, forKey: .parentID)
        try c.encode(createdAt, forKey: .createdAt)
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try c.decode(UUID.self, forKey: .id)
        self.role = try c.decode(Role.self, forKey: .role)
        self.content = try c.decode(String.self, forKey: .content)
        self.imageAttachments = try c.decodeIfPresent([ChatImageAttachment].self, forKey: .imageAttachments) ?? []
        self.imageDeliveryStatus = try c.decodeIfPresent(
            ImageDeliveryStatus.self,
            forKey: .imageDeliveryStatus
        )
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
        if let storedVisibility = try c.decodeIfPresent(WireVisibility.self, forKey: .wireVisibility) {
            self.wireVisibility = storedVisibility
        } else {
            self.wireVisibility = role == .assistant && Self.isLegacyQuickstartWelcome(content)
                ? .transcriptOnly
                : .model
        }
        // Absent on every row written before branching shipped, and absent on
        // root turns thereafter. ``ChatConversation``'s decode rebuilds the
        // parent chain for a legacy linear array, so ``nil`` here is not
        // assumed to mean "root" until that repair has run.
        self.parentID = try c.decodeIfPresent(UUID.self, forKey: .parentID)
        self.createdAt = try c.decode(Date.self, forKey: .createdAt)
        // Transient by construction — a persisted turn carries no wire
        // trailer, and the next request mints a fresh one.
        self.wireSuffix = nil
    }

    /// Text sent to the model. Document extracts stay out of the visible
    /// ``content`` property but remain part of this turn on every retry and
    /// follow-up request.
    /// ``wireSuffix`` is joined LAST, after the attachment extracts, so a
    /// per-request trailer cannot displace the document text that the
    /// engine's prefix cache needs to find unchanged.
    var modelContent: String {
        let trailer = wireSuffix.flatMap {
            $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : [$0]
        } ?? []
        guard !fileAttachments.isEmpty else {
            guard !trailer.isEmpty else { return content }
            // An empty prose row must not gain a leading blank line.
            let head = content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? [] : [content]
            return (head + trailer).joined(separator: "\n\n")
        }
        let request = content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? "Analyze the attached file and summarize the important findings."
            : content
        return ([request] + fileAttachments.map(\.promptText) + trailer)
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
    ///   * ``promptHadAttachment == false``, OR the prompt is not one
    ///     an attached document could answer
    ///     (``promptIsAttachmentAnswerable``). An answer read off an
    ///     attached file is grounded, not guessed — but a page cannot
    ///     ground "what is today's stock price", has nothing to do
    ///     with "calculate 17*23", and is not what "search for Ada
    ///     Lovelace's biography" asks for.
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
    /// The heuristic leans towards firing — a false-negative (a
    /// silent wrong answer) is the bug we're fixing, and the view
    /// layer makes the caption dismissible (one-shot per session) so
    /// a user who knows better can mute it. But false positives are
    /// NOT free, which the original #308 note understated: the
    /// caption's whole value is that the user believes it, and each
    /// time it appears under a demonstrably correct answer it teaches
    /// them to ignore the next one. That is why the prompt heuristic
    /// matches whole words (see ``promptLooksCalculatorish``) and why
    /// a document-grounded turn is exempt (Gate 1c) instead of
    /// relying on the user to dismiss it — and equally why that
    /// exemption is *narrow*: an attachment on the turn does not make
    /// a live-data question answerable from the page.
    ///
    /// Role-agnostic (assertion-only check); the view layer enforces
    /// "only paint on assistant rows".
    static func shouldFlagToolNotCalled(
        userPrompt: String,
        assistantContent: String,
        toolCalls: [ToolCall]?,
        finishReason: String?,
        toolsRequested: Bool,
        toolSucceededThisTurn: Bool = false,
        promptHadAttachment: Bool = false
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
        // Gate 1c: the user's turn carried a document AND the question
        // is one that document could answer. When it is, the answer is
        // grounded in text the user supplied in the prompt, so
        // "answered without calling any of the available tools" is not
        // a caution — it is the correct behaviour, and no tool on the
        // roster could have improved it. Dogfooding 0.14.1 hit exactly
        // this: a scanned-invoice turn whose grounded, correct total
        // wore the caption, which reads as "this number may be a
        // guess" directly under a number the model had in fact read
        // off the page. Flagging a right answer is not a harmless
        // false positive — it spends the user's trust in the caption,
        // so the next one (a real hallucinated total) gets ignored too.
        //
        // But the exemption has to be narrow, because an attachment is
        // not a general licence — three review rounds each found a
        // prompt that carried a document and still could not be
        // answered from it. So the test is stated positively, as the
        // one shape a page CAN answer: math vocabulary whose operands
        // live on that page. See ``promptIsAttachmentAnswerable`` for
        // the table of what that excludes and why.
        if promptHadAttachment, promptIsAttachmentAnswerable(userPrompt) { return false }
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
        // Gate 4b: a turn that answered in prose and THEN emitted an
        // envelope. The 0.14.1 dogfood repro: 5,413 characters of
        // "Let me read the first page more carefully…" followed by
        // `<tool_call> {"name":"read_document","arguments":{…,"greP":…`,
        // which no parser claimed — so no tool round fired, the model kept
        // generating for six more minutes, and the user watched a sentence
        // that promised an action be followed by nothing at all. The
        // leading-only check below cannot see it, because the artifact is
        // the TAIL of an otherwise real answer.
        if trailingToolCallArtifactProse(in: content) != nil { return true }
        // Not a gate, but the question every reviewer asks here: what about a
        // COMPLETE, well-formed, unfenced example that legitimately ends an
        // answer? Two things cover it. Gates 1-3 are themselves the
        // "parser-rejected" evidence — tools were advertised and the turn came
        // back with no tool call at all, so an envelope the engine's parser
        // could read would have been dispatched and never reached this line.
        // And #513 documents the remainder as an accepted residual: a turn
        // that IS only a raw call shape cannot be told apart from a leak by
        // content, suppression is non-destructive (the raw text stays on the
        // message; copy and export reproduce it verbatim, and since this PR
        // the prose above it renders), and in the target population — a local
        // model whose call the parser lost — a leak is far likelier than a
        // deliberately-requested example. A fenced example, which is how a
        // model actually answers "show me one", is never touched.
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

    /// The prose an assistant turn actually said, when its content is real
    /// text followed by a malformed tool-call envelope — or nil when there
    /// is no such tail.
    ///
    /// Companion to ``contentLooksLikeToolCallArtifact``, which only fires
    /// when the artifact IS the whole turn. Here the answer is genuine and
    /// only its tail is machine syntax, so the prose is kept and the tail is
    /// replaced by ``toolCallArtifactSuppressedCaptionCopy``.
    ///
    /// Conservative in the same three ways as the leading check:
    ///   * The marker must OPEN A LINE and be followed by that format's
    ///     payload — the ``leadingEnvelopeLeak`` gate, applied to the tail,
    ///     plus a block anchor — so an answer that explains `<tool_call>` in
    ///     a sentence is left alone whether or not it fences the example.
    ///     (The DeepSeek U+2581 token is exempt from the anchor: it never
    ///     appears in prose, so there is no inline shape to protect.)
    ///   * A marker inside a fenced code block is never a leak. An answer to
    ///     "show me what a tool call looks like" puts its example in a fence,
    ///     and that fence is the whole point of asking. Fences are parsed
    ///     (``fencedRanges``), not counted: backticks and tildes, three or
    ///     more, closer matching the opener. A fenced marker is SKIPPED, not
    ///     a verdict: the scan carries on to the next candidate, so an answer
    ///     that shows a fenced example and then trails off into a real
    ///     envelope is still caught.
    ///   * There must be real prose before it. With none, the leading check
    ///     owns the turn and this returns nil, so the caption-only render
    ///     stays exactly as it was.
    ///   * The envelope must be the turn's TAIL — inside the terminal run of
    ///     machine syntax (``terminalMachineSyntaxRunStart``). An answer that
    ///     shows a raw call and then explains it keeps its explanation.
    static func trailingToolCallArtifactProse(in content: String) -> String? {
        // Candidate openers for the machine-syntax tail.
        //
        // Strict on purpose: each pattern requires that format's payload to
        // follow the marker (whitespace only in between). The leading check
        // can afford its looser "carries a closing tag somewhere" fallback,
        // because it has already established that the envelope IS the whole
        // turn; a tail cannot. An answer that mentions `<tool_call>` in a
        // sentence and shows a fenced example further down would match from
        // the sentence onward under the loose gate, and eat the explanation
        // along with the example.
        let patterns = [
            // Every XML/bracket envelope must OPEN A LINE (leading whitespace
            // allowed). A leaked call is emitted as its own block after the
            // model stops writing prose; an answer that documents the syntax
            // does it mid-sentence — `Use <tool_call>{"name":"search"}` — and
            // truncating that sentence at the tag is the false positive this
            // detector must not have. The dogfood repro and every other real
            // leak shape put the envelope on its own line.
            #"(?m)^[ \t]*</?(tool_call|function_call)[^>]*>\s*[\{\[<]"#,
            // `<function=NAME>` must also OPEN a payload: JSON, or the nested
            // `<parameter=` block the llama/qwen fragment shape uses. The bare
            // tag is not enough — prose about the syntax carries it too. (The
            // leading check can keep accepting the bare prefix: prose never
            // OPENS a turn with `<function=`.)
            #"(?m)^[ \t]*<function=[^<>\s]+>\s*(?:[\{\[]|<parameter=)"#,
            // A `<parameter=…>` block on its own line AND closed by
            // `</parameter>`. Both halves are required for the same reason:
            // inline inside a sentence it is documentation.
            #"(?m)^[ \t]*<parameter=[^<>\s]+>[\s\S]{0,4096}?</parameter>"#,
            #"(?m)^[ \t]*\[TOOL_CALLS\]\s*[\{\[]"#,
            // The DeepSeek marker is deliberately NOT line-anchored: its
            // U+2581 separators never occur in human prose, so there is no
            // inline-documentation shape to protect and a real emit can follow
            // the last prose character directly. The `<\u{FF5C}` opener is folded
            // into the match so the prose above it does not keep a dangling
            // half-tag.
            "[<\u{FF5C}]*tool\u{2581}calls\u{2581}begin",
        ]

        // The turn must END in machine syntax, and only the terminal run of
        // it is a candidate tail.
        //
        // This is what "tail" means, and without it a genuine answer that
        // shows a raw (unfenced) call on its own line and then EXPLAINS it
        // lost the explanation: suppression ran from the marker to the end of
        // the turn, so the closing sentences went with the example. Confining
        // the search to the terminal machine-syntax run also fixes the general
        // case — example, prose, then a real envelope — because the earlier
        // example is no longer even a candidate.
        guard let runStart = terminalMachineSyntaxRunStart(in: content) else { return nil }
        let searchRange = runStart..<content.endIndex

        // Fences next: one line pass, reused by every pattern below.
        let fenced = fencedRanges(in: content)

        // The earliest UNFENCED candidate across every pattern.
        //
        // Two things this must not do. It must not stop at the first match of
        // a pattern and decide on it alone — a legitimate fenced example
        // earlier in the turn would then hide the genuine unfenced envelope
        // after it, and that envelope renders raw. And it must not cap the
        // number of candidates it will look at — a cap is spent by the
        // examples and loses the real tail that follows them. Instead a match
        // inside a fence advances the cursor past the WHOLE fenced block, so a
        // fence holding a thousand examples costs one step, not a thousand.
        var earliest: String.Index?
        for pattern in patterns {
            var from = runStart
            // `fenced` is ascending and disjoint and a pattern's matches only
            // move forward, so one cursor walks the fence list ONCE per
            // pattern. Asking `fenced.first { … }` per match instead rescans
            // every range from the start, which is quadratic in the number of
            // fenced examples — on the transcript render path.
            var block = 0
            while from < content.endIndex,
                  let found = content.range(
                      of: pattern, options: [.regularExpression],
                      range: from..<searchRange.upperBound
                  ) {
                while block < fenced.count, fenced[block].upperBound <= found.lowerBound {
                    block += 1
                }
                if block < fenced.count, fenced[block].contains(found.lowerBound) {
                    from = max(fenced[block].upperBound, content.index(after: found.lowerBound))
                    continue
                }
                // Matches arrive in increasing order, so the first unfenced
                // one is this pattern's earliest; no need to scan its tail.
                if earliest == nil || found.lowerBound < earliest! { earliest = found.lowerBound }
                break
            }
        }

        // Nothing unfenced, or the turn OPENS with machine syntax: either way
        // this returns nil — in the second case the leading check owns the
        // turn and the caption stands alone. Testing the first UNFENCED
        // candidate (rather than the first candidate of any kind) is what
        // keeps a turn that opens with a fenced example from bailing here.
        guard let start = earliest, start > content.startIndex else { return nil }
        let prose = String(content[content.startIndex..<start])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        // Whitespace only: the artifact is effectively the whole turn, so the
        // leading check owns it and the caption stands alone.
        return prose.isEmpty ? nil : prose
    }

    /// What to render above the suppression caption: the prose of a turn
    /// whose tail was machine syntax, or nil when the artifact was the whole
    /// turn and the caption stands alone.
    static func proseAboveSuppressedToolCallArtifact(content: String) -> String? {
        // Trailing wins when it fires: its own gates already establish that
        // there is real prose before the machine syntax, which the leading
        // check cannot tell (it matches the DeepSeek marker ANYWHERE in the
        // turn, so a prose answer that trailed off into one used to lose the
        // prose as well as the tail).
        trailingToolCallArtifactProse(in: content)
    }

    /// Where the turn's terminal run of machine syntax begins, or nil when
    /// the turn does not end in machine syntax at all.
    ///
    /// Walks lines from the end: a blank line or a machine-syntax line
    /// (``isMachineSyntaxLine``) belongs to the run, and the first line that
    /// reads as prose stops it.
    ///
    /// Deliberately a line-shape test rather than a JSON/XML parse: the run
    /// only has to tell an envelope dump — what a model emits when the parser
    /// lost its call and generation simply stopped — from a sentence, and the
    /// input is by definition syntax no parser could read. Every test is kept
    /// strict, because widening one moves the boundary EARLIER, and an
    /// over-early boundary eats real prose. A closing ```` ``` ```` stops the
    /// run too, which is correct: a turn that ENDS with a fenced example is
    /// showing the example, not leaking a call.
    private static func terminalMachineSyntaxRunStart(in content: String) -> String.Index? {
        var runStart: String.Index?
        var sawContent = false
        var lineStart = content.startIndex
        var index = content.startIndex
        // Forward pass recording the last prose line's successor, which is the
        // same answer as walking backwards and cheaper on String.Index.
        while true {
            let lineEnd = content[index...].firstIndex(of: "\n") ?? content.endIndex
            let trimmed = content[lineStart..<lineEnd]
                .trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty {
                if isMachineSyntaxLine(trimmed) {
                    if runStart == nil { runStart = lineStart }
                    sawContent = true
                } else {
                    // Prose: everything up to and including this line is the
                    // answer, so any run starts after it.
                    runStart = nil
                }
            }
            guard lineEnd < content.endIndex else { break }
            index = content.index(after: lineEnd)
            lineStart = index
        }
        return sawContent ? runStart : nil
    }

    /// True when a line is a piece of an envelope dump rather than a sentence.
    ///
    /// Structural per opener, not "the first character is punctuation". Prose
    /// opens with punctuation often enough that the loose form ate real
    /// content: a Markdown link (`[That syntax](…) is invalid`), a reference
    /// definition (`[1]: …`), a quoted sentence. Each case below accepts the
    /// shape an envelope actually produces and nothing wider.
    private static func isMachineSyntaxLine(_ trimmed: String) -> Bool {
        // Never in prose, wherever it appears.
        if trimmed.contains("tool\u{2581}calls\u{2581}") { return true }
        guard let first = trimmed.first else { return false }
        switch first {
        case "{", "}", "]", ",":
            // These open no sentence BY THEMSELVES, but prose can open with
            // one: "} closes the object; this is why …". A JSON fragment — `}`,
            // `},`, `},{"name":"x"}`, `{"limit": 10,` — carries no unquoted
            // word; a sentence does.
            return hasNoUnquotedWord(trimmed)
        case "[":
            // Reject Markdown first: a link (`[label](url)`) or a reference
            // definition (`[1]: url`). The digit branch below has to accept
            // `[1, 2]`, so `[1]: url` would otherwise read as an array.
            if trimmed.range(
                of: #"^\[[^\]\n]*\]\s*[:(]"#,
                options: .regularExpression) != nil {
                return false
            }
            // A JSON array opening, or the Mistral marker.
            return trimmed.range(
                of: #"^\[(\s*$|\s*[\{\[\]"'\-0-9]|TOOL_CALLS\])"#,
                options: .regularExpression) != nil
        case "\"":
            // A JSON key or a bare string element, not a quoted sentence.
            return trimmed.range(
                of: #"^"(\\.|[^"\\])*"\s*(:|,?$)"#,
                options: .regularExpression) != nil
        case "<":
            // A tag, not prose that happens to open with a less-than sign.
            return trimmed.contains(">")
                && trimmed.range(
                    of: #"^</?[A-Za-z\uFF5C|]"#,
                    options: .regularExpression) != nil
        default:
            return isJSONScalarLine(trimmed)
        }
    }

    /// True when a line carries no word outside a string literal.
    ///
    /// This is what separates a JSON fragment from a sentence that merely
    /// OPENS with a brace or bracket. Words inside string literals do not
    /// count — they are data, and a leaked call is full of them. The JSON
    /// keywords are allowed through unquoted, since `{"ok": true}` is a
    /// fragment; a run that stops matching one of them (`"trus"`) is a word.
    private static func hasNoUnquotedWord(_ line: String) -> Bool {
        let keywords = ["true", "false", "null"]
        var inString = false
        var escaped = false
        var run = ""
        for character in line {
            if escaped { escaped = false; continue }
            if inString, character == "\\" { escaped = true; continue }
            if character == "\"" {
                inString.toggle()
                run = ""
                continue
            }
            if inString { continue }
            guard character.isLetter else { run = ""; continue }
            run.append(character)
            // One letter is never a word here — `1e5` and a bare `n` in a
            // truncated `null` both have to pass.
            guard run.count >= 2 else { continue }
            let lowered = run.lowercased()
            if !keywords.contains(where: { $0.hasPrefix(lowered) }) { return false }
        }
        return true
    }

    /// True for a line that is nothing but a JSON scalar with an optional
    /// trailing comma — the continuation lines of a pretty-printed array
    /// (`10,`, `true`, `null`). Quoted strings and the bracket/brace forms are
    /// already covered by the first-character test.
    private static func isJSONScalarLine(_ trimmed: String) -> Bool {
        var body = Substring(trimmed)
        if body.hasSuffix(",") { body = body.dropLast() }
        body = Substring(body.trimmingCharacters(in: .whitespaces))
        if body == "true" || body == "false" || body == "null" { return true }
        return body.range(
            of: #"^-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][-+]?[0-9]+)?$"#,
            options: .regularExpression
        ) != nil
    }

    /// The character ranges of ``content`` that sit inside a fenced code
    /// block, opening fence line included.
    ///
    /// CommonMark's actual rule, not a count of literal ```` ``` ````
    /// sequences: a fence opens on a line whose first non-space content is a
    /// run of three or more backticks OR tildes, and closes on a later line
    /// whose run uses the SAME character, is at least as long, and carries
    /// nothing but whitespace after it. Counting
    /// triple-backtick occurrences — the shape this check started as —
    /// misreads a ```` ~~~ ```` fence (no backticks at all) and a
    /// four-backtick fence (the standard way to show a nested example) as
    /// unfenced, which turns the example the user asked for into a "leak"
    /// and truncates the answer at it. Same fence vocabulary the release-notes
    /// renderer already accepts.
    static func fencedRanges(in content: String) -> [Range<String.Index>] {
        var ranges: [Range<String.Index>] = []
        var openStart: String.Index?
        var openMarker: FenceRun?
        var index = content.startIndex
        while index < content.endIndex {
            let lineEnd = content[index...].firstIndex(of: "\n") ?? content.endIndex
            let next = lineEnd < content.endIndex ? content.index(after: lineEnd) : content.endIndex
            if let found = fenceRun(in: content[index..<lineEnd]) {
                if let open = openMarker, let start = openStart {
                    // A closer must use the opener's character, be at least as
                    // long, and carry NOTHING but whitespace after the run.
                    // CommonMark gives a closing fence no info string, so a
                    // ```` ```swift ```` line inside a ``` block is content —
                    // treating it as a closer would leave the rest of the
                    // example unfenced and truncate the answer there.
                    if found.marker == open.marker, found.run >= open.run, found.isBare {
                        ranges.append(start..<next)
                        openMarker = nil
                        openStart = nil
                    }
                } else {
                    openMarker = found
                    openStart = index
                }
            }
            index = next
        }
        // An unterminated fence runs to the end of the turn: a streamed answer
        // cut off mid-example is still an example, not a leak.
        if let start = openStart { ranges.append(start..<content.endIndex) }
        return ranges
    }

    /// A fence line's delimiter run: which character, how long, and whether
    /// anything follows it (an info string, which only an OPENER may have).
    private struct FenceRun {
        let marker: Character
        let run: Int
        let isBare: Bool
    }

    /// The fence run a line carries, or nil when the line is not a fence line.
    ///
    /// Indentation is measured in COLUMNS with tabs expanded to the next
    /// four-column stop, because that is what decides the CommonMark cutoff:
    /// four columns in is an indented code block, not a fence opener, and a
    /// single leading tab already reaches four. A backtick fence's info string
    /// may not contain a backtick, which is what keeps inline `` `code` ``
    /// spans off this path.
    private static func fenceRun(in line: Substring) -> FenceRun? {
        var rest = line
        var column = 0
        while let first = rest.first, first == " " || first == "\t" {
            column = first == "\t" ? (column / 4 + 1) * 4 : column + 1
            if column > 3 { return nil }
            rest = rest.dropFirst()
        }
        guard let marker = rest.first, marker == "`" || marker == "~" else { return nil }
        let run = rest.prefix { $0 == marker }.count
        guard run >= 3 else { return nil }
        let info = rest.dropFirst(run)
        if marker == "`", info.contains("`") { return nil }
        return FenceRun(
            marker: marker, run: run,
            isBare: info.allSatisfy { $0 == " " || $0 == "\t" }
        )
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
    /// match is keyword-based and inclusive, but WHOLE-WORD (see
    /// ``containsKeyword``): substring matching flagged any prose
    /// containing "computer", "sometimes" or "surplus", and a
    /// caption under a correct answer costs more than #308 assumed.
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
    /// True when ``keyword`` appears in ``haystack`` as a whole word
    /// (or whole phrase) rather than as a substring of a longer word.
    ///
    /// ``lowered.contains(kw)`` is what shipped with #308, and it
    /// misfires on ordinary English: ``compute`` matches "computer"
    /// and "computing", ``times`` matches "sometimes", ``plus``
    /// matches "surplus", ``minus`` matches "minuscule",
    /// ``forecast`` matches "forecasting", ``sum of`` matches
    /// "consum[er] of". Every one of those turns a normal prose
    /// question into a "calculator-shaped" one, and a short answer
    /// containing any digit then wears the caution. Dogfooding
    /// 0.14.1 tripped it on a document question with "computed" in
    /// the prose.
    ///
    /// A boundary is anything that is not a letter or a digit, plus
    /// the ends of the string — so "compute 17*23", "(compute)" and
    /// "compute." all match while "computer" does not. Multi-word
    /// keywords keep working because only the outer edges of the
    /// phrase are checked.
    static func containsKeyword(_ keyword: String, in haystack: String) -> Bool {
        guard !keyword.isEmpty else { return false }
        func isWordScalar(_ scalar: Unicode.Scalar) -> Bool {
            CharacterSet.alphanumerics.contains(scalar)
        }
        /// A boundary, or a regular English plural immediately followed by
        /// one. codex flagged the regression this closes: moving from
        /// substring to whole-word matching silently dropped "temperatures",
        /// "forecasts" and "current prices", all of which the old substring
        /// match caught and all of which are ordinary live-data questions.
        /// Accepting a trailing "s"/"es" before the boundary keeps the
        /// inflections without reopening the substring bug — "forecasting"
        /// is still rejected (the next scalar is "i"), "forecasted" too
        /// ("ed" is not "es"), and "concurrent" never contained a keyword to
        /// begin with.
        func boundaryFollows(_ index: String.Index) -> Bool {
            if index == haystack.endIndex { return true }
            if !isWordScalar(haystack[index].unicodeScalars.first!) { return true }
            for suffix in ["es", "s"] where haystack[index...].hasPrefix(suffix) {
                let after = haystack.index(index, offsetBy: suffix.count)
                if after == haystack.endIndex
                    || !isWordScalar(haystack[after].unicodeScalars.first!) {
                    return true
                }
            }
            return false
        }
        var searchStart = haystack.startIndex
        while let range = haystack.range(
            of: keyword,
            range: searchStart..<haystack.endIndex
        ) {
            let leftOK = range.lowerBound == haystack.startIndex
                || !isWordScalar(
                    haystack[haystack.index(before: range.lowerBound)]
                        .unicodeScalars.first!
                )
            if leftOK && boundaryFollows(range.upperBound) { return true }
            // Overlapping matches matter ("timestimes"), so advance by
            // one character rather than past the whole keyword.
            searchStart = haystack.index(after: range.lowerBound)
        }
        return false
    }

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

        // The three remaining lanes each live in their own predicate, so
        // Gate 1c can name exactly which of them an attachment neutralises
        // without either copy of the keyword lists drifting.
        if promptContainsMathKeyword(lowered) { return true }
        // Note: codex r1 MAJOR-1 (#308 PR) dropped the bare
        // ``"what is the"`` keyword — it matched every plain factual
        // question ("What is the capital of France?") and false-flagged
        // short prose answers like "Paris." Web-search keywords must
        // point at LIVE / DATED information; an evergreen factual
        // lookup is not the failure mode this caption guards against.
        if promptAsksForLiveData(lowered) { return true }
        if promptAsksForExternalRetrieval(lowered) { return true }

        return false
    }

    /// Math vocabulary with no literal expression — "calculate the total",
    /// "sum of the line items", "what percent of it".
    ///
    /// This is the ONE lane an attachment neutralises (see Gate 1c of
    /// ``shouldFlagToolNotCalled``): the operands can live on the page, so
    /// reading them off it is the grounded, correct answer.
    static func promptContainsMathKeyword(_ prompt: String) -> Bool {
        let lowered = prompt.lowercased()
        let mathKeywords: [String] = [
            "square root", "sqrt", "percent", "calculate", "compute", "solve",
            "divide", "multiply", "sum of", "product of", "plus", "minus",
            "times", "divided by"
        ]
        for kw in mathKeywords where containsKeyword(kw, in: lowered) { return true }
        return false
    }

    /// Asks for something to be fetched from outside the conversation.
    ///
    /// These used to sit with the math keywords on the theory that "look up
    /// the invoice number" is answered by an attached page. codex was right
    /// that the theory does not survive its own counterexample: attach a
    /// résumé and ask to "search for Ada Lovelace's biography" and the page
    /// grounds nothing, yet the exemption silenced the caption on a
    /// completely ungrounded answer — the #308 failure mode itself.
    ///
    /// So external-retrieval language now withholds the exemption. The cost
    /// is accepted knowingly: "look up the invoice number" with the invoice
    /// attached will wear a caution it does not deserve. Of the two errors,
    /// the false negative is the one this feature exists to prevent, and the
    /// caption is dismissible while a silent wrong answer is not.
    static func promptAsksForExternalRetrieval(_ prompt: String) -> Bool {
        let lowered = prompt.lowercased()
        let retrievalKeywords: [String] = ["search for", "look up", "look it up"]
        for kw in retrievalKeywords where containsKeyword(kw, in: lowered) { return true }
        return false
    }

    /// True when an attached document could actually answer `prompt`.
    ///
    /// Stated as what IS exempt rather than as a list of disqualifiers,
    /// because the disqualifier form grew a hole every review round. Of the
    /// four lanes that make a prompt tool-shaped at all
    /// (``promptLooksCalculatorish``), exactly one is answerable from a page
    /// the user attached:
    ///
    /// | lane | attached document can answer it? |
    /// | --- | --- |
    /// | math keywords, no literal expression | **yes** — operands are on the page |
    /// | self-contained arithmetic (`17*23`) | no — prompt brought its own numbers |
    /// | live data (today's price, the weather) | no — no page holds a moving target |
    /// | external retrieval (search for, look up) | no — it asks to leave the document |
    static func promptIsAttachmentAnswerable(_ prompt: String) -> Bool {
        guard !promptAsksForLiveData(prompt) else { return false }
        guard !promptContainsSelfContainedArithmetic(prompt) else { return false }
        guard !promptAsksForExternalRetrieval(prompt) else { return false }
        return promptContainsMathKeyword(prompt)
    }

    /// True when `prompt` names a MOVING target — something that
    /// changes without the conversation changing, so no document the
    /// user attached can contain the answer.
    ///
    /// This is the line that makes Gate 1c of
    /// ``shouldFlagToolNotCalled`` safe to draw. Without it, any
    /// attachment on the turn silences the caption, including for
    /// "here is my portfolio PDF — what is today's stock price?",
    /// where a bare number with no tool call is exactly the
    /// hallucination the caption exists to flag.
    ///
    /// ``"google for"`` lives here rather than with the
    /// retrieval-shaped keywords in ``promptLooksCalculatorish``
    /// because it names an external service outright; ``"search
    /// for"`` / ``"look up"`` do not, and pointing either of those at
    /// an attached document is an ordinary thing for a user to do.
    ///
    /// Whole-word matching throughout (see ``containsKeyword``), so
    /// "temperature" does not fire on "temperatures"' neighbours and
    /// "current" does not fire on "concurrent".
    static func promptAsksForLiveData(_ prompt: String) -> Bool {
        let lowered = prompt.lowercased()
        let liveKeywords: [String] = [
            "google for",
            "latest news", "latest version", "news about",
            "today's", "this week's", "right now",
            "current price", "stock price", "exchange rate",
            "current weather",
            // Looser than the phrases above, and deliberately so: a
            // bare "weather" / "temperature" / "forecast" is always a
            // live-data question, attachment or not.
            "weather", "temperature", "forecast"
        ]
        for kw in liveKeywords where containsKeyword(kw, in: lowered) { return true }
        return false
    }

    /// True when `prompt` carries arithmetic that stands on its own — a
    /// digit and an operator, as in "17*23" or "1200 * 0.15".
    ///
    /// Such a question does not become document-grounded just because a
    /// document happens to be attached: the numbers are in the prompt,
    /// the page is irrelevant, and the calculator is exactly the tool
    /// that should have run. So this withholds Gate 1c's exemption
    /// alongside ``promptAsksForLiveData``.
    ///
    /// Note what it deliberately does NOT cover: math *keywords* with
    /// no literal expression — "what is the total due? calculate it
    /// from the invoice", the 0.14.1 dogfood case — where the operands
    /// live on the page and reading them off it is the grounded,
    /// correct answer. The discriminator is whether the prompt brought
    /// its own numbers.
    static func promptContainsSelfContainedArithmetic(_ prompt: String) -> Bool {
        let hasDigit = prompt.unicodeScalars.contains { scalar in
            scalar.value >= 0x30 && scalar.value <= 0x39
        }
        guard hasDigit else { return false }
        let mathOperators: Set<Character> = ["+", "*", "/", "%", "=", "^"]
        if prompt.contains(where: { mathOperators.contains($0) }) { return true }
        // "-" only counts with a number on each side: a hyphenated filename
        // or a dashed aside ("attached is my resume - what does it say?") is
        // not arithmetic, but "1200-180" and "1200 - 180" both are. codex
        // caught the spaced form being missed, which is the way most people
        // actually type it.
        let scalars = Array(prompt.unicodeScalars)
        func isDigit(_ index: Int) -> Bool {
            scalars.indices.contains(index)
                && scalars[index].value >= 0x30
                && scalars[index].value <= 0x39
        }
        func digitLookingBack(from index: Int) -> Bool {
            var i = index
            while scalars.indices.contains(i), scalars[i] == " " || scalars[i] == "\t" { i -= 1 }
            return isDigit(i)
        }
        func digitLookingForward(from index: Int) -> Bool {
            var i = index
            while scalars.indices.contains(i), scalars[i] == " " || scalars[i] == "\t" { i += 1 }
            return isDigit(i)
        }
        for (offset, scalar) in scalars.enumerated() where scalar == "-" {
            if digitLookingBack(from: offset - 1), digitLookingForward(from: offset + 1) {
                return true
            }
        }
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
