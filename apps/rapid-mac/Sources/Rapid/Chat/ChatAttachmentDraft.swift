import Foundation

/// Product state for attachments waiting in the Chat composer.
///
/// Keeping this outside ``ChatView`` makes the important identity and lifecycle
/// rules directly testable: every input method feeds the same draft, a send
/// atomically consumes it, and a later turn cannot inherit stale attachments.
struct ChatAttachmentDraft: Equatable {
    /// Immutable ownership transfer from the composer to one user turn.
    ///
    /// The arrays are captured before asynchronous chat work starts, so later
    /// composer mutations cannot alter an in-flight request.
    struct Submission: Equatable {
        let images: [ChatImageAttachment]
        let files: [ChatFileAttachment]
    }

    private(set) var images: [ChatImageAttachment] = []
    private(set) var files: [ChatFileAttachment] = []
    private(set) var sourcePaths: [UUID: String] = [:]
    private(set) var fileImportID: UUID?
    private(set) var imageImportID: UUID?
    var notice: String?

    var hasAttachments: Bool { !images.isEmpty || !files.isEmpty }
    var isImportingFiles: Bool { fileImportID != nil || imageImportID != nil }

    /// Appends a single image only when it fits the per-message image budget
    /// (count and combined bytes). Returns `false` (and appends nothing) when
    /// the budget is already exhausted, so a caller can surface a notice. This
    /// is the draft-level companion to ``ChatImageAttachment/importCandidates``:
    /// the pre-read gate bounds the picker/drop, and this bounds the direct
    /// paste path and any late import completion.
    @discardableResult
    mutating func appendImage(
        _ image: ChatImageAttachment,
        sourceURL: URL? = nil
    ) -> Bool {
        guard Self.canAppend(image, to: images) else { return false }
        images.append(image)
        if let sourceURL { sourcePaths[image.id] = Self.attachmentKey(for: sourceURL) }
        return true
    }

    /// Appends each imported image that still fits the per-message image budget,
    /// preserving order and source identity, and returns how many were rejected
    /// for exceeding it. Mirrors how file imports apply their fitted budget;
    /// the existing, already-fitted images always precede the imported ones, so
    /// the shared gate keeps only the imported subset that still fits.
    @discardableResult
    mutating func appendImages(
        _ imported: [(attachment: ChatImageAttachment, sourceURL: URL)]
    ) -> Int {
        let fitted = ChatImageAttachment.fittedForMessage(
            images + imported.map(\.attachment)
        )
        let kept = imported.filter { attachment, _ in
            fitted.contains { $0.id == attachment.id }
        }
        for item in kept {
            images.append(item.attachment)
            sourcePaths[item.attachment.id] = Self.attachmentKey(for: item.sourceURL)
        }
        return imported.count - kept.count
    }

    private static func canAppend(
        _ image: ChatImageAttachment,
        to images: [ChatImageAttachment]
    ) -> Bool {
        guard images.count < ChatImageAttachment.maxImagesPerMessage else { return false }
        let existingBytes = images.reduce(0) { $0 + $1.encodedDataURLByteCount }
        return existingBytes + image.encodedDataURLByteCount
            <= ChatImageAttachment.maxCombinedEncodedImageBytes
    }

    mutating func beginImageImport() -> UUID? {
        guard imageImportID == nil else { return nil }
        let id = UUID()
        // A new selection supersedes an old notice. Any notice produced by
        // that selection after async image decoding starts must survive its
        // later completion.
        notice = nil
        imageImportID = id
        return id
    }

    @discardableResult
    mutating func finishImageImport(
        id: UUID,
        _ imported: [(attachment: ChatImageAttachment, sourceURL: URL)],
        notice: String?
    ) -> Bool {
        guard imageImportID == id else { return false }
        let combinedCount = images.count + imported.count
        let budgetRejectedCount = appendImages(imported)
        var mergedNotice = notice
        if budgetRejectedCount > 0 {
            // The on-disk pre-read estimate can differ from the normalized
            // PNG/JPEG payload. This is the authoritative post-normalization
            // gate, so its late rejection must be visible rather than silently
            // dropping an image that the picker initially admitted.
            let limit: ChatImageAttachment.ImageBudgetLimit =
                combinedCount > ChatImageAttachment.maxImagesPerMessage ? .count : .bytes
            let budgetNotice = ChatImageAttachment.budgetNotice(
                rejectedCount: budgetRejectedCount,
                limit: limit
            )
            mergedNotice = mergedNotice.map { "\(budgetNotice) \($0)" } ?? budgetNotice
        }
        if let mergedNotice {
            if let current = self.notice, current != mergedNotice {
                self.notice = "\(current) \(mergedNotice)"
            } else {
                self.notice = mergedNotice
            }
        }
        imageImportID = nil
        return true
    }

    /// Starts one asynchronous import generation. A second source cannot race
    /// the first because every UI entry point funnels through this method.
    mutating func beginFileImport() -> UUID? {
        guard fileImportID == nil else { return nil }
        let id = UUID()
        fileImportID = id
        return id
    }

    /// Applies results only to the generation that created them.
    ///
    /// A conversation transition can cancel an import while file parsing is
    /// still off-main-thread. Its late completion must not resurrect stale
    /// attachments in the current composer.
    @discardableResult
    mutating func finishFileImport(
        id: UUID,
        _ imported: [(attachment: ChatFileAttachment, sourceURL: URL)],
        notice: String?
    ) -> Bool {
        guard fileImportID == id else { return false }
        files = ChatFileAttachment.fittedForMessage(files + imported.map(\.attachment))
        for item in imported {
            sourcePaths[item.attachment.id] = Self.attachmentKey(for: item.sourceURL)
        }
        self.notice = notice
        fileImportID = nil
        return true
    }

    /// Invalidates only the expected generation, when supplied. This prevents
    /// late cleanup from an older task from cancelling newer work.
    @discardableResult
    mutating func cancelFileImport(id expectedID: UUID? = nil, notice: String? = nil) -> Bool {
        guard let activeID = fileImportID else { return false }
        guard expectedID == nil || expectedID == activeID else { return false }
        fileImportID = nil
        if let notice { self.notice = notice }
        return true
    }

    mutating func removeImage(id: UUID) {
        images.removeAll { $0.id == id }
        sourcePaths[id] = nil
    }

    mutating func removeFile(id: UUID) {
        files.removeAll { $0.id == id }
        sourcePaths[id] = nil
    }

    /// Returns exactly one turn's attachments and clears all transient state.
    /// This is intentionally one mutation so a new import cannot observe an
    /// old source-path map after the visible chips have already disappeared.
    mutating func takeSubmission() -> Submission {
        let submission = Submission(images: images, files: files)
        images = []
        files = []
        sourcePaths = [:]
        notice = nil
        fileImportID = nil
        imageImportID = nil
        return submission
    }

    func filteringAlreadyAttached(_ urls: [URL]) -> (fresh: [URL], duplicates: Int) {
        Self.withoutAlreadyAttached(urls, attached: Set(sourcePaths.values))
    }

    /// Identity for "the same file". Symlinks and `..` are resolved; equal
    /// bytes at distinct real paths deliberately remain separate attachments.
    static func attachmentKey(for url: URL) -> String {
        url.standardizedFileURL.resolvingSymlinksInPath().path
    }

    static func withoutAlreadyAttached(
        _ urls: [URL], attached: Set<String>
    ) -> (fresh: [URL], duplicates: Int) {
        var seen = attached
        var fresh: [URL] = []
        for url in urls where seen.insert(attachmentKey(for: url)).inserted {
            fresh.append(url)
        }
        return (fresh, urls.count - fresh.count)
    }
}

/// Conversation-keyed composer state.
///
/// Async work always writes through the ID it started with. Browsing another
/// conversation therefore neither exposes the old draft nor cancels the new
/// conversation's work; returning restores the original draft and results.
struct ChatAttachmentDraftStore: Equatable {
    struct ImportRequest: Equatable {
        let conversationID: UUID
        let generationID: UUID
    }

    private var drafts: [UUID: ChatAttachmentDraft] = [:]

    subscript(conversationID: UUID) -> ChatAttachmentDraft {
        get { drafts[conversationID] ?? ChatAttachmentDraft() }
        set { drafts[conversationID] = newValue }
    }

    mutating func beginFileImport(conversationID: UUID) -> ImportRequest? {
        var draft = self[conversationID]
        guard let generationID = draft.beginFileImport() else { return nil }
        drafts[conversationID] = draft
        return ImportRequest(conversationID: conversationID, generationID: generationID)
    }

    mutating func beginImageImport(conversationID: UUID) -> ImportRequest? {
        var draft = self[conversationID]
        guard let generationID = draft.beginImageImport() else { return nil }
        drafts[conversationID] = draft
        return ImportRequest(conversationID: conversationID, generationID: generationID)
    }

    @discardableResult
    mutating func finishImageImport(
        request: ImportRequest,
        _ imported: [(attachment: ChatImageAttachment, sourceURL: URL)],
        notice: String?
    ) -> Bool {
        guard var draft = drafts[request.conversationID] else { return false }
        guard draft.finishImageImport(
            id: request.generationID,
            imported,
            notice: notice
        ) else { return false }
        drafts[request.conversationID] = draft
        return true
    }

    /// Completes only an import whose owning conversation still exists in the
    /// store. In particular, a late task cannot recreate a deleted draft.
    @discardableResult
    mutating func finishFileImport(
        request: ImportRequest,
        _ imported: [(attachment: ChatFileAttachment, sourceURL: URL)],
        notice: String?
    ) -> Bool {
        guard var draft = drafts[request.conversationID] else { return false }
        guard draft.finishFileImport(
            id: request.generationID,
            imported,
            notice: notice
        ) else { return false }
        drafts[request.conversationID] = draft
        return true
    }

    /// Releases attachment data for deleted conversations while retaining the
    /// unsaved active conversation, which is not present in history yet.
    ///
    /// Returns the file-attachment ids of every dropped draft so the caller can
    /// delete their cached plaintext. Dropping the draft alone is a leak: a
    /// successful import registers the document's FULL text in
    /// ``DocumentContentCache`` the moment it parses, and once the draft that
    /// held the chip is gone nothing on screen references those documents —
    /// no chip to click, no conversation for ``deleteConversation`` to walk.
    /// The extract would sit in Application Support until the 90-day sweep.
    @discardableResult
    mutating func retainDrafts(for conversationIDs: Set<UUID>) -> [UUID] {
        var discarded: [UUID] = []
        for (conversationID, draft) in drafts where !conversationIDs.contains(conversationID) {
            discarded.append(contentsOf: draft.files.map(\.id))
        }
        drafts = drafts.filter { conversationIDs.contains($0.key) }
        return discarded
    }
}
