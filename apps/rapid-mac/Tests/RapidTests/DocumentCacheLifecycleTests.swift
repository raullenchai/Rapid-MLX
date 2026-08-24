import AppKit
import Foundation
import PDFKit
import Testing
@testable import Rapid

/// Lifecycle contracts for ``DocumentContentCache`` — how an extract is
/// STOPPED and DELETED, as opposed to how it is read.
///
/// Both halves are privacy properties, not housekeeping. An entry holds the
/// complete plaintext of a file the user attached, and the persistent tier
/// keeps it in Application Support across launches. Before these paths existed:
///
///   * removing an attachment or deleting a conversation left `<uuid>.json`
///     behind until unrelated LRU pressure happened to evict it, which for a
///     handful of documents is never — a retention regression against the
///     previous behaviour, where the extract lived inline in the conversation
///     file and went with it; and
///   * a multi-minute OCR pass ran on an unstructured ``Task.detached`` whose
///     handle was discarded, so nothing could cancel it. Removing a 529-page
///     scan left Vision and PDFKit working for minutes on a document nobody
///     would read.
@MainActor
@Suite("Document cache lifecycle")
struct DocumentCacheLifecycleTests {
    /// A cache with its own temp directory, so the disk tier is genuinely
    /// exercised without touching the user's real Application Support tree.
    private func diskCache(
        in directory: URL,
        diskTTL: TimeInterval = 90 * 24 * 60 * 60
    ) -> DocumentContentCache {
        DocumentContentCache(diskDirectory: directory, diskTTL: diskTTL)
    }

    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("doc-cache-\(UUID().uuidString)", isDirectory: true)
    }

    private func diskFile(_ directory: URL, _ id: UUID) -> URL {
        directory.appendingPathComponent("\(id.uuidString).json", isDirectory: false)
    }

    /// An attachment whose full text is already in `cache` — the state after a
    /// real import, where the preview rides on the message and the complete
    /// extract lives in the store.
    private func seedAttachment(
        _ name: String,
        in cache: DocumentContentCache
    ) throws -> ChatFileAttachment {
        let attachment = try ChatFileAttachment(
            filename: name,
            kind: .txt,
            extractedText: "preview of \(name)",
            sourceByteCount: 100
        )
        cache.put(attachment.id, entry: DocumentContentCache.Entry(
            filename: name,
            text: "full text of \(name)"
        ))
        return attachment
    }

    // MARK: - Deletion

    @Test("Removing a document deletes its persisted plaintext, not just the hot copy")
    func removeDeletesTheDiskEntry() throws {
        let dir = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let cache = diskCache(in: dir)

        let id = UUID()
        cache.put(id, entry: DocumentContentCache.Entry(
            filename: "payslip.pdf",
            text: "SALARY 12345 CONFIDENTIAL"
        ))
        #expect(FileManager.default.fileExists(atPath: diskFile(dir, id).path))

        cache.remove(id)

        // Both tiers. A memory-only removal would still leave the plaintext on
        // disk, where the next launch would load it straight back in.
        #expect(cache.get(id) == nil)
        #expect(!FileManager.default.fileExists(atPath: diskFile(dir, id).path))
    }

    @Test("A removed document's text is not recoverable from the cache directory")
    func removedTextIsGoneFromDisk() throws {
        // The assertion the user actually cares about: not "the API returns
        // nil" but "the words are no longer on my disk".
        let dir = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let cache = diskCache(in: dir)

        let id = UUID()
        cache.put(id, entry: DocumentContentCache.Entry(
            filename: "diagnosis.pdf",
            text: "PATIENT NAME AND DIAGNOSIS"
        ))
        cache.remove(id)

        let remaining = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
        for name in remaining {
            let contents = (try? String(contentsOf: dir.appendingPathComponent(name), encoding: .utf8)) ?? ""
            #expect(!contents.contains("DIAGNOSIS"))
        }
    }

    @Test("Removing a document that was never cached is a no-op, not a failure")
    func removeOfUnknownIDIsHarmless() throws {
        // The common case for a conversation restored from history, whose
        // extracts aged out long ago: deletion must not care.
        let dir = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let cache = diskCache(in: dir)

        let kept = UUID()
        cache.put(kept, entry: DocumentContentCache.Entry(filename: "keep.txt", text: "keep me"))

        cache.remove(UUID())

        #expect(cache.get(kept)?.text == "keep me")
        #expect(FileManager.default.fileExists(atPath: diskFile(dir, kept).path))
    }

    @Test("Bulk removal deletes every listed document and leaves the rest")
    func bulkRemovalIsScoped() throws {
        // The shape conversation deletion needs: every attachment on every
        // message goes, and nothing else does.
        let dir = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let cache = diskCache(in: dir)

        let doomed = [UUID(), UUID(), UUID()]
        let survivor = UUID()
        for id in doomed + [survivor] {
            cache.put(id, entry: DocumentContentCache.Entry(filename: "d.txt", text: "text \(id)"))
        }

        cache.remove(contentsOf: doomed)

        for id in doomed {
            #expect(cache.get(id) == nil)
            #expect(!FileManager.default.fileExists(atPath: diskFile(dir, id).path))
        }
        #expect(cache.get(survivor) != nil)
    }

    // MARK: - Conversation deletion

    @Test("Deleting a conversation deletes the documents attached to it")
    func deletingConversationDeletesItsExtracts() throws {
        // The privacy regression this closes: the conversation was the only
        // place these attachments were visible, so after deleting it the user
        // had no way to see — let alone remove — the extracts left behind.
        let dir = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let cache = diskCache(in: dir)

        let storeURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("conv-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: storeURL) }

        let viewModel = ChatViewModel(
            conversationStoreURL: storeURL,
            documentCache: cache
        )

        let attachment = try ChatFileAttachment(
            filename: "contract.pdf",
            kind: .pdf,
            extractedText: "the whole agreement",
            sourceByteCount: 1_000
        )
        cache.put(attachment.id, entry: DocumentContentCache.Entry(
            filename: "contract.pdf",
            text: "the whole agreement, at length"
        ))
        #expect(FileManager.default.fileExists(atPath: diskFile(dir, attachment.id).path))

        viewModel.devSeedMessages([
            ChatMessage(role: .user, content: "summarize", fileAttachments: [attachment])
        ])
        let conversationID = viewModel.activeConversationID
        viewModel.deleteConversation(conversationID)

        #expect(cache.get(attachment.id) == nil)
        #expect(!FileManager.default.fileExists(atPath: diskFile(dir, attachment.id).path))
    }

    @Test("Deleting one conversation leaves another conversation's documents alone")
    func deletingConversationIsScopedToItsOwnDocuments() throws {
        let dir = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let cache = diskCache(in: dir)

        let storeURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("conv-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: storeURL) }

        let viewModel = ChatViewModel(
            conversationStoreURL: storeURL,
            documentCache: cache
        )

        func attach(_ name: String) throws -> ChatFileAttachment {
            try seedAttachment(name, in: cache)
        }

        // First conversation, then archived by starting a second one —
        // ``newConversation`` persists the outgoing transcript on its way out.
        let first = try attach("first.txt")
        viewModel.devSeedMessages([
            ChatMessage(role: .user, content: "one", fileAttachments: [first])
        ])
        let firstID = viewModel.activeConversationID
        viewModel.newConversation()

        let second = try attach("second.txt")
        viewModel.devSeedMessages([
            ChatMessage(role: .user, content: "two", fileAttachments: [second])
        ])

        viewModel.deleteConversation(firstID)

        #expect(cache.get(first.id) == nil)
        #expect(cache.get(second.id) != nil)
    }

    /// A conversation whose visible path and off-path branch carry DIFFERENT
    /// documents, written straight to a store file.
    ///
    /// Built by hand rather than by driving ``editUserMessage``: that path
    /// re-sends the original turn's attachments, so both siblings end up
    /// sharing one id and the "did we walk the branches?" question never
    /// actually gets asked.
    private func seedBranchedConversation(
        storeURL: URL,
        visible: ChatFileAttachment,
        branched: ChatFileAttachment
    ) -> UUID {
        let root = ChatMessage(role: .user, content: "root prompt")
        var onPath = ChatMessage(
            role: .assistant,
            content: "kept answer",
            fileAttachments: [visible]
        )
        onPath.parentID = root.id
        var offPath = ChatMessage(
            role: .assistant,
            content: "superseded answer",
            fileAttachments: [branched]
        )
        offPath.parentID = root.id

        let conversation = ChatConversation(
            id: UUID(),
            title: "branched",
            messages: [root, onPath],
            branches: [offPath],
            activeLeafID: onPath.id,
            createdAt: Date(),
            updatedAt: Date()
        )
        ConversationStore.save([conversation], to: storeURL)
        // Writes go through an async serial queue; the view model below reads
        // the file synchronously on init.
        ConversationStore.flush()
        return conversation.id
    }

    @Test("Deleting a stored conversation also deletes documents on its OTHER branches")
    func deletingStoredConversationCoversBranchedAwayAttachments() throws {
        // ``ChatConversation/messages`` is only the branch the user was
        // looking at. A document attached to a turn they later regenerated
        // away sits in ``branches`` — just as deleted from the user's point of
        // view, and just as much plaintext left in Application Support.
        let dir = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let cache = diskCache(in: dir)

        let storeURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("conv-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: storeURL) }

        let visible = try seedAttachment("visible.txt", in: cache)
        let branched = try seedAttachment("branched.txt", in: cache)
        let conversationID = seedBranchedConversation(
            storeURL: storeURL,
            visible: visible,
            branched: branched
        )

        let viewModel = ChatViewModel(
            conversationStoreURL: storeURL,
            documentCache: cache
        )
        viewModel.deleteConversation(conversationID)

        #expect(cache.get(visible.id) == nil)
        #expect(cache.get(branched.id) == nil)
        #expect(!FileManager.default.fileExists(atPath: diskFile(dir, branched.id).path))
    }

    @Test("Deleting the OPEN conversation also deletes documents on its other branches")
    func deletingActiveConversationCoversBranchedAwayAttachments() throws {
        // Same gap on the live side: ``messages`` holds the visible path and
        // everything else is in ``branchedAway``, so the collection has to
        // read the tree rather than the path.
        let dir = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let cache = diskCache(in: dir)

        let storeURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("conv-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: storeURL) }

        let visible = try seedAttachment("visible.txt", in: cache)
        let branched = try seedAttachment("branched.txt", in: cache)
        let conversationID = seedBranchedConversation(
            storeURL: storeURL,
            visible: visible,
            branched: branched
        )

        let viewModel = ChatViewModel(
            conversationStoreURL: storeURL,
            documentCache: cache
        )
        // Opening it splits the tree into path + branchedAway, which is the
        // shape the live deletion path actually sees.
        viewModel.selectConversation(conversationID)
        #expect(viewModel.activeConversationID == conversationID)
        #expect(!viewModel.messages.contains { $0.content == "superseded answer" })

        viewModel.deleteConversation(conversationID)

        #expect(cache.get(branched.id) == nil)
        #expect(!FileManager.default.fileExists(atPath: diskFile(dir, branched.id).path))
    }

    // MARK: - Message deletion

    @Test("Deleting a subtree deletes the documents attached inside it")
    func deletingSubtreeDeletesItsExtracts() throws {
        // Same privacy contract as conversation deletion, one level down: the
        // deleted bubbles were the only place these attachments were visible.
        let dir = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let cache = diskCache(in: dir)

        let viewModel = ChatViewModel(
            persistsConversations: false,
            documentCache: cache
        )

        let attachment = try ChatFileAttachment(
            filename: "invoice.pdf",
            kind: .pdf,
            extractedText: "amount due",
            sourceByteCount: 1_000
        )
        cache.put(attachment.id, entry: DocumentContentCache.Entry(
            filename: "invoice.pdf",
            text: "the whole invoice, at length"
        ))

        let keptPrompt = ChatMessage(role: .user, content: "hello")
        let doomed = ChatMessage(
            role: .user,
            content: "and this one",
            fileAttachments: [attachment]
        )
        viewModel.devSeedMessages([
            keptPrompt,
            ChatMessage(role: .assistant, content: "hi"),
            doomed,
        ])

        #expect(viewModel.deleteMessage(id: doomed.id))

        #expect(cache.get(attachment.id) == nil)
        #expect(!FileManager.default.fileExists(atPath: diskFile(dir, attachment.id).path))
    }

    @Test("A document still referenced by a surviving branch is not deleted")
    func subtreeDeletionSparesSharedAttachments() throws {
        // Editing a prompt re-sends the SAME attachment down a new branch, so
        // one attachment id can appear on two sibling turns. Deleting one of
        // them must not empty the cache out from under the other.
        let dir = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let cache = diskCache(in: dir)

        let viewModel = ChatViewModel(
            persistsConversations: false,
            documentCache: cache
        )

        let attachment = try ChatFileAttachment(
            filename: "shared.pdf",
            kind: .pdf,
            extractedText: "shared preview",
            sourceByteCount: 1_000
        )
        cache.put(attachment.id, entry: DocumentContentCache.Entry(
            filename: "shared.pdf",
            text: "the shared document"
        ))

        let original = ChatMessage(
            role: .user,
            content: "first draft",
            fileAttachments: [attachment]
        )
        viewModel.devSeedMessages([
            original,
            ChatMessage(role: .assistant, content: "an answer"),
        ])
        // The edit carries the attachment onto a sibling turn.
        _ = viewModel.editUserMessage(
            id: original.id,
            newContent: "second draft",
            alias: "test-model"
        )
        viewModel.stopAndPersist()
        let survivor = try #require(viewModel.messages.first)
        #expect(survivor.fileAttachments.map(\.id) == [attachment.id])

        // Delete the branched-away original — its attachment id is still
        // referenced by the turn on screen.
        #expect(viewModel.deleteMessage(id: original.id))

        #expect(cache.get(attachment.id) != nil)
    }

    // MARK: - TTL

    @Test("An extract older than the TTL is deleted even with room to spare")
    func expiredEntriesAreSweptRegardlessOfCaps() throws {
        // The caps are a SIZE policy: a user who attaches a few documents a
        // year never reaches 64 entries or 512 MB, so without a TTL their
        // plaintext stays on disk for the life of the install.
        let dir = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }

        let stale = UUID()
        do {
            let seeding = diskCache(in: dir)
            seeding.put(stale, entry: DocumentContentCache.Entry(
                filename: "old.pdf",
                text: "last year's tax return"
            ))
        }
        // Backdate past any plausible TTL. The sweep reads modification time,
        // which is also what a real months-old entry would carry.
        try FileManager.default.setAttributes(
            [.modificationDate: Date().addingTimeInterval(-200 * 24 * 60 * 60)],
            ofItemAtPath: diskFile(dir, stale).path
        )

        // A fresh cache sweeps on initialization — the launch path.
        let cache = diskCache(in: dir)
        #expect(!FileManager.default.fileExists(atPath: diskFile(dir, stale).path))
        #expect(cache.get(stale) == nil)
    }

    @Test("An extract within the TTL survives the sweep")
    func freshEntriesSurviveTheSweep() throws {
        // Expiry must not be so eager that reopening a recent conversation
        // stops working — that is the promise the persistent tier makes.
        let dir = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }

        let id = UUID()
        do {
            let seeding = diskCache(in: dir)
            seeding.put(id, entry: DocumentContentCache.Entry(
                filename: "recent.pdf",
                text: "this quarter's report"
            ))
        }

        let cache = diskCache(in: dir)
        #expect(cache.get(id)?.text == "this quarter's report")
    }

    // MARK: - Deletion vs. a publish already in flight

    @Test("A publish that began before removal cannot resurrect the document")
    func removalBeatsAnInFlightPublish() throws {
        // The TOCTOU this closes, deterministically rather than by timing:
        //
        //   1. extraction finishes its work and its cancellation check passes
        //   2. the user removes the document — memory cleared, <uuid>.json gone
        //   3. extraction resumes and publishes
        //
        // Moving the cancellation check closer to the write only narrows step
        // 2's window; it cannot close it, because the task can be descheduled
        // between ANY check and the write that follows. So the guarantee lives
        // in the cache: a publish presents the generation it started with, and
        // removal invalidates it.
        //
        // Interleaving the calls directly is what makes this a proof rather
        // than a stress test — no sleeps, no scheduler dependence.
        let dir = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let cache = diskCache(in: dir)

        let id = UUID()
        // Step 1: the extraction takes its token and completes its work.
        let generation = cache.generation(for: id)

        // Step 2: the user deletes the document while that publish is pending.
        cache.remove(id)

        // Step 3: the extraction resumes and tries to publish. It must fail.
        let published = cache.publish(
            id,
            entry: DocumentContentCache.Entry(
                filename: "confidential.pdf",
                text: "PRIVATE CONTENTS THE USER DELETED"
            ),
            ifGenerationIs: generation
        )

        #expect(published == false)
        #expect(cache.get(id) == nil)
        #expect(!FileManager.default.fileExists(atPath: diskFile(dir, id).path))
        // And the words themselves are nowhere in the cache directory.
        let remaining = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
        for name in remaining {
            let contents = (try? String(contentsOf: dir.appendingPathComponent(name), encoding: .utf8)) ?? ""
            #expect(!contents.contains("PRIVATE CONTENTS"))
        }
    }

    @Test("A publish begun after removal is a normal re-attach and succeeds")
    func removalDoesNotPoisonTheIDForever() throws {
        // The tombstone must not become a permanent ban: re-attaching the same
        // document (or the extraction of a genuinely later attach) has to work.
        // A generation taken AFTER the removal is current, so it publishes.
        let dir = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let cache = diskCache(in: dir)

        let id = UUID()
        cache.put(id, entry: DocumentContentCache.Entry(filename: "a.txt", text: "first"))
        cache.remove(id)

        let generation = cache.generation(for: id)
        let published = cache.publish(
            id,
            entry: DocumentContentCache.Entry(filename: "a.txt", text: "attached again"),
            ifGenerationIs: generation
        )

        #expect(published)
        #expect(cache.get(id)?.text == "attached again")
    }

    @Test("A task cannot register after its pending extraction already finished")
    func completedExtractionCannotLeaveARegisteredTask() async {
        let cache = DocumentContentCache(diskDirectory: nil)
        let id = UUID()
        cache.beginPending(id)
        cache.finishPending(id)

        let completed = Task.detached {}
        await completed.value
        #expect(!cache.registerExtraction(id, task: completed))
        #expect(!cache.hasRegisteredExtraction(id))
    }

    @Test("Progress from another document does not extend a stalled wait")
    func progressIsScopedToItsDocument() async {
        let cache = DocumentContentCache(diskDirectory: nil)
        let stalled = UUID()
        let advancing = UUID()
        cache.put(stalled, entry: .init(filename: "stalled.pdf", text: "partial A"))
        cache.put(advancing, entry: .init(filename: "advancing.pdf", text: "partial B"))
        cache.beginPending(stalled)
        cache.beginPending(advancing)

        let progress = Task.detached {
            for _ in 0..<12 {
                try? await Task.sleep(for: .milliseconds(40))
                cache.reportProgress(advancing)
            }
            cache.finishPending(advancing)
        }

        let started = Date()
        let entry = cache.getAwaitingCompletion(stalled, stallTimeout: 0.15)
        let elapsed = Date().timeIntervalSince(started)
        cache.finishPending(stalled)
        await progress.value

        #expect(entry?.text == "partial A")
        #expect(elapsed < 0.4)
    }

    @Test("Repeated removals each invalidate a publish that straddles them")
    func everyRemovalInvalidates() throws {
        // A generation, not a boolean flag: remove/re-attach/remove must not
        // let a publish from the FIRST attach land after the second removal.
        let dir = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let cache = diskCache(in: dir)

        let id = UUID()
        let firstGeneration = cache.generation(for: id)
        cache.remove(id)
        let secondGeneration = cache.generation(for: id)
        cache.remove(id)

        #expect(!cache.publish(
            id,
            entry: DocumentContentCache.Entry(filename: "a.txt", text: "stale one"),
            ifGenerationIs: firstGeneration
        ))
        #expect(!cache.publish(
            id,
            entry: DocumentContentCache.Entry(filename: "a.txt", text: "stale two"),
            ifGenerationIs: secondGeneration
        ))
        #expect(cache.get(id) == nil)
    }

    @Test("A removal racing a real extraction never leaves the document behind", .timeLimit(.minutes(2)))
    func concurrentRemovalDuringRealExtraction() async throws {
        // The deterministic tests above prove the ordering property. This one
        // exercises the same property through the REAL attach path, where the
        // publish is a background OCR task rather than a direct call — the
        // wiring is what is under test here, not the algorithm.
        let dir = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let cache = diskCache(in: dir)

        let url = try makeScannedPDF(pages: 8)
        defer { try? FileManager.default.removeItem(at: url) }

        let attachment = try ChatFileAttachment(contentsOf: url, cache: cache)
        #expect(cache.hasRegisteredExtraction(attachment.id))

        cache.remove(attachment.id)

        // Well past the point the extraction would have published.
        try? await Task.sleep(for: .seconds(5))
        #expect(cache.get(attachment.id) == nil)
        #expect(!FileManager.default.fileExists(atPath: diskFile(dir, attachment.id).path))
    }

    // MARK: - Cancellation

    /// A PDF whose pages are IMAGES of text, so the only way to read it back
    /// is recognition — the multi-minute path cancellation exists for.
    private func makeScannedPDF(pages: Int) throws -> URL {
        let doc = PDFDocument()
        for index in 0..<pages {
            let size = NSSize(width: 612, height: 300)
            let image = NSImage(size: size)
            image.lockFocus()
            NSColor.white.setFill()
            NSRect(origin: .zero, size: size).fill()
            ("Section \(index) heading text" as NSString).draw(
                in: NSRect(x: 40, y: 40, width: size.width - 80, height: size.height - 80),
                withAttributes: [
                    .font: NSFont.systemFont(ofSize: 28),
                    .foregroundColor: NSColor.black,
                ]
            )
            image.unlockFocus()
            guard let page = PDFPage(image: image) else { continue }
            doc.insert(page, at: doc.pageCount)
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("pdf")
        guard let data = doc.dataRepresentation() else { throw CocoaError(.fileWriteUnknown) }
        try data.write(to: url)
        return url
    }

    @Test("A background extraction registers a handle that can cancel it", .timeLimit(.minutes(2)))
    func extractionIsCancellable() async throws {
        // The regression: both background passes used `Task.detached` and threw
        // the handle away, so `PDFTextRecognizer`'s `Task.isCancelled` check
        // was unreachable and no removal path could stop a running scan.
        let cache = DocumentContentCache(diskDirectory: nil)
        let url = try makeScannedPDF(pages: 8)
        defer { try? FileManager.default.removeItem(at: url) }

        let attachment = try ChatFileAttachment(contentsOf: url, cache: cache)
        // Past the eager OCR window, so a background pass is genuinely running.
        #expect(attachment.totalCharacterCount == nil)
        #expect(cache.hasRegisteredExtraction(attachment.id))

        cache.cancelExtraction(attachment.id)
        #expect(!cache.hasRegisteredExtraction(attachment.id))

        // Cancelling releases any waiter rather than leaving it to time out:
        // the task's own `defer` still clears the pending mark.
        let started = Date()
        _ = cache.getAwaitingCompletion(attachment.id, stallTimeout: 30)
        #expect(Date().timeIntervalSince(started) < 20)
    }

    @Test("Removing a document cancels its extraction and keeps it deleted", .timeLimit(.minutes(2)))
    func removalCancelsAndStaysDeleted() async throws {
        // Ordering matters: a pass still running would call `put` when it
        // finished and re-create the entry `remove` just deleted. Cancelling
        // first — and refusing to publish once cancelled — closes that window.
        let dir = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let cache = diskCache(in: dir)

        let url = try makeScannedPDF(pages: 8)
        defer { try? FileManager.default.removeItem(at: url) }

        let attachment = try ChatFileAttachment(contentsOf: url, cache: cache)
        #expect(cache.hasRegisteredExtraction(attachment.id))

        cache.remove(attachment.id)
        #expect(cache.get(attachment.id) == nil)

        // Give the cancelled pass ample time to reach its publish point. If it
        // published anyway, the document the user deleted would be back.
        try? await Task.sleep(for: .seconds(3))
        #expect(!FileManager.default.fileExists(atPath: diskFile(dir, attachment.id).path))
    }
}
