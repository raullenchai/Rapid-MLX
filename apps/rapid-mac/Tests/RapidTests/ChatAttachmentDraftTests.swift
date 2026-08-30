import Foundation
import Testing

@testable import Rapid

@Suite("Chat attachment draft state")
struct ChatAttachmentDraftTests {
    @Test("HEIC picker/drop import becomes a previewable draft image")
    func heicImportUsesSharedDraftBoundary() throws {
        let fixture = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("__Snapshots__/cheetah-logo-96.heic")
        let conversationID = UUID()
        var store = ChatAttachmentDraftStore()
        let startedRequest = store.beginImageImport(conversationID: conversationID)
        let request = try #require(startedRequest)

        let outcome = ChatView.loadImageAttachments([fixture])
        let accepted = store.finishImageImport(
            request: request,
            outcome.accepted,
            notice: outcome.rejection
        )
        let draft = store[conversationID]

        #expect(accepted)
        #expect(draft.notice == nil)
        #expect(draft.images.count == 1)
        #expect(draft.images.first?.filename == "cheetah-logo-96.jpg")
        #expect(draft.images.first?.mimeType == "image/jpeg")
        #expect(draft.sourcePaths.count == 1)
    }

    @Test("async image completion preserves a rejection from the same mixed selection")
    func mixedSelectionNoticeSurvivesImageCompletion() throws {
        let conversationID = UUID()
        var store = ChatAttachmentDraftStore()
        store[conversationID].notice = "Old notice"
        let startedRequest = store.beginImageImport(conversationID: conversationID)
        let request = try #require(startedRequest)
        #expect(store[conversationID].notice == nil)

        store[conversationID].notice = "That file type isn't supported."
        let image = try makeImage(name: "accepted.png")
        let accepted = store.finishImageImport(
            request: request,
            [(image, URL(fileURLWithPath: "/tmp/accepted.png"))],
            notice: nil
        )

        #expect(accepted)
        #expect(store[conversationID].images == [image])
        #expect(store[conversationID].notice == "That file type isn't supported.")
    }

    @Test("consume atomically clears attachments, identity, notice, and import state")
    func consumeClearsEveryTransientField() throws {
        let image = try makeImage(name: "first.png")
        let file = try makeFile(name: "notes.txt", text: "first turn")
        let imageURL = URL(fileURLWithPath: "/tmp/first.png")
        let fileURL = URL(fileURLWithPath: "/tmp/notes.txt")
        var draft = ChatAttachmentDraft()
        draft.appendImage(image, sourceURL: imageURL)
        let startedImportID = draft.beginFileImport()
        let importID = try #require(startedImportID)
        draft.finishFileImport(id: importID, [(file, fileURL)], notice: "old notice")
        _ = draft.beginFileImport()

        let payload = draft.takeSubmission()

        #expect(payload.images == [image])
        #expect(payload.files == [file])
        #expect(!draft.hasAttachments)
        #expect(draft.sourcePaths.isEmpty)
        #expect(draft.notice == nil)
        #expect(!draft.isImportingFiles)
    }

    @Test("a second turn cannot inherit the first turn's image or file")
    func sequentialTurnsRemainIsolated() throws {
        let firstImage = try makeImage(name: "first.png")
        let secondImage = try makeImage(name: "second.png")
        let firstFile = try makeFile(name: "first.txt", text: "alpha")
        let secondFile = try makeFile(name: "second.txt", text: "beta")
        var draft = ChatAttachmentDraft()

        draft.appendImage(firstImage)
        let startedFirstImportID = draft.beginFileImport()
        let firstImportID = try #require(startedFirstImportID)
        draft.finishFileImport(
            id: firstImportID,
            [(firstFile, URL(fileURLWithPath: "/tmp/first.txt"))],
            notice: nil
        )
        let first = draft.takeSubmission()

        draft.appendImage(secondImage)
        let startedSecondImportID = draft.beginFileImport()
        let secondImportID = try #require(startedSecondImportID)
        draft.finishFileImport(
            id: secondImportID,
            [(secondFile, URL(fileURLWithPath: "/tmp/second.txt"))],
            notice: nil
        )
        let second = draft.takeSubmission()

        #expect(first.images.map(\.filename) == ["first.png"])
        #expect(first.files.map(\.filename) == ["first.txt"])
        #expect(second.images.map(\.filename) == ["second.png"])
        #expect(second.files.map(\.filename) == ["second.txt"])
    }

    @Test("removing an attachment also releases its source identity")
    func removalReleasesIdentity() throws {
        let url = URL(fileURLWithPath: "/tmp/reusable.png")
        let image = try makeImage(name: "reusable.png")
        var draft = ChatAttachmentDraft()
        draft.appendImage(image, sourceURL: url)
        #expect(draft.filteringAlreadyAttached([url]).duplicates == 1)

        draft.removeImage(id: image.id)

        #expect(draft.filteringAlreadyAttached([url]).fresh == [url])
    }

    @Test("all input methods share one path-normalized duplicate filter")
    func duplicateFilterCoversExistingAndBatchDuplicates() {
        let existing = URL(fileURLWithPath: "/tmp/docs/photo.png")
        let sameSpelling = URL(fileURLWithPath: "/tmp/docs/../docs/photo.png")
        let fresh = URL(fileURLWithPath: "/tmp/new.txt")
        var draft = ChatAttachmentDraft()
        let image = try? makeImage(name: "photo.png")
        if let image { draft.appendImage(image, sourceURL: existing) }

        let result = draft.filteringAlreadyAttached([sameSpelling, fresh, fresh])

        #expect(result.fresh == [fresh])
        #expect(result.duplicates == 2)
    }

    @Test("a cancelled import cannot resurrect attachments when it completes late")
    func staleImportCompletionIsIgnored() throws {
        let staleFile = try makeFile(name: "old.txt", text: "old conversation")
        let staleURL = URL(fileURLWithPath: "/tmp/old.txt")
        var draft = ChatAttachmentDraft()
        let startedStaleID = draft.beginFileImport()
        let staleID = try #require(startedStaleID)

        let cancelled = draft.cancelFileImport(notice: "Import canceled after navigation.")
        let accepted = draft.finishFileImport(
            id: staleID,
            [(staleFile, staleURL)],
            notice: "stale notice"
        )

        #expect(!accepted)
        #expect(draft.files.isEmpty)
        #expect(cancelled)
        #expect(draft.notice == "Import canceled after navigation.")
        #expect(!draft.isImportingFiles)
        #expect(draft.filteringAlreadyAttached([staleURL]).fresh == [staleURL])
    }

    @Test("a late old generation cannot overwrite a newer import")
    func importGenerationsDoNotCross() throws {
        let oldFile = try makeFile(name: "old.txt", text: "old")
        let newFile = try makeFile(name: "new.txt", text: "new")
        var draft = ChatAttachmentDraft()
        let startedOldID = draft.beginFileImport()
        let oldID = try #require(startedOldID)
        let cancelledOld = draft.cancelFileImport()
        #expect(cancelledOld)
        let startedNewID = draft.beginFileImport()
        let newID = try #require(startedNewID)

        let acceptedOld = draft.finishFileImport(
            id: oldID,
            [(oldFile, URL(fileURLWithPath: "/tmp/old.txt"))],
            notice: nil
        )
        #expect(!acceptedOld)
        #expect(draft.isImportingFiles)
        let acceptedNew = draft.finishFileImport(
            id: newID,
            [(newFile, URL(fileURLWithPath: "/tmp/new.txt"))],
            notice: nil
        )
        #expect(acceptedNew)
        #expect(draft.files == [newFile])
    }

    @Test("a stale generation cannot cancel the newer active import")
    func staleGenerationCannotCancelNewImport() throws {
        var draft = ChatAttachmentDraft()
        let startedOldID = draft.beginFileImport()
        let oldID = try #require(startedOldID)
        _ = draft.cancelFileImport(id: oldID)
        let startedNewID = draft.beginFileImport()
        let newID = try #require(startedNewID)

        let cancelled = draft.cancelFileImport(
            id: oldID,
            notice: "Old conversation changed."
        )

        #expect(!cancelled)
        #expect(draft.fileImportID == newID)
        #expect(draft.notice == nil)
    }

    @Test("conversation drafts import independently and restore by identity")
    func conversationDraftsRemainIsolated() throws {
        let conversationA = UUID()
        let conversationB = UUID()
        let fileA = try makeFile(name: "a.txt", text: "owned by A")
        let fileB = try makeFile(name: "b.txt", text: "owned by B")
        var store = ChatAttachmentDraftStore()

        let startedA = store.beginFileImport(conversationID: conversationA)
        let importA = try #require(startedA)

        let startedB = store.beginFileImport(conversationID: conversationB)
        let importB = try #require(startedB)
        store.finishFileImport(
            request: importB,
            [(fileB, URL(fileURLWithPath: "/tmp/b.txt"))],
            notice: nil
        )

        // A may finish while B is visible; it writes only through A's key.
        let acceptedA = store.finishFileImport(
            request: importA,
            [(fileA, URL(fileURLWithPath: "/tmp/a.txt"))],
            notice: nil
        )

        #expect(acceptedA)
        #expect(store[conversationA].files == [fileA])
        #expect(store[conversationB].files == [fileB])
    }

    @Test("deleted conversations release drafts and reject late imports")
    func deletedConversationCannotBeRecreatedByCompletion() throws {
        let deletedConversation = UUID()
        let survivingConversation = UUID()
        let file = try makeFile(name: "late.txt", text: "late")
        var store = ChatAttachmentDraftStore()
        let startedImport = store.beginFileImport(conversationID: deletedConversation)
        let importID = try #require(startedImport)

        store.retainDrafts(for: [survivingConversation])
        let accepted = store.finishFileImport(
            request: importID,
            [(file, URL(fileURLWithPath: "/tmp/late.txt"))],
            notice: nil
        )

        #expect(!accepted)
        #expect(!store[deletedConversation].hasAttachments)
        #expect(!store[deletedConversation].isImportingFiles)
    }

    @Test("submission is an immutable snapshot of one composer turn")
    func submissionDoesNotFollowLaterDraftMutations() throws {
        let first = try makeImage(name: "first.png")
        let second = try makeImage(name: "second.png")
        var draft = ChatAttachmentDraft()
        draft.appendImage(first)

        let submission = draft.takeSubmission()
        draft.appendImage(second)

        #expect(submission.images == [first])
        #expect(draft.images == [second])
    }

    @Test("appendImage enforces the per-message image count and byte budget")
    func appendImageEnforcesImageBudget() throws {
        var draft = ChatAttachmentDraft()
        // Fill the count slot.
        for index in 0..<ChatImageAttachment.maxImagesPerMessage {
            let appended = draft.appendImage(try makeImage(name: "\(index).png"))
            #expect(appended)
        }
        // Count budget exhausted: a fitted image is rejected.
        let countRejected = draft.appendImage(try makeImage(name: "overflow.png"))
        #expect(!countRejected)
        #expect(draft.images.count == ChatImageAttachment.maxImagesPerMessage)

        // Byte budget exhausted (measured in exact encoded data-URL bytes):
        // each image individually fits the combined budget, but two that
        // together exceed it cannot both be appended.
        let budget = ChatImageAttachment.maxCombinedEncodedImageBytes
        let almostHalf = rawBytesForEncoded(Int(budget * 6 / 10))
        var byteDraft = ChatAttachmentDraft()
        let large = try makeImage(name: "big.png", bytes: almostHalf)
        let largeAccepted = byteDraft.appendImage(large)
        #expect(largeAccepted)
        let another = try makeImage(name: "another.png", bytes: almostHalf)
        let byteRejected = byteDraft.appendImage(another)
        #expect(!byteRejected)
        #expect(byteDraft.images == [large])
    }

    @Test("appendImages keeps only the batch that fits and reports the rejected count")
    func appendImagesGatesMixedBatch() throws {
        var draft = ChatAttachmentDraft()
        // The first image alone fits the encoded byte budget; together the pair
        // exceeds it, so the second is dropped and counted as rejected.
        let budget = ChatImageAttachment.maxCombinedEncodedImageBytes
        let almostHalf = rawBytesForEncoded(Int(budget * 6 / 10))
        let first = try makeImage(name: "first.png", bytes: almostHalf)
        let second = try makeImage(name: "second.png", bytes: almostHalf)

        let rejectedCount = draft.appendImages([
            (first, URL(fileURLWithPath: "/tmp/first.png")),
            (second, URL(fileURLWithPath: "/tmp/second.png")),
        ])
        #expect(rejectedCount == 1)
        #expect(draft.images == [first])
        #expect(draft.sourcePaths.count == 1)
        #expect(draft.sourcePaths[first.id] != nil)
        #expect(draft.sourcePaths[second.id] == nil)
    }

    @Test("post-normalization budget rejection reports its count and reason")
    func finishImageImportReportsLateBudgetRejection() throws {
        let budget = ChatImageAttachment.maxCombinedEncodedImageBytes
        let almostHalf = rawBytesForEncoded(Int(budget * 6 / 10))
        let first = try makeImage(name: "first.png", bytes: almostHalf)
        let second = try makeImage(name: "second.png", bytes: almostHalf)
        var draft = ChatAttachmentDraft()
        let startedImportID = draft.beginImageImport()
        let importID = try #require(startedImportID)

        let accepted = draft.finishImageImport(
            id: importID,
            [
                (first, URL(fileURLWithPath: "/tmp/first.png")),
                (second, URL(fileURLWithPath: "/tmp/second.png")),
            ],
            notice: nil
        )

        #expect(accepted)
        #expect(draft.images == [first])
        #expect(draft.notice?.contains("1 image was not added") == true)
        #expect(draft.notice?.contains("combined size exceeds") == true)
    }

    @Test("duplicate paths are filtered before the budget gate so they consume budget once")
    func duplicatesAreDedupedBeforeBudgetGate() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-dedup-budget-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        // Six references to one path. Dedup must collapse them to a single
        // candidate so the 4-slot count budget is charged once, not six times.
        let file = root.appendingPathComponent("photo.png")
        try Data(repeating: 0, count: 100).write(to: file)
        let batch = Array(repeating: file, count: 6)

        // What addAttachmentURLs does first: dedup the batch.
        let (fresh, duplicates) = ChatAttachmentDraft.withoutAlreadyAttached(
            batch, attached: []
        )
        #expect(fresh == [file])
        #expect(duplicates == 5)

        // Then the image budget gate sees only the single fresh path and accepts
        // it (and, having been deduped, it is charged once).
        let selection = ChatImageAttachment.importCandidates(
            fresh, existingCount: 0, existingBytes: 0
        )
        #expect(selection.accepted == [file])
        #expect(selection.rejectedCount == 0)

        // If dedup had NOT run first, the raw six would consume four slots:
        // prove the gate alone would have under-promised.
        let withoutDedup = ChatImageAttachment.importCandidates(
            batch, existingCount: 0, existingBytes: 0
        )
        #expect(withoutDedup.rejectedCount > 0)
    }

    @Test("a submission (the wire request) never exceeds the accepted, bounded image set")
    func submissionStaysWithinImageBudget() throws {
        var draft = ChatAttachmentDraft()
        // Push far more images than the budget admits; the draft gate holds the
        // count, and takeSubmission snapshots only what fits. The request is
        // built from this snapshot, so it is bounded by construction.
        for index in 0..<ChatImageAttachment.maxImagesPerMessage + 5 {
            _ = draft.appendImage(try makeImage(name: "\(index).png"))
        }
        let submission = draft.takeSubmission()
        #expect(submission.images.count == ChatImageAttachment.maxImagesPerMessage)
        #expect(submission.images.reduce(0) { $0 + $1.encodedDataURLByteCount }
            <= ChatImageAttachment.maxCombinedEncodedImageBytes)
    }

    @Test("a deleted conversation cannot be resurrected by a late image import")
    func deletedConversationRejectsImageCompletion() throws {
        let deletedConversation = UUID()
        let survivingConversation = UUID()
        var store = ChatAttachmentDraftStore()
        let startedImport = store.beginImageImport(conversationID: deletedConversation)
        let importID = try #require(startedImport)

        let late = try makeImage(name: "late.png")
        store.retainDrafts(for: [survivingConversation])
        let accepted = store.finishImageImport(
            request: importID,
            [(late, URL(fileURLWithPath: "/tmp/late.png"))],
            notice: nil
        )

        #expect(!accepted)
        #expect(!store[deletedConversation].hasAttachments)
        #expect(!store[deletedConversation].isImportingFiles)
    }

    @Test("switching conversations mid-image-import cannot land images in the visible draft")
    func imageImportCannotResurrectAcrossConversations() throws {
        let conversationA = UUID()
        let conversationB = UUID()
        let late = try makeImage(name: "late.png")
        var store = ChatAttachmentDraftStore()

        let startedA = store.beginImageImport(conversationID: conversationA)
        let importA = try #require(startedA)
        // Switch to B after import A started.
        store.beginImageImport(conversationID: conversationB)

        let accepted = store.finishImageImport(
            request: importA,
            [(late, URL(fileURLWithPath: "/tmp/late.png"))],
            notice: nil
        )

        #expect(accepted)
        #expect(store[conversationA].images == [late])
        #expect(store[conversationB].images.isEmpty)
    }

    @Test("ChatView gates images before importing and keeps dedup upstream")
    func imageBudgetWiringIntoChatView() throws {
        let source = try String(
            contentsOf: URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("Sources/Rapid/UI/ChatView.swift"),
            encoding: .utf8
        )
        let stripped = CapabilityChipRenderGateSourceGuardTests
            .stripCommentsAndWhitespace(source)

        // The pre-read budget gate runs before any beginImageImport/read work,
        // charging existing images at their encoded wire form so the incoming
        // batch and the already-attached set speak the same byte budget.
        let gated = stripped.contains(
            "letselection=ChatImageAttachment.importCandidates(urls,existingCount:existing.count,existingBytes:existing.reduce(0){$0+$1.encodedDataURLByteCount})"
        )
        #expect(gated, "addImageURLs no longer gates image candidates before import.")
        #expect(stripped.contains(
            "Self.loadImageAttachments(selection.accepted)"
        ), "addImageURLs must load only the budgeted candidates, not the whole selection.")
        #expect(stripped.contains(
            "attachmentDraft.filteringAlreadyAttached(urls)"
        ), "Dedup must remain the first step so duplicates never reach the budget gate twice.")
    }

    @Test("ChatView writes async completion through the originating conversation key")
    func lifecycleIsWiredIntoChatView() throws {
        let source = try String(
            contentsOf: URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("Sources/Rapid/UI/ChatView.swift"),
            encoding: .utf8
        )
        let stripped = CapabilityChipRenderGateSourceGuardTests
            .stripCommentsAndWhitespace(source)

        #expect(stripped.contains(
            "letimportRequest=attachmentDrafts.beginFileImport(conversationID:viewModel.activeConversationID)"
        ))
        #expect(stripped.contains(
            "attachmentDrafts.finishFileImport(request:importRequest,outcome.0,notice:notice)"
        ))
        #expect(stripped.contains(
            ".onChange(of:viewModel.activeConversationID){_,_inpruneAttachmentDrafts()photoCapabilityNotice.dismiss()}"
        ))
        #expect(stripped.contains(".onChange(of:viewModel.conversations.map(\\.id)){_,_inpruneAttachmentDrafts()}"))
    }

    @Test("Photo capability notices stay informational and transient")
    func photoCapabilityNoticeWiring() throws {
        let source = try String(
            contentsOf: URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("Sources/Rapid/UI/ChatView.swift"),
            encoding: .utf8
        )
        let stripped = CapabilityChipRenderGateSourceGuardTests
            .stripCommentsAndWhitespace(source)

        #expect(stripped.contains("@StateprivatevarphotoCapabilityNotice=PhotoCapabilityNotice()"))
        #expect(stripped.contains("InlineNotice(message:message,tone:.info)"))
        #expect(stripped.contains("InlineNotice(message:attachmentNotice,tone:.error)"))
        #expect(stripped.contains("attachmentDraft.notice=nilphotoCapabilityNotice.present(message,availability:photoAvailability)"))
        #expect(!stripped.contains("attachmentDraft.notice=imageInputUnavailableMessage??"))
        #expect(stripped.contains(".onChange(of:alias){_,_inphotoCapabilityNotice.dismiss()}"))
        #expect(stripped.contains(".onChange(of:photoAvailability){_,availabilityinphotoCapabilityNotice.reconcile(with:availability)}"))
        #expect(stripped.contains(".onChange(of:draft){_,_inphotoCapabilityNotice.dismiss()}"))
        #expect(stripped.contains("guardacknowledgeIfNotReady()else{return}photoCapabilityNotice.dismiss()"))
        #expect(stripped.contains("guardacknowledgeIfNotReady()else{returnfalse}photoCapabilityNotice.dismiss()returnviewModel.editUserMessage"))
        #expect(stripped.contains("guardacknowledgeIfNotReady()else{returnfalse}photoCapabilityNotice.dismiss()returnviewModel.retryAssistantMessage"))
        #expect(stripped.contains("privatefuncsendSuggestion(_text:String){guard!viewModel.isStreaming,!attachmentDraft.isImportingFileselse{return}guardacknowledgeIfNotReady()else{return}photoCapabilityNotice.dismiss()"))
        #expect(stripped.contains("privatefuncchoosePhotos(){photoCapabilityNotice.dismiss()letpanel=NSOpenPanel()"))
        #expect(stripped.contains("privatefuncchooseFiles(){photoCapabilityNotice.dismiss()letpanel=NSOpenPanel()"))
    }

    @Test("Photo capability guidance expires when same-alias availability changes")
    func photoCapabilityNoticeReconcilesAvailability() {
        let textOnly = PhotoCapabilityNotice.Availability(
            supportsImageInput: false,
            unavailableMessage: "Photo mode needs more memory."
        )
        var notice = PhotoCapabilityNotice()
        notice.present("Text chat is ready.", availability: textOnly)

        notice.reconcile(with: textOnly)
        #expect(notice.message == "Text chat is ready.")

        notice.reconcile(with: PhotoCapabilityNotice.Availability(
            supportsImageInput: true,
            unavailableMessage: nil
        ))
        #expect(notice.message == nil)

        notice.present("Old remedy", availability: textOnly)
        notice.reconcile(with: PhotoCapabilityNotice.Availability(
            supportsImageInput: false,
            unavailableMessage: "A new lane-specific remedy"
        ))
        #expect(notice.message == nil)
    }

    private func makeImage(name: String, bytes: Int = 4) throws -> ChatImageAttachment {
        try ChatImageAttachment(
            filename: name,
            mimeType: "image/png",
            data: Data(repeating: 0x47, count: bytes)
        )
    }

    /// A raw byte count whose encoded data-URL size is *at most* `encoded`, so
    /// a test can express "this image encodes to ~60 % of the message budget"
    /// without hard-coding base64 expansion.
    private func rawBytesForEncoded(_ encoded: Int) -> Int {
        let prefix = ChatImageAttachment.encodedDataURLByteCount(mimeType: "image/png", rawBytes: 0)
        return ((encoded - prefix) / 4) * 3
    }

    private func makeFile(name: String, text: String) throws -> ChatFileAttachment {
        try ChatFileAttachment(
            filename: name,
            kind: .txt,
            extractedText: text,
            sourceByteCount: text.utf8.count
        )
    }
}
