import Foundation
import Testing
@testable import Rapid

/// The same file must not attach twice.
///
/// It is not merely redundant. `ChatFileAttachment.fittedForMessage` splits
/// the per-message character budget evenly across attachments, so one PDF
/// added four times sends a quarter of it four times over instead of the whole
/// document once — and the only sign is a "partial" chip on each row.
///
/// Paste, drag-and-drop and the open panel all funnel through the same import,
/// so the guard sits there rather than on any one gesture — and above the
/// image/document split, so both kinds get the same answer.
///
/// One case is deliberately NOT covered: an image pasted as raw clipboard
/// bytes rather than as a file URL. It arrives with no path, so path identity
/// has nothing to compare; deduping it would mean hashing pixels, which is a
/// different decision from the one made here.
@Suite("Attachment de-duplication")
struct AttachmentDedupTests {

    private func url(_ path: String) -> URL { URL(fileURLWithPath: path) }

    /// A filter nothing calls is not a filter.
    ///
    /// The cases below all exercise ``ChatView/withoutAlreadyAttached`` in
    /// isolation, so deleting its one call site leaves every one of them
    /// green — verified by doing exactly that. This session has now produced
    /// three bugs of that shape (a window floor declared but never applied, a
    /// jump-to-bottom button whose two halves were each correct), so the
    /// wiring gets pinned separately from the logic.
    ///
    /// ViewInspector is not in this target (#1492), hence a source guard.
    @Test("The filter is wired into the one path every gesture funnels through")
    func filterIsWiredIntoTheImport() throws {
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
        // Reduced to a Bool first: `#expect` prints the expression it is
        // handed, and the stripped source is the whole of ChatView.
        let wired = stripped.contains(
            "Self.withoutAlreadyAttached(urls,attached:Set(attachedSourcePaths.values))"
        )
        #expect(
            wired,
            """
            ChatView no longer filters incoming URLs through             withoutAlreadyAttached before splitting them into images and             documents. Every add gesture funnels through addAttachmentURLs,             so removing it there re-opens duplicate attachment on all of them.
            """
        )
    }

    @Test("A file already attached is rejected, and counted")
    func rejectsAlreadyAttached() {
        let attached: Set<String> = [ChatView.attachmentKey(for: url("/tmp/report.pdf"))]
        let (fresh, duplicates) = ChatView.withoutAlreadyAttached(
            [url("/tmp/report.pdf")], attached: attached
        )
        #expect(fresh.isEmpty)
        #expect(duplicates == 1)
    }

    /// Selecting one file twice in the open panel is the same mistake as
    /// pasting it twice, and arrives as a single batch.
    @Test("Repeats inside one batch collapse")
    func collapsesRepeatsWithinABatch() {
        let (fresh, duplicates) = ChatView.withoutAlreadyAttached(
            [url("/tmp/a.pdf"), url("/tmp/a.pdf"), url("/tmp/b.csv")],
            attached: []
        )
        #expect(fresh.map(\.lastPathComponent) == ["a.pdf", "b.csv"])
        #expect(duplicates == 1)
    }

    /// Two spellings of one path are one file. Without this, `./a.pdf` and
    /// `a.pdf` reached from different working directories both attach.
    @Test("Path spelling does not create a second file")
    func normalisesPathSpelling() {
        let attached: Set<String> = [ChatView.attachmentKey(for: url("/tmp/docs/a.pdf"))]
        let (fresh, _) = ChatView.withoutAlreadyAttached(
            [url("/tmp/docs/../docs/a.pdf")], attached: attached
        )
        #expect(fresh.isEmpty, "a non-normalised path attached the same file again")
    }

    /// The same bytes at two real paths stay two attachments. Deciding
    /// otherwise would mean reading every candidate before knowing whether we
    /// want it — this is the cost of path identity, recorded rather than
    /// discovered later.
    @Test("Distinct paths remain distinct")
    func distinctPathsAreKept() {
        let attached: Set<String> = [ChatView.attachmentKey(for: url("/tmp/a.pdf"))]
        let (fresh, duplicates) = ChatView.withoutAlreadyAttached(
            [url("/tmp/copy/a.pdf")], attached: attached
        )
        #expect(fresh.count == 1)
        #expect(duplicates == 0)
    }

    /// Images and documents share one filter, so an image already attached is
    /// rejected on the same terms — the split into the two lists happens after.
    @Test("The filter does not care whether the file is an image or a document")
    func imagesAndDocumentsShareTheFilter() {
        let attached: Set<String> = [
            ChatView.attachmentKey(for: url("/tmp/shot.png")),
            ChatView.attachmentKey(for: url("/tmp/report.pdf")),
        ]
        let (fresh, duplicates) = ChatView.withoutAlreadyAttached(
            [url("/tmp/shot.png"), url("/tmp/report.pdf"), url("/tmp/new.csv")],
            attached: attached
        )
        #expect(fresh.map(\.lastPathComponent) == ["new.csv"])
        #expect(duplicates == 2)
    }

    @Test("Nothing attached yet lets everything through")
    func emptyAttachedSetAcceptsAll() {
        let (fresh, duplicates) = ChatView.withoutAlreadyAttached(
            [url("/tmp/a.pdf"), url("/tmp/b.pdf")], attached: []
        )
        #expect(fresh.count == 2)
        #expect(duplicates == 0)
    }

    @Test("A failed import does not shift source paths onto later attachments")
    func failedImportPreservesSourceAssociation() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: directory) }

        let invalid = directory.appendingPathComponent("broken.txt")
        let valid = directory.appendingPathComponent("answer.txt")
        try Data([0xFF]).write(to: invalid)
        try Data("hello".utf8).write(to: valid)

        let outcome = ChatView.loadFileAttachments([invalid, valid])

        #expect(outcome.accepted.count == 1)
        #expect(outcome.accepted.first?.attachment.filename == "answer.txt")
        #expect(outcome.accepted.first?.sourceURL == valid)
        #expect(outcome.rejection != nil)
    }
}
