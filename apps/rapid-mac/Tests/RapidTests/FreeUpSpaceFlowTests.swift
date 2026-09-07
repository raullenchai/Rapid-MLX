import Foundation
import Testing
@testable import Rapid

@Suite("Computer Use free-up-space flow", .serialized)
struct FreeUpSpaceFlowTests {
    @Test("scan includes only old top-level regular files and orders by size")
    func scanIsNarrowAndDeterministic() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let now = Date(timeIntervalSince1970: 2_000_000_000)
        let old = now.addingTimeInterval(-40 * 24 * 60 * 60)

        try fixture.file("large.zip", bytes: 200, modifiedAt: old)
        try fixture.file("small.txt", bytes: 20, modifiedAt: old)
        try fixture.file("recent.bin", bytes: 1_000, modifiedAt: now)
        try fixture.file(".hidden.log", bytes: 2_000, modifiedAt: old)
        try fixture.file("unfinished.crdownload", bytes: 3_000, modifiedAt: old)
        try fixture.directory("old-folder", modifiedAt: old)
        try fixture.symlink("old-link", destination: fixture.root.appendingPathComponent("large.zip"))

        let service = try #require(MacOSDownloadsCleanupService(root: fixture.root))
        let candidates = try await service.scan(now: now)

        #expect(candidates.map(\.name) == ["large.zip", "small.txt"])
        #expect(candidates.map(\.byteCount) == [200, 20])
        #expect(candidates.allSatisfy { $0.ageInDays(relativeTo: now) >= 39 })
    }

    @Test("candidate count is bounded after deterministic ordering")
    func scanCapKeepsLargestCandidates() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let now = Date(timeIntervalSince1970: 2_000_000_000)
        let old = now.addingTimeInterval(-40 * 24 * 60 * 60)
        try fixture.file("one", bytes: 1, modifiedAt: old)
        try fixture.file("two", bytes: 2, modifiedAt: old)
        try fixture.file("three", bytes: 3, modifiedAt: old)

        let service = try #require(MacOSDownloadsCleanupService(
            root: fixture.root,
            maximumCandidates: 2
        ))
        let candidates = try await service.scan(now: now)

        #expect(candidates.map(\.name) == ["three", "two"])
    }

    @Test("Cocoa and wrapped POSIX access failures become permission guidance")
    func permissionErrorsAreClassified() {
        let cocoa = CocoaError(.fileReadNoPermission)
        #expect(MacOSDownloadsCleanupService.scanError(for: cocoa) == .permissionDenied)

        let wrapped = NSError(
            domain: NSCocoaErrorDomain,
            code: CocoaError.fileReadUnknown.rawValue,
            userInfo: [
                NSUnderlyingErrorKey: NSError(
                    domain: NSPOSIXErrorDomain,
                    code: Int(EPERM)
                )
            ]
        )
        #expect(MacOSDownloadsCleanupService.scanError(for: wrapped) == .permissionDenied)
        #expect(MacOSDownloadsCleanupService.scanError(
            for: CocoaError(.fileReadCorruptFile)
        ) == .enumerationFailed)
    }

    @Test("approved file moves through Trash adapter and original path disappears")
    func approvedMoveIsVerified() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let quarantine = fixture.root.deletingLastPathComponent()
            .appendingPathComponent("quarantine-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: quarantine, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: quarantine) }
        let now = Date(timeIntervalSince1970: 2_000_000_000)
        try fixture.file(
            "approved.zip",
            bytes: 64,
            modifiedAt: now.addingTimeInterval(-40 * 24 * 60 * 60)
        )
        let trash: MacOSDownloadsCleanupService.TrashOperation = { url in
            try FileManager.default.moveItem(
                at: url,
                to: quarantine.appendingPathComponent(url.lastPathComponent)
            )
        }
        let candidateService = MacOSDownloadsCleanupService(
            root: fixture.root,
            trashOperation: trash
        )
        let service = try #require(candidateService)
        let candidate = try #require(try await service.scan(now: now).first)

        let outcome = await service.moveToTrash([candidate])

        #expect(outcome.moved == [candidate])
        #expect(outcome.failures.isEmpty)
        #expect(!FileManager.default.fileExists(atPath: candidate.url.path))
        #expect(FileManager.default.fileExists(
            atPath: quarantine.appendingPathComponent(candidate.name).path
        ))
    }

    @Test("a file replaced after review is left in Downloads")
    func replacedFileFailsClosed() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let now = Date(timeIntervalSince1970: 2_000_000_000)
        let old = now.addingTimeInterval(-40 * 24 * 60 * 60)
        let url = try fixture.file("changed.txt", bytes: 20, modifiedAt: old)
        let trash: MacOSDownloadsCleanupService.TrashOperation = { _ in
            Issue.record("trash must not run for changed identity")
        }
        let candidateService = MacOSDownloadsCleanupService(
            root: fixture.root,
            trashOperation: trash
        )
        let service = try #require(candidateService)
        let reviewed = try #require(try await service.scan(now: now).first)
        try FileManager.default.removeItem(at: url)
        try fixture.file("changed.txt", bytes: 21, modifiedAt: old)

        let outcome = await service.moveToTrash([reviewed])

        #expect(outcome.moved.isEmpty)
        #expect(outcome.failures == [FreeUpSpaceMoveFailure(
            candidate: reviewed,
            reason: .changedSinceReview
        )])
        #expect(FileManager.default.fileExists(atPath: url.path))
    }

    @Test("a successful adapter return is not accepted without the move postcondition")
    func noOpTrashAdapterFailsVerification() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let now = Date(timeIntervalSince1970: 2_000_000_000)
        try fixture.file(
            "still-here.txt",
            bytes: 20,
            modifiedAt: now.addingTimeInterval(-40 * 24 * 60 * 60)
        )
        let noOp: MacOSDownloadsCleanupService.TrashOperation = { _ in }
        let candidateService = MacOSDownloadsCleanupService(
            root: fixture.root,
            trashOperation: noOp
        )
        let service = try #require(candidateService)
        let reviewed = try #require(try await service.scan(now: now).first)

        let outcome = await service.moveToTrash([reviewed])

        #expect(outcome.moved.isEmpty)
        #expect(outcome.failures == [FreeUpSpaceMoveFailure(
            candidate: reviewed,
            reason: .verificationFailed
        )])
        #expect(FileManager.default.fileExists(atPath: reviewed.url.path))
    }

    @Test("Trash permission failures are reported without claiming a move")
    func trashPermissionFailureIsClassified() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let now = Date(timeIntervalSince1970: 2_000_000_000)
        try fixture.file(
            "protected.txt",
            bytes: 20,
            modifiedAt: now.addingTimeInterval(-40 * 24 * 60 * 60)
        )
        let denied: MacOSDownloadsCleanupService.TrashOperation = { _ in
            throw NSError(
                domain: NSCocoaErrorDomain,
                code: CocoaError.fileWriteUnknown.rawValue,
                userInfo: [
                    NSUnderlyingErrorKey: NSError(
                        domain: NSPOSIXErrorDomain,
                        code: Int(EACCES)
                    )
                ]
            )
        }
        let service = try #require(MacOSDownloadsCleanupService(
            root: fixture.root,
            trashOperation: denied
        ))
        let reviewed = try #require(try await service.scan(now: now).first)

        let outcome = await service.moveToTrash([reviewed])

        #expect(outcome.moved.isEmpty)
        #expect(outcome.failures == [FreeUpSpaceMoveFailure(
            candidate: reviewed,
            reason: .permissionDenied
        )])
        #expect(FileManager.default.fileExists(atPath: reviewed.url.path))
    }

    @Test("a crafted candidate outside the reviewed root fails closed")
    func candidateOutsideRootIsRejected() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let outside = try Fixture()
        defer { outside.remove() }
        let now = Date(timeIntervalSince1970: 2_000_000_000)
        let old = now.addingTimeInterval(-40 * 24 * 60 * 60)
        try fixture.file("inside.txt", bytes: 20, modifiedAt: old)
        let outsideURL = try outside.file("outside.txt", bytes: 20, modifiedAt: old)
        let service = try #require(MacOSDownloadsCleanupService(root: fixture.root))
        let outsideService = try #require(MacOSDownloadsCleanupService(root: outside.root))
        let crafted = try #require(try await outsideService.scan(now: now).first)

        let outcome = await service.moveToTrash([crafted])

        #expect(outcome.moved.isEmpty)
        #expect(outcome.failures.first?.reason == .changedSinceReview)
        #expect(FileManager.default.fileExists(atPath: outsideURL.path))
    }

    @MainActor
    @Test("review requires selection and confirmation before move")
    func viewModelRequiresTwoExplicitSteps() async throws {
        let candidate = sampleCandidate()
        let expected = FreeUpSpaceMoveOutcome(moved: [candidate])
        let model = FreeUpSpaceFlowViewModel(
            service: ImmediateFreeUpSpaceService(
                candidates: [candidate],
                outcome: expected
            ),
            now: { Date(timeIntervalSince1970: 2_000_000_000) }
        )

        model.scan()
        try await waitUntil { model.phase == .reviewing }
        #expect(model.selectedCandidates.isEmpty)
        model.reviewMove()
        #expect(model.phase == .reviewing)

        model.toggle(candidate)
        model.reviewMove()
        #expect(model.phase == .confirming)
        model.returnToSelection()
        #expect(model.phase == .reviewing)

        model.reviewMove()
        model.moveSelectedToTrash()
        try await waitUntil { model.phase == .finished(expected) }
        #expect(model.phase == .finished(expected))
    }

    @MainActor
    @Test("stopping a scan reports cancellation instead of an empty result")
    func stoppingScanIsNotReportedAsEmpty() async throws {
        let model = FreeUpSpaceFlowViewModel(service: CancellableScanService())

        model.scan()
        await Task.yield()
        model.stop()

        try await waitUntil { model.phase == .failed(.cancelled) }
        #expect(model.candidates.isEmpty)
    }

    private func sampleCandidate() -> FreeUpSpaceCandidate {
        FreeUpSpaceCandidate(
            url: URL(fileURLWithPath: "/tmp/old.zip"),
            identity: FreeUpSpaceFileIdentity(
                device: 1,
                inode: 2,
                size: 100,
                modifiedSeconds: 1_000,
                modifiedNanoseconds: 0
            ),
            modifiedAt: Date(timeIntervalSince1970: 1_000)
        )
    }

    @MainActor
    private func waitUntil(
        _ condition: @escaping @MainActor () -> Bool
    ) async throws {
        let clock = ContinuousClock()
        let deadline = clock.now + .seconds(2)
        while !condition(), clock.now < deadline { await Task.yield() }
        guard condition() else { throw TestWaitError.timedOut }
    }
}

private enum TestWaitError: Error { case timedOut }

private struct ImmediateFreeUpSpaceService: FreeUpSpaceServicing {
    let candidates: [FreeUpSpaceCandidate]
    let outcome: FreeUpSpaceMoveOutcome

    func scan(now: Date) async throws -> [FreeUpSpaceCandidate] { candidates }

    func moveToTrash(
        _ candidates: [FreeUpSpaceCandidate]
    ) async -> FreeUpSpaceMoveOutcome { outcome }
}

private struct CancellableScanService: FreeUpSpaceServicing {
    func scan(now: Date) async throws -> [FreeUpSpaceCandidate] {
        try await Task.sleep(for: .seconds(10))
        return []
    }

    func moveToTrash(
        _ candidates: [FreeUpSpaceCandidate]
    ) async -> FreeUpSpaceMoveOutcome {
        FreeUpSpaceMoveOutcome()
    }
}

private struct Fixture {
    let root: URL

    init() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-free-space-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    @discardableResult
    func file(_ name: String, bytes: Int, modifiedAt: Date) throws -> URL {
        let url = root.appendingPathComponent(name)
        try Data(repeating: 0x41, count: bytes).write(to: url)
        try FileManager.default.setAttributes(
            [.modificationDate: modifiedAt],
            ofItemAtPath: url.path
        )
        return url
    }

    func directory(_ name: String, modifiedAt: Date) throws {
        let url = root.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        try FileManager.default.setAttributes(
            [.modificationDate: modifiedAt],
            ofItemAtPath: url.path
        )
    }

    func symlink(_ name: String, destination: URL) throws {
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent(name),
            withDestinationURL: destination
        )
    }

    func remove() {
        try? FileManager.default.removeItem(at: root)
    }
}
