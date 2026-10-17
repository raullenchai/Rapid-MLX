import AppKit
import Foundation
import Testing
@testable import Rapid

@Suite struct LocalSearchReviewFixTests {
    private let knownReasons = [
        "Unsupported file type", "Too large", "No readable text", "Scanned PDF without text",
        "Password-protected", "Could not be read", "Private or system file", "Audio is not supported yet"
    ]

    @Test func reasonsMapToDistinctCatalogEntriesWithNeighbouringLanguages() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("Sources/Rapid/Resources/Localizable.xcstrings"))
        let catalog = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let strings = try #require(catalog["strings"] as? [String: [String: Any]])
        let languages = Set(try #require(strings["Why files were not indexed"]?["localizations"] as? [String: Any]).keys)
        #expect(Set(knownReasons.map(LocalSearchSkipReason.localized)).count == knownReasons.count)
        #expect(LocalSearchSkipReason.localized("future reason") == String(localized: "File contents could not be indexed"))
        #expect(LocalSearchSkipReason.localized("") == LocalSearchSkipReason.localized("future reason"))
        for reason in knownReasons {
            #expect(LocalSearchSkipReason.localized(reason) == String(localized: String.LocalizationValue(reason)))
            let folder = LocalSearchFolder(id: "f", path: "/files", state: .idle, filesIndexed: 0,
                filesSkipped: 2, chunks: 0, lastIndexedAt: nil, error: nil, skipReasons: [reason: 2])
            #expect(folder.skippedExplanation == "\(LocalSearchSkipReason.localized(reason)): 2")
        }
        let keys = knownReasons + ["File contents could not be indexed", "Matched by file name",
            "Matched by file name — %@", "Indexing paused — it resumes automatically",
            "Indexing paused while a chat response is generating — it resumes automatically"]
        var chineseReasons = Set<String>()
        for key in keys {
            let localizations = try #require(strings[key]?["localizations"] as? [String: [String: Any]])
            #expect(Set(localizations.keys) == languages)
            let unit = try #require(localizations["zh-Hans"]?["stringUnit"] as? [String: String])
            #expect(unit["state"] == "translated")
            let value = try #require(unit["value"])
            #expect(value != key)
            if knownReasons.contains(key) { chineseReasons.insert(value) }
        }
        #expect(chineseReasons.count == knownReasons.count)
    }

    @Test func pausedReasonDecodesOptionallyAndKeepsPolling() throws {
        for (extra, reason) in [("", nil), (#", "paused_reason":null"#, nil),
            (#", "paused_reason":"chat""#, "chat"), (#", "paused_reason":"future""#, "future")] as [(String, String?)] {
            let json = #"{"enabled":true,"model":{"id":"m","downloaded":true,"loaded":true},"folders":[],"indexing":{"active":true,"files_done":2,"files_total":10"#
                + extra + #"},"totals":{"files":2,"chunks":2,"index_bytes":0}}"#
            let status = try LocalSearchClient.decode(LocalSearchStatus.self, from: Data(json.utf8))
            #expect(status.indexing.pausedReason == reason)
            #expect(status.needsPolling)
            let expected: String
            switch reason {
            case nil: expected = String(localized: "Indexing: \(2) of \(10) files")
            case "chat": expected = String(localized: "Indexing paused while a chat response is generating — it resumes automatically")
            default: expected = String(localized: "Indexing paused — it resumes automatically")
            }
            #expect(status.indexing.progressLabel == expected)
        }
        let scanning = LocalSearchStatus.Indexing(active: true, folderId: nil, currentPath: nil, filesDone: 2, filesTotal: 0)
        #expect(scanning.progressLabel == String(localized: "Scanning and indexing: \(2) files processed"))
    }

    @Test func contentIndexingDefaultsAndNameOnlyRowPresentation() throws {
        let base: [String: Any] = ["path": "/files/scan.pdf", "folder_id": "f", "title": "scan",
            "snippet": "Readable excerpt", "highlights": [], "score": 1, "matched": [], "modified_at": 0, "size": 12]
        for field in [nil, NSNull(), true, false] as [Any?] {
            var json = base
            json["content_indexed"] = field
            json["skip_reason"] = "Scanned PDF without text"
            let result = try LocalSearchClient.decode(LocalSearchResult.self, from: JSONSerialization.data(withJSONObject: json))
            let indexed = (field as? Bool) ?? true
            #expect(result.contentIndexed == indexed)
            #expect(result.skipReason == "Scanned PDF without text")
            if indexed {
                #expect(result.nameOnlyExplanation == nil)
                #expect(result.snippetLabel == "Readable excerpt")
            } else {
                #expect(result.snippetLabel == String(localized: "Matched by file name — \(LocalSearchSkipReason.localized("Scanned PDF without text"))"))
            }
        }
        for reason in [nil, NSNull(), "future reason"] as [Any?] {
            var json = base
            json["content_indexed"] = false
            json["snippet"] = ""
            json["skip_reason"] = reason
            let result = try LocalSearchClient.decode(LocalSearchResult.self, from: JSONSerialization.data(withJSONObject: json))
            #expect(result.skipReason == reason as? String)
            #expect(result.snippetLabel == (reason is String
                ? String(localized: "Matched by file name — \(LocalSearchSkipReason.localized("future reason"))")
                : String(localized: "Matched by file name")))
        }
    }
}

@Suite @MainActor struct LocalSearchPreviewTests {
    @MainActor final class Panel: LocalSearchPreviewPanelDriving {
        var isVisible = false
        weak var dataSource: LocalSearchPreviewController?
        weak var delegate: LocalSearchPreviewController?
        var calls: [String] = []
        var onShow: (() -> Void)?
        var defersVisibility = false
        var opening = false
        var dismissal = LocalSearchPreviewDismissal()
        var orderOutCount = 0
        func orderOut() { orderOutCount += 1; isVisible = false }
        func finishOpening() {
            if opening { isVisible = true; opening = false }
            if dismissal.finish(isVisible: isVisible, wantsControl: dataSource != nil) { orderOut() }
        }
        func acquire(_ controller: LocalSearchPreviewController) -> Bool {
            calls.append("acquire"); dataSource = controller; delegate = controller; return true
        }
        func reload() { calls.append("reload") }
        func show() {
            calls.append("show")
            opening = true
            if !defersVisibility { finishOpening() }
            onShow?()
        }
        func release(_ controller: LocalSearchPreviewController) {
            calls.append("release")
            if dismissal.begin(isVisible: isVisible) { orderOut() }
            dataSource = nil; delegate = nil
            _ = dismissal.finish(isVisible: isVisible, wantsControl: false)
        }
    }
    private func nextTurn() async {
        await withCheckedContinuation { continuation in
            DispatchQueue.main.async { continuation.resume() }
        }
    }
    @Test func openingLeavesUpdateTransactionBeforeTouchingPanel() async {
        let panel = Panel()
        let controller = LocalSearchPreviewController(driver: panel)
        controller.toggle(URL(fileURLWithPath: "/tmp/preview.pdf"))
        #expect(panel.calls.isEmpty)
        await nextTurn()
        #expect(panel.calls == ["acquire", "reload", "show"])
        controller.close(); await nextTurn()
    }
    @Test func retainsExactItemIndependentlyOfRequestingState() async {
        let panel = Panel()
        let controller = LocalSearchPreviewController(driver: panel)
        #expect(controller.numberOfPreviewItems(in: nil) == 0)
        var requestingState: URL? = URL(fileURLWithPath: "/tmp/a file.png")
        let expected = requestingState!
        controller.toggle(requestingState!)
        requestingState = nil
        await nextTurn()
        #expect(controller.numberOfPreviewItems(in: nil) == 1)
        #expect(controller.previewPanel(nil, previewItemAt: 0)?.previewItemURL == expected)
        #expect(controller.previewPanel(nil, previewItemAt: 1) == nil)
        controller.close(); await nextTurn()
        #expect(controller.numberOfPreviewItems(in: nil) == 0)
    }
    @Test func toggleSelectionAndDisappearReleaseControl() async {
        let panel = Panel()
        let controller = LocalSearchPreviewController(driver: panel)
        let first = URL(fileURLWithPath: "/tmp/a.pdf")
        let second = URL(fileURLWithPath: "/tmp/b.txt")
        controller.toggle(first); await nextTurn()
        controller.selectionChanged(to: second)
        #expect(controller.item == first as NSURL)
        await nextTurn()
        #expect(controller.item == second as NSURL)
        #expect(panel.calls.last == "reload")
        controller.toggle(first); await nextTurn()
        #expect(panel.isVisible)
        #expect(controller.item == first as NSURL)
        controller.toggle(first); await nextTurn()
        #expect(!panel.isVisible)
        #expect(panel.dataSource == nil && panel.delegate == nil)
        controller.toggle(second); await nextTurn()
        controller.close(); await nextTurn()
        #expect(!panel.isVisible)
        #expect(panel.dataSource == nil && panel.delegate == nil)
        #expect(controller.item == nil)
    }
    @Test func nestedOpeningDoesNotApplyTeardownUntilOpeningReturns() async {
        let panel = Panel()
        let controller = LocalSearchPreviewController(driver: panel)
        panel.onShow = {
            controller.close()
            // Model Quick Look's nested run loop servicing a queued request.
            RunLoop.main.run(until: Date(timeIntervalSinceNow: 0.01))
            #expect(panel.dataSource === controller)
            #expect(controller.item != nil)
        }
        controller.toggle(URL(fileURLWithPath: "/tmp/a.pdf"))
        await nextTurn(); await nextTurn()
        #expect(panel.dataSource == nil && panel.delegate == nil)
    }
    @Test func disappearanceAndAnchorTeardownCloseOnlyOnce() async {
        let panel = Panel()
        let controller = LocalSearchPreviewController(driver: panel)
        controller.toggle(URL(fileURLWithPath: "/tmp/a.png")); await nextTurn()
        controller.close()
        controller.close()
        for _ in 0..<3 { await nextTurn() }
        #expect(panel.calls.filter { $0 == "release" }.count == 1)
        #expect(!panel.isVisible && panel.dataSource == nil && panel.delegate == nil)
    }
    @Test func rapidRequestsKeepOrderAndDoNotReopenAfterDisappear() async {
        let panel = Panel()
        let controller = LocalSearchPreviewController(driver: panel)
        let url = URL(fileURLWithPath: "/tmp/a.pdf")
        controller.toggle(url)
        controller.toggle(url)
        controller.toggle(url)
        controller.close()
        #expect(panel.calls.isEmpty)
        for _ in 0..<5 { await nextTurn() }
        #expect(panel.calls == ["acquire", "reload", "show", "release", "acquire", "reload", "show", "release"])
        #expect(!panel.isVisible && panel.dataSource == nil && panel.delegate == nil)
    }
    @Test func lateOpenOwnedByAnotherControllerIsNotDismissed() {
        for initiallyVisible in [false, true] {
            var dismissal = LocalSearchPreviewDismissal()
            let began = dismissal.begin(isVisible: initiallyVisible)
            #expect(began)
            let shouldOrderOut = dismissal.finish(isVisible: true, wantsControl: false, ownedByOther: true)
            #expect(!shouldOrderOut)
            #expect(!dismissal.closing)
            #expect(!dismissal.awaitingLateOpen)
            let shouldOrderOutAgain = dismissal.finish(isVisible: true, wantsControl: false)
            #expect(!shouldOrderOutAgain)
        }
    }
    @Test func sameFileToggleClosesBeforePanelBecomesVisible() async {
        let panel = Panel()
        panel.defersVisibility = true
        let controller = LocalSearchPreviewController(driver: panel)
        let url = URL(fileURLWithPath: "/tmp/a.txt")
        controller.toggle(url); await nextTurn()
        #expect(!panel.isVisible && panel.opening)
        controller.toggle(url); await nextTurn()
        #expect(panel.opening) // The original open can complete independently of release.
        #expect(panel.orderOutCount == 1) // Cancel even while still hidden.
        panel.finishOpening()
        #expect(panel.orderOutCount == 2) // Dismiss a late completion with no intent.
        #expect(!panel.isVisible)
        #expect(controller.item == nil)
        #expect(panel.dataSource == nil && panel.delegate == nil)
        #expect(panel.calls == ["acquire", "reload", "show", "release"])
        controller.close(); await nextTurn()
    }
    @Test func tripleToggleDuringOpeningRebindsExpectedItem() async {
        let panel = Panel()
        panel.defersVisibility = true
        let controller = LocalSearchPreviewController(driver: panel)
        let url = URL(fileURLWithPath: "/tmp/a.png")
        controller.toggle(url)
        controller.toggle(url)
        controller.toggle(url)
        for _ in 0..<4 { await nextTurn() }
        panel.finishOpening()
        #expect(panel.isVisible)
        #expect(panel.dataSource === controller && panel.delegate === controller)
        #expect(panel.dataSource?.previewPanel(nil, previewItemAt: 0)?.previewItemURL == url)
        #expect(panel.calls == ["acquire", "reload", "show", "release", "acquire", "reload", "show"])
        controller.close(); await nextTurn()
    }
    @Test func selectionDuringOpeningUsesIntendedOpenState() async {
        let panel = Panel()
        panel.defersVisibility = true
        let controller = LocalSearchPreviewController(driver: panel)
        let second = URL(fileURLWithPath: "/tmp/b.txt")
        controller.toggle(URL(fileURLWithPath: "/tmp/a.pdf")); await nextTurn()
        controller.selectionChanged(to: second); await nextTurn()
        panel.finishOpening()
        #expect(controller.item == second as NSURL)
        #expect(panel.calls.last == "reload")
        controller.close(); await nextTurn()
    }
    @Test func nativeDismissalClearsIntentBeforeSameFileToggle() async {
        let panel = Panel()
        let controller = LocalSearchPreviewController(driver: panel)
        let url = URL(fileURLWithPath: "/tmp/a.pdf")
        controller.toggle(url); await nextTurn()
        panel.isVisible = false
        LocalSearchSystemPreviewPanel().didEndControl(controller)
        await nextTurn()
        #expect(controller.item == nil)
        controller.toggle(url); await nextTurn()
        #expect(panel.isVisible && controller.item == url as NSURL)
        controller.close(); await nextTurn()
    }
    @Test func localSearchDoesNotUseSwiftUIPreviewBinding() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let source = try String(contentsOf: root.appendingPathComponent("Sources/Rapid/UI/LocalSearchView.swift"), encoding: .utf8)
        let canonical = SourceGuardSupport.canonicalSource(source, literals: .erase)
        #expect(!canonical.contains(".quickLookPreview("))
    }
}
