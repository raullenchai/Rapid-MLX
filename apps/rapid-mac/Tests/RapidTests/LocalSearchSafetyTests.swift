import AppKit
import Foundation
import Testing
import UniformTypeIdentifiers
@testable import Rapid

@Suite struct LocalSearchFileActionTests {
    // Launch Services is unavailable in restricted test processes. Supply its
    // document decision while exercising real lstat and permission checks.
    private func documentAction(_ url: URL) -> LocalSearchFileAction.Action {
        LocalSearchFileAction.classify(type: .plainText)
    }

    private func withDocument(_ test: (URL) throws -> Void) throws {
        let directory = URL(fileURLWithPath: "/private/tmp/local-search-desktop-file-actions")
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let document = directory.appendingPathComponent("ordinary.txt")
        try Data("A document".utf8).write(to: document)
        try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: document.path)
        try test(document)
    }

    private func expectAction(_ expected: LocalSearchFileAction.Action, for url: URL,
                              classified: LocalSearchFileAction.Classification) {
        var opens = 0, reveals = 0
        let action = LocalSearchFileAction.perform(for: url, classified: classified, classifyDocument: documentAction,
            open: { #expect($0 == url); opens += 1; return true },
            reveal: { #expect($0 == url); reveals += 1 })
        #expect(action == expected)
        #expect(opens == (expected == .open ? 1 : 0))
        #expect(reveals == (expected == .reveal ? 1 : 0))
    }

    @Test func unchangedFileOpensAfterRevalidation() throws {
        try withDocument { url in
            let classified = LocalSearchFileAction.classification(for: url, classifyDocument: documentAction)
            #expect(classified.action == .open)
            expectAction(.open, for: url, classified: classified)
        }
    }

    @Test func replacedExecutableRevealsInsteadOfOpening() throws {
        try withDocument { url in
            let classified = LocalSearchFileAction.classification(for: url, classifyDocument: documentAction)
            #expect(classified.action == .open)
            let replacement = url.deletingLastPathComponent().appendingPathComponent("replacement.txt")
            try Data("#!/bin/sh\necho replaced\n".utf8).write(to: replacement)
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: replacement.path)
            try FileManager.default.removeItem(at: url)
            try FileManager.default.moveItem(at: replacement, to: url)
            expectAction(.reveal, for: url, classified: classified)
        }
    }

    @Test func replacedSymlinkRevealsInsteadOfOpening() throws {
        try withDocument { url in
            let classified = LocalSearchFileAction.classification(for: url, classifyDocument: documentAction)
            #expect(classified.action == .open)
            let target = url.deletingLastPathComponent().appendingPathComponent("target.txt")
            try FileManager.default.moveItem(at: url, to: target)
            try FileManager.default.createSymbolicLink(at: url, withDestinationURL: target)
            expectAction(.reveal, for: url, classified: classified)
        }
    }

    @Test func deletedFileIsUnavailableWithoutOpening() throws {
        try withDocument { url in
            let classified = LocalSearchFileAction.classification(for: url, classifyDocument: documentAction)
            #expect(classified.action == .open)
            try FileManager.default.removeItem(at: url)
            expectAction(.unavailable, for: url, classified: classified)
        }
    }

    @Test func replacedDocumentRevealsDespiteSafeType() throws {
        try withDocument { url in
            let classified = LocalSearchFileAction.classification(for: url, classifyDocument: documentAction)
            #expect(classified.action == .open)
            let replacement = url.deletingLastPathComponent().appendingPathComponent("replacement.txt")
            try Data("Another document".utf8).write(to: replacement)
            try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: replacement.path)
            try FileManager.default.removeItem(at: url)
            try FileManager.default.moveItem(at: replacement, to: url)
            #expect(LocalSearchFileAction.classification(for: url, classifyDocument: documentAction).action == .open)
            expectAction(.reveal, for: url, classified: classified)
        }
    }

    @Test func executablePermissionChangeRevealsDespiteSameIdentity() throws {
        try withDocument { url in
            let classified = LocalSearchFileAction.classification(for: url, classifyDocument: documentAction)
            #expect(classified.action == .open)
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
            #expect(LocalSearchFileAction.classification(for: url, classifyDocument: documentAction).identity == classified.identity)
            expectAction(.reveal, for: url, classified: classified)
        }
    }

    @Test func launchableTypesReveal() {
        for type: UTType in [.application, .applicationBundle, .executable, .script, .shellScript,
                             .diskImage, .aliasFile, .symbolicLink, .bundle] {
            #expect(LocalSearchFileAction.classify(type: type) == .reveal)
        }
        for id in ["com.apple.installer-package", "com.apple.installer-meta-package",
                   "com.apple.shortcuts.workflow", "com.apple.automator-workflow"] {
            #expect(LocalSearchFileAction.classify(type: UTType(id)) == .reveal)
        }
        // Conformance, rather than suffix matching, rejects derived script types.
        let script = UTType(exportedAs: "test.rapid.launchable-script", conformingTo: .script)
        #expect(LocalSearchFileAction.classify(type: script) == .reveal)
    }
    @Test func ordinaryDocumentsAndMediaOpen() {
        for type: UTType in [.plainText, .pdf, .png, .jpeg, .mpeg4Movie, .mp3, .rtf] {
            #expect(LocalSearchFileAction.classify(type: type) == .open, "Type: \(type.identifier)")
            #expect(LocalSearchFileAction.classify(type: type, executable: true) == .reveal)
            #expect(LocalSearchFileAction.classify(type: type, indirect: true) == .reveal)
        }
        #expect(LocalSearchFileAction.classify(type: nil) == .reveal)
        #expect(LocalSearchFileAction.classify(type: .data) == .reveal)
        #expect(LocalSearchFileAction.classify(type: .plainText, directory: true) == .reveal)
        #expect(LocalSearchFileAction.action(for: URL(fileURLWithPath: "/no-such-rapid-search-file")) == .unavailable)
    }
    @Test func filesystemLinksAndExecutableDocumentsReveal() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let directory = root.appendingPathComponent(".build/local-search-action-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let document = directory.appendingPathComponent("ordinary.txt")
        try Data("A document".utf8).write(to: document)
        try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: document.path)
        let link = directory.appendingPathComponent("linked.txt")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: document)
        #expect(LocalSearchFileAction.action(for: link) == .reveal)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: document.path)
        #expect(LocalSearchFileAction.action(for: document) == .reveal)
        let broken = directory.appendingPathComponent("broken.txt")
        try FileManager.default.createSymbolicLink(at: broken, withDestinationURL: directory.appendingPathComponent("missing.txt"))
        #expect(LocalSearchFileAction.action(for: broken) == .reveal)
    }

}

/// Records every real URLSession request. A held first POST exercises disable
/// and completion races across the actual transport, including cancellation.
private final class LocalSearchGateTransport: URLProtocol {
    static let log = Log()
    final class Log: @unchecked Sendable {
        let lock = NSLock()
        var requests: [URLRequest] = []
        var held: LocalSearchGateTransport?
        var holdFolder = false
        var failure = false
        var failFolderAt: Int?
        var folderPaths: [String] = []
        var canceled = 0
        func reset(hold: Bool = false, fail: Bool = false, failFolderAt: Int? = nil) {
            lock.lock(); defer { lock.unlock() }
            requests = []; held = nil; holdFolder = hold; failure = fail; canceled = 0
            self.failFolderAt = failFolderAt; folderPaths = []
        }
        var captured: [URLRequest] { lock.lock(); defer { lock.unlock() }; return requests }
        var hasHeld: Bool { lock.lock(); defer { lock.unlock() }; return held != nil }
        var cancelCount: Int { lock.lock(); defer { lock.unlock() }; return canceled }
        var submittedFolders: [String] { lock.lock(); defer { lock.unlock() }; return folderPaths }
        func setFailure(_ value: Bool) { lock.lock(); failure = value; lock.unlock() }
        func release() {
            lock.lock(); let pending = held; held = nil; lock.unlock()
            pending?.respond()
        }
    }
    private var stopped = false // protected by log.lock
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let path = folderPath()
        Self.log.lock.lock()
        Self.log.requests.append(request)
        if let path { Self.log.folderPaths.append(path) }
        let hold = Self.log.holdFolder && request.url?.lastPathComponent == "folders"
        if hold { Self.log.holdFolder = false; Self.log.held = self }
        Self.log.lock.unlock()
        if !hold { respond() }
    }
    private func folderPath() -> String? {
        guard request.url?.lastPathComponent == "folders", request.httpMethod == "POST" else { return nil }
        var data = request.httpBody ?? Data()
        if data.isEmpty, let stream = request.httpBodyStream {
            stream.open(); defer { stream.close() }
            var buffer = [UInt8](repeating: 0, count: 1024)
            while stream.hasBytesAvailable {
                let count = stream.read(&buffer, maxLength: buffer.count)
                guard count > 0 else { break }
                data.append(contentsOf: buffer.prefix(count))
            }
        }
        return (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["path"] as? String
    }
    func respond() {
        Self.log.lock.lock()
        let canceled = stopped
        let failed = Self.log.failure || (request.url?.lastPathComponent == "folders"
            && Self.log.failFolderAt == Self.log.folderPaths.count)
        Self.log.lock.unlock()
        guard !canceled else { return }
        let route = request.url!.lastPathComponent
        let body: String
        if route == "status" {
            body = #"{"enabled":true,"model":{"id":"test","downloaded":true,"loaded":false},"folders":[{"id":"f","path":"/docs","state":"idle","files_indexed":0,"files_skipped":0,"chunks":0}],"indexing":{"active":true,"files_done":0,"files_total":0},"totals":{"files":0,"chunks":0,"index_bytes":0}}"#
        } else if failed { body = #"{"error":{"message":"Operation failed"}}"# }
        else if route == "search" { body = #"{"results":[],"took_ms":0,"mode":"keyword"}"# }
        else { body = "{}" }
        client?.urlProtocol(self, didReceive: HTTPURLResponse(url: request.url!, statusCode: failed && route != "status" ? 500 : 200, httpVersion: nil, headerFields: nil)!, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(body.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {
        Self.log.lock.lock(); stopped = true; Self.log.canceled += 1; Self.log.lock.unlock()
    }
}

@MainActor extension LocalSearchFeatureGateTests {
    private var endpoint: LocalSearchClient.Endpoint { .init(port: 9123, bearer: nil) }
    private func model() -> LocalSearchViewModel {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [LocalSearchGateTransport.self]
        return LocalSearchViewModel(client: .init(session: URLSession(configuration: configuration)))
    }
    private func waitUntil(_ predicate: () -> Bool) async throws {
        for _ in 0..<200 {
            if predicate() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        Issue.record("Timed out waiting for request")
    }
    @Test func batchRetryResumesAtFailedFolderAndSuccessClearsRetry() async {
        let log = LocalSearchGateTransport.log
        log.reset(failFolderAt: 2)
        let model = model()
        await model.appear(endpoint: endpoint)
        await model.add(paths: ["/A", "/B", "/C"])
        #expect(log.submittedFolders == ["/A", "/B"])
        #expect(model.state == .error("Operation failed"))
        await model.retry()
        #expect(log.submittedFolders == ["/A", "/B", "/B", "/C"])
        #expect(model.state == .indexing)
        let requestCount = log.captured.count
        await model.retry()
        // A cleared retry refreshes status rather than retaining the mutation.
        #expect(log.captured.count == requestCount + 1)
        #expect(log.captured.last?.url?.lastPathComponent == "status")
        #expect(log.submittedFolders == ["/A", "/B", "/B", "/C"])
        model.disappear()
    }
    @Test("Disabled Search cancels client activity even if a stale action fires")
    func disabledClientIsInert() async throws {
        let log = LocalSearchGateTransport.log
        log.reset()
        let model = model()
        await model.appear(endpoint: endpoint)
        #expect(log.captured.count == 1)
        model.disappear()
        await model.refresh()
        model.query = "test"; model.searchNow()
        await model.add(paths: ["/disabled"])
        await model.reindex()
        await model.remove(.init(id: "f", path: "/docs", state: .idle, filesIndexed: 0, filesSkipped: 0, chunks: 0, lastIndexedAt: nil, error: nil))
        await model.retry()
        await model.downloadCompleted(UUID())
        try await Task.sleep(for: .milliseconds(30))
        #expect(log.captured.count == 1)
    }
    @Test func disablingDuringBatchCancelsAndDropsPendingWork() async throws {
        let log = LocalSearchGateTransport.log
        log.reset(hold: true)
        let model = model()
        await model.appear(endpoint: endpoint)
        let batch = Task { await model.add(paths: ["/one", "/two", "/three"]) }
        try await waitUntil { log.hasHeld }
        let completion = Task { await model.downloadCompleted(UUID()) }
        await Task.yield()
        model.disappear()
        await batch.value
        await completion.value
        log.release()
        await model.retry()
        try await Task.sleep(for: .milliseconds(30))
        #expect(log.captured.map { $0.url!.lastPathComponent } == ["status", "folders"])
        #expect(log.cancelCount > 0)
        #expect(!model.isMutating)
    }
    @Test func downloadCompletionQueuesBehindMutation() async throws {
        let log = LocalSearchGateTransport.log
        log.reset(hold: true)
        let model = model()
        await model.appear(endpoint: endpoint)
        let batch = Task { await model.add(paths: ["/one", "/two"]) }
        try await waitUntil { log.hasHeld }
        let id = UUID()
        let completion = Task { await model.downloadCompleted(id) }
        await Task.yield()
        #expect(!log.captured.contains { $0.url?.lastPathComponent == "reindex" })
        log.release()
        await batch.value; await completion.value
        await model.downloadCompleted(id)
        #expect(log.captured.filter { $0.url?.lastPathComponent == "reindex" }.count == 1)
        #expect(log.captured.filter { $0.url?.lastPathComponent == "folders" }.count == 2)
        model.disappear()
    }
    @Test func failedDownloadReindexIsNotConsumed() async {
        let log = LocalSearchGateTransport.log
        log.reset(fail: true)
        let model = model(), id = UUID()
        await model.appear(endpoint: endpoint)
        await model.downloadCompleted(id)
        #expect(model.state == .error("Operation failed"))
        log.setFailure(false)
        await model.retry()
        await model.downloadCompleted(id)
        #expect(log.captured.filter { $0.url?.lastPathComponent == "reindex" }.count == 2)
        model.disappear()
    }
    @Test func disappearanceDropsRetainedRetry() async {
        let log = LocalSearchGateTransport.log
        log.reset(fail: true)
        let model = model()
        await model.appear(endpoint: endpoint)
        await model.add(paths: ["/failed"])
        #expect(model.state == .error("Operation failed"))
        model.disappear()
        log.setFailure(false)
        await model.retry()
        await model.appear(endpoint: endpoint)
        await model.retry()
        #expect(log.captured.filter { $0.url?.lastPathComponent == "folders" }.count == 1)
        model.disappear()
    }
    @Test func pollingPreservesMutationErrorAndRetry() async {
        let log = LocalSearchGateTransport.log
        log.reset(fail: true)
        let model = model()
        await model.appear(endpoint: endpoint)
        await model.add(paths: ["/one"])
        #expect(model.state == .error("Operation failed"))
        await model.refresh()
        #expect(model.state == .error("Operation failed"))
        log.setFailure(false)
        await model.retry()
        #expect(model.state == .indexing)
        #expect(log.captured.filter { $0.url?.lastPathComponent == "folders" }.count == 2)
        model.disappear()
    }
    @Test func pollingPreservesSearchError() async throws {
        let log = LocalSearchGateTransport.log
        log.reset(fail: true)
        let model = model()
        await model.appear(endpoint: endpoint)
        model.query = "notes"; model.searchNow()
        try await waitUntil { model.state == .error("Operation failed") }
        await model.refresh()
        #expect(model.state == .error("Operation failed"))
        model.disappear()
    }
}

private final class LocalSearchThumbnailWork: @unchecked Sendable {
    private let lock = NSLock()
    private var paths: [String] = []
    private var completions: [@Sendable (CGImage?) -> Void] = []
    func start(_ path: String, _ completion: @escaping @Sendable (CGImage?) -> Void) {
        lock.lock(); paths.append(path); completions.append(completion); lock.unlock()
    }
    var started: [String] { lock.lock(); defer { lock.unlock() }; return paths }
    func finishAll() {
        lock.lock(); let current = completions; completions = []; lock.unlock()
        for completion in current { completion(nil) }
    }
}

@Suite @MainActor struct LocalSearchThumbnailSafetyTests {
    @Test func boundedDeduplicatedAndCanceledWork() async throws {
        let work = LocalSearchThumbnailWork()
        let cache = LocalSearchThumbnailCache(quickLookLoader: { path, _, completion in work.start(path, completion) })
        let first = Task { await cache.thumbnail(path: "/first", modifiedAt: 0, size: 1) }
        for _ in 0..<100 where work.started.isEmpty { try await Task.sleep(for: .milliseconds(5)) }
        #expect(work.started == ["/first"])
        let duplicate = Task { await cache.thumbnail(path: "/first", modifiedAt: 0, size: 1) }
        let others = (0..<8).map { index in Task { await cache.thumbnail(path: "/other\(index)", modifiedAt: 0, size: 1) } }
        try await Task.sleep(for: .milliseconds(40))
        #expect(work.started.count == 2)
        #expect(work.started.filter { $0 == "/first" }.count == 1)
        first.cancel()
        #expect(await first.value == nil)
        // One consumer remains; cancelling it releases its own continuation.
        duplicate.cancel()
        for other in others { other.cancel() }
        #expect(await duplicate.value == nil)
        for other in others { #expect(await other.value == nil) }
        work.finishAll()
        try await Task.sleep(for: .milliseconds(40))
        #expect(work.started.count == 2) // canceled queued work was never decoded
        let canceled = Task { () -> NSImage? in
            withUnsafeCurrentTask { $0?.cancel() }
            return await cache.thumbnail(path: "/already-canceled", modifiedAt: 0, size: 1)
        }
        #expect(await canceled.value == nil)
        #expect(work.started.count == 2)
    }
    @Test func completedCacheIsBounded() async throws {
        let work = LocalSearchThumbnailWork()
        let context = try #require(CGContext(data: nil, width: 2, height: 2, bitsPerComponent: 8,
            bytesPerRow: 8, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        let frame = try #require(context.makeImage())
        let cache = LocalSearchThumbnailCache(quickLookLoader: { path, _, completion in
            work.start(path, { _ in }); completion(frame)
        }, cacheCountLimit: 2)
        for index in 0..<8 { #expect(await cache.thumbnail(path: "/\(index)", modifiedAt: 0, size: 1) != nil) }
        #expect(work.started.count == 8)
        // More than the configured count cannot all survive in memory.
        for index in 0..<8 { _ = await cache.thumbnail(path: "/\(index)", modifiedAt: 0, size: 1) }
        #expect(work.started.count > 8)
    }
}

@Suite struct LocalSearchCatalogAndStatusTests {
    @Test func groupedStringsHaveNeighbouringLanguages() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("Sources/Rapid/Resources/Localizable.xcstrings"))
        let catalog = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let strings = try #require(catalog["strings"] as? [String: [String: Any]])
        let languages = Set(try #require(strings["Videos"]?["localizations"] as? [String: Any]).keys)
        for key in ["%@ (%@)", "Show all", "Show all %@"] {
            let localized = try #require(strings[key]?["localizations"] as? [String: [String: Any]])
            #expect(Set(localized.keys) == languages)
            let unit = try #require(localized["zh-Hans"]?["stringUnit"] as? [String: String])
            #expect(unit["state"] == "translated")
            #expect(unit["value"]?.contains("Show all") != true)
        }
    }
    @Test func additiveSemanticCapacityIsOptionalAndSummaryIsSingle() throws {
        func decode(_ extras: String) throws -> LocalSearchStatus {
            try LocalSearchClient.decode(LocalSearchStatus.self, from: Data("""
            {"enabled":true,"model":{"id":"test","downloaded":true,"loaded":false},"folders":[],"indexing":{"active":false,"files_done":0,"files_total":0},"totals":{"files":947,"chunks":0,"index_bytes":0\(extras)}}
            """.utf8))
        }
        let legacy = try decode("")
        #expect(legacy.totals.semanticIndexFull == nil)
        #expect(legacy.totals.indexedSummary == String(localized: "\(947) files indexed"))
        let full = try decode(#", "semantic_index_full":true, "embedded_chunks":50000, "semantic_chunk_budget":50000, "keyword_only_chunks":12, "future_field":{"value":42}, "files_by_kind":{"text":877,"image":61,"video":9}"#)
        #expect(full.totals.semanticIndexFull == true)
        #expect(full.totals.embeddedChunks == 50000)
        #expect(full.totals.indexedSummary == full.totals.filesByKind?.label)
        #expect(!full.needsPolling)
        #expect(try decode(#", "semantic_index_full":null"#).totals.semanticIndexFull == nil)
    }
}
