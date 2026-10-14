import AppKit
import SwiftUI
import Foundation
import Testing
@testable import Rapid

@Suite @MainActor struct LocalSearchThumbnailTests {
    private func frame(_ color: NSColor) throws -> CGImage {
        let context = try #require(CGContext(data: nil, width: 32, height: 32, bitsPerComponent: 8,
            bytesPerRow: 128, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.setFillColor(color.cgColor)
        context.fill(CGRect(x: 0, y: 0, width: 32, height: 32))
        return try #require(context.makeImage())
    }

    @Test func matchedVideoFramesAndCacheIncludeTimestampAndDimensions() async throws {
        let blue = try frame(.blue), pink = try frame(.systemPink)
        let cache = LocalSearchThumbnailCache(frameLoader: { path, time, size in
            #expect(!Thread.isMainThread)
            #expect(path == "/clip.mov")
            return time == 0 ? blue : pink
        }, quickLookLoader: { _, _, completion in
            Issue.record("Matched frame must bypass Quick Look")
            completion(nil)
        })
        let first = try #require(await cache.thumbnail(path: "/clip.mov", modifiedAt: 0, size: 1, timestamp: 0))
        let later = try #require(await cache.thumbnail(path: "/clip.mov", modifiedAt: 0, size: 1, timestamp: 1))
        func color(_ image: NSImage) throws -> NSColor {
            let cg = try #require(image.cgImage(forProposedRect: nil, context: nil, hints: nil))
            return try #require(NSBitmapImageRep(cgImage: cg).colorAt(x: 16, y: 16)?.usingColorSpace(.deviceRGB))
        }
        #expect(try color(first).blueComponent > 0.9)
        #expect(try color(first).redComponent < 0.1)
        #expect(try color(later).redComponent > 0.8)
        #expect(first !== later)
        let cached = await cache.thumbnail(path: "/clip.mov", modifiedAt: 0, size: 1, timestamp: 0)
        #expect(cached === first)
        let smaller = try #require(await cache.thumbnail(path: "/clip.mov", modifiedAt: 0, size: 1,
            timestamp: 0, pixelSize: CGSize(width: 16, height: 16)))
        #expect(smaller !== first)
        let changed = await cache.thumbnail(path: "/clip.mov", modifiedAt: 1, size: 2, timestamp: 0)
        #expect(changed !== first)
    }

    @Test func failedVideoDecodeFallsBackToQuickLook() async throws {
        let fallback = try frame(.systemPink)
        let cache = LocalSearchThumbnailCache(frameLoader: { _, _, _ in
            #expect(!Thread.isMainThread)
            return nil
        }, quickLookLoader: { path, size, completion in
            #expect(!Thread.isMainThread)
            #expect(path == "/clip.mov")
            #expect(size == CGSize(width: 320, height: 240))
            completion(fallback)
        })
        let image = await cache.thumbnail(path: "/clip.mov", modifiedAt: 0, size: 1, timestamp: 0)
        #expect(image != nil)
    }
}

private func searchStatus(folders: [LocalSearchFolder] = [], downloaded: Bool = true, active: Bool = false) -> LocalSearchStatus {
    .init(enabled: true, model: .init(id: "embeddinggemma-2-4bit", downloaded: downloaded, loaded: false),
          folders: folders, indexing: .init(active: active, folderId: nil, currentPath: nil, filesDone: 0, filesTotal: 0),
          totals: .init(files: 0, chunks: 0, indexBytes: 0))
}
private func searchFolder(state: LocalSearchFolder.State = .idle) -> LocalSearchFolder {
    .init(id: "folder", path: "/tmp/docs", state: state, filesIndexed: 0, filesSkipped: 0, chunks: 0, lastIndexedAt: nil, error: nil)
}
private func searchResult(_ title: String) -> LocalSearchResult {
    .init(path: "/tmp/\(title)", folderId: "folder", title: title, snippet: "A file", highlights: [], score: 0.5,
          matched: ["keyword"], page: nil, line: nil, modifiedAt: 0, size: 12)
}

@Suite struct LocalSearchPresentationTests {
    @Test(arguments: [0, 1, 2, 3], [false, true])
    func accessibilityLabelOmitsAbsentComponents(locator: Int, nameOnly: Bool) {
        let page = locator == 1 || locator == 3 ? 2 : nil
        let line = locator == 2 || locator == 3 ? 7 : nil
        var result = LocalSearchResult(path: "/tmp/docs/locked_memo.pdf", folderId: "folder",
            title: "locked_memo.pdf", snippet: "A file", highlights: [], score: 0.5,
            matched: ["keyword"], page: page, line: line, modifiedAt: 0, size: 12)
        result.contentIndexed = !nameOnly
        result.skipReason = nameOnly ? "Password-protected" : nil
        let date = Date(timeIntervalSince1970: 0).formatted(date: .abbreviated, time: .omitted)
        var expected = ["locked_memo.pdf"]
        if let page { expected.append(String(localized: "page \(page)")) }
        if let line { expected.append(String(localized: "line \(line)")) }
        expected.append(nameOnly
            ? String(localized: "Matched by file name — \(LocalSearchSkipReason.localized("Password-protected"))")
            : "A file")
        expected.append(String(localized: "in \("docs")"))
        expected.append(String(localized: "modified \(date)"))
        #expect(result.accessibilityLabel(folders: [searchFolder()]) == expected.joined(separator: ", "))
        #expect(!result.accessibilityLabel(folders: [searchFolder()]).contains(", ,"))
    }

    @Test func accessibilityLabelOmitsEmptySnippet() {
        let result = LocalSearchResult(path: "/tmp/docs/memo.txt", folderId: "folder",
            title: "memo.txt", snippet: " \n", highlights: [], score: 0.5,
            matched: ["keyword"], page: nil, line: nil, modifiedAt: 0, size: 12)
        let date = Date(timeIntervalSince1970: 0).formatted(date: .abbreviated, time: .omitted)
        #expect(result.accessibilityLabel(folders: [searchFolder()]) ==
            ["memo.txt", String(localized: "in \("docs")"), String(localized: "modified \(date)")].joined(separator: ", "))
    }

    @Test func folderShortcutsResolveSystemURLsAndWholeMacCoverage() throws {
        #expect(LocalSearchFolderShortcut.desktop.url == FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask).first)
        #expect(LocalSearchFolderShortcut.documents.url == FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first)
        #expect(LocalSearchFolderShortcut.downloads.url == FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first)
        #expect(LocalSearchFolderShortcut.wholeMac.url?.path == "/")
        #expect(LocalSearchPath.folderTitle("/") == String(localized: "Whole Mac"))
        let wholeMac = LocalSearchFolder(id: "root", path: "/", state: .idle, filesIndexed: 0, filesSkipped: 0, chunks: 0, lastIndexedAt: nil, error: nil)
        let home = LocalSearchFolder(id: "home", path: FileManager.default.homeDirectoryForCurrentUser.path,
                                     state: .idle, filesIndexed: 0, filesSkipped: 0, chunks: 0, lastIndexedAt: nil, error: nil)
        for shortcut in [LocalSearchFolderShortcut.desktop, .documents, .downloads] {
            let url = try #require(shortcut.url)
            #expect(LocalSearchPath.isCovered(url, folders: [wholeMac]))
            #expect(LocalSearchPath.isCovered(url, folders: [home]))
        }
        let docs = searchFolder()
        #expect(LocalSearchPath.isCovered(URL(fileURLWithPath: "/tmp/docs"), folders: [docs]))
        #expect(LocalSearchPath.isCovered(URL(fileURLWithPath: "/tmp/docs/nested"), folders: [docs]))
        #expect(!LocalSearchPath.isCovered(URL(fileURLWithPath: "/tmp/docs-other"), folders: [docs]))
        #expect(!LocalSearchPath.isCovered(URL(fileURLWithPath: "/"), folders: [docs]))
        #expect(LocalSearchFolderShortcut.allCases.map(\.identifier) == ["LocalSearch.AddFolder.Desktop", "LocalSearch.AddFolder.Documents", "LocalSearch.AddFolder.Downloads", "LocalSearch.AddFolder.WholeMac"])
    }

    @Test func abbreviatesOnlyTheHomeDirectoryAndItsDescendants() {
        let home = URL(fileURLWithPath: "/Users/test")
        #expect(LocalSearchPath.abbreviated("/Users/test", home: home) == "~")
        #expect(LocalSearchPath.abbreviated("/Users/test/Documents/My Files/configuration.md", home: home) == "~/Documents/My Files/configuration.md")
        #expect(LocalSearchPath.abbreviated("/Users/test-other/file.md", home: home) == "/Users/test-other/file.md")
        #expect(LocalSearchPath.abbreviated("/tmp/Users/test/file.md", home: home) == "/tmp/Users/test/file.md")
        #expect(LocalSearchPath.abbreviated("relative/file.md", home: home) == "relative/file.md")
        #expect(LocalSearchPath.abbreviated("~/Documents/file.md", home: home) == "~/Documents/file.md")
    }

    @Test func defaultAbbreviationUsesRealUserHome() {
        let home = FileManager.default.homeDirectoryForCurrentUser
        #expect(LocalSearchPath.abbreviated(home.path) == "~")
        #expect(LocalSearchPath.abbreviated(home.appendingPathComponent("Documents/a.md").path) == "~/Documents/a.md")
    }

    @Test func automaticallyCollapsesOnlyForAnActiveQueryWithResults() {
        let disclosure = LocalSearchFoldersDisclosure()
        #expect(disclosure.isExpanded(query: "", hasResults: false))
        #expect(disclosure.isExpanded(query: "files", hasResults: false))
        #expect(!disclosure.isExpanded(query: "files", hasResults: true))
        #expect(disclosure.isExpanded(query: " \n\t", hasResults: true))
        #expect(disclosure.isExpanded(query: "", hasResults: false))
    }

    @Test func explicitExpansionAndCollapseWinForTheSession() {
        var disclosure = LocalSearchFoldersDisclosure()
        disclosure.toggle(query: "files", hasResults: true)
        #expect(disclosure.isExpanded(query: "files", hasResults: true))
        #expect(disclosure.isExpanded(query: "another query", hasResults: true))
        disclosure.toggle(query: "files", hasResults: true)
        #expect(!disclosure.isExpanded(query: "", hasResults: false))
        #expect(!disclosure.isExpanded(query: "files", hasResults: false))

        var initiallyExpanded = LocalSearchFoldersDisclosure()
        initiallyExpanded.toggle(query: "", hasResults: false)
        #expect(!initiallyExpanded.isExpanded(query: "files", hasResults: true))
        #expect(!initiallyExpanded.isExpanded(query: "", hasResults: false))
    }
}

@Suite struct LocalSearchDecodingTests {
    @Test func nullFieldsAndUnknownState() throws {
        let data = Data(#"{"enabled":true,"model":{"id":"embeddinggemma-2-4bit","downloaded":false,"loaded":false},"folders":[{"id":"f","path":"/tmp","state":"future_state","files_indexed":2,"files_skipped":1,"chunks":3,"last_indexed_at":null,"error":null}],"indexing":{"active":false,"folder_id":null,"current_path":null,"files_done":0,"files_total":0},"totals":{"files":2,"chunks":3,"index_bytes":42}}"#.utf8)
        let status = try LocalSearchClient.decode(LocalSearchStatus.self, from: data)
        #expect(status.folders.first?.state == .unknown)
        #expect(status.folders.first?.lastIndexedAt == nil)
        #expect(status.folders.first?.error == nil)
        #expect(status.indexing.folderId == nil)
        #expect(status.indexing.currentPath == nil)
        #expect(status.totals.indexBytes == 42)
        #expect(!status.model.downloaded)
    }
    @Test func resultResponse() throws {
        let data = Data(#"{"results":[{"path":"/tmp/a.pdf","folder_id":"f","title":"a","snippet":"hello","highlights":[[0,5]],"score":0.7,"matched":["semantic","keyword"],"page":null,"line":null,"modified_at":123.5,"size":20}],"took_ms":1.25,"mode":"hybrid"}"#.utf8)
        let response = try LocalSearchClient.decode(LocalSearchResponse.self, from: data)
        #expect(response.results.first?.folderId == "f")
        #expect(response.results.first?.page == nil)
        #expect(response.results.first?.line == nil)
        #expect(response.results.first?.modifiedAt == 123.5)
        #expect(response.tookMs == 1.25)
        #expect(response.mode == "hybrid")
        #expect(response.groups == nil)
    }
    @Test func pdfPageLocator() throws {
        let data = Data(#"{"results":[{"path":"/tmp/a.pdf","folder_id":"f","title":"a","snippet":"Second","highlights":[],"score":0.7,"matched":["keyword"],"page":2,"line":null,"modified_at":123.5,"size":20}],"took_ms":1.25,"mode":"keyword"}"#.utf8)
        let response = try LocalSearchClient.decode(LocalSearchResponse.self, from: data)
        #expect(response.results.first?.page == 2)
        #expect(response.results.first?.line == nil)
    }
    @Test func folderStates() throws {
        for state in ["idle", "queued", "indexing", "waiting_for_model", "missing", "error"] {
            let decoded = try JSONDecoder().decode(LocalSearchFolder.State.self, from: JSONEncoder().encode(state))
            #expect(decoded != .unknown)
        }
    }
}

@Suite struct LocalSearchHighlightTests {
    @Test func clipsAndMergesMalformedRanges() {
        #expect(LocalSearchHighlights.segments("abcdef", ranges: [[-4,2], [1,4], [4,100], [9,10], [4,1], [1], []]) == [.init(text: "abcdef", bold: true)])
        #expect(LocalSearchHighlights.segments("abcdef", ranges: [[2,4]]) == [
            .init(text: "ab", bold: false), .init(text: "cd", bold: true), .init(text: "ef", bold: false)])
        #expect(LocalSearchHighlights.segments("", ranges: [[0,Int.max]]) == [])
    }
    @Test func unicodeOffsets() {
        // Python character offsets: emoji is one scalar, combining accent another.
        #expect(LocalSearchHighlights.segments("a😀e\u{301}z", ranges: [[1,4]]) == [
            .init(text: "a", bold: false), .init(text: "😀e\u{301}", bold: true), .init(text: "z", bold: false)])
    }
}

@MainActor private final class LocalSearchTestBox<Value> {
    var value: Value
    init(_ value: Value) { self.value = value }
}

@Suite @MainActor struct LocalSearchViewModelTests {
    private let endpoint = LocalSearchClient.Endpoint(port: 8000, bearer: "test")
    @Test func emptyAndReadinessStates() async {
        let snapshot = LocalSearchTestBox(searchStatus())
        let model = LocalSearchViewModel(fetchStatus: { _ in snapshot.value })
        #expect(model.state == .engineStopped)
        await model.appear(endpoint: nil)
        #expect(model.state == .engineStopped)
        await model.connect(endpoint)
        #expect(model.state == .noFolders)
        snapshot.value = searchStatus(folders: [searchFolder()], downloaded: false)
        await model.refresh()
        #expect(model.state == .modelMissing)
        snapshot.value = searchStatus(folders: [searchFolder()])
        await model.refresh()
        #expect(model.state == .ready)
        for state: LocalSearchFolder.State in [.queued, .indexing] {
            snapshot.value = searchStatus(folders: [searchFolder(state: state)])
            await model.refresh()
            #expect(model.state == .indexing)
        }
        snapshot.value = searchStatus(folders: [searchFolder()], active: true)
        await model.refresh()
        #expect(model.state == .indexing)
        for state: LocalSearchFolder.State in [.missing, .error, .unknown] {
            snapshot.value = searchStatus(folders: [searchFolder(state: state)])
            await model.refresh()
            #expect(model.status?.folders.first?.state == state)
            #expect(model.state == .ready)
        }
        model.disappear()
        #expect(!model.visible)
        #expect(!searchStatus(folders: [searchFolder(state: .waitingForModel)]).needsPolling)
    }
    @Test func unsupportedAndErrorRetry() async {
        let failure = LocalSearchTestBox<Error?>(LocalSearchClient.Failure.unsupported)
        let model = LocalSearchViewModel(fetchStatus: { _ in
            if let failure = failure.value { throw failure }; return searchStatus()
        })
        await model.appear(endpoint: endpoint)
        #expect(model.state == .unsupported)
        failure.value = LocalSearchClient.Failure.http(500, "Try again")
        await model.refresh()
        #expect(model.state == .error("Try again"))
        failure.value = nil
        await model.retry()
        #expect(model.state == .noFolders)
        model.disappear()
    }
    @Test func debounceAndReturn() async throws {
        var queries: [String] = []
        let model = LocalSearchViewModel(debounce: .milliseconds(30), fetchStatus: { _ in searchStatus(folders: [searchFolder()]) }, fetchSearch: { query, _ in
            queries.append(query)
            return .init(results: [], tookMs: 1, mode: "keyword")
        })
        await model.appear(endpoint: endpoint)
        model.query = "old"; model.queryChanged()
        model.query = "new"; model.queryChanged()
        #expect(model.state == .searching)
        try await Task.sleep(for: .milliseconds(100))
        #expect(queries == ["new"])
        #expect(model.state == .noResults)
        #expect(model.keywordMode)
        model.query = "return"; model.searchNow()
        for _ in 0..<10 { await Task.yield() }
        #expect(queries == ["new", "return"])
        model.query = ""; model.queryChanged()
        #expect(model.state == .ready)
        model.query = "hidden"; model.queryChanged(); model.disappear()
        try await Task.sleep(for: .milliseconds(60))
        #expect(queries == ["new", "return"])
    }
    @Test func cancelsInflightAndRejectsLateResponse() async throws {
        var continuations: [String: CheckedContinuation<LocalSearchResponse, Never>] = [:]
        var cancelled: [String] = []
        let model = LocalSearchViewModel(fetchStatus: { _ in searchStatus(folders: [searchFolder()]) }, fetchSearch: { query, _ in
            let response: LocalSearchResponse = await withCheckedContinuation { continuations[query] = $0 }
            if Task.isCancelled { cancelled.append(query) }
            return response
        })
        await model.appear(endpoint: endpoint)
        model.query = "old"; model.searchNow()
        for _ in 0..<20 { await Task.yield() }
        let old = try #require(continuations["old"])
        model.query = "new"; model.searchNow()
        for _ in 0..<20 { await Task.yield() }
        let new = try #require(continuations["new"])
        new.resume(returning: .init(results: [searchResult("new")], tookMs: 1, mode: "hybrid"))
        for _ in 0..<20 { await Task.yield() }
        old.resume(returning: .init(results: [searchResult("old")], tookMs: 1, mode: "keyword"))
        for _ in 0..<20 { await Task.yield() }
        #expect(model.results.map(\.title) == ["new"])
        #expect(cancelled == ["old"])
        #expect(model.state == .results)
        #expect(!model.keywordMode)
        #expect(model.selectedResult?.title == "new")
        model.disappear()
    }
    @Test func reindexCompletionRefreshesUnchangedTotals() async {
        let snapshot = LocalSearchTestBox(searchStatus(folders: [searchFolder()], active: true))
        let matches = LocalSearchTestBox([searchResult("first"), searchResult("selected")])
        let model = LocalSearchViewModel(fetchStatus: { _ in snapshot.value }, fetchSearch: { _, _ in
            .init(results: matches.value, tookMs: 1, mode: "hybrid")
        })
        await model.appear(endpoint: endpoint)
        model.query = "file"; model.searchNow()
        for _ in 0..<20 { await Task.yield() }
        model.moveSelection(1)
        #expect(model.selectedResult?.title == "selected")
        snapshot.value = searchStatus(folders: [searchFolder()], active: false)
        matches.value = [searchResult("new"), searchResult("selected")]
        await model.refresh()
        #expect(model.results.map(\.title) == ["first", "selected"], "Keep visible results while refreshing")
        for _ in 0..<20 { await Task.yield() }
        #expect(model.results.map(\.title) == ["new", "selected"])
        #expect(model.selectedResult?.title == "selected")
        model.disappear()
    }
    @Test func searchFailureAndRetry() async throws {
        let fail = LocalSearchTestBox(true)
        let model = LocalSearchViewModel(fetchStatus: { _ in searchStatus(folders: [searchFolder()]) }, fetchSearch: { _, _ in
            if fail.value { throw LocalSearchClient.Failure.http(500, "Search failed") }
            return .init(results: [searchResult("ok")], tookMs: 1, mode: "keyword")
        })
        await model.appear(endpoint: endpoint)
        model.query = "file"; model.searchNow()
        for _ in 0..<20 { await Task.yield() }
        #expect(model.state == .error("Search failed"))
        fail.value = false
        await model.retry()
        for _ in 0..<20 { await Task.yield() }
        #expect(model.state == .results)
        model.moveSelection(1)
        #expect(model.selectedResult?.title == "ok")
        model.disappear()
    }
}

private final class LocalSearchRequestRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var requests: [URLRequest] = []
    private var status = 200
    private var body = Data()
    func reply(status: Int = 200, body: Data = Data()) {
        lock.lock(); defer { lock.unlock() }
        self.status = status; self.body = body
    }
    func record(_ request: URLRequest) -> (Int, Data) {
        lock.lock(); defer { lock.unlock() }
        requests.append(request)
        return (status, body)
    }
    var captured: [URLRequest] {
        lock.lock(); defer { lock.unlock() }
        return requests
    }
}
private final class LocalSearchURLProtocol: URLProtocol {
    static let recorder = LocalSearchRequestRecorder()
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let (status, body) = Self.recorder.record(request)
        let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

@Suite(.serialized) struct LocalSearchClientTests {
    private func identityResponse(paths: [String]) throws -> Data {
        let results = paths.map { path -> [String: Any] in
            ["path": path, "folder_id": "f", "title": "file", "snippet": "match",
             "highlights": [], "score": 0.5, "matched": ["keyword"], "modified_at": 0, "size": 12]
        }
        return try JSONSerialization.data(withJSONObject: [
            "results": results, "took_ms": 1, "mode": "keyword",
            "groups": [["kind": "text", "results": results, "total": results.count]]
        ])
    }

    @Test func identicalResultsDecodeWithDistinctIDs() throws {
        let data = try identityResponse(paths: ["/file.txt", "/file.txt", "/file.txt"])
        let response = try LocalSearchClient.decode(LocalSearchResponse.self, from: data)
        #expect(response.results.count == 3)
        #expect(Set(response.results.map(\.id)).count == 3)
        let grouped = try #require(response.groups?.first?.results)
        #expect(Set(grouped.map(\.id)).count == 3)
        #expect(grouped.map(\.id) == response.results.map(\.id))
        #expect(try LocalSearchClient.decode(LocalSearchResponse.self, from: data).results.map(\.id)
            == response.results.map(\.id))
    }

    @Test func uniqueResultsKeepExistingIDsAcrossRefreshes() throws {
        let paths = ["/first.txt", "/second.txt"]
        let response = try LocalSearchClient.decode(LocalSearchResponse.self, from: identityResponse(paths: paths))
        let expected = paths.map { "f:\($0):-1:-1:match" }
        #expect(response.results.map(\.id) == expected)
        #expect(response.groups?.first?.results.map(\.id) == expected)
        let refreshed = try LocalSearchClient.decode(LocalSearchResponse.self,
            from: identityResponse(paths: Array(paths.reversed())))
        #expect(refreshed.results.map(\.id) == Array(expected.reversed()))
    }

    private func client() -> LocalSearchClient {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [LocalSearchURLProtocol.self]
        return LocalSearchClient(session: URLSession(configuration: configuration))
    }
    private func body(_ request: URLRequest) throws -> [String: Any] {
        let data: Data
        if let bytes = request.httpBody { data = bytes }
        else if let stream = request.httpBodyStream {
            stream.open(); defer { stream.close() }
            var bytes = Data()
            var buffer = [UInt8](repeating: 0, count: 1024)
            while stream.hasBytesAvailable {
                let count = stream.read(&buffer, maxLength: buffer.count)
                guard count > 0 else { break }
                bytes.append(contentsOf: buffer.prefix(count))
            }
            data = bytes
        } else { data = Data() }
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
    @Test func authenticatedRequestsAndFrozenBodies() async throws {
        let client = client()
        let recorder = LocalSearchURLProtocol.recorder
        let endpoint = LocalSearchClient.Endpoint(port: 9123, bearer: "test-secret")
        recorder.reply(body: Data(#"{"results":[],"took_ms":0.5,"mode":"keyword"}"#.utf8))
        _ = try await client.search("notes", endpoint: endpoint)
        var request = try #require(recorder.captured.last)
        #expect(request.url?.absoluteString == "http://127.0.0.1:9123/v1/local-search/search")
        #expect(request.httpMethod == "POST")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer test-secret")
        #expect(request.value(forHTTPHeaderField: RapidClientHeader.field) == RapidClientHeader.desktop)
        #expect(request.value(forHTTPHeaderField: "Content-Type") == "application/json")
        var json = try body(request)
        #expect(json["query"] as? String == "notes")
        #expect(json["limit"] as? Int == 50)
        #expect(json["folder_ids"] is NSNull)
        recorder.reply()
        try await client.add(path: "/tmp/My Docs", endpoint: endpoint)
        request = try #require(recorder.captured.last)
        #expect(request.url?.path == "/v1/local-search/folders")
        #expect(try body(request)["path"] as? String == "/tmp/My Docs")
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        try await client.add(path: home, endpoint: endpoint)
        #expect(try body(#require(recorder.captured.last))["path"] as? String == home)
        try await client.reindex(id: nil, endpoint: endpoint)
        request = try #require(recorder.captured.last)
        #expect(request.url?.path == "/v1/local-search/reindex")
        #expect(try body(request)["folder_id"] is NSNull)
        try await client.reindex(id: "f", endpoint: endpoint)
        json = try body(#require(recorder.captured.last))
        #expect(json["folder_id"] as? String == "f")
        recorder.reply(status: 204)
        try await client.remove(id: "folder", endpoint: endpoint)
        request = try #require(recorder.captured.last)
        #expect(request.url?.path == "/v1/local-search/folders/folder")
        #expect(request.httpMethod == "DELETE")
        #expect(request.httpBody == nil)
    }
    @Test func errorEnvelopeAndStatus404() async throws {
        let client = client()
        let endpoint = LocalSearchClient.Endpoint(port: 8000, bearer: nil)
        LocalSearchURLProtocol.recorder.reply(status: 400, body: Data(#"{"error":{"message":"Folder denied","type":"permission_error","code":"denied","param":"path"}}"#.utf8))
        do { try await client.add(path: "/tmp", endpoint: endpoint); Issue.record("Expected HTTP error") }
        catch { #expect(error.localizedDescription == "Folder denied") }
        LocalSearchURLProtocol.recorder.reply(status: 404)
        do { _ = try await client.status(endpoint); Issue.record("Expected unsupported engine") }
        catch { guard case LocalSearchClient.Failure.unsupported = error else { Issue.record("Wrong error: \(error)"); return } }
        do { try await client.remove(id: "removed", endpoint: endpoint); Issue.record("Expected HTTP error") }
        catch { guard case LocalSearchClient.Failure.http(404, _) = error else { Issue.record("Folder 404 is not an engine capability failure"); return } }
    }
    @Test @MainActor func folderMutationErrorRetries() async {
        let client = client()
        let recorder = LocalSearchURLProtocol.recorder
        recorder.reply(status: 500, body: Data(#"{"error":{"message":"Index failed"}}"#.utf8))
        let model = LocalSearchViewModel(client: client, fetchStatus: { _ in searchStatus(folders: [searchFolder()]) })
        await model.appear(endpoint: .init(port: 8000, bearer: "test"))
        await model.add(paths: ["/tmp/docs"])
        #expect(model.state == .error("Index failed"))
        recorder.reply()
        await model.retry()
        #expect(model.state == .ready)
        model.disappear()
    }
    @Test @MainActor func downloadCompletionReindexesOnce() async {
        let client = client()
        LocalSearchURLProtocol.recorder.reply()
        let initialCount = LocalSearchURLProtocol.recorder.captured.count
        let model = LocalSearchViewModel(client: client, fetchStatus: { _ in searchStatus(folders: [searchFolder()]) })
        let endpoint = LocalSearchClient.Endpoint(port: 8000, bearer: "test")
        await model.appear(endpoint: endpoint)
        let jobID = UUID()
        await model.downloadCompleted(jobID)
        await model.downloadCompleted(jobID)
        model.disappear()
        await model.appear(endpoint: endpoint)
        await model.downloadCompleted(jobID)
        #expect(LocalSearchURLProtocol.recorder.captured.count == initialCount + 1)
        #expect(LocalSearchURLProtocol.recorder.captured.last?.url?.path == "/v1/local-search/reindex")
        model.disappear()
    }
}

extension LocalSearchViewModelTests {
    @Test func pollsOnlyVisibleIndexingWork() async throws {
        var calls = 0
        let model = LocalSearchViewModel(fetchStatus: { _ in
            calls += 1
            return searchStatus(folders: [searchFolder(state: .queued)])
        })
        await model.appear(endpoint: endpoint)
        #expect(calls == 1)
        try await Task.sleep(for: .milliseconds(1150))
        #expect(calls == 2)
        model.disappear()
        try await Task.sleep(for: .milliseconds(1150))
        #expect(calls == 2)
    }
    @Test func idleStatusDoesNotPoll() async throws {
        var calls = 0
        let model = LocalSearchViewModel(fetchStatus: { _ in
            calls += 1
            return searchStatus(folders: [searchFolder()])
        })
        await model.appear(endpoint: endpoint)
        try await Task.sleep(for: .milliseconds(1150))
        #expect(calls == 1)
        model.disappear()
    }
    @Test func loadingAndRestartRejectStaleStatus() async throws {
        var pending: CheckedContinuation<LocalSearchStatus, Never>?
        let model = LocalSearchViewModel(fetchStatus: { _ in await withCheckedContinuation { pending = $0 } })
        let appear = Task { await model.appear(endpoint: endpoint) }
        for _ in 0..<20 { await Task.yield() }
        #expect(model.state == .loading)
        let continuation = try #require(pending)
        await model.connect(nil)
        continuation.resume(returning: searchStatus(folders: [searchFolder()]))
        await appear.value
        #expect(model.state == .engineStopped)
        #expect(model.status == nil)
        model.disappear()
    }
}

@Suite @MainActor struct LocalSearchAccessibilityTests {
    @Test func wholeMacAndUnknownOrLargeProgressRemainReadable() async throws {
        let folder = LocalSearchFolder(id: "root", path: "/", state: .indexing, filesIndexed: 42, filesSkipped: 5, chunks: 42, lastIndexedAt: nil, error: nil)
        let base = searchStatus(folders: [folder], active: true)
        var snapshot = LocalSearchStatus(enabled: true, model: base.model, folders: base.folders,
            indexing: .init(active: true, folderId: "root", currentPath: "/Users/test/Documents/example.txt", filesDone: 42, filesTotal: 0), totals: base.totals)
        let model = LocalSearchViewModel(fetchStatus: { _ in snapshot })
        let server = ServerManager(testingState: .ready(alias: "test"), binaryPath: URL(fileURLWithPath: "/usr/bin/true"))
        let stage = GoldenStage(LocalSearchView(server: server, model: model, onStartModel: { true }).environment(DownloadManager()))
        defer { model.disappear() }
        try await stage.wait(for: "unknown progress") { model.status != nil && stage.identifiers().contains("LocalSearch.IndexProgress") }
        #expect(stage.treeText().contains("Whole Mac"))
        #expect(stage.treeText().contains("Scanning and indexing: 42 files processed"))
        snapshot = .init(enabled: true, model: base.model, folders: base.folders,
            indexing: .init(active: true, folderId: "root", currentPath: nil, filesDone: 42, filesTotal: 10_000_000), totals: base.totals)
        await model.refresh()
        try await stage.wait(for: "large progress") { stage.treeText().contains("of") }
        #expect(stage.identifiers().contains("LocalSearch.IndexProgress"))
        let empty = searchStatus()
        snapshot = empty
        await model.refresh()
        try await stage.wait(for: "empty folder menu") { stage.identifiers().contains("LocalSearch.AddFolder.EmptyState") }
        #expect(stage.identifiers().contains("LocalSearch.AddFolder.Standard"))
        #expect(stage.identifiers().contains("LocalSearch.AddFolder.Desktop"))
        #expect(stage.identifiers().contains("LocalSearch.AddFolder.WholeMac"))
        #expect(stage.identifiers().contains("LocalSearch.AddFolder.Choose"))
    }

    @Test func axQueryValueTriggersDebouncedSearch() async throws {
        var queries: [String] = []
        let model = LocalSearchViewModel(debounce: .milliseconds(30),
            fetchStatus: { _ in searchStatus(folders: [searchFolder()], downloaded: false) },
            fetchSearch: { query, _ in
                queries.append(query)
                let result = LocalSearchResult(path: "/tmp/configuration.md", folderId: "folder", title: "configuration",
                    snippet: "A file", highlights: [], score: 0.5, matched: ["keyword"], page: nil, line: nil,
                    modifiedAt: Date.now.timeIntervalSince1970 - (4 * 3_600 + 33 * 60), size: 12)
                return .init(results: [result], tookMs: 1, mode: "keyword")
            })
        let server = ServerManager(testingState: .ready(alias: "test"), binaryPath: URL(fileURLWithPath: "/usr/bin/true"))
        let stage = GoldenStage(LocalSearchView(server: server, model: model, onStartModel: { true })
            .environment(DownloadManager()))
        defer { model.disappear() }
        try await stage.wait(for: "search ready") { model.status != nil }
        for id in ["LocalSearch.QueryField", "LocalSearch.AddFolder", "LocalSearch.Reindex",
                   "LocalSearch.FoldersToggle", "LocalSearch.DownloadModel",
                   "LocalSearch.Folder.0", "LocalSearch.Folder.0.Remove"] {
            #expect(stage.identifiers().contains(id), "Missing AX identifier: \(id)")
        }
        try stage.setValue("old AX query", for: "LocalSearch.QueryField")
        try stage.setValue("files via AX", for: "LocalSearch.QueryField")
        try await stage.wait(for: "AX debounced response") { model.results.count == 1 }
        #expect(model.query == "files via AX")
        #expect(queries == ["files via AX"])
        #expect(stage.identifiers().contains("LocalSearch.ResultsList"))
        #expect(stage.identifiers().contains("LocalSearch.Result.0"))
        #expect(stage.treeText().contains("configuration.md"))
        #expect(stage.treeText().contains("modified"))
        try await stage.wait(for: "folders automatically collapsed") {
            !stage.identifiers().contains("LocalSearch.Folder.0")
        }
        for id in ["LocalSearch.FoldersToggle", "LocalSearch.AddFolder", "LocalSearch.Reindex"] {
            #expect(stage.identifiers().contains(id))
        }
        try stage.press("LocalSearch.FoldersToggle")
        try await stage.wait(for: "explicitly expanded folders") {
            stage.identifiers().contains("LocalSearch.Folder.0")
        }
        await model.refresh()
        #expect(model.foldersDisclosure.isExpanded(query: model.query, hasResults: !model.results.isEmpty))
        model.disappear()
        await model.appear(endpoint: .init(port: 8000, bearer: nil))
        try await stage.wait(for: "results after returning to search") { model.results.count == 1 }
        #expect(model.foldersDisclosure.isExpanded(query: model.query, hasResults: true))
    }
}

extension LocalSearchViewModelTests {
    @Test func searchResponseDiscoversExternalIndexingWithoutSearchLoop() async throws {
        var statusCalls = 0
        var searches = 0
        let model = LocalSearchViewModel(fetchStatus: { _ in
            statusCalls += 1
            return searchStatus(folders: [searchFolder()], active: statusCalls > 1)
        }, fetchSearch: { _, _ in
            searches += 1
            return .init(results: [searchResult("file")], tookMs: 1, mode: "keyword")
        })
        await model.appear(endpoint: endpoint)
        #expect(model.status?.indexing.active == false)
        model.query = "file"; model.searchNow()
        for _ in 0..<20 { await Task.yield() }
        #expect(model.status?.indexing.active == true)
        #expect(statusCalls == 2)
        #expect(searches == 1)
        model.disappear()
    }
}

#if DEBUG
@Suite @MainActor struct LocalSearchHarnessTests {
    @Test func folderRequiresGoldenModeAndAbsolutePath() {
        let key = "RAPID_GUI_LOCAL_SEARCH_FOLDER"
        #expect(LocalSearchViewModel.harnessFolder(environment: [key: "/tmp/docs"]) == nil)
        #expect(LocalSearchViewModel.harnessFolder(environment: ["RAPID_GUI_GOLDEN_MODE": "0", key: "/tmp/docs"]) == nil)
        #expect(LocalSearchViewModel.harnessFolder(environment: ["RAPID_GUI_GOLDEN_MODE": "true", key: "/tmp/docs"]) == nil)
        #expect(LocalSearchViewModel.harnessFolder(environment: ["RAPID_GUI_GOLDEN_MODE": "1"]) == nil)
        #expect(LocalSearchViewModel.harnessFolder(environment: ["RAPID_GUI_GOLDEN_MODE": "1", key: "relative/docs"]) == nil)
        #expect(LocalSearchViewModel.harnessFolder(environment: ["RAPID_GUI_GOLDEN_MODE": "1", key: ""]) == nil)
        #expect(LocalSearchViewModel.harnessFolder(environment: ["RAPID_GUI_GOLDEN_MODE": "1", key: "/tmp/My Docs"]) == "/tmp/My Docs")
    }
}

extension LocalSearchClientTests {
    @Test @MainActor func harnessFolderWaitsForRunningEngineAndAddsOnce() async {
        let client = client()
        LocalSearchURLProtocol.recorder.reply()
        let initialCount = LocalSearchURLProtocol.recorder.captured.count
        let model = LocalSearchViewModel(client: client, fetchStatus: { _ in searchStatus() })
        let environment = ["RAPID_GUI_GOLDEN_MODE": "1", "RAPID_GUI_LOCAL_SEARCH_FOLDER": "/tmp/docs"]
        await model.appear(endpoint: nil)
        await model.addHarnessFolderIfNeeded(environment: environment)
        #expect(LocalSearchURLProtocol.recorder.captured.count == initialCount)
        await model.connect(.init(port: 8000, bearer: "test"))
        await model.addHarnessFolderIfNeeded(environment: ["RAPID_GUI_LOCAL_SEARCH_FOLDER": "/tmp/docs"])
        #expect(LocalSearchURLProtocol.recorder.captured.count == initialCount)
        await model.addHarnessFolderIfNeeded(environment: environment)
        await model.addHarnessFolderIfNeeded(environment: environment)
        model.disappear()
        await model.appear(endpoint: .init(port: 9000, bearer: "new"))
        await model.addHarnessFolderIfNeeded(environment: environment)
        #expect(LocalSearchURLProtocol.recorder.captured.count == initialCount + 1)
        #expect(LocalSearchURLProtocol.recorder.captured.last?.url?.path == "/v1/local-search/folders")
        model.disappear()
    }
}
#endif

@Suite struct LocalSearchMediaTests {
    @Test func resultKindsAndOptionalTimestamp() throws {
        let base: [String: Any] = ["path": "/tmp/movie.mov", "folder_id": "f", "title": "movie",
            "snippet": "", "highlights": [], "score": 0.7, "matched": [], "modified_at": 12, "size": 20]
        for (kind, expected) in [(nil, LocalSearchKind.text), ("text", .text), ("image", .image), ("video", .video), ("future", .text)] {
            var json = base
            if let kind { json["kind"] = kind }
            let result = try LocalSearchClient.decode(LocalSearchResult.self, from: JSONSerialization.data(withJSONObject: json))
            #expect(result.kind == expected)
            #expect(result.timestamp == nil)
            json["timestamp"] = 83.5
            let timed = try LocalSearchClient.decode(LocalSearchResult.self, from: JSONSerialization.data(withJSONObject: json))
            #expect(timed.timestamp == 83.5)
            json["kind"] = NSNull(); json["timestamp"] = NSNull()
            let nulls = try LocalSearchClient.decode(LocalSearchResult.self, from: JSONSerialization.data(withJSONObject: json))
            #expect(nulls.kind == .text)
            #expect(nulls.timestamp == nil)
        }
        var first = searchResult("movie")
        first.kind = .video; first.timestamp = 10
        var second = first; second.timestamp = 20
        #expect(first.id != second.id)
    }

    @Test func statusAdditionsAndMissingFields() throws {
        var json: [String: Any] = ["enabled": true,
            "model": ["id": "model", "downloaded": true, "loaded": false], "folders": [],
            "indexing": ["active": false, "files_done": 0, "files_total": 0],
            "totals": ["files": 0, "chunks": 0, "index_bytes": 0]]
        let old = try LocalSearchClient.decode(LocalSearchStatus.self, from: JSONSerialization.data(withJSONObject: json))
        #expect(old.model.vision != true)
        #expect(old.totals.filesByKind == nil)
        json["model"] = ["id": "model", "downloaded": true, "loaded": true, "vision": true]
        json["totals"] = ["files": 1489, "chunks": 0, "index_bytes": 0,
                          "files_by_kind": ["text": 212, "image": 1240, "video": 37]]
        let new = try LocalSearchClient.decode(LocalSearchStatus.self, from: JSONSerialization.data(withJSONObject: json))
        #expect(new.model.vision == true)
        #expect(new.totals.filesByKind?.text == 212)
        #expect(new.totals.filesByKind?.image == 1240)
        #expect(new.totals.filesByKind?.video == 37)
        json["model"] = ["id": "model", "downloaded": true, "loaded": true, "vision": NSNull()]
        json["totals"] = ["files": 0, "chunks": 0, "index_bytes": 0, "files_by_kind": NSNull()]
        let nulls = try LocalSearchClient.decode(LocalSearchStatus.self, from: JSONSerialization.data(withJSONObject: json))
        #expect(nulls.model.vision != true)
        #expect(nulls.totals.filesByKind == nil)
        let partial = try LocalSearchClient.decode(LocalSearchKindCounts.self, from: Data(#"{"image":2}"#.utf8))
        #expect(partial.text == nil)
        #expect(partial.image == 2)
    }

    @Test func timestampFormatting() {
        for (seconds, text) in [(0.0, "0:00"), (59.4, "0:59"), (83, "1:23"), (3725, "1:02:05")] {
            #expect(LocalSearchTimestamp.formatted(seconds) == text)
        }
        #expect(LocalSearchTimestamp.formatted(.nan) == "0:00")
        #expect(LocalSearchTimestamp.formatted(-1) == "0:00")
    }

    @Test func filterVisibilityAndKinds() {
        #expect(LocalSearchKindFilter.visible(vision: nil) == [.all, .text])
        #expect(LocalSearchKindFilter.visible(vision: false) == [.all, .text])
        #expect(LocalSearchKindFilter.visible(vision: true) == [.all, .text, .image, .video])
        #expect(LocalSearchKindFilter.all.kinds == nil)
        #expect(LocalSearchKindFilter.text.kinds == [.text])
        #expect(LocalSearchKindFilter.image.kinds == [.image])
        #expect(LocalSearchKindFilter.video.kinds == [.video])
        #expect(!LocalSearchKindFilter.all.isGrid)
        #expect(LocalSearchKindFilter.image.isGrid)
        #expect(LocalSearchKindFilter.video.isGrid)
    }
}

extension LocalSearchClientTests {
    @Test func kindsRequestEncoding() async throws {
        let client = client()
        let endpoint = LocalSearchClient.Endpoint(port: 8000, bearer: nil)
        LocalSearchURLProtocol.recorder.reply(body: Data(#"{"results":[],"took_ms":0,"mode":"hybrid"}"#.utf8))
        for kinds: [LocalSearchKind]? in [nil, [.text], [.image], [.video], [.text, .image, .video]] {
            _ = try await client.search("media", kinds: kinds, endpoint: endpoint)
            let json = try body(#require(LocalSearchURLProtocol.recorder.captured.last))
            #expect(json["kinds"] as? [String] == kinds?.map(\.rawValue))
            #expect(json["group_limit"] as? Int == (kinds == nil ? 8 : nil))
        }
    }
}

extension LocalSearchViewModelTests {
    @Test func filterRerunsSearchAndResetsWhenVisionDisappears() async {
        var snapshot = searchStatus(folders: [searchFolder()])
        snapshot = .init(enabled: snapshot.enabled,
            model: .init(id: "vision", downloaded: true, loaded: true, vision: true),
            folders: snapshot.folders, indexing: snapshot.indexing, totals: snapshot.totals)
        let status = LocalSearchTestBox(snapshot)
        var requests: [[LocalSearchKind]?] = []
        let model = LocalSearchViewModel(fetchStatus: { _ in status.value }, fetchMediaSearch: { _, kinds, _ in
            requests.append(kinds)
            return .init(results: [searchResult("file")], tookMs: 0, mode: "hybrid")
        })
        await model.appear(endpoint: endpoint)
        model.query = "file"; model.searchNow()
        for _ in 0..<20 { await Task.yield() }
        model.kindFilter = .image; model.kindFilterChanged()
        for _ in 0..<20 { await Task.yield() }
        #expect(requests.count == 2)
        #expect(requests.last! == [.image])
        status.value = searchStatus(folders: [searchFolder()])
        await model.refresh()
        for _ in 0..<20 { await Task.yield() }
        #expect(model.kindFilter == .all)
        #expect(requests.last! == nil)
        #expect(model.visibleKindFilters == [.all, .text])
        model.disappear()
    }
}

extension LocalSearchAccessibilityTests {
    @Test func mediaSegmentsAndGridRetainAccessibilityIdentifiers() async throws {
        let base = searchStatus(folders: [searchFolder()])
        let status = LocalSearchTestBox(LocalSearchStatus(enabled: true,
            model: .init(id: "vision", downloaded: true, loaded: true, vision: true),
            folders: base.folders, indexing: base.indexing, totals: base.totals))
        var requests: [[LocalSearchKind]?] = []
        let model = LocalSearchViewModel(fetchStatus: { _ in status.value }, fetchMediaSearch: { _, kinds, _ in
            requests.append(kinds)
            var result = searchResult("movie.mov")
            result.kind = .video; result.timestamp = 83
            return .init(results: [result], tookMs: 0, mode: "hybrid")
        })
        let server = ServerManager(testingState: .ready(alias: "test"), binaryPath: URL(fileURLWithPath: "/usr/bin/true"))
        let stage = GoldenStage(LocalSearchView(server: server, model: model, onStartModel: { true })
            .environment(DownloadManager()))
        defer { model.disappear() }
        try await stage.wait(for: "media filters") { stage.identifiers().contains("LocalSearch.Kind.Video") }
        for id in ["LocalSearch.KindFilter", "LocalSearch.Kind.All", "LocalSearch.Kind.Text", "LocalSearch.Kind.Image"] {
            #expect(stage.identifiers().contains(id))
        }
        try stage.setValue("movie", for: "LocalSearch.QueryField")
        try stage.press("LocalSearch.Kind.Video")
        try await stage.wait(for: "video grid") { stage.identifiers().contains("LocalSearch.Result.0") }
        #expect(model.kindFilter == .video)
        #expect(requests.last! == [.video])
        #expect(stage.identifiers().contains("LocalSearch.ResultsList"))
        // The tree helper prefers AXValue over AXLabel for combined cells.
        #expect(stage.tree().contains { $0.id == "LocalSearch.Result.0" && $0.role == "AXButton" })
        #expect(stage.value(of: "LocalSearch.Result.0") == "at 1:23")
        try stage.press("LocalSearch.Kind.All")
        try await stage.wait(for: "list restored") { model.kindFilter == .all && model.state == .results }
        #expect(stage.identifiers().contains("LocalSearch.Result.0"))
        status.value = base
        await model.refresh()
        try await stage.wait(for: "media segments hidden") { !stage.identifiers().contains("LocalSearch.Kind.Video") }
        #expect(!stage.identifiers().contains("LocalSearch.Kind.Image"))
        #expect(stage.identifiers().contains("LocalSearch.Kind.Text"))
    }
}

extension LocalSearchViewModelTests {
    @Test func engineStartsInPlaceAndFailureCanRetry() async throws {
        let model = LocalSearchViewModel(fetchStatus: { _ in searchStatus() })
        await model.appear(endpoint: nil)
        var completion: CheckedContinuation<Bool, Never>?
        let start = Task { await model.startEngine { await withCheckedContinuation { completion = $0 } } }
        for _ in 0..<20 { await Task.yield() }
        #expect(model.state == .engineStarting)
        #expect(model.isStartingEngine)
        var duplicateStarts = 0
        await model.startEngine { duplicateStarts += 1; return true }
        #expect(duplicateStarts == 0)
        model.engineStateChanged(.starting(alias: "current"))
        #expect(model.state == .engineStarting)
        try #require(completion).resume(returning: false)
        await start.value
        if case .error = model.state {} else { Issue.record("Failed startup must offer recovery") }
        await model.startEngine {
            await model.connect(endpoint)
            return true
        }
        #expect(model.state == .noFolders)
        #expect(!model.isStartingEngine)
        model.disappear()
    }

    @Test func serverStartingCrashAndDisconnectAreVisible() async {
        let model = LocalSearchViewModel(fetchStatus: { _ in searchStatus() })
        await model.appear(endpoint: nil)
        model.engineStateChanged(.starting(alias: "current"))
        #expect(model.state == .engineStarting)
        model.engineStateChanged(.crashed(alias: "current", message: "Could not load model"))
        #expect(model.state == .error("Could not load model"))
        model.engineStateChanged(.missing)
        if case .error(let message) = model.state { #expect(message.contains("Settings")) }
        else { Issue.record("Missing engine must explain setup recovery") }
        model.engineStateChanged(.stopped)
        #expect(model.state == .engineStopped)
        await model.connect(endpoint)
        #expect(model.state == .noFolders)
        await model.connect(nil)
        #expect(model.results.isEmpty)
        #expect(model.state == .engineStopped)
        model.disappear()
    }

    @Test func searchWorksDuringDiscoveryAndEmptyResultsChooseRecoveryByState() async {
        let snapshot = LocalSearchTestBox(searchStatus(folders: [searchFolder()], active: true))
        let matches = LocalSearchTestBox([LocalSearchResult]())
        let model = LocalSearchViewModel(fetchStatus: { _ in snapshot.value }, fetchSearch: { _, _ in
            .init(results: matches.value, tookMs: 0, mode: "keyword")
        })
        await model.appear(endpoint: endpoint)
        #expect(model.state == .indexing)
        #expect(model.status?.indexing.filesTotal == 0)
        model.query = "patent"; model.searchNow()
        for _ in 0..<20 { await Task.yield() }
        #expect(model.state == .noResults)
        #expect(model.emptyResultsAction == .wait)
        #expect(model.searchedScope.contains("/tmp/docs"))
        #expect(model.searchedScope.contains("All"))
        model.kindFilter = .text
        #expect(model.emptyResultsAction == .clearFilter)
        model.clearKindFilter()
        #expect(model.kindFilter == .all)
        for _ in 0..<20 { await Task.yield() }
        matches.value = [searchResult("patent.pdf")]
        model.searchNow()
        for _ in 0..<20 { await Task.yield() }
        #expect(model.state == .results)
        #expect(model.isIndexing)
        matches.value = []
        snapshot.value = searchStatus(folders: [searchFolder()])
        await model.refresh()
        for _ in 0..<20 { await Task.yield() }
        #expect(model.state == .noResults)
        #expect(model.emptyResultsAction == .addFolder)
        model.disappear()
    }

    @Test func typingPreservesSelectionAndDiscoveryDoesNotRepeatSearch() async throws {
        let initial = searchStatus(folders: [searchFolder()], active: true)
        let snapshot = LocalSearchTestBox(initial)
        var calls = 0
        let model = LocalSearchViewModel(debounce: .milliseconds(20), fetchStatus: { _ in snapshot.value }, fetchSearch: { _, _ in
            calls += 1
            return .init(results: [searchResult("one.pdf"), searchResult("two.docx")], tookMs: 0, mode: "hybrid")
        })
        await model.appear(endpoint: endpoint)
        model.query = "patent"; model.searchNow()
        for _ in 0..<20 { await Task.yield() }
        model.moveSelection(1)
        let selected = model.selection
        model.query = "patent claims"; model.queryChanged()
        #expect(model.selection == selected)
        #expect(model.results.count == 2)
        try await Task.sleep(for: .milliseconds(80))
        #expect(model.selection == selected)
        let before = calls
        snapshot.value = .init(enabled: true, model: initial.model, folders: initial.folders,
            indexing: .init(active: true, folderId: "folder", currentPath: "/tmp/docs/next.pdf", filesDone: 21, filesTotal: 0), totals: initial.totals)
        await model.refresh()
        for _ in 0..<20 { await Task.yield() }
        #expect(calls == before)
        #expect(model.selection == selected)
        model.moveSelection(-100); #expect(model.selectedResult?.title == "one.pdf")
        model.moveSelection(100); #expect(model.selectedResult?.title == "two.docx")
        model.fileOpenFailed()
        if case .error = model.state {} else { Issue.record("Missing file must explain recovery") }
        model.disappear()
    }

    @Test func removalIsOnlyPendingUntilConfirmedAndCanBeCancelled() async {
        let model = LocalSearchViewModel(fetchStatus: { _ in searchStatus(folders: [searchFolder()]) })
        await model.appear(endpoint: endpoint)
        model.requestRemoval(searchFolder())
        #expect(model.pendingRemoval?.id == "folder")
        #expect(model.status?.folders.count == 1)
        model.pendingRemoval = nil
        await model.confirmRemoval()?.value
        #expect(model.status?.folders.count == 1)
        model.disappear()
    }
}

extension LocalSearchPresentationTests {
    @Test func resultPathsStartAtIndexedRootAndStayWithinPathBoundaries() {
        let root = LocalSearchFolder(id: "folder", path: "/Users/example/Documents", state: .idle,
            filesIndexed: 1, filesSkipped: 0, chunks: 1, lastIndexedAt: nil, error: nil)
        func result(_ path: String) -> LocalSearchResult {
            .init(path: path, folderId: "folder", title: "claim", snippet: "claims", highlights: [], score: 1,
                matched: [], page: 4, line: nil, modifiedAt: 0, size: 10)
        }
        #expect(LocalSearchPath.resultFolder(result("/Users/example/Documents/Case/claim.pdf"), folders: [root]) == "Documents/Case")
        #expect(LocalSearchPath.resultFolder(result("/Users/example/Documents/claim.pdf"), folders: [root]) == "Documents")
        #expect(LocalSearchPath.resultFolder(result("/Users/example/Documents-other/claim.pdf"), folders: [root]) == "/Users/example/Documents-other")
    }

    @Test func folderRecoveryAndCountsHaveNoTechnicalIndexDetails() throws {
        for state: LocalSearchFolder.State in [.missing, .error, .waitingForModel] {
            #expect(searchFolder(state: state).recoveryTitle != nil)
            #expect(!state.title.isEmpty)
        }
        #expect(searchFolder().recoveryTitle == nil)
        #expect(searchFolder().countLabel == "0 files indexed · 0 not indexed")
        #expect(searchFolder().skippedExplanation == nil)
        let base = #"{"id":"f","path":"/tmp","state":"idle","files_indexed":3,"files_skipped":2,"chunks":3}"#
        let absent = try LocalSearchClient.decode(LocalSearchFolder.self, from: Data(base.utf8))
        #expect(absent.skipReasons == nil)
        let reasons = base.dropLast() + #", "skip_reasons":{"Permission denied":2}}"#
        let explained = try LocalSearchClient.decode(LocalSearchFolder.self, from: Data(reasons.utf8))
        #expect(explained.skippedExplanation == "\(LocalSearchSkipReason.localized("Permission denied")): 2")
        let null = base.dropLast() + #", "skip_reasons":null}"#
        #expect(try LocalSearchClient.decode(LocalSearchFolder.self, from: Data(null.utf8)).skipReasons == nil)
    }
}

extension LocalSearchFeatureGateTests {
    @Test func onlyEnablingWithALiveChildNeedsOneRestart() {
        for state: ServerState in [.idle, .missing, .stopped, .crashed(alias: "model", message: "failed")] {
            #expect(LocalSearchFeatureConfig.restartAlias(enabling: true, state: state) == nil)
        }
        for state: ServerState in [.starting(alias: "model"), .ready(alias: "model")] {
            #expect(LocalSearchFeatureConfig.restartAlias(enabling: true, state: state) == "model")
            #expect(LocalSearchFeatureConfig.restartAlias(enabling: false, state: state) == nil)
        }
    }
}

extension LocalSearchAccessibilityTests {
    @Test func firstRunExposesEveryChoiceAndOptionalModelCardAtLargeTextSize() async throws {
        let model = LocalSearchViewModel(fetchStatus: { _ in searchStatus(downloaded: false) })
        let server = ServerManager(testingState: .ready(alias: "test"), binaryPath: URL(fileURLWithPath: "/usr/bin/true"))
        let stage = GoldenStage(LocalSearchView(server: server, model: model, onStartModel: { true })
            .environment(DownloadManager())
            .dynamicTypeSize(.xxxLarge))
        defer { model.disappear() }
        try await stage.waitForIdentifier("LocalSearch.DownloadModel")
        for identifier in ["LocalSearch.AddFolder.Desktop", "LocalSearch.AddFolder.Documents",
                           "LocalSearch.AddFolder.Downloads", "LocalSearch.AddFolder.Standard",
                           "LocalSearch.AddFolder.WholeMac", "LocalSearch.AddFolder.Choose"] {
            #expect(stage.identifiers().contains(identifier))
        }
        #expect(stage.treeText().contains("Files never leave your Mac"))
        #expect(stage.treeText().contains("first index can take a long time"))
        #expect(stage.treeText().contains("1.1 GB"))
        #expect(stage.identifiers().contains("LocalSearch.DocumentFormats"))
        #expect(stage.treeText().contains("older .doc files need conversion"))
        #expect(!stage.identifiers().contains("LocalSearch.KindFilter"))
    }

    @Test func collapsedFoldersStillShowDiscoveryProgressAndIncompleteResults() async throws {
        let model = LocalSearchViewModel(fetchStatus: { _ in searchStatus(folders: [searchFolder()], active: true) },
            fetchSearch: { _, _ in .init(results: [searchResult("claim.pdf")], tookMs: 0, mode: "keyword") })
        let server = ServerManager(testingState: .ready(alias: "test"), binaryPath: URL(fileURLWithPath: "/usr/bin/true"))
        let stage = GoldenStage(LocalSearchView(server: server, model: model, onStartModel: { true }).environment(DownloadManager()))
        defer { model.disappear() }
        try await stage.waitForIdentifier("LocalSearch.QueryField")
        try stage.setValue("claims", for: "LocalSearch.QueryField")
        try await stage.waitForIdentifier("LocalSearch.Result.0")
        #expect(!stage.identifiers().contains("LocalSearch.Folder.0"))
        #expect(stage.identifiers().contains("LocalSearch.IndexProgress"))
        #expect(stage.identifiers().contains("LocalSearch.IncompleteResults"))
        #expect(stage.treeText().contains("files processed"))
        #expect(stage.treeText().contains("modified"))
    }
}

extension LocalSearchClientTests {
    @Test @MainActor func confirmedRemovalSendsDeleteOnceAndLeavesSourceFilesAlone() async throws {
        let client = client()
        LocalSearchURLProtocol.recorder.reply()
        let before = LocalSearchURLProtocol.recorder.captured.count
        let model = LocalSearchViewModel(client: client, fetchStatus: { _ in searchStatus(folders: [searchFolder()]) })
        await model.appear(endpoint: .init(port: 8000, bearer: nil))
        model.requestRemoval(searchFolder())
        #expect(LocalSearchURLProtocol.recorder.captured.count == before)
        let removal = model.confirmRemoval()
        model.pendingRemoval = nil // Same binding reset as native alert dismissal.
        await removal?.value
        await model.confirmRemoval()?.value
        #expect(model.pendingRemoval == nil)
        #expect(LocalSearchURLProtocol.recorder.captured.count == before + 1)
        let request = try #require(LocalSearchURLProtocol.recorder.captured.last)
        #expect(request.httpMethod == "DELETE")
        #expect(request.url?.path == "/v1/local-search/folders/folder")
        model.disappear()
    }
}


extension LocalSearchDecodingTests {
    @Test func groupedResponseDecodesOptionalPreviewAndCounts() throws {
        let json = Data(#"{"results":[],"took_ms":2,"mode":"hybrid","groups":[{"kind":"video","total":12,"results":[{"path":"/clip.mov","folder_id":"f","title":"clip","snippet":"","highlights":[],"score":0.7,"matched":["semantic"],"modified_at":0,"size":20,"kind":"video","timestamp":83}]}]}"#.utf8)
        let response = try LocalSearchClient.decode(LocalSearchResponse.self, from: json)
        #expect(response.groups?.first?.kind == .video)
        #expect(response.groups?.first?.total == 12)
        #expect(response.groups?.first?.results.first?.timestamp == 83)
        let legacy = Data(#"{"results":[],"took_ms":2,"mode":"keyword","groups":null}"#.utf8)
        #expect(try LocalSearchClient.decode(LocalSearchResponse.self, from: legacy).groups == nil)
    }
}

extension LocalSearchPresentationTests {
    @Test func sectionPreviewsKeepServerOrderCountsAndKindLimits() {
        let documents = (0..<10).map { searchResult("doc\($0)") }
        let media = (0..<10).map { index in
            var result = searchResult("image\(index)"); result.kind = .image; return result
        }
        let sections = LocalSearchSection.previews([
            .init(kind: .image, results: media, total: 40),
            .init(kind: .video, results: [], total: 0),
            .init(kind: .text, results: documents, total: 20)])
        #expect(sections.map(\.kind) == [.image, .text])
        #expect(sections.map(\.total) == [40, 20])
        #expect(sections.map { $0.results.count } == [8, 5])
        #expect(sections.first?.results == Array(media.prefix(8)))
        #expect(sections.last?.filter == .text)
    }
}

extension LocalSearchViewModelTests {
    @Test func groupedSelectionCrossesSectionsAndFallsBackForOlderEngine() async {
        let docs = [searchResult("doc0"), searchResult("doc1")]
        let images = (0..<3).map { index in
            var result = searchResult("image\(index)"); result.kind = .image; return result
        }
        var video = searchResult("video"); video.kind = .video
        let response = LocalSearchTestBox(LocalSearchResponse(results: [video], tookMs: 0, mode: "hybrid", groups: [
            .init(kind: .text, results: docs, total: 10),
            .init(kind: .image, results: images, total: 30),
            .init(kind: .video, results: [video], total: 1)]))
        let model = LocalSearchViewModel(fetchStatus: { _ in searchStatus(folders: [searchFolder()]) },
            fetchSearch: { _, _ in response.value })
        await model.appear(endpoint: endpoint)
        defer { model.disappear() }
        model.query = "files"; model.searchNow()
        for _ in 0..<30 { await Task.yield() }
        #expect(model.selectedResult == docs[0])
        #expect(!model.moveSelection(.right))
        model.moveSelection(.down); #expect(model.selectedResult == docs[1])
        model.moveSelection(.down); #expect(model.selectedResult == images[0])
        model.moveSelection(.right); #expect(model.selectedResult == images[1])
        model.moveSelection(.right); #expect(model.selectedResult == images[2])
        model.moveSelection(.right); #expect(model.selectedResult == images[2])
        model.moveSelection(.down); #expect(model.selectedResult == video)
        model.moveSelection(.down); #expect(model.selectedResult == video)
        model.moveSelection(.up); #expect(model.selectedResult == images[0])
        model.moveSelection(.up); #expect(model.selectedResult == docs[1])
        model.moveSelection(.up); #expect(model.selectedResult == docs[0])
        model.moveSelection(.up); #expect(model.selectedResult == docs[0])
        model.kindFilter = .text
        #expect(model.sections == nil)
        #expect(model.displayedResults == [video])
        model.kindFilter = .all
        response.value = .init(results: docs, tookMs: 0, mode: "keyword")
        model.searchNow()
        for _ in 0..<30 { await Task.yield() }
        #expect(model.sections == nil)
        #expect(model.displayedResults == docs)
        model.moveSelection(.down); #expect(model.selectedResult == docs[1])
    }
}


extension LocalSearchViewModelTests {
    @Test func mediaFirstAndSingleSectionSelectionBoundaries() async {
        let docs = [searchResult("doc0"), searchResult("doc1")]
        let images = (0..<3).map { index in
            var result = searchResult("image\(index)"); result.kind = .image; return result
        }
        var video = searchResult("video"); video.kind = .video
        let response = LocalSearchTestBox(LocalSearchResponse(results: [], tookMs: 0, mode: "hybrid", groups: [
            .init(kind: .image, results: images, total: 3),
            .init(kind: .video, results: [video], total: 1),
            .init(kind: .text, results: docs, total: 2)]))
        let model = LocalSearchViewModel(fetchStatus: { _ in searchStatus(folders: [searchFolder()]) },
            fetchSearch: { _, _ in response.value })
        await model.appear(endpoint: endpoint)
        defer { model.disappear() }
        model.query = "files"; model.searchNow()
        for _ in 0..<30 { await Task.yield() }
        #expect(model.state == .results)
        #expect(model.selectedResult == images[0])
        model.moveSelection(.up); #expect(model.selectedResult == images[0])
        model.moveSelection(.right); model.moveSelection(.right)
        model.moveSelection(.down); #expect(model.selectedResult == video)
        model.moveSelection(.down); #expect(model.selectedResult == docs[0])
        model.moveSelection(.down); #expect(model.selectedResult == docs[1])
        model.moveSelection(.up); model.moveSelection(.up)
        #expect(model.selectedResult == video)
        model.selection = nil
        model.moveSelection(.up); #expect(model.selectedResult == docs[1])
        response.value = .init(results: [], tookMs: 0, mode: "hybrid", groups: [
            .init(kind: .image, results: images, total: 3)])
        model.searchNow()
        for _ in 0..<30 { await Task.yield() }
        #expect(model.sections?.count == 1)
        model.moveSelection(.down); #expect(model.selectedResult == images[0])
        model.moveSelection(.down); #expect(model.selectedResult == images[0])
        model.moveSelection(.right); #expect(model.selectedResult == images[1])
        model.query = ""; model.queryChanged()
        #expect(model.sections == nil)
        #expect(model.selectedResult == nil)
    }

    @Test func groupedNativeHeadersCountsAndShowAll() async throws {
        let base = searchStatus(folders: [searchFolder()])
        let status = LocalSearchStatus(enabled: true,
            model: .init(id: "vision", downloaded: true, loaded: true, vision: true),
            folders: base.folders, indexing: base.indexing, totals: base.totals)
        let document = searchResult("notes.md")
        var video = searchResult("clip.mov"); video.kind = .video; video.timestamp = 83
        let model = LocalSearchViewModel(fetchStatus: { _ in status }, fetchMediaSearch: { _, kinds, _ in
            if kinds == [.video] { return .init(results: [video], tookMs: 0, mode: "hybrid") }
            return .init(results: [document], tookMs: 0, mode: "hybrid", groups: [
                .init(kind: .video, results: [video], total: 12),
                .init(kind: .text, results: [document], total: 20)])
        })
        let server = ServerManager(testingState: .ready(alias: "test"), binaryPath: URL(fileURLWithPath: "/usr/bin/true"))
        let stage = GoldenStage(LocalSearchView(server: server, model: model, onStartModel: { true })
            .environment(DownloadManager()))
        defer { model.disappear() }
        try await stage.waitForIdentifier("LocalSearch.QueryField")
        try stage.setValue("clip", for: "LocalSearch.QueryField")
        try await stage.waitForIdentifier("LocalSearch.Section.video.Header")
        #expect(stage.tree().first { $0.id == "LocalSearch.Section.video.Header" }?.text == "Videos (12)")
        #expect(stage.tree().first { $0.id == "LocalSearch.Section.text.Header" }?.text == "Documents (20)")
        #expect(stage.value(of: "LocalSearch.Result.0") == "at 1:23")
        #expect(model.selectedResult == video)
        try stage.press("LocalSearch.Section.video.ShowAll")
        try await stage.wait(for: "full video tab") { model.kindFilter == .video && model.state == .results }
        #expect(!stage.identifiers().contains("LocalSearch.Section.video.Header"))
        #expect(stage.identifiers().contains("LocalSearch.Result.0"))
    }
}

extension LocalSearchAccessibilityTests {
    @Test func pausedIndexingKeepsSearchUsableAndNameOnlyResultsAccessible() async throws {
        let base = searchStatus(folders: [searchFolder()], active: true)
        let status = LocalSearchStatus(enabled: true, model: base.model, folders: base.folders,
            indexing: .init(active: true, folderId: "folder", currentPath: nil, filesDone: 2, filesTotal: 10,
                pausedReason: "chat"), totals: base.totals)
        var result = searchResult("scan.pdf")
        result.contentIndexed = false
        result.skipReason = "Scanned PDF without text"
        let model = LocalSearchViewModel(fetchStatus: { _ in status }, fetchSearch: { _, _ in
            .init(results: [result], tookMs: 0, mode: "keyword")
        })
        let server = ServerManager(testingState: .ready(alias: "test"), binaryPath: URL(fileURLWithPath: "/usr/bin/true"))
        let stage = GoldenStage(LocalSearchView(server: server, model: model, onStartModel: { true })
            .environment(DownloadManager()))
        defer { model.disappear() }
        try await stage.wait(for: "paused indexing message") { stage.treeText().contains(status.indexing.progressLabel) }
        #expect(model.status?.needsPolling == true)
        #expect(!stage.identifiers().contains("LocalSearch.IndexProgress"))
        try stage.setValue("scan", for: "LocalSearch.QueryField")
        try await stage.waitForIdentifier("LocalSearch.Result.0")
        let row = try #require(stage.tree().first { $0.id == "LocalSearch.Result.0" })
        #expect(row.text.contains(result.snippetLabel))
        #expect(stage.identifiers().contains("LocalSearch.ResultsList"))
    }
}
