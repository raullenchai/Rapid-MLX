import AppKit
import Foundation
import PDFKit
import Testing
@testable import Rapid

/// Contracts for ``read_document`` — the tool that makes a large attachment
/// analyzable without forcing its whole text into the prompt.
///
/// The security-relevant assertion is ``unknownDocumentIsRefused``: the tool
/// takes no path, so the ONLY documents it can reach are those the user
/// attached and Rapid cached. Everything else is a miss.
@MainActor
@Suite("read_document")
struct ReadDocumentToolTests {
    /// Memory-only cache so tests never touch the real Application Support tree.
    private func freshCache() -> DocumentContentCache {
        DocumentContentCache(diskDirectory: nil)
    }

    private func store(
        _ text: String,
        filename: String = "report.pdf",
        pageCount: Int? = nil,
        outline: [DocumentContentCache.OutlineNode] = [],
        in cache: DocumentContentCache
    ) -> UUID {
        let id = UUID()
        cache.put(id, entry: DocumentContentCache.Entry(
            filename: filename,
            text: text,
            pageCount: pageCount,
            outline: outline
        ))
        return id
    }

    private func run(
        _ arguments: [String: Any],
        cache: DocumentContentCache,
        stallTimeout: TimeInterval = 30
    ) async -> ToolCallResult {
        let data = try! JSONSerialization.data(withJSONObject: arguments)
        return await ReadDocumentTool.run(
            arguments: String(data: data, encoding: .utf8)!,
            cache: cache,
            stallTimeout: stallTimeout
        )
    }

    private func payload(_ result: ToolCallResult) throws -> [String: Any] {
        try #require(
            JSONSerialization.jsonObject(with: Data(result.content.utf8)) as? [String: Any]
        )
    }

    // MARK: - Reach

    @Test("A document id that was never attached is refused, not looked up")
    func unknownDocumentIsRefused() async throws {
        let cache = freshCache()
        _ = store("attached content", in: cache)

        // A well-formed id nobody registered: the cache is the whole namespace,
        // so this can only miss.
        let result = await run(["document_id": UUID().uuidString], cache: cache)
        #expect(result.isError)
        #expect(result.content.contains("no attached document"))
        #expect(!result.content.contains("attached content"))
    }

    @Test("A path-shaped argument is rejected — this tool addresses no filesystem")
    func pathArgumentIsRejected() async throws {
        let cache = freshCache()
        let result = await run(["document_id": "/etc/passwd"], cache: cache)
        #expect(result.isError)
        #expect(result.content.contains("not a valid document id"))
    }

    // MARK: - Sequential paging

    @Test("A short document is returned whole with no continuation cursor")
    func shortDocumentHasNoNextOffset() async throws {
        let cache = freshCache()
        let id = store("all of it", filename: "notes.txt", in: cache)

        let json = try payload(await run(["document_id": id.uuidString], cache: cache))
        #expect(json["content"] as? String == "all of it")
        #expect(json["has_more"] as? Bool == false)
        #expect(json["next_offset"] == nil)
        #expect(json["filename"] as? String == "notes.txt")
    }

    @Test("A long document pages through its whole text via next_offset")
    func paginationCoversEveryCharacter() async throws {
        let cache = freshCache()
        // Line-broken so the boundary snap has somewhere to land.
        let text = (0..<4000).map { "line \($0) of the document" }.joined(separator: "\n")
        let id = store(text, in: cache)

        var assembled = ""
        var offset = 0
        var calls = 0
        while true {
            calls += 1
            #expect(calls < 100)   // guards an infinite loop if the cursor stalls
            let json = try payload(
                await run(["document_id": id.uuidString, "offset": offset], cache: cache)
            )
            assembled += try #require(json["content"] as? String)
            #expect(json["total_chars"] as? Int == text.count)
            guard json["has_more"] as? Bool == true else { break }
            let next = try #require(json["next_offset"] as? Int)
            #expect(next > offset)   // a cursor that doesn't advance would hang the model
            offset = next
        }
        // The point of the whole feature: nothing is lost past the preview.
        #expect(assembled == text)
        #expect(calls > 1)
    }

    @Test("An offset past the end reports the end instead of erroring")
    func offsetPastEndIsNotAnError() async throws {
        let cache = freshCache()
        let id = store("short", in: cache)

        let json = try payload(
            await run(["document_id": id.uuidString, "offset": 9_000], cache: cache)
        )
        #expect(!(json["has_more"] as? Bool ?? true))
        #expect((json["note"] as? String)?.contains("past the end") == true)
    }

    // MARK: - Grep

    @Test("grep returns matching passages with reusable offsets")
    func grepFindsPassagesAndOffsets() async throws {
        let cache = freshCache()
        let filler = String(repeating: "padding text\n", count: 3_000)
        let text = filler + "The NET REVENUE for Q4 was 12.5M.\n" + filler
        let id = store(text, in: cache)

        let json = try payload(
            await run(["document_id": id.uuidString, "grep": "net revenue"], cache: cache)
        )
        #expect(json["match_count"] as? Int == 1)
        let passages = try #require(json["passages"] as? [[String: Any]])
        #expect(passages.count == 1)
        // Case-insensitive by default, and the surrounding context comes along.
        #expect((passages[0]["text"] as? String)?.contains("NET REVENUE for Q4") == true)

        // The reported offset must be reusable as a sequential cursor.
        let offset = try #require(passages[0]["offset"] as? Int)
        let follow = try payload(
            await run(["document_id": id.uuidString, "offset": offset], cache: cache)
        )
        #expect((follow["content"] as? String)?.contains("NET REVENUE") == true)
    }

    @Test("grep offsets stay correct for non-ASCII text")
    func grepOffsetsSurviveMultibyteText() async throws {
        let cache = freshCache()
        // NSRegularExpression reports UTF-16 ranges while pagination counts
        // Characters. An emoji (surrogate pair) plus CJK text would desync the
        // two if the conversion were skipped, handing back an unusable cursor.
        let text = String(repeating: "文档内容 🎉\n", count: 500)
            + "TARGET LINE\n"
            + String(repeating: "更多内容 🎉\n", count: 500)
        let id = store(text, in: cache)

        let json = try payload(
            await run(["document_id": id.uuidString, "grep": "TARGET LINE"], cache: cache)
        )
        let passages = try #require(json["passages"] as? [[String: Any]])
        let offset = try #require(passages[0]["offset"] as? Int)

        let follow = try payload(
            await run(["document_id": id.uuidString, "offset": offset], cache: cache)
        )
        #expect((follow["content"] as? String)?.contains("TARGET LINE") == true)
    }

    @Test("A pattern with no match says so instead of returning the first page")
    func grepMissIsExplicit() async throws {
        let cache = freshCache()
        let id = store("alpha beta gamma", in: cache)

        let json = try payload(
            await run(["document_id": id.uuidString, "grep": "omega"], cache: cache)
        )
        #expect(json["match_count"] as? Int == 0)
        #expect(json["passages"] == nil)
        #expect((json["note"] as? String)?.contains("No match") == true)
    }

    @Test("An invalid regular expression is a recoverable error, not a crash")
    func invalidRegexIsRecoverable() async throws {
        let cache = freshCache()
        let id = store("content", in: cache)

        let result = await run(["document_id": id.uuidString, "grep": "([unclosed"], cache: cache)
        #expect(result.isError)
        #expect(result.content.contains("invalid regular expression"))
    }

    @Test("A pattern matching everywhere stays within the character budget")
    func grepIsBudgeted() async throws {
        let cache = freshCache()
        let id = store(String(repeating: "a\n", count: 100_000), in: cache)

        let json = try payload(
            await run(["document_id": id.uuidString, "grep": "a"], cache: cache)
        )
        let passages = try #require(json["passages"] as? [[String: Any]])
        #expect(passages.count <= ReadDocumentTool.maxGrepMatches)
        let emitted = passages.reduce(0) { $0 + (($1["text"] as? String)?.count ?? 0) }
        #expect(emitted <= ReadDocumentTool.charBudget)
        // The scan stopped at the cap, so the count is a floor, not a total.
        #expect(json["search_complete"] as? Bool == false)
        #expect((json["note"] as? String)?.contains("at least") == true)
    }

    @Test("A single whole-document grep match is independently bounded")
    func wholeDocumentMatchIsBounded() async throws {
        let cache = freshCache()
        let id = store(String(repeating: "a", count: 100_000), in: cache)

        let result = await run(
            ["document_id": id.uuidString, "grep": "(?s).*"],
            cache: cache
        )
        let json = try payload(result)
        let passages = try #require(json["passages"] as? [[String: Any]])
        let first = try #require(passages.first)
        let match = try #require(first["match"] as? String)

        #expect(match.count == ReadDocumentTool.maxGrepMatchCharacters)
        #expect(first["match_truncated"] as? Bool == true)
        #expect((first["text"] as? String)?.count ?? 0 <= ReadDocumentTool.charBudget)
        #expect(result.content.count < ReadDocumentTool.charBudget
            + ReadDocumentTool.maxGrepMatchCharacters + 2_000)
    }

    // MARK: - Adversarial grep
    //
    // `grep` runs a MODEL-SUPPLIED regular expression over an extract that may
    // hold ``ChatFileAttachment/maxExtractedCharacters``. Everything below is
    // about that combination: the pattern is untrusted input, the corpus is at
    // the size limit, and a document can carry an instruction telling the model
    // which pattern to send. These are resource bounds, not correctness nits.

    /// The largest extract the app will ever hold, filled with distinct lines
    /// so a pattern cannot be answered from a shared prefix.
    private func maximumSizeText() -> String {
        // Built by repeating a large pre-sized block rather than appending in a
        // `while text.count < limit` loop: `String.count` walks the whole string
        // every iteration, which makes the obvious spelling quadratic and hangs
        // the test long before it can assert anything about grep.
        var block = ""
        for line in 0..<10_000 {
            block += "line \(line) of the maximum size document\n"
        }
        let blockLength = block.count
        let repeats = ChatFileAttachment.maxExtractedCharacters / blockLength + 1
        return String(
            String(repeating: block, count: repeats)
                .prefix(ChatFileAttachment.maxExtractedCharacters)
        )
    }

    @Test("A match-everything pattern over a maximum-size document stays bounded", .timeLimit(.minutes(1)))
    func grepDoesNotMaterializeEveryMatch() async throws {
        // The regression: `matches(in:)` allocated an NSTextCheckingResult for
        // EVERY hit before ten were kept, so '.' over a 20,000,000-character
        // extract meant twenty million objects to return ten passages — a
        // locally triggerable memory/CPU exhaustion. Enumeration with an early
        // stop must finish this in the time it takes to find ten matches, not
        // the time it takes to scan the document.
        let cache = freshCache()
        let id = store(maximumSizeText(), in: cache)

        let started = Date()
        let json = try payload(
            await run(["document_id": id.uuidString, "grep": "."], cache: cache)
        )
        let elapsed = Date().timeIntervalSince(started)

        let passages = try #require(json["passages"] as? [[String: Any]])
        #expect(passages.count <= ReadDocumentTool.maxGrepMatches)
        // Deliberately an order of magnitude above what the early stop costs,
        // rather than a tight bound on it. The whole suite runs in parallel and
        // this assertion is wall-clock, so a tight bound measures machine load
        // more than it measures the code. Twenty million allocations is tens of
        // seconds and gigabytes even unloaded, so the regression is still
        // caught — and `.timeLimit` is the backstop if it hangs outright.
        #expect(elapsed < 30)
    }

    @Test("A backtracking pattern is abandoned at the time budget, not run to completion", .timeLimit(.minutes(1)))
    func grepStopsCatastrophicBacktracking() async throws {
        // The match cap alone does not bound WORK: this pattern produces no
        // match at all, so nothing is ever counted, and a naive engine spends
        // exponential time inside one attempt. `.reportProgress` is the only
        // point at which this code regains control mid-attempt.
        let cache = freshCache()
        let id = store(String(repeating: "a", count: 40_000) + "\n", in: cache)

        let started = Date()
        let json = try payload(
            await run(["document_id": id.uuidString, "grep": "(a+)+b"], cache: cache)
        )
        let elapsed = Date().timeIntervalSince(started)

        // Generous for the same reason as above: how often `.reportProgress`
        // fires depends on how much CPU this process is getting. What must hold
        // is that the scan TERMINATES — an unbounded one on 40,000 'a's does
        // not finish in any amount of time worth waiting for.
        #expect(elapsed < 30)
        // Whatever it found, it must not claim the document was fully searched.
        if json["search_complete"] as? Bool == false {
            #expect((json["note"] as? String)?.contains("stopped") == true)
        }
    }

    @Test("An over-long grep pattern is refused before it is compiled")
    func grepPatternLengthIsCapped() async throws {
        // Pattern length is the cheap proxy for how much nesting and
        // alternation the model can hand this engine, so it is bounded before
        // compilation rather than after the scan has already begun.
        let cache = freshCache()
        let id = store("content", in: cache)
        let pattern = String(repeating: "(a|b)", count: 200)

        let result = await run(["document_id": id.uuidString, "grep": pattern], cache: cache)
        #expect(result.isError)
        #expect(result.content.contains("too long"))
    }

    @Test("A completed grep says so, so a full count is distinguishable from a floor")
    func grepReportsWhetherTheScanFinished() async throws {
        let cache = freshCache()
        let id = store("alpha\nbeta\ngamma\n", in: cache)

        let json = try payload(
            await run(["document_id": id.uuidString, "grep": "beta"], cache: cache)
        )
        #expect(json["match_count"] as? Int == 1)
        #expect(json["search_complete"] as? Bool == true)
    }

    // MARK: - Cache round-trip

    @Test("An attached document is registered so read_document can reach it")
    func attachmentRegistersFullTextForTheTool() async throws {
        let cache = freshCache()
        let body = (0..<5_000).map { "row \($0)" }.joined(separator: "\n")
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("txt")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(body.utf8).write(to: url)

        let attachment = try ChatFileAttachment(contentsOf: url, cache: cache)
        // The prompt carries only a preview, but the whole document is reachable.
        #expect(attachment.hasUnshownContent)
        #expect(attachment.extractedText.count < body.count)
        #expect(attachment.totalCharacterCount == body.count)
        #expect(attachment.promptText.contains("read_document"))
        #expect(attachment.promptText.contains(attachment.id.uuidString))

        let json = try payload(
            await run(["document_id": attachment.id.uuidString, "grep": "row 4999"], cache: cache)
        )
        #expect(json["match_count"] as? Int == 1)
    }

    // MARK: - Deferred extraction

    /// Multi-page PDF whose every page carries findable text.
    private func makePDF(pages: Int) -> Data {
        let doc = PDFDocument()
        for index in 0..<pages {
            let view = NSTextView(frame: NSRect(x: 0, y: 0, width: 400, height: 200))
            view.string = "PAGEMARK\(index) content for page \(index)."
            guard let page = PDFDocument(data: view.dataWithPDF(inside: view.bounds))?.page(at: 0) else {
                continue
            }
            doc.insert(page, at: doc.pageCount)
        }
        return doc.dataRepresentation() ?? Data()
    }

    private func writePDF(pages: Int) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("pdf")
        try makePDF(pages: pages).write(to: url)
        return url
    }

    @Test("Attaching a long PDF does not extract every page up front")
    func attachExtractsOnlyThePreviewWindow() async throws {
        // The perceived-stall fix: extracting all 302 pages of a real book
        // cost ~1.9s while the send button was disabled. Only the eager
        // window is read synchronously; the rest lands in the background.
        let cache = freshCache()
        let url = try writePDF(pages: 40)
        defer { try? FileManager.default.removeItem(at: url) }

        let attachment = try ChatFileAttachment(contentsOf: url, cache: cache)
        #expect(attachment.pageCount == 40)
        // Page count comes from PDFDocument, which is cheap; the tail's TEXT
        // is not read yet, so the total length is honestly unknown.
        #expect(attachment.totalCharacterCount == nil)
        #expect(attachment.hasUnshownContent)
        // The envelope must not print "nil" at the model.
        #expect(!attachment.promptText.contains("nil"))
        #expect(attachment.promptText.contains("read_document"))
    }

    @Test("read_document waits for the background pass and sees the last page")
    func deferredExtractionCompletesForTheTool() async throws {
        let cache = freshCache()
        let url = try writePDF(pages: 40)
        defer { try? FileManager.default.removeItem(at: url) }

        let attachment = try ChatFileAttachment(contentsOf: url, cache: cache)
        // Text from beyond the eager window is reachable: the tool blocks on
        // the in-flight extraction rather than reporting a missing document.
        let json = try payload(
            await run(["document_id": attachment.id.uuidString, "grep": "PAGEMARK39"], cache: cache)
        )
        #expect(json["match_count"] as? Int == 1)
    }

    @Test("A short PDF is complete on attach with no pending work")
    func shortPDFNeedsNoBackgroundPass() async throws {
        // Below the eager window nothing is deferred, so the total is known
        // immediately and behaviour is exactly as before the split.
        let cache = freshCache()
        let url = try writePDF(pages: 3)
        defer { try? FileManager.default.removeItem(at: url) }

        let attachment = try ChatFileAttachment(contentsOf: url, cache: cache)
        #expect(attachment.pageCount == 3)
        #expect(attachment.totalCharacterCount != nil)
        #expect(!attachment.hasUnshownContent)
    }

    // MARK: - Outline

    @Test("Outline mode returns the bookmark map in a single call")
    func outlineFromBookmarks() async throws {
        // The whole point: one call yields the shape of a document that would
        // take ~33 sequential slices to read, and each row carries an offset
        // the model can read from next.
        let cache = freshCache()
        let id = store(
            "[Page 1]\nIntro body\n\n[Page 2]\nChapter body",
            pageCount: 2,
            outline: [
                .init(title: "Introduction", depth: 0, page: 1, offset: 0),
                .init(title: "Background", depth: 1, page: 1, offset: 9),
                .init(title: "Chapter 1", depth: 0, page: 2, offset: 21),
            ],
            in: cache
        )

        let json = try payload(await run(["document_id": id.uuidString, "mode": "outline"], cache: cache))
        #expect(json["outline_source"] as? String == "bookmarks")
        let rows = try #require(json["outline"] as? [[String: Any]])
        #expect(rows.count == 3)
        #expect(rows[0]["title"] as? String == "Introduction")
        #expect(rows[1]["depth"] as? Int == 1)
        #expect(rows[2]["offset"] as? Int == 21)
        #expect(json["total_pages"] as? Int == 2)
    }

    @Test("An outline offset can be read back as a sequential cursor")
    func outlineOffsetsAreReadable() async throws {
        let cache = freshCache()
        let body = "[Page 1]\n" + String(repeating: "front matter\n", count: 100) + "TARGET SECTION starts here"
        let target = body.distance(from: body.startIndex, to: body.range(of: "TARGET SECTION")!.lowerBound)
        let id = store(
            body,
            outline: [.init(title: "Target", depth: 0, page: 1, offset: target)],
            in: cache
        )

        let outline = try payload(await run(["document_id": id.uuidString, "mode": "outline"], cache: cache))
        let rows = try #require(outline["outline"] as? [[String: Any]])
        let offset = try #require(rows[0]["offset"] as? Int)

        let read = try payload(await run(["document_id": id.uuidString, "offset": offset], cache: cache))
        #expect((read["content"] as? String)?.hasPrefix("TARGET SECTION") == true)
    }

    @Test("A large outline is trimmed by depth, keeping the document's shape")
    func outlineIsBudgetedByDepth() async throws {
        // Dropping the deepest levels first keeps every chapter visible. The
        // alternative — truncating in order — would return the first 40
        // sub-sub-sections of chapter one and nothing after it.
        let cache = freshCache()
        var nodes: [DocumentContentCache.OutlineNode] = []
        for chapter in 0..<40 {
            nodes.append(.init(title: "Chapter \(chapter) of the document", depth: 0, page: chapter + 1, offset: chapter * 100))
            for section in 0..<20 {
                nodes.append(.init(title: "Section \(chapter).\(section) with a reasonably long heading", depth: 1, page: chapter + 1, offset: chapter * 100 + section))
            }
        }
        let id = store("body", outline: nodes, in: cache)

        let json = try payload(await run(["document_id": id.uuidString, "mode": "outline"], cache: cache))
        let rows = try #require(json["outline"] as? [[String: Any]])
        #expect(rows.count < nodes.count)
        #expect(json["entries_omitted"] as? Int == nodes.count - rows.count)
        // Every top-level chapter survived; only the depth-1 rows went.
        #expect(rows.allSatisfy { ($0["depth"] as? Int) == 0 })
        #expect(rows.count == 40)
    }

    @Test("A document without bookmarks gets an inferred outline")
    func outlineFallsBackToInference() async throws {
        let cache = freshCache()
        let body = """
        [Page 1]
        第 1 章 开始
        正文内容在这里，这一行不是标题因为它足够长而且没有编号前缀。
        1.1 第一节标题
        更多正文。
        第 2 章 继续
        2.1 另一节
        """
        let id = store(body, in: cache)

        let json = try payload(await run(["document_id": id.uuidString, "mode": "outline"], cache: cache))
        #expect(json["outline_source"] as? String == "inferred")
        let titles = try #require(json["outline"] as? [[String: Any]]).compactMap { $0["title"] as? String }
        #expect(titles.contains("第 1 章 开始"))
        #expect(titles.contains("1.1 第一节标题"))
        #expect(titles.contains("第 2 章 继续"))
        // Prose must not be mistaken for a heading.
        #expect(!titles.contains { $0.hasPrefix("正文内容") })
        #expect((json["note"] as? String)?.contains("inferred") == true)
    }

    @Test("A document with no structure at all says so instead of returning nothing")
    func outlineReportsAbsenceOfStructure() async throws {
        let cache = freshCache()
        let id = store("Just some prose with no headings whatsoever, running on for a while.", in: cache)

        let json = try payload(await run(["document_id": id.uuidString, "mode": "outline"], cache: cache))
        let rows = try #require(json["outline"] as? [[String: Any]])
        #expect(rows.isEmpty)
        #expect((json["note"] as? String)?.contains("no detectable section structure") == true)
        // Not an error: "this document is flat" is a real answer.
        #expect((json["total_chars"] as? Int) ?? 0 > 0)
    }

    @Test("A real PDF's bookmarks survive attach and the Codable round trip")
    func outlineSurvivesAttachAndPersistence() async throws {
        let cache = freshCache()
        let url = try writePDF(pages: 5)
        defer { try? FileManager.default.removeItem(at: url) }

        let attachment = try ChatFileAttachment(contentsOf: url, cache: cache)
        let entry = try #require(cache.get(attachment.id))
        // The generated fixture carries no bookmarks, so the cached outline is
        // empty — and must round-trip as empty rather than failing to decode.
        let encoded = try JSONEncoder().encode(entry)
        let decoded = try JSONDecoder().decode(DocumentContentCache.Entry.self, from: encoded)
        #expect(decoded.outline == entry.outline)
        #expect(decoded.text == entry.text)
    }

    @Test("An entry stored before outline support still decodes")
    func outlineIsBackwardCompatible() throws {
        // Older cache files have no "outline" key; a required field would make
        // every previously-cached document undecodable after upgrade.
        let legacy = #"{"filename":"old.pdf","text":"body","pageCount":3}"#
        let decoded = try JSONDecoder().decode(
            DocumentContentCache.Entry.self,
            from: Data(legacy.utf8)
        )
        #expect(decoded.outline.isEmpty)
        #expect(decoded.text == "body")
        #expect(decoded.pageCount == 3)
    }

    @Test("The attachment envelope steers whole-document questions to outline")
    func envelopePointsAtOutlineMode() async throws {
        let cache = freshCache()
        let url = try writePDF(pages: 40)
        defer { try? FileManager.default.removeItem(at: url) }

        let attachment = try ChatFileAttachment(contentsOf: url, cache: cache)
        let prompt = attachment.promptText
        #expect(prompt.contains("mode=\"outline\""))
        #expect(prompt.contains("AS A WHOLE"))
    }

    // MARK: - Scanned documents

    /// A PDF whose pages are IMAGES of text — no text layer at all, which is
    /// what a scanner produces. Built by rasterizing rendered text, so the
    /// only way to read it back is recognition.
    private func makeScannedPDF(pages: [String]) throws -> URL {
        let doc = PDFDocument()
        for body in pages {
            let size = NSSize(width: 612, height: 300)
            let image = NSImage(size: size)
            image.lockFocus()
            NSColor.white.setFill()
            NSRect(origin: .zero, size: size).fill()
            (body as NSString).draw(
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
        guard let data = doc.dataRepresentation() else {
            throw CocoaError(.fileWriteUnknown)
        }
        try data.write(to: url)
        return url
    }

    @Test("A scanned PDF is recognized instead of rejected", .timeLimit(.minutes(1)))
    func scannedPDFIsRecognized() async throws {
        // Before OCR support this threw noExtractableText and the user could
        // not attach the file at all.
        let cache = freshCache()
        let url = try makeScannedPDF(pages: ["Quarterly revenue summary"])
        defer { try? FileManager.default.removeItem(at: url) }

        // The fixture genuinely has no text layer — otherwise this test would
        // pass without exercising recognition at all.
        let rawText = PDFDocument(url: url)?.page(at: 0)?.string ?? ""
        #expect(rawText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

        let attachment = try ChatFileAttachment(contentsOf: url, cache: cache)
        #expect(attachment.kind == .pdf)
        #expect(attachment.extractedText.localizedCaseInsensitiveContains("revenue"))
    }

    @Test("A multi-page scan previews eagerly and finishes in the background", .timeLimit(.minutes(2)))
    func scannedPDFDefersTheTail() async throws {
        // Recognition costs ~0.69 s/page, so only a few pages can run while
        // the user waits; the rest must land later without blocking attach.
        let cache = freshCache()
        let pages = (0..<6).map { "Section \($0) heading text" }
        let url = try makeScannedPDF(pages: pages)
        defer { try? FileManager.default.removeItem(at: url) }

        let attachment = try ChatFileAttachment(contentsOf: url, cache: cache)
        #expect(attachment.pageCount == 6)
        // Beyond the eager OCR window, so the total is not yet known.
        #expect(attachment.totalCharacterCount == nil)
        #expect(attachment.hasUnshownContent)

        // read_document waits for the background recognition to finish, so
        // the last page is reachable even though attach never read it.
        let json = try payload(
            await run(["document_id": attachment.id.uuidString, "grep": "Section 5"], cache: cache)
        )
        #expect((json["match_count"] as? Int) ?? 0 >= 1)
    }

    @Test("A text PDF never pays the recognition cost", .timeLimit(.minutes(1)))
    func textPDFSkipsRecognition() async throws {
        // 40 pages of OCR would take ~28s. This completing quickly is the
        // assertion: the selectable-text path must not rasterize anything.
        let cache = freshCache()
        let url = try writePDF(pages: 40)
        defer { try? FileManager.default.removeItem(at: url) }

        let started = Date()
        let attachment = try ChatFileAttachment(contentsOf: url, cache: cache)
        #expect(Date().timeIntervalSince(started) < 2.0)
        #expect(attachment.extractedText.contains("PAGEMARK0"))
    }

    @Test("A stalled extraction is abandoned rather than waited on forever")
    func stalledExtractionDoesNotHang() async throws {
        // Recognition of a 529-page scan takes ~9 minutes, so no fixed timeout
        // can be right for both it and a text extraction that finishes in
        // milliseconds. The wait is bounded by SILENCE instead: nothing here
        // ever reports progress, so the waiter returns the partial entry.
        let cache = freshCache()
        let id = UUID()
        cache.put(id, entry: DocumentContentCache.Entry(filename: "scan.pdf", text: "page one"))
        cache.beginPending(id)
        defer { cache.finishPending(id) }

        let started = Date()
        let entry = cache.getAwaitingCompletion(id, stallTimeout: 0.3)
        #expect(Date().timeIntervalSince(started) < 3.0)
        #expect(entry?.text == "page one")
    }

    @Test("Progress extends the wait past the stall timeout")
    func progressExtendsTheWait() async throws {
        // The live case: work that keeps reporting outlives a stall timeout
        // far shorter than its total runtime, which is what lets a nine-minute
        // recognition finish under a thirty-second stall bound.
        let cache = freshCache()
        let id = UUID()
        cache.put(id, entry: DocumentContentCache.Entry(filename: "scan.pdf", text: "partial"))
        cache.beginPending(id)

        // Report progress well past the stall timeout, then publish. Timing is
        // one-sided: the test only fails if the waiter gives up EARLY, and the
        // stall bound is an order of magnitude under the total, so a loaded
        // machine makes this more forgiving rather than flaky.
        Task.detached {
            for _ in 0..<5 {
                try? await Task.sleep(for: .milliseconds(60))
                cache.reportProgress(id)
            }
            cache.put(id, entry: DocumentContentCache.Entry(filename: "scan.pdf", text: "complete text"))
            cache.finishPending(id)
        }

        let entry = cache.getAwaitingCompletion(id, stallTimeout: 5.0)
        #expect(entry?.text == "complete text")
    }

    // MARK: - An extraction that never finished
    //
    // A large PDF's preview is persisted BEFORE the background pass replaces
    // it with the whole document. The pending mark lives in memory only, so
    // quitting mid-OCR leaves a disk entry that a later launch cannot tell
    // apart from a finished one — and `read_document` would present the first
    // four pages of a 529-page scan as the complete document.

    @Test("A partial extract survives a relaunch still marked incomplete")
    func incompletenessIsPersisted() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let id = UUID()
        let writing = DocumentContentCache(diskDirectory: directory)
        writing.put(id, entry: DocumentContentCache.Entry(
            filename: "scan.pdf",
            text: "first four pages",
            pageCount: 529,
            isComplete: false
        ))

        // A fresh instance is the relaunch: no pending set, no extraction task.
        let relaunched = DocumentContentCache(diskDirectory: directory)
        let entry = try #require(relaunched.get(id))
        #expect(entry.text == "first four pages")
        #expect(!entry.isComplete)
    }

    @Test("An entry written before completeness tracking still reads as complete")
    func legacyEntriesDefaultToComplete() throws {
        // Those predate deferred extraction's disk exposure; defaulting them
        // to partial would put a spurious warning on every old conversation.
        let json = #"{"filename":"old.pdf","text":"whole document"}"#
        let entry = try JSONDecoder().decode(
            DocumentContentCache.Entry.self,
            from: Data(json.utf8)
        )
        #expect(entry.isComplete)
    }

    @Test("A partial extract reports unreachable continuation, not a phantom page")
    func partialExtractKeepsHasMoreTrue() async throws {
        let cache = freshCache()
        let id = UUID()
        cache.put(id, entry: DocumentContentCache.Entry(
            filename: "scan.pdf",
            text: "the only pages that were ever extracted",
            pageCount: 529,
            isComplete: false
        ))

        let result = await run(["document_id": id.uuidString], cache: cache)
        let body = try payload(result)

        // `has_more` means strictly "this cursor can advance". Overloading it
        // to also mean "the source continues" produced a self-contradictory
        // result — has_more with no next_offset — whose only compliant reading
        // was to re-read the same slice until the twelve-call budget ran out.
        #expect(body["has_more"] as? Bool == false)
        #expect(body["next_offset"] == nil)
        #expect(body["continuation_unavailable"] as? Bool == true)
        #expect(body["extract_complete"] as? Bool == false)
        let note = try #require(body["note"] as? String)
        #expect(note.localizedCaseInsensitiveContains("attach the file again"))
        #expect(note.localizedCaseInsensitiveContains("do not retry"))
    }

    @Test("A partial extract mid-way still offers its next page")
    func partialExtractStillPagesWithinWhatItHas() async throws {
        // Incompleteness of the SOURCE must not suppress paging through the
        // part that was captured.
        let cache = freshCache()
        let id = UUID()
        cache.put(id, entry: DocumentContentCache.Entry(
            filename: "scan.pdf",
            text: String(repeating: "captured text\n", count: 4_000),
            pageCount: 529,
            isComplete: false
        ))

        let body = try payload(await run(["document_id": id.uuidString], cache: cache))

        #expect(body["has_more"] as? Bool == true)
        #expect(body["next_offset"] as? Int != nil)
        #expect(body["continuation_unavailable"] == nil)
        #expect(body["extract_complete"] as? Bool == false)
        // The warning must not tell the model to stop reading while there is
        // still captured text one call away — that abandons cache it has.
        let note = try #require(body["note"] as? String)
        #expect(note.localizedCaseInsensitiveContains("keep reading from 'next_offset'"))
        #expect(!note.localizedCaseInsensitiveContains("will not return more"))
    }

    @Test("An extract stopped by the size ceiling does not advise re-attaching")
    func sizeCeilingRemediationIsNotRetry() async throws {
        // Re-attaching truncates at exactly the same point, so the interrupted
        // pass's advice sends the user around a loop that cannot terminate.
        let cache = freshCache()
        let id = UUID()
        cache.put(id, entry: DocumentContentCache.Entry(
            filename: "huge.pdf",
            text: "as much as Rapid will ever extract",
            pageCount: 40_000,
            isComplete: false,
            hitSizeCeiling: true
        ))

        let body = try payload(await run(["document_id": id.uuidString], cache: cache))
        let note = try #require(body["note"] as? String)

        #expect(note.localizedCaseInsensitiveContains("larger than Rapid can extract"))
        #expect(note.localizedCaseInsensitiveContains("split the file"))
        #expect(!note.localizedCaseInsensitiveContains("attach the file again"))
    }

    @Test("The ceiling reason survives a relaunch alongside the partial text")
    func ceilingReasonIsPersisted() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let id = UUID()
        DocumentContentCache(diskDirectory: directory).put(id, entry: DocumentContentCache.Entry(
            filename: "huge.pdf",
            text: "head only",
            isComplete: false,
            hitSizeCeiling: true
        ))

        let entry = try #require(DocumentContentCache(diskDirectory: directory).get(id))
        #expect(!entry.isComplete)
        #expect(entry.hitSizeCeiling)
    }

    @Test("A complete extract carries no incompleteness warning")
    func completeExtractIsNotAnnotated() async throws {
        let cache = freshCache()
        let id = store("the whole document", in: cache)

        let body = try payload(await run(["document_id": id.uuidString], cache: cache))

        #expect(body["has_more"] as? Bool == false)
        #expect(body["extract_complete"] == nil)
        #expect(body["continuation_unavailable"] == nil)
    }

    @Test("Outline and grep also disclose an unfinished extract")
    func otherModesAnnotateIncompleteness() async throws {
        let cache = freshCache()
        let id = UUID()
        cache.put(id, entry: DocumentContentCache.Entry(
            filename: "scan.pdf",
            text: "第 1 章 总则\n合同的赔偿条款在此。",
            pageCount: 529,
            isComplete: false
        ))

        for arguments in [
            ["document_id": id.uuidString, "mode": "outline"],
            ["document_id": id.uuidString, "grep": "赔偿"],
        ] {
            let body = try payload(await run(arguments, cache: cache))
            #expect(body["extract_complete"] as? Bool == false)
            // Both modes expose a way back into the captured text: outline
            // rows and grep passages carry reusable offsets, while their notes
            // also offer offset=0 as the sequential fallback.
            #expect(body["continuation_unavailable"] == nil)
            let note = try #require(body["note"] as? String)
            #expect(note.localizedCaseInsensitiveContains("attach the file again"))
            #expect(note.localizedCaseInsensitiveContains("captured extract remains readable"))
            #expect(!note.localizedCaseInsensitiveContains("further offsets will not return more"))
        }
    }

    @Test("A stalled wait reports a running extraction, not a permanent interruption")
    func stalledWaitPreservesPendingState() async throws {
        let cache = freshCache()
        let id = UUID()
        cache.put(id, entry: DocumentContentCache.Entry(
            filename: "slow-scan.pdf",
            text: "first recognized page",
            pageCount: 200,
            isComplete: false
        ))
        cache.beginPending(id)
        let generation = cache.generation(for: id)
        let completion = Task.detached {
            try? await Task.sleep(for: .milliseconds(150))
            _ = cache.publish(
                id,
                entry: DocumentContentCache.Entry(
                    filename: "slow-scan.pdf",
                    text: "the complete recognized document",
                    pageCount: 200
                ),
                ifGenerationIs: generation
            )
            cache.finishPending(id)
        }

        let partialResult = await run(
            ["document_id": id.uuidString],
            cache: cache,
            stallTimeout: 0.03
        )
        await completion.value
        let partial = try payload(partialResult)

        #expect(partial["extract_pending"] as? Bool == true)
        #expect(partial["continuation_unavailable"] == nil)
        let note = try #require(partial["note"] as? String)
        #expect(note.localizedCaseInsensitiveContains("background extraction is still running"))
        #expect(note.localizedCaseInsensitiveContains("retry read_document later"))
        #expect(!note.localizedCaseInsensitiveContains("attach the file again"))
        #expect(!note.localizedCaseInsensitiveContains("cannot be resumed"))

        let complete = try payload(await run(["document_id": id.uuidString], cache: cache))
        #expect(complete["content"] as? String == "the complete recognized document")
        #expect(complete["extract_pending"] == nil)
        #expect(complete["extract_complete"] == nil)
    }

    // MARK: - Outline fallback allocation

    @Test("Inferred outline stops scanning once the row cap is reached")
    func inferredOutlineIsBoundedByRows() throws {
        // ``split(separator:)`` materialized the WHOLE `[Substring]` before
        // the row cap could stop anything, so a densely-newlined 20M-character
        // extract allocated millions of slices to return a few hundred rows.
        // Far more headings than the cap, each on its own short line.
        let headings = (1...(ReadDocumentTool.maxOutlineRows * 4))
            .map { "\($0) Section title" }
            .joined(separator: "\n")
        let entry = DocumentContentCache.Entry(filename: "report.pdf", text: headings)

        let rows = ReadDocumentTool.inferredOutline(in: entry)

        #expect(rows.count == ReadDocumentTool.maxOutlineRows * 2)
        #expect(rows.first?.title == "1 Section title")
    }

    @Test("Lazy line scanning reports the same rows, pages and offsets as before")
    func inferredOutlineKeepsOffsets() throws {
        let text = """
        [Page 1]
        1 Introduction
        body text that is far too long to be mistaken for a heading, at length.
        [Page 7]
        1.2 Scope
        第 2 章 定义
        """
        let entry = DocumentContentCache.Entry(filename: "report.pdf", text: text)

        let rows = ReadDocumentTool.inferredOutline(in: entry)

        #expect(rows.map(\.title) == ["1 Introduction", "1.2 Scope", "第 2 章 定义"])
        #expect(rows.map(\.depth) == [0, 1, 0])
        #expect(rows.map(\.page) == [1, 7, 7])
        // Offsets must still address the real character positions, since the
        // model passes them straight back as `offset`.
        for row in rows {
            let offset = try #require(row.offset)
            let start = entry.index(atCharacterOffset: offset)
            #expect(entry.text[start...].hasPrefix(row.title))
        }
    }

    @Test("A trailing line with no newline is still scanned")
    func inferredOutlineReadsFinalLine() throws {
        let entry = DocumentContentCache.Entry(
            filename: "report.pdf",
            text: "preamble prose that says nothing structural at all here.\n3 Conclusion"
        )

        #expect(ReadDocumentTool.inferredOutline(in: entry).map(\.title) == ["3 Conclusion"])
    }
}
