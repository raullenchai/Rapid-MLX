import Foundation

/// Reads only user-attached cache entries by UUID, never filesystem paths.
/// Supports bounded sequential pages, structural outlines and regex search.
enum ReadDocumentTool {
    static let charBudget = 15_000
    static let grepContextRadius = 1_000
    static let maxGrepMatches = 10
    static let maxGrepMatchCharacters = 1_000
    static let maxGrepPatternLength = 200
    static let grepTimeBudget: TimeInterval = 2.0
    static let outlineTokenBudget = 2_000
    static let maxOutlineRows = 400

    static let definition = ToolDefinition(
        name: "read_document",
        description: "Read a document the user attached to this conversation. Attachments show only a short preview inline; use this to see the rest. THREE modes: (1) mode='outline' returns the document's table of contents — section titles with their page and character offset. Start here for any question about the document AS A WHOLE, such as summarizing it or asking what it covers, then read the sections that matter. (2) 'grep' with a regular expression jumps straight to matching passages — use when you already know the term you want. (3) 'offset' reads sequentially from a character position; the result carries 'next_offset' and 'has_more' to continue. Sequential reading is the slowest way to cover a long document, so prefer outline or grep first.",
        parameters: .object([
            "type": .string("object"),
            "properties": .object([
                "document_id": .object([
                    "type": .string("string"),
                    "description": .string("The attachment id, shown in the BEGIN RAPID ATTACHMENT header of the document preview.")
                ]),
                "mode": .object([
                    "type": .string("string"),
                    "enum": .array([.string("outline"), .string("read")]),
                    "description": .string("'outline' returns the section map of the whole document in one call — best for summarizing or working out what the document covers. 'read' (the default) returns document text at 'offset'.")
                ]),
                "offset": .object([
                    "type": .string("integer"),
                    "description": .string("Character offset to read from. Omit (or 0) for the start; pass the 'next_offset' from a previous call to continue. Ignored when 'grep' is set.")
                ]),
                "grep": .object([
                    "type": .string("string"),
                    "description": .string("Regular expression (case-insensitive). Returns the matching passages with surrounding context instead of a sequential page. Use this to find specific terms, sections, or figures in a long document.")
                ])
            ]),
            "required": .array([.string("document_id")])
        ])
    )

    struct Args: Decodable {
        let document_id: String
        let mode: String?
        let offset: Int?
        let grep: String?
    }

    static func run(
        arguments: String,
        cache: DocumentContentCache = .shared,
        stallTimeout: TimeInterval = 30
    ) async -> ToolCallResult {
        let tool = "read_document"
        guard let data = arguments.data(using: .utf8),
              let args = try? JSONDecoder().decode(Args.self, from: data) else {
            return err(tool, "could not parse arguments JSON")
        }

        let rawID = args.document_id.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let id = UUID(uuidString: rawID) else {
            return err(tool, "'\(rawID)' is not a valid document id — use the id from the document's BEGIN RAPID ATTACHMENT header")
        }
        guard let awaited = cache.getAwaitingCompletionStatus(
            id,
            stallTimeout: stallTimeout
        ) else {
            return err(tool, "no attached document with id \(rawID) — Rapid keeps the full text of an attachment for \(DocumentContentCache.retentionDays) days, and this one has expired (or was deleted). Tell the user the document is no longer available to read and ask them to attach the file again.")
        }
        let entry = awaited.entry

        if args.mode?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "outline" {
            return outlineResult(
                id: rawID,
                entry: entry,
                extractionPending: awaited.extractionPending
            )
        }
        if let pattern = args.grep?.trimmingCharacters(in: .whitespacesAndNewlines), !pattern.isEmpty {
            return grepResult(
                tool: tool,
                id: rawID,
                entry: entry,
                pattern: pattern,
                extractionPending: awaited.extractionPending
            )
        }
        return sliceResult(
            id: rawID,
            entry: entry,
            offset: max(0, args.offset ?? 0),
            extractionPending: awaited.extractionPending
        )
    }

    // MARK: - Outline

    static func outlineResult(
        id: String,
        entry: DocumentContentCache.Entry,
        extractionPending: Bool = false
    ) -> ToolCallResult {
        var rows = entry.outline
        var source = "bookmarks"
        if rows.isEmpty {
            rows = inferredOutline(in: entry)
            source = "inferred"
        }

        guard !rows.isEmpty else {
            return result(
                [
                    "document_id": id,
                    "filename": entry.filename,
                    "outline": [],
                    "total_chars": entry.count,
                    "note": "This document has no detectable section structure — no PDF bookmarks and no heading-like lines. Read it sequentially with offset=0, or use 'grep' if you know what you are looking for.",
                ],
                entry: entry,
                capturedContinuationAvailable: true,
                extractionPending: extractionPending
            )
        }

        let (trimmed, keptDepth) = budgeted(rows)
        var payload: [String: Any] = [
            "document_id": id,
            "filename": entry.filename,
            "outline_source": source,
            "outline": trimmed.map { node -> [String: Any] in
                var row: [String: Any] = ["title": node.title, "depth": node.depth]
                if let page = node.page { row["page"] = page }
                if let offset = node.offset { row["offset"] = offset }
                return row
            },
            "total_chars": entry.count,
        ]
        if let pages = entry.pageCount { payload["total_pages"] = pages }

        var note = "Each entry's 'offset' can be passed back as read_document's 'offset' to read that section."
        if trimmed.count < rows.count {
            payload["entries_omitted"] = rows.count - trimmed.count
            if let depth = keptDepth {
                note += " Showing the top \(depth + 1) level(s) of \(rows.count) total entries; deeper subsections are omitted."
            } else {
                note += " Showing \(trimmed.count) of \(rows.count) entries."
            }
        }
        if source == "inferred" {
            note += " This document carries no bookmarks, so the structure was inferred from heading-like lines and may be imperfect."
        }
        payload["note"] = note
        return result(
            payload,
            entry: entry,
            capturedContinuationAvailable: true,
            extractionPending: extractionPending
        )
    }

    /// Drops deepest levels first to preserve the outline's overall shape.
    static func budgeted(
        _ rows: [DocumentContentCache.OutlineNode]
    ) -> (rows: [DocumentContentCache.OutlineNode], keptDepth: Int?) {
        func cost(_ rows: [DocumentContentCache.OutlineNode]) -> Int {
            TokenEstimate.tokens(in: rows.map(\.title).joined(separator: "\n"))
                + rows.count * 12
        }

        if rows.count <= maxOutlineRows, cost(rows) <= outlineTokenBudget {
            return (rows, nil)
        }
        var depth = rows.map(\.depth).max() ?? 0
        while depth > 0 {
            depth -= 1
            let kept = rows.filter { $0.depth <= depth }
            if kept.count <= maxOutlineRows, cost(kept) <= outlineTokenBudget {
                return (kept, depth)
            }
        }
        var kept: [DocumentContentCache.OutlineNode] = []
        for row in rows where row.depth == 0 {
            let next = kept + [row]
            if next.count > maxOutlineRows || cost(next) > outlineTokenBudget { break }
            kept = next
        }
        return (kept, kept.isEmpty ? nil : 0)
    }

    /// Infers a conservative outline from numbered or explicit chapter headings.
    static func inferredOutline(
        in entry: DocumentContentCache.Entry
    ) -> [DocumentContentCache.OutlineNode] {
        let patterns: [(NSRegularExpression, Int)] = [
            (try! NSRegularExpression(pattern: #"^第\s*[0-9一二三四五六七八九十百]+\s*[章篇部]"#), 0),
            (try! NSRegularExpression(pattern: #"^第\s*[0-9一二三四五六七八九十百]+\s*[节節]"#), 1),
            (try! NSRegularExpression(pattern: #"^\d+(\.\d+){0,3}\s+\S"#), -1),
        ]

        var nodes: [DocumentContentCache.OutlineNode] = []
        var offset = 0
        var currentPage: Int?
        // Scan lazily so the row cap also bounds temporary allocations.
        let text = entry.text
        var lineStart = text.startIndex
        while nodes.count < maxOutlineRows * 2 {
            let lineEnd = text[lineStart...].firstIndex(of: "\n") ?? text.endIndex
            let line = text[lineStart..<lineEnd]
            defer {
                offset += line.count + 1
                lineStart = lineEnd < text.endIndex ? text.index(after: lineEnd) : text.endIndex
            }

            if line.hasPrefix("[Page "), line.hasSuffix("]"),
               let page = Int(line.dropFirst("[Page ".count).dropLast()) {
                currentPage = page
                if lineEnd == text.endIndex { break }
                continue
            }

            let title = line.trimmingCharacters(in: .whitespaces)
            if title.count >= 3, title.count <= 90 {
                let range = NSRange(location: 0, length: (title as NSString).length)
                for (regex, fixedDepth) in patterns
                where regex.firstMatch(in: title, range: range) != nil {
                    var depth = fixedDepth
                    if depth < 0 {
                        depth = title.prefix { $0.isNumber || $0 == "." }
                            .filter { $0 == "." }.count
                    }
                    nodes.append(DocumentContentCache.OutlineNode(
                        title: title,
                        depth: depth,
                        page: currentPage,
                        offset: offset
                    ))
                    break
                }
            }
            if lineEnd == text.endIndex { break }
        }
        return nodes
    }

    // MARK: - Sequential paging

    static func sliceResult(
        id: String,
        entry: DocumentContentCache.Entry,
        offset: Int,
        extractionPending: Bool = false
    ) -> ToolCallResult {
        let total = entry.count
        let start = min(max(0, offset), total)
        var end = min(start + charBudget, total)
        // Keep `next_offset` aligned with the emitted line-bounded content.
        if end < total {
            let lower = max(start, end - 500)
            if let nl = lastNewline(in: entry, from: lower, to: end) { end = nl + 1 }
        }
        let content = String(entry.text[entry.index(atCharacterOffset: start)..<entry.index(atCharacterOffset: end)])
        let hasMore = end < total

        var payload: [String: Any] = [
            "document_id": id,
            "filename": entry.filename,
            "content": content,
            "offset": start,
            "total_chars": total,
            // `has_more` describes this captured text, not missing source text.
            "has_more": hasMore,
        ]
        if let pages = entry.pageCount { payload["total_pages"] = pages }
        if hasMore {
            payload["next_offset"] = end
            payload["note"] = "Showing characters \(start)–\(end) of \(total). Call read_document again with offset=\(end) to continue, or pass a 'grep' pattern to jump to a specific passage."
        } else if start >= total && total > 0 {
            payload["note"] = "offset \(start) is at or past the end of this \(total)-character document."
        }
        if !hasMore, !entry.isComplete {
            payload["continuation_unavailable"] = true
        }
        return result(
            payload,
            entry: entry,
            capturedContinuationAvailable: hasMore,
            extractionPending: extractionPending
        )
    }

    /// Adds accurate continuation advice to every incomplete-extract mode.
    static func annotatingIncompleteExtract(
        _ payload: [String: Any],
        entry: DocumentContentCache.Entry,
        capturedContinuationAvailable: Bool = false,
        extractionPending: Bool = false
    ) -> [String: Any] {
        guard !entry.isComplete else { return payload }
        var payload = payload
        payload["extract_complete"] = false
        if extractionPending {
            payload.removeValue(forKey: "continuation_unavailable")
            payload["extract_pending"] = true
        } else if !capturedContinuationAvailable,
                  payload["continuation_unavailable"] == nil {
            payload["continuation_unavailable"] = true
        }

        var warning = "WARNING: only the first \(entry.count) characters of this document were extracted"
        if extractionPending {
            warning += " so far — the background extraction is still running. Do not ask the user to re-attach the file. Retry read_document later for text that has not landed yet."
        } else if entry.hitSizeCeiling {
            warning += " — the document is larger than Rapid can extract, so this is all there will ever be. Attaching the file again will truncate at the same point; do not suggest it. If the user needs the rest, they must split the file or extract the part they care about."
        } else {
            warning += " — the pass that would have read the rest did not finish (it was cancelled, or interrupted by a quit) and cannot be resumed. Tell the user to remove this attachment and attach the file again."
        }
        if capturedContinuationAvailable {
            warning += payload["next_offset"] != nil
                ? " The captured part continues past this slice: keep reading from 'next_offset' before you conclude anything."
                : " The captured extract remains readable: use the offsets above, or offset=0, before you conclude anything."
        } else if !extractionPending {
            warning += " Reading further offsets will not return more; do not retry for the missing part."
        }
        warning += " Do not treat what you have read as the whole document, or conclude anything from the absence of something in it."

        payload["note"] = (payload["note"] as? String).map { "\($0) \(warning)" } ?? warning
        return payload
    }

    // MARK: - Grep

    /// Enumerates model-supplied regex matches with time, count and output caps.
    static func grepResult(
        tool: String,
        id: String,
        entry: DocumentContentCache.Entry,
        pattern: String,
        extractionPending: Bool = false
    ) -> ToolCallResult {
        guard pattern.count <= maxGrepPatternLength else {
            return err(
                tool,
                "grep pattern is too long (\(pattern.count) characters, limit \(maxGrepPatternLength)). Search for a distinctive phrase instead of an elaborate expression."
            )
        }
        let regex: NSRegularExpression
        do {
            regex = try NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
        } catch {
            return err(tool, "invalid regular expression '\(pattern)': \(error.localizedDescription)")
        }

        // Convert UTF-16 regex ranges through String so returned cursors are reusable.
        let text = entry.text
        let ns = text as NSString
        let deadline = Date().addingTimeInterval(grepTimeBudget)
        var passages: [[String: Any]] = []
        var budgetLeft = charBudget
        var searchComplete = true

        regex.enumerateMatches(
            in: text,
            options: [.reportProgress],
            range: NSRange(location: 0, length: ns.length)
        ) { match, _, stop in
            if Date() >= deadline {
                searchComplete = false
                stop.pointee = true
                return
            }
            guard let match else { return }
            guard passages.count < maxGrepMatches, budgetLeft > 0 else {
                searchComplete = false
                stop.pointee = true
                return
            }
            guard let range = Range(match.range, in: text) else { return }

            let lower = text.index(
                range.lowerBound,
                offsetBy: -grepContextRadius,
                limitedBy: text.startIndex
            ) ?? text.startIndex
            let upper = text.index(
                range.upperBound,
                offsetBy: grepContextRadius,
                limitedBy: text.endIndex
            ) ?? text.endIndex
            let passageUpper = text.index(
                lower,
                offsetBy: budgetLeft,
                limitedBy: upper
            ) ?? upper
            let passage = String(text[lower..<passageUpper])
            budgetLeft -= passage.count
            let matchPrefix = text[range].prefix(maxGrepMatchCharacters)
            var passagePayload: [String: Any] = [
                "offset": entry.characterOffset(of: lower),
                "match": String(matchPrefix),
                "text": passage,
            ]
            if matchPrefix.endIndex < range.upperBound {
                passagePayload["match_truncated"] = true
            }
            passages.append(passagePayload)
        }

        guard !passages.isEmpty else {
            var payload: [String: Any] = [
                "document_id": id,
                "filename": entry.filename,
                "grep": pattern,
                "match_count": 0,
                "search_complete": searchComplete,
                "total_chars": entry.count,
            ]
            payload["note"] = searchComplete
                ? "No match for '\(pattern)' in this document. Try a broader pattern, or read sequentially with offset=0."
                : "Search for '\(pattern)' was stopped after \(Int(grepTimeBudget))s without finding a match — the pattern is too expensive to run over this document. Try a plain phrase instead of an elaborate expression."
            return result(
                payload,
                entry: entry,
                capturedContinuationAvailable: true,
                extractionPending: extractionPending
            )
        }

        var payload: [String: Any] = [
            "document_id": id,
            "filename": entry.filename,
            "grep": pattern,
            "match_count": passages.count,
            "search_complete": searchComplete,
            "passages": passages,
            "total_chars": entry.count,
        ]
        if let pages = entry.pageCount { payload["total_pages"] = pages }
        if searchComplete {
            payload["note"] = "Each passage includes surrounding context. Use a passage 'offset' with read_document to read forward from there."
        } else {
            payload["note"] = "Showing the first \(passages.count) matches; the document contains at least that many and the search stopped before reaching the end. Narrow the pattern, or use the 'offset' of a passage to read around it sequentially."
        }
        return result(
            payload,
            entry: entry,
            capturedContinuationAvailable: true,
            extractionPending: extractionPending
        )
    }

    // MARK: - Helpers

    private static func lastNewline(
        in entry: DocumentContentCache.Entry,
        from: Int,
        to: Int
    ) -> Int? {
        guard from < to else { return nil }
        let text = entry.text
        let hi = entry.index(atCharacterOffset: to)
        var idx = entry.index(atCharacterOffset: from)
        var found: Int? = nil
        var pos = from
        while idx < hi {
            if text[idx] == "\n" { found = pos }
            idx = text.index(after: idx)
            pos += 1
        }
        return found
    }

    private static func err(_ tool: String, _ message: String) -> ToolCallResult {
        ToolCallResult(toolCallID: "", content: "\(tool) error: \(message)", isError: true)
    }

    private static func result(
        _ payload: [String: Any],
        entry: DocumentContentCache.Entry,
        capturedContinuationAvailable: Bool,
        extractionPending: Bool
    ) -> ToolCallResult {
        ToolCallResult(
            toolCallID: "",
            content: jsonString(annotatingIncompleteExtract(
                payload,
                entry: entry,
                capturedContinuationAvailable: capturedContinuationAvailable,
                extractionPending: extractionPending
            )),
            isError: false
        )
    }

    static func jsonString(_ payload: [String: Any]) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys]),
              let s = String(data: data, encoding: .utf8) else {
            return "{\"error\":\"failed to encode read_document result\"}"
        }
        return s
    }
}
