import Foundation

/// `read_document` — page through the full text of a document the USER attached.
///
/// ## Why this is not a filesystem tool
///
/// ``BuiltinToolRegistry`` deliberately ships no filesystem tools: this build
/// has no ``SandboxManager``, and a tool that takes a path could be pointed at
/// anything on the user's disk by a model that read an injected instruction out
/// of a document.
///
/// This tool takes no path. It addresses documents by the UUID handle minted
/// when the user attached a file, and resolves it exclusively through
/// ``DocumentContentCache`` — bytes the user already chose to hand over and
/// that Rapid already extracted. A handle that is not in the cache is a miss,
/// not a lookup. So the tool cannot widen what the model can reach beyond what
/// the user attached, and needs no approval prompt of its own: the drop or the
/// file-picker WAS the approval.
///
/// ## Pagination
///
/// Mirrors ``BrowseTool``: a budgeted slice plus a `next_offset` cursor, so a
/// long document is read in bounded pieces instead of being force-fit into one
/// prompt. `grep` short-circuits the linear walk — on a 500-page PDF, paging
/// blindly would burn the whole tool budget before reaching the relevant part.
///
/// ## Outline
///
/// Paging and grep both assume the model knows WHAT it is looking for. A
/// whole-document question ("what does this say?", "summarize it") does not:
/// grep has no term to search, and reading a 302-page book sequentially would
/// need ~33 slices against a budget of 12. `mode: "outline"` answers that
/// shape directly, returning the document's structural map in one call — the
/// 289 headings of a real book cost ~4,300 tokens, and each carries an offset
/// the model can use to then read exactly the section it cares about.
enum ReadDocumentTool {
    /// Characters returned per call. Matches ``BrowseTool/charBudget`` so a
    /// document page and a web page cost the model's context the same.
    static let charBudget = 15_000
    /// Characters of context shown around each `grep` hit.
    static let grepContextRadius = 1_000
    /// Maximum `grep` hits reported in one call. Bounds the result size when a
    /// pattern matches on nearly every line.
    static let maxGrepMatches = 10
    /// Maximum characters copied into a passage's `match` field. The passage
    /// itself is budgeted separately, but a pattern such as `(?s).*` can make
    /// the raw match span the entire 20-million-character document.
    static let maxGrepMatchCharacters = 1_000
    /// Longest `grep` pattern accepted.
    ///
    /// A regex is code the model supplies and this process runs over as much
    /// as 20,000,000 characters. Pattern length is the one cheap proxy for how
    /// much nesting and alternation that code can contain, so it is capped
    /// before compilation rather than after the damage.
    static let maxGrepPatternLength = 200
    /// Wall-clock ceiling on a single `grep` scan.
    ///
    /// The match LIMIT alone does not bound the work: a pattern that matches
    /// nothing scans the whole extract regardless, and one that backtracks
    /// (`(a+)+b`) can spend unbounded time inside a single match attempt.
    /// NSRegularExpression's enumeration block is the only place this code
    /// regains control, so the deadline is checked there and a scan that
    /// overruns returns what it found instead of running to completion.
    static let grepTimeBudget: TimeInterval = 2.0
    /// Token ceiling for an outline response. Roughly a third of a page slice:
    /// an outline is a map, and one that fills the context defeats its purpose.
    static let outlineTokenBudget = 2_000
    /// Headings emitted before the map is trimmed to shallower levels.
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
        cache: DocumentContentCache = .shared
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
        // The security boundary: only documents the user attached (and that are
        // still cached) resolve. Nothing here can reach an arbitrary file.
        //
        // A large PDF finishes extracting on a background task, so this waits
        // for that work rather than reporting a document the user can plainly
        // see as missing.
        guard let entry = cache.getAwaitingCompletion(id) else {
            return err(tool, "no attached document with id \(rawID) — Rapid keeps the full text of an attachment for \(DocumentContentCache.retentionDays) days, and this one has expired (or was deleted). Tell the user the document is no longer available to read and ask them to attach the file again.")
        }

        if args.mode?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "outline" {
            return outlineResult(id: rawID, entry: entry)
        }
        if let pattern = args.grep?.trimmingCharacters(in: .whitespacesAndNewlines), !pattern.isEmpty {
            return grepResult(tool: tool, id: rawID, entry: entry, pattern: pattern)
        }
        return sliceResult(id: rawID, entry: entry, offset: max(0, args.offset ?? 0))
    }

    // MARK: - Outline

    static func outlineResult(
        id: String,
        entry: DocumentContentCache.Entry
    ) -> ToolCallResult {
        // Bookmarks first — authored by whoever made the PDF, with exact
        // pages. Only fall back to guessing from prose when there are none.
        var rows = entry.outline
        var source = "bookmarks"
        if rows.isEmpty {
            rows = inferredOutline(in: entry)
            source = "inferred"
        }

        guard !rows.isEmpty else {
            // Saying so plainly beats returning an empty list, which the model
            // could read as "this document is empty".
            return ToolCallResult(toolCallID: "", content: jsonString([
                "document_id": id,
                "filename": entry.filename,
                "outline": [],
                "total_chars": entry.count,
                "note": "This document has no detectable section structure — no PDF bookmarks and no heading-like lines. Read it sequentially with offset=0, or use 'grep' if you know what you are looking for.",
            ]), isError: false)
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
        return ToolCallResult(toolCallID: "", content: jsonString(payload), isError: false)
    }

    /// Trim an outline to the token budget by dropping the DEEPEST levels
    /// first, then truncating if even the top level overflows.
    ///
    /// Depth-first dropping preserves the map's shape: a reader served every
    /// chapter is better off than one served the first 40 sub-sub-sections of
    /// chapter one and nothing after it.
    static func budgeted(
        _ rows: [DocumentContentCache.OutlineNode]
    ) -> (rows: [DocumentContentCache.OutlineNode], keptDepth: Int?) {
        func cost(_ rows: [DocumentContentCache.OutlineNode]) -> Int {
            TokenEstimate.tokens(in: rows.map(\.title).joined(separator: "\n"))
                // Each row also ships depth/page/offset as JSON — roughly a
                // dozen tokens of envelope apiece.
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
        // Even top-level headings overflow: keep as many as fit, in order.
        var kept: [DocumentContentCache.OutlineNode] = []
        for row in rows where row.depth == 0 {
            let next = kept + [row]
            if next.count > maxOutlineRows || cost(next) > outlineTokenBudget { break }
            kept = next
        }
        return (kept, kept.isEmpty ? nil : 0)
    }

    /// Derive a rough outline from heading-like lines, for documents with no
    /// bookmarks — exported reports, plain text, anything not authored with a
    /// table of contents.
    ///
    /// Deliberately narrow. A permissive heading regex over a 302-page book
    /// matched 882 lines, most of them body text and table-of-contents
    /// residue; a map that wrong is worse than none. So this accepts only
    /// numbered headings and explicit chapter markers, and only on short
    /// standalone lines.
    static func inferredOutline(
        in entry: DocumentContentCache.Entry
    ) -> [DocumentContentCache.OutlineNode] {
        let patterns: [(NSRegularExpression, Int)] = [
            // "第 3 章" / "第三节" — depth comes from the marker itself.
            (try! NSRegularExpression(pattern: #"^第\s*[0-9一二三四五六七八九十百]+\s*[章篇部]"#), 0),
            (try! NSRegularExpression(pattern: #"^第\s*[0-9一二三四五六七八九十百]+\s*[节節]"#), 1),
            // "1 Title", "1.2 Title", "1.2.3 Title" — depth from the dots.
            (try! NSRegularExpression(pattern: #"^\d+(\.\d+){0,3}\s+\S"#), -1),
        ]

        var nodes: [DocumentContentCache.OutlineNode] = []
        var offset = 0
        var currentPage: Int?
        for line in entry.text.split(separator: "\n", omittingEmptySubsequences: false) {
            defer { offset += line.count + 1 }
            guard nodes.count < maxOutlineRows * 2 else { break }

            if line.hasPrefix("[Page "), line.hasSuffix("]"),
               let page = Int(line.dropFirst("[Page ".count).dropLast()) {
                currentPage = page
                continue
            }

            let title = line.trimmingCharacters(in: .whitespaces)
            // Headings are short and standalone; a long line is prose.
            guard title.count >= 3, title.count <= 90 else { continue }
            let range = NSRange(location: 0, length: (title as NSString).length)
            for (regex, fixedDepth) in patterns
            where regex.firstMatch(in: title, range: range) != nil {
                var depth = fixedDepth
                if depth < 0 {
                    // "1.2.3" nests one level per dot.
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
        return nodes
    }

    // MARK: - Sequential paging

    static func sliceResult(
        id: String,
        entry: DocumentContentCache.Entry,
        offset: Int
    ) -> ToolCallResult {
        let total = entry.count
        let start = min(max(0, offset), total)
        var end = min(start + charBudget, total)
        // Snap the cut back to a line boundary when there's more to come, so a
        // page doesn't end mid-line. Keep `end` (and thus next_offset) exactly
        // at the emitted length — no off-by-one in the cursor the model reuses.
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
            "has_more": hasMore,
        ]
        if let pages = entry.pageCount { payload["total_pages"] = pages }
        if hasMore {
            payload["next_offset"] = end
            payload["note"] = "Showing characters \(start)–\(end) of \(total). Call read_document again with offset=\(end) to continue, or pass a 'grep' pattern to jump to a specific passage."
        } else if start >= total && total > 0 {
            payload["note"] = "offset \(start) is at or past the end of this \(total)-character document."
        }
        return ToolCallResult(toolCallID: "", content: jsonString(payload), isError: false)
    }

    // MARK: - Grep

    /// Search the extract for `pattern`, returning at most
    /// ``maxGrepMatches`` passages.
    ///
    /// ## Why this enumerates instead of collecting
    ///
    /// The obvious spelling, ``NSRegularExpression/matches(in:options:range:)``
    /// followed by ``prefix(maxGrepMatches)``, allocates an
    /// `NSTextCheckingResult` for EVERY match before ten are kept. The pattern
    /// is model-supplied and the extract may hold 20,000,000 characters, so
    /// `.` alone would materialize twenty million objects to return ten
    /// passages — a resource exhaustion any document able to reach the model
    /// can trigger. Enumeration with an early `stop` never holds more than the
    /// current match.
    ///
    /// ## Why a deadline as well as a match cap
    ///
    /// The cap bounds OUTPUT, not WORK. A pattern that matches nothing still
    /// scans the entire extract, and one that backtracks (`(a+)+$`) can spend
    /// unbounded time inside a single match attempt without ever producing a
    /// result to count. ``.reportProgress`` makes the enumeration block fire
    /// periodically DURING such an attempt with a nil result, which is the
    /// only point at which this code can regain control; checking the deadline
    /// there bounds both shapes. ``maxGrepPatternLength`` caps how much
    /// nesting the pattern can express in the first place.
    ///
    /// A truncated scan is reported as such rather than presented as a
    /// complete match count, so the model is never told a document contains
    /// exactly the number of hits that happened to fit in the budget.
    static func grepResult(
        tool: String,
        id: String,
        entry: DocumentContentCache.Entry,
        pattern: String
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

        // NSRegularExpression works in UTF-16 offsets; the cache paginates in
        // Character offsets. Convert every reported range back through the
        // String's own index space so a document containing emoji or CJK text
        // cannot produce an offset the caller can't reuse.
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
            // Fires with a nil match while a single attempt is still running.
            // Both paths must honour the deadline or the interruption point is
            // only reachable by patterns that are already cheap.
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
                // Checkpoint-based rather than `distance(from: startIndex)`:
                // the linear spelling would walk the whole prefix per hit.
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
            return ToolCallResult(toolCallID: "", content: jsonString(payload), isError: false)
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
            // "at least": the scan stopped early, so the true total is unknown
            // and reporting `passages.count` as THE count would be a claim this
            // search never established.
            payload["note"] = "Showing the first \(passages.count) matches; the document contains at least that many and the search stopped before reaching the end. Narrow the pattern, or use the 'offset' of a passage to read around it sequentially."
        }
        return ToolCallResult(toolCallID: "", content: jsonString(payload), isError: false)
    }

    // MARK: - Helpers

    /// Index of the last "\n" in the entry's text within [from, to), or nil.
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

    static func jsonString(_ payload: [String: Any]) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys]),
              let s = String(data: data, encoding: .utf8) else {
            return "{\"error\":\"failed to encode read_document result\"}"
        }
        return s
    }
}
