import Foundation

/// Converts an HTML document into readable Markdown for the ``browse`` tool.
///
/// This is a pragmatic "readability-lite" extractor, not a full
/// Mozilla-Readability port: it drops non-content elements (script / style /
/// nav chrome), prefers the main article region when the page marks one, and
/// linearises the remaining block/inline structure into Markdown the model can
/// read. It is intentionally a single linear tokenizer (no regex over HTML, no
/// DOM) so its behaviour on malformed / adversarial markup is bounded and
/// predictable: unknown tags degrade to their text, unbalanced tags can't
/// crash it, and there is no backtracking to blow up on pathological input.
enum HTMLToMarkdown {
    /// Hard cap on the HTML we will process, independent of the fetch byte cap,
    /// so a pathological document can't pin the CPU in the tokenizer.
    static let maxInputChars = 4_000_000

    /// Max characters ``parseTagAt`` will scan for a single tag's closing `>`.
    /// Bounds each parse to O(1)-ish so the tokenizer stays O(n): without it, an
    /// unterminated `<` scans to EOF, and callers that advance by one character
    /// after a failed parse would re-scan the tail for every `<` — O(n²) on a
    /// body like a long run of `<a` with no `>`. A real tag is far shorter than
    /// this; a longer "tag" is treated as literal text.
    static let maxTagScan = 8192

    struct Result {
        let title: String?
        let markdown: String
    }

    static func extract(_ html: String, baseURL: URL? = nil) -> Result {
        let capped = html.count > maxInputChars ? String(html.prefix(maxInputChars)) : html
        let title = extractTitle(capped)
        // Raw-text + hidden elements first (their bodies are NOT parsed as HTML,
        // so they must be removed by literal open→close scanning before the tag
        // tokenizer runs).
        var cleaned = stripComments(capped)
        for tag in ["script", "style", "noscript", "svg", "head", "template", "iframe"] {
            cleaned = stripElement(tag, in: cleaned)
        }
        // Focus on the main content region when the page marks one; this is what
        // drops most nav / header / footer / sidebar boilerplate.
        let region = mainRegion(cleaned)
        var tokenizer = Tokenizer(Array(region), baseURL: baseURL)
        let md = tokenizer.render()
        return Result(title: title, markdown: normalizeWhitespace(md))
    }

    // MARK: - Region selection

    /// Inner HTML of the first `<article>` or `<main>`, else `<body>`, else the
    /// whole (already-cleaned) document.
    static func mainRegion(_ html: String) -> String {
        for tag in ["article", "main", "body"] {
            if let inner = firstElementInner(tag, in: html), !inner.isEmpty {
                return inner
            }
        }
        return html
    }

    /// Inner HTML of the first `<tag …>…</tag>`, tracking nesting depth so a
    /// nested same-named element doesn't close the region early. Case-insensitive.
    static func firstElementInner(_ tag: String, in html: String) -> String? {
        let chars = Array(html)
        let lower = tag.lowercased()
        guard let open = findTag(lower, closing: false, in: chars, from: 0) else { return nil }
        var depth = 1
        var cursor = open.end
        while cursor < chars.count {
            guard let next = findAnyTag(named: lower, in: chars, from: cursor) else { break }
            if next.closing {
                depth -= 1
                if depth == 0 {
                    return String(chars[open.end..<next.start])
                }
            } else if !next.selfClosing {
                depth += 1
            }
            cursor = next.end
        }
        return nil
    }

    private struct TagHit { let start: Int; let end: Int; let closing: Bool; let selfClosing: Bool }

    /// Find the next `<name …>` (or `</name>`) occurrence at/after `from`.
    private static func findAnyTag(named name: String, in chars: [Character], from: Int) -> TagHit? {
        var i = from
        while i < chars.count {
            if chars[i] == "<" {
                if let hit = parseTagAt(i, in: chars), hit.name == name {
                    return TagHit(start: i, end: hit.end, closing: hit.closing, selfClosing: hit.selfClosing)
                }
                // Skip past this tag (or lone '<').
                if let hit = parseTagAt(i, in: chars) { i = hit.end } else { i += 1 }
            } else {
                i += 1
            }
        }
        return nil
    }

    private static func findTag(_ name: String, closing: Bool, in chars: [Character], from: Int) -> (start: Int, end: Int)? {
        var i = from
        while i < chars.count {
            if chars[i] == "<", let hit = parseTagAt(i, in: chars), hit.name == name, hit.closing == closing {
                return (i, hit.end)
            }
            i += 1
        }
        return nil
    }

    // MARK: - Comment / element stripping

    static func stripComments(_ html: String) -> String {
        var out = ""
        out.reserveCapacity(html.count)
        let chars = Array(html)
        var i = 0
        while i < chars.count {
            if chars[i] == "<", i + 3 < chars.count, chars[i+1] == "!", chars[i+2] == "-", chars[i+3] == "-" {
                // Scan to the closing "-->".
                var j = i + 4
                while j + 2 < chars.count && !(chars[j] == "-" && chars[j+1] == "-" && chars[j+2] == ">") {
                    j += 1
                }
                i = (j + 2 < chars.count) ? j + 3 : chars.count
            } else {
                out.append(chars[i])
                i += 1
            }
        }
        return out
    }

    /// Remove every `<tag …>…</tag>` (raw-text / hidden element) by literal
    /// scanning. Case-insensitive; tolerant of a missing close tag (drops to EOF).
    static func stripElement(_ tag: String, in html: String) -> String {
        let chars = Array(html)
        let lower = tag.lowercased()
        var out = ""
        out.reserveCapacity(chars.count)
        var i = 0
        while i < chars.count {
            if chars[i] == "<", let hit = parseTagAt(i, in: chars), hit.name == lower, !hit.closing {
                if hit.selfClosing {
                    i = hit.end
                    continue
                }
                // Skip until the matching close tag (no nesting for raw-text
                // elements; first close wins).
                if let close = findTag(lower, closing: true, in: chars, from: hit.end) {
                    i = close.end
                } else {
                    i = chars.count
                }
            } else {
                out.append(chars[i])
                i += 1
            }
        }
        return out
    }

    static func extractTitle(_ html: String) -> String? {
        let chars = Array(html)
        guard let open = findTag("title", closing: false, in: chars, from: 0),
              let close = findTag("title", closing: true, in: chars, from: open.end),
              open.end <= close.start else { return nil }
        let raw = String(chars[open.end..<close.start])
        let decoded = decodeEntities(raw).trimmingCharacters(in: .whitespacesAndNewlines)
        return decoded.isEmpty ? nil : collapseSpaces(decoded)
    }

    // MARK: - Tag parsing

    private struct ParsedTag { let name: String; let closing: Bool; let selfClosing: Bool; let attributes: String; let end: Int }

    /// Parse a tag starting at `chars[i] == '<'`. Reads to the terminating `>`,
    /// honouring quotes so a `>` inside an attribute value doesn't end it early.
    /// Returns nil for a lone `<` or a `<!`/`<?` directive.
    private static func parseTagAt(_ i: Int, in chars: [Character]) -> ParsedTag? {
        guard i < chars.count, chars[i] == "<" else { return nil }
        var j = i + 1
        guard j < chars.count else { return nil }
        var closing = false
        if chars[j] == "/" { closing = true; j += 1 }
        // A name must start with an ASCII letter; otherwise it's `<!`, `<?`, or a
        // stray '<' in text.
        guard j < chars.count, chars[j].isASCIILetter else { return nil }
        var name = ""
        while j < chars.count, chars[j].isASCIILetter || chars[j].isNumber || chars[j] == "-" {
            name.append(chars[j]); j += 1
        }
        // Scan to the terminating '>', tracking quotes, with INDEX-ONLY work (no
        // per-char string building) so an unterminated tag costs only the scan,
        // not an allocation. Bounded two ways so a malformed tag can't force an
        // O(n) scan (and thus O(n²) overall when every '<' in a malformed body
        // restarts one): a bare unquoted '<' means this is not a well-formed tag
        // — a real tag never contains one — and a tag that runs past
        // ``maxTagScan`` characters without a closing '>' is likewise treated as
        // literal text. Both return nil, and the caller then advances a single
        // character into the just-scanned run, so total work stays linear. The
        // `attributes` string is materialised only once a real `>` is found.
        let attrStart = j
        var quote: Character? = nil
        let scanLimit = min(chars.count, j + maxTagScan)
        var closeIndex: Int? = nil
        while j < scanLimit {
            let ch = chars[j]
            if let q = quote {
                if ch == q { quote = nil }
            } else if ch == "\"" || ch == "'" {
                quote = ch
            } else if ch == ">" {
                closeIndex = j
                break
            } else if ch == "<" {
                return nil   // bare '<' inside a tag → malformed, treat as text
            }
            j += 1
        }
        guard let close = closeIndex else { return nil }   // no closing '>' (or hit cap)
        let attrs = String(chars[attrStart..<close])
        let selfClosing = attrs.hasSuffix("/")
        return ParsedTag(name: name.lowercased(), closing: closing, selfClosing: selfClosing, attributes: attrs, end: close + 1)
    }

    private static func attribute(_ key: String, in attrs: String) -> String? {
        // Linear scan for `key = "…"` / `key='…'` / `key=token`, case-insensitive
        // key. Good enough for href/src/alt; not a spec HTML attribute parser.
        let lowerAttrs = attrs.lowercased()
        let lowerKey = key.lowercased()
        var searchStart = lowerAttrs.startIndex
        while let r = lowerAttrs.range(of: lowerKey, range: searchStart..<lowerAttrs.endIndex) {
            // Must be at a token boundary (preceded by whitespace or start).
            let before = r.lowerBound == lowerAttrs.startIndex ? " " : lowerAttrs[lowerAttrs.index(before: r.lowerBound)]
            var k = r.upperBound
            // Skip spaces before '='.
            while k < lowerAttrs.endIndex, lowerAttrs[k] == " " { k = lowerAttrs.index(after: k) }
            if (before == " " || before == "\t" || before == "\n" || before == "\r"),
               k < lowerAttrs.endIndex, lowerAttrs[k] == "=" {
                // Map back into the ORIGINAL-case string at the same offset.
                let valOffset = lowerAttrs.distance(from: lowerAttrs.startIndex, to: lowerAttrs.index(after: k))
                let value = attrValue(from: attrs, startingAt: valOffset)
                return value
            }
            searchStart = r.upperBound
        }
        return nil
    }

    private static func attrValue(from attrs: String, startingAt offset: Int) -> String {
        let arr = Array(attrs)
        var i = offset
        while i < arr.count, arr[i] == " " { i += 1 }
        guard i < arr.count else { return "" }
        if arr[i] == "\"" || arr[i] == "'" {
            let q = arr[i]; i += 1
            var out = ""
            while i < arr.count, arr[i] != q { out.append(arr[i]); i += 1 }
            return decodeEntities(out)
        }
        var out = ""
        while i < arr.count, arr[i] != " ", arr[i] != "\t", arr[i] != "\n", arr[i] != "\r", arr[i] != ">" {
            out.append(arr[i]); i += 1
        }
        return decodeEntities(out)
    }

    // MARK: - Tokenizer → Markdown

    private struct Tokenizer {
        let chars: [Character]
        let baseURL: URL?
        var out = ""
        var inPre = false
        var linkHrefStack: [String] = []
        var listDepth = 0

        init(_ chars: [Character], baseURL: URL?) {
            self.chars = chars
            self.baseURL = baseURL
            out.reserveCapacity(chars.count)
        }

        mutating func render() -> String {
            var i = 0
            var textBuf = ""
            func flushText() {
                guard !textBuf.isEmpty else { return }
                let decoded = decodeEntities(textBuf)
                out += inPre ? decoded : collapseSpaces(decoded)
                textBuf = ""
            }
            while i < chars.count {
                if chars[i] == "<", let tag = HTMLToMarkdown.parseTagAt(i, in: chars) {
                    flushText()
                    emit(tag)
                    i = tag.end
                } else {
                    textBuf.append(chars[i])
                    i += 1
                }
            }
            flushText()
            return out
        }

        private mutating func newlineBlock() {
            // Ensure a blank line before the next block, without piling up.
            if inPre { out += "\n"; return }
            while out.hasSuffix("\n\n\n") { out.removeLast() }
            if !out.isEmpty && !out.hasSuffix("\n\n") {
                out += out.hasSuffix("\n") ? "\n" : "\n\n"
            }
        }

        private mutating func emit(_ tag: HTMLToMarkdown.ParsedTag) {
            switch tag.name {
            case "br":
                out += "\n"
            case "hr":
                newlineBlock(); out += "---"; newlineBlock()
            case "p", "div", "section", "article", "main", "header", "footer",
                 "aside", "nav", "table", "figure", "figcaption", "dl", "dd", "dt":
                newlineBlock()
            case "h1", "h2", "h3", "h4", "h5", "h6":
                if tag.closing { newlineBlock() }
                else {
                    newlineBlock()
                    let level = Int(String(tag.name.dropFirst())) ?? 1
                    out += String(repeating: "#", count: level) + " "
                }
            case "ul", "ol":
                if tag.closing { listDepth = max(0, listDepth - 1) }
                else { listDepth += 1 }
                newlineBlock()
            case "li":
                if !tag.closing {
                    if !out.hasSuffix("\n") { out += "\n" }
                    out += String(repeating: "  ", count: max(0, listDepth - 1)) + "- "
                }
            case "blockquote":
                newlineBlock(); if !tag.closing { out += "> " }
            case "tr":
                if tag.closing { out += "\n" }
            case "td", "th":
                if tag.closing { out += " | " }
            case "pre":
                if tag.closing { inPre = false; out += "\n```"; newlineBlock() }
                else { newlineBlock(); out += "```\n"; inPre = true }
            case "code":
                if !inPre { out += "`" }
            case "strong", "b":
                out += "**"
            case "em", "i":
                out += "*"
            case "a":
                if tag.closing {
                    let href = linkHrefStack.popLast() ?? ""
                    if href.isEmpty { /* nothing to link */ }
                    else { out += "](\(href))" }
                } else {
                    let href = (HTMLToMarkdown.attribute("href", in: tag.attributes) ?? "").trimmingCharacters(in: .whitespaces)
                    // Only linkify safe, real destinations; drop javascript:/data:
                    // and fragment/empty hrefs to plain text.
                    if let destination = resolvedDestination(href) {
                        out += "["
                        linkHrefStack.append(destination)
                    }
                    else { linkHrefStack.append("") }
                }
            case "img":
                let alt = (HTMLToMarkdown.attribute("alt", in: tag.attributes) ?? "").trimmingCharacters(in: .whitespaces)
                let src = (HTMLToMarkdown.attribute("src", in: tag.attributes) ?? "").trimmingCharacters(in: .whitespaces)
                if let destination = resolvedDestination(src), !alt.isEmpty {
                    out += "![\(alt)](\(destination))"
                }
                else if !alt.isEmpty { out += "[image: \(alt)]" }
            default:
                break   // unknown/inline tag → contributes only its text
            }
        }

        private func isSafeHref(_ href: String) -> Bool {
            guard !href.isEmpty, !href.hasPrefix("#") else { return false }
            // Browsers strip ASCII whitespace and control characters (including
            // leading spaces and EMBEDDED tab/newline/CR) before resolving a
            // URL's scheme, so ``\njavascript:`` and ``jav\tascript:`` are live.
            // Normalise the same way for the scheme allowlist test — this local
            // is only used to classify the scheme, not to rewrite the link.
            let scanned = String(
                href.unicodeScalars.filter { $0.value > 0x20 && $0.value != 0x7F }
            ).lowercased()
            if scanned.hasPrefix("javascript:")
                || scanned.hasPrefix("data:")
                || scanned.hasPrefix("vbscript:") {
                return false
            }
            return true
        }

        private func resolvedDestination(_ raw: String) -> String? {
            guard isSafeHref(raw) else { return nil }
            guard let baseURL else { return raw }
            guard let resolved = URL(string: raw, relativeTo: baseURL)?.absoluteURL,
                  let scheme = resolved.scheme?.lowercased(),
                  scheme == "http" || scheme == "https" else { return nil }
            return resolved.absoluteString
        }
    }

    // MARK: - Text helpers

    /// Collapse any run of ASCII whitespace to a single space (non-pre text).
    static func collapseSpaces(_ s: String) -> String {
        var out = ""
        out.reserveCapacity(s.count)
        var lastWasSpace = false
        for ch in s {
            if ch == " " || ch == "\t" || ch == "\n" || ch == "\r" {
                if !lastWasSpace { out.append(" "); lastWasSpace = true }
            } else {
                out.append(ch); lastWasSpace = false
            }
        }
        return out
    }

    /// Final pass: trim trailing spaces per line, collapse 3+ blank lines to a
    /// single blank line, and trim leading/trailing blank lines.
    static func normalizeWhitespace(_ s: String) -> String {
        var lines = s.components(separatedBy: "\n").map { line -> String in
            var l = line
            while l.hasSuffix(" ") || l.hasSuffix("\t") { l.removeLast() }
            return l
        }
        // Collapse runs of >1 blank line.
        var collapsed: [String] = []
        var blankRun = 0
        for l in lines {
            if l.isEmpty {
                blankRun += 1
                if blankRun <= 1 { collapsed.append(l) }
            } else {
                blankRun = 0
                collapsed.append(l)
            }
        }
        lines = collapsed
        while lines.first?.isEmpty == true { lines.removeFirst() }
        while lines.last?.isEmpty == true { lines.removeLast() }
        return lines.joined(separator: "\n")
    }

    /// Decode the HTML entities that actually show up in body text. Named set is
    /// the common web subset; numeric `&#nnn;` / `&#xhhh;` are fully supported.
    static func decodeEntities(_ s: String) -> String {
        guard s.contains("&") else { return s }
        var out = ""
        out.reserveCapacity(s.count)
        let chars = Array(s)
        var i = 0
        while i < chars.count {
            guard chars[i] == "&" else { out.append(chars[i]); i += 1; continue }
            // Find the terminating ';' within a small window.
            var j = i + 1
            let limit = min(chars.count, i + 12)
            while j < limit, chars[j] != ";" { j += 1 }
            guard j < chars.count, chars[j] == ";" else { out.append("&"); i += 1; continue }
            let name = String(chars[(i+1)..<j])
            if let decoded = decodeOneEntity(name) {
                out.append(decoded)
                i = j + 1
            } else {
                out.append("&"); i += 1
            }
        }
        return out
    }

    private static func decodeOneEntity(_ name: String) -> Character? {
        if name.hasPrefix("#") {
            let numPart = name.dropFirst()
            let value: UInt32?
            if numPart.hasPrefix("x") || numPart.hasPrefix("X") {
                value = UInt32(numPart.dropFirst(), radix: 16)
            } else {
                value = UInt32(numPart, radix: 10)
            }
            if let v = value, let scalar = Unicode.Scalar(v) { return Character(scalar) }
            return nil
        }
        switch name {
        case "amp": return "&"
        case "lt": return "<"
        case "gt": return ">"
        case "quot": return "\""
        case "apos": return "'"
        case "nbsp": return "\u{00A0}"
        case "copy": return "\u{00A9}"
        case "reg": return "\u{00AE}"
        case "trade": return "\u{2122}"
        case "hellip": return "\u{2026}"
        case "mdash": return "\u{2014}"
        case "ndash": return "\u{2013}"
        case "lsquo": return "\u{2018}"
        case "rsquo": return "\u{2019}"
        case "ldquo": return "\u{201C}"
        case "rdquo": return "\u{201D}"
        case "middot": return "\u{00B7}"
        case "bull": return "\u{2022}"
        case "deg": return "\u{00B0}"
        case "euro": return "\u{20AC}"
        case "pound": return "\u{00A3}"
        case "cent": return "\u{00A2}"
        case "times": return "\u{00D7}"
        case "divide": return "\u{00F7}"
        default: return nil
        }
    }
}

private extension Character {
    var isASCIILetter: Bool {
        return (self >= "a" && self <= "z") || (self >= "A" && self <= "Z")
    }
}
