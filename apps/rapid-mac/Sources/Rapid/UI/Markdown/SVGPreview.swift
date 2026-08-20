import AppKit

/// Turns the SVG a model wrote into something a reader can look at.
///
/// ## Why there is no library here
///
/// AppKit renders SVG itself: `NSImage(data:)` accepts an SVG document and
/// returns an image backed by a vector representation, which re-rasterises at
/// whatever size it is drawn. Measured on this codebase's deployment target
/// family — a triangle's hypotenuse has a one-pixel antialiased edge at 64pt
/// and still one pixel at 512pt, so it is genuinely vector and not an upscaled
/// bitmap. Gradients, `transform`, `clipPath`, `<text>` and dashes all render.
///
/// The alternative considered was exyte/SVGView, which is what ChatGPT ships
/// for its *icons* (its code-block preview is a sandboxed WebView, a much
/// larger machine). That library is MIT and would work, but its last
/// functional commit was in 2023 — everything since is README edits — and
/// vendoring three thousand lines of unmaintained parser to do what one
/// framework call already does is a poor trade.
///
/// ## What this deliberately does not do
///
/// It does not execute anything. There is no script engine behind
/// `NSImage(data:)`, and a remote `<image xlink:href="http://…">` is **not**
/// fetched — measured against a local HTTP server that received no request
/// while the document drew. So a model-authored SVG cannot phone home, and
/// this needs no sandbox, no permission prompt and no network policy. That is
/// the whole reason this feature is small enough to be worth having.
enum SVGPreview {

    /// Widest the preview will draw, so a `viewBox` of 4000 does not produce a
    /// wall of image.
    static let maximumHeight: CGFloat = 420

    /// Refuse to even attempt anything larger. A model can emit a megabyte of
    /// path data, and the parse is synchronous on the main thread.
    static let maximumSourceBytes = 128 * 1024

    /// Is this code block worth attempting a preview for?
    ///
    /// A cheap pre-filter, not a decision: ``image(from:)`` is the authority.
    /// Besides rejecting ordinary code, it waits for a likely closing root so
    /// an incomplete streamed document is not reparsed on every token flush.
    ///
    /// The language tag is deliberately ignored. Models label SVG as `svg`,
    /// `xml`, `html`, or nothing at all, and an allowlist was tried and
    /// removed: the only blocks it excluded were mis-tagged documents that
    /// render perfectly well, so it bought nothing that `NSImage` returning
    /// nil does not already buy, and cost a preview on every block a model
    /// labelled wrong.
    nonisolated static func looksLikeSVG(code: String, language: String?) -> Bool {
        guard code.utf8.count <= maximumSourceBytes else { return false }
        let head = code.prefix(2_048)
        guard head.range(of: "<svg", options: .caseInsensitive) != nil else {
            return false
        }
        // Prose or source that merely mentions `<svg` is not a document. The
        // document has to *open* with a tag — an XML prolog, comment or
        // doctype all satisfy that, a `let markup = "…"` does not.
        guard code.unicodeScalars.first(where: { !$0.properties.isWhitespace })?.value == 60 else {
            return false
        }
        // The lexer is linear, so do not run it for every token in a growing
        // stream. A normal document is only eligible when its final non-space
        // bytes are `</svg>`; the root-only form is checked from the front.
        guard hasClosingRootSuffix(code) || isSelfClosingRoot(code) else { return false }
        return hasCompleteSVGRoot(in: code)
    }

    private nonisolated static func hasClosingRootSuffix(_ code: String) -> Bool {
        var trimmed = code[...]
        while trimmed.last?.isWhitespace == true { trimmed.removeLast() }
        // Only do the full backwards check at a plausible document boundary,
        // not after every streamed token. Comments may be arbitrarily long,
        // so the eventual completed boundary must not use a fixed look-back.
        while !endsWithClosingSVGTag(trimmed) {
            if trimmed.hasSuffix("-->"),
               let start = trimmed.range(of: "<!--", options: .backwards) {
                trimmed = trimmed[..<start.lowerBound]
            } else if trimmed.hasSuffix("?>"),
                      let start = trimmed.range(of: "<?", options: .backwards) {
                trimmed = trimmed[..<start.lowerBound]
            } else {
                return false
            }
            while trimmed.last?.isWhitespace == true { trimmed.removeLast() }
        }
        return true
    }

    private nonisolated static func endsWithClosingSVGTag<C: BidirectionalCollection>(
        _ source: C
    ) -> Bool where C.Element == Character {
        var reversed = source.reversed()[...]
        guard reversed.first == ">" else { return false }
        reversed = reversed.dropFirst().drop(while: { $0.isWhitespace })
        return String(reversed.prefix(5).reversed()).lowercased() == "</svg"
    }

    private nonisolated static func containsOnlyXMLTrivia(_ suffix: Substring) -> Bool {
        var rest = suffix[...]
        while true {
            rest = rest.drop(while: { $0.isWhitespace })
            if rest.isEmpty { return true }
            if rest.hasPrefix("<!--"), let end = rest.range(of: "-->") {
                rest = rest[end.upperBound...]
            } else if rest.hasPrefix("<?"), let end = rest.range(of: "?>") {
                rest = rest[end.upperBound...]
            } else {
                return false
            }
        }
    }

    private nonisolated static func isSelfClosingRoot(_ code: String) -> Bool {
        let head = code.prefix(2_048)
        var searchStart = head.startIndex
        while searchStart < head.endIndex,
              let opening = head.range(
                  of: "<svg", options: .caseInsensitive,
                  range: searchStart..<head.endIndex
              ) {
            let candidate = code[opening.lowerBound...]
            var quote: Character?
            var previousNonSpace: Character?
            for index in candidate.indices {
                let character = candidate[index]
                if let currentQuote = quote {
                    if character == currentQuote { quote = nil }
                } else if character == "\"" || character == "'" {
                    quote = character
                } else if character == ">" {
                    let rest = candidate[candidate.index(after: index)...]
                    if previousNonSpace == "/", containsOnlyXMLTrivia(rest) { return true }
                    break
                } else if !character.isWhitespace {
                    previousNonSpace = character
                }
            }
            searchStart = opening.upperBound
        }
        return false
    }

    private nonisolated static func lowercasedASCII(_ byte: UInt8) -> UInt8 {
        (65...90).contains(byte) ? byte + 32 : byte
    }

    private nonisolated static func isASCIIWhitespace(_ byte: UInt8) -> Bool {
        byte == 9 || byte == 10 || byte == 13 || byte == 32
    }

    private nonisolated static func isNameBoundary(_ byte: UInt8) -> Bool {
        byte == 47 || byte == 62 || isASCIIWhitespace(byte)
    }

    /// Avoid the expensive parse until the streamed root is whole. This tiny
    /// lexer ignores comments, CDATA, declarations and quoted attributes, so
    /// text such as `<!-- </svg> -->` cannot trigger a parse on every flush.
    /// ``NSImage`` remains the authority on whether the completed source is
    /// valid SVG.
    private nonisolated static func hasCompleteSVGRoot(in code: String) -> Bool {
        let bytes = Array(code.utf8)
        var index = 0
        var svgDepth = 0

        func starts(with token: [UInt8], at offset: Int) -> Bool {
            offset + token.count <= bytes.count
                && bytes[offset..<(offset + token.count)].elementsEqual(token)
        }

        while index < bytes.count {
            guard bytes[index] == 60 else { index += 1; continue } // `<`

            if starts(with: Array("<!--".utf8), at: index) {
                guard let end = bytes[(index + 4)...].firstRange(of: Array("-->".utf8)) else {
                    return false
                }
                index = end.upperBound
                continue
            }
            if starts(with: Array("<![CDATA[".utf8), at: index) {
                guard let end = bytes[(index + 9)...].firstRange(of: Array("]]>".utf8)) else {
                    return false
                }
                index = end.upperBound
                continue
            }

            // Find the tag end without treating `>` inside a quoted attribute
            // or declaration as markup.
            var end = index + 1
            var quote: UInt8?
            var bracketDepth = 0
            while end < bytes.count {
                let byte = bytes[end]
                if let currentQuote = quote {
                    if byte == currentQuote { quote = nil }
                } else if byte == 34 || byte == 39 {
                    quote = byte
                } else if byte == 91 {
                    bracketDepth += 1
                } else if byte == 93, bracketDepth > 0 {
                    bracketDepth -= 1
                } else if byte == 62, bracketDepth == 0 {
                    break
                }
                end += 1
            }
            guard end < bytes.count else { return false }

            var nameStart = index + 1
            let isClosing = nameStart < end && bytes[nameStart] == 47
            if isClosing { nameStart += 1 }
            while nameStart < end, [9, 10, 13, 32].contains(bytes[nameStart]) {
                nameStart += 1
            }
            let isSVG = nameStart + 3 <= end
                && lowercasedASCII(bytes[nameStart]) == 115
                && lowercasedASCII(bytes[nameStart + 1]) == 118
                && lowercasedASCII(bytes[nameStart + 2]) == 103
                && (nameStart + 3 == end || isNameBoundary(bytes[nameStart + 3]))

            if isSVG {
                if isClosing {
                    guard svgDepth > 0 else { index = end + 1; continue }
                    svgDepth -= 1
                    if svgDepth == 0 { return true }
                } else {
                    var tail = end
                    while tail > nameStart, [9, 10, 13, 32].contains(bytes[tail - 1]) {
                        tail -= 1
                    }
                    if tail > nameStart, bytes[tail - 1] == 47 {
                        if svgDepth == 0 { return true }
                        index = end + 1
                        continue
                    }
                    svgDepth += 1
                }
            }
            index = end + 1
        }
        return false
    }

    /// The rendered document, or nil when it will not parse.
    ///
    /// Nil is the normal outcome mid-stream: half an `<svg` is not a document,
    /// and `NSImage` returns nil for it rather than drawing something partial.
    /// The caller shows the code and nothing else until it completes.
    @MainActor
    static func image(from code: String) -> NSImage? {
        guard code.utf8.count <= maximumSourceBytes else { return nil }
        let data = Data(code.utf8)
        guard SafeSVGValidator.accepts(data), let image = NSImage(data: data) else { return nil }
        let size = image.size
        guard size.width > 0, size.height > 0,
              size.width.isFinite, size.height.isFinite else { return nil }
        return image
    }

    /// A conservative boundary in front of AppKit's decoder. Vector
    /// primitives remain available, but resource-bearing elements, external
    /// entities, filters and excessive structure are refused before the
    /// synchronous framework parser sees model-authored input.
    private final class SafeSVGValidator: NSObject, XMLParserDelegate {
        private static let maximumElements = 2_048
        private static let maximumAttributes = 8_192
        private static let maximumDepth = 64
        private static let forbiddenElements: Set<String> = [
            "audio", "filter", "foreignobject", "iframe", "image", "script", "style", "video",
        ]

        private var elementCount = 0
        private var attributeCount = 0
        private var depth = 0
        private var rejected = false

        static func accepts(_ data: Data) -> Bool {
            // Reject declarations before XMLParser can expand internal
            // entities; delegate limits run too late for a "billion laughs"
            // payload.
            let source = String(decoding: data, as: UTF8.self)
            guard source.range(of: "<!doctype", options: .caseInsensitive) == nil,
                  source.range(of: "<!entity", options: .caseInsensitive) == nil else {
                return false
            }
            let validator = SafeSVGValidator()
            let parser = XMLParser(data: data)
            parser.delegate = validator
            parser.shouldResolveExternalEntities = false
            return parser.parse() && !validator.rejected
        }

        func parser(
            _ parser: XMLParser, didStartElement elementName: String,
            namespaceURI: String?, qualifiedName qName: String?,
            attributes attributeDict: [String: String]
        ) {
            elementCount += 1
            attributeCount += attributeDict.count
            depth += 1
            let localName = elementName.split(separator: ":").last?.lowercased() ?? ""
            if elementCount > Self.maximumElements
                || attributeCount > Self.maximumAttributes
                || depth > Self.maximumDepth
                || Self.forbiddenElements.contains(localName) {
                rejected = true
                parser.abortParsing()
                return
            }

            for (name, value) in attributeDict {
                let key = name.lowercased()
                let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
                let lower = trimmed.lowercased()
                // Namespace identifiers name vocabularies; they are not
                // dereferenced resources.
                if key == "xmlns" || key.hasPrefix("xmlns:") { continue }
                // CSS escapes can disguise both `url` and its scheme. Inline
                // CSS and escaped presentation values are outside the safe
                // subset rather than being normalized by a second CSS parser.
                if key == "style" || value.contains("\\") { rejected = true }
                if (key == "href" || key.hasSuffix(":href"))
                    && !trimmed.isEmpty && !trimmed.hasPrefix("#") {
                    rejected = true
                }
                if ["http:", "https:", "file:", "data:", "ftp:", "//"]
                    .contains(where: lower.contains) {
                    rejected = true
                }
                if !Self.containsOnlyLocalCSSURLs(lower) { rejected = true }
                if key == "d", value.utf8.count > 64 * 1024 { rejected = true }
            }
            if rejected { parser.abortParsing() }
        }

        func parser(
            _ parser: XMLParser, didEndElement elementName: String,
            namespaceURI: String?, qualifiedName qName: String?
        ) {
            depth -= 1
        }

        private static func containsOnlyLocalCSSURLs(_ value: String) -> Bool {
            var rest = value[...]
            while let marker = rest.range(of: "url(") {
                let targetStart = marker.upperBound
                guard let close = rest[targetStart...].firstIndex(of: ")") else { return false }
                var target = rest[targetStart..<close]
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if target.count >= 2,
                   let first = target.first, first == "'" || first == "\"",
                   target.last == first {
                    target.removeFirst()
                    target.removeLast()
                }
                guard target.hasPrefix("#") else { return false }
                rest = rest[rest.index(after: close)...]
            }
            return true
        }

        func parser(
            _ parser: XMLParser, foundExternalEntityDeclarationWithName name: String,
            publicID: String?, systemID: String?
        ) {
            rejected = true
            parser.abortParsing()
        }

        func parser(
            _ parser: XMLParser, foundProcessingInstructionWithTarget target: String,
            data: String?
        ) {
            // The XML declaration is handled by the parser itself. Any PI
            // delivered here (notably `xml-stylesheet`) is an extension point
            // that can carry a resource URL, so the preview refuses it.
            rejected = true
            parser.abortParsing()
        }
    }

    /// The size to draw `image` at inside `width`, preserving its aspect and
    /// never exceeding ``maximumHeight``.
    ///
    /// Vector art has no natural pixel size, so it is scaled to the column
    /// rather than centred at its nominal dimensions — except that it is never
    /// scaled *up* past its own size, because a 24-point icon blown across a
    /// 700-point column looks like a mistake rather than a preview.
    nonisolated static func drawSize(for imageSize: CGSize, inWidth width: CGFloat) -> CGSize {
        guard imageSize.width > 0, imageSize.height > 0, width > 0 else { return .zero }
        let scale = min(
            min(width / imageSize.width, maximumHeight / imageSize.height),
            1
        )
        return CGSize(
            width: imageSize.width * scale,
            height: imageSize.height * scale
        )
    }
}
