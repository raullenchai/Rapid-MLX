import Foundation

/// Which fenced code blocks are Mermaid diagrams.
///
/// The pure half of the feature: no WebKit, no I/O, no actor. Everything that
/// decides whether a reader is offered a diagram lives here so it can be
/// tested exhaustively without spawning a web content process.
///
/// ## Why this is stricter than ``SVGPreview/looksLikeSVG(code:language:)``
///
/// The SVG detector ignores the language tag on the grounds that
/// `NSImage(data:)` is a cheap, synchronous, side-effect-free authority — if
/// the guess is wrong it costs one failed parse. Mermaid's authority is 3.4 MB
/// of JavaScript in another process, so a false positive costs a render and a
/// cache entry. The tag therefore decides when it is present, and only an
/// untagged block falls through to reading the source.
enum MermaidSource {

    /// Mermaid source is prose-sized. The 512 KB the SVG path allows is
    /// meaningless here, and half a megabyte of "diagram" is an attack rather
    /// than a diagram.
    static let maximumSourceBytes = 64 * 1024

    /// Tags a model actually writes for a diagram.
    static let languageTags: Set<String> = ["mermaid", "mmd"]

    /// The word a Mermaid document opens with.
    ///
    /// Pinned by a test rather than derived, so gaining a diagram type is a
    /// deliberate edit. Taken from Mermaid v11's own diagram registry; the
    /// `-beta` names are theirs, not a placeholder of ours.
    static let diagramKeywords: Set<String> = [
        "graph", "flowchart", "flowchart-elk",
        "sequenceDiagram", "classDiagram", "classDiagram-v2",
        "stateDiagram", "stateDiagram-v2", "erDiagram",
        "journey", "gantt", "pie", "mindmap", "timeline", "gitGraph",
        "quadrantChart", "requirementDiagram", "kanban",
        "C4Context", "C4Container", "C4Component", "C4Dynamic", "C4Deployment",
        "sankey-beta", "xychart-beta", "block-beta", "packet-beta",
        "architecture-beta", "radar-beta", "treemap-beta",
    ]

    /// Is this block worth rendering?
    nonisolated static func looksLikeMermaid(code: String, language: String?) -> Bool {
        guard code.utf8.count <= maximumSourceBytes else { return false }

        if let language, !language.isEmpty {
            return languageTags.contains(language.lowercased())
        }
        guard let opening = openingKeyword(of: code) else { return false }
        guard diagramKeywords.contains(where: { $0.lowercased() == opening.opening.lowercased() })
        else { return false }
        // `graph` and `flowchart` are ordinary words, and `graph = nx.Graph()`
        // opens with one followed by a space exactly as `graph TD` does. Their
        // grammar says what may come next, so ask it: a direction, or nothing.
        if ["graph", "flowchart", "flowchart-elk"].contains(opening.opening.lowercased()) {
            guard let next = opening.rest.split(whereSeparator: \.isWhitespace).first else {
                return true
            }
            return directions.contains(next.uppercased())
        }
        return true
    }

    /// The directions Mermaid's flowchart grammar accepts after `graph`.
    static let directions: Set<String> = ["TD", "TB", "BT", "RL", "LR"]

    /// The diagram keyword an untagged block opens with, skipping the things
    /// Mermaid allows in front of one: a `---` frontmatter block, `%%`
    /// comments, and `%%{init: …}%%` directives.
    ///
    /// The keyword must be followed by the end of the line or a separator —
    /// that is what tells `graph TD` from a Python block opening
    /// `graph = nx.Graph()`.
    nonisolated static func openingKeyword(of code: String) -> (opening: String, rest: String)? {
        // CRLF is one extended grapheme cluster in Swift, so splitting on the
        // literal `\n` Character misses it. Split by newline semantics.
        var lines = code.split(
            omittingEmptySubsequences: false, whereSeparator: \.isNewline
        )[...]

        // Frontmatter, if the very first line opens one.
        if lines.first?.trimmingCharacters(in: .whitespacesAndNewlines) == "---" {
            lines = lines.dropFirst()
            guard let close = lines.firstIndex(where: {
                $0.trimmingCharacters(in: .whitespacesAndNewlines) == "---"
            }) else { return nil }
            lines = lines[lines.index(after: close)...]
        }

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty || trimmed.hasPrefix("%%") { continue }
            let word = trimmed.prefix { !$0.isWhitespace && $0 != ":" }
            guard !word.isEmpty else { return nil }
            // Something has to follow the keyword or end the line; a bare
            // `graph` with a `(` after it is code.
            let rest = trimmed.dropFirst(word.count)
            guard rest.isEmpty || rest.first!.isWhitespace || rest.first! == ":" else {
                return nil
            }
            return (String(word), String(rest))
        }
        return nil
    }
}
