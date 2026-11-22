import Foundation

/// What an artifact is. The panel and the inline card switch on this.
enum ArtifactKind: String, Sendable, Equatable {
    case html
    case svg
    case mermaid
}

/// The pure half of the HTML artifact feature: no WebKit, no actor.
///
/// Everything that decides whether a fenced block is offered as a live page
/// lives here so it can be tested without spawning a web content process.
enum ArtifactHTMLSource {

    /// A single-file app with inline CSS and JS is far larger than an SVG, so
    /// the SVG preview's 128 KB cap would refuse ordinary output. Above this
    /// the block stays code only.
    static let maximumSourceBytes = 256 * 1024

    /// Is this block a whole HTML document, as far as a cheap check can say?
    ///
    /// A tag decides when it is present. An untagged block has to open with a
    /// doctype or `<html`, because a bare fragment in an unlabelled fence is
    /// far more often prose than a page.
    nonisolated static func looksLikeHTML(code: String, language: String?) -> Bool {
        guard code.utf8.count <= maximumSourceBytes else { return false }
        let trimmed = code.drop(while: \.isWhitespace)
        guard !trimmed.isEmpty else { return false }

        if let language, !language.isEmpty {
            return language.lowercased() == "html"
        }
        return opens(trimmed, with: "<!doctype html") || opens(trimmed, with: "<html")
    }

    /// `prefix` followed by a tag boundary, so `<htmlish>` is not `<html`.
    private nonisolated static func opens(_ text: Substring, with prefix: String) -> Bool {
        guard text.prefix(prefix.count).lowercased() == prefix else { return false }
        guard let next = text.dropFirst(prefix.count).first else { return true }
        return next == ">" || next == "/" || next.isWhitespace
    }

    /// The document's `<title>`, collapsed to one short line, or nil.
    nonisolated static func title(of source: String) -> String? {
        let head = String(source.prefix(16_384))
        guard let match = titlePattern.firstMatch(
            in: head, range: NSRange(head.startIndex..., in: head)
        ), let range = Range(match.range(at: 1), in: head) else { return nil }
        let collapsed = head[range]
            .split(whereSeparator: \.isWhitespace)
            .joined(separator: " ")
        guard !collapsed.isEmpty else { return nil }
        return collapsed.count > 80 ? String(collapsed.prefix(80)) + "…" : collapsed
    }

    /// Does the page ask for a script or stylesheet from off the machine?
    /// Those are blocked, and the reader is told so rather than shown a page
    /// that is quietly missing its code.
    nonisolated static func referencesRemoteResources(_ source: String) -> Bool {
        remotePattern.firstMatch(
            in: source, range: NSRange(source.startIndex..., in: source)
        ) != nil
    }

    // Both patterns are constants known to compile.
    private static let titlePattern = try! NSRegularExpression(
        pattern: #"<title[^>]*>([\s\S]*?)</title>"#, options: [.caseInsensitive]
    )

    private static let remotePattern = try! NSRegularExpression(
        pattern: #"<script\b[^>]*\bsrc\s*=\s*["']?\s*(?:https?:)?//"#
            + #"|<link\b[^>]*\bhref\s*=\s*["']?\s*(?:https?:)?//"#
            + #"|@import\s*(?:url\(\s*)?["']?\s*(?:https?:)?//"#,
        options: [.caseInsensitive]
    )
}

/// One artifact the panel can show.
struct ArtifactItem: Identifiable, Equatable, Sendable {
    let id: String
    let kind: ArtifactKind
    let source: String
    let title: String
    let referencesRemoteResources: Bool

    init(id: String, kind: ArtifactKind, source: String) {
        self.id = id
        self.kind = kind
        self.source = source
        switch kind {
        case .html:
            title = ArtifactHTMLSource.title(of: source)
                ?? String(localized: "HTML artifact")
            referencesRemoteResources = ArtifactHTMLSource.referencesRemoteResources(source)
        case .svg:
            title = String(localized: "SVG image")
            referencesRemoteResources = false
        case .mermaid:
            title = String(localized: "Mermaid diagram")
            referencesRemoteResources = false
        }
    }

    /// The kind a fenced block is offered as, if any. SVG and Mermaid keep
    /// their existing detectors, and win over HTML: a block tagged `html` that
    /// holds an `<svg>` document is a picture.
    static func kind(code: String, language: String?) -> ArtifactKind? {
        if SVGPreview.looksLikeSVG(code: code, language: language) { return .svg }
        if MermaidSource.looksLikeMermaid(code: code, language: language) { return .mermaid }
        if ArtifactHTMLSource.looksLikeHTML(code: code, language: language) { return .html }
        return nil
    }

    var fileExtension: String {
        switch kind {
        case .html: "html"
        case .svg: "svg"
        case .mermaid: "mmd"
        }
    }

    /// Human-readable size, for the card and the list.
    var sizeDescription: String {
        ByteCountFormatter.string(fromByteCount: Int64(source.utf8.count), countStyle: .file)
    }
}
