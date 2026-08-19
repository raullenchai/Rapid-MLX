import Foundation

/// One renderable block of a compiled markdown document.
///
/// The four cases mirror ChatGPT's `MarkdownBlockStack`, which fans out to
/// `MarkdownTextBlock`, `MarkdownTableBlock`, `MarkdownCodeBlock` and
/// `MarkdownImagesBlock`. That split is not arbitrary: text and code go
/// through TextKit 2 because the fade animator addresses glyphs by
/// `NSTextRange`, while tables and images are layout problems better served by
/// SwiftUI.
///
/// Paragraphs, headings, lists, block quotes and horizontal rules all collapse
/// into `.text`. They are rendered as one attributed string with paragraph
/// styles rather than as separate stacked views — which is what ChatGPT's nine
/// list metrics describe (hanging indents), and what keeps a whole list inside
/// a single addressable text range.
enum MarkdownItem: Equatable, Sendable {
    case text(TextBlock)
    case code(CodeBlock)
    case table(TableBlock)
    case images(ImagesBlock)
    case math(MathBlock)

    public struct TextBlock: Equatable, Sendable {
        /// Rendered attributed content. `NSAttributedString` is not `Sendable`,
        /// so we carry the pieces and build it in the renderer.
        public var runs: [InlineRun]
        public var kind: Kind
        /// Nesting depth for lists and quotes.
        public var depth: Int
        /// Ordered-list index, when applicable.
        public var listIndex: Int?

        public enum Kind: Equatable, Sendable {
            case paragraph
            case heading(level: Int)
            case unorderedListItem
            case orderedListItem
            case blockQuote
            case horizontalRule
        }

        public init(runs: [InlineRun], kind: Kind, depth: Int = 0, listIndex: Int? = nil) {
            self.runs = runs
            self.kind = kind
            self.depth = depth
            self.listIndex = listIndex
        }
    }

    public struct CodeBlock: Equatable, Sendable {
        public var code: String
        public var language: String?
        public init(code: String, language: String?) {
            self.code = code
            self.language = language
        }
    }

    public struct TableBlock: Equatable, Sendable {
        public var header: [[InlineRun]]
        public var rows: [[[InlineRun]]]
        public var alignments: [Alignment]

        public enum Alignment: Equatable, Sendable { case leading, center, trailing }

        public init(header: [[InlineRun]], rows: [[[InlineRun]]], alignments: [Alignment]) {
            self.header = header
            self.rows = rows
            self.alignments = alignments
        }
    }

    public struct ImagesBlock: Equatable, Sendable {
        public var urls: [URL]
        public var altTexts: [String]
        public init(urls: [URL], altTexts: [String]) {
            self.urls = urls
            self.altTexts = altTexts
        }
    }

    /// Display math — `$$...$$` on its own.
    ///
    /// A fifth case where ChatGPT has four. Its `Item.Kind` is
    /// text/table/code/images, and formulas ride inside `Item.Text` as entries
    /// in `_viewAttachments: [Range: TextAttachment<NSView>]` — an `NSView`
    /// pinned to a character range.
    ///
    /// That route is strictly better for *inline* math, because an attachment
    /// flows with the sentence and rewraps with it. It is also a larger change
    /// than this one: the fade animator addresses glyphs by `NSTextRange`, and
    /// an attachment inside a faded range needs its own opacity handling.
    ///
    /// So display math becomes its own block now, and inline math stays plain
    /// text until the attachment path exists. Splitting them keeps this change
    /// reviewable; the inline case is tracked separately.
    public struct MathBlock: Equatable, Sendable {
        public var latex: String
        public init(latex: String) {
            self.latex = latex
        }
    }
}

/// A styled span of inline text.
///
/// Deliberately a value type rather than an `NSAttributedString`: the compiler
/// runs off the main actor and its output is cached, and `NSAttributedString`
/// is neither `Sendable` nor cheap to hash.
struct InlineRun: Equatable, Sendable {
    public var text: String
    public var isStrong: Bool
    public var isEmphasis: Bool
    public var isStrikethrough: Bool
    public var isInlineCode: Bool
    public var link: URL?
    /// LaTeX body when this run IS a piece of inline math, `nil` for prose.
    ///
    /// ``text`` still carries the source spelling (`$x$`) so anything that
    /// only wants characters — copy, VoiceOver, search, width estimation —
    /// keeps working untouched and unaware.
    public var math: String?

    public init(
        text: String,
        isStrong: Bool = false,
        isEmphasis: Bool = false,
        isStrikethrough: Bool = false,
        isInlineCode: Bool = false,
        link: URL? = nil,
        math: String? = nil
    ) {
        self.text = text
        self.isStrong = isStrong
        self.isEmphasis = isEmphasis
        self.isStrikethrough = isStrikethrough
        self.isInlineCode = isInlineCode
        self.link = link
        self.math = math
    }
}

/// A compiled document.
struct MarkdownResult: Equatable, Sendable {
    public var items: [MarkdownItem]
    /// Bumped whenever the compiler produces new blocks.
    ///
    /// The height cache keys on this rather than on the message text: during
    /// streaming the text changes on every flush, and re-hashing a growing
    /// 20K-character buffer per flush is exactly the cost the cache exists to
    /// avoid. This changes at compile cadence instead.
    public var revision: Int

    public init(items: [MarkdownItem] = [], revision: Int = 0) {
        self.items = items
        self.revision = revision
    }

    public static let empty = MarkdownResult()
}
