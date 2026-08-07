import SwiftUI
import MarkdownUI

/// Issue #131: drop-in replacement for ``Markdown(content)`` that
/// renders **display math** (``$$...$$`` and ``\[...\]``) through
/// ``SwiftMath`` while leaving inline math (``$...$`` / ``\(...\)``)
/// as literal source inside the surrounding markdown.
///
/// ## v1 scope: display math only
///
/// Codex r1 P2 (#131) surfaced that splitting markdown around
/// INLINE math corrupts surrounding CommonMark block structure
/// (tables / list items / headings / blockquotes). A row in a
/// table containing ``$x$`` would, with naïve splitting, render
/// as two broken table fragments with a math row jammed in
/// between. STEM responses commonly put formulas in lists and
/// tables.
///
/// Display math (``$$...$$``) doesn't have this problem because
/// models emit it as its own block — separated from prose by
/// blank lines. Splitting markdown around a display-math block
/// preserves the surrounding structure.
///
/// So for v1 we render only display math through ``SwiftMath``.
/// Inline math stays in the markdown body and renders as literal
/// ``$...$`` source — same as before this PR, no regression.
/// A follow-up issue can iterate inline math with a flow layout
/// or a Markdown-block-aware splitter.
///
/// The 2026-08 bracket-delimiter fix (see ``LaTeXSegmenter``) did
/// NOT lift that deferral, and one more constraint was found while
/// re-checking it: routing inline math through MarkdownUI's
/// ``InlineImageProvider`` — the one composition path that keeps a
/// table row or list item intact — caches the rendered image on
/// ``.task(id: inlines)`` in MarkdownUI's ``InlineText``. The cache
/// key is the parsed inline nodes, so it does not re-fire when only
/// the colour scheme or Dynamic-Type size changes, and glyph colour
/// is baked into the image. A light/dark toggle would leave stale
/// black-on-black formulas behind unless the rendering parameters
/// are smuggled into the image URL to perturb the key. That is a
/// design worth doing deliberately, not as a rider on a delimiter
/// fix.
///
/// ## Hot path
///
/// When no display math is detected, this view returns exactly
/// the same view tree as the pre-#131
/// ``Markdown(content).markdownTheme(.rapidChat)`` call — no
/// extra ``VStack``, no extra wrapping cost. That's the 95% of
/// chat replies.
struct LaTeXMarkdownView: View {
    let content: String

    var body: some View {
        // Collapse inline-math segments back into adjacent
        // markdown (v1: codex r1 P2). After this pass the
        // segments are a strict alternation of ``.markdown`` and
        // ``.math(displayMode: true)``.
        let segments = Self.displayMathOnly(LaTeXSegmenter.segment(content))
        Group {
            if segments.count == 1, case .markdown(let only) = segments[0] {
                // Hot path: no display math detected. Identical view
                // tree to the pre-fix shape.
                Markdown(only).markdownTheme(.rapidChat)
            } else {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(Array(segments.enumerated()), id: \.offset) { _, segment in
                        switch segment {
                        case .markdown(let text):
                            Markdown(text).markdownTheme(.rapidChat)
                        case .math(let latex, _):
                            // displayMode is always true here — the
                            // ``displayMathOnly`` filter folded inline
                            // back into markdown.
                            HStack {
                                Spacer(minLength: 0)
                                MathView(latex: latex, displayMode: true)
                                    .padding(.vertical, 4)
                                Spacer(minLength: 0)
                            }
                            .frame(maxWidth: .infinity)
                        }
                    }
                }
            }
        }
        // Issue #304 + #349: chat surfaces render LLM-emitted Markdown.
        // Default SwiftUI OpenURLAction falls through to
        // ``NSWorkspace.shared.open`` which honours ``file://`` —
        // one click can open an arbitrary local file. Restrict to
        // the ``ChatLinkSafety`` allowlist (``http`` / ``https`` /
        // ``mailto``) here so every Markdown link rendered in this
        // view (hot path AND segmented math path) shares the same
        // allow-list. See ``ChatLinkSafety``.
        .chatLinkSafetyFilter()
    }

    /// Codex r1 P2 (#131): fold any inline-math segment back into
    /// the adjacent markdown by re-wrapping with ``$...$``. Keeps
    /// the original prose / table / list shape intact and lets
    /// markdown render the source — same surface as pre-#131 for
    /// inline math. Display math segments pass through unchanged.
    ///
    /// Adjacent markdown segments are concatenated so the result
    /// is a strict alternation of ``.markdown`` / ``.math``.
    ///
    /// ``nonisolated`` so non-MainActor callers (the test suite,
    /// future segmenter callers) can invoke it synchronously. The
    /// implementation is purely value-transforming with no actor
    /// state involved.
    nonisolated static func displayMathOnly(_ segments: [LaTeXSegment]) -> [LaTeXSegment] {
        var out: [LaTeXSegment] = []
        var pendingMarkdown = ""
        func flushMarkdown() {
            if !pendingMarkdown.isEmpty {
                out.append(.markdown(pendingMarkdown))
                pendingMarkdown = ""
            }
        }
        for seg in segments {
            switch seg {
            case .markdown(let text):
                pendingMarkdown += text
            case .math(let latex, let displayMode):
                if displayMode {
                    flushMarkdown()
                    out.append(seg)
                } else {
                    // Re-wrap and inline back into markdown so the
                    // surrounding block (list / table / heading)
                    // stays intact.
                    pendingMarkdown += "$\(latex)$"
                }
            }
        }
        flushMarkdown()
        return out
    }
}
