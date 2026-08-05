import Foundation
import Testing
@testable import Rapid

/// Pins for the "Select text…" stopgap (2026-07 dogfood: selection in
/// the transcript dies at every MarkdownUI block edge, so the user
/// could only ever grab one line / list item at a time).
///
/// The sheet is a single selectable ``Text`` fed by
/// ``SelectTextSheet.selectableText(for:)``. Two contracts matter:
///   * the shown text rides the SAME display sanitiser as the
///     transcript — a future dev passing raw wire content through
///     the sheet would reopen the F-10-4 bidi surface on a copy
///     path, silently;
///   * the sheet body stays ONE ``Text`` with selection enabled —
///     splitting it back into per-block views would quietly
///     reintroduce the exact bug the sheet exists to work around.
@MainActor
@Suite("SelectTextSheet — cross-block selection stopgap")
struct SelectTextSheetTests {

    // MARK: - Content contract

    @Test("Plain prose passes through verbatim (markdown markers included)")
    func plainProseVerbatim() {
        let body = "First paragraph.\n\nSecond paragraph with **bold** and a\n- list item"
        #expect(SelectTextSheet.selectableText(for: body) == body)
    }

    @Test("Sheet text rides the display sanitiser — bidi override never reaches the sheet")
    func sanitiserApplied() {
        let hostile = "safe\u{202E}gnp.exe"
        let shown = SelectTextSheet.selectableText(for: hostile)
        #expect(!shown.unicodeScalars.contains { $0.value == 0x202E })
        // Byte-for-byte the same pipeline the transcript renders and
        // the Copy button writes.
        #expect(shown == ChatTextSanitizer.sanitizeForDisplay(hostile))
    }

    // MARK: - Source-shape guards

    private static var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // RapidTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // repo root
    }

    private func loadSource(_ relativePath: String) throws -> String {
        try String(
            contentsOf: Self.repoRoot.appendingPathComponent(relativePath),
            encoding: .utf8
        )
    }

    @Test("Sheet body keeps its single selectable Text")
    func sheetBodySingleSelectableText() throws {
        let source = try loadSource("Sources/Rapid/UI/SelectTextSheet.swift")
        #expect(
            source.contains(".textSelection(.enabled)"),
            "SelectTextSheet's body Text must stay selectable — it is the sheet's entire reason to exist."
        )
    }

}
