import SwiftUI

/// Accessibility-only model for a GFM table block. MarkdownUI draws tables
/// with `Grid`, which looks correct but flattens to unrelated static text in
/// AppKit's accessibility tree. The chat theme uses this model to provide a
/// native SwiftUI `Table` as the visual block's accessibility representation.
enum MarkdownTableAccessibility {
    struct TableModel: Equatable {
        let headers: [String]
        let rows: [[String]]
    }

    static func parse(_ markdown: String) -> TableModel? {
        let lines = markdown.split(whereSeparator: \Character.isNewline).map(String.init)
        guard lines.count >= 2 else { return nil }
        let headers = cells(in: lines[0])
        let separator = cells(in: lines[1])
        // SwiftUI's macOS 14.0 TableColumnBuilder has no dynamic-column
        // primitive. Cover ordinary chat tables (up to eight columns) and
        // leave wider tables on MarkdownUI's existing text accessibility
        // instead of silently dropping columns from the representation.
        guard !headers.isEmpty,
              headers.count <= 8,
              separator.count == headers.count,
              separator.allSatisfy(isSeparatorCell) else { return nil }

        let rows = lines.dropFirst(2).map { line in
            let parsed = cells(in: line)
            return headers.indices.map { index in
                index < parsed.count ? parsed[index] : ""
            }
        }
        return TableModel(headers: headers, rows: rows)
    }

    private static func cells(in line: String) -> [String] {
        var result: [String] = []
        var current = ""
        var escaped = false
        var codeFenceLength = 0

        for character in line {
            if escaped {
                current.append(character)
                escaped = false
                continue
            }
            if character == "\\" {
                escaped = true
                continue
            }
            if character == "`" {
                codeFenceLength = codeFenceLength == 0 ? 1 : 0
                current.append(character)
                continue
            }
            if character == "|", codeFenceLength == 0 {
                result.append(clean(current))
                current = ""
            } else {
                current.append(character)
            }
        }
        if escaped { current.append("\\") }
        result.append(clean(current))
        if result.first == "" { result.removeFirst() }
        if result.last == "" { result.removeLast() }
        return result
    }

    private static func clean(_ value: String) -> String {
        value.trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "`", with: "")
    }

    private static func isSeparatorCell(_ value: String) -> Bool {
        let trimmed = value.trimmingCharacters(in: CharacterSet(charactersIn: ":"))
        return trimmed.count >= 3 && trimmed.allSatisfy { $0 == "-" }
    }
}

private struct AccessibleMarkdownTableRow: Identifiable {
    let id: Int
    let cells: [String]
}

/// A native `Table` is used only as an accessibility representation; the
/// visible MarkdownUI grid remains unchanged. VoiceOver therefore gains real
/// table navigation and header association without duplicating visual output.
struct AccessibleMarkdownTable: View {
    let model: MarkdownTableAccessibility.TableModel

    private var rows: [AccessibleMarkdownTableRow] {
        model.rows.enumerated().map { .init(id: $0.offset, cells: $0.element) }
    }

    @ViewBuilder var body: some View {
        switch model.headers.count {
        case 1: table1
        case 2: table2
        case 3: table3
        case 4: table4
        case 5: table5
        case 6: table6
        case 7: table7
        default: table8
        }
    }

    private func cell(_ index: Int, in row: AccessibleMarkdownTableRow) -> String {
        index < row.cells.count ? row.cells[index] : ""
    }

    private var table1: some View { Table(rows) {
        TableColumn(model.headers[0]) { row in Text(cell(0, in: row)) }
    }.accessibilityLabel("Markdown table") }
    private var table2: some View { Table(rows) {
        TableColumn(model.headers[0]) { row in Text(cell(0, in: row)) }
        TableColumn(model.headers[1]) { row in Text(cell(1, in: row)) }
    }.accessibilityLabel("Markdown table") }
    private var table3: some View { Table(rows) {
        TableColumn(model.headers[0]) { row in Text(cell(0, in: row)) }
        TableColumn(model.headers[1]) { row in Text(cell(1, in: row)) }
        TableColumn(model.headers[2]) { row in Text(cell(2, in: row)) }
    }.accessibilityLabel("Markdown table") }
    private var table4: some View { Table(rows) {
        TableColumn(model.headers[0]) { row in Text(cell(0, in: row)) }
        TableColumn(model.headers[1]) { row in Text(cell(1, in: row)) }
        TableColumn(model.headers[2]) { row in Text(cell(2, in: row)) }
        TableColumn(model.headers[3]) { row in Text(cell(3, in: row)) }
    }.accessibilityLabel("Markdown table") }
    private var table5: some View { Table(rows) {
        TableColumn(model.headers[0]) { row in Text(cell(0, in: row)) }
        TableColumn(model.headers[1]) { row in Text(cell(1, in: row)) }
        TableColumn(model.headers[2]) { row in Text(cell(2, in: row)) }
        TableColumn(model.headers[3]) { row in Text(cell(3, in: row)) }
        TableColumn(model.headers[4]) { row in Text(cell(4, in: row)) }
    }.accessibilityLabel("Markdown table") }
    private var table6: some View { Table(rows) {
        TableColumn(model.headers[0]) { row in Text(cell(0, in: row)) }
        TableColumn(model.headers[1]) { row in Text(cell(1, in: row)) }
        TableColumn(model.headers[2]) { row in Text(cell(2, in: row)) }
        TableColumn(model.headers[3]) { row in Text(cell(3, in: row)) }
        TableColumn(model.headers[4]) { row in Text(cell(4, in: row)) }
        TableColumn(model.headers[5]) { row in Text(cell(5, in: row)) }
    }.accessibilityLabel("Markdown table") }
    private var table7: some View { Table(rows) {
        TableColumn(model.headers[0]) { row in Text(cell(0, in: row)) }
        TableColumn(model.headers[1]) { row in Text(cell(1, in: row)) }
        TableColumn(model.headers[2]) { row in Text(cell(2, in: row)) }
        TableColumn(model.headers[3]) { row in Text(cell(3, in: row)) }
        TableColumn(model.headers[4]) { row in Text(cell(4, in: row)) }
        TableColumn(model.headers[5]) { row in Text(cell(5, in: row)) }
        TableColumn(model.headers[6]) { row in Text(cell(6, in: row)) }
    }.accessibilityLabel("Markdown table") }
    private var table8: some View { Table(rows) {
        TableColumn(model.headers[0]) { row in Text(cell(0, in: row)) }
        TableColumn(model.headers[1]) { row in Text(cell(1, in: row)) }
        TableColumn(model.headers[2]) { row in Text(cell(2, in: row)) }
        TableColumn(model.headers[3]) { row in Text(cell(3, in: row)) }
        TableColumn(model.headers[4]) { row in Text(cell(4, in: row)) }
        TableColumn(model.headers[5]) { row in Text(cell(5, in: row)) }
        TableColumn(model.headers[6]) { row in Text(cell(6, in: row)) }
        TableColumn(model.headers[7]) { row in Text(cell(7, in: row)) }
    }.accessibilityLabel("Markdown table") }
}
