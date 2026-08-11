import Testing

@testable import Rapid

struct MarkdownTableAccessibilityTests {
    @Test("GFM table becomes headers and rectangular data rows")
    func parsesTable() {
        let table = MarkdownTableAccessibility.parse("""
        | model | size | speed |
        | --- | ---: | :---: |
        | qwen3.5-9b | 5.2 GB | 74 tok/s |
        | llama-3.1-8b | 4.5 GB | 68 tok/s |
        """)
        #expect(table?.headers == ["model", "size", "speed"])
        #expect(table?.rows == [
            ["qwen3.5-9b", "5.2 GB", "74 tok/s"],
            ["llama-3.1-8b", "4.5 GB", "68 tok/s"],
        ])
    }

    @Test("Escaped and inline-code pipes stay inside their cells")
    func preservesPipes() {
        let table = MarkdownTableAccessibility.parse("""
        | name | expression |
        | --- | --- |
        | escaped | a\\|b |
        | code | `x|y` |
        """)
        #expect(table?.rows == [["escaped", "a|b"], ["code", "x|y"]])
    }

    @Test("Pipes inside multi-backtick code spans stay inside their cells")
    func preservesPipesInMultiBacktickCode() {
        let table = MarkdownTableAccessibility.parse("""
        | name | expression |
        | --- | --- |
        | code | ``x`|y`` |
        """)
        #expect(table?.rows == [["code", "x|y"]])
    }

    @Test("Backslashes outside CommonMark escapes remain in accessible text")
    func preservesOrdinaryBackslashes() {
        let table = MarkdownTableAccessibility.parse("""
        | platform | path |
        | --- | --- |
        | Windows | C:\\Models |
        """)
        #expect(table?.rows == [["Windows", "C:\\Models"]])
    }

    @Test("Ordinary prose is not mistaken for a table")
    func rejectsProse() {
        #expect(MarkdownTableAccessibility.parse("hello\nworld") == nil)
    }

    @Test("Tables wider than the faithful macOS 14 representation keep the visual fallback")
    func rejectsMoreThanEightColumns() {
        #expect(MarkdownTableAccessibility.parse("""
        | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
        |---|---|---|---|---|---|---|---|---|
        | a | b | c | d | e | f | g | h | i |
        """) == nil)
    }
}
