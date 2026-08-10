import Testing
@testable import Rapid

@Suite("Literal-aware source guard support")
struct SourceGuardSupportTests {
    @Test("comment markers in strings cannot hide executable code")
    func stringCommentMarkersDoNotHideCode() {
        let source = #"let line = "//"; forbiddenLine(); let block = "/*"; forbiddenBlock()"#
        let canonical = SourceGuardSupport.canonicalSource(source, literals: .preserve)
        #expect(canonical.contains("forbiddenLine()"))
        #expect(canonical.contains("forbiddenBlock()"))
        #expect(canonical.contains(#""//""#))
        #expect(canonical.contains(#""/*""#))
    }

    @Test("actual comments and nested block comments are removed")
    func removesComments() {
        let source = "kept() // removed()\n/* outer /* nested */ still outer */ keptToo()"
        #expect(SourceGuardSupport.canonicalSource(source, literals: .preserve) == "kept()keptToo()")
    }

    @Test("literal braces cannot terminate a balanced source block")
    func literalBracesDoNotAffectBalance() {
        let source = #"func f() { let close = "}"; nested { work() }; forbidden() } after()"#
        let canonical = SourceGuardSupport.canonicalSource(source, literals: .erase)
        let block = SourceGuardSupport.balancedBlock(in: canonical, openingBraceAt: canonical.firstIndex(of: "{")!)
        #expect(block?.contains("forbidden()") == true)
        #expect(block?.contains("after()") == false)
    }

    @Test("raw, multiline, and extended-regex contents cannot mimic source")
    func extendedLiteralContentsAreErased() {
        let source = ##"""
        let raw = #"// }"#; rawCode()
        let multiline = """
        /* } */
        """; multilineCode()
        let regex = #/\/\* } \/\//#; regexCode()
        """##
        let canonical = SourceGuardSupport.canonicalSource(source, literals: .erase)

        #expect(canonical.contains("rawCode()"))
        #expect(canonical.contains("multilineCode()"))
        #expect(canonical.contains("regexCode()"))
        #expect(!canonical.contains("/*"))
    }

    @Test("interpolation expressions remain executable source")
    func interpolationCodeIsCanonicalizedRecursively() {
        let source = ##"""
        func f() {
            let text = "\(outer({ closureCall() }, "value \(nestedCall())"))"
            let raw = #"\#(rawCall())"#
            tailCall()
        }
        afterFunction()
        """##
        let erased = SourceGuardSupport.canonicalSource(source, literals: .erase)
        let preserved = SourceGuardSupport.canonicalSource(source, literals: .preserve)
        let opening = erased.firstIndex(of: "{")!
        let function = SourceGuardSupport.balancedBlock(
            in: erased,
            openingBraceAt: opening
        )

        for call in ["outer(", "closureCall()", "nestedCall()", "rawCall()"] {
            #expect(erased.contains(call), "erased literals hid \(call)")
            #expect(preserved.contains(call), "preserved literals hid \(call)")
        }
        #expect(function?.contains("tailCall()") == true)
        #expect(function?.contains("afterFunction()") == false)
    }

    @Test("balanced blocks fail closed on ambiguous ordinary regex syntax")
    func ordinaryRegexRequiresAParser() {
        let source = #"func f() { let pattern = /https?:\/\/[}]/; hiddenAfterBrace() }"#
        let canonical = SourceGuardSupport.canonicalSource(source, literals: .erase)
        let opening = canonical.firstIndex(of: "{")!

        #expect(canonical.contains("hiddenAfterBrace()"))
        #expect(
            SourceGuardSupport.balancedBlock(
                in: canonical,
                openingBraceAt: opening
            ) == nil
        )
    }

    @Test("preserved ordinary regex keeps exactly one opening slash")
    func preservedOrdinaryRegexDelimiterIsNotDuplicated() {
        let source = "let pattern = /foo/; followingCall()"
        let canonical = SourceGuardSupport.canonicalSource(source, literals: .preserve)

        #expect(canonical == "letpattern=/foo/;followingCall()")
    }

    @Test("bare-regex parentheses cannot end a string interpolation")
    func regexParenthesesInsideInterpolationAreSkipped() {
        let source = ##"""
        let text = "\(use(/[(]/, /[)]/))"; followingCall()
        outsideCall()
        """##
        let canonical = SourceGuardSupport.canonicalSource(source, literals: .erase)

        #expect(canonical.contains("use("))
        #expect(canonical.contains("followingCall()"))
        #expect(canonical.contains("outsideCall()"))
    }
}
