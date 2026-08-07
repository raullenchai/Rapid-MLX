import Foundation
import Testing

@testable import Rapid

/// Static guard against `OpenSettingsAction` coming back.
///
/// Issue #1578 asks for one explicitly: "either a lint/grep check that
/// `openSettings()` never appears outside a doc comment, or deleting the
/// `@Environment(\.openSettings)` properties entirely so the call doesn't
/// compile." Both are done — the properties are gone, and this pins that they
/// stay gone. Deletion alone is not a guard: nothing stops the next view from
/// declaring the property again, and the failure mode is silent (the button
/// renders, responds to the pointer, and does nothing).
///
/// Doc comments are exempt on purpose. Several call sites now carry a comment
/// naming `@Environment(\.openSettings)` as the thing NOT to use, and that
/// explanation is the reason the trap stays shut.
@Suite("openSettings must not come back")
struct OpenSettingsActionAbsenceTests {
    private static var sourcesRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // RapidTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // rapid-mac
            .appendingPathComponent("Sources/Rapid", isDirectory: true)
    }

    /// `AppDelegate.openSettingsWindow` is the WORKING bridge
    /// (`openWindow(id: "settings")`) and shares a prefix with the broken API,
    /// so match the two real spellings rather than the bare word.
    /// `\s*` before the paren so `openSettings ()` cannot slip through.
    private static func forbiddenPattern() throws -> NSRegularExpression {
        try NSRegularExpression(pattern: #"\\\.openSettings|openSettings\s*\("#)
    }

    /// Blank out comments **and string literals**, so a doc comment naming the
    /// API is not an offence while `/* … */ openSettings()` on one line still
    /// is. Blanked characters become spaces rather than being deleted, so the
    /// line and column numbers this suite reports stay honest.
    ///
    /// String literals are tracked for a reason that is not cosmetic. Without
    /// it, `let url = "https://example.com"` opens a line comment at the `//`
    /// inside the string, and every subsequent character on that line — up to
    /// and including a real `openSettings()` — gets blanked. That is a false
    /// NEGATIVE: the guard would go quiet exactly when it should fire. Blanking
    /// literal contents fixes both directions at once, since a string that
    /// merely mentions `openSettings(` then cannot raise a false alarm either.
    ///
    /// Handles nested block comments, `//` to end of line, escapes, multiline
    /// `"""` literals, and raw `#"…"#` literals of any pound count. It is still
    /// not a full Swift lexer: code inside a string interpolation is blanked
    /// along with the literal, so `"\(openSettings())"` would slip past. That
    /// is the one acknowledged blind spot, and it is not a shape anyone writes.
    static func strippingCommentsAndStringLiterals(_ source: String) -> String {
        enum Mode {
            case code
            case lineComment
            case blockComment(depth: Int)
            case string(pounds: Int, multiline: Bool)
        }

        var out = ""
        out.reserveCapacity(source.count)
        var mode = Mode.code
        var index = source.startIndex

        /// Does `literal` start at `position`, without running off the end?
        func matches(_ literal: String, at position: String.Index) -> Bool {
            var cursor = position
            for character in literal {
                guard cursor < source.endIndex, source[cursor] == character else { return false }
                cursor = source.index(after: cursor)
            }
            return true
        }

        /// `count` characters on, clamped to `endIndex` so no step can trap.
        func stepping(_ position: String.Index, by count: Int) -> String.Index {
            source.index(position, offsetBy: count, limitedBy: source.endIndex) ?? source.endIndex
        }

        /// Blank `count` characters, preserving any newline among them so the
        /// output stays line-for-line with the input.
        func blank(_ count: Int) {
            var cursor = index
            for _ in 0..<count {
                guard cursor < source.endIndex else { break }
                out.append(source[cursor] == "\n" ? "\n" : " ")
                cursor = source.index(after: cursor)
            }
        }

        while index < source.endIndex {
            let character = source[index]

            switch mode {
            case .lineComment:
                if character == "\n" { mode = .code }
                blank(1)
                index = source.index(after: index)

            case .blockComment(let depth):
                // Swift block comments nest, so `/* /* */ */` must not end early.
                if matches("*/", at: index) {
                    mode = depth == 1 ? .code : .blockComment(depth: depth - 1)
                    blank(2)
                    index = stepping(index, by: 2)
                } else if matches("/*", at: index) {
                    mode = .blockComment(depth: depth + 1)
                    blank(2)
                    index = stepping(index, by: 2)
                } else {
                    blank(1)
                    index = source.index(after: index)
                }

            case .string(let pounds, let multiline):
                let closing = (multiline ? "\"\"\"" : "\"") + String(repeating: "#", count: pounds)
                // In a raw literal the escape is `\` followed by that many `#`.
                let escape = "\\" + String(repeating: "#", count: pounds)
                if matches(closing, at: index) {
                    mode = .code
                    blank(closing.count)
                    index = stepping(index, by: closing.count)
                } else if matches(escape, at: index) {
                    // Consume the escape AND the character it escapes, so a
                    // `\"` cannot be mistaken for the terminator.
                    blank(escape.count + 1)
                    index = stepping(index, by: escape.count + 1)
                } else {
                    blank(1)
                    index = source.index(after: index)
                }

            case .code:
                // Raw string opener: one or more `#` immediately before a quote.
                var pounds = 0
                var probe = index
                while probe < source.endIndex, source[probe] == "#" {
                    pounds += 1
                    probe = source.index(after: probe)
                }
                if pounds > 0, probe < source.endIndex, source[probe] == "\"" {
                    let multiline = matches("\"\"\"", at: probe)
                    let opener = pounds + (multiline ? 3 : 1)
                    mode = .string(pounds: pounds, multiline: multiline)
                    blank(opener)
                    index = stepping(index, by: opener)
                } else if character == "\"" {
                    let multiline = matches("\"\"\"", at: index)
                    let opener = multiline ? 3 : 1
                    mode = .string(pounds: 0, multiline: multiline)
                    blank(opener)
                    index = stepping(index, by: opener)
                } else if matches("//", at: index) {
                    mode = .lineComment
                    blank(2)
                    index = stepping(index, by: 2)
                } else if matches("/*", at: index) {
                    mode = .blockComment(depth: 1)
                    blank(2)
                    index = stepping(index, by: 2)
                } else {
                    out.append(character)
                    index = source.index(after: index)
                }
            }
        }
        return out
    }

    @Test("no source file calls openSettings() or holds the environment action")
    func noOpenSettingsInSources() throws {
        let fm = FileManager.default
        let pattern = try Self.forbiddenPattern()
        var offenders: [String] = []

        let walker = try #require(fm.enumerator(at: Self.sourcesRoot, includingPropertiesForKeys: nil))
        for case let url as URL in walker where url.pathExtension == "swift" {
            let stripped = Self.strippingCommentsAndStringLiterals(try String(contentsOf: url, encoding: .utf8))
            for (index, line) in stripped.components(separatedBy: .newlines).enumerated() {
                let range = NSRange(line.startIndex..<line.endIndex, in: line)
                guard pattern.firstMatch(in: line, range: range) != nil else { continue }
                offenders.append(
                    "\(url.lastPathComponent):\(index + 1): \(line.trimmingCharacters(in: .whitespaces))"
                )
            }
        }

        #expect(
            offenders.isEmpty,
            """
            This app declares no SwiftUI `Settings` scene, so `OpenSettingsAction` \
            is a silent no-op — a button that renders and does nothing. Use \
            `@Environment(\\.openWindow)` + `openWindow(id: "settings")`, and go \
            through `SettingsRouter.route(_:open:)` if a specific tab is wanted. \
            Offenders:
            \(offenders.joined(separator: "\n"))
            """
        )
    }

    /// The stripper is load-bearing — if it blanked too much, the scan above
    /// would pass vacuously — so its behaviour is pinned directly.
    @Test("the comment stripper keeps code and drops only comments")
    func stripperBehaviour() throws {
        let pattern = try Self.forbiddenPattern()
        func hits(_ source: String) -> Int {
            let stripped = OpenSettingsActionAbsenceTests.strippingCommentsAndStringLiterals(source)
            let range = NSRange(stripped.startIndex..<stripped.endIndex, in: stripped)
            return pattern.numberOfMatches(in: stripped, range: range)
        }

        // Exempt: the explanatory comments the fix deliberately leaves behind.
        #expect(hits("/// use openSettings() never\n") == 0)
        #expect(hits("// @Environment(\\.openSettings)\n") == 0)
        #expect(hits("/* openSettings()\n   still a comment */\n") == 0)

        // Caught: real code, including the shapes a naive line check misses.
        #expect(hits("openSettings()\n") == 1)
        #expect(hits("openSettings ()\n") == 1)
        #expect(hits("@Environment(\\.openSettings) var x\n") == 1)
        #expect(hits("/* c */ openSettings()\n") == 1)
        #expect(hits("let x = 1 // note\nopenSettings()\n") == 1)

        // Not caught: the working bridge, which shares the prefix.
        #expect(hits("AppDelegate.openSettingsWindow?()\n") == 0)

        // Swift block comments nest; an inner `*/` must not end the outer one.
        #expect(hits("/* /* openSettings() */ openSettings() */\nlet a = 1\n") == 0)
        // Code AFTER a nested comment closes is still code.
        #expect(hits("/* /* x */ */ openSettings()\n") == 1)

        // FALSE-NEGATIVE guard. A `//` inside a string literal must not open a
        // comment — otherwise the rest of the line, including a real call, goes
        // unseen. This app hard-codes https:// URLs all over its sources, so
        // this is the everyday case, not a contrived one.
        #expect(hits("let u = \"https://example.com\"; openSettings()\n") == 1)
        #expect(hits("let u = #\"https://example.com\"#; openSettings()\n") == 1)
        #expect(hits("let u = \"\"\"\nhttps://a\n\"\"\"\nopenSettings()\n") == 1)
        // A `/*` inside a string must not open a block comment either, which
        // would otherwise swallow every following line in the file.
        #expect(hits("let s = \"/*\"\nopenSettings()\n") == 1)
        // An escaped quote must not be read as the terminator.
        #expect(hits("let s = \"a\\\" // b\"\nopenSettings()\n") == 1)

        // FALSE-POSITIVE guard, the other half of the same mechanism: a string
        // that merely mentions the API is not a call.
        #expect(hits("let s = \"openSettings()\"\n") == 0)
        #expect(hits("let s = #\"openSettings()\"#\n") == 0)
    }

    /// Degenerate inputs the scan will meet if a source file is ever empty,
    /// a single character, or ends inside an unterminated comment. These must
    /// return, not trap — the earlier implementation indexed past `endIndex`
    /// and crashed the whole suite on an empty file.
    @Test("the comment stripper survives degenerate input")
    func stripperEdgeCases() {
        let strip = OpenSettingsActionAbsenceTests.strippingCommentsAndStringLiterals
        #expect(strip("") == "")
        #expect(strip("x") == "x")
        #expect(strip("/") == "/")
        #expect(strip("//") == "  ")
        #expect(strip("/*") == "  ")
        #expect(strip("\"") == " ")
        #expect(strip("#") == "#")
        #expect(strip("let a = 1") == "let a = 1")
        // Unterminated literals must terminate the walk, not run off the end.
        #expect(strip("let s = \"abc").hasPrefix("let s = "))
        #expect(strip("let s = #\"abc").hasPrefix("let s = "))
        // Unterminated block comment: everything after it is blanked, while
        // newlines survive so reported line numbers stay aligned with the file.
        let unterminated = strip("a\n/* b\nc")
        #expect(unterminated.hasPrefix("a\n"))
        #expect(!unterminated.contains("b"))
        #expect(!unterminated.contains("c"))
        #expect(unterminated.filter { $0 == "\n" }.count == 2)
    }

    /// Line numbers in the failure message are only useful if they are right,
    /// which is why comments are blanked to spaces rather than deleted.
    @Test("stripping preserves line count and column positions")
    func strippingPreservesLineGeometry() {
        let source = """
        let a = 1 // trailing
        /* two
           lines */ let b = 2
        let c = 3
        """
        let stripped = OpenSettingsActionAbsenceTests.strippingCommentsAndStringLiterals(source)
        let before = source.components(separatedBy: "\n")
        let after = stripped.components(separatedBy: "\n")
        #expect(before.count == after.count)
        for (original, blanked) in zip(before, after) {
            #expect(original.count == blanked.count)
        }
        // Code trailing a block-comment close on the same line survives, blanked
        // only up to the `*/`, so its column position is unchanged.
        #expect(after[2] == "            let b = 2")
        #expect(after[3] == "let c = 3")
    }

    /// Guard on the guard: if the path arithmetic breaks, the scan above would
    /// find nothing and pass vacuously.
    @Test("the source tree the scan reads is actually there")
    func sourceTreeIsReachable() {
        var isDirectory: ObjCBool = false
        let exists = FileManager.default.fileExists(
            atPath: Self.sourcesRoot.path,
            isDirectory: &isDirectory
        )
        #expect(exists && isDirectory.boolValue)
    }
}

/// Where each failure-recovery button lands.
///
/// The bug this pins: ``QuickstartView`` held
/// `@Environment(\.openSettings)` and called it for its recovery actions. This
/// app declares no SwiftUI ``Settings`` scene — it uses a real
/// `Window("Settings", id: "settings")` — so `OpenSettingsAction` was a silent
/// no-op and those buttons did **nothing**, at the exact moment a first-run
/// download or model start had already failed. Nothing observed it, because
/// the routing decision lived inside a SwiftUI body and a body is not
/// reachable from this suite.
///
/// So the decision was lifted out into
/// ``SettingsRouter/settingsCategory(for:)`` — pure, static, exhaustive — and
/// these tests assert the DESTINATION of every action, not merely that some
/// function got called. A button rewired to the wrong tab, or to nowhere,
/// fails here.
@Suite("Settings deep-link routing")
struct SettingsDeepLinkRoutingTests {
    /// The whole table in one place. Written out literally, and checked for
    /// totality against ``FailureDiagnosis/Action/allCases``, so a newly added
    /// action cannot ship without someone stating where it goes.
    private static let expected: [FailureDiagnosis.Action: SettingsView.Category?] = [
        .openModelManagement: .modelManagement,
        .openWebSearchSettings: .tools,
        // Handled in place by the view that owns the failed operation; opening
        // Settings for these would be a non-sequitur.
        .retry: nil,
        .restart: nil,
        .switchDownloadSource: nil,
    ]

    @Test("the routing table covers every action")
    func tableIsTotal() {
        #expect(Set(Self.expected.keys) == Set(FailureDiagnosis.Action.allCases))
    }

    @Test("each action routes to the tab the table names")
    func eachActionRoutesToItsTab() {
        for (action, category) in Self.expected {
            #expect(
                SettingsRouter.settingsCategory(for: action) == category,
                "\(action) should route to \(String(describing: category))"
            )
        }
    }

    /// The invariant that would have caught the shipped defect on its own: a
    /// button whose label tells the user it will OPEN something has to have
    /// somewhere to open.
    @Test("every action titled \"Open …\" has a real destination")
    func openTitledActionsHaveDestinations() {
        for action in FailureDiagnosis.Action.allCases where action.title.hasPrefix("Open") {
            #expect(
                SettingsRouter.settingsCategory(for: action) != nil,
                "\(action) is titled \"\(action.title)\" but routes nowhere"
            )
        }
    }

    // MARK: - Diagnosis → button → tab, end to end

    @Test(
        "a failed model start lands the user on Model Management",
        arguments: [FailureDiagnosis.Kind.modelOutOfMemory, .modelLoadFailed]
    )
    func modelFailuresRouteToModelManagement(kind: FailureDiagnosis.Kind) throws {
        let action = try #require(FailureDiagnoser.diagnosis(for: kind).action)
        #expect(action == .openModelManagement)
        #expect(SettingsRouter.settingsCategory(for: action) == .modelManagement)
    }

    @Test("a throttled web search lands the user on Tools")
    func rateLimitedSearchRoutesToTools() throws {
        let action = try #require(
            FailureDiagnoser.diagnosis(for: .webSearchRateLimited).action
        )
        #expect(action == .openWebSearchSettings)
        #expect(SettingsRouter.settingsCategory(for: action) == .tools)
    }

    /// No diagnosis may offer a button that opens nothing. This is the whole
    /// defect class, asserted over every kind the app can produce.
    @Test("no diagnosis offers an \"Open …\" button with nowhere to go")
    func noDiagnosisOffersADeadButton() {
        for kind in FailureDiagnosis.Kind.allCases {
            guard let action = FailureDiagnoser.diagnosis(for: kind).action else { continue }
            guard action.title.hasPrefix("Open") else { continue }
            #expect(
                SettingsRouter.settingsCategory(for: action) != nil,
                "\(kind) offers \"\(action.title)\", which routes nowhere"
            )
        }
    }

    // MARK: - The retired openPermissions action

    /// `openPermissions` is gone. It titled a button "Open Permissions" with
    /// no tab to open: this app's ``SettingsView/Category`` set holds no
    /// folder-grant or file-access control (Settings → Privacy is telemetry
    /// consent), and the conditions that produced it are not reachable here —
    /// ``FailureDiagnoser/toolFailureKind`` derives these two kinds only for
    /// the tool names `run_command`, `read_file`, `list_directory`,
    /// `write_file`, `edit_file`, and ``BuiltinToolRegistry`` deliberately
    /// ships none of them.
    ///
    /// The ``Kind`` cases stay — they are persisted in transcripts and must
    /// keep decoding — but they now carry no action, and their copy no longer
    /// promises an "Allow…" control the app does not have.
    @Test(
        "permission-denied kinds no longer offer a button",
        arguments: [FailureDiagnosis.Kind.commandPermissionDenied, .filePermissionDenied]
    )
    func permissionKindsCarryNoAction(kind: FailureDiagnosis.Kind) {
        let diagnosis = FailureDiagnoser.diagnosis(for: kind)
        #expect(diagnosis.action == nil)
        #expect(!diagnosis.message.contains("Allow"))
        // Still a fault, not a user choice — only the affordance changed.
        #expect(diagnosis.severity == .error)
    }

    @Test("a legacy permission-denied row renders no inline tool-card button")
    func permissionKindsRenderNoInlineButton() {
        for kind in [FailureDiagnosis.Kind.commandPermissionDenied, .filePermissionDenied] {
            #expect(
                FailureDiagnosis.inlineToolCardAction(
                    for: FailureDiagnoser.diagnosis(for: kind),
                    canRouteToSettings: true
                ) == nil
            )
        }
    }

    // MARK: - Ordering

    /// ``SettingsView`` consumes ``SettingsRouter/requestedCategory`` from
    /// `.onAppear`, so the category must be assigned BEFORE the window opens;
    /// assigning afterwards races the read and drops the user on the last-used
    /// tab. ``route`` takes the open as a closure precisely so a call site
    /// cannot sequence the two the wrong way round.
    ///
    /// The spy reads ``requestedCategory`` from INSIDE the open closure — that
    /// is the moment `SettingsView.onAppear` would run — so this pins the
    /// ordering as observed by the window, not merely the end state.
    @MainActor
    @Test(
        "route stages the tab before the window opens",
        arguments: [
            (FailureDiagnosis.Action.openModelManagement, SettingsView.Category.modelManagement),
            (FailureDiagnosis.Action.openWebSearchSettings, SettingsView.Category.tools),
        ]
    )
    func routeStagesBeforeOpening(
        action: FailureDiagnosis.Action,
        expected: SettingsView.Category
    ) {
        let router = SettingsRouter()
        var categoryWhenWindowOpened: SettingsView.Category??
        var opens = 0

        let opened = router.route(action) {
            opens += 1
            categoryWhenWindowOpened = router.requestedCategory
        }

        #expect(opened)
        #expect(opens == 1)
        #expect(categoryWhenWindowOpened == expected)
    }

    @MainActor
    @Test("route(to:) stages a directly-named tab before the window opens")
    func routeToCategoryStagesBeforeOpening() {
        let router = SettingsRouter()
        var categoryWhenWindowOpened: SettingsView.Category??

        // The version pill's deep-link: Settings → App.
        router.route(to: .app) { categoryWhenWindowOpened = router.requestedCategory }

        #expect(categoryWhenWindowOpened == .app)
    }

    @MainActor
    @Test("route never opens the window for an in-place action")
    func routeRefusesInPlaceActions() {
        let router = SettingsRouter()
        router.requestedCategory = .appearance

        for action: FailureDiagnosis.Action in [.retry, .restart, .switchDownloadSource] {
            var opened = false
            #expect(router.route(action) { opened = true } == false)
            #expect(!opened, "\(action) must not open Settings")
            #expect(
                router.requestedCategory == .appearance,
                "\(action) must not overwrite a pending deep-link target"
            )
        }
    }
}
