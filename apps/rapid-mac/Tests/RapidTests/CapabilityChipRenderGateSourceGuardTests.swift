import Foundation
import Testing
@testable import Rapid

/// #133 source-grep regression guard — the ONLY render-site allowed
/// to consume ``ChatView.capabilityChipKinds`` is the gated helper
/// ``ChatView.capabilityChipKinds(forAlias:)``. Any other UI view file
/// that constructs the chip set MUST route through the alias-gated
/// helper too — otherwise a later refactor that adds a new chip row
/// (e.g. on the QuickstartView empty state, or a chip-row variant for
/// the sessions sidebar) could silently re-introduce the #133
/// over-promise bug by going straight to the un-gated static catalog.
///
/// Same shape as ``ChatViewAssistantContentBidiTests`` (PR #329) —
/// scan the source files literally and assert every chip-set
/// construction site goes through the gate. The unit test isn't a
/// substitute for behavioural coverage (``CapabilityChipsAliasGateTests``
/// pins the function-level contract); it's a tripwire against
/// bypass-shape refactors.
///
/// Bypass shapes we explicitly look for:
///
///   1. Direct reference: ``ChatView.capabilityChipKinds`` (the
///      un-gated static) outside the static-defining file is suspect.
///   2. ``Self.capabilityChipKinds`` inside ``ChatView`` body sites
///      that render chips into a view (the chip-row HStack used to
///      use this shape; PR #133 routes it through
///      ``Self.capabilityChipKinds(forAlias:)``).
///   3. ``var something = capabilityChipKinds`` alias-rebinding
///      (a re-export under a new identifier still has to feed through
///      the gate).
@Suite("#133 — capability-chip render-site source-grep regression guard")
struct CapabilityChipRenderGateSourceGuardTests {

    /// Repository root, derived from ``#filePath`` so the test runs
    /// from any cwd (swift test, Xcode, CI).
    private static var sourceRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // RapidTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // repo root
    }

    private func loadSource(_ relativePath: String) throws -> String {
        let url = Self.sourceRoot.appendingPathComponent(relativePath)
        return try String(contentsOf: url, encoding: .utf8)
    }

    // MARK: - ChatView.swift: the chip-row render site

    /// Walk the chip-row neighbourhood (the slice of the file
    /// containing the ``CapabilityChip(icon:`` constructor) and
    /// assert NO call site there uses the un-gated static directly.
    ///
    /// Two bypass shapes are scanned for:
    ///   a) ``ForEach(Self.capabilityChipKinds,`` (the literal old call)
    ///   b) ``ForEach(ChatView.capabilityChipKinds,`` (qualified)
    ///
    /// The matched-call helper at the bottom of
    /// ``ChatViewAssistantContentBidiTests`` was the model for this
    /// search but we deliberately keep this match narrower because
    /// the chip-row body is a small file slice — a fuzzy regex would
    /// pull in unrelated identifier mentions in long comment blocks.
    @Test("ChatView.swift chip-row body contains NO un-gated ForEach over capabilityChipKinds")
    func chipRowHasNoUngatedForEach() throws {
        let source = try loadSource("Sources/Rapid/UI/ChatView.swift")
        let stripped = Self.stripCommentsAndWhitespace(source)
        for bypass in [
            "ForEach(Self.capabilityChipKinds,",
            "ForEach(ChatView.capabilityChipKinds,",
        ] {
            let strippedBypass = Self.stripCommentsAndWhitespace(bypass)
            #expect(
                !stripped.contains(strippedBypass),
                "ChatView.swift contains a bypass shape '\(bypass)' — a ForEach that iterates the un-gated static catalog. This re-opens the #133 over-promise bug: .broken/.unknown aliases will render the full chip row again. Route the ForEach through Self.capabilityChipKinds(forAlias: alias) instead."
            )
        }
    }

    // MARK: - Other view files: no chip-row constructions outside ChatView

    /// No source file outside ``ChatView.swift`` should reference
    /// ``capabilityChipKinds`` at all today — the chip row is owned
    /// by ChatView. If a future view (e.g. an embedded chat-pane in
    /// a sidecar window, the QuickAsk popover, the Quickstart hero)
    /// needs to render the chips, it MUST route through the gated
    /// helper, not the un-gated catalog. This test is the tripwire
    /// that forces the conversation.
    ///
    /// Codex r3 NIT — scan all of ``Sources/Rapid/`` (not just
    /// ``UI/``) so future QuickAsk-style or service-layer consumers
    /// are also covered. The gated-call check is tightened to require
    /// the FULL ``(forAlias:`` opener (stripped form), not just an
    /// opening paren — so a hypothetical ``capabilityChipKinds(other:``
    /// overload would not be silently accepted.
    @Test("No source file outside ChatView.swift references capabilityChipKinds (un-gated)")
    func noOtherFileReferencesUngatedCatalog() throws {
        let sourceTreeRoot = Self.sourceRoot.appendingPathComponent("Sources/Rapid")
        let enumerator = try #require(
            FileManager.default.enumerator(
                at: sourceTreeRoot,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            ),
            "Could not enumerate Sources/Rapid — directory missing?"
        )
        for case let url as URL in enumerator
            where url.pathExtension == "swift"
            && url.lastPathComponent != "ChatView.swift"
        {
            let body = try String(contentsOf: url, encoding: .utf8)
            // Strip the comment + whitespace canonical form so a
            // comment that LITERALLY mentions the identifier (e.g.
            // a "see ChatView.capabilityChipKinds" doc link) does
            // not trip the gate. The shape we forbid is the
            // un-gated catalog ``capabilityChipKinds`` (any form)
            // that does NOT continue into ``(forAlias:``.
            let stripped = Self.stripCommentsAndWhitespace(body)
            let gatedOpener = "(forAlias:"
            var searchStart = stripped.startIndex
            let needle = "capabilityChipKinds"
            while let range = stripped.range(of: needle, range: searchStart..<stripped.endIndex) {
                // Check that what follows the identifier begins with
                // the EXACT gated-call opener ``(forAlias:`` — not
                // just an open paren. A different overload (e.g.
                // ``capabilityChipKinds(forAlias:filter:)`` would
                // still match; a different paren-prefixed identifier
                // like a hypothetical ``capabilityChipKinds(rawAlias:``
                // does NOT.
                let afterIdentifier = stripped[range.upperBound...]
                let isGatedCall = afterIdentifier.hasPrefix(gatedOpener)
                #expect(
                    isGatedCall,
                    "\(url.lastPathComponent) references capabilityChipKinds without the gated '\(gatedOpener)' opener. The un-gated static is the #133 over-promise hole — route through ChatView.capabilityChipKinds(forAlias:) instead."
                )
                searchStart = range.upperBound
            }
        }
    }

    /// Sister check: ChatView itself should not contain a
    /// ``ChatView.capabilityChipKinds`` self-qualified reference that
    /// drops the ``(forAlias:)`` opener. The static catalog is still
    /// publicly accessible (the existing tests pin its shape against
    /// the canonical four) — what's forbidden is a render-side
    /// consumer that goes around the gate.
    ///
    /// Allowed shapes:
    ///   - ``static let capabilityChipKinds: [CapabilityChipKind] =``
    ///     (the catalog declaration)
    ///   - ``return capabilityChipKinds`` (the .known branch of the
    ///     gated helper)
    ///   - ``Self.capabilityChipKinds(forAlias: alias)`` (gated call)
    ///
    /// Forbidden shapes:
    ///   - ``Self.capabilityChipKinds`` followed by ``,`` ``)`` or
    ///     ``.map`` — passing the un-gated catalog into a ForEach /
    ///     map / etc. would round-trip the bug.
    @Test("ChatView.swift: no un-gated 'Self.capabilityChipKinds' render-side consumer")
    func chatViewSelfQualifiedUngatedConsumer() throws {
        let source = try loadSource("Sources/Rapid/UI/ChatView.swift")
        let stripped = Self.stripCommentsAndWhitespace(source)
        // Scan for "Self.capabilityChipKinds" and inspect the next
        // character — anything that isn't "(" means a static-array
        // read (e.g. passed to ForEach / map / count). The catalog
        // declaration site uses "static let capabilityChipKinds" so
        // we don't catch it (no "Self." prefix).
        var searchStart = stripped.startIndex
        let needle = "Self.capabilityChipKinds"
        while let range = stripped.range(of: needle, range: searchStart..<stripped.endIndex) {
            let nextIndex = range.upperBound
            let nextChar: Character
            if nextIndex < stripped.endIndex {
                nextChar = stripped[nextIndex]
            } else {
                nextChar = " "
            }
            #expect(
                nextChar == "(",
                "ChatView.swift has a 'Self.capabilityChipKinds' reference NOT followed by '(' (next char='\(nextChar)') — this is an un-gated read of the static catalog and re-opens the #133 over-promise bug. Pass through Self.capabilityChipKinds(forAlias: alias) instead."
            )
            searchStart = range.upperBound
        }
    }

    /// Codex r1 MAJOR: scan EVERY occurrence of the identifier
    /// ``capabilityChipKinds`` in ``ChatView.swift`` (with or without
    /// the ``Self.`` / ``ChatView.`` qualifier) and require each one
    /// to match one of the explicit ALLOWED shapes below. Any other
    /// shape — alias-rebinding (``let foo = capabilityChipKinds``),
    /// wrapped-RHS (``some(capabilityChipKinds)``), or a fresh
    /// render-side ForEach (``ForEach(capabilityChipKinds,``) — is a
    /// silent re-introduction of the #133 over-promise bug.
    ///
    /// Allowed shapes (each pre-stripped to canonical form):
    ///
    /// 1. ``staticletcapabilityChipKinds:[CapabilityChipKind]=`` —
    ///    the static catalog declaration. Exactly one site.
    /// 2. ``capabilityChipKinds(forAlias:`` — the gated function
    ///    definition AND every call site that goes through the gate.
    /// 3. ``returncapabilityChipKinds`` — the ``.known`` branch of
    ///    ``capabilityChipKinds(forAlias:)`` that returns the full
    ///    catalog. Exactly one site.
    /// 4. ``=capabilityChipKinds`` — the default-argument value in
    ///    ``resolvePendingForcedTool(... chipKinds: [CapabilityChipKind]
    ///    = capabilityChipKinds)``. This is NOT a render site (it's
    ///    pure data passed to the seed-prefix matcher) but it is the
    ///    one legitimate non-render consumer of the un-gated catalog.
    ///    Exactly one site.
    /// 5. ``chipKinds:[CapabilityChipKind]=capabilityChipKinds`` —
    ///    sub-form of 4 (matched separately because the strip may
    ///    glue the parts differently depending on Swift formatting).
    ///
    /// Anything outside this allowlist trips the gate with a pointer
    /// to the specific surrounding bytes so the reviewer can see what
    /// shape was introduced.
    @Test("ChatView.swift: every capabilityChipKinds reference matches an allowed shape")
    func everyChatViewReferenceMatchesAllowedShape() throws {
        let source = try loadSource("Sources/Rapid/UI/ChatView.swift")
        let stripped = Self.stripCommentsAndWhitespace(source)
        // The canonical (stripped) allowed-shape patterns. Each
        // is checked against the bytes IMMEDIATELY surrounding the
        // identifier occurrence — we walk every occurrence and ask
        // "does the context here begin with one of these shapes?"
        let allowedSlicesStartingAtIdentifier: [String] = [
            // Static catalog declaration: ``static let capabilityChipKinds: [CapabilityChipKind] =``
            // Identifier-relative slice: starts AT "capabilityChipKinds" and walks forward.
            "capabilityChipKinds:[CapabilityChipKind]=",
            // Gated function call (call site, e.g.
            // ``Self.capabilityChipKinds(forAlias: alias)``). Walks
            // forward from the identifier into "(forAlias:".
            "capabilityChipKinds(forAlias:",
            // Gated function DEFINITION: ``func capabilityChipKinds(
            // forAlias alias: String)`` — the strip glues the parameter
            // labels together so the canonical form is
            // ``capabilityChipKinds(forAliasalias:``. Keep both shapes
            // explicit so a future Swift API-label tighten (e.g.
            // ``forAlias _:``) trips this rather than silently passing.
            "capabilityChipKinds(forAliasalias:",
        ]
        // Patterns that must IMMEDIATELY PRECEDE the identifier
        // (looking backward from the identifier's start). These are
        // INTENTIONALLY narrow — broader patterns like a bare ``=``,
        // ``let``, ``Self.``, or ``ChatView.`` would allow rebinding
        // shapes (e.g. ``let bypass = ChatView.capabilityChipKinds``)
        // to slip through. Each entry below is tied to a specific
        // legitimate site we don't want to over-fit on.
        //
        // Codex r2 MAJOR: ``Self.`` / ``ChatView.`` were previously
        // here and accepted any qualified READ of the catalog. The
        // forward-allowlist's ``capabilityChipKinds(forAlias:`` entry
        // is sufficient to cover qualified gated CALLS — qualified
        // reads (without the ``(forAlias:`` opener) are now rejected.
        let allowedPrecedingContexts: [String] = [
            // ``return capabilityChipKinds`` — the .known branch of
            // the gated helper. Uniquely identified by the keyword.
            "return",
            // Default argument value in ``resolvePendingForcedTool``:
            // ``chipKinds: [CapabilityChipKind] = capabilityChipKinds``.
            // We require the FULL slice including the parameter name +
            // type + ``=`` so a different identifier-rebinding shape
            // (e.g. ``static let _bypass: [CapabilityChipKind] =``)
            // does NOT match. ``chipKinds`` is the unique parameter
            // label here; if it's later renamed the test trips and
            // forces a deliberate re-allowlist.
            "chipKinds:[CapabilityChipKind]=",
        ]

        var searchStart = stripped.startIndex
        let needle = "capabilityChipKinds"
        while let range = stripped.range(of: needle, range: searchStart..<stripped.endIndex) {
            let (forwardMatches, backwardMatches, forwardSlice, backwardSlice) =
                Self.matchAllowlist(
                    at: range,
                    in: stripped,
                    forwardAllowed: allowedSlicesStartingAtIdentifier,
                    backwardAllowed: allowedPrecedingContexts
                )
            #expect(
                forwardMatches || backwardMatches,
                """
                ChatView.swift contains a 'capabilityChipKinds' reference \
                that matches NEITHER a forward-allowed shape \
                (\(allowedSlicesStartingAtIdentifier)) NOR a backward- \
                allowed context (\(allowedPrecedingContexts)). \
                Surrounding bytes (back='...\(backwardSlice)', \
                fwd='\(forwardSlice)...'). This is a silent re-introduction \
                of the #133 over-promise bug — route through \
                capabilityChipKinds(forAlias:) instead.
                """
            )
            searchStart = range.upperBound
        }
    }

    /// Codex r2: extracted matcher so the production guard and the
    /// meta-pin self-tests run the EXACT same logic. Returns whether
    /// the identifier occurrence at ``range`` matches a forward- or
    /// backward-allowed pattern, plus the slices used for diagnostics.
    /// Window sizes intentionally match: 64 chars forward (covers the
    /// longest forward shape — function-definition with parameter
    /// signature) and 64 chars backward (covers the
    /// ``chipKinds:[CapabilityChipKind]=`` default-arg slice + margin).
    static func matchAllowlist(
        at range: Range<String.Index>,
        in stripped: String,
        forwardAllowed: [String],
        backwardAllowed: [String]
    ) -> (forward: Bool, backward: Bool, forwardSlice: String, backwardSlice: String) {
        let forwardEnd = stripped.index(
            range.upperBound,
            offsetBy: 64,
            limitedBy: stripped.endIndex
        ) ?? stripped.endIndex
        let forwardSlice = String(stripped[range.lowerBound..<forwardEnd])
        let backStart = stripped.index(
            range.lowerBound,
            offsetBy: -64,
            limitedBy: stripped.startIndex
        ) ?? stripped.startIndex
        let backwardSlice = String(stripped[backStart..<range.lowerBound])
        let forwardMatches = forwardAllowed.contains { forwardSlice.hasPrefix($0) }
        let backwardMatches = backwardAllowed.contains { backwardSlice.hasSuffix($0) }
        return (forwardMatches, backwardMatches, forwardSlice, backwardSlice)
    }

    // MARK: - guard self-test: meta-pin that the allowlist actually rejects bypass shapes

    /// Codex r1 MAJOR — self-test the guard. Construct synthetic
    /// "ChatView.swift slice" strings that contain each bypass shape
    /// the guard is supposed to catch, run them through the SAME
    /// matcher the production guard uses (extracted into
    /// ``matchAllowlist``), and assert the test would have FAILED.
    /// Without this self-test a future allowlist over-broadening
    /// would silently re-open the bypass without any signal.
    ///
    /// Codex r2 MAJOR — added qualified-bypass shapes
    /// (``ChatView.capabilityChipKinds`` / ``Self.capabilityChipKinds``)
    /// after dropping ``Self.`` / ``ChatView.`` from the generic
    /// preceding allowlist. A qualified READ of the catalog (without
    /// the ``(forAlias:`` opener) is now correctly rejected.
    ///
    /// Each case below is a stripped synthetic — they don't need to
    /// parse as Swift; the test only exercises the allowlist matcher.
    @Test("Guard meta-pin: synthetic bypass shapes trigger the gate")
    func guardRejectsBypassShapes() {
        // Mirror the exact arrays the production guard uses. If a
        // refactor adds a new entry there, mirror it here BEFORE
        // running this test — or the meta-pin's coverage drifts out
        // of sync with the live guard.
        let allowedSlicesStartingAtIdentifier: [String] = [
            "capabilityChipKinds:[CapabilityChipKind]=",
            "capabilityChipKinds(forAlias:",
            "capabilityChipKinds(forAliasalias:",
        ]
        let allowedPrecedingContexts: [String] = [
            "return",
            "chipKinds:[CapabilityChipKind]=",
        ]

        let bypassCases: [(stripped: String, label: String)] = [
            (
                "staticlet_bypass:[CapabilityChipKind]=capabilityChipKinds",
                "static-let alias-rebind to a non-chipKinds identifier"
            ),
            (
                "letlocalChips=capabilityChipKinds",
                "local-let alias-rebind"
            ),
            (
                "ForEach(capabilityChipKinds,id:\\.title)",
                "un-qualified ForEach over the catalog"
            ),
            (
                "letxs=capabilityChipKinds.map{$0.title}",
                "method call on the un-gated catalog"
            ),
            (
                "varsink:[CapabilityChipKind]=capabilityChipKinds",
                "var rebinding (mutable variant of static-let bypass)"
            ),
            // Codex r2: qualified bypass shapes — a raw read with the
            // type-qualifier MUST be rejected because the forward
            // allowlist only accepts qualified CALLS (with the
            // ``(forAlias:`` opener), not qualified READS.
            (
                "letx=ChatView.capabilityChipKinds",
                "ChatView-qualified raw read (qualified rebinding)"
            ),
            (
                "letx=Self.capabilityChipKinds",
                "Self-qualified raw read (qualified rebinding)"
            ),
            (
                "letxs=ChatView.capabilityChipKinds.map{$0.title}",
                "ChatView-qualified method call on the un-gated catalog"
            ),
            (
                "ForEach(ChatView.capabilityChipKinds,id:\\.title)",
                "ChatView-qualified un-gated ForEach"
            ),
        ]

        for c in bypassCases {
            var rejected = false
            var searchStart = c.stripped.startIndex
            let needle = "capabilityChipKinds"
            while let range = c.stripped.range(of: needle, range: searchStart..<c.stripped.endIndex) {
                let result = Self.matchAllowlist(
                    at: range,
                    in: c.stripped,
                    forwardAllowed: allowedSlicesStartingAtIdentifier,
                    backwardAllowed: allowedPrecedingContexts
                )
                if !(result.forward || result.backward) {
                    rejected = true
                    break
                }
                searchStart = range.upperBound
            }
            #expect(
                rejected,
                "Bypass shape '\(c.label)' (stripped='\(c.stripped)') was NOT rejected by the allowlist matcher. This means the production guard would let this shape slip through — a silent re-opening of #133. Tighten ``allowedPrecedingContexts`` / ``allowedSlicesStartingAtIdentifier``."
            )
        }
    }

    /// Sister meta-pin: the EXACT legitimate shapes the guard whitelists
    /// must still be accepted. A future allowlist tightening that breaks
    /// the legit forms would surface here BEFORE the production gate
    /// test surfaces it as a "no longer compiles"-style noise.
    @Test("Guard meta-pin: known-legit shapes are accepted by the allowlist")
    func guardAcceptsLegitShapes() {
        let allowedSlicesStartingAtIdentifier: [String] = [
            "capabilityChipKinds:[CapabilityChipKind]=",
            "capabilityChipKinds(forAlias:",
            "capabilityChipKinds(forAliasalias:",
        ]
        let allowedPrecedingContexts: [String] = [
            "return",
            "chipKinds:[CapabilityChipKind]=",
        ]

        let legitCases: [(stripped: String, label: String)] = [
            (
                "staticletcapabilityChipKinds:[CapabilityChipKind]=[]",
                "catalog declaration"
            ),
            (
                "returncapabilityChipKinds",
                "return statement inside .known branch"
            ),
            (
                "letgated=Self.capabilityChipKinds(forAlias:alias)",
                "Self-qualified gated call (still allowed — the forward shape matches)"
            ),
            (
                "ChatView.capabilityChipKinds(forAlias:alias)",
                "Type-qualified gated call from another file"
            ),
            (
                "chipKinds:[CapabilityChipKind]=capabilityChipKinds)",
                "default-argument value in resolvePendingForcedTool"
            ),
            (
                "staticfunccapabilityChipKinds(forAliasalias:String)->[CapabilityChipKind]{",
                "function definition (stripped form, Swift API labels glued)"
            ),
        ]

        for c in legitCases {
            var allAccepted = true
            var searchStart = c.stripped.startIndex
            let needle = "capabilityChipKinds"
            while let range = c.stripped.range(of: needle, range: searchStart..<c.stripped.endIndex) {
                let result = Self.matchAllowlist(
                    at: range,
                    in: c.stripped,
                    forwardAllowed: allowedSlicesStartingAtIdentifier,
                    backwardAllowed: allowedPrecedingContexts
                )
                if !(result.forward || result.backward) {
                    allAccepted = false
                    break
                }
                searchStart = range.upperBound
            }
            #expect(
                allAccepted,
                "Legit shape '\(c.label)' (stripped='\(c.stripped)') was REJECTED by the allowlist. A future allowlist tightening over-fired — restore the relevant entry."
            )
        }
    }

    // MARK: - Strip helper (mirrors ChatViewAssistantContentBidiTests)

    /// Strip ``//`` line comments, ``/*…*/`` block comments, and all
    /// whitespace so the source-grep tests can pin against a
    /// canonical form. The same shape ``ChatViewAssistantContentBidiTests``
    /// uses (PR #329 carryover) — we keep it private so the two
    /// suites can't drift on the strip rules.
    static func stripCommentsAndWhitespace(_ source: String) -> String {
        var out = ""
        out.reserveCapacity(source.count)
        var i = source.startIndex
        while i < source.endIndex {
            let c = source[i]
            // Block comment
            if c == "/", source.index(after: i) < source.endIndex,
               source[source.index(after: i)] == "*" {
                var j = source.index(i, offsetBy: 2)
                while j < source.endIndex {
                    if source[j] == "*",
                       source.index(after: j) < source.endIndex,
                       source[source.index(after: j)] == "/" {
                        j = source.index(j, offsetBy: 2)
                        break
                    }
                    j = source.index(after: j)
                }
                i = j
                continue
            }
            // Line comment
            if c == "/", source.index(after: i) < source.endIndex,
               source[source.index(after: i)] == "/" {
                var j = source.index(after: i)
                while j < source.endIndex, source[j] != "\n" {
                    j = source.index(after: j)
                }
                i = j
                continue
            }
            // Strip whitespace
            if c.isWhitespace {
                i = source.index(after: i)
                continue
            }
            out.append(c)
            i = source.index(after: i)
        }
        return out
    }
}
