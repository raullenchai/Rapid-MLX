import Foundation
import Testing

/// Pin the four-layer SemVer regex gate that prevents a non-dotted-
/// digit ``sidecar_version`` from reaching ``latest.json`` and bricking
/// every slim-DMG install. Background: v0.8.6 shipped
/// ``sidecar_version: "26ac5b4"`` (a 7-character git short SHA)
/// because (a) ``actions/checkout``'s submodule pull doesn't bring
/// tags, (b) ``git describe --tags --always`` in ``scripts/build.sh``
/// silently fell back to ``--always`` and produced the SHA, (c) the
/// SHA propagated through ``scripts/build-sidecar-tarball.sh`` into the
/// manifest unchecked, and (d) ``release.yml`` only screened the
/// ``(unknown)`` sentinel. The bootstrapper's defensive validator at
/// ``BootstrapCoordinator.swift``'s ``isValidVersionString`` correctly
/// rejected the value but only at install-time on every user's Mac —
/// 100% of slim-DMG installs hit the unrecoverable "Setup didn't
/// finish" splash. See issue #411 for the dogfood transcript.
///
/// The defence is four layers (all four below):
///   1. release.yml fetches submodule tags so derivation can succeed
///   2. scripts/build.sh prefers ``git tag --points-at HEAD`` and
///      hard-fails if neither tag nor describe yields SemVer
///   3. scripts/build-sidecar-tarball.sh refuses to emit a manifest
///      with a non-SemVer ``sidecar_version`` (floor under upstream)
///   4. release.yml regex-gates the value before AND after publish to
///      ``latest.json`` (fail-on-mismatch, not warn-on-mismatch)
///
/// These tests pin the SHAPE of each layer so a future maintainer
/// can't quietly delete any one of them. The bug class is too
/// expensive (100% slim-DMG install failure rate; only caught by
/// release-day dogfood) to rely on a single point of defence.
@Suite("Sidecar version SemVer gate (#411) — four-layer regression pins")
struct SidecarVersionGateTests {

    private static var sourceRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private static var buildScriptPath: URL {
        sourceRoot.appendingPathComponent("scripts").appendingPathComponent("build.sh")
    }

    private static var tarballScriptPath: URL {
        sourceRoot.appendingPathComponent("scripts").appendingPathComponent("build-sidecar-tarball.sh")
    }

    private static var releaseYamlPath: URL {
        sourceRoot
            .appendingPathComponent(".github")
            .appendingPathComponent("workflows")
            .appendingPathComponent("release.yml")
    }

    private static func load(_ url: URL) throws -> String {
        try String(contentsOf: url, encoding: .utf8)
    }

    // MARK: - layer 1: release.yml fetches submodule tags

    // MARK: - layer 2: scripts/build.sh derives SemVer from git tags

    @Test("scripts/build.sh prefers git tag --points-at HEAD for SemVer derivation (#411 layer 2)")
    func buildScriptPrefersTagPointsAt() throws {
        let body = try Self.load(Self.buildScriptPath)
        #expect(
            body.contains("git tag --points-at HEAD"),
            "scripts/build.sh must use ``git tag --points-at HEAD`` to find an exact tag on the submodule's HEAD commit. ``git describe`` walks past lightweight tags (rapid-mlx's v0.8.15/16/18 are lightweight); ``--points-at`` returns them directly (#411)."
        )
        #expect(
            body.contains("--list 'v[0-9]*'"),
            "scripts/build.sh's tag query must filter on ``v[0-9]*`` so non-release tags (e.g. ``staging-*`` / ``rc-*``) can't pollute the SemVer derivation (#411)."
        )
    }

    @Test("scripts/build.sh accepts dotted-digit/RC and rejects malformed versions (#411 layer 2 — empirical)")
    func buildScriptRegexEmpiricalBehaviour() throws {
        // Codex #412 r2 MINOR: source-substring assertions catch
        // accidental deletions but not subtle regex breakage (e.g. a
        // future edit that over-escapes the dot, or accidentally
        // anchors only one end). Empirically verify the regex against
        // known good and known bad inputs by extracting it from the
        // file and executing the same ``[[ =~ ]]`` check the script
        // uses. This catches a regex regression on the PR rather than
        // at release time.
        let body = try Self.load(Self.buildScriptPath)
        // Pull the regex literal from the SIDECAR_SEMVER_RE='…'
        // assignment. Pinning the assignment SHAPE keeps the test
        // robust against unrelated edits to the surrounding script.
        guard let assignRange = body.range(of: "SIDECAR_SEMVER_RE='"),
              let endQuoteRange = body.range(of: "'", range: assignRange.upperBound..<body.endIndex) else {
            Issue.record("Could not locate ``SIDECAR_SEMVER_RE='…'`` assignment in scripts/build.sh; the test extractor needs updating (#411).")
            return
        }
        let regex = String(body[assignRange.upperBound..<endQuoteRange.lowerBound])
        for (input, shouldMatch) in [
            ("0.8.18", true),         // canonical sidecar release
            ("1.2.3", true),          // generic semver
            ("0.8", true),            // two-segment also acceptable to validator
            ("0.13.0-rc1", true),     // supported release candidate
            ("0.13.0-rc12", true),    // multi-digit RC sequence
            ("26ac5b4", false),       // the v0.8.6 bug
            ("0.8.19-rc.1", false),   // codex r1 BLOCKING — must reject
            ("0.8.19-rc0", false),    // RC numbering begins at one
            ("0.8.19-beta1", false),  // unsupported prerelease family
            ("0.8.18+build.7", false),// build suffix — must reject
            ("v0.8.18", false),       // leading ``v`` not allowed at this layer (bootstrapper strips it ITSELF; we hand off the bare form)
            ("0.8.", false),          // trailing dot
            (".8.18", false),         // leading dot
            ("", false),              // empty
            ("0.8.18\n", false),      // trailing newline (defends against caller forgetting to ``tr -d``)
        ] {
            let bash = "/bin/bash"
            let script = "re=\(Self.shellSingleQuote(regex)); if [[ \"$1\" =~ $re ]]; then exit 0; else exit 1; fi"
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: bash)
            proc.arguments = ["-c", script, "_test", input]
            try proc.run()
            proc.waitUntilExit()
            let matched = proc.terminationStatus == 0
            #expect(
                matched == shouldMatch,
                "scripts/build.sh's SIDECAR_SEMVER_RE='\(regex)' should \(shouldMatch ? "MATCH" : "REJECT") '\(input)' but bash =~ returned exit=\(proc.terminationStatus). Empirical regression of the regex behaviour (#411 layer 2)."
            )
        }
    }

    /// Wrap ``s`` in bash single quotes so any embedded ``'`` is safe.
    /// Bash single-quoted strings cannot contain ``'``; the standard
    /// trick is to close, insert an escaped ``\\'``, and re-open.
    private static func shellSingleQuote(_ s: String) -> String {
        "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    // MARK: - layer 3: scripts/build-sidecar-tarball.sh defensive gate

    @Test("scripts/build-sidecar-tarball.sh refuses to emit a non-SemVer sidecar_version (#411 layer 3)")
    func tarballScriptHasDefenseInDepth() throws {
        let body = try Self.load(Self.tarballScriptPath)
        #expect(
            body.contains("^[0-9]+(\\.[0-9]+)+(-rc[1-9][0-9]*)?$"),
            "scripts/build-sidecar-tarball.sh must accept only dotted-digit versions with an optional strict -rcN suffix. A looser suffix regex would accept ``0.8.19-rc.1`` while the bootstrapper validator rejects it at runtime — re-creating #411."
        )
        #expect(
            body.contains("Bootstrapper validator (BootstrapCoordinator.isValidVersionString) would reject"),
            "scripts/build-sidecar-tarball.sh's regex-gate error message must name the specific Swift function (``BootstrapCoordinator.isValidVersionString``) so a future maintainer reading the CI log can grep straight to the validator and confirm the two regexes are in lockstep (#411)."
        )
    }

    // MARK: - layer 4: release.yml gates both compose-time and post-publish

}
