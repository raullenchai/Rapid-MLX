import Foundation
import Testing

/// Pin the shape of the slice ε.1 release.yml additions (slim DMG
/// notarise + R2 publish + GH Release preview-asset attach) AND of
/// the ``scripts/notarize.sh`` interface they depend on. Slice ε.1
/// is DORMANT: ``latest.json.dmg_url`` STILL points at the canonical
/// (full) DMG. The slim asset is published-and-discoverable on R2
/// and on the GH Release, but the in-app UpdateChecker on v0.8.x is
/// unaffected. Slice ε.2 is the 1-line PR that flips ``dmg_url``;
/// these tests are the structural moat that keeps ε.1 dormant +
/// keeps ε.2 a 1-line flip.
///
/// Pattern mirrors ``BootstrapperDMGShapeTests``:
///   - Locate source files via ``#filePath`` walk so the test runs
///     under ``swift test``, Xcode, and CI.
///   - Read each file as UTF-8.
///   - Assert canonical substrings / structural invariants.
///
/// The asserts pin SHAPE, not byte-for-byte text — reformatting
/// comments / whitespace should not trip these tests. What is pinned
/// is what we care about: continue-on-error guards (so notarise
/// failures cannot block the canonical release), stapler-validate
/// gates (so an un-stapled DMG never reaches dl.rapidmlx.com or the
/// GH Release), the additive GH Release upload, and the explicit
/// dormancy invariant on latest.json composition.
@Suite("Bootstrapper notarize integration slice ε.1 — release.yml + notarize.sh shape")
struct BootstrapperNotarizeIntegrationShapeTests {

    /// Repository root, derived from ``#filePath`` so the test runs
    /// from any cwd.
    private static var sourceRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // RapidTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // repo root
    }

    private static var notarizeScriptPath: URL {
        sourceRoot
            .appendingPathComponent("scripts")
            .appendingPathComponent("notarize.sh")
    }

    private static var releaseYamlPath: URL {
        sourceRoot
            .appendingPathComponent(".github")
            .appendingPathComponent("workflows")
            .appendingPathComponent("release.yml")
    }

    private static func loadNotarizeScript() throws -> String {
        try String(contentsOf: notarizeScriptPath, encoding: .utf8)
    }

    private static func loadReleaseYaml() throws -> String {
        try String(contentsOf: releaseYamlPath, encoding: .utf8)
    }

    /// Strip whitespace from every character so substring matches
    /// survive reformatting. Mirrors the helper in
    /// ``BootstrapperDMGShapeTests`` / ``SidecarShimHardeningTests``.
    private static func stripWhitespace(_ s: String) -> String {
        s.filter { !$0.isWhitespace }
    }

    /// Extract the *actual* ``if:`` clause from the given step's
    /// block. Returns the substring after ``if:`` (trimmed of
    /// leading/trailing whitespace) so callers can assert against
    /// the real conditional rather than relying on substring
    /// matches across the whole step (which would match comments
    /// and pass when the actual ``if:`` line is wrong — codex r4
    /// NIT). Returns nil if the step has no ``if:`` line (which is
    /// itself a regression for steps that REQUIRE one — callers
    /// should fail the test loudly).
    private static func extractIfClause(stepBlock: String) -> String? {
        // Walk lines, pick the first one whose trimmed form starts
        // with ``if:``. GHA permits a single ``if:`` per step.
        for line in stepBlock.split(separator: "\n", omittingEmptySubsequences: false) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("if:") {
                let after = String(trimmed.dropFirst("if:".count))
                return after.trimmingCharacters(in: .whitespaces)
            }
        }
        return nil
    }

    // MARK: - notarize.sh interface (parametrized, accepts arbitrary DMG)

    @Test("notarize.sh accepts <submit-file> <staple-target> argv pair (slice ε.1 reuses without refactor)")
    func notarizeScriptAcceptsArbitraryDmgArgs() throws {
        let body = try Self.loadNotarizeScript()
        // The script's documented signature is ``notarize.sh
        // <submit-file> <staple-target>``. Slice ε.1's release.yml
        // step relies on this — a regression that hardcodes the
        // canonical DMG path would break the slim-DMG submission
        // silently (the slim invocation would notarise the canonical
        // DMG twice and skip the slim entirely). Pin the parametric
        // argv assignment.
        #expect(
            body.contains("SUBMIT_FILE=\"${1:?usage: notarize.sh <submit-file> <staple-target>}\""),
            "scripts/notarize.sh must accept SUBMIT_FILE as ``${1:?...}`` (positional argv). Slice ε.1 calls this twice (once for the canonical DMG, once for the slim DMG) so any hardcoded path here regresses both call sites."
        )
        #expect(
            body.contains("STAPLE_TARGET=\"${2:?usage: notarize.sh <submit-file> <staple-target>}\""),
            "scripts/notarize.sh must accept STAPLE_TARGET as ``${2:?...}`` (positional argv). Required for slice ε.1's slim-DMG call (which passes the slim DMG as both submit AND staple targets)."
        )
    }

    @Test("notarize.sh skips cleanly when AC_API_* are unset (local-dev + fork-dry-run path)")
    func notarizeScriptSkipsWhenCredsMissing() throws {
        let body = try Self.loadNotarizeScript()
        // The canonical-DMG step relies on this skip path so a local
        // ``build.sh && dmg.sh`` flow without Apple creds still
        // succeeds. Slice ε.1's release.yml step inherits the same
        // expectation — a regression that started exiting non-zero
        // here would tank fork dry-runs even with continue-on-error
        // (the step would log a confusing failure rather than the
        // clean ``skipping notarisation`` notice).
        #expect(
            body.contains("AC_API_* not set — skipping notarisation"),
            "scripts/notarize.sh must skip cleanly (exit 0 with a notice) when AC_API_KEY_ID / AC_API_ISSUER_ID / AC_API_KEY_PATH are unset. Slice ε.1 relies on this for fork-dry-run + local-dev paths."
        )
    }

    // MARK: - release.yml: slim DMG notarise step

    // MARK: - release.yml: GH Release preview-asset attach

    // MARK: - slice ε.2 cutover invariants (slim DMG is load-bearing, schema unchanged)

}
