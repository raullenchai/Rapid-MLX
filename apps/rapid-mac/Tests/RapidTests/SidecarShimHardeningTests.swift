import Foundation
import Testing
@testable import Rapid

/// rapid-desktop #361 — bundled sidecar's ``scripts/sidecar-shim.sh``
/// invoked ``python3.12 -u -s -m vllm_mlx.cli`` without ``-P`` or
/// ``PYTHONSAFEPATH=1``. Python's ``-m`` mode unconditionally prepends
/// cwd to ``sys.path[0]``, so a caller that ``cd``s into a directory
/// containing a sibling ``vllm_mlx/cli.py`` hijacks the bundled
/// import path — the dogfood reproducer printed
/// ``rapid-mlx 9.9.9-cwd-poison`` instead of the real version.
///
/// The fix (belt+suspenders) lives in
/// ``scripts/sidecar-shim.sh``:
///   1. ``-P`` on the python exec line (Python 3.11+ idiom)
///   2. ``export PYTHONSAFEPATH=1`` in the env block above
///
/// These tests pin both layers — the source-grep tripwire guards
/// against a future shim rewrite that drops either layer, and the
/// repro guard exercises the actual bundled sidecar (when present)
/// against the original #361 reproducer.
///
/// Pattern mirrors ``ToolUseCapabilitySourceGuardTests`` (PR #343):
/// strip comments/whitespace, then assert canonical substrings.
@Suite("Sidecar shim cwd-hijack hardening (#361)", TestTimeouts.hangProne)
struct SidecarShimHardeningTests {

    /// Repository root, derived from ``#filePath`` so the test runs
    /// from any cwd (swift test, Xcode, CI).
    private static var sourceRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // RapidTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // repo root
    }

    private static var shimSourcePath: URL {
        sourceRoot
            .appendingPathComponent("scripts")
            .appendingPathComponent("sidecar-shim.sh")
    }

    // MARK: - Source-grep tripwire (always runs)

    /// The python exec line MUST carry ``-P`` between ``python3.12``
    /// and ``-u``. ``-P`` is the canonical Python 3.11+ flag that
    /// suppresses the ``-m``-mode cwd prepend. The bundled python is
    /// 3.12 (see ``scripts/build-sidecar.sh`` ``PYTHON_VERSION``) so
    /// the flag is available.
    ///
    /// If this trips, someone reordered or dropped the flag — restore
    /// the canonical line shape:
    ///
    ///     exec "$ROOT/python/bin/python3.12" -P -u -s -m vllm_mlx.cli "$@"
    /// Strip leading whitespace then split into (executable, comment)
    /// halves at the first non-quoted ``#`` (POSIX shell comment).
    /// Returns the executable half so source-grep tests can ignore
    /// commentary on the SAME line. Quote tracking is conservative —
    /// only single/double quotes; backslash escapes inside double
    /// quotes are not unwound (sufficient for our shim's shape).
    static func executablePart(of line: String) -> String {
        var out = ""
        var inSingle = false
        var inDouble = false
        for c in line {
            if c == "'" && !inDouble { inSingle.toggle(); out.append(c); continue }
            if c == "\"" && !inSingle { inDouble.toggle(); out.append(c); continue }
            if c == "#" && !inSingle && !inDouble {
                break // rest of line is a shell comment
            }
            out.append(c)
        }
        return out
    }

    /// Return only the EXECUTABLE lines (no full-line ``#`` comments,
    /// no blank lines, no in-line comments). Used by the source-grep
    /// tripwires below so a commented-out canonical command can't
    /// satisfy the assertion.
    static func executableLines(of source: String) -> [String] {
        source
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { String($0) }
            .map { Self.executablePart(of: $0).trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }

    /// Predicate for "this is an executable ``exec`` line that
    /// launches the bundled python at vllm_mlx.cli". Shared between
    /// the -P guard and the PYTHONSAFEPATH ordering guard so they
    /// reason about the SAME set of candidate exec lines (one set, no
    /// ``first(where:)`` lies if a dead alternate exec exists).
    private static func isPythonExecLine(_ line: String) -> Bool {
        line.hasPrefix("exec ")
            && line.contains("python3.12")
            && line.contains("vllm_mlx.cli")
    }

    @Test("sidecar-shim.sh python exec line contains -P flag")
    func shimExecLineHasDashCapitalP() throws {
        let source = try String(contentsOf: Self.shimSourcePath, encoding: .utf8)
        let lines = Self.executableLines(of: source)
        // Collect ALL executable ``exec ... python3.12 ... vllm_mlx.cli``
        // lines (not just the first). Per codex r2 MINOR: a dead
        // alternate exec carrying ``-P`` before the real broken exec
        // would satisfy a ``first(where:)`` guard and let the bug
        // ship. Require EVERY candidate to carry ``-P`` so any new
        // exec branch must also be hardened.
        let execLines = lines.filter(Self.isPythonExecLine)
        #expect(
            !execLines.isEmpty,
            "sidecar-shim.sh has no ``exec ... python3.12 ... vllm_mlx.cli`` line in an executable position. Either the shim was restructured (e.g. exec collapsed onto a single ``if X; then exec Y; fi``) or the canonical entrypoint changed. If intentional, update this test; otherwise restore the canonical line:\n    exec \"$ROOT/python/bin/python3.12\" -P -u -s -m vllm_mlx.cli \"$@\""
        )
        // Canonical shape: ``..python3.12"-P-u-s-mvllm_mlx.cli..``.
        // Pin ``"-P`` so a refactor that swaps for `--isolated` or
        // moves the flag past the module name (where -m would eat it)
        // trips here. Every candidate exec line must satisfy it.
        for execLine in execLines {
            let stripped = Self.stripWhitespace(execLine)
            #expect(
                stripped.contains("\"-P-u"),
                "sidecar-shim.sh has a python exec line missing the ``-P`` flag (must appear immediately after the python binary path, before ``-u``). Offending line:\n    \(execLine)\n\nWithout ``-P``, Python's ``-m`` mode prepends cwd to sys.path[0] and a sibling ``vllm_mlx/`` directory in the caller's cwd hijacks the bundled import path. See rapid-desktop #361. Restore the canonical line shape:\n    exec \"$ROOT/python/bin/python3.12\" -P -u -s -m vllm_mlx.cli \"$@\""
            )
        }
    }

    /// The env block MUST set ``PYTHONSAFEPATH=1`` alongside the
    /// existing ``PYTHONHOME`` / ``PYTHONPATH`` / ``PYTHONNOUSERSITE``
    /// pins, in an EXECUTABLE position (not a comment) and BEFORE the
    /// first ``exec`` line so the env var is in scope when the bundled
    /// python is launched. (Per codex r2 MINOR — compare against the
    /// MINIMUM index of all candidate exec lines, not just the first,
    /// so an export inserted between two execs doesn't satisfy the
    /// guard by being before SOME exec.)
    ///
    /// This is the static-analysis-friendly belt that survives a
    /// future shim rewrite that loses the ``-P`` arg on the exec line.
    @Test("sidecar-shim.sh env block exports PYTHONSAFEPATH=1 before exec")
    func shimEnvBlockExportsPythonSafePath() throws {
        let source = try String(contentsOf: Self.shimSourcePath, encoding: .utf8)
        let lines = Self.executableLines(of: source)
        let exportIdx = lines.firstIndex(where: {
            Self.stripWhitespace($0).contains("exportPYTHONSAFEPATH=1")
        })
        let execIndices = lines.enumerated()
            .filter { Self.isPythonExecLine($0.element) }
            .map { $0.offset }
        #expect(
            exportIdx != nil,
            "sidecar-shim.sh has no executable ``export PYTHONSAFEPATH=1`` line. This is the env-var belt for the ``-P`` flag — without it, a future shim rewrite that drops the flag silently re-opens the rapid-desktop #361 cwd-poison hijack."
        )
        if let e = exportIdx, let earliest = execIndices.min() {
            #expect(
                e < earliest,
                "sidecar-shim.sh has ``export PYTHONSAFEPATH=1`` at line-index \(e) but the earliest python exec is at line-index \(earliest). The export must be BEFORE every candidate exec so the env var is in scope for the bundled python. Move the export above the first exec line."
            )
        }
    }

    /// Cross-check: the file actually contains the documented
    /// reference to #361 in a comment, so a maintainer browsing the
    /// shim sees WHY the flags exist (and doesn't strip them in a
    /// "cleanup" pass).
    @Test("sidecar-shim.sh documents the #361 fix in comments")
    func shimDocumentsRapidDesktop361() throws {
        let source = try String(contentsOf: Self.shimSourcePath, encoding: .utf8)
        #expect(
            source.contains("#361"),
            "sidecar-shim.sh does not reference rapid-desktop #361 in a comment. The ``-P`` + PYTHONSAFEPATH=1 pair is non-obvious; a comment pointing at the closed issue is required so a future cleanup pass doesn't strip them as ``redundant``."
        )
    }

    // MARK: - Live repro guard (skipped when bundle absent)

    /// Resolve the bundled sidecar's shim under the dev-time build
    /// stage (``build/sidecar-stage/rapid-mlx/bin/rapid-mlx``) — the
    /// canonical output of ``scripts/build-sidecar.sh`` from this
    /// branch.
    ///
    /// We deliberately do NOT fall back to ``/Applications/Rapid-MLX
    /// Desktop.app/...`` because the installed app may be a release
    /// build that PREDATES this branch — running the live repro
    /// against an unpatched installed bundle would fail this test
    /// even though the source fix on disk is correct. Dev machines
    /// that haven't run ``scripts/build-sidecar.sh`` from this branch
    /// silently skip the live repro; CI nightly that runs the build
    /// script before tests exercises the real path.
    private static func locateBundledShim() -> URL? {
        let stagePath = sourceRoot
            .appendingPathComponent("build")
            .appendingPathComponent("sidecar-stage")
            .appendingPathComponent("rapid-mlx")
            .appendingPathComponent("bin")
            .appendingPathComponent("rapid-mlx")
        return FileManager.default.isExecutableFile(atPath: stagePath.path) ? stagePath : nil
    }

    /// Invoke the bundled shim with the given args from ``cwd``,
    /// returning stdout+stderr concatenated. Throws on spawn failure
    /// or non-zero exit (caller inspects output).
    private static func runShim(_ shim: URL, cwd: URL, args: [String]) async throws -> (output: String, exitCode: Int32) {
        // Strip any inherited PYTHONPATH/PYTHONSAFEPATH from the test
        // host so we exercise the shim's OWN env-pinning behaviour
        // rather than accidentally inheriting it from the parent.
        var env = ProcessInfo.processInfo.environment
        env.removeValue(forKey: "PYTHONPATH")
        env.removeValue(forKey: "PYTHONSAFEPATH")
        env.removeValue(forKey: "PYTHONHOME")
        let result = try await TestSubprocess.run(
            executableURL: shim,
            arguments: args,
            currentDirectoryURL: cwd,
            environment: env
        )
        let data = result.standardOutput + result.standardError
        return (String(decoding: data, as: UTF8.self), result.terminationStatus)
    }

    /// Construct the #361 reproducer (poison ``vllm_mlx/cli.py`` in a
    /// tmp dir) and confirm the bundled shim, run from that cwd,
    /// reads the REAL bundled version rather than the poison string.
    ///
    /// Skipped (returns silently) when no bundled shim is present —
    /// developer machines without a freshly-built sidecar shouldn't
    /// fail the suite. CI nightly that runs after build-sidecar.sh
    /// will exercise the real path.
    @Test("Bundled shim invoked from poison cwd returns real version, not poison")
    func bundledShimResistsPoisonCwd() async throws {
        guard let shim = Self.locateBundledShim() else {
            // No bundle present — skip (dev machine without a fresh build).
            return
        }

        let tmp = try makeTempDir(prefix: "rapid-cwd-poison-361-")
        defer { try? FileManager.default.removeItem(at: tmp) }

        let poisonPkg = tmp.appendingPathComponent("vllm_mlx")
        try FileManager.default.createDirectory(at: poisonPkg, withIntermediateDirectories: true)
        try Data().write(to: poisonPkg.appendingPathComponent("__init__.py"))
        let poisonCLI = """
        def main():
            print("rapid-mlx 9.9.9-cwd-poison")

        if __name__ == "__main__":
            main()
        """
        try poisonCLI.write(
            to: poisonPkg.appendingPathComponent("cli.py"),
            atomically: true,
            encoding: .utf8
        )

        let result = try await Self.runShim(shim, cwd: tmp, args: ["--version"])
        #expect(
            !result.output.contains("9.9.9-cwd-poison"),
            "Bundled sidecar shim at \(shim.path) loaded the poison ``vllm_mlx/cli.py`` from the caller's cwd (\(tmp.path)). The #361 cwd-hijack fix (``-P`` + PYTHONSAFEPATH=1 in sidecar-shim.sh) is not effective in the installed bundle. Output:\n\(result.output)"
        )
        // Real version line starts with ``rapid-mlx 0.`` (every shipped
        // version begins with 0. through current). Sanity-check that
        // the happy path also fired so the assertion above isn't
        // vacuously true (e.g. shim crashed before reaching the CLI).
        #expect(
            result.output.contains("rapid-mlx 0."),
            "Bundled sidecar shim at \(shim.path) did not return a recognisable real version. The shim may have crashed before exec'ing the CLI. Output:\n\(result.output)"
        )
    }

    /// Happy-path regression guard: the bundled shim invoked from a
    /// clean cwd (``/tmp``, no poison sibling) still reads the real
    /// version. Without this, a future ``-P`` strengthening that
    /// over-isolates (e.g. blocks the bundled site-packages from
    /// being found at all) would slip through bundledShimResistsPoisonCwd
    /// by virtue of the assertion being satisfied trivially.
    @Test("Bundled shim invoked from clean cwd still reads real version")
    func bundledShimHappyPathFromCleanCwd() async throws {
        guard let shim = Self.locateBundledShim() else {
            return
        }
        let cleanCwd = URL(fileURLWithPath: "/tmp")
        let result = try await Self.runShim(shim, cwd: cleanCwd, args: ["--version"])
        #expect(
            result.exitCode == 0,
            "Bundled sidecar shim at \(shim.path) exited \(result.exitCode) on a clean ``--version`` from /tmp. ``-P`` may have over-isolated the import path. Output:\n\(result.output)"
        )
        #expect(
            result.output.contains("rapid-mlx 0."),
            "Bundled sidecar shim at \(shim.path) did not return a recognisable real version from a clean cwd. Output:\n\(result.output)"
        )
    }

    // MARK: - Helpers

    private func makeTempDir(prefix: String) throws -> URL {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(prefix)\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }

    /// Strip ALL whitespace (incl. tabs/newlines) so source-grep
    /// patterns can pin against a canonical form regardless of shell
    /// indentation. Comments are NOT stripped because shell comments
    /// (``#``) are line-scoped and our needles are line-scoped already.
    static func stripWhitespace(_ source: String) -> String {
        source.filter { !$0.isWhitespace }
    }
}
