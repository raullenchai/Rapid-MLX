import Foundation
import Testing

@Suite("Local release script")
struct ReleaseLocalScriptTests {
    @Test("Local --publish is retired fail-closed before any side effect")
    func publishRetiredFailClosesBeforeSideEffects() throws {
        let script = try String(contentsOf: Self.scriptURL, encoding: .utf8)

        // The refusal is the current contract: --publish is gone from the local
        // tool, and the message names the canonical version-bump flow (#2301).
        #expect(script.contains("❌ --publish is retired and cannot be used to release the Desktop app."))
        #expect(script.contains("Use the canonical release flow instead:"))
        #expect(script.contains("chore: bump version to X.Y.Z"))

        // Ordering is the point of the retirement: the refusal is emitted BEFORE
        // the script sources the operator env file or probes the signing identity,
        // so a disabled public command executes no operator-owned shell and touches
        // no secrets.
        let retired = try #require(script.firstRange(of: "❌ --publish is retired"))
        let envSource = try #require(script.range(of: "if [[ -f \"$ENV_FILE\" ]]"))
        #expect(retired.lowerBound < envSource.lowerBound)

        // No remote tag-push / monotonic-release machinery may be resurrected.
        #expect(!script.contains("RAPID_RELEASE_REMOTE"))
        #expect(!script.contains(#"git push ""#))
        #expect(!script.contains("`gh "))
        #expect(!script.contains("PREVIOUS_BUILD"))
        #expect(!script.contains("(( PLIST_BUILD > PREVIOUS_BUILD ))"))
    }

    @Test("Local --check and the dogfood path remain")
    func checkAndDogfoodPathPreserved() throws {
        let script = try String(contentsOf: Self.scriptURL, encoding: .utf8)

        // The preserved local modes are dogfood (default) and --check; both are
        // still parsed and dispatched.
        #expect(script.contains("MODE=\"dogfood\""))
        #expect(script.contains("MODE=\"check\""))
        #expect(script.contains("$0 --check"))
        #expect(script.contains("# verify signing/notary setup only"))
        #expect(script.contains("--check : report setup, build nothing"))
        #expect(script.contains("For a local build only (no tag, no release)"))
    }

    private static var scriptURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("scripts/release-local.sh")
    }
}
