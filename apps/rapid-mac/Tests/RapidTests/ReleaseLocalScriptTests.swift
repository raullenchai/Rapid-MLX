import Foundation
import Testing

@Suite("Local release script")
struct ReleaseLocalScriptTests {
    @Test("Publish resolves and validates the canonical release remote")
    func canonicalRemoteGuard() throws {
        let script = try String(contentsOf: Self.scriptURL, encoding: .utf8)
        #expect(script.contains("RAPID_RELEASE_REMOTE"))
        #expect(script.contains("raullenchai/Rapid-MLX"))
        #expect(script.contains("RELEASE_FETCH_URL"))
        #expect(script.contains("RELEASE_PUSH_URL"))
        #expect(script.contains(#"^https://github\.com/raullenchai/Rapid-MLX"#))
        #expect(script.contains(#"^git@github\.com:raullenchai/Rapid-MLX"#))
        #expect(script.contains(#"git push "$RELEASE_REMOTE" "$TAG""#))
        #expect(!script.contains(#"git push origin "$TAG""#))
        #expect(!script.contains("git rev-parse origin/main"))
    }

    private static var scriptURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("scripts/release-local.sh")
    }
}
