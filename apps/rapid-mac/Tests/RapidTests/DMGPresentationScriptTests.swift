import Foundation
import Testing

@Suite("DMG presentation scripts")
struct DMGPresentationScriptTests {
    @Test("Validator preserves legacy compatibility and detaches by device")
    func validatorCompatibilityAndCleanup() throws {
        let script = try Self.loadScript("validate-dmg.sh")

        #expect(script.contains("LEGACY_ARTIFACT=1"))
        #expect(script.contains("skipped for pre-v0.5.22 legacy artifact"))
        #expect(script.contains("DEVICE=\"$(printf"))
        #expect(script.contains("{ print $1; exit }"))
        #expect(script.contains("detach_target=\"${DEVICE:-$MOUNT}\""))
        #expect(
            script.firstRange(of: "ATTACHED=1")!.lowerBound
                < script.firstRange(of: "could not determine mounted volume path")!.lowerBound
        )
    }

    @Test("Layout writer discards stale state and verifies Finder readback")
    func layoutPersistenceGuard() throws {
        let script = try Self.loadScript("configure-dmg-layout.sh")

        #expect(script.contains("rm -f \"$MOUNT/.DS_Store\""))
        #expect(script.contains("PERSISTED_LAYOUT=\"$(osascript"))
        #expect(script.contains("EXPECTED_LAYOUT=\"180,228|540,228|96|180,120,900,580\""))
        #expect(script.contains("if [[ \"$PERSISTED_LAYOUT\" != \"$EXPECTED_LAYOUT\" ]]"))
    }

    private static func loadScript(_ name: String) throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(
            contentsOf: root.appendingPathComponent("scripts").appendingPathComponent(name),
            encoding: .utf8
        )
    }
}
