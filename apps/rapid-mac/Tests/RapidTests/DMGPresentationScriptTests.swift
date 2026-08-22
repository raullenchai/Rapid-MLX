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
        #expect(script.contains("BACKGROUND_SOURCE=\"$ROOT/Resources/dmg-background.png\""))
        #expect(script.contains("cp \"$BACKGROUND_SOURCE\" \"$BACKGROUND\""))
        #expect(!script.contains("sips -s format png"))
        Self.expectBackgroundAliasContract(in: script)
    }

    @Test("Final validator requires the persisted Finder background alias")
    func finalBackgroundAliasGuard() throws {
        Self.expectBackgroundAliasContract(in: try Self.loadScript("validate-dmg.sh"))
    }

    @Test("Committed Finder background is a 720x460 PNG")
    func committedBackgroundDimensions() throws {
        let data = try Data(
            contentsOf: Self.sourceRoot
                .appendingPathComponent("Resources")
                .appendingPathComponent("dmg-background.png")
        )
        let bytes = [UInt8](data)

        #expect(bytes.count > 24)
        #expect(Array(bytes.prefix(8)) == [137, 80, 78, 71, 13, 10, 26, 10])
        #expect(String(bytes: bytes[12..<16], encoding: .ascii) == "IHDR")
        #expect(Self.uint32(bytes, at: 16) == 720)
        #expect(Self.uint32(bytes, at: 20) == 460)
    }

    @Test("Bootstrapper final verification uses a normal volume mount")
    func bootstrapperVerificationMountIdentity() throws {
        let script = try Self.loadScript("build-bootstrapper-dmg.sh")

        #expect(script.contains("VERIFY_ATTACH_OUTPUT=\"$(hdiutil attach \"$DMG\" -nobrowse -readonly)\""))
        #expect(script.contains("VERIFY_DEVICE=\"$(printf"))
        #expect(script.contains("detach_target=\"${VERIFY_DEVICE:-$MOUNT}\""))
        #expect(!script.contains("mktemp -d -t rapid-bootstrap-dmg-XXXXXX"))
    }

    private static func expectBackgroundAliasContract(in script: String) {
        #expect(script.contains("strings -a \"$MOUNT/.DS_Store\""))
        #expect(script.contains("backgroundImageAlias"))
        #expect(script.contains("Rapid-MLX Desktop:.background:"))
        #expect(script.contains("/.background/background.png"))
    }

    private static func uint32(_ bytes: [UInt8], at offset: Int) -> UInt32 {
        bytes[offset..<(offset + 4)].reduce(0) { ($0 << 8) | UInt32($1) }
    }

    private static var sourceRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private static func loadScript(_ name: String) throws -> String {
        return try String(
            contentsOf: sourceRoot.appendingPathComponent("scripts").appendingPathComponent(name),
            encoding: .utf8
        )
    }
}
