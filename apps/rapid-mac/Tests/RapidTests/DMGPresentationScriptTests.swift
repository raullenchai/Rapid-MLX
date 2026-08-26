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
        #expect(script.contains("validate final bootstrapper DMG presentation"))
        #expect(script.contains("bash \"$ROOT/scripts/validate-dmg.sh\" \"$DMG\""))
    }

    @Test("Release validates the canonical DMG after stapling and before upload")
    func releasePostStapleValidationOrder() throws {
        // The DMG notarise → final-validate contract moved into the shared
        // desktop-releasable composite when the build internals were extracted
        // from rapid-mac-release.yml (#2301), so the ordering must be asserted
        // against the composite where the steps actually live.
        let composite = try String(
            contentsOf: Self.monorepoRoot
                .appendingPathComponent(".github/actions/desktop-releasable/action.yml"),
            encoding: .utf8
        )
        let notarize = try #require(composite.firstRange(of: "- name: Notarise + staple rapid-mlx-desktop.dmg"))
        let finalValidation = try #require(composite.firstRange(of: "- name: Validate final stapled DMG presentation"))
        #expect(notarize.lowerBound < finalValidation.lowerBound)

        // In the workflow, the composite-use step must run BEFORE the workflow
        // uploads the DMG artifact — nothing uploads a DMG that wasn't final-
        // validated inside the composite.
        let workflow = try String(
            contentsOf: Self.monorepoRoot
                .appendingPathComponent(".github/workflows/rapid-mac-release.yml"),
            encoding: .utf8
        )
        let compositeUse = try #require(workflow.firstRange(of: "uses: ./.github/actions/desktop-releasable"))
        let upload = try #require(workflow.firstRange(of: "- name: Upload workflow artifact"))
        #expect(compositeUse.lowerBound < upload.lowerBound)
    }

    @Test("Structural parser accepts the active icvp background alias")
    func structuralBackgroundAliasPasses() throws {
        let result = try Self.runBackgroundVerifier(
            alias: Self.makeFinderAlias()
        )

        #expect(result.status == 0)
        #expect(result.output.contains("verify-dmg-background: OK"))
    }

    @Test("Structural parser rejects unrelated matching strings")
    func unrelatedBackgroundStringsFail() throws {
        let result = try Self.runBackgroundVerifier(
            alias: Self.makeFinderAlias(posixPath: "/wrong/background.png"),
            trailingData: Data("Rapid-MLX Desktop:.background:/backgroundImageAlias/.background/background.png".utf8)
        )

        #expect(result.status == 1)
        #expect(result.output.contains("path tags target the wrong file"))
    }

    @Test("Structural parser rejects non-image icvp records")
    func nonImageBackgroundTypeFails() throws {
        let result = try Self.runBackgroundVerifier(
            alias: Self.makeFinderAlias(),
            backgroundType: 0
        )

        #expect(result.status == 1)
        #expect(result.output.contains("backgroundType is not image mode"))
    }

    @Test("Structural parser reports a truncated icvp length without traceback")
    func truncatedICVPLengthFailsCleanly() throws {
        var fixture = Data("prefix-icvpblob".utf8)
        fixture.append(contentsOf: [0, 0, 0, 32])
        fixture.append(contentsOf: Data("short".utf8))
        let result = try Self.runVerifier(fixture: fixture)

        #expect(result.status == 1)
        #expect(result.output.contains("icvp blob payload is truncated"))
        #expect(!result.output.contains("Traceback"))
    }

    private static func expectBackgroundAliasContract(in script: String) {
        #expect(script.contains("python3 \"$ROOT/scripts/verify-dmg-background.py\" \"$MOUNT/.DS_Store\""))
        #expect(!script.contains("strings -a \"$MOUNT/.DS_Store\""))
    }

    private static func runBackgroundVerifier(
        alias: Data,
        backgroundType: Int = 2,
        trailingData: Data = Data()
    ) throws -> (status: Int32, output: String) {
        let plist = try PropertyListSerialization.data(
            fromPropertyList: [
                "backgroundImageAlias": alias,
                "backgroundType": backgroundType,
            ],
            format: .binary,
            options: 0
        )
        var fixture = Data("DSStore fixture icvpblob".utf8)
        var length = UInt32(plist.count).bigEndian
        withUnsafeBytes(of: &length) { fixture.append(contentsOf: $0) }
        fixture.append(plist)
        fixture.append(trailingData)

        return try runVerifier(fixture: fixture)
    }

    private static func runVerifier(fixture: Data) throws -> (status: Int32, output: String) {
        let fixtureDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-dmg-ds-store-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: fixtureDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: fixtureDirectory) }
        let fixtureURL = fixtureDirectory.appendingPathComponent(".DS_Store")
        try fixture.write(to: fixtureURL)

        let output = Pipe()
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [
            "python3",
            sourceRoot.appendingPathComponent("scripts/verify-dmg-background.py").path,
            fixtureURL.path,
        ]
        process.standardOutput = output
        process.standardError = output
        try process.run()
        process.waitUntilExit()
        let data = output.fileHandleForReading.readDataToEndOfFile()
        return (process.terminationStatus, String(decoding: data, as: UTF8.self))
    }

    private static func makeFinderAlias(
        posixPath: String = "/.background/background.png"
    ) -> Data {
        // Alias Manager v2 fixed header is 150 bytes. Extension records below
        // are independently serialized as tag/length/value tuples.
        var alias = Data(repeating: 0, count: 150)
        alias[6] = 0
        alias[7] = 2  // Alias Manager record version.
        alias[8] = 0
        alias[9] = 0  // File target.
        Self.writePascal("Rapid-MLX Desktop", to: &alias, at: 10, capacity: 28)
        Self.writePascal("background.png", to: &alias, at: 50, capacity: 64)

        Self.appendAliasTag(0x0000, Data(".background".utf8), to: &alias)
        Self.appendAliasTag(
            0x0002,
            Data("Rapid-MLX Desktop:.background:\u{0}background.png".utf8),
            to: &alias
        )
        Self.appendAliasTag(0x0012, Data(posixPath.utf8), to: &alias)
        Self.appendUInt16(0xFFFF, to: &alias)
        Self.appendUInt16(0, to: &alias)

        let size = UInt16(alias.count)
        alias[4] = UInt8(size >> 8)
        alias[5] = UInt8(size & 0xFF)
        return alias
    }

    private static func writePascal(
        _ value: String,
        to data: inout Data,
        at offset: Int,
        capacity: Int
    ) {
        let bytes = Array(value.utf8)
        precondition(bytes.count < capacity)
        data[offset] = UInt8(bytes.count)
        data.replaceSubrange((offset + 1)..<(offset + 1 + bytes.count), with: bytes)
    }

    private static func appendAliasTag(_ tag: UInt16, _ value: Data, to data: inout Data) {
        Self.appendUInt16(tag, to: &data)
        Self.appendUInt16(UInt16(value.count), to: &data)
        data.append(value)
        if value.count.isMultiple(of: 2) == false {
            data.append(0)
        }
    }

    private static func appendUInt16(_ value: UInt16, to data: inout Data) {
        data.append(UInt8(value >> 8))
        data.append(UInt8(value & 0xFF))
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

    private static var monorepoRoot: URL {
        sourceRoot.deletingLastPathComponent().deletingLastPathComponent()
    }

    private static func loadScript(_ name: String) throws -> String {
        return try String(
            contentsOf: sourceRoot.appendingPathComponent("scripts").appendingPathComponent(name),
            encoding: .utf8
        )
    }
}
