import Foundation
import Testing
@testable import Rapid

@Suite("Mac automation permissions")
struct MacAutomationPermissionsTests {
    @Test("Packaged app can request target-specific browser Automation access")
    func browserAutomationPackagingContract() throws {
        let infoData = try Data(contentsOf: Self.sourceFile("Resources/Info.plist"))
        let info = try #require(
            PropertyListSerialization.propertyList(from: infoData, format: nil)
                as? [String: Any]
        )
        let purpose = try #require(info["NSAppleEventsUsageDescription"] as? String)
        #expect(!purpose.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

        for path in [
            "Resources/Rapid.entitlements",
            "scripts/sidecar-entitlements.plist",
        ] {
            let data = try Data(contentsOf: Self.sourceFile(path))
            let entitlements = try #require(
                PropertyListSerialization.propertyList(from: data, format: nil)
                    as? [String: Any]
            )
            #expect(
                entitlements["com.apple.security.automation.apple-events"] as? Bool
                    == true
            )
        }

        let buildScript = try String(
            contentsOf: Self.sourceFile("scripts/build.sh"), encoding: .utf8
        )
        #expect(!buildScript.contains("SIDECAR_ARGS+=(--skip-codesign --skip-verify)"))
        #expect(buildScript.contains("packaged sidecar Python lacks"))
    }

    @Test("Computer Use is packaged as a stable helper process")
    func computerUseHelperPackagingContract() throws {
        let infoData = try Data(contentsOf: Self.sourceFile(
            "Resources/RapidComputerUseHelper-Info.plist"
        ))
        let info = try #require(
            PropertyListSerialization.propertyList(from: infoData, format: nil)
                as? [String: Any]
        )
        #expect(info["CFBundleIdentifier"] as? String == "com.rapidmlx.rapid.computer-use")
        #expect(info["CFBundleExecutable"] as? String == "RapidComputerUse")
        for key in [
            "NSAccessibilityUsageDescription",
            "NSScreenCaptureUsageDescription",
            "NSAppleEventsUsageDescription",
        ] {
            let purpose = try #require(info[key] as? String)
            #expect(!purpose.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }

        let buildScript = try String(
            contentsOf: Self.sourceFile("scripts/build.sh"), encoding: .utf8
        )
        #expect(buildScript.contains("python/bin/python3.12"))
        #expect(buildScript.contains("Rapid Computer Use.app"))
        #expect(buildScript.contains("com.rapidmlx.rapid.computer-use"))
        #expect(!buildScript.contains("RapidComputerUseHelper/main.c"))
        #expect(buildScript.contains("Computer Use helper has unstable identity"))
    }

    @Test("Computer Use requires both observation and control grants")
    func readinessRequiresBoth() {
        let none = MacAutomationPermissionSnapshot(
            screenRecording: false,
            accessibility: false
        )
        #expect(none.missingForComputerUse == [.screenRecording, .accessibility])
        #expect(!none.isReadyForComputerUse)

        let observationOnly = MacAutomationPermissionSnapshot(
            screenRecording: true,
            accessibility: false
        )
        #expect(observationOnly.missingForComputerUse == [.accessibility])
        #expect(!observationOnly.isReadyForComputerUse)

        let controlOnly = MacAutomationPermissionSnapshot(
            screenRecording: false,
            accessibility: true
        )
        #expect(controlOnly.missingForComputerUse == [.screenRecording])
        #expect(!controlOnly.isReadyForComputerUse)

        let ready = MacAutomationPermissionSnapshot(
            screenRecording: true,
            accessibility: true
        )
        #expect(ready.missingForComputerUse.isEmpty)
        #expect(ready.isReadyForComputerUse)
    }

    @Test("Individual grant lookup is stable")
    func individualLookup() {
        let snapshot = MacAutomationPermissionSnapshot(
            screenRecording: true,
            accessibility: false
        )
        #expect(snapshot.isGranted(.screenRecording))
        #expect(!snapshot.isGranted(.accessibility))
        #expect(MacAutomationPermission.screenRecording.title == "Screen Recording")
        #expect(MacAutomationPermission.accessibility.title == "Accessibility")
    }

    @Test("Automation recovery opens the target-specific privacy pane")
    func automationRecoverySettingsURL() {
        #expect(
            MacAutomationPermissions.automationSettingsURL.absoluteString
                == "x-apple.systempreferences:com.apple.preference.security?Privacy_Automation"
        )
    }

    private static func sourceFile(_ relative: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent(relative)
    }
}
