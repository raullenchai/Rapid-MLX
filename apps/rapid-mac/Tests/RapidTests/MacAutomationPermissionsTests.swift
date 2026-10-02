import Foundation
import Testing
@testable import Rapid

@Suite("Mac automation permissions")
struct MacAutomationPermissionsTests {
    @Test("Browser Automation allowlist matches trusted URL readers")
    func browserAutomationAllowlist() {
        for bundle in [
            "com.apple.Safari",
            "com.apple.SafariTechnologyPreview",
            "com.google.Chrome",
            "com.google.Chrome.beta",
            "com.microsoft.edgemac",
            "com.microsoft.edgemac.Beta",
            "org.chromium.Chromium",
            "org.chromium.Chromium.canary",
        ] {
            #expect(BrowserAutomationAuthorizer.supports(bundleIdentifier: bundle))
        }
        for bundle in ["", "com.apple.TextEdit", "com.example.untrusted"] {
            #expect(!BrowserAutomationAuthorizer.supports(bundleIdentifier: bundle))
        }
    }

    @Test("Browser Automation maps permission results without sending an event")
    func browserAutomationPermissionResults() async {
        let bundle = "com.apple.Safari"
        let authorized = await BrowserAutomationAuthorizer.request(
            bundleIdentifier: bundle, permissionCheck: { _ in noErr }
        )
        let denied = await BrowserAutomationAuthorizer.request(
            bundleIdentifier: bundle,
            permissionCheck: { _ in OSStatus(errAEEventNotPermitted) }
        )
        let unavailable = await BrowserAutomationAuthorizer.request(
            bundleIdentifier: bundle,
            permissionCheck: { _ in OSStatus(procNotFound) }
        )
        let failed = await BrowserAutomationAuthorizer.request(
            bundleIdentifier: bundle, permissionCheck: { _ in -1 }
        )

        #expect(authorized == .authorized)
        #expect(denied == .denied)
        #expect(unavailable == .targetUnavailable)
        #expect(failed == .failed(-1))
    }

    @Test("Browser Automation request has a bounded consent wait")
    func browserAutomationTimeout() async {
        let result = await BrowserAutomationAuthorizer.request(
            bundleIdentifier: "com.apple.Safari",
            timeoutNanoseconds: 1_000_000,
            permissionCheck: { _ in
                Thread.sleep(forTimeInterval: 0.05)
                return noErr
            }
        )
        #expect(result == .timedOut)
    }

    @Test("Unsupported Automation targets never reach the system API")
    func unsupportedBrowserAutomationTargetFailsClosed() async {
        let result = await BrowserAutomationAuthorizer.request(
            bundleIdentifier: "com.example.untrusted",
            permissionCheck: { _ in
                fatalError("unsupported target reached the system API")
            }
        )
        #expect(result == .targetUnavailable)
    }

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
        #expect(
            buildScript.contains(
                "if [[ \"$SKIP_SIDECAR\" != \"1\" ]]; then\n    nested_automation="
            )
        )
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
