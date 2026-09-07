import Testing
@testable import Rapid

@Suite("Mac automation permissions")
struct MacAutomationPermissionsTests {
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
}
