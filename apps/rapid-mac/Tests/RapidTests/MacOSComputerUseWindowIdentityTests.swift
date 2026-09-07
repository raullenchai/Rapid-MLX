import CoreGraphics
import Testing
@testable import Rapid

@Suite("Computer Use window identity")
struct MacOSComputerUseWindowIdentityTests {
    private let display = CGRect(x: 0, y: 0, width: 1920, height: 1080)
    private let point = CGPoint(x: 1493, y: 782)
    private let dockPID: pid_t = 744

    @Test("A trusted display-sized Dock management window is not an occluder")
    func skipsDockBackdrop() {
        #expect(topmost([
            dock(frame: display),
            app(),
        ]) == "calculator")
    }

    @Test("A process merely named Dock remains an occluder")
    func namedDockIsNotTrusted() {
        #expect(topmost([
            dock(frame: display),
            app(),
        ], trustedDockPIDs: []) == "dock")
    }

    @Test("Visible Dock UI smaller than a display remains an occluder")
    func visibleDockRemainsOccluder() {
        #expect(topmost([
            dock(frame: CGRect(x: 1400, y: 740, width: 500, height: 100)),
            app(),
        ]) == "dock")
    }

    @Test("A display lookup failure preserves fail-closed occlusion")
    func missingDisplayIdentityRemainsOccluder() {
        #expect(MacOSComputerUseWindowIdentity.topmostWindowIdentifier(
            at: point,
            records: [dock(frame: display), app()],
            activeDisplayFrames: [],
            trustedDockPIDs: [dockPID]
        ) == "dock")
    }

    @Test("A Dock backdrop may match any active display")
    func matchesSecondaryDisplay() {
        let secondaryDisplay = CGRect(x: 1920, y: 0, width: 1920, height: 1080)
        let secondaryPoint = CGPoint(x: 2000, y: 500)
        let secondaryApp = record(
            identifier: "secondary-app",
            ownerPID: 200,
            ownerName: "Calculator",
            name: "Calculator",
            layer: 0,
            frame: CGRect(x: 1950, y: 400, width: 300, height: 300)
        )
        #expect(MacOSComputerUseWindowIdentity.topmostWindowIdentifier(
            at: secondaryPoint,
            records: [dock(frame: secondaryDisplay), secondaryApp],
            activeDisplayFrames: [display, secondaryDisplay],
            trustedDockPIDs: [dockPID]
        ) == "secondary-app")
    }

    @Test("Wrong-name and wrong-layer Dock windows remain occluders")
    func onlyExactBackdropShapeIsSkipped() {
        let wrongName = record(
            identifier: "wrong-name",
            ownerPID: dockPID,
            ownerName: "Dock",
            name: "Item-0",
            layer: 20,
            frame: display
        )
        let wrongLayer = record(
            identifier: "wrong-layer",
            ownerPID: dockPID,
            ownerName: "Dock",
            name: "Dock",
            layer: 21,
            frame: display
        )
        #expect(topmost([wrongName, app()]) == "wrong-name")
        #expect(topmost([wrongLayer, app()]) == "wrong-layer")
    }

    @Test("An ordinary overlay still blocks the selected window")
    func ordinaryOverlayRemainsOccluder() {
        #expect(topmost([
            dock(frame: display),
            record(
                identifier: "menu",
                ownerPID: 100,
                ownerName: "Other",
                name: "Menu",
                layer: 25,
                frame: CGRect(x: 1480, y: 770, width: 60, height: 60)
            ),
            app(),
        ]) == "menu")
    }

    @Test("Transparent records are ignored without weakening later occlusion")
    func transparentRecordIsIgnored() {
        #expect(topmost([
            record(
                identifier: "transparent",
                ownerPID: 100,
                ownerName: "Other",
                name: nil,
                layer: 30,
                frame: display,
                alpha: 0
            ),
            app(),
        ]) == "calculator")
    }

    private func topmost(
        _ records: [MacOSComputerUseWindowIdentity.WindowRecord],
        trustedDockPIDs: Set<pid_t>? = nil
    ) -> String? {
        MacOSComputerUseWindowIdentity.topmostWindowIdentifier(
            at: point,
            records: records,
            activeDisplayFrames: [display],
            trustedDockPIDs: trustedDockPIDs ?? [dockPID]
        )
    }

    private func dock(frame: CGRect) -> MacOSComputerUseWindowIdentity.WindowRecord {
        record(
            identifier: "dock",
            ownerPID: dockPID,
            ownerName: "Dock",
            name: "Dock",
            layer: 20,
            frame: frame
        )
    }

    private func app() -> MacOSComputerUseWindowIdentity.WindowRecord {
        record(
            identifier: "calculator",
            ownerPID: 200,
            ownerName: "Calculator",
            name: "Calculator",
            layer: 0,
            frame: CGRect(x: 1462, y: 580, width: 230, height: 408)
        )
    }

    private func record(
        identifier: String,
        ownerPID: pid_t,
        ownerName: String?,
        name: String?,
        layer: Int,
        frame: CGRect,
        alpha: Double = 1
    ) -> MacOSComputerUseWindowIdentity.WindowRecord {
        MacOSComputerUseWindowIdentity.WindowRecord(
            identifier: identifier,
            ownerPID: ownerPID,
            ownerName: ownerName,
            name: name,
            layer: layer,
            frame: frame,
            alpha: alpha
        )
    }
}
