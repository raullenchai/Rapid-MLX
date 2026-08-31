import AppKit
import SwiftUI
import Testing

/// Contract tests for ``GoldenStage``: the stage must expose a walkable AX
/// tree, deliver presses into real SwiftUI action closures, surface state
/// changes back through the tree, and materialize sheet and popover
/// windows — all without the window ever being visible or key. Sunk golden
/// flows depend on exactly these guarantees, so a macOS update that breaks
/// one should fail here, in a test that names the broken ingredient,
/// rather than inside a ported product journey.
@MainActor
private final class StageProbeModel: ObservableObject {
    @Published var label = "before-press"
    @Published var showsPopover = false
    @Published var showsSheet = false
    var pressed = false
}

private struct StageProbeView: View {
    @ObservedObject var model: StageProbeModel

    var body: some View {
        VStack {
            Text(model.label)
                .accessibilityIdentifier("StageProbe.Label")
            Button("Press Me") {
                model.pressed = true
                model.label = "after-press"
            }
            .accessibilityIdentifier("StageProbe.Button")
            Button("Popover") { model.showsPopover.toggle() }
                .accessibilityIdentifier("StageProbe.PopoverAnchor")
                .popover(isPresented: $model.showsPopover) {
                    Text("popover-content")
                        .accessibilityIdentifier("StageProbe.PopoverContent")
                        .padding()
                }
            Button("Sheet") { model.showsSheet = true }
                .accessibilityIdentifier("StageProbe.SheetAnchor")
                .sheet(isPresented: $model.showsSheet) {
                    Button("Done") { model.showsSheet = false }
                        .accessibilityIdentifier("StageProbe.SheetDone")
                        .padding()
                }
            TextField("Draft", text: .constant(""))
                .accessibilityIdentifier("StageProbe.Field")
        }
        .frame(width: 400, height: 300)
    }
}

@MainActor
@Suite("GoldenStage contract", .serialized)
struct InProcessAXSpikeTests {
    @Test("The stage walks identifiers, presses, and observes state changes")
    func walkPressObserve() async throws {
        let model = StageProbeModel()
        let stage = GoldenStage(StageProbeView(model: model), size: CGSize(width: 400, height: 300))

        let ids = stage.identifiers()
        #expect(ids.contains("StageProbe.Label"), "AX tree missing label: \(ids)")
        #expect(ids.contains("StageProbe.Button"), "AX tree missing button: \(ids)")
        #expect(stage.treeText().contains("before-press"))

        try stage.press("StageProbe.Button")
        #expect(model.pressed, "press did not reach the SwiftUI action closure")
        try await stage.waitForText("after-press")
    }

    @Test("Sheets materialize as walkable windows and dismiss on press")
    func sheetRoundTrip() async throws {
        let model = StageProbeModel()
        let stage = GoldenStage(StageProbeView(model: model), size: CGSize(width: 400, height: 300))

        try stage.press("StageProbe.SheetAnchor")
        try await stage.waitForIdentifier("StageProbe.SheetDone")
        try stage.press("StageProbe.SheetDone")
        try await stage.waitForIdentifierGone("StageProbe.SheetDone")
        #expect(!model.showsSheet)
    }

    @Test("Popovers materialize because the stage window is ordered in")
    func popoverRoundTrip() async throws {
        let model = StageProbeModel()
        let stage = GoldenStage(StageProbeView(model: model), size: CGSize(width: 400, height: 300))

        try stage.press("StageProbe.PopoverAnchor")
        try await stage.waitForIdentifier("StageProbe.PopoverContent")
        try stage.press("StageProbe.PopoverAnchor")
        try await stage.waitForIdentifierGone("StageProbe.PopoverContent")
    }

    @Test("setValue drives text fields the way the AX driver's set-value does")
    func setValueOnField() async throws {
        let model = StageProbeModel()
        let stage = GoldenStage(StageProbeView(model: model), size: CGSize(width: 400, height: 300))

        try stage.setValue("typed via AX", for: "StageProbe.Field")
        try await stage.wait(for: "field value") {
            stage.value(of: "StageProbe.Field") == "typed via AX"
        }
    }
}
