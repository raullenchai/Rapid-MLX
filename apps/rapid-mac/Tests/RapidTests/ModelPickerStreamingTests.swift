import SwiftUI
import Testing
@testable import Rapid

@MainActor
@Suite("Model picker streaming isolation")
struct ModelPickerStreamingTests {
    @Observable
    final class Inputs {
        var alias = "first-model"
        var frame = 0
    }

    private struct Parent: View {
        @Bindable var inputs: Inputs
        let server: ServerManager
        let downloads: DownloadManager

        var body: some View {
            VStack {
                Text("Frame \(inputs.frame)")
                ModelPickerBar(
                    server: server,
                    downloads: downloads,
                    alias: $inputs.alias,
                    composerStyle: true
                )
            }
        }
    }

    private final class BuildCounter {
        var count = 0
    }

    private struct MenuParent: View {
        @Bindable var inputs: Inputs
        let counter: BuildCounter

        var body: some View {
            let selection = $inputs.alias
            VStack {
                Text("Frame \(inputs.frame)")
                ModelPickerBar.StablePickerMenuContent(key: .init(
                    catalog: [], loading: false, selectedAlias: inputs.alias,
                    showAll: false,
                    hardware: MacHardware(
                        brandString: "Test", family: .m3, tier: .pro,
                        physicalRAMBytes: 18 * 1024 * 1024 * 1024,
                        memoryBandwidthGBs: 150
                    ),
                    deleting: nil, servingAlias: nil, downloading: []
                )) {
                    counter.count += 1
                    return AnyView(VStack {
                        Text(selection.wrappedValue)
                        Button("Select second model") {
                            selection.wrappedValue = "second-model"
                        }
                        .accessibilityIdentifier("StreamingMenu.SelectSecond")
                    })
                }
                .equatable()
            }
        }
    }

    @Test("Retained menu actions write through the binding after transcript updates")
    func retainedMenuActionUpdatesSelection() async throws {
        let inputs = Inputs()
        let counter = BuildCounter()
        let stage = GoldenStage(MenuParent(inputs: inputs, counter: counter))
        try await stage.waitForIdentifier("StreamingMenu.SelectSecond")
        try await Task.sleep(for: .milliseconds(100))
        let initialBuilds = counter.count
        for frame in 1...25 {
            inputs.frame = frame
            try await stage.wait(for: "transcript update") {
                stage.treeText().contains("Frame \(frame)")
            }
        }
        #expect(counter.count <= initialBuilds + 1)
        #expect(inputs.alias == "first-model")
        try stage.press("StreamingMenu.SelectSecond")
        try await stage.wait(for: "menu action selection") {
            inputs.alias == "second-model"
                && stage.treeText().contains("second-model")
        }
    }

    @Test("Transcript frames do not rebuild menu rows, but selection changes do")
    func menuBuilderOnlyRunsForMenuChanges() async throws {
        let inputs = Inputs()
        let counter = BuildCounter()
        let stage = GoldenStage(MenuParent(inputs: inputs, counter: counter))
        try await stage.wait(for: "initial menu") {
            stage.treeText().contains("first-model")
        }
        // Let the hosting window finish its initial layout/environment pass.
        try await Task.sleep(for: .milliseconds(100))
        let initialBuilds = counter.count
        #expect(initialBuilds > 0)
        for frame in 1...25 {
            inputs.frame = frame
            try await stage.wait(for: "transcript update") {
                stage.treeText().contains("Frame \(frame)")
            }
        }
        // SwiftUI can attach the dynamic binding once more on the first
        // parent update; subsequent token frames must reuse the same rows.
        #expect(counter.count <= initialBuilds + 1)
        let buildsBeforeSelection = counter.count
        inputs.alias = "second-model"
        try await stage.wait(for: "updated menu") {
            stage.treeText().contains("second-model")
        }
        #expect(counter.count > buildsBeforeSelection)
    }

    @Test("Selection stays live across streaming parent updates")
    func selectionStillUpdates() async throws {
        let inputs = Inputs()
        let server = ServerManager(testingState: .idle)
        let stage = GoldenStage(Parent(
            inputs: inputs, server: server, downloads: DownloadManager()
        ))
        try await stage.waitForIdentifier("ModelPickerBar.ModelMenu")
        try await stage.wait(for: "initial selection") {
            stage.treeText().contains("first-model")
        }
        for frame in 1...5 {
            inputs.frame = frame
            try await stage.wait(for: "parent update") {
                stage.treeText().contains("Frame \(frame)")
            }
        }
        inputs.alias = "second-model"
        try await stage.wait(for: "changed selection") {
            stage.treeText().contains("second-model")
                && !stage.treeText().contains("first-model")
        }
        inputs.alias = ""
        try await stage.wait(for: "cleared selection") {
            stage.treeText().contains("No model chosen")
        }
    }
}
