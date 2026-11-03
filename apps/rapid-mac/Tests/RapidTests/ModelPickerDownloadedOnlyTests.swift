import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("ModelPicker downloaded-only chat menu")
struct ModelPickerDownloadedOnlyTests {
    private func entry(_ alias: String, cached: Bool = false) -> ModelEntry {
        ModelEntry(alias: alias, hfRepo: nil, sizeOnDisk: nil, cached: cached)
    }

    private func runnable(
        _ entries: [ModelEntry], selected: String = "", serving: String? = nil,
        downloading: Set<String> = []
    ) -> [ModelEntry] {
        ModelPickerVisibility.runnableEntries(
            entries, selectedAlias: selected, servingAlias: serving,
            isDownloading: { downloading.contains($0) }
        )
    }

    @Test func cachedKept() {
        #expect(runnable([entry("installed-4b", cached: true)]).map(\.alias) == ["installed-4b"])
    }

    @Test func uncachedAndInactivePartialDropped() {
        var partial = entry("partial-4b")
        partial.sizeBytes = 1024
        #expect(runnable([entry("remote-4b"), partial]).isEmpty)
    }

    @Test func uncachedSelectedKept() {
        #expect(runnable([entry("selected-4b")], selected: "selected-4b").map(\.alias) == ["selected-4b"])
    }

    @Test func uncachedServingKept() {
        #expect(runnable([entry("serving-4b")], serving: "serving-4b").map(\.alias) == ["serving-4b"])
    }

    @Test func localAndBYOMUseCatalogInstalledState() {
        let entries = ModelCatalog.mergeAvailableAndCached(
            available: [("my-local-model", "/models/custom")],
            cached: [("my-local-model", "/models/custom", "1 GiB"),
                     ("(external)", "local-external-model", "2 GiB")],
            excluded: []
        )
        #expect(entries.count == 2)
        #expect(entries.allSatisfy { $0.cached })
        #expect(runnable(entries).map(\.alias) == entries.map(\.alias))
    }

    @Test func customLaunchPathUsesExistingInstalledState() {
        let cached: [CachedModelInventoryEntry] = [
            (alias: "(external)", hfRepo: "local-model", subfolder: nil,
             size: .bytes(1024), discovery: nil, launchPath: "/models/local-model")
        ]
        let catalog = ModelCatalog.mergeAtomicAndCached(atomic: [], cached: cached, excluded: [])
        let entries = runnable(catalog)
        #expect(entries.contains { $0.alias == "/models/local-model" && $0.cached })
    }

    @Test func downloadingVisibleButNotSelectable() {
        let entries = runnable([entry("loading-4b")], downloading: ["loading-4b"])
        #expect(entries.map(\.alias) == ["loading-4b"])
        let row = ModelPickerVisibility.rowPresentation(
            title: "Loading model", accessibilityLabel: "Loading model, not downloaded",
            isDownloading: true
        )
        #expect(row.title == "Loading model · " + String(localized: "Downloading…"))
        #expect(row.accessibilityLabel.hasSuffix(String(localized: "Downloading…")))
        #expect(!row.isSelectable)
    }

    @Test func installedRowPreservesTitleAndVoiceOver() {
        let row = ModelPickerVisibility.rowPresentation(
            title: "Installed model", accessibilityLabel: "Installed model, downloaded",
            isDownloading: false
        )
        #expect(row.title == "Installed model")
        #expect(row.accessibilityLabel == "Installed model, downloaded")
        #expect(row.isSelectable)
    }

    @Test func allUncachedShowsDownloadedEmptyState() {
        let entries = runnable([entry("remote-4b"), entry("remote-8b")])
        #expect(entries.isEmpty)
        #expect(ModelPickerVisibility.menuState(
            catalogIsEmpty: false, loading: false, hasVisibleRows: !entries.isEmpty
        ) == .noDownloadedModels)
    }

    @Test func loadingAndUnavailableStatesRemainDistinct() {
        #expect(ModelPickerVisibility.menuState(
            catalogIsEmpty: true, loading: true, hasVisibleRows: false
        ) == .loading)
        #expect(ModelPickerVisibility.menuState(
            catalogIsEmpty: true, loading: false, hasVisibleRows: false
        ) == .unavailable)
        #expect(ModelPickerVisibility.menuState(
            catalogIsEmpty: false, loading: false, hasVisibleRows: true
        ) == .rows)
    }

    @Test func runnableFilterPrecedesSizeAndDenylist() {
        let entries = runnable([
            entry("remote-4b"), entry("tiny-0.6b", cached: true),
            entry("selected-0.6b"), entry("ministral-3b-4bit", cached: true)
        ], selected: "selected-0.6b")
        let visible = ModelPickerVisibility.filter(entries, selectedAlias: "selected-0.6b", includeAll: false)
        #expect(visible.map(\.alias) == ["selected-0.6b"])
    }

    @Test func browseAllModelsRoutesBeforeOpeningOnce() {
        let router = SettingsRouter()
        var opens = 0
        ModelPickerBar.browseAllModels(router: router) {
            #expect(router.requestedCategory == .modelManagement)
            opens += 1
        }
        #expect(router.requestedCategory == .modelManagement)
        #expect(opens == 1)
    }

    @Test func browseButtonUsesModelManagementAction() async throws {
        let router = SettingsRouter()
        var opens = 0
        let stage = GoldenStage(ModelPickerBrowseAllModelsButton(router: router) {
            #expect(router.requestedCategory == .modelManagement)
            opens += 1
        })
        try await stage.waitForIdentifier("ModelPickerBar.BrowseAllModels")
        #expect(stage.treeText().contains("Browse all models…"))
        try stage.press("ModelPickerBar.BrowseAllModels")
        #expect(router.requestedCategory == .modelManagement)
        #expect(opens == 1)
    }
}
