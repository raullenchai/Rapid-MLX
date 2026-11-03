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
            catalogIsEmpty: false, loading: false, runnableIsEmpty: entries.isEmpty,
            hasVisibleRows: !entries.isEmpty
        ) == .noDownloadedModels)
    }

    @Test func loadingAndUnavailableStatesRemainDistinct() {
        #expect(ModelPickerVisibility.menuState(
            catalogIsEmpty: true, loading: true, runnableIsEmpty: true, hasVisibleRows: false
        ) == .loading)
        #expect(ModelPickerVisibility.menuState(
            catalogIsEmpty: true, loading: false, runnableIsEmpty: true, hasVisibleRows: false
        ) == .unavailable)
        #expect(ModelPickerVisibility.menuState(
            catalogIsEmpty: false, loading: false, runnableIsEmpty: false, hasVisibleRows: true
        ) == .rows)
    }

    @Test func chatPickerDropsUncachedRemoteEvenWithIncludeAll() {
        for includeAll in [false, true] {
            #expect(ModelPickerVisibility.chatPickerEntries(
                [entry("remote-4b")], selectedAlias: "", servingAlias: nil,
                includeAll: includeAll, isDownloading: { _ in false }
            ).isEmpty)
        }
    }

    @Test func chatPickerCachedTinyRespectsIncludeAll() {
        let catalog = [entry("tiny-0.6b", cached: true)]
        #expect(ModelPickerVisibility.chatPickerEntries(
            catalog, selectedAlias: "", servingAlias: nil,
            includeAll: false, isDownloading: { _ in false }
        ).isEmpty)
        #expect(ModelPickerVisibility.chatPickerEntries(
            catalog, selectedAlias: "", servingAlias: nil,
            includeAll: true, isDownloading: { _ in false }
        ).map(\.alias) == ["tiny-0.6b"])
    }

    @Test func chatPickerKeepsSelectedUncachedTiny() {
        #expect(ModelPickerVisibility.chatPickerEntries(
            [entry("selected-0.6b")], selectedAlias: "selected-0.6b", servingAlias: nil,
            includeAll: false, isDownloading: { _ in false }
        ).map(\.alias) == ["selected-0.6b"])
    }

    @Test func chatPickerKeepsServingUncachedTiny() {
        #expect(ModelPickerVisibility.chatPickerEntries(
            [entry("serving-0.6b")], selectedAlias: "other-4b", servingAlias: "serving-0.6b",
            includeAll: false, isDownloading: { _ in false }
        ).map(\.alias) == ["serving-0.6b"])
    }

    @Test func chatPickerDenylistOverridesSizeExemptions() {
        let broken = "ministral-3b-4bit"
        for includeAll in [false, true] {
            for (selected, serving) in [("", nil), (broken, nil), ("", Optional(broken))] {
                #expect(ModelPickerVisibility.chatPickerEntries(
                    [entry(broken, cached: true)], selectedAlias: selected, servingAlias: serving,
                    includeAll: includeAll, isDownloading: { _ in false }
                ).isEmpty)
            }
        }
    }

    @Test func downloadedButFilteredShowsNoVisibleModels() {
        for alias in ["tiny-0.6b", "ministral-3b-4bit"] {
            let catalog = [entry(alias, cached: true)]
            let runnableEntries = runnable(catalog)
            let visible = ModelPickerVisibility.chatPickerEntries(
                catalog, selectedAlias: "", servingAlias: nil,
                includeAll: false, isDownloading: { _ in false }
            )
            #expect(!runnableEntries.isEmpty)
            #expect(visible.isEmpty)
            #expect(ModelPickerVisibility.menuState(
                catalogIsEmpty: catalog.isEmpty, loading: false,
                runnableIsEmpty: runnableEntries.isEmpty, hasVisibleRows: !visible.isEmpty
            ) == .noVisibleModels)
        }
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
