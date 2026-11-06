import Foundation
import Testing
@testable import Rapid

@Suite("Golden fixture cache identities")
@MainActor
struct GoldenFixtureCatalogTests {
    private func output(_ arguments: [String], variants: Bool = false, curated: Bool = false) async throws -> String {
        let script = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("scripts/fake-rapid-mlx.sh")
        var environment = ProcessInfo.processInfo.environment.filter { !$0.key.hasPrefix("FAKE_") }
        environment["FAKE_CACHED_VARIANTS"] = variants ? "1" : "0"
        environment["FAKE_CACHED_CURATED_TRADEUP"] = curated ? "1" : "0"
        let process = try await TestSubprocess.run(
            executableURL: script, arguments: arguments, environment: environment
        )
        #expect(process.terminationStatus == 0)
        return String(decoding: process.standardOutput, as: UTF8.self)
    }

    private func entries(variants: Bool = false) async throws -> [ModelEntry] {
        let availableOutput = try await output(["models", "--json"], variants: variants)
        let cachedOutput = try await output(["ls"], variants: variants)
        let catalog = try #require(ModelCatalog.parseAvailableJSON(availableOutput))
        let cached = ModelCatalog.parseCached(cachedOutput)
        return ModelCatalog.mergeAvailableAndCached(
            available: catalog.entries, cached: cached, excluded: catalog.excluded
        )
    }

    @Test("Cached curated fixture retains the LFM starter past the six-row cap")
    func cachedStarter() async throws {
        let available = try #require(ModelCatalog.parseAvailableJSON(
            try await output(["models", "--json"], curated: true)
        ))
        let cached = ModelCatalog.parseCached(try await output(["ls"], curated: true))
        let models = ModelCatalog.mergeAvailableAndCached(
            available: available.entries, cached: cached, excluded: available.excluded
        )
        let starter = QuickstartCoordinator.defaultChoice.alias
        #expect(models.contains { $0.alias == starter && $0.cached })
        let machine = MacHardware(
            brandString: "Apple M1", family: .m1, tier: .base,
            physicalRAMBytes: 16 * (1 << 30), memoryBandwidthGBs: 100
        )
        #expect(QuickstartCoordinator.defaultChoice(hardware: machine, catalog: models).alias == starter)
        let list = QuickstartView.shortlist(catalog: models, selection: starter, physicalRAMGB: 16)
        #expect(list.visibleAliases.contains(starter))
        #expect(list.onThisMacCount > 6)
        #expect(available.profiles[starter]?.isTextOnly == true)
        #expect(available.profiles["qwen3.5-4b-4bit"]?.isTextOnly == false)
    }

    @Test("External golden model remains cached, selectable and read-only")
    func externalIdentity() async throws {
        let models = try await entries()
        let external = try #require(models.first { $0.alias == "fake-external-alias" })
        #expect(external.cached)
        #expect(external.isExternal)
        #expect(external.hfRepo == "fake-external-alias")
        #expect(ModelSelectionPurpose.chat.accepts(external))
        #expect(QuickstartView.quickstartCachedModels(models).contains(external))
        let info = try await output(["info", external.alias])
        #expect(info.contains("-> fake-external-alias"))
        let managed = try #require(models.first { $0.alias == "fake-alias" })
        #expect(managed.cached && !managed.isExternal)
        #expect(managed.hfRepo != external.hfRepo)
    }

    @Test("Variant golden fixture keeps the distinct size behind disclosure")
    func variantDisclosure() async throws {
        let models = try await entries(variants: true)
        let presentation = QuickstartView.compactCachedPresentation(
            QuickstartView.quickstartCachedPresentation(models, limit: 6),
            pinnedAliases: ["qwen3-0.6b-4bit"]
        )
        #expect(presentation.primary.map(\.alias) == ["qwen3-0.6b-4bit"])
        #expect(presentation.alternates.contains { $0.alias == "qwen3-0.6b-8bit" })
        #expect(presentation.alternates.contains { $0.alias == "qwen3-4b-4bit" })
    }
}
