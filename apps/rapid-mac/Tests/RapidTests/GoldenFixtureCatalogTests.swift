import Foundation
import Testing
@testable import Rapid

@Suite("Golden fixture cache identities")
@MainActor
struct GoldenFixtureCatalogTests {
    private func output(_ arguments: [String], variants: Bool = false) throws -> String {
        let script = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("scripts/fake-rapid-mlx.sh")
        let process = Process()
        process.executableURL = script
        process.arguments = arguments
        var environment = ProcessInfo.processInfo.environment.filter { !$0.key.hasPrefix("FAKE_") }
        environment["FAKE_CACHED_VARIANTS"] = variants ? "1" : "0"
        process.environment = environment
        let pipe = Pipe()
        process.standardOutput = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        #expect(process.terminationStatus == 0)
        return String(decoding: data, as: UTF8.self)
    }

    private func entries(variants: Bool = false) throws -> [ModelEntry] {
        let catalog = try #require(ModelCatalog.parseAvailableJSON(output(["models", "--json"], variants: variants)))
        let cached = ModelCatalog.parseCached(try output(["ls"], variants: variants))
        return ModelCatalog.mergeAvailableAndCached(
            available: catalog.entries, cached: cached, excluded: catalog.excluded
        )
    }

    @Test("External golden model remains cached, selectable and read-only")
    func externalIdentity() throws {
        let models = try entries()
        let external = try #require(models.first { $0.alias == "fake-external-alias" })
        #expect(external.cached)
        #expect(external.isExternal)
        #expect(external.hfRepo == "fake-external-alias")
        #expect(ModelSelectionPurpose.chat.accepts(external))
        #expect(QuickstartView.quickstartCachedModels(models).contains(external))
        #expect(try output(["info", external.alias]).contains("-> fake-external-alias"))
        let managed = try #require(models.first { $0.alias == "fake-alias" })
        #expect(managed.cached && !managed.isExternal)
        #expect(managed.hfRepo != external.hfRepo)
    }

    @Test("Variant golden fixture keeps the distinct size behind disclosure")
    func variantDisclosure() throws {
        let models = try entries(variants: true)
        let presentation = QuickstartView.compactCachedPresentation(
            QuickstartView.quickstartCachedPresentation(models, limit: 6),
            pinnedAliases: ["qwen3-0.6b-4bit"]
        )
        #expect(presentation.primary.map(\.alias) == ["qwen3-0.6b-4bit"])
        #expect(presentation.alternates.contains { $0.alias == "qwen3-0.6b-8bit" })
        #expect(presentation.alternates.contains { $0.alias == "qwen3-4b-4bit" })
    }
}
