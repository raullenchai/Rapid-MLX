import Testing
@testable import Rapid

@MainActor
@Suite("#1793 — Quickstart can start an existing cached chat model")
struct QuickstartCachedModelTests {
    private func entry(
        _ alias: String,
        cached: Bool = true,
        kind: ModelKind = .chat,
        external: Bool = false
    ) -> ModelEntry {
        ModelEntry(
            alias: alias,
            hfRepo: "mlx-community/\(alias)",
            sizeOnDisk: cached ? "2.9 GiB" : nil,
            cached: cached,
            isExternal: external,
            kind: kind
        )
    }

    @Test("only cached chat entries are eligible and lookup is not presentation-bounded")
    func filtersCachedModels() {
        let rows = (0..<8).map { entry("chat-\($0)") } + [
            entry("not-downloaded", cached: false),
            entry("image", kind: .image),
        ]
        let offered = QuickstartView.quickstartCachedModels(rows)
        #expect(offered.count == 8)
        #expect(offered.map(\.alias) == (0..<8).map { "chat-\($0)" })
        #expect(QuickstartView.canStartWithoutDownload(
            alias: "chat-7",
            cachedModels: rows
        ))
    }

    @Test("a selected cached alias takes the start-only path")
    func cachedAliasNeedsNoDownload() {
        let cached = entry("qwen3.5-4b-4bit")
        #expect(QuickstartView.canStartWithoutDownload(
            alias: cached.alias,
            cachedModels: [cached]
        ))
        #expect(!QuickstartView.canStartWithoutDownload(
            alias: "lfm2.5-1b-4bit",
            cachedModels: [cached]
        ))
    }

    @Test("external cached models retain their usable alias and honest copy")
    func externalChoice() {
        let choice = QuickstartView.choice(forCachedModel: entry("external-chat", external: true))
        #expect(choice.alias == "external-chat")
        #expect(choice.blurb.contains("another MLX app"))
    }

    @Test("cached lookup returns catalog provenance rather than the curated choice repo")
    func cachedLookupUsesCatalogEntry() {
        let cached = ModelEntry(
            alias: QuickstartCoordinator.defaultChoice.alias,
            hfRepo: "local/actual-cached-repo",
            sizeOnDisk: "600 MiB",
            cached: true
        )
        let resolved = QuickstartView.cachedModel(
            alias: QuickstartCoordinator.defaultChoice.alias,
            cachedModels: [cached]
        )
        #expect(resolved?.hfRepo == "local/actual-cached-repo")
    }

    @Test("curated trade-up keeps cached provenance past the six-row display cap")
    func cachedTradeUpPastDisplayCap() {
        let rows = (0..<6).map { entry("cached-\($0)") } + [
            entry("qwen3.5-4b-4bit")
        ]
        let shortlist = QuickstartView.shortlist(
            catalog: rows,
            selection: "qwen3.5-4b-4bit"
        )

        #expect(shortlist.cached.count == 6)
        #expect(QuickstartView.cachedModel(
            alias: "qwen3.5-4b-4bit",
            cachedModels: rows
        )?.sizeOnDisk == "2.9 GiB")
        #expect(QuickstartView.canStartWithoutDownload(
            alias: "qwen3.5-4b-4bit",
            cachedModels: rows
        ))
    }
}
