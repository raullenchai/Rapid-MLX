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

    @Test("first-run cached list collapses quant siblings but not model sizes")
    func collapsesOnlyTrueVariantSiblings() {
        let rows = [
            entry("qwen3-0.6b-8bit"),
            entry("qwen3-0.6b-4bit"),
            entry("qwen3-4b-4bit"),
            entry("llama3-3b-8bit"),
        ]

        let presentation = QuickstartView.quickstartCachedPresentation(rows, limit: 6)

        #expect(presentation.primary.map(\.alias) == [
            "qwen3-0.6b-4bit", "qwen3-4b-4bit", "llama3-3b-8bit",
        ])
        #expect(presentation.alternates.map(\.alias) == ["qwen3-0.6b-8bit"])
    }

    @Test("unknown cached families remain independently visible")
    func unknownFamiliesDoNotCollapse() {
        let rows = [entry("custom-7b-4bit"), entry("custom-7b-8bit")]
        let presentation = QuickstartView.quickstartCachedPresentation(rows, limit: 6)
        #expect(presentation.primary.map(\.alias) == rows.map(\.alias))
        #expect(presentation.alternates.isEmpty)
    }

    @Test("same-family same-size semantic variants remain separate decisions")
    func semanticVariantsDoNotCollapse() {
        let rows = [
            entry("qwen3-4b-instruct-2507-4bit"),
            entry("qwen3-4b-thinking-2507-8bit"),
        ]
        let presentation = QuickstartView.quickstartCachedPresentation(rows, limit: 6)
        #expect(presentation.primary.map(\.alias) == rows.map(\.alias))
        #expect(presentation.alternates.isEmpty)
    }

    @Test("presentation cap counts decisions, not hidden quant siblings")
    func capCountsDistinctModelDecisions() {
        let rows = [
            entry("qwen3-0.6b-8bit"),
            entry("qwen3-0.6b-4bit"),
            entry("qwen3-4b-4bit"),
        ]
        let presentation = QuickstartView.quickstartCachedPresentation(rows, limit: 2)
        #expect(presentation.primary.map(\.alias) == ["qwen3-0.6b-4bit", "qwen3-4b-4bit"])
        #expect(presentation.alternates.map(\.alias) == ["qwen3-0.6b-8bit"])
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
            selection: "qwen3.5-4b-4bit",
            physicalRAMGB: 16
        )

        // #4857: past the compaction threshold only the pinned selection
        // stays in the visible cached rows; the rest sit one disclosure away.
        #expect(shortlist.cached.map(\.alias) == ["qwen3.5-4b-4bit"])
        #expect(shortlist.cachedDecisionCount == 6)
        // The five unpinned bounded rows plus the one the selection displaced.
        #expect(shortlist.cachedAlternates.map(\.alias)
                == (0..<6).map { "cached-\($0)" })
        #expect(shortlist.visibleAliases.contains("qwen3.5-4b-4bit"))
        #expect(QuickstartView.cachedModel(
            alias: "qwen3.5-4b-4bit",
            cachedModels: rows
        )?.sizeOnDisk == "2.9 GiB")
        #expect(QuickstartView.canStartWithoutDownload(
            alias: "qwen3.5-4b-4bit",
            cachedModels: rows
        ))
    }

    // MARK: - #4857 clearer cached rows

    @Test("a cached authored alias reads with its catalogue name, alias kept as identity")
    func cachedRowUsesCatalogueName() {
        let authored = QuickstartCoordinator.defaultChoice
        let choice = QuickstartView.choice(forCachedModel: entry(authored.alias))
        #expect(choice.alias == authored.alias)
        #expect(choice.displayName == authored.displayName)
        #expect(choice.displayName != authored.alias)
        #expect(QuickstartView.cachedRowSecondaryText(for: choice) == authored.alias)
    }

    @Test("an uncatalogued cached alias keeps the alias and no duplicate caption")
    func uncataloguedCachedRowKeepsAlias() {
        let choice = QuickstartView.choice(forCachedModel: entry("bonsai-8b-2bit"))
        #expect(choice.displayName == "bonsai-8b-2bit")
        #expect(QuickstartView.cachedRowSecondaryText(for: choice) == nil)
    }

    @Test("compaction leaves a short cached list untouched")
    func compactionBelowThresholdIsIdentity() {
        let presentation = QuickstartView.CachedPresentation(
            primary: (0..<3).map { entry("chat-\($0)") },
            alternates: [entry("chat-0-8bit")]
        )
        #expect(QuickstartView.compactCachedPresentation(
            presentation, pinnedAliases: []
        ) == presentation)
    }

    @Test("compaction keeps pinned rows in order and demotes the rest ahead of siblings")
    func compactionKeepsPinnedRows() {
        let presentation = QuickstartView.CachedPresentation(
            primary: (0..<5).map { entry("chat-\($0)") },
            alternates: [entry("chat-0-8bit")]
        )
        let compact = QuickstartView.compactCachedPresentation(
            presentation, pinnedAliases: ["chat-3", "chat-1", "not-cached"]
        )
        #expect(compact.primary.map(\.alias) == ["chat-1", "chat-3"])
        #expect(compact.alternates.map(\.alias) == [
            "chat-0", "chat-2", "chat-4", "chat-0-8bit",
        ])
    }

    @Test("compaction with nothing pinned keeps the first row as the anchor")
    func compactionFallsBackToFirstRow() {
        let presentation = QuickstartView.CachedPresentation(
            primary: (0..<4).map { entry("chat-\($0)") },
            alternates: []
        )
        let compact = QuickstartView.compactCachedPresentation(
            presentation, pinnedAliases: ["elsewhere"]
        )
        #expect(compact.primary.map(\.alias) == ["chat-0"])
        #expect(compact.alternates.map(\.alias) == ["chat-1", "chat-2", "chat-3"])
    }

    @Test("a cache-rich shortlist keeps the selected cached model visible")
    func cacheRichShortlistKeepsSelectionVisible() {
        let rows = (0..<5).map { entry("chat-\($0)") } + [
            entry("qwen3.5-4b-4bit")
        ]
        let shortlist = QuickstartView.shortlist(
            catalog: rows,
            selection: "qwen3.5-4b-4bit",
            physicalRAMGB: 16
        )
        #expect(shortlist.cached.map(\.alias) == ["qwen3.5-4b-4bit"])
        #expect(shortlist.cachedAlternates.map(\.alias)
                == rows.prefix(5).map(\.alias))
        #expect(shortlist.cachedDecisionCount == 6)
        #expect(shortlist.visibleAliases.contains("qwen3.5-4b-4bit"))
        #expect(shortlist.yourPick == nil)
        // The headline still counts every cached decision, not the compact row.
        #expect(QuickstartView.selectionNarrative(
            alias: "qwen3.5-4b-4bit",
            cachedModels: rows,
            comparableTradeUps: shortlist.tradeUps,
            cachedRowCount: shortlist.cachedDecisionCount
        ) == .alreadyHere(cachedCount: 6))
    }

    @Test("a compacted authored model renders once, in its own lane")
    func compactedAuthoredRowIsNotDuplicated() {
        let rows = [entry("chat-a"), entry("chat-b"), entry("qwen3.5-9b-4bit"),
                    entry("qwen3.5-4b-4bit")]
        let shortlist = QuickstartView.shortlist(
            catalog: rows,
            selection: "qwen3.5-4b-4bit",
            physicalRAMGB: 16
        )
        #expect(shortlist.cached.map(\.alias) == ["qwen3.5-4b-4bit"])
        #expect(shortlist.tradeUps.map(\.alias).contains("qwen3.5-9b-4bit"))
        #expect(!shortlist.cachedAlternates.map(\.alias).contains("qwen3.5-9b-4bit"))
        #expect(shortlist.cachedAlternates.map(\.alias) == ["chat-a", "chat-b"])
        let all = shortlist.visibleAliases(includeCachedAlternates: true)
        #expect(all.filter { $0 == "qwen3.5-9b-4bit" }.count == 1)
    }

    @Test("a short cached list is not compacted")
    func shortCachedListStaysExpanded() {
        let rows = [entry("chat-a"), entry("chat-b"), entry("qwen3.5-4b-4bit")]
        let shortlist = QuickstartView.shortlist(
            catalog: rows,
            selection: "qwen3.5-4b-4bit",
            physicalRAMGB: 16
        )
        #expect(shortlist.cached.map(\.alias) == rows.map(\.alias))
        #expect(shortlist.cachedAlternates.isEmpty)
    }
}
