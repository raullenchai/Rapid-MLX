import Foundation
import Testing
@testable import Rapid

/// Truth tables behind the Models-surface redesign (issue #507): brand
/// classification (``ModelBrandStyle``), the two compact benchmark
/// meters (``ModelMeter``), and favorite ordering (``ModelFavorites``).
///
/// The view layer (``BrandIcon`` / ``SegmentedBenchMeter`` /
/// ``SettingsModelManagementPanel``) is a thin shell over these pure
/// helpers, mirroring the ``ModelCacheActions`` split — so every branch
/// is pinned here without standing up a SwiftUI host.
@Suite("Models surface redesign — pure logic (#507)")
struct ModelSurfaceRedesignTests {

    // MARK: - ModelBrandStyle.brand

    @Test("brand: each recognised family maps to its case")
    func brandFamilies() {
        #expect(ModelBrandStyle.brand(forAlias: "qwen3.6-35b-4bit") == .qwen)
        #expect(ModelBrandStyle.brand(forAlias: "qwq-32b-8bit") == .qwen)
        #expect(ModelBrandStyle.brand(forAlias: "gpt-oss-20b-4bit") == .gptOss)
        #expect(ModelBrandStyle.brand(forAlias: "gptoss-120b") == .gptOss)
        #expect(ModelBrandStyle.brand(forAlias: "llama-3.3-70b-4bit") == .llama)
        #expect(ModelBrandStyle.brand(forAlias: "gemma-3-27b-8bit") == .gemma)
        #expect(ModelBrandStyle.brand(forAlias: "deepseek-v3-4bit") == .deepseek)
        #expect(ModelBrandStyle.brand(forAlias: "phi-4-mini-4bit") == .phi)
        #expect(ModelBrandStyle.brand(forAlias: "glm-4.6-9b-4bit") == .glm)
        #expect(ModelBrandStyle.brand(forAlias: "smollm-1.7b") == .smollm)
        #expect(ModelBrandStyle.brand(forAlias: "hermes-3-8b") == .hermes)
    }

    @Test("brand: the whole Mistral house resolves to .mistral")
    func brandMistralHouse() {
        for alias in [
            "mistral-small-24b-4bit",
            "devstral-24b-4bit",
            "ministral-8b-4bit",
            "magistral-24b-4bit",
            "codestral-22b-4bit",
        ] {
            #expect(ModelBrandStyle.brand(forAlias: alias) == .mistral)
        }
    }

    @Test("brand: an unlisted family falls back to .other")
    func brandOther() {
        #expect(ModelBrandStyle.brand(forAlias: "bonsai-2b-4bit") == .other)
        #expect(ModelBrandStyle.brand(forAlias: "some-unknown-model") == .other)
    }

    // MARK: - ModelBrandStyle.monogram

    @Test("monogram: recognised brands use their fixed 2-letter mark")
    func monogramRecognised() {
        #expect(ModelBrandStyle.monogram(forAlias: "qwen3.6-35b-4bit") == "Qw")
        #expect(ModelBrandStyle.monogram(forAlias: "gpt-oss-20b-4bit") == "GO")
        #expect(ModelBrandStyle.monogram(forAlias: "deepseek-v3-4bit") == "DS")
    }

    @Test("monogram: .other derives the first two letters, upper-cased")
    func monogramOtherFallback() {
        #expect(ModelBrandStyle.monogram(forAlias: "bonsai-2b-4bit") == "BO")
        // Leading non-letters are skipped so a numeric prefix still
        // yields a legible mark.
        #expect(ModelBrandStyle.monogram(forAlias: "2x-model") == "XM")
    }

    // MARK: - ModelBrandStyle.modelType

    @Test("modelType: -vl aliases are vision, the rest are chat")
    func modelTypeVision() {
        #expect(ModelBrandStyle.modelType(forAlias: "qwen3-vl-30b-4bit") == .vision)
        #expect(ModelBrandStyle.modelType(forAlias: "some-model-vl") == .vision)
        #expect(ModelBrandStyle.modelType(forAlias: "qwen3.6-35b-4bit") == .chat)
        #expect(ModelBrandStyle.modelType(forAlias: "gpt-oss-20b-4bit") == .chat)
    }

    // MARK: - ModelBrandStyle.displayFamily

    @Test("displayFamily: overrides the two aliases ModelInfoCatalog calls Unknown")
    func displayFamilyOverrides() {
        // gpt-oss + devstral aren't substrings of their family names, so
        // ModelInfoCatalog returns "Unknown" — the brand layer fixes both.
        #expect(ModelBrandStyle.displayFamily(forAlias: "gpt-oss-20b-4bit") == "GPT-OSS")
        #expect(ModelBrandStyle.displayFamily(forAlias: "devstral-24b-4bit") == "Devstral")
    }

    @Test("displayFamily: an entirely unknown alias reads 'Model', never 'Unknown'")
    func displayFamilyUnknownIsHumane() {
        let family = ModelBrandStyle.displayFamily(forAlias: "some-unknown-xyz-model")
        #expect(family != "Unknown")
        #expect(family == "Model")
    }

    // MARK: - ModelMeter.segments

    @Test("segments: 0 → 0 blocks, normalizer → all blocks, clamped above")
    func meterSegmentsBoundaries() {
        let axis = BenchScores.Axis.generalReasoning
        let norm = axis.thresholds.normalizer
        #expect(ModelMeter.segments(value: 0, axis: axis) == 0)
        #expect(ModelMeter.segments(value: norm, axis: axis) == ModelMeter.segmentCount)
        // Above the normalizer stays clamped at the full count.
        #expect(ModelMeter.segments(value: norm * 2, axis: axis) == ModelMeter.segmentCount)
        // Negatives clamp to zero rather than going negative.
        #expect(ModelMeter.segments(value: -10, axis: axis) == 0)
    }

    @Test("segments: a mid value rounds to the nearest block")
    func meterSegmentsRounds() {
        // Speed normalizer is 300 t/s; 180 → 3.0 blocks exactly.
        #expect(ModelMeter.segments(value: 180, axis: .speed) == 3)
        // 150/300 = 0.5 → 2.5 → rounds to 3 (banker's-free `.rounded()`).
        #expect(ModelMeter.segments(value: 150, axis: .speed) == 3)
    }

    // MARK: - ModelMeter.level

    @Test("level: great / good / low honour each axis's thresholds")
    func meterLevels() {
        let axis = BenchScores.Axis.generalReasoning // (good: 50, great: 75)
        #expect(ModelMeter.level(value: 80, axis: axis) == .great)
        #expect(ModelMeter.level(value: 75, axis: axis) == .great)
        #expect(ModelMeter.level(value: 60, axis: axis) == .good)
        #expect(ModelMeter.level(value: 50, axis: axis) == .good)
        #expect(ModelMeter.level(value: 40, axis: axis) == .low)
    }

    // MARK: - ModelMeter labels + formatting

    @Test("shortLabel: the compact column names are stable")
    func meterShortLabels() {
        #expect(ModelMeter.shortLabel(for: .generalReasoning) == "Accuracy")
        #expect(ModelMeter.shortLabel(for: .code) == "Code")
        #expect(ModelMeter.shortLabel(for: .tool) == "Tool")
        #expect(ModelMeter.shortLabel(for: .ifeval) == "Instructions")
        #expect(ModelMeter.shortLabel(for: .speed) == "Speed")
    }

    @Test("formatted: rounds to a whole number, suffix-free")
    func meterFormatted() {
        #expect(ModelMeter.formatted(value: 85.6, axis: .generalReasoning) == "86")
        #expect(ModelMeter.formatted(value: 158.2, axis: .speed) == "158")
    }

    // MARK: - ModelMeter.primaryQualityAxis

    @Test("primaryQualityAxis: prefers General & Reasoning, then falls through")
    func primaryQualityAxisFallthrough() {
        let allAround = BenchScores(
            generalReasoning: 70, generalReasoningSource: nil, mmluPro: nil,
            gpqaDiamond: nil, code: 60, tool: 55, ifeval: 80, speedTps: 100
        )
        #expect(ModelMeter.primaryQualityAxis(allAround) == .generalReasoning)

        // A coder alias that only publishes a code score falls to .code.
        let coderOnly = BenchScores(
            generalReasoning: nil, generalReasoningSource: nil, mmluPro: nil,
            gpqaDiamond: nil, code: 65, tool: nil, ifeval: nil, speedTps: nil
        )
        #expect(ModelMeter.primaryQualityAxis(coderOnly) == .code)

        // No quality axis at all → nil.
        let speedOnly = BenchScores(
            generalReasoning: nil, generalReasoningSource: nil, mmluPro: nil,
            gpqaDiamond: nil, code: nil, tool: nil, ifeval: nil, speedTps: 120
        )
        #expect(ModelMeter.primaryQualityAxis(speedOnly) == nil)
    }

    // MARK: - ModelMeter resolved meters (catalog integration)

    @Test("qualityMeter: an unscored alias yields a dashed, em-dash meter")
    func qualityMeterUnscored() {
        let meter = ModelMeter.qualityMeter(for: "definitely-not-a-real-alias-xyz")
        #expect(meter.level == nil)
        #expect(meter.filledSegments == 0)
        #expect(meter.formattedValue == "—")
        // Still labelled so the row renders a quality track, not a blank.
        #expect(meter.label == "Accuracy")
    }

    @Test("speedMeter: an unscored alias yields a dashed, em-dash meter")
    func speedMeterUnscored() {
        let meter = ModelMeter.speedMeter(for: "definitely-not-a-real-alias-xyz")
        #expect(meter.level == nil)
        #expect(meter.filledSegments == 0)
        #expect(meter.formattedValue == "—")
        #expect(meter.label == "Speed")
    }

    @Test("qualityMeter: a scored alias yields a filled, rated meter")
    func qualityMeterScored() throws {
        // Drift-resilient: find any alias the bundled catalog scores on a
        // quality axis rather than hard-coding a name that could change.
        let scored = BenchScoresCatalog.allAliases.first { alias in
            guard let s = BenchScoresCatalog.lookup(alias: alias) else { return false }
            return ModelMeter.primaryQualityAxis(s) != nil
        }
        let alias = try #require(scored, "catalog should score at least one quality axis")
        let meter = ModelMeter.qualityMeter(for: alias)
        #expect(meter.level != nil)
        #expect(meter.formattedValue != "—")
        #expect(meter.filledSegments >= 0)
        #expect(meter.filledSegments <= ModelMeter.segmentCount)
    }

    @Test("qualityMeter: label reflects the resolved axis (a code-only alias reads 'Code', not 'Accuracy')")
    func qualityMeterLabelReflectsResolvedAxis() throws {
        // A coder alias with no published General & Reasoning score
        // resolves its primary quality axis to Code/Tool/Instructions.
        // The meter labels that true axis so a coding pick reads e.g.
        // "Code 66" (its honest coding-bench number) instead of
        // "Accuracy 66", which looks like a failing general-correctness
        // grade and contradicts the Coding card's "best coding-bench"
        // pitch. (Drift-resilient: resolves to whichever code-only alias
        // the catalog carries — e.g. devstral-v2-24b — not a hard-coded
        // name; qwen3-coder-30b's Coding-Index code was nulled in #468.)
        // The meters column header is the axis-agnostic "Quality · Speed"
        // umbrella, so a per-axis row label no longer contradicts it.
        let coder = BenchScoresCatalog.allAliases.first { alias in
            guard let s = BenchScoresCatalog.lookup(alias: alias),
                  let axis = ModelMeter.primaryQualityAxis(s) else { return false }
            return axis != .generalReasoning
        }
        let alias = try #require(coder, "catalog should have a code-only-scored alias")
        let meter = ModelMeter.qualityMeter(for: alias)
        #expect(meter.axis != .generalReasoning) // axis carries the real source
        #expect(meter.label == ModelMeter.shortLabel(for: meter.axis))
        #expect(meter.label != "Accuracy")

        // Positive control: an alias WITH a General & Reasoning score
        // still reads "Accuracy".
        let general = try #require(
            BenchScoresCatalog.allAliases.first { alias in
                BenchScoresCatalog.lookup(alias: alias)?.generalReasoning != nil
            },
            "catalog should have a general-reasoning-scored alias"
        )
        #expect(ModelMeter.qualityMeter(for: general).label == "Accuracy")
    }

    @Test("speedMeter: a speed-scored alias yields a filled, rated meter")
    func speedMeterScored() throws {
        let scored = BenchScoresCatalog.allAliases.first { alias in
            BenchScoresCatalog.lookup(alias: alias)?.speedTps != nil
        }
        let alias = try #require(scored, "catalog should measure at least one speed")
        let meter = ModelMeter.speedMeter(for: alias)
        #expect(meter.level != nil)
        #expect(meter.formattedValue != "—")
        #expect(meter.axis == .speed)
    }

    // MARK: - ModelFavorites.favoritesFirst (pure ordering)

    private func entries(_ aliases: [String]) -> [ModelEntry] {
        aliases.map { ModelEntry(alias: $0, hfRepo: nil, sizeOnDisk: nil, cached: false) }
    }

    @Test("favoritesFirst: pinned aliases float up, relative order preserved")
    func favoritesFirstOrders() {
        let list = entries(["a", "b", "c", "d"])
        let out = ModelFavorites.favoritesFirst(list, favorites: ["c", "a"])
        // Pinned in their ORIGINAL relative order (a before c), then rest.
        #expect(out.map(\.alias) == ["a", "c", "b", "d"])
    }

    @Test("favoritesFirst: empty favorites returns the list unchanged")
    func favoritesFirstEmpty() {
        let list = entries(["a", "b", "c"])
        let out = ModelFavorites.favoritesFirst(list, favorites: [])
        #expect(out.map(\.alias) == ["a", "b", "c"])
    }

    @Test("favoritesFirst: a favorite not in the list is simply absent")
    func favoritesFirstMissing() {
        let list = entries(["a", "b"])
        let out = ModelFavorites.favoritesFirst(list, favorites: ["z"])
        #expect(out.map(\.alias) == ["a", "b"])
    }

    @Test("favoritesFirst: all-favorites keeps the original order")
    func favoritesFirstAll() {
        let list = entries(["a", "b", "c"])
        let out = ModelFavorites.favoritesFirst(list, favorites: ["a", "b", "c"])
        #expect(out.map(\.alias) == ["a", "b", "c"])
    }

    // MARK: - ModelFavorites persistence (scratch defaults)

    @Test("toggle: round-trips through a scratch UserDefaults suite")
    func favoritesTogglePersists() throws {
        let suite = "test.rapid.models.favorites.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }

        #expect(ModelFavorites.load(defaults: defaults).isEmpty)

        let nowOn = ModelFavorites.toggle("qwen3.6-35b-4bit", defaults: defaults)
        #expect(nowOn)
        #expect(ModelFavorites.isFavorite("qwen3.6-35b-4bit", defaults: defaults))
        #expect(ModelFavorites.load(defaults: defaults) == ["qwen3.6-35b-4bit"])

        let nowOff = ModelFavorites.toggle("qwen3.6-35b-4bit", defaults: defaults)
        #expect(!nowOff)
        #expect(!ModelFavorites.isFavorite("qwen3.6-35b-4bit", defaults: defaults))
        #expect(ModelFavorites.load(defaults: defaults).isEmpty)
    }

    // MARK: - Recommended card geometry

    /// The reported defect: the best-pick card's primary call to action
    /// rendered as "Dow…". Nothing truncated the label directly — the
    /// slot it sat in was pinned to 92pt and its contents did not fit,
    /// so AppKit ellipsised the title. Widening the window did not help
    /// because the clamp was on the card, not the window.
    ///
    /// This measures the real thing: AppKit's own metrics for the small
    /// system font, plus the button's chrome, against the width the card
    /// gives that slot. Any future label, control size or padding change
    /// that reintroduces the squeeze fails here.
    @Test("recommended card: every action label fits the slot it renders in")
    func recommendedActionLabelsFitTheirColumn() {
        for title in RecommendedCardLayout.actionButtonTitles {
            let needed = RecommendedCardLayout.buttonWidth(title: title)
            #expect(
                needed <= RecommendedCardLayout.actionColumnWidth,
                "\"\(title)\" needs \(needed)pt but the action column offers \(RecommendedCardLayout.actionColumnWidth)pt"
            )
        }
        for text in RecommendedCardLayout.actionPillTitles {
            let needed = RecommendedCardLayout.pillWidth(text: text)
            #expect(
                needed <= RecommendedCardLayout.actionColumnWidth,
                "\"\(text)\" needs \(needed)pt but the action column offers \(RecommendedCardLayout.actionColumnWidth)pt"
            )
        }
    }

    /// The sweep above can only mean something if the two arrays really
    /// are the labels the card renders — they are hand-maintained, and a
    /// width derived from them would otherwise be self-certifying.
    ///
    /// So: grep the source. Every `Text("…")` inside the card's
    /// ``actionButton`` and inside ``statusBadgeView``'s pill calls must
    /// appear in one of the two arrays. Adding "Download again" to the
    /// button without widening the column trips here rather than shipping
    /// another ellipsis. Same pattern as
    /// ``ToolUseCapabilitySourceGuardTests``.
    @Test("recommended card: the measured label set matches the source")
    func actionLabelArraysMatchTheSource() throws {
        let source = try Self.loadPanelSource()
        let known = Set(
            RecommendedCardLayout.actionButtonTitles
                + RecommendedCardLayout.actionPillTitles
        )

        let buttonBody = try Self.body(
            of: "private func actionButton(for entry: ModelEntry, badge: ModelCacheActions.StatusBadge) -> some View {",
            in: source
        )
        for label in Self.textLiterals(in: buttonBody) {
            #expect(
                known.contains(label),
                """
                actionButton renders Text("\(label)") but RecommendedCardLayout \
                does not measure it, so the card's action column width was \
                derived without it. Add it to actionButtonTitles.
                """
            )
        }

        let badgeBody = try Self.body(
            of: "private func statusBadgeView(_ badge: ModelCacheActions.StatusBadge) -> some View {",
            in: source
        )
        // The pills the CARD can reach. ``statusBadgeView`` is shared with
        // states the card never routes to it, so only these two are
        // required to be measured.
        for label in ["On disk", "In use"] {
            #expect(
                badgeBody.contains("\"\(label)\""),
                "statusBadgeView no longer renders the \"\(label)\" pill the card's width was measured against"
            )
            #expect(known.contains(label))
        }
    }

    /// Panel source, located from ``#filePath`` so the test runs from any
    /// working directory.
    private static func loadPanelSource() throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // RapidTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // apps/rapid-mac
        return try String(
            contentsOf: root.appendingPathComponent(
                "Sources/Rapid/UI/SettingsModelManagementPanel.swift"
            ),
            encoding: .utf8
        )
    }

    /// One function's body, scoped by matching the braces after its
    /// signature. A "up to the next MARK" heuristic silently swallowed
    /// the following functions, which would have made this guard pass on
    /// labels it never actually saw.
    private static func body(of signature: String, in source: String) throws -> String {
        struct MissingDeclaration: Error { let signature: String }
        guard let start = source.range(of: signature) else {
            throw MissingDeclaration(signature: signature)
        }
        // ``signature`` ends with the opening brace, so start at depth 1.
        var depth = 1
        var end = start.upperBound
        while end < source.endIndex, depth > 0 {
            switch source[end] {
            case "{": depth += 1
            case "}": depth -= 1
            default: break
            }
            if depth > 0 { end = source.index(after: end) }
        }
        guard depth == 0 else { throw MissingDeclaration(signature: signature) }
        return String(source[start.upperBound..<end])
    }

    /// Every `Text("…")` string literal in a fragment of Swift source.
    private static func textLiterals(in fragment: String) -> [String] {
        var out: [String] = []
        var remainder = Substring(fragment)
        while let open = remainder.range(of: "Text(\"") {
            let after = remainder[open.upperBound...]
            guard let close = after.firstIndex(of: "\"") else { break }
            let literal = String(after[..<close])
            // Skip interpolated labels — they are not fixed strings and
            // cannot be width-measured up front.
            if !literal.contains("\\(") { out.append(literal) }
            remainder = after[close...]
        }
        return out
    }

    /// The specific label that broke, called out on its own so a
    /// regression names the reported symptom rather than a sweep index.
    @Test("recommended card: \"Download\" renders in full")
    func downloadLabelIsNotTruncated() {
        let needed = RecommendedCardLayout.buttonWidth(title: "Download")
        #expect(needed <= RecommendedCardLayout.actionColumnWidth)
        // And the width is genuinely derived from the labels rather than
        // being a literal that happens to be large today.
        #expect(RecommendedCardLayout.actionColumnWidth >= needed)
        #expect(RecommendedCardLayout.actionColumnWidth < needed + 40)
    }

    /// Why the size caption had to leave the action slot: with it, the
    /// old 92pt column could not fit "Download" alongside it, which is
    /// exactly the arithmetic that produced "Dow…". Pinned so nobody
    /// re-adds a caption there without hitting this.
    @Test("recommended card: a caption beside the button re-creates the squeeze")
    func sizeCaptionBesideTheButtonDoesNotFit() {
        let legacyColumn: CGFloat = 92
        let caption = RecommendedCardLayout.captionWidth("7.6 GB")
        let button = RecommendedCardLayout.buttonWidth(title: "Download")
        #expect(caption + 8 + button > legacyColumn)
    }

    // MARK: - Models table geometry

    /// Defect 3's other half: now that a cached cell carries a figure and
    /// not just two glyphs, the Size column has to be wide enough to hold
    /// it — including on the machines with the largest caches, which are
    /// exactly the users who go looking for this tab.
    @Test("size column: a cached cell fits glyph + measured size + delete")
    func cachedCellFitsTheSizeColumn() {
        for size in ["7.9 GiB", "512 MiB", "123.4 GiB"] {
            let needed = ModelTableLayout.cachedCellWidth(size: size)
            #expect(
                needed <= ModelTableLayout.sizeColumnWidth,
                "\"\(size)\" needs \(needed)pt, column is \(ModelTableLayout.sizeColumnWidth)pt"
            )
        }
        // Why the column had to move at all: the 84pt it replaced would
        // have clipped the largest caches.
        #expect(ModelTableLayout.cachedCellWidth(size: "123.4 GiB") > 84)
    }

    @Test("size column: a not-cached cell still fits after the ~ marker")
    func notCachedCellFitsTheSizeColumn() {
        for size in ["0.5 GB", "42.7 GB", "123.4 GB"] {
            let needed = ModelTableLayout.notCachedCellWidth(size: size)
            #expect(
                needed <= ModelTableLayout.sizeColumnWidth,
                "\"~\(size)\" needs \(needed)pt, column is \(ModelTableLayout.sizeColumnWidth)pt"
            )
        }
    }

    /// Settings is not inside the chat surface's Dynamic Type clamp, so
    /// these `.caption` figures grow with the system text size against a
    /// fixed-width column — the exact shape of the truncation this pass
    /// is fixing, just triggered by the user's text setting instead of
    /// the layout. The cells carry a shrink floor; this pins that the
    /// floor actually covers the non-accessibility sizes (~1.25x) with
    /// the largest figures.
    @Test("size column: cells survive non-accessibility text growth")
    func sizeCellsSurviveTextScaling() {
        let scale: CGFloat = 1.25
        for size in ["7.9 GiB", "123.4 GiB"] {
            #expect(
                ModelTableLayout.fits(
                    ModelTableLayout.cachedCellWidth(size: size), atTextScale: scale
                ),
                "cached \"\(size)\" truncates at \(scale)x text"
            )
            #expect(
                ModelTableLayout.fits(
                    ModelTableLayout.inUseCellWidth(size: size), atTextScale: scale
                ),
                "serving \"\(size)\" truncates at \(scale)x text"
            )
        }
        for size in ["0.5 GB", "123.4 GB"] {
            #expect(
                ModelTableLayout.fits(
                    ModelTableLayout.notCachedCellWidth(size: size), atTextScale: scale
                ),
                "not cached \"~\(size)\" truncates at \(scale)x text"
            )
        }
        // The floor is a bound, not a Dynamic Type pass — say so in the
        // suite rather than implying the AX sizes are covered.
        #expect(
            !ModelTableLayout.fits(
                ModelTableLayout.inUseCellWidth(size: "123.4 GiB"), atTextScale: 2.0
            ),
            "if this now passes, the column reflows and the comment on cellMinimumScaleFactor is stale"
        )
    }

    /// The serving row is a cached row too — it owes the same size, and
    /// the "Serving" label it carries instead of a delete button is wider
    /// than that button, so it needs its own width check.
    @Test("size column: a serving cell fits its size next to \"Serving\"")
    func inUseCellFitsTheSizeColumn() {
        for size in ["7.9 GiB", "35.2 GiB", "123.4 GiB"] {
            let needed = ModelTableLayout.inUseCellWidth(size: size)
            #expect(
                needed <= ModelTableLayout.sizeColumnWidth,
                "serving \"\(size)\" needs \(needed)pt, column is \(ModelTableLayout.sizeColumnWidth)pt"
            )
        }
    }

    // MARK: - Size column: measured vs. estimated

    /// A cached row must quote what was MEASURED on disk
    /// (``ModelEntry.sizeOnDisk``, from `rapid-mlx ls`), verbatim, so the
    /// Model Management table and Settings → Models cannot print
    /// different numbers for the same model.
    @Test("size column: a cached row quotes the measured on-disk size")
    func cachedRowUsesMeasuredSize() {
        let entry = ModelEntry(
            alias: "bonsai-27b-2bit", hfRepo: nil, sizeOnDisk: "7.9 GiB", cached: true
        )
        #expect(SettingsModelManagementPanel.onDiskSizeLabel(entry) == "7.9 GiB")
    }

    /// No measurement, no number. Substituting the alias-derived
    /// estimate here would dress a guess as a measurement — #1550 is the
    /// same confusion in the download strip, where the estimate came in
    /// ~12% under the real bytes.
    @Test("size column: an unmeasured cached row shows nothing, not an estimate")
    func cachedRowWithoutMeasurementFallsBackToNothing() {
        let unmeasured = ModelEntry(
            alias: "qwen3.5-9b-4bit", hfRepo: nil, sizeOnDisk: nil, cached: true
        )
        #expect(SettingsModelManagementPanel.onDiskSizeLabel(unmeasured) == nil)
        // The estimate for that alias exists and is NOT what we show.
        #expect(SettingsModelManagementPanel.downloadSizeLabel("qwen3.5-9b-4bit") != nil)

        let blank = ModelEntry(
            alias: "qwen3.5-9b-4bit", hfRepo: nil, sizeOnDisk: "   ", cached: true
        )
        #expect(SettingsModelManagementPanel.onDiskSizeLabel(blank) == nil)
    }

    /// The two numbers now share a column, so they must not share a
    /// format: the estimate is rendered with a leading "~" by the row,
    /// and the helper hands back the bare figure for that to be added to
    /// exactly once.
    @Test("size column: the download estimate is a bare figure the row marks as approximate")
    func downloadEstimateIsBare() {
        let estimate = SettingsModelManagementPanel.downloadSizeLabel("qwen3.5-9b-4bit")
        #expect(estimate?.hasPrefix("~") == false)
        #expect(estimate?.hasSuffix(" GB") == true)
        // An alias ``ModelSizing`` cannot size at all yields no figure
        // rather than "~0.0 GB".
        #expect(SettingsModelManagementPanel.downloadSizeLabel("") == nil)
    }
}
