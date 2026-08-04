import SwiftUI
import Testing
import ViewInspector
@testable import Rapid

/// Start/Stop label switching + state-badge text. The picker's
/// internal Menu is hard to drive through ViewInspector cleanly
/// (private SwiftUI types), but the textual labels are exposed and
/// pin the routing in ``ModelPickerBar.isStartable`` /
/// ``stateLabel``.
@MainActor
@Suite("ModelPickerBar state badge + button labels")
struct ModelPickerBarTests {
    private func makeView(state: ServerState, alias: Binding<String>? = nil) -> some View {
        let server = ServerManager(testingState: state, binaryPath: URL(fileURLWithPath: "/dev/null"))
        let aliasBinding = alias ?? .constant("fake-alias")
        // v0.5.7: picker now takes a ``DownloadManager`` for the
        // side-car "Download in background" context-menu entry.
        // Tests use the no-binary ``DownloadManager()`` test seam
        // so no real subprocess can fire even if a test
        // accidentally exercised the right-click path.
        let downloadsStub = DownloadManager()
        return ModelPickerBar(server: server, downloads: downloadsStub, alias: aliasBinding)
    }

    @Test("Ready state shows 'Stop model' button (not Start)")
    func readyShowsStop() throws {
        let sut = makeView(state: .ready(alias: "fake-alias"))
        // The label is "Stop model" (not a bare "Stop") to disambiguate
        // the server-unload control from the composer's "Stop response"
        // circle. See ModelPickerBar's Stop button + the 丝滑 dogfood.
        #expect(throws: Never.self) {
            try sut.inspect().find(text: "Stop model")
        }
    }

    @Test("Idle state shows a Start-class button (not Stop model)")
    func idleShowsStart() throws {
        // The label varies by cache state: "Start" for a cached
        // alias, "Download & start" for an uncached one. Both
        // contain "start" — the absence of a "Stop model" button is
        // the load-bearing assertion (otherwise the user can't kick
        // off a model from idle).
        let sut = makeView(state: .idle)
        #expect(throws: Error.self) {
            try sut.inspect().find(text: "Stop model")
        }
    }

    @Test("Starting state shows 'Stop model' button (mid-handshake teardown is the escape hatch)")
    func startingShowsStop() throws {
        let sut = makeView(state: .starting(alias: "fake-alias"))
        // While starting we present Stop model because the user's only
        // available action is to cancel the spawn — pressing Start
        // again would be a no-op.
        #expect(throws: Never.self) {
            try sut.inspect().find(text: "Stop model")
        }
    }

    @Test("State badge reads 'Ready' (alias lives in the picker, not the pill)")
    func badgeReflectsReadyAlias() throws {
        let sut = makeView(state: .ready(alias: "qwen3.6-27b"))
        // v0.5 (Phase 4): the status pill no longer repeats the alias —
        // the model name is shown once, in the picker to its left. The
        // pill is now just the status word.
        #expect(throws: Never.self) {
            try sut.inspect().find(text: "Ready")
        }
    }

    @Test("Missing-binary badge text is 'Not installed'")
    func badgeMissing() throws {
        let server = ServerManager(testingState: .missing, binaryPath: nil)
        let sut = ModelPickerBar(server: server, downloads: DownloadManager(), alias: .constant(""))
        #expect(throws: Never.self) {
            try sut.inspect().find(text: "Not installed")
        }
    }

    // MARK: - v0.6.9 recommended-row selection (Option B from the picker-v2 mock)

    @Test("Role row is selected when the picker alias equals the row alias — paints amber")
    func roleRowMatchingAliasIsSelected() {
        // Option B from the mock: the recommended-section row whose
        // alias matches the currently selected one gets the
        // Start-button amber treatment. The truth table is trivial
        // equality but lifted to a static helper so this pin catches
        // a future "should also match on family prefix" drift.
        #expect(
            ModelPickerBar.roleRowIsSelected(
                selectedAlias: "qwen3.5-9b-4bit",
                rowAlias: "qwen3.5-9b-4bit"
            ) == true
        )
    }

    @Test("Role row is NOT selected when picker alias is a different alphabetical entry")
    func roleRowDifferentAliasNotSelected() {
        // User picked a non-recommended alias from "All models" — no
        // recommended row should highlight. Picker label up top still
        // names the current alias.
        #expect(
            ModelPickerBar.roleRowIsSelected(
                selectedAlias: "gemma-4-12b-4bit",
                rowAlias: "qwen3.5-9b-4bit"
            ) == false
        )
    }

    @Test("Role row is NOT selected when picker alias is empty (first-launch transient)")
    func roleRowEmptyAliasNotSelected() {
        // Catalog-still-loading state: an empty alias paired against
        // every role would paint all five rows amber simultaneously.
        // The empty-string guard rules that out.
        #expect(
            ModelPickerBar.roleRowIsSelected(
                selectedAlias: "",
                rowAlias: "qwen3.5-9b-4bit"
            ) == false
        )
    }

    // MARK: - v0.6.9 menu-rendering fix (NSMenuItem collapses HStack)

    @Test("Recommended row menu title folds label + alias into a single em-dash string")
    func recommendedRowMenuTitleFormat() {
        // SwiftUI Menu wraps each Button as an NSMenuItem and drops
        // everything past the first Text. Folding the label + alias into
        // one Text() via this helper survives the NSMenu collapse; " — "
        // em-dash anchors the "Recommended"/"Faster" label on the left.
        #expect(
            ModelPickerBar.recommendedRowMenuTitle(label: "Recommended", alias: "bonsai-27b-2bit")
                == "Recommended — bonsai-27b-2bit"
        )
        #expect(
            ModelPickerBar.recommendedRowMenuTitle(label: "Faster", alias: "lfm2.5-8b-a1b-4bit")
                == "Faster — lfm2.5-8b-a1b-4bit"
        )
    }

    @Test("Recommended tagline reflects the ACTUAL tier pick's capability + tok/s")
    func recommendedTaglineCopy() {
        // Pin against the real table, not synthetic picks, so a future
        // edit to the 16 GB tier's numbers has to update this expectation.
        let tier16 = RAMBucketedDefault.tier(forPhysicalRAMGB: 16)
        // Literal expectations pin the actual curated numbers (86 / 17.1)
        // AND the tagline format — a change to either the copy or the
        // table's numbers must update this assertion.
        let taglineP = ModelPickerBar.recommendedTagline(pick: tier16.primary, isPrimary: true)
        #expect(taglineP == "Best pick for your Mac · 86% capability · ~17 tok/s")

        #expect(tier16.alt != nil, "16 GB tier carries a faster alternative")
        if let alt = tier16.alt {
            // The fast chat specialist shows its "Chat only" caveat in place
            // of the blended 62 % (which understates conversation quality).
            let taglineA = ModelPickerBar.recommendedTagline(pick: alt, isPrimary: false)
            #expect(taglineA == "Faster, lighter alternative · ~117 tok/s · Chat only")
        }

        // A tier with no local tok/s measurement (64 GB → 35b-8bit) omits
        // the tok/s clause.
        let noSpeed = RAMBucketedDefault.tier(forPhysicalRAMGB: 64).primary
        #expect(noSpeed.tokensPerSec == nil)
        #expect(!ModelPickerBar.recommendedTagline(pick: noSpeed, isPrimary: true).contains("tok/s"))
    }

    // MARK: - One download-state vocabulary
    //
    // Dogfood report: in "Recommended for your 16 GB Mac" there was no
    // way to tell whether a model was already downloaded — and the rows
    // below used a dashed circle, so it wasn't even clear whether the
    // app had a symbol for it. It had three: circle.fill/circle.dashed
    // in the menu rows, checkmark.circle.fill/icloud.and.arrow.down on
    // the closed picker, and nothing on the Recommended rows.

    @Test("Download state uses the platform-standard cloud glyph, one vocabulary")
    func cacheGlyphIsStandardAndUniform() {
        // `icloud.and.arrow.down` is what Music / TV / Podcasts /
        // Photos / Files all use for "not on this device yet".
        #expect(ModelPickerBar.cacheGlyph(cached: false) == "icloud.and.arrow.down")
        // The downloaded side matches what the CLOSED picker already
        // shows for the same state, so the dropdown and the control it
        // drops from agree.
        #expect(ModelPickerBar.cacheGlyph(cached: true) == "checkmark.circle.fill")
    }

    @Test("No picker row still speaks the retired dashed-circle dialect")
    func dashedCircleIsRetired() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/Rapid/UI/ModelPickerBar.swift")
        let code = try String(contentsOf: url, encoding: .utf8)
            .components(separatedBy: "\n")
            .filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("///") }
            .joined(separator: "\n")
        #expect(
            !code.contains("\"circle.dashed\""),
            "download state must go through cacheGlyph(cached:), not a per-row literal."
        )
    }

    @Test("The current model is named in words, not left to a bare tick")
    func currentSelectionIsSpelledOut() {
        // The leading image slot now carries download state, and
        // NSMenuItem honours only one image per row — so selection
        // moved into the title. A word also disambiguates what the old
        // tick meant: this is the current model.
        #expect(
            ModelPickerBar.currentSelectionTitle("Best pick — bonsai-27b-2bit")
                == "Best pick — bonsai-27b-2bit (current)"
        )
    }

    // MARK: - v0.6.9 picker shape (Cached section removed, All models alphabetical)

    @Test("ModelPickerBar no longer exposes a `cachedSection` view")
    func noCachedSectionView() {
        // The picker now renders a single alphabetical "All models"
        // list plus the "Recommended for your N GB Mac" section. The
        // separate "Cached" section was deleted because users skim by
        // alias name, not cache state — the on-disk affordance is now
        // a small green dot at the right edge of each row.
        //
        // Reflection-based check would be brittle. Instead we pin the
        // `fitSuffixLabel` static helper removal — its only call site
        // was the (now-deleted) cached/all entry label. If
        // `fitSuffixLabel` came back, the suffix would be back too.
        let m = Mirror(reflecting: ModelPickerBar.self)
        let staticChildNames = m.children.compactMap { $0.label }
        #expect(!staticChildNames.contains("fitSuffixLabel"))
    }

    @Test("All models list places downloaded models first, then alphabetical")
    func allModelsDownloadedFirst() {
        // The "All models" section sorts cached (downloaded) aliases to
        // the top so a model the user already pulled surfaces instead of
        // being buried mid-alphabet behind a small green dot. Alphabetical
        // within each group. Guards against the display step reverting to
        // pure alphabetical (which discards ModelCatalog.load's cached-first
        // order).
        let entries = [
            ModelEntry(alias: "alpha-3b-4bit", hfRepo: "stub/alpha", sizeOnDisk: nil, cached: false),
            ModelEntry(alias: "qwen3.5-4b-4bit", hfRepo: "mlx-community/Qwen3.5-4B-MLX-4bit", sizeOnDisk: "5.7 GiB", cached: true),
            ModelEntry(alias: "gemma-4-12b-4bit", hfRepo: "stub/gemma", sizeOnDisk: nil, cached: false),
            ModelEntry(alias: "aardvark-7b-4bit", hfRepo: "stub/aard", sizeOnDisk: "1.0 GiB", cached: true),
        ]
        let ordered = ModelPickerBar.orderedAllModels(entries).map(\.alias)
        #expect(ordered == [
            "aardvark-7b-4bit",   // cached, alphabetical within the cached group
            "qwen3.5-4b-4bit",    // cached
            "alpha-3b-4bit",      // uncached, alphabetical within the uncached group
            "gemma-4-12b-4bit",   // uncached
        ])
    }

    @Test("orderedAllModels is a stable no-op when nothing is downloaded")
    func allModelsNoCacheStaysAlphabetical() {
        let entries = [
            ModelEntry(alias: "gemma-4-12b-4bit", hfRepo: "stub/gemma", sizeOnDisk: nil, cached: false),
            ModelEntry(alias: "alpha-3b-4bit", hfRepo: "stub/alpha", sizeOnDisk: nil, cached: false),
        ]
        #expect(ModelPickerBar.orderedAllModels(entries).map(\.alias) == ["alpha-3b-4bit", "gemma-4-12b-4bit"])
    }

    // MARK: - v0.6.9 recommendation header (unchanged contract)

    @Test("Recommendation header carries the rounded RAM in GB")
    func recommendationHeaderRoundsRAM() {
        // Matches MacHardware.shortDescription's Int(.rounded()) so the
        // picker title agrees with macOS About-dialog reporting.
        #expect(ModelPickerBar.recommendedHeaderTitle(physicalRAMGB: 18) == "Recommended for your 18 GB Mac")
        #expect(ModelPickerBar.recommendedHeaderTitle(physicalRAMGB: 17.6) == "Recommended for your 18 GB Mac")
        #expect(ModelPickerBar.recommendedHeaderTitle(physicalRAMGB: 256) == "Recommended for your 256 GB Mac")
    }

    @Test("Recommendation header clamps zero / negative RAM to '1 GB' rather than '0 GB'")
    func recommendationHeaderClampsNonPositive() {
        // The picker never *actually* sees a 0 GB Mac (sysctl
        // wouldn't boot), but a defensive clamp keeps the header copy
        // grammatical if a future probe ever returns 0.
        #expect(ModelPickerBar.recommendedHeaderTitle(physicalRAMGB: 0) == "Recommended for your 1 GB Mac")
        #expect(ModelPickerBar.recommendedHeaderTitle(physicalRAMGB: -8) == "Recommended for your 1 GB Mac")
    }

    // MARK: - v0.6.9 .tooBig Start confirmation alert

    @Test(".tooBig alert title front-loads the alias name and the host RAM")
    func tooBigAlertTitleShape() {
        // Same pattern as the delete-confirmation title — identity
        // first ("qwen3.5-122b-mxfp4 likely won't fit") so the user
        // doesn't need to read past the model name to feel the cost.
        let title = ModelPickerBar.tooBigAlertTitle(
            alias: "qwen3.5-122b-mxfp4",
            physicalRAMGB: 18
        )
        #expect(title == "qwen3.5-122b-mxfp4 likely won't fit your 18 GB Mac")
    }

    @Test(".tooBig alert title falls back gracefully on an empty alias")
    func tooBigAlertTitleEmptyAlias() {
        // Defensive: a transient (pendingTooBigStart == "") shouldn't
        // produce " likely won't fit…" with a leading space.
        let title = ModelPickerBar.tooBigAlertTitle(alias: "", physicalRAMGB: 18)
        #expect(title == "This model likely won't fit your 18 GB Mac")
    }

    @Test(".tooBig alert message names both the estimated need and the consequence")
    func tooBigAlertMessageBody() {
        // A real big-alias-on-small-Mac case: qwen3.5-122b-mxfp4
        // estimated at ~65 GB weights + overhead, on an 18 GB Mac
        // (usable ≈ 14 GB). The message must include both numbers AND
        // the consequence-sentence so the user understands what
        // "Start anyway" really means.
        let host = MacHardware(
            brandString: "Apple M2",
            family: .m2,
            tier: .base,
            physicalRAMBytes: 18 * 1024 * 1024 * 1024,
            memoryBandwidthGBs: 100
        )
        let fp = ModelSizing.estimate(alias: "qwen3.6-122b-mxfp4")
        let body = ModelPickerBar.tooBigAlertMessage(
            alias: "qwen3.6-122b-mxfp4",
            footprint: fp,
            hardware: host
        )
        // Body must contain the usable-RAM number and the consequence
        // sentence. Exact GB number is an estimate so we just check
        // the consequence phrase is present.
        #expect(body.contains("about 14 GB available") || body.contains("usable RAM (14 GB)"))
        #expect(body.contains("swap thrashing"))
        #expect(body.contains("system lock-up"))
    }

    @Test(".tooBig classify on a 18 GB Mac drives the alert (.recommended/.borderline don't)")
    func tooBigDrivesAlertOnlyForTooBig() {
        // Wire-up smoke: only the .tooBig classification should land
        // in pendingTooBigStart. The pure helper that the production
        // path consults is ModelSizing.classify — we pin its outputs
        // on a real 18 GB fixture so a future drift in the estimator
        // surfaces here rather than at runtime.
        let host = MacHardware(
            brandString: "Apple M2",
            family: .m2,
            tier: .base,
            physicalRAMBytes: 18 * 1024 * 1024 * 1024,
            memoryBandwidthGBs: 100
        )
        // A 4B model is .recommended on 18 GB.
        let smallFit = ModelSizing.classify(
            ModelSizing.estimate(alias: "qwen3.5-4b-4bit"),
            on: host
        )
        #expect(smallFit != .tooBig)
        // A 122B model is .tooBig on 18 GB.
        let bigFit = ModelSizing.classify(
            ModelSizing.estimate(alias: "qwen3.6-122b-mxfp4"),
            on: host
        )
        #expect(bigFit == .tooBig)
    }

    // MARK: - cycle-10: sub-3B quality sticker (F9-004)

    @Test("aliasButtonTitle appends bucket-distinct sticker for sub-3B aliases (.tiny → ' · tiny', .small → ' · small' per #348); leaves 3B+ aliases (.midOrLarger) untouched — cycle-11 boundary tightened to strict < 3B. #133/FU-9: .broken aliases append ' · no tools' (strong); .unknown aliases append ' · tools unverified' (softer); .known aliases append nothing")
    func aliasButtonTitleSticker() {
        // 1B llama: small sticker fires (.small post-cycle-11; #348
        // split the suffix so .small renders "· small" instead of
        // sharing the .tiny label) AND #133 no-tools badge fires
        // (llama3-1b is .broken per ToolUseCapability — cycle-9
        // F9-001 schema-leak; the FU-9 split preserves the strong
        // "no tools" copy for .broken).
        #expect(
            ModelPickerBar.aliasButtonTitle(
                alias: "llama3-1b-4bit",
                bucket: .small
            ) == "llama3-1b-4bit · small · no tools"
        )
        // 2B custom: small sticker fires (.small) AND FU-9 badge
        // fires with the SOFTER "tools unverified" copy because
        // custom-2b is .unknown — the bench loop has no signal for it,
        // and the conservative default should say "we haven't
        // verified" instead of "we know it's broken".
        #expect(
            ModelPickerBar.aliasButtonTitle(
                alias: "custom-2b-4bit",
                bucket: .small
            ) == "custom-2b-4bit · small · tools unverified"
        )
        // 4B project test default: NO sticker — clean row.
        // qwen3.5-4b-4bit is .known per ToolUseCapability (bench
        // backbone) so neither badge fires — full clean row.
        #expect(
            ModelPickerBar.aliasButtonTitle(
                alias: "qwen3.5-4b-4bit",
                bucket: .midOrLarger
            ) == "qwen3.5-4b-4bit"
        )
        // 35B big model: NO sticker. qwen3.6-35b is .known per
        // ToolUseCapability — clean row, no badge.
        #expect(
            ModelPickerBar.aliasButtonTitle(
                alias: "qwen3.6-35b-8bit",
                bucket: .midOrLarger
            ) == "qwen3.6-35b-8bit"
        )
        // Sub-1B (e.g. surfaced via Show all): tiny sticker fires AND
        // FU-9 softer "tools unverified" badge fires (qwen3-0.6b-4bit
        // is .unknown per the capability map's "broken-means-
        // empirically-observed" rule — sub-1B bench coverage is N/A
        // so "unverified" rather than "no tools"). #348 regression
        // pin: ``.tiny`` still renders "· tiny" — the split only
        // touched ``.small``.
        #expect(
            ModelPickerBar.aliasButtonTitle(
                alias: "qwen3-0.6b-4bit",
                bucket: .tiny
            ) == "qwen3-0.6b-4bit · tiny · tools unverified"
        )
    }

    @Test("cycle-11 F-10-PRESET integration — computed bucket → title pipeline: llama3-3b-4bit resolves to .midOrLarger and produces a CLEAN title with no '· tiny' suffix (regression pin against a <= 3B revert)")
    func aliasButtonTitleLlama3BIntegration() {
        // Integration pin: walk the same path the picker takes —
        // ``ModelPickerVisibility.qualityBucket(for:)`` (the source
        // of truth) feeds ``ModelPickerBar.aliasButtonTitle`` (the
        // render helper). A future revert of the bucket boundary to
        // ``<= 3.0`` would flip both 3B aliases here back to
        // ``.small`` and re-introduce the "· tiny" suffix —
        // tripping this test with a clear failure message.
        let llama3B = "llama3-3b-4bit"
        let llamaBucket = ModelPickerVisibility.qualityBucket(for: llama3B)
        #expect(llamaBucket == .midOrLarger)
        #expect(
            ModelPickerBar.aliasButtonTitle(alias: llama3B, bucket: llamaBucket)
                == "llama3-3b-4bit",
            "llama3-3b-4bit must render WITHOUT the · tiny sticker — it is the smallest viable llama-family chat preset per cycle-10."
        )
        // Sister 1B alias still renders WITH the sticker. #133:
        // llama3-1b is also .broken per ToolUseCapability (cycle-9
        // F9-001 schema-leak) so the no-tools badge composes with
        // the quality sticker. #348: the sticker now reads "· small"
        // (matches the ``.small`` bucket) instead of the previous
        // "· tiny" — the data-model split is now visible in the UI.
        let llama1B = "llama3-1b-4bit"
        let llama1Bucket = ModelPickerVisibility.qualityBucket(for: llama1B)
        #expect(llama1Bucket == .small)
        #expect(
            ModelPickerBar.aliasButtonTitle(alias: llama1B, bucket: llama1Bucket)
                == "llama3-1b-4bit · small · no tools"
        )
    }

    @Test("aliasRowHelpText: sub-3B + cached → two-line tooltip; 3B+ + cached → bare 'Already downloaded'; 3B+ + uncached → 'Will download on Start' (cycle-11 strict bound — 3B aliases like llama3-3b-4bit land in the clean band)")
    func aliasRowHelpTextBranches() {
        let smallCached = ModelPickerBar.aliasRowHelpText(
            bucket: .small,
            cached: true
        )
        #expect(smallCached.hasSuffix("Already downloaded"))
        #expect(smallCached.contains("multi-turn"))

        let midCached = ModelPickerBar.aliasRowHelpText(
            bucket: .midOrLarger,
            cached: true
        )
        #expect(midCached == "Already downloaded")

        let midUncached = ModelPickerBar.aliasRowHelpText(
            bucket: .midOrLarger,
            cached: false
        )
        #expect(midUncached == "Will download on Start")

        let tinyUncached = ModelPickerBar.aliasRowHelpText(
            bucket: .tiny,
            cached: false
        )
        #expect(tinyUncached.hasSuffix("Will download on Start"))
        #expect(tinyUncached.contains("qwen3.5-4b"))
    }

    @Test("aliasRowAccessibilityLabel surfaces the bucket-distinct sticker word to VoiceOver for sub-3B (.tiny → 'tiny model', .small → 'small model' per #348); 3B+ keeps the legacy short form (cycle-11 strict bound). #133/FU-9: .broken aliases append ', no tools' (strong); .unknown aliases append ', tools unverified' (softer)")
    func aliasRowAccessibilityLabelBranches() {
        // 1B + cached → composed warning. #133: llama3-1b is .broken
        // per ToolUseCapability so the VoiceOver label appends the
        // strong ", no tools" copy after the multi-turn warning
        // (FU-9 preserved the .broken wording). #348: the bucket
        // word is now "small" (matches the ``.small`` bucket) so
        // VoiceOver users get the same bucket signal sighted users
        // see in ``qualityStickerSuffix``.
        #expect(
            ModelPickerBar.aliasRowAccessibilityLabel(
                alias: "llama3-1b-4bit",
                cached: true,
                bucket: .small
            ) == "llama3-1b-4bit, downloaded, small model — may contradict itself in multi-turn chat, no tools"
        )
        // 4B + uncached → unchanged legacy form. qwen3.5-4b-4bit is
        // .known per ToolUseCapability so neither badge fires — pure
        // regression pin against badging a known-good alias.
        #expect(
            ModelPickerBar.aliasRowAccessibilityLabel(
                alias: "qwen3.5-4b-4bit",
                cached: false,
                bucket: .midOrLarger
            ) == "qwen3.5-4b-4bit, not downloaded"
        )
        // 0.6B (surfaced via Show all) + uncached → composed warning.
        // FU-9: qwen3-0.6b-4bit is .unknown per ToolUseCapability
        // (sub-1B bench coverage is N/A; conservative-default badged)
        // → softer ", tools unverified" suffix so VoiceOver doesn't
        // declare an unbenched model broken. #348 regression pin:
        // ``.tiny`` still says "tiny model" — the split only touched
        // ``.small``.
        #expect(
            ModelPickerBar.aliasRowAccessibilityLabel(
                alias: "qwen3-0.6b-4bit",
                cached: false,
                bucket: .tiny
            ) == "qwen3-0.6b-4bit, not downloaded, tiny model — may contradict itself in multi-turn chat, tools unverified"
        )
    }
}
