import Foundation
import Testing
@testable import Rapid

/// cycle-7: pin the sub-1B picker filter so a future alias-naming
/// drift (or a regression in ``ModelSizing.parseParamsBillions``)
/// can't silently re-surface tiny models to the dropdown. The bug
/// (`bug_report.md` cycle-0 P2) was: the rapid-mlx catalog ships 92
/// aliases including `qwen3-0.6b-4bit` / `qwen3-0.6b-8bit`, which
/// hallucinate within 1-2 turns of chat and read as "the app is
/// broken" to first-time users.
///
/// Threshold is **inclusive** at 1.0B (a 1B alias like
/// `gemma3-1b-qat-4bit` / `llama3-1b-4bit` IS shown). Parse failures
/// default to "show" so legitimate custom aliases the user types in
/// don't get accidentally swallowed.
@Suite("ModelPickerVisibility — sub-1B filter (cycle-7)")
struct ModelPickerVisibilityTests {

    // MARK: - shouldShow

    @Test("0.6B (qwen3-0.6b-4bit) is HIDDEN when includeAll is OFF")
    func sub1bHiddenByDefault() {
        #expect(
            ModelPickerVisibility.shouldShow(
                alias: "qwen3-0.6b-4bit",
                selectedAlias: "qwen3.5-4b-4bit",
                includeAll: false
            ) == false
        )
    }

    @Test("0.6B 8-bit variant is also HIDDEN — both qwen3-0.6b SKUs need to drop")
    func sub1b8bitHiddenByDefault() {
        #expect(
            ModelPickerVisibility.shouldShow(
                alias: "qwen3-0.6b-8bit",
                selectedAlias: "qwen3.5-4b-4bit",
                includeAll: false
            ) == false
        )
    }

    @Test("4B alias (qwen3.5-4b-4bit, project default test model) is SHOWN")
    func defaultTestModelShown() {
        #expect(
            ModelPickerVisibility.shouldShow(
                alias: "qwen3.5-4b-4bit",
                selectedAlias: "",
                includeAll: false
            ) == true
        )
    }

    @Test("1.0B boundary is SHOWN — gemma3-1b-qat-4bit at the inclusive lower bound")
    func boundary1bShown() {
        #expect(
            ModelPickerVisibility.shouldShow(
                alias: "gemma3-1b-qat-4bit",
                selectedAlias: "",
                includeAll: false
            ) == true
        )
    }

    @Test("1.0B boundary — llama3-1b-4bit is SHOWN (also at the inclusive bound)")
    func boundary1bLlamaShown() {
        #expect(
            ModelPickerVisibility.shouldShow(
                alias: "llama3-1b-4bit",
                selectedAlias: "",
                includeAll: false
            ) == true
        )
    }

    @Test("1.7B (bonsai-1.7b-unpacked) is SHOWN — above threshold")
    func bonsai17bShown() {
        #expect(
            ModelPickerVisibility.shouldShow(
                alias: "bonsai-1.7b-unpacked",
                selectedAlias: "",
                includeAll: false
            ) == true
        )
    }

    @Test("vibethinker-1.5b is SHOWN — above threshold")
    func vibethinker15bShown() {
        #expect(
            ModelPickerVisibility.shouldShow(
                alias: "vibethinker-1.5b-4bit",
                selectedAlias: "",
                includeAll: false
            ) == true
        )
    }

    @Test("includeAll = true unhides sub-1B")
    func includeAllShowsTiny() {
        #expect(
            ModelPickerVisibility.shouldShow(
                alias: "qwen3-0.6b-4bit",
                selectedAlias: "",
                includeAll: true
            ) == true
        )
    }

    @Test("Currently-selected sub-1B alias is exempt from the filter (don't hide what the user has picked)")
    func selectedSub1bAlwaysShown() {
        // User somehow picked qwen3-0.6b (e.g. via the bundled
        // first-launch path or via Type custom alias). The dropdown
        // must still surface it so they can identify the row.
        #expect(
            ModelPickerVisibility.shouldShow(
                alias: "qwen3-0.6b-4bit",
                selectedAlias: "qwen3-0.6b-4bit",
                includeAll: false
            ) == true
        )
    }

    @Test("Parse-failure aliases (no N b / N m token) return nil from the parser AND default to SHOWN — codex r1 MINOR split")
    func parseFailureDefaultsToShown() {
        // True parse failures — no `\d+(\.\d+)?[bBmM]` token at all.
        // The parser MUST return nil; the visibility check MUST then
        // default to true.
        let parseMisses = [
            "phi-3.5-mini-4bit",       // no N-b / N-m token outside the bit-width
            "my-custom-alias",
            "x",
            "gemma3n",                 // 3n is digits-letter but not b / m
        ]
        for alias in parseMisses {
            #expect(
                ModelPickerVisibility.parseSmallestParamsBillions(alias) == nil,
                "alias \(alias) should NOT parse to a size (true parse failure)"
            )
            #expect(
                ModelPickerVisibility.shouldShow(
                    alias: alias,
                    selectedAlias: "",
                    includeAll: false
                ) == true,
                "alias \(alias) should be shown via parse-failure → default-to-shown branch"
            )
        }
    }

    @Test("Parsed-but-large aliases (16b / 20b) are also shown — different branch from parse-failure")
    func parsedLargeAliasesShown() {
        // These DO parse to a size; they should be shown because
        // the size is >= 1B, not because of the nil-default branch.
        let large: [(String, Double)] = [
            ("deepseek-coder-v2-lite-16b-4bit", 16.0),
            ("gpt-oss-20b-mxfp4-q8", 20.0),
        ]
        for (alias, expected) in large {
            #expect(
                ModelPickerVisibility.parseSmallestParamsBillions(alias) == expected,
                "alias \(alias) should parse to \(expected)B"
            )
            #expect(
                ModelPickerVisibility.shouldShow(
                    alias: alias,
                    selectedAlias: "",
                    includeAll: false
                ) == true
            )
        }
    }

    // MARK: - Million-scale suffix (Nm / NM) — codex r1 MINOR

    @Test("Hypothetical million-scale alias (qwen3-600m-4bit) is HIDDEN — sub-1B via the N-m branch")
    func millionScaleSub1bHidden() {
        // No alias ships this way today, but rapid-mlx COULD rename
        // the tinies to the `-Nm` convention upstream. We pre-empt
        // that by parsing million-scale identifiers and normalising
        // to billions (600m → 0.6B → < 1.0B → hidden).
        #expect(ModelPickerVisibility.parseSmallestParamsBillions("qwen3-600m-4bit") == 0.6)
        #expect(
            ModelPickerVisibility.shouldShow(
                alias: "qwen3-600m-4bit",
                selectedAlias: "",
                includeAll: false
            ) == false
        )
    }

    @Test("Uppercase million suffix (smollm-135M) is also parsed — case-insensitive")
    func millionScaleUppercase() {
        #expect(ModelPickerVisibility.parseSmallestParamsBillions("smollm-135M") == 0.135)
        #expect(
            ModelPickerVisibility.shouldShow(
                alias: "smollm-135M",
                selectedAlias: "",
                includeAll: false
            ) == false
        )
    }

    @Test("Million-scale at 1000m (1B equivalent) is SHOWN — boundary stays inclusive")
    func millionScaleBoundary1000m() {
        #expect(ModelPickerVisibility.parseSmallestParamsBillions("custom-1000m-4bit") == 1.0)
        #expect(
            ModelPickerVisibility.shouldShow(
                alias: "custom-1000m-4bit",
                selectedAlias: "",
                includeAll: false
            ) == true
        )
    }

    @Test("minimax-m2.7-mxfp4 / minimax-m2.5-4bit: 'm' is the family-version letter, NOT a million suffix — parse miss → shown (codex r2 MINOR pin)")
    func minimaxFamilyLetterNotMillionSuffix() {
        // The regex is number-then-suffix (`\d+(\.\d+)?[mM]\b`), so
        // `m2.7` (suffix-then-number) is correctly NOT matched. The
        // visibility helper then defaults to "shown" via the
        // parse-failure branch. Result: MiniMax aliases (235B MoE
        // family) are surfaced, matching the user's expectation.
        // Pinning this so a future regex change that adds
        // suffix-then-number matching (`m\d+(\.\d+)?`) doesn't
        // silently mis-classify these aliases as 2.7B → 2.5B → 2B
        // and break the family-letter convention.
        for alias in ["minimax-m2.7-mxfp4", "minimax-m2.5-4bit"] {
            #expect(
                ModelPickerVisibility.parseSmallestParamsBillions(alias) == nil,
                "alias \(alias) should NOT parse to a size — 'm' is the family letter, not a million suffix"
            )
            #expect(
                ModelPickerVisibility.shouldShow(
                    alias: alias,
                    selectedAlias: "",
                    includeAll: false
                ) == true,
                "alias \(alias) should be shown via the parse-failure branch (MiniMax M2.x is a large MoE family, not a sub-1B model)"
            )
        }
    }

    // MARK: - Smallest-match semantics

    @Test("Two N-b tokens → SMALLEST wins (mix-0.6b-2b-moe hidden via 0.6B branch)")
    func smallestWinsForMultiB() {
        // Hypothetical MoE alias with both an active-params and a
        // weights-size token. The filter cares about whether ANY
        // sub-1B identifier is present (any tiny variant = same
        // first-impression risk), so the smallest match wins.
        #expect(ModelPickerVisibility.parseSmallestParamsBillions("mix-0.6b-2b-moe") == 0.6)
        #expect(
            ModelPickerVisibility.shouldShow(
                alias: "mix-0.6b-2b-moe",
                selectedAlias: "",
                includeAll: false
            ) == false
        )
    }

    @Test("Both sized aliases-with-bit-width tokens (qwen3.6-35b-8bit) only see the size token")
    func bitWidthNotMistakenForParams() {
        // `4bit` / `8bit` end in `t`, so `b\b` is NOT a word
        // boundary there — the parser must skip them. Only the real
        // size token contributes.
        #expect(ModelPickerVisibility.parseSmallestParamsBillions("qwen3.6-35b-8bit") == 35.0)
        #expect(ModelPickerVisibility.parseSmallestParamsBillions("qwen3.5-122b-mxfp4") == 122.0)
    }

    @Test("Truly nameless alias is shown — empty / weird names default to shown")
    func namelessAliasShown() {
        #expect(
            ModelPickerVisibility.shouldShow(
                alias: "x",
                selectedAlias: "",
                includeAll: false
            ) == true
        )
    }

    // MARK: - filter()

    @Test("filter drops sub-1B aliases by default and preserves order")
    func filterDropsTiny() {
        let entries = [
            ModelEntry(alias: "qwen3-0.6b-4bit", hfRepo: nil, sizeOnDisk: nil, cached: true),
            ModelEntry(alias: "qwen3-0.6b-8bit", hfRepo: nil, sizeOnDisk: nil, cached: false),
            ModelEntry(alias: "qwen3.5-4b-4bit", hfRepo: nil, sizeOnDisk: nil, cached: true),
            ModelEntry(alias: "bonsai-1.7b-unpacked", hfRepo: nil, sizeOnDisk: nil, cached: false),
            ModelEntry(alias: "gemma3-1b-qat-4bit", hfRepo: nil, sizeOnDisk: nil, cached: true),
        ]
        let filtered = ModelPickerVisibility.filter(
            entries,
            selectedAlias: "qwen3.5-4b-4bit",
            includeAll: false
        )
        #expect(filtered.map(\.alias) == ["qwen3.5-4b-4bit", "bonsai-1.7b-unpacked", "gemma3-1b-qat-4bit"])
    }

    @Test("filter with includeAll = true is a no-op")
    func filterIncludeAllNoOp() {
        let entries = [
            ModelEntry(alias: "qwen3-0.6b-4bit", hfRepo: nil, sizeOnDisk: nil, cached: false),
            ModelEntry(alias: "qwen3.5-4b-4bit", hfRepo: nil, sizeOnDisk: nil, cached: true),
        ]
        let filtered = ModelPickerVisibility.filter(
            entries,
            selectedAlias: "",
            includeAll: true
        )
        #expect(filtered.count == 2)
    }

    @Test("filter keeps the currently-selected sub-1B alias even when filter is on")
    func filterKeepsSelected() {
        let entries = [
            ModelEntry(alias: "qwen3-0.6b-4bit", hfRepo: nil, sizeOnDisk: nil, cached: true),
            ModelEntry(alias: "qwen3-0.6b-8bit", hfRepo: nil, sizeOnDisk: nil, cached: false),
        ]
        let filtered = ModelPickerVisibility.filter(
            entries,
            selectedAlias: "qwen3-0.6b-4bit",
            includeAll: false
        )
        #expect(filtered.map(\.alias) == ["qwen3-0.6b-4bit"])
    }

    // MARK: - parseSmallestParamsBillions sanity (regression-pin the parser our filter rides on)

    @Test("parseSmallestParamsBillions parses 0.6 from qwen3-0.6b-4bit (sanity pin)")
    func parserSanity06b() {
        #expect(ModelPickerVisibility.parseSmallestParamsBillions("qwen3-0.6b-4bit") == 0.6)
    }

    @Test("parseSmallestParamsBillions parses 1.0 from gemma3-1b-qat-4bit (sanity pin)")
    func parserSanity1b() {
        #expect(ModelPickerVisibility.parseSmallestParamsBillions("gemma3-1b-qat-4bit") == 1.0)
    }

    @Test("parseSmallestParamsBillions parses 1.7 from bonsai-1.7b-unpacked (sanity pin)")
    func parserSanity17b() {
        #expect(ModelPickerVisibility.parseSmallestParamsBillions("bonsai-1.7b-unpacked") == 1.7)
    }

    @Test("parseSmallestParamsBillions parses 122 from qwen3.5-122b-8bit — multi-digit guard")
    func parserSanity122b() {
        #expect(ModelPickerVisibility.parseSmallestParamsBillions("qwen3.5-122b-8bit") == 122.0)
    }

    @Test("Mixed-case 'B' parses too (parser is case-insensitive)")
    func parserMixedCase() {
        #expect(ModelPickerVisibility.parseSmallestParamsBillions("custom-7B-mlx") == 7.0)
    }

    // MARK: - cycle-10: quality buckets (F9-004)

    /// Pin the (params → bucket) truth table at every published
    /// boundary so a future change to ``minParamsBillions`` or
    /// ``smallBucketUpperBoundBillions`` can't silently re-shape the
    /// sticker without a test failure.
    ///
    /// cycle-9 fuzz-correct data backed the original sticker bucket:
    /// llama3-1b-4bit failed 8/10 basic arithmetic + ALL multi-turn
    /// coherence probes.
    ///
    /// cycle-11 (F-10-PRESET) tightened the upper bound from
    /// inclusive (``<= 3.0``) to strict (``< 3.0``). cycle-10
    /// empirical data: ``llama3-3b-4bit`` is desktop-viable (5/5
    /// clean weather tool calls vs cycle-9's 3/5 leaked, stable
    /// multi-turn fact recall, 154 tok/s B=1 decode on M3 Ultra),
    /// so 3.0B aliases now land in ``.midOrLarger`` (no sticker)
    /// rather than ``.small`` (sticker). Pinning ``llama3-3b-4bit``
    /// explicitly so a future revert of the boundary trips this gate.
    @Test("Quality bucket boundaries: 0.6B → tiny, 1.0B → small, 1.5B → small, 3.0B → midOrLarger (cycle-11 tighten), 3.5B → midOrLarger, 4.0B → midOrLarger, 7.0B → midOrLarger, 13.0B → midOrLarger")
    func qualityBucketBoundaries() {
        let cases: [(String, ModelPickerVisibility.QualityBucket)] = [
            // < 1B → tiny (already hidden by default, but sticker
            // still fires when surfaced via the "Show all" toggle).
            ("qwen3-0.6b-4bit", .tiny),
            ("gemma3-0.5b-4bit", .tiny),
            // == 1.0B → small (inclusive lower bound of the sticker
            // bucket — matches the cycle-9 llama3-1b-4bit confirmed
            // failure mode: silent tool-call schema-leak + multi-turn
            // fact-flipping).
            ("llama3-1b-4bit", .small),
            ("gemma3-1b-qat-4bit", .small),
            // 1.5B / 1.7B → small
            ("vibethinker-1.5b-4bit", .small),
            ("bonsai-1.7b-unpacked", .small),
            // == 3.0B → midOrLarger (cycle-11 strict upper bound —
            // 3B is desktop-viable for llama-family chat per cycle-10
            // measurement). This is the most important regression
            // pin in this table: a revert to ``<= 3.0`` would flip
            // both rows below to ``.small`` and trip this test.
            ("llama3-3b-4bit", .midOrLarger),
            ("custom-3b-4bit", .midOrLarger),
            // > 3B → midOrLarger (no sticker, clean row)
            ("custom-3.5b-4bit", .midOrLarger),
            ("qwen3.5-4b-4bit", .midOrLarger),
            ("custom-7b-4bit", .midOrLarger),
            ("custom-13b-4bit", .midOrLarger),
            ("qwen3.6-35b-8bit", .midOrLarger),
        ]
        for (alias, expectedBucket) in cases {
            #expect(
                ModelPickerVisibility.qualityBucket(for: alias) == expectedBucket,
                "alias \(alias) should land in bucket \(expectedBucket)"
            )
        }
    }

    @Test("cycle-11 F-10-PRESET regression pin — llama3-3b-4bit is the smallest llama-family chat-viable alias; it MUST land in .midOrLarger (no 'tiny' sticker) so users see it as a viable smallest pick rather than another can't-trust-it tiny. A revert to <= 3B trips this test.")
    func qualityBucketLlama3BIsMidOrLarger() {
        #expect(
            ModelPickerVisibility.qualityBucket(for: "llama3-3b-4bit")
                == .midOrLarger,
            "llama3-3b-4bit must NOT be in .small — cycle-10 verified 5/5 clean tool calls, stable multi-turn, 154 tok/s decode; it is the smallest viable llama-family chat preset and must NOT carry the discouraging 'tiny' sticker."
        )
        // Sister 1B alias: still .small (cycle-9 confirmed broken
        // tool-call args + multi-turn contradictions). Pinning both
        // sides of the boundary in one test makes the cycle-11 intent
        // unambiguous.
        #expect(
            ModelPickerVisibility.qualityBucket(for: "llama3-1b-4bit")
                == .small,
            "llama3-1b-4bit must stay in .small — cycle-9 verified 3/5 tool-call schema-leak + multi-turn fact-flipping."
        )
    }

    @Test("MoE aliases with both a total-size and an active-band token (qwen3-coder-next-80b-a3b, qwen3.6-35b-a3b-mxfp4) land in .midOrLarger — sticker reads the LARGEST token (total params), NOT the smallest (active band) — codex r1 BLOCKING regression pin")
    func qualityBucketMoEAliasesUseLargestToken() {
        // The cycle-7 shouldShow filter reads the SMALLEST token
        // — "any tiny part = tiny first-impression risk." Quality
        // stickering reads the LARGEST token — "any 3B+ total
        // capacity = NOT in the contradicts-itself band" (cycle-11
        // strict bound moved this from 4B+ to 3B+). The two
        // diverge intentionally on A-NB-MoE aliases, where the
        // smallest token is the active inference band (3B) and the
        // largest is the total weight (80B, 35B, …). The sticker
        // must NOT fire on these — an 80B MoE is the opposite of
        // tiny in chat quality terms even though it inferences with
        // only 3B active params per token.
        //
        // codex r1 BLOCKING regression: a previous draft reused
        // ``parseSmallestParamsBillions`` and would have stickered
        // ``qwen3.6-35b-a3b-mxfp4`` as "tiny" (3B active beats 35B
        // total). Switching to ``ModelSizing.parseParamsBillions``
        // (largest-match) is the fix; this test pins the corner
        // case so a future "make the two helpers share a parser"
        // refactor can't silently regress.
        let moeAliases = [
            "qwen3-coder-next-80b-a3b",
            "qwen3.6-35b-a3b-mxfp4",
            "qwen3.6-30b-a3b",
            // Hypothetical sub-1B-active MoE — total 7B → still
            // .midOrLarger because the chat-quality signal is the
            // total weight count.
            "future-7b-a0.5b-moe",
        ]
        for alias in moeAliases {
            #expect(
                ModelPickerVisibility.qualityBucket(for: alias) == .midOrLarger,
                "MoE alias \(alias) should NOT be stickered — total params govern quality, not active band"
            )
        }
    }

    @Test("Parse-failure alias falls into midOrLarger (no sticker) — phi-3.5-mini-4bit is 3.8B in reality but the alias carries no size token")
    func qualityBucketParseFailureMidOrLarger() {
        // phi-3.5-mini-4bit's HF card shows ~3.8B params but the
        // alias string only carries "3.5" (the version) and "4bit"
        // (the quant). The two-pass parser doesn't see a `\d+b\b`
        // token — `4bit` ends in `t` so `b\b` fails — and returns
        // nil. The bucket then defaults to midOrLarger (no sticker)
        // which matches the user's expectation: phi-3.5-mini is in
        // the 3-7B band that we deliberately don't decorate. If a
        // future upstream rename adds "3.8b" to the alias the bucket
        // will flip to .midOrLarger anyway via the > 3.0 branch.
        #expect(
            ModelPickerVisibility.qualityBucket(for: "phi-3.5-mini-4bit")
                == .midOrLarger
        )
        // Custom HF repo with no size token at all → midOrLarger
        // (no sticker; safer to default-clean for user-typed aliases).
        #expect(
            ModelPickerVisibility.qualityBucket(for: "my-custom-alias")
                == .midOrLarger
        )
    }

    @Test("Quality sticker suffix is '· tiny' for .tiny, '· small' for .small (#348 bucket-distinct), nil for .midOrLarger")
    func qualityStickerSuffixPerBucket() {
        // #348: ``.tiny`` and ``.small`` previously both returned
        // "· tiny", collapsing the data-model split (< 1B vs
        // >= 1B && < 3B) into a single visual label. The buckets are
        // derived from different empirical failure modes (sub-1B:
        // silent first-impression risk; 1-3B: cycle-9 confirmed
        // tool-call schema-leak), so the suffix now mirrors the
        // bucket. A future revert that re-collapses them would
        // re-introduce the "visual theatre" gap this fix closed.
        #expect(
            ModelPickerVisibility.qualityStickerSuffix(for: .tiny) == "· tiny"
        )
        #expect(
            ModelPickerVisibility.qualityStickerSuffix(for: .small) == "· small"
        )
        #expect(
            ModelPickerVisibility.qualityStickerSuffix(for: .midOrLarger) == nil
        )
    }

    @Test("Quality sticker tooltip text — 3B+ rows MUST get nil so the cache-state cue still surfaces alone; sub-3B copy explicitly says 'smaller than 3B' (cycle-11 strict upper bound)")
    func qualityStickerTooltipPerBucket() {
        let tooltip = ModelPickerVisibility.qualityStickerTooltip(for: .small)
        #expect(tooltip != nil)
        // Pin the substring user-visible signal so a copy churn that
        // strips the multi-turn-contradiction warning fails this test.
        #expect(tooltip?.contains("multi-turn") == true)
        #expect(tooltip?.contains("qwen3.5-4b") == true)
        // cycle-11 F-10-PRESET: tooltip MUST say "smaller than 3B" so
        // a user reading the row for a 3B alias (no sticker, no
        // tooltip) doesn't see contradictory copy. A revert to the
        // inclusive "3B and smaller" wording would surface a tooltip
        // covering an alias that no longer carries the sticker —
        // breaking the boundary contract.
        #expect(
            tooltip?.contains("smaller than 3B") == true,
            "cycle-11 tooltip must use strict 'smaller than 3B' wording — matches the < 3.0 upper bound"
        )
        #expect(
            tooltip?.contains("3B and smaller") == false,
            "cycle-11 must NOT use the cycle-10 inclusive 'and smaller' wording — boundary moved"
        )
        // .tiny gets the same tooltip — if a tiny alias is somehow
        // surfaced (Show all toggle, selected alias exempt) the
        // warning still applies.
        #expect(
            ModelPickerVisibility.qualityStickerTooltip(for: .tiny) == tooltip
        )
        // 3B-and-above rows: nil so the row falls back to the bare
        // cache cue.
        #expect(
            ModelPickerVisibility.qualityStickerTooltip(for: .midOrLarger) == nil
        )
    }

    @Test("Composed row tooltip layers quality warning over cache hint with a newline; 3B+ shows cache hint alone (cycle-11 strict bound); both empty → empty")
    func qualityRowHelpTextComposed() {
        // sub-3B (>=1B, <3B) + cached → both lines, quality first.
        let smallCached = ModelPickerVisibility.qualityRowHelpText(
            for: .small,
            cacheHint: "Already downloaded"
        )
        #expect(smallCached.contains("multi-turn"))
        #expect(smallCached.hasSuffix("Already downloaded"))
        #expect(smallCached.contains("\n"))
        // 3B+ + uncached → cache hint alone, no quality line.
        let midUncached = ModelPickerVisibility.qualityRowHelpText(
            for: .midOrLarger,
            cacheHint: "Will download on Start"
        )
        #expect(midUncached == "Will download on Start")
        // 3B+ + empty cache hint → empty (SwiftUI `.help("")` is a no-op).
        let midEmpty = ModelPickerVisibility.qualityRowHelpText(
            for: .midOrLarger,
            cacheHint: ""
        )
        #expect(midEmpty == "")
        // sub-3B + empty cache hint → quality line alone.
        let smallNoCache = ModelPickerVisibility.qualityRowHelpText(
            for: .small,
            cacheHint: ""
        )
        #expect(smallNoCache.contains("multi-turn"))
        #expect(smallNoCache.contains("\n") == false)
    }

    @Test("Sticker does NOT change visibility — bucket and shouldShow are independent helpers; a 1B alias is SHOWN AND stickered, a 0.6B alias is HIDDEN by default but stickered if surfaced via Show all")
    func qualityBucketIndependentFromVisibility() {
        // 1B llama → shown + small bucket
        #expect(
            ModelPickerVisibility.shouldShow(
                alias: "llama3-1b-4bit",
                selectedAlias: "",
                includeAll: false
            ) == true
        )
        #expect(
            ModelPickerVisibility.qualityBucket(for: "llama3-1b-4bit")
                == .small
        )
        // 0.6B qwen → hidden by default + tiny bucket. If the user
        // flips Show all OR has the alias selected, the row surfaces
        // AND the sticker reads "tiny".
        #expect(
            ModelPickerVisibility.shouldShow(
                alias: "qwen3-0.6b-4bit",
                selectedAlias: "",
                includeAll: false
            ) == false
        )
        #expect(
            ModelPickerVisibility.qualityBucket(for: "qwen3-0.6b-4bit")
                == .tiny
        )
    }

    // MARK: - known-broken denylist (issue #1367)

    @Test("ministral-3b-4bit is HIDDEN even though 3B clears the size filter — mlx-vlm lane hangs on first chat (#1367)")
    func brokenMinistralHiddenDespiteSize() {
        // 3.0B parses well above the 1B floor, so ONLY the denylist
        // keeps it off the menu. Without the denylist this is the exact
        // reported footgun: pick it, send "hi", spin forever, 0 tokens.
        #expect(
            ModelPickerVisibility.shouldShow(
                alias: "ministral-3b-4bit",
                selectedAlias: "",
                includeAll: false
            ) == false
        )
    }

    @Test("every gemma-4-e2b SKU is HIDDEN — 0/6 incoherent on both lanes, arch-level not quant-level (#1367)")
    func brokenGemmaE2bFamilyHidden() {
        for alias in [
            "gemma-4-e2b-4bit",
            "gemma-4-e2b-6bit",
            "gemma-4-e2b-8bit",
            "gemma-4-e2b-assistant",
        ] {
            #expect(
                ModelPickerVisibility.shouldShow(
                    alias: alias,
                    selectedAlias: "",
                    includeAll: false
                ) == false,
                "\(alias) must be hidden"
            )
        }
    }

    @Test("denylist WINS over includeAll — Show small models must not reveal a hanging model")
    func denylistBeatsIncludeAll() {
        #expect(
            ModelPickerVisibility.shouldShow(
                alias: "ministral-3b-4bit",
                selectedAlias: "",
                includeAll: true
            ) == false
        )
    }

    @Test("denylist WINS over the selected-alias exemption — a stale persisted pick of a broken model is not kept on the menu")
    func denylistBeatsSelected() {
        #expect(
            ModelPickerVisibility.shouldShow(
                alias: "gemma-4-e2b-4bit",
                selectedAlias: "gemma-4-e2b-4bit",
                includeAll: false
            ) == false
        )
    }

    @Test("evidence-bounded: untested e4b / 3n nano SKUs are NOT denylisted — hiding a maybe-working model is its own harm")
    func untestedNanoNotDenylisted() {
        // These clear both the size filter and the denylist, so they
        // stay shown. #1367 measured only e2b + Ministral-3; the set
        // must not creep to models with no failing evidence.
        for alias in ["gemma-4-e4b-4bit", "gemma-3n-e2b-4bit", "gemma-3n-e4b-4bit"] {
            #expect(ModelPickerVisibility.isKnownBroken(alias) == false, "\(alias)")
            #expect(
                ModelPickerVisibility.shouldShow(
                    alias: alias,
                    selectedAlias: "",
                    includeAll: false
                ) == true,
                "\(alias) must stay shown"
            )
        }
    }

    @Test("filter drops denylisted aliases as well as size-hidden ones")
    func denylistAndSizeHiddenBothDrop() {
        let entries = [
            ModelEntry(alias: "ministral-3b-4bit", hfRepo: nil, sizeOnDisk: nil, cached: true),
            ModelEntry(alias: "qwen3-0.6b-4bit", hfRepo: nil, sizeOnDisk: nil, cached: true),
            ModelEntry(alias: "qwen3.5-4b-4bit", hfRepo: nil, sizeOnDisk: nil, cached: true),
        ]
        let filtered = ModelPickerVisibility.filter(
            entries,
            selectedAlias: "qwen3.5-4b-4bit",
            includeAll: false
        )
        // Broken (ministral) AND size-hidden (0.6b) both drop; only the
        // 4B survives.
        #expect(filtered.map(\.alias) == ["qwen3.5-4b-4bit"])
        // The two are hidden for DIFFERENT reasons, and only one is
        // recoverable: flipping "Show small models" brings back the 0.6b,
        // while ministral stays out however the toggle is set. Pinned here
        // because the filter is now the only thing that distinguishes
        // them — the picker no longer prints a count that could say so.
        let unfiltered = ModelPickerVisibility.filter(
            entries,
            selectedAlias: "qwen3.5-4b-4bit",
            includeAll: true
        )
        #expect(unfiltered.map(\.alias).contains("qwen3-0.6b-4bit"))
        #expect(!unfiltered.map(\.alias).contains("ministral-3b-4bit"))
    }
}
