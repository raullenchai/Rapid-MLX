import Foundation

/// Pure RAM tier → curated model recommendation for "what should this
/// Mac run". Replaces the old 6-bucket × 5-role matrix (Default / Speed /
/// Quality / Coding / Vision) with a much simpler table: per RAM tier, a
/// **smart** pick (the most capable model that fits) and, when a genuinely
/// faster/lighter model is worth surfacing, a **fast** alternative — plus
/// the exact launch flags each pick needs.
///
/// ## Why the roles went away
///
/// The role matrix asked the user to reason about "do I want Speed or
/// Quality or Coding?" before running anything — five rows, most of them
/// duplicated across buckets, several needing footnotes about which alias
/// mis-parses tool calls. The signal that actually decides what a Mac can
/// run is its RAM. So the recommendation is now: read the RAM, show the
/// smartest model we'd run there and — where it helps — a faster/lighter
/// second option. That's it.
///
/// ## Tiers (keyed by machine RAM, rounded DOWN to the nearest floor)
///
/// A machine with N GB gets the tier whose floor is the largest value
/// ≤ N — so a 20 GB Mac gets the 18 GB pick (which fits), NOT the 24 GB
/// pick (which would be borderline). Sub-16 GB Macs clamp to the 16 GB
/// tier; the app's per-row fit warning still flags anything too big, and
/// that tier's fast alternative does fit 8 GB.
///
/// A tier only carries a **fast** alt when it beats the smart pick on
/// speed by a margin worth a second card. The 24 & 32 GB smart picks are
/// already MoE-fast (42 / 60 tok/s), so they stand alone — a slower or
/// much weaker "fast" option there would only mislead.
///
/// | RAM    | 🧠 Smart            | GB   | Cap | tok/s | 🚀 Fast                          |
/// | ------ | ------------------- | ---- | --- | ----- | -------------------------------- |
/// | 16 GB  | bonsai-27b-2bit     | 7.6  | 86% | 17.1  | lfm2.5-8b-a1b-4bit · 117 · chat  |
/// | 18 GB  | bonsai-27b-2bit     | 7.6  | 86% | 17.1  | lfm2.5-8b-a1b-4bit · 117 · chat  |
/// | 24 GB  | gemma-4-26b-4bit    | 14.6 | 87% | 41.7  | — (smart pick already fast)      |
/// | 32 GB  | qwen3.6-35b-4bit    | 20.0 | 87% | 60.0  | — (smart pick already fast)      |
/// | 64 GB  | qwen3.6-35b-8bit    | 37.7 | 87% | —     | qwen3.6-35b-4bit · 87% · 60      |
/// | 96 GB+ | qwen3.5-122b-mxfp4  | 65.0 | 88% | —     | qwen3.6-35b-4bit · 87% · 60      |
///
/// 18 GB deliberately mirrors 16 GB (see the tier comment). The 16/18 GB
/// fast pick is ``lfm2.5-8b-a1b-4bit`` — a chat specialist, so it shows
/// "Chat only" instead of its blended 62 % (which understates conversation
/// and overstates tools/coding). Capability % and tok/s are the
/// maintainer's measured scores (M2/M3); the 64/96 GB smart rows have no
/// local tok/s measurement yet (rendered without a speed figure). The
/// capability column is monotonic non-decreasing by RAM, with ONE
/// deliberate tie documented at the tier: the 64 GB smart 8-bit is floored
/// at its own faster 4-bit alt's 87 % (an 8-bit quant can't display weaker
/// than its 4-bit; its edge is fidelity, not bench points). Every alias is
/// verified to exist in the bundled ``aliases.json`` by
/// ``RAMBucketedDefaultTests``.
enum RAMBucketedDefault {
    /// One recommended model for a RAM tier: the alias, the numbers the
    /// picker shows, and the launch flags it needs to fit/run on that
    /// tier's RAM (e.g. ``--no-mllm`` to drop the vision tower).
    struct Pick: Sendable, Equatable {
        let alias: String
        /// Active-memory footprint in GB (what the model actually uses).
        let footprintGB: Double
        /// Blended capability score 0–100 (tool / coding / reasoning /
        /// general) — the single number that ranks picks.
        let capabilityPct: Int
        /// Measured decode tok/s, or ``nil`` when there is no local
        /// measurement for this tier yet (the largest tiers).
        let tokensPerSec: Double?
        /// Extra ``rapid-mlx serve`` flags this pick needs on its tier,
        /// applied only when the alias is started AS the recommendation
        /// for a Mac at this RAM (see ``launchFlags(forAlias:physicalRAMGB:)``).
        let launchFlags: [String]
        /// An honest one-word caveat shown INSTEAD OF ``capabilityPct`` on
        /// the card (e.g. "Chat only" for ``lfm2.5-8b-a1b-4bit``, a chat
        /// specialist whose blended 62 % understates conversation quality
        /// while overstating tools/coding). ``nil`` for a general-purpose
        /// pick, which shows its capability % as usual.
        var caveat: String? = nil
    }

    /// A RAM tier: the ``floorGB`` it applies from (up to the next tier),
    /// its ``primary`` (smart) pick, and an optional ``alt`` (fast/light)
    /// pick — present only when a faster model is worth a second card.
    struct Tier: Sendable, Equatable {
        let floorGB: Double
        let primary: Pick
        let alt: Pick?

        /// Smart pick first, then the fast alt if present.
        var picks: [Pick] { alt.map { [primary, $0] } ?? [primary] }
    }

    /// The fast/light pick shared by every tier whose smart pick is slow:
    /// an 8B-A1B MoE at ~117 tok/s. A chat specialist, so it carries a
    /// "Chat only" caveat instead of its (misleadingly low) blended score.
    private static let lfm2FastPick = Pick(
        alias: "lfm2.5-8b-a1b-4bit", footprintGB: 4.8, capabilityPct: 62,
        tokensPerSec: 117.3, launchFlags: [], caveat: "Chat only")

    /// The fast/light pick for the big-MoE tiers: the 4-bit Qwen3.6-35B
    /// (same weights as the 32 GB smart pick) — near-equal capability to
    /// the tier's smart model but much faster than an 8-bit / 122B load.
    private static let qwen35bFastPick = Pick(
        alias: "qwen3.6-35b-4bit", footprintGB: 20.0, capabilityPct: 87,
        tokensPerSec: 60.0, launchFlags: [])

    /// Source of truth — ascending by ``floorGB``. A recommendation change
    /// is a one-line edit here, verified by ``RAMBucketedDefaultTests`` and
    /// the standalone ``scripts/verify-recommendation-tiers.swift`` contract
    /// check against the bundled ``aliases.json``.
    static let tiers: [Tier] = [
        Tier(
            floorGB: 16,
            primary: Pick(alias: "bonsai-27b-2bit", footprintGB: 7.6, capabilityPct: 86, tokensPerSec: 17.1, launchFlags: []),
            alt: lfm2FastPick
        ),
        // 18 GB intentionally MIRRORS the 16 GB tier (bonsai smart + lfm2.5
        // fast). An 18 GB Mac has no headroom for a meaningfully stronger
        // model than bonsai-27b-2bit that we'd trust, and the gemma-4-12b
        // that used to sit here read WEAKER (72 %) than 16 GB's bonsai
        // (86 %) — a "more RAM, worse pick" dip. Rather than ship that
        // inversion we keep bonsai here too; the tier stays explicit (not
        // folded into 16 GB) so a future 18 GB-specific pick is a one-line
        // edit. gemma-4-12b remains available in the full "All models" list.
        Tier(
            floorGB: 18,
            primary: Pick(alias: "bonsai-27b-2bit", footprintGB: 7.6, capabilityPct: 86, tokensPerSec: 17.1, launchFlags: []),
            alt: lfm2FastPick
        ),
        // 24 & 32 GB: the smart pick is already MoE-fast (42 / 60 tok/s),
        // so it stands alone — a slower or much-weaker "fast" card here
        // would only mislead.
        Tier(
            floorGB: 24,
            primary: Pick(
                alias: "gemma-4-26b-4bit", footprintGB: 14.6, capabilityPct: 87, tokensPerSec: 41.7,
                launchFlags: ["--no-mllm", "--kv-cache-dtype", "bf16", "--cache-memory-mb", "512"]),
            alt: nil
        ),
        Tier(
            floorGB: 32,
            primary: Pick(alias: "qwen3.6-35b-4bit", footprintGB: 20.0, capabilityPct: 87, tokensPerSec: 60.0, launchFlags: []),
            alt: nil
        ),
        // 64 GB smart pick is the 8-bit of the same Qwen3.6-35B whose 4-bit
        // is the fast alt. Its capability is floored at the 4-bit's measured
        // 87 % — an 8-bit quant is strictly higher-fidelity than its own
        // 4-bit, so it must never DISPLAY below it (an earlier 85 % estimate
        // made the "Best pick" read as weaker than its "Faster" alt). We
        // pin equality, not a fabricated margin: the 8-bit's edge is
        // quantization fidelity on long / hard prompts, which the blended
        // bench doesn't fully resolve, and the alt exists precisely for
        // users who'd rather trade that fidelity for the 4-bit's speed.
        Tier(
            floorGB: 64,
            primary: Pick(alias: "qwen3.6-35b-8bit", footprintGB: 37.7, capabilityPct: 87, tokensPerSec: nil, launchFlags: []),
            alt: qwen35bFastPick
        ),
        Tier(
            floorGB: 96,
            primary: Pick(alias: "qwen3.5-122b-mxfp4", footprintGB: 65.0, capabilityPct: 88, tokensPerSec: nil, launchFlags: []),
            alt: qwen35bFastPick
        ),
    ]

    /// The tier a Mac with ``physicalRAMGB`` lands in: the highest floor
    /// ≤ RAM. A sub-16 GB Mac clamps to the smallest tier.
    static func tier(forPhysicalRAMGB physicalRAMGB: Double) -> Tier {
        var chosen = tiers[0]
        for candidate in tiers where physicalRAMGB >= candidate.floorGB {
            chosen = candidate
        }
        return chosen
    }

    /// Primary alias for a Mac — kept for the first-launch /
    /// ``ServerManager`` callers that need a single default alias.
    static func alias(forPhysicalRAMGB physicalRAMGB: Double) -> String {
        tier(forPhysicalRAMGB: physicalRAMGB).primary.alias
    }

    /// The picks (primary, then optional alt) shown in the picker's
    /// "Recommended for your N GB Mac" section.
    static func picks(forPhysicalRAMGB physicalRAMGB: Double) -> [Pick] {
        tier(forPhysicalRAMGB: physicalRAMGB).picks
    }

    /// Launch flags to apply when starting ``alias`` AS the recommended
    /// model for a Mac at ``physicalRAMGB`` — empty unless the alias is
    /// this Mac's primary or alt pick. This is why a 64 GB Mac that hand-
    /// picks ``gemma-4-26b-4bit`` (the 24 GB tier's pick, not its own)
    /// keeps vision: the flags (``--no-mllm`` …) only ride along with the
    /// recommendation on the tier they were curated for.
    static func launchFlags(forAlias alias: String, physicalRAMGB: Double) -> [String] {
        tier(forPhysicalRAMGB: physicalRAMGB)
            .picks
            .first(where: { $0.alias == alias })?
            .launchFlags ?? []
    }

    /// Is ``alias`` a pick for a Mac that genuinely SITS IN its tier (RAM
    /// ≥ the tier floor)? The picker trusts the curated table's measured
    /// footprints over ``ModelSizing``'s heuristic estimate, which over-
    /// states low-bit / MoE models (it scores the real-7.6 GB
    /// ``bonsai-27b-2bit`` as ~14.8 GB and flags it ``.tooBig`` on a
    /// 16 GB Mac). So starting a recommended pick skips the ``.tooBig``
    /// "Start anyway" gate — the table already vetted it fits this tier.
    ///
    /// The floor guard matters for the sub-16 GB clamp: an 8 GB Mac is
    /// SHOWN the 16 GB tier's picks, but it does NOT sit in that tier, so
    /// its picks stay subject to the ``.tooBig`` gate — bypassing it there
    /// would re-open the OOM hole (bonsai's 7.6 GB exceeds an 8 GB Mac's
    /// usable pool).
    static func isRecommendedPick(alias: String, physicalRAMGB: Double) -> Bool {
        let t = tier(forPhysicalRAMGB: physicalRAMGB)
        return physicalRAMGB >= t.floorGB && t.picks.contains { $0.alias == alias }
    }
}

extension MacHardware {
    /// Primary recommended alias for this Mac's RAM — the first-launch /
    /// picker fallback default. (Quickstart still handles the true
    /// first-touch with the small bundled model; this is the RAM-tier
    /// fallback for the "nothing cached" case.)
    var bucketedDefaultAlias: String {
        RAMBucketedDefault.alias(forPhysicalRAMGB: physicalRAMGB)
    }

    /// Recommended picks (primary, then optional fast alternative) shown
    /// as rows in the "Recommended for your N GB Mac" section — the single
    /// source of truth for both the picker and the Model Management panel,
    /// so they can never drift.
    var recommendedPicks: [RAMBucketedDefault.Pick] {
        RAMBucketedDefault.picks(forPhysicalRAMGB: physicalRAMGB)
    }
}

/// Codex r2 BLOCKING on PR #165. When the bucketed alias is rejected
/// as ``.tooBig`` AND ``sortedRecommended`` is empty (the hardware-
/// floor case: 8 GB Mac, or a future ultra-low-RAM iPad-class
/// device), the picker's original "cached.first ?? catalog.first"
/// fallback could silently re-pick a ``.tooBig`` alias if the user
/// happened to have a large model cached from a previous session.
/// That reintroduces the OOM the codex r1 fix was supposed to close.
///
/// This helper walks the catalog in three priority steps:
///
///   1. **Cached + not-`.tooBig`.** Instant boot, safe to run.
///   2. **Not-`.tooBig` by smallest footprint.** May not be cached
///      but is at least within the ModelSizing budget.
///   3. **Smallest catalog entry overall.** Hardware-floor escape —
///      every alias is `.tooBig` on this Mac, so we hand back the
///      smallest one so the picker still has SOMETHING to default
///      to. The UI will mark the row borderline/warning per its
///      existing per-row classification; this helper just avoids
///      handing back a 122B alias when a 4B alias is sitting next
///      to it.
///
/// Tested directly in ``RAMBucketedDefaultTests`` via a synthetic
/// catalog + hardware fixture; the private ``recommendedDefault``
/// in ``ModelPickerBar`` would otherwise be impossible to unit test
/// without a SwiftUI driver.
enum SafeDefaultFallback {
    static func pick(catalog: [ModelEntry], hardware: MacHardware) -> String? {
        // Codex r3 BLOCKING on #165: ``ModelSizing.estimate`` returns
        // ``paramsBillions == nil`` for aliases like
        // ``qwen3-coder-4bit`` / ``deepseek-v4-flash-2bit`` /
        // ``glm4.5-air-4bit`` where the parameter count isn't a
        // ``<number>b`` token. Those nil-param footprints come out
        // at totalGB = 0 + 1.2 + 2.0 = 3.2 GB and ``classify`` short-
        // circuits to ``.borderline`` — so a naive "sort by total
        // footprint" would float a 20 B coder model ABOVE the known
        // 5.9 GB 4B as the safest default on an 8 GB Mac. The picker
        // would then offer Start on an alias whose real footprint
        // could be 12+ GB, OOMing the Mac.
        //
        // Partition into "known params" and "unknown params" up front
        // and rank known first. Unknown-params aliases are only the
        // absolute last resort (catalog contains nothing else).
        let known = catalog.filter {
            ModelSizing.estimate(alias: $0.alias).paramsBillions != nil
        }
        let unknown = catalog.filter {
            ModelSizing.estimate(alias: $0.alias).paramsBillions == nil
        }

        // Step 1: cached + safe (only over the known-params subset).
        if let safeCached = known.filter(\.cached)
            .first(where: { isSafe($0, on: hardware) }) {
            return safeCached.alias
        }
        // Step 2: smallest known-params alias that fits.
        let knownBySize = known.sorted { lhs, rhs in
            ModelSizing.estimate(alias: lhs.alias).totalGB
                < ModelSizing.estimate(alias: rhs.alias).totalGB
        }
        if let safe = knownBySize.first(where: { isSafe($0, on: hardware) }) {
            return safe.alias
        }
        // Step 3: smallest known-params alias overall (.tooBig but
        // we know what we're getting).
        if let smallestKnown = knownBySize.first {
            return smallestKnown.alias
        }
        // Step 4: catalog has zero parseable aliases. Last resort —
        // hand back the first unknown-params row so the picker has
        // SOMETHING. In practice this branch is unreachable on the
        // real Rapid-MLX catalog (every alias carries a `<n>b` token
        // somewhere) but we keep it for forward-compat with future
        // custom-alias entries the user types in by hand.
        return unknown.first?.alias
    }

    private static func isSafe(_ entry: ModelEntry, on hardware: MacHardware) -> Bool {
        ModelSizing.classify(ModelSizing.estimate(alias: entry.alias), on: hardware) != .tooBig
    }
}

/// Issue #436: pick a fresh-launch picker default that prefers a
/// cached-and-fits alias when the RAM-bucketed default isn't on
/// disk yet. Closes the post-Quickstart UX cliff where a 256 GB
/// M3 Ultra user with ``bonsai-1.7b-2bit`` already pulled to the
/// HF cache still saw a 4.4 GB "Download & start qwen3.6-35b-4bit"
/// CTA — the Quickstart promise ("5-second time-to-first-token")
/// silently lost to the RAM-bucketed default because
/// ``ModelPickerBar.recommendedDefault`` consulted the bucket
/// table without ever checking ``ModelEntry.cached``.
///
/// ## Decision ladder
///
/// 1. **Bucketed default is in catalog AND cached AND fits.** Use it.
///    No surprise — when the high-quality canonical pick is already
///    on disk, the user gets it with zero download (the Quickstart
///    "5-second time-to-first-token" instant-start experience). This
///    is about download cost, not about matching any site table —
///    the canonical pick is defined by this file (see type docstring).
/// 2. **Bucketed isn't cached but ≥1 cached-and-fits alias exists.**
///    Prefer the cached alternative. The user paid for those bytes
///    already; surfacing the 4.4 GB CTA when a runnable model is
///    sitting two clicks away (open picker → pick row → start) is
///    the exact UX cliff the issue documents.
/// 3. **Bucketed default is in catalog AND fits.** Use it. Nothing
///    cached fits, but the canonical pick still runs on this Mac —
///    legacy behaviour preserved.
/// 4. **Bucketed default is ``.tooBig`` OR missing from catalog.**
///    Delegate to ``SafeDefaultFallback`` (the codex r2/r3 hardware-
///    floor escape from #165 — never returns a ``.tooBig`` alias
///    when a smaller one is available, and partitions out
///    unparseable aliases so a coder/flash quant doesn't phantom-
///    classify as small).
///
/// ## Why alphabetical tie-break inside the cached set
///
/// ``localizedStandardCompare`` mirrors ``ModelCatalog.load``'s
/// post-cached sort and ``AutoStartDecision.resolveAlias``'s
/// first-cached fallback — three surfaces, one tie-break. A
/// 256 GB Mac with both ``bonsai-1.7b-2bit`` and ``qwen3.6-35b-4bit``
/// cached would land on ``bonsai-1.7b-2bit`` (alphabetically first),
/// instant-boot. The user swaps via the picker the moment they
/// want the larger model — same affordance as today, but
/// without paying a multi-GB download just to swap back to a
/// model that was already on disk.
///
/// "Smallest by footprint" was considered and rejected — on a
/// catalog where every cached alias has a real ``<n>b`` token the
/// two heuristics converge for the dominant Quickstart case
/// (bonsai-1.7b-2bit wins on both), and alphabetical avoids the
/// estimator-overhead the picker doesn't otherwise need on the
/// hot path.
///
/// ## Why we filter out unparseable-params aliases for step 2
///
/// ``ModelSizing.estimate(alias:).paramsBillions == nil`` (e.g.
/// ``qwen3-coder-4bit`` / ``deepseek-v4-flash-2bit``) phantom-
/// classifies as ``.borderline`` everywhere — see the codex r3
/// rationale on ``SafeDefaultFallback``. If we let those into the
/// cached-and-fits set, a 16 GB Mac with a stray cached coder
/// quant would default to it and silently OOM on Start. The
/// partition mirrors ``SafeDefaultFallback.pick``'s own
/// known-params filter so both helpers refuse the same trap.
///
/// ## Pure-function contract
///
/// Inputs are values, no FS / sysctl / catalog probes. Caller
/// (``ModelPickerBar``) computes ``catalog`` from
/// ``ModelCatalog.load`` and ``hardware`` from ``MacHardware.detect``
/// once and threads the snapshot in. Tested directly in
/// ``CacheAwareDefaultTests``.
enum CacheAwareDefault {
    static func pick(
        catalog: [ModelEntry],
        hardware: MacHardware,
        bucketedDefault: String
    ) -> String? {
        let bucketedEntry = catalog.first(where: { $0.alias == bucketedDefault })
        // The RAM-tier primary is trusted to fit even when ModelSizing's
        // heuristic over-states it (it scores the real-7.6 GB
        // bonsai-27b-2bit as ~14.8 GB → .tooBig on 16 GB). Without this
        // the default falls through to an unrelated fallback instead of
        // the tier's own recommended primary.
        //
        // Scoped INSIDE `bucketedEntry.map` so `bucketedFits` is only ever
        // true for an alias that actually exists in this engine's catalog:
        // on a version skew where the RAM table names an alias the bundled
        // engine doesn't ship, `bucketedEntry == nil` → `bucketedFits ==
        // false` → we fall through to `SafeDefaultFallback` rather than
        // handing back an alias the engine would refuse to serve. (Both
        // return sites below already guard on `bucketedEntry != nil`, so
        // this is defence-in-depth against a future refactor, not a live
        // bug — but keeping the exemption catalog-scoped makes the
        // invariant local and obvious.)
        let bucketedFits = bucketedEntry.map {
            RAMBucketedDefault.isRecommendedPick(
                alias: bucketedDefault, physicalRAMGB: hardware.physicalRAMGB)
                || isSafe($0, on: hardware)
        } ?? false

        // Step 1: bucketed default is on disk AND runnable. No
        // surprise — high-quality canonical pick wins.
        if let entry = bucketedEntry, entry.cached, bucketedFits {
            return bucketedDefault
        }

        // Step 2: any cached + fits alternative — prefer it over a
        // bucketed alias that would trigger a multi-GB pull. Filter
        // out unparseable-params aliases so a cached coder/flash
        // quant doesn't phantom-classify as small and land us in an
        // OOM (codex r3 on #165).
        let cachedCandidates = catalog
            .filter { $0.cached }
            .filter { ModelSizing.estimate(alias: $0.alias).paramsBillions != nil }
            .filter { isSafe($0, on: hardware) }
        if let pick = firstAlphabetical(cachedCandidates) {
            return pick
        }

        // Step 3: nothing cached fits, but the canonical pick still
        // runs on this Mac → legacy bucketed-default branch.
        if bucketedEntry != nil, bucketedFits {
            return bucketedDefault
        }

        // Step 4: bucketed is .tooBig OR missing — hand off to the
        // existing hardware-floor escape so we never silently
        // promote a .tooBig cached alias above a safe one.
        return SafeDefaultFallback.pick(catalog: catalog, hardware: hardware)
    }

    private static func isSafe(_ entry: ModelEntry, on hardware: MacHardware) -> Bool {
        ModelSizing.classify(
            ModelSizing.estimate(alias: entry.alias),
            on: hardware
        ) != .tooBig
    }

    private static func firstAlphabetical(_ entries: [ModelEntry]) -> String? {
        entries
            .map(\.alias)
            .sorted(by: { $0.localizedStandardCompare($1) == .orderedAscending })
            .first
    }
}
