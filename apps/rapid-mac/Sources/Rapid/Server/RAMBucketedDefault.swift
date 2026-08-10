import Foundation

/// Pure RAM tier → curated model recommendation for "what should this
/// Mac run". Replaces the old 6-bucket × 5-role matrix (Default / Speed /
/// Quality / Coding / Vision) with a much simpler table: per RAM tier, a
/// **smart** pick (the most capable model that fits) and a **fast** alternative — plus
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
/// pick (which would be borderline). Sub-8 GB Macs clamp to the 8 GB
/// tier; the app's per-row fit warning still flags anything too big.
///
/// Every tier carries exactly two choices. This keeps the decision stable:
/// choose responsiveness, or trade some responsiveness for capability.
///
/// | RAM    | 🧠 Smart            | GB   | Cap | tok/s | 🚀 Fast                          |
/// | ------ | ------------------- | ---- | --- | ----- | -------------------------------- |
/// |  8 GB  | lfm2.5-2.6b-4bit    |  3.0 | 64% | 93.5  | lfm2.5-1b-4bit · basic · 208.4   |
/// | 16 GB  | qwen3.5-4b-4bit     |  6.0 | 78% | 60.7  | lfm2.5-1b-4bit · basic · 208.4   |
/// | 18 GB  | qwen3.5-9b-4bit     |  8.7 | 82% | 35.7  | qwen3.5-4b-4bit · 78% · 60.7     |
/// | 24 GB  | bonsai-27b-2bit     | 13.0 | 86% | 17.5  | qwen3.5-4b-4bit · 78% · 60.7     |
/// | 32 GB  | gemma-4-26b-4bit    | 17.0 | 87% | 49.5  | qwen3.5-4b-4bit · 78% · 60.7     |
/// | 48 GB  | gemma-4-26b-4bit    | 17.0 | 87% | 49.5  | qwen3.6-35b-4bit · 87% · 60      |
/// | 64 GB  | qwen3.6-35b-8bit    | 37.7 | 87% | —     | qwen3.6-35b-4bit · 87% · 60      |
/// | 96 GB+ | qwen3.5-122b-mxfp4  | 65.0 | 88% | —     | qwen3.6-35b-4bit · 87% · 60      |
///
/// The measured rows use the standard ~8K prompt peak of the complete
/// ``rapid-mlx serve`` process tree on an M2 Pro 32 GB Mac mini, not weight
/// size or an idle/short-prompt RSS. Recommendations require zero new swap,
/// peak below 75% of the tier floor, 8K prefill >=100 tok/s, and decode
/// >=10 tok/s. The 64/96 GB rows predate this run and remain explicitly
/// unmeasured until matching hardware is available.
///
/// These figures must be measured THROUGH serve: a bare ``mlx_lm`` probe
/// does not exercise the product's cache configuration or full process
/// tree. The column is display-only (``pickStatsLine``); safety gates still
/// apply independently at launch.
///
/// The capability column is monotonic non-decreasing by RAM, with ONE
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
        /// Standard ~8K prompt peak footprint in GB for measured picks.
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
    /// its ``primary`` (smart) pick, and its ``alt`` (fast/light) pick.
    struct Tier: Sendable, Equatable {
        let floorGB: Double
        let primary: Pick
        let alt: Pick?

        /// Smart pick first, then the fast alt if present.
        var picks: [Pick] { alt.map { [primary, $0] } ?? [primary] }
    }

    /// Conservative general-purpose laptop picks. Both have verified tool
    /// calling and stay within the same footprint budget enforced at launch.
    private static let qwen4Pick = Pick(
        alias: "qwen3.5-4b-4bit", footprintGB: 6.0, capabilityPct: 78,
        tokensPerSec: 60.7, launchFlags: [])

    private static let qwen9Pick = Pick(
        alias: "qwen3.5-9b-4bit", footprintGB: 8.7, capabilityPct: 82,
        tokensPerSec: 35.7, launchFlags: [])

    /// Latest release-eval mean: Tool 47, Code 50, Reasoning 40, General 50
    /// = 46.75, rounded to 47. The Basic chat caveat remains user-facing.
    private static let lfm1Pick = Pick(
        alias: "lfm2.5-1b-4bit", footprintGB: 1.9, capabilityPct: 47,
        tokensPerSec: 208.4, launchFlags: [], caveat: "Basic chat")

    /// The 8 GB tier's smarter pick: LFM2.5-2.6B, a 2.6 B dense model whose
    /// 30 layers are 22 short-convolution blocks and just 8 GQA. Those 8
    /// attention layers are why it belongs here — the KV cache costs
    /// ~16 KB/token, so a 32 K conversation adds only ~0.5 GB on top of
    /// 1.6 GB of weights. On an M2 Pro it peaks at 3.0 GB on the standard
    /// 8K prompt, decodes at
    /// 93.5 tok/s on the short prompt, and prefills at 473 tok/s at 8K.
    ///
    /// It carries a caveat rather than a capability %, and the caveat is
    /// Liquid's own: they publish this model as "not recommended for
    /// agentic coding and knowledge-heavy tasks". It is post-trained for
    /// tool use and instruction following, and on those it beats models
    /// ~4x its size — but our users drive coding agents, and putting a
    /// bare "64%" on this card would invite exactly the use Liquid warns
    /// against. ``capabilityPct`` is never rendered for a caveat pick; the
    /// 64 below is the mean of the latest local tool, coding, reasoning,
    /// and general release-eval suites; the caveat remains more useful than
    /// presenting that small-suite composite as a universal quality score.
    private static let lfm26Pick = Pick(
        alias: "lfm2.5-2.6b-4bit", footprintGB: 3.0, capabilityPct: 64,
        tokensPerSec: 93.5, launchFlags: [], caveat: "Not for coding")

    /// Latest release-eval mean: Tool 93, Code 90, Reasoning 70, General 90
    /// = 85.75, rounded to 86.
    private static let bonsaiPick = Pick(
        alias: "bonsai-27b-2bit", footprintGB: 13.0, capabilityPct: 86,
        tokensPerSec: 17.5, launchFlags: [])

    private static let gemma26Pick = Pick(
        alias: "gemma-4-26b-4bit", footprintGB: 17.0, capabilityPct: 87,
        tokensPerSec: 49.5,
        launchFlags: ["--no-mllm", "--kv-cache-dtype", "bf16", "--cache-memory-mb", "512"])

    /// Existing reviewed fast/light pick for the unmeasured 48/64/96 GB tiers.
    private static let qwen35bFastPick = Pick(
        alias: "qwen3.6-35b-4bit", footprintGB: 20.0, capabilityPct: 87,
        tokensPerSec: 60.0, launchFlags: [])

    /// Source of truth — ascending by ``floorGB``. A recommendation change
    /// is a one-line edit here, verified by ``RAMBucketedDefaultTests`` and
    /// the standalone ``scripts/verify-recommendation-tiers.swift`` contract
    /// check against the bundled ``aliases.json``.
    static let tiers: [Tier] = [
        // 8 GB was a hole, not a tier. These Macs clamped up to the 16 GB
        // picks, then ``isRecommendedPick``'s floor guard correctly refused
        // to exempt them from the ``.tooBig`` gate — so launch auto-start
        // rejected the only thing it was offered and fell through to
        // ``SafeDefaultFallback``. The user's first run was a model the app
        // had just told them not to run. Nothing in the catalog fit until
        // LFM2.5-2.6B: 3.21 GB standard-8K peak leaves room for macOS on
        // an 8 GB machine; LFM 1B supplies the even lighter fast choice.
        Tier(floorGB: 8, primary: lfm26Pick, alt: lfm1Pick),
        Tier(
            floorGB: 16,
            primary: qwen4Pick,
            alt: lfm1Pick
        ),
        // 18 GB gets the verified 9B general-purpose model plus the 4B fast
        // option. Do not promote bonsai-27b-2bit here: its 13 GB standard
        // 8K peak leaves too little headroom at the 18 GB tier floor.
        Tier(
            floorGB: 18,
            primary: qwen9Pick,
            alt: qwen4Pick
        ),
        Tier(
            floorGB: 24,
            primary: bonsaiPick,
            alt: qwen4Pick
        ),
        Tier(
            floorGB: 32,
            primary: gemma26Pick,
            alt: qwen4Pick
        ),
        Tier(
            floorGB: 48,
            primary: gemma26Pick,
            alt: qwen35bFastPick
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

    /// Quality-aware order for choosing among models that are already on
    /// disk. Begin with the closest tier this Mac qualifies for and walk
    /// downward, preserving smart-before-fast within each tier and removing
    /// aliases repeated across tiers.
    static func preferenceOrder(forPhysicalRAMGB physicalRAMGB: Double) -> [String] {
        var seen: Set<String> = []
        var result: [String] = []
        let effectiveRAMGB = max(physicalRAMGB, tiers[0].floorGB)
        for tier in tiers.reversed() where tier.floorGB <= effectiveRAMGB {
            for alias in tier.picks.map(\.alias) where seen.insert(alias).inserted {
                result.append(alias)
            }
        }
        return result
    }

    /// Launch flags to apply when starting ``alias`` AS the recommended
    /// model for a Mac at ``physicalRAMGB`` — empty unless the alias is
    /// this Mac's primary or alt pick. This is why a 64 GB Mac that hand-
    /// picks ``gemma-4-26b-4bit`` (a 32/48 GB tier pick, not its own)
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
/// ## Cached-candidate ordering
///
/// A cached model is only useful as a default if it is also a good default.
/// Alphabetical order made a 256 GB Mac choose ``bonsai-27b-2bit`` while the
/// stronger ``qwen3.6-35b-4bit`` was already on disk. Rank known candidates
/// by the curated RAM table instead: start at this Mac's tier, walk downward,
/// and keep each tier's smart pick ahead of its fast alternative. This reuses
/// the same measured capability decisions the recommendation UI presents.
/// Aliases outside that table retain an alphabetical fallback; inventing a
/// quality score from parameter count or quantization would be less honest
/// than admitting that no curated comparison exists.
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
    /// Models that remain manually selectable but must never be chosen by an
    /// automatic default policy. Bonsai 1.7B was retired after repeatedly
    /// degenerating in ordinary plain chat; cached weights are not a reason to
    /// resurrect it on relaunch when auto-start is disabled.
    static let retiredAutomaticAliases: Set<String> = ["bonsai-1.7b-2bit"]

    static func pick(
        catalog: [ModelEntry],
        hardware: MacHardware,
        bucketedDefault: String,
        excludedAliases: Set<String> = retiredAutomaticAliases
    ) -> String? {
        let eligibleCatalog = catalog.filter { !excludedAliases.contains($0.alias) }
        let bucketedEntry = eligibleCatalog.first(where: { $0.alias == bucketedDefault })
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
        let cachedCandidates = eligibleCatalog
            .filter { $0.cached }
            .filter { ModelSizing.estimate(alias: $0.alias).paramsBillions != nil }
            .filter { isSafe($0, on: hardware) }
        if let pick = preferredCachedAlias(cachedCandidates, on: hardware) {
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
        return SafeDefaultFallback.pick(catalog: eligibleCatalog, hardware: hardware)
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

    private static func preferredCachedAlias(
        _ entries: [ModelEntry], on hardware: MacHardware
    ) -> String? {
        let aliases = Set(entries.map(\.alias))
        if let curated = RAMBucketedDefault.preferenceOrder(
            forPhysicalRAMGB: hardware.physicalRAMGB
        ).first(where: aliases.contains) {
            return curated
        }
        return firstAlphabetical(entries)
    }
}
