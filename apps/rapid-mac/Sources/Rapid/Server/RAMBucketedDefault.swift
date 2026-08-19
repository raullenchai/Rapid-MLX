import Foundation

private final class RecommendationBundleFinder: NSObject {}

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
/// | 32 GB  | qwen3.8-27b-4bit    | 20.0 | 92% | 40.7  | qwen3.5-4b-4bit · 78% · 60.7     |
/// | 48 GB  | qwen3.8-27b-4bit    | 20.0 | 92% | 40.7  | qwen3.6-35b-4bit · 87% · 60      |
/// | 64 GB  | qwen3.8-27b-4bit    | 20.0 | 92% | 40.7  | qwen3.6-35b-4bit · 87% · 60      |
/// | 96 GB+ | qwen3.8-27b-4bit    | 20.0 | 92% | 40.7  | qwen3.6-35b-4bit · 87% · 60      |
///
/// Every tier from 32 GB up shares one smart pick (AA-Index policy,
/// 2026-08-18): qwen3.8-27b-4bit is the highest-scoring open-weights model
/// the engine serves on the Artificial Analysis Intelligence Index (52 —
/// GPT-5.6-class; the 122B it displaced scores 33), and its measured 8K
/// peak clears every one of those tiers' budgets.
///
/// The measured rows use the standard ~8K prompt peak of the complete
/// ``rapid-mlx serve`` process tree, not weight size or an idle/short-prompt
/// RSS. Recommendations require zero new swap, peak below 75% of the tier
/// floor, 8K prefill >=100 tok/s, and decode >=10 tok/s.
///
/// These figures must be measured THROUGH serve: a bare ``mlx_lm`` probe
/// does not exercise the product's cache configuration or full process
/// tree. The column is display-only (``pickStatsLine``); safety gates still
/// apply independently at launch.
///
/// The capability column is monotonic non-decreasing by RAM, and every
/// smart pick reads at or above its own fast alt — e.g. the 64 GB smart
/// pick (qwen3.8-27b-4bit) shows 92 %, above its qwen3.6-35b-4bit alt's
/// 87 % — so "Best pick" never looks weaker than "Faster". Every alias
/// is verified to exist in the bundled ``aliases.json`` by
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

    /// Decoded from the repository-wide SSOT used by both Rapid Desktop and
    /// ``rapid-mlx recipe``. A missing or malformed table is a packaging error:
    /// silently inventing a fallback would recreate the cross-surface drift this
    /// catalog exists to prevent.
    static let tiers: [Tier] = loadTiers()

    private struct Payload: Decodable {
        let schemaVersion: Int
        let tiers: [RawTier]

        enum CodingKeys: String, CodingKey {
            case schemaVersion = "schema_version"
            case tiers
        }
    }

    private struct RawTier: Decodable {
        let floorGB: Double
        let picks: [RawPick]

        enum CodingKeys: String, CodingKey {
            case floorGB = "floor_gb"
            case picks
        }
    }

    private struct RawPick: Decodable {
        let role: String
        let alias: String
        let footprintGB: Double
        let capabilityPct: Int
        let tokensPerSec: Double?
        let launchFlags: [String]
        let caveat: String?

        enum CodingKeys: String, CodingKey {
            case role, alias, caveat
            case footprintGB = "footprint_gb"
            case capabilityPct = "capability_pct"
            case tokensPerSec = "tokens_per_sec"
            case launchFlags = "launch_flags"
        }

        var pick: Pick {
            Pick(alias: alias, footprintGB: footprintGB,
                 capabilityPct: capabilityPct, tokensPerSec: tokensPerSec,
                 launchFlags: launchFlags, caveat: caveat)
        }
    }

    private static func loadTiers() -> [Tier] {
        guard let url = recommendationResourceURL(),
              let data = try? Data(contentsOf: url),
              let payload = try? JSONDecoder().decode(Payload.self, from: data),
              payload.schemaVersion == 1 else {
            fatalError("model_recommendations.json is missing or invalid")
        }
        precondition(!payload.tiers.isEmpty, "recommendation catalog must contain tiers")
        var previousFloor = -Double.infinity
        return payload.tiers.map { raw in
            precondition(raw.floorGB > previousFloor,
                         "recommendation tiers must be sorted by RAM floor")
            previousFloor = raw.floorGB
            precondition(raw.picks.count == 2 && raw.picks.map(\.role) == ["smart", "fast"],
                         "every RAM tier must contain smart + fast picks")
            return Tier(floorGB: raw.floorGB, primary: raw.picks[0].pick,
                        alt: raw.picks[1].pick)
        }
    }

    private static func recommendationResourceURL() -> URL? {
        if let url = Bundle.main.url(forResource: "model_recommendations", withExtension: "json") {
            return url
        }
        let anchor = Bundle(for: RecommendationBundleFinder.self).bundleURL.deletingLastPathComponent()
        if let bundle = Bundle(url: anchor.appendingPathComponent("Rapid_Rapid.bundle")),
           let url = bundle.url(forResource: "model_recommendations", withExtension: "json") {
            return url
        }
        // SwiftPM tests run from a source checkout. Keep the source fallback
        // pointed at the Python package's canonical file; it is deliberately
        // not a second copied table.
        let sourceCandidate = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Server
            .deletingLastPathComponent() // Rapid
            .deletingLastPathComponent() // Sources
            .deletingLastPathComponent() // rapid-mac
            .deletingLastPathComponent() // apps
            .appendingPathComponent("vllm_mlx/model_recommendations.json")
        if FileManager.default.fileExists(atPath: sourceCandidate.path) {
            return sourceCandidate
        }
        // Some SwiftPM invocations preserve a relative #filePath. Walk upward
        // from their working directory instead of resolving that relative path
        // against an already-nested package directory.
        var directory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        for _ in 0..<6 {
            let candidate = directory.appendingPathComponent("vllm_mlx/model_recommendations.json")
            if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
            directory.deleteLastPathComponent()
        }
        return nil
    }

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
