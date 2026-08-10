import Testing
@testable import Rapid

/// Pin the RAM tier → recommended-pick table. v0.13 replaced the old
/// 6-bucket × 5-role matrix with exactly two choices per RAM tier: a smart
/// primary and a faster/lighter alternative.
///
/// The tiers are keyed by machine RAM and a Mac rounds DOWN to the
/// nearest floor: a 20 GB Mac gets the 18 GB pick (which fits), not the
/// 24 GB pick. Table (see ``RAMBucketedDefault`` docstring for the full
/// numbers):
///
///    8 GB  → lfm2.5-2.6b-4bit  (+ fast lfm2.5-1b-4bit)
///   16 GB  → qwen3.5-4b-4bit   (+ fast lfm2.5-1b-4bit)
///   18 GB  → qwen3.5-9b-4bit   (+ fast qwen3.5-4b-4bit)
///   24 GB  → bonsai-27b-2bit   (+ fast qwen3.5-4b-4bit)
///   32 GB  → gemma-4-26b-4bit  (+ fast qwen3.5-4b-4bit)
///   48 GB  → gemma-4-26b-4bit  (+ fast qwen3.6-35b-4bit)
///   64 GB  → qwen3.6-35b-8bit  (+ fast qwen3.6-35b-4bit)
///   96 GB+ → qwen3.5-122b-mxfp4 (+ fast qwen3.6-35b-4bit)
///
/// Note: the picks trust the maintainer's MEASURED footprints, which are
/// smaller than ``ModelSizing``'s heuristic estimate for low-bit / MoE
/// models — so there is deliberately NO ``ModelSizing.classify`` fit
/// assertion here (that gate over-states these picks; the picker bypasses
/// it for recommended picks via ``isRecommendedPick``).
@Suite("RAMBucketedDefault — RAM tier → recommended pick")
struct RAMBucketedDefaultTests {

    // MARK: - Primary alias per RAM (rounds DOWN to nearest floor)

    @Test("Each RAM lands on the tier whose floor is the largest ≤ its RAM")
    func aliasPerRAM() {
        #expect(RAMBucketedDefault.alias(forPhysicalRAMGB: 8) == "lfm2.5-2.6b-4bit")   // 8 GB is its own tier now
        #expect(RAMBucketedDefault.alias(forPhysicalRAMGB: 16) == "qwen3.5-4b-4bit")
        #expect(RAMBucketedDefault.alias(forPhysicalRAMGB: 17) == "qwen3.5-4b-4bit")
        #expect(RAMBucketedDefault.alias(forPhysicalRAMGB: 18) == "qwen3.5-9b-4bit")
        #expect(RAMBucketedDefault.alias(forPhysicalRAMGB: 24) == "bonsai-27b-2bit")
        #expect(RAMBucketedDefault.alias(forPhysicalRAMGB: 32) == "gemma-4-26b-4bit")
        #expect(RAMBucketedDefault.alias(forPhysicalRAMGB: 48) == "gemma-4-26b-4bit")
        #expect(RAMBucketedDefault.alias(forPhysicalRAMGB: 64) == "qwen3.6-35b-8bit")
        #expect(RAMBucketedDefault.alias(forPhysicalRAMGB: 96) == "qwen3.5-122b-mxfp4")
        #expect(RAMBucketedDefault.alias(forPhysicalRAMGB: 256) == "qwen3.5-122b-mxfp4") // 96 tier
    }

    @Test("A 20 GB Mac rounds DOWN to the 18 GB pick (fits), not up to 24 GB")
    func roundsDownNotUp() {
        #expect(RAMBucketedDefault.alias(forPhysicalRAMGB: 20) == "qwen3.5-9b-4bit")
        #expect(RAMBucketedDefault.alias(forPhysicalRAMGB: 30) == "bonsai-27b-2bit")
    }

    @Test("Pathological zero / negative RAM clamps to the smallest tier, no crash")
    func degenerateRAM() {
        #expect(RAMBucketedDefault.alias(forPhysicalRAMGB: 0) == "lfm2.5-2.6b-4bit")
        #expect(RAMBucketedDefault.alias(forPhysicalRAMGB: -1) == "lfm2.5-2.6b-4bit")
    }

    // MARK: - Picks (primary + optional alt)

    @Test("Each tier surfaces a smart pick + a fast alt where speed warrants it")
    func smartAndFastPicks() {
        // Every tier deliberately exposes the same two-choice decision.
        let smallest = RAMBucketedDefault.picks(forPhysicalRAMGB: 8)
        #expect(smallest.count == 2)
        #expect(smallest[0].alias == "lfm2.5-2.6b-4bit")
        #expect(smallest[0].caveat == "Not for coding")
        #expect(smallest[1].alias == "lfm2.5-1b-4bit")
        let tier16 = RAMBucketedDefault.picks(forPhysicalRAMGB: 16)
        #expect(tier16.count == 2)
        #expect(tier16[0].alias == "qwen3.5-4b-4bit")
        #expect(tier16[1].alias == "lfm2.5-1b-4bit")
        // 18 GB: 9B smart + 4B fast, both verified for tools.
        let tier18 = RAMBucketedDefault.picks(forPhysicalRAMGB: 18)
        #expect(tier18.count == 2)
        #expect(tier18[0].alias == "qwen3.5-9b-4bit")
        #expect(tier18[1].alias == "qwen3.5-4b-4bit")
        #expect(RAMBucketedDefault.picks(forPhysicalRAMGB: 24).map(\.alias)
            == ["bonsai-27b-2bit", "qwen3.5-4b-4bit"])
        #expect(RAMBucketedDefault.picks(forPhysicalRAMGB: 32).map(\.alias)
            == ["gemma-4-26b-4bit", "qwen3.5-4b-4bit"])
        #expect(RAMBucketedDefault.picks(forPhysicalRAMGB: 48).map(\.alias)
            == ["gemma-4-26b-4bit", "qwen3.6-35b-4bit"])
        // 64/96 GB: fast alt is the lighter 4-bit Qwen3.6-35B (no caveat).
        let tier64 = RAMBucketedDefault.picks(forPhysicalRAMGB: 64)
        #expect(tier64.count == 2)
        #expect(tier64[1].alias == "qwen3.6-35b-4bit")
        #expect(tier64[1].caveat == nil)
        let tier96 = RAMBucketedDefault.picks(forPhysicalRAMGB: 96)
        #expect(tier96.count == 2)
        #expect(tier96[1].alias == "qwen3.6-35b-4bit")
    }

    // MARK: - Launch flags travel with the recommendation, gated by RAM

    @Test("Flags apply only when the alias IS the pick for that Mac's RAM")
    func launchFlagsAreRAMGated() {
        #expect(RAMBucketedDefault.launchFlags(forAlias: "qwen3.5-9b-4bit", physicalRAMGB: 18).isEmpty)
        #expect(RAMBucketedDefault.launchFlags(forAlias: "gemma-4-26b-4bit", physicalRAMGB: 32)
            == ["--no-mllm", "--kv-cache-dtype", "bf16", "--cache-memory-mb", "512"])
        #expect(RAMBucketedDefault.launchFlags(forAlias: "qwen3.6-35b-4bit", physicalRAMGB: 48).isEmpty)
        // Hand-picking gemma-26b on a 64 GB Mac (where it is NOT the pick)
        // → no forced flags, so it keeps vision.
        #expect(RAMBucketedDefault.launchFlags(forAlias: "gemma-4-26b-4bit", physicalRAMGB: 64).isEmpty)
        #expect(RAMBucketedDefault.launchFlags(forAlias: "not-a-pick", physicalRAMGB: 32).isEmpty)
    }

    @Test("isRecommendedPick is true only for this Mac's primary or alt, and is floor-gated")
    func isRecommendedPickContract() {
        #expect(RAMBucketedDefault.isRecommendedPick(alias: "qwen3.5-4b-4bit", physicalRAMGB: 16))
        #expect(RAMBucketedDefault.isRecommendedPick(alias: "qwen3.5-9b-4bit", physicalRAMGB: 18))
        #expect(RAMBucketedDefault.isRecommendedPick(alias: "qwen3.5-4b-4bit", physicalRAMGB: 18))
        #expect(!RAMBucketedDefault.isRecommendedPick(alias: "bonsai-27b-2bit", physicalRAMGB: 18))
        #expect(!RAMBucketedDefault.isRecommendedPick(alias: "gemma-4-12b-4bit", physicalRAMGB: 18)) // dropped from picks
        // An 8 GB Mac now SITS IN a tier, so its own pick is exempt from the
        // .tooBig gate — that exemption is the whole point of the tier, since
        // before it every pick offered to an 8 GB Mac was rejected at launch.
        #expect(RAMBucketedDefault.isRecommendedPick(alias: "lfm2.5-2.6b-4bit", physicalRAMGB: 8))
        // The bigger models are still NOT exempt there: they are not this
        // tier's picks, so the OOM hole stays closed.
        #expect(!RAMBucketedDefault.isRecommendedPick(alias: "bonsai-27b-2bit", physicalRAMGB: 8))
        #expect(!RAMBucketedDefault.isRecommendedPick(alias: "qwen3.5-4b-4bit", physicalRAMGB: 8))
        // Below the lowest floor there is still no exemption for anything.
        #expect(!RAMBucketedDefault.isRecommendedPick(alias: "lfm2.5-2.6b-4bit", physicalRAMGB: 4))
        // The fast alt is a recommended pick on its own tiers (64/96 GB),
        // so it also skips the .tooBig gate there.
        #expect(RAMBucketedDefault.isRecommendedPick(alias: "qwen3.6-35b-4bit", physicalRAMGB: 64))
        #expect(RAMBucketedDefault.isRecommendedPick(alias: "qwen3.6-35b-4bit", physicalRAMGB: 96))
    }

    @Test("Laptop recommendations agree with the conservative launch sizing guard")
    func laptopRecommendationsDoNotRequireSizingBypass() {
        for ram in [16.0, 18.0] {
            let hardware = MacHardware(
                brandString: "Apple Silicon", family: .m3, tier: .base,
                physicalRAMBytes: UInt64(ram * Double(1 << 30)),
                memoryBandwidthGBs: 100
            )
            for pick in RAMBucketedDefault.picks(forPhysicalRAMGB: ram) {
                #expect(
                    ModelSizing.classify(ModelSizing.estimate(alias: pick.alias), on: hardware) != .tooBig,
                    "\(pick.alias) must not be recommended and then rejected by the launch budget"
                )
            }
        }
    }

    // MARK: - Table invariants

    @Test("Tier floors are strictly increasing so round-down is unambiguous")
    func floorsAreSorted() {
        let floors = RAMBucketedDefault.tiers.map(\.floorGB)
        for (a, b) in zip(floors, floors.dropFirst()) {
            #expect(a < b, "Tier floors must strictly increase — got \(a) before \(b)")
        }
    }

    @Test("Capability % is a sane 0–100 for every pick")
    func capabilityInRange() {
        for tier in RAMBucketedDefault.tiers {
            for pick in tier.picks {
                #expect(pick.capabilityPct > 0 && pick.capabilityPct <= 100,
                        "\(pick.alias) capability \(pick.capabilityPct) out of range")
            }
        }
    }

    @Test("Every RAM tier exposes exactly smart and fast choices")
    func exactlyTwoChoices() {
        for tier in RAMBucketedDefault.tiers {
            #expect(tier.picks.count == 2, "Tier \(tier.floorGB) GB must have exactly two picks")
        }
    }
}

// MARK: - SafeDefaultFallback (codex r2 BLOCKING on #165)

@Suite("SafeDefaultFallback — hardware-floor fallback never returns .tooBig when avoidable")
struct SafeDefaultFallbackTests {
    private func host(gb: Double) -> MacHardware {
        MacHardware(
            brandString: "Apple M2", family: .m2, tier: .base,
            physicalRAMBytes: UInt64(gb) * 1024 * 1024 * 1024,
            memoryBandwidthGBs: 100
        )
    }

    private func entry(_ alias: String, cached: Bool = false) -> ModelEntry {
        ModelEntry(alias: alias, hfRepo: "synthetic/\(alias)", sizeOnDisk: nil, cached: cached)
    }

    @Test("Codex r2 case: 8 GB Mac with a big cached model — fallback prefers the smaller catalog entry")
    func eightGBWithBigCachedFallsBackToSmaller() {
        // The exact pathology codex flagged: user has a 122B model
        // cached from a friend's recommendation, then opens the app
        // on an 8 GB Air. The pre-r2 fallback would have returned
        // the 122B because `cached.first` ignored fit.
        let catalog = [
            entry("qwen3.5-122b-mxfp4", cached: true),
            entry("qwen3.5-4b-4bit", cached: false),
        ]
        let pick = SafeDefaultFallback.pick(catalog: catalog, hardware: host(gb: 8))
        // On 8 GB everything is .tooBig per ModelSizing, so step 3
        // fires — but step 3 still returns the smallest, which is
        // 4B, not the cached 122B.
        #expect(pick == "qwen3.5-4b-4bit")
    }

    @Test("16 GB Mac with no cached entries — picks the smallest fitting alias from catalog")
    func sixteenGBPicksFittingSmallest() {
        let catalog = [
            entry("qwen3.5-122b-mxfp4"),
            entry("qwen3.6-35b-4bit"),
            entry("qwen3.5-9b-4bit"),
            entry("qwen3.5-4b-4bit"),
        ]
        let pick = SafeDefaultFallback.pick(catalog: catalog, hardware: host(gb: 16))
        // Smallest .recommended-or-.borderline entry that fits 16 GB
        // is 4B (5.9 GB / 12.8 GB usable = .recommended). 9B at 16 GB
        // is borderline (0.676), also acceptable — but smaller wins.
        #expect(pick == "qwen3.5-4b-4bit")
    }

    @Test("32 GB Mac with 4B cached and 27B uncached — prefers cached 4B (cached + safe)")
    func thirtyTwoGBPrefersCachedSafeOverUncachedSafe() {
        let catalog = [
            entry("qwen3.5-4b-4bit", cached: true),
            entry("qwen3.5-9b-4bit", cached: false),
        ]
        let pick = SafeDefaultFallback.pick(catalog: catalog, hardware: host(gb: 32))
        // Both fit (4B and 9B are .recommended at 32 GB). Cached wins
        // the tiebreaker — instant boot.
        #expect(pick == "qwen3.5-4b-4bit")
    }

    @Test("32 GB Mac with 122B cached and 4B uncached — skips the cached .tooBig, returns the safe 4B")
    func thirtyTwoGBSkipsCachedTooBig() {
        let catalog = [
            entry("qwen3.5-122b-mxfp4", cached: true),
            entry("qwen3.5-4b-4bit", cached: false),
        ]
        let pick = SafeDefaultFallback.pick(catalog: catalog, hardware: host(gb: 32))
        // 122B at 32 GB is wildly .tooBig (74 GB needed vs 25.6 GB
        // usable). Even though it's the only cached entry, we skip
        // it for the safe-but-uncached 4B.
        #expect(pick == "qwen3.5-4b-4bit")
    }

    @Test("Empty catalog returns nil rather than crashing")
    func emptyCatalogReturnsNil() {
        let pick = SafeDefaultFallback.pick(catalog: [], hardware: host(gb: 16))
        #expect(pick == nil)
    }

    @Test("Single-entry catalog returns that entry even if .tooBig (last-resort step 3)")
    func singleEntryReturnsItEvenIfTooBig() {
        let catalog = [entry("qwen3.5-122b-mxfp4")]
        let pick = SafeDefaultFallback.pick(catalog: catalog, hardware: host(gb: 8))
        // 122B on 8 GB is wildly .tooBig, but there's literally
        // nothing else — picker has to default to something so it
        // can show the user the disabled-Start state with a
        // ModelSizing reason.
        #expect(pick == "qwen3.5-122b-mxfp4")
    }

    // MARK: - Codex r3 BLOCKING on #165 — unparseable aliases must not win

    /// Aliases like ``qwen3-coder-4bit`` and ``deepseek-v4-flash-2bit``
    /// don't carry a ``<n>b`` token. ``ModelSizing.parseParamsBillions``
    /// returns nil, and ``ModelSizing.estimate`` then reports a phantom
    /// totalGB = 0 + 1.2 + 2.0 = 3.2 GB while ``classify`` short-
    /// circuits to ``.borderline``. The pre-r3 ``SafeDefaultFallback``
    /// would have sorted the unparseable alias to the front of "smallest
    /// safe" and handed the picker a 20+ B coder model as the 8 GB
    /// default — exactly the OOM trap we just spent r1 and r2 closing.
    ///
    /// This test pins the contract: in a mixed catalog of one
    /// unparseable + one known-small alias on an 8 GB Mac, the known
    /// small wins.
    @Test("8 GB Mac with unparseable alias + known small — known small wins (#165 r3)")
    func unparseableAliasYieldsToKnownSmallOn8GB() {
        let catalog = [
            entry("qwen3-coder-4bit"),     // unparseable — no <n>b token
            entry("qwen3.5-4b-4bit"),      // known: 4B
        ]
        let pick = SafeDefaultFallback.pick(catalog: catalog, hardware: host(gb: 8))
        // The pre-r3 sort would put qwen3-coder-4bit first (phantom
        // 3.2 GB < 5.9 GB for 4B). After r3 we partition by
        // paramsBillions != nil; only the 4B is in the known set, so
        // step 3 returns it.
        #expect(pick == "qwen3.5-4b-4bit")
    }

    @Test("16 GB Mac with unparseable alias + 4B + 9B — picks 4B, not the unparseable")
    func unparseableLosesAtAllStepsWhenKnownAvailable() {
        let catalog = [
            entry("deepseek-v4-flash-2bit"),  // unparseable
            entry("glm4.5-air-4bit"),         // unparseable
            entry("qwen3.5-9b-4bit"),         // known: 9B
            entry("qwen3.5-4b-4bit"),         // known: 4B
        ]
        let pick = SafeDefaultFallback.pick(catalog: catalog, hardware: host(gb: 16))
        // Both unparseables would phantom-classify as 3.2 GB
        // .borderline, sorting them ahead of 4B (5.9 GB). r3 filter
        // pushes both out of the running entirely; smallest known fit
        // is 4B.
        #expect(pick == "qwen3.5-4b-4bit")
    }

    @Test("32 GB Mac with cached unparseable + uncached known small — known small still wins")
    func cachedUnparseableLosesToUncachedKnownSafe() {
        // Even cache priority shouldn't promote an unparseable alias.
        // If the user happens to have a coder model cached but is
        // first-launching the app, defaulting to "I can't tell you
        // how big this is, but Start is enabled" is worse than
        // "uncached but known-safe".
        let catalog = [
            entry("qwen3-coder-4bit", cached: true),   // unparseable but cached
            entry("qwen3.5-4b-4bit", cached: false),   // known small, uncached
        ]
        let pick = SafeDefaultFallback.pick(catalog: catalog, hardware: host(gb: 32))
        #expect(pick == "qwen3.5-4b-4bit")
    }

    @Test("Catalog of ONLY unparseable aliases — step 4 last-resort still returns something")
    func onlyUnparseableYieldsLastResort() {
        let catalog = [
            entry("qwen3-coder-4bit"),
            entry("deepseek-v4-flash-2bit"),
        ]
        let pick = SafeDefaultFallback.pick(catalog: catalog, hardware: host(gb: 16))
        // Defensive — in practice the real Rapid-MLX catalog always
        // contains parseable aliases, but a hand-typed custom-alias
        // scenario could leave us here. Picker still needs a non-nil
        // default; order in the catalog wins.
        #expect(pick == "qwen3-coder-4bit")
    }
}

// MARK: - CacheAwareDefault (issue #436)

/// Pin the four-step ladder ``ModelPickerBar.recommendedDefault``
/// consults on every fresh-launch alias resolution. The headline
/// case (the issue's smoking-gun screenshot) is the first test:
/// a 256 GB Mac with the Quickstart model cached should land on
/// ``bonsai-1.7b-2bit`` instead of the bucketed default.
///
/// The remaining tests pin the contract preserved from the legacy
/// path so a future "let's simplify the helper" PR can't silently
/// regress the landing-page promise, the codex r2 ``.tooBig``
/// guard from #165, or the codex r3 unparseable-alias guard.
@Suite("CacheAwareDefault — fresh-launch picker default prefers cached-and-runnable (#436)")
struct CacheAwareDefaultTests {
    private func host(gb: Double) -> MacHardware {
        MacHardware(
            brandString: "Apple M3 Ultra", family: .m3, tier: .ultra,
            physicalRAMBytes: UInt64(gb) * 1024 * 1024 * 1024,
            memoryBandwidthGBs: 800
        )
    }

    private func entry(_ alias: String, cached: Bool = false) -> ModelEntry {
        ModelEntry(alias: alias, hfRepo: "synthetic/\(alias)", sizeOnDisk: nil, cached: cached)
    }

    @Test("Cached preference walks closest eligible tier downward, smart before fast")
    func curatedPreferenceOrder() {
        let order = RAMBucketedDefault.preferenceOrder(forPhysicalRAMGB: 256)
        #expect(order.prefix(4) == [
            "qwen3.5-122b-mxfp4",
            "qwen3.6-35b-4bit",
            "qwen3.6-35b-8bit",
            "gemma-4-26b-4bit",
        ])
        #expect(Set(order).count == order.count)
        #expect(RAMBucketedDefault.preferenceOrder(forPhysicalRAMGB: 4).prefix(2) == [
            "lfm2.5-2.6b-4bit", "lfm2.5-1b-4bit",
        ])
    }

    // MARK: - Headline case (issue #436 repro)

    @Test("256 GB Mac with retired Bonsai cached — picker chooses coherent cached LFM")
    func retiredBonsaiNeverWinsCachedFallback() {
        let catalog = [
            entry("bonsai-1.7b-2bit", cached: true),
            entry("lfm2.5-1b-4bit", cached: true),
            entry("qwen3.5-122b-mxfp4", cached: false),
        ]
        let pick = CacheAwareDefault.pick(
            catalog: catalog,
            hardware: host(gb: 256),
            bucketedDefault: "qwen3.5-122b-mxfp4"
        )
        #expect(pick == "lfm2.5-1b-4bit")
    }

    // MARK: - Step 1: bucketed is cached + fits → use it

    @Test("Step 1: bucketed default already cached + fits — return bucketed")
    func bucketedCachedAndFitsWins() {
        let catalog = [
            entry("bonsai-1.7b-2bit", cached: true),
            entry("qwen3.6-35b-4bit", cached: true),
        ]
        let pick = CacheAwareDefault.pick(
            catalog: catalog,
            hardware: host(gb: 256),
            bucketedDefault: "qwen3.6-35b-4bit"
        )
        // Both cached, both fit, but bucketed wins because step 1
        // fires before the step 2 alphabetical fallback.
        #expect(pick == "qwen3.6-35b-4bit")
    }

    // MARK: - Step 2: cached-and-fits beats not-cached bucketed (the #436 fix)

    @Test("#1581 repro: cached fallback prefers the closest curated tier over alphabetical Bonsai")
    func multipleCachedQualityAwareTieBreak() {
        let catalog = [
            entry("qwen3.5-122b-mxfp4", cached: false),
            entry("bonsai-27b-2bit", cached: true),
            entry("qwen3.6-35b-4bit", cached: true),
        ]
        let pick = CacheAwareDefault.pick(
            catalog: catalog,
            hardware: host(gb: 256),
            bucketedDefault: "qwen3.5-122b-mxfp4"
        )
        #expect(pick == "qwen3.6-35b-4bit")
    }

    @Test("Step 2: bucketed missing from catalog entirely — cached candidate wins")
    func bucketedMissingCachedWins() {
        let catalog = [
            entry("bonsai-1.7b-2bit", cached: true),
            entry("lfm2.5-1b-4bit", cached: true),
        ]
        let pick = CacheAwareDefault.pick(
            catalog: catalog,
            hardware: host(gb: 256),
            bucketedDefault: "future-alias-not-yet-shipped"
        )
        #expect(pick == "lfm2.5-1b-4bit")
    }

    @Test("Step 2: cached candidate must FIT — .tooBig cached alias gets skipped, falls to bucketed")
    func cachedButTooBigSkippedFallsToBucketed() {
        let catalog = [
            entry("qwen3.5-122b-mxfp4", cached: true),  // cached but .tooBig on 18 GB
            entry("qwen3.5-9b-4bit", cached: false),    // bucketed default, fits
        ]
        let pick = CacheAwareDefault.pick(
            catalog: catalog,
            hardware: MacHardware(
                brandString: "Apple M3 Pro", family: .m3, tier: .pro,
                physicalRAMBytes: 18 * 1024 * 1024 * 1024,
                memoryBandwidthGBs: 150
            ),
            bucketedDefault: "qwen3.5-9b-4bit"
        )
        #expect(pick == "qwen3.5-9b-4bit")
    }

    @Test("Step 2: cached candidate with unparseable params — skipped (codex r3 #165 trap)")
    func cachedUnparseableSkipped() {
        let catalog = [
            entry("qwen3-coder-4bit", cached: true),    // unparseable, must NOT win
            entry("qwen3.5-9b-4bit", cached: false),    // bucketed, fits
        ]
        let pick = CacheAwareDefault.pick(
            catalog: catalog,
            hardware: host(gb: 256),
            bucketedDefault: "qwen3.5-9b-4bit"
        )
        // Step 2 rejects qwen3-coder-4bit → step 3 returns bucketed.
        #expect(pick == "qwen3.5-9b-4bit")
    }

    // MARK: - Step 3: bucketed fits but nothing cached → legacy behaviour

    @Test("Step 3: nothing cached — bucketed default wins (legacy parity)")
    func nothingCachedBucketedWins() {
        let catalog = [
            entry("qwen3.6-35b-4bit", cached: false),
            entry("qwen3.5-9b-4bit", cached: false),
        ]
        let pick = CacheAwareDefault.pick(
            catalog: catalog,
            hardware: host(gb: 256),
            bucketedDefault: "qwen3.6-35b-4bit"
        )
        #expect(pick == "qwen3.6-35b-4bit")
    }

    // MARK: - Step 4: bucketed missing AND nothing cached → SafeDefaultFallback escape

    @Test("Step 4: bucketed missing AND nothing cached fits — delegate to SafeDefaultFallback")
    func bucketedMissingNothingCachedFallback() {
        let catalog = [
            entry("qwen3.5-9b-4bit", cached: false),
            entry("qwen3.5-4b-4bit", cached: false),
        ]
        let pick = CacheAwareDefault.pick(
            catalog: catalog,
            hardware: host(gb: 8),
            bucketedDefault: "future-alias-not-yet-shipped"
        )
        // On 8 GB everything's .tooBig per ModelSizing; the
        // SafeDefaultFallback escape hands back the smallest known
        // (4B beats 9B).
        #expect(pick == "qwen3.5-4b-4bit")
    }

    @Test("Step 4: empty catalog returns nil (no crash)")
    func emptyCatalogReturnsNil() {
        let pick = CacheAwareDefault.pick(
            catalog: [],
            hardware: host(gb: 256),
            bucketedDefault: "qwen3.6-35b-4bit"
        )
        #expect(pick == nil)
    }

    // MARK: - Slim-DMG real-world case

    @Test("Slim DMG with retired cache — picker defaults to current coherent starter")
    func slimDMGFreshInstallPrefersQuickstart() {
        let catalog = [
            entry("bonsai-1.7b-2bit", cached: true),
            entry("lfm2.5-1b-4bit", cached: true),
            entry("qwen3.5-4b-4bit", cached: false),
            entry("qwen3.5-9b-4bit", cached: false),
            entry("qwen3.6-35b-4bit", cached: false),
        ]
        let pick = CacheAwareDefault.pick(
            catalog: catalog,
            hardware: host(gb: 256),
            bucketedDefault: "qwen3.5-122b-mxfp4"
        )
        #expect(pick == "lfm2.5-1b-4bit")
    }
}
