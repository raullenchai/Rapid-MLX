import Foundation
import Testing
@testable import Rapid

/// Adversarial input coverage for ``QuickstartCoordinator`` +
/// ``QuickstartView`` pure helpers (issue #284). Pin eligibility
/// against every combination of inputs; throw adversarial strings at
/// the friendly-failure classifier; poison UserDefaults with garbage
/// and assert the coordinator's init degrades gracefully.
@MainActor
@Suite("Quickstart — fuzz", .serialized)
struct QuickstartFuzzTests {

    // MARK: - Independent eligibility table

    /// Independent expected-output: reproduces the documented rule
    /// ("done==false AND lastServed==nil AND server in
    /// {.idle,.stopped}"). Disagreement → bug. First-run is decided
    /// from app-owned state ONLY — the shared HF cache was a gate
    /// (#298) but was removed because unrelated ecosystem models
    /// (Whisper / VAD / forced-aligner from other tools) wrongly
    /// suppressed onboarding for genuinely new users.
    private func expectedEligible(
        done: Bool,
        lastServedAlias: String?,
        serverState: ServerState
    ) -> Bool {
        if done { return false }
        if lastServedAlias != nil { return false }
        switch serverState {
        case .idle, .stopped: return true
        case .ready, .starting, .crashed, .missing: return false
        }
    }

    /// Brute-force every combination: 2 done × 4 lastServed-shapes
    /// × 7 server states = 56 cases. Any disagreement with the
    /// expected table is reported as a bug, not a flaky assertion.
    @Test("QuickstartCoordinator.isEligible: full truth table (56 combinations)")
    func eligibleFullTruthTable() {
        let dones = [false, true]
        let lastServeds: [String?] = [nil, "qwen3.5-4b", "gemma3-1b-qat-4bit", ""]
        let states: [ServerState] = [
            .idle,
            .stopped,
            .ready(alias: "x"),
            .starting(alias: "x"),
            .crashed(alias: "x", message: "boom"),
            .missing,
            .ready(alias: QuickstartCoordinator.defaultChoice.alias),
        ]
        for done in dones {
            for last in lastServeds {
                for state in states {
                    let actual = QuickstartCoordinator.isEligible(
                        done: done,
                        lastServedAlias: last,
                        serverState: state
                    )
                    let expected = expectedEligible(
                        done: done,
                        lastServedAlias: last,
                        serverState: state
                    )
                    #expect(
                        actual == expected,
                        "eligibility drift: done=\(done) last=\(last ?? "nil") state=\(state) actual=\(actual) expected=\(expected)"
                    )
                }
            }
        }
    }

    // MARK: - Failure classifier

    /// 200-iteration random drive of the classifier. The contract:
    ///   * Never returns empty for any non-empty input.
    ///   * Returns a non-empty fallback for empty input.
    ///   * Never crashes on any UTF-8 string (control chars / 1 MB /
    ///     mixed languages / null bytes).
    ///
    /// Codex r3 NIT closure: a separate regression test
    /// (``friendlyFailureWhitespaceFallthrough`` below) documents the
    /// whitespace-only fall-through bug filed as
    /// machinefi/rapid-desktop#290. The 200-iter random pass here is
    /// scoped to "never crash + never empty" only — the targeted
    /// behaviour is pinned in the dedicated test.
    @Test("friendlyFailureMessage: 200 adversarial strings never empty / never crash")
    func friendlyFailureNeverEmpty() {
        let baseSeed: UInt64 = 0xF411_F411_F411_F411
        // A few targeted strings + many random ones
        let targeted: [String] = [
            "",
            String(repeating: "x", count: 1_000_000),
            "Kein Speicherplatz",   // German "no disk space" — should NOT classify (we only match English heuristics)
            "磁盘已满",                // Chinese disk full — same, not classified
            "errno=28 ENOSPC",
            "\n\n\n",
            "\u{00}",
            "\u{202E}flipped",
            "rate-limit",
            "RATE LIMIT",
            "Network down",
            "Connection reset by peer",
            "DNS lookup failed",
            "timeout",
            "Operation timed out",
        ]
        for s in targeted {
            let out = QuickstartView.friendlyFailureMessage(raw: s)
            #expect(!out.isEmpty, "empty classifier output for input '\(s.prefix(64))'")
        }
        for i in 0..<200 {
            var rng = SplitMix64(seed: baseSeed &+ UInt64(i))
            let len = Int(rng.next() % 2048)
            let s = FuzzSessionFactory.randomString(rng: &rng, byteCount: len)
            let out = QuickstartView.friendlyFailureMessage(raw: s)
            // Contract: non-empty output for any input (empty input
            // gets the explicit fallback).
            #expect(!out.isEmpty)
        }
    }

    /// Guard the whitespace-fallback fix for
    /// machinefi/rapid-desktop#290. Pre-fix, ``raw == "   "`` /
    /// ``"\n\n\n"`` / ``"\t\t"`` skipped the ``raw.isEmpty`` branch and
    /// the classifier returned the whitespace verbatim — the failure
    /// card then rendered a visually blank bubble. The fix tightens the
    /// empty-check to ``trimmingCharacters(in: .whitespacesAndNewlines)
    /// .isEmpty`` so every shape of bare whitespace falls back to the
    /// "Download didn't finish" copy. If this test ever flips back to
    /// ``out == r``, the fallback has regressed.
    @Test("friendlyFailureMessage: whitespace-only input falls back (#290 fixed)")
    func friendlyFailureWhitespaceFallthrough() {
        // Empty input → documented "Download didn't finish" fallback.
        let onEmpty = QuickstartView.friendlyFailureMessage(raw: "")
        #expect(onEmpty.contains("didn't finish"))
        // Whitespace-only inputs now take the same fallback as empty —
        // the classifier MUST NOT return the raw whitespace verbatim.
        for r in ["\n\n\n", "   ", "\t\t"] {
            let out = QuickstartView.friendlyFailureMessage(raw: r)
            #expect(out != r,
                    "whitespace-only input passed through verbatim — #290 fix has regressed")
            #expect(out.contains("didn't finish"),
                    "whitespace-only input should take the empty-fallback message")
        }
    }

    /// Classifier outputs for each documented keyword are stable
    /// across 1 KB + control-char prefixes (the keyword still wins).
    @Test("friendlyFailureMessage: keyword classification survives prefix noise")
    func friendlyFailureClassificationStable() {
        // Each pair: (keyword that should be matched, classifier
        // substring we expect in the output).
        let pairs: [(String, String)] = [
            ("429", "rate-limit"),
            ("rate limit", "rate-limit"),
            ("network", "Network error"),
            ("connection", "Network error"),
            ("dns", "Network error"),
            ("timeout", "Network error"),
            ("timed out", "Network error"),
            ("no space", "disk space"),
            ("disk full", "disk space"),
        ]
        for (keyword, marker) in pairs {
            let prefix = "[\u{200B}garbage\n\u{0B}]"
            let raw = prefix + keyword + " then stuff"
            let out = QuickstartView.friendlyFailureMessage(raw: raw)
            #expect(out.contains(marker) || out.lowercased().contains(marker.lowercased()),
                    "classifier failed for keyword '\(keyword)' in noise prefix")
        }
    }

    // MARK: - Coordinator init under UserDefaults poison

    /// Poison the storage keys with non-Bool / non-existent values
    /// and assert the production ``QuickstartCoordinator`` init reads
    /// them without crashing AND exposes a deterministic ``done`` /
    /// ``awaitingWelcomeSeed`` value.
    ///
    /// Cross-suite contention note: ``QuickstartCoordinator`` reads
    /// from ``UserDefaults.standard`` unconditionally (no test seam),
    /// so this test has to mutate the global suite. We save + restore
    /// the original values around every mutation. The pre-existing
    /// ``QuickstartViewTests`` suite touches the same keys; both
    /// suites run on the @MainActor and Swift Testing serializes
    /// @MainActor tests within a single process by default, so cross-
    /// suite collisions are theoretical rather than observed. The
    /// alternative (adding a UserDefaults parameter to the
    /// coordinator's init) is a production refactor and out of scope
    /// for this fuzz harness.
    ///
    /// Codex r2 MAJOR closure: every poison case now carries an
    /// explicit assertion against ``done`` / ``awaitingWelcomeSeed``.
    /// A regression where garbage at the key makes Quickstart
    /// permanently skipped (``done`` reads as ``true`` on integer /
    /// array / dictionary garbage) would surface as a failed
    /// assertion below — what UserDefaults.bool(forKey:) DOES return
    /// for each shape is well-documented + pinned here so a future
    /// Foundation behavioural change is caught at CI time.
    @Test("QuickstartCoordinator.init: poisoned UserDefaults values produce predictable done / awaitingWelcomeSeed values")
    func initDegradesOnPoisonedDefaults() {
        let defaults = UserDefaults.standard
        let originalDone = defaults.object(forKey: QuickstartCoordinator.storageKey)
        let originalAwait = defaults.object(forKey: QuickstartCoordinator.awaitingSeedKey)
        defer {
            if let v = originalDone {
                defaults.set(v, forKey: QuickstartCoordinator.storageKey)
            } else {
                defaults.removeObject(forKey: QuickstartCoordinator.storageKey)
            }
            if let v = originalAwait {
                defaults.set(v, forKey: QuickstartCoordinator.awaitingSeedKey)
            } else {
                defaults.removeObject(forKey: QuickstartCoordinator.awaitingSeedKey)
            }
        }

        // Poison case 1: non-zero integer. UserDefaults.bool(forKey:)
        // promotes any non-zero number to true; document the actual
        // behaviour so a future Foundation change is caught.
        defaults.set(42, forKey: QuickstartCoordinator.storageKey)
        defaults.set(0, forKey: QuickstartCoordinator.awaitingSeedKey)
        let c1 = QuickstartCoordinator()
        #expect(c1.done == true,
                "non-zero integer at storage key should read as true under bool(forKey:)")
        #expect(c1.awaitingWelcomeSeed == false,
                "zero integer at awaitingSeed key should read as false")
        c1._testingReset()

        // Poison case 2: array. UserDefaults.bool(forKey:) returns
        // false for any non-Bool / non-Number / non-string value.
        defaults.set(["a", "b"], forKey: QuickstartCoordinator.storageKey)
        defaults.set([1, 2, 3], forKey: QuickstartCoordinator.awaitingSeedKey)
        let c2 = QuickstartCoordinator()
        #expect(c2.done == false,
                "array at storage key should read as false under bool(forKey:)")
        #expect(c2.awaitingWelcomeSeed == false,
                "array at awaitingSeed key should read as false")
        c2._testingReset()

        // Poison case 3: dictionary at the bool key.
        defaults.set(["key": "value"], forKey: QuickstartCoordinator.storageKey)
        let c3 = QuickstartCoordinator()
        #expect(c3.done == false,
                "dictionary at storage key should read as false")
        c3._testingReset()

        // Poison case 4: empty string. bool(forKey:) for an empty
        // string returns false.
        defaults.set("", forKey: QuickstartCoordinator.storageKey)
        defaults.set("", forKey: QuickstartCoordinator.awaitingSeedKey)
        let c4 = QuickstartCoordinator()
        #expect(c4.done == false,
                "empty string should not parse as true")
        #expect(c4.awaitingWelcomeSeed == false,
                "empty string at awaitingSeed key should not parse as true")
        c4._testingReset()

        // Poison case 5: completely-removed keys (fresh-install
        // shape). The init's spec is `defaults.bool(forKey:)` on a
        // missing key returns false — pin it.
        defaults.removeObject(forKey: QuickstartCoordinator.storageKey)
        defaults.removeObject(forKey: QuickstartCoordinator.awaitingSeedKey)
        let c5 = QuickstartCoordinator()
        #expect(c5.done == false, "missing storage key should produce done=false")
        #expect(c5.awaitingWelcomeSeed == false, "missing awaitingSeed key should produce awaitingWelcomeSeed=false")
        c5._testingReset()
    }

    // MARK: - State machine: no fatalError paths on random input

    /// Drive the coordinator through 200 random sequences of
    /// (enterDownloading / enterStarting / enterFailed / markReady /
    /// releaseInFlight / markDone / clearPendingSeed) and assert it
    /// never crashes or lands in an undefined state.
    @Test("QuickstartCoordinator: 200 random transition sequences never crash")
    func randomTransitionSequencesNeverCrash() async {
        let baseSeed: UInt64 = 0xCAFE_F00D_BABE_4242
        for i in 0..<200 {
            var rng = SplitMix64(seed: baseSeed &+ UInt64(i))
            let coord = QuickstartCoordinator()
            coord._testingReset()
            for _ in 0..<32 {
                switch rng.next() % 8 {
                case 0: coord.enterDownloading()
                case 1: coord.enterStarting()
                case 2: coord.enterFailed(message: "synthetic-\(rng.next())")
                case 3: _ = coord.markReady { (rng.next() & 1) == 0 }
                case 4: coord.releaseInFlight()
                case 5: coord.markDone()
                case 6: coord.clearPendingSeed()
                default: coord._testingReset()
                }
                // Phase must be one of the documented cases — Swift
                // will already enforce this at the type level; this
                // assertion is for the future where someone adds a
                // .switch on Phase that misses a case.
                switch coord.phase {
                case .idle, .lowDiskWarning, .downloading, .starting, .ready, .failed: break
                }
            }
            // Always reset at the end so we don't pollute the global
            // UserDefaults across iterations.
            coord._testingReset()
        }
    }
}
