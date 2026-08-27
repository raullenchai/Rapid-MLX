import Foundation
import Testing
@testable import Rapid

/// Contract for v0.4.14 ``SamplingConfig`` — the persisted store
/// behind Settings → Sampling. The whole point of pinning these is
/// that the v0.4.12 defaults (temperature 0.7, top_p 0.95,
/// max_tokens 4096, repetition_penalty 1.1) are the safe values
/// we ship — a refactor can't quietly shift them without breaking
/// these tests.
@MainActor
@Suite("SamplingConfig defaults + persistence")
final class SamplingConfigTests {
    init() { CIHangWatchdog.noteProgress() }

    /// Names of every suite ``freshDefaults`` minted for this test
    /// instance. Each ``@Test`` gets a fresh instance of the suite
    /// type, so ``deinit`` runs after the test exits — the suite is
    /// removed from the in-memory ``UserDefaults`` registry and the
    /// backing ``~/Library/Preferences/<name>.plist`` is unlinked.
    ///
    /// F-005 lesson: ``UserDefaults(suiteName:)`` flushes to disk on
    /// first write and the file persists forever unless explicitly
    /// removed. Before this RAII wrapper the test suite was leaking
    /// ~900 plists per dev machine after a few weeks of ``swift
    /// test`` runs.
    nonisolated(unsafe) private var createdSuiteNames: [String] = []

    deinit {
        CIHangWatchdog.noteProgress()
        TestDefaultsScope.cleanup(suiteNames: createdSuiteNames)
    }

    /// Issue #139 dedupe — the cleanup mechanics that used to live
    /// here are now shared with the other suite-minting tests via
    /// ``TestDefaultsScope.cleanup``. The regression tests below
    /// still call it directly so they can pin the disk-side
    /// contract without racing against ``deinit``'s non-deterministic
    /// timing.
    private static func cleanup(suiteNames: [String]) {
        TestDefaultsScope.cleanup(suiteNames: suiteNames)
    }

    private func suiteName() -> String {
        TestDefaultsScope.mintSuiteName(prefix: "rapid-sampling-test-")
    }

    private func freshDefaults() -> UserDefaults {
        // Each test gets its own UserDefaults suite so concurrent
        // runs don't clobber each other. ``standard`` would also
        // leak into the user's actual macOS preferences during a
        // local test run.
        let name = suiteName()
        createdSuiteNames.append(name)
        let d = UserDefaults(suiteName: name)!
        d.removePersistentDomain(forName: name)
        return d
    }

    @Test("First-launch defaults match the v0.4.12 hard-coded profile")
    func defaultsMatchV0412() {
        let s = SamplingConfig(defaults: freshDefaults())
        #expect(s.temperature == 0.7)
        #expect(s.topP == 0.95)
        #expect(s.maxTokens == 4096)
        #expect(s.repetitionPenalty == 1.1)
        #expect(s.isAtDefaults)
    }

    @Test("Mutations persist to the backing UserDefaults under the v0 keyspace")
    func mutationsPersist() {
        let d = freshDefaults()
        let prefix = "rapid.sampling.v0"
        let s = SamplingConfig(defaults: d, keyPrefix: prefix)
        s.temperature = 0.42
        s.topP = 0.5
        s.maxTokens = 1024
        s.repetitionPenalty = 1.25

        // Read back via the raw UserDefaults so we pin the wire
        // shape, not just the in-memory state.
        #expect(d.double(forKey: "\(prefix).temperature") == 0.42)
        #expect(d.double(forKey: "\(prefix).topP") == 0.5)
        #expect(d.integer(forKey: "\(prefix).maxTokens") == 1024)
        #expect(d.double(forKey: "\(prefix).repetitionPenalty") == 1.25)
    }

    @Test("Fresh instance reads back the persisted values — survives app relaunch")
    func reloadFromDefaults() {
        let d = freshDefaults()
        let s1 = SamplingConfig(defaults: d)
        s1.temperature = 0.33
        s1.topP = 0.42
        s1.maxTokens = 2048
        s1.repetitionPenalty = 1.07

        // A brand-new instance reading from the same UserDefaults
        // mirrors what happens after a process restart (RapidApp
        // builds a fresh SamplingConfig in init).
        let s2 = SamplingConfig(defaults: d)
        #expect(s2.temperature == 0.33)
        #expect(s2.topP == 0.42)
        #expect(s2.maxTokens == 2048)
        #expect(s2.repetitionPenalty == 1.07)
        #expect(!s2.isAtDefaults)
    }

    @Test("Persisted out-of-range values are clamped on load")
    func persistedValuesClampOnLoad() {
        let d = freshDefaults()
        let prefix = "rapid.sampling.v0"
        d.set(-0.25, forKey: "\(prefix).temperature")
        d.set(5.0, forKey: "\(prefix).topP")
        d.set(-99, forKey: "\(prefix).maxTokens")
        d.set(0.5, forKey: "\(prefix).repetitionPenalty")

        let s = SamplingConfig(defaults: d, keyPrefix: prefix)

        #expect(s.temperature == SamplingConfig.temperatureRange.lowerBound)
        #expect(s.topP == SamplingConfig.topPRange.upperBound)
        #expect(s.maxTokens == SamplingConfig.maxTokensRange.lowerBound)
        #expect(s.repetitionPenalty == SamplingConfig.repetitionPenaltyRange.lowerBound)
    }

    @Test("Direct out-of-range mutations clamp before persisting")
    func directMutationsClampBeforePersisting() {
        let d = freshDefaults()
        let prefix = "rapid.sampling.v0"
        let s = SamplingConfig(defaults: d, keyPrefix: prefix)

        s.temperature = Double.nan
        s.topP = -4.0
        s.maxTokens = 999_999
        s.repetitionPenalty = 2.0

        #expect(s.temperature == SamplingConfig.temperatureDefault)
        #expect(s.topP == SamplingConfig.topPRange.lowerBound)
        #expect(s.maxTokens == SamplingConfig.maxTokensRange.upperBound)
        #expect(s.repetitionPenalty == SamplingConfig.repetitionPenaltyRange.upperBound)
        #expect(d.double(forKey: "\(prefix).temperature") == SamplingConfig.temperatureDefault)
        #expect(d.double(forKey: "\(prefix).topP") == SamplingConfig.topPRange.lowerBound)
        #expect(d.integer(forKey: "\(prefix).maxTokens") == SamplingConfig.maxTokensRange.upperBound)
        #expect(d.double(forKey: "\(prefix).repetitionPenalty") == SamplingConfig.repetitionPenaltyRange.upperBound)
    }

    /// README "Sampling" section "audit batch 12" claims "Out-of-range
    /// / NaN clamping on **both load and write**". The existing
    /// ``directMutationsClampBeforePersisting`` pins NaN on the write
    /// path; this test pins the load half. If a refactor drops the
    /// ``guard value.isFinite`` line in ``clamped`` then ``min/max``
    /// silently leaks NaN back out (``min(max(.nan, 0), 1) == .nan``),
    /// so the user-visible NaN would round-trip through the persisted
    /// store and corrupt every subsequent inference request.
    @Test("Persisted NaN values are clamped on load to the safe default")
    func persistedNaNClampedOnLoad() {
        let d = freshDefaults()
        let prefix = "rapid.sampling.v0"
        d.set(Double.nan, forKey: "\(prefix).temperature")
        d.set(Double.nan, forKey: "\(prefix).topP")
        d.set(Double.nan, forKey: "\(prefix).repetitionPenalty")

        let s = SamplingConfig(defaults: d, keyPrefix: prefix)

        #expect(s.temperature == SamplingConfig.temperatureDefault)
        #expect(s.topP == SamplingConfig.topPDefault)
        #expect(s.repetitionPenalty == SamplingConfig.repetitionPenaltyDefault)
        #expect(s.temperature.isFinite)
        #expect(s.topP.isFinite)
        #expect(s.repetitionPenalty.isFinite)
    }

    /// Same defence as NaN but for the +/-Infinity edge. ``isFinite``
    /// is documented to reject both NaN and infinities, but the
    /// existing tests only exercise NaN on the write path and finite
    /// out-of-range on the load path. An explicit Inf test pins the
    /// full ``isFinite`` contract end-to-end so a regression to
    /// ``value.isNaN`` (which would let +Inf through) is caught.
    @Test("Persisted +Infinity / -Infinity values are clamped on load to the safe default")
    func persistedInfinityClampedOnLoad() {
        let d = freshDefaults()
        let prefix = "rapid.sampling.v0"
        d.set(Double.infinity, forKey: "\(prefix).temperature")
        d.set(-Double.infinity, forKey: "\(prefix).topP")
        d.set(Double.infinity, forKey: "\(prefix).repetitionPenalty")

        let s = SamplingConfig(defaults: d, keyPrefix: prefix)

        #expect(s.temperature == SamplingConfig.temperatureDefault)
        #expect(s.topP == SamplingConfig.topPDefault)
        #expect(s.repetitionPenalty == SamplingConfig.repetitionPenaltyDefault)
    }

    /// Pins +/-Infinity on the write path. ``directMutationsClampBeforePersisting``
    /// only exercises NaN on this side; an Inf assignment must take
    /// the same fallback path so we can't end up persisting an Inf
    /// to UserDefaults that a later load has to re-clamp.
    @Test("Direct Infinity mutations clamp to the safe default before persisting")
    func directInfinityMutationsClamp() {
        let d = freshDefaults()
        let prefix = "rapid.sampling.v0"
        let s = SamplingConfig(defaults: d, keyPrefix: prefix)

        s.temperature = Double.infinity
        s.topP = -Double.infinity
        s.repetitionPenalty = Double.infinity

        #expect(s.temperature == SamplingConfig.temperatureDefault)
        #expect(s.topP == SamplingConfig.topPDefault)
        #expect(s.repetitionPenalty == SamplingConfig.repetitionPenaltyDefault)
        #expect(d.double(forKey: "\(prefix).temperature") == SamplingConfig.temperatureDefault)
        #expect(d.double(forKey: "\(prefix).topP") == SamplingConfig.topPDefault)
        #expect(d.double(forKey: "\(prefix).repetitionPenalty") == SamplingConfig.repetitionPenaltyDefault)
    }

    @Test("resetToDefaults restores all four knobs at once + isAtDefaults flips back to true")
    func resetButton() {
        let s = SamplingConfig(defaults: freshDefaults())
        s.temperature = 0.1
        s.topP = 0.2
        s.maxTokens = 256
        s.repetitionPenalty = 1.4
        #expect(!s.isAtDefaults)

        s.resetToDefaults()

        #expect(s.temperature == 0.7)
        #expect(s.topP == 0.95)
        #expect(s.maxTokens == 4096)
        #expect(s.repetitionPenalty == 1.1)
        #expect(s.isAtDefaults)
    }

    @Test("isAtDefaults catches partial-default profiles — flipping any one knob trips it")
    func isAtDefaultsIsPrecise() {
        let s = SamplingConfig(defaults: freshDefaults())
        #expect(s.isAtDefaults)

        s.temperature = 0.71  // 0.01 off — still NOT at defaults
        #expect(!s.isAtDefaults)
        s.temperature = 0.7
        #expect(s.isAtDefaults)

        s.maxTokens = 4097
        #expect(!s.isAtDefaults)
    }

    /// F-005 regression — pins the disk-side cleanup contract that
    /// keeps the test suite from leaking ``rapid-sampling-test-*``
    /// plists into ``~/Library/Preferences/``.
    ///
    /// Before this fix every ``freshDefaults()`` call minted a suite
    /// (which the OS flushes to disk on first write) but no teardown
    /// removed it; ~900 stale plists accumulated per dev machine.
    /// The class-shaped suite + ``deinit``-triggered ``cleanup``
    /// helper closes that. We can't deterministically observe
    /// ``deinit`` from inside a ``@Test`` (instance lifetime is
    /// model-dependent), so the test calls the same static helper
    /// directly and asserts the suite is gone from the registry.
    @Test("F-005 regression: cleanup helper drops the UserDefaults suite from the registry AND unlinks the on-disk plist")
    func cleanupHelperRemovesSuite() throws {
        // Issue #139 review — go through the shared minter rather
        // than hand-rolling the prefix. This is the canonical
        // reference test for the cleanup contract, so a future
        // contributor cargo-culting from it must end up calling
        // ``TestDefaultsScope.mintSuiteName(prefix:)``, not splicing
        // their own string.
        let name = TestDefaultsScope.mintSuiteName(prefix: "rapid-sampling-test-")
        // Register up front so ``deinit`` re-cleans if this test
        // bails between the ``set`` and the ``cleanup`` call below.
        createdSuiteNames.append(name)
        let d = UserDefaults(suiteName: name)!
        d.set(0.5, forKey: "rapid.sampling.v0.temperature")
        // Round-trip via a second instance pins that the value was
        // really written (not just buffered in the local handle).
        #expect(UserDefaults(suiteName: name)?.double(forKey: "rapid.sampling.v0.temperature") == 0.5)

        // codex r1 P2 — force ``cfprefsd`` to flush BEFORE the
        // ``cleanup`` call. ``UserDefaults`` may keep the write
        // buffered in memory; if the plist never materialises on
        // disk, the post-cleanup ``fileExists`` assertion below
        // would pass vacuously and silently rubber-stamp a future
        // regression that stops unlinking plists. ``synchronize``
        // forces commit; we then ``#require`` the file is on disk
        // so the test fails loudly if the OS ever changes its
        // flushing behaviour rather than producing a false green.
        d.synchronize()
        let path = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Library/Preferences/\(name).plist")
        try #require(
            FileManager.default.fileExists(atPath: path.path),
            "Pre-cleanup invariant: the plist must exist on disk after a synchronised write, otherwise the disk-side cleanup assertion below is vacuous. Path: \(path.path)"
        )

        Self.cleanup(suiteNames: [name])

        // The suite is dropped from the in-memory registry — a fresh
        // handle sees the absent-key state. (We assert the absent-
        // key shape rather than ``.double(forKey:) == 0`` because
        // ``UserDefaults.double`` returns 0 for both 0.0 and missing.)
        let after = UserDefaults(suiteName: name)!
        #expect(after.object(forKey: "rapid.sampling.v0.temperature") == nil)

        // The disk-side half: cleanup must also unlink the
        // ``<name>.plist`` that ``cfprefsd`` flushed above. Without
        // the ``FileManager.removeItem`` call inside ``cleanup`` the
        // file persists as a 42-byte empty plist — exactly how
        // F-005 accumulated ~900 stragglers.
        #expect(!FileManager.default.fileExists(atPath: path.path),
                "Cleanup must unlink ~/Library/Preferences/\(name).plist; it survived at \(path.path)")
    }

    @Test("F-005 regression: freshDefaults registers the minted suite name for teardown")
    func freshDefaultsRegistersForCleanup() {
        let before = createdSuiteNames.count
        _ = freshDefaults()
        _ = freshDefaults()
        #expect(createdSuiteNames.count == before + 2)
        // Every registered name must carry the ``rapid-sampling-test-``
        // prefix so a future test cleanup pass (or a developer's
        // ``find -delete`` salvage script) can match by glob.
        for n in createdSuiteNames.suffix(2) {
            #expect(n.hasPrefix("rapid-sampling-test-"))
        }
    }

    // MARK: - #161 enableThinking

    @Test("#161 default: enableThinking is OFF on first launch — matches ChatGPT / Claude Desktop")
    func enableThinkingDefaultOff() {
        let s = SamplingConfig(defaults: freshDefaults())
        #expect(s.enableThinking == false)
        #expect(SamplingConfig.enableThinkingDefault == false)
    }

    @Test("#161 enableThinking mutation persists under the v0 keyspace")
    func enableThinkingPersists() {
        let d = freshDefaults()
        let prefix = "rapid.sampling.v0"
        let s = SamplingConfig(defaults: d, keyPrefix: prefix)
        s.enableThinking = true
        #expect(d.bool(forKey: "\(prefix).enableThinking") == true)
        s.enableThinking = false
        #expect(d.bool(forKey: "\(prefix).enableThinking") == false)
    }

    @Test("#161 enableThinking survives reinstantiation — the toggle remembers across launches")
    func enableThinkingReloads() {
        let d = freshDefaults()
        let s1 = SamplingConfig(defaults: d)
        s1.enableThinking = true
        let s2 = SamplingConfig(defaults: d)
        #expect(s2.enableThinking == true)
    }

    @Test("#161 resetToDefaults reverts enableThinking + isAtDefaults reflects it")
    func enableThinkingResetAndIsAtDefaults() {
        let s = SamplingConfig(defaults: freshDefaults())
        #expect(s.isAtDefaults)
        s.enableThinking = true
        #expect(!s.isAtDefaults)
        s.resetToDefaults()
        #expect(s.enableThinking == false)
        #expect(s.isAtDefaults)
    }

    @Test(
        "Zero / negative on a Double knob WOULD persist as 0 via UserDefaults.double(forKey:) — verify the absent-key fallback handles that distinction",
        .timeLimit(.minutes(1))
    )
    func zeroDoubleDistinguishedFromAbsent() {
        // The init uses ``object(forKey:)`` precisely so a missing
        // key on first launch doesn't read as 0.0 (which would
        // clobber the safe default). This test pins that semantic.
        let d = freshDefaults()
        // Don't touch the defaults — simulate a fresh install.
        let s = SamplingConfig(defaults: d)
        #expect(s.temperature == 0.7)  // NOT 0.0 from a missing key
    }
}
