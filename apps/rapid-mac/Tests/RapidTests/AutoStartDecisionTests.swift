import Foundation
import Testing
@testable import Rapid

/// Contract for ``AutoStartDecision`` — the pure 3-condition gate
/// that drives the launch-time auto-start of the bundled rapid-mlx
/// sidecar (issue #223).
///
/// The full truth table (2^3 = 8 combinations) is exercised below so
/// any future tweak to the gate's logic surfaces as a single named
/// failure rather than a silent UX regression on real Macs.
///
/// **Repro for the issue** — the original v0.6.11 cliclick walk:
/// cleared ``defaults``, launched the app, and observed the chat
/// surface staying Idle because there was no last-served alias, the
/// production DMG ships no bundled snapshot (``BUNDLE_MODEL=0``), and
/// the helper that should have picked up cached aliases from
/// ``~/.cache/huggingface/hub/`` did not exist. The
/// ``decision_fires_when_only_cached_alias_present`` case below pins
/// that scenario — it would have FAILED before this fix because the
/// helper itself didn't exist and the launch path returned silently.
@Suite("AutoStartDecision — launch-time 3-condition gate (#223)")
struct AutoStartDecisionTests {

    // MARK: - Repro for issue #223 (Step 0 of the fix SOP)

    /// The smoking gun: user has cached aliases from prior sessions,
    /// ``defaults`` was cleared so ``lastServedAlias`` is nil, the
    /// production DMG ships no bundled snapshot so
    /// ``bundledFallbackAlias`` is nil — pre-fix the launch hook fell
    /// through and the sidecar stayed Idle. Post-fix the helper picks
    /// the alphabetically-first cached alias and auto-starts.
    @Test("#223 repro: cached alias on disk + no last-served + no bundled → auto-start picks the cached alias")
    func reproAutoStartFromCachedAlias() {
        let decision = AutoStartDecision.decide(
            lastServedAlias: nil,
            bundledFallbackAlias: nil,
            binaryReachable: true,
            cachedAliases: ["qwen3.5-4b-4bit", "gemma3-1b-qat-4bit"],
            serverState: .idle
        )
        // Sorted pick → "gemma3-1b-qat-4bit" comes before
        // "qwen3.5-4b-4bit" alphabetically.
        #expect(decision == .start(alias: "gemma3-1b-qat-4bit"))
    }

    // MARK: - Full 2^3 truth table over the three gates

    /// One row of the truth table. ``aliasResolvable`` is shorthand
    /// for "any of lastServed / bundled / cached produces a value" —
    /// the helper itself walks the precedence ladder.
    struct TruthRow {
        let aliasResolvable: Bool
        let binaryReachable: Bool
        let modelCached: Bool
        let expected: AutoStartDecision
    }

    @Test("Truth table: 2^3 combinations of (alias × binary × cached) map to documented decisions")
    func truthTable() {
        let presentAlias = "bonsai-1.7b-2bit"
        let rows: [TruthRow] = [
            // 0 0 0 — nothing holds
            .init(
                aliasResolvable: false,
                binaryReachable: false,
                modelCached: false,
                expected: .skip(reason: .binaryMissing)
            ),
            // 0 0 1 — model on disk but no alias resolves (impossible
            // in practice because a cached entry IS a resolvable
            // alias; we still cover the matrix for completeness — the
            // helper picks the cached alias as the resolver fallback,
            // but binary is missing so still skip)
            .init(
                aliasResolvable: false,
                binaryReachable: false,
                modelCached: true,
                expected: .skip(reason: .binaryMissing)
            ),
            // 0 1 0 — binary reachable, no alias, nothing cached
            .init(
                aliasResolvable: false,
                binaryReachable: true,
                modelCached: false,
                expected: .skip(reason: .noResolvableAlias)
            ),
            // 0 1 1 — binary reachable, alias derives from cached set,
            // model cached → start
            .init(
                aliasResolvable: false,
                binaryReachable: true,
                modelCached: true,
                expected: .start(alias: presentAlias)
            ),
            // 1 0 0 — alias resolves but binary missing
            .init(
                aliasResolvable: true,
                binaryReachable: false,
                modelCached: false,
                expected: .skip(reason: .binaryMissing)
            ),
            // 1 0 1 — alias + cached, but binary missing
            .init(
                aliasResolvable: true,
                binaryReachable: false,
                modelCached: true,
                expected: .skip(reason: .binaryMissing)
            ),
            // 1 1 0 — alias + binary, model NOT cached → promptDownload
            .init(
                aliasResolvable: true,
                binaryReachable: true,
                modelCached: false,
                expected: .promptDownload(alias: presentAlias)
            ),
            // 1 1 1 — everything holds → start
            .init(
                aliasResolvable: true,
                binaryReachable: true,
                modelCached: true,
                expected: .start(alias: presentAlias)
            ),
        ]

        for row in rows {
            let lastServed: String? = row.aliasResolvable ? presentAlias : nil
            let cached: Set<String> = row.modelCached ? [presentAlias] : []
            let actual = AutoStartDecision.decide(
                lastServedAlias: lastServed,
                bundledFallbackAlias: nil,
                binaryReachable: row.binaryReachable,
                cachedAliases: cached,
                serverState: .idle
            )
            #expect(
                actual == row.expected,
                "row(alias=\(row.aliasResolvable),bin=\(row.binaryReachable),cached=\(row.modelCached)) → expected \(row.expected), got \(actual)"
            )
        }
    }

    // MARK: - Alias resolution precedence

    @Test("Alias precedence: lastServed wins over bundled and cached")
    func lastServedWins() {
        let decision = AutoStartDecision.decide(
            lastServedAlias: "qwen3.6-35b-4bit",
            bundledFallbackAlias: "bonsai-1.7b-2bit",
            binaryReachable: true,
            cachedAliases: ["gemma3-1b-qat-4bit", "qwen3.6-35b-4bit"],
            serverState: .idle
        )
        #expect(decision == .start(alias: "qwen3.6-35b-4bit"))
    }

    @Test("Alias precedence: bundled wins over cached when no last-served")
    func bundledBeatsCached() {
        let decision = AutoStartDecision.decide(
            lastServedAlias: nil,
            bundledFallbackAlias: "bonsai-1.7b-2bit",
            binaryReachable: true,
            cachedAliases: ["bonsai-1.7b-2bit", "qwen3.6-35b-4bit"],
            serverState: .idle
        )
        // Bundled alias is also in the cached set so we hit ``.start``
        // (real production: bundled snapshot symlinked into HF cache
        // means ``ModelCatalog.load`` reports it as cached).
        #expect(decision == .start(alias: "bonsai-1.7b-2bit"))
    }

    @Test("Alias precedence: lastServed honoured even when not cached → promptDownload")
    func lastServedNotCachedPromptsDownload() {
        let decision = AutoStartDecision.decide(
            lastServedAlias: "qwen3.5-122b-8bit",
            bundledFallbackAlias: nil,
            binaryReachable: true,
            cachedAliases: ["bonsai-1.7b-2bit"],
            serverState: .idle
        )
        // The user's prior intent (qwen3.5-122b-8bit) is honoured,
        // surfaced as a download-aware CTA — don't silently swap to
        // the smaller cached model behind their back.
        #expect(decision == .promptDownload(alias: "qwen3.5-122b-8bit"))
    }

    @Test("Alias precedence: alphabetically-first cached alias chosen when no last-served + no bundled")
    func cachedFallbackUsesAlphabeticalOrder() {
        let decision = AutoStartDecision.decide(
            lastServedAlias: nil,
            bundledFallbackAlias: nil,
            binaryReachable: true,
            cachedAliases: [
                "qwen3.6-35b-4bit",
                "gemma3-1b-qat-4bit",
                "qwen3.5-4b-4bit",
                "phi-4-mini-4bit",
            ],
            serverState: .idle
        )
        // localizedStandardCompare: "gemma3-1b-qat-4bit" first.
        #expect(decision == .start(alias: "gemma3-1b-qat-4bit"))
    }

    // MARK: - Whitespace / empty-string hardening

    @Test("Empty / whitespace-only lastServed treated as nil")
    func whitespaceLastServedTreatedAsNil() {
        let decision = AutoStartDecision.decide(
            lastServedAlias: "   ",
            bundledFallbackAlias: "bonsai-1.7b-2bit",
            binaryReachable: true,
            cachedAliases: ["bonsai-1.7b-2bit"],
            serverState: .idle
        )
        #expect(decision == .start(alias: "bonsai-1.7b-2bit"))
    }

    // MARK: - Idempotency / server-state precedence

    @Test("Idempotency: serverState != .idle short-circuits before any other gate")
    func nonIdleSkips() {
        let states: [ServerState] = [
            .starting(alias: "bonsai-1.7b-2bit"),
            .ready(alias: "bonsai-1.7b-2bit"),
            .stopped,
            .crashed(alias: "bonsai-1.7b-2bit", message: "boom"),
        ]
        for state in states {
            let decision = AutoStartDecision.decide(
                lastServedAlias: "bonsai-1.7b-2bit",
                bundledFallbackAlias: nil,
                binaryReachable: true,
                cachedAliases: ["bonsai-1.7b-2bit"],
                serverState: state
            )
            #expect(
                decision == .skip(reason: .serverNotIdle),
                "state \(state) must skip, got \(decision)"
            )
        }
    }

    @Test("Idempotency: .missing serverState reports binaryMissing (distinct skip reason)")
    func missingStateMaps() {
        let decision = AutoStartDecision.decide(
            lastServedAlias: "bonsai-1.7b-2bit",
            bundledFallbackAlias: nil,
            binaryReachable: false,
            cachedAliases: [],
            serverState: .missing
        )
        #expect(decision == .skip(reason: .binaryMissing))
    }

    // MARK: - Empty-state CTA copy

    @Test("promptDownloadCaption: alias + size → 'Click Start to download <alias> (<size>).'")
    func captionWithSize() {
        let copy = AutoStartDecision.promptDownloadCaption(
            alias: "qwen3.5-122b-8bit",
            sizeText: "~65 GB"
        )
        #expect(copy == "Click Start to download qwen3.5-122b-8bit (~65 GB).")
    }

    @Test("promptDownloadCaption: missing size degrades to size-less variant")
    func captionWithoutSize() {
        let copy = AutoStartDecision.promptDownloadCaption(
            alias: "qwen3.5-4b-4bit",
            sizeText: nil
        )
        #expect(copy == "Click Start to download qwen3.5-4b-4bit.")
    }

    @Test("promptDownloadCaption: empty-string size also degrades cleanly")
    func captionWithEmptySize() {
        let copy = AutoStartDecision.promptDownloadCaption(
            alias: "qwen3.5-4b-4bit",
            sizeText: ""
        )
        #expect(copy == "Click Start to download qwen3.5-4b-4bit.")
    }

    // MARK: - Codex r1 MAJOR: cached-fallback fit rejection

    /// The cached-fallback tier MUST respect ``rejectsAlias`` (in
    /// production wired to ``ModelSizing.classify`` rejecting
    /// ``.tooBig``). A cleared-defaults user with a single oversized
    /// cached model on disk should NOT get auto-OOM'd — the resolver
    /// should fall through to ``.noResolvableAlias`` and let the
    /// existing "Pick a model from the top bar" copy own the frame.
    @Test("Codex r1 MAJOR: cached-fallback respects rejectsAlias; sole oversized cached entry → noResolvableAlias")
    func cachedFallbackRejectsOversized() {
        let decision = AutoStartDecision.decide(
            lastServedAlias: nil,
            bundledFallbackAlias: nil,
            binaryReachable: true,
            cachedAliases: ["qwen3.5-122b-8bit"],
            serverState: .idle,
            rejectsAlias: { $0 == "qwen3.5-122b-8bit" }
        )
        #expect(decision == .skip(reason: .noResolvableAlias))
    }

    @Test("Codex r1 MAJOR: rejectsAlias skips the alphabetically-first cached entry when rejected, picks the next acceptable one")
    func cachedFallbackSkipsRejectedAlias() {
        // Codex r2 MINOR fix: rejected alias MUST be the
        // alphabetically-first candidate, otherwise the predicate
        // is never load-bearing and the test name lies about what
        // it covers. Here ``aaa-huge-model`` sorts first and is
        // rejected; the resolver must skip past it to the next
        // acceptable alias rather than returning ``.noResolvableAlias``.
        let decision = AutoStartDecision.decide(
            lastServedAlias: nil,
            bundledFallbackAlias: nil,
            binaryReachable: true,
            cachedAliases: [
                "aaa-huge-model",      // alphabetically first, rejected
                "qwen3.5-4b-4bit",     // alphabetically next, acceptable
                "qwen3.6-35b-4bit",    // also acceptable
            ],
            serverState: .idle,
            rejectsAlias: { $0 == "aaa-huge-model" }
        )
        // localizedStandardCompare order: aaa-huge-model (rejected) → qwen3.5-4b-4bit (wins).
        #expect(decision == .start(alias: "qwen3.5-4b-4bit"))
    }

    @Test("Codex r1 MAJOR: rejectsAlias does NOT filter lastServed (explicit prior intent honoured)")
    func rejectsAliasDoesNotFilterLastServed() {
        let decision = AutoStartDecision.decide(
            lastServedAlias: "qwen3.5-122b-8bit",
            bundledFallbackAlias: nil,
            binaryReachable: true,
            cachedAliases: ["qwen3.5-122b-8bit"],
            serverState: .idle,
            // Reject everything — proves the predicate is only
            // consulted for the cached-fallback tier.
            rejectsAlias: { _ in true }
        )
        #expect(decision == .start(alias: "qwen3.5-122b-8bit"))
    }

    @Test("Codex r1 MAJOR: rejectsAlias does NOT filter bundled fallback (product-curated)")
    func rejectsAliasDoesNotFilterBundled() {
        let decision = AutoStartDecision.decide(
            lastServedAlias: nil,
            bundledFallbackAlias: "bonsai-1.7b-2bit",
            binaryReachable: true,
            cachedAliases: ["bonsai-1.7b-2bit"],
            serverState: .idle,
            rejectsAlias: { _ in true }
        )
        #expect(decision == .start(alias: "bonsai-1.7b-2bit"))
    }

    @Test("Codex r1 MAJOR: default rejectsAlias is the always-false predicate (back-compat)")
    func defaultRejectsAliasIsNoop() {
        // Existing callers that don't pass ``rejectsAlias`` see the
        // pre-fix behaviour — every cached alias is acceptable.
        let decision = AutoStartDecision.decide(
            lastServedAlias: nil,
            bundledFallbackAlias: nil,
            binaryReachable: true,
            cachedAliases: ["qwen3.5-122b-8bit"],
            serverState: .idle
        )
        #expect(decision == .start(alias: "qwen3.5-122b-8bit"))
    }

    @Test("Codex r2 MINOR: rejectsAlias never invoked when cachedAliases is empty — no crash, clean noResolvableAlias")
    func rejectsAliasNotCalledWithEmptyCache() {
        // Defensive: the predicate must not be invoked on an empty
        // candidate set. We pass a deliberately-throwing predicate
        // to prove the resolver short-circuits before any call site.
        // Test passes if no crash occurs and the decision falls
        // through to ``.noResolvableAlias``.
        var invoked = false
        let decision = AutoStartDecision.decide(
            lastServedAlias: nil,
            bundledFallbackAlias: nil,
            binaryReachable: true,
            cachedAliases: [],
            serverState: .idle,
            rejectsAlias: { _ in invoked = true; return true }
        )
        #expect(decision == .skip(reason: .noResolvableAlias))
        #expect(invoked == false)
    }

    // MARK: - FU-1 (v0.7.19 audit): user opt-out for launch-time auto-start

    /// Persisted-pref OFF must win over EVERY other gate, including
    /// the otherwise-green 3-condition path. A user who turned the
    /// toggle off explicitly chose "don't load a model on launch";
    /// the helper must honour that even when last-served + binary +
    /// cached all hold. Surfaces ``.skip(.userOptedOut)`` so callers
    /// can distinguish this from the binary-missing / no-alias paths
    /// (the empty-state CTA copy in ChatView differs between them —
    /// a user who opted out should NOT see a "Click Start to
    /// download X" caption the way an alias-resolved-but-not-cached
    /// path would).
    @Test("FU-1: userOptedIn=false short-circuits even when all 3 gates pass — never auto-spawns")
    func userOptedOutSkipsEvenWhenAllGatesPass() {
        let decision = AutoStartDecision.decide(
            lastServedAlias: "qwen3.5-4b-4bit",
            bundledFallbackAlias: "bonsai-1.7b-2bit",
            binaryReachable: true,
            cachedAliases: ["qwen3.5-4b-4bit", "bonsai-1.7b-2bit"],
            serverState: .idle,
            userOptedIn: false
        )
        #expect(decision == .skip(reason: .userOptedOut))
    }

    /// Opt-out precedence is HIGHEST — it must also beat the
    /// ``.binaryMissing`` and ``.noResolvableAlias`` skips, not just
    /// ``.start`` / ``.promptDownload``. A future telemetry consumer
    /// bucketing by skip reason needs the opt-out signal to dominate
    /// so "user explicitly said no" is never miscounted as
    /// "infrastructure problem."
    @Test("FU-1: userOptedIn=false beats binaryMissing and noResolvableAlias precedence")
    func userOptedOutPrecedence() {
        // Binary missing + no alias would otherwise return
        // ``.skip(.binaryMissing)``; opt-out must override.
        let binaryMissing = AutoStartDecision.decide(
            lastServedAlias: nil,
            bundledFallbackAlias: nil,
            binaryReachable: false,
            cachedAliases: [],
            serverState: .idle,
            userOptedIn: false
        )
        #expect(binaryMissing == .skip(reason: .userOptedOut))

        // No alias resolves; binary present. Without opt-out this
        // would be ``.skip(.noResolvableAlias)``.
        let noAlias = AutoStartDecision.decide(
            lastServedAlias: nil,
            bundledFallbackAlias: nil,
            binaryReachable: true,
            cachedAliases: [],
            serverState: .idle,
            userOptedIn: false
        )
        #expect(noAlias == .skip(reason: .userOptedOut))

        // Server already running; would normally skip with
        // ``.serverNotIdle``. Opt-out still wins so the skip-reason
        // bucket is consistent across all paths the user took.
        let serverBusy = AutoStartDecision.decide(
            lastServedAlias: "qwen3.5-4b-4bit",
            bundledFallbackAlias: nil,
            binaryReachable: true,
            cachedAliases: ["qwen3.5-4b-4bit"],
            serverState: .ready(alias: "qwen3.5-4b-4bit"),
            userOptedIn: false
        )
        #expect(serverBusy == .skip(reason: .userOptedOut))
    }

    /// Back-compat anchor: the ``userOptedIn`` parameter defaults to
    /// ``true``, so every existing call site (production
    /// ``ContentView`` + every test in this file that doesn't pass
    /// the parameter) sees pre-FU-1 behavior. A future refactor that
    /// changes the default would break dozens of existing tests;
    /// pinning the default value explicitly here surfaces that as a
    /// single named failure with a clear intent comment.
    @Test("FU-1: userOptedIn defaults to true — back-compat for all pre-FU-1 callers")
    func userOptedInDefaultIsTrue() {
        // No ``userOptedIn`` argument — must behave as if the user
        // explicitly opted in (the pre-FU-1 contract).
        let decision = AutoStartDecision.decide(
            lastServedAlias: "qwen3.5-4b-4bit",
            bundledFallbackAlias: nil,
            binaryReachable: true,
            cachedAliases: ["qwen3.5-4b-4bit"],
            serverState: .idle
        )
        #expect(decision == .start(alias: "qwen3.5-4b-4bit"))
    }

    /// FU-1 storage-key / default-value lock. The persisted pref's
    /// key is consumed by both the UI (Settings → Models toggle) and
    /// the launch hook (``ContentView``'s ``@AppStorage``); a typo
    /// in either spot is a silent disconnect (toggle writes one key,
    /// launch hook reads another). Pin both values here so a rename
    /// or default flip surfaces immediately. The ``v1`` suffix is
    /// load-bearing: a future shape change opts into a new key
    /// without inheriting the v1 user's stored value silently.
    @Test("FU-1: AutoStartPreference storage key + default value are pinned (UI ↔ launch-hook contract)")
    func autoStartPreferenceContract() {
        #expect(AutoStartPreference.storageKey == "rapid.server.auto_start_on_launch.v1")
        // Default ``true`` is the load-bearing contract: existing
        // users see no behavior change after upgrade. Flipping this
        // to ``false`` would silently break every v0.7.x install on
        // first launch after the bump.
        #expect(AutoStartPreference.defaultValue == true)
    }

    /// FU-1 mid-session flip safety: a user who toggles the pref ON
    /// mid-session (the toggle was OFF at process launch, the user
    /// then flipped it on in Settings) must still be able to use
    /// the helper with ``userOptedIn: true`` and get the normal
    /// 3-gate behavior. Pins the "the toggle is the SOLE governor —
    /// no hidden process-launch latch" contract so a future refactor
    /// that caches the value into a let-bound constant at app boot
    /// would surface here.
    @Test("FU-1: mid-session opt-in (userOptedIn=true after a launch with it OFF) honors the gate normally")
    func midSessionOptInRespectsGate() {
        let decision = AutoStartDecision.decide(
            lastServedAlias: "qwen3.5-4b-4bit",
            bundledFallbackAlias: nil,
            binaryReachable: true,
            cachedAliases: ["qwen3.5-4b-4bit"],
            serverState: .idle,
            userOptedIn: true
        )
        #expect(decision == .start(alias: "qwen3.5-4b-4bit"))
    }

    // MARK: - Issue #356: SkipReason cardinality contract

    /// Pins the SkipReason case set so a future PR that adds a fifth
    /// reason (e.g. ``.userOptedOutMidSession``) MUST also touch this
    /// test — at which point the dev is forced to audit:
    ///
    ///   * the precedence ladder in ``AutoStartDecision.decide`` (where
    ///     does the new gate sit relative to ``.userOptedOut`` /
    ///     ``.serverNotIdle`` / ``.binaryMissing`` / ``.noResolvableAlias``?);
    ///   * every consumer of ``SkipReason`` — today only ``==``
    ///     comparisons in this test file and ``HideAlwaysOrthogonalToAutoStartTests``;
    ///     tomorrow potentially a telemetry counter or a
    ///     diagnostics-bucketing ``switch reason { default: ... }``
    ///     that would otherwise silently absorb the new case.
    ///
    /// Same shape as the FU-1 ``AutoStartPreference.storageKey`` /
    /// ``defaultValue`` pin above: a single named failure with a
    /// clear intent comment beats a silent UX regression months later.
    /// Pinning the explicit case-name set (not just ``count``) means a
    /// rename also surfaces here — a rename without a deliberate test
    /// update would otherwise leave the count pinned at 4 and slip
    /// through unnoticed.
    @Test("#356: SkipReason cardinality + case-name set is pinned — adding/renaming a case requires explicit test update")
    func skipReasonCardinality() {
        let allCases = AutoStartDecision.SkipReason.allCases
        #expect(allCases.count == 5)
        // Pin the exact case-name set via the raw-string backing so a
        // rename (``.userOptedOut`` → ``.userExplicitlyOptedOut``)
        // surfaces here even though ``count`` is unchanged.
        let names = Set(allCases.map(\.rawValue))
        #expect(names == [
            "userOptedOut",
            "serverNotIdle",
            "binaryMissing",
            "noResolvableAlias",
            // Added 2026-08-05 with the retired-starter swap. This test did
            // exactly what its docstring promised: the suite was dormant at
            // the time, so nothing forced the audit it demands, and the
            // precedence question it asks — where does the new gate sit? —
            // took several review rounds to answer instead of one red test.
            "retiredStarter",
        ])
    }
}
