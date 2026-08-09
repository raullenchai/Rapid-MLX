import Foundation
import Observation

/// Fully-resolved sampling values placed on the streaming wire body.
/// Relocated here from the (now-removed) Presets subsystem because it
/// sits on the ``ChatViewModel`` streaming path, which the minimal
/// menu-bar app keeps.
struct ResolvedSampling: Equatable, Sendable {
    var temperature: Double
    var topP: Double
    var maxTokens: Int
    var repetitionPenalty: Double
    var enableThinking: Bool
}

/// Persisted sampling knobs (temperature, top_p, max_tokens,
/// repetition_penalty) exposed in the Settings → Sampling panel.
/// Backs the wire-body construction inside ``ChatViewModel`` so a
/// power user can deviate from the v0.4.1 hard-coded defaults
/// without touching code.
///
/// Why these four:
///   * ``temperature`` and ``top_p`` — daily-driver knobs for
///     "make this more deterministic" / "let it think wider".
///   * ``max_tokens`` — the hardcoded 4096 default works for chat
///     but a coding-heavy session blowing 8 K of context-window
///     wants a higher ceiling; a tiny-prompt latency-test wants
///     a tighter cap.
///   * ``repetition_penalty`` — the dominant break-out lever
///     when a small fine-tuned hybrid loops. Hidden under a "show
///     advanced" toggle so a casual user can't accidentally
///     disable it (1.0 → no penalty → the model's prone to
///     looping, which looks like "the app is broken").
///
/// ``frequency_penalty`` / ``presence_penalty`` deliberately stay
/// at v0.4.1's 0.0 / 0.0 — see ``ChatStreamClient.Request`` doc
/// for the dispatch rationale. Exposing them would tempt users
/// into the same "scatter into rare Unicode" regression that v0.4.1
/// solved by retiring them.
///
/// Persistence: each value lives under its own ``UserDefaults`` key
/// in the ``v0`` namespace. Bumping ``v0`` → ``v1`` is the safe
/// migration path if we ever change the contract.
@MainActor
@Observable
final class SamplingConfig {
    /// Default sampling profile — matches the v0.4.12 hard-coded
    /// constants. ``ChatStreamClient.Request`` uses these defaults
    /// when nothing else is wired, so any test that doesn't pass an
    /// override still gets the same behaviour as the production
    /// app's first-run state.
    static let temperatureDefault: Double = 0.7
    static let topPDefault: Double = 0.95
    static let maxTokensDefault: Int = 4096
    static let repetitionPenaltyDefault: Double = 1.1
    /// #161: hybrid-thinking models (Qwen 3 / 3.5 / 3.6, GLM 4.7,
    /// the Qwopus series) emit a ``<think>...</think>`` reasoning
    /// trace BEFORE the final answer. On a 4 B / 9 B-class model
    /// with the default ``maxTokens = 4096`` budget, the reasoning
    /// trace alone routinely consumes 10-13 K characters of
    /// completion tokens and the stream hits ``finish_reason =
    /// "length"`` with ``content = ""`` — the UI shows a 25-30 s
    /// spinner and an empty assistant bubble. This is what real
    /// first-time users on 18 GB MacBooks saw as "prompts don't
    /// work" (cliclick triage on 2026-06-14).
    ///
    /// Default OFF matches what ChatGPT / Claude desktop ship — both
    /// hide thinking under the hood by default. Power users who
    /// want chain-of-thought on hard reasoning prompts can flip
    /// this back on per-session from Settings → Sampling.
    static let enableThinkingDefault: Bool = false

    /// Cycle-3 fix (rapid-desktop loop, 2026-06-19) — per-alias
    /// floors that take effect when ``/v1/models`` reports
    /// ``reasoning_parser != null`` (Hermes / DeepSeek-R1 /
    /// VibeThinker / GLM-4.7 / GPT-OSS / Minimax / Qwen3.x and
    /// any future addition). vibethinker-3b-8bit on a trivial
    /// "27 + 45?" tool prompt burns 1,697 reasoning tokens before
    /// emitting the first tool_call; at ``max_tokens = 512`` the
    /// same prompt returns ``content = null, tool_calls = null,
    /// finish_reason = "length"`` (cycle-2 fuzz-correctness P1
    /// repro on 2026-06-19). PR #317 cured the *symptom*
    /// (rendering ``reasoning_content`` with a "Cut off mid-
    /// thought" footer). This floor cures the *cause*: when a
    /// reasoning alias swap fires AND the user hasn't manually
    /// overridden the slider, the effective budget lifts to
    /// values that fit a typical reasoning trace + final answer.
    ///
    /// Chat floor ``2048`` — empirically covers the median
    /// reasoning trace + a short answer on Qwen 3.x reasoning,
    /// DeepSeek-R1-distill, VibeThinker, and the hermes-parsed
    /// reasoning-finetunes we ship in ``aliases.json``. Today the
    /// non-reasoning baseline ``maxTokensDefault = 4096`` already
    /// exceeds this; the constant pins the contract so a future
    /// non-reasoning floor lowering can't silently downshift a
    /// reasoning alias below 2,048.
    ///
    /// Tools floor ``16384`` — a multi-tool prompt on a reasoning
    /// alias routinely emits 3-4k completion tokens before the
    /// *second* tool_call fires (cycle-2 repro at 4,096 used 3,407
    /// for the single-tool 27+45 case). Tools-on auto-scales to
    /// this floor instead of the chat one so a reasoning-finetune
    /// with a heavy preamble doesn't blow the budget mid-tool-call.
    ///
    /// This floor sat at ``4_096`` — identical to
    /// ``maxTokensDefault`` — from the day it was introduced, so
    /// ``max(maxTokens, effectiveToolsFloor)`` resolved to 4,096 on
    /// every default install. **It never lifted anyone by a single
    /// token.** The intent was recorded; the effect was not.
    ///
    /// What that cost, measured on 2026-08-09 through the shipped
    /// GUI against `qwen3.5-4b-4bit` — the 16 GB tier's own
    /// recommendation — asking one live web-research question five
    /// times:
    ///
    /// | outcome | runs |
    /// |---|---|
    /// | tool budget exhausted, no answer | 2 |
    /// | hit max_tokens with EMPTY content | 2 |
    /// | answered | 1 |
    ///
    /// and the one that answered opened with "I can't access
    /// real-time news or browse live pages" before contradicting
    /// itself from the search results it had just been handed.
    ///
    /// Sizing this by measured consumption would be the wrong
    /// instrument. ``max_tokens`` is a safety ceiling, not a budget
    /// allocation: its job is to be high enough that legitimate work
    /// never reaches it and low enough that a runaway is cut off in
    /// tolerable time. Fitting it to a p95 of observed usage kills
    /// the other 5% by construction, and would need re-deriving for
    /// every new tool, model and quantisation.
    ///
    /// So it is set from the two real costs instead:
    ///
    ///   * **Memory: none.** The KV cache grows with tokens actually
    ///     produced. The one place ``max_tokens`` could reserve
    ///     capacity up front is
    ///     ``SchedulerConfig.metal_cap_kv_bytes_per_token``, which
    ///     defaults to ``0`` (disabled) and which the desktop never
    ///     sets. Worst case for this model is 32 KB/token over its 8
    ///     full-attention layers — 512 MB if a turn genuinely runs to
    ///     16,384, against 2.9 GB of weights. Nothing is spent until
    ///     the tokens are.
    ///   * **Time: real.** At ~60 tok/s this is ~4.5 min of decode
    ///     before the ceiling fires, and the tool loop allows up to
    ///     four rounds. That is the price, and it is the right one to
    ///     pay: a turn that runs long and then answers beats a turn
    ///     that stops early and delivers nothing.
    ///
    /// A model that genuinely needs different numbers gets them
    /// per-alias from the engine via
    /// ``ServerModelProfile.reasoningToolsFloor`` rather than by
    /// moving this global.
    ///
    /// Both floors are observed only via ``applyServerProfile``
    /// + ``effectiveMaxTokens(toolsEnabled:)``, so a user who
    /// drags the Settings → Sampling slider AT ALL (above or
    /// below) takes immediate priority — auto-scale never fights
    /// an explicit user choice.
    static let reasoningChatFloor: Int = 2_048
    static let reasoningToolsFloor: Int = 16_384
    /// FU-3 (post-v0.7.19) canonical names. ``reasoningChatFloor``
    /// and ``reasoningToolsFloor`` are the GLOBAL fallbacks the
    /// per-alias override (``ServerModelProfile.reasoningChatFloor``
    /// / ``reasoningToolsFloor``) shadows; the ``default…``
    /// aliases below give call sites a name that reads
    /// unambiguously as "the value used when the profile is silent".
    /// Both pairs intentionally point at the same constant — the
    /// alias exists for readability at the call site, not to let
    /// the two drift apart. A future maintainer who wants to
    /// change one MUST change both (a compile-time test pins this).
    static var defaultReasoningChatFloor: Int { reasoningChatFloor }
    static var defaultReasoningToolsFloor: Int { reasoningToolsFloor }

    /// Min / max ranges the SettingsView sliders clamp to. Picked
    /// to keep "obviously broken" values off the slider — there's
    /// no use case for ``temperature = 5.0`` on this class of
    /// local model, and the user can always reset to defaults.
    static let temperatureRange: ClosedRange<Double> = 0.0 ... 2.0
    static let topPRange: ClosedRange<Double> = 0.05 ... 1.0
    // Floor is 256 (not 128): the Settings stepper moves in 256-token
    // increments, and a sub-256 reply cap has no real use on this class
    // of model. Keeping the canonical clamp aligned to the step means a
    // persisted / server-clamped value can never land on an
    // unreachable 128.
    static let maxTokensRange: ClosedRange<Int> = 256 ... 32_768
    static let repetitionPenaltyRange: ClosedRange<Double> = 1.0 ... 1.5

    private let defaults: UserDefaults
    private let keyPrefix: String

    /// Cycle-3 fix — captured from the last ``applyServerProfile``
    /// call so the request-builder in ``ChatViewModel`` can lift
    /// ``maxTokens`` for a reasoning alias without re-fetching the
    /// profile. ``nil`` for non-reasoning aliases, an old rapid-mlx
    /// server (<0.7.4) that doesn't ship the vendor extension, or
    /// before any profile has been observed. In-memory only; not
    /// persisted, so a relaunch reads ``nil`` until the next
    /// ``.task(id: server.servingAlias)`` fires in ``RapidApp``.
    private(set) var activeReasoningParser: String?

    /// FU-3 (post-v0.7.19) — per-alias override for the chat-mode
    /// floor, snapshotted from ``ServerModelProfile.reasoningChatFloor``
    /// during ``applyServerProfile``. ``nil`` means the global
    /// default (``defaultReasoningChatFloor``) applies. Reset on
    /// ``clearActiveReasoningParser`` so an alias swap can't carry
    /// the previous alias's floor into a new conversation.
    private(set) var activeReasoningChatFloor: Int?

    /// FU-3 — per-alias override for the tools-mode floor, mirror
    /// of ``activeReasoningChatFloor``. Surfaced through
    /// ``effectiveMaxTokens(toolsEnabled: true)``.
    private(set) var activeReasoningToolsFloor: Int?

    /// Issue #363 — engine-advertised context window for the active
    /// alias, captured from ``ServerModelProfile.contextWindow`` on
    /// the last ``applyServerProfile`` call. ``nil`` for an older
    /// rapid-mlx sidecar (< 0.8.4) that doesn't emit the field, or
    /// before any profile has been observed. ``ChatViewModel`` reads
    /// this when constructing a request body — wins over the
    /// per-family heuristic in ``ModelInfoCatalog`` because the
    /// server reads the loaded engine's ``max_position_embeddings``
    /// directly (the canonical source of truth) and the heuristic
    /// drifts out of sync with every long-context release. Reset on
    /// ``clearActiveReasoningParser`` so an alias swap can't carry
    /// the previous alias's window into a new conversation. In-memory
    /// only; not persisted.
    private(set) var activeContextWindow: Int?

    /// Cycle-3 fix — ``true`` when our auto-scale write was the
    /// last thing that touched ``maxTokens``. Any user-initiated
    /// mutation through the public setter flips this back to
    /// ``false`` (see the ``maxTokens.didSet`` override), so the
    /// "respect user override" rule holds without a second flag.
    private var maxTokensIsAutoScaled: Bool = false

    var temperature: Double {
        didSet {
            let clamped = Self.clamped(temperature, to: Self.temperatureRange, fallback: Self.temperatureDefault)
            guard clamped == temperature else {
                temperature = clamped
                return
            }
            persist(\.temperature, value: temperature)
        }
    }
    var topP: Double {
        didSet {
            let clamped = Self.clamped(topP, to: Self.topPRange, fallback: Self.topPDefault)
            guard clamped == topP else {
                topP = clamped
                return
            }
            persist(\.topP, value: topP)
        }
    }
    var maxTokens: Int {
        didSet {
            let clamped = Self.clamped(maxTokens, to: Self.maxTokensRange)
            guard clamped == maxTokens else {
                maxTokens = clamped
                return
            }
            // Cycle-3 fix — a value-changing public write means the
            // user (or our own auto-scale path, which sets the flag
            // back to ``true`` immediately after) just expressed an
            // explicit choice. Reset the auto-scale flag so the
            // ``effectiveMaxTokens`` request-time bump won't fight a
            // slider drag. The auto-scale path uses
            // ``autoScaleMaxTokens`` below which re-flips the flag
            // after the assignment so its own writes don't get
            // counted as user intent.
            maxTokensIsAutoScaled = false
            persist(\.maxTokens, value: maxTokens)
        }
    }
    var repetitionPenalty: Double {
        didSet {
            let clamped = Self.clamped(
                repetitionPenalty,
                to: Self.repetitionPenaltyRange,
                fallback: Self.repetitionPenaltyDefault
            )
            guard clamped == repetitionPenalty else {
                repetitionPenalty = clamped
                return
            }
            persist(\.repetitionPenalty, value: repetitionPenalty)
        }
    }
    /// #161 hybrid-thinking switch. ``false`` (default) sends
    /// ``chat_template_kwargs: {enable_thinking: false}`` on every
    /// turn so a hybrid model skips the ``<think>...</think>`` block
    /// and emits its answer directly. ``true`` omits the kwarg so
    /// hybrid models behave as upstream (thinking ON); non-hybrid
    /// models ignore the kwarg either way.
    var enableThinking: Bool {
        didSet {
            persist(\.enableThinking, value: enableThinking)
        }
    }

    /// Injectable for tests — production code lets this default
    /// to ``.standard`` so the Settings panel mutates the same
    /// store the v0.4.14 first launch reads from.
    init(
        defaults: UserDefaults = .standard,
        keyPrefix: String = "rapid.sampling.v0"
    ) {
        self.defaults = defaults
        self.keyPrefix = keyPrefix
        // Initial read from UserDefaults with the hard-coded
        // defaults as fallback. ``object(forKey:)`` is preferred
        // over the typed accessor because ``.double(forKey:)``
        // returns 0.0 for a missing key — which would silently
        // clobber the user's first-launch defaults with 0 if we
        // didn't double-check existence.
        self.temperature = Self.clamped(
            defaults.object(forKey: "\(keyPrefix).temperature") as? Double ?? Self.temperatureDefault,
            to: Self.temperatureRange,
            fallback: Self.temperatureDefault
        )
        self.topP = Self.clamped(
            defaults.object(forKey: "\(keyPrefix).topP") as? Double ?? Self.topPDefault,
            to: Self.topPRange,
            fallback: Self.topPDefault
        )
        self.maxTokens = Self.clamped(
            defaults.object(forKey: "\(keyPrefix).maxTokens") as? Int ?? Self.maxTokensDefault,
            to: Self.maxTokensRange
        )
        self.repetitionPenalty = Self.clamped(
            defaults.object(forKey: "\(keyPrefix).repetitionPenalty") as? Double
                ?? Self.repetitionPenaltyDefault,
            to: Self.repetitionPenaltyRange,
            fallback: Self.repetitionPenaltyDefault
        )
        self.enableThinking = (defaults.object(forKey: "\(keyPrefix).enableThinking") as? Bool)
            ?? Self.enableThinkingDefault
    }

    /// Apply curated server-provided ``recommended_sampling`` from
    /// a ``ServerModelProfile``. Only writes through to the knobs
    /// **when the user hasn't manually overridden the defaults**
    /// (``isAtDefaults == true``) — a slider that the user has
    /// explicitly dragged represents their intent and we must not
    /// clobber it on alias swap. Returns ``true`` if any value was
    /// applied so callers can surface a "Using calibrated defaults
    /// for {alias}" banner; ``false`` otherwise.
    ///
    /// Why this lives on ``SamplingConfig`` rather than a free
    /// function: the gating decision (``isAtDefaults``) is owned
    /// by the same type that owns the knobs and their persistence;
    /// keeping the apply logic here means there's exactly one
    /// place that decides "do we respect the user's drag, or
    /// over-write it?" and contract tests at this layer pin the
    /// behaviour.
    ///
    /// Why nil checks per-key rather than wholesale-replace: a
    /// curated profile in ``aliases.json`` may set only a subset
    /// (some models only need a tighter temperature; others want
    /// a custom top_p too). Anything the profile doesn't pin stays
    /// at the v0.4.12 default — which is the safe shape.
    @discardableResult
    func applyServerProfile(_ profile: ServerModelProfile) -> Bool {
        // Cycle-3 fix — capture the reasoning-parser signal even
        // when the recommended-sampling block is empty or when the
        // user has manually overridden the sliders. The signal
        // gates the ``effectiveMaxTokens(toolsEnabled:)`` lift at
        // request time, which is independent of the curated-knobs
        // application below. Without this assignment a reasoning
        // alias would silently lose its tools-on floor whenever
        // ``isAtDefaults == false``, which is exactly the user
        // population most likely to be running tool-heavy prompts
        // (they cared enough to touch the sliders).
        activeReasoningParser = profile.reasoningParser
        // FU-3 — snapshot per-alias floor overrides alongside the
        // parser signal so a profile that ships custom floors lifts
        // the budget consistently across the auto-scale + tools-on
        // request-time bump. Captured even for non-reasoning aliases
        // so a future profile that ships floors WITHOUT a parser
        // (probably never, but the API allows it) doesn't silently
        // drop them on the floor.
        activeReasoningChatFloor = profile.reasoningChatFloor
        activeReasoningToolsFloor = profile.reasoningToolsFloor
        // Issue #363 — snapshot the engine-reported context window so
        // ``ChatViewModel``'s trim helper uses the canonical value the
        // server actually enforces. Guard ``> 0`` so a future server-side
        // regression that emits 0 / negative doesn't pin the trim to
        // an absurd value; ``nil`` falls back to the per-family
        // heuristic in ``ModelInfoCatalog``.
        if let serverCtx = profile.contextWindow, serverCtx > 0 {
            activeContextWindow = serverCtx
        } else {
            activeContextWindow = nil
        }
        // Codex r2 NIT — snapshot ``isAtDefaults`` BEFORE the
        // reasoning auto-scale rewrite, so a future change to
        // ``reasoningChatFloor`` that lifts ``maxTokens`` ABOVE
        // ``maxTokensDefault`` doesn't invalidate the curated
        // sampling gate below. Today both knobs are observationally
        // equivalent (the chat floor 2,048 < baseline 4,096), but
        // if a future maintainer raises the chat floor to 8,192
        // for a heavier reasoning alias, the auto-scale would flip
        // ``isAtDefaults`` to false before the curated-sampling
        // block runs, silently dropping the recommended temperature/
        // top_p / repetition_penalty. The snapshot keeps the
        // contract: "fresh install + curated profile applies both;
        // user override of any knob applies NEITHER".
        let pristineAtCall = isAtDefaults
        // Cycle-3 fix — auto-scale ``maxTokens`` to the reasoning
        // chat floor (2,048) when a reasoning alias is observed
        // AND the user hasn't touched the slider. We deliberately
        // gate on ``!maxTokensIsAutoScaled && maxTokens ==
        // maxTokensDefault`` rather than the broader
        // ``isAtDefaults`` so a power user who set a custom
        // temperature still gets the reasoning floor — temperature
        // and max_tokens are independent intent signals; pinning
        // them together would force "all or nothing" semantics.
        //
        // FU-3 — ``effectiveChatFloor`` honours a per-alias
        // override when present; falls back to the global default.
        if let parser = profile.reasoningParser, !parser.isEmpty,
           !maxTokensIsAutoScaled, maxTokens == Self.maxTokensDefault {
            autoScaleMaxTokens(to: max(maxTokens, effectiveChatFloor))
        }
        guard pristineAtCall else { return false }
        guard let sampling = profile.recommendedSampling else { return false }
        var applied = false
        if let t = sampling["temperature"] {
            temperature = Self.clamped(t, to: Self.temperatureRange, fallback: Self.temperatureDefault)
            applied = true
        }
        if let p = sampling["top_p"] {
            topP = Self.clamped(p, to: Self.topPRange, fallback: Self.topPDefault)
            applied = true
        }
        if let r = sampling["repetition_penalty"] {
            repetitionPenalty = Self.clamped(
                r,
                to: Self.repetitionPenaltyRange,
                fallback: Self.repetitionPenaltyDefault
            )
            applied = true
        }
        // ``top_k``, ``min_p``, ``presence_penalty``, ``frequency_penalty``
        // are honoured server-side (rapid-mlx merges them into the wire body)
        // but not yet surfaced as desktop sliders. Silently passing through
        // is fine: the server already takes them; the desktop just doesn't
        // render the knob. If the user later opens Settings → Sampling, the
        // surfaced sliders will be at their v0.4.12 defaults, which is the
        // honest shape — we haven't applied a value to a knob we can't show.
        return applied
    }

    /// Cycle-3 fix — request-time budget that respects the
    /// reasoning-alias floor + tools amplifier without persisting
    /// anything to ``UserDefaults``. ``ChatViewModel`` calls this
    /// at wire-body construction so a tools-on send to a reasoning
    /// alias gets ``≥ reasoningToolsFloor`` even when the
    /// persisted ``maxTokens`` would otherwise sit at the chat
    /// floor. A user who explicitly dragged the slider (any value
    /// other than the auto-scaled ones) takes priority — the bump
    /// is suppressed.
    ///
    /// Why not persist the tools floor: the tools-on lift is a
    /// per-request property (the same conversation may have
    /// tools-on / tools-off rounds depending on whether the
    /// capability chip is loaded). Persisting it would clobber the
    /// chat floor for the next tools-off turn, and the user-facing
    /// "Max Tokens" slider in Settings would mysteriously read
    /// 4,096 on tools-on tabs and 2,048 elsewhere. Keeping it
    /// request-scoped is the honest shape.
    func effectiveMaxTokens(toolsEnabled: Bool) -> Int {
        guard let parser = activeReasoningParser, !parser.isEmpty else {
            return maxTokens
        }
        // Respect user override — only auto-bump when EITHER:
        //   (a) our own auto-scale path wrote the current value
        //       (``maxTokensIsAutoScaled == true``), OR
        //   (b) the value still sits at the v0.4.12 baseline
        //       (``maxTokensDefault``) — i.e. the user has never
        //       touched the slider since first launch, so a parser
        //       observed on a relaunch should still bump.
        //
        // Codex r1 MAJOR — earlier draft also matched
        // ``maxTokens == reasoningChatFloor`` as an auto-scaled
        // landmark, but with today's constants ``autoScaleMaxTokens``
        // never writes ``2_048`` (the chat floor is below the
        // baseline, so ``max(4_096, 2_048) == 4_096``). A user who
        // explicitly drags the slider to ``2_048`` is expressing
        // intent — we must NOT silently bump them to ``4_096`` on
        // tools-on. The dedicated bookkeeping flag captures the
        // "auto vs explicit" distinction precisely; the landmark
        // check is the fresh-install fallback only.
        guard maxTokensIsAutoScaled || maxTokens == Self.maxTokensDefault else {
            return maxTokens
        }
        if toolsEnabled {
            return Self.clamped(max(maxTokens, effectiveToolsFloor), to: Self.maxTokensRange)
        }
        return Self.clamped(max(maxTokens, effectiveChatFloor), to: Self.maxTokensRange)
    }

    /// Resolve the current global sampling knobs into a wire-ready value.
    /// The minimal menu-bar app has a single ephemeral conversation with no
    /// per-session overrides, so this reads the Settings-backed properties
    /// directly; ``maxTokens`` still honours the reasoning-alias floor via
    /// ``effectiveMaxTokens(toolsEnabled:)``.
    func resolved(toolsEnabled: Bool) -> ResolvedSampling {
        ResolvedSampling(
            temperature: temperature,
            topP: topP,
            maxTokens: effectiveMaxTokens(toolsEnabled: toolsEnabled),
            repetitionPenalty: repetitionPenalty,
            enableThinking: enableThinking
        )
    }

    /// FU-3 — resolved chat floor: per-alias override (captured
    /// from ``ServerModelProfile.reasoningChatFloor`` on the last
    /// ``applyServerProfile`` call) when present, otherwise the
    /// global ``defaultReasoningChatFloor``. Clamped to
    /// ``maxTokensRange`` so a hostile server-side value (negative,
    /// 0, or way above ``maxTokensRange.upperBound``) can't escape
    /// the same bounds an honest user-dragged slider obeys.
    var effectiveChatFloor: Int {
        let raw = activeReasoningChatFloor ?? Self.defaultReasoningChatFloor
        return Self.clamped(raw, to: Self.maxTokensRange)
    }

    /// FU-3 — resolved tools floor; mirror of ``effectiveChatFloor``
    /// for the tools-on request path.
    var effectiveToolsFloor: Int {
        let raw = activeReasoningToolsFloor ?? Self.defaultReasoningToolsFloor
        return Self.clamped(raw, to: Self.maxTokensRange)
    }

    /// Cycle-3 fix — internal write that records the result as
    /// auto-scaled (rather than user-initiated). The didSet
    /// observer resets the flag on every assignment, so we set it
    /// back to ``true`` immediately after.
    private func autoScaleMaxTokens(to newValue: Int) {
        maxTokens = newValue
        maxTokensIsAutoScaled = true
    }

    /// Cycle-3 fix (codex r1 MAJOR) — drop the cached parser signal
    /// so a request that races against an alias swap doesn't lift
    /// to the previous alias's floor. Called by
    /// ``RapidApp`` at the TOP of the ``.task(id: server.servingAlias)``
    /// block BEFORE the async ``ServerProfileFetcher.fetch`` runs,
    /// so the window between "user clicked picker → server.ready
    /// fires for new alias" and "fetch returns and applies new
    /// profile" never carries stale parser state. Also called when
    /// the fetch fails (returns nil) so a transient 5xx / timeout
    /// during alias swap doesn't leave a stale signal that
    /// reactivates on the next chat send.
    func clearActiveReasoningParser() {
        activeReasoningParser = nil
        // FU-3 — clear the per-alias floor overrides at the same
        // time so an alias swap can't carry the previous alias's
        // floor into the new conversation. Mirrors the
        // ``resetToDefaults`` clean-slate semantics for the same
        // bookkeeping triplet.
        activeReasoningChatFloor = nil
        activeReasoningToolsFloor = nil
        // Issue #363 — same alias-swap clean-slate semantics: drop
        // the cached context window so the next chat send on a new
        // alias doesn't trim against the previous alias's window
        // until the fresh ``applyServerProfile`` lands.
        activeContextWindow = nil
        // FU-3 r1 P1 (codex) — if our auto-scale path was the last
        // thing that wrote ``maxTokens`` (i.e. the persisted slider
        // value is OUR footprint, not the user's intent), revert it
        // to the v0.4.12 baseline on alias swap. Otherwise a profile
        // that auto-scaled to e.g. 6,000 leaves 6,000 persisted in
        // ``UserDefaults`` even after the alias is dropped — the
        // next non-reasoning alias would silently send 6,000 max_tokens
        // even though the user never chose it. The pre-FU-3 codebase
        // never hit this because ``max(maxTokensDefault, chatFloor)
        // == maxTokensDefault`` (4,096 > 2,048) so the auto-scale
        // write was a no-op; with per-alias overrides now able to
        // exceed the baseline, the no-op assumption no longer holds.
        if maxTokensIsAutoScaled {
            maxTokens = Self.maxTokensDefault
            // ``maxTokens.didSet`` flips ``maxTokensIsAutoScaled``
            // back to ``false`` for us, which is exactly what we want
            // here — the next ``applyServerProfile`` decides afresh
            // whether to auto-scale based on the new alias's floor.
        }
    }

    /// One-button reset for the Settings panel — restores the
    /// v0.4.12 hard-coded profile across all four knobs at once.
    /// The visible "Reset to defaults" button calls this; tests
    /// pin the exact post-reset values so a future tweak to the
    /// defaults can't silently shift behaviour for users who hit
    /// the button thinking they're getting "v0.4.12 behaviour".
    func resetToDefaults() {
        temperature = Self.temperatureDefault
        topP = Self.topPDefault
        maxTokens = Self.maxTokensDefault
        repetitionPenalty = Self.repetitionPenaltyDefault
        enableThinking = Self.enableThinkingDefault
        // Cycle-3 fix — clear the reasoning bookkeeping so a user
        // who hits "Reset" gets a clean ``v0.4.12`` slate.
        // ``activeReasoningParser`` is in-memory only; the next
        // ``applyServerProfile`` call (on the next alias swap or
        // app launch) will re-populate it and re-run the
        // auto-scale.
        activeReasoningParser = nil
        // FU-3 — keep the per-alias floor bookkeeping in lockstep
        // with ``activeReasoningParser`` reset; a user who hits
        // "Reset to defaults" gets the global floors back, not the
        // last alias's override.
        activeReasoningChatFloor = nil
        activeReasoningToolsFloor = nil
        maxTokensIsAutoScaled = false
    }

    /// ``true`` when the current config matches the v0.4.12
    /// hard-coded defaults exactly. Drives the Settings panel's
    /// "Reset" button disabled state — pressing reset while
    /// already on defaults does nothing meaningful.
    var isAtDefaults: Bool {
        temperature == Self.temperatureDefault
            && topP == Self.topPDefault
            && maxTokens == Self.maxTokensDefault
            && repetitionPenalty == Self.repetitionPenaltyDefault
            && enableThinking == Self.enableThinkingDefault
    }

    private func persist<V>(_ keyPath: KeyPath<SamplingConfig, V>, value: V) {
        let key = "\(keyPrefix).\(Self.keyName(for: keyPath))"
        defaults.set(value, forKey: key)
    }

    static func clamped(_ value: Double, to range: ClosedRange<Double>, fallback: Double) -> Double {
        guard value.isFinite else { return fallback }
        return min(max(value, range.lowerBound), range.upperBound)
    }

    static func clamped(_ value: Int, to range: ClosedRange<Int>) -> Int {
        min(max(value, range.lowerBound), range.upperBound)
    }

    /// Compile-time map from KeyPath → UserDefaults suffix. We
    /// could use ``String(describing:)`` but the output is
    /// implementation-defined; an explicit switch is unambiguous
    /// across Swift versions and survives renames the moment a
    /// case stops covering all four knobs.
    private static func keyName(for keyPath: PartialKeyPath<SamplingConfig>) -> String {
        switch keyPath {
        case \SamplingConfig.temperature: return "temperature"
        case \SamplingConfig.topP: return "topP"
        case \SamplingConfig.maxTokens: return "maxTokens"
        case \SamplingConfig.repetitionPenalty: return "repetitionPenalty"
        case \SamplingConfig.enableThinking: return "enableThinking"
        default:
            // Reachable only if a new field is added and this
            // helper isn't updated. Crash loudly in DEBUG, fall
            // back to a stable string in release.
            assertionFailure("Unmapped SamplingConfig keyPath")
            return "unknown"
        }
    }
}
