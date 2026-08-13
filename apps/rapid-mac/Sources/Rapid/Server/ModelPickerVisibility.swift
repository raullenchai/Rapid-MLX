import Foundation

/// cycle-7: filter the model picker's "All models" list so sub-1B
/// aliases (qwen3-0.6b-4bit, qwen3-0.6b-8bit, …) do not surface to
/// first-time users. The rapid-mlx catalog ships ~92 aliases including
/// 600M tiny models intended for unit tests and the bundled cold-boot
/// path — they hallucinate within 1-2 turns of chat and give the user
/// a bad first impression of every other model in the dropdown.
///
/// The threshold is ``minParamsBillions = 1.0`` (inclusive — a 1B
/// model is shown). Parse failures default to "show" so a custom
/// alias the user typed in, or an upstream alias whose name doesn't
/// follow the ``…-Nb…`` convention, isn't accidentally hidden.
///
/// The currently-selected alias is always shown regardless of size —
/// otherwise a user who picked a sub-1B alias (e.g. via the bundled
/// first-launch path, ``BundledModel.firstLaunchAlias``) couldn't
/// see it in the dropdown to identify what's currently in use.
///
/// Power users can flip ``Settings → Models → Show small (<1B) models``
/// to override the filter. The toggle persists via
/// ``UserDefaults`` key ``rapid.picker.show_all_models.v1`` and
/// defaults OFF.
///
/// Tests live in ``ModelPickerVisibilityTests``.
enum ModelPickerVisibility {
    /// Inclusive lower bound on parameter count, in billions. A 1.0B
    /// alias (``gemma3-1b-qat-4bit``, ``llama3-1b-4bit``) is shown;
    /// a 0.6B alias (``qwen3-0.6b-4bit``) is hidden.
    static let minParamsBillions: Double = 1.0

    /// UserDefaults key for the ``Show small (<1B) models`` toggle.
    /// Mirrors the ``rapid.*.v1`` keyspace convention used by
    /// ``AppearanceConfig`` and the sidebar collapsed-section flags.
    static let showAllStorageKey: String = "rapid.picker.show_all_models.v1"

    /// Aliases that are KNOWN-BROKEN for plain text chat on the bundled
    /// engine and must never be offered as a selectable chat model.
    ///
    /// These are small **multimodal-only** checkpoints whose only small
    /// SKU ships as a VLM. Serving them for text through the bundled
    /// rapid-mlx routes to the mlx-vlm lane, where they either hang or
    /// return incoherent output — the exact "switched the picker to X
    /// and it spun forever with zero tokens" footgun. Evidence (issue
    /// #1367, reproduced on a real macos-14 M1 runner, rapid-mlx 0.11.4
    /// / mlx-vlm 0.6.3, and pinned in ``ci.yml``'s L1-smoke comment):
    ///
    ///   * ``gemma-4-e2b-*`` (the Gemma nano "effective-2B" MatFormer):
    ///     **0/6** golden — total incoherence — on BOTH the mlx-vlm lane
    ///     AND the ``--no-mllm`` text lane. It is not a quant artifact
    ///     (0/6 is arch-level), so every e2b SKU shares the break:
    ///     4/6/8-bit and the ``-assistant`` bf16 tune.
    ///   * ``ministral-3b-4bit`` (Ministral-3-3B-Instruct-2512): the
    ///     first chat completion **hangs** under the mlx-vlm lane
    ///     (request times out at 90 s, server goes unreachable). It is
    ///     usable (5/6, flaky) only via ``--no-mllm``, which the desktop
    ///     does not wire per-alias — so the picker path always hits the
    ///     hanging lane.
    ///
    /// Scope is deliberately EVIDENCE-BOUNDED to what #1367 measured.
    /// The ``gemma-4-e4b-*`` / ``gemma-3n-*`` nano SKUs are NOT listed —
    /// they were not tested, and hiding a model that might work is its
    /// own harm (mirrors ``shouldShow``'s parse-failure "default to
    /// shown" bias). Extend this set only when engine testing confirms a
    /// new alias is broken for text chat.
    static let knownBrokenForTextChat: Set<String> = [
        "gemma-4-e2b-4bit",
        "gemma-4-e2b-6bit",
        "gemma-4-e2b-8bit",
        "gemma-4-e2b-assistant",
        "ministral-3b-4bit",
    ]

    /// True when ``alias`` is on the known-broken-for-text-chat
    /// denylist above and must be hidden from the picker regardless of
    /// size or the ``Show small models`` toggle.
    static func isKnownBroken(_ alias: String) -> Bool {
        knownBrokenForTextChat.contains(alias)
    }

    /// Returns true when ``alias`` should be visible in the picker's
    /// "All models" alphabetical list.
    ///
    /// Decision matrix:
    ///   * ``includeAll == true`` → always true (user opted out of
    ///     the filter via Settings).
    ///   * ``alias == selectedAlias`` → always true (don't hide the
    ///     row the user is currently looking at; otherwise they can't
    ///     identify what's actually serving).
    ///   * ``parseSmallestParamsBillions`` returns ``nil`` →
    ///     true. Parse-failure aliases (custom HF repos, future
    ///     naming conventions) default to shown so we never
    ///     accidentally hide a legitimate alias due to a parse miss.
    ///   * Parsed params ``>= minParamsBillions`` → true.
    ///   * Parsed params ``< minParamsBillions`` → false (hidden).
    static func shouldShow(
        alias: String,
        selectedAlias: String,
        includeAll: Bool
    ) -> Bool {
        // Known-broken denylist is ABSOLUTE — it wins over both
        // ``includeAll`` and the currently-selected exemption. Those
        // two escape hatches are about SIZE ("power user wants the
        // tinies" / "don't hide what I picked"); brokenness is a
        // different axis. A model that hangs or garbles on every chat
        // send is never a legitimate option, so ``Show small models``
        // must not reveal it and a stale persisted selection must not
        // keep it on the menu where it can be re-picked. See issue
        // #1367 and ``knownBrokenForTextChat``.
        if isKnownBroken(alias) { return false }
        if includeAll { return true }
        if !selectedAlias.isEmpty && alias == selectedAlias { return true }
        guard let params = parseSmallestParamsBillions(alias) else {
            // Parse failure → default to shown. Hiding an alias the
            // parser can't read would be a worse failure mode than
            // surfacing one extra tiny model.
            return true
        }
        return params >= minParamsBillions
    }

    /// Extract a parameter count from ``alias`` for the
    /// hide-the-tinies filter.
    ///
    /// Two-pass parse so naming variants the upstream
    /// ``ModelSizing.parseParamsBillions`` regex doesn't cover still
    /// hide cleanly. Concretely (codex r1 MINOR):
    ///
    ///   1. ``\d+(\.\d+)?[bB]`` — the ModelSizing regex catches
    ///      ``qwen3-0.6b``, ``gemma3-1b``, ``bonsai-1.7b``,
    ///      ``qwen3.5-122b``.
    ///   2. ``\d+(\.\d+)?[mM]`` — a million-scale fallback that
    ///      ModelSizing doesn't carry. Catches a hypothetical
    ///      ``qwen3-600m-4bit`` / ``smollm-135m`` so a future
    ///      upstream rename to the ``-Nm`` convention doesn't
    ///      silently leak a sub-1B alias back into the picker. The
    ///      value is normalised to billions before comparison
    ///      (600m → 0.6B).
    ///
    /// Both regexes are NUMBER-then-SUFFIX. The ``minimax-m2.7`` /
    /// ``minimax-m2.5`` aliases (codex r2 MINOR) use a
    /// SUFFIX-then-number convention where ``m`` is the family
    /// version letter, not a million-scale identifier — those
    /// correctly fall through to ``nil`` and the visibility helper
    /// defaults to "shown", which matches the user's expectation
    /// (MiniMax M2.x is a 235B MoE family).
    ///
    /// We pick the SMALLEST positive match rather than the largest,
    /// the inverse of ``ModelSizing.parseParamsBillions``. That
    /// parser sizes the host's RAM around the largest weight tensor
    /// (a MoE alias like ``qwen3-coder-next-80b-a3b`` needs 80 GB of
    /// RAM); this filter cares about whether the alias has a
    /// SUB-billion identifier in it at all. Using the smallest match
    /// means an A3B-style alias (80b total, 3B active) sees 3B → ≥
    /// 1B → shown; a hypothetical ``mix-0.6b-2b-moe`` would see
    /// 0.6B → < 1B → hidden, which matches the "if there's a tiny
    /// part of this thing, treat it as tiny for picker purposes"
    /// rule (any sub-1B variant on the menu is the same first-
    /// impression risk).
    static func parseSmallestParamsBillions(_ alias: String) -> Double? {
        let billionMatches = matchedNumbers(in: alias, regex: billionsRegex)
        let millionMatches = matchedNumbers(in: alias, regex: millionsRegex).map { $0 / 1000.0 }
        let candidates = billionMatches + millionMatches
        guard !candidates.isEmpty else { return nil }
        // Smallest > 0; the helper already filters out 0.
        return candidates.min()
    }

    /// Run a `\d+(\.\d+)?\s*<suffix>\b` regex over ``alias`` and
    /// return every captured numeric value. Empty array on regex
    /// compile failure or no match.
    /// Compiled once per suffix. Both are reached once per alias per
    /// ``ModelPickerBar`` body pass, and that bar rebuilds on every streamed
    /// delta — compiling the pattern per call put ICU's ``uregex_open`` on
    /// the hot path of every chat token.
    private static let billionsRegex = try? NSRegularExpression(
        pattern: #"(\d+(?:\.\d+)?)\s*[bB]\b"#
    )
    private static let millionsRegex = try? NSRegularExpression(
        pattern: #"(\d+(?:\.\d+)?)\s*[mM]\b"#
    )

    private static func matchedNumbers(in alias: String, regex: NSRegularExpression?) -> [Double] {
        guard let regex else { return [] }
        let nsAlias = alias as NSString
        let matches = regex.matches(in: alias, range: NSRange(location: 0, length: nsAlias.length))
        var values: [Double] = []
        for m in matches {
            guard m.numberOfRanges >= 2 else { continue }
            let captured = nsAlias.substring(with: m.range(at: 1))
            guard let v = Double(captured), v > 0, v <= 10_000 else { continue }
            values.append(v)
        }
        return values
    }

    /// Filter a catalog. Convenience over ``shouldShow`` for the
    /// picker's ``allAliasesSection``.
    static func filter(
        _ entries: [ModelEntry],
        selectedAlias: String,
        includeAll: Bool
    ) -> [ModelEntry] {
        entries.filter { entry in
            shouldShow(alias: entry.alias, selectedAlias: selectedAlias, includeAll: includeAll)
        }
    }

    /// Count how many entries WOULD be hidden by the filter, given
    // MARK: - cycle-10: quality buckets / picker sticker (F9-004)

    /// Quality bucket for an alias, derived from its parameter count.
    ///
    /// cycle-9 fuzz-correct surfaced that ``llama3-1b-4bit`` failed
    /// 8/10 basic arithmetic prompts and contradicted itself in
    /// multi-turn chat (turn 1: "octopuses have 3 hearts" → turn 2
    /// in the same session: "actually 8 hearts"). Sub-3B chat-tuned
    /// quants (1B llama, 1.5B vibethinker, 1.7B bonsai, 1B gemma3
    /// QAT) are format-correct but semantic garbage on anything
    /// resembling a real conversation. The picker filter from
    /// cycle-7 only hides sub-1B aliases; the rest of the sub-3B
    /// band still surfaces in "All models" unmarked and a casual
    /// user can hit turn-2 contradictions in their very first
    /// session.
    ///
    /// The bucket lifts that signal out so the picker can decorate
    /// sub-3B rows with a "small — try larger for chat" sticker
    /// (per LM Studio / Ollama convention for small-model warnings).
    /// #348: the suffix is bucket-distinct (``.tiny`` -> "· tiny",
    /// ``.small`` -> "· small") so the data-model split between
    /// sub-1B and 1-3B aliases shows up in the picker instead of
    /// both rendering as "tiny". Rows at 3B and above stay
    /// un-stickered: 3B is the boundary where llama-family quality
    /// becomes desktop-viable per cycle-11 (``llama3-3b-4bit``
    /// passed 5/5 weather tool-call schemas at temp=0 vs
    /// ``llama3-1b-4bit``'s 3/5 schema-leak rate, held its
    /// multi-turn arithmetic stable, and decoded at 154 tok/s B=1
    /// on M3 Ultra). 3.8B ``phi-3.5-mini-4bit`` and 4B
    /// ``qwen3.5-4b-4bit`` are real-world-useful for testing and
    /// single-shot Q&A, and the project's own test default is
    /// qwen3.5-4b — stickering them would dilute the warning.
    ///
    /// **cycle-11 boundary tightening (F-10-PRESET):** the original
    /// ``.small`` bucket was ``>= 1B && <= 3B`` inclusive, which
    /// pulled ``llama3-3b-4bit`` into the discouraging sub-3B
    /// sticker band. Empirical cycle-10 testing showed the 3B
    /// llama-family chat quant clears every failure mode cycle-9
    /// flagged on 1B (clean tool-call args, stable multi-turn
    /// facts). Tightening the upper bound to strict ``< 3B`` lets
    /// 3B aliases surface in the picker without the discouraging
    /// small/tiny sticker — they read as the smallest viable
    /// llama-family chat choice rather than another can't-trust-it
    /// tiny.
    enum QualityBucket: Equatable, Sendable {
        /// < 1B. Already hidden by ``shouldShow`` unless ``includeAll``
        /// is on; if surfaced (e.g. the user typed in a sub-1B alias),
        /// the sticker still fires so they know what they're getting.
        case tiny
        /// ``>= 1B`` and ``< 3B``. The cycle-10 sticker bucket,
        /// tightened in cycle-11 to exclude exactly-3B aliases.
        /// ``llama3-1b-4bit`` (1.0B), ``gemma3-1b-qat-4bit`` (1.0B),
        /// ``vibethinker-1.5b-4bit`` (1.5B), and
        /// ``bonsai-1.7b-unpacked`` (1.7B) all land here.
        /// ``llama3-3b-4bit`` (3.0B) does NOT — cycle-10 empirical
        /// data showed 3B is the smallest llama-family chat-viable
        /// size (5/5 clean tool calls, stable multi-turn).
        /// ``phi-3.5-mini-4bit`` (3.8B, per the HF card's true param
        /// count) also does NOT — it parses to no size from the
        /// alias and falls through to ``midOrLarger`` via the
        /// nil-default branch. That's intentional: stickering an
        /// alias whose size the parser can't read would surprise
        /// users who type custom HF repos.
        case small
        /// ``>= 3B``, or parse-failure (default to no sticker so
        /// custom HF aliases the user types in aren't decorated with
        /// a warning the parser can't justify). cycle-11: 3B
        /// llama-family is the smallest desktop-viable chat quant
        /// per cycle-10 measurement, so it lands here.
        case midOrLarger
    }

    /// Strict upper bound of the ``.small`` bucket, in billions. An
    /// alias with parsed params ``< 3.0`` gets the sticker; a 3.0B
    /// alias does NOT. The 3B-and-above band stays unmarked — see
    /// ``QualityBucket`` for the cycle-11 rationale (3B llama-family
    /// is desktop-viable per cycle-10 empirical data).
    static let smallBucketUpperBoundBillions: Double = 3.0

    /// Map ``alias`` to a ``QualityBucket``.
    ///
    /// Quality stickering reads ``ModelSizing.parseParamsBillions``
    /// (LARGEST ``\d+b`` token), not ``parseSmallestParamsBillions``
    /// (smallest). The cycle-7 visibility filter uses the SMALLEST
    /// token because "any sub-1B identifier in the name = first-
    /// impression risk." Quality stickering wants the OPPOSITE
    /// semantics: if a model has any size token ``>= 3B`` (cycle-11
    /// strict upper bound — 3B llama-family is desktop-viable per
    /// cycle-10 measurement), it has the total capacity to NOT
    /// contradict itself in chat, so the sticker should NOT fire.
    /// The two helpers diverge intentionally at the A-NB-MOE corner
    /// case:
    ///
    ///   * ``qwen3-coder-next-80b-a3b`` → SMALLEST sees 3B → would
    ///     wrongly sticker an 80B MoE as "tiny"; LARGEST sees 80B →
    ///     ``.midOrLarger``, no sticker (correct — an 80B MoE is the
    ///     opposite of tiny in chat quality terms; the active-3B
    ///     band only describes inference compute).
    ///   * ``qwen3.6-35b-a3b-mxfp4`` → same story; LARGEST sees 35B →
    ///     ``.midOrLarger``, sticker stays off (correct).
    ///   * Pure-dense aliases (``llama3-1b-4bit``, ``qwen3.5-4b-4bit``,
    ///     ``qwen3.6-35b-8bit``) have ONE size token, so LARGEST and
    ///     SMALLEST agree — the two-parser split affects only
    ///     mixed-band aliases.
    ///
    /// Parse failure → ``.midOrLarger`` so the sticker is never
    /// applied to a custom alias whose size we can't read.
    static func qualityBucket(for alias: String) -> QualityBucket {
        guard let params = ModelSizing.parseParamsBillions(alias) else {
            return .midOrLarger
        }
        if params < minParamsBillions {
            return .tiny
        }
        // cycle-11 F-10-PRESET: strict upper bound (``<``) so a
        // 3.0B alias lands in ``.midOrLarger`` (no sticker), not
        // ``.small``. cycle-10 measurement: ``llama3-3b-4bit`` is
        // the smallest llama-family chat-viable quant (5/5 clean
        // tool calls, stable multi-turn, 154 tok/s decode on M3
        // Ultra). The cycle-10 inclusive bound (``<=``) pulled it
        // into the discouraging "tiny" sticker band; tightening to
        // strict lets the row read as a viable smallest pick.
        if params < smallBucketUpperBoundBillions {
            return .small
        }
        return .midOrLarger
    }

    /// Compact suffix appended to the alias label in the picker
    /// dropdown for ``.small`` and ``.tiny`` rows. ``nil`` for
    /// ``.midOrLarger`` so 3B+ rows stay clean (cycle-11 strict
    /// upper bound — ``llama3-3b-4bit`` lands in ``.midOrLarger``).
    ///
    /// Copy follows the LM Studio / Ollama convention of a one-word
    /// inline tag ("tiny", "small") rather than inventing a new vocab.
    /// NSMenuItem only honours the first ``Text`` inside a SwiftUI
    /// ``Menu`` button, so the sticker rides in the alias title string
    /// itself — see ``ModelPickerBar.aliasButtonTitle``. A leading
    /// "·" separator keeps the alias visually anchored on the left.
    ///
    /// **#348 — bucket-distinct suffix.** Before this fix, ``.small``
    /// returned ``"· tiny"`` (same as ``.tiny``), making the
    /// data-model split between sub-1B and 1-3B aliases invisible in
    /// the UI ("visual theatre"). The two buckets are derived from
    /// different empirical failure modes (sub-1B: silent
    /// first-impression risk; 1-3B: cycle-9 confirmed tool-call
    /// schema-leak + multi-turn fact-flipping) and the comments above
    /// have always treated them as distinct user-facing signals; the
    /// suffix now matches that intent. The tooltip remains a unified
    /// "smaller than 3B" warning since it applies to both buckets.
    static func qualityStickerSuffix(for bucket: QualityBucket) -> String? {
        switch bucket {
        case .tiny:
            return "· tiny"
        case .small:
            return "· small"
        case .midOrLarger:
            return nil
        }
    }

    /// Long-form tooltip surfaced via ``.help()`` on hover for the
    /// ``.tiny`` and ``.small`` rows. Returns ``nil`` for
    /// ``.midOrLarger`` so 3B+ rows fall back to the cache-state
    /// tooltip ("Already downloaded" / "Will download on Start") —
    /// cycle-11 strict upper bound puts 3B aliases in the clean band.
    ///
    /// Copy is honest about what the user will see (turn-2
    /// contradictions, weak math) and points at a concrete fix
    /// (qwen3.5-4b or larger). Matches the ChatGPT-Desktop /
    /// LM-Studio precedent of "facts ok, math/coherence weak"
    /// language for sub-3B models.
    static func qualityStickerTooltip(for bucket: QualityBucket) -> String? {
        switch bucket {
        case .tiny, .small:
            // "Smaller than 3B" matches the cycle-11 strict upper
            // bound: a 3.0B alias does NOT get the sticker and is
            // NOT covered by this warning (cycle-10 empirically
            // verified llama3-3b-4bit is desktop-viable — 5/5 clean
            // tool calls, stable multi-turn). Earlier wording
            // "3B and smaller" matched the cycle-10 inclusive bound;
            // the boundary moved in cycle-11 (F-10-PRESET) so the
            // tooltip copy moved with it. Pinned to the same string
            // in ``ModelPickerVisibilityTests`` so a copy churn that
            // re-introduces the old inclusive wording trips the gate.
            return "Models smaller than 3B may contradict themselves in multi-turn chat. Good for testing or single-shot Q&A. Try qwen3.5-4b or larger for serious use."
        case .midOrLarger:
            return nil
        }
    }

    /// Combined hover-tooltip text for a single alias row: the
    /// quality-sticker copy (when present) plus the cache-state cue
    /// ("Already downloaded" / "Will download on Start"). Composed
    /// here so ``ModelPickerBar.aliasRowHelpText`` and the test suite
    /// agree on the multi-line layout. ``cacheHint`` is appended only
    /// when non-empty so callers can pass ``""`` if no cache info is
    /// known yet.
    static func qualityRowHelpText(for bucket: QualityBucket, cacheHint: String) -> String {
        let qualityLine = qualityStickerTooltip(for: bucket)
        switch (qualityLine, cacheHint.isEmpty) {
        case (nil, true):
            return ""
        case (nil, false):
            return cacheHint
        case (.some(let q), true):
            return q
        case (.some(let q), false):
            return "\(q)\n\(cacheHint)"
        }
    }
}
