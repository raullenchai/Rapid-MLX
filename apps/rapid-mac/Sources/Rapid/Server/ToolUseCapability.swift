import Foundation

/// Desktop-side truth about whether a rapid-mlx alias can actually
/// emit ``tool_calls`` for tool-eligible prompts.
///
/// ## Why this lives in the desktop (not in ``aliases.json``)
///
/// The sidecar's ``/v1/models`` reports a ``tool_call_parser`` per
/// alias (``"hermes"``, ``"llama"``, ``"deepseek_r1"``, …). That field
/// describes which parser is wired up to scrape ``<tool_call>`` tokens
/// out of the model's stream — NOT whether the underlying weights
/// actually produce those tokens. Cycle-11 fuzz F-11-5 caught the
/// gap: ``phi-4-mini-reasoning-4bit`` advertises ``tool_call_parser:
/// "hermes"`` because of profile defaults, but the model itself never
/// emits ``<tool_call>`` for a tool-eligible prompt. It hallucinates
/// the answer ("Tokyo is 16°C, partly cloudy" with ``tool_calls:
/// null``) and the user sees a confidently wrong reply.
///
/// Cycle-4 fuzz F-1 caught the same shape on ``hermes3-8b-4bit`` and
/// cycle-9 fuzz F9-001 caught a parser-level variant on
/// ``llama3-1b-4bit`` (the model emits raw JSON-Schema into
/// ``function.arguments`` instead of a populated args object). Cycle-2
/// fuzz-correctness caught vibethinker emitting a ``<JSON>...</JSON>``
/// wrapper instead of hermes ``<tool_call>...</tool_call>`` — the
/// alias's profile-default hermes parser can't extract any args.
/// Server-side fixes for either gap require rapid-mlx changes that
/// haven't landed; meanwhile the desktop owns its own truth about
/// which aliases are safe to surface a tools UI for.
///
/// ## Three buckets, intentional defaults
///
/// * ``.known`` — empirical bench confirms the alias's family AND
///   size emit well-formed ``tool_calls`` on representative tools-
///   eligible prompts. The classifier looks up the alias's
///   ``(family, sizeBillions)`` tuple against a verified table
///   below so size siblings of every known-good family are covered
///   without enumerating each quant individually.
/// * ``.broken`` — loop fuzz proved silent-degradation OR
///   schema-leak on this alias. Specific to the alias name (no
///   prefix match for the family — we never want to over-broaden
///   ``.broken`` and accidentally lock out a working sibling).
/// * ``.unknown`` — DEFAULT. We have no signal either way; do not
///   regress on aliases that haven't been benched. With #342
///   followup, ``.unknown`` is reserved for legitimately exotic
///   families (brand-new releases the loop hasn't covered),
///   parser=None aliases, non-chat modalities, or sub-min-size
///   variants of otherwise-verified families.
///
/// The picker / Tools UI consumes the bucket. ``.broken`` and
/// ``.unknown`` both collapse the empty-state capability chips
/// (no over-promise) AND fire a picker badge so the user sees the
/// limitation BEFORE clicking a chip. ``.known`` surfaces both.
/// FU-9 splits the badge copy by bucket so the user can tell
/// "empirically broken" apart from "we haven't benched it":
/// ``.broken`` reads ``· no tools`` (unchanged); ``.unknown``
/// reads ``· tools unverified`` (NEW; softer). See
/// ``ToolUseCapability/badgeLabel(for:)``.
///
/// ## When to flip an alias / family
///
/// New empirical evidence (fuzz cycle, manual bench, user report
/// with reproducer) can:
///
///   * Promote a family to verified via ``knownFamilies`` (add a
///     row with the appropriate ``minSizeBillions`` floor and a
///     citation to the cycle / PR that produced the evidence).
///   * Promote ``.unknown`` → ``.broken`` via ``brokenPrefixes``
///     when silent-degradation OR schema-leak is observed.
///   * Demote a family by raising its ``minSizeBillions`` floor or
///     removing the row entirely. The
///     ``ToolUseCapabilityCatalogCoverageTests`` distribution
///     guards catch the drop.
///
/// Do NOT promote ``.broken`` → ``.known`` without a re-bench;
/// the existing entries are pinned because we have evidence the
/// model fails, and "the new quant might be different" is not
/// evidence the model now works.
enum ToolUseConfidence: Sendable, Equatable {
    /// Empirically confirmed: this alias DOES emit ``tool_calls``
    /// for tool-eligible prompts. Surface the Tools UI.
    case known
    /// No signal. Default for new aliases. Collapse the empty-state
    /// chip row + badge the picker row with "tools unverified" so
    /// the user isn't promised tool-call support we can't back up
    /// AND isn't told the model is broken when we just haven't
    /// benched it (FU-9 softer copy split).
    case unknown
    /// Empirically confirmed silent-degradation OR schema-leak.
    /// Treated identically to ``.unknown`` at the chip-row layer
    /// (no tool-promise we can't back), but the picker badge keeps
    /// the strong "no tools" copy so the user can tell empirically-
    /// broken aliases apart from merely-unbenched ones (FU-9).
    case broken
}

/// Static capability map for ``ToolUseConfidence``.
///
/// Match rules (in priority order):
///
///   1. Alias matches a ``brokenPrefixes`` entry → ``.broken``.
///   2. Alias's parsed ``(family, sizeBillions)`` tuple matches a
///      verified row in ``knownFamilies`` → ``.known``.
///   3. Otherwise → ``.unknown``.
///
/// Prefix-match for ``brokenPrefixes`` is case-insensitive on the
/// ASCII alias namespace; rapid-mlx alias keys are always lowercased
/// ASCII so the ``localizedLowercase`` is a no-op on real aliases
/// but keeps the match robust against an upstream casing change.
///
/// Family parsing reuses ``ModelSizing.parseParamsBillions`` so the
/// desktop doesn't carry a third alias-name parser — the sizing
/// estimator already covers ``qwen3.6-27b`` / ``llama-3.1-8b`` /
/// ``gemma-4-12b-qat`` / ``qwen3-coder-next-80b-a3b`` shapes.
enum ToolUseCapability {

    /// One family-level rule: "aliases whose name starts with
    /// ``prefix`` (case-insensitive) AND parse to a size
    /// >= ``minSizeBillions`` are ``.known``." The ``note`` field is
    /// for human readers (test failures cite it back); not consumed
    /// by the matcher.
    ///
    /// ``missingSizeAllowList`` controls what happens when the alias
    /// matches the family prefix but ``ModelSizing.parseParamsBillions``
    /// returns ``nil`` (no ``\d+b`` token in the name — happens for
    /// quant-only suffixes like ``qwen3-coder-4bit`` or ``glm4.5-air-
    /// 4bit``). Default empty set means "any missing-size alias under
    /// this family falls to ``.unknown``". Codex r2 MAJOR tightened
    /// this from a prefix-wide ``allowMissingSize: Bool`` to an
    /// exact-alias allow-list so a user-typed ``qwen3-coder-
    /// experimental`` that shares a verified family prefix but is
    /// NOT in the catalog falls through to ``.unknown`` — only the
    /// catalog-shipped size-less aliases (the 4 known cases below)
    /// promote.
    ///
    /// Verified families come from the fuzz loop + eval suite + the
    /// cross-model bench walks (cycle-6 / cycle-7 / cycle-8 / cycle-10
    /// / cycle-11 / cycle-13 closure verifications). Each row should
    /// cite the cycle / PR that produced the evidence so a future
    /// reviewer can verify the floor was set deliberately.
    struct KnownFamily: Sendable, Equatable {
        let prefix: String
        let minSizeBillions: Double
        /// Catalog-shipped aliases without a ``\d+b`` size token
        /// that promote to ``.known`` via family verdict alone.
        /// Codex r2 MAJOR: each entry is the FULL alias name (not
        /// a suffix); only an exact case-insensitive match against
        /// this list promotes the missing-size path. A future
        /// catalog-bump that ships a new size-less alias must add
        /// the literal name here to opt in — anything else stays
        /// ``.unknown``.
        let missingSizeAllowList: Set<String>
        let note: String

        /// Default initialiser preserves the conservative empty
        /// allow-list so adding a new family row without thinking
        /// about the missing-size case doesn't silently open the
        /// hole (codex r2 MAJOR).
        init(prefix: String, minSizeBillions: Double, missingSizeAllowList: Set<String> = [], note: String) {
            self.prefix = prefix
            self.minSizeBillions = minSizeBillions
            // Pre-lowercase the allow-list so the case-insensitive
            // match in ``confidence(for:)`` is a constant-time hash
            // lookup against the lowercased needle.
            self.missingSizeAllowList = Set(missingSizeAllowList.map { $0.localizedLowercase })
            self.note = note
        }
    }

    /// Aliases proven to silently fail at tool-calling. Each entry
    /// cites the fuzz finding that put it on the list so a future
    /// re-bench can verify or refute it.
    ///
    /// Match is prefix-based to catch quant siblings of the same
    /// weights — e.g. a hypothetical ``hermes3-8b-8bit`` would still
    /// degrade silently (same training) so the prefix ``hermes3-8b``
    /// covers both ``-4bit`` and any future quant.
    ///
    /// IMPORTANT: only add an alias to this list when there is
    /// empirical evidence of silent-degradation OR schema-leak.
    /// "Reasoning models probably can't tool-call" is not enough —
    /// deepseek-r1 / qwen-thinker variants exist that DO emit
    /// tool_calls, and we don't want to lock out an untested sibling.
    static let brokenPrefixes: [String] = [
        // F-11-5 (2026-06-20): phi-4-mini-reasoning-4bit advertises
        // tool_call_parser="hermes" but emits no <tool_call> tokens;
        // model hallucinates the answer with tool_calls: null. Family
        // prefix covers any future ``phi-4-mini-reasoning-Nbit`` quant.
        "phi-4-mini-reasoning",
        // cycle-4 F-1 (2026-06-19): hermes3-8b-4bit silent-degradation
        // — auto tool_choice never emits <tool_call> across 4 prompts;
        // content returns hallucinated prose with fabricated values.
        // rapid-mlx registry already flags model tool_calling="avoid"
        // but server surface is silent. Desktop must guard.
        "hermes3-8b",
        // cycle-9 F9-001 (2026-06-19): llama3-1b-4bit llama parser
        // silently passes raw JSON-Schema wrapper into function.
        // arguments verbatim (``{"type":"object","properties":...}``
        // instead of ``{"location":"Paris"}``); 3/5 weather prompts
        // at temp=0. Independent of PR #322's sub-1B picker filter —
        // a user who keeps the alias selected (or types it manually
        // into the picker) still hits this.
        "llama3-1b",
        // cycle-2 fuzz-correctness (2026-06-19): vibethinker-3b-8bit
        // (and -1.5b-4bit sibling) emit ``<JSON>...</JSON>`` wrapper
        // instead of hermes ``<tool_call>...</tool_call>`` on simple
        // single-tool prompts. The alias claims ``tool_call_parser=
        // hermes`` but the model distribution doesn't match → the
        // parser captures nothing and the raw XML leaks into content.
        // Also a 4-duplicate-tool-call pathology when the model
        // echoes ``<tool_call>`` inside its reasoning trace. Family
        // prefix ``vibethinker-`` covers both shipped quants.
        "vibethinker-",
        // 2026-07-09 recommended-model tool-usability sweep (real
        // desktop wire: toolGuidancePreamble + web_search schema +
        // tool_choice auto, N=6 per alias). phi-4-mini-4bit
        // (non-reasoning; the -reasoning sibling was already broken
        // above) does NOT leak schema — it flatly REFUSES every
        // tool-eligible prompt ("I'm sorry, but I can't assist with
        // that", 6/6) while chatting fine without tools. Stripping
        // tools restores a coherent chat model. Prefix ``phi-4-mini``
        // also subsumes ``phi-4-mini-reasoning`` above (kept there for
        // its own citation).
        "phi-4-mini",
        // 2026-07-09 sweep: deepseek-coder-v2-lite-16b-4bit ships
        // ``tool_call_parser=None`` AND the model invents ad-hoc tool
        // names ("latest_ai_model_releases") wrapped in a DeepSeek-V3
        // ``<｜tool▁calls▁begin｜>`` envelope that no wired parser can
        // recover → 6/6 raw envelope+JSON dumped as content. Chat is
        // fine; strip tools. (Was the 25–36 GB Coding recommendation;
        // replaced by qwen3-coder-30b-4bit.)
        "deepseek-coder-v2-lite",
        // 2026-07-09 sweep (F3): deepseek-r1-8b-4bit
        // (DeepSeek-R1-0528-Qwen3-8B distill) invents a DIFFERENT JSON
        // schema each run ({"action":…}/{"tool":…}/{"results":…}) with
        // wrong tool names → 4/8 raw-JSON leak; no parser can recover
        // arbitrary invented schemas. Chat is coherent without tools.
        // Prefix pinned to the 8B distill ONLY — larger / other R1
        // variants may tool-call and must not be locked out (see the
        // IMPORTANT note above).
        "deepseek-r1-8b",
        // NOTE: the Mistral / Devstral family was here (2026-07-09
        // sweep: PARSER MISCONFIG, not model incapacity — the model
        // emits a textbook ``[TOOL_CALLS]name[ARGS]{json}`` call but
        // the OLD bundled engine routed these aliases through the
        // hermes parser so the call leaked verbatim, 6/6). The engine
        // parser fix (rapid-mlx #1071/#1077 — route the Mistral family
        // to the ``mistral`` parser) is NOW BUNDLED (submodule bump to
        // 7b6a787). Re-verified on the bundled engine: devstral-v2-24b
        // emits a clean ``tool_calls`` (get_weather) with the mistral
        // parser. So ``mistral-`` / ``devstral`` are promoted back to
        // ``knownFamilies`` below (the disposition their comments
        // always pointed to). The broken ``ministral-3b-4bit`` alias was
        // removed from the engine catalog in #1367.
    ]

    /// Empirically verified family rows. Match rule: alias starts
    /// with the family ``prefix`` (case-insensitive) AND parses to a
    /// size ``>= minSizeBillions`` via ``ModelSizing.parseParamsBillions``.
    ///
    /// Family-aware match closes the #342 over-classification hole:
    /// the pre-fix string-prefix list missed ``qwen3.5-9b-4bit``
    /// (didn't start with ``qwen3.5-4b``/``-7b``/``-14b``/...),
    /// every Gemma 4 variant under 26B, every Qwen 3 hermes-parser
    /// size sibling, and the ``mxfp4-q8`` quant suffix of gpt-oss.
    /// Adding one row per family closes all of those at once.
    ///
    /// Order matters only for human-readability (matched first hit
    /// wins for the diagnostic, but each row should be disjoint at
    /// the prefix level so order is irrelevant in practice).
    static let knownFamilies: [KnownFamily] = [
        // MARK: Qwen family — the backbone of the bench loop's tool
        // evidence. Dense + hybrid+MoE variants benched extensively.

        // qwen3.5 — dense + hybrid variants; verified across the eval
        // suite at every shipped size from 4B to 122B. Hermes parser.
        // 3B floor is conservative (smallest shipped is 4B; if a 2B
        // ever ships it'd need its own bench).
        KnownFamily(prefix: "qwen3.5-", minSizeBillions: 3.0, note: "eval-suite + fuzz loop verified 4B/9B/14B/27B/35B/122B"),
        // qwen3.6 — hybrid + MoE family, qwen3_coder_xml parser.
        // cycle-13 F-13-2 confirms /v1/models exposes the parser
        // correctly; PR #318 auto-scale gate verified.
        KnownFamily(prefix: "qwen3.6-", minSizeBillions: 3.0, note: "hybrid+MoE, qwen3_coder_xml parser; cycle-13 F-13-2 closure"),
        // qwen3.8 — hybrid family, hermes parser. Release eval on
        // rapid-mlx/Qwen3.8-27B-mixed-3.5bpw-MLX (our own mixed-precision
        // build) scored 25/30 tool-calling scenarios, including the
        // parallel and sequential multi-tool levels, with no control-token
        // leakage into content.
        KnownFamily(prefix: "qwen3.8-", minSizeBillions: 3.0, note: "hybrid, hermes parser; release eval 25/30 tool scenarios on 27b-mixed-3.5bpw"),
        // qwopus — Qwen+Opus distillation, hermes parser. PR #333
        // closure-verified 5/5 weather tool calls on qwopus-27b-8bit.
        KnownFamily(prefix: "qwopus-", minSizeBillions: 3.0, note: "PR #333 closure-verified 5/5 weather tool calls on 27b-8bit"),
        // qwen3-4b / qwen3-8b / qwen3-coder — bench loop covers
        // qwen3-4b-thinking-2507 + qwen3-4b-instruct-2507; the 8B
        // sibling shares the same parser family. qwen3-coder is the
        // headline coding alias (qwen3-coder-30b benched).
        // Note: ``qwen3-vl-*`` (multimodal) is intentionally excluded
        // — see ``unsupportedQwen3Prefixes`` below.
        // Note: ``qwen3-0.6b-*`` is excluded by the 3B floor.
        KnownFamily(prefix: "qwen3-4b", minSizeBillions: 3.0, note: "loop benches thinking-2507 + instruct-2507"),
        KnownFamily(prefix: "qwen3-8b", minSizeBillions: 3.0, note: "qwen3 family, hermes parser; 8B size sibling of bench-covered 4B"),
        KnownFamily(prefix: "qwen3-coder", minSizeBillions: 3.0, missingSizeAllowList: ["qwen3-coder-4bit"], note: "headline coding alias; qwen3-coder-30b benched in eval suite; bare ``qwen3-coder-4bit`` (no size token) ships in the catalog and promotes via exact allow-list"),
        // qwen2.5 — older eval-suite alias still in catalog.
        KnownFamily(prefix: "qwen2.5", minSizeBillions: 3.0, note: "legacy eval-suite verified at 14B"),

        // MARK: Bonsai (Ternary) family — the first-run starter

        // bonsai — PrismML Ternary Bonsai, a real Qwen3-architecture
        // checkpoint packed at 1.58-bit (MLX-2bit). Verified 6/6 clean
        // tool_calls (hermes parser) on the 1.7B in the eval harness
        // (rapid-mlx PR #1092); it's the first-run starter, so promoting
        // it to .known lets the empty-state capability chips surface.
        // 1.5B floor: the starter is 1.7B (below the 3.0 floor the other
        // rows use), and the smallest shipped ternary is 1.7B.
        KnownFamily(prefix: "bonsai-", minSizeBillions: 1.5, note: "Ternary Bonsai (Qwen3 arch); 6/6 clean tool_calls on 1.7B, hermes parser; rapid-mlx PR #1092. NOT the first-run starter since 2026-08-05 — it degenerates on plain chat; see QuickstartCoordinator.defaultChoice"),

        // MARK: Llama family

        // llama-3.1-8b — cycle-6 F-1/F-2 confirm tool_call path works
        // (the findings themselves are about prompt-injection bias,
        // not silent failure).
        KnownFamily(prefix: "llama-3.1-8b", minSizeBillions: 3.0, note: "cycle-6 F-1/F-2 confirm tool_call emission"),
        // llama3-3b — cycle-10 verified 5/5 clean weather tool calls
        // at temp=0; smallest empirically-good llama. llama3-1b is
        // ``.broken`` via brokenPrefixes (cycle-9 F9-001 schema-leak).
        KnownFamily(prefix: "llama3-3b", minSizeBillions: 3.0, note: "cycle-10 verified 5/5 weather tool calls at temp=0"),
        // llama3-8b — same parser as llama3-3b; size sibling.
        KnownFamily(prefix: "llama3-8b", minSizeBillions: 3.0, note: "size sibling of cycle-10-verified llama3-3b"),

        // MARK: Gemma 4 family

        // gemma-4 — gemma4 parser, cycle-6 cross-model walk + PR #321
        // (PromptProcessingBatch) + PR #323 (tool dispatch placeholder)
        // closure-verified on 26B. 12B / 31B siblings share the
        // gemma4 parser path.
        KnownFamily(prefix: "gemma-4-", minSizeBillions: 3.0, note: "gemma4 parser; cycle-6 cross-model on 26B; PR #321/#323 closure-verified"),

        // MARK: GLM family

        KnownFamily(prefix: "glm4.5-", minSizeBillions: 3.0, missingSizeAllowList: ["glm4.5-air-4bit"], note: "glm47 parser shared with glm4.7; eval-suite verified family; bare ``glm4.5-air-4bit`` (no size token) ships in the catalog and promotes via exact allow-list"),
        KnownFamily(prefix: "glm4.7", minSizeBillions: 3.0, missingSizeAllowList: ["glm4.7-4bit"], note: "glm47 parser, benched in evals/tool_calling .known on 9-cat scorecard; bare ``glm4.7-4bit`` (PR #333 pin) covered via exact allow-list"),
        KnownFamily(prefix: "glm-4.7", minSizeBillions: 3.0, note: "underscore variant of glm4.7 namespace (alias-name convention); no catalog size-less aliases at this prefix"),

        // MARK: GPT-OSS / Minimax / Mistral families

        // gpt-oss — minimax/harmony parser; cycle-7 confirmed clean on
        // safety + tool-edge probes. The mxfp4-q8 quant suffix was the
        // pre-fix sibling miss.
        KnownFamily(prefix: "gpt-oss-", minSizeBillions: 3.0, note: "minimax/harmony parser; cycle-7 safety+tool-edge clean"),
        // minimax-m2.x — minimax parser, 235B MoE family. Loop verifies
        // the parser path on this family.
        KnownFamily(prefix: "minimax-m", minSizeBillions: 3.0, missingSizeAllowList: ["minimax-m2.5-4bit", "minimax-m2.7-mxfp4"], note: "minimax parser; 235B MoE family; aliases ``minimax-m2.5-4bit`` / ``minimax-m2.7-mxfp4`` have no \\d+b token (the m-version is the model name, not a size) — promote via exact allow-list"),
        // NOTE: the Mistral family (``mistral-``, ``devstral``) was
        // demoted to ``brokenPrefixes`` on the 2026-07-09 sweep (hermes
        // parser mis-parsed its ``[TOOL_CALLS]…[ARGS]{…}`` output, 6/6
        // leak). The engine parser fix (rapid-mlx #1071/#1077) is now
        // bundled (submodule 7b6a787) so the family is RE-PROMOTED — see
        // the ``Mistral / Devstral families`` rows further below.

        // MARK: Hermes / Nemotron families (large variants)

        // hermes4 (large; distinct from hermes3-8b which is .broken).
        // 70B size with hermes parser; SFT issue that bit hermes3-8b
        // does not reproduce at this scale (the parser/family are
        // well-understood at 70B per the same hermes-parser path
        // qwen3.5-122b uses).
        KnownFamily(prefix: "hermes4", minSizeBillions: 3.0, note: "hermes parser; 70B scale; distinct from .broken hermes3-8b SFT issue"),

        // MARK: Mistral / Devstral families
        //
        // Re-promoted from ``brokenPrefixes`` after the engine parser
        // fix (rapid-mlx #1071/#1077) landed in the bundled submodule
        // (7b6a787): the family now routes through the ``mistral``
        // tool-call parser instead of ``hermes``, so the textbook
        // ``[TOOL_CALLS]name[ARGS]{json}`` output is parsed cleanly.
        // Bench on the bundled engine: devstral-v2-24b-4bit → clean
        // ``tool_calls`` (get_weather). Parser fix is family-level, so
        // the 24B bench validates the routing for the 119B siblings
        // too.
        KnownFamily(prefix: "mistral-", minSizeBillions: 3.0, note: "mistral parser (rapid-mlx #1071/#1077, bundled 7b6a787); [TOOL_CALLS] format parsed cleanly; devstral-v2-24b family-bench confirms routing"),
        KnownFamily(prefix: "devstral", minSizeBillions: 3.0, note: "mistral parser (rapid-mlx #1071/#1077, bundled 7b6a787); devstral-v2-24b-4bit benched clean on the bundled engine"),
        // nemotron — hermes parser; cycle-7 headline confirmed engine
        // works (continuous batching scales) and parser/family contour
        // matches verified 30B aliases (qwen3-coder-30b).
        KnownFamily(prefix: "nemotron", minSizeBillions: 3.0, note: "cycle-7 headline: engine + parser path verified at 30B scale"),

        // MARK: Gemma 4 e-series (efficient variants)

        // gemma-4-e4b — efficient 4B variant of gemma-4 family; same
        // gemma4 parser. ``gemma-4-e2b`` excluded by 3B floor (sub-3B).
        KnownFamily(prefix: "gemma-4-e4b", minSizeBillions: 3.0, note: "efficient variant; same gemma4 parser as gemma-4-12b/26b"),
    ]

    /// Family prefixes we DELIBERATELY do not promote to ``.known``
    /// even though they share a name root with a verified family.
    /// Defined as a separate list so the family classifier can shadow
    /// the more general ``knownFamilies`` rows without us needing to
    /// rewrite each verified family's prefix to exclude these.
    ///
    /// Each entry below is an alias name shape we have first-hand
    /// reason to keep ``.unknown`` for:
    ///
    ///   * Multimodal-only chat surfaces (``qwen3-vl-*``) — tool-call
    ///     wire shape under vision-language conditioning is not what
    ///     the loop has benched. The text-only parser path may work
    ///     but we don't have evidence yet.
    ///   * Sub-min-size siblings that would otherwise inherit the
    ///     family's verified bucket — covered by the size floor in
    ///     ``confidence(for:)``.
    ///   * Families whose name prefix-matches a verified family but
    ///     belong to a distinct SFT lineage (e.g. ``qwen3-0.6b``
    ///     belongs to qwen3 namespace but is the sub-1B tiny; size
    ///     floor covers it).
    static let unverifiedPrefixOverrides: [String] = [
        // Vision-language siblings of qwen3 — multimodal tool-call
        // surface not benched. Conservative .unknown.
        "qwen3-vl-",
        // qwen3-0.6b — sub-1B; the ToolUseCapability "broken-means-
        // empirically-observed" rule keeps it .unknown rather than
        // .broken. Picker visibility filter handles surfacing.
        "qwen3-0.6b",
    ]

    /// Backwards-compatibility shim: the union of every family prefix
    /// from ``knownFamilies``, exposed so the existing
    /// ``CapabilityChipsAliasGateTests`` / ``ToolUseCapabilityTests``
    /// suite that iterates ``ToolUseCapability.knownPrefixes`` still
    /// has a stable surface. Each entry is guaranteed to round-trip
    /// to ``.known`` via ``confidence(for:)`` (sized at exactly the
    /// family's ``minSizeBillions`` floor, so the family + size guard
    /// passes).
    ///
    /// Use ``knownFamilies`` directly for any new code that needs the
    /// family + size structure; ``knownPrefixes`` is retained only so
    /// the parameterised tests at ``CapabilityChipsAliasGateTests``
    /// (which walks the live ``knownPrefixes`` array) keep covering
    /// the verified-family contract.
    static var knownPrefixes: [String] {
        knownFamilies.map { family in
            // Build a representative alias that satisfies both the
            // prefix and the size floor. The representative is just
            // "prefix" + "minSize" + "b" if the prefix doesn't already
            // include a size token; otherwise the prefix alone is
            // enough.
            let needsSize = !family.prefix.contains(where: { $0.isNumber })
                || !family.prefix.hasSuffix("b")
            if !needsSize {
                // Prefix already includes its full size (e.g.
                // ``llama3-3b``) — return it unchanged.
                return family.prefix
            }
            // Append the family's min size + "b" to satisfy the
            // sizeFloor guard in ``confidence(for:)``. The
            // representative might be slightly larger than what a
            // real alias would look like (e.g. ``qwen3.5-3b``
            // doesn't ship) but it round-trips to .known cleanly,
            // which is all the back-compat suite needs.
            let size = family.minSizeBillions
            let sizeToken: String
            if size == size.rounded() {
                sizeToken = "\(Int(size))b"
            } else {
                sizeToken = "\(size)b"
            }
            return family.prefix.hasSuffix("-")
                ? "\(family.prefix)\(sizeToken)"
                : "\(family.prefix)-\(sizeToken)"
        }
    }

    /// Confidence we have that ``alias`` will emit well-formed
    /// ``tool_calls`` when given a tool-eligible prompt with
    /// ``tool_choice: "auto"``.
    ///
    /// See ``ToolUseConfidence`` for the bucket semantics. Three-step
    /// match (broken denylist → family/size → fallback unknown).
    static func confidence(for alias: String) -> ToolUseConfidence {
        guard !alias.isEmpty else { return .unknown }
        let needle = alias.localizedLowercase

        // Step 1: denylist match (broken takes priority).
        for broken in brokenPrefixes where needle.hasPrefix(broken) {
            return .broken
        }

        // Step 2: deliberate-unknown overrides shadow the family match
        // below — e.g. ``qwen3-vl-8b-4bit`` matches the ``qwen3-``
        // family root but multimodal tool-call surface is unverified.
        for override in unverifiedPrefixOverrides where needle.hasPrefix(override) {
            return .unknown
        }

        // Step 3: family + size guard. An alias is .known when its
        // name starts with a verified family prefix AND parses to a
        // size >= the family's minSizeBillions floor.
        for family in knownFamilies where needle.hasPrefix(family.prefix) {
            // Size floor. ``parseParamsBillions`` returns the LARGEST
            // ``\d+b`` token — for MoE aliases (``qwen3.6-35b-a3b``)
            // that's the total weight count, which is what the
            // tool-call quality contour tracks (35B class, not 3B
            // active params).
            if let size = ModelSizing.parseParamsBillions(needle) {
                if size >= family.minSizeBillions {
                    return .known
                }
                // Family matched but size too small — explicitly
                // .unknown rather than fall through to subsequent
                // family rows (no overlapping family in this table).
                return .unknown
            }
            // No size token in the alias name. Codex r2 MAJOR
            // tightened this from prefix-wide
            // ``allowMissingSize: Bool`` to per-family exact-alias
            // ``missingSizeAllowList: Set<String>``. The 4 catalog
            // aliases without a size token (``qwen3-coder-4bit`` /
            // ``glm4.5-air-4bit`` / ``glm4.7-4bit`` /
            // ``minimax-m2.5-4bit`` / ``minimax-m2.7-mxfp4``) are
            // each listed by full name on their family row; a
            // user-typed ``qwen3-coder-experimental`` that shares
            // the prefix but isn't in the allow-list falls through
            // to ``.unknown`` rather than silently inheriting the
            // family verdict. The needle is already lowercased and
            // the allow-list entries are pre-lowercased at
            // initialisation, so the contains check is
            // case-insensitive.
            if family.missingSizeAllowList.contains(needle) {
                return .known
            }
            return .unknown
        }

        return .unknown
    }

    /// Convenience for the compose-row Tools chip. Returns ``true``
    /// when the chip should be disabled (and a tooltip surfaced)
    /// because the selected alias is known-broken at tool-calling.
    ///
    /// Note: ``.unknown`` returns ``false`` (chip stays enabled).
    /// We do NOT want to regress on aliases the loop hasn't covered.
    static func shouldDisableToolsChip(alias: String) -> Bool {
        confidence(for: alias) == .broken
    }

    /// User-facing tooltip for a disabled Tools chip. Tells the user
    /// WHY tools are off so they don't think the UI is broken — the
    /// alias has been observed to silently ignore tools and the
    /// desktop is being honest about it.
    ///
    /// Kept short — the help-tooltip is one line.
    static let disabledToolsTooltip: String =
        "Tools are unavailable on this model. " +
        "Empirical bench shows it ignores tool calls and hallucinates the answer instead. " +
        "Pick a tool-capable model (e.g. qwen3.5-4b or larger) to enable tools."

    // MARK: - FU-9: per-state picker badge label

    /// Single source of truth for the picker / sticker badge text per
    /// ``ToolUseConfidence`` bucket. Centralised here so every render
    /// site (alias-row title, hover tooltip, VoiceOver label) shares
    /// the same per-state copy and a future label change touches one
    /// constant instead of three call sites.
    ///
    /// Pre-FU-9 the picker rendered a single literal ``"no tools"``
    /// for BOTH ``.broken`` AND ``.unknown``. That collapsed two
    /// distinct states ("empirically tested and known to leak schema"
    /// vs "we have no signal either way") into one alarmist label —
    /// a user who picked an unbenched 4B alias (e.g.
    /// ``bonsai-4b-unpacked``) saw the same sticker as on
    /// ``hermes3-8b-4bit`` even though the bonsai family might be
    /// fine. The split mirrors the bucket semantics:
    ///
    ///   * ``.known`` → ``nil`` (no badge — the alias is verified).
    ///   * ``.broken`` → ``"no tools"`` (UNCHANGED; preserves the
    ///     strong wording the empirical-evidence path earned).
    ///   * ``.unknown`` → ``"tools unverified"`` (NEW; softer copy
    ///     that says "we don't know" instead of "we know it's
    ///     broken").
    ///
    /// The chip-row gate in
    /// ``ChatView.capabilityChipKinds(forAlias:)`` still collapses
    /// chips for BOTH ``.broken`` and ``.unknown`` — the goal is to
    /// avoid over-promising tool support on aliases the loop hasn't
    /// benched, not to ship the chips with a softer caveat.
    static func badgeLabel(for confidence: ToolUseConfidence) -> String? {
        switch confidence {
        case .known:
            return nil
        case .broken:
            return "no tools"
        case .unknown:
            return "tools unverified"
        }
    }

    /// Convenience overload — resolve the confidence for ``alias``
    /// and return its badge label. Empty alias suppressed defensively
    /// so the picker placeholder row (pre-model-selection) never
    /// renders the badge.
    static func badgeLabel(forAlias alias: String) -> String? {
        guard !alias.isEmpty else { return nil }
        return badgeLabel(for: confidence(for: alias))
    }
}
