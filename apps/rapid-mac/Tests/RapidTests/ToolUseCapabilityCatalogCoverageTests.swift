import Foundation
import Testing
@testable import Rapid

/// #342 follow-up — exhaustive coverage check for ``ToolUseCapability``
/// across every alias currently shipped by the bundled rapid-mlx
/// catalog. Pre-fix audit of ``third_party/rapid-mlx/vllm_mlx/aliases.json``
/// at HEAD (commit ``232e63a``) found 21 ``.known`` / 3 ``.broken`` /
/// 68 ``.unknown`` out of 92 aliases — 74% over-classified as
/// ``.unknown`` because the original ``knownPrefixes`` list was a
/// string-prefix array missing size siblings (``qwen3.5-9b-4bit`` did
/// not match the ``qwen3.5-4b`` / ``qwen3.5-7b`` / etc. entries). A
/// single missing size meant every quant of that size fell to
/// ``.unknown`` and the picker rendered a "· no tools" badge on a
/// model that works fine.
///
/// The follow-up replaces the bare string-prefix list with a
/// family-aware classification: an alias resolves to ``.known`` when
/// its family is empirically verified AND its size meets the family's
/// minimum (typically 3B, the smallest empirically-good llama
/// chat-tuned quant per cycle-10).
///
/// This file pins the expected distribution per-alias AND in
/// aggregate. The aggregate counts double as a regression guard:
/// if a future PR re-broadens ``.unknown`` (e.g. by accidentally
/// dropping a family from the verified list), the count moves and
/// this test fails before merge.
///
/// **Data source**: parsed at TEST TIME from the literal aliases.json
/// the bundled rapid-mlx submodule points at. The aliases listed in
/// ``expectedKnown`` / ``expectedBroken`` / ``expectedUnknown`` are
/// the 92 entries at submodule commit ``4ba7053989cd``. If the
/// submodule bumps and surfaces a new alias the catalog test will
/// surface it as "alias '<name>' missing from expectation table" so
/// the next reviewer makes a deliberate classification call rather
/// than letting the new alias default silently.
@Suite("ToolUseCapability — catalog coverage (#342 follow-up)")
struct ToolUseCapabilityCatalogCoverageTests {

    // MARK: - Step-0 failing tests: aliases that MUST resolve to .known after fix

    /// Concrete repro list from the task brief — these aliases were
    /// in ``.unknown`` at HEAD and demonstrably work for tool calls
    /// (their family is verified working, just a size/quant sibling
    /// of the original ``knownPrefixes`` entries). At HEAD these will
    /// fail; after the family-aware fix they pass.
    @Test("aliases newly promoted to .known (Step-0 repro)", arguments: [
        // Qwen 3.5 size siblings missing from HEAD's prefix list
        "qwen3.5-9b-4bit",
        "qwen3.5-9b-8bit",
        "qwen3.5-35b-4bit",
        "qwen3.5-35b-8bit",
        // Qwen 3 hermes-parser size siblings
        "qwen3-coder-4bit",
        "qwen3-coder-30b-4bit",
        "qwen3-8b-4bit",
        "qwen3-8b-8bit",
        "qwen3-4b-8bit",
        // Gemma 4 family — gemma4 parser verified on 26B (PR #321/#323)
        "gemma-4-12b-4bit",
        "gemma-4-12b-8bit",
        "gemma-4-12b-qat-4bit",
        "gemma-4-31b-4bit",
        "gemma-4-31b-8bit",
        "gemma-4-31b-qat-4bit",
        // GPT-OSS sibling form (mxfp4-q8 quant suffix didn't match "gpt-oss-20b")
        "gpt-oss-20b-mxfp4-q8",
    ])
    func aliasNewlyKnown(alias: String) {
        #expect(
            ToolUseCapability.confidence(for: alias) == .known,
            "Alias '\(alias)' should be .known: family is empirically verified and size >= 3B. Pre-fix HEAD returned .unknown because the original knownPrefixes string list missed this size/quant sibling."
        )
    }

    // MARK: - Step-0 failing tests: aliases that MUST resolve to .broken after fix

    /// Vibethinker — emits ``<JSON>`` wrapper instead of hermes
    /// ``<tool_call>``, so the hermes parser cannot extract args.
    /// Pre-fix HEAD: ``.unknown``. Per cycle-2 fuzz-correctness
    /// entries in ``bug_report.md`` ("vibethinker-3b-8bit emits
    /// `<JSON>...</JSON>` wrapper instead of hermes
    /// `<tool_call>...</tool_call>`" + 4-duplicate-tool-call
    /// pathology) the model silently fails at tool-calling. Two
    /// shipped aliases: ``vibethinker-1.5b-4bit`` (sub-3B; also
    /// hidden by picker visibility filter) and
    /// ``vibethinker-3b-8bit``.
    @Test("vibethinker aliases are .broken (cycle-2 <JSON>-wrapper finding)", arguments: [
        "vibethinker-1.5b-4bit",
        "vibethinker-3b-8bit",
    ])
    func vibethinkerIsBroken(alias: String) {
        #expect(
            ToolUseCapability.confidence(for: alias) == .broken,
            "Alias '\(alias)' should be .broken: cycle-2 fuzz-correctness confirmed the model emits <JSON> wrapper instead of hermes <tool_call> markers. The picker badge + chip suppression must fire."
        )
    }

    // MARK: - Regression pins for PR #333 broken denylist

    /// The 3 empirically-broken aliases from PR #333 MUST continue
    /// resolving to ``.broken`` after the refactor — the
    /// denylist semantics must not regress.
    @Test("PR #333 broken aliases remain .broken after refactor", arguments: [
        "phi-4-mini-reasoning-4bit",
        "phi-4-mini-reasoning-8bit",
        "hermes3-8b-4bit",
        "hermes3-8b",
        "llama3-1b-4bit",
        "llama3-1b-8bit",
    ])
    func pr333BrokenRegressionPin(alias: String) {
        #expect(
            ToolUseCapability.confidence(for: alias) == .broken,
            "PR #333 .broken contract regressed on '\(alias)'. The capability map must continue treating this alias as silent-degradation."
        )
    }

    // MARK: - Exhaustive catalog matrix

    /// Per-alias expected confidence for the 92-alias bundled catalog
    /// at submodule commit ``4ba7053989cd``. The matrix below is the
    /// load-bearing contract: a deliberate classification call for
    /// every alias the desktop ships against. Adding a new alias
    /// (submodule bump) is a forcing function — the catalog-walk test
    /// surfaces unseen aliases so the reviewer must add a row here.
    ///
    /// Choices:
    /// * ``.known``: family has empirical evidence (fuzz cycle, bench
    ///   loop) of well-formed tool_calls AND alias size >= 3B.
    /// * ``.broken``: empirical evidence of silent-degradation OR
    ///   schema-leak (cycle-2 F-CORR vibethinker, cycle-4 F-1 hermes3,
    ///   cycle-9 F9-001 llama3-1b, cycle-11 F-11-5 phi-4-mini-reasoning).
    /// * ``.unknown``: legitimate gap — brand-new family (granite4,
    ///   nanbeige4.1), unverified parser (gemma3 hermes-via-
    ///   profile, qwen3-vl multimodal-only signal), parser=None
    ///   (gemma-3n-e*, phi-3.5-mini, deepseek-coder-v2-lite,
    ///   phi-4-mini, qwen3-0.6b sub-1B), or non-chat modality
    ///   (diffusion-gemma text-diffusion).
    static let expectedMatrix: [String: ToolUseConfidence] = [
        // bonsai — Ternary Bonsai (Qwen3 arch), hermes parser. The
        // 1.7B ternary is the first-run starter: 6/6 clean tool_calls
        // on the eval harness (rapid-mlx PR #1092) → .known. (The old
        // FP16 ``bonsai-*-unpacked`` aliases were dropped in #1092.)
        "bonsai-1.7b-2bit": .known,
        // LiquidAI LFM2.x — engine-side tool parser landed in
        // rapid-mlx #1076, but the desktop bench loop has not yet
        // covered the family, so no ``ToolUseCapability`` KnownFamily
        // row exists and these resolve to .unknown by design. Promote
        // to .known only with first-hand bench evidence.
        "lfm2.5-1b-4bit": .unknown,
        "lfm2.5-8b-a1b-4bit": .unknown,
        "lfm2-24b-a2b-4bit": .unknown,
        // Hy3 preview — tool + reasoning parser landed engine-side
        // (rapid-mlx #1070/#1072), Ultra-only integration matrix; not
        // yet benched by the desktop loop → .unknown until covered.
        "hy3-preview-4bit": .unknown,
        // deepseek — varied. 2026-07-09 recommended-model sweep gave
        // first-hand user-surface evidence on two of these:
        //   * deepseek-coder-v2-lite-16b-4bit — parser=None + invents
        //     ad-hoc tool names → 6/6 raw envelope leak → .broken.
        //   * deepseek-r1-8b-4bit — invents a different JSON schema per
        //     run → 4/8 leak → .broken (prefix pinned to the 8B distill;
        //     the 32B and other R1 variants stay .unknown, untested).
        "deepseek-coder-v2-lite-16b-4bit": .broken,
        "deepseek-r1-32b-4bit": .unknown,
        "deepseek-r1-8b-4bit": .broken,
        "deepseek-v4-flash-2bit": .unknown,
        "deepseek-v4-flash-4bit": .unknown,
        "deepseek-v4-flash-8bit": .unknown,
        // devstral / mistral — Mistral-family. 2026-07-09 sweep: the
        // bundled engine ships these with tool_call_parser=hermes,
        // which can't read their [TOOL_CALLS]…[ARGS]{…} output → 6/6
        // leak. PARSER MISCONFIG, not model incapacity (a swap to the
        // "mistral" parser makes them 6/6 clean). The engine parser fix
        // (rapid-mlx #1071/#1077) is NOW BUNDLED (submodule 7b6a787) and
        // devstral-v2-24b-4bit was re-benched clean on it → re-promoted
        // to .known via the ``devstral`` knownFamily row.
        "devstral-24b-4bit": .known,
        "devstral-v2-24b-4bit": .known,
        // diffusion-gemma — text-diffusion modality, NOT chat.
        // No tool-call surface; keep .unknown.
        "diffusion-gemma-26b-4bit": .unknown,
        "diffusion-gemma-26b-8bit": .unknown,
        // embeddinggemma — sentence-embedding modality (parser=None per
        // aliases.json), no chat / tool-call surface. v0.8.11 submodule
        // bump (PR #379) introduced these aliases.
        "embeddinggemma-300m-6bit": .unknown,
        "embeddinggemma-300m-8bit": .unknown,
        // gemma-3n — parser=None per aliases.json; legitimately unknown.
        "gemma-3n-e2b-4bit": .unknown,
        "gemma-3n-e4b-4bit": .unknown,
        // gemma-4 — gemma4 parser verified on 26B (cycle-6 cross-model);
        // PR #321 + #323 closure-verified. All sizes ≥ 12B safe.
        "gemma-4-12b-4bit": .known,
        "gemma-4-12b-8bit": .known,
        "gemma-4-12b-qat-4bit": .known,
        "gemma-4-12b-qat-8bit": .known,
        "gemma-4-26b-4bit": .known,
        "gemma-4-26b-qat-4bit": .known,
        "gemma-4-31b-4bit": .known,
        "gemma-4-31b-8bit": .known,
        "gemma-4-31b-qat-4bit": .known,
        "gemma-4-31b-qat-8bit": .known,
        // gemma-4 e-series (efficient variants). e2b is sub-3B and
        // efficient-variant unverified — keep .unknown. e4b ≥ 4B per
        // mlx-community labelling — known by family.
        "gemma-4-e2b-4bit": .unknown,
        "gemma-4-e4b-4bit": .known,
        // gemma3 — hermes via profile-default; loop never confirmed
        // tool-call emission on this family. Same hermes-via-profile
        // pattern that failed on phi-4-mini-reasoning; conservative
        // .unknown rather than .known.
        "gemma3-12b-4bit": .unknown,
        "gemma3-1b-4bit": .unknown,
        "gemma3-1b-qat-4bit": .unknown,
        "gemma3-27b-4bit": .unknown,
        "gemma3-27b-qat-4bit": .unknown,
        "gemma3-4b-qat-4bit": .unknown,
        // glm — glm47 parser, verified in eval suite.
        "glm4.5-air-4bit": .known,
        "glm4.7-9b-4bit": .known,
        // gpt-oss — harmony parser (cycle-7 confirmed clean on safety
        // + tool-edge probes). The mxfp4-q8 quant suffix was the
        // pre-fix sibling miss.
        "gpt-oss-20b-mxfp4-q8": .known,
        // granite4 — IBM granite4 family; loop never benched. Unknown.
        "granite4-h-micro-4bit": .unknown,
        "granite4-tiny-4bit": .unknown,
        // hermes3 — empirically broken (cycle-4 F-1 silent-degradation).
        "hermes3-8b-4bit": .broken,
        // hermes4-70b — hermes parser, large model, distinct from
        // hermes3 SFT issue. No loop evidence either way; conservatively
        // .known because the parser/family are well-understood at this
        // scale (loop benches qwen3.5-122b same parser + similar size).
        "hermes4-70b-4bit": .known,
        // llama-3.1 ≥ 8B — cycle-6 F-1/F-2 confirm tool_call path works
        "llama-3.1-8b-4bit": .known,
        "llama-3.1-8b-8bit": .known,
        // llama3-1b — cycle-9 F9-001 schema-leak, .broken
        "llama3-1b-4bit": .broken,
        // llama3-3b — cycle-10 verified 5/5 clean weather tool calls.
        "llama3-3b-4bit": .known,
        // minimax — minimax parser; M2.x is 235B MoE family
        "minimax-m2.5-4bit": .known,
        "minimax-m2.7-mxfp4": .known,
        // mistral-24b — Mistral-family. The 2026-07-09 parser-misconfig
        // leak is fixed now that the "mistral" parser ships bundled
        // (rapid-mlx #1071/#1077, submodule 7b6a787; devstral-v2-24b
        // family-bench clean) → re-promoted to .known.
        "mistral-24b-4bit": .known,
        // nanbeige4.1-3b — brand-new family from cycle-1 fuzz; flagged
        // for hallucination on Swift codegen but no tool-call signal.
        // Conservatively .unknown.
        "nanbeige4.1-3b-4bit": .unknown,
        // nemotron-30b — cycle-7 headline confirmed engine works,
        // but tool-calling specifically not benched. Take the
        // conservative side and mark .known because nemotron's
        // hermes parser + 30B size is on the same family contour as
        // qwen3-coder-30b which is known-good.
        "nemotron-30b-4bit": .known,
        // phi-3.5-mini — parser=None in aliases.json. Unknown.
        "phi-3.5-mini-4bit": .unknown,
        // phi-4-14b — hermes parser; phi-4 distinct from phi-4-mini-
        // reasoning. No first-hand evidence either direction. Take
        // conservative .unknown rather than promote based on family
        // (the phi-4 SFT recipe is non-public and the related
        // ``phi-4-mini-reasoning`` is .broken — that's not strong
        // signal that phi-4-14b works, just that the family is
        // mixed).
        "phi-4-14b-4bit": .unknown,
        // phi-4-mini — 2026-07-09 sweep gave first-hand evidence: the
        // non-reasoning phi-4-mini flatly REFUSES tool-eligible prompts
        // (6/6 "I can't assist with that") while chatting fine without
        // tools → .broken (strip tools, keep coherent chat). Was the
        // ≤16 GB Speed recommendation; replaced by qwen3.5-4b-4bit.
        "phi-4-mini-4bit": .broken,
        // phi-4-mini-reasoning — F-11-5 .broken
        "phi-4-mini-reasoning-4bit": .broken,
        // qwen2.5 — eval-suite verified at 14B; hermes parser.
        "qwen2.5-14b-4bit": .known,
        // qwen3-0.6b — sub-1B, picker-hidden; conservatively .unknown
        // (the cycle-7 ``bundledQwen06bIsUnknown`` regression test
        // explicitly pins this).
        "qwen3-0.6b-4bit": .unknown,
        "qwen3-0.6b-8bit": .unknown,
        // qwen3-4b / qwen3-8b — same parser family as qwen3.5-4b. Loop
        // benches qwen3-4b-thinking-2507 and qwen3-4b-instruct-2507.
        "qwen3-4b-8bit": .known,
        "qwen3-4b-instruct-2507-4bit": .known,
        "qwen3-4b-thinking-2507-4bit": .known,
        "qwen3-8b-4bit": .known,
        "qwen3-8b-8bit": .known,
        // qwen3-coder — hermes parser; same family as qwen3-coder-30b
        // which is the headline coding model.
        "qwen3-coder-30b-4bit": .known,
        "qwen3-coder-4bit": .known,
        // qwen3-vl — multimodal (vision-language). Tool-calling on
        // VL models has different wire-shape considerations; loop
        // hasn't verified tool path. Conservatively .unknown.
        "qwen3-vl-2b-4bit": .unknown,
        "qwen3-vl-30b-4bit": .unknown,
        "qwen3-vl-4b-4bit": .unknown,
        "qwen3-vl-8b-4bit": .unknown,
        // qwen3.5 — the backbone family. All sizes verified through
        // the eval suite or fuzz loop. 122B headline + 4B/9b/27b/35b
        // size siblings.
        // -6bit / -nvfp4 / -mxfp4 quant siblings landed with the v0.9.8
        // sidecar bump (#949 community-curated quant variants). Each is a
        // quant variant of an already-classified Qwen3.5/3.6 alias with
        // the same family and tool-call parser as its -4bit/-8bit siblings
        // (qwen3.5 -> hermes, qwen3.6 -> qwen3_coder_xml), so the same
        // .known classification applies.
        "qwen3.5-122b-6bit": .known,
        "qwen3.5-122b-8bit": .known,
        "qwen3.5-122b-mxfp4": .known,
        "qwen3.5-27b-4bit": .known,
        "qwen3.5-27b-6bit": .known,
        "qwen3.5-27b-8bit": .known,
        "qwen3.5-35b-4bit": .known,
        "qwen3.5-35b-6bit": .known,
        "qwen3.5-35b-8bit": .known,
        "qwen3.5-4b-4bit": .known,
        "qwen3.5-4b-6bit": .known,
        "qwen3.5-4b-8bit": .known,
        "qwen3.5-9b-4bit": .known,
        "qwen3.5-9b-6bit": .known,
        "qwen3.5-9b-8bit": .known,
        // qwen3.6 — hybrid+MoE family, qwen3_coder_xml parser. Loop
        // benches all listed quants.
        "qwen3.6-27b-4bit": .known,
        "qwen3.6-27b-6bit": .known,
        "qwen3.6-27b-8bit": .known,
        "qwen3.6-27b-ud": .known,
        "qwen3.6-35b-4bit": .known,
        "qwen3.6-35b-6bit": .known,
        "qwen3.6-35b-8bit": .known,
        "qwen3.6-35b-dwq": .known,
        "qwen3.6-35b-mxfp4": .known,
        "qwen3.6-35b-nvfp4": .known,
        "qwen3.6-35b-ud": .known,
        // qwen3.8 — hybrid family, hermes parser. The mixed-3.5bpw entry
        // is our own build; release eval scored 25/30 tool scenarios.
        "qwen3.8-27b-4bit": .known,
        "qwen3.8-27b-mixed-3.5bpw": .known,
        // qwopus — Qwen+Opus distillation, hermes parser. PR #333
        // closure-verified 5/5 weather tool calls on qwopus-27b-8bit.
        "qwopus-27b-4bit": .known,
        "qwopus-27b-8bit": .known,
        // qwopus-9b — sibling at 9B, same family.
        "qwopus-9b-4bit": .known,
        // smollm3-3b — small-family-3B, hermes parser. Loop hasn't
        // benched tool-calling specifically. Conservatively .unknown
        // (3B small-llm SFT tends to schema-leak per cycle-9).
        "smollm3-3b-4bit": .unknown,
        // ui-tars — specialised GUI agent family with bespoke
        // ``ui_tars`` tool_call + reasoning parsers (distinct from
        // hermes/qwen). Loop has never benched UI-TARS tool-calling
        // and the family's output shape is screenshot-grounded GUI
        // action selection, not generic tool_call wire-protocol.
        // Conservatively ``.unknown`` until empirically verified.
        // v0.8.11 submodule bump (PR #379) introduced these aliases.
        "ui-tars-1.5-7b-4bit": .unknown,
        "ui-tars-1.5-7b-6bit": .unknown,
        "ui-tars-1.5-7b-8bit": .unknown,
        "ui-tars-72b-dpo-4bit": .unknown,
        "ui-tars-7b-dpo-4bit": .unknown,
        "ui-tars-7b-dpo-6bit": .unknown,
        "ui-tars-7b-dpo-8bit": .unknown,
        "ui-tars-7b-sft-4bit": .unknown,
        "ui-tars-7b-sft-8bit": .unknown,
        // vibethinker — cycle-2 <JSON>-wrapper. .broken.
        "vibethinker-1.5b-4bit": .broken,
        "vibethinker-3b-8bit": .broken,
        // ── v0.8.19 submodule bump (rapid-mlx v0.9.8 → v0.9.9) ──
        // New aliases pulled in by the engine bump. The loop has not
        // benched tool-calling on the exotic families below, so they
        // stay conservatively .unknown (same rule as ui-tars): a
        // capability chip must never over-promise.
        //
        // Mistral Small 4 — ``mistral-`` family. The 2026-07-09
        // parser-misconfig leak is fixed now that the "mistral" parser
        // ships bundled (rapid-mlx #1071/#1077, submodule 7b6a787). The
        // fix is family-level (routes ALL Mistral aliases to the
        // mistral parser), so the devstral-v2-24b bench validates the
        // 119B siblings too → re-promoted to .known.
        "mistral-small-4-119b": .known,
        "mistral-small-4-119b-4bit": .known,
        "mistral-small-4-119b-8bit": .known,
        // glm-5.2-reap50 — GLM 5.2 (REAP-pruned); NOT the glm4.5/4.7
        // namespace the catalog verifies, and the loop has not benched
        // the 5.2 tool path. Conservative .unknown.
        "glm-5.2-reap50": .unknown,
        // kimi-k2.6 — Moonshot Kimi K2.6; agentic-capable upstream but
        // never benched by the loop here. Conservative .unknown until
        // empirically verified.
        "kimi-k2.6": .unknown,
        // qwen3-0.6b / qwen3-1.7b — sub-3B Qwen3; excluded by the 3B
        // family floor (tool-calling unreliable at this size, and the
        // picker filters sub-1B). .unknown.
        "qwen3-0.6b": .unknown,
        "qwen3-1.7b": .unknown,
        "qwen3-1.7b-4bit": .unknown,
        // tmax-9b / tmax-27b — TMAX family; unbenched by the loop, no
        // catalog family row. Conservative .unknown.
        "tmax-9b": .unknown,
        "tmax-9b-6bit": .unknown,
        "tmax-9b-8bit": .unknown,
        "tmax-9b-bf16": .unknown,
        "tmax-27b": .unknown,
        "tmax-27b-6bit": .unknown,
        "tmax-27b-8bit": .unknown,
        // holo3.1-35b-a3b — Holo GUI/computer-use family; screenshot-
        // grounded action selection, not generic tool_call wire-protocol
        // (same rationale as ui-tars). Conservative .unknown.
        "holo3.1-35b-a3b": .unknown,
        "holo3.1-35b-a3b-8bit": .unknown,
        // ── v0.10.5 submodule bump reconciliation (2026-07-09, #514) ──
        // The catalog grew from 128 → 158 aliases across the v0.9.9 →
        // v0.10.5 engine bumps (gpt-oss quant fan-out, qwen3.6 mtp/optiq/
        // ud variants + bare no-quant keys, gemma-4 assistant/optiq/qat
        // tunes + the e2b/e4b efficient series). Each new alias is
        // classified by the SAME family+size guard the loop already
        // verified — no new empirical claim, just size-sibling coverage
        // of an already-verified family — so the catalog walk stops
        // reporting them as "unclassified". Closes rapid-desktop#514.
        //
        // gpt-oss — harmony/minimax parser (family verified, cycle-7).
        // Every quant of the 20B/120B weights + the safeguard-20b
        // moderation variant share the parser path → .known.
        "gpt-oss-20b": .known,
        "gpt-oss-20b-4bit": .known,
        "gpt-oss-20b-8bit": .known,
        "gpt-oss-20b-mxfp4-q4": .known,
        "gpt-oss-120b": .known,
        "gpt-oss-120b-4bit": .known,
        "gpt-oss-120b-mxfp4-q4": .known,
        "gpt-oss-120b-mxfp4-q8": .known,
        "gpt-oss-safeguard-20b": .known,
        // qwen3.6 — qwen3_coder_xml parser (family verified; 27b/35b
        // swept 2026-07-09). Bare no-quant keys + mtp/optiq/ud quant
        // variants are size siblings of the already-.known 27b/35b → .known.
        "qwen3.6-27b": .known,
        "qwen3.6-27b-mtp-4bit": .known,
        "qwen3.6-27b-optiq-4bit": .known,
        "qwen3.6-27b-ud-3bit": .known,
        "qwen3.6-27b-ud-6bit": .known,
        "qwen3.6-35b": .known,
        "qwen3.6-35b-mtp-4bit": .known,
        "qwen3.6-35b-optiq-4bit": .known,
        // gemma-4 — gemma4 parser (family verified on 12B/26B/31B, swept
        // 2026-07-09). assistant/optiq/qat tunes of the 12B/26B/31B and
        // the efficient e4b (~4B, >= 3B floor) → .known; the efficient
        // e2b (~2B, BELOW the 3B family floor) → .unknown, same rule that
        // keeps every sub-3B sibling conservative.
        "gemma-4-12b-assistant": .known,
        "gemma-4-12b-optiq-4bit": .known,
        "gemma-4-12b-qat-assistant-4bit": .known,
        "gemma-4-26b-8bit": .known,
        "gemma-4-26b-assistant": .known,
        "gemma-4-31b-assistant": .known,
        "gemma-4-e4b-6bit": .known,
        "gemma-4-e4b-8bit": .known,
        "gemma-4-e4b-assistant": .known,
        "gemma-4-e4b-optiq-4bit": .known,
        "gemma-4-e2b-6bit": .unknown,
        "gemma-4-e2b-8bit": .unknown,
        "gemma-4-e2b-assistant": .unknown,
    ]

    // MARK: - Family-pattern edge cases

    /// Family classification must handle a representative set of
    /// edge cases without crashing or false-positive matching.
    @Test("Family classifier edge cases", arguments: [
        // Empty alias — bedrock contract from PR #333; tests pre-existing
        ("", ToolUseConfidence.unknown),
        // Random unknown alias — must not match any family.
        ("random-alias-xyz", ToolUseConfidence.unknown),
        // Just a number — must not crash.
        ("9b", ToolUseConfidence.unknown),
        // Verified family root without trailing ``-`` or size suffix.
        // ``qwen3.5`` doesn't START with ``qwen3.5-`` (the family row's
        // prefix has the trailing ``-`` to anchor on the version
        // boundary) so no family matches and it falls to .unknown.
        ("qwen3.5", ToolUseConfidence.unknown),
        // Family prefix + sub-min size — must NOT promote to .known
        // by accident.
        ("qwen3.5-1b-4bit", ToolUseConfidence.unknown),
        // Bare family name with size = min, no quant — should be
        // .known via family bucket.
        ("qwen3.5-4b", ToolUseConfidence.known),
        // Family + size + odd suffix — quant suffix not in normal
        // set ("test"). Should still match via family + size.
        ("qwen3.5-4b-4bit-test", ToolUseConfidence.known),
        // Codex r1 MAJOR + r2 MAJOR: hypothetical user-typed aliases
        // that collide with a verified family prefix but carry no
        // size evidence. Must be .unknown — even for families with
        // a ``missingSizeAllowList`` (qwen3-coder, glm4.5-, glm4.7,
        // minimax-m), because the allow-list is exact-alias not
        // prefix-wide (codex r2 tightening).
        ("qwen3-coder-experimental", ToolUseConfidence.unknown),
        // ^ qwen3-coder family has ``missingSizeAllowList=
        //   ["qwen3-coder-4bit"]`` (codex r2 tightening). The
        //   experimental alias is NOT in the allow-list and has no
        //   size token, so it falls through to .unknown.
        ("qwen3-coder-4bit", ToolUseConfidence.known),
        // ^ The single catalog allow-listed missing-size alias for
        //   this family — must still resolve .known.
        ("glm4.5-experimental", ToolUseConfidence.unknown),
        // ^ glm4.5- family has ``missingSizeAllowList=["glm4.5-air-4bit"]``
        //   only; experimental falls through.
        ("glm4.7-experimental", ToolUseConfidence.unknown),
        // ^ glm4.7 family has ``missingSizeAllowList=["glm4.7-4bit"]``
        //   only; experimental falls through.
        ("minimax-m-experimental", ToolUseConfidence.unknown),
        // ^ minimax-m family has ``missingSizeAllowList=
        //   ["minimax-m2.5-4bit","minimax-m2.7-mxfp4"]`` only;
        //   experimental falls through.
        ("hermes4-experimental", ToolUseConfidence.unknown),
        // ^ hermes4 family has empty missingSizeAllowList (only
        //   ``hermes4-70b-4bit`` ships in catalog, has size token),
        //   so this hypothetical falls through to .unknown.
        ("nemotron-preview", ToolUseConfidence.unknown),
        // ^ nemotron family empty missingSizeAllowList; only
        //   ``nemotron-30b-4bit`` ships; experimental falls through.
        ("devstral-preview", ToolUseConfidence.unknown),
        // ^ bonsai-starter bump: the ``devstral`` brokenPrefix is gone
        //   (the mistral parser now ships bundled) and the ``devstral``
        //   knownFamily is size-gated (>= 3.0). A size-less
        //   ``devstral-preview`` parses no ``<n>b`` token, so it falls
        //   through both the (removed) broken step and the size guard
        //   to .unknown — the correct "not yet benched at this shape"
        //   verdict.
        ("mistral-preview", ToolUseConfidence.unknown),
        // ^ same as devstral-preview — no broken prefix any more, and
        //   the ``mistral-`` knownFamily needs a parseable size >= 3.0
        //   which a bare ``mistral-preview`` lacks → .unknown.
        ("qwen3.5-experimental", ToolUseConfidence.unknown),
        // ^ qwen3.5- family empty missingSizeAllowList; bare
        //   experimental falls through.
    ])
    func familyEdgeCase(alias: String, expected: ToolUseConfidence) {
        #expect(
            ToolUseCapability.confidence(for: alias) == expected,
            "Family classifier returned wrong bucket for edge case '\(alias)'."
        )
    }

    // MARK: - Helpers

    /// Load every alias key from the bundled ``aliases.json`` snapshot
    /// at submodule commit ``4ba7053989cd``. The JSON lives at
    /// ``third_party/rapid-mlx/vllm_mlx/aliases.json``; we resolve
    /// the path via ``#filePath`` so the test runs from any cwd.
    static func loadCatalogAliases() throws -> [String] {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // RapidTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // repo root
            .appendingPathComponent("third_party/rapid-mlx/vllm_mlx/aliases.json")
        let data = try Data(contentsOf: url)
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NSError(domain: "ToolUseCapabilityCatalogCoverageTests", code: 1, userInfo: [NSLocalizedDescriptionKey: "aliases.json did not decode as object"])
        }
        return Array(obj.keys)
    }
}
