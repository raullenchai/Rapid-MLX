import Foundation

/// Issue #1717: the engine's throughput knobs (KV-cache precision, prefix
/// caching) reach `serve` only as CLI flags. This is the desktop-side value
/// type for the subset we expose, resolved into `serve` argv by
/// ``launchFlags``.
///
/// **Sparse by construction.** Every field is optional and `nil` means "the
/// app passes nothing for this knob", not "the app passes the engine default".
/// A user who never opens the panel therefore produces an empty flag list and
/// an argv byte-identical to today's — which is the issue's "defaults are
/// unchanged for users who never open the panel" acceptance criterion. It also
/// means an engine-side default change reaches those users, instead of being
/// frozen by a value the app snapshotted at first launch.
///
/// **Audited surface only.** The exposed set is deliberately smaller than the
/// flag families the issue names. `--kv-bits`, `--kv-group-size`,
/// `--draft-model` and `--num-draft-tokens` are in the engine's
/// deprecated-no-op block (`vllm_mlx/cli.py`: "consumed-and-discarded: stored
/// on ``args`` but never read"), so exposing them would ship switches wired to
/// nothing. Speculative decoding uses the alias registry's audited preset and
/// emits canonical `--speculative-config` JSON rather than a legacy flag.
struct ModelPerfConfig: Codable, Equatable, Sendable {
    /// KV-cache precision / compression. One knob, not two, because the
    /// engine treats them as one: `vllm_mlx/cli.py` resolves `--kv-cache-dtype`
    /// only `if not args.kv_cache_turboquant and not args.kv_cache_quantization`
    /// — with TurboQuant on, the dtype flag is silently ignored because
    /// TurboQuant owns the V cache. Modelling these as two independent
    /// controls would let the panel show "int8 + TurboQuant" while the engine
    /// ran TurboQuant alone, which is precisely the "neither they nor we can
    /// tell which toggle did it" failure the issue opens with.
    var kvCacheMode: KVCacheMode?

    /// Prefix caching on/off. The engine defaults this on; the flag pair
    /// exists so an operator can A/B it.
    var prefixCacheEnabled: Bool?

    /// Prefix-cache memory ceiling in MB. `nil` leaves the engine's
    /// auto-detection (~20% of RAM) alone.
    var cacheMemoryMB: Int?

    /// Explicit opt-in to the selected alias's registry-advertised preset.
    /// `nil` is off; the method is persisted so launch remains deterministic.
    var speculativePreset: SpeculativeDecodingPreset?

    init(
        kvCacheMode: KVCacheMode? = nil,
        prefixCacheEnabled: Bool? = nil,
        cacheMemoryMB: Int? = nil,
        speculativePreset: SpeculativeDecodingPreset? = nil
    ) {
        self.kvCacheMode = kvCacheMode
        self.prefixCacheEnabled = prefixCacheEnabled
        self.cacheMemoryMB = cacheMemoryMB
        self.speculativePreset = speculativePreset
    }

    /// True when the user has not overridden anything. Such a config is
    /// dropped rather than persisted, so the store never accumulates rows
    /// that encode "no opinion".
    var isEmpty: Bool {
        kvCacheMode == nil && prefixCacheEnabled == nil && cacheMemoryMB == nil
            && speculativePreset == nil
    }

    /// Bounds for ``cacheMemoryMB``. The floor is the engine's own smallest
    /// useful budget; below it the cache thrashes and the knob reads as
    /// "prefix caching is broken" rather than "I set it too low".
    static let cacheMemoryMBRange: ClosedRange<Int> = 256 ... 32_768

    /// `serve` argv contributed by this config, in a stable order.
    ///
    /// Order matters only for reproducibility in tests and in the
    /// `--dev-snapshot` output; the engine's argparse is order-insensitive
    /// across these flags. Each branch is skipped entirely when the field is
    /// `nil`, so an untouched config contributes an empty array.
    var launchFlags: [String] {
        var flags: [String] = []
        if let kvCacheMode {
            flags.append(contentsOf: kvCacheMode.launchFlags)
        }
        if let prefixCacheEnabled {
            // Distinct flags rather than a value — the engine models this as
            // a `--enable-prefix-cache` / `--disable-prefix-cache` pair.
            flags.append(prefixCacheEnabled ? "--enable-prefix-cache" : "--disable-prefix-cache")
        }
        if let cacheMemoryMB {
            let clamped = min(max(cacheMemoryMB, Self.cacheMemoryMBRange.lowerBound),
                              Self.cacheMemoryMBRange.upperBound)
            flags.append(contentsOf: ["--cache-memory-mb", String(clamped)])
        }
        return flags
    }

    /// Alias-aware rendering used by the app's launch pipeline. The preset was
    /// advertised by aliases.json through `rapid-mlx models`; persisting that
    /// exact repo keeps relaunch deterministic and avoids a second allowlist.
    func launchFlags(forAlias alias: String) -> [String] {
        var flags = launchFlags
        let normalizedAlias = alias.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalizedAlias.isEmpty, let speculativePreset else { return flags }
        let payload: String
        switch speculativePreset.method {
        case .mtp:
            guard let model = speculativePreset.model,
                  let tokens = speculativePreset.tokens else { return flags }
            payload = #"{"method":"mtp","model":"\#(model)","num_speculative_tokens":\#(tokens)}"#
        case .suffix:
            payload = #"{"method":"suffix"}"#
        }
        flags.append(contentsOf: ["--speculative-config", payload])
        return flags
    }

    /// Decode the audited subset from an already-resolved argv fragment.
    /// This is how residency receives the same measured recommendation +
    /// user override that a cold ``serve`` spawn receives.
    init(launchFlags: [String]) {
        self.init()
        var index = 0
        while index < launchFlags.count {
            let flag = launchFlags[index]
            let value = index + 1 < launchFlags.count ? launchFlags[index + 1] : nil
            switch flag {
            case "--kv-cache-dtype":
                if let value { kvCacheMode = KVCacheMode(rawValue: value) }
                index += 2
            case "--kv-cache-turboquant":
                if value == "v4" { kvCacheMode = .turboquantV4 }
                if value == "k8v4" { kvCacheMode = .turboquantK8V4 }
                index += 2
            case "--enable-prefix-cache":
                prefixCacheEnabled = true
                index += 1
            case "--disable-prefix-cache":
                prefixCacheEnabled = false
                index += 1
            case "--cache-memory-mb":
                if let value, let parsed = Int(value) { cacheMemoryMB = parsed }
                index += 2
            case "--speculative-config":
                if let value,
                   let data = value.data(using: .utf8),
                   let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let method = payload["method"] as? String {
                    if method == "mtp", let model = payload["model"] as? String,
                       let tokens = payload["num_speculative_tokens"] as? Int {
                        speculativePreset = SpeculativeDecodingPreset(
                            method: .mtp, model: model, tokens: tokens
                        )
                    } else if method == "suffix" {
                        speculativePreset = SpeculativeDecodingPreset(
                            method: .suffix, model: nil, tokens: nil
                        )
                    }
                }
                index += 2
            default:
                index += 1
            }
        }
    }

    /// Flag names this config can emit. Used by ``ServerManager`` to drop the
    /// RAM-tier recommendation's value for a knob the user has an explicit
    /// opinion about, so the two never both land on argv.
    static let ownedFlagNames: Set<String> = [
        "--kv-cache-dtype",
        "--kv-cache-turboquant",
        "--enable-prefix-cache",
        "--disable-prefix-cache",
        "--cache-memory-mb",
        "--speculative-config",
    ]
}

/// The KV-cache precision options the audit cleared for exposure.
///
/// Values map onto `--kv-cache-dtype` (bf16 / int8 / int4) and
/// `--kv-cache-turboquant` (v4 / k8v4). The legacy `--kv-cache-quantization`
/// spelling is deliberately absent: the engine documents it as "[deprecated
/// alias of --kv-cache-dtype int8]", and it is the flag that trips the
/// hard mutual-exclusion error against TurboQuant.
enum KVCacheMode: String, Codable, CaseIterable, Sendable {
    case bf16
    case int8
    case int4
    case turboquantV4 = "turboquant-v4"
    case turboquantK8V4 = "turboquant-k8v4"

    var launchFlags: [String] {
        switch self {
        case .bf16: return ["--kv-cache-dtype", "bf16"]
        case .int8: return ["--kv-cache-dtype", "int8"]
        case .int4: return ["--kv-cache-dtype", "int4"]
        case .turboquantV4: return ["--kv-cache-turboquant", "v4"]
        case .turboquantK8V4: return ["--kv-cache-turboquant", "k8v4"]
        }
    }

    /// Short label for the picker.
    var title: String {
        switch self {
        case .bf16: return "Full precision (bf16)"
        case .int8: return "8-bit"
        case .int4: return "4-bit"
        case .turboquantV4: return "TurboQuant V4"
        case .turboquantK8V4: return "TurboQuant K8V4"
        }
    }

    /// The issue requires each control to "state the trade-off in one line
    /// each, next to the control — what it buys, what it costs, and whether it
    /// can change output". These are those lines; they live next to the enum
    /// so a new case cannot be added without one.
    var tradeOff: String {
        switch self {
        case .bf16:
            return "Most memory, no quality loss. Slowest on long contexts."
        case .int8:
            return "Half the KV memory. Safe for hard math and reasoning."
        case .int4:
            return "Quarter the KV memory, fastest long-context decode. Can change output on AIME-class math."
        case .turboquantV4:
            return "Compresses the value cache only. Experimental; can change output."
        case .turboquantK8V4:
            return "~4.6× KV compression on dense models. Experimental; can change output."
        }
    }

    /// Whether picking this can change what the model writes, as opposed to
    /// only how fast it writes it. Drives the warning affordance in the panel.
    var canChangeOutput: Bool {
        switch self {
        case .bf16, .int8: return false
        case .int4, .turboquantV4, .turboquantK8V4: return true
        }
    }

    /// Sliding-window (Gemma 3/4, GPT-OSS) and MLA (DeepSeek V3+, Kimi K2.5)
    /// families auto-downgrade to bf16 inside the engine regardless of this
    /// setting. The panel uses this to say so up front rather than let the
    /// user believe a setting took that the engine overrode.
    var isSubjectToArchitectureDowngrade: Bool {
        self != .bf16
    }
}
