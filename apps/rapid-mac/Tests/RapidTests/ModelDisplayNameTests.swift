import Foundation
import Testing
@testable import Rapid

@Suite("Model display names")
struct ModelDisplayNameTests {
    // Pin the existing formatter before adding the unknown-family fallback.
    @Test(arguments: [
        ("qwen3.5-4b-4bit", "Qwen 3.5 · 4B"),
        ("qwen3.5-9b-8bit", "Qwen 3.5 · 9B · 8-bit"),
        ("qwen3-4b-thinking-2507-4bit", "Qwen 3 · 4B · Thinking 2507"),
        ("qwen3.8-27b-tensorfold", "Qwen 3.8 · 27B · TensorFold"),
        ("llama-3.1-8b-8bit", "Llama 3.1 · 8B · 8-bit"),
        ("gemma3-12b-4bit", "Gemma 3 · 12B"),
        ("phi-4-mini-4bit", "Phi 4 · Mini"),
        ("mistral-24b-4bit", "Mistral · 24B"),
        ("deepseek-r1-8b-4bit", "DeepSeek · 8B · R1"),
        ("bonsai-1.7b-2bit", "Bonsai · 1.7B · 2-bit"),
        ("lfm2.5-1b-4bit", "LFM2.5 · 1.2B"),
        ("ornith-1.5-9b-bf16", "Ornith 1.5 · 9B · 16-bit"),
        ("flux-schnell", "FLUX.1 Schnell"),
        ("whisper-large-v3-turbo", "Whisper Large v3 Turbo")
    ])
    func existingNames(alias: String, expected: String) {
        #expect(ModelDisplayName.name(for: alias) == expected)
    }

    @Test(arguments: [
        ("ui-tars-1.5-7b-4bit", "UI-TARS 1.5 · 7B"),
        ("ui-tars-7b-sft-4bit", "UI-TARS · 7B · SFT"),
        ("granite-4.2-3b-4bit", "Granite 4.2 · 3B"),
        ("nemotron-3.5-bolt-e4b-8bit", "Nemotron 3.5 Bolt E4B · 8-bit"),
        ("nemotron-labs-diffusion-3b-4bit", "Nemotron Labs Diffusion · 3B"),
        ("tmax-9b", "TMAX · 9B"),
        ("minicpm5-1b-4bit", "MiniCPM 5 · 1B"),
        ("minicpm-v4.7-1b-8bit", "MiniCPM-V 4.7 · 1B · 8-bit"),
        ("muse-glimmer-30b-4bit", "Muse Glimmer · 30B"),
        ("vibethinker-1.5b-4bit", "VibeThinker · 1.5B"),
        ("nanbeige4.1-3b-4bit", "Nanbeige 4.1 · 3B"),
        ("qwopus-9b-4bit", "Qwopus · 9B"),
        ("diffusion-gemma-26b-4bit", "Diffusion Gemma · 26B"),
        ("holo3.1-35b-a3b", "Holo 3.1 · 35B · A3B"),
        ("glm5.3-flash-4bit", "GLM 5.3 Flash"),
        ("minimax-m2.5-4bit", "MiniMax M2.5"),
        ("spark-x2.6-4b-4bit", "Spark X2.6 · 4B"),
        ("g9v3-39a5b-4bit", "G9V3 39A5B"),
        ("lfm2-24b-a2b-4bit", "LFM 2 · 24B · A2B"),
        ("flint-40b-a2.5b-4bit", "Flint · 40B · A2.5B"),
        ("hy3-preview-4bit", "HY3 Preview"),
        ("kimi-k2.6", "Kimi K2.6"),
        ("ling-3.0-tiny-4bit", "Ling 3.0 Tiny"),
        ("mimo-v2.6-flash-4bit", "MiMo V2.6 Flash"),
        ("north-mini-code-4bit", "North Mini Code"),
        ("sage-a1.5-4bit", "Sage A1.5"),
        // Future families work without adding a catalog or spelling-table entry.
        ("newfamily2.1-7b-instruct-8bit", "Newfamily 2.1 · 7B · Instruct · 8-bit")
    ])
    func aliasFallback(alias: String, expected: String) {
        #expect(ModelDisplayName.name(for: alias) == expected)
        #expect(ModelDisplayName.name(for: alias.uppercased()) == expected)
    }

    @Test func bundledCatalogHasRecognizableNames() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("rapid_mlx/aliases.json")
        let catalog = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        #expect(catalog.count > 200) // Fail closed if the full shipped catalog stops loading.
        let genericWords: Set<String> = ["", "model", "unknown", "default", "llm"]
        for alias in catalog.keys.sorted() {
            let name = ModelDisplayName.name(for: alias)
            let family = ModelBrandStyle.displayFamily(forAlias: alias)
            #expect(!name.hasPrefix("Model ·"), "\(alias): \(name)")
            #expect(!genericWords.contains(name.lowercased()), "\(alias): \(name)")
            // Settings composes its size/precision subtitle from this family;
            // pickers and download titles use ModelDisplayName directly.
            #expect(!genericWords.contains(family.lowercased()), "\(alias): \(family)")
        }
    }

    @Test func establishedFamiliesRemainExact() {
        for (alias, expected) in [
            ("qwen3.5-4b-4bit", "Qwen 3.5"),
            ("gemma-4-e4b-8bit", "Gemma 4"),
            ("llama-3.1-8b-8bit", "Llama 3.1"),
            ("gpt-oss-20b-4bit", "GPT-OSS"),
            ("devstral-24b-4bit", "Devstral"),
            ("lfm3-1.7b-8bit", "LFM3")
        ] {
            #expect(ModelBrandStyle.displayFamily(forAlias: alias) == expected)
        }
        #expect(ModelInfoCatalog.familyAndContext(for: "granite-4.2-3b-4bit").contextWindow == nil)
        #expect(ModelDisplayName.name(for: "unknown-model") == "unknown-model")
        for alias in ["gpt-oss-20b", "gpt-oss-20b-4bit", "gpt-oss-120b", "gpt-oss-120b-4bit", "devstral-24b-4bit"] {
            #expect(ModelDisplayName.name(for: alias) == alias)
        }
        #expect(ModelBrandStyle.displayFamily(forAlias: "model-7b-4bit") != "Model")
    }
}
