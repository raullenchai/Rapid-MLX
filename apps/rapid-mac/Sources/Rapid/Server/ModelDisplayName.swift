import Foundation

extension ModelDisplayName {
    /// Raw Hugging Face identities have exactly two components. Local paths,
    /// URLs and catalog aliases must keep their existing presentation.
    nonisolated static func repositoryName(for alias: String) -> String? {
        let parts = alias.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2, parts.allSatisfy({ !$0.isEmpty }),
              !alias.contains(where: { $0.isWhitespace }) else { return nil }
        return String(parts[1])
    }

    /// Compact labels omit the org unless another listed repository has the
    /// same name. Keep the full string here; the view owns tail truncation.
    nonisolated static func compactName(for alias: String, among aliases: [String] = []) -> String {
        // A curated name (speech, image, catalog) wins over the raw repo ID.
        let display = name(for: alias)
        guard display == alias, let shortName = repositoryName(for: alias) else { return display }
        let collides = aliases.contains { $0 != alias && repositoryName(for: $0) == shortName }
        return collides ? alias : shortName
    }

    nonisolated static let imageNames: [String: String] = [
        "flux2-klein-4b": "FLUX.2 Klein · 4B",
        "flux2-klein-4b-bf16": "FLUX.2 Klein · 4B · BF16",
        "bonsai-image-4b-2bit": "Bonsai Image · 4B · 2-bit",
        "flux-schnell": "FLUX.1 Schnell",
        "z-image-turbo": "Z-Image Turbo",
        "qwen-image": "Qwen Image",
        "qwen-image-2.1": "Qwen Image 2.1",
        "qwen-image-2.1-bf16": "Qwen Image 2.1 · BF16",
        "qwen-image-2.1-turbo": "Qwen Image 2.1 Turbo",
        "qwen-image-edit": "Qwen Image Edit",
        "hidream-o1-dev": "HiDream O1 Dev",
        "sdxl-base": "SDXL Base",
        "sd35-large-4bit": "Stable Diffusion 3.5 Large"
    ]

    // Keep explicit video spellings here, alongside image names, so every
    // surface shares family casing, workload, size, and precision labels.
    nonisolated static let videoNames: [String: String] = [
        "cogvideox-fun-5b-q4": "CogVideoX-Fun · 5B · 4-bit",
        "ltx-2.3-mlx-q4": "LTX 2.3 · 4-bit",
        "ltx-2.5-mlx-q8": "LTX 2.5 · 8-bit",
        "minimax-h3": "MiniMax H3",
        "pika-h3-flex": "Pika H3 Flex",
        "wan2.1-t2v-1.3b-bf16": "Wan 2.1 · T2V 1.3B · BF16",
        "wan2.2-i2v-a14b-q8": "Wan 2.2 · I2V A14B · 8-bit",
        "wan2.2-t2v-a14b-bf16": "Wan 2.2 · T2V A14B · BF16",
        "wan2.2-ti2v-5b-bf16": "Wan 2.2 · TI2V 5B · BF16",
        "wan2.2-ti2v-5b-q8": "Wan 2.2 · TI2V 5B · 8-bit"
    ]

    nonisolated static func name(for alias: String) -> String {
        if let choice = QuickstartCoordinator.onboardingChoices.first(where: { $0.alias == alias }) {
            return choice.displayName
        }
        if let name = imageNames[alias.lowercased()] { return name }
        if let name = videoNames[alias.lowercased()] { return name }
        let speech = AudioViewModel.transcriptionDetails(alias: alias, family: nil).displayName
        if speech != alias { return speech }
        let family = ModelInfoCatalog.familyAndContext(for: alias).family
        guard family != "Unknown" else {
            // The brand layer already names these families. Keep their existing
            // titles: bare GPT-OSS and explicit 4-bit aliases can be different
            // repositories, so removing their quantization tags loses identity.
            let displayFamily = ModelBrandStyle.displayFamily(forAlias: alias)
            if displayFamily == "GPT-OSS" || displayFamily == "Devstral" { return alias }
            // Preserve free-text titles. Machine aliases carry a version,
            // size or precision; their identity is readable even for new families.
            guard alias.rangeOfCharacter(from: .decimalDigits) != nil,
                  !alias.contains(where: { $0.isWhitespace || $0 == "/" }) else { return alias }
            let parts = aliasDisplayParts(alias)
            return ([parts.family] + parts.details).joined(separator: " · ")
        }
        let params = ModelSizing.parseParamsBillions(alias)
        let bits = ModelSizing.parseBitsPerWeight(alias)
        let tokens = alias.lowercased().split(whereSeparator: { $0 == "-" || $0 == "_" }).map(String.init)
        let familyEnd = (1...tokens.count).first { end in
            ModelInfoCatalog.familyAndContext(for: tokens.prefix(end).joined(separator: "-")).family == family
        } ?? 0
        if params == nil {
            // Only format machine aliases with explicit quantization. Family
            // lookup accepts substrings; preserve free-text names and speech aliases.
            let familyWithoutFirst = tokens.prefix(familyEnd).dropFirst().joined(separator: "-")
            let audioTokens: Set<String> = ["tts", "stt", "asr", "speech", "voice", "audio"]
            let quantizationTokens: Set<String> = ["2bit", "3bit", "4bit", "6bit", "8bit", "16bit", "bf16", "fp16"]
            guard alias.unicodeScalars.allSatisfy({
                      (97...122).contains($0.value) || (48...57).contains($0.value)
                          || $0 == "." || $0 == "-" || $0 == "_"
                  }),
                  tokens.contains(where: { quantizationTokens.contains($0) }),
                  familyEnd > 0,
                  ModelInfoCatalog.familyAndContext(for: familyWithoutFirst).family != family,
                  !tokens.dropFirst(familyEnd).contains(where: { audioTokens.contains($0) }) else {
                return alias
            }
        }
        var consumedSize = false
        var consumedBits = false
        let variantSpellings = ["tensorfold": "TensorFold"]
        var variants: [String] = []
        for token in tokens.dropFirst(familyEnd) {
            if !consumedSize, token.last == "b", let value = Double(token.dropLast()), value == params {
                consumedSize = true
            } else if !consumedBits, ["2bit", "3bit", "4bit", "6bit", "8bit", "16bit", "bf16", "fp16"].contains(token),
                      ModelSizing.parseBitsPerWeight(token) == bits {
                consumedBits = true
            } else {
                variants.append(token.first?.isLetter == true && Double(token.dropFirst().dropLast()) != nil && token.hasSuffix("b") ? token.uppercased() : (variantSpellings[token] ?? token.capitalized))
            }
        }
        let variant = variants.isEmpty ? "" : " · " + variants.joined(separator: " ")
        let size: String
        if let params {
            let number = params.rounded() == params ? String(format: "%.0f", params) : String(format: "%g", params)
            let redundant = variants.contains { token in
                token.range(of: #"(?<![0-9.])"# + NSRegularExpression.escapedPattern(for: number) + #"B(?![a-zA-Z0-9])"#,
                            options: [.regularExpression, .caseInsensitive]) != nil
            }
            size = redundant ? "" : " · \(number)B"
        } else {
            size = ""
        }
        return family + size + variant + (bits == 4 ? "" : " · \(bits)-bit")
    }

    /// Presentation only: do not add guessed families to the context-window table.
    /// Strip explicit size/precision tokens, retaining every version and variant.
    nonisolated static func aliasDisplayParts(_ alias: String) -> (family: String, details: [String]) {
        var tokens = alias.lowercased().split(whereSeparator: { $0 == "-" || $0 == "_" }).map(String.init)
        // These hyphens belong to the brand rather than separating words.
        if tokens.starts(with: ["ui", "tars"]) { tokens.replaceSubrange(0...1, with: ["ui-tars"]) }
        if tokens.count > 1, tokens[0] == "minicpm", tokens[1].hasPrefix("v"),
           Double(tokens[1].dropFirst()) != nil {
            tokens.replaceSubrange(0...1, with: ["minicpm-v", String(tokens[1].dropFirst())])
        }
        let precision = Set(["2bit", "3bit", "4bit", "6bit", "8bit", "16bit", "bf16", "fp16"])
        tokens.removeAll { precision.contains($0) }
        let sizeIndex = tokens.firstIndex { $0.last == "b" && Double($0.dropLast()) != nil }
        let familyTokens = sizeIndex.map { Array(tokens.prefix($0)) } ?? tokens
        let family = familyTokens.map(aliasWord).joined(separator: " ")
        var details: [String] = []
        if let sizeIndex {
            details.append(tokens[sizeIndex].uppercased())
            let variants = tokens.dropFirst(sizeIndex + 1).map(aliasWord)
            if !variants.isEmpty { details.append(variants.joined(separator: " ")) }
        }
        let bits = ModelSizing.parseBitsPerWeight(alias)
        if bits != 4 { details.append("\(bits)-bit") }
        return (family.isEmpty || family == "Model" ? alias : family, details)
    }

    /// Only spellings whose internal capitals cannot be recovered from lowercase.
    private nonisolated static func aliasWord(_ token: String) -> String {
        let spellings = [
            "ui-tars": "UI-TARS", "minicpm": "MiniCPM", "minicpm-v": "MiniCPM-V",
            "glm": "GLM", "lfm": "LFM", "tmax": "TMAX", "minimax": "MiniMax",
            "mimo": "MiMo", "vibethinker": "VibeThinker", "nanbeige": "Nanbeige",
            "sft": "SFT", "dpo": "DPO", "vl": "VL", "tensorfold": "TensorFold"
        ]
        if let spelling = spellings[token] { return spelling }
        // A name joined to a version (granite4, glm5.3) becomes two words;
        // short version/architecture markers (K2.6, A3B, X2.6) stay together.
        if let digit = token.firstIndex(where: { $0.isNumber }), digit != token.startIndex {
            let root = String(token[..<digit])
            if root.count <= 2 { return token.uppercased() }
            return (spellings[root] ?? root.capitalized) + " " + token[digit...].uppercased()
        }
        return token.capitalized
    }
}
