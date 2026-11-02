import Foundation

extension ModelDisplayName {
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

    nonisolated static func name(for alias: String) -> String {
        if let choice = QuickstartCoordinator.onboardingChoices.first(where: { $0.alias == alias }) {
            return choice.displayName
        }
        if let name = imageNames[alias.lowercased()] { return name }
        let speech = AudioViewModel.transcriptionDetails(alias: alias, family: nil).displayName
        if speech != alias { return speech }
        let family = ModelInfoCatalog.familyAndContext(for: alias).family
        guard family != "Unknown" else { return alias }
        let params = ModelSizing.parseParamsBillions(alias)
        let bits = ModelSizing.parseBitsPerWeight(alias)
        let tokens = alias.lowercased().split(whereSeparator: { $0 == "-" || $0 == "_" }).map(String.init)
        let familyEnd = (1...tokens.count).first { end in
            ModelInfoCatalog.familyAndContext(for: tokens.prefix(end).joined(separator: "-")).family == family
        } ?? 0
        if params == nil {
            // Family lookup accepts substrings; do not swallow an unrelated
            // leading token, or relabel speech aliases as text model families.
            let familyWithoutFirst = tokens.prefix(familyEnd).dropFirst().joined(separator: "-")
            let audioTokens: Set<String> = ["tts", "stt", "asr", "speech", "voice", "audio"]
            guard familyEnd > 0,
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
}
