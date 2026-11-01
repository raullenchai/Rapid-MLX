import Foundation

extension ModelDisplayName {
    nonisolated static func name(for alias: String) -> String {
        if let choice = QuickstartCoordinator.onboardingChoices.first(where: { $0.alias == alias }) {
            return choice.displayName
        }
        let imageNames = [
            "flux2-klein-4b": "FLUX.2 Klein · 4B",
            "flux2-klein-4b-bf16": "FLUX.2 Klein · 4B · BF16",
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
        if let name = imageNames[alias.lowercased()] { return name }
        let speech = AudioViewModel.transcriptionDetails(alias: alias, family: nil).displayName
        if speech != alias { return speech }
        let family = ModelInfoCatalog.familyAndContext(for: alias).family
        guard family != "Unknown", let params = ModelSizing.parseParamsBillions(alias) else { return alias }
        let number = params.rounded() == params ? String(format: "%.0f", params) : String(format: "%g", params)
        let bits = ModelSizing.parseBitsPerWeight(alias)
        let tokens = alias.lowercased().split(whereSeparator: { $0 == "-" || $0 == "_" }).map(String.init)
        let familyEnd = (1...tokens.count).first { end in
            ModelInfoCatalog.familyAndContext(for: tokens.prefix(end).joined(separator: "-")).family == family
        } ?? 0
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
                variants.append(token.uppercased().hasPrefix("A") && Double(token.dropFirst().dropLast()) != nil && token.hasSuffix("b") ? token.uppercased() : (variantSpellings[token] ?? token.capitalized))
            }
        }
        let variant = variants.isEmpty ? "" : " · " + variants.joined(separator: " ")
        return "\(family) · \(number)B" + variant + (bits == 4 ? "" : " · \(bits)-bit")
    }
}
