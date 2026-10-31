import Foundation

extension ModelDisplayName {
    @MainActor static func name(for alias: String) -> String {
        if let choice = QuickstartCoordinator.onboardingChoices.first(where: { $0.alias == alias }) {
            return choice.displayName
        }
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
        var variants: [String] = []
        for token in tokens.dropFirst(familyEnd) {
            if !consumedSize, token.last == "b", let value = Double(token.dropLast()), value == params {
                consumedSize = true
            } else if !consumedBits, ["2bit", "3bit", "4bit", "6bit", "8bit", "16bit", "bf16", "fp16"].contains(token),
                      ModelSizing.parseBitsPerWeight(token) == bits {
                consumedBits = true
            } else {
                variants.append(token.uppercased().hasPrefix("A") && Double(token.dropFirst().dropLast()) != nil && token.hasSuffix("b") ? token.uppercased() : token.capitalized)
            }
        }
        let variant = variants.isEmpty ? "" : " · " + variants.joined(separator: " ")
        return "\(family) · \(number)B" + variant + (bits == 4 ? "" : " · \(bits)-bit")
    }
}
