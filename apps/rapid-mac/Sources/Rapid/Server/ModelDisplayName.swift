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
        return "\(family) · \(number)B" + (bits == 4 ? "" : " · \(bits)-bit")
    }
}
