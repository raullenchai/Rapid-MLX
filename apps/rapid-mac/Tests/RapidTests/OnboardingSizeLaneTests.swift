import Testing
@testable import Rapid

@Suite("Onboarding size lane")
struct OnboardingSizeLaneTests {
    @Test("Short sizes stay in the trailing lane", arguments: ["3.06 GB", "16.32 GB · 92%"])
    func shortSizesAreCompact(sizeText: String) {
        #expect(OnboardingSizeLane.isCompact(sizeText))
    }

    @Test("Long MoE details wrap below the name")
    func longSizeIsNotCompact() {
        #expect(!OnboardingSizeLane.isCompact("20.43 GB · 87% capability · 35B total · 3B active"))
    }

    @Test("Empty sizes have no lane")
    func emptySizeIsNotCompact() {
        #expect(!OnboardingSizeLane.isCompact(""))
    }

    @Test("The trailing lane allows at most 18 characters")
    func compactLengthBoundary() {
        #expect(OnboardingSizeLane.isCompact(String(repeating: "1", count: 18)))
        #expect(!OnboardingSizeLane.isCompact(String(repeating: "1", count: 19)))
    }
}
