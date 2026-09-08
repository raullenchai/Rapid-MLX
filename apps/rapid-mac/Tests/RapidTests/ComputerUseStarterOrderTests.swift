import Foundation
import Testing
@testable import Rapid

/// The Computer Use "Start with a flow" grid leads with the flow a user can
/// actually run. That ordering is product behavior, not view chrome, so it is
/// asserted against the catalog model rather than a rendered grid (#3254).
@Suite("Computer Use starter order")
struct ComputerUseStarterOrderTests {
    @Test("available starters come first")
    func availableStartersLead() {
        let ranks = ComputerUseStarter.ordered.map { starter -> Int in
            switch starter.availability {
            case .available: 0
            case .comingSoon: 1
            case .reserved: 2
            }
        }
        // Non-decreasing: every available starter precedes every coming-soon
        // one, which precedes every reserved one.
        #expect(ranks == ranks.sorted())
        #expect(ComputerUseStarter.ordered.first?.availability == .available)
    }

    @Test("ordering preserves every catalog entry exactly once")
    func orderingDropsNothing() {
        let orderedKinds = Set(ComputerUseStarter.ordered.map(\.kind))
        let catalogKinds = Set(ComputerUseStarter.catalog.map(\.kind))
        #expect(ComputerUseStarter.ordered.count == ComputerUseStarter.catalog.count)
        #expect(orderedKinds == catalogKinds)
    }

    @Test("ordering is stable within every availability tier")
    func orderingIsStableWithinEveryTier() {
        // Within a tier, entries keep their catalog order. Verify the ordered
        // subsequence equals the catalog subsequence for EACH tier, so a
        // reversal in any tier — not just the leading one — fails the test.
        for availability: ComputerUseStarter.Availability in [.available, .comingSoon, .reserved] {
            let catalogTier = ComputerUseStarter.catalog
                .filter { $0.availability == availability }
                .map(\.kind)
            let orderedTier = ComputerUseStarter.ordered
                .filter { $0.availability == availability }
                .map(\.kind)
            #expect(catalogTier == orderedTier, "tier \(availability) reordered")
        }
    }
}
