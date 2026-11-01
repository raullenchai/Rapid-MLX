import Foundation
import Testing
@testable import Rapid

@Suite("Community Benchmark localized copy")
@MainActor
struct CommunityBenchmarkCopyTests {
    private let alias = "gemma-4-e4b-4bit"

    @Test(arguments: [(1, false), (3, false), (1, true), (3, true)])
    func strengthenCopy(count: Int, isAtLeast: Bool) {
        let scope = CommunityBenchmarkScope(
            modelAlias: alias, workload: .llm, protocolID: "rapid-text-speed",
            protocolVersion: 1, macProfile: CommunityMacProfile(chip: "Apple M3 Ultra", memoryGiB: 256)
        )
        let branch = CommunityContributionBranch.strengthen(observationCount: count, isAtLeast: isAtLeast)
        let name = ModelDisplayName.name(for: alias)
        let ready = CommunityBenchmarkCopy.readyInvitation(branch: branch, scope: scope)
        let publish = CommunityBenchmarkCopy.publishInvitation(branch: branch, scope: scope)
        for body in [ready.body, publish.body] {
            #expect(body.first?.isUppercase == true || body.first?.isNumber == true)
            if name != alias { #expect(!body.contains(alias)) }
            #expect(body.contains(String(count)))
            #expect(!body.contains("%"))
        }
        // The Run-tab sentence and the publish headline name the model.
        #expect(ready.body.contains(name))
        #expect(publish.headline.contains(name))
        if name != alias { #expect(!publish.headline.contains(alias)) }
        #expect(scope.scopeDescription.contains(name))
    }
}
