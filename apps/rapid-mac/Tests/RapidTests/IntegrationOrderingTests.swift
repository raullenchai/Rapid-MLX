import Testing
@testable import Rapid

/// The Launch page leads with the rows that work.
///
/// The sidecar assembles its list as "config writers first, then adapter
/// profiles" — an artefact of how the registry is built, not a judgement about
/// the rows. It put Cline first, a VS Code extension most users do not have,
/// and Codex sixth behind LangChain.
@Suite("Launch page integration ordering")
struct IntegrationOrderingTests {

    private func target(_ id: String, _ kind: IntegrationTarget.Kind) -> IntegrationTarget {
        IntegrationTarget(id: id, name: id, kind: kind, configPath: nil)
    }

    /// The registry order observed on a real install, trimmed to the head.
    private var registryOrder: [IntegrationTarget] {
        [
            target("cline", .configWriter),
            target("claude-code", .configWriter),
            target("continue-dev", .configWriter),
            target("cursor", .configWriter),
            target("langchain", .adapterProfile),
            target("codex", .adapterProfile),
            target("hermes", .adapterProfile),
        ]
    }

    @Test("Claude Code leads, Codex follows")
    func leadingRowsArePinned() {
        let ordered = IntegrationCatalog.displayOrdered(registryOrder).map(\.id)
        #expect(ordered.first == "claude-code")
        #expect(ordered.dropFirst().first == "codex")
    }

    /// Pinning must not become a reshuffle. Everything the pin does not name
    /// keeps the order the registry gave it — that order encodes the sidecar's
    /// own grouping, and rearranging it silently would be a second change
    /// nobody asked for.
    ///
    /// This does NOT prove the sort is stable: Swift's `sorted(by:)` happens
    /// to preserve input order for equal elements at these sizes even without
    /// an explicit tiebreaker (verified by removing ours — this test still
    /// passed). The tiebreaker in `displayOrdered` guards a documented
    /// non-guarantee, and no test can demonstrate that today.
    @Test("Unpinned rows keep the registry's order")
    func unpinnedRowsAreUndisturbed() {
        let ordered = IntegrationCatalog.displayOrdered(registryOrder).map(\.id)
        let rest = ordered.filter { !IntegrationCatalog.leadingIntegrations.contains($0) }
        #expect(rest == ["cline", "continue-dev", "cursor", "langchain", "hermes"])
    }

    @Test("A missing pinned row is skipped, not faked")
    func absentPinnedRowIsSkipped() {
        let withoutCodex = registryOrder.filter { $0.id != "codex" }
        let ordered = IntegrationCatalog.displayOrdered(withoutCodex).map(\.id)
        #expect(ordered.first == "claude-code")
        #expect(!ordered.contains("codex"))
        #expect(ordered.count == withoutCodex.count)
    }
}
