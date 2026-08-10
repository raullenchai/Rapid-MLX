import Foundation
import Testing
@testable import Rapid

/// Issue #1718 — models another MLX runtime downloaded.
///
/// The engine marks those rows `(external)` in the alias column. They are
/// listed and usable, but must never become a deletable `ModelEntry`: the
/// delete path rebuilds `<hub-root>/models--<repo>`, and an external model
/// does not live there. Offering the delete would either silently miss or
/// remove an unrelated hub entry that happens to share the name.
@Suite("External model rows (#1718)")
struct ExternalModelCatalogTests {

    private static let listing = """
      Cached models (3 on disk)
      ────────────────────────────────────────────────
      Alias                  HF repo                       Size      Modified
      ────────────────────────────────────────────────
      qwen3.5-4b-4bit        mlx-community/Qwen3.5-4B      2.3 GiB   3d ago
      (external)             mlx-community/Outsider-4bit   1.1 GiB   1d ago
      (incomplete)           mlx-community/Partial-4bit    61.0 MiB  2d ago
      """

    @Test("An (external) row survives parsing and keeps its repo")
    func externalRowIsParsed() {
        let rows = ModelCatalog.parseCached(Self.listing)
        let external = rows.first { $0.0 == "(external)" }

        #expect(external != nil, "the row must reach the app — it is a usable model")
        #expect(external?.1 == "mlx-community/Outsider-4bit")
    }

    @Test("An (incomplete) row is still dropped")
    func incompleteRowIsRejected() {
        let rows = ModelCatalog.parseCached(Self.listing)

        #expect(!rows.contains { $0.0 == "(incomplete)" })
    }

    /// The load-bearing test: an external model must not reach the picker as
    /// a cached entry, because cached entries are offered for deletion.
    @Test("An (external) row never becomes a deletable entry")
    func externalRowDoesNotBecomeDeletableEntry() {
        let entries = ModelCatalog.mergeAvailableAndCached(
            available: [],
            cached: ModelCatalog.parseCached(Self.listing),
            excluded: []
        )

        #expect(!entries.contains { $0.alias == "(external)" })
        #expect(!entries.contains { $0.hfRepo == "mlx-community/Outsider-4bit" })
    }

    @Test("A real alias in the same listing is still admitted")
    func realAliasStillWorks() {
        let entries = ModelCatalog.mergeAvailableAndCached(
            available: [],
            cached: ModelCatalog.parseCached(Self.listing),
            excluded: []
        )

        let qwen = entries.first { $0.alias == "qwen3.5-4b-4bit" }
        #expect(qwen?.cached == true)
        #expect(qwen?.hfRepo == "mlx-community/Qwen3.5-4B")
    }

    @Test("Status aliases are recognised by shape, not by an allow-list")
    func statusAliasDetection() {
        #expect(ModelCatalog.isStatusAlias("(external)"))
        #expect(ModelCatalog.isStatusAlias("(unmapped)"))
        #expect(ModelCatalog.isStatusAlias("(incomplete)"))
        #expect(!ModelCatalog.isStatusAlias("qwen3.5-4b-4bit"))
        // A future engine status must be excluded by default rather than
        // silently admitted as a deletable alias.
        #expect(ModelCatalog.isStatusAlias("(whatever-comes-next)"))
    }

    @Test("The extra-roots env key matches the engine's contract")
    func envKeyMatchesEngine() {
        // Cross-process contract with vllm_mlx.cli._external_model_roots.
        // A typo fails silently as "no models found".
        #expect(ModelCatalog.extraModelRootsEnvKey == "RAPID_MLX_EXTRA_MODEL_ROOTS")
    }
}
