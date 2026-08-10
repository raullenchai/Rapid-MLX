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

    /// The point of the issue: a model already on disk must be visible, or
    /// the user re-downloads weights they have. An earlier draft of this fix
    /// dropped external rows entirely — that satisfied "not deletable" by
    /// making them invisible, which is the bug, not the fix.
    @Test("An (external) row reaches the catalog so the user can see and use it")
    func externalRowBecomesAVisibleEntry() {
        let entries = ModelCatalog.mergeAvailableAndCached(
            available: [],
            cached: ModelCatalog.parseCached(Self.listing),
            excluded: []
        )

        let outsider = entries.first { $0.hfRepo == "mlx-community/Outsider-4bit" }
        #expect(outsider != nil, "an on-disk model must not be hidden")
        #expect(outsider?.cached == true, "it is on disk — no re-download prompt")
        #expect(outsider?.isExternal == true, "and it is flagged read-only")
        // The repo is the identifier: ``(external)`` is a status marker, not
        // a name, and the repo is what ``serve`` accepts.
        #expect(outsider?.alias == "mlx-community/Outsider-4bit")
    }

    @Test("External copies merge into an existing catalog alias")
    func externalCopyMarksKnownAliasCached() {
        let cached = [
            ("(external)", "mlx-community/Qwen3.5-4B", "2.3 GiB")
        ]
        let entries = ModelCatalog.mergeAvailableAndCached(
            available: [("qwen3.5-4b-4bit", "mlx-community/Qwen3.5-4B")],
            cached: cached,
            excluded: []
        )

        #expect(entries.count == 1)
        #expect(entries[0].alias == "qwen3.5-4b-4bit")
        #expect(entries[0].cached)
        #expect(entries[0].isExternal)
        #expect(entries[0].sizeOnDisk == "2.3 GiB")
    }

    @Test("A root-level external model matching an alias is not dropped")
    func rootLevelExternalMatchesAlias() {
        let entries = ModelCatalog.mergeAvailableAndCached(
            available: [("local-model", nil)],
            cached: [("(external)", "local-model", "1.0 GiB")],
            excluded: []
        )

        #expect(entries.count == 1)
        #expect(entries[0].alias == "local-model")
        #expect(entries[0].cached)
        #expect(entries[0].isExternal)
    }

    @Test("An excluded external identifier cannot re-enter the chat catalog")
    func excludedExternalStaysExcluded() {
        let entries = ModelCatalog.mergeAvailableAndCached(
            available: [],
            cached: [("(external)", "video-model", "1.0 GiB")],
            excluded: ["video-model"]
        )

        #expect(entries.isEmpty)
    }

    /// The safety half: visible, but never deletable.
    @Test("Deleting an external model is refused at the dispatcher")
    func externalDeletionIsRefused() async {
        let entry = ModelEntry(
            alias: "mlx-community/Outsider-4bit",
            hfRepo: "mlx-community/Outsider-4bit",
            sizeOnDisk: "1.1 GiB",
            cached: true,
            isExternal: true
        )

        let outcome = await ModelCacheActions.runDeletion(
            for: entry,
            binaryPath: URL(fileURLWithPath: "/bin/echo")
        )

        guard case .failure(let message) = outcome else {
            Issue.record("expected refusal, got \(outcome)")
            return
        }
        #expect(message.contains("another app"))
    }

    @Test("A normal cached entry is still deletable")
    func normalEntryIsNotRefused() async {
        let entry = ModelEntry(
            alias: "qwen3.5-4b-4bit",
            hfRepo: "mlx-community/Qwen3.5-4B",
            sizeOnDisk: "2.3 GiB",
            cached: true
        )

        let outcome = await ModelCacheActions.runDeletion(
            for: entry,
            binaryPath: URL(fileURLWithPath: "/nonexistent-binary")
        )

        // It fails (no real binary), but NOT with the external refusal —
        // the guard must not swallow ordinary deletes.
        guard case .failure(let message) = outcome else {
            Issue.record("expected the deliberately invalid binary to fail, got \(outcome)")
            return
        }
        #expect(!message.contains("another app"))
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

    @Test("Selected model root is merged with ambient roots and deduplicated")
    func rootsAreMerged() {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-external-model-root").path
        let canonical = URL(fileURLWithPath: root)
            .standardizedFileURL.resolvingSymlinksInPath().path
        let merged = ModelCatalog.mergedExtraModelRoots(
            existing: "/first:\(root)",
            selected: root + "/."
        )

        #expect(merged == "/first:\(canonical)")
    }

    @Test("Serve child receives the same external root used for discovery")
    func serveEnvironmentCarriesExternalRoot() {
        let env = ServerManager.serveEnvironmentAdditions(
            bearer: "test-token",
            ambient: [ModelCatalog.extraModelRootsEnvKey: "/ambient"],
            modelsFolderOverride: "/selected"
        )

        #expect(env[ModelCatalog.extraModelRootsEnvKey] == "/ambient:/selected")
        #expect(env["HF_HUB_CACHE"] == "/selected")
    }
}
