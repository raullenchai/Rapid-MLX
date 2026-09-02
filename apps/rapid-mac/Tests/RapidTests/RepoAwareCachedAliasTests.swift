import Foundation
import Testing
@testable import Rapid

/// #576 — a downloaded model served by its bare (default-quant) alias
/// (`qwen3-0.6b`) must not be re-detected as uncached just because
/// `rapid-mlx ls` reports the quant-suffixed sibling (`qwen3-0.6b-4bit`).
/// Exercises the three pure helpers that back the repo-aware
/// reconciliation in ``ModelCatalog.load`` without standing up a sidecar.
@Suite("Repo-aware cached-alias reconciliation (#576)")
struct RepoAwareCachedAliasTests {

    // MARK: parseInfoRepo

    @Test("parseInfoRepo extracts the HF repo from the Alias → repo line")
    func parseInfoRepoUnicodeArrow() {
        let out = """
          Alias: qwen3-0.6b → mlx-community/Qwen3-0.6B-4bit

        ┌───────────────────────────────────────┐
        │ Model: mlx-community/Qwen3-0.6B-4bit   │
        """
        #expect(ModelCatalog.parseInfoRepo(out) == "mlx-community/Qwen3-0.6B-4bit")
    }

    @Test("parseInfoRepo accepts an ASCII arrow too")
    func parseInfoRepoAsciiArrow() {
        #expect(
            ModelCatalog.parseInfoRepo("  Alias: qwen3-0.6b -> mlx-community/Qwen3-0.6B-4bit")
                == "mlx-community/Qwen3-0.6B-4bit"
        )
    }

    @Test("parseInfoRepo returns nil without an Alias line")
    func parseInfoRepoNoAliasLine() {
        #expect(ModelCatalog.parseInfoRepo("Model: mlx-community/Qwen3-0.6B-4bit") == nil)
        #expect(ModelCatalog.parseInfoRepo("") == nil)
    }

    @Test("parseInfoRepo rejects an unsafe repo after the arrow")
    func parseInfoRepoRejectsUnsafeRepo() {
        #expect(ModelCatalog.parseInfoRepo("Alias: x → https://evil.test/model?x=1") == nil)
        #expect(ModelCatalog.parseInfoRepo("Alias: x → a/b/c") == nil)  // 3 path parts
    }

    // MARK: siblingCandidateAliases

    @Test("siblingCandidateAliases picks the bare base of a cached quant alias")
    func candidatesBareBase() {
        let entries = [
            ModelEntry(alias: "qwen3-0.6b", hfRepo: nil, sizeOnDisk: nil, cached: false),
            ModelEntry(
                alias: "qwen3-0.6b-4bit",
                hfRepo: "mlx-community/Qwen3-0.6B-4bit",
                sizeOnDisk: "351 MB",
                cached: true
            ),
            // A different quant of the same family that is NOT cached must
            // not become a candidate (the cached -4bit alias is not a
            // "qwen3-0.6b-8bit-" prefix).
            ModelEntry(alias: "qwen3-0.6b-8bit", hfRepo: nil, sizeOnDisk: nil, cached: false),
        ]
        #expect(ModelCatalog.siblingCandidateAliases(entries) == ["qwen3-0.6b"])
    }

    @Test("siblingCandidateAliases returns empty when nothing is cached")
    func candidatesEmptyWhenNoCache() {
        let entries = [
            ModelEntry(alias: "qwen3-0.6b", hfRepo: nil, sizeOnDisk: nil, cached: false),
            ModelEntry(alias: "qwen3-0.6b-4bit", hfRepo: nil, sizeOnDisk: nil, cached: false),
        ]
        #expect(ModelCatalog.siblingCandidateAliases(entries).isEmpty)
    }

    // MARK: remarkCachedByRepo

    @Test("remarkCachedByRepo marks the bare alias cached when repos match")
    func remarkMatchesRepo() {
        let entries = [
            ModelEntry(alias: "qwen3-0.6b", hfRepo: nil, sizeOnDisk: nil, cached: false),
            ModelEntry(
                alias: "qwen3-0.6b-4bit",
                hfRepo: "mlx-community/Qwen3-0.6B-4bit",
                sizeOnDisk: "351 MB",
                cached: true
            ),
        ]
        let out = ModelCatalog.remarkCachedByRepo(
            entries,
            resolvedRepos: ["qwen3-0.6b": "mlx-community/Qwen3-0.6B-4bit"]
        )
        let bare = out.first { $0.alias == "qwen3-0.6b" }
        #expect(bare?.cached == true)
        #expect(bare?.hfRepo == "mlx-community/Qwen3-0.6B-4bit")
        #expect(bare?.sizeOnDisk == "351 MB")  // carried from the cached sibling
    }

    @Test("remarkCachedByRepo leaves the alias uncached when the resolved repo differs")
    func remarkNoFalsePositiveOnDifferentQuant() {
        // Only the 8-bit repo is on disk; the bare alias defaults to 4-bit,
        // so it must stay uncached — the base-prefix heuristic narrows the
        // probe set but the repo equality check is authoritative.
        let entries = [
            ModelEntry(alias: "qwen3-0.6b", hfRepo: nil, sizeOnDisk: nil, cached: false),
            ModelEntry(
                alias: "qwen3-0.6b-8bit",
                hfRepo: "mlx-community/Qwen3-0.6B-8bit",
                sizeOnDisk: "700 MB",
                cached: true
            ),
        ]
        let out = ModelCatalog.remarkCachedByRepo(
            entries,
            resolvedRepos: ["qwen3-0.6b": "mlx-community/Qwen3-0.6B-4bit"]
        )
        #expect(out.first { $0.alias == "qwen3-0.6b" }?.cached == false)
    }

    @Test("structured subfolder cache never proves the repository root")
    func remarkRequiresExactSubfolderIdentity() {
        let merged = ModelCatalog.mergeAvailableAndCached(
            available: [("foo", "org/foo"), ("foo-4bit", "org/foo")],
            cached: [("foo-4bit", "org/foo", "4bit", "1 GiB")],
            excluded: []
        )
        #expect(merged.first { $0.alias == "foo-4bit" }?.sourceSubfolder == "4bit")

        let out = ModelCatalog.remarkCachedByRepo(
            merged, resolvedRepos: ["foo": "org/foo"]
        )
        #expect(out.first { $0.alias == "foo" }?.cached == false)
        #expect(out.first { $0.alias == "foo-4bit" }?.cached == true)
    }

    @Test("remarkCachedByRepo never merges repos that differ only by case")
    func remarkExactCaseMatch() {
        let entries = [
            ModelEntry(alias: "qwen3-0.6b", hfRepo: nil, sizeOnDisk: nil, cached: false),
            ModelEntry(
                alias: "qwen3-0.6b-4bit",
                hfRepo: "mlx-community/Qwen3-0.6B-4bit",
                sizeOnDisk: "351 MB",
                cached: true
            ),
        ]
        let out = ModelCatalog.remarkCachedByRepo(
            entries,
            resolvedRepos: ["qwen3-0.6b": "mlx-community/qwen3-0.6b-4bit"]
        )
        #expect(out.first { $0.alias == "qwen3-0.6b" }?.cached == false)
    }
}
