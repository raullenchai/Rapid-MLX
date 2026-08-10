import Foundation
import Testing
@testable import Rapid

/// Pin the ``ModelDeletion.parseFreedBytes`` contract — the regex
/// that pulls a bytes-freed estimate out of the CLI's stdout. The UI
/// uses this to render a "Freed 3.1 GB" toast; if the CLI ever
/// changes its output format and we silently regress to ``nil``,
/// the toast degrades from helpful to vague.
///
/// We do NOT exercise the subprocess path — that requires a real
/// ``rapid-mlx`` binary, the right HF cache layout, and a model the
/// test is allowed to delete. The smoke for that lives in the
/// ``TestDriver`` (out-of-band; cost real disk).
@Suite("ModelDeletion — bytes-freed parser")
struct ModelDeletionParseTests {
    @Test("Gigabyte suffix — the common case")
    func gigabyteSize() {
        let raw = """
          Alias: qwen3.5-4b-4bit → mlx-community/Qwen3.5-4B-MLX-4bit

          Removing mlx-community/Qwen3.5-4B-MLX-4bit (3.1G) ...
          Done.
        """
        let parsed = ModelDeletion.parseFreedBytes(stdout: raw)
        // 3.1 * 1024^3 ≈ 3,328,599,818 — assert order-of-magnitude
        // rather than exact int (Double rounding).
        #expect(parsed != nil)
        #expect((parsed ?? 0) > 3_300_000_000 && (parsed ?? 0) < 3_400_000_000)
    }

    @Test("Megabyte suffix")
    func megabyteSize() {
        let raw = "Removing org/small-model (640M) ..."
        let parsed = ModelDeletion.parseFreedBytes(stdout: raw)
        #expect(parsed != nil)
        // 640 * 1024 * 1024 = 671,088,640
        #expect(parsed == 671_088_640)
    }

    @Test("Decimal terabyte (unrealistic today but the regex must accept the unit)")
    func terabyteSize() {
        let raw = "Removing org/huge-model (1.5T) ..."
        let parsed = ModelDeletion.parseFreedBytes(stdout: raw)
        #expect(parsed != nil)
        let tb: Double = 1024 * 1024 * 1024 * 1024
        let expected = Int64(1.5 * tb)
        #expect(parsed == expected)
    }

    @Test("Missing parentheses → nil")
    func absentSize() {
        let raw = """
        Removing org/foo ...
        Done.
        """
        #expect(ModelDeletion.parseFreedBytes(stdout: raw) == nil)
    }

    @Test("Garbage payload → nil")
    func malformed() {
        #expect(ModelDeletion.parseFreedBytes(stdout: "(abc)") == nil)
        #expect(ModelDeletion.parseFreedBytes(stdout: "") == nil)
    }

    @Test("'GiB' suffix variant — the regex must tolerate the i + B")
    func gibibyteSuffix() {
        // Some HF tools render sizes as ``GiB`` instead of ``G``.
        // We accept either so the toast stays informative if the
        // CLI ever swaps style.
        let raw = "Removing org/model (2.0GiB) ..."
        #expect(ModelDeletion.parseFreedBytes(stdout: raw) != nil)
    }
}

@Suite("ModelDeletion — cache boundary")
struct ModelDeletionBoundaryTests {
    @Test("HF repo maps to the expected hub cache directory name")
    func repoDirectoryName() {
        #expect(
            ModelDeletion._testingCacheDirectoryName(forRepo: "mlx-community/Qwen3.5-4B-MLX-4bit")
            == "models--mlx-community--Qwen3.5-4B-MLX-4bit"
        )
        #expect(ModelDeletion._testingCacheDirectoryName(forRepo: "mlx-community/../escape") == nil)
        #expect(ModelDeletion._testingCacheDirectoryName(forRepo: "mlx-community/evil..name") == nil)
    }

    @Test("Default cache root honors HF_HUB_CACHE and rejects relative env paths")
    func defaultCacheRoot() throws {
        let tmp = try makeTmpDir()
        let hub = tmp.appendingPathComponent("custom-hub", isDirectory: true)
        let resolved = ModelDeletion._testingDefaultHubCacheRoot(environment: [
            "HF_HUB_CACHE": hub.path,
            "HOME": tmp.path,
        ])
        #expect(resolved?.path == hub.path)

        #expect(ModelDeletion._testingDefaultHubCacheRoot(environment: [
            "HF_HUB_CACHE": "../escape",
            "HOME": tmp.path,
        ]) == nil)
    }

    @Test("Validated target must canonicalize inside the hub root")
    func validatedTargetStaysInsideRoot() throws {
        let tmp = try makeTmpDir()
        defer { try? FileManager.default.removeItem(at: tmp) }
        let hub = tmp.appendingPathComponent("hf", isDirectory: true)
            .appendingPathComponent("hub", isDirectory: true)
        try FileManager.default.createDirectory(at: hub, withIntermediateDirectories: true)

        let target = hub.appendingPathComponent("models--mlx-community--Qwen3", isDirectory: true)
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        #expect(ModelDeletion._testingValidatedDeletionURL(target, hubCacheRoot: hub)?.path == target.path)

        let lexicalEscape = URL(fileURLWithPath: hub.path + "/../outside/models--escape", isDirectory: true)
        #expect(ModelDeletion._testingValidatedDeletionURL(lexicalEscape, hubCacheRoot: hub) == nil)
    }

    /// Pins the explicit ``+ "/"`` trick in ``validatedDeletionURL``.
    /// Without the trailing slash, ``hasPrefix(rootCanonical)`` would
    /// accept a sibling directory whose name happens to start with
    /// the hub root's last path component — e.g. ``/cache/hub-evil/x``
    /// shares a string prefix with ``/cache/hub`` but is a totally
    /// different directory tree. README "Model picker" section "audit
    /// batch 10" claims "hard-prefix check" precisely; if a refactor
    /// drops the slash suffix the gate silently regresses to substring
    /// matching.
    @Test("Sibling root with shared name prefix is rejected (the '/' in rootPrefix matters)")
    func siblingPrefixRoot() throws {
        let tmp = try makeTmpDir()
        defer { try? FileManager.default.removeItem(at: tmp) }
        let hub = tmp.appendingPathComponent("hf", isDirectory: true)
            .appendingPathComponent("hub", isDirectory: true)
        let evilSibling = tmp.appendingPathComponent("hf", isDirectory: true)
            .appendingPathComponent("hub-evil", isDirectory: true)
        try FileManager.default.createDirectory(at: hub, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: evilSibling, withIntermediateDirectories: true)

        // A real directory living under hub-evil — not under hub —
        // whose path string nevertheless shares hub's prefix bytes.
        let candidate = evilSibling.appendingPathComponent("models--mlx-community--Sneak", isDirectory: true)
        try FileManager.default.createDirectory(at: candidate, withIntermediateDirectories: true)

        #expect(ModelDeletion._testingValidatedDeletionURL(candidate, hubCacheRoot: hub) == nil)
        // And the sibling tree must still be on disk after the rejection.
        #expect(FileManager.default.fileExists(atPath: candidate.path))
    }

    /// Pins the ``candidateCanonical != rootCanonical`` guard. The hub
    /// root itself is a real directory; without this check, a caller
    /// that fed the root as the candidate would canonically prefix-match
    /// (``"/cache/hub"`` is a prefix of ``"/cache/hub/"``) and we'd be
    /// asked to ``removeItem(at:)`` the hub cache root, wiping every
    /// model in one click. README "Model picker" section "no
    /// rm-rf-on-edge-case".
    @Test("Candidate that canonicalizes to the hub root itself is rejected")
    func candidateEqualToRootRejected() throws {
        let tmp = try makeTmpDir()
        defer { try? FileManager.default.removeItem(at: tmp) }
        let hub = tmp.appendingPathComponent("hf", isDirectory: true)
            .appendingPathComponent("hub", isDirectory: true)
        try FileManager.default.createDirectory(at: hub, withIntermediateDirectories: true)

        #expect(ModelDeletion._testingValidatedDeletionURL(hub, hubCacheRoot: hub) == nil)
        #expect(FileManager.default.fileExists(atPath: hub.path))
    }

    /// Pins the existence guard. A candidate URL that doesn't resolve
    /// to a real directory on disk must reject — otherwise an
    /// alias-to-repo mapping that drifts (CLI renames a model, cache
    /// dir lags) could end up pointing at a stale path that doesn't
    /// exist and silently no-op without surfacing the drift. The
    /// validator should refuse instead so the UI shows a real failure
    /// toast.
    @Test("Candidate that doesn't exist on disk is rejected")
    func nonExistentCandidateRejected() throws {
        let tmp = try makeTmpDir()
        defer { try? FileManager.default.removeItem(at: tmp) }
        let hub = tmp.appendingPathComponent("hf", isDirectory: true)
            .appendingPathComponent("hub", isDirectory: true)
        try FileManager.default.createDirectory(at: hub, withIntermediateDirectories: true)

        let ghost = hub.appendingPathComponent("models--never--existed", isDirectory: true)
        #expect(FileManager.default.fileExists(atPath: ghost.path) == false)
        #expect(ModelDeletion._testingValidatedDeletionURL(ghost, hubCacheRoot: hub) == nil)
    }

    /// Pins the ``isDirectory`` half of the existence guard explicitly.
    /// A regression that drops the ``isDir.boolValue`` check would let
    /// a regular file living under the hub root canonicalize cleanly
    /// (it exists, the path stays under the prefix, it's not a
    /// symlink) — but ``removeItem(at:)`` against a file deletes the
    /// file. That's not the bounded-deletion contract the README
    /// promises: only HF cache *directories* should ever be removable.
    @Test("Candidate that exists as a regular file (not a directory) is rejected")
    func regularFileCandidateRejected() throws {
        let tmp = try makeTmpDir()
        defer { try? FileManager.default.removeItem(at: tmp) }
        let hub = tmp.appendingPathComponent("hf", isDirectory: true)
            .appendingPathComponent("hub", isDirectory: true)
        try FileManager.default.createDirectory(at: hub, withIntermediateDirectories: true)

        let file = hub.appendingPathComponent("models--mlx-community--Decoy")
        try "decoy contents".write(to: file, atomically: true, encoding: .utf8)
        // Treat it as if it were a directory candidate (the caller
        // appends with ``isDirectory: true`` blindly; the real
        // on-disk shape is what matters for the guard).
        let asDirURL = URL(fileURLWithPath: file.path, isDirectory: true)

        #expect(ModelDeletion._testingValidatedDeletionURL(asDirURL, hubCacheRoot: hub) == nil)
        // And the file must still be on disk — the validator did not
        // accidentally surface it as deletable.
        #expect(FileManager.default.fileExists(atPath: file.path))
    }

    @Test("Symlinked cache target escaping the hub root is rejected")
    func symlinkEscapeRejected() throws {
        let tmp = try makeTmpDir()
        defer { try? FileManager.default.removeItem(at: tmp) }
        let hub = tmp.appendingPathComponent("hf", isDirectory: true)
            .appendingPathComponent("hub", isDirectory: true)
        let outside = tmp.appendingPathComponent("outside", isDirectory: true)
        try FileManager.default.createDirectory(at: hub, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)

        let link = hub.appendingPathComponent("models--mlx-community--Qwen3", isDirectory: true)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: outside)

        #expect(ModelDeletion._testingValidatedDeletionURL(link, hubCacheRoot: hub) == nil)
        #expect(FileManager.default.fileExists(atPath: outside.path))
    }

    @Test("deleteCachedModel removes only the verified cache directory")
    func deleteRemovesVerifiedDirectory() async throws {
        let tmp = try makeTmpDir()
        defer { try? FileManager.default.removeItem(at: tmp) }
        let hub = tmp.appendingPathComponent("hf", isDirectory: true)
            .appendingPathComponent("hub", isDirectory: true)
        try FileManager.default.createDirectory(at: hub, withIntermediateDirectories: true)

        let repo = "mlx-community/Qwen3.5-4B-MLX-4bit"
        let dirName = try #require(ModelDeletion._testingCacheDirectoryName(forRepo: repo))
        let target = hub.appendingPathComponent(dirName, isDirectory: true)
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        try "weights".write(to: target.appendingPathComponent("model.safetensors"), atomically: true, encoding: .utf8)

        let sibling = hub.appendingPathComponent("models--mlx-community--Other", isDirectory: true)
        try FileManager.default.createDirectory(at: sibling, withIntermediateDirectories: true)

        let script = try makeExecutableScript(
            """
            #!/bin/sh
            if [ "$1" = "ls" ]; then
              cat <<'EOF'
            Alias                 HuggingFace repo                         Size
            qwen3.5-4b            mlx-community/Qwen3.5-4B-MLX-4bit         7 B
            EOF
            fi
            exit 0
            """,
            in: tmp
        )

        let outcome = await ModelDeletion.deleteCachedModel(
            binaryPath: script,
            alias: "qwen3.5-4b",
            hubCacheRoot: hub
        )
        guard case .freed(let bytes, let raw) = outcome else {
            Issue.record("Expected successful delete, got \(outcome)")
            return
        }
        #expect((bytes ?? 0) > 0)
        #expect(raw.contains(repo))
        #expect(!FileManager.default.fileExists(atPath: target.path))
        #expect(FileManager.default.fileExists(atPath: sibling.path))
    }

    @Test("Invalid alias is rejected before catalog lookup")
    func invalidAliasRejected() async {
        let outcome = await ModelDeletion.deleteCachedModel(
            binaryPath: URL(fileURLWithPath: "/bin/echo"),
            alias: "../escape"
        )
        #expect(outcome == .failed(message: "That model name isn't valid."))
    }

    @Test("Known non-chat repo can be deleted without re-entering the chat-only catalog")
    func knownRepoDeletesAudioSnapshot() async throws {
        let tmp = try makeTmpDir()
        defer { try? FileManager.default.removeItem(at: tmp) }
        let hub = tmp.appendingPathComponent("hub", isDirectory: true)
        try FileManager.default.createDirectory(at: hub, withIntermediateDirectories: true)

        let repo = "mlx-community/Kokoro-82M-bf16"
        let dirName = try #require(ModelDeletion._testingCacheDirectoryName(forRepo: repo))
        let target = hub.appendingPathComponent(dirName, isDirectory: true)
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        try Data("audio weights".utf8).write(to: target.appendingPathComponent("weights.npz"))

        let outcome = await ModelDeletion.deleteCachedModel(
            binaryPath: URL(fileURLWithPath: "/bin/echo"),
            alias: "kokoro",
            knownRepo: repo,
            hubCacheRoot: hub
        )
        guard case .freed = outcome else {
            Issue.record("Expected successful audio delete, got \(outcome)")
            return
        }
        #expect(!FileManager.default.fileExists(atPath: target.path))
    }

    private func makeTmpDir() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-model-deletion-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func makeExecutableScript(_ body: String, in dir: URL) throws -> URL {
        let script = dir.appendingPathComponent("fake-rapid-mlx.sh")
        try body.write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
        return script
    }
}

/// Pin the contract that the context-menu visibility logic in
/// ``ModelPickerBar`` matches the user's mental model: cached rows
/// expose the affordance, the currently-serving row does not, and
/// rows already in flight don't allow re-fire. We don't render the
/// view here — these are pure boolean checks against the same
/// predicate the view uses, lifted into a static helper so it can
/// be tested without ViewInspector.
@Suite("Cache-row context menu eligibility")
struct CacheRowEligibilityTests {
    private func entry(alias: String, cached: Bool, size: String? = "1.0G") -> ModelEntry {
        ModelEntry(alias: alias, hfRepo: nil, sizeOnDisk: size, cached: cached)
    }

    @Test("Cached row, server idle → eligible")
    func eligibleWhenCached() {
        let e = entry(alias: "qwen3.5-4b-4bit", cached: true)
        #expect(canOfferDelete(entry: e, servingAlias: nil, deletingAlias: nil))
    }

    @Test("Uncached row → never eligible")
    func uncachedRowsNeverDelete() {
        let e = entry(alias: "qwen3.6-122b-mxfp4", cached: false, size: nil)
        #expect(canOfferDelete(entry: e, servingAlias: nil, deletingAlias: nil) == false)
    }

    @Test("Currently-serving alias → not eligible (mmap'd weights)")
    func cantDeleteResident() {
        let e = entry(alias: "qwen3.5-4b-4bit", cached: true)
        #expect(canOfferDelete(entry: e, servingAlias: "qwen3.5-4b-4bit", deletingAlias: nil) == false)
    }

    @Test("In-flight deletion of the same alias → not eligible (no double-fire)")
    func inflightDeletionNotEligible() {
        let e = entry(alias: "qwen3.5-4b-4bit", cached: true)
        #expect(canOfferDelete(entry: e, servingAlias: nil, deletingAlias: "qwen3.5-4b-4bit") == false)
    }

    @Test("In-flight deletion of a SIBLING alias → still eligible")
    func siblingDeletionStillEligible() {
        // Deleting alias A must not freeze alias B's context menu —
        // a power user batch-cleaning their disk should be able to
        // queue the next delete the moment they confirm the current
        // one.
        let e = entry(alias: "qwen3.5-4b-4bit", cached: true)
        #expect(canOfferDelete(entry: e, servingAlias: nil, deletingAlias: "deepseek-v4-flash-2bit"))
    }

    /// Mirror of the predicate the SwiftUI ``.contextMenu`` body in
    /// ``ModelPickerBar.aliasButton`` evaluates. Lifted into a free
    /// function so the truth table is testable without spinning up
    /// ViewInspector against the closure body. Keep this in sync
    /// with the view.
    private func canOfferDelete(
        entry: ModelEntry,
        servingAlias: String?,
        deletingAlias: String?
    ) -> Bool {
        entry.cached
            && servingAlias != entry.alias
            && deletingAlias != entry.alias
    }
}
