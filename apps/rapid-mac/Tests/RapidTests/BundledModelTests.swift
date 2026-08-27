import Foundation
import Testing
@testable import Rapid

/// Contract for the v0.7.1 first-paint bundling (#229).
///
/// Pins:
/// - the bundled alias / HF repo IDs we promise the DMG ships weights for
/// - the HF Hub cache directory name derivation from a repo ID
/// - ``bundledSnapshotURL`` returns nil when the staged tree is absent
///   and the on-disk URL when it exists
/// - ``userHFCacheURL`` honours ``HF_HUB_CACHE`` > ``HF_HOME`` > ``HOME``
///   in the same order huggingface_hub does
/// - ``installBundledSnapshotSymlink`` is idempotent — second call returns
///   ``.alreadyPresent`` and never re-creates the symlink
/// - ``installBundledSnapshotSymlink`` doesn't overwrite a real user
///   download (a regular directory at the target path wins)
/// - ``firstLaunchAlias`` only returns the bundled alias when both
///   (a) there's no last-served alias AND (b) the bundled snapshot is
///   on disk
@Suite("BundledModel — first-paint instant-on (#229)")
struct BundledModelTests {

    // MARK: - Pinned identifiers

    @Test("Bundled alias matches the entry the rapid-mlx submodule ships")
    func bundledAliasIsCanonical() {
        // The aliases.json entry added in raullenchai/Rapid-MLX#1092.
        // Changing the value here without also bumping the submodule
        // pointer + re-running the bundled-snapshot download in
        // ``scripts/build.sh`` would ship a DMG whose weights don't
        // resolve to any alias — exactly the failure mode this
        // constant exists to prevent.
        #expect(BundledModel.bundledAlias == "lfm2.5-1b-4bit")
        #expect(BundledModel.bundledRepoID == "mlx-community/LFM2.5-1.2B-Instruct-4bit")
    }

    @Test("HF cache directory name follows huggingface_hub's models--<owner>--<name> rule")
    func cacheDirNameDerivation() {
        // The HF Hub cache encodes ``owner/name`` as ``models--owner--name``.
        // Both halves matter — a code path that produces ``mlx-community/LFM2.5-1.2B-Instruct-4bit``
        // verbatim (unescaped) would break symlink resolution because that's
        // not the path huggingface_hub looks for.
        #expect(BundledModel.bundledCacheDirName == "models--mlx-community--LFM2.5-1.2B-Instruct-4bit")
    }

    @MainActor
    @Test("Air-gapped bundle remains the explicit low-memory onboarding choice")
    func bundledAliasMatchesLowMemoryChoice() {
        // The upgrade banner fires only when the active alias equals
        // ``BundledModel.bundledAlias`` (see ``UpgradeBannerCoordinator``),
        // and air-gapped builds deliberately keep the smallest authored
        // onboarding choice. Production DMGs bundle no weights, so their
        // hardware-aware starter is independent from this fallback.
        #expect(BundledModel.bundledAlias == QuickstartCoordinator.lowMemoryChoice.alias)
        #expect(QuickstartCoordinator.lowMemoryChoice.hfRepo == BundledModel.bundledRepoID)
    }

    // MARK: - Snapshot URL resolution

    @Test("bundledSnapshotURL returns nil when no bundle resource URL")
    func bundledSnapshotURLNilOutsideBundle() {
        #expect(BundledModel.bundledSnapshotURL(bundleResourceURL: nil) == nil)
    }

    @Test("bundledSnapshotURL returns nil when the staged tree is missing")
    func bundledSnapshotURLNilWhenTreeMissing() throws {
        let tmp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: tmp) }
        // Bundle directory exists but nothing under Resources/models — this
        // is the dev build (SKIP_BUNDLED_MODEL=1) shape.
        #expect(BundledModel.bundledSnapshotURL(bundleResourceURL: tmp) == nil)
    }

    @Test("bundledSnapshotURL returns the staged path when present")
    func bundledSnapshotURLResolves() throws {
        let tmp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: tmp) }
        let staged = tmp
            .appendingPathComponent("models")
            .appendingPathComponent("hf-cache")
            .appendingPathComponent("hub")
            .appendingPathComponent("models--mlx-community--LFM2.5-1.2B-Instruct-4bit")
        try FileManager.default.createDirectory(at: staged, withIntermediateDirectories: true)
        let resolved = BundledModel.bundledSnapshotURL(bundleResourceURL: tmp)
        #expect(resolved?.path == staged.path)
    }

    // MARK: - User HF cache URL precedence

    @Test("userHFCacheURL honours HF_HUB_CACHE first")
    func userCacheHonoursHFHubCacheFirst() {
        let env: [String: String] = [
            "HF_HUB_CACHE": "/custom/hub",
            "HF_HOME": "/somewhere/else",
            "HOME": "/Users/test",
        ]
        let url = BundledModel.userHFCacheURL(environment: env)
        #expect(url?.path == "/custom/hub")
    }

    @Test("userHFCacheURL falls back to HF_HOME/hub when HF_HUB_CACHE missing")
    func userCacheHonoursHFHomeNext() {
        let env: [String: String] = [
            "HF_HOME": "/Users/test/.hf",
            "HOME": "/Users/test",
        ]
        let url = BundledModel.userHFCacheURL(environment: env)
        #expect(url?.path == "/Users/test/.hf/hub")
    }

    @Test("userHFCacheURL honours XDG_CACHE_HOME between HF_HOME and HOME")
    func userCacheHonoursXDG() {
        // huggingface_hub itself falls back to ``$XDG_CACHE_HOME/huggingface``
        // before defaulting to ``~/.cache/huggingface``. A user who sets
        // XDG_CACHE_HOME to e.g. /Volumes/Models/cache (common on Mac
        // setups with model drives) needs us to write the bundled
        // symlink into THAT cache, not ~/.cache/huggingface/, otherwise
        // the sidecar — which respects XDG via huggingface_hub — looks
        // for the model in a different place than we put it.
        let env: [String: String] = [
            "XDG_CACHE_HOME": "/Volumes/Models/cache",
            "HOME": "/Users/test",
        ]
        let url = BundledModel.userHFCacheURL(environment: env)
        #expect(url?.path == "/Volumes/Models/cache/huggingface/hub")
    }

    @Test("userHFCacheURL falls back to HOME/.cache/huggingface/hub")
    func userCacheHonoursHomeLast() {
        let env: [String: String] = ["HOME": "/Users/test"]
        let url = BundledModel.userHFCacheURL(environment: env)
        #expect(url?.path == "/Users/test/.cache/huggingface/hub")
    }

    @Test("userHFCacheURL returns nil with no env vars at all")
    func userCacheNilWithoutEnv() {
        #expect(BundledModel.userHFCacheURL(environment: [:]) == nil)
    }

    // MARK: - Models folder preference override (issue #503)

    @Test("preferredOverride wins over every env-derived tier")
    func preferredOverrideWinsOverEnv() {
        // When the user has pointed Rapid at a validated models folder,
        // the app's cache resolution must prefer it so disk scanning /
        // size / deletion all target the SAME directory the engine
        // reads/writes — even if HF_HUB_CACHE (the strongest env tier)
        // also happens to be set.
        let env: [String: String] = [
            "HF_HUB_CACHE": "/custom/hub",
            "HF_HOME": "/somewhere/else",
            "HOME": "/Users/test",
        ]
        let override = URL(fileURLWithPath: "/Volumes/T7/models", isDirectory: true)
        let url = BundledModel.userHFCacheURL(environment: env, preferredOverride: override)
        #expect(url?.path == "/Volumes/T7/models")
    }

    @Test("nil preferredOverride falls back to the env precedence chain")
    func nilOverrideFallsBackToEnv() {
        // The unplugged-drive case: the caller resolved the override to
        // nil, so app-side resolution must behave exactly as it did
        // before #503 (HF_HUB_CACHE → HF_HOME → XDG → HOME).
        let env: [String: String] = [
            "HF_HUB_CACHE": "/custom/hub",
            "HOME": "/Users/test",
        ]
        let url = BundledModel.userHFCacheURL(environment: env, preferredOverride: nil)
        #expect(url?.path == "/custom/hub")
    }

    // MARK: - Symlink install

    @Test("installBundledSnapshotSymlink lays down the symlink on first call")
    func installCreatesSymlink() throws {
        let bundle = try makeStagedBundle()
        defer { try? FileManager.default.removeItem(at: bundle.root) }
        let home = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: home) }
        let env = ["HOME": home.path]

        let outcome = BundledModel.installBundledSnapshotSymlink(
            bundleResourceURL: bundle.resourceURL,
            environment: env
        )
        #expect(outcome == .installed)
        // Verify the symlink lives where we expect AND points at the
        // staged snapshot. Two assertions because either side can drift.
        let target = home
            .appendingPathComponent(".cache")
            .appendingPathComponent("huggingface")
            .appendingPathComponent("hub")
            .appendingPathComponent("models--mlx-community--LFM2.5-1.2B-Instruct-4bit")
        let attrs = try FileManager.default.attributesOfItem(atPath: target.path)
        #expect((attrs[.type] as? FileAttributeType) == .typeSymbolicLink)
        let dest = try FileManager.default.destinationOfSymbolicLink(atPath: target.path)
        #expect(dest == bundle.snapshotURL.path)
    }

    @Test("installBundledSnapshotSymlink is idempotent — second call is alreadyPresent")
    func installIsIdempotent() throws {
        let bundle = try makeStagedBundle()
        defer { try? FileManager.default.removeItem(at: bundle.root) }
        let home = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: home) }
        let env = ["HOME": home.path]

        let first = BundledModel.installBundledSnapshotSymlink(
            bundleResourceURL: bundle.resourceURL,
            environment: env
        )
        #expect(first == .installed)
        let second = BundledModel.installBundledSnapshotSymlink(
            bundleResourceURL: bundle.resourceURL,
            environment: env
        )
        #expect(second == .alreadyPresent)
    }

    @Test("installBundledSnapshotSymlink does NOT overwrite a real user download")
    func installRespectsExistingDirectory() throws {
        let bundle = try makeStagedBundle()
        defer { try? FileManager.default.removeItem(at: bundle.root) }
        let home = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: home) }
        let env = ["HOME": home.path]

        // Pre-existing real directory (simulates the user having
        // already pulled the model in a previous session).
        let target = home
            .appendingPathComponent(".cache")
            .appendingPathComponent("huggingface")
            .appendingPathComponent("hub")
            .appendingPathComponent("models--mlx-community--LFM2.5-1.2B-Instruct-4bit")
        try FileManager.default.createDirectory(
            at: target,
            withIntermediateDirectories: true
        )
        try "real download".write(
            toFile: target.appendingPathComponent("real-file").path,
            atomically: true,
            encoding: .utf8
        )
        let outcome = BundledModel.installBundledSnapshotSymlink(
            bundleResourceURL: bundle.resourceURL,
            environment: env
        )
        #expect(outcome == .alreadyPresent)
        // The real download survives untouched — its file is still there.
        #expect(FileManager.default.fileExists(
            atPath: target.appendingPathComponent("real-file").path
        ))
    }

    @Test("installBundledSnapshotSymlink leaves a correct symlink untouched (no-op re-link)")
    func installLeavesMatchingSymlinkAlone() throws {
        let bundle = try makeStagedBundle()
        defer { try? FileManager.default.removeItem(at: bundle.root) }
        let home = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: home) }
        let env = ["HOME": home.path]

        // Pre-seed the user cache with a symlink that ALREADY points at
        // the live snapshot — the normal steady-state on every launch
        // after the first. We want the implementation to short-circuit
        // BEFORE doing any disk surgery (no unlink, no recreate).
        let target = home
            .appendingPathComponent(".cache")
            .appendingPathComponent("huggingface")
            .appendingPathComponent("hub")
            .appendingPathComponent("models--mlx-community--LFM2.5-1.2B-Instruct-4bit")
        try FileManager.default.createDirectory(
            at: target.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try FileManager.default.createSymbolicLink(
            at: target,
            withDestinationURL: bundle.snapshotURL
        )
        // Snapshot the link's inode so a silent unlink+recreate would
        // change it — proves the no-op path didn't touch the link.
        let preAttrs = try FileManager.default.attributesOfItem(atPath: target.path)
        let preInode = preAttrs[.systemFileNumber] as? NSNumber

        let outcome = BundledModel.installBundledSnapshotSymlink(
            bundleResourceURL: bundle.resourceURL,
            environment: env
        )
        #expect(outcome == .alreadyPresent)

        let postAttrs = try FileManager.default.attributesOfItem(atPath: target.path)
        #expect((postAttrs[.type] as? FileAttributeType) == .typeSymbolicLink)
        let postInode = postAttrs[.systemFileNumber] as? NSNumber
        #expect(preInode == postInode, "matching symlink must not be re-created")
        let dest = try FileManager.default.destinationOfSymbolicLink(atPath: target.path)
        #expect(dest == bundle.snapshotURL.path)
    }

    @Test("installBundledSnapshotSymlink re-links a stale symlink when the .app moves")
    func installRelinksStaleSymlink() throws {
        // Simulate the .app moving between two paths (e.g. user dragged
        // it from /Applications to /Volumes/External): we stage TWO
        // bundles, point a pre-existing symlink at the OLD one, delete
        // the old bundle, then call install with the NEW bundle. The
        // pre-fix behaviour returned .alreadyPresent and left the user
        // staring at a "model not found" error on first chat.
        let oldBundle = try makeStagedBundle()
        let newBundle = try makeStagedBundle()
        defer { try? FileManager.default.removeItem(at: newBundle.root) }
        let home = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: home) }
        let env = ["HOME": home.path]

        let target = home
            .appendingPathComponent(".cache")
            .appendingPathComponent("huggingface")
            .appendingPathComponent("hub")
            .appendingPathComponent("models--mlx-community--LFM2.5-1.2B-Instruct-4bit")
        try FileManager.default.createDirectory(
            at: target.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try FileManager.default.createSymbolicLink(
            at: target,
            withDestinationURL: oldBundle.snapshotURL
        )
        let oldSnapshotPath = oldBundle.snapshotURL.path
        // Wipe the OLD bundle on disk — this mirrors the user trashing
        // the previous .app after copying to a new location.
        try FileManager.default.removeItem(at: oldBundle.root)

        let outcome = BundledModel.installBundledSnapshotSymlink(
            bundleResourceURL: newBundle.resourceURL,
            environment: env
        )
        #expect(outcome == .relinked(oldDestination: oldSnapshotPath))

        // The symlink now points at the NEW snapshot, end-to-end:
        let attrs = try FileManager.default.attributesOfItem(atPath: target.path)
        #expect((attrs[.type] as? FileAttributeType) == .typeSymbolicLink)
        let dest = try FileManager.default.destinationOfSymbolicLink(atPath: target.path)
        #expect(dest == newBundle.snapshotURL.path)
        // And resolves to a real on-disk file (via the marker that
        // makeStagedBundle drops inside every snapshot).
        let marker = target.appendingPathComponent("snapshot-marker")
        #expect(FileManager.default.fileExists(atPath: marker.path))
    }

    @Test("installBundledSnapshotSymlink re-links a dangling symlink")
    func installRelinksDanglingSymlink() throws {
        // Subtler variant of the stale-symlink case: the prior
        // destination path NEVER existed (we point straight at a tmp
        // path we never created). The pre-fix attributesOfItem check
        // saw .typeSymbolicLink and returned .alreadyPresent; the
        // post-fix code re-links it.
        let bundle = try makeStagedBundle()
        defer { try? FileManager.default.removeItem(at: bundle.root) }
        let home = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: home) }
        let env = ["HOME": home.path]

        let target = home
            .appendingPathComponent(".cache")
            .appendingPathComponent("huggingface")
            .appendingPathComponent("hub")
            .appendingPathComponent("models--mlx-community--LFM2.5-1.2B-Instruct-4bit")
        try FileManager.default.createDirectory(
            at: target.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let danglingDest = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("does-not-exist-" + UUID().uuidString)
        try FileManager.default.createSymbolicLink(
            at: target,
            withDestinationURL: danglingDest
        )

        let outcome = BundledModel.installBundledSnapshotSymlink(
            bundleResourceURL: bundle.resourceURL,
            environment: env
        )
        #expect(outcome == .relinked(oldDestination: danglingDest.path))

        let dest = try FileManager.default.destinationOfSymbolicLink(atPath: target.path)
        #expect(dest == bundle.snapshotURL.path)
    }

    @Test("installBundledSnapshotSymlink is noBundledSnapshot when the staged tree is missing")
    func installNoBundledWhenTreeMissing() throws {
        let bundle = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: bundle) }
        let home = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: home) }
        let env = ["HOME": home.path]
        let outcome = BundledModel.installBundledSnapshotSymlink(
            bundleResourceURL: bundle,
            environment: env
        )
        #expect(outcome == .noBundledSnapshot)
    }

    @Test("installBundledSnapshotSymlink is userCacheUnavailable without HOME / HF env")
    func installNoUserCache() throws {
        let bundle = try makeStagedBundle()
        defer { try? FileManager.default.removeItem(at: bundle.root) }
        let outcome = BundledModel.installBundledSnapshotSymlink(
            bundleResourceURL: bundle.resourceURL,
            environment: [:]
        )
        #expect(outcome == .userCacheUnavailable)
    }

    // MARK: - First-launch alias decision

    @Test("firstLaunchAlias returns the bundled alias on a true fresh install")
    func firstLaunchFreshInstall() throws {
        let bundle = try makeStagedBundle()
        defer { try? FileManager.default.removeItem(at: bundle.root) }
        let alias = BundledModel.firstLaunchAlias(
            lastServedAlias: nil,
            bundleResourceURL: bundle.resourceURL
        )
        #expect(alias == "lfm2.5-1b-4bit")
    }

    @Test("firstLaunchAlias returns nil when the user already has a last-served alias")
    func firstLaunchRespectsLastServed() throws {
        let bundle = try makeStagedBundle()
        defer { try? FileManager.default.removeItem(at: bundle.root) }
        // User had already trade-upped on a previous launch — we must
        // NOT yank them back to the 0.6B bundled model. Verified
        // explicitly because it's the regression the RAMBucketedDefault
        // first-launch path used to silently introduce.
        let alias = BundledModel.firstLaunchAlias(
            lastServedAlias: "qwen3.6-35b-4bit",
            bundleResourceURL: bundle.resourceURL
        )
        #expect(alias == nil)
    }

    @Test("firstLaunchAlias returns nil when bundled snapshot is missing")
    func firstLaunchNilWithoutSnapshot() throws {
        let tmp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: tmp) }
        let alias = BundledModel.firstLaunchAlias(
            lastServedAlias: nil,
            bundleResourceURL: tmp
        )
        // Dev / CI build skipped SKIP_BUNDLED_MODEL — fall through to
        // the existing RAM-bucketed path, signalled by returning nil.
        #expect(alias == nil)
    }

    // MARK: - Helpers

    /// Staged bundle fixture mirroring what ``scripts/build.sh`` lays
    /// down — gives us a real on-disk Resources/models/hf-cache/hub/
    /// tree so the production code path runs unmodified in tests.
    private struct StagedBundle {
        let root: URL
        let resourceURL: URL
        let snapshotURL: URL
    }

    private func makeStagedBundle() throws -> StagedBundle {
        let root = try makeTempDir()
        let resourceURL = root.appendingPathComponent("Resources", isDirectory: true)
        try FileManager.default.createDirectory(at: resourceURL, withIntermediateDirectories: true)
        let snapshot = resourceURL
            .appendingPathComponent("models")
            .appendingPathComponent("hf-cache")
            .appendingPathComponent("hub")
            .appendingPathComponent("models--mlx-community--LFM2.5-1.2B-Instruct-4bit")
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        // Drop a marker file so a future "is this snapshot complete?"
        // check has something to grep for; today the existence of the
        // directory itself is sufficient.
        try "marker".write(
            toFile: snapshot.appendingPathComponent("snapshot-marker").path,
            atomically: true,
            encoding: .utf8
        )
        return StagedBundle(root: root, resourceURL: resourceURL, snapshotURL: snapshot)
    }

    private func makeTempDir() throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("BundledModelTests-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}
