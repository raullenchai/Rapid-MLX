import Foundation
import Testing
@testable import Rapid

/// Contract for the v0.8.8 #414 fix — slim-DMG Quickstart install
/// must publish an HF Hub cache stub so ``rapid-mlx ls`` finds the
/// alias and ``AutoStartDecision.decide`` returns ``.start``.
///
/// Pins:
/// - the canonical Quickstart alias / repoID and the
///   ``models--<owner>--<name>`` cache dir derivation
/// - ``resolveFlatModelDir`` resolves a fresh flat install
///   (``<root>/<alias>/<files>`` — the post-#416 canonical shape the
///   producer now packs) via its PREFERRED branch, AND still resolves
///   a LEGACY nested install (``<root>/<alias>/<alias>/<files>`` —
///   pre-#416 F-DGF-V087-03 shape) via the back-compat fallback so
///   machines that installed a pre-fix tarball keep loading
/// - ``installSnapshotSymlink`` is idempotent — second call returns
///   ``.alreadyPresent`` and does not re-create the stub
/// - ``installSnapshotSymlink`` respects a user's real HF Hub cache
///   entry (regular directory at the target wins, never rebuilt)
/// - ``installSnapshotSymlink`` rebuilds a stub whose leaf symlinks
///   no longer match the live flat model dir (relocation /
///   re-extraction case)
/// - ``installAllSnapshotSymlinks`` reports ``.noQuickstartModel``
///   for known aliases not yet on disk (full-DMG users, fresh
///   sandbox, install-rolled-back case)
@Suite("QuickstartModel — slim-DMG HF cache stub (#414)")
struct QuickstartModelTests {

    // MARK: - Pinned identifiers

    @Test("Known aliases includes the slim-DMG Quickstart model")
    func knownAliasesIsCanonical() {
        // v0.8.x slim DMGs ship bonsai-1.7b-2bit weights to converge
        // with the full-DMG bundled snapshot. Removing this entry
        // without also pruning ``scripts/build-model-tarball.sh``
        // would ship a slim DMG whose Quickstart install can never
        // be auto-started — exactly the failure mode this test
        // exists to prevent.
        let spec = QuickstartModel.knownAliases["bonsai-1.7b-2bit"]
        #expect(spec?.alias == "bonsai-1.7b-2bit")
        #expect(spec?.repoID == "prism-ml/Ternary-Bonsai-1.7B-mlx-2bit")
    }

    @Test("Spec.cacheDirName follows huggingface_hub's models--<owner>--<name> rule")
    func cacheDirNameDerivation() {
        let spec = QuickstartModel.Spec(
            alias: "bonsai-1.7b-2bit",
            repoID: "prism-ml/Ternary-Bonsai-1.7B-mlx-2bit"
        )
        #expect(spec.cacheDirName == "models--prism-ml--Ternary-Bonsai-1.7B-mlx-2bit")
    }

    @Test("Stub revision marker is stable")
    func stubRevisionMarkerIsStable() {
        // Changing this would make every existing stub look "stale"
        // on next launch and force a needless re-link (~50ms × N
        // files). Tested explicitly so a future refactor that wants
        // to rename it has to do so deliberately.
        #expect(QuickstartModel.stubRevisionMarker == "quickstart")
    }

    // MARK: - Flat model dir resolution

    @Test("resolveFlatModelDir resolves a fresh FLAT install via the PREFERRED branch (post-#416 canonical shape)")
    func resolveFlatLayout() throws {
        // #416: the producer (scripts/build-model-tarball.sh) now packs
        // FLAT, so a fresh install lands at <root>/<alias>/<files> and
        // resolves via resolveFlatModelDir's preferred (non-legacy)
        // branch. This is the canonical shape going forward.
        let env = try TestEnv.make()
        defer { env.tearDown() }
        let alias = "bonsai-1.7b-2bit"
        let flat = env.installRoot.appendingPathComponent(alias, isDirectory: true)
        try FileManager.default.createDirectory(at: flat, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: flat.appendingPathComponent("config.json"))

        let resolved = QuickstartModel.resolveFlatModelDir(
            alias: alias,
            installRoot: env.installRoot
        )
        #expect(resolved?.path == flat.path)
    }

    @Test("LEGACY back-compat: resolveFlatModelDir still resolves the nested layout (root/alias/alias/files) — pre-#416 F-DGF-V087-03")
    func resolveNestedLayout() throws {
        // BACK-COMPAT GUARD. Pre-#416 tarballs shipped a leading
        // <alias>/ dir, so commit landed files at
        // <root>/<alias>/<alias>/<files>. The producer is now fixed to
        // pack flat (see scriptPacksFlatNoAliasWrapper), BUT machines
        // that already installed a pre-fix tarball keep that nested
        // shape on disk — ModelInstaller.stage refuses to re-extract
        // over an existing <root>/<alias> (.alreadyInstalled), so the
        // nested fallback in resolveFlatModelDir MUST stay for at least
        // this release so those installs keep loading. Do NOT delete
        // this test or the fallback branch it guards without a
        // migration.
        let env = try TestEnv.make()
        defer { env.tearDown() }
        let alias = "bonsai-1.7b-2bit"
        let nested = env.installRoot
            .appendingPathComponent(alias, isDirectory: true)
            .appendingPathComponent(alias, isDirectory: true)
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: nested.appendingPathComponent("config.json"))

        let resolved = QuickstartModel.resolveFlatModelDir(
            alias: alias,
            installRoot: env.installRoot
        )
        #expect(resolved?.path == nested.path)
    }

    @Test("resolveFlatModelDir returns nil when neither layout is present")
    func resolveNeitherLayoutPresent() throws {
        let env = try TestEnv.make()
        defer { env.tearDown() }
        // Make the alias dir but with no config.json inside (mid-
        // staging case, or a partial extraction we shouldn't act on).
        let aliasDir = env.installRoot.appendingPathComponent("bonsai-1.7b-2bit", isDirectory: true)
        try FileManager.default.createDirectory(at: aliasDir, withIntermediateDirectories: true)
        #expect(QuickstartModel.resolveFlatModelDir(
            alias: "bonsai-1.7b-2bit",
            installRoot: env.installRoot
        ) == nil)
    }

    // MARK: - Install — happy path

    @Test("installSnapshotSymlink creates refs/main + snapshots/<rev>/<symlinks>")
    func installCreatesStub() throws {
        let env = try TestEnv.make()
        defer { env.tearDown() }
        let alias = "bonsai-1.7b-2bit"
        let spec = QuickstartModel.knownAliases[alias]!
        let flat = try stageFlatModel(alias: alias, installRoot: env.installRoot)

        let outcome = QuickstartModel.installSnapshotSymlink(
            spec: spec,
            installRoot: env.installRoot,
            environment: env.envDict
        )
        #expect(outcome == .installed)

        // refs/main contains the marker
        let cacheDir = env.userHubURL.appendingPathComponent(spec.cacheDirName)
        let refsMain = cacheDir
            .appendingPathComponent("refs")
            .appendingPathComponent("main")
        let refsContent = try String(contentsOf: refsMain, encoding: .utf8)
        #expect(refsContent == QuickstartModel.stubRevisionMarker)

        // snapshots/<rev>/<file> exists per flat file, each is a
        // symlink pointing back into the flat model dir
        let snapshotDir = cacheDir
            .appendingPathComponent("snapshots")
            .appendingPathComponent(QuickstartModel.stubRevisionMarker)
        for name in ["config.json", "model.safetensors", "tokenizer.json"] {
            let leaf = snapshotDir.appendingPathComponent(name)
            let attrs = try FileManager.default.attributesOfItem(atPath: leaf.path)
            #expect((attrs[.type] as? FileAttributeType) == .typeSymbolicLink)
            let dest = try FileManager.default.destinationOfSymbolicLink(atPath: leaf.path)
            #expect(dest == flat.appendingPathComponent(name).path)
        }
    }

    @Test("installSnapshotSymlink is idempotent — second call is alreadyPresent")
    func installIsIdempotent() throws {
        let env = try TestEnv.make()
        defer { env.tearDown() }
        let alias = "bonsai-1.7b-2bit"
        let spec = QuickstartModel.knownAliases[alias]!
        _ = try stageFlatModel(alias: alias, installRoot: env.installRoot)

        let first = QuickstartModel.installSnapshotSymlink(
            spec: spec,
            installRoot: env.installRoot,
            environment: env.envDict
        )
        #expect(first == .installed)

        let second = QuickstartModel.installSnapshotSymlink(
            spec: spec,
            installRoot: env.installRoot,
            environment: env.envDict
        )
        #expect(second == .alreadyPresent)
    }

    @Test("installSnapshotSymlink does NOT touch a real user HF Hub cache entry")
    func installRespectsRealUserCache() throws {
        let env = try TestEnv.make()
        defer { env.tearDown() }
        let alias = "bonsai-1.7b-2bit"
        let spec = QuickstartModel.knownAliases[alias]!
        _ = try stageFlatModel(alias: alias, installRoot: env.installRoot)

        // Simulate a real HF Hub download already present (refs/main
        // has a 40-char SHA, not our marker — that's the
        // discriminator).
        let cacheDir = env.userHubURL.appendingPathComponent(spec.cacheDirName)
        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        let realRefsDir = cacheDir.appendingPathComponent("refs")
        try FileManager.default.createDirectory(at: realRefsDir, withIntermediateDirectories: true)
        let realSHA = "abcdef0123456789abcdef0123456789abcdef01"
        try Data(realSHA.utf8).write(to: realRefsDir.appendingPathComponent("main"))
        try Data("downloaded-by-user".utf8).write(
            to: cacheDir.appendingPathComponent("USER-MARKER")
        )

        let outcome = QuickstartModel.installSnapshotSymlink(
            spec: spec,
            installRoot: env.installRoot,
            environment: env.envDict
        )
        #expect(outcome == .alreadyPresent)

        // refs/main UNCHANGED — still the real SHA, not our marker
        let postRefs = try String(
            contentsOf: realRefsDir.appendingPathComponent("main"),
            encoding: .utf8
        )
        #expect(postRefs == realSHA)
        // User's marker file survives untouched
        #expect(FileManager.default.fileExists(
            atPath: cacheDir.appendingPathComponent("USER-MARKER").path
        ))
    }

    @Test("installSnapshotSymlink NEVER follows a user-placed symlink at the cache-dir path (codex r1 BLOCKING)")
    func installRespectsSymlinkAtCacheDir() throws {
        // Codex r1 BLOCKING: if cacheDir is itself a symlink (user
        // redirected models--<owner>--<name> to e.g. a network
        // drive's HF cache), a rebuild path would let writeStub's
        // removeItem(at: refsDir) follow the symlink prefix and
        // delete refs/snapshots inside the SYMLINK TARGET. This
        // test pins the post-fix behaviour: symlink at cacheDir →
        // .alreadyPresent, victim's refs/snapshots untouched.
        let env = try TestEnv.make()
        defer { env.tearDown() }
        let alias = "bonsai-1.7b-2bit"
        let spec = QuickstartModel.knownAliases[alias]!
        _ = try stageFlatModel(alias: alias, installRoot: env.installRoot)

        // Create a "victim" directory that the user-placed symlink
        // points at, complete with their own refs/ and snapshots/
        // sub-trees we must not touch.
        let victim = env.root.appendingPathComponent("victim-cache", isDirectory: true)
        let victimRefs = victim.appendingPathComponent("refs")
        let victimSnapshots = victim.appendingPathComponent("snapshots")
        try FileManager.default.createDirectory(at: victimRefs, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: victimSnapshots, withIntermediateDirectories: true)
        try Data("user-owned-ref".utf8).write(to: victimRefs.appendingPathComponent("main"))
        try Data("user-owned-snapshot".utf8).write(
            to: victimSnapshots.appendingPathComponent("USER-FILE")
        )

        // Make the user HF cache dir + place the symlink at the
        // path our installer would target.
        try FileManager.default.createDirectory(at: env.userHubURL, withIntermediateDirectories: true)
        let cacheDir = env.userHubURL.appendingPathComponent(spec.cacheDirName)
        try FileManager.default.createSymbolicLink(at: cacheDir, withDestinationURL: victim)

        let outcome = QuickstartModel.installSnapshotSymlink(
            spec: spec,
            installRoot: env.installRoot,
            environment: env.envDict
        )
        #expect(outcome == .alreadyPresent)

        // Symlink survives unchanged
        let attrs = try FileManager.default.attributesOfItem(atPath: cacheDir.path)
        #expect((attrs[.type] as? FileAttributeType) == .typeSymbolicLink)
        let dest = try FileManager.default.destinationOfSymbolicLink(atPath: cacheDir.path)
        #expect(dest == victim.path)

        // Victim's refs/snapshots untouched — both file content + dir contents
        let postRefs = try String(
            contentsOf: victimRefs.appendingPathComponent("main"),
            encoding: .utf8
        )
        #expect(postRefs == "user-owned-ref")
        #expect(FileManager.default.fileExists(
            atPath: victimSnapshots.appendingPathComponent("USER-FILE").path
        ))
    }

    @Test("installSnapshotSymlink leaves a dangling user-placed symlink alone (codex r1 BLOCKING)")
    func installRespectsDanglingSymlinkAtCacheDir() throws {
        // Stricter variant: even a DANGLING symlink at cacheDir is
        // user-placed (writeStub never creates symlinks at the
        // cache-dir level), so we must not unlink-and-rebuild — that
        // would silently replace a deliberate redirect with our
        // stub. .alreadyPresent is correct; the user-side fix is
        // theirs to make.
        let env = try TestEnv.make()
        defer { env.tearDown() }
        let alias = "bonsai-1.7b-2bit"
        let spec = QuickstartModel.knownAliases[alias]!
        _ = try stageFlatModel(alias: alias, installRoot: env.installRoot)

        try FileManager.default.createDirectory(at: env.userHubURL, withIntermediateDirectories: true)
        let cacheDir = env.userHubURL.appendingPathComponent(spec.cacheDirName)
        let dangling = env.root.appendingPathComponent("does-not-exist-" + UUID().uuidString)
        try FileManager.default.createSymbolicLink(at: cacheDir, withDestinationURL: dangling)

        let outcome = QuickstartModel.installSnapshotSymlink(
            spec: spec,
            installRoot: env.installRoot,
            environment: env.envDict
        )
        #expect(outcome == .alreadyPresent)

        let attrs = try FileManager.default.attributesOfItem(atPath: cacheDir.path)
        #expect((attrs[.type] as? FileAttributeType) == .typeSymbolicLink)
        let dest = try FileManager.default.destinationOfSymbolicLink(atPath: cacheDir.path)
        #expect(dest == dangling.path)
    }

    @Test("installSnapshotSymlink rebuilds a stale stub whose leaf points wrong")
    func installRebuildsStaleStub() throws {
        let env = try TestEnv.make()
        defer { env.tearDown() }
        let alias = "bonsai-1.7b-2bit"
        let spec = QuickstartModel.knownAliases[alias]!
        let flat = try stageFlatModel(alias: alias, installRoot: env.installRoot)

        // First install — happy path
        #expect(QuickstartModel.installSnapshotSymlink(
            spec: spec,
            installRoot: env.installRoot,
            environment: env.envDict
        ) == .installed)

        // Now corrupt: relocate the flat model dir mid-flight by
        // replacing its config.json with a different file path the
        // stub doesn't know about. Easiest way: overwrite the leaf
        // symlink with one pointing at a bogus path so stubIsIntact
        // detects drift.
        let cacheDir = env.userHubURL.appendingPathComponent(spec.cacheDirName)
        let snapshotDir = cacheDir
            .appendingPathComponent("snapshots")
            .appendingPathComponent(QuickstartModel.stubRevisionMarker)
        let badLeaf = snapshotDir.appendingPathComponent("config.json")
        try FileManager.default.removeItem(at: badLeaf)
        try FileManager.default.createSymbolicLink(
            at: badLeaf,
            withDestinationURL: URL(fileURLWithPath: "/tmp/bogus-destination")
        )

        let outcome = QuickstartModel.installSnapshotSymlink(
            spec: spec,
            installRoot: env.installRoot,
            environment: env.envDict
        )
        #expect(outcome == .relinked)

        // Post-rebuild leaf points back at the live flat model dir
        let dest = try FileManager.default.destinationOfSymbolicLink(atPath: badLeaf.path)
        #expect(dest == flat.appendingPathComponent("config.json").path)
    }

    // MARK: - Install — guard rails

    @Test("installSnapshotSymlink is noQuickstartModel when nothing is staged")
    func installNoQuickstartModel() throws {
        let env = try TestEnv.make()
        defer { env.tearDown() }
        let spec = QuickstartModel.knownAliases["bonsai-1.7b-2bit"]!
        #expect(QuickstartModel.installSnapshotSymlink(
            spec: spec,
            installRoot: env.installRoot,
            environment: env.envDict
        ) == .noQuickstartModel)
    }

    @Test("missing Quickstart source removes only our dangling HF cache stub")
    func missingSourceRemovesOwnedDanglingStub() throws {
        let env = try TestEnv.make()
        defer { env.tearDown() }
        let spec = QuickstartModel.knownAliases["bonsai-1.7b-2bit"]!
        let flat = try stageFlatModel(alias: spec.alias, installRoot: env.installRoot)
        #expect(QuickstartModel.installSnapshotSymlink(
            spec: spec, installRoot: env.installRoot, environment: env.envDict
        ) == .installed)
        try FileManager.default.removeItem(at: flat.deletingLastPathComponent())

        #expect(QuickstartModel.installSnapshotSymlink(
            spec: spec, installRoot: env.installRoot, environment: env.envDict
        ) == .removedStaleStub)
        let cacheDir = env.userHubURL.appendingPathComponent(spec.cacheDirName)
        #expect(!FileManager.default.fileExists(atPath: cacheDir.path))
    }

    @Test("missing Quickstart source never removes a real HF cache entry")
    func missingSourcePreservesRealCache() throws {
        let env = try TestEnv.make()
        defer { env.tearDown() }
        let spec = QuickstartModel.knownAliases["bonsai-1.7b-2bit"]!
        let cacheDir = env.userHubURL.appendingPathComponent(spec.cacheDirName)
        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        let sentinel = cacheDir.appendingPathComponent("user-data")
        try Data("keep".utf8).write(to: sentinel)

        #expect(QuickstartModel.installSnapshotSymlink(
            spec: spec, installRoot: env.installRoot, environment: env.envDict
        ) == .noQuickstartModel)
        #expect(FileManager.default.fileExists(atPath: sentinel.path))
    }

    @Test("missing source removes Quickstart leaves but preserves coexisting HF revisions")
    func missingSourcePreservesCoexistingHFRevision() throws {
        let env = try TestEnv.make()
        defer { env.tearDown() }
        let spec = QuickstartModel.knownAliases["bonsai-1.7b-2bit"]!
        let flat = try stageFlatModel(alias: spec.alias, installRoot: env.installRoot)
        #expect(QuickstartModel.installSnapshotSymlink(
            spec: spec, installRoot: env.installRoot, environment: env.envDict
        ) == .installed)
        let cacheDir = env.userHubURL.appendingPathComponent(spec.cacheDirName)
        let pinnedRef = cacheDir.appendingPathComponent("refs/v1")
        let realSnapshot = cacheDir.appendingPathComponent("snapshots/abc123")
        try Data("abc123".utf8).write(to: pinnedRef)
        try FileManager.default.createDirectory(at: realSnapshot, withIntermediateDirectories: true)
        let realWeight = realSnapshot.appendingPathComponent("weights.safetensors")
        try Data("real".utf8).write(to: realWeight)
        try FileManager.default.removeItem(at: flat.deletingLastPathComponent())

        #expect(QuickstartModel.installSnapshotSymlink(
            spec: spec, installRoot: env.installRoot, environment: env.envDict
        ) == .removedStaleStub)
        #expect(FileManager.default.fileExists(atPath: pinnedRef.path))
        #expect(FileManager.default.fileExists(atPath: realWeight.path))
        #expect(!FileManager.default.fileExists(
            atPath: cacheDir.appendingPathComponent("snapshots/quickstart").path
        ))
    }

    @Test("installSnapshotSymlink is userCacheUnavailable without HOME / HF env")
    func installNoUserCache() throws {
        let env = try TestEnv.make()
        defer { env.tearDown() }
        let alias = "bonsai-1.7b-2bit"
        let spec = QuickstartModel.knownAliases[alias]!
        _ = try stageFlatModel(alias: alias, installRoot: env.installRoot)

        #expect(QuickstartModel.installSnapshotSymlink(
            spec: spec,
            installRoot: env.installRoot,
            environment: [:]
        ) == .userCacheUnavailable)
    }

    // MARK: - Fan-out

    @Test("installAllSnapshotSymlinks reports per-alias outcomes")
    func installAllSnapshotSymlinksReportsPerAliasOutcomes() throws {
        let env = try TestEnv.make()
        defer { env.tearDown() }
        // Stage only one of the known aliases (currently the only
        // known one, but the test still exercises the fan-out loop).
        _ = try stageFlatModel(alias: "bonsai-1.7b-2bit", installRoot: env.installRoot)

        let outcomes = QuickstartModel.installAllSnapshotSymlinks(
            installRoot: env.installRoot,
            environment: env.envDict
        )
        for (alias, _) in QuickstartModel.knownAliases {
            #expect(outcomes[alias] != nil)
        }
        #expect(outcomes["bonsai-1.7b-2bit"] == .installed)
    }

    @Test("installAllSnapshotSymlinks returns noQuickstartModel for every alias when install root is nil")
    func installAllSnapshotSymlinksNilRoot() throws {
        let env = try TestEnv.make()
        defer { env.tearDown() }
        let outcomes = QuickstartModel.installAllSnapshotSymlinks(
            installRoot: nil,
            environment: env.envDict
        )
        for (alias, outcome) in outcomes {
            #expect(outcome == .noQuickstartModel, "alias \(alias) should be noQuickstartModel")
        }
        // Every known alias is represented in the result map.
        #expect(outcomes.count == QuickstartModel.knownAliases.count)
    }

    // MARK: - Helpers

    private struct TestEnv {
        let root: URL
        let installRoot: URL
        let userHubURL: URL

        var envDict: [String: String] {
            // Pin HF_HUB_CACHE explicitly so the test does NOT depend
            // on the runner's real HOME (which would land assertions
            // in the dev's actual ~/.cache/huggingface — exactly the
            // cross-test pollution we don't want).
            ["HF_HUB_CACHE": userHubURL.path]
        }

        static func make() throws -> TestEnv {
            let root = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("QuickstartModelTests-" + UUID().uuidString, isDirectory: true)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            let installRoot = root.appendingPathComponent("quickstart-models", isDirectory: true)
            try FileManager.default.createDirectory(at: installRoot, withIntermediateDirectories: true)
            let userHubURL = root.appendingPathComponent("hub", isDirectory: true)
            return TestEnv(root: root, installRoot: installRoot, userHubURL: userHubURL)
        }

        func tearDown() {
            try? FileManager.default.removeItem(at: root)
        }
    }

    private func stageFlatModel(alias: String, installRoot: URL) throws -> URL {
        // Stage in the FLAT layout so resolveFlatModelDir picks it
        // up. Nested-layout variants exercise resolveFlatModelDir
        // directly in dedicated tests above; the install path itself
        // is shape-agnostic past resolution.
        let flat = installRoot.appendingPathComponent(alias, isDirectory: true)
        try FileManager.default.createDirectory(at: flat, withIntermediateDirectories: true)
        // Mimic the v0.8.x tarball file set (the three pinned in the
        // installCreatesStub assertion). config.json is the
        // resolution sentinel for resolveFlatModelDir, so it must
        // exist for the install path to reach the stub-write phase.
        for name in ["config.json", "model.safetensors", "tokenizer.json"] {
            try Data("stub-\(name)".utf8).write(to: flat.appendingPathComponent(name))
        }
        return flat
    }
}
