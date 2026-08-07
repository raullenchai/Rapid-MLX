import Foundation

/// Slim-DMG (~5.6 MB bootstrapper) installs a Quickstart model into
/// ``~/Library/Application Support/Rapid/quickstart-models/<alias>/``
/// — a flat directory of HuggingFace weights produced by
/// ``scripts/build-model-tarball.sh``. That layout is NOT the HF Hub
/// cache shape (``models--<owner>--<name>/snapshots/<sha>/<files>``),
/// so ``rapid-mlx ls`` — which enumerates ``<HF_HUB_CACHE>/models--*/``
/// — never sees it. ``AutoStartDecision.decide`` consequently returns
/// ``.promptDownload`` instead of ``.start``, the sidecar is never
/// auto-spawned, and the slim-DMG first-launch UX bricks (#414,
/// v0.8.7 dogfood F-DGF-V087-01).
///
/// ## Why this is a desktop-side fix and not a rapid-mlx change
///
/// rapid-mlx's HF-cache scan is the single source of truth across
/// every downstream tool (LM Studio export, HF Hub web links, the
/// sidecar's own ``snapshot_download`` resolution). Teaching it about
/// a desktop-specific install root would burn cross-tool invariants.
/// Instead we make the Quickstart install LOOK like an HF Hub cache
/// entry, the same trick ``BundledModel.installBundledSnapshotSymlink``
/// uses for the bundled .app snapshot.
///
/// ## What we fabricate
///
/// At ``<userHFCache>/models--<owner>--<name>/`` we lay down:
///
/// - ``refs/main`` — a plain file containing the revision marker
///   (``quickstart``). Required by ``huggingface_hub`` to resolve
///   ``snapshot_download(revision="main")`` against the local cache.
/// - ``snapshots/quickstart/<file>`` — one absolute-path symlink per
///   file in the flat Quickstart install, pointing into the directory
///   ``resolveFlatModelDir`` returns:
///   ``<appSupport>/Rapid/quickstart-models/<alias>/`` for a fresh
///   post-#416 flat install (or the legacy
///   ``…/quickstart-models/<alias>/<alias>/`` for a pre-#416 nested
///   install). We
///   intentionally do NOT populate ``blobs/`` — HF Hub's
///   ``scan_cache_dir`` is tolerant of non-symlinked snapshot files
///   (the "non-blob-deduped" path), and the inner files are already on
///   the user's disk; copying them would double the on-disk footprint
///   for zero benefit.
///
/// ## Idempotency
///
/// Three states count as "already correct":
///
/// 1. ``models--<owner>--<name>/`` is a regular directory we didn't
///    create (the user already ran a real ``snapshot_download``). The
///    user's real cache wins — we don't touch it.
/// 2. ``models--<owner>--<name>/`` is our prior stub AND every leaf
///    symlink still resolves to the live Quickstart install. No-op.
/// 3. ``models--<owner>--<name>/`` is our prior stub but the
///    Quickstart install has been moved / re-extracted / partially
///    pruned (any leaf symlink points wrong or dangles, or the marker
///    file's content drifted). We rebuild only the stub directories
///    we own — never touching adjacent caches.
enum QuickstartModel {

    /// Per-alias Quickstart definition. ``alias`` is the
    /// ``rapid-mlx``-side name the picker shows; ``repoID`` is the
    /// HuggingFace coordinate the sidecar's ``snapshot_download``
    /// resolves against.
    ///
    /// New entries: add to ``knownAliases`` and ship a matching
    /// tarball entry in ``scripts/build-model-tarball.sh``. The two
    /// halves are intentionally co-located in the same release PR so
    /// the catalog never references a tarball the slim DMG can't
    /// install.
    struct Spec: Sendable, Equatable {
        let alias: String
        let repoID: String

        var cacheDirName: String {
            "models--" + repoID.replacingOccurrences(of: "/", with: "--")
        }
    }

    /// Aliases the Quickstart tarball ships weights for. Slim DMGs
    /// only carry ``bonsai-1.7b-2bit`` (the same alias bundled in the
    /// full DMG via ``BundledModel``) — the two paths converge on the
    /// same first-launch UX so users can't tell which DMG variant they
    /// downloaded. Kept in lockstep with the release-pipeline pin
    /// (``release.yml`` ``MODEL_ALIAS_CONST``) and the tarball builder
    /// default (``scripts/build-model-tarball.sh``).
    ///
    /// Future Quickstart additions go HERE (and in the tarball
    /// builder), keyed by alias so ``installAllSnapshotSymlinks`` can
    /// fan out without per-alias plumbing.
    static let knownAliases: [String: Spec] = [
        "bonsai-1.7b-2bit": Spec(
            alias: "bonsai-1.7b-2bit",
            repoID: "prism-ml/Ternary-Bonsai-1.7B-mlx-2bit"
        ),
    ]

    /// Revision marker we write into ``refs/main``. Chosen to be
    /// human-recognisable in ``ls`` output ("snapshots/quickstart/")
    /// so an operator inspecting the HF cache by hand can tell our
    /// fabricated stub apart from a real HF Hub revision SHA without
    /// loading any tools.
    ///
    /// The exact string doesn't matter for correctness (HF Hub's
    /// ``snapshot_download(revision="main")`` reads ``refs/main`` then
    /// resolves ``snapshots/<that-value>/``), but stability matters:
    /// changing it across versions would make existing stubs look
    /// "stale" and force every launch to re-link. Pinned via
    /// ``QuickstartModelTests`` so a future refactor that wants to
    /// rename it has to do so deliberately.
    static let stubRevisionMarker: String = "quickstart"

    // MARK: - Install root resolution

    /// Per-process install root for Quickstart models. Matches the
    /// path ``BootstrapCoordinator`` hands to ``ModelInstaller`` at
    /// ``Bootstrapper/BootstrapCoordinator.swift`` line 2126.
    ///
    /// Resolution goes through ``ApplicationSupportLocator`` so a
    /// dogfood / test launch with ``HOME`` overridden lays the
    /// install (and consequently the HF cache stub) under the
    /// overridden path. Pre-#419/#420 fix used
    /// ``FileManager.urls(for:.applicationSupportDirectory)`` which
    /// resolved via ``getpwuid`` and ignored ``$HOME`` — that
    /// inconsistency would cause this launch-time call to read from
    /// the real ``~/Library`` while the BootstrapCoordinator
    /// commit-success hook reads from the overridden path. They MUST
    /// agree, otherwise the stub fabrication would target the wrong
    /// cache.
    static var defaultInstallRoot: URL? {
        ApplicationSupportLocator.applicationSupportRoot()
            .appendingPathComponent("quickstart-models", isDirectory: true)
    }

    /// Resolve the directory holding the flat weight files for a
    /// given alias under the Quickstart install root. Returns ``nil``
    /// when neither the flat (``<root>/<alias>/<files>``) nor the
    /// legacy nested (``<root>/<alias>/<alias>/<files>`` — pre-#416
    /// F-DGF-V087-03 tarball shape) layouts can be observed.
    ///
    /// #416: ``scripts/build-model-tarball.sh`` now packs FLAT, so a
    /// FRESH install lands at ``<root>/<alias>/<files>`` and resolves
    /// via the preferred branch below. The nested branch is retained
    /// as a BACK-COMPAT fallback: machines that already installed a
    /// pre-#416 tarball keep the nested shape on disk, and
    /// ``ModelInstaller.stage`` refuses to re-extract over an existing
    /// ``<root>/<alias>`` (``.alreadyInstalled``), so those installs
    /// are never rebuilt and must keep loading via this fallback. Do
    /// NOT remove the nested branch without a hoist migration. The
    /// probe is anchored on ``config.json`` (every HF model has one)
    /// rather than a directory existence check so a stub-only /
    /// partial extraction doesn't get mistaken for an installed model.
    static func resolveFlatModelDir(
        alias: String,
        installRoot: URL,
        fileManager: FileManager = .default
    ) -> URL? {
        let outer = installRoot.appendingPathComponent(alias, isDirectory: true)
        // Flat (preferred shape — post-#416 tarballs pack flat, so a
        // fresh install lands here).
        let flatConfig = outer.appendingPathComponent("config.json")
        if fileManager.fileExists(atPath: flatConfig.path) {
            return outer
        }
        // Legacy nested (pre-#416 tarball shape — top-level dir matched
        // the alias, so commit produced ``<root>/<alias>/<alias>/<files>``).
        // Retained for back-compat with installs made before #416.
        let nested = outer.appendingPathComponent(alias, isDirectory: true)
        let nestedConfig = nested.appendingPathComponent("config.json")
        if fileManager.fileExists(atPath: nestedConfig.path) {
            return nested
        }
        return nil
    }

    // MARK: - Outcome

    /// Outcome of a single ``installSnapshotSymlink`` call. Mirrors
    /// the shape of ``BundledModel.InstallOutcome`` so the two
    /// installers can share a logging style.
    enum InstallOutcome: Equatable {
        /// A real (non-stub) HF cache entry exists at the target, OR
        /// our prior stub is fully intact + still points at the same
        /// flat model dir. No filesystem mutation.
        case alreadyPresent
        /// Freshly created the stub on this call (refs/main +
        /// snapshots/<rev>/<symlinks>).
        case installed
        /// Our prior stub was incomplete / pointed at a different
        /// flat model dir / was missing the refs marker. Rebuilt only
        /// the stub directories we own (``refs/`` + ``snapshots/``).
        case relinked
        /// No Quickstart install under the install root for this
        /// alias (slim-DMG never ran, or user pruned the install).
        case noQuickstartModel
        /// The Quickstart source is gone and the stale HF-cache stub we
        /// previously fabricated was removed. A real cache entry is never
        /// eligible for this cleanup.
        case removedStaleStub
        /// Resolving the user HF cache failed (no HOME / HF_HOME).
        case userCacheUnavailable
        /// FileManager raised during mkdir / symlink / write.
        /// Stringified so the caller can surface in logs without
        /// ``Error`` plumbing.
        case failed(String)
    }

    // MARK: - Install

    /// Convenience: install stubs for every known alias whose
    /// Quickstart model is on disk. Production callers (ContentView
    /// task entry + BootstrapCoordinator post-commit hook) use this
    /// no-arg form; tests inject ``installRoot`` / ``environment`` /
    /// ``fileManager`` to exercise corner cases without touching the
    /// real Application Support tree.
    ///
    /// Returns ``[alias: outcome]`` for every aliased subdir we
    /// considered (including ``.noQuickstartModel`` for known aliases
    /// not present on disk yet). Caller can ignore the map or log it
    /// at debug; the production wires discard it because every
    /// branch is recoverable.
    @discardableResult
    static func installAllSnapshotSymlinks(
        installRoot: URL? = QuickstartModel.defaultInstallRoot,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        fileManager: FileManager = .default
    ) -> [String: InstallOutcome] {
        guard let installRoot = installRoot else {
            return knownAliases.mapValues { _ in .noQuickstartModel }
        }
        var results: [String: InstallOutcome] = [:]
        for (alias, spec) in knownAliases {
            results[alias] = installSnapshotSymlink(
                spec: spec,
                installRoot: installRoot,
                environment: environment,
                fileManager: fileManager
            )
        }
        return results
    }

    /// Single-alias form. Public for the BootstrapCoordinator commit
    /// hook (which knows exactly which alias was just installed and
    /// avoids re-scanning unrelated aliases).
    static func installSnapshotSymlink(
        spec: Spec,
        installRoot: URL,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        fileManager: FileManager = .default
    ) -> InstallOutcome {
        let flatModelDir = resolveFlatModelDir(
            alias: spec.alias,
            installRoot: installRoot,
            fileManager: fileManager
        )
        guard let userCache = BundledModel.userHFCacheURL(environment: environment) else {
            return .userCacheUnavailable
        }
        let cacheDir = userCache.appendingPathComponent(spec.cacheDirName, isDirectory: true)
        guard let flatModelDir else {
            return removeStaleStubIfOwned(cacheDir: cacheDir, fileManager: fileManager)
                ?? .noQuickstartModel
        }
        do {
            try fileManager.createDirectory(
                at: userCache,
                withIntermediateDirectories: true,
                attributes: nil
            )
        } catch {
            return .failed("create user cache dir: \(error.localizedDescription)")
        }

        // Three "already correct" shapes — return early without
        // mutation. Order matters:
        //   1. ``cacheDir`` is itself a symlink — ALWAYS user-placed.
        //      We never create the cache-dir as a symlink (writeStub
        //      always mkdir's it via createDirectory(... refsDir ...,
        //      withIntermediateDirectories: true)), so a symlink at
        //      this path can only have been placed deliberately
        //      (e.g. ``ln -s /Volumes/Models/cache/... models--...``
        //      to redirect the cache to a network drive). Following
        //      it on rebuild would let our writeStub's
        //      ``removeItem(at: refsDir)`` delete refs/snapshots
        //      inside the symlink target — exactly the data-loss
        //      hazard codex r1 flagged. Treat it as the user's own
        //      and leave it untouched, even if dangling.
        //   2. ``cacheDir`` is a real directory we did NOT create
        //      (real HF Hub download). Leave it alone — user's
        //      real cache always wins.
        //   3. Our prior stub — validate against the live flat model
        //      dir; rebuild only on drift.
        let attrs = try? fileManager.attributesOfItem(atPath: cacheDir.path)
        if let type = attrs?[.type] as? FileAttributeType {
            if type == .typeSymbolicLink {
                // Shape 1 — user-placed symlink. Never touch.
                return .alreadyPresent
            }
            if type == .typeDirectory && !isOurStub(cacheDir: cacheDir, fileManager: fileManager) {
                // Shape 2 — user-downloaded snapshot.
                return .alreadyPresent
            }
            // Shape 3 — our prior stub OR a stub-shaped directory.
            if stubIsIntact(
                cacheDir: cacheDir,
                flatModelDir: flatModelDir,
                fileManager: fileManager
            ) {
                return .alreadyPresent
            }
            // Stub is stale / partial / wrong-target. Rebuild —
            // safe to do removeItem here because we've proven
            // cacheDir is a real directory (not a symlink) AND it
            // matches our stub fingerprint.
            switch writeStub(
                cacheDir: cacheDir,
                flatModelDir: flatModelDir,
                fileManager: fileManager,
                rebuilding: true
            ) {
            case .success: return .relinked
            case .failure(let message): return .failed(message)
            }
        }

        // Fresh install — no entry at all.
        switch writeStub(
            cacheDir: cacheDir,
            flatModelDir: flatModelDir,
            fileManager: fileManager,
            rebuilding: false
        ) {
        case .success: return .installed
        case .failure(let message): return .failed(message)
        }
    }

    // MARK: - Private — stub identity + write

    /// Detect "this is our fabricated stub, not a real HF Hub
    /// download" by the unique combination ``refs/main`` containing
    /// ``stubRevisionMarker`` AND ``snapshots/<stubRevisionMarker>/``
    /// existing. A real HF Hub snapshot writes a 40-char hex SHA into
    /// ``refs/main``, so the marker string is a reliable
    /// discriminator.
    private static func isOurStub(cacheDir: URL, fileManager: FileManager) -> Bool {
        let refsMain = cacheDir
            .appendingPathComponent("refs", isDirectory: true)
            .appendingPathComponent("main")
        guard markerMatches(at: refsMain) else { return false }
        let snapshotDir = cacheDir
            .appendingPathComponent("snapshots", isDirectory: true)
            .appendingPathComponent(stubRevisionMarker, isDirectory: true)
        var isDir: ObjCBool = false
        return fileManager.fileExists(atPath: snapshotDir.path, isDirectory: &isDir) && isDir.boolValue
    }

    private static func markerMatches(at refsMain: URL) -> Bool {
        guard let data = try? Data(contentsOf: refsMain),
              let content = String(data: data, encoding: .utf8) else {
            return false
        }
        return content.trimmingCharacters(in: .whitespacesAndNewlines) == stubRevisionMarker
    }

    /// Remove only the two directories this installer owns when its source
    /// model disappeared. Keeping an unrelated ``blobs`` directory avoids
    /// deleting data from a partial/real Hub download that happened to share
    /// the cache entry after our stub was created.
    private static func removeStaleStubIfOwned(
        cacheDir: URL,
        fileManager: FileManager
    ) -> InstallOutcome? {
        let attrs = try? fileManager.attributesOfItem(atPath: cacheDir.path)
        guard attrs?[.type] as? FileAttributeType == .typeDirectory else {
            return nil
        }
        let refs = cacheDir.appendingPathComponent("refs", isDirectory: true)
        let refsMain = refs.appendingPathComponent("main")
        let snapshots = cacheDir.appendingPathComponent("snapshots", isDirectory: true)
        let quickstartSnapshot = snapshots.appendingPathComponent(
            stubRevisionMarker, isDirectory: true
        )
        guard markerMatches(at: refsMain) else { return nil }
        // Never traverse a cache-internal directory symlink. The marker read
        // above can follow one, so ownership alone is insufficient for safe
        // deletion; both parents must lstat as real directories.
        let refsAttrs = try? fileManager.attributesOfItem(atPath: refs.path)
        guard refsAttrs?[.type] as? FileAttributeType == .typeDirectory else {
            return .failed("stale Quickstart refs path is not a real directory; refusing cleanup")
        }
        let snapshotsAttrs = try? fileManager.attributesOfItem(atPath: snapshots.path)
        if let snapshotsType = snapshotsAttrs?[.type] as? FileAttributeType,
           snapshotsType != .typeDirectory {
            return .failed("stale Quickstart snapshots path is not a real directory; refusing cleanup")
        }
        do {
            // Re-lstat immediately before mutation. A sync daemon may swap
            // any checked directory between ownership validation and here.
            let rootRecheck = try fileManager.attributesOfItem(atPath: cacheDir.path)
            let refsRecheck = try fileManager.attributesOfItem(atPath: refs.path)
            guard rootRecheck[.type] as? FileAttributeType == .typeDirectory,
                  refsRecheck[.type] as? FileAttributeType == .typeDirectory else {
                return .failed("stale Quickstart cache path changed during cleanup; refusing mutation")
            }
            if snapshotsAttrs != nil {
                let snapshotsRecheck = try fileManager.attributesOfItem(atPath: snapshots.path)
                guard snapshotsRecheck[.type] as? FileAttributeType == .typeDirectory else {
                    return .failed("stale Quickstart snapshots path changed during cleanup; refusing mutation")
                }
            }
            // A previous attempt may already have removed the snapshot. The
            // marker remains the retry token until every destructive step is
            // complete.
            if (try? fileManager.attributesOfItem(atPath: quickstartSnapshot.path)) != nil {
                try fileManager.removeItem(at: quickstartSnapshot)
            }
            // Keep the ownership marker until the destructive step succeeds;
            // a transient snapshot-removal failure must remain retryable on
            // the next launch.
            try fileManager.removeItem(at: refsMain)
            if try fileManager.contentsOfDirectory(atPath: refs.path).isEmpty {
                try fileManager.removeItem(at: refs)
            }
            if snapshotsAttrs != nil,
               try fileManager.contentsOfDirectory(atPath: snapshots.path).isEmpty {
                try fileManager.removeItem(at: snapshots)
            }
            let remaining = try fileManager.contentsOfDirectory(atPath: cacheDir.path)
            if remaining.isEmpty {
                try fileManager.removeItem(at: cacheDir)
            }
            return .removedStaleStub
        } catch {
            return .failed("remove stale Quickstart stub: \(error.localizedDescription)")
        }
    }

    /// Verify every leaf symlink in our stub still points at the
    /// expected file inside the current flat model dir. Returns false
    /// on any of: missing snapshot dir, missing leaf, leaf not a
    /// symlink, leaf destination drift, on-disk file in flat dir not
    /// represented as a leaf.
    private static func stubIsIntact(
        cacheDir: URL,
        flatModelDir: URL,
        fileManager: FileManager
    ) -> Bool {
        let snapshotDir = cacheDir
            .appendingPathComponent("snapshots", isDirectory: true)
            .appendingPathComponent(stubRevisionMarker, isDirectory: true)
        guard let flatFiles = try? fileManager.contentsOfDirectory(
            at: flatModelDir,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else {
            return false
        }
        let flatNames = Set(flatFiles.map { $0.lastPathComponent })
        guard let snapFiles = try? fileManager.contentsOfDirectory(
            at: snapshotDir,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else {
            return false
        }
        let snapNames = Set(snapFiles.map { $0.lastPathComponent })
        guard flatNames == snapNames else { return false }
        for name in flatNames {
            let leaf = snapshotDir.appendingPathComponent(name)
            let attrs = try? fileManager.attributesOfItem(atPath: leaf.path)
            guard let type = attrs?[.type] as? FileAttributeType,
                  type == .typeSymbolicLink else {
                return false
            }
            guard let dest = try? fileManager.destinationOfSymbolicLink(atPath: leaf.path) else {
                return false
            }
            let expected = flatModelDir.appendingPathComponent(name).path
            if dest != expected {
                return false
            }
        }
        return true
    }

    private enum WriteResult {
        case success
        case failure(String)
    }

    /// Materialize (or re-materialize) the stub. On ``rebuilding``
    /// we tear down only ``refs/`` and ``snapshots/`` — leaving any
    /// ``blobs/`` from a partial real download untouched, on the
    /// chance the user is mid-migration from a real cache to our
    /// stub-managed one and we'd otherwise destroy data we don't own.
    private static func writeStub(
        cacheDir: URL,
        flatModelDir: URL,
        fileManager: FileManager,
        rebuilding: Bool
    ) -> WriteResult {
        let refsDir = cacheDir.appendingPathComponent("refs", isDirectory: true)
        let snapshotsDir = cacheDir.appendingPathComponent("snapshots", isDirectory: true)
        let snapshotDir = snapshotsDir.appendingPathComponent(stubRevisionMarker, isDirectory: true)

        if rebuilding {
            // Codex r2 MINOR — TOCTOU defense-in-depth: the
            // installSnapshotSymlink guard above proved cacheDir was
            // not a symlink at check time, but an external actor
            // (user, sync daemon, mount-point flip) could have
            // swapped a real directory for a symlink between then
            // and now. Re-lstat here so a removeItem on refsDir /
            // snapshotsDir can't accidentally follow a freshly-
            // installed cacheDir-level symlink prefix into a victim
            // tree. attributesOfItem does NOT follow symlinks at the
            // queried path (vs fileExists which does), so a symlink
            // newly placed at cacheDir reports .typeSymbolicLink
            // here exactly as we want.
            let recheck = try? fileManager.attributesOfItem(atPath: cacheDir.path)
            if let recheckedType = recheck?[.type] as? FileAttributeType,
               recheckedType == .typeSymbolicLink {
                return .failure("cache-dir was swapped to a symlink mid-flight; refusing rebuild to avoid data loss")
            }
            // Only tear down the dirs we own. Catch + swallow per
            // dir; if a removal fails the subsequent mkdir/create
            // will surface the real error.
            try? fileManager.removeItem(at: refsDir)
            try? fileManager.removeItem(at: snapshotsDir)
        }

        do {
            try fileManager.createDirectory(at: refsDir, withIntermediateDirectories: true)
        } catch {
            return .failure("create refs dir: \(error.localizedDescription)")
        }
        let refsMain = refsDir.appendingPathComponent("main")
        do {
            try Data(stubRevisionMarker.utf8).write(to: refsMain, options: [.atomic])
        } catch {
            return .failure("write refs/main: \(error.localizedDescription)")
        }
        do {
            try fileManager.createDirectory(at: snapshotDir, withIntermediateDirectories: true)
        } catch {
            return .failure("create snapshot dir: \(error.localizedDescription)")
        }
        // Enumerate the flat model dir and lay down absolute-path
        // symlinks. Absolute (not relative) so a future user-side
        // ``mv`` of the snapshot dir doesn't silently break the link.
        let flatFiles: [URL]
        do {
            flatFiles = try fileManager.contentsOfDirectory(
                at: flatModelDir,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            )
        } catch {
            return .failure("enumerate flat model dir: \(error.localizedDescription)")
        }
        for enumerated in flatFiles {
            // ``contentsOfDirectory(at:)`` canonicalises the URL
            // (e.g. ``/var/folders/...`` → ``/private/var/folders/...``
            // on macOS), but ``stubIsIntact`` compares the leaf's
            // symlink destination against ``flatModelDir.appendingPath
            // Component(name).path`` — i.e. the caller's path
            // representation. If those two disagree, every idempotent
            // call would observe drift and pointlessly relink. Rebuild
            // the source URL via the caller's flatModelDir so the
            // stored symlink destination matches what stubIsIntact
            // will look for.
            let basename = enumerated.lastPathComponent
            let source = flatModelDir.appendingPathComponent(basename)
            let leaf = snapshotDir.appendingPathComponent(basename)
            do {
                try fileManager.createSymbolicLink(at: leaf, withDestinationURL: source)
            } catch {
                return .failure("symlink \(basename): \(error.localizedDescription)")
            }
        }
        return .success
    }
}
