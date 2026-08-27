import Foundation

/// First-paint UX: a tiny model bundled inside the DMG so the app
/// works the moment the user double-clicks it, without any
/// HuggingFace round-trips.
///
/// ## Why we bundle anything at all
///
/// v0.7.0 shipped with ``RAMBucketedDefault`` choosing a 7+ GB model
/// (``gemma-4-12b-4bit`` on an 18 GB MacBook) as the first-launch
/// default. The user's first impression was a "Downloading 0/9 files"
/// progress bar that either stalled on ``cas-bridge.xethub.hf.co`` or
/// hit the ``huggingface_hub`` 10s read-timeout and never recovered
/// (see #229 / "stuck on download" reports). The chat composer was
/// inactive the entire time. By the time the user had typed "hi" the
/// app had already failed.
///
/// For airgapped builds (``BUNDLE_MODEL=1``) the ~0.6 GB
/// lfm2.5-1b-4bit weights are staged inside the DMG. On first
/// launch the desktop:
///
/// 1. Resolves the bundled model directory at
///    ``Contents/Resources/models/hf-cache/hub/models--mlx-community--LFM2.5-1.2B-Instruct-4bit``
/// 2. Symlinks it into the user's HuggingFace cache
///    (``~/.cache/huggingface/hub/``) the first time the symlink is
///    missing, so the sidecar can ``snapshot_download`` it without a
///    network call.
/// 3. Returns ``bundledAlias`` as the first-launch default so the
///    sidecar gets ``rapid-mlx serve lfm2.5-1b-4bit`` instead of
///    something the user would have to wait minutes to download.
///
/// ## Why lfm2.5-1b-4bit (LFM2.5 1.2B Instruct)
///
/// 2026-08-05: swapped from ``bonsai-1.7b-2bit``. It now stays in lock-step
/// with ``QuickstartCoordinator.lowMemoryChoice``: the hardware-fit automatic
/// starter is larger, while the bundle remains the explicit low-memory and
/// airgapped escape hatch.
///
/// Bonsai was chosen (#1092) on a tool-call eval: 6/6 clean
/// ``tool_calls`` on the 1.7B. That measurement was real but did not
/// cover the thing a starter is actually judged on. On a plain-chat
/// multi-step word problem — no tools — it degenerated 4/4 and
/// terminated 0/4, doubling words within the first line and then
/// looping until it hit ``max_tokens``. See
/// ``QuickstartCoordinator.defaultChoice`` for the full measurements.
///
/// The 1.2B replacement answers correctly and terminates cleanly on
/// every run recorded: 16/16 total, of which 12/12 came from a single
/// controlled repro. Both numbers describe the same result, so quote
/// the 16/16 -- it is the whole sample, not a subset of it. 170 tok/s
/// with no reasoning
/// phase. It is a text-first pick: unlike Bonsai it is not currently
/// listed in ``ToolUseCapability``, so the empty-state capability
/// chips stay hidden until someone measures its tool calls. That is
/// the intended fail-closed posture — do not add it to the known
/// list on the strength of the engine's ``lfm`` parser alone.
///
/// ## Why this lives in ``Server/`` not ``UI/``
///
/// The first-launch alias decision is a server-side concern (drives
/// what ``ServerManager.start(alias:)`` gets called with). The UI
/// just reflects the result. Keeping it next to ``RAMBucketedDefault``
/// + ``ServerManager`` makes that ownership obvious.
enum BundledModel {
    /// Alias the bundled weights resolve to in ``rapid-mlx``'s
    /// ``aliases.json``. Pinned, not derived — bumping the bundled
    /// model is an intentional release action (DMG growth, sniff
    /// re-test, sidecar smoke). Source-of-truth lives in the
    /// submodule at ``third_party/rapid-mlx/vllm_mlx/aliases.json``;
    /// this constant just names the entry we ship weights for.
    static let bundledAlias: String = "lfm2.5-1b-4bit"

    /// HuggingFace repo ID for the bundled weights. The HF cache
    /// layout encodes this as ``models--<owner>--<name>`` on disk.
    /// Kept here next to ``bundledAlias`` so a renamed-upstream
    /// repo doesn't silently break the symlink path.
    static let bundledRepoID: String = "mlx-community/LFM2.5-1.2B-Instruct-4bit"

    /// HF cache directory name for the bundled repo. The HF Hub
    /// snapshot layout uses ``models--<owner>--<name>`` (double-dash
    /// separators). Exposed as a static so tests + the install-
    /// symlink helper agree without re-deriving the rule.
    static var bundledCacheDirName: String {
        "models--" + bundledRepoID.replacingOccurrences(of: "/", with: "--")
    }

    /// Path inside the .app where ``scripts/build.sh`` stages the
    /// pre-downloaded HF Hub snapshot. ``nil`` when running outside
    /// a bundle (unit tests link the executable target directly and
    /// ``Bundle.main.resourceURL`` is the test binary's resource dir,
    /// which doesn't have the bundled snapshot).
    static var bundledSnapshotURL: URL? {
        bundledSnapshotURL(bundleResourceURL: Bundle.main.resourceURL)
    }

    /// Test-injectable form of ``bundledSnapshotURL``. Production
    /// callers use the no-arg accessor above; tests synthesize a
    /// fake ``bundleResourceURL`` pointing at a tmpdir with the
    /// expected sub-tree to exercise the resolution rules without
    /// rebuilding the .app.
    static func bundledSnapshotURL(bundleResourceURL: URL?) -> URL? {
        guard let resourceURL = bundleResourceURL else { return nil }
        let url = resourceURL
            .appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent("hf-cache", isDirectory: true)
            .appendingPathComponent("hub", isDirectory: true)
            .appendingPathComponent(bundledCacheDirName, isDirectory: true)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return nil
        }
        return url
    }

    /// User-side HF cache directory that ``rapid-mlx`` consults at
    /// load time. Defaults to ``~/.cache/huggingface/hub/`` — the
    /// path ``huggingface_hub.constants.HF_HUB_CACHE`` resolves to
    /// when no ``HF_HOME`` / ``HF_HUB_CACHE`` env var is set.
    /// Environment is taken via the argument so tests can pin a
    /// throwaway ``HOME``.
    ///
    /// Precedence:
    ///   0. ``preferredOverride`` — the desktop "Models folder"
    ///      preference (issue #503), when the user pointed Rapid at an
    ///      explicit folder that currently resolves. Wins over every
    ///      env-derived tier so the app's disk view matches where the
    ///      engine reads/writes once the preference is set. Callers pass
    ///      a pre-validated URL (see
    ///      ``ModelsFolderPreference/validatedOverrideURL(defaults:fileManager:)``)
    ///      so an unplugged drive is already resolved to ``nil`` here and
    ///      the env/default fallback applies.
    /// The remaining tiers mirror huggingface_hub's own resolution
    /// (``huggingface_hub.constants``):
    ///   1. ``HF_HUB_CACHE`` — explicit override (any directory).
    ///   2. ``HF_HOME`` + ``/hub`` — explicit HF root.
    ///   3. ``XDG_CACHE_HOME`` + ``/huggingface/hub`` — XDG-aware.
    ///   4. ``HOME`` + ``/.cache/huggingface/hub`` — default.
    static func userHFCacheURL(
        environment: [String: String],
        preferredOverride: URL? = nil
    ) -> URL? {
        if let preferredOverride {
            return preferredOverride
        }
        if let explicit = environment["HF_HUB_CACHE"], !explicit.isEmpty {
            return URL(fileURLWithPath: explicit, isDirectory: true)
        }
        if let hfHome = environment["HF_HOME"], !hfHome.isEmpty {
            return URL(fileURLWithPath: hfHome, isDirectory: true)
                .appendingPathComponent("hub", isDirectory: true)
        }
        if let xdg = environment["XDG_CACHE_HOME"], !xdg.isEmpty {
            return URL(fileURLWithPath: xdg, isDirectory: true)
                .appendingPathComponent("huggingface", isDirectory: true)
                .appendingPathComponent("hub", isDirectory: true)
        }
        if let home = environment["HOME"], !home.isEmpty {
            return URL(fileURLWithPath: home, isDirectory: true)
                .appendingPathComponent(".cache", isDirectory: true)
                .appendingPathComponent("huggingface", isDirectory: true)
                .appendingPathComponent("hub", isDirectory: true)
        }
        return nil
    }

    /// Outcome of ``installBundledSnapshotSymlink`` so callers can
    /// log + tests can assert without crawling the filesystem.
    enum InstallOutcome: Equatable {
        /// Snapshot was already linked / present in the user HF cache;
        /// no work needed.
        case alreadyPresent
        /// We just created the symlink from the user HF cache into the
        /// bundled snapshot.
        case installed
        /// A pre-existing symlink at the target was pointing somewhere
        /// other than the live bundled snapshot (or was dangling), so
        /// we unlinked it and recreated it pointing at the current
        /// snapshot. Surfaces the prior destination (``nil`` when the
        /// link's destination was unresolvable) so the caller can log
        /// "Re-linked stale snapshot from <X>".
        ///
        /// Why a separate case (not ``.installed``): the user moving
        /// ``Rapid-MLX Desktop.app`` between disks is a real-world
        /// scenario we want telemetry on — distinguishing a stale-
        /// symlink re-link from a true first-launch install lets us
        /// see how often the .app gets relocated without adding extra
        /// plumbing.
        case relinked(oldDestination: String?)
        /// Nothing to install — running outside a real .app bundle or
        /// the bundled snapshot directory is missing on disk.
        case noBundledSnapshot
        /// Resolving the user HF cache failed (no HOME / HF_HOME).
        case userCacheUnavailable
        /// FileManager raised during mkdir / symlink. Stringified so
        /// the caller can surface it in logs without ``Error`` plumbing.
        case failed(String)
    }

    /// Idempotently link the bundled snapshot into the user's HF cache.
    /// Safe to call on every launch — when the symlink already exists
    /// (or a real downloaded snapshot is there) we return
    /// ``.alreadyPresent`` without touching disk.
    ///
    /// Why a symlink and not a copy: the bundled snapshot is already
    /// codesigned + sealed inside the .app's resource envelope; a copy
    /// would duplicate ~320 MB on every fresh install for zero benefit.
    /// macOS resolves symlinks transparently inside the HF cache, and
    /// ``huggingface_hub`` doesn't care that the snapshot lives outside
    /// its hub directory.
    ///
    /// Note we link the ``models--<owner>--<name>`` directory itself,
    /// not its contents. The HF Hub cache structure is one directory
    /// per repo; symlinking the whole repo dir is exactly what HF's
    /// ``LocalEntry.symlink_path`` ends up creating on a real download.
    static func installBundledSnapshotSymlink(
        bundleResourceURL: URL? = Bundle.main.resourceURL,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        fileManager: FileManager = .default
    ) -> InstallOutcome {
        guard let snapshot = bundledSnapshotURL(bundleResourceURL: bundleResourceURL) else {
            return .noBundledSnapshot
        }
        guard let userCache = userHFCacheURL(environment: environment) else {
            return .userCacheUnavailable
        }
        let target = userCache.appendingPathComponent(bundledCacheDirName, isDirectory: true)
        // mkdir -p the parent hub/ directory — HF only creates it on
        // the first real download, so on a fresh Mac install it
        // doesn't exist yet.
        do {
            try fileManager.createDirectory(
                at: userCache,
                withIntermediateDirectories: true,
                attributes: nil
            )
        } catch {
            return .failed("create user cache dir: \(error.localizedDescription)")
        }
        // Already present? Two shapes count as "present":
        //   * a regular directory (user already downloaded the model
        //     in a previous session) — leave it alone, the user's own
        //     cache wins so we never overwrite their work.
        //   * a symlink we (or a previous launch) installed — but only
        //     if it STILL points at the current snapshot. If the user
        //     moved Rapid-MLX Desktop.app to a different path the link
        //     now resolves to nowhere, and falling through to
        //     ``createSymbolicLink`` would fail because the path is
        //     already taken (as a dangling link). Re-link in that case.
        //
        // ``attributesOfItem(atPath:)`` does NOT follow symlinks (vs
        // ``fileExists(atPath:)`` which does), so a dangling link still
        // reports ``.typeSymbolicLink`` — exactly what we want here.
        let attrs = try? fileManager.attributesOfItem(atPath: target.path)
        if let type = attrs?[.type] as? FileAttributeType {
            if type == .typeDirectory {
                return .alreadyPresent
            }
            if type == .typeSymbolicLink {
                let existingDest = try? fileManager.destinationOfSymbolicLink(atPath: target.path)
                if existingDest == snapshot.path {
                    return .alreadyPresent
                }
                // Stale or dangling — unlink + recreate.
                // Order matters: unlink THEN createSymbolicLink, so a
                // crash between the two leaves no link rather than the
                // wrong link (a missing link triggers a fresh install
                // on next launch; a wrong link silently breaks HF Hub
                // resolution forever).
                do {
                    try fileManager.removeItem(at: target)
                } catch {
                    return .failed("unlink stale symlink: \(error.localizedDescription)")
                }
                do {
                    try fileManager.createSymbolicLink(
                        at: target,
                        withDestinationURL: snapshot
                    )
                } catch {
                    return .failed("recreate symlink: \(error.localizedDescription)")
                }
                return .relinked(oldDestination: existingDest)
            }
        }
        do {
            try fileManager.createSymbolicLink(
                at: target,
                withDestinationURL: snapshot
            )
        } catch {
            return .failed("create symlink: \(error.localizedDescription)")
        }
        return .installed
    }

    /// First-launch alias decision. Returns ``bundledAlias`` when
    /// (a) the user has no last-served alias (fresh install / explicit
    /// Stop), AND (b) the bundled snapshot is on disk. Otherwise
    /// returns ``nil`` and the caller falls through to its existing
    /// resolution chain (last-served → RAM-bucketed default).
    ///
    /// We intentionally don't override the choice when the user HAS
    /// a last-served alias: a user who already trade-upped to
    /// qwen3.6-35b shouldn't get yanked back to the small starter just
    /// because the bundled snapshot exists.
    static func firstLaunchAlias(
        lastServedAlias: String? = ServerManager.lastServedAlias(),
        bundleResourceURL: URL? = Bundle.main.resourceURL
    ) -> String? {
        guard lastServedAlias == nil else { return nil }
        guard bundledSnapshotURL(bundleResourceURL: bundleResourceURL) != nil else {
            return nil
        }
        return bundledAlias
    }
}
