import Foundation

/// Pure-data pre-flight check for "does the user have enough free
/// disk to land a model download?"
///
/// ## Why this exists
///
/// PR #338 swapped the Quickstart default alias from
/// ``gemma3-1b-qat-4bit`` (~700 MB) to ``qwen3.5-4b-4bit`` (~2.3 GB).
/// F-LWT-1 then swapped to ``qwen3-0.6b-4bit`` (~400 MB); 2026-07-10
/// swapped to ``bonsai-1.7b-2bit`` (~0.5 GB ternary); 2026-08-05
/// swapped to ``lfm2.5-1b-4bit`` (~0.6 GB 4-bit) after Bonsai was found
/// to degenerate on plain chat — see
/// ``QuickstartCoordinator.defaultChoice`` for the receipt.
/// A user with a near-full disk who clicks Get started watches the bar
/// crawl, then either:
///
///   * the download fails partway (corrupt HF cache → forced retry),
///     or
///   * the volume fills and the rest of the system goes wonky.
///
/// We can't *prevent* either shape from upstream (the download tool
/// runs inside ``rapid-mlx pull`` and writes incrementally to
/// ``~/.cache/huggingface/hub/``), but we can read free space BEFORE
/// kicking the subprocess off and surface a truthful warning so the
/// user gets to make an informed call.
///
/// ## Why warn-only (not block)
///
/// Mature competitors do not block on free-space estimates:
///
///   * **LM Studio** shows a yellow "low space" badge in the download
///     dialog but still lets you proceed.
///   * **Ollama** prints a stderr warning and continues — the CLI never
///     refuses a `pull`.
///
/// statfs free-space numbers are also routinely wrong on macOS:
/// purgeable space, snapshot-pinned space, and APFS reserved blocks
/// can each make the kernel under- OR over-report. A hard block built
/// on top of that data would deny the download to users who actually
/// have headroom. The user owns the call; we just surface signal.
///
/// ## Why a separate type (not a free function in ``QuickstartView``)
///
/// Splitting the data probe (``freeBytes``) from the decision
/// (``decide``) lets the unit test pin the decision truth table
/// without touching the filesystem, and lets a future caller (e.g. a
/// pre-flight check baked into ``DownloadManager.startDownload`` so
/// every alias pull benefits, not just Quickstart) reuse the decision
/// logic without duplicating the threshold constant.
enum DiskSpaceProbe {

    /// Minimum free-space threshold for a small Quickstart download.
    ///
    /// Sized for the Quickstart default ``lfm2.5-1b-4bit``
    /// (~0.6 GB on disk):
    ///
    /// Every operand below is binary (GiB), matching the constant itself
    /// (``2 * 1024³``) — mixing a decimal download figure into a binary
    /// budget is how a threshold silently stops covering what it claims.
    ///
    ///   * **0.622 GiB** final footprint — measured, ``du -sm`` reported
    ///     637 MiB for the pulled snapshot, AND
    ///   * ~1.5× transient peak during chunk fetch + dedupe (HF's
    ///     downloader writes an incomplete `.bin` alongside the
    ///     sharded snapshot dir before atomically moving), AND
    ///   * 1 GiB headroom for the rest of the OS, swap, and Console
    ///     log churn during a 1-2 minute pull on a slow link.
    ///
    /// ``0.622 × 1.5 + 1 = 1.933 GiB``, inside the ``2 GiB`` threshold,
    /// which stays put so the surfaced "needs ~Y GB" copy remains a clean
    /// round number. Earlier starters derived the same 2 GiB from smaller
    /// footprints, so the slack has narrowed to **~0.067 GiB (~68 MiB)**.
    ///
    /// That is thin enough that the next bump probably breaks it: solving
    /// ``x × 1.5 + 1 = 2`` puts the ceiling at **0.667 GiB (~683 MiB)** on
    /// disk. If the Quickstart default (see
    /// ``QuickstartCoordinator.defaultChoice``) moves again, re-derive this
    /// constant — do not assume 2 GiB still covers it.
    static let quickstartRequiredBytes: Int64 = 2 * 1024 * 1024 * 1024

    /// Selected-model budget: 1.5× transient download peak plus 1 GiB for
    /// macOS, rounded up to a whole GiB so the warning copy stays legible.
    /// The historical 2 GiB threshold remains the floor for small models.
    static func requiredBytes(downloadBytes: Int64) -> Int64 {
        guard downloadBytes > 0 else { return quickstartRequiredBytes }
        let gib = Double(1 << 30)
        let budget = Double(downloadBytes) * 1.5 + gib
        let rounded = Int64((budget / gib).rounded(.up)) * Int64(1 << 30)
        return max(quickstartRequiredBytes, rounded)
    }

    /// Outcome of ``decide``. Used to drive both the UI banner and a
    /// future telemetry signal (how often does the warning fire for
    /// real users?).
    enum Decision: Equatable {
        /// Free space comfortably exceeds the requirement, or we
        /// couldn't probe free space at all. Either way, the caller
        /// should proceed without surfacing a warning — the
        /// can't-probe case errs on the side of letting the user
        /// continue (HF will surface the real error if the disk truly
        /// can't hold the download).
        case ok
        /// Free space falls below the requirement. The caller should
        /// surface a non-blocking banner with the embedded numbers
        /// so the user can decide whether to continue. Payload is
        /// raw bytes; the view layer formats to GB.
        case warn(freeBytes: Int64, requiredBytes: Int64)
    }

    /// Pure decision: given measured free bytes and a required-bytes
    /// threshold, should the caller surface a low-disk warning?
    ///
    /// Returns ``.ok`` when ``freeBytes`` is ``nil`` (probe failed —
    /// see ``freeBytes(forPath:)``) so a transient FileManager error
    /// can never gate the user out of starting the download.
    static func decide(freeBytes: Int64?, requiredBytes: Int64) -> Decision {
        guard let freeBytes else { return .ok }
        // Defensive: a non-positive requirement degenerates to "no
        // warning ever" — callers shouldn't pass <= 0 but the math
        // shouldn't trap if they do.
        guard requiredBytes > 0 else { return .ok }
        if freeBytes >= requiredBytes {
            return .ok
        }
        return .warn(freeBytes: freeBytes, requiredBytes: requiredBytes)
    }

    /// Read the volume free-space at ``path`` via
    /// ``FileManager.attributesOfFileSystem(forPath:)``.
    ///
    /// Returns ``nil`` when the FileManager call throws (path is on a
    /// volume that vanished mid-launch, sandbox blocks the statfs, the
    /// path resolves through a broken symlink, etc.). The caller MUST
    /// treat ``nil`` as "no signal" — see ``decide``'s ``nil`` branch.
    ///
    /// ## Path resolution
    ///
    /// The caller passes the HuggingFace hub cache root (typically
    /// ``~/.cache/huggingface/hub/``). ``FileManager`` resolves symlinks
    /// internally and reports the underlying volume's free space, so a
    /// user who's redirected ``~/.cache`` to a secondary disk via
    /// symlink (a real pattern on Mac mini setups where ``$HOME`` is on
    /// the boot SSD but ``~/.cache`` points at a Thunderbolt SSD) sees
    /// the correct number, not the boot disk's.
    ///
    /// Walks up the path to the nearest existing ancestor when the
    /// cache root itself doesn't exist on disk yet (a brand-new install
    /// never had ``~/.cache/huggingface/`` created). The walk STOPS at
    /// volume boundaries — see ``crossesVolumeBoundary`` rationale —
    /// so a path under a missing ``/Volumes/<name>`` mount or a broken
    /// external-disk symlink degrades to ``nil`` instead of silently
    /// reporting the boot volume's free space (which would be wildly
    /// wrong: the boot SSD has ~200 GB; the user redirected to an
    /// external precisely BECAUSE they wanted multi-TB headroom).
    /// Codex r1 MAJOR.
    static func freeBytes(forPath path: String) -> Int64? {
        let fm = FileManager.default
        if let value = readFreeBytes(fm: fm, path: path) {
            return value
        }
        // Codex r2 MAJOR: resolve symlinks BEFORE the boundary check
        // so a dangling ``~/.cache → /Volumes/ExternalSSD/cache`` link
        // (external unmounted) is treated the same as
        // ``HF_HOME=/Volumes/ExternalSSD/hf``. Without resolution,
        // ``originalPath`` stays under ``$HOME`` and
        // ``crossesVolumeBoundary`` returns false, the walk continues
        // up to ``$HOME``, and we report the boot volume's free space
        // — exactly the wrong number for the external-cache pattern
        // the redirect was set up to solve.
        //
        // ``URL.resolvingSymlinksInPath`` does NOT resolve dangling
        // symlinks (verified on macOS 14 / 15 / 26 — Foundation only
        // dereferences links that actually point somewhere). We have
        // to walk the path components from root and call
        // ``destinationOfSymbolicLink`` on each so a broken
        // ``~/.cache → /Volumes/ExternalSSD/cache`` is rewritten to
        // its dangling destination before the boundary check.
        let resolved = resolveDanglingSymlinks(path: path, fm: fm)
        // Cache root may not exist yet on first launch. Walk up to
        // the nearest existing ancestor so the statfs lands on the
        // owning volume — but STOP if we'd cross a volume boundary
        // (broken symlink to unmounted /Volumes, HF_HOME pointing at
        // a not-yet-created external-disk path).
        var url = URL(fileURLWithPath: resolved, isDirectory: true)
        while url.path != "/" {
            url.deleteLastPathComponent()
            if crossesVolumeBoundary(originalPath: resolved, walkedTo: url.path) {
                // Original intent was an external / mounted volume
                // that doesn't exist yet. Don't substitute the boot
                // volume's free-space — the surfaced number would
                // contradict the user's actual storage layout. Fail
                // open: caller sees ``nil`` → ``Decision.ok`` → no
                // warning (HF will surface the real error if the
                // external disk truly isn't there at download time).
                return nil
            }
            if fm.fileExists(atPath: url.path) {
                if let value = readFreeBytes(fm: fm, path: url.path) {
                    return value
                }
                break
            }
        }
        return nil
    }

    /// Walk the path components of ``path`` from root and resolve any
    /// symlink along the way — INCLUDING dangling symlinks that point
    /// at a not-yet-existing destination.
    ///
    /// ``URL.resolvingSymlinksInPath`` only dereferences links whose
    /// destination actually exists on disk, which is the wrong shape
    /// for the FU-4 external-cache pattern: a user with
    /// ``~/.cache → /Volumes/ExternalSSD/cache`` whose external disk
    /// is unmounted has a dangling link, and ``resolvingSymlinksInPath``
    /// returns the ``$HOME``-rooted path unchanged. Walking + calling
    /// ``destinationOfSymbolicLink`` per component reaches the
    /// dangling destination so the boundary check fires.
    ///
    /// Handles **chained** symlinks too: when resolving a symlink leaf
    /// produces a path whose intermediate components are themselves
    /// symlinks (e.g. ``~/.cache → some/link/cache`` and
    /// ``some/link → /Volumes/Missing``), the resolved path is fed
    /// back through the walker so all link layers are unwrapped. Total
    /// iterations bounded by ``maxIterations`` so cycles never trap.
    /// Codex r3 MINOR.
    ///
    /// Returns the original path if the iteration budget is exhausted
    /// (defensive fail-shut — the caller's ``Decision.ok`` fail-open
    /// kicks in if the boundary check can't reach a verdict).
    static func resolveDanglingSymlinks(path: String, fm: FileManager = .default) -> String {
        let maxIterations = 128
        var current = path
        for _ in 0..<maxIterations {
            let (resolved, changed) = resolveOnePass(path: current, fm: fm)
            if !changed { return (resolved as NSString).standardizingPath }
            current = resolved
        }
        // Cycle / pathological depth — bail.
        return path
    }

    /// Single forward pass of the component walker. Returns the
    /// possibly-rewritten path AND a "did anything change" flag the
    /// outer loop uses to decide whether another pass is needed.
    private static func resolveOnePass(path: String, fm: FileManager) -> (String, Bool) {
        let components = (path as NSString).pathComponents
        var current = "/"
        var didRewrite = false
        for component in components {
            if component == "/" || component.isEmpty { continue }
            let next = (current as NSString).appendingPathComponent(component)
            if let dest = try? fm.destinationOfSymbolicLink(atPath: next) {
                didRewrite = true
                if (dest as NSString).isAbsolutePath {
                    current = dest
                } else {
                    // Relative destinations resolve against the link's
                    // containing directory (POSIX symlink semantics).
                    current = (current as NSString).appendingPathComponent(dest)
                }
            } else {
                current = next
            }
        }
        return (current, didRewrite)
    }

    /// Convenience: probe the volume backing the HuggingFace hub cache
    /// for the current process's environment. Equivalent to
    /// ``freeBytes(forPath: BundledModel.userHFCacheURL(...).path)`` but
    /// keeps the call site at the Quickstart layer single-line.
    ///
    /// Returns ``nil`` when ``BundledModel.userHFCacheURL`` itself
    /// returns ``nil`` (no ``HOME`` / ``HF_HOME``) — the empty
    /// environment is a pathological shape we don't try to recover
    /// from; the caller treats it as ``Decision.ok``.
    static func freeBytesForHFCache(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int64? {
        guard let url = BundledModel.userHFCacheURL(environment: environment) else {
            return nil
        }
        return freeBytes(forPath: url.path)
    }

    // MARK: - Private

    /// Heuristic: does walking up from ``originalPath`` to
    /// ``walkedTo`` cross a volume boundary on macOS?
    ///
    /// The conservative answer is "yes" for any path that explicitly
    /// roots itself on a non-boot volume:
    ///
    ///   * Anything under ``/Volumes/<name>`` is a mounted external
    ///     (or DMG); walking past ``/Volumes/<name>`` lands on ``/Volumes``,
    ///     a system directory on the boot volume, free-space numbers
    ///     would describe the wrong disk.
    ///   * Anything under ``/private/var/folders`` or similar system
    ///     roots that the user wouldn't realistically point HF_HOME at;
    ///     not enumerated here because the realistic cache shape is
    ///     either ``$HOME/.cache/...`` or ``/Volumes/<name>/...``.
    ///
    /// Returns ``false`` for the default ``$HOME/.cache/huggingface/...``
    /// shape — the walk-up to ``$HOME`` lands on the same APFS volume
    /// the cache leaf would have lived on, so reporting that volume's
    /// free space is correct.
    ///
    /// Codex r1 MAJOR: without this, a user with a broken
    /// ``~/.cache → /Volumes/ExternalSSD/cache`` symlink (or an
    /// ``HF_HOME=/Volumes/ExternalSSD/hf`` set in the env before the
    /// disk was mounted) would see the boot-volume's free space, which
    /// is exactly the WRONG number for the case the disk-redirect
    /// pattern is trying to solve.
    static func crossesVolumeBoundary(originalPath: String, walkedTo: String) -> Bool {
        let volumesPrefix = "/Volumes/"
        guard originalPath.hasPrefix(volumesPrefix) else {
            // Default cache shape lives under $HOME — no boundary
            // concern, walk-up is safe.
            return false
        }
        // Extract the mount point — "/Volumes/<name>" — and check
        // whether the walk has gone above it. If walkedTo is shorter
        // than "/Volumes/<name>" OR equals "/Volumes" / "/", we've
        // crossed the boundary.
        let trimmed = originalPath.dropFirst(volumesPrefix.count)
        guard let nameEnd = trimmed.firstIndex(of: "/") else {
            // Path was exactly "/Volumes/<name>" with no trailing slash;
            // any walk-up crosses the boundary.
            return walkedTo != originalPath
        }
        let mountPoint = volumesPrefix + trimmed[..<nameEnd]
        // Walk-up is safe as long as we're still under the mount point.
        return !(walkedTo == mountPoint || walkedTo.hasPrefix(mountPoint + "/"))
    }

    private static func readFreeBytes(fm: FileManager, path: String) -> Int64? {
        guard fm.fileExists(atPath: path) else { return nil }
        guard let attrs = try? fm.attributesOfFileSystem(forPath: path) else {
            return nil
        }
        // ``FileAttributeKey.systemFreeSize`` is documented as an
        // ``NSNumber`` wrapping a 64-bit unsigned. Read as ``Int64``:
        // even on the largest Mac volumes shipping today (~16 TB),
        // free-bytes fits in a signed 64. If it ever doesn't, ``Int64``
        // overflow returns nil and the probe degrades to "no signal"
        // — same fail-open behaviour as a FileManager throw.
        guard let number = attrs[.systemFreeSize] as? NSNumber else {
            return nil
        }
        let value = number.int64Value
        return value >= 0 ? value : nil
    }
}
