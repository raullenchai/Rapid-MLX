import Foundation

/// Background polling task that watches a single ``models--<owner>--<repo>``
/// directory under the HuggingFace cache and forwards its on-disk size to a
/// ``DownloadProgress`` instance every few seconds.
///
/// The HuggingFace ``snapshot_download`` tqdm bar — the one ``DownloadProgress``
/// parses today — counts FILES, not bytes. On a 6.8 GB / 11-shard model the
/// outer bar reads "0 of 9 files (0%)" for many minutes while the first
/// shard streams silently to disk. Users (rightly) interpret that as "the
/// download is dead." This monitor closes the gap by sampling the cache
/// dir directly so the UI can render real bytes-on-disk progress
/// independent of tqdm cadence.
///
/// Design notes:
///   * **Background actor / Task.detached, NOT @MainActor.** File-system
///     enumeration on a 7 GB hot cache directory takes 10-100 ms — keeping
///     it off the main thread is non-negotiable. Only the
///     ``progress.applyDiskObservation(bytes:)`` call hops onto
///     ``@MainActor``.
///   * **No shell `du` spawn.** Per the task constraints we use
///     ``FileManager.enumerator``. HF cache dirs hold tens to low-hundreds
///     of files — enumeration is plenty fast, and avoiding ``Process``
///     keeps us out of sandbox / signal-handler corners.
///   * **Hardlink-safe.** HF stores blobs once under ``blobs/<sha>`` and
///     hard-links them into ``snapshots/<rev>/<file>``. Counting BOTH
///     paths would double the total. We enumerate the dir but tally each
///     inode at most once (keyed off ``fileResourceIdentifierKey``), so
///     the snapshot side never inflates the byte count.
///   * **Defensive.** A missing dir, an unreadable dir, or an
///     ``HF_HOME`` redirected somewhere we can't reach all leave the
///     bytes channel at ``nil`` — the UI falls back to the existing
///     file-count copy. Never throws into the caller; never aborts the
///     download.
///   * **Stops cleanly.** ``stop()`` cancels the underlying ``Task``;
///     the polling loop checks ``Task.isCancelled`` after every sleep
///     and on every iteration so a teardown completes within one poll
///     interval (default 3 s).
enum HFCacheByteMonitor {
    /// Default poll cadence. Trade-off: shorter → snappier UI; longer →
    /// less CPU from the directory walk. 3 s matches the SettingsModels
    /// reconciliation panel's hot-poll cadence (500 ms there) order-of-
    /// magnitude — visible motion without burning the battery.
    static let defaultPollInterval: TimeInterval = 3.0

    /// Resolve the per-alias cache directory under ``hubCacheRoot`` for
    /// the given HuggingFace ``owner/repo`` path. Mirrors HF's own
    /// ``cache_dir`` convention: ``models--<owner>--<repo>``.
    ///
    /// Returns ``nil`` when the HF path doesn't parse to ``owner/repo``
    /// or to a bare ``repo`` (some community uploads are unnamespaced —
    /// those still get a single-segment cache dir like ``models--gpt2``).
    /// We bound the directory name to ``ModelCatalog.maxHuggingFaceRepoBytes``
    /// before joining so a pathologically long hf_path can never grow
    /// into a writeable region.
    static func cacheDirectoryURL(
        hubCacheRoot: URL,
        hfPath: String
    ) -> URL? {
        guard let sanitized = ModelCatalog.sanitizedHuggingFaceRepo(hfPath) else {
            return nil
        }
        let dirName = "models--" + sanitized.replacingOccurrences(of: "/", with: "--")
        guard dirName.utf8.count <= ModelCatalog.maxHuggingFaceRepoBytes + "models--".utf8.count else {
            return nil
        }
        return hubCacheRoot.appendingPathComponent(dirName, isDirectory: true)
    }

    /// One-shot recursive byte count of ``url`` that counts each inode
    /// at most once. Exposed for tests so they can pin the hardlink
    /// dedup contract without standing up a poller.
    ///
    /// Returns ``0`` if the directory doesn't exist or can't be
    /// enumerated — callers should treat ``0`` as "no observation yet"
    /// and not forward it to ``DownloadProgress.applyDiskObservation``,
    /// which already rejects zero.
    nonisolated static func directoryByteCount(at url: URL) -> Int64 {
        let keys: Set<URLResourceKey> = [
            .isRegularFileKey,
            .isSymbolicLinkKey,
            .fileSizeKey,
            .fileAllocatedSizeKey,
            .totalFileAllocatedSizeKey,
            .fileResourceIdentifierKey,
        ]
        // HF cache layout: ``models--<owner>--<repo>/blobs/<sha>`` holds
        // the real bytes; ``models--…/snapshots/<rev>/<file>`` are hard-
        // links into the blob store. Naive du-style counting on the
        // whole dir would tally both sides and inflate the observed
        // bytes by 2x for a model that's mid-snapshot. We dedupe by
        // ``fileResourceIdentifierKey`` (inode-equivalent on macOS HFS+
        // / APFS) so each blob contributes exactly once.
        var total: Int64 = 0
        var seenInodes: Set<AnyHashable> = []
        let fm = FileManager.default
        guard fm.fileExists(atPath: url.path) else { return 0 }
        let enumerator = fm.enumerator(
            at: url,
            includingPropertiesForKeys: Array(keys),
            options: [],
            errorHandler: { _, _ in true }
        )
        while let file = enumerator?.nextObject() as? URL {
            guard let values = try? file.resourceValues(forKeys: keys),
                  values.isSymbolicLink != true,
                  values.isRegularFile == true else {
                continue
            }
            if let inodeID = values.fileResourceIdentifier as? (any Hashable) {
                let key = AnyHashable(inodeID)
                if seenInodes.contains(key) { continue }
                seenInodes.insert(key)
            }
            let size = values.totalFileAllocatedSize
                ?? values.fileAllocatedSize
                ?? values.fileSize
                ?? 0
            total += Int64(size)
        }
        return total
    }

    /// Token returned by ``start(...)`` — call ``stop()`` to cancel the
    /// poll loop. Holding the handle keeps no resources beyond the
    /// inner ``Task``; dropping it does NOT auto-cancel (so a forgotten
    /// handle won't silently kill a live monitor), but
    /// ``DownloadManager`` and ``ServerManager`` retain theirs for the
    /// lifetime of the matching subprocess.
    final class Handle: @unchecked Sendable {
        private let task: Task<Void, Never>
        private let firstPoll: Task<Bool, Never>

        init(task: Task<Void, Never>, firstPoll: Task<Bool, Never>) {
            self.task = task
            self.firstPoll = firstPoll
        }

        /// Cancel the polling loop. Idempotent — calling twice is a no-op.
        func stop() {
            task.cancel()
        }

        /// Wait for the first filesystem poll and report whether it published
        /// a positive disk observation. Multiple callers share the same result.
        func waitForFirstPoll() async -> Bool {
            await firstPoll.value
        }

        /// Cancel the poll loop and wait until it has fully exited.
        func stopAndWait() async {
            task.cancel()
            await task.value
            _ = await firstPoll.value
        }

        deinit {
            // Defensive: a Handle that goes out of scope should not
            // leak a forever-polling task even if the owner forgot to
            // call stop().
            task.cancel()
        }
    }

    /// Start polling ``cacheDir`` and forwarding observations to
    /// ``progress``. The first observation lands immediately (the loop
    /// counts before its first sleep) so the UI can flip out of "0 B"
    /// copy as soon as the first chunk hits disk.
    ///
    /// ``isCancelled`` is an optional stop predicate evaluated each
    /// iteration in ADDITION to ``Task.isCancelled``. Useful for the
    /// ``ServerManager`` integration where we want to stop polling
    /// once the state machine leaves ``.starting`` even if the Handle
    /// hasn't been explicitly stopped yet.
    @discardableResult
    static func start(
        cacheDir: URL,
        progress: DownloadProgress,
        pollInterval: TimeInterval = defaultPollInterval,
        isCancelled: @Sendable @escaping () -> Bool = { false }
    ) -> Handle {
        let (firstPollStream, firstPollContinuation) = AsyncStream<Bool>.makeStream(
            bufferingPolicy: .bufferingNewest(1)
        )
        let firstPoll = Task {
            var iterator = firstPollStream.makeAsyncIterator()
            return await iterator.next() ?? false
        }
        let task = Task.detached(priority: .utility) {
            defer { firstPollContinuation.finish() }
            var hasSignalledFirstPoll = false
            // Count what's ALREADY on disk before the first poll and
            // seed it as the growth baseline. Pre-existing bytes are a
            // cache hit (full weights) or a resumable partial — either
            // way they are not evidence of a download, and without the
            // baseline a cached model's first observation read as
            // "Downloading 5.6 GB / 5.6 GB · 100%" for the whole
            // mmap/Metal-warm window (2026-07 dogfood). Zero for a
            // fresh alias — that IS the baseline. This runs before the
            // child can plausibly have written anything meaningful
            // (the count lands within the spawn round-trip; the 4 MiB
            // epsilon in DownloadProgress absorbs the race).
            let baseline = directoryByteCount(at: cacheDir)
            await MainActor.run { progress.seedDiskBaseline(bytes: baseline) }
            let intervalNanos = UInt64(max(pollInterval, 0.1) * 1_000_000_000)
            while !Task.isCancelled && !isCancelled() {
                let bytes = directoryByteCount(at: cacheDir)
                var didPublish = false
                if bytes > 0 {
                    didPublish = await MainActor.run {
                        // Re-check at the publish point, not just at the top
                        // of the loop. `directoryByteCount` is a filesystem
                        // walk and the MainActor hop is a suspension — a
                        // `stop()` landing in either window would otherwise
                        // publish an observation the caller already cancelled,
                        // which is exactly what "no further updates after
                        // cancel" promises will not happen. Surfaced on CI,
                        // where the wider window made the race reliable.
                        guard !Task.isCancelled, !isCancelled() else { return false }
                        progress.applyDiskObservation(bytes: bytes)
                        return true
                    }
                }
                if !hasSignalledFirstPoll {
                    hasSignalledFirstPoll = true
                    firstPollContinuation.yield(didPublish)
                    firstPollContinuation.finish()
                }
                try? await Task.sleep(nanoseconds: intervalNanos)
            }
        }
        return Handle(task: task, firstPoll: firstPoll)
    }
}
