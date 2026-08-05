import Foundation

/// Process-wide cache in front of ``ModelCatalog/load(binary:hubCacheOverride:)``.
///
/// Every catalog read forks **at least two** `rapid-mlx` subprocesses — `models`
/// and `ls`, plus one `info` per cached sibling — and five separate surfaces
/// call it (both settings panels, the composer's model picker, launch
/// auto-start, and deletion). None of them shared anything, so simply moving
/// between Models and Model Management paid for a fresh round of process
/// spawns and showed a spinner each time.
///
/// ## Staleness is bounded, not merely time-limited
///
/// Entries are keyed on ``DownloadManager/cacheGeneration``, the counter the app
/// already bumps whenever the on-disk model set changes (download finished,
/// alias deleted, models folder repointed). A cached snapshot is therefore
/// invalidated by the *event* that makes it wrong rather than by a timer that
/// might fire too late — an expiring TTL would still hand out a stale
/// "downloaded" badge for however long it had left to run.
///
/// A short TTL is kept **on top of** that as a backstop for changes nothing
/// announces: a model deleted from `~/.cache/huggingface` by another tool, or
/// by `rapid-mlx` from a terminal.
///
/// ## Serving stale while revalidating
///
/// ``entries(binary:generation:)`` returns any usable snapshot immediately and
/// refreshes in the background. The first read of a generation has nothing to
/// serve and awaits the real load; every read after that is instant. Callers
/// distinguish the two via ``cached(binary:generation:)`` so they can show a
/// spinner only when there is genuinely nothing to show.
actor ModelCatalogCache {
    static let shared = ModelCatalogCache()

    /// Backstop for out-of-band changes (see above).
    ///
    /// Five minutes, not seconds: ``cacheGeneration`` already catches every
    /// mutation the app makes, so this only has to bound how long an edit made
    /// *outside* the app can go unnoticed. An earlier 30s value expired while
    /// the user was simply reading the Settings panel — measured MISSes at
    /// age=33s and age=40s during ordinary tab switching — which is exactly
    /// the needless refetch this cache exists to remove.
    private static let ttl: TimeInterval = 300

    private struct Snapshot {
        let entries: [ModelEntry]
        let generation: UInt
        let fetchedAt: Date
        /// The override in force when this was captured. Pointing the models
        /// folder somewhere else changes what "cached" means for every alias,
        /// so a snapshot taken under a different root cannot be reused.
        let overridePath: String?
    }

    private var snapshot: Snapshot?

    /// Mirror of ``snapshot`` readable without `await`.
    ///
    /// SwiftUI builds `@State` initial values synchronously, so a view cannot
    /// consult an actor before its first frame. Without this, a panel that
    /// re-appears starts at `catalog = []` + `loading = true`, renders the
    /// spinner, and only replaces it once the `.task` resumes — the cache
    /// removes the subprocess cost but the *flash* remains, which is the part
    /// the user actually sees.
    ///
    /// `nonisolated(unsafe)` is sound here for the same reason the actor is
    /// enough elsewhere: writes happen only from inside the actor (a single
    /// serialised context), and the value is a `let`-only struct of immutable
    /// members. A reader can observe a slightly older snapshot, never a torn
    /// one — and "slightly older" is exactly what a cache promises.
    nonisolated(unsafe) private static var lastSnapshot: Snapshot?

    /// Synchronous peek for `@State` seeding. Returns entries only when they
    /// would also satisfy ``cached(binary:generation:)``.
    nonisolated static func seed(generation: UInt) -> [ModelEntry]? {
        guard let snap = lastSnapshot,
              snap.generation == generation,
              snap.overridePath == ModelsFolderPreference.validatedOverrideURL()?.path,
              Date().timeIntervalSince(snap.fetchedAt) < ttl
        else { return nil }
        return snap.entries
    }

    /// In-flight load, so N simultaneous views trigger one set of subprocesses
    /// instead of N. Without this the first paint after launch — picker plus
    /// auto-start plus whichever panel is open — would fan out.
    private var inFlight: Task<[ModelEntry], Never>?

    /// A snapshot only if one is currently valid; never triggers a load.
    ///
    /// Views use this to decide whether to show a loading state: `nil` means
    /// "nothing to display yet", not "nothing exists".
    func cached(binary: URL, generation: UInt) -> [ModelEntry]? {
        guard let snapshot, isFresh(snapshot, generation: generation) else {
            return nil
        }
        return snapshot.entries
    }

    /// The catalog, served from cache when possible.
    ///
    /// Returns immediately with a valid snapshot; otherwise awaits a real load
    /// (joining one already running rather than starting a second).
    func entries(binary: URL, generation: UInt) async -> [ModelEntry] {
        if let snapshot, isFresh(snapshot, generation: generation) {
            return snapshot.entries
        }
        if let inFlight {
            return await inFlight.value
        }
        let override = ModelsFolderPreference.validatedOverrideURL()
        let task = Task<[ModelEntry], Never> {
            await ModelCatalog.load(binary: binary, hubCacheOverride: override)
        }
        inFlight = task
        let loaded = await task.value
        // Stamp with the generation captured at call time. If a mutation
        // landed mid-load the counter has already moved on, so this snapshot
        // is born stale and the next read refetches — which is the correct
        // outcome: the load observed the world before the change.
        snapshot = Snapshot(
            entries: loaded,
            generation: generation,
            fetchedAt: Date(),
            overridePath: override?.path
        )
        Self.lastSnapshot = snapshot
        inFlight = nil
        return loaded
    }

    /// Drop the snapshot. For callers that mutate the model set themselves and
    /// want the next read to be authoritative without waiting on the
    /// generation counter to propagate.
    func invalidate() {
        snapshot = nil
        Self.lastSnapshot = nil
    }

    private func isFresh(_ snapshot: Snapshot, generation: UInt) -> Bool {
        guard snapshot.generation == generation else { return false }
        guard snapshot.overridePath == ModelsFolderPreference.validatedOverrideURL()?.path else {
            return false
        }
        return Date().timeIntervalSince(snapshot.fetchedAt) < Self.ttl
    }
}
