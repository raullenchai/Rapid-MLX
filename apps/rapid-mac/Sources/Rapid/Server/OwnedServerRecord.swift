import Foundation

/// Persisted record of the ``rapid-mlx`` child process this Rapid.app
/// session spawned. Lets ``PortSweep`` on the NEXT launch tell apart
/// "an orphan from our own previous crash" (kill it) from "a different
/// rapid-mlx serve a user is running in their terminal" (leave it
/// alone).
///
/// Before this file existed (issue #20), ``PortSweep`` relied purely on a
/// basename match (``rapid-mlx`` / ``python -m rapid_mlx serve``). That
/// heuristic had two failure modes:
///
///   1. **False positive**: a developer running their own
///      ``rapid-mlx serve`` got SIGTERM'd on the next Rapid.app launch.
///   2. **False negative load-bearing on coincidence**: the basename
///      match could not distinguish "process WE spawned and lost track
///      of after a crash" from "process a different Rapid session
///      spawned" — today both shapes are basename-equivalent so we
///      happened to do the right thing, but that's not guaranteed.
///
/// The record is written atomically on every successful spawn and
/// cleared on clean child exit. ``PortSweep`` reads it (when present)
/// and uses the recorded PGID for a precise group-kill instead of the
/// basename heuristic. A stale record (PID no longer on port, port
/// mismatch) is discarded — the sweep falls through to the existing
/// heuristic path, so behavior is strictly safer than before.
struct OwnedServerRecord: Codable, Equatable {
    /// PID of the spawned child as known to the parent.
    let pid: Int32
    /// Process group ID. ``kill(-pgid, SIGTERM)`` reaches uvicorn workers
    /// the child forked, not just the leader.
    let pgid: Int32
    /// Port the child was bound to. ``PortSweep`` cross-checks this
    /// before trusting the record — if the persisted port doesn't match
    /// the port being swept, the record is considered stale.
    let port: Int
    /// Alias the child was serving. Diagnostic only (surfaced in the
    /// sweep log line); not load-bearing for any kill decision.
    let alias: String
    /// Wall-clock when the record was written. Used purely for the
    /// sweep log line — record age is not the staleness signal
    /// (kernel PID re-use, not elapsed time, is what makes a record
    /// dangerous, and we test that via ``pidsOnPort`` intersection).
    let writtenAt: Date

    /// Absolute path to the persistence file under
    /// ``~/Library/Application Support/Rapid/owned-server.json``. The
    /// JSON envelope is forward-compatible — future fields can be
    /// added with a fresh ``Codable`` decode, and an old version
    /// reading a newer file simply gets the keys it understands.
    static func defaultURL() -> URL {
        // #419/#420 consolidation completion: delegate to
        // ApplicationSupportLocator. The pre-PR-#422 path used
        // FileManager.urls (ignored HOME); the NSHomeDirectory
        // fallback DID honour HOME but only kicked in if the
        // FileManager API returned nil — which never happens on
        // production macOS, so the leak was always-on. The locator
        // unifies the rules.
        ApplicationSupportLocator.applicationSupportRoot()
            .appendingPathComponent("owned-server.json")
    }

    /// Persist via atomic rename. Persistence failure is best-effort:
    /// the next-launch sweep simply falls back to the basename
    /// heuristic, so a partial fs write never escalates into a wrong
    /// kill decision.
    func persist(to url: URL = OwnedServerRecord.defaultURL()) {
        do {
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            let data = try JSONEncoder().encode(self)
            try data.write(to: url, options: [.atomic])
        } catch {
            FileHandle.standardError.write(Data(
                "[server] owned-server.json persist failed: \(error.localizedDescription)\n".utf8
            ))
        }
    }

    /// Remove the persisted record. Called on every clean exit path so
    /// the next launch starts from a known-empty baseline.
    static func clear(at url: URL = OwnedServerRecord.defaultURL()) {
        try? FileManager.default.removeItem(at: url)
    }

    /// Load the previous-session record if one exists and decodes.
    /// Returns nil on any I/O or decoding error — a corrupt record is
    /// treated the same as an absent one (fall through to heuristic).
    static func load(from url: URL = OwnedServerRecord.defaultURL()) -> OwnedServerRecord? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(OwnedServerRecord.self, from: data)
    }
}
