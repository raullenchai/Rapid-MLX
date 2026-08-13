import Foundation
import Observation

/// Parses HuggingFace's tqdm progress lines out of ``rapid-mlx serve``'s
/// stderr stream while the server is starting and surfaces them as
/// structured ``@Observable`` state for SwiftUI.
///
/// We see two kinds of progress on first-time downloads:
///
///   Fetching 16 files:  23%|██▎       | 3/16 [00:42<03:08, 0.07it/s]
///   model-00001-of-00006.safetensors:  41%|████▏     | 2.10G/5.13G [01:28<02:09, 23.4MB/s]
///
/// The outer "Fetching N files" is the most reliable progress signal —
/// HuggingFace hub resolves the snapshot up front, so ``total`` is
/// trustworthy and ``done`` only grows monotonically. The per-file lines
/// add detail (current filename + bytes), but tqdm refresh-in-place
/// means we get many updates per second — we only care about the latest.
///
/// Why a dedicated object (instead of folding this into ``ServerManager``):
/// SwiftUI re-renders the entire view that observes a property whenever
/// the property changes; folding everything into ``ServerManager`` means
/// the log-tail view diffs on every tqdm tick. Keeping ``phase`` in its
/// own ``@Observable`` lets the progress overlay re-render alone.
@MainActor
@Observable
final class DownloadProgress {
    static let maxProgressLineBytes = 8 * 1024

    /// Coarse phase the UI binds to. v0.4.42 split the old monolithic
    /// ``.loading`` case in two so the user can tell what rapid-mlx
    /// is actually doing at every step. The bug we're closing:
    ///
    /// rapid-mlx emits ``Loading model with BatchedEngine: <name>`` at
    /// the very START of ``load_model()`` — BEFORE the HuggingFace
    /// fetch happens (load_model triggers the fetch internally). The
    /// old parser treated that line as the end of download, so a
    /// user staring at the central overlay during a 5-minute first-
    /// time download saw "Loading model into Metal..." for the
    /// entire fetch and rightly wondered if anything was happening
    /// at all.
    ///
    /// New phase layout:
    ///   * ``.idle`` — pre-anything
    ///   * ``.preparing`` — "Loading model with" seen; no tqdm yet
    ///   * ``.fetching`` / ``.downloading`` — tqdm in flight
    ///   * ``.warmingUp`` — "Warming up (compiling Metal shaders)" seen
    enum Phase: Equatable, Sendable {
        case idle
        /// rapid-mlx announced it's about to load. The fetch /
        /// Metal-load may or may not start in the next instant
        /// (depends on cache state). Shows a calm "Preparing
        /// model…" subtitle so the user knows the request landed.
        case preparing
        /// HuggingFace is resolving snapshot files. ``done / total``
        /// counts complete files. ``percent`` mirrors tqdm's % for the
        /// linear bar but isn't strictly necessary (you can derive it
        /// from done/total).
        case fetching(done: Int, total: Int, percent: Int)
        /// A single file is being pulled. ``file`` is the basename;
        /// ``done`` and ``total`` are tqdm-formatted byte counts like
        /// "1.18G" — we keep them as strings because that's what the
        /// user reads, not because we can't parse them. ``eta`` is
        /// tqdm's own time-remaining estimate (e.g. "02:09" or
        /// "1:23:45") — extracted from the `[elapsed<eta, speed]`
        /// bracket. The v1 P1 fix to address "Starting overlay shows
        /// elapsed time, no ETR" (audit gap, ContentView.swift:486–530)
        /// — without an ETA users have no idea whether a 5GB download
        /// will land in 30 seconds or 30 minutes.
        case downloading(file: String, done: String, total: String, percent: Int, speed: String?, eta: String?)
        /// Metal shader compilation + KV-cache warmup. rapid-mlx's
        /// distinct marker for this stage is "Warming up (compiling
        /// Metal shaders)" — emitted from the FastAPI lifespan hook
        /// AFTER the model weights are loaded.
        case warmingUp
    }

    private(set) var phase: Phase = .idle

    /// Wall-clock of the most recent recognised progress tick (any of
    /// `.fetching`, `.downloading`, `.preparing`). The overlay uses
    /// this to surface a "X.X s since last update" caption when the
    /// phase is stuck — first-time users who only see the outer file
    /// counter (e.g. "Downloading 1/12 files") otherwise have no
    /// signal whether the download is still alive during the long
    /// multi-minute window between file flips.
    private(set) var lastTickAt: Date = .distantPast

    /// Bytes currently observed in the HuggingFace cache directory for
    /// this alias. ``nil`` when the cache-directory monitor isn't
    /// running (e.g. before a download has been started, or after the
    /// resolver couldn't find the HF cache root). When populated this
    /// is updated every ~3 s by ``HFCacheByteMonitor`` and gives the UI
    /// a TRUE bytes-on-disk signal that doesn't depend on tqdm output —
    /// HF's "Fetching N files" tqdm bar counts FILES, so a 12-shard
    /// download stays pinned at "0/12 files (0%)" for many minutes
    /// while the first 600 MB shard streams silently.
    ///
    /// Why this isn't folded into ``Phase``: phase tracks what
    /// rapid-mlx's stderr is telling us; bytes are an independent
    /// disk-side observation. Keeping them separate lets the UI fall
    /// through to phase-based copy whenever the cache observation is
    /// missing or stale, AND lets a future caller (e.g. progressive
    /// download UX during ``.preparing``) consult bytes without
    /// re-interpreting the phase enum.
    private(set) var bytesDownloaded: Int64?

    /// Best-known total bytes for the alias's weight files. Set once
    /// at job start by ``DownloadManager`` / ``ServerManager`` from
    /// the model catalog (``ModelSizing.estimate(alias:).weightsGB``
    /// converted to bytes) and never mutated thereafter. ``nil`` when
    /// the alias is unfamiliar and we can't infer params.
    private(set) var totalBytes: Int64?

    /// True once a real MEASURED total (the R2 puller's ``[bytes] D/T``
    /// heartbeat) has superseded the a-priori ``ModelSizing`` estimate
    /// that ``setTotalBytes`` seeds at job start. The estimate rounds
    /// every sub-4-bit quant (2-bit / ternary) up to 4-bit and so
    /// over-states low-bit models ~2x (``bonsai-1.7b-2bit``: 957 MB
    /// estimated vs 495 MB real), which the old monotonic guard then
    /// pinned for the whole download. The heartbeat is ground truth, so
    /// the FIRST one wins even when it is SMALLER than the estimate;
    /// later heartbeats still only grow (replay-shrink guard). #520.
    private(set) var hasMeasuredTotal: Bool = false

    /// True iff the most recent bytes update came from
    /// ``HFCacheByteMonitor.applyObservation`` — i.e. we observed the
    /// cache directory actually growing. Drives the UI's preference
    /// for bytes-based copy over file-count copy: only swap to
    /// "1.2 / 6.8 GB · 18%" once the disk really has bytes; until then
    /// fall through to "Downloading 0/9 files" so we don't show
    /// "0 B" when nothing has hit disk yet.
    private(set) var hasDiskObservation: Bool = false

    /// Bytes already sitting in the alias's cache directory when this
    /// start cycle began — seeded by ``HFCacheByteMonitor`` with its
    /// pre-spawn count (or lazily by the first observation when no
    /// explicit seed arrived). Pre-existing bytes are a cache hit or a
    /// resumable partial, NOT a download.
    private(set) var baselineDiskBytes: Int64?

    /// Latched true the first time bytes provably MOVE this cycle:
    /// the disk count rises above ``baselineDiskBytes`` by more than
    /// ``growthEpsilonBytes``, or a per-file tqdm ``.downloading``
    /// line is parsed (those only fire on real transfers). This — not
    /// the tqdm phase — is the only signal allowed to put the word
    /// "Downloading" on screen for the serve path. 2026-07 dogfood:
    /// the byte monitor runs on EVERY start, and on a cached model
    /// its first poll observes the full weight size already on disk,
    /// so phase-or-observation-driven copy claimed "Downloading
    /// 5.6 GB / 5.6 GB · 100%" for the whole mmap/Metal-warm window.
    /// A static directory never grows, so growth cannot lie.
    ///
    /// Deliberately a LATCH, not a computed property: HF's pull path
    /// alternates phases between files (.downloading → .fetching),
    /// and a non-latched signal would flip the label back to
    /// "Loading" in every inter-file gap.
    private(set) var hasObservedGrowth: Bool = false

    /// Slack for metadata/lockfile churn and heartbeat-vs-directory
    /// drift; far below any real weight shard.
    nonisolated static let growthEpsilonBytes: Int64 = 4 * 1024 * 1024

    /// Seed the pre-spawn baseline. First writer wins — later calls
    /// are no-ops so a slow first poll can't overwrite the true
    /// pre-spawn count with mid-download bytes.
    func seedDiskBaseline(bytes: Int64) {
        guard baselineDiskBytes == nil else { return }
        baselineDiskBytes = max(0, bytes)
    }

    /// What the serve-path start is actually DOING right now, as far
    /// as the observable evidence goes. The two progress surfaces
    /// (picker pill, chat startup banner) key their copy off this so
    /// they cannot disagree — and so neither can say "Downloading"
    /// while zero bytes move.
    enum StartupActivity: Equatable, Sendable {
        /// No signal yet (child just spawned).
        case starting
        /// Bytes are provably moving.
        case downloading
        /// Weights on disk, nothing growing — mmap / load window.
        case loading
        /// Server reported the warm-up phase.
        case warmingUp
    }

    var startupActivity: StartupActivity {
        if case .warmingUp = phase { return .warmingUp }
        if hasObservedGrowth { return .downloading }
        // Per-file tqdm parse implies transfer even before the disk
        // poll catches up.
        if case .downloading = phase { return .downloading }
        if hasDiskObservation { return .loading }
        switch phase {
        case .preparing, .fetching: return .loading
        default: return .starting
        }
    }

    /// Compact "N min left" for the pill's summary line; nil until
    /// total, bytes and rate are all known.
    var etaText: String? {
        guard let total = totalBytes, let bytes = bytesDownloaded,
              let speed = bytesPerSecond, speed > 0, total > bytes else { return nil }
        return Self.formatETA(bytesRemaining: total - bytes, bytesPerSecond: speed)
    }

    /// Recent ``(timestamp, bytes)`` samples accumulated from
    /// ``applyDiskObservation``. Used to estimate a rolling download
    /// rate. Capped at ``maxRateSamples`` so the buffer can't grow
    /// unbounded on long downloads. We keep the raw samples (not a
    /// pre-smoothed value) so the rate calc can re-derive speed
    /// against the OLDEST in-window sample on each call — simpler to
    /// reason about than recursive EMA and easier to test.
    private var rateSamples: [(at: Date, bytes: Int64)] = []

    /// Most recent rate estimate in bytes/second, or ``nil`` if we
    /// don't have a fresh enough window to derive one. Recomputed on
    /// every ``applyDiskObservation`` tick. Exposed read-only so the
    /// subtitle and the tests can both read the same number.
    ///
    /// Suppression rules (all must hold to publish a value):
    ///   * At least 2 samples in the window
    ///   * Oldest sample ≤ ``rateWindowSeconds`` old vs the newest
    ///   * Newest sample ≤ ``rateStaleSeconds`` old vs ``rateNow``
    ///   * Delta bytes > 0 (a shrinking cache means cleanup, not download)
    ///
    /// Why a window rather than an EMA: the heartbeat fires every
    /// 500 ms, but slow network paths can deliver a single chunk
    /// every few seconds. EMA with a fixed alpha overshoots when
    /// chunks arrive in bursts. A "newest − oldest over wall time"
    /// in a 4-second window mirrors what Chrome and HF's own tqdm
    /// do — and matches the user's expectation of "current speed."
    private(set) var bytesPerSecond: Double?

    /// Rolling window length for the rate calc. 4 seconds: long enough
    /// to absorb the 500 ms heartbeat cadence + bursty chunks without
    /// jumping; short enough that the displayed speed catches a real
    /// stall within a few ticks. Chosen empirically; bumping by ±2 s
    /// doesn't visibly change the UX.
    private nonisolated static let rateWindowSeconds: TimeInterval = 4.0

    /// Cap on retained samples. With a 500 ms heartbeat that's 8 in
    /// the window — keep 16 so a momentary slow-down doesn't push
    /// the oldest sample out before we've drawn enough of them.
    private nonisolated static let maxRateSamples = 16

    /// Window of "freshness": if the newest sample is older than this
    /// we publish ``nil``, so the UI doesn't show a stale "5.2 MB/s"
    /// readout after the download actually stalls. 5 s is generous
    /// vs the 500 ms heartbeat but tight enough to feel responsive.
    private nonisolated static let rateStaleSeconds: TimeInterval = 5.0

    /// Reset to ``.idle``. Called by ``ServerManager`` whenever a new
    /// start cycle begins so the previous run's progress doesn't bleed
    /// into the next.
    func reset() {
        phase = .idle
        lastTickAt = .distantPast
        bytesDownloaded = nil
        totalBytes = nil
        hasMeasuredTotal = false
        hasDiskObservation = false
        baselineDiskBytes = nil
        hasObservedGrowth = false
        rateSamples.removeAll(keepingCapacity: true)
        bytesPerSecond = nil
    }

    /// Set the catalog-known total weight size. Called once when a job
    /// starts. Idempotent — passing ``nil`` clears, passing a positive
    /// value installs. Negative / zero is rejected so an unset
    /// ``ModelSizing.weightsGB`` (params unknown → 0) leaves the UI in
    /// "X MB downloaded" mode rather than "X / 0 B (∞%)".
    func setTotalBytes(_ bytes: Int64?) {
        guard let bytes else {
            totalBytes = nil
            return
        }
        totalBytes = bytes > 0 ? bytes : nil
    }

    /// Ingest a fresh disk-observation tick from ``HFCacheByteMonitor``.
    /// Only positive observations advance the byte counter — a transient
    /// read failure (returns 0) is ignored so the UI doesn't flap back
    /// to "0 B downloaded" if a sandbox / permission glitch hides the
    /// dir for one beat.
    ///
    /// Updates ``lastTickAt`` so the "X s since last update" caption
    /// counts disk observations the same way it counts tqdm ticks —
    /// the user gets a live "still downloading" signal even if HF's
    /// tqdm has gone silent between file flips.
    func applyDiskObservation(bytes: Int64) {
        applyDiskObservation(bytes: bytes, at: Date())
    }

    /// Test seam for ``applyDiskObservation``. Production calls go
    /// through the no-arg overload; tests pass a synthesised ``Date``
    /// so the rate window math is deterministic. We don't use a
    /// global clock injection because the rest of the type already
    /// reads ``Date()`` directly and a partial injection would just
    /// confuse the call sites.
    func applyDiskObservation(bytes: Int64, at now: Date) {
        guard bytes > 0 else { return }
        if let baseline = baselineDiskBytes {
            if bytes > baseline + Self.growthEpsilonBytes {
                hasObservedGrowth = true
            }
        } else {
            // No explicit seed (heartbeat-only path): the first
            // observation becomes the baseline, so growth can latch
            // from the second tick onward.
            baselineDiskBytes = bytes
        }
        bytesDownloaded = bytes
        // The catalog/alias total is only an estimate. Once the filesystem
        // proves that estimate was too small, keeping it would produce copy
        // such as "633 MB / 563 MB · 100%". Drop the disproven denominator
        // until a later measured heartbeat supplies a trustworthy total.
        if let total = totalBytes, bytes > total {
            totalBytes = nil
        }
        hasDiskObservation = true
        lastTickAt = now
        recordRateSample(at: now, bytes: bytes)
        bytesPerSecond = computeRate(now: now)
    }

    /// Append a sample, drop anything older than ``rateWindowSeconds``
    /// behind the newest one, and enforce ``maxRateSamples``. Called
    /// only from ``applyDiskObservation(bytes:at:)``.
    private func recordRateSample(at now: Date, bytes: Int64) {
        rateSamples.append((at: now, bytes: bytes))
        let cutoff = now.addingTimeInterval(-Self.rateWindowSeconds)
        while let first = rateSamples.first, first.at < cutoff {
            rateSamples.removeFirst()
        }
        if rateSamples.count > Self.maxRateSamples {
            rateSamples.removeFirst(rateSamples.count - Self.maxRateSamples)
        }
    }

    /// Derive the current download rate from the sample buffer.
    /// Returns ``nil`` if any suppression rule fires; see
    /// ``bytesPerSecond`` for the full list. Pure given the buffer
    /// + ``now`` so it composes with the test seam.
    private func computeRate(now: Date) -> Double? {
        guard let oldest = rateSamples.first, let newest = rateSamples.last else {
            return nil
        }
        guard rateSamples.count >= 2 else { return nil }
        let span = newest.at.timeIntervalSince(oldest.at)
        guard span > 0 else { return nil }
        let staleness = now.timeIntervalSince(newest.at)
        if staleness > Self.rateStaleSeconds { return nil }
        let delta = newest.bytes - oldest.bytes
        guard delta > 0 else { return nil }
        return Double(delta) / span
    }

    /// Ingest one stdout/stderr line from the child. Returns ``true`` if
    /// the line was recognised as a progress tick — the caller may
    /// choose to suppress those from the user-visible log tail to keep
    /// it readable (tqdm produces hundreds of redraws per file).
    @discardableResult
    func ingest(_ line: String) -> Bool {
        // Strip ANSI SGR escapes before any matching. The rapid-mlx R2
        // puller (``vllm_mlx/_mirror.py``) wraps its `[N/M] file R2 (X
        // MB)` completion tags in ``\x1b[2m…\x1b[0m`` DIM/RESET pairs
        // when stdout is a TTY. The desktop spawns the child with a
        // pipe (no TTY), so ``_print_dim`` sees ``is_tty == False`` and
        // emits plain ASCII — but a user running ``rapid-mlx serve``
        // through a wrapper that re-attaches a PTY (or a future
        // change to the puller) would leak the raw escapes into our
        // stdin. Strip them up front so the regex matchers don't have
        // to know about ANSI at all.
        let stripped = line.replacingOccurrences(
            of: "\u{1B}\\[[0-9;]*m",
            with: "",
            options: .regularExpression
        )
        let trimmed = stripped.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        guard trimmed.utf8.count <= Self.maxProgressLineBytes else { return false }

        // Forward-only invariant only kicks in once we've reached
        // ``.warmingUp`` — that's the genuine end of the download +
        // Metal-load pipeline, after which a stray tqdm line would
        // be a real bug (rapid-mlx doesn't re-fetch mid-warmup). We
        // deliberately do NOT apply this to ``.preparing``: the old
        // parser used to treat "Loading model with" as the terminator,
        // which clipped the entire HF fetch window because rapid-mlx
        // emits that log line BEFORE load_model() internally triggers
        // the fetch.
        if case .warmingUp = phase {
            return false
        }

        // 1) R2 puller "Pulling <repo> (R2 mirror, fallback: HF)" /
        // "(mirror direct-layout, fallback: HF)" — banner that marks
        // the start of the per-file R2 phase. Drives the UI out of
        // the "Spinning up rapid-mlx…" copy the moment the puller
        // starts, before any per-file completion lands. R2 + HF tqdm
        // are mutually exclusive per-pull (the puller either runs the
        // R2 phase OR falls through to snapshot_download which emits
        // tqdm), so it's safe to honor whichever marker we see first.
        if matchR2PullerHeader(trimmed) {
            if case .idle = phase {
                phase = .preparing
            }
            lastTickAt = Date()
            return true
        }

        // 2) R2 puller "Found N files (~X.X GB total)" — primes the
        // file counter so the ``.fetching`` bar can render before the
        // first per-file completion line lands. Without this, the UI
        // would still sit at "Spinning up rapid-mlx…" for the entire
        // first-shard download (the R2 puller does not stream
        // mid-shard progress — completion lines fire once per file).
        if let m = matchR2FoundFiles(trimmed) {
            phase = .fetching(done: 0, total: m.totalFiles, percent: 0)
            lastTickAt = Date()
            return true
        }

        // 3a) R2 puller aggregate "[bytes] D/T" heartbeat — emitted at
        // most every 500 ms while ANY worker is streaming a file. Feeds
        // the bytes-on-disk channel directly so the linear progress bar
        // advances smoothly inside a multi-GB single shard download.
        // Without this, the user sits at "5/6 files (83%)" for 60-120 s
        // while one big shard streams silently in the background. Lives
        // here (before matchR2PerFile) so the bytes signal lands even
        // when a per-file completion line is interleaved on the same
        // poll tick — the heartbeat is the more granular signal.
        if let m = matchR2BytesHeartbeat(trimmed) {
            // Codex-style raised in review #2: ``setTotalBytes`` writes through
            // an ``@Observable`` setter, which fans out a property-change
            // notification SwiftUI uses to re-render any view that read
            // ``totalBytes``. A 120 s pull fires ~240 heartbeats with the
            // same ``total`` — gate on inequality so we don't pay the
            // re-render storm for an unchanged value. NIT #5: refuse a
            // shrinking ``total`` — a buggy mirror that replays an older
            // ``D/T`` shouldn't make the subtitle render "6.3 GB / 1.0 GB
            // · 100%" mid-pull. The byte channel itself stays monotonic
            // via ``applyDiskObservation``'s own ``bytes > 0`` guard +
            // ``progressFraction``'s ``[0, 1]`` clamp.
            let newTotal = Int64(m.total)
            if m.total > 0 {
                if !hasMeasuredTotal {
                    // First MEASURED total from the puller supersedes the
                    // a-priori ``ModelSizing`` estimate seeded at job
                    // start — even when it is SMALLER. The estimate rounds
                    // sub-4-bit quants (2-bit / ternary) up to 4-bit and
                    // over-states low-bit models ~2x (bonsai-1.7b-2bit:
                    // 957 MB estimated vs 495 MB real); the heartbeat is
                    // ground truth, so we trust it over the guess. #520.
                    setTotalBytes(newTotal)
                    hasMeasuredTotal = true
                } else if totalBytes ?? 0 < newTotal {
                    // Among measured heartbeats keep the max (NIT #5): a
                    // buggy mirror replaying a stale, smaller D/T must not
                    // shrink the bar mid-pull. The byte channel itself
                    // stays monotonic via ``applyDiskObservation``'s
                    // ``bytes > 0`` guard + ``progressFraction``'s clamp.
                    setTotalBytes(newTotal)
                }
            }
            applyDiskObservation(bytes: Int64(m.done))
            return true
        }

        // 3) R2 puller "[N/M] <filename> R2 (X MB)" / "HF (X MB,
        // fallback)" / "cached (X MB)" / "miss …" — per-file
        // completion line. Advances the file counter and drives the
        // bar from N-1/M to N/M. This is the puller's only per-file
        // signal — it doesn't refresh mid-shard like tqdm does, so
        // the bar will jump per-file rather than be smooth. The
        // mid-shard wait is smoothed by the aggregate ``[bytes] D/T``
        // heartbeat above (v0.7.11+) and ``HFCacheByteMonitor``
        // belt-and-braces.
        if let m = matchR2PerFile(trimmed) {
            let percent = m.total > 0 ? min(100, max(0, m.done * 100 / m.total)) : 0
            phase = .fetching(done: m.done, total: m.total, percent: percent)
            lastTickAt = Date()
            return true
        }

        // 4) Outer "Fetching N files: P%|bar| done/total [eta]"
        if let m = matchFetching(trimmed) {
            phase = .fetching(done: m.done, total: m.total, percent: m.percent)
            lastTickAt = Date()
            return true
        }

        // 5) Per-file "name: P%|bar| done/total [eta, speed]"
        if let m = matchPerFile(trimmed) {
            // Per-file tqdm lines only fire on real transfers — latch
            // growth even before the disk poll catches up.
            hasObservedGrowth = true
            phase = .downloading(
                file: m.file,
                done: m.done,
                total: m.total,
                percent: m.percent,
                speed: m.speed,
                eta: m.eta
            )
            lastTickAt = Date()
            return true
        }

        // 6) "Warming up (compiling Metal shaders)" is rapid-mlx's
        // distinct marker that the fetch + weight-load are done
        // and we're now in the GPU-shader-compile + KV-cache step.
        // This is the genuine endpoint — once we see it, suppress
        // further tqdm transitions.
        if trimmed.contains("compiling Metal shaders") || trimmed.contains("Warmup complete") {
            phase = .warmingUp
            return false
        }

        // 7) "Loading model with BatchedEngine" / "Loading model with
        // mlx-lm" announces the load is starting but DOES NOT mean
        // the fetch is over (it's emitted BEFORE the internal HF
        // fetch). Park us in ``.preparing`` so the spinner has
        // copy other than "Spinning up rapid-mlx…", but stay open
        // to a tqdm line interrupting in the next instant.
        if trimmed.contains("Loading model with") {
            // Only advance if we're idle; if we already saw tqdm
            // we'd rather keep the granular phase.
            if case .idle = phase {
                phase = .preparing
            }
            return false
        }

        return false
    }

    // MARK: - Parsers

    /// Whether ``line`` is the R2 puller's banner — emitted from
    /// ``vllm_mlx/_mirror.py`` as either
    /// ``  Pulling <repo> (R2 mirror, fallback: HF)`` or
    /// ``  Pulling <repo> (mirror direct-layout, fallback: HF)``.
    /// We don't care which variant — both signal that the puller has
    /// started its per-file R2 phase and the user should see "Pulling
    /// from rapid-mlx mirror…" copy instead of "Spinning up rapid-mlx…".
    private func matchR2PullerHeader(_ line: String) -> Bool {
        // The leading two-space indent + bold escape are stripped
        // upstream (ANSI removal in ``ingest``, leading-trim in
        // ``trimmingCharacters``). Check the load-bearing substrings.
        guard line.hasPrefix("Pulling ") else { return false }
        return line.contains("(R2 mirror, fallback: HF)")
            || line.contains("(mirror direct-layout, fallback: HF)")
    }

    private struct R2FoundFilesMatch {
        let totalFiles: Int
    }

    /// Matches the R2 puller's pre-flight file-plan line, e.g.
    /// ``  Found 12 files (~11.0 GB total)`` or ``  Found 12 files``
    /// (when no size is known up front). We only need ``totalFiles`` —
    /// the GB estimate flows through ``HFCacheByteMonitor`` for the
    /// byte-based copy, so we don't compete with it.
    private func matchR2FoundFiles(_ line: String) -> R2FoundFilesMatch? {
        guard line.hasPrefix("Found ") else { return nil }
        // Tokens: ["Found", "<N>", "files", ...]
        let tokens = line.split(whereSeparator: { $0.isWhitespace })
        guard tokens.count >= 3 else { return nil }
        guard tokens[0] == "Found" else { return nil }
        guard let total = Int(tokens[1]), total > 0 else { return nil }
        // "files" — accept singular too in case the puller ever pulls a
        // one-file repo (e.g. a tokenizer-only update).
        guard tokens[2] == "files" || tokens[2] == "file" else { return nil }
        return R2FoundFilesMatch(totalFiles: total)
    }

    /// Whether ``line`` is the R2 puller's aggregate ``[bytes] D/T``
    /// heartbeat. Exposed for ``ServerManager.appendLogLines`` so it
    /// can drop these from the user-visible log tail without re-
    /// invoking the full parser. ANSI-strip + trim mirrors ``ingest``
    /// so a TTY-wrapped emission is correctly recognised.
    ///
    /// Why a static classifier instead of routing the bool out of
    /// ``ingest``: every existing matcher's path already returns
    /// ``true`` for "recognised", and we DON'T want to suppress the
    /// per-file completion / fetching / per-shard tqdm lines — those
    /// fire at most a handful of times per pull and ARE useful in
    /// the log tail. Only the heartbeat (~2 Hz × the entire download)
    /// would evict legitimate startup output, so the suppression
    /// gates exclusively on this prefix.
    nonisolated static func isHeartbeatLogLine(_ line: String) -> Bool {
        let stripped = line.replacingOccurrences(
            of: "\u{1B}\\[[0-9;]*m",
            with: "",
            options: .regularExpression
        )
        return stripped
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .hasPrefix("[bytes]")
    }

    private struct R2BytesHeartbeatMatch {
        let done: Int64
        let total: Int64
    }

    /// Matches the R2 puller's aggregate byte heartbeat, e.g.
    /// ``[bytes] 2147483648/6800000000``. Emitted at most every 500 ms
    /// from whichever worker happens to write a chunk. ``done`` is
    /// cumulative bytes across ALL files (cached + R2 + HF-fallback);
    /// ``total`` is the planned snapshot size (sum of HF-advertised
    /// sizes). v0.7.11 fix for the "stuck at 83%" UX bug — see
    /// ``_ProgressTracker`` in ``vllm_mlx/_mirror.py``.
    private func matchR2BytesHeartbeat(_ line: String) -> R2BytesHeartbeatMatch? {
        guard line.hasPrefix("[bytes]") else { return nil }
        let after = line.dropFirst("[bytes]".count)
            .trimmingCharacters(in: .whitespaces)
        let pieces = after.split(separator: "/", omittingEmptySubsequences: false)
        guard pieces.count == 2,
              let done = Int64(pieces[0]),
              let total = Int64(pieces[1]),
              done >= 0,
              total >= 0 else { return nil }
        return R2BytesHeartbeatMatch(done: done, total: total)
    }

    private struct R2PerFileMatch {
        let done: Int
        let total: Int
    }

    /// Matches the R2 puller's per-file completion line, e.g.
    /// ``  [8/12] model-00001-of-00003.safetensors R2 (4523 MB)``
    /// ``  [3/12] config.json HF (0 MB, fallback)``
    /// ``  [5/12] tokenizer.json cached (1 MB)``
    /// ``  [6/12] foo.bin miss (will retry via HF snapshot_download)``
    ///
    /// We only need ``done`` and ``total`` — the per-file kind and byte
    /// count are surfaced in the underlying CLI log tail; the overlay
    /// just needs to advance the file counter so the bar moves.
    private func matchR2PerFile(_ line: String) -> R2PerFileMatch? {
        guard line.hasPrefix("[") else { return nil }
        guard let close = line.firstIndex(of: "]") else { return nil }
        // Strip the leading '['
        let inside = line[line.index(after: line.startIndex)..<close]
        let pieces = inside.split(separator: "/", omittingEmptySubsequences: false)
        guard pieces.count == 2,
              let done = Int(pieces[0]),
              let total = Int(pieces[1]),
              total > 0,
              done >= 0,
              done <= total else { return nil }
        // Require a non-empty tail after the bracket — protects against
        // matching a stray ``[3/12]`` in a different log line. The tail
        // contains the filename + kind tag (e.g. " config.json R2 (0
        // MB)"); we don't validate its contents because the R2 puller
        // can emit several kind shapes (R2 / HF / cached / miss /
        // sanitized failures) and we don't want this matcher to drift
        // every time _mirror.py grows a new tag.
        let tail = line[line.index(after: close)...].trimmingCharacters(in: .whitespaces)
        guard !tail.isEmpty else { return nil }
        return R2PerFileMatch(done: done, total: total)
    }

    private struct FetchingMatch {
        let done: Int
        let total: Int
        let percent: Int
    }

    /// Matches the outer file-count tqdm line, e.g.
    /// ``Fetching 16 files:  23%|██▎       | 3/16 [00:42<03:08, 0.07it/s]``
    private func matchFetching(_ line: String) -> FetchingMatch? {
        guard line.hasPrefix("Fetching ") else { return nil }
        guard let headerEnd = line.firstIndex(of: ":") else { return nil }
        let header = line[..<headerEnd].split(whereSeparator: { $0.isWhitespace })
        guard header.count == 3,
              header[0] == "Fetching",
              Int(header[1]) != nil,
              header[2] == "files" else { return nil }
        guard let percentMarker = line.range(of: "%|") else { return nil }
        guard percentMarker.lowerBound > headerEnd else { return nil }
        let percentToken = line[..<percentMarker.lowerBound]
            .split(whereSeparator: { $0.isWhitespace })
            .last
        guard let percentToken, let percent = Int(percentToken) else { return nil }

        let afterPercent = line[percentMarker.upperBound...]
        guard let barEnd = afterPercent.firstIndex(of: "|") else { return nil }
        let afterBar = afterPercent[afterPercent.index(after: barEnd)...]
        guard let fraction = afterBar.split(whereSeparator: { $0.isWhitespace }).first else {
            return nil
        }
        let pieces = fraction.split(separator: "/", omittingEmptySubsequences: false)
        guard pieces.count == 2,
              let done = Int(pieces[0]),
              let total = Int(pieces[1]) else { return nil }
        return FetchingMatch(done: done, total: total, percent: percent)
    }

    private struct PerFileMatch {
        let file: String
        let done: String
        let total: String
        let percent: Int
        let speed: String?
        let eta: String?
    }

    /// Matches per-file tqdm output, e.g.
    /// ``model-00001-of-00006.safetensors:  41%|████▏     | 2.10G/5.13G [01:28<02:09, 23.4MB/s]``
    ///
    /// We deliberately don't validate the filename character set — HF
    /// repos contain any UTF-8 — but we DO require the "<percent>%|"
    /// shape and the "[elapsed<remaining, speed]" tail so noise lines
    /// (e.g. INFO logs that happen to contain a colon) don't match.
    private func matchPerFile(_ line: String) -> PerFileMatch? {
        // Skip the "Fetching" prefix — already handled above.
        if line.hasPrefix("Fetching") { return nil }
        guard let nameEnd = line.firstIndex(of: ":") else { return nil }
        let file = String(line[..<nameEnd])
        guard !file.isEmpty else { return nil }

        let afterName = line[line.index(after: nameEnd)...].trimmingCharacters(in: .whitespaces)
        guard let percentMarker = afterName.range(of: "%|") else { return nil }
        let percentText = afterName[..<percentMarker.lowerBound].trimmingCharacters(in: .whitespaces)
        guard let percent = Int(percentText) else { return nil }

        let afterPercent = afterName[percentMarker.upperBound...]
        guard let barEnd = afterPercent.firstIndex(of: "|") else { return nil }
        let afterBar = afterPercent[afterPercent.index(after: barEnd)...].trimmingCharacters(in: .whitespaces)
        guard let fraction = afterBar.split(whereSeparator: { $0.isWhitespace }).first else { return nil }
        let pieces = fraction.split(separator: "/", omittingEmptySubsequences: false)
        guard pieces.count == 2 else { return nil }

        let done = String(pieces[0])
        let total = String(pieces[1])
        guard Self.isByteToken(done), Self.isByteToken(total) else { return nil }
        guard afterBar.contains("[") && afterBar.contains("]") else { return nil }

        let speed = Self.speedToken(in: afterBar)
        let eta = Self.etaToken(in: afterBar)
        // tqdm sometimes flushes a 0% redraw at total=0 before the actual
        // header lands. Suppress those — they make the bar flash empty.
        if total.hasPrefix("0") && percent == 0 { return nil }
        return PerFileMatch(file: file, done: done, total: total, percent: percent, speed: speed, eta: eta)
    }

    // MARK: - Byte-based progress derivation
    //
    // #?? — The HuggingFace tqdm bar that drives ``Phase`` counts FILES,
    // not BYTES. On a 6.8 GB / 11-shard model the outer bar sits at
    // "0/11 files (0%)" for many minutes while the first shard streams
    // silently — users assumed the download had hung. ``DownloadProgress``
    // now carries an independent disk-observation channel
    // (``bytesDownloaded`` + ``totalBytes``) populated by
    // ``HFCacheByteMonitor``. The helpers below let the UI prefer
    // byte-based copy whenever it's available and fall through to the
    // existing tqdm-derived copy otherwise.
    //
    // Both helpers are ``@MainActor``-implicitly via the enclosing
    // class but are pure (no side effects), so they're safe to call
    // from any ``View.body``.

    /// 0…1 fraction the UI should render in the linear progress bar,
    /// preferring bytes-on-disk over tqdm files when both are present.
    ///
    /// Priority order:
    ///   1. ``bytesDownloaded`` / ``totalBytes`` — truest signal; only
    ///      kicks in once we've actually observed positive bytes on
    ///      disk (``hasDiskObservation``). Clamped to [0, 1] so a
    ///      stale cache observation (e.g. user re-downloaded after a
    ///      `huggingface-cli scan-cache` cleanup) can't push the bar
    ///      past 100%.
    ///   2. Per-file tqdm ``.downloading`` percent — within a single
    ///      shard, tqdm's % is the most reliable signal.
    ///   3. Outer ``.fetching`` files-completed percent — the
    ///      "0/9 files" lie, but a better-than-nothing fallback when
    ///      bytes aren't observable (e.g. user redirected HF_HOME to a
    ///      directory we can't read).
    ///
    /// Returns ``nil`` for indeterminate phases (``.idle`` /
    /// ``.preparing`` / ``.warmingUp``) so callers can hide the bar
    /// without collapsing layout.
    var progressFraction: Double? {
        if hasDiskObservation, let bytes = bytesDownloaded, let total = totalBytes, total > 0 {
            let fraction = Double(bytes) / Double(total)
            return max(0.0, min(1.0, fraction))
        }
        switch phase {
        case .downloading(_, _, _, let percent, _, _):
            return Double(percent) / 100.0
        case .fetching(_, _, let percent):
            return Double(percent) / 100.0
        case .idle, .preparing, .warmingUp:
            return nil
        }
    }

    /// Compact subtitle the UI shows below "Downloading model files".
    /// Honours the same priority order as ``progressFraction``:
    ///
    ///   * Bytes + total → ``"1.2 / 6.8 GB · 18%"`` (the desired UX)
    ///   * Bytes only → ``"1.2 GB downloaded"`` (total unknown — still
    ///     trustworthy that bytes are flowing)
    ///   * tqdm-derived per-file → reused phase copy (file basename +
    ///     bytes + speed + ETA)
    ///   * Outer file counter → ``"0 of 9 files (0%)"``
    ///
    /// Returns ``nil`` for indeterminate phases so the caller can
    /// fall through to existing copy ("Spinning up rapid-mlx…").
    var progressSubtitle: String? {
        if hasDiskObservation, let bytes = bytesDownloaded {
            var parts: [String] = []
            if let total = totalBytes, total > 0 {
                let pct = Int((Double(bytes) / Double(total) * 100.0).rounded())
                let clamped = max(0, min(100, pct))
                parts.append("\(Self.formatBytes(bytes)) / \(Self.formatBytes(total)) · \(clamped)%")
                if let speed = bytesPerSecond {
                    parts.append(Self.formatSpeed(bytesPerSecond: speed))
                    if let eta = Self.formatETA(bytesRemaining: total - bytes, bytesPerSecond: speed) {
                        parts.append(eta)
                    }
                }
            } else {
                parts.append("\(Self.formatBytes(bytes)) downloaded")
                if let speed = bytesPerSecond {
                    parts.append(Self.formatSpeed(bytesPerSecond: speed))
                }
            }
            return parts.joined(separator: " · ")
        }
        switch phase {
        case .idle, .preparing, .warmingUp:
            return nil
        case .fetching(let done, let total, let percent):
            return "\(done) of \(total) file\(total == 1 ? "" : "s") (\(percent)%)"
        case .downloading(let file, let done, let total, let percent, let speed, let eta):
            var parts: [String] = ["\(percent)%", "\(file) (\(done)/\(total))"]
            if let speed { parts.append(speed) }
            if let eta { parts.append("ETA \(eta)") }
            return parts.joined(separator: " · ")
        }
    }

    /// Human-friendly byte formatter used by ``progressSubtitle``.
    /// Uses 1024-base ("GB" label but really GiB) to match HF's tqdm
    /// rendering so the same model reads as the same number in both
    /// places. Rounding: 1 decimal place below 100 of any unit (so
    /// 1.2 GB / 6.8 GB), no decimal at ≥ 100 (so 250 MB not 250.4 MB).
    ///
    /// Exposed ``nonisolated static`` for tests + so the UI can format
    /// auxiliary numbers (e.g. "Freed 1.2 GB") consistently.
    nonisolated static func formatBytes(_ bytes: Int64) -> String {
        let absBytes = bytes < 0 ? 0 : Double(bytes)
        let units: [(threshold: Double, suffix: String)] = [
            (1024.0 * 1024.0 * 1024.0 * 1024.0, "TB"),
            (1024.0 * 1024.0 * 1024.0, "GB"),
            (1024.0 * 1024.0, "MB"),
            (1024.0, "KB"),
        ]
        for (threshold, suffix) in units where absBytes >= threshold {
            let value = absBytes / threshold
            if value >= 100 {
                return "\(Int(value.rounded())) \(suffix)"
            }
            return String(format: "%.1f %@", value, suffix)
        }
        return "\(Int(absBytes.rounded())) B"
    }

    /// Format a byte-rate the way Chrome / Finder do: "683 KB/s",
    /// "5.2 MB/s". Uses 1024-base so a number quoted next to the
    /// `formatBytes`-formatted totals reads consistently.
    ///
    /// Below 1 KB/s the readout collapses to "< 1 KB/s" — at that
    /// level the next significant change won't arrive for many
    /// seconds anyway, and "0 B/s" reads like a stall.
    nonisolated static func formatSpeed(bytesPerSecond rate: Double) -> String {
        let value = max(0, rate)
        let units: [(threshold: Double, suffix: String)] = [
            (1024.0 * 1024.0 * 1024.0, "GB/s"),
            (1024.0 * 1024.0, "MB/s"),
            (1024.0, "KB/s"),
        ]
        for (threshold, suffix) in units where value >= threshold {
            let scaled = value / threshold
            if scaled >= 100 {
                return "\(Int(scaled.rounded())) \(suffix)"
            }
            return String(format: "%.1f %@", scaled, suffix)
        }
        // Below 1 KB/s we collapse to a floor so the speed token can't
        // read a literal "0 B/s" or a tiny "317 B/s" next to a still-
        // moving GB counter — both undermine the "this is alive"
        // signal the readout exists to provide.
        return "< 1 KB/s"
    }

    /// Format the remaining time in the same idiom Chrome's download
    /// shelf uses: "< 1 min left", "5 min left", "1 h 23 min left",
    /// "> 24 h" past a day. Returns ``nil`` for inputs that can't
    /// give a sensible answer (zero / negative speed or remaining,
    /// non-finite math). The "≥ 24 h" cap stops absurd readouts
    /// from a 50 KB/s burst on a 100 GB download — users would
    /// stop trusting any ETA that quoted "47 hours."
    nonisolated static func formatETA(bytesRemaining: Int64,
                                      bytesPerSecond rate: Double) -> String? {
        guard rate > 0, bytesRemaining > 0 else { return nil }
        let seconds = Double(bytesRemaining) / rate
        guard seconds.isFinite else { return nil }
        if seconds < 60 { return "< 1 min left" }
        if seconds >= 24 * 3600 { return "> 24 h left" }
        // Compute minutes first, then carry into hours so values that
        // round up to 60 don't render as "60 min left" or "1 h 60 min
        // left". E.g. ``seconds = 7170`` (1h 59.5m) used to render as
        // "1 h 60 min left" — the rounded residue was 60 minutes.
        var hours = Int(seconds / 3600)
        var minutes = Int(((seconds - Double(hours * 3600)) / 60).rounded())
        if minutes == 60 {
            hours += 1
            minutes = 0
        }
        if hours == 0 { return "\(minutes) min left" }
        if minutes == 0 { return "\(hours) h left" }
        return "\(hours) h \(minutes) min left"
    }

    private nonisolated static func speedToken(in tail: String) -> String? {
        guard let open = tail.firstIndex(of: "["),
              let close = tail[open...].firstIndex(of: "]") else {
            return nil
        }
        let bracket = tail[tail.index(after: open)..<close]
        guard let candidate = bracket.split(separator: ",").last?.trimmingCharacters(in: .whitespaces),
              candidate.hasSuffix("B/s") else {
            return nil
        }
        let value = String(candidate.dropLast(3))
        return isByteToken(value) ? candidate : nil
    }

    /// Extracts tqdm's time-remaining estimate from
    /// ``[elapsed<eta, speed]``. Returns nil when:
    ///   * the bracket is malformed,
    ///   * the elapsed-eta separator `<` is missing (e.g. tqdm at 0%
    ///     just shows `[?<?, ?it/s]`),
    ///   * the candidate isn't an `H?H:MM:SS` / `MM:SS` shape.
    /// Accepting only the canonical time-shape protects against tqdm
    /// emitting "?" placeholders before it has a stable estimate —
    /// those should leave the UI showing the elapsed clock alone.
    ///
    /// **Caller contract** (codex r1 NIT): ``tail`` must be the
    /// substring AFTER the tqdm `|` bar terminator — i.e. the
    /// fragment that begins with the byte fraction and contains
    /// only the trailing `[...]`. Passing a full unparsed log line
    /// would let `firstIndex(of: "[")` lock onto an earlier `[` in
    /// the message and silently return garbage. The production
    /// caller (``matchPerFile``) already pre-slices to ``afterBar``;
    /// the symmetric ``speedToken`` carries the same contract.
    nonisolated static func etaToken(in tail: String) -> String? {
        guard let open = tail.firstIndex(of: "["),
              let close = tail[open...].firstIndex(of: "]") else {
            return nil
        }
        let bracket = tail[tail.index(after: open)..<close]
        // The first comma separates elapsed/eta from speed; we want
        // the chunk BEFORE the comma. Inside that chunk, `<` splits
        // elapsed from eta.
        guard let timePart = bracket.split(separator: ",").first else {
            return nil
        }
        let pieces = timePart.split(separator: "<", omittingEmptySubsequences: false)
        guard pieces.count == 2 else { return nil }
        let candidate = pieces[1].trimmingCharacters(in: .whitespaces)
        return isTimeToken(candidate) ? candidate : nil
    }

    /// Whether ``token`` matches `M:SS` / `MM:SS` / `H:MM:SS` /
    /// `HH:MM:SS`. Anything else (notably tqdm's "?" placeholder) is
    /// rejected so we don't pipe garbage to the UI.
    private nonisolated static func isTimeToken(_ token: String) -> Bool {
        let parts = token.split(separator: ":", omittingEmptySubsequences: false)
        guard parts.count == 2 || parts.count == 3 else { return false }
        for part in parts {
            guard !part.isEmpty, part.allSatisfy({ $0.isASCII && $0.isNumber }) else {
                return false
            }
        }
        return true
    }

    /// Accepts every byte/speed token shape ``huggingface_hub`` tqdm
    /// can emit:
    /// * bare numbers (``23.4`` — used by speed scrubber after the
    ///   trailing ``B/s`` is dropped);
    /// * SI-prefixed (``2.10G``, ``2.10GB``);
    /// * IEC-prefixed (``2.10Gi``, ``2.10GiB``) which is what
    ///   ``huggingface_hub`` ≥0.20 produces by default thanks to
    ///   `unit_divisor=1024`.
    ///
    /// #150: prior to this fix the function only accepted the SI
    /// shapes WITHOUT trailing ``B``, so a per-file tqdm line like
    /// ``2.10GiB/5.13GiB [01:28<02:09, 23.4MiB/s]`` silently failed
    /// ``matchPerFile`` and the download overlay stayed pinned to
    /// the outer ``Fetching N files`` counter for the entire
    /// multi-minute download.
    private nonisolated static func isByteToken(_ token: String) -> Bool {
        guard !token.isEmpty else { return false }
        var sawDigit = false
        var sawDot = false
        // State machine:
        //   0 — mantissa (digits + optional '.')
        //   1 — saw SI prefix (K/M/G/T)
        //   2 — saw IEC marker 'i' after the SI prefix
        //   3 — saw trailing 'B' (terminal; nothing else allowed)
        var phase = 0
        for byte in token.utf8 {
            switch phase {
            case 0:
                if byte >= 48 && byte <= 57 {
                    sawDigit = true
                } else if byte == 46, !sawDot {
                    sawDot = true
                } else if byte == 75 || byte == 77 || byte == 71 || byte == 84 {
                    phase = 1
                } else {
                    return false
                }
            case 1:
                if byte == 105 {
                    phase = 2
                } else if byte == 66 {
                    phase = 3
                } else {
                    return false
                }
            case 2:
                if byte == 66 {
                    phase = 3
                } else {
                    return false
                }
            default:
                return false
            }
        }
        return sawDigit
    }
}
