import SwiftUI

/// One-line status strip that lives at the top of the chat surface and
/// shows any in-flight background downloads. Hidden
/// when ``DownloadManager.jobs`` has no running entries.
///
/// v0.5.7 ships this so users can pull a second model in parallel
/// while ``ServerManager`` keeps serving the model they're chatting
/// with. The picker bar itself stays focused on selection; the strip
/// owns the "what's downloading right now" surface — mirrors LM
/// Studio's bottom "Downloads" panel except inline (we don't have
/// the screen budget for a dedicated tab on a 13-inch MacBook).
///
/// Per-job row layout:
///
///   ⏳ qwen3-32b           62% · 5.1G/8.2G · 23.4MB/s    [×]
///   ✅ gemma-4-12b         Downloaded                     [Dismiss]
///   ⚠️  hermes3-8b         pull failed — connection reset [Dismiss]
///
/// The cancel "[×]" sends SIGTERM via ``DownloadManager``;
/// "[Dismiss]" clears a finished job from the row.
struct DownloadStrip: View {
    @Bindable var downloads: DownloadManager

    /// Deep-link channel into Settings, resolved optionally so the strip still
    /// renders in a host that never injected one (previews, the snapshot
    /// harness) — the non-optional form traps at lookup time.
    @Environment(SettingsRouter.self) private var settingsRouter: SettingsRouter?
    /// ``openWindow(id: "settings")``, NOT ``@Environment(\.openSettings)``:
    /// this app declares no SwiftUI ``Settings`` scene, so `OpenSettingsAction`
    /// is a silent no-op here. See ``SettingsRouter``.
    @Environment(\.openWindow) private var openWindow

    /// All known jobs, ordered: running first (by alias), then
    /// terminal states (alpha) so finished rows fall to the bottom
    /// and don't bury the live progress. Computed each render — the
    /// dictionary is small (typically 0-2 entries; 5+ would be
    /// pathological on a single Mac).
    private var orderedJobs: [DownloadManager.Job] {
        let pairs = downloads.jobs.values.map { job -> (DownloadManager.Job, Int) in
            // Sort key: running before terminal. Within each bucket,
            // alphabetical alias gives a stable order so newly added
            // jobs don't jump around mid-scroll.
            switch job.status {
            case .running:                          return (job, 0)
            case .completed:                        return (job, 1)
            case .cancelled, .failed:               return (job, 2)
            }
        }
        return pairs
            .sorted { lhs, rhs in
                if lhs.1 != rhs.1 { return lhs.1 < rhs.1 }
                return lhs.0.alias < rhs.0.alias
            }
            .map { $0.0 }
    }

    var body: some View {
        if !orderedJobs.isEmpty {
            VStack(alignment: .leading, spacing: 4) {
                ForEach(orderedJobs) { job in
                    jobRow(job)
                }
            }
            .padding(.horizontal, 18)
            .padding(.vertical, 6)
            .background(.bar)
            .overlay(Divider(), alignment: .bottom)
        }
    }

    @ViewBuilder
    private func jobRow(_ job: DownloadManager.Job) -> some View {
        if case .failed = job.status {
            HStack(spacing: 8) {
                Text(job.alias)
                    .scaledSystemFont(12, relativeTo: .caption, weight: .medium)
                    .lineLimit(1)
                    .truncationMode(.middle)
                FailureDiagnosisView(
                    diagnosis: FailureDiagnoser.diagnosis(
                        for: job.failureKind ?? .downloadFailed
                    ),
                    onAction: { action in handleFailureAction(action, for: job) }
                )
                trailingAffordance(for: job)
            }
        } else {
            HStack(spacing: 8) {
                statusGlyph(for: job)
                    .frame(width: 16)
                Text(job.alias)
                    .scaledSystemFont(12, relativeTo: .caption, weight: .medium)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(progressDetail(for: job))
                    .scaledSystemFont(11, relativeTo: .caption, design: .monospaced)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer()
                trailingAffordance(for: job)
            }
        }
    }

    private func handleFailureAction(
        _ action: FailureDiagnosis.Action,
        for job: DownloadManager.Job
    ) {
        switch action {
        case .retry:
            downloads.retryDownload(alias: job.alias)
        case .switchDownloadSource:
            downloads.retryDownload(alias: job.alias, source: .huggingFace)
        case .openModelManagement, .openWebSearchSettings:
            // A download job only ever produces ``.retry`` /
            // ``.switchDownloadSource`` today, so these are latent. Wiring
            // them anyway costs two lines and keeps the strip from growing the
            // dead button the Quickstart card just lost, the moment a new
            // failure kind starts routing here.
            settingsRouter?.route(action) { openWindow(id: "settings") }
        case .restart:
            break
        }
    }

    @ViewBuilder
    private func statusGlyph(for job: DownloadManager.Job) -> some View {
        switch job.status {
        case .running:
            ProgressView()
                .controlSize(.small)
                .scaleEffect(0.65)
        case .completed:
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
        case .cancelled:
            Image(systemName: "xmark.circle.fill")
                .foregroundStyle(.secondary)
        case .failed:
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
        }
    }

    /// Builds the human-readable progress line — same parsed phase
    /// the central download overlay uses for the in-band serve
    /// path, so the strip and the overlay can never drift apart.
    private func progressDetail(for job: DownloadManager.Job) -> String {
        switch job.status {
        case .running:
            return DownloadStrip.detail(
                phase: job.progress.phase,
                bytesSubtitle: job.progress.hasDiskObservation
                    ? job.progress.progressSubtitle
                    : nil
            )
        case .completed:
            return "Downloaded — ready to load"
        case .cancelled:
            return "Cancelled"
        case .failed(let message):
            return message
        }
    }

    /// Lifted to a static helper so ``DownloadStripTests`` can pin
    /// the phase → caption truth table without standing up a View
    /// (mirrors ``ModelPickerBar.fitReasonLabel``).
    ///
    /// ``bytesSubtitle`` carries the byte-based progress string
    /// (``"1.2 / 6.8 GB · 18%"``) when the HF cache-dir monitor has
    /// observed real bytes on disk. When present we preface the
    /// existing tqdm-derived copy with it so the user sees real bytes
    /// even when HF's outer "Fetching N files" tqdm is stuck at 0/N.
    static func detail(
        phase: DownloadProgress.Phase,
        bytesSubtitle: String? = nil
    ) -> String {
        let base = phaseDetail(phase)
        guard let bytesSubtitle, !bytesSubtitle.isEmpty else {
            return base
        }
        // Bytes-first; appendix from the tqdm phase only when it adds
        // information beyond the disk observation (e.g. the file name
        // during ``.downloading``). For ``.idle`` / ``.preparing`` /
        // ``.fetching`` the tqdm copy isn't useful on top of the byte
        // string, so we just show bytes.
        switch phase {
        case .idle, .preparing, .fetching:
            return bytesSubtitle
        case .downloading, .warmingUp:
            return "\(bytesSubtitle) · \(base)"
        }
    }

    private static func phaseDetail(_ phase: DownloadProgress.Phase) -> String {
        switch phase {
        case .idle:
            return "Starting…"
        case .preparing:
            return "Preparing…"
        case .fetching(let done, let total, let percent):
            return "\(percent)% · \(done)/\(total) files"
        case .downloading(let file, let done, let total, let percent, let speed, let eta):
            let head = "\(percent)% · \(done)/\(total)"
            let speedTail = speed.map { " · \($0)" } ?? ""
            let etaTail = eta.map { " · ETA \($0)" } ?? ""
            // File basename is the most useful disambiguator when
            // multiple files in one snapshot are mid-transfer; truncate
            // in the View, not here.
            return "\(file) · \(head)\(speedTail)\(etaTail)"
        case .warmingUp:
            // pull never hits warmingUp (it doesn't load the model),
            // but the parser shares the enum with serve. Treat as a
            // success-ish state.
            return "Finalising…"
        }
    }

    @ViewBuilder
    private func trailingAffordance(for job: DownloadManager.Job) -> some View {
        switch job.status {
        case .running:
            Button {
                downloads.cancelDownload(alias: job.alias)
            } label: {
                Image(systemName: "xmark.circle")
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .help("Cancel download (partial files stay in the HuggingFace cache and can resume on retry).")
            .accessibilityLabel("Cancel download of \(job.alias)")
            .accessibilityIdentifier("DownloadStrip.Cancel.\(job.alias)")
        case .completed, .cancelled, .failed:
            Button {
                downloads.dismissJob(alias: job.alias)
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 10))
                    .foregroundStyle(.tertiary)
            }
            .buttonStyle(.plain)
            .help("Dismiss")
            .accessibilityLabel("Dismiss \(job.alias) status")
            .accessibilityIdentifier("DownloadStrip.Dismiss.\(job.alias)")
        }
    }
}
