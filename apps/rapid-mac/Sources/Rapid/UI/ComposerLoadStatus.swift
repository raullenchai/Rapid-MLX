import Foundation

/// What the composer's status slot says while the selected model is being
/// prepared, or has failed to start.
///
/// ## Why it lives in the composer
///
/// Loading used to open a graphite band across the top of the chat. Its
/// contrast against the light canvas made the window top-heavy, and it put
/// the one thing the user was waiting on as far as possible from the control
/// it was gating. The composer is where Send is disabled, so it is where the
/// reason belongs — first launch, relaunch, a model switch and a reload all
/// render here, in one layout.
///
/// ## What it is derived from
///
/// Nothing new. ``ModelReadiness`` already decides which model a live
/// download or start speaks for (``ModelReadiness/serveStateSpeaksForSelection``),
/// so an old model's progress cannot describe a new pick. This type only
/// reshapes that answer — plus the structured byte counts the same
/// ``ModelReadiness/ProgressSnapshot`` already carries — into the rows the
/// composer draws. It holds no state and cannot disagree with the
/// placeholder, the Send tooltip or the hero.
///
/// Decision states (no model, not downloaded, not running) are not here:
/// they need a choice, not a wait, and keep the readiness notice above the
/// field with its action button.
struct ComposerLoadStatus: Equatable {
    enum Phase: Equatable {
        /// Weights are moving onto disk.
        case downloading
        case stopped
        /// On disk; the engine is starting, loading or warming up.
        case loading
        /// The start failed and needs the user.
        case failed
    }

    enum Progress: Equatable {
        /// A real fraction from the download monitor, clamped to 0…1.
        case determinate(Double)
        /// Work is happening but nothing honest can say how far along.
        case indeterminate
        /// Nothing is in flight (a failure).
        case none
    }

    var phase: Phase
    /// Leading text on the first row.
    var title: String
    /// Trailing text on the first row — only with a determinate fraction.
    var percentText: String?
    /// Leading text on the second row: the model, then what has arrived,
    /// or the failure message.
    var detail: String?
    /// Trailing text on the second row — only once the ETA is stable.
    var etaText: String?
    var progress: Progress
    /// Recovery for ``Phase/failed``; nil while work is in flight.
    var action: ModelReadiness.Action?

    var isFailure: Bool { phase == .failed }

    /// The composed VoiceOver label for the whole slot.
    var accessibilityLabel: String {
        [title, detail, percentText, etaText]
            .compactMap { $0 }
            .joined(separator: ", ")
    }

    // MARK: - Resolution

    /// - Parameters:
    ///   - readiness: the window's single readiness value.
    ///   - progress: the snapshot ``ContentView`` resolved for the SAME
    ///     selected alias. Only consulted while ``readiness`` is
    ///     downloading or starting.
    static func resolve(
        readiness: ModelReadiness,
        progress: ModelReadiness.ProgressSnapshot?,
        downloadStatus: DownloadManager.Job.Status? = nil
    ) -> ComposerLoadStatus? {
        if downloadStatus == .cancelled {
            switch readiness {
            case .needsDownload, .downloading: return stoppedStatus(alias: readiness.alias)
            default: break
            }
        }
        switch readiness {
        case .downloading(let alias, _, let fraction):
            let clamped = fraction.flatMap { $0.isFinite ? min(max($0, 0), 1) : nil }
            return ComposerLoadStatus(
                phase: .downloading,
                title: String(localized: "Downloading the model…"),
                percentText: percentText(for: clamped),
                detail: joined(ModelDisplayName.name(for: alias), bytesText(progress)),
                etaText: progress?.etaText,
                progress: clamped.map(Progress.determinate) ?? .indeterminate,
                action: nil
            )
        case .starting(let alias, _):
            // The download percentage is deliberately NOT carried over. On a
            // cached model the byte monitor reads the full weight size on its
            // first poll, so a fraction here is ~1.0 and would claim "100%"
            // for the entire load.
            return ComposerLoadStatus(
                phase: .loading,
                title: loadingTitle(for: progress?.activity),
                percentText: nil,
                detail: ModelDisplayName.name(for: alias),
                etaText: nil,
                progress: .indeterminate,
                action: nil
            )
        case .failed(_, let message, let action):
            return ComposerLoadStatus(
                phase: .failed,
                title: readiness.headline,
                percentText: nil,
                detail: message,
                etaText: nil,
                progress: .none,
                action: action
            )
        case .engineMissing, .sharingCompute, .noModel, .needsDownload,
             .needsStart, .unknownModel, .ready:
            return nil
        }
    }

    private static func stoppedStatus(alias: String?) -> ComposerLoadStatus {
        ComposerLoadStatus(
            phase: .stopped, title: String(localized: "Download stopped"),
            percentText: nil, detail: alias.map { ModelDisplayName.name(for: $0) }, etaText: nil,
            progress: .none, action: nil
        )
    }

    /// Only suppress a global row when the visible composer can control it.
    static func ownsDownload(readiness: ModelReadiness, status: DownloadManager.Job.Status?) -> Bool {
        switch (readiness, status) {
        case (.downloading, .running), (.downloading, .cancelled), (.needsDownload, .cancelled): return true
        default: return false
        }
    }

    /// What VoiceOver should hear when readiness moves from `old` to `new`,
    /// or nil when the change is not worth interrupting for.
    ///
    /// Coarse on purpose: entering a download, entering a load, becoming
    /// ready after work, and failing. A percentage tick, a byte count, an
    /// ETA refresh and a loading → warming-up step are all the same moment
    /// to a listener and stay silent; the progress indicator's own value is
    /// there for anyone who asks.
    struct AnnouncementState: Equatable {
        var readiness: ModelReadiness
        var downloadStatus: DownloadManager.Job.Status?
    }

    static func announcement(from old: AnnouncementState, to new: AnnouncementState) -> String? {
        announcement(from: old.readiness, to: new.readiness,
                     oldDownloadStatus: old.downloadStatus, newDownloadStatus: new.downloadStatus)
    }

    static func announcement(
        from old: ModelReadiness, to new: ModelReadiness,
        oldDownloadStatus: DownloadManager.Job.Status? = nil,
        newDownloadStatus: DownloadManager.Job.Status? = nil
    ) -> String? {
        let before = Moment(old, downloadStatus: oldDownloadStatus)
        let after = Moment(new, downloadStatus: newDownloadStatus)
        guard before != after else { return nil }
        switch after {
        case .downloading, .loading:
            return new.headline
        case .ready:
            // Only after waiting. Selecting a model that is already serving
            // is the user's own action and needs no announcement.
            return before.isWorking ? new.headline : nil
        case .failed:
            return new.accessibilityLabel
        case .stopped:
            return String(localized: "Download stopped")
        case .other:
            return nil
        }
    }

    // MARK: - Helpers

    /// Whole percent, clamped — the byte monitor can overshoot on the final
    /// chunk, and "101%" would contradict the file it just finished.
    static func percentText(for fraction: Double?) -> String? {
        guard let fraction, fraction.isFinite else { return nil }
        return "\(Int((min(max(fraction, 0), 1) * 100).rounded()))%"
    }

    /// "1.7 GB of 2.9 GB", or "2.9 GB downloaded" when the total is unknown
    /// or has been overtaken. Nil until the disk has really been observed —
    /// we never print "0 B".
    static func bytesText(_ progress: ModelReadiness.ProgressSnapshot?) -> String? {
        guard let done = progress?.bytesDownloaded, done > 0 else { return nil }
        let doneText = DownloadProgress.formatBytes(done)
        if let total = progress?.totalBytes, total > 0, done <= total {
            return String(localized: "\(doneText) of \(DownloadProgress.formatBytes(total))")
        }
        return String(localized: "\(doneText) downloaded")
    }

    private static func loadingTitle(for activity: DownloadProgress.StartupActivity?) -> String {
        switch activity {
        case .warmingUp:
            return String(localized: "Warming up…")
        case .loading:
            return String(localized: "Loading the model into memory…")
        case .starting, .downloading, nil:
            // `.downloading` cannot normally reach a `.starting` readiness
            // (resolve maps it to `.downloading`); if it ever does, the
            // neutral phrase is the honest one.
            return String(localized: "Starting the model…")
        }
    }

    private static func joined(_ parts: String?...) -> String? {
        let present = parts.compactMap { $0 }.filter { !$0.isEmpty }
        return present.isEmpty ? nil : present.joined(separator: " · ")
    }

    /// A readiness value with everything that ticks stripped out, so two
    /// values compare equal exactly when nothing worth announcing changed.
    private enum Moment: Equatable {
        case downloading(alias: String)
        case stopped(alias: String)
        case loading(alias: String)
        case ready(alias: String)
        case failed(String)
        case other

        init(_ readiness: ModelReadiness, downloadStatus: DownloadManager.Job.Status?) {
            if downloadStatus == .cancelled {
                switch readiness {
                case .needsDownload(let alias, _), .downloading(let alias, _, _):
                    self = .stopped(alias: alias)
                    return
                default: break
                }
            }
            switch readiness {
            case .downloading(let a, _, _): self = .downloading(alias: a)
            case .starting(let a, _):       self = .loading(alias: a)
            case .ready(let a):             self = .ready(alias: a)
            case .failed:                   self = .failed(readiness.accessibilityLabel)
            case .engineMissing, .sharingCompute, .noModel, .needsDownload,
                 .needsStart, .unknownModel:
                self = .other
            }
        }

        var isWorking: Bool {
            switch self {
            case .downloading, .loading: return true
            default:                     return false
            }
        }
    }
}
