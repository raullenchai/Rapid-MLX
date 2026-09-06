import Foundation

/// A normalized, pure lifecycle for the model behind an Audio workflow.
///
/// Speech and Dictation observe the same catalog, download and serving facts,
/// but used to assign precedence to those facts inside separate SwiftUI view
/// bodies. Keeping the reducer here makes the full lifecycle directly testable
/// and, importantly, rejects completions that belong to a previously selected
/// model.
enum AudioReadinessState: Equatable {
    case catalogPending
    case noModel
    case unknownModel(alias: String)
    case notDownloaded(alias: String, sizeText: String?)
    case downloading(alias: String, detail: String?, fraction: Double?)
    case verifyingDownload(alias: String)
    case downloaded(alias: String)
    case loading(alias: String, detail: String?)
    case ready(alias: String)
    case active(alias: String, activity: Activity)
    case failed(alias: String, message: String)

    enum Activity: Equatable {
        case loadingVoices
        case synthesizing
        case previewingVoice
        case startingCapture
        case recording
        case transcribing
    }

    enum DownloadStatus: Equatable {
        case running(detail: String?, fraction: Double?)
        case completed
        case failed(message: String)
        case cancelled
    }

    struct DownloadSnapshot: Equatable {
        var alias: String
        var status: DownloadStatus
    }

    struct ActivitySnapshot: Equatable {
        var alias: String
        var activity: Activity
    }

    struct LoadingSnapshot: Equatable {
        var alias: String
        var detail: String?
    }

    struct Snapshot: Equatable {
        var alias: String
        var catalogLoaded: Bool
        /// `nil` means a loaded catalog did not contain the selected alias.
        var cached: Bool?
        var sizeText: String?
        var download: DownloadSnapshot?
        var loading: LoadingSnapshot?
        var readyAlias: String?
        var activity: ActivitySnapshot?

        init(
            alias: String,
            catalogLoaded: Bool,
            cached: Bool?,
            sizeText: String? = nil,
            download: DownloadSnapshot? = nil,
            loading: LoadingSnapshot? = nil,
            readyAlias: String? = nil,
            activity: ActivitySnapshot? = nil
        ) {
            self.alias = alias
            self.catalogLoaded = catalogLoaded
            self.cached = cached
            self.sizeText = sizeText
            self.download = download
            self.loading = loading
            self.readyAlias = readyAlias
            self.activity = activity
        }
    }

    static func resolve(_ snapshot: Snapshot) -> Self {
        let alias = snapshot.alias
        guard !alias.isEmpty else { return .noModel }
        guard snapshot.catalogLoaded else { return .catalogPending }
        guard let cached = snapshot.cached else { return .unknownModel(alias: alias) }

        let matchingDownload = snapshot.download.flatMap {
            $0.alias == alias ? $0.status : nil
        }
        let matchingActivity = snapshot.activity.flatMap {
            $0.alias == alias ? $0.activity : nil
        }
        let matchingLoad = snapshot.loading.flatMap {
            $0.alias == alias ? $0 : nil
        }
        let isReady = snapshot.readyAlias == alias

        // Active work is meaningful only after the selected lane is ready.
        // A stale task from model A must never make newly selected model B look
        // active or ready.
        if isReady, let matchingActivity {
            return .active(alias: alias, activity: matchingActivity)
        }
        if case .running(let detail, let fraction) = matchingDownload {
            return .downloading(alias: alias, detail: detail, fraction: fraction)
        }
        if !cached {
            if case .completed = matchingDownload {
                // A successful pull is not proof of a usable checkpoint. The
                // catalog refresh owns the transition to `downloaded`.
                return .verifyingDownload(alias: alias)
            }
            if case .failed(let message) = matchingDownload {
                return .failed(alias: alias, message: message)
            }
        }
        if let matchingLoad {
            return cached
                ? .loading(alias: alias, detail: matchingLoad.detail)
                : .downloading(alias: alias, detail: "Starting the download…", fraction: nil)
        }
        if isReady, cached { return .ready(alias: alias) }

        if !cached {
            switch matchingDownload {
            case .cancelled, nil:
                return .notDownloaded(alias: alias, sizeText: snapshot.sizeText)
            case .running, .completed, .failed:
                // Handled above so working state outranks every stale result.
                preconditionFailure("download states resolve before the fallback")
            }
        }

        return .downloaded(alias: alias)
    }

    @MainActor
    static func downloadSnapshot(
        alias: String,
        job: DownloadManager.Job?
    ) -> DownloadSnapshot? {
        guard let job else { return nil }
        let status: DownloadStatus
        switch job.status {
        case .running:
            status = .running(
                detail: job.progress.progressSubtitle,
                fraction: job.progress.progressFraction
            )
        case .completed:
            status = .completed
        case .failed(let message):
            status = .failed(message: message)
        case .cancelled:
            status = .cancelled
        }
        return .init(alias: alias, status: status)
    }

    /// States that conclusively override the shared server-level readiness.
    /// `downloaded`, pending and unknown states intentionally fall through to
    /// ``ModelReadiness.resolve``, which owns process/crash semantics.
    var modelReadinessOverride: ModelReadiness? {
        switch self {
        case .notDownloaded(let alias, let sizeText):
            return .needsDownload(alias: alias, sizeText: sizeText)
        case .downloading(let alias, let detail, let fraction):
            return .downloading(alias: alias, detail: detail, fraction: fraction)
        case .verifyingDownload(let alias):
            return .starting(alias: alias, detail: "Finishing the download…")
        case .loading(let alias, let detail):
            return .starting(alias: alias, detail: detail)
        case .ready(let alias), .active(let alias, _):
            return .ready(alias: alias)
        case .failed(let alias, let message):
            return .failed(alias: alias, message: message, action: .retry(alias: alias))
        case .catalogPending, .noModel, .unknownModel, .downloaded:
            return nil
        }
    }

    /// Model changes are safe while setup is idle (including a download that
    /// does not own runtime memory), but never during model loading or an
    /// active capture/generation operation.
    var allowsModelSelection: Bool {
        switch self {
        case .loading, .active:
            return false
        case .catalogPending, .noModel, .unknownModel, .notDownloaded,
            .downloading, .verifyingDownload, .downloaded, .ready, .failed:
            return true
        }
    }
}
