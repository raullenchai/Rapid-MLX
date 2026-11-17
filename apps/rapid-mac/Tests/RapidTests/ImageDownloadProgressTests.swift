import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Image download readiness progress")
struct ImageDownloadProgressTests {
    @Test("Download-only job updates banner percent, bytes and speed while server stays idle")
    func liveProgress() throws {
        let alias = "bonsai-image"
        let job = DownloadManager.Job(alias: alias)
        job.progress.setTotalBytes(4_000_000_000)
        let start = Date(timeIntervalSince1970: 1_000)
        for (seconds, bytes) in [(0.0, Int64(1_000_000_000)), (2.0, Int64(2_000_000_000))] {
            job.progress.applyDiskObservation(bytes: bytes, at: start.addingTimeInterval(seconds))
            let progress = try #require(ImagesView.downloadProgressSnapshot(alias: alias, job: job))
            #expect(progress.fraction == Double(bytes) / 4_000_000_000)
            #expect(progress.subtitle?.contains(DownloadProgress.formatBytes(bytes)) == true)
            #expect(progress.subtitle?.contains(DownloadProgress.formatBytes(4_000_000_000)) == true)
            #expect(progress.subtitle?.contains(bytes == 1_000_000_000 ? "25%" : "50%") == true)
            let readiness = ModelReadiness.resolve(
                serverState: .idle, alias: alias, cacheState: .notOnDisk,
                progress: progress, downloadInFlight: true
            )
            #expect(readiness == .downloading(alias: alias, detail: progress.subtitle, fraction: progress.fraction))
            if seconds > 0 {
                #expect(progress.subtitle?.contains("/s") == true)
            }
            // Audio already uses the same job parser; keep that mapping pinned.
            let audio = try #require(AudioReadinessState.downloadSnapshot(alias: alias, job: job))
            #expect(audio.status == .running(detail: progress.subtitle, fraction: progress.fraction))
        }
    }

    @Test("Absent, unrelated and terminal jobs do not supply stale banner progress")
    func noStaleProgress() {
        let downloads = DownloadManager()
        #expect(ImagesView.downloadProgressSnapshot(alias: "selected", job: nil) == nil)
        #expect(ImagesView.downloadProgressSnapshot(alias: "selected", job: DownloadManager.Job(alias: "other")) == nil)
        for exit in [(Int32(0), false), (Int32(1), true), (Int32(1), false)] {
            let job = downloads._testingSeedJob(alias: "selected", retryAttempt: 100)
            job.progress.setTotalBytes(100)
            job.progress.applyDiskObservation(bytes: 50)
            downloads._testingFinish(alias: "selected", status: exit.0, reason: .exit, wasCancelling: exit.1)
            #expect(job.status != .running)
            #expect(ImagesView.downloadProgressSnapshot(alias: "selected", job: job) == nil)
        }
    }
}
