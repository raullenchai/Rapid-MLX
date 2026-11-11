import Darwin
import Foundation
import Testing
@testable import Rapid

@MainActor
struct DownloadCacheAccessTests {
    private let denied = CachedModelDiscoveryFailure(volumeName: "Models SSD")

    private func settle(_ job: DownloadManager.Job, until ready: () -> Bool = { true }) async throws {
        // Generous bound: the parallel suite can delay the main-actor hop.
        let deadline = ContinuousClock.now.advanced(by: .seconds(15))
        while (job.status == .running || !ready()) && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(job.status != .running)
    }

    @Test("Unreadable cache volume prevents the pull; Retry rechecks access")
    func preflightAndRetry() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("harbor-desk-5765-download-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let binary = root.appendingPathComponent("pull")
        let marker = root.appendingPathComponent("started")
        try Data("#!/bin/sh\ntouch '\(marker.path)'\n".utf8).write(to: binary)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: binary.path)
        var checks = 0
        var readable = false
        let manager = DownloadManager(binaryPath: binary, binaryLocator: { binary }, cacheAccessCheck: {
            checks += 1
            guard !readable else { return nil }
            return denied
        })
        defer { manager.beginShutdown(); manager.finishShutdown() }
        #expect(manager.startDownload(alias: "starter"))
        let first = try #require(manager.job(for: "starter"))
        try await settle(first)
        #expect(first.status == .failed(message: denied.downloadMessage))
        #expect(first.cacheAccessFailure == denied)
        #expect(first.failureCause == .other)
        #expect(!FileManager.default.fileExists(atPath: marker.path))
        #expect(checks == 1)
        #expect(manager.retryDownload(alias: "starter"))
        let second = try #require(manager.job(for: "starter"))
        #expect(second.instanceID != first.instanceID)
        try await settle(second, until: { checks == 2 })
        #expect(second.status == first.status)
        #expect(checks == 2)
        #expect(!FileManager.default.fileExists(atPath: marker.path))
        readable = true
        #expect(manager.retryDownload(alias: "starter"))
        let third = try #require(manager.job(for: "starter"))
        try await settle(third, until: { checks == 3 })
        #expect(checks == 3)
        #expect(third.status == .completed)
        #expect(FileManager.default.fileExists(atPath: marker.path))
    }

    @Test("Stalled pulls classify unreadable volumes before retrying", arguments: [0, 3])
    func unreadableStall(attempt: Int) async throws {
        let manager = DownloadManager(cacheAccessCheck: { denied })
        let job = manager._testingSeedJob(alias: "starter", retryAttempt: attempt)
        manager._testingFinish(alias: job.alias, status: 9, reason: .uncaughtSignal, wasStalled: true)
        try await settle(job)
        #expect(job.status == .failed(message: denied.downloadMessage))
        #expect(job.failureCause != .network)
        #expect(job.retryDelaySeconds == nil)
        #expect(manager.canRetryDownload(alias: job.alias))
    }

    @Test("Readable cache retains the existing terminal network stall message")
    func readableStall() async throws {
        let manager = DownloadManager(cacheAccessCheck: { nil })
        let job = manager._testingSeedJob(alias: "starter", retryAttempt: 3)
        manager._testingFinish(alias: job.alias, status: 9, reason: .uncaughtSignal, wasStalled: true)
        try await settle(job)
        #expect(job.status == .failed(
            message: "The download stopped making progress. Check your network and try again."
        ))
        #expect(job.failureCause == .network)
        #expect(job.cacheAccessFailure == nil)
    }

    @Test("Cancellation during preflight prevents a late pull")
    func cancelledPreflight() async throws {
        let binary = URL(fileURLWithPath: "/usr/bin/true")
        let manager = DownloadManager(binaryPath: binary, binaryLocator: { binary }, cacheAccessCheck: {
            try? await Task.sleep(for: .milliseconds(50))
            return nil
        })
        #expect(manager.startDownload(alias: "starter"))
        let job = try #require(manager.job(for: "starter"))
        try await Task.sleep(for: .milliseconds(10))
        manager.cancelDownload(alias: job.alias)
        try await Task.sleep(for: .milliseconds(100))
        #expect(job.status == .cancelled)
        #expect(manager.cacheGeneration == 0)
    }

    @Test("Pending volume consent is classified as the same permission failure")
    func pendingConsentClassification() async {
        let failure = await CachedModelDiscovery.downloadAccessFailure(
                blockingIO: DiscoveryBlockingIO(),
            requestedOverride: nil, environment: ["HF_HUB_CACHE": "/fake/hub"],
            directoryProbe: { _ in throw DiscoveryDeadline.Expired() },
            volumeResolver: { _ in "Models SSD" },
            readDirectory: { _ in Issue.record("Pending cache must not be enumerated") }
        )
        #expect(failure == denied)
    }

    @Test("Read denial uses the same volume classification for default and saved caches", arguments: [false, true])
    func deniedDirectoryRead(savedOverride: Bool) async {
        let requested = savedOverride ? URL(fileURLWithPath: "/fake/custom-cache") : nil
        let errors: [NSError] = [
            NSError(domain: NSPOSIXErrorDomain, code: Int(EPERM)),
            NSError(domain: NSPOSIXErrorDomain, code: Int(EACCES)),
            CocoaError(.fileReadNoPermission) as NSError,
        ]
        for error in errors {
            let failure = await CachedModelDiscovery.downloadAccessFailure(
                blockingIO: DiscoveryBlockingIO(),
                requestedOverride: requested, environment: ["HF_HUB_CACHE": "/fake/hub"],
                directoryProbe: { _ in true }, volumeResolver: { _ in "Models SSD" },
                readDirectory: { url in
                    #expect(url.path == (savedOverride ? "/fake/custom-cache" : "/fake/hub"))
                    throw error
                }
            )
            #expect(failure == denied)
        }
    }

    @Test("Readable or absent cache permits pulling", arguments: [false, true])
    func readableOrAbsentCache(absent: Bool) async {
        let failure = await CachedModelDiscovery.downloadAccessFailure(
                blockingIO: DiscoveryBlockingIO(),
            requestedOverride: nil, environment: ["HF_HUB_CACHE": "/fake/hub"],
            directoryProbe: { _ in
                if absent { throw NSError(domain: NSPOSIXErrorDomain, code: Int(ENOENT)) }
                return true
            },
            volumeResolver: { _ in "Models SSD" }, readDirectory: { _ in }
        )
        #expect(failure == nil)
    }
}
