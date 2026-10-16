import AppKit
@preconcurrency import AVFoundation
@preconcurrency import QuickLookThumbnailing
import SwiftUI

/// Two generators at most; identical cells share work. Canceling the last
/// consumer cancels native generation or skips queued work before it starts.
final class LocalSearchThumbnailCache: @unchecked Sendable {
    static let shared = LocalSearchThumbnailCache()
    private let images = NSCache<NSString, NSImage>()
    typealias FrameLoader = @Sendable (String, Double, CGSize) -> CGImage?
    typealias QuickLookLoader = @Sendable (String, CGSize, @escaping @Sendable (CGImage?) -> Void) -> Void
    private let frameLoader: FrameLoader?
    private let quickLookLoader: QuickLookLoader?
    private let queue = DispatchQueue(label: "rapid.local-search.thumbnails")
    private let maximumConcurrent = 2
    private var active = 0
    private var jobs: [String: Job] = [:]
    private var pending: [String] = []

    private final class Consumer: @unchecked Sendable {
        private let lock = NSLock()
        private var canceled = false
        func cancel() { lock.lock(); canceled = true; lock.unlock() }
        var isCanceled: Bool { lock.lock(); defer { lock.unlock() }; return canceled }
    }
    private final class Job: @unchecked Sendable {
        let key: String
        let path: String
        let timestamp: Double?
        let size: CGSize
        var waiters: [UUID: CheckedContinuation<Delivered, Never>] = [:]
        var running = false
        var cancel: (@Sendable () -> Void)?
        init(key: String, path: String, timestamp: Double?, size: CGSize) {
            self.key = key; self.path = path; self.timestamp = timestamp; self.size = size
        }
    }
    init(frameLoader: FrameLoader? = nil, quickLookLoader: QuickLookLoader? = nil,
         cacheCountLimit: Int = 160, cacheCostLimit: Int = 32 * 1024 * 1024) {
        self.frameLoader = frameLoader
        self.quickLookLoader = quickLookLoader
        images.countLimit = cacheCountLimit
        images.totalCostLimit = cacheCostLimit
    }
    func thumbnail(path: String, modifiedAt: Double, size: Int, timestamp: Double? = nil,
                   pixelSize: CGSize = CGSize(width: 320, height: 240)) async -> NSImage? {
        await deliveredThumbnail(path: path, modifiedAt: modifiedAt, size: size,
                                 timestamp: timestamp, pixelSize: pixelSize).image
    }
    /// Main-actor callers receive the image through a Sendable box: Swift 6.1
    /// rejects sending non-Sendable `NSImage?` across isolation boundaries.
    struct Delivered: @unchecked Sendable { let image: NSImage? }
    func deliveredThumbnail(path: String, modifiedAt: Double, size: Int, timestamp: Double? = nil,
                            pixelSize: CGSize = CGSize(width: 320, height: 240)) async -> Delivered {
        guard !Task.isCancelled else { return Delivered(image: nil) }
        let key = "\(path):\(modifiedAt):\(size):\(timestamp.map { String($0) } ?? "still"):\(pixelSize.width)x\(pixelSize.height)"
        let id = UUID(), consumer = Consumer()
        return await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                queue.async {
                    guard !consumer.isCanceled else { continuation.resume(returning: Delivered(image: nil)); return }
                    if let image = self.images.object(forKey: key as NSString) {
                        continuation.resume(returning: Delivered(image: image)); return
                    }
                    let job: Job
                    if let existing = self.jobs[key], !existing.waiters.isEmpty { job = existing }
                    else if self.jobs[key] != nil {
                        // A canceled native generator is still draining its callback.
                        continuation.resume(returning: Delivered(image: nil)); return
                    } else {
                        job = Job(key: key, path: path, timestamp: timestamp, size: pixelSize)
                        self.jobs[key] = job
                        self.pending.append(key)
                    }
                    job.waiters[id] = continuation
                    self.pump()
                }
            }
        } onCancel: {
            consumer.cancel()
            self.queue.async {
                guard let job = self.jobs[key] else { return }
                job.waiters.removeValue(forKey: id)?.resume(returning: Delivered(image: nil))
                if job.waiters.isEmpty {
                    if job.running { job.cancel?() }
                    else {
                        self.jobs.removeValue(forKey: key)
                        self.pending.removeAll { $0 == key }
                    }
                }
            }
        }
    }
    private func pump() {
        while active < maximumConcurrent, !pending.isEmpty {
            let key = pending.removeFirst()
            guard let job = jobs[key], !job.waiters.isEmpty else { continue }
            job.running = true
            active += 1
            DispatchQueue.global(qos: .userInitiated).async { self.generate(job) }
        }
    }
    private func installCancellation(_ job: Job, _ cancel: @escaping @Sendable () -> Void) {
        queue.async {
            guard self.jobs[job.key] === job else { cancel(); return }
            job.cancel = cancel
            if job.waiters.isEmpty { cancel() }
        }
    }
    private func finish(_ job: Job, _ frame: CGImage?) {
        queue.async {
            guard self.jobs[job.key] === job else { return }
            var image: NSImage?
            if !job.waiters.isEmpty, let frame {
                image = NSImage(cgImage: frame, size: CGSize(width: frame.width, height: frame.height))
                self.images.setObject(image!, forKey: job.key as NSString, cost: frame.bytesPerRow * frame.height)
            }
            let delivered = Delivered(image: image)
            for waiter in job.waiters.values { waiter.resume(returning: delivered) }
            self.jobs.removeValue(forKey: job.key)
            self.active -= 1
            self.pump()
        }
    }
    private func generate(_ job: Job) {
        queue.async {
            guard !job.waiters.isEmpty else { self.finish(job, nil); return }
            DispatchQueue.global(qos: .userInitiated).async {
                if let time = job.timestamp, time.isFinite, time >= 0 {
                    if let loader = self.frameLoader {
                        if let frame = loader(job.path, time, job.size) { self.finish(job, frame) }
                        else { self.fallback(job) }
                    } else {
                        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: URL(fileURLWithPath: job.path)))
                        generator.appliesPreferredTrackTransform = true
                        generator.maximumSize = job.size
                        generator.requestedTimeToleranceBefore = .zero
                        generator.requestedTimeToleranceAfter = .zero
                        generator.generateCGImagesAsynchronously(forTimes: [NSValue(time: CMTime(seconds: time, preferredTimescale: 600))]) { _, image, _, result, _ in
                            if result == .cancelled { self.finish(job, nil) }
                            else if let image { self.finish(job, image) }
                            else { self.fallback(job) }
                        }
                        self.installCancellation(job) {
                            generator.cancelAllCGImageGeneration()
                            self.finish(job, nil)
                        }
                    }
                } else { self.fallback(job) }
            }
        }
    }
    private func fallback(_ job: Job) {
        queue.async {
            guard !job.waiters.isEmpty else { self.finish(job, nil); return }
            DispatchQueue.global(qos: .userInitiated).async {
                if let loader = self.quickLookLoader {
                    loader(job.path, job.size) { self.finish(job, $0) }
                } else {
                    let request = QLThumbnailGenerator.Request(fileAt: URL(fileURLWithPath: job.path),
                        size: job.size, scale: 1, representationTypes: .thumbnail)
                    QLThumbnailGenerator.shared.generateBestRepresentation(for: request) { representation, _ in
                        self.finish(job, representation?.cgImage)
                    }
                    self.installCancellation(job) {
                        QLThumbnailGenerator.shared.cancel(request)
                        // Cancellation need not deliver a representation callback.
                        self.finish(job, nil)
                    }
                }
            }
        }
    }
}

struct LocalSearchThumbnail: View {
    let result: LocalSearchResult
    @State private var image: NSImage?
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 6).fill(RapidTheme.hoverFill)
            if let image {
                Image(nsImage: image).resizable().scaledToFit()
            } else {
                Image(systemName: result.kind == .video ? "video" : "photo")
                    .font(.title2).foregroundStyle(.secondary)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .accessibilityHidden(true)
        .task(id: "\(result.path):\(result.modifiedAt):\(result.size):\(result.timestamp ?? -1)") {
            image = nil
            let delivered = await LocalSearchThumbnailCache.shared.deliveredThumbnail(
                path: result.path, modifiedAt: result.modifiedAt, size: result.size,
                timestamp: result.kind == .video ? result.timestamp : nil)
            guard !Task.isCancelled else { return }
            image = delivered.image
        }
    }
}
