import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("VideoGenViewModel")
struct VideoGenViewModelTests {
    @Test("Catalog fails closed to explicit Video rows and chooses a model that fits")
    func catalogFilteringAndMemoryChoice() async {
        let chat = ModelEntry(alias: "chat", hfRepo: nil, sizeOnDisk: nil, cached: true)
        let tooLarge = ModelEntry(
            alias: "video-32", hfRepo: "org/large", sizeOnDisk: nil, cached: true,
            kind: .video, videoCapabilities: [.textToVideo], minimumMemoryGB: 32
        )
        let fitting = ModelEntry(
            alias: "video-24", hfRepo: "org/fitting", sizeOnDisk: nil, cached: true,
            kind: .video,
            videoCapabilities: [.textToVideo, .imageToVideo], minimumMemoryGB: 24
        )
        let server = ServerManager(
            testingState: .idle,
            binaryPath: URL(fileURLWithPath: "/usr/bin/true")
        )
        let viewModel = VideoGenViewModel(
            server: server,
            client: VideoFakeClient(),
            physicalRAMGB: 24,
            catalogLoader: { _ in [chat, tooLarge, fitting] }
        )

        await viewModel.refreshCatalog()

        #expect(viewModel.videoModels == [tooLarge, fitting])
        #expect(viewModel.selectedAlias == "video-24")
        #expect(viewModel.isSelectedModelEligible)
        #expect(!viewModel.isModelEligible(tooLarge))
        #expect(viewModel.supportedModes == [.text])
    }

    @Test("Live capabilities gate Image mode and submission carries the reference")
    func capabilityDrivenSubmission() async throws {
        let model = ModelEntry(
            alias: "ltx-2.3-mlx-q4", hfRepo: "org/ltx", sizeOnDisk: "9 GB", cached: true,
            kind: .video,
            videoCapabilities: [.textToVideo, .imageToVideo], minimumMemoryGB: 24
        )
        let client = VideoFakeClient()
        let server = ServerManager(
            testingState: .ready(alias: model.alias),
            binaryPath: URL(fileURLWithPath: "/usr/bin/true"),
            activeBearer: "test-bearer"
        )
        let viewModel = VideoGenViewModel(
            server: server,
            client: client,
            physicalRAMGB: 32,
            catalogLoader: { _ in [model] }
        )

        await viewModel.refreshCatalog()
        await viewModel.refreshServerData()
        #expect(viewModel.size == "512x512")
        #expect(viewModel.seconds == 1)
        viewModel.seconds = 4
        viewModel.selectSize("1280x720")
        #expect(viewModel.durationPresets == [1])
        #expect(viewModel.seconds == 1)

        viewModel.selectMode(.image)
        viewModel.prompt = "Ocean waves moving around a black rock"
        #expect(!viewModel.canSubmit)
        viewModel.setReference(.init(
            data: Data("png".utf8), fileName: "rock.png", mimeType: "image/png"
        ))
        #expect(viewModel.canSubmit)

        await viewModel.submit()

        let requests = await client.recordedRequests()
        let request = try #require(requests.first)
        #expect(request.model == model.alias)
        #expect(request.size == "1280x720")
        #expect(request.seconds == 1)
        #expect(request.reference == Data("png".utf8))
        #expect(request.referenceFileName == "rock.png")
        #expect(viewModel.jobs.first?.status == .queued)
        #expect(viewModel.prompt.isEmpty)
    }

    @Test("A running model that exceeds this Mac's memory remains ineligible")
    func runningIneligibleModelCannotSubmit() async {
        let model = ModelEntry(
            alias: "ltx-2.3-mlx-q4", hfRepo: "org/ltx", sizeOnDisk: "9 GB", cached: true,
            kind: .video, videoCapabilities: [.textToVideo], minimumMemoryGB: 64
        )
        let server = ServerManager(
            testingState: .ready(alias: model.alias),
            binaryPath: URL(fileURLWithPath: "/usr/bin/true"),
            activeBearer: "test-bearer"
        )
        let viewModel = VideoGenViewModel(
            server: server,
            client: VideoFakeClient(),
            physicalRAMGB: 32,
            catalogLoader: { _ in [model] }
        )
        await viewModel.refreshCatalog()
        await viewModel.refreshServerData()
        viewModel.prompt = "A short scene"

        #expect(viewModel.isServerReady)
        #expect(!viewModel.isSelectedModelEligible)
        #expect(!viewModel.canSubmit)
    }

    @Test("A late response cannot repopulate a newly selected model")
    func staleCapabilitiesAreDiscarded() async throws {
        let first = ModelEntry(
            alias: "video-a", hfRepo: "org/a", sizeOnDisk: "9 GB", cached: true,
            kind: .video, videoCapabilities: [.textToVideo], minimumMemoryGB: 24
        )
        let second = ModelEntry(
            alias: "video-b", hfRepo: "org/b", sizeOnDisk: "9 GB", cached: true,
            kind: .video, videoCapabilities: [.textToVideo], minimumMemoryGB: 24
        )
        let client = VideoSuspendingClient(
            capabilities: try VideoFakeClient.capabilitiesValue()
        )
        let server = ServerManager(
            testingState: .ready(alias: first.alias),
            binaryPath: URL(fileURLWithPath: "/usr/bin/true"),
            activeBearer: "test-bearer"
        )
        let viewModel = VideoGenViewModel(
            server: server,
            client: client,
            physicalRAMGB: 32,
            catalogLoader: { _ in [first, second] }
        )
        await viewModel.refreshCatalog()

        let refresh = Task { await viewModel.refreshServerData() }
        await client.waitUntilCapabilitiesRequested()
        server._testSetState(.ready(alias: second.alias))
        await viewModel.serverStateDidChange()
        await client.resumeCapabilities()
        await refresh.value

        #expect(viewModel.selectedAlias == first.alias)
        #expect(viewModel.capabilities == nil)
        #expect(viewModel.jobs.isEmpty)
    }

    @Test("Unreconciled history blocks model switching until retry succeeds")
    func historyFailureBlocksModelSwitch() async {
        let first = ModelEntry(
            alias: "ltx-2.3-mlx-q4", hfRepo: "org/ltx", sizeOnDisk: "9 GB", cached: true,
            kind: .video, videoCapabilities: [.textToVideo], minimumMemoryGB: 24
        )
        let second = ModelEntry(
            alias: "video-b", hfRepo: "org/b", sizeOnDisk: "9 GB", cached: true,
            kind: .video, videoCapabilities: [.textToVideo], minimumMemoryGB: 24
        )
        let client = VideoFakeClient(listFailures: 1)
        let server = ServerManager(
            testingState: .ready(alias: first.alias),
            binaryPath: URL(fileURLWithPath: "/usr/bin/true"),
            activeBearer: "test-bearer"
        )
        let viewModel = VideoGenViewModel(
            server: server,
            client: client,
            physicalRAMGB: 32,
            catalogLoader: { _ in [first, second] }
        )
        await viewModel.refreshCatalog()
        await viewModel.refreshServerData()

        #expect(viewModel.needsServerRefresh)
        #expect(!viewModel.canSwitchModels)
        viewModel.selectModel(second.alias)
        #expect(viewModel.selectedAlias == first.alias)

        await viewModel.refreshServerData()
        #expect(viewModel.jobsAreReconciled)
        #expect(viewModel.canSwitchModels)
        viewModel.selectModel(second.alias)
        #expect(viewModel.selectedAlias == second.alias)
    }

    @Test("A stale history selection cannot display a different job")
    func staleHistorySelectionIsIgnored() async {
        let server = ServerManager(
            testingState: .idle,
            binaryPath: URL(fileURLWithPath: "/usr/bin/true")
        )
        let viewModel = VideoGenViewModel(server: server, client: VideoFakeClient())
        let job = VideoJob(
            id: "video_0123456789abcdef0123456789abcdef",
            model: "ltx", prompt: "first", seconds: "1", size: "512x512",
            status: .completed, progress: 100, createdAt: 1, completedAt: 2, error: nil
        )
        viewModel.jobs = [job]
        viewModel.selectedJobID = job.id

        await viewModel.selectJob("video_stale")

        #expect(viewModel.selectedJobID == job.id)
        #expect(viewModel.selectedJob == job)
    }

    @Test("Active jobs keep polling without a mounted Video view")
    func pollingOutlivesView() async throws {
        let model = ModelEntry(
            alias: "ltx-2.3-mlx-q4", hfRepo: "org/ltx", sizeOnDisk: "9 GB", cached: true,
            kind: .video, videoCapabilities: [.textToVideo], minimumMemoryGB: 24
        )
        let client = VideoPollingClient()
        let server = ServerManager(
            testingState: .ready(alias: model.alias),
            binaryPath: URL(fileURLWithPath: "/usr/bin/true"),
            activeBearer: "test-bearer"
        )
        let viewModel = VideoGenViewModel(
            server: server,
            client: client,
            physicalRAMGB: 32,
            pollingInterval: .milliseconds(5),
            catalogLoader: { _ in [model] }
        )
        await viewModel.refreshCatalog()
        await viewModel.refreshServerData()
        viewModel.prompt = "A fox runs through snow"

        await viewModel.submit()
        #expect(viewModel.hasLiveActiveJobs)

        for _ in 0..<100 where viewModel.hasActiveJobs {
            try await Task.sleep(for: .milliseconds(5))
        }

        #expect(viewModel.jobs.first?.status == .completed)
        #expect(!viewModel.hasLiveActiveJobs)
        #expect(viewModel.previewURL?.lastPathComponent == "finished.mp4")
        #expect(await client.listCallCount() >= 3)
    }

    @Test("A persistently missing active job expires after bounded polling")
    func missingActiveJobEventuallyExpires() async throws {
        let model = ModelEntry(
            alias: "ltx-2.3-mlx-q4", hfRepo: "org/ltx", sizeOnDisk: "9 GB", cached: true,
            kind: .video, videoCapabilities: [.textToVideo], minimumMemoryGB: 24
        )
        let client = VideoFakeClient()
        let server = ServerManager(
            testingState: .ready(alias: model.alias),
            binaryPath: URL(fileURLWithPath: "/usr/bin/true"),
            activeBearer: "test-bearer"
        )
        let viewModel = VideoGenViewModel(
            server: server,
            client: client,
            physicalRAMGB: 32,
            pollingInterval: .milliseconds(5),
            catalogLoader: { _ in [model] }
        )
        await viewModel.refreshCatalog()
        await viewModel.refreshServerData()
        viewModel.prompt = "A fox runs through snow"
        await viewModel.submit()

        for _ in 0..<100 where viewModel.hasActiveJobs {
            try await Task.sleep(for: .milliseconds(5))
        }

        #expect(!viewModel.hasActiveJobs)
        #expect(viewModel.jobs.isEmpty)
        #expect(await client.listCallCount() >= 7)
    }

    @Test("A stale active job stops blocking global busy after a server switch")
    func staleJobIsNotGloballyBusy() async {
        let model = ModelEntry(
            alias: "ltx-2.3-mlx-q4", hfRepo: "org/ltx", sizeOnDisk: "9 GB", cached: true,
            kind: .video, videoCapabilities: [.textToVideo], minimumMemoryGB: 24
        )
        let server = ServerManager(
            testingState: .ready(alias: model.alias),
            binaryPath: URL(fileURLWithPath: "/usr/bin/true"),
            activeBearer: "test-bearer"
        )
        let viewModel = VideoGenViewModel(
            server: server,
            client: VideoFakeClient(),
            physicalRAMGB: 32,
            pollingInterval: .seconds(60),
            catalogLoader: { _ in [model] }
        )
        await viewModel.refreshCatalog()
        await viewModel.refreshServerData()
        viewModel.prompt = "A fox runs through snow"
        await viewModel.submit()
        #expect(viewModel.hasLiveActiveJobs)

        server._testSetState(.ready(alias: "chat-model"))
        await viewModel.serverStateDidChange()

        #expect(viewModel.hasActiveJobs)
        #expect(!viewModel.hasLiveActiveJobs)
    }
}

private actor VideoPollingClient: VideoClientProtocol {
    private var listCalls = 0

    func capabilities(port: Int, bearer: String?) async throws -> VideoCapabilities {
        try VideoFakeClient.capabilitiesValue()
    }

    func create(
        _ request: VideoCreateRequest,
        port: Int,
        bearer: String?
    ) async throws -> VideoJob {
        Self.job(status: .queued, progress: 0)
    }

    func list(port: Int, bearer: String?, limit: Int) async throws -> [VideoJob] {
        listCalls += 1
        return listCalls <= 2 ? [] : [Self.job(status: .completed, progress: 100)]
    }

    func delete(id: String, port: Int, bearer: String?) async throws {}

    func content(id: String, port: Int, bearer: String?) async throws -> URL {
        URL(fileURLWithPath: "/tmp/finished.mp4")
    }

    func listCallCount() -> Int { listCalls }

    private nonisolated static func job(status: VideoJobStatus, progress: Int) -> VideoJob {
        VideoJob(
            id: "video_0123456789abcdef0123456789abcdef",
            model: "ltx-2.3-mlx-q4",
            prompt: "A fox runs through snow",
            seconds: "1",
            size: "512x512",
            status: status,
            progress: progress,
            createdAt: 123,
            completedAt: status == .completed ? 456 : nil,
            error: nil
        )
    }
}

private actor VideoFakeClient: VideoClientProtocol {
    private var requests: [VideoCreateRequest] = []
    private var listFailures: Int
    private var listCalls = 0

    init(listFailures: Int = 0) {
        self.listFailures = listFailures
    }

    func capabilities(port: Int, bearer: String?) async throws -> VideoCapabilities {
        try JSONDecoder().decode(VideoCapabilities.self, from: Data(Self.capabilitiesJSON.utf8))
    }

    func create(
        _ request: VideoCreateRequest,
        port: Int,
        bearer: String?
    ) async throws -> VideoJob {
        requests.append(request)
        return VideoJob(
            id: "video_0123456789abcdef0123456789abcdef",
            model: request.model,
            prompt: request.prompt,
            seconds: String(request.seconds),
            size: request.size,
            status: .queued,
            progress: 0,
            createdAt: 123,
            completedAt: nil,
            error: nil
        )
    }

    func list(port: Int, bearer: String?, limit: Int) async throws -> [VideoJob] {
        listCalls += 1
        if listFailures > 0 {
            listFailures -= 1
            throw VideoClientError.transport("history unavailable")
        }
        return []
    }
    func delete(id: String, port: Int, bearer: String?) async throws {}
    func content(id: String, port: Int, bearer: String?) async throws -> URL {
        URL(fileURLWithPath: "/tmp/\(id).mp4")
    }

    func recordedRequests() -> [VideoCreateRequest] { requests }
    func listCallCount() -> Int { listCalls }

    nonisolated static func capabilitiesValue() throws -> VideoCapabilities {
        try JSONDecoder().decode(VideoCapabilities.self, from: Data(capabilitiesJSON.utf8))
    }

    private static let capabilitiesJSON = #"""
    {
      "model":"org/ltx","family":"ltx-2.3",
      "modes":["text-to-video","image-to-video"],
      "limits":{
        "size":{"type":"range","width":{"minimum":256,"maximum":1920,"multiple_of":64},"height":{"minimum":256,"maximum":1920,"multiple_of":64},"also_supported":["1280x720","720x1280"]},
        "seconds":{"minimum":1,"maximum":20,"default":4},
        "fps":{"minimum":1,"maximum":60,"default":24,"fixed":false},
        "frames":{"minimum":9,"maximum":1201,"step":8,"offset":1},
        "workload":{"metric":"pixel_frames","maximum":38141952,"dimension_rounding":"ceil_to_64"},
        "input_reference":{"maximum_bytes":20971520,"maximum_pixels":16777216,"formats":["jpeg","png","webp"]}
      }
    }
    """#
}

private actor VideoSuspendingClient: VideoClientProtocol {
    private let value: VideoCapabilities
    private var capabilitiesRequested = false
    private var capabilitiesContinuation: CheckedContinuation<VideoCapabilities, Error>?

    init(capabilities: VideoCapabilities) {
        value = capabilities
    }

    func capabilities(port: Int, bearer: String?) async throws -> VideoCapabilities {
        capabilitiesRequested = true
        return try await withCheckedThrowingContinuation { continuation in
            capabilitiesContinuation = continuation
        }
    }

    func waitUntilCapabilitiesRequested() async {
        while !capabilitiesRequested { await Task.yield() }
    }

    func resumeCapabilities() {
        capabilitiesContinuation?.resume(returning: value)
        capabilitiesContinuation = nil
    }

    func create(
        _ request: VideoCreateRequest,
        port: Int,
        bearer: String?
    ) async throws -> VideoJob {
        throw VideoClientError.invalidResponse
    }

    func list(port: Int, bearer: String?, limit: Int) async throws -> [VideoJob] { [] }
    func delete(id: String, port: Int, bearer: String?) async throws {}
    func content(id: String, port: Int, bearer: String?) async throws -> URL {
        throw VideoClientError.invalidResponse
    }
}
