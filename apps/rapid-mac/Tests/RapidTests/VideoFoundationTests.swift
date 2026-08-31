import Foundation
import Testing
@testable import Rapid

@Suite("Video foundation")
struct VideoFoundationTests {
    @Test("Video JSON exposes exact request modes and memory floor")
    func parsesMachineReadableVideoCapabilities() throws {
        let output = """
        {"text":[],"audio":[],"image":[],"video":[
          {"alias":"wan-ti2v","hf_path":"org/wan","video_modes":["text-to-video","image-to-video"],"min_memory_gb":32},
          {"alias":"ltx-t2v","hf_path":"org/ltx","video_modes":["text-to-video"],"min_memory_gb":24},
          {"alias":"unknown-mode","hf_path":"org/bad","video_modes":["video-to-video"],"min_memory_gb":24},
          {"alias":"duplicate-mode","hf_path":"org/bad","video_modes":["text-to-video","text-to-video"],"min_memory_gb":24},
          {"alias":"boolean-floor","hf_path":"org/bad","video_modes":["text-to-video"],"min_memory_gb":true},
          {"alias":"missing-floor","hf_path":"org/bad","video_modes":["text-to-video"]}
        ]}
        """

        let rows = ModelCatalog.parseVideoRowsJSON(output)
        #expect(rows.count == 2)
        let wan = try #require(rows.first { $0.alias == "wan-ti2v" })
        #expect(wan.hfRepo == "org/wan")
        #expect(wan.capabilities == [.textToVideo, .imageToVideo])
        #expect(wan.minimumMemoryGB == 32)
        #expect(rows.first { $0.alias == "ltx-t2v" }?.capabilities == [.textToVideo])
    }

    @Test("Video entries join machine-readable catalog to complete cache rows")
    func videoEntriesResolveCacheByRepository() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-video-catalog-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let binary = directory.appendingPathComponent("rapid-mlx")
        let script = """
        #!/bin/sh
        if [ "$1" = "models" ]; then
          printf '%s' '{"text":[],"audio":[],"image":[],"video":[{"alias":"ltx-2.3-mlx-q4","hf_path":"org/ltx","video_modes":["text-to-video","image-to-video"],"min_memory_gb":24},{"alias":"cogvideox-fun-5b-q4","hf_path":"org/cog","video_modes":["text-to-video"],"min_memory_gb":24},{"alias":"ltx-2.5-mlx-q8","hf_path":"org/ltx25","video_modes":["text-to-video","image-to-video"],"min_memory_gb":24}]}'
        else
          cat <<'EOF'
        Cached models (1 on disk)
        Alias       HF repo   Size
        (unmapped)  org/ltx   9.5 GiB
        EOF
        fi
        """
        try script.write(to: binary, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: binary.path)

        let entries = await ModelCatalog.videoEntries(binary: binary, hubCacheOverride: nil)
        let entry = try #require(entries.first)
        #expect(entries.count == 1)
        #expect(entry.alias == "ltx-2.3-mlx-q4")
        #expect(entry.kind == .video)
        #expect(entry.cached)
        #expect(entry.sizeOnDisk == "9.5 GiB")
        #expect(entry.videoCapabilities == [.textToVideo, .imageToVideo])
        #expect(entry.minimumMemoryGB == 24)
    }

    @Test("Video artifacts honor HOME isolation")
    func videoArtifactDirectoryHonorsHome() {
        let directory = ApplicationSupportLocator.videoArtifactsDirectory(
            environment: ["HOME": "/tmp/rapid-video-test"]
        )
        #expect(directory.path == "/tmp/rapid-video-test/Library/Application Support/Rapid/VideoArtifacts")
        #expect(ApplicationSupportLocator.videoArtifactsFolderName == "VideoArtifacts")
    }

    @Test("Video memory floor uses physical capacity, not current usage")
    func videoMemoryFloorUsesPhysicalCapacity() {
        let gib = UInt64(1) << 30
        let busy32GBMac = MemoryProbe.Snapshot(
            totalBytes: 32 * gib,
            usedBytes: 31 * gib
        )
        let idle16GBMac = MemoryProbe.Snapshot(
            totalBytes: 16 * gib,
            usedBytes: gib
        )

        #expect(ServerManager.videoMemoryFloorSatisfied(
            minimumMemoryGB: 32,
            snapshot: busy32GBMac
        ))
        #expect(!ServerManager.videoMemoryFloorSatisfied(
            minimumMemoryGB: 32,
            snapshot: idle16GBMac
        ))
        #expect(!ServerManager.videoMemoryFloorSatisfied(
            minimumMemoryGB: nil,
            snapshot: nil
        ))
        #expect(!ServerManager.videoMemoryFloorSatisfied(
            minimumMemoryGB: 32,
            snapshot: nil
        ))
        let estimated = ServerManager.videoEstimatedFootprintGB(
            minimumMemoryGB: 24
        ) ?? 0
        #expect(abs(estimated - 19.2) < 0.001)
        #expect(ServerManager.videoEstimatedFootprintGB(minimumMemoryGB: nil) == nil)
    }

    @MainActor
    @Test("Video startup applies the catalog working set to live admission")
    func videoStartupUsesCatalogWorkingSetForAdmission() async throws {
        let gib = UInt64(1) << 30
        let artifacts = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-video-admission-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: artifacts) }
        let server = ServerManager(
            testingState: .idle,
            binaryPath: URL(fileURLWithPath: "/usr/bin/true")
        )
        server.videoArtifactsDirectoryProvider = { artifacts }
        server.memorySnapshotProvider = {
            MemoryProbe.Snapshot(
                totalBytes: 32 * gib,
                usedBytes: 20 * gib
            )
        }

        let load = Task {
            await server.ensureVideoServing(
                alias: "ltx-2.3-mlx-q4",
                hfPath: "notapalindrome/ltx23-mlx-av-q4",
                minimumMemoryGB: 24
            )
        }
        for _ in 0 ..< 300 where server.pendingMemoryWarning == nil {
            try await Task.sleep(for: .milliseconds(10))
        }
        let warning = try #require(server.pendingMemoryWarning)
        #expect(warning.severity == .unsafe)
        #expect(abs(warning.footprintGB - 19.2) < 0.001)
        #expect(warning.videoOutputDirectory == artifacts.path)

        server.cancelPendingMemoryLoad(warning)
        #expect(await load.value == false)
    }
}
