import Darwin
import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Benchmark runtime launch")
struct BenchmarkRuntimeLaunchTests {
    @Test("Benchmark invokes the bundled launcher with its Python environment")
    func bundledLauncherEnvironment() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-benchmark-runtime-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let bin = root.appendingPathComponent("bin")
        let pythonBin = root.appendingPathComponent("python/bin")
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: pythonBin, withIntermediateDirectories: true)
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let launcher = bin.appendingPathComponent("rapid-mlx")
        try FileManager.default.copyItem(
            at: packageRoot.appendingPathComponent("scripts/sidecar-shim.sh"), to: launcher
        )
        chmod(launcher.path, 0o755)
        let python = pythonBin.appendingPathComponent("python3.12")
        try """
        #!/bin/sh
        /usr/bin/env
        printf 'ARG=%s\\n' "$@"
        """.write(to: python, atomically: true, encoding: .utf8)
        chmod(python.path, 0o755)
        let output = try await CommunityBenchmarkCommand.run(
            binary: launcher, arguments: ["benchmark", "catalog", "--json"]
        )
        let lines = Set(String(decoding: output, as: UTF8.self).split(separator: "\n").map(String.init))
        let resolvedRoot = root.resolvingSymlinksInPath().path
        #expect(lines.contains("PYTHONHOME=\(resolvedRoot)/python"))
        #expect(lines.contains("PYTHONPATH=\(resolvedRoot)/site-packages"))
        #expect(lines.contains("PYTHONNOUSERSITE=1"))
        #expect(lines.contains("RAPID_MLX_PROCESS_ROLE=desktop-sidecar"))
        // This default comes from the same environment helper used by serve.
        #expect(lines.contains("RAPID_MLX_PREFIX_CACHE_AUTOLOAD=0"))
        #expect(lines.contains("ARG=rapid_mlx.cli"))
        #expect(lines.contains("ARG=catalog"))
    }

    @Test("Integration discovery uses the shared server environment")
    func integrationEnvironment() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-integration-runtime-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let binary = root.appendingPathComponent("rapid-mlx")
        try """
        #!/bin/sh
        [ "$RAPID_MLX_PREFIX_CACHE_AUTOLOAD" = 0 ] || exit 1
        [ "$RAPID_MLX_PROCESS_ROLE" = desktop-sidecar ] || exit 1
        [ "$RAPID_MLX_TELEMETRY" = 0 ] || exit 1
        echo '[{"id":"fixture","name":"Fixture","kind":"config_writer"}]'
        """.write(to: binary, atomically: true, encoding: .utf8)
        chmod(binary.path, 0o755)
        let targets = await IntegrationCatalog.load(binary: binary)
        #expect(targets.map(\.id) == ["fixture"])
    }

    @Test("Catalog command failures retain the actual stderr detail")
    func catalogFailureDetail() async throws {
        do {
            _ = try await CommunityBenchmarkCommand.run(
                binary: URL(fileURLWithPath: "/bin/sh"),
                arguments: ["-c", "echo \"ModuleNotFoundError: No module named 'pydantic'\" >&2; exit 1"]
            )
            Issue.record("Expected catalog failure")
        } catch {
            #expect(error.localizedDescription == "ModuleNotFoundError: No module named 'pydantic'")
        }
    }
}
