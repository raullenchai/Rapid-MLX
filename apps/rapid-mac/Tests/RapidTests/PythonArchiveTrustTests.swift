import Foundation
import Testing

@Suite("Python archive trust")
struct PythonArchiveTrustTests {
    private var script: String {
        get throws {
            let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
                .deletingLastPathComponent().deletingLastPathComponent()
                .appendingPathComponent("scripts/build-sidecar.sh")
            return try String(contentsOf: url, encoding: .utf8)
        }
    }

    @Test("Archive verification preserves valid installs", arguments: ["fresh", "cached", "replace", "bad-download", "bad-format", "transport"])
    func archiveGate(scenario: String) async throws {
        let source = try script
        let start = try #require(source.range(of: "PBS_URL="))
        let end = try #require(source.range(of: "# ----- step 1.5:"))
        let section = String(source[start.lowerBound..<end.lowerBound])
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let harness = #"""
        set -euo pipefail
        cd "$1"
        scenario="$2"
        mkdir -p fixture/python/bin stage out
        printf '#!/bin/sh\nexit 0\n' > fixture/python/bin/python3.12
        chmod +x fixture/python/bin/python3.12
        tar -czf good.tar.gz -C fixture python
        PBS_SHA256=$(shasum -a 256 good.tar.gz | awk '{print $1}')
        PBS_VERSION=test
        PBS_TAG=test
        OUT_DIR="$PWD/out"
        STAGE="$PWD/stage"
        cached="$OUT_DIR/cpython-test-test-${PBS_SHA256}.tar.gz"
        case "$scenario" in
          cached) cp good.tar.gz "$cached" ;;
          replace) printf 'altered' > "$cached" ;;
          bad-format)
            printf 'not a tar archive' > good.tar.gz
            PBS_SHA256=$(shasum -a 256 good.tar.gz | awk '{print $1}') ;;
        esac
        curl() {
          touch downloaded
          if [ "$scenario" = transport ]; then return 22; fi
          while [ "$1" != '-o' ]; do shift; done
          if [ "$scenario" = bad-download ]; then
            printf 'altered' > "$2"
          else
            cp good.tar.gz "$2"
          fi
        }
        """#
        let file = root.appendingPathComponent("run.sh")
        try (harness + "\n" + section).write(to: file, atomically: true, encoding: .utf8)
        let process = try await TestSubprocess.run(
            executableURL: URL(fileURLWithPath: "/bin/bash"),
            arguments: [file.path, root.path, scenario]
        )
        let accepted = ["fresh", "cached", "replace"].contains(scenario)
        #expect((process.terminationStatus == 0) == accepted)
        #expect(FileManager.default.isExecutableFile(atPath: root.appendingPathComponent("stage/python/bin/python3.12").path) == accepted)
        #expect(FileManager.default.fileExists(atPath: root.appendingPathComponent("downloaded").path) == (scenario != "cached"))
        let cachedFiles = try FileManager.default.contentsOfDirectory(atPath: root.appendingPathComponent("out").path)
        #expect(cachedFiles.count == (accepted ? 1 : 0))
    }

    @Test("Version overrides require a reviewed digest", arguments: ["missing", "malformed", "reviewed", "default"])
    func configuration(scenario: String) async throws {
        let source = try script
        let start = try #require(source.range(of: "PBS_TAG="))
        let end = try #require(source.range(of: "# compileall uses"))
        var environment = ProcessInfo.processInfo.environment
        environment.removeValue(forKey: "PBS_SHA256")
        environment["PBS_TAG"] = scenario == "default" ? "20260610" : "custom"
        environment["PBS_VERSION"] = "3.12.13"
        if scenario == "malformed" { environment["PBS_SHA256"] = "invalid" }
        if scenario == "reviewed" { environment["PBS_SHA256"] = String(repeating: "a", count: 64) }
        let process = try await TestSubprocess.run(
            executableURL: URL(fileURLWithPath: "/bin/bash"),
            arguments: ["-c", "set -euo pipefail\n" + String(source[start.lowerBound..<end.lowerBound])],
            environment: environment
        )
        #expect((process.terminationStatus == 0) == ["reviewed", "default"].contains(scenario))
    }
}
