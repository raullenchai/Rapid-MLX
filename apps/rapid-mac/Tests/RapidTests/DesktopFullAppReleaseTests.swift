import Foundation
import Testing

@Suite("Full-app release assembly")
struct DesktopFullAppReleaseTests {
    @Test("Full releases and local engine-less builds pass the configuration guard", arguments: [
        ["SKIP_SIDECAR": "0", "RAPID_MLX_OFFICIAL_RELEASE": "1", "CODESIGN_IDENTITY": "release-identity"],
        ["SKIP_SIDECAR": "1", "RAPID_MLX_OFFICIAL_RELEASE": "0", "CODESIGN_IDENTITY": "-"],
    ])
    func supportedConfigurations(settings: [String: String]) async throws {
        let script = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("scripts/build.sh")
        let source = try String(contentsOf: script, encoding: .utf8)
        // Execute the real configuration section without compiling an entire app.
        let end = try #require(source.range(of: "SIDECAR_SCRIPT="))
        let configuration = String(source[..<end.lowerBound])
        #expect(configuration.contains("signed releases require a bundled engine"))
        var environment = ProcessInfo.processInfo.environment
        environment.merge(settings) { _, new in new }
        let result = try await TestSubprocess.run(
            executableURL: URL(fileURLWithPath: "/bin/bash"),
            arguments: ["-c", configuration, script.path], environment: environment
        )
        #expect(result.terminationStatus == 0)
    }

    @Test("Signed and official releases cannot omit the replacement engine", arguments: [
        ["RAPID_MLX_OFFICIAL_RELEASE": "1"], ["CODESIGN_IDENTITY": "release-identity"],
    ])
    func releaseRequiresEngine(settings: [String: String]) async throws {
        let script = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("scripts/build.sh")
        var environment = ProcessInfo.processInfo.environment
        environment["SKIP_SIDECAR"] = "1"
        environment["RAPID_MLX_OFFICIAL_RELEASE"] = "0"
        environment["CODESIGN_IDENTITY"] = "-"
        environment.merge(settings) { _, new in new }
        let result = try await TestSubprocess.run(
            executableURL: URL(fileURLWithPath: "/bin/bash"), arguments: [script.path],
            environment: environment
        )
        #expect(result.terminationStatus == 1)
        #expect(String(decoding: result.standardError, as: UTF8.self).contains("require a bundled engine"))
    }
}
