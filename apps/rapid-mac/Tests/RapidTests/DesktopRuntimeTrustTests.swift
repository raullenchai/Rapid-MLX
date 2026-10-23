import Foundation
import Security
import Testing
@testable import Rapid

@Suite("Desktop runtime authentication")
struct DesktopRuntimeTrustTests {
    @Test("A sealed runtime runs after a full-app update of an existing slim layout")
    func slimMigration() async throws {
        let fixture = try SignedRuntimeFixture()
        defer { fixture.remove() }
        // Existing slim install: only the historical managed slot is present.
        let legacy = fixture.support.appendingPathComponent("runtime-override/rapid-mlx/bin/rapid-mlx")
        try fixture.write("#!/bin/sh\necho legacy\n", to: legacy, executable: true)
        let oldBytes = try Data(contentsOf: legacy)
        #expect(fixture.locate(requirement: nil) == nil)
        // The whole-app update brings its engine; it never modifies support/.
        try fixture.populateRuntime()
        let requirement = try await fixture.sign()
        let resolution = try #require(fixture.locate(requirement: requirement))
        #expect(resolution.source == .bundled)
        let result = try await TestSubprocess.run(executableURL: resolution.binary)
        #expect(result.terminationStatus == 0)
        #expect(String(decoding: result.standardOutput, as: UTF8.self) == "runnable\n")
        #expect(try Data(contentsOf: legacy) == oldBytes)
    }

    @Test("Changed launcher, modules, metadata and added modules are refused", arguments: [
        "bin/rapid-mlx", "site-packages/module.py", "VERSION", "site-packages/addition.py",
    ])
    func changedResource(path: String) async throws {
        let fixture = try SignedRuntimeFixture()
        defer { fixture.remove() }
        try fixture.populateRuntime()
        let requirement = try await fixture.sign()
        #expect(fixture.locate(requirement: requirement) != nil)
        try fixture.write("changed", to: fixture.runtime.appendingPathComponent(path))
        #expect(fixture.locate(requirement: requirement) == nil)
    }

    @Test("Unsigned and differently signed bundles are refused")
    func untrustedSignature() async throws {
        let fixture = try SignedRuntimeFixture()
        defer { fixture.remove() }
        try fixture.populateRuntime()
        #expect(fixture.locate(requirement: nil) == nil)
        let requirement = try await fixture.sign()
        #expect(fixture.locate(requirement: requirement) != nil)
        #expect(!DesktopRuntimeTrust.authenticate(fixture.binary), "An ad-hoc identity is not the release identity")
        try fixture.write("replacement", to: fixture.runtime.appendingPathComponent("VERSION"))
        _ = try await fixture.sign()
        #expect(fixture.locate(requirement: requirement) == nil)
    }

    @Test("A sealed symlink cannot authorize an external runtime dependency")
    func externalSymlink() async throws {
        let fixture = try SignedRuntimeFixture()
        defer { fixture.remove() }
        try fixture.populateRuntime()
        let external = fixture.root.appendingPathComponent("external.py")
        try fixture.write("external", to: external)
        try FileManager.default.createSymbolicLink(
            at: fixture.runtime.appendingPathComponent("site-packages/link.py"),
            withDestinationURL: external
        )
        let requirement = try await fixture.sign()
        #expect(fixture.locate(requirement: requirement) == nil)
    }

    @Test("An internal dependency symlink remains supported")
    func internalSymlink() async throws {
        let fixture = try SignedRuntimeFixture()
        defer { fixture.remove() }
        try fixture.populateRuntime()
        try FileManager.default.createSymbolicLink(
            atPath: fixture.runtime.appendingPathComponent("site-packages/link.py").path,
            withDestinationPath: "module.py"
        )
        let requirement = try await fixture.sign()
        #expect(fixture.locate(requirement: requirement) != nil)
    }

    @Test("Explicit developer selection can run an unsigned local engine")
    func developerOverride() async throws {
        let fixture = try SignedRuntimeFixture()
        defer { fixture.remove() }
        let developerBinary = fixture.root.appendingPathComponent("checkout/bin/rapid-mlx")
        try fixture.write("#!/bin/sh\necho runnable\n", to: developerBinary, executable: true)
        let resolution = try #require(ServerLocator.locate(
            environment: ["RAPID_BIN": developerBinary.path],
            bundleResourceURL: fixture.resources, applicationSupportURL: fixture.support
        ))
        #expect(resolution.source == .rapidBin)
        #expect(AboutPanel.engineIdentity(resolution: resolution)?.isOverride == true)
        let result = try await TestSubprocess.run(executableURL: resolution.binary)
        #expect(result.terminationStatus == 0)
        #expect(String(decoding: result.standardOutput, as: UTF8.self) == "runnable\n")
    }

    @Test("A cached bundled path is reverified before any engine probe")
    @MainActor
    func cachedResolutionIsNotAuthority() async throws {
        let fixture = try SignedRuntimeFixture()
        defer { fixture.remove() }
        try fixture.populateRuntime()
        let manager = ServerManager(testingState: .idle, binaryPath: fixture.binary, binarySource: .bundled)
        manager.runtimeCapabilitiesProvider = { _ in
            Issue.record("An unverified runtime must not be probed")
            return .conservative
        }
        await manager.start(alias: "test")
        #expect(manager.state == .missing)
        #expect(manager.binaryPath == nil)
        #expect(manager.binaryResolution == nil)
    }

    @Test("Release assembly refuses an engine-less app before building", arguments: [
        ["RAPID_MLX_OFFICIAL_RELEASE": "1"], ["CODESIGN_IDENTITY": "release-identity"],
    ])
    func releaseRequiresEngine(settings: [String: String]) async throws {
        let script = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("scripts/build.sh")
        var environment = ProcessInfo.processInfo.environment
        environment["SKIP_SIDECAR"] = "1"
        environment.merge(settings) { _, new in new }
        let result = try await TestSubprocess.run(
            executableURL: URL(fileURLWithPath: "/bin/bash"), arguments: [script.path],
            environment: environment
        )
        #expect(result.terminationStatus == 1)
        #expect(String(decoding: result.standardError, as: UTF8.self).contains("require a bundled engine"))
    }
}

private struct SignedRuntimeFixture {
    let root: URL
    var app: URL { root.appendingPathComponent("Rapid.app") }
    var resources: URL { app.appendingPathComponent("Contents/Resources") }
    var runtime: URL { resources.appendingPathComponent("rapid-mlx") }
    var binary: URL { runtime.appendingPathComponent("bin/rapid-mlx") }
    var support: URL { root.appendingPathComponent("support") }

    init() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("rapid-runtime-trust-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    func write(_ text: String, to url: URL, executable: Bool = false) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(text.utf8).write(to: url)
        if executable { try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path) }
    }

    func populateRuntime() throws {
        try write("#!/bin/sh\necho runnable\n", to: binary, executable: true)
        try write("1.0.0", to: runtime.appendingPathComponent("VERSION"))
        try write("value = 1\n", to: runtime.appendingPathComponent("site-packages/module.py"))
        let executable = app.appendingPathComponent("Contents/MacOS/Rapid")
        try FileManager.default.createDirectory(at: executable.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: URL(fileURLWithPath: "/usr/bin/true"), to: executable)
        let plist: [String: String] = [
            "CFBundleIdentifier": "com.rapid.runtime-trust-test", "CFBundleExecutable": "Rapid",
            "CFBundlePackageType": "APPL", "CFBundleVersion": "1",
        ]
        try PropertyListSerialization.data(fromPropertyList: plist, format: .xml, options: 0)
            .write(to: app.appendingPathComponent("Contents/Info.plist"))
    }

    func sign() async throws -> SecRequirement {
        let result = try await TestSubprocess.run(
            executableURL: URL(fileURLWithPath: "/usr/bin/codesign"),
            arguments: ["--force", "--sign", "-", app.path]
        )
        #expect(result.terminationStatus == 0, "\(String(decoding: result.standardError, as: UTF8.self))")
        var code: SecStaticCode?
        #expect(SecStaticCodeCreateWithPath(app as CFURL, [], &code) == errSecSuccess)
        var requirement: SecRequirement?
        #expect(SecCodeCopyDesignatedRequirement(try #require(code), [], &requirement) == errSecSuccess)
        return try #require(requirement)
    }

    func locate(requirement: SecRequirement?) -> ServerLocator.Resolution? {
        ServerLocator.locate(
            environment: [:], bundleResourceURL: resources, applicationSupportURL: support,
            authenticateBundle: { binary in
                if let requirement { return DesktopRuntimeTrust.authenticate(binary, requirement: requirement) }
                return DesktopRuntimeTrust.authenticate(binary)
            }
        )
    }

    func remove() { try? FileManager.default.removeItem(at: root) }
}
