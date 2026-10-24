import Foundation
import Security
import Testing
@testable import Rapid

@Suite("Desktop runtime authentication")
struct DesktopRuntimeTrustTests {
    @Test("A swap after selection but before launch verification is refused", arguments: [
        "bin/rapid-mlx", "site-packages/module.py",
    ])
    func swapBeforeLaunchVerification(path: String) async throws {
        let fixture = try SignedRuntimeFixture()
        defer { fixture.remove() }
        try fixture.populateRuntime()
        let requirement = try await fixture.sign()
        let resolution = try #require(fixture.locate(requirement: requirement))
        // Atomic replacement models a changed inode, not only an in-place edit.
        try Data("#!/bin/sh\nexit 17\n".utf8).write(
            to: fixture.runtime.appendingPathComponent(path), options: .atomic
        )
        let process = Process()
        process.executableURL = resolution.binary
        #expect(throws: DesktopRuntimeTrust.LaunchRefused.self) {
            try DesktopRuntimeTrust.run(process) {
                DesktopRuntimeTrust.authenticate($0, requirement: requirement)
            }
        }
        #expect(process.processIdentifier == 0)
    }

    @Test("An intact signed runtime passes the actual launch boundary")
    func verifiedLaunch() async throws {
        let fixture = try SignedRuntimeFixture()
        defer { fixture.remove() }
        try fixture.populateRuntime()
        let requirement = try await fixture.sign()
        let process = Process()
        process.executableURL = fixture.binary
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try DesktopRuntimeTrust.run(process) {
            DesktopRuntimeTrust.authenticate($0, requirement: requirement)
        }
        defer { if process.isRunning { process.terminate() } }
        let deadline = ContinuousClock.now + .seconds(10)
        while process.isRunning && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(!process.isRunning)
        if !process.isRunning { #expect(process.terminationStatus == 0) }
    }

    @Test("Process-group launches reject an unsigned bundled runtime")
    func processGroupRefusal() throws {
        let fixture = try SignedRuntimeFixture()
        defer { fixture.remove() }
        try fixture.populateRuntime()
        #expect(throws: DesktopRuntimeTrust.LaunchRefused.self) {
            try ProcessGroupChild.spawn(
                executableURL: fixture.binary, arguments: [],
                standardInput: .nullDevice, standardOutput: Pipe(), standardError: Pipe()
            )
        }
    }

    @Test("A bundled launcher symlink keeps its app-owned launch path")
    func launcherSymlink() async throws {
        let fixture = try SignedRuntimeFixture()
        defer { fixture.remove() }
        try fixture.populateRuntime()
        let target = fixture.binary.deletingLastPathComponent().appendingPathComponent("launcher")
        try FileManager.default.moveItem(at: fixture.binary, to: target)
        try FileManager.default.createSymbolicLink(atPath: fixture.binary.path, withDestinationPath: "launcher")
        let requirement = try await fixture.sign()
        let resolution = try #require(fixture.locate(requirement: requirement))
        #expect(resolution.binary == fixture.binary)
        try FileManager.default.removeItem(at: fixture.binary)
        try FileManager.default.createSymbolicLink(atPath: fixture.binary.path, withDestinationPath: "/usr/bin/true")
        #expect(throws: DesktopRuntimeTrust.LaunchRefused.self) {
            try DesktopRuntimeTrust.verifyBeforeLaunch(resolution.binary, environment: [:]) {
                DesktopRuntimeTrust.authenticate($0, requirement: requirement)
            }
        }
    }

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
