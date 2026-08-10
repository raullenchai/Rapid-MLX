import Foundation
import Testing

@testable import Rapid

@Suite("About panel engine identity — #1712")
struct AboutPanelEngineIdentityTests {
    @Test("A winning runtime override is visible with its own version and path")
    func runtimeOverrideIdentity() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }

        let resolution = ServerLocator.locate(
            environment: [:],
            bundleResourceURL: fixture.bundleResources,
            applicationSupportURL: fixture.applicationSupport
        )
        let identity = AboutPanel.engineIdentity(resolution: resolution)

        #expect(identity?.source == .runtimeOverride)
        #expect(identity?.version == "99.0.0")
        #expect(identity?.summary == "Engine 99.0.0 · App-managed override")
        #expect(identity?.isOverride == true)
        #expect(identity?.path == fixture.runtimeBinary.path)
    }

    @Test("A bundled engine is labelled without an override warning")
    func bundledIdentity() throws {
        let fixture = try Fixture(runtimeVersion: "0.11.0")
        defer { fixture.remove() }

        let resolution = ServerLocator.locate(
            environment: [:],
            bundleResourceURL: fixture.bundleResources,
            applicationSupportURL: fixture.applicationSupport
        )
        let identity = AboutPanel.engineIdentity(resolution: resolution)

        #expect(identity?.source == .bundled)
        #expect(identity?.version == "0.12.7")
        #expect(identity?.summary == "Engine 0.12.7 · Bundled with app")
        #expect(identity?.isOverride == false)
    }

    @Test("A symlinked override keeps the managed slot's identity and version")
    func symlinkedRuntimeIdentity() throws {
        let fixture = try Fixture(symlinkRuntime: true)
        defer { fixture.remove() }

        let resolution = ServerLocator.locate(
            environment: [:],
            bundleResourceURL: fixture.bundleResources,
            applicationSupportURL: fixture.applicationSupport
        )
        let identity = AboutPanel.engineIdentity(resolution: resolution)
        let resolved = fixture.runtimeBinary.resolvingSymlinksInPath()

        #expect(identity?.source == .runtimeOverride)
        #expect(identity?.version == "99.0.0")
        #expect(identity?.path == resolved.path)
    }

    @Test("No resolved binary produces no misleading engine identity")
    func missingIdentity() {
        #expect(AboutPanel.engineIdentity(resolution: nil) == nil)
    }

    @Test("Shared symlink targets preserve the slot that won version selection")
    func sharedManagedTarget() throws {
        let fixture = try Fixture(sharedManagedTarget: true)
        defer { fixture.remove() }

        let resolution = ServerLocator.locate(
            environment: [:],
            bundleResourceURL: fixture.bundleResources,
            applicationSupportURL: fixture.applicationSupport
        )
        let identity = AboutPanel.engineIdentity(resolution: resolution)

        #expect(identity?.source == .runtimeOverride)
        #expect(identity?.version == "99.0.0")
        #expect(identity?.summary == "Engine 99.0.0 · App-managed override")
        #expect(identity?.isOverride == true)
    }
}

private struct Fixture {
    let root: URL
    let applicationSupport: URL
    let bundleResources: URL
    let runtimeBinary: URL
    let bundledBinary: URL

    init(
        runtimeVersion: String = "99.0.0",
        symlinkRuntime: Bool = false,
        sharedManagedTarget: Bool = false
    ) throws {
        let fm = FileManager.default
        root = fm.temporaryDirectory
            .appendingPathComponent("rapid-about-engine-\(UUID().uuidString)", isDirectory: true)
        applicationSupport = root.appendingPathComponent("support", isDirectory: true)
        bundleResources = root.appendingPathComponent("resources", isDirectory: true)
        runtimeBinary = applicationSupport
            .appendingPathComponent("runtime-override/rapid-mlx/bin/rapid-mlx")
        bundledBinary = bundleResources
            .appendingPathComponent("rapid-mlx/bin/rapid-mlx")
        if sharedManagedTarget {
            let target = root.appendingPathComponent("checkout/bin/rapid-mlx")
            try Self.writeSidecar(binary: target, version: "0.1.0")
            for binary in [runtimeBinary, bundledBinary] {
                try fm.createDirectory(
                    at: binary.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                try fm.createSymbolicLink(at: binary, withDestinationURL: target)
            }
            try Self.writeVersion(runtimeVersion, forBinary: runtimeBinary)
            try Self.writeVersion("0.12.7", forBinary: bundledBinary)
        } else if symlinkRuntime {
            let target = root.appendingPathComponent("checkout/bin/rapid-mlx")
            try Self.writeSidecar(binary: target, version: "0.1.0")
            try fm.createDirectory(
                at: runtimeBinary.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try fm.createSymbolicLink(at: runtimeBinary, withDestinationURL: target)
            let runtimeRoot = runtimeBinary.deletingLastPathComponent().deletingLastPathComponent()
            try Data("\(runtimeVersion)\n".utf8).write(
                to: runtimeRoot.appendingPathComponent("VERSION"), options: .atomic)
        } else {
            try Self.writeSidecar(binary: runtimeBinary, version: runtimeVersion)
        }
        if !sharedManagedTarget {
            try Self.writeSidecar(binary: bundledBinary, version: "0.12.7")
        }
    }

    func remove() {
        try? FileManager.default.removeItem(at: root)
    }

    private static func writeSidecar(binary: URL, version: String) throws {
        let fm = FileManager.default
        try fm.createDirectory(
            at: binary.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try Data("#!/bin/sh\nexit 0\n".utf8).write(to: binary, options: .atomic)
        try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: binary.path)
        try writeVersion(version, forBinary: binary)
    }

    private static func writeVersion(_ version: String, forBinary binary: URL) throws {
        let root = binary.deletingLastPathComponent().deletingLastPathComponent()
        try Data("\(version)\n".utf8).write(
            to: root.appendingPathComponent("VERSION"), options: .atomic)
    }
}
