import Foundation
import Testing
@testable import Rapid

@Suite("ServerLocator bundled engine trust boundary")
struct ServerLocatorVersionSelectionTests {
    @Test("A stale runtime override cannot shadow a newer bundled sidecar")
    func staleRuntimeUsesBundle() throws {
        let fixture = try Fixture(runtimeVersion: "0.10.8", bundledVersion: "0.12.4")
        defer { fixture.remove() }

        let resolved = ServerLocator.find(
            environment: [:],
            bundleResourceURL: fixture.bundleResources,
            applicationSupportURL: fixture.applicationSupport,
            authenticateBundle: { _ in true } // Selection only; real signatures covered by DesktopRuntimeTrustTests.
        )

        #expect(resolved == fixture.bundledBinary.resolvingSymlinksInPath())
    }

    @Test("A newer writable VERSION cannot replace the bundle")
    func newerRuntimeIgnored() throws {
        let fixture = try Fixture(runtimeVersion: "0.13.0", bundledVersion: "0.12.4")
        defer { fixture.remove() }

        let resolved = ServerLocator.find(
            environment: [:],
            bundleResourceURL: fixture.bundleResources,
            applicationSupportURL: fixture.applicationSupport,
            authenticateBundle: { _ in true } // Selection only; real signatures covered by DesktopRuntimeTrustTests.
        )

        #expect(resolved == fixture.bundledBinary.resolvingSymlinksInPath())
    }

    @Test("Equal managed versions cannot replace the bundle")
    func equalRuntimeIgnored() throws {
        let fixture = try Fixture(runtimeVersion: "0.12.4", bundledVersion: "0.12.4")
        defer { fixture.remove() }

        let resolved = ServerLocator.find(
            environment: [:],
            bundleResourceURL: fixture.bundleResources,
            applicationSupportURL: fixture.applicationSupport,
            authenticateBundle: { _ in true } // Selection only; real signatures covered by DesktopRuntimeTrustTests.
        )

        #expect(resolved == fixture.bundledBinary.resolvingSymlinksInPath())
    }

    @Test("An unversioned runtime override cannot shadow a versioned bundle")
    func unversionedRuntimeUsesBundle() throws {
        let fixture = try Fixture(runtimeVersion: nil, bundledVersion: "0.12.4")
        defer { fixture.remove() }

        let resolved = ServerLocator.find(
            environment: [:],
            bundleResourceURL: fixture.bundleResources,
            applicationSupportURL: fixture.applicationSupport,
            authenticateBundle: { _ in true } // Selection only; real signatures covered by DesktopRuntimeTrustTests.
        )

        #expect(resolved == fixture.bundledBinary.resolvingSymlinksInPath())
    }

    @Test("Missing bundle never falls back to writable runtime override")
    func missingBundleFailsClosed() throws {
        let fixture = try Fixture(
            runtimeVersion: "0.12.4",
            bundledVersion: nil,
            createBundledBinary: false
        )
        defer { fixture.remove() }

        let resolved = ServerLocator.find(
            environment: [:],
            bundleResourceURL: fixture.bundleResources,
            applicationSupportURL: fixture.applicationSupport,
            authenticateBundle: { _ in true } // Selection only; real signatures covered by DesktopRuntimeTrustTests.
        )

        #expect(resolved == nil)
    }

    @Test("Explicit RAPID_BIN remains higher priority than both managed slots")
    func rapidBinStillWins() throws {
        let fixture = try Fixture(runtimeVersion: "0.10.8", bundledVersion: "0.12.4")
        defer { fixture.remove() }
        let explicit = fixture.root.appendingPathComponent("explicit-rapid-mlx")
        try Fixture.writeExecutable(at: explicit)

        let resolved = ServerLocator.find(
            environment: ["RAPID_BIN": explicit.path],
            bundleResourceURL: fixture.bundleResources,
            applicationSupportURL: fixture.applicationSupport,
            authenticateBundle: { _ in true } // Selection only; real signatures covered by DesktopRuntimeTrustTests.
        )

        #expect(resolved == explicit.resolvingSymlinksInPath())
    }


}

private struct Fixture {
    let root: URL
    let applicationSupport: URL
    let bundleResources: URL
    let runtimeBinary: URL
    let bundledBinary: URL

    init(
        runtimeVersion: String?,
        bundledVersion: String?,
        createBundledBinary: Bool = true
    ) throws {
        let fm = FileManager.default
        root = fm.temporaryDirectory
            .appendingPathComponent("rapid-server-locator-\(UUID().uuidString)", isDirectory: true)
        applicationSupport = root.appendingPathComponent("support", isDirectory: true)
        bundleResources = root.appendingPathComponent("resources", isDirectory: true)
        runtimeBinary = applicationSupport
            .appendingPathComponent("runtime-override/rapid-mlx/bin/rapid-mlx")
        bundledBinary = bundleResources
            .appendingPathComponent("rapid-mlx/bin/rapid-mlx")

        try Self.writeExecutable(at: runtimeBinary)
        if createBundledBinary {
            try Self.writeExecutable(at: bundledBinary)
        }
        if let runtimeVersion {
            try Self.writeVersion(runtimeVersion, forBinary: runtimeBinary)
        }
        if let bundledVersion, createBundledBinary {
            try Self.writeVersion(bundledVersion, forBinary: bundledBinary)
        }
    }

    func remove() {
        try? FileManager.default.removeItem(at: root)
    }

    static func writeExecutable(at url: URL) throws {
        let fm = FileManager.default
        try fm.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try Data("#!/bin/sh\nexit 0\n".utf8).write(to: url, options: .atomic)
        try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
    }

    private static func writeVersion(_ version: String, forBinary binary: URL) throws {
        let root = binary.deletingLastPathComponent().deletingLastPathComponent()
        try Data("\(version)\n".utf8).write(
            to: root.appendingPathComponent("VERSION"),
            options: .atomic
        )
    }
}
