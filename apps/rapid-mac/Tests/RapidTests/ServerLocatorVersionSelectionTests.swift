import Foundation
import Testing
@testable import Rapid

@Suite("ServerLocator managed-sidecar version selection — #1503")
struct ServerLocatorVersionSelectionTests {
    @Test("A stale runtime override cannot shadow a newer bundled sidecar")
    func staleRuntimeUsesBundle() throws {
        let fixture = try Fixture(runtimeVersion: "0.10.8", bundledVersion: "0.12.4")
        defer { fixture.remove() }

        let resolved = ServerLocator.find(
            environment: [:],
            bundleResourceURL: fixture.bundleResources,
            applicationSupportURL: fixture.applicationSupport
        )

        #expect(resolved == fixture.bundledBinary.resolvingSymlinksInPath())
    }

    @Test("A newer runtime override still wins after an in-app engine update")
    func newerRuntimeWins() throws {
        let fixture = try Fixture(runtimeVersion: "0.13.0", bundledVersion: "0.12.4")
        defer { fixture.remove() }

        let resolved = ServerLocator.find(
            environment: [:],
            bundleResourceURL: fixture.bundleResources,
            applicationSupportURL: fixture.applicationSupport
        )

        #expect(resolved == fixture.runtimeBinary.resolvingSymlinksInPath())
    }

    @Test("Equal managed versions preserve runtime-override priority")
    func equalRuntimeWins() throws {
        let fixture = try Fixture(runtimeVersion: "0.12.4", bundledVersion: "0.12.4")
        defer { fixture.remove() }

        let resolved = ServerLocator.find(
            environment: [:],
            bundleResourceURL: fixture.bundleResources,
            applicationSupportURL: fixture.applicationSupport
        )

        #expect(resolved == fixture.runtimeBinary.resolvingSymlinksInPath())
    }

    @Test("An unversioned runtime override cannot shadow a versioned bundle")
    func unversionedRuntimeUsesBundle() throws {
        let fixture = try Fixture(runtimeVersion: nil, bundledVersion: "0.12.4")
        defer { fixture.remove() }

        let resolved = ServerLocator.find(
            environment: [:],
            bundleResourceURL: fixture.bundleResources,
            applicationSupportURL: fixture.applicationSupport
        )

        #expect(resolved == fixture.bundledBinary.resolvingSymlinksInPath())
    }

    @Test("Slim DMG still uses runtime override when no bundled binary exists")
    func slimDMGUsesRuntime() throws {
        let fixture = try Fixture(
            runtimeVersion: "0.12.4",
            bundledVersion: nil,
            createBundledBinary: false
        )
        defer { fixture.remove() }

        let resolved = ServerLocator.find(
            environment: [:],
            bundleResourceURL: fixture.bundleResources,
            applicationSupportURL: fixture.applicationSupport
        )

        #expect(resolved == fixture.runtimeBinary.resolvingSymlinksInPath())
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
            applicationSupportURL: fixture.applicationSupport
        )

        #expect(resolved == explicit.resolvingSymlinksInPath())
    }

    @Test("Managed comparison rejects malformed versions and accepts v-prefix")
    func comparisonGrammar() {
        #expect(ServerLocator.shouldPreferBundled(
            runtimeOverrideVersion: "not-a-version",
            bundledVersion: "0.12.4"
        ))
        #expect(!ServerLocator.shouldPreferBundled(
            runtimeOverrideVersion: "v0.12.5",
            bundledVersion: "0.12.4"
        ))
        #expect(!ServerLocator.shouldPreferBundled(
            runtimeOverrideVersion: "0.10.8",
            bundledVersion: "development"
        ))
        #expect(!ServerLocator.shouldPreferBundled(
            runtimeOverrideVersion: "0.12.4.0",
            bundledVersion: "0.12.4"
        ))
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
