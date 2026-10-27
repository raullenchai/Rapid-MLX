import Foundation
import Testing
@testable import Rapid

struct ChatOwnerRuntimeTests {
    @Test func developerLayoutsAndRejectedSources() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? fm.removeItem(at: root) }
        let bin = root.appendingPathComponent("bin")
        try fm.createDirectory(at: bin, withIntermediateDirectories: true)
        let binary = bin.appendingPathComponent("rapid-mlx")
        func executable(_ url: URL) throws {
            try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data().write(to: url)
            try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
        }
        try executable(binary)
        let resolution = ServerLocator.Resolution(binary: binary, source: .rapidBin, version: nil)
        let environment = ["PYTHONHOME": "inherited", "PYTHONPATH": "inherited"]
        #expect(throws: ChatOwnerConnectionError.invalidRuntime) {
            try ChatOwnerRuntime.forResolution(resolution, environment: environment)
        }
        let python3 = bin.appendingPathComponent("python3")
        try executable(python3)
        let venv = try ChatOwnerRuntime.forResolution(resolution, environment: environment)
        #expect(venv.interpreter == python3)
        #expect(venv.pythonHome == nil && venv.pythonPath == nil)
        #expect(venv.baseEnvironment["PYTHONHOME"] == nil && venv.baseEnvironment["PYTHONPATH"] == nil)
        try fm.removeItem(at: python3)
        let python = bin.appendingPathComponent("python")
        try executable(python)
        #expect(try ChatOwnerRuntime.forResolution(resolution).interpreter == python)

        let bundledPython = root.appendingPathComponent("python/bin/python3.12")
        let sitePackages = root.appendingPathComponent("site-packages")
        try executable(bundledPython)
        try fm.createDirectory(at: sitePackages, withIntermediateDirectories: true)
        let layout = try ChatOwnerRuntime.forResolution(resolution, environment: environment, authenticateBundle: { _ in
            Issue.record("Developer override must not require bundle authentication")
            return false
        })
        #expect(layout.interpreter == bundledPython)
        #expect(layout.pythonHome == root.appendingPathComponent("python"))
        #expect(layout.pythonPath?.path == sitePackages.path)
        for source in [ServerLocator.ResolvedSource.runtimeOverride, .unknown] {
            #expect(throws: ChatOwnerConnectionError.invalidRuntime) {
                try ChatOwnerRuntime.forResolution(.init(binary: binary, source: source, version: nil))
            }
        }
    }
}
