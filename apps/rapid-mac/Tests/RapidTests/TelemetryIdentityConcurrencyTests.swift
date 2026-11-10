import Foundation
import XCTest
@testable import Rapid

final class TelemetryIdentityConcurrencyTests: XCTestCase {
    private func identityURL() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-identity-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try FileManager.default.removeItem(at: directory) }
        return directory.appendingPathComponent("telemetry-client-id")
    }

    func testCorruptUTF8IsReplacedWithFreshIdentity() throws {
        let url = try identityURL()
        try Data([0xff, 0xfe]).write(to: url)
        let suite = "rapid-identity-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        defaults.set(true, forKey: TelemetryConfig.enabledKey)

        let id = TelemetryConfig.$environmentOverride.withValue([:]) {
            TelemetryIdentity.clientID(defaults: defaults, sharedIDURL: url)
        }

        XCTAssertNotNil(UUID(uuidString: id))
        XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), "\(id)\n")
        XCTAssertEqual(defaults.string(forKey: TelemetryConfig.clientIDKey), id)
        XCTAssertTrue(defaults.bool(forKey: TelemetryConfig.sharedClientIDMigrationKey))
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        XCTAssertEqual((attributes[.posixPermissions] as? NSNumber)?.intValue, 0o600)
    }

    func testUnsupportedExclusivePublishFallsBackToReplacement() throws {
        for code: POSIXErrorCode in [.ENOTSUP, .EOPNOTSUPP, .EPERM, .EXDEV] {
            // Cover absent, empty, and undecodable files on the fallback path.
            for existing in [nil, Data(), Data([0xff])] as [Data?] {
                let url = try identityURL()
                if let existing { try existing.write(to: url) }
                var attempts: [Bool] = []
                let id = TelemetryIdentity.persist("fresh-id", at: url) { data, destination, replace in
                    attempts.append(replace)
                    if !replace { throw POSIXError(code) }
                    try OwnerOnlyFile.writeAtomically(data, to: destination, replaceExisting: replace)
                }

                XCTAssertEqual(id, "fresh-id")
                XCTAssertEqual(attempts, [false, true])
                XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), "fresh-id\n")
                let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
                XCTAssertEqual((attributes[.posixPermissions] as? NSNumber)?.intValue, 0o600)
            }
        }
    }

    func testUnsupportedExclusivePublishAdoptsConcurrentWinner() throws {
        let url = try identityURL()
        var attempts: [Bool] = []
        let id = TelemetryIdentity.persist("fresh-id", at: url) { _, destination, replace in
            attempts.append(replace)
            try Data("winner-id\n".utf8).write(to: destination)
            throw POSIXError(.ENOTSUP)
        }

        XCTAssertEqual(id, "winner-id")
        XCTAssertEqual(attempts, [false])
        XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), "winner-id\n")
    }

    func testReadErrorDoesNotReplaceExistingFile() throws {
        let url = try identityURL()
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
        for code: POSIXErrorCode in [.EEXIST, .ENOTSUP] {
            var attempts: [Bool] = []
            let id = TelemetryIdentity.persist("fresh-id", at: url) { _, _, replace in
                attempts.append(replace)
                throw POSIXError(code)
            }
            XCTAssertNil(id)
            XCTAssertEqual(attempts, [false])
        }
    }

    private final class Results: @unchecked Sendable {
        let lock = NSLock()
        var ids: [String] = []

        func append(_ id: String) {
            lock.lock()
            defer { lock.unlock() }
            ids.append(id)
        }
    }

    func testConcurrentFirstRunAdoptsOneSharedID() throws {
        for _ in 0..<20 {
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("rapid-identity-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let url = directory.appendingPathComponent("telemetry-client-id")
            let results = Results()
            DispatchQueue.concurrentPerform(iterations: 16) { _ in
                let suite = "rapid-identity-\(UUID().uuidString)"
                let defaults = UserDefaults(suiteName: suite)!
                defer { defaults.removePersistentDomain(forName: suite) }
                defaults.set(true, forKey: TelemetryConfig.enabledKey)
                // CI sets the telemetry kill switch; task-locals do not cross
                // concurrentPerform threads, so pin an empty environment here.
                let id = TelemetryConfig.$environmentOverride.withValue([:]) {
                    TelemetryIdentity.clientID(defaults: defaults, sharedIDURL: url)
                }
                XCTAssertEqual(defaults.string(forKey: TelemetryConfig.clientIDKey), id)
                XCTAssertTrue(defaults.bool(forKey: TelemetryConfig.sharedClientIDMigrationKey))
                results.append(id)
            }
            let stored = try String(contentsOf: url, encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            XCTAssertEqual(results.ids.count, 16)
            XCTAssertEqual(Set(results.ids), [stored])
            let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
            XCTAssertEqual((attributes[.posixPermissions] as? NSNumber)?.intValue, 0o600)
            XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path), [url.lastPathComponent])
        }
    }
}
