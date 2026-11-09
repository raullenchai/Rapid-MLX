import Foundation
import XCTest
@testable import Rapid

final class TelemetryIdentityConcurrencyTests: XCTestCase {
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
