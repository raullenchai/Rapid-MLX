import Darwin
import Foundation
import Testing
@testable import Rapid

@Suite("Telemetry consent v2 shared state", .serialized, .pinnedTelemetryEnvironment)
struct TelemetryConsentV2Tests {
    private func directory(_ label: String) throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("rapid-consent-v2-\(label)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func defaults(_ label: String) -> UserDefaults {
        let name = "rapid-consent-v2-\(label)-\(UUID().uuidString)"
        let value = UserDefaults(suiteName: name)!
        value.removePersistentDomain(forName: name)
        return value
    }

    private func consentURL(_ directory: URL) -> URL {
        directory.appendingPathComponent("telemetry-consent.yaml")
    }

    private func writeJSON(_ object: [String: Any], to url: URL) throws {
        var data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        data.append(0x0A)
        try data.write(to: url)
    }

    private func json(at url: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: url)
        return try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    @Test("Marker merge preserves every existing and unknown field")
    func mergePreservesUnknownKeys() throws {
        let dir = try directory("merge")
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeJSON([
            "desktop_consent": false,
            "prompted_at": "2026-09-01T01:02:03Z",
            "schema_version": 7,
            "junk": "keep-me",
        ], to: consentURL(dir))

        #expect(TelemetryConsent.writeMergedConsent(
            updates: [:], raiseNoticeRevisionTo: 1, directory: dir, replaceUnreadable: false
        ))
        let stored = try json(at: consentURL(dir))
        #expect(stored["desktop_consent"] as? Bool == false)
        #expect(stored["prompted_at"] as? String == "2026-09-01T01:02:03Z")
        #expect(stored["schema_version"] as? Int == 7)
        #expect(stored["junk"] as? String == "keep-me")
        #expect(stored["notice_revision_seen"] as? Int == 1)
    }

    @Test("Busy permanent lock retries then falls back without corrupting state")
    func lockRetryFallsBack() throws {
        let dir = try directory("lock")
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeJSON(["consent": true, "junk": "safe"], to: consentURL(dir))
        let lock = dir.appendingPathComponent("telemetry-consent.yaml.lock")
        let fd = lock.path.withCString { open($0, O_CREAT | O_RDWR, mode_t(0o600)) }
        #expect(fd >= 0)
        #expect(flock(fd, LOCK_EX) == 0)
        defer {
            _ = flock(fd, LOCK_UN)
            close(fd)
        }

        let started = Date()
        #expect(TelemetryConsent.writeMergedConsent(
            updates: ["desktop_consent": true],
            raiseNoticeRevisionTo: nil,
            directory: dir,
            replaceUnreadable: true
        ))
        #expect(Date().timeIntervalSince(started) >= 1.4)
        let stored = try json(at: consentURL(dir))
        #expect(stored["consent"] as? Bool == true)
        #expect(stored["desktop_consent"] as? Bool == true)
        #expect(stored["junk"] as? String == "safe")
        #expect(FileManager.default.fileExists(atPath: lock.path))
    }

    @Test("Disclosure marker is raised and never lowered")
    func markerNeverLowers() throws {
        let dir = try directory("raise")
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeJSON(["notice_revision_seen": 9], to: consentURL(dir))
        #expect(TelemetryConsent.writeMergedConsent(
            updates: [:], raiseNoticeRevisionTo: 1, directory: dir, replaceUnreadable: false
        ))
        #expect(try json(at: consentURL(dir))["notice_revision_seen"] as? Int == 9)
    }

    @Test("Settings writes both consent scopes and never touches the notice marker")
    func settingsWritesBothScopes() throws {
        let dir = try directory("settings")
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeJSON(["notice_revision_seen": 9, "junk": "keep"], to: consentURL(dir))
        TelemetryConsent.record(
            enabled: false, version: "0.15.0rc1",
            defaults: defaults("settings"), telemetryDirectory: dir
        )
        let stored = try json(at: consentURL(dir))
        #expect(stored["consent"] as? Bool == false)
        #expect(stored["desktop_consent"] as? Bool == false)
        #expect(stored["prompted_version"] as? String == "0.15.0rc1")
        #expect(stored["notice_revision_seen"] as? Int == 9)
        #expect(stored["junk"] as? String == "keep")
    }

    @Test("Settings on stays locally dark when a presented notice could not persist its marker")
    func settingsOnCannotBypassMissingMarker() throws {
        let dir = try directory("settings-no-marker")
        defer { try? FileManager.default.removeItem(at: dir) }
        let userDefaults = defaults("settings-no-marker")
        TelemetryConsent.record(
            enabled: true, version: "0.15.0", defaults: userDefaults, telemetryDirectory: dir
        )
        #expect(!TelemetryConfig.isEnabled(defaults: userDefaults))
        let stored = try json(at: consentURL(dir))
        #expect(stored["consent"] as? Bool == true)
        #expect(stored["desktop_consent"] as? Bool == true)
        #expect(stored["notice_revision_seen"] == nil)
    }

    @Test(arguments: ["0.14.0", "0.14.9rc1"])
    func legacyRefusalMigratesAfterPresentation(_ recordedVersion: String) throws {
        let dir = try directory("legacy")
        defer { try? FileManager.default.removeItem(at: dir) }
        let userDefaults = defaults("legacy")
        try writeJSON([
            "consent": false,
            "desktop_consent": false,
            "prompted_version": recordedVersion,
            "prompted_at": "keep",
        ], to: consentURL(dir))

        #expect(TelemetryConsent.needsNotice(environment: [:], telemetryDirectory: dir))
        let result = TelemetryConsent.noticePresented(
            version: "0.15.0", defaults: userDefaults, environment: [:], telemetryDirectory: dir
        )
        #expect(result == .init(persisted: true, uploadAllowedThisRun: false))
        let stored = try json(at: consentURL(dir))
        #expect(stored["consent"] as? Bool == true)
        #expect(stored["desktop_consent"] as? Bool == true)
        #expect(stored["prompted_version"] as? String == "0.15.0")
        #expect(stored["prompted_at"] as? String == "keep")
        #expect(stored["notice_revision_seen"] as? Int == 1)
        #expect(!TelemetryConfig.isEnabled(defaults: userDefaults))
    }

    @Test(arguments: ["0.15.0", "0.15.0rc1", nil, "0.0.0"] as [String?])
    func currentOrUnknownRefusalIsByteIdentical(_ recordedVersion: String?) throws {
        let dir = try directory("current")
        defer { try? FileManager.default.removeItem(at: dir) }
        var record: [String: Any] = ["consent": false, "desktop_consent": false]
        if let recordedVersion { record["prompted_version"] = recordedVersion }
        try writeJSON(record, to: consentURL(dir))
        let before = try Data(contentsOf: consentURL(dir))

        #expect(!TelemetryConsent.needsNotice(environment: [:], telemetryDirectory: dir))
        let result = TelemetryConsent.noticePresented(
            version: "0.15.0", defaults: defaults("current"),
            environment: [:], telemetryDirectory: dir
        )
        #expect(!result.persisted)
        #expect(try Data(contentsOf: consentURL(dir)) == before)
    }

    @Test("A refusal with the marker and a true record with the marker need no action")
    func markerMakesExistingDecisionFinal() throws {
        for consent in [false, true] {
            let dir = try directory("marker-\(consent)")
            defer { try? FileManager.default.removeItem(at: dir) }
            try writeJSON([
                "consent": consent,
                "desktop_consent": consent,
                "prompted_version": "0.14.0",
                "notice_revision_seen": 1,
            ], to: consentURL(dir))
            let before = try Data(contentsOf: consentURL(dir))
            #expect(!TelemetryConsent.needsNotice(environment: [:], telemetryDirectory: dir))
            #expect(try Data(contentsOf: consentURL(dir)) == before)
        }
    }

    @Test("Absent consent gets only the disclosure marker")
    func absentConsentGetsMarkerOnly() throws {
        let dir = try directory("absent")
        defer { try? FileManager.default.removeItem(at: dir) }
        let userDefaults = defaults("absent")
        let result = TelemetryConsent.noticePresented(
            version: "0.15.0", defaults: userDefaults, environment: [:], telemetryDirectory: dir
        )
        #expect(result == .init(persisted: true, uploadAllowedThisRun: true))
        let stored = try json(at: consentURL(dir))
        #expect(stored["consent"] == nil)
        #expect(stored["notice_revision_seen"] as? Int == 1)
        #expect(stored["schema_version"] as? Int == 1)
        #expect(TelemetryConfig.isEnabled(defaults: userDefaults))
    }

    @Test("Kill switches suppress both notice and marker write")
    func killSwitchesDoNotWrite() throws {
        for environment in [
            ["DO_NOT_TRACK": "1"],
            ["RAPID_MLX_TELEMETRY": "0"],
            ["CI": "true"],
        ] {
            let dir = try directory("kill")
            defer { try? FileManager.default.removeItem(at: dir) }
            try writeJSON(["junk": "unchanged"], to: consentURL(dir))
            let before = try Data(contentsOf: consentURL(dir))
            #expect(!TelemetryConsent.needsNotice(
                environment: environment, telemetryDirectory: dir
            ))
            let result = TelemetryConsent.noticePresented(
                version: "0.15.0", defaults: defaults("kill"),
                environment: environment, telemetryDirectory: dir
            )
            #expect(!result.persisted)
            #expect(try Data(contentsOf: consentURL(dir)) == before)
        }
    }

    @Test("Presentation with an unwritable destination leaves existing bytes untouched")
    func failedPresentationDoesNotWrite() throws {
        let parent = try directory("failed-presentation")
        defer { try? FileManager.default.removeItem(at: parent) }
        let notADirectory = parent.appendingPathComponent("blocked")
        let original = Data("do-not-touch\n".utf8)
        try original.write(to: notADirectory)

        #expect(TelemetryConsent.needsNotice(environment: [:], telemetryDirectory: notADirectory))
        let result = TelemetryConsent.noticePresented(
            version: "0.15.0", defaults: defaults("failed-presentation"),
            environment: [:], telemetryDirectory: notADirectory
        )
        #expect(!result.persisted)
        #expect(try Data(contentsOf: notADirectory) == original)
    }

    @Test("Python safe-dump YAML shape is readable by Swift")
    func pythonYAMLIsReadable() throws {
        let dir = try directory("yaml")
        defer { try? FileManager.default.removeItem(at: dir) }
        let literal = """
        consent: true
        desktop_consent: true
        notice_revision_seen: 1
        prompted_at: '2026-09-21T12:00:00Z'
        prompted_version: 0.15.0
        schema_version: 1
        """
        try Data(literal.utf8).write(to: consentURL(dir))
        let shared = try #require(TelemetryConsent.readSharedConsent(at: consentURL(dir)))
        #expect(shared.engine == true)
        #expect(shared.desktop == true)
        #expect(shared.noticeRevisionSeen == 1)
        #expect(shared.promptedVersion == "0.15.0")
    }

    @Test("Consent file is 0600 and directory is 0700 after atomic replacement")
    func secureModes() throws {
        let dir = try directory("modes")
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: dir.path)
        TelemetryConsent.record(
            enabled: false, version: "0.15.0", defaults: defaults("modes"), telemetryDirectory: dir
        )
        let fileMode = try #require(
            FileManager.default.attributesOfItem(atPath: consentURL(dir).path)[.posixPermissions] as? NSNumber
        ).intValue & 0o777
        let directoryMode = try #require(
            FileManager.default.attributesOfItem(atPath: dir.path)[.posixPermissions] as? NSNumber
        ).intValue & 0o777
        #expect(fileMode == 0o600)
        #expect(directoryMode == 0o700)
    }

    @Test("Swift cutoff and disclosure constants match Python")
    func constantsDoNotDrift() throws {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let python = try String(
            contentsOf: repoRoot.appendingPathComponent("rapid_mlx/telemetry/consent_decision.py"),
            encoding: .utf8
        )
        #expect(python.contains("DEFAULT_ON_CUTOFF: Final = \"\(TelemetryConsent.defaultOnCutoff)\""))
        #expect(python.contains("DISCLOSURE_REVISION: Final = \(TelemetryConsent.disclosureRevision)"))
    }

    @Test("Simulated launch evidence shows merge and current-refusal no-op")
    func simulatedLaunchEvidence() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("rapid-consent-v2-evidence-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }

        let mergedDirectory = root.appendingPathComponent("merge")
        try FileManager.default.createDirectory(at: mergedDirectory, withIntermediateDirectories: true)
        try writeJSON([
            "consent": true,
            "desktop_consent": true,
            "prompted_at": "2026-09-21T12:00:00Z",
            "prompted_version": "0.14.9",
            "schema_version": 1,
        ], to: consentURL(mergedDirectory))
        let mergeBefore = try String(contentsOf: consentURL(mergedDirectory), encoding: .utf8)
        _ = TelemetryConsent.noticePresented(
            version: "0.15.0", defaults: defaults("evidence-merge"),
            environment: [:], telemetryDirectory: mergedDirectory
        )
        let mergeAfter = try String(contentsOf: consentURL(mergedDirectory), encoding: .utf8)
        let merged = try json(at: consentURL(mergedDirectory))
        #expect(merged["desktop_consent"] as? Bool == true)
        #expect(merged["prompted_at"] as? String == "2026-09-21T12:00:00Z")
        #expect(merged["notice_revision_seen"] as? Int == 1)

        let refusalDirectory = root.appendingPathComponent("refusal")
        try FileManager.default.createDirectory(at: refusalDirectory, withIntermediateDirectories: true)
        try writeJSON([
            "consent": false,
            "desktop_consent": false,
            "prompted_at": "2026-09-21T12:05:00Z",
            "prompted_version": "0.15.0",
            "schema_version": 1,
        ], to: consentURL(refusalDirectory))
        let refusalBefore = try Data(contentsOf: consentURL(refusalDirectory))
        _ = TelemetryConsent.noticePresented(
            version: "0.15.0", defaults: defaults("evidence-refusal"),
            environment: [:], telemetryDirectory: refusalDirectory
        )
        let refusalAfter = try Data(contentsOf: consentURL(refusalDirectory))
        #expect(refusalAfter == refusalBefore)

        print("T12 merge before:\n\(mergeBefore)")
        print("T12 merge after:\n\(mergeAfter)")
        print("T12 current refusal byte-identical: \(refusalAfter == refusalBefore)")
    }
}
