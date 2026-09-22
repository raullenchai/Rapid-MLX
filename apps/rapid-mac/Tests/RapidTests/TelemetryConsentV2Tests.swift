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

    private func write(_ text: String, to directory: URL) throws {
        try Data(text.utf8).write(to: consentURL(directory))
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

    @Test("A lock held by another process does not block the main actor")
    func busyLockDoesNotBlockMainActor() async throws {
        let dir = try directory("main-actor-lock")
        defer { try? FileManager.default.removeItem(at: dir) }
        let lockURL = dir.appendingPathComponent("telemetry-consent.yaml.lock")
        let locker = Process()
        let ready = Pipe()
        locker.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        locker.arguments = [
            "-c",
            "import fcntl,sys,time; f=open(sys.argv[1], 'a+'); fcntl.flock(f, fcntl.LOCK_EX); print('ready', flush=True); time.sleep(1.2); fcntl.flock(f, fcntl.LOCK_UN)",
            lockURL.path,
        ]
        locker.environment = [
            "HOME": NSHomeDirectory(),
            "PATH": "/usr/bin:/bin",
            "USER": "rc",
        ]
        locker.standardOutput = ready
        locker.standardError = Pipe()
        try locker.run()
        defer {
            if locker.isRunning { locker.terminate() }
        }
        let signal = ready.fileHandleForReading.readData(ofLength: 6)
        #expect(String(decoding: signal, as: UTF8.self) == "ready\n")

        let started = ContinuousClock.now
        let write = Task {
            let result = await TelemetryConsent.noticePresented(
                version: "0.15.0",
                defaults: self.defaults("main-actor-lock"),
                environment: [:],
                telemetryDirectory: dir
            )
            return (result, ContinuousClock.now)
        }

        var previousSample = await MainActor.run { ContinuousClock.now }
        var maximumGap = Duration.zero
        while locker.isRunning {
            try await Task.sleep(for: .milliseconds(10))
            let sample = await MainActor.run { ContinuousClock.now }
            maximumGap = max(maximumGap, previousSample.duration(to: sample))
            previousSample = sample
        }
        let (result, completed) = await write.value
        #expect(maximumGap < .milliseconds(100))
        #expect(started.duration(to: completed) >= .seconds(1))
        #expect(result == .init(persisted: true, uploadAllowedThisRun: true))
        #expect(FileManager.default.fileExists(atPath: consentURL(dir).path))
    }

    @Test("A directory chmod failure does not turn a successful consent write into failure")
    func directoryChmodIsBestEffort() throws {
        struct InjectedFailure: Error {}

        let dir = try directory("chmod-failure")
        defer { try? FileManager.default.removeItem(at: dir) }
        let persisted = TelemetryConsent.writeMergedConsent(
            updates: ["desktop_consent": true],
            raiseNoticeRevisionTo: 1,
            directory: dir,
            replaceUnreadable: true,
            setDirectoryPermissions: { _, _ in throw InjectedFailure() }
        )

        #expect(persisted)
        #expect(FileManager.default.fileExists(atPath: consentURL(dir).path))
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
    func settingsWritesBothScopes() async throws {
        let dir = try directory("settings")
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeJSON(["notice_revision_seen": 9, "junk": "keep"], to: consentURL(dir))
        await TelemetryConsent.record(
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

    @Test("A failed Settings opt-out reports failure and never claims the sidecar is off")
    func failedSettingsOptOutIsVisible() async throws {
        let dir = try directory("settings-write-failure")
        defer { try? FileManager.default.removeItem(at: dir) }
        // A directory at the consent path makes atomic replacement fail even
        // when the containing directory itself is writable.
        try FileManager.default.createDirectory(at: consentURL(dir), withIntermediateDirectories: false)
        let userDefaults = defaults("settings-write-failure")
        userDefaults.set(true, forKey: TelemetryConfig.enabledKey)

        let persisted = await TelemetryConsent.record(
            enabled: false, version: "0.15.0",
            defaults: userDefaults, telemetryDirectory: dir
        )

        #expect(!persisted)
        #expect(TelemetryConfig.isEnabled(defaults: userDefaults, environment: [:]))
        #expect(FileManager.default.fileExists(atPath: consentURL(dir).path))
    }

    @Test("Settings on stays locally dark when the shared consent write fails")
    func failedSettingsOptInRestoresPreviousState() async throws {
        let dir = try directory("settings-on-write-failure")
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(at: consentURL(dir), withIntermediateDirectories: false)
        let userDefaults = defaults("settings-on-write-failure")
        userDefaults.set(false, forKey: TelemetryConfig.enabledKey)

        let persisted = await TelemetryConsent.record(
            enabled: true, version: "0.15.0", defaults: userDefaults, telemetryDirectory: dir
        )

        #expect(!persisted)
        #expect(!TelemetryConfig.isEnabled(defaults: userDefaults))
        #expect(FileManager.default.fileExists(atPath: consentURL(dir).path))
    }

    @Test("Explicit Settings on below the cutoff enables Desktop and writes shared consent")
    func explicitSettingsOptInBelowCutoff() async throws {
        let dir = try directory("settings-on-pre-cutoff")
        defer { try? FileManager.default.removeItem(at: dir) }
        let userDefaults = defaults("settings-on-pre-cutoff")

        let persisted = await TelemetryConsent.record(
            enabled: true, version: "0.14.3", defaults: userDefaults, telemetryDirectory: dir
        )

        #expect(persisted)
        #expect(TelemetryConfig.isEnabled(defaults: userDefaults))
        let stored = try json(at: consentURL(dir))
        #expect(stored["consent"] as? Bool == true)
        #expect(stored["desktop_consent"] as? Bool == true)
        #expect(stored["prompted_version"] as? String == "0.14.3")
        #expect(stored["notice_revision_seen"] == nil)
    }

    @Test("Explicit Settings on above the cutoff needs no notice marker")
    func explicitSettingsOptInAboveCutoffWithoutMarker() async throws {
        let dir = try directory("settings-on-current-no-marker")
        defer { try? FileManager.default.removeItem(at: dir) }
        let userDefaults = defaults("settings-on-current-no-marker")

        let persisted = await TelemetryConsent.record(
            enabled: true, version: "0.15.0", defaults: userDefaults, telemetryDirectory: dir
        )

        #expect(persisted)
        #expect(TelemetryConfig.isEnabled(defaults: userDefaults))
        let stored = try json(at: consentURL(dir))
        #expect(stored["consent"] as? Bool == true)
        #expect(stored["desktop_consent"] as? Bool == true)
        #expect(stored["prompted_version"] as? String == "0.15.0")
        #expect(stored["notice_revision_seen"] == nil)
    }

    @Test(arguments: ["0.14.0", "0.14.9rc1"])
    func legacyRefusalMigratesAfterPresentation(_ recordedVersion: String) async throws {
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
        let result = await TelemetryConsent.noticePresented(
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
    func currentOrUnknownRefusalIsByteIdentical(_ recordedVersion: String?) async throws {
        let dir = try directory("current")
        defer { try? FileManager.default.removeItem(at: dir) }
        var record: [String: Any] = ["consent": false, "desktop_consent": false]
        if let recordedVersion { record["prompted_version"] = recordedVersion }
        try writeJSON(record, to: consentURL(dir))
        let before = try Data(contentsOf: consentURL(dir))

        #expect(!TelemetryConsent.needsNotice(environment: [:], telemetryDirectory: dir))
        let result = await TelemetryConsent.noticePresented(
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
    func absentConsentGetsMarkerOnly() async throws {
        let dir = try directory("absent")
        defer { try? FileManager.default.removeItem(at: dir) }
        let userDefaults = defaults("absent")
        let result = await TelemetryConsent.noticePresented(
            version: "0.15.0", defaults: userDefaults, environment: [:], telemetryDirectory: dir
        )
        #expect(result == .init(persisted: true, uploadAllowedThisRun: true))
        let stored = try json(at: consentURL(dir))
        #expect(stored["consent"] == nil)
        #expect(stored["notice_revision_seen"] as? Int == 1)
        #expect(stored["schema_version"] as? Int == 1)
        #expect(TelemetryConfig.isEnabled(defaults: userDefaults))
    }

    @Test(arguments: ["0.14.9", "0.15.0", "0.15.0rc1"])
    func runningVersionGatesTheV2Path(_ runningVersion: String) async throws {
        let dir = try directory("runtime-cutoff-\(runningVersion)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let userDefaults = defaults("runtime-cutoff-\(runningVersion)")
        userDefaults.set(true, forKey: TelemetryConfig.enabledKey)
        try write(
            "consent: false\nprompted_version: 0.14.3\nlegacy: keep\n",
            to: dir
        )
        let before = try Data(contentsOf: consentURL(dir))

        TelemetryConsent.synchronizeExistingDecision(
            version: runningVersion, defaults: userDefaults, telemetryDirectory: dir
        )
        let needsNotice = TelemetryConsent.needsNotice(
            version: runningVersion, environment: [:], telemetryDirectory: dir
        )
        let result = await TelemetryConsent.noticePresented(
            version: runningVersion, defaults: userDefaults,
            environment: [:], telemetryDirectory: dir
        )

        if runningVersion == "0.14.9" {
            #expect(TelemetryConfig.isEnabled(defaults: userDefaults))
            #expect(!needsNotice)
            #expect(result == .init(persisted: false, uploadAllowedThisRun: false))
            #expect(try Data(contentsOf: consentURL(dir)) == before)
        } else {
            #expect(!TelemetryConfig.isEnabled(defaults: userDefaults))
            #expect(needsNotice)
            #expect(result == .init(persisted: true, uploadAllowedThisRun: false))
            #expect(try json(at: consentURL(dir))["notice_revision_seen"] as? Int == 1)
        }
    }

    @Test("Settings keeps writing plain consent below the v2 cutoff")
    func settingsStillWritesBelowCutoff() async throws {
        let dir = try directory("pre-cutoff-settings")
        defer { try? FileManager.default.removeItem(at: dir) }
        await TelemetryConsent.record(
            enabled: false, version: "0.14.9",
            defaults: defaults("pre-cutoff-settings"), telemetryDirectory: dir
        )
        let stored = try json(at: consentURL(dir))
        #expect(stored["consent"] as? Bool == false)
        #expect(stored["desktop_consent"] as? Bool == false)
        #expect(stored["prompted_version"] as? String == "0.14.9")
        #expect(stored["notice_revision_seen"] == nil)
    }

    @Test("Kill switches suppress both notice and marker write")
    func killSwitchesDoNotWrite() async throws {
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
            let result = await TelemetryConsent.noticePresented(
                version: "0.15.0", defaults: defaults("kill"),
                environment: environment, telemetryDirectory: dir
            )
            #expect(!result.persisted)
            #expect(try Data(contentsOf: consentURL(dir)) == before)
        }
    }

    @Test("Presentation with an unwritable destination leaves existing bytes untouched")
    func failedPresentationDoesNotWrite() async throws {
        let parent = try directory("failed-presentation")
        defer { try? FileManager.default.removeItem(at: parent) }
        let notADirectory = parent.appendingPathComponent("blocked")
        let original = Data("do-not-touch\n".utf8)
        try original.write(to: notADirectory)

        #expect(TelemetryConsent.needsNotice(environment: [:], telemetryDirectory: notADirectory))
        let result = await TelemetryConsent.noticePresented(
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

    @Test(arguments: [
        ("yes", true), ("Yes", true), ("YES", true),
        ("true", true), ("True", true), ("TRUE", true),
        ("on", true), ("On", true), ("ON", true),
        ("no", false), ("No", false), ("NO", false),
        ("false", false), ("False", false), ("FALSE", false),
        ("off", false), ("Off", false), ("OFF", false),
    ])
    func yaml11BooleansMatchPython(_ scalar: String, _ expected: Bool) throws {
        let dir = try directory("yaml11-\(scalar)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try write("consent: \(scalar)\n", to: dir)
        let shared = try #require(TelemetryConsent.readSharedConsent(at: consentURL(dir)))
        #expect(shared.engine == expected)
    }

    @Test(arguments: ["nO", "TrUe", "YeS", "oN", "fAlSe"])
    func yaml11MixedCaseWordsAreStringsLikePyYAML(_ scalar: String) throws {
        let dir = try directory("yaml11-string-\(scalar)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try write("consent: \(scalar)\n", to: dir)
        let shared = try #require(TelemetryConsent.readSharedConsent(at: consentURL(dir)))
        #expect(shared.engine == nil)
    }

    @Test(arguments: ["1.0", "1.5", "true"])
    func noticeMarkerRequiresAnIntegerScalar(_ scalar: String) throws {
        let dir = try directory("integer-marker-\(scalar)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try write("notice_revision_seen: \(scalar)\n", to: dir)
        let shared = try #require(TelemetryConsent.readSharedConsent(at: consentURL(dir)))
        #expect(shared.noticeRevisionSeen == nil)
        #expect(TelemetryConsent.needsNotice(environment: [:], telemetryDirectory: dir))
    }

    @Test(arguments: ["", "# comments only\n", "consent: [unclosed\n"])
    func unreadableDocumentsNeverBecomeAbsent(_ contents: String) async throws {
        let dir = try directory("unreadable")
        defer { try? FileManager.default.removeItem(at: dir) }
        try write(contents, to: dir)
        let before = try Data(contentsOf: consentURL(dir))

        #expect(TelemetryConsent.readSharedConsent(at: consentURL(dir)) == nil)
        #expect(!TelemetryConsent.needsNotice(
            version: "0.15.0", environment: [:], telemetryDirectory: dir
        ))
        let result = await TelemetryConsent.noticePresented(
            version: "0.15.0", defaults: defaults("unreadable"),
            environment: [:], telemetryDirectory: dir
        )
        #expect(result == .init(persisted: false, uploadAllowedThisRun: false))
        #expect(try Data(contentsOf: consentURL(dir)) == before)
    }

    @Test("Unknown nested YAML survives a consent merge")
    func unknownNestedYAMLSurvivesMerge() throws {
        let dir = try directory("nested")
        defer { try? FileManager.default.removeItem(at: dir) }
        try write(
            """
            consent: true
            future:
              nested: keep
              flags:
                - one
                - two
            """ + "\n",
            to: dir
        )

        #expect(TelemetryConsent.writeMergedConsent(
            updates: [:], raiseNoticeRevisionTo: 1,
            directory: dir, replaceUnreadable: false
        ))
        let stored = try json(at: consentURL(dir))
        let future = try #require(stored["future"] as? [String: Any])
        #expect(future["nested"] as? String == "keep")
        #expect(future["flags"] as? [String] == ["one", "two"])
    }

    @Test("Consent file is 0600 and directory is 0700 after atomic replacement")
    func secureModes() async throws {
        let dir = try directory("modes")
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: dir.path)
        await TelemetryConsent.record(
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
    func simulatedLaunchEvidence() async throws {
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
        _ = await TelemetryConsent.noticePresented(
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
        _ = await TelemetryConsent.noticePresented(
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
