import Foundation
import Testing
@testable import Rapid

/// Pin the contract of the telemetry pipeline so a future refactor
/// can't silently break the wire-body shape the
/// ``telemetry.rapidmlx.com`` Worker expects, the opt-in default,
/// or the per-install identity persistence.
@MainActor
@Suite("Telemetry pipeline — schema, identity, opt-out")
final class TelemetryTests {

    // Every defaults-touching test mints its OWN private
    // ``UserDefaults(suiteName:)`` and passes it explicitly to the
    // product API, so the parallel test pool can never race the
    // shared ``.standard`` domain between a write and its read-back
    // (issue #530 — the flaky ``clientIDPersists`` /
    // ``enabled*`` residue). Suites are reclaimed on ``deinit`` via
    // the shared cleanup helper so we don't leak 42-byte plists into
    // ``~/Library/Preferences`` (issue #139).
    nonisolated(unsafe) private var createdSuiteNames: [String] = []
    deinit { TestDefaultsScope.cleanup(suiteNames: createdSuiteNames) }

    private func freshDefaults() -> UserDefaults {
        let name = TestDefaultsScope.mintSuiteName(prefix: "rapid-telemetry-test-")
        createdSuiteNames.append(name)
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        return defaults
    }

    private func temporaryTelemetryDirectory(_ label: String) throws -> URL {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent(
                "rapid-telemetry-\(label)-\(UUID().uuidString)",
                isDirectory: true
            )
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        return directory
    }

    // MARK: - Opt-out

    @Test("TelemetryConfig.isEnabled defaults to false until consent is recorded")
    func disabledUntilConsent() {
        let defaults = freshDefaults()
        #expect(TelemetryConfig.isEnabled(defaults: defaults) == false)
        #expect(TelemetryConsent.needsDecision(defaults: defaults))
    }

    @Test("TelemetryConfig.isEnabled honours an explicit false override")
    func optOutHonoured() {
        let defaults = freshDefaults()
        defaults.set(false, forKey: TelemetryConfig.enabledKey)
        #expect(TelemetryConfig.isEnabled(defaults: defaults) == false)
    }

    @Test("TelemetryConfig.isEnabled honours an explicit true override")
    func explicitOptIn() {
        let defaults = freshDefaults()
        defaults.set(true, forKey: TelemetryConfig.enabledKey)
        #expect(TelemetryConfig.isEnabled(defaults: defaults) == true)
    }

    // MARK: - Identity

    @Test("clientID is stable across calls within one launch")
    func clientIDStableWithinLaunch() {
        let defaults = freshDefaults()
        defaults.set(true, forKey: TelemetryConfig.enabledKey)
        let first = TelemetryIdentity.clientID(defaults: defaults)
        let second = TelemetryIdentity.clientID(defaults: defaults)
        #expect(first == second)
        #expect(!first.isEmpty)
    }

    @Test("clientID persists to UserDefaults so a later launch sees the same value")
    func clientIDPersists() {
        let defaults = freshDefaults()
        defaults.set(true, forKey: TelemetryConfig.enabledKey)
        let written = TelemetryIdentity.clientID(defaults: defaults)
        let readBack = defaults.string(forKey: TelemetryConfig.clientIDKey)
        #expect(readBack == written)
    }

    @Test("clientID fits inside the Worker's 64-char schema limit")
    func clientIDFitsSchemaCap() {
        let defaults = freshDefaults()
        defaults.set(true, forKey: TelemetryConfig.enabledKey)
        let id = TelemetryIdentity.clientID(defaults: defaults)
        #expect(id.count <= 64)
    }

    @Test("engine shared client ID wins and is mirrored to UserDefaults")
    func sharedClientIDWins() throws {
        let defaults = freshDefaults()
        defaults.set(true, forKey: TelemetryConfig.enabledKey)
        defaults.set("desktop-legacy-id", forKey: TelemetryConfig.clientIDKey)
        let directory = try temporaryTelemetryDirectory("shared-wins")
        defer { try? FileManager.default.removeItem(at: directory) }
        let sharedURL = directory.appendingPathComponent("telemetry-client-id")
        try Data("engine-shared-id\n".utf8).write(to: sharedURL)

        let resolved = TelemetryIdentity.clientID(
            defaults: defaults,
            sharedIDURL: sharedURL
        )

        #expect(resolved == "engine-shared-id")
        #expect(defaults.string(forKey: TelemetryConfig.clientIDKey) == resolved)
        #expect(defaults.bool(forKey: TelemetryConfig.sharedClientIDMigrationKey))
    }

    @Test("legacy desktop client ID migrates to the engine file once")
    func legacyClientIDMigrates() throws {
        let defaults = freshDefaults()
        defaults.set(true, forKey: TelemetryConfig.enabledKey)
        defaults.set("desktop-legacy-id", forKey: TelemetryConfig.clientIDKey)
        let directory = try temporaryTelemetryDirectory("legacy-migration")
        defer { try? FileManager.default.removeItem(at: directory) }
        let sharedURL = directory.appendingPathComponent("telemetry-client-id")

        let resolved = TelemetryIdentity.clientID(
            defaults: defaults,
            sharedIDURL: sharedURL
        )

        #expect(resolved == "desktop-legacy-id")
        #expect(try String(contentsOf: sharedURL, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines) == resolved)
        #expect(defaults.bool(forKey: TelemetryConfig.sharedClientIDMigrationKey))
    }

    @Test("deleting a migrated shared ID rotates instead of resurrecting the old ID")
    func sharedClientIDResetRotates() throws {
        let defaults = freshDefaults()
        defaults.set(true, forKey: TelemetryConfig.enabledKey)
        defaults.set("old-desktop-id", forKey: TelemetryConfig.clientIDKey)
        defaults.set(true, forKey: TelemetryConfig.sharedClientIDMigrationKey)
        let directory = try temporaryTelemetryDirectory("identity-reset")
        defer { try? FileManager.default.removeItem(at: directory) }
        let sharedURL = directory.appendingPathComponent("telemetry-client-id")

        let resolved = TelemetryIdentity.clientID(
            defaults: defaults,
            sharedIDURL: sharedURL
        )

        #expect(resolved != "old-desktop-id")
        #expect(UUID(uuidString: resolved) != nil)
        #expect(defaults.string(forKey: TelemetryConfig.clientIDKey) == resolved)
    }

    // MARK: - Shared consent

    @Test("accepting records desktop opt-in and engine-compatible shared state")
    func consentAcceptsAndSharesState() throws {
        let defaults = freshDefaults()
        let directory = try temporaryTelemetryDirectory("consent-yes")
        defer { try? FileManager.default.removeItem(at: directory) }

        TelemetryConsent.record(
            enabled: true,
            version: "0.10.8",
            defaults: defaults,
            telemetryDirectory: directory
        )

        #expect(TelemetryConfig.isEnabled(defaults: defaults))
        #expect(!TelemetryConsent.needsDecision(defaults: defaults))
        #expect(FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("telemetry-client-id").path
        ))
        let data = try Data(contentsOf: directory.appendingPathComponent("telemetry-consent.yaml"))
        let json = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        #expect(json["consent"] as? Bool == true)
        #expect(json["desktop_consent"] as? Bool == true)
        #expect(json["prompted_version"] as? String == "0.10.8")
        #expect(json["schema_version"] as? Int == 1)
    }

    @Test("declining records consent without creating a client ID")
    func consentDeclinesWithoutIdentity() throws {
        let defaults = freshDefaults()
        let directory = try temporaryTelemetryDirectory("consent-no")
        defer { try? FileManager.default.removeItem(at: directory) }

        TelemetryConsent.record(
            enabled: false,
            version: "0.10.8",
            defaults: defaults,
            telemetryDirectory: directory
        )

        #expect(!TelemetryConfig.isEnabled(defaults: defaults))
        #expect(!FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("telemetry-client-id").path
        ))
        let data = try Data(contentsOf: directory.appendingPathComponent("telemetry-consent.yaml"))
        let json = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        #expect(json["consent"] as? Bool == false)
    }

    @Test("shared consent deletion clears desktop decision for re-prompt")
    func sharedConsentResetReprompts() throws {
        let defaults = freshDefaults()
        defaults.set(true, forKey: TelemetryConfig.enabledKey)
        defaults.set(true, forKey: TelemetryConfig.sharedConsentMigrationKey)
        let directory = try temporaryTelemetryDirectory("consent-reset")
        defer { try? FileManager.default.removeItem(at: directory) }

        TelemetryConsent.synchronizeExistingDecision(
            version: "0.10.8",
            defaults: defaults,
            telemetryDirectory: directory
        )

        #expect(TelemetryConsent.needsDecision(defaults: defaults))
        #expect(!TelemetryConfig.isEnabled(defaults: defaults))
    }

    @Test("engine-only consent does not silently authorize desktop telemetry")
    func engineOnlyConsentDoesNotAuthorizeDesktop() throws {
        let defaults = freshDefaults()
        let directory = try temporaryTelemetryDirectory("engine-consent")
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data("consent: true\nprompted_version: 0.11.0\nschema_version: 1\n".utf8)
            .write(to: directory.appendingPathComponent("telemetry-consent.yaml"))
        try Data("engine-existing-id\n".utf8)
            .write(to: directory.appendingPathComponent("telemetry-client-id"))

        TelemetryConsent.synchronizeExistingDecision(
            version: "0.10.8",
            defaults: defaults,
            telemetryDirectory: directory
        )

        #expect(!TelemetryConfig.isEnabled(defaults: defaults))
        #expect(TelemetryConsent.needsDecision(defaults: defaults))
        #expect(defaults.string(forKey: TelemetryConfig.clientIDKey) == nil)
        #expect(!defaults.bool(forKey: TelemetryConfig.sharedConsentMigrationKey))
    }

    @Test("shared consent is the source of truth after migration")
    func migratedSharedConsentWins() throws {
        let defaults = freshDefaults()
        defaults.set(true, forKey: TelemetryConfig.enabledKey)
        defaults.set(true, forKey: TelemetryConfig.sharedConsentMigrationKey)
        let directory = try temporaryTelemetryDirectory("shared-consent-wins")
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data("consent: false\ndesktop_consent: false\nprompted_version: 0.11.0\nschema_version: 1\n".utf8)
            .write(to: directory.appendingPathComponent("telemetry-consent.yaml"))

        TelemetryConsent.synchronizeExistingDecision(
            version: "0.10.8",
            defaults: defaults,
            telemetryDirectory: directory
        )

        #expect(!TelemetryConfig.isEnabled(defaults: defaults))
        #expect(!TelemetryConsent.needsDecision(defaults: defaults))
    }

    @Test("malformed shared consent line (comment-only value) is tolerated, not a launch crash-loop")
    func malformedSharedConsentDoesNotTrap() throws {
        let defaults = freshDefaults()
        let directory = try temporaryTelemetryDirectory("malformed-consent")
        defer { try? FileManager.default.removeItem(at: directory) }
        // A shared file whose `consent` value is only an inline comment.
        // `pieces[1].split(separator: "#", maxSplits: 1)[0]` used to
        // fatal-trap here — `String.split` omits empty subsequences, so
        // `"#".split(separator: "#")` is `[]` and `[0]` is out of range.
        // Because this parse runs at launch via
        // ``synchronizeExistingDecision``, the trap crash-loops the app
        // on a corrupt shared file (which is written by BOTH the desktop
        // and the rapid-mlx engine). The contract is "treat a malformed
        // shared file as absent and re-prompt" — reaching the asserts at
        // all is the core regression pin.
        try Data("consent:#\n".utf8)
            .write(to: directory.appendingPathComponent("telemetry-consent.yaml"))

        TelemetryConsent.synchronizeExistingDecision(
            version: "0.10.8",
            defaults: defaults,
            telemetryDirectory: directory
        )

        #expect(TelemetryConsent.needsDecision(defaults: defaults))
        #expect(!TelemetryConfig.isEnabled(defaults: defaults))
    }

    // MARK: - Event schema

    @Test("session_start event carries every required top-level field the Worker validates")
    func sessionStartHasRequiredFields() throws {
        let event = TelemetryEvent.sessionStart(
            version: "0.5.12",
            platform: TelemetryEvent.Platform(
                app: "rapid-desktop",
                os: "macos",
                os_version: "26.0.0",
                arch: "arm64"
            )
        )
        let data = try JSONEncoder().encode(event)
        let json = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        // Mirror the Worker's validateEvent() field list verbatim —
        // a missing one here = a 400 from the collector.
        #expect(json["schema_version"] as? Int == 1)
        #expect((json["client_id"] as? String)?.isEmpty == false)
        #expect((json["session_id"] as? String)?.isEmpty == false)
        #expect(json["rapid_mlx_version"] as? String == "0.5.12")
        #expect(json["event"] as? String == "session_start")
        #expect((json["timestamp"] as? String)?.isEmpty == false)
        let platform = try #require(json["platform"] as? [String: Any])
        #expect(platform["app"] as? String == "rapid-desktop")
        #expect(platform["os"] as? String == "macos")
        #expect(platform["arch"] as? String == "arm64")
    }

    @Test("error event carries the same required fields plus the error_* extras the dashboard reads")
    func errorEventCarriesExtras() throws {
        let event = TelemetryEvent.error(
            version: "0.5.12",
            platform: TelemetryEvent.Platform(
                app: "rapid-desktop", os: "macos",
                os_version: "26.0.0", arch: "arm64"
            ),
            errorType: "uncaught_exception",
            errorMessage: "NSInternalInconsistencyException: bad selector",
            stackFrames: ["0 Rapid 0x0001 main + 42"],
            context: "chat_send"
        )
        let data = try JSONEncoder().encode(event)
        let json = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        #expect(json["event"] as? String == "error")
        #expect(json["error_type"] as? String == "uncaught_exception")
        #expect((json["error_message"] as? String)?.contains("bad selector") == true)
        let frames = try #require(json["stack_frames"] as? [String])
        #expect(frames.count == 1)
        #expect(json["context"] as? String == "chat_send")
    }

    // MARK: - Platform builder

    @Test("currentPlatform tags the app as rapid-desktop so analytics can split clients")
    func platformTagsApp() {
        let p = TelemetryClient.currentPlatform()
        #expect(p.app == "rapid-desktop")
        #expect(p.os == "macos")
        #expect(!p.os_version.isEmpty)
        #expect(p.arch == "arm64" || p.arch == "x86_64" || p.arch == "unknown")
    }

    // MARK: - Machine identity (chip + bucketed memory)

    @Test("currentPlatform now reports a chip brand so desktop machines appear in the per-chip breakdown")
    func platformCarriesChip() {
        let p = TelemetryClient.currentPlatform()
        // Every CI + dev host this runs on is a real Mac, so the sysctl
        // key resolves — the whole point of this change is that the
        // field is populated (was nil/absent before) so desktop
        // machines stop being invisible next to CLI ones.
        let chip = try! #require(p.chip)
        #expect(!chip.isEmpty)
        // Apple Silicon brand strings start with "Apple" (e.g.
        // "Apple M4 Max"); on the arm64 CI fleet this pins that the
        // value is the real brand, not the generic arch fallback.
        #if arch(arm64)
        #expect(chip.hasPrefix("Apple"))
        #endif
    }

    @Test("currentPlatform chip matches the raw sysctl brand string the engine also reads")
    func platformChipMatchesSysctl() {
        #if arch(x86_64)
        // Do not transmit Intel's detailed SKU/frequency-bearing brand string.
        #expect(TelemetryClient.chipBrand() == "Intel")
        #else
        // Read the same key the engine shells out to
        // (`sysctl -n machdep.cpu.brand_string`) and confirm the Swift
        // reader returns the byte-identical, whitespace-trimmed value —
        // so desktop + CLI bucket into the same analytics label.
        var size = 0
        _ = sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        var buffer = [CChar](repeating: 0, count: size)
        _ = sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0)
        let bytes = buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        let raw = String(bytes: bytes, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        #expect(TelemetryClient.chipBrand() == raw)
        #endif
    }

    @Test("currentPlatform reports bucketed memory matching this host's rounded physical RAM")
    func platformCarriesBucketedMemory() {
        let p = TelemetryClient.currentPlatform()
        let mem = try! #require(p.memory_gb)
        // Bucket is the rounded physical RAM — strictly positive on any
        // real machine, and it must equal the bucket of the raw byte
        // count (no raw bytes ever leave the process).
        #expect(mem > 0)
        let expected = TelemetryClient.bucketMemoryGB(
            ProcessInfo.processInfo.physicalMemory
        )
        #expect(mem == expected)
    }

    @Test("bucketMemoryGB rounds to coarse GB tiers exactly like the engine's bucket_memory_gb")
    func memoryBucketingMatchesEngineTiers() {
        let giB: UInt64 = 1024 * 1024 * 1024
        // Whole-GiB Mac configs map to their integer GB (the common case).
        #expect(TelemetryClient.bucketMemoryGB(8 * giB) == 8)
        #expect(TelemetryClient.bucketMemoryGB(16 * giB) == 16)
        #expect(TelemetryClient.bucketMemoryGB(24 * giB) == 24)
        #expect(TelemetryClient.bucketMemoryGB(64 * giB) == 64)
        #expect(TelemetryClient.bucketMemoryGB(128 * giB) == 128)
        // Non-positive clamps to 0 (mirrors the engine's `<= 0` guard).
        #expect(TelemetryClient.bucketMemoryGB(0) == 0)
        // Sub-GB rounds to nearest (0.4 GiB → 0, 0.6 GiB → 1).
        #expect(TelemetryClient.bucketMemoryGB(UInt64(0.4 * Double(giB))) == 0)
        #expect(TelemetryClient.bucketMemoryGB(UInt64(0.6 * Double(giB))) == 1)
        // Round-half-to-even: 1.5 → 2, 2.5 → 2 — mirrors Python round().
        #expect(TelemetryClient.bucketMemoryGB(UInt64(1.5 * Double(giB))) == 2)
        #expect(TelemetryClient.bucketMemoryGB(UInt64(2.5 * Double(giB))) == 2)
    }

    @Test("session_start encodes chip + memory_gb when present (parity with the engine platform shape)")
    func platformEncodesMachineFields() throws {
        let event = TelemetryEvent.sessionStart(
            version: "0.5.12",
            platform: TelemetryEvent.Platform(
                app: "rapid-desktop",
                os: "macos",
                os_version: "26.0.0",
                arch: "arm64",
                chip: "Apple M4 Max",
                memory_gb: 48
            )
        )
        let data = try JSONEncoder().encode(event)
        let json = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        let platform = try #require(json["platform"] as? [String: Any])
        #expect(platform["app"] as? String == "rapid-desktop")
        #expect(platform["chip"] as? String == "Apple M4 Max")
        #expect(platform["memory_gb"] as? Int == 48)
    }

    @Test("chip + memory_gb are omitted on the wire when nil so the addition is backward-compatible")
    func platformOmitsMachineFieldsWhenNil() throws {
        // A build that couldn't read the brand (chip == nil) must not
        // emit a null/placeholder key — the field is simply absent, so
        // an older worker sees the exact legacy 4-field platform shape.
        let event = TelemetryEvent.sessionStart(
            version: "0.5.12",
            platform: TelemetryEvent.Platform(
                app: "rapid-desktop",
                os: "macos",
                os_version: "26.0.0",
                arch: "arm64"
            )
        )
        let data = try JSONEncoder().encode(event)
        let json = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        let platform = try #require(json["platform"] as? [String: Any])
        #expect(platform["chip"] == nil)
        #expect(platform["memory_gb"] == nil)
        // The discriminator the CLI-vs-App split must key off stays set.
        #expect(platform["app"] as? String == "rapid-desktop")
    }

    // MARK: - Crash marker directory

    @Test("crash marker directory is created on demand under Application Support")
    func markerDirectoryExists() {
        let dir = CrashReporter.markerDirectory
        #expect(FileManager.default.fileExists(atPath: dir.path))
        #expect(dir.path.contains("Application Support/Rapid/crash-markers"))
    }

    // MARK: - sendBatch acceptance contract

    @Test("sendBatch returns true when telemetry is opted out so cleanup runs")
    func sendBatchOptedOutReturnsTrue() async {
        let defaults = freshDefaults()
        defaults.set(false, forKey: TelemetryConfig.enabledKey)
        let dummy = TelemetryEvent.sessionStart(
            version: "0.0.0",
            platform: TelemetryClient.currentPlatform()
        )
        var client = TelemetryClient()
        client.defaults = defaults
        let accepted = await client.sendBatch([dummy])
        // Opt-out short-circuits BEFORE any network call yet still
        // signals "you may delete the local copies" — otherwise an
        // opted-out user accumulates markers across launches forever.
        #expect(accepted == true)
    }

    @Test("sendBatch returns true for an empty input so cleanup runs even on the no-op path")
    func sendBatchEmptyReturnsTrue() async {
        let defaults = freshDefaults()
        defaults.set(true, forKey: TelemetryConfig.enabledKey)
        var client = TelemetryClient()
        client.defaults = defaults
        let accepted = await client.sendBatch([])
        #expect(accepted == true)
    }

    @Test("sendBatch returns false when the request transport errors so markers stay on disk")
    func sendBatchTransportErrorReturnsFalse() async {
        let defaults = freshDefaults()
        defaults.set(true, forKey: TelemetryConfig.enabledKey)
        // Point the client at a session that always fails (zero
        // timeout). We can't override TelemetryConfig.endpoint without
        // a hook, but we CAN swap in a URLSession whose request
        // resolution will always fail fast.
        let cfg = URLSessionConfiguration.ephemeral
        cfg.timeoutIntervalForRequest = 0.001
        cfg.timeoutIntervalForResource = 0.001
        let failing = URLSession(configuration: cfg)
        var client = TelemetryClient()
        client.session = failing
        client.defaults = defaults
        let dummy = TelemetryEvent.sessionStart(
            version: "0.0.0",
            platform: TelemetryClient.currentPlatform()
        )
        let accepted = await client.sendBatch([dummy])
        #expect(accepted == false)
    }

    // MARK: - Signal marker shape

    @Test("Pre-cached signal marker JSON decodes into the same CrashMarker shape flush expects")
    func signalMarkerDecodesAsCrashMarker() throws {
        // Mirror the exact string shape ``prebuildSignalMarkers``
        // bakes at install time, then prove it survives the
        // round-trip ``flushPendingCrashReports`` performs. Catches
        // any future refactor that diverges the signal-context
        // envelope from CrashMarker's required keys (which is what
        // codex flagged in round 1 — the previous envelope used
        // ``"signal"`` instead of ``"session_id"`` and got dropped).
        let session = "11111111-2222-3333-4444-555555555555"
        let version = "0.5.13"
        let json = "{\"session_id\":\"\(session)\",\"version\":\"\(version)\",\"error_type\":\"signal\",\"error_message\":\"crashed with SIGABRT (6)\"}"
        let data = Data(json.utf8)
        let marker = try JSONDecoder().decode(CrashMarker.self, from: data)
        #expect(marker.session_id == session)
        #expect(marker.version == version)
        #expect(marker.error_type == "signal")
        #expect(marker.error_message.contains("SIGABRT"))
    }

    // MARK: - Crash attribution

    @Test("error event honours an explicit sessionID override so flushed crashes attribute to the crashed launch")
    func errorEventSessionIDOverride() throws {
        let crashedSession = "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE"
        let event = TelemetryEvent.error(
            version: "0.5.12",
            platform: TelemetryClient.currentPlatform(),
            errorType: "uncaught_exception",
            errorMessage: "boom",
            stackFrames: [],
            context: nil,
            sessionID: crashedSession
        )
        let data = try JSONEncoder().encode(event)
        let json = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        #expect(json["session_id"] as? String == crashedSession)
        // And the version comes from the crashed-launch marker,
        // not the current bundle.
        #expect(json["rapid_mlx_version"] as? String == "0.5.12")
    }

    @Test("error event falls back to the current session when no override is supplied")
    func errorEventSessionIDFallback() throws {
        let event = TelemetryEvent.error(
            version: "0.5.13",
            platform: TelemetryClient.currentPlatform(),
            errorType: "uncaught_exception",
            errorMessage: "boom",
            stackFrames: [],
            context: nil
        )
        let data = try JSONEncoder().encode(event)
        let json = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        #expect(json["session_id"] as? String == TelemetryConfig.sessionID)
    }
}

private struct TelemetryAuditDefaultsSandbox {
    private let suiteName: String
    private let suite: UserDefaults
    private let previousEnabled: Any?
    private let previousClientID: Any?

    init() {
        let name = "rapid-telemetry-audit-batch-8-\(UUID().uuidString)"
        suiteName = name
        suite = UserDefaults(suiteName: name)!
        suite.removePersistentDomain(forName: name)
        previousEnabled = UserDefaults.standard.object(forKey: TelemetryConfig.enabledKey)
        previousClientID = UserDefaults.standard.object(forKey: TelemetryConfig.clientIDKey)
        UserDefaults.standard.removeObject(forKey: TelemetryConfig.enabledKey)
        UserDefaults.standard.removeObject(forKey: TelemetryConfig.clientIDKey)
        UserDefaults.standard.addSuite(named: name)
    }

    func setTelemetryEnabled(_ enabled: Bool) {
        suite.set(enabled, forKey: TelemetryConfig.enabledKey)
    }

    func suiteString(forKey key: String) -> String? {
        suite.string(forKey: key)
    }

    func tearDown() {
        UserDefaults.standard.removeSuite(named: suiteName)
        suite.removePersistentDomain(forName: suiteName)
        UserDefaults.standard.removeObject(forKey: TelemetryConfig.enabledKey)
        UserDefaults.standard.removeObject(forKey: TelemetryConfig.clientIDKey)
        // Issue #139: ``removePersistentDomain`` + ``removeSuite``
        // leave a 42-byte empty plist sitting in
        // ``~/Library/Preferences/`` because ``cfprefsd`` doesn't
        // unlink it. Use the shared cleanup helper to actually
        // remove the file and force the daemon to flush first so
        // it doesn't race our unlink.
        TestDefaultsScope.cleanup(suiteNames: [suiteName])
        if let previousEnabled {
            UserDefaults.standard.set(previousEnabled, forKey: TelemetryConfig.enabledKey)
        }
        if let previousClientID {
            UserDefaults.standard.set(previousClientID, forKey: TelemetryConfig.clientIDKey)
        }
    }
}

final class TelemetryAuditURLProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) private static var statusCode: Int = 200
    nonisolated(unsafe) private static var transportError: Error?
    nonisolated(unsafe) static var requestCount = 0

    static func stub(statusCode: Int) {
        Self.statusCode = statusCode
        transportError = nil
        requestCount = 0
    }

    static func stub(error: Error) {
        transportError = error
        requestCount = 0
    }

    static func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [TelemetryAuditURLProtocol.self] + (config.protocolClasses ?? [])
        return URLSession(configuration: config)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.requestCount += 1
        if let transportError = Self.transportError {
            client?.urlProtocol(self, didFailWithError: transportError)
            return
        }
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: Self.statusCode,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "application/json"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data())
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

@MainActor
@Suite("Telemetry audit batch 8 contracts", .serialized)
struct TelemetryAuditBatch8Contracts {
    private func platform() -> TelemetryEvent.Platform {
        TelemetryEvent.Platform(
            app: "rapid-desktop",
            os: "macos",
            os_version: "26.0.0",
            arch: "arm64"
        )
    }

    private func event() -> TelemetryEvent {
        TelemetryEvent.sessionStart(version: "0.0.0", platform: platform())
    }

    @Test("redact scrubs home directories, caps long strings, and leaves safe strings alone")
    func redactContract() {
        #expect(
            TelemetryEvent.redact("/Users/raullen/work/foo", cap: 128)
                == "/Users/<redacted>/work/foo"
        )
        #expect(
            TelemetryEvent.redact("/home/raullen/bar", cap: 128)
                == "/home/<redacted>/bar"
        )
        #expect(TelemetryEvent.redact("abcdefghijk", cap: 5) == "abcde…")

        let plain = "plain diagnostic message with no filesystem path"
        #expect(TelemetryEvent.redact(plain, cap: 128) == plain)
    }

    @Test("error factory redacts message, frames, and context")
    func errorFactoryAppliesRedaction() throws {
        let defaults = TelemetryAuditDefaultsSandbox()
        defer { defaults.tearDown() }
        defaults.setTelemetryEnabled(true)

        let stackFrames = (0..<35).map {
            "frame \($0) /Users/dev\($0)/project\($0)/File.swift"
        }
        let event = TelemetryEvent.error(
            version: "0.0.0",
            platform: platform(),
            errorType: "uncaught_exception",
            errorMessage: "failed while reading /Users/raullen/work/secret.txt",
            stackFrames: stackFrames,
            context: "model alias from /home/raullen/private-model"
        )

        #expect(event.error_message == "failed while reading /Users/<redacted>/work/secret.txt")
        let frames = try #require(event.stack_frames)
        #expect(frames.count == 30)
        #expect(frames.allSatisfy { $0.contains("/Users/<redacted>/") })
        #expect(frames.allSatisfy { !$0.contains("/Users/dev") })
        #expect(frames.last?.contains("frame 29") == true)
        #expect(frames.contains { $0.contains("frame 30") } == false)
        #expect(event.context == "model alias from /home/<redacted>/private-model")
    }

    @Test("clientID returns opted-out placeholder without persisting")
    func clientIDOptedOutDoesNotPersist() {
        let defaults = TelemetryAuditDefaultsSandbox()
        defer { defaults.tearDown() }
        defaults.setTelemetryEnabled(false)

        #expect(TelemetryConfig.isEnabled == false)
        #expect(TelemetryIdentity.clientID() == "00000000-0000-0000-0000-000000000000")
        #expect(UserDefaults.standard.string(forKey: TelemetryConfig.clientIDKey) == nil)
        #expect(defaults.suiteString(forKey: TelemetryConfig.clientIDKey) == nil)
    }

    @Test("clientID persists one real UUID when opted in")
    func clientIDOptedInPersistsAndReusesUUID() {
        let defaults = TelemetryAuditDefaultsSandbox()
        defer { defaults.tearDown() }
        defaults.setTelemetryEnabled(true)

        #expect(TelemetryConfig.isEnabled == true)
        let first = TelemetryIdentity.clientID()
        let second = TelemetryIdentity.clientID()
        #expect(UUID(uuidString: first) != nil)
        #expect(first != "00000000-0000-0000-0000-000000000000")
        #expect(second == first)
        #expect(UserDefaults.standard.string(forKey: TelemetryConfig.clientIDKey) == first)
    }

    @Test("sendBatch returns true for permanent 4xx rejection")
    func sendBatchReturnsTrueFor4xx() async {
        let defaults = TelemetryAuditDefaultsSandbox()
        defer { defaults.tearDown() }
        defaults.setTelemetryEnabled(true)
        TelemetryAuditURLProtocol.stub(statusCode: 422)
        var client = TelemetryClient()
        client.session = TelemetryAuditURLProtocol.session()

        let accepted = await client.sendBatch([event()])

        #expect(accepted == true)
        #expect(TelemetryAuditURLProtocol.requestCount == 1)
    }

    @Test("sendBatch returns false for transient 5xx response")
    func sendBatchReturnsFalseFor5xx() async {
        let defaults = TelemetryAuditDefaultsSandbox()
        defer { defaults.tearDown() }
        defaults.setTelemetryEnabled(true)
        TelemetryAuditURLProtocol.stub(statusCode: 503)
        var client = TelemetryClient()
        client.session = TelemetryAuditURLProtocol.session()

        let accepted = await client.sendBatch([event()])

        #expect(accepted == false)
        #expect(TelemetryAuditURLProtocol.requestCount == 1)
    }

    @Test("sendBatch returns false for transport error")
    func sendBatchReturnsFalseForTransportError() async {
        let defaults = TelemetryAuditDefaultsSandbox()
        defer { defaults.tearDown() }
        defaults.setTelemetryEnabled(true)
        TelemetryAuditURLProtocol.stub(error: URLError(.notConnectedToInternet))
        var client = TelemetryClient()
        client.session = TelemetryAuditURLProtocol.session()

        let accepted = await client.sendBatch([event()])

        #expect(accepted == false)
        #expect(TelemetryAuditURLProtocol.requestCount == 1)
    }

    @Test("sendBatch returns true when disabled even with non-empty events")
    func sendBatchDisabledReturnsTrueForNonEmptyEvents() async {
        let defaults = TelemetryAuditDefaultsSandbox()
        defer { defaults.tearDown() }
        defaults.setTelemetryEnabled(false)
        TelemetryAuditURLProtocol.stub(error: URLError(.cannotConnectToHost))
        var client = TelemetryClient()
        client.session = TelemetryAuditURLProtocol.session()

        let accepted = await client.sendBatch([event()])

        #expect(accepted == true)
        #expect(TelemetryAuditURLProtocol.requestCount == 0)
    }

    // MARK: - README L102-104 unpinned audit-batch-8 surfaces

    /// Pin ``NoRedirectDelegate``'s contract: any 3xx from the
    /// telemetry endpoint must terminate the request locally instead
    /// of letting ``URLSession`` replay the body to the new host.
    /// Audit batch 8 T1 specifically calls out the "307/308 to an
    /// arbitrary host" scenario; if a refactor replaces the ``nil``
    /// argument with ``request`` (e.g. for "follow same-origin
    /// redirects" UX), the README "redirect-free" claim regresses
    /// silently.
    @Test("NoRedirectDelegate denies every HTTP redirect by completing with nil")
    func noRedirectDelegateDeniesRedirects() async {
        let delegate = NoRedirectDelegate()
        // Synthesise a redirect-shaped response. The delegate's only
        // job is the completionHandler argument; we don't actually
        // run a URLSession round-trip here.
        let response = HTTPURLResponse(
            url: URL(string: "https://attacker.example/exfil")!,
            statusCode: 307,
            httpVersion: "HTTP/1.1",
            headerFields: nil
        )!
        let newRequest = URLRequest(url: URL(string: "https://attacker.example/exfil")!)
        var observed: URLRequest? = newRequest

        await withCheckedContinuation { continuation in
            // The delegate is not actor-isolated; create a dummy
            // URLSessionTask placeholder via a no-op session. We do
            // not start a real task; the delegate only reads the
            // arguments.
            let dummySession = URLSession(configuration: .ephemeral)
            let task = dummySession.dataTask(with: URLRequest(url: URL(string: "https://telemetry.rapidmlx.com")!))
            delegate.urlSession(
                dummySession,
                task: task,
                willPerformHTTPRedirection: response,
                newRequest: newRequest,
                completionHandler: { result in
                    observed = result
                    continuation.resume()
                }
            )
            task.cancel()
        }

        #expect(observed == nil, "Redirect was honoured — body would have replayed to attacker.example")
    }

    /// Pin the **wiring** between ``noRedirectSession`` and
    /// ``NoRedirectDelegate``. Without this assertion a refactor
    /// that drops ``delegate: NoRedirectDelegate()`` from the
    /// session initialiser leaves the per-method delegate test
    /// (above) green while the live session reverts to
    /// ``URLSession``'s default redirect-follow behaviour — the
    /// README "redirect-free" claim regresses end-to-end. Reading
    /// ``URLSession.delegate`` back asserts the live session is the
    /// thing the audit-batch-8 T1 fix actually installed.
    @Test("noRedirectSession is wired to NoRedirectDelegate, not the default redirect-follower")
    func noRedirectSessionDelegateIsInstalled() {
        let delegate = TelemetryClient.noRedirectSession.delegate
        #expect(delegate is NoRedirectDelegate, "noRedirectSession lost its NoRedirectDelegate wiring — URLSession will silently follow redirects")
    }

    /// Pin that a freshly-constructed ``TelemetryClient`` defaults to
    /// ``noRedirectSession``. Codex r2 NIT: a future refactor could
    /// leave ``noRedirectSession`` hardened but stop using it as the
    /// client's default session (e.g. switching the property default
    /// to ``URLSession.shared``). The READ would still show
    /// ``noRedirectSession`` is well-formed; live traffic would
    /// silently route through the shared session and re-enable
    /// redirect-follow + persistent cookies. The identity check is
    /// the cheapest possible end-to-end binding.
    @Test("TelemetryClient().session defaults to noRedirectSession (not URLSession.shared)")
    func telemetryClientDefaultsToHardenedSession() {
        let client = TelemetryClient()
        #expect(client.session === TelemetryClient.noRedirectSession)
    }

    /// Pin the ``noRedirectSession`` configuration attributes that
    /// satisfy the README "no-cookie" + "ephemeral" claims. Each
    /// attribute closes a distinct passive-tracking surface:
    ///   * ``httpCookieAcceptPolicy = .never`` — refuses Set-Cookie
    ///   * ``httpShouldSetCookies = false`` — refuses to send a
    ///     cookie even if storage already contains one
    ///   * ephemeral configuration — disk cache stays at zero so a
    ///     second process can't read prior telemetry from /var/cache
    /// A regression toggling any single attribute would leave the
    /// other two assertions green and the README claim partially
    /// untrue.
    @Test("noRedirectSession is configured cookie-less and ephemeral")
    func noRedirectSessionConfigIsHardened() {
        let cfg = TelemetryClient.noRedirectSession.configuration
        #expect(cfg.httpCookieAcceptPolicy == .never)
        #expect(cfg.httpShouldSetCookies == false)
        // .ephemeral has zero disk cache by definition; pin that
        // explicitly so a regression to .default (which gets a
        // ~250 MB on-disk cache by default) goes red.
        #expect(cfg.urlCache?.diskCapacity ?? 0 == 0)
        // Pin the User-Agent so a refactor that swaps in
        // "URLSession/N.M" (the default) loses our analytics
        // signature for distinguishing rapid-desktop traffic from
        // the worker's other clients.
        let ua = cfg.httpAdditionalHeaders?["User-Agent"] as? String
        #expect(ua == "rapid-desktop-telemetry/1")
    }

    /// Pin ``maxBodyBytes`` and its < ``noRedirectSession`` 256 KB
    /// Worker-ceiling relationship. Audit batch 8 T2 leaves a 56 KB
    /// headroom for upstream framing; a refactor that hoists the
    /// constant to a "clean 256 * 1024" value would silently start
    /// pushing payloads that exceed the Worker's hard cap.
    @Test("maxBodyBytes is below the documented 256 KB Worker ceiling")
    func maxBodyBytesLeavesHeadroomBelowWorkerCap() {
        let workerHardCap = 256 * 1024
        #expect(TelemetryClient.maxBodyBytes == 200 * 1024)
        #expect(TelemetryClient.maxBodyBytes < workerHardCap)
        // Ensure the headroom is meaningful (>= 32 KB) so a future
        // bump to the Worker's framing overhead doesn't quietly
        // sneak the in-band ceiling above the hard cap.
        #expect(workerHardCap - TelemetryClient.maxBodyBytes >= 32 * 1024)
    }

    /// Pin the oversized-batch drop on ``sendBatch``. Audit batch 8
    /// T2: a batch that JSON-encodes beyond ``maxBodyBytes`` would
    /// 413 forever; the contract is to short-circuit (no network
    /// call) AND return ``true`` so the caller's cleanup retires
    /// the markers instead of retrying every launch.
    @Test("sendBatch on an oversized payload returns true without firing the request")
    func sendBatchOversizedReturnsTrueWithoutNetwork() async {
        let defaults = TelemetryAuditDefaultsSandbox()
        defer { defaults.tearDown() }
        defaults.setTelemetryEnabled(true)
        TelemetryAuditURLProtocol.stub(statusCode: 200)
        var client = TelemetryClient()
        client.session = TelemetryAuditURLProtocol.session()

        // Build one event with a paddable free-text field. The cap
        // is 512 chars for error_message + 256 per frame; build the
        // batch by repeating a deterministic event many times so the
        // overall JSON crosses maxBodyBytes (200 KB). Each event is
        // ~1 KB on the wire, so ~250 events takes us across.
        let padded = String(repeating: "x", count: 480)
        let oversized: [TelemetryEvent] = (0..<260).map { _ in
            TelemetryEvent.error(
                version: "0.0.0",
                platform: platform(),
                errorType: "signal",
                errorMessage: padded,
                stackFrames: Array(repeating: padded, count: 1),
                context: nil
            )
        }
        // Sanity: confirm the encoded batch genuinely exceeds the cap.
        // If this assertion fails, the test is no longer exercising
        // the oversized branch and needs a bigger payload.
        let envelope: [String: [TelemetryEvent]] = ["batch": oversized]
        let encoded = try? JSONEncoder().encode(envelope)
        #expect((encoded?.count ?? 0) > TelemetryClient.maxBodyBytes)

        let accepted = await client.sendBatch(oversized)

        #expect(accepted == true, "Oversized batch must retire markers, not block their cleanup")
        #expect(
            TelemetryAuditURLProtocol.requestCount == 0,
            "Oversized batch was dispatched to the network — should have short-circuited"
        )
    }
}
