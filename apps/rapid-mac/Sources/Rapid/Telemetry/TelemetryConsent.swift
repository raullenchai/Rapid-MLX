import Darwin
import Foundation

/// Bridges Desktop telemetry state to rapid-mlx's shared consent record.
/// JSON is deliberately written into the engine's `.yaml` file: JSON is valid
/// YAML and lets Desktop preserve fields it does not own without shipping a
/// second YAML encoder.
enum TelemetryConsent {
    /// Keep these in lockstep with `rapid_mlx.telemetry.consent_decision`.
    static let disclosureRevision = 1
    static let defaultOnCutoff = "0.15.0"

    private static let lockRetrySeconds: TimeInterval = 1.5
    private static let lockRetryIntervalMicroseconds: useconds_t = 50_000

    struct SharedConsent {
        let engine: Bool?
        let desktop: Bool?
        let noticeRevisionSeen: Int?
        let promptedVersion: String?

        var currentNoticeWasSeen: Bool {
            (noticeRevisionSeen ?? Int.min) >= disclosureRevision
        }
    }

    struct NoticePresentationResult: Equatable {
        let persisted: Bool
        let uploadAllowedThisRun: Bool
    }

    private enum NoticeWrite {
        case none
        case markerOnly
        case migrateLegacyRefusal
    }

    /// Reconcile Desktop's local sender gate with the shared v2 record. A
    /// missing/currently-unseen disclosure stays off until the banner really
    /// appears and its marker write succeeds.
    static func synchronizeExistingDecision(
        version: String = TelemetryClient.currentVersion()
    ) {
        synchronizeExistingDecision(
            version: version,
            defaults: .standard,
            telemetryDirectory: TelemetryIdentity.sharedTelemetryDirectory()
        )
    }

    static func synchronizeExistingDecision(
        version _: String,
        defaults: UserDefaults,
        telemetryDirectory: URL
    ) {
        let url = consentURL(in: telemetryDirectory)
        let alreadyMigrated = defaults.bool(
            forKey: TelemetryConfig.sharedConsentMigrationKey
        )

        guard let mapping = readConsentMapping(at: url) else {
            // Unreadable is absent for decision purposes, but is never
            // overwritten by the automatic notice writer.
            defaults.removeObject(forKey: TelemetryConfig.enabledKey)
            return
        }

        if mapping.isEmpty {
            // Honour `telemetry reset`: an old local answer must not
            // resurrect a deleted shared record. A pre-v2 local preference
            // is likewise held off until the new disclosure is visible.
            if alreadyMigrated {
                defaults.removeObject(forKey: TelemetryConfig.enabledKey)
            } else if defaults.object(forKey: TelemetryConfig.enabledKey) != nil {
                defaults.set(false, forKey: TelemetryConfig.enabledKey)
            }
            return
        }

        let shared = sharedConsent(from: mapping)
        let enabled: Bool
        if shared.engine == false || shared.desktop == false {
            enabled = false
        } else if shared.currentNoticeWasSeen {
            // With no explicit refusal, the current disclosure marker is the
            // default-on authorisation (decision-table rows 3 and 9).
            enabled = true
        } else {
            enabled = false
        }
        defaults.set(enabled, forKey: TelemetryConfig.enabledKey)
        defaults.set(true, forKey: TelemetryConfig.sharedConsentMigrationKey)
        if enabled {
            synchronizeClientID(defaults: defaults, telemetryDirectory: telemetryDirectory)
        }
    }

    static func needsNotice(
        environment: [String: String] = TelemetryConfig.environment,
        telemetryDirectory: URL = TelemetryIdentity.sharedTelemetryDirectory()
    ) -> Bool {
        guard !TelemetryConfig.killSwitchActive(environment: environment),
              let mapping = readConsentMapping(at: consentURL(in: telemetryDirectory))
        else { return false }
        return noticeWrite(for: sharedConsent(from: mapping)) != .none
    }

    /// Called by the banner's `onAppear`, never when launch merely schedules
    /// the banner. Failed persistence leaves the local sender off and causes a
    /// later launch to try the disclosure again.
    static func noticePresented(
        version: String = TelemetryClient.currentVersion(),
        defaults: UserDefaults = .standard,
        environment: [String: String] = TelemetryConfig.environment,
        telemetryDirectory: URL = TelemetryIdentity.sharedTelemetryDirectory()
    ) -> NoticePresentationResult {
        guard !TelemetryConfig.killSwitchActive(environment: environment),
              let initial = readConsentMapping(at: consentURL(in: telemetryDirectory))
        else {
            return NoticePresentationResult(persisted: false, uploadAllowedThisRun: false)
        }

        let action = noticeWrite(for: sharedConsent(from: initial))
        guard action != .none else {
            return NoticePresentationResult(persisted: false, uploadAllowedThisRun: false)
        }

        var committedAction = NoticeWrite.none
        let persisted = writeMergedConsent(
            updates: [:],
            raiseNoticeRevisionTo: nil,
            directory: telemetryDirectory,
            replaceUnreadable: false,
            customMerge: { current in
                let currentAction = noticeWrite(for: sharedConsent(from: current))
                guard currentAction != .none else { return nil }
                var merged = current
                if currentAction == .migrateLegacyRefusal {
                    merged["consent"] = true
                    merged["desktop_consent"] = true
                    merged["prompted_version"] = version
                }
                if merged["schema_version"] == nil { merged["schema_version"] = 1 }
                let seen = intValue(merged["notice_revision_seen"]) ?? 0
                merged["notice_revision_seen"] = max(seen, disclosureRevision)
                committedAction = currentAction
                return merged
            }
        )
        guard persisted else {
            return NoticePresentationResult(persisted: false, uploadAllowedThisRun: false)
        }

        defaults.set(true, forKey: TelemetryConfig.sharedConsentMigrationKey)
        if committedAction == .migrateLegacyRefusal {
            // Decision-table row 4: migration authorises the next run, never
            // the run that reversed the pre-default-on refusal.
            defaults.set(false, forKey: TelemetryConfig.enabledKey)
            return NoticePresentationResult(persisted: true, uploadAllowedThisRun: false)
        }

        let finalShared = readSharedConsent(at: consentURL(in: telemetryDirectory))
        let enabled = finalShared.map {
            $0.desktop != false && $0.engine != false
        } ?? false
        defaults.set(enabled, forKey: TelemetryConfig.enabledKey)
        if enabled {
            synchronizeClientID(defaults: defaults, telemetryDirectory: telemetryDirectory)
        }
        return NoticePresentationResult(persisted: true, uploadAllowedThisRun: enabled)
    }

    static func record(
        enabled: Bool,
        version: String = TelemetryClient.currentVersion()
    ) {
        // Reports captured while off must never become eligible merely because
        // Settings was switched on later.
        if enabled && !TelemetryConfig.isEnabled {
            CrashReporter.discardPendingCrashReports()
        }
        record(
            enabled: enabled,
            version: version,
            defaults: .standard,
            telemetryDirectory: TelemetryIdentity.sharedTelemetryDirectory()
        )
    }

    static func record(
        enabled: Bool,
        version: String,
        defaults: UserDefaults,
        telemetryDirectory: URL
    ) {
        // Settings never writes the disclosure marker. An explicit "on"
        // therefore becomes effective only when the launch notice has already
        // established the current marker; a failed notice write stays dark.
        let markerSeen = readSharedConsent(at: consentURL(in: telemetryDirectory))?
            .currentNoticeWasSeen == true
        let effectiveEnabled = enabled && markerSeen
        defaults.set(effectiveEnabled, forKey: TelemetryConfig.enabledKey)
        if effectiveEnabled {
            synchronizeClientID(defaults: defaults, telemetryDirectory: telemetryDirectory)
        }
        let persisted = writeMergedConsent(
            updates: [
                "consent": enabled,
                // This separate scope prevents a CLI-only choice from
                // authorising Desktop's broader redacted crash diagnostics.
                "desktop_consent": enabled,
                "prompted_at": ISO8601DateFormatter().string(from: Date()),
                "prompted_version": version,
            ],
            raiseNoticeRevisionTo: nil,
            directory: telemetryDirectory,
            replaceUnreadable: true
        )
        defaults.set(persisted, forKey: TelemetryConfig.sharedConsentMigrationKey)
    }

    private static func synchronizeClientID(
        defaults: UserDefaults,
        telemetryDirectory: URL
    ) {
        _ = TelemetryIdentity.clientID(
            defaults: defaults,
            sharedIDURL: telemetryDirectory.appendingPathComponent(
                "telemetry-client-id", isDirectory: false
            )
        )
    }

    private static func noticeWrite(for shared: SharedConsent) -> NoticeWrite {
        if shared.currentNoticeWasSeen { return .none }
        if shared.engine == false {
            return isLegacyVersion(shared.promptedVersion) ? .migrateLegacyRefusal : .none
        }
        return .markerOnly
    }

    private static func isLegacyVersion(_ version: String?) -> Bool {
        guard let version,
              let recorded = releaseTriple(version),
              let cutoff = releaseTriple(defaultOnCutoff)
        else { return false }
        return recorded < cutoff
    }

    private static func releaseTriple(_ version: String) -> (Int, Int, Int)? {
        let pattern = #"^([0-9]+)\.([0-9]+)\.([0-9]+)(?:(?:rc|a|b)[0-9]+|\.dev[0-9]+)?$"#
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let match = regex.firstMatch(
                in: version,
                range: NSRange(version.startIndex..., in: version)
              ), match.range == NSRange(version.startIndex..., in: version),
              let majorRange = Range(match.range(at: 1), in: version),
              let minorRange = Range(match.range(at: 2), in: version),
              let patchRange = Range(match.range(at: 3), in: version),
              let major = Int(version[majorRange]),
              let minor = Int(version[minorRange]),
              let patch = Int(version[patchRange]),
              (major, minor, patch) != (0, 0, 0)
        else { return nil }
        return (major, minor, patch)
    }

    static func readSharedConsent(at url: URL) -> SharedConsent? {
        guard let mapping = readConsentMapping(at: url) else { return nil }
        return sharedConsent(from: mapping)
    }

    private static func sharedConsent(from mapping: [String: Any]) -> SharedConsent {
        SharedConsent(
            engine: boolValue(mapping["consent"]),
            desktop: boolValue(mapping["desktop_consent"]),
            noticeRevisionSeen: intValue(mapping["notice_revision_seen"]),
            promptedVersion: mapping["prompted_version"] as? String
        )
    }

    private static func boolValue(_ value: Any?) -> Bool? {
        guard let number = value as? NSNumber,
              CFGetTypeID(number) == CFBooleanGetTypeID()
        else { return nil }
        return number.boolValue
    }

    private static func intValue(_ value: Any?) -> Int? {
        guard let number = value as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID()
        else { return nil }
        return number.intValue
    }

    /// Reads Desktop JSON or the engine's flat YAML mapping. Inline YAML
    /// comments remain defensively stripped: a comment-only scalar must never
    /// turn launch into a fatal subscript trap.
    private static func readConsentMapping(at url: URL) -> [String: Any]? {
        guard FileManager.default.fileExists(atPath: url.path) else { return [:] }
        guard let data = try? Data(contentsOf: url) else { return nil }
        if let json = try? JSONSerialization.jsonObject(with: data),
           let mapping = json as? [String: Any] {
            return mapping
        }
        guard let text = String(data: data, encoding: .utf8) else { return nil }
        var mapping: [String: Any] = [:]
        for rawLine in text.split(whereSeparator: \.isNewline) {
            let trimmed = rawLine.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty || trimmed.hasPrefix("#") { continue }
            let pieces = rawLine.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false)
            guard pieces.count == 2 else { return nil }
            let key = pieces[0].trimmingCharacters(in: .whitespaces)
            guard !key.isEmpty else { return nil }
            let value = pieces[1]
                .prefix(while: { $0 != "#" })
                .trimmingCharacters(in: .whitespaces)
            guard !value.isEmpty else { return nil }
            mapping[key] = yamlScalar(value)
        }
        return mapping
    }

    private static func yamlScalar(_ value: String) -> Any {
        let lowered = value.lowercased()
        if lowered == "true" { return true }
        if lowered == "false" { return false }
        if lowered == "null" || value == "~" { return NSNull() }
        if let integer = Int(value) { return integer }
        if value.count >= 2,
           (value.hasPrefix("\"") && value.hasSuffix("\""))
            || (value.hasPrefix("'") && value.hasSuffix("'")) {
            return String(value.dropFirst().dropLast())
        }
        return value
    }

    private static func consentURL(in directory: URL) -> URL {
        directory.appendingPathComponent("telemetry-consent.yaml", isDirectory: false)
    }

    /// Read-merge-write under the permanent sibling flock. Lock open/acquire
    /// failures fall back to the same unlocked merge-and-atomic-replace so a
    /// stale root-owned lock cannot block an explicit Settings choice.
    @discardableResult
    static func writeMergedConsent(
        updates: [String: Any],
        raiseNoticeRevisionTo revision: Int?,
        directory: URL,
        replaceUnreadable: Bool,
        customMerge: (([String: Any]) -> [String: Any]?)? = nil
    ) -> Bool {
        let fm = FileManager.default
        do {
            try fm.createDirectory(
                at: directory,
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700]
            )
            try fm.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
        } catch {
            return false
        }

        let url = consentURL(in: directory)
        let lockURL = directory.appendingPathComponent(
            "telemetry-consent.yaml.lock", isDirectory: false
        )

        func mergeAndReplace() -> Bool {
            let existing = readConsentMapping(at: url)
            guard existing != nil || replaceUnreadable else { return false }
            let current = existing ?? [:]
            if let customMerge {
                guard let merged = customMerge(current) else { return false }
                return atomicReplace(mapping: merged, at: url)
            }
            var merged = current
            for (key, value) in updates { merged[key] = value }
            if merged["schema_version"] == nil { merged["schema_version"] = 1 }
            if let revision {
                let current = intValue(merged["notice_revision_seen"]) ?? 0
                merged["notice_revision_seen"] = max(current, revision)
            }
            return atomicReplace(mapping: merged, at: url)
        }

        let lockFD = lockURL.path.withCString {
            open($0, O_CREAT | O_RDWR, mode_t(0o600))
        }
        guard lockFD >= 0 else { return mergeAndReplace() }

        let deadline = Date().addingTimeInterval(lockRetrySeconds)
        var acquired = false
        while true {
            if flock(lockFD, LOCK_EX | LOCK_NB) == 0 {
                acquired = true
                break
            }
            let error = errno
            let remaining = deadline.timeIntervalSinceNow
            guard [EACCES, EAGAIN, EINTR].contains(error), remaining > 0 else { break }
            let remainingMicroseconds = useconds_t(remaining * 1_000_000)
            usleep(min(lockRetryIntervalMicroseconds, remainingMicroseconds))
        }

        guard acquired else {
            close(lockFD)
            return mergeAndReplace()
        }
        defer {
            _ = flock(lockFD, LOCK_UN)
            close(lockFD)
        }
        return mergeAndReplace()
    }

    private static func atomicReplace(mapping: [String: Any], at url: URL) -> Bool {
        guard JSONSerialization.isValidJSONObject(mapping),
              var data = try? JSONSerialization.data(
                withJSONObject: mapping,
                options: [.prettyPrinted, .sortedKeys]
              )
        else { return false }
        data.append(0x0A)

        var template = Array("\(url.path).tmp.XXXXXX".utf8CString)
        let descriptor = mkstemp(&template)
        guard descriptor >= 0 else { return false }
        let temporaryPath = String(
            decoding: template.prefix(while: { $0 != 0 }).map { UInt8(bitPattern: $0) },
            as: UTF8.self
        )
        var descriptorIsOpen = true
        var shouldRemoveTemporary = true
        defer {
            if descriptorIsOpen { close(descriptor) }
            if shouldRemoveTemporary { _ = unlink(temporaryPath) }
        }

        let wroteAll = data.withUnsafeBytes { buffer -> Bool in
            guard let base = buffer.baseAddress else { return data.isEmpty }
            var offset = 0
            while offset < buffer.count {
                let count = Darwin.write(descriptor, base.advanced(by: offset), buffer.count - offset)
                if count < 0 && errno == EINTR { continue }
                guard count > 0 else { return false }
                offset += count
            }
            return true
        }
        guard wroteAll,
              fsync(descriptor) == 0,
              fchmod(descriptor, mode_t(0o600)) == 0,
              Darwin.close(descriptor) == 0
        else { return false }
        descriptorIsOpen = false
        guard rename(temporaryPath, url.path) == 0 else { return false }
        shouldRemoveTemporary = false
        return true
    }
}
