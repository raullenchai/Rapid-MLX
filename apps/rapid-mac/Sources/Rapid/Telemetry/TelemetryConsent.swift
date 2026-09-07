import Foundation

/// Bridges the desktop privacy choice to rapid-mlx's existing consent
/// state. JSON is deliberately written into the engine's `.yaml` file:
/// JSON is valid YAML, and using `JSONSerialization` avoids maintaining
/// a second ad-hoc YAML encoder in the desktop app.
enum TelemetryConsent {
    /// A decision is owed only when none is stored AND the environment has
    /// not already settled the question: under a kill switch (explicit,
    /// `DO_NOT_TRACK`, or a CI marker) nothing can be sent, so the invitation
    /// is never shown — the engine skips its first-run prompt the same way.
    static func needsDecision(
        defaults: UserDefaults = .standard,
        environment: [String: String] = TelemetryConfig.environment
    ) -> Bool {
        !TelemetryConfig.killSwitchActive(environment: environment)
            && defaults.object(forKey: TelemetryConfig.enabledKey) == nil
    }

    /// One-time migration for users who explicitly changed the old
    /// Settings toggle. That explicit desktop choice becomes the shared
    /// decision; after migration the shared file is authoritative.
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
        version: String,
        defaults: UserDefaults,
        telemetryDirectory: URL
    ) {
        let localDecision = defaults.object(
            forKey: TelemetryConfig.enabledKey
        ) as? Bool
        let alreadyMigrated = defaults.bool(
            forKey: TelemetryConfig.sharedConsentMigrationKey
        )
        let consentURL = telemetryDirectory.appendingPathComponent(
            "telemetry-consent.yaml",
            isDirectory: false
        )
        if FileManager.default.fileExists(atPath: consentURL.path) {
            if alreadyMigrated {
                // After migration the shared engine file is the single
                // source of truth. Only desktop_consent authorizes the
                // desktop's broader redacted crash diagnostics; a CLI-only
                // `consent: true` must not silently expand that scope.
                if let sharedDecision = readSharedConsent(at: consentURL)?
                    .desktop {
                    defaults.set(
                        sharedDecision,
                        forKey: TelemetryConfig.enabledKey
                    )
                    if sharedDecision {
                        synchronizeClientID(
                            defaults: defaults,
                            telemetryDirectory: telemetryDirectory
                        )
                    }
                } else {
                    // The engine treats a malformed record as absent and
                    // re-prompts; desktop must do the same.
                    defaults.removeObject(forKey: TelemetryConfig.enabledKey)
                }
                return
            }

            if let localDecision {
                // One-time legacy migration: a desktop user who had
                // explicitly changed the old toggle keeps that choice.
                if localDecision {
                    synchronizeClientID(
                        defaults: defaults,
                        telemetryDirectory: telemetryDirectory
                    )
                }
                let persisted = writeSharedConsent(
                    enabled: localDecision,
                    version: version,
                    directory: telemetryDirectory
                )
                defaults.set(
                    persisted,
                    forKey: TelemetryConfig.sharedConsentMigrationKey
                )
            } else if let sharedDecision = readSharedConsent(at: consentURL)?
                .desktop {
                // A desktop decision survived while UserDefaults did
                // not (for example, app reinstall). Reuse that exact
                // desktop-scoped choice.
                defaults.set(sharedDecision, forKey: TelemetryConfig.enabledKey)
                defaults.set(true, forKey: TelemetryConfig.sharedConsentMigrationKey)
                if sharedDecision {
                    synchronizeClientID(
                        defaults: defaults,
                        telemetryDirectory: telemetryDirectory
                    )
                }
            }
            return
        }

        guard let localDecision else { return }

        if alreadyMigrated {
            // The shared decision existed after migration and is now
            // missing. Honour engine `telemetry reset`: clear desktop's
            // answer so the next main window asks again.
            defaults.removeObject(forKey: TelemetryConfig.enabledKey)
            return
        }

        if localDecision {
            synchronizeClientID(
                defaults: defaults,
                telemetryDirectory: telemetryDirectory
            )
        }
        let persisted = writeSharedConsent(
            enabled: localDecision,
            version: version,
            directory: telemetryDirectory
        )
        defaults.set(
            persisted,
            forKey: TelemetryConfig.sharedConsentMigrationKey
        )
    }

    private static func synchronizeClientID(
        defaults: UserDefaults,
        telemetryDirectory: URL
    ) {
        _ = TelemetryIdentity.clientID(
            defaults: defaults,
            sharedIDURL: telemetryDirectory.appendingPathComponent(
                "telemetry-client-id",
                isDirectory: false
            )
        )
    }

    /// Read both desktop-written JSON (valid YAML) and the engine's
    /// original small YAML mapping. The fallback intentionally accepts
    /// only the single `consent: true|false` scalar we own; it is not a
    /// general-purpose YAML parser.
    private struct SharedConsent {
        let engine: Bool
        let desktop: Bool?
    }

    private static func readSharedConsent(at url: URL) -> SharedConsent? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        if let json = try? JSONSerialization.jsonObject(with: data),
           let object = json as? [String: Any],
           let consent = object["consent"] as? Bool {
            return SharedConsent(
                engine: consent,
                desktop: object["desktop_consent"] as? Bool
            )
        }
        guard let text = String(data: data, encoding: .utf8) else { return nil }
        var engine: Bool?
        var desktop: Bool?
        for line in text.split(whereSeparator: \.isNewline) {
            let pieces = line.split(separator: ":", maxSplits: 1)
            guard pieces.count == 2 else { continue }
            let key = pieces[0].trimmingCharacters(in: .whitespaces)
            guard key == "consent" || key == "desktop_consent" else { continue }
            // Strip an inline `# comment` without indexing into
            // `split`'s result: `String.split` omits empty
            // subsequences, so a value that is *only* a comment
            // (e.g. `consent:#` or `consent: # note`) yields `[]`
            // and `[0]` would fatal-trap at launch — turning a
            // malformed shared file into a crash-loop instead of the
            // "treat as absent and re-prompt" the design promises.
            // `prefix(while:)` yields "" in that case, which falls
            // through to the malformed → `return nil` branch below.
            let value = pieces[1]
                .prefix(while: { $0 != "#" })
                .trimmingCharacters(in: .whitespaces)
                .lowercased()
            let parsed: Bool
            if value == "true" {
                parsed = true
            } else if value == "false" {
                parsed = false
            } else {
                return nil
            }
            if key == "consent" { engine = parsed }
            if key == "desktop_consent" { desktop = parsed }
        }
        guard let engine else { return nil }
        return SharedConsent(engine: engine, desktop: desktop)
    }

    static func record(
        enabled: Bool,
        version: String = TelemetryClient.currentVersion()
    ) {
        // Reports captured before explicit desktop consent must never
        // become eligible merely because the user opts in later.
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

    /// Testable core. Recording to UserDefaults happens even if the
    /// shared directory is unwritable; both telemetry pipelines remain
    /// fail-open with respect to app behavior.
    static func record(
        enabled: Bool,
        version: String,
        defaults: UserDefaults,
        telemetryDirectory: URL
    ) {
        defaults.set(enabled, forKey: TelemetryConfig.enabledKey)
        if enabled {
            _ = TelemetryIdentity.clientID(
                defaults: defaults,
                sharedIDURL: telemetryDirectory.appendingPathComponent(
                    "telemetry-client-id",
                    isDirectory: false
                )
            )
        }
        let persisted = writeSharedConsent(
            enabled: enabled,
            version: version,
            directory: telemetryDirectory
        )
        defaults.set(
            persisted,
            forKey: TelemetryConfig.sharedConsentMigrationKey
        )
    }

    @discardableResult
    private static func writeSharedConsent(
        enabled: Bool,
        version: String,
        directory: URL
    ) -> Bool {
        let payload: [String: Any] = [
            "consent": enabled,
            // rapid-mlx ignores unknown keys. Keeping a separate marker
            // prevents its narrower CLI disclosure from authorizing
            // desktop crash diagnostics without a desktop decision.
            "desktop_consent": enabled,
            "prompted_at": ISO8601DateFormatter().string(from: Date()),
            "prompted_version": version,
            "schema_version": 1,
        ]
        guard var data = try? JSONSerialization.data(
            withJSONObject: payload,
            options: [.prettyPrinted, .sortedKeys]
        ) else { return false }
        data.append(0x0A)

        let fm = FileManager.default
        let url = directory.appendingPathComponent(
            "telemetry-consent.yaml",
            isDirectory: false
        )
        do {
            try fm.createDirectory(
                at: directory,
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700]
            )
            try data.write(to: url, options: .atomic)
            try fm.setAttributes(
                [.posixPermissions: 0o600],
                ofItemAtPath: url.path
            )
            return true
        } catch {
            // Consent persistence cannot be allowed to break launch or
            // Settings. Desktop's local decision still gates its sender.
            return false
        }
    }
}
