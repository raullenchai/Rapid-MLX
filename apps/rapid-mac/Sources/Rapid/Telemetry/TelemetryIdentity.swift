import Foundation

/// Stable per-install + per-launch identifiers for telemetry events.
///
/// ``client_id`` is shared with the bundled rapid-mlx engine through
/// ``~/.rapid-mlx/telemetry-client-id``. Keeping one identifier means
/// a desktop session and its embedded engine requests count as one
/// anonymous install in distinct-user rollups instead of two. The
/// legacy UserDefaults value remains as a migration/fallback copy.
///
/// ``session_id`` lives in ``TelemetryConfig.sessionID`` and rotates
/// on every process launch.
enum TelemetryIdentity {
    /// Sentinel returned when telemetry is opted out at event-build
    /// time. Encodes into the wire envelope as a non-routable
    /// placeholder; the gate in ``TelemetryClient.send`` drops the
    /// envelope before it leaves the process, so this value never
    /// hits the network in practice. Keeps the type system honest
    /// (we still produce a ``String``) without persisting a real
    /// install ID on disk for an opted-out user.
    static let optedOutPlaceholder = "00000000-0000-0000-0000-000000000000"

    /// Production accessor. Reads the engine's shared identity first,
    /// then migrates the legacy UserDefaults value, then creates a new
    /// UUID. Tests use the overload below so they never touch the real
    /// home directory.
    static func clientID() -> String {
        clientID(
            defaults: .standard,
            sharedIDURL: sharedClientIDURL()
        )
    }

    /// Isolated accessor retained for existing tests and callers that
    /// inject a private defaults suite. No shared filesystem path is
    /// consulted in this shape.
    static func clientID(defaults: UserDefaults) -> String {
        clientID(defaults: defaults, sharedIDURL: nil)
    }

    /// Returns the persistent per-install client identifier,
    /// generating one on first call. The Worker accepts strings up
    /// to 64 bytes; a stock UUID is 36 bytes so we fit with room.
    ///
    /// Codex audit batch 8 finding T5 (P3): the previous shape
    /// persisted a fresh UUID on every event-build call, even for
    /// users who had opted out — leaving a persistent ``client_id``
    /// in UserDefaults for a user who never sent telemetry. Now
    /// we short-circuit when ``TelemetryConfig.isEnabled`` is
    /// false: no read, no write, no UUID, no persistent identifier.
    ///
    /// The ``defaults`` parameter defaults to ``.standard`` (the real
    /// per-install store) so product call sites are unchanged; tests
    /// pass a private ``UserDefaults(suiteName:)`` so a parallel
    /// sibling can't clear ``clientIDKey`` between the write and the
    /// read-back (issue #530).
    static func clientID(
        defaults: UserDefaults,
        sharedIDURL: URL?
    ) -> String {
        guard TelemetryConfig.isEnabled(defaults: defaults) else { return optedOutPlaceholder }

        if let sharedIDURL,
           let shared = try? String(contentsOf: sharedIDURL, encoding: .utf8),
           let existing = usableID(shared) {
            defaults.set(existing, forKey: TelemetryConfig.clientIDKey)
            defaults.set(true, forKey: TelemetryConfig.sharedClientIDMigrationKey)
            return existing
        }

        if let stored = defaults.string(forKey: TelemetryConfig.clientIDKey),
           let existing = usableID(stored) {
            if let sharedIDURL {
                let alreadyMigrated = defaults.bool(
                    forKey: TelemetryConfig.sharedClientIDMigrationKey
                )
                if !alreadyMigrated {
                    if persist(existing, at: sharedIDURL) {
                        defaults.set(
                            true,
                            forKey: TelemetryConfig.sharedClientIDMigrationKey
                        )
                    }
                    return existing
                }
                // The shared file existed after migration and is now
                // gone: honour engine `telemetry reset` by rotating
                // instead of restoring the old UserDefaults copy.
            } else {
                return existing
            }
        }

        let fresh = UUID().uuidString
        defaults.set(fresh, forKey: TelemetryConfig.clientIDKey)
        if let sharedIDURL {
            let persisted = persist(fresh, at: sharedIDURL)
            defaults.set(
                persisted,
                forKey: TelemetryConfig.sharedClientIDMigrationKey
            )
        }
        return fresh
    }

    /// Engine-compatible state directory. ``HOME`` is read at call
    /// time so dogfood/test launches with an isolated home never touch
    /// the user's production identity.
    static func sharedTelemetryDirectory(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> URL {
        let home: URL
        if let raw = environment["HOME"], raw.hasPrefix("/") {
            home = URL(fileURLWithPath: raw, isDirectory: true)
        } else {
            home = FileManager.default.homeDirectoryForCurrentUser
        }
        return home.appendingPathComponent(".rapid-mlx", isDirectory: true)
    }

    static func sharedClientIDURL(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> URL {
        sharedTelemetryDirectory(environment: environment)
            .appendingPathComponent("telemetry-client-id", isDirectory: false)
    }

    private static func usableID(_ raw: String) -> String? {
        let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty,
              value.utf8.count <= 64,
              value.unicodeScalars.allSatisfy({
                  !CharacterSet.controlCharacters.contains($0)
              }) else {
            return nil
        }
        return value
    }

    @discardableResult
    private static func persist(_ value: String, at url: URL) -> Bool {
        let fm = FileManager.default
        do {
            try fm.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700]
            )
            // Identity bytes are owner-only from creation.
            try OwnerOnlyFile.writeAtomically(Data("\(value)\n".utf8), to: url)
            return true
        } catch {
            // Telemetry state must never make the app fail. The
            // UserDefaults copy still gives desktop events a stable ID.
            return false
        }
    }
}
