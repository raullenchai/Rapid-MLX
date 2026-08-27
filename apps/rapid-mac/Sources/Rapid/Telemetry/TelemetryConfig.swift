import Foundation

/// Static knobs for the anonymous-usage telemetry pipeline.
///
/// We deliberately ship one endpoint and one schema version. A
/// version bump goes into the source — no remote config, no kill
/// switch, no A/B. The privacy contract reduces to "the code in
/// this directory plus this one URL"; users can audit it.
///
/// Endpoint is the same Cloudflare Worker the rapid-mlx Python
/// server posts to (telemetry.rapidmlx.com → R2). The Worker
/// validates ``schema_version == 1`` and the small required-fields
/// set; unknown fields pass through, so we can carry rapid-desktop-
/// specific extras (``error_type``, ``stack_frames``, ``context``)
/// without forking the Worker.
enum TelemetryConfig {
    /// Production endpoint. See ``endpoint`` for the (loopback-only)
    /// developer override.
    static let productionEndpoint = URL(string: "https://telemetry.rapidmlx.com/v1/events")!

    /// Where event batches POST. Normally ``productionEndpoint``. A
    /// developer may set ``RAPID_MLX_TELEMETRY_ENDPOINT`` to a **loopback**
    /// URL (`http://127.0.0.1:…` / `http://localhost:…`) to point the
    /// pipeline at a local capture server and audit exactly what the app
    /// sends. The override is restricted to loopback + `http` so it can
    /// never redirect a real user's telemetry to an arbitrary remote host.
    static var endpoint: URL {
        resolveEndpoint(environment: ProcessInfo.processInfo.environment)
    }

    /// Testable core of ``endpoint``. Honors ``RAPID_MLX_TELEMETRY_ENDPOINT``
    /// ONLY when it parses to a loopback ``http`` URL (``127.0.0.1`` /
    /// ``localhost``); a remote host, an ``https`` remote, or garbage all
    /// fall back to ``productionEndpoint`` so the override can never exfil
    /// a real user's telemetry off-box.
    static func resolveEndpoint(environment: [String: String]) -> URL {
        if let raw = environment["RAPID_MLX_TELEMETRY_ENDPOINT"],
           let url = URL(string: raw),
           url.scheme == "http",
           let host = url.host,
           host == "127.0.0.1" || host == "localhost" {
            return url
        }
        return productionEndpoint
    }

    /// Schema version the Worker requires on every event.
    static let schemaVersion = 1

    /// UserDefaults key for the explicit telemetry decision. An
    /// absent value means the user has not answered yet and telemetry
    /// stays OFF. The post-value invitation records either ``true`` or
    /// ``false``; Settings updates the same key later.
    static let enabledKey = "com.rapidmlx.rapid.telemetry.enabled"

    /// UserDefaults key for the per-install ``client_id`` UUID.
    /// Legacy migration/fallback copy of the engine-shared ID. A
    /// `rapid-mlx telemetry reset` rotates it on the next opt-in.
    static let clientIDKey = "com.rapidmlx.rapid.telemetry.clientID"

    /// One-time migration markers. Once desktop state has been copied
    /// into the engine's shared files, a later missing file means the
    /// user ran `rapid-mlx telemetry reset` (or deleted it manually),
    /// not that desktop should resurrect the legacy value.
    static let sharedClientIDMigrationKey =
        "com.rapidmlx.rapid.telemetry.sharedClientIDMigrated"
    static let sharedConsentMigrationKey =
        "com.rapidmlx.rapid.telemetry.sharedConsentMigrated"

    /// Per-app-launch session UUID. Lives only for the process.
    static let sessionID = UUID().uuidString

    /// User-facing on/off check. Default is ``false`` until the user
    /// explicitly opts in.
    static var isEnabled: Bool { isEnabled(defaults: .standard) }

    /// Testable variant that reads the opt-out flag from an injected
    /// ``UserDefaults`` rather than the process-wide ``.standard``
    /// domain. Product code uses the ``isEnabled`` property (which
    /// resolves to ``.standard``); tests pass a private
    /// ``UserDefaults(suiteName:)`` so a parallel sibling test can't
    /// clear the shared key between this read and a matching write
    /// (issue #530 — flaky ``clientIDPersists`` under the parallel
    /// test pool).
    static func isEnabled(defaults: UserDefaults) -> Bool {
        defaults.object(forKey: enabledKey) as? Bool ?? false
    }
}
