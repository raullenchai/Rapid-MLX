import Foundation

/// The single role boundary for every rapid-mlx process owned by Desktop.
///
/// App-owned commands are sidecars even when they are short-lived CLI probes:
/// they must never print or claim the shared telemetry disclosure on behalf of
/// the GUI. Direct assignment deliberately overrides an ambient spoofed role.
enum EngineProcessEnvironment {
    static let roleKey = "RAPID_MLX_PROCESS_ROLE"
    static let sidecarRole = "desktop-sidecar"

    nonisolated static func sidecar(
        _ environment: [String: String]
    ) -> [String: String] {
        var result = environment
        result[roleKey] = sidecarRole
        return result
    }
}
