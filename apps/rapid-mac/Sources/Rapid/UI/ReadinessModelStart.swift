import Foundation

/// The one server capability a readiness action needs: bring `alias` up,
/// switching away from whatever is currently resident. Narrowed to a single
/// method so a test can supply a spy without depending on the whole `final`
/// `ServerManager` (which spawns real child processes).
@MainActor
protocol ReadinessServing {
    func ensureServing(alias: String, hfPath: String?) async -> Bool
}

extension ServerManager: ReadinessServing {}

/// Where a readiness action ("Load model", "Retry", "Download and start")
/// turns into a server call.
///
/// It goes through ``ServerManager/ensureServing(alias:hfPath:)`` — NEVER
/// ``start`` — because `start` is cold-start only: it no-ops when a DIFFERENT
/// model is already resident. A Chat readiness action fired while an Images
/// checkpoint is loaded would then silently do nothing (#1739). The Chat and
/// Images readiness handlers were separate copies of this call and drifted —
/// Chat regressed to `start` twice — so the rule now lives in one place they
/// both route through, with a behavioural test.
enum ReadinessModelStart {
    @MainActor
    static func perform(_ server: ReadinessServing, alias: String, hfPath: String?) async {
        _ = await server.ensureServing(alias: alias, hfPath: hfPath)
    }
}
