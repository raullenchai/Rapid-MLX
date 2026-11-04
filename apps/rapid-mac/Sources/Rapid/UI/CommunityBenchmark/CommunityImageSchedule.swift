import Foundation

/// The schedule encoded by registered image protocol IDs. Unknown IDs must
/// not silently inherit the legacy protocol's 20-step workload.
struct CommunityImageSchedule: Equatable {
    let steps: Int
    /// Nil means CFG is disabled; otherwise preserve the protocol's decimal.
    let cfg: String?

    init?(protocolID: String) {
        if protocolID == "rapid-image-speed" {
            steps = 20
            cfg = "3.5"
            return
        }
        guard let match = protocolID.wholeMatch(
            of: #/rapid-image-speed-([0-9]+)step-(?:nocfg|cfg([0-9]+)p([0-9]+))/#
        ), let count = Int(match.1), count > 0 else { return nil }
        steps = count
        cfg = match.2.flatMap { whole in match.3.map { "\(whole).\($0)" } }
    }

    var displayName: String {
        if let cfg { return String(localized: "\(steps) steps, CFG \(cfg)") }
        return String(localized: "\(steps) steps, no CFG")
    }
}
