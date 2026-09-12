import Foundation

/// Suppression for the access-log lines the app itself generates, so
/// the user-visible "Server log" drawer shows the child's own output.
///
/// The drawer is a 200-line ring buffer fed from the child's
/// stdout/stderr. Two app-side loops poll the child on a timer:
/// `ServerManager`'s `/healthz` probe and the 5-second
/// `/v1/models/residency` refresh that keeps the model rail current
/// (see `ContentView`'s residency `.task`). Uvicorn logs one access
/// line per request, so the two loops alone write several lines a
/// minute — enough to evict the entire buffer within a few minutes of
/// idling. Dogfooding 0.14.1, opening the drawer after a chat session
/// showed *only* those lines: the startup banner, the resolved model
/// path, and every warning worth reading had already scrolled off, so
/// the one surface that answers "what is the server doing?" answered
/// "polling itself".
///
/// This is the same move `DownloadProgress.isHeartbeatLogLine` makes
/// for the R2 puller's `[bytes]` heartbeat, for the same reason: a
/// machine-generated line at a fixed cadence carries no signal a human
/// reads, and its real cost is the signal it evicts.
///
/// Deliberately narrow:
///
///   * Only the paths the app polls on a timer. A `/v1/chat/completions`
///     or `/v1/models/load` line is a *user* action and stays.
///   * Only successful (2xx) polls. A `/healthz` that answers 503, or a
///     residency refresh that 500s, is exactly what someone opening the
///     drawer needs to see, so those lines are kept.
///   * Only the access-log shape — the quoted `GET … HTTP/1.1` request
///     line followed by a status code. A warning or traceback that
///     merely mentions `/healthz` in prose is not an access line and is
///     never suppressed.
///
/// Pure function, no shared state, matching `LogScrubber`'s contract.
enum ServerLogNoise {
    /// Endpoints the app polls on a timer, quoted for use in a regex
    /// alternation. Extend this when a new app-side poll loop lands.
    static let polledPaths = ["/healthz", "/v1/models/residency"]

    /// Uvicorn's default access format is
    /// `%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s`,
    /// e.g. `INFO:     127.0.0.1:57230 - "GET /healthz HTTP/1.1" 200 OK`.
    /// The optional query-string branch covers a future poll that
    /// carries parameters; the status branch is pinned to 2xx so
    /// failures survive.
    private static let accessLinePattern: String = {
        let alternation = polledPaths
            .map { NSRegularExpression.escapedPattern(for: $0) }
            .joined(separator: "|")
        return "\"(?:GET|HEAD) (?:\(alternation))(?:\\?[^ \"]*)? HTTP/[0-9.]+\" 2[0-9][0-9]"
    }()

    /// True when `line` is a successful access-log line for one of the
    /// app's own polling endpoints, and so should not reach the
    /// user-visible log tail.
    static func isAppPollAccessLine(_ line: String) -> Bool {
        // Colours are off when the child's stdout is a pipe (uvicorn
        // defaults `use_colors` to `isatty()`), but strip them anyway so
        // a future `--use-colors` or a TTY-attached child can't smuggle
        // the noise back in past the quote anchor.
        let stripped = line.replacingOccurrences(
            of: "\u{1B}\\[[0-9;]*m",
            with: "",
            options: .regularExpression
        )
        return stripped.range(of: accessLinePattern, options: .regularExpression) != nil
    }
}
