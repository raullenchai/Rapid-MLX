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
///   * Only loopback clients. A `/healthz` from another machine on the
///     LAN is someone testing reachability, and dropping it would hide
///     the one line that proves the request arrived.
///   * Only successful (2xx) polls. A `/healthz` that answers 503, or a
///     residency refresh that 500s, is exactly what someone opening the
///     drawer needs to see, so those lines are kept.
///   * Only a line that is *entirely* an access-log line. The pattern is
///     anchored at both ends, so a warning or traceback that merely
///     mentions `/healthz` — even one that quotes a whole request line
///     back at you, e.g. `WARNING: probe failed: sent "GET /healthz
///     HTTP/1.1" got 200 with an empty body` — is never suppressed.
///
/// Pure function, no shared state, matching `LogScrubber`'s contract.
enum ServerLogNoise {
    /// Endpoints the app polls on a timer, quoted for use in a regex
    /// alternation. Extend this when a new app-side poll loop lands.
    static let polledPaths = ["/healthz", "/v1/models/residency"]

    /// Uvicorn's default access format is
    /// `%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s`,
    /// e.g. `INFO:     127.0.0.1:57230 - "GET /healthz HTTP/1.1" 200 OK`.
    /// The server runs `uvicorn.run(app, …)` with no `log_config`, so
    /// uvicorn installs its own default config and that *is* the shape
    /// on the wire.
    ///
    /// Anchored at both ends (`^…$`), which is the whole safety
    /// argument: the line must be nothing but an access line, so prose
    /// that quotes a request line back at the reader stays visible. The
    /// leading group absorbs either uvicorn's padded `levelprefix`
    /// (`INFO:     `) or a `LEVEL:logger.name:` prefix, in case the
    /// access logger is ever left to propagate to a root handler using
    /// Python's default `%(levelname)s:%(name)s:%(message)s` format. The
    /// optional query-string branch covers a future poll that carries
    /// parameters; the status branch is pinned to 2xx so failures
    /// survive; the trailing group is the HTTP reason phrase
    /// (`OK`, `No Content`, `Non-Authoritative Information`).
    private static let accessLinePattern: String = {
        let alternation = polledPaths
            .map { NSRegularExpression.escapedPattern(for: $0) }
            .joined(separator: "|")
        let levelPrefix = "(?:[A-Z]+:(?:[A-Za-z0-9_.]+:)?[ \\t]*)?"
        // Loopback only. codex caught that any client address matched, so a
        // `/healthz` from a phone on the LAN — exactly the request someone
        // debugging "why can't my other machine reach the server?" opened the
        // drawer to look for — was filed as the app's own noise. Uvicorn
        // renders `client_addr` as `host:port` (`127.0.0.1:57230`), with the
        // port absent when the transport reports no peer.
        let loopbackHost = "(?:127\\.0\\.0\\.1|localhost|\\[::1\\]|::1)"
        let clientAddr = "\(loopbackHost)(?::[0-9]{1,5})?"
        let requestLine = "\"(?:GET|HEAD) (?:\(alternation))(?:\\?[^ \"]*)? HTTP/[0-9.]+\""
        let status = "2[0-9][0-9]"
        let reasonPhrase = "(?: [A-Za-z][A-Za-z'\\- ]*)?"
        return "^\(levelPrefix)\(clientAddr) - \(requestLine) \(status)\(reasonPhrase)$"
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
        // Trimmed because the pattern is `$`-anchored and lines arrive from a
        // pipe: a trailing `\r` (or indentation on a continuation line) must
        // not be the thing that decides whether the filter fires.
        let trimmed = stripped.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.range(of: accessLinePattern, options: .regularExpression) != nil
    }
}
