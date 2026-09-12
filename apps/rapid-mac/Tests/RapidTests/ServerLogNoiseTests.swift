import Foundation
import Testing
@testable import Rapid

/// 0.14.1 dogfood: the "Server log" drawer contained nothing but the app's own
/// `/healthz` and `/v1/models/residency` polls. Both loops run on a timer
/// (health probe; 5-second residency refresh), uvicorn logs one access line
/// per request, and the drawer is a 200-line ring buffer — so the startup
/// banner, the resolved model path and every warning worth reading had already
/// scrolled off by the time anyone opened it.
///
/// These tests pin the narrow shape of the suppression. The risk of a filter
/// like this is that it hides something a user needed, so the negative cases
/// below matter more than the positive ones.
@Suite("Server log noise suppression")
struct ServerLogNoiseTests {

    @Test("Successful polls on the app's own endpoints are suppressed")
    func suppressesSuccessfulPolls() {
        #expect(ServerLogNoise.isAppPollAccessLine(
            #"INFO:     127.0.0.1:57230 - "GET /healthz HTTP/1.1" 200 OK"#))
        #expect(ServerLogNoise.isAppPollAccessLine(
            #"INFO:     127.0.0.1:57231 - "GET /v1/models/residency HTTP/1.1" 200 OK"#))
        // HEAD, HTTP/1.0, a query string, and a 2xx that is not 200.
        #expect(ServerLogNoise.isAppPollAccessLine(
            #"INFO:     127.0.0.1:1 - "HEAD /healthz HTTP/1.0" 204 No Content"#))
        #expect(ServerLogNoise.isAppPollAccessLine(
            #"INFO:     127.0.0.1:1 - "GET /v1/models/residency?verbose=1 HTTP/1.1" 200 OK"#))
    }

    @Test("A FAILED poll survives — that is what the drawer is for")
    func keepsFailedPolls() {
        // Someone opens the log drawer precisely because the server is not
        // answering. Hiding the evidence would be the worse bug.
        #expect(!ServerLogNoise.isAppPollAccessLine(
            #"INFO:     127.0.0.1:1 - "GET /healthz HTTP/1.1" 503 Service Unavailable"#))
        #expect(!ServerLogNoise.isAppPollAccessLine(
            #"INFO:     127.0.0.1:1 - "GET /v1/models/residency HTTP/1.1" 500 Internal Server Error"#))
        #expect(!ServerLogNoise.isAppPollAccessLine(
            #"INFO:     127.0.0.1:1 - "GET /healthz HTTP/1.1" 404 Not Found"#))
    }

    @Test("User-driven requests survive")
    func keepsUserTraffic() {
        #expect(!ServerLogNoise.isAppPollAccessLine(
            #"INFO:     127.0.0.1:1 - "POST /v1/chat/completions HTTP/1.1" 200 OK"#))
        #expect(!ServerLogNoise.isAppPollAccessLine(
            #"INFO:     127.0.0.1:1 - "POST /v1/models/load HTTP/1.1" 200 OK"#))
        #expect(!ServerLogNoise.isAppPollAccessLine(
            #"INFO:     127.0.0.1:1 - "GET /v1/models HTTP/1.1" 200 OK"#))
    }

    @Test("Prose that merely mentions an endpoint survives")
    func keepsProseMentioningEndpoints() {
        // The access-log shape (quoted request line + version + status) is the
        // discriminator, so a warning, a traceback frame or a banner naming
        // the endpoint is never swallowed.
        #expect(!ServerLogNoise.isAppPollAccessLine(
            "WARNING: /healthz answered slowly (2.4s) — model still loading"))
        #expect(!ServerLogNoise.isAppPollAccessLine(
            "  File \"/app/vllm_mlx/server.py\", line 1, in healthz"))
        #expect(!ServerLogNoise.isAppPollAccessLine(
            "INFO: probe endpoint is /v1/models/residency"))
        #expect(!ServerLogNoise.isAppPollAccessLine(""))
    }

    @Test("A diagnostic that QUOTES a poll's request line survives")
    func keepsProseQuotingAnAccessLine() {
        // codex, reviewing this PR: an unanchored pattern would suppress any
        // line containing the quoted fragment, including the one message a
        // user most needs when health checks are lying to the app. The
        // pattern is anchored at both ends, so a line is only dropped when it
        // is nothing but an access line.
        #expect(!ServerLogNoise.isAppPollAccessLine(
            #"WARNING: probe failed: sent "GET /healthz HTTP/1.1" 200 but the body was empty"#))
        #expect(!ServerLogNoise.isAppPollAccessLine(
            #"ERROR: residency refresh loop is wedged; last line was 127.0.0.1:1 - "GET /v1/models/residency HTTP/1.1" 200 OK"#))
        // A trailing comment on an otherwise well-formed access line means
        // someone wrapped it in prose, so it is not uvicorn's own output.
        #expect(!ServerLogNoise.isAppPollAccessLine(
            #"INFO:     127.0.0.1:1 - "GET /healthz HTTP/1.1" 200 OK  <-- took 2.4s, model still loading"#))
    }

    @Test("The access logger propagating to a root handler is still suppressed")
    func suppressesPropagatedAccessLines() {
        // The server calls `uvicorn.run(app, ...)` with no `log_config`, so
        // uvicorn's padded `levelprefix` is the shape today. If the access
        // logger is ever left to propagate to a root handler instead, Python's
        // default `%(levelname)s:%(name)s:%(message)s` is what comes out, and
        // the filter must not quietly stop working.
        #expect(ServerLogNoise.isAppPollAccessLine(
            #"INFO:uvicorn.access:127.0.0.1:57230 - "GET /healthz HTTP/1.1" 200 OK"#))
        // IPv6 client address, and a line arriving with a trailing CR from the
        // pipe.
        #expect(ServerLogNoise.isAppPollAccessLine(
            "INFO:     [::1]:57230 - \"GET /healthz HTTP/1.1\" 200 OK\r"))
    }

    @Test("Colourised access lines are suppressed too")
    func suppressesColourisedLines() {
        // uvicorn bolds the request line when `use_colors` is on. The app
        // spawns the child on a pipe so colours are off today, but a future
        // TTY-attached child must not smuggle the noise back in.
        let bold = "\u{1B}[32mINFO\u{1B}[0m:     127.0.0.1:1 - \"\u{1B}[1mGET /healthz HTTP/1.1\u{1B}[0m\" \u{1B}[32m200 OK\u{1B}[0m"
        #expect(ServerLogNoise.isAppPollAccessLine(bold))
    }
    @Test("An access line from another machine is kept")
    func keepsAccessLinesFromOtherClients() {
        // codex round 6: any client address matched, so a reachability probe
        // from a phone on the LAN was filed as the app's own polling noise —
        // and that line is exactly what someone debugging "why can't my other
        // machine reach this?" opened the drawer to find.
        #expect(!ServerLogNoise.isAppPollAccessLine(
            #"INFO:     192.168.1.42:51234 - "GET /healthz HTTP/1.1" 200 OK"#))
        #expect(!ServerLogNoise.isAppPollAccessLine(
            #"INFO:     10.0.0.7:8080 - "GET /v1/models/residency HTTP/1.1" 200 OK"#))
        // Loopback in each of the forms uvicorn can render it stays filtered.
        #expect(ServerLogNoise.isAppPollAccessLine(
            #"INFO:     127.0.0.1:57230 - "GET /healthz HTTP/1.1" 200 OK"#))
        #expect(ServerLogNoise.isAppPollAccessLine(
            #"INFO:     ::1:57230 - "GET /healthz HTTP/1.1" 200 OK"#))
        #expect(ServerLogNoise.isAppPollAccessLine(
            #"INFO:     localhost:57230 - "GET /healthz HTTP/1.1" 200"#))
    }

}
