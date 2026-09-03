import Foundation
import Testing
@testable import Rapid

/// Pins the single-source-of-truth contract for the chat client's
/// default base URL (audit P1 `ChatStreamClient.swift:139` —
/// hard-coded `http://127.0.0.1:8000`). The literal port now
/// lives in exactly one place — `PortSweep.defaultPort` — and
/// the default-constructed client picks it up. If a future
/// refactor re-introduces a literal in either place, this suite
/// fails CI.
@Suite("ChatStreamClient default base URL — single source of truth")
struct ChatStreamClientDefaultBaseURLTests {
    @Test("Default base URL is derived from PortSweep.defaultPort, not a hardcoded literal")
    func default_base_url_matches_port_sweep() {
        let expected = URL(string: "http://127.0.0.1:\(PortSweep.defaultPort)")!
        #expect(ChatStreamClient.defaultBaseURL == expected)
    }

    @Test("Default-constructed client picks up the SoT-derived default")
    func default_init_uses_default_base_url() {
        let client = ChatStreamClient()
        #expect(client.baseURL == ChatStreamClient.defaultBaseURL)
    }

    @Test("Caller-supplied baseURL overrides the default — re-target path")
    func explicit_baseURL_overrides_default() {
        let custom = URL(string: "http://127.0.0.1:9999")!
        let client = ChatStreamClient(baseURL: custom)
        #expect(client.baseURL == custom)
        #expect(client.baseURL != ChatStreamClient.defaultBaseURL)
    }

    /// The default base URL must be a loopback HTTP URL. A future
    /// refactor that swaps the default to a remote address would
    /// blow up the privacy/local-first guarantee — this pin
    /// surfaces it loudly.
    @Test("Default base URL is HTTP + loopback (127.0.0.1)")
    func default_is_loopback_http() {
        let url = ChatStreamClient.defaultBaseURL
        #expect(url.scheme == "http",
                "Default must stay HTTP — we're on loopback, TLS is overhead. Switched to remote? Update audit doc.")
        #expect(url.host == "127.0.0.1",
                "Default must stay on loopback IP literal — DNS round-trip on chat startup is unacceptable. Got host: \(url.host ?? "<nil>")")
    }

    /// Codex r1 NIT-3 / r2 NIT-A: SoT comparison alone won't flag
    /// the case where someone intentionally moves
    /// `PortSweep.defaultPort` without auditing the other surfaces
    /// that hard-code a literal port for documentation or scripting
    /// reasons. Pin the absolute port number so a deliberate port
    /// change is a multi-place edit — this test fails loudly, the
    /// reviewer reads the list below, and bumps every pin in the
    /// same PR.
    ///
    /// Downstream surfaces re-audited at the 0.13.x 7659 default
    /// change (grep `8000` / `7659` confirmed — `.github/workflows/`
    /// has none today, so do NOT cite a release.yml smoke probe):
    ///   * `docs/userflows.md` — (already clean of a literal port)
    ///   * `docs/plans/v1-prod-readiness-gaps.md` — (already clean)
    ///   * `scripts/fake-rapid-mlx.sh` — (already clean)
    ///   * `Sources/Rapid/TestDriver.swift` — (already clean)
    ///   * `Sources/Rapid/Server/ServerManager.swift` — doc comments
    @Test("Default port is the documented 7659 — bump fails loudly if PortSweep.defaultPort moves")
    func default_port_is_pinned_7659() {
        #expect(ChatStreamClient.defaultBaseURL.port == 7659,
                "If you intentionally changed PortSweep.defaultPort, update this pin AND every docs/scripts surface listed in the doc-comment above.")
    }

    /// Codex r1 BLOCKING: the re-target site in `ChatViewModel.send()`
    /// previously open-coded `URL(string: "http://127.0.0.1:\(port)")!`
    /// — a duplicate of the SoT shape. The whole point of
    /// `loopbackURL(port:)` is to be the sole construction site.
    /// Pin its output so a future refactor that switches host or
    /// scheme has exactly one place to land.
    @Test("loopbackURL(port:) is the sole scheme+host constructor")
    func loopback_url_pins_scheme_and_host() {
        let url = ChatStreamClient.loopbackURL(port: 9001)
        #expect(url.scheme == "http")
        #expect(url.host == "127.0.0.1")
        #expect(url.port == 9001)
        #expect(url.absoluteString == "http://127.0.0.1:9001")
    }

    @Test("defaultBaseURL is equivalent to loopbackURL(port: PortSweep.defaultPort)")
    func default_equals_loopback_helper() {
        #expect(ChatStreamClient.defaultBaseURL
                == ChatStreamClient.loopbackURL(port: PortSweep.defaultPort))
    }
}
