import Foundation
import Testing
import Darwin
@testable import Rapid

/// Contract for ``PortAllocator`` — the v0.5.6 fix for "rapid-mlx
/// crashed mid-chat because vite/jupyter holds :8000."
///
/// The allocator is intentionally side-effecting (it shells out to
/// ``lsof`` via ``PortSweep`` and binds real sockets), so these tests
/// run against real ephemeral sockets on the test host rather than
/// against a mock. To keep the test isolated from whatever the test
/// host is actually doing, each case picks its own port window in
/// the ephemeral high range (60000+) so a developer running rapid-mlx
/// in another terminal can't false-fail us.
@Suite("PortAllocator — v0.5.6 fallback contract")
struct PortAllocatorTests {

    /// Pin a port by holding a real LISTEN socket on it. Returned
    /// closure releases. Tests use this to make a candidate port
    /// "look busy" to the allocator.
    private func pin(port: Int, host: String = "127.0.0.1") -> (() -> Void)? {
        let sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)
        guard sock >= 0 else { return nil }
        var yes: Int32 = 1
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &yes,
                   socklen_t(MemoryLayout<Int32>.size))
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = UInt16(port).bigEndian
        addr.sin_addr.s_addr = inet_addr(host)
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        let bindOk = withUnsafePointer(to: &addr) { ptr -> Int32 in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                Darwin.bind(sock, sockPtr,
                            socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bindOk == 0 else { close(sock); return nil }
        // ``listen`` makes the port appear as LISTEN to lsof — a
        // bare bind would still occupy the port at the bind-probe
        // layer, but lsof skips non-LISTEN sockets and we want the
        // sweep's lsof query to surface this pid (or skip it, since
        // it's not rapid-owned). LISTEN is the realistic shape.
        _ = Darwin.listen(sock, 1)
        return { close(sock) }
    }

    @Test("Empty candidate list returns nil")
    func emptyCandidatesReturnsNil() {
        #expect(PortAllocator.allocate(candidates: [], host: "127.0.0.1") == nil)
    }

    @Test("First free port wins")
    func firstFreeWins() throws {
        // 60000 should be available on any sane CI host — pick a
        // fresh window so a prior failed test's leaked socket
        // doesn't poison us.
        let result = PortAllocator.allocate(
            candidates: [60_001, 60_002, 60_003],
            host: "127.0.0.1"
        )
        #expect(result == 60_001)
    }

    @Test("Foreign-held port is skipped, next port wins")
    func foreignHeldSkips() throws {
        // Pin :60010 with our own LISTEN socket. Since it's not a
        // rapid-mlx process, ``PortSweep`` won't kill it, and the
        // allocator's bind-probe will fail → walks to :60011.
        guard let release = pin(port: 60_010) else {
            Issue.record("Could not pin :60010 to set up the test")
            return
        }
        defer { release() }
        let result = PortAllocator.allocate(
            candidates: [60_010, 60_011, 60_012],
            host: "127.0.0.1"
        )
        #expect(result == 60_011)
    }

    @Test("All candidates busy returns nil")
    func allBusyReturnsNil() throws {
        guard let r1 = pin(port: 60_020),
              let r2 = pin(port: 60_021)
        else {
            Issue.record("Could not pin :60020/60021 to set up the test")
            return
        }
        defer { r1(); r2() }
        let result = PortAllocator.allocate(
            candidates: [60_020, 60_021],
            host: "127.0.0.1"
        )
        #expect(result == nil)
    }

    @Test("canBind probes one port and reports honestly")
    func canBindHonest() throws {
        // Free port — canBind should report true.
        #expect(PortAllocator.canBind(port: 60_030, host: "127.0.0.1"))

        // Pinned port — canBind should report false.
        guard let release = pin(port: 60_031) else {
            Issue.record("Could not pin :60031 to set up the test")
            return
        }
        defer { release() }
        #expect(!PortAllocator.canBind(port: 60_031, host: "127.0.0.1"))
    }

    @Test("Default candidate window is 7659..7668 plus 8000..8009 legacy fallback")
    func defaultWindow() {
        // Primary window is the RMLX phone-keypad range; 8000…8009 is
        // probed after it so existing 127.0.0.1:8000 clients keep working.
        #expect(PortAllocator.candidatePorts == Array(7659...7668) + Array(8000...8009))
        #expect(PortAllocator.candidatePorts.count == 20)
    }

    // MARK: - #455 RAPID_DESKTOP_PORT override (test-harness isolation)

    @Test("RAPID_DESKTOP_PORT pins the candidate window to a single port")
    func envOverridePinsSinglePort() {
        let ports = PortAllocator.resolveCandidatePorts(
            environment: ["RAPID_DESKTOP_PORT": "8505"]
        )
        #expect(ports == [8505])
    }

    @Test("RAPID_DESKTOP_PORT surrounding whitespace is tolerated")
    func envOverrideTrimsWhitespace() {
        #expect(
            PortAllocator.resolveCandidatePorts(
                environment: ["RAPID_DESKTOP_PORT": "  9123 "]
            ) == [9123]
        )
    }

    @Test("Unset / invalid / out-of-range RAPID_DESKTOP_PORT falls back to the default window")
    func envOverrideFallsBackWhenInvalid() {
        let fallbacks: [[String: String]] = [
            [:],                                        // unset
            ["RAPID_DESKTOP_PORT": ""],                 // empty
            ["RAPID_DESKTOP_PORT": "not-a-number"],     // non-numeric
            ["RAPID_DESKTOP_PORT": "0"],                // below range
            ["RAPID_DESKTOP_PORT": "70000"],            // above 65535
            ["RAPID_DESKTOP_PORT": "-1"],               // negative
        ]
        for env in fallbacks {
            #expect(
                PortAllocator.resolveCandidatePorts(environment: env) == Array(7659...7668) + Array(8000...8009),
                "invalid override \(env) must fall back to the default window"
            )
        }
    }

    @Test("Invalid port and host inputs return false instead of trapping")
    func invalidProbeInputsReturnFalse() {
        #expect(!PortAllocator.canBind(port: -1, host: "127.0.0.1"))
        #expect(!PortAllocator.canBind(port: 70_000, host: "127.0.0.1"))
        #expect(!PortAllocator.canBind(port: 60_040, host: "not a host"))
    }
}

@Suite("PortSweep — port arg refactor (v0.5.6 prerequisite)")
struct PortSweepPortArgTests {

    @Test("Zero-arg sweep targets the default port")
    func zeroArgUsesDefault() {
        // The zero-arg form is just a one-line shim — the contract
        // we care about is that it routes to ``defaultPort``.
        // Behaviour-test by running it and asserting no crash; the
        // bigger ``isRapidOwnedCommand`` ownership semantics are
        // covered by ``PortSweepTests`` and unaffected here.
        _ = PortSweep.sweep()  // must not throw / crash
        #expect(PortSweep.defaultPort == 7659)
    }

    @Test("Back-compat shim: PortSweep.port == defaultPort")
    func backCompatShim() {
        // ``PortSweep.port`` is kept as a computed property so older
        // call sites that read ``PortSweep.port`` keep working
        // during the v0.5.6 migration window. The shim must point
        // at the same source of truth as ``defaultPort``.
        #expect(PortSweep.port == PortSweep.defaultPort)
    }
}
