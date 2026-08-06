import Foundation

/// Server-Side-Request-Forgery guard for the ``browse`` tool.
///
/// `browse` turns a model-supplied URL into an HTTP GET whose response body is
/// fed back to the model. Without a guard that is a pivot into the user's
/// private network: `http://169.254.169.254/…` (cloud/EC2 metadata),
/// `http://127.0.0.1:…` (local admin panels), `http://192.168.…` (LAN
/// devices). So every host the fetch would contact — the initial URL AND every
/// redirect hop — is resolved and checked against a blocklist of loopback,
/// link-local, private, CGNAT, and other special-use ranges (IPv4 and IPv6),
/// and only `http`/`https` are allowed.
///
/// Scope of the guarantee. This layer resolves the host and rejects the request
/// before it runs if ANY resolved address is private — for the initial URL and
/// for every redirect hop. That confines the *hostnames the model can name* to
/// ones whose DNS currently points at public space. It is deliberately NOT a
/// closed IP-confinement proof, because these residuals cannot be closed at
/// this layer:
///
///  1. DNS-rebind TOCTOU. We resolve and validate, then `URLSession` re-resolves
///     when it connects; a hostile resolver can hand us a public address and the
///     socket a private one. `URLSession` exposes no hook to pin the validated
///     address while preserving TLS SNI / certificate validation, so the pinned
///     connection can't be expressed here.
///  2. System / PAC HTTP proxies. If the OS is configured with a proxy, the
///     proxy resolves and connects on its own, on a path our local resolution
///     never sees — so a locally-public result does not prove the proxy's
///     connection is public.
///  3. Network-specific NAT64 prefixes. We unwrap the well-known (`64:ff9b::/96`)
///     and RFC 8215 local-use (`64:ff9b:1::/48`) prefixes to range-check the
///     embedded IPv4, but a site can deploy NAT64 under any prefix we can't know
///     without that network's config, so a private IPv4 tunnelled through a
///     custom NAT64 prefix reads as global IPv6 (see ``ParsedIP/nat64EmbeddedV4``).
///
/// These are acknowledged residuals, not covered by this guard. They are not
/// left bare, though: the ``BrowseApprovalStore`` gate makes the user see and
/// approve the exact host before the first request (and again on any
/// cross-origin redirect), so exploitation requires the user to approve browsing
/// an attacker-controlled domain; and the action is only a read-only GET whose
/// body returns to the model — no code executes, no request body is sent. A
/// fully IP-confined fetch would need a custom transport that connects to the
/// validated address while setting TLS SNI/Host to the original hostname; that
/// is out of scope for this tool and tracked as future hardening.
enum BrowseSSRFGuard {
    enum Rejection: Error, Equatable {
        case badURL
        case blockedScheme(String)
        case noHost
        case unresolvable(String)
        case blockedAddress(host: String, address: String)

        var message: String {
            switch self {
            case .badURL:
                return "not a valid absolute URL"
            case .blockedScheme(let s):
                return "scheme '\(s)' is not allowed (only http/https)"
            case .noHost:
                return "URL has no host"
            case .unresolvable(let h):
                return "could not resolve host '\(h)'"
            case .blockedAddress(let host, let address):
                return "host '\(host)' resolves to a private/loopback address (\(address)) and cannot be browsed"
            }
        }
    }

    static let allowedSchemes: Set<String> = ["http", "https"]

    /// Validate a URL's scheme + host end-to-end: the host is resolved to its
    /// concrete addresses (or parsed directly if it is an IP literal) and every
    /// address is range-checked. Throws ``Rejection`` on the first problem.
    static func validate(_ url: URL) async throws {
        guard let scheme = url.scheme?.lowercased() else { throw Rejection.badURL }
        guard allowedSchemes.contains(scheme) else { throw Rejection.blockedScheme(scheme) }
        guard let host = url.host, !host.isEmpty else { throw Rejection.noHost }

        // Strip an IPv6 literal's brackets (URL.host keeps them off, but be
        // defensive if a raw string sneaks through elsewhere).
        let bareHost = host.hasPrefix("[") && host.hasSuffix("]")
            ? String(host.dropFirst().dropLast())
            : host

        // An IP literal in the URL never touches DNS — validate it directly so
        // `http://127.0.0.1` / `http://[::1]` are caught before any request.
        if let literal = ParsedIP(bareHost) {
            if literal.isBlocked {
                throw Rejection.blockedAddress(host: host, address: literal.canonical)
            }
            return
        }

        let addresses = try await resolve(bareHost)
        guard !addresses.isEmpty else { throw Rejection.unresolvable(host) }
        for ip in addresses where ip.isBlocked {
            throw Rejection.blockedAddress(host: host, address: ip.canonical)
        }
    }

    /// Resolve a hostname to every A / AAAA address. Runs `getaddrinfo` off the
    /// cooperative pool (it blocks). Returns `[]` (→ ``unresolvable``) on
    /// lookup failure rather than throwing a raw errno.
    ///
    /// Hard-bounded by ``resolveTimeout``: ``getaddrinfo`` is a blocking syscall
    /// that neither Swift task cancellation nor a task-group race can interrupt
    /// (a task group would still await the blocked child). Without a bound, a
    /// host whose resolver stalls would pin the ``browse`` call — and the chat
    /// turn awaiting it, unresponsive to Stop — for the OS resolver's own
    /// timeout (~30 s). We run the lookup fire-and-forget and resume from
    /// whichever finishes first, the lookup or the deadline; the orphaned
    /// lookup completes on its own and its late result is dropped by the
    /// single-resume guard. Throws ``Rejection.unresolvable`` on timeout so the
    /// fetch fails closed.
    static let resolveTimeout: TimeInterval = 8

    static func resolve(_ host: String) async throws -> [ParsedIP] {
        let gate = SingleResume()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation {
                (continuation: CheckedContinuation<[ParsedIP], Error>) in
                gate.attach(continuation)
                DispatchQueue.global(qos: .userInitiated).async {
                    gate.resume(.success(blockingLookup(host)))
                }
                DispatchQueue.global().asyncAfter(deadline: .now() + resolveTimeout) {
                    gate.resume(.failure(Rejection.unresolvable(host)))
                }
            }
        } onCancel: {
            // Stop pressed mid-lookup: settle the caller immediately instead of
            // waiting out the deadline. The orphaned getaddrinfo finishes on its
            // own and its result is dropped by the single-resume guard.
            gate.resume(.failure(CancellationError()))
        }
    }

    /// Synchronous ``getaddrinfo`` — blocks the calling thread until the OS
    /// resolver returns. Only ever run on a background queue by ``resolve``.
    private static func blockingLookup(_ host: String) -> [ParsedIP] {
        var hints = addrinfo(
            ai_flags: 0,
            ai_family: AF_UNSPEC,
            ai_socktype: SOCK_STREAM,
            ai_protocol: IPPROTO_TCP,
            ai_addrlen: 0,
            ai_canonname: nil,
            ai_addr: nil,
            ai_next: nil
        )
        var result: UnsafeMutablePointer<addrinfo>?
        let status = getaddrinfo(host, nil, &hints, &result)
        guard status == 0, let first = result else {
            // NXDOMAIN / no address → treat as unresolvable, not a crash.
            return []
        }
        defer { freeaddrinfo(first) }
        var out: [ParsedIP] = []
        var node: UnsafeMutablePointer<addrinfo>? = first
        while let cur = node {
            if let sa = cur.pointee.ai_addr {
                if let ip = ParsedIP(sockaddr: sa) { out.append(ip) }
            }
            node = cur.pointee.ai_next
        }
        return out
    }
}

/// Single-resume bridge so ``getaddrinfo`` and the resolve deadline can race:
/// the first to fire resumes the continuation, the loser is dropped. The lock
/// makes the once-only transition safe across the two background threads.
private final class SingleResume: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<[ParsedIP], Error>?
    private var resumed = false
    private var pendingResult: Result<[ParsedIP], Error>?

    /// Store the continuation, or settle it right away if a resume (e.g. the
    /// cancellation handler) already fired before it was attached.
    func attach(_ continuation: CheckedContinuation<[ParsedIP], Error>) {
        lock.lock()
        if resumed {
            let pending = pendingResult
            lock.unlock()
            continuation.resume(with: pending ?? .failure(CancellationError()))
            return
        }
        self.continuation = continuation
        lock.unlock()
    }

    func resume(_ result: Result<[ParsedIP], Error>) {
        lock.lock()
        if resumed {
            lock.unlock()
            return
        }
        resumed = true
        pendingResult = result
        let continuation = self.continuation
        self.continuation = nil
        lock.unlock()
        continuation?.resume(with: result)
    }
}

/// A resolved IP address (v4 or v6) with the range checks the guard needs.
/// Parsed either from a textual literal or from a `sockaddr` `getaddrinfo`
/// returned. All comparisons work on the raw bytes so there is no string
/// canonicalisation to fool.
struct ParsedIP: Equatable {
    enum Family: Equatable { case v4, v6 }
    let family: Family
    /// 4 bytes for v4, 16 for v6 (network order / big-endian, as stored).
    let bytes: [UInt8]

    /// Parse a textual literal via `inet_pton` (no DNS). Returns nil if the
    /// string is not a valid IPv4 or IPv6 literal.
    init?(_ literal: String) {
        var v4 = in_addr()
        if literal.withCString({ inet_pton(AF_INET, $0, &v4) }) == 1 {
            // `s_addr` already holds the 4 address bytes in NETWORK order in
            // memory (a.b.c.d → [a, b, c, d]); read them as-is. (Applying
            // `.bigEndian` would byte-swap on a little-endian host and reverse
            // the octets — a silent, dangerous SSRF misclassification.)
            var netAddr = v4.s_addr
            self.family = .v4
            self.bytes = withUnsafeBytes(of: &netAddr) { Array($0) }
            return
        }
        var v6 = in6_addr()
        if literal.withCString({ inet_pton(AF_INET6, $0, &v6) }) == 1 {
            self.family = .v6
            self.bytes = withUnsafeBytes(of: &v6) { Array($0) }
            return
        }
        return nil
    }

    /// Extract the address bytes from a `sockaddr` (`AF_INET` / `AF_INET6`).
    init?(sockaddr sa: UnsafePointer<sockaddr>) {
        switch Int32(sa.pointee.sa_family) {
        case AF_INET:
            let bytes = sa.withMemoryRebound(to: sockaddr_in.self, capacity: 1) { sin -> [UInt8] in
                // s_addr is already network-order in memory (see the literal
                // init) — read the raw bytes without a byte-swap.
                var netAddr = sin.pointee.sin_addr.s_addr
                return withUnsafeBytes(of: &netAddr) { Array($0) }
            }
            self.family = .v4
            self.bytes = bytes
        case AF_INET6:
            let bytes = sa.withMemoryRebound(to: sockaddr_in6.self, capacity: 1) { sin6 -> [UInt8] in
                var addr = sin6.pointee.sin6_addr
                return withUnsafeBytes(of: &addr) { Array($0) }
            }
            self.family = .v6
            self.bytes = bytes
        default:
            return nil
        }
    }

    /// Human-readable form for error messages (best-effort; falls back to a
    /// byte dump). Never used for a security decision — the checks are on bytes.
    var canonical: String {
        switch family {
        case .v4:
            return bytes.map(String.init).joined(separator: ".")
        case .v6:
            var addr = in6_addr()
            withUnsafeMutableBytes(of: &addr) { raw in
                for (i, b) in bytes.enumerated() where i < 16 { raw[i] = b }
            }
            var buf = [CChar](repeating: 0, count: Int(INET6_ADDRSTRLEN))
            if inet_ntop(AF_INET6, &addr, &buf, socklen_t(INET6_ADDRSTRLEN)) != nil {
                let utf8 = buf.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
                return String(decoding: utf8, as: UTF8.self)
            }
            return bytes.map { String(format: "%02x", $0) }.joined()
        }
    }

    /// True when the address falls in any loopback / link-local / private /
    /// CGNAT / reserved / multicast range that a browse must never reach.
    var isBlocked: Bool {
        switch family {
        case .v4:
            return Self.isBlockedV4(bytes)
        case .v6:
            // IPv4-mapped (::ffff:a.b.c.d) and NAT64 carry a v4 address inside
            // the v6 word — a `::ffff:127.0.0.1` or a NAT64-encoded loopback
            // would otherwise slip past the v6 checks, so validate the embedded
            // v4 too.
            if isV4Mapped { return Self.isBlockedV4(Array(bytes[12..<16])) }
            if let embedded = nat64EmbeddedV4 { return Self.isBlockedV4(embedded) }
            return isBlockedV6
        }
    }

    private var isV4Mapped: Bool {
        // ::ffff:0:0/96 — first 10 bytes zero, bytes 10-11 == 0xFF.
        guard bytes.count == 16 else { return false }
        for i in 0..<10 where bytes[i] != 0 { return false }
        return bytes[10] == 0xFF && bytes[11] == 0xFF
    }

    /// Embedded IPv4 for the well-known NAT64 prefix (RFC 6052 `64:ff9b::/96`)
    /// and the RFC 8215 local-use prefix (`64:ff9b:1::/48`), per the RFC 6052
    /// bit layout. Other NETWORK-SPECIFIC NAT64 prefixes can't be recognised
    /// without that network's configuration; reaching one needs a NAT64 gateway
    /// present AND its prefix known to the model, so it is a documented residual
    /// (same class as DNS-rebind, see the type doc).
    private var nat64EmbeddedV4: [UInt8]? {
        guard bytes.count == 16 else { return nil }
        // WKP 64:ff9b::/96 → v4 in the low 32 bits (bytes 12-15).
        if Array(bytes[0..<12]) == [0x00, 0x64, 0xFF, 0x9B, 0, 0, 0, 0, 0, 0, 0, 0] {
            return Array(bytes[12..<16])
        }
        // RFC 8215 64:ff9b:1::/48 → /48 layout: v4 = bytes 6,7,9,10 (byte 8 is
        // the reserved "u" octet, skipped).
        if Array(bytes[0..<6]) == [0x00, 0x64, 0xFF, 0x9B, 0x00, 0x01] {
            return [bytes[6], bytes[7], bytes[9], bytes[10]]
        }
        return nil
    }

    static func isBlockedV4(_ b: [UInt8]) -> Bool {
        guard b.count == 4 else { return true }   // malformed → fail closed
        let a = b[0], c = b[1]
        // 0.0.0.0/8 — "this host"
        if a == 0 { return true }
        // 10.0.0.0/8 — private
        if a == 10 { return true }
        // 100.64.0.0/10 — CGNAT
        if a == 100 && (c & 0xC0) == 0x40 { return true }
        // 127.0.0.0/8 — loopback
        if a == 127 { return true }
        // 169.254.0.0/16 — link-local (incl. 169.254.169.254 metadata)
        if a == 169 && c == 254 { return true }
        // 172.16.0.0/12 — private
        if a == 172 && (c & 0xF0) == 0x10 { return true }
        // 192.0.0.0/24 (IETF protocol assignments) + 192.0.2.0/24 (TEST-NET-1)
        if a == 192 && c == 0 && (b[2] == 0 || b[2] == 2) { return true }
        // 192.168.0.0/16 — private
        if a == 192 && c == 168 { return true }
        // 198.18.0.0/15 — benchmarking
        if a == 198 && (c == 18 || c == 19) { return true }
        // 198.51.100.0/24 + 203.0.113.0/24 — TEST-NET-2 / TEST-NET-3
        if a == 198 && c == 51 && b[2] == 100 { return true }
        if a == 203 && c == 0 && b[2] == 113 { return true }
        // 224.0.0.0/4 multicast + 240.0.0.0/4 reserved (covers 255.255.255.255)
        if a >= 224 { return true }
        return false
    }

    private var isBlockedV6: Bool {
        guard bytes.count == 16 else { return true }   // malformed → fail closed
        // ::/128 unspecified and ::1/128 loopback
        if bytes[0..<15].allSatisfy({ $0 == 0 }) && (bytes[15] == 0 || bytes[15] == 1) {
            return true
        }
        // fc00::/7 — unique local addresses (fc00::/8 + fd00::/8)
        if (bytes[0] & 0xFE) == 0xFC { return true }
        // fe00::/8 — link-local (fe80::/10) + deprecated site-local (fec0::/10,
        // which the earlier fe80-only check let through) + reserved. No global
        // unicast lives here (that is 2000::/3), so block the whole /8.
        if bytes[0] == 0xFE { return true }
        // ff00::/8 — multicast
        if bytes[0] == 0xFF { return true }
        return false
    }
}
