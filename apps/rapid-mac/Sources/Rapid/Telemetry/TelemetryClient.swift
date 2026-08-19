import Foundation

/// Async sender for ``TelemetryEvent`` batches.
///
/// Fire-and-forget: callers don't await the response and don't get
/// surfaced an error. The Worker's contract is "best effort"; a
/// dropped event is preferable to blocking the UI on a slow network.
///
/// Opt-out check happens here, so call sites don't have to. If
/// ``TelemetryConfig.isEnabled`` is ``false`` we drop the batch on
/// the floor — no network call, no retry, no log line.
///
/// ``@unchecked Sendable`` because both stored properties are
/// thread-safe reference types with value-copy semantics and no
/// mutable shared state: ``URLSession`` is already ``Sendable`` and
/// ``UserDefaults`` is documented thread-safe but not yet marked
/// ``Sendable`` by Foundation (hence ``@unchecked``, mirroring
/// ``NoRedirectDelegate`` below). Concurrent reads of either from two
/// copies of this struct are safe. Marking it explicitly lets a
/// caller inject an actor-isolated ``defaults`` (e.g. a ``@MainActor``
/// test's private suite) and still `await client.sendBatch(...)` on
/// the nonisolated method without tripping Swift 6 region-based
/// sending diagnostics (issue #530).
struct TelemetryClient: @unchecked Sendable {
    /// Codex audit batch 8 finding T1 (P2): a 307/308 from
    /// ``telemetry.rapidmlx.com`` would replay the POST body to an
    /// arbitrary host with ``URLSession.shared``'s default redirect
    /// policy. Use a dedicated session whose delegate rejects every
    /// redirect — the endpoint is hard-coded in ``TelemetryConfig``,
    /// so a redirect can only ever be an attack or a misconfigured
    /// edge node, neither of which we want to honour with user data.
    var session: URLSession = TelemetryClient.noRedirectSession

    /// Store the opt-out check reads. Defaults to ``.standard`` (the
    /// real per-install prefs) so product construction is unchanged;
    /// tests inject a private ``UserDefaults(suiteName:)`` so the
    /// opt-out flag can't be raced by a sibling test touching the
    /// shared ``.standard`` domain (issue #530). Mirrors the
    /// injectable ``session`` above.
    var defaults: UserDefaults = .standard

    /// Hard cap on a single POST body. The Worker rejects > 256 KB so
    /// anything larger than that is guaranteed-fail traffic. We cap
    /// at 200 KB to leave room for upstream framing and Cloudflare
    /// metadata. Codex audit batch 8 T2 (P2).
    static let maxBodyBytes = 200 * 1024

    /// Dedicated ``URLSession`` for telemetry POSTs. Built once.
    /// Configured to:
    ///   * Refuse to follow HTTP redirects (delegate denies all).
    ///   * Disable HTTP cookies — telemetry is auth-less and a
    ///     cookie would just be passive tracking surface.
    ///   * Use ephemeral storage so we never accumulate cache
    ///     entries that could be read back by another process.
    static let noRedirectSession: URLSession = {
        let cfg = URLSessionConfiguration.ephemeral
        cfg.httpCookieAcceptPolicy = .never
        cfg.httpShouldSetCookies = false
        cfg.httpAdditionalHeaders = ["User-Agent": "rapid-desktop-telemetry/1"]
        return URLSession(
            configuration: cfg,
            delegate: NoRedirectDelegate(),
            delegateQueue: nil
        )
    }()

    /// POSTs the event as a single-event payload
    /// (``{"batch":[<event>]}``). The Worker accepts both single-
    /// event and batched shapes; we always wrap in batch so the
    /// payload stays the same on the wire when we add real
    /// batching later.
    func send(_ event: TelemetryEvent) async {
        guard TelemetryConfig.isEnabled(defaults: defaults) else { return }
        let envelope: [String: [TelemetryEvent]] = ["batch": [event]]
        guard let body = try? JSONEncoder().encode(envelope) else { return }
        // Codex audit batch 8 T2 (P2): single oversized error event
        // would exceed the Worker's 256 KB cap. Drop instead of
        // wasting a network round-trip for a guaranteed 413.
        guard body.count <= TelemetryClient.maxBodyBytes else { return }
        var req = URLRequest(url: TelemetryConfig.endpoint)
        req.httpMethod = "POST"
        req.timeoutInterval = 5
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = body
        _ = try? await session.data(for: req)
    }

    /// Flush multiple events in one request. Used at launch to
    /// drain any crash markers left by the previous run — sending
    /// them one by one would multiply the network round-trip when
    /// a single batch fits well under the Worker's 256 KB cap.
    ///
    /// Returns ``true`` iff the Worker accepted the batch (2xx
    /// status). Caller uses the return value to decide whether to
    /// retire the on-disk crash markers or keep them for next
    /// launch — a network error or 5xx must NOT delete the only
    /// copy of the report.
    ///
    /// Opt-out + empty input both return ``true`` so the caller's
    /// cleanup path runs (there is nothing meaningful to retry).
    @discardableResult
    func sendBatch(_ events: [TelemetryEvent]) async -> Bool {
        guard TelemetryConfig.isEnabled(defaults: defaults) else { return true }
        guard !events.isEmpty else { return true }
        let envelope: [String: [TelemetryEvent]] = ["batch": events]
        guard let body = try? JSONEncoder().encode(envelope) else { return false }
        // Codex audit batch 8 T2 (P2): refuse oversized batches.
        // Returning ``true`` lets the caller retire the markers —
        // the batch would 413 forever; better to drop than retry.
        guard body.count <= TelemetryClient.maxBodyBytes else { return true }
        var req = URLRequest(url: TelemetryConfig.endpoint)
        req.httpMethod = "POST"
        req.timeoutInterval = 8
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = body
        do {
            let (_, response) = try await session.data(for: req)
            guard let http = response as? HTTPURLResponse else { return false }
            let code = http.statusCode
            if (200..<300).contains(code) {
                return true
            }
            // Codex audit batch 8 T3 (P2): classify response. 4xx
            // means the server permanently rejected the payload —
            // retrying every launch with the same marker just burns
            // bandwidth and re-pollutes Worker logs with the same
            // failed schema validation. Treat 4xx as "delete on
            // disk" by returning ``true``. 5xx and transport errors
            // (catch branch below) stay retryable.
            if (400..<500).contains(code) {
                return true
            }
            return false
        } catch {
            return false
        }
    }

    /// Build the ``platform`` field from the current process'
    /// macOS + arch + bundle metadata. Pulled out so tests can
    /// build a deterministic ``Platform`` without invoking the
    /// real ``ProcessInfo`` / ``Bundle``.
    static func currentPlatform() -> TelemetryEvent.Platform {
        let os = ProcessInfo.processInfo.operatingSystemVersion
        let osStr = "\(os.majorVersion).\(os.minorVersion).\(os.patchVersion)"
        #if arch(arm64)
        let arch = "arm64"
        #elseif arch(x86_64)
        let arch = "x86_64"
        #else
        let arch = "unknown"
        #endif
        return TelemetryEvent.Platform(
            app: "rapid-desktop",
            os: "macos",
            os_version: osStr,
            arch: arch,
            chip: chipBrand(),
            memory_gb: bucketMemoryGB(totalMemoryBytes())
        )
    }

    /// Apple Silicon chip brand — e.g. ``"Apple M4 Max"`` — read via
    /// ``sysctlbyname("machdep.cpu.brand_string", …)``. This is the
    /// exact same sysctl key the engine's
    /// ``redact._read_chip_brand()`` shells out to
    /// (``sysctl -n machdep.cpu.brand_string``), so the two clients
    /// produce byte-identical Apple Silicon brand strings and bucket into the
    /// same per-chip analytics label. Intel is deliberately reduced to the
    /// coarse label ``"Intel"`` because its raw brand includes detailed SKU
    /// and clock information. Whitespace-trimmed to match the engine's
    /// ``.strip()``; returns ``nil`` (field omitted on the
    /// wire) if the value is unreadable or empty rather than shipping
    /// a placeholder that would pollute the chip breakdown.
    static func chipBrand() -> String? {
        #if arch(x86_64)
        // Intel's brand string includes the exact CPU SKU and clock speed,
        // which is substantially more identifying than an Apple chip family.
        // Desktop analytics only needs a coarse legacy-Mac bucket.
        return "Intel"
        #else
        var size = 0
        // First call sizes the buffer; a failure or zero length means
        // the key is unavailable (should not happen on macOS, but we
        // degrade to "field absent" rather than trap).
        guard sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0,
              size > 0 else {
            return nil
        }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0) == 0 else {
            return nil
        }
        // Do not rely on `String(cString:)`: although this sysctl normally
        // includes a trailing NUL, its contract is the returned byte count.
        // Decode only the bytes written and reject malformed UTF-8.
        let bytes = buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        guard let decoded = String(bytes: bytes, encoding: .utf8) else {
            return nil
        }
        let brand = decoded.trimmingCharacters(in: .whitespacesAndNewlines)
        return brand.isEmpty ? nil : brand
        #endif
    }

    /// Total physical RAM in bytes. ``physicalMemory`` is the same
    /// quantity the engine reads via ``psutil.virtual_memory().total``.
    static func totalMemoryBytes() -> UInt64 {
        ProcessInfo.processInfo.physicalMemory
    }

    /// Round a byte count to the nearest GB (GiB, 1024³), returning 0 for an
    /// empty reading. A faithful port of the engine's
    /// ``redact.bucket_memory_gb`` — coarse tiers so exact byte counts
    /// can't fingerprint a machine, and the two telemetry sources land
    /// on identical integers for the same hardware. Uses
    /// round-half-to-even (``.toNearestOrEven``) to mirror Python's
    /// ``round()`` semantics exactly; in practice Mac RAM configs are
    /// whole GiB multiples so the tie-break never fires, but matching
    /// it keeps the two implementations provably equivalent.
    static func bucketMemoryGB(_ bytes: UInt64) -> Int {
        guard bytes > 0 else { return 0 }
        let gib = 1024.0 * 1024.0 * 1024.0
        return Int((Double(bytes) / gib).rounded(.toNearestOrEven))
    }

    /// Current bundle short-version. Falls back to ``"0.0.0"`` so a
    /// dev ``swift run`` (no Info.plist) still produces a valid
    /// event rather than crashing on a force-unwrap.
    static func currentVersion() -> String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String
            ?? "0.0.0"
    }
}

/// Codex audit batch 8 T1 (P2): refuse to follow redirects on the
/// telemetry POST. Returning ``nil`` from ``willPerformHTTPRedirection``
/// tells ``URLSession`` to deliver the redirect response to the caller
/// as-is rather than replaying the request body to the new location.
final class NoRedirectDelegate: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        completionHandler(nil)
    }
}
