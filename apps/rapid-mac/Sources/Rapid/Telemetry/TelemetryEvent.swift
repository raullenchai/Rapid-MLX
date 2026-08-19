import Foundation

/// Codable shape of one telemetry event.
///
/// Matches the rapid-mlx telemetry-worker contract verbatim
/// (``schema_version``, ``client_id``, ``session_id``,
/// ``rapid_mlx_version``, ``event``, ``timestamp``, ``platform``).
/// The Worker passes unknown top-level fields through, so we can
/// stash rapid-desktop specifics under ``extras`` and they land
/// in R2 alongside the validated core.
///
/// ``rapid_mlx_version`` is a misnomer for a rapid-desktop client —
/// the field name is fixed by the Worker schema. We populate it
/// with the rapid-desktop CFBundleShortVersionString and tag the
/// platform with ``app: "rapid-desktop"`` so analytics can split
/// the two clients without a schema migration.
struct TelemetryEvent: Codable, Sendable {
    enum EventKind: String, Codable, Sendable {
        case sessionStart = "session_start"
        case sessionEnd = "session_end"
        case error
    }

    var schema_version: Int
    var client_id: String
    var session_id: String
    var rapid_mlx_version: String
    var event: EventKind
    var timestamp: String
    var platform: Platform
    var error_type: String?
    var error_message: String?
    var stack_frames: [String]?
    var context: String?

    struct Platform: Codable, Sendable {
        var app: String
        var os: String
        var os_version: String
        var arch: String
        /// Apple Silicon chip brand (e.g. ``"Apple M4 Max"``), read via
        /// ``sysctlbyname("machdep.cpu.brand_string")``. Mirrors the
        /// engine's ``redact._read_chip_brand()`` so a desktop machine
        /// buckets into the SAME per-chip label the CLI reports, instead
        /// of being invisible behind the generic ``arch`` ("arm64").
        /// Legacy Intel Macs use the coarse fixed label ``"Intel"``.
        ///
        /// Optional on the wire (``encodeIfPresent`` omits it when nil)
        /// so this is a backward-compatible schema addition: an older
        /// telemetry-worker that doesn't read ``chip`` ignores the extra
        /// key, and a build that can't read the brand simply omits it.
        ///
        /// Analytics uses ``app == "rapid-desktop"`` as the authoritative
        /// surface discriminator; chip is hardware metadata, not identity.
        ///
        /// The ``= nil`` default keeps the synthesized memberwise
        /// initializer callable with the original four labels, so every
        /// existing ``Platform(app:os:os_version:arch:)`` call site (and
        /// the decode path) stays source-compatible.
        var chip: String? = nil
        /// Total physical RAM rounded to the nearest GB. Mirrors the
        /// engine's ``redact.bucket_memory_gb`` so desktop + CLI machines
        /// aggregate uniformly and exact byte counts can't fingerprint a
        /// machine. Never carries raw bytes. Optional/back-compat like
        /// ``chip`` above.
        var memory_gb: Int? = nil
    }

    /// Build a session_start (no error fields populated).
    static func sessionStart(
        version: String,
        platform: Platform
    ) -> TelemetryEvent {
        TelemetryEvent(
            schema_version: TelemetryConfig.schemaVersion,
            client_id: TelemetryIdentity.clientID(),
            session_id: TelemetryConfig.sessionID,
            rapid_mlx_version: version,
            event: .sessionStart,
            timestamp: ISO8601DateFormatter().string(from: Date()),
            platform: platform,
            error_type: nil,
            error_message: nil,
            stack_frames: nil,
            context: nil
        )
    }

    /// Build an error event from a crash marker the previous launch
    /// left behind.
    ///
    /// ``sessionID`` overrides the default (current-process)
    /// ``TelemetryConfig.sessionID`` so a flushed crash report is
    /// attributed to the launch that ACTUALLY crashed — not the
    /// launch that's now reporting it. The dashboard correlates
    /// errors against the prior ``session_start`` by ``session_id``;
    /// without the override every crash looks like it happened in
    /// the reporting launch, which masks repeat-crash signal and
    /// breaks per-version regression tracking when the user
    /// crashed on N-1 and upgraded to N before relaunching.
    ///
    /// Codex audit batch 8 finding T4 (P2): ``errorMessage``,
    /// ``stackFrames``, and ``context`` could carry user-identifying
    /// data — an ``NSException.reason`` that references a path under
    /// ``/Users/<name>/work/secret-project``, a stack frame with the
    /// user's home directory, or a context label that captures a
    /// model alias the user is fine-tuning. We redact at build time
    /// so the on-disk crash marker AND the wire envelope share the
    /// same scrubbed shape — easier audit boundary than scrubbing
    /// once on send.
    static func error(
        version: String,
        platform: Platform,
        errorType: String,
        errorMessage: String,
        stackFrames: [String],
        context: String?,
        sessionID: String? = nil
    ) -> TelemetryEvent {
        // Clamp `errorType` to the closed set — see
        // `allowedErrorTypes` for the rationale. A future caller
        // that passes a user-controlled string (e.g.
        // `NSException.name.rawValue`) would otherwise leak PII
        // AND blow up dashboard cardinality. Falling through to
        // `"unknown"` is louder than silently dropping the event
        // (the operator sees an `unknown` slice and can grep the
        // codebase for the offending call site).
        let clampedType = TelemetryEvent.allowedErrorTypes.contains(errorType)
            ? errorType
            : "unknown"
        return TelemetryEvent(
            schema_version: TelemetryConfig.schemaVersion,
            client_id: TelemetryIdentity.clientID(),
            session_id: sessionID ?? TelemetryConfig.sessionID,
            rapid_mlx_version: version,
            event: .error,
            timestamp: ISO8601DateFormatter().string(from: Date()),
            platform: platform,
            error_type: clampedType,
            error_message: TelemetryEvent.redact(errorMessage, cap: 512),
            stack_frames: stackFrames.prefix(30).map {
                TelemetryEvent.redact($0, cap: 256)
            },
            context: context.map { TelemetryEvent.redact($0, cap: 128) }
        )
    }

    /// Scrub the most common PII leak vectors from a free-form
    /// string before it lands in telemetry. Codex audit batch 8 T4
    /// + v1 audit P1 (TelemetryEvent — verify error events don't
    /// include full paths, model aliases, message snippets).
    ///
    /// Three layers, in order:
    ///
    ///   1. **Token scrub** — delegate to `LogScrubber.scrub` so a
    ///      stack frame or error message that captured an
    ///      Authorization header / `HF_TOKEN=` env line / etc.
    ///      surfaces with the value redacted. Bug-class shared
    ///      with the log drawer, so the audit boundary is one
    ///      function for the whole app.
    ///   2. **Path scrub** — strip the username segment from the
    ///      well-known home/tmp-shaped paths so a stack frame like
    ///      `/Users/raullen/work/secret-project/foo.swift` becomes
    ///      `/Users/<redacted>/work/secret-project/foo.swift`.
    ///      Covers: `/Users/<name>/`, `/home/<name>/`,
    ///      `/private/var/folders/<two>/<random>/` (macOS sandbox
    ///      temp dirs that contain the user's container ID).
    ///   3. **Length cap** — bound the field so a runaway message
    ///      can't smuggle large blobs through. Always last so the
    ///      `<redacted>` marker can't be sliced mid-token.
    ///
    /// Caller-side discipline still required for surfaces this
    /// can't infer: model alias names, prompt fragments, paths
    /// outside the well-known home shapes. Those should never
    /// reach an error_message in the first place — telemetry
    /// uses a closed set of message templates by convention.
    static func redact(_ s: String, cap: Int) -> String {
        var out = LogScrubber.scrub(s)
        // /Users/raullen/work/... → /Users/<redacted>/work/...
        out = out.replacingOccurrences(
            of: #"/Users/[^/\s]+/"#,
            with: "/Users/<redacted>/",
            options: .regularExpression
        )
        // Linux-style /home/<name>/... → /home/<redacted>/...
        out = out.replacingOccurrences(
            of: #"/home/[^/\s]+/"#,
            with: "/home/<redacted>/",
            options: .regularExpression
        )
        // macOS sandbox temp / Container paths frequently leak the
        // user's container ID, which is a per-Apple-ID hash that
        // can be cross-referenced. Strip both segments past
        // `/private/var/folders/`. The trailing terminator
        // `(?=/|\s|$)` matches a slash (path continues),
        // whitespace, OR end-of-string — pre-fix the regex
        // required `/` which silently leaked the container ID
        // when an `NSException.reason` ended mid-segment
        // (codex r1 BLOCKING-2).
        out = out.replacingOccurrences(
            of: #"/private/var/folders/[^/\s]+/[^/\s]+(?=/|\s|$)"#,
            with: "/private/var/folders/<redacted>/<redacted>",
            options: .regularExpression
        )
        // Same shape without the `/private` prefix — Foundation
        // sometimes returns paths via the canonical, non-private
        // form depending on the resolver.
        out = out.replacingOccurrences(
            of: #"/var/folders/[^/\s]+/[^/\s]+(?=/|\s|$)"#,
            with: "/var/folders/<redacted>/<redacted>",
            options: .regularExpression
        )
        if out.count > cap {
            out = String(out.prefix(cap)) + "…"
        }
        return out
    }

    /// Closed set of `error_type` values telemetry is allowed to
    /// emit. Telemetry's free-text fields run through `redact()`,
    /// but `error_type` is dashboard-faceted and so MUST stay a
    /// known enumeration — a future codepath that passes a
    /// user-controlled string here would (a) create high-
    /// cardinality dashboard noise and (b) leak per-event PII if
    /// the value was an exception name carrying a path or alias.
    ///
    /// Enforced inside `error(errorType:…)` — values outside the
    /// set are clamped to `"unknown"` rather than silently
    /// dropped so an operator who sees the `unknown` slice on
    /// the dashboard can grep the codebase for the offending
    /// call site. Adding a new error_type therefore requires a
    /// deliberate addition here, and the test below pins the
    /// set so a silent extension is impossible.
    static let allowedErrorTypes: Set<String> = [
        "unclean_shutdown",
        "uncaught_exception",
        "signal",
    ]
}
