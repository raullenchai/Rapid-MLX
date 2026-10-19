import AppKit
import Darwin
import Foundation
import RapidCrashHandler

/// Catches crashes that escape the Swift runtime and surfaces them
/// as ``error`` events on the next launch.
///
/// We cover three crash classes, in decreasing order of fidelity:
///
///   1. **Uncaught Objective-C exceptions** —
///      ``NSSetUncaughtExceptionHandler`` fires on the main thread
///      before AppKit aborts. Heap-safe; we capture the full reason
///      + symbolicated call stack.
///   2. **Fatal signals** (SIGABRT from ``fatalError`` / Swift
///      precondition / ``assert``, plus SIGSEGV / SIGBUS / SIGILL
///      / SIGFPE). Signal handlers run in an interrupted context
///      where heap allocation and most Foundation calls are
///      undefined — so we only ``write(2)`` a tiny pre-allocated
///      marker file and re-raise. The marker tells the next launch
///      "we crashed with signal N at time T."
///   3. **Unclean shutdowns** — a "started" marker is written
///      synchronously on launch and deleted on clean ``terminate``.
///      Anything still on disk next launch that doesn't match
///      classes 1 or 2 is reported as an unclean shutdown. False-
///      positive on force-quit / OS reboot is acceptable; the goal
///      is "we'd rather over-report than miss a v0.5.9-class
///      silent abort."
///
/// On next launch, ``flushPendingCrashReports`` reads the marker
/// directory, sends one ``error`` event per file, and deletes the
/// file on a successful POST. Files that fail to send stay on disk
/// for the launch after that — best-effort retry, no exponential
/// backoff.
///
/// **Signal-safety contract (issue #24).** The signal handler runs
/// in an interrupted context where malloc, stdio, NSObject, Swift
/// reference counting, and most Foundation calls are undefined. We
/// uphold the contract by:
///
///   * Installing handlers via ``sigaction(2)`` with a pure-C
///     handler from the ``RapidCrashHandler`` target (F1) — no
///     Swift trampoline, no captures, no thunks.
///   * Pre-allocating every byte the handler writes — the marker
///     path C string and the 5 per-signal JSON envelopes are
///     ``malloc``'d at install time and handed to the C arena as
///     raw ``UnsafePointer<CChar>`` + length (F2).
///   * Reading state through a pure-C extern struct, not Swift
///     statics — ``_swift_beginAccess`` is async-signal-unsafe.
///   * Guarding the rich-marker-already-written check with
///     ``volatile sig_atomic_t`` (F7) — the only race-safe type the
///     C standard guarantees for the writer/handler boundary.
enum CrashReporter {
    /// Per-launch marker file directory. Lives under Application
    /// Support so it survives crashes (unlike Library/Caches which
    /// macOS may evict). Each launch writes one file; clean exits
    /// remove it; crashes leave it behind for the next launch.
    static let markerDirectory: URL = {
        // #420: resolve through ApplicationSupportLocator so dogfood
        // / test instances launched with HOME override write crash
        // markers to THEIR isolated path, not the real user's
        // ~/Library/Application Support/Rapid/crash-markers/.
        // Pre-fix code used FileManager.urls(for:...) which goes
        // through NSSearchPathForDirectoriesInDomains and resolves
        // the LOGGED-IN USER's home via getpwuid regardless of
        // $HOME. Zero-impact for real installs (production never
        // overrides HOME); unblocks dogfood test isolation.
        let dir = ApplicationSupportLocator.applicationSupportRoot()
            .appendingPathComponent("crash-markers", isDirectory: true)
        // Codex audit batch 7 F9 (P3): marker files may carry the
        // process command-line context ("chat_send", "model_pull")
        // which leaks the user's workflow. Restrict the directory
        // to owner-only (0700) so other local users on a shared Mac
        // can't read them. The per-file 0600 is enforced inside the
        // signal handler's open(2) call.
        try? FileManager.default.createDirectory(
            at: dir,
            withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700]
        )
        try? FileManager.default.setAttributes(
            [.posixPermissions: 0o700],
            ofItemAtPath: dir.path
        )
        return dir
    }()

    /// The file this launch will populate. Held as a static so the
    /// signal handler can reach a pre-resolved path without
    /// calling ``URL`` APIs inside the signal context.
    private static let markerURL: URL = {
        markerDirectory.appendingPathComponent(
            "\(TelemetryConfig.sessionID).json",
            isDirectory: false
        )
    }()

    /// Context label stamped into every crash marker. Fixed at
    /// ``"startup"`` today — the former runtime setter was removed as
    /// dead code (no caller ever narrowed the context). Kept as a
    /// single source of truth for the marker's ``context`` field so a
    /// future re-introduction has one place to wire.
    private static let contextLabel: String = "startup"

    /// Install all hooks and write the launch marker. Idempotent;
    /// calling twice is a no-op.
    static func install() {
        guard !installed else { return }
        allocateSignalState()
        writeLaunchMarker()
        installExceptionHandler()
        installSignalHandlers()
        installed = true
    }

    /// Test-only entry point: clears the installed-once latch, frees
    /// the malloc'd buffers, and zeroes the C arena so a follow-up
    /// ``install`` call can rebuild from scratch. Production code
    /// never touches this.
    static func _resetForTesting() {
        installed = false
        deallocateSignalState()
        rapid_crash_arena_reset()
    }

    /// Test-only readback. Goes through the C-level
    /// ``rapid_crash_arena_snapshot`` helper which returns a by-value
    /// copy — the live extern struct is ``static`` in C so Swift 6
    /// strict-concurrency doesn't see it as shared mutable state.
    static var _state: SignalStateSnapshot {
        let snap = rapid_crash_arena_snapshot()
        return SignalStateSnapshot(
            installed: installed,
            markerPath: snap.marker_path.map { String(cString: $0) },
            detailedFlagValue: snap.detailed_flag,
            sigabrtLen: snap.sigabrt_len,
            sigsegvLen: snap.sigsegv_len,
            sigbusLen: snap.sigbus_len,
            sigillLen: snap.sigill_len,
            sigfpeLen: snap.sigfpe_len
        )
    }

    struct SignalStateSnapshot {
        let installed: Bool
        let markerPath: String?
        let detailedFlagValue: sig_atomic_t
        let sigabrtLen: Int
        let sigsegvLen: Int
        let sigbusLen: Int
        let sigillLen: Int
        let sigfpeLen: Int
    }

    nonisolated(unsafe) private static var installed = false

    // Swift owns the malloc'd buffers handed to the C arena. We keep
    // pointers here so test-only ``_resetForTesting`` can free them
    // before zeroing the arena; production never deallocates.
    nonisolated(unsafe) private static var ownedMarkerPath: UnsafeMutablePointer<CChar>? = nil
    nonisolated(unsafe) private static var ownedSigabrtBuf: UnsafeMutablePointer<CChar>? = nil
    nonisolated(unsafe) private static var ownedSigsegvBuf: UnsafeMutablePointer<CChar>? = nil
    nonisolated(unsafe) private static var ownedSigbusBuf: UnsafeMutablePointer<CChar>? = nil
    nonisolated(unsafe) private static var ownedSigillBuf: UnsafeMutablePointer<CChar>? = nil
    nonisolated(unsafe) private static var ownedSigfpeBuf: UnsafeMutablePointer<CChar>? = nil

    /// Called on clean shutdown by ``RapidApp`` so the next launch
    /// doesn't misclassify a graceful exit as an unclean one.
    static func recordCleanShutdown() {
        try? FileManager.default.removeItem(at: markerURL)
    }

    /// Internal hook the file-scope NSException handler uses to
    /// tell the signal handler "don't overwrite the rich marker I
    /// just wrote." Delegates to the C extern so the
    /// ``volatile sig_atomic_t`` store is the canonical race-safe
    /// shape (issue #24 F7).
    static func markDetailedMarkerWritten() {
        rapid_crash_mark_detailed_written()
    }

    /// Read marker files written by previous launches, build error
    /// events from them, POST in one batch, and delete on success.
    static func flushPendingCrashReports() async {
        let fm = FileManager.default
        let myFile = markerURL.lastPathComponent
        guard let files = try? fm.contentsOfDirectory(
            at: markerDirectory,
            includingPropertiesForKeys: nil
        ) else { return }
        let pending = files.filter {
            $0.pathExtension == "json" && $0.lastPathComponent != myFile
        }
        guard !pending.isEmpty else { return }
        let platform = TelemetryClient.currentPlatform()
        var events: [TelemetryEvent] = []
        for file in pending {
            guard let data = try? Data(contentsOf: file),
                  let marker = try? JSONDecoder().decode(CrashMarker.self, from: data)
            else {
                try? fm.removeItem(at: file)
                continue
            }
            // Attribute the crash to the launch that crashed, not
            // the launch reporting it. The dashboard correlates
            // ``error`` events back to the matching ``session_start``
            // by ``session_id``; passing the current process's
            // values would re-attribute every flushed crash to the
            // reporting launch, masking repeat-crash patterns and
            // breaking per-version regression tracking when the
            // user updated between crashes.
            events.append(.error(
                version: marker.version,
                platform: platform,
                errorType: marker.error_type,
                errorMessage: marker.error_message,
                stackFrames: marker.stack_frames ?? [],
                context: marker.context,
                sessionID: marker.session_id
            ))
        }
        let accepted = await TelemetryClient().sendBatch(events)
        // Only retire the files the Worker actually accepted. If the
        // batch failed (offline, 5xx, timeout) leave them on disk so
        // the next launch retries — losing the only copy of a crash
        // report in exactly the transient-network case where it
        // matters defeats the whole pipeline. ``sendBatch`` also
        // returns ``true`` on opt-out so the cleanup path runs and
        // doesn't accumulate markers forever for that user.
        guard accepted else { return }
        for file in pending {
            try? fm.removeItem(at: file)
        }
    }

    /// Drop reports created before the user made an explicit telemetry
    /// choice. This is called immediately before a first-time opt-in so
    /// accepting the disclosure cannot retroactively upload a crash
    /// from an earlier, non-consenting launch.
    static func discardPendingCrashReports() {
        let fm = FileManager.default
        let myFile = markerURL.lastPathComponent
        guard let files = try? fm.contentsOfDirectory(
            at: markerDirectory,
            includingPropertiesForKeys: nil
        ) else { return }
        for file in files where file.pathExtension == "json"
            && file.lastPathComponent != myFile {
            try? fm.removeItem(at: file)
        }
    }

    // MARK: - Pre-allocated signal-context state

    /// Build the C strings the C arena needs and hand them off.
    /// Allocation happens on the main thread at launch where the
    /// heap is healthy; the signal handler only reads, never
    /// allocates.
    private static func allocateSignalState() {
        let session = TelemetryConfig.sessionID
        let version = TelemetryClient.currentVersion()

        ownedMarkerPath = duplicateAsCString(markerURL.path, nulTerminated: true).buf

        let abrt = duplicateAsCString(envelopeJSON("SIGABRT", SIGABRT, sessionID: session, version: version), nulTerminated: false)
        let segv = duplicateAsCString(envelopeJSON("SIGSEGV", SIGSEGV, sessionID: session, version: version), nulTerminated: false)
        let bus  = duplicateAsCString(envelopeJSON("SIGBUS",  SIGBUS,  sessionID: session, version: version), nulTerminated: false)
        let ill  = duplicateAsCString(envelopeJSON("SIGILL",  SIGILL,  sessionID: session, version: version), nulTerminated: false)
        let fpe  = duplicateAsCString(envelopeJSON("SIGFPE",  SIGFPE,  sessionID: session, version: version), nulTerminated: false)

        ownedSigabrtBuf = abrt.buf
        ownedSigsegvBuf = segv.buf
        ownedSigbusBuf  = bus.buf
        ownedSigillBuf  = ill.buf
        ownedSigfpeBuf  = fpe.buf

        // Hand pointers to the C arena. The C side stores them as
        // plain ``extern struct`` fields the signal handler reads
        // without invoking ``_swift_beginAccess``.
        rapid_crash_arena_install(
            ownedMarkerPath,
            ownedSigabrtBuf, abrt.len,
            ownedSigsegvBuf, segv.len,
            ownedSigbusBuf,  bus.len,
            ownedSigillBuf,  ill.len,
            ownedSigfpeBuf,  fpe.len
        )
    }

    /// Test-only counterpart to ``allocateSignalState``. Frees the
    /// malloc'd buffers BEFORE zeroing the C arena (otherwise the
    /// arena forgets them and they leak forever).
    private static func deallocateSignalState() {
        ownedMarkerPath?.deallocate(); ownedMarkerPath = nil
        ownedSigabrtBuf?.deallocate(); ownedSigabrtBuf = nil
        ownedSigsegvBuf?.deallocate(); ownedSigsegvBuf = nil
        ownedSigbusBuf?.deallocate();  ownedSigbusBuf  = nil
        ownedSigillBuf?.deallocate();  ownedSigillBuf  = nil
        ownedSigfpeBuf?.deallocate();  ownedSigfpeBuf  = nil
    }

    /// Shape must match ``CrashMarker``'s required keys so
    /// ``flushPendingCrashReports`` decodes the file successfully
    /// instead of treating it as garbage.
    private static func envelopeJSON(
        _ name: String,
        _ sig: Int32,
        sessionID: String,
        version: String
    ) -> String {
        // None of session / version / name contain quotes or
        // backslashes, so plain interpolation is safe JSON.
        "{\"session_id\":\"\(sessionID)\",\"version\":\"\(version)\",\"error_type\":\"signal\",\"error_message\":\"crashed with \(name) (\(sig))\"}"
    }

    private static func duplicateAsCString(_ s: String, nulTerminated: Bool)
        -> (buf: UnsafeMutablePointer<CChar>, len: Int)
    {
        let bytes = Array(s.utf8)
        let capacity = nulTerminated ? bytes.count + 1 : bytes.count
        let ptr = UnsafeMutablePointer<CChar>.allocate(capacity: capacity)
        for i in 0..<bytes.count {
            ptr[i] = CChar(bitPattern: bytes[i])
        }
        if nulTerminated {
            ptr[bytes.count] = 0
        }
        return (ptr, bytes.count)
    }

    // MARK: - Marker format

    private static func writeLaunchMarker() {
        let marker = CrashMarker(
            session_id: TelemetryConfig.sessionID,
            version: TelemetryClient.currentVersion(),
            error_type: "unclean_shutdown",
            error_message: "Process exited without recording a clean shutdown.",
            stack_frames: nil,
            context: contextLabel
        )
        guard let data = try? JSONEncoder().encode(marker) else { return }
        // Keep marker bytes owner-only from creation.
        try? OwnerOnlyFile.writeAtomically(data, to: markerURL)
    }

    // MARK: - Exception handler

    private static func installExceptionHandler() {
        NSSetUncaughtExceptionHandler(crashReporterExceptionHook)
    }

    // MARK: - Signal handlers

    /// Install the C ``rapid_crash_signal_handler`` via
    /// ``sigaction(2)``. Issue #24 F1: this replaces the previous
    /// ``signal(sig) { sig in ... }`` Swift closure trampoline, which
    /// dispatched through a witness table and re-entered the Swift
    /// runtime — async-signal-unsafe. The C handler reads the C
    /// arena directly with no runtime calls.
    private static func installSignalHandlers() {
        var action = sigaction()
        action.__sigaction_u.__sa_handler = rapid_crash_signal_handler
        sigemptyset(&action.sa_mask)
        // SA_RESTART matches the previous BSD ``signal(2)`` behaviour
        // — concurrent slow syscalls don't EINTR-fail when SIGABRT
        // lands on another thread. SA_SIGINFO is intentionally NOT
        // set: the three-arg form is not async-signal-safe without
        // further care and we don't need siginfo.
        action.sa_flags = SA_RESTART
        for sig in [SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGFPE] {
            sigaction(sig, &action, nil)
        }
    }
}

/// Disk shape of a per-launch marker. Written on launch with
/// ``error_type == "unclean_shutdown"`` and overwritten by an
/// exception or signal handler if those fire first. Hoisted to
/// fileprivate so the ``@convention(c)`` exception hook below can
/// reference it without re-declaring the struct.
struct CrashMarker: Codable {
    var session_id: String
    var version: String
    var error_type: String
    var error_message: String
    var stack_frames: [String]?
    var context: String?
}

/// `@convention(c)` exception hook hoisted out of ``CrashReporter``.
/// ``NSSetUncaughtExceptionHandler`` takes a bare C function
/// pointer, which forbids context capture — so the function lives
/// at file scope and reaches into ``CrashReporter``'s static state
/// directly. We're on the main thread, AppKit hasn't aborted yet,
/// and the heap is intact, so full Foundation is safe.
private func crashReporterExceptionHook(_ ex: NSException) {
    let frames = ex.callStackSymbols
    let marker = CrashMarker(
        session_id: TelemetryConfig.sessionID,
        version: TelemetryClient.currentVersion(),
        error_type: "uncaught_exception",
        error_message: "\(ex.name.rawValue): \(ex.reason ?? "")",
        stack_frames: Array(frames.prefix(30)),
        context: nil
    )
    if let data = try? JSONEncoder().encode(marker) {
        let url = CrashReporter.markerDirectory.appendingPathComponent(
            "\(TelemetryConfig.sessionID).json",
            isDirectory: false
        )
        // Keep marker bytes owner-only from creation.
        try? OwnerOnlyFile.writeAtomically(data, to: url)
        // Tell the signal handler to leave the file alone. The
        // runtime will follow this write with objc_terminate ->
        // SIGABRT on the same thread; without this flag the signal
        // handler would O_TRUNC over the rich payload with the
        // generic "crashed with SIGABRT (6)" envelope and the next
        // launch would report a hollow error event.
        CrashReporter.markDetailedMarkerWritten()
    }
}
