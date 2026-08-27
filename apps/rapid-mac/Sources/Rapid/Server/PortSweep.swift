import Foundation

/// Clears port 8000 of leftover `rapid-mlx` processes from previous Rapid
/// sessions that exited uncleanly (Force Quit, crash, SIGKILL). Without
/// this, a fresh launch would spawn a second backend that races the
/// orphan for the port and pins another 4-80 GB of model weights in
/// memory on top of the first.
///
/// Mirrors `sweep_port_8000` in the v0.1 Tauri reference
/// (`src-tauri/src/server.rs`), which is the v0.1.13 cross-session
/// orphan fix.
///
/// Strategy: `netstat -anv -p tcp` lists every TCP listener without walking
/// mounted filesystems, we select the pids holding the target port, then send
/// each one SIGTERM, wait 1.5 s for graceful uvicorn shutdown, then
/// SIGKILL anyone still alive. A developer running rapid-mlx in their
/// own terminal can opt out via `RAPID_DESKTOP_NO_PORT_SWEEP=1`.
enum PortSweep {
    /// Filesystem-independent listener inventory. Exposed internally so the
    /// removable-volume regression test pins the executable as well as the
    /// output parser; replacing this with a file-descriptor scanner would
    /// reintroduce the picker-time mount traversal fixed in #2343.
    static let socketTableProbeExecutable = URL(fileURLWithPath: "/usr/sbin/netstat")
    static let socketTableProbeArguments = ["-anv", "-p", "tcp"]

    /// The opportunistic launch-time sweep, so a spawn can wait it out
    /// instead of racing it. Detached at launch (never blocking the UI); a
    /// ``ServerManager.start`` that lands while it is still running awaits it
    /// via ``awaitLaunchSweep`` rather than allocating a port the sweep is
    /// about to clear — otherwise an auto-start can spawn onto the candidate
    /// port before the sweep's ``lsof`` snapshot, and the sweep then reaps the
    /// server this launch just started.
    @MainActor private static var launchSweep: Task<Void, Never>?

    /// Kick off the launch sweep. Idempotent: a second call is a no-op.
    @MainActor static func startLaunchSweep(port: Int) {
        guard launchSweep == nil else { return }
        launchSweep = Task.detached(priority: .userInitiated) {
            sweep(port: port)
        }
    }

    /// Await the launch sweep if one is in flight. Returns immediately when
    /// none was started or it already finished.
    @MainActor static func awaitLaunchSweep() async {
        await launchSweep?.value
    }

    /// The default port rapid-mlx serve binds on. ``PortAllocator``
    /// walks ``defaultPort … defaultPort + 9`` when the default port
    /// is held by a foreign process. ``sweep()`` (no arg) targets the
    /// default port so legacy single-port callers stay one-line.
    static let defaultPort: Int = 8000

    /// Back-compat shim — older call sites reach for ``PortSweep.port``.
    /// New code should prefer ``ServerManager.activePort`` (the actual
    /// port the running rapid-mlx is bound to) when wiring URLs.
    static var port: Int { defaultPort }

    /// Upper bound on how long we wait for a SIGTERM'd process (group)
    /// to exit before escalating to SIGKILL. Unchanged from the
    /// original fixed ``Thread.sleep(1.5)`` — what changed is that we
    /// now POLL toward this deadline instead of always burning it.
    private static let sigtermGrace: TimeInterval = 1.5

    /// Poll granularity while waiting for death. 50 ms is far below
    /// human perception yet coarse enough that the ``kill(_:0)`` probes
    /// cost nothing measurable.
    private static let deathPollInterval: TimeInterval = 0.05

    /// Block until ``isAlive`` reports false or ``grace`` elapses.
    /// Returns true when the target actually died inside the window.
    ///
    /// Replaces an unconditional ``Thread.sleep(forTimeInterval: 1.5)``
    /// after each SIGTERM. That sleep burned the FULL 1.5 s even when
    /// uvicorn exited in 80 ms, and ``PortAllocator.allocate()`` runs
    /// this sweep once per candidate port (8000…8009) — so a machine
    /// with several stale ports paid 1.5 s each, serially, for
    /// processes that were already gone. Polling keeps the identical
    /// worst-case bound while collapsing the common case to roughly
    /// one poll interval.
    @discardableResult
    private static func awaitDeath(
        within grace: TimeInterval,
        isAlive: () -> Bool
    ) -> Bool {
        let deadline = Date().addingTimeInterval(grace)
        while Date() < deadline {
            if !isAlive() { return true }
            Thread.sleep(forTimeInterval: deathPollInterval)
        }
        return !isAlive()
    }

    /// ``kill(-pgid, 0)`` probes process-group liveness without
    /// signalling. Errno ``EPERM`` means "the group exists but we lack
    /// permission to signal it" — that is very much alive, so it must
    /// NOT be read as death. Only ``ESRCH`` (no such group) is death.
    private static func processGroupIsAlive(_ pgid: Int32) -> Bool {
        if kill(-pgid, 0) == 0 { return true }
        return errno == EPERM
    }

    /// Single-pid counterpart to ``processGroupIsAlive``. Same EPERM
    /// caveat applies.
    private static func pidIsAlive(_ pid: Int32) -> Bool {
        if kill(pid, 0) == 0 { return true }
        return errno == EPERM
    }

    /// Sweep ``defaultPort`` — kept as a zero-arg overload for the
    /// app's launch hook which always runs against the canonical
    /// rapid-mlx port.
    @discardableResult
    static func sweep() -> [Int32] {
        sweep(port: defaultPort)
    }

    /// Returns the list of pids that were touched (before the sweep
    /// started). Callers can surface that in the UI if useful; the
    /// current shell does not.
    @discardableResult
    static func sweep(port: Int) -> [Int32] {
        if ProcessInfo.processInfo.environment["RAPID_DESKTOP_NO_PORT_SWEEP"] != nil {
            return []
        }
        let raw = pidsOnPort(port: port)
        // codex r1 (#20): if NOTHING is listening on the swept port
        // but a record still references it, our owned child exited
        // after persisting the record (crash, hard kill, app died
        // before clear hooks fired). Drop the stale record now so
        // a kernel PID-reuse to an unrelated process later can't be
        // matched against this dead record on a future launch.
        guard !raw.isEmpty else {
            if let stale = OwnedServerRecord.load(), stale.port == port {
                FileHandle.standardError.write(Data(
                    "[server] sweep_port_\(port): clearing stale ownership record (pid \(stale.pid), pgid \(stale.pgid), alias \(stale.alias)) — port has no listener\n".utf8
                ))
                OwnedServerRecord.clear()
            }
            return []
        }

        // #20: prefer the persisted ``OwnedServerRecord`` over the
        // basename heuristic when we have one. The record was written
        // at spawn time by ``ServerManager`` and only persists when
        // we genuinely owned a child; a matching record + still-alive
        // PID-on-port is an authoritative ownership signal and lets
        // us PGID-kill the whole uvicorn tree precisely without
        // touching anything else lsof returns.
        //
        // Stale-record handling: if the record exists but the recorded
        // PID is NOT in the lsof set (process died and the port is
        // now held by something else), discard the record so the next
        // launch doesn't trust it again. Fall through to the existing
        // heuristic — the record is an upgrade, not a replacement.
        if let record = OwnedServerRecord.load(), record.port == port {
            if raw.contains(record.pid) {
                // Codex r2 (#20 P2): PID reuse race. macOS recycles
                // PIDs aggressively — if our owned-server.json sat on
                // disk across a long uptime (Sleep + Wake, hibernate),
                // the recorded PID could now belong to an unrelated
                // user process that just happens to be listening on
                // the same port AND be a process-group leader. Blindly
                // PGID-killing here would clobber that user's tree.
                //
                // Verify identity via the same command-line predicate
                // the heuristic path uses before signalling. If the
                // command matches our owned shape (rapid-mlx serve OR
                // python -m vllm_mlx serve) the recorded pgid is
                // trustworthy; if not, drop the record and fall
                // through to the existing per-PID heuristic which
                // will skip the foreign process explicitly.
                let identityVerified = recordIdentityHolds(record)
                if identityVerified {
                    FileHandle.standardError.write(Data(
                        "[server] sweep_port_\(port): PGID-kill \(record.pgid) (recorded pid \(record.pid), alias \(record.alias))\n".utf8
                    ))
                    _ = kill(-record.pgid, SIGTERM)
                    // Poll toward the grace deadline rather than
                    // always sleeping through it — see ``awaitDeath``.
                    awaitDeath(within: sigtermGrace) {
                        processGroupIsAlive(record.pgid)
                    }
                    if processGroupIsAlive(record.pgid) {
                        _ = kill(-record.pgid, SIGKILL)
                    }
                    OwnedServerRecord.clear()
                    return [record.pid]
                }
                // Identity check failed — the recorded PID now belongs
                // to a foreign process (or `ps` failed to read it).
                // Drop the stale record and let the heuristic path
                // below verify each pid individually.
                let cmdSummary = processCommand(pid: record.pid)?.prefix(60) ?? "(nil)"
                FileHandle.standardError.write(Data(
                    "[server] sweep_port_\(port): record pid \(record.pid) failed identity check (cmd=\(cmdSummary)), discarding record + falling through to heuristic\n".utf8
                ))
                OwnedServerRecord.clear()
            } else {
                // Record refers to a port we're sweeping but the PID
                // is gone — stale. Discard so we don't keep retrying
                // every launch.
                OwnedServerRecord.clear()
            }
        }

        // Codex round-1 finding: ownership-verify each pid BEFORE
        // signalling. The previous version blindly SIGTERM/SIGKILL'd
        // every pid `lsof` returned, which would clobber a user's
        // unrelated dev server on :8000.
        var owned: [Int32] = []
        var skipped: [(Int32, String)] = []
        for pid in raw {
            let cmd = processCommand(pid: pid)
            if let cmd, isRapidOwnedCommand(cmd) {
                owned.append(pid)
            } else {
                skipped.append((pid, cmd ?? "(unknown)"))
            }
        }
        if !skipped.isEmpty {
            FileHandle.standardError.write(Data(
                "[server] sweep_port_\(port): skipping non-rapid pids: \(skipped.map { "\($0.0)(\($0.1.prefix(60)))" })\n".utf8
            ))
        }
        guard !owned.isEmpty else { return [] }

        FileHandle.standardError.write(Data(
            "[server] sweep_port_\(port): terminating \(owned.count) rapid-mlx orphan(s): \(owned)\n".utf8
        ))
        for pid in owned {
            _ = kill(pid, SIGTERM)
        }
        // Give uvicorn a moment for graceful shutdown; observed up to ~1 s
        // in the wild. Polls instead of always burning the full grace —
        // the overwhelmingly common case is "already gone in <100 ms",
        // and this sweep runs once PER candidate port on the start path.
        awaitDeath(within: sigtermGrace) {
            owned.contains(where: pidIsAlive)
        }

        // Codex round-2 finding: the previous SIGKILL pass took a
        // FRESH ``lsof`` snapshot and SIGKILL'd anything that looked
        // rapid-owned. If the original process exited during the 1.5 s
        // grace AND a NEW rapid-mlx process bound :8000 in that
        // window (e.g. the user's own ``rapid-mlx serve`` in a
        // terminal), we'd kill the wrong process. Intersect with the
        // pids we actually SIGTERM'd so SIGKILL only targets the
        // exact survivors of the first pass.
        let ownedSet = Set(owned)
        let survivorsRaw = pidsOnPort(port: port)
        let survivors = survivorsRaw.filter { ownedSet.contains($0) }
        if !survivors.isEmpty {
            FileHandle.standardError.write(Data(
                "[server] sweep_port_\(port): SIGKILL \(survivors.count) survivor(s): \(survivors)\n".utf8
            ))
            for pid in survivors {
                _ = kill(pid, SIGKILL)
            }
        }
        return owned
    }

    /// Ask the kernel's socket table for TCP listeners, then select the exact
    /// local port. Do not use `lsof` here: even an internet-socket-filtered
    /// invocation walks the mount table and stats mounted filesystems. A model
    /// switch can call this once per candidate port, so an unrelated mounted
    /// installer or external disk was touched merely by opening the picker and
    /// could trigger macOS removable-volume consent (#2343).
    ///
    /// `netstat` reads the networking table directly. `-n` disables name
    /// resolution, `-a` includes listeners, `-v` includes the owning
    /// `process:pid` field, and `-p tcp` excludes unrelated protocols. The
    /// ownership check below remains authoritative before any signal is sent.
    private static func pidsOnPort(port: Int) -> [Int32] {
        let task = Process()
        task.executableURL = socketTableProbeExecutable
        task.arguments = socketTableProbeArguments
        guard let data = runCapturingStdout(task) else { return [] }
        return parseListeningPIDs(data, port: port)
    }

    /// Run ``task`` to completion and return its stdout bytes, draining
    /// stdout AND stderr concurrently. Returns nil only when the spawn
    /// itself threw (missing binary) so callers can distinguish "couldn't
    /// launch" from "launched and produced nothing".
    ///
    /// The concurrent drain is the load-bearing part. The previous shape
    /// at all three call sites here was:
    ///
    ///     task.standardError = Pipe()   // created, NEVER read
    ///     task.waitUntilExit()          // <- blocks forever
    ///     drainPipeToEOF(stdout...)     // <- never reached
    ///
    /// A pipe holds ~64 KiB. Once a child fills an undrained pipe it
    /// blocks in ``write(2)`` and never exits, while the parent blocks in
    /// ``waitUntilExit()`` waiting for that exit — a textbook deadlock
    /// with no timeout on either side. ``lsof`` reaches that volume on
    /// machines with stale network mounts (a ``WARNING: can't stat()``
    /// line per unreachable filesystem); reading stdout only AFTER the
    /// wait means even stdout alone can hang on a long pid list.
    ///
    /// This mattered more than a stuck helper: ``PortSweep`` runs from
    /// ``AppDelegate`` init and from the @MainActor start path, so the
    /// deadlock froze the whole app at launch with no way out.
    /// Verified: a repro of the old shape had to be SIGKILLed at 10 s.
    ///
    /// Both handles are drained on concurrent queues so neither pipe can
    /// backpressure the child, and ``waitUntilExit`` is called only after
    /// both reach EOF — which the child's exit guarantees, since exit
    /// closes both write ends.
    private static func runCapturingStdout(_ task: Process) -> Data? {
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        task.standardOutput = stdoutPipe
        task.standardError = stderrPipe
        do {
            try task.run()
        } catch {
            return nil
        }

        // Read both ends concurrently. ``readToEndSafely`` degrades a
        // bad descriptor to empty Data rather than raising the
        // uncatchable NSException — see ``drainPipeToEOF``.
        //
        // stderr goes to a background queue and its bytes are DISCARDED
        // — we drain it purely so the child can never block writing to
        // a full pipe. stdout is then read on THIS thread, which both
        // pipes being drained in parallel is all the deadlock fix
        // requires. Keeping the captured result on the stack (rather
        // than a shared `var` written from the background queue) is
        // what keeps this free of cross-thread mutable state, so no
        // lock is needed and the compiler's Sendable capture checking
        // is satisfied by construction.
        let stderrHandle = stderrPipe.fileHandleForReading
        let stderrDrain = DispatchWorkItem { _ = stderrHandle.readToEndSafely() }
        DispatchQueue.global(qos: .userInitiated).async(execute: stderrDrain)

        let stdoutData = stdoutPipe.fileHandleForReading.readToEndSafely()

        // Both reads must reach EOF before the wait — the child's exit
        // closes both write ends, so this cannot outlive the child.
        stderrDrain.wait()
        task.waitUntilExit()
        return stdoutData
    }

    /// Parse macOS `netstat -anv -p tcp` output. Kept as a pure seam so the
    /// listener-only and exact-port contracts do not depend on the host's live
    /// sockets. Field positions are defined by netstat's own header; malformed
    /// or permission-redacted rows fail closed.
    static func parseListeningPIDs(_ data: Data, port: Int) -> [Int32] {
        guard let text = String(data: data, encoding: .utf8) else { return [] }
        guard (1...65_535).contains(port) else { return [] }

        enum OwnerColumn {
            case processAndPID(Int)
            case numericPID(Int)
        }

        let portSuffix = ".\(port)"
        var result: [Int32] = []
        var seen: Set<Int32> = []
        var ownerColumn: OwnerColumn?
        for line in text.split(separator: "\n") {
            let fields = line.split(whereSeparator: { $0.isWhitespace })

            // Header labels contain two two-word address columns, while each
            // data row contains one token per address. Derive the owner column
            // from the table's own header so both supported macOS layouts work:
            // current releases expose `process:pid`; older releases expose
            // separate numeric `pid` / `epid` columns.
            if fields.first == "Proto" {
                let addressLabelAdjustment = 2
                if let index = fields.firstIndex(of: "process:pid"),
                   index >= addressLabelAdjustment {
                    ownerColumn = .processAndPID(index - addressLabelAdjustment)
                } else if let index = fields.firstIndex(of: "pid"),
                          fields.contains("epid"),
                          index >= addressLabelAdjustment {
                    ownerColumn = .numericPID(index - addressLabelAdjustment)
                } else {
                    ownerColumn = nil
                }
                continue
            }

            guard let ownerColumn,
                  fields.count > 5,
                  fields[0] == "tcp4" || fields[0] == "tcp6",
                  fields[5] == "LISTEN",
                  fields[3].hasSuffix(portSuffix) else { continue }

            let pid: Int32?
            switch ownerColumn {
            case let .processAndPID(index):
                guard fields.indices.contains(index),
                      let separator = fields[index].lastIndex(of: ":") else { continue }
                pid = Int32(fields[index][fields[index].index(after: separator)...])
            case let .numericPID(index):
                guard fields.indices.contains(index) else { continue }
                pid = Int32(fields[index])
            }

            guard let pid,
                  pid > 0,
                  seen.insert(pid).inserted else { continue }
            result.append(pid)
        }
        return result
    }

    /// Drain a child process's pipe to EOF using the *throwing*
    /// ``FileHandle/readToEnd()`` rather than ``readDataToEndOfFile()``.
    ///
    /// ``readDataToEndOfFile()`` raises an Objective-C
    /// ``NSFileHandleOperationException`` ("Bad file descriptor") when
    /// the underlying descriptor has gone bad (EBADF) — e.g. when a
    /// child is reaped and its pipe FD is closed concurrently under
    /// heavy load. That exception is NOT catchable from Swift, so it
    /// ``SIGABRT``s the whole process. It was observed aborting the
    /// parallel test suite mid-run (a ``ps``/``lsof`` child's pipe FD
    /// racing teardown), and the same crash could take down the app on
    /// launch since the port sweep runs on every server start.
    ///
    /// ``readToEnd()`` bridges the identical failure to a catchable
    /// Swift error, so a bad descriptor degrades to empty ``Data`` —
    /// every caller here already treats empty as "no pids" / "no
    /// command", i.e. the sweep simply finds nothing rather than
    /// crashing.
    static func drainPipeToEOF(_ handle: FileHandle) -> Data {
        handle.readToEndSafely()
    }

    /// Codex r2 (#20 P2) gate: verify the recorded PID's live command
    /// matches the rapid-owned shape before trusting the record's
    /// pgid for a kill. Lifted out so a unit test can pin the
    /// "record points at a foreign reused PID → identity fails →
    /// foreign group stays alive" contract without spawning a real
    /// child or mocking lsof.
    static func recordIdentityHolds(_ record: OwnedServerRecord) -> Bool {
        guard let cmd = processCommand(pid: record.pid) else {
            // ``ps`` couldn't read the pid (process gone or unprivileged).
            // Treat as identity failure — the recorded pgid is no
            // longer trustworthy.
            return false
        }
        return isRapidOwnedCommand(cmd)
    }

    /// Read the executable command for ``pid`` via ``ps -p <pid> -o command=``.
    /// Returns nil if the pid no longer exists or the command can't be
    /// determined. We compare against the rapid-mlx marker substrings
    /// below to ensure the sweep never sends a signal to an unrelated
    /// process that happens to be listening on :8000.
    static func processCommand(pid: Int32) -> String? {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/bin/ps")
        task.arguments = ["-p", String(pid), "-o", "command="]
        guard let data = runCapturingStdout(task) else { return nil }
        guard let s = String(data: data, encoding: .utf8) else { return nil }
        let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    /// True when ``command`` looks like a rapid-mlx-spawned process.
    ///
    /// Codex round-4 finding: a plain substring match (``contains
    /// "rapid-mlx"``) was too wide — it would also match
    /// ``rapid-mlx-proxy``, ``rapid-mlx-watcher``, any unrelated
    /// script with that name in its argv, and especially a user
    /// running ``rapid-mlx serve`` from their own terminal (which
    /// is NOT an orphan from THIS Rapid.app process).
    ///
    /// PR #26 codex meta-review finding 5 (P1 regression): the
    /// previous shape ONLY accepted basename ``rapid-mlx``, but
    /// editable pip installs and dev-mode runs use the
    /// ``python -m vllm_mlx serve`` form where the basename is
    /// ``python3`` / ``python3.12``. Those orphans were leaking
    /// through the sweep and holding GPU memory across launches.
    ///
    /// Issue #170 follow-up: the Homebrew formula ships
    /// ``rapid-mlx`` as a tiny ``#!.../libexec/bin/python3.12``
    /// shebang script. The kernel resolves the shebang by exec-ing
    /// the interpreter, so ``ps -o command=`` records the executable
    /// as the Python framework binary (basename ``Python`` — capital
    /// P, which the existing ``.lowercased()`` already handles) and
    /// argv[1] as the ``rapid-mlx`` script path. That is neither
    /// Form 1 (basename != "rapid-mlx") nor Form 2 (no
    /// ``-m vllm_mlx``), so a Force-Quit orphan from a brew install
    /// survived the sweep and held the port + GPU memory across
    /// launches — defeating the whole PR #142 / #20 cleanup chain
    /// on the most common install channel. Form 3 catches that
    /// exact shape.
    ///
    /// Require the EXACT executable shape that ``ServerManager``
    /// is permitted to launch:
    ///
    ///   * Form 1 — basename == "rapid-mlx" (brew/pip console-script
    ///     install path) and argv contains the ``serve`` subcommand
    ///   * Form 2 — basename matches ``python`` / ``python3`` /
    ///     ``python3.N`` and argv carries ``-m vllm_mlx serve``
    ///   * Form 3 — basename matches ``python*`` AND the first
    ///     non-flag argv token is a path whose basename is
    ///     ``rapid-mlx`` AND argv contains the ``serve`` verb
    ///     (Homebrew / pipx Python-shebang console-script shape)
    ///
    /// Anything else is reported and left alone.
    static func isRapidOwnedCommand(_ command: String) -> Bool {
        // ``ps -o command=`` returns the full argv joined by single
        // spaces, starting with the executable path. Codex r2 #170:
        // a naive first-space split mangles paths-with-spaces (most
        // notably macOS pipx's current default at
        // ``~/Library/Application Support/pipx/venvs/...``). Use
        // ``splitCommand`` which walks tokens and stops only at a
        // boundary that looks like a NEW argv element (a ``-`` flag
        // or a fresh absolute path).
        let trimmed = command.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return false }
        let (exe, argv) = splitCommand(trimmed)
        let exeBase = (exe as NSString).lastPathComponent.lowercased()
        let argvLower = argv.lowercased()

        // Form 1: brew / pip console-script — basename "rapid-mlx"
        // running the server subcommand. Other rapid-mlx commands
        // (pull, ls, models, rm) are not app-owned port orphans.
        if exeBase == "rapid-mlx" {
            return containsServeVerb(argvLower)
        }

        // Form 2: ``python -m vllm_mlx serve``. Match any python
        // basename (``python``, ``python3``, ``python3.12`` etc.)
        // because pyenv / virtualenv / homebrew vary the suffix.
        // Form 3: ``python <path-to>/rapid-mlx serve …``. The kernel
        // resolves a ``#!/path/python`` shebang on the brew/pipx
        // ``rapid-mlx`` console-script by exec-ing the interpreter
        // with the script path as argv[1]; ``ps -o command=`` then
        // shows ``Python /opt/homebrew/Cellar/.../rapid-mlx serve …``.
        // Both forms share the python-basename gate, so try them
        // in order.
        if isPythonBasename(exeBase) {
            // Require BOTH the module flag and the serve verb so a
            // user running ``python -m vllm_mlx pull foo`` is
            // correctly classified as NOT a server orphan.
            if containsModuleFlag(argvLower) && containsServeVerb(argvLower) {
                return true
            }
            // Form 3: the first non-flag argv token must be a path
            // whose basename is ``rapid-mlx`` AND ``serve`` must
            // appear later in argv. Restricting to the FIRST
            // non-flag token guards against a user running
            // ``python my_script.py --bin /opt/.../rapid-mlx --serve``
            // where ``rapid-mlx`` is referenced only as a flag value
            // rather than as the script being executed.
            if firstNonFlagArgvTokenIsRapidMlx(argv: argv)
                && containsServeVerb(argvLower) {
                return true
            }
            return false
        }

        return false
    }

    /// Split ``trimmed`` (the output of ``ps -o command=``) into the
    /// executable path + the rest of argv. Handles paths-with-spaces
    /// by walking tokens and stopping the executable run only at a
    /// boundary that looks like a NEW argv element — specifically:
    ///
    ///   * a token starting with ``-`` (an interpreter flag like
    ///     ``-E -s`` or ``-m``); OR
    ///   * a token starting with ``/`` that is NOT a continuation
    ///     of the executable path (i.e. we already have at least
    ///     one absolute-path token in the run); OR
    ///   * the literal first whitespace if the run-so-far ALREADY
    ///     ends in a basename we recognize as either ``rapid-mlx``
    ///     or a python interpreter (``python``, ``python3``,
    ///     ``python3.N``). We trust the recognized basename to
    ///     mark the executable boundary — this is the fast path
    ///     for all non-spaced installs and any case where the
    ///     interpreter itself has no spaces.
    ///
    /// Returns ``(exe, argv)`` where ``exe`` is the reconstructed
    /// (possibly multi-token) executable path and ``argv`` is the
    /// trimmed remainder. ``exe`` will be the whole string when no
    /// argv tail is present.
    private static func splitCommand(_ trimmed: String) -> (String, String) {
        let tokens = trimmed.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        guard !tokens.isEmpty else { return ("", "") }

        // Fast path: the first token's basename already matches a
        // recognized interpreter / rapid-mlx — no path-with-spaces
        // reconstruction needed.
        let firstBase = (tokens[0] as NSString).lastPathComponent.lowercased()
        if firstBase == "rapid-mlx" || isPythonBasename(firstBase) {
            let tail = tokens.dropFirst().joined(separator: " ")
            return (tokens[0], tail)
        }

        // Slow path: the first token does NOT match. Extend the
        // executable run by joining subsequent tokens with ``" "``
        // until either:
        //   (a) the joined basename matches a recognized shape
        //       (this is the path-with-spaces success case), or
        //   (b) the NEXT token starts with ``-`` or ``/`` — that's
        //       a definite argv boundary, so the run ends here even
        //       if the basename is unrecognized (we fall back to
        //       returning the run as-is and let the caller's
        //       ``isPythonBasename`` / ``exeBase == "rapid-mlx"``
        //       gates do their thing).
        var runEnd = 0 // last index included in the exe run
        for idx in 1..<tokens.count {
            let candidate = tokens[0...idx].joined(separator: " ")
            let base = (candidate as NSString).lastPathComponent.lowercased()
            if base == "rapid-mlx" || isPythonBasename(base) {
                runEnd = idx
                break
            }
            // ``-flag`` or ``/new-path`` — definite argv boundary.
            // Don't include ``tokens[idx]`` in the exe run.
            if tokens[idx].hasPrefix("-") || tokens[idx].hasPrefix("/") {
                break
            }
            runEnd = idx
        }
        let exe = tokens[0...runEnd].joined(separator: " ")
        let tail = tokens.dropFirst(runEnd + 1).joined(separator: " ")
        return (exe, tail)
    }

    /// True when the first non-flag token of ``argv`` resolves to a
    /// path whose basename is ``rapid-mlx``. Skips leading interpreter
    /// flags (anything starting with ``-``) so ``python -E -s
    /// /path/rapid-mlx serve`` still matches, but a stray
    /// ``--bin /path/rapid-mlx`` reference somewhere later in argv
    /// does NOT (that would be a flag VALUE, not the script being
    /// executed). Returns early on a ``-m`` so the Form 2 dispatch
    /// above is never double-counted here.
    ///
    /// Codex r2 #170: macOS pipx now defaults to
    /// ``~/Library/Application Support/pipx/venvs/rapid-mlx/...``,
    /// which contains a literal space inside the script path. ``ps
    /// -o command=`` emits the path verbatim, so naive whitespace
    /// tokenization mis-segments the path. After landing on the
    /// FIRST non-flag token, we greedily extend the candidate path
    /// by joining subsequent tokens with ``" "`` until either the
    /// joined basename is ``rapid-mlx`` (success) or the next
    /// token is a definite argv boundary (``-`` flag or fresh ``/``
    /// path), at which point we stop. Restricting the join to the
    /// FIRST non-flag token preserves the "flag VALUE" rejection
    /// for the ``--bin /path/rapid-mlx --serve`` shape.
    ///
    /// Case sensitivity: the basename comparison is case-INsensitive
    /// to match the existing ``exeBase`` lowercase convention, but
    /// the underlying filename ``rapid-mlx`` is always lowercase on
    /// every install channel we ship (brew formula + pipx +
    /// console-script) so this is effectively an exact match.
    private static func firstNonFlagArgvTokenIsRapidMlx(argv: String) -> Bool {
        let tokens = argv.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        // Locate the first non-flag token (= start of the SCRIPT being
        // executed). Bail on ``-m`` so the Form 2 module-run shape is
        // not double-counted here.
        var scriptStart: Int = -1
        for (idx, token) in tokens.enumerated() {
            if token == "-m" { return false }
            // Skip plain interpreter flags like ``-E``, ``-s``,
            // ``-I``, ``-OO``. The first non-flag token is argv[1]
            // for the SCRIPT being executed.
            if token.hasPrefix("-") { continue }
            scriptStart = idx
            break
        }
        guard scriptStart >= 0 else { return false }

        // Try increasingly long joins of [scriptStart ... end] until
        // the joined basename is ``rapid-mlx``. The fast path
        // (no embedded space in the script path) is the single-token
        // case checked first.
        for endIdx in scriptStart..<tokens.count {
            let candidate = tokens[scriptStart...endIdx].joined(separator: " ")
            let base = (candidate as NSString).lastPathComponent.lowercased()
            if base == "rapid-mlx" {
                return true
            }
            // If the NEXT token is a definite argv boundary, stop
            // extending the candidate — anything beyond this is the
            // script's own argv (verb + args), not part of the path.
            let nextIdx = endIdx + 1
            if nextIdx < tokens.count {
                let next = tokens[nextIdx]
                if next.hasPrefix("-") || next.hasPrefix("/") {
                    return false
                }
            }
        }
        return false
    }

    private static func containsServeVerb(_ argv: String) -> Bool {
        argv.split(whereSeparator: { $0.isWhitespace }).contains("serve")
    }

    private static func containsModuleFlag(_ argv: String) -> Bool {
        // Match ``-m vllm_mlx`` or ``-m vllm_mlx.<submodule>``
        // (e.g. ``vllm_mlx.cli``, ``vllm_mlx.main``) as adjacent
        // tokens — guards against a stray ``vllm_mlx`` substring
        // in a file path. Both the top-level package run shape
        // and the submodule run shape are reachable depending on
        // how rapid-mlx was installed.
        let tokens = argv.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        // Iterate over ADJACENT PAIRS, never a manually-constructed
        // ``0..<(count - 1)``. On an empty argv that expression is
        // ``0..<(-1)``, and Swift traps on Range construction
        // (`lowerBound <= upperBound`) BEFORE the loop body or any
        // `where` clause can guard it — the previous
        // ``where i + 1 < tokens.count`` was dead code for exactly the
        // input that crashes.
        //
        // Empty argv is reachable in production, not theoretical: a
        // Python started with no arguments (``python3 < script.py``,
        // a bare REPL, ``python -``) reports a single-token
        // ``ps -o command=`` line, so ``splitCommand`` returns an
        // empty argv tail. That token's basename is ``Python`` on the
        // macOS framework build, which passes ``isPythonBasename``
        // and routes straight here. Any such process merely LISTENING
        // on a swept port crashed the app on launch, because
        // ``RapidApp`` runs the sweep during ``AppDelegate`` init.
        // Verified: repro trapped at this line with exit 133 (SIGTRAP).
        for (flag, modName) in zip(tokens, tokens.dropFirst()) {
            guard flag == "-m" else { continue }
            if modName == "vllm_mlx" || modName.hasPrefix("vllm_mlx.") {
                return true
            }
        }
        return false
    }

    private static func isPythonBasename(_ basename: String) -> Bool {
        guard basename.hasPrefix("python") else { return false }
        let suffix = basename.dropFirst("python".count)
        // Accept "", "3", "3.12" etc. Reject "pythonista" or
        // "python-launcher".
        if suffix.isEmpty { return true }
        let allowedSuffix = CharacterSet(charactersIn: "0123456789.")
        return suffix.unicodeScalars.allSatisfy { allowedSuffix.contains($0) }
    }
}
