import Darwin
import Foundation
import Testing
@testable import Rapid

@Suite("PortSweep process identity")
struct PortSweepTests {
    @Test("Current socket-table parser returns only exact TCP listeners without filesystem discovery")
    func currentSocketTableParserIsListenerAndPortScoped() {
        #expect(PortSweep.socketTableProbeExecutable.path == "/usr/sbin/netstat")
        #expect(PortSweep.socketTableProbeArguments == ["-anv", "-p", "tcp"])

        let output = """
        Active Internet connections (including servers)
        Proto Recv-Q Send-Q  Local Address Foreign Address (state) rxbytes txbytes rhiwat shiwat process:pid state
        tcp4 0 0 127.0.0.1.8000 *.* LISTEN 0 0 131072 131072 python3.12:4321 00000
        tcp6 0 0 ::1.8000 *.* LISTEN 0 0 131072 131072 rapid-mlx:4321 00000
        tcp4 0 0 127.0.0.1.18000 *.* LISTEN 0 0 131072 131072 foreign:9999 00000
        tcp4 0 0 127.0.0.1.8000 127.0.0.1.50000 ESTABLISHED 0 0 131072 131072 curl:7777 00102
        tcp4 0 0 127.0.0.1.8000 *.* LISTEN 0 0 131072 131072 unknown:* 00000
        """

        let pids = PortSweep.parseListeningPIDs(Data(output.utf8), port: 8000)

        #expect(pids == [4321])
        #expect(PortSweep.parseListeningPIDs(Data(output.utf8), port: 18_000) == [9999])
        #expect(PortSweep.parseListeningPIDs(Data(output.utf8), port: 0).isEmpty)
    }

    @Test("Current socket-table parser includes dual-stack tcp46 listeners")
    func currentSocketTableParserIncludesDualStackListeners() {
        let output = """
        Active Internet connections (including servers)
        Proto Recv-Q Send-Q  Local Address Foreign Address (state) rxbytes txbytes rhiwat shiwat process:pid state
        tcp46 0 0 *.48844 *.* LISTEN 0 0 131072 131072 Python:77187 00000
        """

        #expect(PortSweep.parseListeningPIDs(Data(output.utf8), port: 48_844) == [77_187])
    }

    @Test("Legacy socket-table parser reads the separate numeric pid column")
    func legacySocketTableParserReadsNumericPIDColumn() {
        let output = """
        Active Internet connections (including servers)
        Proto Recv-Q Send-Q  Local Address Foreign Address (state) rhiwat shiwat pid epid state options
        tcp4 0 0 127.0.0.1.8000 *.* LISTEN 131072 131072 2468 0 00000 00000000
        tcp6 0 0 ::1.8000 *.* LISTEN 131072 131072 2468 0 00000 00000000
        tcp4 0 0 127.0.0.1.18000 *.* LISTEN 131072 131072 9753 0 00000 00000000
        tcp4 0 0 127.0.0.1.8000 127.0.0.1.50000 ESTABLISHED 131072 131072 8642 0 00102 00000000
        """

        #expect(PortSweep.parseListeningPIDs(Data(output.utf8), port: 8000) == [2468])
        #expect(PortSweep.parseListeningPIDs(Data(output.utf8), port: 18_000) == [9753])
    }

    @Test("Socket-table parser fails closed when the owner layout is absent or malformed")
    func socketTableParserRejectsUnknownOwnerLayout() {
        let output = """
        Proto Recv-Q Send-Q Local Address Foreign Address (state) rxbytes txbytes rhiwat shiwat owner state
        tcp4 0 0 127.0.0.1.8000 *.* LISTEN 0 0 131072 131072 4321 00000
        """

        #expect(PortSweep.parseListeningPIDs(Data(output.utf8), port: 8000).isEmpty)
    }

    @Test("rapid-mlx serve command is considered sweep-owned")
    func rapidMlxServeAccepted() {
        #expect(PortSweep.isRapidOwnedCommand("/opt/homebrew/bin/rapid-mlx serve qwen3.5-4b --port 8000"))
    }

    /// Regression: a bad/closed pipe descriptor must degrade to empty
    /// ``Data``, NOT crash the process. ``drainPipeToEOF`` replaced
    /// ``FileHandle.readDataToEndOfFile()``, which raises an uncatchable
    /// ``NSFileHandleOperationException`` ("Bad file descriptor") on a
    /// closed/reaped FD and ``SIGABRT``s the whole run — observed
    /// aborting the parallel test suite when a ``ps``/``lsof`` child's
    /// pipe raced teardown. Closing the read end here deterministically
    /// forces the EBADF path: on the OLD code this test would crash; on
    /// the throwing ``readToEnd()`` path it returns empty and passes.
    @Test("drainPipeToEOF returns empty on a closed descriptor instead of crashing")
    func drainPipeToEOFOnClosedDescriptorDoesNotCrash() throws {
        let pipe = Pipe()
        let reader = pipe.fileHandleForReading
        try reader.close()
        let data = PortSweep.drainPipeToEOF(reader)
        #expect(data.isEmpty)
    }

    @Test("rapid-mlx non-server commands are not sweep-owned")
    func rapidMlxPullRejected() {
        #expect(!PortSweep.isRapidOwnedCommand("/opt/homebrew/bin/rapid-mlx pull qwen3.5-4b"))
        #expect(!PortSweep.isRapidOwnedCommand("/opt/homebrew/bin/rapid-mlx ls"))
    }

    @Test("similarly named executables are rejected")
    func similarlyNamedExecutablesRejected() {
        #expect(!PortSweep.isRapidOwnedCommand("/tmp/rapid-mlx-proxy serve --port 8000"))
        #expect(!PortSweep.isRapidOwnedCommand("/tmp/not-rapid-mlx serve --port 8000"))
    }

    // PR #26 codex meta-review finding 5 (P1): editable pip installs
    // and dev-mode runs spawn the server as ``python -m vllm_mlx serve``
    // (or the ``.cli`` submodule shape). Those processes ARE app-owned
    // orphans across launches and must be swept; the earlier test
    // here asserted the WRONG behavior (rejecting them) and was
    // letting them leak GPU memory across launches.
    @Test("python -m vllm_mlx serve forms are sweep-owned")
    func pythonModuleServeAccepted() {
        #expect(PortSweep.isRapidOwnedCommand(
            "/usr/bin/python3 -m vllm_mlx serve qwen3.5-4b --port 8000"
        ))
        #expect(PortSweep.isRapidOwnedCommand(
            "/usr/bin/python3 -m vllm_mlx.cli serve qwen --port 8000"
        ))
        #expect(PortSweep.isRapidOwnedCommand(
            "/opt/homebrew/Cellar/python@3.12/3.12.7/bin/python3.12 -m vllm_mlx serve qwen --port 8000"
        ))
    }

    @Test("python -m vllm_mlx non-server commands are not sweep-owned")
    func pythonModuleNonServeRejected() {
        #expect(!PortSweep.isRapidOwnedCommand("/usr/bin/python3 -m vllm_mlx pull qwen3.5-4b"))
        #expect(!PortSweep.isRapidOwnedCommand("/usr/bin/python3 -m vllm_mlx.cli models"))
    }

    @Test("python without -m vllm_mlx is not sweep-owned")
    func pythonWithoutModuleFlagRejected() {
        // A user running an unrelated python web server on :8000
        // must NOT be killed even if their argv happens to contain
        // the word "serve".
        #expect(!PortSweep.isRapidOwnedCommand("/usr/bin/python3 some_script.py serve --port 8000"))
        // Stray ``vllm_mlx`` substring in a path (not after -m) is
        // not a module run.
        #expect(!PortSweep.isRapidOwnedCommand("/usr/bin/python3 /tmp/vllm_mlx_helper.py serve"))
    }

    @Test("python-launcher style executables are not python")
    func pythonLikeExecutablesRejected() {
        // ``pythonista`` / ``python-launcher`` / ``pythonw`` are
        // common non-CPython binaries we should NOT treat as the
        // python that runs ``-m vllm_mlx``.
        #expect(!PortSweep.isRapidOwnedCommand("/usr/bin/pythonista -m vllm_mlx serve --port 8000"))
        #expect(!PortSweep.isRapidOwnedCommand("/usr/bin/python-launcher -m vllm_mlx serve --port 8000"))
    }

    // MARK: - Issue #170: Homebrew Python-shebang console-script form
    //
    // The Homebrew formula ships ``rapid-mlx`` as a script with a
    // ``#!/opt/homebrew/Cellar/.../libexec/bin/python3.12`` shebang.
    // The kernel resolves the shebang by exec-ing the interpreter,
    // so ``ps -o command=`` records the executable as the Python
    // framework binary (basename ``Python`` — capital P, hence the
    // existing ``.lowercased()`` on ``exeBase``) and argv[1] as the
    // ``rapid-mlx`` script path. Form 1 missed it (basename !=
    // "rapid-mlx") and Form 2 missed it (no ``-m vllm_mlx``), so
    // Force-Quit orphans from brew installs survived PR #142 /
    // #20's cleanup chain. Form 3 closes the gap.
    //
    // The shape below was captured live via ``ps -A -o command=``
    // against a brew install of rapid-mlx 0.7.3 on Apple Silicon
    // before opening the PR.

    @Test("Homebrew Python-shebang rapid-mlx serve is sweep-owned (#170)")
    func homebrewPythonShebangAccepted() {
        // Apple Silicon brew prefix — exact ``ps`` output.
        // Capital ``Python`` exercises the lowercase guard.
        #expect(PortSweep.isRapidOwnedCommand(
            "/opt/homebrew/Cellar/python@3.12/3.12.13_4/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python /opt/homebrew/Cellar/rapid-mlx/0.7.3/libexec/bin/rapid-mlx serve qwen3.5-4b-4bit --port 8765"
        ))
        // Intel-mac brew prefix.
        #expect(PortSweep.isRapidOwnedCommand(
            "/usr/local/Cellar/python@3.12/3.12.7/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python /usr/local/Cellar/rapid-mlx/0.7.3/libexec/bin/rapid-mlx serve qwen3.5-4b --port 8000"
        ))
        // pipx venv shape — interpreter + script live next to each
        // other inside the venv bin dir.
        #expect(PortSweep.isRapidOwnedCommand(
            "/Users/raullen/.local/pipx/venvs/rapid-mlx/bin/python3.12 /Users/raullen/.local/pipx/venvs/rapid-mlx/bin/rapid-mlx serve qwen3.5-4b --port 8000"
        ))
        // Interpreter flags before the script path must not block
        // the match — ``python -E -s /path/rapid-mlx serve`` is
        // still the brew console-script shape.
        #expect(PortSweep.isRapidOwnedCommand(
            "/opt/homebrew/bin/python3.12 -E -s /opt/homebrew/Cellar/rapid-mlx/0.7.3/libexec/bin/rapid-mlx serve qwen3.5-4b --port 8000"
        ))
    }

    @Test("Homebrew Python-shebang rapid-mlx non-server commands are not sweep-owned (#170)")
    func homebrewPythonShebangNonServeRejected() {
        // Other rapid-mlx subcommands (pull, ls, models) are NOT
        // app-owned port orphans — they shouldn't be killed even
        // if a user happens to be running one in the background.
        #expect(!PortSweep.isRapidOwnedCommand(
            "/opt/homebrew/Cellar/python@3.12/3.12.13_4/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python /opt/homebrew/Cellar/rapid-mlx/0.7.3/libexec/bin/rapid-mlx pull qwen3.5-4b"
        ))
        #expect(!PortSweep.isRapidOwnedCommand(
            "/opt/homebrew/bin/python3 /opt/homebrew/Cellar/rapid-mlx/0.7.3/libexec/bin/rapid-mlx ls"
        ))
    }

    @Test("python with rapid-mlx referenced only as a flag value is not sweep-owned (#170)")
    func pythonRapidMlxAsFlagValueRejected() {
        // ``python my_script.py --bin /opt/.../rapid-mlx --serve``
        // mentions ``rapid-mlx`` only as the VALUE of a ``--bin``
        // flag, not as the script being executed. Form 3 must not
        // be fooled by that — the first non-flag argv token is
        // ``my_script.py``, which is the script being run.
        #expect(!PortSweep.isRapidOwnedCommand(
            "/usr/bin/python3 /tmp/my_script.py --bin /opt/homebrew/Cellar/rapid-mlx/0.7.3/libexec/bin/rapid-mlx --serve --port 8000"
        ))
        // Even with the literal word ``serve`` later in argv, the
        // script is ``some_helper.py`` not ``rapid-mlx``, so it's
        // a user process and must be left alone.
        #expect(!PortSweep.isRapidOwnedCommand(
            "/usr/bin/python3 /tmp/some_helper.py --bin /usr/local/bin/rapid-mlx serve"
        ))
    }

    @Test("python-shebang form requires the serve verb (#170)")
    func pythonShebangRequiresServeVerb() {
        // Without ``serve`` later in argv, the brew console-script
        // shape collapses to ``python /path/rapid-mlx`` — which
        // could be a help invocation, a no-arg call, etc. Refuse
        // to sweep without the explicit serve verb.
        #expect(!PortSweep.isRapidOwnedCommand(
            "/opt/homebrew/bin/python3.12 /opt/homebrew/Cellar/rapid-mlx/0.7.3/libexec/bin/rapid-mlx"
        ))
        #expect(!PortSweep.isRapidOwnedCommand(
            "/opt/homebrew/bin/python3.12 /opt/homebrew/Cellar/rapid-mlx/0.7.3/libexec/bin/rapid-mlx --help"
        ))
    }

    // MARK: - Codex r2 #170: paths-with-spaces (macOS pipx default)
    //
    // macOS pipx now defaults to
    // ``~/Library/Application Support/pipx/venvs/rapid-mlx/...``,
    // a path with a literal space. ``ps -o command=`` emits the
    // path verbatim without quoting/escaping, so naive whitespace
    // tokenization mis-segments the path. ``splitCommand`` and
    // ``firstNonFlagArgvTokenIsRapidMlx`` both extend their token
    // runs until the basename is recognized.

    @Test("macOS pipx default install path with spaces is sweep-owned (#170 codex r2)")
    func macosPipxSpacedPathAccepted() {
        // The exact shape from ``ServerLocator.swift`` line 249:
        // ``~/Library/Application Support/pipx/venvs/rapid-mlx/bin/...``.
        // Both the interpreter path AND the script path contain a
        // literal space. The fast-path basename match must rebuild
        // both.
        #expect(PortSweep.isRapidOwnedCommand(
            "/Users/raullen/Library/Application Support/pipx/venvs/rapid-mlx/bin/python3.12 /Users/raullen/Library/Application Support/pipx/venvs/rapid-mlx/bin/rapid-mlx serve qwen3.5-4b --port 8000"
        ))
        // Form 1 equivalent — the brew/pip console-script invoked
        // directly (no python in front) but the install path
        // contains a space. ``exe`` reconstruction must still
        // recover the ``rapid-mlx`` basename.
        #expect(PortSweep.isRapidOwnedCommand(
            "/Users/raullen/Library/Application Support/pipx/venvs/rapid-mlx/bin/rapid-mlx serve qwen3.5-4b --port 8000"
        ))
    }

    @Test("spaced path with rapid-mlx as flag value is still rejected (#170 codex r2)")
    func macosPipxSpacedPathFlagValueRejected() {
        // The greedy basename-rebuild must NOT be fooled by a
        // ``--bin <spaced path>/rapid-mlx`` flag value into
        // sweeping a foreign python script. Form 3's
        // "first non-flag argv token" gate already runs on the
        // script being executed (``/tmp/my_script.py``) — the
        // ``rapid-mlx`` mention after ``--bin`` is a flag VALUE,
        // not the script.
        #expect(!PortSweep.isRapidOwnedCommand(
            "/usr/bin/python3 /tmp/my_script.py --bin /Users/raullen/Library/Application Support/pipx/venvs/rapid-mlx/bin/rapid-mlx --serve"
        ))
    }

    // MARK: - Codex r2 (#20 P2): PID-reuse identity gate

    @Test("recordIdentityHolds(record) → false when the record pid resolves to a foreign command")
    func recordIdentityFalseForForeignPid() {
        // The test runner itself is by definition NOT a rapid-mlx
        // process. A record that claims our own PID must fail the
        // identity gate, otherwise PortSweep.sweep() would PGID-kill
        // the test runner the next time a stale owned-server.json
        // pointed at a reused PID matching a foreign port-binder.
        let record = OwnedServerRecord(
            pid: getpid(),
            pgid: getpgrp(),
            port: 8000,
            alias: "qwen3.5-4b",
            writtenAt: Date(timeIntervalSince1970: 0)
        )
        #expect(
            !PortSweep.recordIdentityHolds(record),
            "identity gate must refuse a record whose pid resolves to a non-rapid-mlx command — otherwise PID reuse turns the record-first path into a friendly-fire weapon"
        )
    }

    @Test("recordIdentityHolds(record) → false when the record pid no longer exists")
    func recordIdentityFalseForDeadPid() {
        // pid 1 is launchd (always alive) and 99999 is plausibly
        // unused on a typical Mac. Pick a pid that ``ps -p`` will
        // refuse to read; either ps returns empty stdout or fails.
        // Identity gate must treat "ps couldn't read" as a failure
        // so a long-dead record doesn't accidentally pass.
        let record = OwnedServerRecord(
            pid: 99_999_999, // out of macOS pid_t range guarantees failure
            pgid: 99_999_999,
            port: 8000,
            alias: "qwen3.5-4b",
            writtenAt: Date(timeIntervalSince1970: 0)
        )
        #expect(
            !PortSweep.recordIdentityHolds(record),
            "identity gate must refuse a record whose pid no longer resolves to any process — the recorded pgid is meaningless"
        )
    }
}
