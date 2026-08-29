import Darwin
import Foundation

struct TestSubprocessResult: Sendable {
    let terminationStatus: Int32
    let standardOutput: Data
    let standardError: Data
}

enum TestSubprocessError: Error, CustomStringConvertible {
    case timedOut(command: String, seconds: TimeInterval, pid: pid_t)

    var description: String {
        switch self {
        case let .timedOut(command, seconds, pid):
            return "test subprocess timed out after \(seconds)s (pid \(pid)): \(command); process sample was emitted above"
        }
    }
}

/// Async, bounded launcher for test-only subprocesses.
///
/// A synchronous ``Process.waitUntilExit()`` occupies a cooperative executor
/// worker when called from an async or MainActor test. On a low-core runner,
/// enough such waits can prevent the tasks responsible for making the child
/// exit from running at all. This helper instead waits on a native thread,
/// independent of Swift concurrency, while a second native-thread watchdog
/// bounds both process exit and pipe drainage. A timeout emits one bounded
/// sample for the process-group leader, then escalates group-wide TERM to KILL
/// and closes the capture descriptors so an escaped descendant cannot retain
/// them indefinitely.
enum TestSubprocess {
    static func run(
        executableURL: URL,
        arguments: [String] = [],
        currentDirectoryURL: URL? = nil,
        environment: [String: String]? = nil,
        timeout: TimeInterval = 30,
        sampleOnTimeout: Bool = true,
        sampleDuration: Int = 3
    ) async throws -> TestSubprocessResult {
        precondition(timeout > 0, "test subprocess timeout must be positive")
        precondition(sampleDuration > 0, "sample duration must be positive")

        let stdoutPipe = try SpawnPipe()
        let stderrPipe: SpawnPipe
        do {
            stderrPipe = try SpawnPipe()
        } catch {
            stdoutPipe.closeBoth()
            throw error
        }

        let state = TestProcessCompletionState()
        let stdoutCapture = AsyncPipeCapture(
            FileHandle(fileDescriptor: stdoutPipe.readFD, closeOnDealloc: true),
            onFinish: { state.pipeFinished() }
        )
        let stderrCapture = AsyncPipeCapture(
            FileHandle(fileDescriptor: stderrPipe.readFD, closeOnDealloc: true),
            onFinish: { state.pipeFinished() }
        )

        var fileActions: posix_spawn_file_actions_t?
        var attributes: posix_spawnattr_t?
        var writeEndsOpen = true
        defer {
            if writeEndsOpen {
                Darwin.close(stdoutPipe.writeFD)
                Darwin.close(stderrPipe.writeFD)
            }
        }
        try check(posix_spawn_file_actions_init(&fileActions))
        defer { posix_spawn_file_actions_destroy(&fileActions) }
        try check(posix_spawnattr_init(&attributes))
        defer { posix_spawnattr_destroy(&attributes) }

        try check(posix_spawn_file_actions_adddup2(&fileActions, stdoutPipe.writeFD, STDOUT_FILENO))
        try check(posix_spawn_file_actions_adddup2(&fileActions, stderrPipe.writeFD, STDERR_FILENO))
        for descriptor in [
            stdoutPipe.readFD, stdoutPipe.writeFD,
            stderrPipe.readFD, stderrPipe.writeFD,
        ] {
            try check(posix_spawn_file_actions_addclose(&fileActions, descriptor))
        }
        if let currentDirectoryURL {
            let result = currentDirectoryURL.path.withCString { path in
                // The non-portable Darwin spelling is present across the
                // oldest CI SDK through macOS 26. The standardized spelling
                // is runtime-available on 26 but absent from older SDK headers,
                // so merely placing it behind #available does not compile.
                posix_spawn_file_actions_addchdir_np(&fileActions, path)
            }
            try check(result)
        }

        // A dedicated process group lets the timeout own descendants even
        // after the direct child exits while a grandchild still holds one of
        // our pipe descriptors open.
        try check(posix_spawnattr_setflags(&attributes, Int16(POSIX_SPAWN_SETPGROUP)))
        try check(posix_spawnattr_setpgroup(&attributes, 0))

        let argv = [executableURL.path] + arguments
        let inheritedEnvironment = environment ?? ProcessInfo.processInfo.environment
        let envp = inheritedEnvironment.sorted { $0.key < $1.key }
            .map { "\($0.key)=\($0.value)" }
        var pid: pid_t = 0
        let spawnResult = withMutableCStrings(argv) { argvPointer in
            withMutableCStrings(envp) { environmentPointer in
                executableURL.path.withCString { executablePath in
                    posix_spawn(
                        &pid,
                        executablePath,
                        &fileActions,
                        &attributes,
                        argvPointer,
                        environmentPointer
                    )
                }
            }
        }
        Darwin.close(stdoutPipe.writeFD)
        Darwin.close(stderrPipe.writeFD)
        writeEndsOpen = false
        guard spawnResult == 0 else {
            stdoutCapture.cancel()
            stderrCapture.cancel()
            throw POSIXError(POSIXErrorCode(rawValue: spawnResult) ?? .EIO)
        }
        let spawnedPID = pid

        Thread.detachNewThread {
            var rawStatus: Int32 = 0
            var result: pid_t
            repeat {
                result = Darwin.waitpid(spawnedPID, &rawStatus, 0)
            } while result == -1 && errno == EINTR
            if result == spawnedPID {
                state.processExited(rawStatus: rawStatus)
            } else {
                state.processExited(rawStatus: 127 << 8)
            }
        }

        let command = ([executableURL.path] + arguments)
            .map(\.debugDescription)
            .joined(separator: " ")
        Thread.detachNewThread {
            guard state.beginTimeoutIfIncomplete(after: timeout) else { return }
            // Pipe ownership ends at the caller's deadline. In particular, a
            // setsid() descendant cannot extend capture lifetime while process
            // diagnostics and termination continue on this native thread.
            stdoutCapture.cancel()
            stderrCapture.cancel()
            if sampleOnTimeout {
                sample(pid: spawnedPID, duration: sampleDuration, command: command)
            }
            terminate(pid: spawnedPID, state: state)
            state.timeoutCleanupFinished()
        }

        let exit = await withTaskCancellationHandler {
            await state.value()
        } onCancel: {
            Thread.detachNewThread {
                terminate(pid: spawnedPID, state: state)
                stdoutCapture.cancel()
                stderrCapture.cancel()
            }
        }
        async let standardOutput = stdoutCapture.value()
        async let standardError = stderrCapture.value()
        let output = await standardOutput
        let errorOutput = await standardError
        try Task.checkCancellation()
        if exit.timedOut {
            throw TestSubprocessError.timedOut(
                command: command,
                seconds: timeout,
                pid: spawnedPID
            )
        }
        return TestSubprocessResult(
            terminationStatus: terminationStatus(from: exit.rawStatus),
            standardOutput: output,
            standardError: errorOutput
        )
    }

    private static func sample(pid: pid_t, duration: Int, command: String) {
        // Sampling every descendant serially makes a fork-heavy timeout scale
        // with process count. One leader sample keeps diagnostics within the
        // caller-selected duration while the group ID remains in the message.
        let message = "TestSubprocess: timeout; sampling process-group leader \(pid) for \(duration)s: \(command)\n"
        FileHandle.standardError.write(Data(message.utf8))
        let sampler = Process()
        sampler.executableURL = URL(fileURLWithPath: "/usr/bin/sample")
        sampler.arguments = [String(pid), String(duration), "-file", "/dev/stderr"]
        sampler.standardOutput = FileHandle.standardError
        sampler.standardError = FileHandle.standardError
        let finished = DispatchSemaphore(value: 0)
        sampler.terminationHandler = { _ in finished.signal() }
        do {
            try sampler.run()
            // Symbolication can outlive the requested sampling period. Give it
            // one fixed grace second, then stop the diagnostic so termination
            // never depends on an unbounded Process wait.
            if finished.wait(timeout: .now() + .seconds(duration + 1)) == .timedOut {
                sampler.terminate()
                if finished.wait(timeout: .now() + .milliseconds(250)) == .timedOut {
                    _ = Darwin.kill(sampler.processIdentifier, SIGKILL)
                }
            }
        } catch {
            let failure = "TestSubprocess: sample failed for pid \(pid): \(error)\n"
            FileHandle.standardError.write(Data(failure.utf8))
        }
    }

    private static func terminate(pid: pid_t, state: TestProcessCompletionState) {
        signalProcessTree(pid: pid, signal: SIGTERM)
        let deadline = Date(timeIntervalSinceNow: 2)
        while processTreeExists(pid: pid), Date() < deadline {
            Thread.sleep(forTimeInterval: 0.01)
        }
        if processTreeExists(pid: pid) {
            signalProcessTree(pid: pid, signal: SIGKILL)
        }
        // waitpid normally publishes immediately after TERM/KILL. Keep one
        // final fixed grace window so the returned timeout represents a reaped
        // child, but never let a kernel-side wait stall the caller forever.
        if !state.waitForProcessExit(seconds: 1) {
            state.forceProcessExitIfMissing(rawStatus: SIGKILL)
        }
    }

    private static func signalProcessTree(pid: pid_t, signal: Int32) {
        // Signal both the group and its leader. The direct child may have
        // changed its own group after spawn; addressing its PID still gives
        // the waitpid thread a deterministic path to reap it.
        _ = Darwin.kill(-pid, signal)
        _ = Darwin.kill(pid, signal)
    }

    private static func processTreeExists(pid: pid_t) -> Bool {
        Darwin.kill(-pid, 0) == 0 || Darwin.kill(pid, 0) == 0
    }

    private static func terminationStatus(from rawStatus: Int32) -> Int32 {
        let signal = rawStatus & 0x7f
        return signal == 0 ? (rawStatus >> 8) & 0xff : signal
    }

    private static func check(_ result: Int32) throws {
        guard result != 0 else { return }
        throw POSIXError(POSIXErrorCode(rawValue: result) ?? .EIO)
    }

    private static func withMutableCStrings<Result>(
        _ strings: [String],
        _ body: (UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) -> Result
    ) -> Result {
        let storage = strings.map { strdup($0) }
        defer { storage.forEach { free($0) } }
        var pointers = storage + [nil]
        return pointers.withUnsafeMutableBufferPointer { buffer in
            body(buffer.baseAddress!)
        }
    }
}

private struct SpawnPipe {
    let readFD: Int32
    let writeFD: Int32

    init() throws {
        var descriptors = [Int32](repeating: -1, count: 2)
        let result = descriptors.withUnsafeMutableBufferPointer { buffer in
            Darwin.pipe(buffer.baseAddress!)
        }
        guard result == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        readFD = descriptors[0]
        writeFD = descriptors[1]
    }

    func closeBoth() {
        Darwin.close(readFD)
        Darwin.close(writeFD)
    }
}

private final class TestProcessCompletionState: @unchecked Sendable {
    struct Exit: Sendable {
        let rawStatus: Int32
        let timedOut: Bool
    }

    private let condition = NSCondition()
    private var exit: Exit?
    private var rawStatus: Int32?
    private var finishedPipeCount = 0
    private var timedOut = false
    private var timeoutCleanupComplete = true
    private var valueRequested = false
    private var continuation: CheckedContinuation<Exit, Never>?

    func processExited(rawStatus: Int32) {
        condition.lock()
        self.rawStatus = rawStatus
        condition.broadcast()
        let completion = completeIfReadyLocked()
        condition.unlock()
        if let completion {
            completion.continuation?.resume(returning: completion.value)
        }
    }

    func pipeFinished() {
        condition.lock()
        finishedPipeCount += 1
        condition.broadcast()
        let completion = completeIfReadyLocked()
        condition.unlock()
        if let completion {
            completion.continuation?.resume(returning: completion.value)
        }
    }

    func value() async -> Exit {
        await withCheckedContinuation { continuation in
            condition.lock()
            precondition(!valueRequested, "TestProcessCompletionState supports one consumer")
            valueRequested = true
            if let exit {
                condition.unlock()
                continuation.resume(returning: exit)
            } else {
                self.continuation = continuation
                condition.unlock()
            }
        }
    }

    func waitForProcessExit(seconds: TimeInterval) -> Bool {
        condition.lock()
        defer { condition.unlock() }
        let deadline = Date(timeIntervalSinceNow: seconds)
        while rawStatus == nil && condition.wait(until: deadline) {}
        return rawStatus != nil
    }

    func forceProcessExitIfMissing(rawStatus: Int32) {
        condition.lock()
        if self.rawStatus == nil {
            self.rawStatus = rawStatus
        }
        condition.broadcast()
        let completion = completeIfReadyLocked()
        condition.unlock()
        if let completion {
            completion.continuation?.resume(returning: completion.value)
        }
    }

    func beginTimeoutIfIncomplete(after seconds: TimeInterval) -> Bool {
        condition.lock()
        defer { condition.unlock() }
        let deadline = Date(timeIntervalSinceNow: seconds)
        while exit == nil && condition.wait(until: deadline) {}
        guard exit == nil else { return false }
        timedOut = true
        timeoutCleanupComplete = false
        return true
    }

    func timeoutCleanupFinished() {
        condition.lock()
        timeoutCleanupComplete = true
        let completion = completeIfReadyLocked()
        condition.unlock()
        if let completion {
            completion.continuation?.resume(returning: completion.value)
        }
    }

    private func completeIfReadyLocked() -> (
        continuation: CheckedContinuation<Exit, Never>?,
        value: Exit
    )? {
        guard exit == nil,
              let rawStatus,
              finishedPipeCount == 2,
              timeoutCleanupComplete
        else { return nil }
        let value = Exit(rawStatus: rawStatus, timedOut: timedOut)
        exit = value
        let continuation = continuation
        self.continuation = nil
        condition.broadcast()
        return (continuation, value)
    }
}

private final class AsyncPipeCapture: @unchecked Sendable {
    private let lock = NSLock()
    private let handle: FileHandle
    private var data = Data()
    private var finished = false
    private var valueRequested = false
    private var continuation: CheckedContinuation<Data, Never>?
    private let onFinish: @Sendable () -> Void

    init(_ handle: FileHandle, onFinish: @escaping @Sendable () -> Void) {
        self.handle = handle
        self.onFinish = onFinish
        handle.readabilityHandler = { [weak self] readable in
            guard let self else { return }
            let chunk = readable.availableData
            if chunk.isEmpty {
                finish()
            } else {
                lock.lock()
                data.append(chunk)
                lock.unlock()
            }
        }
    }

    func value() async -> Data {
        await withCheckedContinuation { continuation in
            lock.lock()
            precondition(!valueRequested, "AsyncPipeCapture supports one consumer")
            valueRequested = true
            if finished {
                let data = data
                lock.unlock()
                continuation.resume(returning: data)
            } else {
                self.continuation = continuation
                lock.unlock()
            }
        }
    }

    func cancel() {
        finish()
    }

    private func finish() {
        lock.lock()
        guard !finished else {
            lock.unlock()
            return
        }
        finished = true
        let data = data
        let continuation = continuation
        self.continuation = nil
        lock.unlock()
        // Only the caller that won the one-shot transition above mutates the
        // FileHandle. EOF delivery and timeout cancellation can race here,
        // but the losing path returns before touching the handle.
        handle.readabilityHandler = nil
        handle.closeFile()
        continuation?.resume(returning: data)
        onFinish()
    }
}
