import Darwin
import Foundation

/// Hosted-runner guard for a stranded main queue or cooperative executor.
///
/// This deliberately uses a Foundation thread rather than Swift concurrency:
/// the failure under investigation strands the cooperative executor and the
/// MainActor, so a Task-based timeout cannot run to report the blocked stacks.
enum CIHangWatchdog {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var started = false
    nonisolated(unsafe) private static var lastMain = Date.timeIntervalSinceReferenceDate
    nonisolated(unsafe) private static var lastCooperative = Date.timeIntervalSinceReferenceDate
    private static let quietLimit: TimeInterval =
        ProcessInfo.processInfo.environment["RAPID_DESKTOP_WATCHDOG_QUIET_SECONDS"] == "180"
            ? 180 : 90

    static func noteProgress() {
        lock.lock()
        let now = Date.timeIntervalSinceReferenceDate
        lastMain = now
        lastCooperative = now
        let shouldStart = !started
        started = true
        lock.unlock()

        guard shouldStart else { return }
        Thread.detachNewThread {
            monitor()
        }
    }

    static func stalledLane(
        now: TimeInterval,
        lastMain: TimeInterval,
        lastCooperative: TimeInterval,
        limit: TimeInterval
    ) -> String? {
        if now - lastMain >= limit { return "main" }
        if now - lastCooperative >= limit { return "cooperative" }
        return nil
    }

    private static func recordHeartbeat(main: Bool) {
        lock.lock()
        if main {
            lastMain = Date.timeIntervalSinceReferenceDate
        } else {
            lastCooperative = Date.timeIntervalSinceReferenceDate
        }
        lock.unlock()
    }

    private static func monitor() {
        while true {
            Thread.sleep(forTimeInterval: 5)
            DispatchQueue.main.async {
                recordHeartbeat(main: true)
            }
            Task.detached {
                recordHeartbeat(main: false)
            }
            lock.lock()
            let now = Date.timeIntervalSinceReferenceDate
            let lane = stalledLane(
                now: now, lastMain: lastMain, lastCooperative: lastCooperative, limit: quietLimit
            )
            let quietFor = now - (lane == "main" ? lastMain : lastCooperative)
            lock.unlock()
            guard let lane else { continue }

            write("CI hang watchdog: no \(lane) heartbeat for \(Int(quietFor))s; sampling blocked test process\n")
            let ownPID = getpid()
            sample(pid: ownPID)
            for childPID in childPIDs(of: ownPID) where isSwiftTestProcess(pid: childPID) {
                sample(pid: childPID)
            }
            write("CI hang watchdog: samples complete; terminating test process with exit(3)\n")
            fflush(stdout)
            fflush(stderr)
            exit(3)
        }
    }

    private static func sample(pid: pid_t) {
        write("===== /usr/bin/sample \(pid) 3 -file /dev/stdout =====\n")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/sample")
        process.arguments = [String(pid), "3", "-file", "/dev/stdout"]
        process.standardOutput = FileHandle.standardOutput
        process.standardError = FileHandle.standardError
        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            write("CI hang watchdog: sample failed for pid \(pid): \(error)\n")
        }
        write("===== end sample pid=\(pid) =====\n")
    }

    private static func childPIDs(of parentPID: pid_t) -> [pid_t] {
        let process = Process()
        let output = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/pgrep")
        process.arguments = ["-P", String(parentPID)]
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return []
        }
        let data = output.fileHandleForReading.readDataToEndOfFile()
        return String(decoding: data, as: UTF8.self)
            .split(whereSeparator: \Character.isWhitespace)
            .compactMap { pid_t($0) }
    }

    private static func isSwiftTestProcess(pid: pid_t) -> Bool {
        let process = Process()
        let output = Pipe()
        process.executableURL = URL(fileURLWithPath: "/bin/ps")
        process.arguments = ["-p", String(pid), "-o", "comm="]
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return false
        }
        let command = String(decoding: output.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        let name = URL(fileURLWithPath: command.trimmingCharacters(in: .whitespacesAndNewlines)).lastPathComponent
        return name == "swiftpm-testing" || name == "xctest"
    }

    private static func write(_ message: String) {
        FileHandle.standardError.write(Data(message.utf8))
    }
}
