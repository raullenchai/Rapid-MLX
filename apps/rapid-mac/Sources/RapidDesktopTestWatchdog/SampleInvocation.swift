import Foundation

/// Real-host sampler: captures a hung process's stack with `/usr/bin/sample`
/// plus a snapshot of process state (`ps`) and memory pressure (`vm_stat`),
/// all written into a single `.txt` artifact for CI to upload.
///
/// The invocation pattern mirrors the existing in-process
/// `CIHangWatchdog.sample(pid:)` (Tests/RapidTests/Support/CIHangWatchdog.swift)
/// so a developer who has read one recognizes the other; the addition here is
/// that the whole set is redirected into a named artifact file rather than
/// `/dev/stdout`, plus `ps`/`vm_stat` give the RAM/memory-pressure context that
/// lets a reader tell a "stuck waiting on memory" hang from a pure deadlock.
public enum SampleInvocation {

    /// Capture the stack + system state for `pid` into `artifactURL`.
    public static func capture(
        pid: pid_t,
        config: WatchdogConfig = WatchdogConfig(),
        artifactURL: URL
    ) throws {
        try FileManager.default.createDirectory(
            at: artifactURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        // Fresh materialize the artifact (FileHandle(forWritingTo:) alone
        // fails when the file doesn't yet exist), then write the sections.
        if !FileManager.default.fileExists(atPath: artifactURL.path) {
            try Data().write(to: artifactURL)
        }
        let handle = try FileHandle(forWritingTo: artifactURL)
        try handle.truncate(atOffset: 0)
        try handle.seek(toOffset: 0)
        try writeSections(into: handle, pid: pid, config: config)
        try handle.close()
    }

    private static func writeSections(into handle: FileHandle, pid: pid_t, config: WatchdogConfig) throws {
        func write(_ s: String) throws {
            try handle.write(contentsOf: Data(s.utf8))
        }

        try write("Rapid Desktop test-suite hang capture\n")
        try write("captured at: \(Date())\n")
        try write("wrapped PID: \(pid)\n")
        try write("sample duration: \(config.sampleDurationSeconds)s\n\n")

        // Process state + RAM context first, so a reader sees memory pressure
        // even if the stack sample below is truncated.
        try write("===== ps -p \(pid) -o pid,ppid,state,%cpu,%mem,rss,etime,comm =====\n")
        try write(shellOutput("/bin/ps", ["-p", String(pid), "-o", "pid,ppid,state,%cpu,%mem,rss,etime,comm"]))
        // SwiftPM waits on its test-runner child. Sampling only swift-test
        // leaves that child invisible when the parent is blocked in wait().
        // Capture executable names, never argv, so command-line secrets stay
        // out of the uploaded artifact.
        let processTable = shellOutput("/bin/ps", ["-axo", "pid=,ppid=,state=,etime=,comm="])
        let descendants = descendantProcesses(of: pid, processTable: processTable)
        try write("\n===== descendant process tree (pid,ppid,state,etime,comm) =====\n")
        if descendants.isEmpty {
            try write("<none>\n")
        } else {
            for child in descendants { try write("\(child.summary)\n") }
        }
        try write("\n===== vm_stat =====\n")
        try write(shellOutput("/usr/bin/vm_stat", []))
        try write("\n===== memory pressure -Q =====\n")
        try write(shellOutput("/usr/bin/memory_pressure", ["-Q"]))

        // The stack capture proper.
        try write("\n===== /usr/bin/sample \(pid) \(config.sampleDurationSeconds) -file ... =====\n")
        try write(streamingSample(pid: pid, seconds: config.sampleDurationSeconds))
        // At most two test runners: the bounded artifact still shows the
        // process actually executing tests if SwiftPM itself is idle.
        let runners = descendants.filter {
            let name = $0.command.lowercased()
            return name.contains("swiftpm-testing") || name.contains("xctest")
                || name.contains("rapidtests")
        }
        for child in runners.prefix(2) {
            try write("\n===== /usr/bin/sample child \(child.pid) \(config.sampleDurationSeconds) =====\n")
            try write(streamingSample(pid: child.pid, seconds: config.sampleDurationSeconds))
        }
        try write("===== end capture \(pid) =====\n")
    }

    struct ProcessEntry: Equatable {
        let pid: pid_t
        let parentPID: pid_t
        let summary: String

        var command: String {
            summary.split(maxSplits: 4, whereSeparator: \.isWhitespace).last.map(String.init) ?? ""
        }
    }

    /// Return only descendants of the wrapped process. The cap prevents a
    /// runaway process tree from filling the CI artifact or delaying cleanup.
    static func descendantProcesses(
        of rootPID: pid_t,
        processTable: String,
        limit: Int = 32
    ) -> [ProcessEntry] {
        let entries: [ProcessEntry] = processTable.split(separator: "\n").compactMap { line in
            let fields = line.split(maxSplits: 4, whereSeparator: \.isWhitespace)
            guard fields.count == 5,
                  let pid = pid_t(fields[0]),
                  let parent = pid_t(fields[1])
            else { return nil }
            return ProcessEntry(
                pid: pid,
                parentPID: parent,
                summary: fields.map(String.init).joined(separator: " ")
            )
        }
        var known: Set<pid_t> = [rootPID]
        var descendants: [ProcessEntry] = []
        var cursor = 0
        var parents = [rootPID]
        while cursor < parents.count && descendants.count < max(0, limit) {
            let parent = parents[cursor]
            cursor += 1
            for entry in entries where entry.parentPID == parent && !known.contains(entry.pid) {
                known.insert(entry.pid)
                parents.append(entry.pid)
                descendants.append(entry)
                if descendants.count >= limit { break }
            }
        }
        return descendants
    }

    /// `/usr/bin/sample` streams to stdout; capture it into the artifact.
    private static func streamingSample(pid: pid_t, seconds: Int) -> String {
        let process = Process()
        let pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/sample")
        process.arguments = [String(pid), String(seconds)]
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            let maxSampleBytes = 256 * 1024
            let captured = String(decoding: data.prefix(maxSampleBytes), as: UTF8.self)
            return data.count > maxSampleBytes
                ? captured + "\n<sample truncated at \(maxSampleBytes) bytes>\n"
                : captured
        } catch {
            return "<sample failed: \(error)>"
        }
    }

    /// Run a command and return its combined stdout (best-effort, non-fatal).
    private static func shellOutput(_ executable: String, _ args: [String]) -> String {
        let process = Process()
        let pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = args
        process.standardOutput = pipe
        process.standardError = pipe
        do {
            try process.run()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            return String(decoding: data, as: UTF8.self)
        } catch {
            return "<'\(executable)' failed: \(error)>"
        }
    }
}
