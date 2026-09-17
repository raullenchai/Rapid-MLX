import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Local workspace tools")
final class LocalWorkspaceToolsTests {
    nonisolated(unsafe) private var suiteNames: [String] = []
    deinit { TestDefaultsScope.cleanup(suiteNames: suiteNames) }

    private func approval() -> LocalToolApprovalStore {
        let name = TestDefaultsScope.mintSuiteName(prefix: "rapid-local-tools-")
        suiteNames.append(name)
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        return LocalToolApprovalStore(defaults: defaults)
    }

    private func fixtureDirectory() throws -> URL {
        let root = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Caches/Rapid-MLXTests", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func runApproved(
        name: String,
        arguments: String,
        store: LocalToolApprovalStore
    ) async -> ToolCallResult {
        let task = Task {
            await LocalWorkspaceTools.run(
                ToolCall(id: "test-call", name: name, arguments: arguments),
                approval: store
            )
        }
        while store.pendingRequest == nil { await Task.yield() }
        store.answer(.allowOnce)
        return await task.value
    }

    @Test("search finds text locally and never needs a web fallback")
    func searchFindsContent() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try "Project Orchid meets Thursday in Redwood.".write(
            to: root.appendingPathComponent("notes.txt"), atomically: true, encoding: .utf8
        )
        let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "path": root.path, "query": "Project Orchid",
        ]), encoding: .utf8))

        let result = await runApproved(name: "local_search", arguments: arguments, store: approval())

        #expect(!result.isError)
        #expect(result.content.contains("notes.txt"))
        #expect(result.content.contains("Thursday"))
    }

    @Test("search does not follow a symlink outside the approved folder")
    func searchSkipsSymlinkDescendants() async throws {
        let root = try fixtureDirectory()
        let outside = try fixtureDirectory()
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: outside)
        }
        try "RAPID_PRIVATE_NEEDLE".write(
            to: outside.appendingPathComponent("private.txt"),
            atomically: true,
            encoding: .utf8
        )
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("linked-folder"),
            withDestinationURL: outside
        )
        let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "path": root.path, "query": "RAPID_PRIVATE_NEEDLE",
        ]), encoding: .utf8))

        let result = await runApproved(name: "local_search", arguments: arguments, store: approval())

        #expect(!result.isError)
        #expect(!result.content.contains("private.txt"))
        #expect(result.content.contains(#""matches":[]"#))
    }

    @Test("write requires approval and creates the exact UTF-8 file")
    func writeCreatesExactFile() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let output = root.appendingPathComponent("proposal.md")
        let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "path": output.path, "content": "# Proposal\nLocal first.\n",
        ]), encoding: .utf8))
        let store = approval()

        let result = await runApproved(name: "local_write", arguments: arguments, store: store)

        #expect(!result.isError)
        #expect(try String(contentsOf: output, encoding: .utf8) == "# Proposal\nLocal first.\n")
        #expect(!store.isGranted("local_write"))
    }

    @Test("write never replaces an existing file without overwrite")
    func writeUsesExclusiveCreate() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let output = root.appendingPathComponent("existing.txt")
        try "original".write(to: output, atomically: true, encoding: .utf8)
        let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "path": output.path, "content": "replacement",
        ]), encoding: .utf8))

        let result = await runApproved(name: "local_write", arguments: arguments, store: approval())

        #expect(result.isError)
        #expect(try String(contentsOf: output, encoding: .utf8) == "original")
    }

    @Test("overwrite replaces a symlink instead of writing through it")
    func writeDoesNotFollowDestinationSymlink() async throws {
        let root = try fixtureDirectory()
        let outside = try fixtureDirectory()
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: outside)
        }
        let target = outside.appendingPathComponent("target.txt")
        let output = root.appendingPathComponent("output.txt")
        try "outside-original".write(to: target, atomically: true, encoding: .utf8)
        try FileManager.default.createSymbolicLink(at: output, withDestinationURL: target)
        let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "path": output.path, "content": "approved-output", "overwrite": true,
        ]), encoding: .utf8))

        let result = await runApproved(name: "local_write", arguments: arguments, store: approval())

        #expect(!result.isError, Comment(rawValue: result.content))
        #expect(try String(contentsOf: output, encoding: .utf8) == "approved-output")
        #expect(try String(contentsOf: target, encoding: .utf8) == "outside-original")
        #expect((try output.resourceValues(forKeys: [.isSymbolicLinkKey])).isSymbolicLink != true)
    }

    @Test("run uses argv without a shell and captures output")
    func commandRunsWithoutShell() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "command": "python3",
            "arguments": ["-c", "print('RAPID_LOCAL_OK')"],
            "working_directory": root.path,
            "timeout_seconds": 5,
        ]), encoding: .utf8))

        let result = await runApproved(name: "local_run", arguments: arguments, store: approval())

        #expect(!result.isError)
        #expect(result.content.contains("exit_code: 0"))
        #expect(result.content.contains("RAPID_LOCAL_OK"))
    }

    @Test("run drains output written immediately before process exit")
    func commandPreservesTrailingOutput() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "command": "python3",
            "arguments": ["-c", "import os;os.write(1,b'RAPID_TRAILING_BYTES')"],
            "working_directory": root.path,
        ]), encoding: .utf8))

        let result = await runApproved(name: "local_run", arguments: arguments, store: approval())

        #expect(!result.isError)
        #expect(result.content.contains("RAPID_TRAILING_BYTES"))
    }

    @Test("run compiles C inside the approved workspace sandbox")
    func commandCompilesC() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try "#include <stdio.h>\nint main(void){puts(\"RAPID_C_OK\");return 0;}\n".write(
            to: root.appendingPathComponent("main.c"),
            atomically: true,
            encoding: .utf8
        )
        let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "command": "gcc",
            "arguments": ["main.c", "-o", "main"],
            "working_directory": root.path,
        ]), encoding: .utf8))

        let result = await runApproved(name: "local_run", arguments: arguments, store: approval())

        #expect(!result.isError, Comment(rawValue: result.content))
        #expect(FileManager.default.isExecutableFile(atPath: root.appendingPathComponent("main").path))
    }

    @Test("run sandbox blocks reads elsewhere in the home folder")
    func commandCannotReadOutsideWorkingDirectory() async throws {
        let root = try fixtureDirectory()
        let outside = try fixtureDirectory()
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: outside)
        }
        let secret = outside.appendingPathComponent("secret.txt")
        try "RAPID_SANDBOX_SECRET".write(to: secret, atomically: true, encoding: .utf8)
        let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "command": "python3",
            "arguments": ["-c", "print(open(\"\(secret.path)\").read())"],
            "working_directory": root.path,
        ]), encoding: .utf8))

        let result = await runApproved(name: "local_run", arguments: arguments, store: approval())

        #expect(result.isError)
        #expect(!result.content.contains("RAPID_SANDBOX_SECRET"))
    }

    @Test("run captures at most 64 KB per output stream")
    func commandOutputIsBounded() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "command": "python3",
            "arguments": ["-c", "import sys;sys.stdout.write('x'*200000)"],
            "working_directory": root.path,
        ]), encoding: .utf8))

        let result = await runApproved(name: "local_run", arguments: arguments, store: approval())

        #expect(!result.isError)
        #expect(result.content.utf8.count < 66_000)
    }

    @Test("run force-stops a process that ignores termination")
    func commandTimeoutIsBounded() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "command": "python3",
            "arguments": [
                "-c",
                "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(10)",
            ],
            "working_directory": root.path,
            "timeout_seconds": 1,
        ]), encoding: .utf8))
        let started = Date()

        let result = await runApproved(name: "local_run", arguments: arguments, store: approval())

        #expect(result.isError)
        #expect(result.content.contains("timed out"))
        #expect(Date().timeIntervalSince(started) < 3)
    }

    @Test("run timeout stops child processes in the approved process group")
    func commandTimeoutStopsChildren() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let marker = root.appendingPathComponent("escaped-child.txt")
        let child = "import time;time.sleep(2);open('escaped-child.txt','w').write('escaped')"
        let parent = "import signal,subprocess,time;subprocess.Popen(['python3','-c',\"\(child)\"]);signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(10)"
        let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "command": "python3",
            "arguments": ["-c", parent],
            "working_directory": root.path,
            "timeout_seconds": 1,
        ]), encoding: .utf8))

        let result = await runApproved(name: "local_run", arguments: arguments, store: approval())
        try await Task.sleep(for: .seconds(2))

        #expect(result.isError)
        #expect(result.content.contains("timed out"))
        #expect(!FileManager.default.fileExists(atPath: marker.path))
    }

    @Test("paths outside the user's home fail closed before approval")
    func outsideHomeIsRejected() async {
        let store = approval()
        let result = await LocalWorkspaceTools.run(
            ToolCall(
                id: "test-call",
                name: "local_read",
                arguments: #"{"path":"/etc/hosts"}"#
            ),
            approval: store
        )
        #expect(result.isError)
        #expect(result.content.contains("inside"))
        #expect(store.pendingRequest == nil)
    }
}
