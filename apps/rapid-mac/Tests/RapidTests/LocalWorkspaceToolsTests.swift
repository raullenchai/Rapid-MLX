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
