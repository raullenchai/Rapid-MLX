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
        // A tool that answers before asking for approval (a preflight
        // refusal) must surface as a result, not as an unbounded spin: the
        // 2026-09-18 CI hang sat in this loop with nothing left to run.
        final class Outcome { var result: ToolCallResult? }
        let outcome = Outcome()
        let task = Task {
            let result = await LocalWorkspaceTools.run(
                ToolCall(id: "test-call", name: name, arguments: arguments),
                approval: store
            )
            outcome.result = result
            return result
        }
        let deadline = ContinuousClock.now + .seconds(30)
        while store.pendingRequest == nil {
            if let early = outcome.result { return early }
            if ContinuousClock.now > deadline {
                task.cancel()
                return ToolCallResult(
                    toolCallID: "test-call",
                    content: "\(name) never requested approval within 30 s",
                    isError: true,
                    executed: false
                )
            }
            try? await Task.sleep(for: .milliseconds(2))
        }
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

    @Test("search matches every query word, not only the exact phrase")
    func searchMatchesAllWordsOfTheQuery() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        // Neither the file name nor the text contains "orchid notes" as one
        // phrase; the words are hyphenated in the name and apart in the body.
        try "Notes on the orchid meeting: Thursday 3:30 PM in Redwood.".write(
            to: root.appendingPathComponent("orchid-notes.txt"), atomically: true, encoding: .utf8
        )
        try "Nothing about flowers here.".write(
            to: root.appendingPathComponent("other.txt"), atomically: true, encoding: .utf8
        )
        let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "path": root.path, "query": "orchid notes",
        ]), encoding: .utf8))

        let result = await runApproved(name: "local_search", arguments: arguments, store: approval())

        #expect(!result.isError)
        #expect(result.content.contains("orchid-notes.txt"))
        #expect(result.content.contains("Thursday"))
        #expect(!result.content.contains("other.txt"))
        #expect(LocalWorkspaceTools.searchTerms(for: "Orchid, notes!") == ["orchid", "notes"])
        // A single word never widens into an all-words match, and a query
        // missing one word from the text is not a match.
        #expect(!LocalWorkspaceTools.matchesAllTerms(["orchid"], in: "orchid"))
        #expect(!LocalWorkspaceTools.matchesAllTerms(["orchid", "cactus"], in: "orchid notes"))
        #expect(!LocalWorkspaceTools.matchesAllTerms(["art", "note"], in: "party notebook"))
        #expect(LocalWorkspaceTools.snippetRange(query: "art note", terms: ["art", "note"], in: "party notebook") == nil)
        #expect(LocalWorkspaceTools.snippetRange(query: "orchid cactus", terms: ["orchid", "cactus"], in: "orchid notes") == nil)
        let manyTerms = (0..<200).map { "term\($0)" }
        let nearLimitText = String(repeating: "padding ", count: 100_000)
            + manyTerms.joined(separator: " filler ")
        let longQueryRange = try #require(LocalWorkspaceTools.snippetRange(
            query: manyTerms.joined(separator: " "),
            terms: manyTerms,
            in: nearLimitText
        ))
        #expect(String(nearLimitText[longQueryRange]) == "term0")
    }

    @Test("run expands a leading ~/ in argv like the shell the model imitates")
    func commandExpandsHomeInArguments() throws {
        let home = URL(fileURLWithPath: "/Users/example")
        #expect(LocalWorkspaceTools.expandingHomeArguments(
            ["-o", "~/Documents/app", "~/Documents/app.c", "~notme", "-Wall", "~"],
            command: "clang",
            home: home
        ) == ["-o", "/Users/example/Documents/app", "/Users/example/Documents/app.c", "~notme", "-Wall", "~"])
        #expect(LocalWorkspaceTools.expandingHomeArguments(
            ["-c", "print('~/literal')"], command: "python3", home: home
        ) == ["-c", "print('~/literal')"])
        #expect(LocalWorkspaceTools.expandingHomeArguments(
            ["~/Documents/app.py", "~/literal"], command: "python3", home: home
        ) == ["/Users/example/Documents/app.py", "~/literal"])
        #expect(LocalWorkspaceTools.expandingHomeArguments(
            ["-W", "ignore", "~/Documents/app.py"], command: "python3", home: home
        ) == ["-W", "ignore", "/Users/example/Documents/app.py"])
        #expect(LocalWorkspaceTools.expandingHomeArguments(
            ["~/Documents/app.py"], command: "/usr/bin/python3", home: home
        ) == ["/Users/example/Documents/app.py"])
        #expect(LocalWorkspaceTools.expandingHomeArguments(
            ["~/Documents/app.c"], command: "/usr/bin/clang", home: home
        ) == ["/Users/example/Documents/app.c"])
        #expect(LocalWorkspaceTools.expandingHomeArguments(
            ["-D", "~/literal", "-I", "~/Documents/include", "~/Documents/app.c"],
            command: "clang",
            home: home
        ) == ["-D", "~/literal", "-I", "/Users/example/Documents/include", "/Users/example/Documents/app.c"])
        #expect(LocalWorkspaceTools.expandingHomeArguments(
            ["-I~/Documents/include", "-F~/Documents/frameworks", "-o~/Documents/app", "~/Documents/app.c"],
            command: "clang",
            home: home
        ) == ["-I/Users/example/Documents/include", "-F/Users/example/Documents/frameworks", "-o/Users/example/Documents/app", "/Users/example/Documents/app.c"])
        #expect(LocalWorkspaceTools.expandingHomeArguments(
            ["run", "~/Documents/project"], command: "go", home: home
        ) == ["run", "/Users/example/Documents/project"])
        #expect(LocalWorkspaceTools.expandingHomeArguments(
            ["run", "~/Documents/main.go", "~/literal"], command: "go", home: home
        ) == ["run", "/Users/example/Documents/main.go", "~/literal"])
    }

    @Test("tool results report paths relative to the home directory")
    func resultsReportHomeRelativePaths() throws {
        let home = URL(fileURLWithPath: "/Users/example")
        #expect(LocalWorkspaceTools.displayPath(URL(fileURLWithPath: "/Users/example/Documents/winter.md"), home: home) == "~/Documents/winter.md")
        #expect(LocalWorkspaceTools.displayPath(URL(fileURLWithPath: "/Users/example"), home: home) == "~")
        #expect(LocalWorkspaceTools.displayPath(URL(fileURLWithPath: "/Users/examples/x.txt"), home: home) == "/Users/examples/x.txt")
        #expect(LocalWorkspaceTools.displayPath(URL(fileURLWithPath: "/tmp/x.txt"), home: home) == "/tmp/x.txt")
    }

    @Test("run still accepts the pre-0.14.3 arguments key and the args spelling")
    func commandAcceptsLegacyArgumentKeys() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        for key in ["arguments", "args"] {
            let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
                "command": "python3",
                key: ["-c", "print('RAPID_LEGACY_OK')"],
                "working_directory": root.path,
                "timeout_seconds": 5,
            ]), encoding: .utf8))

            let result = await runApproved(name: "local_run", arguments: arguments, store: approval())

            #expect(!result.isError, "key \(key)")
            #expect(result.content.contains("RAPID_LEGACY_OK"), "key \(key)")
        }
    }

    @Test("run rejects a present argument list with the wrong wire type")
    func commandRejectsMalformedArgumentList() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "command": "python3",
            "argv": "-c print('must not run')",
            "working_directory": root.path,
        ]), encoding: .utf8))

        let result = await runApproved(
            name: "local_run", arguments: arguments, store: approval()
        )

        #expect(result.isError)
        #expect(result.content == "local_run arguments are invalid")
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

    @Test("search excludes protected descendants of an approved ancestor")
    func searchSkipsProtectedDescendants() {
        let home = FileManager.default.homeDirectoryForCurrentUser
        #expect(LocalWorkspaceTools.isProtectedSearchURL(
            home.appendingPathComponent("Library/Keychains/login.keychain-db")
        ))
        #expect(!LocalWorkspaceTools.isProtectedSearchURL(
            home.appendingPathComponent("Library/Notes/notes.sqlite")
        ))
    }

    @Test("only immutable system and selected toolchain executables run in place")
    func packageManagerExecutablesAreStaged() {
        let developer = URL(fileURLWithPath: "/Applications/Xcode.app/Contents/Developer")
        #expect(LocalWorkspaceTools.isImmutableSystemExecutable(
            URL(fileURLWithPath: "/usr/bin/python3"), developerDirectories: [developer]
        ))
        #expect(LocalWorkspaceTools.isImmutableSystemExecutable(
            developer.appendingPathComponent("usr/bin/swift"), developerDirectories: [developer]
        ))
        #expect(!LocalWorkspaceTools.isImmutableSystemExecutable(
            URL(fileURLWithPath: "/opt/homebrew/Cellar/node/24/bin/node"),
            developerDirectories: [developer]
        ))
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
        #expect(result.toolCallID == "test-call")
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

    @Test("write never replaces an existing directory")
    func writeRejectsDirectoryDestination() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let output = root.appendingPathComponent("existing", isDirectory: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
        let child = output.appendingPathComponent("keep.txt")
        try "keep".write(to: child, atomically: true, encoding: .utf8)
        let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "path": output.path, "content": "replacement", "overwrite": true,
        ]), encoding: .utf8))

        let result = await runApproved(name: "local_write", arguments: arguments, store: approval())

        #expect(result.isError)
        #expect(result.content.contains("regular files or symbolic links"))
        #expect(try String(contentsOf: child, encoding: .utf8) == "keep")
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

    @Test("write rejects a destination replaced while approval is open")
    func writePinsDestinationIdentity() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let output = root.appendingPathComponent("output.txt")
        try "approved-original".write(to: output, atomically: true, encoding: .utf8)
        let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "path": output.path, "content": "approved-output", "overwrite": true,
        ]), encoding: .utf8))
        let store = approval()
        let task = Task {
            await LocalWorkspaceTools.run(
                ToolCall(id: "write", name: "local_write", arguments: arguments), approval: store
            )
        }
        while store.pendingRequest == nil { await Task.yield() }
        try FileManager.default.removeItem(at: output)
        try "replacement".write(to: output, atomically: true, encoding: .utf8)
        store.answer(.allowOnce)

        let result = await task.value
        #expect(result.isError)
        #expect(result.content.contains("destination changed"))
        #expect(try String(contentsOf: output, encoding: .utf8) == "replacement")
    }

    @Test("write rejects a parent folder replaced while approval is open")
    func writePinsParentIdentity() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let parent = root.appendingPathComponent("approved", isDirectory: true)
        let moved = root.appendingPathComponent("moved", isDirectory: true)
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: false)
        let output = parent.appendingPathComponent("output.txt")
        let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "path": output.path, "content": "must-not-write",
        ]), encoding: .utf8))
        let store = approval()
        let task = Task {
            await LocalWorkspaceTools.run(
                ToolCall(id: "write", name: "local_write", arguments: arguments), approval: store
            )
        }
        while store.pendingRequest == nil { await Task.yield() }
        try FileManager.default.moveItem(at: parent, to: moved)
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: false)
        store.answer(.allowOnce)

        let result = await task.value
        #expect(result.isError)
        #expect(!FileManager.default.fileExists(atPath: output.path))
        #expect(!FileManager.default.fileExists(atPath: moved.appendingPathComponent("output.txt").path))
    }

    @Test("run uses argv without a shell and captures output")
    func commandRunsWithoutShell() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "command": "python3",
            "argv": ["-c", "print('RAPID_LOCAL_OK')"],
            "working_directory": root.path,
            "timeout_seconds": 5,
        ]), encoding: .utf8))

        let result = await runApproved(name: "local_run", arguments: arguments, store: approval())

        #expect(!result.isError)
        #expect(result.content.contains("exit_code: 0"))
        #expect(result.content.contains("RAPID_LOCAL_OK"))
    }

    @Test("run rejects make because Makefiles execute shell recipes")
    func commandRejectsMake() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "command": "make",
            "argv": ["--version"],
            "working_directory": root.path,
        ]), encoding: .utf8))

        let store = approval()
        let result = await LocalWorkspaceTools.run(
            ToolCall(id: "make", name: "local_run", arguments: arguments), approval: store
        )

        #expect(result.isError)
        #expect(store.pendingRequest == nil)
    }

    @Test("run refuses a workspace that contains protected folders")
    func commandCannotExposeHomeAsWorkspace() async {
        let store = approval()
        let result = await LocalWorkspaceTools.run(
            ToolCall(
                id: "home",
                name: "local_run",
                arguments: #"{"command":"python3","working_directory":"~"}"#
            ),
            approval: store
        )

        #expect(result.isError)
        #expect(result.content.contains("protected"))
        #expect(store.pendingRequest == nil)
    }

    @Test("run drains output written immediately before process exit")
    func commandPreservesTrailingOutput() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "command": "python3",
            "argv": ["-c", "import os;os.write(1,b'RAPID_TRAILING_BYTES')"],
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
            "argv": ["main.c", "-o", "main"],
            "working_directory": root.path,
        ]), encoding: .utf8))

        let result = await runApproved(name: "local_run", arguments: arguments, store: approval())

        #expect(!result.isError, Comment(rawValue: result.content))
        #expect(FileManager.default.isExecutableFile(atPath: root.appendingPathComponent("main").path))
    }

    @Test("compiler rejects in-process plugin and indirect argument escapes before approval", arguments: [
        ["-Xclang", "-load", "-Xclang", "plugin.dylib", "main.c"],
        ["-fplugin=plugin.dylib", "main.c"],
        ["-fpass-plugin=plugin.dylib", "main.c"],
        ["@workspace-flags.rsp"],
        ["--config=workspace.cfg", "main.c"],
        ["-cc1", "-load", "plugin.dylib", "main.c"],
        ["-Xlinker", "-plugin", "main.c"],
        ["-Wl,-plugin,plugin.dylib", "main.c"],
        ["-mllvm", "-load=plugin.dylib", "main.c"],
    ])
    func compilerCannotLoadWorkspaceCode(arguments: [String]) async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let payload = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "command": "clang",
            "arguments": arguments,
            "working_directory": root.path,
        ]), encoding: .utf8))
        let store = approval()

        let result = await LocalWorkspaceTools.run(
            ToolCall(id: "compiler-escape", name: "local_run", arguments: payload), approval: store
        )

        #expect(result.isError)
        #expect(result.content.contains("plugins and indirect argument files"))
        #expect(store.pendingRequest == nil)
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
            "argv": ["-c", "print(open(\"\(secret.path)\").read())"],
            "working_directory": root.path,
        ]), encoding: .utf8))

        let result = await runApproved(name: "local_run", arguments: arguments, store: approval())

        #expect(result.isError)
        #expect(!result.content.contains("RAPID_SANDBOX_SECRET"))
    }

    @Test("run sandbox blocks reads outside the home and approved workspace")
    func commandCannotReadMachineFiles() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "command": "python3",
            "argv": ["-c", "print(open('/etc/hosts').read())"],
            "working_directory": root.path,
        ]), encoding: .utf8))

        let result = await runApproved(name: "local_run", arguments: arguments, store: approval())

        #expect(result.isError)
        #expect(!result.content.contains("localhost"))
    }

    @Test("run sandbox blocks machine data under Library")
    func commandCannotReadLibraryData() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "command": "python3",
            "argv": ["-c", "import os;print(os.listdir('/Library'))"],
            "working_directory": root.path,
        ]), encoding: .utf8))

        let result = await runApproved(name: "local_run", arguments: arguments, store: approval())

        #expect(result.isError)
    }

    @Test("approved interpreter cannot launch an unrelated executable")
    func commandCannotLaunchArbitraryExecutable() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let marker = root.appendingPathComponent("escaped.txt")
        let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "command": "python3",
            "argv": ["-c", "import subprocess;subprocess.run(['/usr/bin/touch','escaped.txt'],check=True)"],
            "working_directory": root.path,
        ]), encoding: .utf8))

        let result = await runApproved(name: "local_run", arguments: arguments, store: approval())

        #expect(result.isError)
        #expect(!FileManager.default.fileExists(atPath: marker.path))
    }

    @Test("run rejects a working directory retargeted while approval is open")
    func commandPinsApprovedWorkingDirectory() async throws {
        let first = try fixtureDirectory()
        let second = try fixtureDirectory()
        let link = first.deletingLastPathComponent().appendingPathComponent(UUID().uuidString)
        defer {
            try? FileManager.default.removeItem(at: link)
            try? FileManager.default.removeItem(at: first)
            try? FileManager.default.removeItem(at: second)
        }
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: first)
        let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "command": "python3",
            "argv": ["-c", "print('SHOULD_NOT_RUN')"],
            "working_directory": link.path,
        ]), encoding: .utf8))
        let store = approval()
        let task = Task {
            await LocalWorkspaceTools.run(
                ToolCall(id: "run", name: "local_run", arguments: arguments), approval: store
            )
        }
        while store.pendingRequest == nil { await Task.yield() }
        try FileManager.default.removeItem(at: link)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: second)
        store.answer(.allowOnce)

        let result = await task.value
        #expect(result.isError)
        #expect(result.content.contains("approved file changed"))
        #expect(!result.content.contains("SHOULD_NOT_RUN"))
    }

    @Test("run rejects executable bytes changed while approval is open")
    func commandPinsApprovedExecutableDigest() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let executable = root.appendingPathComponent("local-tool")
        let original = Data("#!/bin/sh\necho ORIGINAL\n".utf8)
        try original.write(to: executable)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o700], ofItemAtPath: executable.path
        )
        let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "command": executable.path,
            "working_directory": root.path,
        ]), encoding: .utf8))
        let store = approval()
        let task = Task {
            await LocalWorkspaceTools.run(
                ToolCall(id: "run", name: "local_run", arguments: arguments), approval: store
            )
        }
        while store.pendingRequest == nil { await Task.yield() }
        let handle = try FileHandle(forWritingTo: executable)
        try handle.seek(toOffset: 0)
        try handle.write(contentsOf: Data(repeating: 65, count: original.count))
        try handle.close()
        store.answer(.allowOnce)

        let result = await task.value
        #expect(result.isError)
        #expect(result.content.contains("changed after approval"))
        #expect(!result.content.contains("ORIGINAL"))
    }

    @Test("protected paths are rejected case-insensitively before approval")
    func protectedPathCaseDoesNotBypassGuard() async {
        let store = approval()
        let result = await LocalWorkspaceTools.run(
            ToolCall(
                id: "test-call",
                name: "local_read",
                arguments: #"{"path":"~/.SSH/id_rsa"}"#
            ),
            approval: store
        )
        #expect(result.isError)
        #expect(result.content.contains("protected"))
        #expect(store.pendingRequest == nil)
    }

    @Test("all hidden paths are rejected before approval")
    func hiddenPathsAreRejected() async {
        let store = approval()
        let result = await LocalWorkspaceTools.run(
            ToolCall(
                id: "hidden-read",
                name: "local_read",
                arguments: #"{"path":"~/.aws/credentials"}"#
            ),
            approval: store
        )
        #expect(result.isError)
        #expect(result.toolCallID == "hidden-read")
        #expect(result.content.contains("hidden"))
        #expect(store.pendingRequest == nil)
    }

    @Test("session read grants stay scoped to the approved path")
    func readGrantDoesNotAuthorizeAnotherFile() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let first = root.appendingPathComponent("first.txt")
        let second = root.appendingPathComponent("second.txt")
        try "one".write(to: first, atomically: true, encoding: .utf8)
        try "two".write(to: second, atomically: true, encoding: .utf8)
        let store = approval()
        let firstArguments = #"{"path":"\#(first.path)"}"#

        let initial = Task {
            await LocalWorkspaceTools.run(
                ToolCall(id: "first", name: "local_read", arguments: firstArguments),
                approval: store
            )
        }
        while store.pendingRequest == nil { await Task.yield() }
        store.answer(.alwaysAllowTool)
        #expect(!(await initial.value).isError)

        // The exact file is session-approved and returns without prompting.
        let repeated = await LocalWorkspaceTools.run(
            ToolCall(id: "repeat", name: "local_read", arguments: firstArguments),
            approval: store
        )
        #expect(!repeated.isError)
        #expect(store.pendingRequest == nil)

        // A different file must still stop at a fresh consent sheet.
        let secondTask = Task {
            await LocalWorkspaceTools.run(
                ToolCall(
                    id: "second",
                    name: "local_read",
                    arguments: #"{"path":"\#(second.path)"}"#
                ),
                approval: store
            )
        }
        while store.pendingRequest == nil { await Task.yield() }
        store.answer(.deny)
        #expect((await secondTask.value).isError)
    }

    @Test("session read grants do not authorize a replacement at the same path")
    func readGrantPinsGrantedIdentity() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("notes.txt")
        let displaced = root.appendingPathComponent("approved.txt")
        try "approved".write(to: file, atomically: true, encoding: .utf8)
        let store = approval()
        let arguments = #"{"path":"\#(file.path)"}"#

        let initial = Task {
            await LocalWorkspaceTools.run(
                ToolCall(id: "first", name: "local_read", arguments: arguments),
                approval: store
            )
        }
        while store.pendingRequest == nil { await Task.yield() }
        store.answer(.alwaysAllowTool)
        #expect(!(await initial.value).isError)

        try FileManager.default.moveItem(at: file, to: displaced)
        try "replacement".write(to: file, atomically: true, encoding: .utf8)
        let replacement = Task {
            await LocalWorkspaceTools.run(
                ToolCall(id: "second", name: "local_read", arguments: arguments),
                approval: store
            )
        }
        while store.pendingRequest == nil { await Task.yield() }
        #expect(store.pendingRequest?.toolName == "local_read")
        store.answer(.deny)
        #expect((await replacement.value).failureKind == .userDeclined)
    }

    @Test("session read grants cannot follow a retargeted symlink")
    func readGrantDoesNotFollowRetargetedSymlink() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let first = root.appendingPathComponent("first.txt")
        let second = root.appendingPathComponent("second.txt")
        let link = root.appendingPathComponent("current.txt")
        try "one".write(to: first, atomically: true, encoding: .utf8)
        try "two".write(to: second, atomically: true, encoding: .utf8)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: first)
        let store = approval()
        let arguments = #"{"path":"\#(link.path)"}"#

        let initial = Task {
            await LocalWorkspaceTools.run(
                ToolCall(id: "first", name: "local_read", arguments: arguments),
                approval: store
            )
        }
        while store.pendingRequest == nil { await Task.yield() }
        store.answer(.alwaysAllowTool)
        #expect(!(await initial.value).isError)

        try FileManager.default.removeItem(at: link)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: second)
        let retargeted = Task {
            await LocalWorkspaceTools.run(
                ToolCall(id: "second", name: "local_read", arguments: arguments),
                approval: store
            )
        }
        while store.pendingRequest == nil { await Task.yield() }
        #expect(store.pendingRequest?.toolName == "local_read")
        store.answer(.deny)
        #expect((await retargeted.value).failureKind == .userDeclined)
    }

    @Test("read rejects a file replaced while approval is open")
    func readPinsApprovedFileIdentity() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("notes.txt")
        let original = root.appendingPathComponent("original.txt")
        try "approved".write(to: file, atomically: true, encoding: .utf8)
        let store = approval()
        let task = Task {
            await LocalWorkspaceTools.run(
                ToolCall(id: "read", name: "local_read", arguments: #"{"path":"\#(file.path)"}"#),
                approval: store
            )
        }
        while store.pendingRequest == nil { await Task.yield() }
        try FileManager.default.moveItem(at: file, to: original)
        try "replacement".write(to: file, atomically: true, encoding: .utf8)
        store.answer(.allowOnce)

        let result = await task.value
        #expect(result.isError)
        #expect(result.content.contains("approved file changed"))
        #expect(!result.content.contains("replacement"))
    }

    @Test("trash rejects symbolic links before approval")
    func trashRejectsSymbolicLinks() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let target = root.appendingPathComponent("target.txt")
        let link = root.appendingPathComponent("current.txt")
        try "one".write(to: target, atomically: true, encoding: .utf8)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
        let store = approval()
        let result = await LocalWorkspaceTools.run(
            ToolCall(
                id: "trash", name: "local_trash",
                arguments: #"{"path":"\#(link.path)"}"#
            ),
            approval: store
        )
        #expect(result.isError)
        #expect(result.content.contains("symbolic links"))
        #expect(store.pendingRequest == nil)
        #expect(FileManager.default.fileExists(atPath: target.path))
    }

    @Test("trash rejects a parent folder replaced while approval is open")
    func trashPinsParentIdentity() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let parent = root.appendingPathComponent("approved", isDirectory: true)
        let moved = root.appendingPathComponent("moved", isDirectory: true)
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: false)
        let file = parent.appendingPathComponent("notes.txt")
        try "approved".write(to: file, atomically: true, encoding: .utf8)
        let store = approval()
        let task = Task {
            await LocalWorkspaceTools.run(
                ToolCall(
                    id: "trash-parent", name: "local_trash",
                    arguments: #"{"path":"\#(file.path)"}"#
                ),
                approval: store
            )
        }
        while store.pendingRequest == nil { await Task.yield() }
        try FileManager.default.moveItem(at: parent, to: moved)
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: false)
        let replacement = parent.appendingPathComponent("notes.txt")
        try "replacement".write(to: replacement, atomically: true, encoding: .utf8)
        store.answer(.allowOnce)

        let result = await task.value
        #expect(result.isError)
        #expect(result.content.contains("approved file changed"))
        #expect(try String(contentsOf: replacement, encoding: .utf8) == "replacement")
        #expect(FileManager.default.fileExists(atPath: moved.appendingPathComponent("notes.txt").path))
    }

    @Test("read rejects files over its hard byte limit")
    func readRejectsOversizedFiles() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("large.txt")
        try Data(repeating: 65, count: 512_001).write(to: file)
        let arguments = #"{"path":"\#(file.path)"}"#

        let result = await runApproved(name: "local_read", arguments: arguments, store: approval())

        #expect(result.isError)
        #expect(result.content.contains("exceeds 512 KB"))
    }

    @Test("run captures at most 64 KB per output stream")
    func commandOutputIsBounded() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "command": "python3",
            "argv": ["-c", "import sys;sys.stdout.write('x'*200000)"],
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
            "argv": [
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

    @Test("approved interpreters cannot fork detached background work")
    func commandCannotForkDetachedChildren() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let marker = root.appendingPathComponent("escaped-child.txt")
        let child = "import time;time.sleep(1);open('escaped-child.txt','w').write('escaped')"
        let parent = "import subprocess;subprocess.Popen(['python3','-c',\"\(child)\"],start_new_session=True)"
        let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "command": "python3",
            "argv": ["-c", parent],
            "working_directory": root.path,
            "timeout_seconds": 5,
        ]), encoding: .utf8))

        let result = await runApproved(name: "local_run", arguments: arguments, store: approval())
        try await Task.sleep(for: .seconds(1.5))

        #expect(result.isError)
        #expect(!FileManager.default.fileExists(atPath: marker.path))
    }

    @Test("approved Swift scripts cannot fork detached background work",
          .enabled(if: LocalWorkspaceTools.swiftScriptToolchainIsAvailable(),
                   "no root-owned Xcode/CLT swift toolchain on this Mac (CI runner)"))
    func swiftCommandCannotForkDetachedChildren() async throws {
        let root = try fixtureDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let marker = root.appendingPathComponent("escaped-swift-child.txt")
        let script = """
        import Darwin
        let child = fork()
        if child < 0 { print("FORK_BLOCKED"); exit(7) }
        if child == 0 {
            _ = setsid()
            sleep(1)
            let fd = open("escaped-swift-child.txt", O_WRONLY | O_CREAT, 0o600)
            if fd >= 0 {
                let bytes = Array("escaped".utf8)
                _ = bytes.withUnsafeBytes { write(fd, $0.baseAddress, $0.count) }
                close(fd)
            }
            exit(0)
        }
        """
        try script.write(
            to: root.appendingPathComponent("fork-attempt.swift"),
            atomically: true,
            encoding: .utf8
        )
        let arguments = try #require(String(data: JSONSerialization.data(withJSONObject: [
            "command": "swift",
            "argv": ["fork-attempt.swift"],
            "working_directory": root.path,
            "timeout_seconds": 5,
        ]), encoding: .utf8))

        let result = await runApproved(name: "local_run", arguments: arguments, store: approval())
        try await Task.sleep(for: .seconds(1.5))

        #expect(result.isError)
        #expect(result.content.contains("FORK_BLOCKED"))
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
