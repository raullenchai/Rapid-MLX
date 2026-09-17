import Darwin
import Foundation

/// A deliberately small local-computer surface for conversational work.
/// There is no shell tool: commands are an executable plus an argument array,
/// paths stay inside the current user's home directory, and mutations require
/// per-call approval.
enum LocalWorkspaceTools {
    private final class BoundedOutputBuffer: @unchecked Sendable {
        private let lock = NSLock()
        private var storage = Data()

        func append(_ data: Data) {
            lock.lock()
            defer { lock.unlock() }
            guard storage.count < 65_536 else { return }
            storage.append(data.prefix(65_536 - storage.count))
        }

        func snapshot() -> Data {
            lock.lock()
            defer { lock.unlock() }
            return storage
        }
    }

    private final class BoundedPipeCapture: @unchecked Sendable {
        let pipe = Pipe()
        private let buffer = BoundedOutputBuffer()
        private let finished = DispatchGroup()
        private let finishLock = NSLock()
        private var didFinish = false

        init() {
            finished.enter()
            let reader = Thread { [self] in
                let descriptor = pipe.fileHandleForReading.fileDescriptor
                var chunk = [UInt8](repeating: 0, count: 8_192)
                while true {
                    let count = Darwin.read(descriptor, &chunk, chunk.count)
                    if count > 0 {
                        buffer.append(Data(chunk.prefix(count)))
                    } else if count == 0 {
                        break
                    } else if errno != EINTR {
                        break
                    }
                }
                markFinished()
            }
            reader.name = "Rapid local command output"
            reader.qualityOfService = .userInitiated
            reader.start()
        }

        func closeParentWriter() {
            try? pipe.fileHandleForWriting.close()
        }

        func finish() -> Data {
            _ = finished.wait(timeout: .now() + 1)
            try? pipe.fileHandleForReading.close()
            return buffer.snapshot()
        }

        private func markFinished() {
            finishLock.lock()
            guard !didFinish else {
                finishLock.unlock()
                return
            }
            didFinish = true
            finishLock.unlock()
            finished.leave()
        }
    }

    static let searchDefinition = ToolDefinition(
        name: "local_search",
        description: "Search filenames and UTF-8 text inside a local folder. Use this—not web_search—when the user asks to find something on this Mac. Results include matching paths and short text snippets.",
        parameters: .object([
            "type": .string("object"),
            "properties": .object([
                "path": .object(["type": .string("string"), "description": .string("Folder inside the user's home directory. Use '~' when the user says 'my files' without naming a folder.")]),
                "query": .object(["type": .string("string"), "description": .string("Case-insensitive filename or text to find.")]),
            ]),
            "required": .array([.string("path"), .string("query")]),
            "additionalProperties": .bool(false),
        ])
    )

    static let readDefinition = ToolDefinition(
        name: "local_read",
        description: "Read a UTF-8 text file on this Mac. Use an absolute path inside the user's home directory. Output is capped; do not use this for web URLs.",
        parameters: pathSchema(description: "Absolute file path, or a path beginning with ~, inside the user's home directory.")
    )

    static let writeDefinition = ToolDefinition(
        name: "local_write",
        description: "Create or replace one UTF-8 text file on this Mac. The user approves the exact path and content every time. If no destination was requested, use '~/Rapid Workspace/<descriptive-name>'.",
        parameters: .object([
            "type": .string("object"),
            "properties": .object([
                "path": .object(["type": .string("string"), "description": .string("Absolute path or ~/ path inside the user's home directory. Default to ~/Rapid Workspace when the user omitted a destination.")]),
                "content": .object(["type": .string("string"), "description": .string("Complete UTF-8 file contents.")]),
                "overwrite": .object(["type": .string("boolean"), "description": .string("Set true only when the user asked to replace an existing file.")]),
            ]),
            "required": .array([.string("path"), .string("content")]),
            "additionalProperties": .bool(false),
        ])
    )

    static let trashDefinition = ToolDefinition(
        name: "local_trash",
        description: "Move one local file to the macOS Trash so it remains recoverable. Never deletes folders. The user must approve every action.",
        parameters: pathSchema(description: "Absolute file path, or a path beginning with ~, inside the user's home directory.")
    )

    static let runDefinition = ToolDefinition(
        name: "local_run",
        description: "Run a development command without a shell, for example clang, cc, go, swift, python3, node, ruby, or a compiled executable inside the user's home directory. Pass arguments separately. working_directory is optional and defaults to ~/Rapid Workspace. The user approves every command; execution times out after at most 30 seconds.",
        parameters: .object([
            "type": .string("object"),
            "properties": .object([
                "command": .object(["type": .string("string"), "description": .string("Allowed command name or absolute path to a local executable.")]),
                "arguments": .object(["type": .string("array"), "items": .object(["type": .string("string")])]),
                "working_directory": .object(["type": .string("string"), "description": .string("Absolute or ~/ working directory inside the user's home directory. Use ~/Rapid Workspace for generated code when no destination was requested.")]),
                "cwd": .object(["type": .string("string"), "description": .string("Alias for working_directory.")]),
                "timeout_seconds": .object(["type": .string("integer"), "minimum": .number(1), "maximum": .number(30)]),
            ]),
            "required": .array([.string("command")]),
            "additionalProperties": .bool(false),
        ])
    )

    static let definitions = [
        searchDefinition, readDefinition, writeDefinition, trashDefinition, runDefinition,
    ]

    private static func pathSchema(description: String) -> CodableJSON {
        .object([
            "type": .string("object"),
            "properties": .object([
                "path": .object(["type": .string("string"), "description": .string(description)]),
            ]),
            "required": .array([.string("path")]),
            "additionalProperties": .bool(false),
        ])
    }

    private struct PathArgs: Decodable { let path: String }
    private struct SearchArgs: Decodable { let path: String; let query: String }
    private struct WriteArgs: Decodable { let path: String; let content: String; let overwrite: Bool? }
    private struct RunArgs: Decodable {
        let command: String
        let arguments: [String]?
        let workingDirectory: String?
        let cwd: String?
        let timeoutSeconds: Int?

        enum CodingKeys: String, CodingKey {
            case command, arguments, cwd
            case workingDirectory = "working_directory"
            case timeoutSeconds = "timeout_seconds"
        }
    }

    @MainActor
    static func run(
        _ call: ToolCall,
        approval: LocalToolApprovalStore
    ) async -> ToolCallResult {
        let name = call.function.name
        let persistent = name == "local_search" || name == "local_read"
        let title: String
        switch name {
        case "local_search": title = "Search local files?"
        case "local_read": title = "Read this local file?"
        case "local_write": title = "Write this local file?"
        case "local_trash": title = "Move this file to Trash?"
        case "local_run": title = "Run this command?"
        default: return failure("Unknown local tool \(name)", executed: false)
        }

        // Reject malformed or out-of-scope actions before showing consent UI.
        // Approval should always describe an action Rapid can actually run.
        if let rejected = preflight(name, arguments: call.function.arguments) {
            return rejected
        }
        let grantScope = persistent ? approvalScope(name, arguments: call.function.arguments) : nil

        switch await approval.requestApproval(
            toolName: name,
            title: title,
            argumentsJSON: call.function.arguments,
            grantScope: grantScope,
            allowsPersistentGrant: persistent
        ) {
        case .allowOnce, .alwaysAllowTool: break
        case .deny:
            return ToolCallResult(toolCallID: "", content: "The user declined \(name). Continue without it.", isError: true, failureKind: .userDeclined, executed: false)
        case .unavailable:
            return ToolCallResult(toolCallID: "", content: "\(name) was cancelled before approval.", isError: true, failureKind: .userDeclined, executed: false)
        }

        return await Task.detached(priority: .userInitiated) {
            switch name {
            case "local_search": return search(call.function.arguments)
            case "local_read": return read(call.function.arguments)
            case "local_write": return write(call.function.arguments)
            case "local_trash": return trash(call.function.arguments)
            case "local_run": return runCommand(call.function.arguments)
            default: return failure("Unknown local tool \(name)", executed: false)
            }
        }.value
    }

    private static func preflight(_ name: String, arguments: String) -> ToolCallResult? {
        do {
            switch name {
            case "local_search":
                guard let args = decode(SearchArgs.self, arguments) else {
                    return failure("local_search arguments are invalid", executed: false)
                }
                guard !args.query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    return failure("local_search query is empty", executed: false)
                }
                let url = try safeURL(args.path)
                var isDirectory: ObjCBool = false
                guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
                    return failure("local_search path is not a folder", executed: false)
                }
            case "local_read":
                guard let args = decode(PathArgs.self, arguments) else {
                    return failure("local_read arguments are invalid", executed: false)
                }
                _ = try safeURL(args.path)
            case "local_write":
                guard let args = decode(WriteArgs.self, arguments) else {
                    return failure("local_write arguments are invalid", executed: false)
                }
                guard args.content.utf8.count <= 512_000 else {
                    return failure("local_write content exceeds 512 KB", executed: false)
                }
                _ = try validatedLexicalURL(args.path)
            case "local_trash":
                guard let args = decode(PathArgs.self, arguments) else {
                    return failure("local_trash arguments are invalid", executed: false)
                }
                let url = try safeURL(args.path)
                let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isDirectoryKey])
                guard values.isRegularFile == true, values.isDirectory != true else {
                    return failure("local_trash only moves regular files, never folders", executed: false)
                }
            case "local_run":
                guard let args = decode(RunArgs.self, arguments) else {
                    return failure("local_run arguments are invalid", executed: false)
                }
                _ = try safeURL(args.workingDirectory ?? args.cwd ?? "~/Rapid Workspace", mustExist: false)
                let allowed = Set(["clang", "cc", "gcc", "go", "swift", "python3", "node", "ruby"])
                if !allowed.contains(args.command) {
                    let executable = try safeURL(args.command)
                    guard FileManager.default.isExecutableFile(atPath: executable.path) else {
                        return failure("local_run command is not in the allowlist or is not a local executable", executed: false)
                    }
                }
            default:
                return failure("Unknown local tool \(name)", executed: false)
            }
            return nil
        } catch {
            return failure("\(name) error: \(error.localizedDescription)", executed: false)
        }
    }

    private static func decode<T: Decodable>(_ type: T.Type, _ arguments: String) -> T? {
        guard let data = arguments.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(type, from: data)
    }

    private static func approvalScope(_ name: String, arguments: String) -> String? {
        let rawPath: String?
        switch name {
        case "local_search": rawPath = decode(SearchArgs.self, arguments)?.path
        case "local_read": rawPath = decode(PathArgs.self, arguments)?.path
        default: rawPath = nil
        }
        guard let rawPath else { return nil }
        return try? safeURL(rawPath).path
    }

    private static func validatedLexicalURL(_ path: String) throws -> URL {
        let home = FileManager.default.homeDirectoryForCurrentUser.standardizedFileURL.resolvingSymlinksInPath()
        let expanded: String
        if path == "~" {
            expanded = home.path
        } else if path.hasPrefix("~/") {
            expanded = home.appendingPathComponent(String(path.dropFirst(2))).path
        } else {
            guard path.hasPrefix("/") else { throw LocalError("path must be absolute or begin with ~/") }
            expanded = path
        }
        let url = URL(fileURLWithPath: expanded).standardizedFileURL
        let homePrefix = home.path.hasSuffix("/") ? home.path : home.path + "/"
        guard url.path == home.path || url.path.hasPrefix(homePrefix) else {
            throw LocalError("path must stay inside \(home.path)")
        }
        let relative = String(url.path.dropFirst(homePrefix.count))
        let comparisonRelative = relative.lowercased()
        let protected = [".ssh", ".gnupg", "Library/Keychains"]
        guard !protected.contains(where: {
            let comparisonProtected = $0.lowercased()
            return comparisonRelative == comparisonProtected
                || comparisonRelative.hasPrefix(comparisonProtected + "/")
        }) else {
            throw LocalError("that protected location is unavailable")
        }
        return url
    }

    private static func safeURL(_ path: String, mustExist: Bool = true) throws -> URL {
        let lexicalURL = try validatedLexicalURL(path)
        let url = lexicalURL.resolvingSymlinksInPath()
        _ = try validatedLexicalURL(url.path)
        if mustExist, !FileManager.default.fileExists(atPath: url.path) {
            throw LocalError("path does not exist")
        }
        return url
    }

    /// Keep the approved filename as the filename that is actually opened.
    /// Writes reject symlinked parent paths rather than silently resolving the
    /// approval to a different destination.
    private static func safeWriteURL(_ path: String) throws -> URL {
        let lexicalURL = try validatedLexicalURL(path)
        let lexicalParent = lexicalURL.deletingLastPathComponent()
        var isDirectory: ObjCBool = false
        if !FileManager.default.fileExists(atPath: lexicalParent.path, isDirectory: &isDirectory) {
            try FileManager.default.createDirectory(at: lexicalParent, withIntermediateDirectories: true)
            isDirectory = true
        }
        guard isDirectory.boolValue else { throw LocalError("local_write parent path is not a folder") }
        let resolvedParent = try safeURL(lexicalParent.path)
        guard resolvedParent.path == lexicalParent.path else {
            throw LocalError("local_write parent path may not contain symbolic links")
        }
        return resolvedParent.appendingPathComponent(lexicalURL.lastPathComponent, isDirectory: false)
    }

    private static func search(_ arguments: String) -> ToolCallResult {
        guard let args = decode(SearchArgs.self, arguments) else { return failure("local_search arguments are invalid") }
        let query = args.query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return failure("local_search query is empty") }
        do {
            let root = try safeURL(args.path)
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: root.path, isDirectory: &isDirectory), isDirectory.boolValue else {
                return failure("local_search path is not a folder")
            }
            let keys: [URLResourceKey] = [
                .isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey,
            ]
            guard let enumerator = FileManager.default.enumerator(
                at: root,
                includingPropertiesForKeys: keys,
                options: [.skipsHiddenFiles, .skipsPackageDescendants]
            ) else { return failure("local_search could not open the folder") }
            var visitedEntries = 0
            var scanned = 0
            var matches: [[String: String]] = []
            let needle = query.lowercased()
            while visitedEntries < 2_000,
                  scanned < 500,
                  matches.count < 20,
                  let url = enumerator.nextObject() as? URL {
                visitedEntries += 1
                let values = try? url.resourceValues(forKeys: Set(keys))
                if values?.isDirectory == true {
                    if values?.isSymbolicLink == true || (try? safeURL(url.path)) == nil {
                        enumerator.skipDescendants()
                    }
                    continue
                }
                guard values?.isRegularFile == true,
                      values?.isSymbolicLink != true,
                      let safeFile = try? safeURL(url.path)
                else { continue }
                scanned += 1
                let filenameMatch = safeFile.lastPathComponent.lowercased().contains(needle)
                var snippet: String?
                if (values?.fileSize ?? 0) <= 1_000_000,
                   let data = try? Data(contentsOf: safeFile, options: [.mappedIfSafe]),
                   let text = String(data: data, encoding: .utf8),
                   let range = text.range(of: query, options: [.caseInsensitive]) {
                    let lower = text.index(range.lowerBound, offsetBy: -80, limitedBy: text.startIndex) ?? text.startIndex
                    let upper = text.index(range.upperBound, offsetBy: 160, limitedBy: text.endIndex) ?? text.endIndex
                    snippet = String(text[lower..<upper]).replacingOccurrences(of: "\n", with: " ")
                }
                if filenameMatch || snippet != nil {
                    var item = ["path": safeFile.path]
                    if let snippet { item["snippet"] = snippet }
                    matches.append(item)
                }
            }
            let payload: [String: Any] = [
                "root": root.path,
                "query": query,
                "visited_entries": visitedEntries,
                "scanned_files": scanned,
                "truncated": visitedEntries >= 2_000 || scanned >= 500 || matches.count >= 20,
                "matches": matches,
            ]
            let data = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
            return ToolCallResult(toolCallID: "", content: String(decoding: data, as: UTF8.self))
        } catch { return failure("local_search error: \(error.localizedDescription)") }
    }

    private static func read(_ arguments: String) -> ToolCallResult {
        guard let args = decode(PathArgs.self, arguments) else { return failure("local_read arguments are invalid") }
        do {
            let url = try safeURL(args.path)
            let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
            guard values.isRegularFile == true else { return failure("local_read only reads regular files") }
            guard (values.fileSize ?? 0) <= 512_000 else { return failure("local_read file exceeds 512 KB") }
            let data = try Data(contentsOf: url)
            guard let text = String(data: data, encoding: .utf8) else { return failure("local_read supports UTF-8 text files only") }
            return ToolCallResult(toolCallID: "", content: "Path: \(url.path)\n\(text)")
        } catch { return failure("local_read error: \(error.localizedDescription)") }
    }

    private static func write(_ arguments: String) -> ToolCallResult {
        guard let args = decode(WriteArgs.self, arguments) else { return failure("local_write arguments are invalid") }
        guard args.content.utf8.count <= 512_000 else { return failure("local_write content exceeds 512 KB") }
        do {
            let url = try safeWriteURL(args.path)
            let parent = url.deletingLastPathComponent()
            var isDirectory: ObjCBool = false
            _ = FileManager.default.fileExists(atPath: parent.path, isDirectory: &isDirectory)
            guard isDirectory.boolValue else {
                return failure("local_write parent path is not a folder")
            }
            try secureWrite(Data(args.content.utf8), to: url, overwrite: args.overwrite == true)
            return ToolCallResult(toolCallID: "", content: "Wrote \(args.content.utf8.count) bytes to \(url.path)")
        } catch { return failure("local_write error: \(error.localizedDescription)") }
    }

    private static func secureWrite(_ data: Data, to url: URL, overwrite: Bool) throws {
        let parent = url.deletingLastPathComponent()
        let parentFD = Darwin.open(parent.path, O_RDONLY | O_DIRECTORY | O_CLOEXEC)
        guard parentFD >= 0 else { throw posixError("could not open the destination folder") }
        defer { Darwin.close(parentFD) }

        let finalName = url.lastPathComponent
        guard !finalName.isEmpty, finalName != ".", finalName != "..", !finalName.contains("/") else {
            throw LocalError("destination filename is invalid")
        }
        let temporaryName = ".rapid-write-\(UUID().uuidString).tmp"
        let openedName = overwrite ? temporaryName : finalName
        let descriptor = openedName.withCString {
            Darwin.openat(
                parentFD,
                $0,
                O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
                mode_t(0o600)
            )
        }
        guard descriptor >= 0 else {
            if errno == EEXIST, !overwrite {
                throw LocalError("refused to replace an existing file without overwrite=true")
            }
            throw posixError("could not create the destination file")
        }
        var shouldRemoveTemporary = overwrite
        defer {
            Darwin.close(descriptor)
            if shouldRemoveTemporary {
                temporaryName.withCString { _ = Darwin.unlinkat(parentFD, $0, 0) }
            }
        }

        try data.withUnsafeBytes { rawBuffer in
            guard let base = rawBuffer.baseAddress else { return }
            var written = 0
            while written < rawBuffer.count {
                let count = Darwin.write(
                    descriptor,
                    base.advanced(by: written),
                    rawBuffer.count - written
                )
                guard count > 0 else { throw posixError("could not write the destination file") }
                written += count
            }
        }
        guard Darwin.fsync(descriptor) == 0 else {
            throw posixError("could not sync the destination file")
        }
        if overwrite {
            let renamed = temporaryName.withCString { temporaryPointer in
                finalName.withCString { finalPointer in
                    Darwin.renameat(parentFD, temporaryPointer, parentFD, finalPointer)
                }
            }
            guard renamed == 0 else { throw posixError("could not replace the destination file") }
            shouldRemoveTemporary = false
        }
    }

    private static func posixError(_ context: String) -> LocalError {
        LocalError("\(context): \(String(cString: strerror(errno)))")
    }

    private static func trash(_ arguments: String) -> ToolCallResult {
        guard let args = decode(PathArgs.self, arguments) else { return failure("local_trash arguments are invalid") }
        do {
            let url = try safeURL(args.path)
            let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isDirectoryKey])
            guard values.isRegularFile == true, values.isDirectory != true else {
                return failure("local_trash only moves regular files, never folders")
            }
            var resultingURL: NSURL?
            try FileManager.default.trashItem(at: url, resultingItemURL: &resultingURL)
            return ToolCallResult(toolCallID: "", content: "Moved \(url.path) to Trash. It can be recovered from Finder.")
        } catch { return failure("local_trash error: \(error.localizedDescription)") }
    }

    private static func runCommand(_ arguments: String) -> ToolCallResult {
        guard let args = decode(RunArgs.self, arguments) else { return failure("local_run arguments are invalid") }
        do {
            let requestedWorkingDirectory = args.workingDirectory ?? args.cwd ?? "~/Rapid Workspace"
            let cwd = try safeURL(requestedWorkingDirectory, mustExist: false)
            var isDirectory: ObjCBool = false
            if !FileManager.default.fileExists(atPath: cwd.path, isDirectory: &isDirectory),
               requestedWorkingDirectory == "~/Rapid Workspace" {
                try FileManager.default.createDirectory(at: cwd, withIntermediateDirectories: true)
                isDirectory = true
            }
            guard FileManager.default.fileExists(atPath: cwd.path, isDirectory: &isDirectory), isDirectory.boolValue else {
                return failure("local_run working_directory is not a folder")
            }
            let allowed = Set(["clang", "cc", "gcc", "go", "swift", "python3", "node", "ruby", "make"])
            let executable: URL
            var processArguments = args.arguments ?? []
            if allowed.contains(args.command) {
                if ["clang", "cc", "gcc"].contains(args.command) {
                    let compilerCandidates = [
                        "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang",
                        "/Library/Developer/CommandLineTools/usr/bin/clang",
                        "/usr/bin/clang",
                    ]
                    guard let compiler = compilerCandidates.first(where: {
                        FileManager.default.isExecutableFile(atPath: $0)
                    }) else {
                        return failure("local_run could not find an installed C compiler")
                    }
                    executable = URL(fileURLWithPath: compiler)
                    let sdkCandidates = [
                        "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk",
                        "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk",
                    ]
                    if let sdk = sdkCandidates.first(where: {
                        FileManager.default.fileExists(atPath: $0)
                    }) {
                        processArguments.insert(contentsOf: ["-isysroot", sdk], at: 0)
                    }
                } else {
                    executable = URL(fileURLWithPath: "/usr/bin/env")
                    processArguments.insert(args.command, at: 0)
                }
            } else {
                executable = try safeURL(args.command)
                guard FileManager.default.isExecutableFile(atPath: executable.path) else {
                    return failure("local_run command is not in the allowlist or is not a local executable")
                }
            }

            let temporary = cwd.appendingPathComponent(".rapid-tmp-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: false)
            defer { try? FileManager.default.removeItem(at: temporary) }
            let sandboxArguments = [
                "-p", sandboxProfile(
                    home: FileManager.default.homeDirectoryForCurrentUser.resolvingSymlinksInPath(),
                    workingDirectory: cwd,
                    temporaryDirectory: temporary,
                    executable: executable
                ),
                executable.path,
            ] + processArguments
            let environment = [
                "HOME": FileManager.default.homeDirectoryForCurrentUser.path,
                "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
                "TMPDIR": temporary.path,
                "LANG": "en_US.UTF-8",
            ]
            let timeout = min(max(args.timeoutSeconds ?? 15, 1), 30)
            let outcome = try spawnSandboxed(
                arguments: sandboxArguments,
                environment: environment,
                workingDirectory: cwd,
                timeout: TimeInterval(timeout)
            )
            let out = String(decoding: outcome.stdout, as: UTF8.self)
            let err = String(decoding: outcome.stderr, as: UTF8.self)
            let content = "exit_code: \(outcome.exitCode)\(outcome.timedOut ? " (timed out)" : "")\nstdout:\n\(out)\nstderr:\n\(err)"
            return ToolCallResult(
                toolCallID: "",
                content: content,
                isError: outcome.timedOut || outcome.exitCode != 0
            )
        } catch { return failure("local_run error: \(error.localizedDescription)") }
    }

    private struct CommandOutcome {
        let exitCode: Int32
        let timedOut: Bool
        let stdout: Data
        let stderr: Data
    }

    private static func spawnSandboxed(
        arguments: [String],
        environment: [String: String],
        workingDirectory: URL,
        timeout: TimeInterval
    ) throws -> CommandOutcome {
        let stdout = BoundedPipeCapture()
        let stderr = BoundedPipeCapture()
        defer {
            stdout.closeParentWriter()
            stderr.closeParentWriter()
        }
        var actions: posix_spawn_file_actions_t?
        var attributes: posix_spawnattr_t?
        guard posix_spawn_file_actions_init(&actions) == 0,
              posix_spawnattr_init(&attributes) == 0 else {
            throw LocalError("could not initialize the command launcher")
        }
        defer {
            posix_spawn_file_actions_destroy(&actions)
            posix_spawnattr_destroy(&attributes)
        }
        guard posix_spawn_file_actions_adddup2(
            &actions, stdout.pipe.fileHandleForWriting.fileDescriptor, STDOUT_FILENO
        ) == 0,
        posix_spawn_file_actions_adddup2(
            &actions, stderr.pipe.fileHandleForWriting.fileDescriptor, STDERR_FILENO
        ) == 0,
        posix_spawn_file_actions_addclose(
            &actions, stdout.pipe.fileHandleForReading.fileDescriptor
        ) == 0,
        posix_spawn_file_actions_addclose(
            &actions, stderr.pipe.fileHandleForReading.fileDescriptor
        ) == 0,
        posix_spawn_file_actions_addclose(
            &actions, stdout.pipe.fileHandleForWriting.fileDescriptor
        ) == 0,
        posix_spawn_file_actions_addclose(
            &actions, stderr.pipe.fileHandleForWriting.fileDescriptor
        ) == 0,
        posix_spawn_file_actions_addchdir_np(&actions, workingDirectory.path) == 0,
        posix_spawnattr_setflags(&attributes, Int16(POSIX_SPAWN_SETPGROUP)) == 0,
        posix_spawnattr_setpgroup(&attributes, 0) == 0 else {
            throw LocalError("could not configure the command launcher")
        }

        let argvStrings = ["sandbox-exec"] + arguments
        let environmentStrings = environment.map { "\($0.key)=\($0.value)" }.sorted()
        var argv = argvStrings.map { strdup($0) } + [nil]
        var envp = environmentStrings.map { strdup($0) } + [nil]
        defer {
            argv.dropLast().forEach { free($0) }
            envp.dropLast().forEach { free($0) }
        }
        var pid: pid_t = 0
        let spawnStatus = argv.withUnsafeMutableBufferPointer { argvBuffer in
            envp.withUnsafeMutableBufferPointer { environmentBuffer in
                posix_spawn(
                    &pid,
                    "/usr/bin/sandbox-exec",
                    &actions,
                    &attributes,
                    argvBuffer.baseAddress,
                    environmentBuffer.baseAddress
                )
            }
        }
        guard spawnStatus == 0, pid > 0 else {
            throw LocalError("could not start the approved command: \(String(cString: strerror(spawnStatus)))")
        }
        stdout.closeParentWriter()
        stderr.closeParentWriter()

        var status: Int32 = 0
        var reaped = false
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            let waited = waitpid(pid, &status, WNOHANG)
            if waited == pid {
                reaped = true
                break
            }
            if waited == -1, errno != EINTR { throw posixError("could not monitor the approved command") }
            Thread.sleep(forTimeInterval: 0.02)
        }
        let timedOut = !reaped
        if timedOut {
            _ = kill(-pid, SIGTERM)
            let grace = Date().addingTimeInterval(0.5)
            while Date() < grace {
                let waited = waitpid(pid, &status, WNOHANG)
                if waited == pid {
                    reaped = true
                    break
                }
                Thread.sleep(forTimeInterval: 0.02)
            }
            if !reaped {
                _ = kill(-pid, SIGKILL)
                let killDeadline = Date().addingTimeInterval(1)
                while Date() < killDeadline {
                    let waited = waitpid(pid, &status, WNOHANG)
                    if waited == pid {
                        reaped = true
                        break
                    }
                    Thread.sleep(forTimeInterval: 0.02)
                }
            }
        }
        let exitCode: Int32
        if !reaped {
            exitCode = -1
        } else if status & 0x7f == 0 {
            exitCode = (status >> 8) & 0xff
        } else if status & 0x7f != 0x7f {
            exitCode = 128 + (status & 0x7f)
        } else {
            exitCode = -1
        }
        return CommandOutcome(
            exitCode: exitCode,
            timedOut: timedOut,
            stdout: stdout.finish(),
            stderr: stderr.finish()
        )
    }

    private static func sandboxProfile(
        home: URL,
        workingDirectory: URL,
        temporaryDirectory: URL,
        executable: URL
    ) -> String {
        func quoted(_ path: String) -> String {
            "\"" + path.replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"") + "\""
        }
        return """
        (version 1)
        (allow default)
        (deny network*)
        (deny mach-lookup)
        ; Keep runtimes and toolchains usable, but deny data-bearing locations
        ; outside the exact approved workspace. The more-specific workspace
        ; allow is the only exception within the user's home directory.
        (deny file-read* (subpath \(quoted(home.path))))
        (deny file-read* (subpath "/Users"))
        (deny file-read* (subpath "/Volumes"))
        (deny file-read* (subpath "/private/etc"))
        (allow file-read*
            (subpath \(quoted(workingDirectory.path)))
            (subpath \(quoted(temporaryDirectory.path)))
            (literal \(quoted(executable.path))))
        (deny file-write*)
        (allow file-write*
            (subpath \(quoted(workingDirectory.path)))
            (subpath \(quoted(temporaryDirectory.path))))
        (deny process-exec
            (literal "/bin/sh") (literal "/bin/zsh") (literal "/bin/bash")
            (literal "/usr/bin/osascript") (literal "/usr/bin/open"))
        """
    }

    private static func failure(_ message: String, executed: Bool = true) -> ToolCallResult {
        ToolCallResult(toolCallID: "", content: message, isError: true, executed: executed)
    }

    private struct LocalError: LocalizedError {
        let message: String
        init(_ message: String) { self.message = message }
        var errorDescription: String? { message }
    }
}
