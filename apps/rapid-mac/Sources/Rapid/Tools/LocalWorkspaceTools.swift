import Darwin
import CryptoKit
import Foundation

@_silgen_name("proc_listchildpids")
private func rapidProcListChildPIDs(
    _ parent: pid_t,
    _ buffer: UnsafeMutableRawPointer?,
    _ bufferSize: Int32
) -> Int32

/// A deliberately small local-computer surface for conversational work.
/// There is no shell tool: commands are an executable plus an argument array,
/// paths stay inside the current user's home directory, and mutations require
/// per-call approval.
enum LocalWorkspaceTools {
    private static let allowedCommands = Set([
        "clang", "cc", "gcc", "go", "swift", "python3", "node", "ruby",
    ])

    private struct FileIdentity: Equatable, Sendable {
        let device: dev_t
        let inode: ino_t
    }

    /// The object that was shown in the approval sheet. Keeping the descriptor
    /// open pins the inode while the sheet is visible; execution also verifies
    /// that the approved pathname still names that same inode.
    private final class PinnedPath: @unchecked Sendable {
        let url: URL
        let identity: FileIdentity
        let descriptor: Int32
        private let presentedURL: URL?
        private let presentedIdentity: FileIdentity?

        init(url: URL, directory: Bool, presentedURL: URL? = nil) throws {
            let capturedPresentedURL: URL?
            let capturedPresentedIdentity: FileIdentity?
            if let presentedURL, presentedURL.path != url.path {
                capturedPresentedURL = presentedURL
                capturedPresentedIdentity = try fileIdentity(at: presentedURL)
            } else {
                capturedPresentedURL = nil
                capturedPresentedIdentity = nil
            }
            let flags = O_RDONLY | O_NOFOLLOW | O_CLOEXEC | (directory ? O_DIRECTORY : 0)
            descriptor = Darwin.open(url.path, flags)
            guard descriptor >= 0 else {
                throw posixError("could not open the object awaiting approval")
            }
            var metadata = stat()
            guard fstat(descriptor, &metadata) == 0 else {
                Darwin.close(descriptor)
                throw posixError("could not inspect the object awaiting approval")
            }
            let expectedType = directory ? S_IFDIR : S_IFREG
            guard metadata.st_mode & S_IFMT == expectedType else {
                Darwin.close(descriptor)
                throw LocalError(directory ? "approved path is not a folder" : "approved path is not a regular file")
            }
            self.url = url
            identity = FileIdentity(device: metadata.st_dev, inode: metadata.st_ino)
            self.presentedURL = capturedPresentedURL
            presentedIdentity = capturedPresentedIdentity
        }

        init(
            parentDescriptor: Int32,
            filename: String,
            url: URL,
            directory: Bool
        ) throws {
            let flags = O_RDONLY | O_NOFOLLOW | O_CLOEXEC | (directory ? O_DIRECTORY : 0)
            descriptor = filename.withCString {
                Darwin.openat(parentDescriptor, $0, flags)
            }
            guard descriptor >= 0 else {
                throw posixError("could not open the object awaiting approval")
            }
            var metadata = stat()
            guard fstat(descriptor, &metadata) == 0 else {
                Darwin.close(descriptor)
                throw posixError("could not inspect the object awaiting approval")
            }
            let expectedType = directory ? S_IFDIR : S_IFREG
            guard metadata.st_mode & S_IFMT == expectedType else {
                Darwin.close(descriptor)
                throw LocalError(
                    directory
                        ? "approved path is not a folder"
                        : "approved path is not a regular file"
                )
            }
            self.url = url
            identity = FileIdentity(device: metadata.st_dev, inode: metadata.st_ino)
            self.presentedURL = nil
            presentedIdentity = nil
        }

        deinit { Darwin.close(descriptor) }

        func verifyPathStillNamesPinnedObject() throws {
            guard try fileIdentity(at: url) == identity else {
                throw LocalError("the approved file changed while approval was open")
            }
            if let presentedURL, let presentedIdentity,
               try fileIdentity(at: presentedURL) != presentedIdentity {
                throw LocalError("the approved file changed while approval was open")
            }
        }

        var approvalScope: String {
            "\(url.path)\u{0}\(identity.device)\u{0}\(identity.inode)"
        }
    }

    private struct ApprovedRun: @unchecked Sendable {
        let arguments: RunArgs
        let workingDirectory: PinnedPath?
        let creationHome: PinnedPath?
        let requestedWorkingDirectory: URL
        let executable: PinnedPath
        let processArguments: [String]
        let helperClass: HelperClass
        let developerDirectories: [PinnedPath]
        let executableDigest: Data?
    }

    private struct ApprovedWrite: @unchecked Sendable {
        let arguments: WriteArgs
        let parent: PinnedPath?
        let creationHome: PinnedPath?
        let requestedParent: URL
        let filename: String
        let expectedIdentity: FileIdentity?
    }

    private enum HelperClass: Sendable {
        case none, compiler, swift, go
    }

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
        description: "Run a development command without a shell, for example clang, cc, go, swift, python3, node, ruby, or a compiled executable inside the user's home directory. Put the command name in command and its arguments in the argv string array. working_directory is optional and defaults to ~/Rapid Workspace. The user approves every command; execution times out after at most 30 seconds.",
        parameters: .object([
            "type": .string("object"),
            "properties": .object([
                "command": .object(["type": .string("string"), "description": .string("Allowed command name or absolute path to a local executable.")]),
                "argv": .object(["type": .string("array"), "items": .object(["type": .string("string")]), "description": .string("Command arguments, one string each.")]),
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
        /// Command arguments. The wire key is `argv`; the pre-0.14.3 key
        /// `arguments` and the common `args` spelling are still accepted so a
        /// small model that reaches for either keeps working.
        let argv: [String]?
        let workingDirectory: String?
        let cwd: String?
        let timeoutSeconds: Int?

        enum CodingKeys: String, CodingKey {
            case command, argv, arguments, args, cwd
            case workingDirectory = "working_directory"
            case timeoutSeconds = "timeout_seconds"
        }

        init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            command = try container.decode(String.self, forKey: .command)
            if container.contains(.argv) {
                argv = try container.decode([String].self, forKey: .argv)
            } else if container.contains(.arguments) {
                argv = try container.decode([String].self, forKey: .arguments)
            } else if container.contains(.args) {
                argv = try container.decode([String].self, forKey: .args)
            } else {
                argv = nil
            }
            workingDirectory = try container.decodeIfPresent(String.self, forKey: .workingDirectory)
            cwd = try container.decodeIfPresent(String.self, forKey: .cwd)
            timeoutSeconds = try container.decodeIfPresent(Int.self, forKey: .timeoutSeconds)
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
        default: return withToolCallID(failure("Unknown local tool \(name)", executed: false), call.id)
        }

        // Reject malformed or out-of-scope actions before showing consent UI.
        // Approval should always describe an action Rapid can actually run.
        if let rejected = preflight(name, arguments: call.function.arguments) {
            return withToolCallID(rejected, call.id)
        }
        let approvedPath: PinnedPath?
        let approvedParent: PinnedPath?
        let approvedTrash: PinnedPath?
        let approvedRun: ApprovedRun?
        let approvedWrite: ApprovedWrite?
        do {
            switch name {
            case "local_search":
                let args = try requireDecoded(SearchArgs.self, call.function.arguments)
                approvedPath = try PinnedPath(
                    url: safeURL(args.path), directory: true,
                    presentedURL: validatedLexicalURL(args.path)
                )
                approvedRun = nil
                approvedWrite = nil
                approvedParent = nil
                approvedTrash = nil
            case "local_read":
                let args = try requireDecoded(PathArgs.self, call.function.arguments)
                approvedPath = try PinnedPath(
                    url: safeURL(args.path), directory: false,
                    presentedURL: validatedLexicalURL(args.path)
                )
                approvedRun = nil
                approvedWrite = nil
                approvedParent = nil
                approvedTrash = nil
            case "local_trash":
                let args = try requireDecoded(PathArgs.self, call.function.arguments)
                let fileURL = try validatedLexicalURL(args.path)
                let lexicalParent = fileURL.deletingLastPathComponent()
                let resolvedParent = try safeURL(lexicalParent.path)
                guard resolvedParent.path == lexicalParent.path else {
                    throw LocalError("local_trash parent path may not contain symbolic links")
                }
                let pinnedParent = try PinnedPath(url: resolvedParent, directory: true)
                approvedParent = pinnedParent
                approvedPath = try PinnedPath(
                    parentDescriptor: pinnedParent.descriptor,
                    filename: fileURL.lastPathComponent,
                    url: fileURL,
                    directory: false
                )
                approvedTrash = try PinnedPath(
                    url: FileManager.default.homeDirectoryForCurrentUser
                        .appendingPathComponent(".Trash"),
                    directory: true
                )
                approvedRun = nil
                approvedWrite = nil
            case "local_write":
                let args = try requireDecoded(WriteArgs.self, call.function.arguments)
                approvedWrite = try prepareWrite(args)
                approvedPath = nil
                approvedRun = nil
                approvedParent = nil
                approvedTrash = nil
            case "local_run":
                let args = try requireDecoded(RunArgs.self, call.function.arguments)
                approvedRun = try prepareRun(args)
                approvedPath = nil
                approvedWrite = nil
                approvedParent = nil
                approvedTrash = nil
            default:
                approvedPath = nil
                approvedRun = nil
                approvedWrite = nil
                approvedParent = nil
                approvedTrash = nil
            }
        } catch {
            return withToolCallID(
                failure("\(name) error: \(error.localizedDescription)", executed: false), call.id
            )
        }
        // A remembered read/search grant belongs to the approved object, not
        // merely to a pathname that can later be replaced with another inode.
        let grantScope = persistent ? approvedPath?.approvalScope : nil

        switch await approval.requestApproval(
            toolName: name,
            title: title,
            argumentsJSON: call.function.arguments,
            grantScope: grantScope,
            allowsPersistentGrant: persistent
        ) {
        case .allowOnce, .alwaysAllowTool: break
        case .deny:
            return ToolCallResult(toolCallID: call.id, content: "The user declined \(name). Continue without it.", isError: true, failureKind: .userDeclined, executed: false)
        case .unavailable:
            return ToolCallResult(toolCallID: call.id, content: "\(name) was cancelled before approval.", isError: true, failureKind: .userDeclined, executed: false)
        }

        let result = await Task.detached(priority: .userInitiated) {
            switch name {
            case "local_search": return search(call.function.arguments, approved: approvedPath)
            case "local_read": return read(approved: approvedPath)
            case "local_write": return write(approved: approvedWrite)
            case "local_trash": return trash(
                approved: approvedPath, parent: approvedParent, trashDirectory: approvedTrash
            )
            case "local_run": return runCommand(approved: approvedRun)
            default: return failure("Unknown local tool \(name)", executed: false)
            }
        }.value
        return withToolCallID(result, call.id)
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
                let lexicalURL = try validatedLexicalURL(args.path)
                var metadata = stat()
                guard lexicalURL.path.withCString({ lstat($0, &metadata) }) == 0,
                      metadata.st_mode & S_IFMT != S_IFLNK else {
                    return failure("local_trash refuses symbolic links", executed: false)
                }
            case "local_run":
                guard let args = decode(RunArgs.self, arguments) else {
                    return failure("local_run arguments are invalid", executed: false)
                }
                _ = try safeWorkingDirectory(
                    args.workingDirectory ?? args.cwd ?? "~/Rapid Workspace",
                    mustExist: false
                )
                if !allowedCommands.contains(args.command) {
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

    private static func requireDecoded<T: Decodable>(
        _ type: T.Type, _ arguments: String
    ) throws -> T {
        guard let decoded = decode(type, arguments) else {
            throw LocalError("arguments are invalid")
        }
        return decoded
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
        let components = relative.split(separator: "/")
        guard !components.contains(where: { $0.hasPrefix(".") }) else {
            throw LocalError("hidden or protected files and folders are unavailable")
        }
        let protected = ["Library/Keychains"]
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

    private static func safeWorkingDirectory(
        _ path: String, mustExist: Bool = true
    ) throws -> URL {
        let url = try safeURL(path, mustExist: mustExist)
        let candidate = url.path.lowercased()
        let home = FileManager.default.homeDirectoryForCurrentUser
            .standardizedFileURL.resolvingSymlinksInPath()
        let protected = [".ssh", ".gnupg", "Library/Keychains"].map {
            home.appendingPathComponent($0).standardizedFileURL.path.lowercased()
        }
        let prefix = candidate.hasSuffix("/") ? candidate : candidate + "/"
        guard !protected.contains(where: { $0 == candidate || $0.hasPrefix(prefix) }) else {
            throw LocalError("working directory would expose a protected location")
        }
        return url
    }

    private static func fileIdentity(at url: URL) throws -> FileIdentity {
        var metadata = stat()
        let status = url.path.withCString { Darwin.lstat($0, &metadata) }
        guard status == 0 else { throw posixError("could not inspect the approved file") }
        return FileIdentity(device: metadata.st_dev, inode: metadata.st_ino)
    }

    private static func prepareWrite(_ args: WriteArgs) throws -> ApprovedWrite {
        let lexicalURL = try validatedLexicalURL(args.path)
        let lexicalParent = lexicalURL.deletingLastPathComponent()
        var isDirectory: ObjCBool = false
        var parent: PinnedPath?
        var creationHome: PinnedPath?
        if !FileManager.default.fileExists(atPath: lexicalParent.path, isDirectory: &isDirectory) {
            let home = FileManager.default.homeDirectoryForCurrentUser.standardizedFileURL
                .resolvingSymlinksInPath()
            guard lexicalParent.path == home.appendingPathComponent("Rapid Workspace").path else {
                throw LocalError("local_write parent folder must already exist")
            }
            creationHome = try PinnedPath(url: home, directory: true)
        } else {
            guard isDirectory.boolValue else { throw LocalError("local_write parent path is not a folder") }
            let resolvedParent = try safeURL(lexicalParent.path)
            guard resolvedParent.path == lexicalParent.path else {
                throw LocalError("local_write parent path may not contain symbolic links")
            }
            parent = try PinnedPath(url: resolvedParent, directory: true)
        }
        let filename = lexicalURL.lastPathComponent
        guard !filename.isEmpty, filename != ".", filename != "..", !filename.contains("/") else {
            throw LocalError("destination filename is invalid")
        }
        let expectedIdentity = try parent.map {
            try entryIdentity(parentDescriptor: $0.descriptor, filename: filename)
        } ?? nil
        return ApprovedWrite(
            arguments: args,
            parent: parent,
            creationHome: creationHome,
            requestedParent: lexicalParent,
            filename: filename,
            expectedIdentity: expectedIdentity
        )
    }

    private static func search(
        _ arguments: String, approved: PinnedPath?
    ) -> ToolCallResult {
        guard let args = decode(SearchArgs.self, arguments) else { return failure("local_search arguments are invalid") }
        let query = args.query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return failure("local_search query is empty") }
        do {
            guard let approved else { return failure("local_search approval expired") }
            try approved.verifyPathStillNamesPinnedObject()
            let root = approved.url
            var state = SearchState(query: query)
            searchDirectory(
                descriptor: approved.descriptor,
                displayURL: root,
                depth: 0,
                state: &state
            )
            let payload: [String: Any] = [
                "root": root.path,
                "query": query,
                "visited_entries": state.visitedEntries,
                "scanned_files": state.scannedFiles,
                "truncated": state.isAtLimit,
                "matches": state.matches,
            ]
            let data = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
            return ToolCallResult(toolCallID: "", content: String(decoding: data, as: UTF8.self))
        } catch { return failure("local_search error: \(error.localizedDescription)") }
    }

    private struct SearchState {
        let query: String
        let needle: String
        /// Lower-cased words of the query. A file matches when every word
        /// appears in its name or its text, so "orchid notes" still finds
        /// `orchid-notes.txt` and a note whose words are not adjacent.
        let terms: [String]
        var visitedEntries = 0
        var scannedFiles = 0
        var matches: [[String: String]] = []

        init(query: String) {
            self.query = query
            needle = query.lowercased()
            terms = LocalWorkspaceTools.searchTerms(for: query)
        }

        var isAtLimit: Bool {
            visitedEntries >= 2_000 || scannedFiles >= 500 || matches.count >= 20
        }
    }

    /// Lower-cased query words, punctuation stripped, in query order. A query
    /// that is a single word yields one term, so the exact-phrase path and
    /// the all-words path agree.
    static func searchTerms(for query: String) -> [String] {
        query.lowercased()
            .split { !($0.isLetter || $0.isNumber) }
            .map(String.init)
            .filter { !$0.isEmpty }
    }

    /// Whole-query substring first (exact phrase), then every query word
    /// anywhere in the text. Returns the range that anchors the snippet.
    static func snippetRange(query: String, terms: [String], in text: String) -> Range<String.Index>? {
        if let exact = text.range(of: query, options: [.caseInsensitive]) { return exact }
        guard terms.count > 1 else { return nil }
        var missing = Set(terms)
        var firstRanges: [String: Range<String.Index>] = [:]
        var start: String.Index?
        var index = text.startIndex
        while index < text.endIndex {
            let character = text[index]
            if character.isLetter || character.isNumber {
                if start == nil { start = index }
            } else if let tokenStart = start {
                let word = String(text[tokenStart..<index]).lowercased()
                if missing.remove(word) != nil {
                    firstRanges[word] = tokenStart..<index
                    if missing.isEmpty { return firstRanges[terms[0]] }
                }
                start = nil
            }
            index = text.index(after: index)
        }
        if let tokenStart = start {
            let word = String(text[tokenStart..<text.endIndex]).lowercased()
            if missing.remove(word) != nil {
                firstRanges[word] = tokenStart..<text.endIndex
            }
        }
        return missing.isEmpty ? firstRanges[terms[0]] : nil
    }

    private static func snippetRange(for state: SearchState, in text: String) -> Range<String.Index>? {
        snippetRange(query: state.query, terms: state.terms, in: text)
    }

    static func matchesAllTerms(_ terms: [String], in lowercasedText: String) -> Bool {
        guard terms.count > 1 else { return false }
        let words = Set(searchTerms(for: lowercasedText))
        return terms.allSatisfy(words.contains)
    }

    /// Walk from the approved directory descriptor rather than reopening its
    /// pathname. Every descendant is inspected/opened relative to a pinned
    /// parent descriptor with O_NOFOLLOW, so renames and symlink swaps cannot
    /// redirect a search after consent.
    private static func searchDirectory(
        descriptor: Int32,
        displayURL: URL,
        depth: Int,
        state: inout SearchState
    ) {
        guard !state.isAtLimit, depth < 64 else { return }
        let duplicate = Darwin.dup(descriptor)
        guard duplicate >= 0, let directory = fdopendir(duplicate) else {
            if duplicate >= 0 { Darwin.close(duplicate) }
            return
        }
        defer { closedir(directory) }

        let packageExtensions = Set(["app", "bundle", "framework", "pkg", "xcodeproj"])
        while !state.isAtLimit, let entry = readdir(directory) {
            let name = withUnsafePointer(to: &entry.pointee.d_name) { pointer in
                pointer.withMemoryRebound(to: CChar.self, capacity: Int(MAXNAMLEN) + 1) {
                    String(cString: $0)
                }
            }
            guard name != ".", name != "..", !name.hasPrefix(".") else { continue }
            state.visitedEntries += 1
            var metadata = stat()
            let inspected = name.withCString {
                fstatat(descriptor, $0, &metadata, AT_SYMLINK_NOFOLLOW)
            }
            guard inspected == 0 else { continue }
            let displayChild = displayURL.appendingPathComponent(name)
            // Approval of a broad ancestor (for example ~/Library) must not
            // implicitly grant access to credential stores below it.
            guard !isProtectedSearchURL(displayChild) else { continue }
            switch metadata.st_mode & S_IFMT {
            case S_IFDIR:
                guard !packageExtensions.contains(displayChild.pathExtension.lowercased()) else { continue }
                let child = name.withCString {
                    Darwin.openat(descriptor, $0, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
                }
                guard child >= 0 else { continue }
                searchDirectory(
                    descriptor: child,
                    displayURL: displayChild,
                    depth: depth + 1,
                    state: &state
                )
                Darwin.close(child)
            case S_IFREG:
                state.scannedFiles += 1
                let lowercasedName = name.lowercased()
                let filenameMatch = lowercasedName.contains(state.needle)
                    || Self.matchesAllTerms(state.terms, in: lowercasedName)
                var snippet: String?
                if metadata.st_size <= 1_000_000 {
                    let file = name.withCString {
                        Darwin.openat(descriptor, $0, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
                    }
                    if file >= 0 {
                        defer { Darwin.close(file) }
                        if let data = readData(descriptor: file, limit: 1_000_000),
                           let text = String(data: data, encoding: .utf8),
                           let range = Self.snippetRange(for: state, in: text) {
                            let lower = text.index(range.lowerBound, offsetBy: -80, limitedBy: text.startIndex) ?? text.startIndex
                            let upper = text.index(range.upperBound, offsetBy: 160, limitedBy: text.endIndex) ?? text.endIndex
                            snippet = String(text[lower..<upper]).replacingOccurrences(of: "\n", with: " ")
                        }
                    }
                }
                if filenameMatch || snippet != nil {
                    var item = ["path": displayChild.path]
                    if let snippet { item["snippet"] = snippet }
                    state.matches.append(item)
                }
            default:
                continue
            }
        }
    }

    static func isProtectedSearchURL(_ url: URL) -> Bool {
        let home = FileManager.default.homeDirectoryForCurrentUser
            .standardizedFileURL.resolvingSymlinksInPath()
        let candidate = url.standardizedFileURL.path.lowercased()
        let protected = home.appendingPathComponent("Library/Keychains", isDirectory: true)
            .standardizedFileURL.path.lowercased()
        return candidate == protected || candidate.hasPrefix(protected + "/")
    }

    private static func readData(descriptor: Int32, limit: Int) -> Data? {
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 16_384)
        while data.count <= limit {
            let count = Darwin.read(descriptor, &buffer, min(buffer.count, limit + 1 - data.count))
            if count == 0 { return data }
            if count < 0 {
                if errno == EINTR { continue }
                return nil
            }
            data.append(contentsOf: buffer.prefix(count))
        }
        return nil
    }

    private static func read(approved: PinnedPath?) -> ToolCallResult {
        do {
            guard let approved else { return failure("local_read approval expired") }
            try approved.verifyPathStillNamesPinnedObject()
            let url = approved.url
            let descriptor = approved.descriptor
            guard lseek(descriptor, 0, SEEK_SET) >= 0 else {
                return failure("local_read could not seek the approved file")
            }
            var metadata = stat()
            guard fstat(descriptor, &metadata) == 0,
                  metadata.st_mode & S_IFMT == S_IFREG,
                  FileIdentity(device: metadata.st_dev, inode: metadata.st_ino) == approved.identity else {
                return failure("local_read only reads regular files")
            }
            guard metadata.st_size <= 512_000 else {
                return failure("local_read file exceeds 512 KB")
            }
            var data = Data()
            var buffer = [UInt8](repeating: 0, count: 16_384)
            while data.count <= 512_000 {
                let count = Darwin.read(descriptor, &buffer, buffer.count)
                if count == 0 { break }
                if count < 0 {
                    if errno == EINTR { continue }
                    return failure("local_read could not read the file")
                }
                data.append(contentsOf: buffer.prefix(count))
            }
            guard data.count <= 512_000 else {
                return failure("local_read file exceeds 512 KB")
            }
            guard let text = String(data: data, encoding: .utf8) else { return failure("local_read supports UTF-8 text files only") }
            return ToolCallResult(toolCallID: "", content: "Path: \(url.path)\n\(text)")
        } catch { return failure("local_read error: \(error.localizedDescription)") }
    }

    private static func write(approved: ApprovedWrite?) -> ToolCallResult {
        guard let approved else { return failure("local_write approval expired") }
        let args = approved.arguments
        guard args.content.utf8.count <= 512_000 else { return failure("local_write content exceeds 512 KB") }
        do {
            let parent = try approvedDirectory(
                existing: approved.parent,
                creationHome: approved.creationHome,
                requestedURL: approved.requestedParent
            )
            try secureWrite(
                Data(args.content.utf8),
                parent: parent,
                filename: approved.filename,
                expectedIdentity: approved.expectedIdentity,
                overwrite: args.overwrite == true
            )
            let url = parent.url.appendingPathComponent(approved.filename)
            return ToolCallResult(toolCallID: "", content: "Wrote \(args.content.utf8.count) bytes to \(Self.displayPath(url))")
        } catch { return failure("local_write error: \(error.localizedDescription)") }
    }

    private static func secureWrite(
        _ data: Data,
        parent: PinnedPath,
        filename: String,
        expectedIdentity: FileIdentity?,
        overwrite: Bool
    ) throws {
        let parentFD = parent.descriptor
        let finalName = filename
        let currentIdentity = try entryIdentity(
            parentDescriptor: parentFD,
            filename: finalName
        )
        guard currentIdentity == expectedIdentity else {
            throw LocalError("the approved destination changed while approval was open")
        }
        if currentIdentity != nil {
            var metadata = stat()
            let inspected = finalName.withCString {
                fstatat(parentFD, $0, &metadata, AT_SYMLINK_NOFOLLOW)
            }
            guard inspected == 0 else {
                throw posixError("could not inspect the approved destination")
            }
            let kind = metadata.st_mode & S_IFMT
            guard kind == S_IFREG || kind == S_IFLNK else {
                throw LocalError("local_write only replaces regular files or symbolic links")
            }
        }
        if !overwrite, currentIdentity != nil {
            throw LocalError("refused to replace an existing file without overwrite=true")
        }
        let temporaryName = ".rapid-write-\(UUID().uuidString).tmp"
        let descriptor = temporaryName.withCString {
            Darwin.openat(
                parentFD,
                $0,
                O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
                mode_t(0o600)
            )
        }
        guard descriptor >= 0 else {
            throw posixError("could not create the destination file")
        }
        var shouldRemoveTemporary = true
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

        if let expectedIdentity {
            guard overwrite else {
                throw LocalError("refused to replace an existing file without overwrite=true")
            }
            let backupName = ".rapid-replaced-\(UUID().uuidString).tmp"
            let movedOriginal = finalName.withCString { finalPointer in
                backupName.withCString { backupPointer in
                    renameatx_np(parentFD, finalPointer, parentFD, backupPointer, UInt32(RENAME_EXCL))
                }
            }
            guard movedOriginal == 0 else { throw posixError("could not secure the approved destination") }
            let movedIdentity = try entryIdentity(
                parentDescriptor: parentFD,
                filename: backupName
            )
            guard movedIdentity == expectedIdentity else {
                let restored = backupName.withCString { backupPointer in
                    finalName.withCString { finalPointer in
                        renameatx_np(parentFD, backupPointer, parentFD, finalPointer, UInt32(RENAME_EXCL))
                    }
                }
                guard restored == 0 else {
                    let backupURL = parent.url.appendingPathComponent(backupName)
                    throw LocalError(
                        "the approved destination changed; the original remains recoverable at \(backupURL.path)"
                    )
                }
                throw LocalError("the approved destination changed while approval was open")
            }
            let installed = temporaryName.withCString { temporaryPointer in
                finalName.withCString { finalPointer in
                    renameatx_np(parentFD, temporaryPointer, parentFD, finalPointer, UInt32(RENAME_EXCL))
                }
            }
            if installed != 0 {
                let installError = posixError("could not replace the destination file")
                let restored = backupName.withCString { backupPointer in
                    finalName.withCString { finalPointer in
                        renameatx_np(parentFD, backupPointer, parentFD, finalPointer, UInt32(RENAME_EXCL))
                    }
                }
                guard restored == 0 else {
                    let backupURL = parent.url.appendingPathComponent(backupName)
                    throw LocalError(
                        "write failed and the original remains recoverable at \(backupURL.path): "
                            + installError.localizedDescription
                    )
                }
                throw installError
            }
            let removedBackup = backupName.withCString {
                Darwin.unlinkat(parentFD, $0, 0)
            }
            if removedBackup != 0 {
                let cleanupError = posixError("could not remove the secured file backup")
                finalName.withCString { _ = Darwin.unlinkat(parentFD, $0, 0) }
                let restored = backupName.withCString { backupPointer in
                    finalName.withCString { finalPointer in
                        renameatx_np(
                            parentFD, backupPointer,
                            parentFD, finalPointer,
                            UInt32(RENAME_EXCL)
                        )
                    }
                }
                guard restored == 0 else {
                    throw LocalError(
                        "write failed and the original remains in \(backupName): "
                            + cleanupError.localizedDescription
                    )
                }
                throw cleanupError
            }
            shouldRemoveTemporary = false
        } else {
            let installed = temporaryName.withCString { temporaryPointer in
                finalName.withCString { finalPointer in
                    renameatx_np(parentFD, temporaryPointer, parentFD, finalPointer, UInt32(RENAME_EXCL))
                }
            }
            guard installed == 0 else { throw posixError("could not create the destination file") }
            shouldRemoveTemporary = false
        }
    }

    private static func entryIdentity(
        parentDescriptor: Int32, filename: String
    ) throws -> FileIdentity? {
        var metadata = stat()
        let status = filename.withCString {
            fstatat(parentDescriptor, $0, &metadata, AT_SYMLINK_NOFOLLOW)
        }
        if status == 0 {
            return FileIdentity(device: metadata.st_dev, inode: metadata.st_ino)
        }
        if errno == ENOENT { return nil }
        throw posixError("could not inspect the destination")
    }

    /// Resolve the directory pinned before approval, or create the one
    /// product-owned default folder only after approval. The open home
    /// descriptor prevents a symlink swap from redirecting creation.
    private static func approvedDirectory(
        existing: PinnedPath?, creationHome: PinnedPath?, requestedURL: URL
    ) throws -> PinnedPath {
        if let existing {
            try existing.verifyPathStillNamesPinnedObject()
            return existing
        }
        guard let creationHome else {
            throw LocalError("approved folder is unavailable")
        }
        try creationHome.verifyPathStillNamesPinnedObject()
        let expected = creationHome.url.appendingPathComponent("Rapid Workspace")
            .standardizedFileURL
        guard requestedURL.standardizedFileURL.path == expected.path else {
            throw LocalError("only the default Rapid Workspace folder may be created")
        }
        let status = "Rapid Workspace".withCString {
            mkdirat(creationHome.descriptor, $0, mode_t(0o700))
        }
        guard status == 0 || errno == EEXIST else {
            throw posixError("could not create Rapid Workspace")
        }
        let pinned = try PinnedPath(url: expected, directory: true)
        guard pinned.url.deletingLastPathComponent().path == creationHome.url.path else {
            throw LocalError("the approved folder escaped the home directory")
        }
        return pinned
    }

    private static func posixError(_ context: String) -> LocalError {
        LocalError("\(context): \(String(cString: strerror(errno)))")
    }

    private static func trash(
        approved: PinnedPath?, parent: PinnedPath?, trashDirectory: PinnedPath?
    ) -> ToolCallResult {
        do {
            guard let approved, let parent, let trashDirectory else {
                return failure("local_trash approval expired")
            }
            try approved.verifyPathStillNamesPinnedObject()
            try parent.verifyPathStillNamesPinnedObject()
            try trashDirectory.verifyPathStillNamesPinnedObject()
            let url = approved.url
            let parentFD = parent.descriptor
            let trashFD = trashDirectory.descriptor

            let sourceName = url.lastPathComponent
            let destinationName = "\(sourceName).rapid-\(UUID().uuidString)"
            // renameatx_np(RENAME_EXCL) is the single atomic mutation. Verify
            // the moved inode afterwards; in the vanishingly small race between
            // the pre-check and rename, put the unapproved object back.
            try approved.verifyPathStillNamesPinnedObject()
            try parent.verifyPathStillNamesPinnedObject()
            let renamed = sourceName.withCString { sourcePointer in
                destinationName.withCString { destinationPointer in
                    renameatx_np(parentFD, sourcePointer, trashFD, destinationPointer, UInt32(RENAME_EXCL))
                }
            }
            guard renamed == 0 else { throw posixError("could not move the approved file to Trash") }
            let movedIdentity = try destinationName.withCString { pointer -> FileIdentity in
                var metadata = stat()
                guard fstatat(trashFD, pointer, &metadata, AT_SYMLINK_NOFOLLOW) == 0 else {
                    throw posixError("could not verify the trashed file")
                }
                return FileIdentity(device: metadata.st_dev, inode: metadata.st_ino)
            }
            guard movedIdentity == approved.identity else {
                let restored = destinationName.withCString { destinationPointer in
                    sourceName.withCString { sourcePointer in
                        renameatx_np(trashFD, destinationPointer, parentFD, sourcePointer, UInt32(RENAME_EXCL))
                    }
                }
                if restored == 0 {
                    return failure("local_trash refused because the approved file changed; the replacement was restored to \(url.path)")
                }
                let recoveryName = ".\(sourceName).rapid-recovered-\(UUID().uuidString)"
                let recovered = destinationName.withCString { destinationPointer in
                    recoveryName.withCString { recoveryPointer in
                        renameatx_np(trashFD, destinationPointer, parentFD, recoveryPointer, UInt32(RENAME_EXCL))
                    }
                }
                if recovered == 0 {
                    let recoveryURL = url.deletingLastPathComponent().appendingPathComponent(recoveryName)
                    return failure("local_trash refused because the approved file changed; the replacement was recovered at \(recoveryURL.path)")
                }
                let recoverableURL = trashDirectory.url.appendingPathComponent(destinationName)
                return failure("local_trash refused because the approved file changed; the replacement remains recoverable at \(recoverableURL.path)")
            }
            return ToolCallResult(toolCallID: "", content: "Moved \(Self.displayPath(url)) to Trash. It can be recovered from Finder.")
        } catch { return failure("local_trash error: \(error.localizedDescription)") }
    }

    private static func prepareRun(_ args: RunArgs) throws -> ApprovedRun {
        let requestedWorkingDirectory = args.workingDirectory ?? args.cwd ?? "~/Rapid Workspace"
        let cwd = try safeWorkingDirectory(requestedWorkingDirectory, mustExist: false)
        var isDirectory: ObjCBool = false
        var workingDirectory: PinnedPath?
        var creationHome: PinnedPath?
        if !FileManager.default.fileExists(atPath: cwd.path, isDirectory: &isDirectory),
           requestedWorkingDirectory == "~/Rapid Workspace" {
            let home = FileManager.default.homeDirectoryForCurrentUser.standardizedFileURL
                .resolvingSymlinksInPath()
            creationHome = try PinnedPath(url: home, directory: true)
        } else {
            guard FileManager.default.fileExists(atPath: cwd.path, isDirectory: &isDirectory),
                  isDirectory.boolValue else {
                throw LocalError("local_run working_directory is not a folder")
            }
            workingDirectory = try PinnedPath(
                url: cwd, directory: true,
                presentedURL: try validatedLexicalURL(requestedWorkingDirectory)
            )
        }

        var processArguments = Self.expandingHomeArguments(
            args.argv ?? [], command: args.command
        )
        let executable: URL
        let executablePresentedURL: URL?
        let helperClass: HelperClass
        let developerDirectories = try installedDeveloperDirectories()
        let developerPaths = developerDirectories.map(\.url.path)
        if ["clang", "cc", "gcc"].contains(args.command) {
            try validateCompilerArguments(processArguments)
            executable = try firstExecutable(
                developerPaths.flatMap { root in [
                    "\(root)/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang",
                    "\(root)/usr/bin/clang",
                ] } + ["/usr/bin/clang"],
                command: args.command
            )
            executablePresentedURL = nil
            helperClass = .compiler
            let sdkCandidates = developerPaths.flatMap { root in [
                "\(root)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk",
                "\(root)/SDKs/MacOSX.sdk",
            ] }
            if let sdk = sdkCandidates.first(where: { FileManager.default.fileExists(atPath: $0) }) {
                processArguments.insert(contentsOf: ["-isysroot", sdk], at: 0)
            }
        } else if allowedCommands.contains(args.command) {
            let candidates: [String]
            switch args.command {
            case "python3": candidates = developerPaths.map {
                "\($0)/Library/Frameworks/Python3.framework/Versions/3.9/Resources/Python.app/Contents/MacOS/Python"
            } + ["/usr/bin/python3"]
            case "node": candidates = ["/opt/homebrew/bin/node", "/usr/local/bin/node", "/usr/bin/node"]
            case "ruby": candidates = ["/opt/homebrew/bin/ruby", "/usr/local/bin/ruby", "/usr/bin/ruby"]
            case "go": candidates = ["/opt/homebrew/bin/go", "/usr/local/bin/go", "/usr/bin/go"]
            case "swift": candidates = developerPaths.map {
                "\($0)/Toolchains/XcodeDefault.xctoolchain/usr/bin/swift-frontend"
            }
            default: throw LocalError("local_run command is not in the allowlist")
            }
            executable = try firstExecutable(candidates, command: args.command)
            executablePresentedURL = nil
            helperClass = args.command == "swift" ? .swift : (args.command == "go" ? .go : .none)
            if args.command == "swift" {
                guard processArguments.contains(where: { $0.hasSuffix(".swift") }),
                      let developer = developerPaths.first,
                      let sdk = [
                          "\(developer)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk",
                          "\(developer)/SDKs/MacOSX.sdk",
                      ].first(where: { FileManager.default.fileExists(atPath: $0) }) else {
                    throw LocalError("local_run swift requires a .swift script and an installed macOS SDK")
                }
                let resources = "\(developer)/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift"
                let prebuiltRoot = "\(resources)/macosx/prebuilt-modules"
                let versions = (try? FileManager.default.contentsOfDirectory(atPath: prebuiltRoot)) ?? []
                let version = versions.sorted { lhs, rhs in
                    lhs.compare(rhs, options: .numeric) == .orderedAscending
                }.last
                #if arch(arm64)
                let architectureCandidates = ["arm64", "arm64e"]
                #else
                let architectureCandidates = ["x86_64"]
                #endif
                guard let version,
                      let architecture = architectureCandidates.first(where: {
                          FileManager.default.fileExists(
                              atPath: "\(prebuiltRoot)/\(version)/Swift.swiftmodule/\($0)-apple-macos.swiftmodule"
                          )
                      }) else {
                    throw LocalError("local_run swift could not find a compatible prebuilt standard library")
                }
                let prebuilt = "\(prebuiltRoot)/\(version)"
                processArguments.insert(contentsOf: [
                    "-interpret", "-sdk", sdk, "-resource-dir", resources,
                    "-prebuilt-module-cache-path", prebuilt,
                    "-target", "\(architecture)-apple-macosx\(version)",
                ], at: 0)
            }
        } else {
            executablePresentedURL = try validatedLexicalURL(args.command)
            executable = try safeURL(args.command)
            guard FileManager.default.isExecutableFile(atPath: executable.path) else {
                throw LocalError("local_run command is not in the allowlist or is not a local executable")
            }
            helperClass = .none
        }

        let pinnedExecutable = try PinnedPath(
            url: executable.resolvingSymlinksInPath(), directory: false,
            presentedURL: executablePresentedURL
        )
        var executableMetadata = stat()
        guard fstat(pinnedExecutable.descriptor, &executableMetadata) == 0 else {
            throw posixError("could not inspect the approved executable")
        }
        // Only SIP-protected system/toolchain executables run in place.
        // Package-manager prefixes are mutable even when Rapid selected the
        // path from its allowlist, so pin their bytes and execute a staged copy.
        let executableDigest = isImmutableSystemExecutable(
            pinnedExecutable.url,
            developerDirectories: developerDirectories.map(\.url)
        ) ? nil : try digest(descriptor: pinnedExecutable.descriptor)
        return ApprovedRun(
            arguments: args,
            workingDirectory: workingDirectory,
            creationHome: creationHome,
            requestedWorkingDirectory: cwd,
            executable: pinnedExecutable,
            processArguments: processArguments,
            helperClass: helperClass,
            developerDirectories: developerDirectories,
            executableDigest: executableDigest
        )
    }

    /// Clang must retain `process-fork` so its signed toolchain helpers can run,
    /// but that exception is only safe when workspace-controlled code cannot be
    /// loaded into the compiler process itself. Reject indirect argument files,
    /// frontend passthrough, and every supported plugin-loading spelling before
    /// asking the user to approve the command.
    /// Expand a leading `~/` only for command positions that are filesystem
    /// operands. `local_run` is argv-based rather than shell-based, so literal
    /// data passed to a program must remain byte-for-byte unchanged.
    static func expandingHomeArguments(
        _ arguments: [String],
        command: String,
        home: URL = FileManager.default.homeDirectoryForCurrentUser
    ) -> [String] {
        // Resolve symlinks the same way the sandbox profile does.
        let homePath = canonicalPath(home.standardizedFileURL.path)
        let commandName = command.split(separator: "/").last.map(String.init) ?? command
        let pathIndexes: Set<Int>
        switch commandName {
        case "clang", "cc", "gcc":
            let pathOperandOptions: Set<String> = [
                "-o", "-I", "-F", "-include", "-include-pch",
                "-isystem", "-iquote", "-iframework",
            ]
            let nonPathOperandOptions: Set<String> = ["-D", "-U", "-x", "-std"]
            var indexes: Set<Int> = []
            var nextIsPath = false
            var nextIsLiteral = false
            var afterTerminator = false
            for index in arguments.indices {
                let argument = arguments[index]
                if nextIsPath {
                    indexes.insert(index)
                    nextIsPath = false
                    continue
                }
                if nextIsLiteral {
                    nextIsLiteral = false
                    continue
                }
                if afterTerminator {
                    indexes.insert(index)
                } else if argument == "--" {
                    afterTerminator = true
                } else if pathOperandOptions.contains(argument) {
                    nextIsPath = true
                } else if nonPathOperandOptions.contains(argument) {
                    nextIsLiteral = true
                } else if !argument.hasPrefix("-") {
                    indexes.insert(index)
                }
            }
            pathIndexes = indexes
        case "python3", "node", "ruby":
            let optionOperands: Set<String>
            switch commandName {
            case "python3": optionOperands = ["-W", "-X"]
            case "node": optionOperands = ["-r", "--require", "--loader", "--import", "--conditions"]
            default: optionOperands = ["-I", "-r", "-C", "-E"]
            }
            var scriptIndex: Int?
            var index = 0
            while index < arguments.count {
                let argument = arguments[index]
                if ["-c", "-e", "--eval", "-m"].contains(argument) { break }
                if optionOperands.contains(argument) {
                    index += 2
                    continue
                }
                if argument == "--", index + 1 < arguments.count {
                    scriptIndex = index + 1
                    break
                }
                if !argument.hasPrefix("-") {
                    scriptIndex = index
                    break
                }
                index += 1
            }
            pathIndexes = scriptIndex.map { [$0] } ?? []
        case "swift":
            pathIndexes = Set(arguments.indices.filter { arguments[$0].hasSuffix(".swift") })
        case "go":
            var indexes: Set<Int> = []
            var fileMode = false
            if arguments.first == "run" {
                for index in arguments.indices.dropFirst() {
                    let argument = arguments[index]
                    if argument == "--" { break }
                    if argument.hasPrefix("-") { continue }
                    if indexes.isEmpty {
                        indexes.insert(index)
                        fileMode = argument.hasSuffix(".go")
                        if !fileMode { break }
                    } else if fileMode, argument.hasSuffix(".go") {
                        indexes.insert(index)
                    } else {
                        break
                    }
                }
            }
            pathIndexes = indexes
        default:
            pathIndexes = []
        }
        return arguments.enumerated().map { index, argument in
            if pathIndexes.contains(index), argument.hasPrefix("~/") {
                return homePath + argument.dropFirst(1)
            }
            if ["clang", "cc", "gcc"].contains(commandName) {
                let joinedPathPrefixes = ["-iframework", "-isystem", "-iquote", "-I", "-F", "-o"]
                if let prefix = joinedPathPrefixes.first(where: {
                    argument.hasPrefix($0 + "~/")
                }) {
                    let operand = argument.dropFirst(prefix.count)
                    return prefix + homePath + operand.dropFirst(1)
                }
            }
            return argument
        }
    }

    /// The path a tool result reports back to the model and the transcript.
    /// The user asked for `~/Documents/x`; echoing the resolved absolute path
    /// makes small models conclude the file "went somewhere else".
    static func displayPath(
        _ url: URL,
        home: URL = FileManager.default.homeDirectoryForCurrentUser
    ) -> String {
        let path = canonicalPath(url.standardizedFileURL.path)
        let homePath = canonicalPath(home.standardizedFileURL.path)
        if path == homePath { return "~" }
        if path.hasPrefix(homePath + "/") { return "~" + path.dropFirst(homePath.count) }
        return path
    }

    /// The kernel path of `path`, via POSIX `realpath`. Foundation's
    /// `resolvingSymlinksInPath()` drops the `/private` prefix
    /// (`/private/tmp/x` becomes `/tmp/x`), but sandbox profile filters match
    /// the kernel path, so a home or workspace under `/private/tmp` or
    /// `/private/var` would otherwise be denied every read and write.
    static func canonicalPath(_ path: String) -> String {
        guard let resolved = realpath(path, nil) else { return path }
        defer { free(resolved) }
        return String(cString: resolved)
    }

    static func validateCompilerArguments(_ arguments: [String]) throws {
        let exactDenied = Set([
            "-Xclang", "-cc1", "-cc1as", "-cc1gen-reproducer",
            "-load", "-plugin", "--config", "-config", "-mllvm",
        ])
        let prefixesDenied = [
            "@", "-X", "-Wl,", "-Wa,", "-Wp,", "--config=", "-config=",
            "-fplugin", "-fpass-plugin",
        ]
        guard !arguments.contains(where: { argument in
            exactDenied.contains(argument)
                || prefixesDenied.contains(where: argument.hasPrefix)
        }) else {
            throw LocalError("local_run compiler plugins and indirect argument files are unavailable")
        }
    }

    static func isImmutableSystemExecutable(
        _ executable: URL,
        developerDirectories: [URL]
    ) -> Bool {
        let path = executable.standardizedFileURL.path
        if path.hasPrefix("/usr/bin/") || path.hasPrefix("/bin/") {
            return true
        }
        return developerDirectories.contains { root in
            let rootPath = root.standardizedFileURL.path
            return path.hasPrefix(rootPath + "/")
        }
    }

    /// Developer roots in the same precedence order as the active toolchain.
    /// CI and beta-Xcode users commonly select an app named `Xcode_26.x.app`;
    /// hard-coding `/Applications/Xcode.app` makes the executable launch but
    /// leaves its framework and SDK outside the sandbox.
    /// Whether `local_run swift` can resolve a toolchain on this Mac: a
    /// root-owned Xcode or Command Line Tools with the default toolchain's
    /// `swift-frontend`. A user-owned `Xcode_26.x.app` (common on CI
    /// runners) fails closed, so callers and tests can tell "unsupported
    /// here" from a broken sandbox.
    static func swiftScriptToolchainIsAvailable() -> Bool {
        guard let roots = try? installedDeveloperDirectories() else { return false }
        return roots.contains { root in
            FileManager.default.isExecutableFile(
                atPath: root.url.path + "/Toolchains/XcodeDefault.xctoolchain/usr/bin/swift-frontend"
            )
        }
    }

    private static func installedDeveloperDirectories() throws -> [PinnedPath] {
        var candidates: [String] = []
        if let environment = ProcessInfo.processInfo.environment["DEVELOPER_DIR"],
           environment.hasPrefix("/") {
            candidates.append(environment)
        }
        let selected = URL(fileURLWithPath: "/var/db/xcode_select_link")
            .resolvingSymlinksInPath().path
        if selected != "/var/db/xcode_select_link" { candidates.append(selected) }
        candidates += [
            "/Applications/Xcode.app/Contents/Developer",
            "/Library/Developer/CommandLineTools",
        ]
        var seen = Set<String>()
        return try candidates.compactMap { path in
            var candidate = URL(fileURLWithPath: path).standardizedFileURL
            if candidate.pathExtension == "app" {
                candidate.appendPathComponent("Contents/Developer", isDirectory: true)
            }
            candidate = candidate.resolvingSymlinksInPath()
            let normalized = candidate.path
            guard seen.insert(normalized).inserted else { return nil }
            let isXcode = normalized.hasPrefix("/Applications/")
                && normalized.hasSuffix(".app/Contents/Developer")
            let isCommandLineTools = normalized == "/Library/Developer/CommandLineTools"
            guard isXcode || isCommandLineTools else { return nil }
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: normalized, isDirectory: &isDirectory),
                  isDirectory.boolValue else { return nil }
            var metadata = stat()
            guard lstat(normalized, &metadata) == 0, metadata.st_uid == 0 else { return nil }
            return try PinnedPath(url: candidate, directory: true)
        }
    }

    private static func firstExecutable(_ candidates: [String], command: String) throws -> URL {
        guard let candidate = candidates.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) else {
            throw LocalError("local_run could not find an installed \(command) executable")
        }
        return URL(fileURLWithPath: candidate).resolvingSymlinksInPath()
    }

    private static func runCommand(approved: ApprovedRun?) -> ToolCallResult {
        do {
            guard let approved else { return failure("local_run approval expired") }
            let workingDirectory = try approvedDirectory(
                existing: approved.workingDirectory,
                creationHome: approved.creationHome,
                requestedURL: approved.requestedWorkingDirectory
            )
            try approved.executable.verifyPathStillNamesPinnedObject()
            let args = approved.arguments
            let cwd = try currentURL(for: workingDirectory.descriptor)
            var processArguments = approved.processArguments
            for developerDirectory in approved.developerDirectories {
                try developerDirectory.verifyPathStillNamesPinnedObject()
            }

            let temporaryName = ".rapid-tmp-\(UUID().uuidString)"
            let created = temporaryName.withCString {
                mkdirat(workingDirectory.descriptor, $0, mode_t(0o700))
            }
            guard created == 0 else { throw posixError("could not create the private command folder") }
            let temporary = cwd.appendingPathComponent(temporaryName)
            defer { try? FileManager.default.removeItem(at: temporary) }
            if args.command == "swift" {
                processArguments.insert(contentsOf: [
                    "-module-cache-path", temporary.appendingPathComponent("modules").path,
                ], at: 0)
            }
            let executable = try stableExecutable(
                approved.executable,
                expectedDigest: approved.executableDigest,
                in: temporary
            )
            let sandboxArguments = [
                "-p", sandboxProfile(
                    workingDirectory: cwd,
                    temporaryDirectory: temporary,
                    executable: executable,
                    originalExecutable: approved.executable.url,
                    helperClass: approved.helperClass,
                    developerDirectories: approved.developerDirectories,
                    approvedCommand: args.command
                ),
                executable.path,
            ] + processArguments
            var environment = [
                "HOME": FileManager.default.homeDirectoryForCurrentUser.path,
                "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
                "TMPDIR": temporary.path,
                "LANG": "en_US.UTF-8",
            ]
            if args.command == "go" {
                let original = approved.executable.url
                environment["GOROOT"] = original.deletingLastPathComponent().deletingLastPathComponent().path
            }
            if args.command == "swift", let developer = approved.developerDirectories.first {
                environment["DEVELOPER_DIR"] = developer.url.path
            }
            let timeout = min(max(args.timeoutSeconds ?? 15, 1), 30)
            let outcome = try spawnSandboxed(
                arguments: sandboxArguments,
                environment: environment,
                workingDirectoryDescriptor: workingDirectory.descriptor,
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

    private static func stableExecutable(
        _ approved: PinnedPath,
        expectedDigest: Data?,
        in temporaryDirectory: URL
    ) throws -> URL {
        if expectedDigest == nil {
            return approved.url
        }

        let destination = temporaryDirectory.appendingPathComponent("approved-tool")
        let output = Darwin.open(
            destination.path,
            O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
            mode_t(0o700)
        )
        guard output >= 0 else { throw posixError("could not stage the approved executable") }
        defer { Darwin.close(output) }
        guard lseek(approved.descriptor, 0, SEEK_SET) >= 0 else {
            throw posixError("could not read the approved executable")
        }
        var buffer = [UInt8](repeating: 0, count: 64 * 1_024)
        while true {
            let count = Darwin.read(approved.descriptor, &buffer, buffer.count)
            if count == 0 { break }
            if count < 0 {
                if errno == EINTR { continue }
                throw posixError("could not read the approved executable")
            }
            var written = 0
            while written < count {
                let result = buffer.withUnsafeBytes { rawBuffer in
                    Darwin.write(output, rawBuffer.baseAddress!.advanced(by: written), count - written)
                }
                guard result > 0 else { throw posixError("could not stage the approved executable") }
                written += result
            }
        }
        guard let expectedDigest,
              try digest(descriptor: output) == expectedDigest else {
            throw LocalError("the approved executable changed after approval")
        }
        guard fsync(output) == 0, fchmod(output, mode_t(0o700)) == 0 else {
            throw posixError("could not finalize the approved executable")
        }
        return destination
    }

    private static func digest(descriptor: Int32) throws -> Data {
        guard lseek(descriptor, 0, SEEK_SET) >= 0 else {
            throw posixError("could not hash the approved executable")
        }
        var hasher = SHA256()
        var buffer = [UInt8](repeating: 0, count: 64 * 1_024)
        while true {
            let count = Darwin.read(descriptor, &buffer, buffer.count)
            if count == 0 { break }
            if count < 0 {
                if errno == EINTR { continue }
                throw posixError("could not hash the approved executable")
            }
            hasher.update(data: Data(buffer.prefix(count)))
        }
        return Data(hasher.finalize())
    }

    private struct CommandOutcome {
        let exitCode: Int32
        let timedOut: Bool
        let stdout: Data
        let stderr: Data
    }

    private static func currentURL(for descriptor: Int32) throws -> URL {
        var path = [CChar](repeating: 0, count: Int(PATH_MAX))
        guard fcntl(descriptor, F_GETPATH, &path) == 0 else {
            throw posixError("could not locate the approved working directory")
        }
        let terminator = path.firstIndex(of: 0) ?? path.endIndex
        let pathString = String(
            decoding: path[..<terminator].map { UInt8(bitPattern: $0) },
            as: UTF8.self
        )
        let url = URL(fileURLWithPath: pathString).standardizedFileURL
        _ = try validatedLexicalURL(url.path)
        return url
    }

    private static func spawnSandboxed(
        arguments: [String],
        environment: [String: String],
        workingDirectoryDescriptor: Int32,
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
        guard posix_spawn_file_actions_init(&actions) == 0 else {
            throw LocalError("could not initialize the command launcher")
        }
        defer { posix_spawn_file_actions_destroy(&actions) }
        guard posix_spawnattr_init(&attributes) == 0 else {
            throw LocalError("could not initialize the command launcher")
        }
        defer { posix_spawnattr_destroy(&attributes) }
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
        posix_spawn_file_actions_addfchdir_np(&actions, workingDirectoryDescriptor) == 0,
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
        var descendants = Set<pid_t>()
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            collectDescendants(of: pid, into: &descendants)
            let waited = waitpid(pid, &status, WNOHANG)
            if waited == pid {
                reaped = true
                break
            }
            if waited == -1, errno != EINTR {
                let monitorError = posixError("could not monitor the approved command")
                _ = kill(-pid, SIGKILL)
                while waitpid(pid, &status, 0) == -1, errno == EINTR {}
                throw monitorError
            }
            Thread.sleep(forTimeInterval: 0.02)
        }
        collectDescendants(of: pid, into: &descendants)
        let timedOut = !reaped
        if timedOut {
            _ = kill(-pid, SIGTERM)
            signalProcesses(descendants, signal: SIGTERM)
            let grace = Date().addingTimeInterval(0.5)
            while Date() < grace {
                collectDescendants(of: pid, into: &descendants)
                collectDescendants(of: Array(descendants), into: &descendants)
                let waited = waitpid(pid, &status, WNOHANG)
                if waited == pid {
                    reaped = true
                    break
                }
                Thread.sleep(forTimeInterval: 0.02)
            }
            if !reaped {
                _ = kill(-pid, SIGKILL)
                signalProcesses(descendants, signal: SIGKILL)
                // SIGKILL cannot be ignored. Once it has been sent, perform an
                // EINTR-safe blocking reap so a slow kernel teardown cannot
                // leave a zombie behind after the one-second polling window.
                while true {
                    let waited = waitpid(pid, &status, 0)
                    if waited == pid {
                        reaped = true
                        break
                    }
                    if waited == -1, errno == EINTR { continue }
                    break
                }
            }
        }
        // The approved command owns a fresh process group. Even when its
        // leader exits normally, descendants must not survive the bounded
        // action and continue writing in the background.
        terminateProcessGroup(pid)
        collectDescendants(of: Array(descendants), into: &descendants)
        terminateProcesses(descendants)
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

    private static func childPIDs(of parent: pid_t) -> [pid_t] {
        guard parent > 0 else { return [] }
        var capacity = 32
        while capacity <= 4_096 {
            var children = [pid_t](repeating: 0, count: capacity)
            let count = children.withUnsafeMutableBytes { buffer in
                rapidProcListChildPIDs(parent, buffer.baseAddress, Int32(buffer.count))
            }
            guard count >= 0 else { return [] }
            if Int(count) < capacity {
                return Array(children.prefix(Int(count))).filter { $0 > 0 }
            }
            capacity *= 2
        }
        return []
    }

    private static func collectDescendants(
        of parent: pid_t,
        into descendants: inout Set<pid_t>
    ) {
        var pending = [parent]
        while let current = pending.popLast() {
            for child in childPIDs(of: current) where descendants.insert(child).inserted {
                pending.append(child)
            }
        }
    }

    private static func collectDescendants(
        of parents: [pid_t],
        into descendants: inout Set<pid_t>
    ) {
        for parent in parents { collectDescendants(of: parent, into: &descendants) }
    }

    private static func signalProcesses(_ processes: Set<pid_t>, signal: Int32) {
        for process in processes where process > 0 { _ = kill(process, signal) }
    }

    private static func terminateProcesses(_ processes: Set<pid_t>) {
        guard !processes.isEmpty else { return }
        signalProcesses(processes, signal: SIGTERM)
        let grace = Date().addingTimeInterval(0.1)
        while Date() < grace {
            if processes.allSatisfy({ kill($0, 0) == -1 && errno == ESRCH }) { return }
            Thread.sleep(forTimeInterval: 0.01)
        }
        signalProcesses(processes, signal: SIGKILL)
    }

    private static func terminateProcessGroup(_ leader: pid_t) {
        guard leader > 0 else { return }
        _ = kill(-leader, SIGTERM)
        let grace = Date().addingTimeInterval(0.1)
        while Date() < grace {
            if kill(-leader, 0) == -1, errno == ESRCH { return }
            Thread.sleep(forTimeInterval: 0.01)
        }
        _ = kill(-leader, SIGKILL)
        let reapDeadline = Date().addingTimeInterval(1)
        while Date() < reapDeadline {
            if kill(-leader, 0) == -1, errno == ESRCH { return }
            Thread.sleep(forTimeInterval: 0.01)
        }
    }

    private static func sandboxProfile(
        workingDirectory: URL,
        temporaryDirectory: URL,
        executable: URL,
        originalExecutable: URL,
        helperClass: HelperClass,
        developerDirectories: [PinnedPath],
        approvedCommand: String
    ) -> String {
        func quoted(_ path: String) -> String {
            "\"" + path.replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"") + "\""
        }
        let workingDirectoryPath = canonicalPath(workingDirectory.path)
        let temporaryDirectoryPath = canonicalPath(temporaryDirectory.path)
        var readableFilters = [
            "(literal \"/\")",
            "(subpath \(quoted(workingDirectoryPath)))",
            "(subpath \(quoted(temporaryDirectoryPath)))",
            "(literal \(quoted(canonicalPath(executable.path))))",
            "(subpath \"/System\")",
            "(subpath \"/usr/lib\")",
            "(subpath \"/usr/share\")",
            "(literal \"/dev/null\")",
            "(literal \"/dev/zero\")",
            "(literal \"/dev/random\")",
            "(literal \"/dev/urandom\")",
            "(subpath \"/private/var/db/timezone\")",
        ]
        var metadataFilters: [String] = []
        let developerPaths = developerDirectories.map(\.url.path)
        if developerPaths.contains(where: {
            originalExecutable.path.hasPrefix($0 + "/")
        }) || helperClass != .none || approvedCommand == "python3" {
            readableFilters += developerPaths.map { "(subpath \(quoted($0)))" }
            metadataFilters += [
                "(literal \"/Applications\")",
                "(literal \"/Library\")",
                "(literal \"/Library/Developer\")",
            ]
            for root in developerPaths where root.hasPrefix("/Applications/") {
                let contents = URL(fileURLWithPath: root).deletingLastPathComponent().path
                let application = URL(fileURLWithPath: contents).deletingLastPathComponent().path
                metadataFilters += [
                    "(literal \(quoted(application)))",
                    "(literal \(quoted(contents)))",
                ]
            }
        }
        if originalExecutable.path.hasPrefix("/opt/homebrew/") {
            // Homebrew's executable symlinks resolve into Cellar and its
            // linked libraries resolve through opt. Never expose the whole
            // prefix: /opt/homebrew/etc can contain unrelated credentials.
            readableFilters += [
                "(subpath \"/opt/homebrew/Cellar\")",
                "(subpath \"/opt/homebrew/opt\")",
            ]
            metadataFilters += [
                "(literal \"/opt\")",
                "(literal \"/opt/homebrew\")",
            ]
        } else if originalExecutable.path.hasPrefix("/usr/local/") {
            readableFilters += [
                "(subpath \"/usr/local/Cellar\")",
                "(subpath \"/usr/local/opt\")",
                "(subpath \(quoted(originalExecutable.deletingLastPathComponent().path)))",
            ]
            metadataFilters.append("(literal \"/usr/local\")")
        }

        var executableFilters = ["(literal \(quoted(canonicalPath(executable.path))))"]
        if approvedCommand == "python3" {
            executableFilters += [
                "(literal \"/opt/homebrew/bin/python3\")",
                "(literal \"/usr/local/bin/python3\")",
                "(literal \"/usr/bin/python3\")",
                "(literal \"/Applications/Xcode.app/Contents/Developer/usr/bin/python3\")",
                "(literal \"/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/Resources/Python.app/Contents/MacOS/Python\")",
            ]
        }
        switch helperClass {
        case .none:
            break
        case .compiler:
            executableFilters += developerPaths.flatMap { root in [
                "(subpath \(quoted(root + "/Toolchains")))",
                "(subpath \(quoted(root + "/usr/bin")))",
                "(subpath \(quoted(root + "/Platforms/MacOSX.platform/Developer/usr/bin")))",
            ] } + ["(literal \"/usr/bin/ld\")"]
        case .swift:
            executableFilters += developerPaths.flatMap { root in [
                "(subpath \(quoted(root + "/Toolchains")))",
                "(subpath \(quoted(root + "/usr/bin")))",
                "(subpath \(quoted(root + "/Platforms/MacOSX.platform/Developer/usr/bin")))",
            ] }
        case .go:
            if originalExecutable.path.hasPrefix("/opt/homebrew/") {
                executableFilters.append("(subpath \"/opt/homebrew/Cellar/go\")")
            } else if originalExecutable.path.hasPrefix("/usr/local/") {
                executableFilters.append("(subpath \(quoted(originalExecutable.deletingLastPathComponent().deletingLastPathComponent().path)))")
            }
        }
        let readFilters = readableFilters.joined(separator: "\n                ")
        let metadataRule = metadataFilters.isEmpty ? "" : """
        (allow file-read-metadata
            \(metadataFilters.joined(separator: "\n            ")))
        """
        let processFilters = executableFilters.joined(separator: "\n                ")
        // Interpreters and user-selected binaries execute user-authored code
        // in-process. Deny fork there so a child cannot call setsid(), become
        // orphaned between process-tree polls, and outlive the approved action.
        // Toolchain drivers retain fork solely for their constrained helpers.
        let mayForkTrustedHelpers = helperClass == .compiler || helperClass == .go
        let forkRule = mayForkTrustedHelpers ? "" : "(deny process-fork)"
        return """
        (version 1)
        (allow default)
        (deny network*)
        (deny mach-lookup)
        \(forkRule)
        ; File reads are default-denied. Restore only the approved workspace,
        ; private temp folder, exact executable, immutable runtime data, and
        ; the toolchain explicitly implied by the approved command.
        (deny file-read*
            (require-not (require-any
                \(readFilters))))
        \(metadataRule)
        (deny file-write*)
        (allow file-write*
            (subpath \(quoted(workingDirectoryPath)))
            (subpath \(quoted(temporaryDirectoryPath))))
        ; Interpreters may execute themselves recursively. Compilers receive
        ; only their explicit platform toolchain helpers.
        (deny process-exec
            (require-not (require-any
                \(processFilters))))
        """
    }

    private static func failure(_ message: String, executed: Bool = true) -> ToolCallResult {
        ToolCallResult(toolCallID: "", content: message, isError: true, executed: executed)
    }

    private static func withToolCallID(_ result: ToolCallResult, _ id: String) -> ToolCallResult {
        ToolCallResult(
            toolCallID: id,
            content: result.content,
            isError: result.isError,
            failureKind: result.failureKind,
            executed: result.executed
        )
    }

    private struct LocalError: LocalizedError {
        let message: String
        init(_ message: String) { self.message = message }
        var errorDescription: String? { message }
    }
}
