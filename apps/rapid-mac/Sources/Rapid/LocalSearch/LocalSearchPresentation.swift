import Foundation

enum LocalSearchPath {
    static func folderTitle(_ path: String) -> String {
        path == "/" ? String(localized: "Whole Mac") : abbreviated(path)
    }

    static func resultFolder(_ result: LocalSearchResult, folders: [LocalSearchFolder]) -> String {
        let parent = URL(fileURLWithPath: result.path).deletingLastPathComponent().path
        guard let root = folders.first(where: { $0.id == result.folderId })?.path else { return abbreviated(parent) }
        if root == "/" { return abbreviated(parent) }
        guard parent == root || parent.hasPrefix(root + "/") else { return abbreviated(parent) }
        return URL(fileURLWithPath: root).lastPathComponent + parent.dropFirst(root.count)
    }

    static func isCovered(_ url: URL, folders: [LocalSearchFolder]) -> Bool {
        let path = url.resolvingSymlinksInPath().path
        return folders.contains {
            let root = URL(fileURLWithPath: $0.path).resolvingSymlinksInPath().path
            return root == "/" || path == root || path.hasPrefix(root + "/")
        }
    }

    /// Use the real user home, including when the app runs in a container.
    static func abbreviated(_ path: String, home: URL = FileManager.default.homeDirectoryForCurrentUser) -> String {
        let prefix = home.path
        guard prefix != "/" else { return path }
        if path == prefix { return "~" }
        guard path.hasPrefix(prefix + "/") else { return path }
        return "~" + path.dropFirst(prefix.count)
    }
}

struct LocalSearchFoldersDisclosure {
    private var userExpanded: Bool?

    func isExpanded(query: String, hasResults: Bool) -> Bool {
        userExpanded ?? (query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !hasResults)
    }

    mutating func toggle(query: String, hasResults: Bool) {
        userExpanded = !isExpanded(query: query, hasResults: hasResults)
    }
}


enum LocalSearchKind: String, Codable, Sendable {
    case text, image, video
    init(from decoder: Decoder) throws {
        self = Self(rawValue: try decoder.singleValueContainer().decode(String.self)) ?? .text
    }
}

struct LocalSearchKindCounts: Decodable, Sendable, Equatable {
    let text: Int?
    let image: Int?
    let video: Int?
    var label: String {
        String(localized: "\((text ?? 0).formatted()) documents · \((image ?? 0).formatted()) images · \((video ?? 0).formatted()) videos")
    }
}

enum LocalSearchKindFilter: CaseIterable {
    case all, text, image, video
    static func visible(vision: Bool?) -> [Self] {
        vision == true ? allCases : [.all, .text]
    }
    var kinds: [LocalSearchKind]? {
        switch self {
        case .all: nil
        case .text: [.text]
        case .image: [.image]
        case .video: [.video]
        }
    }
    var isGrid: Bool { self == .image || self == .video }
    var title: String {
        switch self {
        case .all: String(localized: "All")
        case .text: String(localized: "Documents")
        case .image: String(localized: "Images")
        case .video: String(localized: "Videos")
        }
    }
    var identifier: String {
        switch self {
        case .all: "LocalSearch.Kind.All"
        case .text: "LocalSearch.Kind.Text"
        case .image: "LocalSearch.Kind.Image"
        case .video: "LocalSearch.Kind.Video"
        }
    }
}

enum LocalSearchTimestamp {
    static func formatted(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds >= 0, seconds < Double(Int.max) else { return "0:00" }
        let total = Int(seconds.rounded(.down))
        if total >= 3600 { return String(format: "%d:%02d:%02d", total / 3600, total / 60 % 60, total % 60) }
        return String(format: "%d:%02d", total / 60, total % 60)
    }
}

enum LocalSearchFolderShortcut: String, CaseIterable {
    case desktop = "Desktop", documents = "Documents", downloads = "Downloads", wholeMac = "WholeMac"

    var title: String {
        switch self {
        case .desktop: String(localized: "Desktop")
        case .documents: String(localized: "Documents")
        case .downloads: String(localized: "Downloads")
        case .wholeMac: String(localized: "Whole Mac")
        }
    }

    var url: URL? {
        let directory: FileManager.SearchPathDirectory
        switch self {
        case .desktop: directory = .desktopDirectory
        case .documents: directory = .documentDirectory
        case .downloads: directory = .downloadsDirectory
        case .wholeMac: return URL(fileURLWithPath: "/", isDirectory: true)
        }
        return FileManager.default.urls(for: directory, in: .userDomainMask).first
    }

    var identifier: String { "LocalSearch.AddFolder." + rawValue }
}


extension LocalSearchFolder {
    var countLabel: String { String(localized: "\(filesIndexed) files indexed · \(filesSkipped) not indexed") }
    var recoveryTitle: String? {
        switch state {
        case .missing: String(localized: "Locate folder…")
        case .error: String(localized: "Try again")
        case .waitingForModel: String(localized: "Download search model")
        default: nil
        }
    }
    var skippedExplanation: String? {
        guard let skipReasons, !skipReasons.isEmpty else { return nil }
        return skipReasons.sorted { $0.key < $1.key }.map { reason, count in
            String(localized: "\(LocalSearchSkipReason.localized(reason)): \(count)")
        }.joined(separator: "\n")
    }
}

/// The preview follows server section order while keeping each kind's ranking.
struct LocalSearchSection: Identifiable, Equatable {
    let kind: LocalSearchKind
    let results: [LocalSearchResult]
    let total: Int
    var id: String { kind.rawValue }
    var filter: LocalSearchKindFilter {
        switch kind { case .text: .text; case .image: .image; case .video: .video }
    }
    var title: String { filter.title }
    static func previews(_ groups: [LocalSearchGroup]) -> [Self] {
        groups.compactMap { group in
            guard !group.results.isEmpty else { return nil }
            return Self(kind: group.kind, results: Array(group.results.prefix(group.kind == .text ? 5 : 8)), total: group.total)
        }
    }
}


enum LocalSearchSkipReason {
    static func localized(_ reason: String) -> String {
        switch reason {
        case "Unsupported file type": String(localized: "Unsupported file type")
        case "Too large": String(localized: "Too large")
        case "No readable text": String(localized: "No readable text")
        case "Scanned PDF without text": String(localized: "Scanned PDF without text")
        case "Password-protected": String(localized: "Password-protected")
        case "Could not be read": String(localized: "Could not be read")
        case "Private or system file": String(localized: "Private or system file")
        case "Audio is not supported yet": String(localized: "Audio is not supported yet")
        default: String(localized: "File contents could not be indexed")
        }
    }
}

extension LocalSearchStatus.Indexing {
    var progressLabel: String {
        if let pausedReason {
            return pausedReason == "chat"
                ? String(localized: "Indexing paused while a chat response is generating — it resumes automatically")
                : String(localized: "Indexing paused — it resumes automatically")
        }
        return filesTotal > 0
            ? String(localized: "Indexing: \(filesDone) of \(filesTotal) files")
            : String(localized: "Scanning and indexing: \(filesDone) files processed")
    }
}

extension LocalSearchResult {
    var nameOnlyExplanation: String? {
        guard !contentIndexed else { return nil }
        guard let skipReason else { return String(localized: "Matched by file name") }
        return String(localized: "Matched by file name — \(LocalSearchSkipReason.localized(skipReason))")
    }
    var snippetLabel: String { nameOnlyExplanation ?? snippet }

    func accessibilityLabel(folders: [LocalSearchFolder]) -> String {
        let folder = LocalSearchPath.resultFolder(self, folders: folders)
        let date = Date(timeIntervalSince1970: modifiedAt).formatted(date: .abbreviated, time: .omitted)
        let components: [String?] = [
            URL(fileURLWithPath: path).lastPathComponent,
            page.map { String(localized: "page \($0)") },
            line.map { String(localized: "line \($0)") },
            snippetLabel,
            String(localized: "in \(folder)"),
            String(localized: "modified \(date)")
        ]
        return components.compactMap { $0 }
            .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            .joined(separator: ", ")
    }
}
