import AppKit
import Foundation
import Observation

/// Proper nouns handed to the ASR model as a decoding hint.
///
/// The size cap is not a UI nicety — it is the whole design constraint. Measured
/// on Qwen3-ASR-1.7B and whisper-large-v3-turbo with identical audio, repeated
/// runs with zero variance:
///
/// | terms | outcome                                    |
/// |-------|--------------------------------------------|
/// |     0 | `herdr` → "Herder", `throughput` → "Throbot" |
/// |  7–16 | both correct                                |
/// | 30–35 | back to "Herder" / "Thruppet"               |
///
/// Attention gets diluted as the list grows, so a longer list is actively worse
/// than a short one. Worse, the effect is not separable per term: dropping an
/// unrelated entry has been observed to regress a *different* one. That is why
/// terms carry an explicit active/parked distinction instead of everything being
/// sent at once, and why the UI surfaces the budget rather than hiding it.
@MainActor
@Observable
final class DictationVocabulary {
    /// Upper bound on terms sent with any single request.
    static let activeLimit = 20

    private(set) var terms: [Term] = []
    private(set) var suggestions: [String] = []
    private(set) var isScanning = false

    struct Term: Codable, Identifiable, Hashable, Sendable {
        var text: String
        var isActive: Bool
        /// Bumped whenever the user fixes a transcript containing this term, so
        /// words that actually cause trouble win the limited budget.
        var corrections: Int

        var id: String { text }

        init(text: String, isActive: Bool = true, corrections: Int = 0) {
            self.text = text
            self.isActive = isActive
            self.corrections = corrections
        }
    }

    private let storeURL: URL
    /// Preserve mutation order on disk. Independent detached writes can finish
    /// out of order, so a quick Add → Remove may otherwise resurrect the term
    /// on the next launch when the older Add snapshot lands last.
    private let persistenceQueue = DispatchQueue(label: "ai.rapidmlx.dictation-vocabulary")

    init(storeURL: URL? = nil) {
        self.storeURL = storeURL ?? Self.defaultStoreURL()
        load()
    }

    // MARK: - Derived

    var activeTerms: [Term] {
        terms.filter(\.isActive).prefix(Self.activeLimit).map { $0 }
    }

    var activeCount: Int { min(terms.filter(\.isActive).count, Self.activeLimit) }

    var isOverBudget: Bool { terms.filter(\.isActive).count > Self.activeLimit }

    /// The string sent as the `context` form field. Empty when nothing is
    /// active, in which case the field is omitted entirely — an empty hint still
    /// costs decoding attention.
    var contextPrompt: String {
        let words = activeTerms.map(\.text)
        guard !words.isEmpty else { return "" }
        return "专有名词 / proper nouns: " + words.joined(separator: ", ") + "。"
    }

    // MARK: - Mutation

    func add(_ text: String, active: Bool = true) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !terms.contains(where: { $0.text == trimmed }) else { return }
        // A newly added term is the one the user cares about right now, so it
        // goes to the front where it is least likely to be crowded out.
        terms.insert(Term(text: trimmed, isActive: active), at: 0)
        suggestions.removeAll { $0 == trimmed }
        save()
    }

    func remove(_ text: String) {
        terms.removeAll { $0.text == text }
        save()
    }

    func setActive(_ text: String, _ active: Bool) {
        guard let index = terms.firstIndex(where: { $0.text == text }) else { return }
        terms[index].isActive = active
        save()
    }

    /// Records that a transcript had to be corrected to this term, and makes
    /// sure it is in the budget next time.
    func noteCorrection(to text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        if let index = terms.firstIndex(where: { $0.text == trimmed }) {
            terms[index].corrections += 1
            terms[index].isActive = true
            let promoted = terms.remove(at: index)
            terms.insert(promoted, at: 0)
        } else {
            terms.insert(Term(text: trimmed, isActive: true, corrections: 1), at: 0)
        }
        save()
    }

    // MARK: - Discovery

    /// Users do not maintain word lists by hand, so the first list has to come
    /// from somewhere. Local project directories and installed apps are where
    /// personal jargon actually lives.
    func scanForSuggestions() async {
        guard !isScanning else { return }
        isScanning = true
        defer { isScanning = false }

        let known = Set(terms.map { $0.text.lowercased() })
        let found = await Task.detached(priority: .utility) {
            Self.discoverCandidates()
        }.value

        suggestions = found
            .filter { !known.contains($0.lowercased()) }
            .sorted { $0.localizedCaseInsensitiveCompare($1) == .orderedAscending }
    }

    func dismissSuggestion(_ text: String) {
        suggestions.removeAll { $0 == text }
    }

    /// Wait until all mutations queued before this call have reached disk.
    /// Used by deterministic tests and any future shutdown flush path.
    func waitForPersistence() async {
        await withCheckedContinuation { continuation in
            persistenceQueue.async { continuation.resume() }
        }
    }

    private nonisolated static func discoverCandidates() -> [String] {
        let fm = FileManager.default
        let home = fm.homeDirectoryForCurrentUser
        var names: Set<String> = []

        for folder in ["work", "Developer", "Projects", "src", "Code", "repos"] {
            let dir = home.appendingPathComponent(folder, isDirectory: true)
            guard let entries = try? fm.contentsOfDirectory(
                at: dir,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            ) else { continue }
            for entry in entries.prefix(200) {
                let isDir = (try? entry.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory
                guard isDir == true else { continue }
                names.insert(entry.lastPathComponent)
            }
        }

        if let apps = try? fm.contentsOfDirectory(
            at: URL(fileURLWithPath: "/Applications"),
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) {
            for app in apps where app.pathExtension == "app" {
                names.insert(app.deletingPathExtension().lastPathComponent)
            }
        }

        return names.filter(isLikelyProperNoun).sorted()
    }

    /// Keeps names an ASR model plausibly gets wrong and drops ones it already
    /// knows. Ordinary dictionary words are exactly what a modern model handles
    /// correctly unaided, and spending budget on them costs accuracy elsewhere.
    nonisolated static func isLikelyProperNoun(_ name: String) -> Bool {
        guard name.count >= 3, name.count <= 24 else { return false }
        guard name.rangeOfCharacter(from: .whitespacesAndNewlines) == nil
                || name.split(separator: " ").count <= 2 else { return false }
        guard name.rangeOfCharacter(from: CharacterSet(charactersIn: "._@#")) == nil else {
            return false
        }
        if commonWords.contains(name.lowercased()) { return false }

        let hasDigit = name.rangeOfCharacter(from: .decimalDigits) != nil
        let hasHyphen = name.contains("-")
        // Interior capitals (vLLM, MacBook) or an all-caps run mark a name that
        // is spelled, not spoken as an ordinary word.
        let interiorUppercase = name.dropFirst().rangeOfCharacter(from: .uppercaseLetters) != nil

        return hasDigit || hasHyphen || interiorUppercase || !isDictionaryWord(name)
    }

    /// Looks the name up in the system word list.
    ///
    /// The earlier shape of this test — "all lowercase and short, so probably a
    /// real word" — rejected exactly the names worth hinting: `herdr` fit the
    /// pattern perfectly and never got suggested, while being the kind of
    /// invented spelling a recogniser reliably mangles. A real dictionary is
    /// the only way to tell an ordinary word from a coined one.
    private nonisolated static func isDictionaryWord(_ name: String) -> Bool {
        systemWords.contains(name.lowercased())
    }

    /// `/usr/share/dict/words` ships with macOS. Loaded once, lazily, and only
    /// off the main actor (discovery already runs detached). An empty set is a
    /// safe degradation: every candidate then falls through to the shape rules
    /// above, which over-suggests rather than hiding real names.
    private nonisolated static let systemWords: Set<String> = {
        guard let contents = try? String(
            contentsOfFile: "/usr/share/dict/words", encoding: .utf8
        ) else { return [] }
        return Set(contents.split(separator: "\n").map { $0.lowercased() })
    }()

    private nonisolated static let commonWords: Set<String> = [
        "documents", "downloads", "desktop", "library", "pictures", "movies",
        "music", "public", "applications", "system", "users", "shared",
        "safari", "mail", "notes", "calendar", "reminders", "maps", "photos",
        "messages", "facetime", "preview", "terminal", "utilities", "books",
        "home", "news", "stocks", "shortcuts", "weather", "clock", "podcasts",
        "test", "tests", "temp", "tmp", "build", "dist", "node_modules",
        "backup", "archive", "old", "new", "data", "src", "bin", "lib",
    ]

    // MARK: - Persistence

    private static func defaultStoreURL() -> URL {
        ApplicationSupportLocator.applicationSupportRoot()
            .appendingPathComponent("Dictation", isDirectory: true)
            .appendingPathComponent("vocabulary.json")
    }

    private func load() {
        guard let data = try? Data(contentsOf: storeURL),
              let decoded = try? JSONDecoder().decode([Term].self, from: data) else { return }
        terms = decoded
    }

    private func save() {
        let url = storeURL
        let snapshot = terms
        persistenceQueue.async {
            try? FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            guard let data = try? encoder.encode(snapshot) else { return }
            try? data.write(to: url, options: .atomic)
        }
    }
}
