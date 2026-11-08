import Foundation

/// Browser-reference rules mirror rapid_mlx/model_reference.py. No disk or network IO.
enum HuggingFaceReference: Equatable, Sendable {
    case model(repo: String, isURL: Bool)
    case invalidURL

    static func detect(_ input: String) -> Self? {
        var value = input.trimmingCharacters(in: .whitespacesAndNewlines)
        let wrappers: [Character: Character] = ["\"": "\"", "'": "'", "<": ">"]
        while value.count >= 2, let first = value.first, wrappers[first] == value.last {
            value = String(value.dropFirst().dropLast()).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        let hostPattern = #"(?i)^(?:https?://)?(?:huggingface\.co|www\.huggingface\.co|hf\.co|hf-mirror\.com)(?:[/?#]|$)"#
        let isURL = value.range(of: hostPattern, options: .regularExpression) != nil
        let parts: [String]
        if isURL {
            guard let url = URLComponents(string: value.contains("://") ? value : "https://" + value),
                  ["huggingface.co", "www.huggingface.co", "hf.co", "hf-mirror.com"].contains(url.host?.lowercased() ?? ""),
                  url.port == nil, url.user == nil, url.password == nil else { return .invalidURL }
            // Keep percent escapes: the CLI validates the raw path components.
            parts = url.percentEncodedPath.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
                .components(separatedBy: "/")
            if ["datasets", "spaces", "collections", "models"].contains(parts[0].lowercased()) {
                return .invalidURL
            }
        } else {
            parts = value.components(separatedBy: "/")
            guard parts.count == 2 else { return nil }
            // Reserved HF routes must not become repository IDs when pasted bare.
            if ["datasets", "spaces", "collections", "models"].contains(parts[0].lowercased()) { return nil }
        }
        guard parts.count >= 2, parts.prefix(2).allSatisfy({
            $0.range(of: #"^[A-Za-z0-9_][A-Za-z0-9_.-]*$"#, options: .regularExpression) != nil
        }) else { return isURL ? .invalidURL : nil }
        return .model(repo: parts.prefix(2).joined(separator: "/"), isURL: isURL)
    }

    static func clipboardPrefill(_ text: String?) -> String? {
        guard let text, case .model(_, true) = detect(text) else { return nil }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func catalogMatch(_ input: String, in entries: [ModelEntry]) -> ModelEntry? {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        let repo: String? = if case .model(let repo, _) = detect(input) { repo } else { nil }
        return entries.first {
            ($0.alias == trimmed || (repo != nil && $0.hfRepo == repo))
        }
    }
}

struct HuggingFaceModelCheck: Decodable, Equatable, Sendable {
    enum Verdict: String, Decodable, CaseIterable, Sendable {
        case ok, unsupported, tooBig = "too_big", unknown, invalidReference = "invalid_reference"
    }
    let schema: Int
    let reference: String
    let repo: String?
    let alias: String?
    let inCatalog: Bool
    let verdict: Verdict
    let reason: String?
    let message: String?
    let downloadBytes: Int64?
    let format: String?
    let capabilities: [String]
    let requiredMemoryBytes: Int64?
    let fits: Bool?
    let cached: Bool

    enum CodingKeys: String, CodingKey {
        case schema, reference, repo, alias, verdict, reason, message, format, capabilities, fits, cached
        case inCatalog = "in_catalog", downloadBytes = "download_bytes", requiredMemoryBytes = "required_memory_bytes"
    }

    enum RowState: Equatable { case catalog, download, downloadAnyway, warning }
    var rowState: RowState {
        if inCatalog { return .catalog }
        guard repo != nil else { return .warning }
        switch verdict {
        case .ok: return .download
        case .tooBig: return .downloadAnyway
        case .unsupported, .unknown, .invalidReference: return .warning
        }
    }

    var entry: ModelEntry? {
        guard let repo else { return nil }
        return ModelEntry(alias: repo, hfRepo: repo, sizeOnDisk: nil, cached: cached,
                          downloadSizeBytes: downloadBytes, isUncataloged: true)
    }

    var communityMeta: String {
        ([repo?.components(separatedBy: "/").first, format].compactMap { $0 }
            + capabilities + [String(localized: "community upload, not tested")]).joined(separator: " · ")
    }

    func memoryWarning(physicalRAMGB: Double) -> String {
        guard let requiredMemoryBytes else {
            return message ?? String(localized: "This model may need more memory than this Mac has.")
        }
        let required = Int(ceil(Double(requiredMemoryBytes) / Double(1 << 30)))
        let available = Int(physicalRAMGB.rounded())
        return String(localized: "Needs about \(required) GB memory; this Mac has \(available) GB.")
    }
}

extension ModelEntry {
    var isFromHuggingFace: Bool {
        guard let hfRepo, case .model(_, false) = HuggingFaceReference.detect(hfRepo),
              !isExternal, isBuiltinProfile != true else { return false }
        return isBuiltinProfile == false || isUncataloged || alias == hfRepo
    }
}
