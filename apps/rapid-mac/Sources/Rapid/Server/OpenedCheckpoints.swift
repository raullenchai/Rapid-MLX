import Foundation
import Darwin

/// References to user-owned checkpoints. This preference never grants deletion
/// authority and is deliberately separate from the managed download folder.
enum OpenedCheckpoints {
    static let storageKey = "rapid.models.openedCheckpoints"
    static let limit = 32

    static func canonicalPath(_ path: String) -> String? {
        guard path.hasPrefix("/"), path.utf8.count <= 4096,
              !path.unicodeScalars.contains(where: { $0.value < 32 || $0.value == 127 }) else { return nil }
        // Match the engine's os.path.realpath, including macOS /var aliases;
        // Foundation's URL normalization alone does not canonicalize all of them.
        if let resolved = realpath(path, nil) {
            defer { free(resolved) }
            return String(cString: resolved)
        }
        return URL(fileURLWithPath: path).standardizedFileURL.path
    }

    static func paths(defaults: UserDefaults = .standard) -> [String] {
        var seen: Set<String> = []
        return (defaults.stringArray(forKey: storageKey) ?? []).compactMap(canonicalPath)
            .filter { seen.insert($0).inserted }.prefix(limit).map { $0 }
    }

    @MainActor static func register(_ check: LocalCheckpointCheck, defaults: UserDefaults = .standard) throws {
        guard check.certified, let path = check.path else { throw LocalCheckpointError.rejected(check.reason ?? "inspection_failed") }
        var references = paths(defaults: defaults)
        guard !references.contains(path) else { return }
        guard references.count < limit else { throw LocalCheckpointError.limit }
        references.append(path)
        defaults.set(references, forKey: storageKey)
    }

    @MainActor static func remove(_ path: String, defaults: UserDefaults = .standard) {
        // Removing a reference is not ModelCacheActions deletion, even when the
        // directory is missing or happens to share a managed model's basename.
        defaults.set(paths(defaults: defaults).filter { $0 != path }, forKey: storageKey)
    }

    static func merging(_ entries: [ModelEntry], checks: [LocalCheckpointCheck]) -> [ModelEntry] {
        var result = entries
        var seen: Set<String> = []
        for check in checks where check.certified {
            guard let path = check.path, let bytes = check.sizeBytes, seen.insert(path).inserted,
                  !result.contains(where: { $0.alias == path }) else { continue }
            if let known = check.knownModel, let index = result.firstIndex(where: { $0.alias == known }) {
                if result[index].cached { continue } // Preserve actual inventory ownership.
                result.remove(at: index) // Never offer a download for a certified local copy.
            }
            result.append(ModelEntry(
                alias: path, hfRepo: nil,
                sizeOnDisk: ModelSize.bytes(bytes).displayText,
                cached: true, sizeBytes: check.sizeBytes, isExternal: true,
                cachedChatDiscovery: CachedChatDiscovery(ready: true, fit: check.fit,
                    mlxNative: check.mlxNative ?? false, modifiedEpoch: check.modifiedEpoch ?? 0),
                isUncataloged: true
            ))
        }
        return result
    }
}

struct LocalCheckpointEnvelope: Decodable {
    let schema: Int
    let localCheckpoints: [LocalCheckpointCheck]
}

struct LocalCheckpointCheck: Decodable, Sendable {
    let requestedPath: String
    let path: String?
    let ready: Bool
    let reason: String?
    let launchPath: String?
    let knownModel: String?
    let sizeBytes: Int64?
    let fit: String?
    let mlxNative: Bool?
    let modifiedEpoch: Double?

    var certified: Bool {
        ready && path != nil && path == OpenedCheckpoints.canonicalPath(requestedPath)
            && path == launchPath && (sizeBytes ?? 0) > 0
            && ["yes", "tight"].contains(fit ?? "")
            && mlxNative != nil && (modifiedEpoch?.isFinite == true)
    }
}

enum LocalCheckpointError: LocalizedError {
    case rejected(String), limit, unavailable
    var errorDescription: String? {
        switch self {
        case .limit:
            String(localized: "Remove an existing checkpoint reference before adding another.")
        case .unavailable:
            String(localized: "The engine could not inspect this directory. Update the engine and try again.")
        case .rejected(let reason):
            switch reason {
            case "missing_directory", "invalid_path":
                String(localized: "Choose an existing checkpoint directory.")
            case "unsafe_symlink":
                String(localized: "This checkpoint contains links outside the selected directory.")
            case "incomplete_checkpoint":
                String(localized: "This directory is missing complete MLX checkpoint weights.")
            case "missing_chat_template", "missing_tokenizer":
                String(localized: "This checkpoint needs a tokenizer vocabulary and chat template.")
            case "unsupported_chat_model", "unsupported_quantization", "unsupported_or_oversize":
                String(localized: "This checkpoint is not a supported chat model for this engine and Mac.")
            default:
                String(localized: "The checkpoint metadata or runtime could not be verified. Check the files and engine installation.")
            }
        }
    }
}
