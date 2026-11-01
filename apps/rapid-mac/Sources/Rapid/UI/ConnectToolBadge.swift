import AppKit
import SwiftUI

/// Agent identity stays independent of installed applications and appearance.
enum ConnectToolIdentity {
    static func bundleIdentifiers(for id: String) -> [String] {
        switch id {
        case "claude-code": return ["com.anthropic.claudefordesktop"]
        case "codex": return ["com.openai.codex"]
        case "cursor": return ["com.todesktop.230313mzl4w4u92"]
        case "goose": return ["com.block.goose"]
        case "opencode": return ["ai.opencode.desktop"]
        default: return []
        }
    }

    static func monogram(for id: String, name: String) -> String {
        switch id {
        case "claude-code": return "CC"
        case "codex": return "Cx"
        case "pi": return "Pi"
        case "opencode": return "OC"
        case "hermes": return "H"
        case "goose": return "G"
        case "qwen-code": return "Q"
        case "crush": return "Cr"
        case "aider": return "Ai"
        case "cline": return "Cl"
        case "continue-dev": return "Co"
        case "cursor": return "Cu"
        case "deepseek-harness": return "DS"
        case "langchain": return "LC"
        case "openhands": return "OH"
        case "pydanticai": return "Pd"
        case "smolagents": return "Sm"
        case "kilo-code": return "K"
        default:
            let words = name.split(whereSeparator: { $0.isWhitespace })
                .map { $0.filter { $0.isLetter || $0.isNumber } }
                .filter { !$0.isEmpty }
            let initials = words.prefix(2).compactMap(\.first)
            return initials.isEmpty ? "?" : String(initials).uppercased()
        }
    }

    static func hue(for id: String) -> Double {
        switch id {
        case "claude-code": return 0.06
        case "codex": return 0.44
        case "pi": return 0.76
        case "opencode": return 0.56
        case "hermes": return 0.85
        case "goose": return 0.12
        case "qwen-code": return 0.68
        case "crush": return 0.94
        case "aider": return 0.34
        case "cline": return 0.60
        case "continue-dev": return 0.16
        case "cursor": return 0.48
        case "deepseek-harness": return 0.64
        case "langchain": return 0.30
        case "openhands": return 0.08
        case "pydanticai": return 0.90
        case "smolagents": return 0.20
        case "kilo-code": return 0.52
        default:
            // FNV-1a is stable across launches, unlike Swift's randomized Hasher.
            let hash = id.utf8.reduce(UInt64(14_695_981_039_346_656_037)) {
                ($0 ^ UInt64($1)) &* 1_099_511_628_211
            }
            return Double(hash % 1_000_000) / 1_000_000
        }
    }
}

@MainActor
enum ConnectToolIcon {
    case app(NSImage)
    case monogram(String)

    static func resolve(id: String, name: String, lookup: (String) -> NSImage?) -> Self {
        for bundleID in ConnectToolIdentity.bundleIdentifiers(for: id) {
            if let image = lookup(bundleID) { return .app(image) }
        }
        return .monogram(ConnectToolIdentity.monogram(for: id, name: name))
    }
}

/// Cache misses too: rendering another row or appearance never repeats a lookup.
@MainActor
private enum ConnectToolAppIconCache {
    private enum CachedIcon {
        case found(NSImage)
        case missing
    }

    private static var results: [String: CachedIcon] = [:]

    static func image(for bundleID: String) -> NSImage? {
        if let result = results[bundleID] {
            if case .found(let image) = result { return image }
            return nil
        }
        guard let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundleID) else {
            results[bundleID] = .missing
            return nil
        }
        let image = NSWorkspace.shared.icon(forFile: url.path)
        results[bundleID] = .found(image)
        return image
    }
}

struct ConnectToolBadge: View {
    let id: String
    let name: String
    @Environment(\.colorScheme) private var colorScheme

    private var tint: Color {
        Color(hue: ConnectToolIdentity.hue(for: id),
              saturation: colorScheme == .dark ? 0.45 : 0.6,
              brightness: colorScheme == .dark ? 0.95 : 0.6)
    }

    var body: some View {
        Group {
            switch ConnectToolIcon.resolve(id: id, name: name, lookup: ConnectToolAppIconCache.image) {
            case .app(let image):
                Image(nsImage: image)
                    .resizable()
                    .scaledToFit()
            case .monogram(let initials):
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .fill(tint.opacity(colorScheme == .dark ? 0.24 : 0.16))
                    .overlay {
                        Text(initials)
                            .font(.system(size: 11, weight: .semibold, design: .rounded))
                            .foregroundStyle(tint)
                    }
                    .overlay {
                        RoundedRectangle(cornerRadius: 7, style: .continuous)
                            .strokeBorder(RapidTheme.hairline, lineWidth: 0.5)
                    }
            }
        }
        .frame(width: 28, height: 28)
        .accessibilityHidden(true)
    }
}
