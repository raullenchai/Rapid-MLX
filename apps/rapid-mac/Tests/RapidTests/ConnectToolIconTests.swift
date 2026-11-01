import AppKit
import Testing
@testable import Rapid

@Suite("Connect tool icons")
struct ConnectToolIconTests {
    @Test("Only agents with their own desktop application have bundle candidates")
    func bundleIdentifiers() {
        let mappings = [
            "claude-code": "com.anthropic.claudefordesktop",
            "codex": "com.openai.codex",
            "cursor": "com.todesktop.230313mzl4w4u92",
            "goose": "com.block.goose",
            "opencode": "ai.opencode.desktop"
        ]
        for (id, bundleID) in mappings {
            #expect(ConnectToolIdentity.bundleIdentifiers(for: id) == [bundleID])
        }
        for id in ["pi", "cline", "continue-dev", "kilo-code", "unknown"] {
            #expect(ConnectToolIdentity.bundleIdentifiers(for: id).isEmpty)
        }
    }

    @Test("Monograms preserve agent identities and derive unknown names")
    func monograms() {
        let overrides = [
            "claude-code": "CC", "codex": "Cx", "pi": "Pi", "opencode": "OC",
            "hermes": "H", "goose": "G", "qwen-code": "Q", "crush": "Cr",
            "aider": "Ai", "cline": "Cl", "continue-dev": "Co", "cursor": "Cu",
            "deepseek-harness": "DS", "langchain": "LC", "openhands": "OH",
            "pydanticai": "Pd", "smolagents": "Sm", "kilo-code": "K"
        ]
        for (id, initials) in overrides {
            #expect(ConnectToolIdentity.monogram(for: id, name: "Ignored name") == initials)
            #expect((0..<1).contains(ConnectToolIdentity.hue(for: id)))
        }
        for (name, initials) in [
            ("Agent", "A"), ("Local Agent", "LA"), (" Local Agent Extra ", "LA"),
            ("!!local? @agent!", "LA"), ("... Agent", "A"), ("7 tools", "7T"),
            ("", "?"), (" !!! ", "?")
        ] {
            #expect(ConnectToolIdentity.monogram(for: "unknown", name: name) == initials)
        }
    }

    @Test("Unknown hues are stable and within the color wheel")
    func fallbackHue() {
        let hue = ConnectToolIdentity.hue(for: "unknown-agent")
        #expect(hue == ConnectToolIdentity.hue(for: "unknown-agent"))
        #expect((0..<1).contains(hue))
        // Pin the stable hash across processes, not only repeated calls.
        #expect(hue == 0.965621)
    }

    @MainActor
    @Test("Resolution uses an installed app icon or the agent monogram")
    func resolution() {
        let image = NSImage(size: NSSize(width: 28, height: 28))
        var candidates: [String] = []
        let resolved = ConnectToolIcon.resolve(id: "codex", name: "Codex") { bundleID in
            candidates.append(bundleID)
            return image
        }
        #expect(candidates == ["com.openai.codex"])
        if case .app(let actual) = resolved {
            #expect(actual === image)
        } else {
            Issue.record("Expected installed app icon")
        }
        let missing = ConnectToolIcon.resolve(id: "codex", name: "Codex") { _ in nil }
        if case .monogram(let initials) = missing {
            #expect(initials == "Cx")
        } else {
            Issue.record("Expected monogram for missing app")
        }
        let unmapped = ConnectToolIcon.resolve(id: "pi", name: "Pi") { _ in
            Issue.record("CLI agents must not look up an editor application")
            return image
        }
        if case .monogram(let initials) = unmapped {
            #expect(initials == "Pi")
        } else {
            Issue.record("Expected monogram for CLI agent")
        }
    }
}
