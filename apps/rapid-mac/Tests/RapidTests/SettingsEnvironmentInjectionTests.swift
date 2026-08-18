import Foundation
import Testing
@testable import Rapid

/// Every observable a Settings panel reads must be injected by the Settings
/// scene.
///
/// SwiftUI does not warn about a missing `@Environment` observable — it traps.
/// The failure is invisible until somebody opens that one category, and then
/// the app dies with `EnvironmentValues.subscript.getter` in the backtrace and
/// nothing naming the type.
///
/// This is not hypothetical: Settings → Developer shipped its first build
/// reading `QuickstartCoordinator`, which the main window injected and the
/// Settings window did not. It compiled, every unit test passed, and clicking
/// the row killed the app.
///
/// ``SettingsVisualFoundationTests/everyCategoryKeepsItsStateOwner`` pins the
/// other half — that a panel still *declares* what it needs. Declaring and
/// providing are different mistakes; this covers the provider side.
@Suite("Settings environment injection")
struct SettingsEnvironmentInjectionTests {

    private static var sourceRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func source(_ relativePath: String) throws -> String {
        try String(
            contentsOf: Self.sourceRoot.appendingPathComponent(relativePath),
            encoding: .utf8
        )
    }

    private func matches(_ pattern: String, in text: String, group: Int = 1) -> [String] {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        let range = NSRange(text.startIndex..., in: text)
        return regex.matches(in: text, range: range).compactMap {
            Range($0.range(at: group), in: text).map { r in String(text[r]) }
        }
    }

    /// The `Window("Settings", …)` scene body, sliced out by brace balance so
    /// the main window's own (larger) injection list can't stand in for it.
    private func settingsSceneBody(_ app: String) throws -> String {
        guard let anchor = app.range(of: #"Window("Settings""#) else {
            Issue.record("RapidApp no longer declares Window(\"Settings\")")
            return ""
        }
        guard let open = app[anchor.upperBound...].firstIndex(of: "{") else { return "" }
        var depth = 0
        var index = open
        while index < app.endIndex {
            if app[index] == "{" { depth += 1 }
            if app[index] == "}" {
                depth -= 1
                if depth == 0 { return String(app[open...index]) }
            }
            index = app.index(after: index)
        }
        return ""
    }

    @Test("The Settings scene injects every observable its panels read")
    func settingsSceneProvidesEveryPanelDependency() throws {
        let app = try source("Sources/Rapid/RapidApp.swift")

        // name → type, from `@State private var server: ServerManager`.
        var typeOfProperty: [String: String] = [:]
        let declarations = matches(
            #"@State\s+private\s+var\s+(\w+)\s*:\s*(\w+)"#, in: app, group: 0
        )
        for declaration in declarations {
            let name = matches(#"var\s+(\w+)\s*:"#, in: declaration).first
            let type = matches(#":\s*(\w+)"#, in: declaration).first
            if let name, let type { typeOfProperty[name] = type }
        }
        #expect(!typeOfProperty.isEmpty, "the @State declaration scrape found nothing")

        let scene = try settingsSceneBody(app)
        #expect(!scene.isEmpty, "could not slice the Settings scene body")
        let injected = Set(
            matches(#"\.environment\((\w+)\)"#, in: scene).compactMap { typeOfProperty[$0] }
        )

        // Every Settings panel, including the debug-only one when present.
        var panels = [
            "Sources/Rapid/UI/SettingsView.swift",
            "Sources/Rapid/UI/SettingsToolsPanel.swift",
            "Sources/Rapid/UI/SettingsConnectorsPanel.swift",
            "Sources/Rapid/UI/SettingsModelManagementPanel.swift",
            "Sources/Rapid/UI/SettingsPerformancePanel.swift",
        ]
        #if DEBUG
        panels.append("Sources/Rapid/UI/SettingsDeveloperPanel.swift")
        #endif

        for path in panels {
            let required = Set(matches(#"@Environment\((\w+)\.self\)"#, in: try source(path)))
            for type in required.sorted() {
                #expect(
                    injected.contains(type),
                    """
                    \(path) reads \(type) from the environment, and the \
                    Settings scene in RapidApp.swift never injects it. SwiftUI \
                    traps — not warns — the first time that category is opened.
                    """
                )
            }
        }
    }
}
