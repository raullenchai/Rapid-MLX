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

    /// The panels the Settings window renders, in both build configurations.
    private var settingsPanelSources: [String] {
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
        return panels
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

        for path in settingsPanelSources {
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

    /// ``DevSnapshot/settingsShell(category:size:)`` is the *second* hand-built
    /// environment chain that renders the real `SettingsView`, for the
    /// `RAPID_DEV_SNAPSHOT_DIR` capture run. It has to provide the same
    /// observables the scene does, and nothing was checking that.
    ///
    /// That gap shipped: adding the language picker to Settings → Appearance
    /// gave `SettingsView` a new `@Environment(LanguageConfig.self)`, the scene
    /// got it and the harness did not — so the capture run died with
    /// `EnvironmentValues.subscript.getter` in the backtrace and no type named,
    /// after writing 149 of its 221 PNGs. The failure is silent in CI because
    /// the harness only runs when `RAPID_DEV_SNAPSHOT_DIR` is set.
    @Test("The dev snapshot harness injects every observable its panels read")
    func snapshotHarnessProvidesEveryPanelDependency() throws {
        let snapshot = try source("Sources/Rapid/DevSnapshot.swift")

        // name → type, from the harness's own sources of observables:
        //   `runIfRequested(… appearance: AppearanceConfig, …)`
        //   `let snapshotMemory = MemoryStore(…)`
        var typeOfName: [String: String] = [:]
        if let open = snapshot.range(of: "runIfRequested(") {
            let body = snapshot[open.upperBound...]
            if let close = body.firstIndex(of: ")") {
                for pair in matches(#"(\w+):\s*([A-Z]\w*)"#, in: String(body[..<close]), group: 0) {
                    let name = matches(#"(\w+)\s*:"#, in: pair).first
                    let type = matches(#":\s*([A-Z]\w*)"#, in: pair).first
                    if let name, let type { typeOfName[name] = type }
                }
            }
        }
        for pair in matches(#"let\s+(\w+)\s*=\s*([A-Z]\w*)"#, in: snapshot, group: 0) {
            let name = matches(#"let\s+(\w+)"#, in: pair).first
            let type = matches(#"=\s*([A-Z]\w*)"#, in: pair).first
            if let name, let type { typeOfName[name] = type }
        }
        #expect(!typeOfName.isEmpty, "the DevSnapshot observable scrape found nothing")

        guard let anchor = snapshot.range(of: "func settingsShell(") else {
            Issue.record("DevSnapshot no longer declares settingsShell(category:size:)")
            return
        }
        guard let open = snapshot[anchor.upperBound...].firstIndex(of: "{") else { return }
        var depth = 0
        var index = open
        var shell = ""
        while index < snapshot.endIndex {
            if snapshot[index] == "{" { depth += 1 }
            if snapshot[index] == "}" {
                depth -= 1
                if depth == 0 { shell = String(snapshot[open...index]); break }
            }
            index = snapshot.index(after: index)
        }
        #expect(!shell.isEmpty, "could not slice the settingsShell body")

        var injected = Set(
            matches(#"\.environment\((\w+)\)"#, in: shell).compactMap { typeOfName[$0] }
        )
        // The one dotted form the harness uses: the chat view model owns the
        // instruction store, and Settings reads it directly.
        if shell.contains(".environment(chat.customInstructions)") {
            injected.insert("CustomInstructionsConfig")
        }

        for path in settingsPanelSources {
            let required = Set(matches(#"@Environment\((\w+)\.self\)"#, in: try source(path)))
            for type in required.sorted() {
                #expect(
                    injected.contains(type),
                    """
                    \(path) reads \(type) from the environment, and \
                    DevSnapshot.settingsShell never injects it. The \
                    RAPID_DEV_SNAPSHOT_DIR capture run traps — not warns — the \
                    first time it walks to that category, and CI never sees it \
                    because the harness does not run there.
                    """
                )
            }
        }
    }
}
