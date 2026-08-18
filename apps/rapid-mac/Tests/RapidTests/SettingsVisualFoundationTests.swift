import AppKit
import Foundation
import SwiftUI
import Testing
@testable import Rapid

/// Pins for the UI-1 Settings visual migration.
///
/// The migration's claim is narrow and checkable: every Settings panel
/// draws from one set of tokens and one set of components, no category
/// lost a control or a state owner, and the window's committed widths
/// fit the smallest window it supports. These tests hold that claim;
/// the screenshots hold everything about it that is a matter of taste.
///
/// Source-level guards use the same canonical (comment- and
/// literal-stripped) form the existing ``SourceGuardSupport`` helpers
/// produce, so a value inside a comment or a doc-string cannot pass or
/// fail one by accident.
@Suite("Settings visual foundation (UI-1)")
@MainActor
// The whole suite runs on the main actor: several tests construct SwiftUI
// style/metric types whose initializers the CI toolchain (stricter default
// isolation than the local one) treats as MainActor-isolated. Individual
// @MainActor markers below become redundant but stay harmless.
struct SettingsVisualFoundationTests {

    private static var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // RapidTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // package root
    }

    /// Every Settings source this phase migrated.
    private static let settingsSources = [
        "Sources/Rapid/UI/SettingsView.swift",
        "Sources/Rapid/UI/SettingsToolsPanel.swift",
        "Sources/Rapid/UI/SettingsConnectorsPanel.swift",
        "Sources/Rapid/UI/SettingsModelManagementPanel.swift",
        "Sources/Rapid/UI/SettingsPerformancePanel.swift",
        "Sources/Rapid/UI/MCPServerEditorSheet.swift",
    ]

    /// Canonical form of each source, computed once per process.
    ///
    /// Six of the tests below read the same five files. Stripping is not
    /// free (it walks every character to elide comments and literals), and
    /// this suite runs alongside wall-clock-sensitive streaming tests —
    /// paying it thirty times instead of five measurably slowed the whole
    /// run and pushed two unrelated timing tests over their thresholds.
    /// Immutable, computed once on first touch. A `let` dictionary keyed
    /// by path is `Sendable`; a mutable cache would need a lock, which is
    /// more machinery than five files justify.
    private static let strippedSources: [String: String] = {
        var out: [String: String] = [:]
        for path in settingsSources {
            let url = packageRoot.appendingPathComponent(path)
            guard let body = try? String(contentsOf: url, encoding: .utf8) else { continue }
            out[path] = CapabilityChipRenderGateSourceGuardTests
                .stripCommentsAndWhitespace(body)
        }
        return out
    }()

    private func strippedSource(_ relativePath: String) throws -> String {
        if let hit = Self.strippedSources[relativePath] { return hit }
        // Not one of the five migrated panels — read it on demand.
        let url = Self.packageRoot.appendingPathComponent(relativePath)
        let body = try String(contentsOf: url, encoding: .utf8)
        return CapabilityChipRenderGateSourceGuardTests
            .stripCommentsAndWhitespace(body)
    }

    // MARK: - Navigation and ownership

    @MainActor
    @Test("Every category still exists, with a title and an icon")
    func everyCategorySurvives() {
        // ``performance`` joined in #1717 (per-model engine knobs), as
        // ``connectors`` did in #1716. The guard's job is to make a category
        // change deliberate and reviewed, not to freeze the list forever —
        // update this literal alongside the enum when a feature adds one.
        var expected: Set<String> = [
            "modelManagement", "instructions", "tools", "connectors", "performance",
            "appearance", "privacy", "app",
        ]
        // `swift test` builds debug, so the debug-only category is present
        // here and absent from the shipped binary. Conditioning the literal
        // rather than hardcoding it keeps `swift test -c release` honest too.
        #if DEBUG
        expected.insert("developer")
        #endif
        let actual = Set(SettingsView.Category.allCases.map(\.rawValue))
        #expect(
            actual == expected,
            """
            UI-1 is a visual migration and must not add, remove, or rename a \
            Settings category as a side effect. Got \(actual.sorted()).
            """
        )
        for category in SettingsView.Category.allCases {
            #expect(!category.title.isEmpty, "\(category.rawValue) lost its title")
            #expect(!category.iconName.isEmpty, "\(category.rawValue) lost its icon")
        }
    }

    @MainActor
    @Test("Category order is unchanged, so arrow-key navigation is unchanged")
    func categoryOrderIsStable() {
        var expectedOrder = [
            "modelManagement", "instructions", "tools", "connectors", "performance",
            "appearance", "privacy", "app",
        ]
        #if DEBUG
        expectedOrder.append("developer")
        #endif
        #expect(SettingsView.Category.allCases.map(\.rawValue) == expectedOrder)
        // #1717: ``performance`` sits after ``connectors`` — both are
        // "what the engine is wired to do", ahead of the presentation and
        // app-level sections.
        #expect(SettingsView.category(.connectors, movedBy: 1) == .performance)
        #expect(SettingsView.category(.performance, movedBy: 1) == .appearance)
        // The rail's ↑/↓ handler walks this order and clamps at the ends.
        #expect(SettingsView.category(.modelManagement, movedBy: -1) == nil)
        // Developer sits after App in debug, so App is only the last row in
        // a release build. Both spellings assert the same property: arrowing
        // past the final row clamps rather than wrapping.
        #if DEBUG
        #expect(SettingsView.category(.app, movedBy: 1) == .developer)
        #expect(SettingsView.category(.developer, movedBy: 1) == nil)
        #expect(SettingsView.category(.developer, movedBy: -1) == .app)
        #else
        #expect(SettingsView.category(.app, movedBy: 1) == nil)
        #endif
        #expect(SettingsView.category(.modelManagement, movedBy: 1) == .instructions)
        #expect(SettingsView.category(.app, movedBy: -1) == .privacy)
    }

    /// Each panel reaches its state through the SwiftUI environment. A
    /// migration that dropped an `@Environment` line would compile and
    /// then crash at runtime the first time that category was opened —
    /// SwiftUI traps on a missing observable — so the ownership is
    /// pinned at the source level where it can fail in CI instead.
    @Test("No category loses its state owner")
    func everyCategoryKeepsItsStateOwner() throws {
        let requirements: [(path: String, owners: [String])] = [
            ("Sources/Rapid/UI/SettingsView.swift", [
                "@Environment(AppearanceConfig.self)",
                "@Environment(CustomInstructionsConfig.self)",
                "@Environment(SettingsRouter.self)",
                "@Environment(ServerManager.self)",
                "@Environment(UpdateChecker.self)",
                "@Environment(SparkleUpdateController.self)",
                "@Environment(DockVisibilityPromptStore.self)",
            ]),
            ("Sources/Rapid/UI/SettingsToolsPanel.swift", [
                "@Environment(ChatViewModel.self)",
                "@Environment(WebSearchConfig.self)",
                "@Environment(BrowseApprovalStore.self)",
            ]),
            ("Sources/Rapid/UI/SettingsConnectorsPanel.swift", [
                "@Environment(MCPConfigStore.self)",
                "@Environment(MCPCatalog.self)",
                "@Environment(MCPToolApprovalStore.self)",
                "@Environment(MCPToolRegistry.self)",
                "@Environment(ServerManager.self)",
            ]),
            ("Sources/Rapid/UI/SettingsModelManagementPanel.swift", [
                "@Environment(ServerManager.self)",
                "@Environment(DownloadManager.self)",
            ]),
        ]
        for (path, owners) in requirements {
            let source = try strippedSource(path)
            for owner in owners {
                #expect(
                    source.contains(owner),
                    "\(path) no longer declares \(owner) — that category's panel would trap on open."
                )
            }
        }
    }

    @Test("Persisted settings keys are untouched")
    func persistedKeysAreUnchanged() throws {
        // The migration repainted controls; it must not have moved where
        // any of them reads or writes.
        let panel = try strippedSource("Sources/Rapid/UI/SettingsModelManagementPanel.swift")
        #expect(panel.contains("@AppStorage(ModelPickerVisibility.showAllStorageKey)"))
        #expect(panel.contains("@AppStorage(AutoStartPreference.storageKey)"))
    }

    // MARK: - Shared components are actually used

    @Test("Every Settings panel is built from the shared section components")
    func panelsUseSharedComponents() throws {
        for path in Self.settingsSources where !path.hasSuffix("MCPServerEditorSheet.swift") {
            let source = try strippedSource(path)
            #expect(
                source.contains("SettingsSection(")
                    || source.contains("settingsGroupedCard(")
                    || source.contains("InstructionEditorSection("),
                "\(path) draws no shared Settings section — it is still hand-rolling a card."
            )
        }
        // The editor sheet keeps its native grouped Form, but must still
        // use the shared header + button tiers.
        let sheet = try strippedSource("Sources/Rapid/UI/MCPServerEditorSheet.swift")
        #expect(sheet.contains("SectionHeader("))
        #expect(sheet.contains("buttonStyle(.rapidPrimary)"))
        #expect(sheet.contains("buttonStyle(.rapidSecondary)"))
    }

    @Test("Settings uses the shared button tiers, not the native ones")
    func panelsUseSharedButtonTiers() throws {
        for path in Self.settingsSources {
            let source = try strippedSource(path)
            for native in ["buttonStyle(.borderedProminent)", "buttonStyle(.bordered)"] {
                #expect(
                    !source.contains(native),
                    """
                    \(path) still uses \(native). Native prominent buttons \
                    inherit the scene tint, which paints amber with white text \
                    at ~2:1 — the contrast failure RapidPrimaryButtonStyle \
                    exists to prevent.
                    """
                )
            }
        }
    }

    @Test("Destructive Settings actions use the destructive hierarchy")
    func destructiveActionsAreStyled() throws {
        let panel = try strippedSource("Sources/Rapid/UI/SettingsModelManagementPanel.swift")
        #expect(
            panel.contains("buttonStyle(.rapidDestructiveCompact)"),
            "The recommended card's Delete must read as destructive."
        )
        #expect(
            panel.contains("tint:RapidTheme.statusError"),
            "The table's delete glyphs must carry the error tint."
        )
    }

    // MARK: - One set of tokens

    @Test("No Settings panel paints a raw colour or a one-off radius")
    func panelsUseOnlyTokens() throws {
        // Substrings checked against the stripped source, where comments
        // and string literals are gone — so a token name mentioned in a
        // doc-comment cannot trip this.
        let banned = [
            "Color.red", "Color.orange", "Color.green", "Color.blue", "Color.purple",
            "foregroundStyle(.orange)", "foregroundStyle(.red)", "foregroundStyle(.green)",
            "RapidTheme.cardRadius",
            "RapidTheme.brandTint",
        ]
        for path in Self.settingsSources {
            let source = try strippedSource(path)
            for needle in banned {
                #expect(
                    !source.contains(needle),
                    """
                    \(path) still contains `\(needle)`. Settings draws from the \
                    semantic tokens now: statusError / statusWarning / \
                    statusReady / statusIdle for meaning, brandPrimaryTint for \
                    selection, Radius.card for containers.
                    """
                )
            }
        }
    }

    @Test("No Settings panel invents a font size")
    func panelsUseTheTypeRamp() throws {
        let banned = [
            "font(.title)", "font(.title2)", "font(.title3)",
            "font(.headline)", "font(.subheadline)",
            "font(.callout)", "font(.caption)", "font(.caption2)",
            "font(.body)",
        ]
        for path in Self.settingsSources {
            let source = try strippedSource(path)
            for needle in banned {
                #expect(
                    !source.contains(needle),
                    """
                    \(path) still uses `\(needle)`. The Settings window had four \
                    heading sizes because panels reached past the ramp; every \
                    role now exists on RapidFont.
                    """
                )
            }
        }
    }

    // MARK: - UI-1 refinement invariants

    /// Nothing in Settings may put white on an amber fill.
    ///
    /// The review found it on segmented controls; the general rule is
    /// that ink on ``brandPrimary`` comes from ``brandOnAccent`` and
    /// from nowhere else.
    @Test("No Settings surface writes white on an accent fill")
    func noWhiteOnAccent() throws {
        for path in Self.settingsSources + [
            "Sources/Rapid/UI/SettingsControlStyles.swift",
            "Sources/Rapid/UI/Components/RapidButtonStyles.swift",
        ] {
            let source = try strippedSource(path)
            for needle in ["foregroundStyle(.white)", "foregroundStyle(Color.white)"] {
                #expect(
                    !source.contains(needle),
                    "\(path) puts white text on a control — use RapidTheme.brandOnAccent."
                )
            }
        }
    }

    /// The destructive fill is much lighter in Dark, so its ink has to
    /// change with it — white on the Dark coral is ~2.4:1.
    @MainActor
    @Test("Destructive ink adapts to its fill")
    func destructiveInkAdapts() throws {
        let light = try #require(Self.resolve(RapidTheme.destructiveActionLabel, appearance: .aqua))
        let dark = try #require(Self.resolve(RapidTheme.destructiveActionLabel, appearance: .darkAqua))
        #expect(light != dark, "Destructive ink must not be one colour for two very different reds.")
        #expect(light.redComponent > 0.9, "Light mode: white on the deep brick fill.")
        #expect(dark.redComponent < 0.3, "Dark mode: near-black on the light coral fill.")
    }

    @Test("brandOnAccent is dark ink and identical in Light and Dark")
    @MainActor
    func accentInkIsDarkInBothAppearances() throws {
        let light = try #require(Self.resolve(RapidTheme.brandOnAccent, appearance: .aqua))
        let dark = try #require(Self.resolve(RapidTheme.brandOnAccent, appearance: .darkAqua))
        // The amber fill does not change between appearances, so its ink
        // must not either — flipping to white in Dark is the exact 2:1
        // pairing this token exists to prevent.
        #expect(light == dark)
        // Dark ink: comfortably below mid-grey on every channel.
        #expect(light.redComponent < 0.35)
        #expect(light.greenComponent < 0.35)
        #expect(light.blueComponent < 0.35)
    }

    /// The native segmented style is what produced white-on-amber, so it
    /// must not come back in Settings.
    @Test("Settings uses the shared segmented control, not the native style")
    func segmentedControlIsShared() throws {
        for path in Self.settingsSources {
            let source = try strippedSource(path)
            #expect(
                !source.contains("pickerStyle(.segmented)"),
                """
                \(path) uses .pickerStyle(.segmented). Its selected segment                 takes the ambient tint with WHITE text and exposes no hook to                 change it — use RapidSegmentedControl.
                """
            )
        }
        let panel = try strippedSource("Sources/Rapid/UI/SettingsModelManagementPanel.swift")
        #expect(panel.contains("RapidSegmentedControl("))
    }

    @Test("The regular button tiers share one height")
    func regularButtonTiersShareAHeight() {
        // A Cancel/Save pair in a sheet must not step. Emphasis is the
        // fill, not four extra points of height.
        #expect(RapidPrimaryButtonStyle().height == RapidTheme.ControlHeight.medium)
        #expect(RapidSecondaryButtonStyle().height == RapidTheme.ControlHeight.medium)
        #expect(RapidDestructiveButtonStyle().height == RapidTheme.ControlHeight.medium)
        // ...and the compact tiers agree too.
        #expect(RapidPrimaryButtonStyle(height: RapidTheme.ControlHeight.small).height
                == RapidSecondaryButtonStyle(height: RapidTheme.ControlHeight.small).height)
    }

    @Test("The settings toggle reserves a real gutter and a stable column")
    func toggleGutterIsGenerous() {
        // The review asked for 20–24pt of clear space so a three-line
        // description can never run under the switch.
        #expect(TrailingSettingsToggleStyle.gutter >= 20)
        #expect(TrailingSettingsToggleStyle.controlColumnWidth > 0)
    }

    @Test("Card insets meet the reviewed minimums")
    func cardInsetsAreGenerous() {
        #expect(SettingsCardMetrics.regularInset >= 24)
        #expect(SettingsCardMetrics.compactInset >= 20)
        #expect(SettingsCardMetrics.inset(isCompact: false) == SettingsCardMetrics.regularInset)
        #expect(SettingsCardMetrics.inset(isCompact: true) == SettingsCardMetrics.compactInset)
    }

    /// Tool rows lead with a human name; the wire identifier stays the
    /// key for state, dispatch and accessibility.
    @Test("Built-in tools present human names without losing their identifiers")
    func toolDisplayNames() {
        #expect(SettingsToolsPanel.displayName(for: "web_search") == "Web Search")
        #expect(SettingsToolsPanel.displayName(for: "browse") == "Browse Web Page")
        #expect(SettingsToolsPanel.displayName(for: "weather") == "Weather")
        // An unmapped tool degrades to its identifier rather than to
        // nothing, so a newly registered tool is never nameless.
        #expect(SettingsToolsPanel.displayName(for: "future_tool") == "future_tool")

        // Summaries are short enough to sit inside the three-line cap.
        for tool in ["web_search", "browse", "weather"] {
            let summary = SettingsToolsPanel.summary(for: tool, fallback: "")
            #expect(!summary.isEmpty)
            #expect(summary.count < 110, "\(tool) summary is long enough to wrap past three lines")
        }
        // An unmapped tool falls back to the engine's own description.
        #expect(SettingsToolsPanel.summary(for: "future_tool", fallback: "raw") == "raw")
    }

    @Test("Tool rows still key their state on the wire identifier")
    func toolRowsKeyOnWireIdentifier() throws {
        let source = try strippedSource("Sources/Rapid/UI/SettingsToolsPanel.swift")
        // The toggle binding and the AX identifier must both use the raw
        // name, not the display name.
        #expect(source.contains("toolBinding(name)"))
        #expect(source.contains(#"accessibilityIdentifier("Settings.Tools.Toggle.\(def.function.name)")"#))
    }

    // MARK: - Removed tokens have no consumers

    @Test("Tokens removed by this migration have no remaining consumers")
    func removedTokensAreUnreferenced() throws {
        let removed = [
            // Superseded by ``surfaceSidebar``; the cool grey rail.
            "RapidTheme.sidebarSurface",
            // The tray glyph is unconditionally a template now.
            "MenuBarStatus.glyphIsTemplate",
            "glyphIsTemplate(",
        ]
        let sourcesRoot = Self.packageRoot.appendingPathComponent("Sources")
        let files = FileManager.default
            .enumerator(at: sourcesRoot, includingPropertiesForKeys: nil)?
            .compactMap { $0 as? URL }
            .filter { $0.pathExtension == "swift" } ?? []
        #expect(!files.isEmpty, "Found no Swift sources to scan — the walk is broken.")

        for file in files {
            let body = try String(contentsOf: file, encoding: .utf8)
            // Raw-substring pre-filter first: canonicalising ~160 files to
            // prove a negative is wasted work, and this suite shares a
            // process with tests that measure wall-clock latency. Only a
            // file that mentions the symbol at all is worth stripping —
            // and stripping is still what decides, so a mention inside a
            // comment (which is how the removals are documented) passes.
            guard removed.contains(where: { body.contains($0) }) else { continue }
            let stripped = CapabilityChipRenderGateSourceGuardTests
                .stripCommentsAndWhitespace(body)
            for token in removed {
                #expect(
                    !stripped.contains(token),
                    "\(file.lastPathComponent) still references removed symbol `\(token)`."
                )
            }
        }
    }

    // MARK: - Light and Dark both resolve

    /// Every token this phase added is dynamic. A token that resolves to
    /// the same colour in both appearances is either a mistake or a
    /// deliberate constant — and the ones below are deliberately NOT
    /// constant, so equality means somebody dropped the dark branch.
    @MainActor
    @Test("New semantic tokens resolve differently in Light and Dark")
    func tokensResolveInBothAppearances() {
        let cases: [(name: String, color: Color)] = [
            ("statusReadyTint", RapidTheme.statusReadyTint),
            ("surfaceCanvas", RapidTheme.surfaceCanvas),
            ("surfaceSidebar", RapidTheme.surfaceSidebar),
            ("surfaceRaised", RapidTheme.surfaceRaised),
            ("surfaceCode", RapidTheme.surfaceCode),
            ("hairline", RapidTheme.hairline),
            ("hairlineStrong", RapidTheme.hairlineStrong),
            ("statusError", RapidTheme.statusError),
            ("statusErrorTint", RapidTheme.statusErrorTint),
        ]
        for (name, color) in cases {
            let light = Self.resolve(color, appearance: .aqua)
            let dark = Self.resolve(color, appearance: .darkAqua)
            #expect(light != nil, "\(name) did not resolve in Light")
            #expect(dark != nil, "\(name) did not resolve in Dark")
            #expect(
                light != dark,
                "\(name) resolves identically in Light and Dark — its dynamic provider lost a branch."
            )
        }
    }

    /// Text tokens delegate to the system's dynamic label colours, so
    /// they only have to RESOLVE — asserting they differ would be
    /// asserting something about macOS, not about this code.
    @MainActor
    @Test("Text tokens resolve in both appearances")
    func textTokensResolve() {
        for (name, color) in [
            ("textPrimary", RapidTheme.textPrimary),
            ("textSecondary", RapidTheme.textSecondary),
            ("textTertiary", RapidTheme.textTertiary),
            ("textDisabled", RapidTheme.textDisabled),
        ] {
            #expect(Self.resolve(color, appearance: .aqua) != nil, "\(name) failed in Light")
            #expect(Self.resolve(color, appearance: .darkAqua) != nil, "\(name) failed in Dark")
        }
    }

    @MainActor
    private static func resolve(
        _ color: Color,
        appearance name: NSAppearance.Name
    ) -> NSColor? {
        guard let appearance = NSAppearance(named: name) else { return nil }
        var resolved: NSColor?
        appearance.performAsCurrentDrawingAppearance {
            resolved = NSColor(color).usingColorSpace(.deviceRGB)
        }
        return resolved
    }

    // MARK: - Responsive contract

    /// The arithmetic behind "no horizontal clipping at 720×480".
    ///
    /// This is the assertion the audit could not make from source: the
    /// old panel committed a 158pt meters column and a 124pt size column
    /// inside a hard 600pt content box, and nothing checked that the
    /// model name had anywhere to go.
    @Test(
        "The models table leaves a readable name at every supported width",
        arguments: [
            CGFloat(720),   // the window floor
            CGFloat(900),   // the size the brief calls out
            CGFloat(1200),
            CGFloat(1600),  // wide — the column caps, it does not sprawl
        ]
    )
    func modelsTableFitsAtSupportedWidths(windowWidth: CGFloat) {
        let column = SettingsView.contentColumnWidth(forWindowWidth: windowWidth)
        #expect(column > 0, "No content column at \(windowWidth)pt")
        #expect(
            column <= RapidTheme.Layout.pageMaxWidth,
            "The content column must cap at the shared page measure, not sprawl."
        )

        let showsMeters = !SettingsView.isCompact(forWindowWidth: windowWidth)
        let committed = ModelTableLayout.committedRowWidth(showsMeters: showsMeters)
        let remaining = column - committed
        #expect(
            remaining >= ModelTableLayout.minimumNameWidth,
            """
            At a \(windowWidth)pt window the models table commits \(committed)pt \
            of a \(column)pt column, leaving \(remaining)pt for the model name \
            (floor \(ModelTableLayout.minimumNameWidth)pt). Either the compact \
            threshold or a column width is wrong.
            """
        )
    }

    @Test("The window floor puts the models table into its compact layout")
    func floorIsCompact() {
        #expect(
            SettingsView.isCompact(forWindowWidth: SettingsView.minWindowWidth),
            """
            At the 720pt floor the meters column has to stand down; if this \
            flips, the model name is being squeezed instead.
            """
        )
        #expect(
            !SettingsView.isCompact(forWindowWidth: 900),
            "900pt is comfortably wide enough for the meters column."
        )
    }

    @Test("Nothing the shell commits to is wider than the narrowest window")
    func shellFitsItsOwnFloor() {
        let floor = SettingsView.minWindowWidth
        let committed = SettingsView.railWidth
            + 1  // divider
            + SettingsView.contentColumnWidth(forWindowWidth: floor)
            + RapidTheme.Space.xl * 2  // page insets
        #expect(
            committed <= floor,
            "The shell commits \(committed)pt inside a \(floor)pt window."
        )
    }
}
