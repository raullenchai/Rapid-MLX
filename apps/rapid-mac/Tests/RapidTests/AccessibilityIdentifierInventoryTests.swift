import Foundation
import Testing
@testable import Rapid

/// Inventory pin for the `.accessibilityIdentifier(...)` values the GUI
/// golden-flow harness addresses controls by.
///
/// `scripts/gui-golden-flows.sh` + `scripts/rapid-ax.swift` drive the app by
/// `AXIdentifier` and nothing else — an unnamed control is a control the suite
/// cannot reach, and `docs/gui-golden-flows.md` says so out loud ("Prefer a
/// stable `.accessibilityIdentifier(...)` in product code"). Several surfaces
/// shipped with either no identifier at all (Settings → Tools, the
/// conversation-row overflow menu) or with an SF Symbol name leaking through
/// as one (`pin`, `doc.on.doc`, `pencil`, `arrow.clockwise`) — the latter is
/// worse than nothing, because it looks deliberate and silently changes the
/// moment somebody swaps the glyph.
///
/// ViewInspector is not in this target (#1492), so — like
/// ``SidebarConversationDeleteConfirmationTests`` — the wiring is pinned by a
/// source-grep guard over the canonical (comment- and whitespace-stripped)
/// form of each view file. Deleting or renaming one of these identifiers trips
/// the test with the identifier and its file named, so the harness change lands
/// in the same PR as the product change.
@Suite("Accessibility identifier inventory")
struct AccessibilityIdentifierInventoryTests {

    private static var sourceRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // RapidTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // package root
    }

    private func strippedSource(_ relativePath: String) throws -> String {
        let url = Self.sourceRoot.appendingPathComponent(relativePath)
        let body = try String(contentsOf: url, encoding: .utf8)
        return CapabilityChipRenderGateSourceGuardTests.stripCommentsAndWhitespace(body)
    }

    /// The literal identifier expression each surface must declare, in the
    /// canonical stripped form. Interpolated entries keep their `\(…)` so the
    /// pin also fails if somebody swaps the per-item key (tool name, provider
    /// id, conversation id, message id) for display text.
    private func assertDeclared(
        _ identifiers: [String],
        in relativePath: String,
        surface: String
    ) throws {
        let stripped = try strippedSource(relativePath)
        for identifier in identifiers {
            let shape = ".accessibilityIdentifier(\(identifier))"
            #expect(
                stripped.contains(shape),
                """
                \(surface): \(relativePath) no longer declares \
                \(shape). Golden flows address this control by AXIdentifier — \
                removing or renaming it makes the control unreachable. Update \
                scripts/gui-golden-flows.sh and this inventory together.
                """
            )
        }
    }

    // MARK: - Settings → Developer (debug builds only)

    /// The panel and its controls exist only in a debug build, and so does
    /// this pin — asserting the file's contents from a release test run would
    /// fail against source that is deliberately compiled out.
    #if DEBUG
    @Test("Settings → Developer names its panel and every control")
    func settingsDeveloperPanelIdentifiers() throws {
        try assertDeclared(
            [
                #""Settings.Developer.Panel""#,
                #""Settings.Developer.Scope.Preferences""#,
                #""Settings.Developer.Scope.Conversations""#,
                #""Settings.Developer.Scope.Telemetry""#,
                #""Settings.Developer.Reonboard""#,
                #""Settings.Developer.ConfirmReonboard""#,
                #""Settings.Developer.CancelReonboard""#,
            ],
            in: "Sources/Rapid/UI/SettingsDeveloperPanel.swift",
            surface: "Settings → Developer"
        )
    }
    #endif

    // MARK: - Settings → Tools

    @Test("Settings → Performance names its panel and every control")
    func settingsPerformancePanelIdentifiers() throws {
        try assertDeclared(
            [
                #""Settings.Performance.Panel""#,
                #""Settings.Performance.NoModel""#,
                #""Settings.Performance.RestartNotice""#,
                #""Settings.Performance.ModelPicker""#,
                #""Settings.Performance.KVMode""#,
                #""Settings.Performance.PrefixCache""#,
                #""Settings.Performance.CacheBudget""#,
                #""Settings.Performance.CacheBudgetAutomatic""#,
                #""Settings.Performance.Reset""#,
                #""Settings.Performance.AppliesNextLoad""#,
            ],
            in: "Sources/Rapid/UI/SettingsPerformancePanel.swift",
            surface: "Settings → Performance"
        )
        let source = try strippedSource("Sources/Rapid/UI/SettingsPerformancePanel.swift")
        for id in ["Settings.Performance.Prefix.Default",
                   "Settings.Performance.Prefix.On",
                   "Settings.Performance.Prefix.Off"] {
            #expect(
                source.contains("identifier:\"\(id)\""),
                "Settings → Performance: segmented option \(id) is no longer addressable."
            )
        }
    }

    /// The whole panel shipped with ZERO identifiers: the three tool switches,
    /// the web-search backend radio group, and the browsing auto-approve
    /// switch all had correct AX roles and no name.
    @Test("Settings → Tools names every control it offers")
    func settingsToolsPanelIdentifiers() throws {
        try assertDeclared(
            [
                // Keyed on the tool's WIRE name (web_search / browse /
                // weather), which is also what the request body carries.
                #""Settings.Tools.Toggle.\(def.function.name)""#,
                // Radio group, then one radio per backend keyed on the
                // provider's raw value.
                #""Settings.Tools.WebSearch.Backend""#,
                #""Settings.Tools.WebSearch.Backend.\(provider.id)""#,
                #""Settings.Tools.WebSearch.KeyField.\(provider.id)""#,
                #""Settings.Tools.WebSearch.SaveKey.\(provider.id)""#,
                // The "Get a <provider> key" Link is interactive too — it
                // opens the provider's dashboard. It was missed on the first
                // pass precisely because it only renders for a provider that
                // has a `keyDashboardURL`, so it is absent from an AX dump
                // taken with any other backend selected. "Every control" has
                // to mean every control, including the conditional ones.
                #""Settings.Tools.WebSearch.KeyDashboardLink.\(provider.id)""#,
                #""Settings.Tools.Browse.AutoApproveToggle""#,
            ],
            in: "Sources/Rapid/UI/SettingsToolsPanel.swift",
            surface: "Settings → Tools"
        )
    }

    // MARK: - Settings → Connectors (issue #1716)

    /// The connector surface is where a user authorises a program on their Mac
    /// to be driven by a model, so every control on it has to be reachable by
    /// the golden-flow harness — not just the happy-path ones.
    @Test("Settings → Connectors names every control it offers")
    func settingsConnectorsPanelIdentifiers() throws {
        try assertDeclared(
            [
                #""Settings.Connectors.MasterToggle""#,
                #""Settings.Connectors.AddButton""#,
                #""Settings.Connectors.SubsystemError""#,
                #""Settings.Connectors.RestartButton""#,
                #""Settings.Connectors.ConfirmRemove""#,
                #""Settings.Connectors.CancelRemove""#,
                // Per-server rows, keyed on the server's name — which is also
                // the namespace half of every tool it exposes, not display text.
                #""Settings.Connectors.Row.Status.\(entry.name)""#,
                #""Settings.Connectors.Row.Toggle.\(entry.name)""#,
                #""Settings.Connectors.Row.Menu.\(entry.name)""#,
                #""Settings.Connectors.Row.Edit.\(entry.name)""#,
                #""Settings.Connectors.Row.Remove.\(entry.name)""#,
                // Per-tool switch, keyed on the namespaced wire name.
                #""Settings.Connectors.Tool.Toggle.\(name)""#,
                #""Settings.Connectors.AutoApproveToggle""#,
                #""Settings.Connectors.ResetApprovals""#,
            ],
            in: "Sources/Rapid/UI/SettingsConnectorsPanel.swift",
            surface: "Settings → Connectors"
        )
    }

    @Test("The connector editor sheet names every field")
    func mcpServerEditorSheetIdentifiers() throws {
        try assertDeclared(
            [
                #""Settings.Connectors.Editor.Name""#,
                #""Settings.Connectors.Editor.Transport""#,
                #""Settings.Connectors.Editor.Command""#,
                #""Settings.Connectors.Editor.URL""#,
                #""Settings.Connectors.Editor.Enabled""#,
                #""Settings.Connectors.Editor.Allow""#,
                #""Settings.Connectors.Editor.Cancel""#,
            ],
            in: "Sources/Rapid/UI/MCPServerEditorSheet.swift",
            surface: "Settings → Connectors → editor"
        )
        // The two code editors route their identifier through the shared
        // `codeEditor(text:height:axIdentifier:)` builder so the modifier
        // lands on the `TextEditor` itself (the AX-identifier gate checks
        // the control, and a wrapper-level duplicate would give the AX
        // driver two hits). Pin the literal at the call sites instead —
        // the identifier STRINGS are unchanged, so golden flows are
        // unaffected.
        let sheet = try strippedSource("Sources/Rapid/UI/MCPServerEditorSheet.swift")
        for id in ["Settings.Connectors.Editor.AddArgument",
                   "Settings.Connectors.Editor.AddEnv"] {
            #expect(
                sheet.contains(#"axIdentifier:"\#(id)""#),
                """
                Settings → Connectors → editor: MCPServerEditorSheet.swift no \
                longer passes \(id) into codeEditor(text:height:axIdentifier:). \
                Golden flows address this field by AXIdentifier — update \
                scripts/gui-golden-flows.sh and this inventory together.
                """
            )
        }
    }

    /// The consent prompt is the last thing standing between a model and a
    /// program on the user's Mac. Its three buttons must be addressable.
    @Test("The MCP tool approval sheet names its three decisions")
    func mcpApprovalSheetIdentifiers() throws {
        try assertDeclared(
            [
                #""ToolApproval.MCP.Deny""#,
                #""ToolApproval.MCP.AlwaysAllow""#,
                #""ToolApproval.MCP.Allow""#,
            ],
            in: "Sources/Rapid/UI/ContentView.swift",
            surface: "MCP tool approval sheet"
        )
    }

    /// Behaviour guard for the constraint the identifiers ride on: a settings
    /// switch must stay a REAL `Toggle` with the native `.switch` style, so it
    /// keeps reporting `AXCheckBox` with a value that tracks state and flips
    /// when pressed. #1608 already had to walk back a hand-rolled
    /// `HStack` + `.accessibilityRepresentation` version of this style; naming
    /// the toggles must not tempt anyone back into it.
    @Test("TrailingSettingsToggleStyle still renders a native switch Toggle")
    func trailingToggleStyleKeepsNativeSemantics() throws {
        let stripped = try strippedSource("Sources/Rapid/UI/SettingsControlStyles.swift")
        // The invariant is that a REAL `Toggle` bound to the
        // configuration is what draws the switch — not that the Toggle
        // is the syntactic root of `makeBody`. UI-1's refinement pass
        // wraps it in an `HStack` to hold the label column and the
        // trailing gutter, which is layout around the control and does
        // not touch its semantics: the switch is still a `Toggle`, still
        // `.switch`, and still carries its own AXCheckBox role and
        // value. The `.accessibilityRepresentation` assertion below is
        // the one that actually catches a regression to #1608's
        // hand-rolled row, and it is unchanged.
        #expect(
            stripped.contains("Toggle(isOn:binding){"),
            "TrailingSettingsToggleStyle must wrap a real Toggle — a hand-rolled row loses the AXCheckBox role and its value."
        )
        #expect(
            stripped.contains(".toggleStyle(.switch)"),
            "TrailingSettingsToggleStyle must keep the native .switch style."
        )
        #expect(
            !stripped.contains(".accessibilityRepresentation"),
            "TrailingSettingsToggleStyle must not re-introduce the #1608 .accessibilityRepresentation shim — the real Toggle already carries the semantics."
        )
    }

    // MARK: - Settings → Privacy

    /// The second panel `no-dead-controls` stopped on. Same shape of gap as
    /// Tools: a working telemetry switch and three working policy links, none
    /// of them addressable.
    @Test("Settings → Privacy names its toggle and its policy links")
    func settingsPrivacyPanelIdentifiers() throws {
        try assertDeclared(
            [
                #""Settings.Privacy.TelemetryToggle""#,
                // Named for the document each link opens, not for its visible
                // label — "License (EULA)" is exactly the sort of string that
                // gets reworded.
                #""Settings.Privacy.Link.PrivacyPolicy""#,
                #""Settings.Privacy.Link.License""#,
                #""Settings.Privacy.Link.Credits""#,
            ],
            in: "Sources/Rapid/UI/SettingsView.swift",
            surface: "Settings → Privacy"
        )
    }

    // MARK: - Sidebar conversation row

    /// The row itself was already addressable; its controls were not. The
    /// overflow `Menu` reported `missing value`, its items were unreachable,
    /// and the hover pin button's identifier was the SF Symbol name `pin`.
    @Test("Conversation row controls and menu items are addressable")
    func sidebarConversationIdentifiers() throws {
        try assertDeclared(
            [
                #""Sidebar.Conversation.Menu.\(conv.id.uuidString)""#,
                // Menu items — shared by the ··· menu and the right-click
                // context menu, so exactly one set of ids covers both.
                #""Sidebar.Conversation.Action.Rename""#,
                #""Sidebar.Conversation.Action.Delete""#,
                // Delete confirmation. These ride AppKit's alert bridge
                // rather than an ordinary SwiftUI button; measured on this
                // branch the bridge does forward them (the dialog is an
                // AXSheet whose two AXButton children carry these ids).
                #""Sidebar.DeleteConversation.Confirm""#,
                #""Sidebar.DeleteConversation.Keep""#,
            ],
            in: "Sources/Rapid/UI/SidebarView.swift",
            surface: "Sidebar conversation row"
        )
        // The state-dependent ids are pure helpers, so they get pinned by
        // value rather than by grep — and the render sites must go through
        // them rather than re-deriving the string inline.
        let stripped = try strippedSource("Sources/Rapid/UI/SidebarView.swift")
        for shape in [
            ".accessibilityIdentifier(Self.pinControlIdentifier(for:conv))",
            ".accessibilityIdentifier(Self.pinMenuItemIdentifier(for:conv))",
            ".accessibilityIdentifier(Self.archiveMenuItemIdentifier(for:conv))",
        ] {
            #expect(
                stripped.contains(shape),
                "SidebarView must apply \(shape) — an inline ternary would drift from the helper the tests pin."
            )
        }
    }

    /// The pin / archive controls change what they DO with the row's state, so
    /// their identifiers change with it: a golden flow presses `…Action.Pin`
    /// and then asserts `…Action.Unpin` is what is now offered, which is a
    /// real observable-state assertion rather than a sleep.
    @Test("Pin and archive identifiers name the action the press performs")
    func stateDependentIdentifiersFollowTheAction() {
        let id = UUID()
        func conversation(pinned: Bool = false, archived: Bool = false) -> ChatConversation {
            ChatConversation(
                id: id,
                title: "Chat",
                messages: [],
                createdAt: Date(),
                updatedAt: Date(),
                isPinned: pinned,
                isArchived: archived
            )
        }

        #expect(
            SidebarView.pinControlIdentifier(for: conversation())
                == "Sidebar.Conversation.Pin.\(id.uuidString)"
        )
        #expect(
            SidebarView.pinControlIdentifier(for: conversation(pinned: true))
                == "Sidebar.Conversation.Unpin.\(id.uuidString)"
        )
        #expect(
            SidebarView.pinMenuItemIdentifier(for: conversation())
                == "Sidebar.Conversation.Action.Pin"
        )
        #expect(
            SidebarView.pinMenuItemIdentifier(for: conversation(pinned: true))
                == "Sidebar.Conversation.Action.Unpin"
        )
        #expect(
            SidebarView.archiveMenuItemIdentifier(for: conversation())
                == "Sidebar.Conversation.Action.Archive"
        )
        #expect(
            SidebarView.archiveMenuItemIdentifier(for: conversation(archived: true))
                == "Sidebar.Conversation.Action.Unarchive"
        )
    }

    /// The pin control's id must be derived from the conversation id, never
    /// from the glyph. A bare `"pin"` anywhere in an identifier position is the
    /// exact regression this closes.
    @Test("Sidebar identifiers are never SF Symbol names")
    func sidebarIdentifiersAreNotSymbolNames() throws {
        let stripped = try strippedSource("Sources/Rapid/UI/SidebarView.swift")
        for symbol in ["pin", "pin.slash", "ellipsis", "archivebox", "pencil"] {
            #expect(
                !stripped.contains(".accessibilityIdentifier(\"\(symbol)\")"),
                "SidebarView declares the SF Symbol name '\(symbol)' as an accessibility identifier — identifiers must be semantic, not glyph names."
            )
        }
    }

    // MARK: - Chat message actions

    /// Copy / edit / retry (and the two edit-mode controls) previously
    /// surfaced as `doc.on.doc`, `pencil` and `arrow.clockwise`.
    @Test("Message action buttons carry semantic, per-message identifiers")
    func messageActionIdentifiers() throws {
        let stripped = try strippedSource("Sources/Rapid/UI/ChatView.swift")
        // The id builder: fixed English action key + the message id, so a
        // transcript of several rows addresses the right one.
        #expect(
            stripped.contains(#""ChatView.Message.\(action).\(message.id.uuidString)""#),
            "MessageRow.actionIdentifier must key on the action name AND the message id — a shared id would make every row's Copy button the same element."
        )
        try assertDeclared(
            [
                #"actionIdentifier("Copy")"#,
                #"actionIdentifier("Edit")"#,
                #"actionIdentifier("Retry")"#,
                #"actionIdentifier("CancelEdit")"#,
                #"actionIdentifier("SaveEdit")"#,
            ],
            in: "Sources/Rapid/UI/ChatView.swift",
            surface: "Chat message actions"
        )
        for symbol in ["doc.on.doc", "pencil", "arrow.clockwise", "checkmark", "xmark"] {
            #expect(
                !stripped.contains(".accessibilityIdentifier(\"\(symbol)\")"),
                "ChatView declares the SF Symbol name '\(symbol)' as an accessibility identifier — identifiers must be semantic, not glyph names."
            )
        }
    }

    // MARK: - Tool approval

    /// The `browse` per-fetch approval is this app's tool-approval prompt. It
    /// is a real SwiftUI `.sheet`, so — unlike a `confirmationDialog` — its
    /// buttons are ordinary SwiftUI buttons and carry identifiers directly.
    ///
    /// The three *answers* are named; the enclosing stack deliberately is not.
    /// An accessibility modifier on a container that is not its own
    /// accessibility element applies to the elements it contains, so naming
    /// the wrapper risks stamping that name across its descendants — including
    /// over the buttons a flow has to press. A flow asserts the prompt is up
    /// by waiting for `ToolApproval.Browse.Allow`.
    @Test("The browse tool-approval answers are addressable")
    func browseApprovalIdentifiers() throws {
        try assertDeclared(
            [
                #""ToolApproval.Browse.Allow""#,
                #""ToolApproval.Browse.AlwaysAllow""#,
                #""ToolApproval.Browse.Deny""#,
            ],
            in: "Sources/Rapid/UI/ContentView.swift",
            surface: "Browse tool approval"
        )
    }

    // MARK: - Onboarding

    /// Onboarding's addressable controls, including the Ready surface added
    /// with Onboarding V3.
    ///
    /// Ready matters more than the rest here: it is the one screen a golden
    /// flow can now be *stuck* on. Readiness no longer dismisses setup, so a
    /// harness that cannot find and press Start chatting never reaches the
    /// app at all — it hangs on a screen that looks finished.
    @Test("Onboarding names every step control, including the Ready confirmation")
    func onboardingIdentifiers() throws {
        try assertDeclared(
            [
                #""Quickstart.GetStarted""#,
                #""Quickstart.Skip""#,
                #""Quickstart.BrowseAll""#,
                #""Quickstart.LowDisk.Continue""#,
                #""Quickstart.LowDisk.Cancel""#,
                #""Quickstart.Memory.SwitchToLowMemory""#,
                #""Quickstart.Memory.LoadAnyway""#,
                #""Quickstart.Memory.Cancel""#,
                #""Quickstart.Ready.StartChatting""#,
            ],
            in: "Sources/Rapid/UI/QuickstartView.swift",
            surface: "Onboarding"
        )
        // The rail and the footer lane moved into the Direction D design
        // system when the wizard's centred-card chrome was replaced. The
        // identifiers did NOT move: golden flows address all three, and
        // `Quickstart.Progress` in particular is matched with its exact
        // spoken label on three screens.
        try assertDeclared(
            [
                #""Quickstart.Progress""#,
                #""Quickstart.Footer.Back""#,
                #""Quickstart.Footer.Primary""#,
            ],
            in: "Sources/Rapid/UI/OnboardingDirectionD.swift",
            surface: "Onboarding shell"
        )
        try assertDeclared(
            [
                #""Quickstart.Choice.\(choice.alias)""#,
                #""Quickstart.CatalogRow.\(alias)""#,
            ],
            in: "Sources/Rapid/UI/OnboardingComponents.swift",
            surface: "Onboarding model rows"
        )
        // Review download's read-only content. These reach the accessibility
        // tree through a component parameter (``OnboardingFactRow`` and
        // ``OnboardingInlineNote`` both forward `identifier` to
        // `.accessibilityIdentifier`), so they are pinned in their declared
        // form rather than by the modifier shape the helper above matches.
        //
        // `Quickstart.Review.Incompatible` is the one a harness needs in order
        // to assert that a WON'T FIT row explains itself: without it the only
        // evidence that the screen said anything is a screenshot.
        let review = try strippedSource("Sources/Rapid/UI/QuickstartView.swift")
        for identifier in [
            "Quickstart.Review.Alias",
            "Quickstart.Review.Size",
            "Quickstart.Review.CachedStatus",
            "Quickstart.Review.Memory",
            "Quickstart.Review.UsableMemory",
            "Quickstart.Review.FreeSpace",
            "Quickstart.Review.Repo",
            "Quickstart.Review.Incompatible",
        ] {
            #expect(
                review.contains(#"identifier:"\#(identifier)""#),
                """
                Onboarding Review: QuickstartView.swift no longer declares \
                identifier: "\(identifier)". Golden flows read Review's facts \
                by AXIdentifier — removing one makes that fact unassertable.
                """
            )
        }
    }

    // MARK: - Launch integrations

    @Test("Launch integration copy buttons are individually addressable")
    func launchIntegrationCopyIdentifiers() throws {
        try assertDeclared(
            [#""Launch.Integration.Copy.\(tool.id)""#],
            in: "Sources/Rapid/UI/ConnectToolsView.swift",
            surface: "Launch integration rows"
        )
    }

    /// The positive test above would still pass if someone re-added the
    /// container identifier, bringing the propagation bug back with it. Naming
    /// the wrapper is the mistake, so the absence has to be asserted directly.
    @Test("The tool-approval sheet's container stays unnamed")
    func browseApprovalSheetContainerIsNotNamed() throws {
        let stripped = try strippedSource("Sources/Rapid/UI/ContentView.swift")
        #expect(
            !stripped.contains(#".accessibilityIdentifier("ToolApproval.Browse.Sheet")"#),
            """
            The approval sheet's root stack must NOT carry an identifier. An \
            accessibility modifier on a container that is not its own \
            accessibility element is applied to the elements it contains, so \
            naming the wrapper can stamp that name across its descendants — \
            including over the Allow/Deny buttons a flow has to press. Assert \
            the prompt is up by waiting for ToolApproval.Browse.Allow instead.
            """
        )
    }
}
