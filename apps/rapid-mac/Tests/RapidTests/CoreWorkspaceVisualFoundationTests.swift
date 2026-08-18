import AppKit
import Foundation
import SwiftUI
import Testing
@testable import Rapid

/// Pins for the UI-2 Slice 1 core-workspace visual refresh.
///
/// The claim this slice makes is narrow and checkable: the shell, Chat
/// and Images draw from one refreshed token set; the selected row is
/// legible by something other than a hue; the lifecycle band flexes only
/// in height; and not one production string, action, binding or
/// accessibility identifier moved. These tests hold that claim. What is
/// a matter of taste — whether the mineral canvas actually feels calmer
/// than the bone one — is held by the screenshots, not here.
///
/// Source-level guards use the same canonical (comment- and
/// literal-stripped) form the existing ``SourceGuardSupport`` helpers
/// produce, so a token named inside a doc comment can neither satisfy
/// nor break one.
@Suite("Core workspace visual foundation (UI-2 Slice 1)")
@MainActor
// Main-actor for the same reason ``SettingsVisualFoundationTests`` is: the
// CI toolchain treats several SwiftUI initialisers as MainActor-isolated
// under its stricter default isolation.
struct CoreWorkspaceVisualFoundationTests {

    private static var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // RapidTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // package root
    }

    private func strippedSource(_ relativePath: String) throws -> String {
        let url = Self.packageRoot.appendingPathComponent(relativePath)
        let body = try String(contentsOf: url, encoding: .utf8)
        return CapabilityChipRenderGateSourceGuardTests.stripCommentsAndWhitespace(body)
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

    /// Perceived luminance, for the contrast arithmetic below.
    private static func relativeLuminance(_ color: NSColor) -> Double {
        func channel(_ raw: CGFloat) -> Double {
            let v = Double(raw)
            return v <= 0.03928 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4)
        }
        return 0.2126 * channel(color.redComponent)
            + 0.7152 * channel(color.greenComponent)
            + 0.0722 * channel(color.blueComponent)
    }

    private static func contrastRatio(_ a: NSColor, _ b: NSColor) -> Double {
        let la = relativeLuminance(a)
        let lb = relativeLuminance(b)
        return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)
    }

    /// Straight-line distance in device RGB. A crude perceptual proxy,
    /// but the right one for the question these tests ask, which is
    /// "could a person tell these two planes apart at all?"
    private static func rgbDistance(_ a: NSColor, _ b: NSColor) -> Double {
        let dr = Double(a.redComponent - b.redComponent)
        let dg = Double(a.greenComponent - b.greenComponent)
        let db = Double(a.blueComponent - b.blueComponent)
        return (dr * dr + dg * dg + db * db).squareRoot()
    }

    // MARK: - The defect the selection rule exists to fix

    /// The v1.0 selected row was an amber TINT on the rail plus an amber
    /// label — two colour signals, both weak, and nothing else. This
    /// pins the replacement's load-bearing part.
    ///
    /// Note what is deliberately NOT asserted: that the new fill is
    /// *stronger* than the amber tint it replaced. It is not, and it is
    /// not meant to be. Under the refined rule the fill only has to say
    /// "this row is different"; the thing that says "this row is
    /// SELECTED" is the amber bar, which is a hard-edged shape with real
    /// contrast against the rail. Writing this test the other way round
    /// was the mistake — it measured the quiet part and let the loud part
    /// go unchecked.
    @Test("The selection bar carries real contrast against the rail")
    func selectionBarSeparatesFromTheRail() throws {
        for appearance in [NSAppearance.Name.aqua, .darkAqua] {
            let rail = try #require(Self.resolve(RapidTheme.surfaceSidebar, appearance: appearance))
            let fill = try #require(Self.resolve(RapidTheme.selectionFill, appearance: appearance))
            let bar = try #require(Self.resolve(RapidTheme.selectionBar, appearance: appearance))

            // 3:1 is the non-text threshold, and the bar is a graphic
            // object rather than a glyph.
            #expect(
                Self.contrastRatio(rail, bar) >= 3,
                """
                \(appearance.rawValue): the selection bar measures \
                \(Self.contrastRatio(rail, bar)):1 against the rail. The bar is \
                the whole signal now — if it stops separating, selection is \
                invisible again.
                """
            )
            #expect(
                Self.contrastRatio(fill, bar) >= 3,
                "\(appearance.rawValue): the bar disappears against the selected row's own fill."
            )
            // The fill still has to be doing something, however quiet.
            #expect(
                Self.rgbDistance(rail, fill) > 0,
                "\(appearance.rawValue): the selection fill is identical to the rail."
            )
        }
    }

    /// Hover must stay a clear step BELOW selection, or pointing at a row
    /// looks like having chosen it.
    @Test("Hover is weaker than selection on the rail")
    func hoverIsWeakerThanSelection() throws {
        for appearance in [NSAppearance.Name.aqua, .darkAqua] {
            let rail = try #require(Self.resolve(RapidTheme.surfaceSidebar, appearance: appearance))
            let hover = try #require(Self.resolve(RapidTheme.hoverFill, appearance: appearance))
            let selected = try #require(Self.resolve(RapidTheme.selectionFill, appearance: appearance))

            #expect(
                Self.rgbDistance(rail, hover) < Self.rgbDistance(rail, selected),
                "\(appearance.rawValue): hover reads at least as strong as selection."
            )
        }
    }

    /// Selection cannot be carried by colour alone. The rail draws a
    /// hard-edged amber bar and thickens the label, and both have to
    /// survive somebody "simplifying" the row later.
    @Test("The rail marks selection with a bar and a weight, not only a fill")
    func selectionUsesNonColourSignals() throws {
        let source = try strippedSource("Sources/Rapid/UI/SidebarView.swift")
        #expect(
            source.contains("RapidTheme.selectionBar"),
            "The sidebar no longer draws the amber selection bar."
        )
        #expect(
            source.contains("RapidTheme.Layout.selectionBarWidth"),
            "The selection bar stopped reading its width from the shared token."
        )
        #expect(
            source.contains("isSelected?RapidFont.bodyEmphasis:RapidFont.body"),
            "The nav row's label no longer thickens when selected."
        )
        #expect(
            source.contains("isActive?RapidFont.bodyEmphasis:RapidFont.body"),
            "The conversation row's title no longer thickens when active."
        )
    }

    /// The command palette is the same conversation list reached another
    /// way, so it has to answer "which row am I on" identically.
    @Test("The command palette uses the same selection rule as the rail")
    func searchPanelMatchesTheRailSelection() throws {
        let source = try strippedSource("Sources/Rapid/UI/ConversationSearchView.swift")
        #expect(source.contains("RapidTheme.selectionFill"))
        #expect(source.contains("RapidTheme.selectionBar"))
    }

    // MARK: - Amber budget

    /// One amber moment per surface. In the composer that moment is the
    /// send disc, so the readiness notice 40pt above it must not be a
    /// second amber block.
    @Test("The readiness notice does not paint a second amber block")
    func readinessBannerIsNeutralExceptOnFailure() throws {
        let source = try strippedSource("Sources/Rapid/UI/ReadinessBanner.swift")
        #expect(
            source.contains("readiness.isFailure?RapidTheme.statusErrorTint:RapidTheme.surfaceRaised"),
            """
            The readiness banner's background is no longer neutral-on-success. \
            An amber-tinted plate here competes with the send disc, which is \
            the composer's one amber moment.
            """
        )
    }

    /// Ink on amber is a near-black graphite, never white, and it is
    /// appearance-independent because the fill is. Re-pinned here because
    /// this slice repaints several amber surfaces and the pairing is the
    /// one that fails silently — white on #EFA23A measures ~2:1.
    @Test("Ink on amber clears AA in both appearances")
    func inkOnAmberIsLegible() throws {
        for appearance in [NSAppearance.Name.aqua, .darkAqua] {
            let amber = try #require(Self.resolve(RapidTheme.brandPrimary, appearance: appearance))
            let ink = try #require(Self.resolve(RapidTheme.brandOnAccent, appearance: appearance))
            let ratio = Self.contrastRatio(amber, ink)
            #expect(
                ratio >= 4.5,
                "\(appearance.rawValue): ink on amber measures \(ratio):1."
            )
        }
    }

    /// The band's two ink tiers have to be readable on its graphite
    /// ground — including the quiet tier, which carries the byte counts
    /// and the ETA a user is actively reading during a long download.
    @Test("Both band ink tiers are readable on the band ground")
    func bandInkIsLegible() throws {
        for appearance in [NSAppearance.Name.aqua, .darkAqua] {
            let ground = try #require(Self.resolve(RapidTheme.surfaceBand, appearance: appearance))
            let primary = try #require(Self.resolve(RapidTheme.bandInk, appearance: appearance))
            let secondary = try #require(Self.resolve(RapidTheme.bandInkSecondary, appearance: appearance))

            #expect(
                Self.contrastRatio(ground, primary) >= 7,
                "\(appearance.rawValue): band ink measures \(Self.contrastRatio(ground, primary)):1."
            )
            // 4.5 rather than 7: this tier is deliberately quiet, but it
            // is still prose a person reads, not decoration.
            #expect(
                Self.contrastRatio(ground, secondary) >= 4.5,
                """
                \(appearance.rawValue): the band's supporting ink measures \
                \(Self.contrastRatio(ground, secondary)):1 — the byte counts and \
                ETA are the part of a long download a user actually watches.
                """
            )
        }
    }

    /// Amber has to stand out on the band, or the progress fill and the
    /// percentage — the band's entire reason for existing — recede into
    /// their own background.
    @Test("Amber reads as progress on the band ground")
    func amberIsVisibleOnTheBand() throws {
        for appearance in [NSAppearance.Name.aqua, .darkAqua] {
            let ground = try #require(Self.resolve(RapidTheme.surfaceBand, appearance: appearance))
            let track = try #require(Self.resolve(RapidTheme.bandTrack, appearance: appearance))
            let amber = try #require(Self.resolve(RapidTheme.brandPrimary, appearance: appearance))

            #expect(Self.contrastRatio(ground, amber) >= 4.5)
            #expect(
                Self.contrastRatio(track, amber) >= 3,
                "\(appearance.rawValue): the progress fill does not separate from its own track."
            )
        }
    }

    // MARK: - The band flexes in height only

    /// The reason the band is horizontal rather than a side column: chat
    /// has a fixed 720pt reading measure, and at the 720pt window floor a
    /// vertical priority area would take width straight out of the
    /// conversation. Turned sideways, width is never contested — so the
    /// only thing allowed to respond to window width is the height.
    @Test("The band steps 132 / 112 / 44 across the three verified widths")
    func bandHeightSteps() {
        #expect(LifecycleBand.height(for: RapidTheme.Layout.Breakpoint.wide) == 132)
        #expect(LifecycleBand.height(for: RapidTheme.Layout.Breakpoint.mid) == 112)
        #expect(LifecycleBand.height(for: RapidTheme.Layout.Breakpoint.floor) == 44)
    }

    /// Monotonic, and bounded at both ends. A window wider than 1440 or
    /// narrower than 720 must not fall off a cliff into a zero or a
    /// runaway height.
    @Test("Band height is monotonic and bounded outside the verified widths")
    func bandHeightIsMonotonic() {
        let widths: [CGFloat] = [320, 640, 719, 720, 999, 1000, 1439, 1440, 2560, 5120]
        var previous = LifecycleBand.height(for: widths[0])
        for width in widths.dropFirst() {
            let height = LifecycleBand.height(for: width)
            #expect(height >= previous, "height fell going from a narrower width to \(width)")
            #expect(height >= 44 && height <= 132, "height \(height) at width \(width) is out of range")
            previous = height
        }
    }

    /// Only the floor collapses to the single-line form, and the floor is
    /// where the detail line is dropped — so this predicate decides
    /// whether a string disappears and is worth pinning on its own.
    @Test("Only the compact step collapses the band to one line")
    func bandCompactStep() {
        #expect(LifecycleBand.isCompact(width: RapidTheme.Layout.Breakpoint.floor))
        #expect(!LifecycleBand.isCompact(width: RapidTheme.Layout.Breakpoint.mid))
        #expect(!LifecycleBand.isCompact(width: RapidTheme.Layout.Breakpoint.wide))
    }

    /// The percentage is the one thing the band formats itself, so its
    /// edges are its own to get right. The byte monitor can overshoot on
    /// the final chunk; "101%" would be the band contradicting the file
    /// it just finished downloading.
    @Test("Band percentage clamps and rounds")
    func bandPercentText() {
        #expect(LifecycleBand.percentText(for: nil) == nil)
        #expect(LifecycleBand.percentText(for: 0) == "0%")
        #expect(LifecycleBand.percentText(for: 0.594) == "59%")
        #expect(LifecycleBand.percentText(for: 1) == "100%")
        #expect(LifecycleBand.percentText(for: 1.04) == "100%")
        #expect(LifecycleBand.percentText(for: -0.2) == "0%")
    }

    // MARK: - The band and the banner never speak at once

    /// The band renders the same ``ModelReadiness`` the composer's notice
    /// does. Showing both would print one fact twice, 400pt apart, in two
    /// visual languages — so the notice stands down for exactly the
    /// states the band takes.
    @Test("The composer notice stands down while the band is open")
    func bannerSuppressedUnderTheBand() throws {
        let source = try strippedSource("Sources/Rapid/UI/ChatView.swift")
        #expect(
            source.contains("if!readiness.isReady&&!showsLifecycleBand{"),
            "The readiness banner no longer stands down while the band is open."
        )
        #expect(
            source.contains("varshowsLifecycleBand:Bool{readiness.isWorking}"),
            """
            The band's gate changed. It must be exactly ModelReadiness.isWorking: \
            anything wider would open a graphite band over a decision the user \
            has not made yet, and streaming in particular must never open one.
            """
        )
    }

    /// Suppressing the banner is only safe because the states it covers
    /// carry no renderable action — otherwise a control (and its
    /// `Readiness.Action` identifier) would vanish with it.
    @Test("The states the band covers carry no renderable action")
    func bandStatesHaveNoRenderableAction() {
        let banded: [ModelReadiness] = [
            .downloading(alias: "gemma-4-26b-4bit", detail: "8.4 GB of 14.2 GB", fraction: 0.59),
            .starting(alias: "gemma-4-26b-4bit", detail: "Loading the model into memory…"),
        ]
        for state in banded {
            #expect(state.isWorking, "\(state) should be a banded state")
            #expect(
                state.action == nil,
                """
                \(state) now offers an action. The composer notice is suppressed \
                while the band is open, so that button — and the Readiness.Action \
                identifier the golden flows address it by — would disappear from \
                the surface entirely.
                """
            )
        }
    }

    /// Everything that is NOT a banded state keeps its notice, and the
    /// actionable ones keep their button.
    @Test("Decision states keep the composer notice and its action")
    func decisionStatesKeepTheirNotice() {
        let decisions: [ModelReadiness] = [
            .noModel,
            .needsDownload(alias: "gemma-4-26b-4bit", sizeText: "~14.2 GB"),
            .needsStart(alias: "gemma-4-26b-4bit"),
            .unknownModel(alias: "something-custom"),
            .failed(alias: "gemma-4-26b-4bit", message: "Couldn't start", action: .retry(alias: "gemma-4-26b-4bit")),
        ]
        for state in decisions {
            #expect(!state.isWorking, "\(state) must not open the band")
            #expect(!state.isReady, "\(state) must still show a notice")
        }
    }

    // MARK: - The band invents no copy

    /// Every sentence in the band is read off ``ModelReadiness``. The
    /// band formats exactly one thing of its own — the percentage — and
    /// a source guard is the only way to keep a stray literal out of a
    /// view this test cannot instantiate.
    @Test("The band renders readiness copy and nothing of its own")
    func bandUsesOnlyReadinessCopy() throws {
        let raw = try String(
            contentsOf: Self.packageRoot
                .appendingPathComponent("Sources/Rapid/UI/Components/LifecycleBand.swift"),
            encoding: .utf8
        )
        let source = CapabilityChipRenderGateSourceGuardTests.stripCommentsAndWhitespace(raw)
        #expect(source.contains("readiness.headline"))
        #expect(source.contains("readiness.detail"))
        #expect(source.contains("readiness.progressFraction"))
        #expect(source.contains("readiness.accessibilityLabel"))
        // The canonicaliser PRESERVES string literals (minus whitespace),
        // so a hard-coded sentence would survive verbatim. Every Text in
        // this file must therefore be reading a readiness property; a
        // `Text("` anywhere means somebody typed copy into the band.
        #expect(
            !source.contains("Text(\""),
            """
            LifecycleBand has grown a literal string. Every word it shows must \
            come off ModelReadiness, or the band can describe a moment \
            differently from the composer placeholder and the Send tooltip that \
            describe the same one.
            """
        )
    }

    // MARK: - Reduce Motion

    /// The Images HUD's sliding segment and breathing dot are perpetual
    /// loops driven by a ``TimelineView`` clock in the parent, not by an
    /// ``Animation`` the environment can suppress — so they ran
    /// regardless of the setting until this slice. Both now route through
    /// the shared predicate.
    @Test("The Images HUD loops stop under Reduce Motion")
    func imagesHUDHonoursReduceMotion() throws {
        let source = try strippedSource("Sources/Rapid/UI/ImagesView.swift")
        let occurrences = source.components(separatedBy: "RapidMotion.shouldPulse(").count - 1
        #expect(
            occurrences >= 2,
            """
            Expected both perpetual loops in the Images HUD — the indeterminate \
            progress slide and the breathing status dot — to be gated on \
            RapidMotion.shouldPulse; found \(occurrences).
            """
        )
    }

    /// The predicate itself, re-pinned here because two new call sites
    /// now depend on it suppressing a loop rather than shortening one.
    @Test("A perpetual loop is fully suppressed, not merely slowed")
    func perpetualLoopsAreSuppressed() {
        #expect(RapidMotion.shouldPulse(isAnimating: true, reduceMotion: false))
        #expect(!RapidMotion.shouldPulse(isAnimating: true, reduceMotion: true))
        #expect(!RapidMotion.shouldPulse(isAnimating: false, reduceMotion: false))
    }

    // MARK: - Ornament budget

    /// The three ornaments this slice removed from the Images progress
    /// bar. Amber is a signal here; a two-stop blend with a moving
    /// highlight and a halo reads as decoration wrapped around the
    /// signal, and the brand spec allows no gradient on the amber axis
    /// anywhere else in the app.
    @Test("The Images progress bar carries no gradient, sheen or glow")
    func imagesProgressBarIsUnornamented() throws {
        let source = try strippedSource("Sources/Rapid/UI/ImagesView.swift")
        #expect(
            !source.contains("fillGradient"),
            "The amber→gold gradient is back on the diffusion progress bar."
        )
        #expect(
            !source.contains("RapidTheme.brandAmber.opacity(0.55)"),
            "The amber glow is back under the diffusion progress bar."
        )
        #expect(
            !source.contains("RapidTheme.brandAmber.opacity(0.9)"),
            "The amber halo is back around the generating status dot."
        )
    }

    // MARK: - Chat empty state

    /// The mascot is the brand moment for Chat's empty state and for
    /// first run. Repeating it on every empty surface turned a greeting
    /// into wallpaper, so Images renders the aspect preview instead.
    @Test("The cheetah appears on the Chat empty state and not on Images")
    func mascotIsReservedForChat() throws {
        let chat = try strippedSource("Sources/Rapid/UI/ChatView.swift")
        let images = try strippedSource("Sources/Rapid/UI/ImagesView.swift")
        #expect(chat.contains("CheetahLogo("), "The Chat empty state lost the shipped cheetah asset.")
        #expect(
            !images.contains("CheetahLogo("),
            "The cheetah is back on the Images surface, where §17 V2 removes it."
        )
    }

    /// The plate is gone and the greeting is at display scale. Both are
    /// properties of the composition rather than of a token, so a source
    /// guard is what holds them.
    @Test("The Chat hero drops its backplate and takes the display tier")
    func chatHeroIsDisplayScaleAndPlateless() throws {
        let source = try strippedSource("Sources/Rapid/UI/ChatView.swift")
        #expect(source.contains("marksOnBackplate:false"))
        #expect(source.contains("titleEmphasis:.display"))
    }

    /// The display tier has to actually be bigger than the page tier, and
    /// its tracking has to be negative — SF opens up as it grows, and an
    /// untracked 34pt line reads as display type from a slide deck.
    @Test("The display tier outranks the page tier")
    func displayTierIsLarger() {
        #expect(RapidFont.displayTitle != RapidFont.pageTitle)
        #expect(RapidFont.displayTitleTracking < 0)
    }

    // MARK: - Nothing moved

    /// The identifiers this slice's surfaces are addressed by. The GUI
    /// golden flows drive the app by `AXIdentifier` and nothing else, so
    /// a visual pass that renames one is a broken harness, not a
    /// restyle. ``AccessibilityIdentifierInventoryTests`` owns the full
    /// inventory; this is the Slice 1 subset, pinned in the PR that
    /// touched these files.
    @Test("Slice 1 surfaces keep every accessibility identifier")
    func identifiersSurvivedTheRestyle() throws {
        let expectations: [String: [String]] = [
            "Sources/Rapid/UI/SidebarView.swift": [
                "Sidebar.NewChat",
                "Sidebar.Images",
                "Sidebar.Audio",
                "Sidebar.Launch",
                "Sidebar.Residency",
                "Toolbar.SearchChats",
                "Sidebar.DeleteConversation.Confirm",
                "Sidebar.DeleteConversation.Keep",
                "Sidebar.Conversation.Action.Rename",
                "Sidebar.Conversation.Action.Delete",
            ],
            "Sources/Rapid/UI/ChatView.swift": [
                "ChatView.SendOrStopButton",
                "ChatView.AddAttachments",
                "ChatView.ConversationInstructions",
            ],
            "Sources/Rapid/UI/ImagesView.swift": [
                "Images.Stage",
                "Images.EmptyState",
                "Images.Prompt",
                "Images.Cancel",
                "Images.Gallery",
                "Images.Result.Edit",
                "Images.Result.Save",
            ],
            "Sources/Rapid/UI/ReadinessBanner.swift": [
                "Readiness.Action",
            ],
        ]
        for (path, identifiers) in expectations {
            let source = try strippedSource(path)
            for identifier in identifiers {
                #expect(
                    source.contains(".accessibilityIdentifier(\"\(identifier)\")"),
                    "\(path) no longer declares \(identifier)."
                )
            }
        }
    }

    /// The nav order and the SF Symbols behind it. §15's audit turns on
    /// these four rows being untouched, and a symbol swap is the kind of
    /// thing a visual pass does without noticing.
    @Test("Navigation keeps its four rows, their symbols and their order")
    func navigationIsUnchanged() throws {
        let source = try strippedSource("Sources/Rapid/UI/SidebarView.swift")
        let rows: [(title: String, symbol: String)] = [
            ("New Chat", "square.and.pencil"),
            ("Images", "photo"),
            ("Audio", "waveform"),
            ("Launch", "paperplane"),
        ]
        var searchStart = source.startIndex
        for row in rows {
            // The canonicaliser strips whitespace INSIDE preserved string
            // literals too, so "New Chat" arrives here as "NewChat".
            let title = row.title.filter { !$0.isWhitespace }
            let needle = "title:\"\(title)\",systemImage:\"\(row.symbol)\""
            guard let found = source.range(of: needle, range: searchStart..<source.endIndex) else {
                Issue.record("\(row.title) / \(row.symbol) is missing, renamed, or out of order.")
                return
            }
            searchStart = found.upperBound
        }
    }

    /// The brand lockup added to the rail must stay decoration. Without
    /// this it becomes the first element in every VoiceOver traversal of
    /// the sidebar, ahead of the first thing you can actually do there —
    /// and it would add an AX node the identifier inventory never
    /// accounted for.
    @Test("The sidebar brand lockup is accessibility-hidden")
    func brandLockupIsDecorative() throws {
        let source = try strippedSource("Sources/Rapid/UI/SidebarView.swift")
        guard let lockup = source.range(of: "varbrandLockup:someView{") else {
            Issue.record("The brand lockup is gone.")
            return
        }
        let block = SourceGuardSupport.balancedBlock(
            in: source,
            openingBraceAt: source.index(before: lockup.upperBound)
        )
        #expect(
            block?.contains(".accessibilityHidden(true)") == true,
            "The sidebar brand lockup is now announced to VoiceOver."
        )
    }

    /// The composer's control lane is unchanged, and the send action is
    /// still the composer's one amber moment.
    @Test("The composer keeps its controls and its single amber action")
    func composerControlsAreIntact() throws {
        let source = try strippedSource("Sources/Rapid/UI/ChatView.swift")
        #expect(source.contains("ModelPickerBar("))
        #expect(source.contains("sendOrStopButton"))
        #expect(
            source.contains("sendEnabled?RapidTheme.brandPrimary:Color.clear"),
            "The send disc is no longer the composer's amber moment."
        )
    }

    // MARK: - Responsive contract

    /// Breakpoints belong to the view receiving the geometry: Chat's detail
    /// pane, not the outer window. Pin the conversion so screenshots cannot
    /// accidentally test one coordinate system while production uses another.
    @Test("Window review widths map to the intended detail breakpoints")
    func windowWidthsMapToDetailBreakpoints() {
        #expect(720 - SidebarView.columnIdealWidth == RapidTheme.Layout.Breakpoint.floor)
        #expect(1000 - SidebarView.columnIdealWidth == RapidTheme.Layout.Breakpoint.mid)
        #expect(1440 - SidebarView.columnIdealWidth == RapidTheme.Layout.Breakpoint.wide)
    }

    /// A declared floor the shell never applies is not a floor.
    ///
    /// ``ContentView/minWindowWidth`` spent a while as a constant read only by
    /// the test below, while the window's real minimum came from whatever the
    /// rail's 176pt column minimum and the 440pt detail floor happened to add
    /// up to — about 616pt. The number and the window could disagree
    /// indefinitely and nothing would notice. `.windowResizability` reads the
    /// content's minimum size, so the shell has to state it.
    ///
    /// ViewInspector is not in this target (#1492), so this is a source guard
    /// in the same shape as ``AccessibilityIdentifierInventoryTests``.
    @Test("The declared window floor is applied to the shell")
    func windowFloorIsEnforced() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/Rapid/UI/ContentView.swift")
        let source = try String(contentsOf: url, encoding: .utf8)
        let stripped = CapabilityChipRenderGateSourceGuardTests
            .stripCommentsAndWhitespace(source)
        // Reduced to a Bool before the expectation: `#expect` prints the
        // expression it was given, and handing it the stripped source dumps
        // the whole of ContentView into the failure output.
        let applied = stripped.contains(
            ".frame(minWidth:Self.minWindowWidth,minHeight:Self.minWindowHeight)"
        )
        #expect(
            applied,
            "ContentView declares minWindowWidth but no longer applies it — .windowResizability(.contentMinSize) takes the window floor from the content, so without this the constant is decoration."
        )
    }

    /// The detail pane's floor plus the widest rail has to fit inside the
    /// window floor, or the shell clips horizontally before any surface
    /// gets a say.
    @Test("The rail and the detail floor fit inside the window floor")
    func shellFitsAtTheWindowFloor() {
        let committed = SidebarView.columnIdealWidth + 440
        #expect(
            committed <= ContentView.minWindowWidth,
            """
            The rail (\(SidebarView.columnIdealWidth)) plus the detail floor (440) \
            commits \(committed)pt inside a \(ContentView.minWindowWidth)pt window.
            """
        )
    }
}
