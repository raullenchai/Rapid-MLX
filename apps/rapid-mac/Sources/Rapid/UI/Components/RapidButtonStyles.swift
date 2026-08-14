import SwiftUI

/// The app's four button tiers.
///
/// Before v1.0 every call site hand-rolled its own button: some used
/// ``.borderedProminent`` (which paints the system accent, or — after
/// the scene-root `.tint` — amber with unreadable white text), some
/// built a `Capsule` with a hand-picked fill, some used `.bordered`
/// with a `.tint` override. The result was five different heights and
/// three different meanings for "this is the important one".
///
/// The tiers, and the single question that picks between them:
///
///   * ``rapidPrimary`` — is this THE action on this surface? Amber
///     fill, graphite label. At most one per view.
///   * ``rapidSecondary`` — a real action, but not the main one.
///     Outlined, neutral label, optional steel-blue accent.
///   * ``rapidTertiary`` — borderless text/icon. Present, not
///     competing.
///   * ``rapidDestructive`` — Stop, Delete, and nothing else.
///
/// All four share height, corner radius, icon size, and the full
/// hover / pressed / disabled / focus set, so tiers can sit next to
/// each other on one baseline without any per-site padding fixes.
///
/// Heights come from ``RapidTheme.ControlHeight``: primary is 36pt,
/// everything else defaults to 32pt with a 28pt compact variant. That
/// keeps every control at or above the 28pt hit-target floor.

// MARK: - Primary

/// Amber fill, graphite label. The one high-emphasis action.
///
/// The label colour is the load-bearing detail: white on #EFA23A is
/// ~2.0:1 and was the single most common contrast failure in the
/// pre-v1.0 surface. ``RapidTheme.onBrandPrimary`` is ~9:1.
struct RapidPrimaryButtonStyle: ButtonStyle {
    // Explicit init on every style in this file: a `private` stored
    // property (the `@Environment` below) makes the SYNTHESISED
    // memberwise initialiser private too, so call sites in other files
    // could not construct these with arguments.
    @Environment(\.isEnabled) private var isEnabled
    /// Fill the available width — for full-width CTAs in cards/sheets.
    var expands: Bool
    var height: CGFloat

    /// Default height is ``ControlHeight/medium`` (32) — the SAME
    /// regular-command height as ``RapidSecondaryButtonStyle`` and
    /// ``RapidDestructiveButtonStyle``.
    ///
    /// It was ``large`` (36), which meant "regular command button" had
    /// two heights depending on tier: a Cancel/Save pair in a sheet came
    /// out 32/36 and visibly stepped. Emphasis is carried by the FILL,
    /// not by being 4pt taller. The hero, full-width case keeps 36 via
    /// ``rapidPrimaryWide``.
    init(expands: Bool = false, height: CGFloat = RapidTheme.ControlHeight.medium) {
        self.expands = expands
        self.height = height
    }

    func makeBody(configuration: Configuration) -> some View {
        RapidButtonSurface(
            configuration: configuration,
            isEnabled: isEnabled,
            expands: expands,
            height: height,
            foreground: RapidTheme.primaryActionLabel,
            fill: RapidTheme.primaryActionFill,
            stroke: nil
        )
    }
}

// MARK: - Secondary

/// Outlined, neutral label. A real action that is not the primary one.
///
/// ``accented`` swaps the label to steel blue for utility/informational
/// actions (Copy config, Test, Open docs) — the secondary brand colour
/// doing secondary-brand work instead of filling primary buttons.
struct RapidSecondaryButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled
    /// Treat this as a utility control: neutral at rest, steel blue
    /// under the pointer. This is the default shape for Copy / Reveal /
    /// per-row actions.
    var utility: Bool
    var expands: Bool
    var height: CGFloat
    /// Explicit label colour, overriding every default. Needed where a
    /// button signals transient state through colour (Copy → Copied
    /// flips to the ready green) — an outer `.foregroundStyle` can't
    /// reach inside a ButtonStyle.
    var foreground: Color?

    init(
        utility: Bool = false,
        expands: Bool = false,
        height: CGFloat = RapidTheme.ControlHeight.medium,
        foreground: Color? = nil
    ) {
        self.utility = utility
        self.expands = expands
        self.height = height
        self.foreground = foreground
    }

    func makeBody(configuration: Configuration) -> some View {
        RapidButtonSurface(
            configuration: configuration,
            isEnabled: isEnabled,
            expands: expands,
            height: height,
            foreground: foreground ?? RapidTheme.secondaryActionLabel,
            hoverForeground: (foreground == nil && utility)
                ? RapidTheme.utilityActionHover
                : nil,
            fill: RapidTheme.surfaceRaised,
            stroke: RapidTheme.hairlineStrong
        )
    }
}

// MARK: - Tertiary

/// Borderless text or icon button. No fill until hovered.
struct RapidTertiaryButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled
    /// A genuine text link. This is one of the few sanctioned uses of
    /// steel blue at rest.
    var link: Bool
    var height: CGFloat

    init(link: Bool = false, height: CGFloat = RapidTheme.ControlHeight.small) {
        self.link = link
        self.height = height
    }

    func makeBody(configuration: Configuration) -> some View {
        RapidButtonSurface(
            configuration: configuration,
            isEnabled: isEnabled,
            expands: false,
            height: height,
            foreground: link ? RapidTheme.linkLabel : .primary,
            hoverForeground: link ? nil : RapidTheme.utilityActionHover,
            fill: .clear,
            stroke: nil
        )
    }
}

// MARK: - Destructive

/// Stop / Delete. Red fill, white label.
struct RapidDestructiveButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled
    var expands: Bool
    var height: CGFloat

    init(expands: Bool = false, height: CGFloat = RapidTheme.ControlHeight.medium) {
        self.expands = expands
        self.height = height
    }

    func makeBody(configuration: Configuration) -> some View {
        RapidButtonSurface(
            configuration: configuration,
            isEnabled: isEnabled,
            expands: expands,
            height: height,
            foreground: RapidTheme.destructiveActionLabel,
            fill: RapidTheme.destructiveActionFill,
            stroke: nil
        )
    }
}

// MARK: - Shared surface

/// The one place button chrome is drawn. Every tier routes through
/// here, which is what keeps their metrics and interaction states
/// identical by construction rather than by convention.
private struct RapidButtonSurface: View {
    let configuration: ButtonStyleConfiguration
    let isEnabled: Bool
    let expands: Bool
    let height: CGFloat
    let foreground: Color
    /// Label colour under the pointer. ``nil`` keeps ``foreground``.
    /// This is how utility controls earn steel blue on hover without
    /// wearing it at rest.
    var hoverForeground: Color? = nil
    let fill: Color
    let stroke: Color?

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var hovering = false

    private var resolvedForeground: Color {
        guard isEnabled, hovering, let hoverForeground else { return foreground }
        return hoverForeground
    }

    var body: some View {
        configuration.label
            .font(RapidFont.bodyEmphasis)
            .foregroundStyle(resolvedForeground)
            // Icons inside a Label track the text size rather than the
            // SF Symbol default, so a `Label("Copy", systemImage:)`
            // can't render a glyph two sizes larger than its own word.
            .imageScale(.small)
            .lineLimit(1)
            .padding(.horizontal, RapidTheme.Space.md)
            .frame(maxWidth: expands ? .infinity : nil)
            .frame(height: height)
            .background(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.button, style: .continuous)
                    .fill(fill)
            )
            .overlay {
                // Hover and pressed are drawn as a wash ON TOP of the
                // tier's own fill, so one pair of values reads correctly
                // over amber, over white, and over nothing.
                //
                // Both values must therefore be translucent. ``hoverFill`` is
                // not — it is the opaque plane colour rows paint behind their
                // content — so this position takes ``hoverWash``.
                if isEnabled, configuration.isPressed || hovering {
                    RoundedRectangle(cornerRadius: RapidTheme.Radius.button, style: .continuous)
                        .fill(configuration.isPressed ? RapidTheme.pressedFill : RapidTheme.hoverWash)
                }
            }
            .overlay {
                if let stroke {
                    RoundedRectangle(cornerRadius: RapidTheme.Radius.button, style: .continuous)
                        .strokeBorder(stroke, lineWidth: 1)
                }
            }
            .opacity(isEnabled ? 1 : RapidTheme.disabledOpacity)
            .contentShape(RoundedRectangle(cornerRadius: RapidTheme.Radius.button, style: .continuous))
            .onHover { hovering = $0 }
            .rapidAnimation(RapidMotion.quick, value: hovering)
            // A 1pt scale nudge on press. Suppressed under Reduce
            // Motion, where the wash alone carries the feedback.
            .scaleEffect(configuration.isPressed && !reduceMotion ? 0.98 : 1.0)
            .rapidAnimation(RapidMotion.quick, value: configuration.isPressed)
        // Focus ring: deliberately NOT overridden here. SwiftUI draws
        // the system ring for a focused Button using the environment
        // tint, and ``RapidApp`` tints every scene root with the brand
        // amber — so keyboard focus already lands on the brand colour.
        // Hand-rolling a ring would mean re-implementing focusability
        // and would change Tab order, which is out of scope for a
        // visual-only phase.
    }
}

// MARK: - Ergonomics

extension ButtonStyle where Self == RapidPrimaryButtonStyle {
    /// The single high-emphasis action on a surface. Amber, graphite label.
    static var rapidPrimary: RapidPrimaryButtonStyle { .init() }
    /// Full-width primary — cards, sheets, onboarding footers. Keeps the
    /// 36pt hero height: a CTA that spans a card is the one place the
    /// extra weight is doing work.
    static var rapidPrimaryWide: RapidPrimaryButtonStyle {
        .init(expands: true, height: RapidTheme.ControlHeight.large)
    }
    /// 28pt filled action for dense rows — pairs with
    /// ``rapidSecondaryCompact`` on the same baseline.
    static var rapidPrimaryCompact: RapidPrimaryButtonStyle {
        .init(height: RapidTheme.ControlHeight.small)
    }
}

extension ButtonStyle where Self == RapidSecondaryButtonStyle {
    /// Outlined neutral action.
    static var rapidSecondary: RapidSecondaryButtonStyle { .init() }
    /// Outlined utility action — neutral at rest, steel blue on hover.
    /// The default for Copy / per-row actions.
    static var rapidSecondaryUtility: RapidSecondaryButtonStyle { .init(utility: true) }
    /// 28pt outlined action for dense rows.
    static var rapidSecondaryCompact: RapidSecondaryButtonStyle {
        .init(height: RapidTheme.ControlHeight.small)
    }
    /// 28pt outlined utility action for dense rows.
    static var rapidSecondaryCompactUtility: RapidSecondaryButtonStyle {
        .init(utility: true, height: RapidTheme.ControlHeight.small)
    }
}

extension ButtonStyle where Self == RapidTertiaryButtonStyle {
    /// Borderless text / icon button. Neutral, steel blue on hover.
    static var rapidTertiary: RapidTertiaryButtonStyle { .init() }
    /// A genuine text link — steel blue at rest.
    static var rapidLink: RapidTertiaryButtonStyle { .init(link: true) }
}

extension ButtonStyle where Self == RapidDestructiveButtonStyle {
    /// Stop / Delete.
    static var rapidDestructive: RapidDestructiveButtonStyle { .init() }
    /// 28pt destructive action for dense rows.
    static var rapidDestructiveCompact: RapidDestructiveButtonStyle {
        .init(height: RapidTheme.ControlHeight.small)
    }
}
