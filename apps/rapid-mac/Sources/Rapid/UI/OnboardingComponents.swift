import SwiftUI

/// The selectable model rows of first-run setup, in the Direction D visual
/// language (Paper 05.1.B state 04, 05.2.C, 05.2.D).
///
/// ## What changed, and what did not
///
/// The previous versions of these cards carried the wizard's old centred-card
/// styling: benchmark meters, a `ViewThatFits` two-row fallback, and a shared
/// footer scaffold. Direction D replaces the composition — a fixed heading
/// column beside a list of quiet cards, with exactly one amber moment on the
/// screen — but not the contract. Selection is still a single click that never
/// navigates, double-click is still a shortcut for the visible primary, and a
/// row that cannot run on this Mac is still inert rather than merely dimmed.
///
/// Every card is a pure ``View`` over ``RapidTheme`` and ``OnboardingD``, so a
/// screen can be composed without a coordinator and rendered without an app.

// MARK: - Row activation

extension View {
    /// Wire the double-click half of the activation contract onto a selectable
    /// model row.
    ///
    /// Paper 05.2.G — "One action, three inputs". A single click selects and
    /// never navigates. A double-click performs whatever the visible footer
    /// primary currently says — Review download on an uncached pick, Start
    /// existing model on a cached one — and does nothing at all when that
    /// primary is disabled.
    ///
    /// A *simultaneous* gesture, deliberately: the row's own Button still
    /// fires on the first click, so a double-click selects and then activates.
    /// That ordering is what keeps this a shortcut for the primary rather than
    /// a second hidden route with rules of its own — the reading Paper 05.2.J
    /// · S6 supersedes, where double-click always opened Review even for a
    /// model already on disk.
    @ViewBuilder
    func modelRowActivation(_ onActivate: (() -> Void)?) -> some View {
        if let onActivate {
            simultaneousGesture(TapGesture(count: 2).onEnded { onActivate() })
        } else {
            self
        }
    }
}

// MARK: - Model monogram

/// The flat identity tile Direction D uses for a model.
///
/// Reuses ``ModelBrandStyle/monogram(forAlias:)`` so the letter is the same one
/// the rest of the app shows for that family, but paints it flat rather than
/// with ``BrandIcon``'s per-brand gradient. Setup is the one surface where the
/// rule is a single strong colour moment per screen — eleven saturated brand
/// gradients down a list would spend that budget before the primary action got
/// any of it. Settings → Models keeps ``BrandIcon`` and is untouched.
struct OnboardingModelTile: View {
    enum Tone {
        /// The recommended starter — the one identity that gets a colour.
        case feature
        /// Everything else.
        case neutral
        /// A model this Mac cannot run.
        case muted
    }

    let alias: String
    var size: CGFloat = 32
    var tone: Tone = .neutral

    private var monogram: String { ModelBrandStyle.monogram(forAlias: alias) }

    var body: some View {
        RoundedRectangle(cornerRadius: size * 0.25, style: .continuous)
            .fill(background)
            .frame(width: size, height: size)
            .overlay {
                Text(monogram)
                    .scaledSystemFont(size * 0.42, relativeTo: .body, weight: .semibold)
                    .foregroundStyle(foreground)
                    .minimumScaleFactor(0.6)
                    .lineLimit(1)
            }
            .accessibilityHidden(true)
    }

    private var background: Color {
        switch tone {
        case .feature: return RapidTheme.brandSecondaryTint
        case .neutral, .muted: return RapidTheme.surfaceCode
        }
    }

    private var foreground: Color {
        switch tone {
        case .feature: return RapidTheme.brandSecondary
        case .neutral: return RapidTheme.textSecondary
        case .muted: return RapidTheme.textTertiary
        }
    }
}

// MARK: - Card chrome

/// The shared card shell. Selection is a 2pt amber border rather than a fill,
/// and the horizontal padding drops by the same point so the content does not
/// shift when a row is picked.
private struct OnboardingCardChrome: ViewModifier {
    let selected: Bool
    var fill: Color = RapidTheme.surfaceRaised
    var verticalPadding: CGFloat
    var horizontalPadding: CGFloat = 18

    func body(content: Content) -> some View {
        content
            .padding(.vertical, verticalPadding)
            .padding(.horizontal, selected ? horizontalPadding - 1 : horizontalPadding)
            .background(
                RoundedRectangle(cornerRadius: OnboardingD.cardRadius, style: .continuous)
                    .fill(fill)
            )
            .overlay(
                RoundedRectangle(cornerRadius: OnboardingD.cardRadius, style: .continuous)
                    .strokeBorder(
                        selected ? RapidTheme.brandPrimary : RapidTheme.hairline,
                        lineWidth: selected ? 2 : 1
                    )
            )
            .contentShape(RoundedRectangle(cornerRadius: OnboardingD.cardRadius, style: .continuous))
    }
}

private extension View {
    func onboardingCard(
        selected: Bool,
        fill: Color = RapidTheme.surfaceRaised,
        verticalPadding: CGFloat,
        horizontalPadding: CGFloat = 18
    ) -> some View {
        modifier(OnboardingCardChrome(
            selected: selected,
            fill: fill,
            verticalPadding: verticalPadding,
            horizontalPadding: horizontalPadding
        ))
    }
}

// MARK: - Recommended starter

/// The starter card. The only row that carries prose, attribute pills and a
/// coloured monogram — it is the recommendation, and the composition says so
/// without a meter or a benchmark claim.
struct QuickstartRecommendedCard: View {
    let choice: QuickstartModelChoice
    let selected: Bool
    let sizeText: String
    var onActivate: (() -> Void)? = nil
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack(alignment: .top, spacing: 14) {
                OnboardingModelTile(alias: choice.alias, size: 38, tone: .feature)
                VStack(alignment: .leading, spacing: 7) {
                    HStack(spacing: 9) {
                        Text(choice.displayName)
                            .scaledSystemFont(15, weight: .semibold)
                            .foregroundStyle(RapidTheme.textPrimary)
                        if choice.isStarter {
                            OnboardingBadge(text: "START HERE", tone: .ink)
                        }
                        Spacer(minLength: 8)
                        if !sizeText.isEmpty {
                            Text(sizeText)
                                .scaledSystemFont(12, design: .monospaced)
                                .foregroundStyle(RapidTheme.textSecondary)
                                .fixedSize()
                        }
                    }
                    Text(choice.blurb)
                        .scaledSystemFont(13)
                        .foregroundStyle(RapidTheme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                    HStack(spacing: 7) {
                        OnboardingAttributePill(text: "Instant")
                        OnboardingAttributePill(text: "Runs on any Mac")
                    }
                    .padding(.top, 2)
                }
                OnboardingSelectionGlyph(isSelected: selected)
            }
            .onboardingCard(selected: selected, verticalPadding: 18)
        }
        .buttonStyle(.plain)
        .modelRowActivation(onActivate)
        .accessibilityIdentifier("Quickstart.Choice.\(choice.alias)")
        .accessibilityAddTraits(selected ? .isSelected : [])
        .accessibilityLabel(accessibilityText)
    }

    /// Fold the framing, size and attribute pills into the spoken label — the
    /// button overrides its children, so VoiceOver otherwise hears the name
    /// and blurb only.
    private var accessibilityText: String {
        var parts = [choice.displayName]
        if choice.isStarter { parts.append("recommended starter") }
        parts.append(choice.blurb)
        if !sizeText.isEmpty { parts.append("download \(sizeText)") }
        parts.append("instant, runs on any Mac")
        return parts.joined(separator: ". ")
    }
}

// MARK: - Low-memory fallback

/// The explicit below-quality-floor escape hatch. It names the capability cost
/// so "lowest memory" is never read as "same model, only faster".
struct QuickstartLowMemoryCard: View {
    let choice: QuickstartModelChoice
    let selected: Bool
    let sizeText: String
    var onActivate: (() -> Void)? = nil
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack(alignment: .top, spacing: 14) {
                OnboardingModelTile(alias: choice.alias, size: 32)
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 9) {
                        Text(choice.displayName)
                            .scaledSystemFont(14, weight: .semibold)
                            .foregroundStyle(RapidTheme.textPrimary)
                        OnboardingBadge(text: "LOWEST MEMORY", tone: .ready)
                        Spacer(minLength: 8)
                        if !sizeText.isEmpty {
                            Text(sizeText)
                                .scaledSystemFont(12, design: .monospaced)
                                .foregroundStyle(RapidTheme.textSecondary)
                                .fixedSize()
                        }
                    }
                    Text(choice.blurb)
                        .scaledSystemFont(12)
                        .foregroundStyle(RapidTheme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                OnboardingSelectionGlyph(isSelected: selected)
            }
            .onboardingCard(selected: selected, verticalPadding: 15)
        }
        .buttonStyle(.plain)
        .modelRowActivation(onActivate)
        .accessibilityIdentifier("Quickstart.Choice.\(choice.alias)")
        .accessibilityAddTraits(selected ? .isSelected : [])
        .accessibilityLabel(
            "\(choice.displayName). Lowest memory. \(choice.blurb)"
                + (sizeText.isEmpty ? "" : " Download \(sizeText)")
        )
    }
}

// MARK: - Compact trade-up / cached row

/// A one-line choice: name, one-line blurb, size, selection. Used for the
/// bigger trade-ups and for models already on this Mac.
struct QuickstartCompactCard: View {
    let choice: QuickstartModelChoice
    let selected: Bool
    let sizeText: String
    var isCached: Bool = false
    var onActivate: (() -> Void)? = nil
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 14) {
                OnboardingModelTile(alias: choice.alias, size: 28)
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 9) {
                        Text(choice.displayName)
                            .scaledSystemFont(14, weight: .semibold)
                            .foregroundStyle(RapidTheme.textPrimary)
                            .lineLimit(1)
                        if isCached {
                            OnboardingBadge(text: "ON THIS MAC", tone: .ready)
                        }
                    }
                    if !choice.blurb.isEmpty {
                        Text(choice.blurb)
                            .scaledSystemFont(12)
                            .foregroundStyle(RapidTheme.textSecondary)
                            .lineLimit(2)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                Spacer(minLength: 8)
                if !sizeText.isEmpty {
                    Text(sizeText)
                        .scaledSystemFont(12, design: .monospaced)
                        .foregroundStyle(RapidTheme.textSecondary)
                        .fixedSize()
                }
                OnboardingSelectionGlyph(isSelected: selected)
            }
            .onboardingCard(selected: selected, verticalPadding: 13)
        }
        .buttonStyle(.plain)
        .modelRowActivation(onActivate)
        .accessibilityIdentifier("Quickstart.Choice.\(choice.alias)")
        .accessibilityAddTraits(selected ? .isSelected : [])
        .accessibilityLabel(accessibilityText)
    }

    private var accessibilityText: String {
        var parts = [choice.displayName]
        if !choice.blurb.isEmpty { parts.append(choice.blurb) }
        if isCached {
            parts.append(sizeText.isEmpty ? "on disk" : "on disk \(sizeText)")
        } else if !sizeText.isEmpty {
            parts.append("download \(sizeText)")
        }
        return parts.joined(separator: ". ")
    }
}

// MARK: - Catalogue row

/// One row of in-window Browse all models (Paper 05.2.C).
///
/// Fixed slots, not gaps: the badge lane and the size lane are pinned widths so
/// they form vertical columns down a list of 175 rows whose names vary wildly
/// in length. Without them the sizes wander and the list stops being scannable.
struct OnboardingCatalogRow: View {
    let alias: String
    /// Hugging Face repo, shown under the alias. Falls back to the memory
    /// explanation when the row cannot run here.
    let subtitle: String
    let sizeText: String
    let selected: Bool
    /// False when ``ModelSizing`` says this will not run on this Mac.
    ///
    /// Drives the muted treatment — dimmed monogram, tertiary text, canvas
    /// fill instead of raised, hollow selection glyph — and nothing else. The
    /// row stays a live control: Paper 05.2.D allows opening its detail, and
    /// the refusal happens on the primary of the screen that opens, not by
    /// making the row itself unclickable. See ``OnboardingModelSelection``.
    let isAvailable: Bool
    let badges: [OnboardingCatalogRow.Badge]
    var onActivate: (() -> Void)? = nil
    let onTap: () -> Void

    struct Badge: Identifiable {
        let id = UUID()
        let text: String
        let tone: OnboardingBadge.Tone
    }

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 14) {
                OnboardingModelTile(
                    alias: alias,
                    size: 32,
                    tone: isAvailable ? .neutral : .muted
                )
                VStack(alignment: .leading, spacing: 2) {
                    Text(alias)
                        .scaledSystemFont(14, weight: .semibold)
                        .foregroundStyle(isAvailable ? RapidTheme.textPrimary : RapidTheme.textTertiary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Text(subtitle)
                        .scaledSystemFont(11, design: .monospaced)
                        .foregroundStyle(RapidTheme.textTertiary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer(minLength: 8)
                HStack(spacing: 6) {
                    Spacer(minLength: 0)
                    ForEach(badges) { badge in
                        OnboardingBadge(text: badge.text, tone: badge.tone)
                    }
                }
                .frame(width: OnboardingD.rowBadgeSlot, alignment: .trailing)
                Text(sizeText)
                    .scaledSystemFont(12, design: .monospaced)
                    .foregroundStyle(isAvailable ? RapidTheme.textSecondary : RapidTheme.textTertiary)
                    .frame(width: OnboardingD.rowSizeSlot, alignment: .trailing)
                OnboardingSelectionGlyph(isSelected: selected, isEnabled: isAvailable)
            }
            .padding(.horizontal, selected ? 17 : 18)
            .frame(height: OnboardingD.rowHeight)
            .background(
                RoundedRectangle(cornerRadius: OnboardingD.cardRadius, style: .continuous)
                    .fill(isAvailable ? RapidTheme.surfaceRaised : RapidTheme.surfaceCanvas)
            )
            .overlay(
                RoundedRectangle(cornerRadius: OnboardingD.cardRadius, style: .continuous)
                    .strokeBorder(
                        selected ? RapidTheme.brandPrimary : RapidTheme.hairline,
                        lineWidth: selected ? 2 : 1
                    )
            )
            .contentShape(RoundedRectangle(cornerRadius: OnboardingD.cardRadius, style: .continuous))
        }
        .buttonStyle(.plain)
        .modelRowActivation(onActivate)
        .accessibilityIdentifier("Quickstart.CatalogRow.\(alias)")
        .accessibilityAddTraits(selected ? .isSelected : [])
        .accessibilityLabel(accessibilityText)
        .accessibilityHint(accessibilityHint)
    }

    private var accessibilityText: String {
        var parts = [alias, subtitle]
        parts.append(contentsOf: badges.map(\.text.localizedLowercase))
        if !sizeText.isEmpty { parts.append(sizeText) }
        return parts.joined(separator: ". ")
    }

    /// Spoken only for the row that cannot run, because only there is the
    /// honest answer the surprising one: the row IS live, and nothing but an
    /// explanation is behind it. A sighted user infers that from the dimming
    /// plus the badge; the hint is how the same inference reaches VoiceOver,
    /// which is told the label and the traits and would otherwise have to
    /// guess from "won't fit" whether the row does anything at all.
    ///
    /// Runnable rows get none. Their behaviour is the norm the rest of the
    /// list establishes, and the one sentence that would cover them both is
    /// not true of either: a cached row does not open Review, it starts.
    private var accessibilityHint: String {
        isAvailable
            ? ""
            : "Cannot run on this Mac. Opens a read-only explanation — nothing will be downloaded."
    }
}
