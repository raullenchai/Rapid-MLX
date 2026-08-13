import SwiftUI

/// Shared primitives for the first-run onboarding wizard (#1524).
///
/// These are the reusable pieces the ``QuickstartView`` steps compose:
/// a brand mark, a top bar with step progress, a Back/primary footer, an
/// attribute chip, and the two model-choice cards. The patterns are
/// borrowed from FluidVoice's onboarding framework (a shared footer
/// scaffold + progress affordance + selectable cards) but rebuilt on
/// ``RapidTheme`` tokens — we deliberately did NOT port FluidVoice's
/// whole theme-as-EnvironmentValue system for one wizard.
///
/// Everything here is a thin, host-free ``View`` over ``RapidTheme`` +
/// the #507 components (``BrandIcon`` / ``ModelMeter``), so the wizard
/// can be snapshot-rendered offscreen without a running app.

// MARK: - Brand mark

/// The circular Rapid brand mark — an amber-tint disc with the
/// ``RapidMark`` speed streaks. A pure ``Shape`` (no bundled asset), so
/// it renders in the test bundle unlike the image-backed logo views.
struct OnboardingBrandMark: View {
    var size: CGFloat = 26

    var body: some View {
        ZStack {
            Circle()
                .fill(RapidTheme.brandAmberTint)
                .frame(width: size, height: size)
            RapidMark()
                .fill(RapidTheme.brandAmber)
                .frame(width: size * 0.58, height: size * 0.38)
        }
        .accessibilityHidden(true)
    }
}

// MARK: - Step progress

/// Honest wizard progress. The old three capsules looked like a carousel
/// page control even though onboarding only advances through its explicit
/// buttons (#1792). A labelled linear meter communicates position without
/// advertising swipe, drag, or arbitrary-page navigation.
/// ``current`` is 0-indexed.
///
/// ``total`` defaults to ``QuickstartCoordinator.Step.total`` so the public
/// step count lives in exactly one place. Onboarding V3 (Paper 05.1.G,
/// "Four public steps, and Ready is confirmed") makes that four — Welcome,
/// Choose a model, Download, Start — and no "Step N of 3" language survives
/// anywhere in production.
struct OnboardingStepProgress: View {
    let current: Int
    var total: Int = QuickstartCoordinator.Step.total

    var body: some View {
        VStack(alignment: .trailing, spacing: 5) {
            Text("Step \(current + 1) of \(total)")
                .scaledSystemFont(10, weight: .medium)
                .foregroundStyle(.secondary)
            ProgressView(value: Double(current + 1), total: Double(total))
                .progressViewStyle(.linear)
                .tint(RapidTheme.brand)
                .frame(width: 92)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityIdentifier("Quickstart.Progress")
        // A custom SwiftUI container is AXUnknown on macOS and drops its
        // AXValue. Keep the full status in the label so VoiceOver receives it
        // reliably instead of announcing only "Setup progress".
        .accessibilityLabel("Setup progress, step \(current + 1) of \(total)")
    }
}

// MARK: - Top bar

/// The interior-step header: brand mark + wordmark on the left, step
/// dots on the right. ``step`` is the public macro step this screen
/// belongs to — never a raw ordinal, so a screen can't drift out of the
/// four-step model by miscounting.
struct OnboardingTopBar: View {
    let step: QuickstartCoordinator.Step

    var body: some View {
        HStack(spacing: 10) {
            OnboardingBrandMark(size: 26)
            Text("Rapid-MLX").scaledSystemFont(13, weight: .semibold)
            Spacer()
            OnboardingStepProgress(current: step.rawValue)
        }
    }
}

// MARK: - Footer

/// The shared wizard footer: an optional Back on the left, a prominent
/// amber primary pill on the right. Wires Return → primary and (when
/// Back is present) Escape → back, so every step gets identical keyboard
/// behaviour for free.
struct OnboardingWizardFooter: View {
    let primaryTitle: String
    var primaryEnabled: Bool = true
    var onBack: (() -> Void)?
    let onPrimary: () -> Void

    var body: some View {
        HStack {
            if let onBack {
                Button(action: onBack) {
                    Text("Back")
                        .scaledSystemFont(13, weight: .medium)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .keyboardShortcut(.cancelAction)
                .accessibilityIdentifier("Quickstart.Footer.Back")
            }
            Spacer()
            Button(action: onPrimary) {
                Text(primaryTitle)
                    .scaledSystemFont(13, weight: .semibold)
                    .foregroundStyle(.white)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 9)
                    .background(Capsule().fill(RapidTheme.amber.opacity(primaryEnabled ? 1 : 0.4)))
            }
            .buttonStyle(.plain)
            .disabled(!primaryEnabled)
            .keyboardShortcut(.defaultAction)
            .accessibilityIdentifier("Quickstart.Footer.Primary")
        }
    }
}

// MARK: - Attribute chip

/// A small icon+label pill used on the starter card ("Instant", "Runs
/// on any Mac") in place of benchmark meters — the starter has no
/// published benchmark, so a qualitative chip is honest where a meter
/// would be a dashed blank (decision (b), #1524).
struct OnboardingAttrChip: View {
    let symbol: String
    let text: String
    let foreground: Color
    let background: Color

    var body: some View {
        Label(text, systemImage: symbol)
            .labelStyle(.titleAndIcon)
            .scaledSystemFont(10.5, weight: .medium)
            .foregroundStyle(foreground)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(Capsule().fill(background))
    }
}

// MARK: - Model choice cards

/// The explicit below-quality-floor escape hatch. It looks distinct from
/// both the recommended starter and benchmarked trade-ups, and puts the
/// capability cost on screen so "lower memory" is never read as "same model,
/// only faster".
struct QuickstartLowMemoryCard: View {
    let choice: QuickstartModelChoice
    let selected: Bool
    let sizeText: String
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack(alignment: .top, spacing: 11) {
                BrandIcon(alias: choice.alias, size: 32)
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 7) {
                        Text(choice.displayName).scaledSystemFont(13, weight: .semibold)
                        Text("LOWEST MEMORY")
                            .scaledSystemFont(8.5, weight: .bold)
                            .foregroundStyle(RapidTheme.green)
                            .padding(.horizontal, 6).padding(.vertical, 2)
                            .background(Capsule().fill(RapidTheme.green.opacity(0.12)))
                        Spacer()
                        if !sizeText.isEmpty {
                            Text(sizeText).scaledSystemFont(11).foregroundStyle(.secondary)
                        }
                    }
                    Text(choice.blurb)
                        .scaledSystemFont(11)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                selectionGlyph(selected: selected, size: 18)
            }
            .padding(.horizontal, 14).padding(.vertical, 11)
            .background(cardFill(selected: selected))
            .overlay(cardStroke(selected: selected))
            .contentShape(Rectangle())
        }
        .buttonStyle(.pressableCard)
        .accessibilityIdentifier("Quickstart.Choice.\(choice.alias)")
        .accessibilityAddTraits(selected ? .isSelected : [])
        .accessibilityLabel("\(choice.displayName). Lowest memory. \(choice.blurb) Download \(sizeText)")
    }
}

/// The recommended starter card: brand icon + name + "START HERE" badge
/// + a qualitative blurb + attribute chips (NO meters — the starter
/// card is deliberately qualitative; decision (b)). Selectable; the whole card is the
/// tap target.
struct QuickstartRecommendedCard: View {
    let choice: QuickstartModelChoice
    let selected: Bool
    let sizeText: String
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack(alignment: .top, spacing: 13) {
                BrandIcon(alias: choice.alias, size: 38)
                VStack(alignment: .leading, spacing: 7) {
                    HStack(spacing: 8) {
                        Text(choice.displayName).scaledSystemFont(15, weight: .semibold)
                        if choice.isStarter {
                            Text("START HERE")
                                .scaledSystemFont(9, weight: .bold)
                                .foregroundStyle(RapidTheme.brand)
                                .padding(.horizontal, 6).padding(.vertical, 2)
                                .background(Capsule().fill(RapidTheme.brand.opacity(0.14)))
                        }
                        Spacer()
                        if !sizeText.isEmpty {
                            Text(sizeText).scaledSystemFont(12).foregroundStyle(.secondary)
                        }
                    }
                    Text(choice.blurb)
                        .scaledSystemFont(12)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    HStack(spacing: 7) {
                        OnboardingAttrChip(symbol: "bolt.fill", text: "Instant",
                                           foreground: RapidTheme.amberDeep, background: RapidTheme.amberTint)
                        OnboardingAttrChip(symbol: "checkmark.seal.fill", text: "Runs on any Mac",
                                           foreground: RapidTheme.green, background: RapidTheme.green.opacity(0.12))
                    }
                    .padding(.top, 1)
                }
                selectionGlyph(selected: selected, size: 20)
            }
            .padding(14)
            .background(cardFill(selected: selected))
            .overlay(cardStroke(selected: selected))
            .contentShape(Rectangle())
        }
        .buttonStyle(.pressableCard)
        .accessibilityIdentifier("Quickstart.Choice.\(choice.alias)")
        .accessibilityAddTraits(selected ? .isSelected : [])
        .accessibilityLabel(accessibilityText)
    }

    /// Fold the "recommended starter" framing, size, and the attribute
    /// chips into the spoken label — the button overrides its children,
    /// so VoiceOver users otherwise hear only the name + blurb.
    private var accessibilityText: String {
        var parts = [choice.displayName]
        if choice.isStarter { parts.append("recommended starter") }
        parts.append(choice.blurb)
        if !sizeText.isEmpty { parts.append("download \(sizeText)") }
        parts.append("instant, runs on any Mac")
        return parts.joined(separator: ". ")
    }
}

/// A compact bigger-option row: brand icon + name + blurb, inline
/// Accuracy·Speed numbers (real benchmark values via ``ModelMeter``),
/// size, and a selection glyph. Selectable.
struct QuickstartCompactCard: View {
    let choice: QuickstartModelChoice
    let selected: Bool
    let sizeText: String
    var isCached: Bool = false
    let onTap: () -> Void

    private var accValue: String { ModelMeter.qualityMeter(for: choice.alias).formattedValue }
    private var spdValue: String { ModelMeter.speedMeter(for: choice.alias).formattedValue }

    var body: some View {
        // Prefer the elegant single row; when the detail pane is too narrow
        // for the full name beside the fixed meter/size columns (~380pt at
        // the 640pt window floor), fall back to a two-row layout so the
        // model name — the key identifier (4B vs 9B) — is NEVER truncated
        // (memory #459/#464). ViewThatFits picks the wide row only while its
        // full-name ideal width fits.
        Button(action: onTap) {
            ViewThatFits(in: .horizontal) {
                wideRow
                narrowRow
            }
            .padding(.horizontal, 14).padding(.vertical, 11)
            .background(cardFill(selected: selected))
            .overlay(cardStroke(selected: selected))
            .contentShape(Rectangle())
        }
        .buttonStyle(.pressableCard)
        .accessibilityIdentifier("Quickstart.Choice.\(choice.alias)")
        .accessibilityAddTraits(selected ? .isSelected : [])
        .accessibilityLabel(accessibilityText)
    }

    /// Wide layout: icon + name, then Accuracy·Speed meters, size, glyph —
    /// all on one row. Measurement is driven by the name's full ideal
    /// width (the trade-up cards carry no blurb — it's the recommended
    /// starter card that gets prose), so ViewThatFits drops to
    /// ``narrowRow`` exactly when the name would otherwise be squeezed.
    private var wideRow: some View {
        HStack(spacing: 10) {
            BrandIcon(alias: choice.alias, size: 30)
            Text(choice.displayName).scaledSystemFont(13, weight: .medium).lineLimit(1)
            Spacer(minLength: 8)
            metric("Accuracy", accValue)
            metric("Speed", spdValue)
            sizeLabel
            selectionGlyph(selected: selected, size: 18)
        }
    }

    /// Narrow fallback: name (never truncated) + size on the first line,
    /// the meters on a second line — so the 4B/9B identifier always reads.
    private var narrowRow: some View {
        HStack(spacing: 10) {
            BrandIcon(alias: choice.alias, size: 30)
            VStack(alignment: .leading, spacing: 5) {
                HStack(spacing: 6) {
                    Text(choice.displayName).scaledSystemFont(13, weight: .medium).lineLimit(1).fixedSize()
                    Spacer(minLength: 6)
                    sizeLabel
                }
                HStack(spacing: 14) {
                    inlineMetric("Accuracy", accValue)
                    inlineMetric("Speed", spdValue)
                    Spacer(minLength: 0)
                }
            }
            selectionGlyph(selected: selected, size: 18)
        }
    }

    @ViewBuilder private var sizeLabel: some View {
        if !sizeText.isEmpty {
            Text(sizeText).scaledSystemFont(12).foregroundStyle(.secondary).fixedSize()
        }
    }

    /// Fold the accuracy / speed / size read-outs into the spoken label —
    /// the button's own label overrides the child ``Text``s, so without
    /// this VoiceOver users lose the exact numbers sighted users compare.
    private var accessibilityText: String {
        var parts = ["\(choice.displayName). \(choice.blurb)"]
        parts.append("Accuracy \(accValue), speed \(spdValue)")
        if isCached {
            parts.append(sizeText.isEmpty ? "on disk" : "on disk \(sizeText)")
        } else if !sizeText.isEmpty {
            parts.append("download \(sizeText)")
        }
        return parts.joined(separator: ". ")
    }

    // #546: the metric VALUE keeps its Font-level `.monospacedDigit()`
    // (so the Accuracy/Speed numbers stay column-aligned), which
    // `.scaledSystemFont` can't carry — so it scales via a local
    // `@ScaledMetric` instead. 12/11pt at the default size, unchanged.
    @ScaledMetric(relativeTo: .body) private var metricValueSize: CGFloat = 12
    @ScaledMetric(relativeTo: .body) private var inlineMetricValueSize: CGFloat = 11

    private func metric(_ label: String, _ value: String) -> some View {
        VStack(spacing: 1) {
            Text(value).font(.system(size: metricValueSize, weight: .semibold).monospacedDigit())
            Text(label).scaledSystemFont(9).foregroundStyle(.tertiary)
        }
        .fixedSize()
    }

    private func inlineMetric(_ label: String, _ value: String) -> some View {
        HStack(spacing: 3) {
            Text(label).scaledSystemFont(9).foregroundStyle(.tertiary)
            Text(value).font(.system(size: inlineMetricValueSize, weight: .semibold).monospacedDigit())
        }
        .fixedSize()
    }
}

// MARK: - Shared card chrome

private func selectionGlyph(selected: Bool, size: CGFloat) -> some View {
    Image(systemName: selected ? "checkmark.circle.fill" : "circle")
        .font(.system(size: size))
        .foregroundStyle(selected ? RapidTheme.brand : Color.secondary.opacity(0.4))
}

private func cardFill(selected: Bool) -> some View {
    RoundedRectangle(cornerRadius: RapidTheme.cardRadius, style: .continuous)
        .fill(selected ? RapidTheme.brandTint : RapidTheme.card)
}

private func cardStroke(selected: Bool) -> some View {
    RoundedRectangle(cornerRadius: RapidTheme.cardRadius, style: .continuous)
        .stroke(selected ? RapidTheme.brand.opacity(0.5) : RapidTheme.hairline,
                lineWidth: selected ? 1.5 : 1)
}
