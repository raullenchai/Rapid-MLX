import SwiftUI

// MARK: - Relay status bar

/// The bar across the top of the Share workbench.
///
/// Replaces the old "community strip", which showed contributing-Mac counts,
/// requests waiting, and a highest-demand model. None of those three exist:
/// QuickSilver publishes availability, never demand, and there is no queue to
/// have a depth. What belongs here instead is the one fact the user needs
/// before pressing the button — whether this Mac is ready to connect — plus the
/// standing clarification that traffic is live, not queued work to claim.
///
/// Drawn on the band's own darker header (Paper `#17191D` over `#22252B`) so it
/// reads as part of the workbench rather than a light shelf above it.
struct ShareComputeRelayStatusBar: View {
    /// What the relay side of the connection is doing right now.
    enum Status: Equatable {
        case readyToConnect
        case connecting
        case online
        case reconnecting
        case failed

        var title: String {
            switch self {
            case .readyToConnect: return String(localized: "Ready to connect this Mac")
            case .connecting: return String(localized: "Connecting this Mac…")
            case .online: return String(localized: "Serving live requests")
            case .reconnecting: return String(localized: "Reconnecting to the relay…")
            case .failed: return String(localized: "Not connected")
            }
        }

        /// Semantic, never decorative: green only when the relay is actually
        /// usable, amber while in flight, red when it is not.
        var dot: Color {
            switch self {
            case .readyToConnect, .online: return RapidTheme.bandReady
            case .connecting, .reconnecting: return RapidTheme.brandPrimary
            case .failed: return RapidTheme.bandDestructive
            }
        }

        var isInFlight: Bool { self == .connecting || self == .reconnecting }
    }

    let status: Status
    var isNarrow = false

    var body: some View {
        HStack(spacing: RapidTheme.Space.md) {
            HStack(spacing: 14) {
                Group {
                    if status.isInFlight {
                        ProgressView().controlSize(.small)
                    } else {
                        Circle().fill(status.dot).frame(width: 9, height: 9)
                    }
                }
                .frame(width: 9)
                .accessibilityHidden(true)

                VStack(alignment: .leading, spacing: 2) {
                    ShareComputeEyebrow(
                        text: "QuickSilver relay",
                        tone: RapidTheme.bandInkSecondary,
                        size: 11
                    )
                    Text(status.title)
                        .font(.system(size: 15, weight: .medium))
                        .foregroundStyle(RapidTheme.bandInk)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .accessibilityElement(children: .combine)

            Spacer(minLength: RapidTheme.Space.md)

            // The load-bearing clarification. Kept on the bar rather than
            // buried in body copy because "is there a queue of tasks I have to
            // accept?" is the first question this surface gets asked.
            if !isNarrow {
                Text("Live requests · no task queue")
                    .font(.system(size: 12))
                    .foregroundStyle(RapidTheme.bandInkSecondary)
                    .fixedSize()
            }
        }
        .frame(minHeight: 64)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, isNarrow ? 20 : 32)
        .background(RapidTheme.surfaceBandHeader)
        .overlay(alignment: .bottom) {
            Rectangle().fill(RapidTheme.bandHairlineStrong).frame(height: 1)
        }
        .accessibilityIdentifier("ShareCompute.RelayStatus")
    }
}

// MARK: - Share tab

/// "What can I serve right now?"
///
/// One continuous workbench: the relay status bar, the headline and its
/// description, the three connected steps, the model selector, the primary
/// action, and the standing facts — in that reading order, inside a single
/// frame.
///
/// The description sits directly under the headline rather than in a right-hand
/// column. That is not a styling preference: the headline makes a promise
/// ("earn API credits") whose qualification ("live forwarding, no queue") has to
/// be read immediately after it, and a parallel column is read second or not at
/// all.
struct ShareComputeShareTab: View {
    let models: [ShareComputeLocalModel]
    let selected: ShareComputeLocalModel?
    let relayStatus: ShareComputeRelayStatusBar.Status
    let catalogLoaded: Bool
    let isNarrow: Bool
    let requiresConnection: Bool
    @Binding var isPickerOpen: Bool
    let onSelect: (ShareComputeLocalModel) -> Void
    let onStart: () -> Void
    let onOpenPool: () -> Void

    /// Measured width of the selector row, so the picker menu can match it.
    @State private var selectorWidth: CGFloat = 0

    private var readyCount: Int { models.count }

    var body: some View {
        ShareComputeWorkbenchFrame {
            VStack(spacing: 0) {
                ShareComputeRelayStatusBar(status: relayStatus, isNarrow: isNarrow)
                band
            }
        }
    }

    private var band: some View {
        VStack(alignment: .leading, spacing: 18) {
            headline
            Divider().overlay(RapidTheme.bandHairline)
            ShareComputeValuePath(isNarrow: isNarrow)
            Divider().overlay(RapidTheme.bandHairline)
            selector
            assurances
        }
        .padding(.horizontal, isNarrow ? 20 : 32)
        .padding(.top, 28)
        .padding(.bottom, 24)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RapidTheme.surfaceBand)
    }

    // MARK: Headline

    private var headline: some View {
        HStack(alignment: .top, spacing: RapidTheme.Space.lg) {
            VStack(alignment: .leading, spacing: 8) {
                // Paper sets the promise on ONE line, split by colour: the
                // instruction in band ink, the payoff in amber. Two Texts
                // rather than one attributed string so the split survives the
                // wrap to two lines at narrow width.
                headlineText
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel(
                        String(localized: "Serve live requests. Earn API credits.")
                    )

                // Directly under the headline. Says, in one sentence, both what
                // happens (forwarding while online) and what does NOT exist (a
                // queue, a claim step).
                Text("QuickSilver forwards requests to this Mac while it is online. No task queue or manual acceptance.")
                    .font(.system(size: isNarrow ? 13 : 14))
                    .foregroundStyle(RapidTheme.bandInkSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
            // The mascot keeps its full rendered box (the plate is ~50%
            // transparent margin) so nothing clips its tail or motion trail,
            // and it sits opposite the headline rather than over any status
            // text.
            if !isNarrow {
                CommunityMascot(
                    context: .readyInvitation,
                    isAnimating: true,
                    visibleSize: 43
                )
                .frame(width: 92, alignment: .trailing)
            }
        }
    }

    /// Headline, laid out so the colour split never costs a line break.
    ///
    /// Desktop keeps both halves on one line; narrow stacks them. Concatenating
    /// with `+` on desktop lets the two colours share a single line box, which
    /// `HStack` would not guarantee once tracking is applied.
    @ViewBuilder
    private var headlineText: some View {
        let size: CGFloat = isNarrow ? 23 : 29
        let tracking: CGFloat = isNarrow ? -0.4 : -0.58
        if isNarrow {
            VStack(alignment: .leading, spacing: 0) {
                Text("Serve live requests.")
                    .font(.system(size: size, weight: .bold))
                    .tracking(tracking)
                    .foregroundStyle(RapidTheme.bandInk)
                Text("Earn API credits.")
                    .font(.system(size: size, weight: .bold))
                    .tracking(tracking)
                    .foregroundStyle(RapidTheme.brandPrimary)
            }
        } else {
            (
                Text("Serve live requests. ")
                    .font(.system(size: size, weight: .bold))
                    .foregroundColor(RapidTheme.bandInk)
                + Text("Earn API credits.")
                    .font(.system(size: size, weight: .bold))
                    .foregroundColor(RapidTheme.brandPrimary)
            )
            .tracking(tracking)
        }
    }

    // MARK: Selector + action

    private var selector: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 9) {
                // Paper: 11pt on the desktop board, 10pt narrow.
                ShareComputeEyebrow(
                    text: "Model to serve",
                    tone: RapidTheme.bandInkSecondary,
                    size: isNarrow ? 10 : 11
                )
                if catalogLoaded {
                    ShareComputeEyebrow(
                        text: readyCount == 1
                            ? String(localized: "· 1 local model ready")
                            : String(
                                format: String(localized: "· %1$d local models ready"),
                                readyCount
                            ),
                        tone: RapidTheme.bandReady,
                        size: isNarrow ? 10 : 11
                    )
                }
            }

            // Narrow stacks the selector above the action; desktop puts them
            // side by side so the action stays visually attached to the model
            // it will share.
            if isNarrow {
                VStack(spacing: 10) {
                    selectorRow
                    primaryAction
                }
            } else {
                HStack(spacing: 10) {
                    selectorRow
                    primaryAction.frame(width: 300)
                }
            }
        }
    }

    @ViewBuilder
    private var selectorRow: some View {
        if let selected {
            ShareComputeModelSelectorRow(
                model: selected,
                isOpen: isPickerOpen,
                isNarrow: isNarrow,
                action: { isPickerOpen.toggle() }
            )
            // The menu is drawn edge-to-edge with this row, so the row has to
            // report how wide it ended up being.
            .background {
                GeometryReader { proxy in
                    Color.clear
                        .onAppear { selectorWidth = proxy.size.width }
                        .onChange(of: proxy.size.width) { _, width in
                            selectorWidth = width
                        }
                }
                .accessibilityHidden(true)
            }
            .shareComputePicker(
                isPresented: $isPickerOpen,
                models: models,
                selectedID: selected.id,
                width: selectorWidth,
                onSelect: onSelect
            )
        } else {
            ShareComputeNoLocalModelsRow(
                catalogLoaded: catalogLoaded,
                onOpenPool: onOpenPool
            )
        }
    }

    @ViewBuilder
    private var primaryAction: some View {
        if selected == nil {
            ShareComputePrimaryBandButton(
                // NOT "see what the pool needs" — the pool publishes what is
                // online, never what it wants.
                title: String(localized: "See what is online"),
                height: isNarrow ? 48 : 72,
                action: onOpenPool
            )
            .accessibilityIdentifier("ShareCompute.OpenPool")
        } else {
            ShareComputePrimaryBandButton(
                // Both paths open the same review sheet. The label differs
                // because connecting a provider key is a materially different
                // ask from re-joining a pool this Mac is already registered
                // with, and the button has to say which one is about to
                // happen.
                title: requiresConnection
                    ? String(localized: "Connect & serve")
                    : String(localized: "Review & connect"),
                height: isNarrow ? 48 : 72,
                action: onStart
            )
            .accessibilityIdentifier("ShareCompute.Start")
        }
    }

    // MARK: Assurances

    private var assurances: some View {
        AdaptiveAssuranceRow(isNarrow: isNarrow) {
            HStack(spacing: 22) {
                assurance(String(localized: "One model serves at a time"))
                assurance(String(localized: "Your current model returns after Stop"))
            }
        } trailing: {
            HStack(spacing: 10) {
                // What is earned, and on what basis. No throughput estimate and
                // no cap figure: the summary endpoint publishes neither, and a
                // number invented here would read as a commitment.
                assurance(String(localized: "Credits are calculated monthly from metered tokens"))
                Link(destination: ShareComputeDestination.credits) {
                    HStack(spacing: 4) {
                        Text("QuickSilver API credits")
                            .font(.system(size: isNarrow ? 11 : 12, weight: .bold))
                        Image(systemName: "arrow.up.right")
                            .font(.system(size: isNarrow ? 9 : 10, weight: .bold))
                            .accessibilityHidden(true)
                    }
                    .foregroundStyle(RapidTheme.brandPrimary)
                }
                .buttonStyle(.plain)
                .accessibilityLabel(
                    String(localized: "Credits are QuickSilver API credits, not cash. Open the QuickSilver dashboard.")
                )
            }
        }
    }

    private func assurance(_ text: String) -> some View {
        Text(text)
            // Paper: 12pt desktop, 11pt narrow, regular weight in both.
            .font(.system(size: isNarrow ? 11 : 12, weight: .regular))
            .foregroundStyle(RapidTheme.bandInkSecondary)
            .fixedSize(horizontal: false, vertical: true)
    }
}

/// Splits the assurance row onto two lines at narrow width instead of letting
/// either half truncate. The handoff forbids hiding ownership or restoration
/// information to save space, so the row grows downward rather than clipping.
private struct AdaptiveAssuranceRow<Leading: View, Trailing: View>: View {
    let isNarrow: Bool
    @ViewBuilder var leading: Leading
    @ViewBuilder var trailing: Trailing

    var body: some View {
        if isNarrow {
            VStack(alignment: .leading, spacing: 8) {
                leading
                trailing
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        } else {
            HStack(alignment: .center, spacing: RapidTheme.Space.md) {
                leading
                Spacer(minLength: RapidTheme.Space.md)
                trailing
            }
        }
    }
}

// MARK: - Selector rows

/// The selected-model row: one control, entirely clickable.
///
/// Name, size, readiness, and demand all live inside this single button, which
/// is what the handoff means by keeping related information together — the
/// user should never have to look at two places to know what they are about to
/// share.
struct ShareComputeModelSelectorRow: View {
    let model: ShareComputeLocalModel
    let isOpen: Bool
    var isNarrow = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: RapidTheme.Space.md) {
                VStack(alignment: .leading, spacing: 6) {
                    // Desktop fits name, size, and readiness on one line.
                    // Narrow drops size and readiness to the line below rather
                    // than truncating the model name, which is the one value
                    // the user has to be able to read in full before sharing.
                    if isNarrow {
                        Text(model.title)
                            .font(.system(size: 17, weight: .bold))
                            .foregroundStyle(RapidTheme.bandInk)
                            .lineLimit(1)
                            .minimumScaleFactor(0.75)
                        HStack(spacing: 10) {
                            if let size = model.onDiskSize {
                                Text(size)
                                    .font(.system(size: 12))
                                    .foregroundStyle(RapidTheme.bandInkSecondary)
                            }
                            Text("Ready on this Mac")
                                .font(.system(size: 11, weight: .bold))
                                .foregroundStyle(RapidTheme.bandReady)
                        }
                    } else {
                        // Name and size on the first line; readiness moves to
                        // ``secondLine`` so the row does not say "Ready on this
                        // Mac" twice.
                        HStack(alignment: .firstTextBaseline, spacing: 14) {
                            Text(model.title)
                                .font(.system(size: 20, weight: .semibold))
                                .foregroundStyle(RapidTheme.bandInk)
                                .lineLimit(1)
                                .minimumScaleFactor(0.8)
                            if let size = model.onDiskSize {
                                Text(size)
                                    .font(.system(size: 13))
                                    .foregroundStyle(RapidTheme.bandInkSecondary)
                                    .fixedSize()
                            }
                        }
                        secondLine
                    }
                }
                Spacer(minLength: 0)
                Image(systemName: "chevron.down")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(RapidTheme.brandPrimary)
                    .rotationEffect(.degrees(isOpen ? 180 : 0))
                    .frame(width: 28)
                    .accessibilityHidden(true)
            }
            .padding(.leading, 18)
            .padding(.trailing, 16)
            .padding(.vertical, 12)
            .frame(maxWidth: .infinity, minHeight: 72, alignment: .leading)
            .background(
                RapidTheme.bandControl,
                in: RoundedRectangle(cornerRadius: RapidTheme.Radius.button)
            )
            .overlay {
                RoundedRectangle(cornerRadius: RapidTheme.Radius.button)
                    .strokeBorder(RapidTheme.bandControlStroke, lineWidth: 1)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(
            String(
                format: String(localized: "Model to share: %1$@, ready on this Mac"),
                model.title
            )
        )
        .accessibilityHint(String(localized: "Choose a different local model"))
        .accessibilityAddTraits(.isButton)
        .accessibilityIdentifier("ShareCompute.SelectedModel")
    }

    /// Readiness first, then what the model is. Paper leads this line with the
    /// green "Ready on this Mac" because that is the fact gating the button
    /// beside it; the descriptor follows in quiet ink.
    private var secondLine: some View {
        HStack(spacing: 7) {
            Text("Ready on this Mac")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(RapidTheme.bandReady)
            Text("·")
                .font(.system(size: 12))
                .foregroundStyle(RapidTheme.bandInkTertiary)
                .accessibilityHidden(true)
            Text(model.detail)
                .font(.system(size: 12))
                .foregroundStyle(RapidTheme.bandInkSecondary)
                .lineLimit(1)
        }
    }
}

/// What the selector shows when no pool model is downloaded yet.
///
/// Share never offers a download — that is Pool's job — so this row explains
/// the gap and routes there rather than growing a Download Required state the
/// handoff explicitly rules out of this tab.
struct ShareComputeNoLocalModelsRow: View {
    let catalogLoaded: Bool
    let onOpenPool: () -> Void

    var body: some View {
        HStack(spacing: RapidTheme.Space.md) {
            if !catalogLoaded {
                ProgressView().controlSize(.small).accessibilityHidden(true)
            } else {
                Image(systemName: "square.and.arrow.down")
                    .font(.system(size: 15))
                    .foregroundStyle(RapidTheme.bandInkSecondary)
                    .accessibilityHidden(true)
            }
            VStack(alignment: .leading, spacing: 4) {
                Text(
                    catalogLoaded
                        ? String(localized: "No pool model is downloaded yet")
                        : String(localized: "Checking this Mac…")
                )
                .font(.system(size: 15, weight: .bold))
                .foregroundStyle(RapidTheme.bandInk)
                if catalogLoaded {
                    Text("Live Pool shows every supported model, what is online, and what each one would download.")
                        .font(.system(size: 12))
                        .foregroundStyle(RapidTheme.bandInkSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity, minHeight: 72, alignment: .leading)
        .background(
            RapidTheme.bandControl,
            in: RoundedRectangle(cornerRadius: RapidTheme.Radius.button)
        )
        .overlay {
            RoundedRectangle(cornerRadius: RapidTheme.Radius.button)
                .strokeBorder(RapidTheme.bandControlStroke, lineWidth: 1)
        }
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("ShareCompute.NoLocalModels")
        .onTapGesture(perform: onOpenPool)
    }
}
