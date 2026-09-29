import SwiftUI

// MARK: - Reward session strip

/// The light strip above every in-session workbench.
///
/// Three stages, and only one of them is Rapid's: compute is prepared here,
/// then QuickSilver tracks, then rewards live in QuickSilver. Showing the
/// whole arc while highlighting the current stage is what keeps the ownership
/// boundary legible at the moment the user most wants to know who is doing
/// what.
struct ShareComputeRewardSessionStrip: View {
    enum Stage: Int, CaseIterable, Identifiable {
        case preparingCompute
        case providerTracks
        case rewardsInQuickSilver

        var id: Int { rawValue }

        func title(isComplete: Bool) -> String {
            switch self {
            case .preparingCompute:
                // Paper's wording on the in-session strip.
                return isComplete
                    ? String(localized: "Compute connected")
                    : String(localized: "Preparing compute")
            case .providerTracks:
                return String(localized: "QuickSilver tracking")
            case .rewardsInQuickSilver:
                return String(localized: "Rewards in QuickSilver")
            }
        }
    }

    let active: Stage
    /// Stages before ``active`` render as done rather than as pending.
    var completedThrough: Stage?
    var isNarrow = false

    var body: some View {
        HStack(spacing: isNarrow ? 10 : 16) {
            ShareComputeEyebrow(text: "Reward session", tone: RapidTheme.textPrimary, size: 11)
            Rectangle()
                .fill(RapidTheme.hairlineStrong)
                .frame(width: 1, height: 20)
                .accessibilityHidden(true)
            ForEach(Array(Stage.allCases.enumerated()), id: \.element.id) { index, stage in
                stageView(stage)
                if index < Stage.allCases.count - 1 {
                    Image(systemName: "arrow.right")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(RapidTheme.textTertiary)
                        .accessibilityHidden(true)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, isNarrow ? 20 : 28)
        .frame(minHeight: 62)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RapidTheme.surfaceRaised)
        .overlay(alignment: .bottom) {
            Rectangle().fill(RapidTheme.hairlineStrong).frame(height: 1)
        }
        .accessibilityElement(children: .contain)
    }

    private func stageInk(done: Bool, isActive: Bool) -> Color {
        if done { return RapidTheme.statusReady }
        if isActive { return RapidTheme.textPrimary }
        return RapidTheme.brandPrimaryInk
    }

    private func isDone(_ stage: Stage) -> Bool {
        guard let completedThrough else { return false }
        return stage.rawValue <= completedThrough.rawValue
    }

    /// Paper's three treatments, and each one says something different:
    ///
    ///   * DONE — green filled disc, white tick, green label at REGULAR
    ///     weight. A finished stage recedes.
    ///   * ACTIVE — amber filled disc, dark numeral, ink label at BOLD. The
    ///     only emphasised item on the strip.
    ///   * PENDING — amber OUTLINED disc and an amber label, still regular.
    ///     Pending is not disabled: those stages are going to happen, and
    ///     drawing them in neutral grey (which this did) read as "inactive"
    ///     rather than "next".
    private func stageView(_ stage: Stage) -> some View {
        let done = isDone(stage)
        let isActive = stage == active
        return HStack(spacing: 8) {
            Group {
                if done {
                    Image(systemName: "checkmark")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(RapidTheme.surfaceRaised)
                } else {
                    Text("\(stage.rawValue + 1)")
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                        .foregroundStyle(
                            isActive ? RapidTheme.onBrandPrimary : RapidTheme.brandPrimaryInk
                        )
                }
            }
            .frame(width: 22, height: 22)
            .background {
                if done {
                    Circle().fill(RapidTheme.statusReady)
                } else if isActive {
                    Circle().fill(RapidTheme.brandPrimary)
                } else {
                    Circle().strokeBorder(RapidTheme.brandPrimaryDeep, lineWidth: 1)
                }
            }
            if !isNarrow || isActive {
                Text(stage.title(isComplete: done))
                    .font(.system(size: 13, weight: isActive ? .bold : .regular))
                    .foregroundStyle(stageInk(done: done, isActive: isActive))
                    .lineLimit(1)
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(stage.title(isComplete: done))
        .accessibilityValue(
            done
                ? String(localized: "Complete")
                : isActive ? String(localized: "In progress") : String(localized: "Waiting")
        )
    }
}

// MARK: - Preparing

/// The five-step preparation sequence.
///
/// Every step's status comes from a phase the provider actually published —
/// see ``ShareComputePreparationPlan``. Nothing here is simulated or advanced
/// on a timer; a step reads Complete because the provider moved past it.
struct ShareComputePreparingView: View {
    let modelTitle: String
    let rows: [ShareComputePreparationRow]
    let tracking: ShareComputeRewardTracking
    let canStop: Bool
    let isNarrow: Bool
    let onStop: () -> Void

    var body: some View {
        ShareComputeWorkbenchFrame {
            VStack(spacing: 0) {
                ShareComputeRewardSessionStrip(active: .preparingCompute, isNarrow: isNarrow)
                if isNarrow {
                    VStack(spacing: 0) {
                        band
                        rewardPanel
                    }
                } else {
                    HStack(spacing: 0) {
                        band
                        rewardPanel.frame(width: 318)
                    }
                    .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    private var band: some View {
        VStack(alignment: .leading, spacing: 24) {
            HStack(alignment: .top, spacing: RapidTheme.Space.lg) {
                VStack(alignment: .leading, spacing: 4) {
                    ShareComputeEyebrow(text: "Preparing this Mac", size: 11)
                    Text("Starting the shared model")
                        .font(.system(size: isNarrow ? 21 : 25, weight: .bold))
                        .foregroundStyle(RapidTheme.bandInk)
                        .fixedSize(horizontal: false, vertical: true)
                    Text(modelTitle)
                        .font(.system(size: 12))
                        .foregroundStyle(RapidTheme.bandInkSecondary)
                }
                Spacer(minLength: 0)
                ShareComputeStopButton(isEnabled: canStop, action: onStop)
            }

            ShareComputeStepRail(rows: rows, isNarrow: isNarrow)

            HStack(spacing: 8) {
                Image(systemName: "arrow.uturn.backward")
                    .font(.system(size: 11))
                    .foregroundStyle(RapidTheme.bandReady)
                    .accessibilityHidden(true)
                Text("Rapid restores your previous model after this reward session ends.")
                    .font(.system(size: 12))
                    .foregroundStyle(RapidTheme.bandInkSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.top, 14)
            .overlay(alignment: .top) {
                Rectangle().fill(RapidTheme.bandHairline).frame(height: 1)
            }
            .accessibilityElement(children: .combine)
        }
        .padding(.horizontal, isNarrow ? 20 : 30)
        .padding(.vertical, 27)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(RapidTheme.surfaceBand)
    }

    private var rewardPanel: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.lg) {
            VStack(alignment: .leading, spacing: 8) {
                ShareComputeEyebrow(
                    text: "Reward status",
                    tone: RapidTheme.onBrandPrimarySecondary,
                    size: 11
                )
                Text(tracking.headline)
                    .font(.system(size: isNarrow ? 22 : 27, weight: .bold))
                    .foregroundStyle(RapidTheme.onBrandPrimary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: RapidTheme.Space.lg)
            VStack(alignment: .leading, spacing: 8) {
                if tracking.showsNoRewardCountedYet {
                    Text("No reward is counted yet")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundStyle(RapidTheme.onBrandPrimary)
                }
                Text(tracking.detail)
                    .font(.system(size: 11))
                    .foregroundStyle(RapidTheme.onBrandPrimarySecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.top, 14)
            .overlay(alignment: .top) {
                Rectangle().fill(RapidTheme.brandPrimaryHairline).frame(height: 1)
            }
        }
        .padding(27)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(RapidTheme.brandPrimary)
        .accessibilityElement(children: .contain)
    }
}

/// The five numbered circles, their connector line, and their labels.
///
/// The connector is drawn in a background layer whose horizontal inset is
/// exactly half a column, so it starts at the centre of the first circle and
/// ends at the centre of the last one at ANY width. That is the whole point of
/// the fixed-slot rule: the rail is a five-column grid, the circle is centred
/// in its column, and the line is derived from the same geometry rather than
/// positioned by hand for one screenshot.
struct ShareComputeStepRail: View {
    let rows: [ShareComputePreparationRow]
    var isNarrow = false

    private let circleSize: CGFloat = 26

    var body: some View {
        VStack(spacing: 13) {
            ZStack(alignment: .top) {
                connectorLine
                HStack(spacing: 0) {
                    ForEach(rows) { row in
                        circle(for: row).frame(maxWidth: .infinity)
                    }
                }
            }
            .frame(height: circleSize)

            HStack(alignment: .top, spacing: 0) {
                ForEach(rows) { row in
                    label(for: row).frame(maxWidth: .infinity)
                }
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(String(localized: "Preparation steps"))
    }

    private var connectorLine: some View {
        GeometryReader { proxy in
            let columnWidth = proxy.size.width / CGFloat(max(1, rows.count))
            Rectangle()
                .fill(RapidTheme.bandHairlineStrong)
                .frame(height: 2)
                .padding(.horizontal, columnWidth / 2)
                .offset(y: circleSize / 2 - 1)
        }
        .accessibilityHidden(true)
    }

    @ViewBuilder
    private func circle(for row: ShareComputePreparationRow) -> some View {
        // Status carries a distinct GLYPH as well as a colour — a tick, a
        // number, an exclamation mark — so the rail stays readable in
        // monochrome and to a colour-blind reader.
        Group {
            switch row.status {
            // Band-scoped fills with a GRAPHITE glyph, not a white one. The
            // rail lives on the band, where the semantic fills are the light
            // members of each ramp (``bandReady``, ``bandDestructive``); a
            // white tick on a pale green disc is ~1.7:1 and was barely there
            // in Dark, where ``statusReady`` resolves light.
            case .complete, .alreadyRegistered:
                Image(systemName: "checkmark")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(RapidTheme.surfaceBand)
                    .frame(width: circleSize, height: circleSize)
                    .background(RapidTheme.bandReady, in: Circle())
            case .failed:
                Image(systemName: "exclamationmark")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(RapidTheme.surfaceBand)
                    .frame(width: circleSize, height: circleSize)
                    .background(RapidTheme.bandDestructive, in: Circle())
            case .inProgress:
                Text("\(row.step.number)")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundStyle(RapidTheme.brandPrimary)
                    .frame(width: circleSize, height: circleSize)
                    .background(RapidTheme.surfaceBand, in: Circle())
                    .overlay {
                        Circle().strokeBorder(RapidTheme.brandPrimary, lineWidth: 2)
                    }
            case .waiting:
                Text("\(row.step.number)")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(RapidTheme.bandInkTertiary)
                    .frame(width: circleSize, height: circleSize)
                    .background(RapidTheme.surfaceBand, in: Circle())
                    .overlay {
                        Circle().strokeBorder(RapidTheme.bandHairlineStrong, lineWidth: 1)
                    }
            }
        }
        .accessibilityHidden(true)
    }

    private func label(for row: ShareComputePreparationRow) -> some View {
        VStack(spacing: 3) {
            Text(row.step.title)
                .font(.system(size: 11, weight: row.status == .waiting ? .regular : .bold))
                .foregroundStyle(
                    row.status == .waiting ? RapidTheme.bandInkTertiary : RapidTheme.bandInk
                )
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
            Text(row.status.title)
                .font(.system(size: 10))
                .foregroundStyle(statusTint(row.status))
                .multilineTextAlignment(.center)
        }
        .padding(.horizontal, 4)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(row.step.number). \(row.step.title)")
        .accessibilityValue(row.status.title)
    }

    private func statusTint(_ status: ShareComputePreparationStatus) -> Color {
        switch status {
        case .complete, .alreadyRegistered: return RapidTheme.bandReady
        case .inProgress: return RapidTheme.brandPrimary
        case .failed: return RapidTheme.statusError
        case .waiting: return RapidTheme.bandInkSecondary
        }
    }
}

// MARK: - Online

/// Live operational proof while the Mac is in the pool.
///
/// Every figure is read from the provider's own desktop status file: the
/// elapsed clock from `connected_at`, requests from `inflight`, the node id
/// and worker from the registration it published. The reward half is
/// deliberately a separate panel with a separate ground — Rapid's operational
/// truth and QuickSilver's reward activity never share a surface here.
struct ShareComputeOnlineView: View {
    let modelTitle: String
    let worker: String
    let nodeID: String?
    let inflight: Int?
    /// When the node joined the pool. The view ticks its own clock from this
    /// rather than being handed an elapsed value, because the provider's
    /// status file only republishes every few seconds and a clock that
    /// advanced in visible jumps would read as a stalled session.
    let poolJoinedAt: Date?
    let isReconnecting: Bool
    let payoutAccountConnected: Bool
    let canStop: Bool
    let isNarrow: Bool
    let onStop: () -> Void

    var body: some View {
        ShareComputeWorkbenchFrame {
            VStack(spacing: 0) {
                ShareComputeRewardSessionStrip(
                    active: .providerTracks,
                    completedThrough: .preparingCompute,
                    isNarrow: isNarrow
                )
                if isNarrow {
                    VStack(spacing: 0) {
                        band
                        rewardPanel
                    }
                } else {
                    HStack(spacing: 0) {
                        band
                        rewardPanel.frame(width: 438)
                    }
                    .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    private var band: some View {
        VStack(alignment: .leading, spacing: 24) {
            HStack(alignment: .top, spacing: RapidTheme.Space.lg) {
                VStack(alignment: .leading, spacing: 5) {
                    ShareComputeEyebrow(
                        text: isReconnecting ? "Reconnecting" : "Compute connected",
                        tone: isReconnecting ? RapidTheme.brandPrimary : RapidTheme.bandReady,
                        size: 11
                    )
                    Text(
                        isReconnecting
                            ? String(localized: "Reconnecting to the pool")
                            : String(localized: "Your Mac is contributing")
                    )
                    .font(.system(size: isNarrow ? 21 : 25, weight: .bold))
                    .foregroundStyle(RapidTheme.bandInk)
                    .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
                ShareComputeStopButton(isEnabled: canStop, action: onStop)
            }

            metrics

            // The lock statement sits with the model name because that is the
            // question it answers: the user is looking at the model and
            // wondering whether they can change it.
            // Quiet mono, with the node id — and only the node id — carrying
            // the link tone. Paper draws this whole line in a neutral grey;
            // colouring the sentence steel spent the band's one identity
            // colour on a caption and left the id with nothing to stand out
            // against, which is the opposite of what the tone is for.
            Text(lockLine)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(RapidTheme.bandInkTertiary)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
                .accessibilityLabel(
                    String(
                        format: String(localized: "Sharing %1$@. Locked while sharing — stop to change."),
                        modelTitle
                    )
                )
        }
        .padding(.horizontal, isNarrow ? 20 : 30)
        .padding(.vertical, 28)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(RapidTheme.surfaceBand)
    }

    private var lockLine: AttributedString {
        var line = AttributedString(
            [
                modelTitle,
                String(localized: "Locked while sharing — Stop to change"),
                worker,
            ]
            .joined(separator: "  ·  ")
        )
        guard let nodeID, !nodeID.isEmpty else { return line }
        line.append(AttributedString("  ·  "))
        var identifier = AttributedString(nodeID)
        identifier.foregroundColor = RapidTheme.bandLink
        line.append(identifier)
        return line
    }

    private var metrics: some View {
        AdaptiveMetricRow(isNarrow: isNarrow) {
            TimelineView(.periodic(from: .now, by: 1)) { context in
                metric(
                    label: String(localized: "Time shared"),
                    value: poolJoinedAt.map {
                        ShareComputeDuration.clock(context.date.timeIntervalSince($0))
                    } ?? "—",
                    tint: RapidTheme.bandInk,
                    isMonospaced: true
                )
            }
            metric(
                label: String(localized: "Requests now"),
                // `nil` renders an em dash, never a zero: "0 requests" is a
                // measurement, and an absent heartbeat is not one.
                value: inflight.map(String.init) ?? "—",
                tint: RapidTheme.brandPrimary,
                isMonospaced: true
            )
            metric(
                label: String(localized: "Provider"),
                value: isReconnecting
                    ? String(localized: "Reconnecting")
                    : String(localized: "Connected"),
                tint: isReconnecting ? RapidTheme.brandPrimary : RapidTheme.bandReady,
                isMonospaced: false
            )
        }
        .padding(.vertical, 18)
        .overlay(alignment: .top) {
            Rectangle().fill(RapidTheme.bandHairline).frame(height: 1)
        }
        .overlay(alignment: .bottom) {
            Rectangle().fill(RapidTheme.bandHairline).frame(height: 1)
        }
    }

    private func metric(
        label: String,
        value: String,
        tint: Color,
        isMonospaced: Bool
    ) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(label.uppercased())
                .font(.system(size: 11))
                .tracking(0.5)
                .foregroundStyle(RapidTheme.bandInkSecondary)
            Text(value)
                .font(
                    isMonospaced
                        ? .system(size: 28, weight: .bold, design: .monospaced)
                        : .system(size: 15, weight: .bold)
                )
                .monospacedDigit()
                .foregroundStyle(tint)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(label): \(value)")
    }

    private var rewardPanel: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.lg) {
            VStack(alignment: .leading, spacing: 9) {
                HStack(spacing: 9) {
                    Image(systemName: "arrow.right")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundStyle(RapidTheme.onBrandPrimary)
                        .accessibilityHidden(true)
                    ShareComputeEyebrow(
                        text: "QuickSilver tracking",
                        tone: RapidTheme.onBrandPrimarySecondary,
                        size: 11
                    )
                }
                Text("Reward activity is managed in QuickSilver.")
                    .font(.system(size: isNarrow ? 22 : 29, weight: .bold))
                    .foregroundStyle(RapidTheme.onBrandPrimary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: RapidTheme.Space.md)
            payoutCallout
            Text(
                String(
                    format: String(localized: "Sharing %1$@ · %2$@"),
                    modelTitle,
                    payoutAccountConnected
                        ? String(localized: "Provider account connected")
                        : String(localized: "Provider account not reported")
                )
            )
            .font(.system(size: 11))
            .foregroundStyle(RapidTheme.onBrandPrimarySecondary)
            .fixedSize(horizontal: false, vertical: true)
        }
        .padding(28)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(RapidTheme.brandPrimary)
    }

    /// Only claims a connected payout account when the provider actually
    /// published one. Without that field the callout says so rather than
    /// implying money is already routed somewhere.
    private var payoutCallout: some View {
        HStack(alignment: .top, spacing: 11) {
            Image(systemName: payoutAccountConnected ? "dollarsign.circle.fill" : "questionmark.circle")
                .font(.system(size: 20))
                .foregroundStyle(RapidTheme.onBrandPrimary)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 3) {
                Text(
                    payoutAccountConnected
                        ? String(localized: "Payout account connected")
                        : String(localized: "Payout account not reported")
                )
                .font(.system(size: 13, weight: .bold))
                .foregroundStyle(RapidTheme.onBrandPrimary)
                Text("Final usage and rewards appear in QuickSilver.")
                    .font(.system(size: 12))
                    .foregroundStyle(RapidTheme.onBrandPrimarySecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .padding(14)
        .background(
            RapidTheme.brandPrimaryRaised,
            in: RoundedRectangle(cornerRadius: 6)
        )
        .overlay {
            RoundedRectangle(cornerRadius: 6)
                .strokeBorder(RapidTheme.brandPrimaryHairline, lineWidth: 1)
        }
        .accessibilityElement(children: .combine)
    }
}

/// Three metrics side by side, wrapping to a column at narrow width so none of
/// them truncates its value.
private struct AdaptiveMetricRow<Content: View>: View {
    let isNarrow: Bool
    @ViewBuilder var content: Content

    var body: some View {
        if isNarrow {
            VStack(alignment: .leading, spacing: RapidTheme.Space.md) { content }
        } else {
            HStack(alignment: .top, spacing: 22) { content }
        }
    }
}

// MARK: - Session complete

/// The completion receipt.
///
/// Recognition on the amber side, the provider handoff on the white side. No
/// amount appears anywhere, because none was ever returned — the large word is
/// `Complete`, which describes the session, not a payment.
struct ShareComputeSessionCompleteView: View {
    let receipt: ShareComputeReceipt
    let isNarrow: Bool
    let onViewReward: () -> Void
    let onShareAgain: () -> Void

    var body: some View {
        ShareComputeWorkbenchFrame {
            VStack(spacing: 0) {
                ShareComputeRewardSessionStrip(
                    active: .rewardsInQuickSilver,
                    completedThrough: .providerTracks,
                    isNarrow: isNarrow
                )
                if isNarrow {
                    VStack(spacing: 0) {
                        receiptPanel
                        rewardPanel
                    }
                } else {
                    HStack(spacing: 0) {
                        receiptPanel
                        rewardPanel.frame(width: 438)
                    }
                    .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    private var receiptPanel: some View {
        VStack(alignment: .leading, spacing: 24) {
            HStack(alignment: .top, spacing: RapidTheme.Space.md) {
                VStack(alignment: .leading, spacing: 4) {
                    ShareComputeEyebrow(
                        text: "Completion receipt",
                        // This eyebrow sits on the amber panel, so it needs
                        // the amber-panel green, not the app-surface one.
                        tone: RapidTheme.onBrandPrimaryReady,
                        size: 11
                    )
                    Text("Your contribution session is complete")
                        .font(.system(size: isNarrow ? 20 : 24, weight: .bold))
                        .foregroundStyle(RapidTheme.onBrandPrimary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
                Text(Self.timestamp(receipt.endedAt))
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(RapidTheme.onBrandPrimarySecondary)
            }

            HStack(alignment: .firstTextBaseline, spacing: 9) {
                Text("Complete")
                    .font(.system(size: isNarrow ? 40 : 60, weight: .heavy))
                    .foregroundStyle(RapidTheme.onBrandPrimary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.6)
                Text("reward activity continues in QuickSilver")
                    .font(.system(size: 13))
                    .foregroundStyle(RapidTheme.onBrandPrimarySecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            facts
        }
        .padding(.horizontal, isNarrow ? 20 : 30)
        .padding(.vertical, 28)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(RapidTheme.brandPrimary)
    }

    private var facts: some View {
        AdaptiveMetricRow(isNarrow: isNarrow) {
            fact(
                String(localized: "Time shared"),
                ShareComputeDuration.clock(receipt.duration),
                isMonospaced: true
            )
            fact(String(localized: "Worker"), receipt.worker, isMonospaced: false)
            fact(
                // Named explicitly rather than "Previous model": the user is
                // being told the outcome of something Rapid did for them.
                String(localized: "Model restore"),
                receipt.restoreStatus.title,
                isMonospaced: false
            )
        }
        .padding(.top, 16)
        .overlay(alignment: .top) {
            Rectangle().fill(RapidTheme.brandPrimaryHairline).frame(height: 1)
        }
    }

    private func fact(_ label: String, _ value: String, isMonospaced: Bool) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label.uppercased())
                .font(.system(size: 11))
                .tracking(0.5)
                .foregroundStyle(RapidTheme.onBrandPrimarySecondary)
            Text(value)
                .font(
                    isMonospaced
                        ? .system(size: 16, weight: .bold, design: .monospaced)
                        : .system(size: 14, weight: .bold)
                )
                .foregroundStyle(RapidTheme.onBrandPrimary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(label): \(value)")
    }

    private var rewardPanel: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.lg) {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 9) {
                    Image(systemName: "arrow.right")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundStyle(RapidTheme.textPrimary)
                        .accessibilityHidden(true)
                    ShareComputeEyebrow(
                        text: "Reward activity",
                        tone: RapidTheme.textSecondary,
                        size: 11
                    )
                }
                Text("QuickSilver payout account")
                    .font(.system(size: isNarrow ? 20 : 25, weight: .bold))
                    .foregroundStyle(RapidTheme.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)
                Text(receipt.rewardStatus.detailTitle)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(RapidTheme.textPrimary)
                Text("Payment timing and destination remain managed by the provider.")
                    .font(.system(size: 13))
                    .foregroundStyle(RapidTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: RapidTheme.Space.lg)
            VStack(spacing: 10) {
                Button(action: onViewReward) {
                    HStack(spacing: 6) {
                        Text("View reward in QuickSilver")
                            .font(.system(size: 13, weight: .bold))
                            .foregroundStyle(RapidTheme.surfaceRaised)
                        Image(systemName: "arrow.up.right")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundStyle(RapidTheme.brandPrimary)
                            .accessibilityHidden(true)
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 40)
                    .background(
                        RapidTheme.textPrimary,
                        in: RoundedRectangle(cornerRadius: 6)
                    )
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("ShareCompute.ViewReward")

                Button(action: onShareAgain) {
                    Text("Share again")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(RapidTheme.linkLabel)
                        .frame(maxWidth: .infinity)
                        .frame(height: 34)
                        .overlay {
                            RoundedRectangle(cornerRadius: 6)
                                .strokeBorder(RapidTheme.hairlineStrong, lineWidth: 1)
                        }
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("ShareCompute.ShareAgain")
            }
        }
        .padding(28)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(RapidTheme.surfaceRaised)
    }

    static func timestamp(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = Calendar.current.isDateInToday(date) ? .none : .medium
        formatter.timeStyle = .short
        let time = formatter.string(from: date)
        return Calendar.current.isDateInToday(date)
            ? String(format: String(localized: "TODAY · %1$@"), time)
            : time
    }
}
