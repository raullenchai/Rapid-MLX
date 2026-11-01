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
            ShareComputeEyebrow(text: String(localized: "Reward session"), tone: RapidTheme.textPrimary, size: 11)
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
                        .font(.rapid(size: 10, weight: .bold, design: .monospaced))
                        .foregroundStyle(
                            isActive ? RapidTheme.onBrandPrimary : RapidTheme.brandPrimaryInk
                        )
                }
            }
            .frame(width: done ? 22 : UIFontSettings.shared.scaled(22), height: done ? 22 : UIFontSettings.shared.scaled(22))
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
                    .font(.rapid(size: 13, weight: isActive ? .bold : .regular))
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
        ShareSessionCard {
            HStack(alignment: .center) {
                ProgressView().controlSize(.small)
                VStack(alignment: .leading, spacing: 4) {
                    Text("Connecting…").font(.rapid(size: 15, weight: .semibold))
                    Text(modelTitle).font(.rapid(size: 12)).foregroundStyle(.secondary)
                }
                Spacer()
                sessionStopButton(enabled: canStop, action: onStop)
            }
            ForEach(rows) { row in
                HStack {
                    Text(row.step.title)
                    Spacer()
                    Text(row.status.title).foregroundStyle(.secondary)
                }.font(.rapid(size: 12))
            }
            sessionFootnote()
        }
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

    private var circleSize: CGFloat { UIFontSettings.shared.scaled(26) }

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
    func circle(for row: ShareComputePreparationRow) -> some View {
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
                    .font(.rapid(size: 10, weight: .bold, design: .monospaced))
                    .foregroundStyle(RapidTheme.brandPrimary)
                    .frame(width: circleSize, height: circleSize)
                    .background(RapidTheme.surfaceBand, in: Circle())
                    .overlay {
                        Circle().strokeBorder(RapidTheme.brandPrimary, lineWidth: 2)
                    }
            case .waiting:
                Text("\(row.step.number)")
                    .font(.rapid(size: 10, design: .monospaced))
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
                .font(.rapid(size: 11, weight: row.status == .waiting ? .regular : .bold))
                .foregroundStyle(
                    row.status == .waiting ? RapidTheme.bandInkTertiary : RapidTheme.bandInk
                )
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
            Text(row.status.title)
                .font(.rapid(size: 10))
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
        ShareSessionCard {
            HStack(alignment: .center) {
                Circle().fill(isReconnecting ? Color.orange : Color.green).frame(width: 9, height: 9).accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 4) {
                    Text(isReconnecting ? String(localized: "Reconnecting…") : String(localized: "Sharing")).font(.rapid(size: 15, weight: .semibold))
                    Text(isReconnecting ? String(localized: "Reconnecting to the relay.") : String(localized: "Your Mac is online and serving requests."))
                        .font(.rapid(size: 12)).foregroundStyle(.secondary)
                }
                Spacer(minLength: 8)
                sessionStopButton(enabled: canStop, action: onStop)
            }
            HStack(alignment: .top, spacing: 12) {
                TimelineView(.periodic(from: .now, by: 1)) { context in
                    sessionMetric(String(localized: "Time shared"), poolJoinedAt.map { ShareComputeDuration.clock(context.date.timeIntervalSince($0)) } ?? "—")
                }
                sessionMetric(String(localized: "Requests now"), inflight.map(String.init) ?? "—")
                sessionMetric(String(localized: "Provider status"), isReconnecting ? String(localized: "Reconnecting") : String(localized: "Connected"))
            }
            Text(([modelTitle, worker] + (nodeID.map { [$0] } ?? [])).joined(separator: " · "))
                .font(.rapid(size: 12)).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle).textSelection(.enabled)
            sessionFootnote()
        }
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
        ShareSessionCard {
            HStack {
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                Text("Sharing stopped").font(.rapid(size: 15, weight: .semibold))
                Spacer()
                Button("Share again", action: onShareAgain).buttonStyle(.borderedProminent)
                    .accessibilityIdentifier("ShareCompute.ShareAgain")
            }
            HStack(alignment: .top, spacing: 12) {
                sessionMetric(String(localized: "Time shared"), ShareComputeDuration.clock(receipt.duration))
                sessionMetric(String(localized: "Worker"), receipt.worker)
                sessionMetric(String(localized: "Model after sharing"), receipt.restoreStatus.title)
            }
            Text(([receipt.modelTitle, receipt.worker] + (receipt.nodeID.map { [$0] } ?? [])).joined(separator: " · "))
                .font(.rapid(size: 12)).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
            sessionFootnote(action: onViewReward)
        }
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

private struct ShareSessionCard<Content: View>: View {
    @ViewBuilder var content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.lg) { content }
            .padding(RapidTheme.Space.lg).shareGroupedCard()
            .frame(maxWidth: 720, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}
@MainActor private func sessionStopButton(enabled: Bool, action: @escaping () -> Void) -> some View {
    Button("Stop sharing", action: action).buttonStyle(.bordered).tint(.red).disabled(!enabled)
        .accessibilityLabel(String(localized: "Stop sharing this Mac"))
        .accessibilityIdentifier("ShareCompute.Stop")
}
@MainActor private func sessionMetric(_ label: String, _ value: String) -> some View {
    VStack(alignment: .leading, spacing: 4) {
        Text(label).font(.rapid(size: 12)).foregroundStyle(.secondary)
        Text(value).font(.rapid(size: 20, weight: .medium)).monospacedDigit().lineLimit(1)
    }.frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
}
@MainActor private func sessionFootnote(action: (() -> Void)? = nil) -> some View {
    HStack(alignment: .firstTextBaseline, spacing: 8) {
        Image(systemName: "info.circle").accessibilityHidden(true)
        Text("Usage and credits appear in the Credits tab and in QuickSilver.")
        if let action {
            Button("Open QuickSilver ↗", action: action).buttonStyle(.plain)
                .accessibilityIdentifier("ShareCompute.ViewReward")
        } else {
            Link("Open QuickSilver ↗", destination: ShareComputeDestination.rewards)
                .accessibilityIdentifier("ShareCompute.ViewRewardLink")
        }
    }.font(.rapid(size: 12)).foregroundStyle(.secondary)
}
