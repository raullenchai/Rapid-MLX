import SwiftUI

/// The Ready screen: why this result is worth measuring, what will be
/// measured, and one primary action.
///
/// The contribution copy leads, not the control. The previous surface opened
/// with a model menu and a "Run locally" button, which told the user how to
/// operate the feature but never why anyone would.
struct CommunityBenchmarkReadyView: View {
    let model: CommunityBenchmarkModel
    let scope: CommunityBenchmarkScope
    let branch: CommunityContributionBranch
    let isRunEnabled: Bool
    /// Copy explaining that Chat/Images pause for the run, when a model is
    /// currently loaded.
    let serverImpactNote: String?
    /// True below roughly a 900pt window. The fixed-width "What this
    /// measures" panel cannot sit beside the card at that width — it gets
    /// pushed off the trailing edge — so the two stack instead.
    var isNarrow = false
    let onRun: () -> Void
    let onChangeModel: () -> Void
    let onShowTestMethod: () -> Void

    private var invitation: CommunityBenchmarkCopy.ReadyInvitation {
        CommunityBenchmarkCopy.readyInvitation(branch: branch, scope: scope)
    }

    private var doesNotFit: Bool { model.memoryFit == "does_not_fit" }

    var body: some View {
        if isNarrow {
            VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
                contributionCard
                measuresPanel
            }
        } else {
            HStack(alignment: .top, spacing: RapidTheme.Space.xl) {
                contributionCard
                    .frame(maxWidth: .infinity, alignment: .leading)
                measuresPanel
                    .frame(width: 320, alignment: .leading)
            }
        }
    }

    // MARK: - Left: the invitation and the run control

    private var contributionCard: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .top, spacing: RapidTheme.Space.xl) {
                VStack(alignment: .leading, spacing: 0) {
                    Text(invitation.eyebrow)
                        .font(.system(size: 11, weight: .semibold))
                        .tracking(0.7)
                        .foregroundStyle(RapidTheme.brandPrimaryDeep)
                    Text(invitation.headline)
                        .font(.system(size: 26, weight: .semibold))
                        .foregroundStyle(RapidTheme.textPrimary)
                        .padding(.top, RapidTheme.Space.md)
                        .fixedSize(horizontal: false, vertical: true)
                    Text(invitation.body)
                        .font(.system(size: 14))
                        .foregroundStyle(RapidTheme.textSecondary)
                        .padding(.top, RapidTheme.Space.sm)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                CommunityMascot(context: .readyInvitation)
            }

            Divider().padding(.vertical, RapidTheme.Space.lg)

            factsRow

            Divider().padding(.vertical, RapidTheme.Space.lg)

            HStack(spacing: RapidTheme.Space.md) {
                Button(action: onRun) {
                    Label(String(localized: "Run benchmark"), systemImage: "play.fill")
                }
                .buttonStyle(.rapidPrimary)
                .disabled(!isRunEnabled)
                .accessibilityIdentifier("CommunityBenchmark.RunOrStop")

                Button(String(localized: "Change model"), action: onChangeModel)
                    .buttonStyle(.rapidSecondary)
                    .accessibilityIdentifier("CommunityBenchmark.ChangeModel")

                HStack(spacing: 6) {
                    Image(systemName: "lock")
                        .font(.system(size: 11))
                        .accessibilityHidden(true)
                    Text("Private until you publish")
                        .font(RapidFont.body)
                }
                .foregroundStyle(RapidTheme.textSecondary)
                .accessibilityElement(children: .combine)

                Spacer(minLength: 0)
            }

            if let serverImpactNote {
                HStack(alignment: .top, spacing: RapidTheme.Space.sm) {
                    Image(systemName: "info.circle")
                        .font(.system(size: 12))
                        .foregroundStyle(RapidTheme.textTertiary)
                        .accessibilityHidden(true)
                    Text(serverImpactNote)
                        .font(RapidFont.secondary)
                        .foregroundStyle(RapidTheme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(.top, RapidTheme.Space.lg)
            }

            if model.runtimeStatus == "unavailable", let message = model.runtimeMessage {
                Label(message, systemImage: "exclamationmark.triangle.fill")
                    .font(RapidFont.secondary)
                    .foregroundStyle(RapidTheme.statusError)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, RapidTheme.Space.md)
                    .accessibilityIdentifier("CommunityBenchmark.RuntimeUnavailable")
            }
        }
        .padding(RapidTheme.Space.xl)
        .background(
            RapidTheme.surfaceRaised,
            in: RoundedRectangle(cornerRadius: RapidTheme.Radius.card)
        )
        .overlay(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.card)
                .strokeBorder(RapidTheme.hairline)
        )
    }

    /// Model identity plus the three facts the client can always substantiate:
    /// memory fit, download state, and expected duration. No byte counts and
    /// no percentage — the benchmark does not expose a download phase.
    private var factsRow: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: RapidTheme.Space.md) { factLanes }
            VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
                HStack(spacing: RapidTheme.Space.sm) {
                    modelIdentity
                    Spacer(minLength: 0)
                }
                HStack(spacing: RapidTheme.Space.md) { supportingLanes }
            }
        }
    }

    @ViewBuilder
    private var factLanes: some View {
        modelIdentity
        supportingLanes
        Spacer(minLength: 0)
    }

    private var modelIdentity: some View {
        HStack(spacing: RapidTheme.Space.sm) {
            Text(model.entry.alias)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(RapidTheme.textPrimary)
                .lineLimit(1)
                .truncationMode(.tail)
            CommunityWorkloadBadge(workload: scope.workload)
        }
    }

    @ViewBuilder
    private var supportingLanes: some View {
        CommunityFactLane(
            systemImage: doesNotFit ? "exclamationmark.circle" : "checkmark.circle",
            label: doesNotFit
                ? String(localized: "May not fit this Mac")
                : String(localized: "Fits this Mac"),
            detail: model.estimatedMemoryGib.map {
                String(
                    format: String(localized: "%1$d of %2$d GB"),
                    $0,
                    scope.macProfile.memoryGiB
                )
            },
            tone: doesNotFit ? RapidTheme.statusError : RapidTheme.textSecondary
        )
        CommunityFactLane(
            systemImage: "arrow.down.circle",
            label: model.entry.cached
                ? String(localized: "Downloaded")
                : String(localized: "Not downloaded")
        )
        CommunityFactLane(
            systemImage: "clock",
            label: CommunityBenchmarkRunStatus.expectedDuration(for: model.task)
        )
    }

    // MARK: - Right: what this measures

    private var measuresPanel: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.lg) {
            SectionHeader("WHAT THIS MEASURES")
            ForEach(
                Array(
                    CommunityBenchmarkMetrics.measuredQuantities(for: scope.workload).enumerated()
                ),
                id: \.offset
            ) { index, quantity in
                HStack(alignment: .top, spacing: RapidTheme.Space.md) {
                    Text("\(index + 1)")
                        .font(RapidFont.metric)
                        .foregroundStyle(RapidTheme.textTertiary)
                        .frame(width: 14, alignment: .trailing)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(quantity.title)
                            .font(RapidFont.bodyEmphasis)
                            .foregroundStyle(RapidTheme.textPrimary)
                        Text(quantity.detail)
                            .font(RapidFont.secondary)
                            .foregroundStyle(RapidTheme.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .accessibilityElement(children: .combine)
            }
            Button(action: onShowTestMethod) {
                Label(String(localized: "How the test works"), systemImage: "questionmark.circle")
            }
            .buttonStyle(.rapidSecondaryCompact)
            .accessibilityIdentifier("CommunityBenchmark.TestMethodButton")
        }
        .padding(RapidTheme.Space.xl)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RapidTheme.surfaceRaised,
            in: RoundedRectangle(cornerRadius: RapidTheme.Radius.card)
        )
        .overlay(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.card)
                .strokeBorder(RapidTheme.hairline)
        )
    }
}

// MARK: - Running

/// The measurement in progress.
///
/// Shows only progress the CLI actually reports: completed warm-up/measured
/// passes, elapsed time, and an ETA derived from real inter-step timing. When
/// a workload has too few passes for an honest bar (video: one render), or
/// before the first pass lands, the view falls back to an indeterminate
/// "Preparing model" — never a synthesised percentage.
struct CommunityBenchmarkRunningView: View {
    let model: CommunityBenchmarkModel
    let scope: CommunityBenchmarkScope
    let runStartedAt: Date
    /// Everything live, reduced from the CLI's own progress events. The view
    /// renders it; it derives nothing from a clock.
    let progress: CommunityRunProgress
    let plan: CommunityRunPlan
    var isNarrow: Bool = false
    let onStop: () -> Void

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: RapidTheme.Space.xs) {
                    Text("BENCHMARK RUNNING")
                        .font(.system(size: 11, weight: .semibold))
                        .tracking(0.7)
                        .foregroundStyle(RapidTheme.textTertiary)
                    HStack(spacing: RapidTheme.Space.sm) {
                        Text(model.entry.alias)
                            .font(.system(size: 22, weight: .semibold))
                            .lineLimit(1)
                            .truncationMode(.tail)
                        CommunityWorkloadBadge(workload: scope.workload)
                    }
                    Text(scope.macProfile.displayName)
                        .font(RapidFont.metric)
                        .foregroundStyle(RapidTheme.textSecondary)
                }
                Spacer(minLength: RapidTheme.Space.lg)
                Button(action: onStop) {
                    Label(String(localized: "Stop"), systemImage: "stop.fill")
                }
                .buttonStyle(.rapidSecondary)
                .accessibilityIdentifier("CommunityBenchmark.RunOrStop")
            }

            Divider().padding(.vertical, RapidTheme.Space.lg)

            stageStepper
                .padding(.bottom, RapidTheme.Space.lg)

            HStack(alignment: .firstTextBaseline, spacing: RapidTheme.Space.md) {
                Text(progress.stage.activeTitle)
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(RapidTheme.textPrimary)
                    .accessibilityIdentifier("CommunityBenchmark.RunStage")
                if let caption = progress.passCaption {
                    Text(caption)
                        .font(RapidFont.body)
                        .foregroundStyle(RapidTheme.textSecondary)
                        .monospacedDigit()
                        .accessibilityIdentifier("CommunityBenchmark.RunPassCount")
                }
                Spacer(minLength: 0)
            }
            .padding(.bottom, RapidTheme.Space.md)

            progressTrack

            metrics
                .padding(.top, RapidTheme.Space.xl)

            Divider().padding(.vertical, RapidTheme.Space.lg)

            HStack(alignment: .top, spacing: RapidTheme.Space.sm) {
                Image(systemName: "info.circle")
                    .font(.system(size: 12))
                    .foregroundStyle(RapidTheme.textTertiary)
                    .accessibilityHidden(true)
                Text("Chat and Images are paused so the benchmark has the Mac to itself. Your model reloads automatically when it finishes.")
                    .font(RapidFont.secondary)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(RapidTheme.Space.xl)
        .background(
            RapidTheme.surfaceRaised,
            in: RoundedRectangle(cornerRadius: RapidTheme.Radius.card)
        )
        .overlay(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.card)
                .strokeBorder(RapidTheme.hairline)
        )
    }

    /// The stages this workload will actually pass through, with the ones it
    /// has already completed marked. Nothing here is timed: a stage is done
    /// because a later stage's event arrived.
    private var stageStepper: some View {
        let stages = plan.stages
        return HStack(spacing: isNarrow ? RapidTheme.Space.sm : RapidTheme.Space.md) {
            ForEach(Array(stages.enumerated()), id: \.element) { index, stage in
                let isDone = stage < progress.stage
                let isActive = stage == progress.stage
                HStack(spacing: 5) {
                    Image(
                        systemName: isDone
                            ? "checkmark.circle.fill"
                            : (isActive ? "circle.inset.filled" : "circle")
                    )
                    .font(.system(size: 11))
                    .foregroundStyle(
                        isDone || isActive ? RapidTheme.brandPrimary : RapidTheme.textTertiary
                    )
                    // At 900pt the five LLM stage names do not fit beside each
                    // other; the active one still names itself in the headline
                    // directly below, so the labels drop rather than truncate.
                    if !isNarrow || isActive {
                        Text(stage.title)
                            .font(.system(size: 11, weight: isActive ? .semibold : .regular))
                            .foregroundStyle(
                                isActive ? RapidTheme.textPrimary : RapidTheme.textSecondary
                            )
                            .lineLimit(1)
                    }
                }
                if index < stages.count - 1 {
                    Rectangle()
                        .fill(isDone ? RapidTheme.brandPrimary : RapidTheme.hairline)
                        .frame(height: 1)
                        .frame(maxWidth: .infinity)
                }
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityIdentifier("CommunityBenchmark.RunStages")
        .accessibilityLabel(
            String(
                format: String(localized: "Stage %1$@ of %2$d"),
                progress.stage.title,
                stages.count
            )
        )
    }

    /// The mascot rides the leading edge of the determinate bar. With Reduce
    /// Motion on it stays visible but stationary; with no determinate progress
    /// to ride, it sits at the start and the bar becomes indeterminate.
    private var progressTrack: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
            // The mascot is present for the whole run, not only once there is
            // a determinate bar to ride. The `else` branch used to render the
            // bar alone, so during "Getting ready" — which is every second of
            // model loading — the screen had no mascot at all, contradicting
            // this view's own comment.
            GeometryReader { proxy in
                CommunityMascot(context: .running, isAnimating: !reduceMotion)
                    .offset(x: mascotOffset(in: proxy.size.width))
                    .animation(
                        reduceMotion ? nil : .easeInOut(duration: 0.4),
                        value: progress.fraction
                    )
            }
            .frame(height: CommunityMascotContext.running.pointSize)
            .accessibilityHidden(true)

            if let fraction = progress.fraction {
                ProgressView(value: fraction)
                    .progressViewStyle(.linear)
                    .tint(RapidTheme.brandPrimary)
                    .accessibilityIdentifier("CommunityBenchmark.RunProgressBar")
                    .accessibilityLabel(String(localized: "Benchmark progress"))
                    .accessibilityValue(
                        String(
                            format: String(localized: "%1$d percent"),
                            Int((fraction * 100).rounded())
                        )
                    )
            } else {
                // No completed pass yet (or a single-render workload). An
                // indeterminate bar is the only honest shape.
                ProgressView()
                    .progressViewStyle(.linear)
                    .tint(RapidTheme.brandPrimary)
                    .accessibilityIdentifier("CommunityBenchmark.RunProgressBar")
                    .accessibilityLabel(String(localized: "Preparing model"))
            }

            // The CLI's own words while there is nothing to count yet —
            // "Loading …", "Model loaded in 1 s". The reducer has always
            // recorded this and the view never showed it, which is why the
            // preparation phase looked frozen.
            if let statusLine = progress.statusLine, progress.fraction == nil {
                Text(statusLine)
                    .font(RapidFont.secondary)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .accessibilityIdentifier("CommunityBenchmark.RunStatusLine")
            }
        }
    }

    /// Where the mascot sits: at the leading edge until real passes exist,
    /// then at the leading edge of the filled bar.
    private func mascotOffset(in width: CGFloat) -> CGFloat {
        let size = CommunityMascotContext.running.pointSize
        guard let fraction = progress.fraction else { return 0 }
        return max(0, min(width - size, width * fraction - size / 2))
    }

    private var metrics: some View {
        TimelineView(.periodic(from: runStartedAt, by: 1)) { context in
            HStack(alignment: .top, spacing: RapidTheme.Space.xxl) {
                metric(
                    label: String(localized: "ELAPSED"),
                    value: CommunityBenchmarkRunStatus.elapsed(
                        from: runStartedAt, to: context.date
                    ),
                    identifier: "CommunityBenchmark.RunElapsed"
                )
                if let timeLeft = progress.timeLeft {
                    metric(
                        label: String(localized: "TIME LEFT"),
                        value: timeLeft,
                        identifier: "CommunityBenchmark.RunETA"
                    )
                }
                if let measurement = progress.latestMeasurement {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("LATEST MEASUREMENT")
                            .font(.system(size: 10, weight: .semibold))
                            .tracking(0.6)
                            .foregroundStyle(RapidTheme.textTertiary)
                        Text(measurement.value)
                            .font(.system(size: 22, weight: .medium, design: .monospaced))
                            .monospacedDigit()
                            .foregroundStyle(RapidTheme.textPrimary)
                        Text(
                            String(
                                format: String(localized: "on pass %1$d"),
                                measurement.passNumber
                            )
                        )
                        .font(RapidFont.secondary)
                        .foregroundStyle(RapidTheme.textSecondary)
                        .monospacedDigit()
                    }
                    .accessibilityElement(children: .combine)
                    .accessibilityIdentifier("CommunityBenchmark.RunProgress")
                }
                Spacer(minLength: 0)
            }
        }
    }

    private func metric(label: String, value: String, identifier: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.system(size: 10, weight: .semibold))
                .tracking(0.6)
                .foregroundStyle(RapidTheme.textTertiary)
            Text(value)
                .font(.system(size: 22, weight: .medium, design: .monospaced))
                .monospacedDigit()
                .foregroundStyle(RapidTheme.textPrimary)
        }
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier(identifier)
    }
}
