import SwiftUI

/// The local result: headline metric, supporting metrics, community
/// comparison (or an honest explanation of its absence), and one publish
/// action.
///
/// The comparison area is where the two contribution branches diverge most
/// visibly. With published observations, it shows median / observed range /
/// where this run sits. With a known zero, or with the community read
/// unavailable, it shows an empty-data explanation — never a median computed
/// from one local sample.
struct CommunityBenchmarkResultView: View {
    let result: CommunityBenchmarkResult
    let modelAlias: String
    let scope: CommunityBenchmarkScope
    let branch: CommunityContributionBranch
    let observations: CommunityDataState<CommunityObservationSummary>
    let receipt: CommunityBenchmarkReceipt?
    let isPublishing: Bool
    /// True in a ~700pt detail pane. The card's header and action row both
    /// need to reflow there; at that width a single row truncated the date and
    /// the status chip, and pushed the actions off the card entirely.
    var isNarrow: Bool = false
    let onPublish: () -> Void
    let onRunAgain: () -> Void
    let onBenchmarkAnother: () -> Void

    @State private var showsTechnicalDetails = false

    private var metrics: CommunityBenchmarkMetrics.MetricSet {
        CommunityBenchmarkMetrics.metricSet(for: result, workload: scope.workload)
    }

    private var isPublished: Bool { receipt != nil }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            resultHeader
            Divider()
            metricsRow
            Divider()
            comparisonArea
            if !isPublished {
                Divider()
                publishInvitation
            }
            Divider()
            actions
            if showsTechnicalDetails {
                Divider()
                technicalDetails
            }
        }
        .background(
            RapidTheme.surfaceRaised,
            in: RoundedRectangle(cornerRadius: RapidTheme.Radius.card)
        )
        .overlay(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.card)
                .strokeBorder(RapidTheme.hairline)
        )
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("CommunityBenchmark.Result.\(result.id)")
    }

    // MARK: - Header

    private var resultHeader: some View {
        Group {
            if isNarrow {
                // Two lines: identity, then provenance + status. One row here
                // truncated "Sep 5, 9:37 PM" to "Sep…" and the status chip to
                // "SAVED ON THIS M…", which is worse than wrapping.
                VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
                    HStack(spacing: RapidTheme.Space.md) {
                        resultBadge
                        Text(modelAlias)
                            .font(.system(size: 16, weight: .semibold))
                            .lineLimit(1)
                            .truncationMode(.middle)
                        CommunityWorkloadBadge(workload: scope.workload)
                        Spacer(minLength: 0)
                    }
                    HStack(spacing: RapidTheme.Space.sm) {
                        provenanceText
                        Spacer(minLength: RapidTheme.Space.sm)
                        statusChip
                    }
                }
            } else {
                HStack(spacing: RapidTheme.Space.md) {
                    resultBadge
                    Text(modelAlias)
                        .font(.system(size: 16, weight: .semibold))
                        .lineLimit(1)
                        .truncationMode(.tail)
                    CommunityWorkloadBadge(workload: scope.workload)
                    provenanceText
                    Spacer(minLength: RapidTheme.Space.md)
                    statusChip
                }
            }
        }
        .padding(RapidTheme.Space.xl)
    }

    private var resultBadge: some View {
        Image(systemName: "checkmark")
            .font(.system(size: 12, weight: .semibold))
            .foregroundStyle(RapidTheme.statusReady)
            .frame(width: 26, height: 26)
            .background(RapidTheme.statusReadyTint, in: RoundedRectangle(cornerRadius: 7))
            .accessibilityHidden(true)
    }

    private var provenanceText: some View {
        Text(
            "\(scope.macProfile.displayName) · "
                + CommunityBenchmarkResult.formatCompletedAt(result.completedAt)
        )
        .font(RapidFont.metric)
        .foregroundStyle(RapidTheme.textSecondary)
        .lineLimit(1)
        // The date is the point of this line; letting it shrink a little beats
        // truncating it to "Sep…".
        .minimumScaleFactor(isNarrow ? 0.85 : 1)
    }

    private var statusLabel: String {
        Self.statusLabel(isPublished: isPublished, isNarrow: isNarrow)
    }

    /// The chip's text. Pure so the narrow variant can be asserted without a
    /// pixel baseline: at 700pt the long form rendered as "SAVED ON THIS M…",
    /// and a truncated phrase is not a label.
    nonisolated static func statusLabel(isPublished: Bool, isNarrow: Bool) -> String {
        if isPublished { return String(localized: "PUBLISHED") }
        return isNarrow
            ? String(localized: "SAVED HERE")
            : String(localized: "SAVED ON THIS MAC")
    }

    private var statusChip: some View {
        HStack(spacing: 6) {
            Image(systemName: isPublished ? "checkmark.circle.fill" : "lock")
                .font(.system(size: 11))
                .accessibilityHidden(true)
            // "SAVED HERE" is a complete phrase; "SAVED ON THIS M…" is not.
            // A shorter true label beats a longer truncated one.
            Text(statusLabel)
                .font(.system(size: 11, weight: .semibold))
                .tracking(0.3)
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: false)
        }
        .foregroundStyle(isPublished ? RapidTheme.statusReady : RapidTheme.textSecondary)
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(
            isPublished ? RapidTheme.statusReadyTint : RapidTheme.surfaceCanvas,
            in: RoundedRectangle(cornerRadius: 6)
        )
        .accessibilityElement(children: .combine)
    }

    // MARK: - Metrics

    /// `fixedSize(vertical:)` is load-bearing: the column separator is a
    /// `Divider`, which expands to whatever height the parent offers. Without
    /// it the metrics row inflates to fill the window and leaves a dead band
    /// between the numbers and the comparison area.
    private var metricsRow: some View {
        HStack(alignment: .top, spacing: 0) {
            VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
                Text(metrics.headline?.label.uppercased() ?? String(localized: "RESULT"))
                    .font(.system(size: 11, weight: .semibold))
                    .tracking(0.6)
                    .foregroundStyle(RapidTheme.textTertiary)
                if let headline = metrics.headline {
                    CommunityMetricView(metric: headline, isHeadline: true)
                } else if let status = metrics.incompleteStatus {
                    Text(status)
                        .font(.system(size: 22, weight: .medium))
                        .foregroundStyle(RapidTheme.statusError)
                }
                if metrics.headline != nil {
                    Text(metrics.headlineCaption)
                        .font(RapidFont.secondary)
                        .foregroundStyle(RapidTheme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(RapidTheme.Space.xl)
            .frame(maxWidth: .infinity, alignment: .leading)

            if !metrics.supporting.isEmpty {
                Divider()
                VStack(alignment: .leading, spacing: RapidTheme.Space.lg) {
                    Text("ALSO MEASURED")
                        .font(.system(size: 11, weight: .semibold))
                        .tracking(0.6)
                        .foregroundStyle(RapidTheme.textTertiary)
                    // Two per row so an image/video run with two supporting
                    // metrics does not leave three empty cells.
                    ForEach(Array(metrics.supporting.chunked(into: 2)), id: \.first?.id) { pair in
                        HStack(alignment: .top, spacing: RapidTheme.Space.xxl) {
                            ForEach(pair) { metric in
                                CommunityMetricView(metric: metric)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                            if pair.count == 1 {
                                Spacer().frame(maxWidth: .infinity)
                            }
                        }
                    }
                }
                .padding(RapidTheme.Space.xl)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .fixedSize(horizontal: false, vertical: true)
    }

    // MARK: - Comparison

    @ViewBuilder
    private var comparisonArea: some View {
        if branch.allowsComparisonStatistics,
           let summary = observations.value,
           let median = summary.median {
            comparisonStatistics(summary: summary, median: median)
        } else if let placeholder = CommunityBenchmarkCopy.comparisonPlaceholder(
            branch: branch, scope: scope
        ) {
            CommunityUnavailableBand(
                title: placeholder.title,
                message: placeholder.body,
                isLoading: observations.isLoading
            )
            .padding(RapidTheme.Space.xl)
        }
    }

    private func comparisonStatistics(
        summary: CommunityObservationSummary,
        median: Double
    ) -> some View {
        let unit = summary.unit ?? metrics.headline?.unit ?? ""
        // The statistics are fixed-width columns, so on a narrow window they
        // would push past the card's trailing edge. `ViewThatFits` drops to a
        // stacked arrangement rather than clipping or scrolling the page.
        return ViewThatFits(in: .horizontal) {
            comparisonRow(summary: summary, median: median, unit: unit, stacked: false)
            comparisonRow(summary: summary, median: median, unit: unit, stacked: true)
        }
        .padding(RapidTheme.Space.xl)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RapidTheme.surfaceCanvas)
    }

    @ViewBuilder
    private func comparisonRow(
        summary: CommunityObservationSummary,
        median: Double,
        unit: String,
        stacked: Bool
    ) -> some View {
        let layout: AnyLayout = stacked
            ? AnyLayout(VStackLayout(alignment: .leading, spacing: RapidTheme.Space.lg))
            : AnyLayout(HStackLayout(alignment: .top, spacing: RapidTheme.Space.xl))
        layout {
            VStack(alignment: .leading, spacing: 2) {
                Text(
                    String(
                        format: String(localized: "Compared with %1$d published %2$@"),
                        summary.observationCount,
                        summary.observationCount == 1
                            ? String(localized: "result")
                            : String(localized: "results")
                    )
                )
                .font(RapidFont.bodyEmphasis)
                .foregroundStyle(RapidTheme.textPrimary)
                Text(
                    String(
                        format: String(localized: "Same model, same %1$@"),
                        scope.macProfile.displayName
                    )
                )
                .font(RapidFont.secondary)
                .foregroundStyle(RapidTheme.textSecondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            statistic(
                value: String(format: "%.1f %@", median, unit),
                label: String(localized: "Median published")
            )
            if let low = summary.observedMinimum, let high = summary.observedMaximum {
                statistic(
                    value: String(format: "%.1f – %.1f", low, high),
                    label: String(localized: "Observed range")
                )
            }
            if let verdict = comparisonVerdict(median: median, unit: unit) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(verdict.title)
                        .font(RapidFont.bodyEmphasis)
                        .foregroundStyle(RapidTheme.textPrimary)
                    Text(verdict.detail)
                        .font(RapidFont.secondary)
                        .foregroundStyle(RapidTheme.textSecondary)
                }
                .frame(width: 180, alignment: .leading)
            }
        }
    }

    private func statistic(value: String, label: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(value)
                .font(.system(size: 15, design: .monospaced))
                .monospacedDigit()
                .foregroundStyle(RapidTheme.textPrimary)
            Text(label)
                .font(RapidFont.secondary)
                .foregroundStyle(RapidTheme.textSecondary)
        }
        .frame(width: 150, alignment: .leading)
        .accessibilityElement(children: .combine)
    }

    /// Where this run sits relative to the published median. Computed only
    /// from a server-supplied median and this run's own headline value, and
    /// phrased as a comparison, never as a rank.
    private func comparisonVerdict(
        median: Double,
        unit: String
    ) -> (title: String, detail: String)? {
        guard let headline = metrics.headline, let mine = Double(headline.value) else {
            return nil
        }
        let delta = mine - median
        let magnitude = abs(delta)
        guard median > 0 else { return nil }
        if magnitude / median < 0.05 {
            return (
                String(localized: "In line with others"),
                String(
                    format: String(localized: "%1$.1f %2$@ from the median"),
                    magnitude, unit
                )
            )
        }
        let isFaster = Self.isFaster(workload: metrics.workload, delta: delta)
        return (
            isFaster
                ? String(localized: "Faster than the median")
                : String(localized: "Slower than the median"),
            String(
                format: String(localized: "%1$.1f %2$@ %3$@ the median"),
                magnitude,
                unit,
                delta > 0 ? String(localized: "above") : String(localized: "below")
            )
        )
    }

    /// Throughput is better when larger; elapsed time is better when smaller.
    /// Keep that distinction in one testable rule so image/video comparisons
    /// cannot accidentally inherit the language-model direction.
    static func isFaster(workload: CommunityWorkload, delta: Double) -> Bool {
        workload == .llm ? delta > 0 : delta < 0
    }

    // MARK: - Publish invitation

    private var publishInvitation: some View {
        let copy = CommunityBenchmarkCopy.publishInvitation(branch: branch, scope: scope)
        return HStack(alignment: .top, spacing: RapidTheme.Space.md) {
            VStack(alignment: .leading, spacing: 3) {
                Text(copy.headline)
                    .font(RapidFont.bodyEmphasis)
                    .foregroundStyle(RapidTheme.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)
                Text(copy.body)
                    .font(RapidFont.secondary)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .padding(RapidTheme.Space.lg)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RapidTheme.brandPrimaryTint)
        .accessibilityElement(children: .combine)
    }

    // MARK: - Actions

    /// Which actions a completed Result screen offers, and whether each is
    /// currently held.
    ///
    /// A pure function, and the only thing the body renders from, so "is the
    /// exit on screen?" can be asserted directly. SwiftUI does not materialise
    /// per-control accessibility elements in a headless host, so a rendered
    /// tree cannot be asked — and the defect here was exactly a missing
    /// branch: `Benchmark another model` lived inside `if isPublished`, so a
    /// result the user chose not to publish had no way back to Ready.
    struct Action: Equatable, Identifiable, Sendable {
        enum Kind: String, Sendable { case publish, benchmarkAnother, runAgain }
        let kind: Kind
        let isEnabled: Bool

        var id: String { kind.rawValue }
        var accessibilityIdentifier: String {
            switch kind {
            case .publish: return "CommunityBenchmark.Result.Publish"
            case .benchmarkAnother: return "CommunityBenchmark.Result.Another"
            case .runAgain: return "CommunityBenchmark.Result.RunAgain"
            }
        }
    }

    // Pure, so it is callable from anywhere — including a test that has no
    // reason to hop to the main actor to ask what buttons a screen offers.
    nonisolated static func actions(
        isPublished: Bool,
        isCompleted: Bool,
        isPublishing: Bool
    ) -> [Action] {
        var actions: [Action] = []
        // Publish is the only action conditioned on state, because it is the
        // only one that would be a lie: a published run cannot be published a
        // second time, and an incomplete one has nothing to publish.
        if !isPublished {
            actions.append(
                Action(kind: .publish, isEnabled: !isPublishing && isCompleted)
            )
        }
        // Both exits are unconditional. They are held — not hidden — while an
        // upload is in flight, so the user can see where they will be able to
        // go, and a Run again cannot redirect the in-flight receipt onto
        // another run's scope.
        actions.append(Action(kind: .benchmarkAnother, isEnabled: !isPublishing))
        actions.append(Action(kind: .runAgain, isEnabled: !isPublishing))
        return actions
    }

    @ViewBuilder
    private func button(for action: Action) -> some View {
        switch action.kind {
        case .publish:
            Button(action: onPublish) {
                Label(
                    isPublishing
                        ? String(localized: "Publishing…")
                        : String(localized: "Publish to Community Benchmark"),
                    systemImage: "square.and.arrow.up"
                )
            }
            .buttonStyle(.rapidPrimary)
            .disabled(!action.isEnabled)
            .accessibilityIdentifier(action.accessibilityIdentifier)
        case .benchmarkAnother:
            Button(String(localized: "Benchmark another model"), action: onBenchmarkAnother)
                .buttonStyle(.rapidSecondary)
                .disabled(!action.isEnabled)
                .accessibilityIdentifier(action.accessibilityIdentifier)
        case .runAgain:
            Button(action: onRunAgain) {
                Label(String(localized: "Run again"), systemImage: "arrow.clockwise")
            }
            .buttonStyle(.rapidSecondary)
            .disabled(!action.isEnabled)
            .accessibilityIdentifier(action.accessibilityIdentifier)
        }
    }

    private var actions: some View {
        let offered = Self.actions(
            isPublished: isPublished,
            isCompleted: result.isCompleted,
            isPublishing: isPublishing
        )
        return Group {
            if isNarrow {
                // Publish is the long one — "Publish to Community Benchmark"
                // plus two more buttons and the details toggle does not fit in
                // 700pt, and a fixed HStack simply clipped the tail. Stacking
                // keeps every action full-width and fully readable.
                VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
                    ForEach(offered) { action in
                        button(for: action)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    technicalDetailsToggle
                        .padding(.top, RapidTheme.Space.xs)
                }
            } else {
                HStack(spacing: RapidTheme.Space.md) {
                    ForEach(offered) { action in
                        button(for: action)
                    }
                    Spacer(minLength: 0)
                    technicalDetailsToggle
                }
            }
        }
        .padding(RapidTheme.Space.xl)
    }

    private var technicalDetailsToggle: some View {
        Button {
            withAnimation(.easeInOut(duration: 0.15)) { showsTechnicalDetails.toggle() }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: showsTechnicalDetails ? "chevron.down" : "chevron.right")
                    .font(.system(size: 10, weight: .semibold))
                Text("Technical details")
            }
        }
        .buttonStyle(.rapidLink)
        .accessibilityIdentifier("CommunityBenchmark.Result.TechnicalDetails")
        .accessibilityValue(
            showsTechnicalDetails
                ? String(localized: "Expanded")
                : String(localized: "Collapsed")
        )
    }

    private var technicalDetails: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
            detailRow(String(localized: "Protocol"), scope.protocolName)
            detailRow(
                String(localized: "macOS"),
                result.machine.map { "\($0.os.version)" } ?? String(localized: "Unknown")
            )
            detailRow(
                String(localized: "Runtime"),
                "Rapid-MLX \(result.execution.runtime.rapidMLX) · MLX \(result.execution.runtime.mlx)"
            )
            detailRow(String(localized: "Config digest"), result.execution.configDigest)
            detailRow(String(localized: "Run ID"), result.id)
        }
        .padding(RapidTheme.Space.xl)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RapidTheme.surfaceCanvas)
    }

    private func detailRow(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: RapidTheme.Space.md) {
            Text(label)
                .font(RapidFont.secondary)
                .foregroundStyle(RapidTheme.textSecondary)
                .frame(width: 120, alignment: .leading)
            Text(value)
                .font(RapidFont.code)
                .foregroundStyle(RapidTheme.textPrimary)
                .textSelection(.enabled)
                .lineLimit(1)
                .truncationMode(.middle)
        }
        .accessibilityElement(children: .combine)
    }
}

extension Array {
    /// Fixed-size groups, used to lay supporting metrics out two per row.
    func chunked(into size: Int) -> [[Element]] {
        guard size > 0 else { return [self] }
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0..<Swift.min($0 + size, count)])
        }
    }
}
