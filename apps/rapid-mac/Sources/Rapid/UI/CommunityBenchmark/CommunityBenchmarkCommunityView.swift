import SwiftUI

/// The Community tab.
///
/// One reading path: social proof, then the evidence this Mac can use, then
/// the contribution this Mac can make, then the external ranked leaderboard.
/// The ranked destination is deliberately last and quiet — it leaves the app,
/// and it is a different data product from the observations above it.
struct CommunityBenchmarkCommunityView: View {
    let macProfile: CommunityMacProfile
    let pulse: CommunityDataState<CommunityPulse>
    let table: CommunityDataState<[CommunityObservationRow]>
    let coverage: CommunityDataState<[CommunityCoverageGap]>
    @Binding var workload: CommunityWorkload
    let metric: CommunityMetric
    /// True when the window is too narrow for the side-by-side layout. The
    /// contribution task then comes before the longer table.
    let isNarrow: Bool
    /// This installation's pseudonym, once the service has issued one. Nil
    /// before the first publication — there is no identity to show yet.
    var youContributor: CommunityBenchmarkContributor?
    let leaderboardURL: URL
    let onRunModel: (String) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
            pulseBand

            if isNarrow {
                contributionArea
                observationsArea
            } else {
                HStack(alignment: .top, spacing: RapidTheme.Space.xl) {
                    observationsArea
                        .frame(maxWidth: .infinity, alignment: .leading)
                    contributionArea
                        .frame(width: 340, alignment: .leading)
                }
            }

            CommunityLeaderboardLinkRow(destination: leaderboardURL)
        }
    }

    // MARK: - Pulse

    /// Two compact levels, left-aligned and width-capped. Never a hero: it
    /// establishes that a community exists and then gets out of the way.
    @ViewBuilder
    private var pulseBand: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Community pulse")
                    .font(RapidFont.bodyEmphasis)
                    .foregroundStyle(RapidTheme.textPrimary)
                Text("Every character is a real Mac adding a point of evidence.")
                    .font(RapidFont.secondary)
                    .foregroundStyle(RapidTheme.textSecondary)
            }

            switch pulse {
            case .loading:
                HStack(spacing: RapidTheme.Space.sm) {
                    ProgressView().controlSize(.small)
                    Text("Loading community activity…")
                        .font(RapidFont.secondary)
                        .foregroundStyle(RapidTheme.textSecondary)
                }
            case let .unavailable(reason):
                Text(reason.message)
                    .font(RapidFont.secondary)
                    .foregroundStyle(RapidTheme.textTertiary)
            case let .ready(value):
                pulseProof(value)
            }
        }
        .frame(maxWidth: 900, alignment: .leading)
        .padding(.bottom, RapidTheme.Space.md)
        .overlay(alignment: .bottom) {
            Rectangle().fill(RapidTheme.hairline).frame(height: 1)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(String(localized: "Community pulse"))
    }

    private func pulseProof(_ value: CommunityPulse) -> some View {
        Group {
            if isNarrow {
                VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
                    avatarCluster(value)
                    pulseStatistics(value)
                }
            } else {
                HStack(alignment: .center, spacing: 40) {
                    avatarCluster(value)
                    pulseStatistics(value)
                    Spacer(minLength: 0)
                }
            }
        }
    }

    private func pulseStatistics(_ value: CommunityPulse) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            statisticsSentence(value)
            if let last = value.lastContributionAt {
                Text(
                    String(
                        format: String(localized: "Last contribution %1$@"),
                        RelativeDateTimeFormatter().localizedString(for: last, relativeTo: Date())
                    )
                )
                .font(RapidFont.secondary)
                .foregroundStyle(RapidTheme.textTertiary)
            }
        }
    }

    /// Fixed-width cluster: at most five community tiles, one overflow tile,
    /// and this installation's own marker. Its width does not change as the
    /// community grows, which is the whole point of the cap.
    private func avatarCluster(_ value: CommunityPulse) -> some View {
        HStack(spacing: RapidTheme.Space.sm) {
            HStack(spacing: -7) {
                ForEach(value.renderedContributors, id: \.slug) { contributor in
                    CommunityContributorPortrait(
                        contributor: contributor, size: 30, cornerRadius: 7
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 7)
                            .strokeBorder(RapidTheme.surfaceRaised, lineWidth: 2)
                    )
                }
                if let overflow = value.overflowLabel {
                    Text(overflow)
                        .font(.system(size: 11, weight: .medium))
                        .monospacedDigit()
                        .foregroundStyle(RapidTheme.textSecondary)
                        .frame(width: 52, height: 30)
                        .background(
                            RapidTheme.surfaceCanvas,
                            in: RoundedRectangle(cornerRadius: 7)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 7)
                                .strokeBorder(RapidTheme.surfaceRaised, lineWidth: 2)
                        )
                        .accessibilityLabel(
                            String(
                                format: String(localized: "%1$d more contributors"),
                                value.overflowCount
                            )
                        )
                }
            }
            // Only shown once the service has issued this installation a
            // pseudonym. Before the first publication there is no public
            // identity to place in the cluster, and an empty ringed tile reads
            // as a broken portrait rather than as "not yet".
            if let you = youContributor {
                HStack(spacing: 6) {
                    // This installation's own portrait, from the same mapping
                    // as everyone else's — a mascot here would be the one face
                    // in the cluster that identifies nobody.
                    CommunityContributorPortrait(
                        contributor: you, size: 30, cornerRadius: 7
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 7)
                            .strokeBorder(RapidTheme.brandPrimary, lineWidth: 1.5)
                    )
                    Text("YOU")
                        .font(.system(size: 10, weight: .semibold))
                        .tracking(0.4)
                        .foregroundStyle(RapidTheme.brandPrimaryDeep)
                }
                .padding(.leading, RapidTheme.Space.sm)
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(
                    String(
                        format: String(localized: "This Mac, published as %1$@"),
                        you.displayName
                    )
                )
            }
        }
        .fixedSize()
    }

    /// "24 contributors · 63 published runs · 17 models" — one sentence-like
    /// group rather than three separate columns.
    private func statisticsSentence(_ value: CommunityPulse) -> some View {
        // The public feed carries only the newest runs, so its totals are
        // floors. Saying "24 contributors" when the real number is larger
        // would be a false census; "at least 24" is what the data supports.
        HStack(spacing: RapidTheme.Space.sm) {
            statistic(
                value.contributorCount,
                String(localized: "contributors"),
                isBounded: value.isBounded
            )
            separator
            statistic(
                value.publishedRunCount,
                String(localized: "published runs"),
                isBounded: value.isBounded
            )
            separator
            statistic(
                value.modelCount,
                String(localized: "models"),
                isBounded: value.isBounded
            )
        }
        .fixedSize()
    }

    private func statistic(_ count: Int, _ label: String, isBounded: Bool = false) -> some View {
        HStack(spacing: 5) {
            if isBounded {
                Text("at least")
                    .font(RapidFont.secondary)
                    .foregroundStyle(RapidTheme.textTertiary)
            }
            Text("\(count)")
                .font(.system(size: 15, weight: .semibold))
                .monospacedDigit()
                .foregroundStyle(RapidTheme.textPrimary)
            Text(label)
                .font(RapidFont.body)
                .foregroundStyle(RapidTheme.textSecondary)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(isBounded ? "at least " : "")\(count) \(label)")
    }

    private var separator: some View {
        Text("·")
            .font(RapidFont.body)
            .foregroundStyle(RapidTheme.textTertiary)
            .accessibilityHidden(true)
    }

    // MARK: - Observations (primary)

    private var observationsArea: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Performance on Macs like yours")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(RapidTheme.textPrimary)
                Text("Published Rapid-MLX Desktop results for the selected Mac profile. These are observations, not rankings.")
                    .font(RapidFont.body)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack(spacing: RapidTheme.Space.md) {
                Text(macProfile.displayName)
                    .font(RapidFont.body)
                    .padding(.horizontal, RapidTheme.Space.md)
                    .padding(.vertical, 5)
                    .background(
                        RapidTheme.surfaceCanvas,
                        in: RoundedRectangle(cornerRadius: RapidTheme.Radius.segment)
                    )
                    .accessibilityLabel(
                        String(
                            format: String(localized: "Mac profile %1$@"),
                            macProfile.displayName
                        )
                    )

                Picker(String(localized: "Workload"), selection: $workload) {
                    ForEach(CommunityWorkload.allCases, id: \.self) { candidate in
                        Text(candidate.displayName).tag(candidate)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(width: 200)
                .accessibilityIdentifier("CommunityBenchmark.Community.Workload")

                Text(
                    String(
                        format: String(localized: "Metric: %1$@"),
                        metric.displayName
                    )
                )
                .font(RapidFont.secondary)
                .foregroundStyle(RapidTheme.textSecondary)

                Spacer(minLength: 0)
            }

            observationsTable
        }
    }

    @ViewBuilder
    private var observationsTable: some View {
        switch table {
        case .loading:
            CommunityUnavailableBand(
                title: String(localized: "Loading community observations"),
                message: String(localized: "Published results for this Mac profile are on the way."),
                isLoading: true
            )
        case let .unavailable(reason):
            CommunityUnavailableBand(
                title: String(localized: "Community data unavailable"),
                message: reason.message
            )
        case .ready(let rows) where rows.isEmpty:
            // An empty list is NOT proof that nothing has been published. The
            // public feed is bounded to its newest runs, and a directory that
            // cannot establish absence reports `.unavailable` instead — but
            // this branch must also refuse to claim "yours would be the first"
            // in case some other source ever hands back an empty array.
            CommunityUnavailableBand(
                title: String(localized: "No results to show for this selection"),
                message: String(
                    format: String(
                        localized: "No published %1$@ results for an %2$@ came back. That does not mean there are none — try another workload, or check again later."
                    ),
                    workload.displayName,
                    macProfile.displayName
                )
            )
        case let .ready(rows):
            ScrollView(.horizontal, showsIndicators: true) {
                VStack(alignment: .leading, spacing: 0) {
                    HStack(spacing: RapidTheme.Space.md) {
                        Text("MODEL").frame(maxWidth: .infinity, alignment: .leading)
                        Text("OBSERVED RANGE").frame(width: 130, alignment: .trailing)
                        Text("MEDIAN").frame(width: 80, alignment: .trailing)
                        Text("OBSERVATIONS").frame(width: 104, alignment: .trailing)
                    }
                    .font(RapidFont.groupLabel)
                    .tracking(0.4)
                    .foregroundStyle(RapidTheme.textTertiary)
                    .padding(.vertical, RapidTheme.Space.sm)

                    Divider()

                    ForEach(rows) { row in
                        observationRow(row)
                        Divider()
                    }

                    Text("Median comes from published observations. A dash means the bounded feed does not provide an observed range.")
                        .font(RapidFont.secondary)
                        .foregroundStyle(RapidTheme.textTertiary)
                        .padding(.top, RapidTheme.Space.md)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .frame(minWidth: 620, alignment: .leading)
            }
        }
    }

    private func observationRow(_ row: CommunityObservationRow) -> some View {
        HStack(spacing: RapidTheme.Space.md) {
            HStack(spacing: RapidTheme.Space.sm) {
                Text(row.modelAlias)
                    .font(RapidFont.bodyEmphasis)
                    .foregroundStyle(RapidTheme.textPrimary)
                    .lineLimit(1)
                    .truncationMode(.tail)
                if row.summary.includesYours {
                    Text("INCLUDES YOURS")
                        .font(.system(size: 9, weight: .semibold))
                        .tracking(0.3)
                        .foregroundStyle(RapidTheme.brandPrimaryDeep)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(
                            RapidTheme.brandPrimaryTint,
                            in: RoundedRectangle(cornerRadius: 4)
                        )
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            Text(rangeText(row.summary))
                .font(RapidFont.metric)
                .foregroundStyle(RapidTheme.textSecondary)
                .frame(width: 130, alignment: .trailing)

            Text(
                row.summary.median.map {
                    metricText($0, unit: row.summary.unit)
                } ?? "—"
            )
                .font(.system(size: 13, weight: .medium, design: .monospaced))
                .monospacedDigit()
                .foregroundStyle(RapidTheme.textPrimary)
                .frame(width: 80, alignment: .trailing)

            Text(
                row.summary.isBounded
                    ? String(
                        format: String(localized: "at least %1$d"),
                        row.summary.observationCount
                    )
                    : "\(row.summary.observationCount)"
            )
                .font(RapidFont.metric)
                .foregroundStyle(RapidTheme.textSecondary)
                .frame(width: 104, alignment: .trailing)
        }
        .padding(.vertical, RapidTheme.Space.md)
        .accessibilityElement(children: .combine)
    }

    private func rangeText(_ summary: CommunityObservationSummary) -> String {
        guard let low = summary.observedMinimum else { return "—" }
        guard let high = summary.observedMaximum, high != low else {
            return metricText(low, unit: summary.unit)
        }
        let values = String(format: "%.1f – %.1f", low, high)
        guard let unit = summary.unit, !unit.isEmpty else { return values }
        return "\(values) \(unit)"
    }

    private func metricText(_ value: Double, unit: String?) -> String {
        let number = String(format: "%.1f", value)
        guard let unit, !unit.isEmpty else { return number }
        return "\(number) \(unit)"
    }

    // MARK: - Contribution (secondary)

    private var contributionArea: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Where your Mac can help")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(RapidTheme.textPrimary)
                Text(
                    String(
                        format: String(localized: "Same Mac profile as the table — %1$@."),
                        macProfile.displayName
                    )
                )
                .font(RapidFont.body)
                .foregroundStyle(RapidTheme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            }

            switch coverage {
            case .loading:
                CommunityUnavailableBand(
                    title: String(localized: "Loading coverage"),
                    message: String(localized: "Finding models that still need results on this Mac."),
                    isLoading: true
                )
            case let .unavailable(reason):
                CommunityUnavailableBand(
                    title: String(localized: "Coverage unavailable"),
                    message: reason.message
                )
            case let .ready(gaps) where gaps.isEmpty:
                // The feed carries only the newest runs, so "no thin pairings
                // in it" is not "every catalogue model is covered". The old
                // copy made that second claim, which the data cannot support:
                // a model with no results at all is exactly the model that
                // never appears in the feed.
                CommunityUnavailableBand(
                    title: CommunityBenchmarkCopy.coverageAllCovered.title,
                    message: CommunityBenchmarkCopy.coverageAllCovered.message
                )
            case let .ready(gaps):
                coverageList(gaps)
            }
        }
    }

    private func coverageList(_ gaps: [CommunityCoverageGap]) -> some View {
        let mission = gaps.first { $0.isFirstResultOpportunity } ?? gaps[0]
        let rest = gaps.filter { $0.id != mission.id }
        return VStack(alignment: .leading, spacing: 0) {
            missionCard(mission)
            ForEach(rest.prefix(3)) { gap in
                Divider()
                gapRow(gap)
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
        .clipShape(RoundedRectangle(cornerRadius: RapidTheme.Radius.card))
    }

    private func missionCard(_ gap: CommunityCoverageGap) -> some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
            HStack(spacing: 6) {
                Image(systemName: gap.isFirstResultOpportunity ? "flag" : "chart.bar")
                    .font(.system(size: 11))
                    .accessibilityHidden(true)
                Text(
                    gap.isFirstResultOpportunity
                        ? String(localized: "FIRST RESULT NEEDED")
                        : String(localized: "MORE RESULTS NEEDED")
                )
                .font(.system(size: 11, weight: .semibold))
                .tracking(0.5)
            }
            .foregroundStyle(RapidTheme.brandPrimaryDeep)

            Text(gap.modelAlias)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(RapidTheme.textPrimary)
                .lineLimit(1)
                .truncationMode(.tail)

            Text(missionDescription(gap))
                .font(RapidFont.secondary)
                .foregroundStyle(RapidTheme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)

            Button {
                onRunModel(gap.modelAlias)
            } label: {
                Label(String(localized: "Run benchmark"), systemImage: "play.fill")
            }
            .buttonStyle(.rapidPrimaryCompact)
            .accessibilityIdentifier("CommunityBenchmark.Community.RunMission")
        }
        .padding(RapidTheme.Space.lg)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RapidTheme.brandPrimaryTint)
    }

    private func missionDescription(_ gap: CommunityCoverageGap) -> String {
        let coverage = gap.isFirstResultOpportunity
            ? String(
                format: String(localized: "No one has published this model on an %1$@ yet."),
                macProfile.displayName
            )
            : gap.isBounded
                ? String(
                    format: String(localized: "At least %1$d recent published %2$@ on an %3$@."),
                    gap.observationCount,
                    gap.observationCount == 1
                        ? String(localized: "result")
                        : String(localized: "results"),
                    macProfile.displayName
                )
                : String(
                    format: String(localized: "%1$d published %2$@ on an %3$@ so far."),
                    gap.observationCount,
                    gap.observationCount == 1
                        ? String(localized: "result")
                        : String(localized: "results"),
                    macProfile.displayName
                )
        let readiness = gap.isDownloaded
            ? String(localized: "Already downloaded.")
            : gap.downloadSizeGB.map {
                String(format: String(localized: "%1$.1f GB download."), $0)
            } ?? String(localized: "Downloads when the benchmark starts.")
        return "\(coverage) \(readiness)"
    }

    private func gapRow(_ gap: CommunityCoverageGap) -> some View {
        HStack(spacing: RapidTheme.Space.md) {
            VStack(alignment: .leading, spacing: 2) {
                Text(gap.modelAlias)
                    .font(RapidFont.bodyEmphasis)
                    .foregroundStyle(
                        gap.fitsThisMac ? RapidTheme.textPrimary : RapidTheme.textSecondary
                    )
                    .lineLimit(1)
                    .truncationMode(.tail)
                Text(gapDetail(gap))
                    .font(RapidFont.secondary)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .lineLimit(1)
            }
            Spacer(minLength: RapidTheme.Space.sm)
            Button(
                gap.fitsThisMac
                    ? String(localized: "Run")
                    : String(localized: "Run anyway")
            ) {
                onRunModel(gap.modelAlias)
            }
            .buttonStyle(gap.fitsThisMac ? AnyButtonStyle(.rapidSecondaryCompact) : AnyButtonStyle(.rapidLink))
            .font(RapidFont.secondary)
            .accessibilityIdentifier("CommunityBenchmark.Community.Run.\(gap.modelAlias)")
        }
        .padding(RapidTheme.Space.md)
    }

    private func gapDetail(_ gap: CommunityCoverageGap) -> String {
        if !gap.fitsThisMac {
            let needed = gap.requiredMemoryGB.map {
                String(format: String(localized: "needs about %1$d GB"), $0)
            } ?? String(localized: "needs more memory")
            return String(
                format: String(localized: "May not fit · %1$@"),
                needed
            )
        }
        let count = gap.observationCount == 0
            ? String(localized: "No results yet")
            : String(
                format: String(localized: "Only %1$d %2$@"),
                gap.observationCount,
                gap.observationCount == 1
                    ? String(localized: "observation")
                    : String(localized: "observations")
            )
        // Kept short: this row is a 340pt secondary column, and a longer
        // trailing clause truncates before the count is readable.
        return gap.isDownloaded
            ? String(format: String(localized: "%1$@ · downloaded"), count)
            : count
    }
}

/// Type-erased button style so a row can pick between two styles inline.
struct AnyButtonStyle: ButtonStyle {
    private let makeBodyClosure: (Configuration) -> AnyView

    init<Style: ButtonStyle>(_ style: Style) {
        makeBodyClosure = { configuration in
            AnyView(style.makeBody(configuration: configuration))
        }
    }

    func makeBody(configuration: Configuration) -> some View {
        makeBodyClosure(configuration)
    }
}
