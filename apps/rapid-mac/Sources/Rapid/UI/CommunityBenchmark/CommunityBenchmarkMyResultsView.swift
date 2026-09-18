import SwiftUI

/// Local benchmark history.
///
/// Three statuses only — Local, Published, Failed — because that is the full
/// set the client can prove. Publication status is read from the saved
/// receipt, never assumed from "we sent an upload".
struct CommunityBenchmarkMyResultsView: View {
    enum Status: String, CaseIterable, Identifiable, Hashable {
        case all
        case local
        case published
        case failed

        var id: String { rawValue }

        var title: String {
            switch self {
            case .all: return String(localized: "All")
            case .local: return String(localized: "Local")
            case .published: return String(localized: "Published")
            case .failed: return String(localized: "Failed")
            }
        }
    }

    let results: [CommunityBenchmarkResult]
    let receipts: [String: CommunityBenchmarkReceipt]
    let aliasForRepo: (String) -> String
    let workloadForResult: (CommunityBenchmarkResult) -> CommunityWorkload
    let contributor: CommunityBenchmarkContributor?
    /// Exact public totals for this pseudonym, read from the paginated
    /// contributions endpoint. Not `receipts.count`: a receipt can be missing
    /// (upload succeeded, local write failed) or lost to a reinstall, and this
    /// band must not under-report what is publicly attributed to the
    /// contributor.
    let publishedTotals: CommunityDataState<CommunityContributorTotals>
    /// Runs that completed on this Mac and were never published. Local only —
    /// labelled as such so it is never mistaken for a public contribution.
    let localOnlyCount: Int
    let sharingRunID: String?
    let onPublish: (CommunityBenchmarkResult) -> Void
    let onRunFirstBenchmark: () -> Void

    @State private var filter: Status = .all
    @State private var expandedID: String?

    private func status(for result: CommunityBenchmarkResult) -> Status {
        if receipts[result.id] != nil { return .published }
        return result.isCompleted ? .local : .failed
    }

    private var filtered: [CommunityBenchmarkResult] {
        guard filter != .all else { return results }
        return results.filter { status(for: $0) == filter }
    }

    private func count(_ status: Status) -> Int {
        status == .all ? results.count : results.filter { self.status(for: $0) == status }.count
    }

    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.lg) {
            if let contributor {
                identityStrip(contributor)
            }
            if results.isEmpty {
                emptyState
            } else {
                filterBar
                table
            }
        }
    }

    // MARK: - Identity

    private func identityStrip(_ contributor: CommunityBenchmarkContributor) -> some View {
        HStack(spacing: RapidTheme.Space.md) {
            CommunityContributorPortrait(contributor: contributor, size: 38)
            VStack(alignment: .leading, spacing: 2) {
                Text(contributor.displayName)
                    .font(.system(size: 15, weight: .semibold, design: .monospaced))
                    .foregroundStyle(RapidTheme.textPrimary)
                    .textSelection(.enabled)
                Text(totalsSentence)
                    .font(RapidFont.secondary)
                    .foregroundStyle(RapidTheme.textSecondary)
            }
            Spacer(minLength: 0)
            if let url = contributor.profileURL {
                Link(destination: url) {
                    HStack(spacing: 6) {
                        Text("View contributor page")
                        Image(systemName: "arrow.up.right.square").font(.system(size: 11))
                    }
                }
                .buttonStyle(.rapidSecondaryCompact)
                .accessibilityIdentifier("CommunityBenchmark.MyResults.ContributorProfile")
                .accessibilityLabel(contributor.profileURL.map { _ in
                    String(
                        format: String(localized: "View the contributor page for %1$@"),
                        contributor.displayName
                    )
                } ?? "")
            }
        }
        .padding(RapidTheme.Space.lg)
        .background(RapidTheme.brandPrimaryTint, in: RoundedRectangle(cornerRadius: RapidTheme.Radius.card))
        .overlay(alignment: .leading) {
            Rectangle()
                .fill(RapidTheme.brandPrimary)
                .frame(width: 3)
                .clipShape(
                    UnevenRoundedRectangle(
                        topLeadingRadius: RapidTheme.Radius.card,
                        bottomLeadingRadius: RapidTheme.Radius.card
                    )
                )
        }
        .accessibilityElement(children: .contain)
    }

    /// The public total when the server could supply an exact one, plus the
    /// local-only runs kept separate so the two are never added together.
    private var totalsSentence: String {
        let localClause = localOnlyCount > 0
            ? String(
                format: String(localized: " · %1$d kept local"),
                localOnlyCount
            )
            : ""
        switch publishedTotals {
        case .loading:
            return String(localized: "Counting your public contributions…") + localClause
        case .unavailable:
            // No exact server total, so no total is claimed.
            return String(localized: "Public contribution total unavailable") + localClause
        case let .ready(totals):
            let published = String(
                format: String(localized: "%1$d published %2$@ on rapidmlx.com"),
                totals.publishedRunCount,
                totals.publishedRunCount == 1
                    ? String(localized: "contribution")
                    : String(localized: "contributions")
            )
            return published + localClause
        }
    }

    // MARK: - Empty

    private var emptyState: some View {
        VStack(spacing: RapidTheme.Space.md) {
            Text("No benchmarks yet")
                .font(RapidFont.sectionTitle)
                .foregroundStyle(RapidTheme.textPrimary)
            Text("Your results appear here after you run one. They stay on this Mac unless you publish them.")
                .font(RapidFont.body)
                .foregroundStyle(RapidTheme.textSecondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
            Button(String(localized: "Run your first benchmark"), action: onRunFirstBenchmark)
                .buttonStyle(.rapidPrimary)
                .accessibilityIdentifier("CommunityBenchmark.MyResults.RunFirst")
        }
        .padding(RapidTheme.Space.xxl)
        .frame(maxWidth: .infinity)
        .background(
            RapidTheme.surfaceRaised,
            in: RoundedRectangle(cornerRadius: RapidTheme.Radius.card)
        )
        .overlay(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.card)
                .strokeBorder(RapidTheme.hairline)
        )
    }

    // MARK: - Filters

    private var filterBar: some View {
        HStack(spacing: RapidTheme.Space.sm) {
            ForEach(Status.allCases) { status in
                let isSelected = filter == status
                Button {
                    filter = status
                } label: {
                    HStack(spacing: 6) {
                        Text(status.title)
                        Text("\(count(status))")
                            .monospacedDigit()
                            .foregroundStyle(
                                isSelected ? RapidTheme.onBrandPrimary.opacity(0.75)
                                           : RapidTheme.textTertiary
                            )
                    }
                    .font(RapidFont.body)
                    .padding(.horizontal, RapidTheme.Space.md)
                    .padding(.vertical, 5)
                    .background(
                        isSelected ? RapidTheme.brandPrimary : RapidTheme.surfaceCanvas,
                        in: RoundedRectangle(cornerRadius: RapidTheme.Radius.segment)
                    )
                    .foregroundStyle(
                        isSelected ? RapidTheme.onBrandPrimary : RapidTheme.textPrimary
                    )
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
                .accessibilityIdentifier("CommunityBenchmark.MyResults.Filter.\(status.rawValue)")
            }
            Spacer(minLength: 0)
        }
    }

    // MARK: - Table

    private var table: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: RapidTheme.Space.md) {
                Text("MODEL").frame(maxWidth: .infinity, alignment: .leading)
                Text("WORKLOAD").frame(width: 90, alignment: .leading)
                Text("MAIN RESULT").frame(width: 150, alignment: .leading)
                Text("DATE").frame(width: 130, alignment: .leading)
                Text("STATUS").frame(width: 110, alignment: .leading)
                Text("ACTION").frame(width: 90, alignment: .trailing)
            }
            .font(RapidFont.groupLabel)
            .tracking(0.4)
            .foregroundStyle(RapidTheme.textTertiary)
            .padding(.horizontal, RapidTheme.Space.lg)
            .padding(.vertical, RapidTheme.Space.sm)

            Divider()

            ForEach(filtered) { result in
                row(result)
                Divider()
            }

            if filtered.isEmpty {
                Text("No results with this status.")
                    .font(RapidFont.body)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .padding(RapidTheme.Space.lg)
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
    }

    @ViewBuilder
    private func row(_ result: CommunityBenchmarkResult) -> some View {
        let rowStatus = status(for: result)
        let workload = workloadForResult(result)
        let metrics = CommunityBenchmarkMetrics.metricSet(for: result, workload: workload)
        let isExpanded = expandedID == result.id

        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: RapidTheme.Space.md) {
                Button {
                    withAnimation(.easeInOut(duration: 0.15)) {
                        expandedID = isExpanded ? nil : result.id
                    }
                } label: {
                    HStack(spacing: RapidTheme.Space.sm) {
                        Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(RapidTheme.textTertiary)
                        Text(aliasForRepo(result.repoID))
                            .font(RapidFont.bodyEmphasis)
                            .foregroundStyle(RapidTheme.textPrimary)
                            .lineLimit(1)
                            .truncationMode(.tail)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("CommunityBenchmark.MyResults.Expand.\(result.id)")
                .accessibilityLabel(
                    String(
                        format: String(localized: "%1$@, %2$@ details"),
                        aliasForRepo(result.repoID),
                        isExpanded ? String(localized: "hide") : String(localized: "show")
                    )
                )

                Text(workload.displayName)
                    .font(RapidFont.secondary)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .frame(width: 90, alignment: .leading)

                Group {
                    if let headline = metrics.headline {
                        Text(headline.combined)
                            .font(.system(size: 13, design: .monospaced))
                            .monospacedDigit()
                            .foregroundStyle(RapidTheme.textPrimary)
                    } else {
                        Text(metrics.incompleteStatus ?? String(localized: "No result"))
                            .font(RapidFont.secondary)
                            .foregroundStyle(RapidTheme.statusError)
                    }
                }
                .lineLimit(1)
                .frame(width: 150, alignment: .leading)

                Text(CommunityBenchmarkResult.formatCompletedAt(result.completedAt))
                    .font(RapidFont.secondary)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .lineLimit(1)
                    .frame(width: 130, alignment: .leading)

                statusChip(rowStatus)
                    .frame(width: 110, alignment: .leading)

                actionCell(result, status: rowStatus)
                    .frame(width: 90, alignment: .trailing)
            }
            .padding(.horizontal, RapidTheme.Space.lg)
            .padding(.vertical, RapidTheme.Space.md)

            if isExpanded {
                expandedDetail(result, metrics: metrics)
            }
        }
    }

    private func statusChip(_ status: Status) -> some View {
        let tone: Color
        let tint: Color
        let symbol: String
        switch status {
        case .published:
            tone = RapidTheme.statusReady
            tint = RapidTheme.statusReadyTint
            symbol = "checkmark.circle.fill"
        case .failed:
            tone = RapidTheme.statusError
            tint = RapidTheme.statusErrorTint
            symbol = "exclamationmark.circle.fill"
        case .local, .all:
            tone = RapidTheme.textSecondary
            tint = RapidTheme.surfaceCanvas
            symbol = "lock.fill"
        }
        return HStack(spacing: 5) {
            Image(systemName: symbol).font(.system(size: 10)).accessibilityHidden(true)
            Text(status == .all ? Status.local.title.uppercased() : status.title.uppercased())
                .font(.system(size: 10, weight: .semibold))
                .tracking(0.3)
        }
        .foregroundStyle(tone)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(tint, in: RoundedRectangle(cornerRadius: 5))
        .accessibilityElement(children: .combine)
    }

    @ViewBuilder
    private func actionCell(_ result: CommunityBenchmarkResult, status: Status) -> some View {
        switch status {
        case .published:
            if let receipt = receipts[result.id] {
                Link(String(localized: "View online"), destination: receipt.contributionURL)
                    .font(RapidFont.secondary)
                    .accessibilityLabel(receipt.contributionAccessibilityLabel)
                    .accessibilityIdentifier("CommunityBenchmark.Contributor.\(result.id)")
            }
        case .failed:
            EmptyView()
        case .local, .all:
            Button(
                sharingRunID == result.id
                    ? String(localized: "Publishing…")
                    : String(localized: "Publish")
            ) {
                onPublish(result)
            }
            .buttonStyle(.rapidLink)
            .font(RapidFont.secondary)
            .disabled(sharingRunID != nil)
            .accessibilityIdentifier("CommunityBenchmark.Share.\(result.id)")
        }
    }

    private func expandedDetail(
        _ result: CommunityBenchmarkResult,
        metrics: CommunityBenchmarkMetrics.MetricSet
    ) -> some View {
        HStack(alignment: .top, spacing: RapidTheme.Space.xxl) {
            VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
                Text("SECONDARY METRICS")
                    .font(RapidFont.groupLabel)
                    .foregroundStyle(RapidTheme.textTertiary)
                ForEach(metrics.supporting) { metric in
                    HStack(spacing: RapidTheme.Space.md) {
                        Text(metric.label)
                            .font(RapidFont.secondary)
                            .foregroundStyle(RapidTheme.textSecondary)
                            .frame(width: 150, alignment: .leading)
                        Text(metric.combined)
                            .font(RapidFont.metric)
                            .foregroundStyle(RapidTheme.textPrimary)
                    }
                }
                if metrics.supporting.isEmpty {
                    Text("No secondary metrics were recorded for this run.")
                        .font(RapidFont.secondary)
                        .foregroundStyle(RapidTheme.textSecondary)
                }
            }
            VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
                Text("ENVIRONMENT")
                    .font(RapidFont.groupLabel)
                    .foregroundStyle(RapidTheme.textTertiary)
                if let machine = result.machine {
                    detail(
                        String(localized: "Mac"),
                        "\(machine.profile.chip) · \(machine.profile.memoryGib) GB"
                    )
                    detail(String(localized: "macOS"), machine.os.version)
                }
                detail(
                    String(localized: "Rapid-MLX"),
                    "\(result.execution.runtime.rapidMLX) · MLX \(result.execution.runtime.mlx)"
                )
            }
            Spacer(minLength: 0)
        }
        .padding(RapidTheme.Space.lg)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RapidTheme.surfaceCanvas)
    }

    private func detail(_ label: String, _ value: String) -> some View {
        HStack(spacing: RapidTheme.Space.md) {
            Text(label)
                .font(RapidFont.secondary)
                .foregroundStyle(RapidTheme.textSecondary)
                .frame(width: 90, alignment: .leading)
            Text(value)
                .font(RapidFont.metric)
                .foregroundStyle(RapidTheme.textPrimary)
                .lineLimit(1)
                .truncationMode(.tail)
        }
        .accessibilityElement(children: .combine)
    }
}
