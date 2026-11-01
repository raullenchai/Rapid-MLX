import SwiftUI

enum ShareComputeCreditsCopy {
    static func credits(_ formatted: String, bundle: Bundle = .main) -> String { String(localized: "\(formatted) credits", bundle: bundle) }
    static func smallCredit(negative: Bool, bundle: Bundle = .main) -> String { negative ? String(localized: "-<0.01 credits", bundle: bundle) : String(localized: "<0.01 credits", bundle: bundle) }
    static func caps(node: String, pool: String, bundle: Bundle = .main) -> String { String(localized: "Caps: \(node) per node · \(pool) per pool", bundle: bundle) }
    static func scope(_ count: Int, bundle: Bundle = .main) -> String { count == 1 ? String(localized: "Account-wide · 1 node", bundle: bundle) : String(localized: "Account-wide · \(count) nodes", bundle: bundle) }
    static func truncation(bundle: Bundle = .main) -> String { String(localized: "Showing the most recent windows only.", bundle: bundle) }
    static func shown(_ shown: Int, total: Int, bundle: Bundle = .main) -> String { String(localized: "Showing \(shown) of \(total) windows · totals use all of them", bundle: bundle) }
    static func windows(_ count: Int, bundle: Bundle = .main) -> String { count == 1 ? String(localized: "1 accounting window", bundle: bundle) : String(localized: "\(count) accounting windows", bundle: bundle) }
    static func summary(sessions: Int, duration: String, models: Int, bundle: Bundle = .main) -> String { String(localized: "\(sessions) connection periods · \(duration) online · \(models) models served", bundle: bundle) }
    static func row(requests: Int, credit: String, bundle: Bundle = .main) -> String { String(localized: "\(requests) requests · \(credit)", bundle: bundle) }
    static func accessibility(model: String, node: String, requests: Int, input: Int, output: Int, credit: String, accrued: String, status: String, bundle: Bundle = .main) -> String { String(localized: "\(model), node \(node), \(requests) requests, \(input) input and \(output) output tokens, credit \(credit), accrued \(accrued), status \(status)", bundle: bundle) }
}

// Retained for text-layout compatibility with existing callers.
enum ShareComputeLedgerLayout {
    enum Column: CaseIterable {
        case window, model, requests, tokens, credit, status
        func width(fonts: UIFontSettings = .shared) -> CGFloat {
            let base: CGFloat = switch self {
            case .window: 140
            case .model: 210
            case .requests: 82
            case .tokens: 120
            case .credit: 100
            case .status: 90
            }
            return fonts.scaled(base)
        }
    }
}

struct ShareComputeCreditsTab: View {
    let state: ShareComputeLedgerState
    let receipts: [ShareComputeReceipt]
    let localSummary: ShareComputeContributionSummary
    let savedKeyLabel: String?
    let canRefresh: Bool
    let isRefreshing: Bool
    let refreshHoldNote: String?
    let isNarrow: Bool
    let onOpenKeySettings: () -> Void
    let onRefresh: () -> Void
    var now: Date = Date()
    @Environment(\.openURL) private var openURL

    private var account: ShareComputeLedgerAccount? { state.account }
    private var month: String? { account?.latestAllowanceMonth }
    private var totals: ShareComputeLedgerMonthTotals? {
        guard let account, let month else { return nil }
        return account.totals(forAllowanceMonth: month)
    }
    static let visibleRowLimit = 6

    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.lg) {
            summaryCard
            if let account, !account.isEmpty { activityCard(account) }
            localActivity
        }
        .frame(maxWidth: 720, alignment: .leading)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var summaryCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Monthly credits").font(.rapid(size: 16, weight: .semibold))
                    if account != nil { Text(monthTitle).font(.rapid(size: 12)).foregroundStyle(.secondary) }
                }
                Spacer()
                if !state.needsKey { refreshControl }
            }
            if let account, !account.isEmpty {
                if let note = state.staleNote {
                    Text(note).font(.rapid(size: 12)).foregroundStyle(.secondary)
                        .accessibilityIdentifier("ShareCompute.Credits.Stale")
                } else if let hold = refreshHoldNote {
                    Text(hold).font(.rapid(size: 12)).foregroundStyle(.secondary)
                        .accessibilityIdentifier("ShareCompute.Credits.RefreshHold")
                }
                HStack(alignment: .top, spacing: 12) {
                    metric(String(localized: "Credits earned"), totals.map { credit($0.finalCredit) } ?? "—")
                        .accessibilityIdentifier("ShareCompute.Credits.MonthTotal")
                    metric(String(localized: "Requests served"), totals.map { ShareComputeCreditFormatter.count($0.requestCount) } ?? "—")
                    metric(String(localized: "Tokens (in / out)"), totals.map { "\(ShareComputeCreditFormatter.count($0.inputTokens)) / \(ShareComputeCreditFormatter.count($0.outputTokens))" } ?? "—")
                }
                Text(totals?.hasPendingWindows == true
                     ? String(localized: "1 credit = $1 of QuickSilver API usage. Pending windows finalise after the month closes.")
                     : String(localized: "1 credit = $1 of QuickSilver API usage."))
                    .font(.rapid(size: 11)).foregroundStyle(.secondary)
                let caps = account.caps
                if caps.nodeMonthlyUSD > 0 || caps.poolMonthlyUSD > 0 {
                    Text(ShareComputeCreditsCopy.caps(node: credit(caps.nodeMonthlyUSD), pool: credit(caps.poolMonthlyUSD)))
                        .font(.rapid(size: 11)).foregroundStyle(.secondary)
                }
                Text(scope(account)).font(.rapid(size: 11)).foregroundStyle(.secondary)
                cardFooter
            } else {
                stateNote
                if !state.needsKey { cardFooter }
            }
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .creditsGroupedCard()
    }

    private func scope(_ account: ShareComputeLedgerAccount) -> String {
        let nodes = account.nodeIDs.count
        let base = ShareComputeCreditsCopy.scope(nodes)
        return account.isTruncated ? base + " · " + ShareComputeCreditsCopy.truncation() : base
    }
    private func metric(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(label).font(.rapid(size: 12)).foregroundStyle(.secondary)
            Text(value).font(.rapid(size: 18, weight: .semibold)).monospacedDigit()
                .lineLimit(1)
        }.frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityElement(children: .combine)
    }
    private var refreshControl: some View {
        Button(action: onRefresh) {
            HStack(spacing: 5) {
                if isRefreshing { ProgressView().controlSize(.small) }
                else { Image(systemName: "arrow.clockwise") }
                Text("Refresh")
            }
        }
        .controlSize(.small).disabled(!canRefresh)
        .help(refreshHoldNote ?? "")
        .accessibilityLabel(String(localized: "Refresh the credit ledger"))
        .accessibilityHint(refreshHoldNote ?? "")
        .accessibilityIdentifier("ShareCompute.Credits.Refresh")
    }
    private var cardFooter: some View {
        HStack(spacing: 12) {
            if let savedKeyLabel {
                Text(savedKeyLabel).font(.rapid(size: 11, design: .monospaced)).foregroundStyle(.secondary).lineLimit(1)
            }
            Spacer(minLength: 0)
            if savedKeyLabel != nil {
                Button("Manage key…", action: onOpenKeySettings).buttonStyle(.plain)
                    .accessibilityLabel(String(localized: "Manage your QuickSilver key in Settings → Experimental → Share Compute"))
                    .accessibilityIdentifier("ShareCompute.Credits.ManageKey")
            }
            Button("QuickSilver balance ↗") { openURL(ShareComputeDestination.credits) }.buttonStyle(.plain)
                .accessibilityLabel(String(localized: "Open your QuickSilver account balance"))
                .accessibilityIdentifier("ShareCompute.Credits.Balance")
        }.font(.rapid(size: 11)).padding(.top, 4)
    }
    @ViewBuilder private var stateNote: some View {
        switch state {
        case .noKey: note("key", String(localized: "No QuickSilver key yet"), String(localized: "Add your QuickSilver key to see your credits."), "ShareCompute.Credits.NoKey", action: true, addKey: true)
        case .unauthorized: note("exclamationmark.triangle", ShareComputeLedgerError.unauthorized.displayMessage, String(localized: "Update your key in Settings to see your credits."), "ShareCompute.Credits.Unauthorized", action: true)
        case .loading: note("clock", String(localized: "Reading your ledger…"), String(localized: "Your credit activity will appear here."), "ShareCompute.Credits.Loading")
        case .rateLimited: note("hourglass", ShareComputeLedgerError.rateLimited.displayMessage, String(localized: "QuickSilver limits ledger checks per hour."), "ShareCompute.Credits.RateLimited")
        case .unavailable(let error): note("antenna.radiowaves.left.and.right.slash", error.displayMessage, String(localized: "Your local activity below is unaffected."), "ShareCompute.Credits.Unavailable")
        case .loadedEmpty: emptyNote
        case .refreshFailed:
            if let note = state.staleNote {
                Text(note).font(.rapid(size: 12)).foregroundStyle(.secondary)
                    .accessibilityIdentifier("ShareCompute.Credits.Stale")
            }
            emptyNote
        case .loaded, .refreshing: emptyNote
        }
    }
    private var emptyNote: some View {
        note("tray", String(localized: "No accounting windows yet"), String(localized: "Windows appear when your node serves metered requests."), "ShareCompute.Credits.Empty")
    }
    private func note(_ icon: String, _ title: String, _ detail: String, _ id: String, action: Bool = false, addKey: Bool = false) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: icon).foregroundStyle(.secondary).accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 4) {
                    Text(title).font(.rapid(size: 14, weight: .semibold))
                    Text(detail).font(.rapid(size: 12)).foregroundStyle(.secondary)
                }
            }.accessibilityElement(children: .combine).accessibilityIdentifier(id)
            if action {
                Button(action: onOpenKeySettings) {
                    if addKey { Text("Add key") } else { Text("Open Share Compute settings") }
                }
                    .accessibilityLabel(String(localized: "Open Settings → Experimental → Share Compute"))
                    .accessibilityIdentifier("ShareCompute.Credits.OpenKeySettings")
            }
        }
    }
    private func activityCard(_ account: ShareComputeLedgerAccount) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Recent activity").font(.rapid(size: 16, weight: .semibold)).padding(.bottom, 14)
            if !isNarrow {
                HStack(spacing: 10) {
                    Text("Time").frame(width: 100, alignment: .leading)
                    Text("Model").frame(maxWidth: .infinity, alignment: .leading)
                    Text("Requests").frame(width: 65, alignment: .leading)
                    Text("Credits").frame(width: 115, alignment: .leading)
                    Text("Status").frame(width: 80, alignment: .leading)
                }.font(.rapid(size: 11)).foregroundStyle(.secondary).padding(.bottom, 8).accessibilityHidden(true)
            }
            ForEach(account.windows.prefix(Self.visibleRowLimit)) { window in
                ShareComputeLedgerRowView(window: window, isNarrow: isNarrow)
            }
            Text(account.windows.count > Self.visibleRowLimit
                 ? ShareComputeCreditsCopy.shown(Self.visibleRowLimit, total: account.windows.count)
                 : ShareComputeCreditsCopy.windows(account.windows.count))
                .font(.rapid(size: 11)).foregroundStyle(.secondary).padding(.top, 10)
        }.padding(20).frame(maxWidth: .infinity, alignment: .leading).creditsGroupedCard()
    }
    private var localActivity: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("On this Mac").font(.rapid(size: 14, weight: .semibold))
            if receipts.isEmpty {
                Text("No connection periods recorded yet.").accessibilityIdentifier("ShareCompute.EmptyHistory")
            } else {
                Text(ShareComputeCreditsCopy.summary(sessions: localSummary.sessionCount, duration: ShareComputeDuration.total(localSummary.totalShared), models: localSummary.modelCount))
            }
            if account.map({ !$0.isEmpty }) == true {
                Text("Connection periods Rapid recorded on this Mac. Not a QuickSilver settlement record, and not matched to the windows above.")
                    .font(.rapid(size: 11)).foregroundStyle(.tertiary)
            } else {
                Text("Connection periods Rapid recorded on this Mac. Not a QuickSilver settlement record.")
                    .font(.rapid(size: 11)).foregroundStyle(.tertiary)
            }
        }.font(.rapid(size: 12)).foregroundStyle(.secondary)
            .accessibilityIdentifier("ShareCompute.Credits.LocalActivity")
    }
    private var monthTitle: String {
        guard let month else { return String(localized: "No windows yet") }
        let parser = DateFormatter(); parser.dateFormat = "yyyy-MM-dd"; parser.timeZone = TimeZone(identifier: "UTC")
        guard let date = parser.date(from: month) else { return month }
        let display = DateFormatter(); display.setLocalizedDateFormatFromTemplate("MMMMy"); display.timeZone = TimeZone(identifier: "UTC")
        return display.string(from: date)
    }
}

private func credit(_ value: Decimal) -> String {
    if value != 0 && abs(value) < Decimal(string: "0.01")! { return ShareComputeCreditsCopy.smallCredit(negative: value < 0) }
    let formatter = NumberFormatter(); formatter.numberStyle = .decimal
    formatter.minimumFractionDigits = 2; formatter.maximumFractionDigits = 2
    return ShareComputeCreditsCopy.credits(formatter.string(from: NSDecimalNumber(decimal: value)) ?? "0.00")
}

private extension View {
    func creditsGroupedCard() -> some View {
        background(RoundedRectangle(cornerRadius: RapidTheme.Radius.card, style: .continuous).fill(RapidTheme.surfaceRaised))
            .overlay(RoundedRectangle(cornerRadius: RapidTheme.Radius.card, style: .continuous).strokeBorder(RapidTheme.hairline, lineWidth: 1))
    }
}

struct ShareComputeLedgerRowView: View {
    let window: ShareComputeLedgerWindow
    var isNarrow = false
    var body: some View {
        Group {
            if isNarrow {
                VStack(alignment: .leading, spacing: 5) {
                    HStack { Text(periodLabel); Spacer(); statusTag }
                    HStack(spacing: 5) {
                        Text(verbatim: "\(modelLabel) · \(window.nodeID)").lineLimit(1).truncationMode(.middle)
                        Text(verbatim: "· \(ShareComputeCreditsCopy.row(requests: window.requestCount, credit: credit(window.finalCredit)))").fixedSize()
                    }.foregroundStyle(.secondary)
                }
            } else {
                HStack(spacing: 10) {
                    Text(periodLabel).frame(width: 100, alignment: .leading)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(modelLabel).lineLimit(1)
                        Text(window.nodeID).font(.rapid(size: 10, design: .monospaced)).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                    }.frame(maxWidth: .infinity, alignment: .leading)
                    Text(verbatim: "\(window.requestCount)").monospacedDigit().frame(width: 65, alignment: .leading)
                    Text(credit(window.finalCredit)).monospacedDigit().lineLimit(1).fixedSize().frame(width: 115, alignment: .leading)
                    statusTag.frame(width: 80, alignment: .leading)
                }
            }
        }.font(.rapid(size: 11)).padding(.vertical, 9)
            .frame(maxWidth: .infinity, alignment: .leading)
            .overlay(alignment: .bottom) { Rectangle().fill(RapidTheme.hairline).frame(height: 1) }
            .accessibilityElement(children: .combine).accessibilityLabel(accessibilityLabel)
            .accessibilityIdentifier("ShareCompute.Credits.Window.\(window.cursor)")
    }
    private var modelLabel: String { window.modelID ?? String(localized: "Model unavailable") }
    private var periodLabel: String {
        let day = DateFormatter(); day.setLocalizedDateFormatFromTemplate("MMMd")
        let time = DateFormatter(); time.timeStyle = .short
        return "\(day.string(from: window.periodStart)) · \(time.string(from: window.periodStart))"
    }
    private var statusTag: some View {
        let color: Color = switch window.status {
        case .credited: .green
        case .pending: .orange
        case .zero, .unknown: .secondary
        }
        return Text(window.status.title.capitalized).font(.rapid(size: 11, weight: .medium))
            .foregroundStyle(color).padding(.horizontal, 8).padding(.vertical, 4)
            .background(color.opacity(0.13), in: Capsule()).fixedSize()
    }
    private var accessibilityLabel: String {
        let accrued = window.accruedCredit.map(credit) ?? String(localized: "accrued amount unknown")
        return ShareComputeCreditsCopy.accessibility(model: modelLabel, node: window.nodeID, requests: window.requestCount, input: window.inputTokens, output: window.outputTokens, credit: credit(window.finalCredit), accrued: accrued, status: window.status.title)
    }
}
