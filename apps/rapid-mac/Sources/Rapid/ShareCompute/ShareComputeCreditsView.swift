import SwiftUI

/// "What have I earned?" — now backed by QuickSilver's contributor ledger.
///
/// ## The ownership split, made literal
///
/// The dark panel holds the LEDGER: accounting windows QuickSilver metered and
/// settled. The amber panel holds the MONTH summary derived from them. Rapid's
/// own connection receipts survive as a clearly fenced local-activity strip at
/// the bottom of the dark panel — never interleaved with ledger rows, never
/// correlated to one.
///
/// That last rule is not stylistic. A local receipt and a ledger window share
/// no identifier: a receipt is "this Mac was connected from 14:02 to 15:44",
/// a window is "node X served 7 requests for model Y between 04:00 and 04:30".
/// Joining them would require guessing, and a guess presented next to a dollar
/// figure is the worst kind of wrong. So the two sit in separate sections with
/// separate headings, and the local one never shows a credit amount.
///
/// ## Account-wide, not this-Mac
///
/// The read key is account-scoped, so these windows span every node on the
/// account. A user contributing from two machines would otherwise read the
/// other machine's credit as this one's — hence "across N nodes" in the header
/// and node ids on every row.
struct ShareComputeCreditsTab: View {
    let state: ShareComputeLedgerState
    let receipts: [ShareComputeReceipt]
    let localSummary: ShareComputeContributionSummary
    @Binding var readKeyDraft: String
    let readKeyRejection: ShareComputeReadKeyRejection?
    /// `qsprk-…a8c1`. Never the whole key.
    let savedKeyLabel: String?
    /// Whether Refresh is currently permitted. False while a walk runs, while
    /// rate-limited, and inside the minimum interval — the control is disabled
    /// rather than silently ignoring the click.
    let canRefresh: Bool
    let isRefreshing: Bool
    /// Why Refresh is held, when it is. A disabled button with no explanation
    /// reads as a bug.
    let refreshHoldNote: String?
    let isNarrow: Bool
    let onSaveReadKey: () -> Void
    let onRemoveReadKey: () -> Void
    let onRefresh: () -> Void
    var now: Date = Date()

    @Environment(\.openURL) private var openURL

    private var account: ShareComputeLedgerAccount? { state.account }

    private var month: String? {
        account?.latestAllowanceMonth
    }

    private var totals: ShareComputeLedgerMonthTotals? {
        guard let account, let month else { return nil }
        // Summed over EVERY fetched row for the bucket, not the first page.
        return account.totals(forAllowanceMonth: month)
    }

    var body: some View {
        ShareComputeWorkbenchFrame {
            if isNarrow {
                VStack(spacing: 0) {
                    ledgerPanel
                    summaryPanel
                }
            } else {
                HStack(alignment: .top, spacing: 0) {
                    ledgerPanel
                    summaryPanel.frame(width: 400)
                }
                .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    // MARK: - Ledger (left, dark)

    private var ledgerPanel: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            ledgerBody
            localActivity
        }
        .padding(isNarrow ? 20 : 26)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(RapidTheme.surfaceBand)
    }

    private var header: some View {
        HStack(alignment: .top, spacing: RapidTheme.Space.sm) {
            VStack(alignment: .leading, spacing: 4) {
                ShareComputeEyebrow(text: "QuickSilver · Contributor ledger", size: 11)
                Text("Credit ledger")
                    .font(.system(size: isNarrow ? 21 : 25, weight: .semibold))
                    .foregroundStyle(RapidTheme.bandInk)
                Text(subtitle)
                    .font(.system(size: 12))
                    .foregroundStyle(RapidTheme.bandInkSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: RapidTheme.Space.sm)
            if !state.needsReadKey { refreshControl }
        }
    }

    /// Says WHOSE ledger this is. Account-wide is the surprising part, so it
    /// leads.
    private var subtitle: String {
        guard let account, !account.isEmpty else {
            return String(localized: "Accounting windows metered by QuickSilver across every node on your account.")
        }
        let nodes = account.nodeIDs.count
        let base = nodes == 1
            ? String(localized: "Accounting windows metered by QuickSilver for 1 node on your account.")
            : String(
                format: String(localized: "Accounting windows metered by QuickSilver across %d nodes on your account."),
                nodes
            )
        guard account.isTruncated else { return base }
        return base + " " + String(localized: "Showing the most recent windows only.")
    }

    private var refreshControl: some View {
        Button(action: onRefresh) {
            HStack(spacing: 6) {
                if isRefreshing {
                    ProgressView().controlSize(.small)
                } else {
                    Image(systemName: "arrow.clockwise").font(.system(size: 11, weight: .semibold))
                }
                Text("Refresh")
                    .font(.system(size: 11, weight: .medium))
            }
            .foregroundStyle(RapidTheme.bandInkSecondary)
            .padding(.horizontal, 10)
            .frame(height: 26)
            .background(RapidTheme.bandControl, in: RoundedRectangle(cornerRadius: 6))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!canRefresh)
        .opacity(canRefresh ? 1 : RapidTheme.disabledOpacity)
        .help(refreshHoldNote ?? "")
        .accessibilityLabel(String(localized: "Refresh the credit ledger"))
        .accessibilityHint(refreshHoldNote ?? "")
        .accessibilityIdentifier("ShareCompute.Credits.Refresh")
    }

    @ViewBuilder
    private var ledgerBody: some View {
        switch state {
        case .noReadKey:
            note(
                icon: "key",
                title: String(localized: "No read key yet"),
                detail: String(localized: "Add a QuickSilver read key to see the accounting windows your nodes earned credit for."),
                identifier: "ShareCompute.Credits.NoKey"
            )
        case .loading:
            note(
                icon: "clock",
                title: String(localized: "Reading your ledger…"),
                detail: nil,
                identifier: "ShareCompute.Credits.Loading"
            )
        case .unauthorized:
            note(
                icon: "exclamationmark.triangle",
                title: ShareComputeLedgerError.unauthorized.displayMessage,
                detail: String(localized: "Create a new read key in the QuickSilver dashboard and paste it here."),
                identifier: "ShareCompute.Credits.Unauthorized"
            )
        case .rateLimited:
            note(
                icon: "hourglass",
                title: ShareComputeLedgerError.rateLimited.displayMessage,
                detail: String(localized: "QuickSilver allows a limited number of ledger checks per hour."),
                identifier: "ShareCompute.Credits.RateLimited"
            )
        case .unavailable(let error):
            note(
                icon: "antenna.radiowaves.left.and.right.slash",
                title: error.displayMessage,
                detail: String(localized: "Your local activity below is measured on this Mac and is unaffected."),
                identifier: "ShareCompute.Credits.Unavailable"
            )
        case .loadedEmpty:
            note(
                icon: "tray",
                title: String(localized: "No accounting windows yet"),
                detail: String(localized: "QuickSilver opens a window once your node serves metered requests. An empty ledger is a normal reading, not an error."),
                identifier: "ShareCompute.Credits.Empty"
            )
        case .loaded(let account), .refreshing(let account), .refreshFailed(let account, _):
            VStack(alignment: .leading, spacing: 0) {
                if let hold = refreshHoldNote, state.staleNote == nil {
                    Text(hold)
                        .font(.system(size: 11))
                        .foregroundStyle(RapidTheme.bandInkSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.bottom, 10)
                        .accessibilityIdentifier("ShareCompute.Credits.RefreshHold")
                }
                if let staleNote = state.staleNote {
                    Text(staleNote)
                        .font(.system(size: 11))
                        .foregroundStyle(RapidTheme.bandInkSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.bottom, 10)
                        .accessibilityIdentifier("ShareCompute.Credits.Stale")
                }
                columnHeaders
                ForEach(account.windows.prefix(Self.visibleRowLimit)) { window in
                    ShareComputeLedgerRowView(window: window, isNarrow: isNarrow)
                }
                footer(account)
            }
        }
    }

    /// Rows shown before the list stops growing the page. The account walk
    /// fetches everything (totals depend on it); the TABLE only needs enough
    /// to be useful without turning the tab into an unbounded scroll.
    static let visibleRowLimit = 6

    private var columnHeaders: some View {
        Group {
            if !isNarrow {
                HStack(spacing: 0) {
                    headerLabel(String(localized: "Window"), width: 140)
                    headerLabel(String(localized: "Node · Model"), width: 210)
                    headerLabel(String(localized: "Requests"), width: 70)
                    headerLabel(String(localized: "Tokens in/out"), width: 100)
                    headerLabel(String(localized: "Credit"), width: 80)
                    headerLabel(String(localized: "Status"), width: 70)
                }
                .padding(.horizontal, 10)
                .padding(.bottom, 4)
                .overlay(alignment: .bottom) {
                    Rectangle().fill(RapidTheme.bandHairline).frame(height: 1)
                }
            }
        }
    }

    private func headerLabel(_ text: String, width: CGFloat) -> some View {
        Text(text.uppercased())
            .font(.system(size: 8, weight: .bold))
            .tracking(0.5)
            .foregroundStyle(RapidTheme.bandInkTertiary)
            .lineLimit(1)
            .minimumScaleFactor(0.8)
            .frame(width: width, alignment: .leading)
            .accessibilityHidden(true)
    }

    private func footer(_ account: ShareComputeLedgerAccount) -> some View {
        HStack(spacing: RapidTheme.Space.sm) {
            Text(
                account.windows.count > Self.visibleRowLimit
                    ? String(
                        format: String(localized: "Showing %1$d of %2$d windows · totals below use all of them"),
                        Self.visibleRowLimit,
                        account.windows.count
                    )
                    : String(
                        format: String(localized: "%d accounting windows"),
                        account.windows.count
                    )
            )
            .font(.system(size: 10))
            .foregroundStyle(RapidTheme.bandInkSecondary)
            Spacer(minLength: RapidTheme.Space.sm)
        }
        .padding(.top, 8)
    }

    // MARK: Local activity (fenced, separate)

    /// Rapid's own record, kept visibly apart from the ledger above it.
    ///
    /// No credit column, no settlement status, no receipt id presented as a
    /// QuickSilver reference. It answers a different question — "was this Mac
    /// connected?" — and saying so in the heading is what stops a reader
    /// treating an "Ended" receipt as a settled payment.
    private var localActivity: some View {
        VStack(alignment: .leading, spacing: 8) {
            VStack(alignment: .leading, spacing: 3) {
                ShareComputeEyebrow(
                    text: "Rapid · Local activity on this Mac",
                    tone: RapidTheme.bandInkSecondary,
                    size: 10
                )
                Text("Connection periods Rapid recorded on this Mac. Not a QuickSilver settlement record, and not matched to the windows above.")
                    .font(.system(size: 11))
                    .foregroundStyle(RapidTheme.bandInkTertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if receipts.isEmpty {
                Text("No connection periods recorded yet.")
                    .font(.system(size: 12))
                    .foregroundStyle(RapidTheme.bandInkSecondary)
                    .accessibilityIdentifier("ShareCompute.EmptyHistory")
            } else {
                HStack(spacing: 0) {
                    localMetric(
                        "\(localSummary.sessionCount)",
                        String(localized: "Connection periods")
                    )
                    localDivider
                    localMetric(
                        ShareComputeDuration.total(localSummary.totalShared),
                        String(localized: "Online")
                    )
                    localDivider
                    localMetric(
                        localSummary.modelCount == 1
                            ? String(localized: "1 Model")
                            : String(format: String(localized: "%1$d Models"), localSummary.modelCount),
                        String(localized: "Models served")
                    )
                }
            }
        }
        .padding(.top, 14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .overlay(alignment: .top) {
            Rectangle().fill(RapidTheme.bandHairlineStrong).frame(height: 1)
        }
        .accessibilityIdentifier("ShareCompute.Credits.LocalActivity")
    }

    private var localDivider: some View {
        Rectangle()
            .fill(RapidTheme.bandHairline)
            .frame(width: 1, height: 28)
            .padding(.horizontal, isNarrow ? 12 : 20)
            .accessibilityHidden(true)
    }

    private func localMetric(_ value: String, _ label: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(value)
                .font(.system(size: isNarrow ? 17 : 19, weight: .semibold))
                .monospacedDigit()
                .foregroundStyle(RapidTheme.bandInk)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
            Text(label.uppercased())
                .font(.system(size: 9, weight: .semibold))
                .tracking(0.4)
                .foregroundStyle(RapidTheme.bandInkSecondary)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(label): \(value)")
    }

    private func note(
        icon: String,
        title: String,
        detail: String?,
        identifier: String
    ) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: icon)
                .font(.system(size: 13))
                .foregroundStyle(RapidTheme.bandInkTertiary)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(RapidTheme.bandInk)
                    .fixedSize(horizontal: false, vertical: true)
                if let detail {
                    Text(detail)
                        .font(.system(size: 12))
                        .foregroundStyle(RapidTheme.bandInkSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(RapidTheme.Space.md)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RapidTheme.bandControl, in: RoundedRectangle(cornerRadius: RapidTheme.Radius.card))
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier(identifier)
    }

    // MARK: - Summary (right, amber)

    @ViewBuilder
    private var summaryPanel: some View {
        VStack(alignment: .leading, spacing: 14) {
            if state.needsReadKey {
                readKeyOnboarding
            } else {
                monthSummary
            }
        }
        .padding(.horizontal, isNarrow ? 20 : 28)
        .padding(.top, 26)
        .padding(.bottom, 24)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(RapidTheme.brandPrimary)
    }

    // MARK: Read-key onboarding

    /// Compact onboarding. Four things and no more: where the key comes from,
    /// a field to paste it, what saving does, and where it is kept.
    private var readKeyOnboarding: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 6) {
                ShareComputeEyebrow(
                    text: state == .unauthorized ? "Read key rejected" : "Connect your ledger",
                    tone: RapidTheme.onBrandPrimarySecondary,
                    size: 11
                )
                Text(state == .unauthorized ? "Replace your read key" : "Add a read key")
                    .font(.system(size: isNarrow ? 22 : 25, weight: .bold))
                    .foregroundStyle(RapidTheme.onBrandPrimary)
                    .fixedSize(horizontal: false, vertical: true)
                Text(
                    state == .unauthorized
                        ? String(localized: "QuickSilver rejected the saved key — it may have been revoked or replaced. Create a new one and paste it below.")
                        : String(localized: "A read key lets Rapid show your account’s credit ledger. It is read-only: it cannot register nodes or spend credit.")
                )
                .font(.system(size: 13))
                .foregroundStyle(RapidTheme.onBrandPrimarySecondary)
                .fixedSize(horizontal: false, vertical: true)
            }

            Button {
                openURL(ShareComputeDestination.readKeyManagement)
            } label: {
                HStack(spacing: 7) {
                    Text("Open read-only earnings keys")
                        .font(.system(size: 13, weight: .semibold))
                    Image(systemName: "arrow.up.right")
                        .font(.system(size: 11, weight: .bold))
                        .accessibilityHidden(true)
                }
                .foregroundStyle(RapidTheme.onBrandPrimary)
                .frame(maxWidth: .infinity)
                .frame(height: 38)
                .overlay {
                    RoundedRectangle(cornerRadius: 6)
                        .strokeBorder(RapidTheme.onBrandPrimary.opacity(0.5), lineWidth: 1)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel(String(localized: "Open Read-only earnings keys in the QuickSilver Share Compute tab"))
            .accessibilityIdentifier("ShareCompute.Credits.GetReadKey")

            VStack(alignment: .leading, spacing: 6) {
                // SecureField, so the key is never legible on screen and never
                // lands in a screenshot.
                SecureField("qsprk-…", text: $readKeyDraft)
                    .textFieldStyle(.plain)
                    .font(.system(size: 13, design: .monospaced))
                    .foregroundStyle(RapidTheme.onBrandPrimary)
                    .padding(.horizontal, 12)
                    .frame(height: 38)
                    .background(RapidTheme.brandPrimarySurface, in: RoundedRectangle(cornerRadius: 6))
                    .overlay {
                        RoundedRectangle(cornerRadius: 6)
                            .strokeBorder(
                                readKeyRejection == nil
                                    ? RapidTheme.brandPrimaryHairline
                                    : RapidTheme.bandDestructive,
                                lineWidth: 1
                            )
                    }
                    .accessibilityLabel(String(localized: "QuickSilver read key"))
                    .accessibilityIdentifier("ShareCompute.Credits.ReadKeyField")

                if let readKeyRejection {
                    Text(readKeyRejection.message)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(RapidTheme.onBrandPrimary)
                        .fixedSize(horizontal: false, vertical: true)
                        .accessibilityIdentifier("ShareCompute.Credits.ReadKeyError")
                }
            }

            Button(action: onSaveReadKey) {
                Text("Save & load ledger")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(RapidTheme.bandInk)
                    .frame(maxWidth: .infinity)
                    .frame(height: 42)
                    .background(RapidTheme.surfaceBand, in: RoundedRectangle(cornerRadius: 6))
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .disabled(readKeyDraft.isEmpty)
            .opacity(readKeyDraft.isEmpty ? RapidTheme.disabledOpacity : 1)
            .accessibilityIdentifier("ShareCompute.Credits.SaveReadKey")

            Text("Rapid stores the read key in this Mac’s Keychain. It can only read your account’s ledger — it cannot register nodes or spend credit. Your provider key (qsppk-) is never saved.")
                .font(.system(size: 11))
                .foregroundStyle(RapidTheme.onBrandPrimarySecondary)
                .fixedSize(horizontal: false, vertical: true)

            if savedKeyLabel != nil {
                removeKeyButton
            }
        }
    }

    private var removeKeyButton: some View {
        Button(action: onRemoveReadKey) {
            Text("Remove saved key")
                .font(.system(size: 11, weight: .bold))
                .foregroundStyle(RapidTheme.onBrandPrimary)
                .frame(maxWidth: .infinity)
                .frame(height: 32)
                .overlay {
                    RoundedRectangle(cornerRadius: 6)
                        .strokeBorder(RapidTheme.onBrandPrimary.opacity(0.45), lineWidth: 1)
                }
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("ShareCompute.Credits.RemoveReadKey")
    }

    // MARK: Month summary

    private var monthSummary: some View {
        VStack(alignment: .leading, spacing: 13) {
            VStack(alignment: .leading, spacing: 5) {
                ShareComputeEyebrow(
                    text: "Monthly credit summary",
                    tone: RapidTheme.onBrandPrimarySecondary,
                    size: 11
                )
                Text(monthTitle)
                    .font(.system(size: isNarrow ? 23 : 27, weight: .bold))
                    .foregroundStyle(RapidTheme.onBrandPrimary)
                    .fixedSize(horizontal: false, vertical: true)
                Text(monthScope)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(RapidTheme.onBrandPrimarySecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            // Server figures, summed across every fetched row for the bucket.
            HStack(alignment: .top, spacing: RapidTheme.Space.sm) {
                amberField(String(localized: "Requests"), totals.map { ShareComputeCreditFormatter.count($0.requestCount) } ?? "—")
                amberField(String(localized: "Tokens in"), totals.map { ShareComputeCreditFormatter.count($0.inputTokens) } ?? "—")
                amberField(String(localized: "Tokens out"), totals.map { ShareComputeCreditFormatter.count($0.outputTokens) } ?? "—")
            }
            .padding(.vertical, 12)
            .overlay(alignment: .top) {
                Rectangle().fill(RapidTheme.brandPrimaryHairline).frame(height: 1)
            }
            .overlay(alignment: .bottom) {
                Rectangle().fill(RapidTheme.brandPrimaryHairline).frame(height: 1)
            }

            // The headline number. Always the server's `final_credit`, summed —
            // never recomputed from accrued minus a cap.
            VStack(alignment: .leading, spacing: 8) {
                HStack(alignment: .firstTextBaseline, spacing: RapidTheme.Space.sm) {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("FINAL API CREDIT")
                            .font(.system(size: 9, weight: .bold))
                            .tracking(0.4)
                            .foregroundStyle(RapidTheme.onBrandPrimarySecondary)
                        Text(totals.map { ShareComputeCreditFormatter.credit($0.finalCredit) } ?? "—")
                            .font(.system(size: 22, weight: .bold))
                            .monospacedDigit()
                            .foregroundStyle(RapidTheme.onBrandPrimary)
                            .lineLimit(1)
                            .minimumScaleFactor(0.7)
                    }
                    Spacer(minLength: RapidTheme.Space.sm)
                    VStack(alignment: .trailing, spacing: 3) {
                        Text("ACCRUED")
                            .font(.system(size: 9, weight: .bold))
                            .tracking(0.4)
                            .foregroundStyle(RapidTheme.onBrandPrimarySecondary)
                        // `nil` renders `—`, not `$0.00`: a bucket containing a
                        // window with an unknown pre-cap amount has no honest
                        // accrued total.
                        Text(totals.flatMap(\.accruedCredit).map(ShareComputeCreditFormatter.credit) ?? "—")
                            .font(.system(size: 14, weight: .semibold))
                            .monospacedDigit()
                            .foregroundStyle(RapidTheme.onBrandPrimarySecondary)
                    }
                }
                if totals?.hasPendingWindows == true {
                    Text("Some windows are still pending; QuickSilver finalises them after the month closes.")
                        .font(.system(size: 11))
                        .foregroundStyle(RapidTheme.onBrandPrimarySecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RapidTheme.brandPrimaryRaised, in: RoundedRectangle(cornerRadius: 6))
            .overlay {
                RoundedRectangle(cornerRadius: 6)
                    .strokeBorder(RapidTheme.brandPrimaryHairline, lineWidth: 1)
            }
            .accessibilityElement(children: .combine)
            .accessibilityIdentifier("ShareCompute.Credits.MonthTotal")

            if let caps = account?.caps {
                HStack(spacing: RapidTheme.Space.sm) {
                    amberField(
                        String(localized: "Node cap"),
                        ShareComputeCreditFormatter.credit(caps.nodeMonthlyUSD)
                    )
                    amberField(
                        String(localized: "Pool cap"),
                        ShareComputeCreditFormatter.credit(caps.poolMonthlyUSD)
                    )
                }
            }

            // The unit claim, stated plainly. `usd_api_credit` is spendable
            // platform balance; calling it a payout or cash would be false.
            Text(unitExplanation)
                .font(.system(size: 11))
                .foregroundStyle(RapidTheme.onBrandPrimarySecondary)
                .fixedSize(horizontal: false, vertical: true)

            Button {
                openURL(ShareComputeDestination.credits)
            } label: {
                HStack(spacing: 7) {
                    Text("QuickSilver balance")
                        .font(.system(size: 13, weight: .semibold))
                    Image(systemName: "arrow.up.right")
                        .font(.system(size: 11, weight: .bold))
                        .accessibilityHidden(true)
                }
                .foregroundStyle(RapidTheme.bandInk)
                .frame(maxWidth: .infinity)
                .frame(height: 42)
                .background(RapidTheme.surfaceBand, in: RoundedRectangle(cornerRadius: 6))
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            // Balance, not ledger: the ledger is on THIS screen. The hash goes
            // to the account's spendable balance and recharge page.
            .accessibilityLabel(String(localized: "Open your QuickSilver account balance"))
            .accessibilityIdentifier("ShareCompute.Credits.Balance")

            if let savedKeyLabel {
                HStack(spacing: RapidTheme.Space.sm) {
                    Text(savedKeyLabel)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(RapidTheme.onBrandPrimarySecondary)
                        .lineLimit(1)
                    Spacer(minLength: RapidTheme.Space.sm)
                    Button(action: onRemoveReadKey) {
                        Text("Replace")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundStyle(RapidTheme.onBrandPrimary)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(String(localized: "Replace the saved read key"))
                    .accessibilityIdentifier("ShareCompute.Credits.ReplaceReadKey")
                }
            }
        }
    }

    private var monthTitle: String {
        guard let month else { return String(localized: "No windows yet") }
        // `2026-09-01` → `September 2026`.
        let parser = DateFormatter()
        parser.dateFormat = "yyyy-MM-dd"
        parser.timeZone = TimeZone(identifier: "UTC")
        guard let date = parser.date(from: month) else { return month }
        let display = DateFormatter()
        display.setLocalizedDateFormatFromTemplate("MMMMy")
        // Same zone it was parsed in. Formatting a UTC midnight in a local
        // zone west of Greenwich lands on the previous day, which rendered
        // "2026-09-01" as "August 2026".
        display.timeZone = TimeZone(identifier: "UTC")
        return display.string(from: date)
    }

    private var monthScope: String {
        guard let totals else {
            return String(localized: "Account-wide, across every node.")
        }
        let nodes = totals.nodeIDs.count
        return nodes == 1
            ? String(
                format: String(localized: "%1$d windows · 1 node"),
                totals.windowCount
            )
            : String(
                format: String(localized: "%1$d windows · %2$d nodes"),
                totals.windowCount,
                nodes
            )
    }

    private var unitExplanation: String {
        // Prefer QuickSilver's own note when it sends one; it is the
        // authoritative phrasing of what the unit means.
        if let note = account?.note, !note.isEmpty { return note }
        return String(localized: "Amounts are QuickSilver API credit — spendable balance for API usage, not a cash payout. 1.00 equals $1 of API usage.")
    }

    private func amberField(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label.uppercased())
                .font(.system(size: 9, weight: .bold))
                .tracking(0.4)
                .foregroundStyle(RapidTheme.onBrandPrimarySecondary)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
            Text(value)
                .font(.system(size: 14, weight: .semibold))
                .monospacedDigit()
                .foregroundStyle(RapidTheme.onBrandPrimary)
                .lineLimit(1)
                .minimumScaleFactor(0.75)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(label): \(value)")
    }
}

// MARK: - One ledger row

/// One accounting window.
///
/// Fixed-width lanes so `Window`, `Node · Model`, the counts and the credit
/// form columns down the table regardless of how long any node id is.
struct ShareComputeLedgerRowView: View {
    let window: ShareComputeLedgerWindow
    var isNarrow = false

    var body: some View {
        Group {
            if isNarrow { narrow } else { wide }
        }
        .padding(.horizontal, 10)
        .frame(minHeight: isNarrow ? 58 : 46)
        .frame(maxWidth: .infinity, alignment: .leading)
        .overlay(alignment: .bottom) {
            Rectangle().fill(RapidTheme.bandHairline).frame(height: 1)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(accessibilityLabel)
        .accessibilityIdentifier("ShareCompute.Credits.Window.\(window.cursor)")
    }

    private var wide: some View {
        HStack(spacing: 0) {
            Text(periodLabel)
                .font(.system(size: 11))
                .foregroundStyle(RapidTheme.bandInk)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
                .frame(width: 140, alignment: .leading)

            VStack(alignment: .leading, spacing: 1) {
                Text(modelLabel)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(modelTint)
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
                Text(window.nodeID)
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(RapidTheme.bandLink)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            .frame(width: 210, alignment: .leading)

            Text("\(window.requestCount)")
                .font(.system(size: 12))
                .monospacedDigit()
                .foregroundStyle(RapidTheme.bandInk)
                .frame(width: 70, alignment: .leading)

            Text("\(ShareComputeCreditFormatter.count(window.inputTokens)) / \(ShareComputeCreditFormatter.count(window.outputTokens))")
                .font(.system(size: 11))
                .monospacedDigit()
                .foregroundStyle(RapidTheme.bandInkSecondary)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
                .frame(width: 100, alignment: .leading)

            Text(ShareComputeCreditFormatter.credit(window.finalCredit))
                .font(.system(size: 12, weight: .semibold))
                .monospacedDigit()
                .foregroundStyle(RapidTheme.bandInk)
                .frame(width: 80, alignment: .leading)

            statusTag.frame(width: 70, alignment: .leading)
        }
    }

    private var narrow: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: RapidTheme.Space.sm) {
                Text(modelLabel)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(modelTint)
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
                Spacer(minLength: RapidTheme.Space.sm)
                Text(ShareComputeCreditFormatter.credit(window.finalCredit))
                    .font(.system(size: 12, weight: .semibold))
                    .monospacedDigit()
                    .foregroundStyle(RapidTheme.bandInk)
                statusTag
            }
            HStack(spacing: 8) {
                Text(periodLabel)
                    .font(.system(size: 10))
                    .foregroundStyle(RapidTheme.bandInkSecondary)
                Text("·").foregroundStyle(RapidTheme.bandInkTertiary).font(.system(size: 10))
                Text(String(format: String(localized: "%d req"), window.requestCount))
                    .font(.system(size: 10))
                    .monospacedDigit()
                    .foregroundStyle(RapidTheme.bandInkSecondary)
                Spacer(minLength: 0)
                Text(window.nodeID)
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(RapidTheme.bandLink)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
        .padding(.vertical, 8)
    }

    /// A deleted node/model record renders neutrally. Never a guess, and never
    /// blank — a blank cell reads as a rendering fault.
    private var modelLabel: String {
        window.modelID ?? String(localized: "Model unavailable")
    }

    private var modelTint: Color {
        window.modelID == nil ? RapidTheme.bandInkTertiary : RapidTheme.bandInk
    }

    private var periodLabel: String {
        let day = DateFormatter()
        day.setLocalizedDateFormatFromTemplate("MMMd")
        let time = DateFormatter()
        time.timeStyle = .short
        return "\(day.string(from: window.periodStart)) · \(time.string(from: window.periodStart))"
    }

    private var statusTag: some View {
        ShareComputeStatusTag(title: window.status.title, tone: statusTone)
    }

    private var statusTone: ShareComputeTagTone {
        switch window.status {
        case .credited: return .ready
        case .pending: return .selected
        case .zero, .unknown: return .neutral
        }
    }

    private var accessibilityLabel: String {
        let credit = ShareComputeCreditFormatter.credit(window.finalCredit)
        let accrued = window.accruedCredit == nil
            ? String(localized: "accrued amount unknown")
            : String(
                format: String(localized: "accrued %@"),
                ShareComputeCreditFormatter.credit(window.accruedCredit)
            )
        return String(
            format: String(localized: "%1$@, node %2$@, %3$d requests, %4$d input and %5$d output tokens, credit %6$@, %7$@, status %8$@"),
            modelLabel,
            window.nodeID,
            window.requestCount,
            window.inputTokens,
            window.outputTokens,
            credit,
            accrued,
            window.status.title
        )
    }
}
