import SwiftUI

/// "What is online right now?"
///
/// Live Pool reports AVAILABILITY, and only availability. Everything the old
/// Pool tab claimed beyond that — a demand ranking, requests waiting, queue
/// depth, a recommended model — was invented, because the only endpoint that
/// exists is `GET /v1/pool/summary`, which publishes connected / ready / busy
/// node counts and free request slots. Nothing here may imply that the pool
/// wants one model more than another.
///
/// Two panels: a dark capacity workbench on the left reading the summary, and
/// an amber selection panel on the right where the user picks what this Mac
/// would serve. At 720pt they stack rather than compress, because horizontally
/// clipping either one loses a number the page exists to show.
struct ShareComputePoolTab: View {
    let rows: [ShareComputePoolRow]
    let state: ShareComputePoolSummaryState
    let selectedID: String?
    let isNarrow: Bool
    let onSelect: (ShareComputePoolRow) -> Void
    let onContribute: () -> Void
    let onDownload: (ShareComputePoolRow) -> Void
    /// Injected so the golden harness renders a fixed "Updated …" string
    /// instead of one that changes between two captures of the same fixture.
    var now: Date = Date()

    private var selectedRow: ShareComputePoolRow? {
        rows.first { $0.id == selectedID } ?? rows.first
    }

    private var summary: ShareComputePoolSummary? { state.summary }

    var body: some View {
        ShareComputeWorkbenchFrame {
            if isNarrow {
                VStack(spacing: 0) {
                    capacityPanel
                    selectionPanel
                }
            } else {
                HStack(alignment: .top, spacing: 0) {
                    capacityPanel
                    selectionPanel.frame(width: 400)
                }
                .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    // MARK: - Capacity (left, dark)

    private var capacityPanel: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .center, spacing: RapidTheme.Space.sm) {
                ShareComputeEyebrow(text: "Live pool capacity", size: 11)
                Spacer(minLength: RapidTheme.Space.sm)
                freshnessBadge
            }

            VStack(alignment: .leading, spacing: 7) {
                Text("What is online right now")
                    .font(.system(size: isNarrow ? 22 : 27, weight: .semibold))
                    .foregroundStyle(RapidTheme.bandInk)
                    .fixedSize(horizontal: false, vertical: true)
                Text("Availability and capacity only. QuickSilver does not publish a demand queue.")
                    .font(.system(size: 12))
                    .foregroundStyle(RapidTheme.bandInkSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            totals
            modelList
            emptyOrErrorNote

            HStack(alignment: .top, spacing: RapidTheme.Space.sm) {
                Text("Live relay availability · no demand queue")
                    .font(.system(size: 11))
                    .foregroundStyle(RapidTheme.bandInkSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: RapidTheme.Space.sm)
                // Rates and caps are QuickSilver's to state and change. Rapid
                // links to them rather than printing a figure the summary
                // endpoint never returned.
                Text("Credits: metered tokens, monthly")
                    .font(.system(size: 11))
                    .foregroundStyle(RapidTheme.bandInkSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.top, 2)
        }
        .padding(isNarrow ? 20 : 28)
        // maxHeight matters: inside the `.fixedSize(vertical:)` HStack the two
        // panels resolve to the TALLER one's height, and without this the
        // shorter panel leaves a window-coloured gap inside the rounded frame.
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(RapidTheme.surfaceBand)
    }

    /// Says where the numbers came from and how old they are.
    ///
    /// Always formats the server's own `updated_at`, never the local fetch
    /// time — when a refresh fails, this label is the only thing telling the
    /// user that the values beside it are the last good reading.
    private var freshnessBadge: some View {
        HStack(spacing: 8) {
            Group {
                if state.isLoading || state.isRefreshing {
                    ProgressView().controlSize(.small)
                } else {
                    Circle()
                        .fill(summary == nil ? RapidTheme.bandInkTertiary : RapidTheme.bandReady)
                        .frame(width: 7, height: 7)
                }
            }
            .frame(width: 7)
            .accessibilityHidden(true)

            Text(freshnessText)
                .font(.system(size: 11))
                .foregroundStyle(RapidTheme.bandInkSecondary)
        }
        .padding(.horizontal, 10)
        .frame(height: 26)
        .background(RapidTheme.bandControl, in: RoundedRectangle(cornerRadius: 6))
        .fixedSize()
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("ShareCompute.Pool.Freshness")
    }

    private var freshnessText: String {
        guard let summary else {
            return state.isLoading
                ? String(localized: "Reading pool capacity…")
                : String(localized: "Relay aggregate · unavailable")
        }
        return String(
            format: String(localized: "Relay aggregate · %@"),
            ShareComputePoolClock.updatedLabel(summary.updatedAt, now: now)
        )
    }

    // MARK: Totals

    /// The three pool-wide figures, taken from `totals` AS PUBLISHED.
    ///
    /// Never summed from the model rows: the server counts nodes the models
    /// array may not enumerate, so a local sum would quietly under-report.
    private var totals: some View {
        HStack(alignment: .center, spacing: 0) {
            totalCell(
                value: summary.map { "\($0.totals.connectedNodes)" },
                label: ShareComputePoolLabels.connectedTotal,
                tint: RapidTheme.bandInk,
                isFirst: true
            )
            totalCell(
                value: summary.map { "\($0.totals.readyNodes)" },
                label: String(localized: "Ready to Serve"),
                // The one green figure: readiness is the state that decides
                // whether a request can be served at all.
                tint: RapidTheme.bandReady,
                isFirst: false
            )
            totalCell(
                value: summary.map { "\($0.totals.availableSlots)" },
                label: String(localized: "Open Request Slots"),
                tint: RapidTheme.bandInk,
                isFirst: false
            )
        }
        .frame(height: 64)
        .overlay(alignment: .top) {
            Rectangle().fill(RapidTheme.bandHairlineStrong).frame(height: 1)
        }
        .overlay(alignment: .bottom) {
            Rectangle().fill(RapidTheme.bandHairlineStrong).frame(height: 1)
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("ShareCompute.Pool.Totals")
    }

    /// One total. The value is ALWAYS rendered — a real zero prints `0`, and
    /// an absent reading prints `—`. Omitting the number entirely (which the
    /// first draft of this layout did for Connected Machines) leaves a labelled
    /// cell with nothing in it, which reads as a rendering bug.
    private func totalCell(
        value: String?,
        label: String,
        tint: Color,
        isFirst: Bool
    ) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(value ?? ShareComputePoolAvailability.unknownValue)
                .font(.system(size: 22, weight: .semibold))
                .monospacedDigit()
                .foregroundStyle(value == nil ? RapidTheme.bandInkTertiary : tint)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
            Text(label.uppercased())
                .font(.system(size: 10, weight: .semibold))
                .tracking(0.5)
                .foregroundStyle(RapidTheme.bandInkSecondary)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.leading, isFirst ? 0 : 18)
        .overlay(alignment: .leading) {
            if !isFirst {
                Rectangle()
                    .fill(RapidTheme.bandHairlineStrong)
                    .frame(width: 1, height: 40)
                    .accessibilityHidden(true)
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(
            value.map { "\(label): \($0)" }
                ?? String(format: String(localized: "%@: not available"), label)
        )
    }

    // MARK: Model list

    /// Every supported model, in catalog order.
    ///
    /// The list is as long as Rapid's catalog, not as long as the summary's
    /// `models` array — the pool gaining a fourth model must not require a
    /// layout change, and a model the summary omits still belongs on screen
    /// with `—` beside it.
    private var modelList: some View {
        VStack(spacing: 0) {
            ForEach(rows) { row in
                ShareComputePoolRowView(
                    row: row,
                    isSelected: row.id == selectedRow?.id,
                    isNarrow: isNarrow,
                    onSelect: { onSelect(row) }
                )
            }
        }
        .overlay(alignment: .top) {
            Rectangle().fill(RapidTheme.bandHairlineStrong).frame(height: 1)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(String(localized: "Pool models and their live capacity"))
    }

    /// The calm empty state, and the two failure notes.
    ///
    /// A genuinely empty pool is SUCCESS: every counter is a real zero, the
    /// model list and this Mac's readiness still work, and nothing red appears.
    /// Only a load with no data at all is an error.
    @ViewBuilder
    private var emptyOrErrorNote: some View {
        if let error = state.blockingError {
            note(
                icon: "antenna.radiowaves.left.and.right.slash",
                title: error.displayMessage,
                detail: String(localized: "Rapid couldn’t read the pool summary. Your own readiness below is measured on this Mac and is unaffected."),
                tint: RapidTheme.bandInkSecondary,
                identifier: "ShareCompute.Pool.Unavailable"
            )
        } else if let staleNote = state.staleNote {
            note(
                icon: "clock.arrow.circlepath",
                title: staleNote,
                detail: nil,
                tint: RapidTheme.bandInkSecondary,
                identifier: "ShareCompute.Pool.Stale"
            )
        } else if let summary, summary.isPoolEmpty {
            note(
                icon: "moon.zzz",
                title: ShareComputePoolLabels.noneConnected,
                detail: String(localized: "That is a normal reading, not an error. Connect this Mac and it becomes the first."),
                tint: RapidTheme.bandInkSecondary,
                identifier: "ShareCompute.Pool.Empty"
            )
        }
    }

    private func note(
        icon: String,
        title: String,
        detail: String?,
        tint: Color,
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
                        .foregroundStyle(tint)
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

    // MARK: - Selection (right, amber)

    private var selectionPanel: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 6) {
                ShareComputeEyebrow(
                    text: "Community capacity",
                    tone: RapidTheme.onBrandPrimarySecondary,
                    size: 11
                )
                Text(communityHeadline)
                    .font(.system(size: isNarrow ? 24 : 29, weight: .bold))
                    .foregroundStyle(RapidTheme.onBrandPrimary)
                    .fixedSize(horizontal: false, vertical: true)
                Text(communitySubhead)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(RapidTheme.onBrandPrimarySecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if let selectedRow {
                selectedCard(selectedRow)
            }

            modelPicker
            action
        }
        .padding(.horizontal, isNarrow ? 20 : 30)
        .padding(.top, 28)
        .padding(.bottom, 26)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(RapidTheme.brandPrimary)
    }

    /// Only ever states a count the server published.
    ///
    /// "Machines", not "Macs": `totals.connected_nodes` counts every node on
    /// the pool, and the protocol makes no hardware assumption — a Linux box
    /// serving through vLLM registers exactly the way this Mac does. The local
    /// half of this screen still says "this Mac", because that one IS a Mac.
    private var communityHeadline: String {
        guard let summary else { return String(localized: "Pool data unavailable") }
        return ShareComputePoolLabels.machinesOnline(summary.totals.connectedNodes)
    }

    private var communitySubhead: String {
        guard let summary else {
            return String(localized: "Your own readiness below still works.")
        }
        let slots = summary.totals.availableSlots
        return slots == 1
            ? String(localized: "1 request slot available now")
            : String(format: String(localized: "%d request slots available now"), slots)
    }

    /// The model this Mac would serve, with its real local state.
    private func selectedCard(_ row: ShareComputePoolRow) -> some View {
        HStack(alignment: .center, spacing: RapidTheme.Space.md) {
            VStack(alignment: .leading, spacing: 7) {
                ShareComputeEyebrow(
                    text: "Model to serve",
                    tone: RapidTheme.onBrandPrimarySecondary,
                    size: 10
                )
                Text(row.local.title)
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(RapidTheme.onBrandPrimary)
                    .fixedSize(horizontal: false, vertical: true)
                Text(selectedCardStatus(row))
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(
                        row.local.isReady
                            ? RapidTheme.onBrandPrimaryReady
                            : RapidTheme.onBrandPrimarySecondary
                    )
                    .fixedSize(horizontal: false, vertical: true)
                if let storageLabel = row.storage.label, !row.local.isReady {
                    Text(storageLabel)
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(
                            row.storage.isSufficient
                                ? RapidTheme.onBrandPrimarySecondary
                                : RapidTheme.statusError
                        )
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RapidTheme.brandPrimaryRaised,
            in: RoundedRectangle(cornerRadius: 6)
        )
        .overlay {
            RoundedRectangle(cornerRadius: 6)
                .strokeBorder(RapidTheme.brandPrimaryHairline, lineWidth: 1)
        }
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("ShareCompute.Pool.SelectedModel")
    }

    /// Local truth first, then the pool's own switch. A model this Mac has
    /// downloaded is still unusable if the pool has it disabled, and the card
    /// has to say which of the two is blocking.
    private func selectedCardStatus(_ row: ShareComputePoolRow) -> String {
        if row.isDisabledUpstream {
            return String(localized: "Paused by QuickSilver · not accepting nodes")
        }
        if row.local.isReady {
            if let size = row.local.onDiskSize {
                return String(format: String(localized: "Ready on this Mac · %@"), size)
            }
            return String(localized: "Ready on this Mac")
        }
        if let bytes = row.local.estimatedDownloadBytes {
            return String(
                format: String(localized: "Download required · ~%@"),
                ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
            )
        }
        return String(localized: "Download required")
    }

    /// The selectable list. This is the control the redesign brief calls out:
    /// it must look and behave like a picker, not a legend.
    private var modelPicker: some View {
        VStack(spacing: 0) {
            ForEach(Array(rows.enumerated()), id: \.element.id) { index, row in
                pickerRow(row, isLast: index == rows.count - 1)
            }
        }
        .background(RapidTheme.brandPrimarySurface, in: RoundedRectangle(cornerRadius: 6))
        .overlay {
            RoundedRectangle(cornerRadius: 6)
                .strokeBorder(RapidTheme.brandPrimaryHairline, lineWidth: 1)
        }
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .accessibilityElement(children: .contain)
        .accessibilityLabel(String(localized: "Choose the model this Mac would serve"))
        .accessibilityIdentifier("ShareCompute.Pool.ModelPicker")
    }

    private func pickerRow(_ row: ShareComputePoolRow, isLast: Bool) -> some View {
        let isSelected = row.id == selectedRow?.id
        return Button {
            onSelect(row)
        } label: {
            HStack(spacing: RapidTheme.Space.sm) {
                Text(row.local.title)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(RapidTheme.onBrandPrimary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.78)
                    .frame(maxWidth: .infinity, alignment: .leading)
                // Fixed trailing slot so the state words form one lane down the
                // list instead of drifting with each model name's length.
                Text(pickerState(row).uppercased())
                    .font(.system(size: 10, weight: .bold))
                    .tracking(0.4)
                    .foregroundStyle(pickerStateTint(row))
                    .lineLimit(1)
                    .frame(width: 92, alignment: .trailing)
            }
            .padding(.horizontal, 12)
            .frame(height: 42)
            .frame(maxWidth: .infinity)
            .background(isSelected ? RapidTheme.brandPrimaryRaised : Color.clear)
            .overlay(alignment: .leading) {
                // The selected row also carries a solid bar, so selection is
                // not conveyed by a fill difference alone.
                Rectangle()
                    .fill(isSelected ? RapidTheme.onBrandPrimary : Color.clear)
                    .frame(width: 3)
            }
            .overlay(alignment: .bottom) {
                if !isLast {
                    Rectangle()
                        .fill(RapidTheme.brandPrimaryHairline)
                        .frame(height: 1)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(isSelected ? [.isSelected, .isButton] : .isButton)
        .accessibilityLabel("\(row.local.title), \(pickerState(row))")
        .accessibilityIdentifier("ShareCompute.Pool.Model.\(row.id)")
    }

    private func pickerState(_ row: ShareComputePoolRow) -> String {
        if row.isDisabledUpstream { return String(localized: "Disabled") }
        if !row.local.isReady { return String(localized: "Download") }
        if row.id == selectedRow?.id { return String(localized: "Selected") }
        return String(localized: "Ready")
    }

    private func pickerStateTint(_ row: ShareComputePoolRow) -> Color {
        if row.isDisabledUpstream { return RapidTheme.onBrandPrimarySecondary }
        if !row.local.isReady { return RapidTheme.onBrandPrimarySecondary }
        if row.id == selectedRow?.id { return RapidTheme.onBrandPrimary }
        return RapidTheme.onBrandPrimaryReady
    }

    // MARK: Action

    @ViewBuilder
    private var action: some View {
        let plan = ShareComputePoolAction.make(for: selectedRow)
        Button {
            guard let row = selectedRow else { return }
            if plan.startsDownload { onDownload(row) } else { onContribute() }
        } label: {
            HStack(spacing: 8) {
                Text(plan.title)
                    .font(.system(size: 14, weight: .semibold))
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
                Image(systemName: plan.startsDownload ? "arrow.down.circle" : "arrow.right")
                    .font(.system(size: 12, weight: .bold))
                    .accessibilityHidden(true)
            }
            .foregroundStyle(RapidTheme.bandInk)
            .frame(maxWidth: .infinity)
            .frame(height: 48)
            .background(RapidTheme.surfaceBand, in: RoundedRectangle(cornerRadius: 6))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!plan.isEnabled)
        .opacity(plan.isEnabled ? 1 : RapidTheme.disabledOpacity)
        .accessibilityLabel(plan.title)
        .accessibilityIdentifier("ShareCompute.Pool.Contribute")
    }
}

// MARK: - One capacity row

/// A model's live capacity, on the dark workbench.
///
/// The four counters sit in FIXED-WIDTH slots. Gap-based spacing was what let
/// the longest model name push the numbers out of alignment row to row, and it
/// is why a name like "Nemotron 3.5 Lightning 30B · 4-bit" could collide with
/// the first metric.
struct ShareComputePoolRowView: View {
    let row: ShareComputePoolRow
    let isSelected: Bool
    var isNarrow = false
    let onSelect: () -> Void

    /// Paper's lane widths, narrowed at 720pt. 70 rather than 62 because
    /// "CONNECTED" is the longest label and truncating it to "CONNECT…" makes
    /// the column meaningless — the number above it could be anything.
    private var metricWidth: CGFloat { isNarrow ? 70 : 86 }

    var body: some View {
        Button(action: onSelect) {
            HStack(spacing: RapidTheme.Space.sm) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(row.local.title)
                        .font(.system(size: isNarrow ? 14 : 15, weight: .semibold))
                        .foregroundStyle(RapidTheme.bandInk)
                        .lineLimit(1)
                        .minimumScaleFactor(0.72)
                    Text(subtitle)
                        .font(.system(size: 12))
                        .foregroundStyle(
                            row.isDisabledUpstream
                                ? RapidTheme.bandInkTertiary
                                : RapidTheme.bandInkSecondary
                        )
                        .lineLimit(1)
                        .minimumScaleFactor(0.8)
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                metric(String(localized: "Connected"), row.availability.stats?.connectedNodes, tint: RapidTheme.bandInk)
                metric(String(localized: "Ready"), row.availability.stats?.readyNodes, tint: RapidTheme.bandReady)
                metric(String(localized: "Busy"), row.availability.stats?.busyNodes, tint: RapidTheme.bandInk)
                metric(String(localized: "Slots"), row.availability.stats?.availableSlots, tint: RapidTheme.bandInk, alignment: .trailing)
            }
            .padding(.horizontal, 14)
            .frame(height: 74)
            .frame(maxWidth: .infinity)
            .background(isSelected ? RapidTheme.bandSelectionFill : Color.clear)
            .overlay(alignment: .leading) {
                Rectangle()
                    .fill(isSelected ? RapidTheme.brandPrimary : Color.clear)
                    .frame(width: 3)
            }
            .overlay(alignment: .bottom) {
                Rectangle().fill(RapidTheme.bandHairlineStrong).frame(height: 1)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(isSelected ? [.isSelected, .isButton] : .isButton)
        .accessibilityLabel(accessibilityLabel)
        .accessibilityIdentifier("ShareCompute.Pool.Row.\(row.id)")
    }

    /// Three different facts, never conflated: the pool switched this model
    /// off, the pool never mentioned it, or it is simply a supported model.
    private var subtitle: String {
        if row.isDisabledUpstream {
            return String(localized: "Paused by QuickSilver")
        }
        switch row.availability {
        case .unreported:
            return String(localized: "Not reported in this reading")
        case .live, .disabled:
            return row.local.isReady
                ? String(localized: "Supported pool model · ready on this Mac")
                : String(localized: "Supported pool model")
        }
    }

    /// One counter. `nil` renders `—`, never `0`: an unreported model has an
    /// unknown count, and printing zero would assert that nobody is serving it.
    private func metric(
        _ label: String,
        _ value: Int?,
        tint: Color,
        alignment: HorizontalAlignment = .leading
    ) -> some View {
        VStack(alignment: alignment, spacing: 3) {
            Text(value.map { "\($0)" } ?? ShareComputePoolAvailability.unknownValue)
                .font(.system(size: 15, weight: .semibold))
                .monospacedDigit()
                .foregroundStyle(value == nil ? RapidTheme.bandInkTertiary : tint)
            Text(label.uppercased())
                .font(.system(size: 10))
                .tracking(0.4)
                .foregroundStyle(RapidTheme.bandInkSecondary)
                .lineLimit(1)
                // Scale rather than truncate: a metric label that reads
                // "CONNECT…" no longer names which counter it labels.
                .minimumScaleFactor(0.72)
        }
        .frame(width: metricWidth, alignment: alignment == .trailing ? .trailing : .leading)
    }

    private var accessibilityLabel: String {
        guard let stats = row.availability.stats else {
            return String(
                format: String(localized: "%@. Capacity not reported in this reading."),
                row.local.title
            )
        }
        let capacity = String(
            format: String(localized: "%1$d connected, %2$d ready, %3$d busy, %4$d open slots"),
            stats.connectedNodes,
            stats.readyNodes,
            stats.busyNodes,
            stats.availableSlots
        )
        let enabled = row.isDisabledUpstream
            ? String(localized: "Paused by QuickSilver.")
            : String(localized: "Accepting nodes.")
        return "\(row.local.title). \(enabled) \(capacity)."
    }
}
