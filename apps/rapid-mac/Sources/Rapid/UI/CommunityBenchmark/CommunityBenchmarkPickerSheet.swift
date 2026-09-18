import SwiftUI

/// "Choose a model to benchmark".
///
/// A fixed-size sheet: the header and footer never move, and only the model
/// list scrolls. That is what lets a 41-model catalogue stay browsable without
/// the dialog growing past the bottom of the window — the failure mode of the
/// menu-style picker this replaces, which simply listed everything.
struct CommunityBenchmarkPickerSheet: View {
    let listing: CommunityBenchmarkPicker.Listing
    /// Size of the surface the sheet is presented over, so the dialog can be
    /// clamped to the viewport with a margin instead of overflowing a short
    /// window.
    let containerSize: CGSize
    @Binding var query: String
    @Binding var selectedAlias: String
    let onCancel: () -> Void
    let onChoose: (String) -> Void

    @FocusState private var searchFocused: Bool

    /// Row and chrome metrics, named because they are what makes the "eight
    /// or nine complete rows at 1440 × 900, at least six at 900" requirement
    /// hold: 620 − 96 header − 82 footer = 442pt of viewport, which fits two
    /// 28pt group headings plus eight 46pt rows.
    static let rowHeight: CGFloat = 46
    static let groupHeaderHeight: CGFloat = 28
    static let headerHeight: CGFloat = 96
    static let footerHeight: CGFloat = 82

    /// How many complete rows fit at a given sheet height. Exposed so a test
    /// can assert the requirement instead of trusting the arithmetic above.
    static func visibleRowCount(sheetHeight: CGFloat, groupCount: Int = 2) -> Int {
        let viewport = sheetHeight - headerHeight - footerHeight
        let forRows = viewport - CGFloat(groupCount) * groupHeaderHeight
        return max(0, Int(forRows / rowHeight))
    }

    /// Desktop dialog metrics from the design: 680 × 620, clamped into the
    /// viewport with at least a 24pt margin on every side. At a 900 × 600
    /// window this resolves to 680 × 552 and still shows six complete rows.
    static func clampedSheetSize(in container: CGSize) -> CGSize {
        CGSize(
            width: min(680, max(1, container.width - 48)),
            height: min(620, max(1, container.height - 48))
        )
    }

    private var sheetWidth: CGFloat {
        Self.clampedSheetSize(in: containerSize).width
    }

    private var sheetHeight: CGFloat {
        Self.clampedSheetSize(in: containerSize).height
    }

    private var selectedRow: CommunityBenchmarkPicker.Row? {
        listing.sections
            .flatMap(\.rows)
            .first { $0.model.entry.alias == selectedAlias }
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            list
            Divider()
            footer
        }
        .frame(width: sheetWidth, height: sheetHeight)
        .background(RapidTheme.surfaceRaised)
        .accessibilityIdentifier("CommunityBenchmark.Picker")
    }

    // MARK: - Fixed header

    private var header: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
            Text("Choose a model to benchmark")
                .font(.system(size: 17, weight: .semibold))
                .foregroundStyle(RapidTheme.textPrimary)
            HStack(spacing: RapidTheme.Space.sm) {
                Image(systemName: "magnifyingglass")
                    .font(.system(size: 12))
                    .foregroundStyle(RapidTheme.textTertiary)
                    .accessibilityHidden(true)
                TextField(String(localized: "Search models"), text: $query)
                    .textFieldStyle(.plain)
                    .font(RapidFont.body)
                    .focused($searchFocused)
                    .accessibilityLabel(String(localized: "Search models"))
                    .accessibilityIdentifier("CommunityBenchmark.Picker.Search")
                if !query.isEmpty {
                    Button {
                        query = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 12))
                            .foregroundStyle(RapidTheme.textTertiary)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(String(localized: "Clear search"))
                    .accessibilityIdentifier("CommunityBenchmark.Picker.ClearSearch")
                }
                Text(listing.countLabel)
                    .font(RapidFont.secondary)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .monospacedDigit()
                    .accessibilityIdentifier("CommunityBenchmark.Picker.Count")
            }
            .padding(.horizontal, RapidTheme.Space.md)
            .frame(height: RapidTheme.ControlHeight.large)
            .background(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.input)
                    .fill(RapidTheme.surfaceRaised)
            )
            .overlay(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.input)
                    .strokeBorder(
                        searchFocused ? RapidTheme.focusRing : RapidTheme.hairlineStrong,
                        lineWidth: searchFocused ? 1.5 : 1
                    )
            )
        }
        .padding(.horizontal, RapidTheme.Space.xl)
        .padding(.top, RapidTheme.Space.lg)
        .padding(.bottom, RapidTheme.Space.md)
        .frame(height: Self.headerHeight)
    }

    // MARK: - Scrolling list

    private var list: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 0, pinnedViews: [.sectionHeaders]) {
                ForEach(listing.sections) { section in
                    Section {
                        ForEach(section.rows) { row in
                            PickerRow(
                                row: row,
                                isSelected: row.model.entry.alias == selectedAlias
                            ) {
                                selectedAlias = row.model.entry.alias
                            }
                            Divider().padding(.leading, RapidTheme.Space.xl)
                        }
                    } header: {
                        groupHeader(section.title)
                    }
                }
                if listing.isEmpty {
                    Text("No models match this search.")
                        .font(RapidFont.body)
                        .foregroundStyle(RapidTheme.textSecondary)
                        .padding(RapidTheme.Space.xl)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
        .scrollIndicators(.visible)
        .frame(maxHeight: .infinity)
        .accessibilityIdentifier("CommunityBenchmark.Picker.List")
    }

    private func groupHeader(_ title: String) -> some View {
        Text(title)
            .font(RapidFont.groupLabel)
            .tracking(0.5)
            .foregroundStyle(RapidTheme.textTertiary)
            .padding(.horizontal, RapidTheme.Space.xl)
            .frame(height: Self.groupHeaderHeight, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .leading)
            // Opaque so pinned headers never let rows show through as they
            // scroll underneath.
            .background(RapidTheme.surfaceCanvas)
            .overlay(alignment: .bottom) {
                Rectangle().fill(RapidTheme.hairline).frame(height: 1)
            }
            .accessibilityAddTraits(.isHeader)
    }

    // MARK: - Fixed footer

    private var footer: some View {
        VStack(spacing: 0) {
            HStack(spacing: RapidTheme.Space.sm) {
                Image(systemName: "arrow.down.circle")
                    .font(.system(size: 12))
                    .foregroundStyle(RapidTheme.textTertiary)
                    .accessibilityHidden(true)
                // The client has no byte-level download phase to report, so
                // the footer states the behaviour instead of faking progress.
                Text("Models that are not downloaded yet are fetched when the benchmark starts.")
                    .font(RapidFont.secondary)
                    .foregroundStyle(RapidTheme.textSecondary)
                Spacer(minLength: 0)
            }
            .padding(.horizontal, RapidTheme.Space.xl)
            .frame(height: 38)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RapidTheme.surfaceCanvas)

            Divider()

            HStack(spacing: RapidTheme.Space.md) {
                Spacer(minLength: 0)
                Button(String(localized: "Cancel"), action: onCancel)
                    .buttonStyle(.rapidSecondary)
                    .keyboardShortcut(.cancelAction)
                    .accessibilityIdentifier("CommunityBenchmark.Picker.Cancel")
                Button(String(localized: "Choose model")) {
                    onChoose(selectedAlias)
                }
                .buttonStyle(.rapidPrimary)
                .keyboardShortcut(.defaultAction)
                .disabled(!CommunityBenchmarkPicker.canChoose(selectedRow))
                .accessibilityIdentifier("CommunityBenchmark.Picker.Choose")
            }
            .padding(.horizontal, RapidTheme.Space.xl)
            .frame(height: Self.footerHeight - 38)
        }
        .frame(height: Self.footerHeight)
    }
}

/// One model row. Fixed 50pt height whatever the alias length, so a long
/// `mlx-community/...` alias truncates inside its column instead of reflowing
/// the row or widening the dialog.
private struct PickerRow: View {
    let row: CommunityBenchmarkPicker.Row
    let isSelected: Bool
    let onSelect: () -> Void

    private var workload: CommunityWorkload { CommunityWorkload(task: row.model.task) }

    private var doesNotFit: Bool { row.model.memoryFit == "does_not_fit" }

    var body: some View {
        Button(action: onSelect) {
            HStack(spacing: RapidTheme.Space.md) {
                Image(systemName: "checkmark")
                    .font(.system(size: 11, weight: .semibold))
                    .frame(width: 14)
                    .opacity(isSelected ? 1 : 0)
                    .accessibilityHidden(true)

                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: RapidTheme.Space.sm) {
                        Text(row.model.entry.alias)
                            .font(RapidFont.bodyEmphasis)
                            .foregroundStyle(RapidTheme.textPrimary)
                            .lineLimit(1)
                            .truncationMode(.tail)
                        CommunityWorkloadBadge(workload: workload)
                    }
                    if let coverage = row.coverageSentence {
                        Text(coverage)
                            .font(RapidFont.secondary)
                            .foregroundStyle(
                                row.isFirstResultOpportunity
                                    ? RapidTheme.brandPrimaryDeep
                                    : RapidTheme.textSecondary
                            )
                            .lineLimit(1)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                fitColumn
                    .frame(width: 104, alignment: .leading)

                statusColumn
                    .frame(width: 128, alignment: .trailing)
            }
            .padding(.horizontal, RapidTheme.Space.xl)
            .frame(height: CommunityBenchmarkPickerSheet.rowHeight)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .background(isSelected ? RapidTheme.selectionFill : .clear)
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
        .accessibilityIdentifier("CommunityBenchmark.Picker.Row.\(row.model.entry.alias)")
    }

    private var fitColumn: some View {
        HStack(spacing: 5) {
            Image(systemName: doesNotFit ? "exclamationmark.circle" : "checkmark.circle")
                .font(.system(size: 11))
                .accessibilityHidden(true)
            Text(doesNotFit ? String(localized: "May not fit") : String(localized: "Fits"))
                .font(RapidFont.secondary)
                .lineLimit(1)
        }
        .foregroundStyle(doesNotFit ? RapidTheme.statusError : RapidTheme.statusReady)
    }

    private var statusColumn: some View {
        let status = CommunityBenchmarkPicker.downloadStatus(
            row.model, downloadSizeGB: row.downloadSizeGB
        )
        return VStack(alignment: .trailing, spacing: 1) {
            if doesNotFit, let needed = row.model.estimatedMemoryGib {
                Text(
                    String(
                        format: String(localized: "Needs about %1$d GB"),
                        needed
                    )
                )
                .font(RapidFont.secondary)
                .foregroundStyle(RapidTheme.textSecondary)
                .lineLimit(1)
            } else {
                Text(status.title)
                    .font(RapidFont.secondary)
                    .foregroundStyle(RapidTheme.textPrimary)
                    .lineLimit(1)
                if let detail = status.detail {
                    Text(detail)
                        .font(RapidFont.metric)
                        .foregroundStyle(RapidTheme.textSecondary)
                        .lineLimit(1)
                }
            }
        }
    }
}
