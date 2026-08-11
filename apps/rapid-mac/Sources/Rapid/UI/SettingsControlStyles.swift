import SwiftUI

/// Settings rows use a stable trailing control column.
///
/// SwiftUI's stock switch style sizes itself from the label's ideal
/// width, which made a short description place its switch near the
/// middle while longer descriptions happened to push theirs right. This
/// style makes that alignment explicit across every settings panel while
/// retaining the native macOS switch.
///
/// Two things beyond alignment, both from the UI-1 review:
///
///   * **Compact.** The switch renders at ``controlSize(.small)``. At the
///     regular size it was the heaviest object on a settings page —
///     visually louder than the row's own title — which is the wrong
///     emphasis for a control whose job is to be flicked and forgotten.
///   * **Room to breathe.** The label column is separated from the
///     switch by ``gutter``, and the switch sits in a fixed-width
///     column, so a three-line description can never creep under the
///     control and every switch lines up down one edge.
///
/// The switch itself stays native: a `Toggle` with `.switch`, so macOS
/// owns its shape, its animation, its focus ring, its hit target and its
/// accessibility. Only its size and the space around it are ours.
struct TrailingSettingsToggleStyle: ToggleStyle {
    /// Clear space between the end of the text column and the switch.
    /// The review asked for 20–24pt; ``Space.xl`` is 24.
    static let gutter: CGFloat = RapidTheme.Space.xl

    /// Width reserved for the switch. A `.small` macOS switch is ~32pt
    /// wide; reserving a fixed column means rows with and without a
    /// description still line their switches up.
    static let controlColumnWidth: CGFloat = 38

    func makeBody(configuration: Configuration) -> some View {
        let binding = Binding(
            get: { configuration.isOn },
            set: { configuration.isOn = $0 }
        )

        // `.firstTextBaseline` would be ideal, but a switch has no text
        // baseline to align to, so SwiftUI falls back to centring the
        // row — which walks the switch down the page as a description
        // grows. `.top` plus a first-line-height frame pins it to the
        // title line and keeps it there however long the copy gets.
        return HStack(alignment: .top, spacing: Self.gutter) {
            configuration.label
                .frame(maxWidth: .infinity, alignment: .leading)

            Toggle(isOn: binding) { EmptyView() }
                .labelsHidden()
                .toggleStyle(.switch)
                .controlSize(.small)
                .frame(
                    width: Self.controlColumnWidth,
                    height: RapidTheme.ControlHeight.small,
                    alignment: .trailing
                )
        }
        .frame(maxWidth: .infinity)
    }
}

/// The app's one segmented control.
///
/// ## Why this is not `.pickerStyle(.segmented)`
///
/// The native style takes its selected-segment fill from the ambient
/// tint and its selected-segment label from macOS — which pairs the
/// scene's brand amber with **white** text. That is the same ~2:1
/// combination ``RapidTheme/brandOnAccent`` exists to prevent, and
/// SwiftUI exposes no hook to change it: the label colour is decided
/// inside AppKit's `NSSegmentedControl` rendering, not by the ambient
/// `foregroundStyle`. The native control also grew well past the 28–32pt
/// macOS uses at the regular control size, because it scaled with the
/// panels' type.
///
/// So this is a deliberate, documented exception to "prefer native
/// controls": a segmented control is a row of buttons, it carries no
/// system behaviour that is lost by rebuilding it, and rebuilding is
/// what lets the selected segment use dark ink on amber.
///
/// Everything else stays honest to the platform: 30pt tall, a modest 6pt
/// segment radius inside an 8pt track, no capsules, real `Button`s so
/// keyboard focus and AX press behaviour survive, and no layout shift on
/// selection — each segment reserves its selected (semibold) width via a
/// hidden measurement copy.
struct RapidSegmentedControl<Value: Hashable>: View {
    struct Option: Identifiable {
        let value: Value
        let title: String
        /// Optional stable AX identifier for one segment.
        var identifier: String?

        var id: Value { value }

        init(value: Value, title: String, identifier: String? = nil) {
            self.value = value
            self.title = title
            self.identifier = identifier
        }
    }

    @Binding var selection: Value
    let options: [Option]
    /// Spoken name for the whole control.
    var accessibilityLabel: String

    var body: some View {
        HStack(spacing: RapidTheme.Space.xxs) {
            ForEach(options) { option in
                segment(option)
            }
        }
        .padding(RapidTheme.Space.xxs)
        .background(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.button, style: .continuous)
                .fill(RapidTheme.hoverFill)
        )
        .overlay(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.button, style: .continuous)
                .strokeBorder(RapidTheme.hairline, lineWidth: 1)
        )
        .fixedSize(horizontal: true, vertical: false)
        .accessibilityElement(children: .contain)
        .accessibilityLabel(accessibilityLabel)
    }

    @ViewBuilder
    private func segment(_ option: Option) -> some View {
        let isSelected = selection == option.value
        Button {
            selection = option.value
        } label: {
            Text(option.title)
                .font(RapidFont.body)
                .fontWeight(isSelected ? .semibold : .regular)
                // Reserve the SELECTED (semibold) width at all times so
                // the track does not resize when the selection moves.
                .background(
                    Text(option.title)
                        .font(RapidFont.body)
                        .fontWeight(.semibold)
                        .hidden()
                )
                .foregroundStyle(
                    isSelected ? RapidTheme.brandOnAccent : RapidTheme.textSecondary
                )
                .padding(.horizontal, RapidTheme.Space.md)
                .frame(height: RapidTheme.ControlHeight.segmented - RapidTheme.Space.xs)
                .background(
                    RoundedRectangle(cornerRadius: RapidTheme.Radius.segment, style: .continuous)
                        .fill(isSelected ? RapidTheme.brandPrimary : Color.clear)
                )
                .contentShape(
                    RoundedRectangle(cornerRadius: RapidTheme.Radius.segment, style: .continuous)
                )
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
        .accessibilityIdentifier(option.identifier ?? "")
    }
}
