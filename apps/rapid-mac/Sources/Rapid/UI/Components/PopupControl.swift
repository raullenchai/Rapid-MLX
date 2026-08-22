import SwiftUI

/// The chrome every popup control renders through: current value, a
/// chevron, and the app's field surface.
///
/// This exists because AppKit's `NSPopUpButton` — what a raw SwiftUI
/// `Picker` bridges to on macOS — draws its bezel inset a few points
/// inside its layout frame. Put one at the trailing edge of a row and
/// its visible edge floats short of every `.rapidSecondary` button in
/// the same column; the column reads as ragged and no frame math on the
/// call site can fix it. A self-drawn label's visible edge IS its layout
/// edge, so controls built from this chrome sit on the same trailing
/// line as buttons and toggles.
///
/// Use as the `label:` of a `Menu` (with `.menuStyle(.button)`,
/// `.buttonStyle(.plain)`, `.menuIndicator(.hidden)`), which is how the
/// Audio tabs' pickers are built.
struct PopupControlChrome: View {
    let title: String
    var width: CGFloat = 320

    var body: some View {
        HStack(spacing: RapidTheme.Space.sm) {
            Text(title)
                .font(RapidFont.body)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer(minLength: RapidTheme.Space.sm)
            Image(systemName: "chevron.up.chevron.down")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.secondary)
                .accessibilityHidden(true)
        }
        .padding(.horizontal, RapidTheme.Space.md)
        .frame(width: width, height: RapidTheme.ControlHeight.small)
        .background(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous)
                .fill(RapidTheme.surfaceCode)
        )
        .overlay(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous)
                .strokeBorder(RapidTheme.hairlineStrong, lineWidth: 1)
        )
        .contentShape(RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous))
    }
}
