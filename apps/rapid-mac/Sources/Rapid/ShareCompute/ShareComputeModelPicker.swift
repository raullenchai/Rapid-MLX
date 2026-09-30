import SwiftUI

// MARK: - Presentation

extension View {
    /// Attaches the Share model picker to the selected-model row.
    ///
    /// Presented as a popover rather than an in-layout overlay for two
    /// reasons, both load-bearing rather than cosmetic:
    ///
    ///   * the workbench is a clipped rounded frame, and Paper's menu extends
    ///     past its lower edge over the primary action — an overlay inside the
    ///     clip would be cut off, and removing the clip would cost the
    ///     workbench its shape, and
    ///   * a popover is what gives the menu its platform behaviour for free:
    ///     Escape dismisses, clicking away dismisses, focus moves into the
    ///     menu on open and returns to the row on close, and VoiceOver
    ///     announces a presented surface rather than a pile of buttons that
    ///     appeared underneath the one just pressed.
    ///
    /// The popover's own chrome is the single intentional visual deviation
    /// from Paper on this surface; the menu's fill, rows, and selected
    /// treatment are Paper's exactly.
    func shareComputePicker(
        isPresented: Binding<Bool>,
        models: [ShareComputeLocalModel],
        selectedID: String,
        width: CGFloat,
        onSelect: @escaping (ShareComputeLocalModel) -> Void
    ) -> some View {
        popover(isPresented: isPresented, arrowEdge: .bottom) {
            ShareComputeModelPicker(
                models: models,
                selectedID: selectedID,
                width: width,
                onSelect: { model in
                    onSelect(model)
                    isPresented.wrappedValue = false
                }
            )
        }
    }
}

// MARK: - Picker

/// The list of locally ready models.
///
/// Every row is downloaded and ready — Share never renders a Download Required
/// state, because a model that is not on this Mac cannot be shared from it and
/// downloads belong in Pool. The filter that guarantees this lives in
/// ``ShareComputeLocalModel/readyForSharing(_:)``; this view renders whatever
/// it is handed, so there is exactly one place the rule can be broken and one
/// place to test it.
struct ShareComputeModelPicker: View {
    let models: [ShareComputeLocalModel]
    let selectedID: String
    /// The measured width of the selector row this menu belongs to.
    ///
    /// Paper draws the open menu edge-to-edge with the row that opened it, in
    /// both layouts. This used to be a hard-coded 640pt, which is WIDER than
    /// the selector row at the narrow breakpoint (≈576pt inside a 720pt
    /// window) — the menu overhung the control it belongs to and had nowhere
    /// to go but off the window edge. Tracking the row makes it correct at
    /// every width instead of at one.
    var width: CGFloat
    let onSelect: (ShareComputeLocalModel) -> Void

    /// Floor for the measured width, so a menu presented before the first
    /// layout pass (or in a pathologically narrow window) is still a usable
    /// list rather than a sliver.
    private static let minimumWidth: CGFloat = 320

    var body: some View {
        VStack(spacing: 0) {
            ForEach(Array(models.enumerated()), id: \.element.id) { index, model in
                ShareComputeModelOption(
                    model: model,
                    isSelected: model.id == selectedID,
                    showsDivider: index < models.count - 1,
                    action: { onSelect(model) }
                )
            }
        }
        .padding(6)
        .frame(width: max(Self.minimumWidth, width))
        .background(RapidTheme.bandControl)
        .accessibilityElement(children: .contain)
        .accessibilityLabel(String(localized: "Local models available to share"))
        .accessibilityIdentifier("ShareCompute.ModelPicker")
    }
}

/// One option. The entire row is the hit target.
struct ShareComputeModelOption: View {
    let model: ShareComputeLocalModel
    let isSelected: Bool
    let showsDivider: Bool
    let action: () -> Void

    @State private var isHovering = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 14) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(model.title)
                        .font(.system(size: 14, weight: .bold))
                        .foregroundStyle(RapidTheme.bandInk)
                        .lineLimit(1)
                    Text(model.detail)
                        .font(.system(size: 11))
                        .foregroundStyle(RapidTheme.bandInkSecondary)
                        .lineLimit(1)
                }
                Spacer(minLength: RapidTheme.Space.md)
                if let size = model.onDiskSize {
                    Text(size)
                        .font(.system(size: 12))
                        .foregroundStyle(RapidTheme.bandInkSecondary)
                }
                Text("Ready on this Mac")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(RapidTheme.bandReady)
                // Fixed trailing slot so the tick's presence on one row cannot
                // shift the readiness label on the rows around it.
                Image(systemName: "checkmark")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundStyle(RapidTheme.brandPrimary)
                    .opacity(isSelected ? 1 : 0)
                    .frame(width: 16)
                    .accessibilityHidden(true)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .frame(minHeight: 56)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background {
                if isSelected {
                    RoundedRectangle(cornerRadius: 5)
                        .fill(RapidTheme.bandSelectionFill)
                        .overlay {
                            RoundedRectangle(cornerRadius: 5)
                                .strokeBorder(RapidTheme.bandSelectionStroke, lineWidth: 1)
                        }
                } else if isHovering {
                    RoundedRectangle(cornerRadius: 5).fill(RapidTheme.hoverWash)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { isHovering = $0 }
        .overlay(alignment: .bottom) {
            if showsDivider && !isSelected {
                Rectangle()
                    .fill(RapidTheme.bandHairline)
                    .frame(height: 1)
                    .padding(.horizontal, 12)
            }
        }
        // Selection is stated in the label, not left to the amber tick alone.
        .accessibilityLabel(
            [
                model.title,
                model.onDiskSize,
                String(localized: "Ready on this Mac"),
            ]
            .compactMap { $0 }
            .joined(separator: ", ")
        )
        .accessibilityAddTraits(isSelected ? [.isSelected, .isButton] : .isButton)
        .accessibilityIdentifier("ShareCompute.ModelOption.\(model.id)")
    }
}
