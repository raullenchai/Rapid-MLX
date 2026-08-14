import AppKit
import SwiftUI
import Testing
@testable import Rapid

/// Washes drawn OVER a control's own fill have to let it through.
///
/// ``RapidButtonSurface`` paints hover and pressed as an overlay, so that one
/// pair of values reads correctly over an amber primary, a white secondary,
/// and a borderless tertiary. That only works while both values are
/// translucent.
///
/// Phase UI-2 recast ``hoverFill`` as an opaque plane colour — correct for the
/// twelve rows and chips that paint it BEHIND their content, and silently
/// wrong for the one place that paints it in front. It did not touch the
/// button surface; nothing connected the two. Every button in the app then
/// hovered to a solid block with no label, and pressing it brought the label
/// back, because ``pressedFill`` in the same overlay was still a wash.
@Suite("Control washes")
@MainActor
struct ControlWashTests {

    private func alpha(_ color: Color, in appearance: NSAppearance.Name) -> CGFloat {
        var resolved: CGFloat = -1
        NSAppearance(named: appearance)!.performAsCurrentDrawingAppearance {
            resolved = NSColor(color).usingColorSpace(.sRGB)?.alphaComponent ?? -1
        }
        return resolved
    }

    private let appearances: [NSAppearance.Name] = [.aqua, .darkAqua]

    @Test("Over-content washes are translucent in both appearances")
    func washesLetTheControlThrough() {
        for appearance in appearances {
            let hover = alpha(RapidTheme.hoverWash, in: appearance)
            let pressed = alpha(RapidTheme.pressedFill, in: appearance)
            #expect(
                hover > 0 && hover < 1,
                "hoverWash is \(hover) in \(appearance.rawValue) — an opaque overlay hides the label it should be shading"
            )
            #expect(
                pressed > 0 && pressed < 1,
                "pressedFill is \(pressed) in \(appearance.rawValue)"
            )
        }
    }

    /// The pair has an order: pressed reads firmer than hover. Inverting it
    /// makes a button look like it lifts when you press it.
    @Test("Pressed is firmer than hover")
    func pressedIsFirmerThanHover() {
        for appearance in appearances {
            #expect(
                alpha(RapidTheme.pressedFill, in: appearance)
                    > alpha(RapidTheme.hoverWash, in: appearance)
            )
        }
    }

    /// ``hoverFill`` is deliberately opaque and stays that way — this pins the
    /// distinction so the two tokens are not "cleaned up" back into one.
    @Test("The row hover plane stays opaque")
    func rowHoverPlaneIsOpaque() {
        for appearance in appearances {
            #expect(alpha(RapidTheme.hoverFill, in: appearance) == 1)
        }
    }
}
