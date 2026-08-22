import SwiftUI

/// The coloured monogram tile that anchors every model row + card in the
/// Models surface (issue #507). A rounded square with a two-letter mark,
/// filled with a per-brand gradient — the fast visual "scan by family"
/// affordance Superwhisper's brand column validated.
///
/// These are **self-drawn monograms, not vendor logos.** We ship no
/// per-vendor logo assets and won't add bitmaps for a catalog that grows
/// every week; a coloured letter tile is honest, asset-free, and reads
/// at a glance. The brand → case mapping is the pure
/// ``ModelBrandStyle.brand(forAlias:)``; this view owns only the colour.
struct BrandIcon: View {
    let alias: String
    var size: CGFloat = 30

    private var brand: ModelBrand { ModelBrandStyle.brand(forAlias: alias) }
    private var monogram: String { ModelBrandStyle.monogram(forAlias: alias) }

    var body: some View {
        RoundedRectangle(cornerRadius: size * 0.27, style: .continuous)
            .fill(
                LinearGradient(
                    colors: Self.gradient(for: brand),
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            )
            .frame(width: size, height: size)
            .overlay(
                Text(monogram)
                    .font(.system(size: size * 0.37, weight: .heavy, design: .rounded))
                    .foregroundStyle(.white)
                    .minimumScaleFactor(0.6)
                    .lineLimit(1)
                    .padding(.horizontal, 1)
            )
            .accessibilityHidden(true)
    }

    /// Per-brand two-stop gradient. Colours are chosen to be distinct at
    /// a glance and roughly echo each vendor's identity hue, without
    /// claiming to be an official logo. `other` is a neutral graphite so
    /// an unlisted family never borrows a wrong brand colour.
    static func gradient(for brand: ModelBrand) -> [Color] {
        switch brand {
        case .qwen:     return [hex(0x5B5BF0), hex(0x3D3DD6)] // indigo
        case .llama:    return [hex(0x0A84FF), hex(0x0866D6)] // sky
        case .gemma:    return [hex(0xFF9F0A), hex(0xF57C00)] // amber
        case .gptOss:   return [hex(0x12B886), hex(0x0E9E76)] // green
        case .deepseek: return [hex(0x4D6BFE), hex(0x3B54D6)] // blue
        case .mistral:  return [hex(0xFF7043), hex(0xF4511E)] // orange-red
        case .phi:      return [hex(0x5A6572), hex(0x3E4650)] // slate
        case .glm:      return [hex(0x8E5BF0), hex(0x6D3DD6)] // violet
        case .smollm:   return [hex(0x1AA6B7), hex(0x14808C)] // teal
        case .hermes:   return [hex(0x9B59D0), hex(0x7B3FB0)] // purple
        case .ornith:   return [hex(0x2F9E73), hex(0x237A59)] // forest green
        case .other:    return [hex(0x8A8A93), hex(0x6E6E77)] // graphite
        }
    }

    private static func hex(_ rgb: UInt32) -> Color {
        Color(
            red: Double((rgb >> 16) & 0xFF) / 255.0,
            green: Double((rgb >> 8) & 0xFF) / 255.0,
            blue: Double(rgb & 0xFF) / 255.0
        )
    }
}
