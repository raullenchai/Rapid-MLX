import AppKit
import Foundation

/// The official Rapid-MLX **`R` brand mark**, as vector geometry.
///
/// ## Where this comes from
///
/// The three path strings below are copied **verbatim** from the brand's
/// `rapid_icon.svg` (192 × 192 viewBox, three `<path>` elements filled
/// `#0A0A0A` over a white background rect). Nothing here is traced,
/// redrawn, or approximated — the `d` attributes are the official source
/// data, so re-exporting the logo means replacing three strings and
/// nothing else.
///
/// The white background `<rect>` from the SVG is deliberately NOT
/// reproduced: a menu-bar template must be ink-plus-alpha, and a filled
/// tile is exactly the "coloured square" a status item must never be.
///
/// ## Why this exists alongside ``RapidMark``
///
/// ``RapidMark`` is a hand-drawn "three speed streaks" abstraction whose
/// own documentation says to replace it "when the real vector logo is
/// available". It is still what onboarding renders, and this phase does
/// not touch onboarding — so the two coexist for now. This type is the
/// real mark and is currently consumed only by the menu-bar status item.
///
/// ## The menu-bar variant
///
/// ``menuBarTemplateImage(height:)`` renders the **`R` glyph only**
/// (``glyphPathData``), not the full lockup. The two companion streaks
/// are 5.15pt and 4.60pt tall inside a 105pt-tall mark: scaled to a
/// 16pt menu-bar glyph they become 0.78pt and 0.70pt — under one point,
/// i.e. sub-pixel at @1x — and render as grey mush rather than as
/// strokes. Dropping them keeps the mark legible at the size it actually
/// ships at, which is the whole point of a dedicated menu-bar variant.
/// ``fullMarkPath()`` keeps the complete lockup available for any
/// surface with the room for it.
///
/// The `R`'s counter (the enclosed negative space in the bowl) is formed
/// by the glyph path folding back on itself and fills correctly under the
/// non-zero winding rule, which is ``NSBezierPath``'s default.
enum RapidRMark {

    // MARK: - Official geometry (verbatim from rapid_icon.svg)

    /// The `R` itself — `<path>` 1 of 3.
    static let glyphPathData =
        "M68.0642 43.2444C68.171 43.1963 68.7052 43.1736 69.6668 43.1763C80.4041 43.1923 98.0459 43.1963 122.592 43.1883C126.006 43.1883 131.435 43.0721 135.501 43.1362C144.291 43.2684 152.084 47.4231 157.693 54.2902C160.22 57.3832 162.011 60.5283 163.066 63.7255C164.524 68.1486 165.401 73.313 164.816 78.1568C164.338 82.1446 163.691 85.279 162.873 87.56C160.694 93.6605 157.052 98.7314 151.948 102.773C147.434 106.346 142.213 108.289 136.286 108.602C136.169 108.61 133.983 108.701 129.728 108.874C129.639 108.878 129.553 108.906 129.479 108.956C129.405 109.006 129.347 109.075 129.31 109.157C129.274 109.238 129.261 109.328 129.273 109.416C129.286 109.504 129.322 109.587 129.379 109.656L160.437 147.645C160.477 147.693 160.503 147.751 160.512 147.812C160.522 147.874 160.515 147.937 160.491 147.995C160.468 148.053 160.43 148.104 160.38 148.142C160.33 148.181 160.271 148.206 160.209 148.214C159.023 148.358 157.995 148.428 157.124 148.422C149.044 148.369 140.965 148.362 132.885 148.402C131.859 148.408 130.894 148.369 129.988 148.286C129.915 148.28 129.843 148.258 129.778 148.222C129.712 148.186 129.654 148.137 129.608 148.078L85.6686 91.8349C85.6537 91.8154 85.6436 91.7925 85.6394 91.7684C85.6352 91.7442 85.637 91.7195 85.6445 91.6964C85.6521 91.6734 85.6652 91.6527 85.6827 91.6363C85.7002 91.6199 85.7215 91.6083 85.7447 91.6025C86.3644 91.4422 86.92 91.3594 87.4114 91.3541C92.4542 91.3167 105.769 91.3167 127.356 91.3541C130.233 91.3594 132.17 91.2246 133.169 90.9494C138.674 89.439 142.5 84.4269 143.025 78.8139C143.479 73.9473 141.966 69.8327 138.486 66.4699C135.485 63.5692 131.955 62.5075 127.953 62.4995C101.251 62.4594 84.2997 62.4728 77.0988 62.5396C76.0451 62.5476 74.0338 62.6638 73.4128 61.7984C69.2167 55.9356 64.8697 50.0981 60.3718 44.2861C60.0913 43.9282 59.915 43.6477 59.8429 43.4447C59.827 43.3985 59.8224 43.3492 59.8293 43.3009C59.8362 43.2525 59.8545 43.2065 59.8826 43.1666C59.9108 43.1267 59.948 43.0941 59.9913 43.0714C60.0345 43.0487 60.0825 43.0366 60.1314 43.0361L66.1731 43C66.2131 43 66.2533 43.004 66.2933 43.012C67.3697 43.215 67.96 43.2925 68.0642 43.2444Z"

    /// Upper speed streak — `<path>` 2 of 3.
    static let upperStreakPathData =
        "M105.565 77.6575L102.268 82.2489C102.242 82.285 102.207 82.3144 102.168 82.3346C102.128 82.3548 102.084 82.3653 102.04 82.3651H45.8888C45.8374 82.3646 45.7872 82.3499 45.7435 82.3227C45.6999 82.2956 45.6646 82.2569 45.6414 82.211C45.6182 82.1651 45.6081 82.1137 45.6122 82.0625C45.6163 82.0112 45.6344 81.9621 45.6645 81.9204L48.5051 77.962C48.5308 77.9264 48.5645 77.8974 48.6034 77.8772C48.6424 77.857 48.6856 77.8463 48.7294 77.8458L105.337 77.2168C105.388 77.2171 105.438 77.2314 105.482 77.2581C105.525 77.2848 105.561 77.3228 105.584 77.3682C105.608 77.4136 105.618 77.4645 105.615 77.5155C105.612 77.5665 105.594 77.6156 105.565 77.6575Z"

    /// Lower speed streak — `<path>` 3 of 3.
    static let lowerStreakPathData =
        "M27.1184 90.8985L29.5142 87.5691C29.5727 87.4876 29.6496 87.421 29.7388 87.3751C29.828 87.3291 29.9268 87.305 30.0271 87.3047H77.1552C77.2764 87.3045 77.3951 87.3392 77.4972 87.4045C77.5993 87.4699 77.6805 87.5632 77.7311 87.6733C77.7817 87.7834 77.7997 87.9057 77.7828 88.0258C77.766 88.1458 77.715 88.2585 77.636 88.3504L74.7513 91.6797C74.6922 91.7485 74.6191 91.8037 74.5369 91.8417C74.4546 91.8797 74.3652 91.8996 74.2745 91.9001H27.6312C27.515 91.8998 27.4011 91.8674 27.302 91.8067C27.2029 91.7459 27.1225 91.659 27.0696 91.5556C27.0166 91.4521 26.9931 91.3361 27.0017 91.2202C27.0104 91.1043 27.0507 90.993 27.1184 90.8985Z"

    // MARK: - Menu-bar sizing

    /// Height of the drawn glyph, in points, inside the status bar's own
    /// (24pt) slot. Apple's own template glyphs sit in the 15–18pt band;
    /// 16 keeps the `R` present without crowding the bar's edges, and
    /// lands the mark's near-square aspect on a whole-point 16 × 16 box.
    static let menuBarGlyphHeight: CGFloat = 16

    // MARK: - Paths

    /// The `R` alone, in SVG (y-down) coordinates.
    static func glyphPath() -> NSBezierPath? {
        SVGPathData.parse(glyphPathData)
    }

    /// The complete official lockup — `R` plus both streaks — in SVG
    /// coordinates. Not used by the menu bar (see the type docs); kept so
    /// a surface with room for the full mark has one source for it.
    static func fullMarkPath() -> NSBezierPath? {
        guard let glyph = SVGPathData.parse(glyphPathData),
              let upper = SVGPathData.parse(upperStreakPathData),
              let lower = SVGPathData.parse(lowerStreakPathData) else { return nil }
        let combined = NSBezierPath()
        combined.append(glyph)
        combined.append(upper)
        combined.append(lower)
        return combined
    }

    // MARK: - Menu-bar image

    /// A **template** ``NSImage`` of the `R`, sized for the menu bar.
    ///
    /// Resolution-independent: the drawing handler re-runs at whatever
    /// backing scale the display asks for, so the mark is redrawn from
    /// vectors at @1x, @2x and @3x rather than being resampled from one
    /// bitmap. Ink is opaque black with alpha coverage and nothing else —
    /// ``isTemplate`` then lets AppKit paint it black in a light bar,
    /// white in a dark bar, and white again while the menu is open.
    ///
    /// Returns `nil` only if the path data fails to parse, which can
    /// happen exactly once: when somebody replaces the geometry above
    /// with an export using a path command this parser does not cover.
    /// The caller keeps its SF Symbol fallback for that case rather than
    /// installing an empty (invisible) status item.
    static func menuBarTemplateImage(height: CGFloat = menuBarGlyphHeight) -> NSImage? {
        guard let path = glyphPath() else { return nil }
        let bounds = path.bounds
        guard bounds.height > 0, bounds.width > 0 else { return nil }

        let scale = height / bounds.height
        let size = NSSize(
            width: (bounds.width * scale).rounded(),
            height: height
        )

        let image = NSImage(size: size, flipped: false) { rect in
            guard let redrawn = glyphPath() else { return false }
            let transform = NSAffineTransform()
            // Fit the glyph's own bounds to `rect`, flipping y because the
            // SVG's origin is top-left and AppKit's is bottom-left here.
            transform.translateX(by: rect.minX, yBy: rect.maxY)
            transform.scaleX(by: rect.width / bounds.width, yBy: -(rect.height / bounds.height))
            transform.translateX(by: -bounds.minX, yBy: -bounds.minY)
            redrawn.transform(using: transform as AffineTransform)
            NSColor.black.setFill()
            redrawn.fill()
            return true
        }
        // The load-bearing line: macOS owns the colour, so the mark is
        // never "permanently amber" and carries no lifecycle state.
        image.isTemplate = true
        return image
    }
}

// MARK: - SVG path data

/// A deliberately small SVG `d`-attribute reader.
///
/// It covers exactly the command set the official mark uses — `M`, `L`,
/// `H`, `V`, `C`, `Z`, plus their relative forms — and returns `nil` for
/// anything else instead of silently skipping it. That choice matters:
/// a parser that ignores an unknown command renders a subtly wrong logo
/// and nobody notices for a release, whereas a `nil` trips the caller's
/// fallback and the tests below.
///
/// Not a general-purpose SVG parser and not intended to become one. If a
/// future export needs arcs or smooth curves, teach it those commands
/// deliberately — do not make it lenient.
enum SVGPathData {

    static func parse(_ d: String) -> NSBezierPath? {
        var scanner = Scanner(d)
        let path = NSBezierPath()
        var current = CGPoint.zero
        var subpathStart = CGPoint.zero
        var command: Character?
        var started = false

        while true {
            scanner.skipSeparators()
            if scanner.isAtEnd { break }

            if let letter = scanner.peekCommandLetter() {
                scanner.advance()
                command = letter
            }
            guard let active = command else { return nil }
            let relative = active.isLowercase

            switch Character(active.lowercased()) {
            case "m":
                guard let x = scanner.number(), let y = scanner.number() else { return nil }
                current = relative ? CGPoint(x: current.x + x, y: current.y + y) : CGPoint(x: x, y: y)
                path.move(to: current)
                subpathStart = current
                started = true
                // Per the SVG spec, coordinates repeated after a moveto are
                // implicit linetos.
                command = relative ? "l" : "L"

            case "l":
                guard started, let x = scanner.number(), let y = scanner.number() else { return nil }
                current = relative ? CGPoint(x: current.x + x, y: current.y + y) : CGPoint(x: x, y: y)
                path.line(to: current)

            case "h":
                guard started, let x = scanner.number() else { return nil }
                current = CGPoint(x: relative ? current.x + x : x, y: current.y)
                path.line(to: current)

            case "v":
                guard started, let y = scanner.number() else { return nil }
                current = CGPoint(x: current.x, y: relative ? current.y + y : y)
                path.line(to: current)

            case "c":
                guard started,
                      let x1 = scanner.number(), let y1 = scanner.number(),
                      let x2 = scanner.number(), let y2 = scanner.number(),
                      let x = scanner.number(), let y = scanner.number() else { return nil }
                let c1 = relative ? CGPoint(x: current.x + x1, y: current.y + y1) : CGPoint(x: x1, y: y1)
                let c2 = relative ? CGPoint(x: current.x + x2, y: current.y + y2) : CGPoint(x: x2, y: y2)
                let end = relative ? CGPoint(x: current.x + x, y: current.y + y) : CGPoint(x: x, y: y)
                path.curve(to: end, controlPoint1: c1, controlPoint2: c2)
                current = end

            case "z":
                guard started else { return nil }
                path.close()
                current = subpathStart

            default:
                return nil
            }
        }
        return path.isEmpty ? nil : path
    }

    /// Character-wise cursor over the `d` string.
    private struct Scanner {
        private let characters: [Character]
        private var index = 0

        init(_ string: String) {
            characters = Array(string)
        }

        var isAtEnd: Bool { index >= characters.count }

        mutating func advance() { index += 1 }

        mutating func skipSeparators() {
            while index < characters.count,
                  characters[index] == " " || characters[index] == ","
                    || characters[index] == "\n" || characters[index] == "\r"
                    || characters[index] == "\t" {
                index += 1
            }
        }

        mutating func peekCommandLetter() -> Character? {
            skipSeparators()
            guard index < characters.count else { return nil }
            let c = characters[index]
            return c.isLetter ? c : nil
        }

        /// A single SVG number. Handles a leading sign, a bare leading
        /// decimal point, and scientific notation.
        mutating func number() -> CGFloat? {
            skipSeparators()
            let start = index
            if index < characters.count, characters[index] == "-" || characters[index] == "+" {
                index += 1
            }
            var sawDigit = false
            while index < characters.count, characters[index].isNumber {
                index += 1
                sawDigit = true
            }
            if index < characters.count, characters[index] == "." {
                index += 1
                while index < characters.count, characters[index].isNumber {
                    index += 1
                    sawDigit = true
                }
            }
            if sawDigit, index < characters.count,
               characters[index] == "e" || characters[index] == "E" {
                let beforeExponent = index
                index += 1
                if index < characters.count,
                   characters[index] == "-" || characters[index] == "+" {
                    index += 1
                }
                var sawExponentDigit = false
                while index < characters.count, characters[index].isNumber {
                    index += 1
                    sawExponentDigit = true
                }
                if !sawExponentDigit { index = beforeExponent }
            }
            guard sawDigit, let value = Double(String(characters[start..<index])) else {
                index = start
                return nil
            }
            return CGFloat(value)
        }
    }
}
