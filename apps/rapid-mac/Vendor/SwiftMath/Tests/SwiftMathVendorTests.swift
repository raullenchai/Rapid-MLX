import AppKit
import CoreText
import Testing
@testable import SwiftMath

@Suite("Rapid SwiftMath vendor patches")
struct SwiftMathVendorTests {
    @Test("varsigma parses to Greek final sigma")
    func varsigmaGlyph() {
        var error: NSError?
        let list = MTMathListBuilder.build(fromString: #"\varsigma"#, error: &error)
        #expect(error == nil)
        #expect(list?.atoms.count == 1)
        #expect(list?.atoms.first?.nucleus == "\u{03C2}")
    }

    @Test("concurrent CTFont cache access stays coherent")
    func concurrentFontCache() {
        let lock = NSLock()
        var sizes: [CGFloat] = []
        DispatchQueue.concurrentPerform(iterations: 64) { index in
            let expected = CGFloat(12 + index % 4)
            let font = MathFont.latinModernFont.ctFont(withSize: expected)
            lock.withLock { sizes.append(CTFontGetSize(font)) }
        }
        #expect(sizes.count == 64)
        #expect(Set(sizes) == Set([12, 13, 14, 15]))
    }

    @Test("macOS background setter honours its value")
    @MainActor
    func backgroundColorSetter() {
        let view = MTView(frame: .zero)
        view.backgroundColor = .systemRed
        #expect(view.wantsLayer)
        #expect(view.layer?.backgroundColor == NSColor.systemRed.cgColor)

        view.backgroundColor = nil
        #expect(view.layer?.backgroundColor == nil)
    }
}
