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
        let sizes = LockedSizes()
        DispatchQueue.concurrentPerform(iterations: 64) { index in
            let expected = CGFloat(12 + index % 4)
            let font = MathFont.latinModernFont.ctFont(withSize: expected)
            sizes.append(CTFontGetSize(font))
        }
        let observed = sizes.snapshot
        #expect(observed.count == 64)
        #expect(Set(observed) == Set([12, 13, 14, 15]))
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

private final class LockedSizes: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [CGFloat] = []

    func append(_ value: CGFloat) {
        lock.withLock { values.append(value) }
    }

    var snapshot: [CGFloat] {
        lock.withLock { values }
    }
}
