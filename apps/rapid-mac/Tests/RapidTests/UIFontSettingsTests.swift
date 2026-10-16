import Foundation
import SwiftUI
import Testing
@testable import Rapid

@Suite("UI font settings")
struct UIFontSettingsTests {
    private func makeDefaults() -> UserDefaults {
        let name = "UIFontSettingsTests.\(UUID().uuidString)"
        let d = UserDefaults(suiteName: name)!
        d.removePersistentDomain(forName: name)
        return d
    }

    @Test("never returns less than 12pt and scales larger sizes")
    func floorAndScale() {
        let s = UIFontSettings(defaults: makeDefaults())
        #expect(s.scale == 1.0)
        #expect(s.pointSize(9) == 12)
        #expect(s.pointSize(11) == 12)
        #expect(s.pointSize(13) == 13)
        s.scale = 1.5
        #expect(s.pointSize(8) == 12)
        #expect(s.pointSize(10) == 15)
        #expect(s.pointSize(20) == 30)
    }

    @Test("persists the chosen scale")
    func persists() {
        let d = makeDefaults()
        UIFontSettings(defaults: d).scale = 1.3
        #expect(UIFontSettings(defaults: d).scale == 1.3)
    }

    @Test("setScale clamps to 1.0–1.5 and rounds to 0.01; out-of-range saved values fall back")
    func clampAndRound() {
        let d = makeDefaults()
        let s = UIFontSettings(defaults: d)
        s.setScale(2)
        #expect(s.scale == 1.5)
        s.setScale(0.5)
        #expect(s.isDefault)
        s.setScale(1.234)
        #expect(s.scale == 1.23)
        #expect(UIFontScaleControl.label(1.1) == "1.1×")
        d.set(9.0, forKey: UIFontSettings.storageKey)
        #expect(UIFontSettings(defaults: d).scale == 1.0)
    }

    /// Chat body, code, code-block header and composer all route through
    /// the shared setting. At the default multiplier they keep their
    /// designed sizes, except the 11pt header which the floor lifts to 12.
    @Test("chat surfaces use the shared setting and keep their default sizes")
    @MainActor
    func chatSurfacesAtDefault() {
        guard UIFontSettings.shared.isDefault else { return }
        let options = TextKitMarkdownView.options(basePointSize: 15)
        #expect(options.textPointSize == 15)
        #expect(options.codePointSize == 13)
        #expect(options.codeLineHeight == 16)
        #expect(MarkdownCodeBlockView.headerFont.pointSize == 12)
        #expect(ComposeTextEditor.editorFont.pointSize == 15)
    }

    @Test("every text style resolves to at least the floor")
    func textStylesFloored() {
        let s = UIFontSettings(defaults: makeDefaults())
        for style in [Font.TextStyle.caption2, .caption, .footnote, .subheadline, .callout] {
            #expect(s.pointSize(UIFontSettings.basePointSize(style)) >= UIFontSettings.minimumPointSize)
        }
    }
}

extension UIFontSettingsTests {
    @Test("font floor holds across the supported text size range")
    func floorAcrossSupportedRange() {
        let settings = UIFontSettings(defaults: makeDefaults())
        for step in 0...50 {
            settings.setScale(1 + Double(step) / 100)
            for base in 0...40 {
                #expect(settings.pointSize(CGFloat(base)) >= 12)
            }
        }
    }

    @Test("layout slots follow the multiplier without applying a text floor")
    func scaledSlots() {
        let settings = UIFontSettings(defaults: makeDefaults())
        for scale in [1.0, 1.1, 1.25, 1.5] {
            settings.setScale(scale)
            #expect(settings.scaled(0) == 0)
            #expect(settings.scaled(2) == 2 * CGFloat(scale))
            #expect(settings.scaled(22) == 22 * CGFloat(scale))
            #expect(settings.scaled(22) > settings.pointSize(10))
        }
    }

    /// Fonts are inherited by containers and include RapidFont and
    /// scaledSystemFont, so reject shrinking anywhere in the UI tree.
    /// Only a literal 1 is allowed; expressions can conceal a sub-1 factor.
    private static func shrinkingModifiers(in source: String) throws -> [String] {
        let code = source.split(separator: "\n", omittingEmptySubsequences: false)
            .filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("//") }
            .joined(separator: "\n")
        let regex = try NSRegularExpression(pattern: #"\.minimumScaleFactor\s*\(\s*([^)]*)\)"#)
        let nsCode = code as NSString
        return regex.matches(in: code, range: NSRange(location: 0, length: nsCode.length))
            .compactMap { match in
                let argument = nsCode.substring(with: match.range(at: 1))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                return Double(argument) == 1 ? nil : nsCode.substring(with: match.range)
            }
    }

    @Test("shrink guard detects direct, inherited, and expression-based factors")
    func shrinkGuardFixtures() throws {
        for source in [
            #"Text("label").font(.rapid(size: 10)).minimumScaleFactor(0.8)"#,
            #"Text("label").minimumScaleFactor(0.8).font(.rapid(.caption))"#,
            "VStack { Text(value) }.font(RapidFont.caption).minimumScaleFactor(0.75)",
            "Text(value).scaledSystemFont(10).minimumScaleFactor(narrow ? 0.8 : 1)",
            "Text(value).minimumScaleFactor(ModelTableLayout.cellMinimumScaleFactor)",
            "Text(value).minimumScaleFactor(\n 0.9\n)",
        ] {
            #expect(try Self.shrinkingModifiers(in: source).count == 1)
        }
        #expect(try Self.shrinkingModifiers(in: "Text(value).minimumScaleFactor(1.0)").isEmpty)
        #expect(try Self.shrinkingModifiers(in: "// .minimumScaleFactor(0.8)").isEmpty)
        #expect(try Self.shrinkingModifiers(in: "Text(value).font(.rapid(.caption)).lineLimit(1)").isEmpty)
    }

    @Test("UI sources never shrink text below the chosen floored font")
    func uiSourcesDoNotShrink() throws {
        let root = DynamicTypeScaledFontTests.sourceRoot.appendingPathComponent("Sources/Rapid")
        let files = try #require(FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil))
        var scanned = 0
        for case let file as URL in files where file.pathExtension == "swift" {
            scanned += 1
            let source = try String(contentsOf: file, encoding: .utf8)
            #expect(try Self.shrinkingModifiers(in: source).isEmpty, "Text can shrink in \(file.path)")
        }
        #expect(scanned > 100, "Source guard must scan the entire Rapid tree")
    }
}

extension UIFontSettingsTests {
    /// Match only an Image's own modifier chain, stopping at a container
    /// boundary or another view. One nested call covers computed symbols,
    /// colors and frame arguments without scanning a neighboring Text.
    private static func scaledImageFonts(in source: String) throws -> Int {
        let code = source.split(separator: "\n", omittingEmptySubsequences: false)
            .filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("//") }
            .joined(separator: "\n")
        let argument = #"(?:[^(){}]|\([^(){}]*\))*"#
        let pattern = #"\bImage\s*\([ARG]\)(?:\s*\.\w+\([ARG]\))*?\s*(?:\.font\s*\(\s*(?:\.rapid\s*\(|RapidFont\.)|\.scaledSystemFont\s*\()"#
            .replacingOccurrences(of: "[ARG]", with: argument)
        let regex = try NSRegularExpression(pattern: pattern)
        return regex.numberOfMatches(in: code, range: NSRange(code.startIndex..., in: code))
    }

    @Test("image font guard follows modifiers without matching neighboring text")
    func imageFontGuardFixtures() throws {
        for source in [
            #"Image(systemName: "pin.fill").font(.rapid(size: 9))"#,
            "Image(\n systemName: symbol(for: state)\n)\n .foregroundStyle(.secondary)\n .font(.rapid(.caption))",
            #"Image(systemName: "checkmark").accessibilityHidden(true).font(RapidFont.caption)"#,
            #"Image(systemName: "magnifyingglass").scaledSystemFont(32, relativeTo: .title)"#,
            "Image(systemName: symbol)\n .foregroundStyle(.secondary)\n .scaledSystemFont(12)",
            #"Image("logo").scaledSystemFont(12)"#,
            "Image(nsImage: icon).font(.rapid(.caption))",
            #"Image(decorative: "mark").resizable().font(RapidFont.caption)"#,
        ] {
            #expect(try Self.scaledImageFonts(in: source) == 1)
        }
        for source in [
            #"Image(systemName: "pin.fill").font(.system(size: 9))"#,
            "Image(systemName: symbol)\nText(title).font(.rapid(size: 13))",
            "Image(systemName: symbol)\nText(title).scaledSystemFont(13)",
            #"HStack { Image(systemName: symbol); Text(title) }.font(.rapid(.body))"#,
            #"// Image(systemName: "pin.fill").font(.rapid(size: 9))"#,
        ] {
            #expect(try Self.scaledImageFonts(in: source) == 0)
        }
    }

    @Test("decorative images never use UI text scaling directly")
    func uiImagesKeepFixedFonts() throws {
        let root = DynamicTypeScaledFontTests.sourceRoot.appendingPathComponent("Sources/Rapid")
        let files = try #require(FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil))
        var scanned = 0
        for case let file as URL in files where file.pathExtension == "swift" {
            scanned += 1
            let source = try String(contentsOf: file, encoding: .utf8)
            #expect(try Self.scaledImageFonts(in: source) == 0, "Scaled decorative image in \(file.path)")
        }
        #expect(scanned > 100, "Source guard must scan the entire Rapid tree")
    }
}
