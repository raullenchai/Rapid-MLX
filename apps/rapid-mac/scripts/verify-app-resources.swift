#!/usr/bin/env swift
import Foundation
import AppKit
import CoreGraphics

// Promoted from the ad-hoc /tmp/probe_cheetah.swift used to verify v0.5.10's
// fix for the v0.5.9 ship-blocker (SPM Bundle.module accessor fatalError'd
// inside the wrapped .app — see memory/gotcha_spm_bundle_module_app_wrapper.md).
//
// Contract: every PNG declared in Package.swift's executable-target
// `resources:` block must resolve via Bundle.main.url(forResource:withExtension:)
// against the assembled Rapid.app, AND decode as a valid NSImage. Either miss
// is a release-blocker.

func die(_ msg: String, code: Int32 = 1) -> Never {
    FileHandle.standardError.write(Data("\(msg)\n".utf8))
    exit(code)
}

guard CommandLine.arguments.count >= 2 else {
    die("usage: verify-app-resources.swift <path-to-Rapid.app>", code: 2)
}

let appPath = CommandLine.arguments[1]
let appURL = URL(fileURLWithPath: appPath)
guard FileManager.default.fileExists(atPath: appURL.path) else {
    die("FAIL: \(appPath) does not exist")
}
guard let bundle = Bundle(url: appURL) else {
    die("FAIL: Bundle(url:) returned nil for \(appPath)")
}

// Source-of-truth: Package.swift's resources: block. We parse it instead of
// hardcoding so a future PNG addition can't silently bypass this gate.
let scriptURL = URL(fileURLWithPath: CommandLine.arguments[0])
let repoRoot = scriptURL.deletingLastPathComponent().deletingLastPathComponent()
let packageURL = repoRoot.appendingPathComponent("Package.swift")
guard let packageSrc = try? String(contentsOf: packageURL, encoding: .utf8) else {
    die("FAIL: cannot read \(packageURL.path)")
}

// Match e.g. .process("Resources/cheetah.png") or .copy("Resources/foo.png").
let pattern = #"\.(?:process|copy)\(\"Resources/([^\"]+\.png)\"\)"#
let regex = try! NSRegularExpression(pattern: pattern)
let nsSrc = packageSrc as NSString
let matches = regex.matches(in: packageSrc, range: NSRange(location: 0, length: nsSrc.length))
var pngNames: [String] = []
for m in matches where m.numberOfRanges >= 2 {
    let full = nsSrc.substring(with: m.range(at: 1))
    pngNames.append((full as NSString).deletingPathExtension)
}
if pngNames.isEmpty {
    die("FAIL: no PNG resources parsed from \(packageURL.path) — regex drift?")
}

print("verify-app-resources: bundle=\(bundle.bundleURL.lastPathComponent) assets=\(pngNames.count)")

var ok = true
for name in pngNames {
    guard let url = bundle.url(forResource: name, withExtension: "png") else {
        FileHandle.standardError.write(Data("FAIL: \(name).png NOT found via Bundle.url(forResource:)\n".utf8))
        ok = false
        continue
    }
    guard let img = NSImage(contentsOf: url) else {
        FileHandle.standardError.write(Data("FAIL: NSImage decode returned nil for \(url.path)\n".utf8))
        ok = false
        continue
    }
    let s = img.size
    if s.width <= 0 || s.height <= 0 {
        FileHandle.standardError.write(Data("FAIL: degenerate image size \(s.width)x\(s.height) for \(name).png\n".utf8))
        ok = false
        continue
    }
    print("OK: \(name).png \(Int(s.width))x\(Int(s.height)) at \(url.path)")
}

// Localizable.xcstrings: declared as an SPM resource in Package.swift
// but historically the production .app shipped without it (Bundle.main
// only sees flat resources in Contents/Resources/, not the SPM
// Rapid_Rapid.bundle which codesign rejects at the .app root). This
// gate catches the regression — if the catalog isn't bundled, zh-Hans
// users will see English even though `swift test` reports translated.
if packageSrc.contains("Localizable.xcstrings") {
    if let url = bundle.url(forResource: "Localizable", withExtension: "xcstrings"),
       let data = try? Data(contentsOf: url),
       let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
       let strings = json["strings"] as? [String: Any], !strings.isEmpty {
        print("OK: Localizable.xcstrings keys=\(strings.count) at \(url.path)")
    } else {
        FileHandle.standardError.write(Data("FAIL: Localizable.xcstrings NOT found or unreadable in bundle\n".utf8))
        ok = false
    }
}

// SwiftMath must resolve from the assembled app itself. A development build
// can silently fall back to an absolute `.build` checkout, which is precisely
// why the missing shipped resource escaped earlier validation.
if let fontsURL = bundle.url(forResource: "mathFonts", withExtension: "bundle"),
   let fonts = Bundle(url: fontsURL),
   let fontURL = fonts.url(forResource: "latinmodern-math", withExtension: "otf"),
   let tableURL = fonts.url(forResource: "latinmodern-math", withExtension: "plist"),
   let provider = CGDataProvider(url: fontURL as CFURL),
   CGFont(provider) != nil,
   let table = NSDictionary(contentsOf: tableURL),
   (table["version"] as? String) == "1.3",
   fonts.url(forResource: "OFL", withExtension: "txt") != nil,
   fonts.url(forResource: "GUST-FONT-LICENSE", withExtension: "txt") != nil {
    print("OK: SwiftMath fonts and licence notices at \(fontsURL.path)")
} else {
    FileHandle.standardError.write(Data("FAIL: SwiftMath font bundle is missing, invalid, or lacks licence notices\n".utf8))
    ok = false
}

exit(ok ? 0 : 1)
