import Foundation
import Testing

/// Static, network-free checks that every third-party notice which must travel
/// with the binary actually can — and that the mechanism which stages them is
/// exercised, not merely present as text.
///
/// The shipped `.app` (and the DMG around it) is the "distribution" that
/// swift-cmark's BSD-2-Clause and the linked MIT packages ask their notice to
/// accompany. Before #1596 `scripts/build.sh` staged none of them: an accurate
/// `THIRD_PARTY.md` sat in the repo while the download carried no license text
/// at all — a bug nothing in the build could notice.
///
/// Rather than inspect `.build/checkouts` (nondeterministic — a resolved cache
/// that may hold stale entries, or be laid out under a custom scratch path),
/// this suite runs the real staging script, `scripts/stage-licenses.sh`,
/// against constructed fixtures. That deterministically proves both the success
/// path (a package with a license is staged) and the fail-closed path (a
/// package without one aborts the build), and it stays honest if the executable
/// staging call is ever deleted, since the fixtures execute the script itself.
///
/// It also pins the working-tree invariants the real build depends on: the
/// vendored SwiftMath notice exists in-tree, `THIRD_PARTY.md` points at the
/// shipped `Contents/Resources/Licenses/` location, and `build.sh` actually
/// invokes the staging script.
@Suite("Third-party license staging")
struct ThirdPartyLicenseStagingTests {
    /// Repository root, derived from this file's own compile-time location:
    /// `<root>/apps/rapid-mac/Tests/RapidTests/<this file>`.
    private static var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // RapidTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // rapid-mac
            .deletingLastPathComponent()  // apps
            .deletingLastPathComponent()  // <root>
    }

    private static var appRoot: URL {
        repositoryRoot.appendingPathComponent("apps/rapid-mac", isDirectory: true)
    }

    private static var stagingScript: URL {
        appRoot.appendingPathComponent("scripts/stage-licenses.sh")
    }

    // MARK: - Fixture harness

    private struct StagingResult {
        let exitCode: Int32
        let stagedFiles: [String]
        let stderr: String
    }

    /// Run `stage-licenses.sh` against a throwaway fixture tree and report what
    /// it staged. `resolvedBody` is written verbatim as `Package.resolved`;
    /// `checkoutLicenses` maps a checkout directory name to the license file to
    /// drop inside it (nil ⇒ create the directory with no license), and a name
    /// mapped through `omittedCheckouts` is referenced by the resolved file but
    /// its directory is not created at all.
    private static func runStaging(
        resolvedBody: String,
        checkoutLicenses: [String: (filename: String, contents: String)?],
        vendorLicense: String? = "SwiftMath MIT license text",
        createCheckoutsDir: Bool = true
    ) throws -> StagingResult {
        let fm = FileManager.default
        let root = fm.temporaryDirectory
            .appendingPathComponent("lic-fixture-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }

        let resolved = root.appendingPathComponent("Package.resolved")
        try resolvedBody.write(to: resolved, atomically: true, encoding: .utf8)

        let checkouts = root.appendingPathComponent("checkouts", isDirectory: true)
        if createCheckoutsDir {
            try fm.createDirectory(at: checkouts, withIntermediateDirectories: true)
            for (name, license) in checkoutLicenses {
                let dir = checkouts.appendingPathComponent(name, isDirectory: true)
                try fm.createDirectory(at: dir, withIntermediateDirectories: true)
                if let license {
                    try license.contents.write(
                        to: dir.appendingPathComponent(license.filename),
                        atomically: true, encoding: .utf8
                    )
                }
            }
        }

        let vendorLicensePath: String
        if let vendorLicense {
            // Mirror the real source basename (Vendor/SwiftMath/LICENSE) so the
            // staged name is SwiftMath-LICENSE.txt, not SwiftMath-<file>.txt.
            let vendorDir = root.appendingPathComponent("vendor", isDirectory: true)
            try fm.createDirectory(at: vendorDir, withIntermediateDirectories: true)
            let vendor = vendorDir.appendingPathComponent("LICENSE")
            try vendorLicense.write(to: vendor, atomically: true, encoding: .utf8)
            vendorLicensePath = vendor.path
        } else {
            vendorLicensePath = root.appendingPathComponent("does-not-exist").path
        }

        let out = root.appendingPathComponent("Licenses", isDirectory: true)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [
            stagingScript.path,
            resolved.path,
            checkouts.path,
            vendorLicensePath,
            out.path,
        ]
        let errPipe = Pipe()
        process.standardError = errPipe
        process.standardOutput = Pipe()
        try process.run()
        process.waitUntilExit()

        let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
        let staged =
            (try? fm.contentsOfDirectory(atPath: out.path))?.sorted() ?? []
        return StagingResult(
            exitCode: process.terminationStatus,
            stagedFiles: staged,
            stderr: String(data: errData, encoding: .utf8) ?? ""
        )
    }

    /// A minimal `Package.resolved` (v3) body pinning the given repo URLs.
    private static func resolved(locations: [String]) -> String {
        let pins = locations.enumerated().map { index, loc in
            """
                  {
                    "identity" : "pkg\(index)",
                    "kind" : "remoteSourceControl",
                    "location" : "\(loc)",
                    "state" : { "revision" : "deadbeef", "version" : "1.0.0" }
                  }
            """
        }.joined(separator: ",\n")
        return """
            {
              "pins" : [
            \(pins)
              ],
              "version" : 3
            }
            """
    }

    // MARK: - Fixture-driven behavior

    @Test("staging copies each linked package's license plus the vendored one")
    func stagesEveryDeclaredLicense() throws {
        let result = try Self.runStaging(
            resolvedBody: Self.resolved(locations: [
                "https://github.com/example/FakePkg",
                "https://github.com/example/OtherPkg.git",
            ]),
            checkoutLicenses: [
                "FakePkg": (filename: "LICENSE", contents: "MIT for FakePkg"),
                // `.git` suffix in the URL must be stripped to the checkout name,
                // and a `COPYING` (BSD-style) file must be discovered too.
                "OtherPkg": (filename: "COPYING", contents: "BSD for OtherPkg"),
            ]
        )

        #expect(result.exitCode == 0, "staging failed: \(result.stderr)")
        #expect(
            result.stagedFiles == [
                "FakePkg-LICENSE.txt",
                "OtherPkg-COPYING.txt",
                "SwiftMath-LICENSE.txt",
            ],
            "unexpected staged set: \(result.stagedFiles)"
        )
    }

    @Test("staging fails closed when a linked package has no license file")
    func failsWhenPackageHasNoLicense() throws {
        let result = try Self.runStaging(
            resolvedBody: Self.resolved(locations: [
                "https://github.com/example/FakePkg"
            ]),
            checkoutLicenses: ["FakePkg": nil]  // directory exists, no license
        )
        #expect(
            result.exitCode != 0,
            """
            a package without a license must abort the build, but staging \
            returned 0 and staged: \(result.stagedFiles)
            """
        )
    }

    @Test("staging fails closed when a resolved pin has no checkout")
    func failsWhenPinHasNoCheckout() throws {
        let result = try Self.runStaging(
            resolvedBody: Self.resolved(locations: [
                "https://github.com/example/FakePkg"
            ]),
            checkoutLicenses: [:]  // pin referenced, but no FakePkg/ directory
        )
        #expect(
            result.exitCode != 0,
            "a resolved pin with no checkout must abort, but staging returned 0"
        )
    }

    @Test("staging fails closed when no remote pins are present")
    func failsWhenNoPins() throws {
        let result = try Self.runStaging(
            resolvedBody: #"{ "pins" : [], "version" : 3 }"#,
            checkoutLicenses: [:]
        )
        #expect(
            result.exitCode != 0,
            """
            an empty pin set must abort rather than silently staging only the \
            vendored notice
            """
        )
    }

    @Test("staging fails closed when the vendored SwiftMath notice is missing")
    func failsWhenVendoredNoticeMissing() throws {
        let result = try Self.runStaging(
            resolvedBody: Self.resolved(locations: [
                "https://github.com/example/FakePkg"
            ]),
            checkoutLicenses: [
                "FakePkg": (filename: "LICENSE", contents: "MIT for FakePkg")
            ],
            vendorLicense: nil
        )
        #expect(
            result.exitCode != 0,
            "a missing vendored SwiftMath notice must abort the build"
        )
    }

    // MARK: - Working-tree invariants

    @Test("the vendored SwiftMath notice ships from the tree, not a checkout")
    func vendoredSwiftMathLicenseExists() throws {
        let license = Self.appRoot
            .appendingPathComponent("Vendor/SwiftMath/LICENSE")
        let data = try Data(contentsOf: license)
        #expect(
            !data.isEmpty,
            """
            Vendor/SwiftMath/LICENSE is missing or empty; SwiftMath is compiled \
            into the binary and its notice must travel with it (#1596).
            """
        )
    }

    @Test("THIRD_PARTY.md points at the shipped Swift license location")
    func thirdPartyDocReferencesBundledLicenses() throws {
        let doc = Self.appRoot.appendingPathComponent("THIRD_PARTY.md")
        let text = try String(contentsOf: doc, encoding: .utf8)
        #expect(
            text.contains("Contents/Resources/Licenses/"),
            """
            THIRD_PARTY.md must reference Contents/Resources/Licenses/ so the \
            repo document and the shipped bundle cannot disagree (#1596).
            """
        )
    }

    @Test("build.sh actually invokes the license-staging script")
    func buildScriptInvokesStagingScript() throws {
        let script = Self.appRoot.appendingPathComponent("scripts/build.sh")
        let text = try String(contentsOf: script, encoding: .utf8)
        // A real invocation line, not merely a mention in a comment: find a
        // non-comment line that runs the script.
        let invokes = text.split(separator: "\n").contains { rawLine in
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            return !line.hasPrefix("#") && line.contains("scripts/stage-licenses.sh")
        }
        #expect(
            invokes,
            """
            build.sh no longer invokes scripts/stage-licenses.sh on a live \
            (non-comment) line; the license-staging step from #1596 must not be \
            silently removed.
            """
        )
    }
}
