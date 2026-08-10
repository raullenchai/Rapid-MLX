import Foundation
import Testing

/// Static, network-free checks that every third-party notice which must travel
/// with the binary actually can.
///
/// The shipped `.app` (and the DMG around it) is the "distribution" that
/// swift-cmark's BSD-2-Clause and the linked MIT packages ask their notice to
/// accompany. Before #1596 `scripts/build.sh` staged none of them: an accurate
/// `THIRD_PARTY.md` sat in the repo while the download carried no license text
/// at all. A `URL(string:)`-style bug — correct on paper, wrong in the artifact
/// — that nothing in the build could notice.
///
/// This suite closes that at `swift test`, reading the working tree only, in
/// the same spirit as `RepositoryLinkTargetsTests`:
///
/// * the vendored SwiftMath notice exists in-tree (it is compiled into the
///   binary and cannot rely on an ephemeral checkout);
/// * every remote package pinned in `Package.resolved` has a discoverable
///   license file in its resolved checkout — the exact precondition
///   `build.sh` hard-fails on, surfaced earlier and faster here;
/// * `THIRD_PARTY.md` points at the shipped `Contents/Resources/Licenses/`
///   location, so the repo document and the bundle cannot silently diverge;
/// * the staging mechanism in `build.sh` is present, so it cannot be deleted
///   without a test going red.
///
/// Scope, stated plainly: this proves the notices are *stageable* from this
/// checkout. It does not run `build.sh` or inspect an assembled bundle — the
/// build itself is the enforcement that the staged files land under
/// `Contents/`; this suite guards the inputs that build depends on.
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

    /// The conventional license filenames `build.sh` searches for, kept in sync
    /// with `find_license_file` in that script.
    private static let licenseFileNames = [
        "LICENSE", "LICENSE.txt", "LICENSE.md", "LICENCE",
        "COPYING", "COPYING.txt", "COPYRIGHT", "NOTICE",
    ]

    private static func containsLicenseFile(in directory: URL) -> Bool {
        let fm = FileManager.default
        return licenseFileNames.contains {
            fm.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
    }

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

    @Test("build.sh still stages third-party licenses and fails closed")
    func buildScriptStagesLicenses() throws {
        let script = Self.appRoot.appendingPathComponent("scripts/build.sh")
        let text = try String(contentsOf: script, encoding: .utf8)
        for marker in [
            "Contents/Resources/Licenses",
            "stage_license",
            "find_license_file",
            "no license file found for Swift package",
        ] {
            #expect(
                text.contains(marker),
                """
                build.sh no longer contains '\(marker)'; the license-staging \
                mechanism from #1596 must not be silently removed.
                """
            )
        }
    }

    /// The remote SPM checkout directory for a pin: the basename of its
    /// repository URL with any trailing `.git` stripped (SPM's on-disk name,
    /// which preserves the upstream repo's casing rather than the lowercased
    /// identity).
    private static func checkoutName(forLocation location: String) -> String {
        var name = URL(string: location)?.lastPathComponent ?? location
        if name.hasSuffix(".git") {
            name = String(name.dropLast(4))
        }
        return name
    }

    private struct ResolvedPin {
        let identity: String
        let checkoutName: String
    }

    private static func resolvedPins() throws -> [ResolvedPin] {
        let resolved = appRoot.appendingPathComponent("Package.resolved")
        let data = try Data(contentsOf: resolved)
        let root = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let pins = root?["pins"] as? [[String: Any]] ?? []
        return pins.compactMap { pin in
            guard let identity = pin["identity"] as? String else { return nil }
            let location = pin["location"] as? String ?? identity
            return ResolvedPin(
                identity: identity,
                checkoutName: checkoutName(forLocation: location)
            )
        }
    }

    @Test("Package.resolved still pins the license-bearing linked packages")
    func resolvedPinsAreRecognised() throws {
        let identities = Set(try Self.resolvedPins().map(\.identity))
        // swift-cmark is the BSD-2-Clause package whose notice-must-travel
        // requirement motivated #1596; guard it explicitly so an over-loose
        // parse cannot pass on an empty set.
        #expect(
            identities.contains("swift-cmark"),
            """
            Package.resolved no longer pins swift-cmark; if a dependency was \
            removed, update THIRD_PARTY.md and this suite together.
            """
        )
        #expect(!identities.isEmpty, "Package.resolved parsed to zero pins.")
    }

    @Test("every resolved remote package carries a discoverable license file")
    func everyResolvedPackageHasALicense() throws {
        let checkouts = Self.appRoot
            .appendingPathComponent(".build/checkouts", isDirectory: true)
        var isDirectory: ObjCBool = false
        guard
            FileManager.default.fileExists(
                atPath: checkouts.path, isDirectory: &isDirectory
            ), isDirectory.boolValue
        else {
            // `swift test` resolves dependencies before building, so the
            // checkouts are normally present. If a non-standard scratch layout
            // hides them, the build-time hard-fail in build.sh remains the
            // backstop; don't fail spuriously here.
            return
        }

        for pin in try Self.resolvedPins() {
            let dir = checkouts.appendingPathComponent(
                pin.checkoutName, isDirectory: true
            )
            guard
                FileManager.default.fileExists(atPath: dir.path)
            else { continue }
            #expect(
                Self.containsLicenseFile(in: dir),
                """
                Swift package '\(pin.identity)' has no license file in its \
                resolved checkout (\(pin.checkoutName)). build.sh would fail to \
                stage its notice and abort the release build. Confirm the \
                upstream ships a LICENSE/COPYING, or vendor the notice.
                """
            )
        }
    }
}
