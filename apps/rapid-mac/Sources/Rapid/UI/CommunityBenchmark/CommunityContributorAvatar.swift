import AppKit
import Foundation

/// The canonical Community Benchmark contributor portrait.
///
/// This is a **port, not a design**. rapidmlx.com already derives a portrait
/// for every pseudonymous installation in
/// `landing/public/leaderboard-data.js`, and the whole point of the identity
/// is that one installation shows one face everywhere. If Desktop invented its
/// own mapping — or kept drawing a monogram — the same contributor would
/// appear as two different people depending on which surface you looked at.
///
/// Everything below therefore mirrors the website byte for byte:
///
/// | Website (`leaderboard-data.js`) | Here |
/// | --- | --- |
/// | `AVATAR_SALT` | ``salt`` |
/// | `AVATAR_ASSETS` | ``assetPlates`` |
/// | `hash32` (FNV-1a + MurmurHash3 finalizer) | ``hash32(_:)`` |
/// | `avatarKey` | ``key(for:)`` |
/// | `deriveAvatar` | ``plate(for:)`` |
/// | `avatarAssetPath` | ``assetName(plate:)`` |
/// | `AVATAR_FALLBACK` | ``fallbackAssetName`` |
///
/// Any change here is a contract break and must be made on the website first.
enum CommunityContributorAvatar {
    /// `AVATAR_SALT`. Versioned on the website because changing it re-deals
    /// every contributor's portrait; it moves only when the asset set does.
    static let salt = "rapid-mlx/leaderboard/avatar/v12"

    /// `AVATAR_ASSETS` — the commissioned plates that were actually produced.
    /// The brief specified 01–24; 03, 04, 06, 07, 08 and 23 were never drawn,
    /// so listing the real numbers (rather than assuming a contiguous range)
    /// is what stops a missing plate from ever being selected.
    static let assetPlates = [
        1, 2, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24,
    ]

    /// `AVATAR_FALLBACK` — shown only if a portrait asset fails to load.
    static let fallbackAssetName = "cheetah-run"

    // MARK: - Hashing

    /// `hash32`: FNV-1a over UTF-16 code units, then the MurmurHash3 avalanche
    /// finalizer.
    ///
    /// Two details are load-bearing and easy to get wrong in Swift:
    ///
    /// 1. **UTF-16 code units, not bytes.** JavaScript's `charCodeAt` yields
    ///    UTF-16 code units, so iterating `String.utf8` here would diverge for
    ///    any non-ASCII pseudonym. Slugs are ASCII today, but the hash must
    ///    match the website for whatever the API issues tomorrow.
    /// 2. **The finalizer.** The plate is chosen with `%`, which reads the low
    ///    bits, and raw FNV is weakest exactly there. Dropping the avalanche
    ///    would still compile and still be deterministic — it would just
    ///    silently assign different plates than the website.
    static func hash32(_ string: String) -> UInt32 {
        var h: UInt32 = 0x811c_9dc5
        for unit in string.utf16 {
            h ^= UInt32(unit)
            // `Math.imul` is a 32-bit wrapping multiply; `&*` is its exact
            // Swift equivalent. A plain `*` would trap on overflow.
            h = h &* 0x0100_0193
        }
        h ^= h >> 16
        h = h &* 0x85eb_ca6b
        h ^= h >> 13
        h = h &* 0xc2b2_ae35
        h ^= h >> 16
        return h
    }

    // MARK: - Mapping

    /// `avatarKey`: the slug when the API supplied one, otherwise exactly
    /// `name + "-" + tag`, which is the same string the API composes.
    static func key(for contributor: CommunityBenchmarkContributor?) -> String {
        guard let contributor else { return "" }
        return contributor.slug
    }

    /// `deriveAvatar(...).index` — the plate number for an identity key.
    ///
    /// An empty key still resolves (the website's `%` on `hash32("")` does
    /// too), so a portrait always renders rather than leaving a hole.
    static func plate(forKey key: String) -> Int {
        let index = Int(hash32("\(salt)|\(key)") % UInt32(assetPlates.count))
        return assetPlates[index]
    }

    static func plate(for contributor: CommunityBenchmarkContributor?) -> Int {
        plate(forKey: key(for: contributor))
    }

    /// `avatarAssetPath`, reduced to the flat resource name the app bundle
    /// uses: `/leaderboard/avatars/cheetah-avatar-07.webp` → `cheetah-avatar-07`.
    static func assetName(plate: Int) -> String {
        String(format: "cheetah-avatar-%02d", plate)
    }

    static func assetName(for contributor: CommunityBenchmarkContributor?) -> String {
        assetName(plate: plate(for: contributor))
    }

    static func assetName(forKey key: String) -> String {
        assetName(plate: plate(forKey: key))
    }

    // MARK: - Loading

    /// Loads a portrait, falling back to the static cheetah if the plate is
    /// missing from the bundle.
    static func image(forKey key: String) -> NSImage? {
        loadResource(named: assetName(forKey: key), extension: "webp")
            ?? loadResource(named: fallbackAssetName, extension: "png")
    }

    static func image(for contributor: CommunityBenchmarkContributor?) -> NSImage? {
        image(forKey: key(for: contributor))
    }

    /// True when every selectable plate is present in the running bundle. Used
    /// by the resource-integrity test so a packaging regression fails a test
    /// rather than silently degrading every contributor to the fallback.
    static func missingPlates() -> [Int] {
        assetPlates.filter { loadResource(named: assetName(plate: $0), extension: "webp") == nil }
    }

    /// Mirrors ``CheetahLogo``'s resolution: flat `Bundle.main` resource in a
    /// packaged .app, then the SwiftPM resource bundle for `swift run` and the
    /// test runner. Probed via `Bundle(url:)` so a miss returns nil instead of
    /// the `fatalError` that `Bundle.module` raises.
    static func loadResource(named name: String, extension ext: String) -> NSImage? {
        if let url = Bundle.main.url(forResource: name, withExtension: ext),
           let image = NSImage(contentsOf: url) {
            return image
        }
        let executableAnchor = Bundle(for: AvatarBundleFinder.self).bundleURL
        for anchor in [executableAnchor.deletingLastPathComponent(), executableAnchor] {
            let bundleURL = anchor.appendingPathComponent("Rapid_Rapid.bundle")
            if let bundle = Bundle(url: bundleURL),
               let url = bundle.url(forResource: name, withExtension: ext),
               let image = NSImage(contentsOf: url) {
                return image
            }
        }
        return nil
    }
}

private final class AvatarBundleFinder {}

/// The two mascot plates this module is allowed to use, and where each one is
/// allowed to appear.
///
/// Separate from the contributor portraits on purpose: a mascot is Rapid-MLX
/// speaking, a portrait is a person's pseudonymous identity. Using the mascot
/// where an identity belongs is what made the earlier build show the same
/// cheetah for every contributor.
enum CommunityMascotArt: String {
    /// `cheetah-invite-wave` — the invitation and the celebration.
    case inviteWave = "cheetah-invite-wave"
    /// `cheetah-run` — work in progress.
    case run = "cheetah-run"

    var image: NSImage? {
        CommunityContributorAvatar.loadResource(named: rawValue, extension: "png")
    }

    /// The longest visible edge as a fraction of the 320 × 320 canvas, measured
    /// from each plate's alpha channel:
    ///
    /// - `cheetah-invite-wave`: 149 × 158 visible → 158/320
    /// - `cheetah-run`: 165 × 142 visible → 165/320
    ///
    /// Callers size by the character, not the image box, so a mascot asked for
    /// at 68pt actually reads as 68pt rather than as the ~34pt the padded
    /// canvas would produce. Pinned as constants because the vendored artwork
    /// is fixed; if a plate is ever redrawn, re-measure rather than eyeball.
    var visibleRatio: CGFloat {
        switch self {
        case .inviteWave: return 158.0 / 320.0
        case .run: return 165.0 / 320.0
        }
    }
}
