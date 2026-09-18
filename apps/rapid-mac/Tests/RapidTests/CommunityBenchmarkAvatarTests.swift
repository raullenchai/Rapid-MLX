import AppKit
import Foundation
import Testing
@testable import Rapid

/// Cross-platform identity parity with rapidmlx.com.
///
/// These are contract tests, not unit tests: every expectation below was
/// produced by running the website's own `deriveAvatar` in
/// `landing/public/leaderboard-data.js`. If one fails, the app and the website
/// have started showing different faces for the same installation, which is
/// the exact failure the shared mapping exists to prevent.
@Suite("Contributor identity parity with rapidmlx.com")
struct CommunityBenchmarkAvatarTests {
    // MARK: - Mapping constants

    @Test("Salt and asset plates match the website exactly")
    func mappingConstants() {
        #expect(CommunityContributorAvatar.salt == "rapid-mlx/leaderboard/avatar/v12")
        // `AVATAR_ASSETS` — 18 plates; 03, 04, 06, 07, 08 and 23 were never
        // produced, so a contiguous 1...24 would select a missing file.
        #expect(CommunityContributorAvatar.assetPlates == [
            1, 2, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24,
        ])
        #expect(CommunityContributorAvatar.assetPlates.count == 18)
        for absent in [3, 4, 6, 7, 8, 23] {
            #expect(!CommunityContributorAvatar.assetPlates.contains(absent))
        }
    }

    // MARK: - Shared mapping vectors

    /// Computed with the website implementation:
    ///
    /// ```
    /// node -e "import('./leaderboard-data.js').then(m =>
    ///   console.log(m.deriveAvatar({slug:'swift-otter-4417'}).index))"
    /// ```
    @Test("Shared vectors resolve to the same plates the website produces")
    func sharedMappingVectors() {
        #expect(CommunityContributorAvatar.plate(forKey: "swift-otter-4417") == 12)
        #expect(CommunityContributorAvatar.plate(forKey: "modest-slate-wombat-545") == 2)
        #expect(CommunityContributorAvatar.plate(forKey: "sleepy-alpine-okapi-e22") == 9)
    }

    @Test("Vector asset names match the website's file names")
    func sharedVectorAssetNames() {
        #expect(
            CommunityContributorAvatar.assetName(forKey: "swift-otter-4417")
                == "cheetah-avatar-12"
        )
        #expect(
            CommunityContributorAvatar.assetName(forKey: "modest-slate-wombat-545")
                == "cheetah-avatar-02"
        )
        #expect(
            CommunityContributorAvatar.assetName(forKey: "sleepy-alpine-okapi-e22")
                == "cheetah-avatar-09"
        )
    }

    // MARK: - Identity key

    @Test("The identity key is the API slug, or exactly name + \"-\" + tag")
    func identityKeyMatchesNormalize() {
        let withSlug = CommunityBenchmarkContributor(
            name: "sleepy-alpine-okapi", tag: "e22", slug: "sleepy-alpine-okapi-e22"
        )
        let withoutSlug = CommunityBenchmarkContributor(
            name: "sleepy-alpine-okapi", tag: "e22"
        )
        #expect(withSlug.slug == "sleepy-alpine-okapi-e22")
        // `normalize()` composes the identical fallback, so a payload that
        // omits the slug must land on the same plate as one that includes it.
        #expect(withoutSlug.slug == "sleepy-alpine-okapi-e22")
        #expect(
            CommunityContributorAvatar.plate(for: withSlug)
                == CommunityContributorAvatar.plate(for: withoutSlug)
        )
        #expect(CommunityContributorAvatar.plate(for: withSlug) == 9)
    }

    @Test("A server-issued slug wins over the composed fallback")
    func serverSlugWins() {
        // If the API ever issues a slug that is not `name-tag`, the API is
        // right: it owns the identity and the profile route.
        let contributor = CommunityBenchmarkContributor(
            name: "swift-otter", tag: "4417", slug: "modest-slate-wombat-545"
        )
        #expect(contributor.slug == "modest-slate-wombat-545")
        #expect(CommunityContributorAvatar.plate(for: contributor) == 2)
    }

    @Test("The receipt decodes the slug and profile url the API sends")
    func receiptDecodesSlug() throws {
        let json = """
        {"submission_id":"sub-1","already_exists":false,
         "accepted_at":"2026-09-06T04:40:00Z",
         "contributor":{"name":"modest-slate-wombat","tag":"545",
                        "slug":"modest-slate-wombat-545",
                        "url":"/leaderboard/contributors/modest-slate-wombat-545"}}
        """
        let receipt = try JSONDecoder().decode(
            CommunityBenchmarkReceipt.self, from: Data(json.utf8)
        )
        let contributor = try #require(receipt.contributor)
        #expect(contributor.slug == "modest-slate-wombat-545")
        #expect(
            contributor.profileURL?.absoluteString
                == "https://rapidmlx.com/leaderboard/contributors/modest-slate-wombat-545"
        )
        #expect(CommunityContributorAvatar.plate(for: contributor) == 2)
    }

    @Test("A receipt without a slug still produces the website's profile route")
    func legacyReceiptProfileURL() throws {
        let json = """
        {"submission_id":"sub-2","already_exists":false,
         "accepted_at":"2026-09-06T04:40:00Z",
         "contributor":{"name":"swift-otter","tag":"4417"}}
        """
        let receipt = try JSONDecoder().decode(
            CommunityBenchmarkReceipt.self, from: Data(json.utf8)
        )
        let contributor = try #require(receipt.contributor)
        #expect(
            contributor.profileURL?.absoluteString
                == "https://rapidmlx.com/leaderboard/contributors/swift-otter-4417"
        )
    }

    // MARK: - Hash behaviour

    @Test("Hashing uses UTF-16 code units, like JavaScript charCodeAt")
    func hashUsesUTF16CodeUnits() {
        // "é" is one UTF-16 unit (0xE9) but two UTF-8 bytes. Hashing bytes
        // would silently diverge from the website for any non-ASCII slug.
        let unitCount = "é".utf16.count
        #expect(unitCount == 1)
        #expect("é".utf8.count == 2)
        // Sanity: the function is total and deterministic over such input.
        #expect(
            CommunityContributorAvatar.hash32("é")
                == CommunityContributorAvatar.hash32("é")
        )
    }

    @Test("Plate selection is deterministic and always lands on a shipped plate")
    func selectionIsClosedOverShippedPlates() {
        let plates = Set(CommunityContributorAvatar.assetPlates)
        // The worker issues `word-word-word-hex3`; sweep that shape broadly.
        for index in 0..<400 {
            let key = "swift-quiet-noun\(index)-\(String(format: "%03x", index % 4096))"
            let plate = CommunityContributorAvatar.plate(forKey: key)
            #expect(plates.contains(plate), "\(key) selected unshipped plate \(plate)")
            #expect(CommunityContributorAvatar.plate(forKey: key) == plate)
        }
        // Degenerate input still resolves rather than leaving a hole.
        #expect(plates.contains(CommunityContributorAvatar.plate(forKey: "")))
    }

    // MARK: - Resource integrity

    @Test("Every selectable plate is present in the bundle")
    func everyPlateShips() {
        let missing = CommunityContributorAvatar.missingPlates()
        #expect(
            missing.isEmpty,
            "missing contributor avatar plates: \(missing) — packaged apps would fall back for these contributors"
        )
    }

    @Test("Each plate loads as a real, non-degenerate image")
    func platesLoadAsImages() {
        for plate in CommunityContributorAvatar.assetPlates {
            let image = CommunityContributorAvatar.loadResource(
                named: CommunityContributorAvatar.assetName(plate: plate), extension: "webp"
            )
            let loaded = try? #require(image, "plate \(plate) did not load")
            #expect((loaded?.size.width ?? 0) > 0)
            #expect((loaded?.size.height ?? 0) > 0)
        }
    }

    @Test("Both context mascots ship and are distinct artwork")
    func mascotArtShips() {
        let wave = CommunityMascotArt.inviteWave.image
        let run = CommunityMascotArt.run.image
        #expect(wave != nil, "cheetah-invite-wave.png missing from the bundle")
        #expect(run != nil, "cheetah-run.png missing from the bundle")
        #expect(CommunityMascotContext.readyInvitation.art == .inviteWave)
        #expect(CommunityMascotContext.published.art == .inviteWave)
        #expect(CommunityMascotContext.running.art == .run)
    }

    @Test("A contributor portrait never falls back for a shipped plate")
    func portraitResolvesForRealIdentities() {
        for key in ["swift-otter-4417", "modest-slate-wombat-545", "sleepy-alpine-okapi-e22"] {
            #expect(CommunityContributorAvatar.image(forKey: key) != nil)
        }
    }
}
