import Foundation
import Testing

@Suite("Tagged release updater manifest workflow")
struct ReleaseManifestWorkflowTests {
    private static let workflow: String = {
        let here = URL(fileURLWithPath: #filePath)
        let repo = here
            .deletingLastPathComponent() // RapidTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // rapid-mac
            .deletingLastPathComponent() // apps
            .deletingLastPathComponent() // repo root
        return try! String(
            contentsOf: repo.appendingPathComponent(".github/workflows/rapid-mac-release.yml"),
            encoding: .utf8
        )
    }()

    @Test("missing distribution config fails instead of silently skipping")
    func missingConfigFailsClosed() throws {
        #expect(Self.workflow.contains("tagged releases require updater fallback publishing"))
        #expect(!Self.workflow.contains("skipping the optional CDN mirror"))
        for requiredSetting in [
            "CLOUDFLARE_API_TOKEN",
            "CLOUDFLARE_ACCOUNT_ID",
            "RAPID_MAC_DIST_R2_BUCKET",
            "RAPID_MAC_DIST_CDN_BASE",
        ] {
            #expect(Self.workflow.contains("missing+=(\(requiredSetting))"))
        }
        let branchStart = try #require(
            Self.workflow.range(of: "if (( ${#missing[@]} )); then")
        )
        let branchEnd = try #require(
            Self.workflow.range(
                of: "          fi",
                range: branchStart.upperBound..<Self.workflow.endIndex
            )
        )
        let branch = Self.workflow[branchStart.lowerBound..<branchEnd.upperBound]
        #expect(branch.contains("exit 1"))
    }

    @Test("manifest describes the bundled DMG and is committed last")
    func manifestIsPublishedLast() throws {
        let workflow = Self.workflow
        #expect(workflow.contains("{schema_version: 1, version: $version"))
        #expect(workflow.contains("dmg_sha256: $dmg_sha256, dmg_size: $dmg_size"))
        #expect(!workflow.contains("sidecar_url:"))

        let dmgUpload = try #require(workflow.range(of: "${R2_BUCKET}/${VERSIONED_KEY}"))
        let manifestUpload = try #require(workflow.range(of: "${R2_BUCKET}/latest.json"))
        #expect(dmgUpload.lowerBound < manifestUpload.lowerBound)
        let afterManifest = workflow[manifestUpload.upperBound..<workflow.endIndex]
        #expect(!afterManifest.contains("r2 object put"))
        #expect(workflow.contains(#"--cache-control "no-cache, must-revalidate""#))
    }
}
