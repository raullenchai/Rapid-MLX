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

    private static let mirrorJob: Substring = {
        let start = workflow.range(of: "  mirror-dist:")!.lowerBound
        return workflow[start...]
    }()

    @Test("missing distribution config fails instead of silently skipping")
    func missingConfigFailsClosed() throws {
        let job = Self.mirrorJob
        #expect(job.contains("if: startsWith(github.ref, 'refs/tags/')"))
        #expect(job.contains("tagged releases require updater fallback publishing"))
        #expect(!job.contains("skipping the optional CDN mirror"))
        for requiredSetting in [
            "CLOUDFLARE_API_TOKEN",
            "CLOUDFLARE_ACCOUNT_ID",
            "RAPID_MAC_DIST_R2_BUCKET",
            "RAPID_MAC_DIST_CDN_BASE",
        ] {
            #expect(job.contains("missing+=(\(requiredSetting))"))
        }
        let branchStart = try #require(
            job.range(of: "if (( ${#missing[@]} )); then")
        )
        let branchEnd = try #require(
            job.range(
                of: "          fi",
                range: branchStart.upperBound..<job.endIndex
            )
        )
        let branch = job[branchStart.lowerBound..<branchEnd.upperBound]
        #expect(branch.contains("exit 1"))
        let firstUpload = try #require(job.range(of: "r2 object put"))
        #expect(branchEnd.upperBound < firstUpload.lowerBound)
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
        #expect(workflow.contains("rapid-mlx-desktop-${DMG_SHA256}.dmg"))
    }
}
