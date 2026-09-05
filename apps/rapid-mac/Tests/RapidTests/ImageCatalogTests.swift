import Foundation
import Testing
@testable import Rapid

/// Coverage for the image-gen catalog split: `[image:gen]` rows must be
/// parsed for the Images tab AND excluded from the chat catalog, so an image
/// checkpoint can never surface in the chat picker (catalog-integrity, #1603).
@Suite("Image catalog")
struct ImageCatalogTests {
    /// A faithful slice of `rapid-mlx models` output: a chat row, the video
    /// section, and the image section the CLI now emits.
    static let sample = """
      Available models (2 aliases)
      ────────────────────────────
      Alias                 Size       Tools      HF id
      ────────────────────────────
      qwen3.6-27b-4bit      15.0 GiB   hermes     mlx-community/Qwen3.6-27B-4bit
      bonsai-1.7b-2bit      0.9 GiB    —          prism-ml/Bonsai

      Video models (1 aliases)
      ────────────────────────────
      Alias                 Size       Kind        HF id
      ────────────────────────────
      ltx-2.3-mlx-q4        24.0 GiB   [video:gen] notapalindrome/ltx23-mlx-av-q4

      Image models (5 aliases)
      ────────────────────────────
      Alias                 Size       Kind        HF id
      ────────────────────────────
      flux2-klein-4b        4.3 GiB    [image:both] Runpod/FLUX.2-klein-4B-mflux-4bit
      bonsai-image-4b-2bit   3.6 GiB    [image:gen] prism-ml/bonsai-image-ternary-4B-mlx-2bit
      z-image-turbo         5.5 GiB    [image:gen] filipstrand/Z-Image-Turbo-mflux-4bit
      hidream-o1-dev       16.4 GiB    [image:gen] mlx-community/HiDream-O1-Image-Dev-mlx-bf16
      sdxl-base             6.5 GiB     [image:gen] stabilityai/stable-diffusion-xl-base-1.0
    """

    @Test("parseImageRows extracts image rows and their operation")
    func parsesImageRows() {
        let rows = ModelCatalog.parseImageRows(Self.sample)
        #expect(rows.count == 5)

        let aliases = rows.map(\.alias)
        #expect(aliases.contains("flux2-klein-4b"))
        #expect(aliases.contains("bonsai-image-4b-2bit"))
        #expect(aliases.contains("z-image-turbo"))
        #expect(aliases.contains("hidream-o1-dev"))
        #expect(aliases.contains("sdxl-base"))
        // No chat / video alias leaks in.
        #expect(!aliases.contains("qwen3.6-27b-4bit"))
        #expect(!aliases.contains("ltx-2.3-mlx-q4"))

        let klein = rows.first { $0.alias == "flux2-klein-4b" }
        #expect(klein?.hfRepo == "Runpod/FLUX.2-klein-4B-mflux-4bit")
        #expect(klein?.size == "4.3 GiB")
        #expect(klein?.capability == .generationAndEditing)
        let hidream = rows.first { $0.alias == "hidream-o1-dev" }
        #expect(hidream?.hfRepo == "mlx-community/HiDream-O1-Image-Dev-mlx-bf16")
        #expect(hidream?.capability == .generation)
        let sdxl = rows.first { $0.alias == "sdxl-base" }
        #expect(sdxl?.hfRepo == "stabilityai/stable-diffusion-xl-base-1.0")
        #expect(sdxl?.size == "6.5 GiB")
        #expect(sdxl?.capability == .generation)
        let bonsai = rows.first { $0.alias == "bonsai-image-4b-2bit" }
        #expect(bonsai?.hfRepo == "prism-ml/bonsai-image-ternary-4B-mlx-2bit")
        #expect(bonsai?.size == "3.6 GiB")
        #expect(bonsai?.capability == .generation)
    }

    @Test("complete mflux caches are marked downloaded in Images")
    func mfluxCachesReachImagesAsDownloaded() {
        let rows = ModelCatalog.parseImageRows(Self.sample)
        let cached = ModelCatalog.mergeImageRows(
            rows,
            cachedRepos: [
                "Runpod/FLUX.2-klein-4B-mflux-4bit",
                "prism-ml/bonsai-image-ternary-4B-mlx-2bit",
                "filipstrand/Z-Image-Turbo-mflux-4bit",
                "mlx-community/HiDream-O1-Image-Dev-mlx-bf16",
                "stabilityai/stable-diffusion-xl-base-1.0",
            ]
        )

        let klein = cached.first { $0.alias == "flux2-klein-4b" }
        #expect(klein?.cached == true)
        #expect(klein?.imageCapability == .generationAndEditing)
        #expect(cached.first { $0.alias == "bonsai-image-4b-2bit" }?.cached == true)
        #expect(cached.first { $0.alias == "z-image-turbo" }?.cached == true)
        #expect(cached.first { $0.alias == "hidream-o1-dev" }?.cached == true)
        #expect(cached.first { $0.alias == "sdxl-base" }?.cached == true)
    }

    @Test("Image capability rows are excluded from the chat catalog")
    func imageRowsExcludedFromChat() {
        // hasNonChatKindTag now drops image alongside audio/video.
        #expect(ModelCatalog.hasNonChatKindTag(
            "flux2-klein-4b  4.3 GiB  [image:both] Runpod/FLUX.2-klein-4B-mflux-4bit"))
        #expect(ModelCatalog.hasNonChatKindTag(
            "ltx-2.3-mlx-q4  24.0 GiB  [video:gen] repo/ltx"))
        // A plain chat row is not dropped.
        #expect(!ModelCatalog.hasNonChatKindTag(
            "qwen3.6-27b-4bit  15.0 GiB  hermes  mlx-community/Qwen3.6-27B-4bit"))

        // The chat parser drops the image alias entirely.
        let excluded = ModelCatalog.parseExcludedAliases(Self.sample)
        #expect(excluded.contains("flux2-klein-4b"))
        #expect(excluded.contains("bonsai-image-4b-2bit"))
        #expect(excluded.contains("z-image-turbo"))
        #expect(excluded.contains("hidream-o1-dev"))
        #expect(excluded.contains("sdxl-base"))
    }
}
