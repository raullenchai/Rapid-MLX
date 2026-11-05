import Foundation
import SwiftUI
import Testing
@testable import Rapid

@Suite("Hugging Face model references and checks")
struct HuggingFaceModelTests {
    @Test(arguments: [
        "https://huggingface.co/org/name", "https://hf.co/org/name/tree/main?download=1#files",
        "http://hf-mirror.com/org/name/blob/revision/config.json",
        "www.huggingface.co/org/name/resolve/main/config.json", "hf.co/org/name",
        "  <'https://huggingface.co/org/name/tree/other%2Fbranch'>  ", "HTTPS://HF.CO/org/name",
    ])
    func browserReferences(_ text: String) {
        #expect(HuggingFaceReference.detect(text) == .model(repo: "org/name", isURL: true))
    }

    @Test(arguments: ["https://hf.co/datasets/org/name", "https://hf.co/spaces/org/name",
                      "https://hf.co/collections/org/list", "https://hf.co/models", "https://hf.co/org",
                      "https://hf.co/org/%6eame", "https://hf.co//name", "https://hf.co/org/bad%20name"])
    func invalidBrowserReferences(_ text: String) {
        #expect(HuggingFaceReference.detect(text) == .invalidURL)
        #expect(HuggingFaceReference.clipboardPrefill(text) == nil)
    }

    @Test(arguments: ["qwen3", "find a model", "https://example.com/org/name", "https://hf.co.evil/org/name",
                      "/tmp/model", "org/name/tree/main", "datasets/name", "spaces/name", "./model"])
    func ordinarySearch(_ text: String) {
        #expect(HuggingFaceReference.detect(text) == nil)
    }

    @Test func bareRepoAndClipboardGate() {
        #expect(HuggingFaceReference.detect("org/name") == .model(repo: "org/name", isURL: false))
        #expect(HuggingFaceReference.clipboardPrefill("org/name") == nil)
        #expect(HuggingFaceReference.clipboardPrefill(nil) == nil)
        #expect(HuggingFaceReference.clipboardPrefill(" https://hf.co/org/name ") == "https://hf.co/org/name")
    }

    static func json(_ verdict: HuggingFaceModelCheck.Verdict, inCatalog: Bool = false, repo: String? = "org/name") throws -> Data {
        try JSONSerialization.data(withJSONObject: [
            "schema": 1, "reference": "https://hf.co/org/name", "repo": repo as Any? ?? NSNull(),
            "alias": inCatalog ? "catalog-model" : NSNull(), "in_catalog": inCatalog,
            "verdict": verdict.rawValue, "reason": NSNull(), "message": "CLI sentence.",
            "download_bytes": 2_147_483_648, "format": "mlx", "capabilities": ["text_generation"],
            "required_memory_bytes": 34_359_738_368, "fits": false, "cached": false,
        ])
    }

    @Test(arguments: HuggingFaceModelCheck.Verdict.allCases)
    func decodesAndMapsEveryVerdict(_ verdict: HuggingFaceModelCheck.Verdict) throws {
        let result = try JSONDecoder().decode(HuggingFaceModelCheck.self, from: Self.json(verdict))
        #expect(result.verdict == verdict)
        #expect(result.message == "CLI sentence.")
        #expect(result.entry?.alias == "org/name")
        #expect(result.downloadBytes == 2_147_483_648)
        let expected: HuggingFaceModelCheck.RowState = switch verdict {
        case .ok: .download
        case .tooBig: .downloadAnyway
        default: .warning
        }
        #expect(result.rowState == expected)
        #expect(result.communityMeta == "org · mlx · text_generation · community upload, not tested")
        #expect(result.memoryWarning(physicalRAMGB: 16) == "Needs about 32 GB memory; this Mac has 16 GB.")
        let catalog = try JSONDecoder().decode(HuggingFaceModelCheck.self, from: Self.json(verdict, inCatalog: true))
        #expect(catalog.rowState == .catalog)
    }

    @Test func nullableFieldsAndMissingRepoFailClosed() throws {
        let data = Data(#"{"schema":1,"reference":"bad","repo":null,"alias":null,"in_catalog":false,"verdict":"invalid_reference","reason":null,"message":null,"download_bytes":null,"format":null,"capabilities":[],"required_memory_bytes":null,"fits":null,"cached":false}"#.utf8)
        let result = try JSONDecoder().decode(HuggingFaceModelCheck.self, from: data)
        #expect(result.repo == nil)
        #expect(result.rowState == .warning)
        #expect(result.entry == nil)
        let missingRepo = try JSONDecoder().decode(HuggingFaceModelCheck.self, from: Self.json(.ok, repo: nil))
        #expect(missingRepo.rowState == .warning)
    }

    @MainActor @Test func groupingAndCatalogDeduplication() {
        let builtin = ModelEntry(alias: "catalog-model", hfRepo: "org/known", sizeOnDisk: nil, cached: true, isBuiltinProfile: true)
        let hf = ModelEntry(alias: "org/name", hfRepo: "org/name", sizeOnDisk: nil, cached: true)
        let custom = ModelEntry(alias: "custom", hfRepo: "org/custom", sizeOnDisk: nil, cached: true, isBuiltinProfile: false)
        let external = ModelEntry(alias: "external", hfRepo: "org/external", sizeOnDisk: nil, cached: true, isExternal: true)
        let entries = [builtin, hf, custom, external, hf]
        #expect(HuggingFaceReference.catalogMatch("https://hf.co/org/known/tree/main", in: entries) == builtin)
        #expect(HuggingFaceReference.catalogMatch("org/known", in: entries) == builtin)
        #expect(HuggingFaceReference.catalogMatch("catalog-model", in: entries) == builtin)
        #expect(HuggingFaceReference.catalogMatch("org/name", in: entries) == hf)
        #expect(ModelPickerBar.huggingFaceSectionEntries(entries, excluding: ["custom"]) == [hf])
        #expect(!external.isFromHuggingFace)
        #expect(!builtin.isFromHuggingFace)
    }

    @MainActor @Test func routesBeforeOpeningAndPreservesEmptyPrefill() {
        let router = SettingsRouter()
        var opened = 0
        router.addFromHuggingFace(clipboard: "https://hf.co/org/name") {
            #expect(router.requestedCategory == .modelManagement)
            #expect(router.modelSearchRequest?.prefill == "https://hf.co/org/name")
            opened += 1
        }
        let previous = router.modelSearchRequest?.id
        router.addFromHuggingFace(clipboard: "ordinary clipboard") {
            #expect(router.modelSearchRequest?.prefill == nil)
            #expect(router.modelSearchRequest?.id != previous)
            opened += 1
        }
        #expect(opened == 2)
    }

    @MainActor @Test func panelPrefillChecksAndDownloadsNormalizedRepository() async throws {
        let file = URL(fileURLWithPath: #filePath)
        let desktop = file.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let fake = desktop.appendingPathComponent("scripts/fake-rapid-mlx.sh")
        let server = ServerManager(testingState: .stopped, binaryPath: fake)
        let downloads = DownloadManager(binaryPath: fake, binaryLocator: { fake })
        let router = SettingsRouter()
        router.addFromHuggingFace(clipboard: "https://hf.co/org/new-model/tree/main") {}
        let stage = GoldenStage(
            ScrollView {
                SettingsModelManagementPanel()
                    .environment(server).environment(downloads).environment(router)
            }, size: CGSize(width: 1100, height: 1600)
        )
        try await stage.waitForIdentifier("Settings.ModelManagement.HFRow.Download")
        #expect(stage.value(of: "Settings.ModelManagement.Search") == "https://hf.co/org/new-model/tree/main")
        #expect(router.modelSearchRequest == nil)
        #expect(stage.treeText().contains("community upload, not tested"))
        try stage.press("Settings.ModelManagement.HFRow.Download")
        #expect(downloads.jobs["org/new-model"] != nil)
        #expect(downloads.jobs["https://hf.co/org/new-model/tree/main"] == nil)
        downloads.cancelDownload(alias: "org/new-model")
    }

    @Test func checkUsesRealCatalogRunnerAndFakeContract() async throws {
        let file = URL(fileURLWithPath: #filePath)
        let desktop = file.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let fake = desktop.appendingPathComponent("scripts/fake-rapid-mlx.sh")
        let result = try await ModelCatalog.checkHuggingFace(reference: "https://hf.co/org/name/tree/main", binary: fake)
        #expect(result.repo == "org/name")
        #expect(result.rowState == .download)
        let invalid = try await ModelCatalog.checkHuggingFace(reference: "https://hf.co/datasets/org/name", binary: fake)
        #expect(invalid.verdict == .invalidReference)
    }
}
