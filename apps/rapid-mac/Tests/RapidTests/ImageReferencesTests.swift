import Foundation
import Testing
@testable import Rapid

@Suite("Image reference attachments")
struct ImageReferencesTests {
    @Test func catalogLimitsAndOldServerFallback() {
        let limits = ModelCatalog.parseImageReferenceLimitsJSON("""
        {"image":[
          {"alias":"flux3-klein","max_reference_images":10},
          {"alias":"old-server"},
          {"alias":"invalid","max_reference_images":99},
          {"alias":"bool","max_reference_images":true}
        ]}
        """)
        #expect(limits["flux3-klein"] == 10)
        #expect(limits["old-server"] == 1)
        #expect(limits["invalid"] == 1)
        #expect(limits["bool"] == 1)
    }

    @Test func multipartKeepsSingleImageBytesAndRepeatsFileFields() {
        let single = ImageClient.multipartBody(
            boundary: "test", fields: [("prompt", "edit")], fileField: "image",
            fileName: "input.png", fileMime: "image/png", fileData: Data("first".utf8)
        )
        let multiple = ImageClient.multipartBody(
            boundary: "test", fields: [("prompt", "edit")], fileField: "image",
            fileName: "input.png", fileMime: "image/png", fileData: Data("first".utf8),
            additionalFiles: [Data("second".utf8), Data("third".utf8)]
        )
        let text = String(decoding: multiple, as: UTF8.self)
        #expect(text.components(separatedBy: "filename=\"input.png\"").count == 4)
        #expect(text.contains("second\r\n"))
        #expect(text.contains("third\r\n"))
        let original = "--test\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nedit\r\n"
            + "--test\r\nContent-Disposition: form-data; name=\"image\"; filename=\"input.png\"\r\n"
            + "Content-Type: image/png\r\n\r\nfirst\r\n--test--\r\n"
        #expect(single == Data(original.utf8))
    }

    @Test @MainActor func attachmentLimitAndCancellation() {
        let model = ImageGenViewModel(server: ServerManager())
        var entry = ModelEntry(alias: "flux3-klein", hfRepo: "black-forest-labs/flux-3-image-klein", sizeOnDisk: nil, cached: true)
        entry.maxReferenceImages = 3
        model.imageModels = [entry]
        model.selectedAlias = entry.alias
        model.editSource = GeneratedImage(pngData: Data([1]), prompt: "source", isEdit: false)
        let reference = GeneratedImage(pngData: Data([2]), prompt: "reference", isEdit: false)
        for _ in 0..<5 { model.addReference(reference) }
        #expect(model.additionalReferences.count == 2)
        model.selectedAlias = "unknown"
        #expect(model.maxReferenceImages == 1)
        model.cancelEdit()
        #expect(model.additionalReferences.isEmpty)
        #expect(!model.isEditing)
    }
}
