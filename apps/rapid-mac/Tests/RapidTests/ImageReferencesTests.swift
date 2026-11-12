import Foundation
import Testing
@testable import Rapid

@Suite("Image reference attachments")
struct ImageReferencesTests {
    @MainActor private final class ControlledImport {
        private var continuation: CheckedContinuation<Data, Never>?
        private var started: CheckedContinuation<Void, Never>?

        func load() async -> Data {
            await withCheckedContinuation {
                continuation = $0
                started?.resume()
                started = nil
            }
        }

        func waitUntilStarted() async {
            if continuation != nil { return }
            await withCheckedContinuation { started = $0 }
        }

        func finish() {
            continuation?.resume(returning: Data([2]))
            continuation = nil
        }
    }

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

    @MainActor private func editingModel() -> ImageGenViewModel {
        let model = ImageGenViewModel(server: ServerManager())
        var entry = ModelEntry(
            alias: "flux3-klein", hfRepo: "black-forest-labs/flux-3-image-klein", sizeOnDisk: nil,
            cached: true, kind: .image, imageCapability: .generationAndEditing
        )
        entry.maxReferenceImages = 3
        model.imageModels = [entry]
        model.selectedAlias = entry.alias
        model.beginEdit(GeneratedImage(pngData: Data([1]), prompt: "source", isEdit: false))
        model.prompt = "Combine the references"
        return model
    }

    @Test @MainActor func submissionWaitsForEntireImportBatch() async throws {
        let model = editingModel()
        #expect(model.canSubmit)
        var imported = 0
        let task = try #require(model.importEditImages(
            at: [URL(fileURLWithPath: "/first.png"), URL(fileURLWithPath: "/second.png")],
            addingReference: true
        ) { _ in
            #expect(!model.canSubmit)
            #expect(model.additionalReferences.count == imported)
            imported += 1
            return Data([2])
        })
        // Reservation happens before the task gets a chance to run.
        #expect(model.pendingReferenceImports == 2)
        #expect(!model.canSubmit)
        // Pending imports count toward the model's reference limit.
        #expect(model.importEditImages(at: [URL(fileURLWithPath: "/excess.png")], addingReference: true) == nil)
        model.addReference(GeneratedImage(pngData: Data([3]), prompt: "excess", isEdit: false))
        #expect(model.additionalReferences.isEmpty)
        await task.value
        #expect(imported == 2)
        #expect(model.additionalReferences.count == 2)
        #expect(model.pendingReferenceImports == 0)
        #expect(model.canSubmit)
    }

    @Test(arguments: ["cancel", "begin", "replace", "clear"])
    @MainActor func staleReferenceImportIsDropped(change: String) async throws {
        let model = editingModel()
        let replacement = GeneratedImage(pngData: Data([3]), prompt: "replacement", isEdit: false)
        let loader = ControlledImport()
        var imported = 0
        let task = try #require(model.importEditImages(
            at: [URL(fileURLWithPath: "/first.png"), URL(fileURLWithPath: "/second.png")],
            addingReference: true
        ) { _ in
            imported += 1
            return await loader.load()
        })
        await loader.waitUntilStarted()
        #expect(!model.canSubmit)
        switch change {
        case "cancel": model.cancelEdit()
        case "begin": model.beginEdit(replacement)
        case "replace": model.editSource = replacement
        default: model.clearReferences()
        }
        loader.finish()
        await task.value
        #expect(imported == 1)
        #expect(model.additionalReferences.isEmpty)
        #expect(model.pendingReferenceImports == 0)
    }

    @Test @MainActor func importFailureReleasesReservation() async throws {
        let model = editingModel()
        let task = try #require(model.importEditImages(
            at: [URL(fileURLWithPath: "/unreadable.png")], addingReference: true
        ) { _ in
            throw ImportedEditImageError.cannotDecode
        })
        #expect(!model.canSubmit)
        await task.value
        #expect(model.pendingReferenceImports == 0)
        #expect(model.additionalReferences.isEmpty)
        #expect(model.errorMessage != nil)
        #expect(model.canSubmit)
    }

    @Test @MainActor func staleSourceImportAndErrorAreDropped() async throws {
        let model = editingModel()
        let sourceID = model.editSource?.id
        let task = try #require(model.importEditImages(at: [URL(fileURLWithPath: "/source.png")]) { _ in
            model.clearReferences()
            return Data([2])
        })
        await task.value
        #expect(model.editSource?.id == sourceID)
        #expect(model.pendingReferenceImports == 0)

        let failedTask = try #require(model.importEditImages(
            at: [URL(fileURLWithPath: "/unreadable.png")], addingReference: true
        ) { _ in
            model.cancelEdit()
            throw ImportedEditImageError.cannotDecode
        })
        await failedTask.value
        #expect(model.errorMessage == nil)
        #expect(model.pendingReferenceImports == 0)
    }
}
