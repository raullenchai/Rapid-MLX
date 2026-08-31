import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Task-scoped model selection")
struct ModelSelectionPurposeTests {
    private let chat = ModelEntry(
        alias: "qwen3.5-4b-4bit",
        hfRepo: nil,
        sizeOnDisk: nil,
        cached: true
    )
    private let imageGeneration = ModelEntry(
        alias: "z-image-turbo",
        hfRepo: nil,
        sizeOnDisk: nil,
        cached: true,
        kind: .image,
        imageCapability: .generation
    )
    private let imageEditing = ModelEntry(
        alias: "flux2-edit",
        hfRepo: nil,
        sizeOnDisk: nil,
        cached: true,
        kind: .image,
        imageCapability: .editing
    )
    private let imageBoth = ModelEntry(
        alias: "flux2-klein",
        hfRepo: nil,
        sizeOnDisk: nil,
        cached: true,
        kind: .image,
        imageCapability: .generationAndEditing
    )
    private let transcription = ModelEntry(
        alias: "qwen3-asr",
        hfRepo: nil,
        sizeOnDisk: nil,
        cached: true,
        kind: .audio,
        audioCapability: .transcription,
        audioFamily: "qwen3_asr"
    )
    private let alignment = ModelEntry(
        alias: "qwen3-aligner",
        hfRepo: nil,
        sizeOnDisk: nil,
        cached: true,
        kind: .audio,
        audioCapability: .alignment,
        audioFamily: "qwen3_aligner"
    )
    private let presetSpeech = ModelEntry(
        alias: "qwen3-tts-4bit",
        hfRepo: nil,
        sizeOnDisk: nil,
        cached: true,
        kind: .audio,
        audioCapability: .speech,
        audioFamily: "qwen3_tts"
    )
    private let unsupportedSpeech = ModelEntry(
        alias: "kokoro",
        hfRepo: nil,
        sizeOnDisk: nil,
        cached: true,
        kind: .audio,
        audioCapability: .speech,
        audioFamily: "kokoro"
    )

    private var catalog: [ModelEntry] {
        [
            chat,
            imageGeneration,
            imageEditing,
            imageBoth,
            transcription,
            alignment,
            presetSpeech,
            unsupportedSpeech,
        ]
    }

    @Test("Each picker receives only models accepted by its task")
    func purposeFilters() {
        #expect(
            ModelSelectionPurpose.chat.entries(in: catalog).map(\.alias) == [chat.alias]
        )
        #expect(ModelSelectionPurpose.imageGeneration.entries(in: catalog).map(\.alias) == [
            imageGeneration.alias,
            imageBoth.alias,
        ])
        #expect(ModelSelectionPurpose.imageEditing.entries(in: catalog).map(\.alias) == [
            imageEditing.alias,
            imageBoth.alias,
        ])
        #expect(ModelSelectionPurpose.speechToText.entries(in: catalog).map(\.alias) == [
            transcription.alias,
        ])
        #expect(ModelSelectionPurpose.textToSpeech.entries(in: catalog).map(\.alias) == [
            presetSpeech.alias,
        ])
    }

    @Test("Mounted Image, STT, and TTS view models use the shared task policy")
    func viewModelsUsePurposePolicy() {
        let server = ServerManager(testingState: .idle)
        let images = ImageGenViewModel(server: server)
        images.imageModels = catalog
        #expect(images.generationModels.map(\.alias) == [imageGeneration.alias, imageBoth.alias])
        #expect(images.editModels.map(\.alias) == [imageEditing.alias, imageBoth.alias])

        let audio = AudioViewModel(server: server)
        audio.audioModels = catalog
        #expect(audio.transcriptionModels.map(\.alias) == [transcription.alias])
        #expect(audio.speechModels.map(\.alias) == [presetSpeech.alias])
    }

    @Test("Late media classification replaces ASR in Chat but preserves custom text aliases")
    func chatSelectionNormalization() {
        #expect(ModelPickerBar.normalizedChatSelection(
            currentAlias: "",
            catalog: [chat],
            knownNonChatAliases: [],
            fallbackAlias: chat.alias
        ) == chat.alias)
        #expect(ModelPickerBar.normalizedChatSelection(
            currentAlias: "QWEN3-ASR",
            catalog: [chat],
            knownNonChatAliases: ["qwen3-asr"],
            fallbackAlias: chat.alias
        ) == chat.alias)
        #expect(ModelPickerBar.normalizedChatSelection(
            currentAlias: "org/custom-text-model",
            catalog: [chat],
            knownNonChatAliases: ["qwen3-asr"],
            fallbackAlias: chat.alias
        ) == "org/custom-text-model")
    }

    @Test("An authoritative Chat row wins over supplemental media identity")
    func chatCatalogWinsCollision() {
        #expect(ModelPickerBar.normalizedChatSelection(
            currentAlias: chat.alias,
            catalog: [chat],
            knownNonChatAliases: [chat.alias],
            fallbackAlias: nil
        ) == chat.alias)
    }

    @Test("Dictation publishes catalog truth, not its persisted selection")
    func dictationKnownAudioAliases() async {
        let staleSelection = "org/custom-text-model"
        let catalogEntry = transcription
        let server = ServerManager(
            testingState: .idle,
            binaryPath: URL(fileURLWithPath: "/usr/bin/true")
        )
        let dictation = DictationController(
            server: server,
            testingEnabled: false,
            testingModelAlias: staleSelection,
            audioCatalogLoader: { _ in [catalogEntry] }
        )

        #expect(!dictation.knownAudioAliases.contains(staleSelection))
        await dictation.refreshModelCacheState()
        #expect(dictation.knownAudioAliases == [catalogEntry.alias])
    }
}
