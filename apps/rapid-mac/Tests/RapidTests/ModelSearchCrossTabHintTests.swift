import Foundation
import Testing
@testable import Rapid

@Suite("Model search cross-tab hints")
struct ModelSearchCrossTabHintTests {
    private let audio = ModelEntry(alias: "asr-audio", hfRepo: "test/asr", sizeOnDisk: nil, cached: true, taskTypes: [.speechRecognition])
    private let image = ModelEntry(alias: "asr-image", hfRepo: "test/image", sizeOnDisk: nil, cached: false, taskTypes: [.imageGeneration])
    private let chat = ModelEntry(alias: "asr-chat", hfRepo: "test/chat", sizeOnDisk: nil, cached: true, taskTypes: [.textGeneration])

    @Test("Finds audio matches and excludes the current tab")
    func audioMatch() {
        let matches = ModelCacheActions.crossTabMatches(in: [audio, chat], kinds: [.chat, .audio], excluding: .chat, filterMode: .all, query: "asr")
        #expect(matches == [.init(kind: .audio, count: 1)])
        #expect(ModelCacheActions.crossTabMatches(in: [audio], kinds: [.audio], excluding: .audio, filterMode: .all, query: "asr").isEmpty)
    }

    @Test("Empty queries have no cross-tab hints")
    func emptyQuery() {
        for query in ["", "  \n "] {
            #expect(ModelCacheActions.crossTabMatches(in: [audio], kinds: [.audio], excluding: .chat, filterMode: .all, query: query).isEmpty)
        }
    }

    @Test("Preserves tab order and applies the filter mode")
    func orderAndFilter() {
        let catalog = [audio, image, chat]
        let kinds: [ModelKind] = [.image, .audio, .chat]
        #expect(ModelCacheActions.crossTabMatches(in: catalog, kinds: kinds, excluding: .chat, filterMode: .all, query: "asr") == [.init(kind: .image, count: 1), .init(kind: .audio, count: 1)])
        #expect(ModelCacheActions.crossTabMatches(in: catalog, kinds: kinds, excluding: .chat, filterMode: .cached, query: "asr") == [.init(kind: .audio, count: 1)])
    }
}
