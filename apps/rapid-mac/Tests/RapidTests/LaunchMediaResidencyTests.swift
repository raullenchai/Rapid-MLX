import Testing
@testable import Rapid

@MainActor
@Suite("Launch ignores non-chat residency")
struct LaunchMediaResidencyTests {
    private let entries = [
        ModelEntry(alias: "chat-model", hfRepo: nil, sizeOnDisk: nil, cached: true),
        ModelEntry(
            alias: "image-model", hfRepo: nil, sizeOnDisk: nil, cached: true,
            kind: .image
        ),
        ModelEntry(
            alias: "audio-model", hfRepo: nil, sizeOnDisk: nil, cached: true,
            kind: .audio
        ),
        ModelEntry(
            alias: "video-model", hfRepo: nil, sizeOnDisk: nil, cached: true,
            kind: .video
        ),
    ]
    private let mediaAliases: Set<String> = ["image-model", "audio-model", "video-model"]

    @Test("Only chat residents replace the Chat model selection")
    func knownMediaAliasesDoNotSync() {
        #expect(ContentView.shouldSyncChatAlias(
            serving: "chat-model", catalogEntries: entries,
            knownMediaAliases: mediaAliases, section: .chat
        ))
        #expect(!ContentView.shouldSyncChatAlias(
            serving: "image-model", catalogEntries: entries,
            knownMediaAliases: mediaAliases, section: .chat
        ))
        #expect(!ContentView.shouldSyncChatAlias(
            serving: "audio-model", catalogEntries: entries,
            knownMediaAliases: mediaAliases, section: .chat
        ))
        #expect(!ContentView.shouldSyncChatAlias(
            serving: "video-model", catalogEntries: entries,
            knownMediaAliases: mediaAliases, section: .chat
        ))
    }

    @Test("Custom aliases remain eligible as text models")
    func unknownAliasStillSyncs() {
        #expect(ContentView.shouldSyncChatAlias(
            serving: "org/custom-text-model", catalogEntries: entries,
            knownMediaAliases: mediaAliases, section: .chat
        ))
    }

    @Test("Server transitions outside Chat never replace its selection")
    func nonChatSectionsNeverSync() {
        for section in [SidebarSection.audio, .images, .launch] {
            #expect(!ContentView.shouldSyncChatAlias(
                serving: "org/unknown-media-model",
                catalogEntries: entries,
                knownMediaAliases: mediaAliases,
                section: section
            ))
        }
    }

    @Test("A media picker can classify an alias missing from the shared catalog snapshot")
    func mediaCatalogWinsUnknownFallback() {
        #expect(!ContentView.shouldSyncChatAlias(
            serving: "late-audio-model",
            catalogEntries: entries,
            knownMediaAliases: ["late-audio-model"],
            section: .chat
        ))
    }
}
