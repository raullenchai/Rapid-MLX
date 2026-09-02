import Foundation
import Testing
@testable import Rapid

@Suite("Experimental Video gate", .serialized)
struct VideoFeatureGateTests {
    @Test("Video is opt-in when no preference exists")
    func defaultsOff() throws {
        let suite = "rapid.video-gate-tests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }

        #expect(!VideoFeatureConfig.isEnabled(in: defaults))
        defaults.set(true, forKey: VideoFeatureConfig.enabledKey)
        #expect(VideoFeatureConfig.isEnabled(in: defaults))
    }

    @MainActor
    @Test("Disabling while Video is active returns to Chat")
    func disablingRecoversNavigation() {
        #expect(ContentView.sectionAfterVideoGateChange(current: .video, enabled: false) == .chat)
        #expect(ContentView.sectionAfterVideoGateChange(current: .video, enabled: true) == .video)
        #expect(ContentView.sectionAfterVideoGateChange(current: .images, enabled: false) == .images)
    }

    @Test("Video purposes accept only explicit matching capabilities")
    func capabilityFiltering() {
        let chat = ModelEntry(alias: "chat", hfRepo: nil, sizeOnDisk: nil, cached: true)
        let textVideo = ModelEntry(
            alias: "t2v", hfRepo: nil, sizeOnDisk: nil, cached: true,
            kind: .video, videoCapabilities: [.textToVideo], minimumMemoryGB: 24
        )
        let imageVideo = ModelEntry(
            alias: "i2v", hfRepo: nil, sizeOnDisk: nil, cached: true,
            kind: .video, videoCapabilities: [.imageToVideo], minimumMemoryGB: 32
        )

        #expect(ModelSelectionPurpose.textToVideo.entries(in: [chat, textVideo, imageVideo]) == [textVideo])
        #expect(ModelSelectionPurpose.imageToVideo.entries(in: [chat, textVideo, imageVideo]) == [imageVideo])
    }
}
