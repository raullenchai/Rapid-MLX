import Testing
@testable import Rapid

@Suite("App termination lifecycle")
struct AppTerminationTests {
    @Test("Dictation is disarmed before stream and child teardown")
    @MainActor
    func dictationStopsFirst() {
        var events: [String] = []

        AppDelegate.runTerminationSequence(
            stopDictation: { events.append("dictation") },
            stopStream: { events.append("stream") },
            signalServer: { events.append("signal-server") },
            signalDownloads: { events.append("signal-downloads") },
            reapServer: { events.append("reap-server") },
            reapDownloads: { events.append("reap-downloads") },
            flushConversations: { events.append("flush-conversations") },
            flushFolders: { events.append("flush-folders") }
        )

        #expect(events == [
            "dictation",
            "stream",
            "signal-server",
            "signal-downloads",
            "reap-server",
            "reap-downloads",
            "flush-conversations",
            "flush-folders",
        ])
    }
}
