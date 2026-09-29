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
            signalShareCompute: { events.append("signal-share-compute") },
            signalServer: { events.append("signal-server") },
            signalCUAServer: { events.append("signal-cua-server") },
            signalDownloads: { events.append("signal-downloads") },
            reapShareCompute: { events.append("reap-share-compute") },
            reapServer: { events.append("reap-server") },
            reapCUAServer: { events.append("reap-cua-server") },
            reapDownloads: { events.append("reap-downloads") },
            flushConversations: { events.append("flush-conversations") },
            flushFolders: { events.append("flush-folders") }
        )

        #expect(events == [
            "dictation",
            "stream",
            "signal-share-compute",
            "signal-server",
            "signal-cua-server",
            "signal-downloads",
            "reap-server",
            "reap-cua-server",
            "reap-share-compute",
            "reap-downloads",
            "flush-conversations",
            "flush-folders",
        ])
    }
}
