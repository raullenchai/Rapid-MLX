import Testing
@testable import Rapid

@Suite("Dictation model picker download state")
struct DictationModelPickerDownloadTests {
    @Test func states() {
        #expect(TranscriptionDownloadControlState.resolve(cached: true, jobStatus: nil, fraction: nil) == .none)
        #expect(TranscriptionDownloadControlState.resolve(cached: false, jobStatus: nil, fraction: nil) == .download)
        #expect(TranscriptionDownloadControlState.resolve(cached: false, jobStatus: .running, fraction: 0.42) == .downloading(fraction: 0.42))
        #expect(TranscriptionDownloadControlState.resolve(cached: false, jobStatus: .running, fraction: nil) == .downloading(fraction: nil))
        #expect(TranscriptionDownloadControlState.resolve(cached: false, jobStatus: .failed(message: "error"), fraction: nil) == .retry)
        #expect(TranscriptionDownloadControlState.resolve(cached: false, jobStatus: .cancelled, fraction: nil) == .download)
    }
}
