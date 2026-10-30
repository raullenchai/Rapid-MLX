import Foundation
import Testing
@testable import Rapid

@Suite("Desktop funnel failure causes")
struct DesktopFunnelFailureCauseTests {
    @Test func downloadCauses() {
        let cases: [(String, FailureDiagnosis.Kind, DesktopFunnelReporter.DownloadFailureCause)] = [
            ("OSError: [Errno 28] No space left on device", .downloadFailed, .diskFull),
            ("sha256 mismatch for model.safetensors", .downloadFailed, .integrity),
            ("Error: Model 'x' not found on HuggingFace.", .downloadFailed, .notFound),
            ("404 Client Error", .downloadFailed, .notFound),
            ("Error: could not reach Hugging Face for 'x'.", .downloadFailed, .network),
            ("connection reset by peer", .downloadFailed, .network),
            ("status 503 from mirror", .downloadSourceUnavailable, .sourceUnavailable),
            ("boom", .downloadFailed, .other),
            ("", .downloadSourceUnavailable, .sourceUnavailable),
            ("No space left on device ... timed out", .downloadFailed, .diskFull),
        ]
        for (raw, kind, expected) in cases {
            #expect(FailureDiagnoser.downloadFailureCause(raw: raw, kind: kind) == expected)
        }
    }

    @Test func engineCauses() {
        let cases: [(ServerState, DesktopFunnelReporter.EngineStartFailureCause)] = [
            (.crashed(alias: "x", message: "The model ran out of memory and was stopped. ..."), .outOfMemory),
            (.crashed(alias: "x", message: "Couldn't start the model. Restart Rapid-MLX and try again."), .modelLoadFailed),
            (.crashed(alias: "x", message: "The model stopped unexpectedly."), .engineNotRunning),
            (.missing, .other), (.idle, .other),
        ]
        for (state, expected) in cases {
            #expect(FailureDiagnoser.funnelEngineStartCause(for: state) == expected)
        }
    }

    @Test @MainActor func downloadJobCauses() {
        let manager = DownloadManager()
        let disk = manager._testingSeedJob(alias: "disk", source: .huggingFace)
        manager._testingIngestStderr(alias: "disk", line: "OSError: [Errno 28] No space left on device")
        manager._testingFinish(alias: "disk", status: 1, reason: .exit)
        #expect(disk.failureCause == .diskFull)
        let signal = manager._testingSeedJob(alias: "signal", source: .mirror)
        manager._testingFinish(alias: "signal", status: 9, reason: .uncaughtSignal)
        #expect(signal.failureCause == .other)
        let cancelled = manager._testingSeedJob(alias: "cancelled", source: .mirror)
        manager._testingFinish(alias: "cancelled", status: 1, reason: .exit, wasCancelling: true)
        #expect(cancelled.status == .cancelled)
        #expect(cancelled.failureCause == nil)
    }
}
