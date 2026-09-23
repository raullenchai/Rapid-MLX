import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Actionable sidecar startup failures")
struct SidecarStartupFailureTests {
    private let videoMarker =
        "RAPID-MLX-STARTUP-FAILURE: runtime_extra_missing extra=video\n"

    @Test("stderr marker becomes a specific message and Startup Log action")
    func markerMapsToPresentation() {
        let capture = SidecarStartupFailureCapture()
        capture.ingest(Data(videoMarker.utf8), source: .sidecarStderr)

        let failure = capture.failure
        #expect(failure?.reason == .runtimeExtraMissing)
        #expect(failure?.extra == .video)
        #expect(failure?.message ==
            "The installed engine doesn't include Video support. Open Startup Log for installation details.")
        #expect(failure?.action == .openStartupLog)

        let readiness = ModelReadiness.resolve(
            serverState: .crashed(alias: "wan2.2-ti2v-5b-q8", message: failure?.message ?? ""),
            alias: "wan2.2-ti2v-5b-q8",
            cacheState: .onDisk,
            startupFailure: failure
        )
        #expect(readiness.action == .openStartupLog)
        #expect(readiness.detail == failure?.message)
    }

    @Test("marker absent keeps the existing generic exit message")
    func absentMarkerIsGeneric() {
        let manager = ServerManager(testingState: .starting(alias: "plain-model"))
        manager._testSimulateChildExit(
            expectedStop: false,
            status: 2,
            reason: .exit
        )

        #expect(manager.startupFailure == nil)
        #expect(manager.state == .crashed(
            alias: "plain-model",
            message: "The model stopped unexpectedly."
        ))
    }

    @Test("malformed and partial markers are ignored without crashing")
    func malformedMarkerIsGeneric() {
        let capture = SidecarStartupFailureCapture()
        capture.ingest(
            Data("RAPID-MLX-STARTUP-FAILURE: runtime_extra_missing extra=video".utf8),
            source: .sidecarStderr
        )
        #expect(capture.failure == nil)

        let malformed = SidecarStartupFailureCapture()
        malformed.ingest(
            Data("\nRAPID-MLX-STARTUP-FAILURE: unknown extra=video\n".utf8),
            source: .sidecarStderr
        )

        #expect(malformed.failure == nil)
    }

    @Test("a marker split across stderr chunks is accepted")
    func splitMarkerIsAccepted() {
        let capture = SidecarStartupFailureCapture()
        let split = videoMarker.index(videoMarker.startIndex, offsetBy: 31)

        capture.ingest(Data(videoMarker[..<split].utf8), source: .sidecarStderr)
        #expect(capture.failure == nil)
        capture.ingest(Data(videoMarker[split...].utf8), source: .sidecarStderr)

        #expect(capture.failure == SidecarStartupFailure(
            reason: .runtimeExtraMissing,
            extra: .video
        ))
    }

    @Test("an overlong line cannot resynchronize at a marker substring")
    func overlongLineDoesNotResynchronize() {
        let capture = SidecarStartupFailureCapture()

        capture.ingest(Data(String(repeating: "A", count: 513).utf8), source: .sidecarStderr)
        capture.ingest(Data(videoMarker.utf8), source: .sidecarStderr)

        #expect(capture.failure == nil)

        capture.ingest(Data(videoMarker.utf8), source: .sidecarStderr)
        #expect(capture.failure != nil)
    }

    @Test("health success wins over a marker observed before exit handling")
    func healthSuccessWinsCompletionOrder() {
        let capture = SidecarStartupFailureCapture()
        let manager = ServerManager(
            testingState: .starting(alias: "wan2.2-ti2v-5b-q8")
        )

        #expect(capture.recordHealthResponse(statusCode: 200))
        capture.ingest(Data(videoMarker.utf8), source: .sidecarStderr)
        let snapshot = capture.snapshotAtTermination()
        manager._testSimulateChildExit(
            expectedStop: false,
            status: 2,
            reason: .exit,
            startupFailure: snapshot.failure,
            readyObserved: snapshot.readyObserved
        )

        #expect(manager.startupFailure == nil)
        #expect(manager.state == .crashed(
            alias: "wan2.2-ti2v-5b-q8",
            message: "The model stopped unexpectedly."
        ))
    }

    @Test("model output and chat messages cannot spoof a startup failure")
    func untrustedSourcesAreRejected() {
        let capture = SidecarStartupFailureCapture()
        for source in [
            SidecarStartupFailureCapture.Source.sidecarStdout,
            .modelOutput,
            .chatMessage,
        ] {
            capture.ingest(Data(videoMarker.utf8), source: source)
        }

        #expect(capture.failure == nil)
    }

    @Test("a marker after readiness is not a startup failure")
    func markerAfterReadinessIsRejected() {
        let capture = SidecarStartupFailureCapture()
        #expect(capture.recordHealthResponse(statusCode: 204))
        capture.ingest(Data(videoMarker.utf8), source: .sidecarStderr)

        #expect(capture.failure == nil)
    }

    @Test("multiple markers produce one first deterministic failure")
    func multipleMarkersProduceOneMessage() {
        let capture = SidecarStartupFailureCapture()
        let stderr = videoMarker
            + "RAPID-MLX-STARTUP-FAILURE: runtime_broken extra=vision\n"
        capture.ingest(Data(stderr.utf8), source: .sidecarStderr)

        #expect(capture.failure == SidecarStartupFailure(
            reason: .runtimeExtraMissing,
            extra: .video
        ))
    }

    @Test("captured marker replaces generic child-exit copy")
    func markerFlowsIntoCrashState() {
        let capture = SidecarStartupFailureCapture()
        capture.ingest(Data(videoMarker.utf8), source: .sidecarStderr)
        let manager = ServerManager(
            testingState: .starting(alias: "wan2.2-ti2v-5b-q8")
        )

        manager._testSimulateChildExit(
            expectedStop: false,
            status: 2,
            reason: .exit,
            startupFailure: capture.failure
        )

        #expect(manager.startupFailure == capture.failure)
        #expect(manager.state == .crashed(
            alias: "wan2.2-ti2v-5b-q8",
            message: capture.failure?.message ?? ""
        ))
    }

    @Test("failed chat turn uses the structured startup message")
    func chatTurnUsesStructuredMessage() {
        let capture = SidecarStartupFailureCapture()
        capture.ingest(Data(videoMarker.utf8), source: .sidecarStderr)
        let manager = ServerManager(
            testingState: .starting(alias: "wan2.2-ti2v-5b-q8")
        )
        manager._testSimulateChildExit(
            expectedStop: false,
            status: 2,
            reason: .exit,
            startupFailure: capture.failure
        )
        let chat = ChatViewModel(server: manager, persistsConversations: false)
        chat.devSeedMessages([ChatMessage(role: .assistant, status: .streaming)])

        chat.finishWithStartupFailure(
            placeholderIndex: 0,
            alias: "wan2.2-ti2v-5b-q8"
        )

        #expect(chat.messages[0].content == capture.failure?.message)
        #expect(chat.lastError == capture.failure?.message)
    }

    @Test("failed chat turn keeps its generic copy without a marker")
    func chatTurnWithoutMarkerIsGeneric() {
        let manager = ServerManager(
            testingState: .crashed(
                alias: "plain-model",
                message: "The model stopped unexpectedly."
            )
        )
        let chat = ChatViewModel(server: manager, persistsConversations: false)
        chat.devSeedMessages([ChatMessage(role: .assistant, status: .streaming)])

        chat.finishWithStartupFailure(placeholderIndex: 0, alias: "plain-model")

        let generic = "Couldn't start plain-model. Try again, or pick a different model in the box below."
        #expect(chat.messages[0].content == generic)
        #expect(chat.lastError == generic)
    }
}
