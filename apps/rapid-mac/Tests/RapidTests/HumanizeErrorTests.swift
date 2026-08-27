import Foundation
import Testing
@testable import Rapid

/// Contract for ``ChatViewModel.humanize`` — the helper that turns a
/// raw transport / SSE error into the single sentence the user sees
/// in the red bubble + the error banner. The whole point is that
/// "The operation couldn't be completed (NSURLErrorDomain error
/// -1004)" never lands in front of the user — and, since the
/// user-facing copy sweep, neither do engine internals: raw HTTP
/// status codes, server response bodies, the "rapid-mlx" engine name,
/// or transport-layer jargon. The user always gets a plain, actionable
/// recovery path; the diagnostics live in the logs.
///
/// We pin one assertion per branch so a regression that drops a case
/// (or reworms an engine internal back into it) fails loud.
@Suite("ChatViewModel.humanize maps errors to actionable copy")
struct HumanizeErrorTests {
    @Test("streamTruncated points at a restart path without leaking the engine name")
    func streamTruncated() {
        let msg = ChatViewModel.humanize(ChatStreamError.streamTruncated)
        #expect(!msg.contains("rapid-mlx"))
        #expect(msg.lowercased().contains("crash") || msg.lowercased().contains("model"))
        #expect(msg.contains("Restart"))
    }

    @Test("httpStatus never leaks the raw status code to the user")
    func httpStatusEmptyBody() {
        let msg = ChatViewModel.humanize(ChatStreamError.httpStatus(422, ""))
        #expect(!msg.contains("422"))
        #expect(!msg.contains("rapid-mlx"))
        #expect(msg.lowercased().contains("try again") || msg.lowercased().contains("restart"))
    }

    @Test("httpStatus never leaks the server body to the user")
    func httpStatusWithBody() {
        let msg = ChatViewModel.humanize(
            ChatStreamError.httpStatus(400, "alias 'foo' unknown")
        )
        #expect(!msg.contains("400"))
        #expect(!msg.contains("alias 'foo' unknown"))
        #expect(msg.lowercased().contains("try again") || msg.lowercased().contains("restart"))
    }

    @Test("Structured invalid-request reason is available only to the attachment failure path")
    func structuredInvalidRequestReason() {
        let body = #"{"error":{"message":"This model is serving text-only; image input is unsupported.","type":"invalid_request_error","code":"image_input_unsupported"}}"#
        let error = ChatStreamError.httpStatus(400, body)
        #expect(
            error.attachmentFailureMessage
                == "This model is serving text-only; image input is unsupported."
        )
        // The generic humanizer remains deliberately opaque for callers that
        // did not originate an image turn.
        #expect(!ChatViewModel.humanize(error).contains("image input"))
    }

    @Test("Unrelated structured server failures remain private diagnostics")
    func structuredServerFailureReasonStaysPrivate() {
        #expect(ChatStreamError.httpStatus(
            500,
            #"{"error":{"message":"Internal server error"}}"#
        ).attachmentFailureMessage == nil)
        #expect(ChatStreamError.httpStatus(
            503,
            #"{"error":{"message":"Metal is out of memory","type":"server_error","code":"oom"}}"#
        ).attachmentFailureMessage == nil)
    }

    @Test("Unstructured error bodies never become user copy")
    func unstructuredBodiesStayPrivate() {
        #expect(ChatStreamError.httpStatus(
            400,
            "raw internal detail"
        ).attachmentFailureMessage == nil)
    }

    @Test("transport never leaks the raw transport message")
    func transportPassthrough() {
        let msg = ChatViewModel.humanize(
            ChatStreamError.transport("connection reset by peer")
        )
        #expect(!msg.lowercased().contains("transport"))
        #expect(!msg.contains("connection reset by peer"))
        #expect(msg.contains("Restart"))
    }

    @Test("URLError timeout suggests a shorter prompt or restart")
    func urlTimeout() {
        let err = NSError(domain: NSURLErrorDomain, code: NSURLErrorTimedOut)
        let msg = ChatViewModel.humanize(err)
        // v0.5.16 onboarding-copy rewrite (#35) replaced "Timed out"
        // with the user-friendly "stopped responding" phrasing; both
        // the symptom and an actionable tail must still surface.
        #expect(msg.lowercased().contains("stopped responding") || msg.lowercased().contains("timed out"))
        #expect(msg.lowercased().contains("restart") || msg.lowercased().contains("shorter"))
    }

    @Test("URLError cannotConnectToHost points the user at the model bar")
    func urlCannotConnect() {
        let err = NSError(domain: NSURLErrorDomain, code: NSURLErrorCannotConnectToHost)
        let msg = ChatViewModel.humanize(err)
        // Points at the visible affordance ("the model bar at the top
        // to restart it"). Pin both the symptom + the action tail.
        #expect(msg.lowercased().contains("reach"))
        #expect(msg.lowercased().contains("restart"))
    }

    @Test("URLError networkConnectionLost points at model restart")
    func urlNetworkLost() {
        let err = NSError(domain: NSURLErrorDomain, code: NSURLErrorNetworkConnectionLost)
        let msg = ChatViewModel.humanize(err)
        #expect(msg.lowercased().contains("lost") || msg.lowercased().contains("disconnected"))
        #expect(msg.lowercased().contains("restart"))
    }

    @Test("URLError notConnectedToInternet explains the local-only caveat")
    func urlNotConnected() {
        let err = NSError(domain: NSURLErrorDomain, code: NSURLErrorNotConnectedToInternet)
        let msg = ChatViewModel.humanize(err)
        // The model runs entirely on the user's Mac, so a missing
        // internet connection usually doesn't matter — same caveat,
        // plain English. Pin both the symptom + the unusual-state hint.
        #expect(msg.lowercased().contains("network"))
        #expect(
            msg.lowercased().contains("doesn't matter")
            || msg.lowercased().contains("shouldn't matter")
            || msg.lowercased().contains("locally")
            || msg.lowercased().contains("on your mac")
        )
    }

    @Test("Unknown URLError code still surfaces a humanized fallback")
    func urlUnknownCode() {
        // -99 is not one of the cases we map explicitly.
        let err = NSError(
            domain: NSURLErrorDomain,
            code: -99,
            userInfo: [NSLocalizedDescriptionKey: "weird url thing"]
        )
        let msg = ChatViewModel.humanize(err)
        // Unmapped network errors must NOT leak the raw NSURLError code
        // or the localized body — the user gets a generic, actionable
        // recovery path; the raw detail is logged at the call site.
        #expect(msg.lowercased().contains("couldn't reach") || msg.lowercased().contains("network"))
        #expect(!msg.contains("weird url thing"))
        #expect(!msg.contains("NSURLErrorDomain"))
        #expect(msg.lowercased().contains("restart") || msg.lowercased().contains("try again"))
    }

    // MARK: - #471 capacity classification (genuine OOM vs busy vs generic)

    /// The real D-METAL-CAP admission-gate body the sidecar emits when a
    /// request's weights + projected KV exceed the GPU cap.
    private static let metalCapBody =
        "Server is busy (max concurrent requests reached). Retry after the Retry-After delay. "
        + "(Backpressure(error): Metal active 6.7GB + reserved KV 0.0GB + projected KV 7.7GB "
        + "would exceed gpu_memory_utilization cap 11.6GB (0-METAL-CAP))"

    @Test("A genuine out-of-memory 503 reads as a memory problem with a smaller-model path (#471)")
    func httpStatusOutOfMemory() {
        // Note: the sidecar's 503 detail happens to also carry the
        // "max concurrent" phrasing, but the METAL-CAP memory signal must
        // win — the actionable recovery is "smaller model / shorter", not
        // "just wait".
        let msg = ChatViewModel.humanize(ChatStreamError.httpStatus(503, Self.metalCapBody))
        #expect(msg.lowercased().contains("memory"))
        #expect(msg.lowercased().contains("smaller model") || msg.lowercased().contains("shorter"))
        // Engine internals still never surface.
        #expect(!msg.contains("503"))
        #expect(!msg.contains("METAL-CAP"))
        #expect(!msg.lowercased().contains("gpu_memory_utilization"))
        #expect(!msg.contains("rapid-mlx"))
    }

    @Test("A memory cap that trips mid-stream (transport) gets the same OOM copy (#471)")
    func transportOutOfMemory() {
        let msg = ChatViewModel.humanize(
            ChatStreamError.transport("projected KV 7.7GB would exceed gpu_memory_utilization cap (0-METAL-CAP)")
        )
        #expect(msg.lowercased().contains("memory"))
        #expect(msg.lowercased().contains("smaller model") || msg.lowercased().contains("shorter"))
        #expect(!msg.lowercased().contains("transport"))
        #expect(!msg.contains("METAL-CAP"))
    }

    @Test("A pure concurrency-backpressure 503 says wait, NOT downsize (#471)")
    func httpStatusServerBusy() {
        let msg = ChatViewModel.humanize(
            ChatStreamError.httpStatus(503, "Server is busy (max_concurrent_requests=10 reached). Retry after the Retry-After delay.")
        )
        #expect(msg.lowercased().contains("busy") || msg.lowercased().contains("moment"))
        // Busy is transient — must NOT push the user to a smaller model.
        #expect(!msg.lowercased().contains("smaller model"))
        #expect(!msg.contains("503"))
    }

    @Test("A non-capacity httpStatus still falls back to the generic message (no false OOM/busy)")
    func httpStatusGenericUnaffected() {
        let msg = ChatViewModel.humanize(ChatStreamError.httpStatus(400, "alias 'foo' unknown"))
        #expect(!msg.lowercased().contains("memory"))
        #expect(!msg.lowercased().contains("busy"))
        #expect(msg.lowercased().contains("try again") || msg.lowercased().contains("restart"))
    }

    @Test("capacityKind classifies memory, busy, and neither distinctly")
    func capacityKindClassifier() {
        #expect(ChatViewModel.capacityKind(from: "... would exceed gpu_memory_utilization cap ...") == .outOfMemory)
        #expect(ChatViewModel.capacityKind(from: "Metal active 8GB, out of memory") == .outOfMemory)
        #expect(ChatViewModel.capacityKind(from: "max_concurrent_requests=10 reached") == .serverBusy)
        #expect(ChatViewModel.capacityKind(from: "alias 'foo' unknown") == .none)
        #expect(ChatViewModel.capacityKind(from: "") == .none)
        // Case-insensitive on an OOM-specific phrase.
        #expect(ChatViewModel.capacityKind(from: "PROJECTED KV cap") == .outOfMemory)
        // A context-length overflow ("post-build repair prompt would exceed
        // model context …", chat.py) is NOT an OOM — its fix is a shorter
        // prompt, not a smaller model — so it must NOT classify as memory.
        #expect(ChatViewModel.capacityKind(from: "post-build repair prompt would exceed model context window") == .none)
    }

    @Test("A context-length overflow is not misread as out-of-memory (#471 false-positive guard)")
    func contextOverflowIsNotOOM() {
        let msg = ChatViewModel.humanize(
            ChatStreamError.httpStatus(400, "repair prompt would exceed model context")
        )
        // Must fall to the generic message, not the "smaller model" OOM copy.
        #expect(!msg.lowercased().contains("smaller model"))
        #expect(msg.lowercased().contains("try again") || msg.lowercased().contains("restart"))
    }

    @Test("Non-URL, non-ChatStream errors get a generic actionable fallback — the raw body is logged, never shown")
    func unknownErrorFallthrough() {
        struct Custom: LocalizedError {
            var errorDescription: String? { "raw decode failure 0x8badf00d" }
        }
        let msg = ChatViewModel.humanize(Custom())
        // The fallthrough is reached only by system / library errors we
        // didn't author for the user (ChatStreamError is handled above).
        // Their localizedDescription is a raw diagnostic and must NOT
        // surface — the user gets a plain, actionable recovery path.
        #expect(!msg.contains("raw decode failure 0x8badf00d"))
        #expect(msg.lowercased().contains("try again") || msg.lowercased().contains("restart"))
    }
}
