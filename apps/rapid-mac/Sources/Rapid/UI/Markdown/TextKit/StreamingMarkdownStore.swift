import Foundation
import Observation

/// Compiled markdown for the message currently streaming.
///
/// ## Why this exists
///
/// `ChatView`'s transcript is `ForEach(messages) { MessageRow(message: …) }`,
/// and a streaming turn mutates `messages[i].content` on every coalesced SSE
/// batch. SwiftUI therefore rebuilds that row — and everything under it —
/// once per batch, at the transport's cadence rather than the renderer's.
///
/// Measured on a 5 760-character answer streamed at ~313 chars/second: the
/// stream reader's `await MainActor.run` waited **55.5 ms** per hop while the
/// closure it was waiting to run took **0.0 ms**. The main thread was not
/// doing our work; it was rebuilding view trees. End to end the reply took
/// **181 s** against an 18 s transmission. The same fixture in the prototype
/// this renderer came from took 19.9 s, and its hops waited 2.1 ms.
///
/// The prototype avoids this by keeping the streaming text out of the row's
/// inputs: its rows carry an already-compiled `MarkdownResult` that changes on
/// the compiler's 100 ms beat, not the raw string that changes 20× a second.
/// This type is that seam for Rapid — the streaming row reads compiled blocks
/// from here, so an SSE batch no longer invalidates the row.
///
/// Only the streaming message needs it. Settled messages compile once and
/// render through `TextKitMarkdownView` as before.
@MainActor
@Observable
final class StreamingMarkdownStore {

    /// Compiled blocks for the message being streamed.
    private(set) var result: MarkdownResult = .empty
    /// Which message `result` belongs to. Nil when nothing is streaming.
    private(set) var messageID: UUID?

    private let compiler = MarkdownCompiler()
    private var pendingText: String?
    private var flushTask: Task<Void, Never>?
    private var revision = 0

    /// How long to accumulate before recompiling.
    ///
    /// A full compile of a 24 000-character buffer measures 15 ms, so 100 ms
    /// leaves the main thread mostly free, and block structure appearing
    /// within one perceptual beat is fast enough that the delay is invisible.
    private let coalesceInterval: Duration = .milliseconds(100)

    /// Note new streamed text. Compilation happens on the next flush.
    func enqueue(id: UUID, text: String) {
        if messageID != id {
            // A new turn: drop the previous message's blocks rather than
            // letting them show under the new one for a frame.
            messageID = id
            result = .empty
            revision = 0
        }
        pendingText = text
        scheduleFlush()
    }

    /// Compile whatever is queued right now.
    ///
    /// Called when a stream ends so the final tokens are not left waiting on
    /// a timer — a visible pause between the last token and the last words.
    func flushNow() {
        flushTask?.cancel()
        flushTask = nil
        flush()
    }

    /// Forget the current message. The settled row takes over rendering.
    func finish() {
        flushNow()
        messageID = nil
        result = .empty
    }

    private func scheduleFlush() {
        guard flushTask == nil else { return }
        flushTask = Task { [weak self] in
            guard let self else { return }
            try? await Task.sleep(for: self.coalesceInterval)
            guard !Task.isCancelled else { return }
            self.flushTask = nil
            self.flush()
        }
    }

    private func flush() {
        guard let text = pendingText else { return }
        pendingText = nil
        revision += 1
        result = compiler.compile(text, revision: revision, isComplete: false)
    }
}
