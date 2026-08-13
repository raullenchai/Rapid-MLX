import Foundation


/// ⚠️ UNWIRED — not instantiated anywhere in this app. The production
/// streaming compiler is ``StreamingMarkdownStore`` (also a 100 ms debounce,
/// per-message, with a revision token); this class is a prototype port from
/// native-chat that nothing calls. Keep it for a future PR or delete it —
/// either way it must not be mistaken for the live stage-two below.
///
/// Debounced markdown compilation for streaming messages.
///
/// Stage two of a two-stage pipeline. `SSEDeltaCoalescer` (already in the
/// transplanted `ChatStreamClient`) batches transport deltas on a 16ms window
/// and updates the message model; this then compiles markdown on a slower
/// beat and caches the result.
///
///     SSE bytes
///       → SSEDeltaCoalescer (16ms)      transport batching
///       → ChatViewModel.messages[i]     model truth, no rendering
///       → MessageMarkdownCompiler       ← here, 100ms
///       → cached MarkdownResult
///       → MarkdownBlockStack renders
///
/// Why 100ms: a full cmark reparse of a 4KB buffer is ~0.3ms, so the interval
/// is generous on cost; and block-level structure appearing (a list item
/// completing, a fence closing) within one perceptual beat is fast enough that
/// the delay is invisible. ChatGPT ships the same shape as
/// `StreamingCompilationCoalescingPolicy`.
@MainActor
final class MessageMarkdownCompiler {

    private let compiler = MarkdownCompiler()
    private var cache: [UUID: MarkdownResult] = [:]
    private var pending: [UUID: String] = [:]
    private var flushTask: Task<Void, Never>?
    private var revisionCounter = 0

    /// How long to accumulate before recompiling.
    let coalesceInterval: Duration

    /// Called after a coalesced flush produces new results.
    var onFlush: (() -> Void)?

    init(coalesceInterval: Duration = .milliseconds(100)) {
        self.coalesceInterval = coalesceInterval
    }

    /// Compile immediately. For messages that are already complete —
    /// restored history, a user message being sent — where there is nothing
    /// to coalesce and waiting would just delay first paint.
    func compileNow(id: UUID, text: String) -> MarkdownResult {
        revisionCounter += 1
        let result = compiler.compile(text, revision: revisionCounter)
        cache[id] = result
        pending[id] = nil
        return result
    }

    /// Queue a streaming update. Compilation happens on the next flush.
    func enqueue(id: UUID, text: String) {
        pending[id] = text
        scheduleFlush()
    }

    /// Latest compiled result, or an empty document if nothing has compiled
    /// yet. Never compiles as a side effect of being read — a getter that can
    /// trigger a parse turns every render pass into a potential stall.
    func result(for id: UUID) -> MarkdownResult {
        cache[id] ?? .empty
    }

    /// Drop a message's cached result.
    func invalidate(id: UUID) {
        cache[id] = nil
        pending[id] = nil
    }

    func reset() {
        flushTask?.cancel()
        flushTask = nil
        cache.removeAll()
        pending.removeAll()
    }

    /// Force any queued work to compile now — used when a stream ends, so the
    /// final tokens are not left waiting on a timer.
    func flushNow() {
        flushTask?.cancel()
        flushTask = nil
        flush()
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
        guard !pending.isEmpty else { return }
        for (id, text) in pending {
            revisionCounter += 1
            cache[id] = compiler.compile(text, revision: revisionCounter)
        }
        pending.removeAll()
        onFlush?()
    }
}
