import Foundation

/// One completion, taken for the app's own purposes rather than the reader's.
///
/// Both background features — naming a conversation and proposing follow-up
/// questions — need the same thing: ask the model something short, get a
/// string back, and if anything at all goes wrong, get nothing back and say
/// nothing about it. That contract is the whole type.
///
/// ## Why it streams
///
/// The wire body's `stream` is hardcoded `true` (``ChatStreamClient``), and a
/// one-shot `stream: false` path would be the first desktop code to use the
/// server's non-streaming branch — untested here, and worse, invisible to the
/// one harness that can drive this end to end: `scripts/fake-rapid-mlx.sh`
/// answers every chat request with `Content-Type: text/event-stream` and never
/// reads the `stream` field at all, so a JSON-expecting client would decode-fail
/// in the golden flows and silently fall back forever. Accumulating deltas also
/// inherits what ``ChatStreamClient/send(_:bearerToken:onEvent:)`` already
/// owns: bearer injection, the pre-stream retry, HTTP-status bodies drained
/// into a readable error, the SSE line cap, `[DONE]`, and both cancellation
/// shapes. For a ≤96-token reply the framing overhead is noise.
///
/// ## Why it can never be loud
///
/// There is no error surface, no retry, and no log the reader sees — the same
/// contract ``ServerProfileFetcher`` states for its own background fetch:
/// there is no UI affordance for "the title call failed" because there is
/// nothing the reader could do about it. Every failure is `nil`, and every
/// caller treats `nil` as "leave things as they are".
@MainActor
struct BackgroundCompletionClient {

    /// Where to send, resolved at call time.
    ///
    /// Never cached: ``ServerManager/activeBearer`` rotates on every `start()`
    /// and is cleared on stop and on crash, so a handle held across a restart
    /// is a guaranteed 401. The alias comes from the `.ready(alias:)` state
    /// for the same reason it must not be chosen freely — naming a model that
    /// is not resident would issue a real `/v1/models/load`, and with the
    /// assistant replacement group that evicts the model the reader is using.
    struct Target {
        let port: Int
        let bearer: String?
        let alias: String
    }

    /// Test seam. Production leaves this nil and takes
    /// ``ChatStreamClient``'s shared ephemeral session.
    var session: URLSession?

    /// The reply text, or nil.
    ///
    /// `deadline` is a ceiling on the whole call, separate from the client's
    /// inactivity timeout: a background request that is merely slow is one we
    /// would rather abandon than let sit in the engine's batch.
    func complete(
        _ messages: [ChatMessage],
        target: Target,
        maxTokens: Int,
        temperature: Double,
        topP: Double = 0.9,
        deadline: Duration
    ) async -> String? {
        var client = ChatStreamClient(
            baseURL: ChatStreamClient.loopbackURL(port: target.port),
            session: session
        )
        client.requestTimeout = 20

        let request = ChatStreamClient.Request(
            alias: target.alias,
            messages: messages,
            temperature: temperature,
            topP: topP,
            maxTokens: maxTokens,
            // No tools: a background call that advertises them can finish
            // with `tool_calls` and no content at all.
            tools: nil,
            // Mandatory. On a hybrid model with thinking on, the reasoning
            // trace alone exhausts a 24-token budget and the reply comes back
            // `finish_reason: "length"` with empty content — the failure
            // ``SamplingConfig`` documents. Both features would be
            // permanently dead on Qwen 3.x and its relatives.
            enableThinking: false,
            forcedTool: nil
        )
        // Note what is NOT here: any stop sequence. ``Request`` cannot express
        // one, which is the point — the engine's batch-compatibility key is
        // `(frozenset(stop_token_ids), bool(ignore_eos))`, and a request whose
        // key differs from the live batch is requeued until every running
        // request drains. Being unable to set it is a stronger guarantee than
        // remembering not to.

        let sink = TextSink()
        // Two tasks racing rather than a task group: the send closure
        // captures a main-actor client, which a group's `sending` parameter
        // will not accept. Cancelling the work closes the socket, which the
        // engine's disconnect guard turns into an abort that frees its
        // admission slot — so an abandoned call stops costing anything.
        let work = Task { @MainActor in
            try await client.send(request, bearerToken: target.bearer) { event in
                if case .content(let delta) = event { sink.text += delta }
            }
        }
        let timeout = Task {
            try await Task.sleep(for: deadline)
            work.cancel()
        }
        defer { timeout.cancel() }

        do {
            try await work.value
        } catch {
            return nil
        }

        let text = sink.text.trimmingCharacters(in: .whitespacesAndNewlines)
        return text.isEmpty ? nil : text
    }

    /// The accumulator has to be a reference: ``ChatStreamClient/send``'s
    /// handler is an escaping `@MainActor` closure, which cannot capture a
    /// mutable local.
    @MainActor private final class TextSink {
        var text = ""
    }
}
