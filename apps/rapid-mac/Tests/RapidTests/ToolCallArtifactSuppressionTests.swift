import Foundation
import Testing
@testable import Rapid

/// Issue #513 — the render-time safety net that suppresses a raw
/// tool-call artifact (a malformed envelope the engine parser couldn't
/// recover) when tools were advertised but the assistant fired no
/// ``tool_calls``.
///
/// Two layers under test:
///   * ``ChatMessage/contentLooksLikeToolCallArtifact`` — the detector.
///     The headline risk is a FALSE POSITIVE eating a legitimate answer,
///     so the negative cases carry the weight here.
///   * ``ChatMessage/shouldSuppressToolCallArtifact`` — the gate that
///     combines the detector with tools-advertised + zero-tool_calls.
@Suite("Tool-call artifact suppression (#513)")
struct ToolCallArtifactSuppressionTests {

    // MARK: - Detector: positive cases (real leaked artifacts)

    @Test("Hermes <tool_call> envelope at the start of the turn is an artifact")
    func hermesEnvelope() {
        #expect(ChatMessage.contentLooksLikeToolCallArtifact(
            "<tool_call>{\"name\": \"search\", \"arguments\": {\"q\": \"x\"}}</tool_call>"))
    }

    @Test("Truncated <tool_call><parameter=…> fragment (the qwen3.6-27b repro) is an artifact")
    func truncatedHermesFragment() {
        // The exact shape #513 cites: a truncated `<tool_call><parameter=query>…`.
        #expect(ChatMessage.contentLooksLikeToolCallArtifact(
            "<tool_call><parameter=query>weather in Paris"))
    }

    @Test("<function=…> / <parameter=…> lead fragments are artifacts")
    func functionAndParameterFragments() {
        #expect(ChatMessage.contentLooksLikeToolCallArtifact("<function=get_weather>{\"city\":\"NYC\"}"))
        #expect(ChatMessage.contentLooksLikeToolCallArtifact("<parameter=query>london"))
    }

    @Test("Mistral [TOOL_CALLS] marker leading the turn is an artifact")
    func mistralMarker() {
        #expect(ChatMessage.contentLooksLikeToolCallArtifact(
            "[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Berlin\"}"))
    }

    @Test("The canonical OpenAI wire tool-call object is an artifact")
    func openAIWireShape() {
        // The project's own `ToolCall` wire shape (ToolKit.swift). `id` /
        // `type` are not call vocabulary and `arguments` is nested inside
        // `function`, so this is matched structurally, not by key-set.
        #expect(ChatMessage.contentLooksLikeToolCallArtifact(
            "{\"id\": \"call_1\", \"type\": \"function\", \"function\": {\"name\": \"search\", \"arguments\": \"{}\"}}"))
        // A bare `{"function": {...}}` wrapper (no id/type) too.
        #expect(ChatMessage.contentLooksLikeToolCallArtifact(
            "{\"function\": {\"name\": \"get_weather\", \"arguments\": {\"city\": \"NYC\"}}}"))
    }

    @Test("A <JSON>…</JSON> raw wrapper leak is an artifact")
    func jsonWrapperLeak() {
        // The documented raw wrapper (ToolUseCapability) some models emit
        // when the parser can't recover the call.
        #expect(ChatMessage.contentLooksLikeToolCallArtifact(
            "<JSON>{\"name\": \"search\", \"arguments\": {\"q\": \"x\"}}</JSON>"))
    }

    @Test("A truncated canonical OpenAI wire leak (begins with id) is an artifact")
    func truncatedOpenAIWireShape() {
        // Cut off mid-`arguments`, so it never parses and its first key is
        // `id` — not a call key. `"type":"function"` + the args token pin
        // it (codex r2 MAJOR-3).
        #expect(ChatMessage.contentLooksLikeToolCallArtifact(
            "{\"id\": \"call_1\", \"type\": \"function\", \"function\": {\"name\": \"search\", \"arguments\": \"{\\\"q\\\":"))
    }

    @Test("DeepSeek tool-calls-begin envelope is an artifact")
    func deepseekEnvelope() {
        // The U+2581-separated `tool▁calls▁begin` token no wired parser
        // recovered (issue #513 / ToolUseCapability notes).
        #expect(ChatMessage.contentLooksLikeToolCallArtifact(
            "\u{FF1C}\u{FF5C}tool\u{2581}calls\u{2581}begin\u{FF5C}\u{FF1E}get_weather"))
    }

    @Test("A whole-content bare tool-call JSON object is an artifact")
    func bareToolCallJSON() {
        #expect(ChatMessage.contentLooksLikeToolCallArtifact(
            "{\"name\": \"search\", \"arguments\": {\"query\": \"cats\"}}"))
        // The ReAct `action` / `action_input` shape too.
        #expect(ChatMessage.contentLooksLikeToolCallArtifact(
            "{\"action\": \"search\", \"action_input\": {\"query\": \"cats\"}}"))
    }

    @Test("A tool-call JSON object wrapped in a whole-content ```json fence is an artifact")
    func fencedToolCallJSON() {
        #expect(ChatMessage.contentLooksLikeToolCallArtifact(
            "```json\n{\"name\": \"calc\", \"arguments\": {\"a\": 1}}\n```"))
    }

    @Test("A truncated tool-call JSON dump (names a tool + starts its args) is an artifact")
    func truncatedToolCallJSON() {
        #expect(ChatMessage.contentLooksLikeToolCallArtifact(
            "{\"name\": \"search\", \"arguments\": {\"query\": \"unfinis"))
    }

    // MARK: - Detector: negative cases (legitimate content — must NOT match)

    @Test("Ordinary prose is not an artifact")
    func prose() {
        #expect(!ChatMessage.contentLooksLikeToolCallArtifact(
            "Paris is the capital of France. It sits on the Seine."))
    }

    @Test("An answer that merely MENTIONS a tool-call tag in prose is not an artifact")
    func proseMentioningEnvelope() {
        // The marker isn't at the start of the turn — this is a genuine
        // explanation, not a leaked call.
        #expect(!ChatMessage.contentLooksLikeToolCallArtifact(
            "To call a tool, the model emits a <tool_call> block, e.g. <tool_call>{...}</tool_call>."))
        #expect(!ChatMessage.contentLooksLikeToolCallArtifact(
            "Mistral models use a [TOOL_CALLS] prefix — wait, actually that IS a marker, so this line is intentionally led by prose."))
    }

    @Test("A here's-a-JSON-example answer with prose framing is not an artifact")
    func jsonExampleWithProse() {
        #expect(!ChatMessage.contentLooksLikeToolCallArtifact(
            "Here's a JSON example:\n\n```json\n{\"name\": \"Alice\", \"age\": 30}\n```"))
    }

    @Test("A complete JSON answer that names something but carries no args is not an artifact")
    func completeNonCallJSON() {
        // `name` present but no arguments/parameters key, and a non-call
        // key (`age`) — a person record, not a tool call.
        #expect(!ChatMessage.contentLooksLikeToolCallArtifact(
            "{\"name\": \"Alice\", \"age\": 30}"))
        // Pure data object, no tool vocabulary at all.
        #expect(!ChatMessage.contentLooksLikeToolCallArtifact(
            "{\"user\": \"bob\", \"score\": 42}"))
    }

    @Test("A truncated ordinary object (person record cut off) is not an artifact")
    func truncatedNonCallJSON() {
        // Opens with `name` but never spells out an args-ish key — a
        // real leaked call would. Must not be mistaken for one.
        #expect(!ChatMessage.contentLooksLikeToolCallArtifact(
            "{\"name\": \"Bob\", \"age\": 3"))
    }

    @Test("A code answer whose fenced block is non-tool JSON is not an artifact")
    func codeAnswerFencedData() {
        #expect(!ChatMessage.contentLooksLikeToolCallArtifact(
            "```json\n{\"items\": [1, 2, 3], \"total\": 6}\n```"))
    }

    @Test("A tool DEFINITION / JSON-Schema object is not an artifact")
    func toolDefinitionIsNotACall() {
        // `name` + `parameters` where `parameters` is a JSON Schema — this
        // DESCRIBES a tool, it does not call one. `parameters` is not an
        // args key, and the schema keys veto the match.
        #expect(!ChatMessage.contentLooksLikeToolCallArtifact(
            "{\"name\": \"get_weather\", \"parameters\": {\"type\": \"object\"}}"))
        #expect(!ChatMessage.contentLooksLikeToolCallArtifact(
            "{\"name\": \"get_weather\", \"description\": \"Get weather\", \"parameters\": {\"type\": \"object\", \"properties\": {}}}"))
        // Fenced form of the same definition.
        #expect(!ChatMessage.contentLooksLikeToolCallArtifact(
            "```json\n{\"name\": \"search\", \"description\": \"Search the web\", \"parameters\": {\"type\": \"object\"}}\n```"))
    }

    @Test("A canonical OpenAI tool DEFINITION (function wrapper + schema) is not an artifact")
    func openAIToolDefinitionIsNotACall() {
        // The OpenAI tool-DEFINITION wire shape nests name + description +
        // parameters-schema under `function` — the same key a CALL uses.
        // The nested object carries no `arguments`, so it must NOT be
        // suppressed (codex r2 MAJOR-1).
        #expect(!ChatMessage.contentLooksLikeToolCallArtifact(
            "{\"type\": \"function\", \"function\": {\"name\": \"search\", \"description\": \"Search the web\", \"parameters\": {\"type\": \"object\", \"properties\": {}}}}"))
    }

    @Test("An answer that LEADS with a tool token but explains it is not an artifact")
    func markerLeadWithoutPayloadIsNotACall() {
        // The user asked what the token means; the answer opens with it but
        // is followed by prose, not a payload (codex r2 MAJOR-2).
        #expect(!ChatMessage.contentLooksLikeToolCallArtifact(
            "<JSON> is a wrapper some engines use to carry a tool call over the wire."))
        #expect(!ChatMessage.contentLooksLikeToolCallArtifact(
            "<tool_call> is a special XML-style tag the model emits to invoke a tool."))
        #expect(!ChatMessage.contentLooksLikeToolCallArtifact(
            "[TOOL_CALLS] is Mistral's prefix marker; it precedes the function name."))
    }

    @Test("A ```xml example whose body begins <tool_call> is not an artifact")
    func xmlExampleFenceIsNotUnwrapped() {
        // The user asked to SEE the markup; a non-JSON fence must never be
        // unwrapped into the envelope checks (codex r1 MAJOR-1).
        #expect(!ChatMessage.contentLooksLikeToolCallArtifact(
            "```xml\n<tool_call>{\"name\": \"search\"}</tool_call>\n```"))
        // A bare ``` fence whose body is example markup, likewise.
        #expect(!ChatMessage.contentLooksLikeToolCallArtifact(
            "```\n<tool_call>{\"name\": \"search\"}</tool_call>\n```"))
    }

    @Test("A multi-block answer whose first block is tool-shaped is not an artifact")
    func multiBlockAnswerIsNotUnwrapped() {
        // Two fenced blocks — not a single whole-content fence, so it is
        // left intact and its `{`-less prefix fails the JSON branch.
        #expect(!ChatMessage.contentLooksLikeToolCallArtifact(
            "```json\n{\"name\": \"search\", \"arguments\": {}}\n```\n\nAnd here's another:\n\n```json\n{\"x\": 1}\n```"))
    }

    @Test("Empty / whitespace content is not an artifact")
    func emptyContent() {
        #expect(!ChatMessage.contentLooksLikeToolCallArtifact(""))
        #expect(!ChatMessage.contentLooksLikeToolCallArtifact("   \n\t "))
    }

    // MARK: - Gate: shouldSuppressToolCallArtifact

    private let artifact = "{\"name\": \"search\", \"arguments\": {\"q\": \"x\"}}"

    @Test("Gate fires when tools were advertised, no tool_calls, and content is an artifact")
    func gateFires() {
        #expect(ChatMessage.shouldSuppressToolCallArtifact(
            content: artifact, toolCalls: nil, finishReason: "stop", toolsRequested: true))
    }

    @Test("Gate does NOT fire when tools were not advertised")
    func gateNoTools() {
        #expect(!ChatMessage.shouldSuppressToolCallArtifact(
            content: artifact, toolCalls: nil, finishReason: "stop", toolsRequested: false))
    }

    @Test("Gate does NOT fire when the model actually produced a tool_call")
    func gateHasToolCall() {
        let call = ToolCall(id: "1", name: "search", arguments: "{}")
        #expect(!ChatMessage.shouldSuppressToolCallArtifact(
            content: artifact, toolCalls: [call], finishReason: "tool_calls", toolsRequested: true))
    }

    @Test("Gate does NOT fire when finish_reason is tool_calls even with an empty array")
    func gateFinishToolCalls() {
        #expect(!ChatMessage.shouldSuppressToolCallArtifact(
            content: artifact, toolCalls: [], finishReason: "tool_calls", toolsRequested: true))
    }

    @Test("Gate does NOT fire on a genuine answer even with tools advertised + no calls")
    func gateGenuineAnswer() {
        #expect(!ChatMessage.shouldSuppressToolCallArtifact(
            content: "The weather in Paris is 14°C and clear.",
            toolCalls: nil, finishReason: "stop", toolsRequested: true))
    }

    // MARK: - Persistence + copy

    @Test("toolCallArtifactSuppressed round-trips through Codable")
    func flagRoundTrips() throws {
        var msg = ChatMessage(role: .assistant, content: artifact, status: .complete)
        msg.toolCallArtifactSuppressed = true
        let data = try JSONEncoder().encode(msg)
        let decoded = try JSONDecoder().decode(ChatMessage.self, from: data)
        #expect(decoded.toolCallArtifactSuppressed)
    }

    @Test("Old sessions without the key decode the flag as false")
    func backCompatDefaultsFalse() throws {
        // A message JSON that predates #513 — no toolCallArtifactSuppressed key.
        let json = """
        {
          "id": "\(UUID().uuidString)",
          "role": "assistant",
          "content": "hi",
          "reasoning": "",
          "status": "complete",
          "reasoningTruncated": false,
          "contentTruncated": false,
          "toolNotCalledFlagged": false,
          "createdAt": 0
        }
        """
        let decoded = try JSONDecoder().decode(ChatMessage.self, from: Data(json.utf8))
        #expect(!decoded.toolCallArtifactSuppressed)
    }

    // MARK: - Trailing artifact: prose answered, then the tail broke

    /// The 0.14.1 dogfood repro, abbreviated: a real answer that ends by
    /// promising an action, then an envelope nothing claimed. No tool round
    /// fired, so the promise was never kept and the user saw only the
    /// promise — for six minutes.
    @Test("Prose followed by a <tool_call> envelope keeps the prose and flags the tail")
    func trailingEnvelopeKeepsProse() {
        let content = """
        Let me read the first page more carefully with a higher offset:

        <tool_call> {"name":"read_document","arguments":{"document_id":"abc","greP":"Statement
        """
        let prose = ChatMessage.trailingToolCallArtifactProse(in: content)
        #expect(prose == "Let me read the first page more carefully with a higher offset:")
        // Whole-turn detection does NOT fire — the artifact is the tail.
        #expect(!ChatMessage.contentLooksLikeToolCallArtifact(content))
        // …but the gate does, so the row gets the caption.
        #expect(ChatMessage.shouldSuppressToolCallArtifact(
            content: content, toolCalls: [], finishReason: "stop", toolsRequested: true))
        #expect(ChatMessage.proseAboveSuppressedToolCallArtifact(content: content) == prose)
    }

    @Test("A trailing [TOOL_CALLS] / DeepSeek marker is caught the same way")
    func trailingOtherFormats() {
        #expect(ChatMessage.trailingToolCallArtifactProse(
            in: "Sure, let me look that up.\n\n[TOOL_CALLS] [{\"name\":\"search\"}]"
        ) == "Sure, let me look that up.")
        #expect(ChatMessage.trailingToolCallArtifactProse(
            in: "I will check.\n<function=get_weather>{\"city\":\"NYC\"}"
        ) == "I will check.")
    }

    @Test("A trailing tag the answer only TALKS about is not an artifact")
    func trailingProseMentionIsNotAnArtifact() {
        // No payload after the marker → the `leadingEnvelopeLeak` gate holds.
        #expect(ChatMessage.trailingToolCallArtifactProse(
            in: "Hermes-style models wrap their calls in <tool_call> tags."
        ) == nil)
        #expect(ChatMessage.trailingToolCallArtifactProse(
            in: "The prefix Mistral uses is [TOOL_CALLS] and nothing else."
        ) == nil)
    }

    @Test("A fenced example ending the answer is not an artifact")
    func trailingFencedExampleIsNotAnArtifact() {
        let content = """
        Here is what a Hermes call looks like:

        ```xml
        <tool_call>{"name": "search", "arguments": {"q": "x"}}</tool_call>
        ```
        """
        #expect(ChatMessage.trailingToolCallArtifactProse(in: content) == nil)
        #expect(!ChatMessage.shouldSuppressToolCallArtifact(
            content: content, toolCalls: [], finishReason: "stop", toolsRequested: true))
    }

    @Test("An unclosed fence still counts as inside a fence")
    func trailingUnclosedFenceIsNotAnArtifact() {
        let content = "Example:\n\n```xml\n<tool_call>{\"name\": \"search\"}"
        #expect(ChatMessage.trailingToolCallArtifactProse(in: content) == nil)
    }

    @Test("A whole-turn artifact still renders the caption alone")
    func wholeTurnArtifactHasNoProse() {
        let content = "<tool_call>{\"name\": \"search\", \"arguments\": {\"q\": \"x\"}}</tool_call>"
        #expect(ChatMessage.trailingToolCallArtifactProse(in: content) == nil)
        #expect(ChatMessage.proseAboveSuppressedToolCallArtifact(content: content) == nil)
        #expect(ChatMessage.contentLooksLikeToolCallArtifact(content))
    }

    @Test("A trailing artifact is still gated on tools + zero calls")
    func trailingArtifactRespectsTheGates() {
        let content = "Let me look.\n\n<tool_call> {\"name\":\"search\",\"arguments\":{"
        #expect(!ChatMessage.shouldSuppressToolCallArtifact(
            content: content, toolCalls: [], finishReason: "stop", toolsRequested: false))
        #expect(!ChatMessage.shouldSuppressToolCallArtifact(
            content: content,
            toolCalls: [ToolCall(id: "1", name: "search", arguments: "{}")],
            finishReason: "stop",
            toolsRequested: true))
        #expect(!ChatMessage.shouldSuppressToolCallArtifact(
            content: content, toolCalls: [], finishReason: "tool_calls", toolsRequested: true))
    }

    /// The dangerous false positive the strict payload gate exists for: an
    /// answer that names the tag in a sentence and shows the example in a
    /// fence further down. A loose "carries a closing tag somewhere" gate
    /// would start the tail at the sentence and delete the explanation.
    @Test("An answer that names the tag then fences an example keeps all of it")
    func mentionThenFencedExampleIsNotAnArtifact() {
        let content = """
        Hermes models wrap their calls in <tool_call> tags. For example:

        ```xml
        <tool_call>{"name": "search", "arguments": {"q": "x"}}</tool_call>
        ```

        The engine strips them before you see the result.
        """
        #expect(ChatMessage.trailingToolCallArtifactProse(in: content) == nil)
        #expect(!ChatMessage.shouldSuppressToolCallArtifact(
            content: content, toolCalls: [], finishReason: "stop", toolsRequested: true))
    }

    /// A prose turn that trailed off into the DeepSeek marker used to be
    /// suppressed WHOLE (the marker matches anywhere), taking the answer
    /// with it.
    @Test("A trailing DeepSeek marker keeps the prose above it")
    func trailingDeepSeekMarkerKeepsProse() {
        let content = "I will look that up for you.\n<\u{FF5C}tool\u{2581}calls\u{2581}begin\u{FF5C}>"
        #expect(ChatMessage.contentLooksLikeToolCallArtifact(content))
        #expect(
            ChatMessage.proseAboveSuppressedToolCallArtifact(content: content)
                == "I will look that up for you."
        )
    }

    @Test("An ordinary long answer is untouched")
    func ordinaryAnswerHasNoTrailingArtifact() {
        #expect(ChatMessage.trailingToolCallArtifactProse(
            in: "The purchase order number is PO-5592-KX and the total is 14,208.55."
        ) == nil)
    }

    @Test("The suppressed-body caption carries no machine jargon")
    func captionHasNoJargon() {
        let copy = ChatMessage.toolCallArtifactSuppressedCaptionCopy.lowercased()
        #expect(!copy.isEmpty)
        for jargon in ["tool_call", "envelope", "parser", "json", "parameter=", "[tool_calls]"] {
            #expect(!copy.contains(jargon), "user-facing caption must not leak '\(jargon)'")
        }
    }
}
