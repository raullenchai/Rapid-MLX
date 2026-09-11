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

    @Test("A CLOSED fenced example does not hide a real envelope after it")
    func trailingEnvelopeAfterFencedExampleIsStillCaught() {
        // Adversarial review round 1 (codex, blocking): the detector used to
        // take the FIRST match of each pattern and then decide on it alone,
        // so a legitimate fenced example earlier in the turn made the whole
        // function answer nil — and the genuine unfenced envelope after it
        // rendered raw. The scan now skips the fenced candidate and keeps
        // going.
        let content = """
        A Hermes call looks like this:

        ```xml
        <tool_call>{"name": "search", "arguments": {"q": "x"}}</tool_call>
        ```

        Now let me actually run it:

        <tool_call> {"name":"search","arguments":{"q":"Statement
        """
        let prose = ChatMessage.trailingToolCallArtifactProse(in: content)
        // Everything up to the real envelope is kept — the fenced example
        // included, because that example IS part of the answer.
        #expect(prose?.hasPrefix("A Hermes call looks like this:") == true)
        #expect(prose?.hasSuffix("Now let me actually run it:") == true)
        #expect(prose?.contains("```xml") == true)
        #expect(ChatMessage.shouldSuppressToolCallArtifact(
            content: content, toolCalls: [], finishReason: "stop", toolsRequested: true))
    }

    @Test("Prose that documents <parameter=…> inline is not an artifact")
    func trailingParameterMentionIsNotAnArtifact() {
        // Adversarial review round 1 (codex, blocking): `<(function|parameter)=`
        // carried no payload requirement, so an answer EXPLAINING the syntax
        // was truncated at the tag. `<function=` now needs a payload and
        // `<parameter=` needs both its own line and a closing tag.
        for content in [
            "Use <parameter=name> to identify the field, then send the call.",
            "The fragment shape is <function=get_weather> with no arguments at all.",
            "Qwen writes <parameter=query> inline and closes it later; that is the format.",
        ] {
            #expect(ChatMessage.trailingToolCallArtifactProse(in: content) == nil)
            #expect(!ChatMessage.shouldSuppressToolCallArtifact(
                content: content, toolCalls: [], finishReason: "stop", toolsRequested: true))
        }
    }

    @Test("A real <function=…><parameter=…> args block IS an artifact")
    func trailingParameterBlockIsAnArtifact() {
        let content = """
        I will look up the weather for you.

        <function=get_weather>
        <parameter=city>NYC</parameter>
        </function>
        """
        #expect(ChatMessage.trailingToolCallArtifactProse(in: content)
            == "I will look up the weather for you.")
        // …and the `<parameter=` block alone (no `<function=` opener) too,
        // as long as it stands on its own line and closes.
        #expect(ChatMessage.trailingToolCallArtifactProse(
            in: "Checking now.\n<parameter=city>NYC</parameter>"
        ) == "Checking now.")
    }

    @Test("An unclosed fence still counts as inside a fence")
    func trailingUnclosedFenceIsNotAnArtifact() {
        let content = "Example:\n\n```xml\n<tool_call>{\"name\": \"search\"}"
        #expect(ChatMessage.trailingToolCallArtifactProse(in: content) == nil)
    }

    @Test("Tilde and four-backtick fences hide their examples too")
    func trailingNonBacktickFencesAreRespected() {
        // Adversarial review round 2 (codex, blocking): the fence check
        // counted literal ``` runs, so a `~~~` fence (no backticks at all)
        // and a ```` fence (the standard way to show a ``` example nested
        // inside one) both read as UNFENCED — and the example the user asked
        // for was truncated as a leak. Fences are parsed now.
        let tilde = """
        Here is the shape:

        ~~~
        <tool_call>{"name": "search", "arguments": {"q": "x"}}</tool_call>
        ~~~
        """
        #expect(ChatMessage.trailingToolCallArtifactProse(in: tilde) == nil)

        let nested = """
        Here is the shape:

        ````markdown
        ```xml
        <tool_call>{"name": "search", "arguments": {"q": "x"}}</tool_call>
        ```
        ````
        """
        // The inner ``` is shorter than the ```` opener, so it is fence
        // CONTENT and does not close the block — the marker stays covered.
        #expect(ChatMessage.trailingToolCallArtifactProse(in: nested) == nil)
        #expect(!ChatMessage.shouldSuppressToolCallArtifact(
            content: nested, toolCalls: [], finishReason: "stop", toolsRequested: true))
    }

    @Test("A tilde-fenced example does not hide a real envelope after it")
    func trailingEnvelopeAfterTildeFenceIsStillCaught() {
        let content = """
        The shape is:

        ~~~xml
        <tool_call>{"name": "search"}</tool_call>
        ~~~

        Running it now:

        <tool_call> {"name":"search","arguments":{"q":"Statement
        """
        #expect(ChatMessage.trailingToolCallArtifactProse(in: content)?
            .hasSuffix("Running it now:") == true)
    }

    @Test("Fence ranges follow the opener's delimiter and length")
    func fenceRangeParsing() {
        // A closer must match the opener's character AND be at least as long.
        #expect(ChatMessage.fencedRanges(in: "```\nx\n```\nafter").count == 1)
        #expect(ChatMessage.fencedRanges(in: "~~~\nx\n~~~\nafter").count == 1)
        // `~~~` cannot close a ``` fence, so the block runs to the end.
        let mismatched = ChatMessage.fencedRanges(in: "```\nx\n~~~\nafter")
        #expect(mismatched.count == 1)
        #expect(mismatched.first?.upperBound == "```\nx\n~~~\nafter".endIndex)
        // Inline code is not a fence, and neither is a two-character run.
        #expect(ChatMessage.fencedRanges(in: "use `tool_call` here").isEmpty)
        #expect(ChatMessage.fencedRanges(in: "``\nx\n``").isEmpty)
        // Four leading spaces is an indented code block, not a fence opener.
        #expect(ChatMessage.fencedRanges(in: "    ```\nx\n    ```").isEmpty)
    }

    @Test("Any number of fenced examples still cannot hide a real tail")
    func fencedExamplesNeverHideTheTail() {
        // Adversarial review rounds 2-3 (codex): a candidate CAP — shared or
        // per pattern — is spent by the examples and loses the real envelope
        // after them. There is no cap now: a match inside a fence advances
        // the cursor past the whole fenced block, so 200 examples cost 200
        // cheap steps and the tail is still found. Same syntax as the tail on
        // purpose: that is the case a per-pattern cap still got wrong.
        let example = "```xml\n<tool_call>{\"name\": \"s\"}</tool_call>\n```\n"
        let content = "Examples:\n\n"
            + String(repeating: example, count: 200)
            + "\nNow running it:\n\n<tool_call> {\"name\":\"search\",\"arguments\":{\"q\":\"x\"\n"
        #expect(ChatMessage.trailingToolCallArtifactProse(in: content)?
            .hasSuffix("Now running it:") == true)
    }

    @Test("A turn that OPENS with a fenced example is still scanned on")
    func leadingFencedExampleDoesNotEndTheScan() {
        // Adversarial review round 3 (codex, blocking): the start-index guard
        // ran before the fence check, so the leading detector could be handed
        // a turn whose first candidate was merely a fenced example — and the
        // genuine envelope further down rendered raw. The guard now applies to
        // the first UNFENCED candidate.
        let content = """
        ```xml
        <tool_call>{"name": "search", "arguments": {"q": "x"}}</tool_call>
        ```

        That is the shape. Running it:

        <tool_call> {"name":"search","arguments":{"q":"Statement
        """
        #expect(ChatMessage.trailingToolCallArtifactProse(in: content)?
            .hasSuffix("That is the shape. Running it:") == true)
        #expect(ChatMessage.shouldSuppressToolCallArtifact(
            content: content, toolCalls: [], finishReason: "stop", toolsRequested: true))
    }

    @Test("A fence line with an info string never closes a fence")
    func infoStringLineIsFenceContent() {
        // Adversarial review round 3 (codex, blocking): CommonMark gives a
        // CLOSING fence no info string, so a ```` ```swift ```` line inside a
        // ``` block is content. Closing on it left the rest of the example
        // unfenced, and the answer was truncated mid-example.
        let content = """
        Two examples, one block:

        ```
        ```swift
        <tool_call>{"name": "search"}</tool_call>
        ```
        """
        #expect(ChatMessage.trailingToolCallArtifactProse(in: content) == nil)
        let ranges = ChatMessage.fencedRanges(in: content)
        #expect(ranges.count == 1)
        // Trailing whitespace after the run is still a valid closer.
        #expect(ChatMessage.fencedRanges(in: "```\nx\n```   \nafter").count == 1)
    }

    @Test("A leading tab is four columns, not one")
    func tabIndentIsNotAFenceOpener() {
        // Adversarial review round 3 (codex, nit): indentation decides the
        // CommonMark cutoff, and a single tab already reaches column four —
        // an indented code block, not a fence opener.
        #expect(ChatMessage.fencedRanges(in: "\t```\nx\n\t```").isEmpty)
        #expect(ChatMessage.fencedRanges(in: "   ```\nx\n   ```").count == 1)
    }

    @Test("An UNFENCED inline example mid-sentence is not an artifact")
    func trailingInlineExampleIsNotAnArtifact() {
        // Adversarial review round 4 (codex, blocking): the payload gate alone
        // still matched an answer that documents a call inline without a
        // fence, and truncated the sentence at the tag. Envelopes must open a
        // line now — which every real leak shape does, the dogfood repro
        // included.
        for content in [
            "Use <tool_call>{\"name\":\"search\"}</tool_call> to run a search.",
            "Mistral writes [TOOL_CALLS] [{\"name\":\"search\"}] on one line.",
            "The fragment is <function=get_weather>{\"city\":\"NYC\"} in that dialect.",
        ] {
            #expect(ChatMessage.trailingToolCallArtifactProse(in: content) == nil)
            #expect(!ChatMessage.shouldSuppressToolCallArtifact(
                content: content, toolCalls: [], finishReason: "stop", toolsRequested: true))
        }
        // Indented is still "opening a line" — a leak inside a list item.
        #expect(ChatMessage.trailingToolCallArtifactProse(
            in: "Running it:\n  <tool_call> {\"name\":\"search\",\"arguments\":{"
        ) == "Running it:")
    }

    @Test("A raw example the answer goes on to EXPLAIN keeps its explanation")
    func trailingExampleFollowedByProseIsNotAnArtifact() {
        // Adversarial review round 5 (codex, blocking): suppression ran from
        // the marker to the end of the turn without checking that only machine
        // syntax followed, so an answer that showed an unfenced call and then
        // explained it lost the explanation. The envelope has to be the TAIL.
        let content = """
        Here is the call, unfenced:

        <tool_call>{"name":"search","arguments":{"q":"x"}}</tool_call>

        As you can see, the name field picks the tool and arguments carries
        the payload.
        """
        #expect(ChatMessage.trailingToolCallArtifactProse(in: content) == nil)
        #expect(!ChatMessage.shouldSuppressToolCallArtifact(
            content: content, toolCalls: [], finishReason: "stop", toolsRequested: true))
    }

    @Test("Example, prose, then a real envelope keeps everything but the tail")
    func trailingRealEnvelopeAfterRawExampleKeepsTheProse() {
        let content = """
        The shape is:

        <tool_call>{"name":"search","arguments":{"q":"x"}}</tool_call>

        Now running it for real:

        <tool_call> {"name":"search","arguments":{"q":"Statement
        """
        let prose = ChatMessage.trailingToolCallArtifactProse(in: content)
        // The earlier raw example is part of the answer, not the tail.
        #expect(prose?.contains("The shape is:") == true)
        #expect(prose?.hasSuffix("Now running it for real:") == true)
    }

    @Test("A turn ending in a fenced example is never a tail")
    func trailingFenceCloserEndsTheRun() {
        // The closing ``` is not machine syntax for this purpose, so the run
        // never starts and the answer is left alone — which is the same
        // verdict the fence check gives, reached one step earlier.
        let content = """
        Example:

        ```json
        {"name":"search","arguments":{"q":"x"}}
        ```
        """
        #expect(ChatMessage.trailingToolCallArtifactProse(in: content) == nil)
    }

    @Test("A pretty-printed envelope with array values is still a tail")
    func trailingPrettyPrintedEnvelopeIsAnArtifact() {
        // Adversarial review round 6 (codex, blocking): a pretty-printed call
        // puts bare scalars on their own lines, and those reset the terminal
        // run, so the raw envelope stayed visible.
        let content = """
        Let me search for those records:

        <tool_call>
        {
          "name": "search",
          "arguments": {
            "ids": [
              10,
              -2.5,
              true,
              null
            ]
          }
        """
        #expect(ChatMessage.trailingToolCallArtifactProse(in: content)
            == "Let me search for those records:")
        #expect(ChatMessage.shouldSuppressToolCallArtifact(
            content: content, toolCalls: [], finishReason: "stop", toolsRequested: true))
    }

    @Test("Prose that merely ends in a comma or colon is not machine syntax")
    func proseLinesEndingInPunctuationStopTheRun() {
        // The scalar test must stay strict: a looser one moves the boundary
        // EARLIER, and an over-early boundary eats real prose. Here the run
        // has to stop at the last sentence, so the example above it survives.
        let content = """
        The shape is:

        <tool_call>{"name":"search","arguments":{}}</tool_call>

        First, note the name field,
        and second, the arguments object.
        """
        #expect(ChatMessage.trailingToolCallArtifactProse(in: content) == nil)
    }

    @Test("A parsed tool call is never suppressed, however it was written")
    func parsedCallIsNeverSuppressed() {
        // The standing answer to "what about a complete unfenced example?"
        // (raised in adversarial review rounds 5 and 8): gates 1-3 ARE the
        // parser-rejected evidence. An envelope the engine could read comes
        // back as a real `tool_calls` entry, and gate 2 then declines to
        // suppress anything — whatever the content looks like.
        let content = """
        Hermes models emit this:

        <tool_call>{"name":"search","arguments":{"q":"x"}}</tool_call>
        """
        let call = ToolCall(id: "call_1", name: "search", arguments: "{\"q\":\"x\"}")
        #expect(!ChatMessage.shouldSuppressToolCallArtifact(
            content: content, toolCalls: [call], finishReason: "tool_calls",
            toolsRequested: true))
        // And a turn that never advertised tools is out of scope entirely.
        #expect(!ChatMessage.shouldSuppressToolCallArtifact(
            content: content, toolCalls: [], finishReason: "stop",
            toolsRequested: false))
    }

    @Test("Punctuation-led prose after an example stops the terminal run")
    func punctuationLedProseIsNotMachineSyntax() {
        // Adversarial review round 10 (codex, blocking): the run test was
        // "the first character is punctuation", and prose opens with
        // punctuation often enough for that to eat real content — a Markdown
        // link, a reference definition, a quoted sentence. Each opener is
        // matched structurally now.
        let shapes = [
            "[That syntax](https://example.com) is invalid, by the way.",
            "[1]: https://example.com/tool-calling",
            "\"That syntax\" is invalid, by the way.",
            "<- that is what a leaked call looks like.",
        ]
        for tail in shapes {
            let content = """
            Here is the call:

            <tool_call>{"name":"search","arguments":{"q":"x"}}</tool_call>

            \(tail)
            """
            #expect(ChatMessage.trailingToolCallArtifactProse(in: content) == nil)
            #expect(!ChatMessage.shouldSuppressToolCallArtifact(
                content: content, toolCalls: [], finishReason: "stop", toolsRequested: true))
        }
    }

    @Test("Real JSON continuation lines still count as machine syntax")
    func jsonContinuationLinesStayInTheRun() {
        // The other half of round 10: tightening the openers must not drop the
        // shapes an envelope dump actually produces.
        let content = """
        Let me search those records:

        <tool_call>
        {
          "name": "search",
          "arguments": {
            "queries": [
              "first",
              "second"
            ],
            "limit": 10
          }
        }
        """
        #expect(ChatMessage.trailingToolCallArtifactProse(in: content)
            == "Let me search those records:")
    }

    @Test("Prose that opens with a brace or bracket stops the run")
    func braceLedProseIsNotMachineSyntax() {
        // Adversarial review round 11 (codex, blocking): `{ } ] ,` passed
        // unconditionally, so a sentence opening with one — "} closes the
        // object; this is why …" — extended the run and was hidden along with
        // the example above it.
        let shapes = [
            "} closes the object; this is why the call is complete.",
            "] ends the array, and the rest is up to the tool.",
            ", separating the two arguments, is easy to miss.",
            "{ opens it, in case that was not obvious.",
        ]
        for tail in shapes {
            let content = """
            Here is the call:

            <tool_call>{"name":"search","arguments":{"q":"x"}}</tool_call>

            \(tail)
            """
            #expect(ChatMessage.trailingToolCallArtifactProse(in: content) == nil)
        }
    }

    @Test("Multi-call and keyword-valued fragments stay in the run")
    func jsonFragmentLinesStayInTheRun() {
        // The other half of round 11: an unquoted JSON keyword and a
        // `},{`-style multi-call boundary are fragments, not sentences.
        let content = """
        Searching both indexes:

        <tool_call>
        [{"name":"search","arguments":{"q":"x","exact": true}},
        {"name":"search","arguments":{"q":"y","exact": false}}]
        """
        #expect(ChatMessage.trailingToolCallArtifactProse(in: content)
            == "Searching both indexes:")
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
