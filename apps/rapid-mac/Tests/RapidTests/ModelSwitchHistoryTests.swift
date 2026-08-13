import Foundation
import Testing
@testable import Rapid

/// v0.4.35 regression pin for the model-switch silent-failure bug.
///
/// Symptom the user reported (2026-06-10): switching the picker model
/// mid-session → new model returns a blank assistant bubble with no
/// error, no spinner, nothing. The bug was a two-step interaction:
///
///   1. The old model's last assistant turn was sometimes left with
///      ``content = ""`` (mid-stream server kill, hybrid-thinking
///      reasoning-only emission, or model returning [DONE] with
///      empty content).
///   2. That empty assistant row went out as-is to the *new* model's
///      chat template. Qwen / GLM / Hermes templates interpret an
///      empty assistant slot as "the model already finished its
///      reply" and immediately emit EOS — the new model produced no
///      text, the bubble landed blank, and the user assumed the new
///      model was broken.
///
/// Two-pronged defense, both pinned here:
///   * ``filterEmptyAssistantsForWire`` drops the empty rows from the
///     wire body (UI still shows them; this is wire-only).
///   * ``zeroContentFailureMessage`` flags any terminal that produced
///     no prose and no tool calls as a soft failure with actionable
///     copy — so even if the underlying cause changes shape later
///     the user gets a path forward instead of silence.
@MainActor
@Suite("v0.4.35 model-switch silent-failure defense")
struct ModelSwitchHistoryTests {

    // MARK: - filterEmptyAssistantsForWire

    @Test("Empty-prose assistant turn is stripped from wire history")
    func dropsEmptyAssistant() {
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: "hi"),
            ChatMessage(role: .assistant, content: "", status: .failed),
            ChatMessage(role: .user, content: "still there?"),
        ]
        let filtered = ChatViewModel.filterEmptyAssistantsForWire(history)
        #expect(filtered.count == 2)
        #expect(filtered.allSatisfy { $0.role != .assistant || !$0.content.isEmpty })
    }

    @Test("Whitespace-only assistant turn is also stripped — \"\\n\\n\" is just as toxic to chat templates")
    func dropsWhitespaceOnlyAssistant() {
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: "hi"),
            ChatMessage(role: .assistant, content: " \n  \t\n", status: .complete),
        ]
        let filtered = ChatViewModel.filterEmptyAssistantsForWire(history)
        #expect(filtered.count == 1)
        #expect(filtered.first?.role == .user)
    }

    @Test("Stop before first token drops the unanswered user/assistant pair from wire history")
    func dropsZeroTokenCancelledTurn() {
        var stopped = ChatMessage(role: .assistant, content: "", status: .complete)
        stopped.errorMessage = "Stopped."
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: "Earlier answered question"),
            ChatMessage(role: .assistant, content: "Earlier answer", status: .complete),
            ChatMessage(role: .user, content: "Write 500 numbered lines"),
            stopped,
            ChatMessage(role: .user, content: "What is 2 + 2?"),
        ]

        let filtered = ChatViewModel.filterEmptyAssistantsForWire(history)

        #expect(filtered.map(\.content) == [
            "Earlier answered question", "Earlier answer", "What is 2 + 2?",
        ])
    }

    @Test("A partially emitted stopped answer remains valid conversation history")
    func preservesPartialCancelledTurn() {
        var stopped = ChatMessage(role: .assistant, content: "1\n2\n3", status: .complete)
        stopped.errorMessage = "Stopped."
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: "Write 500 numbered lines"),
            stopped,
            ChatMessage(role: .user, content: "Continue"),
        ]

        let filtered = ChatViewModel.filterEmptyAssistantsForWire(history)

        #expect(filtered.count == 3)
        #expect(filtered[1].content == "1\n2\n3")
    }

    @Test("Tool-call assistant (empty prose + populated tool_calls) is preserved — load-bearing for the tool loop")
    func preservesToolCallAssistant() {
        var toolAsst = ChatMessage(role: .assistant, content: "", status: .complete)
        toolAsst.toolCalls = [
            ToolCall(id: "c1", name: "read_file", arguments: "{}")
        ]
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: "read it"),
            toolAsst,
            ChatMessage(role: .tool, content: "{\"ok\":true}"),
        ]
        let filtered = ChatViewModel.filterEmptyAssistantsForWire(history)
        // All three must survive — dropping the tool_calls assistant
        // would leave the .tool message orphaned and the model would
        // refuse the request.
        #expect(filtered.count == 3)
    }

    @Test("Non-empty assistant turns pass through untouched")
    func preservesRealAssistantContent() {
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: "Q1"),
            ChatMessage(role: .assistant, content: "A1", status: .complete),
            ChatMessage(role: .user, content: "Q2"),
            ChatMessage(role: .assistant, content: "A2", status: .complete),
        ]
        let filtered = ChatViewModel.filterEmptyAssistantsForWire(history)
        #expect(filtered.count == 4)
        #expect(filtered.compactMap { $0.role == .assistant ? $0.content : nil } == ["A1", "A2"])
    }

    @Test("User and system rows are never filtered — only assistants are inspected")
    func leavesUserAndSystemAlone() {
        // An empty user / system row is structurally valid and the
        // filter has no business interpreting it.
        let history: [ChatMessage] = [
            ChatMessage(role: .system, content: ""),
            ChatMessage(role: .user, content: ""),
            ChatMessage(role: .assistant, content: "hi"),
        ]
        let filtered = ChatViewModel.filterEmptyAssistantsForWire(history)
        #expect(filtered.count == 3)
    }

    // MARK: - zeroContentFailureMessage

    @Test("Non-empty content returns nil — real completions don't get flagged")
    func realCompletionIsNotAFailure() {
        let msg = ChatViewModel.zeroContentFailureMessage(
            proseContent: "The capital of France is Paris.",
            toolCalls: nil,
            finishReason: "stop"
        )
        #expect(msg == nil)
    }

    @Test("Tool-call terminal with empty prose is NOT a failure — that's just a tool-call turn")
    func toolCallTerminalIsNotAFailure() {
        let calls = [ToolCall(id: "c1", name: "read_file", arguments: "{}")]
        let msg = ChatViewModel.zeroContentFailureMessage(
            proseContent: "",
            toolCalls: calls,
            finishReason: "tool_calls"
        )
        #expect(msg == nil)
    }

    @Test("finish_reason: stop + empty content is flagged with the model-switch hint")
    func emptyStopIsFlagged() {
        let msg = ChatViewModel.zeroContentFailureMessage(
            proseContent: "",
            toolCalls: nil,
            finishReason: "stop"
        )
        #expect(msg != nil)
        #expect(msg?.contains("switching models") == true,
                "User-facing copy must mention the switching-models trigger so the next step is obvious")
    }

    @Test("finish_reason: nil + empty content is flagged — non-conforming server case")
    func emptyNilReasonIsFlagged() {
        let msg = ChatViewModel.zeroContentFailureMessage(
            proseContent: "",
            toolCalls: nil,
            finishReason: nil
        )
        #expect(msg != nil)
    }

    @Test("finish_reason: length + empty content keeps the original truncation explainer")
    func lengthEmptyKeepsOriginalCopy() {
        // Pre-v0.4.35 covered this case; we MUST NOT regress its copy.
        let msg = ChatViewModel.zeroContentFailureMessage(
            proseContent: "",
            toolCalls: nil,
            finishReason: "length"
        )
        #expect(msg?.contains("Max Tokens") == true)
    }

    @Test("Whitespace-only content is treated as empty — \"   \\n\" is not a real reply")
    func whitespaceOnlyContentIsFlagged() {
        let msg = ChatViewModel.zeroContentFailureMessage(
            proseContent: "   \n\t",
            toolCalls: nil,
            finishReason: "stop"
        )
        #expect(msg != nil)
    }

    @Test("#161 length + thinking ON points the user at the Show reasoning toggle, not at Max Tokens alone")
    func lengthEmptyWithThinkingOnPointsAtToggle() {
        let msg = ChatViewModel.zeroContentFailureMessage(
            proseContent: "",
            toolCalls: nil,
            finishReason: "length",
            thinkingEnabled: true
        )
        #expect(msg != nil)
        // The default copy ("Hit the Max Tokens limit before any output")
        // is wrong when reasoning burned the budget — the user is meant
        // to see the toggle name in the message, otherwise the next step
        // is unclear.
        #expect(msg?.contains("Show reasoning") == true,
                "Copy must surface the toggle name when thinkingEnabled=true; got: \(msg ?? "nil")")
    }
}
