import Foundation
import Testing
@testable import Rapid

/// The anti-confabulation guidance used to be prepended to the system row on
/// the round that carried a tool result and stripped again on the next turn.
/// The system row is the head of every prompt and the engine's prefix cache
/// reuses a stored prompt only up to the first differing token, so one web
/// search cost the whole conversation two cold prefills (0.14.1, Qwen3.8-27B,
/// 5.7k-token conversation: ~20 s each against 1.5 s for an append-only turn).
///
/// These tests pin where the guidance lives now — the wire-only trailer of
/// each user row whose turn holds a tool result — and the property that
/// placement buys: the prompt is append-only. The system row and every
/// completed turn are byte-identical between the tool round and the rounds
/// around it, and a row that earned the guidance keeps it on later turns.
@Suite("Tool guidance rides the user rows")
struct ToolGuidanceTrailerTests {

    private static let date = "[CURRENT DATE]\nToday is Friday, 12 September 2026."

    private static func assemble(_ body: [ChatMessage], toolsAdvertised: Bool = true) -> [ChatMessage] {
        var history = ChatViewModel.addingInstructionLayers(
            to: body,
            dateContext: date,
            global: "Answer briefly.",
            conversation: ""
        )
        history = ChatViewModel.stampingClockContext(on: history)
        return ChatViewModel.stampingToolGuidance(on: history, toolsAdvertised: toolsAdvertised)
    }

    private static let asked = Date(timeIntervalSince1970: 1_789_300_000)

    private static let roundOne: [ChatMessage] = {
        [
            ChatMessage(role: .user, content: "Name three colors.", createdAt: asked),
            ChatMessage(role: .assistant, content: "Red, blue, and green.", createdAt: asked),
            ChatMessage(role: .user, content: "weather in Tokyo?", createdAt: asked),
        ]
    }()

    private static let roundTwo: [ChatMessage] = {
        roundOne + [
            ChatMessage(
                role: .assistant,
                toolCalls: [ToolCall(id: "w1", name: "weather", arguments: "{\"city\":\"Tokyo\"}")],
                createdAt: asked
            ),
            ChatMessage(role: .tool, content: "{\"temp_c\": 29.2}", toolCallID: "w1", createdAt: asked),
        ]
    }()

    private static let nextTurn: [ChatMessage] = {
        roundTwo + [
            ChatMessage(role: .assistant, content: "It's 29.2°C in Tokyo.", createdAt: asked),
            ChatMessage(role: .user, content: "what is the capital of France?", createdAt: asked),
        ]
    }()

    @Test("The guidance is a trailer on the newest user row, behind the clock, never in the system row")
    func guidanceLandsOnTheNewestUserRow() {
        let wire = Self.assemble(Self.roundTwo)
        let system = wire[0]
        #expect(system.role == .system)
        #expect(!system.content.contains(ChatViewModel.toolGuidance))

        let newestUser = wire[3]
        #expect(newestUser.role == .user)
        #expect(newestUser.content == "weather in Tokyo?", "the transcript row stays prose-only")
        let suffix = newestUser.wireSuffix ?? ""
        #expect(suffix.hasPrefix("[MESSAGE SENT]"), "the clock trailer still comes first")
        #expect(suffix.hasSuffix(ChatViewModel.toolGuidance))
        #expect(newestUser.modelContent.hasSuffix(ChatViewModel.toolGuidance))

        // No other row carries it: not the earlier user row, not the tool row.
        #expect(wire[1].wireSuffix?.contains(ChatViewModel.toolGuidance) == false)
        #expect(wire[5].role == .tool)
        #expect(wire[5].wireSuffix == nil)
    }

    @Test("The prompt is append-only across the tool round and the turn after it")
    func headOfThePromptIsStableAcrossTheToolRound() {
        let before = Self.assemble(Self.roundOne)
        let during = Self.assemble(Self.roundTwo)
        let after = Self.assemble(Self.nextTurn)

        // Round one (no tool result yet) and round two (tool result in play)
        // differ ONLY from the newest user row on; the system row and the
        // earlier turn are the same bytes on the wire.
        #expect(during[0].content == before[0].content)
        #expect(during[1] == before[1])
        #expect(during[2] == before[2])
        #expect(during[3].modelContent != before[3].modelContent)
        #expect(during[3].content == before[3].content)

        // The next turn extends round two byte-for-byte: the row that earned
        // the guidance keeps it (its turn still holds the tool result), so
        // the engine resumes from the end of round two's prompt instead of
        // re-prefilling from the row where a one-off trailer would have
        // vanished. Only the brand-new user row is unstamped.
        #expect(after[0].content == during[0].content)
        for index in 1..<during.count {
            #expect(after[index] == during[index], "row \(index) must not change once its turn is complete")
        }
        #expect(after[3].wireSuffix?.hasSuffix(ChatViewModel.toolGuidance) == true)
        #expect(after.last?.role == .user)
        #expect(after.last?.wireSuffix?.contains(ChatViewModel.toolGuidance) == false)
    }

    @Test("No guidance without advertised tools or without a tool result this turn (#1549)")
    func gateIsUnchanged() {
        let noTools = Self.assemble(Self.roundTwo, toolsAdvertised: false)
        #expect(!noTools.contains { $0.wireSuffix?.contains(ChatViewModel.toolGuidance) == true })
        let noResult = Self.assemble(Self.roundOne)
        #expect(!noResult.contains { $0.wireSuffix?.contains(ChatViewModel.toolGuidance) == true })
    }

    @Test("A trim that drops the tool result drops the guidance with it")
    func trimmedEvidenceTakesTheInstruction() {
        // The caller stamps the TRIMMED history. Model the trim that keeps the
        // current turn's user row but elides its tool result: the guidance
        // must not claim a tool result that is no longer on the wire.
        var trimmed = Self.roundTwo
        trimmed.removeLast(2)
        let wire = Self.assemble(trimmed)
        #expect(!wire.contains { $0.wireSuffix?.contains(ChatViewModel.toolGuidance) == true })
    }

    @Test("On the wire, each round's body extends the previous one")
    func wireBodySharesThePrefixUpToTheNewestUserRow() async throws {
        func request(_ messages: [ChatMessage]) -> ChatStreamClient.Request {
            ChatStreamClient.Request(alias: "test-model", messages: messages, tools: nil, supportsImageInput: false)
        }
        let oneData = try #require(await WireBodyCaptureProtocol.capture(request(Self.assemble(Self.roundOne))))
        let twoData = try #require(await WireBodyCaptureProtocol.capture(request(Self.assemble(Self.roundTwo))))
        let one = try #require(String(data: oneData, encoding: .utf8))
        let two = try #require(String(data: twoData, encoding: .utf8))
        // The shared prefix must reach past the system row and the first
        // exchange, into the newest user row's own object.
        let marker = "\"content\":\"weather in Tokyo?"
        let oneCut = try #require(one.range(of: marker)?.upperBound)
        let head = String(one[..<oneCut])
        #expect(two.hasPrefix(head),
                "the system row and every earlier row must serialize identically whether or not the round carries a tool result")
        #expect(two.contains(marker))
        #expect(!two.contains("\"content\":\"You have access to tools"),
                "the guidance must not become a system/user row of its own")

        // And the next turn's body carries round two's complete message
        // sequence as its prefix — the stamped user row, its tool rows, all
        // of it — so the engine's exact/prefix lookup sees round two's stored
        // prompt as a prefix of the next turn. Compared as decoded rows, not
        // as a string cut inside a row.
        let threeData = try #require(await WireBodyCaptureProtocol.capture(request(Self.assemble(Self.nextTurn))))
        let twoRows = try Self.wireMessages(twoData)
        let threeRows = try Self.wireMessages(threeData)
        #expect(threeRows.count == twoRows.count + 2)
        #expect(Array(threeRows.prefix(twoRows.count)) == twoRows,
                "every row of round two must serialize identically on the next turn")
    }

    /// The `messages` array of a wire body, each row re-serialized with sorted
    /// keys so rows compare by content.
    private static func wireMessages(_ body: Data) throws -> [String] {
        let object = try #require(try JSONSerialization.jsonObject(with: body) as? [String: Any])
        let rows = try #require(object["messages"] as? [[String: Any]])
        return try rows.map { row in
            let data = try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys])
            return try #require(String(data: data, encoding: .utf8))
        }
    }

    @Test("Stamping is a rebuild: idempotent, and withdrawn everywhere when the tools are")
    func stampIsRebuiltNotAppended() {
        let once = ChatViewModel.stampingToolGuidance(on: Self.assemble(Self.nextTurn), toolsAdvertised: true)
        let twice = ChatViewModel.stampingToolGuidance(on: once, toolsAdvertised: true)
        #expect(twice == once)
        let stampedRow = try? #require(once.first { $0.wireSuffix?.contains(ChatViewModel.toolGuidance) == true })
        #expect(stampedRow?.wireSuffix?.components(separatedBy: ChatViewModel.toolGuidance).count == 2,
                "exactly one copy of the guidance")

        // A history that already carries the guidance (the pre-trim pass) and
        // then loses its tools (final synthesis round, tool toggled off) must
        // not keep a row asserting tool access.
        let withdrawn = ChatViewModel.stampingToolGuidance(on: once, toolsAdvertised: false)
        #expect(!withdrawn.contains { $0.wireSuffix?.contains(ChatViewModel.toolGuidance) == true })
        let pristine = Self.assemble(Self.nextTurn, toolsAdvertised: false)
        #expect(withdrawn.map(\.role) == pristine.map(\.role))
        #expect(withdrawn.map(\.modelContent) == pristine.map(\.modelContent),
                "withdrawing the guidance restores every row's wire bytes")
        // The clock trailer the guidance was joined behind survives intact.
        #expect(withdrawn[3].wireSuffix?.hasPrefix("[MESSAGE SENT]") == true)
        #expect(withdrawn[3].wireSuffix?.hasSuffix("\n\n") == false)
    }

    @Test("The guidance says which message it binds, so an old stamp reads as history")
    func guidanceIsScopedToItsOwnMessage() {
        #expect(ChatViewModel.toolGuidance.contains("bind the answer to THIS message"))
        #expect(ChatViewModel.toolGuidance.contains("A later message that has no tool result of its own is answered normally"))
    }

    @Test("Stripping removes only the terminal component the stamp joined")
    func stripIsExactAndTerminal() {
        let g = ChatViewModel.toolGuidance
        #expect(ChatViewModel.strippingToolGuidance(from: nil) == nil)
        #expect(ChatViewModel.strippingToolGuidance(from: g) == nil)
        #expect(ChatViewModel.strippingToolGuidance(from: "[MESSAGE SENT]\nnow.\n\n" + g) == "[MESSAGE SENT]\nnow.")
        // A trailer that merely quotes the guidance somewhere else is not ours.
        let quoted = "note: " + g + "\n\n[MESSAGE SENT]\nnow."
        #expect(ChatViewModel.strippingToolGuidance(from: quoted) == quoted)
        #expect(ChatViewModel.strippingToolGuidance(from: "   ") == nil)
    }

    @Test("The guidance counts toward the context-window trim budget")
    @MainActor
    func guidanceIsInsideTheTrimBudget() {
        // Same conversation, stamped and unstamped. A budget that fits the
        // unstamped body exactly must NOT fit the stamped one: the guidance
        // rows are real tokens on the wire and the trim has to see them.
        let bare = Self.assemble(Self.nextTurn, toolsAdvertised: false)
        let stamped = Self.assemble(Self.nextTurn)
        #expect(stamped.contains { $0.wireSuffix?.contains(ChatViewModel.toolGuidance) == true })
        // Same accounting as the trim: prose plus tool-call arguments.
        let cost: (ChatMessage) -> Int = { row in
            let toolArgs = (row.toolCalls ?? []).map(\.function.arguments).joined()
            return max(1, TokenEstimate.tokens(in: row.modelContent)
                + (toolArgs.isEmpty ? 0 : TokenEstimate.tokens(in: toolArgs)))
        }
        let bareTokens = bare.reduce(0) { $0 + cost($1) }
        let window = Int((Double(bareTokens) / 0.75).rounded(.up)) + 2
        #expect(ChatViewModel.trimMessagesForContextWindow(bare, contextWindow: window).count == bare.count)
        let trimmed = ChatViewModel.trimMessagesForContextWindow(stamped, contextWindow: window)
        #expect(trimmed.count < stamped.count,
                "a window sized for the prose alone must trim once the guidance rides the rows")
    }
}
