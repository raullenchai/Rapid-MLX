import Foundation
import Testing
@testable import Rapid

@Suite("Base assistant instruction")
struct BaseAssistantInstructionTests {
    private static let dateContext = "[CURRENT DATE]\nToday is Tuesday, August 25, 2026."

    @Test("Base identity and date form exactly one system row without user instructions")
    func baseAndDateWithoutUserInstructions() throws {
        let user = ChatMessage(role: .user, content: "Hello", status: .complete)
        let result = ChatViewModel.addingInstructionLayers(
            to: [user],
            baseInstruction: ChatViewModel.baseAssistantInstruction,
            dateContext: Self.dateContext,
            global: "",
            conversation: ""
        )
        #expect(result.filter { $0.role == .system }.count == 1)
        let system = try #require(result.first?.content)
        #expect(system == ChatViewModel.baseAssistantInstruction + "\n\n" + Self.dateContext)
        #expect(result.dropFirst() == [user])
    }

    @Test("User instructions follow the base and date verbatim in one system row")
    func userInstructionsFollowBaseAndDate() throws {
        let global = "Reply in French.\nPreserve this preference."
        let conversation = "Reply in Japanese.\nUse short paragraphs."
        let result = ChatViewModel.addingInstructionLayers(
            to: [ChatMessage(role: .user, content: "Hello", status: .complete)],
            baseInstruction: ChatViewModel.baseAssistantInstruction,
            dateContext: Self.dateContext,
            global: global,
            conversation: conversation
        )
        #expect(result.filter { $0.role == .system }.count == 1)
        let system = try #require(result.first?.content)
        #expect(system.hasPrefix(ChatViewModel.baseAssistantInstruction + "\n\n" + Self.dateContext))
        let date = try #require(system.range(of: Self.dateContext))
        let globalHeader = try #require(system.range(of: "[GLOBAL USER INSTRUCTIONS]"))
        let conversationHeader = try #require(system.range(of: "[CONVERSATION INSTRUCTIONS - HIGHEST USER PRIORITY]"))
        #expect(date.upperBound < globalHeader.lowerBound)
        #expect(globalHeader.upperBound < conversationHeader.lowerBound)
        #expect(system.contains(global))
        #expect(system.contains(conversation))
    }

    @Test("System bytes stay identical across turns and request instants on the same day")
    func stableAcrossTurnsAndRequestInstants() throws {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = try #require(TimeZone(secondsFromGMT: 0))
        let morning = try #require(calendar.date(from: DateComponents(
            year: 2026, month: 8, day: 25, hour: 8, minute: 10
        )))
        let evening = morning.addingTimeInterval(10 * 60 * 60)
        let history = [
            ChatMessage(role: .user, content: "First", status: .complete, createdAt: morning),
            ChatMessage(role: .assistant, content: "First answer", status: .complete),
            ChatMessage(role: .user, content: "Second", status: .complete),
            ChatMessage(role: .assistant, content: "Second answer", status: .complete),
            ChatMessage(role: .user, content: "Third", status: .complete, createdAt: evening),
        ]
        var systemRows: [String] = []
        for turnCount in [1, 3] {
            for instant in [morning, evening] {
                let result = ChatViewModel.addingInstructionLayers(
                    to: Array(history.prefix(turnCount * 2 - 1)),
                    baseInstruction: ChatViewModel.baseAssistantInstruction,
                    dateContext: ChatViewModel.currentDateContext(now: instant, calendar: calendar),
                    global: "",
                    conversation: ""
                )
                #expect(result.filter { $0.role == .system }.count == 1)
                systemRows.append(try #require(result.first?.content))
            }
        }
        let first = try #require(systemRows.first)
        #expect(first.hasPrefix(ChatViewModel.baseAssistantInstruction))
        #expect(systemRows.allSatisfy { Array($0.utf8) == Array(first.utf8) })
    }

    @Test("Base precedes an existing system row and preserves the remaining layer order")
    func mergesExistingSystemRow() throws {
        let user = ChatMessage(role: .user, content: "Hello", status: .complete)
        let result = ChatViewModel.addingInstructionLayers(
            to: [ChatMessage(role: .system, content: "Existing system", status: .complete), user],
            baseInstruction: ChatViewModel.baseAssistantInstruction,
            dateContext: Self.dateContext,
            memoryContext: "Memory context",
            global: "",
            conversation: ""
        )
        #expect(result.filter { $0.role == .system }.count == 1)
        let system = try #require(result.first?.content)
        #expect(system == ChatViewModel.baseAssistantInstruction + "\n\n" + Self.dateContext
            + "\n\nExisting system\n\nMemory context")
        #expect(result.dropFirst() == [user])
    }

    @Test("Effective preview forwards an explicit base override or nil")
    func previewForwardsBaseInstruction() {
        for base in [String?.none, "Alternate identity"] {
            let preview = ChatViewModel.effectiveSystemPrompt(
                baseInstruction: base,
                dateContext: Self.dateContext,
                global: "",
                conversation: ""
            )
            let wire = ChatViewModel.addingInstructionLayers(
                to: [],
                baseInstruction: base,
                dateContext: Self.dateContext,
                global: "",
                conversation: ""
            )
            #expect(preview == wire.first?.content)
        }
    }

    @Test("Base uses the same normalization as other instruction layers")
    func normalizesBaseInstruction() {
        let normalized = ChatViewModel.addingInstructionLayers(
            to: [], baseInstruction: " \n Identity \n ", global: "", conversation: ""
        )
        #expect(normalized.first?.content == "Identity")
        let blank = ChatViewModel.addingInstructionLayers(
            to: [], baseInstruction: " \n ", global: "", conversation: ""
        )
        #expect(blank.isEmpty)
    }

    @Test("Base identity stays short and contains no digits or per-turn content")
    func baseStaysShortAndStatic() {
        let base = ChatViewModel.baseAssistantInstruction
        #expect(base.count < 400)
        #expect(!base.contains(where: { $0.isNumber }))
    }
}
