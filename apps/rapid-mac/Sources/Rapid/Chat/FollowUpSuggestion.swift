import Foundation

/// Builds and reads the one background completion that proposes what to ask
/// next.
///
/// Same split as ``ConversationTitleSuggestion``: the prompt is a function of
/// the turn, the chips are a function of the reply, and neither needs a
/// server to test.
///
/// ## Why every line must end in a question mark
///
/// It is one predicate, and it does the work of a dozen. A model asked for
/// three questions will often add "Here are three follow-up questions:", or
/// number them, or answer in a markdown table, or drift into a paragraph.
/// Requiring a terminal `?` kills the preamble, the table separator row, the
/// stray prose and any code, while compliant output passes by construction
/// because questions are what was asked for. Softer rules — "drop the first
/// line if it ends in a colon", "strip numbering" — each handle one failure
/// and miss the next.
///
/// ## Why fewer than three is nothing
///
/// One or two lonely chips read as a bug rather than as a feature being
/// modest. The whole thing is garnish; rendering nothing is always an
/// acceptable outcome, and it is the outcome for every reply this parser
/// does not fully understand.
enum FollowUpSuggestion {

    /// Chips shown. Also the exact number required — see the type comment.
    static let count = 3

    /// Per-side excerpt budgets. The answer gets more room than the question
    /// because that is where the threads to pull on are.
    static let userExcerptLimit = 600
    static let assistantExcerptLimit = 1200

    /// Longer than this and it is not a chip a reader can scan.
    static let rejectLongerThan = 80

    // MARK: - Prompt

    static let systemPrompt = """
        You suggest what the user might ask next. Reply with exactly 3 lines. \
        Each line is one short question the user could ask next — 3 to 10 \
        words, ending in a question mark. No numbering, no bullets, no quotes, \
        no other text. Each line is written by the user, addressed to the assistant, \
        and never offers help. CRITICAL: write in the same language the user wrote in.
        """

    /// The last exchange, or nil when there isn't one to follow up on.
    /// Caller is expected to have already decided the turn is worth it —
    /// see ``ChatViewModel/settledTurn(_:)``.
    nonisolated static func messages(
        forTurn transcript: [ChatMessage]
    ) -> [ChatMessage]? {
        guard let assistant = transcript.last,
              assistant.role == .assistant,
              !assistant.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              let user = transcript.last(where: { $0.role == .user })
        else { return nil }

        let cue = ReplyScript.isChinese(user.content)
            ? "用户接下来可能会问的三个问题（用用户的口吻写）："
            : "Three questions the user could type next, written as the user:"
        let prompt = """
            User: \(excerpt(user.content, limit: userExcerptLimit))
            Assistant: \(excerpt(assistant.content, limit: assistantExcerptLimit))

            \(cue)
            """
        return [
            ChatMessage(role: .system, content: systemPrompt),
            ChatMessage(role: .user, content: prompt),
        ]
    }

    private nonisolated static func excerpt(_ text: String, limit: Int) -> String {
        let collapsed = ModelReplyText.collapsingWhitespace(text)
        return collapsed.count > limit ? String(collapsed.prefix(limit)) : collapsed
    }

    // MARK: - Parsing

    /// Exactly ``count`` questions, or an empty array.
    ///
    /// `excluding` is the user's last message: a model that suggests the
    /// question just asked is offering to go in a circle.
    /// `excluding` is the user's last message — a model that suggests the
    /// question just asked is offering to go in a circle. `reference` is what
    /// the language check compares against; it defaults to `excluding` but
    /// should be the answer being followed up on, which is longer and more
    /// representative. A last user message of `SwiftUI?` in a Chinese chat is
    /// not evidence that the chat is English.
    nonisolated static func parse(
        _ raw: String,
        excluding lastUserMessage: String = "",
        reference: String? = nil
    ) -> [String] {
        let reference = reference ?? lastUserMessage
        let excluded = ModelReplyText.collapsingWhitespace(lastUserMessage).lowercased()
        var seen = Set<String>()
        var questions: [String] = []

        for line in ModelReplyText.meaningfulLines(raw) {
            var question = ModelReplyText.strippingListMarker(line)
            question = ModelReplyText.strippingWrappers(question)
            question = ModelReplyText.collapsingWhitespace(question)

            // A row of table pipes or rule dashes carries no letters.
            guard question.contains(where: \.isLetter) else { continue }
            guard question.count <= rejectLongerThan else { continue }
            guard let last = question.last, last == "?" || last == "？" else { continue }

            guard !isAssistantVoice(question) else { continue }

            let key = question.lowercased()
            guard key != excluded, seen.insert(key).inserted else { continue }

            questions.append(question)
            if questions.count == count {
                return sharesScript(questions, reference) ? questions : []
            }
        }
        return []
    }

    /// Obvious offers of help written from the assistant's side (#4938).
    ///
    /// A chip is sent as the user's own message, so "Do you need a forecast?"
    /// or "还需要我进行其他修改吗？" reads as nonsense once clicked. The list is
    /// deliberately narrow: only phrases a user would not say to an assistant.
    /// "Should I…", "Can you…", "Can you provide…", "Which do you prefer?",
    /// "还有其他方法吗？" and "Is there anything else I should know?" are all
    /// ordinary user questions and must pass. A false rejection only costs the
    /// chip row, because fewer than three questions already means no chips.
    nonisolated static func isAssistantVoice(_ question: String) -> Bool {
        let lowered = question.lowercased()
        return assistantOffers.contains { lowered.contains($0) }
    }

    private static let assistantOffers = [
        "do you need", "do you want", "would you like", "will you like",
        "can i help", "may i help", "how can i help", "can i assist",
        "want me to", "like me to", "need me to",
        "anything else you need", "anything else you'd like", "anything else i can",
        "需要我", "我可以帮", "我能帮", "我来帮", "帮你", "帮您",
        "你需要", "您需要", "你想要", "您想要", "你想让", "您想让", "你希望",
        "您希望", "需要帮助",
    ]

    /// Do the suggestions and the conversation use the same writing system?
    ///
    /// Prompting alone does not settle this. Measured over 20 turns on the
    /// bundled 1.2B model, the instruction to match the user's language was
    /// obeyed 10 times; adding a `CRITICAL:` clause took it to 12. Both
    /// numbers mean a reader writing Chinese regularly gets three English
    /// chips, which is worse than no chips — it reads as the app not having
    /// noticed what they said.
    ///
    /// So the last word is a deterministic check rather than a better
    /// sentence. A mismatch throws the whole set away, which is the failure
    /// this feature already uses everywhere else.
    ///
    /// Scoped to CJK-versus-not because that is the mismatch that was
    /// measured. A Cyrillic or Arabic conversation may well have the same
    /// problem, but guessing at a failure mode nobody has watched would be
    /// writing a rule for a bug that has not been seen.
    nonisolated static func sharesScript(_ questions: [String], _ conversation: String) -> Bool {
        // An empty reference tells us nothing, so it cannot veto.
        guard conversation.contains(where: \.isLetter) else { return true }
        let wanted = ReplyScript.usesCJK(conversation)
        // A majority, not "any scalar anywhere". The first version joined the
        // questions and asked whether the result contained a CJK character, so
        // one Chinese word inside one otherwise-English suggestion threw all
        // three away.
        let matching = questions.filter { ReplyScript.usesCJK($0) == wanted }.count
        return matching * 2 > questions.count
    }
}
