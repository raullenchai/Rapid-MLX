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
        words, written in the user's voice, ending in a question mark, in the \
        same language as the conversation. No numbering, no bullets, no \
        quotes, no other text.
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

        let prompt = """
            User: \(excerpt(user.content, limit: userExcerptLimit))
            Assistant: \(excerpt(assistant.content, limit: assistantExcerptLimit))

            Three follow-up questions:
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
    nonisolated static func parse(_ raw: String, excluding lastUserMessage: String = "") -> [String] {
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

            let key = question.lowercased()
            guard key != excluded, seen.insert(key).inserted else { continue }

            questions.append(question)
            if questions.count == count { return questions }
        }
        return []
    }
}
