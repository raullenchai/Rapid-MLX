import Foundation

/// Builds and reads the one background completion that names a conversation.
///
/// Pure and `nonisolated` on both sides: the prompt is a function of the
/// transcript and the title is a function of the reply, so the interesting
/// half of this feature is testable without a server, a view model, or an
/// actor hop. ``ChatViewModel`` owns the request; this owns the words.
///
/// The parser is deliberately lopsided — generous about what it accepts and
/// strict about what it returns. A small local model asked for a title will
/// wrap it in quotes, prefix it with `Title:`, fence it, number it, or write
/// a paragraph explaining its choice. The first four are recoverable and are
/// stripped. The last is not, and is rejected: truncating prose to the
/// sidebar's 42 characters reads worse than the opening words of the user's
/// own question, which is what the row already shows.
///
/// Every rejection returns nil, which ``ChatViewModel/applyGeneratedTitle(_:to:)``
/// turns into "leave the derived title alone". There is no retry, no error
/// state, and nothing the reader can see — the same contract
/// ``ServerProfileFetcher`` states for its own background fetch.
enum ConversationTitleSuggestion {

    /// How much of each side to send. A title needs the topic, not the
    /// transcript, and a short prefill is most of what keeps this call cheap.
    static let excerptLimit = 400

    /// Longer than this and the reply is prose, not a title.
    static let rejectLongerThan = 80

    // MARK: - Prompt

    static let systemPrompt = """
        You name chat conversations. Reply with ONLY the title: 3 to 6 words, \
        no quotes, no trailing punctuation, no markdown, no explanation. \
        Write the title in the same language the conversation is in.
        """

    /// The first user turn and the answer it drew, or nil when the transcript
    /// does not yet hold both.
    nonisolated static func messages(
        forFirstExchange transcript: [ChatMessage]
    ) -> [ChatMessage]? {
        guard let user = transcript.first(where: { $0.role == .user }),
              let assistant = transcript.first(where: {
                  $0.role == .assistant
                      && !$0.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
              })
        else { return nil }

        let prompt = """
            User: \(excerpt(user.content))
            Assistant: \(excerpt(assistant.content))

            Title:
            """
        return [
            ChatMessage(role: .system, content: systemPrompt),
            ChatMessage(role: .user, content: prompt),
        ]
    }

    private nonisolated static func excerpt(_ text: String) -> String {
        let collapsed = ModelReplyText.collapsingWhitespace(text)
        return collapsed.count > excerptLimit
            ? String(collapsed.prefix(excerptLimit))
            : collapsed
    }

    // MARK: - Parsing

    /// The title to show, or nil to keep the derived one.
    nonisolated static func parse(_ raw: String) -> String? {
        guard let line = ModelReplyText.firstMeaningfulLine(raw) else { return nil }
        // Measured before stripping: a paragraph that happens to open with a
        // quote is still a paragraph, and stripping first would let it
        // through at 78 characters.
        guard line.count <= rejectLongerThan else { return nil }

        var title = ModelReplyText.strippingLabel(line, labels: ["title", "标题", "titre", "título"])
        title = ModelReplyText.strippingListMarker(title)
        title = ModelReplyText.strippingWrappers(title)
        title = ModelReplyText.strippingTrailingStop(title)
        title = ModelReplyText.collapsingWhitespace(title)

        guard title.count >= 2 else { return nil }
        guard !ModelReplyText.looksLikeRefusal(title) else { return nil }
        guard title != "New chat" else { return nil }
        // A markdown table row is a row, whatever else it is.
        guard !title.hasPrefix("|") else { return nil }
        // A trailing colon means the model wrote the lead-in to a list and
        // stopped — "Three things, in order:" names nothing.
        guard let last = title.last, last != ":", last != "：" else { return nil }
        // More than one sentence is prose. A title is a phrase; the moment a
        // full stop has text after it we are reading the answer, not a name
        // for it.
        guard !isMultiSentence(title) else { return nil }

        return ConversationStore.capped(title)
    }

    /// True when the line reads as more than one sentence.
    ///
    /// Crude on purpose: a stop with text after it. That also catches
    /// "Dr. Who and the daleks" and "e.g. modular arithmetic", and those are
    /// accepted losses — the two costs are not symmetric. A false reject
    /// keeps the derived title, which is the opening words of the user's own
    /// question and is what the row shows today. A false accept puts a
    /// paragraph in the sidebar. Distinguishing an abbreviation from a
    /// sentence boundary needs a list of abbreviations per language, which is
    /// a great deal of machinery to rescue a title nobody has lost yet.
    private nonisolated static func isMultiSentence(_ title: String) -> Bool {
        if title.contains(". ") { return true }
        // Full-width stops are unambiguous — no abbreviations use them — so
        // only a trailing one is fine.
        return title.dropLast().contains { $0 == "。" || $0 == "！" }
    }
}
