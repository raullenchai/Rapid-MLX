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
        You are a conversation titler. Given a chat, you output a short noun \
        phrase naming what it is about, and nothing else. Use the language of \
        the chat.
        """

    /// The first user turn and the answer it drew, or nil when the transcript
    /// does not yet hold both.
    ///
    /// ## Why the instruction reads the way it does
    ///
    /// Measured against the bundled 1.2B model, which is the one most readers
    /// will see first. An earlier wording — "Reply with ONLY the title: 3 to 6
    /// words, no quotes…" — returned, verbatim and repeatably, `3 to 6 words`
    /// and `Chat conversations`. A model this size will lift a noun phrase out
    /// of the instruction and hand it back, so the instruction must not
    /// contain one that could pass for an answer. Hence a verb ("naming what
    /// it is about") where the old one had a noun ("the title"), and the
    /// constraints moved off the sentence that says what to output.
    ///
    /// Same exchanges after the rewrite, three samples each: a correct
    /// Chinese noun phrase for the Chinese one, `Function reverse` for the
    /// code one, `Deploy command` for the deployment one — against `3`, `3`,
    /// `3` before it.
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
            Chat:
            User: \(excerpt(user.content))
            Assistant: \(excerpt(assistant.content))

            Name it in 2 to 5 words:
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

        var title = ModelReplyText.strippingLabel(line, labels: ["title"])
        title = ModelReplyText.strippingListMarker(title)
        title = ModelReplyText.strippingWrappers(title)
        title = ModelReplyText.strippingTrailingStop(title)
        title = ModelReplyText.collapsingWhitespace(title)

        guard title.count >= 2 else { return nil }
        guard !ModelReplyText.looksLikeRefusal(title) else { return nil }
        guard !isGenericNonTitle(title) else { return nil }
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

    /// Words that name the *thing being titled* rather than what it is about.
    ///
    /// A model with nothing to work from — a bare greeting in any language —
    /// reaches for these, and they are the one output worse than no title at
    /// all: "Chat" in a list of chats says less than the reader's own "hi"
    /// does. Measured on the bundled 1.2B model, a bare greeting returns
    /// `Chat` on every sample, in English, whatever language it was greeted
    /// in — which is why the list needs no translations.
    ///
    /// Note what is *not* here: "Greeting". Naming a greeting as a greeting is
    /// a real answer about a real (if slight) exchange, and a larger model
    /// gives exactly that. This list is only for words that describe the
    /// container.
    private static let genericNonTitles: Set<String> = [
        "chat", "chats", "chat topic", "chat conversations", "conversation",
        "conversations", "topic", "topics", "exchange", "discussion", "dialogue",
        "summary", "title", "untitled", "new chat", "help", "assistance",
        "user assistant", "message", "messages", "response", "answer",
    ]

    private nonisolated static func isGenericNonTitle(_ title: String) -> Bool {
        let key = title
            .lowercased()
            .trimmingCharacters(in: CharacterSet.punctuationCharacters.union(.whitespaces))
        if genericNonTitles.contains(key) { return true }
        return isLengthSpecification(key)
    }

    /// "3 to 6 words" — the reported defect, verbatim.
    ///
    /// A denylist of echoed phrases is whack-a-mole: the instruction that
    /// produced that one has been rewritten, and the next wording would leak
    /// a different literal. This is the general form instead — no
    /// conversation is ever named after how long its name should be — so it
    /// keeps working whatever the instruction says next.
    private nonisolated static func isLengthSpecification(_ key: String) -> Bool {
        let parts = key.split(whereSeparator: { $0 == " " })
        guard parts.count >= 3, parts.count <= 4 else { return false }
        guard Int(parts[0]) != nil else { return false }
        guard ["to", "-", "–"].contains(String(parts[1])) else { return false }
        guard Int(parts[2]) != nil else { return false }
        guard parts.count == 3 || ["word", "words"].contains(String(parts[3]))
        else { return false }
        return true
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
