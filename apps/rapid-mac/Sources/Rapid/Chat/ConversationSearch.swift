import Foundation

/// Pure conversation-search and grouping logic shared by the search surface
/// and its tests. Search stays local: it only inspects the already-loaded
/// history snapshot and never sends transcript content anywhere.
enum ConversationSearch {
    struct Section {
        enum Bucket: Hashable {
            case today
            case yesterday
            case previous7Days
            case previous30Days
            case older
        }

        let bucket: Bucket
        let conversations: [ChatConversation]
    }

    /// Match every whitespace-delimited query term against the conversation's
    /// title or visible user/assistant prose. Terms may land in different
    /// messages, which makes a query such as "swift cache" useful even when
    /// the two words came from different turns.
    static func results(
        in conversations: [ChatConversation],
        matching query: String
    ) -> [ChatConversation] {
        let terms = searchTerms(from: query)
        return conversations
            .filter { conversation in
                guard !terms.isEmpty else { return true }
                let searchableText = [conversation.title] + conversation.messages.compactMap { message in
                    switch message.role {
                    case .user, .assistant:
                        return message.content
                    case .system, .tool, .unknown:
                        return nil
                    }
                }
                return terms.allSatisfy { term in
                    searchableText.contains { text in
                        text.localizedStandardContains(term)
                    }
                }
            }
            .sorted { $0.updatedAt > $1.updatedAt }
    }

    /// Date groups used by the search panel. Archived conversations remain in
    /// these groups because search is their direct recovery path too.
    static func sections(
        for conversations: [ChatConversation],
        now: Date,
        calendar: Calendar = .current
    ) -> [Section] {
        let sorted = conversations.sorted { $0.updatedAt > $1.updatedAt }
        let startOfToday = calendar.startOfDay(for: now)
        let startOfYesterday = calendar.date(byAdding: .day, value: -1, to: startOfToday)
        let weekCutoff = calendar.date(byAdding: .day, value: -7, to: startOfToday)
        let monthCutoff = calendar.date(byAdding: .day, value: -30, to: startOfToday)

        var today: [ChatConversation] = []
        var yesterday: [ChatConversation] = []
        var week: [ChatConversation] = []
        var month: [ChatConversation] = []
        var older: [ChatConversation] = []

        for conversation in sorted {
            if calendar.isDate(conversation.updatedAt, inSameDayAs: now) {
                today.append(conversation)
            } else if let startOfYesterday,
                      conversation.updatedAt >= startOfYesterday,
                      conversation.updatedAt < startOfToday {
                yesterday.append(conversation)
            } else if let weekCutoff, conversation.updatedAt >= weekCutoff {
                week.append(conversation)
            } else if let monthCutoff, conversation.updatedAt >= monthCutoff {
                month.append(conversation)
            } else {
                older.append(conversation)
            }
        }

        return [
            Section(bucket: .today, conversations: today),
            Section(bucket: .yesterday, conversations: yesterday),
            Section(bucket: .previous7Days, conversations: week),
            Section(bucket: .previous30Days, conversations: month),
            Section(bucket: .older, conversations: older),
        ].filter { !$0.conversations.isEmpty }
    }

    private static func searchTerms(from query: String) -> [String] {
        query
            .split(whereSeparator: { $0.isWhitespace })
            .map(String.init)
    }
}
