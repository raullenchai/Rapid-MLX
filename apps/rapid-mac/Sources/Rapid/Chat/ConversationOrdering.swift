import Foundation

/// The small piece of conversation-history mutation that determines sidebar
/// order. Keeping it independent of ``ChatViewModel`` makes the navigation
/// contract executable without constructing the streaming stack.
protocol ConversationOrderingItem: Identifiable {
    var updatedAt: Date { get set }
}

enum ConversationOrdering {
    /// Update an existing row and, only for real activity, mark it recent.
    ///
    /// Selecting a conversation snapshots the row being left, but that is a
    /// navigation event rather than activity. In that case ``touching`` is
    /// false and both its timestamp and array position must remain unchanged.
    static func updating<Item: ConversationOrderingItem>(
        _ items: [Item],
        id: Item.ID,
        touching: Bool,
        at now: Date,
        update: (inout Item) -> Void
    ) -> [Item] where Item.ID: Equatable {
        guard let index = items.firstIndex(where: { $0.id == id }) else {
            return items
        }

        var result = items
        update(&result[index])
        guard touching else { return result }

        result[index].updatedAt = now
        let touched = result.remove(at: index)
        result.insert(touched, at: 0)
        return result
    }
}
