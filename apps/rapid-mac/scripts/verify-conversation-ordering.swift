import Foundation

private struct StubConversation: ConversationOrderingItem, Equatable {
    let id: String
    var updatedAt: Date
    var content: String
}

@main
private enum VerifyConversationOrdering {
    static func main() {
        let old = Date(timeIntervalSince1970: 100)
        let middle = Date(timeIntervalSince1970: 200)
        let newest = Date(timeIntervalSince1970: 300)
        let activityTime = Date(timeIntervalSince1970: 400)
        let initial = [
            StubConversation(id: "A", updatedAt: newest, content: "a"),
            StubConversation(id: "B", updatedAt: middle, content: "b"),
            StubConversation(id: "C", updatedAt: old, content: "c"),
        ]

        // Reproduction contract: leaving A to view B snapshots A's content,
        // but browsing must not make A newer or move any row. Repeating the
        // operation for B is the sequence that used to produce B,A,C.
        let afterLeavingA = ConversationOrdering.updating(
            initial, id: "A", touching: false, at: activityTime
        ) { $0.content = "a-snapshot" }
        let afterLeavingB = ConversationOrdering.updating(
            afterLeavingA, id: "B", touching: false, at: activityTime
        ) { $0.content = "b-snapshot" }

        precondition(afterLeavingB.map(\.id) == ["A", "B", "C"],
                     "selecting older chats must not reorder the sidebar")
        precondition(afterLeavingB[0].updatedAt == newest,
                     "navigation must not refresh updatedAt")
        precondition(afterLeavingB[1].updatedAt == middle,
                     "navigation must preserve the departed row timestamp")
        precondition(afterLeavingB[1].content == "b-snapshot",
                     "non-touching snapshots must still update content")

        let afterActivity = ConversationOrdering.updating(
            afterLeavingB, id: "C", touching: true, at: activityTime
        ) { $0.content = "c-new-turn" }
        precondition(afterActivity.map(\.id) == ["C", "A", "B"],
                     "real conversation activity must move its row to the top")
        precondition(afterActivity[0].updatedAt == activityTime,
                     "real conversation activity must refresh updatedAt")

        print("conversation ordering contract: ok")
    }
}
