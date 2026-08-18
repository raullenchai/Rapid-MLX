import CoreTransferable
import Foundation
import UniformTypeIdentifiers

extension UTType {
    /// Private drag type for a sidebar conversation row.
    ///
    /// A dedicated type rather than reusing `.text` or `.plainText`: those are
    /// produced by every text field, browser and editor on the system, so a
    /// folder advertising them would light up as a drop target for any stray
    /// selection dragged over the rail — and then have to decide what to do
    /// with a payload that isn't a conversation at all. A private identifier
    /// means only rows from this app's own sidebar are ever offered.
    ///
    /// Declared in `Resources/Info.plist` under `UTExportedTypeDeclarations`;
    /// `exportedAs` requires the bundle to own the identifier, and an
    /// undeclared one logs a runtime complaint even though drags still work.
    static let rapidConversationRow = UTType(
        exportedAs: "com.rapidmlx.rapid.conversation-row"
    )
}

/// What a dragged sidebar row carries.
///
/// Just the id — the drop handler resolves it against the live model rather
/// than trusting a snapshot of the conversation taken when the drag began.
/// A drag can outlive what it describes: a streamed token, a rename, or an
/// archive from another surface can all land mid-gesture, and re-reading at
/// drop time means the filing applies to the conversation as it is now.
struct ConversationTransfer: Codable, Transferable, Equatable {
    let conversationID: UUID

    init(_ conversationID: UUID) {
        self.conversationID = conversationID
    }

    static var transferRepresentation: some TransferRepresentation {
        CodableRepresentation(contentType: .rapidConversationRow)
    }
}
