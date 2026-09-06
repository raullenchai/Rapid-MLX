import Foundation

/// Curated, bounded ways into Computer Use. This is deliberately a catalog,
/// not four cards hand-written in a view: availability and safety copy are
/// product behavior and must remain testable without rendering SwiftUI.
struct ComputerUseStarter: Identifiable, Equatable, Sendable {
    enum Kind: String, CaseIterable, Sendable {
        case freeUpSpace
        case tidyInbox
        case draftAndPost
        case orderLunch
    }

    enum Availability: Equatable, Sendable {
        case available
        case comingSoon
    }

    let kind: Kind
    let title: String
    let summary: String
    let systemImage: String
    let applications: String
    let approvalNote: String
    let availability: Availability

    var id: Kind { kind }

    static let catalog: [ComputerUseStarter] = [
        ComputerUseStarter(
            kind: .freeUpSpace,
            title: "Free up space",
            summary: "Review old files in Downloads and move only what you select to Trash.",
            systemImage: "externaldrive.badge.minus",
            applications: "Finder",
            approvalNote: "Nothing moves until you review and approve it.",
            availability: .available
        ),
        ComputerUseStarter(
            kind: .tidyInbox,
            title: "Tidy my inbox",
            summary: "Find newsletters and promotions, then preview what to archive.",
            systemImage: "tray.full",
            applications: "Mail or Gmail",
            approvalNote: "Archiving and deletion will require review.",
            availability: .comingSoon
        ),
        ComputerUseStarter(
            kind: .draftAndPost,
            title: "Draft and post an update",
            summary: "Turn a local TextEdit draft into a post in your signed-in browser.",
            systemImage: "square.and.pencil",
            applications: "TextEdit + browser",
            approvalNote: "Rapid will stop before publishing.",
            availability: .comingSoon
        ),
        ComputerUseStarter(
            kind: .orderLunch,
            title: "Order lunch",
            summary: "Choose from a signed-in lunch site using your preferences and budget.",
            systemImage: "takeoutbag.and.cup.and.straw",
            applications: "Browser",
            approvalNote: "Rapid will stop before checkout.",
            availability: .comingSoon
        )
    ]
}
