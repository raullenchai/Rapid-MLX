import Foundation

/// Curated, bounded ways into Computer Use. This is deliberately a catalog,
/// not cards hand-written in a view: availability and safety copy are
/// product behavior and must remain testable without rendering SwiftUI.
struct ComputerUseStarter: Identifiable, Equatable, Sendable {
    enum Kind: String, CaseIterable, Sendable {
        case freeUpSpace
        case tidyInbox
        case draftAndPost
        case prospectCustomers
        case createDemoVideo
        case reserved
    }

    enum Availability: Equatable, Sendable {
        case comingSoon
        case reserved
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
            availability: .comingSoon
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
            kind: .prospectCustomers,
            title: "Prospect customers",
            summary: "Find matching prospects, research each, and draft personal openers for review.",
            systemImage: "scope",
            applications: "Browser",
            approvalNote: "Rapid will stop before any outreach.",
            availability: .comingSoon
        ),
        ComputerUseStarter(
            kind: .createDemoVideo,
            title: "Create a demo video",
            summary: "Walk through your product, narrate each step, and export a polished demo.",
            systemImage: "video",
            applications: "Browser + Screen Capture",
            approvalNote: "You choose the window and review before export.",
            availability: .comingSoon
        ),
        ComputerUseStarter(
            kind: .reserved,
            title: "More flows are coming",
            summary: "Another broadly useful starter will appear here in a future preview.",
            systemImage: "plus",
            applications: "Coming later",
            approvalNote: "Not available yet.",
            availability: .reserved
        )
    ]
}
