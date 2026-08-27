import SwiftUI

/// Which surface the detail pane shows. Ollama-style: a chat surface and
/// a "Launch" page of connect-your-tools cards. Conversation history
/// ("Older" list) is a later milestone.
enum SidebarSection: Hashable {
    case chat
    case images
    case audio
    case launch
}

/// The left sidebar — Ollama/ChatGPT layout: a "New Chat" action at the
/// top, a "Launch" page entry, then (later) the conversation history. It
/// is the primary column of ``ContentView``'s ``NavigationSplitView``, so
/// macOS gives us the collapse toggle in the toolbar for free.
struct SidebarView: View {
    /// Column metrics. v1.0: narrowed from 190/230/300.
    ///
    /// At the old ideal the rail took ~35% of a 640pt window and ~26% of
    /// a 900pt one for two nav rows and a usually-empty history list —
    /// which is what made the mostly-blank column read as unfinished.
    /// A 200pt rail still fits the longest conversation titles at this
    /// row density while giving the detail pane the width it needs at
    /// the 640pt floor.
    ///
    /// Exposed (rather than inlined at the ``ContentView`` call site) so
    /// the snapshot harness composes the same column it ships.
    static let columnMinWidth: CGFloat = 176
    static let columnIdealWidth: CGFloat = 200
    static let columnMaxWidth: CGFloat = 260

    @Binding var selection: SidebarSection
    /// The chat model — source of the conversation history list + the
    /// active conversation id (for highlighting).
    @Bindable var chat: ChatViewModel
    /// Start a fresh conversation and show the chat surface.
    var onNewChat: () -> Void
    /// Open the window-level conversation search panel.
    var onSearchChats: () -> Void = {}
    /// Open a saved conversation (switches the detail pane back to chat).
    var onSelectConversation: (UUID) -> Void
    /// Optional in isolated snapshot fixtures; the shipping ContentView passes
    /// it so residency and the enforced memory ceiling remain visible globally.
    var server: ServerManager? = nil

    /// The "now" the date buckets are computed against. Rolled forward by
    /// ``dayBoundaryTicker`` at each midnight so an open, untouched sidebar
    /// re-labels yesterday's conversations instead of freezing on the day it
    /// was first rendered. Injected (rather than reading ``Date()`` inside the
    /// section builder) so the roll-over is an observable state change.
    @State private var referenceDate = Date()

    /// The conversation a context-menu "Delete" has staged for removal, shown
    /// in the confirmation dialog. Deleting a conversation is irreversible
    /// (``ConversationStore.save`` atomically overwrites the on-disk store —
    /// no trash, no undo), so — unlike navigating — it must be confirmed first.
    @State private var pendingDeletion: ChatConversation?

    /// The conversation currently being renamed inline, plus its draft text.
    /// Inline (rather than a sheet) because a rename is a one-field edit on a
    /// row the user is already pointing at; a modal for it reads as a much
    /// heavier action than it is.
    @State private var renamingID: UUID?
    @State private var renameDraft = ""

    /// Keyboard focus for the inline rename editor.
    ///
    /// Without this the editor was purely decorative: it never became first
    /// responder, so every keystroke went to whatever already held focus (the
    /// chat composer), and `.onSubmit` / `.onExitCommand` — which only fire
    /// for the FOCUSED view — were unreachable, leaving the row stuck in edit
    /// mode with no way out. See ``renameField(_:)``.
    @FocusState private var renameFieldFocused: Bool

    /// Whether the open editor has actually held focus at least once.
    ///
    /// Cancel-on-focus-loss has to distinguish "the user clicked away" from
    /// "focus has not arrived yet": the editor is created unfocused and only
    /// takes first responder on a later scheduling pass, so reacting to every
    /// `false` would cancel the rename the instant it opened.
    @State private var renameFieldDidFocus = false

    /// Bumped once per rename the user opens, and used as the editor's
    /// `.task` id.
    ///
    /// Keying the focus request on the ROW id would tie "re-request focus" to
    /// "a different row", which is one edit session too coarse: re-opening
    /// Rename on the row already being edited leaves the id unchanged, so the
    /// task would not re-run and the editor would sit there unfocused. A
    /// per-session counter re-runs it for every open, whichever row it lands on.
    @State private var renameSession = 0

    /// Whether the Archived group is expanded. Collapsed by default — the
    /// whole point of archiving is to get those rows out of the way.
    @State private var showArchived = false

    /// Which history row the pointer is over. Tracked here rather than inside
    /// ``SidebarRow`` because the pin / ··· controls are siblings of the row
    /// button (a `Menu` inside a `Button` label is unclickable), so they need
    /// a hover signal the row itself doesn't own.
    @State private var hoveredConversationID: UUID?

    /// Which folders are expanded.
    ///
    /// Expanded-by-default (a folder id is absent until the user collapses
    /// it — see ``isFolderExpanded(_:)``): the user filed those conversations
    /// deliberately, so the useful default is being able to see them. This is
    /// the opposite of Archived, which is collapsed by default precisely
    /// because its whole purpose is getting rows out of the way.
    @State private var collapsedFolderIDs: Set<UUID> = []

    /// The folder-name prompt currently on screen, if any.
    ///
    /// A one-field `.alert` rather than the inline editor the conversation
    /// rows use. The inline path (``renamingID`` / ``renameSession`` /
    /// ``renameFieldDidFocus``) exists because a row rename has to happen in
    /// place, and getting SwiftUI focus to arrive on a field that replaces a
    /// list row took the whole state machine documented on ``renameField(_:)``.
    /// Naming a folder has no such constraint, and paying that complexity a
    /// second time — as a copy or as a risky generalisation of working code —
    /// buys nothing the alert doesn't already give us, including keyboard
    /// commit/cancel for free.
    @State private var folderPrompt: FolderPrompt?
    @State private var folderNameDraft = ""

    /// The folder a "Delete" has staged for confirmation.
    @State private var pendingFolderDeletion: ChatFolder?

    /// The folder a dragged row is currently held over, if any. Drives the
    /// drop highlight so the gesture says where it will land before release.
    @State private var dropTargetFolderID: UUID?

    /// What the folder-name prompt will do when committed.
    enum FolderPrompt: Identifiable, Equatable {
        /// Create a folder. When ``fileConversationID`` is set, the newly
        /// created folder immediately adopts that conversation — this is the
        /// "Move to Folder ▸ New Folder…" path, where making the folder and
        /// filing the row are one user intention rather than two.
        case create(fileConversationID: UUID?)
        case rename(folder: ChatFolder)

        var id: String {
            switch self {
            case let .create(conversationID):
                return "create-\(conversationID?.uuidString ?? "none")"
            case let .rename(folder):
                return "rename-\(folder.id.uuidString)"
            }
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            brandLockup
            row(
                title: "New Chat",
                systemImage: "square.and.pencil",
                isSelected: false,
                action: onNewChat
            )
            .accessibilityIdentifier("Sidebar.NewChat")
            row(
                title: "Images",
                systemImage: "photo",
                isSelected: selection == .images,
                action: { selection = .images }
            )
            .accessibilityIdentifier("Sidebar.Images")
            row(
                title: "Audio",
                systemImage: "waveform",
                isSelected: selection == .audio,
                action: { selection = .audio }
            )
            .accessibilityIdentifier("Sidebar.Audio")
            row(
                title: "Launch",
                systemImage: "paperplane",
                isSelected: selection == .launch,
                action: { selection = .launch }
            )
            .accessibilityIdentifier("Sidebar.Launch")

            // Folders can exist before any conversation does (create one, then
            // file into it), so an empty history no longer means an empty rail.
            if !chat.conversations.isEmpty || !chat.folders.isEmpty {
                // Date-grouped history (#1470), titled with the shared
                // SectionHeader so the groups match the refreshed visual
                // system (#1460) instead of the PR's original inline caption.
                ScrollView {
                    VStack(alignment: .leading, spacing: 1) {
                        folderSectionsList
                        ForEach(historySections, id: \.title) { section in
                            SectionHeader(section.title)
                                .padding(.horizontal, RapidTheme.Space.sm)
                                .padding(.top, RapidTheme.Space.lg)
                                .padding(.bottom, RapidTheme.Space.xs)
                            ForEach(section.conversations) { conv in
                                conversationRow(conv)
                            }
                        }
                        archivedSection
                    }
                }
                .scrollIndicators(.never)
            }

            Spacer(minLength: 0)

            if let server, !server.residency.models.isEmpty {
                residencyFooter(
                    server.residency,
                    preferredAlias: server.servingAlias
                )
            }
        }
        .padding(.horizontal, RapidTheme.Space.sm)
        .padding(.vertical, RapidTheme.Space.md)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        // Column-scoped toolbar content renders beside the native sidebar
        // toggle instead of in the detail column's title group.
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button(action: onSearchChats) {
                    Image(systemName: "magnifyingglass")
                }
                .keyboardShortcut("k", modifiers: .command)
                .help("Search chats - Command-K")
                .accessibilityLabel("Search chats")
                .accessibilityIdentifier("Toolbar.SearchChats")
            }
        }
        .task { await dayBoundaryTicker() }
        // A conversation is a HARD delete with no undo, so confirm first —
        // mirrors the cached-model delete dialog. ``confirmationDialog`` over
        // ``alert`` so the cancel-role button is Return-bound.
        .confirmationDialog(
            Self.deleteConfirmationTitle(for: pendingDeletion),
            isPresented: Binding(
                get: { pendingDeletion != nil },
                set: { if !$0 { pendingDeletion = nil } }
            ),
            titleVisibility: .visible,
            presenting: pendingDeletion
        ) { conv in
            // A ``confirmationDialog`` button is re-hosted by AppKit rather
            // than rendered as an ordinary SwiftUI button, so it was an open
            // question whether `AXIdentifier` survives that hop. It does:
            // measured on a build of this branch, the presented dialog is an
            // `AXSheet` (description "alert") whose two `AXButton` children
            // report these identifiers. Keep that in mind if the deployment
            // target ever moves — the guarantee is empirical, not documented.
            Button("Delete", role: .destructive) {
                chat.deleteConversation(conv.id)
                pendingDeletion = nil
            }
            .accessibilityIdentifier("Sidebar.DeleteConversation.Confirm")
            Button("Keep", role: .cancel) {
                pendingDeletion = nil
            }
            .accessibilityIdentifier("Sidebar.DeleteConversation.Keep")
        } message: { _ in
            Text("This permanently deletes the conversation. It can't be undone.")
        }
        // Deleting a folder is NOT destructive to transcripts, so it gets a
        // plainly-worded confirmation rather than the destructive-delete
        // treatment above — the dialog's job here is to say what will happen
        // to the conversations, which is the only thing the user is unsure of.
        .confirmationDialog(
            pendingFolderDeletion.map { "Delete “\($0.name)”?" } ?? "Delete folder?",
            isPresented: Binding(
                get: { pendingFolderDeletion != nil },
                set: { if !$0 { pendingFolderDeletion = nil } }
            ),
            titleVisibility: .visible,
            presenting: pendingFolderDeletion
        ) { folder in
            Button("Delete Folder", role: .destructive) {
                chat.deleteFolder(folder.id)
                pendingFolderDeletion = nil
            }
            .accessibilityIdentifier("Sidebar.DeleteFolder.Confirm")
            Button("Keep", role: .cancel) {
                pendingFolderDeletion = nil
            }
            .accessibilityIdentifier("Sidebar.DeleteFolder.Keep")
        } message: { _ in
            Text("The conversations in it are kept and move back into the date list.")
        }
        .sheet(
            isPresented: Binding(
                get: { folderPrompt != nil },
                set: { if !$0 { folderPrompt = nil } }
            )
        ) {
            folderPromptSheet
        }
    }

    /// #2050: this prompt used to be a `.alert` with a TextField. SwiftUI
    /// drops the accessibilityIdentifier from a text field inside an alert
    /// (button identifiers propagate; the field's does not), and on the CI
    /// runner an AX value write into an alert field never reaches the
    /// SwiftUI binding — so the golden flow could neither find the field
    /// nor enable Save. A plain sheet exposes the same controls as
    /// ordinary views, where identifiers, value writes, and presses all
    /// behave (fresh-install drives the post-value invitation the same way on the
    /// same runner). The three identifiers are the flow's contract — keep
    /// them stable.
    private var folderPromptSheet: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.lg) {
            Text(folderPromptTitle)
                .font(RapidFont.bodyEmphasis)
            TextField("Folder name", text: $folderNameDraft)
                .textFieldStyle(.roundedBorder)
                .accessibilityIdentifier("Sidebar.Folder.NameField")
                .onSubmit { if canCommitFolderPrompt { commitFolderPrompt() } }
            HStack(spacing: RapidTheme.Space.sm) {
                Spacer()
                Button("Cancel", role: .cancel) { folderPrompt = nil }
                    .keyboardShortcut(.cancelAction)
                    .accessibilityIdentifier("Sidebar.Folder.Prompt.Cancel")
                Button("Save") { commitFolderPrompt() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(!canCommitFolderPrompt)
                    .accessibilityIdentifier("Sidebar.Folder.Prompt.Confirm")
            }
        }
        .padding(RapidTheme.Space.xl)
        .frame(width: 320)
    }

    private var folderPromptTitle: String {
        switch folderPrompt {
        case .rename: return "Rename Folder"
        case .create, .none: return "New Folder"
        }
    }

    private var canCommitFolderPrompt: Bool {
        guard let name = ChatFolder.normalizedName(folderNameDraft) else { return false }
        switch folderPrompt {
        case let .rename(folder):
            return !chat.folderNameExists(name, excluding: folder.id)
        case .create:
            return !chat.folderNameExists(name)
        case .none:
            return false
        }
    }

    /// Open the name prompt, seeding the draft with the current name for a
    /// rename and blank for a new folder.
    private func presentFolderPrompt(_ prompt: FolderPrompt) {
        // Same reasoning as every mutating row action: the prompt takes
        // keyboard focus, so an inline conversation rename must be resolved
        // rather than left pending behind a modal.
        cancelRename()
        switch prompt {
        case .create: folderNameDraft = ""
        case let .rename(folder): folderNameDraft = folder.name
        }
        folderPrompt = prompt
    }

    /// Commit whatever the prompt was for. A blank name is rejected by the
    /// model layer (``ChatFolder.normalizedName``), which leaves the folder
    /// untouched — the same "don't commit an empty name" contract
    /// ``renameConversation`` applies to rows.
    private func commitFolderPrompt() {
        defer {
            folderPrompt = nil
            folderNameDraft = ""
        }
        switch folderPrompt {
        case let .create(conversationID):
            guard let folder = chat.createFolder(named: folderNameDraft) else { return }
            if let conversationID {
                chat.moveConversation(conversationID, toFolder: folder.id)
            }
        case let .rename(folder):
            chat.renameFolder(folder.id, to: folderNameDraft)
        case .none:
            break
        }
    }

    /// The product lockup at the top of the rail: the official Rapid `R`
    /// beside the product name.
    ///
    /// Purely decorative, and marked so. The window already carries the
    /// app name in its title bar and in the menu bar, so announcing it a
    /// third time would put a redundant element ahead of "New Chat" in
    /// every VoiceOver traversal of the sidebar — the rail's first
    /// element should be the first thing you can DO in it. It is also
    /// what keeps this addition free of any new accessibility identifier
    /// or AX node for the identifier inventory to account for.
    ///
    /// The mark is ``RapidRMark``'s template image — the same geometry
    /// the menu-bar status item renders, so the two brand surfaces cannot
    /// drift apart. Template rendering means it tracks the label colour
    /// in both appearances rather than needing a second asset for Dark.
    private var brandLockup: some View {
        HStack(spacing: RapidTheme.Space.sm) {
            if let mark = RapidRMark.menuBarTemplateImage(height: 15) {
                Image(nsImage: mark)
                    .renderingMode(.template)
                    .foregroundStyle(RapidTheme.textPrimary)
                    .frame(width: RapidTheme.Layout.iconSlot, alignment: .center)
            }
            Text("Rapid-MLX")
                .font(RapidFont.bodyEmphasis)
                .foregroundStyle(RapidTheme.textPrimary)
        }
        .padding(.horizontal, RapidTheme.Space.sm)
        .padding(.top, RapidTheme.Space.xs)
        .padding(.bottom, RapidTheme.Space.md)
        .accessibilityHidden(true)
    }

    private func residencyFooter(
        _ snapshot: ModelResidencySnapshot,
        preferredAlias: String?
    ) -> some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.xs) {
            HStack(spacing: RapidTheme.Space.xs) {
                Image(systemName: "memorychip")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.secondary)
                Text("Resident")
                    .font(RapidFont.caption)
                    .foregroundStyle(.secondary)
                Spacer(minLength: 4)
                Text(memorySummary(snapshot))
                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            if snapshot.memoryLimitBytes > 0 {
                ProgressView(
                    value: min(1, Double(snapshot.memoryUsedBytes) / Double(snapshot.memoryLimitBytes))
                )
                .controlSize(.mini)
                .tint(RapidTheme.brandAmber)
            }

            ForEach(snapshot.models.prefix(4)) { model in
                HStack(spacing: RapidTheme.Space.xs) {
                    Image(systemName: model.pinned ? "lock.fill" : "circle.fill")
                        .font(.system(size: model.pinned ? 9 : 6, weight: .semibold))
                        .foregroundStyle(model.pinned ? RapidTheme.brandAmber : .secondary)
                        .frame(width: 12)
                    Text(model.displayName(preferredAlias: preferredAlias))
                        .font(RapidFont.caption)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer(minLength: 4)
                    Text(Self.formatBytes(model.displayBytes))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                .accessibilityIdentifier("Sidebar.ResidentModel.\(model.id)")
            }
        }
        .padding(.horizontal, RapidTheme.Space.sm)
        .padding(.top, RapidTheme.Space.md)
        .padding(.bottom, RapidTheme.Space.sm)
        // A rule, not a gap. The residency block is the one part of the
        // rail that describes the MACHINE rather than the app's
        // navigation, and separating it by whitespace alone left it
        // reading as a fifth, oddly-formatted nav group. A hairline says
        // "different kind of thing" in the width of one pixel.
        .overlay(alignment: .top) {
            Rectangle()
                .fill(RapidTheme.hairline)
                .frame(height: 1)
        }
        .accessibilityIdentifier("Sidebar.Residency")
    }

    private func memorySummary(_ snapshot: ModelResidencySnapshot) -> String {
        let used = Self.formatBytes(snapshot.memoryUsedBytes)
        guard snapshot.memoryLimitBytes > 0 else { return used }
        return "\(used) / \(Self.formatBytes(snapshot.memoryLimitBytes))"
    }

    nonisolated private static func formatBytes(_ bytes: UInt64) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(clamping: bytes), countStyle: .memory)
    }

    // MARK: - Accessibility identifiers

    /// Identifier for a row's hover pin control.
    ///
    /// Named for the ACTION the press performs, not for the control's glyph:
    /// before this existed the button inherited the SF Symbol name (`pin`) as
    /// its `AXIdentifier`, which is an implementation detail that changes the
    /// moment somebody swaps the icon. Keying on the conversation id keeps
    /// rows distinguishable; flipping between `Pin` and `Unpin` means a golden
    /// flow asserts the state change by which identifier is now present,
    /// rather than reading a value off the button.
    nonisolated static func pinControlIdentifier(for conversation: ChatConversation) -> String {
        let action = conversation.isPinned ? "Unpin" : "Pin"
        return "Sidebar.Conversation.\(action).\(conversation.id.uuidString)"
    }

    /// Identifier for the row menu's pin/unpin item. Not keyed on the
    /// conversation — only one menu is ever open, and the item is shared by
    /// the ··· menu and the right-click menu.
    nonisolated static func pinMenuItemIdentifier(for conversation: ChatConversation) -> String {
        conversation.isPinned
            ? "Sidebar.Conversation.Action.Unpin"
            : "Sidebar.Conversation.Action.Pin"
    }

    /// Identifier for the row menu's archive/unarchive item.
    nonisolated static func archiveMenuItemIdentifier(for conversation: ChatConversation) -> String {
        conversation.isArchived
            ? "Sidebar.Conversation.Action.Unarchive"
            : "Sidebar.Conversation.Action.Archive"
    }

    /// Confirmation title for deleting a saved conversation. Unlike a cached
    /// MODEL (re-downloadable, and already gated) a conversation delete is
    /// irreversible, so this always fronts a confirmation.
    nonisolated static func deleteConfirmationTitle(for conversation: ChatConversation?) -> String {
        guard let conversation else { return "Delete this conversation?" }
        let title = conversation.title.trimmingCharacters(in: .whitespacesAndNewlines)
        return title.isEmpty ? "Delete this conversation?" : "Delete “\(title)”?"
    }

    /// The history list split into dated sections, newest first.
    ///
    /// Everything used to sit under a hard-coded "Older" heading, so a
    /// conversation five seconds old was filed as ancient history.
    private var historySections: [HistorySection] {
        SidebarView.sections(
            for: chat.conversations,
            folders: chat.folders,
            now: referenceDate
        )
    }

    /// Advance ``referenceDate`` at each calendar-day boundary for as long as
    /// the sidebar is on screen. Sleeps until the next midnight, bumps the
    /// state (which re-buckets the list), then loops. Cancels with the view.
    private func dayBoundaryTicker() async {
        let calendar = Calendar.current
        while !Task.isCancelled {
            let next =
                calendar.nextDate(
                    after: Date(),
                    matching: DateComponents(hour: 0, minute: 0, second: 0),
                    matchingPolicy: .nextTime
                ) ?? Date().addingTimeInterval(24 * 60 * 60)
            let delay = next.timeIntervalSinceNow
            if delay > 0 {
                try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            }
            if Task.isCancelled { break }
            referenceDate = Date()
        }
    }

    struct HistorySection {
        let title: String
        let conversations: [ChatConversation]
    }

    /// Bucket conversations for the main list. ``now`` is injected rather than
    /// read inside, matching ``RelativeTimestamp`` — it keeps the function pure
    /// so the day boundaries can be exercised without waiting for midnight.
    ///
    /// Pinned rows are lifted into their own leading section and are exempt
    /// from the date buckets entirely: a pin means "keep this where I can see
    /// it", which a "Previous 7 Days" heading would immediately undo. Archived
    /// rows are excluded here — ``archivedConversations`` owns them.
    ///
    /// Conversations filed into an existing folder are excluded too, and are
    /// rendered by ``folderSections(for:folders:)`` instead. Every row appears
    /// in exactly ONE place in the rail: a row showing up under both its
    /// folder and "Yesterday" would make the selection highlight appear twice
    /// and leave "which one does Delete act on?" genuinely ambiguous. Filing
    /// is an explicit instruction about where a conversation lives, so it
    /// outranks the buckets the app derives on its own.
    ///
    /// ``folders`` defaults to empty so the many existing call sites and tests
    /// that predate folders keep compiling and keep their behaviour.
    ///
    /// Uses `Calendar` (not a fixed 86 400s divisor) because the buckets are
    /// *calendar* days: something sent at 23:55 belongs to "Yesterday" once
    /// the clock passes midnight, even though barely any time has elapsed.
    /// Empty buckets produce no section, so no stray headings appear.
    static func sections(
        for conversations: [ChatConversation],
        folders: [ChatFolder] = [],
        now: Date,
        calendar: Calendar = .current
    ) -> [HistorySection] {
        var pinned: [ChatConversation] = []
        var today: [ChatConversation] = []
        var yesterday: [ChatConversation] = []
        var week: [ChatConversation] = []
        var older: [ChatConversation] = []

        // The 7-day cutoff is anchored to the START of today, not to `now` —
        // otherwise the boundary would drift through the day and a
        // conversation could slide between sections while the user watches.
        let startOfToday = calendar.startOfDay(for: now)
        let weekCutoff = calendar.date(byAdding: .day, value: -7, to: startOfToday)

        let liveFolderIDs = Set(folders.map(\.id))

        for conv in conversations where !conv.isArchived {
            // A folderID pointing at a folder that no longer exists falls
            // through to the date buckets rather than vanishing — see
            // ``folderSections(for:folders:)``.
            if let folderID = conv.folderID, liveFolderIDs.contains(folderID) {
                continue
            }
            if conv.isPinned {
                pinned.append(conv)
            } else if calendar.isDate(conv.updatedAt, inSameDayAs: now) {
                today.append(conv)
            } else if calendar.isDateInYesterday(conv.updatedAt) {
                yesterday.append(conv)
            } else if let weekCutoff, conv.updatedAt >= weekCutoff {
                week.append(conv)
            } else {
                older.append(conv)
            }
        }

        return [
            ("Pinned", pinned),
            ("Today", today),
            ("Yesterday", yesterday),
            ("Previous 7 Days", week),
            ("Older", older),
        ]
        .filter { !$0.1.isEmpty }
        .map { HistorySection(title: $0.0, conversations: $0.1) }
    }

    /// One user-created folder plus the conversations filed into it.
    struct FolderSection: Identifiable {
        let folder: ChatFolder
        let conversations: [ChatConversation]
        var id: UUID { folder.id }
    }

    /// Group conversations under their folders.
    ///
    /// Three deliberate behaviours:
    ///
    ///   * **Empty folders still render.** A folder is empty for the whole
    ///     window between creating it and filing something into it; hiding it
    ///     there would make creation look like it silently failed.
    ///   * **Orphans are not dropped.** A ``folderID`` naming a folder that no
    ///     longer exists is ignored here and the row falls through to
    ///     ``sections(for:folders:now:calendar:)``. ``ChatViewModel.deleteFolder``
    ///     already clears those ids eagerly, so this is the belt-and-braces
    ///     path for a hand-edited or partially-written store — and the failure
    ///     mode it prevents (a conversation that exists on disk but appears
    ///     nowhere in the rail) reads exactly like data loss.
    ///   * **Archived rows never appear**, matching the main list: the
    ///     Archived disclosure owns them regardless of filing.
    ///
    /// Within a folder, pinned rows lead and the rest are newest-updated
    /// first — the same ranking the main list uses, applied locally.
    static func folderSections(
        for conversations: [ChatConversation],
        folders: [ChatFolder]
    ) -> [FolderSection] {
        guard !folders.isEmpty else { return [] }

        var grouped: [UUID: [ChatConversation]] = [:]
        for conv in conversations where !conv.isArchived {
            guard let folderID = conv.folderID else { continue }
            grouped[folderID, default: []].append(conv)
        }

        return ChatFolder.displayOrder(folders).map { folder in
            let rows = (grouped[folder.id] ?? []).sorted { lhs, rhs in
                if lhs.isPinned != rhs.isPinned { return lhs.isPinned }
                return lhs.updatedAt > rhs.updatedAt
            }
            return FolderSection(folder: folder, conversations: rows)
        }
    }

    /// Archived rows, newest-updated first. Kept out of ``sections`` so the
    /// main list can never accidentally render one.
    ///
    /// The sort is done here rather than trusted from the caller: the main
    /// list's order is maintained incrementally by ``ConversationOrdering``,
    /// which deliberately leaves a row's position alone for non-activity edits
    /// — and archiving is one of those. So an archived row keeps whatever slot
    /// it held in the live list, which says nothing about its rank among the
    /// other archived ones.
    static func archived(for conversations: [ChatConversation]) -> [ChatConversation] {
        conversations
            .filter { $0.isArchived }
            .sorted { $0.updatedAt > $1.updatedAt }
    }

    private var archivedConversations: [ChatConversation] {
        SidebarView.archived(for: chat.conversations)
    }

    private var folderSections: [FolderSection] {
        SidebarView.folderSections(for: chat.conversations, folders: chat.folders)
    }

    /// Expanded unless explicitly collapsed — see ``collapsedFolderIDs``.
    private func isFolderExpanded(_ id: UUID) -> Bool {
        !collapsedFolderIDs.contains(id)
    }

    /// The user's folders, above the date buckets. Renders nothing at all
    /// when the user has no folders, so the affordance stays invisible until
    /// it's asked for — the same restraint ``archivedSection`` shows.
    @ViewBuilder
    private var folderSectionsList: some View {
        let sections = folderSections
        if !sections.isEmpty {
            ForEach(sections) { section in
                // The whole group is one drop target — header plus any rows
                // it is showing. Restricting the target to the header would
                // mean an expanded folder's own area rejects the drop, which
                // is the part of it the pointer is most likely over.
                VStack(alignment: .leading, spacing: 1) {
                    folderHeader(section)
                    if isFolderExpanded(section.folder.id) {
                        if section.conversations.isEmpty {
                            // A brand-new folder is empty, and an expanded
                            // group showing nothing reads as a rendering bug.
                            Text("No conversations yet")
                                .font(RapidFont.caption)
                                .foregroundStyle(.tertiary)
                                .padding(.horizontal, RapidTheme.Space.sm)
                                .padding(.leading, RapidTheme.Layout.iconSlot)
                                .padding(.vertical, RapidTheme.Space.xs)
                        }
                        ForEach(section.conversations) { conv in
                            conversationRow(conv)
                        }
                    }
                }
                .background(dropHighlight(for: section.folder.id))
                .dropDestination(for: ConversationTransfer.self) { items, _ in
                    file(items, into: section.folder.id)
                } isTargeted: { targeted in
                    dropTargetFolderID = targeted ? section.folder.id : nil
                }
            }
        }
    }

    /// The amber wash + border a folder shows while a row is held over it.
    /// Mirrors the composer's attachment drop affordance so the two drop
    /// targets in the app read as the same gesture.
    @ViewBuilder
    private func dropHighlight(for folderID: UUID) -> some View {
        if dropTargetFolderID == folderID {
            RoundedRectangle(cornerRadius: RapidTheme.Radius.row)
                .fill(RapidTheme.brandPrimary.opacity(0.12))
                .overlay(
                    RoundedRectangle(cornerRadius: RapidTheme.Radius.row)
                        .strokeBorder(RapidTheme.brandPrimary, lineWidth: 1.5)
                )
        }
    }

    /// Apply a drop. Returns whether anything was filed, which is what tells
    /// AppKit to animate the drag as accepted rather than snapping back.
    private func file(_ items: [ConversationTransfer], into folderID: UUID) -> Bool {
        // An open inline rename is resolved first: filing relocates the row
        // between sections, taking any editor inside it off screen along with
        // the focus observer that would have ended the edit.
        cancelRename()
        var filed = false
        for item in items {
            // Re-resolved against the live model — the payload is only an id,
            // and dropping onto the folder a row already sits in is a no-op
            // rather than a spurious save.
            guard let conv = chat.conversations.first(where: { $0.id == item.conversationID }),
                  conv.folderID != folderID || conv.isArchived
            else { continue }
            chat.moveConversation(item.conversationID, toFolder: folderID)
            filed = true
        }
        return filed
    }

    private func folderHeader(_ section: FolderSection) -> some View {
        Button {
            // Collapsing takes any open editor inside the group off screen
            // along with the focus observer that resolves it, leaving the
            // rename pending — the same hazard the Archived toggle handles.
            cancelRename()
            let id = section.folder.id
            if collapsedFolderIDs.contains(id) {
                collapsedFolderIDs.remove(id)
            } else {
                collapsedFolderIDs.insert(id)
            }
        } label: {
            HStack(spacing: RapidTheme.Space.xs) {
                Image(
                    systemName: isFolderExpanded(section.folder.id)
                        ? "chevron.down" : "chevron.right"
                )
                .font(.system(size: 9, weight: .semibold))
                SectionHeader("\(section.folder.name) (\(section.conversations.count))")
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(.secondary)
        .padding(.horizontal, RapidTheme.Space.sm)
        .padding(.top, RapidTheme.Space.lg)
        .padding(.bottom, RapidTheme.Space.xs)
        .accessibilityIdentifier("Sidebar.Folder.Toggle.\(section.folder.axSlug)")
        .contextMenu { folderMenuItems(section.folder) }
    }

    @ViewBuilder
    private func folderMenuItems(_ folder: ChatFolder) -> some View {
        Group {
            Button {
                presentFolderPrompt(.rename(folder: folder))
            } label: {
                Label("Rename Folder", systemImage: "pencil")
            }
            .accessibilityIdentifier("Sidebar.Folder.Action.Rename")
            Divider()
            Button(role: .destructive) {
                cancelRename()
                pendingFolderDeletion = folder
            } label: {
                Label("Delete Folder", systemImage: "trash")
            }
            .accessibilityIdentifier("Sidebar.Folder.Action.Delete")
        }
        // Same template-image reasoning as ``rowMenuItems``.
        .tint(nil)
    }

    /// The collapsed Archived group. Renders nothing at all when empty, so a
    /// user who never archives anything never sees the affordance.
    @ViewBuilder
    private var archivedSection: some View {
        let archived = archivedConversations
        if !archived.isEmpty {
            Button {
                // Collapsing the group would take an archived row's editor off
                // screen along with the focus observer that resolves it, so the
                // rename would sit pending and reappear on the next expand.
                cancelRename()
                showArchived.toggle()
            } label: {
                HStack(spacing: RapidTheme.Space.xs) {
                    Image(systemName: showArchived ? "chevron.down" : "chevron.right")
                        .font(.system(size: 9, weight: .semibold))
                    SectionHeader("Archived (\(archived.count))")
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .padding(.horizontal, RapidTheme.Space.sm)
            .padding(.top, RapidTheme.Space.lg)
            .padding(.bottom, RapidTheme.Space.xs)
            .accessibilityIdentifier("Sidebar.Archived.Toggle")

            if showArchived {
                ForEach(archived) { conv in
                    conversationRow(conv)
                }
            }
        }
    }

    /// One history row — the conversation's derived title, amber-selected
    /// when it's the open one, with a hover-revealed pin toggle and overflow
    /// menu (and the same actions duplicated on right-click, since a context
    /// menu is what a macOS user reaches for first).
    ///
    /// The controls are an OVERLAY rather than trailing content inside the
    /// row's button: a `Menu` nested in a `Button` label is decorative — the
    /// outer button swallows the click, so the ··· would just open the
    /// conversation. Layering them as siblings keeps each one hittable.
    @ViewBuilder
    private func conversationRow(_ conv: ChatConversation) -> some View {
        if renamingID == conv.id {
            renameField(conv)
        } else {
            let isActive = selection == .chat && conv.id == chat.activeConversationID
            // Controls appear on hover OR on the selected row; a pinned row
            // always shows its pin, because otherwise the only signal that a
            // row is pinned would be its position, which reads as an
            // unexplained ordering bug.
            let hovering = hoveredConversationID == conv.id
            let showsControls = hovering || isActive || conv.isPinned
            ZStack(alignment: .trailing) {
                SidebarRow(isSelected: isActive) {
                    // Navigating away resolves any rename in progress rather
                    // than leaving a second row in edit mode behind us. The
                    // focus-loss cancel below normally gets there first; this
                    // is the belt-and-braces path for the case where the
                    // editor never took focus at all.
                    cancelRename()
                    onSelectConversation(conv.id)
                } content: {
                    // History rows carry no icon but keep the same leading
                    // inset as the nav rows above, so titles and nav labels
                    // align down one column instead of stepping in and out.
                    Text(conv.title)
                        .font(isActive ? RapidFont.bodyEmphasis : RapidFont.body)
                        .lineLimit(1)
                        .truncationMode(.tail)
                        .padding(.leading, RapidTheme.Layout.iconSlot + RapidTheme.Space.sm)
                        // Reserve the controls' width so revealing them
                        // re-truncates the title instead of overlapping it.
                        .padding(.trailing, showsControls ? Self.rowControlsWidth : 0)
                }
                .accessibilityIdentifier("Sidebar.Conversation.\(conv.id.uuidString)")
                if showsControls {
                    rowControls(conv, showsPin: hovering || isActive)
                }
            }
            .onHover { hoveredConversationID = $0 ? conv.id : nil }
            .contextMenu { rowMenuItems(conv) }
            // Drag to file into a folder. The preview is the title alone
            // rather than a snapshot of the row: the row carries hover
            // controls and a selection fill, and dragging a picture of those
            // around looks like the control itself came loose.
            .draggable(ConversationTransfer(conv.id)) {
                Text(conv.title)
                    .font(RapidFont.body)
                    .lineLimit(1)
                    .padding(.horizontal, RapidTheme.Space.sm)
                    .padding(.vertical, RapidTheme.Space.xs)
            }
        }
    }

    /// Width the pin + ··· pair occupies, reserved in the title's layout.
    private static let rowControlsWidth =
        RapidTheme.ControlHeight.mini * 2 + RapidTheme.Space.xs

    private func rowControls(_ conv: ChatConversation, showsPin: Bool) -> some View {
        HStack(spacing: 0) {
            if showsPin || conv.isPinned {
                QuietIconButton(
                    symbol: conv.isPinned ? "pin.slash" : "pin",
                    label: conv.isPinned ? "Unpin conversation" : "Pin conversation",
                    size: RapidTheme.ControlHeight.mini
                ) {
                    // Same reasoning as the menu's Pin: a pin moves the row
                    // into its own section, restructuring the list an open
                    // editor lives in.
                    cancelRename()
                    chat.setConversationPinned(conv.id, !conv.isPinned)
                }
                // Without this the button inherits the SF Symbol name
                // ("pin") as its AXIdentifier — see
                // ``pinControlIdentifier(for:)``.
                .accessibilityIdentifier(Self.pinControlIdentifier(for: conv))
            }
            // A native SwiftUI ``Menu`` — so the popup is an ``NSMenu``
            // drawn entirely by macOS. Its padding, corner radius, item
            // height and separator spacing are the system's on macOS 26
            // and are not app-settable; ``.menuStyle`` and
            // ``.menuIndicator`` below style the TRIGGER button, never
            // the popup. The only app-level styling that ever reached
            // the menu content was the scene tint, and ``rowMenuItems``
            // already clears it (see that method's note). Do not try to
            // give this Settings card padding or a custom radius —
            // there is no hook, and a custom popover would cost the
            // keyboard navigation and AX behaviour NSMenu provides.
            Menu {
                rowMenuItems(conv)
            } label: {
                Image(systemName: "ellipsis")
                    .font(.system(size: 11, weight: .semibold))
                    .frame(
                        width: RapidTheme.ControlHeight.mini,
                        height: RapidTheme.ControlHeight.mini
                    )
                    .contentShape(Rectangle())
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            .accessibilityLabel("Conversation actions")
            .accessibilityIdentifier("Sidebar.Conversation.Menu.\(conv.id.uuidString)")
        }
        .foregroundStyle(.secondary)
        .padding(.trailing, RapidTheme.Space.xs)
    }

    /// The row's actions, shared by the hover menu and the right-click menu so
    /// the two can't drift apart.
    ///
    /// Every item that immediately mutates or relocates a conversation resolves
    /// a rename in progress first. Pin and Archive move a row between sections,
    /// which restructures the list an open editor lives in, taking the editor —
    /// and with it the focus observer that would have cancelled the edit — off
    /// screen; so the edit is resolved up front rather than left to a blur that
    /// may never be observed. Delete is the documented exception (see below).
    ///
    /// The whole set is wrapped in a `Group` carrying ``tint(nil)`` — a
    /// `Group`'s modifiers apply to each child, and inside a `Menu` the
    /// children are flattened back out into sibling `NSMenuItem`s, so this
    /// does not nest them into a submenu.
    ///
    /// Why the tint has to be cleared: the scene applies
    /// ``.tint(RapidTheme.brandAmber)`` app-wide (``RapidApp``), which reaches
    /// this menu's content and makes SwiftUI hand AppKit *coloured*
    /// (`isTemplate == false`) menu-item images. AppKit only recolours
    /// TEMPLATE images to `selectedMenuItemTextColor` when a row highlights,
    /// so the amber glyphs stayed amber while their text flipped to white —
    /// the icon and its label disagreeing under the pointer. Dropping the
    /// tint restores template rendering, so each glyph tracks its own row's
    /// text colour at rest and on hover alike.
    @ViewBuilder
    private func rowMenuItems(_ conv: ChatConversation) -> some View {
        Group {
            Button {
                // Tear down any rename already in flight FIRST. Rename → Rename is
                // one continuous edit as far as ``renameFieldFocused`` is
                // concerned: leaving it `true` would rob the new editor of the
                // `false → true` transition its focus gate watches for, and its own
                // eventual blur would then be ignored. Ending the previous cycle
                // outright is what guarantees the transition happens.
                endRename()
                renameDraft = conv.title
                renamingID = conv.id
                renameSession &+= 1
            } label: {
                Label("Rename", systemImage: "pencil")
            }
            .accessibilityIdentifier("Sidebar.Conversation.Action.Rename")
            moveToFolderMenu(conv)
            Button {
                presentFolderPrompt(.create(fileConversationID: conv.id))
            } label: {
                Label("Move to New Folder…", systemImage: "folder.badge.plus")
            }
            .accessibilityIdentifier("Sidebar.Conversation.Action.MoveToNewFolder")
            Divider()
            Button {
                cancelRename()
                chat.setConversationPinned(conv.id, !conv.isPinned)
            } label: {
                Label(
                    conv.isPinned ? "Unpin" : "Pin",
                    systemImage: conv.isPinned ? "pin.slash" : "pin"
                )
            }
            .accessibilityIdentifier(Self.pinMenuItemIdentifier(for: conv))
            Button {
                cancelRename()
                chat.setConversationArchived(conv.id, !conv.isArchived)
            } label: {
                Label(
                    conv.isArchived ? "Unarchive" : "Archive",
                    systemImage: conv.isArchived ? "tray.and.arrow.up" : "archivebox"
                )
            }
            .accessibilityIdentifier(Self.archiveMenuItemIdentifier(for: conv))
            Divider()
            Button {
                ConversationExportPanel.export(conv, format: .markdown)
            } label: {
                Label("Export Markdown…", systemImage: "doc.richtext")
            }
            .accessibilityIdentifier("Sidebar.Conversation.Action.Export.Markdown")
            Button {
                ConversationExportPanel.export(conv, format: .json)
            } label: {
                Label("Export JSON…", systemImage: "curlybraces")
            }
            .accessibilityIdentifier("Sidebar.Conversation.Action.Export.JSON")
            Divider()
            // Delete is the one item that does NOT need an explicit cancel: it only
            // stages a confirmation, and presenting that dialog takes keyboard
            // focus, which the editor's blur handler resolves. Keeping the action
            // body a bare `pendingDeletion = conv` is also what the #1568
            // delete-gate guard pins.
            //
            // Spelled with a trailing `label:` closure rather than the shorter
            // `Button("Delete", …)` so the destructive item carries an icon like
            // every other entry above it — a title-only button renders with a
            // blank icon gutter beside four icon-bearing siblings and looks broken.
            Button(role: .destructive) {
                pendingDeletion = conv
            } label: {
                Label("Delete", systemImage: "trash")
            }
            .accessibilityIdentifier("Sidebar.Conversation.Action.Delete")
        }
        .tint(nil)
    }

    /// The "Move to Folder" submenu for one conversation.
    ///
    /// Creating a folder is a first-level row action; this submenu stays
    /// focused on choosing among existing destinations. "Remove from Folder"
    /// appears only when the row is actually filed.
    @ViewBuilder
    private func moveToFolderMenu(_ conv: ChatConversation) -> some View {
        if !chat.folders.isEmpty || conv.folderID != nil {
            Menu {
                ForEach(ChatFolder.displayOrder(chat.folders)) { folder in
                    Button {
                        cancelRename()
                        chat.moveConversation(conv.id, toFolder: folder.id)
                    } label: {
                        // A checkmark rather than a disabled row: the current
                        // folder still has to be visible in the list.
                        Label(
                            folder.name,
                            systemImage: conv.folderID == folder.id
                                ? "checkmark.circle.fill" : "folder"
                        )
                    }
                    .accessibilityIdentifier(
                        "Sidebar.Conversation.Action.MoveToFolder.\(folder.axSlug)"
                    )
                }
                if !chat.folders.isEmpty, conv.folderID != nil { Divider() }
                if conv.folderID != nil {
                    Button {
                        cancelRename()
                        chat.moveConversation(conv.id, toFolder: nil)
                    } label: {
                        Label("Remove from Folder", systemImage: "folder.badge.minus")
                    }
                    .accessibilityIdentifier("Sidebar.Conversation.Action.MoveToFolder.Remove")
                }
            } label: {
                Label("Move to Folder", systemImage: "folder")
            }
            .accessibilityIdentifier("Sidebar.Conversation.Action.MoveToFolder")
        }
    }

    /// Inline rename editor, occupying the row it replaces.
    ///
    /// Return commits. Escape cancels, and so does losing focus — clicking
    /// another row, or anything else that takes keyboard focus — so a rename
    /// can never be committed by accidentally clicking away. The paths that
    /// remove the editor WITHOUT necessarily moving focus (opening another
    /// conversation, New Chat, Launch, collapsing the Archived group) call
    /// ``cancelRename()`` directly rather than relying on the blur. (Switching
    /// to another app
    /// does NOT cancel: AppKit keeps first responder inside the window, so an
    /// open rename survives a Cmd-Tab, which is the behaviour you want.)
    ///
    /// Finder itself commits on click-away; cancelling is the deliberate
    /// choice here, because a discarded draft costs one retype while a
    /// silently-committed wrong title is a change the user never sees happen.
    ///
    /// Two pieces of wiring make that contract real, and both were missing
    /// when the editor first shipped:
    ///
    ///   * **Focus.** ``renameFieldFocused`` is bound with `.focused(...)` and
    ///     requested from `.task`. Requesting focus in the same render pass
    ///     that installs the field is a documented no-op — the backing AppKit
    ///     field is not in the responder chain yet — whereas `.task` runs once
    ///     the field is on screen. It is keyed on ``renameSession`` so every
    ///     edit the user opens re-focuses, whichever row it lands on. Without
    ///     focus, keystrokes went to the chat composer and `.onSubmit` /
    ///     `.onExitCommand` (which only fire for the focused view) were both
    ///     unreachable.
    ///   * **Hit area.** A bare ``TextField`` is only its intrinsic ~16pt
    ///     tall, so inside this 30pt row the pill drawn behind it was mostly
    ///     dead space: a click on the obvious target landed on nothing,
    ///     resigned first responder and focused nothing at all. The
    ///     `.contentShape` + tap handler make the whole pill focus the field.
    private func renameField(_ conv: ChatConversation) -> some View {
        TextField("Conversation name", text: $renameDraft)
            .textFieldStyle(.plain)
            .font(RapidFont.body)
            .focused($renameFieldFocused)
            .padding(.horizontal, RapidTheme.Space.sm)
            .frame(height: RapidTheme.ControlHeight.row)
            .background(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous)
                    .fill(RapidTheme.hoverFill)
            )
            .contentShape(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous)
            )
            .onTapGesture { renameFieldFocused = true }
            .onSubmit {
                chat.renameConversation(conv.id, to: renameDraft)
                endRename()
            }
            .onExitCommand { cancelRename() }
            .accessibilityIdentifier("Sidebar.Conversation.Rename.\(conv.id.uuidString)")
            .task(id: renameSession) {
                renameFieldDidFocus = false
                // Defer the request to the next scheduling point, so it lands
                // after the update that installs this field rather than during
                // it — a request made mid-update reaches no AppKit field and is
                // silently dropped. (A yield is a scheduler hop, not a
                // guaranteed runloop turn; it is empirically sufficient here
                // and preferable to guessing at a sleep.)
                await Task.yield()
                guard !Task.isCancelled, renamingID == conv.id else { return }
                renameFieldFocused = true
            }
            .onChange(of: renameFieldFocused) { _, focused in
                guard renamingID == conv.id else { return }
                if focused {
                    renameFieldDidFocus = true
                    return
                }
                // Only a loss AFTER focus was actually held is the user
                // clicking away; the opening `false` is just the editor
                // waiting for its first responder.
                guard renameFieldDidFocus else { return }
                cancelRename()
            }
    }

    /// Dismiss the inline editor without committing. Safe to call when no
    /// rename is open.
    private func cancelRename() {
        guard renamingID != nil else { return }
        endRename()
    }

    /// Tear down the editor's state after a commit or a cancel. Clearing
    /// ``renameFieldDidFocus`` here is what stops the focus-loss that
    /// FOLLOWS the dismissal from being read as a second cancel.
    private func endRename() {
        renamingID = nil
        renameDraft = ""
        renameFieldDidFocus = false
        renameFieldFocused = false
    }

    /// One nav row — icon in a fixed-width slot so every label starts on
    /// the same x, whatever the glyph's natural width.
    private func row(
        title: String,
        systemImage: String,
        isSelected: Bool,
        action: @escaping () -> Void
    ) -> some View {
        // Same reasoning as the history rows: leaving for New Chat / Launch
        // resolves a rename in progress instead of stranding it.
        SidebarRow(
            isSelected: isSelected,
            action: {
                cancelRename()
                action()
            }
        ) {
            HStack(spacing: RapidTheme.Space.sm) {
                Image(systemName: systemImage)
                    // Semibold when selected, matching the label. The
                    // glyph and its word are one object; letting only the
                    // text thicken made the icon look like it belonged to
                    // the row above.
                    .font(.system(size: 13, weight: isSelected ? .semibold : .medium))
                    .frame(width: RapidTheme.Layout.iconSlot, alignment: .center)
                Text(title)
                    .font(isSelected ? RapidFont.bodyEmphasis : RapidFont.body)
                    .lineLimit(1)
            }
        }
    }
}

/// Shared chrome for every sidebar row: fixed height, one row radius,
/// one selected treatment, neutral hover.
///
/// ## The selected treatment
///
/// A 3×18pt amber bar in the leading gutter, a neutral fill, and a
/// graphite SEMIBOLD label. Three signals, none of which is a colour
/// difference small enough to miss.
///
/// It replaces the v1.0 pairing of an amber TINT fill plus a deep-amber
/// label, which failed for a reason worth recording: the tint
/// (``brandPrimaryTint``) and the rail it sat on
/// (``surfaceSidebar``) were within a few percent of each other, so on
/// the rail — the one place selection matters most — the fill was
/// effectively invisible, and the whole signal came down to a hue shift
/// in a 13pt label. That is the single least robust way to encode state:
/// it is the first thing lost to colour-blindness, to a dim display, and
/// to peripheral vision.
///
/// The bar is a hard-edged shape, so it survives Increase Contrast and
/// reads at a glance from across the desk; the semibold weight carries
/// the row even in a screenshot with no colour at all. Amber is spent
/// here rather than on the fill because the rail's whole budget is one
/// small brand moment, and a bar is a smaller, sharper one than a slab.
private struct SidebarRow<Content: View>: View {
    let isSelected: Bool
    let action: () -> Void
    @ViewBuilder let content: Content

    @State private var hovering = false

    var body: some View {
        // ax-exempt: each caller attaches its entity-specific identifier to SidebarRow
        Button(action: action) {
            content
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, RapidTheme.Space.sm)
                .frame(height: RapidTheme.ControlHeight.row)
                .background(
                    RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous)
                        .fill(fill)
                )
                // The bar is an OVERLAY on the leading edge rather than a
                // sibling in an HStack: as a sibling it would take layout
                // width, so every label in the rail would shift sideways
                // the moment a row became selected. An overlay marks the
                // row without moving anything in it.
                .overlay(alignment: .leading) {
                    if isSelected {
                        Capsule(style: .continuous)
                            .fill(RapidTheme.selectionBar)
                            .frame(
                                width: RapidTheme.Layout.selectionBarWidth,
                                height: RapidTheme.Layout.selectionBarHeight
                            )
                    }
                }
                .contentShape(
                    RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous)
                )
        }
        .buttonStyle(.plain)
        // Graphite in both states. The label's WEIGHT now carries
        // selection (see ``SidebarView.row`` and ``conversationRow``),
        // which keeps every row in the rail at full reading contrast
        // instead of tinting the selected one down to a brand hue.
        .foregroundStyle(Color.primary)
        .onHover { hovering = $0 }
        .rapidAnimation(RapidMotion.quick, value: hovering)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }

    private var fill: Color {
        if isSelected { return RapidTheme.selectionFill }
        return hovering ? RapidTheme.hoverFill : .clear
    }
}

/// The "Launch" detail page — reuses the connect-your-tools cards. Until
/// the engine ships `rapid-mlx launch <tool>` (issue #1405) these are the
/// copy-the-config cards; that CLI upgrade will turn each into a one-line
/// `rapid-mlx launch …` command without changing this surface's shape.
struct LaunchView: View {
    @Bindable var server: ServerManager
    /// The side-channel downloader the inline model picker uses. Always
    /// supplied by the real ``ContentView``; the dev snapshot harness passes
    /// its own empty manager so the page renders standalone.
    @Bindable var downloads: DownloadManager
    /// The currently-chosen model, bound so the stopped state's inline
    /// picker can change it and the shared readiness re-derives.
    @Binding var alias: String
    /// The window's shared readiness value. Optional so the dev snapshot
    /// harness can render the page standalone; when supplied, the page
    /// describes the model lifecycle in exactly the same words the chat
    /// composer does, and offers the same next step.
    var readiness: ModelReadiness? = nil
    var onReadinessAction: (ModelReadiness.Action) -> Void = { _ in }

    var body: some View {
        ConnectToolsView(
            host: "127.0.0.1",
            port: server.activePort,
            bearer: server.activeBearer ?? "",
            alias: $alias,
            server: server,
            downloads: downloads,
            // The sidecar binary this app owns lives off-PATH (see
            // ``ServerLocator``); pass its absolute path so the launch/agent
            // commands this page generates actually run in a terminal.
            binaryPath: server.binaryPath,
            // Page context: the sidebar owns navigation, so there is no sheet
            // to dismiss. Hide the close ✕ (it used to render as a dead
            // no-op button). A dedicated page-mode header lands with the
            // #1405 Launch redesign.
            onClose: {},
            showsCloseButton: false,
            readiness: readiness,
            onReadinessAction: onReadinessAction
        )
    }
}
