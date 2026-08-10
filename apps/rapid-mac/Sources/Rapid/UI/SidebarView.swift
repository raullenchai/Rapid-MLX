import SwiftUI

/// Which surface the detail pane shows. Ollama-style: a chat surface and
/// a "Launch" page of connect-your-tools cards. Conversation history
/// ("Older" list) is a later milestone.
enum SidebarSection: Hashable {
    case chat
    case images
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

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
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
                title: "Launch",
                systemImage: "paperplane",
                isSelected: selection == .launch,
                action: { selection = .launch }
            )
            .accessibilityIdentifier("Sidebar.Launch")

            if !chat.conversations.isEmpty {
                // Date-grouped history (#1470), titled with the shared
                // SectionHeader so the groups match the refreshed visual
                // system (#1460) instead of the PR's original inline caption.
                ScrollView {
                    VStack(alignment: .leading, spacing: 1) {
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
        .padding(.vertical, RapidTheme.Space.sm)
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
        SidebarView.sections(for: chat.conversations, now: referenceDate)
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
    /// Uses `Calendar` (not a fixed 86 400s divisor) because the buckets are
    /// *calendar* days: something sent at 23:55 belongs to "Yesterday" once
    /// the clock passes midnight, even though barely any time has elapsed.
    /// Empty buckets produce no section, so no stray headings appear.
    static func sections(
        for conversations: [ChatConversation],
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

        for conv in conversations where !conv.isArchived {
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
                        .font(RapidFont.body)
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
                    .font(.system(size: 13, weight: .medium))
                    .frame(width: RapidTheme.Layout.iconSlot, alignment: .center)
                Text(title)
                    .font(RapidFont.body)
                    .lineLimit(1)
            }
        }
    }
}

/// Shared chrome for every sidebar row: fixed height, one row radius,
/// amber selection, neutral hover.
///
/// The selected treatment is the product's canonical "this is chosen"
/// signal — amber tint fill plus the deep-amber label. Deep amber (not
/// raw ``brandPrimary``) because a 13pt label in #EFA23A on the light
/// tint is under 3:1; the deeper shade of the same hue clears AA while
/// reading as the same colour.
private struct SidebarRow<Content: View>: View {
    let isSelected: Bool
    let action: () -> Void
    @ViewBuilder let content: Content

    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            content
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, RapidTheme.Space.sm)
                .frame(height: RapidTheme.ControlHeight.row)
                .background(
                    RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous)
                        .fill(fill)
                )
                .contentShape(
                    RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous)
                )
        }
        .buttonStyle(.plain)
        .foregroundStyle(isSelected ? RapidTheme.brandPrimaryDeep : Color.primary)
        .onHover { hovering = $0 }
        .rapidAnimation(RapidMotion.quick, value: hovering)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }

    private var fill: Color {
        if isSelected { return RapidTheme.brandPrimaryTint }
        return hovering ? RapidTheme.hoverFill : .clear
    }
}

/// The "Launch" detail page — reuses the connect-your-tools cards. Until
/// the engine ships `rapid-mlx launch <tool>` (issue #1405) these are the
/// copy-the-config cards; that CLI upgrade will turn each into a one-line
/// `rapid-mlx launch …` command without changing this surface's shape.
struct LaunchView: View {
    @Bindable var server: ServerManager
    let alias: String
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
            alias: alias,
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
