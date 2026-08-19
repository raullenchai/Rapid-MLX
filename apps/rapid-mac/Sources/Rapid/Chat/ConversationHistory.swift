import Foundation

/// A saved conversation — the unit the sidebar history list shows and the
/// on-disk history persists. ``ChatMessage`` is already ``Codable``, so a
/// conversation serialises as-is.
///
/// ``messages`` holds EVERY branch as an unordered bag of tree nodes, not the
/// visible transcript. Read ``activePath`` for what the user is looking at.
struct ChatConversation: Identifiable, Codable, Equatable {
    let id: UUID
    var title: String
    /// The VISIBLE transcript — the active root-to-leaf path, in order.
    ///
    /// Deliberately still a plain linear conversation, because this is the
    /// field every already-shipped build reads. A downgrade shows the
    /// conversation the user was actually looking at and re-saves it intact;
    /// it loses the off-path alternatives (unavoidable — that build cannot
    /// represent them) but never scrambles the transcript. Storing the whole
    /// node bag here instead would hand an old build a pile of alternatives
    /// with no parent links and let it present them as one linear thread.
    var messages: [ChatMessage]
    /// Nodes that are NOT on the active path: the answers a Regenerate
    /// replaced, the prompts an edit superseded, and everything under them.
    ///
    /// A NEW key, which is also the schema marker. Its presence — not the
    /// shape of the parent links — is what says "this file was written by a
    /// build that understands branching". Shape inference cannot work here: a
    /// user who edits the opening prompt legitimately owns several parentless
    /// roots, which is indistinguishable from a pre-branching linear array.
    var branches: [ChatMessage] = []
    /// Tip of the branch currently on screen. ``nil`` means "resolve the most
    /// recently grown branch", which is also what every conversation written
    /// before branching shipped decodes to.
    var activeLeafID: UUID? = nil
    /// Parent → last-chosen-child, for every fork the user has navigated.
    ///
    /// Stepping to a sibling has to resolve downwards to some leaf, and
    /// "newest child" alone would throw away where the user was: leave a
    /// branch three turns from its tip, look at the alternative, come back,
    /// and you would land at the tip instead of where you left. Persisted so
    /// that survives relaunch too. Stale entries are ignored at read time
    /// rather than pruned, so a deleted branch cannot corrupt navigation.
    var branchChoices: [UUID: UUID] = [:]
    let createdAt: Date
    var updatedAt: Date

    /// Pinned rows sort into their own section above the date buckets and
    /// stay there regardless of how stale ``updatedAt`` gets.
    var isPinned: Bool = false

    /// Archived rows leave the main list entirely. They are NOT deleted —
    /// the transcript is untouched on disk — they just stop competing for
    /// attention with active work, and are reachable through the sidebar's
    /// Archived disclosure.
    var isArchived: Bool = false

    /// Set once the user renames the row. ``ChatViewModel.persistActive``
    /// re-derives ``title`` from the first user turn on every save, which
    /// would silently stomp a manual rename on the next streamed token;
    /// this flag is what makes the derivation skip an owned title.
    var hasCustomTitle: Bool = false

    /// Set once a background completion has written a machine title.
    ///
    /// Deliberately NOT ``hasCustomTitle``. That flag means the *user* owns
    /// the name — only ``ChatViewModel/renameConversation(_:to:)`` sets it,
    /// and nothing may overwrite a title it guards. Reusing it here would
    /// make a generated name indistinguishable from a rename, and the
    /// generator could no longer tell "leave this alone, the user named it"
    /// from "leave this alone, I named it".
    ///
    /// What this buys is narrower: it stops ``ChatViewModel/persistActive``
    /// re-deriving the title from the first user turn on the next save. A
    /// rename still wins outright — it sets ``hasCustomTitle``, and the
    /// generator refuses to write when that is true.
    ///
    /// One known benign interaction: a generated title landing while the
    /// sidebar's inline rename editor is open is invisible, because the row
    /// is showing the editor's `@State` draft. Committing that draft claims
    /// the title as the user's, which is the right outcome; cancelling shows
    /// the generated one. The window is one turn wide, once per conversation.
    var hasGeneratedTitle: Bool = false

    /// Instructions scoped to this conversation. Optional on disk so history
    /// written before custom instructions shipped remains valid.
    var customInstructions: String? = nil

    /// The user-created folder this conversation is filed under, if any.
    ///
    /// Optional for on-disk compatibility, and deliberately a soft reference:
    /// an id pointing at a folder that no longer exists degrades to "unfiled"
    /// at render time rather than hiding the row. See
    /// ``SidebarView/folderSections(for:folders:)``.
    var folderID: UUID? = nil

    init(
        id: UUID,
        title: String,
        messages: [ChatMessage],
        branches: [ChatMessage] = [],
        activeLeafID: UUID? = nil,
        branchChoices: [UUID: UUID] = [:],
        createdAt: Date,
        updatedAt: Date,
        isPinned: Bool = false,
        isArchived: Bool = false,
        hasCustomTitle: Bool = false,
        hasGeneratedTitle: Bool = false,
        customInstructions: String? = nil,
        folderID: UUID? = nil
    ) {
        self.id = id
        self.title = title
        // ``messages`` is the active PATH, so it is linear by construction and
        // its parent chain is derivable. Callers that build one by hand (tests,
        // dev fixtures, a future importer) hand over a bare array with no
        // links; chaining it here means no construction path can express a
        // path whose nodes are not connected.
        //
        // Keyed on ``branches`` being empty, NOT on the shape of the links: a
        // caller supplying real branch data owns the parent links already, and
        // re-chaining them would splice separate branches into one bogus line.
        let linked = branches.isEmpty
            ? MessageTree.repairingLegacyChain(messages)
            : messages
        self.messages = linked
        self.branches = branches
        self.activeLeafID = activeLeafID
        self.branchChoices = branchChoices
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.isPinned = isPinned
        self.isArchived = isArchived
        self.hasCustomTitle = hasCustomTitle
        self.hasGeneratedTitle = hasGeneratedTitle
        self.customInstructions = customInstructions
        self.folderID = folderID
    }

    /// Hand-written so a history file written before pin/archive shipped
    /// still decodes. The synthesised initialiser treats every stored
    /// property as required, so a missing `isPinned` key would throw —
    /// and ``ConversationStore.load`` turns one throw into "the whole
    /// history is corrupt", i.e. an apparently wiped sidebar on upgrade.
    /// Declared explicitly: writing ``encode(to:)`` by hand suppresses the
    /// compiler's synthesis of this enum along with the encoder itself.
    enum CodingKeys: String, CodingKey {
        case id, title, messages, branches, activeLeafID, branchChoices
        case createdAt, updatedAt, isPinned, isArchived, hasCustomTitle
        case customInstructions, folderID
    }

    /// Hand-written to match ``init(from:)``'s string-keyed
    /// ``branchChoices``. Everything else is written exactly as the
    /// synthesised encoder would.
    ///
    /// This has to exist: without it Swift synthesises an encoder that emits
    /// `[UUID: UUID]` as a flat array, the hand-written decoder asks for an
    /// object, and the map would be silently dropped on every reload — the
    /// feature would look like it worked all session and forget on relaunch.
    ///
    /// Compatibility contract, precisely: a conversation that never branched
    /// carries NO conversation-level branching key (`branches`,
    /// `activeLeafID`, `branchChoices` are all omitted). Its rows do carry
    /// `parentID` — an additive per-row key every shipped decoder ignores —
    /// so the output is decodable-identical to a pre-branching file, not
    /// byte-identical. What is load-bearing is the `branches` key's absence,
    /// which is the schema marker ``init(from:)`` keys on.
    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(title, forKey: .title)
        try c.encode(messages, forKey: .messages)
        // Omitted when empty — the key's ABSENCE is the schema signal on the
        // way back in: no `branches` key means "linear transcript, links may
        // be re-derived".
        if !branches.isEmpty {
            try c.encode(branches, forKey: .branches)
        }
        try c.encodeIfPresent(activeLeafID, forKey: .activeLeafID)
        if !branchChoices.isEmpty {
            let flattened = branchChoices.reduce(into: [String: String]()) { acc, entry in
                acc[entry.key.uuidString] = entry.value.uuidString
            }
            try c.encode(flattened, forKey: .branchChoices)
        }
        try c.encode(createdAt, forKey: .createdAt)
        try c.encode(updatedAt, forKey: .updatedAt)
        try c.encode(isPinned, forKey: .isPinned)
        try c.encode(isArchived, forKey: .isArchived)
        try c.encode(hasCustomTitle, forKey: .hasCustomTitle)
        try c.encodeIfPresent(customInstructions, forKey: .customInstructions)
        try c.encodeIfPresent(folderID, forKey: .folderID)
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        title = try c.decode(String.self, forKey: .title)
        // ``branches`` is the schema marker. Its PRESENCE says the writer
        // understood branching, so the stored parent links are authoritative
        // and must be left exactly as they are. Its absence means either a
        // pre-branching file or a conversation that never branched; in both
        // cases the transcript is linear and its links can be re-derived.
        //
        // Inferring this from the tree's shape instead — "no node has a
        // parent, therefore legacy" — is WRONG and was a real bug: editing the
        // opening prompt legitimately produces several parentless roots, which
        // is shape-identical to a legacy array. Re-chaining then spliced two
        // separate branches into one thread and destroyed the branch entry.
        let storedPath = try c.decode([ChatMessage].self, forKey: .messages)
        let storedBranches = try c.decodeIfPresent([ChatMessage].self, forKey: .branches)
        if let storedBranches {
            messages = storedPath
            branches = storedBranches
        } else {
            messages = MessageTree.repairingLegacyChain(storedPath)
            branches = []
        }
        activeLeafID = try c.decodeIfPresent(UUID.self, forKey: .activeLeafID)
        // Stored as a string-keyed object: Swift encodes a `[UUID: UUID]` as
        // a FLAT ARRAY of alternating keys and values, which is unreadable in
        // the file and silently order-dependent. Any entry that fails to
        // parse as a UUID pair is dropped rather than throwing — a corrupt
        // navigation hint must never cost the user the conversation.
        let storedChoices = try c.decodeIfPresent([String: String].self, forKey: .branchChoices) ?? [:]
        branchChoices = storedChoices.reduce(into: [UUID: UUID]()) { acc, entry in
            guard let parent = UUID(uuidString: entry.key),
                  let child = UUID(uuidString: entry.value) else { return }
            acc[parent] = child
        }
        createdAt = try c.decode(Date.self, forKey: .createdAt)
        updatedAt = try c.decode(Date.self, forKey: .updatedAt)
        isPinned = try c.decodeIfPresent(Bool.self, forKey: .isPinned) ?? false
        isArchived = try c.decodeIfPresent(Bool.self, forKey: .isArchived) ?? false
        hasCustomTitle = try c.decodeIfPresent(Bool.self, forKey: .hasCustomTitle) ?? false
        hasGeneratedTitle = try c.decodeIfPresent(Bool.self, forKey: .hasGeneratedTitle) ?? false
        customInstructions = try c.decodeIfPresent(String.self, forKey: .customInstructions)
        folderID = try c.decodeIfPresent(UUID.self, forKey: .folderID)
    }
}

extension ChatConversation: ConversationOrderingItem {}

extension ChatConversation {
    /// Every node across every branch — the conversation as a tree.
    ///
    /// ``messages`` alone is only the visible path; anything the user
    /// regenerated away lives in ``branches``. Orphans are promoted here
    /// rather than at decode time so a hand-edited file cannot strand a
    /// subtree outside every path.
    var allMessages: [ChatMessage] {
        // Dedupe first — path wins over branches — so a corrupt file that
        // carries the same id twice resolves to ONE node before any tree
        // arithmetic sees it; see ``MessageTree/deduplicatingByID``.
        MessageTree.promotingOrphans(MessageTree.deduplicatingByID(messages + branches))
    }

    /// The visible transcript — the root-to-leaf path ending at
    /// ``activeLeafID``, oldest turn first.
    ///
    /// Use this anywhere the user's current conversation is meant: rendering,
    /// the outbound wire body, export, title derivation. For a conversation
    /// that never branched this is just ``messages``.
    var activePath: [ChatMessage] {
        // One code path for every conversation — no `branches.isEmpty` fast
        // path. For a valid linear transcript the derived walk returns the
        // same array, and for a hand-edited one (dangling parents, cycles, a
        // stale leaf pointer) it degrades through the same guards every
        // branched conversation gets instead of returning raw storage.
        MessageTree.activePath(
            in: allMessages,
            activeLeafID: activeLeafID ?? messages.last?.id,
            preferring: branchChoices
        )
    }

    /// Whether any turn here has an alternative. Drives the "this row has
    /// branches" affordance without forcing callers to walk the tree.
    var hasBranches: Bool { !branches.isEmpty }
}

/// On-disk store for the conversation history. One JSON file under
/// Application Support, keyed by bundle identifier so a dogfood-isolated
/// instance (rewritten bundle id) keeps its own history separate from the
/// real app's. Writes are atomic; a missing / unreadable file reads as an
/// empty history (first run).
enum ConversationStore {
    static func fileURL(override: URL? = nil) -> URL {
        if let override { return override }
        // Must go through the locator: a direct FileManager call ignores
        // $HOME overrides, which is the #419/#420 shape a dogfood build
        // depends on (and what ApplicationSupportLocatorTests forbids).
        let base = ApplicationSupportLocator.applicationSupportBase()
        let dir = base.appendingPathComponent(
            Bundle.main.bundleIdentifier ?? "Rapid",
            isDirectory: true
        )
        try? FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true
        )
        return dir.appendingPathComponent("conversations.json")
    }

    static func load(from override: URL? = nil) -> [ChatConversation] {
        let url = fileURL(override: override)
        let fm = FileManager.default
        // A genuinely MISSING file is a normal first run → empty history.
        guard fm.fileExists(atPath: url.path) else { return [] }
        // A file that EXISTS but can't be read (I/O / permissions) or can't
        // be decoded (schema change, corruption) must NOT be silently
        // treated as empty — the next save would atomically overwrite the
        // user's whole history. Side it to `.corrupt-<t>` so it's
        // recoverable AND so the next save writes to a fresh path instead of
        // clobbering the original, then start empty.
        guard let data = try? Data(contentsOf: url),
              let decoded = try? JSONDecoder().decode([ChatConversation].self, from: data)
        else {
            // Unique name (UUID) so the move can't fail on a pre-existing
            // backup → the live file is reliably cleared out of the way
            // before we return empty. A move that still fails here is a
            // same-directory permission/IO fault, in which case the next
            // ``save`` (same dir) also fails — so there's no clobber path.
            let backup = url.deletingLastPathComponent().appendingPathComponent(
                "conversations.corrupt-\(UUID().uuidString).json"
            )
            try? fm.moveItem(at: url, to: backup)
            return []
        }
        // Sort here rather than trusting the file's array order. ``save``
        // writes whatever order the in-memory array happens to be in, which
        // ``ChatViewModel.persistActive`` keeps newest-first only as a side
        // effect of its ``insert(at: 0)`` bubbling. Any path that doesn't go
        // through that bubble — a hand-edited file, a future bulk import, a
        // partially-applied migration — would surface out of order in the
        // sidebar with nothing to correct it. Guaranteeing the invariant at
        // the data boundary costs one sort and removes the whole class.
        return decoded.sorted { $0.updatedAt > $1.updatedAt }
    }

    /// Serial queue for history writes: every save re-encodes the whole
    /// history, so a CONCURRENT queue could let an older snapshot's write
    /// land after a newer one's — overwriting recent turns or resurrecting
    /// a deleted conversation. A serial queue guarantees writes commit in
    /// submission order.
    private static let writeQueue = DispatchQueue(
        label: "com.rapidmlx.rapid.conversation-history-write",
        qos: .utility
    )

    /// Persist off the main actor — the caller (``ChatViewModel``) is
    /// ``@MainActor`` and every save re-encodes the whole history, so
    /// encoding + disk I/O must not run on the main thread or history
    /// growth would stall the UI. The snapshot is passed by value (Codable
    /// value types), so the background write sees a stable copy, and the
    /// serial ``writeQueue`` keeps ordering.
    static func save(_ conversations: [ChatConversation], to override: URL? = nil) {
        writeQueue.async {
            guard let data = try? JSONEncoder().encode(conversations) else { return }
            let url = fileURL(override: override)
            try? FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            // Owner-only (0600): chat transcripts are private. The atomic
            // write would otherwise inherit the umask default (often 0644 =
            // world-readable), exposing history to other local users.
            try? data.write(to: url, options: .atomic)
            try? FileManager.default.setAttributes(
                [.posixPermissions: 0o600], ofItemAtPath: url.path
            )
        }
    }

    /// Block until every queued write has committed. Call from app
    /// termination so the last turn / edit / deletion isn't lost when the
    /// process exits before the async ``save`` lands.
    static func flush() {
        writeQueue.sync {}
    }

    /// Derive a one-line title from the transcript — the first user
    /// message, whitespace-collapsed and length-capped (Ollama shows the
    /// opening message as the row label). Falls back to "New chat".
    static func title(from messages: [ChatMessage]) -> String {
        guard let first = messages.first(where: { $0.role == .user }) else {
            return "New chat"
        }
        let collapsed = first.content
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
        if collapsed.isEmpty, let filename = first.fileAttachments.first?.filename {
            return capped(filename)
        }
        if collapsed.isEmpty { return "New chat" }
        return capped(collapsed)
    }

    /// The sidebar row's one-line budget.
    ///
    /// Extracted so a generated title obeys the identical rule rather than a
    /// second copy of the number: the row is `.lineLimit(1)` either way, and
    /// two caps that drift produce titles that truncate differently depending
    /// on where they came from.
    static func capped(_ title: String) -> String {
        title.count > 42 ? String(title.prefix(42)) + "…" : title
    }
}
