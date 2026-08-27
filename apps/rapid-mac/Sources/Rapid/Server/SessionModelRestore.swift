import Foundation

/// Lane-owned model selections restored when the Desktop launches.
///
/// The process-owning alias is not a user selection: an audio-only fallback
/// may own the sidecar even though the user's chat choice is unchanged. Keep
/// those facts separate so auxiliary lanes can never become the chat model.
struct SessionModelRestore: Equatable, Sendable {
    struct LaunchPlan: Equatable, Sendable {
        let models: SessionModelRestore
        let chatAliasResolved: Bool
        let shouldAutoStart: Bool
    }

    /// Kept at the rc2 key for a migration without data loss; its semantic
    /// owner is now explicitly the chat lane.
    static let chatAliasStorageKey = "rapid.serve.lastAlias"

    let chatAlias: String?
    let dictationAlias: String?
    let speechAlias: String?

    /// A ready process may update chat intent only when the catalog proves it
    /// belongs to the chat lane. Unknown direct aliases fail closed: process
    /// ownership alone is never enough evidence to rewrite user selection.
    static func shouldPersistChatAlias(catalogEntry: ModelEntry?) -> Bool {
        catalogEntry?.kind == .chat
    }

    /// Record a successful ready transition without confusing the process
    /// owner with the user's chat selection. `ServerManager.start` funnels
    /// every ready child through here, including the audio-only fallback from
    /// `ensureVoiceLane`.
    static func persistReadyAlias(
        _ alias: String,
        catalogEntry: ModelEntry?,
        defaults: UserDefaults = .standard
    ) {
        guard shouldPersistChatAlias(catalogEntry: catalogEntry) else { return }
        defaults.set(alias, forKey: chatAliasStorageKey)
    }

    /// Resolve persisted aliases only through authoritative catalog lanes.
    /// `legacyLastAlias` is the rc2-era `rapid.serve.lastAlias` value; accepting
    /// it only as a chat-catalog member makes the migration safe without model
    /// names, repository ids, or hashes.
    static func resolve(
        legacyLastAlias: String?,
        dictationAlias: String?,
        speechAlias: String?,
        catalog: [ModelEntry]
    ) -> SessionModelRestore {
        let chatAliases = catalog.filter { $0.kind == .chat }.map(\.alias)
        let audioAliases = catalog.filter { $0.kind == .audio }.map(\.alias)
        return SessionModelRestore(
            chatAlias: validated(legacyLastAlias, in: chatAliases),
            dictationAlias: validated(dictationAlias, in: audioAliases),
            speechAlias: validated(speechAlias, in: audioAliases)
        )
    }

    /// Resolve lane ownership independently from the auto-start preference.
    /// Callers must publish `models.chatAlias` even when `shouldAutoStart` is
    /// false so onboarding and restore never disagree about a legacy key.
    static func launchPlan(
        legacyLastAlias: String?,
        dictationAlias: String?,
        speechAlias: String?,
        catalog: [ModelEntry],
        autoStartEnabled: Bool,
        emptyCatalogIsAuthoritative: Bool = false
    ) -> LaunchPlan {
        let hasLegacyAlias = legacyLastAlias?.trimmingCharacters(
            in: .whitespacesAndNewlines
        ).isEmpty == false
        // ModelCatalog uses [] as its subprocess-failure sentinel. With a
        // legacy value to classify, absence of rows is therefore unknown —
        // not evidence that the value is non-chat. Keep launch/onboarding
        // pending rather than rejecting it and selecting a fallback model.
        let chatAliasResolved = !hasLegacyAlias
            || !catalog.isEmpty
            || emptyCatalogIsAuthoritative
        return LaunchPlan(
            models: resolve(
                legacyLastAlias: legacyLastAlias,
                dictationAlias: dictationAlias,
                speechAlias: speechAlias,
                catalog: catalog
            ),
            chatAliasResolved: chatAliasResolved,
            shouldAutoStart: autoStartEnabled
                && chatAliasResolved
                && !emptyCatalogIsAuthoritative
        )
    }

    private static func validated(_ raw: String?, in aliases: [String]) -> String? {
        guard let raw else { return nil }
        let alias = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !alias.isEmpty else { return nil }
        return aliases.first {
            $0.caseInsensitiveCompare(alias) == .orderedSame
        }
    }
}
