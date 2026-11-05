import Foundation

/// Shared Step 2 selection contract. A visible, available selection starts
/// directly; an incompatible selection opens an explanation with no commit.
/// Hidden selections retain their alias but cannot trigger a download or load.
enum OnboardingModelSelection {

    // MARK: - Inputs

    enum ListContext: Equatable, Sendable {
        case shortlist
        case catalogue
        /// Blocking detail for a model this Mac cannot run.
        case incompatibleMemory
    }

    /// Whether the catalogue behind the active list has anything to say yet.
    ///
    /// ``ModelCatalog/load(binary:hubCacheOverride:)`` returns `[]` as its
    /// failure sentinel, so "loaded but empty" is not the same claim as "still
    /// loading" — and neither is evidence that a model is absent. Both disable
    /// progression rather than letting the footer act on a list that has not
    /// spoken.
    enum CatalogState: Equatable, Sendable {
        /// The snapshot has not landed. No row can be trusted absent.
        case loading
        /// The snapshot landed and carries rows.
        case ready
        /// The catalogue subprocess failed (the `[]` sentinel).
        case failed
    }

    /// The minimum truth a row contributes to the decision.
    ///
    /// Deliberately three fields and no view type: a row is identified by its
    /// stable alias, never by a display name or a curated label, so the same
    /// model reached from the shortlist and from the catalogue is one model.
    struct Row: Equatable, Sendable, Identifiable {
        /// Canonical rapid-mlx alias. The only identity this contract uses.
        let alias: String
        /// On disk right now, per the catalogue snapshot.
        let isCached: Bool
        /// False when the product can truthfully say this will not run here
        /// — ``ModelSizing/Fit/tooBig``. Unknown-parameter aliases classify
        /// ``borderline`` and stay available, so a custom alias is never
        /// blocked on a guess.
        let isAvailable: Bool

        var id: String { alias }

        init(alias: String, isCached: Bool, isAvailable: Bool = true) {
            self.alias = alias
            self.isCached = isCached
            self.isAvailable = isAvailable
        }
    }

    // MARK: - Output

    enum Action: Equatable, Sendable {
        case reviewIncompatible
        case startExisting
        case downloadAndStart

        var isCommit: Bool {
            switch self {
            case .startExisting, .downloadAndStart: return true
            case .reviewIncompatible: return false
            }
        }
    }

    /// The rendered footer primary: one verb, one action, one availability.
    ///
    /// The control never changes shape or lane between states — only its verb
    /// and whether it is enabled — so a disabled primary still names what it
    /// would do rather than going blank.
    struct Primary: Equatable, Sendable {
        let title: String
        let action: Action
        let isEnabled: Bool
    }

    /// The two commit verbs, stated once. Copy lives here so a test can pin the
    /// derivation without matching a string literal typed into a view.
    enum Verb {
        static let startExisting = String(localized: "Start existing model")
        static let downloadAndStart = String(localized: "Download & start")
    }

    /// The neutral disabled primary. Paper: "Disabled always shows the neutral
    /// verb, never a blank or a third label, so the control's identity is
    /// stable while its availability changes."
    static let disabledPrimary = Primary(
        title: Verb.downloadAndStart,
        action: .downloadAndStart,
        isEnabled: false
    )

    // MARK: - The derivation

    /// Derive the footer primary. Evaluated on every render — never cached,
    /// never assumed across a context switch.
    ///
    /// Order is the contract, and it is the order Paper's table is written in:
    /// the list has to be able to speak before a selection means anything, the
    /// selection has to be visible in it, and the model has to be one this Mac
    /// can run. Only then does cached-ness pick the verb.
    ///
    /// - Parameters:
    ///   - selection: the retained alias, or `nil` when nothing is picked.
    ///   - visibleRows: exactly the rows the user can currently see in the
    ///     active list — already searched, filtered and sorted. Passing the
    ///     unfiltered catalogue here would defeat the whole contract.
    ///   - catalogState: whether the backing snapshot has spoken.
    ///   - context: which list is asking.
    static func primary(
        selection: String?,
        visibleRows: [Row],
        catalogState: CatalogState,
        context: ListContext
    ) -> Primary {
        // The list cannot speak yet, or failed to. Nothing below is knowable.
        switch catalogState {
        case .loading, .failed:
            return disabledPrimary
        case .ready:
            break
        }
        // No results — a search that matched nothing, a filter that excluded
        // everything, or an empty cache under the Cached filter. All three are
        // the same fact: there is nothing here to act on.
        guard !visibleRows.isEmpty else { return disabledPrimary }
        // Stored intent alone is never enough. A retained alias that is not in
        // the visible set keeps its place in memory and loses its actionability.
        guard let selection,
              let row = visibleRows.first(where: { $0.alias == selection })
        else { return disabledPrimary }
        // Cached files do not make an incompatible model runnable.
        if !row.isAvailable {
            if context == .incompatibleMemory {
                return Primary(
                    title: row.isCached ? Verb.startExisting : Verb.downloadAndStart,
                    action: row.isCached ? .startExisting : .downloadAndStart,
                    isEnabled: false
                )
            }
            return Primary(title: Verb.downloadAndStart, action: .reviewIncompatible, isEnabled: true)
        }
        return row.isCached
            ? Primary(title: Verb.startExisting, action: .startExisting, isEnabled: true)
            : Primary(title: Verb.downloadAndStart, action: .downloadAndStart, isEnabled: true)
    }

    /// Whether the retained selection can be COMMITTED on right now — started,
    /// or taken through to a download.
    ///
    /// Exposed on its own for the row-level "is this the live pick" rendering
    /// and for Back-restoration, which must revalidate before it re-selects
    /// (Paper 05.2.G — "the list is rebuilt first, the selection is checked
    /// against it second, the primary is derived third").
    ///
    /// Note this is NOT "can the user do anything with this pick": since Paper
    /// 05.2.D an incompatible selection can still be opened in Review, which is
    /// a navigation and spends nothing. This answers the narrower question the
    /// execution paths care about, and an incompatible pick answers `false` to
    /// it in every context — including from inside Review itself.
    static func isActionable(
        selection: String?,
        visibleRows: [Row],
        catalogState: CatalogState
    ) -> Bool {
        guard case .ready = catalogState else { return false }
        guard let selection else { return false }
        guard let row = visibleRows.first(where: { $0.alias == selection }) else { return false }
        return row.isAvailable
    }

    // MARK: - Availability

    /// Whether this Mac can run the alias. Delegates to the shared
    /// ``ModelSizing/isAvailable(alias:on:catalogEntry:)`` verdict used by
    /// launch, picker, and Share Compute surfaces.
    ///
    /// A curated recommendation is trusted over ``ModelSizing``'s estimate,
    /// which over-states low-bit / MoE footprints (the 32 GB tier's
    /// `qwen3.8-27b-4bit` reads ~22 GB as an estimate but its measured 8K
    /// serve peak is 20.0 GB and fits). This mirrors every start path —
    /// ``ContentView.runLaunchAutoStart``, ``ModelPickerBar.handleStartTap``
    /// and ``CacheAwareDefault`` all gate `.tooBig` on the same
    /// ``RAMBucketedDefault.isRecommendedPick(alias:physicalRAMGB:)`` — so
    /// `.tooBig` is only a veto when the alias isn't THIS Mac's curated
    /// recommendation.
    ///
    /// That predicate is floor-guarded: it is true only for a tier this Mac
    /// genuinely sits in (RAM ≥ the tier's floor). Below the lowest tier's
    /// floor a Mac sits in no tier, so the curated picks it is shown are NOT
    /// exempt and the `.tooBig` veto still applies to them.
    static func isAvailable(
        alias: String,
        hardware: MacHardware,
        catalogEntry: ModelEntry? = nil
    ) -> Bool {
        ModelSizing.isAvailable(alias: alias, on: hardware, catalogEntry: catalogEntry)
    }

    /// Build the row set for a catalogue slice. Cached-ness comes from the
    /// entry the catalogue itself produced, never from copy or grouping.
    static func rows(for entries: [ModelEntry], hardware: MacHardware) -> [Row] {
        entries.map { entry in
            Row(
                alias: entry.alias,
                isCached: entry.cached,
                isAvailable: isAvailable(
                    alias: entry.alias, hardware: hardware, catalogEntry: entry
                )
            )
        }
    }
}
