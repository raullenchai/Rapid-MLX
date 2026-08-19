import Foundation

/// The one selection + progression contract shared by every Step 2 list
/// (Paper 05.2.G — "CTA derivation contract" and "Activation truth table").
///
/// ## Why this is a separate, pure enum
///
/// Step 2 now has three surfaces that can offer the same model — the
/// recommended shortlist, in-window Browse all models, and Review download.
/// Before this existed the chooser derived its footer inline:
///
/// ```swift
/// primaryTitle: canStartWithoutDownload(...) ? "Start existing model" : "Download & start"
/// ```
///
/// That is a two-way branch over one input, and it was correct only because
/// there was exactly one list and every row in it was always visible. Neither
/// holds any more: a catalogue row can be filtered out, searched away, absent
/// while the catalogue loads, missing because the catalogue failed, or too big
/// for this Mac. Duplicating the branch onto two more surfaces would give three
/// sites deciding what the button says, and the interesting cases are precisely
/// the ones a duplicated ternary gets wrong.
///
/// So the derivation lives here, once, as a pure function over the *visible*
/// rows. Paper states the rule as `selection ∩ visible rows in the active
/// list`: stored intent alone is never enough, because a selection the user
/// cannot see is not something they can be said to be choosing.
///
/// ## What it deliberately does NOT do
///
/// It never clears the selection. A pick hidden by a filter or a search is
/// "memory, not intent" — the alias is retained and becomes actionable again
/// the moment it is visible, which is what makes clearing a search restore the
/// previous pick without anything re-picking it. Only *actionability* changes.
///
/// Nothing here reads a label, a badge, a section heading or a card style.
/// Cached-ness arrives as a `Bool` derived from the catalogue snapshot
/// (``QuickstartView/canStartWithoutDownload(alias:cachedModels:)``), and
/// availability from ``ModelSizing/classify(_:on:)`` — the same decision the
/// model picker already disables on. Presentation is never evidence.
enum OnboardingModelSelection {

    // MARK: - Inputs

    /// Which Step 2 list the primary is being derived for.
    ///
    /// Only ``review`` changes the verb, and only for an uncached pick: the
    /// commit lives on one screen (Paper 05.2.H · T5), so "Download & start"
    /// exists there and nowhere else.
    enum ListContext: Equatable, Sendable {
        /// The recommended shortlist (micro-stage 2c).
        case shortlist
        /// In-window Browse all models (micro-stage 2d).
        case catalogue
        /// Review download (micro-stage 2e).
        case review
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

    /// What the footer primary does when activated.
    ///
    /// Two of these navigate and two commit. The split is load-bearing: only
    /// ``startExisting`` and ``downloadAndStart`` may reach ``DownloadManager``,
    /// the disk pre-flight or ``ServerManager/start(alias:hfPath:)``, so a
    /// model that cannot run here is kept out of those two cases by
    /// construction rather than by a check at each call site.
    enum Action: Equatable, Sendable {
        /// Open the Review download micro-stage for an uncached pick.
        case reviewDownload
        /// Open the SAME micro-stage for a pick this Mac cannot run, where it
        /// reads as an explanation rather than a decision (Paper 05.2.D ·
        /// `V3/Onb-2e-ReviewDownload-IncompatibleMemory`).
        ///
        /// Deliberately not folded into ``reviewDownload``. The two arrive at
        /// one screen but mean different things — this one promises no
        /// download and must never acquire one — and naming the difference
        /// here is what lets the derivation, rather than the view, be the
        /// thing that knows an incompatible pick is informational.
        case reviewIncompatible
        /// Start a model already on disk — straight to Step 4, no download
        /// job, no fabricated progress.
        case startExisting
        /// Commit the download. Review download only.
        case downloadAndStart

        /// Whether activating this would spend something — bytes, disk, or a
        /// model load. The one question the execution paths care about.
        var isCommit: Bool {
            switch self {
            case .startExisting, .downloadAndStart: return true
            case .reviewDownload, .reviewIncompatible: return false
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

    /// The three verbs, stated once. Copy lives here so a test can pin the
    /// derivation without matching a string literal typed into a view.
    enum Verb {
        static let reviewDownload = "Review download"
        static let startExisting = "Start existing model"
        static let downloadAndStart = "Download & start"
    }

    /// The neutral disabled primary. Paper: "Disabled always shows the neutral
    /// verb, never a blank or a third label, so the control's identity is
    /// stable while its availability changes."
    static let disabledPrimary = Primary(
        title: Verb.reviewDownload,
        action: .reviewDownload,
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
        // Truthfully won't run on this Mac.
        //
        // This is the one branch where the answer differs by context, and the
        // difference is the whole of Paper 05.2.D's incompatible-memory
        // decision: "Opening the detail of a WON'T FIT row is allowed and
        // informational — the user asked what this model is, and refusing to
        // answer would be worse than answering. The primary is disabled, not
        // hidden, so the shape of the screen never changes between a model
        // that can start and one that cannot."
        //
        // So the LIST offers a way in, and REVIEW is where the refusal lands.
        // Note the order: this is decided before cached-ness, because a model
        // already on disk that cannot run is still a model that cannot run —
        // being downloaded already is not evidence about memory.
        if !row.isAvailable {
            switch context {
            case .shortlist, .catalogue:
                // Enabled, and still the neutral verb: the catalogue's primary
                // reads "Review download" in every state, so selecting a row
                // never relabels the control — only what the screen it opens
                // has to say changes.
                return Primary(
                    title: Verb.reviewDownload,
                    action: .reviewIncompatible,
                    isEnabled: true
                )
            case .review:
                // Paper draws this as "Download & start", greyed. The verb is
                // the one the model WOULD have taken, not a third label and
                // not the neutral one, because on this screen the control's
                // job is to name the thing that is being withheld.
                return Primary(
                    title: row.isCached ? Verb.startExisting : Verb.downloadAndStart,
                    action: row.isCached ? .startExisting : .downloadAndStart,
                    isEnabled: false
                )
            }
        }

        if row.isCached {
            // No download exists to review. This is the same verb on every
            // surface, including Review download itself.
            return Primary(title: Verb.startExisting, action: .startExisting, isEnabled: true)
        }
        switch context {
        case .review:
            return Primary(title: Verb.downloadAndStart, action: .downloadAndStart, isEnabled: true)
        case .shortlist, .catalogue:
            return Primary(title: Verb.reviewDownload, action: .reviewDownload, isEnabled: true)
        }
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

    /// Whether this Mac can run the alias, using the classification the model
    /// picker already disables on. Not a new compatibility claim — the same
    /// ``ModelSizing`` estimate, read in one more place.
    static func isAvailable(alias: String, hardware: MacHardware) -> Bool {
        ModelSizing.classify(ModelSizing.estimate(alias: alias), on: hardware) != .tooBig
    }

    /// Build the row set for a catalogue slice. Cached-ness comes from the
    /// entry the catalogue itself produced, never from copy or grouping.
    static func rows(for entries: [ModelEntry], hardware: MacHardware) -> [Row] {
        entries.map { entry in
            Row(
                alias: entry.alias,
                isCached: entry.cached,
                isAvailable: isAvailable(alias: entry.alias, hardware: hardware)
            )
        }
    }
}
