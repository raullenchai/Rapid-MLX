import SwiftUI

/// Top bar: model picker + Start/Stop + server status pill. Sits above
/// the chat surface, à la ChatGPT Desktop where the model selector lives
/// in the title-bar region.
///
/// v0.6.9 (picker-v2 / Option B mock) reorganises the menu around
/// alias names rather than cache state. F-LWT-1 adds the Quickstart
/// section above Recommended so a user who skipped the Quickstart
/// card can still one-click install the demo model from the picker:
///
///   ┌── Quickstart ─────────────────────────────────┐
///   │  lfm2.5-1b-4bit · Smallest model — fastest   │ ← RAM-blind, F-LWT-1
///   │  first install                                │
///   ├── Recommended for your 18 GB Mac ─────────────┤
///   │  Recommended qwen3.5-9b-4bit   [amber row     │ ← the RAM tier's
///   │              when selected]                   │   smart pick
///   │  Faster      qwen3.5-4b-4bit                  │ ← optional fast/light alt
///   ├── All models (alphabetical) ──────────────────┤
///   │  bonsai-1.7b-unpacked                         │
///   │  deepseek-coder-v2-lite-16b-4bit            ● │ ← green dot = cached
///   │  gemma-4-12b-4bit                           ● │
///   │  qwen3.5-122b-8bit                            │
///   │  …                                            │
///   └──────────────────────────────────────────────┘
///
/// The Quickstart row is deduped from "All models" so the same
/// alias never appears in both sections.
///
/// Recommended section: the RAM tier's smart pick plus an optional
/// faster/lighter alternative (see ``RAMBucketedDefault`` and
/// ``MacHardware/recommendedPicks``). The row whose
/// alias matches the currently selected one paints in the
/// Start-button amber — same accent as the CTA so the selection
/// reads as a single colour story rather than two competing
/// highlights.
///
/// All-models section: every alias in a single list — downloaded
/// models first (each marked with a small green dot at the right
/// edge, present when the HF cache directory exists), then
/// alphabetical, so an alias the user already pulled surfaces at the
/// top instead of being buried mid-alphabet. Kept as ONE list (no
/// separate "Cached" section, so no alias renders twice). No size
/// suffix, no fit suffix. Rows are
/// fully clickable regardless of fit. Chat owns model lifecycle now;
/// this component only owns selection and model details.
struct ModelPickerBar: View {
    @Bindable var server: ServerManager
    /// v0.5.7: side-car downloader so right-clicking an UN-cached
    /// row can offer "Download in background" without disrupting the
    /// active serve. The picker stays the same shape; the extra
    /// affordance is opt-in via the context menu, mirroring how
    /// "Delete from disk" landed in v0.5.2.
    @Bindable var downloads: DownloadManager
    @Binding var alias: String
    /// F-LWT-1: the Quickstart coordinator owns the "user is
    /// mid-Quickstart-flow" gate. The picker reads ``phase`` to
    /// (a) render a dedicated "Quickstart" section above
    /// "Recommended for your <RAM> GB Mac" so a user who skipped
    /// Quickstart can still one-click-install the demo model later,
    /// and (b) disable the Start CTA while a Quickstart download is
    /// in flight so a stray click can't race a second concurrent
    /// download of the same alias. Optional so callers in unit
    /// tests / preview surfaces that don't care about Quickstart
    /// can omit the parameter and get the legacy behaviour.
    var quickstart: QuickstartCoordinator? = nil
    /// Native titlebar presentation used by ``ContentView``. Tests and
    /// standalone previews keep the original full-width strip by default.
    var titlebarStyle: Bool = false

    /// Ollama-style inline presentation: render ONLY the compact model
    /// picker chip (no info button, status badge, or Start/Stop), for
    /// embedding in the chat composer. Start/Stop is intentionally absent
    /// here — the lifecycle is implicit (``ChatViewModel.send`` calls
    /// ``ServerManager.ensureServing`` so the model comes up on first
    /// send). All the catalog/recommendation/download/delete logic still
    /// applies; only the surrounding control strip is dropped.
    var composerStyle: Bool = false
    /// False while a first-launch decision sheet is still pending. Catalog
    /// refresh shells out to `rapid-mlx models/ls`, which can inspect model
    /// caches and must not run before the user's first interaction (#1560).
    var catalogRefreshEnabled: Bool = true
    /// Called only for explicit model-row gestures, never when catalog
    /// initialization fills an empty selection.
    var onUserSelection: (String) -> Void = { _ in }

    /// The alias the Quickstart flow is currently aimed at — the wizard's
    /// live selection (#1524) while a coordinator is attached, else the
    /// fixed starter (lfm2.5-1b-4bit). This is the target the in-flight
    /// picker breadcrumb mirrors onto so the picker never shows a model
    /// that disagrees with what's downloading. The picker's *own*
    /// persistent "Quickstart" demo row is a separate, always-starter
    /// affordance and
    /// uses ``QuickstartCoordinator.defaultChoice`` directly.
    private var quickstartTargetAlias: String {
        quickstart?.selection.alias ?? QuickstartCoordinator.defaultChoice.alias
    }
    /// cycle-7: hide sub-1B aliases (qwen3-0.6b-*) from the
    /// alphabetical "All models" list by default — they hallucinate
    /// within 1-2 turns of chat and give first-time users a bad
    /// impression. Power users can flip the switch in
    /// ``Settings → Models``; the filter then drops to a no-op and
    /// the dropdown shows every alias including the tinies.
    /// See ``ModelPickerVisibility`` for the threshold and copy.
    @AppStorage(ModelPickerVisibility.showAllStorageKey) private var showAllModels: Bool = false
    @State private var catalog: [ModelEntry] = []
    @State private var loadingCatalog = false
    @State private var showCustom = false
    @State private var customDraft = ""
    /// v0.4.18: (i) popover next to the picker showing the selected
    /// alias's family / params / quant / context window / approx RAM /
    /// HF repo. Drives discoverability for users who don't yet know
    /// "qwen3.6-27b" maps to a 27B 4-bit model with a 32k context.
    @State private var showInfo = false
    /// Pointer-over state for the model picker's chrome. The picker is
    /// the only control in this bar without a system bezel, so it has
    /// to grow its own hover response — otherwise it reads as a status
    /// label rather than a menu.
    @State private var pickerHovering = false
    /// Host hardware snapshot. Probed once on first appearance and
    /// reused for every fit-classify call. Doesn't change until the
    /// user reboots into a new RAM module — i.e. never within a
    /// single app session.
    @State private var hardware: MacHardware = MacHardware.detect()

    /// v0.5.2: row queued for cache deletion. The context-menu
    /// "Delete from disk" affordance writes here; the `.alert`
    /// modifier reads it. Cleared on confirm-or-cancel. Held as a
    /// full ``ModelEntry`` (not just the alias) so the confirm copy
    /// can include the on-disk size without re-walking the catalog.
    @State private var pendingDeletion: ModelEntry?

    /// v0.5.2: outcome of the most recent ``rapid-mlx rm``
    /// invocation. Drives a brief toast at the top of the picker
    /// menu so the user gets confirmation without a full sheet.
    /// Cleared after `deletionToastDuration` seconds via a timer task.
    @State private var deletionToast: String?
    @State private var deletionToastGeneration: UInt = 0

    /// v0.5.2: in-flight deletion alias. Used to grey out the row
    /// while ``rapid-mlx rm`` is running so the user can't fire a
    /// second deletion against the same alias before the first
    /// returns. A single Mac shouldn't be deleting two big quants
    /// concurrently anyway — disk IO would saturate.
    @State private var deleting: String?
    @State private var catalogRefreshGeneration: UInt = 0

    // Retained for the legacy lifecycle helpers below, which remain as pure
    // compatibility seams for their existing unit tests. No mounted control
    // writes this state now that chat owns model startup (#1588).
    @State private var pendingTooBigStart: String?

    /// v0.5.2: how long the freed-bytes toast stays visible before
    /// auto-clearing. Picked to be longer than a glance read but
    /// shorter than typical user idle so it doesn't linger across
    /// the next picker interaction.
    private let deletionToastDuration: TimeInterval = 4

    /// Pick a sensible default alias for a fresh install. Ordered:
    ///   1. The user's last-picked alias (if it's still in the catalog,
    ///      handled by the caller via ``@AppStorage`` before this
    ///      function is consulted).
    ///   2. The RAM tier's measured smart pick (#163), if it is present
    ///      in the catalog. ``RAMBucketedDefault`` owns the fit contract;
    ///      this call site must not second-guess the curated tier with a
    ///      separate size heuristic.
    ///   3. Top general recommendation for this Mac.
    ///   4. First cached entry.
    ///   5. First catalog entry.
    /// We bias toward the bucketed / top-recommended slot rather than
    /// "biggest cached model" because on small Macs the biggest
    /// cached model is often the very one that will OOM.
    private func recommendedDefault() -> String? {
        // F-LWT-1: while Quickstart is mid-flow, the picker
        // selection must mirror the Quickstart alias — otherwise
        // the user sees Quickstart downloading the starter model while
        // the picker breadcrumb reads the RAM-bucketed default
        // (e.g. ``qwen3.5-9b-4bit`` on an 18 GB Mac), and a stray
        // Start CTA click would race a second concurrent download.
        // This branch closes the visual half of the 3-way mismatch.
        if ModelPickerBar.isQuickstartInFlight(phase: quickstart?.phase),
           catalog.contains(where: { $0.alias == quickstartTargetAlias }) {
            return quickstartTargetAlias
        }
        // A user who dismisses the Quickstart sheet has declined the guided
        // download flow, not the starter itself.  Keep the picker aimed at the
        // current coherent starter while that user remains Quickstart-eligible
        // (brand-new install, or the retired Bonsai cohort we are rescuing).
        //
        // Without this gate the cache-aware fallback below can immediately
        // resurrect ``bonsai-1.7b-2bit`` merely because its old weights are on
        // disk.  That makes the sheet recommend LFM2.5 while the chat composer
        // behind it says Bonsai — and Skip drops the user directly onto the
        // model retired for degenerating in ordinary plain chat.
        if let quickstart,
           let starter = ModelPickerBar.quickstartEligibleDefault(
               catalog: catalog,
               eligible: QuickstartCoordinator.isEligible(
               done: quickstart.done,
               legacyDone: quickstart.legacyDone,
               lastServedAlias: ServerManager.lastServedAlias(),
               serverState: server.state
               ),
               targetAlias: quickstartTargetAlias
           ) {
            return starter
        }
        // Returning users expect the picker to remember the model they last
        // ran even when they deliberately disable launch-time auto-start.
        // ``ContentView.alias`` is view state, not AppStorage; without this
        // branch a relaunch silently jumps to the first alphabetical cached
        // model. Never restore a retired automatic alias — the stranded
        // Quickstart branch above migrates that cohort to the current starter.
        if let previous = ModelPickerBar.lastServedDefault(
            catalog: catalog,
            lastServedAlias: ServerManager.lastServedAlias()
        ) {
            return previous
        }
        // v0.7.1 #229: on a fresh install the bundled lfm2.5-1b-4bit
        // is on disk and the user has never picked anything else.
        // Prefer it so the picker matches what the ContentView ``.task``
        // auto-restart starts — otherwise the breadcrumb shows a
        // 7+ GB alias while the server is happily running the 320 MB
        // bundled one, and the user thinks the picker is broken.
        // ``BundledModel.firstLaunchAlias`` returns nil when either
        // (a) the user already has a last-served alias or (b) the
        // bundled snapshot isn't present (dev/CI build skipped it),
        // so the existing RAM-bucketed flow below remains the
        // post-first-launch path.
        if let bundled = BundledModel.firstLaunchAlias(),
           catalog.contains(where: { $0.alias == bundled }) {
            return bundled
        }
        // Issue #436: cache-aware default beats the raw bucketed
        // pick. When the user's just-finished Quickstart left
        // ``qwen3-0.6b-4bit`` sitting in the HF cache, the picker
        // used to surface "Download & start qwen3.6-35b-4bit
        // (~4.4 GB)" on a fresh launch — burying the runnable model
        // behind the dropdown and undermining the Quickstart
        // 5-second time-to-first-token promise. ``CacheAwareDefault``
        // applies a four-step ladder:
        //   1. Bucketed default cached + fits → use it (no change).
        //   2. Cached-and-fits alternative exists → prefer it.
        //   3. Bucketed default fits but nothing cached fits → use
        //      it (legacy bucketed-default path).
        //   4. Bucketed default is .tooBig / missing → delegate to
        //      ``SafeDefaultFallback`` (the codex r2/r3 #165 escape).
        // Step 4 preserves every prior contract pinned in
        // ``SafeDefaultFallbackTests``; steps 1 and 3 preserve the
        // landing-page bucket contract pinned in
        // ``RAMBucketedDefaultTests``; step 2 is the new cache-
        // preference that closes #436. The ladder rationale lives
        // on ``CacheAwareDefault``.
        return CacheAwareDefault.pick(
            catalog: catalog,
            hardware: hardware,
            bucketedDefault: hardware.bucketedDefaultAlias
        )
    }

    /// The RAM tier's recommendations for this Mac — the smart pick plus
    /// its optional fast/light alt — paired with the catalog row backing
    /// each alias (or `nil` if the curated alias isn't in the catalog yet —
    /// e.g. rapid-mlx is on an older release that doesn't ship the alias).
    ///
    /// The rows are intentionally NOT decorated with any fit
    /// classification — the bucket table guarantees every entry fits
    /// the host. If a curated alias is missing from the catalog
    /// entirely (forward/backward-compat skew), the row is omitted
    /// so the picker never tries to ``Start`` an alias rapid-mlx
    /// doesn't know about.
    private func recommendedPickRows() -> [(pick: RAMBucketedDefault.Pick, entry: ModelEntry, isPrimary: Bool)] {
        // isPrimary comes from the SOURCE position (index 0 = primary), not
        // the post-filter index — otherwise a primary missing from the
        // catalog (rapid-mlx version skew) would leave the alt at index 0
        // and mislabel it "Recommended" instead of "Faster".
        hardware.recommendedPicks.enumerated().compactMap { index, pick in
            guard let entry = catalog.first(where: { $0.alias == pick.alias }) else {
                return nil
            }
            return (pick, entry, index == 0)
        }
    }

    var body: some View {
        // v0.5 (Phase 4): clean control strip. Left = model picker
        // (the single source of the model name) + its info button.
        // Right = the status pill and the Start/Stop action, grouped
        // together rather than scattered to opposite ends of the bar.
        controlRow
        // Codex round-1 finding: the catalog used to load only once
        // (via plain ``.task``). If the app launched with rapid-mlx
        // missing, ``refreshCatalog`` returned empty, and installing
        // rapid-mlx afterwards never triggered a re-fetch — the
        // picker stayed "catalog unavailable" until app restart.
        // Re-keying ``task(id:)`` on ``server.binaryPath`` runs the
        // catalog refresh again the instant the binary appears.
        // Also keyed on ``cacheGeneration`` so a model deleted (or
        // pulled) from ANY other surface invalidates this snapshot.
        // Keyed on binaryPath alone, the dropdown kept advertising a
        // model the user had just deleted in Settings as still on disk,
        // for the rest of the session.
        .task(id: PickerCatalogKey(
            binaryPath: server.binaryPath,
            cacheGeneration: downloads.cacheGeneration,
            refreshEnabled: catalogRefreshEnabled
        )) {
            guard catalogRefreshEnabled else { return }
            await refreshCatalog(force: true)
        }
        // F-LWT-1: mirror the picker selection to the Quickstart
        // alias the instant the coordinator flips into an in-flight
        // phase (the user clicked Get started). Without this the
        // picker breadcrumb stays on whatever ``recommendedDefault``
        // resolved at catalog-load time (RAM-bucketed default, e.g.
        // ``qwen3.5-9b-4bit`` on an 18 GB Mac) while Quickstart is
        // downloading the starter alias — the 3-way mismatch the bug
        // report screenshot pinned. ``.task(id:)`` re-fires on every
        // ``phase`` transition; the body bails when the gate is off
        // OR the picker is already aimed at the Quickstart alias
        // OR the alias isn't in the catalog yet (the late-arriving
        // catalog refresh will pick it up via ``recommendedDefault``).
        .task(id: quickstartPhaseGateKey) {
            applyQuickstartSelectionMirror()
        }
        .sheet(isPresented: $showCustom) {
            customAliasSheet
        }
        // v0.6 P1 (was v0.5.2): confirm-then-delete cache UI. Migrated
        // from `.alert` to `.confirmationDialog` so the cancel-role
        // button gets the default Return binding instead of the
        // destructive Delete — the old layout let a stray Return on a
        // muscle-memory keyboard shortcut wipe out multi-GB weights
        // with zero recovery. The dialog also surfaces the on-disk
        // size in the title (front-loaded) rather than only in the
        // message body, so a glance at the title is enough to feel
        // the cost.
        .confirmationDialog(
            ModelPickerBar.deletionTitle(for: pendingDeletion),
            isPresented: Binding(
                get: { pendingDeletion != nil },
                set: { if !$0 { pendingDeletion = nil } }
            ),
            titleVisibility: .visible,
            presenting: pendingDeletion
        ) { entry in
            Button("Delete from disk", role: .destructive) {
                Task { await runDeletion(of: entry) }
                pendingDeletion = nil
            }
            Button("Keep on disk", role: .cancel) {
                pendingDeletion = nil
            }
        } message: { entry in
            Text("Removes this model from your Mac. You can download it again later by selecting it.\(entry.sizeOnDisk.map { " Frees \($0)." } ?? "")")
        }
        // v0.5.2: brief toast at the top of the picker after a
        // successful (or failed) deletion. Auto-dismisses after a
        // few seconds so it doesn't linger across the next picker
        // interaction.
        .overlay(alignment: .top) {
            if let toast = deletionToast {
                Text(toast)
                    .font(.caption)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(.thinMaterial, in: Capsule())
                    .overlay(Capsule().strokeBorder(Color.primary.opacity(0.1)))
                    .padding(.top, 4)
                    .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
        // #547 §4/§14: the deletion toast springs in/out; instant under
        // Reduce Motion.
        .rapidAnimation(RapidMotion.standard, value: deletionToast)
    }

    /// The picker and its details affordance.  The app has only ever mounted
    /// this view inside the composer; lifecycle actions live in the readiness
    /// banner and composer, so the old standalone Start/Stop branch was a
    /// second, unreachable implementation of the same flow (#1588).
    private var controlRow: some View {
        HStack(spacing: RapidTheme.Space.xs) {
            modelPicker
                .fixedSize()
            infoButton
        }
    }

    private var modelPicker: some View {
        Menu {
            if loadingCatalog && catalog.isEmpty {
                // A plain ``Text`` in a SwiftUI Menu renders as a
                // disabled, non-selectable row — which is exactly the
                // affordance we want while the catalog is in flight.
                // "Type a model name…" stays reachable so a user who
                // knows what they want isn't blocked on the fetch.
                Text("Fetching models…")
                Divider()
                Button("Type a model name…") { showCustom = true }
            } else if !hasSelectableRows {
                // v0.4.29: previously this state was a dead end — a
                // failed first-load (bootstrapper still installing
                // the sidecar, transient network blip, manual
                // runtime-override deletion) left the user with only
                // "Type alias…" and no way back to a populated picker
                // short of restarting the app. Retry is a one-click
                // rescue.
                //
                // Keyed on "nothing renders", NOT on ``catalog.isEmpty``.
                // A catalog can be non-empty and still produce zero rows —
                // every alias sub-1B with the size filter on, or every one
                // denylisted, and no Quickstart or recommended alias among
                // them. Testing emptiness of the SOURCE rather than of the
                // OUTPUT put that state in the else-branch below, which
                // renders three empty sections and no way out.
                // Honest terminal state: the fetch finished and there is
                // nothing to offer. Non-selectable, and never an
                // indefinite placeholder posing as a model.
                Text("No models available")
                Divider()
                Button("Refresh catalog") {
                    Task { await refreshCatalog(force: true) }
                }
                Button("Type a model name…") { showCustom = true }
            } else {
                // v0.6.9: dropped the separate "Cached" section. Users
                // skim the picker by alias name, not by cache-state —
                // the on-disk affordance is now a small green dot at
                // the right edge of each "All models" row, which keeps
                // a single alphabetical list and removes the cognitive
                // overhead of duplicate aliases appearing in two
                // sections. Cached aliases were rendered once in
                // "Cached" and (since they were filtered out of "All
                // models") never in "All models" — meaning a user who
                // remembered the alias's spelling would scroll right
                // past the cached copy in the wrong section, then
                // grumble that the alias was "missing".
                // No "Refresh catalog" / "Type a model name…" here.
                // Once the catalog HAS models, this menu has exactly one
                // job — pick one — and two maintenance actions sitting
                // above the Quickstart and Recommended sections made the
                // list read as a settings pane. Both remain in the
                // no-selectable-rows branch above, where they are not
                // clutter but the only way out of a failed fetch.
                quickstartSection
                recommendedSection
                allAliasesSection
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: pickerIcon)
                    .foregroundStyle(.secondary)
                    // Decorative — and load-bearing for VoiceOver. Under
                    // `.menuStyle(.button)` SwiftUI promotes EVERY leaf of
                    // the label to its own AXMenuButton element, so leaving
                    // these glyphs visible to AX publishes three duplicate
                    // controls ("Selected" / the alias / "Up And Down
                    // Chevrons") where there is one. Verified by dumping
                    // the AX tree. Do NOT reach for
                    // `.accessibilityElement(children: .ignore)` instead —
                    // it de-duplicates but downgrades the role to AXUnknown.
                    .accessibilityHidden(true)
                // Never renders an internal placeholder as if it were a
                // chosen model — an unresolved alias reads as an
                // instruction instead.
                Text(pickerLabel)
                    .font(RapidFont.secondary)
                    .foregroundStyle(aliasTextStyle)
                    .lineLimit(1)
                    // Middle truncation keeps both ends of a long alias
                    // legible — the family prefix AND the quant suffix,
                    // which is what users disambiguate on. Tail truncation
                    // eats the quant.
                    .truncationMode(.middle)
                Spacer(minLength: 4)
                Image(systemName: "chevron.up.chevron.down")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(chevronStyle)
                    .accessibilityHidden(true)
            }
            .padding(.horizontal, RapidTheme.Space.sm)
            .frame(height: RapidTheme.ControlHeight.small)
            // v1.0: inside the composer this chip sits ON an input
            // surface, so its own bordered pill produced the
            // field-inside-a-field look the redesign is removing. In
            // ``composerStyle`` it is therefore borderless and picks up
            // a fill only on hover; the standalone (non-composer) bar
            // keeps a hairline so it still reads as a control on a bare
            // toolbar. Model selection is a brand moment, so the
            // resolved alias renders in deep amber rather than neutral.
            .background(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous)
                    .fill(pickerFill)
            )
            .overlay(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous)
                    .strokeBorder(pickerStroke, lineWidth: 1)
            )
            // The Spacer stretches the pill across the reserved frame, but
            // SwiftUI hit-tests the label's intrinsic content unless the
            // shape is declared — without this the empty right half of the
            // pill looks clickable and isn't.
            .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        }
        // `.menuStyle(.borderlessButton)` never rendered the label above:
        // it bridges the Menu to an AppKit `SwiftUIPopupButton` (an
        // NSPopUpButton subclass) and transcodes the label into that
        // control's `title` + `image`, discarding the background, the
        // stroke, the padding, the Spacer and the trailing chevron — then
        // tinting the two survivors with RapidApp's accent. That is why
        // the picker read as a blue status label rather than a control:
        // the AppKit subtree under the old style is exactly
        // [NSButtonImageView, NSButtonTextField]. `.menuStyle(.button)`
        // keeps the Menu on SwiftUI's own rendering path, so the chrome
        // survives and SwiftUI installs its own focus ring.
        //
        // The pairing matters: the SDK's deprecation note for
        // `.borderlessButton` suggests `.buttonStyle(.borderless)`, which
        // goes straight back through the AppKit bridge and REPRODUCES the
        // bug (`.accessoryBar` too). It has to be a pure-SwiftUI style.
        //
        // `.plain` rather than the house `.pressable`: a macOS pop-up
        // button stays *highlighted* while its menu is open, it does not
        // shrink, and `configuration.isPressed` stays true for the whole
        // menu-tracking loop — a held 3 % shrink is not the native read.
        // Hover carries the affordance here instead.
        .menuStyle(.button)
        .buttonStyle(.plain)
        // We draw our own disclosure chevron in the label.
        .menuIndicator(.hidden)
        .onHover { pickerHovering = $0 }
        .rapidAnimation(RapidMotion.quick, value: pickerHovering)
        // One vocabulary across every readiness surface (see
        // ``ModelReadiness``): you CHOOSE a model, DOWNLOAD it if
        // needed, START it, and then it is READY. The picker is the
        // control the readiness banner's "Choose a model in the box
        // below" points at, so it has to use that exact word rather
        // than a near-synonym.
        .help(pickerIsUnresolved
              ? "Choose a model"
              : "Model: \(alias) — click to change")
        // Both glyphs are hidden from AX above, so this collapses to a
        // single AXMenuButton announced as "Model, <alias>, pop up button"
        // rather than spelling out SF Symbol names.
        .accessibilityLabel("Model")
        .accessibilityValue(pickerIsUnresolved ? "No model chosen" : alias)
        .accessibilityHint("Choose which model to run")
        .accessibilityIdentifier("ModelPickerBar.ModelMenu")
    }

    /// Chip fill. Borderless-until-hover inside the composer; a faint
    /// standing fill on the standalone bar.
    private var pickerFill: Color {
        if composerStyle {
            return pickerHovering ? RapidTheme.hoverFill : .clear
        }
        return pickerHovering ? RapidTheme.hoverFill : Color.secondary.opacity(0.06)
    }

    /// Chip border. Suppressed inside the composer (the composer already
    /// draws the field edge); a hairline that firms up on hover
    /// elsewhere, so a bezel-less macOS control still reads as live.
    private var pickerStroke: Color {
        if composerStyle {
            return pickerHovering ? RapidTheme.hairlineStrong : .clear
        }
        return pickerHovering ? RapidTheme.hairlineStrong : RapidTheme.hairline
    }

    /// What the composer chip shows. A real alias, or an instruction.
    private var pickerLabel: String {
        ModelDisplayName.configValue(alias: alias) ?? "Choose a model"
    }

    /// True when the chip is showing the instruction rather than a model.
    private var pickerIsUnresolved: Bool {
        ModelDisplayName.isUnresolved(alias)
    }

    /// The alias renders in `.primary`; only the "Choose a model"
    /// placeholder is `.secondary`, like an empty text field.
    ///
    /// Deliberately NOT the accent colour. A real NSPopUpButton draws its
    /// title in `labelColor` on a bezel (Xcode's scheme popup, Mail's
    /// mailbox picker, Finder's arrange-by); accent-coloured text in a
    /// toolbar reads as a link or an active state — which is part of why
    /// the old picker looked like a status badge. The affordance belongs
    /// to the bezel and the chevron, not the text colour. The old blue
    /// was never authored anyway; it leaked from the NSPopUpButton tint.
    private var aliasTextStyle: HierarchicalShapeStyle {
        pickerIsUnresolved ? .secondary : .primary
    }

    /// The disclosure chevron brightens under the pointer — the cheapest
    /// "this is live" cue there is, and the one macOS itself uses on
    /// bezel-less pop-ups.
    private var chevronStyle: HierarchicalShapeStyle {
        pickerHovering ? .primary : .secondary
    }

    /// (i) button next to the picker. Disabled when no alias is
    /// selected (no info to show); the popover anchors to the
    /// button so it appears in-line with the picker bar.
    private var infoButton: some View {
        Button {
            showInfo = true
        } label: {
            Image(systemName: "info.circle")
                .font(.system(size: 14))
        }
        .buttonStyle(.plain)
        .disabled(alias.isEmpty)
        .help("Show model details")
        .accessibilityLabel("Show model details")
        .popover(isPresented: $showInfo, arrowEdge: .top) {
            ModelInfoPopover(
                info: ModelInfoCatalog.info(
                    for: alias,
                    hfRepo: catalog.first(where: { $0.alias == alias })?.hfRepo
                )
            )
            .padding(16)
            .frame(width: 320)
        }
    }

    @ViewBuilder
    private var recommendedSection: some View {
        let rows = recommendedPickRows()
        if !rows.isEmpty {
            Section(recommendedHeaderTitle) {
                ForEach(rows, id: \.entry.alias) { row in
                    recommendedRow(pick: row.pick, entry: row.entry, isPrimary: row.isPrimary)
                }
            }
        }
    }

    /// F-LWT-1: dedicated single-row "Quickstart" section above the
    /// RAM-aware Recommended section. RAM-blind by design — the
    /// Quickstart alias is the same on every Mac (lfm2.5-1b-4bit is
    /// the smallest first-impression install). Persists
    /// across all Quickstart phases including ``.dismissed`` so a
    /// user who closed the Quickstart card can still come back and
    /// one-click install the demo model from the picker.
    ///
    /// Row subtitle ("Smallest model — fastest first install") is
    /// pinned by ``ModelPickerBar.quickstartSubtitle`` so the test
    /// suite catches accidental drift (the section's whole purpose
    /// is to be the bottom-anchored "I just want to try the app"
    /// affordance — the subtitle has to keep that promise).
    /// Would the populated branch of the menu render anything the user can
    /// click?
    ///
    /// Deliberately asks the SAME three helpers the sections themselves
    /// ask — ``quickstartEntry()``, ``recommendedPickRows()`` and the
    /// filter/dedupe/partition chain in ``allAliasesSection`` — rather than
    /// re-deriving "is there anything here" from the raw catalog. A second,
    /// independent notion of emptiness is exactly how the picker ended up
    /// with a branch that believed it had rows while rendering none.
    private var hasSelectableRows: Bool {
        if quickstartEntry() != nil { return true }
        if !recommendedPickRows().isEmpty { return true }
        let filtered = ModelPickerVisibility.filter(
            catalog,
            selectedAlias: alias,
            includeAll: showAllModels
        )
        let deduped = ModelPickerBar.dedupedAllEntries(
            filtered: filtered,
            quickstartRowRendered: quickstartEntry() != nil
        )
        let partition = ModelPickerBar.partitionByFit(deduped, hardware: hardware)
        return !partition.fits.isEmpty || !partition.notFit.isEmpty
    }

    @ViewBuilder
    private var quickstartSection: some View {
        if let entry = quickstartEntry() {
            Section("Quickstart") {
                quickstartRow(entry: entry)
            }
        }
    }

    /// Look up the Quickstart alias in the loaded catalog so the row
    /// shares the cache-state + sizing affordances every other alias
    /// row gets. Returns ``nil`` when the catalog hasn't loaded the
    /// Quickstart alias yet (rapid-mlx older than the 0.6B family
    /// landing — forward/backward-compat skew), in which case the
    /// section is dropped entirely rather than rendering a row that
    /// can't be Started.
    private func quickstartEntry() -> ModelEntry? {
        return catalog.first(where: { $0.alias == QuickstartCoordinator.defaultChoice.alias })
    }

    /// One row for the Quickstart section. Mirrors ``aliasButton``'s
    /// click-to-select behaviour but adds a leading subtitle line so
    /// the row reads as a "demo install" affordance rather than just
    /// another alias entry. Matches the cached/uncached glyph
    /// convention of the All-models rows (Label + leading SF Symbol)
    /// because NSMenuItem inside a SwiftUI ``Menu`` silently drops
    /// trailing icons / multi-line ``VStack`` labels (see the
    /// ``entryLabel`` comment for the same trap).
    private func quickstartRow(entry: ModelEntry) -> some View {
        return Button {
            onUserSelection(entry.alias)
            alias = entry.alias
        } label: {
            Label(
                ModelPickerBar.quickstartRowTitle(alias: entry.alias),
                systemImage: ModelPickerBar.cacheGlyph(cached: entry.cached)
            )
        }
        .disabled(deleting == entry.alias)
        .help(ModelPickerBar.quickstartSubtitle)
        .accessibilityLabel(ModelPickerBar.quickstartRowAccessibilityLabel(
            alias: entry.alias,
            cached: entry.cached
        ))
    }

    /// Subtitle pinned to a single short phrase (~30 chars) so the
    /// "Quickstart" section copy stays anchored even if the file is
    /// later edited around it. The two candidates considered in
    /// F-LWT-1 ("~1.5 min cold install" / "Smallest model — fastest
    /// first impression") collapse to the second because the timing
    /// promise depends on link speed — calling out "smallest /
    /// fastest" is honest regardless of the user's bandwidth.
    static let quickstartSubtitle: String = "Smallest model — fastest first install"

    /// Resolve the safe picker default while the current user is still
    /// eligible for Quickstart. Kept pure so the retired-starter regression
    /// can be pinned without mounting the SwiftUI menu or touching defaults.
    ///
    /// A cached retired model must not outrank the coherent starter merely
    /// because the user chose "Skip for now" in the onboarding sheet.
    static func quickstartEligibleDefault(
        catalog: [ModelEntry],
        eligible: Bool,
        targetAlias: String = QuickstartCoordinator.defaultChoice.alias
    ) -> String? {
        guard eligible, catalog.contains(where: { $0.alias == targetAlias }) else {
            return nil
        }
        return targetAlias
    }

    /// Restore a returning user's last runnable choice without allowing a
    /// retired starter to re-enter any automatic default path.
    static func lastServedDefault(
        catalog: [ModelEntry],
        lastServedAlias: String?,
        excludedAliases: Set<String> = CacheAwareDefault.retiredAutomaticAliases
    ) -> String? {
        guard let alias = lastServedAlias,
              !excludedAliases.contains(alias),
              catalog.contains(where: { $0.alias == alias && $0.cached })
        else { return nil }
        return alias
    }

    /// Pure helper so the row title (which has to ride inside the
    /// single Text NSMenuItem honors) is unit-testable without a
    /// SwiftUI host. Format mirrors ``aliasButtonTitle`` —
    /// ``"<alias> · <subtitle>"`` — so the Quickstart row stays
    /// visually consistent with the other sections.
    static func quickstartRowTitle(alias: String) -> String {
        return "\(alias) · \(quickstartSubtitle)"
    }

    /// VoiceOver label for the Quickstart row.
    static func quickstartRowAccessibilityLabel(alias: String, cached: Bool) -> String {
        let downloaded = cached ? "downloaded" : "not downloaded"
        return "Quickstart: \(alias), \(downloaded). \(quickstartSubtitle)."
    }

    /// Pure helper for the "All models" dedup. When the Quickstart
    /// row is being rendered above, strip the Quickstart alias from
    /// the All-models list so the same alias never appears in two
    /// sections. Lifted out of ``allAliasesSection`` so a
    /// ``ModelPickerBarSectionOrderTests`` case can pin the
    /// invariant directly without standing up a SwiftUI host.
    static func dedupedAllEntries(
        filtered: [ModelEntry],
        quickstartRowRendered: Bool
    ) -> [ModelEntry] {
        guard quickstartRowRendered else { return filtered }
        return filtered.filter { $0.alias != QuickstartCoordinator.defaultChoice.alias }
    }

    /// Order the "All models" list: downloaded (cached) aliases first,
    /// then alphabetical within each group. Surfacing pulled models at
    /// the top means a downloaded alias the user is looking for is not
    /// buried mid-alphabet behind a small green dot in a ~150-row list.
    /// Keeps the single-list shape (each alias renders once, no separate
    /// "Cached" section) rather than reintroducing a duplicate section.
    /// Mirrors ``ModelCatalog.load``'s own cached-first order, which the
    /// alphabetical display step previously discarded.
    static func orderedAllModels(_ entries: [ModelEntry]) -> [ModelEntry] {
        entries.sorted { lhs, rhs in
            if lhs.cached != rhs.cached { return lhs.cached && !rhs.cached }
            return lhs.alias.localizedCaseInsensitiveCompare(rhs.alias) == .orderedAscending
        }
    }

    /// "Recommended for your 18 GB Mac" header. RAM is rounded the
    /// same way ``MacHardware.shortDescription`` rounds it (Int(.
    /// rounded())) so the picker title agrees with the macOS About
    /// dialog the user can verify from.
    private var recommendedHeaderTitle: String {
        Self.recommendedHeaderTitle(physicalRAMGB: hardware.physicalRAMGB)
    }

    /// Pure helper so the header copy is pinnable by tests without
    /// standing up a SwiftUI host. Public-static mirrors the rest of
    /// the file's "lift the truth table out" pattern.
    static func recommendedHeaderTitle(physicalRAMGB: Double) -> String {
        let gb = max(1, Int(physicalRAMGB.rounded()))
        return "Recommended for your \(gb) GB Mac"
    }

    /// v0.6.9: a single alphabetical "All models" list, no separate
    /// "Cached" section. The cache-state affordance is now a small
    /// green dot at the right edge of each row (see ``aliasButton``).
    /// Sorting on `alias` (case-insensitive) so capitalisation
    /// differences in the source aliases.json don't split visually
    /// similar entries.
    ///
    /// cycle-7: sub-1B aliases are filtered out unless the user opts
    /// in via ``Settings → Models → Show small models``. The
    /// currently-selected alias is exempt from the filter (so a user
    /// who has somehow picked a tiny model can still see it).
    ///
    /// The filter no longer announces itself here. A trailing "N small
    /// models hidden" row sat in the same visual weight as the alias
    /// rows directly above it and began with a digit, so it read as a
    /// broken model entry rather than as a note about the list — and
    /// AppKit sized the menu to the aliases and ellipsised it, cutting
    /// exactly the tail that named Settings. The toggle stays
    /// discoverable where it lives, in Settings → Models.
    @ViewBuilder
    private var allAliasesSection: some View {
        let filtered = ModelPickerVisibility.filter(
            catalog,
            selectedAlias: alias,
            includeAll: showAllModels
        )
        // F-LWT-1: dedupe — if the Quickstart row is rendered above,
        // strip the Quickstart alias from the All-models list so the
        // same alias doesn't appear in two sections. The
        // ``dedupedAllEntries`` helper mirrors the visibility check
        // in ``quickstartSection`` (catalog has the alias) so we
        // only suppress when the row is actually being rendered
        // above. Selected-alias safety: even if the user has
        // selected the Quickstart alias, the dedup is safe — the
        // Quickstart row above is clickable and carries the same
        // "select this alias" semantics.
        let deduped = ModelPickerBar.dedupedAllEntries(
            filtered: filtered,
            quickstartRowRendered: quickstartEntry() != nil
        )
        let partition = ModelPickerBar.partitionByFit(deduped, hardware: hardware)
        let sorted = ModelPickerBar.orderedAllModels(partition.fits)
        let notFit = ModelPickerBar.orderedAllModels(partition.notFit)
        if !sorted.isEmpty {
            Section("All models") {
                ForEach(sorted) { entry in
                    aliasButton(entry)
                }
            }
        }
        if !notFit.isEmpty {
            Section("Not fit for this Mac") {
                ForEach(notFit) { entry in
                    aliasButton(entry, notFit: true)
                }
            }
        }
    }

    /// Keep oversized aliases discoverable without presenting them beside
    /// models this Mac can actually run. Download remains available; the
    /// launch-time live-memory guard is still the final safety boundary.
    static func partitionByFit(
        _ entries: [ModelEntry],
        hardware: MacHardware
    ) -> (fits: [ModelEntry], notFit: [ModelEntry]) {
        var fits: [ModelEntry] = []
        var notFit: [ModelEntry] = []
        for entry in entries {
            let fit = ModelSizing.classify(ModelSizing.estimate(alias: entry.alias), on: hardware)
            if fit == .tooBig {
                notFit.append(entry)
            } else {
                fits.append(entry)
            }
        }
        return (fits, notFit)
    }

    /// One row inside the "Recommended for your N GB Mac" section.
    /// Renders the Best pick / Faster label + short blurb + alias. NO
    /// warning icons — the tier table guarantees the alias fits the host
    /// (per the operator-curated invariant in ``RAMBucketedDefault``).
    ///
    /// Codex r1 on PR #196 flagged a real but accepted tension here:
    /// ``ModelSizing.classify`` is a conservative heuristic and rates a
    /// handful of curated picks as ``.tooBig`` on the bottom edge of their
    /// tier (e.g. the real-7.6 GB ``bonsai-27b-2bit`` reads as ~14.8 GB →
    /// ``.tooBig`` on 16 GB). The picker's "All models" branch keeps the
    /// existing ``.tooBig`` click-gate so a user can't click through to a
    /// wildly-oversized alias by browsing. The recommendation rows
    /// intentionally bypass that gate (via ``RAMBucketedDefault/isRecommendedPick``)
    /// because the curated table is the operator's source-of-truth (per
    /// user spec line: "trust the curated tier table — if it's recommended,
    /// it fits"). If a curated entry ever turns out to be a genuine OOM
    /// rather than a heuristic disagreement, the fix is to retune the tier
    /// in ``RAMBucketedDefault`` — never to put a warning glyph back here.
    private func recommendedRow(pick: RAMBucketedDefault.Pick, entry: ModelEntry, isPrimary: Bool) -> some View {
        // v0.6.9-rc: SwiftUI Menu wraps each Button as an NSMenuItem,
        // and NSMenuItem only honours the FIRST Text inside the
        // Button's label — HStack/Spacer/Circle/background fills are
        // silently dropped. The fix is to fold everything into a single
        // Text() via the helper below, with a leading SF Symbol
        // (NSMenuItem DOES honour Label's leading image) carrying the
        // download-state glyph.
        let isSelected = ModelPickerBar.roleRowIsSelected(
            selectedAlias: alias,
            rowAlias: entry.alias
        )
        let baseTitle = ModelPickerBar.recommendedRowMenuTitle(
            label: isPrimary ? "Recommended" : "Faster",
            alias: entry.alias
        )
        let title = isSelected
            ? ModelPickerBar.currentSelectionTitle(baseTitle)
            : baseTitle
        // Tagline = "best pick / faster alternative" + the measured
        // capability and (where we have it) tok/s. Surfaced as the row's
        // NSMenuItem tooltip via ``.help(_)`` and its accessibility label.
        let tagline = ModelPickerBar.recommendedTagline(pick: pick, isPrimary: isPrimary)
        // The tagline rides the row's native NSMenuItem tooltip via
        // ``.help(_)`` (below) — the reliable way to expose it inside the
        // open dropdown, since a SwiftUI ``.popover`` attached to a row
        // inside a ``Menu`` is silently dropped once the dropdown becomes
        // an NSMenu — and doubles as the row's accessibility label.
        return Button {
            onUserSelection(entry.alias)
            alias = entry.alias
        } label: {
            // The image slot carries download state, uniformly with
            // every other row in this dropdown — the Recommended
            // section used to be the only one that never said whether
            // the model was already on disk, which is precisely the
            // section where that matters most.
            Label(title, systemImage: ModelPickerBar.cacheGlyph(cached: entry.cached))
        }
        .disabled(deleting == entry.alias)
        // The recommendation shows only the curated capability / speed
        // tagline — the single source of truth for a pick. The per-axis
        // standard-benchmark meters live in "All models" (Settings), so
        // the recommendation never shows two conflicting sets of numbers
        // for the same model.
        .help(tagline)
        .accessibilityLabel("\(entry.alias). \(tagline)")
    }

    /// Pure helper so the menu title is pinnable by tests. " — " em
    /// dash separator anchors the "Recommended" / "Faster" label on the
    /// left while the alias trails on the right.
    static func recommendedRowMenuTitle(label: String, alias: String) -> String {
        return "\(label) — \(alias)"
    }

    /// Tooltip / caption tagline for a recommended row: whether it's the
    /// best pick or the faster alternative, plus the measured capability
    /// and (where a local measurement exists) decode tok/s. A pick with a
    /// ``caveat`` (e.g. a chat specialist) shows that word in place of the
    /// capability %. Pure so tests can pin the copy without a SwiftUI host.
    static func recommendedTagline(pick: RAMBucketedDefault.Pick, isPrimary: Bool) -> String {
        let lead = isPrimary ? "Best pick for your Mac" : "Faster, lighter alternative"
        var parts = [lead]
        if let caveat = pick.caveat {
            if let tps = pick.tokensPerSec {
                parts.append("~\(Int(tps.rounded())) tok/s")
            }
            parts.append(caveat)
        } else {
            parts.append("\(pick.capabilityPct)% capability")
            if let tps = pick.tokensPerSec {
                parts.append("~\(Int(tps.rounded())) tok/s")
            }
        }
        return parts.joined(separator: " · ")
    }

    /// Mark the row whose alias is the one currently in the picker.
    ///
    /// This used to be a leading `checkmark` glyph, but NSMenuItem
    /// honours exactly one image per row and that slot is now carrying
    /// the download-state glyph — which is the fact a user actually
    /// needs before committing to a multi-gigabyte pull. A word in the
    /// title survives the NSMenu collapse and, unlike a bare tick, says
    /// what it means: this is the current model.
    static func currentSelectionTitle(_ title: String) -> String {
        "\(title) (current)"
    }

    /// Pure helper so tests can pin the "should this recommended row paint
    /// amber?" rule without standing up a SwiftUI host. The rule is
    /// trivially equality, but lifting it out matches the project's
    /// "no truth table inside a view" convention so the test suite
    /// catches a future drift. The empty-alias guard rules out the
    /// transient "catalog still loading, no alias picked yet" state
    /// where every recommended row would otherwise paint amber for "".
    static func roleRowIsSelected(selectedAlias: String, rowAlias: String) -> Bool {
        return !selectedAlias.isEmpty && selectedAlias == rowAlias
    }

    /// The one download-state glyph for every row in the picker's
    /// dropdown.
    ///
    /// The bar had grown three different vocabularies for the same
    /// fact: `circle.fill` / `circle.dashed` in the Quickstart and
    /// "All models" rows, `checkmark.circle.fill` /
    /// `icloud.and.arrow.down` on the closed picker's own label, and
    /// nothing at all on the Recommended rows — so the one section that
    /// asks the user to commit to a multi-gigabyte pull was the one
    /// section that never said whether the pull had already happened.
    ///
    /// `icloud.and.arrow.down` is the de-facto standard for "not on this
    /// device yet" — it is what Music, TV, Podcasts, Photos and Files
    /// all use — and this file already reached for it twice. A dashed
    /// circle is not a download idiom; it reads as a placeholder. The
    /// downloaded side keeps `checkmark.circle.fill`, which is what the
    /// closed picker already shows for exactly this state, so the
    /// dropdown and the control it drops from finally agree.
    static func cacheGlyph(cached: Bool) -> String {
        cached ? "checkmark.circle.fill" : "icloud.and.arrow.down"
    }

    /// `.task(id:)` key for the picker's catalog snapshot: re-fetch
    /// when the engine binary appears/changes, AND whenever the set of
    /// models on disk changes anywhere in the app.
    struct PickerCatalogKey: Equatable {
        let binaryPath: URL?
        let cacheGeneration: UInt
        let refreshEnabled: Bool

        init(binaryPath: URL?, cacheGeneration: UInt, refreshEnabled: Bool = true) {
            self.binaryPath = binaryPath
            self.cacheGeneration = cacheGeneration
            self.refreshEnabled = refreshEnabled
        }
    }

    /// Renders one alias row in the "All models" alphabetical list.
    ///
    /// v0.6.9: the row is now a single alias line with a small green
    /// dot at the right edge when the alias's HF cache directory
    /// exists. The previous build (a) split rows into a separate
    /// "Cached" section and (b) suffixed dimmed "may not fit on your
    /// Mac" text on borderline/tooBig entries — both removed. Fit
    /// classification still drives the Start-click alert (handled
    /// elsewhere) but no longer surfaces hazard text in the picker.
    /// Rows are no longer disabled by ``.tooBig``: a user can pick
    /// any alias from the picker, and the Start CTA wires the actual
    /// guardrail via the confirmation alert.
    private func aliasButton(_ entry: ModelEntry, notFit: Bool = false) -> some View {
        let bucket = ModelPickerVisibility.qualityBucket(for: entry.alias)
        let footprint = ModelSizing.estimate(alias: entry.alias)
        return Button {
            onUserSelection(entry.alias)
            alias = entry.alias
        } label: {
            Label(
                ModelPickerBar.aliasButtonTitle(alias: entry.alias, bucket: bucket)
                    + (notFit ? " · needs ~\(Int(footprint.totalGB.rounded())) GB" : ""),
                systemImage: ModelPickerBar.cacheGlyph(cached: entry.cached)
            )
        }
        .disabled(deleting == entry.alias)
        // cycle-10: rows in the .tiny (< 1B) and .small (>= 1B,
        // cycle-11 < 3B) buckets get a richer hover tooltip surfacing
        // the multi-turn-contradiction risk. 3B+ rows fall back to
        // the cache-state cue. Combined into one string via
        // ``qualityRowHelpText`` so the help() modifier stays a single
        // value and tests can pin the multi-line copy.
        .help(notFit
            ? "Estimated to need about \(Int(footprint.totalGB.rounded())) GB; this Mac has \(Int(hardware.usableRAMGB.rounded())) GB available for models. You can download it, but starting it may fail or destabilize the Mac."
            : ModelPickerBar.aliasRowHelpText(
                alias: entry.alias,
                bucket: bucket,
                cached: entry.cached
            )
        )
        // VoiceOver: the leading SF Symbol carries the cache cue
        // sighted users see, but the symbol's image hint isn't read,
        // so we hand VoiceOver an explicit composed label. Restores
        // the "Downloaded" / "Not downloaded" coverage the removed
        // `cacheStateDot` helper had wired via `.accessibilityLabel`.
        // cycle-10: VoiceOver also gets the sticker word so blind
        // users get the same warning sighted users see in the suffix.
        .accessibilityLabel(
            ModelPickerBar.aliasRowAccessibilityLabel(
                alias: entry.alias,
                cached: entry.cached,
                bucket: bucket
            ) + (notFit ? ". Not fit for this Mac. Needs about \(Int(footprint.totalGB.rounded())) gigabytes" : "")
        )
        // v0.5.2: right-click → "Delete from disk". Cached rows
        // only — uncached models would just trigger "alias not
        // found in cache" from the CLI, so we hide the affordance
        // rather than let the user discover the no-op the hard
        // way. The currently-serving alias is also off-limits
        // (rapid-mlx holds the weights mmap'd; rm would error).
        .contextMenu {
            if entry.cached && server.servingAlias != entry.alias && deleting != entry.alias {
                Button(role: .destructive) {
                    pendingDeletion = entry
                } label: {
                    if let size = entry.sizeOnDisk {
                        Text("Delete from disk (\(size))")
                    } else {
                        Text("Delete from disk")
                    }
                }
            }
            // v0.5.7: side-car download affordance. Only offered when
            // the alias isn't cached and isn't already being pulled.
            // v0.6.9 drops the `.tooBig` gate here — the Start CTA
            // alert handles that case end-to-end, and right-click is
            // a power-user affordance.
            if !entry.cached
                && !downloads.isDownloading(entry.alias) {
                Button {
                    downloads.startDownload(alias: entry.alias, hfPath: entry.hfRepo)
                } label: {
                    Text("Download in background")
                }
            }
        }
    }

    /// v0.6.9-rc: NSMenuItem drops trailing icons inside HStack
    /// labels. The previous shape (Text(alias) + Spacer + Circle())
    /// rendered as just the alias name in the dropdown — the green
    /// dot never reached the user. Fix is a leading SF Symbol via
    /// Label (NSMenuItem renders Label's icon at the gutter) so
    /// cached aliases get a visible cue distinct from uncached ones.
    /// Filled circle = on disk; dashed circle = will download on
    /// Start. Symbol-only differentiation works even when NSMenu
    /// strips the green tint that SwiftUI would otherwise apply.
    ///
    /// cycle-10: the title string now carries an optional bucket
    /// suffix when the alias is in the ``.small`` / ``.tiny`` quality
    /// bucket (< 3B parsed params after the cycle-11 strict-bound
    /// tighten — ``llama3-3b-4bit`` and other 3B aliases stay clean).
    /// #348: the suffix is bucket-distinct (``.tiny`` → " · tiny",
    /// ``.small`` → " · small") so the data-model split shows up in
    /// the picker instead of both buckets sharing the "· tiny" label.
    /// NSMenuItem honours one Text inside the Label so the suffix has
    /// to ride in the title rather than as a trailing badge view;
    /// pulled into ``aliasButtonTitle`` for testability.
    @ViewBuilder
    private func entryLabel(_ entry: ModelEntry, bucket: ModelPickerVisibility.QualityBucket) -> some View {
        Label(
            ModelPickerBar.aliasButtonTitle(alias: entry.alias, bucket: bucket),
            systemImage: ModelPickerBar.cacheGlyph(cached: entry.cached)
        )
    }

    /// Pure helper — compose the alphabetical-row title from the
    /// alias and its quality bucket. The cycle-10 sticker is appended
    /// via a leading " · " separator so the alias remains the visual
    /// anchor and the suffix reads as a footnote rather than a
    /// hyphenated alias rename. Lifted out so tests can pin the
    /// per-bucket output without standing up a SwiftUI host.
    ///
    /// #133: when the alias is non-``.known`` per
    /// ``ToolUseCapability`` (broken or unknown), a second badge is
    /// appended after the quality sticker. The two stickers compose:
    /// a 1-3B model that is ``.broken`` reads
    /// ``"<alias> · small · no tools"`` (the ``.tiny`` sub-1B band
    /// would read ``"<alias> · tiny · no tools"`` — #348 split the
    /// suffix per bucket); a sub-3B model that is ``.unknown`` reads
    /// ``"<alias> · small · tools unverified"`` (or "· tiny" for
    /// sub-1B) (FU-9 — softer copy for the unbenched case so we
    /// don't declare an untested alias broken). NSMenuItem honours only
    /// the first ``Text`` inside a SwiftUI ``Menu`` button label so
    /// the badge has to ride in the title string itself (the same
    /// trade-off the quality sticker made in cycle-10). Order is
    /// quality → tools so the size cue stays adjacent to the alias
    /// name; both badges share the leading " · " separator.
    static func aliasButtonTitle(alias: String, bucket: ModelPickerVisibility.QualityBucket) -> String {
        var title = alias
        if let suffix = ModelPickerVisibility.qualityStickerSuffix(for: bucket) {
            title += " \(suffix)"
        }
        return title
    }

    /// Pure helper — compose the row hover tooltip from the
    /// quality-sticker copy + the cache-state cue. Stays a single
    /// String so SwiftUI's ``.help()`` modifier (one value, not a
    /// view builder) can consume it. Empty string when no quality
    /// warning AND no cache info — SwiftUI ``.help("")`` is a no-op
    /// which is the desired "nothing to surface" outcome.
    ///
    /// #133: when the alias is badged for no-tools, the tooltip
    /// surfaces the WHY ("Tool calls are unverified / unreliable on
    /// this model.") so a sighted user hovering the row understands
    /// the badge before they click. Composed onto a separate line
    /// after the quality-tooltip + cache hint so all three signals
    /// stack without crowding.
    static func aliasRowHelpText(bucket: ModelPickerVisibility.QualityBucket, cached: Bool) -> String {
        return aliasRowHelpText(alias: "", bucket: bucket, cached: cached)
    }

    /// #133 overload — same shape as the original two-arg helper but
    /// adds the tools-badge tooltip when ``alias`` is non-``.known``.
    /// Kept as an overload so existing callers (and the existing
    /// ``ModelPickerBarTests`` suite) still resolve against the
    /// two-arg form without churn.
    ///
    /// FU-9: the appended line now varies by ``ToolUseConfidence``
    /// so the hover tooltip mirrors the per-state badge wording.
    /// ``.broken`` reads as "ignores tool calls" (empirical evidence
    /// earned the strong copy); ``.unknown`` reads as "tools are
    /// unverified" (we don't have a signal either way; don't
    /// declare the alias broken). Both states still suppress the
    /// empty-state chip row — the tooltip just stops conflating
    /// "tested and bad" with "untested".
    static func aliasRowHelpText(
        alias: String,
        bucket: ModelPickerVisibility.QualityBucket,
        cached: Bool
    ) -> String {
        let cacheHint = cached ? "Already downloaded" : "Will download on Start"
        let base = ModelPickerVisibility.qualityRowHelpText(for: bucket, cacheHint: cacheHint)
        let confidence = ToolUseCapability.confidence(for: alias)
        let toolsLine: String
        switch confidence {
        case .known:
            return base
        case .broken:
            toolsLine = "Tools are off on this model — empirical bench shows it ignores tool calls and hallucinates the answer. Chips that promise tools are hidden on the empty-state hero."
        case .unknown:
            toolsLine = "Tool calls are unverified on this model — we have no bench signal yet. Chips that promise tools are hidden on the empty-state hero until verified."
        }
        // Empty alias resolves to .unknown above, but
        // ``shouldBadgeAliasForToolUse`` defensively suppresses for
        // the picker placeholder row; mirror that here so the
        // two-arg overload (alias="") never inherits the FU-9
        // tooltip when no model is selected.
        guard !alias.isEmpty else { return base }
        if base.isEmpty {
            return toolsLine
        }
        return "\(base)\n\(toolsLine)"
    }

    /// Pure helper — compose the VoiceOver label for one alias row
    /// so blind users get the same quality-warning signal sighted
    /// users see in the suffix. Format: ``"<alias>, <downloaded?>,
    /// <quality cue>"`` — comma-separated tokens are how AppKit /
    /// VoiceOver consumes a composed accessibility label.
    ///
    /// #133: when the alias has a non-``nil`` badge per
    /// ``ToolUseCapability.badgeLabel(forAlias:)``, the composed
    /// label appends the per-state copy so blind users get the same
    /// signal sighted users see in the alias-row suffix.
    ///
    /// FU-9: the appended token now varies by ``ToolUseConfidence``
    /// (", no tools" for ``.broken``; ", tools unverified" for
    /// ``.unknown``) so VoiceOver mirrors the visible-label split
    /// instead of pinning both states to "no tools".
    static func aliasRowAccessibilityLabel(
        alias: String,
        cached: Bool,
        bucket: ModelPickerVisibility.QualityBucket
    ) -> String {
        let downloaded = cached ? "downloaded" : "not downloaded"
        let qualityPart: String
        // #348: mirror the visible-suffix split (``.tiny`` → "tiny",
        // ``.small`` → "small") so VoiceOver users get the same
        // bucket signal sighted users see in ``qualityStickerSuffix``.
        // The trailing "may contradict itself in multi-turn chat"
        // clause stays bucket-agnostic — it covers both buckets per
        // the unified tooltip in ``qualityStickerTooltip``.
        switch bucket {
        case .tiny:
            qualityPart = "\(alias), \(downloaded), tiny model — may contradict itself in multi-turn chat"
        case .small:
            qualityPart = "\(alias), \(downloaded), small model — may contradict itself in multi-turn chat"
        case .midOrLarger:
            qualityPart = "\(alias), \(downloaded)"
        }
        return qualityPart
    }

    /// Title string for the delete-from-disk confirmation dialog. The
    /// size is front-loaded so a glance at the title is enough to
    /// gauge the cost; the alias follows in monospaced styling-ish
    /// quotes for visual separation. Pulled out as `static` so the
    /// test suite pins both shapes (with / without a measured size)
    /// without standing up the picker.
    static func deletionTitle(for entry: ModelEntry?) -> String {
        guard let entry = entry else {
            return "Delete this model?"
        }
        if let size = entry.sizeOnDisk {
            return "Delete \"\(entry.alias)\"? This frees \(size)."
        }
        return "Delete \"\(entry.alias)\"?"
    }

    private var pickerIcon: String {
        if let hit = catalog.first(where: { $0.alias == alias }) {
            return hit.cached ? "checkmark.circle.fill" : "icloud.and.arrow.down"
        }
        return "questionmark.circle"
    }

    /// Pure derivation of the subtitle from server state + observed
    /// startup activity. Extracted so unit tests can pin the copy
    /// without standing up a real ``ServerManager``.
    ///
    /// The pill is a SUMMARY, not a second progress bar: the full
    /// byte / speed / ETA read-out lives in the chat's startup banner
    /// (the one detailed home — 2026-07 dedup, the screenshot showed
    /// the identical numbers rendered twice). Here: one word of state
    /// above (``stateLabel``) and, while downloading, "12% · 4 min
    /// left". The elapsed clock is appended by the caller's
    /// TimelineView as before.
    static func progressSubtitle(
        state: ServerState,
        activity: DownloadProgress.StartupActivity,
        fraction: Double?,
        eta: String?
    ) -> String? {
        guard case .starting = state else { return nil }
        switch activity {
        case .starting:
            // v0.4.36: never nil during .starting — the copy always
            // shows so the elapsed clock has a slot to render
            // alongside, and the user gets immediate feedback that
            // the start command was received.
            return "Starting the model…"
        case .loading:
            return "Loading into memory…"
        case .warmingUp:
            return "Warming up…"
        case .downloading:
            guard let fraction else { return "Downloading…" }
            let pct = max(0, min(100, Int((fraction * 100).rounded())))
            // if-let rather than Optional.map: a closure literal inside
            // this @MainActor type inherits main-actor isolation, and
            // this helper is documented as callable from anywhere
            // (SIGTRAP via dispatch_assert_queue when a nonisolated
            // test called it — the closure was the only isolated bit).
            if let eta { return "\(pct)% · \(eta)" }
            return "\(pct)%"
        }
    }

    private var startStopButtons: some View {
        HStack(spacing: 8) {
            if server.isOperating {
                ProgressView()
                    .controlSize(.small)
                // v0.4.28: bare spinner read as "click was lost" on a
                // multi-second graceful shutdown. The starting path
                // already gets its label from the state badge
                // ("Starting / Downloading / Loading <alias>"), so we
                // only surface explicit text here for the stop path —
                // i.e. when the server is still `.ready` but we're
                // mid-``server.stop()``.
                if isStoppingInFlight {
                    Text("Stopping…")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            }
            if isStartable {
                // v0.5: when the action isn't available yet (no model
                // picked, or rapid-mlx missing) we render a *bordered*
                // button rather than a disabled ``.borderedProminent``.
                // A disabled prominent button washes its label to a
                // near-invisible white-on-faint-blue; the bordered
                // variant keeps "Download & start" dark and legible
                // while still clearly reading as inactive.
                if canStart {
                    // v0.6: custom amber fill + dark label (vs
                    // .borderedProminent, which renders white text that's
                    // unreadable on #EFA23A). Matches the New chat CTA.
                    Button {
                        handleStartTap()
                    } label: {
                        Label(startButtonLabel, systemImage: startButtonIcon)
                            .font(.callout.weight(.semibold))
                            .foregroundStyle(startLabelColor)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(
                                RoundedRectangle(cornerRadius: 8, style: .continuous)
                                    .fill(startButtonTint)
                            )
                    }
                    .buttonStyle(.pressable)
                    .help(startButtonHelp)
                    .accessibilityIdentifier("ModelPickerBar.PrimaryButton")
                } else {
                    Button {
                        handleStartTap()
                    } label: {
                        Label(startButtonLabel, systemImage: startButtonIcon)
                    }
                    .buttonStyle(.bordered)
                    .disabled(true)
                    .help(startButtonHelp)
                    .accessibilityIdentifier("ModelPickerBar.PrimaryButton")
                }
            } else {
                Button {
                    Task { await server.stop() }
                } label: {
                    // "Stop model" (not a bare "Stop") disambiguates this
                    // server-unload control from the composer's "Stop
                    // response" circle, which halts only the current
                    // generation. During streaming both are on screen; a
                    // mis-click on a bare "Stop" here unloaded the model
                    // and forced a full reload on the next turn.
                    Label("Stop model", systemImage: "stop.fill")
                }
                .buttonStyle(.bordered)
                .disabled(server.isOperating)
                .help("Stop the running model and free its memory")
                .accessibilityIdentifier("ModelPickerBar.PrimaryButton")
            }
        }
    }

    /// True while ``server.stop()`` is mid-flight. ServerManager keeps
    /// the state at ``.ready`` (or ``.crashed``) until the child has
    /// drained, so an ``isOperating`` flag on top of a ready-ish state
    /// is the canonical signal for "shutting down right now."
    private var isStoppingInFlight: Bool {
        ModelPickerBar.isStoppingInFlight(state: server.state, isOperating: server.isOperating)
    }

    /// Pure helper so tests can pin the (state, isOperating) → "show
    /// Stopping…" truth table without standing up a live
    /// ``ServerManager``. Mirrors the ``MainAreaBranch`` pattern in
    /// ``ContentView``.
    static func isStoppingInFlight(state: ServerState, isOperating: Bool) -> Bool {
        guard isOperating else { return false }
        switch state {
        case .ready, .crashed: return true
        case .starting, .idle, .stopped, .missing: return false
        }
    }

    /// Whether the selected alias is locally cached. Drives the
    /// Start button's label/icon/tint so a user picking an
    /// unseen model has correct expectations BEFORE they click —
    /// the previous build silently triggered a multi-minute HF
    /// download with no warning.
    private var selectedAliasIsCached: Bool {
        catalog.first(where: { $0.alias == alias })?.cached ?? false
    }

    private var startButtonLabel: String {
        selectedAliasIsCached ? "Start" : "Download & start"
    }

    private var startButtonIcon: String {
        selectedAliasIsCached ? "play.fill" : "icloud.and.arrow.down"
    }

    /// Start-button fill. v0.6: the brand amber #EFA23A for a cached
    /// start (matches the New chat CTA); the uncached download click
    /// stays orange to flag "this will take a while."
    private var startButtonTint: Color {
        selectedAliasIsCached ? RapidTheme.amber : .orange
    }

    /// Legible label colour for the Start fill: dark graphite on the
    /// light amber, white on the darker download-orange.
    private var startLabelColor: Color {
        selectedAliasIsCached
            ? Color(nsColor: NSColor(deviceWhite: 0.13, alpha: 1.0))
            : .white
    }

    private var startButtonHelp: String {
        // F-LWT-1: surface the reason when the CTA is gated by a
        // Quickstart flow. Without this the user sees a greyed-out
        // button with the normal "Download from HF, then start"
        // tooltip and reads the gate as a bug.
        if ModelPickerBar.isQuickstartInFlight(phase: quickstart?.phase) {
            return "Quickstart download in progress"
        }
        return selectedAliasIsCached
            ? "Start the model (cached locally)"
            : "Download from Hugging Face, then start. Can take several minutes on first run."
    }

    // MARK: - v0.6.9 tooBig Start guard

    /// Handle a click on the Start (or Download & start) button.
    ///
    /// If ``ModelSizing.classify`` returns ``.tooBig`` for the
    /// currently selected alias on this hardware, surface a
    /// confirmation alert before the spawn so the user has to opt in
    /// to the OOM risk. ``.recommended`` and ``.borderline`` skip the
    /// alert and start directly — borderline is "tight but should
    /// run," and the picker already biases the recommended-section
    /// rows to comfortable fits.
    ///
    /// The alert state is held by ``pendingTooBigStart``; the actual
    /// ``server.start(alias:)`` call fires either immediately (safe
    /// fit) or from the alert's "Start anyway" handler.
    private func handleStartTap() {
        let trimmed = alias.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let fit = ModelSizing.classify(
            ModelSizing.estimate(alias: trimmed),
            on: hardware
        )
        // A recommended pick trusts the curated table's measured
        // footprint over ModelSizing's estimate (which over-states
        // low-bit / MoE models), so it skips the .tooBig gate — the
        // table already vetted it fits this Mac's RAM tier.
        let isRecommended = RAMBucketedDefault.isRecommendedPick(
            alias: trimmed, physicalRAMGB: hardware.physicalRAMGB)
        if fit == .tooBig && !isRecommended {
            pendingTooBigStart = trimmed
            return
        }
        let hfPath = catalog.first(where: { $0.alias == trimmed })?.hfRepo
        // Launch flags are applied inside ServerManager.start (one choke
        // point for every start path), RAM-gated to the recommended pick.
        Task { await server.start(alias: trimmed, hfPath: hfPath) }
    }

    /// Alert title shown when the user tries to Start a ``.tooBig``
    /// alias. Front-loads the host's physical RAM and the alias name
    /// so the user reads "alias-x likely won't fit your 18 GB Mac"
    /// at a glance — same pattern as the delete-confirmation dialog
    /// (size-first, then identity).
    private var tooBigAlertTitle: String {
        ModelPickerBar.tooBigAlertTitle(
            alias: pendingTooBigStart ?? "",
            physicalRAMGB: hardware.physicalRAMGB
        )
    }

    /// Alert body: estimated footprint vs. available RAM + a single
    /// warning sentence. Pulled to a helper so the copy can be pinned
    /// from ``ModelPickerBarTests`` without a SwiftUI host.
    private func tooBigAlertMessage(for aliasToStart: String) -> String {
        ModelPickerBar.tooBigAlertMessage(
            alias: aliasToStart,
            footprint: ModelSizing.estimate(alias: aliasToStart),
            hardware: hardware
        )
    }

    /// Pure helper — produces the title string from alias + RAM. Test
    /// suite pins both the at-1-GB clamp and the rounded display so
    /// future copy churn can't silently regress.
    static func tooBigAlertTitle(alias: String, physicalRAMGB: Double) -> String {
        let gb = max(1, Int(physicalRAMGB.rounded()))
        if alias.isEmpty {
            return "This model likely won't fit your \(gb) GB Mac"
        }
        return "\(alias) likely won't fit your \(gb) GB Mac"
    }

    /// Pure helper — produces the body string from the footprint and
    /// hardware. Two sentences: (1) estimated need vs. usable RAM,
    /// (2) consequence of pressing on. Kept terse — alerts that
    /// span multiple paragraphs read as noise the user just dismisses.
    static func tooBigAlertMessage(
        alias: String,
        footprint: ModelSizing.Footprint,
        hardware: MacHardware
    ) -> String {
        let usable = Int(hardware.usableRAMGB.rounded())
        let total = Int(footprint.totalGB.rounded())
        let firstSentence: String
        if footprint.paramsBillions != nil && total > 0 {
            firstSentence = "Estimated need ≈ \(total) GB; your Mac has about \(usable) GB available for the model."
        } else {
            // Unknown parameter count — skip the numeric comparison
            // rather than report "0 GB needed" which reads as a bug.
            firstSentence = "Estimated footprint exceeds your Mac's usable RAM (\(usable) GB)."
        }
        return "\(firstSentence) Continuing may cause swap thrashing, performance crashes, or system lock-up."
    }

    /// v1.0.1: the last native-blue surface in the app.
    ///
    /// ``.roundedBorder`` drew AppKit's bezel plus the system focus
    /// ring, so the brightest blue in the product appeared the moment
    /// this field took focus — on a sheet whose primary action is
    /// amber. ``RapidTextField`` draws an amber 2px focus border
    /// instead and keeps focus, Return, Escape, and VoiceOver exactly
    /// as they were: Return still commits via the field's `onSubmit`
    /// AND the default-action button, Escape still cancels.
    private var customAliasSheet: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
            SectionHeader(
                "Custom model",
                subtitle: "Type a model name or a Hugging Face path. New models download when you press Start.",
                emphasis: .page
            )

            RapidTextField(
                placeholder: "e.g. qwen3.6-35b",
                text: $customDraft,
                onSubmit: commitCustomAlias
            )

            HStack(spacing: RapidTheme.Space.sm) {
                Spacer()
                Button("Cancel") { showCustom = false }
                    .buttonStyle(.rapidSecondary)
                    .keyboardShortcut(.cancelAction)
                Button("Use", action: commitCustomAlias)
                    .buttonStyle(RapidPrimaryButtonStyle(
                        height: RapidTheme.ControlHeight.medium
                    ))
                    .keyboardShortcut(.return, modifiers: [])
            }
            .padding(.top, RapidTheme.Space.xs)
        }
        .padding(RapidTheme.Space.xl)
        .frame(width: 380)
        .background(RapidTheme.surfaceOverlay)
    }

    /// Commit the typed alias. Shared by Return-in-field and the Use
    /// button so both paths behave identically.
    private func commitCustomAlias() {
        let trimmed = customDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            onUserSelection(trimmed)
            alias = trimmed
        }
        showCustom = false
    }

    private var isStartable: Bool {
        switch server.state {
        case .starting, .ready: return false
        default: return true
        }
    }

    private var canStart: Bool {
        if server.isOperating { return false }
        if case .missing = server.state { return false }
        // F-LWT-1: while Quickstart is mid-flow (low-disk warning,
        // download, serve handoff), the Start CTA disables so a stray
        // click on the still-visible picker can't race a second
        // concurrent download / serve of any alias. Released the
        // instant the coordinator dismisses (.idle phase + not
        // visible per the parent's predicate) OR the flow reaches
        // ``.ready`` / ``.failed``. See ``isQuickstartInFlight`` for
        // the exact phase whitelist.
        if ModelPickerBar.isQuickstartInFlight(phase: quickstart?.phase) {
            return false
        }
        let trimmed = alias.trimmingCharacters(in: .whitespacesAndNewlines)
        return !trimmed.isEmpty
    }

    /// Pure helper: ``true`` iff a Quickstart flow currently owns
    /// the Start CTA. Disabled phases are ``.lowDiskWarning``,
    /// ``.downloading``, ``.starting`` (everything between
    /// "Get started" and "model ready to chat"). ``.idle`` /
    /// ``.ready`` / ``.failed`` release the gate — ``.idle`` means
    /// the user hasn't clicked yet (or dismissed the card),
    /// ``.ready`` means Quickstart finished and ChatView owns the
    /// frame, ``.failed`` means the user can pick a different model
    /// from the picker to recover.
    static func isQuickstartInFlight(phase: QuickstartCoordinator.Phase?) -> Bool {
        guard let phase else { return false }
        switch phase {
        case .lowDiskWarning, .downloading, .starting:
            return true
        case .idle, .ready, .failed:
            return false
        }
    }

    /// Stable hashable key for the ``.task(id:)`` observer that
    /// mirrors the picker selection onto the Quickstart alias.
    /// Folds the in-flight bool into a string AND appends the
    /// catalog-availability bit for the Quickstart alias so the
    /// observer re-fires when the catalog lands the Quickstart row
    /// AFTER the user has already clicked Get started. Without the
    /// second axis, a user who clicks Get started before the catalog
    /// finishes its first refresh would stay pinned on the prior
    /// (RAMBucketed Default) alias even after the catalog gains
    /// ``qwen3-0.6b-4bit`` — the mirror task wouldn't re-fire
    /// because the in-flight bool hadn't changed. Codex r1 MAJOR.
    private var quickstartPhaseGateKey: String {
        return ModelPickerBar.quickstartPhaseGateKey(
            phase: quickstart?.phase,
            catalog: catalog,
            targetAlias: quickstartTargetAlias
        )
    }

    /// Pure helper for the ``.task(id:)`` gate key. Lifted to a
    /// static so the test suite can pin the key transitions
    /// directly:
    ///   * ``off|qs-absent`` → ``off|qs-present`` (catalog landed
    ///     while user hadn't clicked yet — no mirror needed)
    ///   * ``off|qs-absent`` → ``in-flight|qs-absent`` (user clicked
    ///     Get started before catalog loaded — mirror task will
    ///     bail because the alias isn't in the catalog yet, but
    ///     re-fires on the next transition)
    ///   * ``in-flight|qs-absent`` → ``in-flight|qs-present`` (the
    ///     codex r1 race: catalog finally loaded mid-Quickstart-
    ///     download; observer re-fires and mirrors the picker
    ///     selection onto the Quickstart alias)
    ///   * ``in-flight|qs-present`` → ``off|qs-present`` (Quickstart
    ///     completed or user dismissed; mirror gate releases)
    ///
    /// Manual-picker intent contract: while the gate key is
    /// constant, a manual alias swap by the user does NOT re-fire
    /// the mirror (the task id is unchanged). The user's choice
    /// sticks until the Quickstart coordinator transitions out of
    /// the in-flight phase (e.g. ``QuickstartView.handleServerStateChange``
    /// calling ``releaseInFlight`` on a foreign-alias .ready
    /// transition), at which point the gate flips to ``off`` and
    /// the mirror is permanently disengaged for this flow. This
    /// matches QuickstartView's documented "revised intent wins"
    /// behaviour.
    static func quickstartPhaseGateKey(
        phase: QuickstartCoordinator.Phase?,
        catalog: [ModelEntry],
        targetAlias: String = QuickstartCoordinator.defaultChoice.alias
    ) -> String {
        let inFlight = isQuickstartInFlight(phase: phase)
        let catalogHasQuickstart = catalog.contains { $0.alias == targetAlias }
        return "\(inFlight ? "in-flight" : "off")|\(catalogHasQuickstart ? "qs-present" : "qs-absent")"
    }

    /// Mirror the picker's selected alias onto
    /// ``QuickstartCoordinator.alias`` when the coordinator flips
    /// into an in-flight phase. Bails when:
    ///   * the coordinator is nil (legacy / preview surfaces),
    ///   * the gate is off (Quickstart not in flight),
    ///   * the alias is already aimed at the Quickstart target, OR
    ///   * the Quickstart alias isn't in the catalog yet (the next
    ///     catalog refresh will flip ``quickstartPhaseGateKey``'s
    ///     ``qs-absent`` → ``qs-present`` axis and re-fire this
    ///     task — codex r1 MAJOR race fix).
    ///
    /// Manual-picker intent contract: the mirror fires only on the
    /// ``.task(id:)`` re-fire shape (gate-key change), NOT on every
    /// alias mutation. A user who manually picks a different alias
    /// AFTER the mirror has fired keeps their choice — the task id
    /// stays constant and the body doesn't re-run. The downstream
    /// observer in ``QuickstartView.handleServerStateChange`` then
    /// detects the foreign alias and calls ``releaseInFlight``,
    /// which moves the coordinator to ``.ready`` and permanently
    /// disengages the mirror (gate key flips to ``off|``). This
    /// mirrors QuickstartView's documented "revised intent wins"
    /// behaviour (codex r2 MINOR — contract surfaced explicitly).
    private func applyQuickstartSelectionMirror() {
        guard ModelPickerBar.isQuickstartInFlight(phase: quickstart?.phase) else { return }
        let target = quickstartTargetAlias
        guard alias != target else { return }
        guard catalog.contains(where: { $0.alias == target }) else { return }
        alias = target
    }

    /// v1.0: delegates to ``ServerStatusPill``, which owns the single
    /// lifecycle → colour mapping for the whole app.
    ///
    /// The semantics it inherits are the ones this property already
    /// implemented — #129's collapse of ``.idle``/``.stopped`` onto one
    /// off-state colour, amber for starting, green for ready — with two
    /// corrections: the off-state is now the explicit ``statusIdle``
    /// token rather than `.secondary` (which shifts with the
    /// surrounding foreground style), and ``.missing`` reads as an
    /// error rather than a neutral grey, since a missing sidecar is a
    /// fault the user has to act on.
    private var stateColor: Color {
        ServerStatusPill(state: server.state).tint
    }

    private var stateLabel: String {
        Self.stateLabel(
            state: server.state,
            activity: server.downloadProgress.startupActivity
        )
    }

    private var displayedStateLabel: String {
        Self.displayedStateLabel(
            state: server.state,
            activity: server.downloadProgress.startupActivity,
            titlebarStyle: titlebarStyle
        )
    }

    /// The native titlebar deliberately collapses every in-flight phase to
    /// one stable word. The central starting overlay owns file, byte, and
    /// warm-up details without forcing the window toolbar to resize.
    ///
    /// Keyed off ``DownloadProgress.StartupActivity`` (not the tqdm
    /// phase) since the activity-relabel dogfood fix: outside the
    /// titlebar the word must stay a NETWORK truth ("Downloading"
    /// iff bytes provably move), and inside it the collapse makes
    /// the distinction moot anyway.
    static func displayedStateLabel(
        state: ServerState,
        activity: DownloadProgress.StartupActivity,
        titlebarStyle: Bool
    ) -> String {
        if titlebarStyle, case .starting = state {
            return "Starting"
        }
        return stateLabel(state: state, activity: activity)
    }

    /// Pure derivation of the status pill copy from server state +
    /// observed startup activity.
    ///
    /// History, because this word has flip-flopped twice: #130 split
    /// ``.fetching`` off as "Resolving" (cache-hit relaunches flash
    /// that phase with zero bytes moving); #150 collapsed it back to
    /// "Downloading" (a tqdm parse bug parked real downloads on
    /// .fetching, and "Resolving" read as a network stall). Both
    /// failed because they keyed a NETWORK claim off a tqdm PHASE.
    /// The word now keys off ``DownloadProgress.startupActivity`` —
    /// measured byte growth over a pre-spawn baseline — so
    /// "Downloading" appears iff bytes provably move, and a cached
    /// start reads "Loading" for the mmap/Metal window (2026-07
    /// dogfood: the pill claimed "Downloading 5.6 GB / 5.6 GB · 100%"
    /// while switching to an already-downloaded model). A future tqdm
    /// parser regression now degrades to "Loading" (safe), never to a
    /// false "Downloading".
    static func stateLabel(
        state: ServerState,
        activity: DownloadProgress.StartupActivity
    ) -> String {
        switch state {
        case .missing: return "Not installed"
        case .idle, .stopped:
            // #129: visually identical pills meaning two different
            // server-lifecycle states confused users. After a Stop
            // transition we previously rendered "Stopped" with an
            // amber-tinted dot — but the user has no actionable
            // distinction from a cold-launch "Idle" (no warm process,
            // no warm cache, Start re-spawns from scratch in both).
            // Collapse the user-facing copy to a single off-state,
            // matching Ollama / LM Studio. The internal ``ServerState``
            // enum keeps both cases so other surfaces can still react
            // (e.g. session-restore behavior on .stopped).
            return "Idle"
        case .starting:
            // The alias is intentionally omitted — the picker to the
            // left already names the model, so repeating it here was
            // duplicate clutter. ``progressSubtitle`` carries the
            // percent + ETA summary below the label.
            switch activity {
            case .downloading: return "Downloading"
            case .warmingUp: return "Warming up"
            case .loading: return "Loading"
            case .starting: return "Starting"
            }
        case .ready: return "Ready"
        case .crashed: return "Crashed"
        }
    }

    /// v0.5.2: drive ``rapid-mlx rm`` for one alias and refresh the
    /// catalog so the row's green-dot + size column flip to "uncached"
    /// the instant the CLI returns. ``deleting`` gates against a
    /// concurrent second invocation on the same alias; the toast
    /// surfaces success-or-failure for ``deletionToastDuration`` and
    /// then clears.
    ///
    /// Issue #210: the inline toast formatting was lifted into
    /// ``ModelCacheActions`` so the picker and the new
    /// ``SettingsModelManagementPanel`` produce the same
    /// "Deleted X — freed Y" / "Couldn't delete X: …" copy. The
    /// dispatcher there also wraps ``ModelDeletion`` so a future
    /// change to the deletion contract has one site, not two.
    private func runDeletion(of entry: ModelEntry) async {
        guard deleting != entry.alias else { return }
        deleting = entry.alias
        let outcome = await ModelCacheActions.runDeletion(
            for: entry,
            binaryPath: server.binaryPath
        )
        deletionToastGeneration &+= 1
        let toastGeneration = deletionToastGeneration
        switch outcome {
        case .success(let message, _):
            deletionToast = message
        case .failure(let message):
            deletionToast = message
        }
        // Hard refresh — the CLI just mutated the cache, force a
        // re-walk so the row's badges flip to uncached without
        // waiting for the picker's next ``task(id:)`` invalidation.
        downloads.markCacheChanged()
        await refreshCatalog(force: true)
        deleting = nil
        // Auto-clear the toast. Using ``Task.sleep`` keeps the
        // dismissal scoped to the calling actor; the generation check
        // prevents an older timeout from clearing a newer deletion
        // result. UI timing is lifecycle-bound and needs a view host
        // to test, so the race is fixed inline here.
        try? await Task.sleep(nanoseconds: UInt64(deletionToastDuration * 1_000_000_000))
        guard !Task.isCancelled, deletionToastGeneration == toastGeneration else { return }
        deletionToast = nil
    }

    private func refreshCatalog(force: Bool = false) async {
        guard let binary = server.binaryPath else { return }
        if loadingCatalog && !force { return }
        catalogRefreshGeneration &+= 1
        let refreshGeneration = catalogRefreshGeneration
        loadingCatalog = true
        defer {
            if catalogRefreshGeneration == refreshGeneration {
                loadingCatalog = false
            }
        }
        // Route the catalog read through the shared cache (#1470) so re-entering
        // settings does not re-spawn the lister subprocess. Keep the ``loaded``
        // name so the phantom-alias filter below (``entries``) is unchanged.
        let loaded = await ModelCatalogCache.shared.entries(
            binary: binary,
            generation: downloads.cacheGeneration
        )
        guard !Task.isCancelled, catalogRefreshGeneration == refreshGeneration else { return }
        // Belt-and-braces against a phantom alias reaching the UI.
        // ``ModelCatalog.parseAvailable`` already drops engine banner
        // lines, but this guard means a NEW banner shape can at worst
        // produce a missing row — never a selectable fake model, and
        // never a placeholder word promoted into ``alias`` by
        // ``recommendedDefault`` below.
        let entries = loaded.filter { !ModelDisplayName.isUnresolved($0.alias) }
        self.catalog = entries
        // Default selection: only apply if the current alias is blank or
        // not in the catalog (catalog absence still means manual pick).
        if alias.isEmpty || !entries.contains(where: { $0.alias == alias }) {
            if let recommended = recommendedDefault() {
                alias = recommended
            }
        }
    }
}

/// v0.4.18: details popover anchored on the (i) button. Pure layout —
/// receives a fully-resolved `ModelInfo` from `ModelInfoCatalog` and
/// renders header + rows. Kept in this file (alongside the button)
/// because no other surface uses it; if a second caller appears (the
/// Settings model panel, say) extract to `UI/ModelInfoPopover.swift`.
struct ModelInfoPopover: View {
    let info: ModelInfo

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            Divider()
            VStack(alignment: .leading, spacing: 6) {
                infoRow("Family", info.family)
                infoRow("Parameters", info.paramsLabel)
                infoRow("Quantization", info.quantLabel)
                infoRow("Context window", info.contextLabel)
                infoRow("Approx RAM", info.ramLabel)
            }
            if let repo = info.hfRepo, !repo.isEmpty {
                Divider()
                hfRepoRow(repo)
            }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(info.alias)
                .font(.headline)
                .lineLimit(1)
                .textSelection(.enabled)
            Text("\(info.paramsLabel) · \(info.quantLabel) · \(info.contextLabel) context")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private func infoRow(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .monospacedDigit()
                .textSelection(.enabled)
        }
        .font(.callout)
    }

    private func hfRepoRow(_ repo: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Hugging Face repo")
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack(spacing: 6) {
                Text(repo)
                    .font(.callout.monospaced())
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                Spacer()
                Link(destination: URL(string: "https://huggingface.co/\(repo)")!) {
                    Image(systemName: "arrow.up.right.square")
                }
                .help("Open on Hugging Face")
                .accessibilityLabel("Open Hugging Face page")
            }
        }
    }
}

/// Animated state dot. Steady-state for ready / idle / crashed; gentle
/// breathing animation during ``starting`` so the user knows the
/// model is still loading. Pure SwiftUI animation — no timer needed.
///
/// Internal (not `private`) since v1.0: ``ServerStatusPill`` is the
/// shared rendering of ``ServerState`` and needs the same dot, so the
/// dot can no longer be file-scoped to the picker.
struct PulsingStateDot: View {
    let color: Color
    let isAnimating: Bool

    @State private var pulse: Bool = false
    // #547 §14: suppress the breathing loop under Reduce Motion — the dot
    // holds its steady colour and size, so "starting" stays legible (via
    // the picker copy) without the perpetual scale/opacity motion.
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var animating: Bool { isAnimating && !reduceMotion }

    var body: some View {
        Circle()
            .fill(color)
            .frame(width: 8, height: 8)
            // Scale/opacity are gated on `animating` (which folds in
            // !reduceMotion), so a stale `pulse` can never render — the dot
            // rests steady under Reduce Motion regardless of the flag.
            .scaleEffect(animating && pulse ? 1.35 : 1.0)
            .opacity(animating && pulse ? 0.55 : 1.0)
            .animation(
                animating
                    ? RapidMotion.breathe.repeatForever(autoreverses: true)
                    : .default,
                value: pulse
            )
            .onAppear {
                pulse = RapidMotion.shouldPulse(isAnimating: isAnimating, reduceMotion: reduceMotion)
            }
            .onChange(of: isAnimating) { _, new in
                pulse = RapidMotion.shouldPulse(isAnimating: new, reduceMotion: reduceMotion)
            }
            .onChange(of: reduceMotion) { _, reduced in
                // Toggling Reduce Motion at runtime must start / stop the loop.
                pulse = RapidMotion.shouldPulse(isAnimating: isAnimating, reduceMotion: reduced)
            }
    }
}
