import AppKit
import ImageIO
import SwiftUI
import UniformTypeIdentifiers

/// The Images tab. Deliberately mirrors ``ChatView``: a scrollable results
/// area on top and, at the bottom, the *same* compose box — a `surfaceRaised`
/// rounded field with the model picker + submit button clustered at its
/// bottom-right — so model selection and input feel identical across tabs.
struct ImagesView: View {
    @Bindable var viewModel: ImageGenViewModel
    @Bindable var server: ServerManager
    @Environment(\.openWindow) private var openWindow
    @Environment(SettingsRouter.self) private var settingsRouter

    private let contentMaxWidth: CGFloat = RapidTheme.Layout.contentMaxWidth

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @State private var composeFocusToken = 0
    @State private var pickerHovering = false
    /// Bumped when the user tries to submit while gated, so the readiness
    /// banner flashes for attention (same signal ChatView uses).
    @State private var blockedSendAttempts = 0

    var body: some View {
        VStack(spacing: 0) {
            stageAndHistory
            composer
        }
        .background(RapidTheme.surfaceCanvas)
        .task { await viewModel.refreshCatalog() }
    }

    // MARK: - Stage + history

    private var stageAndHistory: some View {
        VStack(spacing: 12) {
            stage
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            if !viewModel.results.isEmpty {
                // Centered on the same column as the composer so the strip
                // reads as part of the layout rather than floating far-left.
                filmstrip
                    .frame(maxWidth: contentMaxWidth)
                    .frame(maxWidth: .infinity)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder
    private var stage: some View {
        ZStack {
            if let active = viewModel.activeImage, let nsImage = NSImage(data: active.pngData) {
                ZStack(alignment: .topTrailing) {
                    Image(nsImage: nsImage)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .clipShape(RoundedRectangle(cornerRadius: 14))
                        .overlay(
                            RoundedRectangle(cornerRadius: 14)
                                .stroke(RapidTheme.hairline, lineWidth: 1)
                        )
                        .accessibilityIdentifier("Images.Stage")
                    // Keep actions as siblings of the named image. Applying
                    // the identifier after `.overlay` makes AppKit inherit it
                    // onto both buttons, erasing their semantic identifiers.
                    resultActionsOverlay(active)
                }
            } else if !viewModel.isGenerating {
                emptyStage
            }

            if viewModel.isGenerating {
                progressHUD.transition(.opacity)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder
    private func resultActionsOverlay(_ image: GeneratedImage) -> some View {
        if !viewModel.isGenerating {
            HStack(spacing: 7) {
                if !viewModel.isEditing {
                    Button {
                        viewModel.beginEdit(image)
                    } label: {
                        Image(systemName: "pencil")
                            .font(.system(size: 12, weight: .semibold))
                            .frame(width: 28, height: 28)
                            .background(.ultraThinMaterial, in: Circle())
                    }
                    .buttonStyle(.plain)
                    .help("Edit image")
                    .accessibilityHint("Edit image")
                    .accessibilityIdentifier("Images.Result.Edit")
                }

                Button {
                    save(image)
                } label: {
                    Image(systemName: "square.and.arrow.down")
                        .font(.system(size: 12, weight: .semibold))
                        .frame(width: 28, height: 28)
                        .background(.ultraThinMaterial, in: Circle())
                }
                .buttonStyle(.plain)
                .help("Save image")
                .accessibilityHint("Save image")
                .accessibilityIdentifier("Images.Result.Save")
            }
            .padding(10)
        }
    }

    /// The empty stage: a scale drawing of what pressing Generate will
    /// produce, then the same readiness-driven copy Chat uses.
    ///
    /// The mascot is gone from this surface on purpose. It is the brand
    /// moment for Chat's empty state and for first run, and repeating it
    /// on every empty surface turned a greeting into wallpaper. What
    /// belongs here instead is the one thing a user opening Images
    /// actually needs to know before typing: the shape and the size of
    /// the thing about to be made. The frame tracks the live aspect and
    /// resolution selections, so changing either previews itself.
    ///
    /// It is a drawing, not a control — every hit target stays in the
    /// composer's aspect and resolution pickers, which own these values.
    private var emptyStage: some View {
        EmptyState(
            title: "Draw anything",
            message: readiness.isReady
                ? "Describe what you want to see, then press Generate."
                : readiness.emptyStateSubtitle,
            hint: readiness.isReady ? nil : readiness.emptyStateHint,
            markDiameter: aspectPreviewSize.height,
            marksOnBackplate: false,
            mark: { aspectPreview },
            actions: { EmptyView() }
        )
        .accessibilityIdentifier("Images.EmptyState")
    }

    /// Longest edge of the preview frame. Sized so the tallest aspect
    /// (3:4 portrait) still leaves the title and subtitle comfortably
    /// above the composer at the 560pt window floor.
    private static let aspectPreviewLongEdge: CGFloat = 150

    private var aspectPreviewSize: CGSize {
        let dimensions = viewModel.aspect.dimensions(for: viewModel.resolution)
        let long = Self.aspectPreviewLongEdge
        guard dimensions.width > 0, dimensions.height > 0 else {
            return CGSize(width: long, height: long)
        }
        let ratio = CGFloat(dimensions.width) / CGFloat(dimensions.height)
        return ratio >= 1
            ? CGSize(width: long, height: long / ratio)
            : CGSize(width: long * ratio, height: long)
    }

    private var aspectPreview: some View {
        VStack(spacing: RapidTheme.Space.sm - 1) {
            Image(systemName: "photo")
                .font(.system(size: 26, weight: .light))
                .foregroundStyle(RapidTheme.textTertiary)
            Text(viewModel.aspect.size(for: viewModel.resolution))
                .font(RapidFont.code)
                .foregroundStyle(RapidTheme.textTertiary)
        }
        .frame(width: aspectPreviewSize.width, height: aspectPreviewSize.height)
        // Dashed, because the frame describes something that does not
        // exist yet. A solid border would read as an image that failed
        // to load.
        .overlay(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.panel, style: .continuous)
                .strokeBorder(
                    RapidTheme.hairlineStrong,
                    style: StrokeStyle(lineWidth: 1.5, dash: [5, 4])
                )
        )
        // Redraws should feel like the picker moving, not like the stage
        // reloading.
        .rapidAnimation(RapidMotion.quick, value: aspectPreviewSize.width)
        .rapidAnimation(RapidMotion.quick, value: aspectPreviewSize.height)
    }

    // MARK: - Readiness (mirrors ChatView: same "load the model first" flow)

    /// Readiness for the selected image model. A healthy sidecar may keep this
    /// engine resident beside the chat engine; otherwise the shared resolver
    /// presents the same on-demand load guidance used by Chat.
    private var readiness: ModelReadiness {
        // In-process image admission leaves the chat sidecar globally
        // `.ready`, so the generic resolver cannot infer that this alias is
        // downloading/loading. Use ServerManager's alias-scoped SSOT; without
        // it the CTA appears dead and remains repeatedly pressable while HF is
        // already writing gigabytes to disk.
        if server.isResidentLoadInFlight(viewModel.selectedAlias) {
            return .starting(
                alias: viewModel.selectedAlias,
                detail: "Downloading or loading the image model…"
            )
        }
        return ModelReadiness.resolve(
            serverState: server.readinessState(for: viewModel.selectedAlias),
            alias: viewModel.selectedAlias,
            cacheState: imageCacheState,
            sizeText: viewModel.imageModels
                .first { $0.alias == viewModel.selectedAlias }?.sizeOnDisk,
            progress: startupProgress,
            // A rejected resident load (e.g. the sidecar cannot serve this
            // model) must surface the engine's reason HERE, on the surface
            // that initiated the load, not only in the log pane (#1838). The
            // resolve precedence rules keep `.ready`/`.starting`/`.downloading`
            // winning over a stale rejection, and `failureApplies` scopes it to
            // the model that actually failed. Failures are keyed per alias in
            // ``ServerManager``, so we read only the outcome for the model
            // this surface asked to load (#1838 follow-up: concurrent loads of
            // different models cannot clobber one another).
            failure: server.residentLoadFailure(for: viewModel.selectedAlias).map {
                ModelReadiness.Failure(message: $0.message, alias: $0.alias)
            }
        )
    }

    private var imageCacheState: ModelReadiness.CacheState {
        guard !viewModel.selectedAlias.isEmpty, viewModel.catalogLoaded else {
            return .catalogPending
        }
        guard let entry = viewModel.imageModels
            .first(where: { $0.alias == viewModel.selectedAlias }) else {
            return .notInCatalog
        }
        return entry.cached ? .onDisk : .notOnDisk
    }

    private var startupProgress: ModelReadiness.ProgressSnapshot? {
        guard case .starting = server.state else { return nil }
        return ModelReadiness.ProgressSnapshot(
            activity: server.downloadProgress.startupActivity,
            subtitle: server.downloadProgress.progressSubtitle,
            fraction: server.downloadProgress.progressFraction
        )
    }

    /// The banner's next-step action: start the sidecar or load the selected
    /// image engine into the already-running process.
    private func handleReadinessAction(_ action: ModelReadiness.Action) {
        switch action {
        case .chooseModel:
            break  // the composer's model picker owns this step
        case .downloadAndStart(let target), .start(let target), .retry(let target):
            let hf = viewModel.imageModels.first { $0.alias == target }?.hfRepo
            // Same shared helper as Chat: ``ensureServing`` (not ``start``),
            // because the user is almost always switching FROM a running chat
            // model TO the image model, and cold-start ``start`` would no-op
            // while that model is resident. See ``ReadinessModelStart``.
            Task { await loadImageModel(target, hfPath: hf) }
        case .restart(let target):
            let hf = viewModel.imageModels.first { $0.alias == target }?.hfRepo
            Task {
                await server.stop()
                await loadImageModel(target, hfPath: hf)
            }
        case .openModelManagement:
            settingsRouter.route(.openModelManagement) {
                openWindow(id: "settings")
            }
        }
    }

    private func loadImageModel(_ alias: String, hfPath: String?) async {
        let entry = viewModel.imageModels.first { $0.alias == alias }
        _ = await server.ensureServing(
            alias: alias,
            hfPath: hfPath,
            estimatedMemoryGB: ModelSizing.residentEstimateGB(
                alias: alias,
                sizeText: entry?.sizeOnDisk
            ),
            imageMode: viewModel.isEditing ? .editing : .generation
        )
        // A cold resident load can change the cache while the sidecar stays
        // globally ready. Refresh this surface so its cached/downloaded copy
        // cannot remain stale after the request settles.
        await viewModel.refreshCatalog()
    }

    private var sendEnabled: Bool {
        viewModel.canSubmit && readiness.sendAllowed
    }

    // MARK: - Progress HUD (the wait, designed)

    private var progressHUD: some View {
        TimelineView(.periodic(from: .now, by: 0.08)) { context in
            let elapsed = viewModel.genStartedAt.map { context.date.timeIntervalSince($0) } ?? 0
            // A 0→1 loop driving the shimmer sweep + status-dot pulse, derived
            // from the frame's date so it animates without stored state.
            let phase = (context.date.timeIntervalSinceReferenceDate
                .truncatingRemainder(dividingBy: 1.6)) / 1.6
            let denoising = viewModel.phase == .denoising && viewModel.progress != nil
            let finalizing = viewModel.phase == .finalizing
            let total = max(viewModel.progress?.total ?? 0, viewModel.estimatedSteps)
            let step = max(1, viewModel.progress?.step ?? 0)
            let fraction = (denoising && total > 0) ? min(1, Double(step) / Double(total)) : 0

            ZStack {
                // Soft scrim so the card reads cleanly over any prior image.
                LinearGradient(colors: [.black.opacity(0.06), .black.opacity(0.34)],
                               startPoint: .top, endPoint: .bottom)
                    .allowsHitTesting(false)

                VStack(spacing: 14) {
                    HStack(spacing: 10) {
                        if denoising || finalizing {
                            // Breathing, and only when Reduce Motion is
                            // off: this is a perpetual loop, so it is
                            // fully suppressed rather than merely slowed.
                            // The decorative glow is gone with it — an
                            // amber dot on graphite already separates,
                            // and a halo around a status indicator is
                            // exactly the ornament this pass removes.
                            let breathing = RapidMotion.shouldPulse(
                                isAnimating: true,
                                reduceMotion: reduceMotion
                            )
                            Circle()
                                .fill(RapidTheme.brandPrimary)
                                .frame(width: 9, height: 9)
                                .scaleEffect(
                                    breathing
                                        ? 0.65 + 0.35 * (0.5 + 0.5 * sin(phase * .pi * 2))
                                        : 1
                                )
                            Text(viewModel.cancelling
                                 ? "Stopping…"
                                 : (finalizing ? "Finalizing image…" : "Generating"))
                                .font(.system(size: 14, weight: .semibold))
                                .foregroundStyle(RapidTheme.bandInk)
                        } else {
                            ProgressView().controlSize(.small)
                            Text(viewModel.cancelling
                                 ? "Stopping…"
                                 : "Warming up \(viewModel.selectedDisplayName)")
                                .font(.system(size: 14, weight: .semibold))
                                .foregroundStyle(RapidTheme.bandInk)
                                .lineLimit(1).truncationMode(.middle)
                        }
                        Spacer(minLength: 8)
                        if denoising {
                            Text("\(step) / \(total)")
                                .font(.system(size: 12, weight: .semibold, design: .monospaced))
                                .foregroundStyle(RapidTheme.bandInkSecondary)
                                .monospacedDigit()
                        }
                        Button { viewModel.cancel() } label: {
                            Image(systemName: "xmark")
                                .font(.system(size: 10, weight: .bold))
                                .foregroundStyle(RapidTheme.bandInk)
                                .frame(width: 22, height: 22)
                                .background(RapidTheme.bandTrack, in: Circle())
                        }
                        .buttonStyle(.plain)
                        .disabled(viewModel.cancelling)
                        .help("Cancel")
                        .accessibilityHint("Cancel")
                        .accessibilityIdentifier("Images.Cancel")
                    }

                    ShimmerProgressBar(
                        fraction: fraction,
                        indeterminate: !denoising || finalizing,
                        phase: phase
                    )
                        .frame(height: 10)

                    HStack {
                        Text(String(format: "%.1fs", max(0, elapsed)))
                            .font(.system(size: 12, weight: .medium, design: .monospaced))
                            .foregroundStyle(RapidTheme.bandInkSecondary)
                            .monospacedDigit()
                        Spacer()
                        // ETA from the denoise-phase clock, not total elapsed —
                        // otherwise cold-load time inflates the per-step estimate.
                        let denoiseElapsed = viewModel.denoiseStartedAt
                            .map { context.date.timeIntervalSince($0) } ?? elapsed
                        Text(finalizing
                             ? "Decoding and saving…"
                             : (denoising
                                ? (etaText(step: step, total: total, elapsed: denoiseElapsed)
                                   ?? "Finalizing next…")
                                : "First run — only happens once"))
                            .font(.system(size: 12, weight: .medium))
                            .foregroundStyle(RapidTheme.bandInkSecondary)
                    }
                }
                .padding(18)
                .frame(width: 340)
                // The same graphite ground the Chat band paints, in the
                // shape this surface's geometry calls for. Images cannot
                // use a band: the subject here IS the canvas, and a strip
                // above it would push the stage down and shrink the one
                // thing the user is watching. A HUD over the image keeps
                // progress on top of its own subject.
                //
                // Previously this card took ``surfaceOverlay``, a
                // near-white plane — so over a bright render it was a
                // pale card on a pale image with a shadow doing all the
                // separation. Graphite separates on its own, in both
                // appearances and over any image.
                .background(RapidTheme.surfaceBand,
                            in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .strokeBorder(RapidTheme.bandTrack, lineWidth: 1)
                )
                .shadow(color: .black.opacity(0.28), radius: 22, y: 10)
            }
        }
    }

    private func etaText(step: Int, total: Int, elapsed: TimeInterval) -> String? {
        guard step > 0, total > step, elapsed > 0 else { return nil }
        let perStep = elapsed / Double(step)
        return "~\(Int((perStep * Double(total - step)).rounded()))s left"
    }

    // MARK: - Filmstrip

    private var filmstrip: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                // Enumerated so each thumb can carry a stable, addressable
                // identifier. Position, not the image's UUID: a UUID differs
                // on every run, which would make the AX structural baseline
                // unrepeatable and turn `image-generation` into a flow that
                // can only ever pass on the run that wrote its baseline.
                // ``results`` is newest-first (``insert(at: 0)``), so thumb 1
                // is always the most recent render.
                ForEach(Array(viewModel.results.enumerated()), id: \.element.id) { index, image in
                    filmstripThumb(image, ordinal: index + 1)
                }
            }
            .padding(.vertical, 2)
        }
        .frame(height: 64)
        .accessibilityIdentifier("Images.Gallery")
    }

    private func filmstripThumb(_ image: GeneratedImage, ordinal: Int) -> some View {
        let selected = viewModel.activeImage?.id == image.id
        return Button {
            viewModel.select(image)
        } label: {
            Group {
                if let nsImage = NSImage(data: image.pngData) {
                    Image(nsImage: nsImage).resizable().aspectRatio(contentMode: .fill)
                } else {
                    Rectangle().fill(RapidTheme.card)
                }
            }
            .frame(width: 56, height: 56)
            .clipShape(RoundedRectangle(cornerRadius: 9))
            .overlay(
                RoundedRectangle(cornerRadius: 9)
                    .stroke(selected ? RapidTheme.brandAmber : RapidTheme.hairline,
                            lineWidth: selected ? 2 : 1)
            )
        }
        .buttonStyle(.plain)
        // The thumb's whole label is an image, so without an identifier and a
        // label it reaches VoiceOver — and the golden flow — as an unnamed
        // button. "A second render produced a second thumbnail" is then
        // unassertable except by counting anonymous buttons, which any
        // unrelated control added to the strip would break.
        //
        // The identifier is the position, not the image's UUID (#1725's first
        // pass): a UUID differs on every run, which would make the AX
        // structural baseline unrepeatable and turn `image-generation` into a
        // flow that can only pass on the run that wrote its baseline. The
        // label still announces selection for VoiceOver, as #1725 intended —
        // the enclosing `Images.Gallery` identifies the strip, not the thumbs
        // inside it, so this is the only place a screen-reader user hears
        // which render is active.
        .accessibilityIdentifier("Images.Gallery.Thumb.\(ordinal)")
        // The label names the render, not just its slot. "Image 2" tells a
        // VoiceOver user only where in the strip they are — every thumb in a
        // gallery of near-identical variations then sounds the same, and the
        // one thing that distinguishes them, the prompt that produced each
        // one, is the caption sighted users can already read.
        //
        // It is also the only thing in the accessibility tree that is derived
        // from the RESULT rather than from its position. Positional labels
        // are satisfied by a gallery that lists two entries and shows the
        // same render for both, which is a real failure mode and one the
        // golden flow could not otherwise see: AX carries no pixel data, so a
        // dump of a duplicated image is byte-identical to a dump of two
        // distinct ones. Binding the label to each entry's own prompt makes
        // the flow's "a second render, not a redraw of the first" assertion
        // actually testable. (It pins the RECORD, not the pixels: two
        // separate entries that somehow carried identical image data would
        // still read as distinct. Proving that would mean publishing a
        // content digest through the UI, which is scaffolding a shipping
        // surface should not carry.)
        .accessibilityLabel(
            selected
                ? "Image \(ordinal), \(image.prompt), selected"
                : "Image \(ordinal), \(image.prompt)"
        )
    }

    // MARK: - Composer (mirrors ChatView's compose box)

    private var composer: some View {
        VStack(spacing: RapidTheme.Space.sm) {
            if let error = viewModel.errorMessage {
                InlineNotice(message: error, tone: .error)
                    .frame(maxWidth: contentMaxWidth)
                    .frame(maxWidth: .infinity)
            }
            if !readiness.isReady {
                ReadinessBanner(
                    readiness: readiness,
                    attentionToken: blockedSendAttempts,
                    onAction: handleReadinessAction
                )
                .frame(maxWidth: contentMaxWidth)
                .frame(maxWidth: .infinity)
            }
            if viewModel.prompt.isEmpty && !viewModel.isEditing {
                starters
                    .frame(maxWidth: contentMaxWidth)
                    .frame(maxWidth: .infinity)
            }
            VStack(spacing: RapidTheme.Space.sm - 2) {
                if let source = viewModel.editSource {
                    editSourceBar(source)
                }
                ComposeField(
                    text: $viewModel.prompt,
                    focusToken: composeFocusToken,
                    isStreaming: viewModel.isGenerating,
                    placeholder: composerPlaceholder,
                    onSubmit: runSubmit,
                    onCancel: { viewModel.cancel() },
                    // Without these the editor inside this tab announces
                    // itself as the CHAT compose field, because that is
                    // ``ComposeField``'s default. ``Images.Prompt`` below sits
                    // on the SwiftUI wrapper and resolves to the placeholder
                    // text, not to the NSTextView, so it cannot stand in.
                    axIdentifier: AutosizingTextView.imagePromptAccessibilityIdentifier,
                    axLabel: AutosizingTextView.imagePromptAccessibilityLabel,
                    axRoleDescription: AutosizingTextView.imagePromptAccessibilityRoleDescription
                )
                .accessibilityIdentifier("Images.Prompt")
                composerControls
            }
            .padding(.horizontal, RapidTheme.Space.md - 2)
            .padding(.vertical, RapidTheme.Space.sm)
            .background(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.input, style: .continuous)
                    .fill(RapidTheme.surfaceRaised)
            )
            .overlay(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.input, style: .continuous)
                    .strokeBorder(RapidTheme.hairlineStrong, lineWidth: 1)
            )
            .frame(maxWidth: contentMaxWidth)
            .frame(maxWidth: .infinity)
        }
        .padding(.horizontal, RapidTheme.Space.xl)
        .padding(.top, RapidTheme.Space.md)
        .padding(.bottom, RapidTheme.Space.lg)
    }

    /// Bottom row of the compose box: canvas controls on the left, then the
    /// inline model picker + submit clustered on the right — the same
    /// `model ▾  ⬆` grouping ChatView uses.
    private var composerControls: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: RapidTheme.Space.sm) {
                if viewModel.isEditing {
                    importButton
                } else {
                    aspectPicker
                    resolutionPicker
                    importButton
                }
                Spacer(minLength: 0)
                modelPicker
                sendOrStopButton
            }

            VStack(spacing: RapidTheme.Space.xs) {
                HStack(spacing: RapidTheme.Space.sm) {
                    if viewModel.isEditing {
                        importButton
                    } else {
                        aspectPicker
                        resolutionPicker
                        importButton
                    }
                    Spacer(minLength: 0)
                }
                HStack(spacing: RapidTheme.Space.sm) {
                    Spacer(minLength: 0)
                    modelPicker
                    sendOrStopButton
                }
            }
        }
    }

    private var composerPlaceholder: String {
        guard readiness.isReady else { return readiness.composerPlaceholder }
        return viewModel.isEditing
            ? "Describe what you want to change…"
            : "Describe the image you want…"
    }

    private func editSourceBar(_ source: GeneratedImage) -> some View {
        HStack(spacing: 9) {
            if let image = NSImage(data: source.pngData) {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: 38, height: 38)
                    .clipShape(RoundedRectangle(cornerRadius: 5))
            }
            VStack(alignment: .leading, spacing: 2) {
                Text("Editing image")
                    .font(RapidFont.secondary)
                    .foregroundStyle(RapidTheme.textPrimary)
                Text(source.prompt)
                    .font(RapidFont.caption)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            Spacer(minLength: 8)
            Button { viewModel.cancelEdit() } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 10, weight: .semibold))
                    .frame(width: 24, height: 24)
            }
            .buttonStyle(.plain)
            .disabled(viewModel.isGenerating)
            .help("Exit image editing")
            .accessibilityLabel("Exit image editing")
            .accessibilityIdentifier("Images.Edit.Exit")
        }
        .padding(.bottom, 2)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("Images.Edit.Source")
    }

    private var importButton: some View {
        Button(action: chooseEditImage) {
            Image(systemName: "photo.badge.plus")
                .font(.system(size: 12, weight: .medium))
                .frame(width: 28, height: RapidTheme.ControlHeight.small)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(viewModel.isGenerating)
        .help(viewModel.isEditing ? "Replace source image" : "Import image to edit")
        .accessibilityLabel(viewModel.isEditing ? "Replace source image" : "Import image to edit")
        .accessibilityIdentifier("Images.Edit.Import")
    }

    private var starters: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 7) {
                ForEach(Array(ImageGenViewModel.starters.enumerated()), id: \.offset) { index, starter in
                    Button {
                        viewModel.use(starter: starter)
                    } label: {
                        Text(starter)
                            .font(.caption)
                            .lineLimit(1)
                            .padding(.horizontal, 11)
                            .padding(.vertical, 6)
                            .background(RapidTheme.card)
                            .clipShape(Capsule())
                            .overlay(Capsule().stroke(RapidTheme.hairline, lineWidth: 1))
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("Images.Starter.\(index)")
                }
            }
        }
    }

    private var aspectPicker: some View {
        HStack(spacing: 4) {
            ForEach(ImageGenViewModel.Aspect.allCases) { ar in
                let on = viewModel.aspect == ar
                Button {
                    viewModel.aspect = ar
                } label: {
                    Text(ar.label)
                        .font(.system(size: 11, weight: .medium))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 5)
                        .background(on ? RapidTheme.hoverFill : Color.clear)
                        .foregroundStyle(on ? Color.primary : Color.secondary)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                }
                .buttonStyle(.plain)
                // ``Aspect.rawValue`` (square / portrait / landscape), NOT
                // ``label`` — the label is display copy ("1:1", "3:4") and a
                // future copy change would silently rename the hook.
                .accessibilityIdentifier("Images.Aspect.\(ar.rawValue)")
                .accessibilityLabel(ar.label)
                .accessibilityAddTraits(on ? .isSelected : [])
            }
        }
    }

    /// Output dimensions are explicit rather than hidden inside the aspect
    /// buttons. The menu keeps the compact composer row stable while still
    /// showing the exact width and height each preset will send to the server.
    private var resolutionPicker: some View {
        Menu {
            ForEach(ImageGenViewModel.Resolution.allCases) { resolution in
                let size = viewModel.aspect.size(for: resolution)
                    .replacingOccurrences(of: "x", with: " × ")
                Button {
                    viewModel.resolution = resolution
                } label: {
                    if viewModel.resolution == resolution {
                        Label(size, systemImage: "checkmark")
                    } else {
                        Text(size)
                    }
                }
                .accessibilityIdentifier("Images.Resolution.\(resolution.rawValue)")
                .accessibilityAddTraits(viewModel.resolution == resolution ? .isSelected : [])
            }
        } label: {
            HStack(spacing: 5) {
                Image(systemName: "ruler")
                    .font(.system(size: 11, weight: .medium))
                    .accessibilityHidden(true)
                Text(viewModel.outputSizeLabel)
                    .font(.system(size: 11, weight: .medium))
                    .monospacedDigit()
                Image(systemName: "chevron.down")
                    .font(.system(size: 8, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .accessibilityHidden(true)
            }
            .foregroundStyle(Color.secondary)
            .padding(.horizontal, 7)
            .frame(height: RapidTheme.ControlHeight.small)
            .contentShape(Rectangle())
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Output resolution: \(viewModel.outputSizeLabel)")
        .accessibilityLabel("Output resolution")
        .accessibilityValue(viewModel.outputSizeLabel)
        .accessibilityIdentifier("Images.Resolution")
    }

    /// The inline model picker — same composer-embedded chip as chat
    /// (``ModelPickerBar`` in `composerStyle`): borderless, a fill on hover,
    /// a cache glyph per row, scaling to any number of image models.
    private var modelPicker: some View {
        Menu {
            if viewModel.selectableModels.isEmpty {
                Text(viewModel.catalogLoaded
                     ? (viewModel.isEditing ? "No image editing models available" : "No image generation models available")
                     : "Loading…")
            } else {
                ForEach(viewModel.selectableModels) { entry in
                    Button {
                        viewModel.selectedAlias = entry.alias
                    } label: {
                        Label(
                            modelRowTitle(entry),
                            systemImage: ModelPickerBar.cacheGlyph(cached: entry.cached)
                        )
                    }
                    // Keyed on the alias, which is what selecting the row
                    // actually writes — so the hook and the effect cannot
                    // drift apart the way a positional index would.
                    .accessibilityIdentifier("Images.Model.\(entry.alias)")
                }
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: viewModel.isEditing ? "pencil.and.scribble" : "photo")
                    .foregroundStyle(.secondary)
                    .accessibilityHidden(true)
                Text(viewModel.selectedAlias.isEmpty ? "Choose a model" : viewModel.selectedAlias)
                    .font(RapidFont.secondary)
                    .foregroundStyle(viewModel.selectedAlias.isEmpty ? .secondary : .primary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Image(systemName: "chevron.up.chevron.down")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(pickerHovering ? .primary : .secondary)
                    .accessibilityHidden(true)
            }
            .padding(.horizontal, RapidTheme.Space.sm)
            .frame(height: RapidTheme.ControlHeight.small)
            .background(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous)
                    .fill(pickerHovering ? RapidTheme.hoverFill : .clear)
            )
            .overlay(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous)
                    .strokeBorder(pickerHovering ? RapidTheme.hairlineStrong : .clear, lineWidth: 1)
            )
            .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .disabled(viewModel.isGenerating)
        .onHover { pickerHovering = $0 }
        .help(viewModel.selectedAlias.isEmpty ? "Choose a model" : "Model: \(viewModel.selectedAlias)")
        // Mirror the tooltip into an accessibility hint: SwiftUI's `.help(_)`
        // reaches AXHelp on macOS 15 but not on macOS 26 for a `Menu` styled as
        // a button, so without this the model the picker resolved to is
        // invisible to VoiceOver and to the golden-flow harness on 26.
        .accessibilityHint(viewModel.selectedAlias.isEmpty ? "Choose a model" : "Model: \(viewModel.selectedAlias)")
        .accessibilityIdentifier("Images.ModelPicker")
    }

    private func modelRowTitle(_ entry: ModelEntry) -> String {
        if let size = entry.sizeOnDisk, !size.isEmpty {
            return "\(entry.alias) · \(size)"
        }
        return entry.alias
    }

    /// Submit / stop, styled exactly like ChatView's send button: an amber
    /// disc when there's something to run, a stop disc while generating.
    @ViewBuilder
    private var sendOrStopButton: some View {
        if viewModel.isGenerating {
            Button { viewModel.cancel() } label: {
                Image(systemName: "stop.fill")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundStyle(RapidTheme.sendButtonIcon)
                    .frame(width: 28, height: 28)
                    .background(Circle().fill(RapidTheme.sendButton))
            }
            .buttonStyle(.plain)
            .disabled(viewModel.cancelling)
            .help("Cancel")
            .accessibilityHint("Cancel")
            .accessibilityIdentifier("Images.Generate")
        } else {
            Button(action: runSubmit) {
                Image(systemName: "arrow.up")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundStyle(sendEnabled ? RapidTheme.onBrandPrimary : Color.secondary)
                    .frame(width: 28, height: 28)
                    .background(Circle().fill(sendEnabled ? RapidTheme.brandPrimary : Color.clear))
                    .overlay(
                        Circle().strokeBorder(
                            sendEnabled ? .clear : RapidTheme.hairlineStrong, lineWidth: 1)
                    )
            }
            .buttonStyle(.plain)
            .disabled(!sendEnabled)
            .help(readiness.isReady
                  ? (viewModel.isEditing ? "Edit image" : "Generate")
                  : "Load the model first")
            // `.help(_)` does not reach AXHelp on macOS 26 for this button
            // (it publishes an identifier but no accessibilityLabel), so mirror
            // the readiness tooltip into a hint — the only signal that
            // distinguishes "ready" from "load the model first" while the
            // button is disabled for an empty prompt on both.
            .accessibilityHint(readiness.isReady
                               ? (viewModel.isEditing ? "Edit image" : "Generate")
                               : "Load the model first")
            .accessibilityIdentifier("Images.Generate")
        }
    }

    // MARK: - Actions

    private func runSubmit() {
        guard sendEnabled else {
            // Not ready (or empty prompt): flash the readiness banner instead
            // of silently doing nothing, so the blocking step is visible.
            if !readiness.sendAllowed { blockedSendAttempts += 1 }
            return
        }
        Task { await viewModel.submit() }
    }

    private func save(_ image: GeneratedImage) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "rapid-image.png"
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try image.pngData.write(to: url)
        } catch {
            // Don't let a disk-full / permission failure look like a success.
            viewModel.errorMessage = "Couldn't save the image: \(error.localizedDescription)"
        }
    }

    private func chooseEditImage() {
        // The native NSOpenPanel's file browser publishes no accessibility
        // identifiers, so a golden flow cannot reach it: neither a
        // pid-targeted CGEvent nor a HID session tap opens its "Go to Folder"
        // sheet on an unattended build/CI runner. To keep the image-edit
        // import journey deterministic, a golden-harness-only seam names the
        // file to import so the exactly-same post-pick path below still runs
        // for real — edit mode, the "Replace source image" affordance, the
        // file name on the source bar, and the fixture's bytes on the wire are
        // all still asserted. The seam requires BOTH RAPID_GUI_GOLDEN_MODE=1
        // (an explicit harness-launch gate) AND a RAPID_SIMULATED_IMPORT_PATH
        // naming a file that actually exists, so a real user — whose launch
        // never sets the golden-mode switch — always gets NSOpenPanel even if
        // an unrelated process leaked an import path into the environment.
        if ProcessInfo.processInfo.environment["RAPID_GUI_GOLDEN_MODE"] == "1",
           let simulated = ProcessInfo.processInfo.environment["RAPID_SIMULATED_IMPORT_PATH"],
           !simulated.isEmpty,
           FileManager.default.fileExists(atPath: simulated) {
            importEditImage(at: URL(fileURLWithPath: simulated))
            return
        }
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        guard panel.runModal() == .OK, let url = panel.url else { return }
        importEditImage(at: url)
    }

    private func importEditImage(at url: URL) {
        Task {
            do {
                let png = try await EditImageImporter.pngData(from: url)
                viewModel.beginEdit(GeneratedImage(
                    pngData: png,
                    prompt: url.deletingPathExtension().lastPathComponent,
                    isEdit: false
                ))
            } catch {
                viewModel.errorMessage = "Couldn't import the image: \(error.localizedDescription)"
            }
        }
    }
}

enum ImportedEditImageError: LocalizedError {
    case tooLarge
    case tooManyPixels
    case unsupportedType
    case cannotDecode

    var errorDescription: String? {
        switch self {
        case .tooLarge: return "Choose an image smaller than 25 MB."
        case .tooManyPixels: return "Choose an image no larger than 8192 px or 40 megapixels."
        case .unsupportedType: return "Choose a PNG or JPEG image."
        case .cannotDecode: return "The selected file isn't a readable image."
        }
    }
}

enum EditImageImporter {
    static let maxDimension = 8192
    static let maxPixelCount = 40_000_000

    static func pngData(from url: URL) async throws -> Data {
        try await Task.detached(priority: .userInitiated) {
            let values = try url.resourceValues(forKeys: [.fileSizeKey])
            guard (values.fileSize ?? 0) <= ImageClient.maxEditImageBytes else {
                throw ImportedEditImageError.tooLarge
            }
            let data = try Data(contentsOf: url, options: .mappedIfSafe)
            return try pngData(from: data)
        }.value
    }

    static func pngData(from data: Data) throws -> Data {
        guard data.count <= ImageClient.maxEditImageBytes else {
            throw ImportedEditImageError.tooLarge
        }
        let metadataOptions = [kCGImageSourceShouldCache: false] as CFDictionary
        guard let source = CGImageSourceCreateWithData(data as CFData, metadataOptions),
              let type = CGImageSourceGetType(source),
              let contentType = UTType(type as String),
              contentType.conforms(to: .png) || contentType.conforms(to: .jpeg) else {
            throw ImportedEditImageError.unsupportedType
        }
        guard let properties = CGImageSourceCopyPropertiesAtIndex(
            source, 0, metadataOptions
        ) as? [CFString: Any],
              let width = (properties[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
              let height = (properties[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue,
              width > 0, height > 0 else {
            throw ImportedEditImageError.cannotDecode
        }
        guard width <= maxDimension,
              height <= maxDimension,
              width <= maxPixelCount / height else {
            throw ImportedEditImageError.tooManyPixels
        }
        let decodeOptions = [kCGImageSourceShouldCacheImmediately: true] as CFDictionary
        guard let image = CGImageSourceCreateImageAtIndex(source, 0, decodeOptions) else {
            throw ImportedEditImageError.cannotDecode
        }
        guard let png = NSBitmapImageRep(cgImage: image)
                .representation(using: .png, properties: [:]),
              png.count <= ImageClient.maxEditImageBytes else {
            throw ImportedEditImageError.tooLarge
        }
        return png
    }
}

/// The diffusion progress bar: a solid amber fill on the lifecycle
/// track. Determinate (the real step fraction) while denoising; a
/// sliding segment while the model warms up and no fraction exists.
/// The only bar in the app that shows a real diffusion step count.
///
/// It used to carry an amber→gold gradient, a white sheen sweeping the
/// fill, and an amber glow. All three are gone. Amber is a signal
/// colour here, and a two-stop blend with a moving highlight on top
/// reads as decoration wrapped around the signal rather than as the
/// signal itself — the same reason the brand spec allows no gradient on
/// the amber axis anywhere else in the app.
///
/// The sliding segment also now stops under Reduce Motion. It is a
/// perpetual loop, which is the canonical vestibular offender, and it
/// was running unconditionally because its clock is a ``TimelineView``
/// in the parent rather than an ``Animation`` the environment could
/// suppress. Reduced, the segment holds still and the state is carried
/// by the surrounding copy, which already names the phase.
private struct ShimmerProgressBar: View {
    var fraction: Double
    var indeterminate: Bool
    var phase: Double  // 0→1, loops to drive the indeterminate slide

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let fillW = indeterminate ? max(1, w * 0.34)
                                      : max(10, w * min(1, max(0, fraction)))
            let sliding = RapidMotion.shouldPulse(
                isAnimating: indeterminate,
                reduceMotion: reduceMotion
            )
            let slideX = sliding ? (w + fillW) * phase - fillW
                                 : (indeterminate ? (w - fillW) / 2 : 0)
            ZStack(alignment: .leading) {
                Capsule().fill(RapidTheme.bandTrack)
                Capsule()
                    .fill(RapidTheme.brandPrimary)
                    .frame(width: fillW)
                    .offset(x: slideX)
                    .animation(indeterminate ? nil : .easeOut(duration: 0.3), value: fraction)
            }
        }
    }
}
