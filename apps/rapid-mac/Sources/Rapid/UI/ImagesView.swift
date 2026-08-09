import AppKit
import SwiftUI

/// The Images tab. Deliberately mirrors ``ChatView``: a scrollable results
/// area on top and, at the bottom, the *same* compose box — a `surfaceRaised`
/// rounded field with the model picker + submit button clustered at its
/// bottom-right — so model selection and input feel identical across tabs.
struct ImagesView: View {
    @Bindable var viewModel: ImageGenViewModel
    @Bindable var server: ServerManager

    private let contentMaxWidth: CGFloat = RapidTheme.Layout.contentMaxWidth

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
                Image(nsImage: nsImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 14))
                    .overlay(
                        RoundedRectangle(cornerRadius: 14).stroke(RapidTheme.hairline, lineWidth: 1)
                    )
                    .overlay(alignment: .topTrailing) { saveOverlay(active) }
                    .accessibilityIdentifier("Images.Stage")
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
    private func saveOverlay(_ image: GeneratedImage) -> some View {
        if !viewModel.isGenerating {
            Button {
                save(image)
            } label: {
                Image(systemName: "square.and.arrow.down")
                    .font(.system(size: 12, weight: .semibold))
                    .frame(width: 28, height: 28)
                    .background(.ultraThinMaterial, in: Circle())
            }
            .buttonStyle(.plain)
            .padding(10)
            .help("Save image")
            .accessibilityIdentifier("Images.Result.Save")
        }
    }

    /// The empty hero — the same shape as ChatView's, with the cheetah mark
    /// and readiness-driven copy. Just "Draw anything" instead of "Ask
    /// anything", and no Connect-tools / Speed actions.
    private var emptyStage: some View {
        EmptyState(
            title: "Draw anything",
            message: readiness.isReady
                ? "Describe what you want to see, then press Generate."
                : readiness.emptyStateSubtitle,
            hint: readiness.isReady ? nil : readiness.emptyStateHint,
            markDiameter: 92,
            mark: { CheetahLogo(size: 68) },
            actions: { EmptyView() }
        )
        .accessibilityIdentifier("Images.EmptyState")
    }

    // MARK: - Readiness (mirrors ChatView: same "load the model first" flow)

    /// Readiness for the selected image model. Because rapid serves one model
    /// per process, this reports "isn't running" whenever the server is
    /// serving something else (e.g. a chat model) — exactly the "load FLUX
    /// first" guidance chat gives, produced by the same resolver.
    private var readiness: ModelReadiness {
        ModelReadiness.resolve(
            serverState: server.state,
            alias: viewModel.selectedAlias,
            cacheState: imageCacheState,
            sizeText: viewModel.imageModels
                .first { $0.alias == viewModel.selectedAlias }?.sizeOnDisk,
            progress: startupProgress,
            failure: nil
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

    /// The banner's next-step action: start (or download-and-start) the
    /// selected image model, switching the single server process to it.
    private func handleReadinessAction(_ action: ModelReadiness.Action) {
        switch action {
        case .chooseModel:
            break  // the composer's model picker owns this step
        case .downloadAndStart(let target), .start(let target), .retry(let target):
            let hf = viewModel.imageModels.first { $0.alias == target }?.hfRepo
            // ``ensureServing`` (not ``start``): the user is almost always
            // switching FROM a running chat model TO the image model. Plain
            // ``start`` is cold-start only — it no-ops when a child is already
            // serving — so it would silently do nothing here; ``ensureServing``
            // stops the current model and brings the target up.
            Task { _ = await server.ensureServing(alias: target, hfPath: hf) }
        }
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
                        if denoising {
                            Circle()
                                .fill(RapidTheme.brandAmber)
                                .frame(width: 9, height: 9)
                                .shadow(color: RapidTheme.brandAmber.opacity(0.9), radius: 5)
                                .scaleEffect(0.65 + 0.35 * (0.5 + 0.5 * sin(phase * .pi * 2)))
                            Text(viewModel.cancelling ? "Stopping…" : "Generating")
                                .font(.system(size: 14, weight: .semibold))
                        } else {
                            ProgressView().controlSize(.small)
                            Text(viewModel.cancelling
                                 ? "Stopping…"
                                 : "Warming up \(viewModel.selectedDisplayName)")
                                .font(.system(size: 14, weight: .semibold))
                                .lineLimit(1).truncationMode(.middle)
                        }
                        Spacer(minLength: 8)
                        if denoising {
                            Text("\(step) / \(total)")
                                .font(.system(size: 12, weight: .semibold, design: .monospaced))
                                .foregroundStyle(.secondary)
                                .monospacedDigit()
                        }
                        Button { viewModel.cancel() } label: {
                            Image(systemName: "xmark")
                                .font(.system(size: 10, weight: .bold))
                                .frame(width: 22, height: 22)
                                .background(Color.primary.opacity(0.08), in: Circle())
                        }
                        .buttonStyle(.plain)
                        .disabled(viewModel.cancelling)
                        .help("Cancel")
                        .accessibilityIdentifier("Images.Cancel")
                    }

                    ShimmerProgressBar(fraction: fraction, indeterminate: !denoising, phase: phase)
                        .frame(height: 10)

                    HStack {
                        Text(String(format: "%.1fs", max(0, elapsed)))
                            .font(.system(size: 11, weight: .medium, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                        Spacer()
                        // ETA from the denoise-phase clock, not total elapsed —
                        // otherwise cold-load time inflates the per-step estimate.
                        let denoiseElapsed = viewModel.denoiseStartedAt
                            .map { context.date.timeIntervalSince($0) } ?? elapsed
                        Text(denoising
                             ? (etaText(step: step, total: total, elapsed: denoiseElapsed) ?? "finishing…")
                             : "First run — only happens once")
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(18)
                .frame(width: 340)
                .background(.ultraThinMaterial,
                            in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .strokeBorder(.white.opacity(0.12), lineWidth: 1)
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
                ForEach(viewModel.results) { image in
                    filmstripThumb(image)
                }
            }
            .padding(.vertical, 2)
        }
        .frame(height: 64)
        .accessibilityIdentifier("Images.Gallery")
    }

    private func filmstripThumb(_ image: GeneratedImage) -> some View {
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
        // Per-item id, mirroring `Sidebar.Conversation.<uuid>`. The
        // enclosing `Images.Gallery` identifies the strip, not the
        // thumbnails inside it — without this the only selectable item in
        // the filmstrip is unreachable by identifier, so a harness can
        // neither pick a specific image nor assert which one is active.
        .accessibilityIdentifier("Images.Thumb.\(image.id)")
        .accessibilityLabel(selected ? "Generated image, selected" : "Generated image")
    }

    // MARK: - Composer (mirrors ChatView's compose box)

    private var composer: some View {
        VStack(spacing: RapidTheme.Space.sm) {
            if !readiness.isReady {
                ReadinessBanner(
                    readiness: readiness,
                    attentionToken: blockedSendAttempts,
                    onAction: handleReadinessAction
                )
                .frame(maxWidth: contentMaxWidth)
                .frame(maxWidth: .infinity)
            } else if let error = viewModel.errorMessage {
                InlineNotice(message: error, tone: .error)
                    .frame(maxWidth: contentMaxWidth)
                    .frame(maxWidth: .infinity)
            }
            if viewModel.prompt.isEmpty {
                starters
                    .frame(maxWidth: contentMaxWidth)
                    .frame(maxWidth: .infinity)
            }
            VStack(spacing: RapidTheme.Space.sm - 2) {
                ComposeField(
                    text: $viewModel.prompt,
                    focusToken: composeFocusToken,
                    isStreaming: viewModel.isGenerating,
                    placeholder: readiness.isReady
                        ? "Describe the image you want…"
                        : readiness.composerPlaceholder,
                    onSubmit: runSubmit,
                    onCancel: { viewModel.cancel() }
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

    /// Bottom row of the compose box: aspect on the left, then the inline
    /// model picker + submit clustered on the right — the same
    /// `model ▾  ⬆` grouping ChatView uses.
    private var composerControls: some View {
        HStack(spacing: RapidTheme.Space.sm) {
            aspectPicker
            Spacer(minLength: 0)
            modelPicker
            sendOrStopButton
        }
    }

    private var starters: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 7) {
                ForEach(ImageGenViewModel.starters, id: \.self) { starter in
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
                    .accessibilityIdentifier("Images.Starter")
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
        .accessibilityIdentifier("Images.Aspect")
    }

    /// The inline model picker — same composer-embedded chip as chat
    /// (``ModelPickerBar`` in `composerStyle`): borderless, a fill on hover,
    /// a cache glyph per row, scaling to any number of image models.
    private var modelPicker: some View {
        Menu {
            if viewModel.imageModels.isEmpty {
                Text(viewModel.catalogLoaded ? "No image models available" : "Loading…")
            } else {
                ForEach(viewModel.imageModels) { entry in
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
                Image(systemName: "photo")
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
        .onHover { pickerHovering = $0 }
        .help(viewModel.selectedAlias.isEmpty ? "Choose a model" : "Model: \(viewModel.selectedAlias)")
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
            .help(readiness.isReady ? "Generate" : "Load the model first")
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
}

/// The diffusion progress bar: a rounded amber→gold gradient fill with a soft
/// glow and a sheen that sweeps across it, over a faint track. Determinate
/// (true step fraction) while denoising; a sliding segment while the model
/// warms up. The only bar in the app that shows a real diffusion step count.
private struct ShimmerProgressBar: View {
    var fraction: Double
    var indeterminate: Bool
    var phase: Double  // 0→1, loops to drive the sheen

    private var fillGradient: LinearGradient {
        LinearGradient(
            colors: [RapidTheme.brandAmber, Color(red: 1.0, green: 0.85, blue: 0.47)],
            startPoint: .leading, endPoint: .trailing)
    }

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let fillW = indeterminate ? max(1, w * 0.34)
                                      : max(10, w * min(1, max(0, fraction)))
            let slideX = indeterminate ? (w + fillW) * phase - fillW : 0
            ZStack(alignment: .leading) {
                Capsule().fill(Color.primary.opacity(0.12))
                ZStack(alignment: .leading) {
                    Capsule().fill(fillGradient)
                    // A narrow highlight sweeping the filled portion.
                    Capsule()
                        .fill(LinearGradient(
                            colors: [.clear, .white.opacity(0.55), .clear],
                            startPoint: .leading, endPoint: .trailing))
                        .frame(width: 64)
                        .offset(x: (fillW + 64) * phase - 64)
                }
                .frame(width: fillW)
                .clipShape(Capsule())
                .shadow(color: RapidTheme.brandAmber.opacity(0.55), radius: 6)
                .offset(x: slideX)
                .animation(indeterminate ? nil : .easeOut(duration: 0.3), value: fraction)
            }
        }
    }
}
