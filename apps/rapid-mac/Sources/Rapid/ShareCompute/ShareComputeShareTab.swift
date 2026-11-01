import SwiftUI

// MARK: - Relay status bar

/// The bar across the top of the Share workbench.
///
/// Replaces the old "community strip", which showed contributing-Mac counts,
/// requests waiting, and a highest-demand model. None of those three exist:
/// QuickSilver publishes availability, never demand, and there is no queue to
/// have a depth. What belongs here instead is the one fact the user needs
/// before pressing the button — whether this Mac is ready to connect — plus the
/// standing clarification that traffic is live, not queued work to claim.
///
/// Drawn on the band's own darker header (Paper `#17191D` over `#22252B`) so it
/// reads as part of the workbench rather than a light shelf above it.
struct ShareComputeRelayStatusBar: View {
    /// What the relay side of the connection is doing right now.
    enum Status: Equatable {
        case readyToConnect
        case connecting
        case online
        case reconnecting
        case failed

        var title: String {
            switch self {
            case .readyToConnect: return String(localized: "Ready to connect this Mac")
            case .connecting: return String(localized: "Connecting this Mac…")
            case .online: return String(localized: "Serving live requests")
            case .reconnecting: return String(localized: "Reconnecting to the relay…")
            case .failed: return String(localized: "Not connected")
            }
        }

        /// Semantic, never decorative: green only when the relay is actually
        /// usable, amber while in flight, red when it is not.
        var dot: Color {
            switch self {
            case .readyToConnect, .online: return RapidTheme.bandReady
            case .connecting, .reconnecting: return RapidTheme.brandPrimary
            case .failed: return RapidTheme.bandDestructive
            }
        }

        var isInFlight: Bool { self == .connecting || self == .reconnecting }
    }

    let status: Status
    var isNarrow = false

    var body: some View {
        HStack(spacing: RapidTheme.Space.md) {
            HStack(spacing: 14) {
                Group {
                    if status.isInFlight {
                        ProgressView().controlSize(.small)
                    } else {
                        Circle().fill(status.dot).frame(width: 9, height: 9)
                    }
                }
                .frame(width: 9)
                .accessibilityHidden(true)

                VStack(alignment: .leading, spacing: 2) {
                    ShareComputeEyebrow(
                        text: String(localized: "QuickSilver relay"),
                        tone: RapidTheme.bandInkSecondary,
                        size: 11
                    )
                    Text(status.title)
                        .font(.rapid(size: 15, weight: .medium))
                        .foregroundStyle(RapidTheme.bandInk)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .accessibilityElement(children: .combine)

            Spacer(minLength: RapidTheme.Space.md)

            // The load-bearing clarification. Kept on the bar rather than
            // buried in body copy because "is there a queue of tasks I have to
            // accept?" is the first question this surface gets asked.
            if !isNarrow {
                Text("Live requests · no task queue")
                    .font(.rapid(size: 12))
                    .foregroundStyle(RapidTheme.bandInkSecondary)
                    .fixedSize()
            }
        }
        .frame(minHeight: 64)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, isNarrow ? 20 : 32)
        .background(RapidTheme.surfaceBandHeader)
        .overlay(alignment: .bottom) {
            Rectangle().fill(RapidTheme.bandHairlineStrong).frame(height: 1)
        }
        .accessibilityIdentifier("ShareCompute.RelayStatus")
    }
}

// MARK: - Share tab

enum ShareComputeSetupStep: Equatable {
    case key, model, start
    static func current(keyDone: Bool, modelDone: Bool) -> Self {
        if !keyDone { return .key }
        if !modelDone { return .model }
        return .start
    }
    static func canStart(keyDone: Bool, modelDone: Bool) -> Bool { keyDone && modelDone }
}

enum ShareComputeModelSetupState: Equatable {
    case ready, candidate, insufficient, downloading, failed(String), none
    static func decide(hasReady: Bool, hasCandidate: Bool, storage: ShareComputeStorageEligibility, downloading: Bool, failure: String?) -> Self {
        if hasReady { return .ready }
        guard hasCandidate else { return .none }
        if downloading { return .downloading }
        if case .notEnoughSpace = storage { return .insufficient }
        if let failure { return .failed(failure) }
        return .candidate
    }
}

struct ShareComputeShareTab: View {
    let models: [ShareComputeLocalModel]
    let candidates: [ShareComputeLocalModel]
    let selected: ShareComputeLocalModel?
    let relayStatus: ShareComputeRelayStatusBar.Status
    let catalogLoaded: Bool
    let isNarrow: Bool
    let requiresConnection: Bool
    let hasSavedKey: Bool
    var savedKeyLabel: String? = nil
    @Binding var isPickerOpen: Bool
    let freeBytes: Int64?
    @Bindable var downloads: DownloadManager
    let onSelect: (ShareComputeLocalModel) -> Void
    let onStart: () -> Void
    let onOpenQuickSilver: () -> Void
    let onAddKey: () -> Void
    let onGetKey: () -> Void
    @State private var selectorWidth: CGFloat = 0
    @State private var candidateID: String?
    private var activeCandidate: ShareComputeLocalModel? { candidates.first { downloads.isDownloading($0.model.alias) } }
    private var candidate: ShareComputeLocalModel? { activeCandidate ?? candidates.first { $0.id == candidateID } ?? candidates.first }
    private var storage: ShareComputeStorageEligibility { .evaluate(downloadBytes: candidate?.estimatedDownloadBytes, freeBytes: freeBytes) }
    private var downloadState: ShareComputeModelSetupState {
        let job = candidate.flatMap { downloads.job(for: $0.model.alias) }
        var failure: String?
        if case .failed(let message) = job?.status { failure = message }
        return .decide(hasReady: selected?.isReady == true, hasCandidate: candidate != nil, storage: storage, downloading: candidate.map { downloads.isDownloading($0.model.alias) } ?? false, failure: failure)
    }
    private func size(_ bytes: Int64?, entry: ModelEntry?) -> String {
        guard let bytes else { return "" }
        let prefix = entry?.displayedModelSize?.byteCount == bytes ? "" : "~"
        return prefix + DownloadProgress.formatBytes(bytes)
    }
    private var keyDone: Bool { hasSavedKey || !requiresConnection }
    private var modelDone: Bool { selected?.isReady == true }
    private var current: ShareComputeSetupStep { .current(keyDone: keyDone, modelDone: modelDone) }
    private var canStart: Bool { ShareComputeSetupStep.canStart(keyDone: keyDone, modelDone: modelDone) }

    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
            VStack(spacing: 0) {
                HStack {
                    Text("Set up sharing").font(.rapid(size: 15, weight: .semibold))
                    Spacer()
                    Text("\((keyDone ? 1 : 0) + (modelDone ? 1 : 0)) of 3 done")
                        .font(.rapid(size: 12)).foregroundStyle(.secondary)
                }.padding(RapidTheme.Space.lg)
                divider
                setupRow(number: 1, title: String(localized: "QuickSilver provider key"), done: keyDone, step: .key) {
                    if keyDone {
                        Text(savedKeyLabel ?? "Key saved")
                    } else {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("One key from your QuickSilver dashboard lets this Mac share and shows your credits.")
                            Button("Don't have one? Get a key ↗", action: onGetKey)
                                .buttonStyle(.link).fixedSize().accessibilityIdentifier("ShareCompute.Setup.GetKey")
                        }
                    }
                } action: {
                    if keyDone {
                        Button("Change…", action: onAddKey).buttonStyle(.plain)
                            .accessibilityIdentifier("ShareCompute.Setup.AddKey")
                    } else {
                        Button("Add key", action: onAddKey).buttonStyle(.borderedProminent)
                            .accessibilityIdentifier("ShareCompute.Setup.AddKey")
                    }
                }
                divider
                setupRow(number: 2, title: String(localized: "Model to share"), done: modelDone, step: .model) {
                    if !catalogLoaded { Text("Checking downloaded models…") }
                    else if models.isEmpty {
                        switch downloadState {
                        case .none: Text("No models can be shared from this Mac right now.")
                        case .insufficient:
                            if case .notEnoughSpace(_, let required) = storage { Text("Not enough free space (\(size(required, entry: candidate?.entry)) needed).") }
                        case .failed(let message): Text(message)
                        default: Text("Download a model to share. \(size(candidate?.estimatedDownloadBytes, entry: candidate?.entry))")
                        }
                    }
                    else if let selected {
                        HStack(spacing: 5) {
                            if !selected.isReady { Circle().fill(.orange).frame(width: 6, height: 6) }
                            Text(selected.isReady ? "Ready on this Mac" : "Download needed")
                        }
                    } else { Text("One model is shared at a time.") }
                } action: {
                    if catalogLoaded && models.isEmpty && candidate == nil {
                        Button("Open QuickSilver", action: onOpenQuickSilver).buttonStyle(.bordered)
                            .accessibilityIdentifier("ShareCompute.Setup.BrowsePool")
                    } else if catalogLoaded && models.isEmpty, let candidate {
                        HStack(spacing: 8) {
                            Menu {
                                ForEach(candidates) { model in
                                    Button("\(model.title) · \(size(model.estimatedDownloadBytes, entry: model.entry))") { candidateID = model.id }
                                        .accessibilityIdentifier("ShareCompute.Setup.DownloadCandidate.\(model.id)")
                                }
                            } label: { Text("\(candidate.title) · \(size(candidate.estimatedDownloadBytes, entry: candidate.entry))").lineLimit(1) }
                            .frame(maxWidth: 260)
                            .disabled(activeCandidate != nil)
                            .accessibilityIdentifier("ShareCompute.Setup.DownloadPicker")
                            if downloadState == .downloading {
                                ProgressView(value: downloads.job(for: candidate.model.alias)?.progress.progressFraction).controlSize(.small).frame(width: 24)
                                    .accessibilityLabel("Model download progress")
                                Text("Downloading…").font(.rapid(size: 12))
                            } else {
                                Button({ if case .failed = downloadState { return "Retry" }; return "Download" }()) {
                                    _ = downloads.startDownload(alias: candidate.model.alias, hfPath: candidate.entry?.hfRepo, totalBytes: candidate.estimatedDownloadBytes)
                                }.buttonStyle(.borderedProminent).disabled(downloadState == .insufficient)
                                    .accessibilityIdentifier("ShareCompute.Setup.Download")
                            }
                        }
                    } else if catalogLoaded { selector }
                }
                divider
                setupRow(number: 3, title: String(localized: "Start sharing"), done: false, step: .start) {
                    Text(canStart ? "This Mac goes online and earns API credits for served tokens." : "Complete the steps above to go online.")
                } action: {
                    if canStart {
                        Button("Start sharing", action: onStart).buttonStyle(.borderedProminent)
                            .accessibilityIdentifier("ShareCompute.Start")
                    } else {
                        Button("Start sharing", action: onStart).buttonStyle(.bordered).disabled(true)
                            .accessibilityIdentifier("ShareCompute.Start")
                    }
                }
            }.shareGroupedCard()
            Link("About QuickSilver API credits ↗", destination: ShareComputeDestination.credits)
                .font(.rapid(size: 12)).accessibilityIdentifier("ShareCompute.CreditsLink")
        }.frame(maxWidth: 720, alignment: .leading).frame(maxWidth: .infinity, alignment: .leading)
    }
    private var divider: some View {
        Rectangle().fill(RapidTheme.hairline).frame(height: 1).padding(.leading, 54)
    }
    private func setupRow<Content: View, Action: View>(number: Int, title: String, done: Bool, step: ShareComputeSetupStep, @ViewBuilder content: () -> Content, @ViewBuilder action: () -> Action) -> some View {
        HStack(alignment: .top, spacing: RapidTheme.Space.md) {
            ZStack {
                Circle().fill(done ? Color.green : (current == step ? Color.accentColor : Color.clear))
                Circle().strokeBorder(done || current == step ? Color.clear : RapidTheme.hairline, lineWidth: 1)
                if done { Image(systemName: "checkmark").font(.system(size: 10, weight: .bold)).foregroundStyle(.white) }
                else { Text("\(number)").font(.rapid(size: 12, weight: .semibold)).foregroundStyle(current == step ? Color.white : Color.secondary) }
            }.frame(width: 22, height: 22).accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 4) {
                if isNarrow {
                    Text(title).font(.rapid(size: 14, weight: .semibold)).frame(minHeight: 22, alignment: .leading)
                    content().font(.rapid(size: 12)).foregroundStyle(.secondary)
                    action().padding(.top, 4)
                } else {
                    HStack(alignment: .center, spacing: RapidTheme.Space.md) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(title).font(.rapid(size: 14, weight: .semibold)).frame(minHeight: 22, alignment: .leading)
                            content().font(.rapid(size: 12)).foregroundStyle(.secondary)
                        }.frame(maxWidth: .infinity, alignment: .leading)
                        action()
                    }
                }
            }.frame(maxWidth: .infinity, alignment: .leading)
        }.padding(RapidTheme.Space.lg).frame(maxWidth: .infinity, minHeight: 64, alignment: .leading)
        .accessibilityElement(children: .contain)
        .accessibilityLabel(String(format: String(localized: "Step %1$d of 3, %2$@, %3$@"), number, title, done ? String(localized: "done") : (current == step ? String(localized: "next") : String(localized: "not done"))))
    }
    @ViewBuilder private var selector: some View {
        if let selected {
            Button { isPickerOpen.toggle() } label: {
                HStack {
                    Text([selected.title, selected.onDiskSize].compactMap { $0 }.joined(separator: " · ")).lineLimit(1)
                    Spacer(minLength: 4)
                    Image(systemName: "chevron.up.chevron.down")
                }.font(.rapid(size: 13)).frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .accessibilityIdentifier("ShareCompute.SelectedModel")
            .background { GeometryReader { proxy in
                Color.clear.onAppear { selectorWidth = proxy.size.width }
                    .onChange(of: proxy.size.width) { _, width in selectorWidth = width }
            }.accessibilityHidden(true) }
            .shareComputePicker(isPresented: $isPickerOpen, models: models, selectedID: selected.id, width: selectorWidth, onSelect: onSelect)
            .frame(maxWidth: isNarrow ? .infinity : 260)
        }
    }

}

// MARK: - Selector rows

/// The selected-model row: one control, entirely clickable.
///
/// Name, size, readiness, and demand all live inside this single button, which
/// is what the handoff means by keeping related information together — the
/// user should never have to look at two places to know what they are about to
/// share.
struct ShareComputeModelSelectorRow: View {
    let model: ShareComputeLocalModel
    let isOpen: Bool
    var isNarrow = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: RapidTheme.Space.md) {
                VStack(alignment: .leading, spacing: 6) {
                    // Desktop fits name, size, and readiness on one line.
                    // Narrow drops size and readiness to the line below rather
                    // than truncating the model name, which is the one value
                    // the user has to be able to read in full before sharing.
                    if isNarrow {
                        Text(model.title)
                            .font(.rapid(size: 17, weight: .bold))
                            .foregroundStyle(.primary)
                            .lineLimit(1)
                        HStack(spacing: 10) {
                            if let size = model.onDiskSize {
                                Text(size)
                                    .font(.rapid(size: 12))
                                    .foregroundStyle(.secondary)
                            }
                            Text("Ready on this Mac")
                                .font(.rapid(size: 11, weight: .bold))
                                .foregroundStyle(Color.green)
                        }
                    } else {
                        // Name and size on the first line; readiness moves to
                        // ``secondLine`` so the row does not say "Ready on this
                        // Mac" twice.
                        HStack(alignment: .firstTextBaseline, spacing: 14) {
                            Text(model.title)
                                .font(.rapid(size: 20, weight: .semibold))
                                .foregroundStyle(.primary)
                                .lineLimit(1)
                            if let size = model.onDiskSize {
                                Text(size)
                                    .font(.rapid(size: 13))
                                    .foregroundStyle(.secondary)
                                    .fixedSize()
                            }
                        }
                        secondLine
                    }
                }
                Spacer(minLength: 0)
                Image(systemName: "chevron.down")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(RapidTheme.brandPrimary)
                    .rotationEffect(.degrees(isOpen ? 180 : 0))
                    .frame(width: 28)
                    .accessibilityHidden(true)
            }
            .padding(.leading, 18)
            .padding(.trailing, 16)
            .padding(.vertical, 12)
            .frame(maxWidth: .infinity, minHeight: 72, alignment: .leading)
            .background(
                RapidTheme.surfaceRaised,
                in: RoundedRectangle(cornerRadius: RapidTheme.Radius.button)
            )
            .overlay {
                RoundedRectangle(cornerRadius: RapidTheme.Radius.button)
                    .strokeBorder(RapidTheme.hairline, lineWidth: 1)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(
            String(
                format: String(localized: "Model to share: %1$@, ready on this Mac"),
                model.title
            )
        )
        .accessibilityHint(String(localized: "Choose a different local model"))
        .accessibilityAddTraits(.isButton)
        .accessibilityIdentifier("ShareCompute.SelectedModel")
    }

    /// Readiness first, then what the model is. Paper leads this line with the
    /// green "Ready on this Mac" because that is the fact gating the button
    /// beside it; the descriptor follows in quiet ink.
    private var secondLine: some View {
        HStack(spacing: 7) {
            Text("Ready on this Mac")
                .font(.rapid(size: 12, weight: .medium))
                .foregroundStyle(Color.green)
            Text("·")
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
                .accessibilityHidden(true)
            Text(model.detail)
                .font(.rapid(size: 12))
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
    }
}


extension View {
    func shareGroupedCard() -> some View {
        background(RapidTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: RapidTheme.Radius.card, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: RapidTheme.Radius.card, style: .continuous).strokeBorder(RapidTheme.hairline, lineWidth: 1))
    }
}
