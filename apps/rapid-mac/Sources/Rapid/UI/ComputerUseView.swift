import SwiftUI

struct ComputerUseView: View {
    private enum Destination { case home, cleanup }
    @State private var destination: Destination = .home
    @State private var candidates: [DownloadCleanupCandidate] = []
    @State private var selected: Set<URL> = []
    @State private var scanning = false
    @State private var errorMessage: String?
    @State private var resultMessage: String?
    @State private var confirmingCleanup = false
    @State private var operationGeneration = 0

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
                switch destination {
                case .home: home
                case .cleanup: cleanup
                }
            }
            .padding(RapidTheme.Space.xl)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(RapidTheme.surfaceCanvas)
        .confirmationDialog(
            cleanupConfirmationTitle,
            isPresented: $confirmingCleanup,
            titleVisibility: .visible
        ) {
            Button("Move selected files to Trash", role: .destructive) { trashSelection() }
                .accessibilityIdentifier("ComputerUse.Cleanup.Confirm")
            Button("Cancel", role: .cancel) {}
                .accessibilityIdentifier("ComputerUse.Cleanup.Cancel")
        } message: {
            Text("Only the selected top-level files in Downloads will move. You can recover them from Trash.")
        }
        .accessibilityIdentifier("ComputerUse.Panel")
    }

    private var home: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
            HStack(alignment: .top) {
                SectionHeader(
                    "Computer Use",
                    subtitle: "Let Rapid handle useful work across apps on this Mac. Everything runs locally.",
                    emphasis: .page
                )
                Spacer()
                Text("EXPERIMENTAL")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(.orange)
                    .padding(.horizontal, 9).padding(.vertical, 5)
                    .background(.orange.opacity(0.1), in: Capsule())
            }

            VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
                Text("Start with a flow").font(.headline)
                LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: 220, maximum: 320), spacing: 12)],
                    spacing: 12
                ) {
                    ForEach(ComputerUseStarter.catalog) { starter in
                        starterCard(starter)
                    }
                }
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("CREATE YOUR OWN").font(.caption2.weight(.bold)).foregroundStyle(.secondary)
                HStack(spacing: 14) {
                    Image(systemName: "record.circle").font(.title2).foregroundStyle(.secondary)
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Teach Rapid a new task").font(.headline)
                        Text("Show Rapid how you work when no starter fits. Planned for the next Computer Use preview.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Coming next") {}
                        .buttonStyle(.rapidSecondaryCompact)
                        .disabled(true)
                        .accessibilityIdentifier("ComputerUse.Teach.ComingNext")
                }
                .padding(16)
                .background(RapidTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 14))
                .overlay(RoundedRectangle(cornerRadius: 14).stroke(.secondary.opacity(0.2)))
            }
        }
    }

    private func starterCard(_ starter: ComputerUseStarter) -> some View {
        Button {
            guard starter.kind == .freeUpSpace else { return }
            destination = .cleanup
            scanDownloads()
        } label: {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Image(systemName: starter.systemImage).font(.title2).foregroundStyle(RapidTheme.brandPrimaryDeep)
                    Spacer()
                    switch starter.availability {
                    case .comingSoon:
                        Text("COMING NEXT").font(.caption2.weight(.bold)).foregroundStyle(.secondary)
                    case .reserved:
                        Text("RESERVED").font(.caption2.weight(.bold)).foregroundStyle(.secondary)
                    case .available:
                        Image(systemName: "chevron.right").foregroundStyle(.secondary)
                    }
                }
                Text(starter.title).font(.headline).foregroundStyle(.primary)
                Text(starter.summary).font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.leading)
                Text(starter.applications).font(.caption2.weight(.medium)).foregroundStyle(.secondary)
                if starter.availability == .available {
                    Label(starter.approvalNote, systemImage: "checkmark.shield")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity, minHeight: 118, alignment: .topLeading)
            .padding(16)
            .background(RapidTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 14))
            .overlay(
                RoundedRectangle(cornerRadius: 14)
                    .stroke(
                        .secondary.opacity(starter.availability == .reserved ? 0.28 : 0.16),
                        style: StrokeStyle(
                            lineWidth: 1,
                            dash: starter.availability == .reserved ? [5] : []
                        )
                    )
            )
        }
        .buttonStyle(.plain)
        .disabled(starter.availability != .available)
        .opacity(starter.availability == .available ? 1 : 0.58)
        .accessibilityIdentifier("ComputerUse.Starter.\(starter.kind.rawValue)")
    }

    private var cleanup: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.lg) {
            Button("Computer Use", systemImage: "chevron.left") { destination = .home }
                .buttonStyle(.rapidTertiary)
                .accessibilityIdentifier("ComputerUse.Cleanup.Back")
            SectionHeader(
                "Free up space",
                subtitle: "Review top-level files in Downloads that have not changed for 90 days. Nothing is selected automatically.",
                emphasis: .page
            )
            if let resultMessage {
                Label(resultMessage, systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                    .accessibilityIdentifier("ComputerUse.Cleanup.Result")
            }
            if scanning {
                HStack { ProgressView(); Text("Checking Downloads…").foregroundStyle(.secondary) }
                    .accessibilityIdentifier("ComputerUse.Cleanup.Loading")
            } else if let errorMessage {
                Text(errorMessage).foregroundStyle(.red)
                    .accessibilityIdentifier("ComputerUse.Cleanup.Error")
                Button("Try again") { scanDownloads() }
                    .buttonStyle(.rapidSecondaryCompact)
                    .accessibilityIdentifier("ComputerUse.Cleanup.Retry")
            } else if candidates.isEmpty {
                ContentUnavailableView(
                    "No old files found",
                    systemImage: "checkmark.circle",
                    description: Text("Rapid found no top-level files in Downloads older than 90 days.")
                )
                .accessibilityIdentifier("ComputerUse.Cleanup.Empty")
            } else {
                Text("\(candidates.count) files found · Select only files you recognize")
                    .font(.callout).foregroundStyle(.secondary)
                VStack(spacing: 0) {
                    ForEach(Array(candidates.enumerated()), id: \.element.id) { index, candidate in
                        Toggle(isOn: selectionBinding(candidate.url)) {
                            HStack {
                                Image(systemName: "doc").foregroundStyle(.secondary)
                                VStack(alignment: .leading) {
                                    Text(candidate.name).lineLimit(1)
                                    Text(candidate.modifiedAt.formatted(date: .abbreviated, time: .omitted))
                                        .font(.caption).foregroundStyle(.secondary)
                                }
                                Spacer()
                                Text(ByteCountFormatter.string(fromByteCount: candidate.byteCount, countStyle: .file))
                                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                            }
                        }
                        .toggleStyle(.checkbox)
                        .accessibilityIdentifier("ComputerUse.Cleanup.File.\(index)")
                        .padding(12)
                        Divider()
                    }
                }
                .background(RapidTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 12))
                HStack {
                    Text("Selected: \(selected.count) files · \(selectedSize)")
                        .font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Button("Move selected to Trash") { confirmingCleanup = true }
                        .buttonStyle(.rapidDestructive)
                        .disabled(selected.isEmpty)
                        .accessibilityIdentifier("ComputerUse.Cleanup.MoveToTrash")
                }
            }
        }
        .accessibilityIdentifier("ComputerUse.Cleanup")
    }

    private var downloadsURL: URL? {
        FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first
    }

    private func scanDownloads(resultMessage: String? = nil) {
        guard let downloadsURL else {
            errorMessage = "Rapid could not locate Downloads on this Mac."
            return
        }
        scanning = true
        errorMessage = nil
        self.resultMessage = nil
        operationGeneration += 1
        let generation = operationGeneration
        Task {
            do {
                let result = try await Task.detached {
                    try DownloadCleanup.scan(downloadsURL: downloadsURL)
                }.value
                guard generation == operationGeneration else { return }
                candidates = result
                selected = []
                self.resultMessage = resultMessage
            } catch {
                guard generation == operationGeneration else { return }
                errorMessage = "Rapid could not read Downloads: \(error.localizedDescription)"
            }
            scanning = false
        }
    }

    private func selectionBinding(_ url: URL) -> Binding<Bool> {
        Binding(get: { selected.contains(url) }, set: { value in
            if value { selected.insert(url) } else { selected.remove(url) }
        })
    }

    private var selectedCandidates: [DownloadCleanupCandidate] {
        candidates.filter { selected.contains($0.url) }
    }

    private var selectedSize: String {
        ByteCountFormatter.string(
            fromByteCount: selectedCandidates.reduce(0) { $0 + $1.byteCount },
            countStyle: .file
        )
    }

    private var cleanupConfirmationTitle: String {
        "Move \(selected.count) selected file\(selected.count == 1 ? "" : "s") (\(selectedSize)) to Trash?"
    }

    private func trashSelection() {
        guard let downloadsURL else { return }
        let selection = selectedCandidates
        let count = selection.count
        let size = selectedSize
        scanning = true
        errorMessage = nil
        resultMessage = nil
        operationGeneration += 1
        let generation = operationGeneration
        Task {
            let result = await Task.detached {
                DownloadCleanup.moveToTrash(selection, downloadsURL: downloadsURL)
            }.value
            guard generation == operationGeneration else { return }
            if let failure = result.failureDescription {
                let prefix = result.movedCount == 0
                    ? "No files were moved."
                    : "Moved \(result.movedCount) of \(count) files to Trash."
                let message = "\(prefix) Cleanup stopped safely: \(failure)"
                do {
                    let refreshed = try await Task.detached {
                        try DownloadCleanup.scan(downloadsURL: downloadsURL)
                    }.value
                    guard generation == operationGeneration else { return }
                    candidates = refreshed
                    selected = []
                } catch {
                    // Preserve the actionable move failure.
                }
                guard generation == operationGeneration else { return }
                errorMessage = message
                scanning = false
            } else {
                scanDownloads(
                    resultMessage: "Moved \(count) file\(count == 1 ? "" : "s") (\(size)) to Trash."
                )
            }
        }
    }
}
