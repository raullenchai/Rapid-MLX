import SwiftUI

struct FreeUpSpaceFlowSheet: View {
    @Environment(\.dismiss) private var dismiss
    @State private var viewModel: FreeUpSpaceFlowViewModel

    init(
        service: (any FreeUpSpaceServicing)? = MacOSDownloadsCleanupService(),
        now: Date = Date()
    ) {
        let resolvedService = service ?? UnavailableFreeUpSpaceService()
        _viewModel = State(initialValue: FreeUpSpaceFlowViewModel(
            service: resolvedService,
            now: { now }
        ))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Free up space")
                        .font(.title2.weight(.semibold))
                    Text("Review old files in Downloads. Rapid moves only the files you approve to Trash.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Close") { dismiss() }
                    .disabled(viewModel.isActive)
                    .buttonStyle(.rapidSecondaryCompact)
                    .accessibilityIdentifier("ComputerUse.FreeSpace.Close")
            }

            Divider()

            switch viewModel.phase {
            case .scanning:
                progress(
                    title: "Reviewing Downloads…",
                    detail: "Rapid is checking file names, sizes, and last-modified dates. Nothing is moving."
                )
                Button("Stop") { viewModel.stop() }
                    .buttonStyle(.rapidSecondaryCompact)
                    .accessibilityIdentifier("ComputerUse.FreeSpace.StopScan")

            case .reviewing:
                review

            case .confirming:
                confirmation

            case .moving:
                progress(
                    title: "Moving approved files to Trash…",
                    detail: "Rapid verifies every file again before moving it and stops between files if you cancel."
                )
                Button("Stop after current file") { viewModel.stop() }
                    .buttonStyle(.rapidSecondaryCompact)
                    .accessibilityIdentifier("ComputerUse.FreeSpace.StopMove")

            case .finished(let outcome):
                result(outcome)

            case .failed(let error):
                failure(error)
            }

            Spacer(minLength: 0)
        }
        .padding(24)
        .frame(width: 720, height: 680)
        .task { viewModel.scan() }
        .onDisappear { viewModel.cancelTask() }
        .interactiveDismissDisabled(viewModel.isActive)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Free up space flow")
    }

    private var review: some View {
        VStack(alignment: .leading, spacing: 14) {
            if viewModel.candidates.isEmpty {
                Label("No old files to review", systemImage: "checkmark.circle.fill")
                    .font(.headline)
                    .foregroundStyle(.green)
                Text("Downloads has no top-level files that have been unchanged for at least 30 days.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Button("Scan again") { viewModel.scan() }
                    .buttonStyle(.rapidSecondaryCompact)
                    .accessibilityIdentifier("ComputerUse.FreeSpace.ScanAgain")
            } else {
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Choose files to move")
                            .font(.headline)
                        Text("Nothing is selected by default. Folders, hidden files, active downloads, and symbolic links are excluded.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    if viewModel.selectedIDs.count == viewModel.candidates.count {
                        Button("Clear") { viewModel.clearSelection() }
                            .buttonStyle(.rapidSecondaryCompact)
                            .accessibilityIdentifier("ComputerUse.FreeSpace.Clear")
                    } else {
                        Button("Select all") { viewModel.selectAll() }
                            .buttonStyle(.rapidSecondaryCompact)
                            .accessibilityIdentifier("ComputerUse.FreeSpace.SelectAll")
                    }
                }

                ScrollView {
                    LazyVStack(spacing: 8) {
                        ForEach(viewModel.candidates) { candidate in
                            candidateRow(candidate)
                        }
                    }
                }
                .frame(maxHeight: 390)
                .accessibilityIdentifier("ComputerUse.FreeSpace.Candidates")

                HStack {
                    Text(selectionSummary)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Review move") { viewModel.reviewMove() }
                        .buttonStyle(.rapidPrimaryCompact)
                        .disabled(viewModel.selectedCandidates.isEmpty)
                        .accessibilityIdentifier("ComputerUse.FreeSpace.ReviewMove")
                }
            }
        }
    }

    private var confirmation: some View {
        VStack(alignment: .leading, spacing: 16) {
            Label("Confirm what moves to Trash", systemImage: "checkmark.shield.fill")
                .font(.headline)
                .foregroundStyle(.orange)
            Text("Rapid will move \(viewModel.selectedCandidates.count) selected \(fileWord(viewModel.selectedCandidates.count)) (\(bytes(viewModel.selectedByteCount))) from Downloads to Trash.")
                .font(.callout)
            VStack(alignment: .leading, spacing: 7) {
                ForEach(viewModel.selectedCandidates.prefix(8)) { candidate in
                    HStack {
                        Image(systemName: "doc")
                            .foregroundStyle(.secondary)
                        Text(candidate.name)
                            .lineLimit(1)
                        Spacer()
                        Text(bytes(candidate.byteCount))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
                if viewModel.selectedCandidates.count > 8 {
                    Text("and \(viewModel.selectedCandidates.count - 8) more…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(12)
            .background(RapidTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 10))

            Text("This is recoverable from Trash. Disk space is not reclaimed until you empty Trash yourself.")
                .font(.caption)
                .foregroundStyle(.secondary)

            HStack {
                Button("Back") { viewModel.returnToSelection() }
                    .buttonStyle(.rapidSecondaryCompact)
                    .accessibilityIdentifier("ComputerUse.FreeSpace.Back")
                Spacer()
                Button("Move to Trash") { viewModel.moveSelectedToTrash() }
                    .buttonStyle(.rapidPrimaryCompact)
                    .accessibilityIdentifier("ComputerUse.FreeSpace.ConfirmMove")
            }
        }
        .accessibilityIdentifier("ComputerUse.FreeSpace.Confirmation")
    }

    private func result(_ outcome: FreeUpSpaceMoveOutcome) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Label(
                outcome.failures.isEmpty && !outcome.wasCancelled
                    ? "Approved files moved to Trash"
                    : "Cleanup stopped safely",
                systemImage: outcome.failures.isEmpty && !outcome.wasCancelled
                    ? "checkmark.circle.fill"
                    : "exclamationmark.circle.fill"
            )
            .font(.headline)
            .foregroundStyle(outcome.failures.isEmpty && !outcome.wasCancelled ? .green : .orange)

            Text("Moved \(outcome.moved.count) \(fileWord(outcome.moved.count)) totaling \(bytes(outcome.movedByteCount)).")
                .font(.callout)
            if !outcome.failures.isEmpty {
                Text("\(outcome.failures.count) \(fileWord(outcome.failures.count)) stayed in Downloads because they changed, could not be moved, or could not be verified.")
                    .font(.callout)
            }
            if outcome.wasCancelled {
                Text("You stopped the run. Files already moved remain in Trash; unprocessed files stayed in Downloads.")
                    .font(.callout)
            }
            Text("Empty Trash yourself when you are ready to reclaim the disk space.")
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack {
                Button("Scan again") { viewModel.scan() }
                    .buttonStyle(.rapidSecondaryCompact)
                    .accessibilityIdentifier("ComputerUse.FreeSpace.ScanAfterResult")
                Spacer()
                Button("Done") { dismiss() }
                    .buttonStyle(.rapidPrimaryCompact)
                    .accessibilityIdentifier("ComputerUse.FreeSpace.Done")
            }
        }
        .accessibilityIdentifier("ComputerUse.FreeSpace.Result")
    }

    private func failure(_ error: FreeUpSpaceScanError) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Label("Rapid could not review Downloads", systemImage: "folder.badge.questionmark")
                .font(.headline)
                .foregroundStyle(.orange)
            Text(errorMessage(error))
                .font(.callout)
            HStack {
                Button("Try again") { viewModel.scan() }
                    .buttonStyle(.rapidSecondaryCompact)
                    .accessibilityIdentifier("ComputerUse.FreeSpace.Retry")
                Spacer()
                Button("Close") { dismiss() }
                    .buttonStyle(.rapidPrimaryCompact)
                    .accessibilityIdentifier("ComputerUse.FreeSpace.CloseAfterFailure")
            }
        }
        .accessibilityIdentifier("ComputerUse.FreeSpace.Failure")
    }

    private func candidateRow(_ candidate: FreeUpSpaceCandidate) -> some View {
        let selected = viewModel.selectedIDs.contains(candidate.id)
        return Button { viewModel.toggle(candidate) } label: {
            HStack(spacing: 12) {
                Image(systemName: selected ? "checkmark.square.fill" : "square")
                    .font(.title3)
                    .foregroundStyle(selected ? RapidTheme.brandPrimaryDeep : .secondary)
                Image(systemName: "doc")
                    .foregroundStyle(.secondary)
                VStack(alignment: .leading, spacing: 3) {
                    Text(candidate.name)
                        .foregroundStyle(.primary)
                        .lineLimit(1)
                    Text("Last changed \(candidate.ageInDays(relativeTo: viewModel.scanDate)) days ago")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Text(bytes(candidate.byteCount))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            .padding(10)
            .background(
                selected ? RapidTheme.brandPrimaryDeep.opacity(0.08) : RapidTheme.surfaceRaised,
                in: RoundedRectangle(cornerRadius: 10)
            )
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(selected ? "Selected" : "Not selected"), \(candidate.name), \(bytes(candidate.byteCount))")
        .accessibilityIdentifier("ComputerUse.FreeSpace.Candidate.\(candidate.id)")
    }

    private func progress(title: String, detail: String) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            ProgressView()
            Text(title).font(.headline)
            Text(detail).font(.caption).foregroundStyle(.secondary)
        }
    }

    private var selectionSummary: String {
        guard !viewModel.selectedCandidates.isEmpty else { return "No files selected" }
        return "\(viewModel.selectedCandidates.count) selected · \(bytes(viewModel.selectedByteCount))"
    }

    private func bytes(_ count: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: count, countStyle: .file)
    }

    private func fileWord(_ count: Int) -> String { count == 1 ? "file" : "files" }

    private func errorMessage(_ error: FreeUpSpaceScanError) -> String {
        switch error {
        case .downloadsUnavailable:
            "The Downloads folder is not available on this Mac."
        case .permissionDenied:
            "macOS did not allow Rapid to read Downloads. Review Files & Folders access in System Settings, then try again."
        case .enumerationFailed:
            "Downloads could not be read. No files were changed."
        case .cancelled:
            "The scan was stopped. No files were changed."
        }
    }
}

private actor UnavailableFreeUpSpaceService: FreeUpSpaceServicing {
    func scan(now: Date) async throws -> [FreeUpSpaceCandidate] {
        throw FreeUpSpaceScanError.downloadsUnavailable
    }

    func moveToTrash(
        _ candidates: [FreeUpSpaceCandidate]
    ) async -> FreeUpSpaceMoveOutcome {
        FreeUpSpaceMoveOutcome()
    }
}
