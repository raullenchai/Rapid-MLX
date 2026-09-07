import Foundation
import Observation

@MainActor
@Observable
final class FreeUpSpaceFlowViewModel {
    enum Phase: Equatable {
        case scanning
        case reviewing
        case confirming
        case moving
        case finished(FreeUpSpaceMoveOutcome)
        case failed(FreeUpSpaceScanError)
    }

    var phase: Phase = .scanning
    var candidates: [FreeUpSpaceCandidate] = []
    var selectedIDs: Set<String> = []
    var scanDate = Date()

    private let service: any FreeUpSpaceServicing
    private let now: @Sendable () -> Date
    private var task: Task<Void, Never>?
    private var generation = 0

    init(
        service: any FreeUpSpaceServicing,
        now: @escaping @Sendable () -> Date = Date.init
    ) {
        self.service = service
        self.now = now
    }

    var selectedCandidates: [FreeUpSpaceCandidate] {
        candidates.filter { selectedIDs.contains($0.id) }
    }

    var selectedByteCount: Int64 {
        selectedCandidates.reduce(0) { $0 + $1.byteCount }
    }

    var isActive: Bool { phase == .scanning || phase == .moving }

    func scan() {
        cancelTask()
        generation += 1
        let requestedGeneration = generation
        phase = .scanning
        candidates = []
        selectedIDs = []
        scanDate = now()
        let requestedScanDate = scanDate
        task = Task { [weak self] in
            guard let self else { return }
            do {
                let result = try await service.scan(now: requestedScanDate)
                guard requestedGeneration == generation else { return }
                task = nil
                candidates = result
                phase = .reviewing
            } catch let error as FreeUpSpaceScanError {
                guard requestedGeneration == generation else { return }
                task = nil
                phase = .failed(error)
            } catch is CancellationError {
                guard requestedGeneration == generation else { return }
                task = nil
                phase = .failed(.cancelled)
            } catch {
                guard requestedGeneration == generation else { return }
                task = nil
                phase = .failed(.enumerationFailed)
            }
        }
    }

    func toggle(_ candidate: FreeUpSpaceCandidate) {
        guard phase == .reviewing else { return }
        if selectedIDs.contains(candidate.id) {
            selectedIDs.remove(candidate.id)
        } else {
            selectedIDs.insert(candidate.id)
        }
    }

    func selectAll() {
        guard phase == .reviewing else { return }
        selectedIDs = Set(candidates.map(\.id))
    }

    func clearSelection() {
        guard phase == .reviewing else { return }
        selectedIDs = []
    }

    func reviewMove() {
        guard phase == .reviewing, !selectedCandidates.isEmpty else { return }
        phase = .confirming
    }

    func returnToSelection() {
        guard phase == .confirming else { return }
        phase = .reviewing
    }

    func moveSelectedToTrash() {
        guard phase == .confirming, task == nil else { return }
        let selected = selectedCandidates
        guard !selected.isEmpty else {
            phase = .reviewing
            return
        }
        generation += 1
        let requestedGeneration = generation
        phase = .moving
        task = Task { [weak self] in
            guard let self else { return }
            let outcome = await service.moveToTrash(selected)
            guard requestedGeneration == generation else { return }
            task = nil
            phase = .finished(outcome)
        }
    }

    func stop() {
        guard isActive else { return }
        task?.cancel()
    }

    func cancelTask() {
        generation += 1
        task?.cancel()
        task = nil
    }
}
