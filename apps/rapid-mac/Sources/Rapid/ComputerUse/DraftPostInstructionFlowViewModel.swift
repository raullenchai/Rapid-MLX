import Foundation
import Observation

extension DraftPostPlanningError {
    var userMessage: String {
        switch self {
        case .instructionMissing:
            "Tell Rapid what to draft and where it should go."
        case .instructionTooLarge:
            "The request is too long for this preview. Keep it under 16 KB."
        case .modelUnavailable:
            "The selected local model is no longer available. Start it and try again."
        case .invalidResponse:
            "The local model could not produce a safe plan. Clarify the request and try again."
        case .responseTooLarge:
            "The local model returned more planning data than this preview accepts."
        case .destinationUnavailable:
            "Rapid could not verify the selected browser's destination. Open the destination page and refresh."
        case .permissionMissing:
            "Allow Screen Recording and Accessibility, then refresh the browser windows."
        case .httpStatus(let status):
            "The local model rejected the planning request (HTTP \(status))."
        case .cancelled:
            "Planning was stopped."
        }
    }
}

@MainActor
@Observable
final class DraftPostInstructionFlowViewModel {
    enum Phase: Equatable {
        case loading
        case ready
        case analyzing
        case reviewing
        case running
        case stopping
        case readyForReview(DraftPostFlowMetrics)
        case planningFailed(DraftPostPlanningError)
        case executionFailed(DraftPostFlowFailure, DraftPostFlowMetrics?)
    }

    var phase: Phase = .loading
    var instruction = ""
    var clarificationQuestion: String?
    var windows: [ComputerUseWindowOption] = []
    var destinationID: String?
    var plan: DraftPostPlan?
    var editableDraft = ""

    private let catalog: any ComputerUseWindowListing
    private var planner: (any DraftPostInstructionPlanning)?
    private let destinationInspector: any ComputerUseBrowserDestinationInspecting
    private let driver: any PreparedDraftPostFlowDriving
    private var task: Task<Void, Never>?
    private var generation = 0
    private var plannedDestinationID: String?
    private var plannedDestination: ComputerUseBrowserDestinationIdentity?

    init(
        catalog: any ComputerUseWindowListing = MacOSComputerUseWindowCatalog(),
        planner: (any DraftPostInstructionPlanning)?,
        destinationInspector: any ComputerUseBrowserDestinationInspecting =
            MacOSDraftPostFlowDriver(),
        driver: any PreparedDraftPostFlowDriving = MacOSDraftPostFlowDriver()
    ) {
        self.catalog = catalog
        self.planner = planner
        self.destinationInspector = destinationInspector
        self.driver = driver
    }

    var destinationOptions: [ComputerUseWindowOption] {
        windows.filter {
            MacOSDraftPostFlowDriver.browserBundles.contains(
                $0.selection.bundleIdentifier
            ) && !$0.windowTitle.trimmingCharacters(
                in: .whitespacesAndNewlines
            ).isEmpty
        }
    }

    var hasPlanner: Bool { planner != nil }

    var plannedDestinationDisplayName: String? {
        guard let plannedDestinationID else { return nil }
        return destinationOptions.first(where: { $0.id == plannedDestinationID })?
            .displayName
    }

    var canAnalyze: Bool {
        guard phase == .ready,
              planner != nil,
              !instruction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              let destinationID
        else { return false }
        return destinationOptions.contains(where: { $0.id == destinationID })
    }

    var canExecute: Bool {
        guard phase == .reviewing,
              plan != nil,
              let plannedDestinationID,
              plannedDestinationID == destinationID,
              plannedDestination != nil,
              !editableDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              editableDraft.utf8.count <= MacOSDraftPostFlowDriver.maximumDraftBytes
        else { return false }
        return destinationOptions.contains(where: { $0.id == plannedDestinationID })
    }

    var isActive: Bool {
        phase == .analyzing || phase == .running || phase == .stopping
    }

    /// Installs the app's current local-model session for future analyses.
    /// Existing analysis tasks capture their planner before launch, and plan
    /// review/browser execution never consult this reference.
    func updatePlanner(_ planner: (any DraftPostInstructionPlanning)?) {
        self.planner = planner
    }

    func load() async {
        cancelTask()
        generation += 1
        let requestedGeneration = generation
        phase = .loading
        do {
            let refreshed = try await catalog.windows()
            guard requestedGeneration == generation else { return }
            windows = refreshed
            if !destinationOptions.contains(where: { $0.id == destinationID }) {
                destinationID = nil
            }
            clearPlan()
            phase = .ready
        } catch let error as ComputerUseWindowCatalogError {
            guard requestedGeneration == generation else { return }
            switch error {
            case .permissionsMissing:
                phase = .executionFailed(.permissionMissing, nil)
            case .unavailable:
                phase = .executionFailed(.dependencyFailure, nil)
            }
        } catch {
            guard requestedGeneration == generation else { return }
            phase = .executionFailed(.dependencyFailure, nil)
        }
    }

    func analyze() {
        guard canAnalyze,
              task == nil,
              let planner,
              let destination = destinationOptions.first(where: { $0.id == destinationID })
        else { return }
        generation += 1
        let requestedGeneration = generation
        let requestedInstruction = instruction
        phase = .analyzing
        clarificationQuestion = nil
        task = Task { [weak self] in
            guard let self else { return }
            let result: Result<DraftPostPlanningResult, DraftPostPlanningError>
            var inspectedDestination: ComputerUseBrowserDestinationIdentity?
            do {
                let destinationIdentity = try await self.destinationInspector.destinationIdentity(
                    for: destination
                )
                let planningResult = try await planner.analyze(
                    instruction: requestedInstruction,
                    browserApplication: destination.applicationName,
                    destinationHost: destinationIdentity.host
                )
                if case .ready = planningResult {
                    let currentIdentity = try await self.destinationInspector
                        .destinationIdentity(for: destination)
                    guard currentIdentity == destinationIdentity else {
                        throw DraftPostPlanningError.destinationUnavailable
                    }
                    inspectedDestination = currentIdentity
                }
                result = .success(planningResult)
            } catch let error as DraftPostPlanningError {
                result = .failure(error)
            } catch let error as DraftPostFlowFailure {
                switch error {
                case .cancelled:
                    result = .failure(.cancelled)
                case .permissionMissing:
                    result = .failure(.permissionMissing)
                default:
                    result = .failure(.destinationUnavailable)
                }
            } catch is CancellationError {
                result = .failure(.cancelled)
            } catch {
                result = .failure(.destinationUnavailable)
            }
            guard requestedGeneration == self.generation else { return }
            self.task = nil
            switch result {
            case .success(.needsClarification(let question)):
                self.clarificationQuestion = question
                self.phase = .ready
            case .success(.ready(let plan)):
                self.plan = plan
                self.editableDraft = plan.draft
                self.plannedDestinationID = destination.id
                self.plannedDestination = inspectedDestination
                self.phase = .reviewing
            case .failure(let error):
                self.phase = .planningFailed(error)
            }
        }
    }

    func execute() {
        guard canExecute,
              task == nil,
              let destination = destinationOptions.first(
                where: { $0.id == plannedDestinationID }
              ), let plannedDestination
        else { return }
        generation += 1
        let requestedGeneration = generation
        let draft = editableDraft
        phase = .running
        let coordinator = PreparedDraftPostFlowCoordinator(driver: driver)
        task = Task { [weak self] in
            let outcome = await coordinator.run(
                draft: draft,
                destination: destination,
                expectedDestination: plannedDestination
            )
            guard let self, requestedGeneration == self.generation else { return }
            self.task = nil
            switch outcome {
            case .readyForReview(let metrics):
                self.phase = .readyForReview(metrics)
            case .failed(let failure, let metrics):
                self.phase = .executionFailed(failure, metrics)
            }
        }
    }

    func editRequest() {
        guard !isActive else { return }
        clearPlan()
        phase = .ready
    }

    func returnToPlan() {
        guard case .executionFailed(let failure, _) = phase,
              failure.permitsReviewedRetry,
              plan != nil,
              !isActive
        else { return }
        phase = .reviewing
    }

    func stop() {
        task?.cancel()
        if phase == .running {
            phase = .stopping
        } else if phase == .analyzing {
            generation += 1
            task = nil
            phase = .ready
        }
    }

    func cancelTask() {
        generation += 1
        task?.cancel()
        task = nil
    }

    private func clearPlan() {
        clarificationQuestion = nil
        plan = nil
        editableDraft = ""
        plannedDestinationID = nil
        plannedDestination = nil
    }
}
