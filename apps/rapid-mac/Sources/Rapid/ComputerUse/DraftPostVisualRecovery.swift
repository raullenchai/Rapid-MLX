import Foundation

/// Runtime-only connection details for the already-running, app-owned local
/// Computer Use model. The bearer remains in memory and is never persisted or
/// included in workflow/ledger state.
struct DraftPostVisualRuntime: Equatable, Sendable {
    typealias SessionValidator = @MainActor @Sendable () -> Bool

    let baseURL: URL
    let model: String
    let bearerToken: String
    private let sessionValidator: SessionValidator

    init?(
        host: String,
        port: Int,
        model: String?,
        bearerToken: String?,
        sessionValidator: @escaping SessionValidator
    ) {
        guard host == "127.0.0.1",
              (1 ... 65_535).contains(port),
              let model,
              !model.isEmpty,
              let bearerToken,
              !bearerToken.isEmpty,
              let baseURL = URL(string: "http://127.0.0.1:\(port)/v1")
        else { return nil }
        self.baseURL = baseURL
        self.model = model
        self.bearerToken = bearerToken
        self.sessionValidator = sessionValidator
    }

    init?(
        profile: ServerModelProfile?,
        selectedAlias: String,
        host: String,
        port: Int,
        bearerToken: String?,
        sessionValidator: @escaping SessionValidator
    ) {
        guard let profile,
              profile.id.caseInsensitiveCompare(selectedAlias) == .orderedSame,
              profile.toolCallParser?.caseInsensitiveCompare("ui_tars") == .orderedSame
        else { return nil }
        self.init(
            host: host,
            port: port,
            model: profile.id,
            bearerToken: bearerToken,
            sessionValidator: sessionValidator
        )
    }

    /// Production constructor: the tuple accepted by this snapshot must still
    /// describe the app-owned server immediately around every visual request.
    @MainActor
    init?(
        profile: ServerModelProfile?,
        selectedAlias: String,
        host: String,
        port: Int,
        bearerToken: String?,
        liveServer server: ServerManager
    ) {
        guard let profile, let bearerToken else { return nil }
        let expectedModel = profile.id
        self.init(
            profile: profile,
            selectedAlias: selectedAlias,
            host: host,
            port: port,
            bearerToken: bearerToken,
            sessionValidator: { [weak server] in
                guard let server,
                      server.host == host,
                      server.activePort == port,
                      server.activeBearer == bearerToken,
                      let currentProfile = server.activeModelProfile
                else { return false }
                return currentProfile.id.caseInsensitiveCompare(expectedModel)
                    == .orderedSame
                    && currentProfile.toolCallParser?.caseInsensitiveCompare(
                        "ui_tars"
                    ) == .orderedSame
                    && server.isModelResident(selectedAlias)
            }
        )
    }

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.baseURL == rhs.baseURL
            && lhs.model == rhs.model
            && lhs.bearerToken == rhs.bearerToken
    }

    func makeRecovery() -> (any DraftPostVisualRecovering)? {
        guard let configuration = try? LocalComputerUseVisualGrounder.Configuration(
            baseURL: baseURL,
            model: model,
            bearerToken: bearerToken,
            deadline: .seconds(30),
            wireContract: .uiTars
        ) else { return nil }
        return MacOSDraftPostVisualRecovery(
            configuration: configuration,
            sessionValidator: sessionValidator
        )
    }
}

/// A bounded bridge from one failed semantic lookup to one visual focus.
/// Each attempt re-captures the exact selected window after inference and
/// requires byte-identical pixels before focusing an AX-verified empty editor.
actor MacOSDraftPostVisualRecovery: DraftPostVisualRecovering {
    typealias Attempt = @Sendable (
        ComputerUseWindowOption,
        String
    ) async throws -> Void

    private static let stepID = "draft-post.visual-composer"
    // A model-format miss and a transient pixel drift are both safe to retry:
    // no input has been emitted yet. The shared workflow model hard-caps this
    // value at three so recovery can never turn into an open-ended agent loop.
    private static let maximumAttempts = 3

    private let attempt: Attempt

    init(
        configuration: LocalComputerUseVisualGrounder.Configuration,
        sessionValidator: @escaping DraftPostVisualRuntime.SessionValidator,
        captureSource: any ComputerUseWindowCapturing =
            ScreenCaptureKitComputerUseCapture(),
        transport: any LocalComputerUseGroundingTransport =
            URLSessionComputerUseGroundingTransport(),
        actuator: any DraftPostComposerActuating = AXDraftPostComposerActuator()
    ) {
        self.attempt = { destination, documentIdentity in
            let vault = ComputerUseObservationVault(maximumArtifacts: 4)
            do {
                let observer = MacOSComputerUseObserver(
                    selections: [Self.stepID: destination.selection],
                    vault: vault,
                    captureSource: captureSource
                )
                let grounder = LocalComputerUseVisualGrounder(
                    configuration: configuration,
                    vault: vault,
                    transport: transport
                )
                let step = Self.composerStep
                let groundingObservation = try await observer.observe(for: step)
                let action = try await Self.withValidatedSession(
                    sessionValidator
                ) {
                    try await grounder.ground(
                        step: step,
                        observation: groundingObservation
                    )
                }
                let currentObservation = try await observer.observe(for: step)
                try Task.checkCancellation()
                try MacOSDraftPostFlowDriver.focusGroundedEmptyComposer(
                    action: action,
                    groundedAgainst: groundingObservation,
                    currentObservation: currentObservation,
                    destination: destination,
                    documentIdentity: documentIdentity,
                    actuator: actuator
                )
                await vault.removeAll()
            } catch {
                await vault.removeAll()
                throw error
            }
        }
    }

    init(attempt: @escaping Attempt) {
        self.attempt = attempt
    }

    /// The local server may restart onto another port or rotate its bearer
    /// while the sheet remains open. Validate immediately around inference so
    /// a screenshot is never sent to, or accepted from, a superseded session.
    static func withValidatedSession<Result: Sendable>(
        _ validator: DraftPostVisualRuntime.SessionValidator,
        operation: @Sendable () async throws -> Result
    ) async throws -> Result {
        guard await validator() else {
            throw DraftPostFlowFailure.dependencyFailure
        }
        let result = try await operation()
        guard await validator() else {
            throw DraftPostFlowFailure.dependencyFailure
        }
        return result
    }

    func focusComposer(
        in destination: ComputerUseWindowOption,
        documentIdentity: String
    ) async throws {
        for attemptNumber in 1 ... Self.maximumAttempts {
            do {
                try Task.checkCancellation()
                try await attempt(destination, documentIdentity)
                return
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                guard attemptNumber < Self.maximumAttempts else {
                    throw DraftPostFlowFailure.composerMissing
                }
            }
        }
        throw DraftPostFlowFailure.composerMissing
    }

    private static var composerStep: LocalWorkflowStep {
        LocalWorkflowStep(
            id: Self.stepID,
            title: "Find the post composer",
            instruction: """
            Locate the empty main post composer in this browser page. Do not choose the \
            address bar, search, reply fields, buttons, Post, Publish, or Send.
            """,
            successCriteria: "The empty main post composer has keyboard focus.",
            risk: .localChange,
            isIdempotent: true,
            maxGroundingAttempts: Self.maximumAttempts
        )
    }
}
