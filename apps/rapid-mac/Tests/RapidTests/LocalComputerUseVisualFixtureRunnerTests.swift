import AppKit
import ApplicationServices
import Foundation
import ScreenCaptureKit
import Testing
@testable import Rapid

@Suite("Local Computer Use visual fixture runner", .serialized)
struct LocalComputerUseVisualFixtureRunnerTests {
    private static let approvalPhrase = "APPROVE_ONE_HARMLESS_CALCULATOR_CLICK"

    @Test("Visual grounding runs through approval, action, verification, and ledger")
    func deterministicComposition() async throws {
        let state = FixtureState()
        let vault = ComputerUseObservationVault()
        let observer = SyntheticFixtureObserver(state: state, vault: vault)
        let transport = OneClickTransport(x: 500, y: 500)
        let grounder = LocalComputerUseVisualGrounder(
            configuration: try LocalComputerUseVisualGrounder.Configuration(
                baseURL: #require(URL(string: "http://127.0.0.1:8377/v1")),
                model: "fixture-model"
            ),
            vault: vault,
            transport: transport
        )
        let approver = FixtureApprover(remainingApprovals: 1)
        let actuator = SyntheticFixtureActuator(state: state)
        let verifier = FixtureVerifier(state: state)
        let ledger = FixtureLedger()
        let executor = LocalWorkflowExecutor(
            observer: observer,
            grounder: grounder,
            actuator: actuator,
            verifier: verifier,
            fallbackResolver: NoFixtureFallback(),
            approver: approver,
            ledger: ledger
        )

        let run = await executor.execute(Self.workflow())

        #expect(run.status == .completed)
        #expect(await transport.requestCount == 1)
        #expect(await approver.requests.count == 1)
        #expect(await actuator.actionCount == 1)
        #expect(await verifier.verificationCount == 1)
        #expect(await state.isPressed)
        #expect(await ledger.events.map(\.kind) == [
            .runStarted,
            .actionGrounded,
            .approvalRequested,
            .approvalGranted,
            .actionPerformed,
            .verificationPassed,
            .runCompleted,
        ])
    }

    @Test("Missing explicit approval stops before the fixture click")
    func approvalFailsClosed() async throws {
        let state = FixtureState()
        let vault = ComputerUseObservationVault()
        let actuator = SyntheticFixtureActuator(state: state)
        let executor = LocalWorkflowExecutor(
            observer: SyntheticFixtureObserver(state: state, vault: vault),
            grounder: LocalComputerUseVisualGrounder(
                configuration: try LocalComputerUseVisualGrounder.Configuration(
                    baseURL: #require(URL(string: "http://127.0.0.1:8377/v1")),
                    model: "fixture-model"
                ),
                vault: vault,
                transport: OneClickTransport(x: 500, y: 500)
            ),
            actuator: actuator,
            verifier: FixtureVerifier(state: state),
            fallbackResolver: NoFixtureFallback(),
            approver: FixtureApprover(remainingApprovals: 0),
            ledger: FixtureLedger()
        )

        let run = await executor.execute(Self.workflow())

        #expect(run.status == .paused(
            stepID: "press-harmless-fixture",
            reason: .approvalUnavailable,
            actionMayHaveOccurred: false
        ))
        #expect(await actuator.actionCount == 0)
        #expect(!(await state.isPressed))
    }

    /// Operator-only end-to-end dogfood. This clears Calculator, sends only
    /// its selected window's pixels to an already
    /// running loopback model, requires a one-use approval for every run,
    /// performs the model-grounded click through the production macOS adapter,
    /// and independently verifies the fixture state. The approval is consumed
    /// once by this test process. It never types, submits,
    /// purchases, deletes, or touches another app.
    ///
    /// Run from a trusted GUI session after granting Screen Recording and
    /// Accessibility to the test process:
    ///
    /// RAPID_LIVE_CUA_VISUAL_FIXTURE=1 \
    /// RAPID_LIVE_CUA_APPROVAL=APPROVE_ONE_HARMLESS_CALCULATOR_CLICK \
    /// RAPID_LIVE_CUA_BASE_URL=http://127.0.0.1:8377/v1 \
    /// RAPID_LIVE_CUA_MODEL=EvoCUA_8B_4bit \
    /// swift test --no-parallel --filter LocalComputerUseVisualFixtureRunnerTests/liveFixture
    @MainActor
    @Test(.enabled(
        if: ProcessInfo.processInfo.environment["RAPID_LIVE_CUA_VISUAL_FIXTURE"] == "1"
    ))
    func liveFixture() async throws {
        let configuration = try Self.liveConfiguration()
        let calculator = try await CalculatorFixture.prepare()
        defer { try? calculator.clear() }

        let vault = ComputerUseObservationVault()
        let observer = MacOSComputerUseObserver(
            selections: ["press-harmless-fixture": calculator.selection],
            vault: vault
        )
        let timedGrounder = TimedFixtureGrounder(base: LocalComputerUseVisualGrounder(
            configuration: try LocalComputerUseVisualGrounder.Configuration(
                baseURL: configuration.baseURL,
                model: configuration.model,
                bearerToken: configuration.bearerToken,
                deadline: .seconds(45)
            ),
            vault: vault
        ))
        let approver = FixtureApprover(remainingApprovals: 1)
        let actuator = DiagnosingFixtureActuator(base: MacOSComputerUseActuator())
        let verifier = CalculatorFixtureVerifier(fixture: calculator)
        let ledger = FixtureLedger()
        let executor = LocalWorkflowExecutor(
            observer: observer,
            grounder: timedGrounder,
            actuator: actuator,
            verifier: verifier,
            fallbackResolver: NoFixtureFallback(),
            approver: approver,
            ledger: ledger
        )
        let clock = ContinuousClock()
        let started = clock.now
        let run = await executor.execute(Self.workflow())
        let endToEndMilliseconds = [Self.milliseconds(started.duration(to: clock.now))]
        if run.status != .completed {
            let events = await ledger.events.map(\.kind.rawValue).joined(separator: ",")
            let action = await timedGrounder.lastActionDescription ?? "none"
            let groundingError = await timedGrounder.lastError ?? "none"
            let actuatorError = await actuator.lastError ?? "none"
            let display = (try? calculator.displayValue()) ?? "unreadable"
            print(
                "RAPID_CUA_VISUAL_FIXTURE_FAILURE status=\(run.status) "
                    + "events=\(events) action=\(action) "
                    + "grounding_error=\(groundingError) "
                    + "actuator_error=\(actuatorError) display=\(display)"
            )
        }
        try #require(run.status == .completed)
        #expect(try calculator.displayValue() == "7")

        let groundingMilliseconds = await timedGrounder.milliseconds
        #expect(groundingMilliseconds.count == 1)
        #expect(endToEndMilliseconds.count == 1)
        #expect(await approver.requests.count == 1)
        #expect(await verifier.verificationCount == 1)
        #expect(await ledger.events.filter { $0.kind == .actionPerformed }.count == 1)
        print("RAPID_CUA_VISUAL_FIXTURE_RESULT " + Self.metricsJSON(
            model: configuration.model,
            groundingMilliseconds: groundingMilliseconds,
            endToEndMilliseconds: endToEndMilliseconds
        ))
    }

    private static func workflow() -> LocalWorkflow {
        LocalWorkflow(
            title: "Harmless Calculator visual grounding fixture",
            steps: [
                LocalWorkflowStep(
                    id: "press-harmless-fixture",
                    title: "Press the harmless fixture button",
                    instruction: "Click the Calculator digit button labeled 7.",
                    successCriteria: "The Calculator display reads exactly 7.",
                    // Deliberately conservative so the fixture exercises the
                    // same fresh approval boundary as a protected action.
                    risk: .externalCommunication,
                    isIdempotent: false,
                    maxGroundingAttempts: 1
                ),
            ]
        )
    }

    private struct LiveConfiguration {
        let baseURL: URL
        let model: String
        let bearerToken: String?
    }

    private static func liveConfiguration() throws -> LiveConfiguration {
        let environment = ProcessInfo.processInfo.environment
        guard environment["RAPID_LIVE_CUA_APPROVAL"] == approvalPhrase,
              let baseURLValue = environment["RAPID_LIVE_CUA_BASE_URL"],
              let baseURL = URL(string: baseURLValue),
              let model = environment["RAPID_LIVE_CUA_MODEL"],
              !model.isEmpty
        else {
            throw LiveFixtureError.invalidOperatorConfiguration
        }
        return LiveConfiguration(
            baseURL: baseURL,
            model: model,
            bearerToken: environment["RAPID_LIVE_CUA_BEARER_TOKEN"]
        )
    }

    private static func milliseconds(_ duration: Duration) -> Double {
        let components = duration.components
        return Double(components.seconds) * 1_000
            + Double(components.attoseconds) / 1_000_000_000_000_000
    }

    private static func metricsJSON(
        model: String,
        groundingMilliseconds: [Double],
        endToEndMilliseconds: [Double]
    ) -> String {
        func summary(_ values: [Double]) -> [String: Double] {
            let sorted = values.sorted()
            return [
                "min_ms": sorted.first ?? 0,
                "median_ms": sorted[sorted.count / 2],
                "max_ms": sorted.last ?? 0,
            ]
        }
        let object: [String: Any] = [
            "runs": endToEndMilliseconds.count,
            "successes": endToEndMilliseconds.count,
            "model": model,
            "grounding": summary(groundingMilliseconds),
            "end_to_end": summary(endToEndMilliseconds),
        ]
        guard let data = try? JSONSerialization.data(
            withJSONObject: object,
            options: [.sortedKeys]
        ) else { return "{}" }
        return String(decoding: data, as: UTF8.self)
    }
}

private enum LiveFixtureError: Error {
    case invalidOperatorConfiguration
    case calculatorUnavailable
    case calculatorAccessibilityUnavailable
    case calculatorStateInvalid
    case coordinateOutsideFixtureTarget
}

private actor FixtureState {
    private(set) var isPressed = false

    func press() { isPressed = true }
}

private struct SyntheticFixtureObserver: LocalWorkflowObserving {
    let state: FixtureState
    let vault: ComputerUseObservationVault

    func observe(for _: LocalWorkflowStep) async throws -> WorkflowObservation {
        let pressed = await state.isPressed
        let observation = WorkflowObservation(
            target: WorkflowInteractionTarget(
                bundleIdentifier: "com.rapidmlx.fixture",
                processIdentifier: 1,
                processLaunchDate: .distantPast,
                windowIdentifier: "fixture",
                windowFrame: WorkflowWindowFrame(x: 0, y: 0, width: 640, height: 480)
            ),
            contentRevision: pressed ? "pressed" : "ready"
        )
        await vault.store(
            ComputerUseObservationArtifact(
                pngData: Data([0x89, 0x50, 0x4E, 0x47]),
                pixelWidth: 640,
                pixelHeight: 480
            ),
            for: observation.id
        )
        return observation
    }
}

private actor SyntheticFixtureActuator: LocalWorkflowActuating {
    let state: FixtureState
    private(set) var actionCount = 0

    init(state: FixtureState) {
        self.state = state
    }

    func perform(
        _ action: GroundedWorkflowAction,
        groundedAgainst _: WorkflowObservation,
        currentObservation _: WorkflowObservation
    ) async throws {
        guard case .click(let x, let y) = action.payload,
              (0.35 ... 0.65).contains(x),
              (0.35 ... 0.65).contains(y)
        else { throw LiveFixtureError.coordinateOutsideFixtureTarget }
        actionCount += 1
        await state.press()
    }
}

private actor FixtureVerifier: LocalWorkflowVerifying {
    let state: FixtureState
    private(set) var verificationCount = 0

    init(state: FixtureState) {
        self.state = state
    }

    func verify(
        step _: LocalWorkflowStep,
        before _: WorkflowObservation,
        after _: WorkflowObservation
    ) async throws -> WorkflowVerification {
        verificationCount += 1
        return await state.isPressed ? .satisfied : .notSatisfied(code: .targetUnchanged)
    }
}

private actor FixtureApprover: LocalWorkflowApproving {
    private var remainingApprovals: Int
    private(set) var requests: [WorkflowApprovalRequest] = []

    init(remainingApprovals: Int) {
        self.remainingApprovals = remainingApprovals
    }

    func requestApproval(
        _ request: WorkflowApprovalRequest,
        timeoutNanoseconds _: UInt64
    ) async -> WorkflowApprovalDecision {
        requests.append(request)
        guard remainingApprovals > 0 else { return .unavailable }
        remainingApprovals -= 1
        return .approved
    }
}

private struct NoFixtureFallback: LocalWorkflowFallbackResolving {
    func fallbackAction(
        identifier _: String,
        step _: LocalWorkflowStep,
        observation _: WorkflowObservation
    ) async throws -> GroundedWorkflowAction? {
        nil
    }
}

private actor FixtureLedger: LocalWorkflowLedgerWriting {
    private(set) var events: [WorkflowLedgerEvent] = []

    func append(_ event: WorkflowLedgerEvent) {
        events.append(event)
    }
}

private actor OneClickTransport: LocalComputerUseGroundingTransport {
    private let x: Int
    private let y: Int
    private(set) var requestCount = 0

    init(x: Int, y: Int) {
        self.x = x
        self.y = y
    }

    func send(
        _: URLRequest,
        maximumResponseBytes _: Int
    ) async throws -> LocalComputerUseGroundingHTTPResponse {
        requestCount += 1
        let arguments = "{\"action\":\"left_click\",\"coordinate\":[\(x),\(y)]}"
        let object: [String: Any] = [
            "choices": [[
                "message": [
                    "tool_calls": [[
                        "type": "function",
                        "function": [
                            "name": "computer_use",
                            "arguments": arguments,
                        ],
                    ]],
                ],
            ]],
        ]
        return LocalComputerUseGroundingHTTPResponse(
            statusCode: 200,
            contentType: "application/json",
            body: try JSONSerialization.data(withJSONObject: object)
        )
    }
}

@MainActor
private final class CalculatorFixture {
    static let bundleIdentifier = "com.apple.calculator"

    let application: NSRunningApplication
    let selection: ComputerUseWindowSelection

    private init(
        application: NSRunningApplication,
        selection: ComputerUseWindowSelection
    ) {
        self.application = application
        self.selection = selection
    }

    static func prepare() async throws -> CalculatorFixture {
        let application = try await runningApplication()
        guard application.activate(options: [.activateAllWindows]) else {
            throw LiveFixtureError.calculatorUnavailable
        }
        for _ in 0 ..< 30 {
            if NSWorkspace.shared.frontmostApplication?.processIdentifier
                    == application.processIdentifier,
               let selection = selection(for: application)
            {
                let fixture = CalculatorFixture(
                    application: application,
                    selection: selection
                )
                try fixture.clear()
                guard try fixture.displayValue() == "0" else {
                    throw LiveFixtureError.calculatorStateInvalid
                }
                return fixture
            }
            try await Task.sleep(for: .milliseconds(100))
        }
        throw LiveFixtureError.calculatorUnavailable
    }

    func clear() throws {
        let root = try applicationElement()
        guard let button = Self.find(root, identifiers: ["AllClear", "Clear"]),
              AXUIElementPerformAction(button, kAXPressAction as CFString) == .success
        else {
            throw LiveFixtureError.calculatorAccessibilityUnavailable
        }
    }

    func displayValue() throws -> String {
        let root = try applicationElement()
        guard let display = Self.find(root, identifiers: ["StandardInputView"]),
              let value = Self.firstStringValue(in: display)
        else {
            throw LiveFixtureError.calculatorAccessibilityUnavailable
        }
        let allowed = CharacterSet(charactersIn: "-0123456789.")
        return String(value.unicodeScalars.filter(allowed.contains))
    }

    private func applicationElement() throws -> AXUIElement {
        let root = AXUIElementCreateApplication(application.processIdentifier)
        guard AXUIElementSetMessagingTimeout(root, 2) == .success else {
            throw LiveFixtureError.calculatorAccessibilityUnavailable
        }
        return root
    }

    private static func runningApplication() async throws -> NSRunningApplication {
        if let running = NSRunningApplication.runningApplications(
            withBundleIdentifier: bundleIdentifier
        ).first {
            return running
        }
        let url = URL(fileURLWithPath: "/System/Applications/Calculator.app")
        return try await withCheckedThrowingContinuation { continuation in
            NSWorkspace.shared.openApplication(
                at: url,
                configuration: NSWorkspace.OpenConfiguration()
            ) { application, error in
                if let application {
                    continuation.resume(returning: application)
                } else {
                    continuation.resume(
                        throwing: error ?? LiveFixtureError.calculatorUnavailable
                    )
                }
            }
        }
    }

    private static func selection(
        for application: NSRunningApplication
    ) -> ComputerUseWindowSelection? {
        guard let launchDate = application.launchDate,
              let records = CGWindowListCopyWindowInfo(
                [.optionOnScreenOnly, .excludeDesktopElements],
                kCGNullWindowID
              ) as? [[CFString: Any]],
              let record = records.first(where: {
                ($0[kCGWindowOwnerPID] as? NSNumber)?.int32Value
                    == application.processIdentifier
                    && ($0[kCGWindowLayer] as? NSNumber)?.intValue == 0
                    && (($0[kCGWindowBounds] as? [String: NSNumber])?["Width"]?.doubleValue ?? 0) > 0
              }),
              let number = record[kCGWindowNumber] as? NSNumber
        else { return nil }
        return ComputerUseWindowSelection(
            bundleIdentifier: bundleIdentifier,
            processIdentifier: application.processIdentifier,
            processLaunchDate: launchDate,
            windowID: number.uint32Value
        )
    }

    private static func find(
        _ root: AXUIElement,
        identifiers: [String],
        depth: Int = 0
    ) -> AXUIElement? {
        guard depth < 32 else { return nil }
        if let identifier = stringAttribute(
            kAXIdentifierAttribute as CFString,
            from: root
        ), identifiers.contains(identifier) {
            return root
        }
        for child in children(of: root).prefix(128) {
            if let match = find(child, identifiers: identifiers, depth: depth + 1) {
                return match
            }
        }
        return nil
    }

    private static func firstStringValue(
        in root: AXUIElement,
        depth: Int = 0
    ) -> String? {
        guard depth < 16 else { return nil }
        if let value = stringAttribute(kAXValueAttribute as CFString, from: root),
           !value.isEmpty {
            return value
        }
        for child in children(of: root).prefix(64) {
            if let value = firstStringValue(in: child, depth: depth + 1) {
                return value
            }
        }
        return nil
    }

    private static func children(of element: AXUIElement) -> [AXUIElement] {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(
            element,
            kAXChildrenAttribute as CFString,
            &value
        ) == .success else { return [] }
        return value as? [AXUIElement] ?? []
    }

    private static func stringAttribute(
        _ attribute: CFString,
        from element: AXUIElement
    ) -> String? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, attribute, &value) == .success
        else { return nil }
        return value as? String
    }
}

private actor CalculatorFixtureVerifier: LocalWorkflowVerifying {
    private let fixture: CalculatorFixture
    private(set) var verificationCount = 0

    init(fixture: CalculatorFixture) {
        self.fixture = fixture
    }

    func verify(
        step _: LocalWorkflowStep,
        before: WorkflowObservation,
        after: WorkflowObservation
    ) async throws -> WorkflowVerification {
        verificationCount += 1
        guard before.target == after.target else {
            return .unsafe(code: .windowChanged)
        }
        let value = try await MainActor.run { try fixture.displayValue() }
        return value == "7" ? .satisfied : .notSatisfied(code: .targetUnchanged)
    }
}

private actor TimedFixtureGrounder: LocalWorkflowGrounding {
    private let base: any LocalWorkflowGrounding
    private(set) var milliseconds: [Double] = []
    private(set) var lastActionDescription: String?
    private(set) var lastError: String?

    init(base: any LocalWorkflowGrounding) {
        self.base = base
    }

    func ground(
        step: LocalWorkflowStep,
        observation: WorkflowObservation
    ) async throws -> GroundedWorkflowAction {
        let clock = ContinuousClock()
        let started = clock.now
        do {
            let result = try await base.ground(step: step, observation: observation)
            milliseconds.append(Self.ms(started.duration(to: clock.now)))
            if case .click(let x, let y) = result.payload {
                lastActionDescription = String(format: "click(%.4f,%.4f)", x, y)
            } else {
                lastActionDescription = "non-click"
            }
            return result
        } catch {
            milliseconds.append(Self.ms(started.duration(to: clock.now)))
            lastError = String(reflecting: error)
            throw error
        }
    }

    private static func ms(_ duration: Duration) -> Double {
        let components = duration.components
        return Double(components.seconds) * 1_000
            + Double(components.attoseconds) / 1_000_000_000_000_000
    }
}

private actor DiagnosingFixtureActuator: LocalWorkflowActuating {
    private let base: any LocalWorkflowActuating
    private(set) var lastError: String?

    init(base: any LocalWorkflowActuating) {
        self.base = base
    }

    func perform(
        _ action: GroundedWorkflowAction,
        groundedAgainst groundingObservation: WorkflowObservation,
        currentObservation: WorkflowObservation
    ) async throws {
        do {
            try await base.perform(
                action,
                groundedAgainst: groundingObservation,
                currentObservation: currentObservation
            )
        } catch {
            lastError = String(reflecting: error)
            throw error
        }
    }
}
