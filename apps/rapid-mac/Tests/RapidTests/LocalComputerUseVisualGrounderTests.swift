import Foundation
import Testing
@testable import Rapid

@Suite("Local Computer Use visual grounder")
struct LocalComputerUseVisualGrounderTests {
    /// Operator-only model compatibility check. It performs no UI action.
    /// Run against an already-started local server with:
    ///
    /// RAPID_LIVE_CUA_GROUNDER=1
    /// RAPID_LIVE_CUA_SCREENSHOT=/path/to/lunch-fixture.png
    /// RAPID_LIVE_CUA_BASE_URL=http://127.0.0.1:8377/v1
    /// RAPID_LIVE_CUA_MODEL=EvoCUA_8B_4bit
    /// swift test --no-parallel --filter LocalComputerUseVisualGrounderTests/liveModel
    @Test(.enabled(
        if: ProcessInfo.processInfo.environment["RAPID_LIVE_CUA_GROUNDER"] == "1"
    ))
    func liveModel() async throws {
        let environment = ProcessInfo.processInfo.environment
        let screenshotPath = try #require(environment["RAPID_LIVE_CUA_SCREENSHOT"])
        let baseURLString = try #require(environment["RAPID_LIVE_CUA_BASE_URL"])
        let baseURL = try #require(URL(string: baseURLString))
        let model = try #require(environment["RAPID_LIVE_CUA_MODEL"])
        let vault = ComputerUseObservationVault()
        let observation = WorkflowObservation(
            target: WorkflowInteractionTarget(
                bundleIdentifier: "com.rapidmlx.fixture",
                processIdentifier: 1,
                processLaunchDate: .distantPast,
                windowIdentifier: "fixture",
                windowFrame: WorkflowWindowFrame(
                    x: 0,
                    y: 0,
                    width: 1280,
                    height: 720
                )
            ),
            contentRevision: "live-model-fixture"
        )
        await vault.store(
            ComputerUseObservationArtifact(
                pngData: try Data(contentsOf: URL(fileURLWithPath: screenshotPath)),
                pixelWidth: 1280,
                pixelHeight: 720
            ),
            for: observation.id
        )
        let grounder = LocalComputerUseVisualGrounder(
            configuration: try LocalComputerUseVisualGrounder.Configuration(
                baseURL: baseURL,
                model: model,
                deadline: .seconds(45)
            ),
            vault: vault
        )
        let probes: [(String, ClosedRange<Double>, ClosedRange<Double>)] = [
            ("Click the child button labeled Eric.", 0.68 ... 0.80, 0.35 ... 0.45),
            ("For Monday, click the meal button labeled Chicken Bowl.", 0.65 ... 0.80, 0.47 ... 0.57),
            ("For Tuesday, click the meal button labeled Pasta Primavera.", 0.54 ... 0.70, 0.60 ... 0.70),
            ("Click Review Order. Do not click Place Order.", 0.55 ... 0.76, 0.68 ... 0.80),
        ]
        for (instruction, expectedX, expectedY) in probes {
            let step = LocalWorkflowStep(
                id: "live-probe",
                title: "Live probe",
                instruction: instruction,
                successCriteria: "The intended fixture target was located.",
                risk: .localChange,
                isIdempotent: true,
                maxGroundingAttempts: 1
            )
            let action = try await grounder.ground(
                step: step,
                observation: observation
            )
            guard case .click(let x, let y) = action.payload else {
                Issue.record("Expected click for \(instruction)")
                continue
            }
            #expect(expectedX.contains(x))
            #expect(expectedY.contains(y))
        }
    }

    @Test("One exact tool call becomes one observation-bound click")
    func validClick() async throws {
        let fixture = try await Fixture(responseBody: Self.response(x: 250, y: 750))

        let action = try await fixture.grounder.ground(
            step: fixture.step,
            observation: fixture.observation
        )

        #expect(action.observationID == fixture.observation.id)
        #expect(action.source == .visualGrounding)
        #expect(action.risk == .externalCommunication)
        #expect(action.safeSummary == "Click the target for Review update")
        guard case .click(let x, let y) = action.payload else {
            Issue.record("Expected one click")
            return
        }
        #expect(abs(x - (250.0 / 999.0)) < 0.000_001)
        #expect(abs(y - (750.0 / 999.0)) < 0.000_001)

        let request = try #require(await fixture.transport.lastRequest)
        #expect(request.url?.absoluteString == "http://127.0.0.1:8377/v1/chat/completions")
        #expect(request.httpMethod == "POST")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer local-secret")
        let body = try #require(request.httpBody)
        let object = try #require(
            JSONSerialization.jsonObject(with: body) as? [String: Any]
        )
        #expect(object["model"] as? String == "EvoCUA_8B_4bit")
        #expect(object["temperature"] as? Int == 0)
        let choice = try #require(object["tool_choice"] as? [String: Any])
        let function = try #require(choice["function"] as? [String: Any])
        #expect(function["name"] as? String == "computer_use")
        let messages = try #require(object["messages"] as? [[String: Any]])
        let userContent = try #require(messages.last?["content"] as? [[String: Any]])
        let image = try #require(userContent.first?["image_url"] as? [String: Any])
        #expect((image["url"] as? String)?.hasPrefix("data:image/png;base64,") == true)
        #expect(userContent.last?["text"] as? String == fixture.step.instruction)
    }

    @Test("Only literal loopback v1 endpoints are accepted")
    func endpointPolicy() {
        let rejected = [
            "https://127.0.0.1:8377/v1",
            "http://localhost:8377/v1",
            "http://127.0.0.2:8377/v1",
            "http://example.com:8377/v1",
            "http://user@127.0.0.1:8377/v1",
            "http://127.0.0.1/v1",
            "http://127.0.0.1:8377/other",
            "http://127.0.0.1:8377/v1?next=evil",
        ]
        for value in rejected {
            #expect(throws: LocalComputerUseVisualGrounderError.invalidConfiguration) {
                _ = try LocalComputerUseVisualGrounder.Configuration(
                    baseURL: #require(URL(string: value)),
                    model: "model"
                )
            }
        }
        #expect(throws: Never.self) {
            let configuration = try LocalComputerUseVisualGrounder.Configuration(
                baseURL: #require(URL(string: "http://[::1]:8377/v1")),
                model: "model"
            )
            #expect(
                configuration.completionURL.absoluteString
                    == "http://[::1]:8377/v1/chat/completions"
            )
        }

        #expect(throws: LocalComputerUseVisualGrounderError.invalidConfiguration) {
            _ = try LocalComputerUseVisualGrounder.Configuration(
                baseURL: #require(URL(string: "http://127.0.0.1:8377/v1")),
                model: "model",
                deadline: .seconds(61)
            )
        }
        #expect(throws: LocalComputerUseVisualGrounderError.invalidConfiguration) {
            _ = try LocalComputerUseVisualGrounder.Configuration(
                baseURL: #require(URL(string: "http://127.0.0.1:8377/v1")),
                model: "model",
                bearerToken: "unsafe\rheader"
            )
        }
        #expect(throws: Never.self) {
            let configuration = try LocalComputerUseVisualGrounder.Configuration(
                baseURL: #require(URL(string: "http://127.0.0.1:8377/v1/")),
                model: "model"
            )
            #expect(
                configuration.completionURL.absoluteString
                    == "http://127.0.0.1:8377/v1/chat/completions"
            )
        }
    }

    @Test("Missing or oversized ephemeral screenshots fail before transport")
    func artifactPolicy() async throws {
        let fixture = try await Fixture(
            responseBody: Self.response(x: 300, y: 300),
            storeArtifact: false
        )
        await #expect(
            throws: LocalComputerUseVisualGrounderError.missingObservationArtifact
        ) {
            _ = try await fixture.grounder.ground(
                step: fixture.step,
                observation: fixture.observation
            )
        }
        #expect(await fixture.transport.requestCount == 0)

        let oversized = try await Fixture(
            responseBody: Self.response(x: 300, y: 300),
            pngData: Data(repeating: 0x61, count: 9),
            maximumScreenshotBytes: 8
        )
        await #expect(throws: LocalComputerUseVisualGrounderError.screenshotTooLarge) {
            _ = try await oversized.grounder.ground(
                step: oversized.step,
                observation: oversized.observation
            )
        }
        #expect(await oversized.transport.requestCount == 0)
    }

    @Test("Request and response allocations remain bounded")
    func allocationPolicy() async throws {
        let request = try await Fixture(
            responseBody: Self.response(x: 300, y: 300),
            maximumScreenshotBytes: 8,
            maximumRequestBytes: 9
        )
        await #expect(throws: LocalComputerUseVisualGrounderError.requestTooLarge) {
            _ = try await request.grounder.ground(
                step: request.step,
                observation: request.observation
            )
        }
        #expect(await request.transport.requestCount == 0)

        let response = try await Fixture(
            responseBody: Self.response(x: 300, y: 300),
            maximumResponseBytes: 8
        )
        await #expect(throws: LocalComputerUseVisualGrounderError.responseTooLarge) {
            _ = try await response.grounder.ground(
                step: response.step,
                observation: response.observation
            )
        }
    }

    @Test("Malformed, multiple, and unsupported model actions fail closed", arguments: [
        Data("{}".utf8),
        Self.response(x: 0, y: 500),
        Self.response(x: 999, y: 500),
        Self.response(x: 500, y: 500, action: "type"),
        Self.response(x: 500, y: 500, extraArgument: true),
        Self.response(x: 500, y: 500, duplicateCall: true),
        Self.responseWithRawCoordinate("[true,500]"),
        Self.responseWithRawCoordinate("[500.0,500]"),
    ])
    func rejectsUnsafeResponses(body: Data) async throws {
        let fixture = try await Fixture(responseBody: body)
        await #expect(throws: LocalComputerUseVisualGrounderError.self) {
            _ = try await fixture.grounder.ground(
                step: fixture.step,
                observation: fixture.observation
            )
        }
    }

    @Test("HTTP status and content type are authoritative")
    func HTTPPolicy() async throws {
        let status = try await Fixture(
            responseBody: Self.response(x: 500, y: 500),
            statusCode: 503
        )
        await #expect(throws: LocalComputerUseVisualGrounderError.httpStatus(503)) {
            _ = try await status.grounder.ground(
                step: status.step,
                observation: status.observation
            )
        }

        let contentType = try await Fixture(
            responseBody: Self.response(x: 500, y: 500),
            contentType: "text/html"
        )
        await #expect(throws: LocalComputerUseVisualGrounderError.invalidHTTPResponse) {
            _ = try await contentType.grounder.ground(
                step: contentType.step,
                observation: contentType.observation
            )
        }
    }

    @Test("The hard deadline cancels a slow local inference")
    func deadline() async throws {
        let fixture = try await Fixture(
            responseBody: Self.response(x: 500, y: 500),
            delay: .seconds(5),
            deadline: .milliseconds(30)
        )
        await #expect(throws: LocalComputerUseVisualGrounderError.deadlineExceeded) {
            _ = try await fixture.grounder.ground(
                step: fixture.step,
                observation: fixture.observation
            )
        }
        #expect(await fixture.transport.cancelledSendCount == 1)
    }

    @Test("Caller cancellation propagates and cancels local inference")
    func cancellation() async throws {
        let fixture = try await Fixture(
            responseBody: Self.response(x: 500, y: 500),
            delay: .seconds(5)
        )
        let task = Task {
            try await fixture.grounder.ground(
                step: fixture.step,
                observation: fixture.observation
            )
        }
        while await fixture.transport.requestCount == 0 {
            await Task.yield()
        }
        task.cancel()
        await #expect(throws: CancellationError.self) {
            _ = try await task.value
        }
        #expect(await fixture.transport.cancelledSendCount == 1)
    }

    private static func response(
        x: Int,
        y: Int,
        action: String = "left_click",
        extraArgument: Bool = false,
        duplicateCall: Bool = false
    ) -> Data {
        var arguments: [String: Any] = [
            "action": action,
            "coordinate": [x, y],
        ]
        if extraArgument { arguments["text"] = "unsafe" }
        let argumentData = try! JSONSerialization.data(withJSONObject: arguments)
        let argumentString = String(decoding: argumentData, as: UTF8.self)
        let call: [String: Any] = [
            "id": "call-1",
            "type": "function",
            "function": [
                "name": "computer_use",
                "arguments": argumentString,
            ],
        ]
        let calls = duplicateCall ? [call, call] : [call]
        return try! JSONSerialization.data(withJSONObject: [
            "choices": [["message": ["tool_calls": calls]]],
        ])
    }

    private static func responseWithRawCoordinate(_ coordinate: String) -> Data {
        let arguments = """
        {\"action\":\"left_click\",\"coordinate\":\(coordinate)}
        """
        return try! JSONSerialization.data(withJSONObject: [
            "choices": [[
                "message": [
                    "tool_calls": [[
                        "function": [
                            "name": "computer_use",
                            "arguments": arguments,
                        ],
                    ]],
                ],
            ]],
        ])
    }

    private struct Fixture {
        let grounder: LocalComputerUseVisualGrounder
        let transport: CapturingGroundingTransport
        let observation: WorkflowObservation
        let step: LocalWorkflowStep

        init(
            responseBody: Data,
            statusCode: Int = 200,
            contentType: String? = "application/json; charset=utf-8",
            delay: Duration = .zero,
            deadline: Duration = .seconds(2),
            pngData: Data = Data([0x89, 0x50, 0x4E, 0x47]),
            maximumScreenshotBytes: Int = 8 * 1024 * 1024,
            maximumRequestBytes: Int = 12 * 1024 * 1024,
            maximumResponseBytes: Int = 512 * 1024,
            storeArtifact: Bool = true
        ) async throws {
            let vault = ComputerUseObservationVault()
            let observation = WorkflowObservation(
                target: WorkflowInteractionTarget(
                    bundleIdentifier: "com.example.fixture",
                    processIdentifier: 42,
                    processLaunchDate: Date(timeIntervalSinceReferenceDate: 10),
                    windowIdentifier: "9",
                    windowFrame: WorkflowWindowFrame(
                        x: 10,
                        y: 20,
                        width: 800,
                        height: 600
                    )
                ),
                contentRevision: "revision"
            )
            if storeArtifact {
                await vault.store(
                    ComputerUseObservationArtifact(
                        pngData: pngData,
                        pixelWidth: 800,
                        pixelHeight: 600
                    ),
                    for: observation.id
                )
            }
            let transport = CapturingGroundingTransport(
                response: LocalComputerUseGroundingHTTPResponse(
                    statusCode: statusCode,
                    contentType: contentType,
                    body: responseBody
                ),
                delay: delay
            )
            self.observation = observation
            self.step = LocalWorkflowStep(
                id: "review",
                title: "Review update",
                instruction: "Click the button labeled Review.",
                successCriteria: "The review screen is visible.",
                risk: .externalCommunication,
                isIdempotent: true,
                maxGroundingAttempts: 2
            )
            self.transport = transport
            self.grounder = LocalComputerUseVisualGrounder(
                configuration: try LocalComputerUseVisualGrounder.Configuration(
                    baseURL: #require(URL(string: "http://127.0.0.1:8377/v1")),
                    model: "EvoCUA_8B_4bit",
                    bearerToken: "local-secret",
                    deadline: deadline,
                    maximumScreenshotBytes: maximumScreenshotBytes,
                    maximumRequestBytes: maximumRequestBytes,
                    maximumResponseBytes: maximumResponseBytes
                ),
                vault: vault,
                transport: transport
            )
        }
    }
}

private actor CapturingGroundingTransport: LocalComputerUseGroundingTransport {
    private let response: LocalComputerUseGroundingHTTPResponse
    private let delay: Duration
    private(set) var requests: [URLRequest] = []
    private(set) var cancelledSendCount = 0

    init(response: LocalComputerUseGroundingHTTPResponse, delay: Duration) {
        self.response = response
        self.delay = delay
    }

    var requestCount: Int { requests.count }
    var lastRequest: URLRequest? { requests.last }

    func send(
        _ request: URLRequest,
        maximumResponseBytes: Int
    ) async throws -> LocalComputerUseGroundingHTTPResponse {
        requests.append(request)
        guard response.body.count <= maximumResponseBytes else {
            throw LocalComputerUseVisualGrounderError.responseTooLarge
        }
        do {
            if delay > .zero { try await Task.sleep(for: delay) }
            return response
        } catch is CancellationError {
            cancelledSendCount += 1
            throw CancellationError()
        }
    }
}
