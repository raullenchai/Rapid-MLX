import Foundation

enum LocalComputerUseVisualGrounderError: Error, Equatable {
    case invalidConfiguration
    case invalidObservation
    case missingObservationArtifact
    case screenshotTooLarge
    case requestTooLarge
    case responseTooLarge
    case deadlineExceeded
    case transportFailure
    case invalidHTTPResponse
    case httpStatus(Int)
    case invalidResponse
    case unsupportedAction
    case invalidCoordinate
}

struct LocalComputerUseGroundingHTTPResponse: Sendable {
    let statusCode: Int
    let contentType: String?
    let body: Data
}

protocol LocalComputerUseGroundingTransport: Sendable {
    func send(
        _ request: URLRequest,
        maximumResponseBytes: Int
    ) async throws -> LocalComputerUseGroundingHTTPResponse
}

/// A bounded, redirect-free transport for the app-owned loopback inference
/// endpoint. The response is consumed incrementally so a broken local peer
/// cannot hand the Desktop app an unbounded allocation.
struct URLSessionComputerUseGroundingTransport: LocalComputerUseGroundingTransport {
    private static let session: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 60
        configuration.timeoutIntervalForResource = 60
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.urlCache = nil
        return URLSession(
            configuration: configuration,
            delegate: LocalComputerUseNoRedirectDelegate(),
            delegateQueue: nil
        )
    }()

    private let session: URLSession

    init(session: URLSession = Self.session) {
        self.session = session
    }

    func send(
        _ request: URLRequest,
        maximumResponseBytes: Int
    ) async throws -> LocalComputerUseGroundingHTTPResponse {
        let (bytes, response) = try await session.bytes(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw LocalComputerUseVisualGrounderError.invalidHTTPResponse
        }

        var body = Data()
        body.reserveCapacity(min(maximumResponseBytes, 64 * 1024))
        for try await byte in bytes {
            try Task.checkCancellation()
            guard body.count < maximumResponseBytes else {
                throw LocalComputerUseVisualGrounderError.responseTooLarge
            }
            body.append(byte)
        }
        return LocalComputerUseGroundingHTTPResponse(
            statusCode: http.statusCode,
            contentType: http.value(forHTTPHeaderField: "Content-Type"),
            body: body
        )
    }
}

private final class LocalComputerUseNoRedirectDelegate: NSObject,
    URLSessionTaskDelegate, @unchecked Sendable
{
    func urlSession(
        _: URLSession,
        task _: URLSessionTask,
        willPerformHTTPRedirection _: HTTPURLResponse,
        newRequest _: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        completionHandler(nil)
    }
}

/// Local-only visual grounding for one compiled workflow step.
///
/// The model receives one ephemeral selected-window screenshot and may return
/// exactly one click. It cannot choose sequence, completion, retry policy,
/// consequence class, typed content, or approval. Those remain owned by
/// ``LocalWorkflowExecutor`` and its verifier/actuator boundaries.
actor LocalComputerUseVisualGrounder: LocalWorkflowGrounding {
    struct Configuration: Equatable, Sendable {
        static let maximumModelBytes = 256
        static let maximumBearerBytes = 4_096
        static let hardMaximumScreenshotBytes = 16 * 1024 * 1024
        static let hardMaximumRequestBytes = 24 * 1024 * 1024
        static let hardMaximumResponseBytes = 4 * 1024 * 1024
        static let hardMaximumDeadline: Duration = .seconds(60)

        let baseURL: URL
        let model: String
        let bearerToken: String?
        let deadline: Duration
        let maximumScreenshotBytes: Int
        let maximumRequestBytes: Int
        let maximumResponseBytes: Int

        init(
            baseURL: URL,
            model: String,
            bearerToken: String? = nil,
            deadline: Duration = .seconds(30),
            maximumScreenshotBytes: Int = 8 * 1024 * 1024,
            maximumRequestBytes: Int = 12 * 1024 * 1024,
            maximumResponseBytes: Int = 512 * 1024
        ) throws {
            guard Self.isAllowedLoopbackBaseURL(baseURL),
                  !model.isEmpty,
                  model.utf8.count <= Self.maximumModelBytes,
                  !model.unicodeScalars.contains(where: CharacterSet.controlCharacters.contains),
                  deadline > .zero,
                  deadline <= Self.hardMaximumDeadline,
                  maximumScreenshotBytes > 0,
                  maximumScreenshotBytes <= Self.hardMaximumScreenshotBytes,
                  maximumRequestBytes > maximumScreenshotBytes,
                  maximumRequestBytes <= Self.hardMaximumRequestBytes,
                  maximumResponseBytes > 0,
                  maximumResponseBytes <= Self.hardMaximumResponseBytes
            else {
                throw LocalComputerUseVisualGrounderError.invalidConfiguration
            }
            if let bearerToken {
                guard !bearerToken.isEmpty,
                      bearerToken.utf8.count <= Self.maximumBearerBytes,
                      !bearerToken.unicodeScalars.contains(
                        where: CharacterSet.controlCharacters.contains
                      )
                else {
                    throw LocalComputerUseVisualGrounderError.invalidConfiguration
                }
            }
            self.baseURL = baseURL
            self.model = model
            self.bearerToken = bearerToken
            self.deadline = deadline
            self.maximumScreenshotBytes = maximumScreenshotBytes
            self.maximumRequestBytes = maximumRequestBytes
            self.maximumResponseBytes = maximumResponseBytes
        }

        var completionURL: URL {
            var components = URLComponents(
                url: baseURL,
                resolvingAgainstBaseURL: false
            )!
            let prefix = components.path.hasSuffix("/")
                ? String(components.path.dropLast())
                : components.path
            components.path = prefix + "/chat/completions"
            return components.url!
        }

        private static func isAllowedLoopbackBaseURL(_ url: URL) -> Bool {
            guard let components = URLComponents(
                url: url,
                resolvingAgainstBaseURL: false
            ), components.scheme == "http",
              let rawHost = components.host?.lowercased()
            else {
                return false
            }
            let host = rawHost.hasPrefix("[") && rawHost.hasSuffix("]")
                ? String(rawHost.dropFirst().dropLast())
                : rawHost
            guard host == "127.0.0.1" || host == "::1",
              let port = components.port,
              (1 ... 65_535).contains(port),
              components.user == nil,
              components.password == nil,
              components.query == nil,
              components.fragment == nil,
              components.path == "/v1" || components.path == "/v1/"
            else {
                return false
            }
            return true
        }
    }

    private let configuration: Configuration
    private let vault: ComputerUseObservationVault
    private let transport: any LocalComputerUseGroundingTransport

    init(
        configuration: Configuration,
        vault: ComputerUseObservationVault,
        transport: any LocalComputerUseGroundingTransport =
            URLSessionComputerUseGroundingTransport()
    ) {
        self.configuration = configuration
        self.vault = vault
        self.transport = transport
    }

    func ground(
        step: LocalWorkflowStep,
        observation: WorkflowObservation
    ) async throws -> GroundedWorkflowAction {
        try Task.checkCancellation()
        guard observation.isStructurallyValid else {
            throw LocalComputerUseVisualGrounderError.invalidObservation
        }
        guard let artifact = await vault.artifact(for: observation.id),
              artifact.isStructurallyValid
        else {
            throw LocalComputerUseVisualGrounderError.missingObservationArtifact
        }
        guard artifact.pngData.count <= configuration.maximumScreenshotBytes else {
            throw LocalComputerUseVisualGrounderError.screenshotTooLarge
        }

        let request = try makeRequest(
            step: step,
            artifact: artifact
        )
        let response = try await sendWithDeadline(request)
        try Task.checkCancellation()
        guard (200 ... 299).contains(response.statusCode) else {
            throw LocalComputerUseVisualGrounderError.httpStatus(response.statusCode)
        }
        guard response.contentType?.lowercased().hasPrefix("application/json") == true else {
            throw LocalComputerUseVisualGrounderError.invalidHTTPResponse
        }
        let coordinate = try Self.decodeSingleClick(response.body)
        return GroundedWorkflowAction(
            observationID: observation.id,
            payload: .click(
                normalizedX: Double(coordinate.x) / 999,
                normalizedY: Double(coordinate.y) / 999
            ),
            source: .visualGrounding,
            safeSummary: "Click the target for \(step.title)",
            risk: step.risk
        )
    }

    private func makeRequest(
        step: LocalWorkflowStep,
        artifact: ComputerUseObservationArtifact
    ) throws -> URLRequest {
        let imageURL = "data:image/png;base64,\(artifact.pngData.base64EncodedString())"
        let tool: [String: Any] = [
            "type": "function",
            "function": [
                "name": "computer_use",
                "description": """
                Locate exactly one requested target in the supplied window screenshot. \
                Coordinates are normalized from 1 to 998 relative to that screenshot.
                """,
                "parameters": [
                    "type": "object",
                    "additionalProperties": false,
                    "properties": [
                        "action": ["type": "string", "enum": ["left_click"]],
                        "coordinate": [
                            "type": "array",
                            "items": ["type": "integer", "minimum": 1, "maximum": 998],
                            "minItems": 2,
                            "maxItems": 2,
                        ],
                    ],
                    "required": ["action", "coordinate"],
                ],
            ],
        ]
        let bodyObject: [String: Any] = [
            "model": configuration.model,
            "messages": [
                [
                    "role": "system",
                    "content": """
                    Locate one target only. Return one computer_use left_click tool call. \
                    Do not type, press keys, finish a workflow, or propose another action.
                    """,
                ],
                [
                    "role": "user",
                    "content": [
                        ["type": "image_url", "image_url": ["url": imageURL]],
                        ["type": "text", "text": step.instruction],
                    ],
                ],
            ],
            "tools": [tool],
            "tool_choice": [
                "type": "function",
                "function": ["name": "computer_use"],
            ],
            "temperature": 0,
            "max_tokens": 384,
            "chat_template_kwargs": ["enable_thinking": true],
        ]
        let body: Data
        do {
            body = try JSONSerialization.data(withJSONObject: bodyObject)
        } catch {
            throw LocalComputerUseVisualGrounderError.invalidConfiguration
        }
        guard body.count <= configuration.maximumRequestBytes else {
            throw LocalComputerUseVisualGrounderError.requestTooLarge
        }

        var request = URLRequest(url: configuration.completionURL)
        request.httpMethod = "POST"
        request.httpBody = body
        request.timeoutInterval = max(1, configuration.deadline.timeInterval)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if let bearer = configuration.bearerToken {
            request.setValue("Bearer \(bearer)", forHTTPHeaderField: "Authorization")
        }
        return request
    }

    private func sendWithDeadline(
        _ request: URLRequest
    ) async throws -> LocalComputerUseGroundingHTTPResponse {
        let transport = transport
        let maximumResponseBytes = configuration.maximumResponseBytes
        let deadline = configuration.deadline
        do {
            return try await withThrowingTaskGroup(
                of: LocalComputerUseGroundingHTTPResponse.self
            ) { group in
                group.addTask {
                    try await transport.send(
                        request,
                        maximumResponseBytes: maximumResponseBytes
                    )
                }
                group.addTask {
                    try await Task.sleep(for: deadline)
                    throw LocalComputerUseVisualGrounderError.deadlineExceeded
                }
                defer { group.cancelAll() }
                guard let first = try await group.next() else {
                    throw LocalComputerUseVisualGrounderError.transportFailure
                }
                return first
            }
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as LocalComputerUseVisualGrounderError {
            throw error
        } catch {
            throw LocalComputerUseVisualGrounderError.transportFailure
        }
    }

    private static func decodeSingleClick(_ data: Data) throws -> (x: Int, y: Int) {
        guard data.count > 0,
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = root["choices"] as? [[String: Any]],
              choices.count == 1,
              let message = choices[0]["message"] as? [String: Any],
              let calls = message["tool_calls"] as? [[String: Any]],
              calls.count == 1,
              let function = calls[0]["function"] as? [String: Any],
              function["name"] as? String == "computer_use",
              let arguments = function["arguments"] as? String,
              arguments.utf8.count <= 4_096,
              let argumentData = arguments.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: argumentData)
                as? [String: Any],
              let decoded = try? JSONDecoder().decode(
                ComputerUseClickArguments.self,
                from: argumentData
              )
        else {
            throw LocalComputerUseVisualGrounderError.invalidResponse
        }
        guard Set(object.keys).isSubset(of: ["action", "coordinate"]),
              decoded.action == "left_click"
        else {
            throw LocalComputerUseVisualGrounderError.unsupportedAction
        }
        guard let rawCoordinates = object["coordinate"] as? [NSNumber],
              rawCoordinates.count == 2,
              rawCoordinates.allSatisfy({ number in
                !["c", "f", "d"].contains(String(cString: number.objCType))
              }),
              decoded.coordinate.count == 2,
              let x = decoded.coordinate.first,
              let y = decoded.coordinate.last,
              (1 ... 998).contains(x),
              (1 ... 998).contains(y)
        else {
            throw LocalComputerUseVisualGrounderError.invalidCoordinate
        }
        return (x, y)
    }
}

private struct ComputerUseClickArguments: Decodable {
    let action: String
    let coordinate: [Int]
}

private extension Duration {
    var timeInterval: TimeInterval {
        let components = self.components
        return TimeInterval(components.seconds)
            + TimeInterval(components.attoseconds) / 1_000_000_000_000_000_000
    }
}
