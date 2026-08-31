import Foundation

/// Extracts durable user memories from a completed conversation turn by
/// asking the active model to review the recent messages and return a
/// structured JSON list of operations.
///
/// The review call is a single non-streaming POST to the same
/// ``v1/chat/completions`` endpoint the chat already uses. It runs in a
/// background task after ``isStreaming`` transitions to ``false``, so it
/// never blocks the UI or competes with an active stream.
struct MemoryExtractor {
    /// Maximum characters per message included in the review prompt.
    /// Open WebUI truncates at 1600 chars (1000 head + 400 tail); we
    /// use a similar bound so a long paste or code block doesn't blow
    /// the small model's context window.
    static let maximumMessageCharacters = 1_200
    /// Maximum number of recent messages included in the review prompt.
    static let maximumReviewedMessages = 16
    /// Upper bound on the non-streaming review request. Memory extraction
    /// is a short single-turn completion; 60 s is generous for a 4B model
    /// on a ~2K-token prompt.
    static let requestTimeout: TimeInterval = 60

    /// Errors surfaced to the caller for logging; never shown to the user
    /// as a banner because memory extraction is a background best-effort.
    enum ExtractError: Error, LocalizedError {
        case serverNotReady
        case httpStatus(Int)
        case emptyResponse
        case parseFailed

        var errorDescription: String? {
            switch self {
            case .serverNotReady: return "Memory extraction skipped: server not ready"
            case .httpStatus(let code): return "Memory extraction HTTP \(code)"
            case .emptyResponse: return "Memory extraction returned an empty response"
            case .parseFailed: return "Memory extraction could not parse model output"
            }
        }
    }

    /// A structured operation the review model returned.
    enum Operation: Equatable, Sendable {
        case add(String)
        case remove(String)
    }

    let baseURL: URL
    let bearerToken: String?

    init(baseURL: URL, bearerToken: String? = nil) {
        self.baseURL = baseURL
        self.bearerToken = bearerToken
    }

    /// Runs the review and returns the parsed operations. The caller
    /// decides how to apply them (typically via ``MemoryStore.upsert``).
    func extract(
        model: String,
        messages: [(role: String, content: String)]
    ) async throws -> [Operation] {
        let transcript = Self.buildTranscript(from: messages)
        guard !transcript.isEmpty else { return [] }

        let prompt = Self.reviewPrompt(transcript: transcript)
        let response = try await Self.sendCompletion(
            baseURL: baseURL,
            bearerToken: bearerToken,
            model: model,
            prompt: prompt
        )
        return Self.parseOperations(from: response)
    }

    // MARK: - Prompt Construction

    /// Formats recent messages into a compact transcript for the review
    /// prompt. Tool messages and system rows are excluded — they are not
    /// user-authored content and would dilute the reviewer's signal.
    static func buildTranscript(
        from messages: [(role: String, content: String)]
    ) -> String {
        let relevant = messages
            .filter { $0.role == "user" || $0.role == "assistant" }
            .suffix(maximumReviewedMessages)
        guard !relevant.isEmpty else { return "" }

        return relevant.map { message in
            let content = message.content.count > maximumMessageCharacters
                ? String(message.content.prefix(maximumMessageCharacters)) + "…"
                : message.content
            return "\(message.role): \(content)"
        }.joined(separator: "\n")
    }

    /// The system-level prompt sent to the reviewer model. Mirrors Open
    /// WebUI's approach: strict JSON schema, conservative rules about
    /// what to remember, and an explicit "return empty" path.
    static func reviewPrompt(transcript: String) -> String {
        """
        Review the following conversation and decide whether any durable user preference or fact should be remembered for future conversations.

        Rules:
        - Save only enduring details: preferences, recurring tasks, environment quirks, long-term context.
        - Do NOT save one-off questions, temporary states, secrets, or transient task steps.
        - Do NOT save anything the user did not explicitly say or confirm.
        - Return ONLY a JSON array. Each element is either:
          {"action": "add", "content": "..."} or {"action": "remove", "content": "..."}
        - "add" saves a new fact. "remove" removes a previously saved fact that is now wrong.
        - Return [] if nothing should change.

        Conversation:
        \(transcript)
        """
    }

    // MARK: - HTTP

    /// Non-streaming POST to the chat-completions endpoint. Reuses the
    /// same URL shape and auth pattern as ``ChatStreamClient`` but
    /// avoids SSE parsing because the memory review doesn't need
    /// progressive output.
    static func sendCompletion(
        baseURL: URL,
        bearerToken: String?,
        model: String,
        prompt: String
    ) async throws -> String {
        let url = ChatStreamClient.chatCompletionsURL(base: baseURL)
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = requestTimeout

        if let bearerToken, !bearerToken.isEmpty {
            request.setValue("Bearer \(bearerToken)", forHTTPHeaderField: "Authorization")
        }

        let body: [String: Any] = [
            "model": model,
            "messages": [
                [
                    "role": "system",
                    "content": "You are a memory reviewer. Return only valid JSON."
                ],
                ["role": "user", "content": prompt]
            ],
            "stream": false,
            "max_tokens": 512,
            "temperature": 0.1
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw ExtractError.serverNotReady
        }
        guard httpResponse.statusCode == 200 else {
            throw ExtractError.httpStatus(httpResponse.statusCode)
        }

        struct CompletionResponse: Decodable {
            struct Choice: Decodable {
                struct Message: Decodable {
                    let content: String?
                }
                let message: Message
            }
            let choices: [Choice]
        }

        guard let decoded = try? JSONDecoder().decode(CompletionResponse.self, from: data),
              let content = decoded.choices.first?.message.content,
              !content.isEmpty
        else {
            throw ExtractError.emptyResponse
        }
        return content
    }

    // MARK: - Parsing

    /// Extracts operations from the model's JSON output. Tolerates
    /// leading/trailing prose (some models wrap JSON in markdown code
    /// fences) by scanning for the first `[` and last `]`.
    static func parseOperations(from content: String) -> [Operation] {
        guard let start = content.firstIndex(of: "["),
              let end = content.lastIndex(of: "]"),
              start < end
        else { return [] }

        let jsonString = String(content[start...end])
        guard let data = jsonString.data(using: .utf8),
              let array = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        else { return [] }

        return array.compactMap { item in
            let action = item["action"] as? String
            let opContent = (item["content"] as? String)
                ?? (item["fact"] as? String)
                ?? (item["memory"] as? String)
            guard let opContent,
                  !opContent.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else { return nil }

            switch action {
            case "add", nil:
                // Some models omit "action" and just emit {"fact": "..."}.
                // Treat a bare fact as an implicit add.
                return .add(opContent)
            case "remove":
                return .remove(opContent)
            default:
                return nil
            }
        }
    }
}
