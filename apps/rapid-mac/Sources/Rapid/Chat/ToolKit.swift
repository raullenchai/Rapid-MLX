import Foundation

/// OpenAI-compatible tool definition. Mirrors the wire shape so we
/// can serialise an array of these directly into the chat-completions
/// request body without an extra mapping layer.
///
/// Used by both:
///   * The model — to know what tools exist and what their parameters
///     look like (the JSON Schema is what the model "reads" in its
///     system prompt construction).
///   * The client — to validate that a returned tool call refers to
///     a tool we know how to run, and to feed the result back as the
///     next message.
struct ToolDefinition: Codable, Equatable, Hashable, Sendable {
    let type: String
    let function: Function

    struct Function: Codable, Equatable, Hashable, Sendable {
        let name: String
        let description: String
        /// JSON Schema (Draft 2020-12) describing the function's input.
        /// We carry it as a ``CodableJSON`` blob rather than typed
        /// fields because the schema shape is open-ended and we don't
        /// want a runtime parse error to block tool registration.
        let parameters: CodableJSON
    }

    init(name: String, description: String, parameters: CodableJSON) {
        self.type = "function"
        self.function = Function(name: name, description: description, parameters: parameters)
    }
}

/// One tool call returned by the assistant turn (or partially returned
/// across streaming deltas, then finalised). The ``arguments`` field is
/// a STRING in the OpenAI spec — the model emits JSON-stringified
/// arguments, and the client decodes it after the stream completes.
/// We preserve that shape verbatim so the wire envelope round-trips
/// without re-encoding noise.
struct ToolCall: Codable, Equatable, Hashable, Sendable, Identifiable {
    let id: String
    let type: String
    let function: Function

    struct Function: Codable, Equatable, Hashable, Sendable {
        let name: String
        /// JSON-stringified arguments. May be empty for no-arg tools.
        let arguments: String
    }

    init(id: String, name: String, arguments: String) {
        self.id = id
        self.type = "function"
        self.function = Function(name: name, arguments: arguments)
    }
}

/// Result of running one tool call. Goes back to the model as the
/// content of a ``role: "tool"`` message keyed by ``tool_call_id``.
struct ToolCallResult: Equatable, Hashable, Sendable {
    let toolCallID: String
    /// Free-form text the model gets to see. Almost always
    /// JSON-stringified by the tool author so the model can pattern-
    /// match against the schema it was told to expect.
    let content: String
    /// True if the tool execution failed (e.g. file not found, denied
    /// by the sandbox). The model gets the error string as ``content``
    /// either way — but the UI styles failed calls differently.
    let isError: Bool
    /// Stable UI classification. Raw ``content`` is still passed back to
    /// the model, while the transcript renders this diagnosis instead.
    let failureKind: FailureDiagnosis.Kind?

    init(
        toolCallID: String,
        content: String,
        isError: Bool = false,
        failureKind: FailureDiagnosis.Kind? = nil
    ) {
        self.toolCallID = toolCallID
        self.content = content
        self.isError = isError
        self.failureKind = failureKind
    }
}

// MARK: - Streaming accumulator

/// Accumulates ``delta.tool_calls`` slices into finalised ``ToolCall``s.
/// The OpenAI spec splits a single tool call across many SSE chunks:
/// the first carries ``id`` and ``function.name``, subsequent ones only
/// append fragments of ``function.arguments``. The ``index`` field is
/// the only stable key across chunks.
///
/// Usage:
///   * Call ``accept`` for each delta the SSE stream emits.
///   * Call ``finalize`` once ``finish_reason: "tool_calls"`` arrives;
///     it returns the calls ordered by index.
struct ToolCallAccumulator {
    private struct Builder {
        var id: String = ""
        var name: String = ""
        var arguments: String = ""
    }

    private var byIndex: [Int: Builder] = [:]

    /// Apply one streamed delta. Missing fields are left as-is so a
    /// chunk that only contains an arguments fragment can extend the
    /// builder without erasing the earlier id/name.
    mutating func accept(_ delta: ToolCallDelta) {
        var b = byIndex[delta.index] ?? Builder()
        if let id = delta.id, !id.isEmpty { b.id = id }
        if let name = delta.function?.name, !name.isEmpty { b.name = name }
        if let args = delta.function?.arguments {
            b.arguments += args
        }
        byIndex[delta.index] = b
    }

    /// Produce final ``ToolCall``s ordered by their delta ``index``.
    /// Empty if no deltas were accepted.
    ///
    /// Codex audit r1 (ToolKit.swift:122): both ``id`` and ``name``
    /// must be non-empty. The pre-audit shape required only
    /// ``id``, which let a malformed stream emit calls with no
    /// function name — the executor then routed them to nowhere
    /// and the tool round-trip silently failed instead of
    /// surfacing a clean "model produced a malformed tool call"
    /// banner. Dropping nameless entries is conservative; the
    /// caller's ``capturedCalls.isEmpty`` branch already covers
    /// the "no usable tool calls produced" path.
    func finalize() -> [ToolCall] {
        byIndex.keys
            .sorted()
            .compactMap { idx -> ToolCall? in
                guard let b = byIndex[idx] else { return nil }
                guard !b.id.isEmpty, !b.name.isEmpty else { return nil }
                return ToolCall(id: b.id, name: b.name, arguments: b.arguments)
            }
    }
}

/// On-the-wire shape of one streamed tool-call delta.
struct ToolCallDelta: Decodable, Equatable, Sendable {
    let index: Int
    let id: String?
    let type: String?
    let function: FunctionDelta?

    struct FunctionDelta: Decodable, Equatable, Sendable {
        let name: String?
        let arguments: String?
    }
}

// MARK: - Codable JSON blob

/// Type-erased JSON value. Used for tool ``parameters`` (the JSON
/// Schema body) and for tool inputs/outputs where we don't want to
/// commit to a typed Swift struct.
enum CodableJSON: Codable, Equatable, Hashable, Sendable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([CodableJSON])
    case object([String: CodableJSON])

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null; return }
        if let b = try? c.decode(Bool.self) { self = .bool(b); return }
        if let d = try? c.decode(Double.self) { self = .number(d); return }
        if let s = try? c.decode(String.self) { self = .string(s); return }
        if let arr = try? c.decode([CodableJSON].self) { self = .array(arr); return }
        if let obj = try? c.decode([String: CodableJSON].self) { self = .object(obj); return }
        throw DecodingError.dataCorruptedError(in: c, debugDescription: "unknown JSON shape")
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null:           try c.encodeNil()
        case .bool(let b):    try c.encode(b)
        case .number(let n):  try c.encode(n)
        case .string(let s):  try c.encode(s)
        case .array(let a):   try c.encode(a)
        case .object(let o):  try c.encode(o)
        }
    }

}

// MARK: - Tool runner contract

/// Anything that can execute a named tool call. ``ToolRegistry``
/// dispatches by ``ToolCall.function.name`` and returns a result the
/// chat loop can feed back to the model.
///
/// Implementations live in ``Sources/Rapid/Tools/`` and are
/// instantiated in ``RapidApp.swift``. The chat view model owns one
/// registry and passes it to every ``send`` so the same tool set is
/// available to every turn in a session.
@MainActor
protocol ToolRegistry: AnyObject, Sendable {
    /// Every tool the registry exposes — used to build the
    /// ``tools:`` array sent on the next request.
    var definitions: [ToolDefinition] { get }

    /// Run one call and return the textual result. The implementation
    /// is responsible for sandboxing / permission prompts; from the
    /// caller's perspective an unauthorised access just returns an
    /// error result, never throws.
    func run(_ call: ToolCall) async -> ToolCallResult
}

/// Schema-driven execution boundary for native model tool calls.
///
/// The model chooses a tool from the definitions advertised on this round.
/// This executor then applies the same policy to every built-in and connector:
/// refuse tools that were not advertised, require an arguments object, remove
/// top-level fields outside the tool's JSON schema, and only then dispatch.
/// Tool-specific intent parsing does not belong here or in the view model.
@MainActor
struct NativeToolCallExecutor {
    let registry: ToolRegistry

    func execute(
        _ call: ToolCall,
        advertised definitions: [ToolDefinition]
    ) async -> ToolCallResult {
        let knownNames = Set(registry.definitions.map { $0.function.name })
        guard let definition = definitions.first(where: {
            $0.function.name == call.function.name
        }) else {
            return ToolCallResult(
                toolCallID: call.id,
                content: Self.refusalMessage(
                    name: call.function.name,
                    allowed: Set(definitions.map { $0.function.name }),
                    known: knownNames
                ) ?? "tool '\(call.function.name)' is unavailable",
                isError: true,
                failureKind: .toolFailed
            )
        }

        guard let normalized = Self.normalized(call, for: definition) else {
            return ToolCallResult(
                toolCallID: call.id,
                content: "tool '\(call.function.name)' error: arguments must be a JSON object matching the advertised schema",
                isError: true,
                failureKind: .toolFailed
            )
        }
        return await registry.run(normalized)
    }

    nonisolated static func refusalMessage(
        name: String,
        allowed: Set<String>,
        known: Set<String>
    ) -> String? {
        if allowed.contains(name) { return nil }
        if known.contains(name) {
            return "tool '\(name)' isn't available in this conversation — answer directly, or ask the user to enable it in Settings."
        }
        let list = allowed.sorted().joined(separator: ", ")
        return "unknown tool '\(name)'\(list.isEmpty ? "" : " — available: \(list)"). Answer directly instead."
    }

    /// Normalize only the generic OpenAI tool envelope. Nested semantics stay
    /// with the tool's own decoder; this boundary intentionally knows nothing
    /// about locations, URLs, queries, or connector-specific values.
    nonisolated static func normalized(
        _ call: ToolCall,
        for definition: ToolDefinition
    ) -> ToolCall? {
        let raw = call.function.arguments.trimmingCharacters(in: .whitespacesAndNewlines)
        let data = Data((raw.isEmpty ? "{}" : raw).utf8)
        guard var object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }

        if case .object(let schema) = definition.function.parameters,
           case .object(let properties)? = schema["properties"]
        {
            object = object.filter { properties.keys.contains($0.key) }
        }
        guard JSONSerialization.isValidJSONObject(object),
              let normalized = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
              let arguments = String(data: normalized, encoding: .utf8)
        else { return nil }
        return ToolCall(id: call.id, name: call.function.name, arguments: arguments)
    }
}

/// Trivial empty registry. Used when no tool plumbing is wired up
/// yet — the chat loop sends ``tools: nil`` and skips the tool round-
/// trip. P4 swaps this for the real ``FilesystemToolRegistry``.
@MainActor
final class EmptyToolRegistry: ToolRegistry {
    var definitions: [ToolDefinition] { [] }
    func run(_ call: ToolCall) async -> ToolCallResult {
        ToolCallResult(toolCallID: call.id, content: "no tool registry wired up", isError: true)
    }
}
