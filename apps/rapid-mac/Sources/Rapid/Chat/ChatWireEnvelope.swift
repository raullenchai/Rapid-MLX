import Foundation

/// Wire-v1 values. Known payloads are strict; unknown event discriminators are
/// consumed as sequence-bearing no-ops by the experimental client.
/// Keep field names/defaults aligned with tests/fixtures/chat_runtime/wire-v1.schema.json.
enum ChatWireError: Error, Equatable {
    case invalidField(String)
    case unknownFields
    case unsupportedVersion
    case invalidFrame
    case frameTooLarge
    case unexpectedEOF
    case wrongTurn
}

private struct WireKey: CodingKey {
    let stringValue: String
    var intValue: Int? { nil }
    init(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
}

private func checkWireKeys(
    _ decoder: Decoder, allowed: Set<String>, eventTypes: Set<String> = []
) throws {
    let fields = try decoder.container(keyedBy: WireKey.self)
    guard Set(fields.allKeys.map(\.stringValue)).isSubset(of: allowed) else {
        throw ChatWireError.unknownFields
    }
    let typeKey = WireKey(stringValue: "type")
    if !eventTypes.isEmpty, fields.contains(typeKey) {
        let type = try fields.decode(String.self, forKey: typeKey)
        guard eventTypes.contains(type) else { throw ChatWireError.invalidField("type") }
    }
}

private extension KeyedDecodingContainer {
    // A missing defaulted field is valid; explicit null is valid only for optionals.
    func value<T: Decodable>(_ type: T.Type, forKey key: Key, default fallback: T) throws -> T {
        contains(key) ? try decode(type, forKey: key) : fallback
    }
}

/// JSON tool arguments and unversioned owner receipts, without Foundation Any.
enum ChatWireJSON: Codable, Equatable, Sendable {
    case null, bool(Bool), integer(Int), number(Double), string(String)
    case array([ChatWireJSON]), object([String: ChatWireJSON])

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let v = try? c.decode(Bool.self) { self = .bool(v) }
        else if let v = try? c.decode(Int.self) { self = .integer(v) }
        else if let v = try? c.decode(Double.self) { self = .number(v) }
        else if let v = try? c.decode(String.self) { self = .string(v) }
        else if let v = try? c.decode([ChatWireJSON].self) { self = .array(v) }
        else { self = .object(try c.decode([String: ChatWireJSON].self)) }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let v): try c.encode(v)
        case .integer(let v): try c.encode(v)
        case .number(let v): try c.encode(v)
        case .string(let v): try c.encode(v)
        case .array(let v): try c.encode(v)
        case .object(let v): try c.encode(v)
        }
    }
}

struct ChatWireContextEvent: Codable, Equatable, Sendable {
    let usedTokens: Int
    let limitTokens: Int
    let systemToolsTokens: Int
    let conversationTokens: Int
    let toolOutputTokens: Int
    let freeTokens: Int
    let percent: Int
    let estimated: Bool
    let compacted: Bool
    let notice: String

    private enum CodingKeys: String, CodingKey {
        case usedTokens = "used_tokens"
        case limitTokens = "limit_tokens"
        case systemToolsTokens = "system_tools_tokens"
        case conversationTokens = "conversation_tokens"
        case toolOutputTokens = "tool_output_tokens"
        case freeTokens = "free_tokens"
        case percent = "percent"
        case estimated = "estimated"
        case compacted = "compacted"
        case notice = "notice"
    }

    init(from decoder: Decoder) throws {
        try checkWireKeys(decoder, allowed: ["type", "used_tokens", "limit_tokens", "system_tools_tokens", "conversation_tokens", "tool_output_tokens", "free_tokens", "percent", "estimated", "compacted", "notice"], eventTypes: ["context"])
        let c = try decoder.container(keyedBy: CodingKeys.self)
        usedTokens = try c.decode(Int.self, forKey: .usedTokens)
        guard usedTokens >= 0 else { throw ChatWireError.invalidField("used_tokens") }
        limitTokens = try c.decode(Int.self, forKey: .limitTokens)
        guard limitTokens >= 1 else { throw ChatWireError.invalidField("limit_tokens") }
        systemToolsTokens = try c.decode(Int.self, forKey: .systemToolsTokens)
        guard systemToolsTokens >= 0 else { throw ChatWireError.invalidField("system_tools_tokens") }
        conversationTokens = try c.decode(Int.self, forKey: .conversationTokens)
        guard conversationTokens >= 0 else { throw ChatWireError.invalidField("conversation_tokens") }
        toolOutputTokens = try c.decode(Int.self, forKey: .toolOutputTokens)
        guard toolOutputTokens >= 0 else { throw ChatWireError.invalidField("tool_output_tokens") }
        freeTokens = try c.decode(Int.self, forKey: .freeTokens)
        guard freeTokens >= 0 else { throw ChatWireError.invalidField("free_tokens") }
        percent = try c.decode(Int.self, forKey: .percent)
        guard percent >= 0, percent <= 100 else { throw ChatWireError.invalidField("percent") }
        estimated = try c.value(Bool.self, forKey: .estimated, default: false)
        compacted = try c.value(Bool.self, forKey: .compacted, default: false)
        notice = try c.value(String.self, forKey: .notice, default: "")
    }
}

struct ChatWirePermissionRequest: Codable, Equatable, Sendable {
    enum PreviewLanguage: String, Codable, Sendable {
        case python = "python"
        case bash = "bash"
        case diff = "diff"
        case text = "text"
    }
    enum Options: String, Codable, Sendable {
        case once = "once"
        case conversation = "conversation"
        case deny = "deny"
    }
    let opID: String
    let revision: Int
    let tool: String
    let title: String
    let previewLanguage: PreviewLanguage
    let summary: String
    let preview: String
    let risk: String
    let reason: String
    let options: [Options]
    let expiresAt: Double
    let subagentID: String?

    private enum CodingKeys: String, CodingKey {
        case opID = "op_id"
        case revision = "revision"
        case tool = "tool"
        case title = "title"
        case previewLanguage = "preview_language"
        case summary = "summary"
        case preview = "preview"
        case risk = "risk"
        case reason = "reason"
        case options = "options"
        case expiresAt = "expires_at"
        case subagentID = "subagent_id"
    }

    init(from decoder: Decoder) throws {
        try checkWireKeys(decoder, allowed: ["type", "op_id", "revision", "tool", "title", "preview_language", "summary", "preview", "risk", "reason", "options", "expires_at", "subagent_id"], eventTypes: ["permission.request"])
        let c = try decoder.container(keyedBy: CodingKeys.self)
        opID = try c.decode(String.self, forKey: .opID)
        revision = try c.decode(Int.self, forKey: .revision)
        guard revision >= 0 else { throw ChatWireError.invalidField("revision") }
        tool = try c.decode(String.self, forKey: .tool)
        title = try c.value(String.self, forKey: .title, default: "Allow this action?")
        previewLanguage = try c.value(PreviewLanguage.self, forKey: .previewLanguage, default: .text)
        summary = try c.decode(String.self, forKey: .summary)
        preview = try c.decode(String.self, forKey: .preview)
        risk = try c.decode(String.self, forKey: .risk)
        reason = try c.decode(String.self, forKey: .reason)
        options = try c.decode([Options].self, forKey: .options)
        expiresAt = try c.decode(Double.self, forKey: .expiresAt)
        subagentID = try c.decodeIfPresent(String.self, forKey: .subagentID)
    }
}

struct ChatWireReasoningDelta: Codable, Equatable, Sendable {
    let partID: String
    let text: String
    let omitted: Bool

    private enum CodingKeys: String, CodingKey {
        case partID = "part_id"
        case text = "text"
        case omitted = "omitted"
    }

    init(from decoder: Decoder) throws {
        try checkWireKeys(decoder, allowed: ["type", "part_id", "text", "omitted"], eventTypes: ["reasoning.delta"])
        let c = try decoder.container(keyedBy: CodingKeys.self)
        partID = try c.decode(String.self, forKey: .partID)
        text = try c.value(String.self, forKey: .text, default: "")
        omitted = try c.value(Bool.self, forKey: .omitted, default: false)
    }
}

struct ChatWireSource: Codable, Equatable, Sendable {
    let sourceID: String
    let callID: String
    let title: String
    let attachmentID: String?
    let url: String?

    private enum CodingKeys: String, CodingKey {
        case sourceID = "source_id"
        case callID = "call_id"
        case title = "title"
        case attachmentID = "attachment_id"
        case url = "url"
    }

    init(from decoder: Decoder) throws {
        try checkWireKeys(decoder, allowed: ["type", "source_id", "call_id", "title", "attachment_id", "url"], eventTypes: ["source"])
        let c = try decoder.container(keyedBy: CodingKeys.self)
        sourceID = try c.decode(String.self, forKey: .sourceID)
        callID = try c.decode(String.self, forKey: .callID)
        title = try c.decode(String.self, forKey: .title)
        attachmentID = try c.decodeIfPresent(String.self, forKey: .attachmentID)
        url = try c.decodeIfPresent(String.self, forKey: .url)
    }
}

struct ChatWireSubagent: Codable, Equatable, Sendable {
    enum Status: String, Codable, Sendable {
        case queued = "queued"
        case running = "running"
        case completed = "completed"
        case error = "error"
        case cancelled = "cancelled"
        case budgetExhausted = "budget_exhausted"
        case timeout = "timeout"
    }
    let id: String
    let callID: String
    let title: String
    let status: Status
    let toolCount: Int
    let summary: String

    private enum CodingKeys: String, CodingKey {
        case id = "id"
        case callID = "call_id"
        case title = "title"
        case status = "status"
        case toolCount = "tool_count"
        case summary = "summary"
    }

    init(from decoder: Decoder) throws {
        try checkWireKeys(decoder, allowed: ["id", "call_id", "title", "status", "tool_count", "summary", "type"], eventTypes: ["subagent.started", "subagent.progress", "subagent.finished"])
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        callID = try c.decode(String.self, forKey: .callID)
        title = try c.decode(String.self, forKey: .title)
        status = try c.decode(Status.self, forKey: .status)
        toolCount = try c.value(Int.self, forKey: .toolCount, default: 0)
        guard toolCount >= 0 else { throw ChatWireError.invalidField("tool_count") }
        summary = try c.value(String.self, forKey: .summary, default: "")
    }
}

struct ChatWireTextDelta: Codable, Equatable, Sendable {
    let partID: String
    let text: String

    private enum CodingKeys: String, CodingKey {
        case partID = "part_id"
        case text = "text"
    }

    init(from decoder: Decoder) throws {
        try checkWireKeys(decoder, allowed: ["type", "part_id", "text"], eventTypes: ["text.delta"])
        let c = try decoder.container(keyedBy: CodingKeys.self)
        partID = try c.decode(String.self, forKey: .partID)
        text = try c.decode(String.self, forKey: .text)
    }
}

struct ChatWireToolOutput: Codable, Equatable, Sendable {
    let callID: String
    let text: String

    private enum CodingKeys: String, CodingKey {
        case callID = "call_id"
        case text = "text"
    }

    init(from decoder: Decoder) throws {
        try checkWireKeys(decoder, allowed: ["type", "call_id", "text"], eventTypes: ["tool.output"])
        let c = try decoder.container(keyedBy: CodingKeys.self)
        callID = try c.decode(String.self, forKey: .callID)
        text = try c.decode(String.self, forKey: .text)
    }
}

struct ChatWireToolResult: Codable, Equatable, Sendable {
    enum DetailsKind: String, Codable, Sendable {
        case diff = "diff"
        case output = "output"
        case text = "text"
    }
    enum Outcome: String, Codable, Sendable {
        case success = "success"
        case error = "error"
    }
    let callID: String
    let result: String
    let label: String
    let target: String
    let summary: String
    let details: [String]
    let detailsKind: DetailsKind
    let omittedLines: Int
    let outcome: Outcome

    private enum CodingKeys: String, CodingKey {
        case callID = "call_id"
        case result = "result"
        case label = "label"
        case target = "target"
        case summary = "summary"
        case details = "details"
        case detailsKind = "details_kind"
        case omittedLines = "omitted_lines"
        case outcome = "outcome"
    }

    init(from decoder: Decoder) throws {
        try checkWireKeys(decoder, allowed: ["type", "call_id", "result", "label", "target", "summary", "details", "details_kind", "omitted_lines", "outcome"], eventTypes: ["tool.result"])
        let c = try decoder.container(keyedBy: CodingKeys.self)
        callID = try c.decode(String.self, forKey: .callID)
        result = try c.decode(String.self, forKey: .result)
        label = try c.value(String.self, forKey: .label, default: "")
        target = try c.value(String.self, forKey: .target, default: "")
        summary = try c.value(String.self, forKey: .summary, default: "")
        details = try c.value([String].self, forKey: .details, default: [])
        detailsKind = try c.value(DetailsKind.self, forKey: .detailsKind, default: .text)
        omittedLines = try c.value(Int.self, forKey: .omittedLines, default: 0)
        guard omittedLines >= 0 else { throw ChatWireError.invalidField("omitted_lines") }
        outcome = try c.value(Outcome.self, forKey: .outcome, default: .success)
    }
}

struct ChatWireToolStarted: Codable, Equatable, Sendable {
    let callID: String
    let name: String
    let label: String
    let target: String
    let summary: String
    let args: [String: ChatWireJSON]

    private enum CodingKeys: String, CodingKey {
        case callID = "call_id"
        case name = "name"
        case label = "label"
        case target = "target"
        case summary = "summary"
        case args = "args"
    }

    init(from decoder: Decoder) throws {
        try checkWireKeys(decoder, allowed: ["type", "call_id", "name", "label", "target", "summary", "args"], eventTypes: ["tool.started"])
        let c = try decoder.container(keyedBy: CodingKeys.self)
        callID = try c.decode(String.self, forKey: .callID)
        name = try c.decode(String.self, forKey: .name)
        label = try c.value(String.self, forKey: .label, default: "")
        target = try c.value(String.self, forKey: .target, default: "")
        summary = try c.value(String.self, forKey: .summary, default: "")
        args = try c.value([String: ChatWireJSON].self, forKey: .args, default: [:])
    }
}

struct ChatWireTurnStarted: Codable, Equatable, Sendable {
    let messageID: String
    let parentMessageID: String?
    let branchID: String
    let model: String
    let thinking: Bool

    private enum CodingKeys: String, CodingKey {
        case messageID = "message_id"
        case parentMessageID = "parent_message_id"
        case branchID = "branch_id"
        case model = "model"
        case thinking = "thinking"
    }

    init(from decoder: Decoder) throws {
        try checkWireKeys(decoder, allowed: ["type", "message_id", "parent_message_id", "branch_id", "model", "thinking"], eventTypes: ["turn.started"])
        let c = try decoder.container(keyedBy: CodingKeys.self)
        messageID = try c.decode(String.self, forKey: .messageID)
        parentMessageID = try c.decodeIfPresent(String.self, forKey: .parentMessageID)
        branchID = try c.decode(String.self, forKey: .branchID)
        model = try c.decode(String.self, forKey: .model)
        thinking = try c.decode(Bool.self, forKey: .thinking)
    }
}

struct ChatWireTurnTerminal: Codable, Equatable, Sendable {
    enum Outcome: String, Codable, Sendable {
        case completed = "completed"
        case cancelled = "cancelled"
        case error = "error"
    }
    let outcome: Outcome
    let finalMessageID: String?
    let usage: ChatWireUsage
    let errorCode: String?

    private enum CodingKeys: String, CodingKey {
        case outcome = "outcome"
        case finalMessageID = "final_message_id"
        case usage = "usage"
        case errorCode = "error_code"
    }

    init(from decoder: Decoder) throws {
        try checkWireKeys(decoder, allowed: ["type", "outcome", "final_message_id", "usage", "error_code"], eventTypes: ["turn.terminal"])
        let c = try decoder.container(keyedBy: CodingKeys.self)
        outcome = try c.decode(Outcome.self, forKey: .outcome)
        finalMessageID = try c.decodeIfPresent(String.self, forKey: .finalMessageID)
        usage = try c.value(ChatWireUsage.self, forKey: .usage, default: ChatWireUsage())
        errorCode = try c.decodeIfPresent(String.self, forKey: .errorCode)
    }
}

struct ChatWireUsage: Codable, Equatable, Sendable {
    let inputTokens: Int
    let outputTokens: Int
    let requests: Int

    private enum CodingKeys: String, CodingKey {
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
        case requests = "requests"
    }

    init(
        inputTokens: Int = 0,
        outputTokens: Int = 0,
        requests: Int = 0
    ) {
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
        self.requests = requests
    }

    init(from decoder: Decoder) throws {
        try checkWireKeys(decoder, allowed: ["input_tokens", "output_tokens", "requests"])
        let c = try decoder.container(keyedBy: CodingKeys.self)
        inputTokens = try c.value(Int.self, forKey: .inputTokens, default: 0)
        guard inputTokens >= 0 else { throw ChatWireError.invalidField("input_tokens") }
        outputTokens = try c.value(Int.self, forKey: .outputTokens, default: 0)
        guard outputTokens >= 0 else { throw ChatWireError.invalidField("output_tokens") }
        requests = try c.value(Int.self, forKey: .requests, default: 0)
        guard requests >= 0 else { throw ChatWireError.invalidField("requests") }
    }
}

struct ChatWireUsageEvent: Codable, Equatable, Sendable {
    let inputTokens: Int
    let outputTokens: Int
    let requests: Int

    private enum CodingKeys: String, CodingKey {
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
        case requests = "requests"
    }

    init(from decoder: Decoder) throws {
        try checkWireKeys(decoder, allowed: ["input_tokens", "output_tokens", "requests", "type"], eventTypes: ["usage"])
        let c = try decoder.container(keyedBy: CodingKeys.self)
        inputTokens = try c.value(Int.self, forKey: .inputTokens, default: 0)
        guard inputTokens >= 0 else { throw ChatWireError.invalidField("input_tokens") }
        outputTokens = try c.value(Int.self, forKey: .outputTokens, default: 0)
        guard outputTokens >= 0 else { throw ChatWireError.invalidField("output_tokens") }
        requests = try c.value(Int.self, forKey: .requests, default: 0)
        guard requests >= 0 else { throw ChatWireError.invalidField("requests") }
    }
}

struct ChatPermissionResponse: Codable, Equatable, Sendable {
    enum Decision: String, Codable, Sendable {
        case allow = "allow"
        case deny = "deny"
    }
    enum Scope: String, Codable, Sendable {
        case once = "once"
        case conversation = "conversation"
    }
    let opID: String
    let revision: Int
    let decision: Decision
    let scope: Scope
    let reason: String

    private enum CodingKeys: String, CodingKey {
        case opID = "op_id"
        case revision = "revision"
        case decision = "decision"
        case scope = "scope"
        case reason = "reason"
    }

    init(
        opID: String,
        revision: Int,
        decision: Decision,
        scope: Scope = .once,
        reason: String = ""
    ) {
        self.opID = opID
        self.revision = revision
        self.decision = decision
        self.scope = scope
        self.reason = reason
    }

    init(from decoder: Decoder) throws {
        try checkWireKeys(decoder, allowed: ["op_id", "revision", "decision", "scope", "reason"])
        let c = try decoder.container(keyedBy: CodingKeys.self)
        opID = try c.decode(String.self, forKey: .opID)
        revision = try c.decode(Int.self, forKey: .revision)
        guard revision >= 0 else { throw ChatWireError.invalidField("revision") }
        decision = try c.decode(Decision.self, forKey: .decision)
        scope = try c.value(Scope.self, forKey: .scope, default: .once)
        reason = try c.value(String.self, forKey: .reason, default: "")
        guard reason.unicodeScalars.count <= 512 else { throw ChatWireError.invalidField("reason") }
    }
}

struct ChatTurnRequest: Codable, Equatable, Sendable {
    enum Surface: String, Codable, Sendable {
        case desktop = "desktop"
        case cli = "cli"
        case plain = "plain"
    }
    enum Approvals: String, Codable, Sendable {
        case `default` = "default"
        case always = "always"
    }
    enum Mode: String, Codable, Sendable {
        case append = "append"
        case regenerate = "regenerate"
    }
    let v: Int
    let turnID: String
    let conversationID: String
    let workspace: String?
    let branchID: String
    let expectedRevision: Int
    let text: String
    let attachmentIDs: [String]
    let inferenceURL: String
    let model: String
    let surface: Surface
    let thinking: Bool
    let actionTools: Bool
    let approvals: Approvals
    let temperature: Double
    let maxTokens: Int
    let contextTokens: Int
    let mode: Mode
    let inferenceAPIKey: String?

    private enum CodingKeys: String, CodingKey {
        case v = "v"
        case turnID = "turn_id"
        case conversationID = "conversation_id"
        case workspace = "workspace"
        case branchID = "branch_id"
        case expectedRevision = "expected_revision"
        case text = "text"
        case attachmentIDs = "attachment_ids"
        case inferenceURL = "inference_url"
        case model = "model"
        case surface = "surface"
        case thinking = "thinking"
        case actionTools = "action_tools"
        case approvals = "approvals"
        case temperature = "temperature"
        case maxTokens = "max_tokens"
        case contextTokens = "context_tokens"
        case mode = "mode"
        case inferenceAPIKey = "inference_api_key"
    }

    init(
        v: Int = 1,
        turnID: String,
        conversationID: String,
        workspace: String? = nil,
        branchID: String,
        expectedRevision: Int,
        text: String = "",
        attachmentIDs: [String] = [],
        inferenceURL: String,
        model: String,
        surface: Surface = .desktop,
        thinking: Bool = false,
        actionTools: Bool = true,
        approvals: Approvals = .`default`,
        temperature: Double = 0.7,
        maxTokens: Int = 32768,
        contextTokens: Int = 32768,
        mode: Mode = .append,
        inferenceAPIKey: String? = nil
    ) {
        self.v = v
        self.turnID = turnID
        self.conversationID = conversationID
        self.workspace = workspace
        self.branchID = branchID
        self.expectedRevision = expectedRevision
        self.text = text
        self.attachmentIDs = attachmentIDs
        self.inferenceURL = inferenceURL
        self.model = model
        self.surface = surface
        self.thinking = thinking
        self.actionTools = actionTools
        self.approvals = approvals
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.contextTokens = contextTokens
        self.mode = mode
        self.inferenceAPIKey = inferenceAPIKey
    }

    init(from decoder: Decoder) throws {
        try checkWireKeys(decoder, allowed: ["v", "turn_id", "conversation_id", "workspace", "branch_id", "expected_revision", "text", "attachment_ids", "inference_url", "model", "surface", "thinking", "action_tools", "approvals", "temperature", "max_tokens", "context_tokens", "mode", "inference_api_key"])
        let c = try decoder.container(keyedBy: CodingKeys.self)
        v = try c.value(Int.self, forKey: .v, default: 1)
        guard v == 1 else { throw ChatWireError.invalidField("v") }
        turnID = try c.decode(String.self, forKey: .turnID)
        guard turnID.range(of: "^[A-Za-z0-9_-]{1,80}$", options: .regularExpression) != nil else { throw ChatWireError.invalidField("turn_id") }
        conversationID = try c.decode(String.self, forKey: .conversationID)
        workspace = try c.decodeIfPresent(String.self, forKey: .workspace)
        guard workspace == nil || workspace!.unicodeScalars.count <= 4096 else { throw ChatWireError.invalidField("workspace") }
        branchID = try c.decode(String.self, forKey: .branchID)
        expectedRevision = try c.decode(Int.self, forKey: .expectedRevision)
        guard expectedRevision >= 0 else { throw ChatWireError.invalidField("expected_revision") }
        text = try c.value(String.self, forKey: .text, default: "")
        guard text.unicodeScalars.count <= 262144 else { throw ChatWireError.invalidField("text") }
        attachmentIDs = try c.value([String].self, forKey: .attachmentIDs, default: [])
        guard attachmentIDs.count <= 32 else { throw ChatWireError.invalidField("attachment_ids") }
        inferenceURL = try c.decode(String.self, forKey: .inferenceURL)
        model = try c.decode(String.self, forKey: .model)
        guard model.unicodeScalars.count >= 1, model.unicodeScalars.count <= 512 else { throw ChatWireError.invalidField("model") }
        surface = try c.value(Surface.self, forKey: .surface, default: .desktop)
        thinking = try c.value(Bool.self, forKey: .thinking, default: false)
        actionTools = try c.value(Bool.self, forKey: .actionTools, default: true)
        approvals = try c.value(Approvals.self, forKey: .approvals, default: .`default`)
        temperature = try c.value(Double.self, forKey: .temperature, default: 0.7)
        guard temperature >= 0, temperature <= 2 else { throw ChatWireError.invalidField("temperature") }
        maxTokens = try c.value(Int.self, forKey: .maxTokens, default: 32768)
        guard maxTokens >= 1, maxTokens <= 32768 else { throw ChatWireError.invalidField("max_tokens") }
        contextTokens = try c.value(Int.self, forKey: .contextTokens, default: 32768)
        guard contextTokens >= 512, contextTokens <= 131072 else { throw ChatWireError.invalidField("context_tokens") }
        mode = try c.value(Mode.self, forKey: .mode, default: .append)
        inferenceAPIKey = try c.decodeIfPresent(String.self, forKey: .inferenceAPIKey)
    }
}

enum ChatWireEventKind: String, CaseIterable, Sendable {
    case context = "context"
    case permissionRequest = "permission.request"
    case reasoningDelta = "reasoning.delta"
    case source = "source"
    case subagentFinished = "subagent.finished"
    case subagentProgress = "subagent.progress"
    case subagentStarted = "subagent.started"
    case textDelta = "text.delta"
    case toolOutput = "tool.output"
    case toolResult = "tool.result"
    case toolStarted = "tool.started"
    case turnStarted = "turn.started"
    case turnTerminal = "turn.terminal"
    case usage = "usage"
}

enum ChatWireEvent: Decodable, Equatable, Sendable {
    case context(ChatWireContextEvent)
    case permissionRequest(ChatWirePermissionRequest)
    case reasoningDelta(ChatWireReasoningDelta)
    case source(ChatWireSource)
    case subagentFinished(ChatWireSubagent)
    case subagentProgress(ChatWireSubagent)
    case subagentStarted(ChatWireSubagent)
    case textDelta(ChatWireTextDelta)
    case toolOutput(ChatWireToolOutput)
    case toolResult(ChatWireToolResult)
    case toolStarted(ChatWireToolStarted)
    case turnStarted(ChatWireTurnStarted)
    case turnTerminal(ChatWireTurnTerminal)
    case usage(ChatWireUsageEvent)
    case unknown(String)

    var type: String {
        switch self {
        case .context: return "context"
        case .permissionRequest: return "permission.request"
        case .reasoningDelta: return "reasoning.delta"
        case .source: return "source"
        case .subagentFinished: return "subagent.finished"
        case .subagentProgress: return "subagent.progress"
        case .subagentStarted: return "subagent.started"
        case .textDelta: return "text.delta"
        case .toolOutput: return "tool.output"
        case .toolResult: return "tool.result"
        case .toolStarted: return "tool.started"
        case .turnStarted: return "turn.started"
        case .turnTerminal: return "turn.terminal"
        case .usage: return "usage"
        case .unknown(let type): return type
        }
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: WireKey.self)
        let type = try c.decode(String.self, forKey: WireKey(stringValue: "type"))
        switch ChatWireEventKind(rawValue: type) {
        case .context: self = .context(try ChatWireContextEvent(from: decoder))
        case .permissionRequest: self = .permissionRequest(try ChatWirePermissionRequest(from: decoder))
        case .reasoningDelta: self = .reasoningDelta(try ChatWireReasoningDelta(from: decoder))
        case .source: self = .source(try ChatWireSource(from: decoder))
        case .subagentFinished: self = .subagentFinished(try ChatWireSubagent(from: decoder))
        case .subagentProgress: self = .subagentProgress(try ChatWireSubagent(from: decoder))
        case .subagentStarted: self = .subagentStarted(try ChatWireSubagent(from: decoder))
        case .textDelta: self = .textDelta(try ChatWireTextDelta(from: decoder))
        case .toolOutput: self = .toolOutput(try ChatWireToolOutput(from: decoder))
        case .toolResult: self = .toolResult(try ChatWireToolResult(from: decoder))
        case .toolStarted: self = .toolStarted(try ChatWireToolStarted(from: decoder))
        case .turnStarted: self = .turnStarted(try ChatWireTurnStarted(from: decoder))
        case .turnTerminal: self = .turnTerminal(try ChatWireTurnTerminal(from: decoder))
        case .usage: self = .usage(try ChatWireUsageEvent(from: decoder))
        case nil: self = .unknown(type)
        }
    }
}

struct ChatWireEnvelope: Decodable, Equatable, Sendable {
    let wireVersion: Int
    let turnID: String
    let seq: Int
    let event: ChatWireEvent

    var eventID: String { "\(turnID):\(seq)" }

    private enum CodingKeys: String, CodingKey {
        case wireVersion = "wire_version", turnID = "turn_id", seq, event
    }

    init(from decoder: Decoder) throws {
        try checkWireKeys(decoder, allowed: ["wire_version", "turn_id", "seq", "event"])
        let c = try decoder.container(keyedBy: CodingKeys.self)
        wireVersion = try c.value(Int.self, forKey: .wireVersion, default: 1)
        guard wireVersion == 1 else { throw ChatWireError.unsupportedVersion }
        turnID = try c.decode(String.self, forKey: .turnID)
        seq = try c.decode(Int.self, forKey: .seq)
        guard seq >= 1 else { throw ChatWireError.invalidField("seq") }
        event = try c.decode(ChatWireEvent.self, forKey: .event)
    }
}
