import Foundation
import Testing
@testable import Rapid

/// "The user said no" is not a malfunction.
///
/// The regression these lock down: a declined `browse` approval used to fall
/// through to ``FailureDiagnosis/Kind/toolFailed``, so clicking **Don't allow**
/// painted a red tool card reading *"The tool couldn't finish. Check its input,
/// then try again."* — the app reporting a fault, blaming the user's own input
/// for it, and offering to retry the very thing they had just refused.
///
/// Four properties are pinned here, and all four matter:
///   1. A decline classifies as ``FailureDiagnosis/Kind/userDeclined`` and
///      reads as an ordinary outcome (``FailureDiagnosis/Severity/notice``,
///      no action button).
///   2. The classification rides an explicit marker from the approval gate —
///      never a guess at the wording of an English sentence.
///   3. Only an actual "no" counts. A cancelled turn, or a prompt that could
///      not be shown, is NOT the user's decision and must not be attributed
///      to them.
///   4. A GENUINE browse failure still classifies as a failure. A fix that
///      quietly reclassified real errors as user choices would hide them.
@MainActor
@Suite("Declined tool diagnosis")
final class DeclinedToolDiagnosisTests {
    nonisolated(unsafe) private var createdSuiteNames: [String] = []
    deinit { TestDefaultsScope.cleanup(suiteNames: createdSuiteNames) }

    private func freshDefaults() -> UserDefaults {
        let name = TestDefaultsScope.mintSuiteName(prefix: "rapid-decline-test-")
        createdSuiteNames.append(name)
        let d = UserDefaults(suiteName: name)!
        d.removePersistentDomain(forName: name)
        return d
    }

    /// Memory-only so a test never reads or writes the real browse cache — and
    /// so the approval gate is actually reached (a fresh cache hit would serve
    /// the page without prompting at all).
    private func memoryOnlyCache() -> BrowseContentCache {
        BrowseContentCache(diskDirectory: nil)
    }

    // MARK: - Declining the initial fetch

    @Test("Clicking Don't allow on a browse prompt is a user decline, not a tool failure")
    func declinedBrowseIsNotAFailure() async throws {
        let approval = BrowseApprovalStore(defaults: freshDefaults())
        async let pending = BrowseTool.run(
            arguments: #"{"url":"https://example.com"}"#,
            approval: approval,
            cache: memoryOnlyCache()
        )

        // The gate suspends on the user; answer it the way the sheet's
        // "Don't allow" button does.
        try await waitForPendingApproval(approval)
        approval.answer(.deny)
        let result = await pending

        #expect(result.failureKind == .userDeclined)
        // The wire text stays factual and unchanged — the MODEL still has to be
        // told, in plain words, that it has no page and why.
        #expect(result.content == "browse error: the user did not approve browsing example.com")
        #expect(result.isError)
    }

    @Test("A declined browse keeps its user-declined kind through tool dispatch")
    func registryPreservesTheDeclinedKind() async throws {
        // ``BuiltinToolRegistry`` re-classifies every result at the dispatch
        // boundary. It must PREFER what the tool reported: re-deriving the kind
        // from the text here would put the decline straight back on the
        // toolFailed lane.
        let approval = BrowseApprovalStore(defaults: freshDefaults())
        let registry = BuiltinToolRegistry(
            browseApproval: approval,
            webSearch: WebSearchConfig(defaults: freshDefaults(), keychain: NoopKeychain())
        )
        async let pending = registry.run(
            ToolCall(id: "call_1", name: "browse", arguments: #"{"url":"https://example.com"}"#)
        )
        try await waitForPendingApproval(approval)
        approval.answer(.deny)
        let result = await pending

        #expect(result.failureKind == .userDeclined)
        #expect(result.toolCallID == "call_1")
    }

    // MARK: - Only an actual "no" is a decline

    @Test("A turn cancelled while the prompt is up is not blamed on the user")
    func cancellationIsNotADecline() async throws {
        // Pressing Stop tears the approval sheet down. The store used to report
        // that as ``.deny``, which would now read back to the user as "you
        // didn't allow this" — a decision they never made.
        let approval = BrowseApprovalStore(defaults: freshDefaults())
        let cache = memoryOnlyCache()
        let task = Task {
            await BrowseTool.run(
                arguments: #"{"url":"https://example.com"}"#,
                approval: approval,
                cache: cache
            )
        }
        try await waitForPendingApproval(approval)
        task.cancel()
        let result = await task.value

        #expect(result.failureKind != .userDeclined)
        #expect(result.isError)
    }

    @Test("A second prompt while one is already open is refused, not counted as a decline")
    func reentrantApprovalIsNotADecline() async throws {
        // Tool execution is serial, so a second pending prompt means something
        // is wrong. The user never saw this URL, so the answer cannot be theirs.
        let approval = BrowseApprovalStore(defaults: freshDefaults())
        async let first = approval.requestApproval(url: "https://example.com", host: "example.com")
        try await waitForPendingApproval(approval)

        let second = await approval.requestApproval(url: "https://other.example", host: "other.example")
        #expect(second == .unavailable)

        approval.answer(.deny)
        #expect(await first == .deny)
    }

    @Test("Auto-approve still lets a fetch through untouched")
    func autoApproveIsUnaffected() async {
        let approval = BrowseApprovalStore(defaults: freshDefaults())
        approval.mode = .autoApproveAll
        let decision = await approval.requestApproval(url: "https://example.com", host: "example.com")
        #expect(decision == .allowOnce)
    }

    @Test("Always allow approves the pending fetch and persists auto-approval")
    func alwaysAllowPersistsAndSkipsFuturePrompts() async throws {
        let defaults = freshDefaults()
        let approval = BrowseApprovalStore(defaults: defaults)

        async let pending = approval.requestApproval(
            url: "https://example.com/article",
            host: "example.com"
        )
        try await waitForPendingApproval(approval)
        approval.alwaysAllow()

        #expect(await pending == .allowOnce)
        #expect(approval.pendingRequest == nil)
        #expect(approval.mode == .autoApproveAll)

        let restored = BrowseApprovalStore(defaults: defaults)
        #expect(restored.mode == .autoApproveAll)
        let next = await restored.requestApproval(
            url: "https://other.example/page",
            host: "other.example"
        )
        #expect(next == .allowOnce)
        #expect(restored.pendingRequest == nil)
    }

    // MARK: - Declining a cross-origin redirect

    @Test("Declining a redirect to another host is a user decline too")
    func declinedRedirectIsNotAFailure() throws {
        // ``redirectGateError`` is the exact function the fetch loop calls at
        // the redirect gate. It is reachable directly only because it is pure:
        // driving the loop itself would need a live redirecting server, and the
        // SSRF guard rightly refuses the loopback address a local one needs.
        let destination = try #require(URL(string: "https://elsewhere.example/page"))
        let thrown = try #require(BrowseTool.redirectGateError(.deny, destination: destination))
        #expect(thrown is BrowseTool.ApprovalDeclined)

        let result = BrowseTool.errorResult(tool: "browse", error: thrown)
        #expect(result.failureKind == .userDeclined)
        #expect(result.content ==
                "browse error: the user did not approve the redirect to elsewhere.example")
    }

    @Test("An allowed redirect does not end the fetch")
    func allowedRedirectContinues() throws {
        let destination = try #require(URL(string: "https://elsewhere.example/page"))
        #expect(BrowseTool.redirectGateError(.allowOnce, destination: destination) == nil)
    }

    @Test("A redirect prompt that could not be shown is a failure, not a decline")
    func unavailableRedirectPromptIsAFailure() throws {
        let destination = try #require(URL(string: "https://elsewhere.example/page"))
        let thrown = try #require(BrowseTool.redirectGateError(.unavailable, destination: destination))
        #expect(!(thrown is BrowseTool.ApprovalDeclined))

        let result = BrowseTool.errorResult(tool: "browse", error: thrown)
        #expect(result.failureKind == nil)   // left to the dispatch-boundary classifier
        #expect(result.isError)
    }

    // MARK: - Genuine failures must STILL be failures

    @Test("A real fetch error is still a tool failure, not a user decline")
    func genuineFetchErrorStaysAFailure() {
        // Same seam, an ordinary error: nothing about the decline lane may
        // swallow a fault the user needs to see.
        let result = BrowseTool.errorResult(
            tool: "browse",
            error: NSError(
                domain: "RapidBrowse",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "HTTP 503 from example.com"]
            )
        )
        #expect(result.failureKind == nil)
        #expect(result.isError)

        let kind = FailureDiagnoser.toolFailureKind(
            toolName: "browse",
            content: result.content,
            isError: result.isError
        )
        #expect(kind == .toolFailed)
        #expect(FailureDiagnoser.diagnosis(for: .toolFailed).severity == .error)
    }

    @Test("A browse call the model malformed is still a tool failure end to end")
    func malformedArgumentsStayAFailure() async {
        let registry = BuiltinToolRegistry(
            browseApproval: BrowseApprovalStore(defaults: freshDefaults()),
            webSearch: WebSearchConfig(defaults: freshDefaults(), keychain: NoopKeychain())
        )
        // Never reaches the approval gate — it fails on its own merits.
        let result = await registry.run(
            ToolCall(id: "call_2", name: "browse", arguments: "not json")
        )
        #expect(result.isError)
        #expect(result.failureKind == .toolFailed)
        #expect(result.failureKind?.severity == .error)
    }

    @Test("An oversized browse page gets specific recovery copy")
    func oversizedBrowsePageIsSpecific() throws {
        let kind = FailureDiagnoser.toolFailureKind(
            toolName: "browse",
            content: "browse error: page exceeded 2 MB cap",
            isError: true
        )
        #expect(kind == .browsePageTooLarge)
        let diagnosis = FailureDiagnoser.diagnosis(for: try #require(kind))
        #expect(diagnosis.message ==
                "This page is too large for Rapid to read at once. Search it or open a smaller page instead.")
        #expect(diagnosis.action == nil)
    }

    @Test("Text that merely LOOKS like a decline is not promoted to one")
    func declineIsNeverInferredFromWording() {
        // The marker is carried by the approval gate, so a page (or a future
        // tool wording its errors differently) cannot talk its way onto the
        // quiet lane by containing the sentence.
        let kind = FailureDiagnoser.toolFailureKind(
            toolName: "browse",
            content: "browse error: the user did not approve browsing example.com",
            isError: true
        )
        #expect(kind == .toolFailed)
    }

    // MARK: - The copy the user actually reads

    @Test("A decline reads as an outcome, with no action button and no blame")
    func declinedCopyIsNeutral() {
        let diagnosis = FailureDiagnoser.diagnosis(for: .userDeclined)
        #expect(diagnosis.message ==
                "You didn't allow this, so there's nothing to show. Ask again if you change your mind.")
        // No "check its input" (the input was fine), and no retry button
        // offering to re-run what the user just refused.
        #expect(diagnosis.action == nil)
        #expect(diagnosis.severity == .notice)
    }

    /// The quiet lane is exactly the outcomes the USER chose, and nothing else.
    ///
    /// Enumerated rather than derived so adding a kind cannot join the quiet
    /// lane by accident: a new case lands on ``.error`` and fails here until
    /// somebody states which lane it belongs on and why. Membership is a strong
    /// claim — it removes the alarm from something that went wrong.
    ///
    /// ``downloadCancelled`` joined in the onboarding-recovery slice. It is the
    /// user stopping a transfer they started, which is the same shape as
    /// declining a permission prompt: nothing malfunctioned, so nothing should
    /// be painted as though it had.
    private static let noticeKinds: Set<FailureDiagnosis.Kind> = [
        .userDeclined,
        .downloadCancelled,
    ]

    @Test("Only the user's own choices are notices; every other kind stays an error")
    func onlyUserChoicesAreNotices() {
        for kind in FailureDiagnosis.Kind.allCases {
            let expected: FailureDiagnosis.Severity =
                Self.noticeKinds.contains(kind) ? .notice : .error
            #expect(kind.severity == expected, "unexpected severity for \(kind.rawValue)")
            #expect(FailureDiagnoser.diagnosis(for: kind).severity == expected)
        }
    }

    @Test("A broken download stays an error even though a cancelled one does not")
    func cancellationDoesNotDragTheFailureLaneWithIt() {
        // The pair the split exists for. If these ever collapse back onto one
        // severity, the copy has collapsed with them.
        #expect(FailureDiagnosis.Kind.downloadCancelled.severity == .notice)
        #expect(FailureDiagnosis.Kind.downloadFailed.severity == .error)
        #expect(FailureDiagnosis.Kind.downloadSourceUnavailable.severity == .error)
    }

    @Test("A throttled backend is something that went wrong, not something the user chose")
    func rateLimitStaysOnTheErrorLane() {
        // #1580's kind arrived alongside this change. A rate limit is the world
        // refusing, not the user — it keeps the red card and its deep-link.
        let diagnosis = FailureDiagnoser.diagnosis(for: .webSearchRateLimited)
        #expect(diagnosis.severity == .error)
        #expect(diagnosis.action == .openWebSearchSettings)
    }

    @Test("A notice never grows an inline button on the tool card")
    func noticeGetsNoInlineToolCardAction() {
        // One rule decides what the card offers. Gating on severity rather than
        // on `action == nil` means a notice that ever gains an action still
        // can't sprout a button here — there is nothing for the app to recover
        // from a decision the user made on purpose.
        #expect(FailureDiagnosis.inlineToolCardAction(
            for: FailureDiagnoser.diagnosis(for: .userDeclined),
            canRouteToSettings: true
        ) == nil)
        // The error lane is untouched: #1580's deep-link still renders.
        #expect(FailureDiagnosis.inlineToolCardAction(
            for: FailureDiagnoser.diagnosis(for: .webSearchRateLimited),
            canRouteToSettings: true
        ) == .openWebSearchSettings)
    }

    // MARK: - Persistence

    @Test("A decline this build wrote is still a decline when the transcript reopens")
    func declinedKindSurvivesAnEncodeDecodeRound() throws {
        // Covers rows written from here on. Declines saved by OLDER builds were
        // persisted as ``.toolFailed`` and are read back as such by design —
        // there is nothing in those rows to tell them apart from a real fault.
        let data = try JSONEncoder().encode(declinedToolRow())
        let restored = try JSONDecoder().decode(ChatMessage.self, from: data)
        #expect(restored.failureKind == .userDeclined)
        let diagnosis = restored.toolFailureDiagnosis(toolName: "browse")
        #expect(diagnosis.kind == .userDeclined)
        #expect(diagnosis.severity == .notice)
        #expect(diagnosis.action == nil)
    }

    @Test("The original key keeps a value older builds can decode")
    func newKindIsNotWrittenWhereOldBuildsReadStrictly() throws {
        // ``FailureDiagnosis.Kind`` decodes strictly in every shipped build,
        // and ``ConversationStore.load`` turns ONE undecodable message into
        // "the whole history is corrupt" — so a downgrade would greet the user
        // with an empty sidebar. The new raw value therefore lives only under
        // a key older builds ignore.
        let data = try JSONEncoder().encode(declinedToolRow())
        let object = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        #expect(object["failureKind"] as? String == "toolFailed")
        #expect(object["failureKindV2"] as? String == "userDeclined")
    }

    @Test("Every kind added after the last release keeps old builds readable")
    func kindsNewerThanTheLastReleaseAllHaveAKnownAncestor() throws {
        // `rapid-mac-v0.12.5` — the newest shipped tag — has neither of these
        // raw values and decodes `failureKind` strictly, so BOTH carry the same
        // hazard: one unknown value sides the entire history file. The encoder
        // covers any kind automatically once ``legacyPersistedKind`` names an
        // ancestor, so what needs pinning is the mapping.
        let shippedInV0_12_5: Set<FailureDiagnosis.Kind> = [
            .modelOutOfMemory, .modelLoadFailed, .engineNotRunning,
            .webSearchOffline, .webSearchUnavailable,
            .commandPermissionDenied, .commandFailed,
            .fileNotFound, .filePermissionDenied, .toolFailed,
            .downloadFailed, .downloadSourceUnavailable, .requestFailed,
        ]
        for kind in FailureDiagnosis.Kind.allCases {
            #expect(
                shippedInV0_12_5.contains(kind.legacyPersistedKind),
                "\(kind.rawValue) maps to \(kind.legacyPersistedKind.rawValue), which no shipped build can decode"
            )
        }
        #expect(FailureDiagnosis.Kind.userDeclined.legacyPersistedKind == .toolFailed)
        // The condition this kind describes used to land on webSearchUnavailable,
        // so that is what an older build should keep showing for it.
        #expect(FailureDiagnosis.Kind.webSearchRateLimited.legacyPersistedKind == .webSearchUnavailable)
        #expect(FailureDiagnosis.Kind.browsePageTooLarge.legacyPersistedKind == .toolFailed)
    }

    @Test("A rate-limited row is downgrade-safe and reopens as itself")
    func rateLimitedRowUsesBothKeys() throws {
        let row = ChatMessage(
            role: .tool,
            content: "web_search error: DuckDuckGo is throttling this Mac",
            status: .failed,
            failureKind: .webSearchRateLimited,
            toolCallID: "call_1"
        )
        let object = try encodedObject(for: row)
        #expect(object["failureKind"] as? String == "webSearchUnavailable")
        #expect(object["failureKindV2"] as? String == "webSearchRateLimited")

        let data = try JSONEncoder().encode(row)
        let restored = try JSONDecoder().decode(ChatMessage.self, from: data)
        #expect(restored.failureKind == .webSearchRateLimited)
    }

    @Test("A row written before the v2 key existed still decodes")
    func legacyOnlyRowStillDecodes() throws {
        var object = try encodedObject(for: ChatMessage(
            role: .tool,
            content: "read_file error: no such file",
            status: .failed,
            failureKind: .fileNotFound,
            toolCallID: "call_1"
        ))
        object.removeValue(forKey: "failureKindV2")

        let data = try JSONSerialization.data(withJSONObject: object)
        let restored = try JSONDecoder().decode(ChatMessage.self, from: data)
        #expect(restored.failureKind == .fileNotFound)
    }

    @Test("A kind from a NEWER build degrades instead of wiping the history")
    func unknownKindDegradesRatherThanThrowing() throws {
        // The mirror image of the downgrade guard above: this build must also
        // survive a transcript written by a future one.
        var object = try encodedObject(for: declinedToolRow())
        object["failureKindV2"] = "someKindFromTheFuture"

        let data = try JSONSerialization.data(withJSONObject: object)
        let restored = try JSONDecoder().decode(ChatMessage.self, from: data)
        #expect(restored.failureKind == .toolFailed)
    }

    @Test("An unknown future kind degrades to the ancestor that build named, not a generic one")
    func unknownKindPrefersTheLegacyValueThatBuildWrote() throws {
        // A future build writes its closest KNOWN ancestor into the original
        // key precisely so an older build lands somewhere sensible. Degrading
        // to a blanket ``.toolFailed`` instead would throw that away and, for
        // e.g. a download-shaped kind, show the wrong recovery action.
        var object = try encodedObject(for: ChatMessage(
            role: .assistant,
            status: .failed,
            failureKind: .downloadSourceUnavailable
        ))
        object["failureKindV2"] = "someFutureDownloadKind"

        let data = try JSONSerialization.data(withJSONObject: object)
        let restored = try JSONDecoder().decode(ChatMessage.self, from: data)
        #expect(restored.failureKind == .downloadSourceUnavailable)
    }

    @Test("The hand-written encoder still round-trips every field")
    func fullMessageRoundTripsUnchanged() throws {
        // ``encode(to:)`` is hand-written for the two-key failure kind, so a
        // dropped line would silently lose part of every saved transcript.
        let original = ChatMessage(
            id: UUID(),
            role: .assistant,
            content: "body",
            reasoning: "trace",
            status: .failed,
            errorMessage: "something to show",
            failureKind: .userDeclined,
            toolCalls: [ToolCall(id: "call_1", name: "browse", arguments: #"{"url":"https://example.com"}"#)],
            toolCallID: "call_1",
            stats: MessageStats(
                elapsedSeconds: 2.5,
                charCount: 128,
                promptTokens: 42,
                completionTokens: 7
            ),
            reasoningTruncated: true,
            contentTruncated: true,
            toolNotCalledFlagged: true,
            toolCallArtifactSuppressed: true,
            createdAt: Date(timeIntervalSinceReferenceDate: 700_000_000)
        )
        let data = try JSONEncoder().encode(original)
        let restored = try JSONDecoder().decode(ChatMessage.self, from: data)
        #expect(restored == original)
    }

    // MARK: - Helpers

    private func declinedToolRow() -> ChatMessage {
        ChatMessage(
            role: .tool,
            content: "browse error: the user did not approve browsing example.com",
            status: .failed,
            errorMessage: FailureDiagnoser.diagnosis(for: .userDeclined).message,
            failureKind: .userDeclined,
            toolCallID: "call_1"
        )
    }

    private func encodedObject(for message: ChatMessage) throws -> [String: Any] {
        let data = try JSONEncoder().encode(message)
        return try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    /// Suspend until the approval gate has published its prompt, so the test
    /// answers a request that actually exists. Bounded by wall-clock rather
    /// than by an iteration count — a loaded machine must not fail correct
    /// code — and it still fails rather than hanging if the prompt never comes.
    private func waitForPendingApproval(
        _ approval: BrowseApprovalStore,
        timeout: Duration = .seconds(10)
    ) async throws {
        let deadline = ContinuousClock.now.advanced(by: timeout)
        while approval.pendingRequest == nil {
            if ContinuousClock.now >= deadline { throw ApprovalNeverArrived() }
            try await Task.sleep(for: .milliseconds(1))
        }
    }

    private struct ApprovalNeverArrived: Error {}
}

/// Keychain double: the web-search key is irrelevant here, and the real
/// keychain would prompt.
private final class NoopKeychain: KeychainStoring, @unchecked Sendable {
    func read(account: String) -> String? { nil }
    func write(account: String, secret: String) -> Bool { true }
    func delete(account: String) -> Bool { true }
}
