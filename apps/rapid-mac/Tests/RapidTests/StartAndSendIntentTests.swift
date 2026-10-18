import Foundation
import Testing
@testable import Rapid

@Suite("Start and send intent")
@MainActor
struct StartAndSendIntentTests {
    private let alias = "qwen3.5-9b-4bit"

    /// Exercise the same intent and draft APIs used by the view, without a
    /// network request. The submit spy stands at the normal send boundary.
    @MainActor
    private final class Composer {
        let model = ChatViewModel(persistsConversations: false)
        let alias = "qwen3.5-9b-4bit"
        var starts = 0
        var sent: [String] = []
        var hasAttachments = false

        func press() {
            if model.startAndSendIntent.begin(conversationID: model.activeConversationID, alias: alias) != nil {
                starts += 1
            }
        }

        @discardableResult
        func observe(_ readiness: ModelReadiness, streaming: Bool = false, importingFiles: Bool = false) -> StartAndSendIntent.Resolution {
            let result = model.startAndSendIntent.resolve(
                readiness: readiness, conversationID: model.activeConversationID,
                selectedAlias: alias, isStreaming: streaming, isImportingFiles: importingFiles
            )
            if case .send = result {
                model.submitComposerDraft(hasAttachments: hasAttachments, canSubmit: { true }) {
                    sent.append($0)
                }
            }
            return result
        }
    }

    @Test("Send starts once and ready sends the current draft once, including attachment-only drafts")
    func sendsCurrentDraftExactlyOnce() throws {
        for attachments in [false, true] {
            let composer = Composer()
            composer.model.composerDraft = "original"
            composer.press()
            let id = try #require(composer.model.startAndSendIntent.pending?.id)
            #expect(composer.observe(.needsStart(alias: alias)) == .wait)
            #expect(composer.observe(.starting(alias: alias, detail: nil)) == .wait)
            composer.press() // Return or button while a start is outstanding.
            #expect(composer.starts == 1)
            composer.model.composerDraft = attachments ? "" : "edited while starting"
            composer.hasAttachments = attachments
            composer.model.startAndSendIntent.finishStartAttempt(id, served: true)
            #expect(composer.observe(.ready(alias: alias)) == .send)
            #expect(composer.sent == [attachments ? "" : "edited while starting"])
            #expect(composer.model.composerDraft == "")
            #expect(composer.observe(.ready(alias: alias)) == .wait)
            composer.model.startAndSendIntent.finishStartAttempt(id, served: true)
            #expect(composer.observe(.ready(alias: alias)) == .wait)
            #expect(composer.sent.count == 1)
        }
    }

    @Test("A ready model waits for file imports, then sends once")
    func waitsForFileImports() throws {
        let composer = Composer()
        composer.model.composerDraft = "keep until files finish"
        composer.hasAttachments = true
        composer.press()
        let pending = try #require(composer.model.startAndSendIntent.pending)
        #expect(composer.observe(.ready(alias: alias), importingFiles: true) == .wait)
        #expect(composer.model.startAndSendIntent.pending == pending)
        #expect(composer.model.composerDraft == "keep until files finish")
        #expect(composer.sent.isEmpty)
        #expect(composer.observe(.ready(alias: alias), importingFiles: false) == .send)
        #expect(composer.model.startAndSendIntent.pending == nil)
        #expect(composer.sent == ["keep until files finish"])
        #expect(composer.observe(.ready(alias: alias)) == .wait)
        #expect(composer.sent.count == 1)
    }

    @Test("A failed start cancels without observations and a later ready model does not send")
    func failedStartWithoutObservations() throws {
        let composer = Composer()
        composer.model.composerDraft = "  keep exactly\n"
        composer.press()
        let id = try #require(composer.model.startAndSendIntent.pending?.id)
        // ChatView can be unmounted throughout startup and failure.
        composer.model.startAndSendIntent.finishStartAttempt(id, served: false)
        #expect(composer.model.startAndSendIntent.pending == nil)
        #expect(composer.model.composerDraft == "  keep exactly\n")
        // A later successful retry from Launch must require a new Send press.
        #expect(composer.observe(.ready(alias: alias)) == .wait)
        #expect(composer.sent.isEmpty)
        #expect(composer.model.composerDraft == "  keep exactly\n")
    }

    @Test("Failure and every unsupported readiness cancel without consuming the draft")
    func cancelsForUnsupportedReadiness() {
        let states: [ModelReadiness] = [
            .failed(alias: alias, message: "start failed", action: .start(alias: alias)),
            .noModel, .needsDownload(alias: alias, sizeText: nil), .engineMissing,
            .sharingCompute, .unknownModel(alias: alias),
            .downloading(alias: alias, detail: nil, fraction: nil),
            .starting(alias: "other", detail: nil), .ready(alias: "other"),
            .needsStart(alias: "other")
        ]
        for state in states {
            let composer = Composer()
            composer.model.composerDraft = "  keep exactly\n"
            composer.press()
            #expect(composer.observe(state) == .cancel)
            #expect(composer.observe(.ready(alias: alias)) == .wait)
            #expect(composer.sent.isEmpty)
            #expect(composer.model.composerDraft == "  keep exactly\n")
        }
    }

    @Test("A finished attempt still needing Start cancels, and late completion cannot finish a newer intent")
    func finishedWithoutStarting() throws {
        let composer = Composer()
        composer.model.composerDraft = "keep me"
        composer.press()
        let oldID = try #require(composer.model.startAndSendIntent.pending?.id)
        composer.model.startAndSendIntent.finishStartAttempt(oldID, served: true)
        #expect(composer.observe(.needsStart(alias: alias)) == .cancel)
        #expect(composer.observe(.ready(alias: alias)) == .wait)
        #expect(composer.sent.isEmpty)
        #expect(composer.model.composerDraft == "keep me")
        composer.press()
        composer.model.startAndSendIntent.finishStartAttempt(oldID, served: true)
        composer.model.startAndSendIntent.finishStartAttempt(oldID, served: false)
        #expect(composer.model.startAndSendIntent.pending?.startAttemptFinished == false)
        #expect(composer.observe(.needsStart(alias: alias)) == .wait)
    }

    @Test("Dismissing the real memory warning finishes ensureServing without starting or sending", .timeLimit(.minutes(1)))
    func memoryWarningDismissal() async throws {
        let composer = Composer()
        composer.model.composerDraft = "keep after cancel"
        composer.press()
        let id = try #require(composer.model.startAndSendIntent.pending?.id)
        let server = ServerManager(testingState: .idle, binaryPath: URL(fileURLWithPath: "/usr/bin/true"))
        server.memorySnapshotProvider = {
            .init(totalBytes: 32 * 1_073_741_824, usedBytes: 30 * 1_073_741_824)
        }
        let start = Task {
            let served = await server.ensureServing(alias: alias)
            composer.model.startAndSendIntent.finishStartAttempt(id, served: served)
            return served
        }
        while server.pendingMemoryWarning == nil { await Task.yield() }
        let warning = try #require(server.pendingMemoryWarning)
        #expect(composer.model.startAndSendIntent.pending?.startAttemptFinished == false)
        #expect(composer.observe(.needsStart(alias: alias)) == .wait)
        server.cancelPendingMemoryLoad(warning)
        #expect(await start.value == false)
        #expect(server.state == .idle)
        #expect(composer.model.startAndSendIntent.pending == nil)
        #expect(composer.observe(.needsStart(alias: alias)) == .wait)
        #expect(composer.observe(.ready(alias: alias)) == .wait)
        #expect(composer.sent.isEmpty)
        #expect(composer.model.composerDraft == "keep after cancel")
    }

    @Test("Switching conversations cancels permanently and preserves both drafts")
    func conversationSwitch() {
        let composer = Composer()
        // Existing conversation plus New Chat, with no disk persistence.
        composer.model.send("seed", alias: alias)
        composer.model.stopAndPersist()
        let first = composer.model.activeConversationID
        composer.model.composerDraft = "first draft"
        composer.model.newConversation()
        composer.model.composerDraft = "second draft"
        composer.model.selectConversation(first)
        composer.press()
        composer.model.newConversation()
        #expect(composer.model.startAndSendIntent.pending == nil)
        #expect(composer.model.composerDraft == "second draft")
        #expect(composer.observe(.ready(alias: alias)) == .wait)
        composer.model.selectConversation(first)
        #expect(composer.observe(.ready(alias: alias)) == .wait)
        #expect(composer.model.composerDraft == "first draft")
        #expect(composer.sent.isEmpty)
    }

    @Test("Context mismatch cancels; a ready model waits for streaming to stop")
    func contextAndStreaming() {
        let conversation = UUID()
        for changeConversation in [false, true] {
            var intent = StartAndSendIntent()
            intent.begin(conversationID: conversation, alias: alias)
            #expect(intent.resolve(
                readiness: .ready(alias: alias),
                conversationID: changeConversation ? UUID() : conversation,
                selectedAlias: changeConversation ? alias : "other", isStreaming: false,
                isImportingFiles: false
            ) == .cancel)
            #expect(intent.pending == nil)
        }
        let composer = Composer()
        composer.press()
        #expect(composer.observe(.ready(alias: alias), streaming: true) == .wait)
        // An emptied draft consumes the intent without submitting anything.
        #expect(composer.observe(.ready(alias: alias)) == .send)
        #expect(composer.model.startAndSendIntent.pending == nil)
        #expect(composer.sent.isEmpty)
    }

    @Test("Only needsStart offers Start & Send; other readiness gates and tooltips are unchanged")
    func predicateAndTooltip() {
        let states: [ModelReadiness] = [
            .noModel, .engineMissing, .sharingCompute,
            .needsDownload(alias: alias, sizeText: nil), .unknownModel(alias: alias),
            .downloading(alias: alias, detail: nil, fraction: nil),
            .starting(alias: alias, detail: nil), .ready(alias: alias),
            .failed(alias: alias, message: "failure", action: nil)
        ]
        for state in states {
            #expect(state.startAndSendAlias == nil)
            #expect(state.composerSendTooltip == state.sendTooltip)
        }
        let stopped = ModelReadiness.needsStart(alias: alias)
        #expect(stopped.startAndSendAlias == alias)
        #expect(!stopped.sendAllowed)
        #expect(stopped.composerSendTooltip == String(localized: "Start & Send"))
    }

    @Test("Composer uses the banner Start path and normal send; Retry/Edit stay ready-only")
    func viewWiring() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        func source(_ path: String) throws -> String {
            CapabilityChipRenderGateSourceGuardTests.stripCommentsAndWhitespace(
                try String(contentsOf: root.appendingPathComponent(path), encoding: .utf8)
            )
        }
        let view = try source("Sources/Rapid/UI/ChatView.swift")
        #expect(view.contains("guardviewModel.startAndSendIntent.pending==nilelse{return}"))
        #expect(view.contains("onReadinessAction(.start(alias:target))"))
        #expect(view.contains("ifcase.send=resolution{send()}"))
        #expect(view.contains("selectedAlias:alias,isStreaming:viewModel.isStreaming,isImportingFiles:attachmentDraft.isImportingFiles"))
        #expect(view.contains(".onChange(of:attachmentDraft.isImportingFiles){_,_inresolveStartAndSend()}"))
        #expect(view.contains(".onDisappear{viewModel.startAndSendIntent=StartAndSendIntent()}"))
        #expect(view.contains(".help(readiness.composerSendTooltip)"))
        #expect(view.contains("?String(localized:\"Sendmessage\"):readiness.composerSendTooltip)"))
        #expect(view.contains("retryEnabled:readiness.sendAllowed"))
        let content = try source("Sources/Rapid/UI/ContentView.swift")
        #expect(content.contains("case.start(lettarget):startModel(target)"))
        #expect(content.contains("letserved=awaitserveReadinessModel(target)"))
        #expect(content.contains("chat.startAndSendIntent.finishStartAttempt(intentID,served:served)"))
    }
}
