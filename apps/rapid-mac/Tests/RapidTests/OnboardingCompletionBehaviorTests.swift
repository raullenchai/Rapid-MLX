import Foundation
import Testing
@testable import Rapid

/// Contract for the Onboarding V3 prerequisite behaviour change
/// (Paper §05.1.G — "Four public steps, and Ready is confirmed",
/// "Readiness does not dismiss setup", "Completion is what persists,
/// not readiness").
///
/// Three things changed, and each of them is a way the app could quietly
/// regress back to the ending this PR retires:
///
///   1. **Four public steps.** The progress model reports Welcome /
///      Choose a model / Download / Start. Micro-states collapse into
///      their macro step and a failure keeps the step that owns it, so
///      nothing can quietly become a fifth step.
///   2. **Ready is a destination.** Readiness parks onboarding on a
///      stable screen. It writes no completion flag and dismisses
///      nothing. The superseded behaviour — dismiss the moment a
///      subprocess reports a listening port — is pinned as forbidden
///      rather than merely unimplemented.
///   3. **Completion is confirmed and idempotent.** Start chatting runs
///      one transaction: seed once, persist once, retire the pending
///      record, release the surface. Repeated readiness notifications
///      and repeated activations are both no-ops.
///
/// The persistence half is deliberately paranoid: a stored alias is a
/// claim about a previous launch, never evidence about this one.
@MainActor
@Suite("Onboarding V3 — four steps, persistent Ready, confirmed completion")
struct OnboardingCompletionBehaviorTests {

    /// A coordinator with every persisted key cleared, so a case never
    /// inherits another case's flags through ``UserDefaults``.
    private func makeCoordinator() -> QuickstartCoordinator {
        let coord = QuickstartCoordinator()
        coord._testingReset()
        return coord
    }

    /// Drop every key this suite can write, including the ones a
    /// deliberately "relaunched" coordinator leaves behind.
    private func clearPersistedState() {
        let coord = QuickstartCoordinator()
        coord._testingReset()
    }

    // MARK: - 1. Four public steps

    @Test("There are exactly four public steps, in order")
    func fourPublicSteps() {
        #expect(QuickstartCoordinator.Step.total == 4)
        #expect(QuickstartCoordinator.Step.allCases == [
            .welcome, .chooseModel, .download, .start
        ])
        #expect(QuickstartCoordinator.Step.welcome.displayNumber == 1)
        #expect(QuickstartCoordinator.Step.chooseModel.displayNumber == 2)
        #expect(QuickstartCoordinator.Step.download.displayNumber == 3)
        #expect(QuickstartCoordinator.Step.start.displayNumber == 4)
    }

    @Test("Welcome is Step 1 of 4; the chooser is Step 2 of 4")
    func welcomeAndChooserSteps() {
        let coord = makeCoordinator()
        #expect(coord.step == .welcome)
        #expect(coord.step.displayNumber == 1)

        coord.advanceToChooseModel()
        #expect(coord.step == .chooseModel)
        #expect(coord.step.displayNumber == 2)

        coord.backToWelcome()
        #expect(coord.step == .welcome)
    }

    @Test("Download is Step 3 — including its low-disk and failure micro-states")
    func downloadIsStepThree() {
        let coord = makeCoordinator()
        coord.enterDownloading()
        #expect(coord.step == .download)
        #expect(coord.step.displayNumber == 3)

        // Insufficient disk is a question about the pull the user just
        // authorised, not a step of its own.
        coord.cancelLowDiskWarning()
        coord.enterLowDiskWarning(freeBytes: 1_000, requiredBytes: 5_000)
        #expect(coord.step == .download)

        // A broken pull must not move the rail — the user is still in
        // Download, which is exactly where Retry puts them back.
        coord.enterFailed(message: "network unreachable", origin: .download)
        #expect(coord.step == .download)
        #expect(coord.step.displayNumber == 3)
    }

    @Test("Starting and Ready are both Step 4 — including a load failure")
    func startingAndReadyAreStepFour() {
        let coord = makeCoordinator()
        coord.enterStarting()
        #expect(coord.step == .start)
        #expect(coord.step.displayNumber == 4)

        coord.enterReady()
        #expect(coord.step == .start)
        #expect(coord.step.displayNumber == 4)

        // The weights are on disk; only the load failed. Sending the user
        // back to Step 3 would imply the download needs redoing.
        coord.enterFailed(message: "could not load", origin: .start)
        #expect(coord.step == .start)
        #expect(coord.step.displayNumber == 4)
    }

    @Test("Every phase maps into the four steps — no failure becomes a fifth")
    func noPhaseEscapesTheFourSteps() {
        let phases: [QuickstartCoordinator.Phase] = [
            .idle,
            .lowDiskWarning(freeBytes: 1, requiredBytes: 2),
            .downloading,
            .starting,
            .ready,
            .dismissed,
            .failed(message: "x", origin: .download),
            .failed(message: "x", origin: .start),
        ]
        for phase in phases {
            for stage in [QuickstartCoordinator.Stage.welcome, .chooseModel] {
                let step = QuickstartCoordinator.step(phase: phase, stage: stage)
                #expect(
                    QuickstartCoordinator.Step.allCases.contains(step),
                    "phase \(phase) escaped the four-step model"
                )
                #expect(step.displayNumber >= 1 && step.displayNumber <= 4)
            }
        }
    }

    @Test("No production surface still says 'of 3'")
    func noThreeStepLanguageSurvives() throws {
        // The step count now lives in exactly one constant, so a
        // regression would have to reintroduce a literal. Grep for the
        // shapes that could: an explicit `total: 3` argument, or a
        // hard-coded three-step progress row.
        for file in ["Sources/Rapid/UI/OnboardingComponents.swift",
                     "Sources/Rapid/UI/QuickstartView.swift"] {
            let body = try Self.strippedSource(file)
            #expect(
                !body.contains("total:3"),
                "\(file) hard-codes a step total; it must read QuickstartCoordinator.Step.total"
            )
        }
    }

    // MARK: - 2. Readiness parks; it does not complete or dismiss

    @Test("Readiness alone does NOT mark onboarding complete")
    func readinessDoesNotComplete() {
        let coord = makeCoordinator()
        coord.enterDownloading()
        coord.enterStarting()
        coord.enterReady()

        #expect(coord.phase == .ready)
        #expect(!coord.done, "readiness must not write the completion flag")
        #expect(
            !UserDefaults.standard.bool(forKey: QuickstartCoordinator.storageKey),
            "readiness must not persist completion"
        )
        #expect(!coord.hasSeededWelcome, "readiness must not seed the welcome message")
    }

    @Test("Readiness alone does NOT dismiss the onboarding surface")
    func readinessDoesNotDismiss() {
        // The superseded ending (Paper: "must not be re-introduced") let
        // readiness release the window. The surface-retention rule is the
        // single gate that decides, so pin it across every phase.
        #expect(ContentView.quickstartRetainsSurface(phase: .ready),
                "Ready must keep the full-window onboarding surface up")
        #expect(ContentView.quickstartRetainsSurface(phase: .downloading))
        #expect(ContentView.quickstartRetainsSurface(phase: .starting))
        #expect(ContentView.quickstartRetainsSurface(
            phase: .failed(message: "x", origin: .download)
        ))
        #expect(ContentView.quickstartRetainsSurface(
            phase: .lowDiskWarning(freeBytes: 1, requiredBytes: 2)
        ))
        // Only the terminal states hand the window back.
        #expect(!ContentView.quickstartRetainsSurface(phase: .dismissed))
        #expect(!ContentView.quickstartRetainsSurface(phase: .idle))
    }

    @Test("Repeated readiness notifications are harmless")
    func repeatedReadinessIsIdempotent() {
        let coord = makeCoordinator()
        coord.enterStarting()

        // Auto-respawn cycle, residency tick, scheduler re-publish: the
        // same serve can announce itself many times.
        for _ in 0..<5 { coord.enterReady() }

        #expect(coord.phase == .ready)
        #expect(!coord.done)
        #expect(coord.pendingReadyAlias == coord.selection.alias)

        // And a late notification after the user has already finished must
        // not drag them back onto a screen they dismissed.
        var seeds = 0
        _ = coord.confirmStartChatting { seeds += 1; return true }
        #expect(coord.phase == .dismissed)
        coord.enterReady()
        #expect(coord.phase == .dismissed, "a late readiness tick must not resurrect Ready")
        #expect(seeds == 1)
    }

    // MARK: - 3. Start chatting is one idempotent transaction

    @Test("Start chatting completes exactly once and seeds exactly once")
    func startChattingCompletesOnce() {
        let coord = makeCoordinator()
        coord.enterStarting()
        coord.enterReady()

        var seeds = 0
        var transitions = 0
        func activate() {
            if coord.confirmStartChatting(seedWelcome: { seeds += 1; return true }) {
                transitions += 1
            }
        }

        // A double-click, a Return that repeats, an accessibility
        // double-activation: all reach the same entry point.
        activate()
        activate()
        activate()

        #expect(transitions == 1, "only the first activation may run the transition")
        #expect(seeds == 1, "the welcome message must be seeded exactly once")
        #expect(coord.done)
        #expect(coord.hasSeededWelcome)
        #expect(coord.phase == .dismissed)
    }

    @Test("Start chatting before Ready is a no-op")
    func startChattingRequiresReady() {
        let coord = makeCoordinator()
        var seeds = 0
        for phase in ["idle", "downloading", "starting"] {
            switch phase {
            case "downloading": coord.enterDownloading()
            case "starting":    coord.enterStarting()
            default:            break
            }
            #expect(!coord.confirmStartChatting { seeds += 1; return true })
        }
        #expect(seeds == 0)
        #expect(!coord.done, "nothing outside Ready may complete onboarding")
    }

    @Test("Completion writes the persistent flag and clears pending provenance")
    func completionClearsPendingProvenance() {
        let coord = makeCoordinator()
        coord.enterStarting()
        coord.enterReady()
        #expect(coord.hasPendingReady, "Ready must record that a confirmation is owed")

        _ = coord.confirmStartChatting { true }

        #expect(coord.done)
        #expect(UserDefaults.standard.bool(forKey: QuickstartCoordinator.storageKey))
        #expect(!coord.hasPendingReady, "a confirmed flow owes no confirmation")
        #expect(
            UserDefaults.standard.string(forKey: QuickstartCoordinator.pendingReadyAliasKey) == nil,
            "the pending record must be erased from disk, not just from memory"
        )
        clearPersistedState()
    }

    // MARK: - 4. Relaunch and persistence

    @Test("Pending Ready survives a coordinator reconstruction")
    func pendingReadySurvivesReconstruction() {
        let coord = makeCoordinator()
        coord.enterStarting()
        coord.enterReady()
        #expect(coord.hasPendingReady)

        // Same shape as a relaunch, and also as a SwiftUI re-mount that
        // rebuilds the coordinator: the record must come from disk.
        let next = QuickstartCoordinator()
        #expect(next.hasPendingReady, "an unconfirmed Ready flow must survive")
        #expect(next.pendingReadyAlias == coord.selection.alias)
        #expect(next.selection.alias == coord.selection.alias,
                "the restored flow must be about the same model")
        #expect(!next.done, "an unconfirmed flow is not a completed one")
        clearPersistedState()
    }

    @Test("Relaunch does NOT fabricate Ready from the stored alias alone")
    func relaunchDoesNotFabricateReady() {
        let coord = makeCoordinator()
        let alias = coord.selection.alias
        coord.enterStarting()
        coord.enterReady()

        let next = QuickstartCoordinator()
        // The claim survived; the STATE did not. Nothing on this launch has
        // yet said the model is up, so claiming Ready here would be the
        // app inventing a readiness it has not observed.
        #expect(next.phase == .idle, "a restored flow must not start in Ready")
        #expect(next.phase != .ready)
        #expect(next.hasPendingReady)
        // It lands on the chooser with the pick restored — the ordinary
        // setup path — rather than re-asking "would you like to begin?".
        #expect(next.stage == .chooseModel)
        #expect(next.selection.alias == alias)
        clearPersistedState()
    }

    @Test("Relaunch with a genuinely ready model returns to the Ready screen")
    func relaunchWithReadyModelReturnsToReady() {
        let coord = makeCoordinator()
        coord.enterStarting()
        coord.enterReady()
        let alias = coord.selection.alias

        let next = QuickstartCoordinator()
        #expect(next.phase == .idle)
        // This is the step the live server observer performs once
        // ``ServerManager`` genuinely reports ``.ready`` for this alias on
        // THIS launch — the evidence the stored record deliberately lacks.
        next.enterReady()
        #expect(next.phase == .ready, "a genuinely ready model returns to the confirmation screen")
        #expect(next.selection.alias == alias)
        #expect(!next.done, "returning to Ready must still not complete anything")
        clearPersistedState()
    }

    @Test("Re-entering Ready is gated on the SERVED alias, not the stored one")
    func readyReEntryIsGatedOnServedAlias() throws {
        // The guard that stops a stored alias becoming a fabricated ready
        // state lives in the view's server observer. Pin its shape: the
        // ready branch must compare the served alias against the live
        // selection before calling ``enterReady``.
        let body = try Self.strippedSource("Sources/Rapid/UI/QuickstartView.swift")
        #expect(
            body.contains("ifcase.ready(letalias)=server.state,alias==coordinator.selection.alias{coordinator.enterReady()"),
            """
            QuickstartView no longer enters Ready by comparing the SERVED \
            alias against the current selection. Re-entering Ready from a \
            stored alias alone would let onboarding claim a model is up \
            without this launch ever observing it.
            """
        )
    }

    @Test("Changing to another model clears stale pending provenance")
    func changingModelClearsPendingProvenance() {
        let coord = makeCoordinator()
        coord.enterStarting()
        coord.enterReady()
        #expect(coord.hasPendingReady)

        // Back out to the chooser and pick something else. The Ready record
        // was about the OLD model; keeping it would offer to confirm a flow
        // the user has walked away from.
        coord.returnToChooser()
        let other = QuickstartCoordinator.onboardingChoices.first {
            $0.alias != coord.selection.alias
        }
        #expect(other != nil)
        guard let other else { return }
        coord.select(other)

        #expect(coord.selection.alias == other.alias)
        #expect(!coord.hasPendingReady, "a different model retires the old Ready record")
        #expect(
            UserDefaults.standard.string(forKey: QuickstartCoordinator.pendingReadyAliasKey) == nil
        )
        clearPersistedState()
    }

    @Test("Re-selecting the SAME model keeps the pending record")
    func reselectingSameModelKeepsProvenance() {
        let coord = makeCoordinator()
        coord.enterStarting()
        coord.enterReady()
        let same = coord.selection
        coord.returnToChooser()
        coord.select(same)
        #expect(coord.hasPendingReady, "re-affirming the same pick is not a change of mind")
        clearPersistedState()
    }

    @Test("Skip retires the pending record without claiming completion")
    func skipClearsPendingWithoutCompleting() {
        let coord = makeCoordinator()
        coord.enterStarting()
        coord.enterReady()
        #expect(coord.hasPendingReady)

        coord.skipForNow()

        #expect(!coord.hasPendingReady, "walking away answers the question Ready was asking")
        #expect(!coord.done, "Skip must keep its existing meaning — onboarding is still owed")
        #expect(
            !UserDefaults.standard.bool(forKey: QuickstartCoordinator.storageKey)
        )
        clearPersistedState()
    }

    @Test("A fresh download retires a stale pending record")
    func freshDownloadClearsPendingProvenance() {
        let coord = makeCoordinator()
        coord.enterStarting()
        coord.enterReady()
        #expect(coord.hasPendingReady)
        coord.returnToChooser()
        coord.enterDownloading()
        #expect(!coord.hasPendingReady, "a fresh pull is a fresh flow")
        clearPersistedState()
    }

    // MARK: - 5. Welcome seeding

    @Test("The welcome message lands in the transcript exactly once")
    func welcomeSeedsExactlyOnce() {
        let chat = ChatViewModel(persistsConversations: false)
        #expect(chat.messages.isEmpty)

        #expect(chat.seedAssistantWelcome("Welcome aboard."))
        #expect(chat.messages.count == 1)
        #expect(chat.messages.first?.role == .assistant)
        #expect(chat.messages.first?.content == "Welcome aboard.")
        #expect(chat.messages.first?.status == .complete,
                "a locally authored welcome must never render as a live stream")

        // A second attempt must not append. In production the coordinator
        // already guarantees one call; this is the defence in depth that
        // makes a stray second call harmless rather than duplicating.
        #expect(!chat.seedAssistantWelcome("Welcome aboard."))
        #expect(chat.messages.count == 1)
    }

    @Test("Seeding never interrupts a conversation that already has content")
    func welcomeNeverInterruptsAnExistingChat() {
        let chat = ChatViewModel(persistsConversations: false)
        chat.devSeedMessages([ChatMessage(role: .user, content: "hello")])
        #expect(!chat.seedAssistantWelcome("Welcome aboard."))
        #expect(chat.messages.count == 1, "no intro may be injected into somebody's chat")
    }

    @Test("Empty welcome copy is refused rather than appended blank")
    func emptyWelcomeIsRefused() {
        let chat = ChatViewModel(persistsConversations: false)
        #expect(!chat.seedAssistantWelcome("   \n  "))
        #expect(chat.messages.isEmpty)
    }

    // MARK: - 6. The completion transaction's wiring

    @Test("Ready renders a Start chatting action with stable identifiers")
    func readyRendersConfirmationAction() throws {
        let body = try Self.strippedSource("Sources/Rapid/UI/QuickstartView.swift")
        #expect(body.contains(#".accessibilityIdentifier("Quickstart.Ready.StartChatting")"#),
                "the completion action must be addressable by the golden-flow harness")
        #expect(!body.contains(#".accessibilityIdentifier("Quickstart.Ready")"#),
                "a container identifier would overwrite the child button's AX identifier")
        #expect(body.contains(#"Button("Startchatting"){completeOnboarding()}"#),
                "the Ready screen must offer the Start chatting action")
        // The confirmation must run the coordinator transaction, not a
        // local shortcut that bypasses seeding / persistence.
        #expect(body.contains("coordinator.confirmStartChatting(seedWelcome:onSeedWelcome)"))
        // No fake work between the click and the app. Direction D renders
        // Ready through the shared outcome block, which has no progress slot
        // at all; pin the absence directly rather than through the retired
        // centred-card call shape.
        #expect(!body.contains("case.ready:OnboardingCenteredCanvas{ProgressView"),
                "Ready must not render a spinner — the model is already up")
        #expect(!body.contains("privatevarreadyCard:someView{ProgressView"),
                "Ready must not render a spinner — the model is already up")
    }

    @Test("Completion announces before it moves focus, and routes to Chat")
    func completionAnnouncesThenFocuses() throws {
        let body = try Self.strippedSource("Sources/Rapid/UI/ContentView.swift")
        // Routing, announcement and focus all belong to the parent half of
        // the transaction, in that order.
        let handoff = "privatefuncfinishOnboardingHandoff(){section=.chat" +
            #"VoiceOverAnnouncer.announce("Setupcomplete.Openingyourfirstchat.")"# +
            "composerFocusRequest&+=1}"
        #expect(
            body.contains(handoff),
            """
            The onboarding completion handoff no longer routes to Chat, \
            announces completion, and requests composer focus in that order. \
            A VoiceOver user who hears the composer described before they \
            hear that setup finished never learns their action worked.
            """
        )
        // The parent half must be gated on the coordinator's verdict so a
        // repeated activation cannot re-run the transition.
        let view = try Self.strippedSource("Sources/Rapid/UI/QuickstartView.swift")
        #expect(
            view.contains("guardcoordinator.confirmStartChatting(seedWelcome:onSeedWelcome)else{return}onCompleted()"),
            "the parent transition must run only when the coordinator says it completed"
        )
    }

    @Test("The composer focus request is plumbed from ContentView into ChatView")
    func composerFocusIsPlumbed() throws {
        let content = try Self.strippedSource("Sources/Rapid/UI/ContentView.swift")
        #expect(content.contains("composerFocusRequest:composerFocusRequest"),
                "ChatView must receive the focus request")
        let chat = try Self.strippedSource("Sources/Rapid/UI/ChatView.swift")
        #expect(chat.contains("varcomposerFocusRequest:Int=0"))
        #expect(
            chat.contains(".onChange(of:composerFocusRequest){_,requestinguardrequest!=0else{return}composeFocusToken&+=1}"),
            "an external focus request must reach the composer's own focus token"
        )
    }

    @Test("Skip clears pending provenance at the call site, not just in theory")
    func skipIsWiredAtTheCallSite() throws {
        let body = try Self.strippedSource("Sources/Rapid/UI/ContentView.swift")
        #expect(body.contains("quickstart.skipForNow()"),
                "the Skip callback must retire any pending Ready record")
    }

    // MARK: - 7. Behaviour outside this contract is untouched

    @Test("Skip, cached-model, low-disk, memory-warning and browse paths still hold")
    func adjacentBehaviourUnchanged() throws {
        let body = try Self.strippedSource("Sources/Rapid/UI/QuickstartView.swift")
        // Skip is still the one genuine dismiss control, and still does not
        // write the completion flag.
        #expect(body.contains(#".accessibilityIdentifier("Quickstart.Skip")"#))
        // Browse all models still has a destination — now the in-window
        // catalogue rather than the Settings window. Paper 05.2.J · S1
        // supersedes the Settings round trip; the control and its identifier
        // survive the change, which is the part this suite is guarding.
        #expect(body.contains(#".accessibilityIdentifier("Quickstart.BrowseAll")"#))
        #expect(body.contains("coordinator.beginBrowsingCatalog()"))
        #expect(!body.contains("settingsRouter.beginQuickstartCatalogRoundTrip()"),
                "onboarding must not hand the catalogue to a second window any more")
        // The low-disk warning is still non-blocking with both exits.
        #expect(body.contains(#".accessibilityIdentifier("Quickstart.LowDisk.Continue")"#))
        #expect(body.contains(#".accessibilityIdentifier("Quickstart.LowDisk.Cancel")"#))
        // The in-sheet memory decision (#1503) still exists in all three arms.
        #expect(body.contains(#".accessibilityIdentifier("Quickstart.Memory.LoadAnyway")"#))
        #expect(body.contains(#".accessibilityIdentifier("Quickstart.Memory.Cancel")"#))
        #expect(body.contains(#".accessibilityIdentifier("Quickstart.Memory.SwitchToLowMemory")"#))
        // A cached model still skips the download and goes straight to serve.
        #expect(body.contains("privatefuncstartCachedModel(_cached:ModelEntry){coordinator.enterStarting()"))
    }

    @Test("A cached-model start still lands on Ready rather than completing itself")
    func cachedModelStartStillRequiresConfirmation() {
        // The cached path is a shorter route into the SAME serving
        // lifecycle, so it must inherit the same ending: park on Ready,
        // wait for the user.
        let coord = makeCoordinator()
        coord.enterStarting()          // startCachedModel's transition
        coord.enterReady()             // server reports ready
        #expect(coord.phase == .ready)
        #expect(!coord.done, "a cached model must not complete onboarding on its own")
        clearPersistedState()
    }

    @Test("The memory-warning decision is still owned by the sheet only while starting")
    func memoryWarningOwnershipUnchanged() {
        let alias = QuickstartCoordinator.defaultChoice.alias
        let warning = ModelSizing.MemoryWarning(
            alias: alias,
            hfPath: nil,
            isAutoRespawn: false,
            severity: .unsafe,
            footprintGB: 3.25,
            freeGB: 0.7,
            totalGB: 18
        )
        #expect(QuickstartView.memoryWarningToPresent(
            phase: .starting, pending: warning, selectionAlias: alias
        ) != nil)
        // Ready is past the load — there is no parked decision to resolve,
        // and the Ready screen must not be replaced by a stale warning.
        #expect(QuickstartView.memoryWarningToPresent(
            phase: .ready, pending: warning, selectionAlias: alias
        ) == nil)
        #expect(QuickstartView.memoryWarningToPresent(
            phase: .dismissed, pending: warning, selectionAlias: alias
        ) == nil)
    }

    // MARK: - Source helpers

    private static var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // RapidTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // package root
    }

    /// Comment- and whitespace-stripped source, matching the canonical form
    /// the rest of the suite's source-grep guards use. ViewInspector is not
    /// in this target (#1492), so view wiring is pinned this way.
    private static func strippedSource(_ relativePath: String) throws -> String {
        let url = packageRoot.appendingPathComponent(relativePath)
        let body = try String(contentsOf: url, encoding: .utf8)
        return CapabilityChipRenderGateSourceGuardTests.stripCommentsAndWhitespace(body)
    }
}
