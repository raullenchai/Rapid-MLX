import AppKit
import ApplicationServices
import Foundation
import Observation
import ScreenCaptureKit

struct ComputerUseWindowOption: Identifiable, Equatable, Sendable {
    let id: String
    let applicationName: String
    let windowTitle: String
    let selection: ComputerUseWindowSelection

    var displayName: String {
        let readableName = windowTitle.isEmpty
            ? applicationName
            : "\(applicationName) — \(windowTitle)"
        return "\(readableName) · Window \(selection.windowID)"
    }
}

enum ComputerUseWindowCatalogError: Error, Equatable {
    case permissionsMissing([MacAutomationPermission])
    case unavailable
}

protocol ComputerUseWindowListing: Sendable {
    func windows() async throws -> [ComputerUseWindowOption]
}

struct MacOSComputerUseWindowCatalog: ComputerUseWindowListing {
    func windows() async throws -> [ComputerUseWindowOption] {
        let permissions = MacAutomationPermissions.snapshot()
        guard permissions.isReadyForComputerUse else {
            throw ComputerUseWindowCatalogError.permissionsMissing(
                permissions.missingForComputerUse
            )
        }
        let content: SCShareableContent
        do {
            content = try await SCShareableContent.excludingDesktopWindows(
                true,
                onScreenWindowsOnly: true
            )
        } catch {
            throw ComputerUseWindowCatalogError.unavailable
        }
        let ownPID = getpid()
        var result: [ComputerUseWindowOption] = []
        for window in content.windows {
            guard window.isOnScreen,
                  window.frame.width >= 240,
                  window.frame.height >= 160,
                  let application = window.owningApplication,
                  application.processID != ownPID,
                  !application.bundleIdentifier.isEmpty,
                  let launchDate = await launchDate(for: application.processID)
            else { continue }
            result.append(ComputerUseWindowOption(
                id: "\(application.processID):\(window.windowID)",
                applicationName: application.applicationName,
                windowTitle: window.title ?? "",
                selection: ComputerUseWindowSelection(
                    bundleIdentifier: application.bundleIdentifier,
                    processIdentifier: application.processID,
                    processLaunchDate: launchDate,
                    windowID: window.windowID
                )
            ))
        }
        return result.sorted {
            let applicationOrder = $0.applicationName.localizedCaseInsensitiveCompare(
                $1.applicationName
            )
            return applicationOrder == .orderedAscending
                || (applicationOrder == .orderedSame
                    && $0.windowTitle.localizedCaseInsensitiveCompare(
                        $1.windowTitle
                    ) == .orderedAscending)
        }
    }

    @MainActor
    private func launchDate(for processIdentifier: pid_t) -> Date? {
        NSRunningApplication(processIdentifier: processIdentifier)?.launchDate
    }
}

enum DraftPostFlowFailure: Error, Equatable, Sendable {
    case sourceIsNotTextEdit
    case destinationIsNotBrowser
    case destinationMismatch
    case targetUnavailable
    case focusChanged
    case draftMissing
    case draftAmbiguous
    case draftTooLarge
    case composerMissing
    case composerAmbiguous
    case composerNotEmpty
    case writeRejected
    case verificationFailed
    case permissionMissing
    case cancelled
    case dependencyFailure
    case accessibilityTreeTooLarge

    var isRecoverable: Bool {
        switch self {
        // A transient focus loss is safe before the write and can be retried.
        // A missing target is terminal because the same error can also surface
        // after a possible mutation; the shared coordinator cannot prove its
        // phase and must never replay the write.
        case .focusChanged:
            true
        default:
            false
        }
    }

    /// Whether the driver proves that no browser content mutation occurred.
    /// Only these failures may return to the already-reviewed plan. Unknown,
    /// rejected-write, and verification failures must start over so the
    /// destination and empty composer are inspected afresh.
    var permitsReviewedRetry: Bool {
        switch self {
        case .sourceIsNotTextEdit, .destinationIsNotBrowser,
             .destinationMismatch, .focusChanged,
             .draftMissing, .draftAmbiguous, .draftTooLarge,
             .composerMissing, .composerAmbiguous, .composerNotEmpty,
             .permissionMissing, .accessibilityTreeTooLarge:
            true
        case .targetUnavailable, .writeRejected, .verificationFailed,
             .cancelled, .dependencyFailure:
            false
        }
    }

    var userMessage: String {
        switch self {
        case .sourceIsNotTextEdit: "Choose a TextEdit window as the draft source."
        case .destinationIsNotBrowser: "Choose a supported browser window as the destination."
        case .destinationMismatch: "The selected browser is no longer on the destination you reviewed."
        case .targetUnavailable: "A selected window is no longer available."
        case .focusChanged: "Rapid could not safely focus the selected window."
        case .draftMissing: "The selected TextEdit document has no readable draft."
        case .draftAmbiguous: "More than one TextEdit document editor was found. Close auxiliary editors and try again."
        case .draftTooLarge: "The draft is too large for this preview (64 KB maximum)."
        case .composerMissing: "No editable post composer was found in the browser window."
        case .composerAmbiguous: "More than one possible composer was found. Close other editors and try again."
        case .composerNotEmpty: "The browser composer already contains text. Clear it before running this flow."
        case .writeRejected: "The browser rejected the local text update."
        case .verificationFailed: "The browser content did not match the reviewed draft."
        case .permissionMissing: "Screen Recording and Accessibility access are required."
        case .cancelled: "The flow was stopped."
        case .dependencyFailure: "The flow stopped because a local system operation failed."
        case .accessibilityTreeTooLarge: "The selected window is too complex for this preview flow."
        }
    }
}

struct DraftPostFlowMetrics: Equatable, Sendable {
    var attempts = 0
    var automaticRecoveries = 0
    var completedSteps = 0
}

enum DraftPostFlowOutcome: Equatable, Sendable {
    case readyForReview(DraftPostFlowMetrics)
    case failed(DraftPostFlowFailure, DraftPostFlowMetrics)
}

protocol DraftPostFlowDriving: Sendable {
    func transferDraft(
        from source: ComputerUseWindowOption,
        to destination: ComputerUseWindowOption
    ) async throws
}

/// The narrower execution capability used after Rapid has generated and the
/// user has reviewed a draft. The draft is already immutable input here: this
/// boundary can place and verify it, but cannot revise it or publish it.
protocol PreparedDraftPostFlowDriving: Sendable {
    func transferPreparedDraft(
        _ draft: String,
        to destination: ComputerUseWindowOption,
        expectedDestination: ComputerUseBrowserDestinationIdentity
    ) async throws
}

struct ComputerUseBrowserDestinationIdentity: Equatable, Sendable {
    let host: String
    let documentIdentity: String
}

protocol ComputerUseBrowserDestinationInspecting: Sendable {
    func destinationIdentity(
        for destination: ComputerUseWindowOption
    ) async throws -> ComputerUseBrowserDestinationIdentity
}

private enum DraftPostTransferRetry {
    static func run(
        maximumAttempts: Int,
        operation: @escaping @Sendable () async throws -> Void
    ) async -> DraftPostFlowOutcome {
        var metrics = DraftPostFlowMetrics()
        for attempt in 1 ... maximumAttempts {
            if Task.isCancelled {
                return .failed(.cancelled, metrics)
            }
            metrics.attempts = attempt
            do {
                try await operation()
                metrics.completedSteps = 3
                return .readyForReview(metrics)
            } catch let failure as DraftPostFlowFailure {
                guard failure.isRecoverable, attempt < maximumAttempts else {
                    return .failed(failure, metrics)
                }
                metrics.automaticRecoveries += 1
            } catch is CancellationError {
                return .failed(.cancelled, metrics)
            } catch {
                // Only typed, known pre-mutation focus/window failures may
                // retry. Unknown adapter failures fail closed because they
                // could have happened after a local mutation.
                return .failed(.dependencyFailure, metrics)
            }
        }
        return .failed(.targetUnavailable, metrics)
    }
}

/// Runs one idempotent local transfer with a strict retry budget. The driver
/// can only populate the composer; publishing is intentionally absent from
/// this protocol and therefore cannot be reached by recovery logic.
actor DraftPostFlowCoordinator {
    private let driver: any DraftPostFlowDriving
    private let maximumAttempts: Int

    init(driver: any DraftPostFlowDriving, maximumAttempts: Int = 3) {
        self.driver = driver
        self.maximumAttempts = min(max(1, maximumAttempts), 3)
    }

    func run(
        source: ComputerUseWindowOption,
        destination: ComputerUseWindowOption
    ) async -> DraftPostFlowOutcome {
        await DraftPostTransferRetry.run(maximumAttempts: maximumAttempts) {
            try await self.driver.transferDraft(from: source, to: destination)
        }
    }
}

/// Runs a generated draft through the same bounded, idempotent browser
/// transfer policy as the original TextEdit-backed preview.
actor PreparedDraftPostFlowCoordinator {
    private let driver: any PreparedDraftPostFlowDriving
    private let maximumAttempts: Int

    init(driver: any PreparedDraftPostFlowDriving, maximumAttempts: Int = 3) {
        self.driver = driver
        self.maximumAttempts = min(max(1, maximumAttempts), 3)
    }

    func run(
        draft: String,
        destination: ComputerUseWindowOption,
        expectedDestination: ComputerUseBrowserDestinationIdentity
    ) async -> DraftPostFlowOutcome {
        await DraftPostTransferRetry.run(maximumAttempts: maximumAttempts) {
            try await self.driver.transferPreparedDraft(
                draft,
                to: destination,
                expectedDestination: expectedDestination
            )
        }
    }
}

/// The complete mutation authority granted to the draft flow. Publishing is
/// structurally impossible because this capability can only focus a composer
/// and set its draft value.
protocol DraftPostComposerActuating: Sendable {
    func focusComposer(_ composer: AXUIElement) throws
    func setDraft(_ draft: String, on composer: AXUIElement) throws
}

/// Optional, local-only recovery for a browser composer that Accessibility
/// cannot identify by an explicit semantic label. The recovery capability may
/// only focus one empty editable element in the selected window; draft writes
/// remain owned by ``DraftPostComposerActuating`` and publishing is absent.
protocol DraftPostVisualRecovering: Sendable {
    func focusComposer(
        in destination: ComputerUseWindowOption,
        documentIdentity: String
    ) async throws
}

struct AXDraftPostComposerActuator: DraftPostComposerActuating {
    func focusComposer(_ composer: AXUIElement) throws {
        guard AXUIElementSetAttributeValue(
            composer,
            kAXFocusedAttribute as CFString,
            kCFBooleanTrue
        ) == .success else {
            throw DraftPostFlowFailure.focusChanged
        }
    }

    func setDraft(_ draft: String, on composer: AXUIElement) throws {
        guard AXUIElementSetAttributeValue(
            composer,
            kAXValueAttribute as CFString,
            draft as CFString
        ) == .success else {
            throw DraftPostFlowFailure.writeRejected
        }
    }
}

/// Accessibility-first implementation for the first bounded starter flow.
/// The user selects both windows. Rapid reads one TextEdit document, writes an
/// empty browser composer, verifies the exact value, and stops. No coordinate
/// action and no publish/send action exists in this adapter.
struct MacOSDraftPostFlowDriver: DraftPostFlowDriving, PreparedDraftPostFlowDriving,
    ComputerUseBrowserDestinationInspecting
{
    static let maximumDraftBytes = 65_536
    private static let textEditBundle = "com.apple.TextEdit"
    static let browserBundles: Set<String> = [
        "com.apple.Safari",
        "com.google.Chrome",
    ]
    private let actuator: any DraftPostComposerActuating
    private let visualRecovery: (any DraftPostVisualRecovering)?

    init(
        actuator: any DraftPostComposerActuating = AXDraftPostComposerActuator(),
        visualRecovery: (any DraftPostVisualRecovering)? = nil
    ) {
        self.actuator = actuator
        self.visualRecovery = visualRecovery
    }

    func transferDraft(
        from source: ComputerUseWindowOption,
        to destination: ComputerUseWindowOption
    ) async throws {
        guard source.selection.bundleIdentifier == Self.textEditBundle else {
            throw DraftPostFlowFailure.sourceIsNotTextEdit
        }
        let draft = try await readDraft(from: source)
        try await transfer(
            draft,
            to: destination,
            source: source,
            expectedDestination: nil
        )
    }

    func transferPreparedDraft(
        _ draft: String,
        to destination: ComputerUseWindowOption,
        expectedDestination: ComputerUseBrowserDestinationIdentity
    ) async throws {
        try await transfer(
            draft,
            to: destination,
            source: nil,
            expectedDestination: expectedDestination
        )
    }

    func destinationIdentity(
        for destination: ComputerUseWindowOption
    ) async throws -> ComputerUseBrowserDestinationIdentity {
        guard Self.browserBundles.contains(destination.selection.bundleIdentifier) else {
            throw DraftPostFlowFailure.destinationIsNotBrowser
        }
        guard MacAutomationPermissions.snapshot().isReadyForComputerUse else {
            throw DraftPostFlowFailure.permissionMissing
        }
        let browserAccessibilityRestoreValue = try await prepareBrowserAccessibility(
            for: destination
        )
        defer {
            if let browserAccessibilityRestoreValue {
                Self.restoreBrowserAccessibility(
                    for: destination,
                    enabled: browserAccessibilityRestoreValue
                )
            }
        }
        let identity = try await browserDocumentIdentity(in: destination)
        guard let host = Self.normalizedDestinationHost(from: identity),
              let normalizedIdentity = Self.normalizedDocumentIdentity(from: identity)
        else {
            throw DraftPostFlowFailure.destinationMismatch
        }
        return ComputerUseBrowserDestinationIdentity(
            host: host,
            documentIdentity: normalizedIdentity
        )
    }

    private func transfer(
        _ draft: String,
        to destination: ComputerUseWindowOption,
        source: ComputerUseWindowOption?,
        expectedDestination: ComputerUseBrowserDestinationIdentity?
    ) async throws {
        guard Self.browserBundles.contains(destination.selection.bundleIdentifier) else {
            throw DraftPostFlowFailure.destinationIsNotBrowser
        }
        guard MacAutomationPermissions.snapshot().isReadyForComputerUse else {
            throw DraftPostFlowFailure.permissionMissing
        }
        guard !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DraftPostFlowFailure.draftMissing
        }
        guard draft.utf8.count <= Self.maximumDraftBytes else {
            throw DraftPostFlowFailure.draftTooLarge
        }
        let browserAccessibilityRestoreValue = try await prepareBrowserAccessibility(
            for: destination
        )
        defer {
            if let browserAccessibilityRestoreValue {
                Self.restoreBrowserAccessibility(
                    for: destination,
                    enabled: browserAccessibilityRestoreValue
                )
            }
        }
        let documentIdentity = try await browserDocumentIdentity(in: destination)
        if let expectedDestination {
            guard Self.browserDestinationMatches(
                currentAddress: documentIdentity,
                expected: expectedDestination
            ) else { throw DraftPostFlowFailure.destinationMismatch }
        }
        var usedVisualRecovery = false
        do {
            try await writeAndVerify(
                draft,
                source: source,
                to: destination,
                documentIdentity: documentIdentity,
                allowFocusedUnlabelledComposer: false,
                focusDestination: true
            )
        } catch DraftPostFlowFailure.composerMissing {
            guard let visualRecovery else {
                throw DraftPostFlowFailure.composerMissing
            }
            usedVisualRecovery = true
            try await visualRecovery.focusComposer(
                in: destination,
                documentIdentity: documentIdentity
            )
            try await Self.verifyAfterVisualRecovery {
                try await writeAndVerify(
                    draft,
                    source: source,
                    to: destination,
                    documentIdentity: documentIdentity,
                    allowFocusedUnlabelledComposer: true,
                    focusDestination: false
                )
            }
        }
        // The destination may now be mutated. Every remaining observation is
        // therefore terminal on failure: recovery must never replay the write.
        do {
            try await Self.verifyDefinitivePostMutationState(
                draft: draft,
                source: source,
                destination: destination,
                documentIdentity: documentIdentity,
                allowFocusedUnlabelledComposer: usedVisualRecovery
            )
        } catch {
            throw DraftPostFlowFailure.verificationFailed
        }
    }

    private func readDraft(from source: ComputerUseWindowOption) async throws -> String {
        try await focus(source)
        return try await Self.runAXWork {
            let window = try Self.exactFocusedWindow(source)
            let candidates = try Self.editableElements(in: window)
                .filter {
                    Self.stringAttribute(
                        kAXRoleAttribute as CFString,
                        from: $0
                    ) == kAXTextAreaRole as String
                }
                .map { Self.stringAttribute(kAXValueAttribute as CFString, from: $0) }
            return try Self.uniqueDraft(in: candidates)
        }
    }

    /// Chromium keeps web content out of its macOS Accessibility tree until an
    /// assistive client requests enhanced UI. Make one balanced request for
    /// this flow only, and avoid touching Safari or a Chrome session that is
    /// already exposing an AXWebArea.
    private func prepareBrowserAccessibility(
        for destination: ComputerUseWindowOption
    ) async throws -> Bool? {
        guard destination.selection.bundleIdentifier == "com.google.Chrome" else {
            return nil
        }
        let alreadyAvailable = try await Self.runAXWork {
            let application = Self.applicationElement(
                destination.selection.processIdentifier
            )
            guard let window = Self.window(matching: destination, in: application) else {
                throw DraftPostFlowFailure.targetUnavailable
            }
            return try Self.allElements(in: window).contains {
                Self.stringAttribute(kAXRoleAttribute as CFString, from: $0)
                    == "AXWebArea"
            }
        }
        guard !alreadyAvailable else { return nil }
        let application = Self.applicationElement(
            destination.selection.processIdentifier
        )
        guard let previousValue = Self.boolAttribute(
            "AXEnhancedUserInterface" as CFString,
            from: application
        ) else {
            throw DraftPostFlowFailure.dependencyFailure
        }
        return try await Self.establishBrowserAccessibilityLease(
            previousValue: previousValue,
            activate: {
                guard AXUIElementSetAttributeValue(
                    application,
                    "AXEnhancedUserInterface" as CFString,
                    kCFBooleanTrue
                ) == .success else {
                    throw DraftPostFlowFailure.dependencyFailure
                }
            },
            settle: {
                try await Task.sleep(for: .seconds(3))
                try Task.checkCancellation()
            },
            restore: { value in
                Self.restoreBrowserAccessibility(
                    for: destination,
                    enabled: value
                )
            }
        )
    }

    /// Transfers cleanup ownership to the caller only after activation has
    /// fully settled. Any error or cancellation before then is balanced here.
    static func establishBrowserAccessibilityLease(
        previousValue: Bool,
        activate: () throws -> Void,
        settle: () async throws -> Void,
        restore: (Bool) -> Void
    ) async throws -> Bool {
        try activate()
        do {
            try await settle()
        } catch {
            restore(previousValue)
            throw error
        }
        return previousValue
    }

    private static func restoreBrowserAccessibility(
        for destination: ComputerUseWindowOption,
        enabled: Bool
    ) {
        guard destination.selection.bundleIdentifier == "com.google.Chrome",
              let running = NSRunningApplication(
                processIdentifier: destination.selection.processIdentifier
              ),
              running.bundleIdentifier == destination.selection.bundleIdentifier,
              running.launchDate == destination.selection.processLaunchDate
        else { return }
        _ = AXUIElementSetAttributeValue(
            applicationElement(destination.selection.processIdentifier),
            "AXEnhancedUserInterface" as CFString,
            enabled ? kCFBooleanTrue : kCFBooleanFalse
        )
    }

    /// Once visual recovery has consumed its one bounded three-attempt budget,
    /// any focus/verification drift is terminal. Converting it to a
    /// non-recoverable failure prevents the outer coordinator from starting a
    /// second visual budget (three-by-three attempts).
    static func verifyAfterVisualRecovery(
        _ operation: () async throws -> Void
    ) async throws {
        do {
            try await operation()
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw DraftPostFlowFailure.verificationFailed
        }
    }

    private func writeAndVerify(
        _ draft: String,
        source: ComputerUseWindowOption?,
        to destination: ComputerUseWindowOption,
        documentIdentity: String,
        allowFocusedUnlabelledComposer: Bool,
        focusDestination: Bool
    ) async throws {
        if focusDestination {
            try await focus(destination)
        }
        try Task.checkCancellation()
        try await Self.runAXWork {
            let window = try Self.exactFocusedBrowserWindow(
                destination,
                documentIdentity: documentIdentity
            )
            let composer = try Self.uniqueComposer(
                in: window,
                allowFocusedUnlabelledComposer: allowFocusedUnlabelledComposer,
                browserBundleIdentifier: destination.selection.bundleIdentifier
            )
            guard let existing = Self.stringAttribute(
                kAXValueAttribute as CFString,
                from: composer
            ) else {
                throw DraftPostFlowFailure.verificationFailed
            }
            guard existing.isEmpty || Self.utf8Matches(existing, draft) else {
                throw DraftPostFlowFailure.composerNotEmpty
            }
            if Self.utf8Matches(existing, draft) {
                let currentWindow = try Self.exactFocusedBrowserWindow(
                    destination,
                    documentIdentity: documentIdentity
                )
                let currentComposer = try Self.uniqueComposer(
                    in: currentWindow,
                    allowFocusedUnlabelledComposer: allowFocusedUnlabelledComposer,
                    browserBundleIdentifier: destination.selection.bundleIdentifier
                )
                guard CFEqual(composer, currentComposer),
                      Self.stringAttribute(
                        kAXValueAttribute as CFString,
                        from: currentComposer
                      ).map({ Self.utf8Matches($0, draft) }) == true
                else {
                    throw DraftPostFlowFailure.verificationFailed
                }
                return
            }
            var settable: DarwinBoolean = false
            guard AXUIElementIsAttributeSettable(
                composer,
                kAXValueAttribute as CFString,
                &settable
            ) == .success, settable.boolValue else {
                throw DraftPostFlowFailure.writeRejected
            }
            try actuator.focusComposer(composer)
            // Re-resolve the exact selected window immediately before the
            // value mutation. Focusing the exact bound editor is allowed, but
            // it must not have moved focus into another window.
            let currentWindow = try Self.exactFocusedBrowserWindow(
                destination,
                documentIdentity: documentIdentity
            )
            let currentComposer = try Self.uniqueComposer(
                in: currentWindow,
                allowFocusedUnlabelledComposer: allowFocusedUnlabelledComposer,
                browserBundleIdentifier: destination.selection.bundleIdentifier
            )
            guard CFEqual(composer, currentComposer) else {
                throw DraftPostFlowFailure.verificationFailed
            }
            guard let currentValue = Self.stringAttribute(
                kAXValueAttribute as CFString,
                from: currentComposer
            ) else {
                throw DraftPostFlowFailure.verificationFailed
            }
            guard currentValue.isEmpty || Self.utf8Matches(currentValue, draft) else {
                throw DraftPostFlowFailure.composerNotEmpty
            }
            if let source {
                let currentSource = try Self.readDraftWithoutFocusing(from: source)
                guard Self.utf8Matches(currentSource, draft) else {
                    throw DraftPostFlowFailure.verificationFailed
                }
            }
            let authorizedWindow = try Self.exactFocusedBrowserWindow(
                destination,
                documentIdentity: documentIdentity
            )
            let authorizedComposer = try Self.uniqueComposer(
                in: authorizedWindow,
                allowFocusedUnlabelledComposer: allowFocusedUnlabelledComposer,
                browserBundleIdentifier: destination.selection.bundleIdentifier
            )
            guard CFEqual(currentComposer, authorizedComposer),
                  let authorizedValue = Self.stringAttribute(
                    kAXValueAttribute as CFString,
                    from: authorizedComposer
                  ), authorizedValue.isEmpty
            else {
                throw DraftPostFlowFailure.verificationFailed
            }
            // This is the final cancellation boundary before the only content
            // mutation. No suspension occurs between this check and the write.
            try Task.checkCancellation()
            try actuator.setDraft(draft, on: authorizedComposer)
            // Once content may have changed, no focus/window error is safe to
            // retry. Collapse every post-mutation observation failure into a
            // terminal verification failure.
            try Self.verifyAfterMutation {
                let verifiedWindow = try Self.exactFocusedBrowserWindow(
                    destination,
                    documentIdentity: documentIdentity
                )
                let verifiedComposer = try Self.uniqueComposer(
                    in: verifiedWindow,
                    allowFocusedUnlabelledComposer: allowFocusedUnlabelledComposer,
                    browserBundleIdentifier: destination.selection.bundleIdentifier,
                    focusedUnlabelledValue: draft
                )
                return CFEqual(authorizedComposer, verifiedComposer)
            }
        }
    }

    static func utf8Matches(_ lhs: String, _ rhs: String) -> Bool {
        lhs.utf8.elementsEqual(rhs.utf8)
    }

    static func normalizedDestinationHost(from address: String) -> String? {
        guard let identity = normalizedDocumentIdentity(from: address),
              let host = URLComponents(string: identity)?.host
        else { return nil }
        return host
    }

    static func normalizedDocumentIdentity(from address: String) -> String? {
        let trimmed = address.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let candidate = trimmed.contains("://") ? trimmed : "https://\(trimmed)"
        guard var components = URLComponents(string: candidate),
              let host = components.host?
            .trimmingCharacters(in: CharacterSet(charactersIn: "."))
            .lowercased(), !host.isEmpty
        else { return nil }
        components.scheme = components.scheme?.lowercased()
        components.host = host
        if (components.scheme == "https" && components.port == 443)
            || (components.scheme == "http" && components.port == 80)
        {
            components.port = nil
        }
        // Preserve the fragment deliberately. Single-page applications often
        // use it as document or account state, so a hash-route change after
        // review invalidates the authorization just like a path change.
        return components.string
    }

    static func browserDestinationMatches(
        currentAddress: String,
        expected: ComputerUseBrowserDestinationIdentity
    ) -> Bool {
        guard let normalizedIdentity = normalizedDocumentIdentity(from: currentAddress),
              utf8Matches(normalizedIdentity, expected.documentIdentity),
              let currentHost = normalizedDestinationHost(from: currentAddress)
        else { return false }
        return currentHost.caseInsensitiveCompare(expected.host) == .orderedSame
    }

    static func verifyAfterMutation(_ verifier: () throws -> Bool) throws {
        do {
            guard try verifier() else {
                throw DraftPostFlowFailure.verificationFailed
            }
        } catch {
            throw DraftPostFlowFailure.verificationFailed
        }
    }

    private static func verifyDefinitivePostMutationState(
        draft: String,
        source: ComputerUseWindowOption?,
        destination: ComputerUseWindowOption,
        documentIdentity: String,
        allowFocusedUnlabelledComposer: Bool
    ) async throws {
        // This detached task is intentionally cancellation-insensitive: after
        // a possible write, Stop must wait for definitive source/destination
        // verification rather than misreporting an unknown mutation state.
        try await Task.detached {
            try await Task.sleep(for: .milliseconds(300))
            if let source {
                let finalSource = try readDraftWithoutFocusing(from: source)
                guard utf8Matches(finalSource, draft) else {
                    throw DraftPostFlowFailure.verificationFailed
                }
            }
            let selection = destination.selection
            guard let running = NSRunningApplication(
                processIdentifier: selection.processIdentifier
            ), running.bundleIdentifier == selection.bundleIdentifier,
                running.launchDate == selection.processLaunchDate
            else { throw DraftPostFlowFailure.verificationFailed }
            let application = applicationElement(selection.processIdentifier)
            guard let window = window(matching: destination, in: application),
                  try currentBrowserDocumentIdentity(
                    in: window,
                    browserBundleIdentifier: destination.selection.bundleIdentifier
                  ) == documentIdentity
            else { throw DraftPostFlowFailure.verificationFailed }
            let composer = try uniqueComposer(
                in: window,
                allowFocusedUnlabelledComposer: allowFocusedUnlabelledComposer,
                browserBundleIdentifier: destination.selection.bundleIdentifier,
                focusedUnlabelledValue: draft
            )
            guard stringAttribute(
                kAXValueAttribute as CFString,
                from: composer
            ).map({ utf8Matches($0, draft) }) == true else {
                throw DraftPostFlowFailure.verificationFailed
            }
        }.value
    }

    private static func exactFocusedBrowserWindow(
        _ destination: ComputerUseWindowOption,
        documentIdentity: String
    ) throws -> AXUIElement {
        let window = try exactFocusedWindow(destination)
        guard browserWindowTitleMatches(
            browserBundleIdentifier: destination.selection.bundleIdentifier,
            currentTitle: stringAttribute(
            kAXTitleAttribute as CFString,
            from: window
            ),
            selectedTitle: destination.windowTitle
        ), try currentBrowserDocumentIdentity(
            in: window,
            browserBundleIdentifier: destination.selection.bundleIdentifier
        ) == documentIdentity
        else {
            throw DraftPostFlowFailure.focusChanged
        }
        return window
    }

    static func browserDocumentMatches(
        currentTitle: String?,
        selectedTitle: String
    ) -> Bool {
        guard let currentTitle, !selectedTitle.isEmpty else { return false }
        return utf8Matches(currentTitle, selectedTitle)
    }

    static func browserWindowTitleMatches(
        browserBundleIdentifier: String,
        currentTitle: String?,
        selectedTitle: String
    ) -> Bool {
        guard let currentTitle, !selectedTitle.isEmpty else { return false }
        if utf8Matches(currentTitle, selectedTitle) { return true }
        guard browserBundleIdentifier == "com.google.Chrome" else { return false }
        return utf8Matches(currentTitle, selectedTitle + " - Google Chrome")
    }

    private func browserDocumentIdentity(
        in destination: ComputerUseWindowOption
    ) async throws -> String {
        return try await Self.runAXWork {
            let selection = destination.selection
            guard let running = NSRunningApplication(
                processIdentifier: selection.processIdentifier
            ), running.bundleIdentifier == selection.bundleIdentifier,
                running.launchDate == selection.processLaunchDate
            else { throw DraftPostFlowFailure.targetUnavailable }
            let application = Self.applicationElement(selection.processIdentifier)
            guard let window = Self.window(matching: destination, in: application)
            else { throw DraftPostFlowFailure.focusChanged }
            return try Self.currentBrowserDocumentIdentity(
                in: window,
                browserBundleIdentifier: selection.bundleIdentifier
            )
        }
    }

    private static func currentBrowserDocumentIdentity(
        in window: AXUIElement,
        browserBundleIdentifier: String
    ) throws -> String {
        guard let windowFrame = elementFrame(window) else {
            throw DraftPostFlowFailure.composerAmbiguous
        }
        let addressFields = try allElements(in: window).filter { element in
            guard isBrowserAddressField(
                element,
                browserBundleIdentifier: browserBundleIdentifier
            ),
                boolAttribute(kAXEnabledAttribute as CFString, from: element) == true,
                let frame = elementFrame(element), windowFrame.intersects(frame)
            else { return false }
            return true
        }
        guard addressFields.count == 1,
              let identity = stringAttribute(
                kAXValueAttribute as CFString,
                from: addressFields[0]
              ), !identity.isEmpty
        else { throw DraftPostFlowFailure.composerAmbiguous }
        return identity
    }

    static func matchesBrowserAddressField(
        browserBundleIdentifier: String,
        role: String?,
        identifier: String?,
        description: String?
    ) -> Bool {
        guard role == kAXTextFieldRole as String else { return false }
        switch browserBundleIdentifier {
        case "com.apple.Safari":
            return identifier == "WEB_BROWSER_ADDRESS_AND_SEARCH_FIELD"
        case "com.google.Chrome":
            return description?.trimmingCharacters(in: .whitespacesAndNewlines)
                .caseInsensitiveCompare("Address and search bar") == .orderedSame
        default:
            return false
        }
    }

    private static func isBrowserAddressField(
        _ element: AXUIElement,
        browserBundleIdentifier: String
    ) -> Bool {
        matchesBrowserAddressField(
            browserBundleIdentifier: browserBundleIdentifier,
            role: stringAttribute(kAXRoleAttribute as CFString, from: element),
            identifier: stringAttribute(kAXIdentifierAttribute as CFString, from: element),
            description: stringAttribute(kAXDescriptionAttribute as CFString, from: element)
        )
    }

    private static func readDraftWithoutFocusing(
        from source: ComputerUseWindowOption
    ) throws -> String {
        let selection = source.selection
        guard let running = NSRunningApplication(
            processIdentifier: selection.processIdentifier
        ), running.bundleIdentifier == selection.bundleIdentifier,
            running.launchDate == selection.processLaunchDate
        else { throw DraftPostFlowFailure.targetUnavailable }
        let application = applicationElement(selection.processIdentifier)
        guard let window = window(matching: source, in: application) else {
            throw DraftPostFlowFailure.targetUnavailable
        }
        let candidates = try editableElements(in: window)
            .filter {
                stringAttribute(kAXRoleAttribute as CFString, from: $0)
                    == kAXTextAreaRole as String
            }
            .map { stringAttribute(kAXValueAttribute as CFString, from: $0) }
        return try uniqueDraft(in: candidates)
    }

    private static func runAXWork<Result: Sendable>(
        _ operation: @escaping @Sendable () throws -> Result
    ) async throws -> Result {
        try Task.checkCancellation()
        return try await withThrowingTaskGroup(of: Result.self) { group in
            group.addTask {
                try Task.checkCancellation()
                return try operation()
            }
            guard let result = try await group.next() else {
                throw CancellationError()
            }
            return result
        }
    }

    private func focus(_ option: ComputerUseWindowOption) async throws {
        let selection = option.selection
        try Task.checkCancellation()
        try await MainActor.run {
            guard let app = NSRunningApplication(processIdentifier: selection.processIdentifier),
                  app.bundleIdentifier == selection.bundleIdentifier,
                  app.launchDate == selection.processLaunchDate
            else { throw DraftPostFlowFailure.targetUnavailable }
            app.activate()
            let application = Self.applicationElement(selection.processIdentifier)
            guard let window = Self.window(matching: option, in: application),
                  AXUIElementPerformAction(window, kAXRaiseAction as CFString) == .success
            else { throw DraftPostFlowFailure.targetUnavailable }
        }
        try await Task.sleep(for: .milliseconds(180))
        try Task.checkCancellation()
        try await MainActor.run {
            _ = try Self.exactFocusedWindow(option)
        }
    }

    private static func exactFocusedWindow(
        _ option: ComputerUseWindowOption
    ) throws -> AXUIElement {
        let selection = option.selection
        guard let running = NSRunningApplication(
            processIdentifier: selection.processIdentifier
        ),
            running.bundleIdentifier == selection.bundleIdentifier,
            running.launchDate == selection.processLaunchDate,
            NSWorkspace.shared.frontmostApplication?.processIdentifier
                == selection.processIdentifier
        else { throw DraftPostFlowFailure.focusChanged }
        let application = applicationElement(selection.processIdentifier)
        guard let selected = window(matching: option, in: application) else {
            throw DraftPostFlowFailure.targetUnavailable
        }
        var focusedValue: CFTypeRef?
        guard AXUIElementCopyAttributeValue(
            application,
            kAXFocusedWindowAttribute as CFString,
            &focusedValue
        ) == .success,
            let focusedValue,
            CFGetTypeID(focusedValue) == AXUIElementGetTypeID(),
            CFEqual(selected, unsafeDowncast(focusedValue, to: AXUIElement.self))
        else { throw DraftPostFlowFailure.focusChanged }
        return selected
    }

    private static func applicationElement(_ processIdentifier: pid_t) -> AXUIElement {
        let application = AXUIElementCreateApplication(processIdentifier)
        AXUIElementSetMessagingTimeout(application, 0.35)
        return application
    }

    private static func window(
        matching option: ComputerUseWindowOption,
        in application: AXUIElement
    ) -> AXUIElement? {
        guard let candidate = window(matching: option.selection, in: application),
              browserWindowTitleMatches(
                browserBundleIdentifier: option.selection.bundleIdentifier,
                currentTitle: stringAttribute(
                    kAXTitleAttribute as CFString,
                    from: candidate
                ),
                selectedTitle: option.windowTitle
              )
        else { return nil }
        return candidate
    }

    private static func window(
        matching selection: ComputerUseWindowSelection,
        in application: AXUIElement
    ) -> AXUIElement? {
        guard let frame = currentFrame(for: selection) else { return nil }
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(
            application,
            kAXWindowsAttribute as CFString,
            &value
        ) == .success,
            let windows = value as? [AXUIElement]
        else { return nil }
        let matches = windows.filter {
            guard let candidate = elementFrame($0) else { return false }
            return MacOSComputerUseWindowIdentity.framesMatch(candidate, frame)
        }
        return matches.count == 1 ? matches[0] : nil
    }

    private static func currentFrame(
        for selection: ComputerUseWindowSelection
    ) -> CGRect? {
        guard let records = CGWindowListCopyWindowInfo(
            [.optionOnScreenOnly, .excludeDesktopElements],
            kCGNullWindowID
        ) as? [[CFString: Any]],
            let record = records.first(where: {
                ($0[kCGWindowNumber] as? NSNumber)?.uint32Value == selection.windowID
                    && ($0[kCGWindowOwnerPID] as? NSNumber)?.int32Value
                        == selection.processIdentifier
            }),
            let bounds = record[kCGWindowBounds] as? [String: NSNumber]
        else { return nil }
        return CGRect(dictionaryRepresentation: bounds as CFDictionary)
    }

    private static func allElements(in root: AXUIElement) throws -> [AXUIElement] {
        var queue: [(AXUIElement, Int)] = [(root, 0)]
        var result: [AXUIElement] = []
        var visited = Set<AXUIElement>()
        var cursor = 0
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: .seconds(2))
        while cursor < queue.count, visited.count < 2_048 {
            guard clock.now < deadline else {
                throw DraftPostFlowFailure.dependencyFailure
            }
            let (element, depth) = queue[cursor]
            cursor += 1
            guard visited.insert(element).inserted else { continue }
            result.append(element)
            guard depth < 32 else { continue }
            var value: CFTypeRef?
            let childrenResult = AXUIElementCopyAttributeValue(
                element,
                kAXChildrenAttribute as CFString,
                &value
            )
            if childrenResult == .success,
                let children = value as? [AXUIElement]
            {
                queue.append(contentsOf: children.map { ($0, depth + 1) })
            } else if childrenResult != .noValue,
                      childrenResult != .attributeUnsupported
            {
                throw DraftPostFlowFailure.dependencyFailure
            }
        }
        guard cursor >= queue.count else {
            throw DraftPostFlowFailure.accessibilityTreeTooLarge
        }
        return result
    }

    private static func editableElements(in root: AXUIElement) throws -> [AXUIElement] {
        try allElements(in: root).filter { element in
            let role = stringAttribute(kAXRoleAttribute as CFString, from: element)
            let subrole = stringAttribute(kAXSubroleAttribute as CFString, from: element)
            return (role == kAXTextAreaRole as String || role == kAXTextFieldRole as String)
                && subrole != kAXSecureTextFieldSubrole as String
        }
    }

    private static func isExplicitComposer(
        _ element: AXUIElement,
        in window: AXUIElement,
        windowFrame: CGRect
    ) -> Bool {
        guard boolAttribute(kAXEnabledAttribute as CFString, from: element) == true,
              let frame = elementFrame(element), frame.width >= 1, frame.height >= 1,
              windowFrame.insetBy(dx: -0.5, dy: -0.5).contains(frame),
              hasVisibleAncestry(element, through: window)
        else { return false }
        guard isSettable(kAXValueAttribute as CFString, on: element) else { return false }
        let fields = [
            stringAttribute(kAXTitleAttribute as CFString, from: element),
            stringAttribute(kAXDescriptionAttribute as CFString, from: element),
            stringAttribute(kAXHelpAttribute as CFString, from: element),
            stringAttribute("AXPlaceholderValue" as CFString, from: element),
        ].compactMap { $0 }
        return fields.contains(where: isExplicitComposerLabel)
    }

    private static func isSettable(_ attribute: CFString, on element: AXUIElement) -> Bool {
        var settable: DarwinBoolean = false
        return AXUIElementIsAttributeSettable(element, attribute, &settable) == .success
            && settable.boolValue
    }

    private static func hasVisibleAncestry(
        _ element: AXUIElement,
        through window: AXUIElement
    ) -> Bool {
        var current = element
        var visited = Set<AXUIElement>()
        for _ in 0 ..< 32 {
            guard visited.insert(current).inserted,
                  boolAttribute("AXHidden" as CFString, from: current) != true
            else { return false }
            if CFEqual(current, window) {
                return true
            }
            var parentValue: CFTypeRef?
            guard AXUIElementCopyAttributeValue(
                current,
                kAXParentAttribute as CFString,
                &parentValue
            ) == .success,
                let parentValue,
                CFGetTypeID(parentValue) == AXUIElementGetTypeID()
            else { return false }
            current = unsafeDowncast(parentValue, to: AXUIElement.self)
        }
        return false
    }

    private static func boolAttribute(
        _ attribute: CFString,
        from element: AXUIElement
    ) -> Bool? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, attribute, &value) == .success,
              let number = value as? NSNumber
        else { return nil }
        return number.boolValue
    }

    private static func uniqueComposer(
        in window: AXUIElement,
        allowFocusedUnlabelledComposer: Bool = false,
        browserBundleIdentifier: String? = nil,
        focusedUnlabelledValue: String = ""
    ) throws -> AXUIElement {
        guard let windowFrame = elementFrame(window) else {
            throw DraftPostFlowFailure.composerMissing
        }
        let matches = try editableElements(in: window).filter {
            isExplicitComposer($0, in: window, windowFrame: windowFrame)
        }
        guard let match = matches.first else {
            guard allowFocusedUnlabelledComposer,
                  let browserBundleIdentifier,
                  let focused = focusedEditableElement(
                    in: window,
                    browserBundleIdentifier: browserBundleIdentifier,
                    requiredValue: focusedUnlabelledValue
                  )
            else { throw DraftPostFlowFailure.composerMissing }
            return focused
        }
        if matches.count > 1 {
            throw DraftPostFlowFailure.composerAmbiguous
        }
        return match
    }

    private static func focusedEditableElement(
        in window: AXUIElement,
        browserBundleIdentifier: String,
        requiredValue: String = ""
    ) -> AXUIElement? {
        guard let application = applicationAncestor(of: window) else { return nil }
        var focusedValue: CFTypeRef?
        guard AXUIElementCopyAttributeValue(
            application,
            kAXFocusedUIElementAttribute as CFString,
            &focusedValue
        ) == .success,
            let focusedValue,
            CFGetTypeID(focusedValue) == AXUIElementGetTypeID()
        else { return nil }
        let focused = unsafeDowncast(focusedValue, to: AXUIElement.self)
        guard isSafeEmptyEditable(
            focused,
            in: window,
            browserBundleIdentifier: browserBundleIdentifier,
            requiredValue: requiredValue
        ) else { return nil }
        return focused
    }

    private static func applicationAncestor(of element: AXUIElement) -> AXUIElement? {
        var current = element
        var visited = Set<AXUIElement>()
        for _ in 0 ..< 32 {
            guard visited.insert(current).inserted else { return nil }
            if stringAttribute(kAXRoleAttribute as CFString, from: current)
                == kAXApplicationRole as String
            {
                return current
            }
            var parentValue: CFTypeRef?
            guard AXUIElementCopyAttributeValue(
                current,
                kAXParentAttribute as CFString,
                &parentValue
            ) == .success,
                let parentValue,
                CFGetTypeID(parentValue) == AXUIElementGetTypeID()
            else { return nil }
            current = unsafeDowncast(parentValue, to: AXUIElement.self)
        }
        return nil
    }

    private static func isSafeEmptyEditable(
        _ element: AXUIElement,
        in window: AXUIElement,
        browserBundleIdentifier: String,
        requiredValue: String = ""
    ) -> Bool {
        let role = stringAttribute(kAXRoleAttribute as CFString, from: element)
        let subrole = stringAttribute(kAXSubroleAttribute as CFString, from: element)
        guard matchesSafeVisualComposer(
            role: role,
            subrole: subrole,
            isBrowserAddressField: isBrowserAddressField(
                element,
                browserBundleIdentifier: browserBundleIdentifier
            ),
            isEnabled: boolAttribute(kAXEnabledAttribute as CFString, from: element),
            value: stringAttribute(kAXValueAttribute as CFString, from: element),
            requiredValue: requiredValue,
            isValueSettable: isSettable(kAXValueAttribute as CFString, on: element)
        ),
              let windowFrame = elementFrame(window),
              let elementFrame = elementFrame(element),
              elementFrame.width >= 1,
              elementFrame.height >= 1,
              windowFrame.insetBy(dx: -0.5, dy: -0.5).contains(elementFrame),
              hasVisibleAncestry(element, through: window)
        else { return false }
        return true
    }

    static func matchesSafeVisualComposer(
        role: String?,
        subrole: String?,
        isBrowserAddressField: Bool,
        isEnabled: Bool?,
        value: String?,
        requiredValue: String,
        isValueSettable: Bool
    ) -> Bool {
        (role == kAXTextAreaRole as String || role == kAXTextFieldRole as String)
            && subrole != kAXSecureTextFieldSubrole as String
            && !isBrowserAddressField
            && isEnabled == true
            && value.map({ utf8Matches($0, requiredValue) }) == true
            && isValueSettable
    }

    /// Converts one observation-bound visual point into focus on an empty
    /// editable element. It never invokes AXPress and rejects buttons, address
    /// bars, secure fields, non-empty fields, and anything outside the exact
    /// selected browser window.
    static func focusGroundedEmptyComposer(
        action: GroundedWorkflowAction,
        groundedAgainst groundingObservation: WorkflowObservation,
        currentObservation: WorkflowObservation,
        destination: ComputerUseWindowOption,
        documentIdentity: String,
        actuator: any DraftPostComposerActuating
    ) throws {
        guard action.observationID == groundingObservation.id,
              groundingObservation.target == currentObservation.target,
              currentObservation.target.bundleIdentifier
                == destination.selection.bundleIdentifier,
              currentObservation.target.processIdentifier
                == destination.selection.processIdentifier,
              currentObservation.target.processLaunchDate
                == destination.selection.processLaunchDate,
              currentObservation.target.windowIdentifier
                == String(destination.selection.windowID),
              case .click(let normalizedX, let normalizedY) = action.payload,
              normalizedX > 0, normalizedX < 1,
              normalizedY > 0, normalizedY < 1
        else { throw DraftPostFlowFailure.composerMissing }

        let frame = currentObservation.target.windowFrame
        let point = CGPoint(
            x: frame.x + normalizedX * frame.width,
            y: frame.y + normalizedY * frame.height
        )
        let window = try exactFocusedBrowserWindow(
            destination,
            documentIdentity: documentIdentity
        )
        if groundingObservation.contentRevision != currentObservation.contentRevision {
            guard let focused = focusedEditableElement(
                in: window,
                browserBundleIdentifier: destination.selection.bundleIdentifier
            ), let focusedFrame = elementFrame(focused),
                isWithinGroundingTolerance(
                    point: point,
                    elementFrame: focusedFrame,
                    windowFrame: CGRect(
                        x: currentObservation.target.windowFrame.x,
                        y: currentObservation.target.windowFrame.y,
                        width: currentObservation.target.windowFrame.width,
                        height: currentObservation.target.windowFrame.height
                    )
                )
            else { throw DraftPostFlowFailure.composerMissing }
        }
        let application = applicationElement(destination.selection.processIdentifier)
        var hit: AXUIElement?
        guard AXUIElementCopyElementAtPosition(
            application,
            Float(point.x),
            Float(point.y),
            &hit
        ) == .success,
            let hit
        else { throw DraftPostFlowFailure.composerMissing }
        guard let composer = try groundedEditable(
            from: hit,
            at: point,
            in: window,
            browserBundleIdentifier: destination.selection.bundleIdentifier
        )
        else { throw DraftPostFlowFailure.composerMissing }

        try actuator.focusComposer(composer)
        let reboundWindow = try exactFocusedBrowserWindow(
            destination,
            documentIdentity: documentIdentity
        )
        guard CFEqual(window, reboundWindow) else {
            throw DraftPostFlowFailure.focusChanged
        }
        guard let focused = focusedEditableElement(
                in: reboundWindow,
                browserBundleIdentifier: destination.selection.bundleIdentifier
              ), CFEqual(composer, focused)
        else { throw DraftPostFlowFailure.focusChanged }
    }

    /// Prefer an exact AX hit. When a small local model lands on the label or
    /// border next to the requested field, recover only if exactly one safe,
    /// empty editor is within a tightly bounded window-relative neighborhood.
    /// This never turns the model point into an unconstrained screen click.
    private static func groundedEditable(
        from hit: AXUIElement,
        at point: CGPoint,
        in window: AXUIElement,
        browserBundleIdentifier: String
    ) throws -> AXUIElement? {
        let safeEditors = try editableElements(in: window).filter { element in
            isSafeEmptyEditable(
                element,
                in: window,
                browserBundleIdentifier: browserBundleIdentifier
            )
        }
        guard authorizesUniqueVisualEditor(candidateCount: safeEditors.count),
              let safeEditor = safeEditors.first
        else { return nil }

        if let exact = editableAncestor(
            from: hit,
            through: window,
            browserBundleIdentifier: browserBundleIdentifier
        ), CFEqual(exact, safeEditor) {
            return safeEditor
        }
        guard let windowFrame = elementFrame(window) else { return nil }
        guard let frame = elementFrame(safeEditor),
              isWithinGroundingTolerance(
                point: point,
                elementFrame: frame,
                windowFrame: windowFrame
              )
        else { return nil }
        return safeEditor
    }

    static func authorizesUniqueVisualEditor(candidateCount: Int) -> Bool {
        candidateCount == 1
    }

    static func isWithinGroundingTolerance(
        point: CGPoint,
        elementFrame: CGRect,
        windowFrame: CGRect
    ) -> Bool {
        guard windowFrame.width > 0, windowFrame.height > 0,
              windowFrame.contains(point),
              windowFrame.insetBy(dx: -0.5, dy: -0.5).contains(elementFrame)
        else { return false }
        let horizontalTolerance = min(64, windowFrame.width * 0.06)
        let verticalTolerance = min(64, windowFrame.height * 0.06)
        return elementFrame.insetBy(
            dx: -horizontalTolerance,
            dy: -verticalTolerance
        ).contains(point)
    }

    private static func editableAncestor(
        from element: AXUIElement,
        through window: AXUIElement,
        browserBundleIdentifier: String
    ) -> AXUIElement? {
        var current = element
        var visited = Set<AXUIElement>()
        for _ in 0 ..< 32 {
            guard visited.insert(current).inserted else { return nil }
            if isSafeEmptyEditable(
                current,
                in: window,
                browserBundleIdentifier: browserBundleIdentifier
            ) {
                return current
            }
            if CFEqual(current, window) { return nil }
            var parentValue: CFTypeRef?
            guard AXUIElementCopyAttributeValue(
                current,
                kAXParentAttribute as CFString,
                &parentValue
            ) == .success,
                let parentValue,
                CFGetTypeID(parentValue) == AXUIElementGetTypeID()
            else { return nil }
            current = unsafeDowncast(parentValue, to: AXUIElement.self)
        }
        return nil
    }

    static func isExplicitComposerLabel(_ raw: String) -> Bool {
        let label = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "’", with: "'")
        return [
            "post text",
            "compose post",
            "write your post",
            "what is happening?",
            "what's happening?",
            "what is on your mind?",
            "what's on your mind?",
        ].contains(label)
    }

    static func uniqueDraft(in candidates: [String?]) throws -> String {
        guard candidates.count == 1 else {
            if candidates.isEmpty {
                throw DraftPostFlowFailure.draftMissing
            }
            throw DraftPostFlowFailure.draftAmbiguous
        }
        guard let draft = candidates[0],
              !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw DraftPostFlowFailure.draftMissing
        }
        return draft
    }

    private static func stringAttribute(
        _ name: CFString,
        from element: AXUIElement
    ) -> String? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, name, &value) == .success else {
            return nil
        }
        return value as? String
    }

    private static func elementFrame(_ element: AXUIElement) -> CGRect? {
        var positionValue: CFTypeRef?
        var sizeValue: CFTypeRef?
        guard AXUIElementCopyAttributeValue(
            element,
            kAXPositionAttribute as CFString,
            &positionValue
        ) == .success,
            AXUIElementCopyAttributeValue(
                element,
                kAXSizeAttribute as CFString,
                &sizeValue
            ) == .success,
            let positionValue,
            let sizeValue,
            CFGetTypeID(positionValue) == AXValueGetTypeID(),
            CFGetTypeID(sizeValue) == AXValueGetTypeID()
        else { return nil }
        var origin = CGPoint.zero
        var size = CGSize.zero
        guard AXValueGetValue(
            unsafeDowncast(positionValue, to: AXValue.self),
            .cgPoint,
            &origin
        ), AXValueGetValue(
            unsafeDowncast(sizeValue, to: AXValue.self),
            .cgSize,
            &size
        ) else { return nil }
        return CGRect(origin: origin, size: size)
    }
}

@MainActor
@Observable
final class DraftPostFlowViewModel {
    enum Phase: Equatable {
        case loading
        case ready
        case running
        case stopping
        case readyForReview(DraftPostFlowMetrics)
        case failed(DraftPostFlowFailure, DraftPostFlowMetrics?)
    }

    var phase: Phase = .loading
    var windows: [ComputerUseWindowOption] = []
    var sourceID: String?
    var destinationID: String?
    private let catalog: any ComputerUseWindowListing
    private let driver: any DraftPostFlowDriving
    private var runTask: Task<Void, Never>?
    private var loadGeneration = 0

    init(
        catalog: any ComputerUseWindowListing = MacOSComputerUseWindowCatalog(),
        driver: any DraftPostFlowDriving = MacOSDraftPostFlowDriver()
    ) {
        self.catalog = catalog
        self.driver = driver
    }

    var sourceOptions: [ComputerUseWindowOption] {
        windows.filter {
            $0.selection.bundleIdentifier == "com.apple.TextEdit"
                && !$0.windowTitle.trimmingCharacters(
                    in: .whitespacesAndNewlines
                ).isEmpty
        }
    }

    var destinationOptions: [ComputerUseWindowOption] {
        windows.filter {
            MacOSDraftPostFlowDriver.browserBundles.contains(
                $0.selection.bundleIdentifier
            ) && !$0.windowTitle.trimmingCharacters(
                in: .whitespacesAndNewlines
            ).isEmpty
        }
    }

    var canRun: Bool {
        guard phase == .ready,
              let sourceID,
              let destinationID,
              sourceID != destinationID
        else { return false }
        return sourceOptions.contains(where: { $0.id == sourceID })
            && destinationOptions.contains(where: { $0.id == destinationID })
    }

    var isActive: Bool {
        phase == .running || phase == .stopping
    }

    func load() async {
        loadGeneration += 1
        let generation = loadGeneration
        phase = .loading
        do {
            let refreshedWindows = try await catalog.windows()
            guard generation == loadGeneration else { return }
            windows = refreshedWindows
            sourceID = nil
            destinationID = nil
            phase = .ready
        } catch let error as ComputerUseWindowCatalogError {
            guard generation == loadGeneration else { return }
            switch error {
            case .permissionsMissing:
                phase = .failed(.permissionMissing, nil)
            case .unavailable:
                phase = .failed(.dependencyFailure, nil)
            }
        } catch {
            guard generation == loadGeneration else { return }
            phase = .failed(.dependencyFailure, nil)
        }
    }

    func run() {
        guard phase == .ready, runTask == nil,
              let source = windows.first(where: { $0.id == sourceID }),
              let destination = windows.first(where: { $0.id == destinationID })
        else { return }
        phase = .running
        let coordinator = DraftPostFlowCoordinator(driver: driver)
        runTask = Task { [weak self] in
            let outcome = await coordinator.run(source: source, destination: destination)
            // A cancellation request may race with the final synchronous write.
            // Report the driver's definitive outcome instead of claiming that
            // cancellation prevented a mutation when it did not.
            guard let self else { return }
            self.runTask = nil
            switch outcome {
            case .readyForReview(let metrics):
                self.phase = .readyForReview(metrics)
            case .failed(let failure, let metrics):
                self.phase = .failed(failure, metrics)
            }
        }
    }

    func stop() {
        runTask?.cancel()
        if phase == .running {
            phase = .stopping
        }
    }
}
