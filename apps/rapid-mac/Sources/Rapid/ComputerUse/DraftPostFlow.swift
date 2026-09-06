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

    var isRecoverable: Bool {
        switch self {
        case .targetUnavailable, .focusChanged:
            true
        default:
            false
        }
    }

    var userMessage: String {
        switch self {
        case .sourceIsNotTextEdit: "Choose a TextEdit window as the draft source."
        case .destinationIsNotBrowser: "Choose a supported browser window as the destination."
        case .targetUnavailable: "A selected window is no longer available."
        case .focusChanged: "Rapid could not safely focus the selected window."
        case .draftMissing: "The selected TextEdit document has no readable draft."
        case .draftAmbiguous: "More than one TextEdit document editor was found. Close auxiliary editors and try again."
        case .draftTooLarge: "The draft is too large for this preview (64 KB maximum)."
        case .composerMissing: "No editable post composer was found in the browser window."
        case .composerAmbiguous: "More than one possible composer was found. Close other editors and try again."
        case .composerNotEmpty: "The browser composer already contains text. Clear it before running this flow."
        case .writeRejected: "The browser rejected the local text update."
        case .verificationFailed: "The browser content did not match the TextEdit draft."
        case .permissionMissing: "Screen Recording and Accessibility access are required."
        case .cancelled: "The flow was stopped."
        case .dependencyFailure: "The flow stopped because a local system operation failed."
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
        var metrics = DraftPostFlowMetrics()
        for attempt in 1 ... maximumAttempts {
            if Task.isCancelled {
                return .failed(.cancelled, metrics)
            }
            metrics.attempts = attempt
            do {
                try await driver.transferDraft(from: source, to: destination)
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
                // retry. An unknown adapter failure could have happened after
                // a local mutation, so it must fail closed.
                return .failed(.dependencyFailure, metrics)
            }
        }
        return .failed(.targetUnavailable, metrics)
    }
}

/// The complete mutation authority granted to the draft flow. Publishing is
/// structurally impossible because this capability can only focus a composer
/// and set its draft value.
@MainActor
protocol DraftPostComposerActuating: Sendable {
    func focusComposer(_ composer: AXUIElement) throws
    func setDraft(_ draft: String, on composer: AXUIElement) throws
}

struct AXDraftPostComposerActuator: DraftPostComposerActuating {
    func focusComposer(_ composer: AXUIElement) throws {
        guard AXUIElementSetAttributeValue(
            composer,
            kAXFocusedAttribute as CFString,
            kCFBooleanTrue
        ) == .success else {
            throw DraftPostFlowFailure.writeRejected
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
struct MacOSDraftPostFlowDriver: DraftPostFlowDriving {
    static let maximumDraftBytes = 65_536
    private static let textEditBundle = "com.apple.TextEdit"
    static let browserBundles: Set<String> = [
        "com.apple.Safari",
    ]
    private let actuator: any DraftPostComposerActuating

    init(actuator: any DraftPostComposerActuating = AXDraftPostComposerActuator()) {
        self.actuator = actuator
    }

    func transferDraft(
        from source: ComputerUseWindowOption,
        to destination: ComputerUseWindowOption
    ) async throws {
        guard source.selection.bundleIdentifier == Self.textEditBundle else {
            throw DraftPostFlowFailure.sourceIsNotTextEdit
        }
        guard Self.browserBundles.contains(destination.selection.bundleIdentifier) else {
            throw DraftPostFlowFailure.destinationIsNotBrowser
        }
        guard MacAutomationPermissions.snapshot().isReadyForComputerUse else {
            throw DraftPostFlowFailure.permissionMissing
        }

        let documentIdentity = try await browserDocumentIdentity(in: destination)
        let draft = try await readDraft(from: source.selection)
        guard !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DraftPostFlowFailure.draftMissing
        }
        guard draft.utf8.count <= Self.maximumDraftBytes else {
            throw DraftPostFlowFailure.draftTooLarge
        }
        try await writeAndVerify(
            draft,
            from: source.selection,
            to: destination,
            documentIdentity: documentIdentity
        )
        // The destination may now be mutated. Every remaining observation is
        // therefore terminal on failure: recovery must never replay the write.
        do {
            let finalSource = try await readDraft(from: source.selection)
            guard Self.utf8Matches(finalSource, draft) else {
                throw DraftPostFlowFailure.verificationFailed
            }
            try await verifyComposer(
                draft,
                in: destination,
                documentIdentity: documentIdentity
            )
        } catch {
            throw DraftPostFlowFailure.verificationFailed
        }
    }

    private func readDraft(from selection: ComputerUseWindowSelection) async throws -> String {
        try await focus(selection)
        return try await MainActor.run {
            let window = try Self.exactFocusedWindow(selection)
            let candidates = Self.editableElements(in: window)
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

    private func writeAndVerify(
        _ draft: String,
        from source: ComputerUseWindowSelection,
        to destination: ComputerUseWindowOption,
        documentIdentity: String
    ) async throws {
        let selection = destination.selection
        try await focus(selection)
        try Task.checkCancellation()
        try await MainActor.run {
            let window = try Self.exactFocusedBrowserWindow(
                destination,
                documentIdentity: documentIdentity
            )
            let composer = try Self.uniqueComposer(in: window)
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
                let currentComposer = try Self.uniqueComposer(in: currentWindow)
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
            let currentComposer = try Self.uniqueComposer(in: currentWindow)
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
            let currentSource = try Self.readDraftWithoutFocusing(from: source)
            guard Self.utf8Matches(currentSource, draft) else {
                throw DraftPostFlowFailure.verificationFailed
            }
            // This is the final cancellation boundary before the only content
            // mutation. No suspension occurs between this check and the write.
            try Task.checkCancellation()
            try actuator.setDraft(draft, on: currentComposer)
            // Once content may have changed, no focus/window error is safe to
            // retry. Collapse every post-mutation observation failure into a
            // terminal verification failure.
            try Self.verifyAfterMutation {
                let verifiedWindow = try Self.exactFocusedBrowserWindow(
                    destination,
                    documentIdentity: documentIdentity
                )
                let verifiedComposer = try Self.uniqueComposer(in: verifiedWindow)
                return CFEqual(currentComposer, verifiedComposer)
                    && Self.stringAttribute(
                        kAXValueAttribute as CFString,
                        from: verifiedComposer
                      ).map { Self.utf8Matches($0, draft) } == true
            }
        }
    }

    static func utf8Matches(_ lhs: String, _ rhs: String) -> Bool {
        lhs.utf8.elementsEqual(rhs.utf8)
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

    private func verifyComposer(
        _ draft: String,
        in destination: ComputerUseWindowOption,
        documentIdentity: String
    ) async throws {
        let selection = destination.selection
        try await focus(selection)
        try await MainActor.run {
            let window = try Self.exactFocusedBrowserWindow(
                destination,
                documentIdentity: documentIdentity
            )
            let composer = try Self.uniqueComposer(in: window)
            guard Self.stringAttribute(
                kAXValueAttribute as CFString,
                from: composer
            ).map({ Self.utf8Matches($0, draft) }) == true else {
                throw DraftPostFlowFailure.verificationFailed
            }
        }
    }

    @MainActor
    private static func exactFocusedBrowserWindow(
        _ destination: ComputerUseWindowOption,
        documentIdentity: String
    ) throws -> AXUIElement {
        let window = try exactFocusedWindow(destination.selection)
        guard browserDocumentMatches(
            currentTitle: stringAttribute(
            kAXTitleAttribute as CFString,
            from: window
            ),
            selectedTitle: destination.windowTitle
        ), try currentBrowserDocumentIdentity(in: window) == documentIdentity
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

    private func browserDocumentIdentity(
        in destination: ComputerUseWindowOption
    ) async throws -> String {
        try await MainActor.run {
            let selection = destination.selection
            guard let running = NSRunningApplication(
                processIdentifier: selection.processIdentifier
            ), running.bundleIdentifier == selection.bundleIdentifier,
                running.launchDate == selection.processLaunchDate
            else { throw DraftPostFlowFailure.targetUnavailable }
            let application = AXUIElementCreateApplication(selection.processIdentifier)
            guard let window = Self.window(matching: selection, in: application),
                  Self.browserDocumentMatches(
                    currentTitle: Self.stringAttribute(
                        kAXTitleAttribute as CFString,
                        from: window
                    ),
                    selectedTitle: destination.windowTitle
                  )
            else { throw DraftPostFlowFailure.focusChanged }
            return try Self.currentBrowserDocumentIdentity(in: window)
        }
    }

    @MainActor
    private static func currentBrowserDocumentIdentity(
        in window: AXUIElement
    ) throws -> String {
        guard let windowFrame = elementFrame(window) else {
            throw DraftPostFlowFailure.composerAmbiguous
        }
        let addressFields = allElements(in: window).filter { element in
            guard stringAttribute(kAXIdentifierAttribute as CFString, from: element)
                == "WEB_BROWSER_ADDRESS_AND_SEARCH_FIELD",
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

    @MainActor
    private static func readDraftWithoutFocusing(
        from selection: ComputerUseWindowSelection
    ) throws -> String {
        guard let running = NSRunningApplication(
            processIdentifier: selection.processIdentifier
        ), running.bundleIdentifier == selection.bundleIdentifier,
            running.launchDate == selection.processLaunchDate
        else { throw DraftPostFlowFailure.targetUnavailable }
        let application = AXUIElementCreateApplication(selection.processIdentifier)
        guard let window = window(matching: selection, in: application) else {
            throw DraftPostFlowFailure.targetUnavailable
        }
        let candidates = editableElements(in: window)
            .filter {
                stringAttribute(kAXRoleAttribute as CFString, from: $0)
                    == kAXTextAreaRole as String
            }
            .map { stringAttribute(kAXValueAttribute as CFString, from: $0) }
        return try uniqueDraft(in: candidates)
    }

    private func focus(_ selection: ComputerUseWindowSelection) async throws {
        try Task.checkCancellation()
        try await MainActor.run {
            guard let app = NSRunningApplication(processIdentifier: selection.processIdentifier),
                  app.bundleIdentifier == selection.bundleIdentifier,
                  app.launchDate == selection.processLaunchDate
            else { throw DraftPostFlowFailure.targetUnavailable }
            app.activate()
            let application = AXUIElementCreateApplication(selection.processIdentifier)
            guard let window = Self.window(matching: selection, in: application),
                  AXUIElementPerformAction(window, kAXRaiseAction as CFString) == .success
            else { throw DraftPostFlowFailure.targetUnavailable }
        }
        try await Task.sleep(for: .milliseconds(180))
        try Task.checkCancellation()
        try await MainActor.run {
            _ = try Self.exactFocusedWindow(selection)
        }
    }

    @MainActor
    private static func exactFocusedWindow(
        _ selection: ComputerUseWindowSelection
    ) throws -> AXUIElement {
        guard let running = NSRunningApplication(
            processIdentifier: selection.processIdentifier
        ),
            running.bundleIdentifier == selection.bundleIdentifier,
            running.launchDate == selection.processLaunchDate,
            NSWorkspace.shared.frontmostApplication?.processIdentifier
                == selection.processIdentifier
        else { throw DraftPostFlowFailure.focusChanged }
        let application = AXUIElementCreateApplication(selection.processIdentifier)
        guard let selected = window(matching: selection, in: application) else {
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

    @MainActor
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

    @MainActor
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

    @MainActor
    private static func allElements(in root: AXUIElement) -> [AXUIElement] {
        var queue: [(AXUIElement, Int)] = [(root, 0)]
        var result: [AXUIElement] = []
        var visited = Set<AXUIElement>()
        var cursor = 0
        while cursor < queue.count, visited.count < 2_048 {
            let (element, depth) = queue[cursor]
            cursor += 1
            guard visited.insert(element).inserted else { continue }
            result.append(element)
            guard depth < 32 else { continue }
            var value: CFTypeRef?
            if AXUIElementCopyAttributeValue(
                element,
                kAXChildrenAttribute as CFString,
                &value
            ) == .success,
                let children = value as? [AXUIElement]
            {
                queue.append(contentsOf: children.map { ($0, depth + 1) })
            }
        }
        return result
    }

    @MainActor
    private static func editableElements(in root: AXUIElement) -> [AXUIElement] {
        allElements(in: root).filter { element in
            let role = stringAttribute(kAXRoleAttribute as CFString, from: element)
            let subrole = stringAttribute(kAXSubroleAttribute as CFString, from: element)
            return (role == kAXTextAreaRole as String || role == kAXTextFieldRole as String)
                && subrole != kAXSecureTextFieldSubrole as String
        }
    }

    @MainActor
    private static func isExplicitComposer(
        _ element: AXUIElement,
        windowFrame: CGRect
    ) -> Bool {
        guard boolAttribute(kAXEnabledAttribute as CFString, from: element) == true,
              boolAttribute("AXHidden" as CFString, from: element) != true,
              let frame = elementFrame(element), frame.width >= 1, frame.height >= 1,
              windowFrame.intersects(frame)
        else { return false }
        var settable: DarwinBoolean = false
        guard AXUIElementIsAttributeSettable(
            element,
            kAXValueAttribute as CFString,
            &settable
        ) == .success, settable.boolValue else { return false }
        let fields = [
            stringAttribute(kAXTitleAttribute as CFString, from: element),
            stringAttribute(kAXDescriptionAttribute as CFString, from: element),
            stringAttribute(kAXHelpAttribute as CFString, from: element),
            stringAttribute("AXPlaceholderValue" as CFString, from: element),
        ].compactMap { $0 }
        return fields.contains(where: isExplicitComposerLabel)
    }

    @MainActor
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

    @MainActor
    private static func uniqueComposer(in window: AXUIElement) throws -> AXUIElement {
        guard let windowFrame = elementFrame(window) else {
            throw DraftPostFlowFailure.composerMissing
        }
        let matches = editableElements(in: window).filter {
            isExplicitComposer($0, windowFrame: windowFrame)
        }
        guard let match = matches.first else {
            throw DraftPostFlowFailure.composerMissing
        }
        if matches.count > 1 {
            throw DraftPostFlowFailure.composerAmbiguous
        }
        return match
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

    @MainActor
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

    @MainActor
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
        case failed(String, DraftPostFlowMetrics?)
    }

    var phase: Phase = .loading
    var windows: [ComputerUseWindowOption] = []
    var sourceID: String?
    var destinationID: String?
    private let catalog: any ComputerUseWindowListing
    private let driver: any DraftPostFlowDriving
    private var runTask: Task<Void, Never>?

    init(
        catalog: any ComputerUseWindowListing = MacOSComputerUseWindowCatalog(),
        driver: any DraftPostFlowDriving = MacOSDraftPostFlowDriver()
    ) {
        self.catalog = catalog
        self.driver = driver
    }

    var sourceOptions: [ComputerUseWindowOption] {
        windows.filter { $0.selection.bundleIdentifier == "com.apple.TextEdit" }
    }

    var destinationOptions: [ComputerUseWindowOption] {
        windows.filter {
            MacOSDraftPostFlowDriver.browserBundles.contains(
                $0.selection.bundleIdentifier
            )
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
        phase = .loading
        do {
            windows = try await catalog.windows()
            sourceID = nil
            destinationID = nil
            phase = .ready
        } catch let error as ComputerUseWindowCatalogError {
            switch error {
            case .permissionsMissing:
                phase = .failed(DraftPostFlowFailure.permissionMissing.userMessage, nil)
            case .unavailable:
                phase = .failed("Rapid could not list the available windows.", nil)
            }
        } catch {
            phase = .failed("Rapid could not list the available windows.", nil)
        }
    }

    func run() {
        guard let source = windows.first(where: { $0.id == sourceID }),
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
                self.phase = .failed(failure.userMessage, metrics)
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
