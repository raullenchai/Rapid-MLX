import Foundation
import Security
import Testing
@testable import Rapid

private final class SettingsKeychainProbe: KeychainStoring, @unchecked Sendable {
    private let lock = NSLock()
    private var reads = 0
    var result: KeychainReadResult = .missing

    var readCount: Int { lock.withLock { reads } }

    func read(account: String) -> String? { nil }
    func readWithoutUserInteraction(account: String) -> KeychainReadResult {
        lock.withLock { reads += 1 }
        return result
    }
    func write(account: String, secret: String) -> Bool { true }
    func delete(account: String) -> Bool { true }
}

private final class SettingsKeychainItems: KeychainItemAccessing, @unchecked Sendable {
    private let lock = NSLock()
    private var states: [String: KeychainReadResult]
    private let rejectedServices: Set<String>

    init(states: [String: KeychainReadResult], rejectedServices: Set<String>) {
        self.states = states
        self.rejectedServices = rejectedServices
    }

    func query(account: String, service: String) -> KeychainReadResult {
        lock.withLock { states["\(service)|\(account)"] ?? .missing }
    }

    func upsert(account: String, service: String, secret: String) -> Bool {
        lock.withLock {
            guard !rejectedServices.contains(service) else { return false }
            states["\(service)|\(account)"] = .found(secret)
            return true
        }
    }

    func remove(account: String, service: String) -> Bool {
        lock.withLock {
            guard !rejectedServices.contains(service) else { return false }
            states.removeValue(forKey: "\(service)|\(account)")
            return true
        }
    }
}

@MainActor
@Suite("Settings web-search key draft commit")
struct SettingsWebSearchKeyDraftTests {
    @Test("Constructing the Tools configuration never reads Keychain")
    func constructionIsKeychainLazy() {
        let keychain = SettingsKeychainProbe()
        _ = WebSearchConfig(defaults: .standard, keychain: keychain)
        #expect(keychain.readCount == 0)
    }

    @Test("An explicit lazy probe preserves the denied state for inline recovery")
    func deniedProbeBecomesInlineState() async {
        let keychain = SettingsKeychainProbe()
        keychain.result = .unavailable
        let config = WebSearchConfig(defaults: .standard, keychain: keychain)

        #expect(config.cachedKeyState(for: .parallel) == .unknown)
        await config.prefetchAPIKey(for: .parallel)

        #expect(keychain.readCount == 1)
        #expect(config.cachedKeyState(for: .parallel) == .unavailable)
        #expect(config.apiKey(for: .parallel) == nil)
        #expect(keychain.readCount == 1, "the denied result is cached instead of repeatedly consulting Security.framework")
    }

    @Test("Developer ID releases and development builds use different services")
    func codeIdentityNamespacesAreSeparated() {
        let release = SystemKeychain.serviceNamespace(
            teamIdentifier: "TEAM123",
            isDeveloperIDApplication: true
        )
        let development = SystemKeychain.serviceNamespace(
            teamIdentifier: "TEAM123",
            isDeveloperIDApplication: false
        )

        #expect(release == "com.rapidmlx.rapid.api-keys.release.TEAM123")
        #expect(development == "com.rapidmlx.rapid.api-keys.development")
        #expect(release != development)
    }

    @Test("Developer ID Application uses a valid system code requirement")
    func developerIDRequirementCompiles() {
        #expect(SystemKeychain.developerIDApplicationRequirementCompiles())
    }

    @Test("Developer ID namespace detection never revalidates the app bundle")
    func developerIDNamespaceCheckSkipsSealedResourcesAndExecutable() {
        let flags = SystemKeychain.developerIDApplicationValidationFlags

        #expect(flags.contains(SecCSFlags(rawValue: kSecCSDoNotValidateResources)))
        #expect(flags.contains(SecCSFlags(rawValue: kSecCSDoNotValidateExecutable)))
    }

    @Test("Explicit save recovers from an inaccessible current-identity item")
    func inaccessibleCurrentSlotUsesRecoverySlot() {
        let account = "rapid.web-search.parallel"
        let primary = "test.release"
        let items = SettingsKeychainItems(
            states: ["\(primary)|\(account)": .unavailable],
            rejectedServices: [primary]
        )
        let store = SystemKeychain(items: items, primaryService: primary)

        #expect(store.readWithoutUserInteraction(account: account) == .unavailable)
        #expect(store.write(account: account, secret: "replacement-key"))
        #expect(store.readWithoutUserInteraction(account: account) == .found("replacement-key"))
    }

    @Test("Direct Swift Keychain I/O calls route through the coordinated adapter")
    func productionKeychainRoutingIsCentralized() throws {
        let source = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources", isDirectory: true)
        let files = try #require(FileManager.default.enumerator(
            at: source, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]
        ))
        let operations = [
            "SecItemCopyMatching(", "SecItemUpdate(", "SecItemAdd(", "SecItemDelete(",
            "SecKeychainFindGenericPassword(", "SecKeychainAddGenericPassword(",
            "SecKeychainFindInternetPassword(", "SecKeychainAddInternetPassword(",
            "SecKeychainItemCopyContent(", "SecKeychainItemModifyContent(", "SecKeychainItemDelete("
        ]
        for case let url as URL in files where url.pathExtension == "swift" {
            guard url.path != source.appendingPathComponent("Rapid/Tools/KeychainStore.swift").path else { continue }
            let code = SourceGuardSupport.canonicalSource(
                try String(contentsOf: url, encoding: .utf8), literals: .erase
            )
            for operation in operations {
                #expect(!code.contains(operation), "Uncoordinated Keychain operation in \(url.lastPathComponent): \(operation)")
            }
        }
    }

    @Test("Legacy Keychain operations suppress authentication and restore the previous UI policy")
    func legacyAuthenticationPolicyIsScoped() {
        var original: DarwinBoolean = false
        #expect(SecKeychainGetUserInteractionAllowed(&original) == errSecSuccess)
        let allowedDuringCall = SecurityKeychainItems.withoutUserInteraction(failure: true) {
            var allowed: DarwinBoolean = true
            #expect(SecKeychainGetUserInteractionAllowed(&allowed) == errSecSuccess)
            return allowed.boolValue
        }
        #expect(!allowedDuringCall)
        var restored: DarwinBoolean = false
        #expect(SecKeychainGetUserInteractionAllowed(&restored) == errSecSuccess)
        #expect(restored.boolValue == original.boolValue)
    }

    @Test("A missing or locked default Keychain refuses insertion without a system dialog")
    func unavailableDefaultKeychainDoesNotInsert() {
        let account = "rapid.keychain-insertion-test.\(UUID().uuidString)"
        let service = "rapid.keychain-insertion-test.\(UUID().uuidString)"
        let items = SecurityKeychainItems(insertionKeychain: { nil })
        #expect(!items.upsert(account: account, service: service, secret: "disposable-test-value"))
        #expect(items.query(account: account, service: service) == .missing)
    }

    @Test("Security status distinguishes a missing item from denied non-interactive access")
    func securityReadStatusMapping() {
        #expect(SecurityKeychainItems.readResult(status: errSecItemNotFound, data: nil) == .missing)
        #expect(SecurityKeychainItems.readResult(status: errSecInteractionNotAllowed, data: nil) == .unavailable)
        #expect(SecurityKeychainItems.readResult(
            status: errSecSuccess,
            data: Data("saved-key".utf8)
        ) == .found("saved-key"))
    }

    @Test("An inaccessible current item never falls back to a stale legacy credential")
    func inaccessibleCurrentSlotStopsLegacyFallback() {
        let account = "rapid.web-search.parallel"
        let primary = "test.release"
        let legacy = "test.legacy"
        let items = SettingsKeychainItems(
            states: [
                "\(primary)|\(account)": .unavailable,
                "\(legacy)|\(account)": .found("stale-key"),
            ],
            rejectedServices: []
        )
        let store = SystemKeychain(
            items: items,
            primaryService: primary,
            legacyMigrationService: legacy
        )

        #expect(store.readWithoutUserInteraction(account: account) == .unavailable)
    }

    @Test("Clearing removes migrated legacy and recovery credentials")
    func clearDeletesEveryAccessibleCredentialSlot() {
        let account = "rapid.web-search.parallel"
        let primary = "test.release"
        let legacy = "test.legacy"
        let items = SettingsKeychainItems(
            states: [
                "\(primary).recovery|\(account)": .found("replacement-key"),
                "\(legacy)|\(account)": .found("migrated-key"),
            ],
            rejectedServices: []
        )
        let store = SystemKeychain(
            items: items,
            primaryService: primary,
            legacyMigrationService: legacy
        )

        #expect(store.delete(account: account))
        #expect(store.readWithoutUserInteraction(account: account) == .missing)
    }

    @Test("A denied removal is masked but never reported as erased")
    func deniedClearUsesTombstoneAndReportsFailure() {
        let account = "rapid.web-search.parallel"
        let primary = "test.release"
        let legacy = "test.legacy"
        let items = SettingsKeychainItems(
            states: ["\(legacy)|\(account)": .found("inaccessible-key")],
            rejectedServices: [legacy]
        )
        let store = SystemKeychain(
            items: items,
            primaryService: primary,
            legacyMigrationService: legacy
        )

        #expect(!store.delete(account: account))
        #expect(store.read(account: account) == nil)
    }

    @Test("Visible backend key status has truthful copy for every cached state")
    func visibleBackendKeyStatusCopy() {
        #expect(SettingsToolsPanel.keyStatusCaption(
            for: .parallel,
            state: .unknown
        ) == "Checking saved key…")
        #expect(SettingsToolsPanel.keyStatusCaption(
            for: .parallel,
            state: .present("secret-never-rendered")
        ) == "A key is stored for Parallel.")
        #expect(SettingsToolsPanel.keyStatusCaption(
            for: .parallel,
            state: .absent
        ) == "No key stored — searches fall back to Keenable until you save one.")
        #expect(SettingsToolsPanel.keyStatusCaption(
            for: .parallel,
            state: .unavailable
        ) == "The saved key can’t be accessed. Enter it again and save to replace it.")
    }

    @Test("Untouched empty SecureField does not clear an existing stored key")
    func untouchedDraftIsUnchanged() {
        #expect(SettingsView.webSearchKeyCommitAction(draft: "", wasEdited: false) == .unchanged)
    }

    @Test("Edited whitespace draft clears the stored key")
    func editedWhitespaceClears() {
        #expect(SettingsView.webSearchKeyCommitAction(draft: "  \n\t ", wasEdited: true) == .clear)
    }

    @Test("Edited key trims before saving")
    func editedKeyTrimsBeforeSave() {
        #expect(SettingsView.webSearchKeyCommitAction(draft: "  BSA-key\n", wasEdited: true) == .save("BSA-key"))
    }

    // v0.6.7 codex r1 P2 — a failed Keychain write shows a
    // "Couldn't save, try again" banner; if the SecureField draft
    // is wiped at the same time the user has nothing to retry with
    // (the SecureField never echoes the existing key back, and the
    // Save button is gated on the dirty flag). Pin both branches.

    @Test("Successful write resets the draft + dirty flag")
    func successfulWriteResetsDraft() {
        #expect(SettingsView.shouldResetWebSearchKeyDraftAfterCommit(keychainWriteSucceeded: true))
    }

    @Test("Failed write keeps the draft so the user can retry without re-pasting")
    func failedWriteKeepsDraftForRetry() {
        #expect(!SettingsView.shouldResetWebSearchKeyDraftAfterCommit(keychainWriteSucceeded: false),
                "Without this, the 'try again' advice in the banner is impossible to follow — the SecureField never echoes the existing key, so the retry has nothing to commit.")
    }
}

/// v0.6.7 — pins the Save-button feedback contract. The transient
/// banner in Settings → Web Search reads its state off
/// ``SettingsView.WebSearchKeySaveFeedback``; the cases must remain
/// distinguishable by generation so back-to-back identical-outcome
/// Saves still retrigger the auto-dismiss task.
@MainActor
@Suite("Settings web-search key Save feedback")
struct SettingsWebSearchKeySaveFeedbackTests {
    @Test("Same kind with different generations compares non-equal")
    func sameKindBumpsViaGeneration() {
        let a = SettingsView.WebSearchKeySaveFeedback.saved(generation: 1)
        let b = SettingsView.WebSearchKeySaveFeedback.saved(generation: 2)
        #expect(a != b,
                "Without the generation bump, SwiftUI .task(id:) would see no change between back-to-back Saves and the auto-dismiss timer would never reschedule.")
    }

    @Test("Distinct kinds compare non-equal regardless of generation")
    func kindsAreDistinct() {
        #expect(SettingsView.WebSearchKeySaveFeedback.saved(generation: 1)
                != SettingsView.WebSearchKeySaveFeedback.cleared(generation: 1))
        #expect(SettingsView.WebSearchKeySaveFeedback.saved(generation: 1)
                != SettingsView.WebSearchKeySaveFeedback.writeFailed(generation: 1))
        #expect(SettingsView.WebSearchKeySaveFeedback.cleared(generation: 1)
                != SettingsView.WebSearchKeySaveFeedback.writeFailed(generation: 1))
    }

    @Test("Same kind + same generation compares equal (idempotent)")
    func identicalEntriesAreEqual() {
        let a = SettingsView.WebSearchKeySaveFeedback.writeFailed(generation: 7)
        let b = SettingsView.WebSearchKeySaveFeedback.writeFailed(generation: 7)
        #expect(a == b)
    }
}
