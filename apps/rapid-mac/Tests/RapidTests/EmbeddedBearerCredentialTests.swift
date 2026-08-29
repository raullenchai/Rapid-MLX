import Foundation
import Testing
@testable import Rapid

@Suite("Embedded bearer credential persistence (issue #2599)")
struct EmbeddedBearerCredentialTests {
    private let now = Date(timeIntervalSince1970: 1_760_000_000)

    private func secret(seed: UInt8) -> String {
        String(repeating: String(format: "%02x", seed), count: 32)
    }

    private func store(
        suiteName: String,
        keychain: KeychainStoring = InMemoryKeychain()
    ) -> EmbeddedBearerCredentialStore {
        let defaults = UserDefaults(suiteName: suiteName)!
        defaults.removePersistentDomain(forName: suiteName)
        return EmbeddedBearerCredentialStore(defaults: defaults, keychain: keychain)
    }

    @Test("Generated one-time material has the canonical bearer shape")
    func generatedMaterialShape() {
        let material = EmbeddedBearerMaterialResolver.resolve(
            lifetime: .perLaunch,
            store: store(suiteName: "bearer-shape"),
            now: now,
            generateSecret: { self.secret(seed: 1) }
        )

        #expect(material.secret.count == 64)
        #expect(material.secret.allSatisfy { $0.isHexDigit && $0.isASCII })
        #expect(!material.isPersisted)
        #expect(material.issue == nil)
    }

    @Test("Per-launch material rotates and never persists either layer")
    func perLaunchRotation() {
        let suiteName = "bearer-per-launch"
        let credentialStore = store(suiteName: suiteName)
        var seed: UInt8 = 1

        let first = EmbeddedBearerMaterialResolver.resolve(
            lifetime: .perLaunch,
            store: credentialStore,
            now: now,
            generateSecret: { seed += 1; return self.secret(seed: seed) }
        )
        let second = EmbeddedBearerMaterialResolver.resolve(
            lifetime: .perLaunch,
            store: credentialStore,
            now: now.addingTimeInterval(1),
            generateSecret: { seed += 1; return self.secret(seed: seed) }
        )

        #expect(first.secret != second.secret)
        #expect(!first.isPersisted)
        #expect(!second.isPersisted)
        #expect(credentialStore.loadCredential() == .missing)
    }

    @Test("Daily material reuses until expiry, then rotates in the Keychain")
    func dailyRotation() {
        let credentialStore = store(suiteName: "bearer-daily")
        let first = EmbeddedBearerMaterialResolver.resolve(
            lifetime: .daily,
            store: credentialStore,
            now: now,
            generateSecret: { self.secret(seed: 2) }
        )
        let reused = EmbeddedBearerMaterialResolver.resolve(
            lifetime: .daily,
            store: credentialStore,
            now: now.addingTimeInterval(24 * 60 * 60 - 1),
            generateSecret: { self.secret(seed: 99) }
        )
        let rotated = EmbeddedBearerMaterialResolver.resolve(
            lifetime: .daily,
            store: credentialStore,
            now: now.addingTimeInterval(24 * 60 * 60 + 1),
            generateSecret: { self.secret(seed: 3) }
        )

        #expect(first.isPersisted)
        #expect(reused.secret == first.secret)
        #expect(rotated.secret != first.secret)
        #expect(rotated.issue == nil)
    }

    @Test("Explicit material survives restarts and only the user rotates it")
    func explicitRotation() {
        let credentialStore = store(suiteName: "bearer-explicit")
        let first = EmbeddedBearerMaterialResolver.resolve(
            lifetime: .explicit,
            store: credentialStore,
            now: now,
            generateSecret: { self.secret(seed: 4) }
        )
        let reused = EmbeddedBearerMaterialResolver.resolve(
            lifetime: .explicit,
            store: credentialStore,
            now: now.addingTimeInterval(10 * 24 * 60 * 60),
            generateSecret: { self.secret(seed: 98) }
        )
        let rotated = EmbeddedBearerMaterialResolver.resolve(
            lifetime: .explicit,
            store: credentialStore,
            now: now.addingTimeInterval(10 * 24 * 60 * 60 + 1),
            generateSecret: { self.secret(seed: 5) }
        )

        #expect(reused.secret == first.secret)
        #expect(rotated.secret == first.secret)
    }

    @Test("Malformed saved credential fails safe to a one-time key")
    func malformedCredentialFailsSafe() {
        let credentialStore = store(
            suiteName: "bearer-corrupted",
            keychain: FixedResultKeychain(readResult: .found("not-a-canonical-bearer"))
        )

        let material = EmbeddedBearerMaterialResolver.resolve(
            lifetime: .daily,
            store: credentialStore,
            now: now.addingTimeInterval(1),
            generateSecret: { self.secret(seed: 6) }
        )

        #expect(!material.isPersisted)
        #expect(material.issue == .corruptedCredential)
    }

    @Test("Unavailable Keychain fails safe to a one-time key")
    func unavailableKeychainFailsSafe() {
        let material = EmbeddedBearerMaterialResolver.resolve(
            lifetime: .daily,
            store: store(
                suiteName: "bearer-unavailable",
                keychain: FixedResultKeychain(readResult: .unavailable)
            ),
            now: now,
            generateSecret: { self.secret(seed: 7) }
        )

        #expect(!material.isPersisted)
        #expect(material.issue == .unavailableKeychain)
    }

    @Test("Keychain write failure keeps the new bearer out of persistence")
    func writeFailureDegradesToOneTime() {
        let material = EmbeddedBearerMaterialResolver.resolve(
            lifetime: .explicit,
            store: store(
                suiteName: "bearer-write-failed",
                keychain: FixedResultKeychain(readResult: .missing, writeSucceeds: false)
            ),
            now: now,
            generateSecret: { self.secret(seed: 8) }
        )

        #expect(!material.isPersisted)
        #expect(material.issue == .writeFailed)
    }

    @Test("Secret storage is separate from UserDefaults policy storage")
    func storageSeparation() {
        let suiteName = "bearer-storage-separation"
        let defaults = UserDefaults(suiteName: suiteName)!
        defaults.removePersistentDomain(forName: suiteName)
        let secret = secret(seed: 9)
        let credentialStore = EmbeddedBearerCredentialStore(
            defaults: defaults,
            keychain: InMemoryKeychain()
        )
        let saved = credentialStore.save(
            EmbeddedBearerCredential(
                secret: secret,
                rotatedAt: now,
                lifetime: .explicit
            )
        )

        let serializedDefaults = defaults.dictionaryRepresentation()
            .values
            .compactMap { value in
                (value as? Data).flatMap { String(data: $0, encoding: .utf8) }
            }

        #expect(saved)
        #expect(serializedDefaults.allSatisfy { !$0.contains(secret) })
        #expect(credentialStore.loadCredential() != .missing)
    }
}

private final class FixedResultKeychain: KeychainStoring, @unchecked Sendable {
    private let readResult: KeychainReadResult
    private let writeSucceeds: Bool

    init(readResult: KeychainReadResult, writeSucceeds: Bool = true) {
        self.readResult = readResult
        self.writeSucceeds = writeSucceeds
    }

    func read(account: String) -> String? {
        guard case .found(let secret) = readResult else { return nil }
        return secret
    }

    func readWithoutUserInteraction(account: String) -> KeychainReadResult {
        readResult
    }

    @discardableResult
    func write(account: String, secret: String) -> Bool {
        writeSucceeds
    }

    @discardableResult
    func delete(account: String) -> Bool {
        true
    }
}

@MainActor
@Suite("ServerManager embedded bearer policy (issue #2599)")
struct ServerManagerEmbeddedBearerTests {
    @Test("Lifetime selection persists separately from the secret")
    func lifetimeSelectionPersists() {
        let suiteName = "server-bearer-lifetime"
        let defaults = UserDefaults(suiteName: suiteName)!
        defaults.removePersistentDomain(forName: suiteName)
        let credentialStore = RecordingCredentialStore()
        let server = ServerManager(
            testingState: .idle,
            sessionDefaults: defaults,
            bearerCredentialStore: credentialStore
        )

        server.setEmbeddedBearerLifetime(.daily)

        #expect(server.embeddedBearerLifetime == .daily)
        #expect(defaults.string(forKey: "rapid.embeddedBearer.lifetime.v1") == "daily")
    }

    @Test("Manual rotation persists only for persisted lifetimes")
    func manualRotation() {
        let dailyCredentialStore = RecordingCredentialStore()
        let dailyServer = ServerManager(
            testingState: .ready(alias: "model"),
            activeBearer: "active-secret",
            sessionDefaults: UserDefaults(suiteName: "server-bearer-daily")!,
            bearerCredentialStore: dailyCredentialStore
        )
        dailyServer.setEmbeddedBearerLifetime(.daily)
        let dailyRotated = dailyServer.rotateEmbeddedBearerNow(now: Date(timeIntervalSince1970: 10))

        #expect(dailyRotated)
        #expect(dailyServer.activeBearer == "active-secret")
        #expect(dailyServer.embeddedBearerStatus == .materialized(
            rotatedAt: Date(timeIntervalSince1970: 10),
            isPersisted: true,
            issue: nil
        ))

        let perLaunchCredentialStore = RecordingCredentialStore()
        let perLaunchServer = ServerManager(
            testingState: .ready(alias: "model"),
            activeBearer: "active-secret",
            sessionDefaults: UserDefaults(suiteName: "server-bearer-per-launch")!,
            bearerCredentialStore: perLaunchCredentialStore
        )
        let perLaunchRotated = perLaunchServer.rotateEmbeddedBearerNow(now: Date(timeIntervalSince1970: 20))

        #expect(!perLaunchRotated)
        #expect(perLaunchCredentialStore.saved.isEmpty)
    }
}

private final class RecordingCredentialStore: EmbeddedBearerCredentialStoring, @unchecked Sendable {
    private(set) var saved: [EmbeddedBearerCredential] = []
    private(set) var cleared = false

    func loadCredential() -> EmbeddedBearerStorageResult {
        .missing
    }

    func hasPersistedCredential() -> Bool {
        !saved.isEmpty
    }

    @discardableResult
    func save(_ credential: EmbeddedBearerCredential) -> Bool {
        saved.append(credential)
        return true
    }

    @discardableResult
    func clear() -> Bool {
        cleared = true
        return true
    }
}
