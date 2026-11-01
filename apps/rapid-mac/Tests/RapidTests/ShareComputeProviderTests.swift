import Foundation
import Testing
@testable import Rapid

final class ShareComputeTestKeychain: KeychainStoring, @unchecked Sendable {
    var values: [String: String] = [:]
    var unavailable = false
    var failWrite = false
    var failDelete = false
    var tombstoneOnFailedDelete = false
    var writes = 0

    func read(account: String) -> String? { values[account] }
    func readWithoutUserInteraction(account: String) -> KeychainReadResult {
        if unavailable { return .unavailable }
        if let value = values[account] { return .found(value) }
        return .missing
    }
    func write(account: String, secret: String) -> Bool {
        if failWrite { return false }
        writes += 1
        values[account] = secret
        return true
    }
    func delete(account: String) -> Bool {
        if failDelete {
            if tombstoneOnFailedDelete { values[account] = "" }
            return false
        }
        values.removeValue(forKey: account)
        return true
    }
}

@Suite("Share Compute · saved provider and offline lifecycle", .serialized)
@MainActor
struct ShareComputeProviderTests {
    static let firstKey = "qsppk-first-fixture-12345678"
    static let secondKey = "qsppk-second-fixture-87654321"

    @Test("Sharing explains why local chat is paused without offering an unusable Start")
    func sharingReadiness() {
        let readiness = ModelReadiness.sharingCompute
        #expect(!readiness.sendAllowed)
        #expect(!readiness.isFailure)
        #expect(!readiness.isWorking)
        #expect(readiness.action == nil)
        #expect(readiness.alias == nil)
        #expect(readiness.statusRole == .idle)
        #expect(readiness.detail?.contains("Stop sharing") == true)
        #expect(readiness.composerPlaceholder.contains("sharing"))
        #expect(readiness.accessibilityLabel.contains(readiness.headline))
    }

    @Test("Provider key validation rejects other keys and malformed pastes", arguments: [
        "", "sk-consumer-123456789", "qsprk-read-only-123456789", "qsppk-short",
        "qsppk-key with spaces", "qsppk-key\nwith-newline", "qsppk-key\u{0}with-control",
        "qsppk-" + String(repeating: "a", count: 4_090),
    ])
    func invalidKeys(raw: String) {
        #expect(ShareComputeProviderKey.validate(raw) == nil)
    }

    @Test("Saved account survives reopen without leaking into preferences")
    func persistenceAndAccountReplacement() throws {
        let suite = "ShareComputeProvider.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let keychain = ShareComputeTestKeychain()
        let store = ShareComputeProviderKeyStore(keychain: keychain)
        let key = try #require(ShareComputeProviderKey.validate(Self.firstKey))
        #expect(store.save(key))
        #expect(key.redactedLabel == "qsppk-…5678")
        let reopened = ShareComputeProviderKeyStore(keychain: keychain)
        #expect(reopened.load() == .found(key))
        #expect(!defaults.dictionaryRepresentation().values.contains {
            String(describing: $0).contains(Self.firstKey)
        })
        #expect(reopened.save(key))
        #expect(reopened.save(try #require(ShareComputeProviderKey.validate(Self.secondKey))))
        #expect(store.load() != .found(key))
    }

    @Test("Failed key-store writes and deletes preserve the saved account")
    func failedStorageDoesNotSwitchAccount() throws {
        let suite = "ShareComputeProviderFailure.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let keychain = ShareComputeTestKeychain()
        let store = ShareComputeProviderKeyStore(keychain: keychain)
        let original = try #require(ShareComputeProviderKey.validate(Self.firstKey))
        #expect(store.save(original))
        keychain.failWrite = true
        #expect(!store.save(try #require(ShareComputeProviderKey.validate(Self.secondKey))))
        #expect(store.load() == .found(original))
        keychain.failDelete = true
        #expect(!store.remove())
        #expect(store.load() == .found(original))
        keychain.failDelete = false
        #expect(store.remove())
        #expect(store.load() == .missing)
    }

    @Test("Failed manager key changes retain the account but require registration again", arguments: [false, true])
    func failedManagerStorageInvalidatesProof(removing: Bool) throws {
        let home = FileManager.default.temporaryDirectory.appendingPathComponent("share-key-write-failure-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: home) }
        let model = ShareComputeModel.desktopSharing[0]
        let marker = ShareComputeManager.registrationURL(catalogID: model.catalogID, home: home)
        let cache = ShareComputeManager.credentialCacheURL(catalogID: model.catalogID, home: home)
        try FileManager.default.createDirectory(at: marker.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: [
            "schema_version": 1, "model": model.catalogID, "alias": model.alias, "worker": "test-mac"
        ]).write(to: marker)
        let cachedNode = Data("{\"node\":\"existing-fixture\"}".utf8)
        try cachedNode.write(to: cache)
        let keychain = ShareComputeTestKeychain()
        keychain.values[ShareComputeProviderKeyStore.account] = Self.firstKey
        keychain.failWrite = !removing
        keychain.failDelete = removing
        let store = ShareComputeProviderKeyStore(keychain: keychain)
        let server = ServerManager(testingState: .ready(alias: "personal-model"))
        let manager = ShareComputeManager(server: server, providerStore: store, home: home)
        #expect(!manager.needsRegistration(for: model, worker: "test-mac"))

        let revision = manager.providerKeyRevision
        let succeeded = removing ? manager.removeProviderKey() : manager.saveProviderKey(Self.secondKey)

        #expect(!succeeded)
        #expect(manager.providerKeyRevision == revision + 1)
        #expect(store.load() == .found(try #require(ShareComputeProviderKey.validate(Self.firstKey))))
        // Invalidate before secret mutation: even a failed switch must never
        // reuse a possibly stale payout binding. The private node cache survives.
        #expect(!FileManager.default.fileExists(atPath: marker.path))
        #expect(manager.needsRegistration(for: model, worker: "test-mac"))
        #expect(try Data(contentsOf: cache) == cachedNode)
        #expect(server.servingAlias == "personal-model")
        #expect(manager.canEditProviderKey)
        #expect(manager.providerKeyMessage != nil)
    }

    @Test("Failed removal refreshes a tombstoned provider key")
    func failedRemovalWithTombstone() {
        let home = FileManager.default.temporaryDirectory.appendingPathComponent("share-key-tombstone-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: home) }
        let keychain = ShareComputeTestKeychain()
        let manager = ShareComputeManager(
            server: ServerManager(testingState: .idle),
            providerStore: ShareComputeProviderKeyStore(keychain: keychain), home: home
        )
        #expect(manager.saveProviderKey(Self.firstKey))
        let ledger = ShareComputeLedgerCoordinator(keychain: keychain, fixture: nil)
        let revision = manager.providerKeyRevision
        keychain.failDelete = true
        keychain.tombstoneOnFailedDelete = true
        #expect(!manager.removeProviderKey())
        #expect(manager.providerKeyRevision == revision + 1)
        #expect(!manager.providerKeyIsStored)
        #expect(manager.savedProviderKeyLabel == nil)
        #expect(manager.providerKeyMessage == "This Mac couldn't remove the provider key. Unlock your Keychain and try again.")
        // The view forwards this revision change to the ledger.
        ledger.providerKeyChanged()
        #expect(ledger.state == .noKey)
        #expect(ledger.savedKeyLabel == nil)
    }

    @Test("Locked, corrupt and read-only credentials remain distinct")
    func credentialBoundaries() {
        let suite = "ShareComputeProviderBoundary.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let keychain = ShareComputeTestKeychain()
        let store = ShareComputeProviderKeyStore(keychain: keychain)
        keychain.unavailable = true
        #expect(store.load() == .unavailable)
        keychain.unavailable = false
        keychain.values[ShareComputeProviderKeyStore.account] = "corrupted"
        #expect(store.load() == .corrupted)
        keychain.values.removeAll()
        // A value under a totally unrelated account (the retired legacy
        // read-key account) must never be picked up by the provider store,
        // and removing the provider key must never touch it either.
        keychain.values[ShareComputeLegacyReadKeyCleanup.account] = "qsprk-read-fixture-123456789"
        #expect(store.load() == .missing)
        #expect(store.remove())
        #expect(keychain.values[ShareComputeLegacyReadKeyCleanup.account] != nil)
    }

    @Test("Unavailable credentials fail before unloading a local model; corrupt keys can be removed")
    func unavailableKeyDoesNotUnload() async throws {
        let home = FileManager.default.temporaryDirectory.appendingPathComponent("share-key-failure-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: home) }
        let keychain = ShareComputeTestKeychain()
        keychain.unavailable = true
        let server = ServerManager(testingState: .ready(alias: "personal-model"), binaryPath: URL(fileURLWithPath: "/usr/bin/false"))
        let manager = ShareComputeManager(server: server, providerStore: ShareComputeProviderKeyStore(keychain: keychain), home: home)
        #expect(manager.providerKeyIsStored)
        #expect(manager.savedProviderKeyLabel == nil)
        await manager.join(model: ShareComputeModel.desktopSharing[0], worker: "test-mac")
        #expect(server.servingAlias == "personal-model")
        if case .failed(let message) = manager.state { #expect(message.contains("Keychain")) }
        else { Issue.record("A locked provider key must be reported before preparation") }
        keychain.unavailable = false
        keychain.values[ShareComputeProviderKeyStore.account] = "corrupted"
        manager.refreshProviderKey()
        #expect(manager.providerKeyIsStored)
        #expect(manager.savedProviderKeyLabel == nil)
        #expect(manager.removeProviderKey())
        #expect(!manager.providerKeyIsStored)
        #expect(server.servingAlias == "personal-model")
    }

    @Test("providerKeyRevision bumps on a same-redacted-label replacement, not on a mere refresh")
    func providerKeyRevisionBumpsOnReplacement() throws {
        let keychain = ShareComputeTestKeychain()
        let server = ServerManager(testingState: .idle)
        let manager = ShareComputeManager(server: server, providerStore: ShareComputeProviderKeyStore(keychain: keychain))
        #expect(manager.providerKeyRevision == 0)

        // Two distinct keys that redact to the SAME label (same last four
        // characters, different secret) — the case the label alone cannot
        // detect.
        let keyA = "qsppk-first-fixture-abcd1234"
        let keyB = "qsppk-second-fixture-wxyz1234"
        let labelA = try #require(ShareComputeProviderKey.validate(keyA)).redactedLabel
        let labelB = try #require(ShareComputeProviderKey.validate(keyB)).redactedLabel
        #expect(labelA == labelB)

        #expect(manager.saveProviderKey(keyA))
        #expect(manager.providerKeyRevision == 1)
        #expect(manager.savedProviderKeyLabel == "qsppk-…1234")

        #expect(manager.saveProviderKey(keyB))
        #expect(manager.providerKeyRevision == 2)
        #expect(manager.savedProviderKeyLabel == "qsppk-…1234")

        // Saving the identical key again is a no-op, not a new write.
        #expect(manager.saveProviderKey(keyB))
        #expect(manager.providerKeyRevision == 2)

        // A mere refresh (no write) must not bump it either.
        manager.refreshProviderKey()
        #expect(manager.providerKeyRevision == 2)

        #expect(manager.removeProviderKey())
        #expect(manager.providerKeyRevision == 3)
    }

    @Test("This release admits only the qualified Qwen27 sharing model")
    func onlyQualifiedModelCanJoin() async {
        let server = ServerManager(testingState: .ready(alias: "personal-model"), binaryPath: URL(fileURLWithPath: "/usr/bin/false"))
        let keychain = ShareComputeTestKeychain()
        let manager = ShareComputeManager(server: server, providerStore: ShareComputeProviderKeyStore(keychain: keychain))
        #expect(ShareComputeModel.desktopSharing.map(\.catalogID) == ["qwen3.8-27b"])
        await manager.join(model: ShareComputeModel.supported[1], worker: "test-mac", providerKey: Self.firstKey)
        #expect(keychain.values.isEmpty)
        #expect(server.servingAlias == "personal-model")
        #expect(manager.canEditProviderKey)
    }

    @Test("Changing a provider key never recursively deletes an unexpected registration directory")
    func unexpectedRegistrationIsPreserved() throws {
        let home = FileManager.default.temporaryDirectory.appendingPathComponent("share-proof-guard-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: home) }
        let marker = ShareComputeManager.registrationURL(catalogID: "qwen3.8-27b", home: home)
        try FileManager.default.createDirectory(at: marker, withIntermediateDirectories: true)
        let sentinel = marker.appendingPathComponent("keep-me")
        try Data("keep".utf8).write(to: sentinel)
        let keychain = ShareComputeTestKeychain()
        keychain.values[ShareComputeProviderKeyStore.account] = Self.firstKey
        let manager = ShareComputeManager(
            server: ServerManager(testingState: .idle),
            providerStore: ShareComputeProviderKeyStore(keychain: keychain), home: home
        )
        #expect(!manager.saveProviderKey(Self.secondKey))
        #expect(keychain.values[ShareComputeProviderKeyStore.account] == Self.firstKey)
        #expect(try String(contentsOf: sentinel, encoding: .utf8) == "keep")
    }

    @Test("Inaccessible registration metadata cannot switch or remove the saved account", .enabled(if: getuid() != 0))
    func registrationPermissionsFailClosed() throws {
        let home = FileManager.default.temporaryDirectory.appendingPathComponent("share-proof-permissions-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: home) }
        let model = ShareComputeModel.desktopSharing[0]
        let marker = ShareComputeManager.registrationURL(catalogID: model.catalogID, home: home)
        let directory = marker.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let proof = try JSONSerialization.data(withJSONObject: [
            "schema_version": 1, "model": model.catalogID, "alias": model.alias, "worker": "test-mac"
        ])
        try proof.write(to: marker)
        try Data("{}".utf8).write(to: ShareComputeManager.credentialCacheURL(catalogID: model.catalogID, home: home))
        let keychain = ShareComputeTestKeychain()
        keychain.values[ShareComputeProviderKeyStore.account] = Self.firstKey
        let manager = ShareComputeManager(
            server: ServerManager(testingState: .idle),
            providerStore: ShareComputeProviderKeyStore(keychain: keychain), home: home
        )
        #expect(manager.hasRegistration(for: model, worker: "test-mac"))
        try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: directory.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path) }
        let saved = manager.saveProviderKey(Self.secondKey)
        let removed = manager.removeProviderKey()
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
        #expect(!saved)
        #expect(!removed)
        #expect(keychain.values[ShareComputeProviderKeyStore.account] == Self.firstKey)
        #expect(try Data(contentsOf: marker) == proof)
        #expect(manager.hasRegistration(for: model, worker: "test-mac"))
    }

    @Test("A dangling registration symlink is refused rather than treated as absent")
    func danglingRegistrationIsPreserved() throws {
        let home = FileManager.default.temporaryDirectory.appendingPathComponent("share-proof-symlink-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: home) }
        let marker = ShareComputeManager.registrationURL(catalogID: "qwen3.8-27b", home: home)
        try FileManager.default.createDirectory(at: marker.deletingLastPathComponent(), withIntermediateDirectories: true)
        let target = home.appendingPathComponent("missing-proof")
        try FileManager.default.createSymbolicLink(at: marker, withDestinationURL: target)
        let keychain = ShareComputeTestKeychain()
        keychain.values[ShareComputeProviderKeyStore.account] = Self.firstKey
        let manager = ShareComputeManager(
            server: ServerManager(testingState: .idle),
            providerStore: ShareComputeProviderKeyStore(keychain: keychain), home: home
        )
        #expect(!manager.saveProviderKey(Self.secondKey))
        #expect(keychain.values[ShareComputeProviderKeyStore.account] == Self.firstKey)
        #expect(try FileManager.default.destinationOfSymbolicLink(atPath: marker.path) == target.path)
    }

    @Test("Real child stop/rejoin reuses node, excludes local inference and leaves models unloaded")
    func offlineAndOnline() async throws {
        let home = FileManager.default.temporaryDirectory.appendingPathComponent("share-on-off-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: home, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: home) }
        let suite = "ShareComputeLifecycle.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let keychain = ShareComputeTestKeychain()
        let store = ShareComputeProviderKeyStore(keychain: keychain)
        let executable = home.appendingPathComponent("fake-provider.py")
        let script = #"""
        #!/usr/bin/python3
        import json, os, signal, sys, time
        from pathlib import Path
        args = sys.argv[1:]
        def arg(name): return args[args.index(name) + 1]
        root = Path(os.environ['HOME'])
        state = root / '.rapid-mlx/quicksilver'
        state.mkdir(parents=True, exist_ok=True)
        key = sys.stdin.readline().strip() if '--provider-key-stdin' in args else ''
        trace = root / 'calls.json'
        calls = json.loads(trace.read_text()) if trace.exists() else []
        calls.append({'args': args, 'had_key': key.startswith('qsppk-'), 'env_has_key': any('qsppk-' in v for v in os.environ.values())})
        trace.write_text(json.dumps(calls))
        model = arg('--quicksilver-model')
        worker = arg('--worker')
        (state / (model + '.json')).write_text('{}')
        (state / (model + '.registration.json')).write_text(json.dumps({'schema_version': 1, 'model': model, 'alias': args[1], 'worker': worker}))
        snapshot = {'schema_version': 1, 'session': arg('--desktop-session'), 'phase': 'online', 'catalog_id': model, 'alias': args[1], 'worker': worker, 'node_id': 'test-node', 'updated_at': time.time()}
        (state / ('desktop-status-' + arg('--desktop-session') + '.json')).write_text(json.dumps(snapshot))
        signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))
        while True: time.sleep(0.01)
        """#
        try Data(script.utf8).write(to: executable)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: executable.path)
        let server = ServerManager(testingState: .idle, binaryPath: executable)
        var manager = ShareComputeManager(server: server, providerStore: store, home: home)
        defer { manager.finishShutdown() }
        let model = ShareComputeModel.desktopSharing[0]
        #expect(manager.saveProviderKey(Self.firstKey))
        await manager.join(model: model, worker: "test-mac")
        try await wait { manager.state == .online }
        #expect(!manager.canEditProviderKey)
        #expect(!manager.saveProviderKey(Self.secondKey))
        #expect(!manager.removeProviderKey())
        await server.start(alias: "personal-model")
        #expect(server.servingAlias == nil)
        manager.leave()
        try await wait { manager.state == .idle && manager.canEditProviderKey }
        #expect(server.servingAlias == nil)
        #expect(manager.lastRestoreOutcome == .unloaded)
        // Reopening the GUI shares the same saved account, without another paste.
        manager = ShareComputeManager(server: server, providerStore: store, home: home)
        #expect(!manager.needsRegistration(for: model, worker: "test-mac"))
        #expect(!manager.requiresProviderKey(for: model, worker: "test-mac"))
        await manager.join(model: model, worker: "test-mac")
        try await wait { manager.state == .online }
        manager.leave()
        try await wait { manager.state == .idle && manager.canEditProviderKey }
        #expect(manager.saveProviderKey(Self.secondKey))
        await manager.join(model: model, worker: "test-mac")
        try await wait { manager.state == .online }
        manager.leave()
        try await wait { manager.state == .idle && manager.canEditProviderKey }
        // Saving the same account again must not invalidate a working node.
        #expect(manager.saveProviderKey("  " + Self.secondKey + "\n"))
        #expect(!manager.needsRegistration(for: model, worker: "test-mac"))
        await manager.join(model: model, worker: "test-mac")
        try await wait { manager.state == .online }
        manager.leave()
        try await wait { manager.state == .idle && manager.canEditProviderKey }
        // Valid node credentials work without reading a temporarily locked account key.
        keychain.unavailable = true
        manager = ShareComputeManager(server: server, providerStore: store, home: home)
        #expect(!manager.requiresProviderKey(for: model, worker: "test-mac"))
        await manager.join(model: model, worker: "test-mac")
        try await wait { manager.state == .online }
        manager.leave()
        try await wait { manager.state == .idle && manager.canEditProviderKey }
        keychain.unavailable = false
        let calls = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: home.appendingPathComponent("calls.json"))) as? [[String: Any]])
        #expect(calls.count == 5)
        #expect(calls.map { $0["had_key"] as? Bool } == [true, false, true, false, false])
        for (index, call) in calls.enumerated() {
            let args = try #require(call["args"] as? [String])
            #expect(args.contains("--reregister") == ([0, 2].contains(index)))
            #expect(!args.joined().contains(Self.firstKey))
            #expect(!args.joined().contains(Self.secondKey))
            #expect(call["env_has_key"] as? Bool == false)
        }
        #expect(manager.removeProviderKey())
        #expect(manager.requiresProviderKey(for: model, worker: "test-mac"))
        #expect(ShareComputeRestoreStatus.unloaded.title == "Unloaded")
    }

    private func wait(_ predicate: () -> Bool) async throws {
        let deadline = ContinuousClock.now + .seconds(10)
        while !predicate(), ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(predicate())
        if !predicate() { throw CocoaError(.executableRuntimeMismatch) }
    }
}
