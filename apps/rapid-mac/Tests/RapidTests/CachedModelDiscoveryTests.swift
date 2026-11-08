import Foundation
import Testing
@testable import Rapid

@Suite(.serialized, .timeLimit(.minutes(2)))
struct CachedModelDiscoveryTests {
    private let binary = URL(fileURLWithPath: "/fake/rapid-mlx")

    /// Deliberately ignores cancellation, like a filesystem read blocked by
    /// an unanswered permission prompt. Released by the test after timeout.
    private actor Provider {
        var continuation: CheckedContinuation<Void, Never>?
        var calls = 0
        func load() async -> [ModelEntry] {
            calls += 1
            if calls == 1 {
                await withCheckedContinuation { continuation = $0 }
            }
            return [ModelEntry(alias: "recovered", hfRepo: nil, sizeOnDisk: nil, cached: true)]
        }
        func release() { continuation?.resume(); continuation = nil }
    }

    private actor ControlledProvider {
        var calls = 0
        private var pending: [Int: CheckedContinuation<[ModelEntry], Error>] = [:]
        func load() async throws -> [ModelEntry] {
            calls += 1
            let call = calls
            return try await withCheckedThrowingContinuation { pending[call] = $0 }
        }
        func succeed(_ call: Int, alias: String) {
            pending.removeValue(forKey: call)?.resume(returning: [
                ModelEntry(alias: alias, hfRepo: nil, sizeOnDisk: nil, cached: true)
            ])
        }
        func fail(_ call: Int) {
            pending.removeValue(forKey: call)?.resume(throwing: CocoaError(.fileReadNoPermission))
        }
    }

    @Test("Retry during discovery redirects an existing waiter to the new result", arguments: [false, true])
    func retryDuringInflightDiscovery(fresh: Bool) async {
        let provider = ControlledProvider()
        let cache = ModelCatalogCache(loader: { _, _ in try await provider.load() },
                                      discoveryTimeout: 5, requestedOverride: { nil })
        let waiter = Task {
            if fresh { return await cache.freshEntries(binary: binary, generation: 0) }
            return await cache.entries(binary: binary, generation: 0)
        }
        while await provider.calls < 1 { await Task.yield() }
        await cache.invalidate()
        let retry = Task { await cache.discoveryResult(binary: binary, generation: 0) }
        while await provider.calls < 2 { await Task.yield() }
        await provider.succeed(2, alias: "retry-model")
        let replacement = await retry.value
        #expect(replacement.failure == nil)
        #expect(replacement.entries.first?.alias == "retry-model")
        #expect(await waiter.value == replacement.entries)
        // The cancelled worker ignores cancellation and fails after Retry won.
        await provider.fail(1)
        let final = await cache.discoveryResult(binary: binary, generation: 0)
        #expect(final.failure == nil)
        #expect(final.entries == replacement.entries)
        #expect(await cache.discoveryFailure == nil)
        #expect(await provider.calls == 2)
    }

    @Test("A different-generation replacement cannot be overwritten by an old waiter")
    func generationReplacementRejectsOldWaiter() async {
        let provider = ControlledProvider()
        let cache = ModelCatalogCache(loader: { _, _ in try await provider.load() },
                                      discoveryTimeout: 5, requestedOverride: { nil })
        let old = Task { await cache.discoveryResult(binary: binary, generation: 0) }
        while await provider.calls < 1 { await Task.yield() }
        let replacement = Task { await cache.discoveryResult(binary: binary, generation: 1) }
        while await provider.calls < 2 { await Task.yield() }
        await provider.succeed(2, alias: "new-generation")
        let current = await replacement.value
        let discarded = await old.value
        #expect(discarded.failure != nil)
        #expect(discarded.entries.isEmpty)
        await provider.succeed(1, alias: "obsolete")
        #expect(await cache.entries(binary: binary, generation: 1) == current.entries)
        #expect(await cache.discoveryFailure == nil)
        #expect(await provider.calls == 2)
    }

    @Test("A failed read retains its failure after a successful Retry clears shared state")
    func failureStaysWithRead() async {
        let provider = ControlledProvider()
        let cache = ModelCatalogCache(loader: { _, _ in try await provider.load() },
                                      volumeResolver: { _ in "Models SSD" }, requestedOverride: { nil })
        let read = Task { await cache.discoveryResult(binary: binary, generation: 0) }
        while await provider.calls < 1 { await Task.yield() }
        await provider.fail(1)
        let failed = await read.value
        await cache.invalidate()
        let retry = Task { await cache.discoveryResult(binary: binary, generation: 0) }
        while await provider.calls < 2 { await Task.yield() }
        await provider.succeed(2, alias: "recovered")
        let recovered = await retry.value
        #expect(recovered.failure == nil)
        #expect(await cache.discoveryFailure == nil)
        #expect(failed.failure != nil)
        #expect(failed.entries.contains { $0.alias == QuickstartCoordinator.defaultChoice.alias })
        if let failure = failed.failure {
            // ContentView wraps startup maintenance and the cache in another
            // deadline. That wrapper must preserve this read's permission copy.
            let forwarded = await CachedModelDiscovery.discover(cacheURL: nil) { throw failure }
            #expect(forwarded.failure == failure)
            #expect(forwarded.failure?.volumeName == "Models SSD")
        }
    }

    @Test("Unanswered permission prompt settles, offers starter, and Retry recovers")
    func timeoutAndRetry() async {
        let provider = Provider()
        let cache = ModelCatalogCache(loader: { _, _ in await provider.load() },
                                      discoveryTimeout: 0.1, volumeResolver: { _ in "Models SSD" })
        let start = ContinuousClock.now
        let entries = await cache.entries(binary: binary, generation: 0)
        #expect(start.duration(to: .now) < .seconds(2))
        #expect(await cache.discoveryFailure == CachedModelDiscoveryFailure(volumeName: "Models SSD"))
        #expect(entries.contains { $0.alias == QuickstartCoordinator.defaultChoice.alias && !$0.cached })
        await assertStarterCanDownload(entries)
        // Presentation fallback is not authoritative absence/presence evidence
        // for sidecar startup or persisted session restoration.
        #expect(await cache.freshEntries(binary: binary, generation: 0).isEmpty)
        // Retry uses this same invalidation/read path in the production UI.
        await cache.invalidate()
        let recovered = await cache.entries(binary: binary, generation: 0)
        #expect(recovered.first?.alias == "recovered")
        #expect(await cache.discoveryFailure == nil)
        // A late success must not overwrite the retry's snapshot or error.
        await provider.release()
        #expect(await cache.entries(binary: binary, generation: 0) == recovered)
    }

    @MainActor
    private func assertStarterCanDownload(_ entries: [ModelEntry]) {
        let starter = QuickstartCoordinator.defaultChoice.alias
        let hardware = MacHardware(brandString: "Test Mac", family: .m3, tier: .base,
                                   physicalRAMBytes: 8 * 1_073_741_824, memoryBandwidthGBs: 100)
        let coordinator = QuickstartCoordinator()
        coordinator.settleDefaultChoice(hardware: hardware, catalog: entries)
        #expect(coordinator.selection.alias == starter)
        let fallback = ModelPickerBar.quickstartEligibleDefault(
            catalog: entries, eligible: true, targetAlias: coordinator.selection.alias
        )
        #expect(ModelPickerBar.normalizedChatSelection(
            currentAlias: "", catalog: entries, knownNonChatAliases: [], fallbackAlias: fallback
        ) == starter)
        let state = QuickstartView.catalogState(catalog: entries, loaded: true)
        let primary = OnboardingModelSelection.primary(
            selection: starter,
            visibleRows: [.init(alias: starter, isCached: false, isAvailable: true)],
            catalogState: state, context: .shortlist
        )
        #expect(primary.isEnabled)
        #expect(primary.action == .downloadAndStart)
    }

    @Test("Don't Allow settles immediately for EPERM, EACCES and Cocoa permission errors")
    func deniedPermission() async {
        let errors: [NSError] = [
            NSError(domain: NSPOSIXErrorDomain, code: Int(EPERM)),
            NSError(domain: NSPOSIXErrorDomain, code: Int(EACCES)),
            CocoaError(.fileReadNoPermission) as NSError,
        ]
        for error in errors {
            let cache = ModelCatalogCache(loader: { _, _ in throw error },
                                          volumeResolver: { _ in "External Models" })
            let start = ContinuousClock.now
            let entries = await cache.entries(binary: binary, generation: 0)
            #expect(start.duration(to: .now) < .seconds(2))
            #expect(await cache.discoveryFailure?.message == "Rapid needs permission to read models on External Models")
            await assertStarterCanDownload(entries)
        }
    }

    private func savedOverride() -> URL {
        let suite = "rapid.tests.discovery.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        ModelsFolderPreference.setStoredPath("/Volumes/Models SSD/custom-cache", defaults: defaults)
        return CachedModelDiscovery.requestedOverrideURL(defaults: defaults)!
    }

    @Test("Saved override validation itself is bounded while consent is unanswered")
    func overrideValidationTimeout() async {
        let override = savedOverride()
        let gate = DispatchSemaphore(value: 0)
        defer { gate.signal() }
        let cache = ModelCatalogCache(
            loader: { _, validated in
                #expect(validated == override)
                return [ModelEntry(alias: "late", hfRepo: nil, sizeOnDisk: nil, cached: true)]
            },
            discoveryTimeout: 0.1,
            requestedOverride: { override },
            directoryProbe: { _ in
                gate.wait() // A synchronous stat waiting for macOS consent.
                return true
            }
        )
        let start = ContinuousClock.now
        let entries = await cache.entries(binary: binary, generation: 0)
        #expect(start.duration(to: .now) < .seconds(2))
        #expect(await cache.discoveryFailure?.volumeName == "Models SSD")
        #expect(await cache.freshEntries(binary: binary, generation: 0).isEmpty)
        await assertStarterCanDownload(entries)
    }

    @Test("Denied saved override never loads the default cache or resolves persisted startup")
    func deniedSavedOverride() async {
        let override = savedOverride()
        let errors: [NSError] = [
            NSError(domain: NSPOSIXErrorDomain, code: Int(EPERM)),
            NSError(domain: NSPOSIXErrorDomain, code: Int(EACCES)),
            CocoaError(.fileReadNoPermission) as NSError,
        ]
        for error in errors {
            let cache = ModelCatalogCache(
                loader: { _, _ in
                    Issue.record("Denied override must stop discovery before any catalog loader runs")
                    return []
                },
                requestedOverride: { override },
                directoryProbe: { url in
                    #expect(url == override)
                    throw error
                }
            )
            let start = ContinuousClock.now
            let entries = await cache.entries(binary: binary, generation: 0)
            #expect(start.duration(to: .now) < .seconds(2))
            #expect(await cache.discoveryFailure?.volumeName == "Models SSD")
            await assertStarterCanDownload(entries)
            let authoritative = await cache.freshEntries(binary: binary, generation: 0)
            #expect(authoritative.isEmpty)
            let plan = SessionModelRestore.launchPlan(
                legacyLastAlias: QuickstartCoordinator.defaultChoice.alias,
                dictationAlias: nil, speechAlias: nil, catalog: authoritative,
                autoStartEnabled: true
            )
            #expect(!plan.chatAliasResolved)
            #expect(!plan.shouldAutoStart)
        }
    }

    @Test("Settings catalog validation preserves denial and Retry reads the saved override")
    func settingsOverrideValidationAndRetry() async {
        let override = savedOverride()
        // Settings uses this shared discovery entry point for its atomic and
        // legacy loaders; neither may run after override validation fails.
        let denied = await CachedModelDiscovery.discoverCatalog(
            requestedOverride: override,
            directoryProbe: { url in
                #expect(url == override)
                throw NSError(domain: NSPOSIXErrorDomain, code: Int(EACCES))
            }
        ) { _ in
            Issue.record("Settings must not scan the default cache after denial")
            return []
        }
        #expect(denied.failure?.volumeName == "Models SSD")
        await assertStarterCanDownload(denied.entries)
        let recovered = await CachedModelDiscovery.discoverCatalog(
            requestedOverride: override, directoryProbe: { _ in true }
        ) { validated in
            #expect(validated == override)
            return [ModelEntry(alias: "recovered", hfRepo: nil, sizeOnDisk: nil, cached: true)]
        }
        #expect(recovered.failure == nil)
        #expect(recovered.entries.first?.cached == true)
    }

    @Test("Missing and non-directory saved overrides still load the default cache")
    func missingSavedOverride() async {
        let override = savedOverride()
        let errors: [NSError?] = [
            nil, // An existing regular file, not a directory.
            NSError(domain: NSPOSIXErrorDomain, code: Int(ENOENT)),
            NSError(domain: NSPOSIXErrorDomain, code: Int(ENOTDIR)),
            CocoaError(.fileNoSuchFile) as NSError,
            CocoaError(.fileReadNoSuchFile) as NSError,
        ]
        for error in errors {
            let cache = ModelCatalogCache(
                loader: { _, validated in
                    #expect(validated == nil)
                    return [ModelEntry(alias: "default-cache", hfRepo: nil, sizeOnDisk: nil, cached: true)]
                },
                requestedOverride: { override },
                directoryProbe: { _ in
                    if let error { throw error }
                    return false
                }
            )
            let entries = await cache.freshEntries(binary: binary, generation: 0)
            #expect(entries.first?.alias == "default-cache")
            #expect(await cache.discoveryFailure == nil)
        }
    }

    @Test("Boot-volume failure and unrelated external-volume errors use generic copy")
    func genericFailure() async {
        for permission in [true, false] {
            let result = await CachedModelDiscovery.discover(
                cacheURL: URL(fileURLWithPath: "/fake/hub"),
                volumeResolver: { _ in permission ? nil : "Models SSD" }
            ) { throw permission ? CocoaError(.fileReadNoPermission) : CocoaError(.fileReadUnknown) }
            #expect(result.failure?.message == "Couldn't read cached models")
        }
    }

    @Test("Both HF-home and hub symlinks identify the volume without enumerating it")
    func symlinkResolution() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let home = root.appendingPathComponent("huggingface")
        let hub = root.appendingPathComponent("hub")
        // Dangling targets are intentional: this test never accesses a volume.
        try FileManager.default.createSymbolicLink(atPath: home.path, withDestinationPath: "/Volumes/Models SSD/hf")
        try FileManager.default.createSymbolicLink(atPath: hub.path, withDestinationPath: "/Volumes/Models SSD/hf/hub")
        #expect(CachedModelDiscovery.externalVolumeName(home.appendingPathComponent("hub")) == "Models SSD")
        #expect(CachedModelDiscovery.externalVolumeName(hub) == "Models SSD")
        #expect(CachedModelDiscovery.externalVolumeName(root) == nil)
    }

    @Test("Volume discovery follows relative link chains and bounds symlink loops")
    func relativeSymlinkResolution() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let link = root.appendingPathComponent("link")
        let target = root.appendingPathComponent("target")
        try FileManager.default.createSymbolicLink(atPath: link.path, withDestinationPath: "./target")
        try FileManager.default.createSymbolicLink(atPath: target.path, withDestinationPath: "/Volumes/Models SSD/hub")
        #expect(CachedModelDiscovery.externalVolumeName(link) == "Models SSD")
        let loop = root.appendingPathComponent("loop")
        try FileManager.default.createSymbolicLink(atPath: loop.path, withDestinationPath: "loop")
        #expect(CachedModelDiscovery.externalVolumeName(loop) == nil)
    }

    @Test("Real directory probe follows symlinks and keeps missing-folder fallback")
    func realOverrideValidation() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let link = root.appendingPathComponent("link")
        try FileManager.default.createSymbolicLink(atPath: link.path, withDestinationPath: root.path)
        #expect(try ModelsFolderPreference.discoveryOverrideURL(link) == link)
        let file = root.appendingPathComponent("file")
        try Data().write(to: file)
        #expect(try ModelsFolderPreference.discoveryOverrideURL(file) == nil)
        #expect(try ModelsFolderPreference.discoveryOverrideURL(file.appendingPathComponent("child")) == nil)
        #expect(try ModelsFolderPreference.discoveryOverrideURL(root.appendingPathComponent("missing")) == nil)
        #expect(try ModelsFolderPreference.discoveryOverrideURL(nil) == nil)
    }

    @Test("Golden-flow legacy catalog and ls fallback are successful discovery")
    func goldenFlowLegacyCatalog() async throws {
        let package = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let fake = package.appendingPathComponent("scripts/fake-rapid-mlx.sh")
        let cache = ModelCatalogCache(requestedOverride: { nil })
        let chat = await cache.entries(binary: fake, generation: 0)
        #expect(await cache.discoveryFailure == nil)
        #expect(chat.contains { $0.alias == "fake-alias" && $0.cached })
        let result = await CachedModelDiscovery.discoverCatalog(requestedOverride: nil) { override in
            if let atomic = await ModelCatalog.productEntries(
                binary: fake, hubCacheOverride: override, openedPaths: []
            ) { return atomic }
            return await ModelCatalog.load(binary: fake, hubCacheOverride: override, openedPaths: [])
        }
        #expect(result.failure == nil)
        #expect(result.entries.contains { $0.alias == "fake-alias" && $0.cached })
    }

    @Test("Successful empty legacy cache is not a discovery failure")
    func emptyLegacyCache() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let script = root.appendingPathComponent("rapid-mlx")
        // Older sidecars can reject JSON but successfully report an empty ls.
        try Data("""
        #!/bin/sh
        case "$*" in
          "models --cached --json") echo 'unknown option --json' >&2; exit 2 ;;
          "ls") echo 'No models cached yet.'; echo 'warning: legacy sidecar' >&2 ;;
          *) echo '{"text": []}' ;;
        esac

        """.utf8).write(to: script)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
        let cache = ModelCatalogCache(requestedOverride: { nil })
        #expect(await cache.entries(binary: script, generation: 0).isEmpty)
        #expect(await cache.discoveryFailure == nil)
    }

    @Test("CLI stderr permission failure reaches discovery state")
    func cliPermissionFailure() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let script = root.appendingPathComponent("rapid-mlx")
        try Data("#!/bin/sh\necho 'PermissionError: [Errno 1] Operation not permitted' >&2\nexit 1\n".utf8).write(to: script)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
        let cache = ModelCatalogCache(volumeResolver: { _ in "Models SSD" })
        let entries = await cache.entries(binary: script, generation: 0)
        #expect(await cache.discoveryFailure?.volumeName == "Models SSD")
        await assertStarterCanDownload(entries)
    }
}
