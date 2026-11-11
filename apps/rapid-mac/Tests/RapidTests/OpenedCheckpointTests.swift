import Foundation
import SwiftUI
import Testing
@testable import Rapid

@Suite("Existing checkpoint references", .serialized, TestTimeouts.hangProne) @MainActor
struct OpenedCheckpointTests {
    func check(_ path: String, known: String? = nil, ready: Bool = true, reason: String? = nil) throws -> LocalCheckpointCheck {
        var value: [String: Any] = [
            "requested_path": OpenedCheckpoints.canonicalPath(path)!, "path": OpenedCheckpoints.canonicalPath(path)!,
            "launch_path": OpenedCheckpoints.canonicalPath(path)!, "ready": ready,
            "size_bytes": 42, "fit": "yes", "mlx_native": true, "modified_epoch": 1,
        ]
        if let known { value["known_model"] = known }
        if let reason { value["reason"] = reason }
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(LocalCheckpointCheck.self, from: JSONSerialization.data(withJSONObject: value))
    }

    func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("opened-checkpoint-\(UUID())")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    func defaults() -> (UserDefaults, String) {
        let name = "opened-checkpoint-\(UUID())"
        return (UserDefaults(suiteName: name)!, name)
    }

    private func viewport<Content: View>(_ content: Content, size: CGSize) -> GoldenStage {
        GoldenStage(ScrollView { content }, size: size)
    }

    func inspector(_ root: URL) async throws -> URL {
        let desktop = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        try Data(desktop.appendingPathComponent("scripts/fake-rapid-mlx.sh").path.utf8)
            .write(to: root.appendingPathComponent("fallback-path.txt"))
        let script = root.appendingPathComponent("inspector.py")
        try Data(#"""
        #!/usr/bin/env python3
        import json, os, pathlib, sys, time
        if '--fixture-ping' in sys.argv: sys.exit(0)
        root=pathlib.Path(__file__).parent
        if len(sys.argv)>2 and sys.argv[1]=='serve' and not sys.argv[2].startswith('-'):
            (root/'served-model.json').write_text(json.dumps(sys.argv[2]))
        if len(sys.argv)>1 and sys.argv[1]=='pull': (root/'pull-called').write_text('called')
        custom=root/'expected-cache.txt'
        if custom.exists() and '--cached' in sys.argv and '--json' in sys.argv:
            print(json.dumps(dict(cached=[dict(alias='qwen3.5-4b-4bit',repo='mlx-community/Qwen3.5-4B-MLX-4bit',state='ok',external=False,size_bytes=42)])))
            sys.exit(0)
        if '--inspect-local' not in sys.argv:
            fallback=pathlib.Path(__file__).with_name('fallback-path.txt').read_text()
            os.execv(fallback,[fallback,*sys.argv[1:]])
        pathlib.Path(__file__).with_name('inspect-called').write_text('called')
        (root/'inspect-cache.txt').write_text(os.environ.get('HF_HUB_CACHE',''))
        mode=pathlib.Path(__file__).with_name('mode.txt')
        mode=mode.read_text() if mode.exists() else ''
        if mode=='unsupported': sys.exit(2)
        if mode=='malformed': print('{}');sys.exit(0)
        if mode=='held-reject':
            (root/'held-started').write_text('started')
            deadline=time.monotonic()+8
            while not (root/'release-inspection').exists() and time.monotonic()<deadline: time.sleep(0.01)
            mode='reject-all'
        if mode=='slow':
            pathlib.Path(__file__).with_name('started').write_text('started')
            time.sleep(0.5)
        paths=[sys.argv[i+1] for i,a in enumerate(sys.argv) if a=='--inspect-local']
        rows=[]
        for p in paths:
            canonical=os.path.realpath(p)
            exists=pathlib.Path(p).is_dir()
            ready=exists and pathlib.Path(p).name!='reject' and mode!='reject-all'
            rows.append(dict(requested_path=p,path=canonical,launch_path=canonical,
                ready=ready,reason=None if ready else ('incomplete_checkpoint' if exists else 'missing_directory'),
                size_bytes=42,fit='yes',mlx_native=True,modified_epoch=1))
            if custom.exists() and os.environ.get('HF_HUB_CACHE')==custom.read_text(): rows[-1]['known_model']='qwen3.5-4b-4bit'
        if mode=='retarget': rows[0]['path']='/different';rows[0]['launch_path']='/different'
        print(json.dumps(dict(schema=1,local_checkpoints=rows)))
        """#.utf8).write(to: script)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: script.path)
        // Fail at the fixture boundary if its interpreter or executable is
        // unavailable, instead of hiding that launch error in catalog probes.
        let ping = try await TestSubprocess.run(executableURL: script,
            arguments: ["--fixture-ping"], timeout: 5)
        guard ping.terminationStatus == 0 else {
            throw NSError(domain: "OpenedCheckpointFixture", code: Int(ping.terminationStatus),
                userInfo: [NSLocalizedDescriptionKey: "The checkpoint test fixture could not run using PATH python3."])
        }
        return script
    }

    @Test func canonicalReferencesDoNotChangeDownloadsOrFiles() throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let (prefs, name) = defaults(); defer { prefs.removePersistentDomain(forName: name) }
        let checkpoint = root.appendingPathComponent("checkpoint with spaces ")
        try FileManager.default.createDirectory(at: checkpoint, withIntermediateDirectories: true)
        let weights = checkpoint.appendingPathComponent("model.safetensors")
        try Data("unchanged external bytes".utf8).write(to: weights)
        let alias = root.appendingPathComponent("alias")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: checkpoint)
        ModelsFolderPreference.setStoredPath("/original download folder", defaults: prefs)
        try OpenedCheckpoints.register(check(checkpoint.path), defaults: prefs)
        try OpenedCheckpoints.register(check(alias.path), defaults: prefs)
        #expect(OpenedCheckpoints.paths(defaults: prefs) == [OpenedCheckpoints.canonicalPath(checkpoint.path)!])
        #expect(ModelsFolderPreference.storedPath(defaults: prefs) == "/original download folder")
        OpenedCheckpoints.remove(OpenedCheckpoints.canonicalPath(checkpoint.path)!, defaults: prefs)
        #expect(OpenedCheckpoints.paths(defaults: prefs).isEmpty)
        #expect(try Data(contentsOf: weights) == Data("unchanged external bytes".utf8))
        #expect(ModelsFolderPreference.storedPath(defaults: prefs) == "/original download folder")
    }

    @Test func inventoryOwnershipAndDistinctPathsArePreserved() throws {
        let managed = ModelEntry(alias: "managed", hfRepo: "org/model", sizeOnDisk: nil, cached: true)
        let external = ModelEntry(alias: "external", hfRepo: "org/external", sizeOnDisk: nil, cached: true, isExternal: true)
        let checks = try [check("/a/same", known: "managed"), check("/b/same", known: "external"),
                          check("/c/same"), check("/d/same"), check("/c/same")]
        let entries = OpenedCheckpoints.merging([managed, external], checks: checks)
        #expect(entries.count == 4)
        #expect(entries[0] == managed)
        #expect(entries[1] == external)
        for entry in entries.suffix(2) {
            #expect(entry.hfRepo == nil)
            #expect(entry.isExternal && entry.cached && entry.isUncataloged)
            #expect(ModelSelectionPurpose.chat.accepts(entry))
        }
        #expect(Set(entries.suffix(2).map(\.alias)) == ["/c/same", "/d/same"])
        let stale = ModelEntry(alias: "managed", hfRepo: "org/model", sizeOnDisk: nil, cached: false)
        let recovered = OpenedCheckpoints.merging([stale], checks: [checks[0]])
        #expect(recovered.count == 1)
        #expect(recovered[0].isExternal && recovered[0].hfRepo == nil)
        #expect(recovered[0].alias == "/a/same")
    }

    @Test func invalidReferencesAndBoundsFailWithoutLosingExistingList() throws {
        let (prefs, name) = defaults(); defer { prefs.removePersistentDomain(forName: name) }
        prefs.set(["relative", "/bad\npath", "/a", "/a"], forKey: OpenedCheckpoints.storageKey)
        #expect(OpenedCheckpoints.paths(defaults: prefs) == ["/a"])
        #expect(OpenedCheckpoints.canonicalPath("/" + String(repeating: "x", count: 4096)) == nil)
        let rejected = try check("/rejected", ready: false, reason: "missing_chat_template")
        #expect(throws: LocalCheckpointError.self) { try OpenedCheckpoints.register(rejected, defaults: prefs) }
        #expect(OpenedCheckpoints.paths(defaults: prefs) == ["/a"])
        prefs.set((0..<32).map { "/checkpoint-\($0)" }, forKey: OpenedCheckpoints.storageKey)
        #expect(throws: LocalCheckpointError.self) { try OpenedCheckpoints.register(check("/extra"), defaults: prefs) }
        #expect(OpenedCheckpoints.paths(defaults: prefs).count == 32)
        #expect(OpenedCheckpoints.merging([], checks: [rejected]).isEmpty)
    }

    @Test func actualRunnerRejectsOldMalformedAndRetargetedResponses() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let binary = try await inspector(root)
        let checks = try await ModelCatalog.inspectLocalCheckpoints(paths: [OpenedCheckpoints.canonicalPath(root.path)!], binary: binary)
        #expect(checks[0].certified)
        let rejected = root.appendingPathComponent("reject")
        try FileManager.default.createDirectory(at: rejected, withIntermediateDirectories: true)
        let denied = try await ModelCatalog.inspectLocalCheckpoints(paths: [OpenedCheckpoints.canonicalPath(rejected.path)!], binary: binary)
        #expect(!denied[0].ready && denied[0].reason == "incomplete_checkpoint")
        for mode in ["unsupported", "malformed"] {
            try Data(mode.utf8).write(to: root.appendingPathComponent("mode.txt"))
            await #expect(throws: LocalCheckpointError.self) {
                try await ModelCatalog.inspectLocalCheckpoints(paths: [OpenedCheckpoints.canonicalPath(root.path)!], binary: binary)
            }
        }
        try Data("retarget".utf8).write(to: root.appendingPathComponent("mode.txt"))
        let retargeted = try await ModelCatalog.inspectLocalCheckpoints(paths: [OpenedCheckpoints.canonicalPath(root.path)!], binary: binary)
        #expect(!retargeted[0].ready && retargeted[0].reason == "identity_changed")
        await #expect(throws: LocalCheckpointError.self) {
            try await ModelCatalog.inspectLocalCheckpoints(paths: [], binary: binary)
        }
    }

    @Test func catalogLoadRechecksReferencesAndFailsClosedOnMissingOrOldEngine() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let binary = try await inspector(root)
        let selected = root.appendingPathComponent("checkpoint")
        try FileManager.default.createDirectory(at: selected, withIntermediateDirectories: true)
        let canonical = OpenedCheckpoints.canonicalPath(selected.path)!
        let entries = await ModelCatalog.load(binary: binary, openedPaths: [canonical])
        let opened = try #require(entries.first { $0.alias == canonical })
        #expect(opened.isExternal && opened.hfRepo == nil && opened.cached)
        try FileManager.default.removeItem(at: selected)
        let missing = await ModelCatalog.load(binary: binary, openedPaths: [canonical])
        #expect(!missing.contains { $0.alias == canonical })
        try Data("unsupported".utf8).write(to: root.appendingPathComponent("mode.txt"))
        let old = await ModelCatalog.load(binary: binary, openedPaths: [canonical])
        #expect(!old.contains { $0.alias == canonical })
    }

    @Test(arguments: [520.0, 900.0]) func nativeOpenAndRemoveReference(width: Double) async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let (prefs, name) = defaults(); defer { prefs.removePersistentDomain(forName: name) }
        let binary = try await inspector(root)
        let selected = root.appendingPathComponent("same name")
        try FileManager.default.createDirectory(at: selected, withIntermediateDirectories: true)
        try Data("external".utf8).write(to: selected.appendingPathComponent("weights-sentinel"))
        ModelsFolderPreference.setStoredPath("/unchanged download folder", defaults: prefs)
        let downloads = DownloadManager(binaryPath: binary, binaryLocator: { binary }, cacheAccessCheck: { nil })
        let server = ServerManager(testingState: .stopped, binaryPath: binary, sessionDefaults: prefs)
        let stage = viewport(SettingsModelManagementPanel(checkpointDefaults: prefs, checkpointDirectory: { selected })
            .environment(server).environment(downloads), size: CGSize(width: width, height: 720))
        try await stage.waitForIdentifier("Settings.ModelManagement.OpenCheckpoint")
        try stage.press("Settings.ModelManagement.OpenCheckpoint")
        let canonical = OpenedCheckpoints.canonicalPath(selected.path)!
        try await stage.wait(for: "registered exact path") { OpenedCheckpoints.paths(defaults: prefs) == [canonical] }
        try await stage.waitForIdentifierGone("Settings.ModelManagement.CheckingCheckpoint")
        #expect(stage.treeText().contains(canonical))
        if let folder = ProcessInfo.processInfo.environment["RAPID_CHECKPOINT_REVIEW_DIR"] {
            try FileManager.default.createDirectory(atPath: folder, withIntermediateDirectories: true)
            try stage.writeScreenshot(to: URL(fileURLWithPath: folder).appendingPathComponent("opened-\(Int(width)).png"))
        }
        try await stage.wait(for: "remove reference actionable") {
            try? stage.press("Settings.ModelManagement.RemoveCheckpoint.\(canonical)")
            return OpenedCheckpoints.paths(defaults: prefs).isEmpty
        }
        #expect(FileManager.default.fileExists(atPath: selected.appendingPathComponent("weights-sentinel").path))
        #expect(ModelsFolderPreference.storedPath(defaults: prefs) == "/unchanged download folder")
        #expect(downloads.jobs.isEmpty)
    }

    @Test func nativePickerCancelDoesNotRegisterOrChangeDownloads() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let (prefs, name) = defaults(); defer { prefs.removePersistentDomain(forName: name) }
        let binary = try await inspector(root)
        let stage = viewport(SettingsModelManagementPanel(checkpointDefaults: prefs, checkpointDirectory: { nil })
            .environment(ServerManager(testingState: .stopped, binaryPath: binary, sessionDefaults: prefs))
            .environment(DownloadManager(binaryPath: binary, binaryLocator: { binary }, cacheAccessCheck: { nil })), size: CGSize(width: 900, height: 720))
        try await stage.waitForIdentifier("Settings.ModelManagement.OpenCheckpoint")
        try stage.press("Settings.ModelManagement.OpenCheckpoint")
        #expect(OpenedCheckpoints.paths(defaults: prefs).isEmpty)
        #expect(ModelsFolderPreference.storedPath(defaults: prefs) == nil)
    }

    @Test func invalidPickedPathShowsReasonBeforeInspectOrRegistration() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let (prefs, domain) = defaults(); defer { prefs.removePersistentDomain(forName: domain) }
        let binary = try await inspector(root)
        let invalid = root.appendingPathComponent("bad\nname")
        try FileManager.default.createDirectory(at: invalid, withIntermediateDirectories: true)
        let downloads = DownloadManager(binaryPath: binary, binaryLocator: { binary }, cacheAccessCheck: { nil })
        let stage = viewport(SettingsModelManagementPanel(checkpointDefaults: prefs, checkpointDirectory: { invalid })
            .environment(ServerManager(testingState: .stopped, binaryPath: binary, sessionDefaults: prefs))
            .environment(downloads), size: CGSize(width: 520, height: 720))
        try await stage.waitForIdentifier("Settings.ModelManagement.OpenCheckpoint")
        try stage.press("Settings.ModelManagement.OpenCheckpoint")
        try await stage.waitForIdentifier("Settings.ModelManagement.DismissCheckpointError")
        #expect(stage.treeText().contains(LocalCheckpointError.rejected("invalid_path").localizedDescription))
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("inspect-called").path))
        #expect(OpenedCheckpoints.paths(defaults: prefs).isEmpty && downloads.jobs.isEmpty)
    }

    @Test(arguments: [520.0, 900.0]) func rejectedDirectoryHasVisibleLocalRemedyAndNoRegistration(width: Double) async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let (prefs, name) = defaults(); defer { prefs.removePersistentDomain(forName: name) }
        let binary = try await inspector(root)
        let selected = root.appendingPathComponent("reject")
        try FileManager.default.createDirectory(at: selected, withIntermediateDirectories: true)
        let downloads = DownloadManager(binaryPath: binary, binaryLocator: { binary }, cacheAccessCheck: { nil })
        let stage = viewport(SettingsModelManagementPanel(checkpointDefaults: prefs, checkpointDirectory: { selected })
            .environment(ServerManager(testingState: .stopped, binaryPath: binary, sessionDefaults: prefs))
            .environment(downloads), size: CGSize(width: width, height: 720))
        try await stage.waitForIdentifier("Settings.ModelManagement.OpenCheckpoint")
        try stage.press("Settings.ModelManagement.OpenCheckpoint")
        try await stage.waitForIdentifier("Settings.ModelManagement.DismissCheckpointError")
        #expect(stage.treeText().contains("missing complete MLX checkpoint weights"))
        #expect(OpenedCheckpoints.paths(defaults: prefs).isEmpty && downloads.jobs.isEmpty)
        if let folder = ProcessInfo.processInfo.environment["RAPID_CHECKPOINT_REVIEW_DIR"] {
            try FileManager.default.createDirectory(atPath: folder, withIntermediateDirectories: true)
            try stage.writeScreenshot(to: URL(fileURLWithPath: folder).appendingPathComponent("rejected-\(Int(width)).png"))
        }
        try stage.press("Settings.ModelManagement.DismissCheckpointError")
        try await stage.waitForIdentifierGone("Settings.ModelManagement.DismissCheckpointError")
    }

    @Test func closingPanelDoesNotPersistALateInspectionReply() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let (prefs, name) = defaults(); defer { prefs.removePersistentDomain(forName: name) }
        let binary = try await inspector(root)
        try Data("slow".utf8).write(to: root.appendingPathComponent("mode.txt"))
        var stage: GoldenStage? = viewport(SettingsModelManagementPanel(checkpointDefaults: prefs, checkpointDirectory: { root })
            .environment(ServerManager(testingState: .stopped, binaryPath: binary, sessionDefaults: prefs))
            .environment(DownloadManager(binaryPath: binary, binaryLocator: { binary }, cacheAccessCheck: { nil })), size: CGSize(width: 900, height: 720))
        try await stage!.waitForIdentifier("Settings.ModelManagement.OpenCheckpoint")
        try stage!.press("Settings.ModelManagement.OpenCheckpoint")
        try await stage!.wait(for: "inspection subprocess started") {
            FileManager.default.fileExists(atPath: root.appendingPathComponent("started").path)
        }
        stage = nil
        try await Task.sleep(for: .milliseconds(800))
        #expect(OpenedCheckpoints.paths(defaults: prefs).isEmpty)
    }

    @Test(arguments: ["invalid_path", "unsafe_symlink", "incomplete_checkpoint", "missing_chat_template", "unsupported_chat_model", "runtime_inventory_unavailable"])
    func typedFailureOffersAnActionableLocalRemedy(reason: String) {
        let message = LocalCheckpointError.rejected(reason).localizedDescription
        #expect(!message.isEmpty)
        #expect(!message.lowercased().contains("download"))
        #expect(LocalCheckpointError.limit.localizedDescription.contains("reference"))
        #expect(LocalCheckpointError.unavailable.localizedDescription.contains("engine"))
    }
}
