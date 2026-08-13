import AppKit
import Foundation
import Observation

/// Hosts the in-app DMG installer is allowed to download from. The
/// initial URL must match one of these AND every HTTP redirect must
/// stay inside the set. Without the redirect-check, the host-allowlist
/// on the initial URL is bypassable by a server that responds with a
/// 302 to an attacker origin. [codex audit r1 Installer.swift:319]
///
/// `dl.rapidmlx.com` is the production R2 origin (v0.6.12 cutover —
/// see UpdateChecker.swift). The GitHub Releases CDN hosts
/// (`objects.githubusercontent.com`, `release-assets.githubusercontent.com`)
/// remain because v0.5.x in-flight clients still see GH URLs in
/// historical manifests, and because the validator must continue to
/// accept the legacy shape until those clients are off-rolls.
///
/// `rapidmlx.com` and `www.rapidmlx.com` are present because the
/// manifest's `html_url` (release-notes link) points at the public
/// landing page. The `UpdateCheckerTests` superset invariant requires
/// every release-allowlist host to also appear here; we keep them in
/// sync even though no DMG is actually served from `rapidmlx.com`
/// (only `dl.rapidmlx.com`). If `rapidmlx.com` ever starts hosting
/// user-supplied content, tighten this to a path-pin rather than a
/// bare host match (see codex r1 N1 on PR #225).
let updateDownloadHostAllowlist: Set<String> = [
    "github.com",
    "www.github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
    "dl.rapidmlx.com",
    "rapidmlx.com",
    "www.rapidmlx.com",
]

/// In-app DMG installer for Rapid.app self-update.
///
/// Companion to ``UpdateChecker`` (which only polls + tells the UI a
/// newer release exists). ``Installer`` drives the actual "download
/// the DMG, swap the bundle in place, relaunch" flow that
/// ``UpdateChecker``'s long-standing comment said was deliberately
/// out of scope:
///
///   > Deliberately not an "Install now" button: replace-while-running
///   > has too many failure modes to ship without a code-signing story.
///
/// We accept those failure modes by gating each step behind a
/// fail-closed check (codesign verify on the mounted bundle, atomic
/// rename with rollback in the post-exit helper script) and surfacing
/// every stage through ``stage`` so the UI can render real progress
/// instead of a single opaque spinner. Modelled after Ollama's
/// ``app/updater/updater_darwin.go`` flow — DMG → mount → copy →
/// detached helper → relaunch. It remains as the migration fallback for local
/// and internal builds that do not carry a Sparkle public key.
///
/// Threat model:
///   * The DMG URL comes from the update worker payload that
///     ``UpdateChecker`` already trusts. HTTPS to GitHub Release CDN
///     is the transport boundary; we don't add a separate SHA256
///     channel today (worker schema_version=1 doesn't surface one).
///     v2 of the worker contract will add ``dmg_sha256`` and this
///     module will start enforcing it — the verify stage already
///     exists in the state machine specifically so that addition is
///     a one-line wiring change.
///   * ``codesign --verify`` (``--strict --deep``) is run on the
///     mounted bundle before we stage it. Production releases are
///     Developer ID signed + hardened-runtime + notarised + stapled
///     (PR #13 + ``.github/workflows/release.yml``) — verified live
///     on v0.5.16 via ``codesign -dvv`` (``Authority=Developer ID
///     Application: MachineFi Inc. (73WQ7ZGSWC)``, ``Notarization
///     Ticket=stapled``, ``spctl: accepted source=Notarized Developer
///     ID``). The verify call catches a corrupt download, a
///     mid-flight bit flip, AND a structurally-tampered resource
///     payload (``--strict`` keeps the resource-rules check on).
///     **Remaining fallback-only gap:** the verify call does
///     NOT pass ``-R "anchor apple generic and identifier
///     \"com.rapidmlx.rapid\" and certificate leaf[subject.OU] =
///     73WQ7ZGSWC"``, so a DMG signed with a *different* valid
///     Developer ID team would also pass. The HTTPS host-allowlist
///     (see top of file) closes the network-substitution path that
///     would otherwise let such a DMG reach this verifier. Signed production
///     builds use Sparkle's EdDSA verification instead.
///   * The post-exit helper is a small bash script we write to disk
///     ourselves and exec via ``/bin/bash``. It never touches anything
///     outside of two paths the parent already chose (the staged copy
///     and the installed location), so a compromised script source is
///     not a useful attack: anyone who can write that file already has
///     filesystem access to swap the app directly.
@MainActor
@Observable
final class Installer {
    /// Per-stage progress reported to the UI. ``downloading`` carries
    /// fractional bytes (0..1) when known; the rest are coarse beats
    /// the user can read as "is anything still happening?" lights.
    /// ``failed`` is sticky — the UI surfaces a "Try again" button
    /// that calls ``reset()`` to drop back to ``.idle``.
    enum Stage: Equatable, Sendable {
        case idle
        case downloading(progress: Double)
        case verifying
        case installing
        case relaunching
        case failed(String)
    }

    private(set) var stage: Stage = .idle

    /// True while a non-terminal stage is active. Used by the UI to
    /// disable a second "Install" click and by the menu bar item to
    /// switch to a "Updating…" label.
    var isRunning: Bool {
        switch stage {
        case .idle, .failed: return false
        case .downloading, .verifying, .installing, .relaunching: return true
        }
    }

    // MARK: - Injection points (production defaults below)

    /// Download a DMG to a local cache path. Reports fractional
    /// progress (0..1) via the callback so the UI can drive a
    /// determinate ``ProgressView``. The callback is fired on the
    /// main actor.
    typealias Downloader = @Sendable (
        _ remote: URL,
        _ progress: @MainActor @Sendable (Double) -> Void
    ) async throws -> URL

    /// Mount the DMG via ``hdiutil attach`` and return the mount
    /// point. Caller is responsible for unmounting via ``unmounter``.
    typealias Mounter = @Sendable (_ dmg: URL) async throws -> URL

    /// Locate ``Rapid.app`` inside a mounted DMG volume. Returns the
    /// full bundle URL.
    typealias AppFinder = @Sendable (_ mountPoint: URL) async throws -> URL

    /// Run ``codesign --verify`` on the mounted bundle. Throws on any
    /// non-zero exit; the message bubbles up to ``.failed``.
    typealias CodesignVerifier = @Sendable (_ app: URL) async throws -> Void

    /// Copy the mounted bundle into a staging directory outside the
    /// DMG so the helper script doesn't need the volume mounted. The
    /// returned URL points at the staged ``Rapid.app``. ``installed``
    /// is the path the new bundle will be renamed into; the copier
    /// chooses a staging location on the SAME volume so the helper's
    /// final ``mv`` is always an atomic same-volume rename, never a
    /// cross-volume copy-then-unlink (which is not crash-safe).
    /// [codex r1 #2]
    typealias StagedCopier = @Sendable (
        _ mountedApp: URL,
        _ installed: URL
    ) async throws -> URL

    /// Detach the DMG volume. Best-effort; we don't surface its
    /// errors because by this point the staged copy is independent.
    typealias Unmounter = @Sendable (_ mountPoint: URL) async -> Void

    /// Spawn the detached helper that waits for the parent PID to
    /// exit, swaps ``staged`` into ``installed``, and relaunches.
    /// Throws if the spawn itself fails — once spawned, errors land
    /// in the helper's log (not surfaced to the user because we've
    /// already torn down).
    typealias HelperSpawner = @Sendable (
        _ staged: URL,
        _ installed: URL,
        _ parentPID: Int32
    ) async throws -> Void

    /// Final shutdown hook. Defaults to ``NSApp.terminate``. Tests
    /// inject a counter so they can assert the swap helper was
    /// spawned BEFORE the terminate call.
    typealias Terminator = @MainActor @Sendable () -> Void

    private let downloader: Downloader
    private let mounter: Mounter
    private let appFinder: AppFinder
    private let codesignVerifier: CodesignVerifier
    private let stagedCopier: StagedCopier
    private let unmounter: Unmounter
    private let helperSpawner: HelperSpawner
    private let terminator: Terminator

    /// Where the currently-running .app lives. We swap a new bundle
    /// into this exact path so users running from
    /// ``/Applications/Rapid.app`` keep that path, users running from
    /// ``~/Downloads/Rapid.app`` keep that path.
    private let installedAppURL: URL

    init(
        installedAppURL: URL = Bundle.main.bundleURL,
        downloader: Downloader? = nil,
        mounter: Mounter? = nil,
        appFinder: AppFinder? = nil,
        codesignVerifier: CodesignVerifier? = nil,
        stagedCopier: StagedCopier? = nil,
        unmounter: Unmounter? = nil,
        helperSpawner: HelperSpawner? = nil,
        terminator: Terminator? = nil
    ) {
        // Canonicalise the install destination before it travels into
        // the helper script. ``Bundle.main.bundleURL`` resolves the
        // running .app's literal launch path, which may traverse a
        // symlink (LaunchServices is happy to launch from
        // ``/Users/u/Downloads/Rapid Symlink`` even when the bundle
        // lives elsewhere). The helper later runs ``mv`` / ``rm -rf``
        // against that path; without canonicalisation a malicious
        // symlink-as-bundle-parent could redirect the swap target.
        // [codex audit r1 Installer.swift:148]
        self.installedAppURL = installedAppURL.resolvingSymlinksInPath()
        self.downloader = downloader ?? Installer.defaultDownloader
        self.mounter = mounter ?? Installer.defaultMounter
        self.appFinder = appFinder ?? Installer.defaultAppFinder
        self.codesignVerifier = codesignVerifier ?? Installer.defaultCodesignVerifier
        self.stagedCopier = stagedCopier ?? Installer.defaultStagedCopier
        self.unmounter = unmounter ?? Installer.defaultUnmounter
        self.helperSpawner = helperSpawner ?? Installer.defaultHelperSpawner
        self.terminator = terminator ?? { NSApp.terminate(nil) }
    }

    /// Reset a failed install back to ``.idle`` so the UI can offer
    /// the user a retry. No-op while a non-terminal stage is in
    /// flight — ``isRunning`` is the authoritative gate.
    func reset() {
        guard !isRunning else { return }
        stage = .idle
    }

    /// Drive the full install flow. Returns to the caller after the
    /// terminator fires (or immediately on failure). Safe to call
    /// concurrently — overlapping calls are coalesced via the
    /// ``isRunning`` guard.
    func install(from remoteDMG: URL) async {
        guard !isRunning else { return }
        // Set the non-idle stage IMMEDIATELY so a second install()
        // call during cleanup hits ``isRunning == true`` and bails.
        // r1 awaited cleanup with ``.value`` BEFORE this assignment,
        // which opened a TOCTOU window: a double-tap could start two
        // parallel pipelines that both raced to swap the same .app.
        // [codex r2 #1]
        stage = .downloading(progress: 0)
        // GC orphaned staged bundles / DMGs left over from a prior
        // retry loop. Fire-and-forget at utility priority — the
        // download starts immediately. The mtime filter inside
        // ``cleanupStaleArtifacts`` ensures we never delete the
        // fresh download/staged bundle this install is about to
        // create. [codex r2 #6 / r1 #7]
        let installedURL = installedAppURL
        Task.detached(priority: .utility) {
            Installer.cleanupStaleArtifacts(near: installedURL)
        }
        do {
            let dmg = try await downloader(remoteDMG) { [weak self] progress in
                guard let self else { return }
                // Clamp before assignment so a misbehaving downloader
                // can't push the UI past 1.0.
                let clamped = max(0, min(1, progress))
                self.stage = .downloading(progress: clamped)
            }
            stage = .verifying
            let mountPoint = try await mounter(dmg)
            // Wrap the rest in a defer that always unmounts —
            // codesign failures and stage-copy failures BOTH need
            // the volume detached or hdiutil leaks until reboot.
            let mountedApp: URL
            let staged: URL
            do {
                mountedApp = try await appFinder(mountPoint)
                try await codesignVerifier(mountedApp)
                stage = .installing
                staged = try await stagedCopier(mountedApp, installedAppURL)
            } catch {
                await unmounter(mountPoint)
                throw error
            }
            // Detach BEFORE spawning the helper. Failure is logged
            // (production unmounter NSLogs to Console.app via the
            // hdiutil exit path) but not surfaced to the UI — by
            // this point the staged bundle is independent of the
            // volume and the install can still succeed; a phantom
            // mount surviving past the relaunch is a Console-trail
            // hygiene issue, not a correctness one. [codex r1 #6]
            await unmounter(mountPoint)
            stage = .relaunching
            let pid = ProcessInfo.processInfo.processIdentifier
            try await helperSpawner(staged, installedAppURL, pid)
            terminator()
        } catch {
            stage = .failed((error as? LocalizedError)?.errorDescription
                            ?? error.localizedDescription)
        }
    }
}

// MARK: - Errors

enum InstallerError: LocalizedError {
    case invalidURL
    case downloadFailed(String)
    case mountFailed(String)
    case appNotFoundInVolume(String)
    case codesignFailed(String)
    case copyFailed(String)
    case helperSpawnFailed(String)

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid update download URL."
        case .downloadFailed(let s):
            return "Download failed: \(s)"
        case .mountFailed(let s):
            return "Couldn't mount the update DMG: \(s)"
        case .appNotFoundInVolume(let s):
            return "Rapid.app missing from update DMG: \(s)"
        case .codesignFailed(let s):
            return "The downloaded update failed signature verification: \(s)"
        case .copyFailed(let s):
            return "Couldn't stage the update for install: \(s)"
        case .helperSpawnFailed(let s):
            return "Couldn't launch the install helper: \(s)"
        }
    }
}

// MARK: - Production implementations

extension Installer {
    /// Download to ``~/Library/Caches/Rapid/updates/<uuid>.dmg``.
    /// Returns the final path. Uses a delegate-driven
    /// ``URLSession.downloadTask(with:completionHandler:)`` so we
    /// can report progress to the UI without buffering the whole
    /// DMG in memory. [codex r2 #4 — was stale after r1's switch
    /// off the ``download(for:)`` async convenience]
    nonisolated static let defaultDownloader: Downloader = { url, progress in
        // The ``progress`` parameter is non-escaping at this call site
        // (closure params in function-typed lets are non-escaping by
        // default), but ``URLSessionDownloadDelegate`` retains the
        // callback past the call boundary. ``withoutActuallyEscaping``
        // is the Swift-supported bridge: we promise the delegate
        // doesn't outlive this function, which is true because we
        // ``finishTasksAndInvalidate`` before returning.
        return try await withoutActuallyEscaping(progress) { escapingProgress in
            try await Installer.runDownload(
                url: url,
                progress: escapingProgress
            )
        }
    }

    /// Production-side worker that actually owns the URLSession +
    /// delegate. Split out so ``defaultDownloader`` can satisfy the
    /// ``Downloader`` typealias (non-escaping closure param) and we
    /// can pass an explicitly-``@escaping`` progress closure here
    /// where the delegate needs to capture it.
    ///
    /// Uses ``URLSession.downloadTask(with:completionHandler:)``
    /// rather than the ``URLSession.download(for:)`` async
    /// convenience because the latter's internal delegate doesn't
    /// reliably route ``didWriteData`` callbacks to a session-level
    /// delegate on macOS 14.0-14.2 — net result there is a progress
    /// bar pinned at 0% until the download completes. The
    /// completion-handler API guarantees the session-level delegate
    /// receives the progress callbacks. [codex r1 #8]
    nonisolated static func runDownload(
        url: URL,
        progress: @escaping @MainActor @Sendable (Double) -> Void
    ) async throws -> URL {
        // Validate the initial URL before we ever touch URLSession.
        // The DMG URL ultimately rides in from a CF Worker payload —
        // a compromised Worker or a man-in-the-middle on a non-HTTPS
        // path could otherwise redirect us at an attacker-controlled
        // origin and we'd happily download whatever it served. The
        // download-host allowlist additionally constrains the FINAL
        // host after redirects via DownloadProgressDelegate's
        // urlSession(_:task:willPerformHTTPRedirection:) hook.
        // [codex audit r1 Installer.swift:319]
        guard let scheme = url.scheme?.lowercased(), scheme == "https" else {
            throw InstallerError.downloadFailed("non-HTTPS URL: \(url)")
        }
        if url.user != nil || url.password != nil {
            throw InstallerError.downloadFailed("URL carries userinfo")
        }
        guard let host = url.host?.lowercased(),
              updateDownloadHostAllowlist.contains(host) else {
            throw InstallerError.downloadFailed(
                "host \(url.host ?? "<nil>") is not in the update-download allowlist"
            )
        }
        let cacheDir = try Installer.cachesDirectory()
        let dest = cacheDir.appendingPathComponent("Rapid-update-\(UUID().uuidString).dmg")
        var request = URLRequest(url: url)
        request.cachePolicy = .reloadIgnoringLocalCacheData
        request.timeoutInterval = 60
        request.setValue("Rapid-Desktop/\(UpdateChecker.bundleVersion())",
                         forHTTPHeaderField: "User-Agent")
        let final: URL = try await withCheckedThrowingContinuation { cont in
            let delegate = DownloadProgressDelegate(progress: progress)
            let session = URLSession(
                configuration: .default,
                delegate: delegate,
                delegateQueue: nil
            )
            let task = session.downloadTask(with: request) { tempURL, response, error in
                defer { session.finishTasksAndInvalidate() }
                if let error {
                    cont.resume(throwing: InstallerError.downloadFailed(error.localizedDescription))
                    return
                }
                guard let tempURL else {
                    cont.resume(throwing: InstallerError.downloadFailed("no temp file"))
                    return
                }
                if let http = response as? HTTPURLResponse,
                   !(200..<300).contains(http.statusCode) {
                    try? FileManager.default.removeItem(at: tempURL)
                    cont.resume(throwing: InstallerError.downloadFailed("HTTP \(http.statusCode)"))
                    return
                }
                do {
                    if FileManager.default.fileExists(atPath: dest.path) {
                        try FileManager.default.removeItem(at: dest)
                    }
                    try FileManager.default.moveItem(at: tempURL, to: dest)
                } catch {
                    cont.resume(throwing: InstallerError.downloadFailed(error.localizedDescription))
                    return
                }
                cont.resume(returning: dest)
            }
            task.resume()
        }
        // Final 100% — last delegate fire may have been slightly
        // under depending on total-bytes-expected accuracy.
        await MainActor.run { progress(1.0) }
        return final
    }

    /// ``hdiutil attach -nobrowse -noverify -mountrandom /tmp <dmg>``.
    /// ``-mountrandom`` picks a unique mount point inside the given
    /// directory so we can run multiple installers without colliding
    /// on a sticky ``/Volumes/Rapid`` name.
    nonisolated static let defaultMounter: Mounter = { dmg in
        // Drop the previous ``-noverify`` flag: that disabled
        // checksum verification of the DMG container, so a corrupted
        // download (bit flip mid-stream, truncated bytes) would
        // happily mount and we'd only learn about the damage on the
        // later codesign step (if at all — codesign verifies the .app
        // contents, not the surrounding container). Verification adds
        // 1-3 seconds to mount but eliminates "partial DMG appears
        // valid" as a failure mode. [codex audit r1 Installer.swift:351]
        let (status, out, err) = try await runProcess(
            executable: "/usr/bin/hdiutil",
            arguments: ["attach", "-nobrowse",
                        "-mountrandom", NSTemporaryDirectory(),
                        "-plist", dmg.path]
        )
        guard status == 0 else {
            throw InstallerError.mountFailed("hdiutil exit \(status): \(err)")
        }
        guard let mount = parseHdiutilMountPoint(plistXML: out) else {
            throw InstallerError.mountFailed("could not parse hdiutil plist")
        }
        return URL(fileURLWithPath: mount)
    }

    /// Walk the mount point's top-level entries looking for a
    /// ``.app`` bundle. We accept any name (Rapid.app today, but a
    /// future build could rename to Rapid Studio.app etc.) — there
    /// should only ever be one bundle in our DMG.
    nonisolated static let defaultAppFinder: AppFinder = { mountPoint in
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(atPath: mountPoint.path) else {
            throw InstallerError.appNotFoundInVolume("empty volume")
        }
        let apps = entries.filter { $0.hasSuffix(".app") && !$0.hasPrefix(".") }
        guard let only = apps.first, apps.count == 1 else {
            throw InstallerError.appNotFoundInVolume(
                apps.isEmpty ? "no .app found" : "multiple .app entries: \(apps)"
            )
        }
        let candidate = mountPoint.appendingPathComponent(only)
        // Reject symlinked .apps inside the DMG: a malicious image
        // could ship a symlink pointing outside the mount (e.g. at
        // ``/Applications/Safari.app``), which the verify + copy
        // stages would then resolve before we caught the swap. Use
        // ``lstat`` semantics (resourceValues with isSymbolicLinkKey
        // and isDirectoryKey) so we see the link itself, not its
        // target. [codex audit r1 Installer.swift:395]
        let lstat = try candidate.resourceValues(
            forKeys: [.isSymbolicLinkKey, .isDirectoryKey]
        )
        if lstat.isSymbolicLink == true {
            throw InstallerError.appNotFoundInVolume(
                "DMG ships .app as symlink: \(only)"
            )
        }
        guard lstat.isDirectory == true else {
            throw InstallerError.appNotFoundInVolume(
                "DMG ships .app as non-directory: \(only)"
            )
        }
        // Defense-in-depth: also confirm the resolved-symlinks path
        // stays under the mount point. Belts-and-braces on the lstat
        // above — any future relaxation of the check above still
        // gets caught here.
        let mountResolved = mountPoint.resolvingSymlinksInPath().standardizedFileURL.path
        let candidateResolved = candidate.resolvingSymlinksInPath().standardizedFileURL.path
        let separator = mountResolved.hasSuffix("/") ? "" : "/"
        guard candidateResolved.hasPrefix(mountResolved + separator) else {
            throw InstallerError.appNotFoundInVolume(
                "candidate escapes mount: \(candidateResolved)"
            )
        }
        return candidate
    }

    /// ``codesign --verify <app>``. Verifies the bundle hash chain —
    /// also works for ad-hoc signed local-dev builds (the unsigned
    /// build path in ``scripts/build.sh``), because Apple's
    /// ``codesign`` re-derives the CDHash from the bundle contents
    /// and matches it against the embedded signature regardless of
    /// signing identity. Production releases are Developer ID signed
    /// + notarised + stapled — see the file header threat model for this
    /// fallback's residual identity-pin gap (no ``-R`` requirement string is
    /// passed).
    ///
    /// We deliberately do NOT pass ``--no-strict`` (which the
    /// original implementation did): ``--no-strict`` disables the
    /// resource-rules check, the very check that catches a tampered
    /// ``Contents/Resources/*.lproj`` payload. Since we have no
    /// SHA256 channel today, codesign-with-resource-check is the
    /// integrity gate. [codex r1 #11]
    nonisolated static let defaultCodesignVerifier: CodesignVerifier = { app in
        // ``--strict`` re-enables the resource-rules check (the
        // historical "weak verification" trap is omitting this) and
        // ``--deep`` walks every nested code resource. Without both
        // flags, a tampered ``Contents/Resources/*.lproj`` or a
        // swapped helper inside the bundle would pass verification
        // even though the bundle is internally inconsistent.
        // [codex audit r1 Installer.swift:421]
        //
        // We deliberately do NOT pass a ``--requirement`` here yet:
        // production builds today are signed with Developer ID
        // (PR #13) but the requirement-string lives in the signing
        // identity, not in code we can pin. The fallback could add
        // ``-R "anchor apple generic and
        // identifier \"com.rapidmlx.rapid\" and certificate
        // leaf[subject.OU] = <TEAMID>"`` so a tampered DMG that
        // re-signs ad-hoc instead of Developer ID can't pass this
        // gate. Signed production updates use Sparkle's EdDSA verification;
        // this strict-deep fallback still catches structural tampering and
        // bit-flip corruption.
        let (status, _, err) = try await runProcess(
            executable: "/usr/bin/codesign",
            arguments: ["--verify", "--strict", "--deep", app.path]
        )
        guard status == 0 else {
            throw InstallerError.codesignFailed(err.isEmpty ? "exit \(status)" : err)
        }
    }

    /// Copy the mounted bundle to a sibling of the installed
    /// bundle. Keeping staging on the same volume as the install
    /// target means the helper's final ``mv`` is always atomic;
    /// staging in ``~/Library/Caches`` (the original implementation)
    /// could fall back to a non-atomic copy-then-unlink across APFS
    /// volumes (rare but possible with FileVault'd external homes).
    /// The leading dot keeps the staged bundle Finder-hidden so the
    /// user doesn't see a transient ``Rapid.app`` ghost during the
    /// install window. [codex r1 #2]
    nonisolated static let defaultStagedCopier: StagedCopier = { mountedApp, installed in
        let staged = stagingURL(near: installed)
        let stagingParent = staged.deletingLastPathComponent()
        do {
            try FileManager.default.createDirectory(
                at: stagingParent,
                withIntermediateDirectories: true
            )
            if FileManager.default.fileExists(atPath: staged.path) {
                try FileManager.default.removeItem(at: staged)
            }
            try FileManager.default.copyItem(at: mountedApp, to: staged)
        } catch {
            throw InstallerError.copyFailed(error.localizedDescription)
        }
        return staged
    }

    /// Compute the staged-bundle URL — sibling of ``installed`` so
    /// rename-into-place is always same-volume.
    nonisolated static func stagingURL(near installed: URL) -> URL {
        let parent = installed.deletingLastPathComponent()
        return parent.appendingPathComponent(".rapid-staged-\(UUID().uuidString).app")
    }

    /// Best-effort GC of stale staging bundles, leftover backup
    /// anchors, and orphaned DMG downloads from prior install
    /// attempts. A user on a flaky network who retries 5 times
    /// before succeeding would otherwise accumulate ~1.5 GB of
    /// staged copies in ``~/Library/Caches/Rapid/updates`` with no
    /// surfaced cleanup. [codex r1 #7]
    ///
    /// **Mtime filter:** never delete anything modified in the last
    /// ``freshThreshold`` seconds (default 60). r2 made the cleanup
    /// fire-and-forget so it races the current install's fresh
    /// download + staged copy; without an age filter the cleanup
    /// could blow away the in-progress DMG mid-flight. 60s is
    /// comfortably longer than any DMG copy + download even on a
    /// slow network. [codex r2 #1/#6]
    ///
    /// Never throws — cleanup failures are not a reason to block an
    /// install. Every error path logs to ``NSLog`` for diagnostics.
    nonisolated static func cleanupStaleArtifacts(
        near installed: URL,
        freshThreshold: TimeInterval = 60,
        now: Date = Date()
    ) {
        let fm = FileManager.default
        let parent = installed.deletingLastPathComponent()
        let bundleName = installed.lastPathComponent
        if let entries = try? fm.contentsOfDirectory(atPath: parent.path) {
            for name in entries {
                guard name.hasPrefix(".rapid-staged-")
                        || name.hasPrefix(bundleName + ".old-") else { continue }
                let url = parent.appendingPathComponent(name)
                guard isStale(url: url, now: now, threshold: freshThreshold, fm: fm) else { continue }
                do {
                    try fm.removeItem(at: url)
                } catch {
                    NSLog("[Installer] cleanup skip %@: %@",
                          url.path, error.localizedDescription)
                }
            }
        }
        let cache: URL
        do {
            cache = try cachesDirectory()
        } catch {
            NSLog("[Installer] cleanup couldn't resolve cache dir: %@",
                  error.localizedDescription)
            return
        }
        if let entries = try? fm.contentsOfDirectory(atPath: cache.path) {
            for name in entries {
                guard name.hasPrefix("Rapid-update-") && name.hasSuffix(".dmg") else { continue }
                let url = cache.appendingPathComponent(name)
                guard isStale(url: url, now: now, threshold: freshThreshold, fm: fm) else { continue }
                do {
                    try fm.removeItem(at: url)
                } catch {
                    NSLog("[Installer] cleanup skip cache %@: %@",
                          url.path, error.localizedDescription)
                }
            }
        }
    }

    /// True if ``url``'s mtime is at least ``threshold`` seconds
    /// older than ``now``. Used by ``cleanupStaleArtifacts`` to
    /// guarantee the GC never touches an in-flight install's
    /// freshly-created files.
    nonisolated static func isStale(
        url: URL,
        now: Date,
        threshold: TimeInterval,
        fm: FileManager = .default
    ) -> Bool {
        guard let attrs = try? fm.attributesOfItem(atPath: url.path),
              let mtime = attrs[.modificationDate] as? Date else {
            // Can't read mtime → conservative: treat as stale only
            // if removing it is harmless. Keep the file to be safe.
            return false
        }
        return now.timeIntervalSince(mtime) > threshold
    }

    nonisolated static let defaultUnmounter: Unmounter = { mountPoint in
        // ``-force`` because the user may still have a Finder window
        // open on the volume; best-effort detach either way. Log any
        // failure to NSLog so a phantom mount shows up in Console.app
        // instead of silently lingering until reboot. [codex r1 #6]
        do {
            let (status, _, err) = try await runProcess(
                executable: "/usr/bin/hdiutil",
                arguments: ["detach", mountPoint.path, "-force"]
            )
            if status != 0 {
                NSLog("[Installer] hdiutil detach exit %d at %@: %@",
                      status, mountPoint.path, err)
            }
        } catch {
            NSLog("[Installer] hdiutil detach spawn failed at %@: %@",
                  mountPoint.path, error.localizedDescription)
        }
    }

    /// Write the post-exit helper script to a temp file, mark it
    /// executable, and spawn it detached with the parent PID + paths
    /// as positional args. Must NOT block on the helper — the parent
    /// is about to call ``terminator()``.
    nonisolated static let defaultHelperSpawner: HelperSpawner = { staged, installed, parentPID in
        let scriptURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("rapid-install-\(UUID().uuidString).sh")
        do {
            try Installer.relaunchHelperScript.write(
                to: scriptURL,
                atomically: true,
                encoding: .utf8
            )
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o755],
                ofItemAtPath: scriptURL.path
            )
        } catch {
            throw InstallerError.helperSpawnFailed(error.localizedDescription)
        }
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/bin/bash")
        task.arguments = [
            scriptURL.path,
            String(parentPID),
            staged.path,
            installed.path,
        ]
        // Drop stdio so the parent's pipes don't keep the child
        // alive past terminate; the script writes its own log to
        // ``~/Library/Logs/Rapid/installer.log``.
        task.standardInput = FileHandle.nullDevice
        task.standardOutput = FileHandle.nullDevice
        task.standardError = FileHandle.nullDevice
        do {
            try task.run()
        } catch {
            throw InstallerError.helperSpawnFailed(error.localizedDescription)
        }
        // Do NOT waitUntilExit — the helper outlives us by design.
    }

    /// Bash script that waits for the parent PID to exit, atomically
    /// swaps the staged bundle into the installed location with
    /// rollback on failure, and ``open``s the new bundle to relaunch.
    /// Kept as a Swift string constant so the test suite can assert
    /// its invariants without scraping a resource file.
    nonisolated static let relaunchHelperScript: String = """
#!/bin/bash
# rapid-install-and-relaunch helper
#
# Spawned detached by Installer.swift's HelperSpawner.
# Args: <parent_pid> <staged_app_path> <installed_app_path>
#
# Contract:
#   1. Wait up to 30s for parent_pid to exit (poll every 200ms).
#   2. If still alive, escalate TERM → KILL (gives the parent
#      a chance to flush state, but we never let a stuck old
#      build block an install indefinitely).
#   3. Rename installed → installed.old-$$ as a rollback anchor.
#   4. Rename staged → installed. On failure, remove the partial
#      target dir first, then roll back from the backup.
#   5. ``open`` the new bundle to relaunch.
#   6. Trash the rollback anchor on success.
#
# Logs land in ~/Library/Logs/Rapid/installer.log so a failed
# install can be diagnosed after the fact.
# Defensive HOME default: launchd-spawned children inherit launchd's
# environment, which may not carry HOME. Without this guard
# ``set -u`` aborts on the LOG_DIR= line below before logging is
# wired and the user sees the app quit with zero diagnostic.
# [codex r1 #4]
HOME="${HOME:-/Users/$(id -un)}"
set -u
PARENT_PID="${1:-}"
STAGED="${2:-}"
INSTALLED="${3:-}"
# Strip trailing slash on both paths so the BACKUP sibling and
# every subsequent rename stay consistent. Bundle.main.bundleURL
# doesn't add one in practice, but the helper has no defense in
# depth without this normalisation.  [codex r1 #1]
INSTALLED="${INSTALLED%/}"
STAGED="${STAGED%/}"
LOG_DIR="${HOME}/Library/Logs/Rapid"
LOG="${LOG_DIR}/installer.log"
mkdir -p "${LOG_DIR}" 2>/dev/null
exec >> "${LOG}" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] installer start pid=${PARENT_PID} staged=${STAGED} installed=${INSTALLED}"
if [ -z "${PARENT_PID}" ] || [ -z "${STAGED}" ] || [ -z "${INSTALLED}" ]; then
  echo "missing args; abort"
  exit 64
fi
if [ ! -d "${STAGED}" ]; then
  echo "staged bundle missing: ${STAGED}"
  exit 65
fi
# Wait up to 30s for parent to exit.
i=0
while [ $i -lt 150 ]; do
  if ! kill -0 "${PARENT_PID}" 2>/dev/null; then
    break
  fi
  sleep 0.2
  i=$((i+1))
done
# Escalate if still alive.
if kill -0 "${PARENT_PID}" 2>/dev/null; then
  echo "parent still alive after 30s, sending TERM"
  kill -TERM "${PARENT_PID}" 2>/dev/null || true
  sleep 1
  kill -KILL "${PARENT_PID}" 2>/dev/null || true
fi
BACKUP="${INSTALLED}.old-$$"
if [ -d "${INSTALLED}" ]; then
  if ! mv "${INSTALLED}" "${BACKUP}"; then
    echo "rename installed → backup failed"
    exit 66
  fi
fi
if ! mv "${STAGED}" "${INSTALLED}"; then
  echo "rename staged → installed failed; rolling back"
  # If the rename partly succeeded, the target now exists as a
  # half-written dir. ``mv`` refuses to overwrite a non-empty dir
  # on same-volume renames, so without an explicit ``rm -rf`` the
  # rollback silently leaves the user with a partial bundle.
  # [codex r1 #3]
  if [ -e "${INSTALLED}" ]; then
    if ! rm -rf "${INSTALLED}"; then
      echo "rollback failed: could not remove partial install"
      exit 68
    fi
  fi
  if [ -d "${BACKUP}" ]; then
    if ! mv "${BACKUP}" "${INSTALLED}"; then
      echo "rollback failed: could not restore backup"
      exit 69
    fi
    echo "rollback complete"
  fi
  exit 67
fi
# Quarantine bit on freshly-extracted-from-DMG bundles trips
# Gatekeeper on relaunch; clear it before ``open``. Ad-hoc signed
# bundles may still surface a one-time "Rapid.app is from the
# Internet" Gatekeeper dialog on first relaunch because we don't
# have a Developer ID to register with ``spctl --add``. That's a
# known limitation of the signing posture, not a script bug.
xattr -dr com.apple.quarantine "${INSTALLED}" 2>/dev/null || true
# Trash backup. Best-effort — leaving it doesn't break anything
# but burns disk on every update.
rm -rf "${BACKUP}" 2>/dev/null
echo "swap complete; launching ${INSTALLED}"
# Retry ``open`` once: LaunchServices transiently 404's when the
# terminated parent's record is still cached and a single failure
# leaves the app closed even though the swap succeeded. Bounded
# retry with a 1s gap covers that window without papering over real
# failures (a second exit code means the bundle is genuinely
# unlaunchable and the user needs to know). [codex audit r1
# Installer.swift:715]
if ! open "${INSTALLED}"; then
  sleep 1
  if ! open "${INSTALLED}"; then
    echo "open failed twice; user must relaunch manually"
    exit 70
  fi
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] installer done"
"""

    /// Build / return ``~/Library/Caches/Rapid/updates``.
    nonisolated static func cachesDirectory() throws -> URL {
        let base = try FileManager.default.url(
            for: .cachesDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let dir = base.appendingPathComponent("Rapid/updates", isDirectory: true)
        try FileManager.default.createDirectory(
            at: dir,
            withIntermediateDirectories: true
        )
        return dir
    }

    /// Pick the first non-empty ``mount-point`` out of an ``hdiutil
    /// attach -plist`` payload.
    ///
    /// Multi-partition DMGs emit one ``system-entities`` dict per
    /// partition; only the HFS+/APFS partition carries a non-empty
    /// ``mount-point``. The original string-scanning impl picked
    /// "the first ``<string>`` after the first ``<key>mount-point</key>``"
    /// which is correct for a single-partition DMG but returns an
    /// empty string (then nil → install fails) if a leading
    /// ``Apple_partition_scheme`` entity has its own empty
    /// ``mount-point`` key. Using ``PropertyListSerialization`` and
    /// scanning ALL entities for the first non-empty value handles
    /// every DMG layout the toolchain produces. [codex r1 #10]
    nonisolated static func parseHdiutilMountPoint(plistXML: String) -> String? {
        // Defensive: also accept the old string-scanning fallback
        // when the payload isn't valid plist (synthetic test inputs
        // sometimes ship a fragment rather than a full <plist>).
        if let propertyList = parseAsPlistAndExtract(plistXML),
           let first = firstNonEmptyMountPoint(in: propertyList) {
            return first
        }
        return scanFirstNonEmptyMountPointString(in: plistXML)
    }

    /// XML → property list. Returns nil if parsing fails (synthetic
    /// fragment, unexpected encoding, etc.) — the caller falls back
    /// to the legacy string scanner.
    nonisolated static func parseAsPlistAndExtract(_ xml: String) -> Any? {
        guard let data = xml.data(using: .utf8) else { return nil }
        return try? PropertyListSerialization.propertyList(
            from: data,
            options: [],
            format: nil
        )
    }

    /// Walk an arbitrary property-list tree, picking the first
    /// non-empty string under a ``mount-point`` key. Dictionary-or-
    /// array-shape-agnostic so future hdiutil output layouts still
    /// parse without code changes.
    nonisolated static func firstNonEmptyMountPoint(in value: Any) -> String? {
        if let dict = value as? [String: Any] {
            if let mp = dict["mount-point"] as? String, !mp.isEmpty {
                return mp
            }
            for (_, sub) in dict {
                if let found = firstNonEmptyMountPoint(in: sub) {
                    return found
                }
            }
        } else if let array = value as? [Any] {
            for entry in array {
                if let found = firstNonEmptyMountPoint(in: entry) {
                    return found
                }
            }
        }
        return nil
    }

    /// Legacy string scanner — kept as the fallback for fragments
    /// that aren't valid <plist>. Walks every ``<key>mount-point</key>``
    /// occurrence and returns the first non-empty ``<string>`` that
    /// follows.
    nonisolated static func scanFirstNonEmptyMountPointString(in plistXML: String) -> String? {
        var cursor = plistXML.startIndex
        while let keyRange = plistXML.range(of: "<key>mount-point</key>", range: cursor..<plistXML.endIndex) {
            let rest = plistXML[keyRange.upperBound...]
            guard let openRange = rest.range(of: "<string>") else { return nil }
            let afterOpen = rest[openRange.upperBound...]
            guard let closeRange = afterOpen.range(of: "</string>") else { return nil }
            let path = String(afterOpen[..<closeRange.lowerBound])
            if !path.isEmpty {
                return path
            }
            cursor = closeRange.upperBound
        }
        return nil
    }

    /// Spawn a process, await exit, return (status, stdout, stderr).
    /// Used by mount / verify only — the actual download streams
    /// through URLSession instead.
    ///
    /// **Pipe-buffer deadlock note.** The previous shape waited for
    /// process exit before draining the pipes. A child that emits
    /// more output than the OS pipe buffer (64 KiB by default) blocks
    /// on ``write(2)`` and never reaches exit, so ``waitUntilExit``
    /// hangs forever and the install spinner sits at "Verifying…"
    /// indefinitely. ``hdiutil attach -plist`` and ``codesign --deep
    /// --verify`` can both exceed 64 KiB on multi-partition DMGs /
    /// large bundles. The fix drains stdout and stderr concurrently
    /// on background queues while ``waitUntilExit`` runs, gated on a
    /// dispatch group so we don't return until BOTH drains finish AND
    /// the child has exited. [codex audit r1 Installer.swift:836]
    nonisolated static func runProcess(
        executable: String,
        arguments: [String]
    ) async throws -> (status: Int32, stdout: String, stderr: String) {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: executable)
        task.arguments = arguments
        let outPipe = Pipe()
        let errPipe = Pipe()
        task.standardOutput = outPipe
        task.standardError = errPipe
        do {
            try task.run()
        } catch {
            throw InstallerError.mountFailed(
                "spawn \(executable) failed: \(error.localizedDescription)"
            )
        }
        let group = DispatchGroup()
        nonisolated(unsafe) var outBuf = Data()
        nonisolated(unsafe) var errBuf = Data()
        group.enter()
        DispatchQueue.global(qos: .utility).async {
            // Crash-safe: readDataToEndOfFile() raises an uncatchable
            // NSException on a bad descriptor (SIGABRTs the process) if
            // the child's pipe FD races teardown. See readToEndSafely().
            outBuf = outPipe.fileHandleForReading.readToEndSafely()
            group.leave()
        }
        group.enter()
        DispatchQueue.global(qos: .utility).async {
            errBuf = errPipe.fileHandleForReading.readToEndSafely()
            group.leave()
        }
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            DispatchQueue.global().async {
                task.waitUntilExit()
                // Wait for both pipe drains to settle. The child may
                // have emitted the last bytes just before exit and we
                // need them in the returned strings.
                group.wait()
                cont.resume()
            }
        }
        return (
            task.terminationStatus,
            String(data: outBuf, encoding: .utf8) ?? "",
            String(data: errBuf, encoding: .utf8) ?? ""
        )
    }
}

// MARK: - URLSession download progress

/// Bridges ``URLSessionDownloadDelegate`` progress callbacks to a
/// main-actor closure the ``Installer`` UI binds. Kept private to
/// this file because nothing else needs to observe download progress.
private final class DownloadProgressDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    let progress: @MainActor @Sendable (Double) -> Void

    init(progress: @MainActor @Sendable @escaping (Double) -> Void) {
        self.progress = progress
    }

    /// Refuse every redirect whose final host isn't in the download
    /// allowlist. GitHub Releases 302's to a signed CDN URL on
    /// objects.githubusercontent.com / release-assets.…; a
    /// compromised Worker (or an attacker MITMing a non-HTTPS hop)
    /// trying to bend the download elsewhere lands here.
    /// [codex audit r1 Installer.swift:319]
    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        guard
            let url = request.url,
            url.scheme?.lowercased() == "https",
            url.user == nil,
            url.password == nil,
            let host = url.host?.lowercased(),
            updateDownloadHostAllowlist.contains(host)
        else {
            completionHandler(nil)
            return
        }
        completionHandler(request)
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard totalBytesExpectedToWrite > 0 else { return }
        let fraction = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
        let cb = progress
        Task { @MainActor in
            cb(fraction)
        }
    }

    /// Required protocol member. ``downloadTask(with:completionHandler:)``
    /// — the API we use — delivers the temp URL through the
    /// completion closure, and URLSession does NOT invoke this
    /// delegate method on a completion-handler task. The stub
    /// remains for the URLSessionDownloadDelegate protocol witness
    /// and for any future refactor that switches back to a
    /// delegate-only download. [codex r2 #3]
    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        // No-op — see docstring; the completion handler owns the
        // temp URL lifetime.
    }
}
