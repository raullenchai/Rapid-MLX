import Foundation
import Testing
@testable import Rapid

/// Contract for the FU-4 / PR #338 low-disk pre-flight check.
///
/// Pins:
/// - the production threshold (2 GB, F-LWT-1 0.6B re-derivation) so a
///   future drop / inflation is loud
/// - decision truth table — including the fail-open ``nil`` branch
///   (probe failure MUST NOT block the user)
/// - real-fs sanity that the statfs path wires up correctly on the
///   build host (we never compare to a magic-number — only that the
///   value is positive and Int64-shaped)
/// - banner copy helpers render the right GB numbers and read as a
///   single sentence
@MainActor
@Suite("DiskSpaceProbe — Quickstart low-disk pre-flight")
struct DiskSpaceProbeTests {

    // MARK: - Pinned threshold

    @Test("Quickstart required-bytes threshold is 2 GB (bonsai-1.7b-2bit re-derivation)")
    func thresholdPinnedAt2GB() {
        // Sized for bonsai-1.7b-2bit (~0.5 GB) × 1.5 transient +
        // 1 GB OS headroom, rounded up to 2 GB (ample margin — even
        // the older ~400 MB starter fit). If the Quickstart
        // alias is ever swapped to something materially bigger /
        // smaller, the threshold here MUST move with it — see
        // ``DiskSpaceProbe`` docstring.
        #expect(DiskSpaceProbe.quickstartRequiredBytes == 2 * 1024 * 1024 * 1024)
    }

    @Test("Automatic starters derive model-sized download headroom")
    func automaticStarterRequirements() {
        let gib: Int64 = 1024 * 1024 * 1024

        #expect(QuickstartView.requiredDiskBytes(
            for: QuickstartCoordinator.compactDefaultChoice
        ) == 4 * gib)
        #expect(QuickstartView.requiredDiskBytes(
            for: QuickstartCoordinator.defaultChoice
        ) == 6 * gib)
        #expect(QuickstartView.requiredDiskBytes(
            for: QuickstartCoordinator.lowMemoryChoice
        ) == 2 * gib)
        #expect(QuickstartCoordinator.defaultChoice.downloadBytes == 3_061_121_321)
        #expect(QuickstartCoordinator.compactDefaultChoice.downloadBytes == 1_601_103_345)
    }

    // MARK: - Decision truth table

    @Test("decide returns .ok when free bytes exceed requirement")
    func decisionOKWhenAmple() {
        let decision = DiskSpaceProbe.decide(
            freeBytes: 100 * 1024 * 1024 * 1024, // 100 GB
            requiredBytes: 5 * 1024 * 1024 * 1024
        )
        #expect(decision == .ok)
    }

    @Test("decide returns .ok when free bytes exactly equal requirement")
    func decisionOKAtBoundary() {
        // Equality is the ok side — a user with exactly the
        // required headroom should not see a warning that contradicts
        // the surfaced numbers ("X GB free, needs X GB").
        let req: Int64 = 5 * 1024 * 1024 * 1024
        #expect(DiskSpaceProbe.decide(freeBytes: req, requiredBytes: req) == .ok)
    }

    @Test("decide returns .warn when free bytes fall below requirement")
    func decisionWarnWhenLow() {
        let free: Int64 = 1 * 1024 * 1024 * 1024 // 1 GB
        let req: Int64 = 5 * 1024 * 1024 * 1024
        let decision = DiskSpaceProbe.decide(freeBytes: free, requiredBytes: req)
        #expect(decision == .warn(freeBytes: free, requiredBytes: req))
    }

    @Test("decide returns .warn when free bytes are zero")
    func decisionWarnWhenZero() {
        // Boundary opposite of ``decisionOKAtBoundary`` — zero free
        // bytes is unambiguously a warning, even if technically
        // ``0 >= 0`` would have flipped to .ok if we'd used the
        // wrong comparison.
        let req: Int64 = 5 * 1024 * 1024 * 1024
        let decision = DiskSpaceProbe.decide(freeBytes: 0, requiredBytes: req)
        #expect(decision == .warn(freeBytes: 0, requiredBytes: req))
    }

    @Test("decide returns .ok when probe failed (nil free bytes)")
    func decisionOKWhenProbeFailed() {
        // Fail-open contract — a transient FileManager error MUST
        // NOT gate the user out of starting the download. The HF
        // downloader will surface the real error if the disk truly
        // can't hold it.
        let decision = DiskSpaceProbe.decide(
            freeBytes: nil,
            requiredBytes: 5 * 1024 * 1024 * 1024
        )
        #expect(decision == .ok)
    }

    @Test("decide returns .ok when requirement is zero or negative")
    func decisionOKWhenRequirementDegenerate() {
        // Defensive: a future caller shouldn't pass <= 0 but the
        // decision math shouldn't trap if they do.
        #expect(
            DiskSpaceProbe.decide(freeBytes: 1024, requiredBytes: 0) == .ok
        )
        #expect(
            DiskSpaceProbe.decide(freeBytes: 1024, requiredBytes: -1) == .ok
        )
    }

    // MARK: - Real-fs sanity

    @Test("freeBytes(forPath:) on / returns a positive Int64 on the build host")
    func freeBytesOnRootReturnsPositive() {
        // We can't pin a magic number (the value depends on the
        // build host) but the call MUST land — a nil return here
        // would indicate the statfs wiring is broken AND the prod
        // pre-flight will silently degrade to "no warning ever".
        let bytes = DiskSpaceProbe.freeBytes(forPath: "/")
        #expect(bytes != nil)
        if let bytes {
            #expect(bytes > 0)
        }
    }

    @Test("freeBytes(forPath:) on a non-existent leaf walks up to the parent volume")
    func freeBytesFallsBackToParent() {
        // Brand-new install case: ``~/.cache/huggingface/hub`` doesn't
        // exist yet, but the volume backing ``$HOME`` does. The probe
        // should walk up the path until it lands a directory that
        // exists so the pre-flight is useful on first launch.
        let bogus = "/tmp/disk-space-probe-does-not-exist-\(UUID().uuidString)/deep/path"
        let bytes = DiskSpaceProbe.freeBytes(forPath: bogus)
        #expect(bytes != nil)
        if let bytes {
            #expect(bytes > 0)
        }
    }

    // MARK: - Codex r1 MAJOR — volume-boundary fail-open

    @Test("freeBytes(forPath:) on a missing /Volumes mount fails open (nil)")
    func freeBytesUnmountedVolumeFailsOpen() {
        // Codex r1 MAJOR: a user with ``HF_HOME=/Volumes/ExternalSSD/hf``
        // set before the external disk is mounted would, under a naive
        // walk-up, get the boot volume's free space (which is exactly
        // the wrong number — the redirect exists precisely BECAUSE the
        // boot SSD is too small). Pre-fix this surfaced a misleading
        // warning; post-fix the probe returns nil → decide() → .ok
        // → no warning, and HF will surface the real "disk missing"
        // error if the volume truly isn't mounted at download time.
        let bogus = "/Volumes/RapidMlxNeverExistsExternalSSD-\(UUID().uuidString)/hf/hub"
        let bytes = DiskSpaceProbe.freeBytes(forPath: bogus)
        #expect(bytes == nil)
    }

    @Test("freeBytes(forPath:) on a dangling ~/.cache → /Volumes/<missing> symlink fails open")
    func freeBytesDanglingSymlinkFailsOpen() throws {
        // Codex r2 MAJOR: the original r1 boundary check only fired
        // when the original path string started with ``/Volumes/``. A
        // user with ``~/.cache → /Volumes/ExternalSSD/cache`` whose
        // external disk is unmounted has the original path under
        // ``$HOME`` — pre-fix, ``crossesVolumeBoundary`` returned
        // false, the walk continued up to ``$HOME``, and the boot
        // volume's free space surfaced as the user's "available"
        // bytes. Post-fix, ``resolveDanglingSymlinks`` walks the
        // dangling link to its declared destination before the
        // boundary check fires.
        let base = "/tmp/disk-space-probe-symlink-\(UUID().uuidString)"
        try FileManager.default.createDirectory(
            atPath: base, withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(atPath: base) }
        let link = base + "/.cache"
        let target = "/Volumes/RapidMlxNeverExists-\(UUID().uuidString)/cache"
        try FileManager.default.createSymbolicLink(
            atPath: link, withDestinationPath: target
        )
        // Probe the path that "would" land under ``$HOME/.cache/...``
        // — symlink resolution must rewrite it to the dangling
        // /Volumes destination, then the boundary check must fire.
        let probePath = link + "/huggingface/hub"
        let bytes = DiskSpaceProbe.freeBytes(forPath: probePath)
        #expect(bytes == nil)
    }

    @Test("freeBytes(forPath:) on chained ~/.cache → /tmp/link → /Volumes/<missing> symlinks fails open")
    func freeBytesChainedSymlinkFailsOpen() throws {
        // Codex r3 MINOR: a single-pass resolver caught the direct
        // ``~/.cache → /Volumes/Missing`` shape but missed chained
        // symlinks — ``~/.cache → some/link/cache`` and
        // ``some/link → /Volumes/Missing`` left ``current`` under the
        // ``$HOME`` prefix, walked back up the home volume, and
        // surfaced the wrong free-space number. Post-fix the walker
        // re-passes until the path stops changing, so multi-hop links
        // are fully unwrapped.
        let base = "/tmp/disk-space-probe-chain-\(UUID().uuidString)"
        try FileManager.default.createDirectory(
            atPath: base, withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(atPath: base) }
        // Build: base/intermediate -> /Volumes/RapidMlxChainMissing
        //        base/.cache       -> intermediate/cache
        let intermediateLink = base + "/intermediate"
        try FileManager.default.createSymbolicLink(
            atPath: intermediateLink,
            withDestinationPath: "/Volumes/RapidMlxChainMissing-\(UUID().uuidString)"
        )
        let cacheLink = base + "/.cache"
        try FileManager.default.createSymbolicLink(
            atPath: cacheLink,
            withDestinationPath: "intermediate/cache"
        )
        let probePath = cacheLink + "/huggingface/hub"
        let bytes = DiskSpaceProbe.freeBytes(forPath: probePath)
        #expect(bytes == nil, "Chained symlink to missing /Volumes mount must fail open")
    }

    @Test("resolveDanglingSymlinks rewrites a dangling intermediate link")
    func resolveDanglingSymlinksTruthTable() throws {
        let base = "/tmp/disk-space-probe-resolve-\(UUID().uuidString)"
        try FileManager.default.createDirectory(
            atPath: base, withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(atPath: base) }
        let link = base + "/.cache"
        let target = "/Volumes/RapidMlxNoSuchDisk/cache"
        try FileManager.default.createSymbolicLink(
            atPath: link, withDestinationPath: target
        )
        let resolved = DiskSpaceProbe.resolveDanglingSymlinks(
            path: link + "/huggingface/hub"
        )
        #expect(resolved == "/Volumes/RapidMlxNoSuchDisk/cache/huggingface/hub")
        // Non-symlinked tail components pass through, but the walker
        // resolves any symlinks along the way — on macOS ``/tmp`` is
        // itself a symlink to ``/private/tmp``, so the suffix is what
        // we assert; the prefix can be either form.
        let plainInput = base + "/plain/path"
        let plain = DiskSpaceProbe.resolveDanglingSymlinks(path: plainInput)
        #expect(plain.hasSuffix("/plain/path"))
    }

    @Test("crossesVolumeBoundary truth table for /Volumes paths")
    func crossesVolumeBoundaryTruthTable() {
        // Default $HOME shape — no boundary concern, walk-up safe.
        #expect(DiskSpaceProbe.crossesVolumeBoundary(
            originalPath: "/Users/jane/.cache/huggingface/hub",
            walkedTo: "/Users/jane"
        ) == false)
        // Walk stays inside the mount → safe.
        #expect(DiskSpaceProbe.crossesVolumeBoundary(
            originalPath: "/Volumes/ModelSSD/hf/hub",
            walkedTo: "/Volumes/ModelSSD/hf"
        ) == false)
        #expect(DiskSpaceProbe.crossesVolumeBoundary(
            originalPath: "/Volumes/ModelSSD/hf/hub",
            walkedTo: "/Volumes/ModelSSD"
        ) == false)
        // Walked above the mount → crossed.
        #expect(DiskSpaceProbe.crossesVolumeBoundary(
            originalPath: "/Volumes/ModelSSD/hf/hub",
            walkedTo: "/Volumes"
        ) == true)
        #expect(DiskSpaceProbe.crossesVolumeBoundary(
            originalPath: "/Volumes/ModelSSD/hf/hub",
            walkedTo: "/"
        ) == true)
    }

    @Test("freeBytesForHFCache returns nil when env has no HOME / HF_HOME")
    func freeBytesForHFCacheReturnsNilOnEmptyEnv() {
        // Pathological environment — ``BundledModel.userHFCacheURL``
        // returns nil, so the probe propagates nil and the
        // decision degrades to ``.ok`` (no warning). Verifies the
        // wiring through the convenience helper.
        let bytes = DiskSpaceProbe.freeBytesForHFCache(environment: [:])
        #expect(bytes == nil)
    }

    @Test("freeBytesForHFCache returns a positive Int64 with a real HOME")
    func freeBytesForHFCacheHappyPath() {
        let env = ["HOME": NSHomeDirectory()]
        let bytes = DiskSpaceProbe.freeBytesForHFCache(environment: env)
        #expect(bytes != nil)
        if let bytes {
            #expect(bytes > 0)
        }
    }

    // MARK: - Banner copy helpers

    @Test("formatBytesForBanner pins MB / GB boundary")
    func formatBytesForBannerBoundary() {
        // Issue #357: < 1 GB must render as "N MB" (no decimals), not
        // "0.X GB". Sticker-pin the exact displayed strings across the
        // boundary so a future regression breaks loudly.
        let mb: Int64 = 1024 * 1024
        let gb: Int64 = 1024 * 1024 * 1024

        // Sub-GB → MB branch
        #expect(QuickstartView.formatBytesForBanner(0) == "0 MB")
        #expect(QuickstartView.formatBytesForBanner(50 * mb) == "50 MB")
        #expect(QuickstartView.formatBytesForBanner(999 * mb) == "999 MB")
        // The exact pathology from the issue: 99 MB used to render
        // as "0.1 GB" — now reads truthfully.
        #expect(QuickstartView.formatBytesForBanner(99 * mb) == "99 MB")
        // Codex r1 MINOR: one byte below the GB cutoff must NOT
        // round up to "1024 MB" (which would cross the cutoff
        // visually and reintroduce the rounding-up problem the GB
        // branch already avoids). Floor-on-integer-division pins this.
        #expect(QuickstartView.formatBytesForBanner(gb - 1) == "1023 MB")

        // ≥ 1 GB → GB branch, one decimal
        #expect(QuickstartView.formatBytesForBanner(gb) == "1.0 GB")
        let oneAndAHalf = Int64(1.5 * Double(gb))
        #expect(QuickstartView.formatBytesForBanner(oneAndAHalf) == "1.5 GB")
        #expect(QuickstartView.formatBytesForBanner(100 * gb) == "100.0 GB")
    }

    @Test("formatBytesForBanner clamps negative inputs to 0")
    func formatBytesForBannerClampsNegative() {
        // Defensive: a future caller shouldn't pass < 0 but the
        // display should never read "-2 MB" or "-2.0 GB".
        #expect(QuickstartView.formatBytesForBanner(-1) == "0 MB")
    }

    @Test("lowDiskBannerBody mentions both numbers and the model name")
    func bannerBodyHasNumbersAndModel() {
        // Codex r3 MINOR (PR #353): source the displayed-bytes from
        // ``DiskSpaceProbe.quickstartRequiredBytes`` rather than a
        // hardcoded literal so a future threshold change re-derives
        // the test surface and a stale displayed-threshold
        // regression fails loudly.
        let free: Int64 = 1 * 1024 * 1024 * 1024
        let req = DiskSpaceProbe.quickstartRequiredBytes
        let body = QuickstartView.lowDiskBannerBody(
            freeBytes: free,
            requiredBytes: req,
            displayName: QuickstartCoordinator.defaultChoice.displayName
        )
        #expect(body.contains(QuickstartView.formatBytesForBanner(free)))
        #expect(body.contains(QuickstartView.formatBytesForBanner(req)))
        #expect(body.contains(QuickstartCoordinator.defaultChoice.displayName))
        // The "Continue anyway?" prompt sets up the button label.
        #expect(body.contains("Continue anyway?"))
    }

    @Test("lowDiskBannerBody renders MB when free space is sub-GB")
    func bannerBodyRendersMBUnderOneGB() {
        // Issue #357 end-to-end: the banner body itself (not just the
        // formatter) must surface MB for a sub-GB free count so a
        // regression at either layer fails this test.
        let free: Int64 = 99 * 1024 * 1024  // 99 MB — the issue's example
        let req = DiskSpaceProbe.quickstartRequiredBytes
        let body = QuickstartView.lowDiskBannerBody(
            freeBytes: free,
            requiredBytes: req,
            displayName: QuickstartCoordinator.defaultChoice.displayName
        )
        #expect(body.contains("99 MB free"))
        // Negative-pin: must NOT render the buggy "0.1 GB free" string.
        #expect(!body.contains("0.1 GB free"))
    }

    @Test("lowDiskAccessibilityLabel narrates both buttons")
    func a11yLabelMentionsBothButtons() {
        // Same codex r3 MINOR (PR #353) — pin against the live threshold.
        let free: Int64 = 1 * 1024 * 1024 * 1024
        let req = DiskSpaceProbe.quickstartRequiredBytes
        let label = QuickstartView.lowDiskAccessibilityLabel(
            freeBytes: free,
            requiredBytes: req,
            displayName: QuickstartCoordinator.defaultChoice.displayName
        )
        #expect(label.lowercased().contains("continue anyway"))
        #expect(label.lowercased().contains("cancel"))
        #expect(label.contains(QuickstartView.formatBytesForBanner(free)))
        #expect(label.contains(QuickstartView.formatBytesForBanner(req)))
    }

    // MARK: - Coordinator state machine

    @Test("enterLowDiskWarning + cancel returns to .idle")
    func coordinatorLowDiskCancelReturnsIdle() {
        let coord = QuickstartCoordinator()
        coord._testingReset()
        let free: Int64 = 1 * 1024 * 1024 * 1024
        let req: Int64 = 5 * 1024 * 1024 * 1024
        coord.enterLowDiskWarning(freeBytes: free, requiredBytes: req)
        let expected: QuickstartCoordinator.Phase =
            .lowDiskWarning(freeBytes: free, requiredBytes: req)
        #expect(coord.phase == expected)
        coord.cancelLowDiskWarning()
        #expect(coord.phase == .idle)
        // Cancel must NOT flip the persistent done flag — a user
        // who cancels the warning should see Quickstart again on
        // the next launch.
        #expect(coord.done == false)
    }

    @Test("enterLowDiskWarning then enterDownloading proceeds normally")
    func coordinatorLowDiskContinueProceeds() {
        let coord = QuickstartCoordinator()
        coord._testingReset()
        let free: Int64 = 1 * 1024 * 1024 * 1024
        let req: Int64 = 5 * 1024 * 1024 * 1024
        coord.enterLowDiskWarning(freeBytes: free, requiredBytes: req)
        coord.enterDownloading()
        #expect(coord.phase == .downloading)
    }

    // MARK: - Codex r1 MINOR — Continue must bypass the probe

    @Test("applyPreflightDecision: .ok fires kickoff, never enters warning phase")
    func applyDecisionOKFiresKickoff() {
        let coord = QuickstartCoordinator()
        coord._testingReset()
        var kickoffCount = 0
        QuickstartView.applyPreflightDecision(
            decision: .ok,
            coordinator: coord,
            onKickoff: { kickoffCount += 1 }
        )
        #expect(kickoffCount == 1)
        #expect(coord.phase == .idle, "ok decision must not flip into .lowDiskWarning")
    }

    @Test("applyPreflightDecision: .warn flips into warning phase, never fires kickoff")
    func applyDecisionWarnDefersKickoff() {
        let coord = QuickstartCoordinator()
        coord._testingReset()
        let free: Int64 = 1 * 1024 * 1024 * 1024
        let req: Int64 = 5 * 1024 * 1024 * 1024
        var kickoffCount = 0
        QuickstartView.applyPreflightDecision(
            decision: .warn(freeBytes: free, requiredBytes: req),
            coordinator: coord,
            onKickoff: { kickoffCount += 1 }
        )
        #expect(kickoffCount == 0, "warn decision must NOT auto-fire download; user owns the call")
        let expected: QuickstartCoordinator.Phase =
            .lowDiskWarning(freeBytes: free, requiredBytes: req)
        #expect(coord.phase == expected)
    }

    @Test("Continue button is wired to kickoffDownload, NOT startQuickstart (source-grep regression)")
    func continueButtonSourceGrep() throws {
        // Codex r1+r2 MINOR: if the Continue button on the low-disk
        // banner re-runs the disk probe (via startQuickstart) and the
        // disk has filled further in the few seconds the banner was
        // visible, the user gets trapped in a warning → continue →
        // warning loop. The runtime fix is that the Continue button
        // calls ``kickoffDownload`` directly.
        //
        // Closure introspection isn't viable in Swift, so the most
        // honest defence against a future regression is a source-grep
        // contract test. Read the production file and assert that the
        // ``lowDiskCard`` body wires its primary button straight to
        // ``kickoffDownload()`` and never to ``startQuickstart()``.
        let candidates = [
            "Sources/Rapid/UI/QuickstartView.swift",
            "../Sources/Rapid/UI/QuickstartView.swift",
            "../../Sources/Rapid/UI/QuickstartView.swift"
        ]
        var contents: String? = nil
        for relative in candidates {
            if let body = try? String(contentsOfFile: relative, encoding: .utf8) {
                contents = body
                break
            }
        }
        // CI may run from a different cwd; fall back to grepping the
        // SwiftPM package root if available.
        if contents == nil,
           let pkgRoot = ProcessInfo.processInfo.environment["SWIFTPM_PACKAGE_ROOT"] {
            let path = pkgRoot + "/Sources/Rapid/UI/QuickstartView.swift"
            contents = try? String(contentsOfFile: path, encoding: .utf8)
        }
        guard let body = contents else {
            // Source unavailable in this CI shape — degrade to a soft
            // skip rather than a false-fail. The behavioural contract
            // is still covered by ``applyDecisionOKFiresKickoff`` +
            // ``applyDecisionWarnDefersKickoff``.
            return
        }
        // Carve out the lowDiskCard body for the assertion. Spans
        // from "private func lowDiskCard" to "private func failedCard"
        // (the next sibling in source order, see the view file
        // structure).
        guard let startRange = body.range(of: "private func lowDiskCard"),
              let endRange = body.range(of: "private func failedCard")
        else {
            Issue.record("Source grep couldn't locate lowDiskCard / failedCard boundary in QuickstartView.swift")
            return
        }
        let card = String(body[startRange.lowerBound..<endRange.lowerBound])
        #expect(card.contains("kickoffDownload()"),
                "lowDiskCard's Continue button must call kickoffDownload() — bypass keeps the user out of the warning loop a re-probe could trap them in.")
        #expect(!card.contains("startQuickstart()"),
                "lowDiskCard MUST NOT call startQuickstart() — that re-runs the probe and risks a warning loop if free space dropped further while the banner was visible.")
    }
}
