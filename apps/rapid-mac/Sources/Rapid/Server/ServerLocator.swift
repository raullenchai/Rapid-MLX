import Foundation

/// Locates the `rapid-mlx` CLI binary that this desktop release owns.
///
/// v0.8.10 cutover: the Phase 1 legacy escape hatch (PATH / brew /
/// pipx / uv) was removed. The slim-bootstrapper DMG (v0.8.9 ε.2) is
/// the canonical install path; every supported install writes the
/// sidecar to ``~/Library/Application Support/Rapid/runtime-override/``
/// via ``BootstrapCoordinator``, or ships it inside the bundle for
/// rare full-bundle builds. A pre-existing user-installed
/// ``rapid-mlx`` on ``PATH`` is intentionally NOT picked up — the
/// desktop and CLI versions would drift silently and "Rapid · up to
/// date" would lie about whichever copy actually answered the spawn.
/// Power users running a dev checkout of rapid-mlx set ``RAPID_BIN``
/// to point at it explicitly; that's still honoured.
///
/// Threat model: `find()` trusts on-disk content. A malicious app with
/// user-level write access can plant a binary at
/// `~/Library/Application Support/Rapid/runtime-override/rapid-mlx/bin/rapid-mlx`
/// and Rapid will execute it on next launch. Mach-O signature
/// verification at the in-app updater install step is a follow-up
/// (same surface ``BootstrapCoordinator`` accepts today).
///
/// Priority chain (high → low):
///
/// 1. **`RAPID_BIN`** — test / dev override env var. Highest priority so
///    integration tests can swap in a fake CLI without touching disk
///    and dev-checkout power users can point at their local build.
/// 2. **managed sidecars, newest version wins** — compares the
///    runtime-override and bundled `VERSION` files whenever both binaries
///    exist. A stale or unversioned runtime override cannot shadow a newer,
///    versioned sidecar shipped by an app update. The runtime override still
///    wins when it is the same version or newer, and remains the only managed
///    candidate for slim DMGs whose bundled slot is empty.
///
///    **runtime-override** — `~/Library/Application Support/Rapid/runtime-override/rapid-mlx/bin/rapid-mlx`
///    written by ``BootstrapCoordinator`` on first launch of the slim
///    bootstrapper DMG, or by the in-app updater when it pulls a
///    newer sidecar from `latest.json`. The `rapid-mlx/` wrapper
///    directory mirrors the top-level entry of the sidecar tarball
///    (``scripts/build-sidecar-tarball.sh`` arcname ``rapid-mlx/...``)
///    AND the bundled layout below — both slots reach the binary via
///    the same final ``rapid-mlx/bin/rapid-mlx`` suffix, so adding a
///    fourth lookup slot in the future stays symmetric. Survives
///    desktop upgrades because it lives outside `Contents/`.
///    **bundled** — `Contents/Resources/rapid-mlx/bin/rapid-mlx`
///    shipped inside a full-bundle DMG (today the slim DMG ships
///    empty here and relies entirely on the bootstrapper to
///    populate the runtime-override slot).
///
/// Return value: ``nil`` when none of the three slots resolves —
/// callers are expected to surface the missing-install UX (re-run
/// the bootstrapper) rather than silently fall back to a sibling
/// install of a different version.
enum ServerLocator {
    struct Resolution: Equatable, Sendable {
        let binary: URL
        let source: ResolvedSource
        let version: String?
    }

    /// Resolve the executable together with the provenance decided by the
    /// same priority/version comparison. Callers that surface diagnostics
    /// must use this instead of trying to reconstruct a winning slot from a
    /// symlink-resolved path (#1712).
    static func locate() -> Resolution? {
        locate(environment: ProcessInfo.processInfo.environment)
    }

    static func locate(environment: [String: String]) -> Resolution? {
        locate(
            environment: environment,
            bundleResourceURL: Self.defaultBundleResourceURL,
            applicationSupportURL: Self.defaultApplicationSupportURL(environment: environment)
        )
    }

    /// Returns the absolute path to `rapid-mlx`, or `nil` if no candidate
    /// resolves to a regular executable file.
    static func find() -> URL? {
        locate()?.binary
    }

    static func find(environment: [String: String]) -> URL? {
        locate(environment: environment)?.binary
    }

    /// Test seam: `bundleResourceURL` and `applicationSupportURL` are
    /// injected so tests can stand up fixtures without touching the
    /// real `Bundle.main` or `~/Library/Application Support`.
    static func find(
        environment: [String: String],
        bundleResourceURL: URL?,
        applicationSupportURL: URL?
    ) -> URL? {
        locate(
            environment: environment,
            bundleResourceURL: bundleResourceURL,
            applicationSupportURL: applicationSupportURL
        )?.binary
    }

    static func locate(
        environment: [String: String],
        bundleResourceURL: URL?,
        applicationSupportURL: URL?
    ) -> Resolution? {
        // 1. ``RAPID_BIN`` — test/dev override. TestDriver chat smoke
        // uses this to swap in ``scripts/fake-rapid-mlx.sh`` so the
        // lifecycle test runs in <2 s. Power users running their own
        // dev checkout of rapid-mlx point this at it.
        if let override = environment["RAPID_BIN"],
           !override.isEmpty,
           let resolved = executableURL(path: override, allowRelative: true) {
            return Resolution(
                binary: resolved,
                source: .rapidBin,
                version: sidecarVersion(forBinary: resolved)
            )
        }

        // Managed sidecars. Resolve both executable slots before choosing:
        // when both exist, VERSION decides which release the desktop owns.
        // This prevents an override left by an older app from shadowing the
        // newer engine that arrived inside a desktop update (#1503).
        //
        // 2a. runtime-override — written by ``BootstrapCoordinator``
        // on first launch of the slim DMG, or by the in-app updater
        // when it pulls a newer rapid-mlx than the bundled one. The
        // ``rapid-mlx/`` wrapper directory matches the top-level entry
        // of the sidecar tarball (preserved through extract + publish)
        // and is symmetric with the bundled slot below — both end in
        // ``rapid-mlx/bin/rapid-mlx``. P0 fix for #430: the path was
        // previously ``runtime-override/bin/rapid-mlx`` (missing the
        // wrapper), so every fresh slim-DMG install completed the
        // bootstrap but immediately landed on the missing-overlay
        // dialog with no recovery. Latent since PR #36 — only
        // exposed v0.8.12 when slim DMG first went live on
        // ``latest.json``; v0.8.10/v0.8.11 silently fell back to the
        // canonical full DMG which hit the bundled slot instead.
        let runtimeOverride = applicationSupportURL?
            .appendingPathComponent("runtime-override/rapid-mlx/bin/rapid-mlx")
        let resolvedRuntimeOverride = runtimeOverride.flatMap {
            executableURL(path: $0.path, allowRelative: false)
        }

        // 2b. bundled — shipped inside the DMG at
        // ``Contents/Resources/rapid-mlx/bin/rapid-mlx``. The slim
        // bootstrapper DMG (v0.8.9 ε.2) ships this slot EMPTY and
        // relies on slot 2 (populated by ``BootstrapCoordinator`` on
        // first launch) to satisfy the lookup. Full-bundle DMGs
        // (developer builds, future ad-hoc artefacts) keep this slot
        // populated as a zero-network fallback. RAPID_BIN remains an
        // unconditional explicit choice; app-managed overrides are compared
        // with this slot so an old bootstrap artifact cannot pin the desktop
        // to an older engine forever.
        let bundled = bundleResourceURL?
            .appendingPathComponent("rapid-mlx/bin/rapid-mlx")
        let resolvedBundled = bundled.flatMap {
            executableURL(path: $0.path, allowRelative: false)
        }

        // There is intentionally no PATH / Homebrew / pipx / uv fallback:
        // if both managed slots miss, callers re-run the supported bootstrap.
        // Read metadata from the managed slots rather than the resolved
        // executable targets. This also preserves the correct VERSION when a
        // launcher is a symlink into a checkout.
        let runtimeVersion = runtimeOverride.flatMap { sidecarVersion(forBinary: $0) }
        let bundledVersion = bundled.flatMap { sidecarVersion(forBinary: $0) }

        switch (resolvedRuntimeOverride, resolvedBundled) {
        case let (runtime?, resolvedBundle?):
            if shouldPreferBundled(
                runtimeOverrideVersion: runtimeVersion,
                bundledVersion: bundledVersion
            ) {
                return Resolution(binary: resolvedBundle, source: .bundled, version: bundledVersion)
            }
            return Resolution(binary: runtime, source: .runtimeOverride, version: runtimeVersion)
        case let (runtime?, nil):
            // Slim DMG: bootstrap owns the only populated sidecar slot.
            return Resolution(
                binary: runtime,
                source: .runtimeOverride,
                version: runtimeVersion
            )
        case let (nil, bundled?):
            return Resolution(
                binary: bundled,
                source: .bundled,
                version: bundledVersion
            )
        case (nil, nil):
            return nil
        }
    }

    /// Decide between two executable, app-managed sidecars.
    ///
    /// A valid bundled version is the trust anchor because it shipped with the
    /// app. A runtime override may shadow it only when its own VERSION parses
    /// and is equal or newer. If the bundle has no valid VERSION we preserve
    /// the historical override-first behaviour: developer/ad-hoc bundles may
    /// omit version metadata, and an unverifiable bundle must not displace a
    /// working bootstrap install.
    static func shouldPreferBundled(
        runtimeOverrideVersion: String?,
        bundledVersion: String?
    ) -> Bool {
        guard let bundled = parsedVersion(bundledVersion) else { return false }
        guard let runtime = parsedVersion(runtimeOverrideVersion) else { return true }
        return compareVersion(runtime, bundled) == .orderedAscending
    }

    /// Read `<sidecar-root>/VERSION` for a `.../bin/rapid-mlx` candidate.
    private static func sidecarVersion(forBinary binary: URL) -> String? {
        let versionFile = binary
            .deletingLastPathComponent() // bin/
            .deletingLastPathComponent() // rapid-mlx/
            .appendingPathComponent("VERSION")
        guard let data = try? Data(contentsOf: versionFile),
              let raw = String(data: data, encoding: .utf8) else {
            return nil
        }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    /// Strict dotted-numeric parser matching the sidecar build gate.
    private static func parsedVersion(_ raw: String?) -> [Int]? {
        guard var value = raw?.trimmingCharacters(in: .whitespacesAndNewlines),
              !value.isEmpty else { return nil }
        if value.hasPrefix("v") || value.hasPrefix("V") {
            value.removeFirst()
        }
        let fields = value.split(separator: ".", omittingEmptySubsequences: false)
        guard fields.count >= 2 else { return nil }
        var parts: [Int] = []
        parts.reserveCapacity(fields.count)
        for field in fields {
            guard !field.isEmpty,
                  field.allSatisfy({ $0 >= "0" && $0 <= "9" }),
                  let part = Int(field) else { return nil }
            parts.append(part)
        }
        return parts
    }

    private static func compareVersion(_ lhs: [Int], _ rhs: [Int]) -> ComparisonResult {
        let width = max(lhs.count, rhs.count)
        for index in 0..<width {
            let left = index < lhs.count ? lhs[index] : 0
            let right = index < rhs.count ? rhs[index] : 0
            if left < right { return .orderedAscending }
            if left > right { return .orderedDescending }
        }
        return .orderedSame
    }

    private static func executableURL(path: String, allowRelative: Bool) -> URL? {
        guard let normalized = normalizedPath(path, allowRelative: allowRelative) else {
            return nil
        }
        var isDirectory: ObjCBool = false
        let fm = FileManager.default
        guard fm.fileExists(atPath: normalized, isDirectory: &isDirectory),
              !isDirectory.boolValue,
              fm.isExecutableFile(atPath: normalized) else {
            return nil
        }
        return URL(fileURLWithPath: normalized).resolvingSymlinksInPath()
    }

    /// Reports which slot in the priority chain a resolved binary came
    /// from. Used by the About panel to show "Bundled (0.7.3)" vs
    /// "App-managed override" vs "RAPID_BIN override" so the user can
    /// tell which CLI is actually running.
    ///
    /// v0.8.10 cutover: ``find()`` no longer walks PATH or scavenges
    /// brew / pipx / uv installs, so those slots were removed here too.
    /// Every path ``find()`` yields matches one of the three live slots
    /// (``RAPID_BIN`` — even when it points at a symlinked brew / pipx
    /// binary — resolves to ``.rapidBin``). ``.unknown`` survives only as
    /// the terminal fallback for a directly-classified path that matches
    /// none of them (a diagnostic / test input, never a real install).
    enum ResolvedSource: String, Equatable, Sendable {
        case rapidBin
        case runtimeOverride
        case bundled
        case unknown

        /// Human-readable origin label for the About panel. Kept short
        /// enough to render in a status pill or chip next to a path.
        var displayLabel: String {
            switch self {
            case .rapidBin:               return "RAPID_BIN override"
            case .runtimeOverride:        return "App-managed override"
            case .bundled:                return "Bundled with app"
            case .unknown:                return "Unknown origin"
            }
        }
    }

    /// Classifies a resolved binary path by which slot it matched. Pure
    /// function — no FS reads.
    ///
    /// Only the three live slots (``RAPID_BIN`` → runtime-override →
    /// bundled) are matched; a path that resolves to none of them is
    /// reported as ``.unknown`` (see the ``ResolvedSource`` note on the
    /// v0.8.10 PATH / brew / pipx / uv removal).
    static func classify(
        resolved: URL,
        environment: [String: String],
        bundleResourceURL: URL?,
        applicationSupportURL: URL?
    ) -> ResolvedSource {
        let path = resolved.standardizedFileURL.path
        // codex r1 MINOR: ``find()`` calls ``resolvingSymlinksInPath()``
        // before returning, so a ``RAPID_BIN`` that pointed at a
        // symlink (e.g. a Cellar-backed shim) arrives here as the
        // resolved target — comparing against the unresolved env value
        // would miss and the override would fall through to ``.unknown``
        // instead of ``.rapidBin``, obscuring the explicit user intent.
        // Resolve the env path symbolically first so the equality check
        // survives the link hop.
        if let override = environment["RAPID_BIN"], !override.isEmpty,
           let normalized = normalizedPath(override, allowRelative: true) {
            let resolvedOverride = URL(fileURLWithPath: normalized)
                .resolvingSymlinksInPath()
                .standardizedFileURL
                .path
            if normalized == path || resolvedOverride == path {
                return .rapidBin
            }
        }
        if let override = applicationSupportURL?
            .appendingPathComponent("runtime-override/rapid-mlx/bin/rapid-mlx")
            .standardizedFileURL.path,
           override == path {
            return .runtimeOverride
        }
        if let bundled = bundleResourceURL?
            .appendingPathComponent("rapid-mlx/bin/rapid-mlx")
            .standardizedFileURL.path,
           bundled == path {
            return .bundled
        }
        // No PATH / brew / pipx / uv slots to match: ``find()`` stopped
        // surfacing them in the v0.8.10 cutover, and a real ``RAPID_BIN``
        // (including a brew / pipx symlink) already matched ``.rapidBin``
        // above. ``.unknown`` is the terminal fallback for a directly-
        // classified path outside all three live slots — a diagnostic or
        // test input, never a binary ``find()`` would actually spawn.
        return .unknown
    }

    /// Convenience overload that uses production defaults for env +
    /// bundle resources + Application Support — the shape the About
    /// panel wants when it just has a resolved URL in hand.
    static func classify(resolved: URL) -> ResolvedSource {
        let env = ProcessInfo.processInfo.environment
        return classify(
            resolved: resolved,
            environment: env,
            bundleResourceURL: defaultBundleResourceURL,
            applicationSupportURL: defaultApplicationSupportURL(environment: env)
        )
    }

    /// Production default bundle resource URL — `nil` when running
    /// outside the app bundle (e.g. unit tests linking the package
    /// directly). Tests inject a real path via the explicit overload.
    private static var defaultBundleResourceURL: URL? {
        Bundle.main.resourceURL
    }

    /// `~/Library/Application Support/Rapid` (created by the in-app
    /// updater when it writes the override). We do NOT mkdir here —
    /// just compute the URL — so a missing directory simply falls
    /// through to the next candidate.
    private static func defaultApplicationSupportURL(environment: [String: String]) -> URL? {
        // Delegate to the canonical locator (#419/#420 consolidation).
        // We keep the Optional-returning shape because the existing
        // ``find()`` callers branch on absence — even though the new
        // helper has a defensive fallback, callers that want to
        // distinguish "couldn't resolve HOME" from "got a real path"
        // still get nil here when HOME is absent / non-absolute.
        guard let home = environment["HOME"], home.hasPrefix("/") else {
            return nil
        }
        _ = home
        return ApplicationSupportLocator.applicationSupportRoot(environment: environment)
    }

    private static func normalizedPath(_ path: String, allowRelative: Bool) -> String? {
        if path.hasPrefix("/") {
            return URL(fileURLWithPath: path).standardizedFileURL.path
        }
        guard allowRelative else { return nil }
        return URL(fileURLWithPath: path).standardizedFileURL.path
    }

    static func _testingNormalizedPath(_ path: String, allowRelative: Bool) -> String? {
        normalizedPath(path, allowRelative: allowRelative)
    }
}
