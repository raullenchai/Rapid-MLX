import Foundation

/// Resolves the engine shipped inside the current desktop app.
/// `RAPID_BIN` remains an explicit developer/test override. Legacy writable
/// runtime-override installations are never discovered automatically: their
/// VERSION files do not authenticate the launcher or its Python imports.
/// Current releases ship the engine inside the signed app's resource seal.
/// A missing bundle requires reinstalling the full desktop app, or setting
/// RAPID_BIN explicitly for a developer build.
enum ServerLocator {
    struct Resolution: Equatable, Sendable {
        let binary: URL
        let source: ResolvedSource
        let version: String?
    }

    /// Resolve the executable together with the provenance decided by the
    /// same resolution decision. Callers that surface diagnostics
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
        applicationSupportURL: URL?,
        authenticateBundle: (URL) -> Bool = DesktopRuntimeTrust.authenticate
    ) -> URL? {
        locate(
            environment: environment,
            bundleResourceURL: bundleResourceURL,
            applicationSupportURL: applicationSupportURL,
            authenticateBundle: authenticateBundle
        )?.binary
    }

    static func locate(
        environment: [String: String],
        bundleResourceURL: URL?,
        applicationSupportURL: URL?,
        authenticateBundle: (URL) -> Bool = DesktopRuntimeTrust.authenticate
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

        // Only the bundled engine is app-managed. A writable VERSION file
        // must never allow a legacy runtime-override to replace this payload.
        guard let bundled = bundleResourceURL?
            .appendingPathComponent("rapid-mlx/bin/rapid-mlx"),
              executableURL(path: bundled.path, allowRelative: false) != nil,
              authenticateBundle(bundled) else {
            return nil
        }
        return Resolution(
            binary: bundled.standardizedFileURL,
            source: .bundled,
            version: sidecarVersion(forBinary: bundled)
        )
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

    /// Origin labels for current resolutions and legacy diagnostic paths.
    /// `locate()` only returns rapidBin or bundled; runtimeOverride remains
    /// available to classify historical paths supplied to diagnostics.
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

    /// Classifies a supplied path for diagnostics, including legacy paths.
    /// This does not authorize execution; use locate() to choose an engine.
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
