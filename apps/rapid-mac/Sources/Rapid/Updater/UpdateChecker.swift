import Foundation
import Observation

/// Lightweight self-update check for Rapid.app.
///
/// rapid-desktop's source lives in a **private** GitHub repo, but
/// release artifacts (DMG + manifest) are published to a public R2
/// bucket fronted by `dl.rapidmlx.com`. The app used to poll that
/// static JSON object directly; as of the update-ping migration it
/// polls a thin Cloudflare Worker on the landing-page domain that
/// returns the **byte-identical manifest** but also counts the poll
/// as an aggregate active-install signal — no proxy of GitHub, no
/// PAT, no GH API rate-limit:
///
///   GET https://rapidmlx.com/api/desktop-update?v=<app-version>
///
/// If that Worker fetch fails for any reason (down, non-2xx, or a
/// manifest that fails decode/validation) the check falls back to the
/// static R2 object it proxies (`dl.rapidmlx.com/latest.json`), so a
/// Worker outage never stops updates — only the active-install count
/// is missed for that poll. See ``fetchWithFallback``.
///     200  {
///            "schema_version": 1,
///            "version": "0.6.11",          // tag with leading "v" stripped
///            "tag_name": "v0.6.11",
///            "html_url": "https://rapidmlx.com/desktop",
///            "notes": "…release body, may be empty…",
///            "published_at": "2026-06-16T16:38:34Z",
///            "dmg_url": "https://dl.rapidmlx.com/rapid-mlx-desktop-0.6.11.dmg",
///            "dmg_sha256": "57806194…",    // forward-compat, not yet read by client
///            "dmg_size": 127234855,        // forward-compat, informational
///            // Optional bootstrapper fields (slice γ + slice δ):
///            "sidecar_url":     "https://dl.rapidmlx.com/rapid-mlx-sidecar-0.6.11.tar.gz",
///            "sidecar_sha256":  "…",
///            "sidecar_size":    125969942,
///            "sidecar_version": "0.8.18",
///            "model_url":       "https://dl.rapidmlx.com/quickstart-bonsai-1.7b-2bit-0.6.11.tar.gz",
///            "model_sha256":    "…",
///            "model_size":      490000000,
///            "model_alias":     "bonsai-1.7b-2bit"
///          }
///     5xx — CF / R2 transient error
///
/// `dmg_sha256` / `dmg_size` are emitted by the desktop release workflow
/// (`.github/workflows/rapid-mac-release.yml`) but the
/// `Release` Codable struct below does NOT decode them — nothing in
/// this process downloads the DMG any more. Sparkle fetches its own
/// appcast and verifies the EdDSA signature over the payload it
/// downloads, so integrity lives there, not here.
/// Codable silently drops unknown keys, so existing manifests don't
/// need a schema bump when those fields appear/disappear.
///
/// `sidecar_*` / `model_*` are bootstrapper-only fields read by
/// ``BootstrapCoordinator`` on first-install (slice γ + slice δ).
/// The in-app UpdateChecker does NOT take action on them. Declaring
/// them here as `Codable Optional` so a future migration (e.g.
/// surfacing "sidecar fix-up download" in Settings, or driving the
/// in-app installer through the bootstrapper machinery) only has
/// to wire UI, not re-touch the Codable type. ALSO: tightens the
/// already-tolerated unknown-field behaviour into an explicit
/// schema record so a publisher-side typo (e.g. `model_alais`)
/// trips a `keyNotFound` in tests rather than silently shipping
/// nil to production. Schema_version stays at 1 — the new fields
/// are additive optional (Codable default-nil on absence), no
/// publisher coordination required to roll them out.
///
/// Decisions:
///
///   * Per-launch only, no timer. Sparkle owns the recurring
///     "is there a newer version" schedule; a second timer beside it was
///     pure duplication. Mac users restart apps weekly+, so per-launch was
///     the dominant cadence anyway.
///   * No background fetch via NSURLSession schedulers.
///   * No auth header — the endpoint is public.
///   * **Privacy contract.** The ONLY thing this request sends is
///     the app's own version, as the `?v=` query parameter (plus the
///     standard `User-Agent`). No device fingerprint, no unique ID,
///     no usage data, nothing about the machine. The Worker records
///     only aggregate counts — version and (Cloudflare-derived)
///     country — and never stores the client IP. The check is a
///     plain fail-open GET: any network or decode error silently
///     leaves the app on its current version (no alert, no crash),
///     and it is skipped entirely when the user has opted out (see
///     ``updateChecksEnabled``).
///   * Installation is Sparkle's job entirely. This type never downloads,
///     verifies or swaps a bundle; it reports what the manifest says so the
///     version pill and the Settings panel have something to render.
@MainActor
@Observable
final class UpdateChecker {
    /// Stripped release payload returned by the manifest. Fields are
    /// the union of what the UI surfaces (version, URL, notes) plus
    /// the raw ``tag_name`` for diagnostics.
    ///
    /// The four ``sidecar*`` and four ``model*`` fields are bootstrapper-
    /// only (slice γ + slice δ). v0.8.x users take the
    /// ``.installed(.bundled)`` short-circuit and never traverse the
    /// install pipeline that consumes them; they live on the type
    /// purely as ``Optional`` so a future migration (e.g. driving the
    /// in-app installer through the bootstrapper machinery) can read
    /// them without re-touching the Codable shape. They are absent on
    /// pre-slice-δ ``latest.json`` payloads and Codable defaults them
    /// to nil — schema_version stays at 1.
    struct Release: Equatable, Sendable, Codable {
        let schemaVersion: Int
        let version: String
        let tagName: String
        let htmlURL: String
        let notes: String
        let publishedAt: String
        let dmgURL: String?
        // P3 slice γ bootstrapper fields. ``BootstrapCoordinator`` has
        // its own decoder (``BootstrapManifest``) that does all the
        // validation; declaring them here too is purely so the
        // UpdateChecker decoder doesn't drop them silently on the
        // floor when a future surface (Settings → "Reinstall sidecar"
        // button, "Apply pending update" CTA) needs to read them.
        let sidecarURL: String?
        let sidecarSHA256: String?
        let sidecarSize: UInt64?
        let sidecarVersion: String?
        // P3 slice δ Quickstart-model fields (this PR). All four
        // MUST be all-present or all-absent on the wire —
        // ``BootstrapCoordinator.validateModelFields`` enforces the
        // contract for the install pipeline. UpdateChecker does not
        // surface them today; they live here purely for forward-
        // compat decoding so a future client surface can consume them without
        // re-touching the type.
        let modelURL: String?
        let modelSHA256: String?
        let modelSize: UInt64?
        let modelAlias: String?

        enum CodingKeys: String, CodingKey {
            case schemaVersion = "schema_version"
            case version
            case tagName = "tag_name"
            case htmlURL = "html_url"
            case notes
            case publishedAt = "published_at"
            case dmgURL = "dmg_url"
            case sidecarURL = "sidecar_url"
            case sidecarSHA256 = "sidecar_sha256"
            case sidecarSize = "sidecar_size"
            case sidecarVersion = "sidecar_version"
            case modelURL = "model_url"
            case modelSHA256 = "model_sha256"
            case modelSize = "model_size"
            case modelAlias = "model_alias"
        }

        /// Explicit initializer so existing call sites (TestDriver,
        /// fixtures, Settings panel snapshot helpers) that pass the
        /// pre-slice-δ field set continue to compile without having
        /// to spell out the eight new optional fields. Memberwise
        /// inits don't grant defaults to ``let`` fields, so without
        /// this we'd break every caller in the codebase the moment
        /// the type gains a property. The bootstrapper / quickstart
        /// fields default to nil so a TestDriver fixture that doesn't
        /// care about them reads the same way it always did.
        init(
            schemaVersion: Int,
            version: String,
            tagName: String,
            htmlURL: String,
            notes: String,
            publishedAt: String,
            dmgURL: String?,
            sidecarURL: String? = nil,
            sidecarSHA256: String? = nil,
            sidecarSize: UInt64? = nil,
            sidecarVersion: String? = nil,
            modelURL: String? = nil,
            modelSHA256: String? = nil,
            modelSize: UInt64? = nil,
            modelAlias: String? = nil
        ) {
            self.schemaVersion = schemaVersion
            self.version = version
            self.tagName = tagName
            self.htmlURL = htmlURL
            self.notes = notes
            self.publishedAt = publishedAt
            self.dmgURL = dmgURL
            self.sidecarURL = sidecarURL
            self.sidecarSHA256 = sidecarSHA256
            self.sidecarSize = sidecarSize
            self.sidecarVersion = sidecarVersion
            self.modelURL = modelURL
            self.modelSHA256 = modelSHA256
            self.modelSize = modelSize
            self.modelAlias = modelAlias
        }
    }

    /// Externally-visible state, bound directly by the MenuBarExtra
    /// menu and any "About" sheet. ``checking`` is a separate flag
    /// from ``lastResult`` so a manual "Check for updates…" click can
    /// show a spinner without clearing the previous result.
    private(set) var checking: Bool = false
    /// The freshest release the manifest reported, regardless of
    /// whether it's newer than ours. ``availableUpdate`` does the
    /// "is this strictly newer than us" comparison.
    private(set) var latest: Release?
    /// Last error from a check, or nil. Cleared on a successful poll.
    private(set) var lastError: String?
    /// When the last check finished (success OR failure). Surfaced for
    /// diagnostics; nothing schedules off it now that the check is
    /// per-launch.
    private(set) var lastCheckedAt: Date?

    /// Currently-installed version, read once from the bundle. Kept
    /// as ``String`` because the comparison helper handles parsing.
    let currentVersion: String

    /// Override hook for tests / TestDriver: lets the harness inject
    /// a closure that returns a canned ``Release`` payload (or throws)
    /// without standing up a real HTTPS server. The production path
    /// uses ``defaultFetch`` which GETs the R2 manifest.
    typealias Fetcher = @Sendable () async throws -> Release
    private let fetcher: Fetcher

    /// Gate deciding whether a check may run at all. Injected so tests
    /// can force opt-out WITHOUT mutating process-global
    /// ``UserDefaults.standard`` — that mutation races any other
    /// parallel test whose ``check()`` reads the same key. Production
    /// leaves this nil and falls back to the env + UserDefaults reader
    /// ``updateChecksEnabled()``.
    ///
    /// Declared AFTER ``fetcher`` in ``init`` so the existing
    /// trailing-closure call sites (`UpdateChecker(currentVersion:) { … }`)
    /// keep binding their closure to ``fetcher`` under the SE-0286
    /// forward-scan rule (which matches a single trailing closure to the
    /// first function-typed parameter). Pass this one by label.
    private let checksEnabled: @Sendable () -> Bool

    init(
        currentVersion: String = UpdateChecker.bundleVersion(),
        fetcher: Fetcher? = nil,
        checksEnabled: (@Sendable () -> Bool)? = nil
    ) {
        self.currentVersion = currentVersion
        self.fetcher = fetcher ?? UpdateChecker.defaultFetch
        self.checksEnabled = checksEnabled ?? { UpdateChecker.updateChecksEnabled() }
    }

    /// Does the latest known release look strictly newer than the
    /// installed app? Returns ``nil`` until ``check()`` has run at
    /// least once or if the comparison can't be made.
    var availableUpdate: Release? {
        guard let latest = latest else { return nil }
        return Self.isNewer(latest.version, than: currentVersion) ? latest : nil
    }

    /// Run a check now. Safe to call any number of times — if one is
    /// already in flight we no-op rather than queue a second.
    @discardableResult
    func check() async -> Release? {
        if checking { return latest }
        // Off switch. When the user has opted out (env var or the
        // UserDefaults toggle, via the injected ``checksEnabled`` gate)
        // we skip the check entirely — NO network call is made, so the
        // endpoint records no active-install signal for this launch. We
        // deliberately do not touch ``lastError`` / ``lastCheckedAt``
        // here: opting out is not a failure, and the UI reads
        // ``availableUpdate`` which stays at whatever the last (if any)
        // successful poll reported.
        guard checksEnabled() else { return latest }
        checking = true
        defer {
            checking = false
            lastCheckedAt = Date()
        }
        do {
            let release = try await fetcher()
            latest = release
            lastError = nil
            return release
        } catch is CancellationError {
            // Scene teardown / SwiftUI .task cancellation should not
            // surface a "failed to check for updates" banner — the
            // user didn't ask, the network round-trip was just
            // racing the window close. Preserve previous state.
            // [codex audit r1 UpdateChecker.swift:124]
            return latest
        } catch let urlError as URLError where urlError.code == .cancelled {
            // Same idea, URLSession's cancellation flavor.
            return latest
        } catch {
            // Failures here are mostly "offline" or "R2 / CF edge
            // briefly 5xx-ing"; we keep the previous ``latest`` so
            // the user doesn't see a transient blip wipe out the
            // update badge.
            lastError = error.localizedDescription
            return latest
        }
    }

    /// Production fetcher — Worker-primary with an R2 fallback.
    ///
    /// Tries the versioned Worker endpoint first (so the poll is
    /// counted as an active-install signal). If the Worker is down,
    /// non-2xx's, or serves a manifest that fails decode/validation, it
    /// falls back to the static R2 object the Worker proxies — so a
    /// Worker outage can never stop clients from seeing updates while
    /// `dl.rapidmlx.com/latest.json` is healthy; the aggregate count is
    /// simply not incremented for that poll. Scene-teardown cancellation
    /// is rethrown (not retried) so a racing window-close preserves
    /// state instead of doubling the request.
    private static let defaultFetch: Fetcher = {
        try await fetchWithFallback(
            primary: endpointURL(forVersion: bundleVersion()),
            fallback: URL(string: fallbackEndpoint),
            using: { try await fetchManifest(from: $0) }
        )
    }

    /// Try ``primary``; on any NON-cancellation failure, retry
    /// ``fallback`` with the same fetch-and-validate primitive. Pure of
    /// ``URLSession`` (the primitive is injected) so the fallback policy
    /// is unit-testable. Cancellation is rethrown so ``check()`` can
    /// preserve state on scene teardown. Throws ``invalidURL`` when the
    /// primary URL couldn't be built, and rethrows the primary's error
    /// when there is no fallback URL.
    nonisolated static func fetchWithFallback(
        primary: URL?,
        fallback: URL?,
        using fetch: @Sendable (URL) async throws -> Release
    ) async throws -> Release {
        guard let primary else { throw UpdateError.invalidURL }
        do {
            return try await fetch(primary)
        } catch {
            if error is CancellationError { throw error }
            if let urlError = error as? URLError, urlError.code == .cancelled { throw error }
            guard let fallback else { throw error }
            return try await fetch(fallback)
        }
    }

    /// GET a manifest URL, enforce HTTP 2xx, decode, and apply
    /// ``validateReleasePayload``. Shared by the Worker fetch and the R2
    /// fallback so both paths enforce the identical contract. The only
    /// data sent is the (public) app version — in the `?v=` query on the
    /// Worker URL, and in the ``User-Agent`` on both.
    ///
    /// ``URLRequest.cachePolicy`` is ``reloadIgnoringLocalCacheData``
    /// because Cloudflare's edge cache is the only cache layer that
    /// matters — a stale local URLCache entry must not survive launches.
    nonisolated static func fetchManifest(from url: URL) async throws -> Release {
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.cachePolicy = .reloadIgnoringLocalCacheData
        request.timeoutInterval = 10
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue(
            "Rapid-Desktop/\(bundleVersion())",
            forHTTPHeaderField: "User-Agent"
        )
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw UpdateError.transport("not http")
        }
        guard (200..<300).contains(http.statusCode) else {
            throw UpdateError.httpStatus(http.statusCode)
        }
        let release: Release
        do {
            release = try JSONDecoder().decode(Release.self, from: data)
        } catch {
            throw UpdateError.decode(String(describing: error))
        }
        // Post-decode validation. The Codable decode only proves the
        // payload shape; a misbehaving (or compromised) manifest
        // publisher can still feed bogus values that travel into the
        // installer + browser-open flows. Reject on contract
        // violation rather than handing the bad payload to
        // downstream UI.
        // [codex audit r1 UpdateChecker.swift:161]
        try validateReleasePayload(release)
        return release
    }

    /// Apply the manifest contract to a decoded ``Release``: schema
    /// pin, version grammar, release URL on the allowlist, and
    /// DMG URL (if present) on the download allowlist. Throws
    /// ``UpdateError.decode`` on any failure so the existing UI
    /// surface treats it as a parse miss.
    nonisolated static func validateReleasePayload(_ release: Release) throws {
        guard release.schemaVersion == 1 else {
            throw UpdateError.decode("unexpected schema_version \(release.schemaVersion)")
        }
        // Version must parse to one OR MORE strictly numeric segments.
        //
        // PR #26 codex meta-review finding 6 (P2): the previous gate
        // used ``numericParts`` which compactMap's ``Int($0)`` over
        // each split, silently DROPPING non-numeric segments. That
        // meant ``"1.evil"`` and ``"1.2.evil"`` both passed validation
        // (yielding [1] and [1, 2] respectively) and the malformed
        // string would then surface in the UI as the displayed
        // "Update available" target. Validate every segment is
        // present and digit-only here; the comparison helper can
        // continue to be lenient when reading bytes off the wire
        // because by the time it runs the version has already been
        // accepted by this gate.
        guard let parts = strictNumericParts(of: release.version),
              !parts.isEmpty else {
            throw UpdateError.decode("version is not numeric: \(release.version)")
        }
        // Release page URL must be HTTPS on the release-host allowlist.
        guard let releaseURL = URL(string: release.htmlURL),
              releaseURL.scheme?.lowercased() == "https",
              releaseURL.user == nil,
              releaseURL.password == nil,
              let releaseHost = releaseURL.host?.lowercased(),
              updateReleaseHostAllowlist.contains(releaseHost) else {
            throw UpdateError.decode("html_url is not in the release allowlist: \(release.htmlURL)")
        }
        // DMG URL is optional; when present it must be HTTPS on the
        // download-host allowlist. Sparkle owns the real download, so this
        // no longer guards a fetch we perform — it keeps a manifest that
        // names an unexpected download host from being accepted as
        // well-formed, which is the shape the status UI trusts.
        if let dmg = release.dmgURL {
            guard let dmgURL = URL(string: dmg),
                  dmgURL.scheme?.lowercased() == "https",
                  dmgURL.user == nil,
                  dmgURL.password == nil,
                  let dmgHost = dmgURL.host?.lowercased(),
                  updateDownloadHostAllowlist.contains(dmgHost) else {
                throw UpdateError.decode("dmg_url is not in the download allowlist: \(dmg)")
            }
        }
    }

    /// Production update-ping endpoint — a thin Cloudflare Worker on
    /// the landing-page domain that returns the byte-identical manifest
    /// the R2 object (`dl.rapidmlx.com/latest.json`) serves, while also
    /// counting the poll as an aggregate active-install signal. Public,
    /// no auth. See the class docstring for the schema + privacy
    /// contract. This is the BASE URL; the app version is appended as
    /// the `?v=` query by ``endpointURL(forVersion:)``.
    nonisolated static let endpoint = "https://rapidmlx.com/api/desktop-update"

    /// R2 fallback manifest — the static object the Worker proxies.
    /// Used ONLY when the Worker fetch fails (down / non-2xx / bad
    /// JSON), so a Worker outage can never stop every client from
    /// seeing updates while `dl.rapidmlx.com/latest.json` is still
    /// healthy. No `?v=` query: R2 serves a static object and counts
    /// nothing, so a poll that falls back is simply not counted as an
    /// active-install signal. See ``fetchWithFallback``.
    nonisolated static let fallbackEndpoint = "https://dl.rapidmlx.com/latest.json"

    /// Build the concrete request URL for ``endpoint`` carrying the
    /// current app version as the sole `?v=` query parameter. Uses
    /// ``URLComponents`` so the version is percent-encoded defensively
    /// — a malformed ``CFBundleShortVersionString`` can never inject
    /// extra query pairs or break the URL. Returns nil only if the
    /// compile-time-constant base fails to parse (never in practice),
    /// in which case ``defaultFetch`` throws ``invalidURL`` and the
    /// fail-open path keeps the app on its current version.
    nonisolated static func endpointURL(forVersion version: String) -> URL? {
        guard var components = URLComponents(string: endpoint) else { return nil }
        components.queryItems = [URLQueryItem(name: "v", value: version)]
        return components.url
    }

    /// UserDefaults key for the update-check opt-out toggle. Default is
    /// ON (checks enabled), matching ``TelemetryConfig.enabledKey``:
    /// the key is absent until the user explicitly flips it, so
    /// ``updateChecksEnabled`` coerces nil → true. No Settings toggle
    /// is wired to this key yet — the env-var kill switches below are
    /// the shipped opt-out; a "Automatically check for updates" toggle
    /// in Settings → App can bind this key later.
    nonisolated static let updateCheckEnabledKey = "com.rapidmlx.rapid.updateCheck.enabled"

    /// Whether the periodic + launch update check should run at all.
    ///
    /// Returns ``false`` (skip the network entirely, record no
    /// active-install signal) when EITHER environment kill switch is
    /// set to a truthy value, OR the UserDefaults toggle has been
    /// explicitly turned off. Env vars win over the stored toggle so a
    /// privacy-conscious launcher can force-disable regardless of the
    /// persisted preference.
    ///
    ///   * ``RAPIDMLX_NO_UPDATE_CHECK`` — app-specific kill switch.
    ///   * ``DO_NOT_TRACK`` — the cross-tool convention (consoledonottrack.com).
    ///
    /// "Truthy" = present and not one of ``""`` / ``"0"`` / ``"false"``
    /// / ``"no"`` (case-insensitive), so `DO_NOT_TRACK=0` still allows
    /// the check per the DNT convention.
    nonisolated static func updateChecksEnabled(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        defaults: UserDefaults = .standard
    ) -> Bool {
        func isTruthy(_ raw: String?) -> Bool {
            guard let raw = raw?.trimmingCharacters(in: .whitespaces).lowercased(),
                  !raw.isEmpty else { return false }
            return !["0", "false", "no"].contains(raw)
        }
        if isTruthy(environment["RAPIDMLX_NO_UPDATE_CHECK"]) { return false }
        if isTruthy(environment["DO_NOT_TRACK"]) { return false }
        return defaults.object(forKey: updateCheckEnabledKey) as? Bool ?? true
    }

    /// Read ``CFBundleShortVersionString`` from the running bundle.
    /// Falls back to ``"0.0.0"`` so the comparison still terminates;
    /// a 0.0.0 install will think every release is newer, which is
    /// the right failure mode (better to nag than to silently miss
    /// updates).
    nonisolated static func bundleVersion() -> String {
        let bundle = Bundle.main
        if let s = bundle.infoDictionary?["CFBundleShortVersionString"] as? String,
           !s.isEmpty {
            return s
        }
        return "0.0.0"
    }

    /// Semver-ish "is A newer than B" comparison.
    ///
    /// We split on ``.``, parse each segment as an ``Int``, and
    /// compare lexicographically. Pre-release suffixes like
    /// ``"0.3.1-rc1"`` are coerced to their numeric prefix
    /// (``[0,3,1]``) — we don't ship pre-releases through the public
    /// update channel today so this is fine. If we ever do, this is
    /// the function to extend (and the only one — every other call
    /// site reads through ``availableUpdate``).
    nonisolated static func isNewer(_ a: String, than b: String) -> Bool {
        let aParts = numericParts(of: a)
        let bParts = numericParts(of: b)
        let n = max(aParts.count, bParts.count)
        for i in 0..<n {
            let av = i < aParts.count ? aParts[i] : 0
            let bv = i < bParts.count ? bParts[i] : 0
            if av != bv { return av > bv }
        }
        return false
    }

    nonisolated private static func numericParts(of v: String) -> [Int] {
        // Strip a leading "v" so "v0.3.1" and "0.3.1" compare the
        // same way (the publish script already strips this when
        // populating `version`, but defence in depth costs nothing).
        var s = v
        if s.hasPrefix("v") || s.hasPrefix("V") {
            s = String(s.dropFirst())
        }
        // Drop a pre-release tag (anything after "-").
        if let dash = s.firstIndex(of: "-") {
            s = String(s[..<dash])
        }
        return s.split(separator: ".").compactMap { Int($0) }
    }

    /// Strict version-grammar parser used at the validation gate.
    /// Returns nil when ANY segment is empty or contains a non-digit
    /// — i.e. "1.evil", "1..2", "1.2.bad", "abc" all fail.
    nonisolated static func strictNumericParts(of v: String) -> [Int]? {
        var s = v
        if s.hasPrefix("v") || s.hasPrefix("V") {
            s = String(s.dropFirst())
        }
        if let dash = s.firstIndex(of: "-") {
            s = String(s[..<dash])
        }
        guard !s.isEmpty else { return nil }
        // ``split`` with the default ``omittingEmptySubsequences: true``
        // would hide a trailing-dot or double-dot case; force false so
        // we can reject those explicitly.
        let segments = s.split(separator: ".", omittingEmptySubsequences: false)
        var out: [Int] = []
        out.reserveCapacity(segments.count)
        for seg in segments {
            guard !seg.isEmpty,
                  seg.unicodeScalars.allSatisfy({ CharacterSet.decimalDigits.contains($0) }),
                  let n = Int(seg) else {
                return nil
            }
            out.append(n)
        }
        return out
    }
}

enum UpdateError: Error, LocalizedError, CustomStringConvertible {
    case invalidURL
    case transport(String)
    case httpStatus(Int)
    case decode(String)

    var description: String {
        switch self {
        case .invalidURL: return "invalid update endpoint"
        case .transport(let s): return "transport error: \(s)"
        case .httpStatus(let n): return "update server returned HTTP \(n)"
        case .decode(let s): return "could not parse update payload: \(s)"
        }
    }

    /// LocalizedError contract — surfaces in ``error.localizedDescription``.
    /// Without this conformance, the Settings → App panel rendered
    /// ``error.localizedDescription`` as the Apple default
    /// "(Rapid.UpdateError error 1.)" — a developer-only string that
    /// leaked into the user-visible "Last check failed: …" banner
    /// (issue surfaced during v0.6.7 polish walk). With LocalizedError
    /// the same call site now returns the prose above so the user
    /// sees something actionable ("transport error: A server with the
    /// specified hostname could not be found").
    var errorDescription: String? { description }
}
