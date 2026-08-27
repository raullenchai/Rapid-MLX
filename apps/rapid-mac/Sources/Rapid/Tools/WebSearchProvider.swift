import Foundation
import Observation

/// Static metadata for one ``web_search`` backend. One value per
/// provider, produced by a single switch in
/// ``WebSearchProvider/descriptor`` — adding a backend means writing
/// ONE descriptor plus a client, not editing half a dozen scattered
/// switches (issue #2040; the old shape needed 8 edits per provider).
struct WebSearchProviderDescriptor: Sendable {
    let displayName: String
    /// Plain-English subtitle for the Settings picker. Tells the
    /// user what they're trading off (cost / signup) without
    /// having to read the docs.
    let subtitle: String
    /// Marketing / landing page for the provider's API. Kept
    /// distinct from ``keyDashboardURL`` — see that field.
    let keyConsoleURL: URL?
    /// Direct link to the API-keys dashboard — the page that lists
    /// existing keys and offers a "create new key" affordance. The
    /// Settings key row links here so the user lands one click from
    /// a usable key instead of on a marketing page. Issue #193.
    let keyDashboardURL: URL?
    /// The provider cannot run without a key. At dispatch a
    /// requiresKey provider with no stored key falls back to the
    /// keyless chain (Keenable → DuckDuckGo).
    let requiresKey: Bool
    /// The provider can use a key at all. True for every
    /// requiresKey provider, and for Keenable — where the key is
    /// optional and lifts the shared keyless rate limit. Drives the
    /// Settings key field + Keychain prefetch.
    let acceptsKey: Bool
    /// A rejected optional key can be removed and the same request can run in
    /// this provider's supported keyless mode. This is deliberately narrower
    /// than generic retry or fallback eligibility.
    let recoversRejectedKeyKeylessly: Bool
    /// Keychain account label for this provider's API key. Distinct
    /// per-provider so one provider's key never leaks into another's
    /// call.
    let keychainAccount: String?
}

/// Which backend does ``web_search`` hit?
///
/// The roster (2026-08, issues #2040–#2043; quality figures from the
/// Artificial Analysis Search Index, 2026-08-18):
///   * **Keenable** — the zero-setup default. No account, no key: the
///     keyless pool allows 1 000 requests/hour per IP. An optional
///     free key lifts the shared cap. Scores above both legacy keyed
///     backends in independent agent-search benchmarking.
///   * **Parallel** — the recommended keyed backend; strongest
///     measured result quality of the providers benchmarked, with a
///     recurring free monthly credit (≈1 000 searches).
///   * **Tavily** — 1 000 queries/month free tier; agent-tuned
///     snippets.
///   * **Brave Search** — kept for existing keys. Brave dropped its
///     card-free tier in Feb 2026: every plan now requires a card on
///     file and auto-bills overage (issue #2043), so it is no longer
///     pitched as a free upgrade.
///   * **DuckDuckGo** — the original v0.3 scrape backend, demoted to
///     backstop: its free HTML endpoint throttles per IP after a
///     handful of searches (measured 2026-08-05) and result quality
///     is poor. Kept because it needs nothing and never bills.
///
/// Keys are stored in the macOS Keychain, NOT UserDefaults — leaking
/// a search key reads as a real privacy bug (the key is
/// account-linked and per-call billing is metered against it).
enum WebSearchProvider: String, CaseIterable, Codable, Identifiable, Sendable {
    // Case order == Settings radio order: the zero-setup default
    // first, then the recommended keyed upgrade, then the rest.
    case keenable
    case parallel
    case tavily
    case brave
    case duckduckgo

    var id: String { rawValue }

    /// The single source of provider metadata. Every legacy
    /// per-property accessor below forwards here so existing call
    /// sites didn't have to churn in the #2040 refactor.
    var descriptor: WebSearchProviderDescriptor {
        switch self {
        case .keenable:
            return WebSearchProviderDescriptor(
                displayName: "Keenable",
                subtitle: "No key needed — the default. A free key lifts the shared hourly limit.",
                keyConsoleURL: URL(string: "https://keenable.ai/"),
                keyDashboardURL: URL(string: "https://keenable.ai/console"),
                requiresKey: false,
                acceptsKey: true,
                recoversRejectedKeyKeylessly: true,
                keychainAccount: "rapid.web-search.keenable"
            )
        case .parallel:
            return WebSearchProviderDescriptor(
                displayName: "Parallel",
                subtitle: "Recommended — the highest-quality backend in our testing. Free key covers about 1 000 searches a month.",
                keyConsoleURL: URL(string: "https://parallel.ai/"),
                keyDashboardURL: URL(string: "https://platform.parallel.ai/"),
                requiresKey: true,
                acceptsKey: true,
                recoversRejectedKeyKeylessly: false,
                keychainAccount: "rapid.web-search.parallel"
            )
        case .tavily:
            return WebSearchProviderDescriptor(
                displayName: "Tavily",
                subtitle: "Requires a free Tavily API key. 1 000 queries/month, agent-tuned snippets.",
                keyConsoleURL: URL(string: "https://app.tavily.com/home"),
                keyDashboardURL: URL(string: "https://app.tavily.com/home"),
                requiresKey: true,
                acceptsKey: true,
                recoversRejectedKeyKeylessly: false,
                keychainAccount: "rapid.web-search.tavily"
            )
        case .brave:
            return WebSearchProviderDescriptor(
                displayName: "Brave Search",
                // Not "a free key" any more: Brave dropped the
                // card-free tier in Feb 2026 — every plan keeps a
                // card on file and overage is auto-billed. The
                // subtitle must say so BEFORE the user follows the
                // key link (issue #2043).
                subtitle: "Requires a Brave Search API key and a card on file — usage past the monthly credit is auto-billed.",
                keyConsoleURL: URL(string: "https://brave.com/search/api/"),
                // The dashboard host is separate from the API host;
                // the API-serving host returns HTTP 403 for this UI
                // route.
                keyDashboardURL: URL(string: "https://api-dashboard.search.brave.com/app/keys"),
                requiresKey: true,
                acceptsKey: true,
                recoversRejectedKeyKeylessly: false,
                keychainAccount: "rapid.web-search.brave"
            )
        case .duckduckgo:
            // The subtitle used to end "works out of the box." It
            // doesn't: the free HTML endpoint throttles per IP after
            // a handful of searches (measured 2026-08-05 — one 200,
            // then 202 non-results pages for every query after it).
            // A user who hits that throttle and comes here looking
            // for the problem must not be told the backend is fine.
            return WebSearchProviderDescriptor(
                displayName: "DuckDuckGo",
                subtitle: "No key required. Backstop only — throttled after a few searches, and result quality is poor.",
                keyConsoleURL: nil,
                keyDashboardURL: nil,
                requiresKey: false,
                acceptsKey: false,
                recoversRejectedKeyKeylessly: false,
                keychainAccount: nil
            )
        }
    }

    // MARK: Forwarding accessors (pre-#2040 API, unchanged call sites)

    var displayName: String { descriptor.displayName }
    var subtitle: String { descriptor.subtitle }
    var keyConsoleURL: URL? { descriptor.keyConsoleURL }
    var keyDashboardURL: URL? { descriptor.keyDashboardURL }
    var requiresKey: Bool { descriptor.requiresKey }
    var acceptsKey: Bool { descriptor.acceptsKey }
    var recoversRejectedKeyKeylessly: Bool { descriptor.recoversRejectedKeyKeylessly }
    var keychainAccount: String? { descriptor.keychainAccount }
}

/// User-facing, persisted web-search configuration. Provider
/// choice lives in UserDefaults; the API key (when needed) lives
/// in Keychain. ``ChatViewModel`` and ``WebSearchTool`` both read
/// this; the Settings panel mutates it.
@MainActor
@Observable
final class WebSearchConfig {
    struct CredentialSnapshot: Equatable, Sendable {
        let key: String?
        let revision: UInt64
    }

    /// Backed by ``UserDefaults`` under a single stable key so a
    /// reset-defaults action (or a corrupted prefs file) doesn't
    /// need to sweep multiple keys. The default is ``.keenable``
    /// (#2041): zero-setup like DuckDuckGo was, but with usable
    /// result quality and no per-handful-of-searches throttle. A
    /// user who ever picked a provider explicitly has a stored raw
    /// value and keeps their choice.
    var provider: WebSearchProvider {
        didSet {
            guard oldValue != provider else { return }
            defaults.set(provider.rawValue, forKey: Self.providerKey)
        }
    }

    private static let providerKey = "rapid.webSearch.provider"

    private let defaults: UserDefaults
    private let keychain: KeychainStoring

    /// Per-account in-memory cache of decrypted secrets. Populated
    /// on first read; ``setAPIKey`` keeps it in sync with the
    /// keychain. Closes issue #23: ``SecItemCopyMatching`` blocks
    /// the main actor across the securityd XPC hop (can spike to
    /// tens of ms when the Keychain agent is cold), and the
    /// ``web_search`` tool fires this on every dispatch.
    private var keyCache: [String: String] = [:]

    /// Accounts we've already hit the keychain for. Separate from
    /// ``keyCache`` so we can distinguish "not in cache, haven't
    /// looked" from "looked, no secret stored" — the latter must
    /// short-circuit ``apiKey(for:)`` without another keychain hit.
    private var probedAccounts: Set<String> = []
    /// Accounts whose no-UI Keychain lookup was refused. Kept distinct from
    /// ordinary absence so Settings can ask for re-entry without ever causing
    /// the macOS login-keychain password dialog.
    private var unavailableAccounts: Set<String> = []
    /// Process-local mutation generation per Keychain account. Value equality
    /// cannot detect an ABA replacement (K → another value → K), so callers
    /// that act on an async response capture this revision with the key.
    private var keyRevisions: [String: UInt64] = [:]

    /// Injection points for tests. Default arguments use the real
    /// system Keychain + the host's standard UserDefaults; the
    /// test suite swaps in an in-memory pair so contracts can be
    /// pinned without polluting the user's actual settings.
    init(
        defaults: UserDefaults = .standard,
        keychain: KeychainStoring = SystemKeychain()
    ) {
        self.defaults = defaults
        self.keychain = keychain
        if let raw = defaults.string(forKey: Self.providerKey),
           let p = WebSearchProvider(rawValue: raw) {
            self.provider = p
        } else {
            self.provider = .keenable
        }
    }

    // MARK: - Key access

    /// Returns the stored key for ``provider`` (if it needs one
    /// and one has been entered). DuckDuckGo always returns nil.
    ///
    /// Cached after first read — subsequent calls do **not** hit
    /// the keychain. ``setAPIKey`` invalidates and refreshes the
    /// cache on every write so the cache stays consistent with
    /// the on-disk Keychain state owned by this process.
    func apiKey(for provider: WebSearchProvider) -> String? {
        guard let account = provider.keychainAccount else { return nil }
        if probedAccounts.contains(account) {
            // Trim whitespace defensively — users frequently paste a
            // key with a trailing newline they didn't notice, and the
            // HTTP header has to land clean or the upstream rejects.
            return keyCache[account]?.trimmingCharacters(in: .whitespacesAndNewlines).nonEmpty
        }
        let result = keychain.readWithoutUserInteraction(account: account)
        probedAccounts.insert(account)
        guard case .found(let value) = result else {
            if result == .unavailable { unavailableAccounts.insert(account) }
            return nil
        }
        unavailableAccounts.remove(account)
        if !value.isEmpty {
            keyCache[account] = value
        }
        return value.trimmingCharacters(in: .whitespacesAndNewlines).nonEmpty
    }

    /// Atomically capture the cached credential value and its mutation
    /// generation on the main actor. External Keychain edits are intentionally
    /// outside this process-owned cache contract, just like ``apiKey(for:)``.
    func credentialSnapshot(for provider: WebSearchProvider) -> CredentialSnapshot {
        guard let account = provider.keychainAccount else {
            return CredentialSnapshot(key: nil, revision: 0)
        }
        return CredentialSnapshot(
            key: apiKey(for: provider),
            revision: keyRevisions[account, default: 0]
        )
    }

    /// Write or clear the key for ``provider``. Passing an empty
    /// / whitespace-only string clears the slot (so the SecureField's
    /// "delete + tab" gesture removes the key instead of storing
    /// an empty record that pretends to be a key).
    ///
    /// **Auto-promote on first key paste.** If the user is on a
    /// keyless backend (the install-default ``.keenable``, or
    /// ``.duckduckgo``) and pastes a key for a key-REQUIRING
    /// provider, we silently flip ``provider`` to the keyed backend.
    ///
    /// Why: a user who goes to the trouble of pasting a
    /// Parallel/Tavily/Brave key has explicitly opted into that
    /// backend; requiring them to ALSO flip the provider picker is
    /// silent-broken UX (their key is stored but never used).
    ///
    /// We only auto-promote from a keyless state. If the user has
    /// already explicitly chosen a keyed backend (say `.parallel`)
    /// and then pastes a key for ANOTHER keyed backend (`.tavily`),
    /// the explicit prior choice wins — they're managing both keys
    /// without re-selecting. Pasting a Keenable key while on
    /// Keenable is not a promotion either — the key is picked up in
    /// place (it lifts the shared rate cap). Clearing a key never
    /// demotes; the user can manually switch back in Settings.
    /// Returns ``true`` when the Keychain write/delete that backs the
    /// requested mutation actually succeeded. v0.6.7 added the
    /// return value to drive the inline "Saved" confirmation in
    /// Settings → Web Search — silently dropping a write (locked DB
    /// / missing entitlement) must not flash a misleading success
    /// toast at the user. Pre-existing call sites that just want
    /// fire-and-forget can keep ignoring the result thanks to
    /// ``@discardableResult``.
    @discardableResult
    func setAPIKey(_ key: String?, for provider: WebSearchProvider) -> Bool {
        guard let account = provider.keychainAccount else { return false }
        let trimmed = key?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if trimmed.isEmpty {
            // ``delete`` returns true on success or ``errSecItemNotFound``;
            // either way the post-condition is "no key on disk", so
            // mark probed + clear the cache. Real failure (e.g. locked
            // DB) returns false — preserve the prior probe state and
            // leave the cache untouched so a later ``apiKey(for:)``
            // can still resolve a value that's actually present on
            // disk.
            if keychain.delete(account: account) {
                keyCache.removeValue(forKey: account)
                unavailableAccounts.remove(account)
                probedAccounts.insert(account)
                keyRevisions[account, default: 0] &+= 1
                return true
            }
            return false
        } else {
            // ``SystemKeychain.write`` returns false on a real
            // Keychain failure (entitlement / locked DB / item-
            // class collision). On failure we must NOT update the
            // in-memory cache (would lie to later ``apiKey(for:)``
            // callers — they'd see the key in memory while the
            // disk-backed Keychain is empty), NOT promote the
            // provider (would leave defaults pointing at a paid
            // backend with no usable key; ``currentProviderUsable``
            // flips to false at dispatch and the user lands on a
            // confusing "selected Brave, getting DDG" state on next
            // launch), AND NOT mark probed — codex r2 P2: negative-
            // caching a failed write would short-circuit
            // ``apiKey(for:)`` to nil for the rest of the process
            // even though a previously-stored key may still be on
            // disk, silently disabling the provider until restart.
            if keychain.write(account: account, secret: trimmed) {
                keyCache[account] = trimmed
                unavailableAccounts.remove(account)
                probedAccounts.insert(account)
                keyRevisions[account, default: 0] &+= 1
                // Auto-promote happens AFTER a successful keychain
                // write so the provider state can never get ahead of
                // the stored key. Gate on ``provider.requiresKey``
                // so a key-optional provider (Keenable) can't
                // trigger promotion via this path — the silent-
                // broken UX only exists for key-REQUIRING backends,
                // and a keyless backend keeps working when a key for
                // it lands.
                if !self.provider.requiresKey && provider.requiresKey {
                    self.provider = provider
                }
                return true
            }
            return false
        }
    }

    /// True if the currently-selected provider has a usable key
    /// (or doesn't need one). Drives the "Effective backend" hint
    /// in Settings — when this is false, the tool will fall back
    /// to the keyless chain (Keenable) at dispatch time.
    var currentProviderUsable: Bool {
        !provider.requiresKey || apiKey(for: provider) != nil
    }

    // MARK: - Lazy async lookup
    //
    // Settings must not touch Keychain merely because its Tools page opened.
    // A required-key backend selection or an API-key field focus calls the
    // single-provider helper below; actual tool dispatch can also resolve its
    // selected provider through ``apiKey(for:)``. Both routes use the same
    // no-authentication-UI storage contract.
    //
    // The pre-existing positive/negative cache contracts in
    // ``WebSearchKeyCacheTests`` are preserved verbatim: ``apiKey(for:)``
    // still does its own one-shot read + probe-bookkeeping, and a
    // ``setAPIKey`` write still primes the cache directly. The new
    // surface only adds an explicit "warm in the background" path plus
    // a read-only cache peek that never hits Keychain.
    //
    // We deliberately do NOT add the prefetch to ``init`` — wiring it
    // into the @State initializer would re-introduce a synchronous
    // Keychain read on the first construction of ``WebSearchConfig``,
    // which happens on the main actor during ``RapidApp`` boot. The
    // caller decides when to warm.

    /// Cache-only read. Returns ``.unknown`` when this provider has
    /// not been probed yet, ``.present(value)`` when the cache holds
    /// a non-empty key, and ``.absent`` when the cache has been told
    /// there is no key (either because a write cleared it or because
    /// a prior ``apiKey(for:)`` / ``prefetchAPIKey`` call resolved to
    /// nil).
    ///
    /// Never touches Keychain. Safe to call from any tight render loop.
    func cachedKeyState(for provider: WebSearchProvider) -> CachedKeyState {
        guard let account = provider.keychainAccount else { return .absent }
        guard probedAccounts.contains(account) else { return .unknown }
        if unavailableAccounts.contains(account) { return .unavailable }
        if let trimmed = keyCache[account]?
            .trimmingCharacters(in: .whitespacesAndNewlines).nonEmpty {
            return .present(trimmed)
        }
        return .absent
    }

    /// Resolve ``cachedKeyState(for:)`` for ``provider`` without blocking
    /// the calling actor on the Keychain XPC hop. Returns after the
    /// cache + probed set have been populated. Safe to call repeatedly;
    /// a probed provider short-circuits without re-reading Keychain.
    ///
    /// The Keychain read itself runs on a detached background task —
    /// that's the whole reason this surface exists. The final cache
    /// mutation hops back onto the main actor so the @Observable
    /// machinery doesn't fire from off-actor and so callers that gate
    /// UI on ``cachedKeyState`` see a coherent snapshot.
    ///
    /// **Cancellation contract** (codex r1 MAJOR). The detached read
    /// itself cannot be interrupted — ``SecItemCopyMatching`` is a
    /// blocking call across securityd XPC and the kernel gives us no
    /// signal to abort it short of killing the process. What we CAN
    /// guarantee is that the result is discarded if the caller's
    /// ``Task`` was cancelled before we mutate ``keyCache``, so a
    /// Web-Search-tab-flick-then-leave never lands a stale read in the
    /// @Observable config (which would in turn flip the UI on a panel
    /// the user has already navigated away from). We also honour an
    /// authoritative concurrent write by re-checking
    /// ``probedAccounts`` after the read returns — race with
    /// ``setAPIKey`` / ``apiKey(for:)`` resolves in favour of the
    /// in-line path because it's the one the user explicitly drove.
    func prefetchAPIKey(for provider: WebSearchProvider) async {
        guard let account = provider.keychainAccount else { return }
        if probedAccounts.contains(account) { return }
        // Honour cancellation BEFORE we cross XPC. If the .task is
        // already cancelled (because the view disappeared between the
        // caller scheduling us and getting here) skip the read
        // entirely.
        if Task.isCancelled { return }
        // Snapshot the keychain handle off-actor: ``KeychainStoring`` is
        // ``Sendable`` so this is safe; capturing ``self`` into the
        // detached task would force ``WebSearchConfig`` to be Sendable,
        // which it isn't (it's main-actor isolated for the @Observable
        // reasons above).
        let keychain = self.keychain
        let result = await Task.detached(priority: .userInitiated) {
            keychain.readWithoutUserInteraction(account: account)
        }.value
        // codex r1 MAJOR: discard the result if our parent task was
        // cancelled while we were waiting on the detached read — the
        // view that triggered us is gone and we must not mutate the
        // @Observable cache (would push an out-of-band UI refresh on
        // whatever screen the user navigated to).
        if Task.isCancelled { return }
        // Re-check probed: another caller may have raced us through
        // ``apiKey(for:)`` / ``setAPIKey`` while we were waiting on the
        // detached read. If so, drop our snapshot — the in-line
        // read / write is authoritative (the write path also primes
        // the cache directly).
        if probedAccounts.contains(account) { return }
        if case .found(let value) = result, !value.isEmpty {
            keyCache[account] = value
        }
        if result == .unavailable { unavailableAccounts.insert(account) }
        probedAccounts.insert(account)
    }

}

/// Result of ``WebSearchConfig.cachedKeyState(for:)``. ``.unknown``
/// is the load-bearing case: it tells the UI "we have NOT probed
/// the Keychain for this account yet, render a neutral placeholder
/// — do not assume absent, and do not block on a fresh read."
enum CachedKeyState: Equatable, Sendable {
    case unknown
    case absent
    case present(String)
    case unavailable

    var hasKey: Bool {
        if case .present = self { return true }
        return false
    }
}

private extension String {
    /// Returns nil for empty / whitespace-only strings. Keeps the
    /// ``apiKey(for:)`` call site free of "is this string actually
    /// present?" boilerplate.
    var nonEmpty: String? {
        let trimmed = self.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}
