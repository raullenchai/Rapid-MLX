import Foundation
import Observation

/// Which backend does ``web_search`` hit? v0.4.41 adds the two
/// dominant paid options on top of the existing free DDG endpoint.
///
/// Why these three:
///   * **DuckDuckGo** — no key, decent quality, the original
///     v0.3 backend. Stays the default so a fresh install keeps
///     working without any setup.
///   * **Brave Search** — 2 000 queries/month free tier; high
///     quality, fast, ad-free index.
///   * **Tavily** — 1 000 queries/month free tier; explicitly
///     designed for LLM agents (returns clean snippets + a
///     ranked list, no SERP cruft).
///
/// Both paid options ask for a long-lived API key. Keys are
/// stored in the macOS Keychain, NOT UserDefaults — leaking a
/// Brave/Tavily key reads as a real privacy bug (the key is
/// account-linked and per-call billing is metered against it).
enum WebSearchProvider: String, CaseIterable, Codable, Identifiable, Sendable {
    case duckduckgo
    case brave
    case tavily

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .duckduckgo: return "DuckDuckGo"
        case .brave:      return "Brave Search"
        case .tavily:     return "Tavily"
        }
    }

    /// Plain-English subtitle for the Settings picker. Tells the
    /// user what they're trading off (cost / signup) without
    /// having to read the docs.
    ///
    /// The DuckDuckGo line used to end "works out of the box." It
    /// doesn't: the free HTML endpoint throttles per IP after a
    /// handful of searches (measured 2026-08-05 — one 200, then 202
    /// non-results pages for every query after it). A user who hits
    /// that throttle and comes here looking for the problem must not
    /// be told the backend is fine.
    var subtitle: String {
        switch self {
        case .duckduckgo:
            return "No key required. Best-effort — throttled after a few searches; the keyed backends aren't."
        case .brave:
            return "Requires a free Brave Search API key. 2 000 queries/month."
        case .tavily:
            return "Requires a free Tavily API key. 1 000 queries/month, agent-tuned snippets."
        }
    }

    /// Where the user goes to mint a key. Shown as an "Open
    /// dashboard" link in Settings so the affordance is one click
    /// instead of "Google for the right URL."
    var keyConsoleURL: URL? {
        switch self {
        case .duckduckgo: return nil
        case .brave:      return URL(string: "https://brave.com/search/api/")
        case .tavily:     return URL(string: "https://app.tavily.com/home")
        }
    }

    /// Direct link to the API-keys dashboard for the provider — the
    /// page that lists existing keys and offers a "create new key"
    /// affordance. Distinct from ``keyConsoleURL`` (the marketing
    /// landing) because the "Upgrade to Brave" / "Upgrade to Tavily"
    /// nudge in Settings + Onboarding wants to drop the user one
    /// click closer to a usable key — landing on a marketing page
    /// adds an extra "click Sign in / Get key" hop. Issue #193.
    ///
    /// For Tavily this resolves to the same /home page as
    /// ``keyConsoleURL`` (the dashboard IS the home view). Brave
    /// splits: ``keyConsoleURL`` points at the public api docs
    /// landing, ``keyDashboardURL`` jumps straight to /app/keys.
    var keyDashboardURL: URL? {
        switch self {
        case .duckduckgo: return nil
        case .brave:      return URL(string: "https://api.search.brave.com/app/keys")
        case .tavily:     return URL(string: "https://app.tavily.com/home")
        }
    }

    /// True only when the provider needs an API key. Lets the
    /// Settings panel hide the SecureField when DDG is selected.
    var requiresKey: Bool {
        switch self {
        case .duckduckgo: return false
        case .brave, .tavily: return true
        }
    }

    /// Keychain account label used to store this provider's
    /// API key. Distinct per-provider so a Brave key never
    /// leaks into a Tavily call (and vice versa).
    var keychainAccount: String? {
        switch self {
        case .duckduckgo: return nil
        case .brave:      return "rapid.web-search.brave"
        case .tavily:     return "rapid.web-search.tavily"
        }
    }
}

/// User-facing, persisted web-search configuration. Provider
/// choice lives in UserDefaults; the API key (when needed) lives
/// in Keychain. ``ChatViewModel`` and ``WebSearchTool`` both read
/// this; the Settings panel mutates it.
@MainActor
@Observable
final class WebSearchConfig {
    /// Backed by ``UserDefaults`` under a single stable key so a
    /// reset-defaults action (or a corrupted prefs file) doesn't
    /// need to sweep multiple keys. The default is ``.duckduckgo``
    /// because that's the only zero-setup option.
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
            self.provider = .duckduckgo
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
        let value = keychain.read(account: account)
        probedAccounts.insert(account)
        if let value, !value.isEmpty {
            keyCache[account] = value
        }
        return value?.trimmingCharacters(in: .whitespacesAndNewlines).nonEmpty
    }

    /// Write or clear the key for ``provider``. Passing an empty
    /// / whitespace-only string clears the slot (so the SecureField's
    /// "delete + tab" gesture removes the key instead of storing
    /// an empty record that pretends to be a key).
    ///
    /// **Auto-promote on first key paste.** If the user is still on
    /// the install-default ``.duckduckgo`` backend and pastes a key
    /// for a paid provider (``.brave`` or ``.tavily``), we silently
    /// flip ``provider`` to the keyed backend.
    ///
    /// Why: DDG HTML scraping is silently rate-limited / anti-bot-
    /// blocked in production today — the `cc=botnet` anomaly modal
    /// returns 0 results without surfacing a real error, so the
    /// tool appears to "work" while delivering nothing. When a user
    /// goes to the trouble of pasting a Brave/Tavily key they have
    /// explicitly opted into a more reliable backend; requiring them
    /// to ALSO flip the provider picker is silent-broken UX (their
    /// key is stored but never used).
    ///
    /// We only auto-promote from the default DDG state. If the user
    /// has already explicitly chosen a paid backend (say `.brave`)
    /// and then pastes a key for the OTHER paid backend (`.tavily`),
    /// the explicit prior choice wins — they're managing both keys
    /// without re-selecting. Clearing a key never demotes; the user
    /// can manually switch back to DDG in Settings.
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
                probedAccounts.insert(account)
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
                probedAccounts.insert(account)
                // Auto-promote happens AFTER a successful keychain
                // write so the provider state can never get ahead of
                // the stored key. Gate on ``provider.requiresKey``
                // so a hypothetical future free provider can't
                // trigger promotion via this path — the silent-
                // broken UX only exists for keyed backends.
                if self.provider == .duckduckgo && provider.requiresKey {
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
    /// to DuckDuckGo at dispatch time.
    var currentProviderUsable: Bool {
        switch provider {
        case .duckduckgo: return true
        case .brave, .tavily: return apiKey(for: provider) != nil
        }
    }

    // MARK: - Async prefetch (cycle-12 P3: Settings → Web Search blocked
    // the UI on tab construction because the panel's view-builder asked
    // ``apiKey(for:)`` for both Brave and Tavily — each call synchronously
    // crosses the securityd XPC hop, and the first cross-process Keychain
    // access against a ``kSecAttrAccessibleWhenUnlockedThisDeviceOnly``
    // item can surface a system permission modal. The UI now warms the
    // cache off the main actor before reading the cache; these two
    // helpers are the seam.
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
        if let trimmed = keyCache[account]?
            .trimmingCharacters(in: .whitespacesAndNewlines).nonEmpty {
            return .present(trimmed)
        }
        return .absent
    }

    /// Warm ``cachedKeyState(for:)`` for ``provider`` without blocking
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
        let value = await Task.detached(priority: .userInitiated) {
            keychain.read(account: account)
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
        if let value, !value.isEmpty {
            keyCache[account] = value
        }
        probedAccounts.insert(account)
    }

    /// Convenience: warm every keyed provider in parallel. Used by the
    /// Settings → Web Search panel's ``.task`` so the cache is populated
    /// once, off the main actor, before the panel re-renders.
    func prefetchAllAPIKeys() async {
        await withTaskGroup(of: Void.self) { group in
            for provider in WebSearchProvider.allCases where provider.requiresKey {
                group.addTask { await self.prefetchAPIKey(for: provider) }
            }
        }
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
