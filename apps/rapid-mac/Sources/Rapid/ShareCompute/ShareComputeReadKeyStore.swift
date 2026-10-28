import Foundation

// MARK: - Validation

/// Why a `qsprk-` value failed format validation.
///
/// ## Legacy — not reachable from the UI
///
/// Rapid used to have a second, read-only `qsprk-` key the Credits tab asked
/// for separately. The owner decision was ONE key: the user-facing credential
/// is now exclusively the `qsppk-` provider key, entered once in
/// Settings → Experimental → Share Compute, and nothing in the app prompts
/// for a `qsprk-` value any more.
///
/// This type, ``ShareComputeReadKey``, and ``ShareComputeReadKeyStore`` stay
/// for ONE reason: an account that already had a `qsprk-` key saved before
/// this change must keep seeing its ledger without being asked to paste
/// anything — see ``ShareComputeLedgerCoordinator``'s provider-key-first,
/// legacy-key-fallback resolution. Nothing writes a NEW `qsprk-` key through
/// this store any more.
enum ShareComputeReadKeyRejection: Error, Equatable, Sendable {
    case empty
    /// Not a `qsprk-` key. Almost always a paste of the wrong credential —
    /// the `qsppk-` provider key, which belongs in its separate
    /// provider credential store.
    case wrongPrefix
    case tooShort
    case tooLong
    /// Newlines, tabs, or other control characters. A newline is the classic
    /// copy-from-a-terminal artefact and would corrupt the header it lands in.
    case containsControlCharacters
    /// The key was valid, but this Mac could not save it.
    case keychainUnavailable
}

/// A validated legacy `qsprk-` key. See the type-level note above: this is
/// migration-only machinery, never shown to the user.
///
/// Deliberately NOT `Codable`, `CustomStringConvertible` or `Encodable`: the
/// compiler should refuse to help anyone serialise it into JSON, a log line,
/// or a screenshot fixture.
struct ShareComputeReadKey: Equatable, Sendable {
    /// The raw secret. Read exactly twice in the app: once to write to the
    /// Keychain, once to build the ledger `Authorization` header.
    let rawValue: String

    static let prefix = "qsprk-"
    /// Prefix plus a short token. Anything below this is a truncated paste.
    static let minimumLength = 16
    /// Generous ceiling. A bound exists so a pathological paste cannot be
    /// handed to Security.framework or a header builder.
    static let maximumLength = 256

    /// Validates and wraps. Trims surrounding whitespace only — interior
    /// control characters are a rejection, not something to silently strip.
    static func validate(_ raw: String) -> Result<ShareComputeReadKey, ShareComputeReadKeyRejection> {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return .failure(.empty) }
        // Prefix is checked before length so pasting a provider key gets the
        // message that names the actual mistake.
        guard trimmed.hasPrefix(prefix) else { return .failure(.wrongPrefix) }
        if trimmed.unicodeScalars.contains(where: {
            CharacterSet.controlCharacters.contains($0) || CharacterSet.whitespacesAndNewlines.contains($0)
        }) {
            return .failure(.containsControlCharacters)
        }
        if trimmed.count < minimumLength { return .failure(.tooShort) }
        if trimmed.count > maximumLength { return .failure(.tooLong) }
        return .success(ShareComputeReadKey(rawValue: trimmed))
    }

    /// A non-secret label for the UI: `qsprk-…a8c1`. Never the whole key.
    ///
    /// Shown so a user with several accounts can tell which key is saved
    /// without Rapid ever re-displaying the secret.
    var redactedLabel: String {
        let tail = rawValue.suffix(4)
        return "\(Self.prefix)…\(tail)"
    }
}

// MARK: - Store

enum ShareComputeReadKeyStoreResult: Equatable, Sendable {
    case found(ShareComputeReadKey)
    case missing
    /// The Keychain answered, but not usably — locked, or an ACL this build
    /// cannot satisfy. Distinct from ``missing`` because the UI must not offer
    /// "paste a key" as the fix for a Keychain that is simply unavailable.
    case unavailable
    /// An item exists but no longer validates (truncated write, or a key
    /// format QuickSilver has since retired). Treated as missing by callers
    /// that need a key, but reported distinctly so the UI can say so.
    case corrupted
}

/// Keychain storage for the LEGACY QuickSilver ledger read key.
///
/// ## Why this is its own type
///
/// Nothing writes a new key through this store any more — see the type-level
/// note on ``ShareComputeReadKeyRejection``. It is read by
/// ``ShareComputeLedgerCoordinator`` only as a fallback, for an account that
/// saved a `qsprk-` key before Rapid moved to one key. Provider credentials
/// use a separate Keychain account/store and travel to registration over
/// stdin; they never enter ledger requests, UserDefaults, argv, or the
/// environment.
///
/// ## Account namespacing
///
/// The account name is deliberately unlike `rapid.web-search.*` and
/// `embedded-engine.bearer.v1`: a collision would make one feature's key
/// readable as another's, and the version suffix leaves room to rotate the
/// format without reading a stale item as a current one.
struct ShareComputeReadKeyStore: Sendable {
    /// Unique across every Keychain account this app writes.
    static let account = "rapid.quicksilver.ledger-read-key.v1"

    private let keychain: any KeychainStoring

    /// Uses the system Keychain. Tests inject ``InMemoryKeychain``.
    init(keychain: any KeychainStoring = SystemKeychain()) {
        self.keychain = keychain
    }

    func load() -> ShareComputeReadKeyStoreResult {
        switch keychain.readWithoutUserInteraction(account: Self.account) {
        case .missing:
            return .missing
        case .unavailable:
            return .unavailable
        case .found(let raw):
            // A tombstoned item (see `SystemKeychain.delete`) is an empty
            // string, which reads as missing rather than corrupted.
            if raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return .missing }
            switch ShareComputeReadKey.validate(raw) {
            case .success(let key): return .found(key)
            case .failure: return .corrupted
            }
        }
    }

    /// Writes a validated key, replacing any existing one.
    ///
    /// Takes ``ShareComputeReadKey`` rather than `String` so there is no way
    /// to reach the Keychain with an unvalidated value — including a `qsppk-`
    /// provider key, which `validate` rejects on its prefix.
    @discardableResult
    func save(_ key: ShareComputeReadKey) -> Bool {
        keychain.write(account: Self.account, secret: key.rawValue)
    }

    /// Removes the stored key. Used by "Remove key" and by the revoked-key
    /// recovery path, so a rejected credential does not sit on disk waiting to
    /// 401 again on the next launch.
    @discardableResult
    func remove() -> Bool {
        keychain.delete(account: Self.account)
    }
}
