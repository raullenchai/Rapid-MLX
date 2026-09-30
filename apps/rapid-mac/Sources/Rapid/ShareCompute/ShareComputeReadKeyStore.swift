import Foundation

// MARK: - Validation

/// Why a pasted QuickSilver read key was refused.
///
/// Validation happens BEFORE the key reaches the Keychain or a request, so a
/// mistyped or mis-pasted value fails locally instead of burning one of the
/// ledger endpoint's 120 requests/hour and coming back as an opaque 401.
enum ShareComputeReadKeyRejection: Error, Equatable, Sendable {
    case empty
    /// Not a `qsprk-` key. Almost always a paste of the wrong credential —
    /// most dangerously the `qsppk-` provider key, which Rapid must never
    /// persist.
    case wrongPrefix
    case tooShort
    case tooLong
    /// Newlines, tabs, or other control characters. A newline is the classic
    /// copy-from-a-terminal artefact and would corrupt the header it lands in.
    case containsControlCharacters
    /// The key was valid, but this Mac could not save it.
    case keychainUnavailable

    var message: String {
        switch self {
        case .empty:
            return String(localized: "Paste your QuickSilver read key to continue.")
        case .wrongPrefix:
            return String(localized: "That isn’t a read key. Read keys start with qsprk-. A provider key (qsppk-) registers nodes and is never saved by Rapid.")
        case .tooShort:
            return String(localized: "That read key looks incomplete.")
        case .tooLong:
            return String(localized: "That read key is longer than QuickSilver issues.")
        case .containsControlCharacters:
            return String(localized: "That read key contains a line break or control character. Paste it as a single line.")
        case .keychainUnavailable:
            return String(localized: "This Mac couldn't save the read key. Unlock your Keychain and try again.")
        }
    }
}

/// A validated `qsprk-` read key.
///
/// The only type the store and the client will accept, so "did anyone check
/// this?" has one answer. Deliberately NOT `Codable`, `CustomStringConvertible`
/// or `Encodable`: the compiler should refuse to help anyone serialise it into
/// JSON, a log line, or a screenshot fixture.
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

/// Keychain storage for the QuickSilver ledger read key.
///
/// ## Why this is its own type
///
/// `qsprk-` is the ONLY QuickSilver credential Rapid may persist. It is
/// account-scoped and read-only: it can list ledger windows and nothing else —
/// it cannot register a node, mint a share key, or spend credit. That is what
/// makes at-rest storage acceptable here and unacceptable for `qsppk-`.
///
/// The provider key (`qsppk-`) keeps its existing contract untouched: typed or
/// piped into the connection sheet, handed to the share subprocess over stdin,
/// cleared from memory immediately after. It is never written here, never in
/// UserDefaults, never in argv, and never in the environment. Giving the read
/// key its own store — rather than a general "QuickSilver credentials" bag —
/// is what keeps that boundary impossible to blur by accident.
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

    /// Defaults to the system Keychain, which applies
    /// `kSecAttrAccessibleWhenUnlockedThisDeviceOnly` — readable only while the
    /// screen is unlocked, and never migrated off this Mac by Keychain sync or
    /// a Time Machine restore. Tests inject ``InMemoryKeychain``.
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
