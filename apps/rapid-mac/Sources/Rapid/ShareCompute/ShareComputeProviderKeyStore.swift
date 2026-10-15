import Foundation

/// Account credential for registering this Mac. It is never serialized into
/// app preferences, process arguments, status files, or contribution receipts.
struct ShareComputeProviderKey: Equatable, Sendable {
    let rawValue: String

    static func validate(_ raw: String) -> ShareComputeProviderKey? {
        let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard value.hasPrefix("qsppk-"), value.utf8.count >= 16,
              value.utf8.count <= 4_095,
              !value.unicodeScalars.contains(where: {
                  CharacterSet.controlCharacters.contains($0)
                      || CharacterSet.whitespacesAndNewlines.contains($0)
              }) else { return nil }
        return Self(rawValue: value)
    }

    var redactedLabel: String { "qsppk-…\(rawValue.suffix(4))" }
}

enum ShareComputeProviderKeyResult: Equatable, Sendable {
    case found(ShareComputeProviderKey)
    case missing
    case unavailable
    case corrupted
}

/// This credential has its own Keychain account, separate from ledger access.
struct ShareComputeProviderKeyStore: Sendable {
    static let account = "rapid.quicksilver.provider-key.v1"

    private let keychain: any KeychainStoring

    init(keychain: any KeychainStoring = SystemKeychain()) {
        self.keychain = keychain
    }

    func load() -> ShareComputeProviderKeyResult {
        switch keychain.readWithoutUserInteraction(account: Self.account) {
        case .missing: return .missing
        case .unavailable: return .unavailable
        case .found(let raw):
            if raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return .missing }
            guard let key = ShareComputeProviderKey.validate(raw) else { return .corrupted }
            return .found(key)
        }
    }

    func save(_ key: ShareComputeProviderKey) -> Bool {
        keychain.write(account: Self.account, secret: key.rawValue)
    }

    func remove() -> Bool { keychain.delete(account: Self.account) }
}
