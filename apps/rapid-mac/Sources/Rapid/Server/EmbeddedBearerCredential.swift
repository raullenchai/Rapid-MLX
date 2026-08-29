import Foundation

/// How long a Desktop-managed embedded-engine bearer remains usable.
enum EmbeddedBearerLifetime: String, CaseIterable, Equatable, Sendable {
    case perLaunch
    case daily
    case explicit

    var storageKey: String { rawValue }

    var displayName: String {
        switch self {
        case .perLaunch: return "Every start"
        case .daily: return "Daily"
        case .explicit: return "Until I rotate"
        }
    }

    var summary: String {
        switch self {
        case .perLaunch:
            return "Generate a one-time key for every model start. Nothing is stored in the Keychain."
        case .daily:
            return "Keep one Keychain-backed key for 24 hours across model starts and app restarts."
        case .explicit:
            return "Keep one Keychain-backed key until you rotate it or change this setting."
        }
    }
}

/// A non-secret policy record. The bearer secret itself never enters
/// UserDefaults.
struct EmbeddedBearerCredentialMetadata: Equatable, Codable, Sendable {
    let lifetime: String
    let rotatedAt: Date
}

struct EmbeddedBearerCredential: Equatable, Sendable {
    let secret: String
    let rotatedAt: Date
    let lifetime: EmbeddedBearerLifetime
}

enum EmbeddedBearerStorageResult: Equatable, Sendable {
    case found(EmbeddedBearerCredential)
    case missing
    case corrupted
    case unavailable
}

enum EmbeddedBearerStorageIssue: Equatable, Sendable {
    case generationFailed
    case missingSecret
    case corruptedCredential
    case unavailableKeychain
    case writeFailed
}

/// One start-time materialization of an embedded API credential. A persisted
/// secret is already in the Keychain; one-time material is never written to
/// any local store.
struct EmbeddedBearerMaterial: Equatable, Sendable {
    let secret: String
    let rotatedAt: Date?
    let isPersisted: Bool
    let issue: EmbeddedBearerStorageIssue?
}

/// Observable, secret-free credential state for Settings.
enum EmbeddedBearerStatus: Equatable, Sendable {
    case notMaterialized
    case materialized(
        rotatedAt: Date?,
        isPersisted: Bool,
        issue: EmbeddedBearerStorageIssue?
    )
}

protocol EmbeddedBearerCredentialStoring: Sendable {
    func loadCredential() -> EmbeddedBearerStorageResult
    func hasPersistedCredential() -> Bool
    @discardableResult func save(_ credential: EmbeddedBearerCredential) -> Bool
    @discardableResult func clear() -> Bool
}

/// ``UserDefaults`` is thread-safe for these small metadata reads/writes but
/// does not expose Sendable conformance in this SDK.
struct EmbeddedBearerCredentialStore: EmbeddedBearerCredentialStoring, @unchecked Sendable {
    private static let account = "embedded-engine.bearer.v1"
    private static let metadataKey = "rapid.embeddedBearer.credential.v1"
    private static let hasPersistedKey = "rapid.embeddedBearer.hasPersistedCredential.v1"
    private static let secretLength = 64

    private let defaults: UserDefaults
    private let keychain: KeychainStoring

    init(defaults: UserDefaults, keychain: KeychainStoring) {
        self.defaults = defaults
        self.keychain = keychain
    }

    func loadCredential() -> EmbeddedBearerStorageResult {
        let keychainResult = keychain.readWithoutUserInteraction(account: Self.account)
        switch keychainResult {
        case .unavailable:
            return .unavailable
        case .missing:
            return .missing
        case .found(let secret):
            guard Self.isValidSecret(secret) else {
                clear()
                return .corrupted
            }
        }

        guard let data = defaults.data(forKey: Self.metadataKey),
              let metadata = try? JSONDecoder().decode(
                  EmbeddedBearerCredentialMetadata.self,
                  from: data
              ),
              EmbeddedBearerLifetime(rawValue: metadata.lifetime) != nil,
              metadata.rotatedAt <= Date().addingTimeInterval(60) else {
            clear()
            return .corrupted
        }

        guard case .found(let secret) = keychainResult else {
            return .missing
        }
        return .found(
            EmbeddedBearerCredential(
                secret: secret,
                rotatedAt: metadata.rotatedAt,
                lifetime: EmbeddedBearerLifetime(rawValue: metadata.lifetime)!
            )
        )
    }

    func hasPersistedCredential() -> Bool {
        defaults.bool(forKey: Self.hasPersistedKey)
    }

    @discardableResult
    func save(_ credential: EmbeddedBearerCredential) -> Bool {
        guard Self.isValidSecret(credential.secret) else { return false }
        guard keychain.write(account: Self.account, secret: credential.secret) else {
            return false
        }

        let metadata = EmbeddedBearerCredentialMetadata(
            lifetime: credential.lifetime.storageKey,
            rotatedAt: credential.rotatedAt
        )
        guard let data = try? JSONEncoder().encode(metadata) else {
            clear()
            return false
        }
        defaults.set(data, forKey: Self.metadataKey)
        defaults.set(true, forKey: Self.hasPersistedKey)
        return true
    }

    @discardableResult
    func clear() -> Bool {
        defaults.removeObject(forKey: Self.metadataKey)
        defaults.removeObject(forKey: Self.hasPersistedKey)
        return keychain.delete(account: Self.account)
    }

    static func isValidSecret(_ secret: String) -> Bool {
        secret.count == secretLength
            && secret.allSatisfy { character in
                character.isHexDigit && character.isASCII
            }
    }
}

enum EmbeddedBearerMaterialResolver {
    static func resolve(
        lifetime: EmbeddedBearerLifetime,
        store: EmbeddedBearerCredentialStoring,
        now: Date,
        generateSecret: () -> String?
    ) -> EmbeddedBearerMaterial {
        let generatedSecret = generateSecret()
        guard let generatedSecret else {
            return EmbeddedBearerMaterial(
                secret: "",
                rotatedAt: nil,
                isPersisted: false,
                issue: .generationFailed
            )
        }

        switch lifetime {
        case .perLaunch:
            store.clear()
            return EmbeddedBearerMaterial(
                secret: generatedSecret,
                rotatedAt: now,
                isPersisted: false,
                issue: nil
            )
        case .daily:
            switch store.loadCredential() {
            case .missing where store.hasPersistedCredential():
                return EmbeddedBearerMaterial(
                    secret: generatedSecret,
                    rotatedAt: now,
                    isPersisted: false,
                    issue: .missingSecret
                )
            case .found(let credential) where credential.lifetime == .daily:
                if now.timeIntervalSince(credential.rotatedAt) < 24 * 60 * 60 {
                    return EmbeddedBearerMaterial(
                        secret: credential.secret,
                        rotatedAt: credential.rotatedAt,
                        isPersisted: true,
                        issue: nil
                    )
                }
            case .corrupted:
                return EmbeddedBearerMaterial(
                    secret: generatedSecret,
                    rotatedAt: now,
                    isPersisted: false,
                    issue: .corruptedCredential
                )
            case .unavailable:
                return EmbeddedBearerMaterial(
                    secret: generatedSecret,
                    rotatedAt: now,
                    isPersisted: false,
                    issue: .unavailableKeychain
                )
            default:
                break
            }
        case .explicit:
            switch store.loadCredential() {
            case .missing where store.hasPersistedCredential():
                return EmbeddedBearerMaterial(
                    secret: generatedSecret,
                    rotatedAt: now,
                    isPersisted: false,
                    issue: .missingSecret
                )
            case .found(let credential) where credential.lifetime == .explicit:
                return EmbeddedBearerMaterial(
                    secret: credential.secret,
                    rotatedAt: credential.rotatedAt,
                    isPersisted: true,
                    issue: nil
                )
            case .corrupted:
                return EmbeddedBearerMaterial(
                    secret: generatedSecret,
                    rotatedAt: now,
                    isPersisted: false,
                    issue: .corruptedCredential
                )
            case .unavailable:
                return EmbeddedBearerMaterial(
                    secret: generatedSecret,
                    rotatedAt: now,
                    isPersisted: false,
                    issue: .unavailableKeychain
                )
            default:
                break
            }
        }

        let credential = EmbeddedBearerCredential(
            secret: generatedSecret,
            rotatedAt: now,
            lifetime: lifetime
        )
        guard store.save(credential) else {
            return EmbeddedBearerMaterial(
                secret: generatedSecret,
                rotatedAt: now,
                isPersisted: false,
                issue: .writeFailed
            )
        }
        return EmbeddedBearerMaterial(
            secret: credential.secret,
            rotatedAt: credential.rotatedAt,
            isPersisted: true,
            issue: nil
        )
    }
}
