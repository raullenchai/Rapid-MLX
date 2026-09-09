import CryptoKit
import DeviceCheck
import Foundation

// App Attest for the Community Benchmark ranked leaderboard.
//
// The leaderboard is a contest (race to #1) with anonymous, account-less
// uploads, so a well-formed fake could take the top spot. App Attest proves a
// submission comes from a genuine, unmodified instance of THIS notarized app
// on real Apple hardware — with no user prompt and no account. DCAppAttestService
// only works inside the signed app, so the assertion is produced here and
// relayed by the bundled engine as X-Rapid-Attest-* headers (see
// `benchmark share --attest-*`). The server verifies against the pinned Apple
// App Attest Root CA; design + server live in rapidmlx.com
// (docs/benchmark-authenticity.md).
//
// Attestation (device-key registration) happens once per install; every later
// upload is just a cheap assertion. All of it is silent.

/// The material the engine relays as request headers on the atomic upload.
struct AttestMaterial: Equatable, Sendable {
    let keyID: String
    let assertionBase64: String
    let challenge: String
}

// MARK: - Seams (so the orchestration is testable without a device/network)

/// DCAppAttestService, behind a protocol. The real calls can only run on a
/// genuine Apple device in a signed app; tests inject a fake.
protocol AttestService: Sendable {
    var isSupported: Bool { get }
    func generateKey() async throws -> String
    func attestKey(_ keyID: String, clientDataHash: Data) async throws -> Data
    func generateAssertion(_ keyID: String, clientDataHash: Data) async throws -> Data
}

/// The two server calls App Attest needs beyond the (engine-run) upload.
protocol AttestHTTP: Sendable {
    func fetchChallenge(_ url: URL) async throws -> String
    func register(_ url: URL, attestation: Data, challenge: String) async throws
}

enum AppAttestError: Error, Equatable {
    case unsupported
    case badChallengeResponse
    case registrationRejected(status: Int, body: String)
}

// MARK: - Client

/// Produces `AttestMaterial` for one upload, or `nil` when this device/build
/// cannot attest (old build, unsupported device, or a transient failure) — in
/// which case the caller uploads un-attested, which simply will not rank.
final class AppAttestClient: Sendable {
    private let service: AttestService
    private let http: AttestHTTP
    private let keychain: any KeychainStoring
    private let keyAccount: String

    init(
        service: AttestService = DeviceCheckAttestService(),
        http: AttestHTTP = URLSessionAttestHTTP(),
        keychain: any KeychainStoring = SystemKeychain(),
        keyAccount: String = "Rapid.communityBenchmark.attestKeyId"
    ) {
        self.service = service
        self.http = http
        self.keychain = keychain
        self.keyAccount = keyAccount
    }

    /// `body` is the exact wire bytes the engine will POST (the preview's
    /// `payload_json`); the assertion signs SHA256(challenge ‖ body) so it
    /// covers precisely what is sent. `target` is the atomic upload URL;
    /// `/challenge` and `/attest` are derived from it.
    func attestMaterial(forBody body: Data, target: URL) async -> AttestMaterial? {
        do {
            return try await make(forBody: body, target: target)
        } catch {
            // Fail soft: an un-attested upload is better than a failed one.
            // (When the server gate is on it will 426; the caller surfaces that.)
            return nil
        }
    }

    private func make(forBody body: Data, target: URL) async throws -> AttestMaterial {
        guard service.isSupported else { throw AppAttestError.unsupported }
        let challengeURL = target.appendingPathComponent("challenge")
        let attestURL = target.appendingPathComponent("attest")

        let keyID = try await ensureRegisteredKey(challengeURL: challengeURL, attestURL: attestURL)

        // Fresh challenge per assertion; sign SHA256(challenge ‖ body).
        let challenge = try await http.fetchChallenge(challengeURL)
        var signed = Data(challenge.utf8)
        signed.append(body)
        let clientDataHash = Data(SHA256.hash(data: signed))
        let assertion = try await service.generateAssertion(keyID, clientDataHash: clientDataHash)
        return AttestMaterial(
            keyID: keyID,
            assertionBase64: assertion.base64EncodedString(),
            challenge: challenge
        )
    }

    /// The device key is minted + attested once and cached in the Keychain.
    private func ensureRegisteredKey(challengeURL: URL, attestURL: URL) async throws -> String {
        if let existing = keychain.read(account: keyAccount), !existing.isEmpty {
            return existing
        }
        let challenge = try await http.fetchChallenge(challengeURL)
        let keyID = try await service.generateKey()
        // attestKey binds the key to SHA256(challenge), matching the server.
        let attestation = try await service.attestKey(
            keyID, clientDataHash: Data(SHA256.hash(data: Data(challenge.utf8)))
        )
        try await http.register(attestURL, attestation: attestation, challenge: challenge)
        // Only persist after the server accepts the registration; a failed
        // attest leaves no key, so the next attempt starts clean.
        keychain.write(account: keyAccount, secret: keyID)
        return keyID
    }
}

// MARK: - Real implementations

struct DeviceCheckAttestService: AttestService {
    private var shared: DCAppAttestService { .shared }

    var isSupported: Bool { DCAppAttestService.shared.isSupported }

    func generateKey() async throws -> String {
        try await withCheckedThrowingContinuation { continuation in
            shared.generateKey { keyID, error in
                if let keyID { continuation.resume(returning: keyID) }
                else { continuation.resume(throwing: error ?? AppAttestError.unsupported) }
            }
        }
    }

    func attestKey(_ keyID: String, clientDataHash: Data) async throws -> Data {
        try await withCheckedThrowingContinuation { continuation in
            shared.attestKey(keyID, clientDataHash: clientDataHash) { attestation, error in
                if let attestation { continuation.resume(returning: attestation) }
                else { continuation.resume(throwing: error ?? AppAttestError.unsupported) }
            }
        }
    }

    func generateAssertion(_ keyID: String, clientDataHash: Data) async throws -> Data {
        try await withCheckedThrowingContinuation { continuation in
            shared.generateAssertion(keyID, clientDataHash: clientDataHash) { assertion, error in
                if let assertion { continuation.resume(returning: assertion) }
                else { continuation.resume(throwing: error ?? AppAttestError.unsupported) }
            }
        }
    }
}

struct URLSessionAttestHTTP: AttestHTTP {
    private let session: URLSession
    init(session: URLSession = .shared) { self.session = session }

    func fetchChallenge(_ url: URL) async throws -> String {
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        let (data, _) = try await session.data(for: request)
        guard
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let challenge = object["challenge"] as? String,
            !challenge.isEmpty
        else { throw AppAttestError.badChallengeResponse }
        return challenge
    }

    func register(_ url: URL, attestation: Data, challenge: String) async throws {
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "attestation": attestation.base64EncodedString(),
            "challenge": challenge,
        ])
        let (data, response) = try await session.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200...299).contains(status) else {
            throw AppAttestError.registrationRejected(
                status: status,
                body: String(data: data, encoding: .utf8) ?? ""
            )
        }
    }
}
