import CryptoKit
import Foundation
import Testing
@testable import Rapid

// The App Attest orchestration (challenge → attest-once → assertion) driven
// through injected fakes. The DCAppAttestService calls themselves only run on
// a real device in a signed app; these pin everything around them: the
// register-once flow, the Keychain caching, the exact bytes the assertion
// signs, and graceful failure. Plus the CLI-argument relay.

private struct FakeHTTP: AttestHTTP {
    let box: Box
    final class Box: @unchecked Sendable {
        var challenges: [String]
        var registerCalls: [(url: URL, attestation: Data, challenge: String)] = []
        var registerError: Error?
        init(challenges: [String]) { self.challenges = challenges }
    }
    func fetchChallenge(_ url: URL) async throws -> String {
        if box.challenges.isEmpty { throw AppAttestError.badChallengeResponse }
        return box.challenges.removeFirst()
    }
    func register(_ url: URL, attestation: Data, challenge: String) async throws {
        if let error = box.registerError { throw error }
        box.registerCalls.append((url, attestation, challenge))
    }
}

private struct FakeService: AttestService {
    let box: Box
    final class Box: @unchecked Sendable {
        var supported: Bool
        var generatedKeys = 0
        var attestCalls: [(keyID: String, clientDataHash: Data)] = []
        var assertionCalls: [(keyID: String, clientDataHash: Data)] = []
        init(supported: Bool) { self.supported = supported }
    }
    var isSupported: Bool { box.supported }
    func generateKey() async throws -> String {
        box.generatedKeys += 1
        return "key-\(box.generatedKeys)"
    }
    func attestKey(_ keyID: String, clientDataHash: Data) async throws -> Data {
        box.attestCalls.append((keyID, clientDataHash))
        return Data("attestation-for-\(keyID)".utf8)
    }
    func generateAssertion(_ keyID: String, clientDataHash: Data) async throws -> Data {
        box.assertionCalls.append((keyID, clientDataHash))
        return Data("assertion".utf8)
    }
}

private final class InMemoryKeychain: KeychainStoring, @unchecked Sendable {
    private let lock = NSLock()
    private var store: [String: String] = [:]
    func read(account: String) -> String? { lock.lock(); defer { lock.unlock() }; return store[account] }
    @discardableResult func write(account: String, secret: String) -> Bool {
        lock.lock(); defer { lock.unlock() }; store[account] = secret; return true
    }
    @discardableResult func delete(account: String) -> Bool {
        lock.lock(); defer { lock.unlock() }; store[account] = nil; return true
    }
}

private let target = URL(string: "https://rapidmlx.com/api/benchmarks/atomic")!

@Suite("Community Benchmark App Attest")
struct CommunityBenchmarkAttestTests {
    @Test("First upload mints + attests a key, then signs the body with a fresh challenge")
    func firstUploadRegistersThenAsserts() async throws {
        let http = FakeHTTP(box: .init(challenges: ["chal-register", "chal-assert"]))
        let service = FakeService(box: .init(supported: true))
        let keychain = InMemoryKeychain()
        let client = AppAttestClient(service: service, http: http, keychain: keychain)

        let body = Data("the-exact-wire-body".utf8)
        let material = try #require(await client.attestMaterial(forBody: body, target: target))

        // Registered once, key cached.
        #expect(service.box.generatedKeys == 1)
        #expect(service.box.attestCalls.count == 1)
        #expect(http.box.registerCalls.count == 1)
        #expect(keychain.read(account: "Rapid.communityBenchmark.attestKeyId") == "key-1")

        // attestKey binds SHA256(register challenge); the derived URLs are right.
        #expect(service.box.attestCalls[0].clientDataHash == Data(SHA256.hash(data: Data("chal-register".utf8))))
        #expect(http.box.registerCalls[0].url.absoluteString == target.absoluteString + "/attest")

        // The assertion signs SHA256(assertion-challenge ‖ body), and the
        // material carries that second challenge — not the registration one.
        #expect(material.keyID == "key-1")
        #expect(material.challenge == "chal-assert")
        var signed = Data("chal-assert".utf8); signed.append(body)
        #expect(service.box.assertionCalls[0].clientDataHash == Data(SHA256.hash(data: signed)))
        #expect(material.assertionBase64 == Data("assertion".utf8).base64EncodedString())
    }

    @Test("A cached key skips registration and only asserts")
    func cachedKeySkipsRegistration() async throws {
        let http = FakeHTTP(box: .init(challenges: ["chal-assert"]))
        let service = FakeService(box: .init(supported: true))
        let keychain = InMemoryKeychain()
        keychain.write(account: "Rapid.communityBenchmark.attestKeyId", secret: "existing-key")
        let client = AppAttestClient(service: service, http: http, keychain: keychain)

        let material = try #require(await client.attestMaterial(forBody: Data("b".utf8), target: target))
        #expect(service.box.generatedKeys == 0)
        #expect(service.box.attestCalls.isEmpty)
        #expect(http.box.registerCalls.isEmpty)
        #expect(material.keyID == "existing-key")
    }

    @Test("An unsupported device yields no material (upload proceeds un-attested)")
    func unsupportedDeviceReturnsNil() async {
        let http = FakeHTTP(box: .init(challenges: ["c"]))
        let service = FakeService(box: .init(supported: false))
        let client = AppAttestClient(service: service, http: http, keychain: InMemoryKeychain())
        #expect(await client.attestMaterial(forBody: Data(), target: target) == nil)
    }

    @Test("A rejected registration yields nil and caches no key")
    func rejectedRegistrationDoesNotCache() async {
        let http = FakeHTTP(box: .init(challenges: ["chal-register", "chal-assert"]))
        http.box.registerError = AppAttestError.registrationRejected(status: 400, body: "nope")
        let service = FakeService(box: .init(supported: true))
        let keychain = InMemoryKeychain()
        let client = AppAttestClient(service: service, http: http, keychain: keychain)

        #expect(await client.attestMaterial(forBody: Data("b".utf8), target: target) == nil)
        #expect(keychain.read(account: "Rapid.communityBenchmark.attestKeyId") == nil)
    }

    @Test("Share arguments relay the attest triple only when present")
    func shareArgumentsRelayAttest() {
        let base = CommunityBenchmarkCommand.benchmarkShareArguments(
            runID: "r", installID: "i", payloadDigest: "pd", bodyDigest: "bd", target: "t"
        )
        #expect(!base.contains("--attest-key-id"))
        #expect(base.last == "--json")

        let attested = CommunityBenchmarkCommand.benchmarkShareArguments(
            runID: "r", installID: "i", payloadDigest: "pd", bodyDigest: "bd", target: "t",
            attest: AttestMaterial(keyID: "kid", assertionBase64: "sig", challenge: "chal")
        )
        #expect(attested.contains("--attest-key-id"))
        for (flag, value) in [("--attest-key-id", "kid"), ("--attest-assertion", "sig"), ("--attest-challenge", "chal")] {
            let index = try! #require(attested.firstIndex(of: flag))
            #expect(attested[index + 1] == value)
        }
        #expect(attested.last == "--json")
    }
}
