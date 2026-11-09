import Foundation
import Testing
@testable import Rapid

@Suite("Desktop LAN API")
@MainActor
struct LANAccessTests {
    private func store() -> EmbeddedBearerCredentialStore {
        let defaults = UserDefaults(suiteName: "LANAccessTests.\(UUID().uuidString)")!
        return EmbeddedBearerCredentialStore(defaults: defaults, keychain: InMemoryKeychain())
    }

    @Test("LAN enable refuses unavailable credentials and stays loopback")
    func credentialRequired() async {
        let server = ServerManager(testingState: .idle, bearerCredentialStore: UnavailableLANStore())
        await server.setLANAccessEnabled(true)
        #expect(!server.lanAccessEnabled)
        #expect(server.lanAccessError != nil)
        #expect(server.host == "127.0.0.1")
    }

    @Test("LAN enable persists its opt-in and requires a reusable key")
    func persistentOptIn() async {
        let credentialStore = store()
        let defaults = UserDefaults(suiteName: "LANOptInTests.\(UUID().uuidString)")!
        let server = ServerManager(
            testingState: .idle, sessionDefaults: defaults, bearerCredentialStore: credentialStore
        )
        #expect(!server.lanAccessEnabled)
        server.setEmbeddedBearerLifetime(.perLaunch)
        await server.setLANAccessEnabled(true)
        #expect(server.lanAccessEnabled)
        #expect(server.embeddedBearerLifetime == .explicit)
        guard case .found(let credential) = credentialStore.loadCredential() else {
            Issue.record("LAN enable did not persist its credential")
            return
        }
        #expect(!credential.secret.isEmpty)
        #expect(defaults.bool(forKey: "rapid.lanAccess.enabled.v1") == true)
        let relaunched = ServerManager(
            testingState: .idle, sessionDefaults: defaults, bearerCredentialStore: credentialStore
        )
        #expect(relaunched.lanAccessEnabled)
        server.setEmbeddedBearerLifetime(.perLaunch)
        #expect(server.embeddedBearerLifetime == .explicit)
        await server.setLANAccessEnabled(false)
        #expect(!server.lanAccessEnabled)
        #expect(defaults.bool(forKey: "rapid.lanAccess.enabled.v1") == false)
    }

    @Test("Persistence failure or failed readback cannot fall back to a one-time key")
    func strictMaterial() {
        #expect(LANAccess.credential(store: UnavailableLANStore()) == nil)
        #expect(LANAccess.credential(store: UnavailableLANStore(read: .missing, saveSucceeds: false)) == nil)
        #expect(LANAccess.credential(store: UnavailableLANStore(read: .missing, saveSucceeds: true)) == nil)
        #expect(LANAccess.credential(store: store(), generateSecret: { "" }) == nil)
    }

    @Test("LAN launch binds IPv4 wildcard, authenticates, and opts into the route boundary")
    func launchContract() {
        let lan = ServerManager.serveArguments(alias: "test-model", host: "127.0.0.1", port: 7659, lanInferenceOnly: true)
        let local = ServerManager.serveArguments(alias: "test-model", host: "127.0.0.1", port: 7659)
        #expect(lan[lan.firstIndex(of: "--host")! + 1] == "0.0.0.0")
        #expect(local[local.firstIndex(of: "--host")! + 1] == "127.0.0.1")
        #expect(!lan.contains("--allow-unauthenticated-public"))
        #expect(!local.contains("--allow-unauthenticated-public"))
        let unsafeExtras = ServerManager.serveArguments(
            alias: "test-model", host: "127.0.0.1", port: 7659, lanInferenceOnly: true,
            extraFlags: ["--allow-unauthenticated-public"]
        )
        #expect(!unsafeExtras.contains("--allow-unauthenticated-public"))
        let env = ServerManager.serveEnvironmentAdditions(bearer: "secret", lanInferenceOnly: true, ambient: [:])
        #expect(env["RAPID_MLX_API_KEY"] == "secret")
        #expect(env["RAPID_MLX_LAN_INFERENCE_ONLY"] == "1")
        let localEnv = ServerManager.serveEnvironmentAdditions(
            bearer: "secret", ambient: ["RAPID_MLX_LAN_INFERENCE_ONLY": "1"]
        )
        #expect(localEnv["RAPID_MLX_LAN_INFERENCE_ONLY"] == nil)
    }

    @Test("LAN URLs use only non-loopback IPv4 addresses and valid ports")
    func addressFormatting() {
        #expect(LANAccess.baseURL(address: "192.168.1.12", port: 7659) == "http://192.168.1.12:7659/v1")
        #expect(LANAccess.baseURL(address: "10.0.0.2", port: 8000) == "http://10.0.0.2:8000/v1")
        for address in ["127.0.0.1", "127.1.2.3", "0.0.0.0", "::1", "fe80::1", "localhost", "invalid"] {
            #expect(LANAccess.baseURL(address: address, port: 7659) == nil)
        }
        #expect(LANAccess.baseURL(address: "10.0.0.2", port: 0) == nil)
        #expect(LANAccess.baseURL(address: "10.0.0.2", port: 65536) == nil)
    }
}

private struct UnavailableLANStore: EmbeddedBearerCredentialStoring {
    var read: EmbeddedBearerStorageResult = .unavailable
    var saveSucceeds = false
    func loadCredential() -> EmbeddedBearerStorageResult { read }
    func hasPersistedCredential() -> Bool { false }
    func save(_ credential: EmbeddedBearerCredential) -> Bool { saveSucceeds }
    func clear() -> Bool { true }
}
