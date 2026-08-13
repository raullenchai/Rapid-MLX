import Foundation
import Testing
@testable import Rapid

@Suite("Sparkle update configuration")
struct SparkleUpdateControllerTests {
    private let publicKey = Data(repeating: 7, count: 32).base64EncodedString()

    @Test("HTTPS feed plus 32-byte EdDSA public key enables Sparkle")
    func validConfiguration() {
        #expect(SparkleUpdateController.hasValidConfiguration([
            "SUFeedURL": "https://dl.rapidmlx.com/appcast.xml",
            "SUPublicEDKey": publicKey,
        ]))
    }

    @Test("local build without injected public key stays on legacy updater")
    func missingPublicKey() {
        #expect(!SparkleUpdateController.hasValidConfiguration([
            "SUFeedURL": "https://dl.rapidmlx.com/appcast.xml",
        ]))
    }

    @Test("configuration rejects insecure feeds and malformed public keys")
    func invalidConfiguration() {
        #expect(!SparkleUpdateController.hasValidConfiguration([
            "SUFeedURL": "http://dl.rapidmlx.com/appcast.xml",
            "SUPublicEDKey": publicKey,
        ]))
        #expect(!SparkleUpdateController.hasValidConfiguration([
            "SUFeedURL": "https://dl.rapidmlx.com/appcast.xml",
            "SUPublicEDKey": Data(repeating: 7, count: 31).base64EncodedString(),
        ]))
    }

    @MainActor
    @Test("existing update-check opt-out also disables Sparkle")
    func updateCheckOptOut() {
        let controller = SparkleUpdateController(
            infoDictionary: [
                "SUFeedURL": "https://dl.rapidmlx.com/appcast.xml",
                "SUPublicEDKey": publicKey,
            ],
            checksEnabled: false
        )
        #expect(!controller.isEnabled)
    }
}
