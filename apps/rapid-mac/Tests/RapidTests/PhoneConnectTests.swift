import Foundation
import Testing
@testable import Rapid

@Suite("Phone connection")
struct PhoneConnectTests {
    @Test func fragmentKey() throws {
        let url = try #require(PhoneConnect.url(address: "192.168.1.5", port: 7659, key: "secret+&/#?=", lanEnabled: true))
        let parts = try #require(URLComponents(url: url, resolvingAgainstBaseURL: false))
        #expect(parts.path == "/connect")
        #expect(parts.query == nil)
        #expect(parts.percentEncodedFragment?.contains("%2B") == true)
        var fragment = URLComponents()
        fragment.percentEncodedQuery = parts.percentEncodedFragment
        #expect(fragment.queryItems?.first?.value == "secret+&/#?=")
        #expect(url.absoluteString.hasPrefix("http://192.168.1.5:7659/connect#k="))
    }

    @Test func lanOffAndInvalidState() {
        #expect(PhoneConnect.url(address: "10.0.0.2", port: 8000, key: "key", lanEnabled: false) == nil)
        #expect(PhoneConnect.url(address: "10.0.0.2", port: 8000, key: "", lanEnabled: true) == nil)
        for address in ["127.0.0.1", "169.254.1.2", "0.0.0.0", "::1", "invalid", "224.0.0.1"] {
            #expect(PhoneConnect.url(address: address, port: 8000, key: "key", lanEnabled: true) == nil)
        }
        #expect(PhoneConnect.url(address: "10.0.0.2", port: 0, key: "key", lanEnabled: true) == nil)
    }

    @Test func selection() {
        #expect(PhoneConnect.orderedAddresses(["10.0.0.2", "192.168.1.5", "127.0.0.1", "169.254.1.2", "10.0.0.2"], primary: "192.168.1.5") == ["192.168.1.5", "10.0.0.2"])
        #expect(PhoneConnect.orderedAddresses(["192.168.1.5", "10.0.0.2"], primary: nil) == ["10.0.0.2", "192.168.1.5"])
    }
}
