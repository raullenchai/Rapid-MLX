import Foundation
import Testing
@testable import Rapid

@Suite("Diagnostics export privacy")
struct DiagnosticsBundleTests {
    @Test("Every exported field redacts fake home and account while retaining context")
    func allFields() {
        let home = "/Users/fake.person+test"
        let report = """
        binary: \(home)/bin/rapid-mlx
        log: sampling defaults from \(home)/.cache/models/snapshots/config.json
        env: HOME=\(home) USER=fake.person+test LOGNAME=fake.person+test
        crash: failed opening \(home)/Library/Logs/crash.txt (user fake.person+test)
        Authorization: Bearer secret-value
        chip: Apple M3 Ultra
        """
        let out = DiagnosticsBundle.redact(report, homeDirectory: home, username: "fake.person+test")
        #expect(!out.contains(home))
        #expect(!out.contains("fake.person+test"))
        #expect(!out.contains("secret-value"))
        #expect(out.contains("<redacted>/.cache/models/snapshots/config.json"))
        #expect(out.contains("HOME=<redacted> USER=<redacted> LOGNAME=<redacted>"))
        #expect(out.contains("<redacted>/Library/Logs/crash.txt"))
        #expect(out.contains("chip: Apple M3 Ultra"))
    }

    @Test("Custom homes, escaped paths and URL paths use the same placeholder")
    func representations() {
        let home = "/Volumes/Accounts/Fake Person"
        let input = #"/Volumes/Accounts/Fake Person/cache \/Volumes\/Accounts\/Fake Person/cache file:///Volumes/Accounts/Fake%20Person/cache"#
        let out = DiagnosticsBundle.redact(input, homeDirectory: home + "/", username: "fake-account")
        #expect(out == "<redacted>/cache <redacted>/cache file://<redacted>/cache")
    }

    @Test("Short account names cannot rewrite redaction placeholders")
    func shortAccount() {
        #expect(DiagnosticsBundle.redact("/Users/a/cache USER=a LOGNAME=a user a failed to load a model", homeDirectory: "/Users/a", username: "a")
            == "<redacted>/cache USER=<redacted> LOGNAME=<redacted> user <redacted> failed to load a model")
    }

    @Test("Percent-encoded allowed characters and separators cannot leak identities")
    func percentEncodings() {
        let inputs = ["/Users/fake%2Bperson/cache", "%2fUsers%2ffake%2bperson/cache",
                      "/Users/%66ake+person/cache"]
        for input in inputs {
            #expect(DiagnosticsBundle.redact(input, homeDirectory: "/Users/fake+person", username: "fake+person")
                == "<redacted>/cache")
        }
        #expect(DiagnosticsBundle.redact("USER=fake%2Bperson", homeDirectory: "/Users/fake+person", username: "fake+person")
            == "USER=<redacted>")
    }

    @Test("Homes match complete components and preserve unrelated directory names")
    func homeBoundary() {
        let input = "/Users/a/cache /Users/alice/cache /Users/a-backup/cache HOME=/Users/a"
        #expect(DiagnosticsBundle.redact(input, homeDirectory: "/Users/a", username: "a")
            == "<redacted>/cache /Users/alice/cache /Users/a-backup/cache HOME=<redacted>")
    }

    @Test("Mixed-case literal and encoded identities remain private")
    func mixedCase() {
        let inputs = ["/Users/FAKE.Person/cache", "/users/%46AKE.Person/cache",
                      "USER=Fake.Person", "USER=%46ake.Person"]
        for input in inputs {
            let output = DiagnosticsBundle.redact(input, homeDirectory: "/Users/fake.person", username: "fake.person")
            #expect(output == (input.hasPrefix("USER=") ? "USER=<redacted>" : "<redacted>/cache"))
        }
    }

    @Test("Empty identities do not corrupt diagnostic text")
    func emptyIdentities() {
        #expect(DiagnosticsBundle.redact("ready /opt/bin", homeDirectory: "", username: "") == "ready /opt/bin")
    }

    @MainActor
    @Test("Report generation applies redaction to state and binary metadata")
    func reportBoundary() {
        let home = "/Users/fake-account"
        let server = ServerManager(
            testingState: .crashed(alias: "fake-account-model", message: "failed \(home)/model/config.json"),
            binaryPath: URL(fileURLWithPath: "\(home)/bin/fake-account-engine")
        )
        server.appendLogLines(["sampling defaults from \(home)/.cache/config.json", "USER=fake-account"])
        let report = DiagnosticsBundle.makeReport(server: server, homeDirectory: home, username: "fake-account")
        #expect(!report.contains(home))
        #expect(!report.contains("fake-account"))
        #expect(report.contains("<redacted>/model/config.json"))
        #expect(report.contains("found (<redacted>-engine)"))
        #expect(report.contains("Recent log"))
        #expect(report.contains("<redacted>/.cache/config.json"))
        #expect(report.contains("USER=<redacted>"))
    }
}
