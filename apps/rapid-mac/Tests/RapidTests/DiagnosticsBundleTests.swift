import Foundation
import Testing
@testable import Rapid

@Suite("Diagnostics export privacy")
struct DiagnosticsBundleTests {
    @Test("Home paths are removed throughout the report, preserving useful suffixes")
    func allFields() {
        let home = "/Users/fake.person+test"
        let input = """
        binary: \(home)/bin/rapid-mlx
        log: loaded \(home)/.cache/models/config.json
        env: HOME=\(home) USER=fake.person+test
        crash: process owned by fake.person+test (Fake Person)
        Authorization: Bearer secret-value
        chip: Apple M3 Ultra
        """
        let out = DiagnosticsBundle.redact(input, homeDirectory: home, username: "fake.person+test", fullName: "Fake Person")
        #expect(!out.contains(home))
        #expect(!out.contains("fake.person+test"))
        #expect(!out.contains("Fake Person"))
        #expect(!out.contains("secret-value"))
        #expect(out.contains("~/.cache/models/config.json"))
        #expect(out.contains("HOME=~ USER=<redacted>"))
        #expect(out.contains("chip: Apple M3 Ultra"))
    }

    @Test("Embedded, backup, escaped, URL and expanded homes are redacted", arguments: [
        "/Users/fake-account/cache", "file:///Users/fake-account/cache",
        "/Volumes/backup/Users/fake-account/cache", #"\/Users\/fake-account/cache"#,
        "%2fUsers%2fFAKE-account/cache", "/Users/%66ake-account/cache",
        #"\u002FUsers\u002Ffake-account/cache"#, #"\u002f\u0075sers\u002ffake-account/cache"#, #"\u002fUsers\u002fother-account\u002fcache"#,
        "~fake-account/cache", "/Users/other-account/cache", "~/cache"
    ])
    func pathForms(_ input: String) {
        let out = DiagnosticsBundle.redact(input, homeDirectory: "/Users/fake-account", username: "fake-account")
        #expect(out.hasSuffix("~/cache") || out.hasSuffix(#"~\u002fcache"#))
        #expect(!out.contains("fake-account"))
        #expect(!out.contains("other-account"))
    }

    @Test("Custom homes and shell-escaped spaces are covered", arguments: [
        "/Volumes/Accounts/Fake Person/cache", #"/Volumes/Accounts/Fake\ Person/cache"#,
        "file:///Volumes/Accounts/Fake%20Person/cache",
        "/backup/Volumes/Accounts/Fake Person/cache"
    ])
    func customHomes(_ input: String) {
        let out = DiagnosticsBundle.redact(input, homeDirectory: "/Volumes/Accounts/Fake Person/", username: "fake")
        #expect(out.hasSuffix("~/cache"))
        #expect(!out.contains("Person"))
    }

    @Test("JSON surrogate pairs in custom home paths are redacted")
    func surrogateHome() {
        let out = DiagnosticsBundle.redact(#"/Volumes/\uD83D\uDE00/cache"#,
                                           homeDirectory: "/Volumes/😀", username: "account")
        #expect(out == "~/cache")
    }

    @Test("Known short and full names match whole tokens, case-insensitively")
    func wholeTokens() {
        let input = "ann ANN Annette joann ann_2 ann2 Ann Lee ANN LEE Ann Leeway xAnn Lee"
        let out = DiagnosticsBundle.redact(input, homeDirectory: "/Users/ann", username: "ann", fullName: "Ann Lee")
        #expect(out == "<redacted> <redacted> Annette joann ann_2 ann2 <redacted> <redacted> <redacted> Leeway xAnn Lee")
        #expect(DiagnosticsBundle.redact("model failed; modelled; remodel", homeDirectory: "", username: "model")
            == "<redacted> failed; modelled; remodel")
    }

    @Test("Names shorter than three characters remain in prose but home paths never do")
    func shortNames() {
        let out = DiagnosticsBundle.redact("a model AB /Users/a/cache /Users/ab/cache", homeDirectory: "/Users/a", username: "a", fullName: "AB")
        #expect(out == "a model AB ~/cache ~/cache")
    }

    @Test("Unknown free-text names are an accepted residual risk")
    func unknownNames() {
        #expect(DiagnosticsBundle.redact("owned by Unknown Person", homeDirectory: "/Users/known", username: "known", fullName: "Known Person")
            == "owned by Unknown Person")
    }

    @Test("Empty identity values do not alter ordinary text")
    func emptyIdentities() {
        #expect(DiagnosticsBundle.redact("ready /opt/bin", homeDirectory: "", username: "") == "ready /opt/bin")
    }

    @MainActor
    @Test("The export boundary covers logs, state, binary metadata and full names")
    func reportBoundary() {
        let home = "/Users/fake-account"
        let server = ServerManager(
            testingState: .crashed(alias: "fake-account-model", message: "Fake Person failed opening \(home)/model/config.json"),
            binaryPath: URL(fileURLWithPath: "\(home)/bin/fake-account-engine")
        )
        server.appendLogLines(["loaded \(home)/.cache/config.json", "USER=fake-account"])
        let report = DiagnosticsBundle.makeReport(server: server, homeDirectory: home, username: "fake-account", fullName: "Fake Person")
        #expect(!report.contains(home))
        #expect(!report.contains("fake-account"))
        #expect(!report.contains("Fake Person"))
        #expect(report.contains("~/model/config.json"))
        #expect(report.contains("found (<redacted>-engine)"))
        #expect(report.contains("~/.cache/config.json"))
    }
}
