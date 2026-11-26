import AppKit
import Foundation
import Testing
import WebKit
@testable import Rapid

@Suite("Artifact sandbox policy")
struct ArtifactSandboxPolicyTests {

    private static var root: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    @Test("The CSP is exactly the contracted policy")
    func csp() {
        #expect(
            ArtifactSandbox.contentSecurityPolicy
                == "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:; connect-src 'none'; frame-src 'none'; form-action 'none'; base-uri 'none'"
        )
    }

    @Test("The rule list blocks everything first, then re-allows only ours, data and blob")
    func ruleList() throws {
        let rules = try #require(
            try JSONSerialization.jsonObject(
                with: Data(ArtifactSandbox.contentRuleListJSON.utf8)
            ) as? [[String: Any]]
        )
        func filter(_ rule: [String: Any]) -> String? {
            (rule["trigger"] as? [String: Any])?["url-filter"] as? String
        }
        func action(_ rule: [String: Any]) -> String? {
            (rule["action"] as? [String: Any])?["type"] as? String
        }
        #expect(rules.count == 4)
        #expect(filter(rules[0]) == ".*")
        #expect(action(rules[0]) == "block")
        #expect(rules.dropFirst().map(filter) == ["^rapid-artifact://", "^data:", "^blob:"])
        #expect(rules.dropFirst().allSatisfy { action($0) == "ignore-previous-rules" })
    }

    @Test("Only the initial document load in the main frame may navigate", arguments: [
        ("rapid-artifact://local/artifact.html", true, false, true),
        ("rapid-artifact://local/artifact.html", true, true, false),
        ("rapid-artifact://local/artifact.html", false, false, false),
        ("rapid-artifact://local/other.html", true, false, false),
        ("rapid-artifact://evil/artifact.html", true, false, true),
        ("http://127.0.0.1:8080/nav", true, false, false),
        ("https://example.com", true, false, false),
        ("file:///etc/passwd", true, false, false),
        ("data:text/html,<b>x</b>", true, false, false),
        ("javascript:alert(1)", true, false, false),
        ("blob:rapid-artifact://local/1", true, false, false),
        ("about:blank", true, false, false),
    ])
    func navigation(_ raw: String, _ main: Bool, _ already: Bool, _ allowed: Bool) throws {
        let url = try #require(URL(string: raw))
        let policy = ArtifactSandbox.policy(
            for: url, isMainFrame: main, initialLoadAlreadyAllowed: already
        )
        #expect((policy == .allow) == allowed, "\(raw)")
    }

    @Test("A nil URL is refused")
    func nilURL() {
        #expect(
            ArtifactSandbox.policy(for: nil, isMainFrame: true, initialLoadAlreadyAllowed: false) == .cancel
        )
    }

    @Test("The meta CSP goes in before the page, after a doctype")
    func documentInjection() throws {
        let meta = "<meta http-equiv=\"Content-Security-Policy\" content=\"\(ArtifactSandbox.contentSecurityPolicy)\">"

        let withDoctype = String(decoding: ArtifactSandbox.document(for: "  <!DOCTYPE html>\n<html><script>1</script></html>"), as: UTF8.self)
        let doctypeEnd = try #require(withDoctype.range(of: "<!DOCTYPE html>"))
        #expect(withDoctype[doctypeEnd.upperBound...].hasPrefix(meta))
        #expect(withDoctype.trimmingCharacters(in: .whitespaces).hasPrefix("<!DOCTYPE html>"))

        let bare = String(decoding: ArtifactSandbox.document(for: "<script>1</script>"), as: UTF8.self)
        #expect(bare.hasPrefix(meta))
        #expect(bare.hasSuffix("<script>1</script>"))

        // A doctype-looking string later in the body is not the doctype.
        let late = String(decoding: ArtifactSandbox.document(for: "<p><!doctype x></p>"), as: UTF8.self)
        #expect(late.hasPrefix(meta))
    }

    @Test("Everything the sandbox is made of stays free of network schemes and bridges")
    func sourceGuard() throws {
        let sandbox = try String(
            contentsOf: Self.root.appendingPathComponent("Sources/Rapid/UI/Artifacts/ArtifactSandbox.swift"),
            encoding: .utf8
        )
        #expect(!sandbox.contains("http:"))
        #expect(!sandbox.contains("https:"))
        #expect(!sandbox.contains("ws://"))

        let directory = Self.root.appendingPathComponent("Sources/Rapid/UI/Artifacts")
        for file in try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        where file.pathExtension == "swift" {
            let text = try String(contentsOf: file, encoding: .utf8)
                .split(separator: "\n")
                .filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("//") }
                .joined(separator: "\n")
            var forbidden = [
                "WKScriptMessageHandler", "add(_ scriptMessageHandler", "addScriptMessageHandler",
                "isInspectable = true", "persistent()",
            ]
            // The isolation script lives in the sandbox file and nowhere else.
            if file.lastPathComponent != "ArtifactSandbox.swift" {
                forbidden += ["addUserScript", "WKUserScript"]
            }
            for term in forbidden {
                #expect(!text.contains(term), "\(file.lastPathComponent) contains \(term)")
            }
        }
    }
}

// MARK: - WebKit

/// An offscreen window hosting a controller's web view. WebKit will not run a
/// view that is in no window.
@MainActor
private final class ArtifactHarness {
    let controller: ArtifactPreviewController
    private let window: NSWindow

    init(
        source: String,
        isolationScript: @escaping (ArtifactSandbox.Isolation) -> String = { $0.script },
        loadTimeout: Duration = .seconds(10),
        pingInterval: Duration = .seconds(5),
        pingTimeout: Duration = .seconds(10)
    ) {
        controller = ArtifactPreviewController(
            source: source, isolationScript: isolationScript, loadTimeout: loadTimeout,
            pingInterval: pingInterval, pingTimeout: pingTimeout
        )
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 400, height: 300),
            styleMask: [.borderless], backing: .buffered, defer: false
        )
        window.isReleasedWhenClosed = false
        controller.hostView.frame = window.contentView!.bounds
        window.contentView?.addSubview(controller.hostView)
        window.orderOut(nil)
    }

    deinit { MainActor.assumeIsolated { window.close() } }

    /// Wait for a state; false if it never arrives.
    func wait(
        until predicate: (ArtifactPreviewController.State) -> Bool, seconds: Double = 10
    ) async -> Bool {
        for _ in 0..<Int(seconds * 20) {
            if predicate(controller.state) { return true }
            try? await Task.sleep(for: .milliseconds(50))
        }
        return predicate(controller.state)
    }

    func loadAndWaitUntilReady() async -> Bool {
        controller.start()
        return await wait { $0 == .ready }
    }

    func evaluate(_ script: String) async throws -> Any? {
        try await controller.webView?.evaluateJavaScript(script)
    }
}

/// The same page, in a web view with none of the guards.
@MainActor
private final class UnguardedHarness {
    private let webView = WKWebView(
        frame: NSRect(x: 0, y: 0, width: 400, height: 300),
        configuration: WKWebViewConfiguration()
    )
    private let window = NSWindow(
        contentRect: NSRect(x: 0, y: 0, width: 400, height: 300),
        styleMask: [.borderless], backing: .buffered, defer: false
    )

    init() {
        window.isReleasedWhenClosed = false
        window.contentView?.addSubview(webView)
        window.orderOut(nil)
    }

    deinit { MainActor.assumeIsolated { window.close() } }

    func load(_ html: String) async {
        webView.loadHTMLString(html, baseURL: nil)
        try? await Task.sleep(for: .milliseconds(1_500))
    }
}

/// One network vector per entry; each is a page that tries to reach the probe.
/// `baselineReachesProbe` is false only where an unguarded view cannot be made
/// to connect without a user gesture, so there is no control to run.
struct ArtifactNetworkVector: Sendable, CustomTestStringConvertible {
    let name: String
    let template: String
    let baselineReachesProbe: Bool
    var testDescription: String { name }

    func body(target: String, port: UInt16) -> String {
        template
            .replacingOccurrences(of: "TARGET", with: target)
            .replacingOccurrences(of: "PORT", with: String(port))
    }

    static let all: [ArtifactNetworkVector] = [
        .init(name: "fetch", template: "<script>fetch('TARGET').catch(function(){})</script>", baselineReachesProbe: true),
        .init(name: "xhr", template: "<script>var x=new XMLHttpRequest();x.open('GET','TARGET');x.send()</script>", baselineReachesProbe: true),
        .init(name: "beacon", template: "<script>navigator.sendBeacon('TARGET','x')</script>", baselineReachesProbe: true),
        .init(name: "websocket", template: "<script>try{new WebSocket('ws://127.0.0.1:PORT/')}catch(e){}</script>", baselineReachesProbe: true),
        .init(name: "script src", template: "<script src='TARGET'></script>", baselineReachesProbe: true),
        .init(name: "img src", template: "<img src='TARGET'>", baselineReachesProbe: true),
        .init(name: "link", template: "<link rel='stylesheet' href='TARGET'>", baselineReachesProbe: true),
        .init(name: "css url", template: "<style>body{background:url('TARGET')}</style>", baselineReachesProbe: true),
        .init(name: "iframe", template: "<iframe src='TARGET'></iframe>", baselineReachesProbe: true),
        .init(name: "location", template: "<script>location = 'TARGET'</script>", baselineReachesProbe: true),
        .init(
            name: "form submit",
            template: "<form id='f' action='TARGET' method='post'></form><script>document.getElementById('f').submit()</script>",
            baselineReachesProbe: true
        ),
        .init(
            name: "webrtc turn",
            template: "<script>try{var pc=new RTCPeerConnection({iceServers:[{urls:'turn:127.0.0.1:PORT?transport=tcp',username:'u',credential:'c'}]});pc.createDataChannel('x');pc.createOffer().then(function(o){return pc.setLocalDescription(o)}).catch(function(){})}catch(e){}</script>",
            baselineReachesProbe: true
        ),
        .init(
            name: "webrtc turn from a child frame",
            template: "<script>try{var f=document.createElement('iframe');document.body.appendChild(f);var pc=new f.contentWindow.RTCPeerConnection({iceServers:[{urls:'turn:127.0.0.1:PORT?transport=tcp',username:'u',credential:'c'}]});pc.createDataChannel('x');pc.createOffer().then(function(o){return pc.setLocalDescription(o)}).catch(function(){})}catch(e){}</script>",
            baselineReachesProbe: true
        ),
        .init(name: "preconnect", template: "<link rel='preconnect' href='TARGET'>", baselineReachesProbe: true),
        .init(name: "window.open", template: "<script>window.open('TARGET')</script>", baselineReachesProbe: false),
    ]
}

@MainActor
@Suite("Artifact web view sandbox")
struct ArtifactSandboxWebKitTests {

    private func page(_ body: String) -> String {
        "<!DOCTYPE html><html><head><title>t</title></head><body>\(body)</body></html>"
    }

    @Test("The configuration is fresh, non-persistent and bridge-free")
    func configuration() async throws {
        let compiled = await ArtifactSandbox.compiledRuleList()
        let ruleList = try #require(compiled)
        let isolation = ArtifactSandbox.Isolation()
        #expect(isolation.token != ArtifactSandbox.Isolation().token)
        let first = ArtifactSandbox.makeConfiguration(
            source: "<html></html>", ruleList: ruleList, isolationScript: isolation.script
        )
        let second = ArtifactSandbox.makeConfiguration(
            source: "<html></html>", ruleList: ruleList, isolationScript: isolation.script
        )
        #expect(!first.websiteDataStore.isPersistent)
        #expect(first.websiteDataStore !== second.websiteDataStore)
        #expect(!first.preferences.javaScriptCanOpenWindowsAutomatically)
        let scripts = first.userContentController.userScripts
        #expect(scripts.count == 1, "only the isolation script, nothing that talks to the app")
        #expect(scripts.first?.source == isolation.script)
        #expect(scripts.first?.injectionTime == .atDocumentStart)
        #expect(scripts.first?.isForMainFrameOnly == false)
        #expect(first.urlSchemeHandler(forURLScheme: ArtifactSandbox.scheme) != nil)
        #expect(first.mediaTypesRequiringUserActionForPlayback == .all)
    }

    @Test("WebRTC globals are locked and a page cannot restore them")
    func webRTCIsLocked() async throws {
        let harness = ArtifactHarness(source: page("<p>x</p>"))
        #expect(await harness.loadAndWaitUntilReady())
        let types = try await harness.evaluate("""
            [typeof RTCPeerConnection, typeof webkitRTCPeerConnection, typeof RTCDataChannel].join(",")
            """)
        #expect(types as? String == "undefined,undefined,undefined")
        let after = try await harness.evaluate("""
            (function () {
              try { window.RTCPeerConnection = function () {}; } catch (e) {}
              try { Object.defineProperty(window, "RTCPeerConnection", { value: function () {} }); } catch (e) {}
              try { delete window.RTCPeerConnection; } catch (e) {}
              return typeof RTCPeerConnection;
            })()
            """)
        #expect(after as? String == "undefined")
    }

    @Test("A load whose WebRTC removal cannot be confirmed fails closed")
    func failsClosedWithoutIsolation() async throws {
        let harness = ArtifactHarness(source: page("<p>x</p>"), isolationScript: { _ in "" })
        harness.controller.start()
        let failed = await harness.wait(until: { $0 == .failed(.sandboxUnavailable) })
        #expect(failed, "state: \(harness.controller.state)")
        #expect(harness.controller.webView == nil)
    }

    @Test("A page cannot forge the isolation proof when the lock-down is absent")
    func isolationProofCannotBeForged() async throws {
        let harness = ArtifactHarness(
            source: page("""
                <script>
                Array.prototype.every = function () { return true; };
                Object.getOwnPropertyDescriptor = function () {
                  return { value: undefined, writable: false, configurable: false };
                };
                </script>
                """),
            isolationScript: { _ in "" }
        )
        harness.controller.start()
        let failed = await harness.wait(until: { $0 == .failed(.sandboxUnavailable) })
        #expect(failed, "state: \(harness.controller.state)")
        #expect(harness.controller.webView == nil)
    }

    @Test("A global that cannot be locked fails the load closed")
    func unlockableGlobalFailsClosed() async throws {
        let harness = ArtifactHarness(
            source: page("<p>x</p>"),
            isolationScript: {
                "Object.defineProperty(window, 'RTCPeerConnection', { value: 1, writable: false, configurable: false });"
                    + $0.script
            }
        )
        harness.controller.start()
        let failed = await harness.wait(until: {
            if case .failed = $0 { return true }
            return false
        })
        #expect(failed, "state: \(harness.controller.state)")
        #expect(harness.controller.webView == nil)
    }

    @Test("A counter page runs and its button works")
    func counter() async throws {
        let harness = ArtifactHarness(source: page("""
            <button id="b" onclick="document.getElementById('c').textContent = ++n">+</button>
            <span id="c">0</span>
            <script>var n = 0;</script>
            """))
        #expect(await harness.loadAndWaitUntilReady())
        #expect(harness.controller.webView?.isInspectable == false)
        _ = try await harness.evaluate("document.getElementById('b').click()")
        _ = try await harness.evaluate("document.getElementById('b').click()")
        let text = try await harness.evaluate("document.getElementById('c').textContent")
        #expect(text as? String == "2")
    }

    @Test("The page runs under the policy it was served with")
    func policyApplies() async throws {
        let harness = ArtifactHarness(source: "<!DOCTYPE html><html><body>x</body></html>")
        #expect(await harness.loadAndWaitUntilReady())
        // The meta is first in <head>, and the page is not in quirks mode.
        let meta = try await harness.evaluate(
            "document.head.firstElementChild && document.head.firstElementChild.httpEquiv"
        )
        #expect((meta as? String)?.lowercased() == "content-security-policy")
        let mode = try await harness.evaluate("document.compatMode")
        #expect(mode as? String == "CSS1Compat")
        let blocked = try await harness.evaluate("""
            (function () { try { new Function("return 1")(); return "allowed"; } catch (e) { return "blocked"; } })()
            """)
        #expect(blocked as? String == "blocked", "the policy does not grant unsafe-eval")
    }

    @Test("Remote scripts and styles do not load")
    func remoteResourcesDoNotLoad() async throws {
        let probe = try #require(LocalRequestProbe())
        defer { probe.stop() }
        let harness = ArtifactHarness(source: page("""
            <script src="http://127.0.0.1:\(probe.port)/lib.js"></script>
            <script>window.libLoaded = typeof lib !== "undefined";</script>
            """))
        #expect(await harness.loadAndWaitUntilReady())
        try await Task.sleep(for: .milliseconds(800))
        #expect(try await harness.evaluate("window.libLoaded") as? Bool == false)
        #expect(probe.requestCount == 0)
    }

    // MARK: Network denial

    @Test("The probe counts a real connection from an unguarded page")
    func positiveControl() async throws {
        let probe = try #require(LocalRequestProbe())
        defer { probe.stop() }
        #expect(probe.requestCount == 0)
        let unguarded = UnguardedHarness()
        await unguarded.load(page("<img src='http://127.0.0.1:\(probe.port)/control'>"))
        #expect(probe.requestCount >= 1, "the probe cannot see a web view's connections; every denial here is vacuous")
    }

    @Test("Network vectors reach nothing from the sandbox", arguments: ArtifactNetworkVector.all)
    func vectorsAreDenied(_ vector: ArtifactNetworkVector) async throws {
        let probe = try #require(LocalRequestProbe())
        defer { probe.stop() }
        let target = "http://127.0.0.1:\(probe.port)/leak"
        let html = page(vector.body(target: target, port: probe.port))

        if vector.baselineReachesProbe {
            let unguarded = UnguardedHarness()
            await unguarded.load(html)
            #expect(probe.requestCount >= 1, "\(vector.name): the unguarded page never connected, so the denial below proves nothing")
        }
        let baseline = probe.requestCount

        let harness = ArtifactHarness(source: html)
        #expect(await harness.loadAndWaitUntilReady(), "the page never ran, so the denial would be vacuous")
        try await Task.sleep(for: .milliseconds(1_500))
        #expect(probe.requestCount == baseline, "\(vector.name): the sandbox let a request through")
        withExtendedLifetime(harness) {}
    }

    @Test("A page cannot navigate itself away from the document")
    func cannotNavigateAway() async throws {
        let harness = ArtifactHarness(source: page("<p>stay</p>"))
        #expect(await harness.loadAndWaitUntilReady())
        _ = try? await harness.evaluate("location.href = 'rapid-artifact://local/artifact.html?again=1'")
        _ = try? await harness.evaluate("location.href = 'about:blank'")
        try await Task.sleep(for: .milliseconds(800))
        #expect(harness.controller.webView?.url == ArtifactSandbox.documentURL)
    }

    @Test("Dialogs, new windows and file panels are answered without a modal")
    func dialogsDoNotBlock() async throws {
        let harness = ArtifactHarness(source: page("<p>x</p>"))
        #expect(await harness.loadAndWaitUntilReady())
        let result = try await harness.evaluate("""
            [String(window.open('about:blank')), String(confirm('x')), String(prompt('x'))].join(",")
            """)
        #expect(result as? String == "null,false,null")
        _ = try await harness.evaluate("alert('x')")
        #expect(harness.controller.state == .ready)
    }

    // MARK: Lifecycle

    @Test("A page that never finishes loading is torn down, and the app stays responsive")
    func loadWatchdog() async throws {
        let harness = ArtifactHarness(
            source: page("<script>while (true) {}</script>"), loadTimeout: .milliseconds(800)
        )
        harness.controller.start()
        let failed = await harness.wait(until: { $0 == .failed(.timedOut) })
        #expect(failed)
        #expect(harness.controller.webView == nil)
    }

    @Test("A page that stops answering after it loads is torn down")
    func livenessWatchdog() async throws {
        let harness = ArtifactHarness(
            source: page("<script>window.onload = function () { setTimeout(function () { while (true) {} }, 100); };</script>"),
            loadTimeout: .milliseconds(300), pingInterval: .milliseconds(300), pingTimeout: .milliseconds(800)
        )
        harness.controller.start()
        let failed = await harness.wait(until: { $0 == .failed(.unresponsive) }, seconds: 15)
        #expect(failed, "state: \(harness.controller.state)")
        #expect(harness.controller.webView == nil)
    }

    @Test("Teardown releases the view; reload builds a new one")
    func teardownAndReload() async throws {
        let harness = ArtifactHarness(source: page("<p>x</p>"))
        #expect(await harness.loadAndWaitUntilReady())
        #expect(harness.controller.webViewsCreated == 1)
        #expect(harness.controller.hostView.subviews.count == 1)

        harness.controller.teardown()
        #expect(harness.controller.webView == nil)
        #expect(harness.controller.hostView.subviews.isEmpty)
        #expect(harness.controller.state == .idle)

        harness.controller.reload()
        #expect(await harness.wait(until: { $0 == .ready }))
        #expect(harness.controller.webViewsCreated == 2)
        #expect(harness.controller.hostView.subviews.count == 1)
    }

    @Test("Starting twice does not build two views")
    func startIsIdempotent() async throws {
        let harness = ArtifactHarness(source: page("<p>x</p>"))
        harness.controller.start()
        harness.controller.start()
        #expect(await harness.wait(until: { $0 == .ready }))
        harness.controller.start()
        #expect(harness.controller.webViewsCreated == 1)
    }
}
