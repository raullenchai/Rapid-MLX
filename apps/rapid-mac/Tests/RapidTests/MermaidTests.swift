import WebKit
import AppKit
import Foundation
import Network
import Testing
@testable import Rapid

@Suite("Mermaid source detection")
struct MermaidSourceTests {

    private let flowchart = "graph TD\n  A[Start] --> B{Choice}\n  B -->|yes| C[Go]"

    // MARK: - The tag decides, when there is one

    @Test("A tagged diagram is detected", arguments: ["mermaid", "Mermaid", "MMD", "mmd"])
    func taggedIsDetected(_ language: String) {
        #expect(MermaidSource.looksLikeMermaid(code: flowchart, language: language))
    }

    /// The deliberate inversion of the SVG rule, and worth stating so nobody
    /// "fixes" it later: `SVGPreview` ignores the tag because `NSImage` is a
    /// cheap synchronous authority. Mermaid's authority is 3.4 MB of
    /// JavaScript in another process, so a wrong guess costs a real render.
    @Test("A diagram tagged as something else is refused", arguments: [
        "python", "swift", "text", "json",
    ])
    func misTaggedIsRefused(_ language: String) {
        #expect(!MermaidSource.looksLikeMermaid(code: flowchart, language: language))
    }

    // MARK: - Untagged blocks are read

    @Test("Every diagram keyword opens a diagram")
    func everyKeywordIsDetected() {
        for keyword in MermaidSource.diagramKeywords {
            // `graph`/`flowchart` require a direction; the rest take anything.
            let head = keyword.hasPrefix("graph") || keyword.hasPrefix("flowchart")
                ? "\(keyword) TD" : "\(keyword) Demo"
            #expect(
                MermaidSource.looksLikeMermaid(code: "\(head)\n  A --> B", language: nil),
                "\(keyword) was not detected"
            )
            #expect(
                MermaidSource.looksLikeMermaid(code: "\(keyword)\n  A --> B", language: nil),
                "\(keyword) alone on its line was not detected"
            )
        }
    }

    /// Pinned by value so gaining a diagram type is a deliberate two-line
    /// diff rather than something that drifts in.
    @Test("The keyword set is what we think it is")
    func keywordSetIsPinned() {
        #expect(MermaidSource.diagramKeywords.count == 30)
        for expected in ["graph", "flowchart", "sequenceDiagram", "classDiagram",
                         "stateDiagram-v2", "erDiagram", "gantt", "pie", "mindmap",
                         "timeline", "gitGraph"] {
            #expect(MermaidSource.diagramKeywords.contains(expected))
        }
    }

    @Test("Frontmatter and comments are skipped")
    func preambleIsSkipped() {
        #expect(MermaidSource.looksLikeMermaid(
            code: "---\ntitle: Demo\n---\ngraph TD\n  A --> B", language: nil
        ))
        #expect(MermaidSource.looksLikeMermaid(
            code: "%% a comment\n%%{init: {'theme':'dark'}}%%\ngraph TD\n  A --> B", language: nil
        ))
    }

    /// The trailing-separator rule: a keyword has to open a document, not
    /// merely start an identifier.
    /// `graph` and `flowchart` are ordinary words. Their grammar says a
    /// direction comes next, so an assignment or a call is not a diagram.
    @Test("Code that merely starts with the word is refused", arguments: [
        "graph = nx.Graph()\ngraph.add_edge(1, 2)",
        "graph TDD_helper = 1",
        "flowchart is a kind of diagram used for...",
        "graph_data = load()",
        "pie(values, labels)",
        "timeline.append(event)",
    ])
    func lookalikeCodeIsRefused(_ code: String) {
        #expect(!MermaidSource.looksLikeMermaid(code: code, language: nil))
    }

    @Test("Nothing to render", arguments: ["", "   ", "print(\"hi\")", "# A heading"])
    func nonDiagramsRefused(_ code: String) {
        #expect(!MermaidSource.looksLikeMermaid(code: code, language: nil))
    }

    /// The render happens in another process and its cost is real, so the
    /// size guard fires before anything is asked of it.
    @Test("An oversized source is refused before anything else")
    func oversizedRefused() {
        let huge = "graph TD\n" + String(repeating: "  A --> B\n", count: 20_000)
        #expect(huge.utf8.count > MermaidSource.maximumSourceBytes)
        #expect(!MermaidSource.looksLikeMermaid(code: huge, language: "mermaid"))
    }
}

/// What the vendored library check actually checks.
@Suite("Mermaid library integrity")
struct MermaidLibraryTests {

    @Test("The vendored library is present and matches its digest")
    func vendoredLibraryLoads() throws {
        let data = try #require(
            MermaidLibrary.load(),
            "Vendor/mermaid/mermaid.min.js is missing or does not match its .sha256"
        )
        #expect(data.count > 1_000_000, "the bundle looks truncated")
    }

    /// The point of a digest rather than a size band: a file that is the
    /// right length and the wrong bytes has to fail too.
    @Test("A tampered library is refused")
    func tamperedLibraryIsRefused() throws {
        let url = try #require(MermaidLibrary.developmentVendorURL)
            .appendingPathComponent("mermaid.min.js")
        var data = try Data(contentsOf: url)
        let real = MermaidLibrary.digest(of: data)
        data[data.count / 2] = data[data.count / 2] &+ 1
        #expect(MermaidLibrary.digest(of: data) != real)
        #expect(data.count == (try Data(contentsOf: url)).count, "same length, different bytes")
    }
}

/// The claim that lets this feature ship without a sandbox or a permission
/// prompt: a model-authored diagram cannot reach the network.
///
/// Deliberately **not** gated behind an opt-in trait. `HermeticTraits` gates
/// suites whose *results* are host-dependent; "did anything connect to this
/// socket" is not. A security assertion that skips quietly on the machine
/// where it matters is worth nothing.
@MainActor
@Suite("Mermaid network denial")
struct MermaidNetworkDenialTests {

    /// This suite's own renderer. Swift Testing builds a `struct` suite once
    /// per test, so every test gets a clean cache, failure budget and web
    /// view, and no suite can reset another's while it is mid-render.
    private let renderer = MermaidRenderer()

    /// **Runs first, and everything else in this file depends on it.**
    ///
    /// Every other assertion here is of the form `requestCount == 0`, which
    /// is exactly the shape that passes when the probe is broken. This one
    /// proves the probe can count — without it, eleven green tests would
    /// prove nothing at all. The SVG suite shipped without this and its
    /// no-network assertion was, until now, unfalsifiable.
    @Test("The probe counts a real connection")
    func probePositiveControl() async throws {
        let probe = try #require(LocalRequestProbe())
        defer { probe.stop() }
        #expect(probe.requestCount == 0)

        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(probe.port)/x")!)
        request.timeoutInterval = 2
        _ = try? await URLSession.shared.data(for: request)

        try await Task.sleep(for: .milliseconds(400))
        #expect(probe.requestCount >= 1, "the probe cannot see connections; every other test here is vacuous")
    }

    /// A diagram whose label smuggles a remote image. Mermaid renders it,
    /// WebKit is asked to load it, and nothing must arrive.
    @Test("A diagram cannot fetch a remote image")
    func diagramCannotFetch() async throws {
        let probe = try #require(LocalRequestProbe())
        defer { probe.stop() }
        let source = """
            graph TD
              A["<img src='http://127.0.0.1:\(probe.port)/leak.png'>"] --> B[End]
            """
        _ = await renderer.image(source: source, theme: .light)
        try await Task.sleep(for: .milliseconds(600))
        #expect(probe.requestCount == 0, "the diagram reached the network")
    }

    /// The navigation rule, asked directly.
    ///
    /// The previous version of this rendered a `click … href` directive and
    /// asserted nothing connected. It passed with the whole delegate deleted:
    /// an offscreen page never activates an anchor, so the assertion was true
    /// for reasons unrelated to navigation policy. That mattered, because the
    /// setup race this PR also fixes left seven of eight web views with no
    /// navigation delegate at all — a layer disappearing in practice, with
    /// nothing able to see it.
    @Test("Only the private scheme may navigate", arguments: [
        ("rapid-mermaid://local/host.html", true),
        ("http://127.0.0.1:8080/nav", false),
        ("https://example.com", false),
        ("file:///etc/passwd", false),
        ("data:text/html,<b>x</b>", false),
        ("javascript:alert(1)", false),
        ("about:blank", false),
    ])
    func onlyThePrivateSchemeNavigates(_ raw: String, _ allowed: Bool) throws {
        let url = try #require(URL(string: raw))
        let policy = MermaidNavigationPolicy.policy(for: url)
        #expect((policy == .allow) == allowed, "\(raw)")
    }

    @Test("A nil URL is refused")
    func nilURLRefused() {
        #expect(MermaidNavigationPolicy.policy(for: nil) == .cancel)
    }

    /// And end to end: a diagram carrying a link still reaches nothing.
    @Test("A click directive cannot navigate")
    func clickDirectiveCannotNavigate() async throws {
        let probe = try #require(LocalRequestProbe())
        defer { probe.stop() }
        let source = """
            graph TD
              A[Start] --> B[End]
              click A href "http://127.0.0.1:\(probe.port)/nav"
            """
        _ = await renderer.image(source: source, theme: .light)
        try await Task.sleep(for: .milliseconds(600))
        #expect(probe.requestCount == 0)
    }

    /// The delegate as it is actually installed, refusing a real navigation.
    ///
    /// Every other assertion in this suite reaches the policy either
    /// statically (`policy(for:)`) or through a page that never activates a
    /// link, so none of them can tell an installed delegate from a deleted
    /// one. This one loads the real private-scheme page, tells that page to
    /// navigate itself somewhere else — the only way a diagram could try —
    /// and asks a listening socket what turned up.
    ///
    /// The unguarded pass is not decoration. `requestCount == 0` is also what
    /// you get when the navigation never happened for reasons of its own, so
    /// the identical page is made to connect first with no delegate attached.
    /// Only after that does zero mean the policy stopped it.
    ///
    /// Note `withExtendedLifetime`. `navigationDelegate` is weak; a caller
    /// that assigns a freshly constructed policy without holding it reads back
    /// nil and allows everything, which from the socket's side is
    /// indistinguishable from a policy that does not work.
    @Test("An installed delegate refuses a navigation off the host page")
    func installedDelegateRefusesNavigation() async throws {
        let probe = try #require(LocalRequestProbe())
        defer { probe.stop() }
        let target = "http://127.0.0.1:\(probe.port)/inpage"

        let unguarded = MermaidNavigationBoundaryHarness(delegate: nil)
        try await unguarded.loadHostPage()
        await unguarded.navigate(to: target)
        #expect(
            probe.requestCount >= 1,
            "the page could not reach the socket even unguarded; the assertion below would be vacuous"
        )

        let baseline = probe.requestCount

        let policy = MermaidNavigationPolicy()
        let guarded = MermaidNavigationBoundaryHarness(delegate: policy)
        #expect(guarded.webView.navigationDelegate != nil, "the policy was released before it could decide anything")
        try await guarded.loadHostPage()
        await guarded.navigate(to: target)

        #expect(probe.requestCount == baseline, "the installed delegate let a navigation reach the network")
        #expect(
            guarded.webView.url == MermaidHostPage.hostPageURL,
            "the web view left the host page"
        )
        withExtendedLifetime(policy) {}
    }
}

/// The renderer's web view, minus the rendering.
///
/// Built by hand rather than reaching into `renderer` because
/// the point is to drive navigation on demand, and the renderer deliberately
/// offers no way to do that.
@MainActor
private final class MermaidNavigationBoundaryHarness {
    let webView: WKWebView

    init(delegate: WKNavigationDelegate?) {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        configuration.setURLSchemeHandler(
            MermaidBoundaryStubHandler(), forURLScheme: MermaidHostPage.scheme
        )
        webView = WKWebView(frame: CGRect(x: 0, y: 0, width: 200, height: 200), configuration: configuration)
        webView.navigationDelegate = delegate
    }

    /// Polls rather than waiting on `didFinish`.
    ///
    /// An earlier version installed a `MermaidNavigationPolicy` here to await
    /// the load, which quietly guarded the unguarded control — it connected to
    /// nothing, and the whole test would have passed for the wrong reason. The
    /// control caught it. Nothing may be attached to this web view that the
    /// caller did not ask for.
    func loadHostPage() async throws {
        webView.load(URLRequest(url: MermaidHostPage.hostPageURL))
        for _ in 0..<40 {
            if webView.url != nil && !webView.isLoading { return }
            try await Task.sleep(for: .milliseconds(50))
        }
    }

    /// A fixed wait: a refusal produces no event to wait for, and the
    /// unguarded control has to be given the same budget to succeed in.
    func navigate(to urlString: String) async {
        webView.evaluateJavaScript("location.href = '\(urlString)'") { _, _ in }
        try? await Task.sleep(for: .milliseconds(900))
    }
}

/// Serves a bare page. The real handler also serves 3.4 MB of Mermaid, which
/// this test has no use for.
private final class MermaidBoundaryStubHandler: NSObject, WKURLSchemeHandler {
    func webView(_ webView: WKWebView, start task: WKURLSchemeTask) {
        let html = Data("<html><body>boundary</body></html>".utf8)
        task.didReceive(URLResponse(
            url: task.request.url!, mimeType: "text/html",
            expectedContentLength: html.count, textEncodingName: "utf-8"
        ))
        task.didReceive(html)
        task.didFinish()
    }

    func webView(_ webView: WKWebView, stop task: WKURLSchemeTask) {}
}

/// The rules, read as text. Cheap, and they fail loudly if someone loosens
/// the policy without noticing.
@Suite("Mermaid host page policy")
struct MermaidHostPagePolicyTests {

    @Test("The deny-all rule list is well formed and denies first")
    func ruleListDeniesFirst() throws {
        let data = Data(MermaidHostPage.contentRuleListJSON.utf8)
        let rules = try #require(
            try JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        )
        #expect(rules.count == 2)
        let first = try #require(rules.first)
        #expect((first["trigger"] as? [String: Any])?["url-filter"] as? String == ".*")
        #expect((first["action"] as? [String: Any])?["type"] as? String == "block")
        // Order is load-bearing: the allow has to come second.
        let second = try #require(rules.last)
        #expect((second["action"] as? [String: Any])?["type"] as? String == "ignore-previous-rules")
    }

    @Test("The host page names no remote origin")
    func hostPageHasNoRemoteOrigin() {
        let html = MermaidHostPage.html
        #expect(!html.contains("http:"))
        #expect(!html.contains("https:"))
        #expect(html.contains("default-src 'none'"))
        #expect(html.contains("connect-src 'none'"))
    }

    /// Mermaid lets a `%%{init: …}%%` directive inside the diagram override
    /// configuration. Without the `secure` list, a model-authored diagram can
    /// turn `htmlLabels` back on and inject markup into this page.
    @Test("Diagram directives cannot re-enable HTML labels")
    func directivesCannotEnableHTMLLabels() {
        let html = MermaidHostPage.html
        #expect(html.contains("securityLevel: \"strict\""))
        #expect(html.contains("\"htmlLabels\""))
        #expect(html.contains("suppressErrorRendering: true"))
    }
}

/// The end-to-end claim: a diagram becomes a picture with the right shape.
///
/// This is the only test that proves the vendored library is real and that
/// the whole pipeline — scheme handler, host page, Mermaid, snapshot —
/// actually joins up.
@MainActor
@Suite("Mermaid rendering")
struct MermaidRenderingTests {

    /// This suite's own renderer. Swift Testing builds a `struct` suite once
    /// per test, so every test gets a clean cache, failure budget and web
    /// view, and no suite can reset another's while it is mid-render.
    private let renderer = MermaidRenderer()

    @Test("Snapshot dimensions are bounded before bitmap allocation")
    func snapshotDimensionsAreBounded() {
        #expect(MermaidRenderer.acceptsSnapshot(width: 2_000, height: 2_000))
        #expect(!MermaidRenderer.acceptsSnapshot(width: 4_097, height: 100))
        #expect(!MermaidRenderer.acceptsSnapshot(width: 100, height: 4_097))
        #expect(!MermaidRenderer.acceptsSnapshot(width: 4_000, height: 4_000))
        #expect(!MermaidRenderer.acceptsSnapshot(width: Int.max, height: 2))
    }

    @Test("A timed-out render releases the serial queue")
    func timedOutRenderReleasesQueue() async {
        let renderer = MermaidRenderer(
            renderTimeout: .milliseconds(100),
            javaScriptEvaluator: { webView, source, theme, completion in
                guard source != "hang forever" else { return }
                webView.callAsyncJavaScript(
                    "return await __rapidRender(source, theme);",
                    arguments: ["source": source, "theme": theme.rawValue],
                    in: nil,
                    in: .page,
                    completionHandler: completion
                )
            }
        )

        let started = ContinuousClock.now
        #expect(await renderer.image(source: "hang forever", theme: .light) == nil)
        #expect(ContinuousClock.now - started < .seconds(2))

        let recovered = await renderer.image(
            source: "graph TD\n  Recovered --> Queue", theme: .light
        )
        #expect(recovered != nil)
    }

    @Test("A timed-out snapshot releases the serial queue")
    func timedOutSnapshotReleasesQueue() async {
        var attempts = 0
        let renderer = MermaidRenderer(
            renderTimeout: .milliseconds(100),
            snapshotter: { webView, configuration, completion in
                attempts += 1
                guard attempts > 1 else { return }
                webView.takeSnapshot(
                    with: configuration, completionHandler: completion
                )
            }
        )

        #expect(await renderer.image(
            source: "graph TD\n  Snapshot --> Timeout", theme: .light
        ) == nil)
        let recovered = await renderer.image(
            source: "graph TD\n  Snapshot --> Recovered", theme: .light
        )
        #expect(recovered != nil)
    }

    @Test("Diagrams of every common kind render", arguments: [
        ("flowchart", "graph TD\n  A[Start] --> B{Choice}\n  B -->|yes| C[Go]"),
        ("sequence", "sequenceDiagram\n  A->>B: hello\n  B-->>A: hi"),
        ("class", "classDiagram\n  Animal <|-- Duck\n  Animal : +int age"),
        ("state", "stateDiagram-v2\n  [*] --> Idle\n  Idle --> Busy: go"),
        ("er", "erDiagram\n  CUSTOMER ||--o{ ORDER : places"),
        ("pie", "pie title Share\n  \"A\" : 40\n  \"B\" : 60"),
    ])
    func diagramsRender(_ name: String, _ source: String) async {
        let image = await renderer.image(source: source, theme: .light)
        guard let image else {
            Issue.record("\(name) produced no image")
            return
        }
        // Non-degenerate: a blank or one-pixel result would satisfy "not nil"
        // and satisfy nothing a reader cares about.
        #expect(image.size.width > 40, "\(name) is too narrow to be a diagram")
        #expect(image.size.height > 40, "\(name) is too short to be a diagram")
    }

    /// The picture has to have ink in it. An earlier design returned Mermaid's
    /// SVG for AppKit to draw, and AppKit produced correctly-sized, entirely
    /// empty boxes — every "is it nil / is it the right size" assertion
    /// passed while the feature was useless.
    /// Compared against a text-free control rendered the same way, not
    /// against an absolute threshold.
    ///
    /// The first version counted dark samples and required more than a
    /// hundred. Measured, a diagram of two *empty* boxes scores **higher**
    /// than the real one — a smaller drawing is magnified more into the fixed
    /// rect, and borders count while the pale fill does not. It caught only a
    /// fully blank image, which is not the regression it was written for:
    /// AppKit's SVG renderer produced correctly-sized boxes with every label
    /// missing.
    @Test("The picture contains its labels")
    func pictureHasInk() async throws {
        let labelled = try #require(
            await renderer.image(
                source: "graph TD\n  A[Alphabet] --> B[Beetroot]", theme: .light
            )
        )
        let bare = try #require(
            await renderer.image(
                source: "graph TD\n  A[ ] --> B[ ]", theme: .light
            )
        )
        let image = labelled
        let rep = try #require(NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: 300, pixelsHigh: 200,
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
        ))
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        NSColor.white.setFill()
        NSRect(x: 0, y: 0, width: 300, height: 200).fill()
        image.draw(in: NSRect(x: 0, y: 0, width: 300, height: 200))
        NSGraphicsContext.restoreGraphicsState()

        var ink = 0
        for x in stride(from: 0, to: 300, by: 3) {
            for y in stride(from: 0, to: 200, by: 3) {
                guard let colour = rep.colorAt(x: x, y: y) else { continue }
                let luminance = 0.3 * colour.redComponent
                    + 0.59 * colour.greenComponent + 0.11 * colour.blueComponent
                if colour.alphaComponent > 0.1 && luminance < 0.8 { ink += 1 }
            }
        }
        #expect(ink > 100, "the diagram drew almost nothing — \(ink) inked samples")
        // The labelled diagram must carry strictly more dark ink than the same
        // diagram with empty nodes, at the same drawn size. Text is the only
        // difference between them.
        #expect(
            darkInk(labelled, side: 400) > darkInk(bare, side: 400),
            "the labels did not draw"
        )
    }

    /// Diagrams that render at the same time must not swap pictures.
    ///
    /// One web view serves every diagram, and a render is two separate awaits
    /// against it: `__rapidRender` draws into `#stage` and measures, then
    /// `takeSnapshot` reads the pixels back. Nothing between those two points
    /// stops a second diagram from starting, replacing the stage, and leaving
    /// the first one to photograph the second one's drawing.
    ///
    /// This is the ordinary case rather than a rare one: an answer with four
    /// diagrams settles them on a single flush, so all four start together.
    /// The symptom is a block showing a neighbour's picture, at its own
    /// block's height.
    ///
    /// Compared by pixels, not by size: `render` assigns the snapshot the
    /// size it measured, so a swapped picture still reports its own block's
    /// dimensions. A size-based version of this test passed against the bug.
    @Test("Concurrent diagrams keep their own pictures")
    @MainActor
    func concurrentRendersDoNotSwap() async throws {
        let diagrams = [
            "pie title Share\n  \"A\" : 40\n  \"B\" : 60",
            "graph TD\n  A[Order placed] --> B{In stock?}\n  B -->|yes| C[Reserve]\n  B -->|no| D[Notify]",
            "sequenceDiagram\n  Alice->>Bob: hello there\n  Bob-->>Alice: hi back",
            "classDiagram\n  Animal <|-- Duck\n  Animal : +int age",
        ]

        // Alone, from cold — the picture each diagram is entitled to.
        renderer.resetForTesting()
        var alone: [String] = []
        for source in diagrams {
            alone.append(fingerprint(try #require(
                await renderer.image(source: source, theme: .light)
            )))
        }
        // The premise: these four actually look different. Without it the
        // comparison below could pass on four identical pictures.
        #expect(Set(alone).count == diagrams.count,
                "the fixtures are not visually distinct, so a swap would be invisible")

        // Together, from cold. Unstructured tasks so they are genuinely in
        // flight at once, which a sequential loop would not be.
        renderer.resetForTesting()
        let started = diagrams.map { source in
            Task { @MainActor () -> NSImage? in
                await renderer.image(source: source, theme: .light)
            }
        }
        var together: [String] = []
        for task in started {
            together.append(fingerprint(try #require(await task.value)))
        }

        for index in diagrams.indices {
            #expect(
                together[index] == alone[index],
                "diagram \(index) photographed a different diagram"
            )
        }
    }

    /// A coarse picture of the picture: dark or not, on a fixed grid. Enough
    /// to tell two diagrams apart, coarse enough not to notice antialiasing.
    @MainActor
    private func fingerprint(_ image: NSImage, side: Int = 48) -> String {
        guard let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: side, pixelsHigh: side,
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
        ) else { return "" }
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        NSColor.white.setFill()
        NSRect(x: 0, y: 0, width: CGFloat(side), height: CGFloat(side)).fill()
        image.draw(in: NSRect(x: 0, y: 0, width: CGFloat(side), height: CGFloat(side)))
        NSGraphicsContext.restoreGraphicsState()

        var bits = ""
        for y in 0..<side {
            for x in 0..<side {
                guard let colour = rep.colorAt(x: x, y: y) else { bits += "0"; continue }
                let luminance = 0.3 * colour.redComponent
                    + 0.59 * colour.greenComponent + 0.11 * colour.blueComponent
                bits += luminance < 0.75 ? "1" : "0"
            }
        }
        return bits
    }

    /// Dark samples at a fixed size — dark enough to mean glyphs rather than
    /// the pale node fill Mermaid's default theme uses.
    private func darkInk(_ image: NSImage, side: Int) -> Int {
        guard let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: side, pixelsHigh: side,
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
        ) else { return 0 }
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        NSColor.white.setFill()
        NSRect(x: 0, y: 0, width: side, height: side).fill()
        image.draw(in: NSRect(x: 0, y: 0, width: side, height: side))
        NSGraphicsContext.restoreGraphicsState()

        var ink = 0
        for x in stride(from: 0, to: side, by: 2) {
            for y in stride(from: 0, to: side, by: 2) {
                guard let colour = rep.colorAt(x: x, y: y) else { continue }
                let luminance = 0.3 * colour.redComponent
                    + 0.59 * colour.greenComponent + 0.11 * colour.blueComponent
                if colour.alphaComponent > 0.1 && luminance < 0.35 { ink += 1 }
            }
        }
        return ink
    }

    @Test("A malformed diagram renders nothing and is remembered")
    func malformedIsRefusedAndCached() async {
        let bad = "graph TD\n  A --> ((("
        #expect(await renderer.image(source: bad, theme: .light) == nil)
        #expect(renderer.isKnownBad(source: bad, theme: .light))
        // Remembered, so a rebuilt row does not pay for it again.
        #expect(renderer.cachedImage(source: bad, theme: .light) == nil)
    }

    /// The theme is part of the cache key. Conflating them is the mistake the
    /// inline-formula cache made: `NSColor.labelColor` compares equal in both
    /// appearances, so the light bitmap was served back in dark.
    @Test("Light and dark are cached apart")
    func themesAreCachedApart() async throws {
        let source = "graph TD\n  A[Theme] --> B[Test]"
        let light = try #require(await renderer.image(source: source, theme: .light))
        let dark = try #require(await renderer.image(source: source, theme: .dark))
        #expect(light !== dark)
        #expect(renderer.cachedImage(source: source, theme: .light) === light)
        #expect(renderer.cachedImage(source: source, theme: .dark) === dark)
    }
}

/// Auto-reveal, and the two rules that keep it from being annoying.
@MainActor
@Suite("Preview auto-reveal", .serialized)
struct PreviewAutoRevealTests {

    /// The shared renderer, deliberately, and the only suite that still uses
    /// it. These tests drive `MarkdownCodeBlockView`, which reaches for
    /// ``MermaidRenderer/shared`` itself, so an injected instance would sit
    /// there unwritten while the assertions read an empty cache. Their
    /// neighbours own private instances, which leaves nobody to race with;
    /// `.serialized` covers this suite's own tests, which share the cache.
    private let renderer = MermaidRenderer.shared

    private let svg = """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 50" width="100" height="50">
          <rect width="100" height="50" fill="teal"/>
        </svg>
        """

    private func block(_ code: String, _ language: String?, isFinal: Bool = true)
        -> MarkdownCodeBlockView {
        let view = MarkdownCodeBlockView(options: MarkdownOptions())
        // Keep the cache assertions below independent of the host's current
        // appearance. CI and developer Macs may run in either mode.
        view.appearance = NSAppearance(named: .aqua)
        view.frame = NSRect(x: 0, y: 0, width: 400, height: 200)
        view.configure(
            code: code, language: language, options: MarkdownOptions(), isFinal: isFinal
        )
        return view
    }

    private func previewButton(_ view: NSView) -> NSButton? {
        view.subviews.compactMap { $0 as? NSButton }
            .first { $0.accessibilityIdentifier() == "CodeBlock.Preview" }
    }

    /// The picture a reader asked for by writing a diagram is not asked for
    /// twice. The button then offers the source, so it reads "Code".
    @Test("A renderable document opens as its picture")
    func opensAsPicture() throws {
        let view = block(svg, "svg")
        #expect(previewButton(view)?.title == "Code")
        // The card is the picture's height, not the source's.
        let sourceOnly = block("print(\"hi\")", "swift")
        #expect(view.height(forWidth: 400) != sourceOnly.height(forWidth: 400))
    }

    /// Auto-reveal is a default, not an override: closing it has to stick,
    /// including across the re-configure that every streaming flush performs.
    @Test("Closing the preview sticks")
    func closingSticks() throws {
        let view = block(svg, "svg")
        let button = try #require(previewButton(view))
        _ = button.target?.perform(button.action, with: button)
        #expect(button.title == "Preview")
        let closedHeight = view.height(forWidth: 400)

        view.configure(code: svg, language: "svg", options: MarkdownOptions())
        #expect(button.title == "Preview", "a re-configure reopened what the reader closed")
        #expect(view.height(forWidth: 400) == closedHeight)
    }

    /// A diagram is only drawn once it has stopped being rewritten. Without
    /// this, every prefix of a streaming diagram is a separate render — ten a
    /// second, each failing, each taking a cache slot.
    ///
    /// The renderer is warmed first, and the wait is generous. The previous
    /// version slept 300 ms on a possibly-cold renderer, which is shorter
    /// than a first render takes — so with the gate deleted it still passed
    /// when run alone, and only failed when another suite had happened to
    /// warm things up. Swift Testing does not promise an order, so that made
    /// the gate effectively unguarded.
    @Test("A diagram is not drawn while it is still being written")
    func noRenderWhileStreaming() async {
        // Warm: after this, a render of a *final* block is fast, so a wait
        // that finds nothing means nothing was asked for.
        _ = await renderer.image(
            source: "graph TD\n  Warm[Warm] --> Up[Up]", theme: .light
        )
        let source = "graph TD\n  A[Streaming] --> B[Partial]"
        _ = block(source, "mermaid", isFinal: false)
        try? await Task.sleep(for: .milliseconds(1_500))
        #expect(renderer.cachedImage(source: source, theme: .light) == nil)
        #expect(!renderer.isKnownBad(source: source, theme: .light))

        // And the control: the same block, final, does get drawn.
        _ = block(source, "mermaid", isFinal: true)
        try? await Task.sleep(for: .milliseconds(1_500))
        #expect(renderer.cachedImage(source: source, theme: .light) != nil)
    }

    @Test("Changing appearance requests a matching diagram theme")
    func appearanceChangeRendersMatchingTheme() async throws {
        let source = "graph TD\n  Light --> Dark"
        let view = block(source, "mermaid")
        try? await Task.sleep(for: .milliseconds(1_500))
        let light = try #require(renderer.cachedImage(source: source, theme: .light))

        view.appearance = NSAppearance(named: .darkAqua)
        view.viewDidChangeEffectiveAppearance()
        try? await Task.sleep(for: .milliseconds(1_500))
        let dark = try #require(renderer.cachedImage(source: source, theme: .dark))
        #expect(light !== dark)
    }

    @Test("Reusing a row reclassifies identical source")
    func reusedRowReclassifiesIdenticalSource() async throws {
        let source = "graph TD\n  Same --> Source"
        let view = block(source, "mermaid")
        try? await Task.sleep(for: .milliseconds(1_500))
        let button = try #require(previewButton(view))
        #expect(!button.isHidden)

        view.configure(
            code: source, language: "swift", options: MarkdownOptions(), isFinal: true
        )
        #expect(button.isHidden)

        view.configure(
            code: source, language: "mermaid", options: MarkdownOptions(), isFinal: true
        )
        #expect(!button.isHidden)
        #expect(button.title == "Code")
    }

    @Test("A replacement diagram receives the default auto-reveal")
    func replacementDiagramAutoReveals() async throws {
        let first = "graph TD\n  First --> Closed"
        let second = "graph TD\n  Second --> Opens"
        let view = block(first, "mermaid")
        try? await Task.sleep(for: .milliseconds(1_500))
        let button = try #require(previewButton(view))
        _ = button.target?.perform(button.action, with: button)
        #expect(button.title == "Preview")

        view.configure(
            code: second, language: "mermaid", options: MarkdownOptions(), isFinal: true
        )
        try? await Task.sleep(for: .milliseconds(1_500))
        #expect(button.title == "Code")
        #expect(!button.isHidden)
    }
}

/// One web view, however many diagrams arrive at once.
///
/// `@MainActor` does not stop reentrancy at a suspension point, and setting
/// the renderer up has three of them. Without a shared setup task, a message
/// carrying several diagrams that go final together built one web content
/// process each — and since the host window and the navigation policy are
/// single slots, and `WKWebView.navigationDelegate` is weak, all but the last
/// ended up with no window and no navigation delegate. A layer of the network
/// defence disappearing is not merely wasteful.
@MainActor
@Suite("Mermaid renderer setup")
struct MermaidSetupTests {

    /// This suite's own renderer. Swift Testing builds a `struct` suite once
    /// per test, so every test gets a clean cache, failure budget and web
    /// view, and no suite can reset another's while it is mid-render.
    private let renderer = MermaidRenderer()

    @Test("Initial navigation has a deadline")
    func initialNavigationHasDeadline() async {
        let navigation = MermaidNavigationPolicy()
        let started = ContinuousClock.now
        let loaded = await navigation.waitForLoad(timeout: .milliseconds(100))
        #expect(!loaded)
        #expect(ContinuousClock.now - started < .seconds(2))
    }

    @Test("Content-process termination settles initial navigation")
    func contentProcessTerminationSettlesNavigation() async {
        var reportedTermination = false
        let navigation = MermaidNavigationPolicy {
            reportedTermination = true
        }
        async let loaded = navigation.waitForLoad(timeout: .seconds(5))
        navigation.webViewWebContentProcessDidTerminate(WKWebView())
        let didLoad = await loaded
        #expect(!didLoad)
        #expect(reportedTermination)
    }

    @Test("Concurrent first renders share one web view")
    func concurrentFirstRendersShareOneWebView() async {
        renderer.resetForTesting()
        defer { renderer.resetForTesting() }
        #expect(renderer.webViewsCreated == 0)

        // Distinct sources, so `inFlight` cannot be what dedups them.
        let sources = (1...6).map { "graph TD\n  A\($0)[Node \($0)] --> B\($0)[End \($0)]" }
        // `async let`, not a task group: the group's sending checks reject a
        // `@MainActor` child here, and what matters is only that six requests
        // are outstanding together.
        async let a: NSImage? = renderer.image(source: sources[0], theme: .light)
        async let b: NSImage? = renderer.image(source: sources[1], theme: .light)
        async let c: NSImage? = renderer.image(source: sources[2], theme: .light)
        async let d: NSImage? = renderer.image(source: sources[3], theme: .light)
        async let e: NSImage? = renderer.image(source: sources[4], theme: .light)
        async let f: NSImage? = renderer.image(source: sources[5], theme: .light)
        _ = await (a, b, c, d, e, f)
        #expect(
            renderer.webViewsCreated == 1,
            "built \(renderer.webViewsCreated) web views for six diagrams"
        )
        // And they all actually rendered.
        for source in sources {
            #expect(renderer.cachedImage(source: source, theme: .light) != nil)
        }
    }
}
