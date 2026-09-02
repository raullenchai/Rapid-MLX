import AppKit
import WebKit

/// Draws a Mermaid diagram, offscreen, and hands back a picture.
///
/// ## Why a web view at all, when the SVG preview needs none
///
/// ``SVGPreview`` argues — correctly, for the SVG a person or a model writes
/// by hand — that AppKit renders SVG natively and a web view would be a much
/// larger machine for no gain. Mermaid's output is a different animal.
/// Measured against this exact library:
///
/// * `<foreignObject>` labels, which Mermaid uses for flowchart, class,
///   state, ER, mindmap and journey diagrams, are **not drawn at all** by
///   AppKit. The diagram comes out as empty boxes.
/// * With `htmlLabels: false` the labels become native `<text>` and do draw,
///   but land outside their shapes — AppKit does not honour the
///   text-positioning attributes Mermaid relies on.
/// * `marker` arrowheads are never drawn, in either mode.
///
/// So the picture has to come from the engine that agrees with the emitter.
/// WebKit is that engine.
///
/// ## Why it is still not a live web view
///
/// The web view never enters the view hierarchy. It renders into an offscreen
/// window, is snapshotted, and the result is an `NSImage` drawn by the same
/// `draw(_:)` path the SVG preview uses. That keeps every property that made
/// the SVG preview affordable: no accessibility subtree (so the 26 committed
/// golden-flow baselines are untouched), no participation in layout, no
/// asynchronous height arriving after `height(forWidth:)` has already
/// answered, and nothing to tear down when the transcript rebuilds a row.
///
/// ## Network
///
/// Four layers, none of which is redundant — see ``MermaidHostPage``. Two of
/// them are directly measured by `MermaidNetworkDenialTests`, which stands up
/// a loopback listener and asserts nothing ever connects to it. The listener
/// has a positive control, so an assertion that cannot fail is not mistaken
/// for one that passes.
@MainActor
final class MermaidRenderer {

    static let shared = MermaidRenderer()

    enum Theme: String, Hashable {
        case light
        case dark

        init(_ appearance: NSAppearance) {
            let match = appearance.bestMatch(from: [.aqua, .darkAqua])
            self = (match == .darkAqua) ? .dark : .light
        }
    }

    typealias JavaScriptEvaluator = @MainActor (
        _ webView: WKWebView,
        _ source: String,
        _ theme: Theme,
        _ completion: @escaping @MainActor @Sendable (Result<Any, Error>) -> Void
    ) -> Void
    typealias Snapshotter = @MainActor (
        _ webView: WKWebView,
        _ configuration: WKSnapshotConfiguration,
        _ completion: @escaping @MainActor @Sendable (NSImage?, Error?) -> Void
    ) -> Void
    typealias ReadinessEvaluator = @MainActor (
        _ webView: WKWebView,
        _ completion: @escaping @MainActor @Sendable (Result<Any, Error>) -> Void
    ) -> Void

    /// A preview is drawn at 2×. Bounding both axes and point area keeps a
    /// model-authored diagram from turning into an unbounded bitmap request.
    nonisolated static let maximumPointDimension = 4_096
    nonisolated static let maximumPointArea = 4_000_000

    private let renderTimeout: Duration
    private let readinessEvaluator: ReadinessEvaluator?
    private let javaScriptEvaluator: JavaScriptEvaluator?
    private let snapshotter: Snapshotter?
    private var abortActiveOperation: (@MainActor () -> Void)?

    private struct Key: Hashable {
        let source: String
        let theme: Theme
    }

    private enum Entry {
        case image(NSImage)
        /// Remembered as firmly as a success. Without this a malformed
        /// diagram is retried on every appearance change and every rebuilt
        /// row, forever.
        case failed
    }

    /// `Task.value` requires a Sendable payload even though this renderer and
    /// every image consumer are main-actor isolated. This wrapper expresses
    /// that narrow ownership invariant without claiming NSImage is generally
    /// safe to move between executors.
    private struct RenderedImage: @unchecked Sendable {
        let value: NSImage
    }

    private enum RenderOutcome: Sendable {
        case image(RenderedImage)
        case sourceRejected
        case infrastructureFailure
    }

    /// Bounded by both entry count and retained bitmap cost. A single accepted
    /// 2× snapshot can occupy about 64 MB, so count alone is not a memory
    /// bound for a model response containing many large diagrams.
    private static let capacity = 32
    private let cacheByteLimit: Int
    private var cache: [Key: Entry] = [:]
    private var order: [Key] = []
    private(set) var cachedImageBytes = 0
    private var inFlight: [Key: Task<RenderedImage?, Never>] = [:]

    /// Rendering happens in a window so WebKit has somewhere to draw. It is
    /// never ordered front and never joins the app's window list in any way
    /// the reader can see.
    private var window: NSWindow?
    private var webView: WKWebView?
    private var loaded = false
    /// The setup, shared by everyone who arrives while it is running.
    ///
    /// `@MainActor` does not stop reentrancy at a suspension point, and
    /// `preparedWebView()` has three: compiling the rule list, waiting for the
    /// page, and probing it. Without this, eight diagrams going final on the
    /// same flush built eight web content processes, compiled eight rule
    /// lists, and read and hashed the 3.4 MB library eight times — and since
    /// `window` and `navigationPolicy` are single slots and
    /// `WKWebView.navigationDelegate` is weak, seven of the eight ended up
    /// with no host window and no navigation delegate at all. The fourth
    /// layer of the network defence silently absent is not a cost, it is a
    /// hole. Same shape as `inFlight` below, for the same reason.
    private var setup: Task<WKWebView?, Never>?

    /// How many web views this renderer has built. A test seam: the whole
    /// point of ``setup`` is that this stays at one however many diagrams
    /// arrive together, and there is no other way to see it.
    private(set) var webViewsCreated = 0

    /// After this many hard failures the feature turns itself off for the
    /// session. A diagram that wedges or crashes the content process must not
    /// be able to make the app respawn it in a loop.
    private static let failureBudget = 3
    private var failures = 0

    /// The app uses ``shared`` and nothing else. This is reachable so tests
    /// can own an instance outright: the cache, the failure budget and the web
    /// view are all mutable state on the object, and suites sharing one
    /// instance under Swift Testing's default parallelism read each other's
    /// resets. That surfaced as "Light and dark are cached apart" failing only
    /// when run alongside its neighbours.
    init(
        renderTimeout: Duration = .seconds(8),
        readinessEvaluator: ReadinessEvaluator? = nil,
        javaScriptEvaluator: JavaScriptEvaluator? = nil,
        snapshotter: Snapshotter? = nil,
        // Keep this a literal. Swift 6.1 can crash in SILGen when a default
        // argument reaches through this @MainActor type to a static product.
        cacheByteLimit: Int = 268_435_456
    ) {
        self.renderTimeout = renderTimeout
        self.readinessEvaluator = readinessEvaluator
        self.javaScriptEvaluator = javaScriptEvaluator
        self.snapshotter = snapshotter
        self.cacheByteLimit = max(0, cacheByteLimit)
    }

    // MARK: - The synchronous half

    /// The picture, if it has already been drawn. This is what makes the
    /// button's press synchronous: by the time it is visible, the answer is
    /// in hand.
    func cachedImage(source: String, theme: Theme) -> NSImage? {
        guard case .image(let image)? = cache[Key(source: source, theme: theme)] else {
            return nil
        }
        return image
    }

    /// Has this diagram already been tried and failed? Callers use it to stop
    /// asking.
    func isKnownBad(source: String, theme: Theme) -> Bool {
        if case .failed? = cache[Key(source: source, theme: theme)] { return true }
        return false
    }

    // MARK: - The asynchronous half

    /// Draw it, or return nil.
    ///
    /// Concurrent callers for the same diagram share one render — a
    /// conversation holding the same diagram twice pays once.
    func image(source: String, theme: Theme) async -> NSImage? {
        let key = Key(source: source, theme: theme)
        switch cache[key] {
        case .image(let image): return image
        case .failed: return nil
        case nil: break
        }
        if let running = inFlight[key] { return await running.value?.value }

        let task = Task { [weak self] () -> RenderedImage? in
            guard let self else { return nil }
            let outcome = await self.render(source: source, theme: theme)
            let image: RenderedImage?
            switch outcome {
            case .image(let rendered):
                self.store(.image(rendered.value), for: key)
                image = rendered
            case .sourceRejected:
                self.store(.failed, for: key)
                image = nil
            case .infrastructureFailure:
                // A rebuilt WebKit process may succeed on the next attempt.
                // Do not turn a host failure into a permanent source verdict.
                image = nil
            }
            self.inFlight[key] = nil
            return image
        }
        inFlight[key] = task
        return await task.value?.value
    }

    private func store(_ entry: Entry, for key: Key) {
        if let replaced = cache.removeValue(forKey: key) {
            cachedImageBytes -= Self.cacheCost(of: replaced)
            order.removeAll { $0 == key }
        }
        cache[key] = entry
        order.append(key)
        cachedImageBytes += Self.cacheCost(of: entry)
        while order.count > Self.capacity || cachedImageBytes > cacheByteLimit {
            guard !order.isEmpty else { break }
            let evicted = order.removeFirst()
            if let removed = cache.removeValue(forKey: evicted) {
                cachedImageBytes -= Self.cacheCost(of: removed)
            }
        }
    }

    private static func cacheCost(of entry: Entry) -> Int {
        guard case .image(let image) = entry else { return 0 }
        let representationBytes = image.representations.compactMap { representation -> Int? in
            guard representation.pixelsWide > 0, representation.pixelsHigh > 0 else {
                return nil
            }
            let (pixels, overflow) = representation.pixelsWide.multipliedReportingOverflow(
                by: representation.pixelsHigh
            )
            guard !overflow else { return Int.max }
            let (bytes, byteOverflow) = pixels.multipliedReportingOverflow(by: 4)
            return byteOverflow ? Int.max : bytes
        }.max()
        if let representationBytes { return representationBytes }

        // Injected/test images can have no bitmap representation. Production
        // snapshots are 2×, so preserve the same conservative cost model.
        let width = max(0, Int(ceil(image.size.width * 2)))
        let height = max(0, Int(ceil(image.size.height * 2)))
        let (pixels, overflow) = width.multipliedReportingOverflow(by: height)
        guard !overflow else { return Int.max }
        let (bytes, byteOverflow) = pixels.multipliedReportingOverflow(by: 4)
        return byteOverflow ? Int.max : bytes
    }

    // MARK: - The web view

    /// The tail of the render queue.
    ///
    /// One web view serves every diagram, and a render is two separate awaits
    /// against it: `__rapidRender` draws into `#stage` and measures it, then
    /// `takeSnapshot` reads the pixels back. Nothing between those two points
    /// stopped a second diagram from starting, replacing the stage, and
    /// leaving the first to photograph the second one's drawing.
    ///
    /// An answer with four diagrams settles them on one flush, so all four
    /// start together — measured, three of the four came back holding a
    /// neighbour's picture, on every run. The snapshot is also assigned the
    /// size that *was* measured, so the wrong picture still arrives at the
    /// right block's dimensions: it looks like a diagram that rendered as
    /// something else, cropped.
    private var tail: Task<Void, Never>?

    private func render(source: String, theme: Theme) async -> RenderOutcome {
        let previous = tail
        let task = Task { @MainActor [weak self] () -> RenderOutcome in
            _ = await previous?.value
            guard let self else { return .infrastructureFailure }
            return await self.renderExclusively(source: source, theme: theme)
        }
        tail = Task { @MainActor in _ = await task.value }
        return await task.value
    }

    /// Draws one diagram, with the web view to itself. Only ever called from
    /// ``render(source:theme:)``, which is what guarantees that.
    private func renderExclusively(source: String, theme: Theme) async -> RenderOutcome {
        guard failures < Self.failureBudget else { return .infrastructureFailure }
        guard let webView = await preparedWebView() else { return .infrastructureFailure }

        let measured: MermaidRenderMeasurement?
        do {
            measured = try await evaluateRender(
                in: webView, source: source, theme: theme
            )
        } catch {
            // A wedged render leaves the content process unusable; drop the
            // whole thing so the next diagram starts clean.
            failures += 1
            teardown()
            return .infrastructureFailure
        }
        guard let measured,
              Self.acceptsSnapshot(width: measured.width, height: measured.height)
        else { return .sourceRejected }

        do {
            let image = try await snapshot(
                webView: webView, measurement: measured
            )
            return .image(image)
        } catch {
            failures += 1
            teardown()
            return .infrastructureFailure
        }
    }

    nonisolated static func acceptsSnapshot(width: Int, height: Int) -> Bool {
        guard width > 0, height > 0,
              width <= maximumPointDimension,
              height <= maximumPointDimension
        else { return false }
        let (area, overflow) = width.multipliedReportingOverflow(by: height)
        return !overflow && area <= maximumPointArea
    }

    /// Evaluate model-authored diagram source behind a hard deadline. WebKit
    /// does not guarantee that cancelling its async wrapper interrupts a
    /// wedged content process, so this callback bridge resumes independently;
    /// the caller tears the abandoned view down before the queue advances.
    private func evaluateRender(
        in webView: WKWebView,
        source: String,
        theme: Theme
    ) async throws -> MermaidRenderMeasurement? {
        let gate = MermaidEvaluationGate()
        return try await withCheckedThrowingContinuation {
            (continuation: CheckedContinuation<MermaidRenderMeasurement?, Error>) in
            let completion: @MainActor @Sendable (Result<Any, Error>) -> Void = { result in
                guard gate.claim() else { return }
                self.abortActiveOperation = nil
                switch result {
                case .success(let value):
                    guard let dictionary = value as? [String: Any],
                          dictionary["ok"] as? Bool == true,
                          let width = dictionary["width"] as? Int,
                          let height = dictionary["height"] as? Int
                    else {
                        continuation.resume(returning: nil)
                        return
                    }
                    continuation.resume(returning: MermaidRenderMeasurement(
                        width: width, height: height
                    ))
                case .failure:
                    continuation.resume(throwing: MermaidRenderError.evaluationFailed)
                }
            }

            abortActiveOperation = {
                guard gate.claim() else { return }
                continuation.resume(throwing: MermaidRenderError.contentProcessTerminated)
            }

            if let javaScriptEvaluator {
                javaScriptEvaluator(webView, source, theme, completion)
            } else {
                webView.callAsyncJavaScript(
                    "return await __rapidRender(source, theme);",
                    // Arguments, never interpolation: the source is
                    // model-authored and must not be able to become code.
                    arguments: ["source": source, "theme": theme.rawValue],
                    in: nil,
                    in: .page,
                    completionHandler: completion
                )
            }

            Task { [renderTimeout] in
                try? await Task.sleep(for: renderTimeout)
                guard gate.claim() else { return }
                self.abortActiveOperation = nil
                continuation.resume(throwing: MermaidRenderError.timedOut)
            }
        }
    }

    private func snapshot(
        webView: WKWebView,
        measurement: MermaidRenderMeasurement
    ) async throws -> RenderedImage {
        let surfaceSize = CGSize(width: measurement.width, height: measurement.height)
        // `WKSnapshotConfiguration.rect` is clipped to the view's drawable
        // surface. The host starts small to keep idle WebKit cheap, then grows
        // only after the measured dimensions pass the allocation bounds.
        webView.frame = CGRect(origin: .zero, size: surfaceSize)
        window?.setContentSize(surfaceSize)
        webView.layoutSubtreeIfNeeded()

        let configuration = WKSnapshotConfiguration()
        configuration.rect = CGRect(
            x: 0, y: 0, width: measurement.width, height: measurement.height
        )
        // Twice the point size: the result is a bitmap, and a preview that
        // was crisp only on a non-Retina display would be a regression on
        // every Mac this app supports.
        configuration.snapshotWidth = NSNumber(value: measurement.width * 2)

        let gate = MermaidEvaluationGate()
        return try await withCheckedThrowingContinuation {
            (continuation: CheckedContinuation<RenderedImage, Error>) in
            let completion: @MainActor @Sendable (NSImage?, Error?) -> Void = {
                image, error in
                guard gate.claim() else { return }
                self.abortActiveOperation = nil
                if error != nil {
                    continuation.resume(throwing: MermaidRenderError.snapshotFailed)
                    return
                }
                guard let image else {
                    continuation.resume(throwing: MermaidRenderError.snapshotFailed)
                    return
                }
                // The snapshot arrives at pixel dimensions; restate it in
                // points so `SVGPreview.drawSize` measures it like SVG.
                image.size = CGSize(width: measurement.width, height: measurement.height)
                continuation.resume(returning: RenderedImage(value: image))
            }

            abortActiveOperation = {
                guard gate.claim() else { return }
                continuation.resume(throwing: MermaidRenderError.contentProcessTerminated)
            }

            if let snapshotter {
                snapshotter(webView, configuration, completion)
            } else {
                webView.takeSnapshot(with: configuration, completionHandler: completion)
            }

            Task { [renderTimeout] in
                try? await Task.sleep(for: renderTimeout)
                guard gate.claim() else { return }
                self.abortActiveOperation = nil
                continuation.resume(throwing: MermaidRenderError.timedOut)
            }
        }
    }

    private func preparedWebView() async -> WKWebView? {
        if let webView, loaded { return webView }
        if let setup { return await setup.value }

        let task = Task { [weak self] () -> WKWebView? in
            guard let self else { return nil }
            let view = await self.buildWebView()
            self.setup = nil
            return view
        }
        setup = task
        return await task.value
    }

    private func buildWebView() async -> WKWebView? {
        webViewsCreated += 1
        guard let ruleList = await compiledRuleList() else {
            // Setup failure counts against the budget too. Without this every
            // distinct diagram re-attempts the whole thing — recompiling the
            // rule list, re-reading and re-hashing 3.4 MB — forever.
            failures += 1
            // Fail closed. Without the rule list a subresource load would go
            // straight to the network, and a preview that is silently
            // unprotected is worse than one that is silently unavailable.
            return nil
        }
        guard let library = MermaidLibrary.load() else {
            failures += 1
            return nil
        }

        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        configuration.setURLSchemeHandler(
            MermaidSchemeHandler(library: library), forURLScheme: MermaidHostPage.scheme
        )
        configuration.userContentController.add(ruleList)
        configuration.preferences.javaScriptCanOpenWindowsAutomatically = false

        let view = WKWebView(
            frame: CGRect(x: 0, y: 0, width: 1_400, height: 1_400),
            configuration: configuration
        )
        let navigation = MermaidNavigationPolicy { [weak self] in
            self?.webContentProcessDidTerminate()
        }
        view.navigationDelegate = navigation
        self.navigationPolicy = navigation

        // WebKit will not paint, and `takeSnapshot` will not produce content,
        // for a view that is in no window. The window is offscreen and never
        // ordered front.
        let host = NSWindow(
            contentRect: view.frame, styleMask: [.borderless],
            backing: .buffered, defer: false
        )
        host.isReleasedWhenClosed = false
        host.contentView?.addSubview(view)
        host.orderOut(nil)
        window = host
        webView = view

        view.load(URLRequest(url: MermaidHostPage.hostPageURL))
        guard await navigation.waitForLoad(timeout: renderTimeout) else {
            failures += 1
            teardown()
            return nil
        }

        // The check on the thing that actually fails: a truncated library
        // resolves by name and then dies inside `render`.
        guard (try? await evaluateReadiness(in: view)) == true else {
            failures += 1
            teardown()
            return nil
        }

        loaded = true
        return view
    }

    /// Probe the host page behind the same hard deadline as rendering. The
    /// async WebKit convenience API can wait forever when the content process
    /// wedges after navigation; a late callback must not retain ``setup`` and
    /// permanently block every later diagram.
    private func evaluateReadiness(in webView: WKWebView) async throws -> Bool {
        let gate = MermaidEvaluationGate()
        return try await withCheckedThrowingContinuation {
            (continuation: CheckedContinuation<Bool, Error>) in
            let completion: @MainActor @Sendable (Result<Any, Error>) -> Void = { result in
                guard gate.claim() else { return }
                self.abortActiveOperation = nil
                switch result {
                case .success(let value):
                    continuation.resume(returning: value as? Bool == true)
                case .failure:
                    continuation.resume(throwing: MermaidRenderError.evaluationFailed)
                }
            }

            abortActiveOperation = {
                guard gate.claim() else { return }
                continuation.resume(throwing: MermaidRenderError.contentProcessTerminated)
            }

            if let readinessEvaluator {
                readinessEvaluator(webView, completion)
            } else {
                webView.evaluateJavaScript("window.__rapidReady === true") { value, error in
                    if let error {
                        completion(.failure(error))
                    } else {
                        completion(.success(value ?? NSNull()))
                    }
                }
            }

            Task { [renderTimeout] in
                try? await Task.sleep(for: renderTimeout)
                guard gate.claim() else { return }
                self.abortActiveOperation = nil
                continuation.resume(throwing: MermaidRenderError.timedOut)
            }
        }
    }

    private var navigationPolicy: MermaidNavigationPolicy?

    private func webContentProcessDidTerminate() {
        // Active render failures are charged by their throwing evaluate/
        // snapshot path; initial-load failures are charged by buildWebView.
        // Between renders neither path runs, so charge that idle crash here.
        if loaded, abortActiveOperation == nil {
            failures += 1
        }
        let abort = abortActiveOperation
        abortActiveOperation = nil
        abort?()
        teardown()
    }

    /// Test seam for the delegate callback while the prepared renderer is idle.
    func simulateIdleContentProcessTerminationForTesting() {
        webContentProcessDidTerminate()
    }

    /// Test seam: put the renderer back to cold. Nothing in the app calls
    /// this — the web view is dropped only when a render wedges it.
    func resetForTesting() {
        teardown()
        cache.removeAll()
        order.removeAll()
        cachedImageBytes = 0
        failures = 0
        webViewsCreated = 0
    }

    private func teardown() {
        abortActiveOperation = nil
        webView?.navigationDelegate = nil
        webView?.removeFromSuperview()
        webView = nil
        window?.close()
        window = nil
        navigationPolicy = nil
        loaded = false
    }

    private func compiledRuleList() async -> WKContentRuleList? {
        await withCheckedContinuation { continuation in
            WKContentRuleListStore.default()?.compileContentRuleList(
                forIdentifier: "rapid-mermaid-deny-all",
                encodedContentRuleList: MermaidHostPage.contentRuleListJSON
            ) { list, _ in continuation.resume(returning: list) }
                ?? continuation.resume(returning: nil)
        }
    }
}

private enum MermaidRenderError: Error {
    case contentProcessTerminated
    case evaluationFailed
    case snapshotFailed
    case timedOut
}

private struct MermaidRenderMeasurement: Sendable {
    let width: Int
    let height: Int
}

/// The JavaScript callback may arrive after the deadline. Exactly one path
/// owns the continuation, even when WebKit replies concurrently with timeout.
private final class MermaidEvaluationGate: @unchecked Sendable {
    private let lock = NSLock()
    private var claimed = false

    func claim() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !claimed else { return false }
        claimed = true
        return true
    }
}

/// Serves exactly two files, by name, from memory. Everything else fails.
private final class MermaidSchemeHandler: NSObject, WKURLSchemeHandler {
    private let library: Data

    init(library: Data) { self.library = library }

    func webView(_ webView: WKWebView, start task: WKURLSchemeTask) {
        guard let url = task.request.url else {
            task.didFailWithError(URLError(.badURL)); return
        }
        let body: Data
        let mime: String
        switch url.path {
        case MermaidHostPage.hostPagePath:
            body = Data(MermaidHostPage.html.utf8)
            mime = "text/html"
        case MermaidHostPage.libraryPath:
            body = library
            mime = "text/javascript"
        default:
            task.didFailWithError(URLError(.unsupportedURL))
            return
        }
        task.didReceive(URLResponse(
            url: url, mimeType: mime,
            expectedContentLength: body.count, textEncodingName: "utf-8"
        ))
        task.didReceive(body)
        task.didFinish()
    }

    func webView(_ webView: WKWebView, stop task: WKURLSchemeTask) {}
}

/// Allows the one page and cancels everything else.
@MainActor
final class MermaidNavigationPolicy: NSObject, WKNavigationDelegate {
    private var continuation: CheckedContinuation<Bool, Never>?
    private var finished: Bool?
    private let onTermination: @MainActor () -> Void

    init(onTermination: @escaping @MainActor () -> Void = {}) {
        self.onTermination = onTermination
    }

    func waitForLoad(timeout: Duration = .seconds(8)) async -> Bool {
        if let finished { return finished }
        return await withCheckedContinuation { continuation in
            self.continuation = continuation
            Task { [weak self] in
                try? await Task.sleep(for: timeout)
                self?.settle(false)
            }
        }
    }

    private func settle(_ value: Bool) {
        guard finished == nil else { return }
        finished = value
        continuation?.resume(returning: value)
        continuation = nil
    }

    @MainActor
    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationAction: WKNavigationAction,
        decisionHandler: @escaping @MainActor @Sendable (WKNavigationActionPolicy) -> Void
    ) {
        decisionHandler(Self.policy(for: navigationAction.request.url))
    }

    /// The decision, as a function of the URL.
    ///
    /// Kept separate so the rule can be asserted directly, cheaply, for every
    /// scheme. It is not a substitute for exercising the installed delegate —
    /// `MermaidNavigationBoundaryTests` loads a real loopback URL through a
    /// real `WKWebView` and asserts a listening server never sees it.
    nonisolated static func policy(for url: URL?) -> WKNavigationActionPolicy {
        url?.scheme == MermaidHostPage.scheme ? .allow : .cancel
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) { settle(true) }

    func webView(
        _ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error
    ) { settle(false) }

    func webView(
        _ webView: WKWebView,
        didFailProvisionalNavigation navigation: WKNavigation!,
        withError error: Error
    ) { settle(false) }

    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        settle(false)
        onTermination()
    }
}
