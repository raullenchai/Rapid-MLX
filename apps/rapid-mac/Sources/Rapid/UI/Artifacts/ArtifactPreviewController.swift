import AppKit
import Observation
import WebKit

/// Owns one sandboxed web view for one artifact, and everything that can go
/// wrong with it.
///
/// The web view is built lazily and only after the content rule list has
/// compiled, so a failure there leaves no page at all rather than an
/// unprotected one. The content process is out of process, so a runaway script
/// cannot hang the app; what the watchdogs add is noticing it and giving the
/// reader a way out. Every failure tears the view down, which ends the process.
@MainActor
@Observable
final class ArtifactPreviewController: NSObject {

    enum Failure: Equatable {
        case sandboxUnavailable
        case loadFailed
        case timedOut
        case unresponsive
        case contentProcessTerminated
    }

    enum State: Equatable {
        case idle
        case loading
        case ready
        case failed(Failure)
    }

    private(set) var state: State = .idle

    @ObservationIgnored let hostView = NSView()
    @ObservationIgnored private(set) var webView: WKWebView?
    /// Test seam: how many web views this controller has built.
    @ObservationIgnored private(set) var webViewsCreated = 0

    @ObservationIgnored private let source: String
    @ObservationIgnored private let loadTimeout: Duration
    @ObservationIgnored private let pingInterval: Duration
    @ObservationIgnored private let pingTimeout: Duration
    @ObservationIgnored private var delegate: ArtifactWebDelegate?
    @ObservationIgnored private var startTask: Task<Void, Never>?
    @ObservationIgnored private var watchdog: Task<Void, Never>?
    @ObservationIgnored private var generation = 0

    init(
        source: String,
        loadTimeout: Duration = .seconds(10),
        pingInterval: Duration = .seconds(5),
        pingTimeout: Duration = .seconds(10)
    ) {
        self.source = source
        self.loadTimeout = loadTimeout
        self.pingInterval = pingInterval
        self.pingTimeout = pingTimeout
        super.init()
    }

    /// Load the page. A no-op while one is loading or showing.
    func start() {
        guard startTask == nil, webView == nil else { return }
        state = .loading
        generation += 1
        let current = generation
        startTask = Task { [weak self] in
            let ruleList = await ArtifactSandbox.compiledRuleList()
            guard let self, self.generation == current else { return }
            self.startTask = nil
            guard let ruleList else {
                self.fail(.sandboxUnavailable)
                return
            }
            self.build(ruleList: ruleList)
        }
    }

    /// Throw the page away and load it again in a new web view.
    func reload() {
        teardown()
        start()
    }

    /// Stop everything and release the web view, which ends its process.
    func teardown() {
        generation += 1
        startTask?.cancel()
        startTask = nil
        watchdog?.cancel()
        watchdog = nil
        discardWebView()
        state = .idle
    }

    private func build(ruleList: WKContentRuleList) {
        let configuration = ArtifactSandbox.makeConfiguration(
            source: source, ruleList: ruleList
        )
        let view = WKWebView(frame: hostView.bounds, configuration: configuration)
        view.autoresizingMask = [.width, .height]
        view.isInspectable = false
        let delegate = ArtifactWebDelegate { [weak self] event in self?.handle(event) }
        view.navigationDelegate = delegate
        view.uiDelegate = delegate
        self.delegate = delegate
        hostView.addSubview(view)
        webView = view
        webViewsCreated += 1

        view.load(URLRequest(url: ArtifactSandbox.documentURL))
        startWatchdog()
    }

    private func handle(_ event: ArtifactWebDelegate.Event) {
        guard webView != nil else { return }
        switch event {
        case .didFinish:
            if state == .loading { state = .ready }
        case .didFail:
            fail(.loadFailed)
        case .contentProcessTerminated:
            fail(.contentProcessTerminated)
        }
    }

    private func fail(_ failure: Failure) {
        watchdog?.cancel()
        watchdog = nil
        discardWebView()
        state = .failed(failure)
    }

    private func discardWebView() {
        webView?.stopLoading()
        webView?.navigationDelegate = nil
        webView?.uiDelegate = nil
        webView?.removeFromSuperview()
        webView = nil
        delegate = nil
    }

    // MARK: Watchdog

    /// Covers the load, then pings the page. A page that never finishes
    /// loading, or stops answering after it has, is torn down. The page's own
    /// script is the only thing that can stall either, and it runs in another
    /// process, so this loop is never blocked by it.
    private func startWatchdog() {
        watchdog?.cancel()
        watchdog = Task { [weak self, loadTimeout, pingInterval] in
            try? await Task.sleep(for: loadTimeout)
            guard let self, !Task.isCancelled else { return }
            if self.state == .loading {
                self.fail(.timedOut)
                return
            }
            while !Task.isCancelled {
                try? await Task.sleep(for: pingInterval)
                guard !Task.isCancelled, let view = self.webView, self.state == .ready
                else { return }
                if await self.answers(view) == false {
                    self.fail(.unresponsive)
                    return
                }
            }
        }
    }

    private func answers(_ view: WKWebView) async -> Bool {
        let gate = ArtifactOneShot()
        let timeout = pingTimeout
        return await withCheckedContinuation { continuation in
            view.evaluateJavaScript("1") { _, _ in
                if gate.claim() { continuation.resume(returning: true) }
            }
            Task {
                try? await Task.sleep(for: timeout)
                if gate.claim() { continuation.resume(returning: false) }
            }
        }
    }
}

/// Exactly one of the page's reply and the deadline resumes the continuation.
private final class ArtifactOneShot: @unchecked Sendable {
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
