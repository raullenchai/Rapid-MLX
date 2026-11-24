import Foundation
import WebKit

/// The rules that keep a model-written page from reaching anything.
///
/// Kept in one small file on purpose: everything a security review has to read
/// is here, and a source-guard test asserts there is no network scheme literal
/// in it. The layers are redundant by design:
///
/// 1. A fresh non-persistent data store per load: no cookies, storage or
///    cache survive the page, and none are shared between artifacts.
/// 2. A Content-Security-Policy, as a response header and as a `<meta>` that
///    is the first thing in the document. It is the only layer that covers
///    `connect-src` (fetch, XHR, WebSocket, `sendBeacon`).
/// 3. A content rule list that blocks every URL except our own scheme, `data:`
///    and `blob:`. The scheme handler is only consulted for its own scheme, so
///    without this a subresource on any other scheme goes straight to the
///    network.
/// 4. A navigation delegate that allows the initial document load and nothing
///    else, and a UI delegate that refuses new windows, media capture, file
///    panels and downloads.
/// 5. WebRTC removed before any page script runs. An `RTCPeerConnection` with
///    its own ICE server opens sockets that neither the CSP nor the content
///    rules see, and WebKit offers no switch for it. The load is only shown
///    once the removal is confirmed, otherwise it fails closed.
///
/// There is deliberately no `WKScriptMessageHandler`: the page has no way to
/// call into the app. The one user script only deletes globals.
enum ArtifactSandbox {

    /// A scheme of our own: not `file:`, which can expose the disk, and not
    /// `loadHTMLString`, which lands on an `about:blank` origin.
    static let scheme = "rapid-artifact"
    static let documentPath = "/artifact.html"

    static var documentURL: URL { URL(string: "\(scheme)://local\(documentPath)")! }

    /// Everything that can create or drive a peer connection.
    static let blockedGlobals = [
        "RTCPeerConnection", "webkitRTCPeerConnection", "RTCDataChannel",
        "RTCIceCandidate", "RTCSessionDescription", "RTCCertificate",
        "RTCRtpSender", "RTCRtpReceiver", "RTCRtpTransceiver",
        "RTCDtlsTransport", "RTCIceTransport", "RTCSctpTransport",
    ]

    private static var blockedGlobalsLiteral: String {
        "[" + blockedGlobals.map { "\"\($0)\"" }.joined(separator: ",") + "]"
    }

    /// Runs first in every frame. The properties become non-writable and
    /// non-configurable, so a page cannot put them back, and no other realm is
    /// reachable under the CSP (no frames, workers or objects).
    static var isolationScript: String {
        """
        (function () {
          var names = \(blockedGlobalsLiteral);
          for (var i = 0; i < names.length; i++) {
            try {
              Object.defineProperty(window, names[i], { value: undefined, writable: false, configurable: false });
            } catch (e) {}
          }
        })();
        """
    }

    /// Evaluates to true only if every global above is locked down.
    static var isolationCheck: String {
        """
        (function () {
          var names = \(blockedGlobalsLiteral);
          return names.every(function (n) {
            var d = Object.getOwnPropertyDescriptor(window, n);
            return !!d && d.configurable === false && d.writable === false && d.value === undefined;
          });
        })()
        """
    }

    static let contentSecurityPolicy = [
        "default-src 'none'",
        "script-src 'unsafe-inline'",
        "style-src 'unsafe-inline'",
        "img-src data: blob:",
        "font-src data:",
        "connect-src 'none'",
        "frame-src 'none'",
        "form-action 'none'",
        "base-uri 'none'",
    ].joined(separator: "; ")

    /// Block everything, then re-allow what is ours. Order matters: each
    /// `ignore-previous-rules` has to come after the block. WebKit's filter
    /// grammar has no alternation, hence one rule per allowed prefix.
    static let contentRuleListJSON = """
        [
          { "trigger": { "url-filter": ".*" },
            "action": { "type": "block" } },
          { "trigger": { "url-filter": "^\(scheme)://" },
            "action": { "type": "ignore-previous-rules" } },
          { "trigger": { "url-filter": "^data:" },
            "action": { "type": "ignore-previous-rules" } },
          { "trigger": { "url-filter": "^blob:" },
            "action": { "type": "ignore-previous-rules" } }
        ]
        """

    /// The page as served: the CSP `<meta>` goes in before anything the model
    /// wrote, right after a doctype if there is one (ahead of it would push
    /// the document into quirks mode). The parser files a `<meta>` seen before
    /// `<head>` into an implicit head, and the model's own `<html>` and
    /// `<head>` tags then merge into it.
    static func document(for source: String) -> Data {
        let meta = "<meta http-equiv=\"Content-Security-Policy\" "
            + "content=\"\(contentSecurityPolicy)\">"
        var html = source
        if let doctype = html.range(
            of: #"^\s*<!doctype[^>]*>"#, options: [.regularExpression, .caseInsensitive]
        ) {
            html.insert(contentsOf: meta, at: doctype.upperBound)
        } else {
            html = meta + html
        }
        return Data(html.utf8)
    }

    /// A fresh configuration. Built per load, never reused.
    @MainActor
    static func makeConfiguration(
        source: String, ruleList: WKContentRuleList, isolationScript: String = isolationScript
    ) -> WKWebViewConfiguration {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        configuration.setURLSchemeHandler(
            ArtifactSchemeHandler(document: document(for: source)), forURLScheme: scheme
        )
        configuration.userContentController.add(ruleList)
        configuration.userContentController.addUserScript(WKUserScript(
            source: isolationScript, injectionTime: .atDocumentStart,
            forMainFrameOnly: false, in: .page
        ))
        configuration.preferences.javaScriptCanOpenWindowsAutomatically = false
        configuration.mediaTypesRequiringUserActionForPlayback = .all
        return configuration
    }

    /// Compile the deny-all list. Nil fails the load closed: a preview that is
    /// silently unprotected is worse than one that is unavailable.
    @MainActor
    static func compiledRuleList() async -> WKContentRuleList? {
        await withCheckedContinuation { continuation in
            WKContentRuleListStore.default()?.compileContentRuleList(
                forIdentifier: "rapid-artifact-deny-all",
                encodedContentRuleList: contentRuleListJSON
            ) { list, _ in continuation.resume(returning: list) }
                ?? continuation.resume(returning: nil)
        }
    }

    /// The navigation rule as a function of the request, so it can be asserted
    /// for every scheme without a web view.
    nonisolated static func policy(
        for url: URL?, isMainFrame: Bool, initialLoadAlreadyAllowed: Bool
    ) -> WKNavigationActionPolicy {
        guard isMainFrame, !initialLoadAlreadyAllowed,
              let url, url.scheme == scheme, url.path == documentPath
        else { return .cancel }
        return .allow
    }
}

/// Serves exactly one document, from memory. Everything else fails.
private final class ArtifactSchemeHandler: NSObject, WKURLSchemeHandler {
    private let document: Data

    init(document: Data) { self.document = document }

    func webView(_ webView: WKWebView, start task: WKURLSchemeTask) {
        guard let url = task.request.url, url.scheme == ArtifactSandbox.scheme,
              url.path == ArtifactSandbox.documentPath,
              let response = HTTPURLResponse(
                url: url, statusCode: 200, httpVersion: "HTTP/1.1",
                headerFields: [
                    "Content-Type": "text/html; charset=utf-8",
                    "Content-Length": String(document.count),
                    "Content-Security-Policy": ArtifactSandbox.contentSecurityPolicy,
                    "Cache-Control": "no-store",
                ]
              )
        else {
            task.didFailWithError(URLError(.unsupportedURL))
            return
        }
        task.didReceive(response)
        task.didReceive(document)
        task.didFinish()
    }

    func webView(_ webView: WKWebView, stop task: WKURLSchemeTask) {}
}

/// Navigation and UI policy for the artifact web view.
@MainActor
final class ArtifactWebDelegate: NSObject, WKNavigationDelegate, WKUIDelegate {

    enum Event: Equatable {
        case didFinish
        case didFail
        case contentProcessTerminated
    }

    private let onEvent: @MainActor (Event) -> Void
    private var initialLoadAllowed = false

    init(onEvent: @escaping @MainActor (Event) -> Void) {
        self.onEvent = onEvent
    }

    // MARK: Navigation

    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationAction: WKNavigationAction,
        decisionHandler: @escaping @MainActor @Sendable (WKNavigationActionPolicy) -> Void
    ) {
        let policy = ArtifactSandbox.policy(
            for: navigationAction.request.url,
            isMainFrame: navigationAction.targetFrame?.isMainFrame ?? false,
            initialLoadAlreadyAllowed: initialLoadAllowed
        )
        if policy == .allow { initialLoadAllowed = true }
        decisionHandler(policy)
    }

    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationResponse: WKNavigationResponse,
        decisionHandler: @escaping @MainActor @Sendable (WKNavigationResponsePolicy) -> Void
    ) {
        decisionHandler(
            navigationResponse.response.url?.scheme == ArtifactSandbox.scheme ? .allow : .cancel
        )
    }

    func webView(
        _ webView: WKWebView, navigationAction: WKNavigationAction, didBecome download: WKDownload
    ) {
        download.cancel { _ in }
    }

    func webView(
        _ webView: WKWebView, navigationResponse: WKNavigationResponse, didBecome download: WKDownload
    ) {
        download.cancel { _ in }
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        onEvent(.didFinish)
    }

    func webView(
        _ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error
    ) { onEvent(.didFail) }

    func webView(
        _ webView: WKWebView,
        didFailProvisionalNavigation navigation: WKNavigation!,
        withError error: Error
    ) { onEvent(.didFail) }

    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        onEvent(.contentProcessTerminated)
    }

    // MARK: UI

    func webView(
        _ webView: WKWebView,
        createWebViewWith configuration: WKWebViewConfiguration,
        for navigationAction: WKNavigationAction,
        windowFeatures: WKWindowFeatures
    ) -> WKWebView? { nil }

    func webView(
        _ webView: WKWebView,
        requestMediaCapturePermissionFor origin: WKSecurityOrigin,
        initiatedByFrame frame: WKFrameInfo,
        type: WKMediaCaptureType,
        decisionHandler: @escaping @MainActor @Sendable (WKPermissionDecision) -> Void
    ) { decisionHandler(.deny) }

    func webView(
        _ webView: WKWebView,
        runOpenPanelWith parameters: WKOpenPanelParameters,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping @MainActor @Sendable ([URL]?) -> Void
    ) { completionHandler(nil) }

    // Answered at once so a page cannot stall the app behind a modal.
    func webView(
        _ webView: WKWebView,
        runJavaScriptAlertPanelWithMessage message: String,
        initiatedByFrame frame: WKFrameInfo
    ) async {}

    func webView(
        _ webView: WKWebView,
        runJavaScriptConfirmPanelWithMessage message: String,
        initiatedByFrame frame: WKFrameInfo
    ) async -> Bool { false }

    func webView(
        _ webView: WKWebView,
        runJavaScriptTextInputPanelWithPrompt prompt: String,
        defaultText: String?,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping @MainActor @Sendable (String?) -> Void
    ) { completionHandler(nil) }
}
