import AppKit
import QuickLookUI
import SwiftUI

/// Small seam for exercising lifetime and scheduling without a system preview panel.
@MainActor protocol LocalSearchPreviewPanelDriving: AnyObject {
    var isVisible: Bool { get }
    func acquire(_ controller: LocalSearchPreviewController) -> Bool
    func reload()
    func show()
    func release(_ controller: LocalSearchPreviewController)
    func didEndControl(_ controller: LocalSearchPreviewController)
}

extension LocalSearchPreviewPanelDriving {
    func didEndControl(_ controller: LocalSearchPreviewController) {}
}

/// A view-lifetime reference, independent of result rows and SwiftUI value updates.
/// All panel work crosses a main-queue boundary. Quick Look can spin a nested event
/// loop while opening; queued requests must also wait for that operation to return.
@MainActor final class LocalSearchPreviewController: NSResponder,
    @preconcurrency QLPreviewPanelDataSource, QLPreviewPanelDelegate {
    private let driver: any LocalSearchPreviewPanelDriving
    private var pending: [@MainActor () -> Void] = []
    private var applying = false
    private(set) var item: NSURL?
    fileprivate var wantsControl = false
    private var controlLifetime: LocalSearchPreviewController?
    weak var window: NSWindow?

    init(driver: any LocalSearchPreviewPanelDriving = LocalSearchSystemPreviewPanel()) {
        self.driver = driver
        super.init()
    }
    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    fileprivate func enqueue(_ action: @escaping @MainActor () -> Void) {
        pending.append(action)
        DispatchQueue.main.async { [self] in drain() }
    }
    private func drain() {
        guard !applying, !pending.isEmpty else { return }
        applying = true
        let action = pending.removeFirst()
        action()
        applying = false
        if !pending.isEmpty { DispatchQueue.main.async { [self] in drain() } }
    }
    func toggle(_ url: URL) {
        enqueue { [self] in
            // Visibility can lag both opening and animated dismissal. Toggle the
            // requested item, just as the original view binding toggled its URL.
            if wantsControl, item == url as NSURL { closeNow(); return }
            item = url as NSURL
            wantsControl = true
            guard driver.acquire(self) else { closeNow(); return }
            driver.reload()
            driver.show()
        }
    }
    func selectionChanged(to url: URL?) {
        enqueue { [self] in
            guard wantsControl else { return }
            guard let url else { closeNow(); return }
            item = url as NSURL
            driver.reload()
        }
    }
    func close() { enqueue { [self] in closeNow() } }
    private func closeNow() {
        // onDisappear and representable dismantling can both request cleanup.
        // Reissuing QL's animated close can cancel its in-flight dismissal.
        guard wantsControl || item != nil else { return }
        // Keep the item alive until Quick Look relinquishes its assign references.
        wantsControl = false
        driver.release(self)
        item = nil
    }
    override func acceptsPreviewPanelControl(_ panel: QLPreviewPanel!) -> Bool {
        MainActor.assumeIsolated { wantsControl }
    }
    override func beginPreviewPanelControl(_ panel: QLPreviewPanel!) {
        MainActor.assumeIsolated { bind(panel) }
    }
    override func endPreviewPanelControl(_ panel: QLPreviewPanel!) {
        MainActor.assumeIsolated {
            unbind(panel)
            driver.didEndControl(self)
        }
    }
    fileprivate func bind(_ panel: QLPreviewPanel) {
        controlLifetime = self
        panel.dataSource = self
        panel.delegate = self
    }
    fileprivate func unbind(_ panel: QLPreviewPanel) {
        if panel.dataSource === self { panel.dataSource = nil }
        if panel.delegate === self { panel.delegate = nil }
        controlLifetime = nil
    }
    func numberOfPreviewItems(in panel: QLPreviewPanel!) -> Int { item == nil ? 0 : 1 }
    func previewPanel(_ panel: QLPreviewPanel!, previewItemAt index: Int) -> (any QLPreviewItem)! {
        index == 0 ? item : nil
    }
}

/// Keeps watching an opening panel even after cancellation first reports it hidden.
struct LocalSearchPreviewDismissal {
    private(set) var closing = false
    private(set) var awaitingLateOpen = false

    mutating func begin(isVisible: Bool) -> Bool {
        guard !closing else { return false }
        closing = true
        awaitingLateOpen = !isVisible
        return true
    }

    /// Returns whether a late open with no remaining intent needs another orderOut.
    mutating func finish(isVisible: Bool, wantsControl: Bool, ownedByOther: Bool = false) -> Bool {
        if ownedByOther {
            closing = false
            awaitingLateOpen = false
            return false
        }
        if awaitingLateOpen, isVisible {
            awaitingLateOpen = false
            if !wantsControl {
                closing = true
                return true
            }
        }
        if !isVisible { closing = false }
        return false
    }
}

@MainActor final class LocalSearchSystemPreviewPanel: LocalSearchPreviewPanelDriving {
    private var panel: QLPreviewPanel?
    private weak var ownerWindow: NSWindow?
    private weak var owner: LocalSearchPreviewController?
    private weak var lastController: LocalSearchPreviewController?
    private var presented = false
    private var dismissal = LocalSearchPreviewDismissal()
    private var closing: Bool { dismissal.closing }
    nonisolated(unsafe) private var closeObserver: NSObjectProtocol?
    deinit {
        if let closeObserver { NotificationCenter.default.removeObserver(closeObserver) }
    }
    var isVisible: Bool { panel?.isVisible == true }
    func acquire(_ controller: LocalSearchPreviewController) -> Bool {
        finishClosingIfHidden(reopen: false)
        guard let window = controller.window else { return false }
        guard let panel = panel ?? QLPreviewPanel.shared() else { return false }
        self.panel = panel
        owner = controller
        lastController = controller
        if ownerWindow == nil {
            ownerWindow = window
            controller.nextResponder = window.nextResponder
            window.nextResponder = controller
        }
        if closing { return true }
        panel.updateController()
        if panel.currentController as AnyObject? === controller { controller.bind(panel) }
        // Hidden panels may defer responder discovery until makeKeyAndOrderFront.
        return !panel.isVisible || panel.currentController as AnyObject? === controller
    }
    func reload() {
        finishClosingIfHidden()
        guard !closing, let owner, panel?.currentController as AnyObject? === owner else { return }
        panel?.reloadData()
    }
    func show() {
        finishClosingIfHidden(reopen: false)
        if closing { return }
        panel?.makeKeyAndOrderFront(nil)
        presented = true
        // Hidden panels can grant control during opening. Reload after that
        // handoff so a cached item from the previous session is replaced too.
        reload()
    }
    func release(_ controller: LocalSearchPreviewController) {
        finishClosingIfHidden(reopen: false)
        let owned = panel?.currentController as AnyObject? === controller
        // Remove only our responder, preserving any other hooks in the chain.
        var predecessor: NSResponder? = ownerWindow
        while let responder = predecessor {
            if responder.nextResponder === controller {
                responder.nextResponder = controller.nextResponder
                break
            }
            predecessor = responder.nextResponder
        }
        if owned, !closing {
            // Notify Quick Look after changing the responder chain, while visible,
            // so endPreviewPanelControl releases the assign references.
            panel?.updateController()
        }
        // Teardown may remove us from the responder chain before the deferred
        // close runs. Close our now-unowned panel, but never another owner's.
        let shouldHide = (owned || presented)
            && (panel?.currentController == nil || panel?.currentController as AnyObject? === controller)
        presented = false
        // A later open/close can arrive while the previous dismissal is still
        // running. Leave QL's controller and data untouched until it finishes;
        // updating/reloading an already dismissing panel can strand it busy.
        if shouldHide, !closing {
            if let panel, dismissal.begin(isVisible: panel.isVisible) {
                // A new intent can open and close again while the previous
                // session is still dismissing. Reissuing orderOut cancels QL's
                // animation and can strand the panel without a data source.
                if closeObserver == nil {
                    closeObserver = NotificationCenter.default.addObserver(
                        forName: NSWindow.didChangeOcclusionStateNotification, object: panel, queue: .main
                    ) { [weak self, weak controller] _ in
                        // Observe dismissal while the view owns the controller;
                        // observation must not extend either object's lifetime.
                        DispatchQueue.main.async { [weak self, weak controller] in
                            controller?.enqueue { [weak self] in self?.finishClosingIfHidden() }
                        }
                    }
                }
                panel.orderOut(nil)
                controller.enqueue { [weak self] in self?.finishClosingIfHidden() }
            }
            // Quick Look can cache currentController across hidden sessions without
            // sending endControl. Detach assign references while still the owner.
            if let panel, panel.currentController as AnyObject? === controller {
                controller.unbind(panel)
                panel.reloadData()
            }
        }
        controller.nextResponder = nil
        ownerWindow = nil
        owner = nil
    }
    func didEndControl(_ controller: LocalSearchPreviewController) {
        // Native Space/Escape dismissal ends intent too. A programmatic close
        // may already have a later open waiting for its animation to finish.
        if !closing, controller.wantsControl { controller.close() }
    }
    // Reloads and queued completion checks resume a pending open. Acquire/show
    // continue opening themselves; release completes closing without reopening.
    private func finishClosingIfHidden(reopen: Bool = true) {
        let wasClosing = closing
        // The shared panel may have been claimed by a new view while our open
        // was still in flight. Neither ownership reference may belong to it.
        let ownedByOther = (panel?.currentController != nil
            && panel?.currentController as AnyObject? !== lastController)
            || (panel?.dataSource != nil && panel?.dataSource !== lastController)
        if dismissal.finish(isVisible: isVisible, wantsControl: owner?.wantsControl == true,
                            ownedByOther: ownedByOther) {
            panel?.orderOut(nil)
            return
        }
        guard !closing else { return }
        if !dismissal.awaitingLateOpen {
            if let closeObserver { NotificationCenter.default.removeObserver(closeObserver) }
            closeObserver = nil
        }
        guard wasClosing, !ownedByOther else { return }
        if reopen, let owner, owner.wantsControl {
            if acquire(owner) { reload(); show() }
        }
    }

}

/// Supplies the owning window without stealing keyboard focus or changing panel
/// state during NSViewRepresentable updates/layout. Teardown is deferred too.
struct LocalSearchPreviewAnchor: NSViewRepresentable {
    let controller: LocalSearchPreviewController
    func makeNSView(context: Context) -> Anchor { Anchor(controller: controller) }
    func updateNSView(_ nsView: Anchor, context: Context) {}
    static func dismantleNSView(_ nsView: Anchor, coordinator: ()) { nsView.controller.close() }
    final class Anchor: NSView {
        let controller: LocalSearchPreviewController
        init(controller: LocalSearchPreviewController) {
            self.controller = controller
            super.init(frame: .zero)
        }
        required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }
        override func viewDidMoveToWindow() { super.viewDidMoveToWindow(); controller.window = window }
    }
}
