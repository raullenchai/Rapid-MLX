import Foundation
import Testing
@testable import Rapid

/// Pins the tray menu's ordering and conditional branches.
///
/// The rendered ``NSMenu`` is not AX-introspectable (see
/// ``MenuBarStatus``), so the strongest regression guard for the menu's
/// dynamic content — the status line, the update row, and now the
/// "Copy API endpoint" row — is to pin the pure ``menuItems(_:)`` model
/// directly. These tests assert the presence/enablement/order of rows
/// the way ``MenuBarController.rebuildMenu`` actually renders them, not
/// against a mock menu.
@Suite("Menu-bar status menu model")
struct MenuBarStatusTests {

    /// The ``MenuBarStatus.MenuBarAction`` carried by the tray's
    /// copy-endpoint row.
    private let copyAction = MenuBarStatus.MenuBarAction.copyEndpoint

    /// Locate the copy-endpoint row (if any) in a rendered menu.
    private func copyRow(in items: [MenuBarStatus.MenuBarItem])
        -> (title: String, enabled: Bool)?
    {
        guard let row = items.first(where: {
            if case .button(let action, _, _, _) = $0 { return action == .copyEndpoint }
            return false
        }) else { return nil }
        guard case .button(_, let title, let enabled, _) = row else { return nil }
        return (title, enabled)
    }

    @Test("The 'Copy API endpoint' row appears, enabled, directly under the status line, when serving")
    func copyEndpointRowPresentWhenServing() {
        let items = MenuBarStatus.menuItems(
            state: .ready(alias: "lfm2.5-1.2b"),
            hasUpdate: false,
            updateVersion: "",
            checking: false,
            baseURL: "http://127.0.0.1:8000/v1"
        )

        // Exactly one copy row, correctly labelled and enabled.
        let count = items.filter {
            if case .button(let a, _, _, _) = $0 { return a == copyAction }
            return false
        }.count
        #expect(count == 1, "Expected exactly one Copy API endpoint row, got \(count)")

        let row = copyRow(in: items)
        #expect(row?.title == "Copy API endpoint")
        #expect(row?.enabled == true, "The copy row must be enabled while the server is serving")

        // It sits immediately below the status line — the highest-value
        // action floats toward the top for serve-type users.
        let statusIndex = items.firstIndex {
            if case .status = $0 { return true }
            return false
        }
        let copyIndex = items.firstIndex {
            if case .button(let a, _, _, _) = $0 { return a == copyAction }
            return false
        }
        #expect(
            copyIndex == statusIndex.map { $0 + 1 },
            "Copy row (at \(copyIndex.map(String.init) ?? "nil")) must sit right under the status line"
        )
    }

    @Test("The 'Copy API endpoint' row is absent when the backend is not serving")
    func copyEndpointRowAbsentWhenNotServing() {
        // An idle backend — the controller passes baseURL: nil — must not
        // surface a dead "copy" click.
        let items = MenuBarStatus.menuItems(
            state: .idle,
            hasUpdate: false,
            updateVersion: "",
            checking: false,
            baseURL: nil
        )
        #expect(
            copyRow(in: items) == nil,
            "No Copy API endpoint row may render while the server is not serving"
        )
    }

    @Test("No copy row for a ready state that reports no resolvable URL")
    func readyWithoutPortHasNoCopyRow() {
        // A `.ready` state with a nil baseURL (controller couldn't
        // resolve host/port — e.g. server deallocated) must not render
        // the row: copying an unknown URL would be a dead click.
        let items = MenuBarStatus.menuItems(
            state: .ready(alias: "demo"),
            hasUpdate: false,
            updateVersion: "",
            checking: false,
            baseURL: nil
        )
        #expect(copyRow(in: items) == nil)
    }
}
