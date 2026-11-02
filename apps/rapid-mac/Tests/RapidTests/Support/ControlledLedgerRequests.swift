import Foundation
@testable import Rapid

/// Registers each request before notifying the test; completions are delivered
/// individually so cancellation and response ordering can be exercised exactly.
actor ControlledLedgerRequests {
    private var pending: [String: CheckedContinuation<ShareComputeLedgerAccount, Error>] = [:]
    private var observers: [String: CheckedContinuation<Void, Never>] = [:]

    func load(_ key: ShareComputeProviderKey) async throws -> ShareComputeLedgerAccount {
        try await withCheckedThrowingContinuation { continuation in
            pending[key.rawValue] = continuation
            observers.removeValue(forKey: key.rawValue)?.resume()
        }
    }

    func waitForRequest(_ key: ShareComputeProviderKey) async {
        guard pending[key.rawValue] == nil else { return }
        await withCheckedContinuation { observers[key.rawValue] = $0 }
    }

    func complete(_ key: ShareComputeProviderKey, with result: Result<ShareComputeLedgerAccount, ShareComputeLedgerError>) {
        pending.removeValue(forKey: key.rawValue)?.resume(with: result.mapError { $0 as Error })
    }
}
