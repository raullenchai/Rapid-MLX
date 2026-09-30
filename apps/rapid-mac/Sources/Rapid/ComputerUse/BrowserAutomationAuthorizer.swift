import ApplicationServices
import Foundation

enum BrowserAutomationAuthorization: Equatable, Sendable {
    case authorized
    case denied
    case targetUnavailable
    case timedOut
    case failed(OSStatus)
}

/// Requests browser Automation from the signed foreground app before a CUA run.
///
/// The helper remains responsible for reading and validating the live URL. This
/// boundary only establishes the main app's target-specific Apple Events grant
/// in direct response to Start. macOS attributes helper-launched Apple Events
/// to the responsible main app, so waiting for the helper's first URL read can
/// leave the consent request unresolved without creating a TCC decision.
enum BrowserAutomationAuthorizer {
    typealias PermissionCheck = @Sendable (String) -> OSStatus

    static let defaultTimeoutNanoseconds: UInt64 = 35_000_000_000

    static func supports(bundleIdentifier: String) -> Bool {
        let bundle = bundleIdentifier.lowercased()
        return bundle == "com.apple.safari"
            || bundle == "com.apple.safaritechnologypreview"
            || bundle.hasPrefix("com.google.chrome")
            || bundle.hasPrefix("com.microsoft.edgemac")
            || bundle.hasPrefix("org.chromium.chromium")
    }

    static func request(
        bundleIdentifier: String,
        timeoutNanoseconds: UInt64 = defaultTimeoutNanoseconds,
        permissionCheck: @escaping PermissionCheck = determinePermission
    ) async -> BrowserAutomationAuthorization {
        guard supports(bundleIdentifier: bundleIdentifier) else {
            return .targetUnavailable
        }
        return await withCheckedContinuation { continuation in
            let completion = AuthorizationCompletion(continuation)
            Task.detached(priority: .userInitiated) {
                completion.resume(map(permissionCheck(bundleIdentifier)))
            }
            Task.detached {
                try? await Task.sleep(nanoseconds: timeoutNanoseconds)
                completion.resume(.timedOut)
            }
        }
    }

    private static func determinePermission(bundleIdentifier: String) -> OSStatus {
        var target = AEAddressDesc()
        let status = bundleIdentifier.withCString { bytes in
            AECreateDesc(
                DescType(typeApplicationBundleID),
                bytes,
                bundleIdentifier.utf8.count,
                &target
            )
        }
        guard status == noErr else { return OSStatus(status) }
        defer { AEDisposeDesc(&target) }
        return AEDeterminePermissionToAutomateTarget(
            &target, AEEventClass(typeWildCard), AEEventID(typeWildCard), true
        )
    }

    private static func map(_ status: OSStatus) -> BrowserAutomationAuthorization {
        switch status {
        case noErr:
            return .authorized
        case OSStatus(errAEEventNotPermitted):
            return .denied
        case OSStatus(procNotFound):
            return .targetUnavailable
        default:
            return .failed(status)
        }
    }
}

private final class AuthorizationCompletion: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<BrowserAutomationAuthorization, Never>?

    init(_ continuation: CheckedContinuation<BrowserAutomationAuthorization, Never>) {
        self.continuation = continuation
    }

    func resume(_ result: BrowserAutomationAuthorization) {
        lock.lock()
        let pending = continuation
        continuation = nil
        lock.unlock()
        pending?.resume(returning: result)
    }
}
