import Foundation

/// Builds `OpenURLAction` completion handlers that are safe to invoke from
/// any thread.
///
/// When the URL opens, `OpenURLAction` calls its completion from
/// LaunchServices' `com.apple.launchservices.open-queue`, not from the main
/// thread. The `Rapid` target compiles in Swift 6 language mode, so a
/// completion written as a closure literal inside a SwiftUI `body` inherits
/// the view's `@MainActor` isolation, and because it is handed to an SDK API
/// that predates strict concurrency, the compiler emits a runtime executor
/// check at the closure's entry. That check runs before the closure's first
/// statement, so even a body that opens with `Task { @MainActor in … }`
/// traps (`EXC_BREAKPOINT` in `_swift_task_checkIsolatedSwift`) the moment
/// the browser accepts the URL. Rapid-MLX Desktop 0.16.2's "See what's new"
/// button crashed this way.
///
/// The closure built here is formed in a `nonisolated` context and typed
/// `@Sendable`, so it carries no actor isolation and no entry check. It
/// hops to the main actor itself and runs `action` only when the URL was
/// accepted, so a declined open leaves UI state untouched.
enum OpenURLCompletion {
    nonisolated static func onMainActorIfAccepted(
        _ action: @escaping @MainActor @Sendable () -> Void
    ) -> @Sendable (Bool) -> Void {
        { accepted in
            Task { @MainActor in
                guard accepted else { return }
                action()
            }
        }
    }
}
