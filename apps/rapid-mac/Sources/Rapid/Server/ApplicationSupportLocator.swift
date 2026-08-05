import Foundation

/// Canonical resolution of ``~/Library/Application Support/Rapid``
/// that honours ``$HOME`` overrides.
///
/// ## Why this exists (#419 + #420)
///
/// Foundation offers two ways to find Application Support:
///
/// 1. ``FileManager.default.urls(for: .applicationSupportDirectory,
///    in: .userDomainMask)`` — internally calls
///    ``NSSearchPathForDirectoriesInDomains`` which resolves through
///    ``getpwuid(geteuid())``. The result is the LOGGED-IN USER's
///    real home directory, ignoring any ``$HOME`` override the
///    process was launched with.
/// 2. ``$HOME + /Library/Application Support`` — honours the env
///    var by construction.
///
/// Production users never override ``$HOME``, so on real machines
/// the two resolve to the same path. Dogfood / test harnesses,
/// however, routinely override ``$HOME`` to isolate write paths
/// (e.g. launching the .app under ``HOME=/tmp/dogfood-v088/home``).
/// In v0.8.8 dogfood (#414), a slim-DMG test instance loaded the
/// user's real prod ``sessions.json`` because ``SessionStore`` used
/// shape (1) — the dogfood agent saw real chat content + the wrong
/// active alias, derailing the auto-spawn check. ``CrashReporter``
/// has the same bug, polluting the user's real
/// ``~/Library/Application Support/Rapid/crash-markers/`` with
/// dogfood-instance markers.
///
/// This locator is the single source of truth callers reach for.
/// Both pre-existing helpers
/// (``BootstrapCoordinator.defaultApplicationSupportRoot`` +
/// ``ServerLocator.defaultApplicationSupportURL``) delegate here so
/// a future refactor that wants to tweak the resolution rule has
/// exactly one place to change instead of N.
///
/// ## Resolution order
///
///   1. ``$HOME`` (if set and absolute) — appends
///      ``Library/Application Support/Rapid``. Honours dogfood
///      overrides + matches what shell `$HOME` expansion would do.
///   2. ``FileManager.default.urls(for: .applicationSupportDirectory,
///      in: .userDomainMask).first`` — defensive fallback for the
///      "no HOME set" pathology (sandboxed launch helpers, broken
///      env). Production never reaches this branch.
///   3. ``NSTemporaryDirectory`` last-resort — keeps callers
///      crash-free in the (effectively unreachable) double-fallback
///      case rather than forcing every caller to handle nil.
enum ApplicationSupportLocator {

    /// Folder name inside ``Library/Application Support`` we claim.
    /// Pinned here (not derived) so a rename has exactly one site to
    /// touch.
    static let folderName: String = "Rapid"

    /// Production accessor — reads ``ProcessInfo.processInfo.environment``.
    /// All non-test callers use this shape.
    static func applicationSupportRoot() -> URL {
        applicationSupportRoot(environment: ProcessInfo.processInfo.environment)
    }

    /// Test-injectable shape. Pass an explicit environment dictionary
    /// to exercise the HOME-override / HOME-unset branches without
    /// mutating the real process environment (which Swift Testing
    /// can't isolate per-test).
    static func applicationSupportRoot(environment: [String: String]) -> URL {
        applicationSupportBase(environment: environment)
            .appendingPathComponent(folderName, isDirectory: true)
    }

    /// ``Library/Application Support`` itself, resolved by the same
    /// HOME-first ladder but WITHOUT appending ``folderName``.
    ///
    /// Exists for the one caller that legitimately claims a different
    /// subdirectory: ``ConversationStore`` keys its folder on the bundle
    /// identifier so a dogfood build (rewritten bundle id) keeps its
    /// history separate. Before this it hand-rolled the FileManager call
    /// and so ignored ``$HOME`` — the exact #419/#420 shape, and precisely
    /// what the source-scan test in ``ApplicationSupportLocatorTests``
    /// forbids. Route new callers through here rather than re-deriving.
    static func applicationSupportBase(environment: [String: String]) -> URL {
        if let home = environment["HOME"], home.hasPrefix("/") {
            return URL(fileURLWithPath: home, isDirectory: true)
                .appendingPathComponent("Library", isDirectory: true)
                .appendingPathComponent("Application Support", isDirectory: true)
        }
        if let base = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first {
            return base
        }
        return URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
    }

    /// Production accessor for ``applicationSupportBase``.
    static func applicationSupportBase() -> URL {
        applicationSupportBase(environment: ProcessInfo.processInfo.environment)
    }
}
