import Foundation
import ObjectiveC

/// Redirects Foundation-level string lookups — `String(localized:)`,
/// `NSLocalizedString`, and any direct
/// ``Bundle/localizedString(forKey:value:table:)`` — to a chosen `.lproj`
/// instead of whatever language macOS picked for the process.
///
/// ## Scope: this is the *second* half of the language switch
///
/// SwiftUI copy is **not** handled here. A `Text("…")` resolves its
/// `LocalizedStringKey` against the ``EnvironmentValues/locale`` in scope, so
/// ``RapidApp`` carries the whole SwiftUI surface by putting
/// ``LanguageConfig/resolvedLocale`` into the environment — which also gets
/// the re-render, because an environment change invalidates the views that
/// read it. This class covers the calls SwiftUI does not own: AppKit menu
/// items, `NSAlert` bodies, window titles, and the `String(localized:)`
/// helpers in `ModelReadiness` and friends.
///
/// Both halves read the same ``LanguageConfig``, so they cannot disagree.
///
/// ## Why not `AppleLanguages`
///
/// Writing `AppleLanguages` into `UserDefaults` is the documented Apple route,
/// but it only takes effect on the *next* launch. That reads as a broken
/// control: the user picks 简体中文, the picker snaps to it, and nothing on
/// screen changes.
///
/// ## How it works
///
/// The method is replaced on ``Bundle`` **once**, exchanged with
/// ``Rapid_localizedString(forKey:value:table:)``. Exchanging swaps the two
/// *implementations*, so after the exchange the original AppKit code answers
/// the `Rapid_localizedString` selector and our interception answers
/// `localizedString`. The pass-through therefore spells the original call as
/// `self.Rapid_localizedString(…)`; writing `self.localizedString(…)` there
/// would re-enter the interception forever.
///
/// The interception is a strict pass-through except for one case: the receiver
/// is ``Bundle/main`` **and** a language override is active. That scoping is
/// deliberate — a blanket redirect would also rewrite lookups inside Sparkle
/// and MarkdownUI, whose string tables have nothing to do with our catalog and
/// whose `.lproj` directories we do not control.
///
/// ## Cost
///
/// One swizzle per process (installed when ``activate(code:)`` is first
/// reached), a lock on each lookup, and one cached `Bundle` per language code.
/// Localization lookups are not hot enough for the lock to matter, and the
/// alternative — reaching into SwiftUI to invalidate every rendered string —
/// is not possible.
///
/// ## What this does *not* do
///
/// It changes **copy**, not **formatting**. `Locale.current`-driven date and
/// number formatting still follows the system locale, the same way a process
/// launched with `-AppleLanguages` would. Callers that own a formatter read
/// ``LanguageConfig/resolvedLocale`` for that reason.
final class BundleLanguageOverride: @unchecked Sendable {
    static let shared = BundleLanguageOverride()

    /// Guards ``activeCode`` and ``cachedBundles``. `localizedString` is called
    /// from whatever thread is rendering, so this cannot be main-actor state.
    private let lock = NSLock()
    private var activeCode: String?
    private var cachedBundles: [String: Bundle] = [:]

    /// `dispatch_once` is unavailable from Swift, so installation is guarded by
    /// this flag. Reaching it twice is harmless (see ``installOverrideOnce``).
    nonisolated(unsafe) private static var didInstallOverride = false

    private init() {
        Self.installOverrideOnce()
    }

    /// Point lookups at `code`'s `.lproj`, or back at the process localization
    /// when `code` is nil. Safe to call repeatedly; safe from any thread.
    static func activate(code: String?) {
        shared.setActive(code: code)
    }

    /// The bundle a lookup should be answered from, or nil for "unchanged".
    /// Separate from ``resolve(forKey:value:table:receiver:)`` so tests can
    /// assert the redirect without driving the swizzle.
    func bundle(for receiver: Bundle, code: String?) -> Bundle? {
        guard receiver === Bundle.main, let code else { return nil }
        lock.lock()
        defer { lock.unlock() }
        if let cached = cachedBundles[code] { return cached }
        guard
            let path = Bundle.main.path(forResource: code, ofType: "lproj"),
            let bundle = Bundle(path: path)
        else {
            // A `.lproj` this build never compiled. Leaving the lookup alone is
            // the correct degradation: the app keeps the system language it
            // would have used anyway, instead of blanking every string.
            return nil
        }
        cachedBundles[code] = bundle
        return bundle
    }

    /// Answer `key` for `receiver`, honouring the active override.
    ///
    /// - Parameter original: the un-swizzled lookup, so the miss path can hand
    ///   the question back to AppKit rather than duplicating its fallback
    ///   rules (missing key → `value` → key itself).
    func resolve(
        receiver: Bundle,
        key: String,
        value: String?,
        table: String?,
        original: (Bundle, String, String?, String?) -> String
    ) -> String {
        guard let bundle = bundle(for: receiver, code: currentCode()) else {
            return original(receiver, key, value, table)
        }
        return bundle.localizedString(forKey: key, value: value, table: table)
    }

    private func setActive(code: String?) {
        lock.lock()
        activeCode = code
        lock.unlock()
    }

    private func currentCode() -> String? {
        lock.lock()
        defer { lock.unlock() }
        return activeCode
    }

    /// Exchange the implementation on ``Bundle`` exactly once per process.
    private static func installOverrideOnce() {
        guard !didInstallOverride else { return }
        didInstallOverride = true
        guard
            let original = class_getInstanceMethod(
                Bundle.self,
                #selector(Bundle.localizedString(forKey:value:table:))
            ),
            let replacement = class_getInstanceMethod(
                Bundle.self,
                #selector(Bundle.Rapid_localizedString(forKey:value:table:))
            )
        else {
            // Without both halves there is nothing to exchange. Shipping the
            // un-swizzled behaviour is the correct degradation: the language
            // picker then behaves like the system setting, which is what the
            // rest of macOS does anyway.
            return
        }
        method_exchangeImplementations(original, replacement)
    }
}

extension Bundle {
    /// Swizzle target — see ``BundleLanguageOverride``. The name is
    /// deliberately not `localizedString…`, because after the exchange this
    /// selector is what reaches the original AppKit implementation.
    @objc dynamic func Rapid_localizedString(
        forKey key: String,
        value: String?,
        table tableName: String?
    ) -> String {
        BundleLanguageOverride.shared.resolve(
            receiver: self,
            key: key,
            value: value,
            table: tableName,
            original: { receiver, key, value, table in
                receiver.Rapid_localizedString(forKey: key, value: value, table: table)
            }
        )
    }
}
