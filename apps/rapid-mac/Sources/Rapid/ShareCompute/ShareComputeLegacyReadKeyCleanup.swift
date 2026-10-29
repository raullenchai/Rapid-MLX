import Foundation

/// Best-effort, one-shot cleanup of the legacy `qsprk-` ledger read key.
///
/// ## History
///
/// Rapid used to have a second, read-only `qsprk-` key the Credits tab asked
/// for separately, then — briefly — a read-only MIGRATION fallback that kept
/// an already-saved `qsprk-` key working until a `qsppk-` provider key took
/// over. Owner decision: drop the fallback outright rather than keep
/// maintaining its edge cases. Share Compute has not publicly launched, so
/// very few accounts, if any, ever held a read key — ``ShareComputeLedgerCoordinator``
/// now reads ONLY the provider key; a provider-missing account sees the
/// "add your key" onboarding state, full stop.
///
/// ## What this does
///
/// Deletes whatever is left of the old read key and its retirement marker
/// from the Keychain, once, so an account that saved one does not keep a
/// dead credential sitting on disk. The result is deliberately ignored:
/// there is nothing to do differently if a deletion fails, and nothing ever
/// reads either account again regardless.
enum ShareComputeLegacyReadKeyCleanup {
    /// The old read key's account. Matches what a pre-this-change build
    /// wrote; never written by any build after it.
    static let account = "rapid.quicksilver.ledger-read-key.v1"
    /// The short-lived retirement marker from the migration-fallback design
    /// this cleanup replaces. Also just deleted, never read.
    static let retiredAccount = "rapid.quicksilver.ledger-read-key.retired.v1"

    static func run(keychain: any KeychainStoring) {
        keychain.delete(account: account)
        keychain.delete(account: retiredAccount)
    }
}
