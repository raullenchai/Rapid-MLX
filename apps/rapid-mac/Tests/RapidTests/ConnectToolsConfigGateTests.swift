import Testing
@testable import Rapid

/// Behavioral truth-table for the Connect-tools disabled-Copy gate
/// (`ConnectToolsView.configIsReady`), factored out of the view so the full
/// question — "is it safe to let the user copy a snippet right now?" — is a
/// testable decision rather than an untested view-local expression.
///
/// The gate is the one behavior #2297's stopped state must not regress: when
/// the model is not serving, the page still renders the endpoint shape and
/// the integration rows as documentation, but every Copy action must stay
/// disabled until the endpoint/key/model actually exist. Copying a
/// half-filled template (a placeholder that *looks* like a command) is the
/// silent-failure defect the gate exists to prevent — see #1470's key
/// generation and the pre-#2297 behavior that let a user paste a placeholder.
/// ``ConnectToolsView`` is a ``@MainActor`` SwiftUI view, so its static
/// ``configIsReady`` is main-actor-isolated — same shape as
/// ``ModelPickerBarSectionOrderTests``. The test body is pure and touches no
/// live state; the annotation just matches the isolation of the function
/// under test.
@MainActor
@Suite("Connect tools config gate — stopped-state Copy disabled (#2297)")
struct ConnectToolsConfigGateTests {

    /// The #2297 stopped state: the page supplies a readiness value and the
    /// model is NOT serving (`modelServing == false`). Even with every static
    /// value present (real port, minted bearer, resolved model), the gate
    /// under test must say NOT ready — otherwise a user could paste an
    /// integration command or key that points at a placeholder. This is the
    /// regression guard for "Copy on a placeholder".
    ///
    /// Scope (deliberately narrower than "every Copy button stays disabled"):
    /// this pins the gate that disables the runtime-only Copy controls — the
    /// API-key row (placeholder disables Copy) and every integration
    /// command's ``Launch.Integration.Copy.*`` button, which ``toolsSection``
    /// feeds ``configReady`` (== this boolean) into ``ConnectToolRow.isReady``
    /// → ``.disabled(!isReady)``. The base-URL and already-resolved Model rows
    /// are real values and correctly stay copyable while stopped; this test
    /// does not claim otherwise. The rendering-level "integration rows
    /// present-but-disabled while stopped" is asserted by the
    /// ``launch-integrations`` AX golden + native journey.
    @Test("Stopped state: gate NOT ready, so API-key and integration Copy stay disabled even when values fill in")
    func stoppedStateNeverCopyable() {
        #expect(!ConnectToolsView.configIsReady(
            hasPort: true,
            hasBearer: true,
            modelResolved: true,
            modelServing: false // the model is not running (#2297 stopped estate)
        ))
    }

    /// Each individual missing ingredient independently forces NOT ready.
    /// The gate is ANDed across all four inputs, so removing any one must
    /// drop the whole config below the copy threshold.
    @Test("Any single missing ingredient disables Copy")
    func everySingleMissingIngredientDisables() {
        // No real port yet.
        #expect(!ConnectToolsView.configIsReady(
            hasPort: false, hasBearer: true, modelResolved: true, modelServing: true
        ))
        // No bearer minted yet ("Created when the server starts").
        #expect(!ConnectToolsView.configIsReady(
            hasPort: true, hasBearer: false, modelResolved: true, modelServing: true
        ))
        // No model name resolved yet (placeholder row).
        #expect(!ConnectToolsView.configIsReady(
            hasPort: true, hasBearer: true, modelResolved: false, modelServing: true
        ))
        // Model resolved but not serving.
        #expect(!ConnectToolsView.configIsReady(
            hasPort: true, hasBearer: true, modelResolved: true, modelServing: false
        ))
    }

    /// Fully complete + serving is exactly the single copy-ready state.
    @Test("Only a complete, serving config is copyable")
    func completeServingConfigIsCopyable() {
        #expect(ConnectToolsView.configIsReady(
            hasPort: true,
            hasBearer: true,
            modelResolved: true,
            modelServing: true
        ))
    }

    /// `nil` modelServing is the dev-snapshot harness path, which has no live
    /// server to resolve readiness against. It must preserve the pre-#2297
    /// behavior of trusting the static values alone (never `false` out of
    /// nowhere), so standalone renders still show live content.
    @Test("Dev-snapshot path (no readiness) trusts static values")
    func nilServingPreservesLegacyTrust() {
        #expect(ConnectToolsView.configIsReady(
            hasPort: true, hasBearer: true, modelResolved: true, modelServing: nil
        ))
        // Even the harness still needs the static ingredients.
        #expect(!ConnectToolsView.configIsReady(
            hasPort: false, hasBearer: true, modelResolved: true, modelServing: nil
        ))
    }

    /// Exhaustive check across all 24 input combinations (three booleans ×
    /// the three-state `modelServing` = 2·2·2·3): the gate reduces exactly to
    /// `hasPort && hasBearer && modelResolved && (modelServing ?? true)`.
    /// Locks the boolean shape so a future change to the formula (e.g. OR-ing
    /// a clause, making serving optional) cannot silently widen Copy without
    /// a review noticing which combinations flipped.
    @Test("Exhaustive truth table matches the ANDed definition")
    func exhaustiveTruthTable() {
        let servingStates: [Bool?] = [nil, false, true]
        for hasPort in [false, true] {
            for hasBearer in [false, true] {
                for modelResolved in [false, true] {
                    for modelServing in servingStates {
                        let expected =
                            hasPort && hasBearer && modelResolved && (modelServing ?? true)
                        #expect(
                            ConnectToolsView.configIsReady(
                                hasPort: hasPort,
                                hasBearer: hasBearer,
                                modelResolved: modelResolved,
                                modelServing: modelServing
                            ) == expected
                        )
                    }
                }
            }
        }
    }
}
