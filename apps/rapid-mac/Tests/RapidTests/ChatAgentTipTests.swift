import Foundation
import Testing
@testable import Rapid

@Suite("Tool connection prompt", .serialized)
@MainActor
struct ToolConnectionPromptTests {
    private func withDefaults(_ body: (UserDefaults) -> Void) {
        let suite = "ToolConnectionPromptTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        body(defaults)
    }

    @Test func successStreamingAndRelaunch() {
        withDefaults { defaults in
            var events: [TelemetryEvent.Activation.Kind] = []
            let prompt = ToolConnectionPrompt(defaults: defaults) { events.append($0) }
            #expect(!prompt.shouldPresent(isStreaming: false))
            prompt.recordSuccess()
            #expect(!prompt.shouldPresent(isStreaming: true))
            #expect(prompt.shouldPresent(isStreaming: false))
            prompt.didShow()
            prompt.didShow()
            #expect(events == [.toolCardShown])
            #expect(!prompt.shouldPresent(isStreaming: true))
            #expect(prompt.shouldPresent(isStreaming: false))
            let relaunched = ToolConnectionPrompt(defaults: defaults, report: { _ in })
            relaunched.recordSuccess()
            #expect(!relaunched.shouldPresent(isStreaming: false))
        }
    }

    @Test func dismissalPersistsAndDoesNotRepeat() {
        withDefaults { defaults in
            var events: [TelemetryEvent.Activation.Kind] = []
            let prompt = ToolConnectionPrompt(defaults: defaults) { events.append($0) }
            prompt.recordSuccess()
            prompt.didShow()
            prompt.dismiss()
            prompt.dismiss()
            prompt.complete(.copyURL)
            #expect(events == [.toolCardShown, .toolCardDismissed])
            #expect(defaults.bool(forKey: "chatAgentTipDismissed"))
            #expect(!prompt.shouldPresent(isStreaming: false))
            #expect(!ToolConnectionPrompt(defaults: defaults, report: { _ in })
                .shouldPresent(isStreaming: false))
        }
    }

    @Test(arguments: ToolConnectionPrompt.Action.allCases)
    func actionsPersistCompletion(action: ToolConnectionPrompt.Action) {
        withDefaults { defaults in
            var events: [TelemetryEvent.Activation.Kind] = []
            let prompt = ToolConnectionPrompt(defaults: defaults) { events.append($0) }
            prompt.recordSuccess()
            prompt.didShow()
            prompt.complete(action)
            prompt.complete(action)
            prompt.dismiss()
            #expect(events == [.toolCardShown, action.milestone])
            #expect(defaults.bool(forKey: "chatAgentTipCompleted"))
            #expect(!defaults.bool(forKey: "chatAgentTipDismissed"))
            #expect(!prompt.shouldPresent(isStreaming: false))
            #expect(!ToolConnectionPrompt(defaults: defaults, report: { _ in })
                .shouldPresent(isStreaming: false))
        }
    }

    @Test func existingDismissalIsRespected() {
        withDefaults { defaults in
            defaults.set(true, forKey: "chatAgentTipEligible")
            defaults.set(true, forKey: "chatAgentTipDismissed")
            let prompt = ToolConnectionPrompt(defaults: defaults) { _ in
                Issue.record("Dismissed prompt must not send events")
            }
            prompt.recordSuccess()
            prompt.didShow()
            #expect(!prompt.shouldPresent(isStreaming: false))
        }
    }
}
