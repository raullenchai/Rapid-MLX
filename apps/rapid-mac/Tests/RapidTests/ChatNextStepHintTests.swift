import Foundation
import Testing
@testable import Rapid

@MainActor
struct ChatNextStepHintTests {
    @Test("Successful reply claims once, including across relaunch")
    func shownOnce() {
        let name = "ChatNextStepHintTests.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let hint = ChatNextStepHint(defaults: defaults)
        #expect(!hint.isVisible)
        hint.replySucceeded()
        #expect(hint.isVisible)
        hint.replySucceeded()
        #expect(hint.isVisible)
        let relaunched = ChatNextStepHint(defaults: defaults)
        relaunched.replySucceeded()
        #expect(!relaunched.isVisible)
        hint.dismiss()
        #expect(!hint.isVisible)
        hint.replySucceeded()
        #expect(!hint.isVisible)
    }

    @Test("Opening Launch before or after a reply permanently suppresses the hint", arguments: [false, true])
    func launchSuppresses(afterReply: Bool) {
        let name = "ChatNextStepHintTests.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let hint = ChatNextStepHint(defaults: defaults)
        if afterReply { hint.replySucceeded() }
        hint.dismiss()
        #expect(!hint.isVisible)
        hint.replySucceeded()
        #expect(!hint.isVisible)
        let relaunched = ChatNextStepHint(defaults: defaults)
        relaunched.replySucceeded()
        #expect(!relaunched.isVisible)
    }
}
