import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Local effective use and prompt schedule")
struct EffectiveUseTests {
    private func defaults() -> UserDefaults {
        let name = "EffectiveUseTests.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        return defaults
    }

    @Test("Every delivered outcome has its own count; days remain bounded")
    func countsAndDays() {
        let defaults = defaults()
        var day = Date(timeIntervalSince1970: 1_800_000_000)
        let uses = EffectiveUse(defaults: defaults, now: { day })
        let kinds: [ProductValueKind] = [
            .chatReply, .dictationTranscript, .generatedImage, .editedImage,
            .synthesizedSpeech
        ]
        for kind in kinds { uses.record(kind) }
        #expect(uses.total == kinds.count)
        #expect(uses.distinctDays == 1)
        #expect(uses.firstDay == Calendar.current.startOfDay(for: day))
        for kind in kinds { #expect(uses.count(kind) == 1) }
        day = Calendar.current.date(byAdding: .day, value: 1, to: day)!
        uses.record(.chatReply)
        #expect(uses.distinctDays == 2)
        #expect(uses.count(.chatReply) == 2)
    }

    @Test("Existing star history seeds the counter once")
    func migration() {
        let defaults = defaults()
        defaults.set(35, forKey: PromptScheduler.Keys.totalSuccessfulActions)
        let uses = EffectiveUse(defaults: defaults)
        #expect(uses.total == 35)
        #expect(uses.distinctDays == 2)
        uses.record(.chatReply)
        #expect(EffectiveUse(defaults: defaults).total == 36)
        #expect(defaults.integer(forKey: PromptScheduler.Keys.totalSuccessfulActions) == 35)
    }

    @Test("Newsletter starts at two outcomes, wins priority, and never returns")
    func newsletter() {
        let defaults = defaults()
        let prompt = PromptScheduler(defaults: defaults, quietWindow: .zero,
                                     presentationActive: true)
        prompt.productValueDelivered(.chatReply)
        #expect(!prompt.newsletterPresented)
        prompt.productValueDelivered(.generatedImage)
        #expect(prompt.newsletterPresented)
        prompt.completeNewsletter()
        #expect(!prompt.newsletterPresented)
        let next = PromptScheduler(defaults: defaults, quietWindow: .zero,
                                   presentationActive: true)
        next.startMonitoringUserActivity()
        #expect(!next.newsletterPresented)
    }

    @Test("Background threshold waits for the workspace and a later launch")
    func backgroundAndCooldown() {
        let defaults = defaults()
        var date = Date(timeIntervalSince1970: 1_800_000_000)
        let prompt = PromptScheduler(defaults: defaults, now: { date }, quietWindow: .zero)
        prompt.productValueDelivered(.chatReply)
        prompt.productValueDelivered(.chatReply)
        #expect(!prompt.newsletterPresented)
        prompt.startMonitoringUserActivity()
        #expect(prompt.newsletterPresented)
        prompt.completeNewsletter()
        defaults.set(10, forKey: EffectiveUse.Keys.total)
        defaults.set(2, forKey: EffectiveUse.Keys.distinctDays)
        let sameDay = PromptScheduler(defaults: defaults, now: { date }, quietWindow: .zero,
                                      presentationActive: true)
        sameDay.startMonitoringUserActivity()
        #expect(!sameDay.isPresented)
        date.addTimeInterval(PromptScheduler.deferralInterval)
        let later = PromptScheduler(defaults: defaults, now: { date }, quietWindow: .zero,
                                    presentationActive: true)
        later.startMonitoringUserActivity()
        #expect(later.isPresented)
    }
}
