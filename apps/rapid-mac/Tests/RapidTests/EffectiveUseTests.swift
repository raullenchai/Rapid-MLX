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

        let recent = self.defaults()
        recent.set(1, forKey: PromptScheduler.Keys.totalSuccessfulActions)
        #expect(EffectiveUse(defaults: recent).distinctDays == 1)
    }

    @Test("A time-zone change keeps the same written calendar day, and old dates migrate once")
    func calendarDayAcrossTimeZones() {
        let defaults = defaults()
        var east = Calendar(identifier: .gregorian)
        east.timeZone = TimeZone(secondsFromGMT: 14 * 3_600)!
        var west = Calendar(identifier: .gregorian)
        west.timeZone = TimeZone(secondsFromGMT: -12 * 3_600)!
        let first = Date(timeIntervalSince1970: 1_800_000_000)
        let eastDay = east.dateComponents([.year, .month, .day], from: first)
        let later = west.date(from: eastDay)!.addingTimeInterval(3_600)
        let uses = EffectiveUse(defaults: defaults, calendar: east, now: { first })
        uses.record(.chatReply)
        let writtenDay = defaults.string(forKey: EffectiveUse.Keys.lastDay)
        EffectiveUse(defaults: defaults, calendar: west, now: { later }).record(.chatReply)
        #expect(defaults.string(forKey: EffectiveUse.Keys.lastDay) == writtenDay)
        #expect(uses.distinctDays == 1)

        defaults.set(east.startOfDay(for: first), forKey: EffectiveUse.Keys.lastDay)
        _ = EffectiveUse(defaults: defaults, calendar: east, now: { first })
        #expect(defaults.string(forKey: EffectiveUse.Keys.lastDay) == writtenDay)
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
        let prompt = PromptScheduler(defaults: defaults, now: { date }, quietWindow: .zero,
                                     windowVisible: { true })
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

    @Test("A legacy star deferral also delays the newsletter")
    func legacyCooldown() {
        let defaults = defaults()
        let date = Date(timeIntervalSince1970: 1_800_000_000)
        defaults.set(2, forKey: PromptScheduler.Keys.totalSuccessfulActions)
        defaults.set(date.addingTimeInterval(PromptScheduler.deferralInterval),
                     forKey: PromptScheduler.Keys.deferredUntil)
        let prompt = PromptScheduler(defaults: defaults, now: { date }, quietWindow: .zero,
                                     presentationActive: true)
        prompt.startMonitoringUserActivity()
        #expect(!prompt.newsletterPresented)
    }

    @Test("Future promo and star timestamps are clamped once after a clock correction")
    func futureTimestamps() {
        let date = Date(timeIntervalSince1970: 1_800_000_000)
        var current = date
        let newsletterDefaults = defaults()
        newsletterDefaults.set(2, forKey: EffectiveUse.Keys.total)
        newsletterDefaults.set(date.addingTimeInterval(30 * 24 * 3_600),
                               forKey: PromptScheduler.PromoKeys.lastShown)
        let newsletter = PromptScheduler(defaults: newsletterDefaults, now: { current },
                                         quietWindow: .zero, presentationActive: true)
        newsletter.startMonitoringUserActivity()
        #expect(!newsletter.newsletterPresented)
        #expect(newsletterDefaults.object(forKey: PromptScheduler.PromoKeys.lastShown) as? Date == date)
        current.addTimeInterval(PromptScheduler.deferralInterval)
        let later = PromptScheduler(defaults: newsletterDefaults, now: { current },
                                    quietWindow: .zero, presentationActive: true)
        later.startMonitoringUserActivity()
        #expect(later.newsletterPresented)

        current = date
        let starDefaults = defaults()
        starDefaults.set(true, forKey: PromptScheduler.PromoKeys.newsletterCompleted)
        starDefaults.set(10, forKey: EffectiveUse.Keys.total)
        starDefaults.set(2, forKey: EffectiveUse.Keys.distinctDays)
        starDefaults.set(date.addingTimeInterval(30 * 24 * 3_600),
                         forKey: PromptScheduler.Keys.deferredUntil)
        let star = PromptScheduler(defaults: starDefaults, now: { current },
                                   quietWindow: .zero, presentationActive: true)
        star.startMonitoringUserActivity()
        #expect(!star.isPresented)
        #expect(starDefaults.object(forKey: PromptScheduler.Keys.deferredUntil) as? Date
                == date.addingTimeInterval(PromptScheduler.deferralInterval))
        current.addTimeInterval(PromptScheduler.deferralInterval)
        let nextStar = PromptScheduler(defaults: starDefaults, now: { current },
                                       quietWindow: .zero, presentationActive: true)
        nextStar.startMonitoringUserActivity()
        #expect(nextStar.isPresented)
    }
}
