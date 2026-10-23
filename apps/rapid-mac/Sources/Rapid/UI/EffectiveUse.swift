import Foundation

/// Local record of successful outcomes. Calendar days use the Mac's current calendar.
@MainActor
final class EffectiveUse {
    enum Keys {
        static let total = "Rapid.effectiveUse.total.v1"
        static let firstDay = "Rapid.effectiveUse.firstDay.v1"
        static let lastDay = "Rapid.effectiveUse.lastDay.v1"
        static let distinctDays = "Rapid.effectiveUse.distinctDays.v1"
    }

    private let defaults: UserDefaults
    private let calendar: Calendar
    private let now: () -> Date

    init(defaults: UserDefaults = .standard, calendar: Calendar = .current,
         now: @escaping () -> Date = Date.init) {
        self.defaults = defaults
        self.calendar = calendar
        self.now = now
        if defaults.object(forKey: Keys.total) == nil {
            let previous = max(0, defaults.integer(forKey: PromptScheduler.Keys.totalSuccessfulActions))
            defaults.set(previous, forKey: Keys.total)
            if previous > 0 {
                let today = calendar.startOfDay(for: now())
                // The old counter has no dates. Preserve mature installs' eligibility.
                defaults.set(calendar.date(byAdding: .day, value: -1, to: today), forKey: Keys.firstDay)
                defaults.set(today, forKey: Keys.lastDay)
                defaults.set(2, forKey: Keys.distinctDays)
            }
        }
    }

    var total: Int { defaults.integer(forKey: Keys.total) }
    var distinctDays: Int { defaults.integer(forKey: Keys.distinctDays) }
    var firstDay: Date? { defaults.object(forKey: Keys.firstDay) as? Date }

    func count(_ kind: ProductValueKind) -> Int {
        defaults.integer(forKey: "Rapid.effectiveUse.\(kind.storageKey).v1")
    }

    func record(_ kind: ProductValueKind) {
        let day = calendar.startOfDay(for: now())
        if firstDay == nil { defaults.set(day, forKey: Keys.firstDay) }
        if let last = defaults.object(forKey: Keys.lastDay) as? Date {
            if !calendar.isDate(last, inSameDayAs: day) {
                defaults.set(distinctDays == Int.max ? Int.max : distinctDays + 1,
                             forKey: Keys.distinctDays)
                defaults.set(day, forKey: Keys.lastDay)
            }
        } else {
            defaults.set(1, forKey: Keys.distinctDays)
            defaults.set(day, forKey: Keys.lastDay)
        }
        defaults.set(total == Int.max ? Int.max : total + 1, forKey: Keys.total)
        let key = "Rapid.effectiveUse.\(kind.storageKey).v1"
        let count = defaults.integer(forKey: key)
        defaults.set(count == Int.max ? Int.max : count + 1, forKey: key)
    }
}
