import Foundation

/// Relative "how long ago" stamp for the message hover toolbar.
///
/// A pure static formatter with the reference date INJECTED, in the
/// ``AssistantStatsFormatter`` mould — deliberately not
/// `RelativeDateTimeFormatter`, which is locale/now-dependent and
/// `@MainActor`-cached in this codebase, so its exact switchovers
/// can't be pinned by tests. The spec is the user's own:
/// minutes for the first hour, hours for the first day, then days,
/// and onward through weeks / months / years.
enum RelativeTimestamp {
    /// Compact toolbar form: "just now", "12m ago", "5h ago",
    /// "3d ago", "2w ago", "4mo ago", "1y ago".
    static func label(from created: Date, to now: Date) -> String {
        let (value, unit) = bucket(from: created, to: now)
        switch unit {
        case .now: return String(localized: "just now")
        case .minutes: return String(localized: "\(value)m ago")
        case .hours: return String(localized: "\(value)h ago")
        case .days: return String(localized: "\(value)d ago")
        case .weeks: return String(localized: "\(value)w ago")
        case .months: return String(localized: "\(value)mo ago")
        case .years: return String(localized: "\(value)y ago")
        }
    }

    /// Spelled-out VoiceOver form: "12 minutes ago", "1 hour ago".
    static func accessibilityLabel(from created: Date, to now: Date) -> String {
        let (value, unit) = bucket(from: created, to: now)
        switch unit {
        case .now: return String(localized: "just now")
        case .minutes: return value == 1 ? String(localized: "1 minute ago") : String(localized: "\(value) minutes ago")
        case .hours: return value == 1 ? String(localized: "1 hour ago") : String(localized: "\(value) hours ago")
        case .days: return value == 1 ? String(localized: "1 day ago") : String(localized: "\(value) days ago")
        case .weeks: return value == 1 ? String(localized: "1 week ago") : String(localized: "\(value) weeks ago")
        case .months: return value == 1 ? String(localized: "1 month ago") : String(localized: "\(value) months ago")
        case .years: return value == 1 ? String(localized: "1 year ago") : String(localized: "\(value) years ago")
        }
    }

    enum Unit { case now, minutes, hours, days, weeks, months, years }

    /// The single source of truth for the switchover points, so the
    /// two render forms can never disagree. Calendar-free fixed
    /// divisors (30-day month, 365-day year) — the toolbar stamp is a
    /// rough age indicator, not a date computation, and fixed divisors
    /// are what make the boundaries pinnable.
    static func bucket(from created: Date, to now: Date) -> (value: Int, unit: Unit) {
        let seconds = max(0, now.timeIntervalSince(created))
        if seconds < 60 { return (0, .now) }
        let minutes = Int(seconds / 60)
        if minutes < 60 { return (minutes, .minutes) }
        let hours = Int(seconds / 3_600)
        if hours < 24 { return (hours, .hours) }
        let days = Int(seconds / 86_400)
        if days < 7 { return (days, .days) }
        let weeks = Int(seconds / 604_800)
        if weeks < 5 { return (weeks, .weeks) }
        let months = Int(seconds / 2_592_000)
        if months < 12 { return (months, .months) }
        return (Int(seconds / 31_536_000), .years)
    }
}
