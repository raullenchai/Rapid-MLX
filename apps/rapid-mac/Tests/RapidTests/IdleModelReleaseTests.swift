import Foundation
import Testing
@testable import Rapid

@Suite("Idle model release")
struct IdleModelReleaseTests {
    @Test func decision() {
        let now = Date(timeIntervalSince1970: 10_000)
        func decide(_ setting: Int = 900, _ age: TimeInterval = 901, _ busy: Bool = false, _ loaded: Bool = true, _ operating: Bool = false) -> Bool {
            IdleModelReleasePreference.shouldUnload(setting: setting, lastActivity: now.addingTimeInterval(-age), inFlight: busy, now: now, loaded: loaded, operating: operating)
        }
        #expect(!decide(0))
        #expect(!decide(900, 899))
        #expect(decide())
        #expect(!decide(900, 901, true))
        #expect(!decide(900, 901, false, false))
        #expect(!decide(900, 901, false, true, true))
    }
    @Test func defaultAndPersistence() {
        let name = "IdleModelReleaseTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        #expect(IdleModelReleasePreference.value(in: defaults) == 0)
        defaults.set(1800, forKey: IdleModelReleasePreference.storageKey)
        #expect(IdleModelReleasePreference.value(in: defaults) == 1800)
        defaults.set(123, forKey: IdleModelReleasePreference.storageKey)
        #expect(IdleModelReleasePreference.value(in: defaults) == 0)
    }
}

@Suite("Server activity idle decisions")
struct ServerActivityIdleTests {
    @Test func activityOverridesLegacyClock() {
        let now = Date()
        func decision(_ activity: ServerActivity?) -> Bool {
            IdleModelReleasePreference.shouldUnload(setting: 900, lastActivity: now.addingTimeInterval(-1000), inFlight: false, now: now, loaded: true, operating: false, activity: activity)
        }
        #expect(decision(nil))
        #expect(!decision(ServerActivity(inFlight: 1, idleSeconds: 1000)))
        #expect(!decision(ServerActivity(inFlight: 0, idleSeconds: 899)))
        #expect(decision(ServerActivity(inFlight: 0, idleSeconds: 900)))
    }

    @Test func activityAndLocalClockMustBothBeIdle() {
        let now = Date()
        let idleServer = ServerActivity(inFlight: 0, idleSeconds: 5000)
        func decision(age: TimeInterval, busy: Bool) -> Bool {
            IdleModelReleasePreference.shouldUnload(setting: 900, lastActivity: now.addingTimeInterval(-age), inFlight: busy, now: now, loaded: true, operating: false, activity: idleServer)
        }
        #expect(!decision(age: 10, busy: false))
        #expect(!decision(age: 1000, busy: true))
        #expect(decision(age: 1000, busy: false))
    }

    @Test func decodeActivityAndLegacySnapshot() throws {
        let base = "\"memory_limit_bytes\":0,\"memory_used_bytes\":0,\"idle_ttl_seconds\":0,\"loads_total\":0,\"evictions_total\":0,\"models\":[]"
        let decoder = JSONDecoder()
        let old = try decoder.decode(ModelResidencySnapshot.self, from: Data("{\(base)}".utf8))
        let new = try decoder.decode(ModelResidencySnapshot.self, from: Data("{\(base),\"activity\":{\"in_flight\":2,\"idle_seconds\":0}}".utf8))
        #expect(old.activity == nil)
        #expect(new.activity == ServerActivity(inFlight: 2, idleSeconds: 0))
    }
}
