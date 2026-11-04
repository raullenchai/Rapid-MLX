import Foundation
import Testing
@testable import Rapid

@Suite("Idle model release")
struct IdleModelReleaseTests {
    @Test func modes() {
        #expect(IdleModelReleasePreference.choices == [0, 300, 900, 1800, 3600])
        for supportsNative in [false, true] {
            #expect(IdleModelReleasePreference.mode(setting: 0, runtimeSupportsNativeUnload: supportsNative) == .off)
            #expect(IdleModelReleasePreference.mode(setting: 123, runtimeSupportsNativeUnload: supportsNative) == .off)
            for setting in [300, 900, 1800, 3600] {
                #expect(IdleModelReleasePreference.mode(setting: setting, runtimeSupportsNativeUnload: supportsNative)
                    == (supportsNative ? .native : .legacyStop))
            }
        }
    }

    @Test func nativeUnloadStillStopsForIdleAudioLanes() throws {
        // Native unload frees chat models with the API online; only a
        // resident audio lane (no engine idle release) keeps the stop path.
        #expect(!IdleModelReleasePreference.requiresProcessStop(nativeUnloadActive: true, residentAudioLanes: 0))
        #expect(IdleModelReleasePreference.requiresProcessStop(nativeUnloadActive: true, residentAudioLanes: 1))
        #expect(IdleModelReleasePreference.requiresProcessStop(nativeUnloadActive: false, residentAudioLanes: 0))

        func model(_ state: String) throws -> ResidentModelStatus {
            try JSONDecoder().decode(ResidentModelStatus.self, from: Data("""
            {"id":"m","model_path":"m","aliases":[],"modality":"text","state":"\(state)",
             "pinned":true,"primary":true,"active_requests":0,"estimated_bytes":1,
             "measured_bytes":null,"idle_seconds":1}
            """.utf8))
        }
        // A standby primary is idle under native unload but blocks the legacy path.
        #expect(!IdleModelReleasePreference.blocksProcessStop(try model("standby"), nativeUnloadActive: true))
        #expect(IdleModelReleasePreference.blocksProcessStop(try model("standby"), nativeUnloadActive: false))
        for state in ["loading", "unloading", "evicting"] {
            #expect(IdleModelReleasePreference.blocksProcessStop(try model(state), nativeUnloadActive: true))
        }
        #expect(!IdleModelReleasePreference.blocksProcessStop(try model("resident"), nativeUnloadActive: true))
        #expect(!IdleModelReleasePreference.blocksProcessStop(try model("resident"), nativeUnloadActive: false))
    }

    @Test func restartOnChange() {
        func decide(_ setting: Int = 300, _ spawned: Int? = 0, ready: Bool = true,
                    operating: Bool = false, requests: Bool = false,
                    loads: Bool = false, alias: String? = "chat") -> IdleModelReleasePreference.RestartDecision {
            IdleModelReleasePreference.restartDecision(
                setting: setting, secondsAtSpawn: spawned, isReady: ready,
                isOperating: operating, hasActiveRequests: requests,
                hasResidentLoadsInFlight: loads, alias: alias
            )
        }
        #expect(decide() == .restart(alias: "chat"))
        #expect(decide(0, 300) == .restart(alias: "chat"))
        #expect(decide(300, 300) == .none)
        #expect(decide(300, nil) == .none)
        #expect(decide(123) == .none)
        #expect(decide(ready: false) == .none)
        #expect(decide(alias: nil) == .none)
        #expect(decide(operating: true) == .deferred)
        #expect(decide(requests: true) == .deferred)
        #expect(decide(loads: true) == .deferred)
        #expect(decide(alias: "preferred-text") == .restart(alias: "preferred-text"))
    }

    @Test func decision() {
        let now = Date(timeIntervalSince1970: 10_000)
        func decide(_ setting: Int = 900, _ age: TimeInterval = 901, _ busy: Bool = false, _ loaded: Bool = true, _ operating: Bool = false) -> Bool {
            IdleModelReleasePreference.shouldUnload(setting: setting, lastActivity: now.addingTimeInterval(-age), inFlight: busy, now: now, loaded: loaded, operating: operating, activity: ServerActivity(inFlight: 0, idleSeconds: 5000))
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
        defaults.set(300, forKey: IdleModelReleasePreference.storageKey)
        #expect(IdleModelReleasePreference.value(in: defaults) == 300)
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
        #expect(!decision(nil))  // no server activity clock: never auto-release
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
