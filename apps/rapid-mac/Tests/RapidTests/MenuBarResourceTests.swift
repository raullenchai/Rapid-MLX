import Foundation
import Testing
@testable import Rapid

@MainActor
struct MenuBarResourceTests {
    private func resident(_ id: String, state: String = "resident", pinned: Bool = false,
                          primary: Bool = false, active: Int = 0, measured: UInt64? = nil) -> ResidentModelStatus {
        ResidentModelStatus(id: id, modelPath: "org/\(id)", aliases: [id], modality: "text",
                            state: state, pinned: pinned, primary: primary, activeRequests: active,
                            estimatedBytes: 2_000_000_000, measuredBytes: measured, idleSeconds: 0)
    }

    private func snapshot(_ models: [ResidentModelStatus], used: UInt64 = 4_000_000_000,
                          activity: ServerActivity? = nil,
                          audio: [ResidentAudioLaneStatus] = []) -> ModelResidencySnapshot {
        ModelResidencySnapshot(memoryLimitBytes: 12_000_000_000, memoryUsedBytes: used,
                               memoryAvailableBytes: nil, idleTTLSeconds: 0, loadsTotal: 0,
                               evictionsTotal: 0, models: models, audioLanes: audio, activity: activity)
    }

    @Test func idleAndUnavailableAreDifferent() {
        let unavailable = MenuBarResourceModel(residency: nil, totalBytes: nil, status: nil)
        #expect(unavailable.memoryLabel == "— / —")
        #expect(unavailable.activityLabel == "—")
        #expect(unavailable.rows.isEmpty)
        #expect(unavailable.safety == nil)
        let idle = MenuBarResourceModel(residency: .empty, totalBytes: 16_000_000_000, status: nil)
        #expect(idle.memoryLabel == "0.00 GB / 16.00 GB")
        #expect(idle.activityLabel == "Idle")
        #expect(!idle.canUnloadAll)
    }

    @Test func oneAndTwoModelsUseEndpointBytesAndPhysicalRAM() {
        for models in [[resident("first")], [resident("first"), resident("second", measured: 3_000_000_000)]] {
            let model = MenuBarResourceModel(residency: snapshot(models), totalBytes: 16_000_000_000, status: nil)
            #expect(model.rows.map(\.id) == models.map(\.id))
            #expect(model.rows.first?.memoryLabel == "2.00 GB")
            #expect(model.memoryLabel == "4.00 GB / 16.00 GB")
            #expect(model.fraction == 0.25)
            #expect(model.canUnloadAll)
            if models.count == 2 { #expect(model.rows.last?.memoryLabel == "3.00 GB") }
        }
    }

    @Test func onlyMemoryHoldingModelsAppearAndUnloadRespectsServerGuards() {
        let models = [resident("idle"), resident("busy", active: 1), resident("pinned", pinned: true),
                      resident("primary", primary: true), resident("loading", state: "loading"),
                      resident("standby", state: "unloaded"), resident("evicting", state: "evicting")]
        let model = MenuBarResourceModel(residency: snapshot(models), totalBytes: nil, status: nil)
        #expect(model.rows.map(\.id) == ["idle", "busy", "pinned", "primary", "loading"])
        #expect(model.rows.map(\.canUnload) == [true, false, false, false, false])
        #expect(!model.canUnloadAll)
        #expect(model.rows[4].memoryLabel == "Reloading…")
        #expect(model.fraction == nil)
    }

    @Test func thresholdsAreExactlyTheMemoryPreflightThresholds() {
        for used: UInt64 in [94, 95, 100, 101] {
            let model = MenuBarResourceModel(residency: snapshot([], used: used), totalBytes: 100, status: nil)
            #expect(model.safety == ModelSizing.memorySafety(footprintGB: 0, usedBytes: used, totalBytes: 100))
        }
        #expect(MenuBarResourceModel(residency: snapshot([], used: 101), totalBytes: 100, status: nil).fraction == 1)
        #expect(MenuBarResourceModel(residency: .empty, totalBytes: 0, status: nil).safety == nil)
    }

    @Test func generationRateComesFromStatusAndDisappearsWhenIdle() throws {
        let status = try MetalMemoryClient.decodeStatus(Data(#"{"num_running":1,"generation_tps":42.5}"#.utf8))
        #expect(status?.generationTPS == 42.5)
        let active = MenuBarResourceModel(residency: snapshot([], activity: ServerActivity(inFlight: 1, idleSeconds: 0)),
                                          totalBytes: nil, status: status)
        #expect(active.isActive)
        #expect(active.activityLabel.contains("42.5 tok/s"))
        #expect(!active.canUnloadAll)
        let idle = MenuBarResourceModel(residency: .empty, totalBytes: nil,
                                       status: MetalMemorySnapshot(active: nil, cache: nil, peak: nil, numRunning: 0, generationTPS: 42.5))
        #expect(idle.activityLabel == "Idle")
        for rate in [Double.nan, .infinity, -1] {
            let model = MenuBarResourceModel(residency: snapshot([resident("busy", active: 1)]), totalBytes: nil,
                                            status: MetalMemorySnapshot(active: nil, cache: nil, peak: nil, generationTPS: rate))
            #expect(model.activityLabel == "Generating")
        }
    }

    @Test func audioAndExternalRequestsCountAsActivity() {
        let audio = ResidentAudioLaneStatus(lane: "tts", model: "voice", state: "resident", activeRequests: 1)
        let model = MenuBarResourceModel(residency: snapshot([], audio: [audio]), totalBytes: nil, status: nil)
        #expect(model.isActive)
        #expect(!model.canUnloadAll)
        #expect(model.audioRows.count == 1)
        #expect(model.audioRows.first?.memoryLabel == "—")
    }

    @Test func statusOnlyRateIsRetainedAndNegativeRateIsUnavailable() throws {
        let rateOnly = try MetalMemoryClient.decodeStatus(Data(#"{"generation_tps":42.5}"#.utf8))
        #expect(rateOnly != nil)
        #expect(rateOnly?.generationTPS == 42.5)
        let negative = try MetalMemoryClient.decodeStatus(Data(#"{"generation_tps":-1}"#.utf8))
        #expect(negative?.generationTPS == nil)
    }
}
