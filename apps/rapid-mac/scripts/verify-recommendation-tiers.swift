#!/usr/bin/env swift
//
// verify-recommendation-tiers.swift — runnable contract check for the
// RAM-tier model recommendation (RAMBucketedDefault).
//
// The Tests/RapidTests XCTest/Swift-Testing suite is excluded from the
// SwiftPM manifest (see Package.swift) and `import Testing` isn't
// resolvable from the command line in this toolchain, so this dependency-
// free script is the CI-runnable verification of the recommendation
// contract: run `swift apps/rapid-mac/scripts/verify-recommendation-tiers.swift`.
//
// It decodes the same repository-wide JSON SSOT as RAMBucketedDefault and the
// Python CLI, then exercises the desktop-specific selection contracts.
//

import Foundation

// Faithful copies of the pure logic under test (RapidTests is excluded
// from Package.swift, so this standalone script is the real verification).

struct Pick: Equatable, Decodable {
    let role: String
    let alias: String
    let footprintGB: Double
    let capabilityPct: Int
    let tokensPerSec: Double?
    let launchFlags: [String]
    let caveat: String?
    enum CodingKeys: String, CodingKey {
        case role, alias, caveat
        case footprintGB = "footprint_gb"
        case capabilityPct = "capability_pct"
        case tokensPerSec = "tokens_per_sec"
        case launchFlags = "launch_flags"
    }
}
struct Tier: Equatable, Decodable {
    let floorGB: Double
    let picks: [Pick]
    var primary: Pick { picks[0] }
    var alt: Pick? { picks.count > 1 ? picks[1] : nil }
    enum CodingKeys: String, CodingKey {
        case floorGB = "floor_gb"
        case picks
    }
}
struct Payload: Decodable { let tiers: [Tier] }
let sourceURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("vllm_mlx/model_recommendations.json")
let tiers = try JSONDecoder().decode(Payload.self, from: Data(contentsOf: sourceURL)).tiers
// Mirror SettingsModelManagementPanel.pickStatsLine / ModelPickerBar
// tagline: a caveat replaces the capability %, speed leads, caveat trails.
func pickStatsLine(_ pick: Pick) -> String {
    var parts = [String(format: "%.1f GB", pick.footprintGB)]
    if let caveat = pick.caveat {
        if let tps = pick.tokensPerSec { parts.append("~\(Int(tps.rounded())) tok/s") }
        parts.append(caveat)
    } else {
        parts.append("\(pick.capabilityPct)% capability")
        if let tps = pick.tokensPerSec { parts.append("~\(Int(tps.rounded())) tok/s") }
    }
    return parts.joined(separator: " · ")
}
func tier(forPhysicalRAMGB ram: Double) -> Tier {
    var chosen = tiers[0]
    for c in tiers where ram >= c.floorGB { chosen = c }
    return chosen
}
func alias(forPhysicalRAMGB ram: Double) -> String { tier(forPhysicalRAMGB: ram).primary.alias }
func picks(forPhysicalRAMGB ram: Double) -> [Pick] { tier(forPhysicalRAMGB: ram).picks }
func launchFlags(forAlias a: String, physicalRAMGB ram: Double) -> [String] {
    tier(forPhysicalRAMGB: ram).picks.first(where: { $0.alias == a })?.launchFlags ?? []
}
func isRecommendedPick(alias a: String, physicalRAMGB ram: Double) -> Bool {
    let t = tier(forPhysicalRAMGB: ram)
    return ram >= t.floorGB && t.picks.contains { $0.alias == a }
}
// Mirror ServerManager.serveArguments append order.
func serveArguments(alias a: String, host: String, port: Int, extraFlags: [String]) -> [String] {
    var args = ["serve", a, "--host", host, "--port", String(port),
                "--cors-origins", "http://127.0.0.1", "http://localhost"]
    args.append(contentsOf: extraFlags)
    return args
}

var fails = 0
func check(_ c: Bool, _ m: String) { if c { print("  ✓ \(m)") } else { print("  ✗ FAIL: \(m)"); fails += 1 } }

print("Tier mapping (round DOWN to nearest floor):")
check(alias(forPhysicalRAMGB: 8)  == "lfm2.5-2.6b-4bit",    "8GB has its own tier")
check(alias(forPhysicalRAMGB: 4)  == "lfm2.5-2.6b-4bit",    "sub-8GB clamps DOWN to the 8 tier")
// MacHardware can report 0 on a probe failure; it must not crash or pick a
// 122B model. Mirrors RAMBucketedDefaultTests.pathologicalRAM.
check(alias(forPhysicalRAMGB: 0)  == "lfm2.5-2.6b-4bit",    "0GB clamps to the smallest tier")
check(alias(forPhysicalRAMGB: -1) == "lfm2.5-2.6b-4bit",    "negative RAM clamps to the smallest tier")
check(alias(forPhysicalRAMGB: 15) == "lfm2.5-2.6b-4bit",    "15GB rounds DOWN to 8, not up to 16")
check(alias(forPhysicalRAMGB: 16) == "qwen3.5-4b-4bit",     "16GB → qwen3.5-4b")
check(alias(forPhysicalRAMGB: 17) == "qwen3.5-4b-4bit",     "17GB → still 16 tier")
check(alias(forPhysicalRAMGB: 18) == "qwen3.5-9b-4bit",     "18GB → qwen3.5-9b")
check(alias(forPhysicalRAMGB: 20) == "qwen3.5-9b-4bit",     "20GB rounds DOWN to 18 (qwen9), not 24")
check(alias(forPhysicalRAMGB: 24) == "bonsai-27b-2bit",     "24GB → bonsai-27b")
check(alias(forPhysicalRAMGB: 30) == "bonsai-27b-2bit",     "30GB → 24 tier")
check(alias(forPhysicalRAMGB: 32) == "qwen3.8-27b-4bit",    "32GB → qwen3.8-27b")
check(alias(forPhysicalRAMGB: 48) == "qwen3.8-27b-4bit",    "48GB → qwen3.8-27b")
check(alias(forPhysicalRAMGB: 64) == "qwen3.8-27b-4bit",    "64GB → qwen3.8-27b")
check(alias(forPhysicalRAMGB: 96) == "qwen3.8-27b-4bit",    "96GB → qwen3.8-27b")
check(alias(forPhysicalRAMGB: 256) == "qwen3.8-27b-4bit",   "256GB → 96 tier (qwen3.8-27b)")

print("Smart + fast picks per tier:")
for floor in [8.0, 16.0, 18.0, 24.0, 32.0, 48.0, 64.0, 96.0] {
    check(picks(forPhysicalRAMGB: floor).count == 2, "\(Int(floor))GB has exactly two choices")
}
check(picks(forPhysicalRAMGB: 18).count == 2, "18GB has smart + fast")
check(picks(forPhysicalRAMGB: 18)[1].alias == "qwen3.5-4b-4bit", "18GB fast is qwen3.5-4b")
check(picks(forPhysicalRAMGB: 24)[0].alias == "bonsai-27b-2bit", "24GB smart is bonsai")
check(picks(forPhysicalRAMGB: 32)[0].alias == "qwen3.8-27b-4bit", "32GB smart is qwen3.8-27b")
check(picks(forPhysicalRAMGB: 48)[1].alias == "qwen3.6-35b-4bit", "48GB retains reviewed Qwen3.6 fast pick")
// 64/96 GB → fast alt is the lighter 4-bit Qwen3.6-35B.
check(picks(forPhysicalRAMGB: 64).count == 2, "64GB has smart + fast")
check(picks(forPhysicalRAMGB: 64)[1].alias == "qwen3.6-35b-4bit", "64GB fast is qwen3.6-35b-4bit")
check(picks(forPhysicalRAMGB: 96).count == 2, "96GB has smart + fast")
check(picks(forPhysicalRAMGB: 96)[1].alias == "qwen3.6-35b-4bit", "96GB fast is qwen3.6-35b-4bit")
check(picks(forPhysicalRAMGB: 256)[1].alias == "qwen3.6-35b-4bit", "256GB (96 tier) fast is qwen3.6-35b-4bit")

print("Capability column: a smart pick never DISPLAYS weaker than its own fast alt (codex r8 MAJOR):")
// 64 GB: the smart pick (qwen3.8-27b-4bit) reads 92 %, above its
// qwen3.6-35b-4bit alt's 87 % — "Best pick" must never look worse than
// "Faster". Pin smart >= alt for every tier that carries an alt.
for ram in [8.0, 16.0, 18.0, 24.0, 32.0, 48.0, 64.0, 96.0] {
    let ps = picks(forPhysicalRAMGB: ram)
    if ps.count == 2, ps[1].caveat == nil {
        // Only compare when the alt shows a %; a caveat alt (lfm2.5) opts out.
        check(ps[0].capabilityPct >= ps[1].capabilityPct,
              "\(Int(ram))GB smart (\(ps[0].capabilityPct)%) not shown below its fast alt (\(ps[1].capabilityPct)%)")
    }
}
check(picks(forPhysicalRAMGB: 64)[0].capabilityPct == 92, "64GB smart (qwen3.8-27b) reads 92%, above its 35b-4bit alt")

print("Smart-pick capability is monotonic non-decreasing by RAM (no 'more RAM, worse pick' dip):")
let floors = [8.0, 16.0, 18.0, 24.0, 32.0, 48.0, 64.0, 96.0]
for (a, b) in zip(floors, floors.dropFirst()) {
    let ca = picks(forPhysicalRAMGB: a)[0].capabilityPct
    let cb = picks(forPhysicalRAMGB: b)[0].capabilityPct
    check(cb >= ca, "\(Int(b))GB smart (\(cb)%) >= \(Int(a))GB smart (\(ca)%)")
}
// 18 GB steps up from 16 GB (qwen3.5-9b smart at 82 %, over 16 GB's
// qwen3.5-4b at 78 %); the old gemma-4-12b pick was dropped from the tiers.
check(picks(forPhysicalRAMGB: 18)[0].alias == "qwen3.5-9b-4bit" && picks(forPhysicalRAMGB: 18)[0].capabilityPct == 82, "18GB smart = qwen9 82%")
check(!tiers.flatMap(\.picks).contains { $0.alias == "gemma-4-12b-4bit" }, "gemma-4-12b is no longer a tier pick")

print("Smallest tier shows an honest capability caveat:")
// Liquid publishes 2.6B as "not recommended for agentic coding" — our users
// drive coding agents, so the card must say so instead of showing a bare %.
let smart8 = picks(forPhysicalRAMGB: 8)[0]
check(picks(forPhysicalRAMGB: 8).count == 2, "8GB has smart + fast")
check(smart8.caveat == "Not for coding", "8GB pick carries the coding caveat")
check(pickStatsLine(smart8) == "3.0 GB · ~94 tok/s · Not for coding", "8GB stats line drops the % for the caveat")
check(smart8.footprintGB < 4.0, "8GB pick must leave room for macOS on an 8GB machine")
let fast96 = picks(forPhysicalRAMGB: 96)[1]
check(fast96.caveat == nil, "96GB fast (qwen3.6-35b-4bit) is general-purpose — no caveat")
check(pickStatsLine(fast96) == "20.0 GB · 87% capability · ~60 tok/s", "general-purpose fast pick keeps its capability %")
let smart16 = picks(forPhysicalRAMGB: 16)[0]
check(pickStatsLine(smart16) == "6.0 GB · 78% capability · ~61 tok/s", "16GB qwen4 renders measured 8K peak")
let smart96 = picks(forPhysicalRAMGB: 96)[0]
check(pickStatsLine(smart96) == "20.0 GB · 92% capability · ~41 tok/s", "96GB smart renders measured 8K peak + speed")

print("Launch flags travel with the recommendation, gated by RAM:")
check(launchFlags(forAlias: "lfm2.5-2.6b-4bit", physicalRAMGB: 8).isEmpty, "8GB lfm2.5-2.6b → no flags")
check(launchFlags(forAlias: "qwen3.5-9b-4bit", physicalRAMGB: 18).isEmpty, "18GB qwen9 → no flags")
// No tier carries launch flags since the gemma pick retired with the
// AA-Index roster — qwen3.8-27b-4bit runs bare (MTP lives in the alias).
// The gating machinery is still exercised via non-pick and foreign-tier lookups.
check(launchFlags(forAlias: "qwen3.8-27b-4bit", physicalRAMGB: 32).isEmpty, "32GB qwen3.8-27b → no flags")
check(launchFlags(forAlias: "qwen3.6-35b-4bit", physicalRAMGB: 48).isEmpty, "48GB reviewed 35b-4bit → no flags")
check(launchFlags(forAlias: "gemma-4-26b-4bit", physicalRAMGB: 64).isEmpty, "gemma-26b on 64GB (not a pick) keeps vision")
check(launchFlags(forAlias: "some-random", physicalRAMGB: 32).isEmpty, "unknown alias → no flags")

print("isRecommendedPick is floor-gated (the floor is now 8 GB, not 16):")
check(isRecommendedPick(alias: "qwen3.5-4b-4bit", physicalRAMGB: 16), "qwen4 IS recommended on 16GB")
check(isRecommendedPick(alias: "lfm2.5-2.6b-4bit", physicalRAMGB: 8), "lfm2.5-2.6b IS recommended on 8GB (its own tier)")
check(!isRecommendedPick(alias: "bonsai-27b-2bit", physicalRAMGB: 8), "bonsai still NOT recommended on 8GB (not in the 8 tier)")
check(!isRecommendedPick(alias: "lfm2.5-2.6b-4bit", physicalRAMGB: 4), "sub-floor 4GB Mac gets NO exemption even from its own tier pick")
check(isRecommendedPick(alias: "qwen3.5-9b-4bit", physicalRAMGB: 18), "qwen9 recommended on 18GB")
check(!isRecommendedPick(alias: "bonsai-27b-2bit", physicalRAMGB: 18), "bonsai is not recommended below 24GB")
check(!isRecommendedPick(alias: "gemma-4-12b-4bit", physicalRAMGB: 18), "gemma-12b NOT recommended anywhere (dropped from picks)")
check(isRecommendedPick(alias: "bonsai-27b-2bit", physicalRAMGB: 24), "bonsai recommended on 24GB")
check(isRecommendedPick(alias: "qwen3.8-27b-4bit", physicalRAMGB: 32), "qwen3.8-27b recommended on 32GB")
check(isRecommendedPick(alias: "qwen3.8-27b-4bit", physicalRAMGB: 96), "qwen3.8-27b recommended on 96GB")
check(!isRecommendedPick(alias: "qwen3.5-122b-mxfp4", physicalRAMGB: 96), "122b-mxfp4 no longer a pick anywhere (AA-Index roster)")
check(!isRecommendedPick(alias: "qwen3.8-27b-4bit", physicalRAMGB: 24), "qwen3.8-27b NOT recommended below 32GB (8K peak 20 GB misses the 24 GB tier budget)")
// The fast alt is a recommended pick on its tiers too (skips the .tooBig gate).
check(isRecommendedPick(alias: "qwen3.6-35b-4bit", physicalRAMGB: 64), "qwen3.6-35b-4bit fast alt recommended on 64GB")
check(isRecommendedPick(alias: "qwen3.6-35b-4bit", physicalRAMGB: 96), "qwen3.6-35b-4bit fast alt recommended on 96GB")
check(launchFlags(forAlias: "qwen3.6-35b-4bit", physicalRAMGB: 96).isEmpty, "fast alt qwen3.6-35b-4bit → no forced flags")

// Mirror the `.tooBig && !isRecommendedPick` predicate shared by every
// start path that consults ModelSizing: ContentView.runLaunchAutoStart's
// AutoStartDecision `rejectsAlias`, ModelPickerBar.handleStartTap, the
// ContentView switch gate, and CacheAwareDefault.bucketedFits. A
// recommended pick trusts the curated footprint, so it is NEVER rejected
// even when ModelSizing over-estimates it as .tooBig.
func rejectsForAutoStart(alias a: String, modelSizingSaysTooBig tooBig: Bool, physicalRAMGB ram: Double) -> Bool {
    tooBig && !isRecommendedPick(alias: a, physicalRAMGB: ram)
}

print("Launch auto-start never rejects the curated recommendation (codex r6 MAJOR):")
check(rejectsForAutoStart(alias: "bonsai-27b-2bit", modelSizingSaysTooBig: true, physicalRAMGB: 16),
      "16GB: bonsai is no longer exempt from ModelSizing .tooBig")
check(rejectsForAutoStart(alias: "bonsai-27b-2bit", modelSizingSaysTooBig: true, physicalRAMGB: 8),
      "8GB: bonsai is not this tier's pick → auto-start still rejects .tooBig")
// The hole this tier closes: before it, EVERY pick offered to an 8 GB Mac was
// rejected by auto-start, so first run fell through to SafeDefaultFallback.
check(!rejectsForAutoStart(alias: "lfm2.5-2.6b-4bit", modelSizingSaysTooBig: true, physicalRAMGB: 8),
      "8GB: its own tier pick survives auto-start (the hole this tier closes)")
check(rejectsForAutoStart(alias: "some-huge-70b", modelSizingSaysTooBig: true, physicalRAMGB: 16),
      "non-recommended .tooBig alias is still rejected")
check(!rejectsForAutoStart(alias: "some-small-model", modelSizingSaysTooBig: false, physicalRAMGB: 16),
      "a model that fits is never rejected")

print("serveArguments appends flags AFTER cors-origins:")
let noFlag = serveArguments(alias: "qwen3.6-35b-4bit", host: "127.0.0.1", port: 8000, extraFlags: [])
check(noFlag == ["serve", "qwen3.6-35b-4bit", "--host", "127.0.0.1", "--port", "8000", "--cors-origins", "http://127.0.0.1", "http://localhost"], "no-flag arg shape unchanged (SpawnArgumentsTests contract)")
let withFlag = serveArguments(alias: "gemma-4-26b-4bit", host: "127.0.0.1", port: 8000, extraFlags: ["--no-mllm", "--kv-cache-dtype", "bf16", "--cache-memory-mb", "512"])
check(withFlag.suffix(5) == ["--no-mllm", "--kv-cache-dtype", "bf16", "--cache-memory-mb", "512"], "flags trail the array")
check(withFlag.firstIndex(of: "--no-mllm")! > withFlag.firstIndex(of: "--cors-origins")!, "--no-mllm comes after --cors-origins (terminates nargs)")

// ---------------------------------------------------------------------------
// Quickstart eligibility — the retired-starter carve-out
//
// Faithful copy of QuickstartCoordinator.isEligible + retiredStarters. The
// Python contract test pins the *contents* of retiredStarters against the
// source text; it cannot execute the gate, so inverting the condition or
// dropping the `done` check would stay green there. These cases are the
// executable half.

enum FakeServerState { case idle, stopped, ready, starting, crashed, missing }

let retiredStarters: Set<String> = ["bonsai-1.7b-2bit"]

func isStranded(_ lastServedAlias: String?) -> Bool {
    guard let alias = lastServedAlias else { return false }
    return retiredStarters.contains(alias)
}

// Gates 1 + 2 only — the persisted "does this install still owe the user
// onboarding?" half. #1589 split this out of isEligible in the source so the
// launch auto-start path could ask the SAME question before it moves
// serverState; the mirror follows the same shape.
func onboardingOwed(done: Bool, legacyDone: Bool = false, lastServedAlias: String?) -> Bool {
    guard !done else { return false }
    let stranded = isStranded(lastServedAlias)
    guard !(legacyDone && !stranded) else { return false }
    if lastServedAlias != nil, !stranded {
        return false
    }
    return true
}

func isEligible(done: Bool, legacyDone: Bool = false, lastServedAlias: String?, serverState: FakeServerState) -> Bool {
    guard onboardingOwed(
        done: done,
        legacyDone: legacyDone,
        lastServedAlias: lastServedAlias
    ) else { return false }
    switch serverState {
    case .idle, .stopped: return true
    case .ready, .starting, .crashed, .missing: return false
    }
}

print("Quickstart eligibility:")
check(isEligible(done: false, lastServedAlias: nil, serverState: .idle),
      "brand-new user (no serve yet) sees the card")
check(isEligible(done: false, lastServedAlias: "bonsai-1.7b-2bit", serverState: .idle),
      "stranded on the retired starter → card returns (the point of the carve-out)")
check(!isEligible(done: false, lastServedAlias: "qwen3.5-9b-4bit", serverState: .idle),
      "traded up to another model → never re-onboarded")
check(!isEligible(done: false, lastServedAlias: "lfm2.5-1b-4bit", serverState: .idle),
      "already on the CURRENT starter → not re-onboarded (no onboarding loop)")
check(!isEligible(done: true, lastServedAlias: "bonsai-1.7b-2bit", serverState: .idle),
      "done flag still wins over the carve-out — dismissal is permanent")
check(!isEligible(done: true, lastServedAlias: nil, serverState: .idle),
      "done flag wins for a new user too")
check(!isEligible(done: false, legacyDone: true, lastServedAlias: nil, serverState: .idle),
      "dismissed under v1, never served → the v2 bump must NOT resurrect the card")
check(!isEligible(done: false, legacyDone: true, lastServedAlias: "qwen3.5-9b-4bit", serverState: .idle),
      "dismissed under v1 and on another model → still dismissed")
check(isEligible(done: false, legacyDone: true, lastServedAlias: "bonsai-1.7b-2bit", serverState: .idle),
      "dismissed under v1 but stranded on the retired starter → rescued anyway")
for busy in [FakeServerState.ready, .starting, .crashed, .missing] {
    check(!isEligible(done: false, lastServedAlias: "bonsai-1.7b-2bit", serverState: busy),
          "server busy (\(busy)) suppresses the card even for the stranded cohort")
}

// ---------------------------------------------------------------------------
// Auto-start must not resume a retired starter
//
// Auto-start defaults to ON. Without this guard the rescue above is
// decorative: the stranded user launches, we restart the broken model,
// serverState leaves .idle, and Quickstart's third gate hides the card.
// Mirrors the ordering in AutoStartDecision.decide — resolution first, then
// the retired check, then the on-disk check.

enum FakeDecision: Equatable { case start(String), promptDownload(String), skip(String) }

func isRetiredStarter(_ alias: String) -> Bool { retiredStarters.contains(alias) }

func decideResume(lastServedAlias: String?, cachedAliases: Set<String>,
                  serverState: FakeServerState, userOptedIn: Bool = true,
                  quickstartDone: Bool = false, legacyDone: Bool = false,
                  firstRunDecisionPending: Bool = false) -> FakeDecision {
    if !userOptedIn { return .skip("userOptedOut") }
    // #1589: both first-run gates sit ABOVE the serverState switch. Below it
    // they could only observe the race, never prevent it — auto-start is the
    // thing that moves the state every downstream predicate then reads.
    if firstRunDecisionPending { return .skip("firstRunDecisionPending") }
    if onboardingOwed(done: quickstartDone, legacyDone: legacyDone,
                      lastServedAlias: lastServedAlias) {
        return .skip("onboardingPending")
    }
    guard case .idle = serverState else { return .skip("serverNotIdle") }
    guard let alias = lastServedAlias else { return .skip("noResolvableAlias") }
    if !quickstartDone && isRetiredStarter(alias) { return .skip("retiredStarter") }
    return cachedAliases.contains(alias) ? .start(alias) : .promptDownload(alias)
}

print("Auto-start vs retired starters:")
// A stranded user is one the onboarding gate now catches first — the reason
// changed from "retiredStarter" to the broader "onboardingPending", the
// property (nothing is resumed, state stays .idle) did not.
check(decideResume(lastServedAlias: "bonsai-1.7b-2bit",
                   cachedAliases: ["bonsai-1.7b-2bit"], serverState: .idle)
        == .skip("onboardingPending"),
      "retired starter on disk is NOT resumed — state stays .idle so the card can show")
check(decideResume(lastServedAlias: "qwen3.5-9b-4bit",
                   cachedAliases: ["qwen3.5-9b-4bit"], serverState: .idle)
        == .start("qwen3.5-9b-4bit"),
      "a normal model still auto-starts (the guard is not a blanket off-switch)")
check(decideResume(lastServedAlias: "lfm2.5-1b-4bit",
                   cachedAliases: ["lfm2.5-1b-4bit"], serverState: .idle)
        == .start("lfm2.5-1b-4bit"),
      "the CURRENT starter auto-starts normally")

check(decideResume(lastServedAlias: "bonsai-1.7b-2bit",
                   cachedAliases: ["bonsai-1.7b-2bit"], serverState: .idle,
                   quickstartDone: true)
        == .start("bonsai-1.7b-2bit"),
      "dismissed the rescue → auto-start comes back (no dead end: neither card nor start)")
check(!isEligible(done: true, lastServedAlias: "bonsai-1.7b-2bit", serverState: .idle),
      "…and the card stays down for them, so the two move together")

// The end-to-end property the two halves buy together.
check(decideResume(lastServedAlias: "bonsai-1.7b-2bit",
                   cachedAliases: ["bonsai-1.7b-2bit"], serverState: .idle)
        == .skip("onboardingPending")
      && isEligible(done: false, lastServedAlias: "bonsai-1.7b-2bit", serverState: .idle),
      "end-to-end: stranded user launches → no auto-start → card IS shown")

// ---------------------------------------------------------------------------
// #1589 — onboarding must win the launch race
//
// The reported defect: a never-onboarded user on a Mac with anything in the
// shared HF cache got a model auto-started, which moved serverState off .idle
// and made the wizard unreachable. The executable half of that contract.

print("Onboarding vs launch auto-start (#1589):")
check(decideResume(lastServedAlias: nil, cachedAliases: ["bonsai-27b-2bit"],
                   serverState: .idle) == .skip("onboardingPending"),
      "never-onboarded user with a cached model does NOT auto-start")
check(isEligible(done: false, lastServedAlias: nil, serverState: .idle),
      "…and because nothing started, the wizard IS eligible")
check(!isEligible(done: false, lastServedAlias: nil, serverState: .starting),
      "the pre-fix mechanism: an auto-started model would have hidden the wizard")
check(decideResume(lastServedAlias: "qwen3.5-9b-4bit",
                   cachedAliases: ["qwen3.5-9b-4bit"], serverState: .idle,
                   quickstartDone: true) == .start("qwen3.5-9b-4bit"),
      "returning user still gets their last-used model restored")
check(decideResume(lastServedAlias: "qwen3.5-9b-4bit",
                   cachedAliases: ["qwen3.5-9b-4bit"], serverState: .idle,
                   quickstartDone: true, firstRunDecisionPending: true)
        == .skip("firstRunDecisionPending"),
      "nothing loads behind the unanswered first-run consent sheet")
// The invariant, over the whole persisted state space: auto-start never
// starts a model for a user the wizard would still be offered to.
for done in [false, true] {
    for legacy in [false, true] {
        for last in [nil, "qwen3.5-9b-4bit", "bonsai-1.7b-2bit"] as [String?] {
            let decision = decideResume(
                lastServedAlias: last,
                cachedAliases: ["qwen3.5-9b-4bit", "bonsai-1.7b-2bit", "bonsai-27b-2bit"],
                serverState: .idle, quickstartDone: done, legacyDone: legacy)
            let eligible = isEligible(done: done, legacyDone: legacy,
                                      lastServedAlias: last, serverState: .idle)
            var started = false
            if case .start = decision { started = true }
            check(!(eligible && started),
                  "invariant: wizard eligible ⇒ no auto-start "
                    + "(done=\(done) legacy=\(legacy) last=\(last ?? "nil"))")
        }
    }
}

print(fails == 0 ? "\nALL PASS" : "\n\(fails) FAILURE(S)")
exit(fails == 0 ? 0 : 1)
