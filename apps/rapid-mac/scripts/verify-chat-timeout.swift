#!/usr/bin/env swift
//
// verify-chat-timeout.swift — runnable contract check for the chat
// stream client's INACTIVITY-timeout policy.
//
// The Tests/RapidTests XCTest/Swift-Testing suite is excluded from the
// SwiftPM manifest (see Package.swift) and `import Testing` isn't
// resolvable from the command line in this toolchain, so this
// dependency-free script is the CI-runnable verification of the
// contract: run `swift apps/rapid-mac/scripts/verify-chat-timeout.swift`.
//
// Why this exists (regression guard):
//   A local rapid-mlx server that connects but goes silent used to
//   strand the chat send for 600 s (a remote-API-shaped inactivity
//   timeout). On a mid-session model switch that surfaced as a dead,
//   error-less spinner for minutes with idle CPU — the user force-quit
//   long before the timeout fired. The fix bounds the per-request
//   INACTIVITY window (max wait for the next byte, reset on each byte)
//   to a local-appropriate value so a wedged send fails as an
//   actionable, retryable row instead of hanging.
//
// Both halves read the REAL source (not re-declared copies), so a
// production-only regression fails here:
//   * Part 1 pins the bounded timeout VALUES + their wiring in
//     ChatStreamClient.swift (and rejects an over-large, too-small,
//     mismatched, or commented-out setting — mutation-verified).
//   * Part 2 pins the UX contract in FailureDiagnosis.swift — that a
//     timed-out request stays a RETRYABLE failure, never a non-retryable
//     dead end.
//
// Scope/limitation (accepted): this is a SOURCE-level contract, not an
// executed one — it cannot import ``ChatStreamClient`` because the SPM
// test target is stripped (Package.swift), so it cannot prove at runtime
// that a silent endpoint yields ``NSURLErrorTimedOut``. That end-to-end
// coverage is part of the deferred v1.0 test-suite work, exactly as the
// sibling ``verify-recommendation-tiers.swift`` accepts for the same
// reason. The bounded-timeout MECHANISM (URLSession streaming
// ``bytes(for:)`` firing ``URLError.timedOut`` at the configured bound)
// was verified behaviorally against a real stalled socket during
// development; this guard keeps the VALUES + WIRING that feed it honest.

import Foundation

// The local-loopback inactivity band [min, max].
//   * Upper (180 s): the only legitimate silent window is first-token
//     prefill; 180 s catches a wedged server in ~3 min instead of 10.
//   * Lower (120 s): the window must comfortably clear a cold first-token
//     prefill. Documented worst case is ~80 s (a 27B on a 4K prompt), so
//     the floor sits above it with headroom — a regression to a value
//     that could clip that prefill (e.g. 60 s), or to a tiny 0.001/0,
//     must FAIL here, not just an over-large one.
let maxInactivityBudget: Double = 180
let minInactivityBudget: Double = 120

var failures: [String] = []
func check(_ cond: Bool, _ msg: String) {
    if !cond { failures.append(msg) }
}

let scriptURL = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
let sourcesRoot = scriptURL
    .deletingLastPathComponent()          // scripts/
    .deletingLastPathComponent()          // rapid-mac/
    .appendingPathComponent("Sources/Rapid")

/// Remove `/* ... */` block comments (non-nesting; sufficient for the
/// contract sites here) so a check can't be defeated by block-commenting
/// out a real statement. A char scan rather than a regex so a multi-line
/// block is handled without dotall flags.
func stripBlockComments(_ s: String) -> String {
    var out = ""
    out.reserveCapacity(s.count)
    var i = s.startIndex
    while i < s.endIndex {
        if s[i] == "/", s.index(after: i) < s.endIndex, s[s.index(after: i)] == "*" {
            // Skip to the closing */ (or end of file).
            var j = s.index(i, offsetBy: 2, limitedBy: s.endIndex) ?? s.endIndex
            while j < s.endIndex {
                if s[j] == "*", s.index(after: j) < s.endIndex, s[s.index(after: j)] == "/" {
                    j = s.index(j, offsetBy: 2)
                    break
                }
                j = s.index(after: j)
            }
            out.append(" ")   // keep tokens on either side from fusing
            i = j
        } else {
            out.append(s[i])
            i = s.index(after: i)
        }
    }
    return out
}

/// Read a source file with comments removed, so a check can't be
/// satisfied (or defeated) by text that is commented out — e.g.
/// commenting out the timeout-wiring line (with `//` OR `/* */`) must NOT
/// keep the guard green. Block comments are stripped first, then any
/// whole-line `//` comment; the contract sites this script pins are all
/// standalone statements, never trailing comments.
func readSource(_ relative: String) -> String? {
    guard let raw = try? String(contentsOf: sourcesRoot.appendingPathComponent(relative), encoding: .utf8)
    else { return nil }
    return stripBlockComments(raw)
        .split(separator: "\n", omittingEmptySubsequences: false)
        .filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("//") }
        .joined(separator: "\n")
}

// MARK: - Part 1: ChatStreamClient timeout VALUES (real source)

guard let client = readSource("Chat/ChatStreamClient.swift") else {
    print("FAIL — cannot read ChatStreamClient.swift under \(sourcesRoot.path)")
    exit(1)
}

/// Pull the numeric literal assigned in the FIRST line matching `needle`.
/// Returns nil when the marker is absent (also a failure — the contract
/// site moved and this guard would silently pass otherwise).
func firstAssignedNumber(in source: String, after needle: String) -> Double? {
    for rawLine in source.split(separator: "\n") {
        let line = String(rawLine)
        guard line.contains(needle) else { continue }
        guard let eq = line.range(of: "=") else { continue }
        var digits = ""
        for ch in line[eq.upperBound...].drop(while: { $0 == " " }) {
            if ch.isNumber || ch == "." { digits.append(ch) } else { break }
        }
        return Double(digits)
    }
    return nil
}

func checkInBand(_ value: Double?, _ label: String) -> Double? {
    guard let value else {
        failures.append("Could not find `\(label)` assignment in ChatStreamClient.swift.")
        return nil
    }
    check(value >= minInactivityBudget && value <= maxInactivityBudget,
          "\(label) is \(value)s — must be within [\(minInactivityBudget), \(maxInactivityBudget)]s: "
            + "large enough to clear a cold first-token prefill, small enough that a silent "
            + "server fails fast instead of hanging like a remote API.")
    return value
}

let reqTimeout = checkInBand(
    firstAssignedNumber(in: client, after: "var requestTimeout: TimeInterval"),
    "requestTimeout default")
let sessionInactivity = checkInBand(
    firstAssignedNumber(in: client, after: "config.timeoutIntervalForRequest"),
    "sharedSession timeoutIntervalForRequest")

// The per-request override and the shared-session default must agree, or
// a caller reusing the session without a per-request override silently
// gets a different (unbounded-relative) policy than a normal send.
if let reqTimeout, let sessionInactivity {
    check(reqTimeout == sessionInactivity,
          "requestTimeout (\(reqTimeout)s) and sharedSession timeoutIntervalForRequest "
            + "(\(sessionInactivity)s) must match so every send gets the same inactivity bound.")
}

// The TOTAL-response cap is a separate, larger backstop; assert it exists
// and is >= the CONFIGURED inactivity bound (not a fixed literal), so a
// valid future retune within the band (e.g. 120s inactivity / 300s total)
// still passes. A total cap tighter than a single inter-byte gap makes no
// sense.
if let sessionResource = firstAssignedNumber(in: client, after: "config.timeoutIntervalForResource") {
    let inactivityBound = reqTimeout ?? maxInactivityBudget
    check(sessionResource >= inactivityBound,
          "sharedSession timeoutIntervalForResource is \(sessionResource)s — must be >= the "
            + "\(inactivityBound)s inactivity bound (total cap can't be tighter than a gap).")
} else {
    failures.append("Could not find `config.timeoutIntervalForResource` in ChatStreamClient.swift.")
}

// A bounded VALUE is inert unless it is actually applied to the request.
// Pin the wiring so this stays a behavioral guard: if the send path stops
// assigning requestTimeout to the URLRequest, the number above no longer
// governs anything and this must fail.
check(client.contains("req.timeoutInterval = requestTimeout"),
      "ChatStreamClient.send no longer wires `req.timeoutInterval = requestTimeout` — the bounded "
        + "value is not applied to the request, so the inactivity guard is inert.")

// MARK: - Part 2: FailureDiagnosis retryability contract (real source)

// The bounded timeout above is only useful if a timed-out chat request
// surfaces as a RETRYABLE failure. That classification lives in the real
// FailureDiagnoser; pin it against the actual source so a regression
// there (e.g. re-bucketing a timeout as a non-retryable dead end) fails
// this guard.
guard let diag = readSource("Services/FailureDiagnosis.swift") else {
    print("FAIL — cannot read FailureDiagnosis.swift under \(sourcesRoot.path)")
    exit(1)
}

// Collapse ALL whitespace runs to single spaces so multi-line switch
// arms and single-space stop tokens match regardless of indentation/wrap.
let diagFlat = diag.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")

/// The substring of `s` starting after `start` and ending at the FIRST
/// of `stops` (or end of string). Used to isolate ONE switch arm / block
/// so an assertion can't accidentally match text belonging to a later
/// arm (codex r4 finding on the prefix-window search).
func region(_ s: String, after start: String, until stops: [String]) -> Substring? {
    guard let r = s.range(of: start) else { return nil }
    let rest = s[r.upperBound...]
    let end = stops.compactMap { rest.range(of: $0)?.lowerBound }.min() ?? rest.endIndex
    return rest[..<end]
}

// Isolate the NSURLErrorDomain `switch` body: from the switch to its
// closing — everything a timeout's classification depends on lives here.
// The switch arms open no braces, so the first `}` after the switch
// header is the switch's own close — bound the region there.
if let domainSwitch = region(diagFlat, after: "switch ns.code {", until: ["}"]),
   let defaultRange = domainSwitch.range(of: "default:") {
    let explicitArm = domainSwitch[..<defaultRange.lowerBound]   // non-default cases
    let defaultArm = domainSwitch[defaultRange.upperBound...]    // the fall-through
    // (a) A timeout must NOT sit in the explicit (non-retryable
    //     `.engineNotRunning`) arm — it must fall through `default`.
    //     Case-insensitive substring so BOTH spellings are caught: the
    //     `NSURLErrorTimedOut` Int constant this switch uses today, and a
    //     `.timedOut` (URLError.Code) form a refactor might introduce.
    check(!explicitArm.lowercased().contains("timedout"),
          "FailureDiagnosis classifies a timeout (NSURLErrorTimedOut / .timedOut) in the "
            + "non-retryable NSURLError arm — a timed-out chat request must fall through to "
            + ".requestFailed (retryable), not a dead end.")
    // (b) …and that `default` must actually return `.requestFailed`, or
    //     the fall-through lands on some other (possibly non-retryable)
    //     kind. Asserting absence-before-default alone would miss this.
    check(defaultArm.contains("return .requestFailed"),
          "FailureDiagnosis's NSURLErrorDomain default no longer returns .requestFailed — a "
            + "timed-out chat request may fall through to a non-retryable diagnosis.")
} else {
    failures.append("Could not locate the NSURLErrorDomain switch in FailureDiagnosis.swift.")
}

// (c) The `.requestFailed` kind (what a timeout lands on) must map to a
//     Retry action. Bound the search to `.requestFailed`'s OWN arm — up
//     to the next `case`/`default:` or the switch close — so it cannot
//     borrow a `.retry` from a neighbouring arm (codex r4 finding).
if let requestFailedArm = region(diagFlat, after: "case .requestFailed:",
                                 until: [" case ", " default:", " } "]) {
    check(requestFailedArm.contains("action = .retry"),
          "FailureDiagnosis .requestFailed no longer maps to a Retry action — a timed-out "
            + "chat request would not offer recovery.")
} else {
    failures.append("Could not find `case .requestFailed:` in FailureDiagnosis.swift.")
}

// MARK: - Report

if failures.isEmpty {
    print("OK — chat inactivity timeout is in [\(Int(minInactivityBudget)), \(Int(maxInactivityBudget))]s "
        + "and a timed-out request stays retryable.")
    exit(0)
} else {
    for f in failures { print("FAIL — \(f)") }
    exit(1)
}
