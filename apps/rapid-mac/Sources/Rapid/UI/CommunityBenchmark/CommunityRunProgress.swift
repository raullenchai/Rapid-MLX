import Foundation

/// The live state of a running benchmark, reduced from the CLI's RS-tagged
/// progress stream.
///
/// The Running screen used to derive everything from two scalars — a step
/// counter and an elapsed clock — so a real run showed a bar and a timer and
/// nothing else. Every field here is instead a named fact with one source:
/// which stage the run is in, how many passes have actually completed, and
/// what the newest measured value was. Nothing is interpolated from a timer,
/// and nothing is shown before an event establishes it.
struct CommunityRunProgress: Equatable, Sendable {
    /// The newest real measurement, with the pass it came from.
    struct Measurement: Equatable, Sendable {
        /// Pre-formatted by the producer (`46.1 tok/s`), never recomputed here.
        let value: String
        /// 1-based, counting warmups — the same number `passesComplete` uses,
        /// so "46.1 tok/s on pass 8" and "8 of 12 passes" agree.
        let passNumber: Int
    }

    var stage: CommunityRunStage = .gettingReady
    /// Passes the CLI has reported *finishing*. Never incremented by a timer,
    /// an estimate, or a line that merely announces intent.
    var passesComplete: Int = 0
    /// Total passes the protocol declares, or nil for a shape with no useful
    /// denominator (a single-render video run).
    var totalPasses: Int?
    /// The newest progress line, for the status row.
    var statusLine: String?
    var latestMeasurement: Measurement?
    /// `~2:40 left`, or nil while there is not enough evidence to say.
    var timeLeft: String?
    /// True once the archive write has started, so "Saving to this Mac" is a
    /// report rather than a guess about what happens next.
    var isSaving: Bool = false

    /// Determinate progress, or nil while the bar must stay indeterminate.
    /// Requires a real completed pass: a bar that starts filling before any
    /// work finishes is describing the clock, not the run.
    var fraction: Double? {
        guard let totalPasses, totalPasses > 0, passesComplete > 0 else { return nil }
        return min(1, Double(passesComplete) / Double(totalPasses))
    }

    /// `8 of 12 passes complete`, or nil when there is no denominator.
    var passCaption: String? {
        guard let totalPasses, passesComplete > 0 else { return nil }
        return String(
            format: String(localized: "%1$d of %2$d passes complete"),
            min(passesComplete, totalPasses),
            totalPasses
        )
    }
}

/// The named stages of a benchmark, in the order a run passes through them.
///
/// Ordered so the stepper can render "done / active / upcoming" without a
/// second source of truth, and `Comparable` so the reducer can keep it
/// monotonic — a second case's warmup must not walk the stepper backwards.
enum CommunityRunStage: Int, Comparable, Sendable, CaseIterable {
    case gettingReady
    case warmingUp
    /// A text protocol with a single case: there is no short/long split to
    /// narrate, and naming one would describe a protocol this run is not using.
    case measuring
    case shortReplies
    case longReplies
    /// Image and video generation, which has no short/long split.
    case rendering
    case saving

    static func < (lhs: Self, rhs: Self) -> Bool { lhs.rawValue < rhs.rawValue }

    var title: String {
        switch self {
        case .gettingReady: return String(localized: "Getting ready")
        case .warmingUp: return String(localized: "Warming up")
        case .measuring: return String(localized: "Measuring")
        case .shortReplies: return String(localized: "Short replies")
        case .longReplies: return String(localized: "Long replies")
        case .rendering: return String(localized: "Rendering")
        case .saving: return String(localized: "Saving to this Mac")
        }
    }

    /// A present-tense headline for the active stage.
    var activeTitle: String {
        switch self {
        case .gettingReady: return String(localized: "Getting ready")
        case .warmingUp: return String(localized: "Warming up")
        case .measuring: return String(localized: "Measuring")
        case .shortReplies: return String(localized: "Measuring short replies")
        case .longReplies: return String(localized: "Measuring long replies")
        case .rendering: return String(localized: "Rendering")
        case .saving: return String(localized: "Saving to this Mac")
        }
    }
}

/// The registered protocol a run is executing, as the protocol declares it.
///
/// This used to be `case text(caseCount: 2, warmupRounds: 1, measuredRounds: 5)`
/// — the shape of `rapid-community-speed` v2, written into the client. A v3
/// with different rounds, or any protocol with a different case list, would
/// have kept showing "of 12" while counting to something else. The counts
/// belong to the protocol, so they travel with it: the runner emits a `plan`
/// event before the first pass and the reducer adopts it.
struct CommunityRunPlan: Equatable, Sendable {
    /// One declared workload case and its round budget.
    struct Case: Equatable, Sendable {
        let id: String
        let warmupRounds: Int
        let measuredRounds: Int

        var passes: Int { max(0, warmupRounds) + max(0, measuredRounds) }
    }

    let workload: CommunityWorkload
    let cases: [Case]

    init(workload: CommunityWorkload, cases: [Case]) {
        self.workload = workload
        self.cases = cases
    }

    /// The shape to assume before the plan event arrives — and the only shape
    /// an older CLI that never sends one will ever report. Derived from the
    /// registered protocol for the task, and superseded the moment the run
    /// declares its own.
    static func assumed(for task: ModelTask) -> Self {
        switch task {
        case .imageGeneration:
            return Self(
                workload: .image,
                cases: [Case(id: "t2i-1024-square", warmupRounds: 1, measuredRounds: 1)]
            )
        case .videoGeneration:
            return Self(
                workload: .video,
                cases: [Case(id: "t2v-480p-81f", warmupRounds: 0, measuredRounds: 1)]
            )
        default:
            // `rapid-community-speed` v1 and v2 as registered today.
            return Self(
                workload: .llm,
                cases: [
                    Case(id: "pp512-tg128", warmupRounds: 1, measuredRounds: 5),
                    Case(id: "pp2048-tg512", warmupRounds: 1, measuredRounds: 5),
                ]
            )
        }
    }

    /// Total warmup + measured passes, or nil where a denominator would be
    /// meaningless. A single-render video run has one pass: a bar that jumps
    /// 0 → 100% says less than an honest spinner.
    var totalPasses: Int? {
        let total = cases.reduce(0) { $0 + $1.passes }
        return total > 1 ? total : nil
    }

    /// The stage sequence this run will actually pass through.
    ///
    /// Not `allCases`, and not a constant: an image run has no short/long
    /// reply split, and a single-case text protocol has no "long replies" to
    /// promise. Showing a stage the run will never enter is a promise it
    /// cannot keep.
    var stages: [CommunityRunStage] {
        let hasWarmup = cases.contains { $0.warmupRounds > 0 }
        var stages: [CommunityRunStage] = [.gettingReady]
        if hasWarmup { stages.append(.warmingUp) }
        switch workload {
        case .llm:
            stages.append(contentsOf: cases.count >= 2 ? [.shortReplies, .longReplies] : [.measuring])
        case .image, .video:
            stages.append(.rendering)
        }
        stages.append(.saving)
        return stages
    }

    /// The stage a pass in `caseIndex` belongs to.
    func stage(forCaseIndex caseIndex: Int, isWarmup: Bool) -> CommunityRunStage {
        switch workload {
        case .llm:
            guard cases.count >= 2 else { return isWarmup ? .warmingUp : .measuring }
            // The second case's *warmup* already belongs to the long-reply
            // stage: mapping every warmup to "Warming up" would walk the
            // stepper backwards halfway through the run.
            if caseIndex >= 1 { return .longReplies }
            return isWarmup ? .warmingUp : .shortReplies
        case .image, .video:
            return isWarmup ? .warmingUp : .rendering
        }
    }
}

/// The `plan` event the runner emits before the first pass.
///
/// `{"event":"plan","task_type":…,"cases":[{"case_id":…,"warmup_rounds":…,
/// "measured_rounds":…}],"total_passes":…}` — see `_announce_plan` in
/// `rapid_mlx/community_bench/local_runner.py`.
struct CommunityRunPlanEvent: Decodable, Sendable {
    struct Case: Decodable, Sendable {
        let caseID: String?
        let warmupRounds: Int?
        let measuredRounds: Int?
        enum CodingKeys: String, CodingKey {
            case caseID = "case_id"
            case warmupRounds = "warmup_rounds"
            case measuredRounds = "measured_rounds"
        }
    }
    let event: String
    let taskType: String?
    let cases: [Case]?

    enum CodingKeys: String, CodingKey {
        case event, cases
        case taskType = "task_type"
    }

    /// Nil unless this really is a plan event naming at least one case.
    var plan: CommunityRunPlan? {
        guard event == "plan", let cases, !cases.isEmpty,
              let workload = CommunityWorkload(taskType: taskType ?? "")
        else { return nil }
        let declared = cases.compactMap { declared -> CommunityRunPlan.Case? in
            guard let id = declared.caseID else { return nil }
            return CommunityRunPlan.Case(
                id: id,
                warmupRounds: declared.warmupRounds ?? 0,
                measuredRounds: declared.measuredRounds ?? 0
            )
        }
        guard !declared.isEmpty else { return nil }
        return CommunityRunPlan(
            workload: workload,
            cases: declared
        )
    }
}

/// Reduces RS-tagged CLI progress lines into ``CommunityRunProgress``.
///
/// Lives off the main actor: the pipe reader hands lines over from a detached
/// task, so the reducer is lock-protected and returns a value snapshot for the
/// caller to hop to the main actor with. Keeping the state here rather than in
/// `@State` is what makes the whole path testable against a real child
/// process instead of only against the parser's return values.
final class CommunityRunProgressBox: @unchecked Sendable {
    private let lock = NSLock()
    /// The protocol's shape. Starts as the assumed one so a run is never
    /// unreadable, and is replaced by the run's own `plan` event — which
    /// arrives before the first pass — the moment it lands.
    private var plan: CommunityRunPlan
    private var hasDeclaredPlan = false
    private var state: CommunityRunProgress
    /// Case ids in first-seen order, so "the second case" is identified by the
    /// stream rather than by a hardcoded id the protocol may rename.
    private var caseOrder: [String] = []
    private var firstPassAt: Date?
    private var lastPassAt: Date?

    init(plan: CommunityRunPlan) {
        self.plan = plan
        state = CommunityRunProgress(totalPasses: plan.totalPasses)
    }

    var snapshot: CommunityRunProgress {
        lock.lock()
        defer { lock.unlock() }
        return state
    }

    /// The plan currently in force, for the stepper.
    var currentPlan: CommunityRunPlan {
        lock.lock()
        defer { lock.unlock() }
        return plan
    }

    /// Applies one raw stderr line. Returns the new state, or nil when the
    /// line is not tagged progress — a traceback, a warning, or the untagged
    /// failure document must never reach the screen or move the bar.
    @discardableResult
    func apply(line: String, at now: Date) -> CommunityRunProgress? {
        // Structured events first: they are longer than a status line and are
        // never displayed, so they do not go through the display bound.
        if let body = CommunityBenchmarkRunStatus.strippedEvent(from: line) {
            guard let event = CommunityRunEvent(body) else { return nil }
            lock.lock()
            defer { lock.unlock() }
            apply(event: event, at: now)
            return state
        }
        guard let stripped = CommunityBenchmarkRunStatus.strippedProgress(from: line) else {
            return nil
        }
        lock.lock()
        defer { lock.unlock() }

        state.statusLine = stripped

        if let event = CommunityRunEvent(stripped) {
            apply(event: event, at: now)
        }
        return state
    }

    /// Folds one parsed event into the state. Called under the lock.
    private func apply(event: CommunityRunEvent, at now: Date) {
        switch event {
        case let .planDeclared(declared):
            // Adopted once. A second plan event mid-run would mean the
            // denominator changed under a bar the user is watching.
            if !hasDeclaredPlan {
                hasDeclaredPlan = true
                plan = declared
                state.totalPasses = declared.totalPasses
            }
        case .saving:
            state.isSaving = true
            state.stage = max(state.stage, .saving)
            // The archive write is the end of measurement: stop counting
            // down toward a pass that will never arrive.
            state.timeLeft = nil
        case let .passCompleted(caseID, phase, measurement):
            register(caseID: caseID)
            state.passesComplete += 1
            state.stage = max(state.stage, stage(for: caseID, phase: phase))
            if let measurement {
                state.latestMeasurement = CommunityRunProgress.Measurement(
                    value: measurement, passNumber: state.passesComplete
                )
            }
            if firstPassAt == nil { firstPassAt = now }
            lastPassAt = now
            state.timeLeft = estimatedTimeLeft(now: now)
        }
    }

    // MARK: - Derivation

    private func register(caseID: String) {
        guard !caseOrder.contains(caseID) else { return }
        caseOrder.append(caseID)
    }

    /// The stage a pass belongs to.
    ///
    /// The case *index* comes from the stream's own ordering rather than from
    /// matching ids against the plan: a run reports its cases in the order it
    /// executes them, and that ordering is what the stepper narrates.
    private func stage(
        for caseID: String, phase: CommunityRunEvent.Phase
    ) -> CommunityRunStage {
        let index = caseOrder.firstIndex(of: caseID) ?? 0
        return plan.stage(forCaseIndex: index, isWarmup: phase == .warmup)
    }

    /// Time left, from completed-pass timestamps only.
    ///
    /// The rate is the span between the first and last completion divided by
    /// the number of *intervals*, which excludes the one-off model load. Two
    /// completions are required: one interval is not a rate, and a figure
    /// invented before then is the number the user sits and watches.
    private func estimatedTimeLeft(now: Date) -> String? {
        guard let totalPasses = state.totalPasses,
              let firstPassAt, let lastPassAt,
              state.passesComplete >= 2,
              state.passesComplete < totalPasses
        else { return nil }
        let perPass = max(0, lastPassAt.timeIntervalSince(firstPassAt))
            / Double(state.passesComplete - 1)
        guard perPass > 0 else { return nil }
        let projected = perPass * Double(totalPasses - state.passesComplete)
        let remaining = projected - max(0, now.timeIntervalSince(lastPassAt))
        guard remaining > 0 else { return String(localized: "wrapping up…") }
        let seconds = Int(remaining.rounded())
        return String(format: String(localized: "~%d:%02d left"), seconds / 60, seconds % 60)
    }
}

/// One meaningful thing the CLI reported, parsed out of its human line.
///
/// The progress stream is prose (`rapid_mlx/community_bench/local_runner.py`
/// formats it for a terminal), so this is where prose becomes an event. Only
/// lines that mark real, completed work are events; plan announcements and
/// estimates are status text and nothing more.
enum CommunityRunEvent: Equatable, Sendable {
    enum Phase: Equatable, Sendable { case warmup, measured }

    /// `pp512-tg128  warmup` / `pp512-tg128  round 3/5   46.1 tok/s`
    case passCompleted(caseID: String, phase: Phase, measurement: String?)
    /// The local archive write has begun.
    case saving
    /// The run declared the protocol it is executing, before any pass.
    case planDeclared(CommunityRunPlan)

    init?(_ stripped: String) {
        // Structured events are compact JSON; prose never begins with `{`, so
        // the two are separated on the first character.
        if stripped.hasPrefix("{") {
            guard let data = stripped.data(using: .utf8),
                  let event = try? JSONDecoder().decode(
                      CommunityRunPlanEvent.self, from: data
                  ),
                  let plan = event.plan
            else { return nil }
            self = .planDeclared(plan)
            return
        }
        if Self.savingPattern.firstMatch(
            in: stripped,
            range: NSRange(stripped.startIndex..., in: stripped)
        ) != nil {
            self = .saving
            return
        }
        let tokens = stripped.split(separator: " ").map(String.init)
        guard tokens.count >= 2 else { return nil }
        let caseID = tokens[0]
        // The phase is always the SECOND token. Matching a bare "warmup" or
        // "round" substring anywhere would let "Benchmarking … 1 warmup + 5
        // measured rounds" and "Estimated time remaining … from the warmup
        // rate" each advance the bar.
        if tokens[1] == "warmup" {
            self = .passCompleted(caseID: caseID, phase: .warmup, measurement: nil)
            return
        }
        guard tokens[1] == "round", tokens.count >= 3,
              tokens[2].range(of: #"^\d+/\d+$"#, options: .regularExpression) != nil
        else { return nil }
        self = .passCompleted(
            caseID: caseID,
            phase: .measured,
            measurement: Self.measurement(in: tokens)
        )
    }

    /// `46.1 tok/s` out of the trailing tokens, or nil when the round reported
    /// no rate (image/video rounds say `generating...`).
    private static func measurement(in tokens: [String]) -> String? {
        for (index, token) in tokens.enumerated() where token == "tok/s" {
            guard index > 0,
                  Double(tokens[index - 1]) != nil
            else { return nil }
            return "\(tokens[index - 1]) \(token)"
        }
        return nil
    }

    /// Matches the archive-write line the CLI emits around persistence.
    private static let savingPattern = try! NSRegularExpression(
        pattern: #"^Sav(ing|ed)\b.*\bthis Mac\b"#, options: [.caseInsensitive]
    )
}
