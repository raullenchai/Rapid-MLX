import Foundation

/// Runs the bundled `rapid-mlx bench` for the in-app "Speed on this Mac"
/// card, and drives the community-leaderboard submission.
///
/// Two paths, both delegating to the proven CLI so the app never
/// reimplements the benchmark or the submission wire format:
///   * ``run`` → `rapid-mlx bench <alias>` (freeform), parses the
///     `Throughput: N tok/s` summary line for display.
///   * ``submit`` → `rapid-mlx bench <alias> --submit` (the
///     standardized B=1 community runner that POSTs to
///     rapidmlx.com/api/benchmarks). The CLI asks for a y/N consent on
///     stdin; the app shows its OWN consent first, then pipes "y" so the
///     user only answers once.
@MainActor
@Observable
final class BenchmarkRunner {
    enum Phase: Equatable {
        case idle
        case running
        case done(BenchmarkResult)
        case failed(String)
    }

    enum SubmitPhase: Equatable {
        case idle
        case submitting
        case submitted
        case failed(String)
    }

    private(set) var phase: Phase = .idle
    private(set) var submitPhase: SubmitPhase = .idle

    /// Public leaderboard the submission lands on.
    static let boardURL = URL(string: "https://rapidmlx.com/leaderboard")!

    /// DEV-ONLY: force a phase so the snapshot harness can render the
    /// result / submitted states without a live sidecar.
    func devSeed(phase: Phase, submit: SubmitPhase = .idle) {
        self.phase = phase
        self.submitPhase = submit
    }

    // MARK: - Benchmark (display)

    func run(binary: URL, alias: String, chip: String) async {
        guard !alias.isEmpty else {
            phase = .failed("Choose a model first.")
            return
        }
        phase = .running
        let output = await Self.runProcess(
            binary: binary, args: ["bench", alias], stdinLine: nil
        )
        switch output {
        case .failure(let msg):
            phase = .failed(msg)
        case .success(let text):
            guard let result = Self.parse(text, alias: alias, chip: chip) else {
                phase = .failed("Couldn't read the benchmark result.")
                return
            }
            // A zero/garbage throughput means the run produced no tokens
            // (e.g. the model errored mid-generation but the summary line
            // still printed "0.00 tok/s" and the process exited cleanly).
            // Never surface a confident "0 tokens / second" — and never
            // let a zero reach the community leaderboard.
            guard result.throughputTPS > 0 else {
                phase = .failed(
                    "The benchmark didn't produce a usable number — the model generated no tokens. Try again, or restart the model.")
                return
            }
            phase = .done(result)
        }
    }

    // MARK: - Submit (community leaderboard)

    func submit(binary: URL, alias: String) async {
        guard !alias.isEmpty else {
            submitPhase = .failed("Choose a model first.")
            return
        }
        submitPhase = .submitting
        // The CLI prints the exact payload and asks "submit? [y/N]" on
        // stdin. The app already showed its own consent, so answer "y".
        let output = await Self.runProcess(
            binary: binary, args: ["bench", alias, "--submit"], stdinLine: "y\n"
        )
        switch output {
        case .failure(let msg):
            submitPhase = .failed(msg)
        case .success(let text):
            // The runner prints "Your numbers are on the board at …" on
            // success. Fall back to a generic success if the wording
            // shifts but the process exited cleanly.
            if text.localizedCaseInsensitiveContains("on the board")
                || text.localizedCaseInsensitiveContains("leaderboard")
                || text.localizedCaseInsensitiveContains("submitted") {
                submitPhase = .submitted
            } else if text.localizedCaseInsensitiveContains("error")
                || text.localizedCaseInsensitiveContains("failed") {
                submitPhase = .failed(Self.firstErrorLine(text) ?? "Submission failed.")
            } else {
                submitPhase = .submitted
            }
        }
    }

    // MARK: - Parsing

    /// Pulls the throughput + tokens/second out of the freeform bench
    /// summary. Format (rapid-mlx bench):
    ///     Tokens/second: 781.55
    ///     Throughput: 836.26 tok/s
    static func parse(_ text: String, alias: String, chip: String) -> BenchmarkResult? {
        let throughput = firstDouble(in: text, pattern: #"Throughput:\s*([\d.]+)\s*tok"#)
        let tokensPerSec = firstDouble(in: text, pattern: #"Tokens/second:\s*([\d.]+)"#)
        // Prefer throughput (end-to-end); fall back to tokens/second.
        guard let primary = throughput ?? tokensPerSec else { return nil }
        return BenchmarkResult(
            alias: alias,
            chip: chip,
            throughputTPS: primary,
            tokensPerSecond: tokensPerSec ?? primary
        )
    }

    private static func firstDouble(in text: String, pattern: String) -> Double? {
        guard let re = try? NSRegularExpression(pattern: pattern) else { return nil }
        let range = NSRange(text.startIndex..., in: text)
        guard let m = re.firstMatch(in: text, range: range),
              m.numberOfRanges > 1,
              let r = Range(m.range(at: 1), in: text) else { return nil }
        return Double(text[r])
    }

    private static func firstErrorLine(_ text: String) -> String? {
        text.split(separator: "\n")
            .first(where: { $0.localizedCaseInsensitiveContains("error") })
            .map { String($0).trimmingCharacters(in: .whitespaces) }
    }

    // MARK: - Subprocess

    private enum RunOutput {
        case success(String)
        case failure(String)
    }

    /// Runs `binary args…`, optionally writing one line to stdin, and
    /// returns combined stdout+stderr text. Off the main actor.
    private nonisolated static func runProcess(
        binary: URL, args: [String], stdinLine: String?
    ) async -> RunOutput {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let task = Process()
                task.executableURL = binary
                task.arguments = args
                let out = Pipe()
                let err = Pipe()
                task.standardOutput = out
                task.standardError = err
                let inPipe = Pipe()
                if stdinLine != nil { task.standardInput = inPipe }

                do {
                    try task.run()
                } catch {
                    continuation.resume(returning: .failure(
                        "Couldn't start the benchmark: \(error.localizedDescription)"))
                    return
                }
                if let line = stdinLine {
                    inPipe.fileHandleForWriting.write(Data(line.utf8))
                    try? inPipe.fileHandleForWriting.close()
                }
                // Read fully before waitUntilExit to avoid a pipe-buffer
                // deadlock on a chatty child.
                let outData = out.fileHandleForReading.readDataToEndOfFile()
                let errData = err.fileHandleForReading.readDataToEndOfFile()
                task.waitUntilExit()
                let combined = (String(data: outData, encoding: .utf8) ?? "")
                    + "\n"
                    + (String(data: errData, encoding: .utf8) ?? "")
                if task.terminationStatus == 0 {
                    continuation.resume(returning: .success(combined))
                } else {
                    let tail = combined
                        .split(separator: "\n")
                        .suffix(4)
                        .joined(separator: " ")
                    continuation.resume(returning: .failure(
                        tail.isEmpty ? "Benchmark exited with code \(task.terminationStatus)." : tail))
                }
            }
        }
    }
}

struct BenchmarkResult: Equatable {
    let alias: String
    let chip: String
    /// End-to-end throughput in tokens/second (the headline number).
    let throughputTPS: Double
    /// Decode tokens/second.
    let tokensPerSecond: Double
}
