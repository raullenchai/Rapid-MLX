import SwiftUI

/// "How the test works" — the user-facing methodology sheet.
///
/// Deliberately contains no case IDs, protocol digests, CLI syntax, or JSON.
/// Those live behind "Review the exact data" on the publish confirmation,
/// where someone auditing a payload will look for them. This sheet answers a
/// different question: is this measurement fair, and what do the numbers mean?
struct CommunityBenchmarkTestMethodSheet: View {
    let modelAlias: String
    let workload: CommunityWorkload
    let onDone: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
                    VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
                        Text("How the test works")
                            .font(RapidFont.pageTitle)
                            .foregroundStyle(RapidTheme.textPrimary)
                        Text(
                            "The benchmark runs entirely on this Mac. It uses fixed prompts and settings every time, so a result from one Mac can be compared with a result from another."
                        )
                        .font(RapidFont.body)
                        .foregroundStyle(RapidTheme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                    }

                    VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
                        SectionHeader(
                            String(
                                format: String(localized: "WHAT IT RUNS ON %1$@"),
                                modelAlias
                            )
                        )
                        ForEach(Array(passes.enumerated()), id: \.offset) { index, pass in
                            passRow(index: index + 1, pass: pass)
                        }
                        Text(closingNote)
                            .font(RapidFont.secondary)
                            .foregroundStyle(RapidTheme.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
                        SectionHeader(String(localized: "WHAT YOU GET BACK"))
                        ForEach(
                            CommunityBenchmarkMetrics.measuredQuantities(for: workload),
                            id: \.title
                        ) { quantity in
                            VStack(alignment: .leading, spacing: 2) {
                                Text(quantity.title)
                                    .font(RapidFont.bodyEmphasis)
                                    .foregroundStyle(RapidTheme.textPrimary)
                                Text(quantity.detail)
                                    .font(RapidFont.secondary)
                                    .foregroundStyle(RapidTheme.textSecondary)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }

                    VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
                        note(
                            systemImage: "pause.circle",
                            text: String(
                                localized: "Chat and Images pause so the benchmark has the Mac to itself. Your previous model reloads automatically when it finishes."
                            )
                        )
                        note(
                            systemImage: "lock",
                            text: String(
                                localized: "Nothing is published automatically. The result is saved on this Mac, and publishing is always a separate, explicit step."
                            )
                        )
                    }
                }
                .padding(RapidTheme.Space.xl)
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            Divider()

            HStack {
                Spacer()
                Button(String(localized: "Done"), action: onDone)
                    .buttonStyle(.rapidPrimary)
                    .keyboardShortcut(.defaultAction)
                    .accessibilityIdentifier("CommunityBenchmark.TestMethod.Done")
            }
            .padding(.horizontal, RapidTheme.Space.xl)
            .padding(.vertical, RapidTheme.Space.lg)
        }
        .frame(width: 640, height: 640)
        .background(RapidTheme.surfaceRaised)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("CommunityBenchmark.TestMethod")
    }

    // MARK: - Workload description

    private struct Pass {
        let title: String
        let detail: String
        let rounds: String
    }

    private var passes: [Pass] {
        switch workload {
        case .llm:
            return [
                Pass(
                    title: String(localized: "A short prompt"),
                    detail: String(localized: "512 tokens in, 128 tokens of reply out"),
                    rounds: String(localized: "1 warm-up round\n5 measured rounds")
                ),
                Pass(
                    title: String(localized: "A long prompt"),
                    detail: String(localized: "2 048 tokens in, 512 tokens of reply out"),
                    rounds: String(localized: "1 warm-up round\n5 measured rounds")
                ),
            ]
        case .image:
            return [
                Pass(
                    title: String(localized: "One image"),
                    detail: String(localized: "1024 × 1024 at 20 steps, fixed prompt and seed"),
                    rounds: String(localized: "1 warm-up render\n1 measured render")
                )
            ]
        case .video:
            return [
                Pass(
                    title: String(localized: "One video"),
                    detail: String(localized: "832 × 480, 81 frames, fixed prompt and seed"),
                    rounds: String(localized: "1 measured render")
                )
            ]
        }
    }

    private var closingNote: String {
        switch workload {
        case .llm:
            return String(
                localized: "One request at a time. The warm-up round is thrown away; only the five measured rounds count."
            )
        case .image:
            return String(
                localized: "The warm-up render is thrown away; only the measured render counts."
            )
        case .video:
            return String(
                localized: "A single measured render, so there is no round-by-round progress or estimate."
            )
        }
    }

    private func passRow(index: Int, pass: Pass) -> some View {
        HStack(alignment: .top, spacing: RapidTheme.Space.md) {
            Text("\(index)")
                .font(RapidFont.metric)
                .foregroundStyle(RapidTheme.textSecondary)
                .frame(width: 22, height: 22)
                .background(RapidTheme.surfaceCanvas, in: RoundedRectangle(cornerRadius: 6))
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(pass.title)
                    .font(RapidFont.bodyEmphasis)
                    .foregroundStyle(RapidTheme.textPrimary)
                Text(pass.detail)
                    .font(RapidFont.secondary)
                    .foregroundStyle(RapidTheme.textSecondary)
            }
            Spacer(minLength: RapidTheme.Space.md)
            Text(pass.rounds)
                .font(RapidFont.secondary)
                .foregroundStyle(RapidTheme.textSecondary)
                .multilineTextAlignment(.trailing)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }

    private func note(systemImage: String, text: String) -> some View {
        HStack(alignment: .top, spacing: RapidTheme.Space.sm) {
            Image(systemName: systemImage)
                .font(.system(size: 13))
                .foregroundStyle(RapidTheme.textTertiary)
                .accessibilityHidden(true)
            Text(text)
                .font(RapidFont.secondary)
                .foregroundStyle(RapidTheme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
