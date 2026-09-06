import SwiftUI

struct ComputerUseView: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
                HStack(alignment: .top) {
                    SectionHeader(
                        "Computer Use",
                        subtitle: "Let Rapid handle useful work across apps on this Mac. Everything runs locally.",
                        emphasis: .page
                    )
                    Spacer()
                    Text("EXPERIMENTAL")
                        .font(.caption2.weight(.bold))
                        .foregroundStyle(.orange)
                        .padding(.horizontal, 9)
                        .padding(.vertical, 5)
                        .background(.orange.opacity(0.1), in: Capsule())
                }

                VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
                    Text("Start with a flow").font(.headline)
                    LazyVGrid(
                        columns: [GridItem(.adaptive(minimum: 260, maximum: 320), spacing: 12)],
                        spacing: 12
                    ) {
                        ForEach(ComputerUseStarter.catalog) { starter in
                            starterCard(starter)
                        }
                    }
                    // Three cards at their maximum width plus two gaps. The
                    // adaptive grid can still collapse to two or one column.
                    .frame(maxWidth: 984, alignment: .leading)
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text("CREATE YOUR OWN")
                        .font(.caption2.weight(.bold))
                        .foregroundStyle(.secondary)
                    HStack(spacing: 14) {
                        Image(systemName: "record.circle")
                            .font(.title2)
                            .foregroundStyle(.secondary)
                        VStack(alignment: .leading, spacing: 3) {
                            Text("Teach Rapid a new task").font(.headline)
                            Text("Show Rapid how you work when no starter fits. Planned for the next Computer Use preview.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button("Coming next") {}
                            .buttonStyle(.rapidSecondaryCompact)
                            .disabled(true)
                            .accessibilityIdentifier("ComputerUse.Teach.ComingNext")
                    }
                    .padding(16)
                    .background(RapidTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 14))
                    .overlay(RoundedRectangle(cornerRadius: 14).stroke(.secondary.opacity(0.2)))
                }
            }
            .padding(RapidTheme.Space.xl)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(RapidTheme.surfaceCanvas)
        .accessibilityIdentifier("ComputerUse.Panel")
    }

    private func starterCard(_ starter: ComputerUseStarter) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: starter.systemImage)
                    .font(.title2)
                    .foregroundStyle(RapidTheme.brandPrimaryDeep)
                Spacer()
                Text(starter.availability == .reserved ? "RESERVED" : "COMING NEXT")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(.secondary)
            }
            Text(starter.title).font(.headline)
            Text(starter.summary)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(starter.applications)
                .font(.caption2.weight(.medium))
                .foregroundStyle(.secondary)
            Label(starter.approvalNote, systemImage: "checkmark.shield")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, minHeight: 118, alignment: .topLeading)
        .padding(16)
        .background(RapidTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 14))
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .stroke(
                    .secondary.opacity(starter.availability == .reserved ? 0.28 : 0.16),
                    style: StrokeStyle(
                        lineWidth: 1,
                        dash: starter.availability == .reserved ? [5] : []
                    )
                )
        )
        .accessibilityIdentifier("ComputerUse.Starter.\(starter.kind.rawValue)")
    }
}
