import SwiftUI

/// The three top-level surfaces of the module.
enum CommunityBenchmarkTab: String, CaseIterable, Identifiable, Hashable {
    case run
    case myResults
    case community

    var id: String { rawValue }

    var title: String {
        switch self {
        case .run: return String(localized: "Run")
        case .myResults: return String(localized: "My Results")
        case .community: return String(localized: "Community")
        }
    }
}

/// Where the brand cheetah is allowed to appear.
///
/// The mascot specification is a *restriction*, not a decoration budget: the
/// character may celebrate contribution and accompany work in progress, and it
/// must stay away from every surface where a person is deciding what to make
/// public or reading about something that went wrong. Encoding the allowed
/// contexts as a type means a future call site has to name its context, and a
/// reviewer can grep for the ones that exist.
enum CommunityMascotContext {
    /// The Ready invitation — upper-right of the contribution copy.
    case readyInvitation
    /// Beside the progress track while a benchmark runs.
    case running
    /// The publication celebration.
    case published

    var pointSize: CGFloat {
        switch self {
        case .readyInvitation: return 68
        case .running: return 40
        case .published: return 80
        }
    }

    /// The website plate this context uses. Ready and Published invite and
    /// celebrate, so they wave; Running is work in progress, so it runs.
    var art: CommunityMascotArt {
        switch self {
        case .readyInvitation, .published: return .inviteWave
        case .running: return .run
        }
    }

    /// The frame to draw in so that ``pointSize`` describes the *visible*
    /// character rather than the image box.
    ///
    /// Both plates are 320 × 320 canvases with roughly half of that being
    /// transparent margin (invite-wave 149 × 158, run 165 × 142). Fitting the
    /// canvas to a 68pt box therefore renders a ~34pt cheetah, which is why
    /// the character looked shrunken next to a 26pt headline. Dividing by the
    /// measured visible ratio restores the intended optical size without
    /// touching the vendored artwork.
    var renderedBoxSize: CGFloat { (pointSize / art.visibleRatio).rounded() }
}

/// The brand cheetah, restricted to the approved contexts.
///
/// Deliberately has no initialiser that omits ``context`` — a mascot without a
/// declared context cannot be placed on a privacy, warning, failure,
/// cancellation, or technical-payload surface by accident.
struct CommunityMascot: View {
    let context: CommunityMascotContext
    var isAnimating = false

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        // The frame is the RENDERED box, not the visible character: the plate
        // is ~50% transparent margin, so reserving only the character's size
        // let the artwork spill out and collide with the headline beneath it
        // on the Published sheet. The margin is just whitespace in the layout.
        mascotImage
            .opacity(context == .running && !isAnimating ? 0.75 : 1)
            .animation(
                reduceMotion || !isAnimating
                    ? nil
                    : .easeInOut(duration: 0.9).repeatForever(autoreverses: true),
                value: isAnimating
            )
            .accessibilityHidden(true)
    }

    @ViewBuilder
    private var mascotImage: some View {
        if let image = context.art.image {
            Image(nsImage: image)
                .resizable()
                .interpolation(.high)
                .scaledToFit()
                .frame(width: context.renderedBoxSize, height: context.renderedBoxSize)
        } else {
            // The asset is bundled at build time, so this is "the .app got
            // corrupted" territory; the existing brand mark keeps the
            // composition intact rather than leaving a hole.
            CheetahLogo(size: context.pointSize)
        }
    }
}

extension CommunityMascotContext: Equatable {}

/// A contributor's canonical portrait.
///
/// The plate comes from ``CommunityContributorAvatar``, which is a byte-for-byte
/// port of the website's mapping — so the same installation shows the same face
/// in My Results, in the publish dialog, on the success sheet, in the Community
/// pulse, and on its rapidmlx.com profile page.
///
/// Decorative by construction: the pseudonym is always rendered as live text
/// beside it, so labelling the artwork would make VoiceOver announce the same
/// installation twice. This matches `avatarHTML` on the website.
struct CommunityContributorPortrait: View {
    /// The identity key — a slug. Passing the key rather than the contributor
    /// lets the pulse render portraits for other people's installations.
    let key: String
    var size: CGFloat = 34
    var cornerRadius: CGFloat = 8

    init(key: String, size: CGFloat = 34, cornerRadius: CGFloat = 8) {
        self.key = key
        self.size = size
        self.cornerRadius = cornerRadius
    }

    init(contributor: CommunityBenchmarkContributor, size: CGFloat = 34, cornerRadius: CGFloat = 8) {
        self.init(key: contributor.slug, size: size, cornerRadius: cornerRadius)
    }

    var body: some View {
        Group {
            if let image = CommunityContributorAvatar.image(forKey: key) {
                Image(nsImage: image)
                    .resizable()
                    .interpolation(.high)
                    .scaledToFill()
            } else {
                RapidTheme.surfaceCanvas
            }
        }
        .frame(width: size, height: size)
        .clipShape(RoundedRectangle(cornerRadius: cornerRadius))
        .accessibilityHidden(true)
    }
}

// MARK: - Small shared parts

/// The LLM / IMAGE / VIDEO chip beside a model alias.
struct CommunityWorkloadBadge: View {
    let workload: CommunityWorkload

    var body: some View {
        Text(workload.displayName.uppercased())
            .font(.system(size: 10, weight: .semibold))
            .tracking(0.3)
            .foregroundStyle(RapidTheme.brandSecondary)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(RapidTheme.brandSecondaryTint, in: RoundedRectangle(cornerRadius: 4))
            .accessibilityLabel(
                String(
                    format: String(localized: "%1$@ workload"),
                    workload.displayName
                )
            )
    }
}

/// A single fact in the Ready facts row: icon, label, optional monospaced
/// detail. Fixed-width icon slot so the lane labels align.
struct CommunityFactLane: View {
    let systemImage: String
    let label: String
    var detail: String? = nil
    var tone: Color = RapidTheme.textSecondary

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: systemImage)
                .font(.system(size: 12))
                .frame(width: 14)
                .foregroundStyle(tone)
            Text(label)
                .font(RapidFont.body)
                .foregroundStyle(tone)
            if let detail {
                Text(detail)
                    .font(RapidFont.metric)
                    .foregroundStyle(RapidTheme.textTertiary)
            }
        }
        .fixedSize()
        .accessibilityElement(children: .combine)
    }
}

/// A headline metric: big value, small unit, label underneath.
struct CommunityMetricView: View {
    let metric: CommunityBenchmarkMetrics.Metric
    var isHeadline = false

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(alignment: .firstTextBaseline, spacing: 4) {
                Text(metric.value)
                    .font(
                        isHeadline
                            ? .system(size: 44, weight: .semibold, design: .monospaced)
                            : .system(size: 20, weight: .medium, design: .monospaced)
                    )
                    .monospacedDigit()
                if let unit = metric.unit {
                    Text(unit)
                        .font(
                            isHeadline
                                ? .system(size: 18, weight: .regular, design: .monospaced)
                                : .system(size: 13, design: .monospaced)
                        )
                        .foregroundStyle(RapidTheme.textSecondary)
                }
            }
            Text(metric.label)
                .font(RapidFont.secondary)
                .foregroundStyle(RapidTheme.textSecondary)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(metric.label): \(metric.combined)")
    }
}

/// The band that explains why a community claim is missing. Used wherever a
/// median, range, coverage count, or pulse statistic would otherwise sit.
struct CommunityUnavailableBand: View {
    let title: String
    let message: String
    var isLoading = false

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            if isLoading {
                ProgressView()
                    .controlSize(.small)
                    .accessibilityHidden(true)
            } else {
                Image(systemName: "chart.bar.doc.horizontal")
                    .font(.system(size: 13))
                    .foregroundStyle(RapidTheme.textTertiary)
                    .accessibilityHidden(true)
            }
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(RapidFont.bodyEmphasis)
                    .foregroundStyle(RapidTheme.textPrimary)
                Text(message)
                    .font(RapidFont.secondary)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .padding(RapidTheme.Space.md)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RapidTheme.surfaceCanvas, in: RoundedRectangle(cornerRadius: RapidTheme.Radius.card))
        .accessibilityElement(children: .combine)
    }
}

/// The anonymous contributor identity returned by a publish receipt.
///
/// The portrait is derived from the receipt's slug using the website's own
/// mapping, so this row shows exactly the face rapidmlx.com shows for the same
/// installation. Nothing about the person is collected or inferred — the slug
/// is the only input.
struct CommunityContributorIdentity: View {
    let contributor: CommunityBenchmarkContributor
    var showsDestination = true

    var body: some View {
        HStack(spacing: RapidTheme.Space.md) {
            CommunityContributorPortrait(contributor: contributor)
            VStack(alignment: .leading, spacing: 2) {
                Text(contributor.displayName)
                    .font(.system(size: 13, weight: .medium, design: .monospaced))
                    .foregroundStyle(RapidTheme.textPrimary)
                    .textSelection(.enabled)
                if showsDestination, let url = contributor.profileURL {
                    Text(url.absoluteString.replacingOccurrences(of: "https://", with: ""))
                        .font(RapidFont.code)
                        .foregroundStyle(RapidTheme.textTertiary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(RapidTheme.Space.md)
        .background(RapidTheme.surfaceCanvas, in: RoundedRectangle(cornerRadius: RapidTheme.Radius.card))
        .accessibilityElement(children: .combine)
        .accessibilityLabel(
            String(
                format: String(localized: "Anonymous contributor %1$@"),
                contributor.displayName
            )
        )
    }
}

/// The compact secondary row that links out to the verified ranked
/// leaderboard. Visually quieter than the observations table and the
/// contribution panel, and always last in the reading order.
struct CommunityLeaderboardLinkRow: View {
    let destination: URL

    var body: some View {
        Link(destination: destination) {
            HStack(spacing: RapidTheme.Space.sm) {
                Image(systemName: "trophy")
                    .font(.system(size: 12))
                    .foregroundStyle(RapidTheme.textTertiary)
                    .accessibilityHidden(true)
                Text("View the verified global leaderboard")
                    .font(RapidFont.bodyEmphasis)
                    .foregroundStyle(RapidTheme.brandSecondary)
                Text("Verified CLI runs only — the ranked dataset lives on rapidmlx.com")
                    .font(RapidFont.secondary)
                    .foregroundStyle(RapidTheme.textTertiary)
                    .lineLimit(1)
                    .truncationMode(.tail)
                Spacer(minLength: RapidTheme.Space.sm)
                Image(systemName: "arrow.up.right.square")
                    .font(.system(size: 12))
                    .foregroundStyle(RapidTheme.textTertiary)
                    .accessibilityHidden(true)
            }
            .padding(.vertical, RapidTheme.Space.md)
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("CommunityBenchmark.Community.Leaderboard")
        .overlay(alignment: .top) {
            Rectangle()
                .fill(RapidTheme.hairline)
                .frame(height: 1)
        }
        .accessibilityLabel(String(localized: "View the verified global leaderboard on rapidmlx.com"))
    }
}
