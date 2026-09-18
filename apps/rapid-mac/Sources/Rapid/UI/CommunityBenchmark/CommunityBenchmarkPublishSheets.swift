import AppKit
import SwiftUI

/// "Publish this result to the Community Benchmark?"
///
/// Leads with the privacy boundary — what becomes public and what never does —
/// and keeps the raw payload behind a disclosure. The previous dialog opened
/// with a wall of JSON, which is exactly backwards: most people are deciding
/// whether to share, not auditing a wire format.
///
/// Deliberately mascot-free. This is a decision surface.
struct CommunityBenchmarkShareConfirmationSheet: View {
    let preview: CommunityBenchmarkUploadPreview
    /// The alias assigned by a previous publication, when this installation
    /// already has one. Nil before the first receipt exists — the client must
    /// not preview or invent an identity it has not been given.
    let knownContributor: CommunityBenchmarkContributor?
    let isPublishing: Bool
    let onCancel: () -> Void
    let onPublish: () -> Void

    @State private var showsPayload = false
    @State private var didCopy = false

    /// What the submission actually contains.
    ///
    /// Derived from the preview, not asserted. The static list used to claim
    /// "Model name and quantisation", and for any run off a warm cache that
    /// was false: the projection replaces the measured quantization with
    /// `{kind: unknown}` and drops the resolved revision before sending. A
    /// consent dialog that overstates what it publishes is worse than no
    /// dialog.
    nonisolated static func sharedItems(
        for preview: CommunityBenchmarkUploadPreview
    ) -> [String] {
        var items = [String(localized: "Mac model, memory and core counts")]
        if let identity = preview.publishedIdentity {
            items.append(
                String(
                    format: String(localized: "The model's repository id (%1$@)"),
                    identity.repoID
                )
            )
            // Only claimed when the wire really carries it.
            if let quantization = identity.quantization.displayName {
                items.append(
                    String(
                        format: String(localized: "Its quantisation (%1$@)"),
                        quantization
                    )
                )
            }
            if identity.resolvedRevision != nil {
                items.append(String(localized: "The exact checkpoint revision"))
            }
        } else {
            items.append(String(localized: "The model's repository id"))
        }
        items.append(String(localized: "macOS and Rapid-MLX versions"))
        items.append(String(localized: "The timings and memory figures above"))
        return items
    }

    private static let neverShared: [String] = [
        String(localized: "Your name or any account"),
        String(localized: "Hardware serial or UUID"),
        String(localized: "Prompts or model output"),
        String(localized: "Files or file paths"),
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
                Text("Publish this result to the Community Benchmark?")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(RapidTheme.textPrimary)
                Text("It will be published publicly under your anonymous contributor alias on rapidmlx.com.")
                    .font(RapidFont.body)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(RapidTheme.Space.xl)

            identitySection
                .padding(.horizontal, RapidTheme.Space.xl)
                .padding(.bottom, RapidTheme.Space.lg)

            Divider()

            HStack(alignment: .top, spacing: 0) {
                fieldColumn(
                    systemImage: "checkmark.circle",
                    tone: RapidTheme.statusReady,
                    title: String(localized: "SHARED"),
                    items: Self.sharedItems(for: preview)
                )
                Divider()
                fieldColumn(
                    systemImage: "nosign",
                    tone: RapidTheme.textTertiary,
                    title: String(localized: "NEVER SHARED"),
                    items: Self.neverShared
                )
            }
            .fixedSize(horizontal: false, vertical: true)

            if !preview.withheld.isEmpty {
                Divider()
                withheldSection
            }

            Divider()

            payloadDisclosure

            Divider()

            HStack(spacing: RapidTheme.Space.md) {
                Text("The service briefly sees your IP address for rate limiting. It is not stored in the benchmark record.")
                    .font(RapidFont.caption)
                    .foregroundStyle(RapidTheme.textTertiary)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: RapidTheme.Space.md)
                Button(String(localized: "Cancel"), action: onCancel)
                    .buttonStyle(.rapidSecondary)
                    .keyboardShortcut(.cancelAction)
                    .accessibilityIdentifier("CommunityBenchmark.Share.Cancel")
                Button(isPublishing ? String(localized: "Publishing…") : String(localized: "Publish"), action: onPublish)
                    .buttonStyle(.rapidPrimary)
                    .keyboardShortcut(.defaultAction)
                    .disabled(isPublishing)
                    .accessibilityIdentifier("CommunityBenchmark.Share.Confirm")
            }
            .padding(RapidTheme.Space.xl)
        }
        .frame(width: 620)
        .frame(maxHeight: showsPayload ? 720 : nil)
        .background(RapidTheme.surfaceRaised)
        .accessibilityIdentifier("CommunityBenchmark.Share.Confirmation")
    }

    /// The facts the archive keeps and the submission does not.
    ///
    /// Shown, not buried: the CLI narrows the payload so the service will
    /// accept it, and a projection the user cannot inspect is indistinguishable
    /// from data loss.
    private var withheldSection: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
            HStack(spacing: 6) {
                Image(systemName: "eye.slash")
                    .font(.system(size: 11))
                    .accessibilityHidden(true)
                Text("KEPT ON THIS MAC, NOT PUBLISHED")
                    .font(.system(size: 11, weight: .semibold))
                    .tracking(0.5)
            }
            .foregroundStyle(RapidTheme.textTertiary)

            ForEach(preview.withheld) { fact in
                VStack(alignment: .leading, spacing: 2) {
                    Text("\(fact.fieldName) — \(fact.value)")
                        .font(RapidFont.secondary)
                        .foregroundStyle(RapidTheme.textPrimary)
                        .fixedSize(horizontal: false, vertical: true)
                    Text(fact.reason)
                        .font(RapidFont.caption)
                        .foregroundStyle(RapidTheme.textTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .accessibilityElement(children: .combine)
            }

            Text("Your saved copy of this result keeps these details. Only the submission is narrowed.")
                .font(RapidFont.caption)
                .foregroundStyle(RapidTheme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(RapidTheme.Space.xl)
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityIdentifier("CommunityBenchmark.Share.Withheld")
    }

    @ViewBuilder
    private var identitySection: some View {
        if let knownContributor {
            VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
                CommunityContributorIdentity(contributor: knownContributor, showsDestination: false)
                Text("This alias belongs to this installation of Rapid-MLX. Everything you publish from this Mac appears together on one contributor page, so people can see a consistent set of results without knowing who you are.")
                    .font(RapidFont.secondary)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        } else {
            // No receipt has been returned yet, so there is no alias to show.
            // Previewing a generated one would be inventing an identity the
            // service has not assigned.
            Text("The service assigns this installation an anonymous alias the first time you publish. Everything you publish from this Mac then appears together on one contributor page.")
                .font(RapidFont.secondary)
                .foregroundStyle(RapidTheme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func fieldColumn(
        systemImage: String,
        tone: Color,
        title: String,
        items: [String]
    ) -> some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
            HStack(spacing: 6) {
                Image(systemName: systemImage)
                    .font(.system(size: 11))
                    .foregroundStyle(tone)
                    .accessibilityHidden(true)
                Text(title)
                    .font(RapidFont.groupLabel)
                    .tracking(0.4)
                    .foregroundStyle(RapidTheme.textSecondary)
            }
            ForEach(items, id: \.self) { item in
                Text(item)
                    .font(RapidFont.body)
                    .foregroundStyle(RapidTheme.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(RapidTheme.Space.xl)
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .contain)
        .accessibilityLabel(title)
    }

    private var payloadDisclosure: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button {
                withAnimation(.easeInOut(duration: 0.15)) { showsPayload.toggle() }
            } label: {
                HStack(spacing: RapidTheme.Space.sm) {
                    Image(systemName: showsPayload ? "chevron.down" : "chevron.right")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(RapidTheme.textSecondary)
                        .accessibilityHidden(true)
                    Text(showsPayload ? String(localized: "Hide exact data") : String(localized: "Review the exact data"))
                        .font(RapidFont.bodyEmphasis)
                        .foregroundStyle(RapidTheme.textPrimary)
                    Text("Raw JSON and destination")
                        .font(RapidFont.secondary)
                        .foregroundStyle(RapidTheme.textTertiary)
                    Spacer(minLength: 0)
                }
                .padding(.horizontal, RapidTheme.Space.xl)
                .padding(.vertical, RapidTheme.Space.md)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("CommunityBenchmark.Share.PayloadDisclosure")
            .accessibilityAddTraits(.isButton)
            .accessibilityValue(showsPayload ? String(localized: "Expanded") : String(localized: "Collapsed"))

            if showsPayload {
                VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
                    payloadFact(String(localized: "Destination"), preview.target)
                    payloadFact(String(localized: "Payload digest"), preview.payloadDigest)
                    ScrollView {
                        Text(preview.payloadJSON)
                            .font(RapidFont.code)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(RapidTheme.Space.md)
                    }
                    // Fixed height so expanding the payload cannot push the
                    // Cancel/Publish buttons off the bottom of the sheet.
                    .frame(height: 220)
                    .background(
                        RapidTheme.surfaceCode,
                        in: RoundedRectangle(cornerRadius: RapidTheme.Radius.code)
                    )
                    HStack {
                        Button(didCopy ? String(localized: "Copied") : String(localized: "Copy JSON")) {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(preview.payloadJSON, forType: .string)
                            didCopy = true
                        }
                        .buttonStyle(.rapidSecondaryCompact)
                        .accessibilityIdentifier("CommunityBenchmark.Share.CopyPayload")
                        Spacer()
                    }
                }
                .padding(.horizontal, RapidTheme.Space.xl)
                .padding(.bottom, RapidTheme.Space.lg)
            }
        }
    }

    private func payloadFact(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: RapidTheme.Space.sm) {
            Text(label)
                .font(RapidFont.secondary)
                .foregroundStyle(RapidTheme.textSecondary)
                .frame(width: 110, alignment: .leading)
            Text(value)
                .font(RapidFont.code)
                .foregroundStyle(RapidTheme.textPrimary)
                .lineLimit(1)
                .truncationMode(.middle)
                .textSelection(.enabled)
        }
        .accessibilityElement(children: .combine)
    }
}

// MARK: - Published

/// The publication celebration. The one restrained mascot moment in the
/// publish flow.
///
/// The headline is chosen from the branch the Result screen showed, not from
/// the receipt alone, so a first-reference claim is only ever made when the
/// count was known to be zero before publishing.
struct CommunityBenchmarkPublishedSheet: View {
    let celebration: CommunityBenchmarkCopy.PublishedCelebration
    let receipt: CommunityBenchmarkReceipt
    /// Set when the upload succeeded but the local receipt could not be
    /// written, so My Results may still show the run as local.
    let receiptNotSavedWarning: String?
    let onDone: () -> Void

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var hasAppeared = false

    var body: some View {
        VStack(spacing: 0) {
            VStack(spacing: RapidTheme.Space.lg) {
                // Scale-only entrance. The mascot must never start at zero
                // opacity: if the animation does not run — Reduce Motion, a
                // static render, a dropped transaction — the celebration
                // would appear with an empty hole where the character is.
                CommunityMascot(context: .published)
                    .scaleEffect(hasAppeared || reduceMotion ? 1 : 0.94)

                VStack(spacing: RapidTheme.Space.sm) {
                    Text(celebration.headline)
                        .font(.system(size: 20, weight: .semibold))
                        .foregroundStyle(RapidTheme.textPrimary)
                    Text(celebration.body)
                        .font(RapidFont.body)
                        .foregroundStyle(RapidTheme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .multilineTextAlignment(.center)

                if let contributor = receipt.contributor {
                    CommunityContributorIdentity(contributor: contributor)
                }

                if let receiptNotSavedWarning {
                    InlineNotice(message: receiptNotSavedWarning, tone: .warning)
                }
            }
            .padding(RapidTheme.Space.xl)
            .frame(maxWidth: .infinity)

            Divider()

            HStack(spacing: RapidTheme.Space.md) {
                Spacer(minLength: 0)
                Button(String(localized: "Done"), action: onDone)
                    .buttonStyle(.rapidSecondary)
                    .accessibilityIdentifier("CommunityBenchmark.Published.Done")
                Link(destination: receipt.contributionURL) {
                    HStack(spacing: 6) {
                        Text("View your contribution")
                        Image(systemName: "arrow.up.right.square")
                            .font(.system(size: 11))
                    }
                }
                .buttonStyle(.rapidPrimary)
                .accessibilityLabel(receipt.contributionAccessibilityLabel)
                .accessibilityIdentifier("CommunityBenchmark.Published.Contribution")
            }
            .padding(RapidTheme.Space.xl)
        }
        .frame(width: 540)
        .background(RapidTheme.surfaceRaised)
        .accessibilityIdentifier("CommunityBenchmark.Published")
        .onAppear {
            guard !reduceMotion else { hasAppeared = true; return }
            withAnimation(.spring(response: 0.42, dampingFraction: 0.82)) {
                hasAppeared = true
            }
        }
    }
}
