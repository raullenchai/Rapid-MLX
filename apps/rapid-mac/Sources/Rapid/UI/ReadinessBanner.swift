import SwiftUI

/// The one rendering of ``ModelReadiness``.
///
/// Sits directly above the compose field and answers, in one row: what
/// state the model is in, why sending is unavailable, and the single
/// next thing to do about it. The chat hero reads its copy off the same
/// ``ModelReadiness`` value (``emptyStateSubtitle`` / ``emptyStateHint``),
/// so the two surfaces are two renderings of one fact rather than two
/// independently-maintained descriptions.
///
/// Visual system: this is deliberately built from parts that already
/// shipped — ``PulsingStateDot``, the ``RapidTheme`` status tokens, the
/// ``InlineNotice`` tint pairing, and ``RapidPrimaryButtonStyle``. It
/// introduces no new shape, colour, or type size. Amber remains the
/// working/active hue; red is reserved for genuine faults; steel blue
/// does not appear.
///
/// The action button takes the PRIMARY tier rather than the secondary
/// one the empty state uses. That is not an escalation for its own sake:
/// while the model is not ready the composer's Send button is disabled,
/// so Start genuinely is the one high-emphasis action on this surface.
/// It steps back down the moment Send becomes live, because the banner
/// hides entirely in ``ModelReadiness/ready``.
struct ReadinessBanner: View {
    let readiness: ModelReadiness
    /// Bumped by the composer each time a send is attempted while gated.
    /// Drives a brief emphasis so a blocked Return is never silent.
    var attentionToken: Int = 0
    var onAction: (ModelReadiness.Action) -> Void

    // Reduce Motion is handled inside the pieces this composes:
    // ``PulsingStateDot`` suppresses its own breathing loop, and
    // ``rapidAnimation`` resolves to an instant change. The emphasis
    // below therefore still *happens* under Reduce Motion — it just
    // snaps rather than fading, which is the correct behaviour for a
    // feedback cue the user is waiting on.
    @State private var attentionActive = false

    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
            HStack(alignment: .center, spacing: RapidTheme.Space.sm) {
                PulsingStateDot(color: tint, isAnimating: readiness.isWorking)
                    .accessibilityHidden(true)

                VStack(alignment: .leading, spacing: 1) {
                    Text(readiness.headline)
                        .font(RapidFont.bodyEmphasis)
                        .foregroundStyle(.primary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    if let detail = readiness.detail {
                        Text(detail)
                            .font(RapidFont.caption)
                            .foregroundStyle(.secondary)
                            // Byte counts and ETAs update every 500 ms;
                            // monospaced digits stop the line jittering
                            // its own width as they tick.
                            .monospacedDigit()
                            .lineLimit(2)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                if let action = readiness.action, action.isRenderable {
                    Button {
                        onAction(action)
                    } label: {
                        Label(action.title, systemImage: action.systemImage)
                    }
                    .buttonStyle(RapidPrimaryButtonStyle(
                        height: RapidTheme.ControlHeight.small
                    ))
                    .fixedSize()
                    .accessibilityIdentifier("Readiness.Action")
                }
            }

            // Determinate only when a real fraction exists. An
            // indeterminate bar here would compete with the pulsing dot
            // for the same "something is happening" job while implying a
            // precision we do not have.
            if let fraction = readiness.progressFraction {
                ProgressView(value: min(max(fraction, 0), 1), total: 1)
                    .progressViewStyle(.linear)
                    .tint(RapidTheme.brandPrimary)
                    .accessibilityHidden(true)
            }
        }
        .padding(.horizontal, RapidTheme.Space.md)
        .padding(.vertical, RapidTheme.Space.sm)
        // A fixed floor, so the composer's top edge does not step up and
        // down as the user moves between "no model chosen", "not
        // downloaded" and "not running". Those are three renderings of
        // one moment; the surface should not reflow between them.
        .frame(minHeight: 48)
        .background(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.button, style: .continuous)
                .fill(background)
        )
        .overlay(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.button, style: .continuous)
                .strokeBorder(
                    borderTint.opacity(attentionActive ? 0.85 : 1),
                    lineWidth: attentionActive ? 1.5 : 1
                )
        )
        .accessibilityElement(children: .contain)
        .accessibilityLabel(readiness.accessibilityLabel)
        .rapidAnimation(RapidMotion.quick, value: attentionActive)
        .task(id: attentionToken) {
            // Token 0 is the initial value — only a real bump (a blocked
            // send) should flash, otherwise the banner would emphasise
            // itself the moment it appears.
            guard attentionToken > 0 else { return }
            attentionActive = true
            try? await Task.sleep(nanoseconds: 1_400_000_000)
            guard !Task.isCancelled else { return }
            attentionActive = false
        }
    }

    /// The four status tokens, keyed off the same role vocabulary
    /// ``ServerStatusPill`` uses for ``ServerState``.
    private var tint: Color {
        switch readiness.statusRole {
        case .idle:    return RapidTheme.statusIdle
        case .working: return RapidTheme.statusWorking
        case .ready:   return RapidTheme.statusReady
        case .error:   return RapidTheme.statusError
        }
    }

    /// Failure keeps its red tint; everything else sits on a plain raised
    /// surface.
    ///
    /// The amber tint this used to paint in every non-failure state was a
    /// second amber block roughly 40pt above the send disc — the one
    /// amber moment the composer is allowed. Two of them competing meant
    /// neither read as the thing to look at. The dot still carries the
    /// state's colour, which is where a status hue belongs: on the
    /// indicator, not on the whole plate behind the sentence.
    private var background: Color {
        readiness.isFailure ? RapidTheme.statusErrorTint : RapidTheme.surfaceRaised
    }

    /// Matching edge — the status hue on a failure, a plain hairline
    /// otherwise, so a neutral notice does not draw a coloured outline
    /// around itself for no reason.
    private var borderTint: Color {
        readiness.isFailure ? tint.opacity(0.35) : RapidTheme.hairline
    }
}
