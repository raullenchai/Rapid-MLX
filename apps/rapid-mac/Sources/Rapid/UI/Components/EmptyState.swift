import SwiftUI

/// The centred "nothing here yet" block: brand mark, title, one line of
/// supporting copy, an optional hint, and optional secondary actions.
///
/// Sizing is deliberately restrained. The pre-v1.0 chat empty state
/// used a 60pt disc with a 27pt glyph pinned 96pt from the top of the
/// transcript, which pushed the whole composition off-centre and made
/// a 640pt-tall window feel like a mostly-empty poster. Here the disc
/// is 44pt, the block is vertically centred by its container, and the
/// content column is width-capped so it stays a considered object
/// rather than stretching with the window.
///
/// The mark is generic over its content rather than type-erased through
/// ``AnyView``: the empty state's brand moment is a real bundled image
/// (``CheetahLogo``) on the chat surface but a plain SF Symbol
/// elsewhere, and both should stay statically typed so SwiftUI can
/// diff them properly.
///
/// Actions are ``rapidSecondaryCompact`` by contract: an empty state
/// offers side-doors, and none of them should out-shout the real
/// primary action on the surface (in chat, the composer).
struct EmptyState<Mark: View, Actions: View>: View {
    /// How loudly the title speaks.
    ///
    /// ``page`` (20pt) is right for an empty state that shares a window
    /// with other content — a results pane, a settings list. ``display``
    /// (34/40) is for the case where the empty state IS the window:
    /// the chat surface at rest has nothing else in it, so a 20pt line
    /// floating in 1440pt of canvas reads as a caption that lost its
    /// picture rather than as the product greeting you.
    enum TitleEmphasis {
        case page
        case display
    }

    let title: String
    var message: String? = nil
    /// A quieter third line — e.g. "First message will download X".
    var hint: String? = nil
    /// Diameter of the mark's frame — and, when ``marksOnBackplate`` is
    /// true, of the tinted disc behind it. 44 suits a small SF Symbol.
    var markDiameter: CGFloat = 44
    /// Whether the mark sits on a tinted disc.
    ///
    /// Direction D turns this off for the chat mascot. The plate was
    /// doing two jobs badly: it framed an illustration that already has
    /// its own silhouette, and — being amber-tinted — it spent a second
    /// amber moment on a surface whose budget is one. A symbol mark still
    /// wants the plate, because a 19pt glyph with no ground is a stray
    /// mark rather than an object.
    var marksOnBackplate: Bool = true
    var titleEmphasis: TitleEmphasis = .page
    @ViewBuilder var mark: Mark
    @ViewBuilder var actions: Actions

    var body: some View {
        VStack(spacing: RapidTheme.Space.md) {
            ZStack {
                if marksOnBackplate {
                    Circle()
                        .fill(RapidTheme.brandPrimaryTint)
                        .frame(width: markDiameter, height: markDiameter)
                }
                mark
            }
            // The mark is decoration; the title carries the meaning.
            .accessibilityHidden(true)
            .padding(.bottom, RapidTheme.Space.xs)

            VStack(spacing: RapidTheme.Space.xs) {
                titleText
                if let message {
                    Text(message)
                        .font(messageFont)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if let hint {
                    Text(hint)
                        .font(RapidFont.caption)
                        .foregroundStyle(.tertiary)
                        .multilineTextAlignment(.center)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            // The actions row is skipped entirely — not merely emptied —
            // when a caller uses the no-actions initialiser. An
            // ``HStack`` holding ``EmptyView`` still carries the top
            // padding below, which reads as a stray gap under the
            // subtitle on a surface that has no side-door actions at all.
            if Actions.self != EmptyView.self {
                HStack(spacing: RapidTheme.Space.sm) {
                    actions
                }
                .buttonStyle(.rapidSecondaryCompact)
                .padding(.top, RapidTheme.Space.xs)
            }
        }
        .frame(maxWidth: contentMaxWidth)
        .padding(.horizontal, RapidTheme.Space.xl)
    }

    @ViewBuilder
    private var titleText: some View {
        switch titleEmphasis {
        case .page:
            Text(title)
                .font(RapidFont.pageTitle)
                .foregroundStyle(.primary)
                .multilineTextAlignment(.center)
        case .display:
            Text(title)
                .font(RapidFont.displayTitle)
                // SF tightens as it grows; at 34pt the default spacing
                // reads loose enough to look like tracked-out display
                // type from a slide deck.
                .tracking(RapidFont.displayTitleTracking)
                .foregroundStyle(.primary)
                .multilineTextAlignment(.center)
        }
    }

    /// The supporting line steps up with the title. A 12pt subtitle
    /// under a 34pt display line reads as a footnote that wandered in
    /// from another surface.
    private var messageFont: Font {
        titleEmphasis == .display ? RapidFont.displaySubtitle : RapidFont.secondary
    }

    /// Display compositions get the reading measure; page-tier ones keep
    /// the tighter 380 they were laid out for.
    private var contentMaxWidth: CGFloat {
        titleEmphasis == .display ? RapidTheme.Layout.decisionMaxWidth : 380
    }
}

// MARK: - Convenience initialisers

extension EmptyState where Mark == EmptyStateSymbolMark {
    /// SF Symbol mark — for surfaces that aren't the brand moment.
    init(
        symbol: String,
        title: String,
        message: String? = nil,
        hint: String? = nil,
        @ViewBuilder actions: () -> Actions
    ) {
        self.init(
            title: title,
            message: message,
            hint: hint,
            markDiameter: 44,
            mark: { EmptyStateSymbolMark(symbol: symbol) },
            actions: actions
        )
    }
}

extension EmptyState where Actions == EmptyView {
    /// No side-door actions.
    init(
        title: String,
        message: String? = nil,
        hint: String? = nil,
        markDiameter: CGFloat = 44,
        marksOnBackplate: Bool = true,
        titleEmphasis: TitleEmphasis = .page,
        @ViewBuilder mark: () -> Mark
    ) {
        self.init(
            title: title,
            message: message,
            hint: hint,
            markDiameter: markDiameter,
            marksOnBackplate: marksOnBackplate,
            titleEmphasis: titleEmphasis,
            mark: mark,
            actions: { EmptyView() }
        )
    }
}

extension EmptyState where Mark == EmptyStateSymbolMark, Actions == EmptyView {
    /// SF Symbol mark, no actions.
    init(symbol: String, title: String, message: String? = nil, hint: String? = nil) {
        self.init(
            title: title,
            message: message,
            hint: hint,
            markDiameter: 44,
            mark: { EmptyStateSymbolMark(symbol: symbol) },
            actions: { EmptyView() }
        )
    }
}

/// The default SF Symbol mark, sized to sit inside the 44pt disc.
struct EmptyStateSymbolMark: View {
    let symbol: String

    var body: some View {
        Image(systemName: symbol)
            .font(.system(size: 19, weight: .semibold))
            .foregroundStyle(RapidTheme.brandPrimaryDeep)
    }
}
