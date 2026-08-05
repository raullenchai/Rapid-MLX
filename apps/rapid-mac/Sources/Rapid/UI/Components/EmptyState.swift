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
    let title: String
    var message: String? = nil
    /// A quieter third line — e.g. "First message will download X".
    var hint: String? = nil
    /// Diameter of the tinted backing disc. 44 suits a small SF Symbol;
    /// the chat surface passes ~92 so the mascot reads as a real brand
    /// moment rather than a favicon.
    var markDiameter: CGFloat = 44
    @ViewBuilder var mark: Mark
    @ViewBuilder var actions: Actions

    var body: some View {
        VStack(spacing: RapidTheme.Space.md) {
            ZStack {
                Circle()
                    .fill(RapidTheme.brandPrimaryTint)
                    .frame(width: markDiameter, height: markDiameter)
                mark
            }
            // The mark is decoration; the title carries the meaning.
            .accessibilityHidden(true)
            .padding(.bottom, RapidTheme.Space.xs)

            VStack(spacing: RapidTheme.Space.xs) {
                Text(title)
                    .font(RapidFont.pageTitle)
                    .foregroundStyle(.primary)
                if let message {
                    Text(message)
                        .font(RapidFont.secondary)
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

            HStack(spacing: RapidTheme.Space.sm) {
                actions
            }
            .buttonStyle(.rapidSecondaryCompact)
            .padding(.top, RapidTheme.Space.xs)
        }
        .frame(maxWidth: 380)
        .padding(.horizontal, RapidTheme.Space.xl)
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
        @ViewBuilder mark: () -> Mark
    ) {
        self.init(
            title: title,
            message: message,
            hint: hint,
            markDiameter: markDiameter,
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
