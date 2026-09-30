import SwiftUI

/// The review step between pressing the primary action and starting a session.
///
/// It exists because joining the pool has three consequences the user should
/// agree to explicitly — their current model gets paused, third-party work
/// starts running locally, and the model comes back on Stop — and because a
/// first-time join needs a provider key. A Mac that is already registered
/// still sees the review (the consequences have not changed) but is not asked
/// for a key again.
struct ShareComputeConnectionReview: View {
    let modelTitle: String
    let requiresProviderKey: Bool
    @Binding var worker: String
    @Binding var providerKey: String
    let onCancel: () -> Void
    let onConnect: () -> Void

    @FocusState private var focus: Field?

    private enum Field: Hashable { case worker, key }

    private var canConnect: Bool {
        let trimmedWorker = worker.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedWorker.isEmpty else { return false }
        guard requiresProviderKey else { return true }
        let trimmedKey = providerKey.trimmingCharacters(in: .whitespacesAndNewlines)
        return !trimmedKey.isEmpty && ShareComputeManager.isValidProviderKey(trimmedKey)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.lg) {
            header
            assurances
            Divider()
            fields
            Spacer(minLength: 0)
            actions
        }
        .padding(RapidTheme.Space.xl)
        .frame(width: 560)
        .background(RapidTheme.surfaceOverlay)
        // A focused SecureField makes macOS offer its own "Passwords…"
        // AutoFill button, which is drawn by the system OVER this sheet —
        // in the review captures it landed on the sentence saying Rapid does
        // not store the key. It cannot be styled or dismissed from here, so
        // a review capture simply opens the sheet with nothing focused. The
        // gate is the shared golden-mode one; every normal launch still
        // lands the caret in the field the user has to fill.
        .onAppear {
            guard !ContentView.suppressesReviewChrome() else { return }
            focus = requiresProviderKey ? .key : .worker
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Review & connect this Mac")
                .font(.system(size: 19, weight: .bold))
                .foregroundStyle(RapidTheme.textPrimary)
            Text(modelTitle)
                .font(RapidFont.body)
                .foregroundStyle(RapidTheme.textSecondary)
        }
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isHeader)
    }

    private var assurances: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
            assurance(
                systemImage: "pause.circle",
                title: String(localized: "Pause current model"),
                detail: String(localized: "This prevents two large models from competing for unified memory.")
            )
            assurance(
                systemImage: "checkmark.shield",
                title: String(localized: "Run accepted work locally"),
                detail: String(localized: "QuickSilver work runs on this Mac; reward activity remains in your provider account.")
            )
            assurance(
                systemImage: "arrow.uturn.backward",
                title: String(localized: "Restore after Stop"),
                detail: String(localized: "Rapid leaves the pool and reloads your previous model automatically.")
            )
        }
    }

    private func assurance(
        systemImage: String,
        title: String,
        detail: String
    ) -> some View {
        HStack(alignment: .top, spacing: RapidTheme.Space.md) {
            Image(systemName: systemImage)
                .font(.system(size: 13))
                .foregroundStyle(RapidTheme.textSecondary)
                // Fixed slot so the three titles align down a lane regardless
                // of glyph width.
                .frame(width: 26, height: 26)
                .background(RapidTheme.surfaceCanvas, in: Circle())
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(RapidFont.bodyEmphasis)
                    .foregroundStyle(RapidTheme.textPrimary)
                Text(detail)
                    .font(RapidFont.secondary)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .accessibilityElement(children: .combine)
    }

    private var fields: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
            VStack(alignment: .leading, spacing: 5) {
                ShareComputeEyebrow(text: "Worker name", tone: RapidTheme.textSecondary, size: 10)
                TextField("", text: $worker)
                    .textFieldStyle(.roundedBorder)
                    .focused($focus, equals: .worker)
                    .accessibilityLabel(String(localized: "Worker name for this Mac"))
                    .accessibilityIdentifier("ShareCompute.Worker")
            }

            if requiresProviderKey {
                VStack(alignment: .leading, spacing: 5) {
                    HStack {
                        ShareComputeEyebrow(
                            text: "QuickSilver provider key",
                            tone: RapidTheme.textSecondary,
                            size: 10
                        )
                        Spacer(minLength: RapidTheme.Space.sm)
                        Link(destination: ShareComputeDestination.provider) {
                            HStack(spacing: 4) {
                                Text("Get a key").font(RapidFont.secondary)
                                Image(systemName: "arrow.up.right")
                                    .font(.system(size: 9, weight: .semibold))
                                    .accessibilityHidden(true)
                            }
                            .foregroundStyle(RapidTheme.linkLabel)
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel(String(localized: "Get a provider key from QuickSilver"))
                        .accessibilityIdentifier("ShareCompute.ProviderKeyLink")
                    }
                    // Opted out of content-type inference: a QuickSilver
                    // provider key is not a website password, and nothing in
                    // the keychain can usefully fill it. (This quietens the
                    // semantics, not the system AutoFill button — that one
                    // follows focus; see the `.onAppear` above.)
                    SecureField("qsppk-…", text: $providerKey)
                        .textContentType(.none)
                        .textFieldStyle(.roundedBorder)
                        .focused($focus, equals: .key)
                        .accessibilityLabel(String(localized: "QuickSilver provider key"))
                        .accessibilityIdentifier("ShareCompute.ProviderKey")
                    Text("Sent directly to the local provider process. Rapid does not save it in app settings.")
                        .font(RapidFont.caption)
                        .foregroundStyle(RapidTheme.textTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    private var actions: some View {
        HStack(spacing: RapidTheme.Space.md) {
            Spacer(minLength: 0)
            Button(String(localized: "Cancel"), action: onCancel)
                .buttonStyle(.rapidSecondary)
                .keyboardShortcut(.cancelAction)
                .accessibilityIdentifier("ShareCompute.Cancel")
            Button(
                requiresProviderKey
                    ? String(localized: "Connect & Share")
                    : String(localized: "Join the pool"),
                action: onConnect
            )
            .buttonStyle(.rapidPrimary)
            .keyboardShortcut(.defaultAction)
            .disabled(!canConnect)
            .accessibilityIdentifier("ShareCompute.Connect")
        }
    }
}
