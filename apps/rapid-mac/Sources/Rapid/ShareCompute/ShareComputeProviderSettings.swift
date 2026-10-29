import SwiftUI

/// Preferences survive offline sessions; saving them never starts inference.
struct ShareComputeProviderSettings: View {
    @Bindable var manager: ShareComputeManager
    @AppStorage("Rapid.shareCompute.worker")
    private var worker = Host.current().localizedName ?? "Mac"
    @State private var draft = ""

    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
            Text("Share Qwen 27B with QuickSilver")
                .font(RapidFont.bodyEmphasis)
            Text("Requires an Apple Silicon Mac with 48 GB+ memory. Starting sharing unloads your local model. Stop takes this Mac offline and releases the sharing model; local models stay unloaded.")
                .font(RapidFont.secondary)
                .foregroundStyle(RapidTheme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)

            TextField("Worker name", text: $worker)
                .textFieldStyle(.roundedBorder)
                .disabled(!manager.canEditProviderKey)
                .accessibilityLabel("Share Compute worker name")
                .accessibilityIdentifier("Settings.ShareCompute.Worker")
            Text("A name for this Mac. Use a different name on each Mac.")
                .font(RapidFont.caption)
                .foregroundStyle(RapidTheme.textSecondary)

            if let label = manager.savedProviderKeyLabel {
                Text("Saved provider key: \(label)")
                    .font(RapidFont.secondary)
            }
            SecureField("Provider key · qsppk-…", text: $draft)
                .textContentType(.none)
                .textFieldStyle(.roundedBorder)
                .disabled(!manager.canEditProviderKey)
                .accessibilityLabel("QuickSilver provider key")
                .accessibilityIdentifier("Settings.ShareCompute.ProviderKey")
            Text("Saved securely in this Mac’s Keychain. You can stop sharing and come back without pasting it again. The same key also powers the Credits tab — Rapid’s one QuickSilver credential, nothing extra to paste there.")
                .font(RapidFont.caption)
                .foregroundStyle(RapidTheme.textSecondary)

            HStack {
                Button("Save key") {
                    if manager.saveProviderKey(draft) { draft = "" }
                }
                .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !manager.canEditProviderKey)
                .accessibilityIdentifier("Settings.ShareCompute.SaveProviderKey")
                Button("Remove saved key") {
                    if manager.removeProviderKey() { draft = "" }
                }
                .disabled(!manager.providerKeyIsStored || !manager.canEditProviderKey)
                .accessibilityIdentifier("Settings.ShareCompute.RemoveProviderKey")
                Spacer()
                Link("Get a provider key", destination: ShareComputeDestination.provider)
                    .accessibilityIdentifier("Settings.ShareCompute.GetProviderKey")
            }
            if let message = manager.providerKeyMessage {
                Text(message)
                    .font(RapidFont.secondary)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .accessibilityIdentifier("Settings.ShareCompute.ProviderKeyStatus")
            }
            if !manager.canEditProviderKey {
                Text("Stop sharing before changing the worker or provider key.")
                    .font(RapidFont.caption)
                    .foregroundStyle(RapidTheme.textSecondary)
            }
        }
        .onAppear { manager.refreshProviderKey() }
        .onDisappear { draft = "" }
    }
}
