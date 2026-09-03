import SwiftUI

/// Port picker for the desktop server. Mirrors MTPLX "path" + port
/// control but for Rapid: lets the user pick their own number instead
/// of fighting over 8000. Stored in UserDefaults so it survives
/// restarts and feeds PortAllocator.storedPort() -> candidatePorts.
///
/// Placed in Connectors / Launch where the URL is already rendered
/// (ConnectToolsView.openAIBaseURL), so the snippet updates live.
/// Settings > Developer is also a fine home — move this view if you
/// prefer that placement; the storage key does not change.
struct DesktopServerPortField: View {
    @AppStorage("rapid.desktop.port") private var storedPortText: String = ""
    @State private var draft: String = ""
    @State private var saved: Bool = false

    private var parsed: Int? {
        Int(draft.trimmingCharacters(in: .whitespaces))
    }
    private var isValid: Bool {
        guard let p = parsed else { return false }
        return (1...65535).contains(p) && p >= 1024
    }
    private var isDirty: Bool { draft != storedPortText }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Server port")
                .font(.headline)
            HStack(spacing: 8) {
                TextField("7659", text: $draft)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 80)
                    .accessibilityIdentifier("ConnectTools.ServerPort.Field")
                    .onAppear { draft = storedPortText }
                Button(saved ? "Saved" : "Save") {
                    let t = draft.trimmingCharacters(in: .whitespaces)
                    if t.isEmpty {
                        UserDefaults.standard.removeObject(forKey: PortAllocator.storedPortKey)
                        storedPortText = ""
                    } else if let p = Int(t), (1...65_535).contains(p) {
                        UserDefaults.standard.set(p, forKey: PortAllocator.storedPortKey)
                        storedPortText = String(p)
                    }
                    saved = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { saved = false }
                }
                .disabled(!isDirty || (!draft.isEmpty && !isValid))
                .accessibilityIdentifier("ConnectTools.ServerPort.Save")
                if !draft.isEmpty && !isValid {
                    Text("1…65535, 1024+ recommended")
                        .font(.caption)
                        .foregroundStyle(.red)
                } else {
                    Text("Takes effect on next start. Restart to apply.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Text("Default 7659 (R M L X on a phone). 8000 fallback still probed. Like MTPLX path + port, this lets your gateway keep 8000.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
