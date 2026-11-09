import SwiftUI

/// Shared by API security settings and the integration endpoint panel.
struct LANAccessPanel: View {
    @Bindable var server: ServerManager
    @State private var showingPhoneConnect = false
    @State private var network = LANAddressMonitor()

    private var ready: Bool {
        if case .ready = server.state { return !server.isChangingLANAccess }
        return false
    }

    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
            Button("Connect phone") { showingPhoneConnect = true }
                .accessibilityIdentifier("LANAccess.ConnectPhone")
                .sheet(isPresented: $showingPhoneConnect) { PhoneConnectSheet(server: server) }

            Toggle("Allow access from other devices on your network", isOn: Binding(
                get: { server.lanAccessEnabled },
                set: { enabled in Task { await server.setLANAccessEnabled(enabled) } }
            ))
            .disabled(server.isChangingLANAccess || server.isOperating)
            .accessibilityIdentifier("Settings.API.LAN.Enabled")

            Text("Changing network access restarts a running model and interrupts active requests.")
                .font(RapidFont.caption)
                .foregroundStyle(.secondary)

            if server.isChangingLANAccess {
                ProgressView(String(localized: "Restarting the API with the new network setting…"))
                    .accessibilityIdentifier("Settings.API.LAN.Restarting")
            }
            if let error = server.lanAccessError {
                Text(error)
                    .foregroundStyle(RapidTheme.statusError)
                    .accessibilityIdentifier("Settings.API.LAN.Error")
            }
            if server.lanAccessEnabled {
                Text("LAN access uses a persistent API key. Traffic is plain HTTP on your local network. Only inference APIs are available to other devices.")
                    .font(RapidFont.caption)
                    .foregroundStyle(.secondary)
                if !ready && !server.isChangingLANAccess {
                    Text("Start the model to make the LAN API available.")
                        .font(RapidFont.caption)
                }
                if network.addresses.isEmpty {
                    Text("No network IPv4 address is available.")
                        .font(RapidFont.caption)
                }
                ForEach(network.addresses, id: \.self) { address in
                    if let url = LANAccess.baseURL(address: address, port: server.activePort) {
                        CopyableRow(label: "LAN URL", value: url, identifierSuffix: address)
                        let key = server.activeBearer ?? ""
                        let command = "curl '\(url)/models' -H 'Authorization: Bearer \(key)'"
                        let maskedCommand = "curl '\(url)/models' -H 'Authorization: Bearer ••••••••'"
                        CopyableRow(
                            label: "LAN curl example", value: ready && !key.isEmpty ? command : "",
                            masked: true, maskedDisplay: maskedCommand, identifierSuffix: address
                        )
                    }
                }
            }
        }
        .onAppear { network.start() }
        .onDisappear { network.stop() }
    }
}
