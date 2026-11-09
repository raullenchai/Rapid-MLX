import CoreImage.CIFilterBuiltins
import SwiftUI

struct PhoneConnectSheet: View {
    @Bindable var server: ServerManager
    @Environment(\.dismiss) private var dismiss
    @State private var network = LANAddressMonitor()
    @State private var selectedAddress = ""

    private var ready: Bool {
        if case .ready = server.state { return !server.isChangingLANAccess }
        return false
    }

    private var address: String { network.addresses.contains(selectedAddress) ? selectedAddress : network.addresses.first ?? "" }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    Text("Connect phone").font(.title2)
                    Spacer()
                    Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
                }
                if !server.lanAccessEnabled {
                    Text("Enable LAN access to connect your phone. This saves a persistent API key and restarts a running model, interrupting active requests.")
                    Button("Enable LAN access") {
                        Task { await server.setLANAccessEnabled(true) }
                    }
                    .disabled(server.isChangingLANAccess || server.isOperating)
                } else if !ready {
                    Text("Start a chat model on your Mac to connect your phone.")
                } else if network.addresses.isEmpty {
                    Text("No LAN IPv4 address found. Connect your Mac to Wi-Fi.")
                } else {
                    if network.addresses.count > 1 {
                        Picker("Network address", selection: Binding(
                            get: { address }, set: { selectedAddress = $0 }
                        )) {
                            ForEach(network.addresses, id: \.self) { Text($0).tag($0) }
                        }
                    }
                    if let key = server.activeBearer,
                       let url = PhoneConnect.url(address: address, port: server.activePort, key: key, lanEnabled: server.lanAccessEnabled),
                       let base = LANAccess.baseURL(address: address, port: server.activePort) {
                        if let image = qrImage(url.absoluteString) {
                            Image(nsImage: image)
                                .interpolation(.none)
                                .resizable()
                                .scaledToFit()
                                .frame(width: 280, height: 280)
                                .padding(12)
                                .background(.white)
                                .accessibilityLabel("Scan with iPhone Camera to open Rapid")
                        }
                        CopyableRow(label: "Phone URL", value: url.absoluteString)
                        CopyableRow(label: "Base URL", value: base)
                        CopyableRow(label: "API key", value: key, masked: true)
                    } else {
                        Text("No active API key. Restart the model and try again.")
                    }
                }
                if server.isChangingLANAccess { ProgressView("Applying LAN setting…") }
                if let error = server.lanAccessError { Text(error).foregroundStyle(.red) }
                Text("Keep both devices on the same Wi-Fi and the Mac running. Traffic is plain HTTP. The QR contains your API key; share it only with people you trust.")
                    .font(.caption).foregroundStyle(.secondary)
            }.padding(24)
        }
        .frame(width: 520, height: 680)
        .onAppear { network.start() }
        .onDisappear { network.stop() }
    }

    private func qrImage(_ value: String) -> NSImage? {
        let filter = CIFilter.qrCodeGenerator()
        filter.message = Data(value.utf8)
        filter.correctionLevel = "M"
        guard let output = filter.outputImage,
              let cg = CIContext().createCGImage(output, from: output.extent) else { return nil }
        return NSImage(cgImage: cg, size: NSSize(width: cg.width, height: cg.height))
    }
}
