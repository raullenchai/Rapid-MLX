import SwiftUI

/// Add or edit one MCP server (issue #1716).
///
/// This is the form that replaces hand-authoring `mcp.json`. It deliberately
/// validates the name locally — the engine namespaces every tool as
/// `server__tool`, so a name with a space in it produces tool names the model
/// can never call, and finding that out through "the model just ignores that
/// server" is a bad afternoon.
///
/// The engine still gets the final say on the command itself
/// (`vllm_mlx/mcp/security.py` allowlists what may be spawned). We don't
/// duplicate that list here: it moves independently of the app, and a
/// client-side copy that drifts would either block something valid or promise
/// something that then fails at connect. The rejection reason comes back on
/// the server row instead.
struct MCPServerEditorSheet: View {
    /// nil when adding.
    let original: MCPServerConfig?
    let onSave: (MCPServerConfig) -> Void
    let onCancel: () -> Void

    @State private var name: String
    @State private var transport: MCPServerConfig.Transport
    @State private var command: String
    @State private var url: String
    @State private var enabled: Bool
    /// Args and env are edited as text — one per line, `KEY=value` for env.
    /// A table of add/remove rows is more clicks for the same result, and this
    /// shape pastes straight out of any MCP README.
    @State private var argsText: String
    @State private var envText: String

    init(
        original: MCPServerConfig?,
        onSave: @escaping (MCPServerConfig) -> Void,
        onCancel: @escaping () -> Void
    ) {
        self.original = original
        self.onSave = onSave
        self.onCancel = onCancel
        _name = State(initialValue: original?.name ?? "")
        _transport = State(initialValue: original?.transport ?? .stdio)
        _command = State(initialValue: original?.command ?? "")
        _url = State(initialValue: original?.url ?? "")
        _enabled = State(initialValue: original?.enabled ?? true)
        _argsText = State(initialValue: (original?.args ?? []).joined(separator: "\n"))
        _envText = State(initialValue: (original?.env ?? [:])
            .sorted { $0.key < $1.key }
            .map { "\($0.key)=\($0.value)" }
            .joined(separator: "\n"))
    }

    /// The value that would be saved. Built fresh each render so the inline
    /// validation message and the Save button can never disagree about it.
    private var draft: MCPServerConfig {
        MCPServerConfig(
            name: name.trimmingCharacters(in: .whitespaces),
            transport: transport,
            command: transport == .stdio
                ? command.trimmingCharacters(in: .whitespaces)
                : nil,
            args: transport == .stdio ? Self.parseLines(argsText) : [],
            env: transport == .stdio ? Self.parseEnv(envText) : [:],
            url: transport == .sse ? url.trimmingCharacters(in: .whitespaces) : nil,
            enabled: enabled,
            timeout: original?.timeout ?? 30
        )
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(original == nil ? "Add connector" : "Edit “\(original?.name ?? "")”")
                .font(.headline)

            Form {
                TextField("Name", text: $name)
                    .accessibilityIdentifier("Settings.Connectors.Editor.Name")
                Text("Letters, numbers, dashes and underscores. Becomes the prefix on every tool this connector exposes.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Picker("Type", selection: $transport) {
                    ForEach(MCPServerConfig.Transport.allCases, id: \.self) { t in
                        Text(t.displayName).tag(t)
                    }
                }
                .accessibilityIdentifier("Settings.Connectors.Editor.Transport")

                switch transport {
                case .stdio:
                    TextField("Command", text: $command)
                        .accessibilityIdentifier("Settings.Connectors.Editor.Command")
                    Text("For example `uvx` or `npx`. Rapid's engine only runs commands on its allowlist.")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    VStack(alignment: .leading, spacing: 4) {
                        Text("Arguments — one per line")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        TextEditor(text: $argsText)
                            .font(.system(.callout, design: .monospaced))
                            .frame(height: 64)
                            .overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))
                            .accessibilityIdentifier("Settings.Connectors.Editor.AddArgument")
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        Text("Environment — one KEY=value per line")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        TextEditor(text: $envText)
                            .font(.system(.callout, design: .monospaced))
                            .frame(height: 52)
                            .overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))
                            .accessibilityIdentifier("Settings.Connectors.Editor.AddEnv")
                    }

                case .sse:
                    TextField("URL", text: $url)
                        .accessibilityIdentifier("Settings.Connectors.Editor.URL")
                    Text("An http:// or https:// endpoint speaking MCP over SSE.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Toggle("Enabled", isOn: $enabled)
                    .accessibilityIdentifier("Settings.Connectors.Editor.Enabled")
            }
            .formStyle(.grouped)

            if let why = draft.validationError, !name.isEmpty || !command.isEmpty || !url.isEmpty {
                // Held back until the user has typed something — an empty form
                // that scolds you before you start is noise, not guidance.
                Label(why, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack {
                Spacer()
                Button("Cancel", action: onCancel)
                    .keyboardShortcut(.cancelAction)
                    .accessibilityIdentifier("Settings.Connectors.Editor.Cancel")
                Button("Save") { onSave(draft) }
                    .keyboardShortcut(.defaultAction)
                    .disabled(draft.validationError != nil)
                    .accessibilityIdentifier("Settings.Connectors.Editor.Allow")
            }
        }
        .padding(20)
        .frame(width: 480)
    }

    /// Non-empty, whitespace-trimmed lines. `static` so the parsing can be
    /// pinned by tests without standing up the view.
    static func parseLines(_ text: String) -> [String] {
        text.split(whereSeparator: { $0 == "\n" || $0 == "\r" })
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }

    /// `KEY=value` per line. Splits on the FIRST `=` so a value containing one
    /// survives; a line with no `=` is skipped rather than becoming an empty
    /// key the engine would then have to reject.
    static func parseEnv(_ text: String) -> [String: String] {
        var out: [String: String] = [:]
        for line in parseLines(text) {
            guard let idx = line.firstIndex(of: "=") else { continue }
            let key = String(line[line.startIndex..<idx]).trimmingCharacters(in: .whitespaces)
            let value = String(line[line.index(after: idx)...])
            if !key.isEmpty { out[key] = value }
        }
        return out
    }
}
