import AppKit
import SwiftUI

/// "Connect your tools" — the second post-install call-to-action.
///
/// Once the local server is running it speaks the OpenAI and Anthropic
/// wire formats on `127.0.0.1`, so any coding tool that lets you point
/// at a custom base URL can use it for free. This sheet turns that from
/// "read the docs and assemble a config" into one click per tool.
///
/// The endpoint + key are passed in from the live ``ServerManager`` so
/// every copied snippet is correct for the current run (the port floats
/// 8000–8009 and the bearer rotates each start).
struct ConnectToolsView: View {
    let host: String
    let port: Int
    let bearer: String
    let alias: String
    var onClose: () -> Void
    /// Whether to render the top-right dismiss "✕". True in sheet context
    /// (the caller's ``onClose`` dismisses the sheet). False when embedded
    /// as a navigation PAGE (the Launch sidebar section), where there is no
    /// sheet to dismiss — showing a dead ✕ that does nothing was a real
    /// papercut. The sidebar owns navigation, so it passes false.
    var showsCloseButton: Bool = true

    private var openAIBaseURL: String { "http://\(host):\(port)/v1" }
    private var anthropicBaseURL: String { "http://\(host):\(port)" }
    private var modelName: String { alias.isEmpty ? "local" : alias }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            ScrollView {
                cardContent
            }
        }
        // Size follows context, the same way ``showsCloseButton`` does.
        //
        // 460×560 is a sheet's size. As the Launch *page* it left the cards in
        // a narrow column stranded in the middle of a 1200pt window, and the
        // hard 560pt height forced a second scrollbar inside a pane that was
        // already tall enough to show everything. Filling the pane instead
        // lets the cards use the width that is actually there.
        .frame(
            width: showsCloseButton ? 460 : nil,
            height: showsCloseButton ? 560 : nil
        )
        .frame(
            maxWidth: showsCloseButton ? nil : .infinity,
            maxHeight: showsCloseButton ? nil : .infinity,
            alignment: .topLeading
        )
        .background(RapidTheme.canvas)
    }

    /// The scrollable card list. Factored out so the snapshot harness can
    /// render it inside a fixed frame (``ImageRenderer`` collapses
    /// ``ScrollView`` content to zero height).
    @ViewBuilder
    var cardContent: some View {
        VStack(spacing: 12) {
            ForEach(tools) { tool in
                ConnectToolCard(tool: tool)
            }
            endpointFootnote
        }
        .padding(20)
        // Filling the pane is not the same as stretching to it. On a wide
        // window the cards would otherwise run the full 1200pt and the
        // one-line instructions would span the screen; capping the measure
        // keeps them readable while the surrounding pane still expands.
        .frame(maxWidth: showsCloseButton ? .infinity : 720)
    }

    private var header: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Connect your tools")
                    .font(.title3.weight(.semibold))
                Text("Point any editor at your local server. It's free and stays on your Mac.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                // The key only exists while the server is running, and it is
                // regenerated on every start. Without this the cards render
                // `API key:` followed by nothing — a config that looks
                // complete, copies clean, and fails at the far end with a 401.
                if bearer.isEmpty {
                    Label(
                        "Server not running — start a chat to generate the key.",
                        systemImage: "exclamationmark.circle"
                    )
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .padding(.top, 4)
                }
            }
            Spacer()
            if showsCloseButton {
                Button {
                    onClose()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title2)
                        .foregroundStyle(.tertiary)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Close")
            }
        }
        .padding(20)
    }

    private var endpointFootnote: some View {
        VStack(alignment: .leading, spacing: 6) {
            Divider()
            Text("Endpoint")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            CopyableRow(label: "OpenAI base URL", value: openAIBaseURL)
            CopyableRow(label: "Anthropic base URL", value: anthropicBaseURL)
            CopyableRow(label: "API key", value: bearer, masked: true)
            CopyableRow(label: "Model", value: modelName)
        }
        .padding(.top, 4)
    }

    // MARK: - Tool definitions

    private var tools: [ConnectTool] {
        [
            ConnectTool(
                id: "cursor",
                name: "Cursor",
                symbol: "cursorarrow.rays",
                blurb: "Settings → Models → add an OpenAI-compatible model with this base URL and key.",
                snippet: """
                Base URL: \(openAIBaseURL)
                API key:  \(bearer)
                Model:    \(modelName)
                """
            ),
            ConnectTool(
                id: "claude-code",
                name: "Claude Code",
                symbol: "terminal",
                blurb: "Export these before launching `claude` — it speaks the Anthropic format.",
                snippet: """
                export ANTHROPIC_BASE_URL=\(anthropicBaseURL)
                export ANTHROPIC_API_KEY=\(bearer)
                export ANTHROPIC_MODEL=\(modelName)
                """
            ),
            ConnectTool(
                id: "codex",
                name: "Codex",
                symbol: "chevron.left.forwardslash.chevron.right",
                blurb: "Export these before launching `codex` — OpenAI-compatible.",
                snippet: """
                export OPENAI_BASE_URL=\(openAIBaseURL)
                export OPENAI_API_KEY=\(bearer)
                """
            ),
        ]
    }
}

/// One tool's copyable card.
private struct ConnectTool: Identifiable {
    let id: String
    let name: String
    let symbol: String
    let blurb: String
    let snippet: String
}

private struct ConnectToolCard: View {
    let tool: ConnectTool
    @State private var copied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 9) {
                Image(systemName: tool.symbol)
                    .font(.system(size: 15, weight: .medium))
                    .foregroundStyle(RapidTheme.brand)
                    .frame(width: 22)
                Text(tool.name)
                    .font(.headline)
                Spacer()
                Button {
                    copy()
                } label: {
                    Label(copied ? "Copied" : "Copy config",
                          systemImage: copied ? "checkmark" : "doc.on.doc")
                        .font(.callout.weight(.medium))
                }
                .buttonStyle(.borderedProminent)
                .tint(copied ? RapidTheme.green : RapidTheme.brand)
            }
            Text(tool.blurb)
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Text(tool.snippet)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.primary)
                .textSelection(.enabled)
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(RapidTheme.sidebarSurface, in: RoundedRectangle(cornerRadius: 8))
        }
        .padding(14)
        .background(RapidTheme.card, in: RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(RapidTheme.hairline, lineWidth: 1)
        )
    }

    private func copy() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(tool.snippet, forType: .string)
        withAnimation { copied = true }
        Task {
            try? await Task.sleep(nanoseconds: 1_600_000_000)
            withAnimation { copied = false }
        }
    }
}

/// A labelled value with a trailing copy button; masks secrets by default.
private struct CopyableRow: View {
    let label: String
    let value: String
    var masked: Bool = false
    @State private var reveal = false
    @State private var copied = false

    private var shown: String {
        guard masked, !reveal else { return value }
        return String(repeating: "•", count: min(value.count, 16))
    }

    var body: some View {
        HStack(spacing: 8) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 118, alignment: .leading)
            Text(shown)
                .font(.system(.caption, design: .monospaced))
                .lineLimit(1)
                .truncationMode(.middle)
                .textSelection(.enabled)
            Spacer(minLength: 4)
            if masked {
                Button {
                    reveal.toggle()
                } label: {
                    Image(systemName: reveal ? "eye.slash" : "eye")
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .accessibilityLabel(reveal ? "Hide key" : "Show key")
            }
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(value, forType: .string)
                copied = true
                Task {
                    try? await Task.sleep(nanoseconds: 1_200_000_000)
                    copied = false
                }
            } label: {
                Image(systemName: copied ? "checkmark" : "doc.on.doc")
                    .foregroundStyle(copied ? RapidTheme.green : RapidTheme.brand)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Copy \(label)")
        }
    }
}
