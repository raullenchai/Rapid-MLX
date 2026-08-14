import AppKit
import SwiftUI

/// "Connect your agents" — the second post-install call-to-action.
///
/// Once the local server is running it speaks the OpenAI and Anthropic
/// wire formats on `127.0.0.1`, so any coding tool that lets you point
/// at a custom local base URL can use it for free. This sheet turns that from
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
    /// The absolute path to the `rapid-mlx` sidecar binary this app owns
    /// (``ServerLocator`` resolution). The Desktop app deliberately does NOT
    /// install its sidecar onto the user's `PATH` (see ``ServerLocator`` —
    /// PATH/brew/pipx/uv are intentionally not used), so generated launch
    /// commands must reference the binary by its absolute path or pasting
    /// them in a terminal fails with `command not found: rapid-mlx`.
    /// ``nil`` (the dev snapshot harness) falls back to the bare `rapid-mlx`
    /// so the page still renders.
    var binaryPath: URL? = nil
    var onClose: () -> Void
    /// Whether to render the top-right dismiss "✕". True in sheet context
    /// (the caller's ``onClose`` dismisses the sheet). False when embedded
    /// as a navigation PAGE (the Launch sidebar section), where there is no
    /// sheet to dismiss — showing a dead ✕ that does nothing was a real
    /// papercut. The sidebar owns navigation, so it passes false.
    var showsCloseButton: Bool = true
    /// The window's shared readiness value, when the caller has one.
    ///
    /// Before this, the page derived readiness twice and locally, and
    /// rendered BOTH results at once: a header line saying "start a
    /// chat to generate the key" and a body notice saying "Start a
    /// model to generate your local endpoint and key". Two verbs, two
    /// placements, one condition — and neither offered a way to do the
    /// thing it asked for. Supplying ``readiness`` replaces both with
    /// the same banner (and the same next-step action) the composer
    /// shows. ``nil`` keeps the legacy local sentence for the dev
    /// snapshot harness, which has no live server to resolve against.
    var readiness: ModelReadiness? = nil
    var onReadinessAction: (ModelReadiness.Action) -> Void = { _ in }
    @State private var integrationTargets: [IntegrationTarget] = []

    private var openAIBaseURL: String { "http://\(host):\(port)/v1" }
    private var anthropicBaseURL: String { "http://\(host):\(port)" }
    private var serverOrigin: String { "http://\(host):\(port)" }

    /// The shell command that invokes this app's `rapid-mlx` sidecar.
    ///
    /// The sidecar never lands on the user's `PATH` (see ``binaryPath``), so
    /// the copied launch/agent commands must call it by absolute path. When
    /// no binary is resolved (dev snapshot) we fall back to the bare command.
    private var cliCommand: String {
        guard let binary = binaryPath else { return "rapid-mlx" }
        return IntegrationLaunchCommand.shellQuote(binary.path)
    }

    /// The model id to publish in a config, or ``nil`` when no real
    /// model is resolved yet. Deliberately not defaulted to a
    /// plausible-looking literal — a copied config carrying a made-up
    /// model name fails later, somewhere else, with a worse error.
    private var resolvedModel: String? { ModelDisplayName.configValue(alias: alias) }

    /// What the user sees in the `Model` row while nothing is resolved.
    private var modelDisplay: String { resolvedModel ?? "Not started yet" }

    /// The model slot inside a displayed snippet. When nothing is
    /// resolved the snippet shows an obvious angle-bracket placeholder
    /// rather than a plausible-looking literal — and Copy is disabled
    /// in that state anyway, so this text is read, never pasted.
    private var snippetModel: String { resolvedModel ?? "<start a model first>" }

    /// The REAL key, for the clipboard only (``ConnectTool.snippet``). Never
    /// painted on screen — see ``snippetKeyMasked``.
    private var snippetKey: String { bearer.isEmpty ? "<starts with your server>" : bearer }

    /// Masked key for the always-visible snippet (``ConnectTool.displaySnippet``).
    /// The API-key ``CopyableRow`` above masks the bearer behind an eye toggle,
    /// but the config snippets rendered right below it interpolated the raw key
    /// in cleartext — so a screenshot of this page (the Launch "connect your
    /// tools" surface users naturally share) leaked the bearer despite the dots
    /// above. Render dots here; the real key still reaches the clipboard on Copy
    /// and stays revealable via the API-key row's eye. Mirrors ``CopyableRow``'s
    /// masking (bullets, length-capped so the key length isn't leaked either).
    private var snippetKeyMasked: String {
        bearer.isEmpty ? "<starts with your server>" : String(repeating: "•", count: min(bearer.count, 16))
    }

    /// A config is only complete once the server is actually listening
    /// (so the port is real) AND it has minted a bearer AND a model is
    /// resolved. Anything less and the snippets below would be a
    /// half-filled template.
    private var configReady: Bool {
        port > 0 && !bearer.isEmpty && resolvedModel != nil
    }

    /// One concise sentence naming what is missing.
    private var readinessMessage: String {
        if resolvedModel == nil && bearer.isEmpty {
            return "Start a model to generate your local endpoint and key."
        }
        if bearer.isEmpty {
            return "Your API key is created when the local server starts."
        }
        return "Start a model to fill in the model name."
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            ScrollView {
                cardContent
            }
        }
        // Sheet context keeps a fixed dialog size. Page context (the
        // Launch sidebar section) FILLS its column instead of sitting in a
        // 460pt box inside a resizable pane — that fixed frame is why the
        // page had a hard right edge and could not use the window it was
        // given. main already fixes the same #1470 defect via this modifier,
        // so we keep it rather than the PR's inline frame.
        .modifier(ConnectToolsFrame(fixedSize: showsCloseButton))
        .background(RapidTheme.surfaceCanvas)
        .task {
            integrationTargets = await IntegrationCatalog.load()
        }
    }

    /// The scrollable content. Factored out so the snapshot harness can
    /// render it inside a fixed frame (``ImageRenderer`` collapses
    /// ``ScrollView`` content to zero height).
    ///
    /// v1.0 structure: one endpoint card and one tools card, each a
    /// group of hairline-separated rows. Previously this was three
    /// free-floating cards, each of which contained a second nested
    /// card for its snippet — card-inside-card three times over, which
    /// is what produced the tall, heavy stack.
    @ViewBuilder
    var cardContent: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
            // Honest about readiness rather than presenting a
            // half-filled template as a working config. The sheet is
            // always reachable (see ChatView's empty-state CTA), so
            // this is the surface that has to explain the "not yet"
            // case instead of the button hiding it.
            //
            // One notice, never two. When the window supplies a shared
            // readiness value it wins outright — it is the same object
            // the composer renders, so the two surfaces cannot disagree,
            // and unlike the old local sentence it carries the action
            // that resolves the problem.
            if let readiness, !readiness.isReady {
                ReadinessBanner(readiness: readiness, onAction: onReadinessAction)
            } else if !configReady {
                // Either no readiness was supplied (dev snapshot), or the
                // model is up but a value is still missing — a narrow
                // case, but silence there would leave a half-filled
                // config looking complete.
                InlineNotice(message: readinessMessage, tone: .info)
            }
            endpointSection
            toolsSection
        }
        .frame(maxWidth: RapidTheme.Layout.pageMaxWidth, alignment: .leading)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(RapidTheme.Space.xl)
    }

    private var header: some View {
        HStack(alignment: .top, spacing: RapidTheme.Space.md) {
            VStack(alignment: .leading, spacing: RapidTheme.Space.xs) {
                SectionHeader(
                    "Connect your agents",
                    subtitle: "Connect any agent or editor that supports a local base URL. It's free and stays on your Mac.",
                    emphasis: .page
                )
                // The #1470 "start a chat to generate the key" hint used
                // to live here, duplicating the body's readiness notice
                // with a different verb. The fact it carried — the key
                // only exists while the server runs — is preserved by
                // the API key row's own placeholder ("Created when the
                // server starts") and, when the caller supplies one, by
                // the readiness banner below, which also says how to
                // fix it. One statement, in one place, with an action.
            }
            Spacer()
            if showsCloseButton {
                SheetCloseButton(action: onClose)
                    .accessibilityIdentifier("ConnectTools.Close")
            }
        }
        .frame(maxWidth: RapidTheme.Layout.pageMaxWidth, alignment: .leading)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, RapidTheme.Space.xl)
        .padding(.vertical, RapidTheme.Space.lg)
    }

    /// The live connection values, promoted above the per-tool rows.
    ///
    /// Every snippet below is assembled from exactly these four values,
    /// so showing them once at the top means the per-tool rows no longer
    /// have to repeat the full base URL + key + model three times.
    private var endpointSection: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
            SectionHeader("Endpoint")
            VStack(spacing: 0) {
                // Base URLs are real information at all times — the
                // loopback address and port are known before anything
                // starts — so they render at full reading contrast and
                // stay copyable. Only values that genuinely don't exist
                // yet (the key, the model) recede and disable their
                // Copy control.
                CopyableRow(label: "OpenAI base URL", value: openAIBaseURL)
                rowDivider
                CopyableRow(label: "Anthropic base URL", value: anthropicBaseURL)
                rowDivider
                CopyableRow(
                    label: "API key",
                    value: bearer,
                    masked: true,
                    placeholder: "Created when the server starts"
                )
                rowDivider
                CopyableRow(
                    label: "Model",
                    value: resolvedModel ?? "",
                    placeholder: modelDisplay
                )
            }
            .groupedCard()
        }
    }

    private var toolsSection: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
            SectionHeader("Editors and agents")
            VStack(spacing: 0) {
                ForEach(Array(tools.enumerated()), id: \.element.id) { index, tool in
                    if index > 0 { rowDivider }
                    ConnectToolRow(tool: tool, isReady: configReady)
                }
            }
            .groupedCard()
        }
    }

    private var rowDivider: some View {
        Rectangle()
            .fill(RapidTheme.hairline)
            .frame(height: 1)
            .padding(.leading, RapidTheme.Space.md)
    }

    // MARK: - Tool definitions

    private var tools: [ConnectTool] {
        guard !integrationTargets.isEmpty else { return legacyTools }
        return integrationTargets.map { target in
            let isWriter = target.kind == .configWriter
            let command: String
            let displayCommand: String
            if isWriter {
                command = IntegrationLaunchCommand.configWriter(
                    id: target.id, serverURL: serverOrigin, key: snippetKey, model: snippetModel, cli: cliCommand
                )
                displayCommand = IntegrationLaunchCommand.configWriter(
                    id: target.id, serverURL: serverOrigin, key: snippetKeyMasked, model: snippetModel, cli: cliCommand
                )
            } else {
                command = IntegrationLaunchCommand.adapterGuide(
                    id: target.id, baseURL: openAIBaseURL, model: snippetModel, cli: cliCommand
                )
                displayCommand = command
            }
            let destination = target.configPath.map { " It writes \($0)." } ?? ""
            let cursorCaveat = target.id == "cursor"
                ? " Cursor requires a public HTTPS endpoint; localhost cannot be reached by Cursor's backend."
                : ""
            return ConnectTool(
                id: target.id,
                name: target.name,
                symbol: isWriter ? "slider.horizontal.3" : "point.3.connected.trianglepath.dotted",
                blurb: isWriter
                    ? "Configure this client to use Rapid-MLX.\(destination)\(cursorCaveat)"
                    : "View this adapter's setup guide for the local endpoint.",
                snippet: command,
                displaySnippet: displayCommand
            )
        }
    }

    /// Available while an older or missing sidecar cannot expose the registry.
    private var legacyTools: [ConnectTool] {
        [
            ConnectTool(
                id: "claude-code",
                name: "Claude Code",
                symbol: "terminal",
                blurb: "Launch with this connection for one session. Your shell environment stays unchanged.",
                snippet: AgentLaunchCommand.claude(
                    baseURL: anthropicBaseURL, key: snippetKey, model: snippetModel
                ),
                displaySnippet: AgentLaunchCommand.claude(
                    baseURL: anthropicBaseURL, key: snippetKeyMasked, model: snippetModel
                )
            ),
            ConnectTool(
                id: "codex",
                name: "Codex",
                symbol: "chevron.left.forwardslash.chevron.right",
                blurb: "Launch with an isolated Rapid provider for one session. Your existing Codex provider and shell environment stay unchanged.",
                snippet: AgentLaunchCommand.codex(
                    baseURL: openAIBaseURL, key: snippetKey, model: snippetModel
                ),
                displaySnippet: AgentLaunchCommand.codex(
                    baseURL: openAIBaseURL, key: snippetKeyMasked, model: snippetModel
                )
            ),
            ConnectTool(
                id: "hermes",
                name: "Hermes",
                symbol: "bolt.horizontal.circle",
                blurb: "Launch with this connection and model for one session. Your shell environment stays unchanged.",
                snippet: AgentLaunchCommand.hermes(
                    baseURL: openAIBaseURL, key: snippetKey, model: snippetModel
                ),
                displaySnippet: AgentLaunchCommand.hermes(
                    baseURL: openAIBaseURL, key: snippetKeyMasked, model: snippetModel
                )
            ),
        ]
    }
}

/// Process-scoped launch commands shown by the Connect agents surface.
///
/// These deliberately use inline `env` assignments rather than `export`, so
/// copying a command cannot alter the user's shell after the agent exits.
/// Codex additionally receives a throwaway home because its interactive CLI
/// has no top-level `--ignore-user-config` flag (that flag belongs only to the
/// non-interactive `exec` subcommand in Codex 0.146). The temporary home keeps
/// the Rapid provider isolated without rewriting `~/.codex/config.toml`.
enum AgentLaunchCommand {
    static func claude(baseURL: String, key: String, model: String) -> String {
        "env ANTHROPIC_BASE_URL=\(baseURL) ANTHROPIC_API_KEY=\(key) ANTHROPIC_MODEL=\(model) claude"
    }

    static func codex(baseURL: String, key: String, model: String) -> String {
        "env CODEX_HOME=\"$(mktemp -d)\" OPENAI_API_KEY=\(key) codex -m \(model) "
            + "-c 'model_provider=\"rapid-mlx\"' "
            + "-c 'model_providers.rapid-mlx={name=\"Rapid-MLX\",base_url=\"\(baseURL)\",env_key=\"OPENAI_API_KEY\",wire_api=\"responses\"}'"
    }

    static func hermes(baseURL: String, key: String, model: String) -> String {
        "env OPENAI_BASE_URL=\(baseURL) OPENAI_API_KEY=\(key) HERMES_INFERENCE_MODEL=\(model) "
            + "hermes --provider openai-api --ignore-user-config"
    }
}

/// Merges the resolved off-PATH sidecar path into the launch/agent commands
/// the Connect page hands the user. Kept out of any SwiftUI ``View`` so it is
/// not inferred ``@MainActor`` — these are pure string functions callable from
/// synchronous, nonisolated tests.
enum IntegrationLaunchCommand {
    /// Single-quote a shell word so spaces / special characters in an
    /// absolute path (e.g. "Application Support") can't break a pasted
    /// command. Embedded single quotes are escaped by closing, backslash-
    /// escaping, and reopening. Pure string logic — deliberately not a UI type.
    static func shellQuote(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    static func configWriter(id: String, serverURL: String, key: String, model: String, cli: String) -> String {
        "env RAPID_MLX_API_KEY=\(key) \(cli) launch \(id) --server-url \(serverURL) --model \(model)"
    }

    static func adapterGuide(id: String, baseURL: String, model: String, cli: String) -> String {
        "\(cli) agents \(id) --base-url \(baseURL) --model \(model)"
    }
}

/// Applies the page-vs-sheet frame. A plain `if` around `.frame` would
/// change the view's type between branches, so the choice is wrapped in
/// a modifier instead.
private struct ConnectToolsFrame: ViewModifier {
    let fixedSize: Bool

    func body(content: Content) -> some View {
        if fixedSize {
            content.frame(width: 460, height: 560)
        } else {
            content.frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
    }
}

/// A group of rows presented as one card: raised fill, one card radius,
/// one hairline. The rows inside are plain — no nested containers.
private extension View {
    func groupedCard() -> some View {
        self
            .background(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.card, style: .continuous)
                    .fill(RapidTheme.surfaceRaised)
            )
            .overlay(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.card, style: .continuous)
                    .strokeBorder(RapidTheme.hairline, lineWidth: 1)
            )
    }
}

/// One agent's copyable, process-scoped launch command.
private struct ConnectTool: Identifiable {
    let id: String
    let name: String
    let symbol: String
    let blurb: String
    /// The command with the REAL key — placed on the clipboard by Copy, never
    /// rendered on screen.
    let snippet: String
    /// The same command with the key masked — the ONLY form painted on screen,
    /// so a screenshot can't leak the bearer.
    let displaySnippet: String
}

/// One agent row: icon, name, description, a Copy action, and its launch
/// command — all on a shared alignment grid.
///
/// The three changes that flatten this surface:
///
///   1. It is a ROW in a shared card, not its own card. No card-inside-
///      card, and the three tools now read as one scannable list.
///   2. Copy command steps down from `.borderedProminent` (a filled
///      steel-blue block, repeated three times, which read as the most
///      important thing on the page) to a compact outlined secondary
///      with a steel-blue label — the utility action it actually is.
///   3. The snippet sits on ``surfaceCode`` with tighter padding and a
///      smaller mono size, so it reads as inset reference material
///      rather than a second card.
///
/// Every tool, value, and action is preserved exactly.
private struct ConnectToolRow: View {
    let tool: ConnectTool
    /// False until the endpoint, key, and model are all real. Copying a
    /// half-filled snippet is worse than not offering it.
    var isReady: Bool = true
    @State private var copied = false

    /// Fixed icon column. Everything textual in the row starts at
    /// ``iconColumn`` + ``iconGap`` so title, description, and snippet
    /// share one left edge.
    private static let iconColumn: CGFloat = 20
    private static let iconGap: CGFloat = 12
    private static var textInset: CGFloat { iconColumn + iconGap }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header: icon | title | spacer | fixed-width action.
            // ``firstTextBaseline`` rather than centre so the glyph sits
            // on the title's baseline instead of floating between the
            // title and the description below it.
            HStack(alignment: .firstTextBaseline, spacing: Self.iconGap) {
                Image(systemName: tool.symbol)
                    .font(.system(size: 14, weight: .medium))
                    // v1.0.2: amber. These are brand/integration marks
                    // on a page that had drifted entirely neutral;
                    // steel blue is off this surface altogether.
                    .foregroundStyle(RapidTheme.brandPrimaryDeep)
                    .frame(width: Self.iconColumn, alignment: .center)
                    .accessibilityHidden(true)

                Text(tool.name)
                    .font(RapidFont.bodyEmphasis)
                    .lineLimit(1)

                Spacer(minLength: RapidTheme.Space.md)

                Button(action: copy) {
                    Label(copied ? "Copied" : "Copy command",
                          systemImage: copied ? "checkmark" : "doc.on.doc")
                }
                .buttonStyle(RapidSecondaryButtonStyle(
                    utility: true,
                    height: RapidTheme.ControlHeight.small,
                    foreground: copied ? RapidTheme.utilityActionSuccess : nil
                ))
                .disabled(!isReady)
                // Fixed width so all three buttons form a clean right
                // column instead of each sizing to its own label (the
                // "Copied" flip would otherwise resize the button and
                // shift the row).
                .frame(width: 132)
                .help(isReady
                      ? "Copy this agent's one-session launch command"
                      : "Start a model to generate a valid key and launch command.")
                .accessibilityLabel(copied ? "Copied \(tool.name) command" : "Copy \(tool.name) command")
                .accessibilityIdentifier("Launch.Integration.Copy.\(tool.id)")
            }

            // Description and snippet share the title's column.
            VStack(alignment: .leading, spacing: 0) {
                Text(tool.blurb)
                    .font(RapidFont.secondary)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    // Capped so a one-line blurb can't stretch to the
                    // card edge and break after a single trailing word.
                    .frame(maxWidth: 420, alignment: .leading)
                    .padding(.top, RapidTheme.Space.sm)

                // Masked form only — the real key never touches the screen, so
                // a screenshot of this page can't leak the bearer. Copy still
                // puts the real key (``tool.snippet``) on the clipboard.
                Text(tool.displaySnippet)
                    .font(RapidFont.code)
                    .foregroundStyle(.secondary)
                    .lineSpacing(3)
                    .textSelection(.enabled)
                    .padding(.horizontal, RapidTheme.Space.md)
                    .padding(.vertical, RapidTheme.Space.md - 2)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: RapidTheme.Radius.code, style: .continuous)
                            .fill(RapidTheme.surfaceCode)
                    )
                    .padding(.top, RapidTheme.Space.md)
            }
            .padding(.leading, Self.textInset)
        }
        .padding(.horizontal, RapidTheme.Space.xl - 4)
        .padding(.vertical, RapidTheme.Space.lg + 1)
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
    /// Shown instead of the value when there is nothing real to show.
    /// Its presence also disables Copy — an empty clipboard write is a
    /// silent failure the user only discovers in their editor.
    var placeholder: String? = nil
    @State private var reveal = false
    @State private var copied = false

    private var hasValue: Bool { !value.isEmpty }

    private var shown: String {
        guard hasValue else { return placeholder ?? "—" }
        guard masked, !reveal else { return value }
        return String(repeating: "•", count: min(value.count, 16))
    }

    var body: some View {
        HStack(spacing: RapidTheme.Space.sm) {
            Text(label)
                .font(RapidFont.secondary)
                .foregroundStyle(.secondary)
                .frame(width: 132, alignment: .leading)
            // Monospaced is correct here — this is an endpoint / key /
            // model id, one of the four sanctioned mono uses. The
            // not-yet placeholder drops to the prose font and tertiary,
            // so it can't be mistaken for a value worth copying.
            // Values render at full contrast. A not-yet placeholder is
            // one step down — clearly secondary, still comfortably
            // readable. It is NOT `.tertiary`: that made whole rows look
            // switched off when the row's own label and the sentence it
            // carries are both perfectly legitimate information.
            Text(shown)
                .font(hasValue ? RapidFont.code : RapidFont.secondary)
                .foregroundStyle(hasValue ? AnyShapeStyle(.primary) : AnyShapeStyle(.secondary))
                .lineLimit(1)
                .truncationMode(.middle)
                // Always selectable: the two `textSelection` states are
                // different types so they can't share a ternary, and
                // the real guard against pasting a placeholder is the
                // disabled Copy button below, not selection.
                .textSelection(.enabled)
            Spacer(minLength: RapidTheme.Space.xs)
            if masked, hasValue {
                QuietIconButton(
                    symbol: reveal ? "eye.slash" : "eye",
                    label: reveal ? "Hide key" : "Show key",
                    size: RapidTheme.ControlHeight.mini
                ) {
                    reveal.toggle()
                }
                .accessibilityIdentifier("ConnectTools.Reveal.\(label)")
            }
            QuietIconButton(
                symbol: copied ? "checkmark" : "doc.on.doc",
                label: "Copy \(label)",
                help: hasValue
                    ? "Copy \(label)"
                    : "Start a model to generate a valid key and configuration.",
                tint: copied ? RapidTheme.utilityActionSuccess : nil,
                size: RapidTheme.ControlHeight.mini
            ) {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(value, forType: .string)
                copied = true
                Task {
                    try? await Task.sleep(nanoseconds: 1_200_000_000)
                    copied = false
                }
            }
            .disabled(!hasValue)
            .accessibilityIdentifier("ConnectTools.Copy.\(label)")
        }
        .padding(.horizontal, RapidTheme.Space.md)
        .frame(height: RapidTheme.ControlHeight.medium)
    }
}
