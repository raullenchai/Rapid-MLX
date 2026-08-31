import Observation
import SwiftUI

/// A compact keyboard entry point for primary navigation and existing app
/// actions. The first version deliberately uses a fixed command registry so
/// every action follows the same lifecycle as its menu or toolbar equivalent.
struct CommandPaletteView: View {
    let onRun: (CommandPalette.Command) -> Void
    let onDismiss: () -> Void

    @State private var query = ""
    @State private var selectedCommandID: CommandPalette.Command.ID?
    @State private var hoveredCommandID: CommandPalette.Command.ID?
    @FocusState private var queryFieldFocused: Bool

    private var results: [CommandPalette.Command] {
        CommandPalette.filter(CommandPalette.Command.allCases, matching: query)
    }

    var body: some View {
        VStack(spacing: 0) {
            searchHeader
            Divider()
                .overlay(RapidTheme.hairlineStrong)
            commandList
        }
        .background {
            RapidTheme.surfaceOverlay
                .accessibilityElement()
                .accessibilityIdentifier("CommandPalette.Panel")
        }
        .clipShape(RoundedRectangle(cornerRadius: RapidTheme.Radius.card, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.card, style: .continuous)
                .strokeBorder(RapidTheme.hairlineStrong, lineWidth: 1)
        )
        .shadow(color: Color.black.opacity(0.22), radius: 28, x: 0, y: 12)
        .onExitCommand(perform: onDismiss)
        .onMoveCommand(perform: moveSelection)
        .onChange(of: query) { _, _ in
            selectedCommandID = results.first?.id
        }
        .task {
            await Task.yield()
            guard !Task.isCancelled else { return }
            selectedCommandID = results.first?.id
            queryFieldFocused = true
        }
    }

    private var searchHeader: some View {
        HStack(spacing: RapidTheme.Space.sm) {
            Image(systemName: "command")
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(.secondary)
                .accessibilityHidden(true)

            TextField("Type a command", text: $query)
                .textFieldStyle(.plain)
                .font(RapidFont.body)
                .focused($queryFieldFocused)
                .onSubmit { runSelectedCommand() }
                .accessibilityIdentifier("CommandPalette.Field")

            SheetCloseButton(action: onDismiss)
                .accessibilityIdentifier("CommandPalette.Close")
        }
        .padding(.horizontal, RapidTheme.Space.lg)
        .frame(height: 56)
    }

    private var commandList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 1) {
                    ForEach(results) { command in
                        commandRow(command)
                            .id(command.id)
                    }
                    if results.isEmpty {
                        emptyState
                    }
                }
                .padding(RapidTheme.Space.md)
            }
            .scrollIndicators(.never)
            .onChange(of: selectedCommandID) { _, id in
                guard let id else { return }
                proxy.scrollTo(id, anchor: .center)
            }
        }
    }

    private func commandRow(_ command: CommandPalette.Command) -> some View {
        let hovering = hoveredCommandID == command.id
        let selected = selectedCommandID == command.id
        return Button {
            onRun(command)
        } label: {
            HStack(spacing: RapidTheme.Space.md) {
                Image(systemName: command.systemImage)
                    .font(.system(size: 14, weight: .medium))
                    .frame(width: RapidTheme.Layout.iconSlot)
                Text(command.title)
                    .font(selected ? RapidFont.bodyEmphasis : RapidFont.body)
                Spacer(minLength: 0)
            }
            .padding(.horizontal, RapidTheme.Space.md)
            .frame(height: 40)
            .background(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous)
                    .fill(selected ? RapidTheme.selectionFill : (hovering ? RapidTheme.hoverFill : .clear))
            )
            .overlay(alignment: .leading) {
                if selected {
                    Capsule(style: .continuous)
                        .fill(RapidTheme.selectionBar)
                        .frame(
                            width: RapidTheme.Layout.selectionBarWidth,
                            height: RapidTheme.Layout.selectionBarHeight
                        )
                }
            }
            .contentShape(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous)
            )
        }
        .buttonStyle(.plain)
        .foregroundStyle(Color.primary)
        .onHover {
            hoveredCommandID = $0 ? command.id : nil
        }
        .accessibilityAddTraits(selected ? .isSelected : [])
        .accessibilityIdentifier("CommandPalette.Command.\(command.rawValue)")
    }

    private var emptyState: some View {
        VStack(spacing: RapidTheme.Space.sm) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 22, weight: .regular))
                .foregroundStyle(.secondary)
            Text("No commands match")
                .font(RapidFont.body)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, RapidTheme.Space.xxl)
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("CommandPalette.Empty")
    }

    private func runSelectedCommand() {
        guard let id = selectedCommandID ?? results.first?.id,
              let command = results.first(where: { $0.id == id }) else { return }
        onRun(command)
    }

    private func moveSelection(_ direction: MoveCommandDirection) {
        guard direction == .up || direction == .down, !results.isEmpty else { return }
        let currentIndex = selectedCommandID.flatMap { id in
            results.firstIndex(where: { $0.id == id })
        }
        let nextIndex: Int
        switch direction {
        case .up:
            nextIndex = max(0, (currentIndex ?? 1) - 1)
        case .down:
            nextIndex = min(results.count - 1, (currentIndex ?? -1) + 1)
        default:
            return
        }
        selectedCommandID = results[nextIndex].id
    }
}

enum CommandPalette {
    enum Command: String, CaseIterable, Identifiable {
        case newChat
        case searchChats
        case images
        case audio
        case launch
        case settings
        case modelManagement
        case connectors
        case serverLogs
        case exportDiagnostics
        case checkUpdates

        var id: String { rawValue }

        var title: LocalizedStringKey {
            switch self {
            case .newChat: return "New chat"
            case .searchChats: return "Search chats"
            case .images: return "Open Images"
            case .audio: return "Open Audio"
            case .launch: return "Open Launch"
            case .settings: return "Open Settings"
            case .modelManagement: return "Open Model Management"
            case .connectors: return "Open Connectors"
            case .serverLogs: return "Show or hide server logs"
            case .exportDiagnostics: return "Export diagnostics…"
            case .checkUpdates: return "Check for updates"
            }
        }

        var systemImage: String {
            switch self {
            case .newChat: return "square.and.pencil"
            case .searchChats: return "magnifyingglass"
            case .images: return "photo"
            case .audio: return "waveform"
            case .launch: return "paperplane"
            case .settings: return "gearshape"
            case .modelManagement: return "externaldrive.fill"
            case .connectors: return "powerplug.fill"
            case .serverLogs: return "terminal"
            case .exportDiagnostics: return "stethoscope"
            case .checkUpdates: return "arrow.triangle.2.circlepath"
            }
        }

        var keywords: String {
            switch self {
            case .newChat: return "create conversation message chat new"
            case .searchChats: return "find conversation history search chat"
            case .images: return "image generation gallery photo"
            case .audio: return "speech transcription voice audio"
            case .launch: return "connect agents tools launch"
            case .settings: return "preferences configuration settings"
            case .modelManagement: return "models download cache storage"
            case .connectors: return "mcp tools integrations connectors"
            case .serverLogs: return "diagnostics output terminal logs"
            case .exportDiagnostics: return "support report troubleshoot diagnostics"
            case .checkUpdates: return "upgrade version software update"
            }
        }

        var searchTitle: String {
            switch self {
            case .newChat: return String(localized: String.LocalizationValue("New chat"))
            case .searchChats: return String(localized: String.LocalizationValue("Search chats"))
            case .images: return String(localized: String.LocalizationValue("Open Images"))
            case .audio: return String(localized: String.LocalizationValue("Open Audio"))
            case .launch: return String(localized: String.LocalizationValue("Open Launch"))
            case .settings: return String(localized: String.LocalizationValue("Open Settings"))
            case .modelManagement: return String(localized: String.LocalizationValue("Open Model Management"))
            case .connectors: return String(localized: String.LocalizationValue("Open Connectors"))
            case .serverLogs: return String(localized: String.LocalizationValue("Show or hide server logs"))
            case .exportDiagnostics: return String(localized: String.LocalizationValue("Export diagnostics…"))
            case .checkUpdates: return String(localized: String.LocalizationValue("Check for updates"))
            }
        }
    }

    static func filter(
        _ commands: [Command],
        matching query: String
    ) -> [Command] {
        let terms = normalizedTerms(query)
        guard !terms.isEmpty else { return commands }

        return commands.filter { command in
            let haystack = normalizedTerms("\(command.searchTitle) \(command.keywords)")
                .joined(separator: " ")
            return terms.allSatisfy { haystack.contains($0) }
        }
    }

    private static func normalizedTerms(_ value: String) -> [String] {
        value.folding(
            options: [.caseInsensitive, .diacriticInsensitive],
            locale: .current
        )
        .split(whereSeparator: \.isWhitespace)
        .map(String.init)
    }
}

@MainActor
@Observable
final class CommandPaletteRequestCoordinator {
    private(set) var requestID: UInt = 0
    private(set) var lastConsumedRequestID: UInt = 0

    func open() {
        requestID &+= 1
    }

    func consume(_ requestID: UInt) -> Bool {
        guard requestID > lastConsumedRequestID else { return false }
        lastConsumedRequestID = requestID
        return true
    }
}
