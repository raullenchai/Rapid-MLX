import SwiftUI

/// Cleans up a raw role string for display (the server wire value is a compact
/// dashed identifier such as ``speech-input`` / ``image-generation``). Returns
/// nil when the string is unknown/empty so the UI can skip it rather than show
/// a raw token.
@MainActor
enum ResidentRoleLabel {
    static func displayName(for role: String) -> String? {
        switch role {
        case "assistant": return "Conversation"
        case "speech-input", "speech_input": return "Speech Input"
        case "speech-output", "speech_output": return "Speech Output"
        case "alignment": return "Forced Alignment"
        case "image-generation", "image_generation": return "Image Generation"
        case "video-generation", "video_generation": return "Video Generation"
        default: return nil
        }
    }
}

/// The user-facing conflict presented when a voice workflow cannot be admitted.
///
/// Desktop consumes the server's typed capacity authority verbatim: it never
/// estimates memory or derives roles from aliases. Memory quantities are shown
/// only when the server reported them; recovery actions are offered only when
/// the server declared them valid. Destructive unload is never implied by the
/// originating action (e.g. a microphone press) — it is always an explicit,
/// reversible button here.
@MainActor
struct ResidentRoleConflictSheet: View {
    let conflict: ResidentRoleConflict
    let requestedModelAlias: String
    let onAction: (ResidentRecoveryAction) -> Void
    let onCancel: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.lg) {
            header

            ScrollView {
                VStack(alignment: .leading, spacing: RapidTheme.Space.lg) {
                    reasonSection
                    if !conflict.residentRoles.isEmpty {
                        residentRolesSection
                    }
                    Divider().overlay(RapidTheme.hairline)
                    actionsSection
                }
            }

            Divider().overlay(RapidTheme.hairline)
            cancelButtonRow
        }
        .padding(RapidTheme.Space.lg)
        .frame(width: 460)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.xs) {
            Label {
                Text("This voice workflow can't start right now")
                    .font(.headline)
            } icon: {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.yellow)
            }
            Text(conflict.message)
                .font(RapidFont.body)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// The requested role/model and the server's memory quantities.
    private var reasonSection: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
            SectionHeader("What was requested")
            VStack(alignment: .leading, spacing: RapidTheme.Space.xxs) {
                Text(requestedModelAlias).font(RapidFont.body)
                if let requestedRole = conflict.requestedRole,
                   let label = ResidentRoleLabel.displayName(for: requestedRole) {
                    Text("Role: \(label)")
                        .font(RapidFont.caption)
                        .foregroundStyle(.secondary)
                }
                memoryLine(
                    "Memory requested",
                    gib: conflict.requestedGib
                )
                memoryLine(
                    "In use",
                    gib: conflict.usedGib
                )
                memoryLine(
                    "Budget",
                    gib: conflict.limitGib
                )
            }
            .padding(RapidTheme.Space.sm)
            .frame(maxWidth: .infinity, alignment: .leading)
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

    @ViewBuilder
    private func memoryLine(_ label: String, gib: Double?) -> some View {
        if let gib {
            HStack {
                Text(label).font(RapidFont.caption)
                Spacer()
                Text(gibString(gib)).font(RapidFont.caption.monospacedDigit())
            }
        }
    }

    /// The conflicting runtime roles (server-declared), in user-readable units.
    private var residentRolesSection: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
            SectionHeader("Currently loaded and conflicting")
            ForEach(conflict.residentRoles) { role in
                HStack(spacing: RapidTheme.Space.sm) {
                    Image(systemName: "cube.fill")
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(roleLabel(role.role))
                            .font(RapidFont.body)
                        if let modelID = role.modelID, !modelID.isEmpty {
                            Text(modelID)
                                .font(RapidFont.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                        }
                    }
                    Spacer(minLength: RapidTheme.Space.md)
                    if let gib = role.reservedGib {
                        Text(gibString(gib)).font(RapidFont.caption.monospacedDigit())
                    }
                }
                .accessibilityElement(children: .combine)
                .accessibilityLabel(roleAccessibilityLabel(role))
            }
            .padding(.vertical, RapidTheme.Space.xxs)
        }
    }

    private func roleLabel(_ role: String) -> String {
        ResidentRoleLabel.displayName(for: role) ?? role
    }

    private func roleAccessibilityLabel(_ role: ResidentRoleStatus) -> String {
        var parts = [roleLabel(role.role)]
        if let modelID = role.modelID, !modelID.isEmpty { parts.append(modelID) }
        if let gib = role.reservedGib {
            parts.append("\(gibString(gib)) reserved")
        }
        return parts.joined(separator: ", ")
    }

    /// Only the recovery actions the server declared valid become buttons.
    /// Each destructive/unload action is an explicit user choice.
    private var actionsSection: some View {
        let actions = conflict.scopedRecoveryActions
        return VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
            SectionHeader("How to proceed")
            if actions.isEmpty {
                Text("No automatic fixes are available. Choose Cancel and free memory another way.")
                    .font(RapidFont.body)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(actions, id: \.self) { action in
                    Button {
                        onAction(action)
                    } label: {
                        Label(action.title, systemImage: actionSystemImage(action))
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.rapidPrimary)
                    .accessibilityIdentifier("RoleConflict.Action.\(action.rawValue)")
                    .accessibilityHint(action.hint)
                }
            }
        }
    }

    private func actionSystemImage(_ action: ResidentRecoveryAction) -> String {
        switch action {
        case .selectSmallerSpeechInput: return "arrow.down.circle"
        case .stopSpeechOutput: return "speaker.slash"
        case .unloadAssistant: return "eject.fill"
        }
    }

    private var cancelButtonRow: some View {
        HStack {
            Spacer()
            Button("Cancel", role: .cancel) {
                onCancel()
            }
            .keyboardShortcut(.cancelAction)
            .accessibilityIdentifier("RoleConflict.Cancel")
        }
    }

    private func gibString(_ gib: Double) -> String {
        if !gib.isFinite { return "—" }
        if gib >= 9.5 { return "\(Int(gib.rounded())) GB" }
        return String(format: "%.1f GB", gib)
    }
}
