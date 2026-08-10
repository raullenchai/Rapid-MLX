import AppKit
import SwiftUI

/// Window scene that drives the in-app DMG install flow. Opened from
/// the MenuBarExtra "Update available — vX.Y.Z" item.
///
/// Three states map to three sub-views:
///
///   * ``Installer.Stage.idle`` (and the user hasn't kicked off yet)
///     → "release notes + Install / Open release page" picker.
///   * Any running stage → determinate ``ProgressView`` for the
///     download fraction plus a status label for the rest of the
///     pipeline; close button disabled to avoid the user thinking
///     they can cancel mid-stream (we don't have a cancel surface
///     wired yet — install is short enough that adding one is
///     reasonable follow-up, not blocking).
///   * ``.failed`` → red banner, the original picker re-enabled so
///     "Open release page" stays the always-works fallback.
///
/// Modelled after LM Studio's update sheet: single window, no
/// surrounding chrome, the in-app path is primary and the browser
/// fallback is one click away on every state.
struct UpdateInstallView: View {
    enum Presentation: Equatable {
        case update(UpdateChecker.Release)
        case checking
        case upToDate
        case unavailable(String?)
    }

    @Environment(UpdateChecker.self) private var updater
    @Environment(Installer.self) private var installer
    @Environment(\.dismissWindow) private var dismissWindow

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            header
            Divider()
            ScrollView {
                releaseNotes
                    .padding(.vertical, 4)
            }
            .frame(minHeight: 80, maxHeight: 180)
            Divider()
            footer
        }
        .padding(20)
        .frame(minWidth: 460, idealWidth: 480, maxWidth: 560,
               minHeight: 320, idealHeight: 360, maxHeight: 520)
    }

    private var header: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "arrow.down.app.fill")
                .font(.system(size: 28, weight: .regular))
                .foregroundStyle(.tint)
            VStack(alignment: .leading, spacing: 2) {
                Text(headerTitle)
                    .font(.headline)
                    .accessibilityIdentifier("UpdateInstall.Title")
                if case .update(let release) = presentation {
                    Text("Rapid-MLX v\(release.version) — you have v\(updater.currentVersion)")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                } else if presentation == .upToDate {
                    Text("You're already on the latest version (v\(updater.currentVersion)).")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                } else if presentation == .checking {
                    Text("Checking the update channel…")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                } else {
                    Text("Update information isn't available right now.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
        }
    }

    @ViewBuilder
    private var releaseNotes: some View {
        if case .update(let release) = presentation,
           !release.notes.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            // Render the CHANGELOG Markdown properly instead of dumping
            // raw ``##`` / ``-`` / ``**`` characters (pre-v0.8.18 the
            // bare ``Text(release.notes)`` showed the literal source).
            ReleaseNotesMarkdown(raw: release.notes)
        } else if case .update = presentation {
            Text("No release notes provided.")
                .font(.callout)
                .foregroundStyle(.secondary)
        } else if presentation == .upToDate {
            Text("Rapid-MLX is up to date. We'll let you know when a newer version is ready.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .accessibilityIdentifier("UpdateInstall.UpToDate")
        } else if presentation == .checking {
            ProgressView("Checking for updates…")
                .controlSize(.small)
        } else if case .unavailable(let message) = presentation {
            Text(message ?? "Check your connection and try again.")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var footer: some View {
        switch installer.stage {
        case .idle:
            idleFooter
        case .downloading(let progress):
            let displayProgress = Self.displayProgress(progress)
            progressFooter(
                label: "Downloading update… (\(Int(displayProgress * 100))%)",
                progress: displayProgress
            )
        case .verifying:
            progressFooter(label: "Verifying signature…", progress: nil)
        case .installing:
            progressFooter(label: "Installing update…", progress: nil)
        case .relaunching:
            progressFooter(label: "Restarting Rapid-MLX…", progress: nil)
        case .failed(let message):
            failedFooter(message: message)
        }
    }

    private var idleFooter: some View {
        Group {
            if case .update = presentation {
                updateFooter
            } else {
                HStack {
                    Spacer()
                    Button("Close") { dismissWindow(id: "update-install") }
                        .keyboardShortcut(.cancelAction)
                        .accessibilityIdentifier("UpdateInstall.Close")
                    Button(updater.checking ? "Checking…" : "Check Again") {
                        Task { await updater.check() }
                    }
                    .disabled(updater.checking)
                    .accessibilityIdentifier("UpdateInstall.CheckAgain")
                }
            }
        }
    }

    private var updateFooter: some View {
        VStack(alignment: .trailing, spacing: 8) {
            HStack {
                Button("Open release page") { openReleasePage() }
                    .buttonStyle(.bordered)
                Spacer()
                Button("Later") { dismissWindow(id: "update-install") }
                    .keyboardShortcut(.cancelAction)
                Button {
                    startInstall()
                } label: {
                    Text(canInstall ? "Install and Restart" : "Install (DMG not available)")
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(!canInstall)
            }
            if !canInstall {
                Text("This release doesn't ship a DMG yet — use the release page link to download manually.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .trailing)
            }
        }
    }

    private func progressFooter(label: String, progress: Double?) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            if let progress {
                ProgressView(value: Self.displayProgress(progress))
            } else {
                ProgressView()
                    .progressViewStyle(.linear)
            }
            HStack {
                Text(label)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Open release page") { openReleasePage() }
                    .buttonStyle(.link)
            }
        }
    }

    static func displayProgress(_ progress: Double) -> Double {
        guard progress.isFinite else { return 0 }
        return min(max(progress, 0), 1)
    }

    static func resolvePresentation(
        availableUpdate: UpdateChecker.Release?,
        latest: UpdateChecker.Release?,
        checking: Bool,
        lastError: String?
    ) -> Presentation {
        if let availableUpdate { return .update(availableUpdate) }
        if checking { return .checking }
        if latest != nil { return .upToDate }
        return .unavailable(lastError)
    }

    private var presentation: Presentation {
        Self.resolvePresentation(
            availableUpdate: updater.availableUpdate,
            latest: updater.latest,
            checking: updater.checking,
            lastError: updater.lastError
        )
    }

    private var headerTitle: String {
        switch presentation {
        case .update: "A new version of Rapid-MLX is available"
        case .checking: "Checking for updates"
        case .upToDate: "Rapid-MLX is up to date"
        case .unavailable: "Unable to check for updates"
        }
    }

    private func failedFooter(message: String) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                Text(message)
                    .font(.callout)
                    .textSelection(.enabled)
            }
            HStack {
                Button("Open release page") { openReleasePage() }
                    .buttonStyle(.bordered)
                Spacer()
                Button("Try again") {
                    installer.reset()
                    startInstall()
                }
                .buttonStyle(.borderedProminent)
                .disabled(!canInstall)
            }
        }
    }

    private var canInstall: Bool {
        guard let release = updater.availableUpdate else { return false }
        guard let dmg = release.dmgURL, !dmg.isEmpty else { return false }
        return URL(string: dmg) != nil
    }

    private func startInstall() {
        guard let release = updater.availableUpdate,
              let dmgString = release.dmgURL,
              let dmgURL = URL(string: dmgString) else {
            return
        }
        Task { @MainActor in
            await installer.install(from: dmgURL)
        }
    }

    private func openReleasePage() {
        guard let release = updater.availableUpdate else { return }
        guard let url = URL(string: release.htmlURL),
              let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
              components.scheme?.lowercased() == "https",
              let host = components.host?.lowercased(),
              updateReleaseHostAllowlist.contains(host) else {
            return
        }
        NSWorkspace.shared.open(url)
    }
}
