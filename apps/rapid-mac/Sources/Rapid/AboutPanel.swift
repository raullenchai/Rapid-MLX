import AppKit
import SwiftUI

/// A clean, branded About window for Rapid-MLX.
///
/// Replaces the bare system ``orderFrontStandardAboutPanel`` (which
/// surfaced the internal ``rapid-mlx`` binary path + an out-of-date
/// repo link — jargon a user shouldn't see). Bug-report diagnostics
/// now live behind Settings → Diagnostics ("Export diagnostics…"), so
/// the About window can stay a simple, on-brand credit: mark, version,
/// one-line what-it-is, and the public links.
enum AboutPanel {
    struct EngineIdentity: Equatable, Sendable {
        let version: String?
        let source: ServerLocator.ResolvedSource
        let path: String

        var summary: String {
            let versionLabel = version.map { "Engine \($0)" } ?? "Engine version unknown"
            return "\(versionLabel) · \(source.displayLabel)"
        }

        var isOverride: Bool {
            source == .runtimeOverride || source == .rapidBin
        }
    }

    private static let website = "https://rapidmlx.com"
    private static let repoURL = "https://github.com/raullenchai/Rapid-MLX"
    /// The policy in the repository, not `rapidmlx.com/privacy` — that page
    /// has never been published and 404s, so the About window's "Privacy"
    /// link opened nothing. `apps/rapid-mac/PRIVACY.md` is the real,
    /// maintained document (and the same one Settings → Privacy now opens).
    /// Point this back at the website once the page exists.
    /// ``RepositoryLinkTargetsTests`` pins the path.
    private static let privacyURL =
        "https://github.com/raullenchai/Rapid-MLX/blob/main/apps/rapid-mac/PRIVACY.md"

    /// Retained so the window isn't deallocated the moment ``show``
    /// returns (an unheld ``NSWindow`` closes itself).
    @MainActor private static var window: NSWindow?

    @MainActor
    static func show(server: ServerManager) {
        NSApplication.shared.activate(ignoringOtherApps: true)
        if let existing = window {
            existing.makeKeyAndOrderFront(nil)
            existing.center()
            return
        }
        let view = AboutView(
            version: bundleShortVersion(),
            build: bundleBuildNumber(),
            engine: engineIdentity(resolution: server.binaryResolution),
            website: website,
            repoURL: repoURL,
            privacyURL: privacyURL
        )
        let host = NSHostingController(rootView: view)
        let win = NSWindow(contentViewController: host)
        win.styleMask = [.titled, .closable, .fullSizeContentView]
        win.titlebarAppearsTransparent = true
        win.titleVisibility = .hidden
        win.isMovableByWindowBackground = true
        win.title = "About Rapid-MLX"
        win.setContentSize(NSSize(width: 360, height: 340))
        win.center()
        win.isReleasedWhenClosed = false
        window = win
        win.makeKeyAndOrderFront(nil)
    }

    // MARK: - Version helpers

    /// Reads ``<resourceURL>/rapid-mlx/VERSION`` written by
    /// ``scripts/build.sh`` at .app build time (the bundled sidecar's
    /// ``git describe``). ``nil`` for ``SKIP_SIDECAR=1`` dev builds.
    static func bundledRapidMlxVersion(
        resourceURL: URL? = Bundle.main.resourceURL
    ) -> String? {
        guard let resourceURL else { return nil }
        let versionFile = resourceURL.appendingPathComponent("rapid-mlx/VERSION")
        guard let data = try? Data(contentsOf: versionFile),
              let raw = String(data: data, encoding: .utf8) else {
            return nil
        }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    static func bundleShortVersion() -> String {
        if let v = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String,
           !v.isEmpty {
            return v
        }
        return "0.0.0"
    }

    static func bundleBuildNumber() -> String? {
        Bundle.main.infoDictionary?["CFBundleVersion"] as? String
    }

    /// Describe the binary the server will actually spawn, rather than the
    /// sidecar merely shipped inside this app bundle. A runtime override can
    /// legitimately win version selection; surfacing that fact prevents a
    /// dogfood session from silently attributing its behaviour to the wrong
    /// engine (#1712).
    static func engineIdentity(
        resolution: ServerLocator.Resolution?
    ) -> EngineIdentity? {
        guard let resolution else { return nil }
        return EngineIdentity(
            version: resolution.version,
            source: resolution.source,
            path: resolution.binary.path
        )
    }
}

/// The branded About content.
private struct AboutView: View {
    let version: String
    let build: String?
    let engine: AboutPanel.EngineIdentity?
    let website: String
    let repoURL: String
    let privacyURL: String

    private var versionLine: String {
        if let build, !build.isEmpty, build != version {
            return "Version \(version) (\(build))"
        }
        return "Version \(version)"
    }

    var body: some View {
        VStack(spacing: 14) {
            ZStack {
                Circle()
                    .fill(RapidTheme.brandAmberTint)
                    .frame(width: 64, height: 64)
                Image(systemName: "bolt.fill")
                    .font(.system(size: 30, weight: .semibold))
                    .foregroundStyle(RapidTheme.brandAmber)
            }
            .padding(.top, 8)

            Text("Rapid-MLX")
                .font(.title2.weight(.bold))

            Text(versionLine)
                .font(.callout)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)

            if let engine {
                HStack(spacing: 5) {
                    if engine.isOverride {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                    }
                    Text(engine.summary)
                }
                .font(.caption.weight(engine.isOverride ? .semibold : .regular))
                .foregroundStyle(engine.isOverride ? .primary : .secondary)
                .help(engine.path)
                .accessibilityLabel("\(engine.summary). Path: \(engine.path)")
                .textSelection(.enabled)
            }

            Text("Fast, private AI that runs on your Mac.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.horizontal, 8)

            HStack(spacing: 8) {
                aboutLink("rapidmlx.com", website)
                Text("·").foregroundStyle(.tertiary)
                aboutLink("GitHub", repoURL)
                Text("·").foregroundStyle(.tertiary)
                aboutLink("Privacy", privacyURL)
            }
            .font(.callout)
            .padding(.top, 2)

            Spacer(minLength: 0)

            Text("© 2026 · Runs 100% on your Mac")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .padding(24)
        .frame(width: 360, height: 340)
        .background(RapidTheme.canvas)
    }

    private func aboutLink(_ label: String, _ urlString: String) -> some View {
        Group {
            if let url = URL(string: urlString) {
                Link(label, destination: url)
                    .tint(RapidTheme.brand)
                    .accessibilityIdentifier("About.Link.\(label)")
            } else {
                Text(label)
            }
        }
    }
}
