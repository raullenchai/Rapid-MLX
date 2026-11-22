import AppKit
import Foundation
import UniformTypeIdentifiers

/// The panel toolbar's side effects, apart from the views so they can be
/// tested without one.
@MainActor
enum ArtifactPanelActions {

    enum SaveFormat {
        case source
        case png
    }

    static func copySource(of item: ArtifactItem) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(item.source, forType: .string)
    }

    /// A file name for the save panel: the title with anything a file system
    /// or a path separator would object to replaced, never empty.
    static func suggestedFileName(for item: ArtifactItem, format: SaveFormat = .source) -> String {
        let unsafe = CharacterSet(charactersIn: "/\\:*?\"<>|").union(.controlCharacters)
        let cleaned = item.title.unicodeScalars
            .map { unsafe.contains($0) ? "-" : String($0) }
            .joined()
            .trimmingCharacters(in: .whitespaces.union(CharacterSet(charactersIn: ".")))
        let base = cleaned.isEmpty ? "artifact" : String(cleaned.prefix(60))
        return "\(base).\(format == .png ? "png" : item.fileExtension)"
    }

    /// Ask where to save, then write. Returns an error message, or nil on
    /// success or cancel.
    static func save(_ item: ArtifactItem, format: SaveFormat = .source, png: Data? = nil) -> String? {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = suggestedFileName(for: item, format: format)
        panel.canCreateDirectories = true
        if let type = UTType(filenameExtension: format == .png ? "png" : item.fileExtension) {
            panel.allowedContentTypes = [type]
        }
        guard panel.runModal() == .OK, let url = panel.url else { return nil }
        let data: Data
        switch format {
        case .source: data = Data(item.source.utf8)
        case .png:
            guard let png else { return String(localized: "The image isn't ready to save yet.") }
            data = png
        }
        do {
            try data.write(to: url, options: .atomic)
            return nil
        } catch {
            return String(localized: "Couldn't save the file: \(error.localizedDescription)")
        }
    }

    /// PNG bytes for a rendered diagram.
    static func pngData(for image: NSImage) -> Data? {
        guard let tiff = image.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiff) else { return nil }
        return bitmap.representation(using: .png, properties: [:])
    }

    // MARK: Open in browser

    static let temporaryDirectoryName = "rapid-artifacts"
    private static let temporaryFileLifetime: TimeInterval = 24 * 60 * 60

    /// Write the page where only this user can read it, and drop leftovers
    /// from earlier opens. The file is handed to a browser, so it is the one
    /// place an artifact leaves the sandbox.
    static func writeTemporaryHTML(
        _ item: ArtifactItem, in base: URL = FileManager.default.temporaryDirectory
    ) throws -> URL {
        let directory = base.appendingPathComponent(temporaryDirectoryName, isDirectory: true)
        let manager = FileManager.default
        try manager.createDirectory(
            at: directory, withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700]
        )
        let cutoff = Date().addingTimeInterval(-temporaryFileLifetime)
        for url in (try? manager.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.contentModificationDateKey]
        )) ?? [] {
            let modified = (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?
                .contentModificationDate
            if let modified, modified < cutoff { try? manager.removeItem(at: url) }
        }
        let file = directory.appendingPathComponent("\(UUID().uuidString).html")
        guard manager.createFile(
            atPath: file.path, contents: Data(item.source.utf8),
            attributes: [.posixPermissions: 0o600]
        ) else { throw CocoaError(.fileWriteUnknown) }
        return file
    }

    /// Confirm, then open the page in the default browser. Returns an error
    /// message, or nil on success or cancel.
    static func openInBrowser(_ item: ArtifactItem) -> String? {
        let alert = NSAlert()
        alert.messageText = String(localized: "Open this page in your browser?")
        alert.informativeText = String(localized:
            "In the browser the page leaves Rapid's sandbox: its scripts can reach the internet and your browser's data. Only continue if you trust it."
        )
        alert.addButton(withTitle: String(localized: "Cancel"))
        alert.addButton(withTitle: String(localized: "Open in browser"))
        guard alert.runModal() == .alertSecondButtonReturn else { return nil }
        do {
            let file = try writeTemporaryHTML(item)
            let workspace = NSWorkspace.shared
            if let probe = URL(string: "https://localhost"),
               let browser = workspace.urlForApplication(toOpen: probe) {
                workspace.open(
                    [file], withApplicationAt: browser,
                    configuration: NSWorkspace.OpenConfiguration()
                )
            } else {
                workspace.open(file)
            }
            return nil
        } catch {
            return String(localized: "Couldn't open the page: \(error.localizedDescription)")
        }
    }
}
