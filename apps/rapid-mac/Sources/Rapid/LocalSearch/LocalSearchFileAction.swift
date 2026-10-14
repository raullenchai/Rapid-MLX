import Darwin
import Foundation
import UniformTypeIdentifiers

/// Only known document/media types are eligible for the default viewer.
/// Launchable, indirect, and unknown types always stay in Finder.
enum LocalSearchFileAction {
    enum Action { case open, reveal, unavailable }

    struct Identity: Equatable {
        let device: dev_t
        let inode: ino_t
        let type: mode_t
    }
    struct Classification {
        let action: Action
        let identity: Identity?
    }

    private static func metadata(for url: URL) -> stat? {
        var info = stat()
        guard url.isFileURL, url.withUnsafeFileSystemRepresentation({ path in
            path.map { lstat($0, &info) } ?? -1
        }) == 0 else { return nil }
        return info
    }

    private static func identity(_ info: stat) -> Identity {
        Identity(device: info.st_dev, inode: info.st_ino, type: info.st_mode & mode_t(S_IFMT))
    }

    static func classify(type: UTType?, executable: Bool = false,
                         indirect: Bool = false, directory: Bool = false) -> Action {
        guard !executable, !indirect, !directory, let type else { return .reveal }
        let launchable: [UTType] = [.application, .applicationBundle, .executable, .script,
                                    .shellScript, .diskImage, .aliasFile, .symbolicLink, .bundle]
        let launchableIDs = ["com.apple.installer-package", "com.apple.installer-meta-package",
                             "com.apple.shortcuts.workflow", "com.apple.automator-workflow",
                             "com.apple.applescript.script", "com.apple.applescript.script-bundle"]
        if launchable.contains(where: { type.conforms(to: $0) }) || launchableIDs.contains(where: {
            type.identifier == $0 || UTType($0).map { type.conforms(to: $0) } == true
        }) { return .reveal }
        // Include common concrete types too: a restricted process can have
        // type identities without access to the system's ancestor declarations.
        // Unknown identities still fail closed; this is never suffix matching.
        let documents: [UTType] = [.text, .image, .audiovisualContent, .pdf, .rtf,
                                   .spreadsheet, .presentation, .compositeContent,
                                   .plainText, .utf8PlainText, .utf16PlainText,
                                   .png, .jpeg, .heic, .tiff, .gif,
                                   .mpeg4Movie, .quickTimeMovie, .mp3, .wav]
        return documents.contains { type.conforms(to: $0) } ? .open : .reveal
    }

    static func action(for url: URL) -> Action {
        classification(for: url).action
    }

    private static func documentAction(for url: URL) -> Action {
        // A fresh URL avoids cached resource values during reclassification.
        let freshURL = URL(fileURLWithPath: url.path)
        guard let values = try? freshURL.resourceValues(forKeys: [.contentTypeKey, .isAliasFileKey, .isRegularFileKey, .isDirectoryKey]) else {
            return .reveal
        }
        return classify(type: values.contentType, indirect: values.isAliasFile == true,
                        directory: values.isDirectory == true)
    }

    // Inject only the system type/alias lookup; identity and permission checks
    // always use the real filesystem, including in restricted test processes.
    static func classification(for url: URL,
                               classifyDocument: (URL) -> Action = documentAction) -> Classification {
        guard let info = metadata(for: url) else { return Classification(action: .unavailable, identity: nil) }
        let captured = identity(info)
        // lstat never follows a link. Only non-executable regular files qualify.
        guard captured.type == mode_t(S_IFREG), info.st_mode & 0o111 == 0 else {
            return Classification(action: .reveal, identity: captured)
        }
        let action = classifyDocument(url)
        guard let after = metadata(for: url) else { return Classification(action: .unavailable, identity: nil) }
        guard identity(after) == captured, after.st_mode & 0o111 == 0 else {
            return Classification(action: .reveal, identity: captured)
        }
        return Classification(action: action, identity: captured)
    }

    /// Revalidate immediately before handing the path to the viewer. Callbacks
    /// keep filesystem regression tests from launching applications or Finder.
    @discardableResult
    static func perform(for url: URL, classified: Classification,
                        classifyDocument: (URL) -> Action = documentAction,
                        open: (URL) -> Bool, reveal: (URL) -> Void) -> Action {
        let current = classification(for: url, classifyDocument: classifyDocument)
        guard current.action != .unavailable else { return .unavailable }
        guard classified.action == .open, current.action == .open,
              let captured = classified.identity, current.identity == captured else {
            reveal(url)
            return .reveal
        }
        return open(url) ? .open : .unavailable
    }
}
