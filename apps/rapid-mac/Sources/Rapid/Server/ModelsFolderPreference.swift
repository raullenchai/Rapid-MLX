import Darwin
import Foundation

/// User preference: an explicit folder where Rapid keeps the models it
/// downloads, overriding the default internal location.
///
/// ## Why this exists (issue #503, @LewnWorx)
///
/// A user with a large shared model collection on a dedicated external
/// drive (280 GB on a 4 TB Thunderbolt NVMe, nothing on the internal
/// disk) wants Rapid to download + load from THAT folder instead of
/// re-filling the internal drive. The engine already honours an
/// explicit models directory end-to-end; this preference is the
/// desktop-side control that feeds it, plus the single source of truth
/// every app-side disk view reads so "what the app shows" matches
/// "where the engine actually reads and writes".
///
/// ## Ownership + precedence
///
///   * The absolute path the user picked in Settings is stored under
///     ``storageKey`` in ``UserDefaults`` (namespaced ``rapid.models.*``
///     consistent with ``rapid.serve.lastAlias`` /
///     ``rapid.window.hideDockChoice``).
///   * ``validatedOverrideURL`` returns a URL ONLY when the stored path
///     is present AND currently resolves to a real directory. So an
///     unplugged external drive transparently resolves to ``nil`` and
///     every caller falls back to the default location instead of
///     failing a model load (#503 point 4 — non-fatal, no crash).
///   * All app-side cache resolvers (``BundledModel.userHFCacheURL``,
///     ``ModelDeletion``, the download byte monitors, ``ModelCatalog``,
///     ``DownloadManager``) and the spawned engine's models-directory
///     env read from the SAME validated override, so the app's disk
///     view can never drift from where the engine operates.
///
/// ## Testability
///
/// Every accessor takes ``defaults`` / ``fileManager`` so unit tests
/// drive the validity + unplugged-drive fallback branches without
/// touching the real ``UserDefaults.standard`` or the live filesystem.
enum ModelsFolderPreference {
    typealias DirectoryProbe = @Sendable (URL) throws -> Bool

    /// Unlike fileExists, this preserves permission errors from stat().
    static func isDirectory(_ url: URL) throws -> Bool {
        var info = stat()
        guard stat(url.path, &info) == 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        return info.st_mode & S_IFMT == S_IFDIR
    }

    /// Discovery retains the missing/non-directory fallback policy, but must
    /// not silently scan a different cache when the saved folder is denied.
    /// Call only inside the discovery worker: even stat() can await consent.
    static func discoveryOverrideURL(
        _ requested: URL?,
        directoryProbe: DirectoryProbe = isDirectory
    ) throws -> URL? {
        guard let requested else { return nil }
        do {
            return try directoryProbe(requested) ? requested : nil
        } catch {
            let error = error as NSError
            if error.domain == NSPOSIXErrorDomain,
               [Int(ENOENT), Int(ENOTDIR)].contains(error.code) { return nil }
            if error.domain == NSCocoaErrorDomain,
               [CocoaError.fileNoSuchFile.rawValue, CocoaError.fileReadNoSuchFile.rawValue].contains(error.code) {
                return nil
            }
            throw error
        }
    }

    /// UserDefaults key holding the user's chosen models folder as an
    /// absolute path. Absent / blank means "use the default location".
    static let storageKey = "rapid.models.folderOverride"

    /// The raw stored absolute path, trimmed. ``nil`` when unset or
    /// blank — a blank string is treated as "no override" so a cleared
    /// field never manufactures a bogus root.
    static func storedPath(defaults: UserDefaults = .standard) -> String? {
        guard let raw = defaults.string(forKey: storageKey) else { return nil }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    /// Persist (``nil`` / blank clears) the chosen folder. Trimmed on
    /// the way in so a stray trailing space can't defeat the
    /// ``storedPath`` blank check on the way out.
    static func setStoredPath(_ path: String?, defaults: UserDefaults = .standard) {
        let trimmed = path?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if trimmed.isEmpty {
            defaults.removeObject(forKey: storageKey)
        } else {
            defaults.set(trimmed, forKey: storageKey)
        }
    }

    /// ``true`` when the user has chosen a custom folder — regardless of
    /// whether it currently resolves. Drives the "Use default" reset
    /// affordance and the unavailable-drive warning in Settings.
    static func hasCustomFolder(defaults: UserDefaults = .standard) -> Bool {
        storedPath(defaults: defaults) != nil
    }

    /// The stored folder as a validated directory URL, or ``nil`` when
    /// no folder is set OR the folder is not currently a reachable
    /// directory (external drive unplugged, path deleted). Callers treat
    /// ``nil`` as "use the default location".
    static func validatedOverrideURL(
        defaults: UserDefaults = .standard,
        fileManager: FileManager = .default
    ) -> URL? {
        validate(path: storedPath(defaults: defaults), fileManager: fileManager)
    }

    /// ``true`` when a custom folder IS set but does NOT currently
    /// resolve to a directory — i.e. the app has fallen back to the
    /// default location and should tell the user why (the drive is
    /// probably unplugged). ``false`` in the healthy case and when no
    /// custom folder is set at all.
    static func customFolderUnavailable(
        defaults: UserDefaults = .standard,
        fileManager: FileManager = .default
    ) -> Bool {
        guard storedPath(defaults: defaults) != nil else { return false }
        return validatedOverrideURL(defaults: defaults, fileManager: fileManager) == nil
    }

    /// Pure validator seam. Returns the path as a directory URL only
    /// when it is an absolute path pointing at a real directory.
    /// Exposed so tests pin the unplugged-drive fallback without
    /// standing up ``UserDefaults``.
    static func validate(path: String?, fileManager: FileManager = .default) -> URL? {
        guard let path, path.hasPrefix("/") else { return nil }
        var isDir: ObjCBool = false
        guard fileManager.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue else {
            return nil
        }
        return URL(fileURLWithPath: path, isDirectory: true)
    }
}
