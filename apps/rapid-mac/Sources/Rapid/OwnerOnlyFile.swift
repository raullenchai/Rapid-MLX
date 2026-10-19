import Darwin
import Foundation

/// Atomic replacement whose temporary and published bytes are owner-only.
enum OwnerOnlyFile {
    static func writeAtomically(
        _ data: Data,
        to url: URL,
        beforeWrite: (_ temporaryPath: String, _ descriptor: Int32) throws -> Void = { _, _ in }
    ) throws {
        let directory = url.deletingLastPathComponent()
        let prefix = directory.appendingPathComponent(".\(url.lastPathComponent).tmp.XXXXXX")
        var template = Array(prefix.path.utf8CString)
        // Close-on-exec so a concurrently spawned child cannot inherit the descriptor.
        let descriptor = mkostemp(&template, O_CLOEXEC)
        guard descriptor >= 0 else { throw posixError() }
        let temporaryPath = String(
            decoding: template.prefix(while: { $0 != 0 }).map { UInt8(bitPattern: $0) },
            as: UTF8.self
        )
        var descriptorIsOpen = true
        var shouldRemoveTemporary = true
        defer {
            if descriptorIsOpen { Darwin.close(descriptor) }
            if shouldRemoveTemporary { _ = unlink(temporaryPath) }
        }

        try beforeWrite(temporaryPath, descriptor)

        try data.withUnsafeBytes { buffer in
            guard let base = buffer.baseAddress else { return }
            var offset = 0
            while offset < buffer.count {
                let count = Darwin.write(descriptor, base.advanced(by: offset), buffer.count - offset)
                if count < 0 && errno == EINTR { continue }
                guard count >= 0 else { throw posixError() }
                guard count > 0 else { throw POSIXError(.EIO) }
                offset += count
            }
        }
        guard fchmod(descriptor, mode_t(0o600)) == 0 else { throw posixError() }
        guard fsync(descriptor) == 0 else { throw posixError() }
        let closeResult = Darwin.close(descriptor)
        // Darwin releases the descriptor even when close reports an error.
        descriptorIsOpen = false
        guard closeResult == 0 else { throw posixError() }
        guard rename(temporaryPath, url.path) == 0 else { throw posixError() }
        shouldRemoveTemporary = false
    }

    private static func posixError() -> POSIXError {
        POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
    }
}
