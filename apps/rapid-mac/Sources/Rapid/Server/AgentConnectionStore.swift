import Foundation
import Darwin

/// Shared loopback connection for Desktop and the bundled CLI. The directory
/// and files are owner-only; no secret is passed in argv or written to logs.
enum AgentConnectionStore {
    static var directory: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".rapid-mlx/connect", isDirectory: true)
    }

    static func readKey() -> String? {
        let path = directory.appendingPathComponent("api-key")
        guard !isLink(directory), !isLink(path),
              let attributes = try? FileManager.default.attributesOfItem(atPath: path.path),
              let mode = attributes[.posixPermissions] as? NSNumber,
              mode.intValue & 0o077 == 0,
              let data = try? Data(contentsOf: path),
              let key = String(data: data, encoding: .ascii)?.trimmingCharacters(in: .whitespacesAndNewlines),
              key.count == 64,
              key.allSatisfy({ $0.isASCII && $0.isHexDigit }) else { return nil }
        return key
    }

    static func readPort() -> Int? {
        let path = directory.appendingPathComponent("port")
        guard !isLink(directory), !isLink(path),
              let attributes = try? FileManager.default.attributesOfItem(atPath: path.path),
              let mode = attributes[.posixPermissions] as? NSNumber,
              mode.intValue & 0o077 == 0,
              let text = try? String(contentsOf: path, encoding: .ascii),
              let port = Int(text.trimmingCharacters(in: .whitespacesAndNewlines)),
              (1...65_535).contains(port) else { return nil }
        return port
    }

    @discardableResult
    static func save(key: String, port: Int) -> Bool {
        guard key.count == 64, key.allSatisfy({ $0.isASCII && $0.isHexDigit }),
              (1...65_535).contains(port), !isLink(directory) else { return false }
        do {
            try FileManager.default.createDirectory(
                at: directory,
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700]
            )
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o700], ofItemAtPath: directory.path
            )
            return write(key + "\n", name: "api-key")
                && write("\(port)\n", name: "port")
        } catch {
            return false
        }
    }

    @discardableResult
    static func clearKey() -> Bool {
        let path = directory.appendingPathComponent("api-key")
        guard !isLink(directory), !isLink(path) else { return false }
        do {
            if FileManager.default.fileExists(atPath: path.path) {
                try FileManager.default.removeItem(at: path)
            }
            return true
        } catch {
            return false
        }
    }

    private static func write(_ content: String, name: String) -> Bool {
        let path = directory.appendingPathComponent(name).path
        let fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW, 0o600)
        guard fd >= 0 else { return false }
        defer { close(fd) }
        guard fchmod(fd, 0o600) == 0 else { return false }
        let bytes = Array(content.utf8)
        return bytes.withUnsafeBytes { buffer in
            guard let base = buffer.baseAddress else { return false }
            return Darwin.write(fd, base, buffer.count) == buffer.count
        }
    }

    private static func isLink(_ path: URL) -> Bool {
        (try? FileManager.default.destinationOfSymbolicLink(atPath: path.path)) != nil
    }
}
