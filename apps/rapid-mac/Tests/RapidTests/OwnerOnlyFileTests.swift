import Darwin
import Foundation
import XCTest
@testable import Rapid

final class OwnerOnlyFileTests: XCTestCase {
    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-owner-only-file-tests", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try FileManager.default.removeItem(at: url) }
        return url
    }

    private func mode(_ url: URL) throws -> Int? {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        return (attributes[.posixPermissions] as? NSNumber)?.intValue
    }

    func testNewFileIsOwnerOnlyUnderPermissiveUmask() throws {
        let dir = try directory()
        let url = dir.appendingPathComponent("private.json")
        let previous = umask(0o022)
        defer { umask(previous) }
        let data = Data(repeating: 0x61, count: 256 * 1024)
        try OwnerOnlyFile.writeAtomically(data, to: url)
        XCTAssertEqual(try mode(url), 0o600)
        XCTAssertEqual(try Data(contentsOf: url), data)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: dir.path), [url.lastPathComponent])
    }

    func testOverwriteReplacesContentsAndTightensPermissions() throws {
        let dir = try directory()
        let url = dir.appendingPathComponent("private.json")
        try Data("old".utf8).write(to: url)
        try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: url.path)
        XCTAssertEqual(try mode(url), 0o644)
        let previous = umask(0o022)
        defer { umask(previous) }
        let data = Data("new contents".utf8)
        try OwnerOnlyFile.writeAtomically(data, to: url)
        XCTAssertEqual(try mode(url), 0o600)
        XCTAssertEqual(try Data(contentsOf: url), data)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: dir.path), [url.lastPathComponent])
    }

    func testMissingDirectoryThrowsWithoutLeavingFiles() throws {
        let dir = try directory()
        let url = dir.appendingPathComponent("missing/private.json")
        XCTAssertThrowsError(try OwnerOnlyFile.writeAtomically(Data("private".utf8), to: url)) {
            XCTAssertEqual(($0 as? POSIXError)?.code, .ENOENT)
        }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: dir.path), [])
    }

    func testRenameFailureRemovesTemporaryFile() throws {
        let dir = try directory()
        let destination = dir.appendingPathComponent("existing-directory")
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: false)
        XCTAssertThrowsError(try OwnerOnlyFile.writeAtomically(Data("private".utf8), to: destination)) {
            XCTAssertTrue($0 is POSIXError)
        }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: dir.path), [destination.lastPathComponent])
    }

    func testEmptyFile() throws {
        let url = try directory().appendingPathComponent("empty")
        try OwnerOnlyFile.writeAtomically(Data(), to: url)
        XCTAssertEqual(try mode(url), 0o600)
        XCTAssertEqual(try Data(contentsOf: url), Data())
    }
}
