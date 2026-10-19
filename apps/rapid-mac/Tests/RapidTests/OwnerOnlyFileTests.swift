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

    func testNewFileIsOwnerOnly() throws {
        let dir = try directory()
        let url = dir.appendingPathComponent("private.json")
        let data = Data(repeating: 0x61, count: 256 * 1024)
        try OwnerOnlyFile.writeAtomically(data, to: url)
        XCTAssertEqual(try mode(url), 0o600)
        XCTAssertEqual(try Data(contentsOf: url), data)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: dir.path), [url.lastPathComponent])
    }

    func testTemporaryFileIsOwnerOnlyBeforeWriting() throws {
        let dir = try directory()
        let url = dir.appendingPathComponent("private.json")
        let data = Data("private contents".utf8)
        var observedTemporaryFile = false

        try OwnerOnlyFile.writeAtomically(data, to: url) { temporaryPath, descriptor in
            observedTemporaryFile = true
            XCTAssertNotEqual(fcntl(descriptor, F_GETFD) & FD_CLOEXEC, 0)
            let temporaryURL = URL(fileURLWithPath: temporaryPath)
            XCTAssertTrue(FileManager.default.fileExists(atPath: temporaryPath))
            XCTAssertEqual(temporaryURL.deletingLastPathComponent().path, dir.path)
            XCTAssertEqual(try mode(temporaryURL), 0o600)
            let attributes = try FileManager.default.attributesOfItem(atPath: temporaryPath)
            XCTAssertEqual((attributes[.size] as? NSNumber)?.intValue, 0)
            XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
        }

        XCTAssertTrue(observedTemporaryFile)
        XCTAssertEqual(try mode(url), 0o600)
        XCTAssertEqual(try Data(contentsOf: url), data)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: dir.path), [url.lastPathComponent])
    }

    func testThrowingBeforeWritePropagatesErrorAndRemovesTemporaryFile() throws {
        enum TestError: Error, Equatable { case beforeWrite }
        let dir = try directory()
        let url = dir.appendingPathComponent("private.json")
        var observedTemporaryFile = false

        XCTAssertThrowsError(try OwnerOnlyFile.writeAtomically(Data("private".utf8), to: url) { temporaryPath, _ in
            observedTemporaryFile = true
            XCTAssertTrue(FileManager.default.fileExists(atPath: temporaryPath))
            throw TestError.beforeWrite
        }) {
            XCTAssertEqual($0 as? TestError, .beforeWrite)
        }

        XCTAssertTrue(observedTemporaryFile)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: dir.path), [])
    }

    func testOverwriteReplacesContentsAndTightensPermissions() throws {
        let dir = try directory()
        let url = dir.appendingPathComponent("private.json")
        try Data("old".utf8).write(to: url)
        try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: url.path)
        XCTAssertEqual(try mode(url), 0o644)
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
