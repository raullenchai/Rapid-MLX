import Foundation
import Security

/// The release signing identity is the trust anchor; candidate files never are.
/// Checking the outer resource seal covers the launcher, Python modules and
/// native dependencies together, rather than just checking the interpreter.
enum DesktopRuntimeTrust {
    struct LaunchRefused: LocalizedError {
        var errorDescription: String? { "The bundled engine could not be verified. Reinstall the full Desktop app." }
    }

    /// Recheck at the synchronous launch boundary, after any asynchronous
    /// preparation. Other helper executables are not managed engines. Keep the
    /// app-relative path intact so changing a symlink cannot change this policy.
    /// RAPID_BIN is the deliberate, documented developer trust bypass.
    static func verifyBeforeLaunch(
        _ binary: URL,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        authenticateBundle: (URL) -> Bool = authenticate
    ) throws {
        if let override = environment["RAPID_BIN"], !override.isEmpty,
           URL(fileURLWithPath: override).resolvingSymlinksInPath() == binary.resolvingSymlinksInPath() {
            return
        }
        guard binary.path.hasSuffix(".app/Contents/Resources/rapid-mlx/bin/rapid-mlx") else { return }
        guard authenticateBundle(binary) else { throw LaunchRefused() }
    }

    static func run(
        _ process: Process,
        authenticateBundle: (URL) -> Bool = authenticate
    ) throws {
        guard let binary = process.executableURL else { throw LaunchRefused() }
        try verifyBeforeLaunch(binary, authenticateBundle: authenticateBundle)
        try process.run()
    }

    static func authenticate(_ binary: URL) -> Bool {
        // Public release identity, pinned in source rather than read from the
        // candidate bundle or its writable metadata. Rotation needs review.
        let releaseIdentity = """
        anchor apple generic and identifier "com.rapidmlx.rapid" \
        and certificate leaf[subject.OU] = "73WQ7ZGSWC" \
        and certificate leaf[field.1.2.840.113635.100.6.1.13] exists
        """
        var requirement: SecRequirement?
        guard SecRequirementCreateWithString(releaseIdentity as CFString, [], &requirement) == errSecSuccess,
              let requirement else { return false }
        return authenticate(binary, requirement: requirement)
    }

    /// Tests supply the designated requirement of a genuinely signed fixture.
    /// Production always uses the pinned release identity above.
    static func authenticate(_ binary: URL, requirement: SecRequirement) -> Bool {
        let resources = binary.deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let app = resources.deletingLastPathComponent().deletingLastPathComponent()
        guard resources.lastPathComponent == "Resources",
              resources.deletingLastPathComponent().lastPathComponent == "Contents",
              app.pathExtension == "app",
              binary.standardizedFileURL.path == resources
                .appendingPathComponent("rapid-mlx/bin/rapid-mlx").standardizedFileURL.path,
              containedRuntime(resources.appendingPathComponent("rapid-mlx"), in: app) else {
            return false
        }
        var code: SecStaticCode?
        guard SecStaticCodeCreateWithPath(app as CFURL, [], &code) == errSecSuccess,
              let code else { return false }
        let flags = SecCSFlags(rawValue: kSecCSStrictValidate | kSecCSCheckAllArchitectures | kSecCSCheckNestedCode)
        return SecStaticCodeCheckValidity(code, flags, requirement) == errSecSuccess
    }

    private static func containedRuntime(_ root: URL, in app: URL) -> Bool {
        // Resource seals authenticate symlink text, not arbitrary external
        // targets. Keep every runtime dependency inside the sealed app.
        let prefix = app.resolvingSymlinksInPath().path + "/"
        guard root.resolvingSymlinksInPath().path.hasPrefix(prefix) else { return false }
        var failed = false
        guard let entries = FileManager.default.enumerator(
            at: root, includingPropertiesForKeys: [.isSymbolicLinkKey],
            errorHandler: { _, _ in failed = true; return false }
        ) else { return false }
        for case let entry as URL in entries {
            guard let values = try? entry.resourceValues(forKeys: [.isSymbolicLinkKey]) else { return false }
            if values.isSymbolicLink == true,
               !entry.resolvingSymlinksInPath().path.hasPrefix(prefix) { return false }
        }
        return !failed
    }
}
