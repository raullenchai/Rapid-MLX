import Foundation
import Security

/// The release signing identity is the trust anchor; candidate files never are.
/// Checking the outer resource seal covers the launcher, Python modules and
/// native dependencies together, rather than just checking the interpreter.
enum DesktopRuntimeTrust {
    /// A successful seal check is session authority for this app-relative
    /// binary path. Failures are never cached. The lock also joins concurrent
    /// launch/prewarm checks so they cannot re-hash the same bundle in parallel.
    /// Residual risk: tampering after success during this running session is
    /// not detected until relaunch. The bundle is not immutable; this extends
    /// the accepted same-user verification-to-use race for the session.
    final class Session: @unchecked Sendable {
        private let lock = NSLock()
        private var successful: Set<String> = []
        private var checks: [String: Int] = [:]
        private var launches: [String: Int] = [:]

        func authenticate(_ binary: URL, check: () -> Bool) -> Bool {
            lock.lock()
            defer { lock.unlock() }
            let key = binary.standardizedFileURL.path
            if successful.contains(key) { return true }
            checks[key, default: 0] += 1
            guard check() else { return false }
            successful.insert(key)
            return true
        }

        func recordLaunch(_ binary: URL) {
            lock.lock()
            defer { lock.unlock() }
            launches[binary.standardizedFileURL.path, default: 0] += 1
        }

        func activity(_ binary: URL) -> (checks: Int, launches: Int) {
            lock.lock()
            defer { lock.unlock() }
            let key = binary.standardizedFileURL.path
            return (checks[key, default: 0], launches[key, default: 0])
        }
    }

    static let session = Session()

    /// Start before catalog discovery or the first engine launch. A launch
    /// racing this task joins the same verification and cannot bypass it.
    static func prewarm(bundleResourceURL: URL? = Bundle.main.resourceURL) {
        guard let binary = bundleResourceURL?.appendingPathComponent("rapid-mlx/bin/rapid-mlx"),
              binary.path.hasSuffix(".app/Contents/Resources/rapid-mlx/bin/rapid-mlx") else { return }
        Task.detached(priority: .userInitiated) { _ = authenticate(binary) }
    }

    struct LaunchRefused: LocalizedError {
        var errorDescription: String? { "The bundled engine could not be verified. Reinstall the full Desktop app." }
    }

    /// Require session authentication at the launch boundary, after asynchronous
    /// preparation. Other helper executables are not managed engines. Keep the
    /// app-relative path intact so changing a symlink cannot change this policy.
    /// RAPID_BIN is the deliberate, documented developer trust bypass.
    static func verifyBeforeLaunch(
        _ binary: URL,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        authenticateBundle: (URL) -> Bool = authenticateUncached
    ) throws {
        if let override = environment["RAPID_BIN"], !override.isEmpty,
           URL(fileURLWithPath: override).resolvingSymlinksInPath() == binary.resolvingSymlinksInPath() {
            return
        }
        guard binary.path.hasSuffix(".app/Contents/Resources/rapid-mlx/bin/rapid-mlx") else { return }
        guard session.authenticate(binary, check: { authenticateBundle(binary) }) else { throw LaunchRefused() }
        session.recordLaunch(binary)
    }

    static func run(
        _ process: Process,
        authenticateBundle: (URL) -> Bool = authenticateUncached
    ) throws {
        guard let binary = process.executableURL else { throw LaunchRefused() }
        try verifyBeforeLaunch(binary, authenticateBundle: authenticateBundle)
        try process.run()
    }

    static func authenticate(_ binary: URL) -> Bool {
        session.authenticate(binary) { authenticateUncached(binary) }
    }

    static func authenticateUncached(_ binary: URL) -> Bool {
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
