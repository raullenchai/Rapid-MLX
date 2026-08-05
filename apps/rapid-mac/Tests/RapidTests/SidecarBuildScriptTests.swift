import Foundation
import Testing

@Suite("Sidecar build script")
struct SidecarBuildScriptTests {
    @Test("Bundling bootstraps pip with the pinned embedded Python")
    func usesEmbeddedPythonForBuildTimeInstalls() throws {
        let script = try String(contentsOf: Self.scriptURL, encoding: .utf8)
        let embeddedPip = #""$STAGE/python/bin/python3.12" -m pip install"#

        #expect(!script.contains("require python3.12"),
                "The script downloads pinned Python 3.12 itself and must not require a host copy.")
        #expect(script.components(separatedBy: embeddedPip).count - 1 == 2,
                "Both dependency installs must use the pinned interpreter extracted into the bundle.")
        #expect(!script.contains("\npython3.12 -m pip install"),
                "A host Python can resolve wheels for the wrong ABI and make the bundle unloadable.")
    }

    @Test("Pinned Python downloads are retried, validated, and committed atomically")
    func pythonDownloadIsResilient() throws {
        let script = try String(contentsOf: Self.scriptURL, encoding: .utf8)

        #expect(script.contains(#"PBS_TAR_TMP="${PBS_TAR}.tmp""#))
        #expect(script.contains(#"tar -tzf "$PBS_TAR""#),
                "A truncated cached archive must be rejected before extraction.")
        #expect(script.contains("--retry-all-errors"),
                "HTTP/2 framing and other transient transport errors must be retried.")
        #expect(script.contains(#"-o "$PBS_TAR_TMP""#),
                "Downloads must not write directly to the trusted cache path.")
        #expect(script.contains(#"mv "$PBS_TAR_TMP" "$PBS_TAR""#),
                "Only a validated temporary archive may become the trusted cache.")
    }

    @Test("Bytecode compilation can run without process semaphores")
    func compileallConcurrencyIsConfigurable() throws {
        let script = try String(contentsOf: Self.scriptURL, encoding: .utf8)

        #expect(script.contains(#"COMPILEALL_JOBS="${COMPILEALL_JOBS:-0}""#),
                "Normal builds should retain automatic compileall parallelism.")
        #expect(script.contains(#"-j "$COMPILEALL_JOBS""#),
                "Restricted builders need a single-process compileall override.")
    }

    private static var scriptURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("scripts/build-sidecar.sh")
    }

    private static var appBuildScriptURL: URL {
        scriptURL.deletingLastPathComponent().appendingPathComponent("build.sh")
    }
}
