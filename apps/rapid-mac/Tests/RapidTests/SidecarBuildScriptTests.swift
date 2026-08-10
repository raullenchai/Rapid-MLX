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
        // Pin the EQUALITY, not a count. The invariant is "every build-time
        // install runs on the pinned interpreter", and a hard-coded total
        // re-breaks the moment a dependency is added (bundling mflux for the
        // Images tab made this 2 -> 3) while saying nothing about whether the
        // NEW install honours the rule. Comparing the two tallies fails only
        // when an install actually escapes the pinned interpreter.
        let anyPip = "-m pip install"
        let embeddedInstalls = script.components(separatedBy: embeddedPip).count - 1
        let allInstalls = script.components(separatedBy: anyPip).count - 1
        #expect(embeddedInstalls >= 2,
                "The engine and vision-stack installs must both still be here.")
        #expect(embeddedInstalls == allInstalls,
                """
                Every build-time pip install must use the pinned interpreter \
                extracted into the bundle; \(allInstalls - embeddedInstalls) \
                of \(allInstalls) do not.
                """)
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

    @Test("Desktop vision stack is exact-pinned and metadata-validated")
    func visionStackIsReproducibleAndCompatible() throws {
        let script = try String(contentsOf: Self.scriptURL, encoding: .utf8)

        #expect(script.contains("'mlx-vlm==0.6.3'"))
        #expect(!script.contains("'mlx-vlm>=0.6.3,!=0.6.4,<0.7'"),
                "The no-deps sidecar install must never float within a range.")
        #expect(script.contains("from packaging.requirements import Requirement"))
        #expect(script.contains("actual not in req.specifier"),
                "Post-install validation must reject incompatible installed dependency versions.")
    }

    @Test("Desktop image stack is pinned and its torch-free proof fails closed")
    func imageStackIsPinnedAndProvenTorchFree() throws {
        let script = try String(contentsOf: Self.scriptURL, encoding: .utf8)

        #expect(script.contains("'mflux==0.18.1'"),
                "The no-deps sidecar install must never float within a range.")
        // The Images tab is only shippable because mflux's module-level
        // `import torch` is deferred into the three torch-only loading modes —
        // bundling torch itself would be +363 MB against a 500 MB cap. Both
        // halves of that argument have to fail closed, or a future mflux bump
        // ships an Images tab that dies on every generation: the patch must
        // refuse to guess when weight_loader.py has been reshaped, and the
        // import probe must prove the result needs no torch.
        #expect(script.contains("no longer has the eager import"),
                "The torch-deferral patch must abort on an unrecognised weight_loader.py.")
        #expect(script.contains("has no single-line def for"),
                "The torch-deferral patch must abort when a target function moves.")
        #expect(script.contains("mflux still pulls torch at import time"),
                "A post-patch import probe must prove the image lane needs no torch.")
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
