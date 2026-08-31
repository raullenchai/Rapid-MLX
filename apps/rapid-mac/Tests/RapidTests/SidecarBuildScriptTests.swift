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
        let constraints = try String(contentsOf: Self.constraintsURL, encoding: .utf8)

        #expect(constraints.contains("mlx-vlm==0.6.17"))
        #expect(!script.contains("'mlx-vlm>=0.6.3,!=0.6.4,<0.7'"),
                "The no-deps sidecar install must never float within a range.")
        #expect(script.contains(#"--constraint "$SIDECAR_CONSTRAINTS""#),
                "Every install must consume the shared release constraint set.")
        #expect(script.contains("$REPO_ROOT/scripts/check-sidecar-distributions.py"),
                "Every sidecar build must execute the tested metadata-coherence gate.")
        #expect(script.contains("SIDECAR_VISION_SMOKE_MODEL"))
        #expect(script.contains("$REPO_ROOT/scripts/smoke-sidecar-vision.py"),
                "Configured release-candidate builds must run real image inference through the sidecar.")
        #expect(script.contains(#"PYTHONPATH="$STAGE/site-packages" PYTHONNOUSERSITE=1"#),
                "The image gate must resolve its pinned snapshot with the bundled runtime, not host Python.")
        for architecture in [
            "diffusion_gemma", "gemma3", "gemma3n", "gemma4", "gemma4_unified",
            "qwen3_5", "qwen3_5_moe", "qwen3_vl", "qwen3_vl_moe",
        ] {
            #expect(script.contains(architecture),
                    "Every Desktop-advertised vision architecture needs a bundled import smoke.")
        }
        #expect(script.contains(#"find_spec("cv2") is None"#))
        #expect(script.contains(#"find_spec("torch") is None"#))
        #expect(script.contains(#"find_spec("torchvision") is None"#),
                "The reduced vision bundle must prove its advertised paths stay torch/cv2-free.")
    }

    @Test("Desktop MLX wheels target the app's minimum macOS")
    func mlxWheelsMatchDeploymentTarget() throws {
        let script = try String(contentsOf: Self.scriptURL, encoding: .utf8)

        #expect(script.contains("--platform macosx_14_0_arm64"),
                "A newer build host must not make the sidecar require that host's macOS.")
        #expect(script.contains("^Tag: .*macosx_14_0_arm64$"),
                "The build must fail closed if pip does not install the requested compatible wheels.")
        #expect(script.contains(#""mlx==${MLX_VERSION}""#))
        #expect(script.contains(#""mlx-metal==${MLX_METAL_VERSION}""#),
                "Core and metallib wheels must be replaced as one matched pair.")
    }

    @Test("Desktop image stack is pinned and its torch-free proof fails closed")
    func imageStackIsPinnedAndProvenTorchFree() throws {
        let script = try String(contentsOf: Self.scriptURL, encoding: .utf8)
        let constraints = try String(contentsOf: Self.constraintsURL, encoding: .utf8)

        #expect(constraints.contains("mflux==0.19.0"),
                "The no-deps sidecar install must never float within a range.")
        // The Images tab is only shippable because mflux's module-level
        // `import torch` is deferred into the three torch-only loading modes —
        // bundling torch itself would be +363 MB against a 550 MB cap. Both
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
        #expect(script.contains("mflux/models/common/pid_decoder/pid_weight_mapping.py"),
                "The Qwen Image import path must defer PiD's optional torch checkpoint converter.")
        #expect(script.contains(#"importlib.import_module("mflux.models.qwen.variants.txt2img.qwen_image")"#),
                "The bundle build must prove qwen-image itself imports without torch.")
        #expect(script.contains("SIDECAR_IMAGE_SMOKE_MODEL"),
                "Release-candidate builds must opt into a real image-generation model.")
        #expect(script.contains("$REPO_ROOT/scripts/smoke-sidecar-image.py"),
                "The image runtime needs a real generated-PNG gate, not only an import probe.")
        let visionSmoke = script.range(of: "smoke-sidecar-vision.py")?.lowerBound
        let imageSmoke = script.range(of: "smoke-sidecar-image.py")?.lowerBound
        #expect(visionSmoke != nil)
        #expect(imageSmoke != nil)
        if let visionSmoke, let imageSmoke {
            #expect(visionSmoke < imageSmoke,
                    "Real Metal model loads must remain serialized on the release host.")
        }
    }

    @Test("Desktop sidecar bundles and smokes both audio lanes")
    func audioRuntimeIsBundled() throws {
        let script = try String(contentsOf: Self.scriptURL, encoding: .utf8)
        let pyproject = try String(contentsOf: Self.pyprojectURL, encoding: .utf8)

        #expect(pyproject.contains("\naudio-desktop = ["),
                "The bounded desktop audio dependency group must remain separately installable.")
        #expect(pyproject.contains(#""mlx-audio>=0.2.9,<0.4.4""#))
        #expect(pyproject.contains(#""soundfile>=0.12.0""#))
        #expect(script.contains(#""${RAPID_MLX_INSTALL_TARGET}[audio-desktop]""#),
                "The desktop sidecar must install the bounded desktop audio dependency set.")
        #expect(script.contains(#"RAPID_MLX_WHEEL="${RAPID_MLX_WHEEL:-}""#),
                "Release builds must be able to promote the exact candidate wheel into the sidecar.")
        #expect(script.contains("from mlx_audio.stt.utils import load_model"),
                "The build smoke must import the transcription loader, not only mlx_audio's package root.")
        #expect(script.contains("from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor"),
                "The build smoke must prove the processor implementation survives trimming.")
        #expect(script.contains(#"-not -path "*/transformers/models/whisper/feature_extraction_whisper.py""#),
                "Whisper's processor fallback cannot work when its feature extractor is trimmed.")
        #expect(script.contains("from mlx_audio.tts.generate import load_model"),
                "The build smoke must import the speech loader.")
        #expect(script.contains("from mlx_audio.tts.models.qwen3_tts import Model"),
                "The smoke must cover the preset-voice family exposed by the desktop picker.")
        #expect(script.contains("from scipy import signal"),
                "The smoke must cover the resampler after SciPy trimming.")
        #expect(script.contains("TTSEngine.__new__(TTSEngine).to_bytes"),
                "The smoke must encode a WAV after scipy.io has been trimmed.")
        #expect(script.contains(#"-not -name qwen3_tts -not -name __pycache__"#),
                "Only model-family directories outside Qwen3 TTS may be removed.")
        #expect(!script.contains(#"rm -rf "$STAGE/site-packages/mlx_audio/tts/models""#),
                "The trim must never remove the complete TTS model directory.")
        #expect(script.contains(#"MACHO_BASELINE_COUNT="${MACHO_BASELINE_COUNT:-172}""#),
                "The signing baseline must match the measured post-audio bundle, less the orphaned libpython dylib dropped in step 3.")
    }

    /// The shared libpython optimization must stay behind the tested guard.
    @Test("The unreferenced libpython dylib uses the fail-closed guard")
    func guardsOrphanedLibpythonRemoval() throws {
        let script = try String(contentsOf: Self.scriptURL, encoding: .utf8)

        #expect(script.contains(#""$REPO_ROOT/scripts/prune-unused-libpython.sh" "$STAGE""#))
        #expect(!script.contains(#"rm -f "$STAGE/python/lib/libpython3.12.dylib""#),
                "build-sidecar must not bypass the consumer scan with an unconditional deletion.")
        #expect(!script.contains(#"strip -x "$STAGE/python/bin/python3.12""#),
                "Local CPython symbols are required for useful native crash frames.")
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

    private static var constraintsURL: URL {
        scriptURL.deletingLastPathComponent().appendingPathComponent("sidecar-constraints.txt")
    }

    private static var pyprojectURL: URL {
        scriptURL
            .deletingLastPathComponent() // scripts
            .deletingLastPathComponent() // rapid-mac
            .deletingLastPathComponent() // apps
            .deletingLastPathComponent() // repository root
            .appendingPathComponent("pyproject.toml")
    }
}
