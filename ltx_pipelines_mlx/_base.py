"""Shared base class for LTX-2 pipelines — composition blocks + proxy properties + common helpers.

Private to ``ltx_pipelines_mlx``. Each public pipeline class
(``TI2VidOneStagePipeline``, ``TI2VidTwoStagesPipeline``, ``ICLoraPipeline``,
``A2VidPipelineTwoStage``, ``RetakePipeline``, …) subclasses this facade and
implements its own ``generate_*`` + ``generate_and_save`` entry points. The
facade itself does not define a generation method — matching the upstream
Lightricks/LTX-2 design where each pipeline file owns its API.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder
from ltx_core_mlx.model.audio_vae.bwe import VocoderWithBWE
from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig
from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder, VideoEncoder
from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel
from ltx_core_mlx.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorV2
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.weights import apply_quantization, load_split_safetensors
from ltx_pipelines_mlx.utils.constants import DEFAULT_NEGATIVE_PROMPT
from ltx_pipelines_mlx.utils.progress import phase


class BasePipeline:
    """Shared facade for all LTX-2 pipelines.

    Owns the composition blocks (prompt encoder, image/audio conditioners,
    decoders), proxy properties for legacy ``self.text_encoder`` /
    ``self.vae_*`` attribute access, and the common load / encode / decode
    helpers. Concrete generation logic lives on subclasses
    (``TI2VidOneStagePipeline``, ``TI2VidTwoStagesPipeline``, etc.) which
    define their own ``generate_*`` + ``generate_and_save`` entry points.

    Args:
        model_dir: Path to model weights or HuggingFace repo ID.
        low_memory: If True, aggressively free memory between stages.
        low_ram_streaming: If True, stream transformer blocks from
            mmap'd safetensors instead of materializing all 48 blocks.
            Cuts transformer peak RSS from ~10-12 GB (q8) or ~22 GB
            (bf16) to ~0.6 GB. Adds ~48 sync points per forward, so
            slightly slower per step. Compatible with ``_pending_loras``
            via per-block ``BlockLoraSource`` bind-time fusion.
        verbose: If True (default), print phase markers to stderr around
            long silent stages (Gemma load/encode, transformer load,
            decoder load, video decode). CLI maps this from ``--quiet``.
    """

    def __init__(
        self,
        model_dir: str,
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
        low_memory: bool = True,
        low_ram_streaming: bool = False,
        verbose: bool = True,
    ):
        self.model_dir = self._resolve_model_dir(model_dir)
        self._gemma_model_id = gemma_model_id
        self.low_memory = low_memory
        self.low_ram_streaming = low_ram_streaming
        self.verbose = verbose
        self._loaded = False

        if self.low_ram_streaming:
            # Disable Metal heap cache before any allocation. With cache enabled,
            # MLX retains "recently freed" buffers in a heap that defeats
            # streaming on macOS unified memory. Setting before any module load
            # ensures Gemma + decoder allocations also benefit.
            mx.set_cache_limit(0)

        self._dev_transformer: str | None = None

        # Composition blocks — own each component's load/free lifecycle. The
        # ``self.text_encoder`` / ``self.vae_encoder`` / etc. attributes
        # below are properties that proxy these blocks for backward compat
        # with subclasses that read/write them directly.
        from ltx_pipelines_mlx.utils.blocks import (
            AudioConditioner as _AudioConditionerBlock,
        )
        from ltx_pipelines_mlx.utils.blocks import (
            AudioDecoder as _AudioDecoderBlock,
        )
        from ltx_pipelines_mlx.utils.blocks import (
            ImageConditioner as _ImageConditionerBlock,
        )
        from ltx_pipelines_mlx.utils.blocks import (
            PromptEncoder as _PromptEncoderBlock,
        )
        from ltx_pipelines_mlx.utils.blocks import (
            VideoDecoder as _VideoDecoderBlock,
        )

        self.prompt_encoder = _PromptEncoderBlock(self.model_dir, gemma_model_id)
        self.image_conditioner = _ImageConditionerBlock(self.model_dir)
        self.audio_conditioner = _AudioConditionerBlock(self.model_dir)
        self.video_decoder_block = _VideoDecoderBlock(self.model_dir, verbose=self.verbose)
        self.audio_decoder_block = _AudioDecoderBlock(self.model_dir)

        self.dit: LTXModel | None = None
        self.video_patchifier = VideoLatentPatchifier()
        self.audio_patchifier = AudioPatchifier()

    # -------------------- proxy properties to blocks --------------------
    # Subclasses still read/write these as direct attributes; the property
    # tier proxies them onto the underlying composition blocks so memory
    # frees propagate (e.g. ``self.text_encoder = None`` releases the
    # block's strong ref too).

    @property
    def text_encoder(self) -> GemmaLanguageModel | None:
        return self.prompt_encoder._text_encoder

    @text_encoder.setter
    def text_encoder(self, value: GemmaLanguageModel | None) -> None:
        self.prompt_encoder._text_encoder = value

    @property
    def feature_extractor(self) -> GemmaFeaturesExtractorV2 | None:
        return self.prompt_encoder._feature_extractor

    @feature_extractor.setter
    def feature_extractor(self, value: GemmaFeaturesExtractorV2 | None) -> None:
        self.prompt_encoder._feature_extractor = value

    @property
    def vae_encoder(self) -> VideoEncoder | None:
        return self.image_conditioner._encoder

    @vae_encoder.setter
    def vae_encoder(self, value: VideoEncoder | None) -> None:
        self.image_conditioner._encoder = value

    @property
    def vae_decoder(self) -> VideoDecoder | None:
        return self.video_decoder_block._decoder

    @vae_decoder.setter
    def vae_decoder(self, value: VideoDecoder | None) -> None:
        self.video_decoder_block._decoder = value

    @property
    def audio_decoder(self) -> AudioVAEDecoder | None:
        return self.audio_decoder_block._audio_decoder

    @audio_decoder.setter
    def audio_decoder(self, value: AudioVAEDecoder | None) -> None:
        self.audio_decoder_block._audio_decoder = value

    @property
    def vocoder(self) -> VocoderWithBWE | None:
        return self.audio_decoder_block._vocoder

    @vocoder.setter
    def vocoder(self, value: VocoderWithBWE | None) -> None:
        self.audio_decoder_block._vocoder = value

    @property
    def audio_encoder(self) -> object | None:
        return self.audio_conditioner._encoder

    @audio_encoder.setter
    def audio_encoder(self, value: object | None) -> None:
        self.audio_conditioner._encoder = value

    @property
    def audio_processor(self) -> object | None:
        return self.audio_conditioner._processor

    @audio_processor.setter
    def audio_processor(self, value: object | None) -> None:
        self.audio_conditioner._processor = value

    @staticmethod
    def _resolve_model_dir(model_dir: str) -> Path:
        """Inheritance wrapper around :func:`utils._orchestration.resolve_model_dir`."""
        from ltx_pipelines_mlx.utils._orchestration import resolve_model_dir as _impl

        return _impl(model_dir)

    @staticmethod
    def _fuse_pending_loras(
        transformer_weights: dict[str, mx.array],
        lora_paths: list[tuple[str, float]],
    ) -> dict[str, mx.array]:
        """Inheritance wrapper around :func:`utils._orchestration.fuse_pending_loras`."""
        from ltx_pipelines_mlx.utils._orchestration import fuse_pending_loras as _impl

        return _impl(transformer_weights, lora_paths)

    # ------------------------------------------------------------------
    # Shared component loading methods (used by subclass pipelines)
    # ------------------------------------------------------------------

    def _load_text_encoder(self) -> None:
        """Load Gemma + connector via the :class:`PromptEncoder` block."""
        with phase("Loading text encoder (Gemma)", verbose=self.verbose):
            self.prompt_encoder.load()

    def _encode_text_with_negative(self, prompt: str) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Load text encoder, encode prompt + negative prompt, materialize, free encoder.

        The two encode calls are materialized **separately** (intermediate
        materialize between positive and negative) so they don't merge into
        a single lazy graph that would queue 2x the Gemma + connector
        forwards into one Metal command buffer — under sustained system
        contention, that combined buffer exceeds the macOS GPU watchdog
        on <=48 GB Macs at HQ shapes (e.g. 1280x704x97).

        Returns:
            Tuple of (video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds).
        """
        _materialize = getattr(mx, "eval")  # noqa: B009 -- security hook flags mx.eval pattern

        self._load_text_encoder()

        with phase("Encoding prompt", verbose=self.verbose):
            video_embeds, audio_embeds = self._encode_text(prompt)
            _materialize(video_embeds, audio_embeds)
            neg_video_embeds, neg_audio_embeds = self._encode_text(DEFAULT_NEGATIVE_PROMPT)
            _materialize(neg_video_embeds, neg_audio_embeds)

        # Free text encoder before loading heavy components
        self.text_encoder = None
        self.feature_extractor = None
        aggressive_cleanup()

        return video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds

    def _load_vae_encoder(self) -> None:
        """Load VAE encoder via the :class:`ImageConditioner` block."""
        self.image_conditioner.load()

    def _load_audio_encoder(self) -> None:
        """Load audio VAE encoder + processor via the AudioConditioner block."""
        self.audio_conditioner.load()

    def _load_decoders(self) -> None:
        """Load VAE decoder + audio decoder + vocoder via composition blocks."""
        with phase("Loading decoders (VAE + audio + vocoder)", verbose=self.verbose):
            self.video_decoder_block.load()
            self.audio_decoder_block.load()

    def _load_dev_transformer(self) -> LTXModel:
        """Load the dev (non-distilled) transformer; honors ``_pending_loras``."""
        assert self._dev_transformer is not None, "_dev_transformer must be set before calling _load_dev_transformer()"
        dev_path = self.model_dir / self._dev_transformer
        if not dev_path.exists():
            raise FileNotFoundError(
                f"Dev transformer not found: {dev_path}\n"
                "This pipeline requires the dev model for CFG guidance.\n"
                "Use: --model dgrauet/ltx-2.3-mlx-q8"
            )
        return self._load_transformer_with_optional_streaming(dev_path)

    @staticmethod
    def _resolve_safetensors(model_dir: Path, stem: str) -> Path:
        """Return the path for a (possibly versioned) safetensors file.

        Prefers explicitly versioned files (``{stem}-*.safetensors``, e.g.
        ``transformer-distilled-1.1.safetensors``) over the unversioned exact
        name, taking the alphabetically latest when multiple versions exist.
        Falls back to ``{stem}.safetensors`` when no versioned file is found,
        and returns the canonical exact path when nothing exists so callers
        surface a clear FileNotFoundError.

        Note: selection is lexicographic, not semantic. This is correct for the
        current single-digit minor scheme (``1.0`` < ``1.1``) but would order
        ``-1.10`` *below* ``-1.9``. Revisit with a numeric version key if LTX
        ever ships double-digit minors.
        """
        versioned = sorted(model_dir.glob(f"{stem}-*.safetensors"))
        if versioned:
            return versioned[-1]
        return model_dir / f"{stem}.safetensors"

    def _load_transformer_with_optional_streaming(self, transformer_path: Path) -> LTXModel:
        """Load a transformer from ``transformer_path``; honors ``_pending_loras``.

        The single entry point for DiT construction across every pipeline's
        ``load()``. Routes through :func:`utils._orchestration.load_transformer`
        when no LoRAs are pending, or fuses LoRA deltas into the weight dict
        before quantization when ``self._pending_loras`` is set by the CLI.

        In ``low_ram_streaming`` mode, LoRAs are attached as
        :class:`BlockLoraSource` objects on the :class:`StreamingLTXModel`
        wrapper — fusion happens per-block at each bind rather than
        materialising the full weight dict. This mirrors
        :meth:`ICLoraPipeline._fuse_loras`'s streaming branch.
        """
        pending_loras = getattr(self, "_pending_loras", None)
        with phase(f"Loading transformer ({transformer_path.name})", verbose=self.verbose):
            if not pending_loras:
                from ltx_pipelines_mlx.utils._orchestration import load_transformer as _impl

                return _impl(transformer_path, low_ram_streaming=self.low_ram_streaming)

            if self.low_ram_streaming:
                from ltx_core_mlx.loader.block_streaming import BlockLoraSource
                from ltx_core_mlx.loader.sd_ops import (
                    LTXV_LORA_BLOCK_PREFIX,
                    LTXV_LORA_COMFY_RENAMING_MAP,
                )
                from ltx_pipelines_mlx.utils._orchestration import load_transformer as _load_impl
                from ltx_pipelines_mlx.utils._orchestration import resolve_lora_path

                model = _load_impl(transformer_path, low_ram_streaming=True)
                sources: list = list(object.__getattribute__(model, "_lora_sources"))
                for lora_path, strength in pending_loras:
                    resolved = resolve_lora_path(lora_path)
                    sources.append(
                        BlockLoraSource(
                            resolved,
                            block_prefix=LTXV_LORA_BLOCK_PREFIX,
                            strength=strength,
                            sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                        )
                    )
                object.__setattr__(model, "_lora_sources", sources)
                return model

            transformer_weights = load_split_safetensors(transformer_path, prefix="transformer.")
            transformer_weights = self._fuse_pending_loras(transformer_weights, pending_loras)
            dit = LTXModel(LTXModelConfig.from_checkpoint_dir(transformer_path.parent))
            apply_quantization(dit, transformer_weights)
            dit.load_weights(list(transformer_weights.items()))
            # Force materialisation so the phase marker reports actual load
            # work (MLX is lazy — load_weights builds a graph only). Mirrors
            # the same guard in utils._orchestration.load_transformer.
            _materialize = getattr(mx, "eval")  # noqa: B009 -- mx.eval is the MLX graph materialiser
            _materialize(dit.parameters())
            aggressive_cleanup()
            return dit

    def _decode_and_save_video(
        self,
        video_latent: mx.array,
        audio_latent: mx.array,
        output_path: str,
        *,
        frame_rate: float,
    ) -> str:
        """Inheritance wrapper around :func:`utils._orchestration.decode_and_save_video`."""
        from ltx_pipelines_mlx.utils._orchestration import decode_and_save_video as _impl

        with phase("Decoding video + audio + muxing", verbose=self.verbose):
            return _impl(
                self.video_decoder_block,
                self.audio_decoder_block,
                video_latent,
                audio_latent,
                output_path,
                frame_rate=frame_rate,
                low_memory=self.low_memory,
            )

    # ------------------------------------------------------------------
    # Original one-stage pipeline methods
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load DiT + VAE + decoders from disk.

        Subclasses' ``generate_*`` methods own the text encoder
        lifecycle: they encode the prompt and free Gemma before
        calling :meth:`load`, so this method intentionally does NOT
        reload Gemma. Doing so would thrash the Metal heap (7.5 GB
        load/mmap + free) right before the 10 GB DiT — a documented
        cause of macOS GPU watchdog crashes under sustained system
        contention.
        """
        if self._loaded:
            return

        model_dir = self.model_dir

        # Stage 1: DiT (largest component); LoRA fusion happens inside the wrapper.
        if self.dit is None:
            transformer_path = model_dir / "transformer.safetensors"
            if not transformer_path.exists():
                transformer_path = self._resolve_safetensors(model_dir, "transformer-distilled")

            self.dit = self._load_transformer_with_optional_streaming(transformer_path)

        # Stage 2: VAE + audio (smaller components)
        self._load_decoders()

        self._loaded = True

    def _encode_text_and_load(self, prompt: str) -> tuple[mx.array, mx.array]:
        """Encode text, then finish loading remaining components.

        In low_memory mode this ensures Gemma is freed before the
        transformer is loaded, keeping peak memory under control.
        """
        # Load text encoder + connector if not already loaded
        self._load_text_encoder()

        # Encode text
        video_embeds, audio_embeds = self._encode_text(prompt)
        # NOTE: mx.eval is MLX graph evaluation, NOT Python eval()
        mx.eval(video_embeds, audio_embeds)

        # Free text encoder before loading heavy components
        if self.low_memory:
            self.text_encoder = None
            self.feature_extractor = None
            aggressive_cleanup()

        # Load remaining components (DiT, VAE, vocoder)
        self.load()

        return video_embeds, audio_embeds

    def _encode_text(self, prompt: str) -> tuple[mx.array, mx.array]:
        """Encode prompt to (video, audio) embeddings via the PromptEncoder block.

        Inheritance-API thin wrapper. New code should prefer
        ``self.prompt_encoder(prompt)`` directly (composition style).
        """
        return self.prompt_encoder.encode(prompt)

    @staticmethod
    def _pre_denoise_flush(video_state: LatentState, audio_state: LatentState) -> None:
        """Force-materialise noised states before starting the denoise loop.

        Flushing here prevents the macOS Metal GPU watchdog from firing
        (``MTLCommandBufferErrorInternal`` code 14) when a large lazy graph
        — VAE encoding, conditioning blend, or noise addition — accumulates
        and is submitted as a single oversized command buffer.  Completing
        the graph in its own buffer before the denoise loop starts keeps each
        subsequent Metal command buffer within the watchdog window.

        Applies to all Apple Silicon Macs: the bug has been observed on
        M2 Max 64 GB and is not specific to the <=48 GB tier.
        """
        # NOTE: mx.eval is MLX graph evaluation, NOT Python eval()
        mx.eval(video_state.latent, video_state.clean_latent, audio_state.latent)

    @staticmethod
    def _save_waveform(waveform: mx.array, path: str, sample_rate: int = 48000) -> None:
        """Inheritance wrapper around :func:`utils._orchestration.save_waveform`."""
        from ltx_pipelines_mlx.utils._orchestration import save_waveform as _impl

        _impl(waveform, path, sample_rate=sample_rate)
