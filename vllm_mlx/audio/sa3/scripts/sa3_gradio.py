"""SA3 MLX — gradio web UI (Apple Silicon).

The MLX sibling of optimized/tensorRT/scripts/sa3_gradio.py, with every
generation mode wired (the TRT one only exposes text-to-audio):
  - Model picker: sm-music / sm-sfx / medium (hot-swap; models cache in unified
    memory, LRU-evicted — first switch loads weights, subsequent instant)
  - CFG 0-10 next to seconds/steps (0 = negative prompt takes over, 0.5 = halfway
    between prompts, 1 = off, >1 = extrapolate) + negative prompt/APG under Advanced
  - Audio-to-audio: guide audio + init_noise_level (whole clip starts from its
    latents); global sigma_max under Advanced for prompt-only generations
  - Inpainting: separate reference audio + start/end range sliders (kept bit-exact
    outside the range). Combinable with a2a: the regenerated span then starts
    from the guide audio instead of noise.
  - Spectrogram-as-player: 3-band tinted stereo mel spectrogram (numpy port — no
    torch) with a white playhead; click to seek. Only one audio plays at a time.
  - History: previous generations in a scroll panel (newest on top, scroll
    position anchored while new items arrive), each replayable.
  - Output options: Auto-play, Auto-download, Infinite Radio (pre-generates the
    next clip while the current one plays, swaps it in when playback ends),
    MP3 (V0, via ffmpeg) or WAV saving.

Launch:
    ./sa3-gradio                  # share=True by default, sm-music + same-s
    ./sa3-gradio --dit medium
    ./sa3-gradio --no-share       # local-only
"""
from __future__ import annotations
import argparse
import base64
import copy
import html as html_lib
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import wave
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPTS_DIR))

import mlx.core as mx  # noqa: E402

from sa3_mlx import (  # noqa: E402
    DIT_CHOICES, DECODER_CHOICES, ENCODER_CHOICES, T5GEMMA_NPZ_REL,
    SAMPLE_RATE, SAMPLES_PER_LATENT,
    load_dit, load_decoder, load_encoder, read_wav, patch_audio, _free_to_pool,
)
from models.defs.sa3_pipeline import (  # noqa: E402
    apply_prompt_padding, build_pingpong_schedule, sample_flow_pingpong,
    patched_decode, load_conditioner_from_npz,
)
from models.defs.lora_merge import (  # noqa: E402
    prepare_loras, apply_conditioner_lora, LoraError,
)
from mlx.utils import tree_flatten  # noqa: E402
from models.defs.t5gemma_mlx import T5Gemma  # noqa: E402
from weights import ensure_local  # noqa: E402
from spec import render_spectrogram_png  # noqa: E402

OUTPUT_DIR = REPO / "output" / "gradio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Local LoRA library: drop .safetensors under loras/<model>/ and they appear in
# the per-slot dropdown (scanned per selected DiT; hidden when empty).
LORAS_DIR = REPO / "loras"
LORA_DIR_NAMES = {"medium": "sa3-medium", "sm-sfx": "sa3-sm-sfx",
                  "sm-music": "sa3-sm-music"}
for _d in LORA_DIR_NAMES.values():
    (LORAS_DIR / _d).mkdir(parents=True, exist_ok=True)
_DD_NONE = ""   # dropdown placeholder value ("--- Choose a LoRA ---")


def _lora_dd_choices(dit_name: str) -> list:
    """Dropdown choices for ./loras/<model>/: placeholder + (name, abspath)."""
    d = LORAS_DIR / LORA_DIR_NAMES.get(dit_name, dit_name)
    hits = sorted(d.glob("*.safetensors")) if d.is_dir() else []
    return [("--- Choose a LoRA ---", _DD_NONE)] + [(p.name, str(p)) for p in hits]
MIN_SIGMA = 0.01
# MP3 (V0) saving needs ffmpeg; without it we save WAV and hide the choice.
FFMPEG = shutil.which("ffmpeg") is not None
FORMAT_MP3, FORMAT_WAV = "Save to MP3 (V0)", "Save to WAV"
DEFAULT_DECODERS = {name: cfg["default_decoder"] for name, cfg in DIT_CHOICES.items()}
# Trained max clip length per model (repo README model table).
MAX_SECONDS = {"sm-music": 120, "sm-sfx": 120, "medium": 380}

# MLX ≥0.31 registers GPU streams PER THREAD (ThreadLocalStream) while gradio
# runs each handler on a rotating anyio worker thread — cross-thread MLX use
# then dies with "There is no Stream(gpu, 0) in current thread". The categorical
# fix: ALL MLX work (pre-warm + every generation) runs on one dedicated owner
# thread; handlers submit to it and wait. This also serializes generations.
import concurrent.futures as _cf
_MLX_EXECUTOR = _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")


def mlx_call(fn, *args, **kwargs):
    """Run fn on the MLX owner thread and return its result (re-raises errors)."""
    return _MLX_EXECUTOR.submit(fn, *args, **kwargs).result()


def read_audio_any(path: str) -> np.ndarray:
    """(2, T) float32 @ 44.1 kHz from any common upload format.

    Layered: read_wav handles 16-bit/44.1k natively and shells out to ffmpeg
    for the rest when installed; if that fails (no ffmpeg), soundfile's bundled
    libsndfile decodes mp3/flac/ogg/24-bit/48 kHz directly, followed by a
    linear resample to 44.1 kHz."""
    try:
        return read_wav(path)
    except Exception:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)   # (T, ch)
        y = data.T
        y = np.stack([y[0], y[0]]) if y.shape[0] == 1 else y[:2]
        if sr != SAMPLE_RATE:
            in_len = y.shape[-1]
            new_len = int(round(in_len * SAMPLE_RATE / sr))
            scale = in_len / new_len
            pos = np.clip((np.arange(new_len) + 0.5) * scale - 0.5, 0, in_len - 1)
            y = np.stack([np.interp(pos, np.arange(in_len), ch) for ch in y])
        return np.ascontiguousarray(y, dtype=np.float32)


def condense_prompt(prompt: str) -> str:
    """Prompt → filename fragment (the main repo's verbose-naming rule):
    filesystem-special characters become hyphens, capped at 150 chars."""
    prompt = re.sub(r'[\\/:*?"<>|]', '-', prompt)[:150]
    return prompt or "_"


def verbose_basename(prompt, negative_prompt, cfg, sigma_max, seed) -> str:
    """prompt[.neg-…].cfg{scale}[.smx{σ}].{seed} — matches the main repo's
    gradio 'verbose' file naming (cfg segment only when cfg != 1)."""
    base = condense_prompt(prompt)
    if negative_prompt and negative_prompt.strip():
        base += ".neg-" + condense_prompt(negative_prompt.strip())
    if cfg != 1.0:
        base += f".cfg{cfg:g}"
    if sigma_max != 1.0:
        base += f".smx{sigma_max:g}"
    return f"{base}.{seed}"


def _save_wav(pcm_int16, out_path):
    """pcm_int16: (T, 2) int16 interleaved."""
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm_int16.tobytes())


# ── Model caches (unified memory; ~0.9-2.8 GB per DiT, 0.2-1.7 GB per codec) ──
# The MLX DiT bakes RoPE/mask lengths at load, so the DiT cache key includes
# T_lat. Everything else is length-independent and cached by name.
_t5: T5Gemma | None = None
_cond_cache: dict[str, tuple] = {}                       # dit -> (padding_emb, secs_embedder)
_dit_cache: dict[tuple[str, int], object] = {}           # (dit, T_lat) -> model
_dit_lru: list[tuple[str, int]] = []
_DIT_CACHE_MAX = 2
_dec_cache: dict[str, tuple] = {}                        # decoder -> (model, chunk_fn, chunk_cfg)
_enc_cache: dict[str, tuple] = {}                        # decoder -> (model, pad_modulo)
# Encoded-latent cache: rerunning a2a/inpainting with the same input audio skips
# the encoder entirely. Keyed on the file's CONTENT hash (gradio re-uploads get
# fresh temp paths) + encoder + latent length; values are ~160 KB fp32 arrays.
_latent_cache: dict[tuple, np.ndarray] = {}
_LATENT_CACHE_MAX = 32


def get_t5() -> T5Gemma:
    global _t5
    if _t5 is None:
        _t5 = T5Gemma.from_npz(str(ensure_local(T5GEMMA_NPZ_REL)))
    return _t5


def get_conditioner(dit_name: str):
    if dit_name not in _cond_cache:
        _cond_cache[dit_name] = load_conditioner_from_npz(
            str(ensure_local(DIT_CHOICES[dit_name]["ckpt"])), prefix="cond.")
    return _cond_cache[dit_name]


def _lora_sig(specs, num_steps: int) -> tuple:
    """Identity of a LoRA config on a cached DiT: file (path+mtime+size —
    gradio temp uploads keep their path within a session), strength, raw step
    range, and the schedule length (the plan's step sets depend on it)."""
    if not specs:
        return ()
    sig = []
    for s in specs:
        st = os.stat(s["path"])
        sig.append((s["path"], st.st_mtime_ns, st.st_size,
                    float(s["strength"]), s["steps"], int(num_steps)))
    return tuple(sig)


def _reconcile_lora(model, dit_name: str, specs, num_steps: int) -> None:
    """Apply / replace / remove the LoRA config on a cached base DiT, in place.

    A cached model always carries base weights + its current LoraStepPlan
    state. Changing config = exact-inverse clear back to base, then a fresh
    plan built from the live module weights (gate_all=True — a permanent merge
    couldn't be undone in place). One boundary transition each way (~26/80 ms),
    so tweaking strength or step ranges between generations never reloads the
    DiT. Same config → no-op (before_step(0) rewinds the plan every run)."""
    sig = _lora_sig(specs, num_steps)
    if sig == getattr(model, "_gr_lora_sig", ()):
        return
    old = getattr(model, "_lora_plan", None)
    if old is not None:
        old.clear()
    model._lora_plan, model._gr_lora_sig = None, ()
    if not specs:
        return
    wd_view = dict(tree_flatten(model.parameters()))
    try:
        plan = prepare_loras(wd_view, specs, num_steps=num_steps,
                             log=lambda m: print(f"  {m}"), gate_all=True)
    except LoraError as e:
        raise LoraError(f"{e} — selected model: {dit_name}") from None
    if plan is None:
        return
    plan.attach(model)
    # prepare_loras applied the step-1 state to the DICT slots (fp32) — point
    # the live modules at the new arrays, in the model dtype.
    swaps = []
    for lkey, mod in plan.mods.items():
        if wd_view[lkey] is not mod.weight:
            mod.weight = wd_view[lkey].astype(mod.weight.dtype)
            swaps.append(mod.weight)
    if swaps:
        mx.eval(swaps)
    model._lora_plan, model._gr_lora_sig = plan, sig


def get_dit(dit_name: str, T_lat: int, dtype, lora_specs=None, num_steps: int = 8):
    key = (dit_name, T_lat)
    if key in _dit_cache:
        if key in _dit_lru:
            _dit_lru.remove(key)
        _dit_lru.append(key)
        model, load_ms = _dit_cache[key], 0.0
    else:
        while len(_dit_cache) >= _DIT_CACHE_MAX:
            oldest = _dit_lru.pop(0)
            print(f"  ← LRU-evicting DiT {oldest}")
            _dit_cache.pop(oldest, None)
            _free_to_pool()
        t0 = time.time()
        model, _ = load_dit(dit_name, T_lat=T_lat, dtype=dtype)
        load_ms = (time.time() - t0) * 1000
        _dit_cache[key] = model
        _dit_lru.append(key)
    _reconcile_lora(model, dit_name, lora_specs, num_steps)
    return model, load_ms


def get_decoder(decoder_name: str):
    if decoder_name not in _dec_cache:
        _dec_cache[decoder_name] = load_decoder(decoder_name, mx.float32)
    return _dec_cache[decoder_name]


def get_encoder(decoder_name: str):
    if decoder_name not in _enc_cache:
        _enc_cache[decoder_name] = load_encoder(decoder_name, mx.float32)
    return _enc_cache[decoder_name]


# ── One full generation (mirrors sa3_mlx.py main(), all modes) ─────────────
def run_generation(dit_name: str, decoder_name: str, prompt: str,
                   negative_prompt: str, seconds: float, steps: int,
                   seed: int, cfg: float, apg: float, sigma_max: float,
                   a2a_audio_path: str | None = None,
                   inpaint_audio_path: str | None = None,
                   inpaint_range_sec=None,
                   lora_specs=None):
    """Returns (audio_np (2,T) float32, timings dict).

    a2a and inpainting are independent and combinable:
      - a2a_audio_path: the whole generation STARTS from this audio's latents
        (noise = lat*(1-σmax) + noise*σmax)
      - inpaint_audio_path + inpaint_range_sec: kept bit-exact outside the range
        (local_add_cond context + per-step paste-back)
      - both: the inpainted span regenerates FROM the a2a guide instead of pure noise
    """
    t = {}
    mx.set_default_device(mx.gpu)   # belt-and-braces; normally on the MLX owner thread
    dtype = mx.float16   # MLX canonical DiT dtype
    T_lat = max(1, math.ceil(seconds * SAMPLE_RATE / SAMPLES_PER_LATENT))

    # 1. T5Gemma
    t0 = time.time()
    enc = get_t5()
    embeds, mask = enc.encode([prompt], max_len=256)
    mx.eval(embeds, mask)
    t["t5_ms"] = (time.time() - t0) * 1000

    # 2. Conditioning
    t0 = time.time()
    padding_emb, secs_embedder = get_conditioner(dit_name)
    if lora_specs:
        # Seconds-conditioner deltas (underfit adapters carry one) apply to the
        # whole generation — conditioning runs once, before sampling. Work on a
        # copy so the cached embedder stays pristine.
        W_lora, _n_cond = apply_conditioner_lora(secs_embedder.W, lora_specs)
        if _n_cond:
            secs_embedder = copy.copy(secs_embedder)
            secs_embedder.W = W_lora
    embeds = embeds.astype(dtype)
    embeds_padded = apply_prompt_padding(embeds, mask, padding_emb.astype(dtype))
    seconds_embed = secs_embedder(seconds).astype(dtype)
    cross_attn = mx.concatenate([embeds_padded, seconds_embed], axis=1)
    global_cond = seconds_embed[:, 0, :]
    null_cross_attn = None
    if cfg != 1.0:
        if negative_prompt and negative_prompt.strip():
            neg_embeds, neg_mask = enc.encode([negative_prompt.strip()], max_len=256)
            mx.eval(neg_embeds, neg_mask)
            neg_padded = apply_prompt_padding(neg_embeds.astype(dtype), neg_mask,
                                              padding_emb.astype(dtype))
            null_cross_attn = mx.concatenate([neg_padded, seconds_embed], axis=1)
        else:
            null_cross_attn = mx.zeros_like(cross_attn)
        mx.eval(null_cross_attn)
    mx.eval(cross_attn, global_cond)
    t["cond_ms"] = (time.time() - t0) * 1000

    # 3a. encode the provided audio inputs → latents (each is optional).
    # Latents are memoized by (file content, encoder, T_lat) so re-running
    # a2a/inpainting with the same input audio skips the encoder.
    def encode_audio(path):
        import hashlib
        ck = (hashlib.sha1(Path(path).read_bytes()).hexdigest(), decoder_name, T_lat)
        if ck in _latent_cache:
            t["enc_cached"] = t.get("enc_cached", 0) + 1
            return mx.array(_latent_cache[ck]).astype(dtype)
        enc_model, pad_mod = get_encoder(decoder_name)
        enc_T_lat = T_lat
        if (T_lat * 16) % pad_mod != 0:
            enc_T_lat = math.ceil((T_lat * 16) / pad_mod) * pad_mod // 16
        target_samples = enc_T_lat * SAMPLES_PER_LATENT
        audio_np = read_audio_any(path)
        if audio_np.shape[-1] >= target_samples:
            audio_np = audio_np[:, :target_samples]
        else:
            audio_np = np.pad(audio_np, ((0, 0), (0, target_samples - audio_np.shape[-1])))
        patches_np = patch_audio(audio_np[None, ...], patch_size=256)
        lat = enc_model(mx.array(patches_np))[..., :T_lat]
        mx.eval(lat)
        while len(_latent_cache) >= _LATENT_CACHE_MAX:
            _latent_cache.pop(next(iter(_latent_cache)))
        _latent_cache[ck] = np.array(lat.astype(mx.float32))
        return lat.astype(dtype)

    a2a_latents = None      # start-point guide (whole clip)
    ctx_latents = None      # inpaint context (kept outside the range)
    t["enc_ms"] = 0.0
    if a2a_audio_path:
        t0 = time.time()
        a2a_latents = encode_audio(a2a_audio_path)
        t["enc_ms"] += (time.time() - t0) * 1000
    if inpaint_audio_path and inpaint_range_sec is not None:
        t0 = time.time()
        ctx_latents = encode_audio(inpaint_audio_path)
        t["enc_ms"] += (time.time() - t0) * 1000

    # 3b. DiT + pingpong sample
    dit_model, t["dit_load_ms"] = get_dit(dit_name, T_lat, dtype,
                                          lora_specs=lora_specs, num_steps=steps)
    sigmas = build_pingpong_schedule(steps, sigma_max=sigma_max, use_logsnr_shift=True)

    key = mx.random.key(seed)
    pure_noise = mx.random.normal((1, 256, T_lat), dtype=dtype, key=key)
    # a2a start-point mix is independent of inpainting: when both are set, the
    # inpainted span regenerates FROM the guide audio instead of pure noise
    # (the kept span is pasted back from ctx_latents every step regardless).
    if a2a_latents is not None:
        noise = a2a_latents * (1.0 - sigma_max) + pure_noise * sigma_max
    else:
        noise = pure_noise
    mx.eval(noise)

    local_add_cond = None
    paste_back = None
    if ctx_latents is not None:
        s0 = max(0, int(round(inpaint_range_sec[0] * SAMPLE_RATE / SAMPLES_PER_LATENT)))
        s1 = min(T_lat, int(round(inpaint_range_sec[1] * SAMPLE_RATE / SAMPLES_PER_LATENT)))
        mask_np = np.ones((1, 1, T_lat), dtype=np.float32)
        mask_np[:, :, s0:s1] = 0.0
        keep = mx.array(mask_np)
        masked_input = ctx_latents.astype(mx.float32) * keep
        local_add_cond = mx.concatenate([keep, masked_input], axis=1).transpose(0, 2, 1).astype(dtype)
        paste_back = (ctx_latents, keep)

    def model_fn(x, tt):
        if cfg == 1.0:
            return dit_model(x, tt, cross_attn, global_cond, local_add_cond=local_add_cond)
        x2 = mx.concatenate([x, x], axis=0)
        t2 = mx.concatenate([tt, tt], axis=0)
        cross2 = mx.concatenate([cross_attn, null_cross_attn], axis=0)
        global2 = mx.concatenate([global_cond, global_cond], axis=0)
        lac2 = None if local_add_cond is None else mx.concatenate([local_add_cond, local_add_cond], axis=0)
        v_batched = dit_model(x2, t2, cross2, global2, local_add_cond=lac2)
        cond_v, uncond_v = mx.split(v_batched, 2, axis=0)
        sigma = tt.reshape(-1, 1, 1).astype(mx.float32)
        cond_d = x.astype(mx.float32) - cond_v.astype(mx.float32) * sigma
        uncond_d = x.astype(mx.float32) - uncond_v.astype(mx.float32) * sigma
        diff = cond_d - uncond_d
        # cfg < 1 is the INTERPOLATION regime: cfg_d = lerp(uncond_d, cond_d, cfg),
        # so 0 = pure negative/uncond branch, 0.5 = halfway between both prompts.
        # APG's orthogonal projection would bend that line — it only applies to the
        # extrapolation regime (cfg > 1) it was designed for.
        if apg <= 0.0 or cfg < 1.0:
            cfg_diff = diff
        else:
            norm = mx.sqrt((cond_d * cond_d).sum(axis=(-2, -1), keepdims=True))
            unit = cond_d / mx.maximum(norm, 1e-8)
            parallel = (diff * unit).sum(axis=(-2, -1), keepdims=True) * unit
            diff_orth = diff - parallel
            cfg_diff = diff_orth if apg >= 1.0 else (apg * diff_orth + (1.0 - apg) * diff)
        cfg_d = cond_d + (cfg - 1.0) * cfg_diff
        cfg_v = (x.astype(mx.float32) - cfg_d) / sigma
        return cfg_v.astype(x.dtype)

    t0 = time.time()
    _plan = getattr(dit_model, "_lora_plan", None)
    latents = sample_flow_pingpong(model_fn, noise, sigmas, seed=seed + 1,
                                   paste_back=paste_back,
                                   before_step=_plan.sync if _plan else None)
    mx.eval(latents)
    t["sample_ms"] = (time.time() - t0) * 1000

    # 4. Decode (fp32 codec; parity-aware chunk dispatch from sa3_mlx.py)
    t0 = time.time()
    decoder, chunk_fn, (chunk, ovl) = get_decoder(decoder_name)
    latents_fp32 = latents.astype(mx.float32)
    kernel = chunk + 2 * ovl
    if T_lat > kernel:
        patches = chunk_fn(decoder, latents_fp32, chunk, ovl)
    elif T_lat % 2 == 0:
        patches = decoder(latents_fp32)
    elif T_lat > 6:
        patches = chunk_fn(decoder, latents_fp32, 2, 2)
    else:
        latents_even = mx.concatenate([latents_fp32, latents_fp32[..., -1:]], axis=-1)
        patches = decoder(latents_even)[..., : T_lat * 16]
    mx.eval(patches)
    t["decode_ms"] = (time.time() - t0) * 1000

    # 5. Unpatch + trim
    audio = patched_decode(patches, patch_size=256, channels=2)
    mx.eval(audio)
    audio_np = np.array(audio.astype(mx.float32))[0]
    requested = int(round(seconds * SAMPLE_RATE))
    if audio_np.shape[-1] > requested:
        audio_np = audio_np[..., :requested]

    t["T_lat"] = T_lat
    t["samples"] = audio_np.shape[-1]
    t["inference_ms"] = sum(t.get(k, 0) for k in
                            ("t5_ms", "cond_ms", "enc_ms", "dit_load_ms", "sample_ms", "decode_ms"))
    t["realtime"] = (audio_np.shape[-1] / SAMPLE_RATE) / max(t["inference_ms"] / 1000, 1e-9)
    return audio_np, t


# ── HTML rendering (inline handlers only — gradio HTML runs attributes, not
# <script> tags) ────────────────────────────────────────────────────────────
# One-at-a-time playback. Chrome does NOT stop a playing <audio> when it's
# removed from the DOM (verified via CDP: swapped-out elements keep playing as
# detached ghosts), and querySelectorAll can't see detached nodes — so every
# player registers itself in window._sa3All on play, and pause-others sweeps
# that registry (reaching ghosts) plus the document. Entries that are detached
# AND paused drop from the registry so ghosts can be garbage-collected.
# A DETACHED element's play event must never silence the living player (the
# swap can briefly leave a superseded copy whose pending play() resolves late).
# gradio's slider track fill lives in an inline CSS var (--range_progress) that
# is only written on user input / first mount — a slider REmounted by a
# visibility toggle (LoRA slots on model switch / add) comes back without it and
# renders a blank track. Recompute it client-side after any remount batch;
# idempotent for healthy sliders (same formula gradio uses).
_JS_FIX_SLIDERS = (
    "() => setTimeout(() => {"
    "document.querySelectorAll('input[type=range]').forEach(r => {"
    "const min = +r.min || 0, max = +r.max || 100, v = +r.value;"
    "if (max > min) r.style.setProperty('--range_progress',"
    "((v - min) / (max - min) * 100) + '%');"
    "});}, 150)"
)

_JS_PAUSE_OTHERS = (
    "if(this.isConnected){"
    "var t=this;var L=window._sa3All=(window._sa3All||[]);"
    "if(L.indexOf(t)<0)L.push(t);"
    "L.forEach(function(o){if(o!==t){try{o.pause()}catch(e){}}});"
    "document.querySelectorAll('audio').forEach(function(o){if(o!==t)o.pause();});"
    "window._sa3All=L.filter(function(o){return o===t||o.isConnected;});}"
    "else{this.pause();}")
_JS_PLAYHEAD = ("var p=this.closest('.blk').querySelector('.ph');"
                "if(p&&this.duration)p.style.left=(this.currentTime/this.duration*100)+'%';")
# Main-player position ledger (for Hotswap): every timeupdate/play/pause stamps
# the current position so a freshly swapped-in element can resume exactly there.
# Guards (all verified via CDP against the live page):
#  - isConnected: teardown fires a pause AND a final timeupdate on the removed
#    element — detached elements never write the ledger.
#  - unresumed hotswap candidates (data-hs set, hsd not yet) don't write either:
#    with autoplay, the NEW element's 'play' event fires BEFORE loadedmetadata
#    and would stamp t≈0 over the old clip's position right before the resume
#    handler reads it.
_JS_POS_RECORD = ("if(this.isConnected&&!(this.dataset.hs==='1'&&!this.dataset.hsd))"
                  "{window._sa3Pos={t:this.currentTime,playing:!this.paused,ts:Date.now()};}")
# timeupdate variant: a PAUSED element only fires timeupdate on seeks — e.g. the
# resume handler's own currentTime assignment. Stamping playing:false there
# poisons the ledger for any second render of the same clip (gradio sometimes
# renders a component update twice), which froze the handoff chain. While
# paused, only real pause events may write.
_JS_POS_RECORD_TU = ("if(this.isConnected&&!this.paused&&"
                     "!(this.dataset.hs==='1'&&!this.dataset.hsd))"
                     "{window._sa3Pos={t:this.currentTime,playing:true,ts:Date.now()};}")
# Resilient play: Chrome's autoplay policy can reject programmatic play() once
# the transient user activation from the Generate click has expired (seconds),
# leaving the clip correctly positioned but paused. Try, retry shortly after,
# and as a last resort arm a one-shot listener so the user's next click or
# keypress anywhere resumes playback.
# Every attempt re-checks isConnected: a superseded (detached) element must
# never revive itself from a queued retry and fight the current player.
_JS_TRY_PLAY = (
    "var A=this;A.play().catch(function(){setTimeout(function(){"
    "if(!A.isConnected)return;"
    "A.play().catch(function(){console.warn('play blocked by autoplay policy — "
    "will resume on next interaction');"
    "var f=function(){if(A.isConnected)A.play();"
    "document.removeEventListener('pointerdown',f,true);"
    "document.removeEventListener('keydown',f,true);};"
    "document.addEventListener('pointerdown',f,true);"
    "document.addEventListener('keydown',f,true);});},150);});")
# Hotswap resume: if the previous main audio was playing when this one arrived,
# jump to its position (+ the split-second since the last stamp) and keep going;
# beyond the new clip's duration -> start at zero. Guarded (hsd) so it applies
# once, on whichever of loadedmetadata/canplay fires first.
_JS_HOTSWAP = ("if(this.isConnected&&this.dataset.hs==='1'&&!this.dataset.hsd&&this.duration){"
               "this.dataset.hsd=1;var s=window._sa3Pos;"
               "this.dataset.dbg=s?(s.playing?'playing@'+s.t.toFixed(2):'notplaying@'+s.t.toFixed(2)):'noledger';"
               "if(s&&s.playing){var tt=s.t+(Date.now()-s.ts)/1000;"
               "this.currentTime=(tt<this.duration)?tt:0;" + _JS_TRY_PLAY + "}}")
_JS_SEEK = ("var a=this.closest('.blk').querySelector('audio');"
            "var r=this.getBoundingClientRect();"
            "if(a&&a.duration){a.currentTime=(event.clientX-r.left)/r.width*a.duration;a.play();}")
_JS_PROMOTE = ("var b=document.getElementById('sa3-promote');"
               "if(b){(b.tagName==='BUTTON'?b:b.querySelector('button')||b).click();}")
# Scroll anchoring for the history panel: onscroll continuously records which
# item sits at the viewport top (+offset); after every re-render a hidden
# bootstrap <img onerror> restores that anchor — stick-to-top when at top,
# otherwise keep hovering over the same old item as new ones prepend.
_JS_SCROLL_RECORD = (
    "var c=this,k=null,off=0,ch=c.querySelectorAll('[data-key]');"
    "for(var i=0;i<ch.length;i++){if(ch[i].offsetTop+ch[i].offsetHeight>c.scrollTop)"
    "{k=ch[i].getAttribute('data-key');off=ch[i].offsetTop-c.scrollTop;break}}"
    "window._sa3S={atTop:c.scrollTop<8,key:k,off:off};")
_JS_SCROLL_RESTORE = (
    "var c=document.getElementById('sa3-hist');var s=window._sa3S||{};"
    "if(c){var el=s.key?c.querySelector('[data-key=&quot;'+s.key+'&quot;]'):null;"
    "if(el&&!s.atTop){c.scrollTop=el.offsetTop-(s.off||0)}else{c.scrollTop=0}}"
    "this.remove();")


def _ago(ts) -> str:
    if not ts:
        return ""
    d = max(0.0, time.time() - ts)
    if d < 10:
        return "just now"
    if d < 60:
        return f"{int(d)}s ago"
    if d < 3600:
        return f"{int(d // 60)}m ago"
    if d < 86400:
        return f"{int(d // 3600)}h ago"
    return f"{int(d // 86400)}d ago"


def _lora_specs_from_ui(vals, notes, num_steps):
    """Turn the 3 LoRA slots' flat values (file, dropdown, strength, min, max)×3
    into lora_merge specs. A slot's adapter is the dropped file or, failing
    that, the dropdown pick from ./loras/<model>/; empty slots are skipped. A
    min slider at 1 / max slider at (or beyond) the schedule length mean 'from
    the start' / 'to the last step', so a maxed-out range slider keeps meaning
    'all steps' when Steps changes. Half-sane inputs get a note and the most
    permissive interpretation (never an error — matches the app's
    permissive-by-design generation policy)."""
    specs = []
    for i in range(3):
        path, picked, strength, mn, mxx = vals[5 * i: 5 * i + 5]
        if not path:
            path = picked if picked and picked != _DD_NONE else None
        if not path:
            continue
        mn = int(mn) if mn and int(mn) > 1 else None
        mxx = int(mxx) if mxx and int(mxx) < num_steps else None
        if mn is not None and mn > num_steps:
            notes.append(f"LoRA {i + 1}: min step {mn} is beyond the "
                         f"{num_steps}-step schedule — adapter inactive")
        if mn is not None and mxx is not None and mn > mxx:
            notes.append(f"LoRA {i + 1}: min step {mn} > max step {mxx} — adapter skipped")
            continue
        steps = None if (mn is None and mxx is None) else (mn, mxx)
        specs.append({"path": path, "strength": float(strength), "steps": steps})
    return specs or None


def _lora_disp(specs) -> str:
    """Short human tag per adapter: 'plini×0.8 @2-8, rain @1-4'."""
    tags = []
    for s in specs or []:
        name = Path(s["path"]).stem
        if len(name) > 25:
            name = name[:24] + "…"
        tag = name
        if s["strength"] != 1.0:
            tag += f"×{s['strength']:g}"
        if s["steps"]:
            lo, hi = s["steps"]
            tag += f" @{lo or 1}-{hi if hi is not None else 'end'}"
        tags.append(tag)
    return ", ".join(tags)


def _meta_suffix(entry) -> str:
    """' · cfg 1.5 · noise 0.92 · audio2audio · inpainting' — only the non-defaults.
    The sigma tag reads 'noise' for a2a runs (it IS the init_noise_level there),
    'smx' otherwise."""
    parts = []
    cfg = entry.get("cfg", 1.0)
    smx = entry.get("smx", 1.0)
    mode = entry.get("mode", "")
    is_a2a = "a2a" in mode or mode == "audio-to-audio"
    dit = entry.get("dit", "")
    if dit:
        parts.append({"medium": "med", "sm-music": "sm-mus", "sm-sfx": "sm-sfx"}.get(dit, dit))
    if cfg != 1.0:
        parts.append(f"cfg {cfg:g}")
    if smx != 1.0:
        parts.append(f"{'noise' if is_a2a else 'smx'} {smx:g}")
    if is_a2a:
        parts.append("audio2audio")
    if "inpaint" in mode:
        parts.append("inpainting")
    if entry.get("lora"):
        parts.append(f"lora {entry['lora']}")
    return "".join(f" · {p}" for p in parts)


def _neg_disp(entry) -> str:
    """Labeled negative prompt, shown only when it actually acted (cfg != 1)."""
    neg = (entry.get("neg") or "").strip()
    if neg and entry.get("cfg", 1.0) != 1.0:
        return f' · <span style="opacity:0.75">neg: {html_lib.escape(neg)}</span>'
    return ""


def render_player(entry, *, small=False, autoplay=False, autodl=False, radio=False,
                  bg=None, advance=False, loop=False, hotswap=False):
    """One self-contained player block: audio + caption + seekable spectrogram
    with playhead. Global one-at-a-time playback via onplay pause-others.
    small: audio + spectrogram side by side (half width each), 'Xm ago' caption.
    advance: on ended, hop to the next history item's audio (Auto-play).
    loop: native loop attribute — finished audio restarts (suppresses ended).
    hotswap: main slot only — resume at the previous clip's position on arrival."""
    # assemble per-event handler bodies (a duplicated attribute name would
    # silently drop one handler, so each event is emitted exactly once)
    on_canplay = []
    if small:
        attrs = f'onplay="{_JS_PAUSE_OTHERS}" ontimeupdate="{_JS_PLAYHEAD}"'
    else:
        attrs = (f'onplay="{_JS_PAUSE_OTHERS}{_JS_POS_RECORD}" '
                 f'onpause="{_JS_POS_RECORD}" '
                 f'ontimeupdate="{_JS_PLAYHEAD}{_JS_POS_RECORD_TU}"')
    if hotswap and not small:
        attrs += f' data-hs="1" onloadedmetadata="{_JS_HOTSWAP}"'
        on_canplay.append(_JS_HOTSWAP)   # fallback if loadedmetadata already passed
    if autoplay and not small:
        # the autoplay attribute is subject to the same policy — rescue a
        # policy-paused clip (radio transitions, long generations)
        on_canplay.append("if(this.paused&&!(this.dataset.hs==='1'&&!this.dataset.hsd))"
                          "{" + _JS_TRY_PLAY + "}")
    if autodl:
        js_name = entry["name"].replace("\\", "").replace("'", "\\'")
        on_canplay.append("if(!this.dataset.dld){this.dataset.dld=1;"
                          "var l=document.createElement('a');l.href=this.src;"
                          f"l.download='{js_name}';l.click();}}")
    if on_canplay:
        attrs += f' oncanplay="{"".join(on_canplay)}"'
    if loop:
        attrs += " loop"
    elif radio:
        attrs += f' onended="{_JS_PROMOTE}"'
    elif advance:
        attrs += (' onended="var b=this.closest(\'.blk\');'
                  "var n=b?b.nextElementSibling:null;"
                  "while(n&&!n.classList.contains('blk'))n=n.nextElementSibling;"
                  "if(n){var a=n.querySelector('audio');if(a)a.play();}\"")
    auto = "autoplay " if autoplay else ""
    # Serve audio via gradio's file route instead of a data: URI — the URL ends
    # with the real (verbose) filename, so right-click "Save audio as…" offers
    # it instead of download.mp3, and the page stays light as history grows.
    src = "gradio_api/file=" + urllib.parse.quote(entry["path"], safe="/")
    audio_el = (f'<audio controls {auto}style="width:100%" {attrs} '
                f'src="{src}"></audio>')
    prompt_disp = html_lib.escape(entry["prompt"]) or "<i>(no prompt)</i>"

    spec_core = ""
    if entry.get("spec_b64"):
        height = "height:56px;" if small else ""
        tip = ("3-band tinted stereo mel · red=bass / green=mid / blue=high · "
               "L top, R bottom · click to seek")
        spec_core = (f'<div style="position:relative; cursor:pointer" title="{tip}" '
                     f'onclick="{_JS_SEEK}">'
                     f'<img src="data:image/png;base64,{entry["spec_b64"]}" '
                     f'style="width:100%; {height} display:block; image-rendering:pixelated; '
                     f'border:1px solid #333" alt="spectrogram"/>'
                     f'<div class="ph" style="position:absolute; top:0; bottom:0; left:0%; '
                     f'width:2px; background:#fff; pointer-events:none; '
                     f'box-shadow:0 0 4px rgba(0,0,0,.8)"></div>'
                     f'</div>')

    if small:
        row = (f'<div style="display:flex; gap:8px; align-items:center">'
               f'<div style="flex:1; min-width:0">{audio_el}</div>'
               f'<div style="flex:1; min-width:0">{spec_core}</div></div>')
        cap = (f'<div style="font-size:0.8em; margin:2px 0; color:#888">'
               f'{_ago(entry.get("ts"))} · {prompt_disp}{_neg_disp(entry)} · seed {entry["seed"]}'
               f'{_meta_suffix(entry)}</div>')
        style = f"padding:6px 8px;{' background:' + bg + ';' if bg else ''}"
        return (f'<div class="blk" data-key="{entry["key"]}" style="{style}">'
                f'{row}{cap}</div>')

    return (f'<div class="blk" data-key="{entry["key"]}">'
            f'{audio_el}{spec_core}</div>')


def render_history(hist, advance=False, loop=False):
    if not hist:
        return ""
    # zebra striping instead of separators: light grey vs medium grey rows
    items = "".join(
        render_player(e, small=True, advance=advance, loop=loop,
                      bg="rgba(127,127,127,0.24)" if i % 2 else "rgba(127,127,127,0.08)")
        for i, e in enumerate(hist))
    boot = f'<img src="data:," style="display:none" onerror="{_JS_SCROLL_RESTORE}"/>'
    return (f'<div style="font-weight:600; margin-top:14px">Previous generations ({len(hist)})</div>'
            f'<div id="sa3-hist" onscroll="{_JS_SCROLL_RECORD}" '
            f'style="max-height:480px; overflow-y:auto; position:relative; margin-top:6px; '
            f'padding-right:6px">{boot}{items}</div>')


def render_queue_status(entry=None, generating=False):
    """Subdued status chip for the Infinite Radio queue — no audio/spectrogram,
    just Generating…/Ready (the swap uses the entry held in server state)."""
    if generating:
        body = "generating…"
    elif entry is not None:
        p = html_lib.escape(entry["prompt"]) or "<i>(no prompt)</i>"
        body = f"ready — {p}{_neg_disp(entry)} · seed {entry['seed']}{_meta_suffix(entry)}"
    else:
        return ""
    return (f'<div style="margin-top:2px; padding:6px 10px; '
            f'background:rgba(127,127,127,0.12); border-radius:6px; '
            f'color:#888; font-size:0.85em">'
            f'<b>Queued next</b> · {body}</div>')


# ── Gradio UI ──────────────────────────────────────────────────────────────
def build_ui(initial_dit: str, initial_decoder: str, *, share: bool,
             default_seconds: float, default_steps: int,
             preload_loras: tuple = (), server_port: int | None = None):
    import gradio as gr
    import random as _random

    # Pre-warm the initial pipeline so the first click is fast — ON the MLX
    # owner thread, so all stream/model state lives where generations run.
    warm_T = max(1, math.ceil(default_seconds * SAMPLE_RATE / SAMPLES_PER_LATENT))
    print(f"  pre-warming {initial_dit}+{initial_decoder} (T_lat={warm_T})...")

    def _warm():
        get_t5(); get_conditioner(initial_dit)
        get_dit(initial_dit, warm_T, mx.float16); get_decoder(initial_decoder)
    mlx_call(_warm)

    def on_dit_change(dit_name, cur_seconds, cur_steps, prev, mem, vis, *slot_vals):
        """Model switch: pair the decoder, clamp seconds, and swap the LoRA
        panel to the new model — the outgoing model's slots (files, picks,
        strengths, ranges, visibility) are snapshotted into per-model memory
        and the incoming model's snapshot is restored (or an empty panel if it
        has none). Dropdowns rescan loras/<model>/ (hidden when empty); a
        remembered pick that no longer exists on disk falls back to the
        placeholder."""
        max_s = MAX_SECONDS.get(dit_name, 120)
        ch = _lora_dd_choices(dit_name)
        valid = {v for _, v in ch}
        lbl = f"…or pick from loras/{LORA_DIR_NAMES.get(dit_name, dit_name)}/"

        mem = dict(mem or {})
        mem[prev] = {"vis": list(vis),
                     "slots": [list(slot_vals[5 * i: 5 * i + 5]) for i in range(3)]}
        snap = mem.get(dit_name)
        if snap:
            nvis = list(snap["vis"])
            slots = [list(s) for s in snap["slots"]]
            for s in slots:
                if s[1] not in valid:
                    s[1] = _DD_NONE
            # a slot with an adapter must never be an invisible-but-active one
            nvis = [v or bool(s[0] or s[1] != _DD_NONE)
                    for v, s in zip(nvis, slots)]
        else:
            nvis = [False, False, False]
            slots = [[None, _DD_NONE, 1.0, 1, int(cur_steps)] for _ in range(3)]

        dd_ups = [gr.update(choices=ch, value=slots[i][1], visible=len(ch) > 1,
                            label=lbl) for i in range(3)]
        srow_ups = [gr.update(visible=bool(slots[i][0] or slots[i][1] != _DD_NONE))
                    for i in range(3)]
        row_ups = [gr.update(visible=v) for v in nvis]
        return (gr.update(value=DEFAULT_DECODERS.get(dit_name, "same-s")),
                gr.update(maximum=max_s, value=min(cur_seconds, max_s)),
                *dd_ups, *srow_ups, *row_ups,
                gr.update(visible=not all(nvis)), nvis, dit_name, mem,
                slots[0][0], slots[1][0], slots[2][0],
                *(slots[i][j] for i in range(3) for j in (2, 3, 4)))

    def _generate_entry(dit_name, decoder_name, prompt, negative_prompt,
                        seconds, steps, seed_text, cfg, apg, sigma_max, init_noise,
                        a2a_audio, inpaint_audio, inp_start, inp_end,
                        output_opts, file_format, *lora_vals):
        """Run one generation and package it as a history entry.
        Returns (entry, None) or (None, error_message)."""
        prompt = (prompt or "").strip()
        # Permissive by design: a generation should succeed with whatever IS set.
        # Half-configured features are ignored with a visible note, never an error.
        notes = []
        lora_specs = (_lora_specs_from_ui(lora_vals, notes, int(steps))
                      if lora_vals else None)
        # blank or -1 → random seed, kept small (1-9999) for readability
        try:
            seed = int(seed_text.strip()) if seed_text and seed_text.strip() else -1
        except ValueError:
            seed = -1
            notes.append("seed wasn't an integer — used a random one")
        if seed == -1:
            seed = _random.randint(1, 9999)
        # backstop for API callers bypassing the slider maxima
        max_s = MAX_SECONDS.get(dit_name, 120)
        if seconds > max_s:
            notes.append(f"seconds clamped to {dit_name}'s trained max ({max_s:g}s)")
            seconds = float(max_s)
        # sigma_max governs every generation's schedule start; when guide audio is
        # present (a2a), init_noise_level overrides it (parent-repo gradio semantics).
        sigma_max = float(init_noise) if a2a_audio else float(sigma_max)
        if sigma_max < MIN_SIGMA:
            which = "init_noise_level" if a2a_audio else "sigma_max"
            notes.append(f"{which} 0 runs at {MIN_SIGMA} (model is undefined at t≈0)"
                         + (" — output ≈ the re-encoded input" if a2a_audio else ""))
            sigma_max = MIN_SIGMA

        inpaint_range = None
        if inpaint_audio and inp_end > inp_start and inp_start < seconds:
            if inp_end > seconds:
                notes.append(f"inpaint end clamped to the clip length ({seconds:g}s)")
            inpaint_range = (float(inp_start), float(min(inp_end, seconds)))
        elif inpaint_audio:
            notes.append("inpainting ignored — set the start/end range sliders")
        elif inp_end > inp_start:
            notes.append("inpaint range ignored — no reference audio uploaded")

        opts = output_opts or []
        if ("Infinite Radio" in opts and seed_text and seed_text.strip()
                and seed_text.strip() != "-1"):
            notes.append("Infinite Radio with a fixed seed repeats the same clip — "
                         "clear the seed for endless variety")

        mode = ("a2a+inpaint" if (a2a_audio and inpaint_range) else
                "inpaint" if inpaint_range else
                "audio-to-audio" if a2a_audio else "text-to-audio")
        try:
            audio_np, t = mlx_call(
                run_generation,
                dit_name, decoder_name, prompt, negative_prompt or "",
                float(seconds), int(steps), seed, float(cfg), float(apg),
                float(sigma_max), a2a_audio or None, inpaint_audio or None,
                inpaint_range, lora_specs)
        except LoraError as e:
            return None, f"LoRA error: {e}"
        except Exception as e:
            return None, f"error: {type(e).__name__}: {e}"
        if not np.isfinite(audio_np).all():
            return None, "error: model produced non-finite audio (try a higher σmax or different seed)"

        pcm = (np.clip(audio_np, -1, 1) * 32767.0).astype(np.int16).T   # (T, 2)
        basename = verbose_basename(prompt, negative_prompt, cfg, sigma_max, seed)
        out_path = OUTPUT_DIR / f"{basename}.wav"
        _save_wav(pcm, out_path)
        mime = "audio/wav"
        if FFMPEG and file_format == FORMAT_MP3:
            mp3_path = OUTPUT_DIR / f"{basename}.mp3"
            r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(out_path),
                                "-codec:a", "libmp3lame", "-q:a", "0", str(mp3_path)],
                               capture_output=True)
            if r.returncode == 0 and mp3_path.exists():
                out_path.unlink()
                out_path, mime = mp3_path, "audio/mpeg"
            else:
                notes.append("mp3 encode failed — saved WAV instead")

        spec_b64 = None
        try:
            spec_png = render_spectrogram_png(pcm, sample_rate=SAMPLE_RATE,
                                              width=1200, height=240)
            spec_b64 = base64.b64encode(spec_png).decode("ascii")
        except Exception as e:
            notes.append(f"spectrogram failed: {type(e).__name__}: {e}")

        load_note = (f"DiT-load {t['dit_load_ms']:.0f} ms ·&nbsp; "
                     if t.get("dit_load_ms", 0) > 100 else "")
        cached_tag = " (cached)" if t.get("enc_cached") else ""
        enc_note = f"encode {t['enc_ms']:.0f} ms{cached_tag} ·&nbsp; " if t.get("enc_ms") else ""
        apg_note = f" (apg {apg:g})" if apg != 1.0 else ""
        cfg_note = f"cfg {cfg:g}{apg_note} ·&nbsp; " if cfg != 1.0 else ""
        lora_note = (f"<b>lora</b>: {html_lib.escape(_lora_disp(lora_specs))} ·&nbsp; "
                     if lora_specs else "")
        prompt_disp = html_lib.escape(prompt) or "<i>(no prompt)</i>"
        neg_disp = (f' · <span style="opacity:0.75">neg: {html_lib.escape(negative_prompt.strip())}</span>'
                    if negative_prompt and negative_prompt.strip() and cfg != 1.0 else "")
        timing_html = (
            f"{prompt_disp}{neg_disp} ·&nbsp; "
            f"{load_note}{enc_note}{cfg_note}{lora_note}"
            f"<b>Inference</b>: {t['inference_ms']:.0f} ms "
            f"<span style='color:#888'>(t5={t['t5_ms']:.0f} · sample={t['sample_ms']:.0f} · "
            f"decode={t['decode_ms']:.0f})</span> ·&nbsp; "
            f"<b>{t['realtime']:.1f}× realtime</b> ·&nbsp; "
            f"<b>seed</b>: <code>{seed}</code> ·&nbsp; "
            f"<b>seq_len</b>: {t['T_lat']} ·&nbsp; <b>samples</b>: {t['samples']}"
        )
        if notes:
            timing_html = ("".join(f"<div style='color:#fa3; font-size:0.85em'>note: {n}</div>"
                                   for n in notes) + timing_html)

        entry = {
            "key": f"k{time.time_ns()}",
            "ts": time.time(),
            "dit": dit_name,
            "neg": (negative_prompt or "").strip(),
            "cfg": float(cfg),
            "smx": float(sigma_max),
            "path": str(out_path),
            "mime": mime,
            "name": out_path.name,
            "size_mb": out_path.stat().st_size / 1e6,
            "mode": mode,
            "prompt": prompt,
            "seed": seed,
            "lora": _lora_disp(lora_specs),
            "spec_b64": spec_b64,
            "timing": timing_html,
        }
        return entry, None

    def _present(state, entry, opts, queued_panel, *, force_autoplay=False):
        """Make `entry` the current clip (previous current moves to history top)
        and render all output panels."""
        if state["current"] is not None:
            state["history"].insert(0, state["current"])
        state["current"] = entry
        loop = "Loop" in opts
        main = render_player(entry,
                             autoplay=force_autoplay or "Auto-play" in opts,
                             autodl="Auto-download" in opts,
                             radio="Infinite Radio" in opts and not loop,
                             loop=loop,
                             hotswap="Hotswap" in opts)
        return (main, entry["timing"], "",
                render_history(state["history"], advance="Auto-play" in opts, loop=loop),
                queued_panel, state)

    def generate(dit_name, decoder_name, prompt, negative_prompt,
                 seconds, steps, seed_text, cfg, apg, sigma_max, init_noise,
                 a2a_audio, inpaint_audio, inp_start, inp_end,
                 output_opts, file_format, *lora_and_state):
        *lora_vals, state = lora_and_state
        entry, err_msg = _generate_entry(
            dit_name, decoder_name, prompt, negative_prompt, seconds, steps,
            seed_text, cfg, apg, sigma_max, init_noise, a2a_audio,
            inpaint_audio, inp_start, inp_end, output_opts, file_format,
            *lora_vals)
        if entry is None:
            return (gr.update(), gr.update(),
                    f"<span style='color:#f88'>{err_msg}</span>",
                    gr.update(), gr.update(), state)
        # a manual Generate makes any queued clip stale (settings may have changed);
        # the chained pregen refills it when Infinite Radio is on
        state["queued"] = None
        opts = output_opts or []
        queued_panel = render_queue_status(generating=True) if "Infinite Radio" in opts else ""
        return _present(state, entry, opts, queued_panel)

    def promote(dit_name, decoder_name, prompt, negative_prompt,
                seconds, steps, seed_text, cfg, apg, sigma_max, init_noise,
                a2a_audio, inpaint_audio, inp_start, inp_end,
                output_opts, file_format, *lora_and_state):
        """Infinite Radio: swap the pre-generated clip in when playback ends.
        Falls back to a full generate if the queue is empty. Either way the
        next track ALWAYS autoplays — Auto-play only governs whether a clip
        starts when nothing was playing; radio transitions are continuations."""
        *lora_vals, state = lora_and_state
        q = state.get("queued")
        if q is None:
            entry, err_msg = _generate_entry(
                dit_name, decoder_name, prompt, negative_prompt, seconds, steps,
                seed_text, cfg, apg, sigma_max, init_noise, a2a_audio,
                inpaint_audio, inp_start, inp_end, output_opts, file_format,
                *lora_vals)
            if entry is None:
                return (gr.update(), gr.update(),
                        f"<span style='color:#f88'>{err_msg}</span>",
                        gr.update(), gr.update(), state)
            q = entry
        state["queued"] = None
        return _present(state, q, output_opts or [],
                        render_queue_status(generating=True), force_autoplay=True)

    def pregen(dit_name, decoder_name, prompt, negative_prompt,
               seconds, steps, seed_text, cfg, apg, sigma_max, init_noise,
               a2a_audio, inpaint_audio, inp_start, inp_end,
               output_opts, file_format, *lora_and_state):
        """Chained after generate/promote: pre-generate the NEXT clip while the
        current one plays (Infinite Radio only)."""
        *lora_vals, state = lora_and_state
        if "Infinite Radio" not in (output_opts or []):
            state["queued"] = None
            return "", state
        entry, err_msg = _generate_entry(
            dit_name, decoder_name, prompt, negative_prompt, seconds, steps,
            seed_text, cfg, apg, sigma_max, init_noise, a2a_audio,
            inpaint_audio, inp_start, inp_end, output_opts, file_format,
            *lora_vals)
        if entry is None:
            return (f"<div style='color:#f88; font-size:0.85em'>queue: {err_msg}</div>",
                    state)
        state["queued"] = entry
        return render_queue_status(entry), state

    # sa3-promote must stay MOUNTED for the onended click to find it —
    # visible=False would remove it from the DOM entirely, so hide via CSS.
    # sa3-out: collapse the column's flex gap + wrapper padding so the
    # spectrogram caption / timing line / queued status sit tightly together.
    _css = ("#sa3-promote{display:none !important}"
            "#sa3-out{gap:4px !important}"
            "#sa3-out .html-container{padding:0 !important; margin:0 !important}")
    with gr.Blocks(title="SA3 MLX", css=_css) as demo:
        gr.Markdown(
            "# SA3 MLX — Apple Silicon\n"
            "Text-to-audio, CFG + negative prompt, audio-to-audio, inpainting. "
            "First use of a model loads weights, subsequent runs are cached."
        )
        st = gr.State({"current": None, "queued": None, "history": []})
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    dit_dd = gr.Dropdown(label="DiT model", choices=list(DIT_CHOICES.keys()),
                                         value=initial_dit, scale=1)
                    decoder_dd = gr.Dropdown(label="Decoder (codec)",
                                             choices=list(DECODER_CHOICES.keys()),
                                             value=initial_decoder, scale=1)
                with gr.Row():
                    prompt = gr.Textbox(label="Prompt", lines=2, scale=6,
                                        placeholder="e.g. 'Impending tribal, epic orchestral buildup'")
                    seed = gr.Textbox(label="Seed (optional)", max_lines=1, value="",
                                      scale=1, min_width=80)
                with gr.Row():
                    seconds = gr.Slider(label="Seconds", minimum=1,
                                        maximum=MAX_SECONDS.get(initial_dit, 120),
                                        value=default_seconds, step=1)
                    steps = gr.Slider(label="Steps", minimum=1, maximum=16,
                                      value=default_steps, step=1)
                    cfg = gr.Slider(label="CFG", minimum=0.0, maximum=10.0,
                                    value=1.0, step=0.1)

                with gr.Accordion("Advanced", open=False):
                    with gr.Row():
                        apg = gr.Slider(label="APG (only applies when CFG > 1)",
                                        minimum=0.0, maximum=1.0, value=1.0, step=0.05)
                        sigma_global = gr.Slider(label="sigma_max",
                                                 minimum=0.0, maximum=1.0, value=1.0, step=0.01)
                    negative_prompt = gr.Textbox(label="Negative prompt", lines=1)

                with gr.Accordion("LoRA", open=False):
                    _dd0 = _lora_dd_choices(initial_dit)
                    lora_inputs, lora_rows, lora_rm_btns = [], [], []
                    lora_dds, lora_files, lora_srows = [], [], []
                    # --lora on the CLI preloads slot(s) 0..N: the trained ckpt
                    # is active on launch (underfit's "load the LoRA by default").
                    for _i in range(1, 4):
                        _prel = preload_loras[_i - 1] if _i - 1 < len(preload_loras) else None
                        with gr.Group(visible=_prel is not None) as _grp:
                            with gr.Row(equal_height=True):
                                _lf = gr.File(label=f"LoRA {_i} (.safetensors)",
                                              file_types=[".safetensors"],
                                              type="filepath", scale=1, height=88,
                                              value=_prel)
                                _ld = gr.Dropdown(
                                    label=f"…or pick from loras/{LORA_DIR_NAMES[initial_dit]}/",
                                    choices=_dd0, value=_DD_NONE, scale=1,
                                    visible=len(_dd0) > 1)
                                _rm = gr.Button("✕", size="sm", scale=0,
                                                min_width=36)
                            with gr.Row(visible=_prel is not None) as _srow:
                                _ls = gr.Slider(label="strength", minimum=0.0,
                                                maximum=10.0, value=1.0, step=0.1,
                                                scale=2, min_width=110)
                                _ln = gr.Slider(label="Min step", minimum=1,
                                                maximum=default_steps, value=1,
                                                step=1, scale=1, min_width=80)
                                _lx = gr.Slider(label="Max step", minimum=1,
                                                maximum=default_steps,
                                                value=default_steps, step=1,
                                                scale=1, min_width=80)
                        lora_rows.append(_grp)
                        lora_rm_btns.append(_rm)
                        lora_dds.append(_ld)
                        lora_files.append(_lf)
                        lora_srows.append(_srow)
                        lora_inputs += [_lf, _ld, _ls, _ln, _lx]
                    lora_add_btn = gr.Button("+ Add LoRA", size="sm")
                    lora_vis = gr.State([k < len(preload_loras) for k in range(3)])
                    # Per-model LoRA memory: switching DiT saves the outgoing
                    # model's slots and restores the incoming model's.
                    lora_mem = gr.State({})
                    prev_dit = gr.State(initial_dit)

                with gr.Accordion("Audio-to-audio (guide the whole clip)", open=False):
                    a2a_audio = gr.Audio(label="Guide audio — generation starts from its latents",
                                         type="filepath")
                    sigma_slider = gr.Slider(
                        label="init_noise_level (1.0 = prompt, ~0.92 = fusion, 0 = input)",
                        minimum=0.0, maximum=1.0, value=0.92, step=0.01)

                with gr.Accordion("Inpainting (regenerate a span of reference audio)", open=False):
                    inpaint_audio = gr.Audio(label="Reference audio — kept bit-exact outside the range",
                                             type="filepath")
                    with gr.Row():
                        inp_start = gr.Slider(label="Start (s)", minimum=0, maximum=120,
                                              value=0, step=0.5)
                        inp_end = gr.Slider(label="End (s)", minimum=0, maximum=120,
                                            value=0, step=0.5)

                with gr.Accordion("Output", open=False):
                    output_opts = gr.CheckboxGroup(
                        ["Auto-play", "Auto-download", "Infinite Radio", "Loop", "Hotswap"],
                        value=["Auto-play"], label="Playback Options")
                    opts_state = gr.State(["Auto-play"])
                    file_format = gr.Radio(
                        [FORMAT_MP3, FORMAT_WAV] if FFMPEG else [FORMAT_WAV],
                        value=FORMAT_MP3 if FFMPEG else FORMAT_WAV,
                        label="Format", visible=FFMPEG)

            with gr.Column(scale=2, elem_id="sa3-out"):
                generate_btn = gr.Button("Generate", variant="primary", size="lg",
                                         elem_id="sa3-generate")
                promote_btn = gr.Button("", elem_id="sa3-promote")   # CSS-hidden, DOM-present
                gr.Markdown("**Output**")
                output_player = gr.HTML()
                timing = gr.HTML()
                error_box = gr.HTML()
                queued_html = gr.HTML()
                history_html = gr.HTML()

        def _sync_dd_vis(dit_name):
            """Second batch after any group (re)mount: gradio 6 drops visibility
            updates sent to components inside a hidden container (the component
            isn't in the DOM), so dropdown show/hide must be applied AFTER the
            slot groups are mounted — chained via .then() on every event that
            reveals them."""
            v = len(_lora_dd_choices(dit_name)) > 1
            return [gr.update(visible=v) for _ in range(3)]

        dit_dd.change(on_dit_change,
                      inputs=[dit_dd, seconds, steps, prev_dit, lora_mem,
                              lora_vis] + lora_inputs,
                      outputs=[decoder_dd, seconds] + lora_dds + lora_srows
                              + lora_rows
                              + [lora_add_btn, lora_vis, prev_dit, lora_mem]
                              + lora_files
                              + [c for i in range(3)
                                 for c in lora_inputs[5 * i + 2: 5 * i + 5]]
                      ).then(_sync_dd_vis, inputs=[dit_dd], outputs=lora_dds
                      ).then(None, js=_JS_FIX_SLIDERS)

        def on_seconds_change(sec):
            return gr.update(maximum=sec), gr.update(maximum=sec)
        seconds.change(on_seconds_change, inputs=[seconds], outputs=[inp_start, inp_end])

        def on_opts_change(opts, prev):
            """Loop and Infinite Radio are mutually exclusive — the one just
            ticked wins, the other unticks."""
            opts = opts or []
            if "Loop" in opts and "Infinite Radio" in opts:
                added = set(opts) - set(prev or [])
                drop = "Infinite Radio" if "Loop" in added else "Loop"
                opts = [o for o in opts if o != drop]
                return gr.update(value=opts), opts
            return gr.update(), opts
        output_opts.change(on_opts_change, inputs=[output_opts, opts_state],
                           outputs=[output_opts, opts_state])

        def lora_add(vis, dit_name, f1, d1, f2, d2, f3, d3):
            """Reveal the first hidden LoRA slot; hide the button at 3/3.
            Slider-row and dropdown visibility are recomputed from slot content
            and the current model's library on every add, so a re-opened slot
            can never inherit a stale visible state — sliders only show once
            THIS slot has an adapter, and the dropdown only when
            loras/<model>/ has any."""
            vis = list(vis)
            for i, v in enumerate(vis):
                if not v:
                    vis[i] = True
                    break
            srows = [gr.update(visible=bool(f or (d and d != _DD_NONE)))
                     for f, d in ((f1, d1), (f2, d2), (f3, d3))]
            return ([gr.update(visible=v) for v in vis] + srows
                    + [gr.update(visible=not all(vis)), vis])

        lora_add_btn.click(lora_add,
                           inputs=[lora_vis, dit_dd]
                           + [c for i in range(3)
                              for c in lora_inputs[5 * i: 5 * i + 2]],
                           outputs=lora_rows + lora_srows
                           + [lora_add_btn, lora_vis]
                           ).then(_sync_dd_vis, inputs=[dit_dd], outputs=lora_dds
                           ).then(None, js=_JS_FIX_SLIDERS)

        def _lora_remove(idx):
            def _rm(vis, cur_steps):
                vis = list(vis)
                vis[idx] = False
                return ([gr.update(visible=v) for v in vis]
                        + [gr.update(visible=True), vis,
                           # reset the slot: file, dropdown, sliders (hidden again)
                           None, gr.update(value=_DD_NONE),
                           1.0, 1, int(cur_steps),
                           gr.update(visible=False)])
            _rm.__name__ = f"lora_remove_{idx + 1}"
            return _rm
        for _idx, _btn in enumerate(lora_rm_btns):
            _btn.click(_lora_remove(_idx), inputs=[lora_vis, steps],
                       outputs=lora_rows + [lora_add_btn, lora_vis]
                       + lora_inputs[5 * _idx: 5 * _idx + 5]
                       + [lora_srows[_idx]])

        # A slot's strength/step sliders appear once an adapter is loaded —
        # via file drop OR dropdown pick — and the two sources are exclusive
        # (loading one resets the other). All three events are user-driven
        # (upload/clear/input), so the programmatic resets here can't loop.
        def _lora_wire_slot(idx):
            lf, ld, srow = lora_files[idx], lora_dds[idx], lora_srows[idx]

            def _up(f):
                return gr.update(visible=bool(f)), gr.update(value=_DD_NONE)
            _up.__name__ = f"lora_file_{idx + 1}"
            lf.upload(_up, inputs=[lf], outputs=[srow, ld]
                      ).then(None, js=_JS_FIX_SLIDERS)

            def _cl(dd):
                return gr.update(visible=bool(dd and dd != _DD_NONE))
            _cl.__name__ = f"lora_file_clear_{idx + 1}"
            lf.clear(_cl, inputs=[ld], outputs=[srow])

            def _pick(dd, f):
                loaded = bool(dd and dd != _DD_NONE)
                return (gr.update(visible=loaded or bool(f)),
                        gr.update(value=None) if loaded else gr.update())
            _pick.__name__ = f"lora_pick_{idx + 1}"
            ld.input(_pick, inputs=[ld, lf], outputs=[srow, lf]
                     ).then(None, js=_JS_FIX_SLIDERS)
        for _idx in range(3):
            _lora_wire_slot(_idx)

        def on_steps_change(s):
            # keep the 6 step-range sliders bounded by the schedule length
            return [gr.update(maximum=int(s))] * 6
        steps.change(on_steps_change, inputs=[steps],
                     outputs=[c for i in range(3)
                              for c in lora_inputs[5 * i + 3: 5 * i + 5]])

        ctrl_inputs = [dit_dd, decoder_dd, prompt, negative_prompt,
                       seconds, steps, seed, cfg, apg, sigma_global,
                       sigma_slider, a2a_audio, inpaint_audio,
                       inp_start, inp_end, output_opts, file_format] + lora_inputs
        main_outputs = [output_player, timing, error_box, queued_html, history_html, st]

        # NB: _present returns (player, timing, err, history, queued, state) —
        # map onto components in that order.
        def _reorder(fn):
            def wrapped(*args):
                player, tim, err_, hist, queued, state = fn(*args)
                return player, tim, err_, queued, hist, state
            wrapped.__name__ = fn.__name__
            return wrapped

        generate_btn.click(_reorder(generate), inputs=ctrl_inputs + [st],
                           outputs=main_outputs
                           ).then(pregen, inputs=ctrl_inputs + [st],
                                  outputs=[queued_html, st])
        promote_btn.click(_reorder(promote), inputs=ctrl_inputs + [st],
                          outputs=main_outputs
                          ).then(pregen, inputs=ctrl_inputs + [st],
                                 outputs=[queued_html, st])

        gr.Markdown(
            "<p style='color:#888; font-size:0.85em'>"
            "WAVs saved under <code>output/gradio/</code>. "
            "DiT runs fp16 (MLX canonical), codecs fp32.</p>"
        )

    demo.queue(max_size=16).launch(share=share, server_name="0.0.0.0",
                                   server_port=server_port,
                                   allowed_paths=[str(OUTPUT_DIR)],
                                   prevent_thread_lock=False, show_error=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dit", choices=list(DIT_CHOICES.keys()), default="sm-music",
                    help="Initial DiT bundle (switchable at runtime)")
    ap.add_argument("--decoder", choices=list(DECODER_CHOICES.keys()), default=None,
                    help="Initial decoder. Default: pairs with --dit")
    ap.add_argument("--default-seconds", type=float, default=30.0,
                    help="Length to pre-warm the initial DiT at")
    ap.add_argument("--default-steps", type=int, default=8)
    ap.add_argument("--share", action=argparse.BooleanOptionalAction, default=True,
                    help="Create a public gradio.live URL (default on)")
    ap.add_argument("--lora", action="append", default=None, metavar="PATH",
                    help="Preload a LoRA .safetensors into a slot (active on "
                         "launch). Repeatable for up to 3 slots. underfit uses "
                         "this to open the gradio with a trained checkpoint loaded.")
    ap.add_argument("--port", type=int, default=None,
                    help="Server port (default: gradio picks 7860+).")
    args = ap.parse_args()

    if args.decoder is None:
        args.decoder = DEFAULT_DECODERS[args.dit]
    preload = tuple(args.lora or ())[:3]

    print(f"\n━━━ SA3 MLX — gradio ━━━")
    print(f"  initial dit:     {args.dit}")
    print(f"  initial decoder: {args.decoder}")
    print(f"  models:          {', '.join(DIT_CHOICES.keys())}  (runtime-switchable)")
    if preload:
        print(f"  preloaded LoRA:  {', '.join(preload)}")
    build_ui(args.dit, args.decoder, share=args.share,
             default_seconds=args.default_seconds, default_steps=args.default_steps,
             preload_loras=preload, server_port=args.port)


if __name__ == "__main__":
    main()
