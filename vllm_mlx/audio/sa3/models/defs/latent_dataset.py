"""Torch-free pre-encoded latent dataset for LoRA training (numpy + stdlib only).

Faithful port of the underfit data pipeline (TRAINING_CONVENTIONS.md §7):

- ``PreEncodedLatentDataset`` mirrors stable_audio_3.data.dataset.PreEncodedDataset
  (latent cropping, silence padding, mask handling, min/max-length rejection with
  random redraw) over the underfit pre-encode format: ``<stem>.npy`` latents
  ``[D, T]`` float32 + ``<stem>.json`` sidecar metadata.
- ``build_prompt`` ports underfit's dataset_processing/prompt_templates.py
  (tag/path/fixed sources with weighted balance, 50/50 shuffle-vs-subset tag
  augmentation, trigger token at ``trigger_pct``), falling back to the legacy
  songs_simple.py fixed-order tag prompt when no prompt_config is given.
- ``iterate_batches`` mirrors underfit's tiny-dataset core behavior: when
  ``len(dataset) < batch_size``, sample with replacement for ``batch_size * 100``
  samples per epoch (~100 batches); otherwise a shuffled epoch, drop_last=False.

Deliberate underfit convention: ``seconds_total`` stays the FULL stored duration
and is never recomputed after cropping.
"""

import json
import os
import random

import numpy as np

__all__ = ["PreEncodedLatentDataset", "build_prompt", "iterate_batches"]


# ---------------------------------------------------------------------------
# Prompt building (port of underfit dataset_processing/prompt_templates.py)
# ---------------------------------------------------------------------------

_TAG_DISPLAY = {
    "title": "Title", "artist": "Artist", "album": "Album",
    "genre": "Genre", "label": "Label", "date": "Year",
    "composer": "Composer", "bpm": "BPM", "prompt": "Prompt",
}
_ALL_TAG_KEYS = list(_TAG_DISPLAY.keys())

# Legacy (no prompt_config) fixed tag order — from underfit songs_simple.py.
_LEGACY_TAG_KEYS = ["artist", "title", "date", "bpm", "genre", "label", "album", "composer"]


def _get(metadata, key):
    val = metadata.get(key, "")
    if isinstance(val, (list, tuple)):
        val = val[0] if val else ""
    return str(val).strip()


def _build_tag_prompt(metadata, pc, rng):
    tag_keys = pc.get("tag_keys", _ALL_TAG_KEYS)
    hide_names = pc.get("hide_tag_names", False)
    hide_commas = pc.get("hide_commas", False)
    split_commas = pc.get("split_commas", False)

    parts = []
    for key in tag_keys:
        val = _get(metadata, key)
        if not val:
            continue
        label = _TAG_DISPLAY.get(key, key)
        if split_commas and "," in val:
            for sub in val.split(","):
                sub = sub.strip()
                if sub:
                    parts.append(sub if hide_names else f"{label}: {sub}")
        else:
            parts.append(val if hide_names else f"{label}: {val}")

    if not parts:
        return ""

    if pc.get("shuffle", True):
        # 50% shuffle all, 50% random subset
        if rng.random() < 0.5:
            rng.shuffle(parts)
        else:
            parts = rng.sample(parts, rng.randint(1, len(parts)))

    return (" " if hide_commas else ", ").join(parts)


def _build_path_prompt(metadata, pc):
    relpath = _get(metadata, "relpath")
    if not relpath:
        return ""

    path_opts = pc.get("path_opts", {})
    # Frontend sends camelCase keys
    hide_ext = path_opts.get("hideExt", path_opts.get("hide_ext", False))
    hide_dirs = path_opts.get("hideDirs", path_opts.get("hide_dirs", False))
    hide_topmost = path_opts.get("hideTopmostDir", path_opts.get("hide_topmost_dir", False))

    parts = relpath.replace("\\", "/").split("/")
    filename = parts[-1]
    dirs = parts[:-1]

    if hide_ext:
        dot = filename.rfind(".")
        if dot > 0:
            filename = filename[:dot]

    if hide_dirs:
        if hide_topmost or not dirs:
            return filename
        return dirs[0] + "/" + filename

    if hide_topmost and dirs:
        return os.path.join(*dirs[1:], filename) if len(dirs) > 1 else filename

    return os.path.join(*dirs, filename) if dirs else filename


def _legacy_prompt(metadata, rng):
    """No-prompt_config fallback: underfit songs_simple.py behavior."""
    properties = []
    for key in _LEGACY_TAG_KEYS:
        val = _get(metadata, key)
        if val:
            properties.append(f"{_TAG_DISPLAY[key]}: {val}")

    if not properties:
        return str(metadata.get("text", ""))

    # 50% shuffle all, 50% random subset
    if rng.random() < 0.5:
        rng.shuffle(properties)
    else:
        properties = rng.sample(properties, rng.randint(1, len(properties)))
    return ", ".join(properties)


def build_prompt(metadata, prompt_config, rng):
    """Build one training prompt from sidecar metadata.

    Port of prompt_templates.get_custom_metadata (returns the prompt string
    directly; underfit's ``lyrics`` is always "" and is omitted here). ``rng``
    is a ``random.Random`` so every __getitem__ draw is fresh (per-epoch
    prompt augmentation) yet reproducible from the dataset seed.
    """
    pc = prompt_config

    if pc is None:
        return _legacy_prompt(metadata, rng)

    use_tags = pc.get("use_tags", True)
    use_paths = pc.get("use_paths", False)
    use_fixed = pc.get("use_fixed", False)
    fixed_text = pc.get("fixed_text", "")
    balance = pc.get("balance", {})
    trigger = pc.get("trigger", "")
    trigger_pct = pc.get("trigger_pct", 80)

    # Build candidate prompts for each enabled method
    candidates = []  # (method_name, prompt_text)
    weights = []

    if use_tags:
        candidates.append(("tags", _build_tag_prompt(metadata, pc, rng)))
        weights.append(balance.get("tags", 50))

    if use_paths:
        candidates.append(("paths", _build_path_prompt(metadata, pc)))
        weights.append(balance.get("paths", 50))

    if use_fixed:
        candidates.append(("fixed", fixed_text))
        weights.append(balance.get("fixed", 0))

    # Pick one method based on balance weights
    chosen_method = None
    prompt = ""
    if not candidates:
        prompt = str(metadata.get("text", ""))
    else:
        total = sum(weights)
        if total <= 0:
            chosen_method, prompt = rng.choice(candidates)
        else:
            chosen_method, prompt = rng.choices(candidates, weights=weights, k=1)[0]

    # Prepend trigger token based on trigger_pct
    if trigger and trigger_pct > 0 and rng.random() * 100 < trigger_pct:
        if prompt:
            use_comma = chosen_method == "tags" and not pc.get("hide_commas", False)
            prompt = trigger + (", " if use_comma else " ") + prompt
        else:
            prompt = trigger

    return prompt


# ---------------------------------------------------------------------------
# Dataset (port of PreEncodedDataset semantics, torch-free)
# ---------------------------------------------------------------------------

class PreEncodedLatentDataset:
    """Pre-encoded latent dataset over ``<stem>.npy`` + ``<stem>.json`` pairs.

    Scans ``root_dir`` recursively for underfit pre-encode outputs: ``.npy``
    latents ``[D, T]`` float32 with a matching ``.json`` sidecar containing
    ``seconds_total``, ``seconds_start``, ``audio_samples``, ``latent_shape``,
    ``padding_mask`` (0/1 at latent resolution) plus arbitrary tag keys
    (title/artist/genre/prompt/...). ``silence.npy`` at the root is used to
    pad short items (never served as a sample).

    ``__getitem__`` returns a dict:
      latents       np.float32 ``[D, latent_crop_length]``
      padding_mask  np.bool_ ``[latent_crop_length]`` (True = valid frame)
      prompt        str (freshly randomized every call — per-epoch augmentation)
      seconds_total float — the FULL stored duration (never recomputed on crop)
      relpath       str (path of the .npy relative to root_dir)

    Items outside [min_length_sec, max_length_sec] and items that fail to load
    are rejected by redrawing a different random index, like the reference.
    """

    _MAX_REDRAWS = 100

    def __init__(
        self,
        root_dir,
        latent_crop_length,
        *,
        random_crop=True,
        prompt_config=None,
        min_length_sec=None,
        max_length_sec=None,
        seed=42,
    ):
        self.root_dir = os.path.abspath(root_dir)
        self.latent_crop_length = latent_crop_length
        self.random_crop = random_crop
        self.prompt_config = prompt_config
        self.min_length_sec = min_length_sec
        self.max_length_sec = max_length_sec
        self._rng = random.Random(seed)

        # Silence latent for variable-length padding (reference stores [1, C, N])
        self.silence_latents = None
        silence_path = os.path.join(self.root_dir, "silence.npy")
        if os.path.exists(silence_path):
            silence = np.load(silence_path)
            if silence.ndim == 3:
                silence = silence.squeeze(0)  # [C, N]
            self.silence_latents = np.asarray(silence, dtype=np.float32)

        # Recursive scan for npy+json pairs (sorted for determinism)
        self.filenames = []
        for dirpath, _dirnames, files in os.walk(self.root_dir):
            for fn in files:
                if not fn.endswith(".npy") or fn == "silence.npy":
                    continue
                npy_path = os.path.join(dirpath, fn)
                json_path = npy_path[: -len(".npy")] + ".json"
                if os.path.exists(json_path):
                    self.filenames.append((npy_path, json_path))
        self.filenames.sort()

        if not self.filenames:
            raise ValueError(f"No <stem>.npy + <stem>.json pairs found under {self.root_dir}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        for _ in range(self._MAX_REDRAWS):
            item = self._load_item(idx)
            if item is not None:
                return item
            # Rejected (length filter) or failed to load — redraw a random index
            idx = self._rng.randrange(len(self))
        raise RuntimeError(
            f"Gave up after {self._MAX_REDRAWS} redraws — every drawn item was "
            f"rejected or failed to load (check min/max_length_sec and data files)"
        )

    def _load_item(self, idx):
        latent_filename, md_filename = self.filenames[idx]
        try:
            latents = np.load(latent_filename).astype(np.float32, copy=False)  # [D, T]
            with open(md_filename, "r") as f:
                meta = json.load(f)

            padding_mask = list(meta["padding_mask"])

            crop = self.latent_crop_length
            if crop is not None:
                stored_length = latents.shape[1]

                if stored_length > crop:
                    # Crop: last valid index = index of the last 1 in the mask
                    last_ix = len(padding_mask) - 1 - padding_mask[::-1].index(1)

                    if self.random_crop and last_ix > crop:
                        start = self._rng.randint(0, last_ix - crop)
                    else:
                        start = 0

                    latents = latents[:, start:start + crop]
                    padding_mask = padding_mask[start:start + crop]

                elif stored_length < crop:
                    # Pad with the silence latent (tiled/sliced) if present, else zeros
                    pad_needed = crop - stored_length
                    silence = self.silence_latents

                    if silence is not None:
                        if silence.shape[1] >= pad_needed:
                            silence_pad = silence[:, :pad_needed]
                        else:
                            reps = (pad_needed // silence.shape[1]) + 1
                            silence_pad = np.tile(silence, (1, reps))[:, :pad_needed]
                        latents = np.concatenate([latents, silence_pad], axis=1)
                    else:
                        latents = np.pad(latents, ((0, 0), (0, pad_needed)))

                    # Valid frames from the stored mask, zeros for padding
                    padding_mask = padding_mask[:stored_length] + [0] * pad_needed

            # Deliberate convention: seconds_total is the FULL stored duration —
            # never recomputed after cropping.
            seconds_total = float(meta["seconds_total"])

            if self.min_length_sec is not None and seconds_total < self.min_length_sec:
                return None
            if self.max_length_sec is not None and seconds_total > self.max_length_sec:
                return None

            relpath = os.path.relpath(latent_filename, self.root_dir)
            # Path prompts prefer the sidecar's own relpath (pre_encode writes it)
            meta.setdefault("relpath", relpath)

            prompt = build_prompt(meta, self.prompt_config, self._rng)

            return {
                "latents": np.ascontiguousarray(latents, dtype=np.float32),
                "padding_mask": np.asarray(padding_mask, dtype=np.bool_),
                "prompt": prompt,
                "seconds_total": seconds_total,
                "relpath": relpath,
            }
        except Exception as e:
            print(f"Couldn't load file {latent_filename}: {e}")
            return None


# ---------------------------------------------------------------------------
# Batching (underfit small-dataset oversampling + plain epochs)
# ---------------------------------------------------------------------------

def iterate_batches(dataset, batch_size, *, shuffle=True, seed=42):
    """Yield collated batches for one epoch.

    When ``len(dataset) < batch_size``: draw with replacement for
    ``batch_size * 100`` samples (≈100 batches/epoch) — underfit's tiny-dataset
    core behavior (RandomSampler(replacement=True, num_samples=batch_size*100));
    ``shuffle`` is irrelevant in this mode. Otherwise iterate a (optionally
    shuffled) permutation of the dataset, drop_last=False.

    Each batch: {"latents": np.float32 [B, D, L], "padding_mask": np.bool_ [B, L],
    "prompt": list[str], "seconds_total": list[float], "relpath": list[str]}.
    """
    n = len(dataset)
    if n == 0:
        return

    rng = random.Random(seed)

    if n < batch_size:
        # Oversample with replacement (~100 batches per epoch)
        indices = [rng.randrange(n) for _ in range(batch_size * 100)]
    else:
        indices = list(range(n))
        if shuffle:
            rng.shuffle(indices)

    for i in range(0, len(indices), batch_size):
        items = [dataset[j] for j in indices[i:i + batch_size]]
        yield {
            "latents": np.stack([it["latents"] for it in items]).astype(np.float32, copy=False),
            "padding_mask": np.stack([it["padding_mask"] for it in items]),
            "prompt": [it["prompt"] for it in items],
            "seconds_total": [it["seconds_total"] for it in items],
            "relpath": [it["relpath"] for it in items],
        }
