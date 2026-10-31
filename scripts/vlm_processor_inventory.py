# SPDX-License-Identifier: Apache-2.0
"""Qualify catalog processor metadata without downloading executable code or weights.

Metadata is fetched into a temporary directory; the shared model cache is never written.
"""

import argparse
import json
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from rapid_mlx.model_metadata import config_indicates_multimodal
from rapid_mlx.utils.model_file_guard import (
    AUXILIARY_TRUST_PROCESSOR_TYPES,
    FORCED_TRUST_PROCESSOR_TYPES,
)

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "config.json",
    "tokenizer_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)
NATIVE = {
    "kimi_k25": "kimi",
    "lfm3_vl": "media-backend",
    "prism_hadamard_qwen35": "prism",
    "embedding_gemma2": "embedding",
}
AUX_KEYS = (
    "mtp_draft_model",
    "dflash_draft_model",
    "native_mtp_draft_model",
    "ddtree_draft_model",
)


def metadata(api, repo, revision, temp_dir, subfolder=None, *, config_only=False):
    names = {s.rfilename for s in api.model_info(repo, revision=revision).siblings}
    prefix = f"{subfolder.strip('/')}/" if subfolder else ""
    data = {}
    for name in FILES[:1] if config_only else FILES:
        remote_name = prefix + name
        if remote_name in names:
            # Keep metadata-only snapshots out of the shared model cache.
            local_dir = Path(temp_dir) / repo.replace("/", "--") / revision
            path = hf_hub_download(
                repo, filename=remote_name, revision=revision, local_dir=local_dir
            )
            with open(path, encoding="utf-8") as handle:
                value = json.load(handle)
            if not isinstance(value, dict):
                raise ValueError(f"{repo}@{revision}: {remote_name} is not an object")
            data[name] = value
    return data, sum(n.startswith(prefix) and n.endswith(".py") for n in names)


def declarations(data):
    maps = {}
    files = {}
    for name, value in data.items():
        auto_map = value.get("auto_map")
        if auto_map:
            if not isinstance(auto_map, dict):
                raise ValueError(f"{name}: auto_map is not an object")
            maps[name] = sorted(auto_map)
        if value.get("model_file") is not None:
            files[name] = value["model_file"]
    return maps, files


def route(model_type):
    if model_type in NATIVE:
        return "native:" + NATIVE[model_type]
    if model_type in FORCED_TRUST_PROCESSOR_TYPES | AUXILIARY_TRUST_PROCESSOR_TYPES:
        return "forced-trust"
    return "dependency:consent-honored"


def generate():
    with tempfile.TemporaryDirectory() as temp_dir:
        return _generate(temp_dir)


def _generate(temp_dir):
    catalog = json.loads((ROOT / "rapid_mlx/aliases.json").read_text())
    api = HfApi()
    aliases, auxiliary, pending = {}, {}, {}
    candidates = []
    for alias, entry in catalog.items():
        data, _ = metadata(
            api,
            entry["hf_path"],
            entry["revision"],
            temp_dir,
            entry.get("subfolder"),
            config_only=True,
        )
        config = data.get("config.json", {})
        if (
            entry.get("supports_image_input") is True
            or config_indicates_multimodal(config)
            or config.get("model_type") in {"spark2_5", "lfm3_vl", "kimi_k25"}
        ):
            candidates.append((alias, entry))
    for alias, entry in candidates:
        repo, rev = entry["hf_path"], entry["revision"]
        subfolder = entry.get("subfolder")
        data, python_files = metadata(api, repo, rev, temp_dir, subfolder)
        model_type = data.get("config.json", {}).get("model_type")
        maps, files = declarations(data)
        processor_class = next(
            (
                data.get(name, {}).get("processor_class")
                for name in (
                    "processor_config.json",
                    "preprocessor_config.json",
                    "tokenizer_config.json",
                )
                if data.get(name, {}).get("processor_class") is not None
            ),
            None,
        )
        aliases[alias] = dict(
            hf_path=repo,
            revision=rev,
            subfolder=subfolder,
            model_type=model_type,
            processor_class=processor_class,
            auto_map=maps,
            model_file=files,
            python_files=python_files,
            text_only_pin=bool(entry.get("is_text_only")),
            route=route(model_type),
        )
        for key in AUX_KEYS:
            if entry.get(key):
                aux_repo = entry[key]
                aux_rev = (
                    entry.get("dflash_draft_revision")
                    if key == "dflash_draft_model"
                    else None
                )
                if aux_repo in pending and pending[aux_repo][0] != aux_rev:
                    raise ValueError(f"Conflicting revisions for auxiliary {aux_repo}")
                pending.setdefault(aux_repo, (aux_rev, []))[1].append(f"{alias}:{key}")
    for repo, (rev, users) in pending.items():
        data, python_files = metadata(api, repo, rev or "main", temp_dir)
        maps, files = declarations(data)
        auxiliary[repo] = dict(
            revision=rev,
            pinned=rev is not None,
            used_by=sorted(users),
            model_type=data["config.json"].get("model_type"),
            auto_map=maps,
            model_file=files,
            python_files=python_files,
            loader="drafter weights only (no processor)",
        )
    return dict(
        schema_version=1,
        scope="Metadata qualification of catalog VLM aliases and their auxiliary draft repositories.",
        aliases=aliases,
        auxiliary=auxiliary,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tests/fixtures/vlm_processor_inventory.json",
    )
    args = parser.parse_args()
    result = generate()  # Complete every fetch before touching the output.
    args.output.write_text(
        json.dumps(result, sort_keys=True, indent=1) + "\n", encoding="utf-8"
    )
    aliases = result["aliases"]
    print(
        f"{len(aliases)} aliases, {len(result['auxiliary'])} auxiliaries, {sum(bool(a['auto_map'] or a['model_file']) for a in aliases.values())} declaring auto_map/model_file, {sum(a['route'] == 'forced-trust' for a in aliases.values())} forced-trust routes"
    )


if __name__ == "__main__":
    main()
