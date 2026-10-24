import hashlib
import json
from pathlib import Path

import pytest

from scripts.deepseek_v41_rd_corpus_prepare import prepare, validate_provenance

MANIFEST_PATH = (
    Path(__file__).parents[1] / "scripts/deepseek_v41_rd_corpus_manifest.json"
)
MANIFEST = json.loads(MANIFEST_PATH.read_text())
RELEASED_LICENSES = {
    "HuggingFaceH4/ultrachat_200k": "mit",
    "NousResearch/hermes-function-calling-v1": "Apache-2.0",
    "SWE-bench/SWE-smith-trajectories": (
        "MIT (dataset); BSD-3-Clause (repository excerpts)"
    ),
    "allenai/WildChat-1M": "ODC-BY-1.0",
    "bigcode/humanevalpack": "MIT",
    "cais/mmlu": "MIT",
    "openai/gsm8k": "MIT",
}


def provenance_fixture(tmp_path, name):
    tokenizers = pytest.importorskip("tokenizers")
    manifest = json.loads(MANIFEST_PATH.read_text())
    spec = next(s for s in manifest["slices"] if s["name"] == name)
    source = next(s for s in manifest["sources"] if name in s["slices"])
    dataset = spec["resolved_datasets"][0]
    license_id = (
        "CC-BY-SA-4.0"
        if dataset.startswith("Wikimedia Wikipedia ")
        else RELEASED_LICENSES[dataset]
    )
    assert source["approved_licenses"] == [license_id]
    manifest["slices"] = [spec]
    spec["calibration_tokens"] = spec["heldout_tokens"] = 2
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {"[UNK]": 0, "calibration": 1, "heldout": 2, "document": 3},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest["tokenizer"]["sha256"] = digest(tokenizer_path)
    provenance = {
        "tokenizer_sha256": digest(tokenizer_path),
        "slices": [],
        "raw_files": [],
    }
    exports = {"calibration": {"slices": {}}, "heldout": {"slices": {}}}
    for split, export in exports.items():
        if split == "calibration" and spec["heldout_only"]:
            continue
        path = tmp_path / f"{split}.txt"
        text = f"{split} document"
        path.write_text(text)
        export["slices"][name] = [str(path)]
        provenance["raw_files"].append({"file": str(path), "sha256": digest(path)})
        provenance["slices"].append(
            {
                "split": split,
                "name": name,
                "file": str(path),
                "sha256": digest(path),
                "tokens": 2,
                "role": spec["role"],
                "documents": [
                    {
                        "dataset": dataset,
                        "license": license_id,
                        "revision": manifest["pinned_sources"].get(dataset, "12345"),
                        "row_id": split,
                        "source_split": split,
                        "document_id": split,
                        "group_id": split,
                        "text_sha256": digest(path),
                        "source_file": str(path),
                        "rendered_file": str(path),
                        "selected_char_count": len(text),
                        "export_char_start": 0,
                        "export_char_end": len(text),
                    }
                ],
            }
        )
    return (
        manifest,
        exports["calibration"],
        exports["heldout"],
        provenance,
        tokenizer_path,
    )


@pytest.mark.parametrize("name", [s["name"] for s in MANIFEST["slices"]])
def test_released_dataset_licenses_validate(tmp_path, name):
    manifest, calibration, heldout, provenance, tokenizer = provenance_fixture(
        tmp_path, name
    )
    prepare(manifest, calibration, heldout)
    result = validate_provenance(manifest, calibration, heldout, provenance, tokenizer)
    assert result["documents"] == (1 if manifest["slices"][0]["heldout_only"] else 2)


def test_license_approved_for_another_dataset_is_rejected(tmp_path):
    manifest, calibration, heldout, provenance, tokenizer = provenance_fixture(
        tmp_path, "general_chat_en"
    )
    license_id = RELEASED_LICENSES["NousResearch/hermes-function-calling-v1"]
    assert any(license_id in s["approved_licenses"] for s in manifest["sources"])
    provenance["slices"][0]["documents"][0]["license"] = license_id
    with pytest.raises(
        ValueError,
        match="source license is not approved for dataset HuggingFaceH4/ultrachat_200k",
    ):
        validate_provenance(manifest, calibration, heldout, provenance, tokenizer)
