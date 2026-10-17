# SPDX-License-Identifier: Apache-2.0
"""Tests for scripts/bench_metadata.py (#320 Tier S2)."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from scripts import bench_metadata as meta

ROOT = Path(__file__).resolve().parents[1]


def _payload(**overrides):
    base = {
        "_schema_version": meta.SCHEMA_VERSION,
        "_methodology_hash": "a" * 64,
        "value": 1,
    }
    base.update(overrides)
    return base


class TestMetadataEmission:
    def test_add_metadata_sets_schema_and_hash(self, tmp_path):
        script = tmp_path / "bench_x.py"
        script.write_text("print('hello')\n")
        payload = meta.add_bench_metadata({"value": 1}, script)

        assert payload["_schema_version"] == meta.SCHEMA_VERSION
        assert (
            payload["_methodology_hash"]
            == hashlib.sha256(script.read_bytes()).hexdigest()
        )

    def test_registry_covers_every_bench_script(self):
        discovered = {path.name for path in (ROOT / "scripts").glob("bench_*.py")}

        assert discovered == meta.JSON_BENCH_SCRIPTS | meta.NON_JSON_BENCH_SCRIPTS
        assert not meta.JSON_BENCH_SCRIPTS & meta.NON_JSON_BENCH_SCRIPTS

    @pytest.mark.parametrize("script_name", sorted(meta.JSON_BENCH_SCRIPTS))
    def test_json_emitter_uses_central_writer(self, script_name):
        script = ROOT / "scripts" / script_name
        tree = ast.parse(script.read_text(), filename=str(script))
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

        assert calls & {"write_bench_json", "format_bench_json"}, (
            f"{script_name} bypasses the centralized benchmark JSON writer"
        )
        bypasses = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "json"
                and node.func.attr == "dump"
            ):
                bypasses.append(node.lineno)
            if node.func.attr == "write_text" and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "json"
                and child.func.attr == "dumps"
                for child in ast.walk(node)
            ):
                bypasses.append(node.lineno)

        assert not bypasses, (
            f"{script_name} has raw JSON artifact writer(s) at {bypasses}"
        )

    def test_writer_rejects_unregistered_emitter(self, tmp_path):
        script = tmp_path / "bench_unregistered.py"
        script.write_text("pass\n")

        with pytest.raises(ValueError, match="unregistered benchmark JSON emitter"):
            meta.format_bench_json({}, script)

    def test_writer_emits_stamped_json_without_mutating_payload(self, tmp_path):
        destination = tmp_path / "result.json"
        script = ROOT / "scripts" / "bench_attention.py"
        payload = {"result": 42}

        meta.write_bench_json(destination, payload, script)

        artifact = json.loads(destination.read_text())
        assert payload == {"result": 42}
        assert artifact["result"] == 42
        assert artifact["_schema_version"] == meta.SCHEMA_VERSION
        assert (
            artifact["_methodology_hash"]
            == hashlib.sha256(script.read_bytes()).hexdigest()
        )

    def test_add_metadata_copies_payload_and_overwrites_spoofed_keys(self, tmp_path):
        script = tmp_path / "bench_x.py"
        script.write_text("x = 1\n")
        original = {"_schema_version": 99, "_methodology_hash": "spoofed"}
        payload = meta.add_bench_metadata(original, script)

        assert original == {"_schema_version": 99, "_methodology_hash": "spoofed"}
        assert payload["_schema_version"] == meta.SCHEMA_VERSION
        assert (
            payload["_methodology_hash"]
            == hashlib.sha256(script.read_bytes()).hexdigest()
        )


class TestDeterministicHash:
    def test_hash_changes_with_script_bytes(self, tmp_path):
        script = tmp_path / "bench_x.py"
        script.write_text("a = 1\n")
        h1 = meta.methodology_hash_for_script(script)
        script.write_text("a = 2\n")
        h2 = meta.methodology_hash_for_script(script)

        assert h1 != h2
        assert len(h1) == 64
        assert len(h2) == 64


class TestCompatibility:
    def test_same_payloads_are_compatible(self):
        meta.assert_compatible(_payload(), _payload())

    def test_missing_schema_rejected(self):
        payload = _payload()
        payload.pop("_schema_version")
        with pytest.raises(ValueError, match="_schema_version"):
            meta.assert_compatible(payload)

    def test_non_integer_schema_rejected(self):
        with pytest.raises(ValueError, match="non-integer"):
            meta.assert_compatible(_payload(_schema_version="1"))

    def test_different_schema_rejected(self):
        with pytest.raises(ValueError, match="re-bench"):
            meta.assert_compatible(_payload(), _payload(_schema_version=2))

    def test_missing_hash_rejected(self):
        payload = _payload()
        payload.pop("_methodology_hash")
        with pytest.raises(ValueError, match="_methodology_hash"):
            meta.assert_compatible(payload)

    @pytest.mark.parametrize("digest", ["invalid", "g" * 64, "a" * 63])
    def test_invalid_hash_rejected(self, digest):
        with pytest.raises(ValueError, match="invalid SHA-256"):
            meta.assert_compatible(_payload(_methodology_hash=digest))

    def test_different_hash_rejected(self):
        with pytest.raises(ValueError, match="refusing to merge"):
            meta.assert_compatible(_payload(), _payload(_methodology_hash="b" * 64))

    def test_non_dict_payload_rejected(self):
        with pytest.raises(ValueError, match="_schema_version"):
            meta.assert_compatible([], _payload())
