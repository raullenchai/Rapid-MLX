# SPDX-License-Identifier: Apache-2.0
"""Small synthetic sidecars and CPU-only Rapid load/lookup contracts."""
# ruff: noqa: B023, SIM117

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import signal
import struct
import sys
import tempfile
import threading
import time
import unittest
import weakref
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "rapid_mlx.models.qwen4_ple_sidecar", ROOT / "rapid_mlx/models/qwen4_ple_sidecar.py"
)
sidecar = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sidecar
spec.loader.exec_module(sidecar)


def tiny_config():
    return dict(
        model_type="qwen4_exp",
        text_config=dict(
            hidden_size=8,
            num_hidden_layers=2,
            vocab_size=32,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            linear_num_key_heads=1,
            linear_num_value_heads=3,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_conv_kernel_dim=3,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=4,
            shared_expert_intermediate_size=4,
            hc_count=4,
            hc_lowrank=3,
            layer_types=["linear_attention", "full_attention"],
            indexer_n_heads=2,
            indexer_kv_heads=1,
            indexer_head_dim=4,
            indexer_budget=8,
            indexer_compress_ratio=2,
            ple_layer_ids=[1],
            eos_token_id=31,
            ple_embed_dim=32,
            ngram_size=2,
            heads_per_ngram=1,
            ngram_vocab_size_base=7,
            make_ngram_vocab_size_divisible_by=2,
            split_ngram_parts=2,
        ),
    )


class Artifact:
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / "ple_rows.bin"
        self.prefix = "language_model.model.layers.0.ple.ple_embedding.ngram_embedding"
        self.source_prefix = (
            "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
        )
        rng = np.random.default_rng(9)
        self.words = rng.integers(0, 2**32, (8, 4), dtype=np.uint32)
        self.scales = np.full((8, 1), 0x3D80, dtype=np.uint16)
        self.biases = np.full((8, 1), 0xBE00, dtype=np.uint16)
        self.packed = np.concatenate(
            [
                self.words.view(np.uint8),
                self.scales.view(np.uint8),
                self.biases.view(np.uint8),
            ],
            axis=1,
        )
        self.path.write_bytes(self.packed.tobytes())
        self.tensors = {}
        header, data = {}, b""
        for shard in range(2):
            for part, array, dtype in [
                ("weight", self.words, "U32"),
                ("scales", self.scales, "BF16"),
                ("biases", self.biases, "BF16"),
            ]:
                key = f"{self.source_prefix}.shard_{shard}.{part}"
                values = array[shard * 4 : shard * 4 + 4]
                raw = values.tobytes()
                header[key] = dict(
                    dtype=dtype,
                    shape=list(values.shape),
                    data_offsets=[len(data), len(data) + len(raw)],
                )
                self.tensors[key] = values
                data += raw
        raw_header = json.dumps(header).encode()
        (self.root / "model-ple.safetensors").write_bytes(
            struct.pack("<Q", len(raw_header)) + raw_header + data
        )
        self.index = {"weight_map": {name: "model-ple.safetensors" for name in header}}
        self.config = tiny_config()
        (self.root / "config.json").write_text(json.dumps(self.config))
        self.manifest = dict(
            format="qwen4-ple-rows",
            version=1,
            tensor_prefix=self.prefix,
            dims=32,
            group_size=32,
            bits=4,
            mode="affine",
            weight_bytes=16,
            scales_bytes=2,
            biases_bytes=2,
            row_bytes=20,
            num_shards=2,
            rows_per_shard=4,
            total_rows=8,
            data_offset=0,
            shard_sha256=[
                hashlib.sha256(self.packed[i * 4 : i * 4 + 4].tobytes()).hexdigest()
                for i in range(2)
            ],
        )
        self.save_index()

    def save_index(self):
        blob = json.dumps(self.index).encode()
        (self.root / "model.safetensors.index.json").write_bytes(blob)
        self.manifest["source_index_sha256"] = hashlib.sha256(blob).hexdigest()
        self.save_manifest()

    def save_manifest(self):
        Path(str(self.path) + ".manifest.json").write_text(json.dumps(self.manifest))


class SidecarContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.a = Artifact(self.tmp.name)

    def test_builder_reproduces_and_validates_sidecar(self):
        from rapid_mlx.models.qwen4_ple_build import build_sidecar

        output = self.a.root / "built.bin"
        receipt = build_sidecar(self.a.root, output, chunk_rows=2, validation_rows=2)
        self.assertEqual(output.read_bytes(), self.a.packed.tobytes())
        self.assertEqual(receipt["bytes_written"], len(self.a.packed.tobytes()))
        self.assertEqual(
            receipt["manifest"]["shard_sha256"], self.a.manifest["shard_sha256"]
        )
        with self.assertRaises(FileExistsError):
            build_sidecar(self.a.root, output, chunk_rows=2)

    def test_builder_cli_and_guardrails(self):
        from rapid_mlx.models import qwen4_ple_build as builder

        for value in (0, True, 1.5):
            with self.subTest(chunk_rows=value), self.assertRaises(ValueError):
                builder.build_sidecar(
                    self.a.root, self.a.root / "bad.bin", chunk_rows=value
                )

        output = self.a.root / "cli.bin"
        with mock.patch("builtins.print") as printed:
            self.assertEqual(
                builder.main(
                    [
                        "--model",
                        str(self.a.root),
                        "--output",
                        str(output),
                        "--chunk-rows",
                        "2",
                        "--validation-rows",
                        "0",
                    ]
                ),
                0,
            )
        self.assertEqual(output.read_bytes(), self.a.packed.tobytes())
        printed.assert_called_once()

        # Overwrite removes abandoned partial publications before rebuilding.
        partial = Path(str(output) + ".partial")
        partial_manifest = Path(str(partial) + ".manifest.json")
        partial.write_bytes(b"stale")
        partial_manifest.write_text("stale")
        builder.build_sidecar(
            self.a.root,
            output,
            chunk_rows=2,
            validation_rows=0,
            overwrite=True,
        )
        self.assertFalse(partial.exists())
        self.assertFalse(partial_manifest.exists())

        config_path = self.a.root / "config.json"
        original_config = config_path.read_text()
        try:
            for update, message in [
                ({"model_type": "other"}, "exactly one"),
                (
                    {
                        "text_config": dict(
                            self.a.config["text_config"], split_ngram_parts=True
                        )
                    },
                    "configuration",
                ),
                (
                    {
                        "text_config": dict(
                            self.a.config["text_config"], ple_embed_dim=16
                        )
                    },
                    "multiple",
                ),
            ]:
                config = json.loads(original_config)
                if "text_config" in update:
                    config["text_config"] = update["text_config"]
                else:
                    config.update(update)
                config_path.write_text(json.dumps(config))
                with (
                    self.subTest(message=message),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    builder.build_sidecar(
                        self.a.root,
                        self.a.root / f"{message}.bin",
                        validation_rows=0,
                    )
        finally:
            config_path.write_text(original_config)

        original_index = self.a.index.copy()
        self.a.index = {"weight_map": {}}
        self.a.save_index()
        try:
            with self.assertRaisesRegex(ValueError, "shard_0"):
                builder.build_sidecar(
                    self.a.root, self.a.root / "missing.bin", validation_rows=0
                )
        finally:
            self.a.index = original_index
            self.a.save_index()

        with mock.patch.object(
            builder.shutil,
            "disk_usage",
            return_value=SimpleNamespace(free=0),
        ):
            with self.assertRaisesRegex(OSError, "free space"):
                builder.build_sidecar(
                    self.a.root, self.a.root / "full.bin", validation_rows=0
                )

        failed = self.a.root / "failed.bin"
        with mock.patch.object(
            builder, "validate_artifact", side_effect=ValueError("validation failed")
        ):
            with self.assertRaisesRegex(ValueError, "validation failed"):
                builder.build_sidecar(
                    self.a.root, failed, chunk_rows=2, validation_rows=0
                )
        self.assertFalse(Path(str(failed) + ".partial").exists())
        self.assertFalse(Path(str(failed) + ".partial.manifest.json").exists())

    def test_manifest_config_and_dequant_guardrails(self):
        original_manifest = dict(self.a.manifest)
        cases = [
            ({"format": "other"}, "format"),
            ({"bits": 8}, "q4"),
            ({"dims": 31}, "divisible"),
            ({"total_rows": 7}, "row count"),
            ({"shard_sha256": []}, "SHA256"),
        ]
        for update, message in cases:
            self.a.manifest = dict(original_manifest, **update)
            self.a.save_manifest()
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                sidecar.load_manifest(self.a.path)
        self.a.manifest = dict(original_manifest)
        self.a.save_manifest()

        with self.assertRaisesRegex(ValueError, "weight-file glob"):
            sidecar.load_manifest(self.a.root / "model-test.safetensors")
        for value in (False, -1, 1.5):
            with self.subTest(positive=value), self.assertRaises(ValueError):
                sidecar._positive_int(value, "value")
        for dims, packed in [
            (31, self.a.packed),
            (32, self.a.packed[:, :-1]),
        ]:
            with self.subTest(dims=dims), self.assertRaises(ValueError):
                sidecar.dequant_rows_numpy(packed, dims)
        with self.assertRaisesRegex(ValueError, "512 MiB"):
            sidecar.PLESidecarReader(
                self.a.root, self.a.path, cache_bytes=512 * 1024**2 + 1
            )

        config_path = self.a.root / "config.json"
        original_config = config_path.read_text()
        try:
            config = json.loads(original_config)
            config["model_type"] = "other"
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "exactly one"):
                sidecar.validate_artifact(self.a.root, self.a.path, random_rows=0)
            config = json.loads(original_config)
            config["text_config"]["ple_embed_dim"] = 31
            config["text_config"]["ngram_size"] = 3
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "head/embedding"):
                sidecar.validate_artifact(self.a.root, self.a.path, random_rows=0)
        finally:
            config_path.write_text(original_config)
        self.a.manifest["tensor_prefix"] = "wrong"
        self.a.save_manifest()
        with self.assertRaisesRegex(ValueError, "geometry/prefix"):
            sidecar.validate_artifact(self.a.root, self.a.path, random_rows=0)
        self.a.manifest = dict(original_manifest)
        self.a.save_manifest()
        with self.assertRaisesRegex(ValueError, "exceeds4096"):
            sidecar.validate_artifact(self.a.root, self.a.path, random_rows=4097)

    def test_hugging_face_snapshot_blob_symlink_is_accepted(self):
        cache = self.a.root / "models--owner--model"
        snapshot = cache / "snapshots" / ("a" * 40)
        snapshot.mkdir(parents=True)
        artifact = Artifact(snapshot)
        blobs = cache / "blobs"
        blobs.mkdir()
        source = snapshot / "model-ple.safetensors"
        blob = blobs / "deadbeef"
        source.replace(blob)
        source.symlink_to(blob)
        receipt = sidecar.validate_artifact(snapshot, artifact.path, random_rows=0)
        self.assertEqual(receipt["manifest"]["total_rows"], 8)
        from rapid_mlx.models import qwen4_ple_build as builder

        first = next(iter(artifact.index["weight_map"]))
        self.assertEqual(
            builder._source_tensor_rows(
                snapshot.resolve(), artifact.index["weight_map"], first
            ),
            4,
        )

    def test_validation_and_exact_integer_affine(self):
        receipt = sidecar.validate_artifact(self.a.root, self.a.path)
        self.assertGreaterEqual(receipt["checked_rows"], 4)
        actual = sidecar.dequant_rows_numpy(self.a.packed, 32)
        q = (
            (self.a.words[..., None] >> (np.arange(8, dtype=np.uint32) * 4)) & 15
        ).reshape(8, 32)
        # All these values are exactly representable; this independent formula
        # uses float64 arithmetic before the final expected BF16 bit view.
        expected = (
            (q.astype(np.float64) / 16 - 0.125).astype(np.float32).view(np.uint32)
        )
        self.assertTrue(np.array_equal(actual, (expected >> 16).astype(np.uint16)))

    def test_nested_load_reader_ownership_and_original_cleanup_exception(self):
        outer = SimpleNamespace(close=mock.Mock())
        inner = SimpleNamespace(
            close=mock.Mock(side_effect=RuntimeError("cleanup failed"))
        )
        with sidecar._bound_load_source(self.a.root):
            sidecar.own_load_reader(outer)
            with self.assertRaisesRegex(ValueError, "original failure"):
                with sidecar._bound_load_source(self.a.root / "inner"):
                    sidecar.own_load_reader(inner)
                    raise ValueError("original failure")
            sidecar.require_load_source(self.a.root)
            outer.close.assert_not_called()
        inner.close.assert_called_once()
        outer.close.assert_not_called()
        self.assertIsNone(sidecar._LOAD_READERS.get())
        with self.assertRaisesRegex(ValueError, "bound load"):
            sidecar.own_load_reader(outer)

    def test_concurrent_load_reader_ownership_is_isolated(self):
        readers = [
            SimpleNamespace(close=mock.Mock()),
            SimpleNamespace(close=mock.Mock()),
        ]
        barrier = threading.Barrier(2)
        errors = []

        def worker(index):
            try:
                with sidecar._bound_load_source(self.a.root / str(index)):
                    sidecar.own_load_reader(readers[index])
                    barrier.wait(timeout=2)
                    if index == 0:
                        raise ValueError("failed load")
            except BaseException as exc:
                errors.append((index, str(exc)))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(3)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [(0, "failed load")])
        readers[0].close.assert_called_once()
        readers[1].close.assert_not_called()
        self.assertIsNone(sidecar._LOAD_READERS.get())

    def test_lookup_shape_bounds_close_and_cache(self):
        reader = sidecar.PLESidecarReader(self.a.root, self.a.path, cache_bytes=2 * 532)
        self.addCleanup(reader.close)
        ids = np.array([[0, 1, 0], [2, 7, 2]], dtype=np.int64)
        self.assertTrue(
            np.array_equal(
                reader.lookup_bits(ids),
                sidecar.dequant_rows_numpy(self.a.packed, 32)[ids],
            )
        )
        reader.lookup_bits(np.array([7], dtype=np.int64))
        self.assertGreater(reader.stats["cache_hits"], 0)
        self.assertGreater(reader.stats["cache_evictions"], 0)
        self.assertLessEqual(reader.stats["cache_charged_bytes"], 2 * 532)
        for ids in (np.array([-1]), np.array([8])):
            with self.assertRaises(IndexError):
                reader.lookup_bits(ids)
        with self.assertRaises(ValueError):
            reader.lookup_bits(np.array([0.5]))
        self.assertEqual(
            reader.lookup_bits(np.array([], dtype=np.int64)).shape, (0, 32)
        )
        reader.close()
        with self.assertRaises(RuntimeError):
            reader.lookup_bits(np.array([0]))

    def test_corrupt_edge_and_index_refused(self):
        raw = bytearray(self.a.path.read_bytes())
        raw[0] ^= 1
        self.a.path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "content mismatch"):
            sidecar.validate_artifact(self.a.root, self.a.path, random_rows=0)
        self.a.manifest["source_index_sha256"] = "0" * 64
        self.a.save_manifest()
        with self.assertRaisesRegex(ValueError, "digest"):
            sidecar.validate_artifact(self.a.root, self.a.path)

    def test_geometry_dtype_and_missing_source_refused(self):
        self.a.manifest["row_bytes"] = 99
        self.a.save_manifest()
        with self.assertRaises(ValueError):
            sidecar.validate_artifact(self.a.root, self.a.path)
        self.a.manifest["row_bytes"] = 20
        self.a.index["weight_map"].pop(next(iter(self.a.index["weight_map"])))
        self.a.save_index()
        with self.assertRaisesRegex(ValueError, "exactly"):
            sidecar.validate_artifact(self.a.root, self.a.path)

    def test_dequantization_overflow_guardrails(self):
        cases = [
            (0xFFFFFFFF, 0x7F7F, 0, "affine"),
            (0x11111111, 0x7B00, 0x7F7F, "BF16"),
        ]
        for word, scale, bias, message in cases:
            packed = np.zeros((1, 20), dtype=np.uint8)
            packed[:, :16] = np.full((1, 4), word, dtype=np.uint32).view(np.uint8)
            packed[:, 16:18] = np.array([[scale]], dtype=np.uint16).view(np.uint8)
            packed[:, 18:] = np.array([[bias]], dtype=np.uint16).view(np.uint8)
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                sidecar.dequant_rows_numpy(packed, 32)

    def test_inherited_process_fast_paths_without_coverage_fork(self):
        lookup_reader = sidecar.PLESidecarReader(
            self.a.root, self.a.path, random_rows=0
        )
        self.addCleanup(lookup_reader.close)
        with mock.patch.object(
            sidecar.os, "getpid", return_value=lookup_reader._pid + 1
        ):
            with self.assertRaisesRegex(RuntimeError, "inherited across fork"):
                lookup_reader.lookup_bits(np.array([0]))

        close_reader = sidecar.PLESidecarReader(
            self.a.root, self.a.path, random_rows=0, cache_bytes=532
        )
        close_reader.lookup_bits(np.array([0]))
        with mock.patch.object(
            sidecar.os, "getpid", return_value=close_reader._pid + 1
        ):
            close_reader.close()
        self.assertIsNone(close_reader._fd)
        self.assertEqual(close_reader.stats["cache_rows"], 0)

    def test_nonfinite_parameters_refused(self):
        packed = self.a.packed.copy()
        packed[0, 16:18] = np.array([0x7F80], dtype=np.uint16).view(np.uint8)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            sidecar.dequant_rows_numpy(packed, 32)

    def test_reader_detects_changed_file(self):
        reader = sidecar.PLESidecarReader(self.a.root, self.a.path)
        self.addCleanup(reader.close)
        self.a.path.write_bytes(self.a.path.read_bytes() + b"x")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            reader.lookup_bits(np.array([0]))

    def test_atomic_replacement_during_validation_refused(self):
        original = sidecar.validate_artifact

        def replace(*args, **kwargs):
            receipt = original(*args, **kwargs)
            other = self.a.path.with_suffix(".replacement")
            other.write_bytes(self.a.path.read_bytes())
            other.replace(self.a.path)
            return receipt

        with mock.patch.object(sidecar, "validate_artifact", side_effect=replace):
            with self.assertRaisesRegex(RuntimeError, "changed during"):
                sidecar.PLESidecarReader(self.a.root, self.a.path)

    @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX fork")
    def test_forked_reader_refuses_lookup_and_closes_with_inherited_locked_mutex(self):
        reader = sidecar.PLESidecarReader(
            self.a.root, self.a.path, random_rows=0, cache_bytes=1064
        )
        self.addCleanup(reader.close)
        ids = np.array([0], dtype=np.int64)
        expected = reader.lookup_bits(ids)
        for operation in ("lookup", "close", "concurrent_close"):
            # Simulate fork while a different parent thread owns the mutex.
            # The parent's lock stays held until the child exits. An alarm is
            # a test failure, never an accepted refusal or an unbounded hang.
            with reader._lock:
                child = os.fork()
                if child == 0:
                    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
                    signal.alarm(2)
                    try:
                        if operation == "lookup":
                            try:
                                reader.lookup_bits(ids)
                            except RuntimeError as exc:
                                if "inherited across fork" not in str(exc):
                                    os._exit(2)
                            else:
                                os._exit(3)
                        elif operation == "close":
                            reader.close()
                            reader.close()
                            if (
                                reader._fd is not None
                                or reader.stats["cache_rows"] != 0
                            ):
                                os._exit(4)
                        else:
                            original_close = sidecar.os.close
                            calls = []
                            errors = []

                            def delayed_close(fd):
                                calls.append(fd)
                                time.sleep(0.02)
                                original_close(fd)

                            def closer():
                                try:
                                    reader.close()
                                except BaseException as exc:
                                    errors.append(exc)

                            sidecar.os.close = delayed_close
                            threads = [
                                threading.Thread(target=closer) for _ in range(2)
                            ]
                            for thread in threads:
                                thread.start()
                            for thread in threads:
                                thread.join(0.5)
                            if (
                                any(thread.is_alive() for thread in threads)
                                or errors
                                or len(calls) != 1
                            ):
                                os._exit(6)
                    except BaseException:
                        os._exit(5)
                    os._exit(0)
                _, status = os.waitpid(child, 0)
            self.assertEqual(os.waitstatus_to_exitcode(status), 0, operation)
            # Child close must not close the parent's fd or clear its cache.
            os.fstat(reader._fd)
            self.assertEqual(reader.stats["cache_rows"], 1)
            self.assertTrue(np.array_equal(reader.lookup_bits(ids), expected))

    @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX fork")
    def test_fork_waits_for_parent_descriptor_close_publication(self):
        reader = sidecar.PLESidecarReader(self.a.root, self.a.path, random_rows=0)
        self.addCleanup(reader.close)
        closed = threading.Event()
        finish = threading.Event()
        original_close = sidecar.os.close
        errors = []

        def delayed_close(fd):
            original_close(fd)
            closed.set()
            if not finish.wait(2):
                raise RuntimeError("close publication timeout")

        def closer():
            try:
                reader.close()
            except BaseException as exc:
                errors.append(exc)

        with mock.patch.object(sidecar.os, "close", side_effect=delayed_close):
            thread = threading.Thread(target=closer)
            thread.start()
            self.assertTrue(closed.wait(1))
            timer = threading.Timer(0.05, finish.set)
            timer.start()
            try:
                child = os.fork()
                if child == 0:
                    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
                    signal.alarm(2)
                    # The before-fork resource barrier must wait until the fd
                    # is both closed and published as None, not snapshot stale
                    # fd ownership that could later close an unrelated file.
                    if reader._fd is not None:
                        os._exit(7)
                    reader.close()
                    os._exit(0)
                _, status = os.waitpid(child, 0)
            finally:
                finish.set()
                thread.join(2)
                timer.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(os.waitstatus_to_exitcode(status), 0)

    @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX fork")
    def test_resource_guard_allows_cyclic_reader_finalizer_reentrancy(self):
        reader = sidecar.PLESidecarReader(self.a.root, self.a.path, random_rows=0)
        reader.cycle = reader
        fd = reader._fd
        child = os.fork()
        if child == 0:
            signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
            signal.alarm(2)
            ref = weakref.ref(reader)
            del reader
            with sidecar._RESOURCE_LOCK:
                gc.collect()
            if ref() is not None:
                os._exit(8)
            try:
                os.fstat(fd)
            except OSError:
                os._exit(0)
            os._exit(9)
        try:
            _, status = os.waitpid(child, 0)
            self.assertEqual(os.waitstatus_to_exitcode(status), 0)
            os.fstat(fd)
        finally:
            reader.close()
            del reader.cycle

    def test_low_level_source_and_reader_guardrails(self):
        from rapid_mlx.models import qwen4_ple_build as builder

        source = self.a.root / "model-ple.safetensors"
        original = source.read_bytes()
        header_size = struct.unpack("<Q", original[:8])[0]
        original_header = json.loads(original[8 : 8 + header_size])
        data = original[8 + header_size :]
        weight_map = self.a.index["weight_map"]
        first = sorted(weight_map)[0]

        outside = Path(tempfile.mkstemp()[1])
        self.addCleanup(outside.unlink, missing_ok=True)
        outside.write_bytes(original)
        escaped = dict(weight_map)
        escaped[first] = str(outside)
        with self.assertRaisesRegex(ValueError, "escapes"):
            sidecar._source_refs(
                self.a.root.resolve(), self.a.manifest, escaped, self.a.source_prefix
            )
        with self.assertRaisesRegex(ValueError, "escapes"):
            builder._source_tensor_rows(self.a.root.resolve(), escaped, first)

        for blob, message in [
            (b"x", "truncated"),
            (struct.pack("<Q", 10_000), "header length"),
        ]:
            source.write_bytes(blob)
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                sidecar._source_refs(
                    self.a.root.resolve(),
                    self.a.manifest,
                    weight_map,
                    self.a.source_prefix,
                )
            with self.assertRaisesRegex(ValueError, message):
                builder._source_tensor_rows(self.a.root.resolve(), weight_map, first)

        for update, message in [
            ({"dtype": "F32"}, "dtype/shape"),
            ({"data_offsets": [False, 1]}, "offsets"),
            ({"data_offsets": [-1, 0]}, "byte range"),
        ]:
            header = json.loads(json.dumps(original_header))
            header[first].update(update)
            encoded = json.dumps(header).encode()
            source.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                sidecar._source_refs(
                    self.a.root.resolve(),
                    self.a.manifest,
                    weight_map,
                    self.a.source_prefix,
                )

        header = json.loads(json.dumps(original_header))
        header[first]["shape"] = [True, 4]
        encoded = json.dumps(header).encode()
        source.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
        with self.assertRaisesRegex(ValueError, "shape"):
            builder._source_tensor_rows(self.a.root.resolve(), weight_map, first)
        source.write_bytes(original)

        self.a.path.write_bytes(self.a.path.read_bytes() + b"x")
        with self.assertRaisesRegex(ValueError, "file size"):
            sidecar.load_manifest(self.a.path)
        self.a.path.write_bytes(self.a.packed.tobytes())

        reader = sidecar.PLESidecarReader(self.a.root, self.a.path, random_rows=0)
        self.addCleanup(reader.close)
        with mock.patch.object(sidecar.os, "pread", return_value=b""):
            with self.assertRaisesRegex(RuntimeError, "short"):
                reader.lookup_bits(np.array([0]))
        real_fstat = sidecar.os.fstat
        current = real_fstat(reader._fd)
        changed = SimpleNamespace(
            st_dev=current.st_dev,
            st_ino=current.st_ino,
            st_size=current.st_size + 1,
            st_mtime_ns=current.st_mtime_ns,
        )
        with mock.patch.object(sidecar.os, "fstat", side_effect=[current, changed]):
            with self.assertRaisesRegex(RuntimeError, "during lookup"):
                reader.lookup_bits(np.array([0]))
        reader.__del__()
        self.assertIsNone(reader._fd)

    def test_builder_short_read_and_output_size_guardrails(self):
        from rapid_mlx.models import qwen4_ple_build as builder

        with mock.patch.object(builder.os, "pread", return_value=b""):
            with self.assertRaisesRegex(OSError, "short PLE source read"):
                builder.build_sidecar(
                    self.a.root,
                    self.a.root / "short.bin",
                    chunk_rows=2,
                    validation_rows=0,
                )

        original_stat = Path.stat

        def wrong_partial_size(path, *args, **kwargs):
            result = original_stat(path, *args, **kwargs)
            if str(path).endswith(".partial"):
                return SimpleNamespace(st_size=result.st_size + 1)
            return result

        with mock.patch.object(builder.Path, "stat", new=wrong_partial_size):
            with self.assertRaisesRegex(OSError, "output size mismatch"):
                builder.build_sidecar(
                    self.a.root,
                    self.a.root / "size.bin",
                    chunk_rows=2,
                    validation_rows=0,
                )


class CPULoadContracts(SidecarContracts):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx

        mx.set_default_device(mx.cpu)
        cls.mx = mx
        from rapid_mlx.models import qwen4_exp

        cls.qwen = qwen4_exp

    def test_model_args_loader_and_installer_guardrails(self):
        from rapid_mlx.models import qwen4_ple_nvme as adapter

        text = self.a.config["text_config"]
        for kwargs, message in [
            ({"ple_nvme_sidecar": "x"}, "both sidecar"),
            (
                {"ple_nvme_sidecar": 1, "ple_nvme_model_path": "x"},
                "nonempty strings",
            ),
            ({"ple_nvme_cache_bytes": True}, "between0"),
        ]:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                self.qwen.ModelArgs(model_type="qwen4_exp", text_config=text, **kwargs)
        flat = dict(text, model_type="qwen4_exp")
        self.assertIsInstance(self.qwen.ModelArgs.from_dict(flat), self.qwen.ModelArgs)

        with self.assertRaisesRegex(ValueError, "between0"):
            adapter.load_file_backed_qwen4(self.a.root, self.a.path, cache_bytes=True)
        with mock.patch.dict(os.environ, {"MLX_QWEN4_PLE_NVME": str(self.a.path)}):
            with self.assertRaisesRegex(ValueError, "clear MLX"):
                adapter.load_file_backed_qwen4(self.a.root, self.a.path)

        reader = SimpleNamespace(
            manifest=self.a.manifest,
            stats={},
            close=mock.Mock(),
        )
        module = adapter.FileBackedPLEEmbedding(reader)
        adapter.close_file_backed_ple(
            SimpleNamespace(named_modules=lambda: [("ple", module)])
        )
        reader.close.assert_called_once()
        sidecar._after_fork_child()

        def model_and_weights():
            model = self.qwen.Model(self.qwen.ModelArgs.from_dict(self.a.config))
            weights = {
                key.replace(self.a.source_prefix, self.a.prefix).replace(
                    ".shard_", ".shards."
                ): self.mx.array(values).view(self.mx.bfloat16)
                if not key.endswith(".weight")
                else self.mx.array(values)
                for key, values in self.a.tensors.items()
            }
            return model, weights

        model, weights = model_and_weights()
        model.model.layers[0].ple = None
        with (
            sidecar._bound_load_source(self.a.root),
            self.assertRaisesRegex(ValueError, "exactly one"),
        ):
            adapter.install_file_backed_ple(model, weights, self.a.path, self.a.root)

        model, weights = model_and_weights()
        model.model.layers[0].ple.ple_embedding.ngram_embedding = object()
        with (
            sidecar._bound_load_source(self.a.root),
            self.assertRaisesRegex(ValueError, "unmodified"),
        ):
            adapter.install_file_backed_ple(model, weights, self.a.path, self.a.root)

        model, weights = model_and_weights()
        resident = model.model.layers[0].ple.ple_embedding.ngram_embedding
        resident.rows_per_shard += 1
        with (
            sidecar._bound_load_source(self.a.root),
            self.assertRaisesRegex(ValueError, "geometry differs"),
        ):
            adapter.install_file_backed_ple(model, weights, self.a.path, self.a.root)

        model, weights = model_and_weights()
        resident = model.model.layers[0].ple.ple_embedding.ngram_embedding
        resident.shards[1].weight = self.mx.zeros((3, 32))
        with (
            sidecar._bound_load_source(self.a.root),
            self.assertRaisesRegex(ValueError, "dimensions are inconsistent"),
        ):
            adapter.install_file_backed_ple(model, weights, self.a.path, self.a.root)

        model, weights = model_and_weights()
        weights[self.a.prefix + ".alias"] = self.mx.array([0])
        with (
            sidecar._bound_load_source(self.a.root),
            self.assertRaisesRegex(ValueError, "unexpected PLE tensor aliases"),
        ):
            adapter.install_file_backed_ple(model, weights, self.a.path, self.a.root)

        model, weights = model_and_weights()
        fake = SimpleNamespace(parameters=lambda: {"bad": 1})
        with (
            mock.patch.object(adapter, "FileBackedPLEEmbedding", return_value=fake),
            sidecar._bound_load_source(self.a.root),
            self.assertRaisesRegex(AssertionError, "owns MLX parameters"),
        ):
            adapter.install_file_backed_ple(model, weights, self.a.path, self.a.root)

    def test_full_strict_cpu_loader_removes_all_resident_ple_parameters(self):
        from mlx.utils import tree_flatten

        mx = self.mx
        base = self.qwen.Model(self.qwen.ModelArgs.from_dict(self.a.config))
        target = {
            key: value
            for key, value in tree_flatten(base.parameters())
            if self.a.prefix not in key
        }
        mx.save_safetensors(str(self.a.root / "model-target.safetensors"), target)
        self.a.index["weight_map"].update(
            {key: "model-target.safetensors" for key in target}
        )
        self.a.save_index()
        self.a.config["model_file"] = "forbidden.py"
        (self.a.root / "forbidden.py").write_text(
            "raise RuntimeError('custom model executed')"
        )
        (self.a.root / "config.json").write_text(json.dumps(self.a.config))
        from rapid_mlx.models.qwen4_ple_nvme import load_file_backed_qwen4

        loaded, _ = load_file_backed_qwen4(self.a.root, self.a.path, cache_bytes=1064)
        self.assertIs(type(loaded), self.qwen.Model)
        flat = tree_flatten(loaded.parameters())
        self.assertFalse(any(self.a.prefix in key for key, _ in flat))
        table = loaded.model.layers[0].ple.ple_embedding.ngram_embedding
        self.assertIsNotNone(table._reader._fd)
        self.assertIsNone(sidecar._LOAD_READERS.get())
        self.assertEqual(table.parameters(), {})
        self.assertEqual(loaded._ple_offload_receipt["removed_tensors"], 6)
        self.assertEqual(loaded._ple_offload_receipt["resident_ple_tensors"], 0)
        ids = np.array([[0, 7, 0]], dtype=np.int64)
        actual = table(mx.array(ids)).view(mx.uint16)
        mx.eval(actual)
        self.assertTrue(
            np.array_equal(
                np.asarray(actual), sidecar.dequant_rows_numpy(self.a.packed, 32)[ids]
            )
        )
        table(mx.array([[7]], dtype=mx.int64))
        mx.eval(table(mx.array([[7]], dtype=mx.int64)))
        self.assertGreater(table.stats["cache_hits"], 0)
        self.assertLessEqual(table.stats["cache_charged_bytes"], 1064)
        table.close()
        self.assertEqual(mx.default_device(), mx.cpu)

    def test_strict_load_failure_closes_reader_even_with_retained_traceback(self):
        from mlx.utils import tree_flatten

        from rapid_mlx.models import qwen4_ple_nvme as adapter

        mx = self.mx
        model = self.qwen.Model(self.qwen.ModelArgs.from_dict(self.a.config))
        target = {
            key: value
            for key, value in tree_flatten(model.parameters())
            if self.a.prefix not in key
        }
        target.pop("language_model.model.embed_tokens.weight")
        mx.save_safetensors(str(self.a.root / "model-target.safetensors"), target)
        self.a.index["weight_map"].update(
            {key: "model-target.safetensors" for key in target}
        )
        self.a.save_index()
        refs = []
        original_reader = adapter.PLESidecarReader

        def track(*args, **kwargs):
            reader = original_reader(*args, **kwargs)
            refs.append(weakref.ref(reader))
            return reader

        held_error = None
        with mock.patch.object(adapter, "PLESidecarReader", side_effect=track):
            try:
                adapter.load_file_backed_qwen4(self.a.root, self.a.path)
            except ValueError as exc:
                held_error = exc
        self.assertIsNotNone(held_error)
        self.assertIn("embed_tokens.weight", str(held_error))
        reader = refs[0]()
        self.assertIsNotNone(reader)  # retained partial model in traceback
        self.assertIsNone(reader._fd)
        self.assertEqual(reader.stats["cache_rows"], 0)
        self.assertIsNone(sidecar._LOAD_READERS.get())
        self.assertEqual(mx.default_device(), mx.cpu)

    def test_loader_identity_failure_remains_inside_reader_ownership_scope(self):
        import mlx_lm.utils as utils

        from rapid_mlx.models import qwen4_ple_nvme as adapter

        reader = SimpleNamespace(close=mock.Mock())

        def wrong_model(*args, **kwargs):
            sidecar.own_load_reader(reader)
            return SimpleNamespace(), self.a.config

        with mock.patch.object(utils, "load_model", side_effect=wrong_model):
            with self.assertRaisesRegex(RuntimeError, "exact vendored"):
                adapter.load_file_backed_qwen4(self.a.root, self.a.path)
        reader.close.assert_called_once()

    def test_installer_cleanup_error_does_not_replace_original_validation_error(self):
        from rapid_mlx.models import qwen4_ple_nvme as adapter

        model = self.qwen.Model(self.qwen.ModelArgs.from_dict(self.a.config))
        reader = sidecar.PLESidecarReader(self.a.root, self.a.path, random_rows=0)
        real_close = reader.close
        self.addCleanup(real_close)
        reader.close = mock.Mock(side_effect=OSError("cleanup failed"))
        with mock.patch.object(adapter, "PLESidecarReader", return_value=reader):
            with self.assertRaisesRegex(ValueError, "missing or invalid"):
                with sidecar._bound_load_source(self.a.root):
                    adapter.install_file_backed_ple(model, {}, self.a.path, self.a.root)

    def test_arbitrary_source_override_refused_before_model_construction(self):
        args = self.qwen.ModelArgs.from_dict(
            dict(
                self.a.config,
                ple_nvme_sidecar=str(self.a.path),
                ple_nvme_model_path=str(self.a.root),
            )
        )
        with self.assertRaisesRegex(ValueError, "source must be bound"):
            self.qwen.Model(args)
        with sidecar._bound_load_source(self.a.root / "different"):
            with self.assertRaisesRegex(ValueError, "source must be bound"):
                self.qwen.Model(args)

    def _production_lane(
        self, stack, *, load_error=None, tokenizer_error=None, chat_template="existing"
    ):
        import mlx_lm.utils as mlx_utils

        from rapid_mlx import model_aliases
        from rapid_mlx.models import qwen4_ple_nvme as adapter
        from rapid_mlx.utils import chat_template_registry
        from rapid_mlx.utils import tokenizer as loader

        stack.enter_context(
            mock.patch.dict(
                "os.environ",
                {
                    "RAPID_MLX_QWEN4_PLE_NVME": str(self.a.path),
                    "RAPID_MLX_QWEN4_PLE_CACHE_BYTES": "1064",
                },
            )
        )
        stack.enter_context(
            mock.patch.object(model_aliases, "resolve_profile", return_value=None)
        )
        for name in ("_resolve_subfolder_checkpoint", "_local_snapshot_if_cached"):
            stack.enter_context(
                mock.patch.object(loader, name, return_value=str(self.a.root.resolve()))
            )
        for name in (
            "validate_local_model_file",
            "_try_inject_mtp_post_load",
            "_apply_chat_template_sidecar",
            "augment_eos_token_ids_from_generation_config",
            "repair_byte_level_decoder",
            "_post_load_ubc_evict",
        ):
            stack.enter_context(mock.patch.object(loader, name))
        stack.enter_context(
            mock.patch.object(loader, "_model_requires_remote_code", return_value=False)
        )
        stack.enter_context(
            mock.patch.object(
                loader, "apply_remote_code_policy", return_value=({}, False)
            )
        )
        stack.enter_context(
            mock.patch.object(
                loader, "_neutralize_unbundled_template_types", return_value={}
            )
        )
        stack.enter_context(
            mock.patch.object(chat_template_registry, "resolve_chat_template")
        )
        model = SimpleNamespace()
        safe = stack.enter_context(
            mock.patch.object(
                adapter,
                "load_file_backed_qwen4",
                return_value=(model, self.a.config),
                side_effect=load_error,
            )
        )
        close = stack.enter_context(mock.patch.object(adapter, "close_file_backed_ple"))
        stack.enter_context(
            mock.patch.object(
                mlx_utils,
                "load_tokenizer",
                return_value=SimpleNamespace(chat_template=chat_template),
                side_effect=tokenizer_error,
            )
        )
        fallback = stack.enter_context(
            mock.patch.object(loader, "_load_model_with_fallback_impl")
        )
        return loader, safe, close, fallback, model

    def test_production_optin_binds_resolved_source_and_never_falls_back(self):
        with ExitStack() as stack:
            loader, safe, close, fallback, model = self._production_lane(stack)
            result = loader.load_model_with_fallback(
                "alias", lazy=True, return_config=True, return_source=True
            )
            safe.assert_called_once_with(
                str(self.a.root.resolve()),
                str(self.a.path),
                cache_bytes=1064,
                lazy=True,
            )
            self.assertIs(result[0], model)
            self.assertEqual(result[-1], str(self.a.root.resolve()))
            fallback.assert_not_called()
            close.assert_not_called()
        with ExitStack() as stack:
            loader, safe, close, fallback, model = self._production_lane(
                stack, load_error=ValueError("invalid sidecar")
            )
            with self.assertRaisesRegex(ValueError, "invalid sidecar"):
                loader.load_model_with_fallback("alias")
            fallback.assert_not_called()

    def test_production_applies_sidecar_chat_template_when_missing(self):
        with ExitStack() as stack:
            loader, _, _, _, _ = self._production_lane(stack, chat_template=None)
            loader.load_model_with_fallback("alias")
            loader._apply_chat_template_sidecar.assert_called_once()

    def test_production_postload_failure_closes_reader(self):
        with ExitStack() as stack:
            loader, safe, close, fallback, model = self._production_lane(
                stack, tokenizer_error=RuntimeError("tokenizer failed")
            )
            with self.assertRaisesRegex(RuntimeError, "tokenizer failed"):
                loader.load_model_with_fallback("alias")
            close.assert_called_once_with(model)
            fallback.assert_not_called()

    def test_bad_sanitized_tensor_refuses_before_replacement(self):
        mx = self.mx
        model = self.qwen.Model(self.qwen.ModelArgs.from_dict(self.a.config))
        weights = {
            key.replace(self.a.source_prefix, self.a.prefix).replace(
                ".shard_", ".shards."
            ): mx.array(values).view(mx.bfloat16)
            if not key.endswith(".weight")
            else mx.array(values)
            for key, values in self.a.tensors.items()
        }
        weights.pop(next(iter(weights)))
        from rapid_mlx.models.qwen4_ple_nvme import install_file_backed_ple

        with (
            sidecar._bound_load_source(self.a.root),
            self.assertRaisesRegex(ValueError, "missing or invalid"),
        ):
            install_file_backed_ple(model, weights, self.a.path, self.a.root)
        self.assertIsInstance(
            model.model.layers[0].ple.ple_embedding.ngram_embedding,
            self.qwen.ShardedEmbedding,
        )


if __name__ == "__main__":
    unittest.main()
