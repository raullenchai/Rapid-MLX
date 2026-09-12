# SPDX-License-Identifier: Apache-2.0
"""Small synthetic sidecars and CPU-only Rapid load/lookup contracts."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('vllm_mlx.models.qwen4_ple_sidecar', ROOT/'vllm_mlx/models/qwen4_ple_sidecar.py')
sidecar = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sidecar
spec.loader.exec_module(sidecar)


def tiny_config():
    return dict(model_type='qwen4_exp', text_config=dict(hidden_size=8, num_hidden_layers=2,
        vocab_size=32, num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        linear_num_key_heads=1, linear_num_value_heads=3, linear_key_head_dim=4,
        linear_value_head_dim=4, linear_conv_kernel_dim=3, num_experts=4,
        num_experts_per_tok=2, moe_intermediate_size=4, shared_expert_intermediate_size=4,
        hc_count=4, hc_lowrank=3, layer_types=['linear_attention','full_attention'],
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=4, indexer_budget=8,
        indexer_compress_ratio=2, ple_layer_ids=[1], eos_token_id=31, ple_embed_dim=32,
        ngram_size=2, heads_per_ngram=1, ngram_vocab_size_base=7,
        make_ngram_vocab_size_divisible_by=2, split_ngram_parts=2))


class Artifact:
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / 'ple_rows.bin'
        self.prefix = 'language_model.model.layers.0.ple.ple_embedding.ngram_embedding'
        rng = np.random.default_rng(9)
        self.words = rng.integers(0, 2**32, (8, 4), dtype=np.uint32)
        self.scales = np.full((8, 1), 0x3d80, dtype=np.uint16)
        self.biases = np.full((8, 1), 0xbe00, dtype=np.uint16)
        self.packed = np.concatenate([self.words.view(np.uint8), self.scales.view(np.uint8), self.biases.view(np.uint8)], axis=1)
        self.path.write_bytes(self.packed.tobytes())
        self.tensors = {}
        header, data = {}, b''
        for shard in range(2):
            for part, array, dtype in [('weight', self.words, 'U32'), ('scales', self.scales, 'BF16'), ('biases', self.biases, 'BF16')]:
                key = f'{self.prefix}.shard_{shard}.{part}'
                values = array[shard*4:shard*4+4]
                raw = values.tobytes()
                header[key] = dict(dtype=dtype, shape=list(values.shape), data_offsets=[len(data), len(data)+len(raw)])
                self.tensors[key] = values
                data += raw
        raw_header = json.dumps(header).encode()
        (self.root/'model-ple.safetensors').write_bytes(struct.pack('<Q',len(raw_header))+raw_header+data)
        self.index = {'weight_map': {name:'model-ple.safetensors' for name in header}}
        self.config = tiny_config()
        (self.root/'config.json').write_text(json.dumps(self.config))
        self.manifest = dict(format='qwen4-ple-rows', version=1, tensor_prefix=self.prefix, dims=32,
            group_size=32,bits=4,mode='affine',weight_bytes=16,scales_bytes=2,biases_bytes=2,
            row_bytes=20,num_shards=2,rows_per_shard=4,total_rows=8,data_offset=0,
            shard_sha256=[hashlib.sha256(self.packed[i*4:i*4+4].tobytes()).hexdigest() for i in range(2)])
        self.save_index()

    def save_index(self):
        blob=json.dumps(self.index).encode()
        (self.root/'model.safetensors.index.json').write_bytes(blob)
        self.manifest['source_index_sha256']=hashlib.sha256(blob).hexdigest()
        self.save_manifest()

    def save_manifest(self):
        Path(str(self.path)+'.manifest.json').write_text(json.dumps(self.manifest))


class SidecarContracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.a=Artifact(self.tmp.name)

    def test_validation_and_exact_integer_affine(self):
        receipt=sidecar.validate_artifact(self.a.root,self.a.path)
        self.assertGreaterEqual(receipt['checked_rows'],4)
        actual=sidecar.dequant_rows_numpy(self.a.packed,32)
        q=((self.a.words[...,None] >> (np.arange(8,dtype=np.uint32)*4)) &15).reshape(8,32)
        # All these values are exactly representable; this independent formula
        # uses float64 arithmetic before the final expected BF16 bit view.
        expected=(q.astype(np.float64)/16-0.125).astype(np.float32).view(np.uint32)
        self.assertTrue(np.array_equal(actual,(expected>>16).astype(np.uint16)))

    def test_lookup_shape_bounds_close_and_cache(self):
        reader=sidecar.PLESidecarReader(self.a.root,self.a.path,cache_bytes=2*532)
        self.addCleanup(reader.close)
        ids=np.array([[0,1,0],[2,7,2]],dtype=np.int64)
        self.assertTrue(np.array_equal(reader.lookup_bits(ids),sidecar.dequant_rows_numpy(self.a.packed,32)[ids]))
        reader.lookup_bits(np.array([7],dtype=np.int64))
        self.assertGreater(reader.stats['cache_hits'],0)
        self.assertGreater(reader.stats['cache_evictions'],0)
        self.assertLessEqual(reader.stats['cache_charged_bytes'],2*532)
        for ids in (np.array([-1]),np.array([8])):
            with self.assertRaises(IndexError): reader.lookup_bits(ids)
        with self.assertRaises(ValueError): reader.lookup_bits(np.array([0.5]))
        self.assertEqual(reader.lookup_bits(np.array([],dtype=np.int64)).shape,(0,32))
        reader.close()
        with self.assertRaises(RuntimeError): reader.lookup_bits(np.array([0]))

    def test_corrupt_edge_and_index_refused(self):
        raw=bytearray(self.a.path.read_bytes()); raw[0]^=1; self.a.path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'content mismatch'): sidecar.validate_artifact(self.a.root,self.a.path,random_rows=0)
        self.a.manifest['source_index_sha256']='0'*64; self.a.save_manifest()
        with self.assertRaisesRegex(ValueError,'digest'): sidecar.validate_artifact(self.a.root,self.a.path)

    def test_geometry_dtype_and_missing_source_refused(self):
        self.a.manifest['row_bytes']=99; self.a.save_manifest()
        with self.assertRaises(ValueError): sidecar.validate_artifact(self.a.root,self.a.path)
        self.a.manifest['row_bytes']=20; self.a.index['weight_map'].pop(next(iter(self.a.index['weight_map'])))
        self.a.save_index()
        with self.assertRaisesRegex(ValueError,'exactly'): sidecar.validate_artifact(self.a.root,self.a.path)

    def test_nonfinite_parameters_refused(self):
        packed=self.a.packed.copy(); packed[0,16:18]=np.array([0x7f80],dtype=np.uint16).view(np.uint8)
        with self.assertRaisesRegex(ValueError,'nonfinite'): sidecar.dequant_rows_numpy(packed,32)

    def test_reader_detects_changed_file(self):
        reader=sidecar.PLESidecarReader(self.a.root,self.a.path); self.addCleanup(reader.close)
        self.a.path.write_bytes(self.a.path.read_bytes()+b'x')
        with self.assertRaisesRegex(RuntimeError,'changed'): reader.lookup_bits(np.array([0]))

    def test_atomic_replacement_during_validation_refused(self):
        original=sidecar.validate_artifact
        def replace(*args,**kwargs):
            receipt=original(*args,**kwargs)
            other=self.a.path.with_suffix('.replacement'); other.write_bytes(self.a.path.read_bytes())
            other.replace(self.a.path)
            return receipt
        with mock.patch.object(sidecar,'validate_artifact',side_effect=replace):
            with self.assertRaisesRegex(RuntimeError,'changed during'): sidecar.PLESidecarReader(self.a.root,self.a.path)


class CPULoadContracts(SidecarContracts):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        cls.mx=mx
        from vllm_mlx.models import qwen4_exp
        cls.qwen=qwen4_exp

    def test_full_strict_cpu_loader_removes_all_resident_ple_parameters(self):
        from mlx.utils import tree_flatten
        from mlx_lm.utils import load_model
        mx=self.mx
        base=self.qwen.Model(self.qwen.ModelArgs.from_dict(self.a.config))
        target={key:value for key,value in tree_flatten(base.parameters()) if self.a.prefix not in key}
        mx.save_safetensors(str(self.a.root/'model-target.safetensors'),target)
        self.a.index['weight_map'].update({key:'model-target.safetensors' for key in target})
        self.a.save_index()
        self.a.config['model_file']='forbidden.py'
        (self.a.root/'forbidden.py').write_text("raise RuntimeError('custom model executed')")
        (self.a.root/'config.json').write_text(json.dumps(self.a.config))
        from vllm_mlx.models.qwen4_ple_nvme import load_file_backed_qwen4
        loaded,_=load_file_backed_qwen4(self.a.root,self.a.path,cache_bytes=1064)
        self.assertIs(type(loaded),self.qwen.Model)
        flat=tree_flatten(loaded.parameters())
        self.assertFalse(any(self.a.prefix in key for key,_ in flat))
        table=loaded.model.layers[0].ple.ple_embedding.ngram_embedding
        self.assertEqual(table.parameters(),{})
        self.assertEqual(loaded._ple_offload_receipt['removed_tensors'],6)
        self.assertEqual(loaded._ple_offload_receipt['resident_ple_tensors'],0)
        ids=np.array([[0,7,0]],dtype=np.int64)
        actual=table(mx.array(ids)).view(mx.uint16); mx.eval(actual)
        self.assertTrue(np.array_equal(np.asarray(actual),sidecar.dequant_rows_numpy(self.a.packed,32)[ids]))
        table(mx.array([[7]],dtype=mx.int64)); mx.eval(table(mx.array([[7]],dtype=mx.int64)))
        self.assertGreater(table.stats['cache_hits'],0)
        self.assertLessEqual(table.stats['cache_charged_bytes'],1064)
        table.close()
        self.assertEqual(mx.default_device(),mx.cpu)

    def test_arbitrary_source_override_refused_before_model_construction(self):
        args=self.qwen.ModelArgs.from_dict(dict(self.a.config,ple_nvme_sidecar=str(self.a.path),
                                                ple_nvme_model_path=str(self.a.root)))
        with self.assertRaisesRegex(ValueError,'source must be bound'):
            self.qwen.Model(args)
        with sidecar._bound_load_source(self.a.root/'different'):
            with self.assertRaisesRegex(ValueError,'source must be bound'):
                self.qwen.Model(args)

    def _production_lane(self, stack, *, load_error=None, tokenizer_error=None):
        from vllm_mlx.utils import tokenizer as loader
        from vllm_mlx.models import qwen4_ple_nvme as adapter
        from vllm_mlx import model_aliases
        from vllm_mlx.utils import chat_template_registry
        import mlx_lm.utils as mlx_utils
        stack.enter_context(mock.patch.dict('os.environ', {'RAPID_MLX_QWEN4_PLE_NVME':str(self.a.path),
                                                          'RAPID_MLX_QWEN4_PLE_CACHE_BYTES':'1064'}))
        stack.enter_context(mock.patch.object(model_aliases,'resolve_profile',return_value=None))
        for name in ('_resolve_subfolder_checkpoint','_local_snapshot_if_cached'):
            stack.enter_context(mock.patch.object(loader,name,return_value=str(self.a.root.resolve())))
        for name in ('validate_local_model_file','_try_inject_mtp_post_load','_apply_chat_template_sidecar',
                     'augment_eos_token_ids_from_generation_config','repair_byte_level_decoder','_post_load_ubc_evict'):
            stack.enter_context(mock.patch.object(loader,name))
        stack.enter_context(mock.patch.object(loader,'_model_requires_remote_code',return_value=False))
        stack.enter_context(mock.patch.object(loader,'apply_remote_code_policy',return_value=({},False)))
        stack.enter_context(mock.patch.object(loader,'_neutralize_unbundled_template_types',return_value={}))
        stack.enter_context(mock.patch.object(chat_template_registry,'resolve_chat_template'))
        model=SimpleNamespace()
        safe=stack.enter_context(mock.patch.object(adapter,'load_file_backed_qwen4',
            return_value=(model,self.a.config),side_effect=load_error))
        close=stack.enter_context(mock.patch.object(adapter,'close_file_backed_ple'))
        stack.enter_context(mock.patch.object(mlx_utils,'load_tokenizer',
            return_value=SimpleNamespace(chat_template='existing'),side_effect=tokenizer_error))
        fallback=stack.enter_context(mock.patch.object(loader,'_load_model_with_fallback_impl'))
        return loader,safe,close,fallback,model

    def test_production_optin_binds_resolved_source_and_never_falls_back(self):
        with ExitStack() as stack:
            loader,safe,close,fallback,model=self._production_lane(stack)
            result=loader.load_model_with_fallback('alias',lazy=True,return_config=True,return_source=True)
            safe.assert_called_once_with(str(self.a.root.resolve()),str(self.a.path),cache_bytes=1064,lazy=True)
            self.assertIs(result[0],model)
            self.assertEqual(result[-1],str(self.a.root.resolve()))
            fallback.assert_not_called(); close.assert_not_called()
        with ExitStack() as stack:
            loader,safe,close,fallback,model=self._production_lane(stack,load_error=ValueError('invalid sidecar'))
            with self.assertRaisesRegex(ValueError,'invalid sidecar'):
                loader.load_model_with_fallback('alias')
            fallback.assert_not_called()

    def test_production_postload_failure_closes_reader(self):
        with ExitStack() as stack:
            loader,safe,close,fallback,model=self._production_lane(stack,tokenizer_error=RuntimeError('tokenizer failed'))
            with self.assertRaisesRegex(RuntimeError,'tokenizer failed'):
                loader.load_model_with_fallback('alias')
            close.assert_called_once_with(model)
            fallback.assert_not_called()

    def test_bad_sanitized_tensor_refuses_before_replacement(self):
        mx=self.mx
        model=self.qwen.Model(self.qwen.ModelArgs.from_dict(self.a.config))
        weights={key.replace('.shard_', '.shards.'): mx.array(values).view(mx.bfloat16) if not key.endswith('.weight') else mx.array(values)
                 for key,values in self.a.tensors.items()}
        weights.pop(next(iter(weights)))
        from vllm_mlx.models.qwen4_ple_nvme import install_file_backed_ple
        with sidecar._bound_load_source(self.a.root), self.assertRaisesRegex(ValueError,'missing or invalid'):
            install_file_backed_ple(model,weights,self.a.path,self.a.root)
        self.assertIsInstance(model.model.layers[0].ple.ple_embedding.ngram_embedding,self.qwen.ShardedEmbedding)


if __name__=='__main__':
    unittest.main()
