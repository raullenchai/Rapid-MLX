# SPDX-License-Identifier: Apache-2.0
"""Recorder behavior for GET /experts, without a model or a GPU."""

from __future__ import annotations

import inspect
import unittest

from fastapi import HTTPException

from rapid_mlx import expert_stats
from rapid_mlx.routes.experts import experts


class ExpertStatsTest(unittest.TestCase):
    def setUp(self):
        expert_stats.reset()
        expert_stats._patched.clear()

    def tearDown(self):
        expert_stats.WINDOW = 256
        expert_stats.reset()
        expert_stats._patched.clear()

    def test_rows_flatten_a_batch_of_tokens(self):
        self.assertEqual(expert_stats._rows([[1, 2], [3, 4]]), [[1, 2], [3, 4]])
        self.assertEqual(expert_stats._rows([[[8, 1], [2, 3]]]), [[8, 1], [2, 3]])
        self.assertEqual(expert_stats._rows([4, 5, 6]), [[4, 5, 6]])

    def test_note_keeps_a_window_and_the_running_count(self):
        expert_stats._set_counts((4, 2))
        expert_stats.note(1, [[0, 1]])
        expert_stats.note(1, [[2, 3]])
        # The same experts on the next token are a real routing, not a
        # second projection of the previous one.
        expert_stats.note(1, [[2, 3]])
        body = expert_stats.snapshot()
        self.assertEqual(body["n_expert"], 4)
        self.assertEqual(body["n_expert_used"], 2)
        self.assertEqual(body["n_tokens"], 3)
        self.assertEqual(body["window"], 256)
        layer = body["layers"][0]
        self.assertEqual(layer["il"], 1)
        self.assertEqual(layer["tokens"], [[0, 1], [2, 3], [2, 3]])
        self.assertEqual(layer["recent"], [1, 1, 2, 2])
        self.assertEqual(layer["n_tokens"], 3)

    def test_window_drops_the_oldest_hit(self):
        expert_stats.WINDOW = 2
        expert_stats.note(0, [[0]])
        expert_stats.note(0, [[1]])
        expert_stats.note(0, [[1]])
        layer = expert_stats.snapshot()["layers"][0]
        self.assertEqual(layer["tokens"], [[1], [1]])
        self.assertEqual(layer["recent"], [0, 2])
        self.assertEqual(layer["n_tokens"], 3)

    def test_nested_call_records_once(self):
        class Block:
            def __call__(self, x, indices):
                return Inner()(x, indices)

        class Inner:
            def __call__(self, x, indices):
                return indices

        expert_stats._patch(Block)
        expert_stats._patch(Inner)
        expert_stats._enabled = True
        block = Block()
        inner = Inner()
        block._expert_stats_layer = 3
        inner._expert_stats_layer = 3
        block(None, [1, 2])
        body = expert_stats.snapshot()
        self.assertEqual(body["n_tokens"], 1)
        self.assertEqual(body["layers"][0]["tokens"], [[1, 2]])

    def test_patch_keeps_the_call_signature(self):
        class Block:
            def __call__(self, x, indices):
                return indices

        expert_stats._patch(Block)
        self.assertEqual(
            tuple(inspect.signature(Block.__call__).parameters),
            ("self", "x", "indices"),
        )

    def test_fused_call_records_once_and_a_repeat_still_counts(self):
        class Linear:
            def __call__(self, x, indices, sorted_indices=False):
                return indices

        class Block:
            def __init__(self):
                self.up = Linear()
                self.down = Linear()

            def __call__(self, x, indices):
                raise AssertionError("fusion replaced this body")

        def fused(self, x, indices):
            self.up(x, indices, sorted_indices=False)
            self.down(x, indices)
            return indices

        Block.__call__ = fused
        expert_stats._patch(Block)
        expert_stats._patch(Linear)
        expert_stats._enabled = True
        block = Block()
        block._expert_stats_layer = 4
        block.up._expert_stats_layer = 4
        block.down._expert_stats_layer = 4
        block(None, [3, 5])
        block(None, [3, 5])
        body = expert_stats.snapshot()
        self.assertEqual(body["n_tokens"], 2)
        self.assertEqual(body["layers"][0]["tokens"], [[3, 5], [3, 5]])

    def test_bind_stamps_the_layer_in_the_module_name(self):
        class Receiver:
            def __call__(self, x, indices):
                return indices

        expert_stats._patch(Receiver)
        expert_stats._enabled = True
        module = Receiver()

        class Model:
            model_type = "dummy"

            def named_modules(self):
                return [("language_model.layers.12.mlp.experts", module)]

        self.assertEqual(expert_stats.bind(Model()), 1)
        module(None, [[4, 7]])
        body = expert_stats.snapshot()
        self.assertEqual(body["layers"][0]["il"], 12)
        self.assertEqual(body["model"], "dummy")

    def test_bind_clears_the_previous_model_window(self):
        class Receiver:
            def __call__(self, x, indices):
                return indices

        expert_stats._patch(Receiver)
        expert_stats._enabled = True
        first = Receiver()
        second = Receiver()

        class Model:
            def __init__(self, module, name):
                self._module = module
                self.model_type = name

            def named_modules(self):
                return [("layers.0.experts", self._module)]

        expert_stats.bind(Model(first, "one"))
        first(None, [[1]])
        self.assertEqual(expert_stats.snapshot()["n_tokens"], 1)
        expert_stats.bind(Model(second, "two"))
        body = expert_stats.snapshot()
        self.assertEqual(body["n_tokens"], 0)
        self.assertEqual(body["layers"], [])
        self.assertEqual(body["model"], "two")
        second(None, [[9]])
        self.assertEqual(expert_stats.snapshot()["layers"][0]["tokens"], [[9]])

    def test_unbind_retires_only_the_owning_model(self):
        class Receiver:
            def __call__(self, x, indices):
                return indices

        expert_stats._patch(Receiver)
        expert_stats._enabled = True
        module = Receiver()

        class Model:
            model_type = "owned"

            def named_modules(self):
                return [("layers.1.experts", module)]

        model = Model()
        expert_stats.bind(model)
        module(None, [[2]])
        expert_stats.unbind(object())
        self.assertEqual(expert_stats.snapshot()["n_tokens"], 1)
        expert_stats.unbind(model)
        self.assertEqual(expert_stats.snapshot()["layers"], [])

    def test_bind_rewraps_after_fusion_replaces_the_callable(self):
        class Switch:
            def __call__(self, x, indices):
                return ("stock", indices)

        expert_stats._patch(Switch)
        expert_stats._enabled = True

        def fused(self, x, indices):
            return ("fused", indices)

        Switch.__call__ = fused
        self.assertFalse(getattr(Switch.__call__, "_expert_stats_wrapped", False))

        module = Switch()

        class Model:
            def named_modules(self):
                return [("layers.0.mlp", module)]

        expert_stats.bind(Model())
        self.assertTrue(getattr(Switch.__call__, "_expert_stats_wrapped", False))
        module(None, [4, 5])
        self.assertEqual(expert_stats.snapshot()["n_tokens"], 1)

    def test_expert_stats_keeps_inkling_decode_eager(self):
        from rapid_mlx.models.inkling2 import layers as inkling_layers

        expert_stats._enabled = False
        self.assertFalse(inkling_layers._expert_stats_active())
        expert_stats._enabled = True
        self.assertTrue(inkling_layers._expert_stats_active())

    def test_route_is_absent_until_enabled(self):
        with self.assertRaises(HTTPException) as caught:
            experts()
        self.assertEqual(caught.exception.status_code, 404)
        expert_stats._enabled = True
        self.assertEqual(experts()["layers"], [])


if __name__ == "__main__":
    unittest.main()
