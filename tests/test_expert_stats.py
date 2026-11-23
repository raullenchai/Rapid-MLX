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

    def tearDown(self):
        expert_stats.WINDOW = 256
        expert_stats.reset()

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
            def named_modules(self):
                return [("language_model.layers.12.mlp.experts", module)]

        self.assertEqual(expert_stats.bind(Model()), 1)
        module(None, [[4, 7]])
        self.assertEqual(expert_stats.snapshot()["layers"][0]["il"], 12)

    def test_route_is_absent_until_enabled(self):
        with self.assertRaises(HTTPException) as caught:
            experts()
        self.assertEqual(caught.exception.status_code, 404)
        expert_stats._enabled = True
        self.assertEqual(experts()["layers"], [])


if __name__ == "__main__":
    unittest.main()
