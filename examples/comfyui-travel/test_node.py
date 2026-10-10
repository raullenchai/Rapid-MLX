# SPDX-License-Identifier: Apache-2.0
"""Contract tests for network/error and tensor boundaries; no model weights."""

import base64
import importlib.util
import io
import json
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from PIL import Image

spec = importlib.util.spec_from_file_location(
    "travel_node", Path(__file__).parent / "custom_nodes/rapid_mlx/__init__.py"
)
node = importlib.util.module_from_spec(spec)
spec.loader.exec_module(node)


class NodeContract(unittest.TestCase):
    def generate(self):
        return node.RapidMLXImage().generate(
            "http://127.0.0.1:1", "qwen-image-2.1", "test", 256, 256, 40, 42
        )

    def test_rgb_tensor_and_request_contract(self):
        png = io.BytesIO()
        Image.new("RGBA", (3, 2), (255, 0, 128, 255)).save(png, "PNG")
        response = io.BytesIO(
            json.dumps(
                {"data": [{"b64_json": base64.b64encode(png.getvalue()).decode()}]}
            ).encode()
        )
        with patch.object(
            node.urllib.request, "urlopen", return_value=response
        ) as call:
            (image,) = self.generate()
        self.assertEqual(tuple(image.shape), (1, 2, 3, 3))
        self.assertAlmostEqual(float(image[0, 0, 0, 0]), 1.0)
        self.assertEqual(float(image[0, 0, 0, 1]), 0.0)
        self.assertEqual(json.loads(call.call_args.args[0].data)["seed"], 42)

    def test_http_errors_preserve_server_reason(self):
        error = urllib.error.HTTPError(
            "http://localhost", 409, "Conflict", {}, io.BytesIO(b"wrong_image_endpoint")
        )
        with (
            patch.object(node.urllib.request, "urlopen", side_effect=error),
            self.assertRaisesRegex(RuntimeError, "409.*wrong_image_endpoint"),
        ):
            self.generate()

    def test_cancelled_and_missing_images_fail(self):
        for result in ({"cancelled": True, "data": []}, {"data": []}):
            with (
                self.subTest(result=result),
                patch.object(
                    node.urllib.request,
                    "urlopen",
                    return_value=io.BytesIO(json.dumps(result).encode()),
                ),
                self.assertRaises(RuntimeError),
            ):
                self.generate()

    def test_unreachable_server_is_actionable(self):
        with (
            patch.object(
                node.urllib.request,
                "urlopen",
                side_effect=urllib.error.URLError("refused"),
            ),
            self.assertRaisesRegex(RuntimeError, "Start rapid-mlx serve"),
        ):
            self.generate()


if __name__ == "__main__":
    unittest.main()
