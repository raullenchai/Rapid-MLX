# SPDX-License-Identifier: Apache-2.0
"""CPU-only admission contracts; isolate tensor-free methods from runtime imports."""

import ast
import logging
import sys
import time
from contextvars import ContextVar
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from rapid_mlx.config import get_config, reset_config
from rapid_mlx.request import ClientRequestError
from rapid_mlx.service.helpers import context_overflow_from_client_error

ROOT = Path(__file__).resolve().parents[1]


def _load_function(path, name, namespace, class_name=None):
    namespace.setdefault("__name__", "rapid_mlx.mllm_batch_generator")
    namespace.setdefault(
        "__package__",
        "rapid_mlx.engine" if class_name == "BatchedEngine" else "rapid_mlx",
    )
    tree = ast.parse((ROOT / path).read_text())
    if class_name:
        tree = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
        )
    node = next(
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )
    node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace
    )
    return namespace[name]


@pytest.fixture(autouse=True)
def cpu_admission(monkeypatch):
    reset_config()
    prepare = _load_function(
        "rapid_mlx/engine/batched.py", "_prepare_mllm_messages", {}, "BatchedEngine"
    )
    module = ModuleType("rapid_mlx.engine.batched")
    module._admission_token_context = ContextVar("test_token", default=None)
    module._admission_engine_context = ContextVar("test_admission", default=None)
    module.BatchedEngine = SimpleNamespace(_prepare_mllm_messages=prepare)
    monkeypatch.setitem(sys.modules, "rapid_mlx.engine.batched", module)
    generator_module = ModuleType("rapid_mlx.mllm_batch_generator")
    generator_module.__package__ = "rapid_mlx"
    generator_module.ClientRequestError = ClientRequestError
    for name in (
        "_vision_patch_unit",
        "_vision_pixel_ceiling",
        "_effective_vision_pixel_cap",
        "_reduced_vision_pixel_cap",
        "_expand_sized_image_prompt",
    ):
        _load_function(
            "rapid_mlx/mllm_batch_generator.py", name, generator_module.__dict__
        )
    monkeypatch.setitem(sys.modules, "rapid_mlx.mllm_batch_generator", generator_module)

    yield
    reset_config()


@pytest.mark.parametrize(
    "image_type,object_url",
    [("image_url", True), ("input_image", True), ("input_image", False)],
)
def test_mllm_image_preflight_and_immutability(image_type, object_url):
    from tests.test_context_length_exceeded import (
        test_mllm_text_preflight_does_not_count_image_data_as_text,
    )

    test_mllm_text_preflight_does_not_count_image_data_as_text(image_type, object_url)


def test_mllm_capped_preflight_converts_pydantic_tools():
    from tests.test_context_length_exceeded import (
        test_mllm_capped_preflight_converts_pydantic_tools as check,
    )

    check()


@pytest.mark.parametrize("failure", ["missing", "raises", "empty"])
def test_mllm_unavailable_count_is_not_overflow(failure):
    from tests.test_context_length_exceeded import (
        test_operational_prompt_cap_fails_closed_when_mllm_count_unavailable,
    )

    test_operational_prompt_cap_fails_closed_when_mllm_count_unavailable(failure)


@pytest.mark.parametrize("expanded_tokens", [270, 271])
def test_expanded_prompt_operational_cap_before_vision(monkeypatch, expanded_tokens):
    generator_path = "rapid_mlx/mllm_batch_generator.py"
    check = _load_function(
        generator_path,
        "_enforce_preprocessed_context_window",
        {"ClientRequestError": ClientRequestError},
    )
    process = _load_function(
        generator_path,
        "_process_prompt_steps",
        {
            "_enforce_preprocessed_context_window": check,
            "logger": logging.getLogger(__name__),
            "time": time,
        },
        "MLLMBatchGenerator",
    )
    cache_module = ModuleType("mlx_lm.models.cache")
    cache_module.make_prompt_cache = lambda *a: None
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache_module)
    generator = SimpleNamespace(
        model=SimpleNamespace(args=SimpleNamespace(max_position_embeddings=1000)),
        processor=SimpleNamespace(tokenizer=object()),
        _media_mrope_restore=lambda: None,
        _preprocess_request=lambda req: setattr(
            req, "input_ids", SimpleNamespace(size=expanded_tokens)
        ),
    )
    request = SimpleNamespace(max_tokens=100, input_ids=None)
    get_config().max_prompt_tokens = 270

    class AdmittedError(Exception):
        pass

    class Stats:
        @property
        def prompt_tokens(self):
            raise AdmittedError

    generator._stats = Stats()
    if expanded_tokens > 270:
        with pytest.raises(ClientRequestError) as exc:
            next(process(generator, [request], cooperative=False))
        translated = context_overflow_from_client_error(exc.value)
        assert translated is not None
        assert translated.prompt_tokens == expanded_tokens
        assert translated.limit == 270
        assert translated.error_payload["code"] == "context_length_exceeded"
        assert "--max-prompt-tokens" in translated.error_payload["message"]
    else:
        with pytest.raises(AdmittedError):
            next(process(generator, [request], cooperative=False))


@pytest.mark.parametrize(
    "pixel_cap,budget,image_count,sizing_supported",
    [
        (0, 8192, 1, True),
        (65536, 8192, 1, True),
        (0, 128, 1, True),
        (0, 8192, 2, True),
        (0, 8192, 1, False),
    ],
)
def test_count_tokens_includes_sized_image_expansion(
    monkeypatch, pixel_cap, budget, image_count, sizing_supported
):
    import base64
    import io
    from unittest.mock import Mock

    from PIL import Image

    from rapid_mlx.api.tool_calling import convert_tools_for_template
    from rapid_mlx.models import mllm
    from rapid_mlx.models.lfm3_vl_processor import (
        IMAGE_TOKEN_ID,
        LFM3VLProcessor,
        expand_image_tokens,
        preprocess_images,
    )
    from rapid_mlx.service.helpers import count_prompt_tokens

    image_loader = Mock(wraps=mllm.process_image_input)
    monkeypatch.setattr(mllm, "process_image_input", image_loader)
    previous_temp_files = set(mllm._temp_manager._files)

    image = Image.new("RGB", (512, 512))
    data = io.BytesIO()
    image.save(data, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()
    token_ids = [11] * 7 + [IMAGE_TOKEN_ID] * image_count + [22] * 7
    tokenizer = SimpleNamespace(encode=lambda *a, **kw: token_ids)
    processor = (
        LFM3VLProcessor(tokenizer)
        if sizing_supported
        else SimpleNamespace(tokenizer=tokenizer)
    )
    render = _load_function(
        "rapid_mlx/engine/batched.py",
        "render_request_prompt",
        {
            "convert_tools_for_template": convert_tools_for_template,
            "extract_multimodal_content": lambda messages: (
                messages,
                [url] * image_count,
                [],
            ),
        },
        "BatchedEngine",
    )
    engine = SimpleNamespace(
        _loaded=True,
        _is_mllm=True,
        _processor=processor,
        tokenizer=tokenizer,
        _model=SimpleNamespace(
            config=SimpleNamespace(
                vision_config=SimpleNamespace(patch_size=16, spatial_merge_size=2)
            )
        ),
        _mllm_scheduler=SimpleNamespace(
            config=SimpleNamespace(
                vision_min_pixels=0,
                vision_max_pixels=pixel_cap,
                vision_prefill_token_budget=budget,
            )
        ),
        _prepare_cache_stable_messages=lambda m: (m, None),
        _render_chat_prompt=lambda *a, **kw: ("image prompt", 0),
        _prepare_harmony_no_thinking_prompt=lambda *a, **kw: (token_ids, None),
    )
    # Independently execute the generation processor's two-pass layout.
    effective_cap = (
        min(pixel_cap, (budget // image_count) * 1024)
        if pixel_cap
        else (budget // image_count) * 1024
    )
    for _ in range(2):
        _, grids = preprocess_images(
            [image] * image_count, max_pixels=min(effective_cap, 1048576)
        )
        expected = len(expand_image_tokens(token_ids, grids))
        reduced = int(effective_cap * budget / expected)
        if expected <= budget or reduced < 1024:
            break
        effective_cap = reduced
    if pixel_cap == 0 and budget == 8192:
        assert expected == 14 + 256 * image_count
    if not sizing_supported:
        expected = len(token_ids)
    messages = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}
    ]
    prompt = render(engine, messages)
    assert count_prompt_tokens(engine, prompt) == expected

    # Exercise the endpoint with its real adapter and renderer, isolating
    # only engine readiness and request-policy checks from model loading.
    import asyncio
    from unittest.mock import AsyncMock

    from fastapi import HTTPException

    from rapid_mlx.api.anthropic_adapter import (
        AnthropicContentError,
        AnthropicOutputConfigError,
        anthropic_to_openai,
    )
    from rapid_mlx.api.anthropic_models import AnthropicRequest
    from rapid_mlx.api.utils import extract_multimodal_content

    class Engine:
        render_request_prompt = render

    route_engine = Engine()
    route_engine.__dict__.update(engine.__dict__)
    route_engine.is_mllm = True
    route_engine.preserve_native_tool_format = False
    prepare = _load_function(
        "rapid_mlx/routes/anthropic.py",
        "_prepare_anthropic_engine_messages",
        {"extract_multimodal_content": extract_multimodal_content},
    )
    count_endpoint = _load_function(
        "rapid_mlx/routes/anthropic.py",
        "count_anthropic_tokens",
        {
            "HTTPException": HTTPException,
            "get_engine": lambda *a: route_engine,
            "ensure_engine_ready": AsyncMock(),
            "AnthropicRequest": AnthropicRequest,
            "anthropic_to_openai": anthropic_to_openai,
            "AnthropicContentError": AnthropicContentError,
            "AnthropicOutputConfigError": AnthropicOutputConfigError,
            "_apply_anthropic_thinking_defaults": lambda r: None,
            "_prepare_anthropic_engine_messages": prepare,
            "get_config": get_config,
            "_resolve_enable_thinking": lambda r: False,
            "_effective_enable_thinking": lambda *a: False,
            "convert_tools_for_template": convert_tools_for_template,
            "count_prompt_tokens": count_prompt_tokens,
        },
    )
    request = SimpleNamespace(
        json=AsyncMock(
            return_value={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64.b64encode(data.getvalue()).decode(),
                                },
                            }
                        ]
                        * image_count,
                    }
                ]
            }
        ),
        headers={},
    )
    assert asyncio.run(count_endpoint(request)) == {"input_tokens": expected}
    assert image_loader.call_count == (2 * image_count if sizing_supported else 0)
    assert set(mllm._temp_manager._files) == previous_temp_files
