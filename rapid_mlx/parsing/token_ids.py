# SPDX-License-Identifier: Apache-2.0
"""Opt-in dispatch for parsers that preserve generated token identity."""

import inspect


def supports_token_method(parser, name):
    """Ignore dynamic mock attributes unless the capability was explicit."""
    try:
        inspect.getattr_static(parser, name)
    except AttributeError:
        return False
    return callable(getattr(parser, name, None))


def _dispatch(parser, name, args, ids, **kwargs):
    capability = name + "_with_tokens"
    if ids is not None and supports_token_method(parser, capability):
        kwargs["delta_token_ids" if name.endswith("streaming") else "token_ids"] = ids
        name = capability
    return getattr(parser, name)(*args, **kwargs)


def extract_reasoning(parser, text, *, token_ids=None, **kwargs):
    return _dispatch(parser, "extract_reasoning", (text,), token_ids, **kwargs)


def extract_reasoning_streaming(parser, *texts, delta_token_ids=None):
    return _dispatch(parser, "extract_reasoning_streaming", texts, delta_token_ids)


def extract_tool_calls(parser, text, request=None, *, token_ids=None):
    if token_ids is not None and supports_token_method(
        parser, "extract_tool_calls_with_tokens"
    ):
        return parser.extract_tool_calls_with_tokens(
            text, request=request, token_ids=token_ids
        )
    return parser.extract_tool_calls(text, request)


def extract_tool_calls_streaming(parser, *texts, request=None, delta_token_ids=None):
    return _dispatch(
        parser, "extract_tool_calls_streaming", texts, delta_token_ids, request=request
    )
