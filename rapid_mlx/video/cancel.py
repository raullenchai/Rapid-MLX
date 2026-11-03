"""Cooperative cancellation at video generation stage and step boundaries."""

import threading
from contextlib import contextmanager
from functools import wraps
from importlib import import_module


class VideoGenerationCancelled(RuntimeError):  # noqa: N818 - shared cancellation contract
    """A video request was cancelled at a safe boundary."""


class VideoCancelToken:
    def __init__(self):
        self._event = threading.Event()

    def cancel(self):
        self._event.set()

    @property
    def cancelled(self):
        return self._event.is_set()

    def check(self):
        if self.cancelled:
            raise VideoGenerationCancelled("Video generation was cancelled.")

    def wait(self, timeout=None):
        return self._event.wait(timeout)


def step_iter(iterable, token):
    for item in iterable:
        token.check()
        yield item
    token.check()


def cancelling_tqdm(original, token):
    """Keep progress-bar behavior while checking even disabled step bars."""

    class Progress:
        def __init__(self, *args, **kwargs):
            token.check()
            self._bar = original(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._bar, name)

        def __iter__(self):
            yield from step_iter(self._bar, token)

        def update(self, *args, **kwargs):
            token.check()
            return self._bar.update(*args, **kwargs)

        def __enter__(self):
            self._bar.__enter__()
            return self

        def __exit__(self, *args):
            return self._bar.__exit__(*args)

    return Progress


@contextmanager
def scoped_step_hooks(token, *module_names):
    """Temporarily hook step bars while the process generation lock is held."""
    patched = []
    try:
        if token is not None:
            token.check()
            for name in module_names:
                try:
                    module = import_module(name)
                except ImportError:
                    continue
                original = getattr(module, "tqdm", None)
                if original is not None:
                    patched.append((module, original))
                    module.tqdm = cancelling_tqdm(original, token)
        yield
    finally:
        for module, original in reversed(patched):
            module.tqdm = original


def cleanup_on_cancel(function):
    """Remove partial output and release activation caches on the worker thread."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            token = kwargs.get("cancel_token")
            if token is not None:
                token.check()
            return function(*args, **kwargs)
        except Exception as exc:
            if not isinstance(exc, VideoGenerationCancelled) and not (
                token is not None and token.cancelled
            ):
                raise
            output = kwargs.get("output_path")
            if output is not None:
                output.unlink(missing_ok=True)
                output.with_suffix(".temp.mp4").unlink(missing_ok=True)
            from .minimax_h3 import _clear_mlx_cache

            _clear_mlx_cache()
            if isinstance(exc, VideoGenerationCancelled):
                raise
            raise VideoGenerationCancelled("Video generation was cancelled.") from exc

    return wrapped
