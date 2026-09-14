from vllm_mlx.engine import batched
from vllm_mlx.prompt_host_cache import PromptHostCache


class _Tokenizer:
    chat_template = "template-v1"
    init_kwargs = {"_commit_hash": "revision-v1"}

    def __init__(self):
        self.encode_calls = 0

    def encode(self, prompt):
        self.encode_calls += 1
        return [ord(char) for char in prompt]


def _engine(tokenizer, *, max_entries=8, max_bytes=4096):
    engine = batched.BatchedEngine.__new__(batched.BatchedEngine)
    engine._is_mllm = False
    engine._processor = None
    engine._tokenizer = tokenizer
    engine._model_name = "test-model"
    engine._prompt_host_cache = PromptHostCache(
        max_entries=max_entries, max_bytes=max_bytes, enabled=True
    )
    return engine


def test_render_cache_is_exact_and_preserves_mapping_order(monkeypatch):
    tokenizer = _Tokenizer()
    engine = _engine(tokenizer)
    calls = []

    def render(_applicator, messages, **kwargs):
        calls.append((messages, kwargs))
        return f"render-{len(calls)}"

    monkeypatch.setattr(batched, "shared_apply_chat_template", render)
    first = [{"role": "user", "content": "hello", "meta": {"a": 1, "b": 2}}]
    reordered = [{"role": "user", "content": "hello", "meta": {"b": 2, "a": 1}}]

    assert engine._apply_chat_template(first) == "render-1"
    assert engine._apply_chat_template(first) == "render-1"
    assert engine._apply_chat_template(reordered) == "render-2"
    assert len(calls) == 2
    assert engine._prompt_host_cache.stats()["hits_by_kind"]["render"] == 1


def test_combined_lru_enforces_entry_and_byte_limits():
    cache = PromptHostCache(max_entries=2, max_bytes=120, enabled=True)
    one = cache.fingerprint({"one": 1})
    two = cache.fingerprint({"two": 2})
    three = cache.fingerprint({"three": 3})

    assert cache.put_render(one, "a" * 40)
    assert cache.put_render(two, "b" * 40)
    assert cache.get_render(one) == "a" * 40  # one is now MRU
    assert cache.put_render(three, "c" * 40)

    assert cache.get_render(two) is None
    assert cache.get_render(one) == "a" * 40
    stats = cache.stats()
    assert stats["entries"] == 2
    assert stats["current_bytes"] <= stats["max_bytes"]
    assert stats["evictions"] == 1


def test_replacing_an_entry_updates_the_byte_ledger():
    cache = PromptHostCache(max_entries=2, max_bytes=120, enabled=True)
    fingerprint = cache.fingerprint({"same": True})

    assert cache.put_render(fingerprint, "short")
    assert cache.put_render(fingerprint, "a little longer")

    assert cache.get_render(fingerprint) == "a little longer"
    assert cache.stats()["current_bytes"] == len("a little longer")


def test_cache_limits_must_be_positive():
    for max_entries, max_bytes in ((0, 1), (1, 0)):
        try:
            PromptHostCache(max_entries=max_entries, max_bytes=max_bytes)
        except ValueError as exc:
            assert "must be positive" in str(exc)
        else:  # pragma: no cover - assertion branch
            raise AssertionError("non-positive cache limit was accepted")


def test_oversize_and_uncacheable_inputs_fail_open_without_retention():
    cache = PromptHostCache(max_entries=2, max_bytes=8, enabled=True)

    assert cache.fingerprint({"bad": object()}) is None
    assert cache.get_render(None) is None
    assert not cache.put_render(cache.fingerprint({"large": 1}), "too large")
    stats = cache.stats()
    assert stats["entries"] == 0
    assert stats["uncacheable_bypasses"] == 1
    assert stats["oversize_bypasses"] == 1


def test_disabled_cache_never_retains_or_counts_lookup(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_PROMPT_HOST_CACHE", "0")
    cache = PromptHostCache()
    fingerprint = cache.fingerprint({"prompt": "hello"})

    assert not cache.put_render(fingerprint, "hello")
    assert cache.get_render(fingerprint) is None
    assert cache.stats()["entries"] == 0
    assert cache.stats()["lookups"] == 0


def test_clear_can_reset_stats_for_model_lifecycle_isolation():
    cache = PromptHostCache(enabled=True)
    fingerprint = cache.fingerprint({"prompt": "hello"})
    assert cache.put_render(fingerprint, "hello")
    assert cache.get_render(fingerprint) == "hello"

    assert cache.clear(reset_stats=True) == 1
    stats = cache.stats()
    assert stats["entries"] == 0
    assert stats["hits"] == 0
    assert stats["stores"] == 0
    assert stats["invalidations"] == 0
