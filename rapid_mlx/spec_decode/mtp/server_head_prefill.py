# SPDX-License-Identifier: Apache-2.0
"""Bounded, request-local prompt head history for singleton text prefill.

On prefix hits only the computed suffix is captured; head positions start at
zero. Head KV stays out of the shared backbone prefix cache. No model or batch
class is patched globally.
"""


class HeadPrefillCapture:
    def __init__(self, model, mx, quantize_cache_fn):
        from rapid_mlx.tuning.mtp import prefill_block_size

        self.prefill_block_size = prefill_block_size(model)
        self.model = model
        self.mx = mx
        self.quantize_cache_fn = quantize_cache_fn
        self.cache = model.make_mtp_cache()
        self.hidden = None
        self.first_token = None
        self.prompt_length = None

    def __call__(self, inputs, cache=None):
        rows = inputs.shape[1]
        block = self.prefill_block_size
        boundary = rows // block * block if block else rows
        if 0 < boundary < rows:
            # Match the measured prefix-snapshot boundary before capturing head
            # history. Target and head arithmetic both depend on forward width.
            first = self(inputs[:, :boundary], cache=cache)
            last = self(inputs[:, boundary:], cache=cache)
            return self.mx.concatenate([first, last], axis=1)
        logits, hidden = self.model(inputs, cache=cache, return_hidden=True)
        # A chunk's last hidden waits for the next chunk's first token. Never
        # append the last prompt hidden here: it seeds the bootstrap chain.
        paired_hidden = hidden[:, :-1, :]
        paired_ids = inputs[:, 1:]
        if self.hidden is not None:
            paired_hidden = self.mx.concatenate([self.hidden, paired_hidden], axis=1)
            paired_ids = inputs
        if paired_ids.shape[1]:
            self.model.mtp_forward(paired_hidden, paired_ids, self.cache)
            self.quantize_cache_fn(self.cache)
        self.hidden = hidden[:, -1:, :]
        # Finish the graph before the next target chunk; retain only one row.
        # Empty KVCache.state dereferences keys.shape. A one-token suffix
        # has a hidden row but no shifted head pair (and no allocated KV).
        if paired_ids.shape[1]:
            self.mx.eval(self.hidden, [c.state for c in self.cache])
        else:
            self.mx.eval(self.hidden)
        return logits


def install_prompt_capture(batch_gen, model, mx, eligible, quantize_cache_fn):
    """Capture eligible singleton prompt batches; discard on route misses.

    Head work is provisional until the scheduler selects the MTP route. New
    arrivals may still force plain batching; those captures are discarded.
    """
    captures = {}
    prompt_batch = batch_gen._prompt_batch
    original_class = type(prompt_batch)

    class CapturingPromptBatch(original_class):
        def _capture(self):
            if len(self.uids) != 1 or not eligible(self.uids[0]):
                for uid in self.uids:
                    captures.pop(uid, None)
                return None
            uid = self.uids[0]
            if uid not in captures:
                captures[uid] = HeadPrefillCapture(model, mx, quantize_cache_fn)
            return captures[uid]

        def prompt(self, tokens):
            capture = self._capture()
            previous = self.model
            if capture is not None:
                self.model = capture
            try:
                return super().prompt(tokens)
            except BaseException:
                for uid in self.uids:
                    captures.pop(uid, None)
                raise
            finally:
                self.model = previous

        def generate(self, tokens):
            capture = self._capture()
            previous = self.model
            if capture is not None:
                self.model = capture
            try:
                generation = super().generate(tokens)
                generation.model = previous
                if capture is not None:
                    capture.first_token = int(generation._next_tokens[0].item())
                    capture.prompt_length = len(generation.tokens[0])
                return generation
            except BaseException:
                # generate may have emptied self.uids before raising.
                captures.clear()
                raise
            finally:
                self.model = previous

    prompt_batch.__class__ = CapturingPromptBatch

    def close():
        captures.clear()
        prompt_batch.__class__ = original_class

    return captures, close
