# LFM3-VL checkpoint layout

The pinned MLX 8-bit revision `83336d5be3874a2bdd3297bde77b2114dbd3a816`
and bf16 revision `de545dfbd29e5db006b7c3320d80b902283e904a` expose
respectively 732 and 424 tensor entries in their safetensors metadata. The
vision depthwise kernels have MLX layout `[1024, 3, 3, 1]` in both. Text
weights in the 8-bit checkpoint use U32 packed weights with BF16 affine scales
and biases; the other checkpoint uses BF16 tensors. The native model's weight
validation accepts the key sets and shapes of both pinned headers. This is a
metadata check only; numerical equivalence and serving qualification remain
separate work.
