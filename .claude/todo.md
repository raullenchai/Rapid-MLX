# TODO

## MTP 实测 Qwen3.5-27B
1. `pip install git+https://github.com/AirRunner/mlx-lm.git@feat/mtp-native`
2. 重新量化 Qwen3.5-27B（MTP 权重会保留）:
   ```bash
   python3.12 -m mlx_lm.convert --hf-path Qwen/Qwen3.5-27B \
     --mlx-path ~/.lmstudio/models/local/Qwen3.5-27B-4bit-mtp \
     --quantize --q-bits 4 --q-group-size 64
   ```
3. 验证 MTP 权重保留: 检查 model.safetensors.index.json 有 mtp.* keys
4. 跑 benchmark: baseline vs --enable-mtp (预期 ~1.5x)
5. 如果 PR #990 效果好，考虑简化我们的 monkey-patch（保留为 fallback）
