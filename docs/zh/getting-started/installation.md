# 安装

## 系统要求

- 搭载 Apple Silicon（M1/M2/M3/M4/M5）的 macOS
- Python 3.10+

## 使用 uv 安装（推荐）

```bash
git clone https://github.com/waybarrios/vllm-mlx.git
cd vllm-mlx

uv pip install -e .
```

## 使用 pip 安装

```bash
git clone https://github.com/waybarrios/vllm-mlx.git
cd vllm-mlx

pip install -e .
```

### 可选：视觉支持

使用 transformers 进行视频处理：

```bash
pip install -e ".[vision]"
```

### 可选：音频支持（STT/TTS）

```bash
pip install mlx-audio
```

### 可选：向量嵌入

```bash
pip install mlx-embeddings
```

## 安装内容说明

- `mlx`, `mlx-lm`, `mlx-vlm` - MLX 框架及模型库
- `transformers`, `tokenizers` - HuggingFace 库
- `opencv-python` - 视频处理
- `gradio` - 对话界面
- `psutil` - 资源监控
- `mlx-audio`（可选）- 语音转文字与文字转语音
- `mlx-embeddings`（可选）- 文本向量嵌入

## 验证安装

```bash
# Check CLI commands
vllm-mlx --help
vllm-mlx-bench --help
vllm-mlx-chat --help

# Test with a small model
vllm-mlx-bench --model mlx-community/Llama-3.2-1B-Instruct-4bit --prompts 1
```

## 故障排查

### 找不到 MLX

请确认您使用的是 Apple Silicon 设备：
```bash
uname -m  # Should output "arm64"
```

### 模型下载失败

请检查网络连接及 HuggingFace 访问权限。部分模型需要身份验证：
```bash
huggingface-cli login
```

### 内存不足

请使用更小的量化模型：
```bash
vllm-mlx serve mlx-community/Llama-3.2-1B-Instruct-4bit
```
