# 用 ComfyUI 和 Rapid MLX 在 Mac 上批量生成旅行海报

月球夜市、土星度假村、深海列车：我们给一家虚构的旅行社做了一组去不了的目的地海报。整个生图流程在 Mac 上运行，ComfyUI 负责工作流，Rapid-MLX 负责调用 Qwen-Image 2.1 做 MLX 推理。同一套流程可以换成商品概念图、活动海报或内容配图的批量制作。

![六个虚构目的地的旅行海报](media/contact-sheet.jpg)

## 工作流如何连接

这个 demo 使用两个独立的 Python 进程。浏览器打开 ComfyUI，生成节点通过本机 HTTP 请求 Rapid-MLX。返回的 PNG 被转换成 ComfyUI 的标准 IMAGE 张量，再交给内置 Save Image 节点。

```text
目的地列表和固定 seed
        ↓
ComfyUI 队列 → Rapid-MLX 生图节点
                      ↓ POST /v1/images/generations
                 Rapid-MLX → MLX → Qwen-Image 2.1
                      ↓ base64 PNG
                 ComfyUI Save Image
                      ↓
              原图 + 每张图的执行记录
```

Rapid-MLX 的图像 runtime 集成了 mflux，这个 demo 复用其现有生图 API。这样可以继续使用 ComfyUI 的工作流与队列，同时把模型加载和 Apple Silicon 推理交给 Rapid-MLX。ComfyUI 不需要再安装一份 Qwen 扩散权重，两个进程也可以使用独立的依赖环境。

## 启动本地模型服务

安装 Rapid-MLX 的 image extra，并启动 Qwen-Image 2.1：

```bash
pip install 'rapid-mlx[image]'
rapid-mlx serve qwen-image-2.1 --host 127.0.0.1 --port 18427
```

这个别名使用约 8.9 GiB 的预量化模型下载，包含 MLX 原生 q4 扩散模型与 q4 文本编码器，默认 40 步。本文展示机为 M3 Ultra、256 GB 统一内存；这里的实测不能推导成小内存 Mac 的表现。模型细节见 [Qwen-Image 2.1 使用说明](../../docs/models/families/qwen-image-2.1.md)。

先用一个请求验证服务：

```bash
curl http://127.0.0.1:18427/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-image-2.1","prompt":"A cinematic lunar night market travel poster, with the headline LUNAR NIGHT MARKET","size":"1024x1024","steps":40,"seed":4200,"response_format":"b64_json"}'
```

返回的是 `data[0].b64_json`，内容为 PNG 字节的 base64 编码。响应里的 `cancelled` 可以用于判断生成是否被取消。

## 把 Rapid MLX 接到 ComfyUI

将示例中的 `custom_nodes/rapid_mlx` 目录链接或复制到 ComfyUI 的 `custom_nodes/`，重启 ComfyUI，然后导入 `workflow.json`。完整安装命令和经过验证的 ComfyUI commit 见 [demo README](README.md)。

![从执行历史加载已完成结果的 ComfyUI 工作流](media/workflow.png)

这个流程可以使用 ComfyUI 的 CPU 模式：

```bash
python main.py --cpu --listen 127.0.0.1 --port 8189
```

这里的 `--cpu` 只约束 ComfyUI 自己的 Torch 进程。模型推理发生在另一个 Rapid-MLX 进程里，通过 MLX 使用 Metal。

节点中最关键的处理只有三件事：发送生图参数、解码 PNG、转换像素布局。ComfyUI 的 IMAGE 约定为 `[batch, height, width, channels]`，RGB 浮点像素范围为 0 到 1。

```python
payload = {
    "model": model,
    "prompt": prompt,
    "size": f"{width}x{height}",
    "steps": steps,
    "seed": seed,
    "n": 1,
    "response_format": "b64_json",
}
# POST payload to /v1/images/generations, decode data[0].b64_json
pixels = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
return (torch.from_numpy(pixels).unsqueeze(0),)
```

Torch 在这里承载的是返回的像素。完整节点还处理 HTTP 错误、连接失败、取消和空结果；如果服务需要 API key，从 ComfyUI 进程环境读取，避免把 key 写进工作流或图片元数据。实现依据是 [ComfyUI 官方图片张量约定](https://docs.comfy.org/custom-nodes/backend/images_and_masks)。

## 批量生成六个目的地

每张海报使用共同的美术方向：复古未来主义、电影感光线、有纵深的建筑与微小旅客。具体场景、标题和 seed 放在 `destinations.json`。例如月球夜市的场景是玻璃穹顶、灯笼、面摊和天空中的地球；土星度假村则是漂浮在星环上方的装饰艺术酒店。

```bash
python3 examples/comfyui-travel/batch.py \
  --output examples/comfyui-travel/media/posters
```

脚本逐张向 ComfyUI 的 `/prompt` 提交任务，轮询 `/history/{prompt_id}`，确认 Save Image 成功后通过 `/view` 下载图片，再提交下一张。这种批量制作指的是连续处理多个任务；Rapid-MLX 的图像通道一次执行一张，避免并发扩散管线争用统一内存。[ComfyUI 官方 API 文档](https://docs.comfy.org/development/comfyui-server/comms_routes)介绍了这些队列和历史接口。

每张图同时保存一个 JSON 记录，其中包括 prompt、seed、分辨率、步数、ComfyUI 任务 ID、完整请求耗时和图片 SHA256。再次运行时，脚本会检查已有文件和参数，跳过完整的输出。修改参数后使用新的输出目录，就能保留不同实验的证据。

固定 seed 方便对比，但不保证跨版本、硬件得到逐字节相同的图片。ComfyUI 也会缓存相同输入的节点结果；想要新样本就改变 seed。中断 ComfyUI 不会自动取消 Rapid-MLX 已经收到的 HTTP 请求，超时后应先检查正在运行的任务。

## 这次 demo 的实测

在 M3 Ultra 上，六张入选图片的服务端生成总耗时约为每张 189–205 秒，参数均为 1024×1024、40 步。客户端记录还包含 ComfyUI 排队与轮询时间。这是一轮使用已缓存权重的 demo 实测，不能作为受控硬件比较；[完整执行记录](VERIFIED.md)给出了版本、seed 和逐张耗时。

## 图片里的文字与最终排版

每个 prompt 明确给出标题、副标题和底部的 `RAPID TRAVEL`，原始海报中的文字由模型生成。这里没有把标题贴回原图来掩盖模型的文字表现，因此发布前需要检查拼写、重复文字和可读性。

这一轮里，月球海报曾把 `DARK` 拼成 `DANK`，土星海报也出现了副标题字母错误。我保留了两个原始候选，缩短副标题并更换 seed 重跑，最终展示使用检查过的版本。

不同目的地是独立生成的图片，共同 prompt 可以帮助保持视觉方向，但不等于角色或建筑身份一致性的保证。制作连续剧情时，需要额外设计参考图或编辑流程。

## 制作适合分享的展示片

```bash
python examples/comfyui-travel/render.py \
  --input examples/comfyui-travel/media/posters \
  --output examples/comfyui-travel/media
```

脚本生成一张六宫格预览和一段 1080p、H.264、无声展示片，每个目的地停留三秒，再以两秒 Rapid-MLX 品牌页收尾。FFmpeg 为静态海报加入轻微推近和淡入淡出；画面外的品牌说明由脚本排版。

这段片子是海报动画，不是 Qwen 生成的连续视频。若要继续做真正的图生视频，可以把选好的原图交给 Rapid-MLX 的视频模型，但那需要独立的模型与验证；本文的可复现范围是本地批量生图。

## 换成自己的内容

把 `destinations.json` 换成你的商品、活动或故事场景，保留统一美术方向，再逐张调整具体信息。先运行 `--limit 1` 检查服务与画面，再跑完整批次。图片、执行参数、工作流和用于分享的素材都在本地，Rapid-MLX 提供推理服务，ComfyUI 把它组织成可以重复执行的创作流程。
