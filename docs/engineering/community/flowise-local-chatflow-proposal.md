I maintain [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX). I'd like to propose a small local text-chat recipe using Flowise's **existing OpenAI Custom Model node** with a model served on an Apple Silicon Mac. No new provider node is needed.

The intended path is: official MiniCPM5 MLX weights → Rapid-MLX local API → OpenAI Custom Model → a simple text chatflow. The custom node accepts a free-text model name, which is useful for a local model outside the standard OpenAI model list.

Proposed setup when both backends run directly on the same Mac:

```bash
brew install rapid-mlx
rapid-mlx serve minicpm5-2b-4bit --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/v1/models
```

Node settings:

| Field | Initial value |
| --- | --- |
| Model Name | Exact `data[].id` returned by `/v1/models` |
| Base Path | `http://127.0.0.1:8000/v1` |
| Credential | Placeholder `local-only` if required by the SDK; use a real matching key for an authenticated server |
| Max Tokens | `128` for a short connection check |
| Temperature | `0.7` |
| Streaming | Off for the first check; test on separately |

Connect this to a simple chat chain and ask for a short greeting. A later RAG example needs a separately configured embedder; agent tools require their own tool/result-loop check.

Important deployment detail: the custom node calls `checkDenyList(basePath)`, so an administrator's network restrictions still apply. A container's localhost is not the Mac host, and hosted Flowise cannot reach the Mac's loopback address. The recipe should explain a permitted, authenticated network setup rather than advise disabling those controls.

**Current status:** this is an Ideas proposal, not a tested/exported chatflow. I inspected the current [ChatOpenAICustom implementation](https://github.com/FlowiseAI/Flowise/blob/main/packages/components/nodes/chatmodels/ChatOpenAICustom/ChatOpenAICustom.ts), including `modelName`, `basepath`, streaming and credential handling. I have not run the Flowise UI or exported a flow for this submission. If useful, the deliverable would be a versioned flow JSON, screenshot and basic-chat/streaming results before posting in Show and Tell, as CONTRIBUTING requests.

This post was AI-assisted under the project owner's direction. Rapid-MLX has anonymous metadata telemetry enabled by default; set `DO_NOT_TRACK=1` before launching to opt out. This is a community integration proposal, not an official partnership.

Is a local OpenAI Custom Model recipe useful for the docs/examples, and which chatflow template would be the best starting point?
