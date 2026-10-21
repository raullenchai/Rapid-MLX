# Rerank

Install Rapid-MLX on an Apple Silicon Mac and restart Terminal:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
```

rapid-mlx provides a Cohere/Jina-compatible `/v1/rerank` endpoint for the model
loaded with `--rerank-model`. Run it next to a chat model or alone:

```bash
rapid-mlx serve --rerank-model mlx-community/Qwen3-Reranker-0.6B-mxfp8
```

In another Terminal window:

```bash
curl http://localhost:8000/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{"query": "capital of China", "documents": ["Beijing is the capital.", "Cookies were invented in 1938."], "top_n": 1}'
```

## Supported Models

Each reranker is trained on one input format, so Rapid-MLX detects the model
family at startup and builds the format that family expects. Other checkpoints
are rejected at startup. When a model's name does not reveal its family, for
example a fine-tune or a renamed local copy, name the family with
`--rerank-model-family`; the model type and architecture must still match.

Supported reranker families include Qwen3, BGE, mxbai, XLM-RoBERTa, and Jina. Run `rapid-mlx serve --help` for `--rerank-model-family` when a local checkpoint name does not identify its family. Use the [model catalog](https://rapidmlx.com/docs/aliases.html) to choose a current checkpoint.

The generative families and jina-reranker-v3 run on the base install.
XLM-RoBERTa cross-encoders need the `[embeddings]` extra. jina-reranker-v3 is
licensed CC BY-NC 4.0 (non-commercial use only).

The generative families and XLM-RoBERTa cross-encoders report the sigmoid of
the model's relevance logit as `relevance_score`.

jina-reranker-v3 is listwise: it reads the query and up to 125 documents in one
prompt and scores each document by the similarity of its hidden state to the
query's, so a document's score depends on the other documents in the request.
Its `relevance_score` is that cosine similarity mapped to 0 to 1. Its query is
limited to 512 tokens and `--rerank-max-length` applies to each document.

## Request and Response

`documents` accepts strings or `{"text": ...}` objects. Results are sorted by
`relevance_score` (0 to 1) and include the document text unless
`return_documents` is `false`; `top_n` limits the result count.

## Query, Instruction and Prompt

Besides the documents, a reranker's input is built from up to three texts:

- **Query**: the search text to rank documents against. Always sent per
  request in the `query` field.
- **Instruction**: the retrieval task the query belongs to. Instruction-aware
  rerankers rank best with a task-specific instruction. The optional
  `instruction` request field sets it per request; otherwise
  `--rerank-instruction` applies, and without it the family default. Clients
  that only send the standard rerank fields get the server's instruction.
- **Prompt**: the fixed judging prompt that tells the model how to rate
  relevance. Rerankers are trained with one fixed prompt, so it can only be
  replaced server-wide with `--rerank-prompt`.

For a family without an instruction, the `instruction` request field is
rejected with `400`. For a family without an instruction or judging prompt,
`--rerank-instruction` or `--rerank-prompt` stops the server at startup.

## Limits

Two limits apply. `--rerank-overflow-policy` governs only per-document token length:

- `--rerank-max-documents N` (default: 1000): excess documents always produce
  `400` with `code: "too_many_documents"` before scoring, under either overflow
  policy. Set 0 or a negative value to disable the cap.
- `--rerank-max-length` (`auto`: the family's pair length from the table,
  capped at the model's maximum): with `truncate`, a document that does not
  fit together with the query is shortened and a warning is logged; with
  `error`, the request is rejected with `400` and `code: "input_too_long"`.

Empty `documents` and an empty `instruction` return `400`. Omit `instruction`
to use the server or family default.

When the shared retrieval worker has 64 pending jobs (running plus queued),
new requests receive `503` with `code: "server_busy"` and `Retry-After: 1`.
Retry after one second. Cancelled requests occupy a slot until their work finishes.

## Model Management

Without `--rerank-model` the endpoint answers `503` with
`code: "no_rerank_model"`. `--rerank-model` pins `/v1/rerank` to one model at
startup; there is no hot-swap. A request whose `model` names a different model
gets `400` with `code: "model_not_found"`.
