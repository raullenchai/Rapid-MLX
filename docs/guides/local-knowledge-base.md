# Build a private knowledge base on your Mac

Put your shop's FAQs and policies in a folder, then ask questions and get answers with citations. Everything runs on your Mac. Use the Python script below or a chat app with a document feature.

## What you need

- An Apple Silicon Mac.
- Rapid-MLX installed. Use the installer below; see [Installation](../getting-started/installation.md) for details.
- A folder of text documents.

```sh
curl -fsSL https://rapidmlx.com/install.sh | bash
```

Local Search supports TXT, Markdown, PDFs with a text layer, DOCX, PPTX, XLSX, and HTML. DOC, RTF, Pages, and scanned PDFs are not supported. See the [full file-type table](local-search.md) for details.

## Pick a model for your Mac's memory

Choose the Smart pick for your Mac's memory:

| Mac memory | Chat model | Approximate model RAM |
| --- | --- | --- |
| 8 GB | `lfm2.5-2.6b-4bit` | 3 GB |
| 16 GB | `qwen3.5-4b-4bit` | 6 GB |
| 18 GB | `qwen3.5-9b-4bit` | 9 GB |
| 24 GB | `bonsai-27b-2bit` | 13 GB |
| 32 GB and more | `qwen3.8-27b-4bit` | 20 GB |

The knowledge-base pieces also use `embeddinggemma-2-4bit`, a 1.1 GB download. This model turns text into numbers that help find related passages.

Run `rapid-mlx recipe` to print the recommendation for this Mac. On a 256 GB Mac, it printed:

```
Recommended for this 256.0 GB Mac (96 GB tier)

1. Smart — qwen3.8-27b-4bit · cached
   20.0 GB RAM · 92% capability
   rapid-mlx serve qwen3.8-27b-4bit
```

The examples below use `qwen3.5-4b-4bit`, the 16 GB pick. Substitute your model. On 8–16 GB Macs, close other large apps. Smaller models follow the “answer only from the documents” instruction less reliably, so check their answers.

## Step 1: Put your documents in one folder

```sh
export KB=~/shop-kb
mkdir -p "$KB/docs"
# copy your FAQ, policy, and price files into $KB/docs
```

This guide uses a synthetic yarn shop, Maple & Thread. Its sample folder has seven files: `classes.md`, `gift-cards.md`, `loyalty-program.md`, `returns-policy.md`, `shipping.md`, `store-hours.txt`, and `wholesale-faq.md`.

For example, `returns-policy.md` contains:

```markdown
# Returns and refunds

Maple & Thread accepts returns of unused items within 30 days of delivery.
Items must be in original packaging with tags attached.

- Refunds go back to the original payment method within 5 business days of
  receiving the return.
- Sale items marked "Final sale" cannot be returned or exchanged.
- Hand-dyed yarn that has been wound into a ball cannot be returned.
- Return shipping is free for exchanges; for refunds we deduct a $6 label fee.

To start a return, email returns@mapleandthread.example with your order number.
```

Use one topic per file and plain headings. Put the facts customers ask about in sentences. The answers shown below come from this sample; your answers will depend on your files.

## Step 2: Start the server with your knowledge base

```sh
rapid-mlx pull embeddinggemma-2-4bit
rapid-mlx serve qwen3.5-4b-4bit --port 8461 \
  --enable-local-search --local-search-dir "$KB/index" \
  --embedding-model embeddinggemma-2-4bit
```

- `--enable-local-search` turns on folder indexing and search.
- `--local-search-dir` keeps this index separate. The default directory is `~/.rapid-mlx/local-search`.
- `--embedding-model` enables `/v1/embeddings`. It is only needed for the chat-app and build-your-own sections below.

Local Search never downloads its embedding model on its own. That is why the first command pulls it explicitly.

The server prints `Ready: http://127.0.0.1:8461`. Leave it running. Open a second Terminal window and run `export KB=~/shop-kb` again there. Use that window for the next steps.

## Step 3: Import the folder

Folder paths must be absolute. `$KB` expands to an absolute path:

```sh
curl -s -X POST localhost:8461/v1/local-search/folders \
  -H 'Content-Type: application/json' \
  -d "{\"path\": \"$KB/docs\"}"
```

Output, trimmed:

```json
{"id":"f7e4fa9c092049a6bd7bed0be32072d1","path":"/Users/you/shop-kb/docs","state":"queued","files_indexed":0,"chunks":0,...}
```

Check progress:

```sh
curl -s localhost:8461/v1/local-search/status | python3 -m json.tool
```

In the completed sample run, status showed these fields:

| Status field | Value |
| --- | --- |
| Folder `state` | `idle` |
| Folder `files_indexed` | `7` |
| Folder `chunks` | `7` |
| Folder `files_skipped` | `0` |
| Model `downloaded` | `true` |
| Model `loaded` | `true` |

Indexing the seven small files took a few seconds. A folder moves from `queued` to `indexing`, then `idle` when done. `waiting_for_model` means the embedding model is missing; search is keyword-only until you pull it.

Try a search:

```sh
curl -s -X POST localhost:8461/v1/local-search/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "do points expire", "limit": 3}'
```

The sample search returned `mode: "hybrid"` in 10–65 ms (the first search is the slowest). Its two hits were:

| Rank | File | Line | Matched |
| --- | --- | --- | --- |
| 1 | `loyalty-program.md` | 1 | `["semantic", "keyword"]` |
| 2 | `gift-cards.md` | 1 | `["semantic"]` |

The path and line number give you a citation. Scores rank results; they are not probabilities.

## Step 4: Ask questions with citations (Python)

Set up a Python environment:

```sh
python3 -m venv "$KB/venv"
"$KB/venv/bin/pip" install openai requests
```

Save this as `$KB/ask.py`:

```python
"""Answer a question from your local knowledge base, with citations."""

import sys
from pathlib import Path

import requests
from openai import OpenAI

BASE = "http://127.0.0.1:8461"
client = OpenAI(base_url=f"{BASE}/v1", api_key="not-needed")


def retrieve(question, limit=3):
    r = requests.post(
        f"{BASE}/v1/local-search/search",
        json={"query": question, "limit": limit, "kinds": ["text"]},
        timeout=60,
    )
    r.raise_for_status()
    passages = []
    for hit in r.json()["results"]:
        if not hit.get("content_indexed", True):
            continue  # matched by file name only; no readable text
        # The snippet is a ~500-character excerpt of the matched chunk.
        # Find it in the file and add the text around it. PDFs and Office
        # files are not plain text, so they fall back to the snippet alone.
        try:
            source = Path(hit["path"]).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            source = ""
        pos = source.find(hit["snippet"]) if hit["snippet"] else -1
        text = source[max(0, pos - 400) : pos + 1100] if pos >= 0 else hit["snippet"]
        passages.append((f"{Path(hit['path']).name}:{hit['line'] or 1}", text))
    return passages


def ask(question):
    passages = retrieve(question)
    if not passages:
        return "I couldn't find that in the documents.", []
    context = "\n\n".join(f"[{src}]\n{text}" for src, text in passages)
    reply = client.chat.completions.create(
        model="default",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a customer-support assistant. Answer ONLY from the "
                    "documents below. Cite the source in square brackets, e.g. "
                    "[shipping.md:1]. If the documents do not contain the answer, "
                    "say you don't know and suggest contacting the shop.\n\n"
                    + context
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0.2,
        max_tokens=300,
        reasoning_effort="none",
    )
    return reply.choices[0].message.content, [src for src, _ in passages]


if __name__ == "__main__":
    answer, sources = ask(" ".join(sys.argv[1:]))
    print(answer)
    print("Sources:", ", ".join(sources))
```

Run:

```sh
cd "$KB"
venv/bin/python ask.py "Can I return yarn I already wound into a ball?"
venv/bin/python ask.py "How much is express shipping and how fast is it?"
venv/bin/python ask.py "Do you offer gift wrapping?"
```

The sample run returned these answers, in order:

```
No, you cannot return hand-dyed yarn that has been wound into a ball. [returns-policy.md:1]
Sources: returns-policy.md:1, wholesale-faq.md:1
```

```
Express shipping costs $18 and takes 2 business days. [shipping.md:1]
Sources: shipping.md:1
```

```
I don't have information about gift wrapping services in the provided documents. You may want to contact the shop directly at (503) 555-0147 during store hours.
Sources: gift-cards.md:1, store-hours.txt:1, shipping.md:1
```

The script searches for the top three passages, then puts them in a system prompt that asks the model to answer only from those passages and cite them. `reasoning_effort="none"` turns off thinking for fast, short answers. `model="default"` uses whichever chat model the server loaded.

Search returns only a short snippet of the matched passage, so the script finds that snippet in the file and adds the surrounding text. This works for UTF-8 text files such as TXT and Markdown; for PDF and Office files the model sees only the snippet. See [the grounded-chat tracking issue](https://github.com/raullenchai/Rapid-MLX/issues/5575).

The plain chat endpoint does not look at your documents by itself. Asked directly in the sample check, the model gave a generic answer: “Most clothing and craft stores have a specific policy...”

## Keep it up to date

After editing files, rescan:

```sh
curl -s -X POST localhost:8461/v1/local-search/reindex \
  -H 'Content-Type: application/json' -d '{"folder_id": null}'
```

This returns HTTP 202. Wait for the folder to return to `idle`.

In the sample check, we appended “Holiday hours: we close at 3 pm on December 24 and December 31.” to `store-hours.txt`, then reindexed. Running `venv/bin/python ask.py "What time do you close on Christmas Eve?"` returned:

```
We close at 3 pm on December 24. [store-hours.txt:1]
Sources: store-hours.txt:1
```

While running, the server also rescans folders about every 30 minutes. There is no instant file watcher. To remove a folder, use `DELETE /v1/local-search/folders/{id}` with its folder ID.

## Use it from a chat app (Open WebUI)

Rapid-MLX does not yet ground chat in the Local Search index ([tracking issue](https://github.com/raullenchai/Rapid-MLX/issues/5575)). Use a chat app with its own document feature and point both its chat and embedding settings at Rapid-MLX.

This path was verified with Open WebUI 0.13.0. It needs uv; see [Install with uv](../getting-started/installation.md#install-with-uv) for the uv installer. Keep the Rapid-MLX server running and start the chat app in another Terminal window, with `export KB=~/shop-kb` set there too:

```sh
uv venv --python 3.11 "$KB/owui"
VIRTUAL_ENV="$KB/owui" uv pip install open-webui
DATA_DIR="$KB/owui-data" WEBUI_AUTH=False ENABLE_OLLAMA_API=False \
  OPENAI_API_BASE_URL=http://127.0.0.1:8461/v1 OPENAI_API_KEY=not-needed \
  RAG_EMBEDDING_ENGINE=openai \
  RAG_OPENAI_API_BASE_URL=http://127.0.0.1:8461/v1 RAG_OPENAI_API_KEY=not-needed \
  RAG_EMBEDDING_MODEL=embeddinggemma-2-4bit \
  "$KB/owui/bin/open-webui" serve --host 127.0.0.1 --port 8462
```

1. Open <http://127.0.0.1:8462>.
2. Go to **Workspace → Knowledge**, create **Shop FAQ**, and upload the files from `$KB/docs`.
3. In a new chat, pick `qwen3.5-4b-4bit` (or your chosen chat model).
4. Type `#`, select **Shop FAQ**, and ask a question.

The sample answers included Open WebUI's numbered citations:

```
No, you cannot return hand-dyed yarn that has already been wound into a ball. According to the return policy, hand-dyed yarn that has been wound into a ball cannot be returned [1].
```

```
Yes, Stitch Club points do expire. They expire 12 months after they are earned [1].
```

Open WebUI keeps its own copy of the documents and its own index in `DATA_DIR`, here `$KB/owui-data`. This is separate from Local Search. Update the documents in the chat app when they change.

`WEBUI_AUTH=False` is for single-user use on this Mac only. Binding to `127.0.0.1` keeps the app off the network. The model list also shows the embedding model; do not pick it for chat.

Open WebUI does not add EmbeddingGemma's task prefixes. Results were fine on this sample, but retrieval quality can be lower than Local Search.

## Privacy

The Rapid-MLX server listens on `127.0.0.1` only by default. In this setup, documents, embeddings, and answers stay on your Mac. The Local Search index is a SQLite database in `$KB/index`, with owner-only permissions.

Models download once from Hugging Face. After that, no document content is sent anywhere in these local workflows. `rapid-mlx telemetry` shows and changes the usage-telemetry setting.

Removing a folder deletes its index entries. It is not secure erasure: SQLite can retain deleted content in unused pages or companion files. See [Local folder search](local-search.md) for details.

## Limits

- There is no built-in grounded chat yet ([tracking issue](https://github.com/raullenchai/Rapid-MLX/issues/5575)). Use the script or the chat app's document feature.
- Only one folder indexes at a time.
- There is no OCR. Scanned PDFs need OCR first.
- DOC, RTF, and Pages are unsupported. In a sample check, an RTF file appeared in status as `skip_reasons: {"Unsupported file type": 1}`.
- CSV, JSON, and YAML are keyword-only, not semantic.
- Only the first 64 chunks, roughly 75 KB, of each file are embedded. The rest stays keyword-only.
- Small models can still miss or blend facts. Keep a human in the loop for refunds and legal answers.

## Troubleshooting

| What you see | Cause and next step |
| --- | --- |
| `{"error":{"message":"path must be absolute without '..'", ... "code":"invalid_folder_path"}}` | A relative path such as `docs` was submitted. Use a full path such as `"$KB/docs"`. |
| Folder `waiting_for_model`, or search `mode: "keyword"` | The embedding model is missing. Run `rapid-mlx pull embeddinggemma-2-4bit`, then poll status. |
| 404 on `/v1/local-search/...` | Restart the server with `--enable-local-search`. |
| `/v1/embeddings` returns 503, `no_embedding_model` | Restart the server with `--embedding-model embeddinggemma-2-4bit`. |
| An answer ignores documents or makes things up | Check the Sources line, ask narrower questions, and use the bigger model for your RAM tier. |
| Files are skipped | Check `skip_reasons` in status for the cause. |
| Indexing pauses with `indexing.paused_reason: "chat"` | This is normal while chatting. |

## Build your own RAG with embeddings

Retrieval-augmented generation (RAG) means finding relevant text before asking the chat model to answer. For your own retrieval system, request vectors from `/v1/embeddings`:

```sh
curl -s http://127.0.0.1:8461/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"embeddinggemma-2-4bit","input":["task: search result | query: When do you close on Saturday?"],"dimensions":256}'
```

The response has `"model":"mlx-community/embeddinggemma-2-4bit"`, a 256-number embedding, and `"usage":{"prompt_tokens":16,"total_tokens":16}`.

Save this ranking example as `$KB/embed_rank.py`:

```python
from pathlib import Path

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8461/v1", api_key="not-needed")
MODEL = "embeddinggemma-2-4bit"

# This example reads plain-text files only (PDF/Office need text extraction first).
docs = {
    p.name: p.read_text(encoding="utf-8")
    for p in sorted(Path("docs").iterdir())
    if p.is_file() and p.suffix.lower() in {".md", ".txt"}
}
if not docs:
    raise SystemExit("No .md or .txt files in docs/")


def embed(texts):
    resp = client.embeddings.create(model=MODEL, input=texts, dimensions=256)
    return [d.embedding for d in resp.data]


# EmbeddingGemma 2 expects task prefixes: one for documents, one for queries.
doc_vecs = embed([f"title: {name} | text: {text}" for name, text in docs.items()])
query = "Is there a fee if I send something back for a refund?"
(q_vec,) = embed([f"task: search result | query: {query}"])

# Vectors are normalized, so the dot product is the cosine similarity.
scores = sorted(
    ((sum(a * b for a, b in zip(q_vec, v)), name) for name, v in zip(docs, doc_vecs)),
    reverse=True,
)
for score, name in scores[:3]:
    print(f"{score:.3f}  {name}")
```

Run `cd "$KB" && venv/bin/python embed_rank.py`. The sample output was:

```
0.708  returns-policy.md
0.645  gift-cards.md
0.641  shipping.md
```

The last digit can vary slightly between runs.

Whole files work as documents only when they are short. For real documents, split text into roughly 1,000-character chunks and store vectors in a vector database or SQLite. Queries and documents must use the same dimensions. See [Embeddings](embeddings.md) for endpoint details and [Rerank](rerank.md) for reranking results.

## Clean up

Stop the server with **Ctrl-C**. If you started Open WebUI, stop it with **Ctrl-C** in its Terminal window too. `rm -rf "$KB/index"` deletes the Local Search index. You can also delete `$KB/owui-data` to remove Open WebUI's document copy and index.

## See also

- [Local folder search](local-search.md)
- [Embeddings](embeddings.md)
- [Rerank](rerank.md)
- [AI client compatibility](ai-clients.md)
- [Python API](python-api.md)
