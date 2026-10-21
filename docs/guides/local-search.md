# Local folder search

Install Rapid-MLX on an Apple Silicon Mac, restart Terminal, and start a local server with folder search enabled:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
# Restart Terminal, then:
rapid-mlx serve qwen3.5-4b-4bit --enable-local-search
```

In another Terminal window, add a folder and search it:

```bash
curl -sS http://127.0.0.1:8000/v1/local-search/folders \
  -H 'Content-Type: application/json' \
  -d "{\"path\":\"$HOME/Documents\"}"
curl -sS http://127.0.0.1:8000/v1/local-search/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"project notes"}'
```

The server prints its actual port; replace `8000` if it chose another. Folder indexing runs after registration. Check progress with `curl -sS http://127.0.0.1:8000/v1/local-search/status`.

## Semantic search

Folder search works with keywords until the separate embedding checkpoint is available. For semantic results, pull the catalog alias, then check status again:

```bash
rapid-mlx pull embeddinggemma-2-4bit
curl -sS http://127.0.0.1:8000/v1/local-search/status
```

The embedding model is separate from `--embedding-model` for `/v1/embeddings`. Folder search does not download it automatically. The status response reports model availability and indexing progress. Searches report `mode` so clients can tell when keyword-only results are in use.

## Supported content and privacy

- Text, Markdown, source code, JSON, YAML, CSV, and HTML are searchable by text.
- PDFs with an extractable text layer and Office documents (DOCX, PPTX, XLSX) are searchable by extracted text. Scan-only PDFs need OCR first.
- Common image formats and sampled video frames can be searched semantically when the embedding model is installed and the installed decoders support the files.
- Hidden files, secret-looking files, system folders, application bundles, symlinks, and known private locations are excluded. The index stays on this Mac.

The default SQLite index is under `~/.rapid-mlx/local-search/`. Use `--local-search-dir /absolute/index-dir` with `serve` to choose a dedicated directory. Deleting a registered folder removes its searchable records; ordinary SQLite deletion is not secure erasure.

## API routes

- `GET /v1/local-search/status` reports model, folders, and indexing state.
- `POST /v1/local-search/folders` registers `{"path":"/absolute/folder"}`.
- `DELETE /v1/local-search/folders/{id}` removes a registered folder.
- `POST /v1/local-search/reindex` queues a scan; send `{"folder_id":null}` for all folders.
- `POST /v1/local-search/search` accepts `{"query":"text"}` and optional folder, kind, and result-limit filters.

These routes are available only when `--enable-local-search` is set. The normal API key and rate limit rules apply. Run `rapid-mlx serve --help` for current server options.
