# Local folder search (MVP pilot)

Local search indexes user-selected folders and searches their text, images, and
sampled video frames with EmbeddingGemma 2 plus keyword retrieval. It is opt-in
and independent
of the model locked by `--embedding-model` for `/v1/embeddings`.

```sh
rapid-mlx serve <chat-model> --enable-local-search
# Optional: put the index in a chosen persistent directory.
rapid-mlx serve <chat-model> --enable-local-search --local-search-dir /absolute/index-dir
```

Without `--enable-local-search`, all local-search routes return 404 and are
absent from the API schema. The default index is
`~/.rapid-mlx/local-search/index.sqlite3`; `RAPID_MLX_STATE_DIR` changes the
state directory. The index stores extracted text and vectors locally in a
single SQLite database with WAL companion files. Delete a folder registration
to remove its searchable files and chunks. SQLite may retain deleted content
in free pages or WAL until it reuses/checkpoints them; deletion is not secure
erasure.

The index directory is owner-only (`0700`); the database and SQLite WAL,
shared-memory and rollback-journal files are owner-only (`0600`), regardless of
umask. The database is created with this mode before SQLite first opens it;
new sidecars inherit its mode. On open, existing index files are tightened only
when owned by the current user. The index directory itself is tightened only
when it contains exclusively the database and its `-wal`, `-shm`, and `-journal`
companions; a directory already at `0700` may contain unrelated entries because
its permissions need no change. Parent directories are never chmodded. Choose
a dedicated index directory. Foreign ownership, symlinks, hard-linked or
non-regular index files, and permission-repair failures refuse opening; status
returns a readable structured `store_open_failed` error (503).

Provision `mlx-community/embeddinggemma-2-4bit` explicitly with `rapid-mlx pull`
before expecting semantic results. These routes never download a model. While
it is absent, files are indexed for keywords, folders report
`waiting_for_model`, and searches return `mode: "keyword"`. Polling status after
provisioning queues waiting folders for embedding. The separate embedding
engine loads on the first indexing/search call that needs it. It remains
resident until server shutdown; existing idle residency management does not
cover this embedding engine.

Supported document types:

| Type | Extracted content | Locator |
| --- | --- | --- |
| UTF-8 text, Markdown, source code, JSON, YAML, CSV, HTML | Text; HTML tags/scripts/styles removed | One-based starting line |
| PDF | Pages with extractable text; mixed PDFs retain their text pages | One-based `page` |
| DOCX | Body paragraphs and tables, headers/footers, footnotes/endnotes, comments | One-based starting line in extracted text |
| PPTX | Slide text and speaker notes, in presentation order | One-based slide number in `page` |
| XLSX | Shared/inline strings and stored cell values, in workbook sheet order | One-based sheet index in `page` |
| DOC, XLS, PPT, RTF, Pages | Unsupported; convert to a supported format first | — |

Office extraction uses ZIP/XML without additional dependencies. Spreadsheet
extraction reads at most 10,000 cells per sheet; formulas contribute their stored
values if present and are not evaluated. The 20 MiB cap also bounds total
uncompressed Office archive bytes and extracted output; oversized archives are
rejected before inflation. Office archives admit at most 4,096 entries and a
1,000:1 compression ratio per entry. Each XML part admits at most 64 levels
and 100,000 elements; entity declarations are rejected. Presentation slides
and workbook sheets are capped at 1,000 per file.
PDFs have no page-count cap; cumulative extracted text and the parser deadline
bound extraction. External Office relationships are not followed.
PDFs with pages but fewer than three alphanumeric characters across their
extracted text are classified as scanned/image-only. Encrypted PDFs are skipped,
even when an empty password could unlock them. There is no OCR: scanned PDFs
need prior OCR, and image indexing does not add keyword-searchable text.

Images: PNG, JPG/JPEG, WebP, BMP, GIF (first frame), and TIF/TIFF. HEIC works
only if the installed Pillow can already decode it; otherwise it is skipped as
`Unsupported file type`. No additional HEIC dependency is installed. Videos: MP4, MOV,
M4V, MKV, and WebM, subject to the installed OpenCV decoder's codec support.
Video sampling seeks to one frame every five seconds. Videos longer than the
120-frame budget instead sample 120 evenly spaced positions from the first to
last frame. Timestamps are seconds at the selected frame (rounded to the
nearest frame). Consecutive samples with mean absolute RGB difference below
2 on a 0–255 scale after downscaling to 32×32 are omitted. At most four decoded
frames are retained per extraction batch; the full video is never buffered or
decoded. Corrupt/undecodable media is skipped without aborting the folder.
Audio is unsupported and its checkpoint weights are always dropped. If the
lazily imported PDF parser is unavailable, PDFs are counted as unsupported skips.

The per-file caps are 20 MiB for text, 50 MiB for images, and 4 GiB for videos.
Images and video frames admit at most 67,108,864 decoded pixels and 16,384
pixels on either axis. Image dimensions are checked before pixel loading; Pillow
decompression-bomb warnings and errors skip the file as “Too large”. Video
metadata is checked before frame reads, and actual RGB-compatible uint8 frames
are checked before resizing/conversion (at most 48 MiB of decoded pixel data).
The pixel checks bound what the engine keeps and converts. The native video
decoder’s own transient allocations for a malformed file are not bounded by
these checks.
Files producing more than 65,536 text chunks are skipped as “Too large”.
Folder discovery descends at most 64 directory levels; deeper directories are
omitted. There is no limit on the number of entries across the tree.
Only the first 64 text chunks per file (roughly 75 KB, including PDF, DOCX, PPTX, and XLSX
text) are embedded; all remaining chunks stay keyword-searchable with their
line or page locators. The entire index retains at most 262,144 embedded chunks
(`MAX_CORPUS_EMBEDDED_CHUNKS`), across all folders and modalities. Later files
remain keyword-only; removing indexed files frees capacity for reindexing.
Files partially embedded because the corpus budget ran out are recorded as
incomplete and upgraded on a later reindex when additional capacity is available.
Unchanged complete files and budget-limited files with no additional room do
not re-embed.
Images need one free slot; videos conservatively need 120 free slots before
embedding, so a nearly full corpus may keep a video keyword-only. This bounds the
cost of large generated logs and data files and reduces repetitive semantic
candidates. Existing indexes keep their chunks and first 64 text vectors;
excess vectors are removed on opening without re-embedding.
Conventional credentials, secrets, passwords, password vaults (`*.kdbx`),
`.netrc`, RSA keys, and service-account JSON filenames share the central private
exclusion policy and the “Private or system file” skip reason.
Hidden and secret-looking entries, Git,
node_modules, virtual environments, build/cache directories, application bundles
and other macOS packages (including photo libraries, frameworks, Xcode projects,
and localization bundles), binary files, and symlinks are skipped. System locations `/System`, `/Library`, `/usr`, `/bin`, `/sbin`, `/private`,
`/dev`, `/cores`, `/opt`, `/Network`, and `/Volumes`, plus every user's
`/Users/*/Library` and the current user's `~/Library`, are excluded by resolved
absolute path while walking a broader root such as Whole Mac or home. An
explicitly selected root inside an excluded location is accepted and indexed,
including iCloud Drive under `~/Library/Mobile Documents`, scratch folders under
`/private/tmp`, and external volumes. Unreadable
folders (including permission/TCC denials) count as skipped and do not abort
indexing. `files_skipped` counts excluded
directory entries once, without traversing/counting their descendants, plus
skipped files. Chunks target 1200 characters with 180 characters of overlap.

Folder paths must identify existing absolute directories without `..`;
hidden/secret-looking directories and package bundles remain rejected as roots. `/` is accepted as
Whole Mac. The desktop Add Folder menu offers Desktop, Documents, Downloads,
Whole Mac, and a multi-select Choose Folder picker. Whole Mac skips system files
and app internals; its first indexing can take a long time. The home directory itself
and folders outside home are supported. Paths resolve before registration; adding
the same resolved path or a folder inside an existing root returns the covering
folder (HTTP 200), and the desktop refreshes silently. Shortcuts already covered
by an existing root show a checkmark and are disabled. Adding a containing folder
creates a new root (HTTP 201) and absorbs nested registrations in one transaction,
preserving their indexed files, chunks, and vectors without re-embedding unchanged
files. Active indexing of an absorbed root is cancelled and queued under the new
root. Removing that root removes all its files, including the absorbed files.
Indexing commits each file
atomically, reuses unchanged content, removes deleted files, and resumes on
server restart. While the engine is running, a background pass every minute
queues folders last indexed more than 30 minutes ago, when indexing is idle.
Opening Local Search also silently requests a rescan of each non-root folder
last indexed more than 2 minutes ago when none is indexing or queued. The Whole
Mac root (`/`) is excluded from this on-appear refresh to avoid walking the whole
disk on every visit. Unchanged
files keep their existing rows and embeddings. Reindex remains available for an
immediate rescan; there is no filesystem watcher in this pilot. Only one folder
indexes at a time. Removing an active
folder cancels its indexing; server shutdown also cancels work between batches.
Changing the stored model id or chunking version invalidates affected vectors
while preserving keyword text until rebuilding finishes. Schema 2 and chunking 3
rebuild earlier text-only indexes, including previously skipped media. Schema 3
upgrades schemas 1 and 2 in place: derive file signatures from stored chunk text in a
transaction. Schema 2 preserves chunk IDs, vectors, locators, and file statuses
without source reads or re-embedding. Schema 1 retains the existing media/chunking
upgrade, preserving keyword text and locators while queuing a vector rebuild. A newer or unrecognized schema is
refused before any SQL schema, data, or WAL change (the owner-only permission
and ACL hardening of the index files, which never changes their contents, still
runs first); routes return HTTP 503 with the structured
`unsupported_index_schema` error and instructions to update Rapid-MLX or choose
a new index directory. Discovery and indexing proceed in bounded batches, with
documents preferred within each directory. Each file becomes searchable when
committed while the walk continues; there is no full-root pre-walk. During discovery, `files_total`
is 0 (unknown), `files_done` counts processed files, and the desktop displays
indeterminate progress with that count. Deleted files are removed after the walk
completes; cancellation does not purge undiscovered files.

All routes use the same API-key and rate-limit dependencies as embeddings.
Errors have the standard `{"error": {"message", "type", "code", "param"}}`
envelope. All field names are snake_case.

| Method | Route | Request / response |
| --- | --- | --- |
| GET | `/v1/local-search/status` | Enabled/model availability, folders, indexing progress, totals |
| POST | `/v1/local-search/folders` | `{"path": "/absolute/folder"}` → Folder (201 new, 200 existing); queues indexing |
| DELETE | `/v1/local-search/folders/{id}` | Drops that folder's files/chunks; 204 |
| POST | `/v1/local-search/reindex` | `{"folder_id": null}` for all folders, or a folder id; status body, 202 |
| POST | `/v1/local-search/search` | `{"query": "search text", "limit": 20, "folder_ids": null, "kinds": null}` → results, took_ms, mode |

Status is shaped as:

```json
{
  "enabled": true,
  "model": {"id": "mlx-community/embeddinggemma-2-4bit", "downloaded": false, "loaded": false, "vision": false},
  "folders": [],
  "indexing": {"active": false, "folder_id": null, "current_path": null, "files_done": 0, "files_total": 0},
  "totals": {"name_only_files": 0, "files": 0, "chunks": 0, "index_bytes": 0, "files_by_kind": {"text": 0, "image": 0, "video": 0}}
}
```

Folder fields: `id`, `path`, `state`, `files_indexed`, `files_skipped`, `chunks`,
`last_indexed_at` (Unix seconds or null), `error` (string or null), and additive
optional `skip_reasons` (reason-to-file-count object, empty when no files were
skipped). Reasons use this fixed English vocabulary:

| Reason | Meaning |
| --- | --- |
| Unsupported file type | Unsupported format, binary text, unavailable PDF parser or HEIC support, or XML entities |
| Too large | Source, archive expansion, or extracted output exceeds its size cap |
| No readable text | Supported document yielded no non-whitespace text |
| Scanned PDF without text | PDF has pages but no usable text layer; OCR is needed |
| Password-protected | Encrypted PDF |
| Could not be read | Read, parse, or media-decoding failure |
| Private or system file | Hidden or secret-looking individual file |
| Audio is not supported yet | Recognized audio format |

Every content skip in the table except `Private or system file` retains a
**name-only** entry. Scanned, encrypted, unreadable, oversized and unsupported
files, including audio, can be found by file name or stem. They have no extracted
content chunk or vector and consume no semantic slots; content words cannot
match them. Extension and size rejection do not read file content. Private files
and files under pruned hidden, system or package directories never surface by
name. Explicitly selected roots inside location exclusions remain separate from
an ancestor registration if that ancestor's walk would skip them.

Name-only entries retain the existing skipped-file status, fingerprint and reason.
They still count in `files_skipped` and `skip_reasons`, and remain excluded from
`files_indexed`, `totals.files` and `totals.files_by_kind`. The additive integer
`totals.name_only_files` counts searchable name-only entries. Each folder admits
at most 200,000 (`MAX_NAME_ONLY_FILES_PER_FOLDER`), in stored file-ID order;
additional skipped files contribute only to skip counts. Existing skipped rows
become searchable without deleting the index or reading unchanged content.
Rename, modification, deletion and folder removal/absorption update these entries
on reindex. A file that becomes readable transitions to normal content indexing.

For example, `skip_reasons: {"Scanned PDF without text": 3, "Unsupported file type": 2}`.
Each file's reason and fingerprint are persisted. Counts survive restarts and
incremental reindexing. Unchanged `Could not be read` entries are retried on
reindex because permissions can change without changing size or modification
time. Other unchanged skipped files are not read or extracted again.
Changed files are retried and deleted files disappear from the counts after a
completed walk. Files skipped purely by directory pruning (system directories,
packages, hidden directories, dependencies) are **not counted individually** in
`skip_reasons`; unreadable directories and symlinks also contribute only to the
existing aggregate `files_skipped`, so the reason counts need not sum to it.

Existing indexes gain reason counts without a full rebuild: stored legacy skip
codes are translated on opening. The next incremental run refreshes Office
files, previously skipped PDFs, and documents previously indexed without any
text chunks; previously excluded individual private files gain metadata-only
reasons during discovery. Unrelated chunks and vectors are retained.

States are
`idle`, `queued`, `indexing`, `waiting_for_model`, `missing`, and `error`.
Index size includes WAL/shared-memory files. `files_by_kind` counts indexed,
keyword-only, and pending files by kind; skipped files are excluded. `vision`
is true when the loaded engine or locally available checkpoint supports images,
even before model loading. Polling status does not construct a model.

Search `limit` is 1–100 (default 20). `folder_ids: null` searches all folders;
an empty list searches none. Unknown folder ids return 404. Optional `kinds` is
an array of `"text"`, `"image"`, and/or `"video"`; null searches all kinds and an
empty array searches none. Unsupported kinds return 400.

Optional `group_limit` is 1–50. When supplied for a scope requesting multiple
kinds (including `kinds: null`), the response also includes `groups`, an ordered
array of `{kind, results, total}`. Empty groups are omitted. Each group ranks its
own files with lexical tiers first, then semantic ranking, returns at most
`group_limit` results. A group's `total` counts matching files of that kind
within the candidate window: up to the top `max(100, limit × 5)` keyword files
across kinds, plus up to 100 admitted semantic files per kind. Each file counts
once, so `total` is at most `max(100, limit × 5) + 100` per kind (just
`max(100, limit × 5)` for keyword-only search). It may be lower than the true
count for very broad queries.
Groups with lexical evidence at their head precede other groups; otherwise
their heads' standardized semantic strengths determine section order, with a
stable kind order for ties. The flat `results` and its current standardized
ordering remain unchanged. Single-kind requests omit `groups`.

The desktop's All tab requests groups and displays up to five document rows and
eight image or video thumbnails per section, in response order. Section headers
show the admitted count; Show all opens the corresponding full kind tab.
Up/Down traverses document rows and crosses sections; Left/Right traverses a
media row. Return opens the selection, Space previews, Cmd-Return reveals in
Finder, and Cmd-C copies its path. Section headers are accessible to VoiceOver.
Older engines without `groups` retain the flat results view.

When the final normal result list is empty and semantic search is available,
Search returns up to three closest files (also bounded by `limit`), ordered by
positive semantic similarity. Folder and kind filters, per-file grouping, and
file-safety checks still apply. These weak suggestions use the existing string
array field `"matched": ["closest"]`; no new required fields are added, so older
clients can decode them. Their `score` is semantic similarity, not a relevance
probability. Normal hits retain their existing admission, ranking, and markers;
closest files are never mixed into normal hits. Keyword-only mode still returns
an empty list when nothing matches. Group previews contain only these same
closest files when requested. Desktop labels an entirely closest result list
“No exact matches · Closest files” and retains the empty state for zero results.

Results contain
`path`, `folder_id`, `title`, `snippet`, `highlights`, `score`, `matched`, `page`,
`line`, `modified_at` (Unix seconds), `size` (bytes), `kind`, and `timestamp`.
Two additive optional fields are emitted on every result: `content_indexed`
(boolean, true for normal results) and `skip_reason` (null normally; the skip
label for name-only results). Name-only results have `content_indexed: false`,
empty `snippet` and `highlights`, `matched: ["keyword"]`, and null `page`/`line`.
They compete in the same filename lexical tiers as other files. Their extension
places them in the image or video group when recognized, otherwise the text
(documents) group.
`timestamp` is seconds for video and null otherwise. Media has empty `snippet`,
empty `highlights`, null `page`/`line`, and matches on visual semantics and/or
filename keywords. Video results collapse to the best qualifying frame per file;
its timestamp survives filename keyword fusion. Scores are bounded
0–1 ranking scores; normal hits have `matched` containing `semantic`, `keyword`, or both. Hybrid
search preserves explicit lexical evidence before fusing weaker matches. A term
is distinctive when it occurs in at most `max(1, floor(0.10 × indexed files))`
files with chunks, across the whole index. Repeated chunks, keyword-only tails,
and embedding coverage do not change that file frequency. A keyword chunk must
contain every distinctive term in its text or filename. Literal compound
identifiers remain mandatory. When none are distinctive, the entire literal
query substring is required in the body; a basename containing all query terms
can also match independently. FTS5 trigram candidates retain BM25 ordering with
equal body/filename column weights. Short terms use literal substring matching;
short-only queries scan inside SQLite where possible. Keyword matching uses at
most 64 distinct terms (`MAX_KEYWORD_TERMS`), preferring distinctive terms, then
longer terms, with lexical order as a deterministic tie breaker. The original
query remains available for phrase evidence and semantic search. This bounds
SQLite expression depth for all queries within the 8,192-character API limit.
If SQLite still rejects query processing, the route returns a structured 400
`invalid_query` error rather than an unhandled 500. Frequency checks stop once
a term is known to be common, and their bounded cache is invalidated on index
changes.

Ranking distinguishes four ordered evidence classes: all query terms as whole
filename tokens, ordered full phrase in the body, basename substrings, and
partial keyword/semantic evidence. Body whitespace folds so PDF line wrapping
can match; punctuation remains literal for identifiers. Filename tokens include
the stem and extension, split on non-alphanumeric characters, digit/letter and
camelCase boundaries, and compare with Unicode casefolding. CJK query terms can
match substrings within name tokens. A single whole filename word qualifies;
`memo` in `memory_notes.md` does not. Filename matches outrank content-only hits,
including content substrings. Within the filename tier, full-stem equality comes
first, then the fraction of name tokens covered by the query, then existing
lexical tie breakers. Indexed and name-only files use the same rule, including
group previews and lexical evidence for group ordering. A full phrase outranks
semantic-only and partial keyword hits, even when a weaker result has both legs.
Within the phrase class, lexical ordering is preserved. A phrase occurring in several
documents cannot identify which copy, summary, or manuscript the user intended.
These classes express lookup behavior; their boundaries are not fitted to an
evaluation query set. Scores map the four classes to disjoint intervals in 0–1
and remain ranking scores, not probabilities.

Semantic eligibility is defined once in `rapid_mlx/local_search/eligibility.py`.
JSON, JSONL/NDJSON, CSV/TSV, logs, XML, lock files, YAML, TOML, INI, CFG/CONF,
PLIST and source maps are keyword-only: their names, keys and literal values
remain searchable, with the same line/page locators, without embedding cost.
CSV is treated as a raw data export. XLSX remains semantic because its extracted
sheet names, headings and descriptive cells can form a readable report; numeric
or encoded noise is still subject to the chunk guard. Prose/document formats
(MD, TXT, RST, TEX, HTML, PDF, DOCX, PPTX) and ordinary source code remain
semantic candidates.

A cheap, deterministic chunk guard rejects dominant ASCII encoding runs of at
least 128 characters (including wrapped base64/hex), compact JSON-shaped data
of at least 256 characters, and punctuation-free runs of at least 32 ASCII words
where at least 80% are at most three letters, at least 50% at most two, and at
least 80% are distinct. The JSON check requires at least 90% data syntax and
less than 2% whitespace. No dictionary, tokenizer or language model is used;
non-ASCII letters cannot trigger the tiny-word/encoding rules, and normal code
syntax does not trigger the tiny-word/data rules. This conservative shape check
leaves short or ambiguous content eligible and cannot identify every nonsense
passage. Guarded chunks keep their text and locators for keyword retrieval.

On first open after this policy update, a versioned streaming pass through the
stored text drops only newly ineligible vectors. It preserves chunk/file IDs,
FTS entries, fingerprints, locators and eligible vectors, without reading source
files or re-embedding. Intentionally keyword-only files are complete (`indexed`
in the internal file status); they are not repeatedly upgraded when the model
becomes available. Ordinary files indexed while the model was absent remain
eligible for the existing upgrade. Subsequent opens skip the policy pass.
SQLite can reuse the freed vector space; the database file need not shrink,
and the first migration can temporarily grow its WAL.

For each kind, admit a file's best positive-cosine chunk when it meets the index's
mean cosine between chunks from different files capped by a pooled background,
or exceeds that query's mean
file score by the normal-tail cutoff for a Bonferroni-corrected 5% false-alarm
budget, keeping at most 100 files per kind.
For fewer than four files of a kind, or zero query-score variance, return up to
three best positive-cosine semantic neighbours instead of applying an
unreliable outlier estimate. Keyword admission remains independent.

The background is computed from normalized vector sums, without a quadratic
pair matrix, and cached for the current folder/kind scope. Within-file pairs are
excluded so overlapping chunks and repeated frames do not supply their own
background evidence. The additional pool takes one unnormalized mean of normalized
chunk vectors per eligible file, across all kinds in the selected folders, and
averages cross-file centroid dot products. This equals the expected cosine when
sampling two different files uniformly, then one chunk from each. Each kind's
background is capped by that shared-space estimate, so a coherent kind cannot
set a stricter ordinary-similarity gate than the folder's overall background.
The pool includes every kind even for a single-kind request, keeping tab and
All-tab admission consistent. It is computed with sums, without pair matrices;
one-file scopes retain the existing positive-cosine behavior. The outlier comparison uses one best score per file and
the population standard deviation; 5% is a conventional statistical budget,
not a fitted model/corpus cosine. Its normal-tail assumption is a heuristic:
correlated, non-normal embedding scores mean it does not guarantee a measured
5% false-positive rate or establish relevance. No independent background exists
in a one-file scope, so only the positive-cosine check applies there; a uniform
multi-file score distribution has no outlier rescue. Boundary comparisons allow
`1e-6` relative error for float32 accumulation, rather than a relevance margin.
The 100-file cap follows the API's maximum result window; replayed datasets
become development evidence, not fresh held-out validation. No fixed text or
media cosine floors remain.

Semantic-only hits compare each kind's per-query standardized cosine (using the mean and population standard deviation of every file's best chunk), interleaving kinds whose normal-CDF strengths differ by at most the existing 5% statistical budget while preserving cosine order within each kind; singleton or uniform backgrounds receive zero standardized strength.
Lexical hits retain their positions, and these normal-tail strengths are ordering heuristics rather than relevance probabilities.
Weaker evidence uses reciprocal rank fusion with k=60 and the existing rank-one
5% and rank-two/three 2% bonuses. Text uses its strongest lexical chunk for a
keyword hit; video keeps its best qualifying semantic frame and timestamp.
Exact byte-content copies of text/document files with a content hash are adjacent
with the strongest group score, while all paths remain discoverable. Identical
images/videos are listed individually. Similar but nonidentical content remains separate
to preserve potentially meaningful differences. Highlights cover distinctive
terms, or the literal query when none are distinctive.

Search caches scoped metadata rather than chunk bodies and reads bodies only
for keyword candidates and returned snippets. Replacement, removal, folder
absorption, and metadata updates invalidate that cache. The frequency cache
retains at most 1,024 terms to bound memory, independently of corpus size; eviction
only repeats a computation and never changes a result.

Whole-file query-term rarity evidence reads one compressed vocabulary per
candidate text file using its primary key; it never scans corpus-wide FTS
postings or candidate chunk bodies. Each vocabulary contains distinct casefolded
whitespace-delimited tokens, including punctuation, identifiers and CJK runs.
Substring matching inside those tokens retains morphological and CJK evidence;
this support signal is separate from phrase retrieval. Semantic-only hits within
each kind use cosine similarity plus 0.1 times rarity-weighted term coverage;
literal keyword priority and admission gates are preserved. Kind previews use
the same evidence. The signature is written
in the same transaction as the file's chunks and deleted with the file.
`MAX_FILE_SIGNATURE_BYTES` bounds the uncompressed UTF-8 vocabulary, counting
one separator byte per distinct token, to 256 KiB per file; compressed storage
adds at most 1 KiB of format overhead. A file exceeding this bound stores a NULL
signature and supplies zero term support, rather than evidence from an arbitrary
prefix. Reads enforce the same cap; malformed signatures also supply zero term
support. Its keyword/semantic retrieval and vectors remain available. Query work
is bounded by candidate file count × vocabulary cap × bounded query term count,
independent of corpus postings and document chunk count. Schema 3 adds these
signatures without rebuilding embeddings.

CPU replay timings exclude query embedding and model loading;
real embedding latency and responsiveness under chat load still need hardware
measurement.

Highlights are `[start, end]` pairs, end-exclusive Unicode character offsets
within the plain snippet, never HTML or Markdown markup. Native clients must
convert these character offsets to their own string index representation,
rather than treating them as UTF-16 byte/code-unit offsets.

Media embedding batches contain at most four images/frames. Text batches contain
at most four chunks, sorted by length within each file. Both yield between
batches. Before every indexing embedding batch, Local Search checks the engine's
read-only generation activity (including admitted, running, and queued requests).
While chat is pending, background embeddings wait with cancellable exponential
polling from 10 ms up to 100 ms; no model-worker work is submitted by that wait.
Status includes the optional `indexing.paused_reason: "chat"` while waiting and
omits it after resuming or cancellation. Sustained chat can pause indexing
indefinitely, visibly in status. Indexing detects idle within 100 ms, plus event
loop scheduling and any interactive query already using the embedding worker.
Search-query embeddings remain interactive and bypass this wait; waiting for
chat does not hold their admission lock. Generation is checked again after
acquiring that lock, before submitting a background batch.

An embedding batch already submitted cannot be preempted. A chat token arriving
then can still incur the remaining wall time of one batch of up to four images
(or four text chunks), including first-use embedding model loading and result
materialisation. There is no fixed millisecond bound; measure the slowest
four-image batch on the target hardware. FIFO admission before the model-worker
boundary limits a waiting query's
indexing-induced model wait to the remaining duration of one batch of at most
four inputs. This is a bound on batch count, not a fixed number of milliseconds;
model loading, inference, and other engine clients have their own latency.
For hardware qualification, repeat the same 120-token streaming request on the
M3 Ultra with the qwen3.5-4b chat model, both idle and during a fresh 950-file
index. Record total completion time, first-token latency, p95/max inter-token
gaps, pause status, time from last generation ending to the next embedding
submission, and four-image batch wall time (cold and warm). Also check sustained
chat, queued requests, search during a pause, and cancellation during a pause.
Whole-file SQLite/FTS writes run on a separate WAL writer, so searches continue
on the reader connection during an uncommitted indexing transaction. After commit,
the reader publishes at most one file's vector changes (64 text chunks or 120
video frames); keyword-only tails are not visited during that publication.
Cross-file length sorting is deferred to
preserve the current per-file commit and cancellation/resume behavior.
MLX construction/loading, inference, result materialisation, and teardown use
`BatchedEngine.execute_on_model_worker`, sharing generation’s owning worker.
Engines without that boundary fail closed to keyword search and expose
`semantic_unavailable_reason` in status. If the selected model worker changes,
semantic work fails closed until Local Search is restarted; cached model objects
are never forwarded to a different worker. Cancellation drains an in-flight model
call before releasing its engine on that same worker. The
`/v1/embeddings` and `/v1/rerank` routes use their own models on a separate
retrieval worker thread that loads and runs them.
File reads, extraction, chunking, and eligibility run off the HTTP event loop;
a separate single CPU worker owns SQLite and search computation. Every root
component opens relative to a pinned directory descriptor with `O_NOFOLLOW`;
the root descriptor stays open throughout discovery and extraction. Search
checks source paths through that same component-wise no-follow discipline;
name-only hits use metadata checks without opening file content.
HTML parses in-process on the CPU worker with the 20 MiB input/output cap.
PDF and Office parsing use killable subprocesses with a 512 MiB memory budget.
Their wall deadline is `min(120, 10 + 4 × input size in MiB)` seconds; the same
budget sets the child CPU limit, rounded up to whole seconds. Timeout/failure
becomes “Could not be read”.
Parser output is read incrementally and capped at 160 MiB of JSON (including
escaping); an over-budget child is killed and reaped. Cancellation kills and
reaps the child. Allocation limits apply on Linux; on
macOS, resident memory is sampled every 20 ms and an over-budget child is killed
(the peak can exceed the budget between samples). PDF and Office parsers do not
run in threads. CPU-only tests use a deterministic fake embedder; real model
retrieval and chat latency under load
remain the pilot's hardware qualification work.

The native loader and `EmbeddingEngine` use `vision=False` by default. Only the
separate local-search engine enables the vision tower/projector. The text-only
engine constructs no vision tower and filters all media weights; the existing
text embedding forward, token guards, `/v1/embeddings` schema, and media rejection
are unchanged. Vision-enabled engines expose `embed_images` for PIL images or
local image paths, using the checkpoint's image processor settings. They still
reject media tokens through their text embedding methods. No audio tower is
constructed in either mode.

CPU regression coverage uses fake embeddings, generated Pillow images, and a
small OpenCV-written video. Real checkpoint strict loading, prompted text-to-image
retrieval, mixed-aspect batching, and chat responsiveness during four-frame
batches remain reviewer hardware qualification. The separate resident vision
engine increases memory use compared with the prior text-only local-search
engine. Codec support and HEIC availability depend on the installed environment.


Memory and work budgets: 262,144 × 768 × 4 bytes = 768 MiB of retained float32
vectors with 768 dimensions (in general, 4 × chunk count × dimensions bytes). Float16 SQLite storage is 384 MiB of vector
payload. One float32 arena reserves capacity up to the budget, touching rows
as embeddings arrive; chunk-vector lookups are views into that same arena.
Freed slots are reused without growing or copying the arena. Exact scoring uses
one matrix-vector operation and NumPy reductions to select each file's best
chunk, retaining only scalar score/index arrays as scratch. File membership and folder-level moments update after committed file changes;
per-file sums use temporary scratch rather than retaining a second corpus of
centroids, so background
statistics require neither vector-table reads nor per-file/chunk query loops.
Scope/index arrays and the lexical allowed-ID table are reused for a warm view.
Lexical scope uses a correlated membership check after the FTS match; combining
FTS MATCH with a rowid IN list can make SQLite repeat the match for every corpus
chunk. The vector budget is enforced again when
opening an older index and at every replacement transaction. Status totals add
`semantic_index_full` and `semantic_chunk_budget`; keyword indexing continues
when full. Lexical queries apply scope and a 4,096-candidate SQL limit per body
or filename leg before returning bodies to Python (at most 8,192 bodies, about
9.4 MiB of ASCII chunk text). Metadata, Python object overhead, SQLite caches,
source buffers, the current file’s chunks, and model weights are additional;
this is a vector/work budget, not a total-process RSS limit. Keyword-only corpus
metadata and disk usage still grow with selected content. Local Search performs
CPU search work off-loop so indexing and searching do not occupy the HTTP thread.

Local availability requires complete weights plus `tokenizer_config.json` and
`tokenizer.json`. A checkpoint with a vision configuration additionally requires
`processor_config.json`; without it the snapshot is reported as not downloaded,
so indexing remains keyword-only until the required assets are present. Optional
repository files do not affect availability.
