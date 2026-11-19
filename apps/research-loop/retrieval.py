"""Portable project retrieval: data + injected embed/chat/rerank callables only."""

import math
import re
from collections import Counter

STOPWORDS = set(
    [
        "a",
        "an",
        "the",
        "is",
        "of",
        "to",
        "and",
        "in",
        "for",
        "what",
        "how",
        "does",
        "we",
        "with",
        "should",
        "et",
        "al",
        "which",
        "it",
        "do",
        "as",
        "by",
        "its",
        "on",
        "are",
        "was",
        "were",
        "this",
        "that",
        "from",
        "about",
    ]
)
BM25_VERSION = 2


def terms(text):
    # Keep decimal hyperparameters, section IDs and hyphenated identifiers intact.
    return [
        t
        for t in re.findall(r"\w+(?:[./-]\w+)*", text.casefold())
        if t not in STOPWORDS
    ]


def bm25_index(chunks, files):
    # Robertson & Zaragoza (2009), BM25, k1=1.2, b=.75:
    # https://doi.org/10.1561/1500000019
    documents = [
        Counter(
            terms(
                " ".join(
                    (
                        c["text"],
                        ((c.get("section_title") or "") + " ") * 2,
                        files.get(c["file"], {}).get("label") or c["file"],
                    )
                )
            )
        )
        for c in chunks
    ]
    df = Counter(t for doc in documents for t in doc)
    lengths = [sum(d.values()) for d in documents]
    return dict(
        version=BM25_VERSION,
        documents=documents,
        df=dict(df),
        lengths=lengths,
        average=sum(lengths) / max(1, len(lengths)),
    )


def lexical(query, stats, eligible):
    scores = {}
    n = len(stats["documents"])
    for i in eligible:
        doc, length = stats["documents"][i], stats["lengths"][i]
        score = 0
        for term in set(terms(query)):
            tf = doc.get(term, 0)
            df = stats["df"].get(term, 0)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            score += (
                idf
                * tf
                * 2.2
                / (tf + 1.2 * (0.25 + 0.75 * length / max(1, stats["average"])))
            )
        if score > 0:
            scores[i] = score
    return sorted(scores, key=lambda i: (-scores[i], i))


def rrf(rankings, weights=None):
    # Cormack, Clarke & Büttcher (SIGIR 2009), original RRF with k=60.
    # https://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf
    scores = Counter()
    weights = [1.0] * len(rankings) if weights is None else weights
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, i in enumerate(dict.fromkeys(ranking), 1):
            scores[i] += weight / (60 + rank)
    return scores


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def unit(v):
    if not v or any(not math.isfinite(x) for x in v):
        raise ValueError("Invalid embedding")
    norm = math.sqrt(dot(v, v))
    if not norm:
        raise ValueError("Invalid embedding norm")
    return [x / norm for x in v]


def scope(data, query):
    tokens = set(terms(query))
    named = {
        name
        for name, f in data["files"].items()
        if f.get("metadata", {}).get("first_author_surname", "").casefold() in tokens
    }
    return [
        i
        for i, c in enumerate(data["chunks"])
        if (c.get("section_title") or "").casefold() != "references"
        and (not named or c["file"] in named)
    ]


def specific(data, query):
    tokens = terms(query)
    if len(tokens) <= 7 or re.search(r"\bsection\b|§|\bpaper\b", query, re.I):
        return True
    for name, f in data["files"].items():
        metadata = f.get("metadata", {})
        names = [
            name,
            f.get("label"),
            metadata.get("title"),
            metadata.get("first_author_surname"),
        ]
        if any(
            n
            and re.search(
                r"(?<!\w)" + re.escape(n.casefold()) + r"(?!\w)", query.casefold()
            )
            for n in names
        ):
            return True
    return any(
        c.get("section_title") and c["section_title"].casefold() in query.casefold()
        for c in data["chunks"]
    )


def queries(data, query, chat):
    result = [query]
    if chat and not specific(data, query):
        try:
            rewritten = chat(query)  # One bounded call, including on malformed output.
            if isinstance(rewritten, dict):
                rewritten = rewritten.get("queries")
            if isinstance(rewritten, list):
                for q in rewritten[:3]:
                    if (
                        isinstance(q, str)
                        and 0 < len(q.strip()) <= 300
                        and q.strip() not in result
                    ):
                        result.append(q.strip())
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return result


def retrieve(
    data,
    query,
    embed,
    chat=None,
    rerank=None,
    goal="",
    limit=6,
    stats=None,
    *,
    bm25=False,
    multiquery=False,
    mmr=False,
    lexical_weight=0.05,
    rewrite_weight=0.2,
):
    chunks, vectors = data["chunks"], data["vectors"]
    eligible = scope(data, query)
    if not eligible or limit <= 0:
        return []
    search_queries = queries(data, query, chat if multiquery else None)
    embedded = embed(
        [
            f"task: search result | query: {q}\nResearch goal: {goal}"
            for q in search_queries
        ]
    )
    if len(embedded) != len(search_queries):
        raise ValueError("Embedding count mismatch")
    if bm25 and (not stats or stats.get("version") != BM25_VERSION):
        stats = bm25_index(chunks, data["files"])
    rankings, weights = [], []
    for number, (q, v) in enumerate(zip(search_queries, embedded, strict=True)):
        v = unit(v)
        if len(v) != len(vectors[0]):
            raise ValueError("Embedding dimensions changed; re-index literature.")
        dense = sorted(
            (i for i in eligible if dot(v, vectors[i]) >= 0.5),
            key=lambda i: (-dot(v, vectors[i]), i),
        )
        # Admission is on cosine only; RRF values are never cosine thresholds.
        # Rewrites share a fixed budget so additional paraphrases cannot outvote
        # the original question. Lexical evidence may admit low-cosine passages.
        weight = 1.0 if number == 0 else rewrite_weight / (len(search_queries) - 1)
        rankings.append(dense)
        weights.append(weight)
        if bm25:
            rankings.append(lexical(q, stats, eligible))
            weights.append(weight * lexical_weight)
    scores = rrf(rankings, weights)
    if not scores:
        return []
    candidates = sorted(scores, key=lambda i: (-scores[i], i))[:20]
    relevance = {i: scores[i] / max(scores.values()) for i in candidates}
    if rerank:
        try:
            values = rerank(query, [chunks[i]["text"] for i in candidates])
            if len(values) != len(candidates) or any(
                not math.isfinite(x) for x in values
            ):
                raise ValueError("Invalid reranking scores")
            low, high = min(values), max(values)
            if high > low:
                relevance = {
                    i: (s - low) / (high - low)
                    for i, s in zip(candidates, values, strict=True)
                }
        except (OSError, ValueError, KeyError, TypeError):
            pass
    # Carbonell & Goldstein (SIGIR 1998), optional MMR:
    # https://doi.org/10.1145/290941.291025
    # Experimental only: similarity is content based, never paper adjacency.
    selected = []

    def redundancy(i, j):
        return max(0, dot(vectors[i], vectors[j]))

    while candidates and len(selected) < limit:
        i = max(
            candidates,
            key=lambda i: (
                relevance[i]
                - (
                    0.1 * max((redundancy(i, j) for j in selected), default=0)
                    if mmr
                    else 0
                ),
                scores[i],
                -i,
            ),
        )
        candidates.remove(i)
        if any(
            chunks[i]["file"] == chunks[j]["file"]
            and chunks[i].get("page") == chunks[j].get("page")
            and max(chunks[i]["start"], chunks[j]["start"])
            < min(chunks[i]["end"], chunks[j]["end"])
            for j in selected
        ):
            continue
        selected.append(i)
    return selected
