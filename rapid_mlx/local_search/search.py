# SPDX-License-Identifier: Apache-2.0
"""Reciprocal rank fusion and plain snippets with Unicode character offsets."""

import math
import re
from pathlib import Path
from statistics import NormalDist

KINDS = ("text", "image", "video")
# A conventional family-wise false-alarm budget, not a fitted cosine cutoff.
# The normal-tail approximation is a heuristic, not a relevance probability.
OUTLIER_FAMILY_ALPHA = 0.05
# Match the API's maximum result window; internal work stays bounded.
SEMANTIC_FILE_CAP = 100
SEMANTIC_MIN_FILES = 4
SEMANTIC_FALLBACK_FILES = 3
# Give a short head from every admitted modality a chance before long tails.
SEMANTIC_HEAD_FILES = 3
# Compare standardized outliers in normal-tail mass, using the admission budget.
# This is an ordering heuristic, not a calibrated relevance probability.
SEMANTIC_NEAR_TIE = OUTLIER_FAMILY_ALPHA


def query_statistics(
    scores: list[tuple[int, float]], rows: dict
) -> tuple[list[int], float, float]:
    """Use every file's best chunk, including files rejected by admission."""
    ids = top_files([i for i, _ in scores], rows, len(scores)) if scores else []
    values = dict(scores)
    samples = [values[i] for i in ids]
    mean = math.fsum(samples) / len(samples) if samples else 0.0
    variance = (
        math.fsum((s - mean) ** 2 for s in samples) / len(samples) if samples else 0.0
    )
    return ids, mean, math.sqrt(variance)


def semantic_ranking(
    scores: list[tuple[int, float]],
    rows: dict,
    background: float,
    *,
    statistics: tuple[list[int], float, float] | None = None,
) -> list[int]:
    """Admit above ordinary index similarity or a query-specific rare outlier.

    Estimate the query null from best-per-file scores so repeated chunks/frames
    do not create extra observations. Bonferroni corrects for the files tested.
    With fewer than four files or zero variance, retain at most three best
    positive-cosine neighbours because the outlier estimate is degenerate.
    """
    ids, mean, deviation = (
        statistics if statistics is not None else query_statistics(scores, rows)
    )
    values = dict(scores)
    samples = [values[i] for i in ids]
    if len(samples) < SEMANTIC_MIN_FILES or deviation <= 1e-12:
        return [i for i in ids if values[i] > 0][:SEMANTIC_FALLBACK_FILES]
    outlier = math.inf
    if len(samples) > 1:
        if deviation > 0 and max(samples) > min(samples):
            z = NormalDist().inv_cdf(1 - OUTLIER_FAMILY_ALPHA / len(samples))
            outlier = mean + z * deviation
    threshold = max(0.0, min(background, outlier))
    # The query dot products are float32; the cached background sums are float64.
    # Allow float32 accumulation error at the boundary, not a relevance margin.
    return [
        i
        for i in ids
        if values[i] > 0
        and (values[i] >= threshold or math.isclose(values[i], threshold, rel_tol=1e-6))
    ][:SEMANTIC_FILE_CAP]


def query_terms(query: str) -> list[str]:
    # Preserve literal identifiers/paths; trailing sentence punctuation is excluded.
    return list(dict.fromkeys(re.findall(r"\w+(?:[./-]\w+)*", query, re.UNICODE)))


def name_tokens(value: str) -> list[str]:
    """Split before casefolding so Unicode case and camel boundaries survive."""
    tokens = []
    for word in re.findall(r"[^\W_]+", value):
        start = 0
        for i in range(1, len(word)):
            previous, current = word[i - 1], word[i]
            if (
                previous.isdigit() != current.isdigit()
                or (previous.islower() and current.isupper())
                or (
                    previous.isupper()
                    and current.isupper()
                    and i + 1 < len(word)
                    and word[i + 1].islower()
                )
            ):
                tokens.append(word[start:i].casefold())
                start = i
        tokens.append(word[start:].casefold())
    return tokens


def filename_match(query: str, terms: list[str], filename: str) -> tuple[bool, float]:
    """Full-stem equality and name-token coverage; zero coverage means no match."""
    tokens = name_tokens(filename)
    wanted = {token for term in terms for token in name_tokens(term)}

    def matches(term, token):
        return term == token or (
            re.search(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", term) is not None
            and term in token
        )

    if not wanted or not all(
        any(matches(term, token) for token in tokens) for term in wanted
    ):
        return False, 0.0
    covered = sum(any(matches(term, token) for term in wanted) for token in tokens)
    return name_tokens(query) == name_tokens(Path(filename).stem), covered / len(tokens)


def lexical_tier(query: str, terms: list[str], text: str, filename: str) -> int:
    """Evidence classes, not calibrated relevance weights.

    Whole filename tokens > ordered phrase > basename substrings > partial.
    Whitespace folds in bodies; punctuation stays literal for identifiers.
    Filename word matching includes stem and extension tokens.
    """
    phrase = " ".join(query.casefold().split())
    if filename_match(query, terms, filename)[1]:
        return 3
    if phrase and phrase in " ".join(text.casefold().split()):
        return 2
    if terms and all(term.casefold() in filename.casefold() for term in terms):
        return 1
    return 0


def fuse(
    semantic: list[int],
    keyword: list[int],
    k: int = 60,
    *,
    rows: dict | None = None,
    semantic_legs: list[list[int]] | None = None,
    semantic_scores: dict[int, float] | None = None,
    semantic_standardized: dict[int, float] | None = None,
    keyword_evidence: dict[int, int] | None = None,
    limit: int | None = None,
) -> list[tuple[int, float, list[str]]]:
    legs = semantic_legs if semantic_legs is not None else [semantic]
    head_paths = (
        {rows[i]["path"] for leg in legs for i in leg[:SEMANTIC_HEAD_FILES]}
        if rows
        else set()
    )
    if semantic_scores is not None:
        legs = [
            sorted(
                {i for leg in legs for i in leg},
                key=lambda i: (-(semantic_scores or {})[i], i),
            )
        ]
    evidence = dict(keyword_evidence or {})
    if rows is not None:
        path_evidence: dict[str, int] = {}
        for i in keyword:
            path = rows[i]["path"]
            path_evidence[path] = max(path_evidence.get(path, 0), evidence.get(i, 0))
        representatives: dict[str, int] = {}
        for leg in legs:
            for i in leg:
                representatives.setdefault(rows[i]["path"], i)
        keyword_representatives: dict[str, int] = {}
        for i in keyword:
            # The first keyword chunk is the strongest, not the last chunk.
            keyword_representatives.setdefault(rows[i]["path"], i)
        for path, i in keyword_representatives.items():
            if rows[i].get("kind", "text") == "text":
                representatives[path] = i
            else:
                representatives.setdefault(path, i)
        legs = [[representatives[rows[i]["path"]] for i in leg] for leg in legs]
        keyword = [representatives[rows[i]["path"]] for i in keyword]
        evidence = {
            i: path_evidence.get(rows[i]["path"], 0) for i in representatives.values()
        }
    # Collapse every leg before scoring; repeated chunks cannot cast extra votes.
    legs = [list(dict.fromkeys(leg)) for leg in legs]
    keyword = list(dict.fromkeys(keyword))
    scores: dict[int, float] = {}
    matched: dict[int, list[str]] = {}
    for name, ranking in [("semantic", leg) for leg in legs] + [("keyword", keyword)]:
        for rank, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
            if rank <= 3:
                scores[chunk_id] += (0.05 if rank == 1 else 0.02) / (k + 1)
            if name not in matched.setdefault(chunk_id, []):
                matched[chunk_id].append(name)
    # A file belongs to exactly one modality: at most one semantic + keyword leg.
    maximum = 2 * 1.05 / (k + 1)
    normalized = {i: min(1.0, score / maximum) for i, score in scores.items()}
    if keyword_evidence is not None:
        # Four evidence classes occupy disjoint score bands. Two weak votes
        # cannot defeat a literal phrase; RRF orders the weaker classes.
        keyword_rank = {i: rank for rank, i in enumerate(keyword, 1)}
        normalized = {
            i: (
                evidence.get(i, 0)
                + (
                    (k + 1) / (k + keyword_rank[i])
                    if evidence.get(i, 0) >= 2
                    else score
                )
            )
            / 4
            for i, score in normalized.items()
        }
        # Within literal classes preserve lexical ordering. Semantic support
        # cannot turn a weaker phrase hit into the best exact lookup.
    ranking = sorted(scores, key=lambda i: (-normalized[i], i))
    if rows is not None and semantic_standardized is not None:
        # Replace only semantic-only slots: every lexical hit retains its position.
        slots = [n for n, i in enumerate(ranking) if matched[i] == ["semantic"]]
        queues: dict[str, list[int]] = {kind: [] for kind in KINDS}
        for n in slots:
            i = ranking[n]
            queues[rows[i]["kind"]].append(i)
        for queue in queues.values():
            queue.sort(key=lambda i: (-(semantic_scores or {})[i], i))
        used = dict.fromkeys(KINDS, 0)
        ordered = []
        strength = {i: NormalDist().cdf(z) for i, z in semantic_standardized.items()}
        while any(queues.values()):
            best = max(strength[q[0]] for q in queues.values() if q)
            eligible = [
                kind
                for kind, q in queues.items()
                if q and strength[q[0]] >= best - SEMANTIC_NEAR_TIE
            ]
            kind = min(eligible, key=lambda kind: (used[kind], KINDS.index(kind)))
            ordered.append(queues[kind].pop(0))
            used[kind] += 1
        slot_scores = [normalized[ranking[n]] for n in slots]
        for n, i, score in zip(slots, ordered, slot_scores):
            ranking[n] = i
            normalized[i] = score
    if rows is not None:
        # Exact byte-content copies remain discoverable at every path, adjacent
        # to their strongest hit. No approximate deduplication: similar reports
        # can differ in the one fact a professional is looking for.
        groups: dict[tuple[str, str], list[int]] = {}
        for i in ranking:
            row = rows[i]
            identity = (row.get("kind", "text"), row.get("content_hash") or row["path"])
            groups.setdefault(identity, []).append(i)
        ranking = [i for group in groups.values() for i in group]
        for group in groups.values():
            for i in group:
                normalized[i] = normalized[group[0]]
        if limit is not None and limit > 0 and semantic_standardized is None:
            # Preserve lexical evidence and the existing order. Reserve a short
            # semantic head per kind by replacing only the weakest other tails,
            # rather than moving media ahead of stronger text hits.
            selected = set([i for i in ranking if evidence.get(i, 0) > 0][:limit])
            for i in ranking:
                if len(selected) >= limit:
                    break
                if rows[i]["path"] in head_paths:
                    selected.add(i)
            for i in ranking:
                if len(selected) >= limit:
                    break
                selected.add(i)
            ranking = [i for i in ranking if i in selected] + [
                i for i in ranking if i not in selected
            ]
    return [(i, normalized[i], matched[i]) for i in ranking]


def top_files(ranking: list[int], rows: dict, count: int) -> list[int]:
    seen = set()
    result = []
    for chunk_id in ranking:
        path = rows[chunk_id]["path"]
        if path not in seen:
            seen.add(path)
            result.append(chunk_id)
            if len(result) >= count:
                break
    return result


def grouped_rankings(
    keyword: list[int],
    *,
    rows: dict,
    semantic_legs: list[list[int]],
    semantic_scores: dict[int, float] | None,
    semantic_standardized: dict[int, float],
    keyword_evidence: dict[int, int],
) -> list[tuple[str, list[tuple[int, float, list[str]]]]]:
    """Rank independent kind previews without changing the flat fusion policy."""
    groups = []
    for kind in KINDS:
        ranking = fuse(
            [],
            [i for i in keyword if rows[i]["kind"] == kind],
            rows=rows,
            semantic_legs=[
                [i for leg in semantic_legs for i in leg if rows[i]["kind"] == kind]
            ],
            semantic_scores=semantic_scores,
            keyword_evidence=keyword_evidence,
        )
        if ranking:
            groups.append((kind, ranking))
    # A lexical representative may differ from the best semantic chunk. Compare
    # the best-per-file statistic, rather than assigning that representative zero.
    strength: dict[str, float] = {}
    for i, z in semantic_standardized.items():
        path = rows[i]["path"]
        strength[path] = max(strength.get(path, -math.inf), z)
    path_evidence: dict[str, int] = {}
    for i in keyword:
        path = rows[i]["path"]
        path_evidence[path] = max(
            path_evidence.get(path, 0), keyword_evidence.get(i, 0)
        )
    groups.sort(
        key=lambda group: (
            -int(path_evidence.get(rows[group[1][0][0]]["path"], 0) > 0),
            -strength.get(rows[group[1][0][0]]["path"], 0.0),
            KINDS.index(group[0]),
        )
    )
    return groups


def snippet(
    text: str, terms: list[str], size: int = 500
) -> tuple[str, list[list[int]]]:
    # IGNORECASE yields positions in the original text (casefold can expand ß).
    pattern = "|".join(re.escape(term) for term in sorted(terms, key=len, reverse=True))
    matches = list(re.finditer(pattern, text, re.IGNORECASE)) if pattern else []
    readable_start = 0
    for line in text.splitlines(keepends=True):
        if any(c.isalnum() for c in line) and not re.match(
            r"^\s*(?:`{3,}|~{3,})", line
        ):
            break
        readable_start += len(line)
    # A punctuation-only chunk still produces a nonempty excerpt.
    if readable_start == len(text):
        readable_start = 0
    while readable_start < len(text) and text[readable_start].isspace():
        readable_start += 1
    # Keep original-text offsets while selecting complete line/sentence starts.
    starts = {readable_start}
    starts.update(
        m.end()
        for m in re.finditer(r"\n\s*|[.!?;](?:[\"'’”)]*)\s+|[。！？；，]\s*", text)
        if m.end() >= readable_start
    )
    if matches:
        preferred = max(0, matches[0].start() - 100)

        def quality(start):
            contained = [
                m for m in matches if start <= m.start() and m.end() <= start + size
            ]
            return (
                len({m.group().casefold() for m in contained}),
                len(contained),
                -abs(start - preferred),
                -start,
            )

        start = max(starts, key=quality)
    else:
        start = readable_start
    end = min(len(text), start + size)
    result = text[start:end]
    highlights = [
        [m.start() - start, m.end() - start]
        for m in matches
        if m.start() >= start and m.end() <= end
    ]
    return result, highlights
