"""Paired retrieval eval; never mutates the demo. Run --help for Hub options."""

import argparse
import hashlib
import inspect
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import literature
import retrieval
from literature import normalize

# Hand-written paper+section targets, checked against the read-only grokking snapshot.
CASES = [
    ("What does Power report about learning time curves?", "2201.02177", "3.1.1"),
    ("Which ablations and tricks does Power test?", "2201.02177", "3.3"),
    ("How does Power visualize learned embeddings?", "2201.02177", "3.4"),
    ("What binary operations does Power use?", "2201.02177", "A.1.1"),
    ("What generalization measures does Power compare?", "2201.02177", "A.5"),
    ("What is Liu's LU mechanism for grokking?", "2210.01117", "2"),
    ("How does Liu study a teacher student setup?", "2210.01117", "3"),
    ("What does Liu find for algorithmic datasets?", "2210.01117", "5.1"),
    ("How does Liu explain grokking on MNIST?", "2210.01117", "5.2"),
    ("How does Liu relate generalization time to weight decay?", "2210.01117", "C"),
    ("How does Nanda's Fourier multiplication algorithm work?", "2301.05217", "3.1"),
    ("What progress measures does Nanda define?", "2301.05217", "5.1"),
    (
        "What are Nanda's memorization, circuit formation, and cleanup phases?",
        "2301.05217",
        "5.2",
    ),
    (
        "Which interventions could make generalization happen sooner rather than only memorizing the examples?",
        "2301.05217",
        "5.3",
    ),
    (
        "What experiments might separate a useful algorithm from incidental correlations in a trained network?",
        "2301.05217",
        "4.4",
    ),
]


# Authored from paper text/headings, before inspecting any new retrieval output.
# The original 15 are all dev (their scores were already exposed in round 6).
# Five new dev questions and ten fresh held-out questions; IDs/split never shuffled.
ADDED_CASES = [
    (
        "What AdamW beta2, warmup updates and minibatch size does Power use?",
        "2201.02177",
        "A.1.2",
        "lexical",
    ),
    (
        "In Liu Figure 2, what do alpha = 0.5 and alpha = 2.0 change?",
        "2210.01117",
        "3",
        "lexical",
    ),
    (
        "How does Nanda define restricted loss and excluded loss?",
        "2301.05217",
        "5.1",
        "lexical",
    ),
    (
        "Why can smaller training sets take much longer to learn a rule even when eventual accuracy is unchanged?",
        "2201.02177",
        "3.1.1",
        "conceptual",
    ),
    (
        "How can changing the size of initial weights make a handwritten-digit classifier learn its rule late?",
        "2210.01117",
        "4",
        "conceptual",
    ),
    (
        "What are the 2 layers, width 128 and 4 attention heads used by Power?",
        "2201.02177",
        "A.1.2",
        "lexical",
    ),
    (
        "In Power learning time curves, what happens near 25-30% training data when it decreases by 1%?",
        "2201.02177",
        "3.1.1",
        "lexical",
    ),
    (
        "What does Liu Figure 8 show about MNIST representation messiness?",
        "2210.01117",
        "5.2",
        "lexical",
    ),
    (
        "What does Liu Figure 10 test about time to generalize versus weight decay?",
        "2210.01117",
        "C",
        "lexical",
    ),
    (
        "Why does Nanda use P = 113, 30% training inputs and 512 hidden units?",
        "2301.05217",
        "3",
        "lexical",
    ),
    (
        "What are Nanda's key frequencies 14, 35, 41, 42, 52 in Figure 3?",
        "2301.05217",
        "4.1",
        "lexical",
    ),
    (
        "Does Nanda find the slingshot mechanism necessary for grokking?",
        "2301.05217",
        "D.2",
        "lexical",
    ),
    (
        "How can destroying selected parts of a network establish that its proposed internal computation actually causes its predictions?",
        "2301.05217",
        "4.4",
        "conceptual",
    ),
    (
        "Why might familiar image features require less reorganization during learning than representations for arithmetic?",
        "2210.01117",
        "5.2",
        "conceptual",
    ),
    (
        "How can measuring a hidden algorithm reveal learning before the test score improves?",
        "2301.05217",
        "5.1",
        "conceptual",
    ),
]
QUESTIONS = [
    dict(
        id=f"q{i + 1:02d}",
        query=q,
        paper=p,
        section=s,
        subset="conceptual",
        split="dev",
    )
    for i, (q, p, s) in enumerate(CASES)
]
QUESTIONS += [
    dict(
        id=f"q{i + 16:02d}",
        query=q,
        paper=p,
        section=s,
        subset=subset,
        split="dev" if i < 5 else "held-out",
    )
    for i, (q, p, s, subset) in enumerate(ADDED_CASES)
]


class DiskCache:
    """Content-addressed per-item cache; batch shape and ablation order don't matter."""

    def __init__(self, folder, namespace, offline=False):
        self.folder = Path(folder) / "cache" / namespace
        self.folder.mkdir(parents=True, exist_ok=True)
        self.offline = offline

    def call(self, kind, key, function):
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
        path = self.folder / f"{kind}-{digest}.json"
        if path.exists():
            return json.loads(path.read_text())
        if self.offline:
            raise RuntimeError(f"Offline cache miss: {kind} {digest}")
        value = function()
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value))
        temporary.replace(path)
        return value

    def embedder(self, embed):
        def cached(texts):
            # Small bounded batches, with every item persisted immediately.
            paths = [
                self.folder
                / (
                    "embed-"
                    + hashlib.sha256(json.dumps(t, sort_keys=True).encode()).hexdigest()
                    + ".json"
                )
                for t in texts
            ]
            missing = list(
                dict.fromkeys(
                    t for t, p in zip(texts, paths, strict=True) if not p.exists()
                )
            )
            if missing and self.offline:
                raise RuntimeError("Offline cache miss: embedding")
            for start in range(0, len(missing), 16):
                batch = missing[start : start + 16]
                values = embed(batch)
                normalize(values, len(batch))
                for text, value in zip(batch, values, strict=True):
                    self.call("embed", text, lambda value=value: value)
            return [json.loads(p.read_text()) for p in paths]

        return cached


def evaluate(
    data,
    embed,
    chat=None,
    rerank=None,
    cases=None,
    lexical_weight=0.05,
    rewrite_weight=0.2,
    round6=None,
):
    cases = QUESTIONS if cases is None else cases
    cases = [
        dict(
            id=f"q{i + 1:02d}",
            query=c[0],
            paper=c[1],
            section=c[2],
            split="dev",
            subset="conceptual",
        )
        if not isinstance(c, dict)
        else c
        for i, c in enumerate(cases)
    ]
    modes = {
        "dense": dict(bm25=False),
        "dense+bm25": dict(bm25=True),
        "dense+bm25+multiquery": dict(bm25=True, multiquery=True),
        "full": dict(bm25=True, multiquery=True, mmr=True),
    }
    result = {name: [] for name in ["legacy-r5", *modes, "default"]}
    if round6:
        result["round6"] = []
    if rerank:
        result.update({name + "+rerank": [] for name in modes})
    for case in cases:
        query = case["query"]
        rankings = {"legacy-r5": literature.baseline_rank(data, query, embed)}
        rankings["default"] = rankings["legacy-r5"]
        if round6:
            rankings["round6"] = round6.retrieve(data, query, embed, chat, rerank)
        for name, options in modes.items():
            rankings[name] = retrieval.retrieve(
                data,
                query,
                embed,
                chat,
                lexical_weight=lexical_weight,
                rewrite_weight=rewrite_weight,
                **options,
            )
            if rerank:
                rankings[name + "+rerank"] = retrieval.retrieve(
                    data,
                    query,
                    embed,
                    chat,
                    rerank,
                    lexical_weight=lexical_weight,
                    rewrite_weight=rewrite_weight,
                    **options,
                )
        for name, indices in rankings.items():
            rank = next(
                (
                    r
                    for r, i in enumerate(indices[:6], 1)
                    if case["paper"] in data["chunks"][i]["file"]
                    and data["chunks"][i].get("section_number") == case["section"]
                ),
                None,
            )
            result[name].append(
                dict(
                    case,
                    rank=rank,
                    retrieved=[
                        dict(
                            index=i,
                            file=data["chunks"][i]["file"],
                            section=data["chunks"][i].get("section_number"),
                        )
                        for i in indices
                    ],
                )
            )
    return result


def metrics(rows):
    n = len(rows)
    return {
        **{
            f"recall@{k}": sum(r["rank"] is not None and r["rank"] <= k for r in rows)
            / n
            if n
            else 0
            for k in (1, 3, 6)
        },
        "MRR": sum(1 / r["rank"] if r["rank"] else 0 for r in rows) / n if n else 0,
    }


def report(result):
    for split in ("dev", "held-out"):
        for subset in ("all", "lexical", "conceptual"):
            selected = {
                name: [
                    r
                    for r in rows
                    if r["split"] == split
                    and (subset == "all" or r["subset"] == subset)
                ]
                for name, rows in result.items()
            }
            n = len(next(iter(selected.values())))
            if not n:
                continue
            print(f"\n{split} / {subset} (n={n})")
            print(f"{'pipeline':32}   R@1   R@3   R@6    MRR")
            for name, rows in selected.items():
                m = metrics(rows)
                print(f"{name:32} " + " ".join(f"{v:5.3f}" for v in m.values()))
        rows = [r for r in next(iter(result.values())) if r["split"] == split]
        if rows:
            print(
                "\nPer-question ranks ("
                + split
                + "): "
                + " ".join(r["id"] for r in rows)
            )
            for name, values in result.items():
                print(
                    f"{name:32} "
                    + " ".join(
                        str(r["rank"] or "-").rjust(3)
                        for r in values
                        if r["split"] == split
                    )
                )
    # A failed holdout is a failed gate, never an invitation to tune on it.
    rows = lambda name: [r for r in result[name] if r["split"] == "held-out"]
    if rows("dense"):
        baseline = metrics(rows("dense"))
        lexical_dense = metrics([r for r in rows("dense") if r["subset"] == "lexical"])
        gates = {}
        for name in ("full", "default"):
            m = metrics(rows(name))
            lex = metrics([r for r in rows(name) if r["subset"] == "lexical"])
            gates[name] = (
                m["MRR"] >= baseline["MRR"]
                and m["recall@6"] >= baseline["recall@6"]
                and lex["MRR"] > lexical_dense["MRR"]
                and lex["recall@6"] >= lexical_dense["recall@6"]
            )
        print(
            "\nHeld-out gate (MRR/R@6 >= dense, lexical MRR > dense): "
            + json.dumps(gates)
        )
        return gates
    return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", default="http://127.0.0.1:4800")
    parser.add_argument(
        "--token-file",
        type=Path,
        help="Token stays out of argv/output; not needed offline",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(
            "/private/tmp/rapid-mlx-hub/demo-data/plugins/research-loop/data/projects/grokking"
        ),
    )
    parser.add_argument(
        "--scratch", type=Path, default=Path("/private/tmp/rapid-mlx-hub-eval-r7")
    )
    parser.add_argument("--embedding-model")
    parser.add_argument("--chat-model")
    parser.add_argument("--split", choices=("dev", "held-out", "both"), default="both")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="No discovery/model calls; fail on missing cache entries",
    )
    parser.add_argument("--lexical-weight", type=float, default=0.05)
    parser.add_argument("--rewrite-weight", type=float, default=0.2)
    parser.add_argument(
        "--round6-source",
        type=Path,
        help="Optional preserved round-6 retrieval.py for paired diagnosis",
    )
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    # Refuse any destination in the live project, including symlink aliases.
    if args.scratch.resolve().is_relative_to(
        args.project.resolve()
    ) or args.project.resolve().is_relative_to(args.scratch.resolve()):
        parser.error("scratch and source project must be separate directories")
    source = (args.project / "literature_index/index.json").read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    project = args.scratch / digest[:16]
    project.mkdir(exist_ok=True)
    manifest = project / "models.json"
    if args.offline:
        info = json.loads(manifest.read_text())
        embed = chat = ranker = None
    else:
        if not args.token_file:
            parser.error("--token-file is required online")
        os.environ.update(
            RAPID_HUB_URL=args.hub.rstrip("/"),
            RAPID_HUB_TOKEN=args.token_file.read_text().strip(),
            RAPID_PROJECT="retrieval-eval",
        )
        import hub_client

        models, _, embed_model = hub_client.discover()
        embed = literature.HTTPEmbedder(
            args.hub.rstrip("/") + "/hub/v1", args.embedding_model or embed_model
        )
        chat_model = args.chat_model or next(iter(models), None)
        chat = (
            literature.HTTPQueryChat(args.hub.rstrip("/") + "/hub/v1", chat_model)
            if chat_model
            else None
        )
        ranker = literature.configured_reranker()
        info = dict(
            identity=embed.identity,
            embedding_model=embed.model,
            chat_model=chat_model,
            rerank_model=ranker.model if ranker else None,
            hub=args.hub,
        )
        manifest.write_text(json.dumps(info))
    namespace = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()[
        :16
    ]
    cache = DiskCache(project, namespace, args.offline)
    cached_embed = cache.embedder(embed)
    # Include the prompt implementation in keys so prompt changes invalidate rewrites.
    prompt_version = hashlib.sha256(
        inspect.getsource(literature.HTTPQueryChat).encode()
    ).hexdigest()
    cached_chat = (
        (lambda q: cache.call("rewrite", [prompt_version, q], lambda: chat(q)))
        if info["chat_model"]
        else None
    )
    cached_ranker = (
        (lambda q, texts: cache.call("rerank", [q, texts], lambda: ranker(q, texts)))
        if info["rerank_model"]
        else None
    )
    index = project / "literature_index/index.json"
    if not index.exists():
        index.parent.mkdir(exist_ok=True)
        data = json.loads(source)
        if literature.HTTPEmbedder.snapshot_identity(None, data) != info["identity"]:
            print(
                "Certifying copied chunk vectors with resolved embedding model.",
                flush=True,
            )
            data["vectors"] = normalize(
                cached_embed(
                    [
                        f"title: {Path(c['file']).stem} | text: {c['text']}"
                        for c in data["chunks"]
                    ]
                ),
                len(data["chunks"]),
            )
            data.update(identity=info["identity"], identity_resolved=True)
        index.write_text(json.dumps(data))
        shutil.copy2(args.project / "project.json", project / "project.json")
    data = literature.snapshot(project)
    if data["identity"] != info["identity"]:
        raise ValueError("Scratch model changed; use a different --scratch directory")
    data["bm25"] = retrieval.bm25_index(data["chunks"], data["files"])
    cases = [c for c in QUESTIONS if args.split == "both" or c["split"] == args.split]
    round6 = None
    if args.round6_source:
        import importlib.util

        spec = importlib.util.spec_from_file_location("round6", args.round6_source)
        round6 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(round6)
    print(
        "Cache:",
        project,
        "| rerank:",
        info["rerank_model"] or "unavailable (skipped)",
        flush=True,
    )
    result = evaluate(
        data,
        cached_embed,
        cached_chat,
        cached_ranker,
        cases,
        args.lexical_weight,
        args.rewrite_weight,
        round6,
    )
    gates = report(result)
    output = project / f"results-{args.split}.json"
    output.write_text(
        json.dumps(
            dict(
                models=info,
                source_sha256=digest,
                questions=QUESTIONS,
                settings=dict(
                    lexical_weight=args.lexical_weight,
                    rewrite_weight=args.rewrite_weight,
                ),
                results=result,
                gates=gates,
            ),
            indent=2,
        )
    )
    print("\nDetailed results:", output)


if __name__ == "__main__":
    main()
