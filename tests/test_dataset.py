import json
import random
import shlex
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from rapid_mlx import cli, dataset


def prose(label, words=270):
    return label + " " + "ordinary, original prose. " * ((words - 1) // 3) + "End."


def argv(source, out):
    return ["dataset", "--from", "docs", "--input", str(source), "--out", str(out)]


def run_args(source, out, seed=0):
    return cli.build_parser().parse_args([*argv(source, out), "--seed", str(seed)])


def fake_generate(instruction, passage):
    assert "neutral phrasing" in instruction
    if passage.startswith("fail"):
        raise RuntimeError("generation failed")
    if passage.startswith("malformed"):
        return "There are no bullets here."
    if instruction.startswith("Summarize"):
        return "" if passage.startswith("empty-summary") else passage.split()[0]
    return f"An introduction to be ignored\n- {passage.split()[0]}"


def rows(out, split):
    return [
        json.loads(line)["messages"]
        for line in (out / f"{split}.jsonl").read_text().splitlines()
    ]


def test_docs_cli_document_holdout(tmp_path, monkeypatch, capsys):
    source = tmp_path / "private-input"
    (source / "nested").mkdir(parents=True)
    for doc in range(5):
        path = source / ("nested" if doc else "") / f"Doc{doc}.md"
        path.write_text(
            "# Chapter\n\n" + "\n\n".join(prose(f"d{doc}p{i}") for i in range(10))
        )
    (source / "extra.pdf").touch()
    (source / "duplicate.md").symlink_to(source / "Doc0.md")
    (source / "notes.txt").write_text("tiny")
    out = tmp_path / "output space"
    out.mkdir()
    from rapid_mlx.telemetry import consent_runtime

    def load(model):
        assert model == "mlx-community/Qwen3.5-4B-MLX-4bit"
        assert consent_runtime.kill_switch_active()
        return fake_generate

    monkeypatch.setattr(dataset, "_load_generator", load)
    elapsed = iter((10.0, 12.375))
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", *argv(source, out)])
    with monkeypatch.context() as patch:
        patch.setattr(dataset, "perf_counter", lambda: next(elapsed))
        with pytest.raises(SystemExit) as exit_info:
            cli.main()
    assert exit_info.value.code == 0
    stdout = capsys.readouterr().out
    assert stdout.count("  assistant: ") == 3
    assert dataset.EXPECTATIONS in stdout
    guidance, command, runtime = stdout.splitlines()[-3:]
    assert "Pick a text model that supports adapters" in guidance
    assert shlex.split(command) == [
        "rapid-mlx",
        "finetune",
        "<model>",
        "--data",
        str(out),
    ]
    assert runtime == "Elapsed time: 2.38s"
    card_text = (out / "dataset_card.json").read_text()
    assert "private-input" not in card_text and "ordinary" not in card_text
    card = json.loads(card_text)
    assert card["source_files"] == [*(f"Doc{i}.md" for i in range(5)), "notes.txt"]
    docs, transitions = set(), 0
    for split, count in zip(("train", "valid", "eval"), (20, 10, 20), strict=True):
        stats = card["splits"][split]
        assert docs.isdisjoint(stats["files"])
        docs.update(stats["files"])
        samples = rows(out, split)
        assert stats["samples"] == len(samples) == count
        for user, answer in samples:
            label = answer["content"].split()[0]
            assert answer == {"role": "assistant", "content": prose(label)}
            assert user["role"] == "user" and user["content"].endswith(f"- {label}")
            if user["content"].startswith("Previous"):
                transitions += 1
                previous = label[:-1] + str(int(label[-1]) - 1)
                assert user["content"].splitlines()[1] == previous
    assert transitions == 15

    monkeypatch.setattr(dataset, "_load_generator", lambda model: fake_generate)
    for seed in (0, 5):
        repeat = tmp_path / f"dataset-{seed}"
        assert dataset.dataset_command(run_args(source, repeat, seed)) == 0
        assert (rows(repeat, "train") == rows(out, "train")) == (seed == 0)


def test_single_document_and_failed_outline(tmp_path, monkeypatch, capsys):
    source = tmp_path / "paper.txt"
    labels = ("empty-summary", "b", "fail", "malformed", "e")
    first = "\n\n".join(map(prose, labels))
    source.write_text(f"First\n====\n{first}\n\nLast\n====\n{prose('held')}")
    monkeypatch.setattr(dataset, "_load_generator", lambda model: fake_generate)
    out = tmp_path / "out"
    assert dataset.dataset_command(run_args(source, out)) == 0
    card = json.loads((out / "dataset_card.json").read_text())
    assert card["skipped_outlines"] == 2
    assert card["words"]["dropped"]["failed_outlines"] == sum(
        len(prose(label).split()) for label in ("fail", "malformed")
    )
    assert card["split_method"] == {"eval": "final_sections", "valid": "passages"}
    assert "held" in rows(out, "eval")[0][-1]["content"]
    assert all(
        "Previous passage" not in user["content"] for user, _ in rows(out, "train")
    )
    stderr = capsys.readouterr().err
    assert "weaker" in stderr and "few-shot" in stderr and "--batch-size 1" in stderr


def test_tex_and_markdown_cleanup(tmp_path):
    tex = tmp_path / "paper.tex"
    tex.write_text(r"""\documentclass{article}
\title{Hidden preamble}
\begin{document}
\maketitle
\section*[Short]{Heading}
% a comment
\subsection{Subheading}
A \textbf{bold} sentence with 15\% price \$20, citation \cite[p. 1]{a}, ref \ref{x}.
\begin{table*}\begin{tabular}TABLE\end{tabular}\end{table*}
\begin{figure}FIGURE\end{figure}
\begin{align*}MATH\end{align*}
Text $inline$ and \[display\] and \(math\) and $$display$$.
\bibliography{bib}
\end{document}
Trailing garbage""")
    dropped = dict.fromkeys(dataset._DROP_REASONS, 0)
    groups = dataset._sections(tex.read_text(), ".tex", dropped)
    sections = [section for group in groups for section in group]
    assert len(sections) == 3
    body = sections[-1]
    assert "A bold sentence with 15% price $20, citation , ref ." in body
    assert "Text  and  and  and ." in body
    for removed in (
        "preamble",
        "comment",
        "FIGURE",
        "TABLE",
        "MATH",
        "bibliography",
        "garbage",
    ):
        assert removed not in "".join(sections)

    md = tmp_path / "article.md"
    md.write_text(
        "---\nfrontmatter\n---\n# Title\n## First\n"
        + prose("one")
        + "\n```\n## Hidden\n```\n## References\n"
        + prose("boilerplate")
        + "\n### Nested references\nIgnored\n## Second\n"
        + prose("two")
    )
    documents, _ = dataset._read_docs(md, dropped)
    assert documents == [("article.md", [[prose("one")], [prose("two")]])]


def test_markdown_cleanup_and_word_accounting(tmp_path, monkeypatch, capsys):
    front_matter = "---\ntitle: metadata\n..."
    code_and_tables = (
        "````python\n## Hidden code heading\n```\nprint('hidden code')\n````\n\n"
        "~~~sh\necho hidden\n~~~\n\n"
        "| hidden table | cells |\n| --- | --- |\n| first | second |\n\n"
        "<table>\n<tr><td>hidden HTML table</td></tr>\n</table>\n\n"
        "    hidden indented code"
    )
    references = (
        "## References\nhidden citation\n### Nested refs\nhidden nested citation"
    )
    definition = '[source]: https://example.test/ref "reference title"'
    first = 'An [inline link](https://example.test/a(b) "link title") stays.'
    second = (
        'A <span data-label="hidden attribute">raw tag</span> and <!-- hidden comment --> '
        "[![nested [alt link](/url) text](/image)](/wrapper) a [reference link][source] stay."
    )
    head = "# Article\n\n## First"
    subhead = "## Long"
    tail_head = "### Tail"
    short_head = "# Separate"
    trailing = "A short trailing section."
    isolated = "Too short to keep."
    source = tmp_path / "article.md"
    raw = "\n\n".join(
        (
            front_matter,
            head,
            first,
            code_and_tables,
            subhead,
            prose("body"),
            second,
            tail_head,
            trailing,
            references,
            short_head,
            isolated,
            definition,
        )
    )
    source.write_text(raw)
    out = tmp_path / "out"
    monkeypatch.setattr(dataset, "_load_generator", lambda model: fake_generate)
    assert dataset.dataset_command(run_args(source, out)) == 0
    kept_first = "An inline link stays."
    kept_second = "A raw tag and   a reference link stay."
    passage = "\n\n".join((kept_first, prose("body"), kept_second, trailing))
    assert rows(out, "train")[0][1]["content"] == passage
    card = json.loads((out / "dataset_card.json").read_text())
    assert card["words"] == {
        "read": len(raw.split()),
        "kept": len(passage.split()),
        "dropped": {
            "front_matter": len(front_matter.split()),
            "code_tables": len(code_and_tables.split()),
            "references": len(references.split()) + len(definition.split()),
            "headings": len(" ".join((head, subhead, tail_head, short_head)).split()),
            "markdown_markup": len((first + " " + second).split())
            - len((kept_first + " " + kept_second).split()),
            "latex_markup": 0,
            "too_short": len(isolated.split()),
            "failed_outlines": 0,
        },
    }
    totals = card["words"]
    dropped = sum(totals["dropped"].values())
    assert totals["read"] == totals["kept"] + dropped
    stdout = capsys.readouterr().out
    assert f"{totals['read']} read, {totals['kept']} kept, {dropped} dropped" in stdout
    for reason, label in dataset._DROP_REASONS.items():
        assert f"{label}: {totals['dropped'][reason]}" in stdout


def test_short_sections_merge_in_order_without_crossing_top_level_headings(tmp_path):
    short = [f"Section {i} " + "word " * 50 for i in range(6)]
    md = tmp_path / "post.md"
    md.write_text(
        short[0]
        + "\n\n## Second\n"
        + short[1]
        + "\n\n# Article\n## Third\n"
        + short[2]
        + "\n\n### Fourth\n"
        + short[3]
        + "\n\n# Other\n"
        + short[4]
        + "\n\n# Final\n"
        + short[5]
    )
    dropped = dict.fromkeys(dataset._DROP_REASONS, 0)
    documents, read = dataset._read_docs(md, dropped)
    passages = ["\n\n".join(part.rstrip() for part in short[i : i + 2]) for i in (0, 2)]
    assert documents == [(md.name, [[passage] for passage in passages])]
    assert dropped["too_short"] == 104
    assert read == sum(len(passage.split()) for passage in passages) + sum(
        dropped.values()
    )


def test_passages_preserve_prose_and_split_long_paragraphs():
    merged = prose("short", 108) + "\n\n\n " + prose("more", 108)
    assert list(dataset._passages(merged)) == [merged]
    natural = [prose("first", 300), prose("second", 270)]
    assert list(dataset._passages("\n\n".join(natural))) == natural
    long = prose("long", 547)
    chunks = list(dataset._passages(long))
    assert len(chunks) == 2
    assert " ".join(chunks) == long
    assert all(200 <= len(chunk.split()) <= 500 for chunk in chunks)
    assert list(dataset._passages("too short")) == []


@pytest.mark.parametrize(
    ("name", "ratio", "out_exists", "error"),
    [
        ("paper.PDF", 0.15, False, "PDF input is out of scope"),
        ("paper.html", 0.15, False, "Expected a .md"),
        ("empty.md", 0.15, False, "No usable prose"),
        ("paper.md", float("nan"), False, "--val-ratio"),
        ("paper.md", 0.15, True, "--out must be a new or empty directory"),
    ],
)
def test_input_errors(tmp_path, capsys, name, ratio, out_exists, error):
    out = tmp_path / "out"
    if out_exists:
        out.mkdir()
        (out / "test.jsonl").touch()
    args = run_args(tmp_path / name, out)
    args.val_ratio = ratio
    assert dataset.dataset_command(args) == 1
    assert error in capsys.readouterr().err


def test_tiny_dataset_warns_and_still_writes(tmp_path, monkeypatch, capsys):
    source = tmp_path / "tiny.md"
    source.write_text(prose("fail"))
    monkeypatch.setattr(dataset, "_load_generator", lambda model: fake_generate)
    out = tmp_path / "out"
    assert dataset.dataset_command(run_args(source, out)) == 0
    assert "at least one train and one valid sample" in capsys.readouterr().err
    assert rows(out, "train") == []
    stats = json.loads((out / "dataset_card.json").read_text())["words"]
    assert stats["read"] == stats["dropped"]["failed_outlines"]
    assert stats["kept"] == 0


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ([1, 1], (1, 1, 0)),
        ([1, 10], (6, 4, 1)),
        ([5, 5, 8], (4, 4, 10)),
        ([5, 5, 7], (5, 5, 7)),
        ([10] * 8, (30, 10, 40)),
        ([2, 2, 24, 24], (2, 2, 48)),
    ],
)
def test_split_edge_cases(counts, expected):
    groups = [[(0, doc, i) for i in range(count)] for doc, count in enumerate(counts)]
    ratio = 0.5 if len(counts) == 8 else 0.15
    splits, _, _ = dataset._split(groups, [1] * len(groups), ratio, random.Random(0))
    assert tuple(map(len, splits)) == expected


def test_generator_reuses_local_loader(tmp_path, monkeypatch, capsys):
    import psutil

    from rapid_mlx import _mlx_compat, model_sizes

    (tmp_path / "model.safetensors").write_bytes(b"fake weights")
    calls = []
    monkeypatch.setattr(_mlx_compat, "install", lambda: calls.append("compat"))
    monkeypatch.setattr(model_sizes, "size_bytes", lambda _: None)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=1))

    template = Mock(return_value="formatted")
    generate = Mock(return_value="- fact")
    sampler = Mock(return_value="greedy")
    tokenizer = SimpleNamespace(apply_chat_template=template)

    def load(ref):
        assert calls == ["compat"] and ref == str(tmp_path)
        return "model", tokenizer

    for name, attr, function in (
        ("mlx_lm", "generate", generate),
        ("mlx_lm.sample_utils", "make_sampler", sampler),
        ("rapid_mlx.utils.tokenizer", "load_model_with_fallback", load),
    ):
        module = ModuleType(name)
        setattr(module, attr, function)
        monkeypatch.setitem(sys.modules, name, module)
    complete = dataset._load_generator(str(tmp_path))
    assert complete("Outline.", "Local content") == "- fact"
    template.assert_called_once_with(
        [
            {
                "role": "user",
                "content": "Outline.\n\n<passage>\nLocal content\n</passage>",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    sampler.assert_called_once_with(temp=0)
    generate.assert_called_once_with(
        "model", tokenizer, prompt="formatted", max_tokens=192, sampler="greedy"
    )
    assert "exceed available memory" in capsys.readouterr().err
