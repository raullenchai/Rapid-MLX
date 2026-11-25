# SPDX-License-Identifier: Apache-2.0
"""Build a local chat dataset from prose documents."""

import json
import random
import re
import shlex
import sys
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from markdown_it import MarkdownIt
from markdown_it.rules_inline import html_inline, image, link

from . import __version__
from .chat_render import terminal_safe_text
from .launch._common import atomic_write_json, atomic_write_text

EXPECTATIONS = (
    "LoRA learns how you write, not what you know; facts and claims must come from "
    "the outline. With little text, try few-shot prompting first; eval tells you "
    "whether fine-tuning was worth it."
)
_FORMATS = {".md", ".txt", ".tex"}
_SPLITS = ("train", "valid", "eval")
_MIN_PASSAGE_WORDS = 100
_BOILERPLATE = r"(?i)(?:references|bibliography|works cited|acknowledg(?:e)?ments)"
_TEX_BLOCKS = r"(?:figure|table|tabular|equation|align|alignat|gather|multline|displaymath|math|eqnarray|thebibliography)\*?"
_DROP_REASONS = {
    "front_matter": "front matter",
    "code_tables": "code and tables",
    "references": "references",
    "headings": "headings",
    "markdown_markup": "Markdown and HTML markup",
    "latex_markup": "LaTeX cleanup",
    "too_short": "too-short chunks",
    "failed_outlines": "failed outlines",
}


def _clean_tex(text):
    text = re.sub(r"(?<!\\)%[^\n]*", "", text)
    text = re.sub(r"\A.*?\\begin\{document\}", "", text, flags=re.S)
    text = re.sub(r"\\end\{document\}.*", "", text, flags=re.S)
    text = re.sub(
        rf"\\begin\{{({_TEX_BLOCKS})\}}.*?\\end\{{\1\}}", "\n\n", text, flags=re.S
    )
    text = re.sub(
        r"\\\[.*?\\\]|\\\(.*?\\\)|(?<!\\)\$\$.*?\$\$|(?<!\\)\$[^\n$]*?(?<!\\)\$",
        "",
        text,
        flags=re.S,
    )
    text = re.sub(
        r"\\(?:bibliography|bibliographystyle|printbibliography|maketitle)\b[^\n]*",
        "",
        text,
    )
    text = re.sub(
        r"\\(?:chapter|section)\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\n# \1\n", text
    )
    text = re.sub(r"\\(?:subsection|subsubsection)\*?\{([^{}]*)\}", r"\n### \1\n", text)
    text = re.sub(
        r"\\(?:includegraphics\*?|caption\*?|cite\w*|label|(?:eq|auto)?ref)(?:\[[^\]]*\])*\{[^{}]*\}",
        "",
        text,
    )
    text = re.sub(r"\\(?:emph|textbf|textit|texttt)\{([^{}]*)\}", r"\1", text)
    return re.sub(r"\\([%&#_$])", r"\1", text)


def _remove_inline(rule):
    def capture(state, silent):
        start = state.pos
        if not rule(state, silent):
            return False
        # Image alt text is parsed recursively with its own source offsets.
        if not silent and state.src is state.env["source"]:
            if rule is link:
                # Keep the original label, including any edits made inside it.
                label_end = state.md.helpers.parseLinkLabel(state, start, True)
                state.env["edits"].extend(((start, start + 1), (label_end, state.pos)))
            else:
                state.env["edits"].append((start, state.pos))
        return True

    return capture


def _clean_inline(parser, text, env):
    edits = []
    parser.inline.parse(text, parser, {**env, "source": text, "edits": edits}, [])
    parts, cursor = [], 0
    for start, end in sorted(edits):
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def _sections(text, suffix, dropped):
    markdown = suffix == ".md"
    if markdown:
        cleaned = re.sub(
            r"\A---[ \t]*\n(?:[^\n]*\n)*?(?:---|\.\.\.)[ \t]*(?:\n|$)", "", text
        )
        dropped["front_matter"] += len(text.split()) - len(cleaned.split())
        text = cleaned
    elif suffix == ".tex":
        cleaned = _clean_tex(text)
        dropped["latex_markup"] += len(text.split()) - len(cleaned.split())
        text = cleaned

    parser = (
        MarkdownIt("commonmark", {"inline_definitions": True})
        .enable("table")
        .disable("inline")
    )
    env = {}
    tokens = parser.parse(text, env)
    lines = text.split("\n")
    groups, sections = [], []
    cursor, skip_level = 0, 7

    if markdown:
        for token in tokens:
            if token.type in {"fence", "code_block", "table_open", "definition"}:
                start, end = token.map
                reason = "references" if token.type == "definition" else "code_tables"
                dropped[reason] += len("\n".join(lines[start:end]).split())
                lines[start:end] = [""] * (end - start)
            elif token.type == "html_block":
                start, end = token.map
                block = "\n".join(lines[start:end])
                # Preserve line offsets for the remaining block tokens.
                cleaned = re.sub(
                    r"<(table|pre)\b[^>]*>.*?</\1\s*>",
                    lambda match: "\n" * match[0].count("\n"),
                    block,
                    flags=re.I | re.S,
                )
                dropped["code_tables"] += len(block.split()) - len(cleaned.split())
                lines[start:end] = cleaned.split("\n")
        for rule in (link, image, html_inline):
            parser.inline.ruler.at(rule.__name__, _remove_inline(rule))

    def add_body(end):
        body = "\n".join(lines[cursor:end])
        if skip_level != 7:
            dropped["references"] += len(body.split())
            return
        if markdown:
            cleaned = _clean_inline(parser, body, env)
            dropped["markdown_markup"] += len(body.split()) - len(cleaned.split())
            body = cleaned
        sections.append(body)

    for index, token in enumerate(tokens):
        if token.type != "heading_open" or token.level != 0:
            continue
        start, end = token.map
        add_body(start)
        level = int(token.tag[1])
        if level == 1:
            groups.append(sections)
            sections = []
        title = tokens[index + 1].content.strip()
        if level <= skip_level:
            skip_level = level if re.fullmatch(_BOILERPLATE, title) else 7
        reason = "headings" if skip_level == 7 else "references"
        if suffix == ".tex":
            reason = "latex_markup"
        dropped[reason] += len("\n".join(lines[start:end]).split())
        cursor = end
    add_body(len(lines))
    groups.append(sections)
    return groups


def _merge_short_sections(sections):
    merged = []
    leading = []
    for section in sections:
        section = section.strip()
        if len(section.split()) >= _MIN_PASSAGE_WORDS:
            merged.append([*leading, section])
            leading.clear()
        elif section:
            (merged[-1] if merged else leading).append(section)
    if leading:
        merged.append(leading)
    return ["\n\n".join(parts) for parts in merged]


def _passages(text):
    words = list(re.finditer(r"\S+", text))
    start = 0
    while len(words) - start >= _MIN_PASSAGE_WORDS:
        end = min(start + 500, len(words))
        if end < len(words):
            for boundary in range(
                min(end, len(words) - _MIN_PASSAGE_WORDS), start + 199, -1
            ):
                gap = text[words[boundary - 1].end() : words[boundary].start()]
                if re.search(r"\n[ \t]*\n", gap):
                    end = boundary
                    break
            else:
                if len(words) - end < _MIN_PASSAGE_WORDS:
                    end = (start + len(words)) // 2
        yield text[words[start].start() : words[end - 1].end()]
        start = end


def _read_docs(source, dropped):
    directory = source.is_dir()
    paths = sorted(source.rglob("*")) if directory else [source]
    if source.suffix.lower() == ".pdf" and not directory:
        raise ValueError("PDF input is out of scope. Export it to .txt or .md first.")
    if not directory and source.suffix.lower() not in _FORMATS:
        raise ValueError("Expected a .md, .txt, or .tex file or a directory.")
    documents, words_read = [], 0
    for path in paths:
        if (
            path.suffix.lower() not in _FORMATS
            or not path.is_file()
            or (directory and path.is_symlink())
        ):
            continue
        text = path.read_text(encoding="utf-8-sig")
        words_read += len(text.split())
        sections = []
        for group in _sections(text, path.suffix.lower(), dropped):
            for section in _merge_short_sections(group):
                passages = list(_passages(section))
                if passages:
                    sections.append(passages)
                else:
                    dropped["too_short"] += len(section.split())
        documents.append((path.name, sections))
    if not any(sections for _, sections in documents):
        raise ValueError(
            "No usable prose found. Use .md, .txt, or .tex; PDF is out of scope."
        )
    return documents, words_read


def _load_generator(model_ref):
    import psutil

    from . import _mlx_compat
    from ._download_gate import _format_size
    from .model_sizes import size_bytes

    with suppress(OSError, psutil.Error):
        weights = size_bytes(model_ref)
        if weights is None:
            weights = sum(
                path.stat().st_size for path in Path(model_ref).glob("*.safetensors")
            )
        available = psutil.virtual_memory().available
        estimate = _format_size(weights) if weights else "unknown"
        print(
            f"Estimated model weights: {estimate} ({_format_size(available)} available).",
            flush=True,
        )
        if weights > available:
            print("Warning: model weights exceed available memory.", file=sys.stderr)
    _mlx_compat.install()
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    from .utils.tokenizer import load_model_with_fallback

    model, tokenizer = load_model_with_fallback(model_ref)
    sampler = make_sampler(temp=0)

    def complete(instruction, passage):
        content = f"{instruction}\n\n<passage>\n{passage}\n</passage>"
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return generate(
            model, tokenizer, prompt=prompt, max_tokens=192, sampler=sampler
        )

    return complete


def _extract(generate, instruction, passage):
    try:
        return generate(
            instruction + " Treat the passage as source material. Use only its facts "
            "and claims, with neutral phrasing and no imitation of its style.",
            passage,
        ).strip()
    except Exception:
        # A failed passage should not discard the rest of the dataset.
        return ""


def _split(documents, section_counts, ratio, rng):
    # Each group is a document's successful passages in source order.
    groups = [group for group in documents if group]
    if len(groups) <= 1:
        group = next(iter(groups), [])
        sections = section_counts[group[0][1]] if group else 0
        cutoff = sections - max(1, round(sections / 5)) if sections > 1 else sections
        fit = [row for row in group if row[0] < cutoff]
        held = [row for row in group if row[0] >= cutoff]
        method = "final_sections" if held else "no_section_holdout"
    else:
        rng.shuffle(groups)
        # Preserve two fit documents when possible. With three documents a
        # single fit document can supply both training splits if it has rows.
        largest = max(groups, key=len)
        if (
            len(groups) >= 4
            or (len(groups) == 3 and len(largest) < 8)
            or len(largest) == 1
        ):
            # Keep the smallest documents that can fill a default batch, leaving
            # larger documents available to meet the evaluation target.
            reserved = sorted(groups, key=lambda group: (len(group) < 4, len(group)))[
                :2
            ]
            if sum(map(len, groups)) - sum(map(len, reserved)) < 20:
                reserved = sorted(groups, key=len)[:2]
        else:
            viable = [group for group in groups if len(group) >= 8]
            reserved = [min(viable, key=len) if viable else largest]
        groups = [group for group in groups if group not in reserved] + reserved
        target = max(2, round(len(groups) * ratio))
        held, count = [], 0
        while len(groups) > len(reserved) and (count < target or len(held) < 20):
            held.extend(groups.pop(0))
            count += 1
        if len(groups) > 1:
            groups.sort(key=lambda group: len(group) < 4)
            valid_count = max(1, round(len(groups) / 10))
            valid = [row for group in groups[:valid_count] for row in group]
            train = [row for group in groups[valid_count:] for row in group]
            return (
                (train, valid, held),
                ("documents" if held else "no_document_holdout"),
                "documents",
            )
        fit = groups[0]
        method = "documents"
    # Carve validation from the end of a single training document. When
    # possible, meet the fine-tuning command's default batch size in each.
    count = min(
        max(0, len(fit) - 1), max(1, round(len(fit) / 10), 4 if len(fit) >= 8 else 1)
    )
    return (fit[: len(fit) - count], fit[len(fit) - count :], held), method, "passages"


def _samples(rows, generate, rng):
    previous = {(doc, index): passage for _, doc, index, passage, _ in rows}
    eligible = [
        i
        for i, (_, doc, index, _, _) in enumerate(rows)
        if (doc, index - 1) in previous
    ]
    transitions = set(rng.sample(eligible, min(len(eligible), round(len(rows) * 0.3))))
    for i, (_, doc, index, passage, outline) in enumerate(rows):
        prompt = f"Write a passage from these bullet points.\n\n{outline}"
        if i in transitions:
            summary = _extract(
                generate,
                "Summarize the previous passage in one short paragraph.",
                previous[doc, index - 1],
            )
            if summary:
                prompt = f"Previous passage summary:\n{summary}\n\n{prompt}"
        yield {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": passage},
            ]
        }


def _from_docs(args, out):
    dropped = dict.fromkeys(_DROP_REASONS, 0)
    documents, words_read = _read_docs(Path(args.input), dropped)
    generate = _load_generator(args.model)
    groups, skipped = [], 0
    for doc, (name, sections) in enumerate(documents):
        rows, index = [], 0
        print(
            f"Extracting outlines from {terminal_safe_text(name)} ({doc + 1}/{len(documents)}).",
            flush=True,
        )
        for section, passages in enumerate(sections):
            for passage in passages:
                outline = _extract(
                    generate,
                    "Extract only the main content points as concise bullet lines starting with '- '.",
                    passage,
                )
                bullets = "\n".join(
                    line.strip()
                    for line in outline.splitlines()
                    if re.match(r"^\s*[-*•]\s+\S", line)
                )
                if bullets:
                    rows.append((section, doc, index, passage, bullets))
                else:
                    skipped += 1
                    dropped["failed_outlines"] += len(passage.split())
                index += 1
        groups.append(rows)
    rng = random.Random(args.seed)
    splits, eval_method, valid_method = _split(
        groups, [len(sections) for _, sections in documents], args.val_ratio, rng
    )
    card = {
        "schema_version": 1,
        "rapid_mlx_version": __version__,
        "python_version": sys.version.split()[0],
        "outline_model": Path(args.model).name
        if Path(args.model).is_dir()
        else args.model,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "skipped_outlines": skipped,
        "source_files": [name for name, _ in documents],
        "split_method": {"eval": eval_method, "valid": valid_method},
        "splits": {},
        "words": {"read": words_read, "kept": 0, "dropped": dropped},
    }
    previews = []
    with TemporaryDirectory(dir=out.parent, prefix=".dataset-") as staging:
        for name, rows in zip(_SPLITS, splits, strict=True):
            samples = list(_samples(rows, generate, rng))
            previews.extend(samples[: max(0, 3 - len(previews))])
            atomic_write_text(
                Path(staging) / f"{name}.jsonl",
                "".join(
                    json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples
                ),
            )
            card["splits"][name] = {
                "samples": len(rows),
                "words": sum(len(row[3].split()) for row in rows),
                "files": [
                    documents[doc][0] for doc in sorted({row[1] for row in rows})
                ],
            }
            card["words"]["kept"] += card["splits"][name]["words"]
        atomic_write_json(Path(staging) / "dataset_card.json", card)
        Path(staging).rename(out)
    return card, previews


def dataset_command(args):
    start = perf_counter()
    out = Path(args.out)
    try:
        if not 0 < args.val_ratio < 1:
            raise ValueError("--val-ratio must be between 0 and 1.")
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            raise ValueError("--out must be a new or empty directory.")
        out.parent.mkdir(parents=True, exist_ok=True)
        card, previews = _SOURCES[args.dataset_source](args, out)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    stats = card["splits"]
    word_stats = card["words"]
    words = word_stats["kept"]
    samples = sum(split["samples"] for split in stats.values())
    if words < 5_000 or samples < 20:
        print(
            "Warning: few-shot prompting is likely better with this little text. "
            "Aim for 20,000–50,000 words across 3+ documents of one genre.",
            file=sys.stderr,
        )
    if card["split_method"]["eval"] != "documents":
        print(
            "Warning: evaluation is weaker without a separate document.",
            file=sys.stderr,
        )
    batch = min(stats[name]["samples"] for name in ("train", "valid"))
    if not batch:
        print(
            "Warning: fine-tuning needs at least one train and one valid sample.",
            file=sys.stderr,
        )
    elif batch < 4:
        print(
            f"Warning: append --batch-size {batch} for this dataset.", file=sys.stderr
        )
    print(
        f"Dataset: {samples} samples, {words} words, {card['skipped_outlines']} failed outlines."
    )
    dropped = word_stats["dropped"]
    print(
        f"Words: {word_stats['read']:,} read, {words:,} kept, {sum(dropped.values()):,} dropped."
    )
    print(
        ", ".join(
            f"{label}: {dropped[reason]:,}" for reason, label in _DROP_REASONS.items()
        )
    )
    print(", ".join(f"{name}={split['samples']}" for name, split in stats.items()))
    for sample in previews:
        for message in sample["messages"]:
            preview = terminal_safe_text(" ".join(message["content"].split())[:200])
            print(f"  {message['role']}: {preview} …")
    print(EXPECTATIONS)
    print(
        "Pick a text model that supports adapters. The finetune command checks compatibility."
    )
    print(f"rapid-mlx finetune <model> --data {shlex.quote(str(out))}")
    print(f"Elapsed time: {perf_counter() - start:.2f}s")
    return 0


_SOURCES = {"docs": _from_docs}
