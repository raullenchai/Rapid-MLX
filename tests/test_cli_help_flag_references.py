"""Ensure help references flags available in the relevant command context."""

import argparse
import re

from rapid_mlx.cli_parser import build_parser
from rapid_mlx.cua.cli import build_parser as build_cua_parser

FLAG = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]*[a-z0-9]")
COMMAND = re.compile(r"\brapid-mlx[ \t]+")
CODE = re.compile(r"`+([^`]+)`+")
CHIP = re.compile(r"\bM[1-5]\b")

# These texts explicitly describe flags forwarded to serve, not local options.
PASSTHROUGH = {("share",): {"epilog"}, ("service", "install"): {"serve_args"}}


def parsers(root):
    seen = set()

    def walk(parser, path):
        yield path, parser
        if id(parser) in seen:
            return
        seen.add(id(parser))
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, child in action.choices.items():
                    yield from walk(child, path + (name,))

    result = list(walk(root, ()))
    # CUA dispatches to a separate parser after the top-level remainder.
    result.extend(
        (("cua",) + path, parser) for path, parser in walk(build_cua_parser(), ())
    )
    return result


def owned(parser):
    return {flag for action in parser._actions for flag in action.option_strings}


def texts(parser):
    yield "description", parser.description
    yield "epilog", parser.epilog
    for action in parser._actions:
        if action.help != argparse.SUPPRESS:
            yield action.dest, action.help
    for group in parser._action_groups:
        yield "group description", group.description


def unknown_flags(text, local_flags, all_parsers):
    def command_flags(words):
        path = ()
        for word in words:
            if path + (word,) not in all_parsers:
                break
            path += (word,)
        if not path:
            return set()
        return set().union(*(owned(p) for p in all_parsers[path]))

    contexts = []
    for span in CODE.finditer(text):
        words = span.group(1).split()
        if words and words[0] == "rapid-mlx":
            words = words[1:]
        contexts.append((span.start(), span.end(), command_flags(words)))
    for invocation in COMMAND.finditer(text):
        end = re.search(r"[,;:.] | \(| — |\n", text[invocation.end() :])
        end = invocation.end() + end.start() if end else len(text)
        contexts.append(
            (invocation.end(), end, command_flags(text[invocation.end() : end].split()))
        )
    return {
        match.group()
        for match in FLAG.finditer(text)
        if match.group() not in local_flags
        and not any(
            start <= match.start() < end and match.group() in flags
            for start, end, flags in contexts
        )
    }


def test_qualified_flag_context_is_limited_to_each_occurrence():
    all_parsers = {}
    for path, parser in parsers(build_parser()):
        all_parsers.setdefault(path, []).append(parser)
    assert unknown_flags(
        "Use `rapid-mlx serve --port 8000`. Also --bogus.", set(), all_parsers
    ) == {"--bogus"}
    assert unknown_flags(
        "Use `serve --port 8000`. Also --port.", set(), all_parsers
    ) == {"--port"}
    assert unknown_flags(
        "Use rapid-mlx serve --port 8000, then set --host on this command.",
        set(),
        all_parsers,
    ) == {"--host"}
    assert not unknown_flags("rapid-mlx models --cached", set(), all_parsers)


def test_help_flag_references_belong_to_the_relevant_command():
    entries = parsers(build_parser())
    all_parsers = {}
    for path, parser in entries:
        all_parsers.setdefault(path, []).append(parser)
    errors = []
    seen = set()
    for path, parser in entries:
        if id(parser) in seen:
            continue
        seen.add(id(parser))
        for location, text in texts(parser):
            if not isinstance(text, str) or text == argparse.SUPPRESS:
                continue
            allowed = owned(parser)
            if location in PASSTHROUGH.get(path, set()):
                allowed |= owned(all_parsers[("serve",)][0])
            unknown = unknown_flags(text, allowed, all_parsers)
            if unknown:
                errors.append(
                    f"{' '.join(path) or '<root>'} {location}: {sorted(unknown)}"
                )
    assert not errors, "\n" + "\n".join(errors)


def test_benchmark_and_share_help_regressions():
    entries = parsers(build_parser())
    all_parsers = dict(entries)
    for path, parser in entries:
        if path and path[0] == "benchmark":
            for location, text in texts(parser):
                if isinstance(text, str) and text != argparse.SUPPRESS:
                    assert not CHIP.search(text), (path, location, text)
    share = all_parsers[("share",)]
    rate_limit = next(a for a in share._actions if "--rate-limit" in a.option_strings)
    assert not CHIP.search(rate_limit.help)
    run = all_parsers[("benchmark", "run")]
    help_by_flag = {
        flag: action.help for action in run._actions for flag in action.option_strings
    }
    assert "--reasoning" not in help_by_flag["--kv-cache-dtype"]
    assert "#" not in help_by_flag["--kv-cache-dtype"]
    assert "--force-spec-decode" not in help_by_flag["--no-spec-decode"]
    spec_help = help_by_flag["--speculative-config"].lower()
    assert "suffix" not in spec_help or "suffix decoding is refused" in spec_help
