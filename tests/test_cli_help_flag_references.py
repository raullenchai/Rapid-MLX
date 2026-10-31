"""Ensure help references flags available in the relevant command context."""

import argparse
import re

from rapid_mlx.cli_parser import build_parser
from rapid_mlx.cua.cli import build_parser as build_cua_parser

FLAG = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]*[a-z0-9]")
COMMAND = re.compile(r"\brapid-mlx\s+([a-z][a-z0-9-]*(?:\s+[a-z][a-z0-9-]*)?)")
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

    result = dict(walk(root, ()))
    # CUA dispatches to a separate parser after the top-level remainder.
    result.update(
        {
            ("cua",) + path: parser
            for path, parser in walk(build_cua_parser(), ())
            if path
        }
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


def qualified_flags(text, all_parsers):
    # Commands may be written as full invocations or as backtick examples.
    candidates = [match.group(1).split() for match in COMMAND.finditer(text)]
    for span in CODE.findall(text):
        words = span.split()
        if words and words[0] == "rapid-mlx":
            words = words[1:]
        candidates.append(words)
    result = set()
    for words in candidates:
        path = ()
        for word in words:
            if path + (word,) not in all_parsers:
                break
            path += (word,)
        if path:
            result.update(owned(all_parsers[path]))
    return result


def test_help_flag_references_belong_to_the_relevant_command():
    all_parsers = parsers(build_parser())
    errors = []
    seen = set()
    for path, parser in all_parsers.items():
        if id(parser) in seen:
            continue
        seen.add(id(parser))
        for location, text in texts(parser):
            if not isinstance(text, str) or text == argparse.SUPPRESS:
                continue
            allowed = owned(parser) | qualified_flags(text, all_parsers)
            if location in PASSTHROUGH.get(path, set()):
                allowed |= owned(all_parsers[("serve",)])
            unknown = set(FLAG.findall(text)) - allowed
            if unknown:
                errors.append(
                    f"{' '.join(path) or '<root>'} {location}: {sorted(unknown)}"
                )
    assert not errors, "\n" + "\n".join(errors)


def test_benchmark_and_share_help_regressions():
    all_parsers = parsers(build_parser())
    for path, parser in all_parsers.items():
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
