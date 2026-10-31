"""Ensure help references flags available in the relevant command context."""

import argparse
import re

from rapid_mlx.cli_parser import build_parser
from rapid_mlx.cua.cli import build_parser as build_cua_parser

FLAG = re.compile(r"(?<![\w-])--[a-z](?:[a-z0-9-]*[a-z0-9])?")
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
            return False, set()
        return True, set().union(*(owned(p) for p in all_parsers[path]))

    contexts = []
    for span in CODE.finditer(text):
        words = span.group(1).split()
        if words and words[0] == "rapid-mlx":
            words = words[1:]
        qualified, flags = command_flags(words)
        contexts.append((span.start(), span.end(), qualified, flags))
    for invocation in COMMAND.finditer(text):
        line_start = text.rfind("\n", 0, invocation.start()) + 1
        shell_line = not text[line_start : invocation.start()].strip()
        tokens = list(re.finditer(r"\S+", text[invocation.end() :]))
        path = ()
        index = 0
        end = invocation.end()
        while index < len(tokens):
            word = tokens[index].group()
            if path + (word,) not in all_parsers:
                break
            path += (word,)
            end = invocation.end() + tokens[index].end()
            index += 1
        if shell_line:
            line_end = text.find("\n", invocation.end())
            if line_end == -1:
                line_end = len(text)
            comment = text.find(" #", invocation.end(), line_end)
            end = comment if comment != -1 else line_end
            qualified, flags = command_flags(path)
            contexts.append((invocation.end(), end, qualified, flags))
            continue
        expects_value = False
        while index < len(tokens):
            token = tokens[index]
            word = token.group()
            if word.startswith("--"):
                expects_value = True
            elif expects_value:
                expects_value = False
            else:
                break
            end = invocation.end() + token.end()
            if word.endswith((",", ";", ":", ".", ")")):
                break
            index += 1
        qualified, flags = command_flags(path)
        contexts.append((invocation.end(), end, qualified, flags))
    unknown = set()
    for match in FLAG.finditer(text):
        qualified_flags = [
            flags
            for start, end, qualified, flags in contexts
            if qualified and start <= match.start() < end
        ]
        if qualified_flags:
            if not any(match.group() in flags for flags in qualified_flags):
                unknown.add(match.group())
        elif match.group() not in local_flags:
            unknown.add(match.group())
    return unknown


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
    assert unknown_flags("rapid-mlx models --host", {"--host"}, all_parsers) == {
        "--host"
    }
    assert unknown_flags("`rapid-mlx models --host`", {"--host"}, all_parsers) == {
        "--host"
    }
    assert not unknown_flags("rapid-mlx serve --host", {"--host"}, all_parsers)
    assert unknown_flags(
        "Use rapid-mlx serve --port 8000 and then use --host on this command",
        set(),
        all_parsers,
    ) == {"--host"}
    assert unknown_flags(
        "Use rapid-mlx serve --port 8000, then set --host.", set(), all_parsers
    ) == {"--host"}
    assert not unknown_flags(
        "Run rapid-mlx serve --port 8000 --host 0.0.0.0", set(), all_parsers
    )
    assert unknown_flags("Use --x here.", set(), all_parsers) == {"--x"}
    assert not unknown_flags(
        "  rapid-mlx serve qwen3.5-9b-4bit --port 8000 # server",
        set(),
        all_parsers,
    )
    assert not unknown_flags(
        "  rapid-mlx launch claude-code --model x", set(), all_parsers
    )
    assert unknown_flags(
        "  rapid-mlx serve model --port 8000 # --host", set(), all_parsers
    ) == {"--host"}


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
