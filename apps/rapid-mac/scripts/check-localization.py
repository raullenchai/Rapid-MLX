#!/usr/bin/env python3
"""Pragmatic Swift UI localization discovery; no third-party dependencies.

Tracks direct UI arguments and same-file String bindings. Dynamic server/user
content is outside this heuristic; native catalog extraction remains required.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Identity, URLs, paths, symbols and unit-only readouts are not prose.
ALLOW = {"Rapid-MLX", "GitHub", "Hugging Face", "QuickSilver", "MCP", "LLM", "macOS", "R", "URL", "herdr", "yyyy-MM-dd", "UTC", "MMMMy", "MMMd"}


def balanced(source, start):
    """End of a parenthesized expression, skipping strings and comments."""
    depth, i = 1, start
    while i < len(source):
        if source.startswith('//', i):
            end = source.find('\n', i)
            i = len(source) if end < 0 else end
        elif source[i] == '"':
            i = literal_end(source, i)
        else:
            if source[i] == '(': depth += 1
            if source[i] == ')':
                depth -= 1
                if depth == 0: return i + 1
            i += 1
    return len(source)


def literal_end(source, start):
    i = start + 1
    while i < len(source):
        if source.startswith('\\(', i): i = balanced(source, i + 2)
        elif source[i] == '\\': i += 2
        elif source[i] == '"': return i + 1
        else: i += 1
    return i


def literals(source):
    i = 0
    while i < len(source):
        if source.startswith('//', i):
            end = source.find('\n', i)
            i = len(source) if end < 0 else end
        elif source.startswith('/*', i):
            end = source.find('*/', i + 2)
            i = len(source) if end < 0 else end + 2
        elif source[i] == '"':
            end = literal_end(source, i)
            yield i, end, source[i + 1:end - 1]
            i = end
        else: i += 1


def nested_prose(value):
    i = 0
    while i < len(value):
        if value.startswith('\\(', i):
            end = balanced(value, i + 2)
            if unlocalized(value[i + 2:end - 1]): return True
            i = end
        else: i += 1
    return False


def prose(value):
    # Remove complete interpolations before deciding whether literal prose exists.
    out, i = '', 0
    while i < len(value):
        if value.startswith('\\(', i): i = balanced(value, i + 2)
        else:
            out += value[i]
            i += 1
    return nested_prose(value) or (bool(re.search('[A-Za-z]{2,}', out)) and out not in ALLOW and not re.match(r'(https?://|/|~/)', out))


def unlocalized(expression):
    # Remove localized calls, preserving offsets for diagnostics.
    for match in reversed(list(re.finditer(r'(?:String|LocalizedStringKey)\s*\(\s*(?:localized\s*:)?', expression))):
        if expression.startswith('String', match.start()) and 'localized' not in match.group(): continue
        end = balanced(expression, expression.index('(', match.start()) + 1)
        expression = expression[:match.start()] + ' ' * (end - match.start()) + expression[end:]
    return [(a, b, s) for a, b, s in literals(expression) if prose(s)]


def scan(path):
    source = path.read_text()
    # Mask comments and literals for call discovery, retaining exact offsets.
    code = source
    for a, b, _ in reversed(list(literals(source))): code = code[:a] + ' ' * (b-a) + code[b:]
    code = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/', lambda m: re.sub(r'[^\n]', ' ', m.group()), code)
    bindings = {}
    for m in re.finditer(r'\b(?:let|var)\s+(\w+)(?:\s*:\s*String\??)?\s*=', code):
        start = m.end()
        while start < len(source) and source[start].isspace(): start += 1
        end = start
        while end < len(source) and source[end] != '\n':
            if source[end] == '"': end = literal_end(source, end)
            elif source[end] == '(': end = balanced(source, end + 1)
            else: end += 1
        expr = source[start:end]
        if unlocalized(expr): bindings.setdefault(m[1], []).append((m.end(), expr))
    hits = []
    pattern = r'\b(Text|Button|Label|Toggle|accessibilityLabel|accessibilityValue|accessibilityHint|help)\s*\('
    for m in re.finditer(pattern, code):
        end = balanced(source, m.end())
        expr = source[m.end():end - 1]
        # Only inspect a control's title, not its action or view builder.
        if m[1] in {'Button', 'Label', 'Toggle'}:
            i = 0
            while i < len(expr):
                if expr[i] == '"': i = literal_end(expr, i)
                elif expr[i] == '(': i = balanced(expr, i + 1)
                elif expr[i] == ',':
                    expr = expr[:i]
                    break
                else: i += 1
        raw = unlocalized(expr)
        direct_literal = expr.lstrip().startswith('"') and literal_end(expr.strip(), 0) == len(expr.strip())
        if direct_literal and m[1] != 'Text':
            raw = [x for x in raw if '\\(' in x[2]]  # SwiftUI literal overload localizes static copy.
        if direct_literal and m[1] == 'Text': raw = [x for x in raw if nested_prose(x[2])]
        # A variable initialized with an English literal/interpolation is String.
        if re.fullmatch(r'\w+', expr.strip()):
            raw += [(0, 0, e) for _, e in bindings.get(expr.strip(), [])]
        if raw:
            line = source.count('\n', 0, m.start()) + 1
            hits.append((line, m[1], expr.strip().replace('\n', ' ')[:180]))
    # Trace local String-producing accessibility helpers as well as bindings.
    helper_names = set()
    for match in re.finditer(r'(?:\.(?:help|accessibilityLabel|accessibilityValue|accessibilityHint)|\bText)\s*\(', code):
        expression = source[match.end():balanced(source, match.end()) - 1]
        helper_names.update(re.findall(r'\b\w+\b', expression))
    declaration = r'\b(?:func|var)\s+(\w+)[^{}\n]*(?:->\s*String\??|:\s*String\??)\s*\{'
    for match in re.finditer(declaration, code):
        if match[1] == 'accessibilityIdentifier': continue
        if match[1] not in helper_names and not re.search(r'accessibility|tooltip|[Hh]elp', match[1]): continue
        end, depth = match.end(), 1
        while end < len(code) and depth:
            if code[end] == '{': depth += 1
            if code[end] == '}': depth -= 1
            end += 1
        body = source[match.end():end - 1]
        if 'localizedString(forKey:' in body: continue  # Explicit localized helper.
        for start, finish, value in unlocalized(body):
            if body[finish:].lstrip().startswith(':'): continue  # Switch wire identifier.
            before = body[:start].rstrip()
            if before.endswith(('==', '!=')): continue  # Protocol comparison value.
            line = source.count('\n', 0, match.end() + start) + 1
            hits.append((line, match[1], value))
    return hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-strings', type=Path,
                        help='Optional directory emitted by swiftc -emit-localized-strings-path; checks exact typed keys')
    args = parser.parse_args()
    count = 0
    for path in sorted((ROOT / 'Sources/Rapid').rglob('*.swift')):
        if path.name == 'DevSnapshot.swift': continue  # Development fixtures, not shipped UI.
        for line, kind, expr in scan(path):
            print(f'{path.relative_to(ROOT)}:{line}: {kind}: {expr}')
            count += 1
    if args.compiler_strings:
        catalog = json.loads((ROOT / 'Sources/Rapid/Resources/Localizable.xcstrings').read_text())['strings']
        for path in sorted(args.compiler_strings.glob('*.stringsdata')):
            if path.stem == 'DevSnapshot': continue
            data = json.loads(path.read_text())
            for entry in data.get('tables', {}).get('Localizable', []):
                key = entry['key']
                readable = re.sub(r'%(?:\d+\$)?[-+ #0]*(?:\d+)?(?:\.\d+)?(?:lld|llu|ld|lu|d|u|@|f|lf|%)', '', key)
                if not prose(readable): continue
                zh = catalog.get(key, {}).get('localizations', {}).get('zh-Hans', {})
                unit = zh.get('stringUnit') or zh.get('variations', {}).get('plural', {}).get('other', {}).get('stringUnit', {})
                if unit.get('state') != 'translated' or not unit.get('value'):
                    print(f"{data['source']}:{entry['location']['startingLine']}: missing exact zh-Hans key: {key}")
                    count += 1
    print(f'Localization sweep: {count} remaining hits')
    return bool(count)


if __name__ == '__main__': sys.exit(main())
