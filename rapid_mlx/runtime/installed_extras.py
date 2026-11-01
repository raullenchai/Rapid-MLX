# SPDX-License-Identifier: Apache-2.0
"""Infer installed optional features from distribution metadata."""


def installed_extras() -> list[str]:
    """Mirror install.sh's installed_extras_of_venv; a parity test locks them together."""
    try:
        import importlib.metadata as metadata
        import re

        try:
            from packaging.requirements import Requirement
        except ImportError:
            try:
                from pip._vendor.packaging.requirements import Requirement
            except ImportError:
                Requirement = None

        def normalize(name):
            return re.sub(r"[-_.]+", "-", name).lower()

        try:
            requirements = metadata.distribution("rapid-mlx").requires or []
        except metadata.PackageNotFoundError:
            try:
                requirements = metadata.distribution("vllm-mlx").requires or []
            except metadata.PackageNotFoundError:
                return []
        installed = {
            normalize(d.metadata["Name"]): d.version
            for d in metadata.distributions()
            if d.metadata.get("Name")
        }
        base = set()
        extras = {}
        for requirement in requirements:
            name = re.match(r"[A-Za-z0-9._-]+", requirement)
            if not name:
                continue
            marker = requirement.partition(";")[2]
            extra = re.search(
                r"\bextra\s*==\s*[\"\x27]([^\"\x27]+)[\"\x27]", marker, re.I
            )
            if Requirement is not None:
                try:
                    req = Requirement(requirement)
                    if req.marker and not req.marker.evaluate(
                        {"extra": extra.group(1) if extra else ""}
                    ):
                        continue
                except Exception:
                    continue
            else:
                if re.search(r"\bpython_(?:full_)?version\b", marker, re.I):
                    continue
                if re.search(
                    r"\b(?:platform_system\s*==\s*[\"\x27]Windows[\"\x27]|sys_platform\s*==\s*[\"\x27]win32[\"\x27])",
                    marker,
                    re.I,
                ):
                    continue
            name = normalize(name.group())
            if extra:
                extras.setdefault(normalize(extra.group(1)), []).append(
                    (name, req if Requirement is not None else None)
                )
            else:
                base.add(name)
        pending = list(base)
        while pending:
            name = pending.pop()
            try:
                dependencies = metadata.requires(name) or []
            except metadata.PackageNotFoundError:
                continue
            for dependency in dependencies:
                match = re.match(r"[A-Za-z0-9._-]+", dependency)
                if not match:
                    continue
                if Requirement is not None:
                    try:
                        req = Requirement(dependency)
                        if req.marker and not req.marker.evaluate({"extra": ""}):
                            continue
                    except Exception:
                        continue
                child = normalize(match.group())
                if child not in base:
                    base.add(child)
                    pending.append(child)
        selected = []
        for extra, requirements in extras.items():
            distinct = [(name, req) for name, req in requirements if name not in base]
            if (
                extra not in {"dev", "test", "all"}
                and not extra.startswith("ci-")
                and distinct
                and all(
                    name in installed
                    and (
                        req is None
                        or req.specifier.contains(installed[name], prereleases=True)
                    )
                    for name, req in distinct
                )
            ):
                selected.append(extra)
        return sorted(selected)
    except Exception:
        return []
