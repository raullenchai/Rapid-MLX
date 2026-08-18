"""Keep user-facing project surfaces branded as Rapid-MLX."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TEXT_ROOTS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs",
    REPO_ROOT / "examples",
)
OLD_BRAND = "vllm-mlx"

# Rapid-MLX derives from https://github.com/waybarrios/vllm-mlx, and section 4
# of the Apache License requires that attribution to travel with the code. The
# upstream name is therefore allowed *exactly* where that attribution lives —
# the README Acknowledgements link — and nowhere else. Do not "clean up" these
# strings to satisfy branding: dropping them puts the project outside the
# licence it received the code under. See NOTICE for the full statement.
ATTRIBUTION_EXEMPTIONS = ("[vLLM-MLX](https://github.com/waybarrios/vllm-mlx)",)
TEXT_SUFFIXES = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def _public_text_files() -> list[Path]:
    files: list[Path] = []
    for root in PUBLIC_TEXT_ROOTS:
        if root.is_file():
            files.append(root)
        else:
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
            )
    return files


def _markdown_section(text: str, header: str) -> str:
    """Return the body of the ``## <header>`` section (up to the next ``## ``)."""
    body: list[str] = []
    capturing = False
    for line in text.splitlines():
        if line.startswith("## "):
            capturing = line.strip() == header
            continue
        if capturing:
            body.append(line)
    return "\n".join(body)


def test_public_surfaces_do_not_use_old_brand() -> None:
    readme = (REPO_ROOT / "README.md").resolve()
    offenders = []
    for path in _public_text_files():
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        # The licence-required attribution lives only in the README
        # Acknowledgements, so the exemption applies there and nowhere else:
        # a copy of that exact link in any other doc/example is still an
        # offender, matching the "README — and nowhere else" rule above.
        if path.resolve() == readme:
            for exemption in ATTRIBUTION_EXEMPTIONS:
                content = content.replace(exemption, "")
        if OLD_BRAND in content.lower():
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == [], (
        "Use Rapid-MLX on user-facing surfaces; keep old names only in "
        "explicit compatibility code and the licence attribution listed in "
        f"ATTRIBUTION_EXEMPTIONS. Offenders: {offenders}"
    )


def test_upstream_attribution_survives_branding() -> None:
    """The other half of the branding rule: attribution must stay put.

    ``test_public_surfaces_do_not_use_old_brand`` pushes the old name out of
    user-facing text. Without this test that pressure eventually removes the
    upstream credit too, which is what section 4 of the Apache License requires
    us to keep. Both must hold at once.
    """
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://github.com/waybarrios/vllm-mlx" in readme, (
        "README lost the upstream attribution for the project this code "
        "derives from — see NOTICE."
    )
    # The exemption is a single Acknowledgements link. Pin it to *exactly once*
    # and *inside the Acknowledgements section* — the old brand it legitimately
    # carries can't then proliferate elsewhere in the README (each stray copy is
    # old-brand text the file-level exemption would otherwise wave through).
    ack = _markdown_section(readme, "## Acknowledgements")
    for exemption in ATTRIBUTION_EXEMPTIONS:
        assert readme.count(exemption) == 1, (
            f"expected exactly one {exemption!r} in README, "
            f"found {readme.count(exemption)}"
        )
        assert exemption in ack, (
            f"{exemption!r} must live in the README Acknowledgements section"
        )

    notice = (REPO_ROOT / "NOTICE").read_text(encoding="utf-8")
    missing = [
        upstream
        for upstream in (
            "waybarrios/vllm-mlx",
            "aigc-apps/VideoX-Fun",
            "Stability-AI/stable-audio-3",
            "mgriebling/SwiftMath",
        )
        if upstream not in notice
    ]
    assert missing == [], f"NOTICE lost required attribution for: {missing}"
