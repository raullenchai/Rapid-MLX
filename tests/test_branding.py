"""Keep user-facing project surfaces branded as Rapid-MLX."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TEXT_ROOTS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs",
    REPO_ROOT / "examples",
)
OLD_BRAND = "vllm-mlx"
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


def test_public_surfaces_do_not_use_old_brand() -> None:
    offenders = []
    for path in _public_text_files():
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if OLD_BRAND in content.lower():
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == [], (
        "Use Rapid-MLX on user-facing surfaces; keep old names only in "
        f"explicit compatibility code. Offenders: {offenders}"
    )
