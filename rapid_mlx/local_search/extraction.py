# SPDX-License-Identifier: Apache-2.0
"""Bounded extraction from explicitly chosen directories."""

from __future__ import annotations

import errno
import fnmatch
import io
import os
import posixpath
import warnings
import xml.etree.ElementTree as ET
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from .eligibility import KEYWORD_ONLY_SUFFIXES

MAX_BYTES = 20 * 1024 * 1024
MAX_EMBEDDED_CHUNKS_PER_FILE = 64
IMAGE_MAX_BYTES = 50 * 1024 * 1024
MAX_MEDIA_PIXELS = 67_108_864
VIDEO_MAX_DECODED_BYTES = 48 * 1024 * 1024
MAX_MEDIA_DIMENSION = 16_384
MAX_ARCHIVE_ENTRIES = 4096
MAX_COMPRESSION_RATIO = 1000
MAX_XML_DEPTH = 64
MAX_XML_ELEMENTS = 100_000
MAX_DOCUMENT_PAGES = 1000
MAX_DIRECTORY_DEPTH = 64
VIDEO_MAX_BYTES = 4 * 1024 * 1024 * 1024
IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".heic",
}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}
# Public status vocabulary; legacy persisted codes are translated without reads.
SKIP_REASONS = {
    "unsupported": "Unsupported file type",
    "size": "Too large",
    "empty": "No readable text",
    "scanned": "Scanned PDF without text",
    "password": "Password-protected",
    "read": "Could not be read",
    "private": "Private or system file",
    "audio": "Audio is not supported yet",
}
AUDIO_SUFFIXES = {
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".aiff",
    ".wma",
    ".opus",
}
OFFICE_SUFFIXES = {".docx", ".pptx", ".xlsx"}
LEGACY_OFFICE_SUFFIXES = {".doc", ".xls", ".ppt", ".rtf", ".pages"}
XLSX_MAX_CELLS_PER_SHEET = 10_000
EXTRACTION_VERSION = "1"


def skip_reason(reason: str) -> str:
    if reason in SKIP_REASONS.values():
        return reason
    if "size cap" in reason:
        return SKIP_REASONS["size"]
    if reason in {
        "unsupported",
        "unsupported HEIC",
        "binary",
        "unsupported XML entities",
    }:
        return SKIP_REASONS["unsupported"]
    return SKIP_REASONS["read"]


VIDEO_INTERVAL_SECONDS = 5.0
VIDEO_MAX_FRAMES = 120
FRAME_DIFFERENCE_THRESHOLD = 2.0  # Mean absolute RGB difference, on the 0..255 scale.


def file_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    return (
        "image"
        if suffix in IMAGE_SUFFIXES
        else "video"
        if suffix in VIDEO_SUFFIXES
        else "text"
    )


TEXT_SUFFIXES = set(
    [
        ".txt",
        ".md",
        ".markdown",
        ".rst",
        ".py",
        ".pyi",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".swift",
        ".m",
        ".h",
        ".c",
        ".cpp",
        ".cc",
        ".rs",
        ".go",
        ".java",
        ".kt",
        ".rb",
        ".php",
        ".sh",
        ".zsh",
        ".bash",
        ".sql",
        ".css",
        ".scss",
        ".json",
        ".jsonl",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".csv",
        ".tsv",
        ".xml",
        ".html",
        ".htm",
        ".log",
        ".vue",
        ".svelte",
        ".tex",
    ]
)
TEXT_SUFFIXES.update(KEYWORD_ONLY_SUFFIXES)

# Lower-case: matched case-insensitively, like the default macOS volume.
SKIP_DIRS = {
    "node_modules",
    "venv",
    "env",
    "__pycache__",
    "build",
    "dist",
    "target",
    "vendor",
    "deriveddata",
    "cmakefiles",
    "cache",
    "caches",
}
PACKAGE_SUFFIXES = (
    ".app",
    ".bundle",
    ".framework",
    ".photoslibrary",
    ".photolibrary",
    ".xcodeproj",
    ".xcworkspace",
    ".lproj",
    ".plugin",
    ".appex",
    ".kext",
    ".prefpane",
    ".rtfd",
    ".pages",
    ".numbers",
    ".keynote",
)
SKIP_PREFIXES = tuple(
    Path("/" + name)
    for name in (
        "System",
        "Library",
        "usr",
        "bin",
        "sbin",
        "private",
        "dev",
        "cores",
        "opt",
        "Network",
        "Volumes",
    )
)
USERS_DIRECTORY = Path("/Users")

SECRET_PATTERNS = (
    ".env*",
    "credentials",
    "secrets",
    "secret",
    "passwords",
    "password",
    "credentials.*",
    "secrets.*",
    "secret.*",
    "passwords.*",
    "password.*",
    "*.kdbx",
    "*.kdb",
    ".netrc",
    "*_rsa",
    "*_rsa.*",
    "service-account*.json",
    "service_account*.json",
    "*credentials*.json",
    "*.keystore",
    "*.pem",
    "*.key",
    "id_rsa*",
    "id_ed25519*",
    "*.keychain*",
    "*.p12",
    "*.pfx",
)


@dataclass(frozen=True)
class Unit:
    text: str
    page: int | None = None


class SkippedError(ValueError):
    """A file cannot safely be indexed (reason is suitable for persisted status)."""


class FolderAccessDeniedError(PermissionError):
    """The selected folder's own entries cannot be listed."""


def excluded_location(path: Path, root: Path) -> bool:
    """Absolute exclusions apply while walking a broader root.

    A root the user picked explicitly wins: an exclusion does not apply when
    the root itself lives inside it (an external volume, iCloud Drive under
    ``~/Library``, a scratch directory under ``/private``).
    """
    path = path.resolve()
    prefixes = [*SKIP_PREFIXES, Path.home().resolve() / "Library"]
    if path.is_relative_to(USERS_DIRECTORY):
        parts = path.relative_to(USERS_DIRECTORY).parts
        if len(parts) >= 2 and parts[1] == "Library":
            prefixes.append(USERS_DIRECTORY / parts[0] / "Library")
    return any(
        path.is_relative_to(prefix) and not root.is_relative_to(prefix)
        for prefix in prefixes
    )


def excluded(path: Path, *, directory: bool = False) -> bool:
    name = path.name
    return (
        name.startswith(".")
        or any(fnmatch.fnmatch(name.lower(), p) for p in SECRET_PATTERNS)
        or (
            directory
            and (name.lower() in SKIP_DIRS or name.lower().endswith(PACKAGE_SUFFIXES))
        )
    )


def broad_folder_root(path: Path) -> bool:
    """Reject system/container roots, while allowing explicitly selected subfolders."""
    path = path.resolve()
    roots = (
        *SKIP_PREFIXES,
        USERS_DIRECTORY,
        Path("/home"),
        Path("/tmp"),
        Path("/var"),
        Path("/etc"),
        Path("/proc"),
        Path("/sys"),
    )
    return path == Path(path.anchor) or any(path == root.resolve() for root in roots)


def validate_folder(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("path must be absolute without '..'")
    path = path.resolve(strict=True)
    if broad_folder_root(path):
        raise ValueError(
            "path is a filesystem or system root; select a specific subfolder"
        )
    if not path.is_dir():
        raise ValueError("path must be a directory")
    if excluded_location(path, path) or any(
        excluded(parent, directory=True) for parent in (path, *path.parents)
    ):
        raise ValueError("path is hidden, sensitive, or an excluded directory")
    # macOS privacy can allow stat on a folder while denying access to its entries.
    # Probe before the pyvenv.cfg stat, which also needs access to those entries.
    try:
        with os.scandir(path):
            pass
    except OSError as exc:
        if isinstance(exc, PermissionError) or exc.errno in (errno.EPERM, errno.EACCES):
            raise FolderAccessDeniedError(*exc.args) from exc
        raise
    if (path / "pyvenv.cfg").exists():
        raise ValueError("path is hidden, sensitive, or an excluded directory")
    return path


@contextmanager
def open_root(root: Path):
    """Pin every absolute component; O_NOFOLLOW on the leaf alone is unsafe."""
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("root must be absolute without '..'")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open("/", flags)
    try:
        for part in root.parts[1:]:
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


def scan_batches(root: Path, root_fd: int | None = None):
    """Discovery and reads share one pinned root, including across renames."""
    if excluded_location(root, root) or any(
        excluded(p, directory=True) for p in (root, *root.parents)
    ):
        yield [], 1
        return
    if root_fd is None:
        with open_root(root) as fd:
            yield from scan_batches(root, fd)
        return
    skipped = 0

    def onerror(exc):
        nonlocal skipped
        skipped += 1

    def walk(fd, base, depth=0):
        nonlocal skipped
        try:
            with os.scandir(fd) as entries:
                dirs, names = [], []
                for entry in entries:
                    (dirs if entry.is_dir(follow_symlinks=False) else names).append(
                        entry.name
                    )
        except OSError as exc:
            onerror(exc)
            yield base, [], [], fd
            return
        if depth >= MAX_DIRECTORY_DEPTH:
            skipped += len(dirs)
            dirs.clear()
        yield base, dirs, names, fd
        for name in dirs:
            try:
                child = os.open(
                    name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
                )
            except OSError as exc:
                onerror(exc)
                continue
            try:
                yield from walk(child, str(Path(base) / name), depth + 1)
            finally:
                os.close(child)

    for base, dirs, names, fd in walk(root_fd, "."):
        relative = Path(base)
        files: list[Path] = []
        retained_dirs = []
        for name in dirs:
            path = root / relative / name
            import stat

            try:
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                child = os.open(
                    name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
                )
                try:
                    try:
                        os.stat("pyvenv.cfg", dir_fd=child, follow_symlinks=False)
                        venv = True
                    except FileNotFoundError:
                        venv = False
                finally:
                    os.close(child)
                skip = (
                    stat.S_ISLNK(info.st_mode)
                    or excluded(path, directory=True)
                    or excluded_location(path, root)
                    or venv
                )
            except OSError:
                skip = True
            if skip:
                skipped += 1
            else:
                retained_dirs.append(name)
        dirs[:] = retained_dirs
        for name in sorted(names, key=lambda n: (file_kind(Path(n)) != "text", n)):
            import stat

            try:
                link = stat.S_ISLNK(
                    os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode
                )
            except OSError:
                link = True
            if link:
                skipped += 1
            else:
                files.append(root / relative / name)
            if len(files) >= 64:
                yield files, skipped
                files, skipped = [], 0
        yield files, skipped
        skipped = 0


def scan(root: Path) -> tuple[list[Path], int]:
    """Compatibility helper for callers that need a complete inventory."""
    files: list[Path] = []
    skipped = 0
    for batch, count in scan_batches(root):
        # Preserve this inventory helper's exclusion contract. The indexer keeps
        # excluded regular files as metadata-only skips, never reading them.
        files.extend(path for path in batch if not excluded(path))
        skipped += count + sum(excluded(path) for path in batch)
    return sorted(files), skipped


def stat_safe(path: Path, root: Path):
    """Check a name-only source without opening its content or following links."""
    import stat

    relative = path.relative_to(root)
    with open_root(root) as root_fd:
        fd = os.dup(root_fd)
        try:
            for part in relative.parts[:-1]:
                child = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
                )
                os.close(fd)
                fd = child
            info = os.stat(relative.name, dir_fd=fd, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                raise SkippedError("unsupported")
            return info
        finally:
            os.close(fd)


@contextmanager
def open_safe(path: Path, root: Path, max_bytes: int, root_fd: int | None = None):
    """Pin the regular source inode without following any path component links."""
    relative = path.relative_to(root)
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if root_fd is None:
        with (
            open_root(root) as pinned,
            open_safe(path, root, max_bytes, pinned) as source,
        ):
            yield source
        return
    fd = os.dup(root_fd)
    try:
        for part in relative.parts[:-1]:
            child = os.open(part, flags | os.O_DIRECTORY, dir_fd=fd)
            os.close(fd)
            fd = child
        child = os.open(relative.name, flags | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(child, "rb") as stream:
            import stat

            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise SkippedError("unsupported")
            if info.st_size > max_bytes:
                raise SkippedError("size cap")
            yield stream, info
            after = os.fstat(stream.fileno())
            if (info.st_size, info.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise OSError("file changed during extraction; reindex to retry")
    finally:
        os.close(fd)


def read_safe(path: Path, root: Path, max_bytes: int = MAX_BYTES, root_fd=None):
    with open_safe(path, root, max_bytes, root_fd) as (stream, info):
        data = stream.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise SkippedError("size cap")
        return data, info


@dataclass(frozen=True)
class MediaFrame:
    image: object
    timestamp: float | None = None


def sample_timestamps(frame_count: int, fps: float) -> list[float]:
    import math

    import numpy as np

    if frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise SkippedError("undecodable video duration")
    duration = frame_count / fps
    if not math.isfinite(duration):
        raise SkippedError("size cap")
    count = math.ceil(duration / VIDEO_INTERVAL_SECONDS)
    if count > VIDEO_MAX_FRAMES:
        return [
            float(t) for t in np.linspace(0, (frame_count - 1) / fps, VIDEO_MAX_FRAMES)
        ]
    return [i * VIDEO_INTERVAL_SECONDS for i in range(count)]


def media_frames(path: Path, root: Path, root_fd=None):
    """Treat decoder-specific failures as per-file skips, never folder failures."""
    try:
        yield from _decode_media_frames(path, root, root_fd)
    except (SkippedError, OSError):
        raise
    except Exception as exc:
        reason = (
            "unsupported HEIC"
            if path.suffix.lower() == ".heic"
            else "media extraction failed"
        )
        raise SkippedError(reason) from exc


def _check_media_dimensions(width, height):
    import math

    if (
        not math.isfinite(width)
        or not math.isfinite(height)
        or width <= 0
        or height <= 0
        or max(width, height) > MAX_MEDIA_DIMENSION
        or width * height > MAX_MEDIA_PIXELS
    ):
        raise SkippedError("decoded size cap")


def _decode_media_frames(path: Path, root: Path, root_fd=None):
    """Yield bounded sampled frames; the caller consumes at most four at once.

    cv2 opens the pinned descriptor, not the replaceable registered pathname.
    No full video byte buffer or full-video decode is materialised.
    """
    import numpy as np
    from PIL import Image, ImageOps

    kind = file_kind(path)
    cap = VIDEO_MAX_BYTES if kind == "video" else IMAGE_MAX_BYTES
    with open_safe(path, root, cap, root_fd) as (stream, _):
        if kind == "image":
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(stream) as image:
                        _check_media_dimensions(*image.size)
                        image.seek(0)
                        _check_media_dimensions(*image.size)
                        frame = ImageOps.exif_transpose(image).convert("RGB")
                yield MediaFrame(frame)
            except (
                Image.DecompressionBombWarning,
                Image.DecompressionBombError,
            ) as exc:
                raise SkippedError("decoded size cap") from exc
            except SkippedError:
                raise
            except (OSError, ValueError) as exc:
                reason = (
                    "unsupported HEIC"
                    if path.suffix.lower() == ".heic"
                    else "undecodable image"
                )
                raise SkippedError(reason) from exc
            return
        import cv2

        capture = cv2.VideoCapture(f"/dev/fd/{stream.fileno()}")
        try:
            if not capture.isOpened():
                raise SkippedError("undecodable video")
            _check_media_dimensions(
                capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
            )
            fps = capture.get(cv2.CAP_PROP_FPS)
            timestamps = sample_timestamps(
                int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), fps
            )
            last_retained = None
            decoded = 0
            for timestamp in timestamps:
                frame_index = round(timestamp * fps)
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                    continue
                ok, pixels = capture.read()
                if not ok:
                    continue
                if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8:
                    raise SkippedError("undecodable video")
                _check_media_dimensions(pixels.shape[1], pixels.shape[0])
                if pixels.nbytes > VIDEO_MAX_DECODED_BYTES:
                    raise SkippedError("decoded size cap")
                decoded += 1
                small = cv2.resize(
                    pixels, (32, 32), interpolation=cv2.INTER_AREA
                ).astype(np.float32)
                duplicate = (
                    last_retained is not None
                    and float(np.abs(small - last_retained).mean())
                    < FRAME_DIFFERENCE_THRESHOLD
                )
                if duplicate:
                    continue
                # Accumulate slow changes relative to the last retained sample.
                last_retained = small
                yield MediaFrame(
                    Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)),
                    frame_index / fps,
                )
            if not decoded:
                raise SkippedError("undecodable video frames")
        except cv2.error as exc:
            raise SkippedError("undecodable video") from exc
        finally:
            capture.release()


def next_media_batch(iterator, size: int = 4):
    from itertools import islice

    return list(islice(iterator, size))


class _HTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.suppressed = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.suppressed += 1
        elif tag in {"p", "div", "br", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.suppressed = max(0, self.suppressed - 1)
        elif tag in {"p", "div", "li", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.suppressed:
            self.parts.append(data)


def extract(path: Path, data: bytes, max_bytes: int = MAX_BYTES) -> list[Unit]:
    if len(data) > max_bytes:
        raise SkippedError("size cap")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise SkippedError("unsupported") from None
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise SkippedError(SKIP_REASONS["password"])
        units = []
        total = 0
        for number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            total += len(text.encode("utf-8"))
            if total > max_bytes:
                raise SkippedError("expanded size cap")
            if text.strip():
                units.append(Unit(text, number))
        # A few isolated glyphs/page numbers are not a readable text layer.
        if reader.pages and sum(c.isalnum() for u in units for c in u.text) < 3:
            raise SkippedError(SKIP_REASONS["scanned"])
        return units
    if suffix in OFFICE_SUFFIXES:
        return _extract_office(suffix, data, max_bytes)
    if suffix in AUDIO_SUFFIXES:
        raise SkippedError(SKIP_REASONS["audio"])
    if suffix not in TEXT_SUFFIXES and path.name.lower() not in {
        "makefile",
        "dockerfile",
        "license",
        "readme",
    }:
        raise SkippedError("unsupported")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise SkippedError("binary") from None
    if any(ord(c) < 32 and c not in "\n\r\t\f" for c in text):
        raise SkippedError("binary")
    if suffix in {".html", ".htm"}:
        parser = _HTML()
        parser.feed(text)
        text = "".join(parser.parts)
    if len(text.encode("utf-8")) > max_bytes:
        raise SkippedError("expanded size cap")
    return [Unit(text)]


_WORD = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_DRAWING = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_SHEET = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


class _OfficeXML:
    """Read bounded OOXML parts; never expand entities or follow external links."""

    def __init__(self, archive, max_bytes):
        self.archive = archive
        self.remaining = max_bytes
        self.cache = {}
        entries = archive.infolist()
        if (
            len(entries) > MAX_ARCHIVE_ENTRIES
            or sum(info.file_size for info in entries) > max_bytes
            or any(
                info.file_size > max(1, info.compress_size) * MAX_COMPRESSION_RATIO
                for info in entries
            )
        ):
            raise SkippedError("expanded size cap")

    def read(self, name):
        if name not in self.cache:
            with self.archive.open(name) as stream:
                xml = stream.read(self.remaining + 1)
            if len(xml) > self.remaining:
                raise SkippedError("expanded size cap")
            self.remaining -= len(xml)
            declarations = xml.replace(b"\0", b"").upper()
            if b"<!DOCTYPE" in declarations or b"<!ENTITY" in declarations:
                raise SkippedError("unsupported XML entities")
            parser = ET.XMLPullParser(events=("start", "end"))
            depth = elements = 0
            root = None
            for offset in range(0, len(xml), 4096):
                parser.feed(xml[offset : offset + 4096])
                for event, element in parser.read_events():
                    if event == "start":
                        if root is None:
                            root = element
                        depth += 1
                        elements += 1
                        if depth > MAX_XML_DEPTH or elements > MAX_XML_ELEMENTS:
                            raise SkippedError("expanded size cap")
                    else:
                        depth -= 1
            parser.close()
            self.cache[name] = root
        return self.cache[name]

    def relationships(self, part):
        directory, basename = posixpath.split(part)
        name = posixpath.join(directory, "_rels", basename + ".rels")
        if name not in self.archive.namelist():
            return {}
        return {
            rel.attrib["Id"]: (
                rel.attrib.get("Type", "").rsplit("/", 1)[-1],
                posixpath.normpath(posixpath.join(directory, rel.attrib["Target"]))
                if not rel.attrib["Target"].startswith("/")
                else rel.attrib["Target"].lstrip("/"),
            )
            for rel in self.read(name)
            if rel.attrib.get("TargetMode") != "External"
        }


def _join_bounded(parts, separator, max_bytes):
    retained = []
    total = 0
    for text in parts:
        total += len(text.encode("utf-8")) + (len(separator) if retained else 0)
        if total > max_bytes:
            raise SkippedError("expanded size cap")
        retained.append(text)
    return separator.join(retained)


def _paragraphs(root, ns, max_bytes):
    parts = []
    total = 0
    for paragraph in root.iter(ns + "p"):
        texts = []
        for element in paragraph.iter(ns + "t"):
            text = element.text or ""
            total += len(text.encode("utf-8"))
            if total > max_bytes:
                raise SkippedError("expanded size cap")
            texts.append(text)
        total += 1
        if total > max_bytes:
            raise SkippedError("expanded size cap")
        parts.append("".join(texts))
    return "\n".join(parts)


def _extract_office(suffix, data, max_bytes) -> list[Unit]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        xml = _OfficeXML(archive, max_bytes)
        if suffix == ".docx":
            names = ["word/document.xml"] + sorted(
                name
                for name in archive.namelist()
                if name.startswith("word/")
                and (
                    posixpath.basename(name).startswith(("header", "footer"))
                    and name.endswith(".xml")
                    or name
                    in {"word/footnotes.xml", "word/endnotes.xml", "word/comments.xml"}
                )
            )
            text = _join_bounded(
                (_paragraphs(xml.read(name), _WORD, max_bytes) for name in names),
                "\n",
                max_bytes,
            )
            return [Unit(text)]
        units = []
        total = 0
        if suffix == ".pptx":
            part = "ppt/presentation.xml"
            relationships = xml.relationships(part)
            ns = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
            parts = [
                relationships[e.attrib[_REL_ID]][1]
                for e in xml.read(part).iter(ns + "sldId")
            ]
            if len(parts) > MAX_DOCUMENT_PAGES:
                raise SkippedError("expanded size cap")
            for number, slide in enumerate(parts, 1):
                targets = [slide] + [
                    target
                    for kind, target in xml.relationships(slide).values()
                    if kind == "notesSlide"
                ]
                text = _join_bounded(
                    (
                        _paragraphs(xml.read(target), _DRAWING, max_bytes - total)
                        for target in targets
                    ),
                    "\n",
                    max_bytes - total,
                )
                total += len(text.encode("utf-8"))
                units.append(Unit(text, number))
        else:
            shared = []
            if "xl/sharedStrings.xml" in archive.namelist():
                shared = [
                    "".join(t.text or "" for t in si.iter(_SHEET + "t"))
                    for si in xml.read("xl/sharedStrings.xml").iter(_SHEET + "si")
                ]
            part = "xl/workbook.xml"
            relationships = xml.relationships(part)
            for number, sheet in enumerate(xml.read(part).iter(_SHEET + "sheet"), 1):
                if number > MAX_DOCUMENT_PAGES:
                    raise SkippedError("expanded size cap")
                target = relationships[sheet.attrib[_REL_ID]][1]
                cells = []
                for index, cell in enumerate(xml.read(target).iter(_SHEET + "c")):
                    if index >= XLSX_MAX_CELLS_PER_SHEET:
                        break
                    value = cell.findtext(_SHEET + "v", "")
                    if cell.attrib.get("t") == "s":
                        value = shared[int(value)] if value else ""
                    elif cell.attrib.get("t") == "inlineStr":
                        value = "".join(t.text or "" for t in cell.iter(_SHEET + "t"))
                    total += len(value.encode("utf-8")) + 1
                    if total > max_bytes:
                        raise SkippedError("expanded size cap")
                    cells.append(value)
                units.append(Unit("\n".join(cells), number))
        return units
