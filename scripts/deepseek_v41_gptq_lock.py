"""Process locks and fail-closed reservations for offline conversion artifacts.

Lock files are permanent: unlinking them would let contenders lock different
inodes. A remote driver reserves artifacts between SSH requests; its children
still take the OS locks, so even two children of that driver cannot overlap.
An interrupted driver retains its reservation for explicit operator recovery.
After verifying all driver/child processes have stopped, use the resource paths
and owner recorded in fullrun.lock/reservation.json with reserve_artifacts(...,
release=True), then remove the marker and directory. Never unlink lock files.
"""

import fcntl
import hashlib
import os
from contextlib import ExitStack, contextmanager
from pathlib import Path

LOCK_ROOT = Path("/tmp") / f"rapid-mlx-gptq-locks-{os.getuid()}"
OWNER_ENV = "RAPID_MLX_GPTQ_LOCK_OWNER"


@contextmanager
def artifact_locks(paths, owner=None):
    """Lock canonical artifact paths for the entire operation, or fail closed."""
    LOCK_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    with ExitStack() as stack:
        files = []
        for path in sorted({str(Path(p).resolve()) for p in paths if p is not None}):
            key = hashlib.sha256(path.encode()).hexdigest()
            stream = stack.enter_context((LOCK_ROOT / key).open("a+"))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"conversion artifact is busy: {path}") from None
            stream.seek(0)
            reservation = stream.read()
            if reservation and reservation != owner:
                raise RuntimeError(f"conversion artifact is reserved: {path}")
            files.append(stream)
        yield files


def reserve_artifacts(paths, owner, *, release=False):
    """Change durable reservations only while holding every associated OS lock."""
    if not owner:
        raise ValueError("a nonempty conversion lock owner is required")
    with artifact_locks(paths, owner) as files:
        for stream in files:
            stream.seek(0)
            stream.truncate()
            stream.write("" if release else owner)
            stream.flush()
            os.fsync(stream.fileno())


def command_locks(args):
    # Include read inputs too: assembly/qualification must not race conversion.
    names = (
        "source",
        "parts",
        "scratch",
        "quality_scratch",
        "reference",
        "build",
        "staging",
        "scan",
        "destination",
        "output",
        "cache",
        "verify_cache",
    )
    paths = [getattr(args, name, None) for name in names]
    if getattr(args, "scratch", None) is not None:
        paths.append(args.scratch / "heldout")
    return artifact_locks(paths, os.environ.get(OWNER_ENV))
