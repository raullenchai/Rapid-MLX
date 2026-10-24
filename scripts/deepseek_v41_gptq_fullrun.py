#!/usr/bin/env python3
"""Studio driver for checkpoint-gated M5 GPTQ. No MLX import or work at import time."""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

HEADROOM = 20_000_000_000
# Downloads need their payload size plus room for metadata/logs and transient
# local writes, not the remote worker's 20 GB model-execution reserve.
LOCAL_EXPORT_RESERVE = 1_000_000_000
METADATA_TIMEOUT = 3600
COMMAND_TIMEOUT = 86400
TRANSFER_TIMEOUT = 86400
SSH_OPTIONS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=15",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ServerAliveCountMax=3",
]
SCRATCH_ROOT = (
    Path(
        os.environ.get(
            "DSV41_SCRATCH_ROOT",
            str(Path(tempfile.gettempdir()) / "rapid-mlx-dsv41-1017"),
        )
    )
    / "fullrun-driver"
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolved_recipe(path):
    # Read the reviewed CLI defaults as literal data, without importing MLX.
    tree = ast.parse(Path(__file__).with_name("deepseek_v41_gptq.py").read_text())
    defaults = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "DEFAULTS" for t in node.targets)
    )
    result = {kw.arg: ast.literal_eval(kw.value) for kw in defaults.keywords}
    supplied = json.loads(Path(path).read_text())
    if set(supplied) - set(result):
        raise ValueError("unknown recipe keys")
    result.update(supplied)
    return result


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def safe_name(name):
    if (
        not isinstance(name, str)
        or not name
        or Path(name).is_absolute()
        or ".." in Path(name).parts
    ):
        raise ValueError("unsafe manifest path")
    return name


def lifecycle(manifest):
    """Map ALL tensors to layers; any non-layer use pins a shard permanently."""
    n = manifest["config"].get("text_config", manifest["config"])["num_hidden_layers"]
    result = {
        safe_name(s): {"layers": set(), "pinned": False} for s in manifest["shards"]
    }
    if not result or type(n) is not int or n < 1:
        raise ValueError("empty or invalid manifest")
    for name, tensor in manifest["tensors"].items():
        path = safe_name(tensor["path"])
        if path not in result:
            raise ValueError("tensor references unknown shard")
        match = re.match(r"^layers\.(\d+)\.", name)
        if match:
            layer = int(match[1])
            if layer >= n:
                raise ValueError("out of range layer")
            result[path]["layers"].add(layer)
        else:
            if name.startswith("layers."):
                raise ValueError("unrecognized layer tensor")
            result[path]["pinned"] = True
    if any(not r["layers"] and not r["pinned"] for r in result.values()):
        raise ValueError("unmapped shard; refusing cleanup")
    if set.union(*(r["layers"] for r in result.values())) != set(range(n)):
        raise ValueError("manifest must cover every layer")
    for meta in manifest["shards"].values():
        if type(meta["size"]) is not int or meta["size"] <= 0:
            raise ValueError("invalid shard size")
    return result


def needed(plan, layer, count):
    return sorted(
        s
        for s, r in plan.items()
        if r["pinned"] or r["layers"] & {layer, min(layer + 1, count - 1)}
    )


def deletable(plan, layer, checkpoint, identity):
    """A drift row or completion file alone NEVER authorizes deletion."""
    validate_checkpoint(checkpoint, identity)
    if checkpoint["next_layer"] <= layer:
        raise ValueError("layer has not committed")
    return sorted(
        s for s, r in plan.items() if not r["pinned"] and max(r["layers"]) <= layer
    )


def validate_checkpoint(checkpoint, identity, count=40):
    if (
        checkpoint["identity"] != identity
        or type(checkpoint["next_layer"]) is not int
        or not 0 <= checkpoint["next_layer"] <= count
    ):
        raise ValueError("invalid checkpoint/run identity")


def sync_parent(path):
    descriptor = os.open(Path(path).parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w") as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    sync_parent(path)


# Requests travel on stdin, never by interpolating code/data into a shell command.
# No commands print environment variables, credentials, or arbitrary subprocess output.
def tool_manifest(root):
    root = Path(root)
    return {
        str(p.relative_to(root)): sha(p)
        for directory in ("scripts", "rapid_mlx")
        for p in sorted((root / directory).rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }


REMOTE = r"""
import hashlib,json,os,signal,struct,subprocess,sys,time
from pathlib import Path
q=json.load(sys.stdin)
def hashfile(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''): h.update(b)
 return h.hexdigest()
def info(p,meta=None):
 if not p.exists():return None
 if p.is_symlink() or not p.is_file():raise ValueError('expected regular file')
 out={'size':p.stat().st_size}
 if meta is None or 'sha256' in meta:out['sha256']=hashfile(p)
 if meta and 'header_sha256' in meta:
  with p.open('rb') as f:
   raw=f.read(8);n=struct.unpack('<Q',raw)[0]
   if n>100000000 or n+8>out['size']:raise ValueError('invalid header size')
   out['header_sha256']=hashlib.sha256(raw+f.read(n)).hexdigest()
 return out
op=q['op']
if op=='json':
 p=Path(q['path']);out=json.loads(p.read_text()) if p.exists() else None
elif op=='tools':
 root=Path(q['path'])
 out={str(p.relative_to(root)):hashfile(p) for directory in ('scripts','rapid_mlx') for p in sorted((root/directory).rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
elif op=='info':out=info(Path(q['path']),q.get('meta'))
elif op=='free':
 s=os.statvfs(q['path']);out=s.f_bavail*s.f_frsize
elif op=='mkdir':
 Path(q['path']).mkdir(parents=True,exist_ok=True);out=True
elif op in ('lock','unlock'):
 sys.path.insert(0,q['repo'])
 from scripts.deepseek_v41_gptq_lock import reserve_artifacts
 p=Path(q['path'])
 if op=='lock':
  p.mkdir()
  try:
   with (p/'reservation.json').open('w') as marker:
    json.dump({'resources':q['resources'],'owner':q['owner']},marker);marker.flush();os.fsync(marker.fileno())
   reserve_artifacts(q['resources'],q['owner'])
  except BaseException:
   (p/'reservation.json').unlink(missing_ok=True)
   p.rmdir()
   raise
 else:
  reserve_artifacts(q['resources'],q['owner'],release=True)
  (p/'reservation.json').unlink()
  p.rmdir()
 out=True
elif op=='delete':
 # Recheck the authoritative checkpoint immediately before EVERY unlink.
 cp=json.loads(Path(q['checkpoint']).read_text())
 if cp['identity']!=q['identity'] or cp['next_layer']<=q['layer']:raise ValueError('uncommitted cleanup')
 p=Path(q['path']);root=Path(q['root']).resolve();resolved=p.resolve()
 if resolved==root or not resolved.is_relative_to(root):raise ValueError('cleanup path outside authorized root')
 if p.is_symlink():raise ValueError('refuse symlink cleanup')
 if p.exists() and not p.is_file():raise ValueError('cleanup requires a regular file')
 p.unlink(missing_ok=True);out=True
elif op=='recover_init':
 root=Path(q['path'])
 if (root/'state/checkpoint.json').exists():raise ValueError('initialization already committed')
 tag=str(time.time_ns());out=[]
 for name in ('parts','state'):
  p=root/name
  if p.exists():
   target=root/('abandoned-init-'+tag+'-'+name)
   p.rename(target);out.append(target.name)
elif op=='run':
 with Path(q['log']).open('ab') as f:
  process=subprocess.Popen(q['argv'],cwd=q['cwd'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,**q.get('env',{})})
  timed_out=False
  try:code=process.wait(timeout=q.get('command_timeout',86400))
  except subprocess.TimeoutExpired:
   try:os.killpg(process.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   try:process.wait(timeout=10)
   except subprocess.TimeoutExpired:
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    process.wait()
   timed_out=True
 out={'command_finished':True,'returncode':process.returncode,'timed_out':timed_out}
else:raise ValueError('unknown operation')
print(json.dumps(out))
"""


class Remote:
    def __init__(self, host, python):
        if host.startswith("-") or not re.fullmatch(r"[\w.@-]+", host):
            raise ValueError("invalid SSH host")
        self.host, self.python = host, python
        self.indeterminate = False

    def call(self, op, **kw):
        request = dict(op=op, **kw)
        timeout = METADATA_TIMEOUT
        if op == "run":
            self.indeterminate = True
            request["command_timeout"] = COMMAND_TIMEOUT
            timeout = COMMAND_TIMEOUT + 30
        try:
            result = subprocess.run(
                [
                    "ssh",
                    *SSH_OPTIONS,
                    self.host,
                    shlex.join([self.python, "-c", REMOTE]),
                ],
                input=json.dumps(request),
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"remote {op} timed out; inspect remote state before retry; no cleanup authorized"
            ) from None
        if result.returncode:
            raise RuntimeError(
                f"remote {op} failed (exit {result.returncode}); no cleanup authorized"
            )
        value = json.loads(result.stdout)
        if op == "run":
            if (
                not isinstance(value, dict)
                or value.get("command_finished") is not True
                or type(value.get("returncode")) is not int
                or type(value.get("timed_out")) is not bool
            ):
                raise RuntimeError(
                    "remote command completion is unverified; retain lock"
                )
            self.indeterminate = False
            if value["returncode"] or value["timed_out"]:
                raise RuntimeError(
                    "remote command finished unsuccessfully; inspect step log before retry"
                )
            return True
        return value

    def transfer(self, source, target, upload):
        # -L dereferences the HF snapshot's blob symlinks. No --delete.
        argv = (
            ["rsync", "-aLI", "--", str(source), f"{self.host}:{shlex.quote(target)}"]
            if upload
            else [
                "rsync",
                "-aI",
                "--",
                f"{self.host}:{shlex.quote(str(source))}",
                str(target),
            ]
        )
        argv[1:1] = ["-e", shlex.join(["ssh", *SSH_OPTIONS])]
        try:
            p = subprocess.run(argv, capture_output=True, timeout=TRANSFER_TIMEOUT)
        except subprocess.TimeoutExpired:
            raise RuntimeError("rsync timed out; no cleanup authorized") from None
        if p.returncode:
            raise RuntimeError("rsync failed; no cleanup authorized")


class Driver:
    def __init__(self, args, remote):
        self.a, self.r = args, remote
        self.manifest = json.loads(args.manifest.read_text())
        self.plan = lifecycle(self.manifest)
        if (
            self.manifest["config"].get("text_config", self.manifest["config"])[
                "num_hidden_layers"
            ]
            != 40
        ):
            raise ValueError("fullrun requires the 40-layer source")
        self.recipe = resolved_recipe(args.recipe)
        if (
            self.recipe.get("pilot") is not False
            or len(set(self.recipe.get("dense_8bit_layers", []))) != 8
            or any(
                type(i) is not int or not 0 <= i < 40
                for i in self.recipe["dense_8bit_layers"]
            )
        ):
            raise ValueError(
                "provide a full selected recipe with pilot:false and eight dense layers"
            )
        self.root = args.remote_work + "/fullrun"
        self.stage = args.remote_work + "/stage"
        self.parts = self.root + "/parts"
        self.state = self.root + "/state"
        self.reference = self.root + "/reference"
        self.identity = None
        self.lock_owner = uuid.uuid4().hex

    def log(self, step, **kw):
        print(json.dumps(dict(time=time.time(), step=step, **kw)), flush=True)

    def room(self, size):
        available = self.r.call("free", path=self.a.remote_work)
        self.log(
            "disk_checked", available=available, incoming_bytes=size, headroom=HEADROOM
        )
        if available < size + HEADROOM:
            raise ValueError(
                f"insufficient M5 disk: need {size + HEADROOM}, available {available}"
            )

    def upload(self, source, target, meta=None):
        source = Path(source)
        size = source.stat().st_size
        meta = meta or dict(size=size, sha256=sha(source))
        if size != meta["size"]:
            raise ValueError("local source size mismatch")
        if "sha256" in meta and sha(source) != meta["sha256"]:
            raise ValueError("local source hash mismatch")
        got = self.r.call("info", path=target, meta=meta)
        if got and all(
            got.get(k) == v
            for k, v in meta.items()
            if k in ("size", "sha256", "header_sha256")
        ):
            self.log("present", file=Path(target).name)
            return
        self.room(size)
        self.r.call("mkdir", path=str(Path(target).parent))
        self.r.transfer(source, target, True)
        got = self.r.call("info", path=target, meta=meta)
        if not got or any(
            got.get(k) != v
            for k, v in meta.items()
            if k in ("size", "sha256", "header_sha256")
        ):
            raise ValueError("transferred source verification failed")
        self.log("upload_verified", file=Path(target).name, bytes=size)

    def stage_reference(self, layer):
        self.upload(
            self.a.reference_source / "reference.json",
            self.reference + "/reference.json",
        )
        for i in sorted({0, layer, layer + 1}):
            root = self.a.reference_source / f"boundary-{i}"
            record = json.loads((root / "complete.json").read_text())
            if record["identity"] != self.ref_identity:
                raise ValueError("reference boundary identity mismatch")
            for name, h in record["files"].items():
                safe_name(name)
                src = root / name
                self.upload(
                    src,
                    self.reference + f"/boundary-{i}/" + name,
                    dict(size=src.stat().st_size, sha256=h),
                )
            self.upload(
                root / "complete.json", self.reference + f"/boundary-{i}/complete.json"
            )

    def checkpoint(self):
        cp = self.r.call("json", path=self.state + "/checkpoint.json")
        validate_checkpoint(cp, self.identity)
        return cp

    def command(self, command, *extra):
        argv = [
            self.a.remote_python,
            self.a.remote_work + "/repo/scripts/deepseek_v41_gptq.py",
            command,
            "--source",
            self.stage,
            "--manifest",
            self.root + "/source-manifest.json",
            "--recipe",
            self.root + "/recipe.json",
            "--parts",
            self.parts,
            "--scratch",
            self.state,
            "--reference",
            self.reference,
            *extra,
        ]
        self.log(
            "command_start",
            command=command,
            layer=extra[-1] if command == "run" else None,
        )
        self.r.call(
            "run",
            argv=argv,
            cwd=self.a.remote_work + "/repo",
            log=self.root + "/commands.log",
            env={"RAPID_MLX_GPTQ_LOCK_OWNER": self.lock_owner},
        )
        self.log("command_exit", command=command)

    def export(self, layer):
        cp = self.checkpoint()
        if cp["next_layer"] <= layer:
            raise ValueError("cannot export uncommitted layer")
        record_name = f"layer-{layer}-complete.json"
        record = self.r.call("json", path=self.parts + "/" + record_name)
        from scripts.deepseek_v41_gptq_amend import validate_record

        validate_record(self.run_metadata, layer, record)
        required = {f"model-layer-{layer}.safetensors", f"layer-{layer}-stats.json"}
        if set(record["files"]) != required:
            raise ValueError("unexpected completion files")
        for name, h in record["files"].items():
            target = self.a.destination / name
            if not target.exists() or sha(target) != h:
                remote = self.r.call("info", path=self.parts + "/" + name)
                if remote["sha256"] != h:
                    raise ValueError("remote part hash mismatch")
                tmp = target.with_suffix(target.suffix + ".partial")
                if (
                    shutil.disk_usage(self.a.destination).free
                    < remote["size"] + LOCAL_EXPORT_RESERVE
                ):
                    raise ValueError("insufficient local export space")
                self.r.transfer(self.parts + "/" + name, tmp, False)
                if sha(tmp) != h:
                    raise ValueError("export hash mismatch")
                with tmp.open("rb") as stream:
                    os.fsync(stream.fileno())
                tmp.replace(target)
            with target.open("rb") as stream:
                os.fsync(stream.fileno())
            sync_parent(target)
            self.log("export_verified", layer=layer, file=name, sha256=h)
        # Snapshot the JSONL to a temporary path: later appends cannot affect exports.
        with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT, prefix="gptq-drift-") as tmp:
            path = Path(tmp) / "heldout-drift.jsonl"
            drift_info = self.r.call("info", path=self.parts + "/heldout-drift.jsonl")
            if (
                not drift_info
                or shutil.disk_usage(path.parent).free
                < drift_info["size"] + LOCAL_EXPORT_RESERVE
            ):
                raise ValueError(
                    "insufficient local drift export space or missing drift"
                )
            self.r.transfer(self.parts + "/heldout-drift.jsonl", path, False)
            if not drift_info or sha(path) != drift_info["sha256"]:
                raise ValueError("drift export hash mismatch")
            rows = [
                json.loads(line)
                for line in path.read_text().splitlines()
                if line.strip()
            ]
        rows = [row for row in rows if row["layer"] == layer]
        if (
            len(rows) != 1
            or rows[0].get("candidate_identity")
            != digest(dict(kind="gptq_parts", run=record["run_identity"]))
            or rows[0].get("reference_identity") != self.ref_identity
        ):
            raise ValueError("expected exactly one committed heldout drift row")
        atomic(self.a.destination / f"layer-{layer}-heldout-drift.json", rows[0])
        atomic(self.a.destination / record_name, record)
        atomic(
            self.a.destination / f"layer-{layer}-export.json",
            dict(
                identity=self.identity,
                layer=layer,
                files={
                    n: sha(self.a.destination / n)
                    for n in (
                        *sorted(required),
                        record_name,
                        f"layer-{layer}-heldout-drift.json",
                    )
                },
            ),
        )
        self.log("export_committed", layer=layer)

    def cleanup(self, layer):
        cp = self.checkpoint()
        for name in deletable(self.plan, layer, cp, self.identity):
            self.r.call(
                "delete",
                path=self.stage + "/" + name,
                root=self.stage,
                checkpoint=self.state + "/checkpoint.json",
                identity=self.identity,
                layer=layer,
            )
            self.log("source_released", layer=layer, file=name)

    def collect_metadata(self):
        for name in (
            "run.json",
            "source-manifest.json",
            "heldout-reference.json",
            "quality-config.json",
            "engram_token_map.json",
        ):
            info = self.r.call("info", path=self.parts + "/" + name)
            if info:
                target = self.a.destination / name
                temporary = target.with_suffix(target.suffix + ".partial")
                if (
                    shutil.disk_usage(self.a.destination).free
                    < info["size"] + LOCAL_EXPORT_RESERVE
                ):
                    raise ValueError("insufficient local metadata export space")
                self.r.transfer(self.parts + "/" + name, temporary, False)
                if sha(temporary) != info["sha256"]:
                    raise ValueError("metadata hash mismatch")
                with temporary.open("rb") as stream:
                    os.fsync(stream.fileno())
                temporary.replace(target)
                sync_parent(target)
                self.log("metadata_verified", file=name, sha256=info["sha256"])

    def cleanup_reference(self, layer):
        # After commit/export, retain zero and the next layer's input boundary.
        for i in range(1, layer + 1):
            root = self.a.reference_source / f"boundary-{i}"
            record = json.loads((root / "complete.json").read_text())
            for name in [*record["files"], "complete.json"]:
                safe_name(name)
                self.r.call(
                    "delete",
                    path=self.reference + f"/boundary-{i}/" + name,
                    root=self.reference,
                    checkpoint=self.state + "/checkpoint.json",
                    identity=self.identity,
                    layer=layer,
                )
        self.log("old_references_released", through_layer=layer)

    def run(self):
        # Preflight ALL source descriptors, reference commits and immutable inputs
        # before creating any remote state or invoking conversion.
        self.a.destination.mkdir(parents=True, exist_ok=True)
        SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
        reference = json.loads((self.a.reference_source / "reference.json").read_text())
        self.ref_identity = reference["identity"]
        if reference["source_identity"] != digest(self.manifest):
            raise ValueError("reference/source mismatch")
        for i in range(41):
            record = json.loads(
                (self.a.reference_source / f"boundary-{i}/complete.json").read_text()
            )
            if record["identity"] != self.ref_identity:
                raise ValueError("missing/mixed original reference boundaries")
        for name, meta in self.manifest["shards"].items():
            if (self.a.source / name).stat().st_size != meta["size"]:
                raise ValueError("source shard absent or size mismatch")
        from scripts.deepseek_v41_public_ruler import read_slice_manifest

        calibration_bytes, _, slices = read_slice_manifest(self.a.calibration)
        if any(re.fullmatch(r"[A-Za-z0-9_-]+", name) is None for name in slices):
            raise ValueError("calibration slice names must be safe filename components")
        lock_request = dict(
            path=self.a.remote_work + "/fullrun.lock",
            repo=self.a.remote_work + "/repo",
            resources=[self.stage, self.parts, self.state, self.reference],
            owner=self.lock_owner,
        )
        self.r.call("lock", **lock_request)
        try:
            self.r.call("mkdir", path=self.root)
            binding = dict(
                source=digest(self.manifest),
                recipe=self.recipe,
                reference=self.ref_identity,
                calibration=hashlib.sha256(calibration_bytes).hexdigest(),
                calibration_files={
                    str(path): sha(path) for files in slices.values() for path in files
                },
                tool=tool_manifest(Path(__file__).parent.parent),
            )
            prior = self.r.call("json", path=self.root + "/driver-inputs.json")
            local_prior = self.a.destination / "driver-inputs.json"
            if not local_prior.exists() and any(
                p.name != "fullrun.lock" for p in self.a.destination.iterdir()
            ):
                raise ValueError(
                    "unbound nonempty destination; use a fresh parts directory"
                )
            if (
                prior is None
                and self.r.call("json", path=self.parts + "/run.json") is not None
            ):
                raise ValueError(
                    "unbound existing remote run; use fresh fullrun directories"
                )
            if (prior and prior != binding) or (
                local_prior.exists() and json.loads(local_prior.read_text()) != binding
            ):
                raise ValueError("driver inputs changed; use fresh run directories")
            with tempfile.TemporaryDirectory(
                dir=SCRATCH_ROOT, prefix="gptq-inputs-"
            ) as tmp:
                path = Path(tmp) / "driver-inputs.json"
                atomic(path, binding)
                self.upload(path, self.root + "/driver-inputs.json")
            atomic(local_prior, binding)
            self.upload(self.a.manifest, self.root + "/source-manifest.json")
            self.upload(self.a.recipe, self.root + "/recipe.json")
            # Require reviewed tooling to have been synced separately; never mutate a pilot's repo.
            if (
                self.r.call("tools", path=self.a.remote_work + "/repo")
                != binding["tool"]
            ):
                raise ValueError(
                    "remote tooling differs; sync reviewed source tree before launch"
                )
            for name in ("config.json", "tokenizer.json"):
                self.upload(self.a.source / name, self.stage + "/" + name)
            cp = self.r.call("json", path=self.state + "/checkpoint.json")
            if cp is None:
                archived = self.r.call("recover_init", path=self.root)
                if archived:
                    self.log(
                        "incomplete_initialization_preserved", directories=archived
                    )
                for name in needed(self.plan, 0, 40):
                    self.upload(
                        self.a.source / name,
                        self.stage + "/" + name,
                        self.manifest["shards"][name],
                    )
                self.stage_reference(0)
                portable = {"slices": {}}
                for name, files in slices.items():
                    portable["slices"][name] = []
                    for i, src in enumerate(files):
                        rel = f"text/{name}-{i}.txt"
                        self.upload(
                            src,
                            self.root + "/corpus/" + rel,
                            dict(
                                size=src.stat().st_size,
                                sha256=binding["calibration_files"][str(src)],
                            ),
                        )
                        portable["slices"][name].append(rel)
                with tempfile.TemporaryDirectory(
                    dir=SCRATCH_ROOT, prefix="gptq-corpus-"
                ) as tmp:
                    path = Path(tmp) / "calibration.json"
                    atomic(path, portable)
                    self.upload(path, self.root + "/corpus/calibration.json")
                self.command(
                    "init",
                    "--slices",
                    self.root + "/corpus/calibration.json",
                    "--tokens",
                    "40960",
                )
            run = self.r.call("json", path=self.parts + "/run.json")
            if (
                run["recipe"] != self.recipe
                or run["source_identity"] != digest(self.manifest)
                or run["heldout_reference_identity"] != self.ref_identity
            ):
                raise ValueError(
                    "initialized run does not match full recipe/source/reference"
                )
            from scripts.deepseek_v41_gptq_amend import epochs

            epochs(run)
            self.run_metadata = run
            self.identity = run["identity"]
            self.collect_metadata()
            # Recover export interrupted after checkpoint commit BEFORE any cleanup.
            cp = self.checkpoint()
            for layer in range(cp["next_layer"]):
                self.export(layer)
            if cp["next_layer"]:
                self.cleanup(cp["next_layer"] - 1)
                self.cleanup_reference(cp["next_layer"] - 1)
            for layer in range(cp["next_layer"], 40):
                for name in needed(self.plan, layer, 40):
                    self.upload(
                        self.a.source / name,
                        self.stage + "/" + name,
                        self.manifest["shards"][name],
                    )
                self.stage_reference(layer)
                self.command("run", "--layer", str(layer))
                if self.checkpoint()["next_layer"] != layer + 1:
                    raise ValueError("run failed to commit exactly one layer")
                self.export(layer)
                self.cleanup(layer)
                self.cleanup_reference(layer)
            self.collect_metadata()
            drift = [
                json.loads(
                    (self.a.destination / f"layer-{i}-heldout-drift.json").read_text()
                )
                for i in range(40)
            ]
            drift_path = self.a.destination / "heldout-drift.jsonl"
            temporary = drift_path.with_suffix(".jsonl.partial")
            with temporary.open("w") as stream:
                for row in drift:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(drift_path)
            atomic(
                self.a.destination / "fullrun-complete.json",
                dict(identity=self.identity, next_layer=40),
            )
            self.log("complete", next_layer=40)
        finally:
            if getattr(self.r, "indeterminate", False):
                self.log(
                    "remote_lock_retained",
                    reason="command status unknown; verify remote process termination before removing the lock and retrying",
                )
            else:
                active_error = sys.exc_info()[1]
                try:
                    self.r.call("unlock", **lock_request)
                except Exception as error:
                    if active_error is None:
                        raise
                    active_error.add_note(f"Remote unlock also failed: {error}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", required=True)
    p.add_argument("--remote-work", required=True)
    p.add_argument(
        "--remote-python",
        required=True,
    )
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--reference-source", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    a = p.parse_args()
    if not Path(a.remote_work).is_absolute() or ".." in Path(a.remote_work).parts:
        p.error("remote-work must be an absolute workspace path")
    try:
        a.destination.mkdir(parents=True, exist_ok=True)
        from scripts.deepseek_v41_gptq_lock import artifact_locks

        with (
            artifact_locks([a.destination]),
            (a.destination / "fullrun.lock").open("a") as lock,
        ):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            Driver(a, Remote(a.host, a.remote_python)).run()
    except Exception as e:
        # Print a controlled diagnostic, never SSH/rsync output or environment.
        print(
            json.dumps(dict(step="failed", error=type(e).__name__, detail=str(e))),
            flush=True,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
