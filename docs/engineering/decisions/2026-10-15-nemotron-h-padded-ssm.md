# Nemotron-H right-padded recurrent prefill

Owner: Vector. Branch: `vector/nemotron-h-padded-ssm-state`.

## Failure and affected path

The text model loader installs a compatibility shim for the shared Nemotron-H
SSM path. In the installed dependency version 0.31.3, the mixer reads
`ArraysCache.lengths` but omits it from `ssm_update`. The shared SSD recurrence
also advances incoming state through padded positions, preserves exhausted
rows only for negative remaining lengths, and caps its state-gather position
by the configured step instead of the current chunk size.

The batched engine's text scheduler uses `PromptProcessingBatch`, whose
`prompt` method right-pads unequal prompts, calls cache `prepare(lengths=...)`,
processes chunks, and finalizes the caches. This reaches the defective path
for both fresh prompts and restored recurrent prefix caches. Prefix snapshots
can consequently retain incorrect state; restoring a correct snapshot does
not protect a subsequent padded suffix from this defect. Unpadded prefill and
ordinary single-token decode do not need padding corrections.

## Correction and boundaries

`rapid_mlx.patches.nemotron_h_ssm` installs idempotently from the canonical text
loader, following the existing model compatibility patches. It replaces the
shared recurrence in its defining module so previously imported dispatcher
aliases also see the correction. It forwards Nemotron-H cache lengths, derives
a valid-token mask when only lengths are provided, masks incoming-state decay,
preserves rows with zero or negative remaining lengths, and bounds state
gathering by each chunk's actual size. Masked single-token prefill uses this
recurrence because the decode kernel has no padding interface.

The existing timestep preprocessing, transition dtype, state accumulation,
residual connection, model parameters, convolution implementation, cache
lifecycle, and unmasked Metal decode dispatcher remain authoritative. No
site-packages files are modified. Remove the shim once the dependency ships
both the shared recurrence corrections and the Nemotron-H length forwarding.
Installation is restricted to dependency version 0.31.3 and the reviewed
recurrence, dispatcher and mixer calling conventions. Unknown versions,
missing or opaque symbols, changed parameter kinds/defaults, and missing
package metadata produce a warning and leave runtime functions unchanged.
A process-local lock serializes installation; the loader installs before
model construction, and repeated calls preserve function identity.

The opt-in Super VL text adapter (`rapid_mlx_nemotron_h: true`) subclasses the
vendored model, whose mixer overrides `_ssm` and already forwards lengths to
its own padding-aware recurrence. It does **not** use this patched mixer.
Neither the adapter nor the vendored Bolt/Puzzle implementation is modified.
The new checked-loader regression verifies both flagged and unflagged native
`nemotron_h` artifacts with random tiny weights, real prompt batching, and
subsequent decode. The flag selects the adapter; unflagged models use this
shim. The separate vision runtime remains outside scope.

## Regression contract

`tests/test_nemotron_h_ssm.py` is included in the Apple CI roster. A tiny
random-weight hybrid model (16 hidden channels, two SSM heads, 32 state
channels, one Mamba and one attention layer) compares valid prefill logits,
convolution and SSM states, and subsequent decode logits against independent
requests. It covers lengths 0, 255, 256, 257, 300 and 333; fresh and incoming
prefix states; prefill chunks of 1, 256 and 333; and the real prompt batch path.
An independent token recurrence additionally checks grouped heads, no explicit
mask, negative lengths, and lengths extending beyond a partial final chunk.
Left padding, unpadded chunking, loader installation, and ordinary decode
dispatch have controls. Zero-token rows retain incoming state exactly.

All eight model/prompt-batch cases fail against the unchanged dependency in
the original CPU reproduction. The prior CPU compatibility set passed 63 tests.
GPU validation on Apple M3 Ultra (256 GB), Python 3.12.14, MLX 0.32.4 and model
dependency 0.31.3 passed all 23 focused tests with default settings and all 48
focused/Apple-CI-roster/compatibility tests with `MLX_ENABLE_TF32=0`. These runs
execute Metal decode in the numerical comparisons; only the explicit dispatcher
contract uses a kernel stub.

## Cached-checkpoint GPU evidence

Checkpoint: `mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit`, cached
snapshot `55ac8c89261109b36c04371cd3f479a4594208c8`. No download or checkpoint
modification was needed. `scripts/validate_nemotron_h_padding.py` loads through
the canonical loader and uses the production prompt-processing batch. Two
prompts have 5 and 308 tokens. Their prefixes (4 and 307 tokens) are processed
with a 256-token prefill step; the final prompt token exposes next-token logits,
followed by eight greedy tokens. Independent runs use the same chunk size.

The numerical control promotes floating parameters/activations to float32,
retains packed 4-bit weights, and disables TF32. Base: `origin/main` at
`ed77e0b33`; candidate implementation: `bc56335f2` plus the validation script.

| Revision | Prompt tokens | First next-token max absolute logit error | Worst error over eight steps | Eight greedy tokens match |
| --- | ---: | ---: | ---: | --- |
| Base | 5 | 7.983969 | 18.153801 | No |
| Candidate | 5 | 0.0000133514 | 0.0000181198 | Yes |
| Base | 308 | 0.00196362 | 0.00196362 | Yes |
| Candidate | 308 | 0.00196362 | 0.00196362 | Yes |

The untouched long row supplies a numerical control: its residual error is
identical on both revisions. The float32 comparison uses `atol=0.003,
rtol=0.0002` for every logit at every step; the base fails and the candidate
passes. The initial tighter `atol=0.0002` bound failed only on this unchanged
long control after the fix. The short-row correction is much tighter than the
final allowance, with peak error below 0.000019.

At original checkpoint precision, the candidate also matches both eight-token
greedy continuations, while the base diverges on the short row. Logits are not
bitwise equivalent: first-step max errors are 1.578125 for the candidate short
row and 0.8125 for the long row, with the long-row errors again identical on
both revisions. The checkpoint-precision diagnostic fails its logit bound on
both revisions. This evidence verifies recurrent padding correctness, not
bitwise batch invariance at checkpoint precision or a quality/performance claim.

Reproduce on a host with this cached snapshot (all GPU commands require the
shared lock):

```sh
mkdir -p /private/tmp/nemotron-ssm-fix-validation
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MLX_ENABLE_TF32=0
RAPID_MODEL="$HOME/.cache/huggingface/hub/models--mlx-community--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit/snapshots/55ac8c89261109b36c04371cd3f479a4594208c8"
lockf -k /var/tmp/rapid-mlx-large-model.lock .venv/bin/python -m pytest tests/test_nemotron_h_ssm.py tests/test_ci_apple_coverage_union.py tests/test_mlx_compat.py -q
lockf -k /var/tmp/rapid-mlx-large-model.lock .venv/bin/python scripts/validate_nemotron_h_padding.py --float32 --model "$RAPID_MODEL" --output /private/tmp/nemotron-ssm-fix-validation/branch.json
```

For the negative control, run the same candidate script and Python environment
with `PYTHONPATH` pointing at a detached base checkout. Base exits 1 after
writing its JSON evidence; candidate exits 0. The script reports raw errors,
allclose decisions, token IDs and decoded continuations before asserting.
Changed-line coverage against `origin/main` is 135/135 executable lines (100%),
including the compatibility shim, loader installation and validation script.


## HTTP control after the 0.16.2 rebase (2026-10-16)

The normal rebase onto `origin/main` at
`a1b3b1d21323540eee59d7d610b802afd4d97ae6` completed without conflicts.
Environment: Apple M3 Ultra, 256 GB; Python 3.12.14, MLX 0.32.4,
model dependency 0.31.3. All model execution holds the shared `lockf` lock
and the Studio lease owned by `rapid-mlx/vector-ssm-fix`. No downloads.

HTTP requests use `/v1/completions`, temperature 0, max_tokens 64, non-streaming,
and the same 5/308-token prompts as the direct numerical control. Each prompt
runs alone first; then a barrier releases two client threads together. Compare
complete returned text, including early EOS. Prefix, response and disk caches
are disabled. No routing override is used. The server is launched via the CLI
entry point with a read-only trace on `PromptProcessingBatch.prompt`.

Code evidence: `Scheduler._create_batch_generator` forwards the configured
prefill batch size into the dependency's `BatchGenerator`; native text models
use this scheduler. The hybrid profile does not force this text lane into a
singleton prefill. Server logs report `mllm=False`, continuous batching, two
requests admitted in the same scheduling turn (`running=2 waiting=0`), and
`PREFILL_PROOF rows=2 lengths=[307, 4]` (sometimes reversed). That trace occurs
for every measured pair on both revisions and both models. HTTP reaches the
right-padded recurrence; this is not a serialized-hybrid false negative.

| Model / revision | Pairs | Short response matches alone | Long response matches alone | Two-row prefills |
| --- | ---: | ---: | ---: | ---: |
| Lightning 30B 4-bit / main | 5 | 0/5 | 5/5 | 5/5 |
| Lightning 30B 4-bit / candidate | 5 | 5/5 | 5/5 | 5/5 |
| Super VL extracted text 3-bit / main | 1 | 0/1 | 1/1 | 1/1 |
| Super VL extracted text 3-bit / candidate | 1 | 0/1 | 1/1 | 1/1 |

The Lightning short request produces 64 tokens alone, but only three in every
main concurrent control. Candidate concurrent responses match the full
standalone responses in all five runs. A preliminary untraced main run also
reproduced 0/5 short and 5/5 long matches.

Super VL uses the existing local artifact
`/Volumes/RTL-2T/scratch-archive/rapid-mlx/super-vl-120b/text-3bit-mixed`.
Its flagged adapter already owns the padding fix. Both standalone outputs and
both concurrent outputs are identical between main and candidate, while the
short concurrent output differs from standalone on both. This PR therefore
makes **no Super VL full-response parity improvement claim**. Batch precision
and greedy-response drift in this artifact remain a separate Vector follow-up;
no scheduler or adapter precision changes are included to force a match.
The tiny adapter regression proves recurrence correctness, not full-size
checkpoint batch invariance. Earlier checkpoint-precision numerical caveats
remain applicable.

Logs and response JSON are in `/private/tmp/rapid-mlx-ssm-http/`:
`main-lightning-traced`, `branch-lightning`, `main-super`, `branch-super`.
Measured ports were 53155, 52784, 53342 and 54112 respectively; all were
allocated from free loopback ports. Only owned server process groups were
stopped, and each server exited after shutdown. The lease is released after
all validation processes stop.

### Exact HTTP reproduction

Create the two scratch scripts below (kept out of `scripts/` because this is
one-off investigation instrumentation). Set `RAPID_PYTHON` to the worktree's
`.venv/bin/python` and `RAPID_CANDIDATE` to its absolute checkout path. The same
Python environment is used for both source trees; `PYTHONPATH` selects source.
Acquire the Studio lease first, then:

```sh
mkdir -p /private/tmp/rapid-mlx-ssm-http
git worktree add --detach /private/tmp/rapid-mlx-ssm-http/main a1b3b1d21323540eee59d7d610b802afd4d97ae6
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MLX_ENABLE_TF32=0
export RAPID_PYTHON="$PWD/.venv/bin/python" RAPID_CANDIDATE="$PWD"
lockf -k /var/tmp/rapid-mlx-large-model.lock "$RAPID_PYTHON" /private/tmp/rapid-mlx-ssm-http/http_check.py --root /private/tmp/rapid-mlx-ssm-http/main --model mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit --tag main-lightning-traced
lockf -k /var/tmp/rapid-mlx-large-model.lock "$RAPID_PYTHON" /private/tmp/rapid-mlx-ssm-http/http_check.py --root "$RAPID_CANDIDATE" --model mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit --tag branch-lightning
lockf -k /var/tmp/rapid-mlx-large-model.lock "$RAPID_PYTHON" /private/tmp/rapid-mlx-ssm-http/http_check.py --root /private/tmp/rapid-mlx-ssm-http/main --model /Volumes/RTL-2T/scratch-archive/rapid-mlx/super-vl-120b/text-3bit-mixed --tag main-super --repeats 1
lockf -k /var/tmp/rapid-mlx-large-model.lock "$RAPID_PYTHON" /private/tmp/rapid-mlx-ssm-http/http_check.py --root "$RAPID_CANDIDATE" --model /Volumes/RTL-2T/scratch-archive/rapid-mlx/super-vl-120b/text-3bit-mixed --tag branch-super --repeats 1
```

`/private/tmp/rapid-mlx-ssm-http/serve_trace.py`:

```python
import importlib, logging, runpy
from rapid_mlx import _mlx_compat
_mlx_compat.install()
generate=importlib.import_module('mlx_lm.generate')
original=generate.PromptProcessingBatch.prompt
def traced(self, prompts, *args, **kwargs):
 logging.getLogger('ssm_http_proof').warning('PREFILL_PROOF rows=%d lengths=%s',len(prompts),[len(p) for p in prompts])
 return original(self,prompts,*args,**kwargs)
generate.PromptProcessingBatch.prompt=traced
runpy.run_module('rapid_mlx.cli',run_name='__main__')
```

`/private/tmp/rapid-mlx-ssm-http/http_check.py`:

```python
import argparse, concurrent.futures, json, os, signal, socket, subprocess, sys, threading, time, urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--model',required=True);p.add_argument('--tag',required=True);p.add_argument('--repeats',type=int,default=5);a=p.parse_args()
out=Path('/private/tmp/rapid-mlx-ssm-http');sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close();assert port not in [7659,7660,8451]
env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MLX_ENABLE_TF32='0',PYTHONPATH=a.root)
cmd=[sys.executable,str(out/'serve_trace.py'),'serve',a.model,'--host','127.0.0.1','--port',str(port),'--no-preflight','--disable-prefix-cache','--disable-disk-caches','--response-cache-entries','0','--no-spec-decode','--log-level','DEBUG','--yes']
log=open(out/(a.tag+'.log'),'w');server=subprocess.Popen(cmd,cwd=a.root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
def get(path,payload=None):
 data=None if payload is None else json.dumps(payload).encode();r=urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=data,headers={'Content-Type':'application/json'});return json.load(urllib.request.urlopen(r,timeout=600))
try:
 for i in range(600):
  if server.poll() is not None:raise RuntimeError(f'server exited {server.returncode}')
  try:
   models=get('/v1/models');break
  except Exception:time.sleep(1)
 else:raise TimeoutError('startup')
 model=models['data'][0]['id'];prompts=['The capital of France is','Explain why the sky is blue in simple terms. '*30+'\nGive a concise answer:']
 def request(i,barrier=None):
  if barrier:barrier.wait()
  response=get('/v1/completions',dict(model=model,prompt=prompts[i],temperature=0,max_tokens=64,stream=False))
  return response
 alone=[request(i) for i in range(2)];pairs=[]
 for run in range(a.repeats):
  barrier=threading.Barrier(2)
  with concurrent.futures.ThreadPoolExecutor(2) as pool:
   futures=[pool.submit(request,i,barrier) for i in range(2)];responses=[f.result() for f in futures]
  matches=[r['choices'][0]['text']==alone[i]['choices'][0]['text'] for i,r in enumerate(responses)];pairs.append(dict(matches=matches,responses=responses));print(a.tag,run,matches,flush=True)
 report=dict(command=cmd,root=a.root,model=a.model,port=port,prompts=prompts,alone=alone,pairs=pairs)
 (out/(a.tag+'.json')).write_text(json.dumps(report,indent=2))
finally:
 os.killpg(server.pid,signal.SIGTERM)
 try:server.wait(timeout=40)
 except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
 log.close()
```

## Final review and verification

Adversarial review covered masking, incoming-state decay, chunk-local gather
bounds, dtype, restored caches, masked one-token tails, installer concurrency,
idempotence, and installed-runtime compatibility. The original installer lacked
version/signature guards; these were added. Independent review then reproduced
an early missing-symbol dereference: fixed with safe symbol lookup, guarded
attribute access, and regressions for removed functions, methods and classes.
Unsupported runtimes now skip atomically before either function is replaced.
The ordinary decoder remains the dependency dispatcher and kernel. No kernel,
scheduler, catalog, release or prefix-cache policy changes are included.

The Apple CI roster retains `tests/test_nemotron_h_ssm.py`. Final validation
includes that file, `test_mlx_compat.py`, `test_ci_apple_coverage_union.py`, the
Super VL and vendored Bolt suites, conversion and metadata tests. The full
relevant set passes 130 tests; Ruff check/format pass. Changed-line coverage
against the rebased main is 151/151 executable lines (100%), including both
the loader hook and the opt-in validation script. The float32 cached-checkpoint control was repeated
after rebase and retains the documented errors and both eight-token matches.
The checkpoint-precision logit diagnostic still fails its declared tolerance,
as documented above; it is diagnostic evidence, not a passing test claim.
