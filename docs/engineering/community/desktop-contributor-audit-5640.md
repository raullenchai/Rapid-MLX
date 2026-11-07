# Desktop contributor requests: discussion #1763

Audit baseline: `aa4d679ec` (main), 2026-11-06. Owner: Pixel; network and
public API decisions remain with Atlas. Tracking: #5640; source discussion:
https://github.com/raullenchai/Rapid-MLX/discussions/1763.

| Request | Audited status | Evidence and disposition |
| --- | --- | --- |
| Stable API bearer across model starts | Implemented | #2639 merged; #2599 closed. `EmbeddedBearerCredential.swift` and `ServerManager.activeBearer` implement bounded/persistent lifetimes. Do not resurrect superseded #2422. |
| Desktop API clients over LAN | Missing, loopback intentional | `ServerManager.host` is `127.0.0.1`; `PortAllocator.canBind` documents avoiding LAN exposure without consent. Requires an explicit authenticated opt-in design from Atlas. Closed #2370 also requested browser chat and model management; that broader Web UI is not this task. |
| Host memory visibility | Implemented | `MemoryProbe.snapshot` samples host VM statistics; `MemoryPill` shows system used/total. This is not the app's own footprint. |
| Metal allocator visibility | Engine implemented, Desktop missing at baseline | `routes/health.py` exposes `/v1/status.metal` active/cache/peak; `scheduler.py` and `mllm_scheduler.py` divide bytes by 1e9. This change adds a read-only Settings → Performance view using those existing authenticated counters. No new API. |
| Per-process / weights / prefix-cache breakdown | Still missing | Allocator counters overlap process memory and cannot establish a weights-versus-prefix partition. Do not infer the requested breakdown by subtraction. A process-footprint view needs a separate measurement contract, including which child processes count. |
| User-selected model collection directory | Implemented | `SettingsModelManagementPanel.chooseModelsFolder`, `ModelsFolderPreference`, and `ModelCatalog.mergedExtraModelRoots` propagate a selected absolute directory to discovery and serve. External rows remain usable but cannot be deleted through the managed-cache action. |
| Select one arbitrary flat checkpoint directory directly | Partial, remaining UX gap | `_scan_external_model_dirs` in `cli.py` scans `<root>/<name>` and `<root>/<publisher>/<name>`, not the selected root itself. A supported complete checkpoint in those layouts is discoverable. `ModelCatalog` accepts engine-certified absolute `launch_path` for identity collisions; this is not a general file-open workflow. `QuickstartModel` bridges bundled flat weights and is not arbitrary-directory import. Track a focused direct-open UX separately. |
| Manual retain-cache control | Intentionally not resumed | The contributor withdrew this proposal after the automatic pressure eviction behavior was explained. Do not add a redundant manual cache switch. |

## Scope and verification

The Metal view refreshes only while the Performance panel is mounted. A sample
belongs to the exact server session, port, and bearer; stop/restart/rotation
invalidates it. Missing fields, failed requests, and invalid values render a
dash. It labels decimal GB and explains that these counters must not be added
to host memory or presented as weights/prefix-cache components.

Relevant existing coverage: `ExternalModelCatalogTests`,
`ModelsFolderPreferenceTests`, `MemoryProbeTests`, `tests/test_cli_models.py`
(external scanning), and `tests/test_local_models.py` (launch identities).
New focused coverage: `MetalMemorySnapshotTests` checks optional/invalid
measurements, decimal GB presentation, and authenticated loopback routing to
the allocated port. A live model/Metal runtime has not been exercised by this
source audit; no model downloads or performance claims are part of this work.

## Handoff

Atlas: decide whether authenticated LAN opt-in merits a dedicated design;
do not reopen #2370 as a prerequisite. Pixel: track direct checkpoint opening
in #5650 separately from collection-root selection, keeping discovery's completeness,
containment, identity, and external-deletion safeguards. Engine/Vector: define
process ownership and measurement semantics before promising a per-process or
weights/prefix split. No delivery date or compatibility expansion is promised.

## CI follow-up

Desktop run `42336462092` compiled successfully but rejected five new strings
without Simplified Chinese translations (ten assertions across two catalog
guards). The catalog now translates the section title, three counters, and
measurement explanation. `LocalizationTests|MetalMemory` passes all 35 tests.

The same run observed the intended eight-node Metal section before the Model
heading in Performance. The three Performance AX baselines now include that
exact structure, including the unsupported-MTP state reached after the enabled
state. No golden assertion was relaxed. Actual GUI replay with the head's CI-built release app passed both
`settings-persistence` and `settings-mtp` (enabled and unsupported states).
A scratch-only status fixture also exercised the real app: stopped engine →
three dashes; live engine → 1.25/0.50/2.75 GB; HTTP 503 → three dashes; restored
endpoint → values again; terminated child → three dashes; restarted child →
values again. This verifies presentation and session recovery without claiming
a real Metal/model measurement or downloading model weights.

Adversarial review identified a masked 401 regression in the test fixture. The
unauthorized response now includes valid Metal data, so only HTTP-status
rejection can make that case return nil. Negative control: temporarily replace
`(200...299).contains(http.statusCode)` with `http.statusCode >= 100`;
`fetchHandlesMissingUnsupportedAndFailedResponses` must fail its nil assertion.
The mutation is restored before the positive suite.
