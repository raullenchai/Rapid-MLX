import Foundation
import Testing
@testable import Rapid

/// Parser cases for ``DownloadProgress``. Migrated from
/// ``TestDriver.runDownloadProgress`` — same input strings, same
/// expectations, now expressed as Swift Testing ``@Test`` cases.
/// The strings are pasted directly from real ``rapid-mlx serve``
/// stderr (cold-start of a ~30 GB Qwen 3.6 35B-A3B fetch) and exist
/// to pin the tqdm format we negotiated; do not "tidy them up."
@MainActor
@Suite("DownloadProgress parser")
struct DownloadProgressTests {
    @Test("Outer 'Fetching N files' tqdm tick parses to .fetching")
    func fetchingLine() {
        let progress = DownloadProgress()
        progress.ingest("Fetching 16 files:  31%|███▏      | 5/16 [00:42<01:32, 0.12it/s]")
        guard case .fetching(let done, let total, let percent) = progress.phase else {
            Issue.record("expected .fetching, got \(progress.phase)")
            return
        }
        #expect(done == 5)
        #expect(total == 16)
        #expect(percent == 31)
    }

    @Test("Per-file tqdm tick parses to .downloading with speed + ETA")
    func perFileLine() {
        let progress = DownloadProgress()
        progress.ingest("model-00001-of-00006.safetensors:  41%|████▏     | 2.10G/5.13G [01:28<02:09, 23.4MB/s]")
        guard case .downloading(let file, let done, let total, let percent, let speed, let eta) = progress.phase else {
            Issue.record("expected .downloading, got \(progress.phase)")
            return
        }
        #expect(file == "model-00001-of-00006.safetensors")
        #expect(done == "2.10G")
        #expect(total == "5.13G")
        #expect(percent == 41)
        #expect(speed == "23.4MB/s")
        #expect(eta == "02:09")
    }

    /// #150 root-cause regression. ``huggingface_hub`` ≥0.20 ships its
    /// per-file tqdm with `unit_divisor=1024`, which renders bytes as
    /// IEC-suffixed `2.10GiB/5.13GiB` (and speed as `23.4MiB/s`)
    /// rather than the SI `2.10G/5.13G` we used to match. The old
    /// ``isByteToken`` rejected the `i` character and silently
    /// dropped EVERY per-file line in production — leaving the
    /// overlay stuck on the outer ``Fetching N files`` counter for
    /// the entire multi-minute first-time download. Pin both the
    /// IEC-suffix bytes AND the IEC-suffix speed end-to-end so this
    /// can't drift again.
    @Test("#150: per-file tqdm with IEC suffixes (GiB / MiB) parses to .downloading")
    func perFileLineIECSuffix() {
        let progress = DownloadProgress()
        progress.ingest("model-00001-of-00006.safetensors:  41%|████▏     | 2.10GiB/5.13GiB [01:28<02:09, 23.4MiB/s]")
        guard case .downloading(let file, let done, let total, let percent, let speed, let eta) = progress.phase else {
            Issue.record("expected .downloading on IEC tqdm, got \(progress.phase)")
            return
        }
        #expect(file == "model-00001-of-00006.safetensors")
        #expect(done == "2.10GiB")
        #expect(total == "5.13GiB")
        #expect(percent == 41)
        #expect(speed == "23.4MiB/s")
        #expect(eta == "02:09")
    }

    @Test("#150: lastTickAt advances on every recognised tick (UI stall caption gate)")
    func lastTickAtAdvances() {
        let progress = DownloadProgress()
        #expect(progress.lastTickAt == .distantPast)
        progress.ingest("Fetching 12 files:  8%| | 1/12 [00:05<01:00, ?it/s]")
        #expect(progress.lastTickAt > .distantPast)
        let firstTick = progress.lastTickAt
        // Hot-pause a hair so the second timestamp is strictly later.
        Thread.sleep(forTimeInterval: 0.01)
        progress.ingest("Fetching 12 files: 16%| | 2/12 [00:10<01:00, ?it/s]")
        #expect(progress.lastTickAt > firstTick)
    }

    @Test("#150: reset() rewinds lastTickAt so a stale stall caption from a previous run can't bleed into the next")
    func resetClearsLastTickAt() {
        let progress = DownloadProgress()
        progress.ingest("Fetching 12 files:  8%| | 1/12 [00:05<01:00, ?it/s]")
        #expect(progress.lastTickAt > .distantPast)
        progress.reset()
        #expect(progress.lastTickAt == .distantPast)
    }

    @Test("v0.6 P1: ETA extracted from tqdm bracket — MM:SS shape")
    func etaTokenSimple() {
        #expect(DownloadProgress.etaToken(in: "[01:28<02:09, 23.4MB/s]") == "02:09")
    }

    @Test("v0.6 P1: ETA accepts H:MM:SS for long downloads")
    func etaTokenWithHours() {
        #expect(DownloadProgress.etaToken(in: "[02:14:30<1:23:45, 800kB/s]") == "1:23:45")
    }

    @Test("v0.6 P1: ETA returns nil for tqdm '?' placeholder (early ticks)")
    func etaTokenPlaceholder() {
        // tqdm emits "[?<?, ?it/s]" before it stabilises an estimate;
        // we must NOT pipe that to the UI.
        #expect(DownloadProgress.etaToken(in: "[?<?, ?it/s]") == nil)
    }

    @Test("v0.6 P1: ETA returns nil for malformed brackets")
    func etaTokenMalformed() {
        #expect(DownloadProgress.etaToken(in: "no brackets here") == nil)
        // Missing the `<elapsed<eta` separator inside the bracket.
        #expect(DownloadProgress.etaToken(in: "[no separator, 23.4MB/s]") == nil)
        // Time-shape failure: letters where digits should be.
        #expect(DownloadProgress.etaToken(in: "[01:28<unknown, 23.4MB/s]") == nil)
    }

    @Test("v0.4.42: 'Loading model with' line transitions to .preparing, NOT past download")
    func preparingTransition() {
        // The previous parser treated this line as the end of
        // download. The user-visible regression (2026-06-10): a
        // 5-minute first-time fetch showed "Loading model into
        // Metal..." for the entire window because rapid-mlx emits
        // "Loading model with" BEFORE load_model() internally
        // triggers the HF fetch. .preparing is the new resting
        // place; tqdm will overwrite if/when it arrives.
        let progress = DownloadProgress()
        progress.ingest("INFO:vllm_mlx.server:Loading model with BatchedEngine: mlx-community/Qwen3.6-35B-A3B-4bit")
        guard case .preparing = progress.phase else {
            Issue.record("expected .preparing after 'Loading model with', got \(progress.phase)")
            return
        }
    }

    @Test("v0.4.42 BUG FIX: tqdm fetch after 'Loading model with' overrides .preparing")
    func fetchOverridesPreparing() {
        // Codifies the exact regression the user hit on v0.4.41:
        // rapid-mlx emits "Loading model with BatchedEngine: ..."
        // FIRST, then load_model() internally triggers the HF fetch
        // (which emits the tqdm lines). The old parser locked into
        // .loading on the first line and forward-only invariant
        // blocked the tqdm — so a 5-minute download showed
        // "Loading model into Metal..." with no progress.
        let progress = DownloadProgress()
        progress.ingest("INFO:vllm_mlx.server:Loading model with BatchedEngine: mlx-community/Qwen3.5-9B-4bit")
        progress.ingest("Fetching 13 files: 0%| | 0/13 [00:00<?, ?it/s]")
        guard case .fetching(let done, let total, _) = progress.phase else {
            Issue.record("BUG: fetch line did not override .preparing, phase=\(progress.phase)")
            return
        }
        #expect(done == 0)
        #expect(total == 13)
    }

    @Test("v0.4.42: 'compiling Metal shaders' transitions to .warmingUp")
    func warmingUpFromShaders() {
        let progress = DownloadProgress()
        progress.ingest("INFO:vllm_mlx.server:Warming up (compiling Metal shaders)...")
        guard case .warmingUp = progress.phase else {
            Issue.record("expected .warmingUp after 'compiling Metal shaders', got \(progress.phase)")
            return
        }
    }

    @Test("v0.4.42: 'Warmup complete' also transitions to .warmingUp (kept on screen until /healthz green)")
    func warmingUpFromComplete() {
        let progress = DownloadProgress()
        progress.ingest("INFO:vllm_mlx.server:Warmup complete (12.4s)")
        guard case .warmingUp = progress.phase else {
            Issue.record("expected .warmingUp after 'Warmup complete', got \(progress.phase)")
            return
        }
    }

    @Test("v0.4.42: 'Starting server on http... (warming up...)' does NOT trip .warmingUp")
    func startingServerLineIsNotWarmup() {
        // The CLI's pre-uvicorn announcement says "warming up" in
        // plain English but it's not the actual warmup phase. We
        // must not transition until the lifespan hook emits the
        // real "compiling Metal shaders" marker.
        let progress = DownloadProgress()
        progress.ingest("  Starting server on http://localhost:8000 (warming up — this can take a few seconds)")
        guard case .idle = progress.phase else {
            Issue.record("Premature .warmingUp from CLI announcement, got \(progress.phase)")
            return
        }
    }

    @Test("v0.4.42: forward-only invariant moved to .warmingUp (was .loading)")
    func forwardOnlyOnWarmup() {
        // Once we've reached .warmingUp, no late tqdm flush from the
        // child process can drag us back to .fetching — that would
        // be the kind of UI flicker the previous test was guarding
        // against. The invariant moved from .loading to .warmingUp
        // because .loading no longer exists.
        let progress = DownloadProgress()
        progress.ingest("INFO:vllm_mlx.server:Warming up (compiling Metal shaders)...")
        progress.ingest("Fetching 14 files: 100%|██████████| 14/14 [00:00<00:00, 14749.20it/s]")
        guard case .warmingUp = progress.phase else {
            Issue.record("forward-only invariant violated, phase=\(progress.phase)")
            return
        }
    }

    @Test("reset() returns phase to .idle")
    func resetClears() {
        let progress = DownloadProgress()
        progress.ingest("Fetching 16 files:  31%|███▏      | 5/16 [00:42<01:32, 0.12it/s]")
        progress.reset()
        guard case .idle = progress.phase else {
            Issue.record("expected .idle after reset, got \(progress.phase)")
            return
        }
    }

    @Test("Non-progress noise lines do not change phase")
    func noiseRejection() {
        let progress = DownloadProgress()
        progress.ingest("INFO:     Started server process [62981]")
        guard case .idle = progress.phase else {
            Issue.record("noise leaked into phase, got \(progress.phase)")
            return
        }
    }

    // MARK: - R2 puller format (rapid-mlx >= 0.7.6, _mirror.py)
    //
    // The rapid-mlx ``download_with_mirror_fallback`` puller has its
    // OWN progress format that's mutually exclusive with the HF
    // ``snapshot_download`` tqdm format the rest of this file covers.
    // When the catalog says a model is mirrored (or a custom mirror is
    // configured), the puller runs the R2 phase and prints one
    // completion line PER FILE. There is no mid-shard refresh — a
    // multi-GB shard sits quiet until it finishes, then prints
    // ``[N/M] file R2 (X MB)``. tqdm only enters the picture if the
    // R2 phase decides to fall back to ``snapshot_download``.
    //
    // The strings below are pasted from ``vllm_mlx/_mirror.py`` lines
    // 1012-1046 and 1340-1343 (rapid-mlx 0.7.29 SHA c397b2d) so the
    // parser stays pinned to the production wire format.

    @Test("R2 puller 'Pulling <repo> (R2 mirror, fallback: HF)' header transitions .idle → .preparing")
    func r2PullerHeaderR2Mirror() {
        let progress = DownloadProgress()
        let line = "  Pulling mlx-community/gemma-4-12B-it-qat-4bit (R2 mirror, fallback: HF)"
        let ok = progress.ingest(line)
        #expect(ok, "R2 puller banner should be recognised as progress")
        guard case .preparing = progress.phase else {
            Issue.record("expected .preparing after R2 mirror banner, got \(progress.phase)")
            return
        }
    }

    @Test("R2 puller 'Pulling <repo> (mirror direct-layout, fallback: HF)' also transitions to .preparing")
    func r2PullerHeaderDirectLayout() {
        let progress = DownloadProgress()
        let line = "  Pulling mlx-community/gemma-4-12B-it-qat-4bit (mirror direct-layout, fallback: HF)"
        let ok = progress.ingest(line)
        #expect(ok)
        guard case .preparing = progress.phase else {
            Issue.record("expected .preparing after direct-layout banner, got \(progress.phase)")
            return
        }
    }

    @Test("R2 puller 'Found N files (~X.X GB total)' primes total file count to .fetching(0/N, 0%)")
    func r2PullerFoundFilesWithSize() {
        let progress = DownloadProgress()
        progress.ingest("  Pulling mlx-community/gemma-4-12B-it-qat-4bit (R2 mirror, fallback: HF)")
        let ok = progress.ingest("  Found 12 files (~11.0 GB total)")
        #expect(ok, "Found-files line should be recognised as progress")
        guard case .fetching(let done, let total, let percent) = progress.phase else {
            Issue.record("expected .fetching after Found-files, got \(progress.phase)")
            return
        }
        #expect(done == 0)
        #expect(total == 12)
        #expect(percent == 0)
    }

    @Test("R2 puller 'Found N files' (size unknown variant) still primes total file count")
    func r2PullerFoundFilesNoSize() {
        let progress = DownloadProgress()
        let ok = progress.ingest("  Found 12 files")
        #expect(ok)
        guard case .fetching(let done, let total, let percent) = progress.phase else {
            Issue.record("expected .fetching after Found-files (no size), got \(progress.phase)")
            return
        }
        #expect(done == 0)
        #expect(total == 12)
        #expect(percent == 0)
    }

    @Test("R2 puller per-file 'R2 (X MB)' line advances file counter + percent")
    func r2PullerPerFileR2Hit() {
        let progress = DownloadProgress()
        progress.ingest("  Found 12 files (~11.0 GB total)")
        let ok = progress.ingest("  [8/12] model-00001-of-00003.safetensors R2 (4523 MB)")
        #expect(ok)
        guard case .fetching(let done, let total, let percent) = progress.phase else {
            Issue.record("expected .fetching after R2 per-file, got \(progress.phase)")
            return
        }
        #expect(done == 8)
        #expect(total == 12)
        // 8/12 = 66.67% → integer floor 66
        #expect(percent == 66)
    }

    @Test("R2 puller per-file 'HF (X MB, fallback)' line also advances the counter")
    func r2PullerPerFileHFFallback() {
        let progress = DownloadProgress()
        let ok = progress.ingest("  [3/12] config.json HF (0 MB, fallback)")
        #expect(ok)
        guard case .fetching(let done, let total, _) = progress.phase else {
            Issue.record("expected .fetching after HF fallback per-file, got \(progress.phase)")
            return
        }
        #expect(done == 3)
        #expect(total == 12)
    }

    @Test("R2 puller per-file 'cached (X MB)' line advances the counter")
    func r2PullerPerFileCached() {
        let progress = DownloadProgress()
        let ok = progress.ingest("  [5/12] tokenizer.json cached (1 MB)")
        #expect(ok)
        guard case .fetching(let done, let total, _) = progress.phase else {
            Issue.record("expected .fetching after cached per-file, got \(progress.phase)")
            return
        }
        #expect(done == 5)
        #expect(total == 12)
    }

    @Test("R2 puller per-file 'miss (will retry via HF snapshot_download)' line advances the counter")
    func r2PullerPerFileMiss() {
        let progress = DownloadProgress()
        let ok = progress.ingest("  [6/12] foo.bin miss (will retry via HF snapshot_download)")
        #expect(ok)
        guard case .fetching(let done, let total, _) = progress.phase else {
            Issue.record("expected .fetching after miss per-file, got \(progress.phase)")
            return
        }
        #expect(done == 6)
        #expect(total == 12)
    }

    @Test("R2 puller per-file with ANSI DIM/RESET escapes still parses (TTY-routed child)")
    func r2PullerPerFileWithAnsiEscapes() {
        // When stdout is a TTY ``_print_dim`` emits the line wrapped in
        // ``\x1b[2m…\x1b[0m`` DIM/RESET pairs. The desktop pipes the
        // child so this should be rare, but a wrapper that re-attaches
        // a PTY would leak the raw escapes — the parser must strip
        // them before matching. The line below interleaves DIM at the
        // bracket and at the tag end (matching the exact f-string
        // template at _mirror.py:1340-1342).
        let progress = DownloadProgress()
        let line = "  \u{1B}[2m[7/12]\u{1B}[0m model-00002-of-00003.safetensors \u{1B}[2mR2 (4612 MB)\u{1B}[0m"
        let ok = progress.ingest(line)
        #expect(ok)
        guard case .fetching(let done, let total, _) = progress.phase else {
            Issue.record("expected .fetching after ANSI-wrapped R2 per-file, got \(progress.phase)")
            return
        }
        #expect(done == 7)
        #expect(total == 12)
    }

    @Test("R2 puller banner with ANSI BOLD/DIM escapes still parses")
    func r2PullerHeaderWithAnsiEscapes() {
        let progress = DownloadProgress()
        let line = "  \u{1B}[1mPulling mlx-community/gemma-4-12B-it-qat-4bit\u{1B}[0m \u{1B}[2m(R2 mirror, fallback: HF)\u{1B}[0m"
        let ok = progress.ingest(line)
        #expect(ok)
        guard case .preparing = progress.phase else {
            Issue.record("expected .preparing after ANSI-wrapped banner, got \(progress.phase)")
            return
        }
    }

    @Test("R2 puller full sequence: banner → Found → per-file walks .fetching done from 0 → 12")
    func r2PullerFullSequence() {
        // Pin the user-visible UX of the gemma-4-12b-qat-4bit cold pull
        // the bug report (#?? — "Spinning up rapid-mlx… no progress bar
        // for 12 minutes") was filed against. With the parser fix the
        // overlay walks .preparing → .fetching(0/12, 0%) → … →
        // .fetching(12/12, 100%) instead of sitting at .idle/.preparing
        // for the entire R2 phase.
        let progress = DownloadProgress()
        progress.ingest("  Pulling mlx-community/gemma-4-12B-it-qat-4bit (R2 mirror, fallback: HF)")
        guard case .preparing = progress.phase else {
            Issue.record("expected .preparing after banner, got \(progress.phase)")
            return
        }
        progress.ingest("  Found 12 files (~11.0 GB total)")
        guard case .fetching(0, 12, 0) = progress.phase else {
            Issue.record("expected .fetching(0,12,0) after Found, got \(progress.phase)")
            return
        }
        progress.ingest("  [1/12] README.md R2 (0 MB)")
        progress.ingest("  [2/12] config.json R2 (0 MB)")
        progress.ingest("  [12/12] model-00003-of-00003.safetensors R2 (3041 MB)")
        guard case .fetching(let done, let total, let percent) = progress.phase else {
            Issue.record("expected .fetching at end of R2 phase, got \(progress.phase)")
            return
        }
        #expect(done == 12)
        #expect(total == 12)
        #expect(percent == 100)
    }

    @Test("R2 puller banner does NOT trigger off plain noise lines")
    func r2PullerBannerNoiseRejection() {
        let progress = DownloadProgress()
        // "Pulling" without the parenthesised mode — must not match.
        progress.ingest("INFO: Pulling latest revision from huggingface")
        guard case .idle = progress.phase else {
            Issue.record("noise 'Pulling' line leaked into phase, got \(progress.phase)")
            return
        }
        // ``[N/M]`` without trailing kind tag — must not match (could
        // be e.g. a unit-test progress line in some downstream tool).
        progress.ingest("[3/12]")
        guard case .idle = progress.phase else {
            Issue.record("bare bracket counter leaked into phase, got \(progress.phase)")
            return
        }
        // ``Found`` with non-integer — must not match.
        progress.ingest("Found XYZ files")
        guard case .idle = progress.phase else {
            Issue.record("non-integer Found line leaked into phase, got \(progress.phase)")
            return
        }
    }

    @Test("R2 puller per-file rejects malformed counters (done > total)")
    func r2PullerPerFileRejectsOutOfRange() {
        let progress = DownloadProgress()
        // ``[13/12]`` — clearly malformed (more done than total). The
        // matcher requires ``done <= total`` so this falls through
        // unmatched and the phase stays .idle.
        progress.ingest("  [13/12] foo.bin R2 (0 MB)")
        guard case .idle = progress.phase else {
            Issue.record("out-of-range counter leaked into phase, got \(progress.phase)")
            return
        }
    }

    // MARK: - R2 aggregate byte heartbeat (v0.7.11 fix for "stuck at 83%")

    @Test("R2 byte heartbeat drives bytes-on-disk + total")
    func r2BytesHeartbeat() {
        let progress = DownloadProgress()
        let ok = progress.ingest("  [bytes] 2147483648/6800000000")
        #expect(ok)
        #expect(progress.bytesDownloaded == 2147483648)
        #expect(progress.totalBytes == 6800000000)
        // Byte path wins over phase fall-through.
        let frac = progress.progressFraction ?? -1
        #expect(frac > 0.3 && frac < 0.4)
    }

    @Test("R2 byte heartbeat is monotonic across ticks")
    func r2BytesHeartbeatMonotonic() {
        let progress = DownloadProgress()
        progress.ingest("  [bytes] 1000000/6800000000")
        progress.ingest("  [bytes] 100000000/6800000000")
        progress.ingest("  [bytes] 3400000000/6800000000")
        #expect(progress.bytesDownloaded == 3400000000)
        // 50% expected at 3.4 / 6.8.
        let frac = progress.progressFraction ?? -1
        #expect(frac >= 0.49 && frac <= 0.51)
    }

    @Test("R2 byte heartbeat with leading ANSI escapes parses")
    func r2BytesHeartbeatWithANSI() {
        let progress = DownloadProgress()
        // ``_print_dim`` would never wrap the heartbeat with ANSI in
        // production (the emitter prints plain), but a wrapper that
        // re-attaches a PTY could. The ANSI stripper inside ``ingest``
        // should normalise.
        let line = "  \u{1B}[2m[bytes]\u{1B}[0m 500/1000"
        let ok = progress.ingest(line)
        #expect(ok)
        #expect(progress.bytesDownloaded == 500)
        #expect(progress.totalBytes == 1000)
    }

    @Test("R2 byte heartbeat rejects malformed payload")
    func r2BytesHeartbeatRejectsMalformed() {
        let progress = DownloadProgress()
        // Non-numeric — must not be claimed as a heartbeat.
        let ok1 = progress.ingest("  [bytes] foo/bar")
        #expect(!ok1)
        // Missing slash.
        let ok2 = progress.ingest("  [bytes] 1234")
        #expect(!ok2)
        // Negative bytes — ``Int64("-1")`` succeeds (returns -1), so
        // rejection relies on the explicit ``done >= 0`` / ``total
        // >= 0`` guards inside ``matchR2BytesHeartbeat``. Don't relax
        // those without re-reading this test.
        let ok3 = progress.ingest("  [bytes] -1/100")
        #expect(!ok3)
        let ok4 = progress.ingest("  [bytes] 100/-1")
        #expect(!ok4)
        // Multiple slashes — pieces.count != 2.
        let ok5 = progress.ingest("  [bytes] 100/200/300")
        #expect(!ok5)
        // Empty done.
        let ok6 = progress.ingest("  [bytes] /100")
        #expect(!ok6)
        // Empty total.
        let ok7 = progress.ingest("  [bytes] 100/")
        #expect(!ok7)
        // No observation should have leaked through.
        #expect(progress.bytesDownloaded == nil)
        #expect(progress.totalBytes == nil)
    }

    @Test("R2 byte heartbeat with total=0 does NOT clobber a known total")
    func r2BytesHeartbeatPreservesKnownTotal() {
        let progress = DownloadProgress()
        // Catalog-priming path: ServerManager.installStartupByteMonitor
        // calls setTotalBytes(estimate) before the puller boots. A
        // first heartbeat with total=0 (HF didn't expose sizes) must
        // NOT erase the prior estimate.
        progress.setTotalBytes(1_000_000)
        progress.ingest("  [bytes] 50000/0")
        // Heartbeat itself is still claimed (line was recognised), but
        // setTotalBytes was guarded by ``m.total > 0``.
        #expect(progress.totalBytes == 1_000_000)
        // Bytes observation still landed.
        #expect(progress.bytesDownloaded == 50_000)
    }

    @Test("R2 byte heartbeat refuses a shrinking total")
    func r2BytesHeartbeatMonotonicTotal() {
        let progress = DownloadProgress()
        progress.ingest("  [bytes] 1000/1000000")
        #expect(progress.totalBytes == 1_000_000)
        // A buggy mirror replays an OLDER, smaller D/T pair. The
        // matcher must refuse the shrink — otherwise progressSubtitle
        // would render "X / smaller-Y · 100%" until the next legit
        // heartbeat with the real total fires.
        progress.ingest("  [bytes] 500/500000")
        #expect(progress.totalBytes == 1_000_000)
        // A bigger total IS accepted (HF metadata refresh mid-pull).
        progress.ingest("  [bytes] 500/2000000")
        #expect(progress.totalBytes == 2_000_000)
    }

    @Test("First measured heartbeat overrides a larger a-priori estimate (#520)")
    func r2BytesHeartbeatOverridesInflatedEstimate() {
        let progress = DownloadProgress()
        // ServerManager seeds the ModelSizing estimate at job start. For
        // a 2-bit / ternary alias that estimate is ~2x inflated
        // (bonsai-1.7b-2bit: 957 MB estimated vs 495 MB real).
        progress.setTotalBytes(957_000_000)
        #expect(progress.hasMeasuredTotal == false)
        // The puller's first real [bytes] heartbeat reports the true,
        // SMALLER total. It must REPLACE the estimate rather than be
        // rejected by the monotonic guard — the bug was the bar staying
        // pinned at the inflated 957 MB for the entire download.
        progress.ingest("  [bytes] 50000000/495000000")
        #expect(progress.totalBytes == 495_000_000)
        #expect(progress.hasMeasuredTotal == true)
        // Once a measured total exists, a later stale/smaller replay is
        // still refused (NIT #5 replay-shrink guard stays in force).
        progress.ingest("  [bytes] 60000000/400000000")
        #expect(progress.totalBytes == 495_000_000)
        // A genuinely larger refresh is still accepted.
        progress.ingest("  [bytes] 60000000/500000000")
        #expect(progress.totalBytes == 500_000_000)
    }

    @Test("R2 byte heartbeat with done>total discards the contradictory total")
    func r2BytesHeartbeatDiscardsOverrunTotal() {
        let progress = DownloadProgress()
        // Edge case: a mirror that lied about Content-Length might
        // deliver more bytes than expected. The byte channel ITSELF
        // is what the UI binds to; the clamp lives in
        // ``progressFraction``. Don't let the bar ever read > 100%.
        progress.ingest("  [bytes] 1500/1000")
        #expect(progress.progressFraction == nil)
        #expect(progress.totalBytes == nil)
        #expect(progress.progressSubtitle == "1.5 KB downloaded")
    }

    @Test("An estimate smaller than observed bytes is not shown as a contradictory total (#1550)")
    func estimatedTotalBelowObservedBytesBecomesUnknown() {
        let progress = DownloadProgress()
        progress.setTotalBytes(563 * 1024 * 1024)
        progress.seedDiskBaseline(bytes: 0)
        progress.applyDiskObservation(bytes: 633 * 1024 * 1024)

        #expect(progress.totalBytes == nil)
        #expect(progress.progressFraction == nil)
        #expect(progress.progressSubtitle == "633 MB downloaded")
    }

    @Test("ServerManager log-suppression classifier matches plain + ANSI heartbeats")
    func heartbeatLogLineClassifier() {
        #expect(DownloadProgress.isHeartbeatLogLine("  [bytes] 100/200"))
        #expect(DownloadProgress.isHeartbeatLogLine("[bytes] 100/200"))
        // ANSI escapes wrapping the prefix still classify.
        #expect(
            DownloadProgress.isHeartbeatLogLine(
                "  \u{1B}[2m[bytes]\u{1B}[0m 100/200"
            )
        )
        // Non-heartbeat lines that the user WANTS to see are NOT
        // suppressed.
        #expect(!DownloadProgress.isHeartbeatLogLine("  [1/12] config.json R2 (0 MB)"))
        #expect(!DownloadProgress.isHeartbeatLogLine("  Pulling foo/bar (R2 mirror)"))
        #expect(!DownloadProgress.isHeartbeatLogLine("INFO: server started"))
    }

    @Test("R2 byte heartbeat preempts the 83%-stuck file-count fallback")
    func r2BytesHeartbeatBeatsFileCountForProgress() {
        let progress = DownloadProgress()
        // Replay the v0.7.10 sequence: banner → Found → 5 of 6 per-file
        // completions land → desktop sits at 5/6 = 83% during the big
        // shard. With the heartbeat live, the bar moves smoothly.
        progress.ingest("  Pulling mlx-community/gemma-4-12B-it-qat-4bit (R2 mirror, fallback: HF)")
        progress.ingest("  Found 6 files (~7.0 GB total)")
        for n in 1...5 {
            progress.ingest("  [\(n)/6] file\(n).bin R2 (10 MB)")
        }
        // File-count phase pinned at 83% — that's exactly the bug.
        if case .fetching(_, _, let pinnedPercent) = progress.phase {
            #expect(pinnedPercent >= 80 && pinnedPercent <= 84)
        } else {
            Issue.record("expected .fetching after 5/6 R2 completions")
        }
        // Heartbeat lands mid-shard at 50% of total bytes.
        progress.ingest("  [bytes] 3500000000/7000000000")
        // progressFraction should now reflect bytes, not file count.
        let frac = progress.progressFraction ?? -1
        #expect(frac >= 0.49 && frac <= 0.51)
        // Heartbeat advances to 78% — file count would still be 83%, but
        // bytes are the authoritative signal.
        progress.ingest("  [bytes] 5500000000/7000000000")
        let frac2 = progress.progressFraction ?? -1
        #expect(frac2 >= 0.78 && frac2 <= 0.79)
    }

    @Test("R2 byte heartbeat populates progressSubtitle with X / Y GB · Z%")
    func r2BytesHeartbeatSubtitle() {
        let progress = DownloadProgress()
        progress.ingest("  [bytes] 1288490188/6871947674") // ~1.2 / 6.4 GiB
        let subtitle = progress.progressSubtitle
        #expect(subtitle?.contains("GB") == true)
        #expect(subtitle?.contains("·") == true)
        #expect(subtitle?.contains("%") == true)
    }

    // MARK: - v0.7.12: bytes/s + ETA

    /// The first observation can't compute a rate — by definition there's
    /// only one sample. The UI must hide the speed token in that case so
    /// it doesn't read "0 KB/s" the moment the bar appears.
    @Test("v0.7.12: bytesPerSecond is nil before two samples land")
    func bytesPerSecondNilBeforeSecondSample() {
        let progress = DownloadProgress()
        let t0 = Date(timeIntervalSince1970: 1_000_000)
        progress.applyDiskObservation(bytes: 100_000_000, at: t0)
        #expect(progress.bytesPerSecond == nil)
    }

    /// Two samples one second apart, 1 MB apart → 1 MB/s exactly.
    @Test("v0.7.12: bytesPerSecond = (new − old) / span over the in-window samples")
    func bytesPerSecondTwoSamples() {
        let progress = DownloadProgress()
        let t0 = Date(timeIntervalSince1970: 1_000_000)
        progress.applyDiskObservation(bytes: 100_000_000, at: t0)
        progress.applyDiskObservation(bytes: 101_048_576, at: t0.addingTimeInterval(1.0))
        guard let rate = progress.bytesPerSecond else {
            Issue.record("expected a published rate after two samples")
            return
        }
        // Allow ±5% drift for floating-point.
        #expect(rate >= 1_048_576 * 0.95 && rate <= 1_048_576 * 1.05)
    }

    /// Bursty arrivals — 3 samples 500 ms apart, total +1.5 MB → average
    /// 1 MB/s over the 1 s window. We deliberately keep the math simple
    /// (newest − oldest in the window) so the answer is stable.
    @Test("v0.7.12: bytesPerSecond averages over the rolling window, not just the last delta")
    func bytesPerSecondAveragesWindow() {
        let progress = DownloadProgress()
        let t0 = Date(timeIntervalSince1970: 2_000_000)
        progress.applyDiskObservation(bytes: 0,           at: t0)
        progress.applyDiskObservation(bytes: 500_000,     at: t0.addingTimeInterval(0.5))
        progress.applyDiskObservation(bytes: 1_000_000,   at: t0.addingTimeInterval(1.0))
        guard let rate = progress.bytesPerSecond else {
            Issue.record("expected a published rate")
            return
        }
        // 1 MB across 1 s → 1_000_000 B/s.
        #expect(rate >= 950_000 && rate <= 1_050_000)
    }

    /// Caller stops feeding samples mid-flight (engine wedged, network
    /// disconnected). On the next ``applyDiskObservation`` past the
    /// staleness threshold the published rate must drop to ``nil`` so
    /// the subtitle doesn't keep reading a phantom "5 MB/s" forever.
    /// Direct test: feed two samples one second apart, then a third
    /// six seconds later (past ``rateStaleSeconds`` from itself? no — we
    /// rely on the staleness check against the newest sample, but
    /// the OLDEST sample being purged means the post-staleness window
    /// only holds the brand-new one — which alone can't yield a rate).
    @Test("v0.7.12: bytesPerSecond clears when no fresh sample arrives within the staleness window")
    func bytesPerSecondClearsOnStall() {
        let progress = DownloadProgress()
        let t0 = Date(timeIntervalSince1970: 3_000_000)
        progress.applyDiskObservation(bytes: 1_000_000, at: t0)
        progress.applyDiskObservation(bytes: 2_000_000, at: t0.addingTimeInterval(1.0))
        #expect(progress.bytesPerSecond != nil)
        // 10 s of silence, then one fresh sample. The window-trim purges
        // the two old samples; only the new one survives → no rate.
        progress.applyDiskObservation(bytes: 2_100_000, at: t0.addingTimeInterval(11.0))
        #expect(progress.bytesPerSecond == nil)
    }

    /// HuggingFace's ``huggingface-cli scan-cache`` cleanup can briefly
    /// shrink the on-disk size between observations (the puller hardlinks
    /// blobs into a snapshot dir, then deletes the original blob). The
    /// rate calc must NOT publish a negative number — that would render
    /// as "< 1 KB/s" momentarily and worse, the ETA branch could compute
    /// against bogus inputs. Suppress instead.
    @Test("v0.7.12: bytesPerSecond suppressed on shrinking buffer (cleanup tick)")
    func bytesPerSecondShrinkingDelta() {
        let progress = DownloadProgress()
        let t0 = Date(timeIntervalSince1970: 4_000_000)
        progress.applyDiskObservation(bytes: 5_000_000, at: t0)
        progress.applyDiskObservation(bytes: 4_500_000, at: t0.addingTimeInterval(0.5))
        #expect(progress.bytesPerSecond == nil)
    }

    /// ``reset()`` must wipe the sample buffer AND the published rate
    /// so a stale "12 MB/s" from the previous run can't bleed into the
    /// next overlay (which would happen if the new run started slow).
    @Test("v0.7.12: reset() clears the sample buffer + bytesPerSecond")
    func bytesPerSecondClearedByReset() {
        let progress = DownloadProgress()
        let t0 = Date(timeIntervalSince1970: 5_000_000)
        progress.applyDiskObservation(bytes: 1_000_000, at: t0)
        progress.applyDiskObservation(bytes: 2_000_000, at: t0.addingTimeInterval(1.0))
        #expect(progress.bytesPerSecond != nil)
        progress.reset()
        #expect(progress.bytesPerSecond == nil)
    }

    /// Numeric formatter thresholds — these gate the entire subtitle
    /// readability so pin the cliffs explicitly.
    @Test("v0.7.12: formatSpeed switches units like Chrome / Finder")
    func formatSpeedThresholds() {
        // < 1 KB/s collapses to the "< 1 KB/s" floor so a 0 B/s readout
        // can never appear next to a still-moving GB counter.
        #expect(DownloadProgress.formatSpeed(bytesPerSecond: 0) == "< 1 KB/s")
        #expect(DownloadProgress.formatSpeed(bytesPerSecond: 512) == "< 1 KB/s")
        // ≥ 1 KB/s but below 100 → 1-decimal precision (matches `formatBytes`).
        #expect(DownloadProgress.formatSpeed(bytesPerSecond: 1024) == "1.0 KB/s")
        #expect(DownloadProgress.formatSpeed(bytesPerSecond: 683 * 1024) == "683 KB/s")
        // MB/s and GB/s switch points.
        let oneMB = 1024.0 * 1024.0
        #expect(DownloadProgress.formatSpeed(bytesPerSecond: oneMB) == "1.0 MB/s")
        #expect(DownloadProgress.formatSpeed(bytesPerSecond: 23.4 * oneMB) == "23.4 MB/s")
        let oneGB = 1024.0 * 1024.0 * 1024.0
        #expect(DownloadProgress.formatSpeed(bytesPerSecond: 1.5 * oneGB) == "1.5 GB/s")
    }

    /// ETA cliffs map onto Chrome's "less than a minute" / "N min" /
    /// "H h M min" idiom. Past 24 h we cap so a momentarily-slow burst
    /// doesn't quote "47 hours left" and trash user trust.
    @Test("v0.7.12: formatETA uses < 1 min / N min / H h M min idiom, caps at > 24 h")
    func formatETAIdiom() {
        let oneMB = 1024.0 * 1024.0
        // 1 MB remaining at 10 MB/s → 0.1 s → "< 1 min left".
        #expect(DownloadProgress.formatETA(bytesRemaining: Int64(oneMB), bytesPerSecond: 10 * oneMB) == "< 1 min left")
        // 60 MB at 1 MB/s → 60 s → "1 min left".
        #expect(DownloadProgress.formatETA(bytesRemaining: Int64(60 * oneMB), bytesPerSecond: oneMB) == "1 min left")
        // 300 MB at 1 MB/s → 5 min.
        #expect(DownloadProgress.formatETA(bytesRemaining: Int64(300 * oneMB), bytesPerSecond: oneMB) == "5 min left")
        // 3600 + 23·60 s ≈ 1 h 23 min.
        let oneGB = 1024.0 * 1024.0 * 1024.0
        let bigRemaining = Int64((3600 + 23 * 60) * oneMB)
        #expect(DownloadProgress.formatETA(bytesRemaining: bigRemaining, bytesPerSecond: oneMB) == "1 h 23 min left")
        // Exact-hour boundary collapses "H h 0 min" into "H h".
        let twoHours = Int64(2 * 3600 * oneMB)
        #expect(DownloadProgress.formatETA(bytesRemaining: twoHours, bytesPerSecond: oneMB) == "2 h left")
        // Past 24 h caps.
        #expect(DownloadProgress.formatETA(bytesRemaining: Int64(50 * oneGB), bytesPerSecond: 500 * 1024) == "> 24 h left")
        // Pathological inputs return nil rather than infinite-formatted strings.
        #expect(DownloadProgress.formatETA(bytesRemaining: 0, bytesPerSecond: oneMB) == nil)
        #expect(DownloadProgress.formatETA(bytesRemaining: Int64(oneMB), bytesPerSecond: 0) == nil)
        #expect(DownloadProgress.formatETA(bytesRemaining: -1, bytesPerSecond: oneMB) == nil)
    }

    /// Round 1 BLOCKING fix. Used to render "60 min left" at sec ∈
    /// [3570, 3599] and "1 h 60 min left" at the equivalent boundary
    /// inside the hour branch. Carry-the-1 handles both cases.
    @Test("v0.7.12 round-1: formatETA carries minutes==60 into hours instead of rendering '60 min'")
    func formatETAMinuteCarry() {
        let oneMB = 1024.0 * 1024.0
        // 3570 s @ 1 MB/s: round(3570/60) = round(59.5) = 60 → must
        // render "1 h left", NOT "60 min left".
        #expect(DownloadProgress.formatETA(bytesRemaining: Int64(3570 * oneMB),
                                          bytesPerSecond: oneMB) == "1 h left")
        // 7170 s @ 1 MB/s: hours=1, residue=3570, round(59.5)=60 →
        // must render "2 h left", NOT "1 h 60 min left".
        #expect(DownloadProgress.formatETA(bytesRemaining: Int64(7170 * oneMB),
                                          bytesPerSecond: oneMB) == "2 h left")
    }

    /// First-sample contract pin. The bytes-only branch of
    /// ``progressSubtitle`` must read exactly "X downloaded" with no
    /// trailing " · S MB/s" before two samples land, because
    /// ``bytesPerSecond`` is intentionally nil at that point. This
    /// guards the existing `HFCacheByteMonitorTests` assertion that
    /// pins the byte-only subtitle copy.
    @Test("v0.7.12 round-1: progressSubtitle stays 'X downloaded' (no speed tail) after a single observation")
    func progressSubtitleSingleObservationStable() {
        let progress = DownloadProgress()
        let t0 = Date(timeIntervalSince1970: 8_000_000)
        progress.applyDiskObservation(bytes: 750_000, at: t0)
        let subtitle = progress.progressSubtitle ?? ""
        #expect(subtitle == "732 KB downloaded")
        #expect(subtitle.contains("/s") == false)
    }

    /// End-to-end: heartbeats land, the rate stabilises, the subtitle
    /// gains the speed + ETA suffix in the order documented in the
    /// header comment. This is the production read-out the user will
    /// see; if this test passes the UX works.
    @Test("v0.7.12: progressSubtitle gains speed + ETA suffix once the rate is known")
    func progressSubtitleSpeedETA() {
        let progress = DownloadProgress()
        let t0 = Date(timeIntervalSince1970: 6_000_000)
        // Two heartbeats one second apart, 1 MB delta on a 10 MB target.
        progress.setTotalBytes(10 * 1024 * 1024)
        progress.applyDiskObservation(bytes: 1 * 1024 * 1024, at: t0)
        progress.applyDiskObservation(bytes: 2 * 1024 * 1024, at: t0.addingTimeInterval(1.0))
        let subtitle = progress.progressSubtitle ?? ""
        // Bytes branch reflects the LATEST observation (2 MiB of 10 MiB
        // → 20%), then the speed token, then the ETA, joined by " · ".
        #expect(subtitle.contains("2.0 MB / 10.0 MB"))
        #expect(subtitle.contains("20%"))
        #expect(subtitle.contains("MB/s") || subtitle.contains("KB/s"))
        #expect(subtitle.contains("left"))
    }

    /// When the total is unknown (HF didn't expose sizes), the subtitle
    /// falls through to "X downloaded" — but the speed readout still
    /// helps the user feel things are alive, so we append it even
    /// without a total. ETA is correctly absent in this branch (we
    /// can't compute it without a target).
    @Test("v0.7.12: progressSubtitle appends speed without ETA when totalBytes is unknown")
    func progressSubtitleSpeedWithoutTotal() {
        let progress = DownloadProgress()
        let t0 = Date(timeIntervalSince1970: 7_000_000)
        // Deliberately do NOT call setTotalBytes.
        progress.applyDiskObservation(bytes: 500_000, at: t0)
        progress.applyDiskObservation(bytes: 1_500_000, at: t0.addingTimeInterval(1.0))
        let subtitle = progress.progressSubtitle ?? ""
        #expect(subtitle.contains("downloaded"))
        #expect(subtitle.contains("MB/s") || subtitle.contains("KB/s"))
        #expect(subtitle.contains("left") == false)
    }
}
