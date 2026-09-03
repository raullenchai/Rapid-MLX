# Manzanita SwiftPM cache qualification

Date: 2026-09-03  
Repository: `raullenchai/Rapid-MLX`  
Branch: `perf/mzr-swiftpm-cache`  
Runner profile: `manzanita-standard` (M4 Pro host, 6 vCPU / 22 GiB VM)  
Guest: macOS 15.7.7, Xcode 16.4, Swift 6.1.2  

## Result

Do **not** ship the tested GitHub-hosted archive of `apps/rapid-mac/.build`.
It is correct, but its transfer and save costs erase nearly all of the compiler
time it recovers. Keep the production workflow unchanged while evaluating a
node-local, trust-partitioned cache.

The existing serial test policy is load-bearing. Both attempted parallel modes
failed with 22 issues, so parallel execution cannot be enabled for the complete
suite without first isolating shared state or defining a proven-safe shard.

## Correctness qualification

PR #2968 changed only runner labels (plus actionlint configuration). The build
commands and test scope remained unchanged. A representative MZR run completed
3,424 tests with no failures. The earlier GitHub-hosted sample ran 3,431 tests
on a different revision; the difference is source history, not skipped tests.
The sampled MZR and GitHub logs each contained 1,039 warnings.

Dependency resolution is not fully reproducible yet: `Package.resolved` is
ignored and not committed. A lockfile should be committed before a persistent
cross-revision build cache is promoted.

## Measurements

Durations are GitHub job/step wall times. Swift's own reported durations are in
parentheses where available.

| Run | Result | Job | Restore | `swift build` | Test compile | Test execution | Cache save |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GitHub-hosted baseline, job `100595370314` | pass, 3,431 tests | 9:15 | n/a | 2:45 | 2:10 | 3:51 | n/a |
| MZR baseline, job `100673332273` | pass, 3,424 tests | 3:07 | n/a | 0:53 (0:49) | 0:34 | 1:25 | n/a |
| MZR cold archive, job `100689925752` | pass, 3,424 tests | 3:38 | 0:00 | 1:00 (0:56) | 0:43 | 1:27 | 0:08 |
| MZR warm archive, job `100693127668` | pass, 3,424 tests | 3:33 | 0:14 | 0:38 (0:34) | 0:39 | 1:28 | 0:18 |
| MZR default parallel, job `100686698553` | **fail**, 22 issues | 3:11 | cold | 0:55 | included below | 0:40 | not saved |
| MZR 2-worker parallel, job `100688133452` | **fail**, 22 issues | 3:59 | cold | 1:01 | included below | 0:45 | not saved |

The archive compressed to 472,091,036 bytes. The warm run downloaded it at
about 41 MB/s. Because a restore-key (rather than the new revision's exact key)
matched, `actions/cache` uploaded another immutable per-revision archive after
the successful job.

The warm archive reduced the two compile phases by 21 seconds gross relative to
the cold archive, but spent 32 seconds restoring and saving. End-to-end it was
only five seconds faster than the cold archive and 26 seconds slower than the
existing uncached MZR baseline. Run-to-run load variation is larger than the net
benefit.

### Swift compiler CAS probe

Apple Swift 6.1.2 exposes `-cache-compile-job`, `-cas-path`, and cache remarks.
This looked more promising than moving the path-sensitive scratch tree, but the
compiler rejects caching unless SwiftPM uses its experimental explicit-module
build. Rapid cannot currently build in that mode: SwiftPM reports an unknown
dependency target for the Sparkle binary module and fails to resolve the
`RapidDesktopTestWatchdog` test dependency, ending in an internal error. The
compiler CAS path is therefore not production-compatible with this package
graph today. Revisit it after explicit-module builds support this graph; do not
silently make an experimental driver mode part of the required CI check.

## Why caching alone cannot produce 5x

The GitHub-hosted reference is 555 seconds, so a 5x target is 111 seconds. The
serial tests alone take 85-88 seconds after compilation. That leaves only
23-26 seconds for VM setup, checkout, cache materialization, application build,
test build, and post-checks. An archive of `.build` cannot meet that budget.

Reaching approximately 5x therefore requires both:

1. a low-latency node-local compiler/build cache; and
2. a correctness-preserving reduction in test wall time (safe sharding,
   state isolation, or test selection), while retaining a serial full-suite
   merge gate.

## Recommended cache hierarchy

### L0: golden image

Bake Xcode/Swift, global SwiftPM repository downloads, CocoaPods/npm metadata,
and other immutable tools into the Tart golden image. APFS clone-on-write makes
these available to each job without a network transfer.

### L1: node-local NVMe hot cache

Store successful cache generations on the physical runner. Key them by
organization, repository, trust lane, architecture, macOS/Xcode/Swift/SDK,
lockfile/manifest digest, and build flags. Give each job a private writable APFS
clone; atomically promote only a successful trusted generation. Never mount one
read-write `.build` or DerivedData directory into concurrent customer VMs.

Untrusted pull requests may read a trusted default-branch seed but must write to
their own PR/merge-ref namespace and must never promote into the trusted lane.

### L2: NAS/object-store archive

Use the NAS for golden-image distribution, cold cache replication, and disaster
recovery, not as the live `.build` or DerivedData filesystem. Compilation is
metadata- and small-file-heavy; local NVMe should stay on the hot path.

GitHub's cache service can remain a portable fallback for small dependency
archives, but the measured full `.build` archive is not a useful hot cache.

## DerivedData and test plan

This measured `build` job is SwiftPM-native and uses `.build`; Xcode DerivedData
does not participate. DerivedData should be qualified separately on
`gui-app-build` and `gui-golden-flows`, keyed and isolated with the same trust
rules.

For the 3,424-test SwiftPM suite:

1. preserve the serial full-suite gate now;
2. inventory tests that mutate process globals, ports, filesystem locations,
   WKWebView state, singleton/static state, and model residency;
3. move pure tests into deterministic shards and keep the stateful set serial;
4. compare unioned test identifiers against the serial discovery list so no
   test silently disappears; and
5. require repeated green runs under concurrent fleet load before changing the
   required check.

The failed parallel experiments affected GitHub star, IP-pinned transport,
Mermaid, preview auto-reveal, SSE coalescing, SVG safety, runtime probing,
video-generation, web-tools, and model-memory/state suites. Core-count limiting
alone did not make them safe.

## References

- GitHub dependency caching: <https://docs.github.com/en/actions/concepts/workflows-and-actions/dependency-caching>
- GitHub cache matching and scope: <https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching>
- SwiftPM reset/cache/scratch paths: <https://docs.swift.org/swiftpm/documentation/packagemanagerdocs/packagereset/>
- Tart host directory mounts: <https://tart.run/quick-start/>
- Orchard: <https://github.com/openai/orchard>
- Graft image and cache guidance: <https://github.com/Arborist-sh/graft/blob/main/docs/images-and-caching.md>
- Cirrus Labs Omni Cache: <https://github.com/cirruslabs/omni-cache>
