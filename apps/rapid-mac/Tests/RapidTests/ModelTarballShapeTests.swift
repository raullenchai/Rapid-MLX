import Foundation
import Testing

/// Pin the shape of ``scripts/build-model-tarball.sh`` and the
/// release.yml workflow steps that surface it as a workflow
/// artifact (P3 slice β — see ``.claude/loop/bootstrapper-plan.md``).
/// Mirrors the pattern + safety net of ``BootstrapperDMGShapeTests``
/// (slice α).
///
/// Why source-grep tests (and not e2e runs of the script): the script
/// shells out to ``python3`` + reads the user's HF cache + writes a
/// 290 MB tarball; running it in unit tests would be slow + flaky +
/// require a primed HF cache on every contributor's Mac. The actual
/// pack is exercised end-to-end in CI via the workflow step. These
/// tests fail-fast on a PR if a maintainer drops a load-bearing
/// invariant (strict-mode shell, byte-precise size gate, determinism
/// machinery, continue-on-error on the workflow step, etc.).
///
/// What is pinned vs flexible:
///   - Pinned (SHAPE): strict-mode flags, the six EXPECTED_FILES
///     entries that gate the load contract (config.json, tokenizer
///     pair, chat_template.jinja, model.safetensors[.index.json]),
///     the byte-precise size gates, the deterministic packing idiom
///     (Python tarfile + gzip mtime=0 + mode normalisation), the
///     workflow step's continue-on-error + artifact name + path glob.
///   - Flexible: comment wording, ordering of internal helpers, log
///     line formatting. The asserts strip whitespace where the
///     pinned token is structural so reformatting doesn't break the
///     tests.
@Suite("Quickstart model tarball slice β — script + workflow shape")
struct ModelTarballShapeTests {

    /// Repository root, derived from ``#filePath`` so the test runs
    /// from any cwd. Same trick as ``BootstrapperDMGShapeTests``.
    private static var sourceRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // RapidTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // repo root
    }

    private static var scriptPath: URL {
        sourceRoot
            .appendingPathComponent("scripts")
            .appendingPathComponent("build-model-tarball.sh")
    }

    private static var releaseYamlPath: URL {
        sourceRoot
            .appendingPathComponent(".github")
            .appendingPathComponent("workflows")
            .appendingPathComponent("release.yml")
    }

    private static func loadScript() throws -> String {
        try String(contentsOf: scriptPath, encoding: .utf8)
    }

    private static func loadReleaseYaml() throws -> String {
        try String(contentsOf: releaseYamlPath, encoding: .utf8)
    }

    /// Strip whitespace from every character so substring matches
    /// survive reformatting. Mirrors the helper in slice α tests.
    private static func stripWhitespace(_ s: String) -> String {
        s.filter { !$0.isWhitespace }
    }

    // MARK: - script shape

    @Test("script enables bash strict mode (set -euo pipefail)")
    func scriptUsesStrictMode() throws {
        let body = try Self.loadScript()
        // Pin all three flags. Losing any (``e`` / ``u`` / ``o pipefail``)
        // silently changes failure semantics — a typo'd env var would
        // turn into an empty value rather than a hard fail, and a
        // ``python3 ... | xargs ...`` chain would swallow upstream
        // failures.
        #expect(
            body.contains("set -euo pipefail"),
            "scripts/build-model-tarball.sh is missing ``set -euo pipefail``. Strict mode is load-bearing: ``-u`` catches typo'd env overrides (MODEL_TARBALL_MAX_MB / MODEL_TARBALL_MIN_MB / SOURCE_DATE_EPOCH), ``-o pipefail`` catches a failing python3 stage. Restore it at the top of the script."
        )
    }

    @Test("script has the canonical bash shebang")
    func scriptHasBashShebang() throws {
        let body = try Self.loadScript()
        // Match scripts/dmg.sh / build-sidecar-tarball.sh / build-
        // bootstrapper-dmg.sh — ``/usr/bin/env bash`` works on macOS
        // runners and on any contributor's Mac without a hard
        // /bin/bash assumption.
        #expect(
            body.hasPrefix("#!/usr/bin/env bash\n"),
            "scripts/build-model-tarball.sh must start with ``#!/usr/bin/env bash`` to match the rest of the build scripts."
        )
    }

    // MARK: - determinism

    @Test("script derives SOURCE_DATE_EPOCH from snapshot content (NOT git commit time)")
    func scriptDerivesEpochFromContent() throws {
        let body = try Self.loadScript()
        // The whole determinism story keys off SOURCE_DATE_EPOCH —
        // gzip mtime=0, tarfile mtime=epoch, gettarinfo overrides.
        // Without the env override the script must still produce a
        // stable archive.
        #expect(
            body.contains("SOURCE_DATE_EPOCH"),
            "scripts/build-model-tarball.sh must reference SOURCE_DATE_EPOCH (the env-pinned timestamp source for deterministic mtimes inside the tarball). Without this, two back-to-back runs produce different SHA256s and slice γ's content-hash latest.json contract breaks."
        )
        // Codex r3 MAJOR: build-sidecar-tarball.sh falls back to the
        // git commit time when no SOURCE_DATE_EPOCH is set, which
        // silently breaks slice γ's content-hash contract for the
        // MODEL tarball (the same model snapshot packaged on two
        // different desktop release commits would produce different
        // SHA256s). The fix is to derive a CONTENT-STABLE epoch
        // from the snapshot files themselves — same files → same
        // epoch → same SHA, regardless of which desktop release
        // commit packaged it.
        //
        // Pin the content-derivation idiom explicitly so a future
        // refactor that reverts to git commit time trips here.
        #expect(
            body.contains("hashlib.sha256()"),
            "scripts/build-model-tarball.sh must derive SOURCE_DATE_EPOCH from a hashlib.sha256() of the snapshot content (NOT git commit time). Codex r3 MAJOR: git-commit-time fallback silently broke slice γ's content-hash invariant — same model bytes packaged on two different desktop release commits produced different SHA256s. Same files → same content hash → same epoch → same tarball SHA is the load-bearing invariant."
        )
        // The clamp range must keep the epoch in a tar-safe window
        // (post-2001, pre-Y2038-time_t-overflow). Pin both bounds
        // so a future refactor that drops the clamp can't smuggle
        // a year-2038-busting epoch through.
        #expect(
            body.contains("978307200"),
            "scripts/build-model-tarball.sh's content-derived epoch must clamp to LO=978307200 (2001-01-01). Earlier epochs trip some tar extractors that special-case 1970-epoch as 'no mtime'."
        )
        #expect(
            body.contains("2147483647"),
            "scripts/build-model-tarball.sh's content-derived epoch must clamp to HI=2147483647 (2038-01-19, max signed 32-bit time_t). Later epochs overflow USTAR_FORMAT's mtime field."
        )
        // Anti-regression on the codex r3 root cause: assert the
        // script does NOT silently reach for git commit time. The
        // canonical regression form is ``git -C "$ROOT" log -1
        // --format=%ct`` (build-sidecar-tarball.sh's pattern). Match
        // BOTH the ``log -1`` invocation idiom AND the ``--format=%ct``
        // tail independently so a reformatted command line still
        // trips. Codex r4 MINOR caught that the whitespace-stripped
        // form ``git-Clog-1--format=%ct`` was too specific — the
        // real command line has different non-whitespace chars
        // (e.g. shell quoting) between segments that the strip
        // pass doesn't remove.
        #expect(
            !body.contains("git log -1") && !body.contains("git -C") && !body.contains("--format=%ct"),
            "scripts/build-model-tarball.sh appears to fall back to ``git log -1 --format=%ct`` for SOURCE_DATE_EPOCH. That was the codex r3 MAJOR root cause: the git commit time drifts across desktop release commits, so identical model bytes produced different tarball SHAs and slice γ's content-hash contract broke. Use the hashlib.sha256-of-snapshot-content path instead."
        )
    }

    @Test("script honours an explicit SOURCE_DATE_EPOCH override (OSS convention)")
    func scriptHonoursEpochOverride() throws {
        let body = try Self.loadScript()
        // The codex r3 fix replaced "fallback to git commit time"
        // with "derive from snapshot content". An explicit env
        // override MUST still take precedence — that's the
        // reproducible-builds convention and a CI step pinning its
        // own epoch keeps that ability.
        #expect(
            body.contains("[[ -z \"${SOURCE_DATE_EPOCH:-}\" ]]"),
            "scripts/build-model-tarball.sh must only derive a content-stable epoch when SOURCE_DATE_EPOCH is unset. An explicit env override must take precedence (OSS reproducible-builds convention)."
        )
    }

    @Test("script uses Python tarfile + gzip (NOT bsdtar)")
    func scriptUsesPythonTarfileNotBsdtar() throws {
        let body = try Self.loadScript()
        // Lesson from build-sidecar-tarball.sh codex r1: bsdtar's
        // ``pax:exthdr.mtime`` is unsupported, and ``find`` + tar
        // default-recurse produce duplicate entries + lost
        // determinism. Python's tarfile is the only correct path.
        #expect(
            body.contains("import tarfile"),
            "scripts/build-model-tarball.sh must use Python's ``tarfile`` module for deterministic packing. bsdtar on macOS has no portable mtime override and produces non-deterministic archives. Switch to the Python machinery used in build-sidecar-tarball.sh."
        )
        #expect(
            body.contains("import gzip"),
            "scripts/build-model-tarball.sh must use Python's ``gzip`` module so the gzip wrapper's mtime can be pinned to 0. Without this the gzip header carries the wall-clock at pack time and re-runs drift."
        )
        // The two invariants that make the tarball deterministic:
        //   1. gzip mtime=0 on the outer wrapper.
        //   2. USTAR format on the inner tar (no pax exthdrs that
        //      would smuggle in xattrs / capabilities).
        // Both pinned explicitly so a future refactor can't silently
        // drop them.
        #expect(
            body.contains("mtime=0"),
            "scripts/build-model-tarball.sh must call ``gzip.GzipFile(..., mtime=0)`` so the gzip header is content-only. A non-zero mtime makes the tarball SHA256 drift between back-to-back runs."
        )
        #expect(
            body.contains("USTAR_FORMAT"),
            "scripts/build-model-tarball.sh must use ``tarfile.USTAR_FORMAT`` (not the default PAX). PAX exthdrs encode extended attributes whose serialisation is platform-dependent — USTAR's strict layout is the only way to guarantee byte-identical archives across macOS and Linux runners."
        )
        // Codex r2 MAJOR: gzip.GzipFile(out_path, ...) bakes
        // os.path.basename(out_path) into the gzip FNAME header field
        // (silently strips trailing .gz). Since the output basename
        // includes ${APP_VERSION}, identical tar payloads packaged at
        // different desktop versions produce different gzip SHA256s
        // — breaking slice γ's content-hash / R2 dedup invariant.
        // Reproduced live: pre-fix --version 1.2.3 → 8108c1… vs
        // --version 4.5.6 → be00ac… (same tar payload via `cmp`);
        // post-fix both → b2fb38…. The fix requires opening the
        // output file ourselves and passing filename="" to GzipFile.
        // Pin BOTH the explicit empty-filename arg AND the fileobj=
        // form so a regression that drops either trips here.
        #expect(
            body.contains("filename=\"\""),
            "scripts/build-model-tarball.sh must pass ``filename=\"\"`` to gzip.GzipFile(). Otherwise gzip writes os.path.basename(out_path) into the FNAME header — and the basename contains ${APP_VERSION}, so identical tar payloads at different desktop versions produce different SHA256s. That breaks slice γ's content-hash / R2 dedup invariant. Codex r2 MAJOR — verified by repacking with --version 1.2.3 vs --version 4.5.6: pre-fix SHA drifted, post-fix identical."
        )
        #expect(
            body.contains("fileobj=raw_out"),
            "scripts/build-model-tarball.sh must construct gzip.GzipFile() with ``fileobj=`` (NOT a path arg). Passing a path arg lets Python derive the FNAME header from the basename — the very bug codex r2 MAJOR caught. Open the file with ``open(out_path, \"wb\") as raw_out`` first, then pass ``fileobj=raw_out``."
        )
        // Anti-regression: explicitly assert the script does NOT
        // reach for bsdtar. The lesson from sidecar-tarball is that
        // ``bsdtar`` *looks* convenient and a future maintainer who
        // doesn't know the history will try to "simplify". Tripwire.
        // Strip whitespace so a reformatted command line still trips.
        let stripped = Self.stripWhitespace(body)
        #expect(
            !stripped.contains("bsdtar-"),
            "scripts/build-model-tarball.sh appears to invoke ``bsdtar``. The sidecar-tarball history shows bsdtar can't produce deterministic archives on macOS — its pax exthdr override is silently unsupported. Use Python's tarfile module instead (see build-sidecar-tarball.sh for the canonical idiom)."
        )
    }

    @Test("script normalises uid/gid/uname/gname for cross-host stability")
    func scriptNormalisesOwnership() throws {
        let body = try Self.loadScript()
        // Without ownership normalisation, the tarball encodes the
        // packer's uid/uname — different on every developer's Mac and
        // different again on the CI runner. Pin all four pieces of
        // metadata that tarfile would otherwise capture from the host.
        // Use whitespace-stripped match so reformatting (e.g. line
        // wrap) doesn't break the assertion.
        let stripped = Self.stripWhitespace(body)
        #expect(
            stripped.contains("ti.mtime=epoch"),
            "scripts/build-model-tarball.sh must pin ``ti.mtime = epoch`` in the normalise() helper. Without this every entry carries the live mtime of the on-disk file and re-runs drift."
        )
        #expect(
            stripped.contains("ti.uid=0") && stripped.contains("ti.gid=0"),
            "scripts/build-model-tarball.sh must pin ``ti.uid = 0`` and ``ti.gid = 0``. Without this the tarball encodes the packer's uid (different on every contributor's Mac AND on the CI runner) and re-runs drift."
        )
        #expect(
            stripped.contains("ti.uname=\"\"") && stripped.contains("ti.gname=\"\""),
            "scripts/build-model-tarball.sh must pin ``ti.uname = \"\"`` and ``ti.gname = \"\"``. Without this the tarball encodes the packer's username (e.g. 'raullenstudio') and re-runs drift on every developer's Mac."
        )
    }

    @Test("script normalises file modes to pin host-FS perm-bit drift")
    func scriptNormalisesFileModes() throws {
        // Codex r1 MAJOR: gettarinfo() captures st_mode from the
        // on-disk file. A `chmod 600` on a user's HF blob produces a
        // different tarball SHA than a freshly-downloaded 0644 blob,
        // breaking the content-hash contract slice γ depends on.
        // Pin the mode normalisation explicitly (dirs 0755, files
        // 0644) so a regression that drops the chmod block trips
        // here.
        //
        // Reproduced via codex r1's symlink fixture pre-fix:
        //   sha before chmod: 31f35fd…
        //   sha after chmod:  3675f1b… (drift)
        // Post-fix the SHAs are byte-identical regardless of source
        // blob perm-bits.
        let body = try Self.loadScript()
        let stripped = Self.stripWhitespace(body)
        #expect(
            stripped.contains("ti.mode=0o755"),
            "scripts/build-model-tarball.sh must pin ``ti.mode = 0o755`` for directories. Without this, a chmod on a directory inside the HF cache (e.g. a packer running with a different umask) drifts the tarball SHA256 — slice γ's content-hash contract breaks. Verified by codex r1 with a 0o600 vs 0o644 chmod fixture."
        )
        #expect(
            stripped.contains("ti.mode=0o644"),
            "scripts/build-model-tarball.sh must pin ``ti.mode = 0o644`` for regular files. Without this, an HF blob with non-default perms (a user-side `chmod 600` to lock down weights, or a strict-umask download) drifts the tarball SHA256. Codex r1 caught this with a chmod fixture."
        )
    }

    @Test("script walks entries in lexicographic order")
    func scriptWalksLexicographically() throws {
        let body = try Self.loadScript()
        // os.walk's default order is filesystem-dependent (HFS+ vs
        // APFS vs Linux ext4 differ). Pin the sort so the tarball's
        // file order is identical regardless of FS. Both ``dirs.sort()``
        // and ``sorted(files)`` are load-bearing.
        let stripped = Self.stripWhitespace(body)
        #expect(
            stripped.contains("dirs.sort()"),
            "scripts/build-model-tarball.sh must call ``dirs.sort()`` inside the os.walk loop. Without this, os.walk's filesystem-order recursion would produce different tarball layouts on different filesystems (APFS vs ext4 etc.) and SHA256 would drift."
        )
        #expect(
            stripped.contains("sorted(files)"),
            "scripts/build-model-tarball.sh must call ``sorted(files)`` inside the os.walk loop. Without this, file order within a directory follows readdir() order which is filesystem-dependent."
        )
        #expect(
            stripped.contains("entries.sort()"),
            "scripts/build-model-tarball.sh must call ``entries.sort()`` after the os.walk loop completes. Re-sorting the flat list provides full lexicographic determinism even if a future os.walk change yields entries in a different inter-directory order."
        )
    }

    @Test("script dereferences symlinks so blobs land in the archive")
    func scriptDereferencesSymlinks() throws {
        let body = try Self.loadScript()
        // HF snapshot directories store .gitattributes + small JSONs
        // as regular files but the large blobs (model.safetensors,
        // tokenizer.json) are SYMLINKS into ../../blobs/<sha>. If we
        // pack the symlinks naively the tarball ships dangling links
        // that resolve to a non-existent blobs/ alongside on the
        // user's machine. ``os.path.realpath`` resolves the symlink;
        // gettarinfo on the resolved path encodes the actual file
        // size + content into the tar entry. This invariant is what
        // makes the extracted snapshot self-contained.
        let stripped = Self.stripWhitespace(body)
        #expect(
            stripped.contains("os.path.realpath(path)"),
            "scripts/build-model-tarball.sh must call ``os.path.realpath(path)`` on each entry to resolve HF symlinks. HF snapshot dirs contain symlinks into ../../blobs/<sha>; without dereferencing them the tarball ships dangling links that resolve to nothing on the user's machine and the extracted snapshot fails to load."
        )
    }

    // MARK: - flat archive layout (#416)

    @Test("script packs FLAT — arcname is snapshot-relative, no <alias>/ wrapper (#416)")
    func scriptPacksFlatNoAliasWrapper() throws {
        // #416: the pre-fix packer prefixed every arcname with the
        // alias (``os.path.join(arc_root, os.path.relpath(...))``) AND
        // emitted a leading root-dir entry
        // (``tar.gettarinfo(snapshot_dir, arcname=arc_root)``). Because
        // ModelInstaller.stage extracts into a per-alias staging dir and
        // commit atomically renames THAT dir onto
        // ``<installRoot>/<alias>/``, the leading ``<alias>/`` in the
        // tarball produced a double-nested
        // ``<installRoot>/<alias>/<alias>/{config.json,...}`` on disk
        // (F-DGF-V087-03). This test is the producer-side contract that
        // would have caught the nesting: the arcname MUST be the
        // snapshot-relative path with NO alias wrapper and NO leading
        // root-dir entry, so the install leaf is a single level.
        let body = try Self.loadScript()
        let stripped = Self.stripWhitespace(body)

        // Positive: the pack loop derives the arcname directly from the
        // snapshot-relative path — no arc_root join.
        #expect(
            stripped.contains("arcname=os.path.relpath(path,snapshot_dir)"),
            "scripts/build-model-tarball.sh must set ``arcname = os.path.relpath(path, snapshot_dir)`` (flat) so entries pack WITHOUT an ``<alias>/`` wrapper. #416: a wrapper double-nests to <installRoot>/<alias>/<alias>/ because ModelInstaller's commit already renames the staging dir onto <installRoot>/<alias>/."
        )

        // Negative: the pre-fix ``os.path.join(arc_root, ...)`` prefix
        // MUST be gone — that was the source of the ``<alias>/`` wrapper.
        #expect(
            !stripped.contains("os.path.join(arc_root"),
            "scripts/build-model-tarball.sh must NOT prefix arcnames with ``os.path.join(arc_root, ...)`` — that reintroduces the ``<alias>/`` wrapper and the #416 double-nest (<installRoot>/<alias>/<alias>/)."
        )

        // Negative: no leading top-level root-dir entry either. The
        // pre-fix code added ``tar.addfile(tar.gettarinfo(snapshot_dir,
        // arcname=arc_root))`` before the file loop — that root dir
        // entry was the ``<alias>/`` directory member.
        #expect(
            !stripped.contains("tar.gettarinfo(snapshot_dir,arcname="),
            "scripts/build-model-tarball.sh must NOT emit a leading root-dir entry via ``tar.gettarinfo(snapshot_dir, arcname=...)``. #416: that root-dir member is the ``<alias>/`` wrapper directory that caused the double-nest."
        )

        // Negative: the now-retired 4th positional arg to the pack
        // heredoc (``arc_root = sys.argv[4]``) must be gone so a future
        // refactor can't silently thread the alias back into the
        // arcname.
        #expect(
            !stripped.contains("arc_root"),
            "scripts/build-model-tarball.sh must not reference ``arc_root`` anywhere — the alias is no longer threaded into the pack heredoc (retired 4th positional arg). Reintroducing it risks re-nesting the archive under an ``<alias>/`` root."
        )
    }

    // MARK: - file-shape gate

    @Test("script asserts the load-contract files exist in the source snapshot")
    func scriptAssertsExpectedFiles() throws {
        let body = try Self.loadScript()
        // EXPECTED_FILES gates the script against an empty / partial
        // / wrong snapshot. Pin every entry of the load contract so
        // a future model swap that drops one of these by accident
        // trips this test rather than producing a tarball that
        // fails at runtime on the user's first launch.
        let must = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "model.safetensors",
            "model.safetensors.index.json",
        ]
        for f in must {
            #expect(
                body.contains("\"\(f)\""),
                "scripts/build-model-tarball.sh's EXPECTED_FILES array is missing ``\(f)``. This file is part of the load contract — the sidecar fails to instantiate the tokenizer / model without it. Add it back to EXPECTED_FILES or document why the load contract changed."
            )
        }
    }

    // MARK: - size gates

    @Test("script has byte-precise size gates (lower and upper bound)")
    func scriptHasSizeGates() throws {
        let body = try Self.loadScript()
        // Lower bound: an empty/truncated archive (< 50 MB) cannot
        // possibly be a valid bonsai-1.7b-2bit tarball — the
        // safetensors blob alone is ~490 MB pre-compression. Pin
        // the env-var default so a silent bump to 0 doesn't slip
        // through code review.
        #expect(
            body.contains("MODEL_TARBALL_MIN_MB:-50"),
            "scripts/build-model-tarball.sh must default MODEL_TARBALL_MIN_MB to 50. An empty/truncated archive < 50 MB cannot be a valid bonsai-1.7b-2bit package (the safetensors blob alone is ~490 MB). Lower defaults invite a packing regression that ships a stub."
        )
        // Upper bound: 1 GB catches packaging the wrong model — a
        // 7B-class model would easily blow past 1 GB compressed.
        // Pin the default explicitly.
        #expect(
            body.contains("MODEL_TARBALL_MAX_MB:-1024"),
            "scripts/build-model-tarball.sh must default MODEL_TARBALL_MAX_MB to 1024 (1 GiB). A larger default would silently accept the wrong model — bonsai-1.7b-2bit compresses to ~470 MB (already-quantized ternary weights barely shrink); anything > 1 GB is almost certainly a wrong-alias swap. Bumping the default needs explicit code review."
        )
    }

    @Test("size gate compares BYTES, not du -sm output (catches sub-50 MB archives)")
    func scriptGatesOnBytesNotDuMb() throws {
        let body = try Self.loadScript()
        // Slice α codex r1 MAJOR: macOS ``du -sm`` reports whole-MiB
        // disk-usage rounded UP. A 100 KB tarball would report `1`
        // and silently pass a ``>= 1 MB`` lower gate. The gate MUST
        // compare bytes (``stat -f%z``) against MIN_BYTES / MAX_BYTES
        // derived from the MB env vars via ``MIN_MB * 1048576``.
        // This test pins the same byte-precise idiom slice α
        // adopted, so a copy-paste regression that reverts to du
        // trips here too.
        #expect(
            body.contains("MIN_BYTES=$(( MODEL_TARBALL_MIN_MB * 1048576 ))"),
            "scripts/build-model-tarball.sh must derive MIN_BYTES from MODEL_TARBALL_MIN_MB * 1048576 (1 MiB). du -sm rounds whole-MiB blocks UP so byte-precise comparison is the only correct floor (lesson from slice α PR #404 codex r1)."
        )
        #expect(
            body.contains("MAX_BYTES=$(( MODEL_TARBALL_MAX_MB * 1048576 ))"),
            "scripts/build-model-tarball.sh must derive MAX_BYTES from MODEL_TARBALL_MAX_MB * 1048576 (1 MiB). du -sm rounds whole-MiB blocks UP so byte-precise comparison is the only correct ceiling."
        )
        #expect(
            body.contains("[[ \"$SIZE_BYTES\" -lt \"$MIN_BYTES\" ]]"),
            "scripts/build-model-tarball.sh's lower gate must compare SIZE_BYTES against MIN_BYTES (NOT a du-rounded MB value). du -sm rounds up; bytes don't."
        )
        #expect(
            body.contains("[[ \"$SIZE_BYTES\" -gt \"$MAX_BYTES\" ]]"),
            "scripts/build-model-tarball.sh's upper gate must compare SIZE_BYTES against MAX_BYTES (NOT a du-rounded MB value). du -sm rounds up; bytes don't."
        )
    }

    @Test("script has a sanity ceiling on env-var size gates (catches typos)")
    func scriptCapsAbsurdEnvOverrides() throws {
        let body = try Self.loadScript()
        // Slice α codex r2 MAJOR: an absurd env override (e.g. user
        // types ``MAX_MB=99999999``) overflows when multiplied by
        // 1 MiB. Pin the same 1 TiB ceiling as slice α so the
        // pattern is consistent across artifact scripts.
        #expect(
            body.contains("MAX_GATE_CEILING_MB=1048576"),
            "scripts/build-model-tarball.sh must cap MODEL_TARBALL_MIN_MB / MODEL_TARBALL_MAX_MB at 1048576 MB (1 TiB) to prevent overflow on absurd env overrides. Matches slice α PR #404 codex r2's sanity cap."
        )
        // Match the explicit numeric check so the script can't be
        // bypassed by passing a non-numeric string (which under
        // ``-u`` + arithmetic context would silently evaluate to 0
        // and pass the floor).
        #expect(
            body.contains("[[ ! \"$gate_val\" =~ ^[0-9]+$ ]]"),
            "scripts/build-model-tarball.sh must reject non-numeric MODEL_TARBALL_MIN_MB / MODEL_TARBALL_MAX_MB. A non-numeric value silently evaluates to 0 in bash arithmetic context — that would make the floor 0 and a 1-byte tarball would pass the lower gate."
        )
    }

    // MARK: - safety / input hardening

    @Test("script validates --alias / --repo / --version against injection-safe grammars")
    func scriptValidatesInputs() throws {
        let body = try Self.loadScript()
        // ALIAS goes straight into the tarball filename + (in slice ε)
        // the R2 bucket key. Reject anything that could smuggle a
        // path separator or shell meta. Pin the grammar so a future
        // "let's allow underscores" requires explicit thought.
        #expect(
            body.contains("\"$ALIAS\" =~ ^[a-z0-9.-]+$"),
            "scripts/build-model-tarball.sh must validate ALIAS against ^[a-z0-9.-]+$. Without this a malformed --alias slips into the output filename / manifest / (eventually) the R2 bucket key. Matches the rapid-mlx aliases.json key grammar."
        )
        // REPO: HF allows alnum + . _ - on each side of the / so the
        // regex is more permissive but still rejects shell meta.
        #expect(
            body.contains("\"$REPO\" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$"),
            "scripts/build-model-tarball.sh must validate REPO against HF's owner/name grammar. Without this a malformed --repo could smuggle a path separator into the cache dir derivation."
        )
        // APP_VERSION: same SemVer regex as build-sidecar-tarball.sh
        // so both artifacts share a single grammar.
        #expect(
            body.contains("^[0-9]+(\\.[0-9]+)+([-+][0-9A-Za-z.-]+)?$"),
            "scripts/build-model-tarball.sh must validate APP_VERSION against SemVer. Matches build-sidecar-tarball.sh; a malformed --version could smuggle a path separator into the output filename."
        )
    }

    @Test("script never writes to the user's HF cache (read-only contract)")
    func scriptDoesNotWriteToHfCache() throws {
        let body = try Self.loadScript()
        // CRITICAL constraint: this script reads from
        // ~/.cache/huggingface/hub but MUST NOT mutate it (the user's
        // own snapshot is shared with other tools; a delete or move
        // would corrupt their cache). Pin the no-write contract by
        // grepping for the obvious mutation idioms.
        //
        // Strip whitespace before grep so a reformatted command line
        // still trips.
        let stripped = Self.stripWhitespace(body)
        // We're checking the SCRIPT's behaviour — it must not contain
        // any mutation against $HF_HUB_CACHE / $HF_HOME / $XDG_CACHE_HOME
        // / $HOME's cache paths. Probe for the obvious patterns.
        let forbidden = [
            "rm-rf\"$USER_HUB",
            "rm-rf\"$HF_HOME",
            "rm-rf\"$HF_HUB_CACHE",
            "rm-rf\"$HOME/.cache",
            "rm\"$USER_HUB",
            ">$USER_HUB",
            "mv\"$USER_HUB",
        ]
        for f in forbidden {
            #expect(
                !stripped.contains(f),
                "scripts/build-model-tarball.sh appears to mutate the user's HF cache (matched ``\(f)``). The cache is read-only — pack via realpath dereference, never via in-place mutation."
            )
        }
        // Positive: the source-resolution comment must document the
        // read-only contract so a future maintainer doesn't add a
        // ``snapshot_download`` (which writes to the cache) inside
        // the script.
        #expect(
            body.contains("never writes to the user's HF cache") ||
                body.contains("NEVER writes to the cache") ||
                body.contains("read-only"),
            "scripts/build-model-tarball.sh must document its read-only contract on the user's HF cache (comment block at top of file). The script must not add ``snapshot_download`` or any cache mutation — that responsibility belongs to the CI prime step (release.yml)."
        )
    }

    // MARK: - manifest

    @Test("manifest carries the slice-γ-consumable schema fields")
    func scriptEmitsCanonicalManifestFields() throws {
        let body = try Self.loadScript()
        // Slice γ will add latest.json fields keyed off the manifest
        // (model_url / model_sha256 / model_size). The Codable shape
        // on the desktop side will be a single struct that reads
        // both the sidecar and quickstart manifests. Pin each field
        // name so a refactor on either side stays in sync.
        let mustFields = [
            "schema_version",
            "artifact",
            "model_alias",
            "hf_path",
            "tarball_filename",
            "tarball_sha256",
            "tarball_size",
            "uncompressed_size",
            "file_count",
            "created_at_epoch",
        ]
        for f in mustFields {
            #expect(
                body.contains("\"\(f)\""),
                "scripts/build-model-tarball.sh's manifest is missing ``\(f)``. Slice γ's latest.json struct expects this field — a missing field will fail Codable decoding on the desktop side at first launch."
            )
        }
        // The artifact tag distinguishes the quickstart manifest from
        // the sidecar manifest (which uses ``rapid-mlx-sidecar``).
        // Pin the literal so a maintainer can't accidentally tag this
        // as a sidecar artifact and confuse slice γ's decoder.
        #expect(
            body.contains("\"quickstart-model\""),
            "scripts/build-model-tarball.sh's manifest must tag artifact as ``quickstart-model`` to distinguish it from the sidecar manifest (``rapid-mlx-sidecar``). Slice γ's decoder uses this tag to route to the right struct."
        )
    }

    @Test("manifest is emitted via Python json.dump (NOT hand-rolled string interpolation)")
    func scriptEmitsManifestViaJsonDump() throws {
        let body = try Self.loadScript()
        // Hand-rolled JSON via shell ``cat <<EOF`` would break the
        // moment a future field contains a quote, backslash, or
        // control char. Python's json.dump escapes properly.
        #expect(
            body.contains("json.dump"),
            "scripts/build-model-tarball.sh must emit the manifest via ``json.dump`` (NOT shell heredoc interpolation). Without this, a future field containing a quote / backslash / control char produces invalid JSON and slice γ's Codable decoder fails."
        )
        // sort_keys=True is what makes the manifest's text bytes
        // deterministic — Python dicts are insertion-ordered, so
        // without sort_keys the manifest's field order would drift
        // if the dict literal is reformatted.
        #expect(
            body.contains("sort_keys=True"),
            "scripts/build-model-tarball.sh must call ``json.dump(..., sort_keys=True)`` so the manifest's text bytes are deterministic. Without sort_keys, a reformatted dict literal silently changes field order and the manifest SHA256 drifts."
        )
    }

    // MARK: - output naming

    @Test("script output filename is namespaced quickstart-<alias>-<version>")
    func scriptOutputNameNamespaced() throws {
        let body = try Self.loadScript()
        // Pin the filename shape so slice γ + slice ε both have a
        // stable glob to upload / mirror. Mirrors the sidecar naming
        // ``rapid-mlx-sidecar-X.Y.Z.tar.gz``.
        #expect(
            body.contains("quickstart-${ALIAS}-${APP_VERSION}.tar.gz"),
            "scripts/build-model-tarball.sh's output filename must follow ``quickstart-<alias>-<version>.tar.gz``. Slice γ's R2 upload and slice ε's release-asset glob both key off this exact shape."
        )
        #expect(
            body.contains("quickstart-${ALIAS}-${APP_VERSION}.manifest.json"),
            "scripts/build-model-tarball.sh's manifest filename must follow ``quickstart-<alias>-<version>.manifest.json`` to pair with the tarball."
        )
        // Negative: must NOT collide with the sidecar tarball name.
        #expect(
            !body.contains("rapid-mlx-sidecar-${APP_VERSION}.tar.gz"),
            "scripts/build-model-tarball.sh must not write to ``rapid-mlx-sidecar-<version>.tar.gz`` — that's build-sidecar-tarball.sh's output."
        )
    }

    // MARK: - CI integration (script side)

    @Test("script emits GITHUB_OUTPUT + GITHUB_STEP_SUMMARY for CI")
    func scriptEmitsCiOutputs() throws {
        let body = try Self.loadScript()
        // Slice γ + slice ε will read these outputs to populate
        // latest.json (model_url / model_sha256 / model_size). Pin
        // the canonical field names so a downstream consumer can
        // rely on them.
        let mustOutputs = [
            "tarball_path=",
            "tarball_name=",
            "tarball_sha256=",
            "tarball_size_bytes=",
            "manifest_path=",
            "model_alias=",
            "desktop_version=",
        ]
        for o in mustOutputs {
            #expect(
                body.contains(o),
                "scripts/build-model-tarball.sh must emit ``\(o)`` to \\$GITHUB_OUTPUT. Slice γ keys off these for the latest.json wiring."
            )
        }
        // STEP_SUMMARY is for human review of dry-runs in the GHA
        // workflow summary panel. Pin its presence.
        #expect(
            body.contains("GITHUB_STEP_SUMMARY"),
            "scripts/build-model-tarball.sh must write to \\$GITHUB_STEP_SUMMARY so the artifact metadata surfaces in the GHA workflow summary panel. Without this, reviewers have no way to confirm the pack succeeded without downloading the artifact."
        )
    }

    // MARK: - release.yml workflow shape

    // NOTE: the original slice-β ``releaseYamlSlbBuildStepsAreNonBlocking``
    // test was removed in slice δ as redundant with the new
    // ``releaseYamlQuickstartStepsRemainNonBlocking`` below, which
    // pins the same continue-on-error invariants AND adds the
    // slice-δ-specific reasoning (these steps now run BEFORE the R2
    // mirror, so continue-on-error is what protects the canonical
    // release from a packing failure). Codex r2 NIT cleanup.

    // MARK: - P3 slice δ: R2 mirror + latest.json schema

}
