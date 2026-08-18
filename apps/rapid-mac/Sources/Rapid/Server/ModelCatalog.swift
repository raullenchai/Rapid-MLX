import Foundation

/// What a model is *for*. Drives the capability tabs in Model Management —
/// chat, image, and audio models are managed side by side but never mixed in
/// one list (and are picked in different surfaces). Video is reserved for
/// when the video lane surfaces manageable aliases.
enum ModelKind: String, Sendable, Hashable, CaseIterable, Identifiable {
    case chat, image, audio, video
    var id: String { rawValue }
    /// Tab label in Model Management.
    var tabLabel: String {
        switch self {
        case .chat: return "Chat"
        case .image: return "Image"
        case .audio: return "Audio"
        case .video: return "Video"
        }
    }
}

/// The user-facing operation an audio checkpoint can actually perform.
/// The engine's registry intentionally groups forced alignment under `stt`
/// and several reference-driven models under `tts`; the desktop needs the
/// finer distinction so its simple transcription and preset-voice pickers do
/// not offer models that require inputs those flows do not collect.
enum AudioModelCapability: String, Sendable, Hashable {
    case transcription
    case alignment
    case speech
    case voiceCloning
    case voiceDesign

    var supportsTranscription: Bool { self == .transcription }
    var supportsPresetSpeech: Bool { self == .speech }
}

/// The request shape an image checkpoint accepts. This comes from the CLI's
/// explicit CLI capability tag, never from UI name matching.
enum ImageModelCapability: String, Sendable, Hashable {
    case generation
    case editing
    case generationAndEditing

    var supportsGeneration: Bool { self != .editing }
    var supportsEditing: Bool { self != .generation }
    var label: String {
        switch self {
        case .generation: return "Image generation"
        case .editing: return "Image editing"
        case .generationAndEditing: return "Image generation and editing"
        }
    }
}

/// Audited speculative-decoding preset advertised by the alias registry.
/// `nil` on ``ModelEntry`` means the alias explicitly has no usable preset
/// (or an older sidecar did not advertise one), so Settings fails closed.
struct SpeculativeDecodingPreset: Codable, Sendable, Hashable {
    enum Method: String, Codable, Sendable, Hashable { case suffix, mtp }
    let method: Method
    let model: String?
    let tokens: Int?

    var displayName: String { method == .mtp ? "MTP" : "Suffix decoding" }
}

/// One model in the rapid-mlx catalog. The picker UI groups cached vs.
/// uncached so the user knows which aliases boot instantly vs. which
/// trigger an HF download on first ``serve``.
struct ModelEntry: Identifiable, Hashable, Sendable {
    /// rapid-mlx alias (the string passed to ``rapid-mlx serve <alias>``).
    /// Always non-empty and unique within a catalog.
    let alias: String
    /// HF repo this alias resolves to, when known. Surfaced as a caption
    /// under the alias in the picker.
    let hfRepo: String?
    /// Size on disk, only set for entries discovered via ``rapid-mlx ls``.
    /// Shown as a right-aligned column in the picker.
    let sizeOnDisk: String?
    /// True when the alias is in ``rapid-mlx ls`` (downloaded already).
    /// Drives a green dot in the picker so the user can tell at a glance
    /// which models start in seconds vs. which trigger a 5-80 GB pull.
    let cached: Bool
    /// True for a model another MLX runtime downloaded, found outside the
    /// hub cache (#1718).
    ///
    /// Such a model is listed and usable, but must never be offered for
    /// deletion: the delete path rebuilds ``<hub-root>/models--<repo>``,
    /// which is not where it lives, so the delete would either miss or
    /// remove an unrelated hub entry of the same name. We did not download
    /// it and cannot manage it.
    var isExternal: Bool = false

    /// What the model is for. Defaults to ``.chat`` so every existing
    /// construction site keeps working; the image catalog tags ``.image``.
    var kind: ModelKind = .chat

    /// Audio-only metadata parsed from the engine registry table. `nil` for
    /// chat/image/video rows.
    var audioCapability: AudioModelCapability? = nil
    var audioFamily: String? = nil

    /// Image-only operation metadata. `nil` for chat/audio/video rows.
    var imageCapability: ImageModelCapability? = nil

    /// Chat-only speculative preset parsed from the engine's alias SSOT.
    var speculativeDecodingPreset: SpeculativeDecodingPreset? = nil

    var id: String { alias }
}

/// Loads the rapid-mlx alias catalog by shelling out to the CLI. The
/// picker depends on this *before* the server is spawned, so we can't
/// use ``GET /v1/models``; the text output of ``rapid-mlx models`` and
/// ``rapid-mlx ls`` is the cheapest source.
///
/// The Tauri reference at ``archive/tauri-v0.1`` parsed the same two
/// commands; the Swift port keeps the parsing centralised here so the
/// picker view stays a thin shell.
///
/// Thread model: ``load(binary:)`` is an async API that fans out to two
/// short-lived subprocesses concurrently. Cancellation propagates to the
/// children via ``Task.checkCancellation`` between phases.
enum ModelCatalog {
    static let maxAliasBytes = 128
    static let maxHuggingFaceRepoBytes = 192
    static let maxSubprocessStdoutBytes = 1_048_576
    /// Tiny Whisper is fast, but its transcription accuracy is below the
    /// desktop workflow's quality floor. Keep the engine alias available to
    /// CLI users while omitting it from the GUI catalog.
    private static let hiddenDesktopAudioAliases: Set<String> = ["whisper-tiny"]
    private static let maxSubprocessStderrBytes = 256 * 1024
    private static let pipeReadChunkBytes = 16 * 1024

    /// Engine env var naming the directories to scan for models another MLX
    /// runtime downloaded (a JSON string array; the engine also accepts the
    /// legacy ``os.pathsep`` representation from shells and older builds).
    ///
    /// Kept as a named constant because it is a cross-process contract with
    /// ``vllm_mlx.cli._external_model_roots`` — a typo on either side fails
    /// silently as "no models found", which reads as an empty disk rather
    /// than as a broken lookup.
    static let extraModelRootsEnvKey = "RAPID_MLX_EXTRA_MODEL_ROOTS"

    /// Merge an explicit Settings folder with any roots inherited from the
    /// launcher. Root order is precedence order, so ambient roots stay first
    /// and the selected folder is appended only when it is not already the
    /// same canonical directory.
    static func mergedExtraModelRoots(existing: String?, selected: String?) -> String? {
        var roots: [String] = []
        var seen: Set<String> = []
        let inherited: [String] = {
            guard let existing, !existing.isEmpty else { return [] }
            if let data = existing.data(using: .utf8),
               let decoded = try? JSONSerialization.jsonObject(with: data) as? [String] {
                return decoded
            }
            return existing.split(separator: ":").map(String.init)
        }()
        let candidates = inherited + [selected].compactMap { $0 }
        for candidate in candidates {
            let trimmed = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            let canonical = URL(fileURLWithPath: trimmed)
                .standardizedFileURL.resolvingSymlinksInPath().path
            guard seen.insert(canonical).inserted else { continue }
            roots.append(canonical)
        }
        guard !roots.isEmpty,
              let data = try? JSONSerialization.data(withJSONObject: roots),
              let encoded = String(data: data, encoding: .utf8) else { return nil }
        return encoded
    }

    /// All known aliases plus their installation status. Empty array on
    /// any failure — the caller should fall back to a plain text field.
    /// We deliberately swallow errors here rather than throwing because
    /// a missing binary / malformed catalog should never block the user
    /// from typing a custom alias.
    ///
    /// ``hubCacheOverride`` (issue #503) points the ``rapid-mlx ls``
    /// probe at the user's chosen models folder so the ``cached`` /
    /// size-on-disk columns reflect what's actually in the folder the
    /// engine reads from. Defaults to the validated "Models folder"
    /// preference so every catalog surface (picker checkmarks, Model
    /// Management, upgrade nudges) stays consistent with the engine
    /// without each call site having to thread it; ``nil`` inherits the
    /// ambient environment (the default location). Tests pass an
    /// explicit value to pin behaviour without touching UserDefaults.
    static func load(
        binary: URL,
        hubCacheOverride: URL? = ModelsFolderPreference.validatedOverrideURL()
    ) async -> [ModelEntry] {
        async let availableTask: (
            entries: [(String, String?)],
            excluded: Set<String>,
            speculative: [String: SpeculativeDecodingPreset]
        ) =
            listAvailableWithExclusions(binary: binary)
        async let cachedTask: [(String, String?, String?)] = listCached(
            binary: binary,
            hubCacheOverride: hubCacheOverride
        )

        let (available, excludedAliases, speculativeCapabilities) = await availableTask
        let cached = await cachedTask

        var entries = mergeAvailableAndCached(
            available: available,
            cached: cached,
            excluded: excludedAliases
        )

        // Repo-aware cached marking (issue #576): a bare alias like
        // ``qwen3-0.6b`` and its default-quant alias ``qwen3-0.6b-4bit``
        // are two catalog rows that resolve to the SAME HF repo, but
        // ``rapid-mlx ls`` only reports the quant-suffixed one. Matching
        // ``cached`` by exact alias string above therefore left the bare
        // alias marked uncached — the picker hid its cached dot and
        // launch-time auto-start (which persists the bare alias as
        // ``lastServedAlias``) kicked off a spurious 0-byte "re-download"
        // on every relaunch. Reconcile by HF repo: resolve the bare
        // siblings via ``rapid-mlx info`` (bounded to aliases that are a
        // base-prefix of a cached alias, so no-cache paths spawn zero
        // extra subprocesses) and re-mark them cached when the repo
        // matches a cached row.
        entries = await remarkSiblingsCachedByRepo(entries, binary: binary)

        // Apply capability metadata after cache reconciliation, whose pure
        // rebuilds intentionally know nothing about the CLI profile table.
        entries = entries.map { entry in
            var enriched = entry
            enriched.speculativeDecodingPreset = speculativeCapabilities[entry.alias]
            return enriched
        }

        // Sort: cached first, then alphabetic within each group. The
        // user is most likely to pick something they already have on
        // disk.
        entries.sort { lhs, rhs in
            if lhs.cached != rhs.cached { return lhs.cached && !rhs.cached }
            return lhs.alias.localizedStandardCompare(rhs.alias) == .orderedAscending
        }
        return entries
    }

    // MARK: - Repo-aware cache reconciliation (#576)

    /// Bounded IO step used by ``load``: for each uncached catalog entry
    /// that is a base-prefix of a cached alias (its default-quant
    /// sibling), resolve its HF repo via ``rapid-mlx info`` and re-mark
    /// it cached when the repo matches a cached row. Returns ``entries``
    /// unchanged when there are no such candidates (the common fresh /
    /// no-cache path) so we never spawn ``info`` for nothing. The
    /// candidate probes fan out concurrently — wall-clock is one ``info``
    /// call, not the sum.
    private static func remarkSiblingsCachedByRepo(
        _ entries: [ModelEntry],
        binary: URL
    ) async -> [ModelEntry] {
        let candidates = siblingCandidateAliases(entries)
        guard !candidates.isEmpty else { return entries }

        var resolved: [String: String] = [:]
        await withTaskGroup(of: (String, String?).self) { group in
            for alias in candidates {
                group.addTask { (alias, await resolveRepo(binary: binary, alias: alias)) }
            }
            for await (alias, repo) in group {
                if let repo { resolved[alias] = repo }
            }
        }
        guard !resolved.isEmpty else { return entries }
        return remarkCachedByRepo(entries, resolvedRepos: resolved)
    }

    /// Pure: uncached aliases worth an ``info`` probe — those that are a
    /// strict base-prefix of some cached alias (e.g. ``qwen3-0.6b`` when
    /// ``qwen3-0.6b-4bit`` is cached). Excludes aliases already cached
    /// and de-duplicates. Kept separate from the IO so the candidate
    /// rule is unit-testable without a sidecar.
    ///
    /// The base-prefix rule only *narrows* which aliases we probe; the
    /// authoritative decision is still the HF-repo equality check in
    /// ``remarkCachedByRepo``, so a sibling that resolves to a different
    /// quant's repo (``qwen3-0.6b`` → ``…-4bit`` while only ``…-8bit`` is
    /// cached) is probed but correctly left uncached — no false positive.
    static func siblingCandidateAliases(_ entries: [ModelEntry]) -> [String] {
        let cachedAliases = entries.filter { $0.cached }.map(\.alias)
        guard !cachedAliases.isEmpty else { return [] }
        var seen: Set<String> = []
        var out: [String] = []
        for entry in entries where !entry.cached {
            let alias = entry.alias
            guard !seen.contains(alias) else { continue }
            if cachedAliases.contains(where: { $0.hasPrefix(alias + "-") }) {
                seen.insert(alias)
                out.append(alias)
            }
        }
        return out
    }

    /// Pure: re-mark uncached entries whose resolved HF repo equals a
    /// cached entry's repo. The rebuilt entry carries the cached repo +
    /// size so the picker caption / size column match the sibling that
    /// is actually on disk. ``resolvedRepos`` maps alias → HF repo.
    /// Matching is exact on the sanitized repo string — never
    /// case-folded — so two repos differing only by case are never
    /// merged.
    static func remarkCachedByRepo(
        _ entries: [ModelEntry],
        resolvedRepos: [String: String]
    ) -> [ModelEntry] {
        var cachedByRepo: [String: (repo: String, size: String?)] = [:]
        for entry in entries where entry.cached {
            if let repo = sanitizedHuggingFaceRepo(entry.hfRepo) {
                cachedByRepo[repo] = (repo, entry.sizeOnDisk)
            }
        }
        guard !cachedByRepo.isEmpty else { return entries }

        return entries.map { entry in
            guard !entry.cached,
                  let raw = resolvedRepos[entry.alias],
                  let repo = sanitizedHuggingFaceRepo(raw),
                  let hit = cachedByRepo[repo]
            else { return entry }
            return ModelEntry(
                alias: entry.alias,
                hfRepo: hit.repo,
                sizeOnDisk: hit.size,
                cached: true
            )
        }
    }

    /// Resolves a single alias to its HF repo via ``rapid-mlx info``.
    /// Returns nil on any failure (missing binary, unknown alias,
    /// unparseable output) — the caller then leaves the entry uncached,
    /// preserving the pre-#576 behaviour for that alias.
    private static func resolveRepo(binary: URL, alias: String) async -> String? {
        guard isSafeAlias(alias) else { return nil }
        let output = await runRapidMlx(binary: binary, args: ["info", alias])
        return parseInfoRepo(output)
    }

    /// Pure parser for ``rapid-mlx info <alias>`` stdout. Extracts the HF
    /// repo from the ``Alias: <alias> → <repo>`` line (both the U+2192
    /// arrow and an ASCII ``->`` are accepted). Returns nil when no such
    /// line is present or the repo fails ``sanitizedHuggingFaceRepo``.
    static func parseInfoRepo(_ output: String) -> String? {
        for rawLine in output.split(separator: "\n", omittingEmptySubsequences: true) {
            let line = String(rawLine)
            guard line.contains("Alias:") else { continue }
            let arrow: String
            if line.contains("→") {
                arrow = "→"
            } else if line.contains("->") {
                arrow = "->"
            } else {
                continue
            }
            guard let range = line.range(of: arrow, options: .backwards) else { continue }
            let tail = line[range.upperBound...].trimmingCharacters(in: .whitespaces)
            if let repo = sanitizedHuggingFaceRepo(tail) {
                return repo
            }
        }
        return nil
    }

    // MARK: - Parsing helpers

    /// Runs ``rapid-mlx models`` and parses the column-aligned text
    /// output. Returns ``(alias, hfRepo)`` pairs — ``hfRepo`` is unset
    /// because the bare ``models`` listing doesn't include it (cached
    /// rows do, but available rows don't). Empty array on any failure.
    private static func listAvailable(binary: URL) async -> [(String, String?)] {
        let output = await runRapidMlx(binary: binary, args: ["models"])
        return parseAvailable(output)
    }

    /// Merges the ``models`` and ``ls`` listings into catalog rows.
    ///
    /// Pure so the exclusion rule is testable without spawning the
    /// engine: the decisive condition is that a cached alias which was
    /// deliberately withheld from ``models`` must NOT be re-admitted here.
    static func mergeAvailableAndCached(
        available: [(String, String?)],
        cached: [(String, String?, String?)],
        excluded: Set<String>
    ) -> [ModelEntry] {
        var cachedIndex: [String: (hfRepo: String?, size: String?)] = [:]
        for (alias, hf, size) in cached where !alias.isEmpty && !isStatusAlias(alias) {
            cachedIndex[alias] = (hf, size)
        }
        var externalIndex: [String: (hfRepo: String?, size: String?)] = [:]
        for (alias, hf, size) in cached where alias == "(external)" {
            guard let identifier = hf else { continue }
            externalIndex[identifier] = (hf, size)
        }

        var entries: [ModelEntry] = []
        var seenAliases: Set<String> = []
        var consumedExternal: Set<String> = []
        for (alias, hfHint) in available {
            seenAliases.insert(alias)
            let cachedHit = cachedIndex[alias]
            let externalIdentifier: String? = {
                if externalIndex[alias] != nil,
                   !consumedExternal.contains(alias) { return alias }
                if let hfHint, externalIndex[hfHint] != nil,
                   !consumedExternal.contains(hfHint) { return hfHint }
                return nil
            }()
            if let externalIdentifier { consumedExternal.insert(externalIdentifier) }
            entries.append(ModelEntry(
                alias: alias,
                hfRepo: cachedHit?.hfRepo ?? hfHint ?? externalIdentifier,
                sizeOnDisk: cachedHit?.size
                    ?? externalIdentifier.flatMap { externalIndex[$0]?.size },
                cached: cachedHit != nil || externalIdentifier != nil,
                isExternal: cachedHit == nil && externalIdentifier != nil
            ))
        }
        // A cached model with no row in ``rapid-mlx models`` is unusual
        // but possible if the user pinned an alias by hand in their
        // rapid-mlx config. Surface them anyway so they show up in the
        // picker (otherwise the user can't pick them without typing) —
        // except for the ones ``parseAvailable`` deliberately withheld.
        // ``rapid-mlx ls`` has no modality tag, so without this check a
        // cached audio or video model has no row in ``models`` for
        // exactly the reason it must stay hidden, and would be re-admitted
        // here on that basis (#1603).
        for (alias, hf, size) in cached
        where !alias.isEmpty
            && !isStatusAlias(alias)
            && !seenAliases.contains(alias)
            && !excluded.contains(alias) {
            entries.append(ModelEntry(
                alias: alias,
                hfRepo: hf,
                sizeOnDisk: size,
                cached: true
            ))
        }

        // Models another MLX runtime downloaded (#1718). These arrive with
        // ``(external)`` in the alias column — a status marker, not a name —
        // so the repo is the only identifier they have, and it is what
        // ``serve`` accepts for them.
        //
        // They are admitted so the user can SEE and USE a model already on
        // disk; that is the entire point of the issue. What they must not be
        // is deletable, which ``isExternal`` conveys to the UI. Dropping them
        // here instead would satisfy "not deletable" by making them invisible
        // — and leave the user re-downloading weights they already have.
        for (alias, hf, size) in cached
        where alias == "(external)" {
            guard let repo = hf,
                  !consumedExternal.contains(repo),
                  !seenAliases.contains(repo),
                  !excluded.contains(repo) else { continue }
            seenAliases.insert(repo)
            entries.append(ModelEntry(
                alias: repo,
                hfRepo: repo,
                sizeOnDisk: size,
                cached: true,
                isExternal: true
            ))
        }
        return entries
    }

    /// Whether the alias column holds a status marker rather than a name.
    ///
    /// These rows carry a real repo but must never become a `ModelEntry`:
    /// an entry is addressed by alias and, once `cached`, is offered for
    /// deletion — and deletion rebuilds `<hub-root>/models--<repo>`.
    /// `(external)` (#1718) lives outside that root entirely, so admitting
    /// one would offer a delete that either silently misses or removes an
    /// unrelated hub entry of the same name.
    static func isStatusAlias(_ alias: String) -> Bool {
        alias.hasPrefix("(") && alias.hasSuffix(")")
    }

    /// Runs ``rapid-mlx models`` and returns both the chat-capable rows
    /// and the aliases deliberately withheld from them.
    ///
    /// ``rapid-mlx ls`` carries no modality tag, so filtering
    /// ``parseAvailable`` alone is not enough: ``load`` re-admits any
    /// cached alias that has no row in ``models``, which would hand a
    /// cached audio or video model straight back to the picker through
    /// the side door. Pairing the two parses closes that (#1603).
    private static func listAvailableWithExclusions(
        binary: URL
    ) async -> (
        entries: [(String, String?)],
        excluded: Set<String>,
        speculative: [String: SpeculativeDecodingPreset]
    ) {
        let output = await runRapidMlx(binary: binary, args: ["models"])
        return (
            parseAvailable(output),
            parseExcludedAliases(output),
            parseSpeculativeCapabilities(output)
        )
    }

    /// Image aliases with explicit generation/edit capabilities for the Images tab's
    /// model picker. Parsed from the same ``rapid-mlx models`` output the
    /// chat catalog reads, but keeping ONLY the image rows the chat catalog
    /// deliberately excludes. ``cached`` is resolved by cross-referencing
    /// ``rapid-mlx ls`` on HF repo id, so the picker can show which image
    /// models boot instantly vs. which trigger a multi-GB pull.
    static func imageEntries(
        binary: URL,
        hubCacheOverride: URL? = ModelsFolderPreference.validatedOverrideURL()
    ) async -> [ModelEntry] {
        async let modelsOut = runRapidMlx(binary: binary, args: ["models"])
        async let cachedTask: [(String, String?, String?)] = listCached(
            binary: binary,
            hubCacheOverride: hubCacheOverride
        )
        let rows = parseImageRows(await modelsOut)
        let cachedRepos = Set((await cachedTask).compactMap { $0.1 })
        return mergeImageRows(rows, cachedRepos: cachedRepos)
    }

    /// Join the engine's image catalog to its runnable-cache view. Keeping
    /// this seam pure pins the cross-process contract: component-layout
    /// mflux snapshots reported by `rapid-mlx ls` must reach the Images UI as
    /// cached even though they intentionally have no root `config.json`.
    static func mergeImageRows(
        _ rows: [(
            alias: String,
            hfRepo: String?,
            size: String?,
            capability: ImageModelCapability
        )],
        cachedRepos: Set<String>
    ) -> [ModelEntry] {
        return rows.map { row in
            ModelEntry(
                alias: row.alias,
                hfRepo: row.hfRepo,
                sizeOnDisk: row.size,
                cached: row.hfRepo.map { cachedRepos.contains($0) } ?? false,
                kind: .image,
                imageCapability: row.capability
            )
        }
    }

    /// Audio aliases for Settings and the dedicated Audio surface. Cached
    /// state is matched by HF repo because `rapid-mlx ls` currently reports
    /// audio snapshots as `(unmapped)` rather than reverse-mapping the audio
    /// registry's alias.
    static func audioEntries(
        binary: URL,
        hubCacheOverride: URL? = ModelsFolderPreference.validatedOverrideURL()
    ) async -> [ModelEntry] {
        async let modelsOut = runRapidMlx(binary: binary, args: ["models"])
        async let cachedTask: [(String, String?, String?)] = listCached(
            binary: binary,
            hubCacheOverride: hubCacheOverride
        )
        let rows = parseAudioRows(await modelsOut).filter {
            isDesktopAudioAliasVisible($0.alias)
        }
        let cachedByRepo = Dictionary(
            (await cachedTask).compactMap { alias, repo, size -> (String, String?)? in
                guard let repo else { return nil }
                return (repo, size)
            },
            uniquingKeysWith: { first, _ in first }
        )
        return rows.map { row in
            ModelEntry(
                alias: row.alias,
                hfRepo: row.hfRepo,
                sizeOnDisk: row.hfRepo.flatMap { cachedByRepo[$0] } ?? row.size,
                cached: row.hfRepo.map { cachedByRepo[$0] != nil } ?? false,
                kind: .audio,
                audioCapability: audioCapability(
                    alias: row.alias,
                    subtype: row.subtype,
                    family: row.family
                ),
                audioFamily: row.family
            )
        }
    }

    static func isDesktopAudioAliasVisible(_ alias: String) -> Bool {
        !hiddenDesktopAudioAliases.contains(alias)
    }

    /// Parse rows shaped as:
    /// `kokoro  338.9 MiB  [audio:tts]  kokoro  mlx-community/Kokoro...`.
    /// The family column is retained because the registry's broad `stt`/`tts`
    /// type does not distinguish aligners and reference-driven speech models.
    static func parseAudioRows(
        _ output: String
    ) -> [(alias: String, hfRepo: String?, size: String?, subtype: String, family: String?)] {
        var rows: [(String, String?, String?, String, String?)] = []
        for rawLine in output.split(separator: "\n", omittingEmptySubsequences: true) {
            let line = String(rawLine).trimmingCharacters(in: .whitespaces)
            let fields = line.split(whereSeparator: { $0.isWhitespace }).map(String.init)
            guard let alias = fields.first, isSafeAlias(alias),
                  let tagIdx = fields.firstIndex(where: {
                      $0.hasPrefix("[audio:") && $0.hasSuffix("]")
                  }) else { continue }
            let tag = fields[tagIdx]
            let subtype = String(tag.dropFirst("[audio:".count).dropLast())
            guard subtype == "tts" || subtype == "stt" else { continue }
            let family = tagIdx + 1 < fields.count ? fields[tagIdx + 1] : nil
            let hfRepo = tagIdx + 2 < fields.count
                ? sanitizedHuggingFaceRepo(fields[tagIdx + 2])
                : nil
            let size = tagIdx > 1 ? fields[1..<tagIdx].joined(separator: " ") : nil
            rows.append((alias, hfRepo, size, subtype, family))
        }
        return rows
    }

    /// Refine the registry's broad audio type into the operations the desktop
    /// currently exposes. Kept pure so capability filtering is regression
    /// tested without starting an audio engine.
    static func audioCapability(
        alias: String,
        subtype: String,
        family: String?
    ) -> AudioModelCapability {
        if subtype == "stt" {
            return family == "qwen3_aligner" ? .alignment : .transcription
        }
        let lower = alias.lowercased()
        if lower.contains("voicedesign") { return .voiceDesign }
        if family == "indextts" || lower == "qwen3-tts-clone" { return .voiceCloning }
        return .speech
    }

    /// Parse operation-tagged image rows into catalog metadata.
    /// Row shape (see cli.py image section):
    /// ``flux2-klein-4b  4.3 GiB  [image:both] Runpod/FLUX...`` or
    /// ``z-image-turbo  5.5 GiB  [image:gen] filipstrand/Z-Image...``.
    static func parseImageRows(
        _ output: String
    ) -> [(
        alias: String,
        hfRepo: String?,
        size: String?,
        capability: ImageModelCapability
    )] {
        var rows: [(String, String?, String?, ImageModelCapability)] = []
        for rawLine in output.split(separator: "\n", omittingEmptySubsequences: true) {
            let line = String(rawLine).trimmingCharacters(in: .whitespaces)
            let fields = line.split(whereSeparator: { $0.isWhitespace }).map(String.init)
            guard let alias = fields.first, isSafeAlias(alias),
                  let tagIdx = fields.firstIndex(where: {
                      $0 == "[image:gen]" || $0 == "[image:edit]"
                          || $0 == "[image:both]"
                  }) else { continue }
            let capability: ImageModelCapability
            switch fields[tagIdx] {
            case "[image:edit]": capability = .editing
            case "[image:both]": capability = .generationAndEditing
            default: capability = .generation
            }
            let hfRepo = tagIdx + 1 < fields.count ? fields[tagIdx + 1] : nil
            let size = tagIdx > 1 ? fields[1..<tagIdx].joined(separator: " ") : nil
            rows.append((alias, hfRepo, size, capability))
        }
        return rows
    }

    /// True when the line carries a non-chat Kind tag in its own column.
    ///
    /// Matching the bare substring ``"[audio:"`` would let any row whose
    /// HF id or description happened to contain those characters
    /// disappear from the catalog. Require a whole whitespace-delimited
    /// token of the shape the engine actually prints — ``[audio:tts]``,
    /// ``[audio:stt]``, ``[video:gen]`` — without hardcoding the
    /// subtypes, which the engine derives from its registries and may
    /// extend.
    static func hasNonChatKindTag(_ line: String) -> Bool {
        for field in line.split(whereSeparator: { $0.isWhitespace }) {
            guard field.hasPrefix("["), field.hasSuffix("]") else { continue }
            let body = field.dropFirst().dropLast()
            guard let colon = body.firstIndex(of: ":") else { continue }
            let kind = body[body.startIndex..<colon]
            let subtype = body[body.index(after: colon)...]
            guard kind == "audio" || kind == "video" || kind == "image" else { continue }
            guard !subtype.isEmpty, subtype.allSatisfy({ $0.isLetter || $0 == "-" }) else {
                continue
            }
            return true
        }
        return false
    }

    /// Aliases ``parseAvailable`` drops for being a non-chat modality.
    ///
    /// Deliberately narrow: only rows carrying an explicit engine-side
    /// Kind tag count. Banner lines, dividers and headers are noise, not
    /// exclusions, and must not end up suppressing a real model.
    static func parseExcludedAliases(_ output: String) -> Set<String> {
        var excluded: Set<String> = []
        for rawLine in output.split(separator: "\n", omittingEmptySubsequences: true) {
            let line = String(rawLine).trimmingCharacters(in: .whitespaces)
            guard hasNonChatKindTag(line) else { continue }
            let token = line.split(maxSplits: 1, whereSeparator: { $0.isWhitespace }).first
            guard let alias = token.map(String.init), isSafeAlias(alias) else { continue }
            excluded.insert(alias)
        }
        return excluded
    }

    /// Runs ``rapid-mlx ls`` (cached models). Returns
    /// ``(alias, hfRepo, sizeOnDisk)`` tuples. ``hubCacheOverride``
    /// (issue #503) points the probe at the user's chosen models folder
    /// so the listing reflects what's on the folder the engine reads
    /// from, not the default location.
    private static func listCached(
        binary: URL,
        hubCacheOverride: URL?
    ) async -> [(String, String?, String?)] {
        let output = await runRapidMlx(
            binary: binary,
            args: ["ls"],
            hubCacheOverride: hubCacheOverride
        )
        return parseCached(output)
    }

    /// Parses the ``rapid-mlx models`` output. The format (v0.6.83) is a
    /// header line, a divider, then rows like::
    ///
    ///     bonsai-1.7b            hermes           glm4         ✓          avoid       —
    ///
    /// Columns are space-aligned but with multiple internal spaces, so a
    /// simple ``components(separatedBy: " ")`` won't work — we use the
    /// first whitespace token as the alias.
    static func parseAvailable(_ output: String) -> [(String, String?)] {
        var entries: [(String, String?)] = []
        for rawLine in output.split(separator: "\n", omittingEmptySubsequences: true) {
            let line = String(rawLine).trimmingCharacters(in: .whitespaces)
            // Skip headers, dividers, summary lines.
            if line.isEmpty { continue }
            if line.hasPrefix("Available models") { continue }
            if line.hasPrefix("Alias") { continue }
            if line.allSatisfy({ $0 == "─" || $0 == "-" || $0.isWhitespace }) { continue }
            // Audio-only aliases belong to the dedicated Audio surface and
            // Model Management tab, never the chat picker or chat auto-start.
            // Skip the section header — whose first token would otherwise
            // pass ``isSafeAlias`` as a phantom "Audio" model. Tagged rows are
            // rejected by ``hasNonChatKindTag`` below using a complete Kind
            // token, so an HF id that merely contains "[audio:" cannot hide an
            // unrelated chat row.
            if line.hasPrefix("Audio models") { continue }
            // Video-generation aliases, same reasoning and same shape.
            // A ``video-gen`` model has no tokenizer and no
            // ``stream_chat``, so it can never answer a chat request; the
            // sidecar exits 2 before binding a port when the video extras
            // are absent, and the user is told only "Couldn't start X.
            // Try again" — advice that will fail identically forever,
            // after a download of up to 64 GiB (#1603). The engine tags
            // these rows ``[video:gen]`` under a "Video models (N
            // aliases)" section; skip the header (its first token would
            // otherwise pass ``isSafeAlias`` and leak a phantom "Video"
            // model) and every tagged row.
            if line.hasPrefix("Video models") { continue }
            if hasNonChatKindTag(line) { continue }
            // Skip engine/server banner lines that can share stdout with
            // the table.
            //
            // The engine prints "Loading model with BatchedEngine: …"
            // and uvicorn prints "INFO:     Uvicorn running on …". Both
            // are prose, and the "first whitespace token is the alias"
            // rule turns them into phantom models — which is exactly how
            // a selectable model literally named "Loading" reached the
            // picker, and from there ``recommendedDefault`` put the word
            // "Loading" in the composer as if the user had chosen it.
            //
            // Matching on the banner prefix (rather than blacklisting
            // the word) keeps a genuine alias that merely starts with
            // those letters safe.
            if isBannerLine(line) { continue }
            // Catalog rows are column-aligned with runs of 2+ spaces.
            // Requiring a second column keeps prose footers out of the
            // catalog. In particular, current engines end with
            // "Size is an approximate download footprint ..."; taking the
            // first whitespace token alone promoted a phantom model named
            // "Size" into Settings and the picker.
            let columns = splitOnMultiSpace(line)
            guard columns.count >= 2 else { continue }
            let alias = columns[0]
            guard !alias.isEmpty else { continue }
            guard isSafeAlias(alias) else { continue }
            entries.append((alias, nil))
        }
        return entries
    }

    /// Parse the final `Preset` column emitted by `rapid-mlx models`. Keeping
    /// this separate preserves the long-standing lightweight tuple contract
    /// of ``parseAvailable`` while still carrying alias-profile capability
    /// metadata into Settings. Unknown/missing values fail closed.
    static func parseSpeculativeCapabilities(
        _ output: String
    ) -> [String: SpeculativeDecodingPreset] {
        // Older sidecars do not have this column. Do not reinterpret a
        // reasoning-parser or alias token named "Suffix" as capability data.
        let advertisesPresetColumn = output.split(separator: "\n").contains { line in
            let fields = line.split(whereSeparator: { $0.isWhitespace })
            return fields.first == "Alias" && fields.last == "Preset"
        }
        guard advertisesPresetColumn else { return [:] }
        let chatAliases = Set(parseAvailable(output).map(\.0))
        var capabilities: [String: SpeculativeDecodingPreset] = [:]
        for rawLine in output.split(separator: "\n", omittingEmptySubsequences: true) {
            let fields = rawLine.split(whereSeparator: { $0.isWhitespace }).map(String.init)
            guard let alias = fields.first, chatAliases.contains(alias),
                  let preset = fields.last else { continue }
            if preset.caseInsensitiveCompare("Suffix") == .orderedSame {
                capabilities[alias] = SpeculativeDecodingPreset(
                    method: .suffix, model: nil, tokens: nil
                )
                continue
            }
            let parts = preset.split(separator: "@", omittingEmptySubsequences: false)
            guard parts.count == 3,
                  String(parts[0]).caseInsensitiveCompare("MTP") == .orderedSame,
                  let model = sanitizedHuggingFaceRepo(String(parts[1])),
                  let tokens = Int(parts[2]), tokens > 0 else { continue }
            capabilities[alias] = SpeculativeDecodingPreset(
                method: .mtp, model: model, tokens: tokens
            )
        }
        return capabilities
    }

    /// Log/banner and catalog-notice lines the engine or its HTTP server
    /// can share stdout with the table. None of these are catalog rows, and
    /// every one of them would otherwise yield a phantom alias from its
    /// first token ("Loading", "INFO:", "Uvicorn", "No", …).
    ///
    /// Shared by `parseCached` and `parseAvailable`, so a single entry here
    /// fixes every consumer of the catalog (picker, Onboarding, Settings).
    ///
    /// Pure + `static` so the set is one list rather than a chain of
    /// `hasPrefix` calls buried in the parse loop.
    static func isBannerLine(_ line: String) -> Bool {
        // Match the full banner grammar, not a bare word. An alias is
        // ASCII `[A-Za-z0-9._-]` with no spaces or colons (``isSafeAlias``),
        // so a genuine alias row for a model literally named "Loading",
        // "Uvicorn", or "Traceback" is `<name><2+ spaces><size>` — which
        // none of these prefixes match, while the real banners
        // ("Loading model with …", "Uvicorn running on …", "Traceback
        // (most recent call last):") all do. `INFO:`/`WARNING:`/`ERROR:`
        // carry a colon and so can never collide with an alias.
        let bannerPrefixes = [
            "Loading model",
            "INFO:",
            "WARNING:",
            "ERROR:",
            "Uvicorn running",
            "Traceback (",
            // The empty-cache notice `rapid-mlx ls` prints in place of a
            // "Cached models" table when the disk is cold:
            //   "No models cached yet. Run 'rapid-mlx pull …' …"
            // Its single-space prose tokenizes (parseCached splits on ANY
            // whitespace) into alias "No", repo "models", size "cached yet." —
            // a selectable phantom that dead-ends model start
            // (raullenchai/Rapid-MLX#1918). Matching the full "No models
            // cached yet" prefix (not the bare word "No") keeps a genuine
            // alias named "No" — whose row is "No" + 2+ spaces + repo — safe.
            "No models cached yet",
        ]
        return bannerPrefixes.contains { line.hasPrefix($0) }
    }

    /// Parses ``rapid-mlx ls`` output. Each row has the alias in the
    /// first column, HF repo in the second, size on disk in the third.
    /// ``(unmapped)`` aliases are kept verbatim so the caller can skip
    /// them upstream.
    static func parseCached(_ output: String) -> [(String, String?, String?)] {
        var entries: [(String, String?, String?)] = []
        for rawLine in output.split(separator: "\n", omittingEmptySubsequences: true) {
            let line = String(rawLine).trimmingCharacters(in: .whitespaces)
            if line.isEmpty { continue }
            if line.hasPrefix("Cached models") { continue }
            if line.hasPrefix("Alias") { continue }
            if line.allSatisfy({ $0 == "─" || $0 == "-" || $0.isWhitespace }) { continue }
            // Same banner guard as ``parseAvailable`` — `ls` shares the
            // engine's stdout too.
            if isBannerLine(line) { continue }
            // Alias and HF repo IDs cannot contain whitespace, so parse
            // those fields as tokens rather than by visual column spacing.
            // Rich leaves only one space after a repo that fills the column
            // width (for example Qwen3-TTS CustomVoice). A 2+-space split
            // would merge that repo with "2.2 GiB" and reject it as invalid.
            let fields = line.split(whereSeparator: { $0.isWhitespace }).map(String.init)
            guard fields.count >= 2 else { continue }
            let alias = fields[0]
            // ``(unmapped)`` and ``(external)`` are the two status values the
            // engine may put in the alias column that still carry a real
            // repo. Every other parenthesized value — ``(incomplete)`` — is
            // dropped on purpose so a half-downloaded directory never reads
            // as ready. ``(external)`` marks a model another MLX runtime
            // downloaded (#1718): usable, but not ours to delete, since the
            // delete path rebuilds ``<hub-root>/models--<repo>`` and that is
            // not where it lives.
            guard alias == "(unmapped)" || alias == "(external)" || isSafeAlias(alias) else {
                continue
            }
            let hf = sanitizedHuggingFaceRepo(fields[1])
            // Size is a two-token cell ("2.2 GiB"); anything shorter means the
            // row carried no measured size.
            let size: String?
            if fields.count >= 4 {
                size = "\(fields[2]) \(fields[3])"
            } else {
                size = fields.count == 3 ? fields[2] : nil
            }
            entries.append((alias, hf, size))
        }
        return entries
    }

    static func isSafeAlias(_ alias: String) -> Bool {
        guard !alias.isEmpty, alias.utf8.count <= maxAliasBytes else { return false }
        guard let first = alias.utf8.first, isASCIILetterOrDigit(first) else { return false }
        return alias.utf8.allSatisfy { byte in
            isASCIILetterOrDigit(byte) || byte == 45 || byte == 46 || byte == 95
        }
    }

    static func sanitizedHuggingFaceRepo(_ repo: String?) -> String? {
        guard let repo else { return nil }
        let trimmed = repo.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              trimmed.utf8.count <= maxHuggingFaceRepoBytes,
              trimmed != "-" && trimmed != "—" else {
            return nil
        }

        let parts = trimmed.split(separator: "/", omittingEmptySubsequences: false)
        guard (1...2).contains(parts.count) else { return nil }
        for part in parts {
            guard !part.isEmpty, part != ".", part != ".." else { return nil }
            guard part.utf8.allSatisfy({ byte in
                isASCIILetterOrDigit(byte) || byte == 45 || byte == 46 || byte == 95
            }) else {
                return nil
            }
        }
        return trimmed
    }

    private static func isASCIILetterOrDigit(_ byte: UInt8) -> Bool {
        (byte >= 48 && byte <= 57) || (byte >= 65 && byte <= 90) || (byte >= 97 && byte <= 122)
    }

    /// Splits a string on runs of 2+ whitespace characters. Used to
    /// parse the column-aligned ``rapid-mlx ls`` output.
    private static func splitOnMultiSpace(_ line: String) -> [String] {
        var result: [String] = []
        var current = ""
        var spaceRun = 0
        for ch in line {
            if ch == " " || ch == "\t" {
                spaceRun += 1
            } else {
                if spaceRun >= 2 && !current.isEmpty {
                    result.append(current)
                    current = ""
                }
                if spaceRun >= 1 && !current.isEmpty && spaceRun < 2 {
                    // Single-space within a column (e.g. "5d ago") — keep as-is.
                    current.append(" ")
                }
                current.append(ch)
                spaceRun = 0
            }
        }
        if !current.isEmpty { result.append(current) }
        return result.map { $0.trimmingCharacters(in: .whitespaces) }
    }

    /// Shells out to ``rapid-mlx <args>``. Returns the stdout as a
    /// UTF-8 string, or empty on any failure. The subprocess is launched
    /// without a shell to keep argv exact.
    ///
    /// Codex round-1 finding: previously we only drained stdout AFTER
    /// the child exited. If rapid-mlx emitted enough stderr to fill
    /// the OS pipe buffer (~64 KB on macOS), the child would block on
    /// the next write and never exit — the catalog task would hang
    /// the picker indefinitely. Drain stdout AND stderr concurrently
    /// via separate background reader tasks while the child runs.
    private static func runRapidMlx(
        binary: URL,
        args: [String],
        hubCacheOverride: URL? = nil
    ) async -> String {
        let processBox = CatalogProcessBox()
        return await withTaskCancellationHandler {
            await withCheckedContinuation { (continuation: CheckedContinuation<String, Never>) in
            let task = Process()
            task.executableURL = binary
            task.arguments = args
            // Issue #503: when the user pointed Rapid at a custom models
            // folder, run the probe with that folder so ``rapid-mlx ls``
            // enumerates the right directory. The helper preserves the
            // ambient environment either way, adding only the optional cache
            // paths and the #1415 telemetry opt-out for internal probes.
            task.environment = probeEnvironment(
                ambient: ProcessInfo.processInfo.environment,
                hubCacheOverride: hubCacheOverride
            )
            let stdout = Pipe()
            let stderr = Pipe()
            task.standardOutput = stdout
            task.standardError = stderr
            // Codex round-4 finding: my round-3 attempt still had
            // a race — the ``drainGroup.enter`` calls happened AFTER
            // ``task.run()``. A fast-exit child (``rapid-mlx ls``
            // <5 ms) would fire the termination handler BEFORE the
            // drainers entered the group; ``drainGroup.wait`` with
            // zero entries returns immediately, the continuation
            // resumes with empty stdout, and the picker shows an
            // empty catalog.
            //
            // Correct ordering: start the drainers FIRST so the
            // group already has both entries before the child exits.
            // On launch failure, close the pipe write ends so the
            // drainers see EOF and the drainGroup-protected resume
            // path still runs.
            let stdoutBox = DataBox()
            let stderrBox = DataBox()
            let drainGroup = DispatchGroup()
            let resumedBox = ResumedFlag()

            drainGroup.enter()
            DispatchQueue.global(qos: .utility).async {
                stdoutBox.data = readPipeData(
                    stdout.fileHandleForReading,
                    maxBytes: maxSubprocessStdoutBytes
                )
                drainGroup.leave()
            }
            drainGroup.enter()
            DispatchQueue.global(qos: .utility).async {
                stderrBox.data = readPipeData(
                    stderr.fileHandleForReading,
                    maxBytes: maxSubprocessStderrBytes
                )
                drainGroup.leave()
            }

            task.terminationHandler = { _ in
                drainGroup.wait()
                processBox.clear(task)
                if resumedBox.tryConsume() {
                    let text = String(data: stdoutBox.data, encoding: .utf8) ?? ""
                    continuation.resume(returning: text)
                }
            }

            processBox.set(task)
            do {
                try task.run()
                // The child now holds its own dup of both write ends, so
                // drop OUR copies. While the parent keeps a write end
                // open the pipe can never reach EOF — ``readPipeData``
                // then blocks forever even after the child exits, and
                // ``terminationHandler``'s ``drainGroup.wait()`` deadlocks
                // the continuation with it. The launch-failure branch
                // below has always closed them; the success path is where
                // a long-lived or hung child actually makes it matter.
                try? stdout.fileHandleForWriting.close()
                try? stderr.fileHandleForWriting.close()
                processBox.terminateIfCancelled()
            } catch {
                // Close write ends so the drainers see EOF instead
                // of blocking forever on a never-written pipe.
                try? stdout.fileHandleForWriting.close()
                try? stderr.fileHandleForWriting.close()
                drainGroup.wait()
                processBox.clear(task)
                if resumedBox.tryConsume() {
                    continuation.resume(returning: "")
                }
                return
            }
            }
        } onCancel: {
            processBox.cancel()
        }
    }

    private static func readPipeData(_ handle: FileHandle, maxBytes: Int) -> Data {
        var data = Data()
        while true {
            let chunk: Data?
            do {
                chunk = try handle.read(upToCount: pipeReadChunkBytes)
            } catch {
                break
            }
            guard let chunk, !chunk.isEmpty else { break }
            let remaining = maxBytes - data.count
            if remaining > 0 {
                data.append(contentsOf: chunk.prefix(remaining))
            }
        }
        return data
    }

    static func _testingRunRapidMlx(binary: URL, args: [String]) async -> String {
        await runRapidMlx(binary: binary, args: args)
    }

    /// Environment for app-owned, read-only catalog probes. These invocations
    /// are implementation details, not engine sessions: one picker refresh can
    /// execute `models`, `ls`, and several `info` commands. Letting each emit a
    /// lifecycle pair inflated usage telemetry and made app shutdown wait on
    /// probe POSTs (#1415). Override even an ambient opt-in; the real `serve`
    /// child has its own environment path and keeps telemetry enabled.
    nonisolated static func probeEnvironment(
        ambient: [String: String],
        hubCacheOverride: URL?
    ) -> [String: String] {
        var env = ambient
        env["DO_NOT_TRACK"] = "1"
        if let hubCacheOverride {
            env["HF_HUB_CACHE"] = hubCacheOverride.path
            // Issue #1718: scan both Hugging Face and external-runtime layouts.
            env[extraModelRootsEnvKey] = mergedExtraModelRoots(
                existing: env[extraModelRootsEnvKey],
                selected: hubCacheOverride.path
            )
        }
        return env
    }
}

/// Mutable reference box for letting two background drainer closures
/// write into shared storage without capture-rules complaints. Pure
/// internal helper for ``ModelCatalog.runRapidMlx``.
private final class DataBox: @unchecked Sendable {
    var data: Data = .init()
}

/// Single-shot atomic flag shared between the launch-failure and
/// terminationHandler paths so the checked continuation is resumed
/// exactly once. Codex round-3 finding: pre-install termination
/// handler races with launch-failure resume.
private final class ResumedFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var consumed = false
    func tryConsume() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if consumed { return false }
        consumed = true
        return true
    }
}

/// Cancellation bridge for ``ModelCatalog.runRapidMlx``. The async
/// catalog load is cancellable; the short-lived child process must be
/// signalled too or a picker refresh can leave orphaned rapid-mlx
/// subprocesses behind.
private final class CatalogProcessBox: @unchecked Sendable {
    private let lock = NSLock()
    private var process: Process?
    private var cancelled = false

    func set(_ process: Process) {
        lock.lock()
        self.process = process
        let shouldCancel = cancelled
        lock.unlock()
        if shouldCancel {
            terminate(process)
        }
    }

    func cancel() {
        lock.lock()
        cancelled = true
        let process = self.process
        lock.unlock()
        if let process {
            terminate(process)
        }
    }

    func terminateIfCancelled() {
        lock.lock()
        let shouldCancel = cancelled
        let process = self.process
        lock.unlock()
        if shouldCancel, let process {
            terminate(process)
        }
    }

    func clear(_ process: Process) {
        lock.lock()
        if self.process === process {
            self.process = nil
        }
        lock.unlock()
    }

    private func terminate(_ process: Process) {
        guard process.isRunning else { return }
        process.terminate()
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 2.0) {
            if process.isRunning {
                kill(process.processIdentifier, SIGKILL)
            }
        }
    }
}
