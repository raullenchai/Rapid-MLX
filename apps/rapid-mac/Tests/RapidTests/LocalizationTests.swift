import Foundation
import Testing
@testable import Rapid

/// Pin the shape of ``Localizable.xcstrings`` so a future drift can't
/// silently break the zh-Hans surface, and prove that
/// ``NSLocalizedString`` resolves through the catalog when the runtime
/// language is forced to Simplified Chinese.
@Suite("Localizable.xcstrings — catalog shape and zh-Hans resolution")
struct LocalizationTests {

    private static let experimentalTemplateKeys = [
        "experimental.video.generating",
        "experimental.video.memory_requirement",
        "experimental.benchmark.rounds",
        "experimental.cua.step",
        "experimental.cua.step_finished",
        "experimental.cua.switched",
        "experimental.cua.run",
        "experimental.cua.app_unavailable",
        "experimental.cua.window_unavailable",
        "experimental.cua.targets_unavailable",
        "experimental.cua.targets_unavailable_retry"
    ]

    private static let photoHintCatalogKeys = [
        "image_input.unavailable.legacy_model",
        "image_input.unavailable.text_lane_forced",
        "image_input.unavailable.speculative_decode",
        "image_input.unavailable.vision_memory_insufficient",
        "image_input.unavailable.vision_runtime_unsupported",
        "image_input.unavailable.vision_features_unavailable",
        "image_input.unavailable.text_checkpoint",
        "image_input.unavailable.generic_text_lane"
    ]

    /// Look up the catalog from the test bundle. The .xcstrings file
    /// is declared as a resource on the Rapid executable target, so
    /// at test time it lives next to the test bundle's bundleURL
    /// under the host process's resource lookup chain. We probe the
    /// known SPM bundle path first, then fall back to the source
    /// tree path which is always present in a CI checkout.
    private func catalogURL() throws -> URL {
        let candidates: [URL] = [
            Bundle.module.url(forResource: "Localizable", withExtension: "xcstrings"),
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("Sources/Rapid/Resources/Localizable.xcstrings")
        ].compactMap { $0 }

        return try #require(
            candidates.first { FileManager.default.fileExists(atPath: $0.path) },
            "Localizable.xcstrings not found on any candidate path"
        )
    }

    private func loadCatalog() throws -> [String: Any] {
        let url = try catalogURL()
        let data = try Data(contentsOf: url)
        let any = try JSONSerialization.jsonObject(with: data)
        return try #require(any as? [String: Any])
    }

    private func experimentalSurfaceSources() throws -> [String] {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = root.appendingPathComponent("Sources/Rapid", isDirectory: true)
        let manager = FileManager.default
        var urls = [
            source.appendingPathComponent("UI/CommunityBenchmarkView.swift"),
            source.appendingPathComponent("UI/CommunityBenchmarkFeatureConfig.swift"),
            source.appendingPathComponent("UI/CUASection.swift"),
            source.appendingPathComponent("UI/ComputerUseView.swift"),
            source.appendingPathComponent("UI/VideoView.swift")
        ]
        for directory in ["UI/CommunityBenchmark", "ShareCompute", "ComputerUse", "Video"] {
            let url = source.appendingPathComponent(directory, isDirectory: true)
            urls += try manager.contentsOfDirectory(
                at: url,
                includingPropertiesForKeys: nil
            ).filter { $0.pathExtension == "swift" }
        }
        return urls.map(\.path).sorted()
    }

    private func rapidSource(_ relativePath: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/Rapid/\(relativePath)")
    }

    /// Covers the literals the native extractor skips because an interpolation's
    /// type (and so its exact `%@` / `%lld` specifier) is only known to the
    /// compiler. Specifiers are compared loosely here; every literal whose key
    /// can be derived exactly is checked by `nativeExtractedSourceKeysTranslated`.
    @Test("Interpolated source literals have Simplified Chinese catalog entries")
    func interpolatedSourceLiteralsTranslated() throws {
        let strings = try #require(try loadCatalog()["strings"] as? [String: Any])
        // Skip an interpolation as a balanced expression, including quoted strings
        // inside it, so nested calls and nil-coalescing literals stay intact.
        func interpolationEnd(_ text: String, from start: String.Index) -> String.Index? {
            var i = start
            var depth = 1
            var quoted = false
            while i < text.endIndex {
                let c = text[i]
                if c == "\\" {
                    i = text.index(after: i)
                    if i < text.endIndex { i = text.index(after: i) }
                    continue
                }
                if c == "\"" { quoted.toggle() }
                if !quoted {
                    if c == "(" { depth += 1 }
                    if c == ")" {
                        depth -= 1
                        if depth == 0 { return text.index(after: i) }
                    }
                }
                i = text.index(after: i)
            }
            return nil
        }
        func normalize(_ value: String) -> String {
            var result = ""
            var i = value.startIndex
            while i < value.endIndex {
                if value[i...].hasPrefix(#"\("#),
                   let end = interpolationEnd(value, from: value.index(i, offsetBy: 2)) {
                    result += "{arg}"
                    i = end
                } else {
                    result.append(value[i])
                    i = value.index(after: i)
                }
            }
            return result.replacingOccurrences(of: #"\n"#, with: "\n").replacingOccurrences(
                of: #"%(?:\d+\$)?(?:@|lld|ld|d|(?:\.\d+)?l?f)"#,
                with: "{arg}", options: .regularExpression
            ).replacingOccurrences(of: "%%", with: "%")
        }
        let indexed = Dictionary(grouping: strings.keys, by: normalize)
        let pattern = #"(?:String\s*\(\s*localized\s*:|\b(?:Text|Button|Toggle|Label|Section|Picker|LabeledContent)\s*\(|\.(?:help|navigationTitle|accessibilityLabel|accessibilityHint)\s*\()\s*\""#
        let regex = try NSRegularExpression(pattern: pattern)
        let enumerator = try #require(FileManager.default.enumerator(at: rapidSource(""), includingPropertiesForKeys: nil))
        let allowlist: Set<String> = ["LLM", "macOS", "Rapid-MLX", "MCP", "Hugging Face", "QuickSilver", "R"]
        for case let url as URL in enumerator where url.pathExtension == "swift" {
            if url.lastPathComponent == "DevSnapshot.swift" || url.path.contains("/UI/Markdown/") { continue }
            let source = try String(contentsOf: url, encoding: .utf8)
            for match in regex.matches(in: source, range: NSRange(source.startIndex..., in: source)) {
                guard let prefix = Range(match.range, in: source) else { continue }
                let lineStart = source[..<prefix.lowerBound].lastIndex(of: "\n").map { source.index(after: $0) } ?? source.startIndex
                if source[lineStart..<prefix.lowerBound].trimmingCharacters(in: .whitespaces).hasPrefix("//") { continue }
                let start = prefix.upperBound
                var end = start
                while end < source.endIndex {
                    if source[end...].hasPrefix(#"\("#),
                       let next = interpolationEnd(source, from: source.index(end, offsetBy: 2)) {
                        end = next
                    } else if source[end] == "\\" {
                        end = source.index(after: end)
                        if end < source.endIndex { end = source.index(after: end) }
                    } else if source[end] == "\"" {
                        break
                    } else {
                        end = source.index(after: end)
                    }
                }
                guard end < source.endIndex else { continue }
                let after = source[source.index(after: end)...].drop(while: \.isWhitespace)
                let before = source[..<prefix.lowerBound].reversed().drop(while: \.isWhitespace)
                if after.first == "+" || before.first == "+" { continue }
                let literal = String(source[start..<end]).replacingOccurrences(of: #"\""#, with: "\"")
                if allowlist.contains(literal) || normalize(literal).replacingOccurrences(of: "{arg}", with: "").rangeOfCharacter(from: .letters) == nil { continue }
                let translated = (indexed[normalize(literal)] ?? []).contains { key in
                    guard let entry = strings[key] as? [String: Any],
                          let localizations = entry["localizations"] as? [String: Any],
                          let zh = localizations["zh-Hans"] as? [String: Any] else { return false }
                    let plural = ((zh["variations"] as? [String: Any])?["plural"] as? [String: Any])?["other"] as? [String: Any]
                    guard let unit = (plural ?? zh)["stringUnit"] as? [String: Any] else { return false }
                    return unit["state"] as? String == "translated" && !(unit["value"] as? String ?? "").isEmpty
                }
                #expect(translated, "Missing zh-Hans: \(literal) in \(url.lastPathComponent)")
            }
        }
    }

    private func rapidSources() throws -> [String] {
        let source = rapidSource("")
        let enumerator = try #require(FileManager.default.enumerator(
            at: source, includingPropertiesForKeys: nil
        ))
        let excludedSource = rapidSource("DevSnapshot.swift")
        var sources: [String] = []
        for case let url as URL in enumerator where url.pathExtension == "swift" {
            if url.standardizedFileURL == excludedSource.standardizedFileURL { continue }
            sources.append(url.path)
        }
        return sources.sorted()
    }

    @Test("Every native-extracted source key carries zh-Hans")
    func nativeExtractedSourceKeysTranslated() async throws {
        let sources = try rapidSources()
        try #require(!sources.isEmpty, "No Swift sources found for localization extraction")
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-localization-extract-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: output) }

        let extractor = try await TestSubprocess.run(
            executableURL: URL(fileURLWithPath: "/usr/bin/xcrun"),
            arguments: [
                "xcstringstool", "extract", "--avoid-arg-placeholder",
                "--modern-localizable-strings", "--SwiftUI",
                "--output-format", "xcstrings", "--output-directory", output.path
            ] + sources,
            timeout: 30
        )
        try #require(
            extractor.terminationStatus == 0,
            "xcstringstool extraction failed: \(String(decoding: extractor.standardError, as: UTF8.self))"
        )

        let extractedData = try Data(contentsOf: output.appendingPathComponent("Localizable.xcstrings"))
        let extracted = try #require(
            try JSONSerialization.jsonObject(with: extractedData) as? [String: Any]
        )
        let extractedStrings = try #require(extracted["strings"] as? [String: Any])
        let strings = try #require(try loadCatalog()["strings"] as? [String: Any])
        let allowlist: Set<String> = ["7659", "URL", "herdr", "https://huggingface.co", "✓", "R"]
        for key in extractedStrings.keys.sorted() {
            if allowlist.contains(key) || key.rangeOfCharacter(from: .letters) == nil { continue }
            let entry = strings[key] as? [String: Any]
            let localizations = entry?["localizations"] as? [String: Any]
            let zh = localizations?["zh-Hans"] as? [String: Any]
            let plural = ((zh?["variations"] as? [String: Any])?["plural"] as? [String: Any])?["other"] as? [String: Any]
            let units = [zh?["stringUnit"], plural?["stringUnit"]].compactMap { $0 as? [String: Any] }
            let translated = units.contains { unit in
                unit["state"] as? String == "translated" && !(unit["value"] as? String ?? "").isEmpty
            }
            #expect(translated, "Missing translated zh-Hans for exact source key: \(key)")
        }
    }

    @Test("Benchmark translations use the intended terminology")
    func benchmarkTranslations() throws {
        let strings = try #require(try loadCatalog()["strings"] as? [String: Any])
        let expected = ["Run": "运行", "Change model": "更换模型", "The most unified memory the model held at once.": "模型运行期间占用的统一内存峰值。"]
        for (key, value) in expected {
            let entry = try #require(strings[key] as? [String: Any])
            let localizations = try #require(entry["localizations"] as? [String: Any])
            let zh = try #require(localizations["zh-Hans"] as? [String: Any])
            let unit = try #require(zh["stringUnit"] as? [String: Any])
            #expect(unit["value"] as? String == value)
        }
        for (_, raw) in strings {
            guard let entry = raw as? [String: Any], let localizations = entry["localizations"] as? [String: Any], let zh = localizations["zh-Hans"] as? [String: Any] else { continue }
            func containsBadTranslation(_ value: Any) -> Bool {
                if let string = value as? String { return string.contains("跑步") }
                if let dictionary = value as? [String: Any] { return dictionary.values.contains(where: containsBadTranslation) }
                if let array = value as? [Any] { return array.contains(where: containsBadTranslation) }
                return false
            }
            #expect(!containsBadTranslation(zh))
        }
    }

    @Test("Catalog parses as valid xcstrings JSON with the expected top-level shape")
    func catalogShape() throws {
        let json = try loadCatalog()
        #expect(json["sourceLanguage"] as? String == "en")
        #expect(json["version"] as? String == "1.0")
        let strings = try #require(json["strings"] as? [String: Any])
        #expect(!strings.isEmpty)
    }

    @Test("Canonical user-visible keys carry a zh-Hans translation")
    func canonicalKeysTranslated() throws {
        let json = try loadCatalog()
        let strings = try #require(json["strings"] as? [String: Any])

        // Pick a few high-visibility keys spanning compose, sidebar,
        // settings, about, status — if any of these regress to
        // untranslated, the Chinese surface is visibly broken.
        let mustHaveZH: [String] = [
            "Send a message…",
            "New chat",
            "Search chats",
            "Today",
            "Previous 30 Days",
            "No chats match",
            "Settings",
            "Appearance",
            "Privacy",
            "About Rapid-MLX",
            "Ready",
            "Downloading",
            "Stopped"
        ]

        for key in mustHaveZH {
            let entry = try #require(
                strings[key] as? [String: Any],
                "Missing catalog entry for key: \(key)"
            )
            let localizations = try #require(entry["localizations"] as? [String: Any])
            let zh = try #require(
                localizations["zh-Hans"] as? [String: Any],
                "Missing zh-Hans for key: \(key)"
            )
            let unit = try #require(zh["stringUnit"] as? [String: Any])
            #expect(unit["state"] as? String == "translated")
            let value = try #require(unit["value"] as? String)
            #expect(!value.isEmpty)
        }
    }

    @Test("Code block Copy and Copied use localized keys in every state")
    func codeBlockCopyLabels() throws {
        let json = try loadCatalog()
        let strings = try #require(json["strings"] as? [String: Any])
        for (key, value) in [("Copy", "复制"), ("Copied", "已复制")] {
            let entry = try #require(strings[key] as? [String: Any])
            let localizations = try #require(entry["localizations"] as? [String: Any])
            let zh = try #require(localizations["zh-Hans"] as? [String: Any])
            let unit = try #require(zh["stringUnit"] as? [String: Any])
            #expect(unit["value"] as? String == value)
            #expect(unit["state"] as? String == "translated")
        }
        let source = try String(contentsOf: rapidSource(
            "UI/Markdown/TextKit/Render/MarkdownCodeBlockView.swift"
        ))
        #expect(source.components(separatedBy: "copyButton.title = String(localized: \"Copy\")").count == 3)
        #expect(source.contains("copyButton.title = String(localized: \"Copied\")"))
        #expect(!source.contains("\"复制\""))
        #expect(!source.contains("\"已复制\""))
    }

    @Test("Every entry that declares a zh-Hans block has a non-empty translated value")
    func noPartialZHEntries() throws {
        let json = try loadCatalog()
        let strings = try #require(json["strings"] as? [String: Any])

        for (key, raw) in strings {
            guard
                let entry = raw as? [String: Any],
                let localizations = entry["localizations"] as? [String: Any],
                let zh = localizations["zh-Hans"] as? [String: Any]
            else {
                continue
            }
            let plural = (zh["variations"] as? [String: Any])?["plural"] as? [String: Any]
            let other = plural?["other"] as? [String: Any]
            let unit = try #require((other ?? zh)["stringUnit"] as? [String: Any], "Missing stringUnit for \(key)")
            let value = unit["value"] as? String ?? ""
            #expect(!value.isEmpty, "Empty zh-Hans value for key: \(key)")
            #expect(
                (unit["state"] as? String) == "translated",
                "zh-Hans not marked translated for key: \(key)"
            )
        }
    }

    @Test("Every photo-unavailable remedy has reviewed English and zh-Hans catalog values")
    func localizedPhotoHintsUseStableCatalogKeys() throws {
        let json = try loadCatalog()
        let strings = try #require(json["strings"] as? [String: Any])

        #expect(
            Set(ImageInputAvailability.PhotoHint.allCases.map(\.rawValue))
                == Set(Self.photoHintCatalogKeys),
            "The production photo-hint key set and the reviewed catalog contract must move together."
        )

        for hint in ImageInputAvailability.PhotoHint.allCases {
            let key = hint.rawValue
            let entry = try #require(
                strings[key] as? [String: Any],
                "Missing photo-hint catalog entry for key: \(key)"
            )
            let localizations = try #require(entry["localizations"] as? [String: Any])
            for language in ["en", "zh-Hans"] {
                let localization = try #require(
                    localizations[language] as? [String: Any],
                    "Missing \(language) photo-hint value for key: \(key)"
                )
                let unit = try #require(localization["stringUnit"] as? [String: Any])
                #expect(unit["state"] as? String == "translated")
                #expect(!(unit["value"] as? String ?? "").isEmpty)
                if language == "en" {
                    #expect(
                        unit["value"] as? String == hint.englishValue,
                        "The catalog's source copy and the fail-safe English value diverged for \(key)."
                    )
                }
            }
        }
    }

    /// The surfaces a Chinese-locale user reported as English: sidebar
    /// destinations, the chat hero and composer, the Settings rail, and the
    /// model list. Each key is spelled exactly as the call site spells it,
    /// including the format specifier an interpolation compiles to.
    @Test("Sidebar, chat hero, Settings rail and model list keys carry zh-Hans")
    func reportedSurfacesTranslated() throws {
        let json = try loadCatalog()
        let strings = try #require(json["strings"] as? [String: Any])

        let mustHaveZH: [String] = [
            "New Chat", "Images", "Audio", "Agent",
            "Ask anything", "Send a message…", "Chatting with %@",
            "Model Management", "System Prompt", "Memory", "Tools",
            "Performance", "Experimental", "Appearance", "Privacy", "App",
            "Recommended for your %@", "Best pick", "Faster", "In use",
            "Download", "All models", "All", "Cached", "Not cached"
        ]

        for key in mustHaveZH {
            let entry = try #require(
                strings[key] as? [String: Any],
                "Missing catalog entry for key: \(key)"
            )
            let localizations = try #require(entry["localizations"] as? [String: Any])
            let zh = try #require(
                localizations["zh-Hans"] as? [String: Any],
                "Missing zh-Hans for key: \(key)"
            )
            let unit = try #require(zh["stringUnit"] as? [String: Any])
            #expect(!(unit["value"] as? String ?? "").isEmpty, "Empty zh-Hans for key: \(key)")
        }
    }

    @Test("Every native-extracted experimental-surface key carries zh-Hans")
    func experimentalSurfaceSourceKeysTranslated() async throws {
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-localization-extract-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: output) }

        let extractor = try await TestSubprocess.run(
            executableURL: URL(fileURLWithPath: "/usr/bin/xcrun"),
            arguments: [
                "xcstringstool", "extract", "--avoid-arg-placeholder",
                "--modern-localizable-strings", "--SwiftUI",
                "--output-format", "xcstrings", "--output-directory", output.path
            ] + (try experimentalSurfaceSources())
        )
        #expect(
            extractor.terminationStatus == 0,
            "xcstringstool extraction failed: \(String(decoding: extractor.standardError, as: UTF8.self))"
        )

        let extractedData = try Data(contentsOf: output.appendingPathComponent("Localizable.xcstrings"))
        let extracted = try #require(
            try JSONSerialization.jsonObject(with: extractedData) as? [String: Any]
        )
        let extractedStrings = try #require(extracted["strings"] as? [String: Any])
        let catalogStrings = try #require(try loadCatalog()["strings"] as? [String: Any])

        for key in extractedStrings.keys where !key.isEmpty {
            let entry = try #require(
                catalogStrings[key] as? [String: Any],
                "Experimental surface key is absent from the catalog: \(key)"
            )
            let localizations = try #require(entry["localizations"] as? [String: Any])
            let zh = try #require(
                localizations["zh-Hans"] as? [String: Any],
                "Experimental surface key lacks zh-Hans: \(key)"
            )
            let unit = try #require(zh["stringUnit"] as? [String: Any])
            #expect(unit["state"] as? String == "translated")
            #expect(!(unit["value"] as? String ?? "").isEmpty)
        }
    }

    @Test("Computed experimental copy stays connected to production localization boundaries")
    func computedExperimentalCopyUsesProductionBoundaries() throws {
        let requirements: [String: [String]] = [
            "UI/VideoView.swift": [
                "returnString(localized:\"Findingvideomodels…\")",
                "returnExperimentalSurfaceCopy.videoProgress(job.progress)"
            ],
            "Video/VideoGenViewModel.swift": [
                "returnExperimentalSurfaceCopy.videoMemory(minimum:Int(minimum.rounded()),available:Int(physicalRAMGB.rounded()))"
            ],
            "UI/CommunityBenchmarkView.swift": [
                "returnExperimentalSurfaceCopy.benchmarkRounds(rounds)"
            ],
            "UI/CUASection.swift": [
                "returnExperimentalSurfaceCopy.cuaStep(event.step??0,instruction:instruction)",
                "returnExperimentalSurfaceCopy.cuaStepFinished(event.step??0)",
                "returnExperimentalSurfaceCopy.cuaSwitched(to:destination)",
                "returnExperimentalSurfaceCopy.cuaRun(event.status??\"ended\")"
            ],
            "ComputerUse/CUAViewModel.swift": [
                "returnExperimentalSurfaceCopy.cuaAppUnavailable(hint:hint)",
                "returnExperimentalSurfaceCopy.cuaWindowUnavailable(hint:hint)",
                "returnExperimentalSurfaceCopy.cuaTargetsUnavailable(message:message,hint:hint)",
                "returnExperimentalSurfaceCopy.cuaTargetsUnavailable(error:Self.describe(error))"
            ]
        ]

        for (path, expectedCalls) in requirements {
            let source = try String(contentsOf: rapidSource(path), encoding: .utf8)
            let canonical = SourceGuardSupport.canonicalSource(source, literals: .preserve)
            for expectedCall in expectedCalls {
                #expect(
                    canonical.contains(expectedCall),
                    "Production localization boundary was removed from \(path): \(expectedCall)"
                )
            }
        }

        let catalogStrings = try #require(try loadCatalog()["strings"] as? [String: Any])
        for key in Self.experimentalTemplateKeys {
            let entry = try #require(
                catalogStrings[key] as? [String: Any],
                "Missing computed-copy catalog contract: \(key)"
            )
            let localizations = try #require(entry["localizations"] as? [String: Any])
            for language in ["en", "zh-Hans"] {
                let unit = try #require(
                    (localizations[language] as? [String: Any])?["stringUnit"] as? [String: Any],
                    "Missing \(language) computed-copy value: \(key)"
                )
                #expect(unit["state"] as? String == "translated")
                #expect(!(unit["value"] as? String ?? "").isEmpty)
            }
        }
    }

    /// A translation that drops, adds, or retypes a format argument renders
    /// garbage (or reads a wrong-typed vararg) only in that language, where
    /// no English-run test would see it.
    @Test("Every zh-Hans value keeps the format arguments of its source string")
    func translationsPreserveFormatArguments() throws {
        let json = try loadCatalog()
        let strings = try #require(json["strings"] as? [String: Any])

        for (key, raw) in strings {
            guard
                let entry = raw as? [String: Any],
                let localizations = entry["localizations"] as? [String: Any],
                let zhUnit = (localizations["zh-Hans"] as? [String: Any])?["stringUnit"] as? [String: Any],
                let zhValue = zhUnit["value"] as? String
            else {
                continue
            }
            // Symbolic keys ("image_input.unavailable…") carry their English
            // in an explicit `en` value; everything else is keyed by it. A
            // symbolic key with no `en` value has its source in code, out of
            // this test's reach.
            let enUnit = (localizations["en"] as? [String: Any])?["stringUnit"] as? [String: Any]
            let enValue = enUnit?["value"] as? String
            if enValue == nil, key.wholeMatch(of: /[a-z0-9_]+(\.[a-z0-9_]+)+/) != nil { continue }
            let source = enValue ?? key
            #expect(
                Self.formatArguments(in: zhValue) == Self.formatArguments(in: source),
                "zh-Hans format arguments diverge from the source for key: \(key)"
            )
        }
    }

    /// Position-resolved argument types of a format string, so a translation
    /// may reorder arguments with `%2$@` but not change what they are.
    private static func formatArguments(in format: String) -> [String] {
        let pattern = /%(?:(\d+)\$)?(lld|ld|d|@|\.\d+f|f|%)/
        var implicitPosition = 0
        var arguments: [(Int, String)] = []
        for match in format.matches(of: pattern) {
            let type = String(match.output.2)
            guard type != "%" else { continue }
            implicitPosition += 1
            let position = match.output.1.flatMap { Int($0) } ?? implicitPosition
            arguments.append((position, type))
        }
        return arguments.sorted { $0.0 < $1.0 }.map { "\($0.0):\($0.1)" }
    }

    /// Catalog keys for interpolated copy are whatever the compiler derives
    /// from the call site ("Chatting with \(alias)" -> "Chatting with %@").
    /// Resolving real interpolations against the compiled table proves the
    /// hand-written keys match that derivation instead of merely existing.
    @Test("Interpolated call-site strings resolve against the compiled zh-Hans table")
    func compiledCatalogLocalizesInterpolatedCopy() async throws {
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-localization-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: output) }

        let compiler = try await TestSubprocess.run(
            executableURL: URL(fileURLWithPath: "/usr/bin/xcrun"),
            arguments: [
                "xcstringstool", "compile", try catalogURL().path,
                "--output-directory", output.path,
                "--serialization-format", "binary"
            ]
        )
        #expect(
            compiler.terminationStatus == 0,
            "xcstringstool failed: \(String(decoding: compiler.standardError, as: UTF8.self))"
        )
        let zhBundle = try #require(
            Bundle(url: output.appendingPathComponent("zh-Hans.lproj", isDirectory: true)),
            "xcstringstool did not emit a loadable zh-Hans localization bundle"
        )
        let enBundle = try #require(
            Bundle(url: output.appendingPathComponent("en.lproj", isDirectory: true)),
            "xcstringstool did not emit a loadable English localization bundle"
        )

        let alias = "qwen3.8-27b-4bit"
        let count = 3
        #expect(String(localized: "New Chat", bundle: zhBundle) == "新对话")
        #expect(String(localized: "Model Management", bundle: zhBundle) == "模型管理")
        #expect(String(localized: "Chatting with \(alias)", bundle: zhBundle) == "正在与 qwen3.8-27b-4bit 对话")
        #expect(String(localized: "Download \(alias) first", bundle: zhBundle) == "请先下载 qwen3.8-27b-4bit")
        #expect(String(localized: "Type a message, then Start & Send to start \(alias).", bundle: zhBundle) == "输入消息后，即可启动 qwen3.8-27b-4bit 并发送。")
        // Composer load status: positional arguments must survive, or the
        // byte counts would print in the wrong order.
        let done = "1.7 GB", total = "2.9 GB"
        #expect(String(localized: "\(done) of \(total)", bundle: zhBundle) == "1.7 GB / 2.9 GB")
        #expect(String(localized: "\(done) of \(total)", bundle: enBundle) == "1.7 GB of 2.9 GB")
        #expect(String(localized: "\(total) downloaded", bundle: zhBundle) == "已下载 2.9 GB")
        #expect(String(localized: "Downloading the model…", bundle: zhBundle) == "正在下载模型…")
        #expect(String(localized: "You can type while the model gets ready…", bundle: zhBundle) == "模型准备期间可以先输入…")
        #expect(String(localized: "Private AI, right on your Mac.", bundle: zhBundle) == "私密的 AI，就在你的 Mac 上。")
        #expect(String(localized: "Archived (\(count))", bundle: zhBundle) == "已归档 (3)")
        #expect(String(localized: "\(count) models", bundle: zhBundle) == "3 个模型")
        let tab = "Audio models"
        let one = 1
        #expect(String(localized: "\(one) matches in \(tab) →", bundle: enBundle) == "1 match in Audio models →")
        #expect(String(localized: "\(count) matches in \(tab) →", bundle: enBundle) == "3 matches in Audio models →")
        #expect(String(localized: "\(count) matches in \(tab) →", bundle: zhBundle) == "Audio models中有 3 个匹配项 →")
        #expect(String(localized: "Finding video models…", bundle: zhBundle) == "寻找视频模型…")
        #expect(String(localized: "Balanced capacity and memory use", bundle: zhBundle) == "平衡容量和内存使用")
        #expect(String(localized: "Cancel queued video?", bundle: zhBundle) == "取消排队的视频？")
        #expect(String(localized: "Finding video models…", bundle: enBundle) == "Finding video models…")
        #expect(String(localized: "Balanced capacity and memory use", bundle: enBundle) == "Balanced capacity and memory use")
        #expect(ExperimentalSurfaceCopy.videoProgress(42, bundle: zhBundle) == "生成中 · 42%")
        #expect(ExperimentalSurfaceCopy.videoMemory(minimum: 48, available: 32, bundle: zhBundle) == "至少需要 \(DownloadProgress.formatMemory(Int64(48) * (1 << 30))) 统一内存；这台 Mac 有 32 GB。")
        #expect(ExperimentalSurfaceCopy.benchmarkRounds(5, bundle: zhBundle) == "5 轮")
        #expect(ExperimentalSurfaceCopy.cuaStep(2, instruction: "点击继续", bundle: zhBundle) == "第 2 步：点击继续")
        #expect(ExperimentalSurfaceCopy.cuaStepFinished(2, bundle: zhBundle) == "第 2 步已完成")
        #expect(ExperimentalSurfaceCopy.cuaSwitched(to: "Safari", bundle: zhBundle) == "已切换到 Safari")
        #expect(ExperimentalSurfaceCopy.cuaRun("ended", bundle: zhBundle) == "运行状态：ended")
        #expect(ExperimentalSurfaceCopy.cuaAppUnavailable(hint: " 请检查窗口。", bundle: zhBundle) == "应用已不再打开。请打开后重试。 请检查窗口。")
        #expect(ExperimentalSurfaceCopy.cuaWindowUnavailable(hint: " 请检查窗口。", bundle: zhBundle) == "Rapid 找不到此任务所需的应用项目。请打开后重试。 请检查窗口。")
        #expect(ExperimentalSurfaceCopy.cuaTargetsUnavailable(message: "未找到", hint: " 请检查窗口。", bundle: zhBundle) == "Rapid 找不到此任务所需的应用：未找到 请检查窗口。")
        #expect(ExperimentalSurfaceCopy.cuaTargetsUnavailable(error: "连接失败", bundle: zhBundle) == "Rapid 找不到此任务所需的应用：连接失败 请重试。")
        #expect(ExperimentalSurfaceCopy.videoProgress(42, bundle: enBundle) == "Generating · 42%")
        #expect(ExperimentalSurfaceCopy.videoMemory(minimum: 48, available: 32, bundle: enBundle) == "Needs at least \(DownloadProgress.formatMemory(Int64(48) * (1 << 30))) unified memory; this Mac has 32 GB.")
    }

    @Test("Share Compute checklist and session labels resolve before String view helpers")
    func compiledShareComputeLabels() async throws {
        let output = FileManager.default.temporaryDirectory.appendingPathComponent("rapid-localization-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: output) }
        let compiler = try await TestSubprocess.run(executableURL: URL(fileURLWithPath: "/usr/bin/xcrun"), arguments: ["xcstringstool", "compile", try catalogURL().path, "--output-directory", output.path, "--serialization-format", "binary"])
        #expect(compiler.terminationStatus == 0)
        let zhBundle = try #require(Bundle(url: output.appendingPathComponent("zh-Hans.lproj", isDirectory: true)))
        let checklist = try String(contentsOf: rapidSource("ShareCompute/ShareComputeShareTab.swift"), encoding: .utf8)
        #expect(checklist.contains("String(localized: \"QuickSilver provider key\")"))
        #expect(String(localized: "QuickSilver provider key", bundle: zhBundle) == "QuickSilver 提供商密钥")
        #expect(checklist.contains("String(localized: \"Model to share\")"))
        #expect(String(localized: "Model to share", bundle: zhBundle) == "要分享的模型")
        #expect(checklist.contains("String(localized: \"Start sharing\")"))
        #expect(String(localized: "Start sharing", bundle: zhBundle) == "开始分享")
        let session = try String(contentsOf: rapidSource("ShareCompute/ShareComputeSessionViews.swift"), encoding: .utf8)
        #expect(session.contains("String(localized: \"Time shared\")"))
        #expect(String(localized: "Time shared", bundle: zhBundle) == "共享时间")
        #expect(session.contains("String(localized: \"Requests now\")"))
        #expect(String(localized: "Requests now", bundle: zhBundle) == "当前请求数")
        #expect(session.contains("String(localized: \"Provider status\")"))
        #expect(String(localized: "Provider status", bundle: zhBundle) == "提供商状态")
        #expect(session.contains("String(localized: \"Worker\")"))
        #expect(String(localized: "Worker", bundle: zhBundle) == "工作节点")
        #expect(session.contains("String(localized: \"Model after sharing\")"))
        #expect(String(localized: "Model after sharing", bundle: zhBundle) == "分享结束后的模型状态")
        #expect(session.contains("String(localized: \"Reconnecting\")"))
        #expect(String(localized: "Reconnecting", bundle: zhBundle) == "重新连接")
        #expect(session.contains("String(localized: \"Connected\")"))
        #expect(String(localized: "Connected", bundle: zhBundle) == "已连接")
    }

    @Test("Images and Share Compute String copy resolves against compiled zh-Hans")
    func compiledImagesAndShareComputeStringCopy() async throws {
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-localization-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: output) }
        let compiler = try await TestSubprocess.run(
            executableURL: URL(fileURLWithPath: "/usr/bin/xcrun"),
            arguments: ["xcstringstool", "compile", try catalogURL().path,
                        "--output-directory", output.path, "--serialization-format", "binary"]
        )
        try #require(compiler.terminationStatus == 0)
        let zhBundle = try #require(Bundle(url: output.appendingPathComponent("zh-Hans.lproj", isDirectory: true)))
        // Keep the source literal and the compiler-derived catalog key separate:
        // interpolations must remain localizable before they become plain String.
        let contracts: [(path: String, literal: String, key: String, zh: String)] = [
            ("UI/ImagesView.swift", #"Draw anything"#, "Draw anything", "随心绘制"),
            ("UI/ImagesView.swift", #"Describe what you want to see, then press Generate."#, "Describe what you want to see, then press Generate.", "描述你想看到的画面，然后点击生成。"),
            ("UI/ImagesView.swift", #"Create images locally, then keep generating offline."#, "Create images locally, then keep generating offline.", "在本地生成图像，离线也能继续创作。"),
            ("UI/ImagesView.swift", #"Pick a starter below while Rapid gets the model ready."#, "Pick a starter below while Rapid gets the model ready.", "Rapid 准备模型期间，可以先从下方选一个示例。"),
            ("UI/ImagesView.swift", #"Describe what you want to change…"#, "Describe what you want to change…", "描述你想要的修改…"),
            ("UI/ImagesView.swift", #"Describe the image you want…"#, "Describe the image you want…", "描述你想要的图像…"),
            ("UI/ImagesView.swift", #"512² default"#, "512² default", "默认 512²"),
            ("UI/ImagesView.swift", #"\(entry.imageDefaultSteps ?? ImageGenViewModel.seedSteps(for: entry.alias)) steps"#, "%lld steps", "%lld 步"),
            ("UI/ImagesView.swift", #"Downloading or loading the image model…"#, "Downloading or loading the image model…", "正在下载或加载图像模型…"),
            ("UI/ImagesView.swift", #"Stopping…"#, "Stopping…", "正在停止…"),
            ("UI/ImagesView.swift", #"Finalizing image…"#, "Finalizing image…", "正在完成图像…"),
            ("UI/ImagesView.swift", #"Generating"#, "Generating", "生成中"),
            ("UI/ImagesView.swift", #"Warming up \(viewModel.selectedDisplayName)"#, "Warming up %@", "正在预热 %@"),
            ("UI/ImagesView.swift", #"Decoding and saving…"#, "Decoding and saving…", "正在解码并保存…"),
            ("UI/ImagesView.swift", #"First run — only happens once"#, "First run — only happens once", "首次运行 — 仅需一次"),
            ("UI/ImagesView.swift", #"Estimating…"#, "Estimating…", "正在估算…"),
            ("UI/ImagesView.swift", #"~\(max(1, Int(secondsRemaining.rounded())))s left"#, "~%llds left", "剩余约 %lld 秒"),
            ("UI/ImagesView.swift", #"Image \(ordinal), \(image.prompt), selected"#, "Image %lld, %@, selected", "图像 %lld，%@，已选中"),
            ("UI/ImagesView.swift", #"Image \(ordinal), \(image.prompt)"#, "Image %lld, %@", "图像 %lld，%@"),
            ("UI/ImagesView.swift", #"Replace source image"#, "Replace source image", "替换源图像"),
            ("UI/ImagesView.swift", #"Import image to edit"#, "Import image to edit", "导入要编辑的图像"),
            ("UI/ImagesView.swift", #"No image editing models available"#, "No image editing models available", "没有可用的图像编辑模型"),
            ("UI/ImagesView.swift", #"No image generation models available"#, "No image generation models available", "没有可用的图像生成模型"),
            ("UI/ImagesView.swift", #"Loading…"#, "Loading…", "加载中…"),
            ("UI/ImagesView.swift", #"Choose a model"#, "Choose a model", "选择模型"),
            ("UI/ImagesView.swift", #"Model: \(viewModel.selectedAlias)"#, "Model: %@", "模型：%@"),
            ("UI/ImagesView.swift", #"Model: \(ImageReadinessPresentation.rowTitle(entry))"#, "Model: %@", "模型：%@"),
            ("UI/ImagesView.swift", #"Edit image"#, "Edit image", "编辑图像"),
            ("UI/ImagesView.swift", #"Generate"#, "Generate", "生成"),
            ("UI/ImagesView.swift", #"Load the model first"#, "Load the model first", "请先加载模型"),
            ("UI/ImagesView.swift", #"Couldn't save the image: \(error.localizedDescription)"#, "Couldn't save the image: %@", "无法保存图像：%@"),
            ("UI/ImagesView.swift", #"Couldn't import the image: \(error.localizedDescription)"#, "Couldn't import the image: %@", "无法导入图像：%@"),
            ("UI/ImagesView.swift", #"Choose an image smaller than \(DownloadProgress.formatBytes(Int64(ImageClient.maxEditImageBytes)))."#, "Choose an image smaller than %@.", "请选择小于 %@ 的图像。"),
            ("UI/ImagesView.swift", #"Choose an image no larger than 8192 px or 40 megapixels."#, "Choose an image no larger than 8192 px or 40 megapixels.", "请选择尺寸不超过 8192 像素且总像素不超过 4000 万的图像。"),
            ("UI/ImagesView.swift", #"Choose a PNG or JPEG image."#, "Choose a PNG or JPEG image.", "请选择 PNG 或 JPEG 图像。"),
            ("UI/ImagesView.swift", #"The selected file isn't a readable image."#, "The selected file isn't a readable image.", "无法读取所选文件中的图像。"),
            ("Images/ImageClient.swift", #"The image model isn't running yet."#, "The image model isn't running yet.", "图像模型尚未运行。"),
            ("Images/ImageClient.swift", #"Image request failed (HTTP \(status))."#, "Image request failed (HTTP %lld).", "图像请求失败（HTTP %lld）。"),
            ("Images/ImageClient.swift", #"The server returned no image."#, "The server returned no image.", "服务器未返回图像。"),
            ("Images/ImageClient.swift", #"Malformed server response."#, "Malformed server response.", "服务器响应格式错误。"),
            ("Images/ImageGenViewModel.swift", #"the model"#, "the model", "模型"),
            ("ShareCompute/ShareComputeShareTab.swift", #"Key saved"#, "Key saved", "密钥已保存"),
            ("ShareCompute/ShareComputeShareTab.swift", #"Ready on this Mac"#, "Ready on this Mac", "在这台 Mac 上准备就绪"),
            ("ShareCompute/ShareComputeShareTab.swift", #"Download needed"#, "Download needed", "需要下载"),
            ("ShareCompute/ShareComputeShareTab.swift", #"Retry"#, "Retry", "重试"),
            ("ShareCompute/ShareComputeShareTab.swift", #"Download"#, "Download", "下载"),
            ("ShareCompute/ShareComputeShareTab.swift", #"This Mac goes online and earns API credits for served tokens."#, "This Mac goes online and earns API credits for served tokens.", "这台 Mac 将上线，并按处理的 token 赚取 API 积分。"),
            ("ShareCompute/ShareComputeShareTab.swift", #"Complete the steps above to go online."#, "Complete the steps above to go online.", "完成上方步骤即可上线。"),
            ("ShareCompute/ShareComputeSessionViews.swift", #"Reconnecting…"#, "Reconnecting…", "正在重新连接…"),
            ("ShareCompute/ShareComputeSessionViews.swift", #"Sharing"#, "Sharing", "正在分享"),
            ("ShareCompute/ShareComputeSessionViews.swift", #"Reconnecting to the relay."#, "Reconnecting to the relay.", "正在重新连接中继。"),
            ("ShareCompute/ShareComputeSessionViews.swift", #"Your Mac is online and serving requests."#, "Your Mac is online and serving requests.", "你的 Mac 已上线，正在处理请求。"),
            ("ShareCompute/ShareComputeManager.swift", #"Rapid-MLX's local engine is not available."#, "Rapid-MLX's local engine is not available.", "Rapid-MLX 的本地引擎不可用。"),
            ("ShareCompute/ShareComputeManager.swift", #"That model is not available in the QuickSilver pool."#, "That model is not available in the QuickSilver pool.", "QuickSilver 池中没有该模型。"),
            ("ShareCompute/ShareComputeManager.swift", #"The provider key couldn't be saved."#, "The provider key couldn't be saved.", "无法保存提供商密钥。"),
            ("ShareCompute/ShareComputeManager.swift", #"Save your QuickSilver provider key to connect this Mac."#, "Save your QuickSilver provider key to connect this Mac.", "保存你的 QuickSilver 提供商密钥以连接这台 Mac。"),
            ("ShareCompute/ShareComputeManager.swift", #"Rapid couldn't unload the current model safely."#, "Rapid couldn't unload the current model safely.", "Rapid 无法安全卸载当前模型。"),
            ("ShareCompute/ShareComputeManager.swift", #"Rapid couldn't start Share Compute."#, "Rapid couldn't start Share Compute.", "Rapid 无法启动共享算力。"),
            ("ShareCompute/ShareComputeManager.swift", #"Share Compute stopped unexpectedly."#, "Share Compute stopped unexpectedly.", "共享算力意外停止。"),
            ("ShareCompute/ShareComputeManager.swift", #"Share Compute reported an unsupported status."#, "Share Compute reported an unsupported status.", "共享算力报告了不支持的状态。"),
        ]
        for contract in contracts {
            let source = try String(contentsOf: rapidSource(contract.path), encoding: .utf8)
            #expect(
                source.contains(#"String(localized: ""# + contract.literal + #"")"#),
                "Missing localization lookup in \(contract.path): \(contract.literal)"
            )
            #expect(zhBundle.localizedString(forKey: contract.key, value: nil, table: nil) == contract.zh)
        }
        // Exercise real interpolations as well as the raw compiled templates.
        let ordinal = 2, steps = 8, status = 500
        let prompt = "a whale", alias = "z-image"
        #expect(String(localized: "\(steps) steps", bundle: zhBundle) == "8 步")
        #expect(String(localized: "Image \(ordinal), \(prompt), selected", bundle: zhBundle) == "图像 2，a whale，已选中")
        #expect(String(localized: "Image request failed (HTTP \(status)).", bundle: zhBundle) == "图像请求失败（HTTP 500）。")
        #expect(String(localized: "Model: \(alias)", bundle: zhBundle) == "模型：z-image")
    }

    @Test("Share Compute Credits captions resolve against the compiled zh-Hans table")
    func compiledShareComputeCreditsCopy() async throws {
        let output = FileManager.default.temporaryDirectory.appendingPathComponent("rapid-localization-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: output) }
        let compiler = try await TestSubprocess.run(executableURL: URL(fileURLWithPath: "/usr/bin/xcrun"), arguments: ["xcstringstool", "compile", try catalogURL().path, "--output-directory", output.path, "--serialization-format", "binary"])
        #expect(compiler.terminationStatus == 0, "xcstringstool failed: \(String(decoding: compiler.standardError, as: UTF8.self))")
        let zhBundle = try #require(Bundle(url: output.appendingPathComponent("zh-Hans.lproj", isDirectory: true)))
        let enBundle = try #require(Bundle(url: output.appendingPathComponent("en.lproj", isDirectory: true)))
        #expect(String(localized: "Credits earned", bundle: zhBundle) == "获得的积分")
        #expect(String(localized: "Requests served", bundle: zhBundle) == "处理的请求数")
        #expect(String(localized: "Tokens (in / out)", bundle: zhBundle) == "token（输入 / 输出）")
        #expect(String(localized: "No QuickSilver key yet", bundle: zhBundle) == "还没有 QuickSilver 密钥")
        #expect(String(localized: "Reading your ledger…", bundle: zhBundle) == "正在读取账本…")
        #expect(ShareComputeCreditsCopy.credits("2.00", bundle: zhBundle) == "2.00 积分")
        #expect(ShareComputeCreditsCopy.smallCredit(negative: false, bundle: zhBundle) == "<0.01 积分")
        #expect(ShareComputeCreditsCopy.smallCredit(negative: true, bundle: zhBundle) == "-<0.01 积分")
        #expect(ShareComputeCreditsCopy.caps(node: "A", pool: "B", bundle: zhBundle) == "上限：每个节点 A · 每个池 B")
        #expect(ShareComputeCreditsCopy.scope(1, bundle: zhBundle) == "整个账户 · 1 个节点")
        #expect(ShareComputeCreditsCopy.scope(2, bundle: zhBundle) == "整个账户 · 2 个节点")
        #expect(ShareComputeCreditsCopy.truncation(bundle: zhBundle) == "仅显示最近的窗口。")
        #expect(ShareComputeCreditsCopy.shown(6, total: 9, bundle: zhBundle) == "显示 6 / 9 个结算窗口 · 总计包含全部窗口")
        #expect(ShareComputeCreditsCopy.windows(1, bundle: zhBundle) == "1 个结算窗口")
        #expect(ShareComputeCreditsCopy.windows(3, bundle: zhBundle) == "3 个结算窗口")
        #expect(ShareComputeCreditsCopy.summary(sessions: 2, duration: "1h", models: 3, bundle: zhBundle) == "2 个连接时段 · 在线 1h · 服务了 3 个模型")
        #expect(ShareComputeCreditsCopy.row(requests: 4, credit: "2 积分", bundle: zhBundle) == "4 次请求 · 2 积分")
        #expect(ShareComputeCreditsCopy.accessibility(model: "M", node: "N", requests: 2, input: 3, output: 4, credit: "C", accrued: "A", status: "S", bundle: zhBundle) == "M，节点 N，2 次请求，输入 3 个 token，输出 4 个 token，积分 C，累计 A，状态 S")
        #expect(ShareComputeCreditsCopy.scope(2, bundle: enBundle) == "Account-wide · 2 nodes")
        #expect(ShareComputeCreditsCopy.windows(1, bundle: enBundle) == "1 accounting window")
        let source = try String(contentsOf: rapidSource("ShareCompute/ShareComputeCreditsView.swift"), encoding: .utf8)
        #expect(source.range(of: #"Text\("[^"\n]*\\\("#, options: .regularExpression) == nil)
        #expect(!source.contains("metric(\""))
        #expect(source.range(of: #"note\("[^"\n]*",\s*""#, options: .regularExpression) == nil)
    }

    @Test("Compiled zh-Hans catalog resolves through the production photo-hint path")
    func compiledCatalogLocalizesPhotoHints() async throws {
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-localization-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: output) }

        let compiler = try await TestSubprocess.run(
            executableURL: URL(fileURLWithPath: "/usr/bin/xcrun"),
            arguments: [
                "xcstringstool", "compile", try catalogURL().path,
                "--output-directory", output.path,
                "--serialization-format", "binary"
            ]
        )
        #expect(
            compiler.terminationStatus == 0,
            "xcstringstool failed: \(String(decoding: compiler.standardError, as: UTF8.self))"
        )

        let zhBundle = try #require(
            Bundle(url: output.appendingPathComponent("zh-Hans.lproj", isDirectory: true)),
            "xcstringstool did not emit a loadable zh-Hans localization bundle"
        )
        let memory = ImageInputAvailability.resolve(
            fallbackSupportsImageInput: true,
            profile: ServerModelProfile(
                id: "model",
                capabilities: ["text", "vision"],
                servingLane: "text",
                servingLaneReason: "vision_memory_insufficient"
            ),
            localizationBundle: zhBundle
        )
        #expect(
            memory.unavailableMessage
                == "此模型的文字聊天可以正常使用。照片模式需要的内存超过这台 Mac 的容量；如需添加照片，请选择内存需求更低的视觉模型。"
        )

        let legacy = ImageInputAvailability.resolve(
            fallbackSupportsImageInput: false,
            profile: nil,
            localizationBundle: zhBundle
        )
        #expect(legacy.unavailableMessage == "此模型不支持照片。要添加照片，请选择支持视觉的模型。")
    }
}
