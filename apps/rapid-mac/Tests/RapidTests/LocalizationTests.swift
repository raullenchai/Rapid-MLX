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
            let unit = try #require(zh["stringUnit"] as? [String: Any], "Missing stringUnit for \(key)")
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
        #expect(String(localized: "Archived (\(count))", bundle: zhBundle) == "已归档 (3)")
        #expect(String(localized: "\(count) models", bundle: zhBundle) == "3 个模型")
        #expect(String(localized: "Finding video models…", bundle: zhBundle) == "寻找视频模型…")
        #expect(String(localized: "Balanced capacity and memory use", bundle: zhBundle) == "平衡容量和内存使用")
        #expect(String(localized: "Cancel queued video?", bundle: zhBundle) == "取消排队的视频？")
        #expect(String(localized: "Finding video models…", bundle: enBundle) == "Finding video models…")
        #expect(String(localized: "Balanced capacity and memory use", bundle: enBundle) == "Balanced capacity and memory use")
        #expect(ExperimentalSurfaceCopy.videoProgress(42, bundle: zhBundle) == "生成中 · 42%")
        #expect(ExperimentalSurfaceCopy.videoMemory(minimum: 48, available: 32, bundle: zhBundle) == "至少需要 48 GB 统一内存；这台 Mac 有 32 GB。")
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
        #expect(ExperimentalSurfaceCopy.videoMemory(minimum: 48, available: 32, bundle: enBundle) == "Needs at least 48 GB unified memory; this Mac has 32 GB.")
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
