import Foundation

/// Telemetry v2 event registry — the desktop half of the strict validator.
///
/// This reads the SAME physical `events.json` the engine reads
/// (`rapid_mlx/telemetry/events.json`). There is no second copy in this
/// package: `scripts/build.sh` copies that one file into the shipped `.app`,
/// SwiftPM tests resolve it straight out of the source checkout, and
/// `tests/test_telemetry_registry_drift.py` fails if a duplicate ever appears.
/// The enums deliberately live in the JSON and nowhere in Swift source —
/// a second hand-maintained copy is exactly the drift this block prevents.
///
/// Strict semantics, identical to `rapid_mlx/telemetry/registry.py` and to
/// Orca's `src/main/telemetry/validator.ts`. Every validation failure drops
/// the WHOLE event. A registry-declared `only_when` condition is the sole
/// filtering rule: a valid property is omitted when its condition is false.
///
/// - unknown event name                       -> `nil`
/// - unknown property key                     -> `nil` (event dropped)
/// - missing required property                -> `nil`
/// - wrong type / out-of-enum / out-of-range  -> `nil`
/// - string past its length cap or pattern    -> `nil`
///
/// Nothing here transmits. It only answers "is this shape allowed on the
/// wire?"; the PostHog transport lands in a later block.
enum TelemetryValue: Equatable, Sendable {
    case bool(Bool)
    case int(Int)
    case string(String)
}

/// Anchor class for the SPM resource-bundle walk — same trick
/// `BenchScores` uses, for the same reason (`Bundle.module` assert-crashes
/// on a miss, and the production `.app` has flat resources).
private final class TelemetryRegistryBundleFinder {}

struct TelemetryRegistry: Sendable {
    /// One property's declared shape. The `kind` set is closed: `enum`,
    /// `bool`, `int` and `model_id` for event properties, plus `version`
    /// and `uuid` which only the common props may use.
    struct PropertySpec: Sendable {
        let kind: String
        let required: Bool
        let enumName: String?
        let min: Int?
        let max: Int?
        let pattern: String?
        let maxLength: Int?
        let onlyWhen: [String: [String]]?
    }

    struct ModelIDSpec: Sendable {
        let pattern: String
        let maxLength: Int
    }

    let registryVersion: Int
    let enums: [String: [String]]
    let modelID: ModelIDSpec
    let commonProps: [String: PropertySpec]
    let events: [String: [String: PropertySpec]]

    // MARK: - Loading

    /// The bundled registry, parsed once. `nil` when the resource is
    /// missing or malformed — that is a packaging bug, and every
    /// `validate` call then fails closed rather than guessing.
    static let shared: TelemetryRegistry? = load()

    static func resourceURL() -> URL? {
        if let url = Bundle.main.url(forResource: "events", withExtension: "json") {
            return url
        }
        let anchor = Bundle(for: TelemetryRegistryBundleFinder.self)
            .bundleURL
            .deletingLastPathComponent()
        let bundleURL = anchor.appendingPathComponent("Rapid_Rapid.bundle")
        if let bundle = Bundle(url: bundleURL),
           let url = bundle.url(forResource: "events", withExtension: "json") {
            return url
        }
        if let url = Bundle(for: TelemetryRegistryBundleFinder.self)
            .url(forResource: "events", withExtension: "json") {
            return url
        }
        // SwiftPM tests run from a source checkout. Keep the source fallback
        // pointed at the Python package's canonical file; it is deliberately
        // not a second copied registry (same rule as
        // `model_recommendations.json` in `RAMBucketedDefault`).
        let sourceCandidate = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Telemetry
            .deletingLastPathComponent() // Rapid
            .deletingLastPathComponent() // Sources
            .deletingLastPathComponent() // rapid-mac
            .deletingLastPathComponent() // apps
            .appendingPathComponent("rapid_mlx/telemetry/events.json")
        if FileManager.default.fileExists(atPath: sourceCandidate.path) {
            return sourceCandidate
        }
        // Some SwiftPM invocations preserve a relative `#filePath`. Walk
        // upward from the working directory instead.
        var directory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        for _ in 0..<6 {
            let candidate = directory
                .appendingPathComponent("rapid_mlx/telemetry/events.json")
            if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
            directory.deleteLastPathComponent()
        }
        return nil
    }

    static func load() -> TelemetryRegistry? {
        guard let url = resourceURL(), let data = try? Data(contentsOf: url) else {
            return nil
        }
        return decode(data)
    }

    /// Exposed so tests can parse the engine's copy straight off disk and
    /// prove both halves read the same bytes.
    static func decode(_ data: Data) -> TelemetryRegistry? {
        guard
            let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
            let version = root["registry_version"] as? Int,
            let rawEnums = root["enums"] as? [String: Any],
            let rawModelID = root["model_id"] as? [String: Any],
            let pattern = rawModelID["pattern"] as? String,
            let maxLength = rawModelID["max_length"] as? Int,
            let rawCommon = root["common_props"] as? [String: Any],
            let rawEvents = root["events"] as? [String: Any]
        else { return nil }

        var enums: [String: [String]] = [:]
        for (name, value) in rawEnums {
            guard name.hasPrefix("_") == false else { continue }
            guard
                let body = value as? [String: Any],
                let values = body["values"] as? [String]
            else { return nil }
            enums[name] = values
        }

        var events: [String: [String: PropertySpec]] = [:]
        for (name, value) in rawEvents {
            guard name.hasPrefix("_") == false else { continue }
            guard
                let body = value as? [String: Any],
                let rawProps = body["props"] as? [String: Any]
            else { return nil }
            guard let props = decodeSpecs(rawProps) else { return nil }
            guard validateOnlyWhen(props, enums: enums) else { return nil }
            events[name] = props
        }

        guard let common = decodeSpecs(rawCommon) else { return nil }

        return TelemetryRegistry(
            registryVersion: version,
            enums: enums,
            modelID: ModelIDSpec(pattern: pattern, maxLength: maxLength),
            commonProps: common,
            events: events
        )
    }

    /// `_`-prefixed keys are documentation for human readers (the JSON has
    /// no comment syntax) and are skipped everywhere, in both languages.
    private static func decodeSpecs(_ raw: [String: Any]) -> [String: PropertySpec]? {
        var out: [String: PropertySpec] = [:]
        for (name, value) in raw where name.hasPrefix("_") == false {
            guard let body = value as? [String: Any], let kind = body["kind"] as? String
            else { return nil }
            let onlyWhen: [String: [String]]?
            if let rawOnlyWhen = body["only_when"] {
                guard let decoded = rawOnlyWhen as? [String: [String]] else { return nil }
                onlyWhen = decoded
            } else {
                onlyWhen = nil
            }
            out[name] = PropertySpec(
                kind: kind,
                required: body["required"] as? Bool ?? false,
                enumName: body["enum"] as? String,
                min: body["min"] as? Int,
                max: body["max"] as? Int,
                pattern: body["pattern"] as? String,
                maxLength: body["max_length"] as? Int,
                onlyWhen: onlyWhen
            )
        }
        return out
    }

    /// Conditional metadata is registry schema, so validate it at decode time
    /// even when callers omit the conditional property from an event.
    private static func validateOnlyWhen(
        _ specs: [String: PropertySpec],
        enums: [String: [String]]
    ) -> Bool {
        for spec in specs.values {
            guard let onlyWhen = spec.onlyWhen else { continue }
            guard onlyWhen.isEmpty == false else { return false }
            for (controller, allowed) in onlyWhen {
                guard controller.isEmpty == false,
                      allowed.isEmpty == false,
                      allowed.allSatisfy({ $0.isEmpty == false }),
                      let controllerSpec = specs[controller],
                      controllerSpec.kind == "enum",
                      let enumName = controllerSpec.enumName,
                      let controllerValues = enums[enumName],
                      allowed.allSatisfy(controllerValues.contains)
                else { return false }
            }
        }
        return true
    }

    // MARK: - Validation

    func validate(_ eventName: String, _ props: [String: TelemetryValue])
        -> [String: TelemetryValue]?
    {
        guard let specs = events[eventName] else {
            TelemetryRegistryLog.once("<unknown-event>", "unknown event \(eventName)")
            return nil
        }
        return validate(props: props, against: specs, label: eventName)
    }

    func validateCommon(_ props: [String: TelemetryValue]) -> [String: TelemetryValue]? {
        validate(props: props, against: commonProps, label: "common_props")
    }

    private func validate(
        props: [String: TelemetryValue],
        against specs: [String: PropertySpec],
        label: String
    ) -> [String: TelemetryValue]? {
        for key in props.keys where specs[key] == nil {
            // Whole event dropped, NOT the key stripped: a caller that got
            // one key wrong has told us nothing about the rest.
            TelemetryRegistryLog.once(label, "\(label): unknown property \(key) — event dropped")
            return nil
        }

        var out: [String: TelemetryValue] = [:]
        for (name, spec) in specs {
            guard let value = props[name] else {
                if spec.required {
                    TelemetryRegistryLog.once(label, "\(label): missing required \(name)")
                    return nil
                }
                continue
            }
            if let onlyWhen = spec.onlyWhen {
                let conditionMet = onlyWhen.allSatisfy { controller, allowed in
                    guard case .string(let actual)? = props[controller] else { return false }
                    return allowed.contains(actual)
                }
                if conditionMet == false { continue }
            }
            guard check(value, against: spec) else {
                TelemetryRegistryLog.once(label, "\(label): property \(name) rejected")
                return nil
            }
            out[name] = value
        }
        return out
    }

    private func check(_ value: TelemetryValue, against spec: PropertySpec) -> Bool {
        switch spec.kind {
        case "bool":
            if case .bool = value { return true }
            return false
        case "int":
            guard case .int(let n) = value, let lo = spec.min, let hi = spec.max else {
                return false
            }
            return n >= lo && n <= hi
        case "enum":
            guard case .string(let s) = value,
                  let name = spec.enumName,
                  let values = enums[name]
            else { return false }
            return values.contains(s)
        case "model_id":
            guard case .string(let s) = value, s.count <= modelID.maxLength else {
                return false
            }
            return Self.matches(s, modelID.pattern)
        case "version":
            guard case .string(let s) = value,
                  let pattern = spec.pattern,
                  s.count <= (spec.maxLength ?? 64)
            else { return false }
            return Self.matches(s, pattern)
        case "uuid":
            guard case .string(let s) = value, s.count <= (spec.maxLength ?? 64) else {
                return false
            }
            return Self.matches(s, Self.uuidPattern)
        default:
            // An unrecognised kind means the registry itself is malformed.
            // Fail closed rather than let an unchecked value through.
            return false
        }
    }

    /// RFC 4122 shape, any version — the same expression the Python
    /// validator applies to `install_id` / `session_id`.
    static let uuidPattern =
        "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"

    /// Whole-string match. The JSON patterns are already `^…$`-anchored for
    /// Python's `re.fullmatch`; `NSRegularExpression` tolerates the anchors
    /// and the explicit range check makes the intent independent of them.
    private static func matches(_ value: String, _ pattern: String) -> Bool {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return false }
        let range = NSRange(value.startIndex..<value.endIndex, in: value)
        guard let match = regex.firstMatch(in: value, options: [], range: range) else {
            return false
        }
        return match.range == range
    }
}

/// Rate-limited debug log: one line per label per process, matching the
/// Python side. A misbehaving caller must not be able to flood the app log.
private enum TelemetryRegistryLog {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var seen: Set<String> = []

    static func once(_ key: String, _ message: String) {
        lock.lock()
        let isNew = seen.insert(key).inserted
        lock.unlock()
        guard isNew else { return }
        #if DEBUG
        FileHandle.standardError.write(Data("[telemetry] registry: \(message)\n".utf8))
        #endif
    }
}
