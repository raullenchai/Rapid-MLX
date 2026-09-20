import Foundation
import Testing
@testable import Rapid

/// Telemetry v2 registry — the desktop half of the strict validator.
///
/// Every test here is written so that removing the rule it guards turns it
/// red; the fault-injection pass before merge confirmed that one by one.
/// The point of the suite is not that `events.json` parses — it is that the
/// app cannot put an unreviewed shape on the wire.
@Suite("TelemetryRegistry — strict validation of the shared events.json")
struct TelemetryRegistryTests {

    /// The shared registry, resolved through the same lookup the app uses.
    private var registry: TelemetryRegistry {
        guard let registry = TelemetryRegistry.shared else {
            fatalError("events.json is missing — check scripts/build.sh and the source fallback")
        }
        return registry
    }

    private var validServe: [String: TelemetryValue] {
        [
            "model": .string("qwen3.5-4b-4bit"),
            "model_type": .string("llm"),
            "auto_selected": .bool(true),
            "quant": .string("4bit")
        ]
    }

    // MARK: - Loading

    @Test("the registry resolves and carries the release-1 event set")
    func registryLoads() {
        #expect(registry.registryVersion == 1)
        for name in [
            "app_opened", "active_day", "model_pulled", "model_pull_failed",
            "model_served", "model_serve_failed", "capability_rejected",
            "inference_bucket_reached", "agent_configured",
            "agent_configure_failed", "telemetry_opted_out",
            "telemetry_opted_in"
        ] {
            #expect(registry.events[name] != nil, "missing event \(name)")
        }
    }

    @Test("the Swift loader reads the engine's canonical file, not a copy")
    func readsEngineCanonicalFile() throws {
        let url = try #require(TelemetryRegistry.resourceURL())
        // In a source checkout the fallback must land on the Python
        // package's file. In a built .app it is the flat bundle copy that
        // scripts/build.sh took from that same file.
        #expect(
            url.path.contains("rapid_mlx/telemetry/events.json")
                || url.path.hasSuffix("Resources/events.json")
        )
        let data = try Data(contentsOf: url)
        #expect(TelemetryRegistry.decode(data) != nil)
    }

    // MARK: - Strictness

    @Test("a well-formed event passes through unchanged")
    func acceptsValidEvent() throws {
        let out = try #require(registry.validate("model_served", validServe))
        #expect(out.count == 4)
        #expect(out["quant"] == .string("4bit"))
    }

    @Test("an unknown event name is dropped")
    func rejectsUnknownEvent() {
        #expect(registry.validate("model_teleported", [:]) == nil)
    }

    @Test("an unknown property drops the WHOLE event, it is not stripped")
    func rejectsUnknownProperty() {
        var props = validServe
        props["prompt"] = .string("hello")
        #expect(registry.validate("model_served", props) == nil)
    }

    @Test("a missing required property is dropped")
    func rejectsMissingRequired() {
        var props = validServe
        props.removeValue(forKey: "model_type")
        #expect(registry.validate("model_served", props) == nil)
    }

    @Test("a value outside the declared enum is dropped")
    func rejectsOutOfEnum() {
        var props = validServe
        props["model_type"] = .string("telepathy")
        #expect(registry.validate("model_served", props) == nil)
    }

    @Test("a wrong-typed value is dropped")
    func rejectsWrongType() {
        var props = validServe
        props["auto_selected"] = .string("true")
        #expect(registry.validate("model_served", props) == nil)
    }

    @Test("a model id outside the pattern or over the cap is dropped")
    func rejectsBadModelID() {
        for bad in [
            "/Users/someone/models/secret",
            "org/name/extra",
            "my model",
            String(repeating: "a", count: 200)
        ] {
            var props = validServe
            props["model"] = .string(bad)
            #expect(registry.validate("model_served", props) == nil, "accepted \(bad)")
        }
        for good in ["qwen3.5-4b-4bit", "mlx-community/Qwen3.5-4B-MLX-4bit", "<custom>", "<local>"] {
            var props = validServe
            props["model"] = .string(good)
            #expect(registry.validate("model_served", props) != nil, "rejected \(good)")
        }
    }

    @Test("an out-of-range int is dropped, and a bool is not an int")
    func rejectsOutOfRangeInt() {
        var common = validCommon
        common["memory_gb"] = .int(-1)
        #expect(registry.validateCommon(common) == nil)

        common = validCommon
        common["memory_gb"] = .int(999_999)
        #expect(registry.validateCommon(common) == nil)

        common = validCommon
        common["memory_gb"] = .bool(true)
        #expect(registry.validateCommon(common) == nil)
    }

    @Test("a _failed twin accepts error_class alone and rejects it missing")
    func failedTwinContract() {
        #expect(registry.validate("model_serve_failed", ["error_class": .string("insufficient_memory")]) != nil)
        #expect(registry.validate("model_serve_failed", ["model": .string("<custom>")]) == nil)
    }

    // MARK: - Common props

    private var validCommon: [String: TelemetryValue] {
        [
            "app_version": .string("0.15.0"),
            "surface": .string("desktop"),
            "os": .string("darwin"),
            "os_version": .string("25.3"),
            "arch": .string("arm64"),
            "chip": .string("m3-max"),
            "memory_gb": .int(64),
            "install_id": .string("6F1B1D3E-4A2B-4C9D-8E7F-0A1B2C3D4E5F"),
            "session_id": .string("6f1b1d3e-4a2b-4c9d-8e7f-0a1b2c3d4e5f"),
            "channel": .string("stable"),
            "nth_model_served": .int(3),
            "days_since_first_run_bucket": .string("2-6")
        ]
    }

    @Test("the desktop envelope validates without python_version")
    func acceptsDesktopCommonProps() throws {
        let out = try #require(registry.validateCommon(validCommon))
        #expect(out["python_version"] == nil)
    }

    @Test("a non-UUID install id is dropped")
    func rejectsBadInstallID() {
        var common = validCommon
        common["install_id"] = .string("not-a-uuid")
        #expect(registry.validateCommon(common) == nil)
    }

    @Test("an unknown common property drops the envelope")
    func rejectsUnknownCommonProperty() {
        var common = validCommon
        common["country"] = .string("US")
        #expect(registry.validateCommon(common) == nil)
    }
}
