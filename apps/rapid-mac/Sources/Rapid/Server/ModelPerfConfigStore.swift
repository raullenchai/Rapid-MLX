import Foundation
import Observation

/// Issue #1717: per-model performance overrides, persisted across launches and
/// resolved into `serve` argv at spawn.
///
/// **Per-model, not global** — the issue's requirement, and the reason this is
/// a dictionary keyed by alias rather than a flat set of preferences: "the
/// right KV setting for a 4B dense model is not the right one for a 35B MoE".
/// It mirrors how ``SamplingConfig`` already carries per-alias context and
/// reasoning floors.
///
/// **Sparse.** Only aliases the user has actually touched get a row, and an
/// override that is reset back to "no opinion" removes the row rather than
/// writing an explicit copy of the current default. Two consequences the issue
/// asks for: an untouched install contributes no flags at all, and an engine
/// default that changes in a later release reaches every user who never
/// expressed an opinion.
///
/// Storage is a single `UserDefaults` JSON blob rather than a file under
/// `~/.config`, unlike ``MCPConfigStore``. The engine does not read this — it
/// is desktop-side state that becomes argv — so there is no interop shape to
/// honour, and `UserDefaults` keeps it inside the app sandbox with the rest of
/// the app's settings.
@MainActor
@Observable
final class ModelPerfConfigStore {
    static let storageKey = "rapid.perf.modelOverrides.v1"

    private let defaults: UserDefaults

    /// Alias → override. Absent key means "no opinion"; see the sparseness
    /// note above.
    private(set) var overrides: [String: ModelPerfConfig] = [:]

    /// Non-nil when the persisted blob existed but could not be decoded.
    /// Surfaced in the panel: silently showing "everything is default" over a
    /// blob we failed to read would misreport the user's own settings back to
    /// them.
    private(set) var loadError: String?

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        load()
    }

    func load() {
        loadError = nil
        guard let data = defaults.data(forKey: Self.storageKey) else {
            overrides = [:]
            return
        }
        do {
            overrides = try JSONDecoder().decode([String: ModelPerfConfig].self, from: data)
        } catch {
            overrides = [:]
            loadError = "Could not read saved performance settings: \(error.localizedDescription)"
        }
    }

    /// The user's override for `alias`, or an all-`nil` config when they have
    /// none. Callers can read fields directly; `nil` fields mean "engine
    /// default", never a concrete value.
    func config(forAlias alias: String) -> ModelPerfConfig {
        overrides[normalize(alias)] ?? ModelPerfConfig()
    }

    /// True when this alias has any explicit override — drives the "modified"
    /// affordance and enables the reset action in the panel.
    func hasOverride(forAlias alias: String) -> Bool {
        !(overrides[normalize(alias)] ?? ModelPerfConfig()).isEmpty
    }

    /// Store `config` for `alias`. An empty config removes the row instead of
    /// persisting a no-op, which keeps ``hasOverride(forAlias:)`` honest and
    /// stops the blob growing one entry per model the user merely looked at.
    func setConfig(_ config: ModelPerfConfig, forAlias alias: String) {
        let key = normalize(alias)
        guard !key.isEmpty else { return }
        if config.isEmpty {
            overrides.removeValue(forKey: key)
        } else {
            overrides[key] = config
        }
        persist()
    }

    /// The issue asks for "reset to measured default" as a single action.
    /// Clearing the override IS that reset: with no override the alias falls
    /// back to ``RAMBucketedDefault``'s benchmarked launch flags when it is
    /// this Mac's recommended pick, and to the engine's own defaults
    /// otherwise. Nothing needs to know what the measured numbers are.
    func resetToDefaults(forAlias alias: String) {
        let key = normalize(alias)
        guard overrides[key] != nil else { return }
        overrides.removeValue(forKey: key)
        persist()
    }

    /// Every alias the user has an opinion about, sorted for a stable panel.
    var configuredAliases: [String] {
        overrides.keys.sorted()
    }

    /// `serve` argv contributed by this alias's override. Empty when the user
    /// has no opinion — the untouched-install path.
    func launchFlags(forAlias alias: String) -> [String] {
        config(forAlias: alias).launchFlags(forAlias: alias)
    }

    // MARK: - Internals

    /// Aliases arrive from the model picker, from persisted settings, and from
    /// the server's own `servingAlias`. Case and surrounding whitespace differ
    /// across those sources; without normalizing, the same model can hold two
    /// rows and the user's setting appears to be ignored.
    private func normalize(_ alias: String) -> String {
        alias.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }

    private func persist() {
        do {
            let data = try JSONEncoder().encode(overrides)
            defaults.set(data, forKey: Self.storageKey)
        } catch {
            // Encoding a dictionary of optionals cannot realistically fail,
            // but swallowing it silently would leave the in-memory state
            // diverged from disk with no trace. Surface it the same way a
            // load failure surfaces.
            loadError = "Could not save performance settings: \(error.localizedDescription)"
        }
    }
}
