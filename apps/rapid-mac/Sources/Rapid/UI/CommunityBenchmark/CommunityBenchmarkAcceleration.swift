import Foundation

/// Presents the effective language settings recorded with a run.
enum CommunityBenchmarkAcceleration {
    typealias Language = CommunityBenchmarkResult.Execution.Task.Language

    static func summary(_ language: Language?) -> String? {
        guard let language else { return nil }
        var parts: [String] = []
        let spec = language.speculativeDecoding
        let draft = spec?.maxDraftTokens.map { " k=\($0)" } ?? ""
        if let kind = language.runtime?.kind {
            let name = ["native_mtp": "native MTP", "dflash": "DFlash", "companion_dspark": "DSpark"][kind] ?? kind
            let algorithm = language.runtime?.algorithm.map { " \($0)" } ?? ""
            let externalDraft = spec?.method == "mtp" && spec?.assistantSource == "external_model" ? ", external draft" : ""
            parts.append("\(name)\(algorithm)\(draft) (dedicated runtime\(externalDraft))")
        } else if let method = spec?.method, method != "none" {
            let name = method == "mtp" ? "MTP" : method
            let suffix = method == "mtp" && spec?.assistantSource == "external_model" ? " (external draft)" : ""
            parts.append("\(name)\(draft)\(suffix)")
        }
        if let kv = language.kvCache, kv.mode == "quantized" {
            var label = "KV"
            if let dtype = kv.dtype { label += " \(dtype)" }
            if let scheme = kv.scheme { label += " (\(scheme))" }
            parts.append(label)
        }
        if language.prefixCacheEnabled == true { parts.append("prefix cache") }
        if parts.isEmpty && language.pflash?.mode == nil { parts.append("no accelerations") }
        if let chunk = language.prefillChunkSize { parts.append("prefill chunk \(chunk)") }
        if let mode = language.pflash?.mode { parts.append("PFlash \(mode)") }
        return parts.joined(separator: " · ")
    }
}
