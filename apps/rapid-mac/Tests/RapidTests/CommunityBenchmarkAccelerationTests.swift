import Foundation
import Testing
@testable import Rapid

@Suite("Community Benchmark acceleration")
struct CommunityBenchmarkAccelerationTests {
    @Test("Effective language settings decode and summarize", arguments: [
        ("{\"speculative_decoding\":{\"method\":\"none\"},\"prefill_chunk_size\":2048}", "Standard (no speed-ups)"),
        ("{\"speculative_decoding\":{\"method\":\"mtp\",\"max_draft_tokens\":3,\"assistant_source\":\"embedded_head\"}}", "MTP k=3"),
        ("{\"speculative_decoding\":{\"method\":\"mtp\",\"max_draft_tokens\":3,\"assistant_source\":\"external_model\"}}", "MTP k=3 (external draft)"),
        ("{\"runtime\":{\"kind\":\"native_mtp\"},\"speculative_decoding\":{\"method\":\"mtp\",\"max_draft_tokens\":3,\"assistant_source\":\"external_model\"}}", "native MTP k=3 (dedicated runtime, external draft)"),
        ("{\"runtime\":{\"kind\":\"native_mtp\"},\"speculative_decoding\":{\"method\":\"mtp\",\"max_draft_tokens\":3,\"assistant_source\":\"embedded_head\"}}", "native MTP k=3 (dedicated runtime)"),
        ("{\"runtime\":{\"kind\":\"dflash\",\"algorithm\":\"dflash2\"},\"speculative_decoding\":{\"max_draft_tokens\":15}}", "DFlash dflash2 k=15 (dedicated runtime)"),
        ("{\"kv_cache\":{\"mode\":\"quantized\",\"dtype\":\"q8\",\"scheme\":\"affine\"},\"prefix_cache_enabled\":true,\"prefill_chunk_size\":2048}", "KV q8 (affine) · prefix cache"),
        ("{\"pflash\":{\"mode\":\"on\"}}", "PFlash on"),
        ("{\"pflash\":{}}", "Standard (no speed-ups)"),
        ("{\"speculative_decoding\":{\"method\":\"future\"}}", "future"),
        ("{\"runtime\":{\"kind\":\"future\"}}", "future (dedicated runtime)"),
    ])
    func summaries(json: String, expected: String) throws {
        let language = try JSONDecoder().decode(CommunityBenchmarkAcceleration.Language.self, from: Data(json.utf8))
        #expect(CommunityBenchmarkAcceleration.summary(language) == expected)
    }

    @Test("Missing language hides the row")
    func missing() {
        #expect(CommunityBenchmarkAcceleration.summary(nil) == nil)
    }
}
