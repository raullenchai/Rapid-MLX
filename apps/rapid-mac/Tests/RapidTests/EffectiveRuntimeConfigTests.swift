import Foundation
import Testing
@testable import Rapid

@Suite("Effective runtime configuration DTO")
struct EffectiveRuntimeConfigTests {
    @Test("Decodes values and provenance from the server")
    func decodesWireContract() throws {
        let json = #"""
        {
          "model": "qwen-test",
          "schema_version": 1,
          "fields": [
            {
              "field": "prefill_step_size",
              "value": 512,
              "source": "performance_profile",
              "source_id": "model-profile:qwen",
              "reason_code": "profile_recommendation",
              "trace": [{
                "value": 512,
                "source": "performance_profile",
                "source_id": "model-profile:qwen",
                "reason_code": "profile_recommendation",
                "action": "overridden"
              }]
            },
            {
              "field": "gpu_memory_utilization",
              "value": null,
              "source": "global_default",
              "source_id": "runtime-defaults:v1",
              "reason_code": "global_default",
              "trace": []
            }
          ]
        }
        """#

        let snapshot = try JSONDecoder().decode(
            EffectiveRuntimeConfigSnapshot.self,
            from: Data(json.utf8)
        )

        #expect(snapshot.schemaVersion == 1)
        #expect(snapshot.hasSupportedSchema)
        #expect(snapshot.model == "qwen-test")
        #expect(snapshot.belongs(to: "QWEN-TEST"))
        #expect(!snapshot.belongs(to: "another-model"))
        #expect(snapshot.fields[0].value == .int(512))
        #expect(snapshot.fields[0].provenanceText == "Model profile")
        #expect(snapshot.fields[0].trace[0].action == "overridden")
        #expect(snapshot.fields[1].value == .null)
        #expect(snapshot.fields[1].value.displayText == "Automatic")
    }

    @Test("Unknown future provenance remains readable")
    func preservesUnknownSource() throws {
        let json = #"""
        {"field":"kv_cache_dtype","value":"bf16","source":"future_source",
         "source_id":"future:v2","reason_code":"future_reason","trace":[]}
        """#
        let field = try JSONDecoder().decode(
            EffectiveRuntimeField.self,
            from: Data(json.utf8)
        )
        #expect(field.value == .string("bf16"))
        #expect(field.provenanceText == "future:v2")
    }

    @Test("Rejects an unknown envelope schema")
    func rejectsUnknownSchema() {
        let snapshot = EffectiveRuntimeConfigSnapshot(
            model: "qwen-test",
            schemaVersion: 2,
            fields: []
        )
        #expect(!snapshot.hasSupportedSchema)
    }

    @MainActor
    @Test("Refresh identity changes when the child becomes ready")
    func refreshIdentityIncludesReadiness() {
        let starting = ServerManager(
            testingState: .starting(alias: "qwen-test"),
            activeBearer: "test-token"
        )
        let ready = ServerManager(
            testingState: .ready(alias: "qwen-test"),
            activeBearer: "test-token"
        )

        #expect(starting.effectiveRuntimeConfigRefreshID.hasSuffix(":not-ready"))
        #expect(ready.effectiveRuntimeConfigRefreshID.hasSuffix(":ready"))
    }
}
