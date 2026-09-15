import CoreGraphics
import Foundation
import Testing
@testable import Rapid

@Suite("Personal Intelligence composer control")
struct PersonalIntelligenceControlTests {
    @Test("The custom mark is one centered four-point shape at composer size")
    func glyphGeometry() {
        let bounds = CGRect(x: 0, y: 0, width: 15, height: 15)
        let path = PersonalIntelligenceGlyph().path(in: bounds)

        #expect(!path.isEmpty)
        #expect(path.boundingRect == bounds)
        #expect(path.contains(CGPoint(x: bounds.midX, y: bounds.midY)))
        #expect(!path.contains(CGPoint(x: 1, y: 1)))
    }

    @Test("Only empirically verified aliases can default into the runtime")
    func modelGate() {
        let verified = [
            "minicpm5-2b-4bit",
            "qwen3.5-4b-4bit",
            "qwen3.8-27b-4bit",
            "llama3-3b-4bit",
            "gemma-4-12b-4bit",
            "glm4.7-4bit",
            "gpt-oss-20b-4bit",
            "devstral-v2-24b-4bit",
        ]
        let excluded = [
            "phi-4-mini-4bit",
            "hermes3-8b-4bit",
            "deepseek-r1-8b-4bit",
            "qwen3-vl-8b-4bit",
            "future-model-7b",
        ]

        for alias in verified {
            #expect(PersonalIntelligenceConfig.supportsModel(alias))
        }
        for alias in excluded {
            #expect(!PersonalIntelligenceConfig.supportsModel(alias))
        }
    }

    @Test("Losing model support stops an active Personal Intelligence run")
    func unsupportedModelTransitionIsFailClosed() throws {
        let source = try String(
            contentsOf: URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("Sources/Rapid/UI/ChatView.swift"),
            encoding: .utf8
        )
        let stripped = CapabilityChipRenderGateSourceGuardTests
            .stripCommentsAndWhitespace(source)

        #expect(stripped.contains(
            ".onChange(of:personalIntelligenceSupportsModel){_,supportedinguard!supportedelse{return}stopAgentIfNeeded()}"
        ))
    }
}
