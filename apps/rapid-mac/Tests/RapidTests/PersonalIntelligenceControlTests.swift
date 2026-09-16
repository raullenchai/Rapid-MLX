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

    @Test("Only models bound to a tuned harness can enter Personal Intelligence")
    func modelGate() {
        let miniProfile = ServerModelProfile(
            id: "minicpm5-2b-4bit",
            toolCallParser: "minicpm",
            personalIntelligenceProfile: "minicpm5-2b",
            personalIntelligenceQualification: "minicpm5-2b-q4-v1"
        )
        #expect(
            PersonalIntelligenceConfig.harnessProfile(
                for: "minicpm5-2b-4bit",
                serverProfile: miniProfile
            )
                == "minicpm5-2b"
        )
        let excluded = [
            // Tool calling support does not imply that the complete Personal
            // Intelligence harness has been tuned for this exact model.
            "qwen3.5-4b-4bit",
            "qwen3.8-27b-4bit",
            "llama3-3b-4bit",
            "gemma-4-e2b-4bit",
            "gemma-4-12b-4bit",
            "glm4.7-9b-4bit",
            "gpt-oss-20b-4bit",
            "devstral-v2-24b-4bit",
            "phi-4-mini-4bit",
            "hermes3-8b-4bit",
            "deepseek-r1-8b-4bit",
            "qwen3-vl-8b-4bit",
            "future-model-7b",
            // A recognized tool-capable family is not enough: a future or
            // custom alias needs explicit end-to-end product qualification.
            "qwen3.5-future-999b",
        ]

        for alias in excluded {
            #expect(!PersonalIntelligenceConfig.supportsModel(
                alias,
                serverProfile: ServerModelProfile(
                    id: alias,
                    toolCallParser: "hermes"
                )
            ))
        }
        #expect(!PersonalIntelligenceConfig.supportsModel(
            "minicpm5-2b-4bit",
            serverProfile: nil
        ))
        #expect(!PersonalIntelligenceConfig.supportsModel(
            "qwen3.5-4b-4bit",
            serverProfile: miniProfile
        ))
    }

    @Test("Unsupported models stay themselves and never offer an implicit switch")
    func unsupportedModelDoesNotSwitchModels() throws {
        let sourceRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/Rapid/UI")
        let chat = try String(
            contentsOf: sourceRoot.appendingPathComponent("ChatView.swift"),
            encoding: .utf8
        )
        let control = try String(
            contentsOf: sourceRoot.appendingPathComponent("PersonalIntelligenceControl.swift"),
            encoding: .utf8
        )
        let stripped = CapabilityChipRenderGateSourceGuardTests
            .stripCommentsAndWhitespace(chat)

        #expect(control.contains("Personal Intelligence hasn’t been tuned for"))
        #expect(!control.contains("Use MiniCPM5-2B"))
        #expect(!control.contains("UseRecommendedModel"))
        #expect(!stripped.contains("onPersonalIntelligenceModelSelection"))
    }

    @Test("First-use acceptance cannot strand staged attachments")
    func introductionUsesTheAttachmentGuard() throws {
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
            "privatefuncacceptPersonalIntelligenceIntroduction(){guard!attachmentDraft.hasAttachmentselse{"
        ))
    }

    @Test("Manual toggles update the default and conversation switches reconcile it")
    func manualPreferenceAndConversationInheritanceAreWired() throws {
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
            "setPersonalIntelligence(!agentModeEnabled,updatesPreference:true)"
        ))
        #expect(stripped.contains(
            "ifupdatesPreference{personalIntelligencePreferred=enabled}"
        ))
        #expect(stripped.contains(
            "returnPersonalIntelligenceRunBinding(conversationID:viewModel.activeConversationID,"
        ))
        #expect(stripped.contains(
            ".onChange(of:viewModel.conversations.map(\\.id)){_,_inreconcilePersonalIntelligenceStates()}"
        ))
        #expect(stripped.contains(
            ".onChange(of:personalIntelligenceBinding){oldBinding,newBindinginifnewBinding.invalidatesRun(boundTo:oldBinding){stopAgentIfNeeded()}"
        ))
        #expect(stripped.contains(
            "viewModel.locallyCreatedConversationID==activeID?[activeID]:[]"
        ))
    }

    @Test("Hover explains without arming consent shortcuts")
    func hoverIsInformationalOnly() throws {
        let sourceRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/Rapid/UI")
        let chat = try String(
            contentsOf: sourceRoot.appendingPathComponent("ChatView.swift"),
            encoding: .utf8
        )
        let control = try String(
            contentsOf: sourceRoot.appendingPathComponent("PersonalIntelligenceControl.swift"),
            encoding: .utf8
        )
        let stripped = CapabilityChipRenderGateSourceGuardTests
            .stripCommentsAndWhitespace(chat)

        #expect(stripped.contains(
            ".onHover{hoveringinifhovering{personalIntelligencePopoverShowsActions=falsepersonalIntelligencePopoverOpenedByHover=trueshowsPersonalIntelligenceInfo=true}elseifpersonalIntelligencePopoverOpenedByHover,!personalIntelligencePopoverShowsActions{showsPersonalIntelligenceInfo=false}}"
        ))
        #expect(!control.contains(".keyboardShortcut"))
    }

}
