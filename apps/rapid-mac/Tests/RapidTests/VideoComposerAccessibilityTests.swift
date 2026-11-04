import Foundation
import Testing

@Suite("Video composer accessibility")
struct VideoComposerAccessibilityTests {
    @Test("Composer controls have distinct stable IDs without an inherited container ID")
    func controlIdentifiers() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let source = try String(contentsOf: root.appendingPathComponent("Sources/Rapid/UI/VideoView.swift"), encoding: .utf8)
        // A container identifier propagates to SwiftUI descendants and can
        // override even explicit child identifiers in the accessibility tree.
        #expect(!source.contains(".accessibilityIdentifier(\"Video.Composer\")"))
        #expect(!source.contains(".accessibilityIdentifier(\"Video.LastFrame\")"))
        #expect(!source.contains(".accessibilityIdentifier(\"Video.References\")"))
        let controls = ["ModePicker", "Prompt", "Size", "Duration", "Generate", "ModelMenu",
                        "Reference.Add", "Reference.Remove", "LastFrame.Add", "LastFrame.Remove",
                        "References.Add", "RetryServerData", "ManageModels", "CancelDownload", "DownloadModel", "StartModel"]
        for control in controls {
            let modifier = ".accessibilityIdentifier(\"Video.Composer.\(control)\")"
            #expect(source.components(separatedBy: modifier).count == 2, "Missing or duplicate ID: \(control)")
        }
        #expect(source.contains(#".accessibilityIdentifier("Video.Composer.References.Remove.\(index)")"#))
    }
}
