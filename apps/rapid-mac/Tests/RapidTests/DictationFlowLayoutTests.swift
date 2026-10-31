import AppKit
import SwiftUI
import Testing
@testable import Rapid

@Suite("Dictation flow layout")
@MainActor
struct DictationFlowLayoutTests {
    @Test("oversized titles wrap before placing the next item")
    func oversizedTitle() {
        let spacing: CGFloat = 4
        let result = FlowLayout.frames(maxWidth: 200, spacing: spacing, measure: [
            { proposal in
                proposal.width == 200 ? CGSize(width: 200, height: 60) : CGSize(width: 600, height: 20)
            },
            { _ in CGSize(width: 50, height: 20) },
        ])
        #expect(result.frames[0].width <= 200)
        #expect(result.frames[0].height == 60)
        #expect(result.frames[1].minY >= result.frames[0].maxY + spacing)
        #expect(result.size.width <= 200)
        #expect(result.size.height >= 60 + spacing + 20)
    }

    @Test("long model titles wrap in a hosting view")
    func hostedTitle() {
        let title = String(repeating: "modelname ", count: 20)
        let host = NSHostingView(rootView:
            FlowLayout(spacing: 4) {
                Text(title)
                Text("offline")
            }
            .frame(width: 200)
            .fixedSize(horizontal: false, vertical: true)
        )
        let singleLine = NSHostingView(rootView: Text("modelname"))
        #expect(host.fittingSize.height > 3 * singleLine.fittingSize.height)
    }
}
