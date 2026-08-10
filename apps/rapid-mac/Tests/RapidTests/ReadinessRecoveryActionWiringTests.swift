import Foundation
import Testing
@testable import Rapid

@Suite("Readiness recovery action wiring")
struct ReadinessRecoveryActionWiringTests {
    private static var contentViewSource: String {
        get throws {
            let packageRoot = URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
            let url = packageRoot
                .appendingPathComponent("Sources/Rapid/UI/ContentView.swift")
            let body = try String(contentsOf: url, encoding: .utf8)
            return CapabilityChipRenderGateSourceGuardTests
                .stripCommentsAndWhitespace(body)
        }
    }

    @Test("Restart tears down the engine before serving")
    func restartStopsBeforeEnsuringServing() throws {
        let source = try Self.contentViewSource
        #expect(
            source.contains("case.restart(lettarget):chat.clearStaleErrorBanner()restartModel(target)"),
            "The Restart CTA must use the explicit restart path rather than Retry."
        )

        guard let signature = source.range(
            of: "privatefuncrestartModel(_target:String){"
        ) else {
            Issue.record("ContentView.restartModel could not be found")
            return
        }
        var depth = 1
        var index = signature.upperBound
        var functionEnd: String.Index?
        while index < source.endIndex {
            switch source[index] {
            case "{": depth += 1
            case "}":
                depth -= 1
                if depth == 0 { functionEnd = index }
            default: break
            }
            if functionEnd != nil { break }
            index = source.index(after: index)
        }
        guard let functionEnd else {
            Issue.record("ContentView.restartModel has no closing brace")
            return
        }
        let function = String(source[signature.lowerBound...functionEnd])
        #expect(
            function.contains("awaitserver.stop()_=awaitserver.ensureServing(alias:target,hfPath:hfPath)"),
            "Restart must stop the failed engine before bringing the selected model back."
        )
    }
}
