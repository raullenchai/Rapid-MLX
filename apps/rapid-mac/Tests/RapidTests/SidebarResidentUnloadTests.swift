import Testing
@testable import Rapid

@Suite("Sidebar resident unload")
struct SidebarResidentUnloadTests {
    @Test("Memory summary keeps one shared unit in the narrow sidebar")
    func compactMemorySummary() {
        let gib = UInt64(1) << 30
        #expect(
            SidebarView.memorySummary(
                usedBytes: 6 * gib,
                limitBytes: 204 * gib
            ) == "6 / 204 GB"
        )
    }

    @Test("Unload is available only while the resident pool is idle")
    func disabledState() {
        #expect(!SidebarView.residentUnloadDisabled(
            isOperating: false,
            chatIsStreaming: false,
            hasActiveRequests: false
        ))
        #expect(SidebarView.residentUnloadDisabled(
            isOperating: true,
            chatIsStreaming: false,
            hasActiveRequests: false
        ))
        #expect(SidebarView.residentUnloadDisabled(
            isOperating: false,
            chatIsStreaming: true,
            hasActiveRequests: false
        ))
        #expect(SidebarView.residentUnloadDisabled(
            isOperating: false,
            chatIsStreaming: false,
            hasActiveRequests: true
        ))
    }

    @Test("Accessible copy names the scope and memory released")
    func accessibleCopy() {
        let gib = UInt64(1) << 30
        #expect(
            SidebarView.residentUnloadLabel(
                modelCount: 1,
                memoryUsedBytes: 6 * gib
            ) == "Unload model and free 6 GB"
        )
        #expect(
            SidebarView.residentUnloadLabel(
                modelCount: 2,
                memoryUsedBytes: 10 * gib
            ) == "Unload all models and free 10 GB"
        )
    }

    @Test("Help explains blocked and in-progress states")
    func helpCopy() {
        #expect(
            SidebarView.residentUnloadHelp(
                isOperating: false,
                hasActiveResponse: false,
                enabledLabel: "Unload model and free 6 GB"
            ) == "Unload model and free 6 GB"
        )
        #expect(
            SidebarView.residentUnloadHelp(
                isOperating: false,
                hasActiveResponse: true,
                enabledLabel: "unused"
            ) == "Stop the active response before unloading models"
        )
        #expect(
            SidebarView.residentUnloadHelp(
                isOperating: true,
                hasActiveResponse: true,
                enabledLabel: "unused"
            ) == "Unloading models…"
        )
    }
}
