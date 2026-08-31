import Testing
@testable import Rapid

@Suite("Command palette")
struct CommandPaletteTests {
    @Test("Filtering accepts all terms across titles and keywords")
    func filteringRequiresEveryTerm() {
        let titleResults = CommandPalette.filter(
            CommandPalette.Command.allCases,
            matching: "chat new"
        )
        let imageTitleResults = CommandPalette.filter(
            CommandPalette.Command.allCases,
            matching: "open images"
        )
        let updateTitleResults = CommandPalette.filter(
            CommandPalette.Command.allCases,
            matching: "check for updates"
        )
        let punctuationResults = CommandPalette.filter(
            CommandPalette.Command.allCases,
            matching: "export diagnostics"
        )

        #expect(titleResults == [.newChat])
        #expect(imageTitleResults == [.images])
        #expect(updateTitleResults == [.checkUpdates])
        #expect(punctuationResults == [.exportDiagnostics])
    }

    @Test("Filtering is case-insensitive and ignores diacritics")
    func filteringNormalizesQuery() {
        let results = CommandPalette.filter(
            CommandPalette.Command.allCases,
            matching: "CHÂT NEW"
        )

        #expect(results == [.newChat])
    }

    @Test("A request is consumed once and survives window recreation only until shown")
    @MainActor
    func requestConsumptionIsAppScoped() {
        let coordinator = CommandPaletteRequestCoordinator()

        coordinator.open()
        #expect(coordinator.consume(coordinator.requestID))
        #expect(!coordinator.consume(coordinator.requestID))

        coordinator.open()
        let requestID = coordinator.requestID
        #expect(coordinator.consume(requestID))
        #expect(!coordinator.consume(requestID))
    }

    @Test("The fixed registry keeps IDs unique and includes primary actions")
    func registryIsStableAndActionable() {
        let commands = CommandPalette.Command.allCases

        #expect(Set(commands.map(\.id)).count == commands.count)
        #expect(commands.contains(.newChat))
        #expect(commands.contains(.searchChats))
        #expect(commands.contains(.settings))
        #expect(commands.contains(.modelManagement))
        #expect(commands.contains(.serverLogs))
        #expect(commands.contains(.exportDiagnostics))
    }
}
