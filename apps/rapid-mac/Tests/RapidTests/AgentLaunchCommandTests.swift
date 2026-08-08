import Testing
@testable import Rapid

@Suite("Connect agents — process-scoped launch commands")
struct AgentLaunchCommandTests {
    private let base = "http://127.0.0.1:8000/v1"
    private let key = "local-key"
    private let model = "qwen-test"

    @Test("Every command is one process-scoped line, never an export")
    func commandsDoNotMutateTheShellEnvironment() {
        let commands = [
            AgentLaunchCommand.claude(baseURL: base, key: key, model: model),
            AgentLaunchCommand.codex(baseURL: base, key: key, model: model),
            AgentLaunchCommand.hermes(baseURL: base, key: key, model: model),
        ]

        for command in commands {
            #expect(command.hasPrefix("env "))
            #expect(!command.contains("export "))
            #expect(!command.contains("\n"))
            #expect(command.contains(key))
            #expect(command.contains(model))
        }
    }

    @Test("Interactive Codex uses an isolated temporary home")
    func codexInteractiveCommandUsesSupportedIsolation() {
        let command = AgentLaunchCommand.codex(baseURL: base, key: key, model: model)

        #expect(command.contains(#"CODEX_HOME="$(mktemp -d)""#))
        #expect(command.contains(" codex -m qwen-test "))
        #expect(!command.contains("codex --ignore-user-config"))
        #expect(command.contains(#"wire_api="responses""#))
    }

    @Test("Claude and Hermes retain their native provider variables")
    func providerSpecificVariablesRemainComplete() {
        let claude = AgentLaunchCommand.claude(baseURL: base, key: key, model: model)
        #expect(claude.contains("ANTHROPIC_BASE_URL=\(base)"))
        #expect(claude.contains("ANTHROPIC_API_KEY=\(key)"))
        #expect(claude.hasSuffix("ANTHROPIC_MODEL=\(model) claude"))

        let hermes = AgentLaunchCommand.hermes(baseURL: base, key: key, model: model)
        #expect(hermes.contains("OPENAI_BASE_URL=\(base)"))
        #expect(hermes.contains("HERMES_INFERENCE_MODEL=\(model)"))
        #expect(hermes.hasSuffix("hermes --provider openai-api --ignore-user-config"))
    }
}
