import Testing
@testable import Rapid

@Suite("Connect agents — process-scoped launch commands")
struct AgentLaunchCommandTests {
    @Test("registry commands carry authentication only for config writers")
    func registryCommands() {
        #expect(IntegrationLaunchCommand.configWriter(
            id: "cline", serverURL: "http://127.0.0.1:8000", key: "secret", model: "model", cli: "rapid-mlx"
        ) == "env RAPID_MLX_API_KEY=secret rapid-mlx launch cline --server-url http://127.0.0.1:8000 --model model")
        #expect(IntegrationLaunchCommand.adapterGuide(
            id: "aider", baseURL: "http://127.0.0.1:8000/v1", model: "model", cli: "rapid-mlx"
        ) == "rapid-mlx agents aider --base-url http://127.0.0.1:8000/v1 --model model")
    }

    /// The Desktop app owns an off-PATH sidecar (see ``ServerLocator``), so
    /// the launch/agent commands it hands the user must invoke the binary by
    /// its absolute, shell-quoted path — otherwise pasting them yields
    /// `command not found: rapid-mlx`.
    @Test("registry commands reference the off-PATH sidecar binary by absolute path when resolved")
    func registryCommandsUseResolvedBinaryPath() {
        let cli = IntegrationLaunchCommand.shellQuote(
            "/Users/alice/Library/Application Support/Rapid/runtime-override/rapid-mlx/bin/rapid-mlx"
        )
        #expect(cli == "'/Users/alice/Library/Application Support/Rapid/runtime-override/rapid-mlx/bin/rapid-mlx'")
        #expect(cli.hasPrefix("'") && cli.hasSuffix("'"))

        // Local key so the generator input and the prefix assertion cannot
        // drift apart (the suite property is a different value).
        let key = "secret"
        let writer = IntegrationLaunchCommand.configWriter(
            id: "claude-code",
            serverURL: "http://127.0.0.1:8004",
            key: key,
            model: "model",
            cli: cli
        )
        #expect(writer.contains(cli))
        #expect(writer.hasPrefix("env RAPID_MLX_API_KEY=\(key) \(cli) launch claude-code"))
        #expect(!writer.contains(" rapid-mlx "))

        let guide = IntegrationLaunchCommand.adapterGuide(
            id: "aider",
            baseURL: "http://127.0.0.1:8004/v1",
            model: "model",
            cli: cli
        )
        #expect(guide.hasPrefix("\(cli) agents aider"))
        #expect(!guide.contains(" rapid-mlx "))
    }

    @Test("shell quoting escapes spaces and embedded single quotes")
    func shellQuoteHandlesSpacesAndApostrophes() {
        let spaced = IntegrationLaunchCommand.shellQuote("/tmp/Rapid App Support/rapid-mlx")
        #expect(spaced == "'/tmp/Rapid App Support/rapid-mlx'")

        // A path containing a single quote must be escaped so the pasted
        // command still runs as a single argument.
        let apostrophe = IntegrationLaunchCommand.shellQuote("/tmp/it's/rapid-mlx")
        #expect(apostrophe == "'/tmp/it'\\''s/rapid-mlx'")
    }

    /// A nil binary (dev snapshot) intentionally falls back to the bare
    /// command so the page still renders.
    @Test("registry commands fall back to the bare command when no binary is resolved")
    func registryCommandsFallBackToBareCommand() {
        let writer = IntegrationLaunchCommand.configWriter(
            id: "cline", serverURL: "http://127.0.0.1:8000", key: "secret", model: "model", cli: "rapid-mlx"
        )
        #expect(writer.contains(" rapid-mlx launch cline "))
    }

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
