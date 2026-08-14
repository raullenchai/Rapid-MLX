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
            #expect(!command.contains("export "))
            #expect(!command.contains("\n"))
            #expect(command.contains(key))
            #expect(command.contains(model))
        }
    }

    /// The rule these commands exist to honour: nothing they touch survives
    /// the process. `env` prefixes and `mktemp` scratch dirs both satisfy it;
    /// a path under the user's home does not. This is the assertion that
    /// actually distinguishes the two, so it names the home directories rather
    /// than matching on a command prefix.
    @Test("No command writes anywhere under the user's home")
    func commandsStayOutOfTheUserHome() {
        let commands = [
            AgentLaunchCommand.claude(baseURL: base, key: key, model: model),
            AgentLaunchCommand.codex(baseURL: base, key: key, model: model),
            AgentLaunchCommand.hermes(baseURL: base, key: key, model: model),
        ]

        for command in commands {
            #expect(!command.contains("~/"))
            #expect(!command.contains("$HOME"))
        }
    }

    @Test("Interactive Codex uses an isolated temporary home")
    func codexInteractiveCommandUsesSupportedIsolation() {
        let command = AgentLaunchCommand.codex(baseURL: base, key: key, model: model)

        #expect(command.contains("d=$(mktemp -d)"))
        #expect(command.contains(#"CODEX_HOME="$d""#))
        #expect(command.contains(#"trap 'rm -rf "$d"' EXIT"#))
        #expect(command.contains(" codex -m qwen-test "))
        #expect(!command.contains("codex --ignore-user-config"))
        #expect(command.contains(#"wire_api="responses""#))
    }

    /// Claude Code reads `settings.json` in preference to the environment for
    /// `ANTHROPIC_BASE_URL`, so an `env`-only command is silently ignored for
    /// any user who already routes Claude Code elsewhere. `--settings <file>`
    /// is the supported way to win that precedence fight without editing the
    /// file they own — the same mechanism amux uses.
    @Test("Claude routes through a throwaway settings file, not the user's")
    func claudeUsesTemporarySettingsFile() {
        let claude = AgentLaunchCommand.claude(baseURL: base, key: key, model: model)

        #expect(claude.contains("mktemp -d"))
        #expect(claude.contains("claude --settings "))
        // The blob is JSON under `env`, which is where Claude Code reads these.
        #expect(claude.contains("\"ANTHROPIC_BASE_URL\":\"\(base)\""))
        #expect(claude.contains("\"ANTHROPIC_API_KEY\":\"\(key)\""))
        #expect(claude.contains("\"ANTHROPIC_MODEL\":\"\(model)\""))
        // Exactly one non-empty credential, so a shell that exports the other
        // (CC Switch and friends do) cannot make Claude Code refuse to choose.
        #expect(claude.contains("\"ANTHROPIC_AUTH_TOKEN\":\"\""))
        // Cleaned up however the session ends, so the key does not outlive it.
        #expect(claude.contains("trap "))
        // The whole point: the user's own settings file is never named.
        #expect(!claude.contains(".claude/settings.json"))
    }

    /// Codex and Hermes each accept a one-session connection, and the Launch
    /// page must offer it rather than the registry's documentation printer.
    ///
    /// This is the regression the test exists for: the sidecar registry
    /// classifies both as `adapter_profile`, so the page generated `rapid-mlx
    /// agents codex ...` — a command that prints a setup guide. The button
    /// said "run" and what ran was a page of Markdown; Codex never started,
    /// even though the command that starts it was already written and sitting
    /// unreachable in the fallback list.
    @Test("Codex and Hermes launch the client, not a setup guide")
    func sessionLaunchCommandsStartTheClient() {
        let codex = AgentLaunchCommand.codex(baseURL: base, key: key, model: model)
        #expect(codex.contains(" codex "))
        #expect(!codex.contains(" agents "))

        let hermes = AgentLaunchCommand.hermes(baseURL: base, key: key, model: model)
        #expect(hermes.contains(" hermes "))
        #expect(!hermes.contains(" agents "))

        // The guide generator is what these must NOT be.
        let guide = IntegrationLaunchCommand.adapterGuide(
            id: "codex", baseURL: base, model: model, cli: "rapid-mlx"
        )
        #expect(guide.contains(" agents codex "))
        #expect(codex != guide)
    }

    @Test("Hermes retains its native provider variables")
    func providerSpecificVariablesRemainComplete() {
        let hermes = AgentLaunchCommand.hermes(baseURL: base, key: key, model: model)
        #expect(hermes.contains("OPENAI_BASE_URL=\(base)"))
        #expect(hermes.contains("HERMES_INFERENCE_MODEL=\(model)"))
        #expect(hermes.hasSuffix("hermes --provider openai-api --ignore-user-config"))
    }
}
