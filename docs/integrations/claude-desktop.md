# Claude Desktop gateway

Claude Desktop can connect to Rapid-MLX through its official in-app third-party
inference settings. Rapid-MLX does not edit Claude Desktop configuration or
launch the app.

1. Start a tool-capable local model and run `rapid-mlx launch claude-desktop
   --model <served-model>`. Use `--opus-model`, `--sonnet-model`, and
   `--haiku-model` to assign different served models to the three tiers.
2. In Claude Desktop, choose Help → Troubleshooting → Enable Developer Mode.
3. Choose Developer → Configure Third-Party Inference → Gateway. Paste the
   Gateway URL and API key printed by the command (also shown on Rapid's Launch
   page). Select the default Bearer authentication scheme and test connection.
4. Select Apply Changes.

The command only creates Rapid's private tier mapping and stable local API key.
If an already-running server uses an older key, restart it before testing the
new connection; the command prints this condition.
The integration is off until the command runs. `rapid-mlx launch claude-desktop
--remove` disables tier discovery and alias resolution in Rapid. The standard
`/v1/models` entries retain their existing JSON fields; extra discoverable
Claude tier entries carry `anthropic_family_tier` and `is_family_default`.
Rapid accepts both its advertised `claude-<tier>-rapid-mlx` IDs and the
documented bare `opus`, `sonnet`, and `haiku` aliases on `/v1/messages`.

Claude Desktop's documented requirements and field names:
[gateway](https://claude.com/docs/third-party/claude-desktop/gateway) and
[in-app configuration](https://claude.com/docs/third-party/claude-desktop/in-app-configuration).
