# MCP & Tool Calling

rapid-mlx supports the Model Context Protocol (MCP) for integrating external tools with LLMs.

## How Tool Calling Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Tool Calling Flow                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. User Request                                                    │
│     ─────────────────►  "List files in /tmp"                       │
│                                                                     │
│  2. LLM Generates Tool Call                                         │
│     ─────────────────►  tool_calls: [{                             │
│                           name: "list_directory",                   │
│                           arguments: {path: "/tmp"}                 │
│                         }]                                          │
│                                                                     │
│  3. App Executes Tool via MCP                                       │
│     ─────────────────►  MCP Server executes list_directory         │
│                         Returns: ["file1.txt", "file2.txt"]        │
│                                                                     │
│  4. Tool Result Sent Back to LLM                                    │
│     ─────────────────►  role: "tool", content: [...]               │
│                                                                     │
│  5. LLM Generates Final Response                                    │
│     ─────────────────►  "The /tmp directory contains 2 files..."   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Create MCP Config

Create `mcp.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    }
  }
}
```

### 2. Start Server with MCP

```bash
rapid-mlx serve qwen3.5-4b-4bit --mcp-config mcp.json
```

### 3. Verify MCP Status

```bash
# Check MCP status
curl http://localhost:8000/v1/mcp/status

# List available tools
curl http://localhost:8000/v1/mcp/tools
```

## Tool Calling Example

```python
import json
import httpx

BASE_URL = "http://localhost:8000"

# 1. Get available tools. Each item is {name, description, server, parameters};
#    /v1/chat/completions expects the OpenAI function-tool shape, so map them.
#    Tool names come back namespaced ("filesystem__list_directory").
mcp_tools = httpx.get(f"{BASE_URL}/v1/mcp/tools").json()["tools"]
tools = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["parameters"],
        },
    }
    for t in mcp_tools
]

# 2. Send request with tools
response = httpx.post(
    f"{BASE_URL}/v1/chat/completions",
    json={
        "model": "default",
        "messages": [{"role": "user", "content": "List files in /tmp"}],
        "tools": tools,
        "max_tokens": 1024
    }
)

result = response.json()
message = result["choices"][0]["message"]

# 3. Check for tool calls
if message.get("tool_calls"):
    tool_call = message["tool_calls"][0]

    # 4. Execute tool via MCP. The namespaced tool_name is enough —
    #    the server resolves which MCP server owns the tool.
    exec_response = httpx.post(
        f"{BASE_URL}/v1/mcp/execute",
        json={
            "tool_name": tool_call["function"]["name"],
            "arguments": json.loads(tool_call["function"]["arguments"])
        }
    )
    tool_result = exec_response.json()
    # Response shape: {tool_name, content, is_error, error_message}

    # 5. Send result back to LLM — the content on success, the error
    #    message on failure, so the model reacts to what actually happened.
    if tool_result["is_error"]:
        tool_content = f"Tool error: {tool_result['error_message']}"
    else:
        content = tool_result["content"]
        tool_content = content if isinstance(content, str) else json.dumps(content)

    messages = [
        {"role": "user", "content": "List files in /tmp"},
        message,
        {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": tool_content
        }
    ]

    final_response = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        json={"model": "default", "messages": messages, "tools": tools}
    )
    print(final_response.json()["choices"][0]["message"]["content"])
```

## MCP Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/mcp/status` | GET | Check MCP status (alias of `/v1/mcp/servers`) |
| `/v1/mcp/servers` | GET | Per-server connection state, tool counts, and errors |
| `/v1/mcp/tools` | GET | List available tools |
| `/v1/mcp/execute` | POST | Execute a tool |
| `/v1/mcp/reload` | POST | Re-read the config file and rebuild every connection |

`/v1/mcp/servers` (and its `/v1/mcp/status` alias) also carry two top-level
fields alongside `servers`:

* `error` — why MCP is not running, when it is not. An empty `servers` list
  with `error: null` means "configured and healthy, zero servers"; with an
  `error` it means MCP could not start at all. Bringing MCP up is **not**
  fatal to the server: a missing config file or an unstartable server leaves
  the rest of the API working and reports the reason here.
* `configured` — whether a config path is known at all, so a client can tell
  "no `--mcp-config` was passed" from "config present but broken".

A server entry that fails security validation is dropped rather than failing
the whole config load, and is listed with `state: "error"` and the validator's
reason — so a typo in one entry does not silently remove the others.

`POST /v1/mcp/reload` re-reads the same config path the server was started
with and reconnects everything, so a config edit applies without restarting
the model. It sits on the same auth gate as the other control-plane routes
(`Authorization: Bearer` or `x-api-key`) because it spawns whatever local
programs the config names. A reload that fails still returns 200 with the
reason in `error` — a connector that won't start is a normal, fixable state,
and the per-server rows are still worth rendering.

> **Desktop app.** Rapid-MLX Desktop drives all of the above from
> **Settings → Connectors** — add/edit servers, see connection state, switch
> individual tools off, and approve each tool the first time the model calls
> it. You do not need to write `mcp.json` by hand there.

## Example MCP Servers

### Filesystem

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    }
  }
}
```

### GitHub

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "your-token"
      }
    }
  }
}
```

### PostgreSQL

```json
{
  "mcpServers": {
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "DATABASE_URL": "postgresql://user:pass@localhost/db"
      }
    }
  }
}
```

### Brave Search

```json
{
  "mcpServers": {
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "your-key"
      }
    }
  }
}
```

## Multiple MCP Servers

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "your-token"
      }
    }
  }
}
```

## Interactive MCP Chat

The built-in chat command can act as the MCP host. The config may contain one
or more entries under `mcpServers` (the legacy `servers` key is also accepted):

```bash
rapid-mlx chat qwen3.5-4b-4bit --mcp-config mcp.json
```

This also works when chat connects to an existing model server:

```bash
rapid-mlx chat --base-url http://localhost:8000 --mcp-config mcp.json
```

The MCP servers run in the chat process. The model server only receives
standard function-tool schemas, tool calls, and tool-result messages.

## Supported Tool Formats

rapid-mlx ships over 25 tool call parser modules covering all major model families. See [Tool Calling](tool-calling.md) for the full list of parsers, aliases, and examples.

## Security

rapid-mlx includes security measures to prevent command injection attacks via MCP servers.

### Command Whitelist

Only trusted commands are allowed by default:

| Category | Allowed Commands |
|----------|-----------------|
| Node.js | `npx`, `npm`, `node` |
| Python | `uvx`, `uv`, `python`, `python3`, `pip`, `pipx` |
| Docker | `docker` |
| MCP Servers | `mcp-server-*` (official servers) |

### Blocked Patterns

The following patterns are blocked to prevent injection attacks:

- Command chaining: `;`, `&&`, `||`, `|`
- Command substitution: `` ` ``, `$()`
- Path traversal: `../`
- Dangerous env vars: `LD_PRELOAD`, `PATH`, `PYTHONPATH`

### Example: Blocked Attack

```json
{
  "mcpServers": {
    "malicious": {
      "command": "bash",
      "args": ["-c", "rm -rf /"]
    }
  }
}
```

This config will be rejected:
```
ValueError: MCP server 'malicious': Command 'bash' is not in the allowed commands whitelist.
```

### Development Mode (Unsafe)

For development only, you can bypass security validation:

```json
{
  "mcpServers": {
    "custom": {
      "command": "my-custom-server",
      "skip_security_validation": true
    }
  }
}
```

**WARNING**: Never use `skip_security_validation` in production!

### Custom Whitelist

To add custom commands to the whitelist programmatically:

```python
from vllm_mlx.mcp import MCPCommandValidator, set_validator

# Add custom commands
validator = MCPCommandValidator(
    custom_whitelist={"my-trusted-server", "another-server"}
)
set_validator(validator)
```

## Tool Execution Sandboxing

Beyond command validation, rapid-mlx provides runtime sandboxing for tool executions:

### Sandbox Features

| Feature | Description |
|---------|-------------|
| Tool Allowlisting | Only permit specific tools to execute |
| Tool Blocklisting | Block specific dangerous tools |
| Argument Validation | Block dangerous patterns in tool arguments |
| Rate Limiting | Limit tool calls per minute |
| Audit Logging | Track all tool executions |

### Blocked Argument Patterns

Tool arguments are validated for dangerous patterns:

- Path traversal: `../`
- System directories: `/etc/`, `/proc/`, `/sys/`
- Root access: `/root/`, `~root`

### High-Risk Tool Detection

Tools matching these patterns trigger security warnings:

- `execute`, `run_command`, `shell`, `eval`, `exec`, `system`, `subprocess`

### Custom Sandbox Configuration

```python
from vllm_mlx.mcp import ToolSandbox, set_sandbox

# Create sandbox with custom settings
sandbox = ToolSandbox(
    # Only allow specific tools (whitelist mode)
    allowed_tools={"read_file", "list_directory"},

    # Block specific tools (blacklist mode)
    blocked_tools={"execute_command", "run_shell"},

    # Rate limit: max 30 calls per minute
    max_calls_per_minute=30,

    # Optional audit callback
    audit_callback=lambda audit: print(f"Tool: {audit.tool_name}, Success: {audit.success}"),
)
set_sandbox(sandbox)
```

### Accessing Audit Logs

```python
from vllm_mlx.mcp import get_sandbox

sandbox = get_sandbox()

# Get recent audit entries
entries = sandbox.get_audit_log(limit=50)

# Filter by tool name
file_ops = sandbox.get_audit_log(tool_filter="file")

# Get only errors
errors = sandbox.get_audit_log(errors_only=True)

# Clear audit log
sandbox.clear_audit_log()
```

### Sensitive Data Redaction

Audit logs automatically redact sensitive fields (password, token, secret, key, credential, auth) and truncate large values.

## Troubleshooting

### MCP server not connecting

Check that the MCP server command is correct:
```bash
npx -y @modelcontextprotocol/server-filesystem /tmp
```

### Tool not executing

Verify tool is available:
```bash
curl http://localhost:8000/v1/mcp/tools | jq '.tools[].name'
```

### Tool call not parsed

Ensure you're using a model that supports function calling (Qwen3, Llama-3.2-Instruct).

### Command not in whitelist

If you see "Command X is not in the allowed commands whitelist", either:
1. Use an allowed command (see whitelist above)
2. Add the command to a custom whitelist
3. Use `skip_security_validation: true` (development only)
