# Long agent tasks: context and tool calls

A coding agent sends the model a conversation containing your request, earlier
messages, tool calls, and tool results. This conversation is the model's
context. Each model has a context window, measured in tokens, that limits how
much it can handle at once.

## Why context fills up

A tool can return a whole file, a long command log, or many search results.
Those results become part of the conversation, along with the agent's replies.
Even a short task can fill a window if it reads enough material. The model also
needs room in the window to generate its next reply.

If the prompt reaches the context limit and leaves no room for generation, the
server returns a `context_length_exceeded` error. It asks you to reduce the
messages, start a new session, or use a model with a larger window. The server
can also reject a prompt that exceeds its configured prompt limit. When the
prompt fits but the requested reply is too long for the remaining room, the
server reduces the reply limit to fit. A client may also manage its own
conversation history, so check its settings and messages.

## Keep a task manageable

- Ask the agent to read only the files or sections it needs.
- Ask for short tool output: filter search results and command logs, and avoid
  printing large files when a small excerpt will do.
- Break a large job into smaller steps. Keep a short summary of decisions and
  remaining work so you can carry it into a new session.
- Start a new session for a new task. This keeps unrelated history out of the
  next prompt.
- If the context fills, start a fresh session with the summary and the next
  concrete step. If your agent offers conversation compaction, you can use it
  to shorten the history first.

## Choose a model and leave memory headroom

For long tasks, choose a tool-capable model with a larger context window. Check
the model's advertised window and any server or agent limits; the effective
limit can be lower. Leave unified memory free for the growing context and other
applications as well as the model itself. A model that barely fits may not leave
room for a long conversation.

Small models of 4B parameters or less may stop calling tools during long chats.
They can imitate earlier turns instead of choosing the next tool reliably. For
long agent work, try a larger tool-capable model that fits comfortably in your
Mac's memory. See the [tool-calling guide](tool-calling.md) for model and parser
setup and the [coding agent guide](../agents/matrix.md) for launch options.

## Reuse the beginning of a conversation

Prefix caching can speed up repeated prompts in multi-turn work by reusing
previously processed tokens. It is enabled by default on the server. Reuse
depends on an exact matching token prefix and available cache space; a cache hit
is not guaranteed. Caching saves processing time but does not increase the
model's context window.

Keep the system instructions and tool definitions stable when possible. The
[cache prefill guide](cache-prefill.md) explains how to warm a prefix and how
disk cache persistence works. The [server guide](server.md) lists cache and
context settings.

## Launch a coding agent

Install a supported agent, then run:

```bash
rapid-mlx launch <agent>
```

Run `rapid-mlx launch list` to see integrations. The [CLI reference](../reference/cli.md)
explains the launch workflow.
