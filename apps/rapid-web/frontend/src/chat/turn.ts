import { streamChat, type ToolCall, type ToolDefinition } from '@/api/chat';
import { callConnectorTool, updateConnectorSettings, type ConnectorState } from '@/api/connectors';
import { asApiError } from '@/api/errors';
import { useStore, wireTurns } from '@/state/store';
import type { MessageNode } from '@/state/types';
import { activePath, branchAnchor, siblings } from './MessageTree';
import {
  advertisedConnectorTools,
  displaySafe as displaySafeText,
  displaySafeResult,
  formatArguments,
  gateConnectorCall,
  isConnectorTool,
  loadConnectorState,
} from './connectors';
import { composeSystemPrompt } from './instructions';
import { streamingStore } from './StreamingStore';
import {
  MAX_TOOL_EXECUTIONS,
  TOOL_BUDGET_MESSAGE,
  TOOL_GUIDANCE,
  advertised,
  displaySafe,
  execute,
  gate,
  loadTools,
} from './tools';

/**
 * Running a turn.
 *
 * All four entry points — send, retry, edit-and-resend, and the branch walk
 * that follows a deletion — go through `runTurn`, so there is one definition
 * of what a turn does and one place the stream is wired up.
 */

let inFlight: AbortController | null = null;

export function isStreaming(): boolean {
  return inFlight !== null;
}

export function stopTurn(): void {
  inFlight?.abort();
  // A stop while an approval sheet is open must settle the promise the tool
  // loop is parked on, or the turn never unwinds.
  useStore.getState().answerApproval('unavailable');
}

interface RunOptions {
  /** The node the answer is written into. Created by the caller. */
  assistantId: string;
  alias: string | null;
}

/**
 * Stream one answer into `assistantId`.
 *
 * A logical turn may take several round trips: the model asks for tools, the
 * results are appended as `tool` nodes, and a fresh assistant node opens for
 * the answer they inform. Each round commits its own node exactly once —
 * nothing per-token touches the app store, so a 400 ms persist debounce is
 * not stringifying the whole store ten times a second.
 */
export async function runTurn({ assistantId, alias }: RunOptions): Promise<void> {
  const controller = new AbortController();
  inFlight = controller;

  // BEFORE the first await, not inside `runOneStream`. The caller has already
  // appended a `streaming` node, and that row renders whatever this store
  // holds — so anything awaited in between leaves the PREVIOUS turn's answer
  // on screen under the new prompt. Two round trips happen below, and on a
  // tunnel they are slow enough to read.
  streamingStore.start();

  const startedAt = performance.now();
  const catalogue = await loadToolsSafely();
  // The connector state is read once per turn: the panel can arm one
  // mid-session, but a mid-turn change would leave the model advertised a
  // tool that vanished before it asked for it.
  const connectors = await loadConnectorState();
  const approvedOrigins = new Set<string>();
  // Grants approved during this turn. `always` is written through to the
  // server, but a plain `allowOnce` must not re-prompt for the same tool in
  // the same answer — the user already said yes to it.
  const approvedTools = new Set<string>();
  let executions = 0;
  let currentId = assistantId;
  let sawToolResult = false;

  try {
    for (;;) {
      const store = useStore.getState();
      // Advertised only while there is budget left. The final round runs with
      // no tools at all, so the model has to answer from what it has rather
      // than asking for a call that would be refused.
      const tools =
        executions < MAX_TOOL_EXECUTIONS
          ? [
              ...advertised(catalogue.tools, store.settings.enabledTools),
              ...advertisedConnectorTools(connectors),
            ]
          : [];

      const outcome = await runOneStream({
        nodeId: currentId,
        // Re-read each round: the previous round appended nodes.
        turns: currentTurns(currentId, sawToolResult),
        alias,
        tools,
        startedAt,
        controller,
      });

      if (outcome.kind !== 'toolCalls') return;

      const spent = tools.length === 0;
      const results: Array<{ call: ToolCall; content: string; failed: boolean }> = [];
      for (const call of outcome.calls) {
        // Every requested call gets a result row, including one that was
        // never run: the wire shape is assistant(tool_calls) -> tool per id,
        // and a missing answer makes the next request malformed.
        if (spent || executions >= MAX_TOOL_EXECUTIONS) {
          results.push({ call, content: TOOL_BUDGET_MESSAGE, failed: true });
          continue;
        }
        executions += 1;
        results.push({
          call,
          ...(isConnectorTool(connectors, call.function.name)
            ? await runConnectorCall(call, {
                connectors,
                approvedTools,
                signal: controller.signal,
              })
            : await runOneCall(call, {
                catalogue,
                enabled: store.settings.enabledTools,
                approvedOrigins,
                signal: controller.signal,
              })),
        });
      }

      if (controller.signal.aborted) return;

      for (const { call, content, failed } of results) {
        useStore.getState().appendNode({
          parentId: currentLeaf(),
          role: 'tool',
          content,
          // The status is what the chip paints from. Deriving it by matching
          // the prose instead would misread any result that merely mentions
          // an error, and miss every one worded differently.
          status: failed ? 'failed' : 'complete',
          toolCallId: call.id,
        });
      }
      sawToolResult = true;

      // That round was already the tools-disabled one and it asked anyway.
      // Its rows say why; re-entering would loop forever against a model that
      // keeps asking for calls it can no longer make.
      if (spent) return;

      currentId = useStore.getState().appendNode({
        parentId: currentLeaf(),
        role: 'assistant',
        content: '',
        status: 'streaming',
      });
    }
  } finally {
    if (inFlight === controller) inFlight = null;
  }
}

/** The catalogue, or an empty one. A server without tools is not an error. */
async function loadToolsSafely(): Promise<{
  tools: ToolDefinition[];
  approvalRequired: Set<string>;
}> {
  try {
    return await loadTools();
  } catch {
    return { tools: [], approvalRequired: new Set() };
  }
}

/**
 * The turns to send, with both instruction layers merged into one system
 * message and the guidance preamble folded in once a tool result is in the
 * history. One row, never two: local chat templates often reject a second
 * system message.
 */
function currentTurns(nodeId: string, sawToolResult: boolean) {
  const store = useStore.getState();
  const conversation = activeConversation();
  return wireTurns(
    pathExcluding(nodeId),
    composeSystemPrompt({
      global: store.settings.system,
      conversation: conversation?.customInstructions ?? '',
      ...(sawToolResult ? { guidance: TOOL_GUIDANCE } : {}),
    }),
  );
}

/** Run one call through the gate, returning what the model will read. */
async function runOneCall(
  call: ToolCall,
  options: {
    catalogue: { tools: ToolDefinition[]; approvalRequired: Set<string> };
    enabled: string[];
    approvedOrigins: Set<string>;
    signal: AbortSignal;
  },
): Promise<{ content: string; failed: boolean }> {
  const names = advertised(options.catalogue.tools, options.enabled).map(
    (tool) => tool.function.name,
  );
  const decision = gate(call, {
    advertised: new Set(names),
    approvalRequired: options.catalogue.approvalRequired,
    approvedOrigins: options.approvedOrigins,
    autoApprove: useStore.getState().settings.autoApproveBrowsing,
  });

  if (decision.kind === 'refuse') return { content: decision.reason, failed: true };

  if (decision.kind === 'approve') {
    const answer = await useStore.getState().askApproval({
      kind: 'browse',
      url: displaySafe(decision.url),
      host: decision.host,
    });
    if (answer === 'declined') {
      return {
        content: `${call.function.name} error: the user did not approve fetching ${decision.host}`,
        failed: true,
      };
    }
    if (answer === 'unavailable') {
      return {
        content: `${call.function.name} error: the approval prompt for ${decision.host} could not be shown`,
        failed: true,
      };
    }
    options.approvedOrigins.add(decision.origin);
  }

  try {
    const result = await execute(call, {
      advertised: names,
      approvedOrigins: [...options.approvedOrigins],
      signal: options.signal,
    });
    // A redirect left every approved origin. The server stopped rather than
    // following it, so the user answers for the new host and the call reruns.
    if (result.needs_approval) {
      const answer = await useStore.getState().askApproval({
        kind: 'browse',
        url: displaySafe(result.needs_approval.url),
        host: result.needs_approval.host,
      });
      if (answer !== 'allowed') return { content: result.content, failed: true };
      options.approvedOrigins.add(new URL(result.needs_approval.url).origin);
      const retried = await execute(call, {
        advertised: names,
        approvedOrigins: [...options.approvedOrigins],
        signal: options.signal,
      });
      return { content: retried.content, failed: retried.is_error };
    }
    return { content: result.content, failed: result.is_error };
  } catch (cause) {
    const error = asApiError(cause);
    return { content: `${call.function.name} error: ${error.message}`, failed: true };
  }
}

/**
 * Run one connector call through the gate.
 *
 * Approval is asked once per tool per turn even without a durable grant: the
 * user already said yes to this tool in this answer, and re-prompting for
 * every call of a paginated tool is how a consent dialog gets clicked
 * through blind.
 */
async function runConnectorCall(
  call: ToolCall,
  options: {
    connectors: ConnectorState | null;
    approvedTools: Set<string>;
    signal: AbortSignal;
  },
): Promise<{ content: string; failed: boolean }> {
  const name = call.function.name;
  const decision = gateConnectorCall(call, options.connectors);

  if (decision.kind === 'refuse') return { content: decision.reason, failed: true };

  if (decision.kind === 'approve' && !options.approvedTools.has(name)) {
    const answer = await useStore.getState().askApproval({
      kind: 'tool',
      // Every one of these is connector-supplied, so a bidi or zero-width
      // scalar in a tool's metadata must not be able to spoof the prompt.
      tool: displaySafeText(decision.tool),
      server: displaySafeText(decision.server),
      short: displaySafeText(decision.short),
      args: displaySafeText(formatArguments(decision.args)),
    });
    if (answer === 'declined') {
      return {
        content: `The user declined to run '${name}'. Continue without it.`,
        failed: true,
      };
    }
    if (answer === 'unavailable') {
      return {
        content: `'${name}' was not run — the request was cancelled before it could be approved.`,
        failed: true,
      };
    }
    options.approvedTools.add(name);
    if (answer === 'always') {
      // Best effort: a grant that failed to persist costs one more prompt
      // next turn, which is the safe direction to fail in.
      void updateConnectorSettings({ tool: name, grant: true }).catch(() => {});
    }
  }

  try {
    const result = await callConnectorTool({
      name,
      arguments: call.function.arguments,
      signal: options.signal,
    });
    return { content: displaySafeResult(result.content), failed: result.is_error };
  } catch (cause) {
    const error = asApiError(cause);
    return { content: `${name} error: ${error.message}`, failed: true };
  }
}

type StreamOutcome =
  | { kind: 'done' }
  | { kind: 'failed' }
  | { kind: 'toolCalls'; calls: ToolCall[] };

/** One request/response cycle, committed into `nodeId`. */
async function runOneStream(options: {
  nodeId: string;
  turns: ReturnType<typeof wireTurns>;
  alias: string | null;
  tools: ToolDefinition[];
  startedAt: number;
  controller: AbortController;
}): Promise<StreamOutcome> {
  const { nodeId, alias, controller } = options;
  const store = useStore.getState();

  streamingStore.start();

  let firstTokenAt: number | null = null;
  let engineTokens: number | null = null;
  let calls: ToolCall[] = [];

  try {
    const deltas = streamChat({
      turns: options.turns,
      model: alias,
      temperature: store.settings.temperature,
      topP: store.settings.topP,
      maxTokens: store.settings.maxTokens,
      ...(options.tools.length ? { tools: options.tools } : {}),
      signal: controller.signal,
    });

    for await (const delta of deltas) {
      switch (delta.kind) {
        case 'content':
          firstTokenAt ??= performance.now();
          streamingStore.appendContent(delta.text);
          break;
        case 'reasoning':
          // TTFT is not stamped here: reasoning arrives before the answer
          // does, and counting it would report a time-to-first-token that
          // does not correspond to any token the user can read.
          streamingStore.appendReasoning(delta.text);
          break;
        case 'usage':
          engineTokens = delta.completionTokens;
          break;
        case 'toolCalls':
          calls = delta.calls;
          break;
      }
    }

    commit({
      assistantId: nodeId,
      startedAt: options.startedAt,
      firstTokenAt,
      engineTokens,
      alias,
      status: 'complete',
      ...(calls.length ? { toolCalls: calls } : {}),
    });
    return calls.length ? { kind: 'toolCalls', calls } : { kind: 'done' };
  } catch (cause) {
    if (controller.signal.aborted) {
      // A stopped turn KEEPS what arrived. The user pressed stop because they
      // had read enough, not because they wanted it thrown away.
      commit({
        assistantId: nodeId,
        startedAt: options.startedAt,
        firstTokenAt,
        engineTokens,
        alias,
        status: 'complete',
      });
      return { kind: 'done' };
    }

    const error = asApiError(cause);
    streamingStore.flush();
    const { content, reasoning } = streamingStore.current();

    useStore.getState().patchNode(nodeId, {
      content,
      ...(reasoning ? { reasoning } : {}),
      status: 'failed',
      error: { type: error.type, message: error.message },
      ...(alias ? { model: alias } : {}),
    });
    return { kind: 'failed' };
  }
}

interface CommitOptions {
  assistantId: string;
  alias: string | null;
  startedAt: number;
  firstTokenAt: number | null;
  engineTokens: number | null;
  status: 'complete';
  toolCalls?: ToolCall[];
}

function commit({
  assistantId,
  startedAt,
  firstTokenAt,
  engineTokens,
  alias,
  status,
  toolCalls,
}: CommitOptions): void {
  // Flush first: the last few tokens must not be sitting on a timer when the
  // final content is read.
  streamingStore.flush();
  const { content, reasoning } = streamingStore.current();

  const elapsedMs = performance.now() - startedAt;
  // The engine's own count when `stream_options.include_usage` produced one.
  // The character estimate is a fallback and is MARKED as one, because it is
  // off by a wide and model-dependent margin.
  const estimated = engineTokens === null;
  const tokens = engineTokens ?? Math.round(content.length / 4);

  useStore.getState().patchNode(assistantId, {
    content,
    ...(reasoning ? { reasoning } : {}),
    ...(toolCalls?.length ? { toolCalls } : {}),
    status,
    ...(alias ? { model: alias } : {}),
    stats: {
      ttftMs: firstTokenAt === null ? null : firstTokenAt - startedAt,
      tokens,
      tps: elapsedMs > 0 && tokens > 0 ? tokens / (elapsedMs / 1000) : null,
      tokensEstimated: estimated,
    },
  });
}

// ------------------------------------------------------------ entry points

/** Send a new prompt at the end of the visible transcript. */
export function send(text: string): void {
  const store = useStore.getState();
  if (store.activeId === null) store.createConversation();

  const alias = store.selectedByKind.text;
  useStore.getState().appendNode({
    parentId: currentLeaf(),
    role: 'user',
    content: text,
    status: 'complete',
  });

  const assistantId = useStore.getState().appendNode({
    parentId: currentLeaf(),
    role: 'assistant',
    content: '',
    status: 'streaming',
  });

  void runTurn({ assistantId, alias });
}

/**
 * Re-answer an existing prompt.
 *
 * Message-addressed, not "regenerate the last turn": the path is rewound to
 * just AFTER the owning user prompt, so the prompt is reused and the new
 * answer lands as a true sibling of the old one. Rewinding past the prompt
 * and re-sending its text would append a duplicate prompt and put the two
 * answers in different branches entirely.
 */
export function retry(nodeId: string): void {
  const store = useStore.getState();
  const conversation = activeConversation();
  if (!conversation) return;

  const node = conversation.nodes.find((candidate) => candidate.id === nodeId);
  if (!node) return;

  const anchor = branchAnchor(nodeId, conversation.nodes);
  const anchorNode = conversation.nodes.find((candidate) => candidate.id === anchor);
  const parentId = anchorNode?.parentId ?? node.parentId;

  const assistantId = store.appendNode({
    parentId,
    role: 'assistant',
    content: '',
    status: 'streaming',
  });

  void runTurn({
    assistantId,
    alias: store.selectedByKind.text,
  });
}

/**
 * Edit a user message and re-send it.
 *
 * The new prompt is a SIBLING of the original, so the pre-edit prompt and
 * everything under it survive one `‹1/2›` away. Stays on the same
 * conversation: forking to a new one here means every edit grows a
 * near-identical row in the sidebar until the list is unusable.
 */
export function editAndResend(nodeId: string, text: string): void {
  const store = useStore.getState();
  const conversation = activeConversation();
  if (!conversation) return;

  const node = conversation.nodes.find((candidate) => candidate.id === nodeId);
  if (!node) return;

  store.appendNode({
    parentId: node.parentId,
    role: 'user',
    content: text,
    status: 'complete',
  });

  const assistantId = useStore.getState().appendNode({
    parentId: currentLeaf(),
    role: 'assistant',
    content: '',
    status: 'streaming',
  });

  void runTurn({
    assistantId,
    alias: store.selectedByKind.text,
  });
}

/** Step to the previous or next alternative at this fork. */
export function switchBranch(nodeId: string, direction: -1 | 1): void {
  const conversation = activeConversation();
  if (!conversation) return;

  const anchor = branchAnchor(nodeId, conversation.nodes);
  if (anchor === null) return;

  const group = siblings(anchor, conversation.nodes);
  const index = group.findIndex((candidate) => candidate.id === anchor);
  const next = group[index + direction];
  // Bounded: the ends are a no-op, which is what the disabled arrows say.
  if (!next) return;

  useStore.getState().setActiveLeaf(next.id);
}

/** Where a node sits in its sibling group, or null if it has no alternatives. */
export function branchPosition(
  nodeId: string,
  nodes: MessageNode[],
): { index: number; total: number } | null {
  const anchor = branchAnchor(nodeId, nodes);
  if (anchor === null) return null;
  const group = siblings(anchor, nodes);
  if (group.length <= 1) return null;
  return {
    index: group.findIndex((candidate) => candidate.id === anchor),
    total: group.length,
  };
}

// ----------------------------------------------------------------- helpers

function activeConversation() {
  const state = useStore.getState();
  return state.conversations.find((conversation) => conversation.id === state.activeId) ?? null;
}

function currentLeaf(): string | null {
  return activeConversation()?.activeLeafId ?? null;
}

/** The visible path with the in-flight placeholder removed. */
function pathExcluding(assistantId: string): MessageNode[] {
  const conversation = activeConversation();
  if (!conversation) return [];
  return activePath(
    conversation.nodes,
    conversation.activeLeafId,
    conversation.branchChoices,
  ).filter((node) => node.id !== assistantId);
}
