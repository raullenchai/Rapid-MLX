import { expect, test, type Page } from '@playwright/test';
import { startStub, toolCallFrames } from './stub-server';

/**
 * Connectors (MCP).
 *
 * The panel authors a file the ENGINE reads and spawns programs from, and the
 * chat loop then offers those programs' tools to the model. Both halves are
 * walked here, because the interesting failures are on the seam: a tool
 * switched off in Settings must not reach a request body, and a call must not
 * run before the user has approved it.
 */

const READY = { engineState: 'ready' as const, model: 'qwen3-4b' };

/** One connector, connected, exposing two tools. */
const CONNECTED = {
  enabled: true,
  servers: [
    {
      name: 'filesystem',
      transport: 'stdio' as const,
      command: 'npx',
      args: ['-y', '@modelcontextprotocol/server-filesystem', '/Users/x'],
      env: {},
      url: null,
      enabled: true,
      timeout: 30,
      summary: 'npx -y @modelcontextprotocol/server-filesystem /Users/x',
    },
  ],
  load_error: null,
  config_path: '/tmp/mcp.json',
  engine_servers: [
    {
      name: 'filesystem',
      state: 'connected',
      transport: 'stdio',
      tools_count: 2,
      error: null,
    },
  ],
  engine_reachable: true,
  subsystem_error: null,
  configured: true,
  needs_restart: false,
  engine_running: true,
  tools: [
    {
      name: 'filesystem__read_file',
      description: 'Read a file from disk',
      server: 'filesystem',
      parameters: { type: 'object', properties: { path: { type: 'string' } } },
    },
    {
      name: 'filesystem__write_file',
      description: 'Write a file to disk',
      server: 'filesystem',
      parameters: { type: 'object', properties: { path: { type: 'string' } } },
    },
  ],
  disabled_tools: [],
  granted_tools: [],
  auto_approve_all: false,
};

async function openConnectors(page: Page) {
  await page.getByLabel('Open sidebar').click();
  await page.getByRole('button', { name: 'Settings' }).click();
  await page.getByRole('button', { name: 'Connectors', exact: true }).click();
}

async function send(page: Page, text: string) {
  // Clicked, not Return: the default project is a phone, where a bare Return
  // is a newline.
  await page.getByLabel('Message').fill(text);
  await page.getByRole('button', { name: 'Send' }).click();
}

test('connectors are off until switched on, and the panel says what that means', async ({
  page,
}) => {
  const stub = await startStub(READY);
  try {
    await page.goto(stub.baseURL);
    await openConnectors(page);

    // Off is the default because a connector is a program that runs on the
    // user's machine — so nothing below the switch is shown yet.
    const master = page.getByRole('switch', { name: 'Enable connectors' });
    await expect(master).toHaveAttribute('aria-checked', 'false');
    await expect(page.getByRole('button', { name: 'Add…' })).toBeHidden();

    await master.click();
    await expect(page.getByRole('button', { name: 'Add…' })).toBeVisible();
    expect(stub.scenario.connectors.enabled).toBe(true);
  } finally {
    await stub.close();
  }
});

test('a connector can be added, and the file gets what was typed', async ({ page }) => {
  const stub = await startStub({ ...READY, connectors: { ...CONNECTED, servers: [] } });
  try {
    await page.goto(stub.baseURL);
    await openConnectors(page);
    await page.getByRole('button', { name: 'Add…' }).click();

    await page.getByLabel('Name').fill('filesystem');
    await page.getByLabel('Command').fill('npx');
    // One per line: a space-separated field would split a path containing a
    // space into two arguments.
    await page.getByLabel('Arguments').fill('-y\n@modelcontextprotocol/server-filesystem\n/tmp');
    await page.getByRole('button', { name: 'Save' }).click();

    const saved = stub.scenario.connectors.servers[0];
    expect(saved?.name).toBe('filesystem');
    expect(saved?.args).toEqual(['-y', '@modelcontextprotocol/server-filesystem', '/tmp']);
    // Scoped to the row: the server's name appears again on every tool badge
    // below, so a document-wide lookup resolves several.
    await expect(page.getByText('npx -y @modelcontextprotocol/server-filesystem')).toBeVisible();
  } finally {
    await stub.close();
  }
});

test('a name that could not become a tool name cannot be saved', async ({ page }) => {
  const stub = await startStub({ ...READY, connectors: { ...CONNECTED, servers: [] } });
  try {
    await page.goto(stub.baseURL);
    await openConnectors(page);
    await page.getByRole('button', { name: 'Add…' }).click();

    // `my server__read_file` is not a legal function name, so the model
    // could never call it — the user finds out here rather than through a
    // connector whose tools mysteriously never run.
    await page.getByLabel('Name').fill('my server');
    await page.getByLabel('Command').fill('npx');
    await expect(page.getByRole('button', { name: 'Save' })).toBeDisabled();
    await expect(page.getByText(/letters, numbers/)).toBeVisible();

    await page.getByLabel('Name').fill('my-server');
    await expect(page.getByRole('button', { name: 'Save' })).toBeEnabled();
  } finally {
    await stub.close();
  }
});

test('a row says whether its server actually connected', async ({ page }) => {
  const stub = await startStub({
    ...READY,
    connectors: {
      ...CONNECTED,
      engine_servers: [
        {
          name: 'filesystem',
          state: 'error',
          transport: 'stdio',
          tools_count: 0,
          error: 'command not found: npx',
        },
      ],
    },
  });
  try {
    await page.goto(stub.baseURL);
    await openConnectors(page);

    // The reason, not just a failure: this is the question the panel exists
    // to answer, and the config file cannot answer it at all.
    await expect(page.getByText('command not found: npx')).toBeVisible();
  } finally {
    await stub.close();
  }
});

test('the restart banner appears only when a restart could fix it', async ({ page }) => {
  const stub = await startStub({
    ...READY,
    connectors: { ...CONNECTED, configured: false, needs_restart: true },
  });
  try {
    await page.goto(stub.baseURL);
    await openConnectors(page);

    // `--mcp-config` is read once at spawn, so a child started before the
    // switch was flipped cannot pick connectors up. The banner carries a
    // real button rather than telling the user to go cycle the model
    // themselves.
    await expect(page.getByText(/Restart the model to finish/)).toBeVisible();
    await page.getByRole('button', { name: 'Restart' }).click();

    expect(stub.scenario.connectorRestarts).toBe(1);
    await expect(page.getByText(/Restart the model to finish/)).toBeHidden();
  } finally {
    await stub.close();
  }
});

test('a connector tool is offered to the model and runs once approved', async ({ page }) => {
  const stub = await startStub({
    ...READY,
    connectors: CONNECTED,
    chatFrames: [
      toolCallFrames([
        {
          id: 'call_1',
          name: 'filesystem__read_file',
          arguments: '{"path":"/tmp/notes.txt"}',
        },
      ]),
      [`data: ${JSON.stringify({ choices: [{ delta: { content: 'The file says hi.' } }] })}\n\n`],
    ],
    connectorResults: { filesystem__read_file: { content: '[FILE] a.txt\n[DIR] notes' } },
  });
  try {
    await page.goto(stub.baseURL);
    await send(page, 'read my notes');

    // Scoped to the dialog: the transcript's tool chip shows the same
    // arguments once the call has run.
    const prompt = page.getByRole('alertdialog');
    await expect(prompt.getByText('Run read_file?')).toBeVisible();
    // The prompt names the SERVER as well as the tool: "run read_file?" is
    // unanswerable without knowing whose read_file it is.
    await expect(prompt.getByText(/from the “filesystem” connector/)).toBeVisible();
    // And the exact arguments the model chose.
    await expect(prompt.getByText('/tmp/notes.txt')).toBeVisible();

    await page.getByRole('button', { name: 'Allow once' }).click();
    await expect(page.getByText('The file says hi.')).toBeVisible();

    expect(stub.scenario.connectorCalls).toHaveLength(1);
    expect(stub.scenario.connectorCalls[0]?.name).toBe('filesystem__read_file');
    const followUp = stub.scenario.chatRequests[1] as {
      messages?: Array<{ role?: string; content?: string }>;
    };
    expect(followUp.messages?.find((message) => message.role === 'tool')?.content).toBe(
      '[FILE] a.txt\n[DIR] notes',
    );
    // Allowing once must NOT persist: the next turn asks again.
    expect(stub.scenario.connectors.granted_tools).toEqual([]);

    const sent = stub.scenario.chatRequests[0]?.tools as Array<{ function: { name: string } }>;
    expect(sent.map((tool) => tool.function.name)).toContain('filesystem__read_file');
  } finally {
    await stub.close();
  }
});

test('declining a connector call is an ordinary answer, not a crash', async ({ page }) => {
  const stub = await startStub({
    ...READY,
    connectors: CONNECTED,
    chatFrames: [
      toolCallFrames([
        { id: 'call_1', name: 'filesystem__write_file', arguments: '{"path":"/etc/passwd"}' },
      ]),
      [`data: ${JSON.stringify({ choices: [{ delta: { content: 'Understood.' } }] })}\n\n`],
    ],
  });
  try {
    await page.goto(stub.baseURL);
    await send(page, 'overwrite that file');

    await page.getByRole('button', { name: "Don't allow" }).click();

    await expect(page.getByText('Understood.')).toBeVisible();
    // Nothing ran, and the model was told why so it can carry on.
    expect(stub.scenario.connectorCalls).toHaveLength(0);
  } finally {
    await stub.close();
  }
});

test('always allow is remembered, and the panel can take it back', async ({ page }) => {
  const stub = await startStub({
    ...READY,
    connectors: CONNECTED,
    chatFrames: [
      toolCallFrames([
        { id: 'call_1', name: 'filesystem__read_file', arguments: '{"path":"/tmp/a"}' },
      ]),
      [`data: ${JSON.stringify({ choices: [{ delta: { content: 'Read it.' } }] })}\n\n`],
    ],
  });
  try {
    await page.goto(stub.baseURL);
    await send(page, 'read a');

    await page.getByRole('button', { name: 'Always allow' }).click();
    await expect(page.getByText('Read it.')).toBeVisible();

    // Polled, not sampled: the grant is written through best-effort so the
    // answer is never held up by it, which means it can land after the text.
    await expect
      .poll(() => stub.scenario.connectors.granted_tools)
      .toEqual(['filesystem__read_file']);

    await openConnectors(page);
    await expect(page.getByText('1 tool permanently allowed.')).toBeVisible();
    await page.getByRole('button', { name: 'Reset' }).click();

    // The grant is per tool, so revoking is too — and the next call asks
    // again. Asserted through the panel, which only says this once the
    // response that cleared them has landed.
    await expect(page.getByText('No tools are permanently allowed.')).toBeVisible();
    expect(stub.scenario.connectors.granted_tools).toEqual([]);
  } finally {
    await stub.close();
  }
});

test('a tool switched off is neither advertised nor run', async ({ page }) => {
  const stub = await startStub({
    ...READY,
    connectors: CONNECTED,
    chatFrames: [
      toolCallFrames([
        { id: 'call_1', name: 'filesystem__write_file', arguments: '{"path":"/tmp/a"}' },
      ]),
      [`data: ${JSON.stringify({ choices: [{ delta: { content: 'Cannot.' } }] })}\n\n`],
    ],
  });
  try {
    await page.goto(stub.baseURL);
    await openConnectors(page);
    await page.getByRole('switch', { name: 'filesystem__write_file' }).click();
    await page.keyboard.press('Escape');

    await send(page, 'write a file');
    await expect(page.getByText('Cannot.')).toBeVisible();

    const sent = stub.scenario.chatRequests[0]?.tools as Array<{ function: { name: string } }>;
    const names = sent.map((tool) => tool.function.name);
    expect(names).toContain('filesystem__read_file');
    expect(names).not.toContain('filesystem__write_file');
    // A model can emit the name anyway, so the page refuses it locally
    // rather than relying on the request body alone.
    expect(stub.scenario.connectorCalls).toHaveLength(0);
  } finally {
    await stub.close();
  }
});

test('nothing is offered while the master switch is off', async ({ page }) => {
  const stub = await startStub({
    ...READY,
    // Tools still listed: the running child keeps its connectors loaded
    // until it is restarted, so "off" has to be enforced rather than
    // inferred from an empty list.
    connectors: { ...CONNECTED, enabled: false },
    chatFrames: [[`data: ${JSON.stringify({ choices: [{ delta: { content: 'Hi.' } }] })}\n\n`]],
  });
  try {
    await page.goto(stub.baseURL);
    await send(page, 'hello');
    await expect(page.getByText('Hi.')).toBeVisible();

    const sent = stub.scenario.chatRequests[0]?.tools as Array<{ function: { name: string } }>;
    expect(sent.map((tool) => tool.function.name)).not.toContain('filesystem__read_file');
  } finally {
    await stub.close();
  }
});

test('removing a connector takes its consent with it', async ({ page }) => {
  const stub = await startStub({
    ...READY,
    connectors: { ...CONNECTED, granted_tools: ['filesystem__read_file'] },
  });
  try {
    await page.goto(stub.baseURL);
    await openConnectors(page);

    await page.getByRole('button', { name: 'Actions for filesystem' }).click();
    await page.getByRole('menuitem', { name: 'Remove' }).click();
    // The program itself is not uninstalled, and the dialog says so.
    await expect(page.getByText(/isn't uninstalled/)).toBeVisible();
    await page.getByRole('button', { name: 'Remove' }).click();

    expect(stub.scenario.connectors.servers).toEqual([]);
    // A grant is keyed on `server__tool`, which a re-added connector would
    // inherit — so it goes with the removal.
    expect(stub.scenario.connectors.granted_tools).toEqual([]);
  } finally {
    await stub.close();
  }
});
