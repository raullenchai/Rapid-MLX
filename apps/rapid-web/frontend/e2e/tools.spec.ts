import { expect, test } from '@playwright/test';
import { startStub, toolCallFrames } from './stub-server';

/**
 * Tool calling.
 *
 * The loop lives in the page — it is what streams the answer — while each
 * call runs on the server, because a browser cannot fetch a cross-origin
 * provider. So the round trip crosses both, and these specs walk it end to
 * end rather than testing either half alone.
 */

const READY = { engineState: 'ready' as const, model: 'qwen3-4b' };

async function send(page: import('@playwright/test').Page, text: string) {
  // Clicked, not Return: the default project is a phone, where a bare Return
  // is a newline.
  await page.getByLabel('Message').fill(text);
  await page.getByRole('button', { name: 'Send' }).click();
}

test('a tool call runs and its answer informs the reply', async ({ page }) => {
  const stub = await startStub({
    ...READY,
    chatFrames: [
      toolCallFrames([{ id: 'call_1', name: 'weather', arguments: '{"location":"Paris"}' }]),
      [`data: ${JSON.stringify({ choices: [{ delta: { content: 'It is 18°C.' } }] })}\n\n`],
    ],
    toolResults: { weather: { content: 'Current weather for Paris: 18.0°C' } },
  });
  try {
    await page.goto(stub.baseURL);
    await send(page, 'weather in Paris?');

    await expect(page.getByText('It is 18°C.')).toBeVisible();

    // The chip names the tool that ran. Its result is collapsed on success —
    // the answer above is what the user asked for — and one click away.
    const chip = page.getByText('weather', { exact: true }).first();
    await expect(chip).toBeVisible();
    await chip.click();
    await expect(page.getByText('Current weather for Paris: 18.0°C')).toBeVisible();

    expect(stub.scenario.toolCalls).toHaveLength(1);
    expect(stub.scenario.toolCalls[0]?.name).toBe('weather');
    // Reassembled from two fragments, which is the accumulator's job.
    expect(stub.scenario.toolCalls[0]?.arguments).toBe('{"location":"Paris"}');
  } finally {
    await stub.close();
  }
});

test('the enabled tools are advertised on the request', async ({ page }) => {
  const stub = await startStub(READY);
  try {
    await page.goto(stub.baseURL);
    await send(page, 'hello');
    await expect(page.getByText('Hello there.')).toBeVisible();

    const sent = stub.scenario.chatRequests[0]?.tools as Array<{
      function: { name: string };
    }>;
    expect(sent.map((tool) => tool.function.name).sort()).toEqual(['browse', 'weather']);
    expect(stub.scenario.chatRequests[0]?.tool_choice).toBe('auto');
  } finally {
    await stub.close();
  }
});

test('a tool-call turn with no prose is sent back as history', async ({ page }) => {
  const stub = await startStub({
    ...READY,
    chatFrames: [
      toolCallFrames([{ id: 'call_1', name: 'weather', arguments: '{}' }]),
      [`data: ${JSON.stringify({ choices: [{ delta: { content: 'Done.' } }] })}\n\n`],
    ],
    toolResults: { weather: { content: '18°C' } },
  });
  try {
    await page.goto(stub.baseURL);
    await send(page, 'weather?');
    await expect(page.getByText('Done.')).toBeVisible();

    // The turn that only asked for a tool has no content. Filtering on empty
    // content would drop it and orphan the result underneath it.
    const messages = stub.scenario.chatRequests[1]?.messages as Array<Record<string, unknown>>;
    const assistant = messages.find((turn) => turn.role === 'assistant');
    expect(assistant?.tool_calls).toHaveLength(1);

    const result = messages.find((turn) => turn.role === 'tool');
    expect(result?.content).toBe('18°C');
    expect(result?.tool_call_id).toBe('call_1');
  } finally {
    await stub.close();
  }
});

test('a dispatch-only turn carries no actions of its own', async ({ page }) => {
  const stub = await startStub({
    ...READY,
    chatFrames: [
      toolCallFrames([{ id: 'call_1', name: 'weather', arguments: '{}' }]),
      [
        `data: ${JSON.stringify({ choices: [{ delta: { content: 'It is 18°C.' } }] })}\n\n`,
        `data: ${JSON.stringify({ usage: { completion_tokens: 41 } })}\n\n`,
      ],
    ],
    toolResults: { weather: { content: '18°C' } },
  });
  try {
    await page.goto(stub.baseURL);
    await send(page, 'weather?');
    await expect(page.getByText('It is 18°C.')).toBeVisible();

    // The turn that only dispatched the call is a step, not an answer: Copy
    // would copy nothing and its throughput belongs to the reply below, which
    // prints its own. So exactly one assistant action row, not two.
    await expect(page.getByRole('button', { name: 'Retry' })).toHaveCount(1);
    await expect(page.getByText(/tokens/)).toHaveCount(1);
  } finally {
    await stub.close();
  }
});

test('a dispatch turn that also narrates still carries no actions', async ({ page }) => {
  // Small models leak the call into their prose as well as emitting it. That
  // turn is still a dispatch, and testing for empty content put a full action
  // and stats row between the call and its own result.
  const narrating = toolCallFrames([
    { id: 'call_1', name: 'web_search', arguments: '{"query":"weather"}' },
  ]);
  const stub = await startStub({
    ...READY,
    chatFrames: [
      [
        `data: ${JSON.stringify({
          choices: [{ delta: { content: '{"name": "web_search"}' } }],
        })}\n\n`,
        ...narrating,
        `data: ${JSON.stringify({ usage: { completion_tokens: 20 } })}\n\n`,
      ],
      [
        `data: ${JSON.stringify({ choices: [{ delta: { content: 'It is 18°C.' } }] })}\n\n`,
        `data: ${JSON.stringify({ usage: { completion_tokens: 41 } })}\n\n`,
      ],
    ],
    toolResults: { web_search: { content: '18°C' } },
  });
  try {
    await page.goto(stub.baseURL);
    await send(page, 'weather?');
    await expect(page.getByText('It is 18°C.')).toBeVisible();

    // The echo is the same dispatch a second time, and the chip below already
    // says what ran — on screen it just looks like the model malfunctioned.
    await expect(page.getByText('{"name": "web_search"', { exact: false })).toHaveCount(0);

    await expect(page.getByRole('button', { name: 'Retry' })).toHaveCount(1);
    await expect(page.getByText(/tokens/)).toHaveCount(1);
  } finally {
    await stub.close();
  }
});

test('a disabled tool is neither advertised nor runnable', async ({ page }) => {
  const stub = await startStub({
    ...READY,
    chatFrames: [
      toolCallFrames([{ id: 'call_1', name: 'browse', arguments: '{"url":"https://x.example"}' }]),
      [`data: ${JSON.stringify({ choices: [{ delta: { content: 'Cannot.' } }] })}\n\n`],
    ],
  });
  try {
    await page.goto(stub.baseURL);
    await page.getByLabel('Open sidebar').click();
    await page.getByRole('button', { name: 'Settings' }).click();
    await page.getByRole('button', { name: 'Tools', exact: true }).click();
    await page.getByRole('switch', { name: 'browse' }).click();
    await page.keyboard.press('Escape');

    await send(page, 'read this page');
    await expect(page.getByText('Cannot.')).toBeVisible();

    // Only weather was offered, and the call for the disabled tool never
    // reached the server: a model can emit one anyway, so the page refuses
    // it locally rather than relying on the request body alone.
    const sent = stub.scenario.chatRequests[0]?.tools as Array<{ function: { name: string } }>;
    expect(sent.map((tool) => tool.function.name)).toEqual(['weather']);
    expect(stub.scenario.toolCalls).toHaveLength(0);
    await expect(page.getByText(/unknown tool 'browse'/)).toBeVisible();
  } finally {
    await stub.close();
  }
});

test('browsing asks before it fetches, and declining is an ordinary answer', async ({ page }) => {
  const stub = await startStub({
    ...READY,
    chatFrames: [
      toolCallFrames([
        { id: 'call_1', name: 'browse', arguments: '{"url":"https://evil.example/leak"}' },
      ]),
      [`data: ${JSON.stringify({ choices: [{ delta: { content: 'Not fetched.' } }] })}\n\n`],
    ],
  });
  try {
    await page.goto(stub.baseURL);
    await send(page, 'read that page');

    // The model chose this URL, so the user sees the exact destination.
    await expect(
      page.getByText('https://evil.example/leak', { exact: true }),
    ).toBeVisible();
    await page.getByRole('button', { name: "Don't allow" }).click();

    await expect(page.getByText('Not fetched.')).toBeVisible();
    // Declined means the request never left.
    expect(stub.scenario.toolCalls).toHaveLength(0);
  } finally {
    await stub.close();
  }
});

test('an approved fetch runs', async ({ page }) => {
  const stub = await startStub({
    ...READY,
    chatFrames: [
      toolCallFrames([
        { id: 'call_1', name: 'browse', arguments: '{"url":"https://example.com/a"}' },
      ]),
      [`data: ${JSON.stringify({ choices: [{ delta: { content: 'The page says hi.' } }] })}\n\n`],
    ],
    toolResults: { browse: { content: '{"content":"hi"}' } },
  });
  try {
    await page.goto(stub.baseURL);
    await send(page, 'read that page');

    await page.getByRole('button', { name: 'Allow', exact: true }).click();
    await expect(page.getByText('The page says hi.')).toBeVisible();
    expect(stub.scenario.toolCalls).toHaveLength(1);
  } finally {
    await stub.close();
  }
});

test('the loop stops at the budget instead of calling forever', async ({ page }) => {
  // Every round asks for another call. Without a cap this never terminates.
  const asking = toolCallFrames([{ id: 'call_x', name: 'weather', arguments: '{}' }]);
  const stub = await startStub({
    ...READY,
    chatFrames: [asking, asking, asking, asking, asking],
    toolResults: { weather: { content: '18°C' } },
  });
  try {
    await page.goto(stub.baseURL);
    await send(page, 'loop please');

    // The fourth call is refused rather than run, and every requested call
    // still gets a row so the wire shape stays valid.
    await expect(page.getByText('weather', { exact: true }).nth(3)).toBeVisible();
    await page.getByText('weather', { exact: true }).nth(3).click();
    await expect(page.getByText(/could not finish after 3 tool calls/).first()).toBeVisible();
    expect(stub.scenario.toolCalls).toHaveLength(3);
  } finally {
    await stub.close();
  }
});

test('a server with no tools sends none and still answers', async ({ page }) => {
  const stub = await startStub({ ...READY, tools: [] });
  try {
    await page.goto(stub.baseURL);
    await send(page, 'hello');

    await expect(page.getByText('Hello there.')).toBeVisible();
    // Absent, not an empty array: some templates emit a tool preamble for
    // `tools: []` and then talk about tools that do not exist.
    expect(stub.scenario.chatRequests[0]).not.toHaveProperty('tools');
  } finally {
    await stub.close();
  }
});
