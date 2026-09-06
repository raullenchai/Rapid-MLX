import { expect, test } from '@playwright/test';
import { startStub } from './stub-server';

/**
 * The two instruction layers.
 *
 * Both are merged into ONE system row before the request leaves, because
 * local chat templates often reject a second system message — so what the
 * wire carries is what these specs assert on.
 */

const READY = { engineState: 'ready' as const, model: 'qwen3-4b' };

async function setGlobal(page: import('@playwright/test').Page, text: string) {
  await page.getByLabel('Open sidebar').click();
  await page.getByRole('button', { name: 'Settings' }).click();
  await page.getByRole('button', { name: 'System Prompt', exact: true }).click();
  await page.getByRole('textbox', { name: 'System prompt', exact: true }).fill(text);
  await page.keyboard.press('Escape');
}

/**
 * The conversation's own prompt is a COMPOSER control, matching rapid-mac —
 * it is a property of the message about to be sent, not a device preference.
 * It commits on Save rather than as the user types.
 */
async function setConversation(page: import('@playwright/test').Page, text: string) {
  await page.getByRole('button', { name: 'Conversation system prompt' }).click();
  await page
    .getByRole('textbox', { name: 'Conversation system prompt', exact: true })
    .fill(text);
  await page.getByRole('button', { name: 'Save' }).click();
}

async function send(page: import('@playwright/test').Page, text: string) {
  await page.getByLabel('Message').fill(text);
  await page.getByRole('button', { name: 'Send' }).click();
}

test('the conversation prompt outranks the global default on the wire', async ({ page }) => {
  const stub = await startStub(READY);
  try {
    await page.goto(stub.baseURL);

    await setGlobal(page, 'Be brief.');
    await setConversation(page, 'Be thorough.');

    await send(page, 'hello');
    await expect(page.getByText('Hello there.')).toBeVisible();

    const messages = stub.scenario.chatRequests[0]?.messages as Array<Record<string, string>>;
    const system = messages.filter((turn) => turn.role === 'system');
    // One row, not two.
    expect(system).toHaveLength(1);

    const content = system[0]?.content ?? '';
    expect(content.indexOf('Be brief.')).toBeLessThan(content.indexOf('Be thorough.'));
    expect(content).toContain('HIGHEST USER PRIORITY');
  } finally {
    await stub.close();
  }
});

test('cancelling the editor leaves the prompt untouched', async ({ page }) => {
  const stub = await startStub(READY);
  try {
    await page.goto(stub.baseURL);
    await setConversation(page, 'Be thorough.');

    // A draft, not a live write: a system prompt silently changing what the
    // next turn does is the case that wants a commit step.
    await page.getByRole('button', { name: 'Conversation system prompt' }).click();
    await page
      .getByRole('textbox', { name: 'Conversation system prompt', exact: true })
      .fill('Discard me.');
    await page.getByRole('button', { name: 'Cancel' }).click();

    await send(page, 'hello');
    await expect(page.getByText('Hello there.')).toBeVisible();

    const messages = stub.scenario.chatRequests[0]?.messages as Array<Record<string, string>>;
    const system = messages.find((turn) => turn.role === 'system');
    expect(system?.content).toBe('Be thorough.');
  } finally {
    await stub.close();
  }
});

test('a conversation prompt belongs to its own conversation', async ({ page }) => {
  const stub = await startStub(READY);
  try {
    await page.goto(stub.baseURL);

    await setConversation(page, 'Only this chat.');

    // A new chat starts clean: the prompt travels with the history, not with
    // the device.
    await page.getByLabel('Open sidebar').click();
    await page.getByRole('button', { name: 'New chat' }).first().click();

    await send(page, 'hello');
    await expect(page.getByText('Hello there.')).toBeVisible();

    const messages = stub.scenario.chatRequests[0]?.messages as Array<Record<string, string>>;
    expect(messages.some((turn) => turn.role === 'system')).toBe(false);
  } finally {
    await stub.close();
  }
});
