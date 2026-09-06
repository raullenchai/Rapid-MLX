import { expect, test } from '@playwright/test';
import { startStub } from './stub-server';

/**
 * A new turn must never show the previous turn's answer.
 *
 * The streaming buffer is a SINGLE module-level store, and any row with
 * `status: 'streaming'` renders whatever is in it. The placeholder node is
 * appended synchronously by `send()`, but `runTurn` makes two fetches
 * (`/api/tools`, `/api/connectors`) before it opens the stream — so unless
 * the buffer is cleared before those awaits, the last answer sits under the
 * new prompt until the first token replaces it.
 *
 * User-reported through a tunnel, where those round trips are slow enough to
 * read. `preStreamDelayMs` is what makes the same window visible here; with
 * it at 0 this passes against the defect.
 */

const FIRST = 'The capital of France is Paris.';
const SECOND = 'Berlin is the capital of Germany.';

function frames(text: string) {
  return [`data: ${JSON.stringify({ choices: [{ delta: { content: text } }] })}\n\n`];
}

/**
 * Record whether `needle` EVER appears in the transcript from now on.
 *
 * A plain `not.toContainText` cannot express this: Playwright retries it
 * until it passes, so text that shows for 700 ms and then vanishes satisfies
 * it — which is precisely the defect. A MutationObserver latches the
 * transient instead of sampling for it.
 */
async function watchForText(page: import('@playwright/test').Page, needle: string) {
  await page.evaluate((text) => {
    const flag = { seen: false };
    (window as unknown as Record<string, unknown>).__sawStale = flag;
    const check = () => {
      const log = document.querySelector('[role="log"]');
      if (log?.textContent?.includes(text)) flag.seen = true;
    };
    check();
    new MutationObserver(check).observe(document.body, {
      subtree: true,
      childList: true,
      characterData: true,
    });
  }, needle);

  return async () =>
    page.evaluate(
      () => ((window as unknown as Record<string, { seen: boolean }>).__sawStale ?? { seen: true }).seen,
    );
}

test('a new conversation does not flash the previous answer', async ({ page }) => {
  const stub = await startStub({
    engineState: 'ready',
    model: 'qwen3-4b',
    chatFrames: [frames(FIRST), frames(SECOND)],
    preStreamDelayMs: 700,
  });
  try {
    await page.goto(stub.baseURL);

    await page.getByLabel('Message').fill('capital of France?');
    await page.getByRole('button', { name: 'Send' }).click();
    await expect(page.getByRole('log')).toContainText(FIRST);

    // Start a fresh conversation, exactly as the report describes.
    await page.getByLabel('Open sidebar').click();
    await page.getByRole('button', { name: 'New Chat' }).click();
    await expect(page.getByRole('dialog', { name: 'Conversations' })).toHaveCount(0);
    // The new conversation is empty, so the old answer is genuinely gone
    // before the watcher starts — anything it sees after this is the bug.
    await expect(page.getByRole('log')).not.toContainText(FIRST);

    const sawStale = await watchForText(page, FIRST);

    await page.getByLabel('Message').fill('capital of Germany?');
    await page.getByRole('button', { name: 'Send' }).click();
    await expect(page.getByRole('log')).toContainText(SECOND);

    expect(await sawStale()).toBe(false);
  } finally {
    await stub.close();
  }
});

test('a second turn in the SAME conversation does not repeat the first answer', async ({
  page,
}) => {
  // The same window without switching conversations. Here the previous answer
  // is legitimately on screen above, so the failure is it appearing TWICE.
  const stub = await startStub({
    engineState: 'ready',
    model: 'qwen3-4b',
    chatFrames: [frames(FIRST), frames(SECOND)],
    preStreamDelayMs: 700,
  });
  try {
    await page.goto(stub.baseURL);

    await page.getByLabel('Message').fill('capital of France?');
    await page.getByRole('button', { name: 'Send' }).click();
    await expect(page.getByRole('log')).toContainText(FIRST);

    await page.getByLabel('Message').fill('and Germany?');
    await page.getByRole('button', { name: 'Send' }).click();

    // Sampled during the pre-stream window: the placeholder is mounted and no
    // token has arrived, which is when the stale buffer used to be painted.
    await expect(page.getByRole('log')).toContainText('and Germany?');
    await expect(page.getByText(FIRST)).toHaveCount(1);

    await expect(page.getByRole('log')).toContainText(SECOND);
  } finally {
    await stub.close();
  }
});
