import { expect, test as base } from '@playwright/test';
import { startStub, chatFrame, type Scenario } from './stub-server';

/**
 * Editing a prompt forks the transcript, and the fork stays reachable.
 *
 * `editAndResend` has always parented the new prompt to the ORIGINAL's parent,
 * so the pre-edit turn survives — but the `‹ 1/2 ›` control only rendered on
 * assistant rows, which left that branch with nothing to walk back to it.
 */

type Stub = Awaited<ReturnType<typeof startStub>>;

const test = base.extend<{ scenario: Partial<Scenario>; stub: Stub }>({
  scenario: [{ frameDelayMs: 3, chatFrames: [chatFrame('ok\n')] }, { option: true }],
  stub: async ({ scenario }, use) => {
    const stub = await startStub(scenario);
    await use(stub);
    await stub.close();
  },
});

async function send(page: import('@playwright/test').Page, text: string) {
  await page.getByLabel('Message').fill(text);
  await page.getByRole('button', { name: 'Send' }).click();
  await expect(page.getByRole('log')).toContainText('ok');
}

/**
 * Rewrite the prompt in place.
 *
 * The editor's Send is scoped to the log: the composer has a Send of its own
 * and it is disabled while a draft is empty, so an unscoped query resolves to
 * a control that never becomes clickable.
 */
async function editPrompt(page: import('@playwright/test').Page, text: string) {
  await page.getByRole('button', { name: 'Edit' }).click();
  await page.getByLabel('Edit message').fill(text);
  await page.getByRole('log').getByRole('button', { name: 'Send' }).click();
}

test('an edited prompt keeps the original one switch away', async ({ page, stub }) => {
  await page.goto(stub.baseURL);
  await send(page, 'first question');

  await editPrompt(page, 'second question');

  const log = page.getByRole('log');
  await expect(log).toContainText('second question');
  await expect(log).not.toContainText('first question');

  // The user row now carries the switcher. Its noun is "version", not
  // "response": the alternatives here are edits of a prompt.
  await expect(log.getByLabel('Version 2 of 2')).toBeVisible();

  await log.getByRole('button', { name: 'Previous version' }).click();
  await expect(log).toContainText('first question');
  await expect(log).not.toContainText('second question');
  await expect(log.getByLabel('Version 1 of 2')).toBeVisible();

  await log.getByRole('button', { name: 'Next version' }).click();
  await expect(log).toContainText('second question');
});

test('the arrows are bounded rather than wrapping', async ({ page, stub }) => {
  await page.goto(stub.baseURL);
  await send(page, 'first question');

  await editPrompt(page, 'second question');
  await expect(page.getByRole('log')).toContainText('second question');

  const log = page.getByRole('log');
  // On the last version, so forward is dead and back is live.
  await expect(log.getByRole('button', { name: 'Next version' })).toBeDisabled();
  await expect(log.getByRole('button', { name: 'Previous version' })).toBeEnabled();

  await log.getByRole('button', { name: 'Previous version' }).click();
  await expect(log.getByRole('button', { name: 'Previous version' })).toBeDisabled();
});

test('switching a prompt carries its answer with it', async ({ page, stub }) => {
  // The switch resolves DOWNWARDS to a leaf, so stepping back to the original
  // prompt must restore the answer that was given to it — not leave the reply
  // from the edited branch stranded under it.
  stub.scenario.chatFrames = [chatFrame('answer to the first')];
  await page.goto(stub.baseURL);
  await send(page, 'first question');

  stub.scenario.chatFrames = [chatFrame('answer to the second')];
  await editPrompt(page, 'second question');

  const log = page.getByRole('log');
  await expect(log).toContainText('answer to the second');

  await log.getByRole('button', { name: 'Previous version' }).click();
  await expect(log).toContainText('answer to the first');
  await expect(log).not.toContainText('answer to the second');
});

test('a retried answer keeps the prompt on one version', async ({ page, stub }) => {
  // Retry forks at the ANSWER, so the prompt above it has no alternatives and
  // must not grow a switcher of its own.
  await page.goto(stub.baseURL);
  await send(page, 'only question');

  await page.getByRole('button', { name: 'Retry' }).click();
  const log = page.getByRole('log');
  await expect(log.getByLabel('Response 2 of 2')).toBeVisible();
  await expect(log.getByLabel(/^Version /)).toHaveCount(0);
});
