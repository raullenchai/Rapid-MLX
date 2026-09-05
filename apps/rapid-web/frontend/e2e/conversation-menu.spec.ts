import { expect, test as base } from '@playwright/test';
import { startStub, chatFrame, type Scenario } from './stub-server';

/**
 * The conversation row's `···` menu, with Pin left outside it.
 *
 * What matters: moving the actions behind a trigger did not make them
 * unreachable, and opening the menu does not activate the row underneath.
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

/** Seed one conversation so the sidebar has a row to act on. */
async function seedConversation(page: import('@playwright/test').Page, baseURL: string) {
  await page.goto(baseURL);
  await page.getByLabel('Message').fill('a seeded conversation');
  await page.getByRole('button', { name: 'Send' }).click();
  await expect(page.getByRole('log')).toContainText('ok');
  await page.getByLabel('Open sidebar').click();
  return page.getByRole('dialog', { name: 'Conversations' });
}

test('every action is reachable from the menu', async ({ page, stub }) => {
  const drawer = await seedConversation(page, stub.baseURL);

  await drawer.getByRole('button', { name: 'Conversation actions' }).first().click();

  // A menu, not a set of loose buttons — so it traps arrow keys and Escape.
  const menu = page.getByRole('menu');
  await expect(menu).toBeVisible();
  for (const name of ['Rename', 'Pin', 'Archive', 'Delete']) {
    await expect(menu.getByRole('menuitem', { name })).toBeVisible();
  }
});

test('opening the menu does not switch conversation', async ({ page, stub }) => {
  // The row behind the trigger is itself a click target. Without
  // stopPropagation the menu opens AND the row activates, which on a
  // different row would silently navigate away from what you were reading.
  const drawer = await seedConversation(page, stub.baseURL);

  await drawer.getByRole('button', { name: 'Conversation actions' }).first().click();

  await expect(page.getByRole('menu')).toBeVisible();

  // The drawer is still on screen: activating the row would have closed it.
  //
  // NOT `expect(drawer).toBeVisible()`. The menu is modal, so Radix marks the
  // background `aria-hidden` while it is open — which is correct, and which
  // takes the drawer out of the accessibility tree that `getByRole` searches.
  // A role query here fails against a MORE correct implementation, so this
  // measures layout instead.
  const stillOnScreen = await page.evaluate(() => {
    const panel = document.querySelector('[aria-label="Conversations"][role="dialog"]');
    return panel ? panel.getBoundingClientRect().width > 0 : false;
  });
  expect(stillOnScreen).toBe(true);
});

test('an action taken from the menu applies', async ({ page, stub }) => {
  const drawer = await seedConversation(page, stub.baseURL);

  await drawer.getByRole('button', { name: 'Conversation actions' }).first().click();
  await page.getByRole('menuitem', { name: 'Archive' }).click();

  // Archived rows leave the default list for a collapsed section: the row is
  // out of the way but still accounted for.
  await expect(drawer.getByRole('button', { name: 'Archived (1)' })).toBeVisible();
  await expect(drawer).not.toContainText('a seeded conversation');

  // Collapsed by DEFAULT — archiving is how a conversation is put out of the
  // way, so expanding it on every visit would undo what it is for.
  await drawer.getByRole('button', { name: 'Archived (1)' }).click();
  await expect(drawer).toContainText('a seeded conversation');

  // And it is archived rather than deleted. Search also still reaches it,
  // which is the archive's other recovery path.
  await drawer.getByRole('button', { name: 'Search conversations' }).click();
  await page.getByPlaceholder('Search conversations').fill('seeded');
  await expect(page.getByRole('dialog')).toContainText('a seeded conversation');
});

test('Delete asks before destroying anything', async ({ page, stub }) => {
  const drawer = await seedConversation(page, stub.baseURL);

  await drawer.getByRole('button', { name: 'Conversation actions' }).first().click();
  await page.getByRole('menuitem', { name: 'Delete' }).click();

  // A confirmation, not an immediate delete — this is the one irreversible
  // action in the menu.
  const confirm = page.getByRole('alertdialog');
  await expect(confirm).toBeVisible();
  await confirm.getByRole('button', { name: 'Cancel' }).click();
  await expect(drawer).toContainText('a seeded conversation');
});
