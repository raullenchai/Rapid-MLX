import { expect, test as base } from '@playwright/test';
import { openModelList, startStub, type Scenario } from './stub-server';

/**
 * Deleting a downloaded model from the model sheet.
 *
 * What has to hold: the trash appears only where there is something to delete,
 * never doubles as a row selection, and asks first.
 */

type Stub = Awaited<ReturnType<typeof startStub>>;

const test = base.extend<{ scenario: Partial<Scenario>; stub: Stub }>({
  // `engineState: 'stopped'` with no model loaded: the server refuses to
  // delete whatever it is running, and the point here is the delete path.
  scenario: [{ engineState: 'stopped', model: null }, { option: true }],
  stub: async ({ scenario }, use) => {
    const stub = await startStub(scenario);
    await use(stub);
    await stub.close();
  },
});

async function openModelSheet(page: import('@playwright/test').Page, baseURL: string) {
  await page.goto(baseURL);
  await expect(page.getByLabel('Message')).toBeVisible();
  return openModelList(page);
}

test('only a downloaded model offers a delete', async ({ page, stub }) => {
  const sheet = await openModelSheet(page, stub.baseURL);

  // qwen3-4b is cached, llama-8b is not. Offering a delete on a model that
  // is not on disk would promise to free space that was never taken.
  await expect(sheet.getByRole('button', { name: 'Delete qwen3-4b' })).toBeVisible();
  await expect(sheet.getByRole('button', { name: 'Delete llama-8b' })).toHaveCount(0);
});

test('deleting asks first and cancelling changes nothing', async ({ page, stub }) => {
  const sheet = await openModelSheet(page, stub.baseURL);

  await sheet.getByRole('button', { name: 'Delete qwen3-4b' }).click();

  const confirm = page.getByRole('alertdialog');
  await expect(confirm).toBeVisible();
  // The size is in the body: the native confirm cannot say what the action
  // costs, which is the whole reason this dialog exists.
  await expect(confirm).toContainText('Frees');

  await confirm.getByRole('button', { name: 'Cancel' }).click();
  await expect(confirm).toBeHidden();
  expect(stub.scenario.removed).toEqual([]);
});

test('confirming deletes and the row stops claiming to be on disk', async ({ page, stub }) => {
  const sheet = await openModelSheet(page, stub.baseURL);

  await sheet.getByRole('button', { name: 'Delete qwen3-4b' }).click();
  await page.getByRole('alertdialog').getByRole('button', { name: 'Delete' }).click();

  await expect.poll(() => stub.scenario.removed).toEqual(['qwen3-4b']);

  // The page re-reads the catalog after a delete rather than waiting out the
  // server's scan TTL, so the affordance goes with the weights.
  await expect(sheet.getByRole('button', { name: 'Delete qwen3-4b' })).toHaveCount(0);
});

test('the delete button does not also select the model', async ({ page, stub }) => {
  // The row behind the trash is itself a click target. Without the two being
  // separate elements, tapping delete would ALSO start a multi-minute load of
  // the model being deleted.
  const sheet = await openModelSheet(page, stub.baseURL);

  await sheet.getByRole('button', { name: 'Delete qwen3-4b' }).click();

  // The confirmation is up and the sheet did not close, which is what
  // choosing a model does.
  await expect(page.getByRole('alertdialog')).toBeVisible();
  await page.getByRole('alertdialog').getByRole('button', { name: 'Cancel' }).click();
  await expect(sheet).toBeVisible();
});

test.describe('when the server refuses', () => {
  test.use({
    scenario: {
      engineState: 'ready',
      model: 'qwen3-4b',
      removeFailure: {
        status: 409,
        type: 'model_in_use',
        message: 'qwen3-4b is the model this server is running.',
      },
    },
  });

  test('the reason is shown verbatim', async ({ page, stub }) => {
    const sheet = await openModelSheet(page, stub.baseURL);

    await sheet.getByRole('button', { name: 'Delete qwen3-4b' }).click();
    await page.getByRole('alertdialog').getByRole('button', { name: 'Delete' }).click();

    // The server names WHICH holder — the engine or an in-flight download —
    // and the fix differs, so a paraphrase would lose the actionable part.
    await expect(page.getByText('qwen3-4b is the model this server is running.')).toBeVisible();
  });
});
