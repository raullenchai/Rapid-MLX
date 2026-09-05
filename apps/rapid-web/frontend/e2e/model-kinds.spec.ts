import { expect, test } from '@playwright/test';
import { openModelList, startStub, stubModel } from './stub-server';

/**
 * Model management, grouped by kind.
 *
 * The tabs are self-hiding: a kind with no rows is absent rather than shown
 * empty, matching `rapid-mac`'s `availableKinds`. An always-present empty tab
 * reads as a broken install.
 */

async function openModels(page: import('@playwright/test').Page, baseURL: string) {
  await page.goto(baseURL);
  await expect(page.getByLabel('Message')).toBeVisible();
  const dialog = await openModelList(page);
  await expect(dialog).toBeVisible();
  return dialog;
}

test('the kind tabs list only kinds that have models', async ({ page }) => {
  const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b' });
  try {
    const dialog = await openModels(page, stub.baseURL);
    const tabs = dialog.getByRole('radiogroup', { name: 'Model type' });
    await expect(tabs).toBeVisible();
    await expect(tabs.getByRole('radio', { name: 'Text' })).toBeVisible();
    await expect(tabs.getByRole('radio', { name: 'Image' })).toBeVisible();
    await expect(tabs.getByRole('radio', { name: 'Audio' })).toBeVisible();
  } finally {
    await stub.close();
  }
});

test('a catalog with only text models shows no tabs at all', async ({ page }) => {
  const stub = await startStub({
    engineState: 'ready',
    model: 'qwen3-4b',
    models: [stubModel({ alias: 'qwen3-4b', cached: true, cached_bytes: 1 })],
  });
  try {
    const dialog = await openModels(page, stub.baseURL);
    // One kind is not a choice, so the control would be a row of chrome
    // offering nothing.
    await expect(dialog.getByRole('radiogroup', { name: 'Model type' })).toHaveCount(0);
    await expect(dialog.getByRole('button', { name: /^qwen3-4b/ })).toBeVisible();
  } finally {
    await stub.close();
  }
});

test('each tab shows only its own kind', async ({ page }) => {
  const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b' });
  try {
    const dialog = await openModels(page, stub.baseURL);

    // Text is the default, and an image alias must not be in it: choosing one
    // for chat dead-ends on the first send.
    await expect(dialog.getByRole('button', { name: /^qwen3-4b/ })).toBeVisible();
    await expect(dialog.getByRole('button', { name: /^flux2-klein-4b/ })).toHaveCount(0);

    await dialog.getByRole('radio', { name: 'Image' }).click();
    await expect(dialog.getByRole('button', { name: /^flux2-klein-4b/ })).toBeVisible();
    await expect(dialog.getByRole('button', { name: /^qwen3-4b/ })).toHaveCount(0);

    await dialog.getByRole('radio', { name: 'Audio' }).click();
    // `-turbo` rather than the bare alias: the catalog carries both, and
    // Playwright's name match is a substring, so `whisper-large-v3` resolves
    // two rows and fails strict mode.
    await expect(dialog.getByRole('button', { name: /^whisper-large-v3-turbo/ })).toBeVisible();
  } finally {
    await stub.close();
  }
});

test('an audio model can be started like any other', async ({ page }) => {
  const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b' });
  try {
    const dialog = await openModels(page, stub.baseURL);
    await dialog.getByRole('radio', { name: 'Audio' }).click();
    await dialog.getByRole('button', { name: /^whisper-large-v3-turbo/ }).click();

    // `serve <audio-alias>` boots in audio mode, so this is a real switch —
    // it is just rarely the right one, since the lane already rides on
    // whatever is serving.
    await expect(page.getByText(/cannot be started here/)).toHaveCount(0);
    expect(stub.scenario.model).toBe('whisper-large-v3-turbo');
  } finally {
    await stub.close();
  }
});

test('the search box filters within the selected kind', async ({ page }) => {
  const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b' });
  try {
    const dialog = await openModels(page, stub.baseURL);
    await dialog.getByLabel('Search models').fill('llama');

    await expect(dialog.getByRole('button', { name: /^llama-8b/ })).toBeVisible();
    await expect(dialog.getByRole('button', { name: /^qwen3-4b/ })).toHaveCount(0);
  } finally {
    await stub.close();
  }
});
