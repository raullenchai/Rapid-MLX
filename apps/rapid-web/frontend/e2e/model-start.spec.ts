import { expect, test } from '@playwright/test';
import { openModelList, startStub, stubModel } from './stub-server';

/**
 * Choosing a model must immediately look like starting one.
 *
 * `selectAlias` nulls the cached status so the band stops describing the
 * PREVIOUS model, and the next status poll is up to 15 s away. Without
 * adopting the load response in between, readiness has no serving state for
 * the new alias and resolves to `needsStart` — so the surface offers a Start
 * button for a model that is already starting. That was user-reported from
 * the composer picker, the one call site that had not hand-rolled the
 * adoption. See `state/startModel.ts`.
 */

const MODELS = [
  stubModel({ alias: 'qwen3-4b', cached: true, cached_bytes: 2_400_000_000 }),
  stubModel({ alias: 'llama-8b', cached: true, cached_bytes: 8_200_000_000 }),
];

test('the composer picker reports starting, not Start', async ({ page }) => {
  const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b', models: MODELS });
  try {
    await page.goto(stub.baseURL);
    await expect(page.getByLabel('Message')).toBeVisible();

    // Stall the poll so the assertion lands in the window BETWEEN choosing
    // and the first status that describes the new model. Without it the
    // round trip arrives within milliseconds and the defect is invisible.
    stub.scenario.statusDelayMs = 30_000;

    await page.getByRole('button', { name: /^Model:/ }).click();
    await page.getByRole('menuitem', { name: 'llama-8b' }).click();

    await expect(page.getByText('Starting llama-8b')).toBeVisible();
    // The whole point: the band must not offer to start what is starting.
    await expect(page.getByRole('button', { name: 'Start' })).toHaveCount(0);
    await expect(page.getByLabel('Message')).toHaveAttribute(
      'placeholder',
      'Starting llama-8b…',
    );
  } finally {
    await stub.close();
  }
});

test('the model list reports starting, not Start', async ({ page }) => {
  const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b', models: MODELS });
  try {
    await page.goto(stub.baseURL);
    await expect(page.getByLabel('Message')).toBeVisible();

    const dialog = await openModelList(page);
    stub.scenario.statusDelayMs = 30_000;
    await dialog.getByRole('button', { name: /^llama-8b/ }).click();

    await expect(page.getByText('Starting llama-8b')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Start' })).toHaveCount(0);
  } finally {
    await stub.close();
  }
});

test('the images picker reports starting, not Start', async ({ page }) => {
  const stub = await startStub({
    engineState: 'ready',
    model: 'qwen3-4b',
    models: [
      ...MODELS,
      stubModel({
        alias: 'flux2-klein-4b',
        kind: 'image',
        cached: true,
        cached_bytes: 4_600_000_000,
        image_capability: 'both',
      }),
    ],
  });
  try {
    await page.goto(stub.baseURL);
    await expect(page.getByLabel('Message')).toBeVisible();
    await page.getByLabel('Open sidebar').click();
    await page.getByRole('button', { name: 'Images' }).click();

    stub.scenario.statusDelayMs = 30_000;
    await page.getByRole('button', { name: /^Model:/ }).click();
    await page.getByRole('menuitem', { name: 'flux2-klein-4b' }).click();

    await expect(page.getByText('Starting flux2-klein-4b')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Start' })).toHaveCount(0);
  } finally {
    await stub.close();
  }
});

test('a failed start still offers Retry', async ({ page }) => {
  // The adoption must not swallow a real failure: the poll that follows
  // reports `failed`, and the band has to act on it.
  const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b', models: MODELS });
  try {
    await page.goto(stub.baseURL);
    await expect(page.getByLabel('Message')).toBeVisible();

    await page.getByRole('button', { name: /^Model:/ }).click();
    await page.getByRole('menuitem', { name: 'llama-8b' }).click();
    await expect(page.getByText('Starting llama-8b')).toBeVisible();

    stub.scenario.engineState = 'failed';
    stub.scenario.detail = 'not enough memory';

    // Scoped to the band: the same sentence is on the empty state and in the
    // live region, so an unanchored match resolves three elements.
    const band = page.getByRole('status').filter({ hasText: 'not enough memory' });
    await expect(band).toContainText("Couldn't start llama-8b");
    await expect(band.getByRole('button', { name: 'Retry' })).toBeVisible();
  } finally {
    await stub.close();
  }
});
