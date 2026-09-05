import { expect, test } from '@playwright/test';
import { startStub, stubModel } from './stub-server';

/**
 * Instruction editing.
 *
 * A mode of the images surface, not a separate page: it is entered by
 * supplying a source image and everything the edit backends ignore — aspect,
 * long edge — disappears while it is on.
 */

async function openImages(page: import('@playwright/test').Page, baseURL: string) {
  await page.goto(baseURL);
  await expect(page.getByLabel('Message')).toBeVisible();
  const drawer = page.getByLabel('Open sidebar');
  if (await drawer.isVisible()) await drawer.click();
  await page.getByRole('button', { name: 'Images' }).click();
}

async function render(page: import('@playwright/test').Page, prompt: string) {
  await page.getByLabel('Prompt').fill(prompt);
  await page.getByRole('button', { name: 'Generate' }).click();
  await expect(page.getByAltText(prompt)).toBeVisible();
}

test('a result can be edited, and the edit chains off it', async ({ page }) => {
  const stub = await startStub({ engineState: 'ready', model: 'flux2-klein-4b' });
  try {
    await openImages(page, stub.baseURL);
    await render(page, 'a cat on a bicycle');

    await page.getByRole('button', { name: 'Edit image' }).click();

    // The strip is the only thing that says why a prompt is now producing a
    // variation of an earlier image.
    await expect(page.getByText('Editing image')).toBeVisible();
    await expect(page.getByLabel('Prompt')).toHaveAttribute(
      'placeholder',
      /what you want to change/,
    );

    await page.getByLabel('Prompt').fill('make it night');
    await page.getByRole('button', { name: 'Edit image', exact: true }).click();

    // Named explicitly: an omitted model resolves to the engine's PRIMARY,
    // which is the chat model whenever the image model was hot-loaded.
    await expect
      .poll(() => stub.scenario.edits)
      .toEqual([{ prompt: 'make it night', model: 'flux2-klein-4b' }]);

    // Still editing, and the instruction is cleared: the next edit acts on
    // what was just produced, not on the original.
    await expect(page.getByText('Editing image')).toBeVisible();
    await expect(page.getByLabel('Prompt')).toHaveValue('');
  } finally {
    await stub.close();
  }
});

test('a file can be imported to edit, under the real CSP', async ({ page }) => {
  const stub = await startStub({ engineState: 'ready', model: 'flux2-klein-4b' });
  try {
    // Imported files use object URLs to avoid Base64 copies, so the real CSP
    // must explicitly admit blob: previews.
    const violations: string[] = [];
    page.on('console', (message) => {
      if (/Content Security Policy/i.test(message.text())) violations.push(message.text());
    });

    await openImages(page, stub.baseURL);

    // A 1x1 PNG, the same one the stub renders.
    // The visible control is a button that clicks a hidden input; the input
    // is what Playwright can set files on.
    await page.locator('input[type="file"]').setInputFiles({
      name: 'beach sunset.png',
      mimeType: 'image/png',
      buffer: Buffer.from(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
        'base64',
      ),
    });

    await expect(page.getByText('Editing image')).toBeVisible();
    // The filename without its extension, matching rapid-mac.
    await expect(page.getByText('beach sunset')).toBeVisible();
    expect(violations).toEqual([]);

    await page.getByLabel('Prompt').fill('make it night');
    await page.getByRole('button', { name: 'Edit image', exact: true }).click();
    await expect
      .poll(() => stub.scenario.edits)
      .toEqual([{ prompt: 'make it night', model: 'flux2-klein-4b' }]);
  } finally {
    await stub.close();
  }
});

test('a file that is not an image is refused with copy that says why', async ({ page }) => {
  const stub = await startStub({ engineState: 'ready', model: 'flux2-klein-4b' });
  try {
    await openImages(page, stub.baseURL);

    // Sniffed from the bytes, not taken from the declared type — some
    // pickers report nothing at all.
    await page.locator('input[type="file"]').setInputFiles({
      name: 'notes.png',
      mimeType: 'image/png',
      buffer: Buffer.from('this is not a png'),
    });

    await expect(page.getByText('Choose a PNG or JPEG image.')).toBeVisible();
    await expect(page.getByText('Editing image')).toHaveCount(0);
  } finally {
    await stub.close();
  }
});

test('the size controls disappear while editing', async ({ page }) => {
  const stub = await startStub({ engineState: 'ready', model: 'flux2-klein-4b' });
  try {
    await openImages(page, stub.baseURL);
    await render(page, 'a cat on a bicycle');

    await expect(page.getByRole('button', { name: 'Long edge' })).toBeVisible();
    await page.getByRole('button', { name: 'Edit image' }).click();

    // Removed, not disabled: the engine derives the canvas from the source
    // image and discards `size`, so a control still present would lie.
    await expect(page.getByRole('button', { name: 'Long edge' })).toHaveCount(0);
    await expect(page.getByRole('radiogroup', { name: 'Aspect ratio' })).toHaveCount(0);

    await page.getByRole('button', { name: 'Exit image editing' }).click();
    await expect(page.getByRole('button', { name: 'Long edge' })).toBeVisible();
  } finally {
    await stub.close();
  }
});

test('the picker offers only models that accept the shape being sent', async ({ page }) => {
  const stub = await startStub({
    engineState: 'ready',
    model: 'flux2-klein-4b',
    models: [
      stubModel({
        alias: 'flux2-klein-4b',
        kind: 'image',
        cached: true,
        cached_bytes: 4_600_000_000,
        image_capability: 'both',
      }),
      stubModel({
        alias: 'z-image-turbo',
        kind: 'image',
        cached: true,
        cached_bytes: 6_000_000_000,
        image_capability: 'generation',
      }),
      stubModel({
        alias: 'qwen-image-edit',
        kind: 'image',
        cached: true,
        cached_bytes: 20_000_000_000,
        image_capability: 'editing',
      }),
    ],
  });
  try {
    await openImages(page, stub.baseURL);

    // Generating: the edit-only checkpoint would 409, so it is not offered.
    await page.getByRole('button', { name: /^Model:/ }).click();
    await expect(page.getByRole('menuitem', { name: 'z-image-turbo' })).toBeVisible();
    await expect(page.getByRole('menuitem', { name: 'qwen-image-edit' })).toHaveCount(0);
    await page.keyboard.press('Escape');

    await render(page, 'a cat on a bicycle');
    await page.getByRole('button', { name: 'Edit image' }).click();

    // Editing: the mirror image.
    await page.getByRole('button', { name: /^Model:/ }).click();
    await expect(page.getByRole('menuitem', { name: 'qwen-image-edit' })).toBeVisible();
    await expect(page.getByRole('menuitem', { name: 'z-image-turbo' })).toHaveCount(0);
  } finally {
    await stub.close();
  }
});

test('a text-to-image model cannot be asked for an edit', async ({ page }) => {
  const stub = await startStub({
    engineState: 'ready',
    model: 'z-image-turbo',
    models: [
      stubModel({
        alias: 'z-image-turbo',
        kind: 'image',
        cached: true,
        cached_bytes: 6_000_000_000,
        image_capability: 'generation',
      }),
    ],
  });
  try {
    await openImages(page, stub.baseURL);
    await render(page, 'a cat on a bicycle');
    await page.getByRole('button', { name: 'Edit image' }).click();

    // Refused here rather than surfaced as the engine's 409: the model IS
    // loaded and running, it just takes the other request shape, which no
    // lifecycle band can say.
    await expect(page.getByText(/z-image-turbo is text-to-image only/)).toBeVisible();
    await expect(page.getByLabel('Prompt')).toHaveAttribute(
      'placeholder',
      'Choose an edit-capable model first',
    );
    await page.getByLabel('Prompt').fill('make it night');
    await expect(page.getByRole('button', { name: 'Edit image', exact: true })).toBeDisabled();
  } finally {
    await stub.close();
  }
});
