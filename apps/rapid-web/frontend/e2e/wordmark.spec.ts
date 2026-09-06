import { expect, test } from '@playwright/test';
import { openModelList, startStub } from './stub-server';

/**
 * The brand mark in the sidebar header.
 *
 * The geometry is copied verbatim from `RapidRMark.swift`, so the risk here is
 * not "is the path right" — it is the two ways an inlined SVG silently fails:
 * it renders at zero size, or it does not follow the theme and becomes an
 * invisible black shape on a dark surface. Both are measured rather than
 * asserted from class names.
 */

test('the mark renders at a real size beside the wordmark', async ({ page }) => {
  const stub = await startStub({});
  try {
    await page.goto(stub.baseURL);
    await expect(page.getByLabel('Message')).toBeVisible();
    await page.getByLabel('Open sidebar').click();

    const box = await page.locator('svg[viewBox="0 0 192 192"]').first().boundingBox();
    expect(box).not.toBeNull();
    // An SVG with no intrinsic size collapses; `em` sizing off the wordmark
    // should put this in the high-teens at the header's `text-lg`.
    expect(box!.width).toBeGreaterThan(12);
    expect(box!.height).toBeGreaterThan(12);
  } finally {
    await stub.close();
  }
});

test('the mark follows the theme rather than staying black', async ({ page }) => {
  const stub = await startStub({});
  try {
    await page.emulateMedia({ colorScheme: 'light' });
    await page.goto(stub.baseURL);
    await expect(page.getByLabel('Message')).toBeVisible();
    await page.getByLabel('Open sidebar').click();

    const mark = page.locator('svg[viewBox="0 0 192 192"]').first();
    // `fill="currentColor"`, so the computed FILL is what matters — reading
    // `color` would pass even if the fill had been hard-coded.
    const fill = () => mark.evaluate((element) => getComputedStyle(element).fill);

    const light = await fill();
    await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'dark'));
    const dark = await fill();

    // Ink on a dark surface must not stay ink. The SVG's white background
    // `<rect>` is deliberately not reproduced for the same reason: it would be
    // a white tile in dark mode.
    expect(dark).not.toBe(light);
  } finally {
    await stub.close();
  }
});

test('the mark is decorative, so the product is announced once', async ({ page }) => {
  const stub = await startStub({});
  try {
    await page.goto(stub.baseURL);
    await expect(page.getByLabel('Message')).toBeVisible();
    await page.getByLabel('Open sidebar').click();

    // The wordmark beside it already carries the name; an unhidden mark would
    // have a screen reader read the product twice in a row.
    const hidden = await page
      .locator('svg[viewBox="0 0 192 192"]')
      .first()
      .getAttribute('aria-hidden');
    expect(hidden).toBe('true');
  } finally {
    await stub.close();
  }
});

/**
 * The sheet chrome, shared by every modal window (Model, Settings).
 *
 * Two things a class-name grep cannot check: that the close control is an icon
 * with an accessible name rather than the word "Done", and that two sheets
 * opened one after the other are the SAME size. The latter used to be a
 * `max-h`, so each sized to its own content — Settings ran to 630px while the
 * model picker sat at 259px, and switching between them made the dialog jump.
 */
test.describe('sheet chrome', () => {
  test.use({ viewport: { width: 1100, height: 900 }, hasTouch: false, isMobile: false });

  test('closes with an icon that still has an accessible name', async ({ page }) => {
    const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b' });
    try {
      await page.goto(stub.baseURL);
      await expect(page.getByLabel('Message')).toBeVisible();
      await openModelList(page);

      const sheet = page.getByRole('dialog', { name: 'Settings' });
      await expect(sheet).toBeVisible();
      // No visible "Done" text, but the control is still reachable by name.
      await expect(sheet.getByText('Done', { exact: true })).toHaveCount(0);
      const close = sheet.getByRole('button', { name: 'Close' });
      await expect(close).toBeVisible();

      await close.click();
      await expect(sheet).toBeHidden();
    } finally {
      await stub.close();
    }
  });

  test('every sheet is the same size on a desktop window', async ({ page }) => {
    const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b' });
    try {
      await page.goto(stub.baseURL);
      await expect(page.getByLabel('Message')).toBeVisible();

      // The sheet zooms in, so a box read the instant it becomes visible is a
      // mid-animation value (measured 611.5 vs 608.9 for two sheets that both
      // settle at exactly the same size). Wait for it to stop changing.
      const settledBox = async (name: string) => {
        const dialog = page.getByRole('dialog', { name });
        await expect(dialog).toBeVisible();
        let last = -1;
        await expect
          .poll(async () => {
            const width = (await dialog.boundingBox())?.width ?? 0;
            const stable = width === last;
            last = width;
            return stable;
          })
          .toBe(true);
        return dialog.boundingBox();
      };

      // The settings window opens on either the model list or the chat
      // preferences depending on which control was used. Both are the same
      // dialog now, so the comparison that still matters is against the
      // search palette — built on `CommandDialog`, NOT on `Sheet`, so it is
      // the one most likely to drift. It shipped at 512 x 310 against the
      // sheets' 640 x 720, and shares the size through `SHEET_DESKTOP_SIZE`.
      await openModelList(page);
      const models = await settledBox('Settings');
      await page.keyboard.press('Escape');
      await expect(page.getByRole('dialog', { name: 'Settings' })).toBeHidden();

      await page.getByRole('button', { name: 'Search conversations' }).click();
      const search = await settledBox('Search conversations');

      expect(models).not.toBeNull();
      expect(search).not.toBeNull();
      expect(search!.width).toBe(models!.width);
      expect(search!.height).toBe(models!.height);
    } finally {
      await stub.close();
    }
  });

  test('the settings window does not resize when the category changes', async ({ page }) => {
    // The panel area is one scroll container whose content is swapped, so a
    // short panel cannot shrink the window under a tall one. With a `max-h`
    // each sized to its own content and switching made the dialog jump.
    const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b' });
    try {
      await page.goto(stub.baseURL);
      await expect(page.getByLabel('Message')).toBeVisible();
      await openModelList(page);

      const dialog = page.getByRole('dialog', { name: 'Settings' });
      await expect(dialog).toBeVisible();
      let last = -1;
      await expect
        .poll(async () => {
          const height = (await dialog.boundingBox())?.height ?? 0;
          const stable = height === last;
          last = height;
          return stable;
        })
        .toBe(true);
      const onModels = await dialog.boundingBox();

      await dialog.getByRole('button', { name: 'Appearance' }).click();
      await expect(dialog.getByRole('heading', { name: 'Appearance' })).toBeVisible();
      const onAppearance = await dialog.boundingBox();

      expect(onAppearance!.height).toBe(onModels!.height);
      expect(onAppearance!.width).toBe(onModels!.width);
    } finally {
      await stub.close();
    }
  });

  test('the model picker has no Refresh button', async ({ page }) => {
    const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b' });
    try {
      await page.goto(stub.baseURL);
      await expect(page.getByLabel('Message')).toBeVisible();
      await openModelList(page);

      const sheet = page.getByRole('dialog', { name: 'Settings' });
      await expect(sheet).toBeVisible();
      // Opening the sheet forces a catalog re-read instead, so the button was
      // a manual step for something that now happens on its own.
      await expect(sheet.getByRole('button', { name: /Refresh/ })).toHaveCount(0);
    } finally {
      await stub.close();
    }
  });

  test('a phone still gets a content-sized bottom sheet', async ({ page }) => {
    // The fixed height is `sm:`-scoped. Applied unconditionally it would push
    // a two-row picker up over most of a phone screen.
    await page.setViewportSize({ width: 390, height: 664 });
    const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b' });
    try {
      await page.goto(stub.baseURL);
      await expect(page.getByLabel('Message')).toBeVisible();
      await openModelList(page);

      const box = await page.getByRole('dialog', { name: 'Settings' }).boundingBox();
      expect(box).not.toBeNull();
      // Full-bleed — not the desktop's fixed block.
      expect(box!.width).toBe(390);
      expect(box!.height).toBeLessThan(664);
    } finally {
      await stub.close();
    }
  });
});
