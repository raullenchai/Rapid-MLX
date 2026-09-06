import { test, expect } from '@playwright/test';
import { startStub } from './stub-server';

/**
 * The iOS 16px input floor, checked as a COMPUTED size.
 *
 * Below 16px Safari zooms on focus and never zooms back out. It is also a
 * cascade question that is easy to misread: `input { font-size: 16px }` is
 * unlayered, and unlayered rules beat anything in `@layer utilities`, so a
 * `text-sm` on an input is silently a no-op. Measuring is the only way to know.
 */
test('every input renders at 16px or more on a phone', async ({ page }) => {
  const stub = await startStub({});
  try {
    await page.goto(stub.baseURL);
    await expect(page.getByLabel('Message')).toBeVisible();
    // Mount the other fields too. Search now lives in a palette rather than
    // in the sidebar, and its input is the one most at risk: it is the field
    // a phone user types into most often after the composer.
    await page.getByLabel('Open sidebar').click();
    await page.getByRole('button', { name: 'Search conversations' }).click();
    await expect(page.getByPlaceholder('Search conversations')).toBeVisible();

    const sizes = await page.evaluate(() =>
      [...document.querySelectorAll('input, textarea, select')].map((element) => ({
        name: element.getAttribute('aria-label') ?? element.id,
        size: Number.parseFloat(getComputedStyle(element).fontSize),
      })),
    );

    expect(sizes.length).toBeGreaterThan(1);
    for (const { name, size } of sizes) {
      expect(size, `${name} is ${size}px`).toBeGreaterThanOrEqual(16);
    }
  } finally {
    await stub.close();
  }
});
