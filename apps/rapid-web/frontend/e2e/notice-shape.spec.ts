import { expect, test as base } from '@playwright/test';
import { openModelList, startStub, type Scenario } from './stub-server';

/**
 * The notice's shape on a wide window. Asserted by MEASUREMENT — a class name
 * in the source proves nothing about what the cascade computed.
 *
 * The suite's one project is a phone, where a full-bleed notice is correct, so
 * this overrides the viewport; without that it would silently test nothing.
 */
type Stub = Awaited<ReturnType<typeof startStub>>;

const test = base.extend<{ scenario: Partial<Scenario>; stub: Stub }>({
  scenario: [
    {
      engineState: 'ready',
      model: 'qwen3-4b',
      removeFailure: {
        status: 409,
        type: 'model_in_use',
        message: 'qwen3-4b is the model this server is running.',
      },
    },
    { option: true },
  ],
  stub: async ({ scenario }, use) => {
    const stub = await startStub(scenario);
    await use(stub);
    await stub.close();
  },
});

test.use({ viewport: { width: 1400, height: 800 }, hasTouch: false, isMobile: false });

test('a notice is capped well short of a wide window', async ({ page, stub }) => {
  await page.goto(stub.baseURL);

  // Provoke a real notice through the delete path rather than injecting one:
  // this is the exact notice the screenshot showed.
  await openModelList(page);
  await page.getByRole('button', { name: 'Delete qwen3-4b' }).click();
  await page.getByRole('alertdialog').getByRole('button', { name: 'Delete' }).click();

  const notice = page.getByRole('status').filter({ hasText: 'in use' });
  await expect(notice).toBeVisible();

  const box = await notice.boundingBox();
  expect(box).not.toBeNull();
  // It used to fill the window. The cap is 28rem; the bound is loose because
  // this asserts "capped", not a specific number.
  expect(box!.width).toBeLessThan(600);
});

test('a notice has no tone-coloured stripe', async ({ page, stub }) => {
  await page.goto(stub.baseURL);

  await openModelList(page);
  await page.getByRole('button', { name: 'Delete qwen3-4b' }).click();
  await page.getByRole('alertdialog').getByRole('button', { name: 'Delete' }).click();

  const notice = page.getByRole('status').filter({ hasText: 'in use' });
  await expect(notice).toBeVisible();

  const border = await notice.evaluate((element) => {
    const style = getComputedStyle(element);
    return {
      leftWidth: style.borderLeftWidth,
      topWidth: style.borderTopWidth,
      leftColor: style.borderLeftColor,
      topColor: style.borderTopColor,
    };
  });

  // A uniform hairline is the card's own edge. The defect was a 3px amber
  // stripe on the left only, so both width and colour must match the top.
  expect(border.leftWidth).toBe(border.topWidth);
  expect(border.leftColor).toBe(border.topColor);
});
