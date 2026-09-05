import { expect, test as base } from '@playwright/test';
import { startStub, type Scenario } from './stub-server';

/**
 * Return-to-send, which the default project cannot cover.
 *
 * `playwright.config.ts` runs one project (iPhone 14), but this behaviour is
 * keyboard-only: on touch a bare Return must stay a newline. So the desktop
 * half needs a context with a fine pointer and the touch half the default one.
 */

type Stub = Awaited<ReturnType<typeof startStub>>;

const test = base.extend<{ scenario: Partial<Scenario>; stub: Stub }>({
  scenario: [{}, { option: true }],
  stub: async ({ scenario }, use) => {
    const stub = await startStub(scenario);
    await use(stub);
    await stub.close();
  },
});

test.describe('with a keyboard', () => {
  // hasTouch/isMobile false is what makes `(hover: none) and (pointer:
  // coarse)` stop matching, which is the condition the composer keys on.
  test.use({ hasTouch: false, isMobile: false, viewport: { width: 1280, height: 800 } });

  test('Return sends the message', async ({ page, stub }) => {
    await page.goto(stub.baseURL);

    const field = page.getByLabel('Message');
    await field.fill('sent with return');
    await field.press('Enter');

    await expect(page.getByRole('log')).toContainText('sent with return');
    // And the draft is consumed, or the next message would repeat it.
    await expect(field).toHaveValue('');
  });

  test('Shift+Return inserts a newline instead of sending', async ({ page, stub }) => {
    // Without this there is no way to write a multi-line prompt at all on a
    // desktop, which is where long prompts are actually written.
    await page.goto(stub.baseURL);

    const field = page.getByLabel('Message');
    await field.fill('first line');
    await field.press('Shift+Enter');
    await field.pressSequentially('second line');

    await expect(field).toHaveValue('first line\nsecond line');
    await expect(page.getByRole('log')).not.toContainText('first line');
  });

  test('a Return that commits an IME candidate does not send', async ({ page, stub }) => {
    // The bug this exists for: typing Chinese, Japanese or Korean ends every
    // candidate selection with Return. If that sends, the message is
    // truncated mid-word on essentially every sentence — and a Latin-only
    // manual test never sees it.
    await page.goto(stub.baseURL);

    const field = page.getByLabel('Message');
    await field.fill('中文');
    await field.evaluate((element) => {
      element.dispatchEvent(
        new KeyboardEvent('keydown', {
          key: 'Enter',
          isComposing: true,
          bubbles: true,
          cancelable: true,
        }),
      );
    });

    await expect(page.getByRole('log')).not.toContainText('中文');
    await expect(field).toHaveValue('中文');
  });
});

test.describe('on touch', () => {
  test('Return inserts a newline rather than sending', async ({ page, stub }) => {
    // The default project is a phone. A bare Return has to stay a newline
    // here: there is no Shift on the software keyboard, so binding send to
    // Return would remove the only way to type one.
    await page.goto(stub.baseURL);

    const field = page.getByLabel('Message');
    await field.fill('typed on a phone');
    await field.press('Enter');

    await expect(page.getByRole('log')).not.toContainText('typed on a phone');
    await expect(field).toHaveValue('typed on a phone\n');
  });
});
