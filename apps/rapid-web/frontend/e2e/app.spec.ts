import { expect, test as base, type Page } from '@playwright/test';
import { startStub, chatFrame, stubModel, type Scenario } from './stub-server';

/**
 * End-to-end specs against the BUILT artifact.
 *
 * Has already paid for itself twice: it caught a TDZ error in the store that
 * made the whole bundle throw on load and render a blank page, and an
 * auto-login path that validated a token without persisting it. Neither is
 * reachable from a unit test — one needs the module evaluated in load order,
 * the other needs a reload.
 */

type Stub = Awaited<ReturnType<typeof startStub>>;

/**
 * One stub per test, torn down automatically.
 *
 * A `let stub` shared across a describe block does NOT work: the config sets
 * `fullyParallel`, so tests run concurrently and each assignment clobbers the
 * previous test's handle — failures that vanish when run alone.
 */
const test = base.extend<{ scenario: Partial<Scenario>; stub: Stub }>({
  scenario: [{}, { option: true }],
  stub: async ({ scenario }, use) => {
    const stub = await startStub(scenario);
    await use(stub);
    await stub.close();
  },
});

/**
 * Seed localStorage BEFORE the app's first script runs.
 *
 * The store reads storage at module evaluation, so this has to precede the
 * navigation. `addInitScript` runs before any page script on every navigation,
 * which is also immune to the store ever becoming lazier about when it reads.
 */
async function seedStorage(page: Page, entries: Record<string, string>) {
  await page.addInitScript((seed: Record<string, string>) => {
    for (const [key, value] of Object.entries(seed)) localStorage.setItem(key, value);
  }, entries);
}

test.describe('boot and auth', () => {
  test('skips the gate entirely when the server needs no token', async ({ page, stub }) => {
    // Compatibility with older servers that advertised auth_required=false.
    await page.goto(stub.baseURL);

    await expect(page.getByLabel('Message')).toBeVisible();
    await expect(page.getByPlaceholder('Access token')).toHaveCount(0);
  });

  test.describe('with a token required', () => {
    test.use({ scenario: { authRequired: true, token: 'secret-token' } });

    test('consumes a fragment token and strips it from the address bar', async ({ page, stub }) => {
      // The token travels in the FRAGMENT so it never reaches an access log, a
      // proxy log, or the tunnel provider's request history. The page's job is
      // to take it and get it out of the URL before it can be screenshotted or
      // land in history.
      await page.goto(`${stub.baseURL}/#token=secret-token`);

      await expect(page.getByLabel('Message')).toBeVisible();
      expect(page.url()).not.toContain('secret-token');
      expect(page.url()).not.toContain('#');

      // And it must survive a reload. The auto-login path originally validated
      // the token without storing it, so a user who had just scanned the QR
      // code was asked to log in again on the very next refresh.
      await page.reload();
      await expect(page.getByLabel('Message')).toBeVisible();
    });

    test('does not persist a token the server rejected', async ({ page, stub }) => {
      // Storing a rejected credential means the next reload silently retries
      // something already known to be wrong.
      await page.goto(stub.baseURL);

      await page.getByPlaceholder('Access token').fill('wrong');
      await page.getByRole('button', { name: 'Enter' }).click();

      await expect(page.getByRole('alert')).toContainText('not accepted');
      expect(await page.evaluate(() => localStorage.getItem('rapid-mlx-web.token'))).toBeNull();
    });
  });
});

test.describe('history migration', () => {
  test('migrates a v1 transcript and does NOT label it "New chat"', async ({ page, stub }) => {
    // A transcript brought forward from v1 never passes through the "touch on
    // write" path, so an underived title left it showing as "New chat"
    // despite having messages.
    await seedStorage(page, {
      'rapid-mlx-web.history': JSON.stringify([
        { role: 'user', content: 'what is a metal shader' },
        { role: 'assistant', content: 'A program that runs on the GPU.' },
      ]),
    });
    await page.goto(stub.baseURL);

    // Below the layout breakpoint the conversation list lives in the drawer.
    await page.getByLabel('Open sidebar').click();
    const drawer = page.getByRole('dialog', { name: 'Conversations' });
    await expect(drawer).toContainText('what is a metal shader');
  });

  test('keeps working when localStorage refuses to persist', async ({ page, stub }) => {
    // Safari private browsing throws on write. Losing persistence is
    // survivable; breaking the send path over it is not.
    await page.addInitScript(() => {
      Storage.prototype.setItem = function throwing() {
        throw new DOMException('QuotaExceededError', 'QuotaExceededError');
      };
    });
    await page.goto(stub.baseURL);

    await page.getByLabel('Message').fill('does this still send');
    await page.getByRole('button', { name: 'Send' }).click();

    await expect(page.getByRole('log')).toContainText('does this still send');
    await expect(page.getByRole('log')).toContainText('Hello there.');
  });
});

test.describe('streaming', () => {
  test.describe('a reply containing code', () => {
    test.use({
      scenario: {
        frameDelayMs: 120,
        chatFrames: [chatFrame('```python\n'), chatFrame('print("hi")\n'), chatFrame('```\n')],
      },
    });

    test('renders an unterminated fence as code, not literal backticks', async ({ page, stub }) => {
      // Every streaming reply containing code flashed raw markdown in the old
      // page until its closing fence arrived. Caught in a screenshot, not by
      // any assertion — so here is the assertion.
      await page.goto(stub.baseURL);

      await page.getByLabel('Message').fill('show me code');
      await page.getByRole('button', { name: 'Send' }).click();

      // Mid-stream, before the closing fence has arrived.
      await expect(page.locator('pre code')).toContainText('print("hi")');
      await expect(page.getByRole('log')).not.toContainText('```');
    });
  });

  test.describe('a long reply', () => {
    test.use({
      scenario: {
        frameDelayMs: 60,
        chatFrames: Array.from({ length: 40 }, (_, index) => chatFrame(`Line ${index}.\n\n`)),
      },
    });

    test('stops following when the user scrolls up', async ({ page, stub }) => {
      // The old page called scrollToBottom() on every frame, so reading back
      // during a long answer was impossible.
      await page.goto(stub.baseURL);

      await page.getByLabel('Message').fill('write a long answer');
      await page.getByRole('button', { name: 'Send' }).click();

      const log = page.getByRole('log');

      // Wait for the transcript to actually OVERFLOW before scrolling up.
      // Asserting on some text having arrived is not the same precondition:
      // a few lines fit on screen, `scrollTop = 0` is then a no-op, no scroll
      // event fires, and the test fails for a reason unrelated to following.
      await expect
        .poll(() => log.evaluate((el) => el.scrollHeight - el.clientHeight))
        .toBeGreaterThan(80);

      // `scrollTop = 0`, not `scrollTo({ top: 0 })`: the latter can animate,
      // and the assertions below would race the animation rather than test
      // the follow behaviour.
      await log.evaluate((element) => {
        element.scrollTop = 0;
      });
      await expect(page.getByRole('button', { name: 'Jump to latest' })).toBeVisible();

      // The view must STAY where the user put it while tokens keep arriving.
      await page.waitForTimeout(400);
      expect(await log.evaluate((element) => element.scrollTop)).toBeLessThan(120);
    });
  });

  test.describe('an interrupted reply', () => {
    test.use({
      scenario: {
        frameDelayMs: 150,
        chatFrames: Array.from({ length: 20 }, (_, index) => chatFrame(`part ${index} `)),
      },
    });

    test('keeps partial content when the user stops a turn', async ({ page, stub }) => {
      // Stop means "I have read enough", not "throw it away".
      await page.goto(stub.baseURL);

      await page.getByLabel('Message').fill('start something long');
      await page.getByRole('button', { name: 'Send' }).click();

      await expect(page.getByRole('log')).toContainText('part 0');
      await page.getByRole('button', { name: 'Stop generating' }).click();

      await expect(page.getByRole('log')).toContainText('part 0');
      await expect(page.getByRole('button', { name: 'Send' })).toBeVisible();
    });
  });
});

test.describe('errors', () => {
  test.describe('when a switch is refused', () => {
    test.use({
      scenario: {
        loadFailure: {
          status: 409,
          type: 'busy_streaming',
          message: 'a chat response is still streaming',
        },
        // BOTH cached, so choosing the second one is a LOAD. An uncached model
        // starts a download instead and never reaches the load endpoint at
        // all — which is what the first version of this test did, and it
        // failed for a reason that had nothing to do with notices.
        models: [
          stubModel({
            alias: 'qwen3-4b',
            size_bytes: 2_400_000_000,
            cached: true,
            cached_bytes: 2_400_000_000,
          }),
          stubModel({
            alias: 'gemma-2b',
            size_bytes: 1_100_000_000,
            cached: true,
            cached_bytes: 1_100_000_000,
          }),
        ],
      },
    });

    test('shows an in-UI notice rather than a window.alert', async ({ page, stub }) => {
      // The old page routed this and five other distinct failures through the
      // same modal alert, discarding the server's error.type on the way.
      let alerted = false;
      page.on('dialog', async (dialog) => {
        alerted = true;
        await dialog.dismiss();
      });

      await page.goto(stub.baseURL);
      await expect(page.getByLabel('Message')).toBeVisible();

      // Through the composer's picker, which is where the model now lives.
      await page.getByRole('button', { name: /^Model:/ }).click();
      await page.getByRole('menuitem', { name: /gemma-2b/ }).click();

      await expect(page.getByText('still streaming')).toBeVisible();
      expect(alerted).toBe(false);
    });
  });

  test.describe('when the first attempt fails', () => {
    test.use({
      scenario: {
        chatFailure: { status: 503, type: 'engine_unavailable', message: 'still loading' },
      },
    });

    test('retrying does not duplicate the user message', async ({ page, stub }) => {
      // Retry is message-addressed: it reuses the prompt rather than
      // re-sending its text, which would append a second copy and split the
      // two answers across different branches.
      await page.goto(stub.baseURL);

      await page.getByLabel('Message').fill('only once');
      await page.getByRole('button', { name: 'Send' }).click();
      await expect(page.getByRole('log')).toContainText('still loading');

      stub.scenario.chatFailure = null;
      await page.getByRole('button', { name: 'Retry' }).first().click();
      await expect(page.getByRole('log')).toContainText('Hello there.');

      const occurrences = await page
        .getByRole('log')
        .evaluate((element) => (element.textContent?.match(/only once/g) ?? []).length);
      expect(occurrences).toBe(1);
    });
  });
});

test.describe('accessibility', () => {
  test('a sheet is a real dialog and traps focus', async ({ page, stub }) => {
    // The old page's sheets were plain divs behind a `hidden` class: content
    // behind them stayed reachable, so they were modal only visually.
    //
    // Asserts behaviour rather than `aria-modal`. Radix omits that attribute
    // deliberately — support is patchy — and hides the background with
    // aria-hidden plus a focus trap instead, which is strictly stronger.
    await page.goto(stub.baseURL);

    await page.getByLabel('Open sidebar').click();
    await page.getByRole('button', { name: 'Settings' }).click();

    const dialog = page.getByRole('dialog', { name: 'Settings' });
    await expect(dialog).toBeVisible();

    // The composer behind the sheet must be out of the accessibility tree
    // and unable to take focus. `toBeHidden` is the wrong check here — it
    // reads CSS visibility, and the composer is still on screen behind the
    // scrim.
    const background = await page.evaluate(() => {
      const composer = document.querySelector<HTMLElement>('textarea[aria-label="Message"]');
      if (!composer) return { hiddenFromA11y: false, tookFocus: true };

      let node: HTMLElement | null = composer;
      let hiddenFromA11y = false;
      while (node) {
        if (node.getAttribute('aria-hidden') === 'true') {
          hiddenFromA11y = true;
          break;
        }
        node = node.parentElement;
      }

      composer.focus();
      return { hiddenFromA11y, tookFocus: document.activeElement === composer };
    });
    expect(background.hiddenFromA11y).toBe(true);
    expect(background.tookFocus).toBe(false);

    await page.keyboard.press('Escape');
    await expect(dialog).toBeHidden();
    await expect(page.getByLabel('Message')).toBeVisible();
  });

  test('escape closes the sheet and leaves the app usable', async ({ page, stub }) => {
    // The old page's single handler closed all three sheets at once.
    await page.goto(stub.baseURL);

    await page.getByLabel('Open sidebar').click();
    await expect(page.getByRole('dialog', { name: 'Conversations' })).toBeVisible();

    await page.keyboard.press('Escape');
    await expect(page.getByRole('dialog', { name: 'Conversations' })).toBeHidden();
    await expect(page.getByLabel('Message')).toBeVisible();
  });

  test('the transcript is a log that does not announce every token', async ({ page, stub }) => {
    // A live transcript would have a screen reader read all of a streamed
    // answer, on every commit. State changes go through a separate region.
    await page.goto(stub.baseURL);
    await expect(page.getByRole('log')).toHaveAttribute('aria-live', 'off');
  });

  test('does not block pinch zoom', async ({ page, stub }) => {
    // maximum-scale=1 in the old page was a WCAG 1.4.4 failure.
    await page.goto(stub.baseURL);

    const viewport = await page.locator('meta[name="viewport"]').getAttribute('content');
    expect(viewport).not.toContain('maximum-scale');
    // But the safe-area opt-in must remain, or the composer sits under the
    // home indicator.
    expect(viewport).toContain('viewport-fit=cover');
  });
});

test.describe('layout', () => {
  test('a phone gets a drawer, not a permanent rail', async ({ page, stub }) => {
    // The rail costs 260px against a 720px reading measure, so below the
    // breakpoint it overlays instead of sitting beside — which also makes it
    // modal, hence the dialog role that the rail deliberately does not have.
    await page.goto(stub.baseURL);
    await expect(page.locator('aside')).toHaveCount(0);

    await page.getByLabel('Open sidebar').click();
    await expect(page.getByRole('dialog', { name: 'Conversations' })).toBeVisible();
  });

  test('the desktop rail is a landmark, not a dialog', async ({ page, stub }) => {
    // A permanent region must NOT trap focus or answer Escape: the transcript
    // beside it has to stay reachable, which is the whole difference between
    // the rail and the drawer.
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto(stub.baseURL);

    const rail = page.locator('aside');
    await expect(rail).toBeVisible();
    await expect(rail).not.toHaveAttribute('aria-modal', 'true');
    // No "open sidebar" affordance, because it is already open.
    await expect(page.getByLabel('Open sidebar')).toHaveCount(0);

    await page.keyboard.press('Escape');
    await expect(rail).toBeVisible();
  });

  test('the model selector lives in the composer, not the rail', async ({ page, stub }) => {
    // Moved out of the sidebar: the model is a property of the message about
    // to be sent, so it belongs beside Send rather than in the navigation.
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto(stub.baseURL);

    await expect(page.getByRole('button', { name: /^Model: qwen3-4b/ })).toBeVisible();
    await expect(page.locator('aside').getByRole('button', { name: /qwen3-4b/ })).toHaveCount(0);
  });
});

test.describe('attach mode', () => {
  test.use({ scenario: { canSwitch: false, model: null, engineState: 'stopped' } });

  test('offers no control that cannot deliver', async ({ page, stub }) => {
    // The dead end this fixes: a disabled chip whose only explanation was a
    // `title`, plus an empty state telling the user to go and press it. There
    // is no hover on a phone, so that tooltip never fired at all.
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto(stub.baseURL);

    // The engine belongs to whoever started it, so the picker is inert text
    // rather than a menu that would refuse every choice in it.
    await expect(page.getByRole('button', { name: /^Model:/ })).toHaveCount(0);
  });

  test('does not send the user to a picker that cannot help', async ({ page, stub }) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto(stub.baseURL);
    await expect(page.getByText('Choose a model in the sidebar')).toHaveCount(0);
  });
});

test.describe('readiness', () => {
  test.describe('with a model that is not running', () => {
    test.use({ scenario: { engineState: 'stopped', model: 'qwen3-4b' } });

    test('gating send keeps the draft and explains why', async ({ page, stub }) => {
      // A gated send must not consume what the user typed for a condition they
      // may not have noticed.
      await page.goto(stub.baseURL);

      const field = page.getByLabel('Message');
      await field.fill('this must survive');
      await page.getByRole('button', { name: 'Send' }).click();

      await expect(field).toHaveValue('this must survive');
      await expect(page.getByRole('log')).not.toContainText('this must survive');
    });

    test('names the blocking step in the composer placeholder', async ({ page, stub }) => {
      await page.goto(stub.baseURL);
      await expect(page.getByLabel('Message')).toHaveAttribute(
        'placeholder',
        /Start qwen3-4b first/,
      );
    });
  });

  test.describe('while the model is loading', () => {
    test.use({
      scenario: { engineState: 'starting', model: 'qwen3-4b', detail: 'loading weights' },
    });

    test('says the model is starting, not a bare spinner', async ({ page, stub }) => {
      await page.goto(stub.baseURL);
      await expect(page.getByText('Starting qwen3-4b')).toBeVisible();
    });
  });
});
