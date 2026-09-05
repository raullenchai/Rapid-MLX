import { expect, test as base } from '@playwright/test';
import { openModelList, startStub, stubModel, type Scenario } from './stub-server';

/**
 * The model list's alignment. Two defects, both invisible to a unit test and
 * to a class-name grep:
 *
 * 1. The alias centred itself. `text-left` was on the row `div`, but the
 *    `<button>` inside takes `text-align: center` from the UA stylesheet,
 *    which beats an inherited value.
 * 2. The badge column moved. The trash was rendered only for cached models
 *    and occupied no space otherwise, so the badges sat in two columns.
 */
type Stub = Awaited<ReturnType<typeof startStub>>;

const MODELS: Scenario['models'] = [
  stubModel({
    alias: 'bonsai-1.7b-2bit',
    size_bytes: 473_000_000,
    cached: true,
    cached_bytes: 473_000_000,
    tool_call_parser: 'hermes',
  }),
  // Not cached, and with no badges at all — the row most likely to drift.
  stubModel({ alias: 'gemma-27b', size_bytes: 17_000_000_000 }),
];

const test = base.extend<{ scenario: Partial<Scenario>; stub: Stub }>({
  scenario: [
    { engineState: 'stopped', model: null, models: MODELS },
    { option: true },
  ],
  stub: async ({ scenario }, use) => {
    const stub = await startStub(scenario);
    await use(stub);
    await stub.close();
  },
});

async function openSheet(page: import('@playwright/test').Page, baseURL: string) {
  await page.goto(baseURL);
  // Wait for the app to mount before probing: `isVisible()` is a snapshot, not
  // a wait, so on a cold page it reports false for everything.
  await expect(page.getByLabel('Message')).toBeVisible();

  const sheet = await openModelList(page);
  await expect(sheet).toBeVisible();

  // Wait for the slide-in to settle. The sheet animates its width, so a
  // measurement taken immediately reads a mid-animation value and every
  // "did this move?" assertion below would compare against a moving target.
  await expect.poll(async () => (await sheet.boundingBox())?.width ?? 0).toBeGreaterThan(0);
  let last = -1;
  await expect
    .poll(async () => {
      const width = (await sheet.boundingBox())?.width ?? 0;
      const settled = width === last;
      last = width;
      return settled;
    })
    .toBe(true);

  return sheet;
}

test('the alias sits flush left, over its own size line', async ({ page, stub }) => {
  await openSheet(page, stub.baseURL);

  for (const alias of ['bonsai-1.7b-2bit', 'gemma-27b']) {
    // Measured with a Range over the TEXT NODE, not the element's box. The
    // alias span is a flex item that shrinks to its content, so its box hugs
    // the left edge whether or not the text inside is centred — asserting on
    // `boundingBox()` passes against the very defect this guards. The glyphs
    // are what the eye sees, so the glyphs are what gets measured.
    const offsets = await page.evaluate((name) => {
      const textLeft = (element: Element | null | undefined) => {
        const node = element?.firstChild;
        if (!node) return null;
        const range = document.createRange();
        range.selectNodeContents(node);
        return range.getBoundingClientRect().left;
      };
      // Scoped to the row BUTTON: the Disk overview card's "Largest" line
      // names the same alias, and an unscoped document lookup grabs that
      // instead — a row this test never renders an assertion about.
      const span = [...document.querySelectorAll('button span')].find(
        (candidate) => candidate.textContent === name,
      );
      return { name: textLeft(span), size: textLeft(span?.nextElementSibling) };
    }, alias);

    expect(offsets.name, alias).not.toBeNull();
    expect(offsets.size, alias).not.toBeNull();
    // Centred, the alias started ~100px right of the size line beneath it.
    expect(Math.abs(offsets.name! - offsets.size!), alias).toBeLessThan(1.5);
  }
});

test('the status badge holds one column regardless of row contents', async ({ page, stub }) => {
  const sheet = await openSheet(page, stub.baseURL);

  const onDisk = await sheet.getByText('on disk', { exact: true }).boundingBox();
  const remote = await sheet.getByText('remote', { exact: true }).boundingBox();
  expect(onDisk).not.toBeNull();
  expect(remote).not.toBeNull();

  // Right edges align: the cached row reserves the trash's slot, so the
  // uncached row's badge does not slide into it.
  const onDiskRight = onDisk!.x + onDisk!.width;
  const remoteRight = remote!.x + remote!.width;
  expect(Math.abs(onDiskRight - remoteRight)).toBeLessThan(1.5);
});

test.describe('on a pointer device', () => {
  // The suite's one project is a phone, where the trash is permanently
  // visible and there is no hover to test. The shift this guards against only
  // happens where the button fades in.
  test.use({ hasTouch: false, isMobile: false, viewport: { width: 900, height: 800 } });

  test('revealing the trash does not move the badge', async ({ page, stub }) => {
    const sheet = await openSheet(page, stub.baseURL);

    const badge = sheet.getByText('on disk', { exact: true });
    const before = await badge.boundingBox();

    // The row, not the Disk overview card's "Largest" line — both name this
    // alias, and only the row has a trash to reveal.
    await sheet.getByRole('button', { name: /^bonsai-1\.7b-2bit/ }).hover();
    await expect(sheet.getByRole('button', { name: 'Delete bonsai-1.7b-2bit' })).toBeVisible();

    const after = await badge.boundingBox();
    // The slot is reserved, so fading the button in shifts nothing.
    expect(Math.abs(before!.x - after!.x)).toBeLessThan(1.5);
  });
});
