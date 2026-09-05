import { expect, test as base } from '@playwright/test';
import { openModelList, startStub, type Scenario } from './stub-server';

/**
 * Download progress reaches the UI by POLLING, not by a stream.
 *
 * The SSE feed this replaced was fine on loopback and unusable through a
 * `trycloudflare` tunnel: headers in 1.8 s, then no body byte in 65 s.
 * Cloudflare strips `X-Accel-Buffering` and padding the first frame did not
 * help. Chat survives the same tunnel because it emits tokens continuously.
 *
 * These specs pin what matters to the user (the bar moves, a running job is
 * picked up on open) without asserting the transport.
 */
type Stub = Awaited<ReturnType<typeof startStub>>;

const test = base.extend<{ scenario: Partial<Scenario>; stub: Stub }>({
  scenario: [{ engineState: 'stopped', model: null }, { option: true }],
  stub: async ({ scenario }, use) => {
    const stub = await startStub(scenario);
    await use(stub);
    await stub.close();
  },
});

async function openSheet(page: import('@playwright/test').Page, baseURL: string) {
  await page.goto(baseURL);
  await expect(page.getByLabel('Message')).toBeVisible();
  return openModelList(page);
}

test.describe('a download already in flight', () => {
  test.use({
    scenario: {
      engineState: 'stopped',
      model: null,
      download: {
        state: 'running',
        alias: 'llama-8b',
        done_bytes: 2_050_000_000,
        total_bytes: 8_200_000_000,
      },
    },
  });

  test('is picked up when the sheet opens', async ({ page, stub }) => {
    const sheet = await openSheet(page, stub.baseURL);

    // Re-attaching matters: a phone that locked its screen mid-pull reopens
    // the sheet and must see real progress rather than an empty strip.
    // Scoped to the strip — the alias also appears in the list behind it.
    await expect(sheet.getByText('25% of 7.6 GB')).toBeVisible();
    await expect(sheet.getByRole('button', { name: 'Cancel' })).toBeVisible();
  });

  test('advances as the server reports more bytes', async ({ page, stub }) => {
    const sheet = await openSheet(page, stub.baseURL);
    await expect(sheet.getByText('25% of 7.6 GB')).toBeVisible();

    // The whole point of the poll: the next tick must pick this up with no
    // interaction from the user.
    stub.scenario.download = {
      state: 'running',
      alias: 'llama-8b',
      done_bytes: 6_150_000_000,
      total_bytes: 8_200_000_000,
    };

    await expect(sheet.getByText('75% of 7.6 GB')).toBeVisible({ timeout: 10_000 });
  });

  test('reports completion and re-reads what is on disk', async ({ page, stub }) => {
    const sheet = await openSheet(page, stub.baseURL);
    await expect(sheet.getByText('25% of 7.6 GB')).toBeVisible();

    stub.scenario.download = {
      state: 'done',
      alias: 'llama-8b',
      done_bytes: 8_200_000_000,
      total_bytes: 8_200_000_000,
    };
    // The catalog now has it cached, which is what the page must re-read.
    const entry = stub.scenario.models.find((model) => model.alias === 'llama-8b');
    if (entry) {
      entry.cached = true;
      entry.cached_bytes = 8_200_000_000;
    }

    await expect(sheet.getByText('done')).toBeVisible({ timeout: 10_000 });
    // Proof the completion triggered a catalog refresh: the row stops saying
    // "remote" and gains the delete affordance a cached model carries.
    await expect(sheet.getByRole('button', { name: 'Delete llama-8b' })).toBeVisible({
      timeout: 10_000,
    });
  });

  test('stops polling once the download finishes', async ({ page, stub }) => {
    const sheet = await openSheet(page, stub.baseURL);
    await expect(sheet.getByText('25% of 7.6 GB')).toBeVisible();

    stub.scenario.download = {
      state: 'done',
      alias: 'llama-8b',
      done_bytes: 8_200_000_000,
      total_bytes: 8_200_000_000,
    };

    // Let the transition settle before measuring: the poll that OBSERVES the
    // completion is itself a request, and it lands after the strip renders.
    // Sampling at that instant reads one request too few and the assertion
    // fails against correct behaviour.
    await expect(sheet.getByText('done', { exact: true })).toBeVisible({ timeout: 10_000 });
    await page.waitForTimeout(2500);

    // The server retains the last finished job forever, so asking again can
    // only ever return the same answer. Without a stop condition this kept
    // requesting once a second for as long as the sheet stayed open — the
    // window below would have added roughly five more.
    const settled = stub.scenario.statusPolls;
    await page.waitForTimeout(5000);
    expect(stub.scenario.statusPolls).toBe(settled);
  });

  test('resumes polling when the next download starts', async ({ page, stub }) => {
    const sheet = await openSheet(page, stub.baseURL);
    stub.scenario.download = {
      state: 'done',
      alias: 'qwen3-4b',
      done_bytes: 2_400_000_000,
      total_bytes: 2_400_000_000,
    };
    await expect(sheet.getByText('done')).toBeVisible({ timeout: 10_000 });
    await page.waitForTimeout(1500);
    const idle = stub.scenario.statusPolls;

    // Stopping must not be permanent. Starting a pull puts a `running` job in
    // the store, which is what has to bring the loop back. The stub's pull
    // route sets the running job, exactly as the real server does.
    await sheet.getByRole('button', { name: /^llama-8b/ }).click();

    await expect.poll(() => stub.scenario.statusPolls, { timeout: 10_000 }).toBeGreaterThan(idle);
  });
});

test('no strip is shown when nothing is downloading', async ({ page, stub }) => {
  const sheet = await openSheet(page, stub.baseURL);

  // The sheet is open and populated...
  await expect(sheet.getByPlaceholder('Search models')).toBeVisible();
  // ...but `idle` renders as absence, not as a zeroed bar. Cancel belongs to
  // the strip alone, so its absence is the strip's absence.
  await expect(sheet.getByRole('button', { name: 'Cancel' })).toHaveCount(0);
});
