import { expect, test as base } from '@playwright/test';
import { startStub, type Scenario } from './stub-server';

/**
 * The sidebar's resident-memory panel.
 *
 * On a phone the sidebar is a drawer, so every spec here opens it first — and
 * a closed drawer polling nothing is itself part of the design.
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

async function openDrawer(page: import('@playwright/test').Page, baseURL: string) {
  await page.goto(baseURL);
  await expect(page.getByLabel('Message')).toBeVisible();
  await page.getByLabel('Open sidebar').click();
  return page.getByRole('dialog', { name: 'Conversations' });
}

test('reports what is resident against the ceiling', async ({ page, stub }) => {
  const drawer = await openDrawer(page, stub.baseURL);
  const panel = drawer.getByLabel('Resident models');

  await expect(panel).toContainText('Resident');
  // 9_750_000_000 / 25 GiB, both binary.
  await expect(panel).toContainText('9.1 GB / 25 GB');
});

test('names each model and sizes it', async ({ page, stub }) => {
  const drawer = await openDrawer(page, stub.baseURL);
  const rows = drawer.getByLabel('Resident models').getByRole('listitem');

  await expect(rows).toHaveCount(2);
  await expect(rows.first()).toContainText('qwen3-4b');
  // The reservation, NOT the smaller measured delta: a lazy engine faults
  // its weights in on the first request.
  await expect(rows.first()).toContainText('5.9 GB');
  await expect(rows.nth(1)).toContainText('bonsai-1.7b-2bit');
  await expect(rows.nth(1)).toContainText('3.2 GB');
});

test('marks the pinned primary', async ({ page, stub }) => {
  const drawer = await openDrawer(page, stub.baseURL);
  const rows = drawer.getByLabel('Resident models').getByRole('listitem');

  await expect(rows.first().getByLabel('Pinned')).toBeVisible();
  await expect(rows.nth(1).getByLabel('Pinned')).toHaveCount(0);
});

test.describe('with nothing resident', () => {
  test.use({
    scenario: {
      residency: { memory_limit_bytes: 25 * 1024 ** 3, memory_used_bytes: 0, models: [] },
    },
  });

  test('shows no panel at all', async ({ page, stub }) => {
    // An empty memory panel is chrome describing nothing. The engine reports
    // this whenever it is unreachable, which is most of a model switch.
    const drawer = await openDrawer(page, stub.baseURL);
    await expect(drawer.getByLabel('Resident models')).toHaveCount(0);
  });
});

test.describe('with no ceiling configured', () => {
  test.use({
    scenario: {
      residency: {
        memory_limit_bytes: 0,
        memory_used_bytes: 5_000_000_000,
        models: [
          {
            id: 'org/qwen3-4b',
            aliases: ['qwen3-4b'],
            state: 'resident',
            pinned: false,
            estimated_bytes: 5_000_000_000,
            measured_bytes: null,
          },
        ],
      },
    },
  });

  test('drops the denominator rather than dividing by zero', async ({ page, stub }) => {
    const drawer = await openDrawer(page, stub.baseURL);
    const panel = drawer.getByLabel('Resident models');

    await expect(panel).toContainText('4.7 GB');
    await expect(panel).not.toContainText('/');
  });
});
