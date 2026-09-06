import { expect, test as base } from '@playwright/test';
import { startStub, type Scenario } from './stub-server';

/**
 * A chat model and an image model loaded at once.
 *
 * The engine keeps text/vision in a single-slot group and gives each media
 * modality its own, so a hot `POST /v1/models/load` leaves both usable. The
 * page has to stop reasoning from `status.model` alone, which names only the
 * PRIMARY — otherwise the images surface reports its own loaded model as
 * missing.
 */
type Stub = Awaited<ReturnType<typeof startStub>>;

const test = base.extend<{ scenario: Partial<Scenario>; stub: Stub }>({
  scenario: [
    {
      // The primary is the CHAT model; the image model rode in on a hot load.
      model: 'qwen3-4b',
      resident: ['qwen3-4b', 'flux2-klein-4b'],
      engineState: 'ready',
    },
    { option: true },
  ],
  stub: async ({ scenario }, use) => {
    const stub = await startStub(scenario);
    await use(stub);
    await stub.close();
  },
});

async function openImages(page: import('@playwright/test').Page, baseURL: string) {
  await page.goto(baseURL);
  await expect(page.getByLabel('Message')).toBeVisible();
  await page.getByLabel('Open sidebar').click();
  await page.getByRole('button', { name: 'Images' }).click();
}

test('the images surface is usable while a chat model is primary', async ({ page, stub }) => {
  await openImages(page, stub.baseURL);

  // Before this change the canvas was hidden behind "choose an image model",
  // because `status.model` named the chat model.
  const prompt = page.getByLabel('Prompt');
  await expect(prompt).toBeVisible();
  await expect(prompt).toBeEnabled();
});

test('a render names its own model rather than the primary', async ({ page, stub }) => {
  const bodies: Array<Record<string, unknown>> = [];
  await page.route('**/api/images/jobs', async (route) => {
    bodies.push(route.request().postDataJSON() as Record<string, unknown>);
    await route.continue();
  });

  await openImages(page, stub.baseURL);
  await page.getByLabel('Prompt').fill('a red cube');
  await page.getByRole('button', { name: 'Generate' }).click();
  await expect(page.getByRole('img', { name: /a red cube/ })).toBeVisible();

  // An omitted `model` resolves to the primary — the CHAT model — which
  // answers 409 image_model_not_loaded.
  expect(bodies[0]?.model).toBe('flux2-klein-4b');
});

test('the chat stays sendable too', async ({ page, stub }) => {
  await page.goto(stub.baseURL);
  await page.getByLabel('Message').fill('still works');
  await page.getByRole('button', { name: 'Send' }).click();
  await expect(page.getByRole('log')).toContainText('still works');
});
